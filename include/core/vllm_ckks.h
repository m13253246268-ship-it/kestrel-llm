/**
 * vllm_ckks.h - CKKS 风格近似 HE（RNS 多素数模数链），纯 C11 自研
 *
 * 设计（axiom_registry.json 对齐）：
 *   - axiom_arith_fixedpoint_encode_001 的推广：实数 x -> 整数 X = round(x * S)，
 *     S = scale = 2^60（本实现）。
 *   - axiom_modal_transition_conservation_005 (iNTT(NTT(x))=x)：槽位打包复用
 *     vllm_fhe 的评估同态（R_t ≅ ∏ F_t），n=1024 槽位。
 *   - 非整数系数多项式（GELU 等）在 BFV 明文模下不可行（模环除法给出模数解），
 *     CKKS 的 scale 管理 + 模数切换（rescale）天然支持——本模块解决该缺口。
 *
 * 架构：
 *   - 模数链：4 个 60-bit 素数 q0..q3（q_i ≡ 1 mod 2n），总 240-bit，支持深 3。
 *   - scale = 2^60：乘法后 scale 翻倍到 2^120，rescale（模数切换切掉一个素数
 *     ≈ 2^60）使 scale 回到 2^60。明文槽位 |z| < 0.5（2^60·z < 2^59 < 任一 q_i，
 *     保证中间层解密无需 CRT）。
 *   - 密文 RNS 表示：分量 x 素数 x 系数（uint64）。
 *   - 乘法：通用 tensor（教科书 negacyclic，正确性优先）；rescale = 模数切换。
 *   - relin：key-switch 使 3 分量回 2 分量（重线性化）。
 *   - 旋转：Galois 自动态 + key-switch（槽位循环移位）。
 *
 * 安全声明（原型级）：噪声源为非密码学 RNG，安全参数未按标准评估；
 * 目标为验证"密文下实数多项式 + 深层运算 + 旋转"的端到端正确性。
 */
#ifndef VLLM_CKKS_H
#define VLLM_CKKS_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef CKKS_N
#define CKKS_N           1024u       /* 环次数 (2^10)；可编译期覆盖 -DCKKS_N=2048 等 */
#endif
#ifndef CKKS_NPRIMES
#define CKKS_NPRIMES     32          /* 模数链素数个数（60-bit×32=1920-bit；常规深 10）。
                                         折叠函数（EvalMod 倍角 sin）~31 深，折叠测试/全链
                                         bootstrapping 用 -DCKKS_NPRIMES=40~128 覆盖 */
#endif
#define CKKS_SCALE_BITS  60          /* scale = 2^60 */
#define CKKS_KEY_HW      8           /* 私钥稀疏三元非零个数（降 hw 控制 bootstrapping 折叠混叠：
                                         混叠 I~hw/2=4，sin 逼近多项式 9 次；原型无安全可接受） */
#define CKKS_MAX_COMP    13          /* 无 relin 多分量上限（深 11：comps = 深度+2） */
#define CKKS_NOISE_CBD   8           /* 中心二项分布参数 eta（σ=√(eta/2)=2，界 [-8,8]） */

typedef struct {
    uint32_t n;          /* 1024 */
    int      nprimes;    /* 当前模数链素数个数（随 rescale 减小） */
    uint64_t q[CKKS_NPRIMES];   /* 模数链（递减） */
    uint64_t scale;      /* 2^60 */
    uint64_t *pts[CKKS_NPRIMES]; /* 每素数评估点 pts[j]=ζ^(2j+1) */
} ckks_ctx_t;

typedef struct {
    uint32_t n;
    int8_t  *s;          /* [n] 稀疏三元 {-1,0,1}（< 任一 q_i） */
} ckks_sk_t;

/* relin 密钥：digit 分解（base 2^20，3 位），D(rk0_d + rk1_d*s) = B^d * s^2 */
#define CKKS_RELIN_DIGITS     3
#define CKKS_RELIN_BASE_BITS  20
typedef struct {
    uint32_t n;
    int      digits;
    uint64_t *rk0[CKKS_RELIN_DIGITS];   /* [digits][nprimes*n] */
    uint64_t *rk1[CKKS_RELIN_DIGITS];
} ckks_rk_t;

/* Galois 密钥（旋转 key-switch）：digit 分解（base 2^20，3 位），
 * D(gk0_d + gk1_d*s) = B^d * σ_k(s)。无分解时 key-switch 噪声 ~ n*q*B ~ 2^73
 * 会爆 60-bit 素数；digit 分解把噪声降到 ~ Σ_d n*2^20*B ~ 2^45。 */
typedef struct {
    uint32_t n;
    int      digits;
    uint64_t *gk0[CKKS_RELIN_DIGITS];   /* [digits][nprimes*n] */
    uint64_t *gk1[CKKS_RELIN_DIGITS];
} ckks_gk_t;

/* 密文：comps x nprimes x n（RNS） */
typedef struct {
    ckks_ctx_t ctx;      /* 密文自身模数链状态 */
    int comps;           /* 2 或 3 */
    uint64_t *c;         /* [comps*nprimes*n] 排列：c[(comp*nprimes+p)*n + i] */
} ckks_ct_t;

/* ---- 初始化/释放 ---- */
int  ckks_ctx_init(ckks_ctx_t *ctx);           /* 构造 4×60-bit 素数链 + 评估点 */
void ckks_ctx_free(ckks_ctx_t *ctx);
int  ckks_sk_gen(ckks_sk_t *sk, uint32_t n);
void ckks_sk_free(ckks_sk_t *sk);
int  ckks_rk_gen(ckks_rk_t *rk, const ckks_sk_t *sk, const ckks_ctx_t *ctx);
void ckks_rk_free(ckks_rk_t *rk);
int  ckks_gk_gen(ckks_gk_t *gk, const ckks_sk_t *sk, const ckks_ctx_t *ctx, int t);
int  ckks_gk_gen_k(ckks_gk_t *gk, const ckks_sk_t *sk, const ckks_ctx_t *ctx, uint64_t k);
void ckks_gk_free(ckks_gk_t *gk);

/* ---- 编码/解码（槽位实数 <-> 明文多项式 RNS） ---- */
int ckks_encode(uint64_t *poly, const double *z, const ckks_ctx_t *ctx);
int ckks_decode(double *z, const uint64_t *poly, const ckks_ctx_t *ctx);

/* ---- 加密/解密 ---- */
int ckks_encrypt(ckks_ct_t *ct, const ckks_sk_t *sk,
                 const uint64_t *poly, const ckks_ctx_t *ctx);
int ckks_decrypt(uint64_t *poly, const ckks_sk_t *sk, const ckks_ct_t *ct);

/* ---- 同态运算 ---- */
int ckks_add(ckks_ct_t *ct, const ckks_ct_t *a, const ckks_ct_t *b);
int ckks_mult(ckks_ct_t *ct, const ckks_ct_t *a, const ckks_ct_t *b); /* tensor, 3 分量 */
int ckks_mult_plain(ckks_ct_t *ct, const ckks_ct_t *a, const uint64_t *poly); /* 密文x明文多项式 */
int ckks_rescale(ckks_ct_t *ct, const ckks_ct_t *in);  /* 模数切换（切一个素数，scale 归一） */
int ckks_modswitch(ckks_ct_t *ct, const ckks_ct_t *in); /* 降级：切素数不 rescale */
int ckks_modraise(ckks_ct_t *ct, const ckks_ct_t *in, int target); /* ModRaise：Garner 扩展提升回满链（bootstrapping 核心） */
int ckks_relin(ckks_ct_t *ct, const ckks_ct_t *in, const ckks_rk_t *rk); /* 3->2 分量 */
int ckks_rotate(ckks_ct_t *ct, const ckks_ct_t *in, const ckks_sk_t *sk,
                const ckks_gk_t *gk, int t); /* 槽位旋转 σ_{5^t}（支持多分量，输出 2 分量） */
int ckks_rotate_k(ckks_ct_t *ct, const ckks_ct_t *in, const ckks_sk_t *sk,
                  const ckks_gk_t *gk, uint64_t k); /* 一般 Galois σ_k（k 任意奇数），同上 */
int ckks_ct_copy(ckks_ct_t *dst, const ckks_ct_t *src);
void ckks_ct_free(ckks_ct_t *ct);

/* RNG 注入（同 vllm_fhe） */
void ckks_set_rng(void (*fn)(uint8_t *out, size_t len));

/* ---- 自测（T12，接入一键判定） ---- */
int ckks_self_test(void);

#ifdef __cplusplus
}
#endif

#endif /* VLLM_CKKS_H */
