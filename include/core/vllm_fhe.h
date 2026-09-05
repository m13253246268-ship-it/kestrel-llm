/**
 * vllm_fhe.h - Lightweight RLWE/BGV-Style Homomorphic Encryption (纯 C11)
 *
 * Axiom-Driven Design (from axiom_registry.json):
 *   - axiom_arith_fixedpoint_encode_001 : 定点实数 -> 明文环元素 (K-bit, homomorphic
 *       addition PASS)
 *   - ntt_isomorphism / axiom_modal_transition_conservation_005 : 多项式乘法在
 *       NTT 域等价（本实现先用教科书 O(n^2) 保证正确性，NTT 加速为后续步骤）
 *   - blas_acceleration : 密文矩阵乘法复用 block/tile 思想
 *
 * 方案：Ring-LWE / BGV 风格 Leveled HE（Fan-Vercauteren BFV）
 *   - R_q = Z_q[x]/(x^n+1)，n=1024
 *   - 密文模 q = 2^64 - 2^32 + 1（Goldilocks prime，64-bit 域）
 *   - 明文模 t=8192（2^13，与 q 互素）
 *   - 缩放因子 Delta = floor(q/t)，明文以 Delta*m 编码（BFV scaling）
 *   - 乘法：3 分量 tensor + (t/q) rescale；解密使用 (1, s, s^2)
 *
 * 设计决策（与 axiom_registry.json 对齐）：
 *   - 定点编码 K bit：映射 axiom_arith_fixedpoint_encode_001（K 可配置）
 *   - 教科书 O(n^2) 多项式乘法（正确性优先，位级可验证）；
 *     NTT 加速（ntt_isomorphism 公理）留作后续 A-B-C-D 步骤
 *
 * 安全声明（原型级）：
 *   - 噪声源为均匀小范围（测试 RNG，可注入替换），非密码学安全强度；
 *     安全参数 (n,q,sigma) 均可调，正式部署需升级为密码学 RNG + 大模数。
 *   - 目标：验证"密文下加/乘/矩阵乘/低阶多项式逼近"的端到端正确性。
 */

#ifndef VLLM_FHE_H
#define VLLM_FHE_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ================================================================
 * 1. 参数
 * ================================================================ */

#define VFHE_N_DEFAULT     1024u          /* 多项式环次数 (2^10) */
#define VFHE_T_DEFAULT     12289u         /* 明文模（素数，t-1=12288=6*2048：支持 n=1024 SIMD 槽位打包；深2需 t<~5150，仅深1） */
#define VFHE_Q_BITS        64             /* 密文模位宽 (Goldilocks) */
#define VFHE_B_NOISE       8              /* 均匀噪声界 [-B, B) */
#define VFHE_KEY_HW        64             /* 私钥稀疏三元 -1/0/1 非零个数 */

typedef struct {
    uint32_t n;          /* 环次数 */
    uint32_t q_bits;     /* 密文模位宽（实际模数 2^64-2^32+1） */
    uint32_t t;          /* 明文模 */
    uint32_t B;          /* 均匀噪声界 */
    uint32_t hw;         /* 私钥非零系数个数 */
} vfhe_params_t;

/** 默认参数（原型级）。 */
const vfhe_params_t *vfhe_params_default(void);

/* ================================================================
 * 2. 密钥与密文
 * ================================================================ */

typedef struct {
    vfhe_params_t p;
    int8_t       *s;    /* [n] 三元 {-1,0,1}, 稀疏 */
} vfhe_sk_t;

typedef struct {
    vfhe_params_t p;
    uint64_t     *a;    /* [n] 均匀随机，系数 mod q (64-bit) */
    uint64_t     *b;    /* [n] b = a*s + e (mod q) */
} vfhe_pk_t;

/* 密文分量数：2 (level1) / 3 (level2) / 5 (level3) ...
 * 不做 relinearization：一次乘法分量数 = deg(a)+deg(b)，
 * 解密使用 1, s, s^2, ... 最高 5 分量（深度 2）。 */
#define VFHE_CT_MAX_COMP 5

typedef struct {
    vfhe_params_t p;
    uint64_t     *c[VFHE_CT_MAX_COMP];  /* [n] 每分量；c[1..4] 可为 NULL */
    int           comps;                /* 实际分量数 (2..5) */
} vfhe_ct_t;

/* ================================================================
 * 3. 核心 API
 * ================================================================ */

/** KeyGen: 生成私钥（稀疏三元）。返回 0 成功。 */
int vfhe_sk_gen(vfhe_sk_t *sk);

/** 从私钥派生公钥: b = a*s + e (mod q)。返回 0 成功。 */
int vfhe_pk_from_sk(vfhe_pk_t *pk, const vfhe_sk_t *sk);

/** 对称加密: m (0 <= m < t)。返回 0 成功。 */
int vfhe_encrypt_sym(vfhe_ct_t *ct, const vfhe_sk_t *sk, uint32_t m);

/** 公钥加密: m (0 <= m < t)。返回 0 成功。 */
int vfhe_encrypt_pub(vfhe_ct_t *ct, const vfhe_pk_t *pk, uint32_t m);

/** 解密: 恢复 0 <= m < t（BFV 缩放语义，必要时调用 vfhe_center 转有符号）。 */
int vfhe_decrypt(uint32_t *m, const vfhe_sk_t *sk, const vfhe_ct_t *ct);

/** 同态加法: ct = ct_a + ct_b (mod q)。 */
int vfhe_add(vfhe_ct_t *ct, const vfhe_ct_t *a, const vfhe_ct_t *b);

/** 同态乘法: ct = ct_a * ct_b (mod q)，3 分量输出。 */
int vfhe_mult(vfhe_ct_t *ct, const vfhe_ct_t *a, const vfhe_ct_t *b);

/** 同态标量乘: ct = ct_in * m (mod q)。 */
int vfhe_mul_scalar(vfhe_ct_t *ct, const vfhe_ct_t *in, uint32_t m);

/**
 * 密文-明文矩阵乘法（激活密文 × 明文权重，密文推理核心算子）:
 *   out[j] = Σ_{i=0}^{nvec-1} in[i] * M[i*ncol + j]   (mod t)
 * @param in   密文向量（长度 nvec，每元素一个密文）
 * @param M    明文矩阵 nvec×ncol（int32 行主序，|M| < t/2 可负）
 * @param out  输出密文向量（长度 ncol，需调用方分配 ncol 个 vfhe_ct_t）
 * 噪声：nvec 次标量乘加，E ~ nvec*t*B << Δ/2（nvec 上限 ~几百）。
 * 返回 0 成功。
 */
int vfhe_matmul_ctvec_plain(vfhe_ct_t *out, const vfhe_ct_t *in, uint32_t nvec,
                            const int32_t *M, uint32_t ncol);

/** 深拷贝。 */
int vfhe_ct_copy(vfhe_ct_t *dst, const vfhe_ct_t *src);

/** 释放密文。 */
void vfhe_ct_free(vfhe_ct_t *ct);

/** 释放密钥。 */
void vfhe_sk_free(vfhe_sk_t *sk);
void vfhe_pk_free(vfhe_pk_t *pk);

/* ================================================================
 * 4. 定点编码 (axiom_arith_fixedpoint_encode_001)
 *    encode: x (real, |x| <= 2^(K-1)) -> m = round(x * 2^K) mod t
 *    decode: m -> x' = (center(m, t) / 2^K)，误差 <= 2^-(K+1)
 * ================================================================ */

/** 定点编码: 实数 x -> 明文值 m (0..t-1)。K 为小数位宽。 */
uint32_t vfhe_fixed_encode(double x, int K);

/** 定点解码: 明文值 m -> 实数 x'（中心化，误差 <= 2^-(K+1)）。 */
double vfhe_fixed_decode(uint32_t m, int K);

/** 中心化: 0..t-1 -> [-t/2, t/2)。 */
int32_t vfhe_center(uint32_t v, uint32_t t);

/* ================================================================
 * 6. RNG 注入 (原型级；默认内置 xorshift，非密码学安全)
 * ================================================================ */

/** 设置随机字节回调（用于注入密码学 RNG）。NULL 恢复内置 xorshift。 */
void vfhe_set_rng(void (*fn)(uint8_t *out, size_t len));

/* ================================================================
 * 7. SIMD 槽位打包 (axiom_ntt_isomorphism: R_t ≅ ∏ F_t 评估同态)
 *    要求明文模 t 为素数且 2n | t-1（默认 t=12289 满足：12288=6*2048）。
 *    一个密文携带 n=1024 个独立槽位；环加/乘 = 逐槽位加/乘。
 * ================================================================ */

/** 计算评估点 pts[j] = ζ^(2j+1)（ζ 为 primitive 2n-th 单位根 mod t）。 */
int vfhe_slot_roots(uint64_t *pts, const vfhe_params_t *p);

/** 打包：槽位值 slots[0..n-1] -> 明文多项式 poly[0..n-1]（R_t 系数）。 */
int vfhe_slot_encode(uint64_t *poly, const uint64_t *slots,
                     const uint64_t *pts, const vfhe_params_t *p);

/** 解包：明文多项式 poly -> 槽位值 slots（在 pts 上评估，同态）。 */
int vfhe_slot_decode(uint64_t *slots, const uint64_t *poly,
                     const uint64_t *pts, const vfhe_params_t *p);

/** 加密明文多项式（打包后任意明文；系数 mod t）。 */
int vfhe_encrypt_sym_poly(vfhe_ct_t *ct, const vfhe_sk_t *sk,
                          const uint64_t *poly);

/** 解密出明文多项式（mod t 系数），可再 vfhe_slot_decode。 */
int vfhe_decrypt_poly(uint64_t *poly, const vfhe_sk_t *sk, const vfhe_ct_t *ct);

/* ================================================================
 * 6. 自测 (PASS/FAIL 自动化判定)
 * ================================================================ */

/**
 * 全链路自测：
 *   T1 定点编码往返精度
 *   T2 对称/公钥 加密-解密往返
 *   T3 同态加法 (含负数中心化)
 *   T4 同态乘法 (深度 1)
 *   T5 混合表达式 ((a*b)+c) 深度 2
 *   T6 定点数同态加/乘
 * 打印逐项 [PASS]/[FAIL]。返回失败数 (0 = OK)。
 */
int vfhe_self_test(void);

#ifdef __cplusplus
}
#endif

#endif /* VLLM_FHE_H */
