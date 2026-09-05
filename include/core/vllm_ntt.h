/**
 * vllm_ntt.h - 负循环 NTT（mod x^n + 1，n = 2^k），公理驱动落地
 *
 * 对齐 axiom_registry.json：
 *   - axiom_isomorphic_superposition_004 (ntt_isomorphism)：
 *     多项式环 ≅ NTT 域，卷积坍缩为逐点乘法（c = a*b mod (x^n+1) 用 NTT 加速）。
 *   - axiom_modal_transition_conservation_005 (ntt_bijection)：
 *     iNTT(NTT(x)) = x，位级可验证的双向映射（自测覆盖）。
 *   - Goldilocks NTT prime：q-1 含 2n 因子（BFV q=2^64-2^32+1，q-1=2^32(2^32-1)；
 *     CKKS 素数 ≡ 1 mod 2n=2048），保证 2n-th 单位根存在。
 *
 * 架构：
 *   - negacyclic NTT：ψ 为 2n-th primitive root（ψ^n = -1），w = ψ² 为 n-th root。
 *     fwd:  A[i] = Σ_j a[j]·ψ^j·w^(ij)（先乘 ψ^j，DIT radix-2 蝶形）
 *     inv:  a[j] = n^{-1}·ψ^{-j}·Σ_i A[i]·w^(-ij)（蝶形用 w^{-1}，最后归一）
 *   - 模乘统一 __int128 % q（兼容 Goldilocks 64-bit 与 60-bit 素数）。
 *   - n 仅支持 2 的幂（本项目 n=1024）。
 */
#ifndef VLLM_NTT_H
#define VLLM_NTT_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint64_t q;          /* 模数（素数，q-1 被 2n 整除） */
    uint32_t n;          /* 环次数（2 的幂，如 1024） */
    uint64_t psi;        /* primitive 2n-th root（ψ^n = -1） */
    uint64_t n_inv;      /* n^{-1} mod q */
    uint64_t qinv;       /* Montgomery: -q^{-1} mod 2^64（R=2^64） */
    uint64_t r2;         /* Montgomery: R^2 mod q（toMont 因子） */
    int      use_mont;   /* q < 2^63 时启用 Montgomery（60-bit 素数安全；
                             64-bit Goldilocks 的 t+m·q 可溢 2^128，走 __int128） */
    uint64_t *wlen_fwd;  /* [log2(n)+1] 每 stage 的 w^(n/len)（Montgomery 域） */
    uint64_t *wlen_inv;  /* [log2(n)+1] 每 stage 的 w^(-n/len)（Montgomery 域） */
    uint64_t *psi_pow;   /* [n] ψ^j（Montgomery 域） */
    uint64_t *inv_psi_pow;/* [n] ψ^{-j}（Montgomery 域） */
    uint32_t *rev;       /* [n] 位反转表 */
    uint64_t *wpow_fwd;  /* [(lg+1)*(n/2)] 每 stage 预计算 w^(j)（Montgomery 域） */
    uint64_t *wpow_inv;  /* [(lg+1)*(n/2)] 每 stage 预计算 w^(-j)（Montgomery 域） */
    uint64_t *bufA;      /* [n] 工作区 */
    uint64_t *bufB;      /* [n] 工作区 */
} ntt_ctx_t;

/* 初始化：找 primitive 2n-th root + 预计算根表（q 需为素数且 2n | q-1） */
int  ntt_ctx_init(ntt_ctx_t *ctx, uint64_t q, uint32_t n);
void ntt_ctx_free(ntt_ctx_t *ctx);

/* 就地正/逆负循环 NTT */
void ntt_negacyclic_fwd(uint64_t *a, const ntt_ctx_t *ctx);
void ntt_negacyclic_inv(uint64_t *a, const ntt_ctx_t *ctx);

/* c = a*b mod (x^n+1)（NTT 逐点乘 + iNTT） */
void ntt_negacyclic_mul(uint64_t *c, const uint64_t *a, const uint64_t *b,
                        const ntt_ctx_t *ctx);

/* 自测（T13）：往返位级 + 与教科书 O(n²) negacyclic 位级对照 + 性能计时。
 * 返回失败项数。 */
int ntt_self_test(void);

#ifdef __cplusplus
}
#endif

#endif /* VLLM_NTT_H */
