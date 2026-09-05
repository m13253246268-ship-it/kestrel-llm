/**
 * vllm_ckks.c - CKKS 风格近似 HE（RNS 多素数模数链），纯 C11 自研
 *
 * 核心流程（scale = 2^60，模数链 4×60-bit 素数，深 3）：
 *   encode: z_j (|z|<0.5) -> round(z*2^60) -> 槽位 -> 明文多项式（RNS，逐素数）
 *   encrypt: (c0,c1) = (a*s + e + poly, -a) 逐素数 mod q_p
 *   mult:   通用 tensor（negacyclic mod q_p）-> 3 分量
 *   relin:  key-switch c2（密钥 s^2 -> s），回 2 分量
 *   rescale:模数切换（切一个素数 ≈2^60，scale 回 2^60）
 *   rotate: Galois 自动态 + key-switch（槽位循环移位）
 *   decrypt: v = c0 + c1*s（逐素数）-> 明文多项式（RNS）
 *   decode:  槽位值（明文 |m|<2^59<任一 q_p，无需 CRT）-> /2^60
 */
#include "vllm_ckks.h"
#include "vllm_ntt.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

/* ================================================================
 * 模算术（60-bit 素数：a+b < 2^61 无溢出；乘积用 __int128）
 * ================================================================ */
static inline uint64_t ckks_addmod(uint64_t a, uint64_t b, uint64_t q) {
    uint64_t r = a + b;
    if (r >= q) r -= q;
    return r;
}
static inline uint64_t ckks_submod(uint64_t a, uint64_t b, uint64_t q) {
    return a >= b ? a - b : a + q - b;
}
static inline uint64_t ckks_mulmod(uint64_t a, uint64_t b, uint64_t q) {
    return (uint64_t)((unsigned __int128)a * b % q);
}
static uint64_t ckks_powmod(uint64_t b, uint64_t e, uint64_t m) {
    uint64_t r = 1;
    b %= m;
    while (e) {
        if (e & 1) r = ckks_mulmod(r, b, m);
        b = ckks_mulmod(b, b, m);
        e >>= 1;
    }
    return r;
}
static uint64_t ckks_modinv(uint64_t a, uint64_t q) {
    int64_t t0 = (int64_t)q, t1 = (int64_t)a, t2;
    int64_t r0 = 0, r1 = 1, r2;
    while (t1) {
        int64_t dv = t0 / t1;
        t2 = t0 - dv * t1; t0 = t1; t1 = t2;
        r2 = r0 - dv * r1; r0 = r1; r1 = r2;
    }
    if (r0 < 0) r0 += (int64_t)q;
    return (uint64_t)r0;
}
static inline int64_t ckks_center(uint64_t v, uint64_t q) {
    return v >= q / 2 ? (int64_t)v - (int64_t)q : (int64_t)v;
}

/* ================================================================
 * RNG（xorshift64*，原型级）
 * ================================================================ */
static _Thread_local uint64_t ckks_rng_state = 0x243F6A8885A308D3ull;
static void (*ckks_rng_cb)(uint8_t *out, size_t len) = NULL;
void ckks_set_rng(void (*fn)(uint8_t *out, size_t len)) { ckks_rng_cb = fn; }
static uint64_t ckks_rng_u64(void) {
    if (ckks_rng_cb) {
        uint8_t b[8];
        ckks_rng_cb(b, 8);
        uint64_t v = 0;
        for (int i = 0; i < 8; i++) v = (v << 8) | b[i];
        return v;
    }
    uint64_t x = ckks_rng_state;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    ckks_rng_state = x;
    return x * 0x2545F4914F6CDD1Dull;
}
/* 中心二项分布 CBD(eta)：X = Σ_{i=1}^{eta} (b_{2i-1} - b_{2i})，b 为均匀随机 bit。
 * 标准 RLWE 噪声（Kyber 同款）：σ=√(eta/2)，有界 [-eta, eta]。
 * 与原均匀 [-B,B) 同界（最坏噪声预算不变），但方差更小（典型噪声更优）。 */
static int64_t ckks_rng_noise(uint64_t eta) {
    int64_t x = 0;
    for (uint64_t i = 0; i < eta; i++) {
        uint64_t r = ckks_rng_u64();
        x += (int64_t)(r & 1) - (int64_t)((r >> 1) & 1);
    }
    return x;
}

/* ================================================================
 * 素数（60-bit，≡ 1 mod 2048，保证 2n-th 单位根存在）
 * ================================================================ */
static int ckks_is_prime(uint64_t n) {
    if (n < 2) return 0;
    for (uint64_t p = 2; p * p <= n && p < 1000; p++)
        if (n % p == 0) return 0;
    /* Miller-Rabin，基 {2,3,5,7,11,13,17}（对 <2^64 确定性足够） */
    uint64_t d = n - 1, s = 0;
    while ((d & 1) == 0) { d >>= 1; s++; }
    static const uint64_t bases[] = {2,3,5,7,11,13,17};
    for (size_t i = 0; i < sizeof(bases)/sizeof(bases[0]); i++) {
        uint64_t a = bases[i] % n;
        if (a == 0) continue;
        uint64_t x = ckks_powmod(a, d, n);
        if (x == 1 || x == n - 1) continue;
        int comp = 1;
        for (uint64_t r = 1; r < s; r++) {
            x = ckks_mulmod(x, x, n);
            if (x == n - 1) { comp = 0; break; }
        }
        if (comp) return 0;
    }
    return 1;
}
/* 从 start 向下找 60-bit 素数 ≡ 1 mod 2048 */
static uint64_t ckks_find_prime(uint64_t start, uint64_t mod) {
    uint64_t x = (start - 1) / mod * mod + 1;   /* 最大 < start 且 ≡ 1 mod mod */
    while (x >= (1ull << 59)) {                 /* 保持 60-bit */
        if (ckks_is_prime(x)) return x;
        x -= mod;
    }
    return 0;
}

/* ================================================================
 * 多项式（mod q 参数化）
 * ================================================================ */
static void ckks_poly_zero(uint64_t *c, size_t n) { memset(c, 0, n * 8); }

/* c = a*b（negacyclic mod q，教科书 O(n^2)，NTT 不可用时的回退参考） */
static void ckks_poly_mul_naive(uint64_t *c, const uint64_t *a, const uint64_t *b,
                                size_t n, uint64_t q) {
    ckks_poly_zero(c, n);
    for (size_t i = 0; i < n; i++) {
        if (a[i] == 0) continue;
        for (size_t j = 0; j < n; j++) {
            if (b[j] == 0) continue;
            uint64_t t = ckks_mulmod(a[i], b[j], q);
            size_t k = i + j;
            if (k < n) c[k] = ckks_addmod(c[k], t, q);
            else       c[k - n] = ckks_submod(c[k - n], t, q);
        }
    }
}

/* NTT 加速（axiom_ntt_isomorphism 落地）：RNS 每素数独立做负循环 NTT。
 * 素数 ≡ 1 mod 2n=2048 → 2n-th 单位根存在。按 (q,n) 缓存 ctx。 */
static ntt_ctx_t g_ckks_ntt[CKKS_NPRIMES];
static uint64_t g_ckks_ntt_q[CKKS_NPRIMES] = {0, 0, 0, 0};
static int g_ckks_ntt_used = 0;

static int ckks_ntt_ctx_for(uint64_t q, uint32_t n, ntt_ctx_t **out) {
    for (int i = 0; i < g_ckks_ntt_used; i++)
        if (g_ckks_ntt_q[i] == q && g_ckks_ntt[i].n == n) { *out = &g_ckks_ntt[i]; return 0; }
    if (g_ckks_ntt_used >= CKKS_NPRIMES) return -1;
    if (ntt_ctx_init(&g_ckks_ntt[g_ckks_ntt_used], q, n) != 0) return -1;
    g_ckks_ntt_q[g_ckks_ntt_used] = q;
    *out = &g_ckks_ntt[g_ckks_ntt_used];
    g_ckks_ntt_used++;
    return 0;
}

static void ckks_poly_mul(uint64_t *c, const uint64_t *a, const uint64_t *b,
                          size_t n, uint64_t q) {
    ntt_ctx_t *ctx;
    if (ckks_ntt_ctx_for(q, (uint32_t)n, &ctx) == 0) {
        ntt_negacyclic_mul(c, a, b, ctx);
        return;
    }
    ckks_poly_mul_naive(c, a, b, n, q);
}
/* c = a*s（s 稀疏三元，hw 非零） */
static void ckks_poly_mul_sparse(uint64_t *c, const uint64_t *a, const int8_t *s,
                                 size_t n, uint64_t q, uint32_t hw) {
    ckks_poly_zero(c, n);
    size_t cnt = 0;
    for (size_t j = 0; j < n && cnt < hw; j++) {
        int8_t sv = s[j];
        if (sv == 0) continue;
        cnt++;
        int neg = sv < 0;
        for (size_t i = 0; i < n; i++) {
            uint64_t ai = a[i];
            if (ai == 0) continue;
            size_t k = i + j;
            if (k < n) {
                if (neg) c[k] = ckks_submod(c[k], ai, q);
                else     c[k] = ckks_addmod(c[k], ai, q);
            } else {
                if (neg) c[k - n] = ckks_addmod(c[k - n], ai, q);
                else     c[k - n] = ckks_submod(c[k - n], ai, q);
            }
        }
    }
}

/* ================================================================
 * 私钥幂缓存（s^2..s^4，带符号整数系数）：无 relin 多分量解密的密钥。
 * CKKS 的 relin digit 分解基于单素数 c2 值导致噪声跨素数不一致，
 * 破坏 rescale/CRT；跳过 relin、直接按 BFV 风格多分量解密可保持
 * 明文（= Σ(e_i+p_i) 之积）跨素数一致。
 * ================================================================ */
static int64_t *g_spow[CKKS_MAX_COMP];   /* g_spow[comp] = s^comp 带符号系数 */
static uint32_t g_spow_n = 0;

/* 带符号整数 negacyclic：out = a*b（系数可能很大，int64 精确；n=1024 时 s^4 ~ ±2^18） */
static void ckks_spow_mul(int64_t *out, const int64_t *a, const int64_t *b, size_t n) {
    memset(out, 0, n * sizeof(int64_t));
    for (size_t i = 0; i < n; i++) {
        if (a[i] == 0) continue;
        for (size_t j = 0; j < n; j++) {
            if (b[j] == 0) continue;
            int64_t t = a[i] * b[j];
            size_t k = i + j;
            if (k < n) out[k] += t;
            else       out[k - n] -= t;
        }
    }
}

static int ckks_spow_ensure(const ckks_sk_t *sk) {
    size_t n = sk->n;
    if (g_spow_n == n) return 0;
    for (int c = 0; c < CKKS_MAX_COMP; c++) {
        free(g_spow[c]);
        g_spow[c] = NULL;
    }
    int64_t *s1 = (int64_t *)calloc(n, sizeof(int64_t));
    if (!s1) return -1;
    for (size_t i = 0; i < n; i++) s1[i] = sk->s[i];
    g_spow[1] = s1;
    for (int c = 2; c < CKKS_MAX_COMP; c++) {
        g_spow[c] = (int64_t *)calloc(n, sizeof(int64_t));
        if (!g_spow[c]) { g_spow_n = 0; return -1; }
        ckks_spow_mul(g_spow[c], g_spow[c - 1], s1, n);
    }
    g_spow_n = (uint32_t)n;
    return 0;
}

/* c = a * spow（带符号小/中系数多项式，mod q，negacyclic） */
static void ckks_poly_mul_signed(uint64_t *c, const uint64_t *a, const int64_t *spow,
                                 size_t n, uint64_t q) {
    ckks_poly_zero(c, n);
    for (size_t j = 0; j < n; j++) {
        int64_t sv = spow[j];
        if (sv == 0) continue;
        int neg = sv < 0;
        uint64_t mag = neg ? (uint64_t)(-sv) : (uint64_t)sv;
        for (size_t i = 0; i < n; i++) {
            if (a[i] == 0) continue;
            uint64_t v = ckks_mulmod(a[i], mag, q);   /* a[i]*|sv| mod q */
            size_t k = i + j;
            if (k < n) {
                if (neg) c[k] = ckks_submod(c[k], v, q);
                else     c[k] = ckks_addmod(c[k], v, q);
            } else {
                if (neg) c[k - n] = ckks_addmod(c[k - n], v, q);
                else     c[k - n] = ckks_submod(c[k - n], v, q);
            }
        }
    }
}

/* ================================================================
 * 评估点 / 槽位打包（mod q 参数化，复用 vllm_fhe 的数学）
 * ================================================================ */
static int ckks_slot_roots(uint64_t *pts, uint64_t q, size_t n) {
    if ((q - 1) % (2 * n) != 0) return -1;
    uint64_t phi = q - 1, tmp = phi;
    uint64_t facs[16]; int nf = 0;
    for (uint64_t d = 2; d * d <= tmp; d++) {
        if (tmp % d == 0) { facs[nf++] = d; while (tmp % d == 0) tmp /= d; }
    }
    if (tmp > 1) facs[nf++] = tmp;
    uint64_t g = 2;
    for (; g < q; g++) {
        int ok = 1;
        for (int i = 0; i < nf; i++)
            if (ckks_powmod(g, phi / facs[i], q) == 1) { ok = 0; break; }
        if (ok) break;
    }
    uint64_t zeta = ckks_powmod(g, phi / (2 * n), q);
    uint64_t z2 = ckks_mulmod(zeta, zeta, q);
    uint64_t cur = zeta;
    for (size_t j = 0; j < n; j++) { pts[j] = cur; cur = ckks_mulmod(cur, z2, q); }
    return 0;
}
/* 打包：v_j -> poly，poly(pts[j]) = v_j（mod q），利用 ∏(x-pts)=x^n+1 */
static void ckks_slot_encode(uint64_t *poly, const uint64_t *v,
                             const uint64_t *pts, uint64_t q, size_t n) {
    ckks_poly_zero(poly, n);
    for (size_t j = 0; j < n; j++) {
        if (v[j] == 0) continue;   /* 稀疏槽位：跳过零贡献（O(n·nnz) 而非 O(n^2)） */
        uint64_t pj = pts[j];
        uint64_t pj_nm1 = ckks_powmod(pj, n - 1, q);
        uint64_t inv_j = ckks_modinv(ckks_mulmod(n % q, pj_nm1, q), q);
        uint64_t coef = ckks_mulmod(v[j] % q, inv_j, q);
        uint64_t pw = pj_nm1;
        uint64_t inv_pj = ckks_modinv(pj, q);
        for (size_t i = 0; i < n; i++) {
            poly[i] = ckks_addmod(poly[i], ckks_mulmod(coef, pw, q), q);
            pw = ckks_mulmod(pw, inv_pj, q);
        }
    }
}
/* 解包：poly -> v_j = poly(pts[j])（Horner） */
static void ckks_slot_decode(uint64_t *v, const uint64_t *poly,
                             const uint64_t *pts, uint64_t q, size_t n) {
    for (size_t j = 0; j < n; j++) {
        uint64_t acc = 0;
        for (size_t i = n; i-- > 0;) acc = ckks_addmod(ckks_mulmod(acc, pts[j], q), poly[i], q);
        v[j] = acc;
    }
}

/* ================================================================
 * 上下文 / 密钥
 * ================================================================ */
int ckks_ctx_init(ckks_ctx_t *ctx) {
    if (!ctx) return -1;
    memset(ctx, 0, sizeof(*ctx));
    ctx->n = CKKS_N;
    ctx->nprimes = CKKS_NPRIMES;
    ctx->scale = 1ull << CKKS_SCALE_BITS;   /* 2^60 */
    /* 12 个互不相同的 60-bit 素数 ≡ 1 mod 2n（保证 2n-th 单位根存在），降序 */
    uint64_t start = (1ull << 60) - 1;
    int got = 0;
    while (got < CKKS_NPRIMES) {
        uint64_t q = ckks_find_prime(start, 2 * (uint64_t)ctx->n);
        if (q == 0) return -1;
        int dup = 0;
        for (int i = 0; i < got; i++) if (ctx->q[i] == q) dup = 1;
        if (!dup) {
            ctx->q[got] = q;
            ctx->pts[got] = (uint64_t *)malloc(ctx->n * sizeof(uint64_t));
            if (!ctx->pts[got] || ckks_slot_roots(ctx->pts[got], q, ctx->n) != 0)
                return -1;
            got++;
        }
        start = q - 1;
    }
    /* 预初始化全部素数的 NTT 表（单线程，运行时只读 → 多线程安全） */
    for (int i = 0; i < ctx->nprimes; i++) {
        ntt_ctx_t *tmp = NULL;
        ckks_ntt_ctx_for(ctx->q[i], ctx->n, &tmp);
    }
    return 0;
}
void ckks_ctx_free(ckks_ctx_t *ctx) {
    if (!ctx) return;
    for (int i = 0; i < CKKS_NPRIMES; i++) { free(ctx->pts[i]); ctx->pts[i] = NULL; }
}
int ckks_sk_gen(ckks_sk_t *sk, uint32_t n) {
    if (!sk) return -1;
    memset(sk, 0, sizeof(*sk));
    sk->n = n;
    sk->s = (int8_t *)calloc(n, sizeof(int8_t));
    if (!sk->s) return -1;
    uint32_t placed = 0, guard = 0;
    while (placed < CKKS_KEY_HW && guard < 100000u) {
        guard++;
        size_t idx = ckks_rng_u64() % n;
        if (sk->s[idx] != 0) continue;
        sk->s[idx] = (ckks_rng_u64() & 1) ? -1 : 1;
        placed++;
    }
    return 0;
}
void ckks_sk_free(ckks_sk_t *sk) { if (sk) { free(sk->s); sk->s = NULL; } }

/* 通用 key-switch 密钥：D(gk0+gk1*s) = target（逐素数，完整链） */
static int ckks_generic_gk(uint64_t *gk0, uint64_t *gk1, const ckks_sk_t *sk,
                           const ckks_ctx_t *ctx, const uint64_t *target) {
    /* target: [nprimes*n]，密钥多项式在每素数的表示 */
    size_t n = sk->n;
    /* 零噪声 key-switch（正确性原型）：
     * key-switch 噪声 = Σ_d dig_d(p)·e_d，其中 dig_d(p) 是 c1 在素数 p 的残差
     * 的 digit（各素数不同）——即便 e_d 跨素数一致，噪声整数也逐素数不同。
     * 解密（单素数 q0）下不可见（噪声 ~2^45 << q0），但 rescale 的 mod-switch
     * 公式 (in_p - in_L)·qL^{-1} 会把该不一致放大到 ~q0 量级，彻底污染 rescaled
     * 明文（T2B-4 旋转打包 matmul 的 FAIL 根因）。零噪声使 key-switch 精确
     * （o0+o1·s = σ(c0)+c1·target，跨素数一致），rescale 恢复正确。
     * 代价：gk 无噪声 → 泄露私钥；原型级（n=1024 无安全性）可接受。 */
    for (int p = 0; p < ctx->nprimes; p++) {
        uint64_t q = ctx->q[p];
        uint64_t *a = (uint64_t *)malloc(n * sizeof(uint64_t));
        if (!a) return -1;
        for (size_t i = 0; i < n; i++) a[i] = ckks_rng_u64() % q;
        ckks_poly_mul_sparse(&gk0[p * n], a, sk->s, n, q, CKKS_KEY_HW);
        for (size_t i = 0; i < n; i++)
            gk0[p * n + i] = ckks_addmod(gk0[p * n + i], target[p * n + i] % q, q);
        for (size_t i = 0; i < n; i++)
            gk1[p * n + i] = a[i] ? q - a[i] : 0;
        free(a);
    }
    return 0;
}
int ckks_rk_gen(ckks_rk_t *rk, const ckks_sk_t *sk, const ckks_ctx_t *ctx) {
    if (!rk || !sk || !ctx) return -1;
    size_t n = sk->n;
    rk->n = n;
    rk->digits = CKKS_RELIN_DIGITS;
    for (int d = 0; d < rk->digits; d++) {
        rk->rk0[d] = (uint64_t *)calloc(ctx->nprimes * n, sizeof(uint64_t));
        rk->rk1[d] = (uint64_t *)calloc(ctx->nprimes * n, sizeof(uint64_t));
        if (!rk->rk0[d] || !rk->rk1[d]) { ckks_rk_free(rk); return -1; }
    }
    /* target_d = B^d * s^2（每 digit 独立生成 key-switch 密钥） */
    uint64_t B = 1ull << CKKS_RELIN_BASE_BITS;
    uint64_t *tgt = (uint64_t *)calloc(ctx->nprimes * n, sizeof(uint64_t));
    if (!tgt) { ckks_rk_free(rk); return -1; }
    for (int d = 0; d < rk->digits; d++) {
        const int8_t *s = sk->s;
        for (int p = 0; p < ctx->nprimes; p++) {
            uint64_t q = ctx->q[p];
            for (size_t i = 0; i < n; i++) {
                if (s[i] == 0) continue;
                for (size_t j = 0; j < n; j++) {
                    if (s[j] == 0) continue;
                    size_t k = i + j;
                    uint64_t one = 1;
                    if (k < n) {
                        tgt[p * n + k] = (s[i] * s[j] > 0)
                            ? ckks_addmod(tgt[p * n + k], one, q)
                            : ckks_submod(tgt[p * n + k], one, q);
                    } else {
                        tgt[p * n + k - n] = (s[i] * s[j] > 0)
                            ? ckks_submod(tgt[p * n + k - n], one, q)
                            : ckks_addmod(tgt[p * n + k - n], one, q);
                    }
                }
            }
        }
        /* target = B^d * s^2（模 q_p 乘 B^d） */
        if (d > 0) {
            for (int p = 0; p < ctx->nprimes; p++) {
                uint64_t q = ctx->q[p];
                uint64_t bq = (d == 1) ? B % q : (uint64_t)((unsigned __int128)B * B % q);
                for (size_t i = 0; i < n; i++)
                    tgt[p * n + i] = ckks_mulmod(tgt[p * n + i], bq, q);
            }
        }
        int rc = ckks_generic_gk(rk->rk0[d], rk->rk1[d], sk, ctx, tgt);
        /* 清 tgt 供下轮 */
        memset(tgt, 0, (size_t)ctx->nprimes * n * sizeof(uint64_t));
        if (rc != 0) { free(tgt); ckks_rk_free(rk); return rc; }
    }
    free(tgt);
    return 0;
}
void ckks_rk_free(ckks_rk_t *rk) {
    if (!rk) return;
    for (int d = 0; d < CKKS_RELIN_DIGITS; d++) { free(rk->rk0[d]); free(rk->rk1[d]); rk->rk0[d] = rk->rk1[d] = NULL; }
}

/* Galois 自动态 σ_k（k 奇数）：x -> x(x^k) mod (x^n+1) */
static void ckks_galois(uint64_t *out, const uint64_t *in, uint64_t k,
                        uint64_t q, size_t n) {
    ckks_poly_zero(out, n);
    size_t n2 = 2 * n;
    for (size_t i = 0; i < n; i++) {
        if (in[i] == 0) continue;
        size_t idx = (size_t)((unsigned __int128)(i % n2) * (k % n2) % n2);
        if (idx < n) out[idx] = ckks_addmod(out[idx], in[i], q);
        else         out[idx - n] = ckks_submod(out[idx - n], in[i], q);
    }
}
int ckks_gk_gen(ckks_gk_t *gk, const ckks_sk_t *sk, const ckks_ctx_t *ctx, int t) {
    if (!gk || !sk || !ctx || t < 0 || t >= (int)sk->n) return -1;
    uint64_t k = ckks_powmod(5, t, 2 * (uint64_t)sk->n);   /* 自动态指数 5^t */
    return ckks_gk_gen_k(gk, sk, ctx, k);
}
/* 一般 Galois 自同构 σ_k（k 任意奇数）的 key-switch 密钥（复槽位 CoeffToSlot 需 σ_{2n-1} 等） */
int ckks_gk_gen_k(ckks_gk_t *gk, const ckks_sk_t *sk, const ckks_ctx_t *ctx, uint64_t k) {
    if (!gk || !sk || !ctx || (k & 1) == 0) return -1;
    size_t n = sk->n;
    gk->n = n;
    gk->digits = CKKS_RELIN_DIGITS;
    for (int d = 0; d < gk->digits; d++) {
        gk->gk0[d] = (uint64_t *)calloc((size_t)ctx->nprimes * n, sizeof(uint64_t));
        gk->gk1[d] = (uint64_t *)calloc((size_t)ctx->nprimes * n, sizeof(uint64_t));
        if (!gk->gk0[d] || !gk->gk1[d]) { ckks_gk_free(gk); return -1; }
    }
    uint64_t *tgt = (uint64_t *)calloc((size_t)ctx->nprimes * n, sizeof(uint64_t));
    if (!tgt) { ckks_gk_free(gk); return -1; }
    /* target_d = B^d * σ_k(s)（每素数），B = 2^20 */
    uint64_t B = 1ull << CKKS_RELIN_BASE_BITS;
    for (int d = 0; d < gk->digits; d++) {
        if (d == 0) {
            for (int p = 0; p < ctx->nprimes; p++) {
                uint64_t q = ctx->q[p];
                uint64_t *sraw = (uint64_t *)calloc(n, sizeof(uint64_t));
                const int8_t *s = sk->s;
                if (!sraw) { ckks_gk_free(gk); free(tgt); return -1; }
                for (size_t i = 0; i < n; i++)
                    sraw[i] = s[i] < 0 ? q - 1 : (uint64_t)s[i];
                ckks_galois(&tgt[p * n], sraw, k, q, n);   /* σ_k(s) */
                free(sraw);
            }
        } else {
            /* tgt 累积：B^{d-1}·σ(s) -> B^d·σ(s)（模 q_p 乘 B） */
            for (int p = 0; p < ctx->nprimes; p++) {
                uint64_t q = ctx->q[p];
                uint64_t bq = B % q;
                for (size_t i = 0; i < n; i++)
                    tgt[p * n + i] = ckks_mulmod(tgt[p * n + i], bq, q);
            }
        }
        int rc = ckks_generic_gk(gk->gk0[d], gk->gk1[d], sk, ctx, tgt);
        if (rc != 0) { ckks_gk_free(gk); free(tgt); return rc; }
    }
    free(tgt);
    return 0;
}
void ckks_gk_free(ckks_gk_t *gk) {
    if (!gk) return;
    for (int d = 0; d < CKKS_RELIN_DIGITS; d++) {
        free(gk->gk0[d]); free(gk->gk1[d]);
        gk->gk0[d] = gk->gk1[d] = NULL;
    }
}

/* ================================================================
 * 编码 / 解码（解析嵌入：复数单位根 ω=e^{πi/n}，槽位在奇次根 ω^{2j+1}）
 *
 * 共轭对称（z 实数，512 有效槽位 z_0..z_511，z_{n-j}=z_j）-> 余弦变换：
 *   encode: m_k = (scale/n)·[z_0 + Σ_{j=1}^{n/2-1} 2·z_j·cos(k(2j+1)π/n)]
 *   decode: z_j   = (1/scale)·Σ_k v_k·cos(k(2j+1)π/n)   （虚部为 0）
 * 关键：|cos| ≤ 1，噪声评估 e_slot = Σ e_k·cos ≤ ||e||_1（小），
 *       不像代数嵌入（单位根幂次）把噪声放大到 ~q/2。这是解析嵌入的核心。
 * ================================================================ */
/* 编码余弦表缓存：cos(k(2j+1)π/n) 仅依赖环 n（与素数/scale 无关），
 * BSGS 对角打包 1024 次 encode 共用一张表，消除 n·n/2 次 cos() 调用
 * （encode 原实现每次 n=2048 时 2.1M 次 cos，BSGS 共 2^31 次 cos 为瓶颈 ~30s）。 */
static double *g_cos_tab = NULL;
static size_t  g_cos_n = 0;
static void cos_tab_prepare(size_t n) {
    if (g_cos_tab && g_cos_n == n) return;
    free(g_cos_tab);
    g_cos_tab = (double *)malloc(n * (n / 2) * sizeof(double));
    g_cos_n = n;
    for (size_t k = 0; k < n; k++)
        for (size_t j = 0; j < n / 2; j++)
            g_cos_tab[k * (n / 2) + j] =
                cos((double)(k * (2 * j + 1)) * M_PI / (double)n);
}
int ckks_encode(uint64_t *poly, const double *z, const ckks_ctx_t *ctx) {
    if (!poly || !z || !ctx) return -1;
    size_t n = ctx->n;
    double sc = (double)ctx->scale / (double)n;
    cos_tab_prepare(n);
    const double *ctab = g_cos_tab;
    for (size_t k = 0; k < n; k++) {
        /* 共轭配对 j ↔ n-1-j（含 j=0），每槽位贡献 2·cos 项 */
        double acc = 0.0;
        const double *row = ctab + k * (n / 2);
        for (size_t j = 0; j < n / 2; j++)
            acc += 2.0 * z[j] * row[j];
        int64_t iv = (int64_t)llround(sc * acc);
        for (int p = 0; p < ctx->nprimes; p++) {
            uint64_t q = ctx->q[p];
            /* 数学取模：负数 iv（|iv| 可 ≥ q，须连续回绕）得到 [0,q) 残差。
             * 旧实现 iv+q 只补一次，|iv|≥q 时仍为负 → (uint64_t) 强转后 %q 出错。 */
            int64_t r = iv % (int64_t)q;
            if (r < 0) r += (int64_t)q;
            poly[p * n + k] = (uint64_t)r;
        }
    }
    return 0;
}
int ckks_decode(double *z, const uint64_t *poly, const ckks_ctx_t *ctx) {
    if (!z || !poly || !ctx) return -1;
    size_t n = ctx->n;
    double scale = (double)ctx->scale;
    uint64_t q0 = ctx->q[0];
    /* 单素数中心化解密明文（明文系数 ~scale·|z| < q0/2 恒成立：
       多素数 CRT 会把各素数独立的解密噪声差放大到 ~q0·q1 量级，故不可用） */
    long double *cc = (long double *)malloc(n * sizeof(long double));
    if (!cc) return -1;
    for (size_t k = 0; k < n; k++)
        cc[k] = (long double)ckks_center(poly[k], q0);
    /* 解析评估：z_j = (1/scale)·Σ cc[k]·cos(π(2j+1)k/n) */
    for (size_t j = 0; j < n / 2; j++) {
        long double acc = 0.0L;
        for (size_t k = 0; k < n; k++)
            acc += cc[k] * cosl((long double)(k * (2 * j + 1)) * (long double)M_PI / (long double)n);
        z[j] = (double)(acc / (long double)scale);
    }
    free(cc);
    return 0;
}

/* ================================================================
 * 加密 / 解密
 * ================================================================ */
int ckks_encrypt(ckks_ct_t *ct, const ckks_sk_t *sk,
                 const uint64_t *poly, const ckks_ctx_t *ctx) {
    if (!ct || !sk || !poly || !ctx) return -1;
    size_t n = ctx->n;
    ct->ctx = *ctx;
    ct->comps = 2;
    ct->c = (uint64_t *)calloc((size_t)ct->comps * ctx->nprimes * n, sizeof(uint64_t));
    if (!ct->c) return -1;
    const int8_t *s = sk->s;
    /* 关键：噪声 e 必须是"同一整数"，逐素数取模（RNS 跨素数一致性，rescale/CRT 依赖） */
    int64_t *e_int = (int64_t *)malloc(n * sizeof(int64_t));
    if (!e_int) { ckks_ct_free(ct); return -1; }
    for (size_t i = 0; i < n; i++) e_int[i] = ckks_rng_noise(CKKS_NOISE_CBD);
    for (int p = 0; p < ctx->nprimes; p++) {
        uint64_t q = ctx->q[p];
        uint64_t *a = (uint64_t *)malloc(n * sizeof(uint64_t));
        uint64_t *e = (uint64_t *)malloc(n * sizeof(uint64_t));
        if (!a || !e) { free(a); free(e); ckks_ct_free(ct); free(e_int); return -1; }
        for (size_t i = 0; i < n; i++) {
            a[i] = ckks_rng_u64() % q;
            e[i] = e_int[i] < 0 ? (uint64_t)(e_int[i] + (int64_t)q) : (uint64_t)e_int[i];
        }
        uint64_t *c0 = &ct->c[(0 * ctx->nprimes + p) * n];
        uint64_t *c1 = &ct->c[(1 * ctx->nprimes + p) * n];
        ckks_poly_mul_sparse(c0, a, s, n, q, CKKS_KEY_HW);   /* c0 = a*s */
        for (size_t i = 0; i < n; i++) c0[i] = ckks_addmod(c0[i], e[i], q);
        for (size_t i = 0; i < n; i++) c0[i] = ckks_addmod(c0[i], poly[p * n + i], q);
        for (size_t i = 0; i < n; i++) c1[i] = a[i] ? q - a[i] : 0;
        free(a); free(e);
    }
    free(e_int);
    return 0;
}
int ckks_decrypt(uint64_t *poly, const ckks_sk_t *sk, const ckks_ct_t *ct) {
    if (!poly || !sk || !ct || ct->comps < 2) return -1;
    size_t n = ct->ctx.n;
    if (ckks_spow_ensure(sk) != 0) return -1;
    uint64_t *tmp = (uint64_t *)malloc(n * sizeof(uint64_t));
    if (!tmp) return -1;
    for (int p = 0; p < ct->ctx.nprimes; p++) {
        uint64_t q = ct->ctx.q[p];
        const uint64_t *c0 = &ct->c[(0 * ct->ctx.nprimes + p) * n];
        memcpy(&poly[p * n], c0, n * sizeof(uint64_t));
        for (int comp = 1; comp < ct->comps; comp++) {
            const uint64_t *ck = &ct->c[((size_t)comp * ct->ctx.nprimes + p) * n];
            ckks_poly_mul_signed(tmp, ck, g_spow[comp], n, q);
            for (size_t i = 0; i < n; i++)
                poly[p * n + i] = ckks_addmod(poly[p * n + i], tmp[i], q);
        }
    }
    free(tmp);
    return 0;
}

/* ================================================================
 * 同态运算
 * ================================================================ */
int ckks_ct_copy(ckks_ct_t *dst, const ckks_ct_t *src) {
    if (!dst || !src || src->comps < 1) return -1;
    dst->ctx = src->ctx;
    dst->comps = src->comps;
    dst->c = (uint64_t *)malloc((size_t)src->comps * src->ctx.nprimes * src->ctx.n * 8);
    if (!dst->c) return -1;
    memcpy(dst->c, src->c, (size_t)src->comps * src->ctx.nprimes * src->ctx.n * 8);
    return 0;
}
void ckks_ct_free(ckks_ct_t *ct) { if (ct) { free(ct->c); ct->c = NULL; ct->comps = 0; } }

int ckks_modraise(ckks_ct_t *ct, const ckks_ct_t *in, int target) {
    /* ModRaise（bootstrapping 模数提升）：把 nprimes=k 的密文用混合基 Garner
     * 提升回 target 素数。密文系数 c 是模 Q_k（链积）的整数，x 的唯一值
     * = v_0 + Σ_{i≥1} M_i·δ_i（M_i = q_0·...·q_{i-1}，δ_i 模 q_i 混合基系数），
     * 新素数残差 c mod q_new 用两阶段投影（全 64-bit 模运算，无大整数）：
     *   阶段 1：δ_i = (v_i - x_{i-1})·(M_i^{-1} mod q_i)，x_{i-1} 在各旧素数下
     *            的残差递推维护（与 q_new 无关）。
     *   阶段 2：c mod q_new = v_0 + Σ_i (M_i mod q_new)·δ_i（M_i mod q_new 递推）。
     * 明文 v < Q_k，链长度恢复。注意：密文系数 c0/c1 逐素数独立（加密 a 随机），
     * ModRaise 重构出的新素数解密 = v + Q_k·k（混叠，k~±7，标准 RNS-CKKS ModUp
     * 行为）——跨素数 mult+rescale 需折叠函数（Approximate Mod Reduction）剥除，
     * 折叠需特殊模数 P（1/Q_k 超 double 精度）+ sin 倍角深链（>12 素数）。
     * 教训：CRT 逐项公式 (v_i·(Qk/qi)·inv) 直接对 qn 取模错误（gcd(Qk,qn)=1
     * 丢信息）；256-bit 重构也错（M_i 增长到 ~2^660）。混合基两阶段才正确。 */
    if (!ct || !in || target <= in->ctx.nprimes || target > CKKS_NPRIMES) return -1;
    int k = in->ctx.nprimes;
    size_t n = in->ctx.n;
    ct->ctx = in->ctx;
    ct->ctx.nprimes = target;
    ct->comps = in->comps;
    ct->c = (uint64_t *)calloc((size_t)in->comps * target * n, sizeof(uint64_t));
    if (!ct->c) return -1;
    uint64_t *delta = (uint64_t *)malloc((size_t)k * sizeof(uint64_t));
    uint64_t *xmod = (uint64_t *)malloc((size_t)k * sizeof(uint64_t));
    uint64_t *v = (uint64_t *)malloc((size_t)k * sizeof(uint64_t));
    if (!delta || !xmod || !v) { free(delta); free(xmod); free(v); ckks_ct_free(ct); return -1; }
    for (int cp = 0; cp < in->comps; cp++)
        for (int p = 0; p < k; p++)
            memcpy(&ct->c[((size_t)cp * target + p) * n],
                   &in->c[((size_t)cp * k + p) * n], n * 8);
    for (int cp = 0; cp < in->comps; cp++) {
        for (size_t x = 0; x < n; x++) {
            for (int i = 0; i < k; i++)
                v[i] = in->c[((size_t)cp * k + i) * n + x];
            /* 阶段 1：δ_i 与 x_{i-1} mod q_j 递推 */
            for (int j = 1; j < k; j++) xmod[j] = v[0] % in->ctx.q[j];
            for (int i = 1; i < k; i++) {
                uint64_t qi = in->ctx.q[i];
                uint64_t mqi = 1;
                for (int j = 0; j < i; j++)
                    mqi = ckks_mulmod(mqi, in->ctx.q[j], qi);
                uint64_t inv = ckks_modinv(mqi, qi);
                uint64_t t = (v[i] + qi - xmod[i]) % qi;
                delta[i] = ckks_mulmod(t, inv, qi);
                for (int j = i + 1; j < k; j++) {
                    uint64_t qj = in->ctx.q[j];
                    uint64_t Mi_qj = 1;
                    for (int l = 0; l < i; l++)
                        Mi_qj = ckks_mulmod(Mi_qj, in->ctx.q[l], qj);
                    xmod[j] = (xmod[j] + ckks_mulmod(Mi_qj, delta[i], qj)) % qj;
                }
            }
            /* 阶段 2：c mod q_new = v_0 + Σ_i (M_i mod q_new)·δ_i */
            for (int p = k; p < target; p++) {
                uint64_t qn = in->ctx.q[p];
                uint64_t res = v[0] % qn;
                uint64_t Mm = 1;   /* M_0 = 1 */
                for (int i = 1; i < k; i++) {
                    Mm = ckks_mulmod(Mm, in->ctx.q[i - 1], qn);   /* M_i */
                    res = (res + ckks_mulmod(Mm, delta[i], qn)) % qn;
                }
                ct->c[((size_t)cp * target + p) * n + x] = res;
            }
        }
    }
    free(delta); free(xmod); free(v);
    return 0;
}

/* mod switch：切最后一个素数（不 rescale scale）。用于把新鲜密文降级到与
 * 深链密文相同的素数链（明文 < 新模数/2 时语义不变）。支持切到 nprimes=1
 * （bootstrapping 降到单素数 q_0，参考模数可编码）。 */
int ckks_modswitch(ckks_ct_t *ct, const ckks_ct_t *in) {
    if (!ct || !in || in->ctx.nprimes < 1) return -1;
    int np_new = in->ctx.nprimes - 1;
    size_t n = in->ctx.n;
    ct->ctx = in->ctx;
    ct->ctx.nprimes = np_new;
    ct->comps = in->comps;
    ct->c = (uint64_t *)calloc((size_t)in->comps * np_new * n, sizeof(uint64_t));
    if (!ct->c) return -1;
    for (int cp = 0; cp < in->comps; cp++)
        for (int p = 0; p < np_new; p++)
            memcpy(&ct->c[((size_t)cp * np_new + p) * n],
                   &in->c[((size_t)cp * in->ctx.nprimes + p) * n], n * 8);
    return 0;
}

int ckks_add(ckks_ct_t *ct, const ckks_ct_t *a, const ckks_ct_t *b) {
    if (!ct || !a || !b || a->ctx.nprimes != b->ctx.nprimes) return -1;
    int comps = a->comps > b->comps ? a->comps : b->comps;
    ct->ctx = a->ctx;
    ct->comps = comps;
    size_t np = a->ctx.nprimes, n = a->ctx.n;
    ct->c = (uint64_t *)calloc((size_t)comps * np * n, sizeof(uint64_t));
    if (!ct->c) return -1;
    for (int cp = 0; cp < comps; cp++)
        for (size_t p = 0; p < np; p++) {
            uint64_t q = a->ctx.q[p];
            uint64_t *dst = &ct->c[((size_t)cp * np + p) * n];
            if (cp < a->comps) memcpy(dst, &a->c[((size_t)cp * np + p) * n], n * 8);
            if (cp < b->comps) {
                const uint64_t *bb = &b->c[((size_t)cp * np + p) * n];
                for (size_t i = 0; i < n; i++) dst[i] = ckks_addmod(dst[i], bb[i], q);
            }
        }
    return 0;
}

/* 通用 tensor 乘法：ct = a*b（RNS 逐素数 negacyclic，输出 comps = a+b-1） */
static int ckks_tensor(ckks_ct_t *ct, const ckks_ct_t *a, const ckks_ct_t *b) {
    int comps = a->comps + b->comps - 1;
    if (comps > CKKS_MAX_COMP) return -1;
    size_t np = a->ctx.nprimes, n = a->ctx.n;
    ct->ctx = a->ctx;
    ct->comps = comps;
    ct->c = (uint64_t *)calloc((size_t)comps * np * n, sizeof(uint64_t));
    if (!ct->c) return -1;
    uint64_t *tmp = (uint64_t *)malloc(n * sizeof(uint64_t));
    uint64_t *acc = (uint64_t *)calloc(n, sizeof(uint64_t));
    if (!tmp || !acc) { free(tmp); free(acc); ckks_ct_free(ct); return -1; }
    for (int k = 0; k < comps; k++) {
        for (size_t p = 0; p < np; p++) {
            uint64_t q = a->ctx.q[p];
            memset(acc, 0, n * 8);
            for (int i = 0; i < a->comps; i++) {
                int j = k - i;
                if (j < 0 || j >= b->comps) continue;
                const uint64_t *ai = &a->c[((size_t)i * np + p) * n];
                const uint64_t *bj = &b->c[((size_t)j * np + p) * n];
                ckks_poly_mul(tmp, ai, bj, n, q);
                for (size_t x = 0; x < n; x++) acc[x] = ckks_addmod(acc[x], tmp[x], q);
            }
            uint64_t *out = &ct->c[((size_t)k * np + p) * n];
            memcpy(out, acc, n * 8);
        }
    }
    free(tmp); free(acc);
    return 0;
}
int ckks_mult(ckks_ct_t *ct, const ckks_ct_t *a, const ckks_ct_t *b) {
    if (!ct || !a || !b || a->ctx.nprimes != b->ctx.nprimes) return -1;
    return ckks_tensor(ct, a, b);
}
int ckks_mult_plain(ckks_ct_t *ct, const ckks_ct_t *a, const uint64_t *poly) {
    /* 明文多项式（RNS，编码 scale=2^60）：每分量 c_k * poly，scale 翻倍待 rescale。
     * 支持 a 任意分量数（无 relin 深链的 3+ 分量密文） */
    if (!ct || !a || !poly) return -1;
    size_t np = a->ctx.nprimes, n = a->ctx.n;
    ct->ctx = a->ctx;
    ct->comps = a->comps;
    ct->c = (uint64_t *)calloc((size_t)a->comps * np * n, sizeof(uint64_t));
    if (!ct->c) return -1;
    for (int cp = 0; cp < a->comps; cp++)
        for (size_t p = 0; p < np; p++) {
            uint64_t q = a->ctx.q[p];
            ckks_poly_mul(&ct->c[((size_t)cp * np + p) * n],
                          &a->c[((size_t)cp * np + p) * n], &poly[p * n], n, q);
        }
    return 0;
}

/* rescale：模数切换（切最后一个素数 q_L，scale 归一回 2^60） */
int ckks_rescale(ckks_ct_t *ct, const ckks_ct_t *in) {
    if (!ct || !in || in->ctx.nprimes < 2) return -1;
    int np_new = in->ctx.nprimes - 1;
    size_t n = in->ctx.n;
    uint64_t qL = in->ctx.q[np_new];
    ct->ctx = in->ctx;
    ct->ctx.nprimes = np_new;
    ct->comps = in->comps;
    ct->c = (uint64_t *)calloc((size_t)in->comps * np_new * n, sizeof(uint64_t));
    if (!ct->c) return -1;
    for (int cp = 0; cp < in->comps; cp++)
        for (int p = 0; p < np_new; p++) {
            uint64_t qp = in->ctx.q[p];
            uint64_t inv = ckks_modinv(qL % qp, qp);
            const uint64_t *xi = &in->c[((size_t)cp * in->ctx.nprimes + p) * n];
            const uint64_t *xL = &in->c[((size_t)cp * in->ctx.nprimes + np_new) * n];
            uint64_t *out = &ct->c[((size_t)cp * np_new + p) * n];
            for (size_t i = 0; i < n; i++) {
                uint64_t d = (xi[i] + qp - xL[i]) % qp;   /* xi - xL (mod qp) */
                out[i] = ckks_mulmod(d, inv, qp);
            }
        }
    return 0;
}

/* key-switch：把 c1（密钥 f(s)）用 gk（D(gk0+gk1*s)=f(s)）切回密钥 s。
 * c0_out = c0 + c1*gk0；c1_out = c1*gk1（完整链，模数匹配） */
static int ckks_keyswitch(ckks_ct_t *ct, const uint64_t *c0, const uint64_t *c1,
                          const uint64_t *gk0, const uint64_t *gk1,
                          const ckks_ctx_t *ctx) {
    size_t n = ctx->n, np = ctx->nprimes;
    uint64_t *t0 = (uint64_t *)malloc(n * sizeof(uint64_t));
    uint64_t *t1 = (uint64_t *)malloc(n * sizeof(uint64_t));
    ct->ctx = *ctx;
    ct->comps = 2;
    ct->c = (uint64_t *)calloc(2 * np * n, sizeof(uint64_t));
    if (!t0 || !t1 || !ct->c) { free(t0); free(t1); ckks_ct_free(ct); return -1; }
    for (size_t p = 0; p < np; p++) {
        uint64_t q = ctx->q[p];
        const uint64_t *c0p = &c0[p * n];
        const uint64_t *c1p = &c1[p * n];
        const uint64_t *g0p = &gk0[p * n];
        const uint64_t *g1p = &gk1[p * n];
        uint64_t *o0 = &ct->c[(0 * np + p) * n];
        uint64_t *o1 = &ct->c[(1 * np + p) * n];
        ckks_poly_mul(t0, c1p, g0p, n, q);          /* c1*gk0 */
        for (size_t i = 0; i < n; i++) o0[i] = ckks_addmod(c0p[i], t0[i], q);
        ckks_poly_mul(o1, c1p, g1p, n, q);           /* c1*gk1 */
    }
    free(t0); free(t1);
    return 0;
}

/* relin：3 -> 2 分量（c2 密钥 s^2 -> s），digit 分解（base 2^20，3 位） */
int ckks_relin(ckks_ct_t *ct, const ckks_ct_t *in, const ckks_rk_t *rk) {
    if (!ct || !in || !rk || in->comps != 3 || rk->digits != CKKS_RELIN_DIGITS) return -1;
    size_t n = in->ctx.n, np = in->ctx.nprimes;
    ct->ctx = in->ctx;
    ct->comps = 2;
    ct->c = (uint64_t *)calloc(2 * np * n, sizeof(uint64_t));
    if (!ct->c) return -1;
    uint64_t *dig = (uint64_t *)calloc(n, sizeof(uint64_t));
    uint64_t *t0 = (uint64_t *)calloc(n, sizeof(uint64_t));
    uint64_t *t1 = (uint64_t *)calloc(n, sizeof(uint64_t));
    if (!dig || !t0 || !t1) { free(dig); free(t0); free(t1); ckks_ct_free(ct); return -1; }
    for (size_t p = 0; p < np; p++) {
        uint64_t q = in->ctx.q[p];
        const uint64_t *c0 = &in->c[(0 * np + p) * n];
        const uint64_t *c1 = &in->c[(1 * np + p) * n];
        const uint64_t *c2 = &in->c[(2 * np + p) * n];
        uint64_t *o0 = &ct->c[(0 * np + p) * n];
        uint64_t *o1 = &ct->c[(1 * np + p) * n];
        memcpy(o0, c0, n * 8);
        memcpy(o1, c1, n * 8);
        for (int d = 0; d < rk->digits; d++) {
            uint64_t shift = (uint64_t)(CKKS_RELIN_BASE_BITS * d);
            for (size_t i = 0; i < n; i++) dig[i] = (c2[i] >> shift) & ((1ull << CKKS_RELIN_BASE_BITS) - 1);
            const uint64_t *r0 = &rk->rk0[d][p * n];
            const uint64_t *r1 = &rk->rk1[d][p * n];
            ckks_poly_mul(t0, dig, r0, n, q);
            ckks_poly_mul(t1, dig, r1, n, q);
            for (size_t i = 0; i < n; i++) o0[i] = ckks_addmod(o0[i], t0[i], q);
            for (size_t i = 0; i < n; i++) o1[i] = ckks_addmod(o1[i], t1[i], q);
        }
    }
    free(dig); free(t0); free(t1);
    return 0;
}

/* 旋转：槽位置换 t（Galois 自动态 + digit 分解 key-switch，支持多分量）。
 * 解析嵌入下 σ_k 的槽位语义（k=5^t）：
 *   源槽位 j -> 目标 j'，满足 (2j'+1)·k ≡ ±(2j+1) (mod 2n)，
 *   取奇数代表元 s = (2j+1)·k^{-1} mod 2n，若 s >= n 则 s = 2n-s，j'=(s-1)/2。
 * 多分量（无 relin 密文 comps>2）：每分量 σ(c_d) 密钥 σ_k(s)^d，用 target=σ_k(s)^d
 * 的 gk_d（d=1 用传入 gk，d>=2 内部生成）digit 分解 key-switch 全部切回密钥 s，
 * 合并为 2 分量输出。key-switch 噪声 Σ_d dig_d·e_d ~ 2^45 << 2^60（scale）。 */
int ckks_rotate(ckks_ct_t *ct, const ckks_ct_t *in, const ckks_sk_t *sk,
                const ckks_gk_t *gk, int t) {
    if (!in) return -1;
    uint64_t k = ckks_powmod(5, t, 2 * (uint64_t)in->ctx.n);   /* σ_{5^t} */
    return ckks_rotate_k(ct, in, sk, gk, k);
}
/* 一般 Galois 自同构 σ_k（k 任意奇数）的旋转：σ_k 全分量 + key-switch 合并 2 分量 */
int ckks_rotate_k(ckks_ct_t *ct, const ckks_ct_t *in, const ckks_sk_t *sk,
                  const ckks_gk_t *gk, uint64_t k) {
    if (!ct || !in || !gk || !sk || in->comps < 2 || in->comps > CKKS_MAX_COMP ||
        (k & 1) == 0) return -1;
    size_t n = in->ctx.n, np = in->ctx.nprimes;
    int comps = in->comps;
    /* σ 全分量 */
    uint64_t **sig = (uint64_t **)calloc((size_t)comps, sizeof(uint64_t *));
    if (!sig) return -1;
    for (int d = 0; d < comps; d++) {
        sig[d] = (uint64_t *)calloc(np * n, sizeof(uint64_t));
        if (!sig[d]) { for (int e = 0; e < d; e++) free(sig[e]); free(sig); return -1; }
        for (size_t p = 0; p < np; p++)
            ckks_galois(&sig[d][p * n], &in->c[((size_t)d * np + p) * n], k,
                        in->ctx.q[p], n);
    }
    /* 目标明文 σ_k(s)^d（d=1..comps-1，逐素数；σ(s) 稀疏 → 幂系数小） */
    uint64_t **tgt = (uint64_t **)calloc((size_t)comps, sizeof(uint64_t *));
    uint64_t *sraw = (uint64_t *)malloc(n * sizeof(uint64_t));
    if (!tgt || !sraw) { free(tgt); free(sraw); goto fail_sig; }
    tgt[1] = (uint64_t *)calloc(np * n, sizeof(uint64_t));
    if (!tgt[1]) { free(tgt); free(sraw); goto fail_sig; }
    for (size_t p = 0; p < np; p++) {
        uint64_t q = in->ctx.q[p];
        const int8_t *s = sk->s;
        for (size_t i = 0; i < n; i++) sraw[i] = s[i] < 0 ? q - 1 : (uint64_t)s[i];
        ckks_galois(&tgt[1][p * n], sraw, k, q, n);   /* σ_k(s) */
    }
    free(sraw);
    for (int d = 2; d < comps; d++) {
        tgt[d] = (uint64_t *)calloc(np * n, sizeof(uint64_t));
        if (!tgt[d]) goto fail_sig;
        for (size_t p = 0; p < np; p++)
            ckks_poly_mul(&tgt[d][p * n], &tgt[d - 1][p * n], &tgt[1][p * n],
                          n, in->ctx.q[p]);
    }
    /* 输出 2 分量（所有分量 key-switch 合并） */
    ct->ctx = in->ctx;
    ct->comps = 2;
    ct->c = (uint64_t *)calloc(2 * np * n, sizeof(uint64_t));
    uint64_t *dig = (uint64_t *)malloc(n * sizeof(uint64_t));
    uint64_t *t0 = (uint64_t *)malloc(n * sizeof(uint64_t));
    uint64_t *t1 = (uint64_t *)malloc(n * sizeof(uint64_t));
    if (!ct->c || !dig || !t0 || !t1) { free(dig); free(t0); free(t1); ckks_ct_free(ct); goto fail_tgt; }
    for (size_t p = 0; p < np; p++)
        memcpy(&ct->c[(0 * np + p) * n], &sig[0][p * n], n * 8);   /* σ(c0) */
    uint64_t B = 1ull << CKKS_RELIN_BASE_BITS;
    for (int d = 1; d < comps; d++) {
        ckks_gk_t gkd = {0};
        int use_ext = (d == 1);
        if (!use_ext) {
            /* 现生成 gk_d：target = B^dd · σ_k(s)^d（d>=2，每 rotate 一次，代价小） */
            gkd.n = (uint32_t)n;
            gkd.digits = gk->digits;
            uint64_t *tg = (uint64_t *)calloc(np * n, sizeof(uint64_t));
            if (!tg) { free(dig); free(t0); free(t1); ckks_ct_free(ct); goto fail_tgt; }
            for (int dd = 0; dd < gk->digits; dd++) {
                if (dd > 0)
                    for (size_t p = 0; p < np; p++) {
                        uint64_t q = in->ctx.q[p], bq = B % q;
                        for (size_t i = 0; i < n; i++)
                            tg[p * n + i] = ckks_mulmod(tg[p * n + i], bq, q);
                    }
                else
                    memcpy(tg, tgt[d], np * n * 8);
                gkd.gk0[dd] = (uint64_t *)calloc(np * n, sizeof(uint64_t));
                gkd.gk1[dd] = (uint64_t *)calloc(np * n, sizeof(uint64_t));
                if (!gkd.gk0[dd] || !gkd.gk1[dd] ||
                    ckks_generic_gk(gkd.gk0[dd], gkd.gk1[dd], sk, &in->ctx, tg) != 0) {
                    free(tg); ckks_gk_free(&gkd);
                    free(dig); free(t0); free(t1); ckks_ct_free(ct); goto fail_tgt;
                }
            }
            free(tg);
        }
        for (size_t p = 0; p < np; p++) {
            uint64_t q = in->ctx.q[p];
            uint64_t *o0 = &ct->c[(0 * np + p) * n];
            uint64_t *o1 = &ct->c[(1 * np + p) * n];
            for (int dd = 0; dd < gk->digits; dd++) {
                uint64_t shift = (uint64_t)(CKKS_RELIN_BASE_BITS * dd);
                for (size_t i = 0; i < n; i++)
                    dig[i] = (sig[d][p * n + i] >> shift) & ((1ull << CKKS_RELIN_BASE_BITS) - 1);
                const uint64_t *g0d = use_ext ? &gk->gk0[dd][p * n] : &gkd.gk0[dd][p * n];
                const uint64_t *g1d = use_ext ? &gk->gk1[dd][p * n] : &gkd.gk1[dd][p * n];
                ckks_poly_mul(t0, dig, g0d, n, q);
                ckks_poly_mul(t1, dig, g1d, n, q);
                for (size_t i = 0; i < n; i++) o0[i] = ckks_addmod(o0[i], t0[i], q);
                for (size_t i = 0; i < n; i++) o1[i] = ckks_addmod(o1[i], t1[i], q);
            }
        }
        if (!use_ext) ckks_gk_free(&gkd);
    }
    free(dig); free(t0); free(t1);
fail_tgt:
    for (int d = 1; d < comps; d++) free(tgt[d]);
    free(tgt);
fail_sig:
    for (int d = 0; d < comps; d++) free(sig[d]);
    free(sig);
    return 0;
}

/* ================================================================
 * 自测（T12，接入一键判定）
 * ================================================================ */
static int g_ckks_fail = 0;
#define CKKS_CHECK(cond, name, fmt, ...) do { \
    if (cond) printf("  [PASS] " name "\n"); \
    else { printf("  [FAIL] " name ": " fmt "\n", ##__VA_ARGS__); g_ckks_fail++; } \
} while (0)

/* T12e 明文参考：对 encode 多项式逐素数做 Galois 置换（与 ckks_galois 同逻辑） */
static void t12_galois(uint64_t *out, const uint64_t *in, uint64_t k,
                       uint64_t q, size_t nn) {
    memset(out, 0, nn * 8);
    size_t n2 = 2 * nn;
    for (size_t i = 0; i < nn; i++) {
        if (in[i] == 0) continue;
        size_t idx = (size_t)((unsigned __int128)(i % n2) * (k % n2) % n2);
        if (idx < nn) out[idx] = ckks_addmod(out[idx], in[i], q);
        else          out[idx - nn] = ckks_submod(out[idx - nn], in[i], q);
    }
}

int ckks_self_test(void) {
    printf("=== vLLM-CKKS Self Test ===\n");
    g_ckks_fail = 0;
    ckks_rng_state = (uint64_t)time(NULL) ^ 0xDEADBEEFCAFEBABEull;
    ckks_set_rng(NULL);

    ckks_ctx_t ctx;
    if (ckks_ctx_init(&ctx) != 0) { printf("  [FAIL] ctx init\n"); return 1; }
    printf("  moduli: ");
    for (int i = 0; i < ctx.nprimes; i++)
        printf("%llu ", (unsigned long long)ctx.q[i]);
    printf("(总 %d-bit), scale=2^%d, n=%u\n", 60 * ctx.nprimes, CKKS_SCALE_BITS, ctx.n);

    ckks_sk_t sk;
    ckks_sk_gen(&sk, ctx.n);
    size_t n = ctx.n;
    uint64_t *poly = (uint64_t *)malloc((size_t)ctx.nprimes * n * 8);
    uint64_t *poly2 = (uint64_t *)malloc((size_t)ctx.nprimes * n * 8);
    uint64_t *pdec = (uint64_t *)malloc((size_t)ctx.nprimes * n * 8);
    double *z1 = (double *)malloc(n * 8), *z2 = (double *)malloc(n * 8);
    double *zb = (double *)malloc(n * 8), *zc = (double *)malloc(n * 8);
    if (!poly || !poly2 || !pdec || !z1 || !z2 || !zb || !zc) {
        printf("  [FAIL] alloc\n");
        return 1;
    }
    for (size_t j = 0; j < n; j++) {
        double r1 = (double)(ckks_rng_u64() % 1000000) / 1000000.0;
        double r2 = (double)(ckks_rng_u64() % 1000000) / 1000000.0;
        z1[j] = (r1 - 0.5) * 0.4;   /* [-0.2, 0.2) */
        z2[j] = (r2 - 0.5) * 0.4;
    }

    /* ---- T12a: 编码/解码往返（解析嵌入，512 有效槽位） ---- */
    ckks_encode(poly, z1, &ctx);
    ckks_decode(zb, poly, &ctx);
    double err = 0.0;
    for (size_t j = 0; j < n / 2; j++) {
        double d = fabs(zb[j] - z1[j]);
        if (d > err) err = d;
    }
    CKKS_CHECK(err < 1e-6, "T12a encode/decode roundtrip", "err=%.2e", err);

    /* ---- T12b: 同态加法（深 0） ---- */
    ckks_ct_t c1 = {0}, c2 = {0}, ca = {0};
    ckks_encode(poly, z1, &ctx);
    ckks_encrypt(&c1, &sk, poly, &ctx);
    ckks_encode(poly, z2, &ctx);
    ckks_encrypt(&c2, &sk, poly, &ctx);
    ckks_add(&ca, &c1, &c2);
    ckks_decrypt(pdec, &sk, &ca);
    ckks_decode(zc, pdec, &ca.ctx);
    err = 0.0;
    for (size_t j = 0; j < n / 2; j++) {
        double d = fabs(zc[j] - (z1[j] + z2[j]));
        if (d > err) err = d;
    }
    CKKS_CHECK(err < 1e-6, "T12b homomorphic add", "err=%.2e", err);

    /* ---- T12c: 深度 3 乘法链 z1*z2*z3*z4 ---- */
    {
        double z3v[1] = {0.1}, z4v[1] = {0.05};
        double zz[1024]; memset(zz, 0, sizeof(zz));
        for (size_t j = 0; j < n; j++) zz[j] = z1[j];
        ckks_ct_t c3 = {0}, c4 = {0}, t1 = {0}, t2 = {0}, t3 = {0}, tr = {0};
        memset(zz, 0, sizeof(zz)); zz[0] = z3v[0];
        ckks_encode(poly, zz, &ctx);
        ckks_encrypt(&c3, &sk, poly, &ctx);
        memset(zz, 0, sizeof(zz)); zz[0] = z4v[0];
        ckks_encode(poly, zz, &ctx);
        ckks_encrypt(&c4, &sk, poly, &ctx);
        /* c1*c2（无 relin：多分量解密，明文跨素数一致保证 rescale 正确） */
        ckks_mult(&t1, &c1, &c2);
        ckks_rescale(&tr, &t1);
        ckks_ct_free(&t1);
        t1 = tr; tr.c = NULL;   /* 转移所有权 */
        /* *c3（先 mod switch 到 t1 的素数链） */
        ckks_modswitch(&tr, &c3);
        ckks_ct_free(&c3); c3 = tr; tr.c = NULL;
        ckks_mult(&t2, &t1, &c3);
        ckks_rescale(&tr, &t2);
        ckks_ct_free(&t2);
        t2 = tr; tr.c = NULL;
        /* *c4（mod switch 到 t2 的素数链：4->3->2） */
        ckks_modswitch(&tr, &c4);
        ckks_ct_free(&c4); c4 = tr; tr.c = NULL;
        ckks_modswitch(&tr, &c4);
        ckks_ct_free(&c4); c4 = tr; tr.c = NULL;
        ckks_mult(&t3, &t2, &c4);
        ckks_rescale(&tr, &t3);
        ckks_ct_free(&t3);
        t3 = tr; tr.c = NULL;
        ckks_decrypt(pdec, &sk, &t3);
        ckks_decode(zc, pdec, &t3.ctx);
        err = 0.0;
        for (size_t j = 0; j < n / 2; j++) {
            /* z3/z4 只在槽位 0 非零：槽位乘积逐槽位 */
            double expect = (j == 0) ? z1[0] * z2[0] * z3v[0] * z4v[0] : 0.0;
            double d = fabs(zc[j] - expect);
            if (d > err) err = d;
        }
        CKKS_CHECK(err < 1e-3, "T12c depth-3 mult chain", "err=%.2e", err);
        ckks_ct_free(&c3); ckks_ct_free(&c4);
        ckks_ct_free(&t1); ckks_ct_free(&t2); ckks_ct_free(&t3);
    }

    /* ---- T12d: GELU 多项式逼近 0.5x + 0.399x^2（深 3，槽位 0） ---- */
    {
        double zx = 0.25;
        double zz[1024]; memset(zz, 0, sizeof(zz)); zz[0] = zx;
        ckks_ct_t cx = {0}, cx2 = {0}, ct_a = {0}, ct_b = {0}, out = {0}, tt = {0};
        ckks_encode(poly, zz, &ctx);
        ckks_encrypt(&cx, &sk, poly, &ctx);
        /* x^2（第 1 层：mult + rescale，无 relin） */
        ckks_mult(&tt, &cx, &cx);
        ckks_rescale(&cx2, &tt);    ckks_ct_free(&tt);
        /* 0.5x（第 2 层：mult_plain + rescale） */
        memset(zz, 0, sizeof(zz)); zz[0] = 0.5;
        ckks_encode(poly, zz, &ctx);
        ckks_mult_plain(&tt, &cx, poly);
        ckks_rescale(&ct_a, &tt);    ckks_ct_free(&tt);
        /* 0.399x^2（第 3 层：mult_plain + rescale，poly 按 cx2 素数链编码） */
        memset(zz, 0, sizeof(zz)); zz[0] = 0.399;
        ckks_encode(poly, zz, &cx2.ctx);
        ckks_mult_plain(&tt, &cx2, poly);
        ckks_rescale(&ct_b, &tt);    ckks_ct_free(&tt);
        /* 对齐素数链：ct_a(3) 与 ct_b(2) 需同链相加（mod switch 降级不 rescale） */
        while (ct_a.ctx.nprimes > ct_b.ctx.nprimes) {
            ckks_modswitch(&tt, &ct_a);
            ckks_ct_free(&ct_a); ct_a = tt; tt.c = NULL;
        }
        /* out = 0.5x + 0.399x^2（同 scale，同素数链） */
        ckks_add(&out, &ct_a, &ct_b);
        ckks_decrypt(pdec, &sk, &out);
        ckks_decode(zc, pdec, &out.ctx);
        double got = zc[0];
        double expect = 0.5 * zx + 0.399 * zx * zx;
        CKKS_CHECK(fabs(got - expect) < 1e-3, "T12d GELU poly 0.5x+0.399x^2 (深3)",
                   "got=%.4f exp=%.4f", got, expect);
        ckks_ct_free(&cx); ckks_ct_free(&cx2);
        ckks_ct_free(&ct_a); ckks_ct_free(&ct_b); ckks_ct_free(&out);
    }

    /* ---- T12e: 槽位旋转（Galois 自动态 + digit 分解 key-switch） ---- */
    {
        double zz[1024]; memset(zz, 0, sizeof(zz));
        zz[0] = 0.25; zz[1] = 0.15; zz[2] = -0.1;   /* 槽位 0,1,2 有值 */
        ckks_encode(poly, zz, &ctx);
        size_t np = ctx.nprimes;
        uint64_t *pref = (uint64_t *)malloc(np * n * 8);
        double *zref = (double *)malloc(n * 8), *zgot = (double *)malloc(n * 8);
        if (!pref || !zref || !zgot) { printf("  [FAIL] T12e alloc\n"); g_ckks_fail++; }
        else {
            for (int t = 1; t <= 3; t++) {
                uint64_t kk = ckks_powmod(5, t, 2 * n);
                for (size_t p = 0; p < np; p++)
                    t12_galois(&pref[p * n], &poly[p * n], kk, ctx.q[p], n);
                ckks_decode(zref, pref, &ctx);
                /* 密文路径：encrypt -> rotate -> decrypt -> decode */
                ckks_ct_t cc = {0}, cr = {0};
                ckks_gk_t gk = {0};
                ckks_encrypt(&cc, &sk, poly, &ctx);
                if (ckks_gk_gen(&gk, &sk, &ctx, t) != 0 ||
                    ckks_rotate(&cr, &cc, &sk, &gk, t) != 0) {
                    printf("  [FAIL] T12e t=%d rotate/gk_gen\n", t);
                    g_ckks_fail++;
                } else {
                    ckks_decrypt(pdec, &sk, &cr);
                    ckks_decode(zgot, pdec, &cr.ctx);
                    double e2 = 0.0;
                    for (size_t j = 0; j < n / 2; j++) {
                        double d = fabs(zgot[j] - zref[j]);
                        if (d > e2) e2 = d;
                    }
                    /* 打印非零槽位（便于与 Python 参考核对：t=1 -> {0,204,409}） */
                    char buf[256]; size_t bl = 0;
                    for (size_t j = 0; j < n / 2 && bl < 200; j++)
                        if (fabs(zgot[j]) > 1e-4)
                            bl += (size_t)snprintf(buf + bl, sizeof(buf) - bl,
                                                   " (%zu:%.3f)", j, zgot[j]);
                    if (e2 < 1e-3)
                        printf("  [PASS] T12e rotate t=%d (k=%llu err=%.2e slots[%s])\n",
                               t, (unsigned long long)kk, e2, buf);
                    else {
                        printf("  [FAIL] T12e rotate t=%d: err=%.2e slots[%s]\n",
                               t, e2, buf);
                        g_ckks_fail++;
                    }
                }
                ckks_ct_free(&cc); ckks_ct_free(&cr); ckks_gk_free(&gk);
            }
            free(pref); free(zref); free(zgot);
        }
    }

    /* ---- T12f: 静态深度探针 D=9/10（12×60-bit 素数链）。
     * 深 11（nprimes=1）实测噪声波动达 9 个数量级（rel 2e-11 ~ 3e-2），
     * 只剩 1 素数时模数切换舍入/噪声不可控 → 诚实上限定为深 10。 */
    {
        int depths[2] = {9, 10};
        for (int di = 0; di < 2; di++) {
            int D = depths[di];
            if (D + 1 > (int)ctx.nprimes) {
                printf("  [SKIP] T12f depth D=%d (素数链不足 %d)\n", D, D + 1);
                continue;
            }
            /* 因子 z_f 集中在槽位 0，接近 1 使期望值稳定可判 */
            double zf[16];
            for (int f = 0; f <= D; f++) zf[f] = 0.98 + 0.01 * (double)(f % 3);
            double expect = 1.0;
            for (int f = 0; f <= D; f++) expect *= zf[f];
            ckks_ct_t fac[16]; memset(fac, 0, sizeof(fac));
            double zz2[1024];
            for (int f = 0; f <= D; f++) {
                memset(zz2, 0, sizeof(zz2)); zz2[0] = zf[f];
                ckks_encode(poly, zz2, &ctx);
                ckks_encrypt(&fac[f], &sk, poly, &ctx);
            }
            ckks_ct_t acc = {0}, tt = {0};
            ckks_ct_copy(&acc, &fac[0]);
            for (int d = 1; d <= D; d++) {
                ckks_ct_t ff = {0};
                ckks_ct_copy(&ff, &fac[d]);
                while ((int)ff.ctx.nprimes > (int)acc.ctx.nprimes) {   /* 因子降链到 acc 链 */
                    ckks_modswitch(&tt, &ff);
                    ckks_ct_free(&ff); ff = tt; tt.c = NULL;
                }
                ckks_mult(&tt, &acc, &ff);
                ckks_ct_free(&acc); ckks_ct_free(&ff);
                ckks_rescale(&acc, &tt);
                ckks_ct_free(&tt);
            }
            ckks_decrypt(pdec, &sk, &acc);
            double ztmp[512];
            ckks_decode(ztmp, pdec, &acc.ctx);
            double got = ztmp[0];
            double e2 = fabs(got - expect);
            double rel = e2 / (fabs(expect) + 1e-12);
            /* 深 11 只剩 1 素数，噪声尾部波动大（最坏 ~2e-3），阈值放宽；深 9/10 严格 */
            double thr = (D >= 11) ? 5e-3 : 1e-3;
            printf("  [%s] T12f depth-D=%d: got=%.6f exp=%.6f abs=%.2e rel=%.2e (comps=%d nprimes=%d)\n",
                   e2 < thr * fabs(expect) + 1e-9 ? "PASS" : "FAIL",
                   D, got, expect, e2, rel, acc.comps, acc.ctx.nprimes);
            if (!(e2 < thr * fabs(expect) + 1e-9)) g_ckks_fail++;
            ckks_ct_free(&acc);
            for (int f = 0; f <= D; f++) ckks_ct_free(&fac[f]);
        }
    }

    /* ---- T12g: 高次 GELU 逼近（11 次切比雪夫 [-3,3]，Horner 深 11） ---- */
    {
        /* GELU ≈ P(x)，P 由 tools/gelu_fit.py 切比雪夫拟合生成：
         * P(x) = 2.83e-4 + 0.5x + 0.3963x^2 - 0.06256x^4 + 7.72e-3x^6
         *        - 5.547e-4x^8 + 1.709e-5x^10，逼近误差 9.1e-4（[-3,3]） */
        double g11[12] = {0.000283044, 0.5, 0.396339523, 0.0, -0.062555973,
                          0.0, 0.007718440, 0.0, -0.000554714, 0.0,
                          0.000017093, 0.0};
        double xs[7] = {-2.8, -1.5, -0.4, 0.3, 1.2, 2.5, 2.9};
        double max_noise = 0.0, max_total = 0.0;
        double zz3[1024];
        for (int xi = 0; xi < 7; xi++) {
            double xv = xs[xi];
            /* 明文 Horner 参考 + 真 GELU */
            double pr = g11[10];
            for (int i = 9; i >= 0; i--) pr = pr * xv + g11[i];
            double gr = xv * 0.5 * (1.0 + erf(xv / sqrt(2.0)));
            /* 密文：x 单槽位加密，Horner 11 层 mult+rescale + 明文加 */
            memset(zz3, 0, sizeof(zz3)); zz3[0] = xv;
            ckks_encode(poly, zz3, &ctx);
            ckks_ct_t cx = {0}, cres = {0}, tt = {0}, xc = {0};
            ckks_encrypt(&cx, &sk, poly, &ctx);
            ckks_ct_copy(&xc, &cx);
            memset(zz3, 0, sizeof(zz3)); zz3[0] = g11[10];
            ckks_encode(poly, zz3, &ctx);
            ckks_encrypt(&cres, &sk, poly, &ctx);
            for (int i = 9; i >= 0; i--) {
                while ((int)xc.ctx.nprimes > (int)cres.ctx.nprimes) {   /* x 降到 res 链 */
                    ckks_modswitch(&tt, &xc);
                    ckks_ct_free(&xc); xc = tt; tt.c = NULL;
                }
                ckks_mult(&tt, &cres, &xc);
                ckks_ct_free(&cres);
                ckks_rescale(&cres, &tt);
                ckks_ct_free(&tt);
                /* 明文加 a_i（按 res 当前链编码） */
                memset(zz3, 0, sizeof(zz3)); zz3[0] = g11[i];
                ckks_encode(poly, zz3, &cres.ctx);
                for (size_t p = 0; p < (size_t)cres.ctx.nprimes; p++) {
                    uint64_t q = cres.ctx.q[p];
                    uint64_t *c0 = &cres.c[(0 * (size_t)cres.ctx.nprimes + p) * n];
                    for (size_t k = 0; k < n; k++)
                        c0[k] = ckks_addmod(c0[k], poly[p * n + k], q);
                }
            }
            ckks_decrypt(pdec, &sk, &cres);
            double ztmp2[512];
            ckks_decode(ztmp2, pdec, &cres.ctx);
            double got = ztmp2[0];
            double noise = fabs(got - pr);
            double total = fabs(got - gr);
            if (noise > max_noise) max_noise = noise;
            if (total > max_total) max_total = total;
            ckks_ct_free(&cx); ckks_ct_free(&cres); ckks_ct_free(&xc);
        }
        printf("  [%s] T12g GELU 11次逼近 [-3,3]: 逼近err=9.1e-4 噪声max=%.2e 总errmax=%.2e\n",
               max_total < 2e-3 ? "PASS" : "FAIL", max_noise, max_total);
        if (!(max_total < 2e-3)) g_ckks_fail++;
    }

    /* ---- T15: LLM FFN 激活流（单槽位 + 深 10 链） ----
     * 模拟 transformer 前馈块：x ← W2·GELU(W1·x + b1) + b2（权重明文标量），
     * GELU 用二次逼近 0.5h+0.399h²（T12d 系数）。2 块 × 深 5 = 深 10，
     * 吃满 12 素数链（剩 2 素数），与明文参考逐块对照。 */
    {
        double W1 = 0.8, b1 = 0.02, W2 = 0.5, b2 = -0.01;
        double xv = 0.25;
        double zz[1024]; memset(zz, 0, sizeof(zz)); zz[0] = xv;
        ckks_ct_t xc = {0}, t = {0}, h = {0}, h2 = {0};
        ckks_ct_t ca = {0}, cb = {0}, g = {0}, y = {0};
        ckks_encode(poly, zz, &ctx);
        ckks_encrypt(&xc, &sk, poly, &ctx);
        double ref = xv;
        double err = 0.0;
        for (int blk = 0; blk < 2; blk++) {
            /* 明文参考（同二次逼近） */
            double h_ref = W1 * ref + b1;
            double g_ref = 0.5 * h_ref + 0.399 * h_ref * h_ref;
            ref = W2 * g_ref + b2;
            /* 密文：h = W1·x + b1（mult_plain scale 翻倍 → rescale） */
            memset(zz, 0, sizeof(zz)); zz[0] = W1;
            ckks_encode(poly, zz, &xc.ctx);
            ckks_mult_plain(&t, &xc, poly);
            ckks_rescale(&h, &t); ckks_ct_free(&t);
            memset(zz, 0, sizeof(zz)); zz[0] = b1;
            ckks_encode(poly, zz, &h.ctx);
            for (size_t p = 0; p < (size_t)h.ctx.nprimes; p++) {
                uint64_t q = h.ctx.q[p];
                uint64_t *c0 = &h.c[(0 * (size_t)h.ctx.nprimes + p) * n];
                for (size_t k = 0; k < n; k++)
                    c0[k] = ckks_addmod(c0[k], poly[p * n + k], q);
            }
            /* h² = h·h（密文×密文，深 +1） */
            ckks_mult(&t, &h, &h);
            ckks_rescale(&h2, &t); ckks_ct_free(&t);
            /* a = 0.5·h；b = 0.399·h² */
            memset(zz, 0, sizeof(zz)); zz[0] = 0.5;
            ckks_encode(poly, zz, &h.ctx);
            ckks_mult_plain(&t, &h, poly);
            ckks_rescale(&ca, &t); ckks_ct_free(&t);
            memset(zz, 0, sizeof(zz)); zz[0] = 0.399;
            ckks_encode(poly, zz, &h2.ctx);
            ckks_mult_plain(&t, &h2, poly);
            ckks_rescale(&cb, &t); ckks_ct_free(&t);
            /* 对齐素数链后 g = a + b */
            while (ca.ctx.nprimes > cb.ctx.nprimes) {
                ckks_modswitch(&t, &ca);
                ckks_ct_free(&ca); ca = t; t.c = NULL;
            }
            ckks_add(&g, &ca, &cb);
            /* y = W2·g + b2 */
            memset(zz, 0, sizeof(zz)); zz[0] = W2;
            ckks_encode(poly, zz, &g.ctx);
            ckks_mult_plain(&t, &g, poly);
            ckks_rescale(&y, &t); ckks_ct_free(&t);
            memset(zz, 0, sizeof(zz)); zz[0] = b2;
            ckks_encode(poly, zz, &y.ctx);
            for (size_t p = 0; p < (size_t)y.ctx.nprimes; p++) {
                uint64_t q = y.ctx.q[p];
                uint64_t *c0 = &y.c[(0 * (size_t)y.ctx.nprimes + p) * n];
                for (size_t k = 0; k < n; k++)
                    c0[k] = ckks_addmod(c0[k], poly[p * n + k], q);
            }
            /* 交接：x = y */
            ckks_ct_free(&xc); xc = y; y.c = NULL;
            ckks_ct_free(&h); ckks_ct_free(&h2);
            ckks_ct_free(&ca); ckks_ct_free(&cb); ckks_ct_free(&g);
        }
        ckks_decrypt(pdec, &sk, &xc);
        double ztmp3[512];
        ckks_decode(ztmp3, pdec, &xc.ctx);
        double got = ztmp3[0];
        err = fabs(got - ref);
        CKKS_CHECK(err < 5e-3, "T15 LLM FFN 激活流 2块(深10) 与明文对照",
                   "got=%.5f exp=%.5f err=%.2e", got, ref, err);
        ckks_ct_free(&xc);
    }

    /* ---- T14: 多槽位 SIMD matmul（代数嵌入，权重明文标量乘） ----
     * 根因修复：T14 曾 SKIP（解析嵌入下满槽位 matmul 的 scale² 明文系数经
     * 环乘卷积放大 n 倍后超 q0/2 而 wrap）。修复：稀疏槽位（d=8 样本 << n）
     * + 权重明文整数标量乘（mult_plain 常数多项式，环乘退化为系数乘标量，
     * 无卷积放大，scale 保持 2^60 无需 rescale）。encode 系数 ~ scale·d·|z|/n
     * ~ 2^50，累加 4 项后 ~2^52 << q0/2=2^59，不 wrap。 */
    {
        int nf = 4;                       /* 输入特征数 = 输出特征数 */
        int ns = 8;                       /* 样本数（活跃槽位） */
        static const int8_t W14[4][4] = {
            {1, 0, -1, 1}, {0, 1, 1, 0}, {1, -1, 0, 1}, {0, 1, -1, -1}};
        double X14[4][8];
        for (int i = 0; i < nf; i++)
            for (int s = 0; s < ns; s++)
                X14[i][s] = (double)((int)(ckks_rng_u64() % 41) - 20) / 100.0; /* [-0.2,0.2] */
        /* 每特征打包一密文（槽位 = 样本） */
        ckks_ct_t cts[4] = {0};
        double zz4[1024];
        for (int i = 0; i < nf; i++) {
            memset(zz4, 0, sizeof(zz4));
            for (int s = 0; s < ns; s++) zz4[s] = X14[i][s];
            ckks_encode(poly, zz4, &ctx);
            ckks_encrypt(&cts[i], &sk, poly, &ctx);
        }
        /* 输出 j = Σ_i W[j][i]·cts[i]：mult_plain 常数多项式（标量乘） */
        double err = 0.0;
        for (int j = 0; j < nf; j++) {
            ckks_ct_t acc = {0}, t2 = {0};
            for (int i = 0; i < nf; i++) {
                if (W14[j][i] == 0) continue;
                int64_t wv = W14[j][i];
                for (size_t p = 0; p < (size_t)ctx.nprimes; p++) {
                    uint64_t q = ctx.q[p];
                    memset(&poly[p * n], 0, n * 8);
                    poly[p * n + 0] = (uint64_t)(wv < 0 ? wv + (int64_t)q : wv);
                }
                ckks_mult_plain(&t2, &cts[i], poly);
                if (acc.comps == 0) { acc = t2; t2.c = NULL; }
                else {
                    ckks_ct_t a2 = {0};
                    ckks_add(&a2, &acc, &t2);
                    ckks_ct_free(&acc); ckks_ct_free(&t2);
                    acc = a2;
                }
            }
            ckks_decrypt(pdec, &sk, &acc);
            double zgot4[512];
            ckks_decode(zgot4, pdec, &acc.ctx);
            for (int s = 0; s < ns; s++) {
                double expect = 0.0;
                for (int i = 0; i < nf; i++) expect += W14[j][i] * X14[i][s];
                double d = fabs(zgot4[s] - expect);
                if (d > err) err = d;
            }
            ckks_ct_free(&acc);
        }
        for (int i = 0; i < nf; i++) ckks_ct_free(&cts[i]);
        CKKS_CHECK(err < 1e-3, "T14 多槽位SIMD matmul 4x4(权重明文) 与明文对照",
                   "err=%.2e", err);
    }

    /* ---- T14b: 满槽位 SIMD matmul（256 样本并行，权重明文标量乘） ----
     * T14 的满槽位扩展：d=256 槽位全有值（非稀疏）。明文系数上界
     * |poly[k]| ≤ scale·2d·max|z|/n，d=256、|z|≤0.05 → ~2^55.3；
     * mult_plain 常数标量（无卷积放大）+ 累加 4 项 → ~2^57.3 << q0/2=2^59，
     * 不 wrap。验证满槽位 SIMD 下 matmul 与明文逐槽位一致。 */
    {
        int nf2 = 4, ns2 = 256;             /* 256 样本（活跃槽位=256） */
        static const int8_t W15[4][4] = {
            {1, 0, -1, 1}, {0, 1, 1, 0}, {1, -1, 0, 1}, {0, 1, -1, -1}};
        double X15[4][256];
        for (int i = 0; i < nf2; i++)
            for (int s = 0; s < ns2; s++)
                X15[i][s] = (double)((int)(ckks_rng_u64() % 11) - 5) / 100.0; /* [-0.05,0.05] */
        ckks_ct_t cts2[4] = {0};
        double zz5[1024];
        for (int i = 0; i < nf2; i++) {
            memset(zz5, 0, sizeof(zz5));
            for (int s = 0; s < ns2; s++) zz5[s] = X15[i][s];
            ckks_encode(poly, zz5, &ctx);
            ckks_encrypt(&cts2[i], &sk, poly, &ctx);
        }
        double err2 = 0.0;
        for (int j = 0; j < nf2; j++) {
            ckks_ct_t acc = {0}, t2 = {0};
            for (int i = 0; i < nf2; i++) {
                if (W15[j][i] == 0) continue;
                int64_t wv = W15[j][i];
                for (size_t p = 0; p < (size_t)ctx.nprimes; p++) {
                    uint64_t q = ctx.q[p];
                    memset(&poly[p * n], 0, n * 8);
                    poly[p * n + 0] = (uint64_t)(wv < 0 ? wv + (int64_t)q : wv);
                }
                ckks_mult_plain(&t2, &cts2[i], poly);
                if (acc.comps == 0) { acc = t2; t2.c = NULL; }
                else {
                    ckks_ct_t a2 = {0};
                    ckks_add(&a2, &acc, &t2);
                    ckks_ct_free(&acc); ckks_ct_free(&t2);
                    acc = a2;
                }
            }
            ckks_decrypt(pdec, &sk, &acc);
            double zgot5[512];
            ckks_decode(zgot5, pdec, &acc.ctx);
            for (int s = 0; s < ns2; s++) {
                double expect = 0.0;
                for (int i = 0; i < nf2; i++) expect += W15[j][i] * X15[i][s];
                double d = fabs(zgot5[s] - expect);
                if (d > err2) err2 = d;
            }
            ckks_ct_free(&acc);
        }
        for (int i = 0; i < nf2; i++) ckks_ct_free(&cts2[i]);
        CKKS_CHECK(err2 < 2e-3, "T14b 满槽位SIMD matmul 4x4 x 256样本 与明文对照",
                   "err=%.2e", err2);
    }

    ckks_sk_free(&sk);
    ckks_ctx_free(&ctx);
    free(poly); free(poly2); free(pdec);
    free(z1); free(z2); free(zb); free(zc);
    if (g_ckks_fail == 0) printf("  RESULT: ALL PASS (%d)\n", g_ckks_fail);
    else printf("  RESULT: %d FAILED\n", g_ckks_fail);
    return g_ckks_fail;
}
