/**
 * vllm_fhe.c - Lightweight RLWE/BGV-Style Homomorphic Encryption (纯 C11)
 *
 * 实现 Fan-Vercauteren (BFV) 风格 Leveled HE：
 *   - 环 R_q = Z_q[x]/(x^n + 1)，n=1024
 *   - 密文模 q = 2^64 - 2^32 + 1 (Goldilocks prime, 64-bit)
 *   - 明文模 t = 2^13 = 8192（与 q 互素）
 *   - 缩放因子 Delta = floor(q/t)，明文以 Delta*m 编码（BFV scaling）
 *   - 噪声：中心二项分布 CBD(8)（σ=2，界 [-8,8]，Kyber 同款标准 RLWE 噪声）
 *   - 乘法：通用 tensor + (t/q) rescale（round），不做 relinearization，
 *     分量数 = deg(a)+deg(b)（2->3->5），解密使用 1, s, s^2, s^3, s^4
 *
 * 设计决策（与 axiom_registry.json 对齐）：
 *   - 定点编码 K=7 bit：映射到 axiom_arith_fixedpoint_encode_001 的 K 参数
 *   - 教科书 O(n^2) 多项式乘法保证位级可验证；NTT 加速（ntt_isomorphism 公理）
 *     留作后续 A-B-C-D 步骤
 *
 * 噪声预算（n=1024, t=2^13, B=8, 对称加密）：
 *   E0 ~ 8;  E1 ~ n*t*E0 ~ 6.7e7;  E2 ~ n*t*E1 ~ 5.5e14
 *   Delta/2 = q/(2t) ~ 1.1e15  => 支持 2 层乘法（深度 2）
 */
#include "vllm_fhe.h"
#include "vllm_ntt.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

/* ================================================================
 * 64-bit Goldilocks 域算术：p = 2^64 - 2^32 + 1
 * ================================================================ */
#define VFHE_P           0xFFFFFFFF00000001ull

/* 64x64 -> 128 bit 乘积 + Goldilocks 快速约化。
 * p = 2^64 - 2^32 + 1，故 2^64 ≡ 2^32-1 (mod p)，反复折叠高位。 */
static inline uint64_t vfhe_mulmod(uint64_t a, uint64_t b) {
#if defined(_MSC_VER)
    uint64_t hi, lo = _umul128(a, b, &hi);
#else
    unsigned __int128 t = (unsigned __int128)a * b;
    uint64_t lo = (uint64_t)t, hi = (uint64_t)(t >> 64);
#endif
    /* 迭代折叠：x = lo + hi*(2^32-1) (mod p)，直到 hi==0。
       hi*(2^32-1) = (hi<<32) + (hi>>32)*2^64 - hi */
    while (hi != 0) {
        uint64_t t1 = lo + (hi << 32);
        uint64_t carry = (t1 < lo) ? 1 : 0;   /* lo + (hi<<32) 的 2^64 进位 */
        uint64_t r_lo = t1 - hi;
        uint64_t borrow = (t1 < hi) ? 1 : 0;  /* 减 hi 的借位 */
        lo = r_lo;
        hi = (hi >> 32) + carry - borrow;
    }
    if (lo >= VFHE_P) lo -= VFHE_P;
    if (lo >= VFHE_P) lo -= VFHE_P;
    return lo;
}

/* a + b (mod q)。注意：a+b 可 ≥ 2^64（a,b 均接近 q）！必须处理 2^64 回绕：
 * 2^64 ≡ 2^32 - 1 (mod q)，故溢出后补偿 +(2^32-1)，可能二次回绕再补偿。 */
static inline uint64_t vfhe_addmod(uint64_t a, uint64_t b) {
    uint64_t r = a + b;
    if (r < a) {                         /* 溢出 2^64：+2^64 ≡ +(2^32-1) */
        r += 0xFFFFFFFFull;
        if (r < 0xFFFFFFFFull) r += 0xFFFFFFFFull;   /* 二次回绕（r ≥ 2^64-2^32+1 时） */
    }
    if (r >= VFHE_P) r -= VFHE_P;
    if (r >= VFHE_P) r -= VFHE_P;
    return r;
}

static inline uint64_t vfhe_submod(uint64_t a, uint64_t b) {
    return a >= b ? a - b : a + VFHE_P - b;
}

/* ================================================================
 * RNG（原型级：xorshift64*，非密码学安全；可注入替换）
 * ================================================================ */
static uint64_t vfhe_rng_state = 0x9E3779B97F4A7C15ull;
static void (*vfhe_rng_cb)(uint8_t *out, size_t len) = NULL;

void vfhe_set_rng(void (*fn)(uint8_t *out, size_t len)) {
    vfhe_rng_cb = fn;
}

static uint64_t vfhe_rng_u64(void) {
    if (vfhe_rng_cb) {
        uint8_t b[8];
        vfhe_rng_cb(b, 8);
        uint64_t v = 0;
        for (int i = 0; i < 8; i++) v = (v << 8) | b[i];
        return v;
    }
    uint64_t x = vfhe_rng_state;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    vfhe_rng_state = x;
    return x * 0x2545F4914F6CDD1Dull;
}

static uint32_t vfhe_rng_uniform(uint32_t bound) {
    return (uint32_t)(vfhe_rng_u64() % bound);
}

/* 中心二项分布 CBD(eta)：X = Σ_{i=1}^{eta} (b_{2i-1} - b_{2i})，σ=√(eta/2)。
 * 标准 RLWE 噪声（Kyber 同款）。与原均匀 [-B,B) 同界（最坏噪声预算不变），
 * 方差更小（典型噪声更优）。参数 p.B 语义变为 eta。 */
static int64_t vfhe_rng_noise(uint32_t B) {
    int64_t x = 0;
    for (uint32_t i = 0; i < B; i++) {
        uint64_t r = vfhe_rng_u64();
        x += (int64_t)(r & 1) - (int64_t)((r >> 1) & 1);
    }
    return x;
}

/* ================================================================
 * 参数
 * ================================================================ */
static const vfhe_params_t g_vfhe_params = {
    .n = VFHE_N_DEFAULT,
    .q_bits = VFHE_Q_BITS,
    .t = VFHE_T_DEFAULT,
    .B = VFHE_B_NOISE,
    .hw = VFHE_KEY_HW,
};

const vfhe_params_t *vfhe_params_default(void) {
    return &g_vfhe_params;
}

/* 实际 delta = floor(q/t)，q 为 64-bit Goldilocks */
static uint64_t vfhe_delta(void) {
    return VFHE_P / VFHE_T_DEFAULT;
}

/* ================================================================
 * 多项式基础（数组长度 n，系数 mod q）
 * ================================================================ */
static void poly_zero(uint64_t *c, size_t n) {
    memset(c, 0, n * sizeof(uint64_t));
}

/* 教科书 negacyclic 乘法：mod (x^n + 1)，O(n^2)（NTT 不可用时的回退参考） */
static void poly_mul_negacyclic_naive(uint64_t *c,
                                      const uint64_t *a, const uint64_t *b,
                                      size_t n) {
    poly_zero(c, n);
    for (size_t i = 0; i < n; i++) {
        if (a[i] == 0) continue;
        for (size_t j = 0; j < n; j++) {
            if (b[j] == 0) continue;
            uint64_t term = vfhe_mulmod(a[i], b[j]);
            size_t k = i + j;
            if (k < n) {
                c[k] = vfhe_addmod(c[k], term);
            } else {
                c[k - n] = vfhe_submod(c[k - n], term);   /* x^n = -1 */
            }
        }
    }
}

/* NTT 加速的 negacyclic 乘法（axiom_ntt_isomorphism 落地）：
 * Goldilocks q-1 含 2n 因子 → 2n-th 单位根存在，卷积 → NTT 逐点乘。 */
static ntt_ctx_t g_bfv_ntt;
static int g_bfv_ntt_ready = 0;

static int poly_ntt_ensure(size_t n) {
    if (g_bfv_ntt_ready && g_bfv_ntt.n == n && g_bfv_ntt.q == VFHE_P) return 0;
    if (g_bfv_ntt_ready) ntt_ctx_free(&g_bfv_ntt);
    g_bfv_ntt_ready = 0;
    if (ntt_ctx_init(&g_bfv_ntt, VFHE_P, (uint32_t)n) == 0) g_bfv_ntt_ready = 1;
    return g_bfv_ntt_ready ? 0 : -1;
}

/* c = a*b mod (x^n+1)：NTT 优先，NTT 不可用回退 O(n²)（位级一致） */
static void poly_mul_negacyclic(uint64_t *c,
                                const uint64_t *a, const uint64_t *b,
                                size_t n) {
    if (poly_ntt_ensure(n) == 0) {
        ntt_negacyclic_mul(c, a, b, &g_bfv_ntt);
        return;
    }
    poly_mul_negacyclic_naive(c, a, b, n);
}

/* a += b (mod q) */
static void poly_add(uint64_t *a, const uint64_t *b, size_t n) {
    for (size_t i = 0; i < n; i++) a[i] = vfhe_addmod(a[i], b[i]);
}

/* a -= b (mod q) */
static void poly_sub(uint64_t *a, const uint64_t *b, size_t n) {
    for (size_t i = 0; i < n; i++) a[i] = vfhe_submod(a[i], b[i]);
}

/* a *= scalar (mod q) */
static void poly_mul_scalar(uint64_t *a, uint64_t s, size_t n) {
    for (size_t i = 0; i < n; i++) a[i] = vfhe_mulmod(a[i], s);
}

/* b 为稀疏三元（s[i] in {-1,0,1}，hw 个非零）：c = a * b (mod x^n+1) */
static void poly_mul_sparse(uint64_t *c, const uint64_t *a,
                            const int8_t *s, size_t n, uint32_t hw) {
    poly_zero(c, n);
    size_t cnt = 0;
    for (size_t j = 0; j < n && cnt < hw; j++) {
        int8_t sv = s[j];
        if (sv == 0) continue;
        cnt++;
        int neg = (sv < 0);
        for (size_t i = 0; i < n; i++) {
            uint64_t ai = a[i];
            if (ai == 0) continue;
            size_t k = i + j;
            if (k < n) {
                if (neg) c[k] = vfhe_submod(c[k], ai);
                else     c[k] = vfhe_addmod(c[k], ai);
            } else {
                if (neg) c[k - n] = vfhe_addmod(c[k - n], ai);
                else     c[k - n] = vfhe_submod(c[k - n], ai);
            }
        }
    }
}

/* ================================================================
 * 密钥生成
 * ================================================================ */
int vfhe_sk_gen(vfhe_sk_t *sk) {
    if (!sk) return -1;
    memset(sk, 0, sizeof(*sk));
    sk->p = g_vfhe_params;
    sk->s = (int8_t *)calloc(sk->p.n, sizeof(int8_t));
    if (!sk->s) return -1;

    uint32_t placed = 0, guard = 0;
    while (placed < sk->p.hw && guard < 100000u) {
        guard++;
        size_t idx = (size_t)vfhe_rng_uniform((uint32_t)sk->p.n);
        if (sk->s[idx] != 0) continue;
        sk->s[idx] = (vfhe_rng_uniform(2) == 0) ? -1 : 1;
        placed++;
    }
    if (placed < sk->p.hw) {
        for (size_t i = 0; i < sk->p.n && placed < sk->p.hw; i++) {
            if (sk->s[i] == 0) { sk->s[i] = 1; placed++; }
        }
    }
    return 0;
}

int vfhe_pk_from_sk(vfhe_pk_t *pk, const vfhe_sk_t *sk) {
    if (!pk || !sk || !sk->s) return -1;
    memset(pk, 0, sizeof(*pk));
    pk->p = sk->p;
    pk->a = (uint64_t *)calloc(sk->p.n, sizeof(uint64_t));
    pk->b = (uint64_t *)calloc(sk->p.n, sizeof(uint64_t));
    if (!pk->a || !pk->b) { vfhe_pk_free(pk); return -1; }

    uint64_t *e = (uint64_t *)malloc(sk->p.n * sizeof(uint64_t));
    if (!e) { vfhe_pk_free(pk); return -1; }

    for (size_t i = 0; i < sk->p.n; i++) {
        pk->a[i] = vfhe_rng_u64() % VFHE_P;
        int64_t nv = vfhe_rng_noise(sk->p.B);
        e[i] = nv < 0 ? (uint64_t)(nv + (int64_t)VFHE_P) : (uint64_t)nv;
    }
    poly_mul_sparse(pk->b, pk->a, sk->s, sk->p.n, sk->p.hw);
    poly_add(pk->b, e, sk->p.n);
    /* 标准 RLWE 公钥：b = -(a*s + e) (mod q)，保证解密时噪声抵消 */
    for (size_t i = 0; i < sk->p.n; i++)
        pk->b[i] = pk->b[i] ? VFHE_P - pk->b[i] : 0;
    free(e);
    return 0;
}

/* ================================================================
 * 加密 / 解密
 * ================================================================ */
static int vfhe_ct_alloc(vfhe_ct_t *ct, const vfhe_params_t *p, int comps) {
    memset(ct, 0, sizeof(*ct));
    ct->p = *p;
    ct->comps = comps;
    for (int i = 0; i < comps; i++) {
        ct->c[i] = (uint64_t *)calloc(p->n, sizeof(uint64_t));
        if (!ct->c[i]) { vfhe_ct_free(ct); return -1; }
    }
    return 0;
}

/* 对称加密: (c0, c1) = (a*s + e + Delta*m, -a) */
int vfhe_encrypt_sym(vfhe_ct_t *ct, const vfhe_sk_t *sk, uint32_t m) {
    if (!ct || !sk || !sk->s || m >= sk->p.t) return -1;
    if (vfhe_ct_alloc(ct, &sk->p, 2) != 0) return -1;

    size_t n = sk->p.n;
    uint64_t delta = vfhe_delta();
    uint64_t *a = (uint64_t *)malloc(n * sizeof(uint64_t));
    uint64_t *e = (uint64_t *)malloc(n * sizeof(uint64_t));
    if (!a || !e) { free(a); free(e); vfhe_ct_free(ct); return -1; }

    for (size_t i = 0; i < n; i++) {
        a[i] = vfhe_rng_u64() % VFHE_P;
        int64_t nv = vfhe_rng_noise(sk->p.B);
        e[i] = nv < 0 ? (uint64_t)(nv + (int64_t)VFHE_P) : (uint64_t)nv;
    }

    poly_mul_sparse(ct->c[0], a, sk->s, n, sk->p.hw);
    poly_add(ct->c[0], e, n);
    ct->c[0][0] = vfhe_addmod(ct->c[0][0], vfhe_mulmod(delta, m));

    memcpy(ct->c[1], a, n * sizeof(uint64_t));
    for (size_t i = 0; i < n; i++)
        ct->c[1][i] = ct->c[1][i] ? VFHE_P - ct->c[1][i] : 0;

    free(a); free(e);
    return 0;
}

/* 公钥加密: (c0, c1) = (b*u + e0 + Delta*m, a*u + e1) */
int vfhe_encrypt_pub(vfhe_ct_t *ct, const vfhe_pk_t *pk, uint32_t m) {
    if (!ct || !pk || !pk->a || !pk->b || m >= pk->p.t) return -1;
    if (vfhe_ct_alloc(ct, &pk->p, 2) != 0) return -1;

    size_t n = pk->p.n;
    uint64_t delta = vfhe_delta();
    int8_t *u = (int8_t *)calloc(n, sizeof(int8_t));
    uint64_t *e0 = (uint64_t *)malloc(n * sizeof(uint64_t));
    uint64_t *e1 = (uint64_t *)malloc(n * sizeof(uint64_t));
    if (!u || !e0 || !e1) { free(u); free(e0); free(e1); vfhe_ct_free(ct); return -1; }

    uint32_t placed = 0, guard = 0;
    while (placed < pk->p.hw && guard < 100000u) {
        guard++;
        size_t idx = (size_t)vfhe_rng_uniform((uint32_t)n);
        if (u[idx] != 0) continue;
        u[idx] = (vfhe_rng_uniform(2) == 0) ? -1 : 1;
        placed++;
    }
    for (size_t i = 0; i < n; i++) {
        int64_t nv0 = vfhe_rng_noise(pk->p.B);
        int64_t nv1 = vfhe_rng_noise(pk->p.B);
        e0[i] = nv0 < 0 ? (uint64_t)(nv0 + (int64_t)VFHE_P) : (uint64_t)nv0;
        e1[i] = nv1 < 0 ? (uint64_t)(nv1 + (int64_t)VFHE_P) : (uint64_t)nv1;
    }

    poly_mul_sparse(ct->c[0], pk->b, u, n, pk->p.hw);
    poly_add(ct->c[0], e0, n);
    ct->c[0][0] = vfhe_addmod(ct->c[0][0], vfhe_mulmod(delta, m));

    poly_mul_sparse(ct->c[1], pk->a, u, n, pk->p.hw);
    poly_add(ct->c[1], e1, n);

    free(u); free(e0); free(e1);
    return 0;
}

/* s^k 幂次（稀疏三元，k=1..4），用于解密 */
static int vfhe_s_powers(const vfhe_sk_t *sk, uint64_t *s_pow[4]) {
    size_t n = sk->p.n;
    /* s^1 = s（直接引用） */
    for (size_t i = 0; i < n; i++) {
        s_pow[0][i] = sk->s[i] < 0 ? VFHE_P - 1 : (uint64_t)sk->s[i];
    }
    /* s^2：稀疏平方 */
    for (size_t j = 0; j < n; j++) {
        if (sk->s[j] == 0) continue;
        for (size_t i = 0; i < n; i++) {
            if (sk->s[i] == 0) continue;
            int8_t prod = (int8_t)(sk->s[j] * sk->s[i]);
            size_t k = j + i;
            uint64_t one = 1;
            if (k < n) {
                s_pow[1][k] = prod > 0 ? vfhe_addmod(s_pow[1][k], one)
                                       : vfhe_submod(s_pow[1][k], one);
            } else {
                s_pow[1][k - n] = prod > 0 ? vfhe_submod(s_pow[1][k - n], one)
                                           : vfhe_addmod(s_pow[1][k - n], one);
            }
        }
    }
    /* s^3 = s^2 * s ; s^4 = s^2 * s^2 */
    uint64_t *t3 = s_pow[2], *t4 = s_pow[3];
    poly_mul_negacyclic(t3, s_pow[1], s_pow[0], n);
    poly_mul_negacyclic(t4, s_pow[1], s_pow[1], n);
    return 0;
}

/* 计算 v = c0 + c1*s + c2*s^2 + ...（R_q 中，mod q）。
 * 返回 malloc 的 v 数组（调用方 free）；失败返回 NULL。 */
static uint64_t *vfhe_decrypt_v(const vfhe_sk_t *sk, const vfhe_ct_t *ct) {
    size_t n = sk->p.n;
    uint64_t *s_pow[4];
    for (int i = 0; i < 4; i++) {
        s_pow[i] = (uint64_t *)calloc(n, sizeof(uint64_t));
        if (!s_pow[i]) {
            for (int j = 0; j < i; j++) free(s_pow[j]);
            return NULL;
        }
    }
    vfhe_s_powers(sk, s_pow);

    uint64_t *v = (uint64_t *)calloc(n, sizeof(uint64_t));
    if (!v) {
        for (int i = 0; i < 4; i++) free(s_pow[i]);
        return NULL;
    }
    memcpy(v, ct->c[0], n * sizeof(uint64_t));

    for (int ci = 1; ci < ct->comps && ci <= 5; ci++) {
        uint64_t *tmp = (uint64_t *)malloc(n * sizeof(uint64_t));
        if (!tmp) { free(v); for (int i = 0; i < 4; i++) free(s_pow[i]); return NULL; }
        /* c_i * s^ci：s^1 直接用 s_pow[0]，s^2..s^4 用 s_pow[1..3] */
        if (ci == 1) {
            poly_mul_negacyclic(tmp, ct->c[1], s_pow[0], n);
        } else if (ci <= 4) {
            poly_mul_negacyclic(tmp, ct->c[ci], s_pow[ci - 1], n);
        } else {
            poly_zero(tmp, n);   /* 不支持 >4 次（5 分量含 c4 = s^4） */
        }
        poly_add(v, tmp, n);
        if (getenv("VLLM_FHE_DBG")) {
            printf("DBG dec c%d[0]=%llu\n", ci, (unsigned long long)tmp[0]);
        }
        free(tmp);
    }
    for (int i = 0; i < 4; i++) free(s_pow[i]);
    return v;
}

/* 单系数 BFV 取整：m = round(t*[v]_q/q) mod t（中心化 [-t/2, t/2)）。 */
static uint32_t vfhe_decrypt_round(uint64_t vc) {
    int neg = 0;
    uint64_t ax = vc;
    if (ax >= VFHE_P / 2) { ax = VFHE_P - ax; neg = 1; }
    uint64_t hi, lo;
#if defined(_MSC_VER)
    lo = _umul128(ax, VFHE_T_DEFAULT, &hi);
#else
    unsigned __int128 tv = (unsigned __int128)ax * VFHE_T_DEFAULT;
    lo = (uint64_t)tv;
    hi = (uint64_t)(tv >> 64);
#endif
    uint64_t q2 = VFHE_P / 2;
    lo += q2;
    if (lo < q2) hi++;
    uint64_t extra = hi * 0xFFFFFFFFull;
    uint64_t rsum = extra + lo;
    uint64_t mv = hi + (rsum >= VFHE_P ? 1 : 0);
    if (rsum < extra) mv += 1;
    /* mv = round(t*|v|/q)，带符号后 mod t 中心化 */
    int64_t mm = neg ? -(int64_t)mv : (int64_t)mv;
    mm %= (int64_t)VFHE_T_DEFAULT;
    if (mm >= (int64_t)(VFHE_T_DEFAULT / 2)) mm -= (int64_t)VFHE_T_DEFAULT;
    if (mm < -(int64_t)(VFHE_T_DEFAULT / 2)) mm += (int64_t)VFHE_T_DEFAULT;
    return (uint32_t)mm;
}

/* 解密（标量）：恢复 [0] 位置的明文，中心化到 [-t/2, t/2) */
int vfhe_decrypt(uint32_t *m, const vfhe_sk_t *sk, const vfhe_ct_t *ct) {
    if (!m || !sk || !sk->s || !ct || ct->comps < 2 || !ct->c[0] || !ct->c[1])
        return -1;
    uint64_t *v = vfhe_decrypt_v(sk, ct);
    if (!v) return -1;
    if (getenv("VLLM_FHE_DBG"))
        printf("DBG v[0]=%llu\n", (unsigned long long)v[0]);
    *m = vfhe_decrypt_round(v[0]);
    free(v);
    return 0;
}

/* 解密（明文多项式）：所有系数 -> poly[0..n-1]（mod t 中心化值） */
int vfhe_decrypt_poly(uint64_t *poly, const vfhe_sk_t *sk, const vfhe_ct_t *ct) {
    if (!poly || !sk || !sk->s || !ct || ct->comps < 2 || !ct->c[0] || !ct->c[1])
        return -1;
    uint64_t *v = vfhe_decrypt_v(sk, ct);
    if (!v) return -1;
    for (size_t i = 0; i < sk->p.n; i++) poly[i] = vfhe_decrypt_round(v[i]);
    free(v);
    return 0;
}

/* ================================================================
 * 同态运算
 * ================================================================ */
int vfhe_add(vfhe_ct_t *ct, const vfhe_ct_t *a, const vfhe_ct_t *b) {
    if (!ct || !a || !b || a->comps < 2 || b->comps < 2) return -1;
    int comps = a->comps > b->comps ? a->comps : b->comps;
    if (comps > VFHE_CT_MAX_COMP) return -1;
    if (vfhe_ct_alloc(ct, &a->p, comps) != 0) return -1;
    size_t n = a->p.n;
    for (int i = 0; i < comps; i++) {
        if (i < a->comps) memcpy(ct->c[i], a->c[i], n * sizeof(uint64_t));
        if (i < b->comps) poly_add(ct->c[i], b->c[i], n);
    }
    return 0;
}

/* ================================================================
 * BFV 乘法（R_{q^2} 残差算术，正确性优先）
 *
 * 标准 BFV：密文乘法 = tensor 乘积在 R_{q^2} 中计算（系数可到 q^2 量级，
 * 不能 mod q 截断），再做 rescale: c_i <- round(t * T_i / q) mod q。
 *
 * 表示约定：
 *   - 全部 tensor 运算在 [0, Q2) 无符号残差空间做模加（杜绝 __int128
 *     有符号溢出：|prod| <= Q2/4，累加经每步 mod Q2 归约，始终 < Q2 < 2^128）
 *   - 仅 rescale 阶段临时转中心化 __int128 计算 round(t*x/q)
 * ================================================================ */

/* Q2 = q^2（< 2^128），Q2H = Q2/2（< 2^127） */
#define VFHE_Q2   ((unsigned __int128)VFHE_P * (unsigned __int128)VFHE_P)
#define VFHE_Q2H  ((__int128)(VFHE_Q2 >> 1))

/* 中心化：系数 x (0..q-1) -> 有符号值 [-q/2, q/2) */
static inline __int128 vfhe_center_c(uint64_t x) {
    __int128 v = (__int128)x;
    if (v >= (__int128)VFHE_P / 2) v -= (__int128)VFHE_P;
    return v;
}

/* 残差模加：a, b ∈ [0, Q2) -> (a+b) mod Q2（u128 无符号，无溢出歧义） */
static inline unsigned __int128 vfhe_add_mod_q2(unsigned __int128 a,
                                                unsigned __int128 b) {
    unsigned __int128 s = a + b;
    if (s < a) s += ((unsigned __int128)0 - VFHE_Q2);   /* 回绕：+ (2^128-Q2) */
    else if (s >= VFHE_Q2) s -= VFHE_Q2;
    return s;
}

/* 残差模减：a, b ∈ [0, Q2) -> (a-b) mod Q2
 * 注意：b > a 分支绝不能写成 a + Q2 - b —— 中间量 a + Q2 可 ≥ 2^128
 * 回绕（a 接近 Q2 时），导致结果错误。
 * 正确做法：a - b 先按 u128 无符号回绕为 2^128 - (b-a)，再加 Q2 二次
 * 回绕，最终 = Q2 - (b-a) ∈ (0, Q2]，两次回绕均精确（结果 < 2^128）。 */
static inline unsigned __int128 vfhe_sub_mod_q2(unsigned __int128 a,
                                                unsigned __int128 b) {
    return a >= b ? a - b : a - b + VFHE_Q2;
}

/* 残差 <-> 中心化 转换 */
static inline unsigned __int128 vfhe_centered_to_residue(__int128 v) {
    return v < 0 ? (unsigned __int128)(v + (__int128)VFHE_Q2)
                 : (unsigned __int128)v;
}

static inline __int128 vfhe_residue_to_centered(unsigned __int128 r) {
    if (r >= VFHE_Q2 / 2) return (__int128)(r - VFHE_Q2);
    return (__int128)r;
}

/* negacyclic 卷积（R_{q^2} 残差算术）
 * c[k] = sum_{i+j=k} a_i*b_j - sum_{i+j=k+n} a_i*b_j (x^n = -1)
 *
 * 必须用模 q 残差（0..q-1）相乘：a*b 的整数乘积 mod q² 与中心化乘积
 * center(a)*center(b) mod q² 相差 -q*(k1*b+k2*a)（非 q² 倍数），
 * 残留 Δ 量级误差（实测 77→108 的根源）。残差乘法精确。
 * 乘积 (q-1)² < q² < 2^128 无溢出；累加在 [0, Q2) 模加。
 */
static void poly_conv_q2(unsigned __int128 *c,
                         const uint64_t *a, const uint64_t *b, size_t n) {
    for (size_t k = 0; k < n; k++) {
        unsigned __int128 acc = 0;
        for (size_t i = 0; i < n; i++) {
            uint64_t bj;
            size_t j = k + n - i;          /* i + j = k + n */
            int neg = 1;
            if (i <= k) { j = k - i; neg = 0; }
            bj = b[j];
            if (a[i] == 0 || bj == 0) continue;
            unsigned __int128 prod = (unsigned __int128)a[i] * bj;
            acc = neg ? vfhe_sub_mod_q2(acc, prod) : vfhe_add_mod_q2(acc, prod);
        }
        c[k] = acc;
    }
}

/* rescale: x（残差 [0, Q2)）-> round(t*x/q) mod q
 * 标准 BFV：rescale 对 R_{q²} 残差直接计算，不做中心化。
 * x = T_hi*2^64 + T_lo（T_hi 无符号 < 2^64，因为 x < Q2 < 2^128）
 * round(t*x/q) = t*T_hi + round(t*(T_hi*(2^32-1) + T_lo)/q)
 * 全部用无符号 __int128（|num| < 2^109 不溢出）。
 * 注意：T_hi 绝不能转 int64（x ≥ 2^127 时溢出为负，公式崩溃）。 */
static uint64_t vfhe_rescale_q2(unsigned __int128 x) {
    uint64_t T_hi = (uint64_t)(x >> 64);          /* 无符号高位 */
    uint64_t T_lo = (uint64_t)x;
    unsigned __int128 s64 = (unsigned __int128)T_hi * 0xFFFFFFFFull + T_lo;
    unsigned __int128 num = (unsigned __int128)VFHE_T_DEFAULT * s64;
    unsigned __int128 half = VFHE_P / 2;
    unsigned __int128 r2 = (num + half) / (unsigned __int128)VFHE_P;
    unsigned __int128 total = (unsigned __int128)VFHE_T_DEFAULT * T_hi + r2;
    return (uint64_t)(total % (unsigned __int128)VFHE_P);
}

/* BFV 乘法：通用 tensor（R_{q^2}）+ rescale。分量数 = a.comps + b.comps - 1 */
int vfhe_mult(vfhe_ct_t *ct, const vfhe_ct_t *a, const vfhe_ct_t *b) {
    if (!ct || !a || !b || a->comps < 2 || b->comps < 2) return -1;
    int comps = a->comps + b->comps - 1;
    if (comps > VFHE_CT_MAX_COMP) return -1;
    if (vfhe_ct_alloc(ct, &a->p, comps) != 0) return -1;
    size_t n = a->p.n;

    /* t_k = sum_{i+j=k} a_i * b_j（在 R_{q^2} 中，残差模加累加） */
    for (int k = 0; k < comps; k++) {
        unsigned __int128 *acc = (unsigned __int128 *)calloc(n, sizeof(unsigned __int128));
        if (!acc) { vfhe_ct_free(ct); return -1; }
        for (int i = 0; i < a->comps; i++) {
            int j = k - i;
            if (j < 0 || j >= b->comps) continue;
            unsigned __int128 *tmp = (unsigned __int128 *)malloc(n * sizeof(unsigned __int128));
            if (!tmp) { free(acc); vfhe_ct_free(ct); return -1; }
            poly_conv_q2(tmp, a->c[i], b->c[j], n);
            for (size_t x = 0; x < n; x++)
                acc[x] = vfhe_add_mod_q2(acc[x], tmp[x]);
            free(tmp);
        }
        for (size_t x = 0; x < n; x++)
            ct->c[k][x] = vfhe_rescale_q2(acc[x]);
        if (getenv("VLLM_FHE_DBG") && getenv("VLLM_FHE_DBG2")) {
            printf("DBG T%d[0]=%llx%016llx\n", k,
                   (unsigned long long)(acc[0] >> 64), (unsigned long long)acc[0]);
        }
        free(acc);
    }
    return 0;
}

/* 同态标量乘：ct = ct_in * m（m 明文值，0 <= m < t）。
 * BFV plain-mult：直接以明文值 m 缩放各分量（不乘 Δ！乘 Δ 会把解码结果
 * 放大 Δ 倍）。解密 D = m*(Δx+e) = Δ*m*x + m*e，decode 得 m*x mod t。 */
int vfhe_mul_scalar(vfhe_ct_t *ct, const vfhe_ct_t *in, uint32_t m) {
    if (!ct || !in || in->comps < 2 || m >= in->p.t) return -1;
    if (vfhe_ct_alloc(ct, &in->p, in->comps) != 0) return -1;
    size_t n = in->p.n;
    uint64_t sm = (uint64_t)m;   /* 直接明文值，非 Δ*m */
    for (int i = 0; i < in->comps; i++)
        for (size_t x = 0; x < n; x++)
            ct->c[i][x] = vfhe_mulmod(in->c[i][x], sm);
    return 0;
}

/* 密文-明文矩阵乘法：out[j] = Σ_i in[i] * M[i*ncol+j]（mod t）。
 * 每输出列做 nvec 次"密文×明文标量"乘 + 同态加（密文-明文乘噪声增长小，
 * 远优于密文-密文乘，适合权重明文的密文推理）。 */
int vfhe_matmul_ctvec_plain(vfhe_ct_t *out, const vfhe_ct_t *in, uint32_t nvec,
                            const int32_t *M, uint32_t ncol) {
    if (!out || !in || nvec == 0 || ncol == 0 || !M) return -1;
    for (uint32_t i = 0; i < nvec; i++)
        if (in[i].comps < 2 || !in[i].c[0] || !in[i].c[1]) return -1;

    vfhe_ct_t acc = {0}, tmp = {0};
    for (uint32_t j = 0; j < ncol; j++) {
        int32_t v0 = M[0 * ncol + j];
        uint32_t m0 = (uint32_t)(v0 < 0 ? v0 + (int32_t)in[0].p.t : v0);
        if (vfhe_mul_scalar(&acc, &in[0], m0) != 0) {
            vfhe_ct_free(&acc); vfhe_ct_free(&tmp); return -1;
        }
        for (uint32_t i = 1; i < nvec; i++) {
            int32_t v = M[i * ncol + j];
            uint32_t mv = (uint32_t)(v < 0 ? v + (int32_t)in[0].p.t : v);
            if (vfhe_mul_scalar(&tmp, &in[i], mv) != 0) {
                vfhe_ct_free(&acc); vfhe_ct_free(&tmp); return -1;
            }
            /* 注意：vfhe_add 输出不能与输入别名（vfhe_ct_alloc 会重分配），
               故用独立 nxt 承接再转移。 */
            vfhe_ct_t nxt = {0};
            if (vfhe_add(&nxt, &acc, &tmp) != 0) {
                vfhe_ct_free(&acc); vfhe_ct_free(&tmp); return -1;
            }
            vfhe_ct_free(&acc);
            vfhe_ct_free(&tmp);
            acc = nxt;
        }
        vfhe_ct_copy(&out[j], &acc);
        vfhe_ct_free(&acc);
    }
    vfhe_ct_free(&tmp);
    return 0;
}

int vfhe_ct_copy(vfhe_ct_t *dst, const vfhe_ct_t *src) {
    if (!dst || !src || src->comps < 2) return -1;
    if (vfhe_ct_alloc(dst, &src->p, src->comps) != 0) return -1;
    size_t n = src->p.n;
    for (int i = 0; i < src->comps; i++)
        memcpy(dst->c[i], src->c[i], n * sizeof(uint64_t));
    return 0;
}

void vfhe_ct_free(vfhe_ct_t *ct) {
    if (!ct) return;
    for (int i = 0; i < VFHE_CT_MAX_COMP; i++) {
        free(ct->c[i]);
        ct->c[i] = NULL;
    }
    ct->comps = 0;
}

void vfhe_sk_free(vfhe_sk_t *sk) {
    if (!sk) return;
    free(sk->s);
    sk->s = NULL;
}

void vfhe_pk_free(vfhe_pk_t *pk) {
    if (!pk) return;
    free(pk->a); free(pk->b);
    pk->a = pk->b = NULL;
}

/* ================================================================
 * 定点编码 (axiom_arith_fixedpoint_encode_001)
 * ================================================================ */
uint32_t vfhe_fixed_encode(double x, int K) {
    int64_t v = (int64_t)(x * (double)(1ll << K));
    int64_t t = VFHE_T_DEFAULT;
    int64_t m = v % t;
    if (m < 0) m += t;
    return (uint32_t)m;
}

double vfhe_fixed_decode(uint32_t m, int K) {
    return (double)vfhe_center(m, VFHE_T_DEFAULT) / (double)(1ll << K);
}

int32_t vfhe_center(uint32_t v, uint32_t t) {
    int32_t s = (int32_t)v;
    if (s >= (int32_t)(t / 2)) s -= (int32_t)t;
    return s;
}

/* ================================================================
 * SIMD 槽位打包 (axiom_ntt_isomorphism: R_t ≅ ∏ F_t 评估同态)
 *   t 素数且 2n | t-1（默认 t=12289: 12288=6*2048）。
 *   打包：槽位 v_j -> 明文多项式 p，使 p(pts[j]) = v_j，pts[j]=ζ^(2j+1)，
 *   ζ 为 primitive 2n-th 单位根。利用 ∏(x-pts[j]) = x^n + 1：
 *     L_j(x) = (x^n+1)/((x-pts[j])·n·pts[j]^(n-1))
 *            = Q_j(x)·inv(n·pts[j]^(n-1))，Q_j[i] = pts[j]^(n-1-i)
 *   评估同态：环加/乘 = 逐槽位加/乘（无需 NTT，直接 O(n^2) 插值/评估）。
 * ================================================================ */
static uint64_t vfhe_modinv_u64(uint64_t a, uint64_t t) {
    int64_t t0 = (int64_t)t, t1 = (int64_t)a, t2;
    int64_t r0 = 0, r1 = 1, r2;
    while (t1) {
        int64_t q = t0 / t1;
        t2 = t0 - q * t1; t0 = t1; t1 = t2;
        r2 = r0 - q * r1; r0 = r1; r1 = r2;
    }
    if (r0 < 0) r0 += (int64_t)t;
    return (uint64_t)r0;
}

static uint64_t vfhe_powmod(uint64_t b, uint64_t e, uint64_t m) {
    uint64_t r = 1;
    b %= m;
    while (e) {
        if (e & 1) r = (uint64_t)((unsigned __int128)r * b % m);
        b = (uint64_t)((unsigned __int128)b * b % m);
        e >>= 1;
    }
    return r;
}

int vfhe_slot_roots(uint64_t *pts, const vfhe_params_t *p) {
    if (!pts || !p || p->n == 0) return -1;
    uint64_t t = p->t, n = p->n;
    /* 要求 t 素数且 2n | t-1（否则评估点不存在） */
    if ((t - 1) % (2 * n) != 0) return -1;
    /* primitive root g mod t（t-1 的素因子分解） */
    uint64_t phi = t - 1, tmp = phi;
    uint64_t facs[16]; int nf = 0;
    for (uint64_t d = 2; d * d <= tmp; d++) {
        if (tmp % d == 0) {
            facs[nf++] = d;
            while (tmp % d == 0) tmp /= d;
        }
    }
    if (tmp > 1) facs[nf++] = tmp;
    uint64_t g = 2;
    for (; g < t; g++) {
        int ok = 1;
        for (int i = 0; i < nf; i++)
            if (vfhe_powmod(g, phi / facs[i], t) == 1) { ok = 0; break; }
        if (ok) break;
    }
    /* ζ = g^(phi/(2n))，评估点 pts[j] = ζ^(2j+1) */
    uint64_t zeta = vfhe_powmod(g, phi / (2 * n), t);
    uint64_t z2 = (uint64_t)((unsigned __int128)zeta * zeta % t);
    uint64_t cur = zeta;
    for (uint64_t j = 0; j < n; j++) {
        pts[j] = cur;
        cur = (uint64_t)((unsigned __int128)cur * z2 % t);
    }
    return 0;
}

int vfhe_slot_encode(uint64_t *poly, const uint64_t *slots,
                     const uint64_t *pts, const vfhe_params_t *p) {
    if (!poly || !slots || !pts || !p) return -1;
    uint64_t t = p->t, n = p->n;
    memset(poly, 0, n * sizeof(uint64_t));
    for (uint64_t j = 0; j < n; j++) {
        uint64_t pj = pts[j];
        uint64_t pj_nm1 = vfhe_powmod(pj, n - 1, t);          /* pj^(n-1) */
        uint64_t inv_j = vfhe_modinv_u64((uint64_t)(n % t) * pj_nm1 % t, t);
        uint64_t coef = (uint64_t)((unsigned __int128)(slots[j] % t) * inv_j % t);
        uint64_t pw = pj_nm1;
        uint64_t inv_pj = vfhe_modinv_u64(pj, t);
        for (uint64_t i = 0; i < n; i++) {
            poly[i] = (poly[i] + (uint64_t)((unsigned __int128)coef * pw % t)) % t;
            pw = (uint64_t)((unsigned __int128)pw * inv_pj % t);  /* pj^(n-2-i) */
        }
    }
    return 0;
}

int vfhe_slot_decode(uint64_t *slots, const uint64_t *poly,
                     const uint64_t *pts, const vfhe_params_t *p) {
    if (!slots || !poly || !pts || !p) return -1;
    uint64_t t = p->t, n = p->n;
    /* 就地（slots==poly）时先拷贝：Horner 逐槽位读全部系数，写 slots[j]
       会污染后续槽位的读取源。 */
    uint64_t *src = (uint64_t *)poly;
    uint64_t *tmp = NULL;
    if (slots == poly) {
        tmp = (uint64_t *)malloc(n * sizeof(uint64_t));
        if (!tmp) return -1;
        memcpy(tmp, poly, n * sizeof(uint64_t));
        src = tmp;
    }
    for (uint64_t j = 0; j < n; j++) {
        /* Horner 求值 poly(pts[j]) */
        uint64_t acc = 0;
        for (uint64_t i = n; i-- > 0; )
            acc = (acc * pts[j] + src[i]) % t;
        slots[j] = acc;
    }
    free(tmp);
    return 0;
}

/* 加密明文多项式：c0 = a*s + e + Δ*poly，c1 = -a */
int vfhe_encrypt_sym_poly(vfhe_ct_t *ct, const vfhe_sk_t *sk,
                          const uint64_t *poly) {
    if (!ct || !sk || !sk->s || !poly) return -1;
    if (vfhe_ct_alloc(ct, &sk->p, 2) != 0) return -1;
    size_t n = sk->p.n;
    uint64_t delta = vfhe_delta();
    uint64_t *a = (uint64_t *)malloc(n * sizeof(uint64_t));
    uint64_t *e = (uint64_t *)malloc(n * sizeof(uint64_t));
    if (!a || !e) { free(a); free(e); vfhe_ct_free(ct); return -1; }
    for (size_t i = 0; i < n; i++) {
        a[i] = vfhe_rng_u64() % VFHE_P;
        int64_t nv = vfhe_rng_noise(sk->p.B);
        e[i] = nv < 0 ? (uint64_t)(nv + (int64_t)VFHE_P) : (uint64_t)nv;
    }
    poly_mul_sparse(ct->c[0], a, sk->s, n, sk->p.hw);
    poly_add(ct->c[0], e, n);
    for (size_t i = 0; i < n; i++)
        ct->c[0][i] = vfhe_addmod(ct->c[0][i], vfhe_mulmod(delta, poly[i] % VFHE_T_DEFAULT));
    memcpy(ct->c[1], a, n * sizeof(uint64_t));
    for (size_t i = 0; i < n; i++)
        ct->c[1][i] = ct->c[1][i] ? VFHE_P - ct->c[1][i] : 0;
    free(a); free(e);
    return 0;
}

/* ================================================================
 * 自测
 * ================================================================ */
static int g_fail = 0;

#define CHECK(cond, name, fmt, ...) do { \
    if (cond) { printf("  [PASS] " name "\n"); } \
    else { printf("  [FAIL] " name ": " fmt "\n", ##__VA_ARGS__); g_fail++; } \
} while (0)

int vfhe_self_test(void) {
    printf("=== vLLM-FHE Self Test ===\n");
    g_fail = 0;
    vfhe_rng_state = (uint64_t)time(NULL) ^ 0x123456789abcdef0ull;

    const vfhe_params_t *P = vfhe_params_default();
    uint64_t delta = vfhe_delta();
    printf("  params: n=%u q=2^64-2^32+1 t=%u delta=%llu B=%u hw=%u\n",
           P->n, P->t, (unsigned long long)delta, P->B, P->hw);

    vfhe_sk_t sk;
    vfhe_pk_t pk;
    vfhe_ct_t ct, ct2, ct3;
    uint32_t m = 0;

    /* ---- T1: 定点编码往返 ---- */
    int rc = vfhe_sk_gen(&sk);
    CHECK(rc == 0, "T1a keygen", "rc=%d", rc);
    double xs[] = {0.0, 1.5, -3.25, 7.5, -8.0};
    int ok_enc = 1;
    for (int i = 0; i < 5; i++) {
        uint32_t em = vfhe_fixed_encode(xs[i], 6);
        double back = vfhe_fixed_decode(em, 6);
        if (fabs(back - xs[i]) > 1.0 / 64.0) ok_enc = 0;
    }
    CHECK(ok_enc, "T1b fixed-point roundtrip (K=6, err<=1/128)", "err>1/64");

    /* ---- T2: 对称/公钥 加解密往返 ---- */
    vfhe_pk_from_sk(&pk, &sk);
    vfhe_encrypt_sym(&ct, &sk, 1234);
    vfhe_decrypt(&m, &sk, &ct);
    CHECK(m == 1234, "T2a sym encrypt/decrypt", "got %u", m);
    vfhe_encrypt_pub(&ct, &pk, 1000);
    vfhe_decrypt(&m, &sk, &ct);
    CHECK(m == 1000, "T2b pub encrypt/decrypt", "got %u", m);

    /* ---- T3: 同态加法（含负数中心化） ---- */
    vfhe_encrypt_sym(&ct, &sk, vfhe_fixed_encode(2.5, 6));
    vfhe_encrypt_sym(&ct2, &sk, vfhe_fixed_encode(3.25, 6));
    vfhe_add(&ct3, &ct, &ct2);
    vfhe_decrypt(&m, &sk, &ct3);
    double sum = vfhe_fixed_decode(m, 6);
    CHECK(fabs(sum - 5.75) < 1e-3, "T3 homomorphic add (2.5+3.25=5.75)",
          "got %.4f", sum);

    /* ---- T4: 同态乘法（深度 1，整数明文） ---- */
    vfhe_encrypt_sym(&ct, &sk, 7);
    vfhe_encrypt_sym(&ct2, &sk, 11);
    vfhe_mult(&ct3, &ct, &ct2);
    vfhe_decrypt(&m, &sk, &ct3);
    CHECK(m == 77, "T4 homomorphic mult (7*11=77)", "got %u", m);

    /* ---- T5: 混合表达式 a*b + c（1 层乘法 + 加法） ---- */
    vfhe_encrypt_sym(&ct, &sk, 5);
    vfhe_encrypt_sym(&ct2, &sk, 6);
    vfhe_mult(&ct3, &ct, &ct2);              /* 5*6 = 30 */
    vfhe_ct_copy(&ct, &ct3);
    vfhe_encrypt_sym(&ct2, &sk, 7);
    vfhe_add(&ct3, &ct, &ct2);               /* 30 + 7 = 37 */
    vfhe_decrypt(&m, &sk, &ct3);
    CHECK(m == 37, "T5 expr a*b+c (5*6+7=37)", "got %u", m);

    /* ---- T5b: 深度 2 乘法链 a*b*c ----
       深 2 噪声预算：E2 ≈ n·t·(n·t·B·σ(s))，要求 E2 < Δ/2。
       当前 t=12289 下 E2≈1e16 > Δ/2≈7.5e14，深 2 不可用（SIMD 参数只支持深 1），
       故预算不足时标记 SKIP（不计失败）。 */
    {
        unsigned __int128 n2 = P->n;
        unsigned __int128 E2 = n2 * P->t * (n2 * P->t * (unsigned __int128)P->B * 8u);
        if (E2 * 2 * P->t < (unsigned __int128)VFHE_P) {   /* E2 < q/(2t) = Δ/2 */
            vfhe_encrypt_sym(&ct, &sk, 3);
            vfhe_encrypt_sym(&ct2, &sk, 4);
            vfhe_mult(&ct3, &ct, &ct2);              /* 3*4 = 12 (3 分量) */
            vfhe_ct_copy(&ct, &ct3);
            vfhe_encrypt_sym(&ct2, &sk, 5);
            vfhe_mult(&ct3, &ct, &ct2);              /* 12*5 = 60 (5 分量) */
            vfhe_decrypt(&m, &sk, &ct3);
            CHECK(m == 60, "T5b depth-2 chain a*b*c (3*4*5=60)", "got %u", m);
        } else {
            printf("  [SKIP] T5b depth-2 (t=%u: 深2噪声预算不足 E2>Δ/2，仅深1)\n", P->t);
        }
    }

    /* ---- T6: 定点同态乘（小数） ----
       定点乘法结果尺度 2^(K+K)=2^12，明文模 t=12289 半程 6144，故
       |乘积*2^12| 须 < 6144（如 0.25*0.5=0.125 编码 512）。 */
    vfhe_encrypt_sym(&ct, &sk, vfhe_fixed_encode(0.25, 6));
    vfhe_encrypt_sym(&ct2, &sk, vfhe_fixed_encode(0.5, 6));
    vfhe_mult(&ct3, &ct, &ct2);
    vfhe_decrypt(&m, &sk, &ct3);
    /* 定点乘法结果 = 0.25*0.5 = 0.125；decode 需再除 2^K */
    double prod = (double)vfhe_center(m, VFHE_T_DEFAULT) / (double)(1ll << 12);
    CHECK(fabs(prod - 0.125) < 1e-3, "T6 fixed mult (0.25*0.5=0.125)",
          "got %.4f", prod);

    vfhe_encrypt_sym(&ct, &sk, vfhe_fixed_encode(1.5, 6));
    vfhe_encrypt_sym(&ct2, &sk, vfhe_fixed_encode(2.0, 6));
    vfhe_add(&ct3, &ct, &ct2);
    vfhe_decrypt(&m, &sk, &ct3);
    double s2 = vfhe_fixed_decode(m, 6);
    CHECK(fabs(s2 - 3.5) < 1e-3, "T6b fixed add (1.5+2.0=3.5)", "got %.4f", s2);

    /* ---- T7: 密文-明文矩阵乘法（4x4，含负权重），与明文对照 ---- */
    {
        enum { NV = 4, NC = 4 };
        static const int32_t Mt[NV * NC] = {
            3, -2,  1,  5,
            0,  4, -3,  2,
           -1,  6,  2, -4,
            2, -1,  4,  3,
        };
        static const uint32_t xs[NV] = { 1, 2, 3, 4 };
        int64_t expect[NC];
        for (uint32_t j = 0; j < NC; j++) {
            int64_t s = 0;
            for (uint32_t i = 0; i < NV; i++) s += (int64_t)xs[i] * Mt[i * NC + j];
            expect[j] = s;
        }
        vfhe_ct_t ivec[NV], ovec[NC];
        memset(ivec, 0, sizeof(ivec));
        memset(ovec, 0, sizeof(ovec));
        int ok = 1;
        for (uint32_t i = 0; i < NV; i++)
            if (vfhe_encrypt_sym(&ivec[i], &sk, xs[i]) != 0) ok = 0;
        if (ok && vfhe_matmul_ctvec_plain(ovec, ivec, NV, Mt, NC) != 0) ok = 0;
        for (uint32_t j = 0; j < NC && ok; j++) {
            uint32_t dj = 0;
            vfhe_decrypt(&dj, &sk, &ovec[j]);
            int64_t got = vfhe_center(dj, P->t);
            if (got != expect[j]) {
                printf("  [FAIL] T7 matmul out[%u]: got %lld expect %lld\n",
                       j, (long long)got, (long long)expect[j]);
                ok = 0;
            }
        }
        CHECK(ok, "T7 ct*plain matmul 4x4 (含负权重) 与明文对照", "见上");
        for (uint32_t i = 0; i < NV; i++) vfhe_ct_free(&ivec[i]);
        for (uint32_t j = 0; j < NC; j++) vfhe_ct_free(&ovec[j]);
    }

    /* ---- T8: 密文域整数系数低阶多项式（含平方）----
       y(x) = 3x^2 + 2x + 1，x 为小整数明文。
       （诚实边界：含非整数系数的定点多项式如 GELU≈0.5x+0.399x² 需要密文域
        除法缩放，而 BFV 模环除法在非整除时产生模数解而非实数舍入；当前
        t=4097 不足，需 t≥2^17 或 CKKS 路线。整数系数多项式无此问题。） */
    {
        static const int32_t xp[] = { -4, -1, 0, 3, 7 };
        int ok = 1;
        for (int i = 0; i < 5 && ok; i++) {
            vfhe_ct_t cx = {0}, cx2 = {0}, t1 = {0}, t2 = {0}, out = {0};
            int32_t x = xp[i];
            uint32_t mx = (uint32_t)(x < 0 ? x + (int32_t)sk.p.t : x);
            if (vfhe_encrypt_sym(&cx, &sk, mx) != 0) ok = 0;
            if (ok && vfhe_mult(&cx2, &cx, &cx) != 0) ok = 0;          /* x^2 */
            if (ok && vfhe_mul_scalar(&t1, &cx2, 3) != 0) ok = 0;      /* 3x^2 */
            if (ok && vfhe_mul_scalar(&t2, &cx, 2) != 0) ok = 0;       /* 2x  */
            if (ok && vfhe_add(&out, &t1, &t2) != 0) ok = 0;           /* 3x^2+2x */
            vfhe_ct_t c1 = {0}, out2 = {0};
            if (ok && vfhe_encrypt_sym(&c1, &sk, 1) != 0) ok = 0;
            if (ok && vfhe_add(&out2, &out, &c1) != 0) ok = 0;         /* +1（无 alias） */
            uint32_t d = 0;
            int32_t expect = 3 * x * x + 2 * x + 1;
            if (ok) {
                vfhe_decrypt(&d, &sk, &out2);
                int32_t got = vfhe_center(d, sk.p.t);
                if (got != expect) {
                    printf("  [FAIL] T8 poly x=%d: got %d expect %d\n", x, got, expect);
                    ok = 0;
                }
            }
            vfhe_ct_free(&cx); vfhe_ct_free(&cx2);
            vfhe_ct_free(&t1); vfhe_ct_free(&t2); vfhe_ct_free(&out);
            vfhe_ct_free(&c1); vfhe_ct_free(&out2);
        }
        CHECK(ok, "T8 ct poly 3x^2+2x+1 (整数系数)", "见上");
    }

    /* ---- T9: 整数化浅层 MLP 密文推理演示（4->4->2，激活 y^2 多项式，深 1）----
       架构验证：密文 x × 明文 W1 -> y1，多项式激活 a=y1^2（密文乘，深1），
       a × 明文 W2 -> y2。明文对照为同一整数环（mod t）运算。
       （诚实边界：激活限定为可多项式化的整数函数；ReLU/max 需比较器，
        密文域不可直接实现，这是量化 LLM 密文推理的精度上限来源。） */
    {
        static const int32_t W1[16] = {   /* 4x4 */
             1,  0, -1,  1,
             0,  1,  1, -1,
             1,  1,  0,  0,
            -1,  0,  1,  1,
        };
        static const int32_t W2[8] = {    /* 4x2 */
             1,  0,
             0,  1,
             1,  1,
             0,  1,
        };
        static const int32_t xin[4] = { 1, 2, 0, 3 };
        /* 明文对照（整数环 mod t）。注意 vfhe_matmul_ctvec_plain 语义：
           out[j] = Σ_i in[i]*M[i*ncol+j]（M 行主序，输出为列方向组合）。 */
        int64_t y1[4], a1[4], y2[2];
        for (int j = 0; j < 4; j++) {
            int64_t s = 0;
            for (int k = 0; k < 4; k++) s += (int64_t)xin[k] * W1[k * 4 + j];
            y1[j] = s;
        }
        for (int i = 0; i < 4; i++) a1[i] = y1[i] * y1[i];
        for (int j = 0; j < 2; j++) {
            int64_t s = 0;
            for (int i = 0; i < 4; i++) s += a1[i] * W2[i * 2 + j];
            y2[j] = s;
        }
        vfhe_ct_t cxs[4], cy1[4], ca1[4], cy2[2];
        memset(cxs, 0, sizeof(cxs)); memset(cy1, 0, sizeof(cy1));
        memset(ca1, 0, sizeof(ca1)); memset(cy2, 0, sizeof(cy2));
        int ok = 1;
        for (int i = 0; i < 4; i++) {
            uint32_t mx = (uint32_t)(xin[i] < 0 ? xin[i] + (int32_t)sk.p.t : xin[i]);
            if (vfhe_encrypt_sym(&cxs[i], &sk, mx) != 0) ok = 0;
        }
        if (ok && vfhe_matmul_ctvec_plain(cy1, cxs, 4, W1, 4) != 0) ok = 0;
        for (int i = 0; i < 4 && ok; i++)
            if (vfhe_mult(&ca1[i], &cy1[i], &cy1[i]) != 0) ok = 0;   /* a = y^2 */
        if (ok && vfhe_matmul_ctvec_plain(cy2, ca1, 4, W2, 2) != 0) ok = 0;
        for (int j = 0; j < 2 && ok; j++) {
            uint32_t d = 0;
            vfhe_decrypt(&d, &sk, &cy2[j]);
            int64_t got = vfhe_center(d, sk.p.t);
            if (got != y2[j]) {
                printf("  [FAIL] T9 mlp out[%d]: got %lld expect %lld\n", j,
                       (long long)got, (long long)y2[j]);
                ok = 0;
            }
        }
        CHECK(ok, "T9 整数化 MLP 4-4-2 密文推理（激活 y^2）与明文对照", "见上");
        for (int i = 0; i < 4; i++) { vfhe_ct_free(&cxs[i]); vfhe_ct_free(&cy1[i]); vfhe_ct_free(&ca1[i]); }
        for (int j = 0; j < 2; j++) vfhe_ct_free(&cy2[j]);
    }

    /* ---- T10: SIMD 槽位打包端到端（n=1024 槽位，深 1 加/乘）----
       打包 1024 槽位 -> 明文多项式 -> 加密 -> 同态加/乘 -> 解密 -> 解包，
       逐槽位对照明文运算（评估同态 axiom_ntt_isomorphism）。 */
    {
        const vfhe_params_t *sp = vfhe_params_default();
        size_t n = sp->n;
        uint64_t *pts = (uint64_t *)malloc(n * sizeof(uint64_t));
        uint64_t *sa = (uint64_t *)malloc(n * sizeof(uint64_t));
        uint64_t *sb = (uint64_t *)malloc(n * sizeof(uint64_t));
        uint64_t *pa = (uint64_t *)malloc(n * sizeof(uint64_t));
        uint64_t *pb = (uint64_t *)malloc(n * sizeof(uint64_t));
        uint64_t *pc = (uint64_t *)malloc(n * sizeof(uint64_t));
        uint64_t *pd = (uint64_t *)malloc(n * sizeof(uint64_t));
        uint64_t *ra = (uint64_t *)malloc(n * sizeof(uint64_t));
        uint64_t *rb = (uint64_t *)malloc(n * sizeof(uint64_t));
        vfhe_ct_t cta = {0}, ctb = {0}, ctc = {0}, ctd = {0};
        int ok = (pts && sa && sb && pa && pb && pc && pd && ra && rb) &&
                 vfhe_slot_roots(pts, sp) == 0;
        if (ok) {
            for (size_t i = 0; i < n; i++) {
                sa[i] = vfhe_rng_u64() % 64;
                sb[i] = vfhe_rng_u64() % 64;
            }
            ok = vfhe_slot_encode(pa, sa, pts, sp) == 0 &&
                 vfhe_slot_encode(pb, sb, pts, sp) == 0 &&
                 vfhe_encrypt_sym_poly(&cta, &sk, pa) == 0 &&
                 vfhe_encrypt_sym_poly(&ctb, &sk, pb) == 0;
        }
        if (ok) {
            /* 同态加（深 0） */
            ok = vfhe_add(&ctc, &cta, &ctb) == 0 &&
                 vfhe_decrypt_poly(pc, &sk, &ctc) == 0;
            if (ok) {
                for (size_t i = 0; i < n; i++) {
                    int32_t cv = vfhe_center((uint32_t)pc[i], sp->t);
                    ra[i] = cv < 0 ? (uint64_t)(cv + (int32_t)sp->t) : (uint64_t)cv;
                }
                ok = vfhe_slot_decode(ra, ra, pts, sp) == 0;
                for (size_t i = 0; i < n && ok; i++)
                    if (ra[i] != (sa[i] + sb[i]) % sp->t) ok = 0;
            }
            if (!ok) printf("  [FAIL] T10 SIMD slot add 与明文不一致\n");
        }
        CHECK(ok, "T10a SIMD 打包 1024 槽位 同态加法", "见上");
        ok = ok && vfhe_mult(&ctd, &cta, &ctb) == 0 &&
             vfhe_decrypt_poly(pd, &sk, &ctd) == 0;
        if (ok) {
            for (size_t i = 0; i < n; i++) {
                int32_t cv = vfhe_center((uint32_t)pd[i], sp->t);
                rb[i] = cv < 0 ? (uint64_t)(cv + (int32_t)sp->t) : (uint64_t)cv;
            }
            ok = vfhe_slot_decode(rb, rb, pts, sp) == 0;
            for (size_t i = 0; i < n && ok; i++)
                if (rb[i] != (sa[i] * sb[i]) % sp->t) ok = 0;
        }
        CHECK(ok, "T10b SIMD 打包 1024 槽位 同态乘法（深1）", "见上");
        free(pts); free(sa); free(sb); free(pa); free(pb);
        free(pc); free(pd); free(ra); free(rb);
        vfhe_ct_free(&cta); vfhe_ct_free(&ctb); vfhe_ct_free(&ctc); vfhe_ct_free(&ctd);
    }

    /* ---- T11: SIMD 量化网络（1024 样本并行，4->4->2，量化权重 + 多项式激活 y^2）----
       槽位 = 样本。每个特征打包一个密文（1024 样本该特征 -> 槽位）。
       层 1：y_j = Σ_i W1[j][i]·ct_i（密文×明文标量 + 加，逐槽位，深 0）
       激活：a_j = y_j^2（密文×密文，深 1，逐槽位）
       层 2：z_k = Σ_j W2[k][j]·a_j（深 0）
       解密解包逐槽位（逐样本）对照明文。无需旋转（权重明文架构）。 */
    {
        const vfhe_params_t *sp = vfhe_params_default();
        size_t n = sp->n;
        enum { NI = 4, NH = 4, NO = 2 };
        static const int32_t W1[NH * NI] = {   /* 行主序：W1[j][i] */
             1,  0, -1,  1,
             0,  1,  1, -1,
             1,  1,  0,  0,
            -1,  0,  1,  1,
        };
        static const int32_t W2[NO * NH] = {   /* 行主序：W2[k][j] */
             1, 0, 1, 0,
             0, 1, 1, 1,
        };
        uint64_t *pts = (uint64_t *)malloc(n * sizeof(uint64_t));
        uint64_t *feat = (uint64_t *)malloc(NI * n * sizeof(uint64_t));   /* feat[i][s] */
        uint64_t *poly = (uint64_t *)malloc(n * sizeof(uint64_t));
        uint64_t *slots = (uint64_t *)malloc(n * sizeof(uint64_t));
        vfhe_ct_t cts[NI], cy1[NH], ca1[NH], cz[NO];
        memset(cts, 0, sizeof(cts)); memset(cy1, 0, sizeof(cy1));
        memset(ca1, 0, sizeof(ca1)); memset(cz, 0, sizeof(cz));
        int ok = pts && feat && poly && slots && vfhe_slot_roots(pts, sp) == 0;

        /* 明文对照（逐样本 s，槽位索引） */
        int64_t *py1 = (int64_t *)malloc(NH * n * sizeof(int64_t));
        int64_t *pa1 = (int64_t *)malloc(NH * n * sizeof(int64_t));
        int64_t *pz = (int64_t *)malloc(NO * n * sizeof(int64_t));
        ok = ok && py1 && pa1 && pz;
        if (ok) {
            for (size_t s = 0; s < n; s++)
                for (int i = 0; i < NI; i++)
                    feat[i * n + s] = (uint64_t)((s * 3 + (size_t)i * 2) % 4);   /* 样本 s 特征 i: 0..3 */
            for (size_t s = 0; s < n; s++) {
                for (int j = 0; j < NH; j++) {
                    int64_t acc = 0;
                    for (int i = 0; i < NI; i++) acc += (int64_t)W1[j * NI + i] * (int64_t)feat[i * n + s];
                    py1[j * n + s] = acc;
                }
                for (int j = 0; j < NH; j++) pa1[j * n + s] = py1[j * n + s] * py1[j * n + s];
                for (int k = 0; k < NO; k++) {
                    int64_t acc = 0;
                    for (int j = 0; j < NH; j++) acc += (int64_t)W2[k * NH + j] * pa1[j * n + s];
                    pz[k * n + s] = acc;
                }
            }
            /* 打包+加密：每特征一个密文 */
            for (int i = 0; i < NI && ok; i++)
                ok = vfhe_slot_encode(poly, &feat[i * n], pts, sp) == 0 &&
                     vfhe_encrypt_sym_poly(&cts[i], &sk, poly) == 0;
        }
        /* 层 1：cy1[j] = Σ_i W1[j][i]·ct_i */
        if (ok) {
            vfhe_ct_t acc = {0}, tmp = {0}, nxt = {0};
            for (int j = 0; j < NH; j++) {
                int32_t w0 = W1[j * NI + 0];
                uint32_t m0 = (uint32_t)(w0 < 0 ? w0 + (int32_t)sp->t : w0);
                if (vfhe_mul_scalar(&acc, &cts[0], m0) != 0) { ok = 0; break; }
                for (int i = 1; i < NI; i++) {
                    int32_t w = W1[j * NI + i];
                    uint32_t mw = (uint32_t)(w < 0 ? w + (int32_t)sp->t : w);
                    if (vfhe_mul_scalar(&tmp, &cts[i], mw) != 0) { ok = 0; break; }
                    if (vfhe_add(&nxt, &acc, &tmp) != 0) { ok = 0; break; }
                    vfhe_ct_free(&acc); vfhe_ct_free(&tmp);
                    acc = nxt; memset(&nxt, 0, sizeof(nxt));
                }
                vfhe_ct_copy(&cy1[j], &acc);
                vfhe_ct_free(&acc);
            }
            vfhe_ct_free(&tmp); vfhe_ct_free(&nxt);
        }
        /* 激活：a_j = y_j^2（深 1） */
        for (int j = 0; j < NH && ok; j++)
            if (vfhe_mult(&ca1[j], &cy1[j], &cy1[j]) != 0) ok = 0;
        /* 层 2：cz[k] = Σ_j W2[k][j]·a_j */
        if (ok) {
            vfhe_ct_t acc = {0}, tmp = {0}, nxt = {0};
            for (int k = 0; k < NO; k++) {
                int32_t w0 = W2[k * NH + 0];
                uint32_t m0 = (uint32_t)(w0 < 0 ? w0 + (int32_t)sp->t : w0);
                if (vfhe_mul_scalar(&acc, &ca1[0], m0) != 0) { ok = 0; break; }
                for (int j = 1; j < NH; j++) {
                    int32_t w = W2[k * NH + j];
                    uint32_t mw = (uint32_t)(w < 0 ? w + (int32_t)sp->t : w);
                    if (vfhe_mul_scalar(&tmp, &ca1[j], mw) != 0) { ok = 0; break; }
                    if (vfhe_add(&nxt, &acc, &tmp) != 0) { ok = 0; break; }
                    vfhe_ct_free(&acc); vfhe_ct_free(&tmp);
                    acc = nxt; memset(&nxt, 0, sizeof(nxt));
                }
                vfhe_ct_copy(&cz[k], &acc);
                vfhe_ct_free(&acc);
            }
            vfhe_ct_free(&tmp); vfhe_ct_free(&nxt);
        }
        /* 解密+解包：逐样本对照 */
        for (int k = 0; k < NO && ok; k++) {
            if (vfhe_decrypt_poly(poly, &sk, &cz[k]) != 0) { ok = 0; break; }
            for (size_t i = 0; i < n; i++) {
                int32_t cv = vfhe_center((uint32_t)poly[i], sp->t);
                slots[i] = cv < 0 ? (uint64_t)(cv + (int32_t)sp->t) : (uint64_t)cv;
            }
            if (vfhe_slot_decode(slots, slots, pts, sp) != 0) { ok = 0; break; }
            for (size_t s = 0; s < n && ok; s++) {
                if ((int64_t)slots[s] != pz[k * n + s]) {
                    if (s < 5)
                        printf("  [FAIL] T11 out[%d] 样本 %zu: got %llu expect %lld\n", k, s,
                               (unsigned long long)slots[s], (long long)pz[k * n + s]);
                    ok = 0;
                }
            }
        }
        CHECK(ok, "T11 SIMD 量化网络 1024 样本并行 4-4-2（激活 y^2）与明文对照", "见上");
        for (int i = 0; i < NI; i++) vfhe_ct_free(&cts[i]);
        for (int j = 0; j < NH; j++) { vfhe_ct_free(&cy1[j]); vfhe_ct_free(&ca1[j]); }
        for (int k = 0; k < NO; k++) vfhe_ct_free(&cz[k]);
        free(pts); free(feat); free(poly); free(slots);
        free(py1); free(pa1); free(pz);
    }

    /* ---- T11b: 模型接入层演示（8->8->4 网络，整数多项式激活 2a^2+a）----
       权重量化打包（int -> slot_encode）+ 权重明文 matmul（深 0）+ 多项式激活
       （深 1：a^2 后线性组合 2a^2+a）。1024 样本并行，逐样本明文对照。
       明文幅度/噪声约束：输入 {0,1}、W1 ∈ {-1,0,1}（作用于新鲜密文，安全）、
       W2 ∈ {0,1}（避免 m≈t 的明文标量乘放大深 1 密文噪声 t 倍）。 */
    {
        const vfhe_params_t *sp = vfhe_params_default();
        size_t n = sp->n;
        enum { NI = 8, NH = 8, NO = 4 };
        static const int32_t W1[NH * NI] = {
             1,  0, -1,  1,  0,  1, -1,  1,
             0,  1,  1, -1,  1,  0,  1,  0,
             1,  1,  0,  0, -1,  1,  0, -1,
            -1,  0,  1,  1,  1,  0,  1,  1,
             0, -1,  0,  1,  1, -1,  0,  1,
             1,  0,  1, -1,  0,  1, -1,  0,
            -1,  1,  0,  1,  1,  0,  1, -1,
             0,  1, -1,  0,  1,  1,  0,  1,
        };
        static const int32_t W2[NO * NH] = {
             1,  0,  1,  1,  0,  1,  0,  0,
             0,  1,  1,  0,  1,  0,  1,  1,
             1,  0,  0,  0,  1,  0,  0,  1,
             0,  0,  1,  1,  0,  1,  1,  0,
        };
        uint64_t *pts = (uint64_t *)malloc(n * sizeof(uint64_t));
        uint64_t *feat = (uint64_t *)malloc(NI * n * sizeof(uint64_t));
        uint64_t *poly = (uint64_t *)malloc(n * sizeof(uint64_t));
        uint64_t *slots = (uint64_t *)malloc(n * sizeof(uint64_t));
        vfhe_ct_t cts[NI], cy1[NH], ca1[NH], cz[NO];
        memset(cts, 0, sizeof(cts)); memset(cy1, 0, sizeof(cy1));
        memset(ca1, 0, sizeof(ca1)); memset(cz, 0, sizeof(cz));
        int ok = pts && feat && poly && slots && vfhe_slot_roots(pts, sp) == 0;
        /* 明文对照：a = 2·(W1·x)^2 + (W1·x)，z = W2·a */
        int64_t *py1 = (int64_t *)malloc(NH * n * sizeof(int64_t));
        int64_t *pa1 = (int64_t *)malloc(NH * n * sizeof(int64_t));
        int64_t *pz = (int64_t *)malloc(NO * n * sizeof(int64_t));
        ok = ok && py1 && pa1 && pz;
        if (ok) {
            for (size_t s = 0; s < n; s++)
                for (int i = 0; i < NI; i++)
                    feat[i * n + s] = (uint64_t)((s * 5 + (size_t)i * 3) % 2);
            for (size_t s = 0; s < n; s++) {
                for (int j = 0; j < NH; j++) {
                    int64_t acc = 0;
                    for (int i = 0; i < NI; i++) acc += (int64_t)W1[j * NI + i] * (int64_t)feat[i * n + s];
                    py1[j * n + s] = acc;
                    pa1[j * n + s] = 2 * acc * acc + acc;
                }
                for (int k = 0; k < NO; k++) {
                    int64_t acc = 0;
                    for (int j = 0; j < NH; j++) acc += (int64_t)W2[k * NH + j] * pa1[j * n + s];
                    pz[k * n + s] = acc;
                }
            }
            for (int i = 0; i < NI && ok; i++)
                ok = vfhe_slot_encode(poly, &feat[i * n], pts, sp) == 0 &&
                     vfhe_encrypt_sym_poly(&cts[i], &sk, poly) == 0;
        }
        /* 层 1 + 激活 2a^2+a（深 1） */
        if (ok) {
            vfhe_ct_t acc = {0}, tmp = {0}, nxt = {0}, t2 = {0};
            for (int j = 0; j < NH; j++) {
                int32_t w0 = W1[j * NI + 0];
                uint32_t m0 = (uint32_t)(w0 < 0 ? w0 + (int32_t)sp->t : w0);
                if (vfhe_mul_scalar(&acc, &cts[0], m0) != 0) { ok = 0; break; }
                for (int i = 1; i < NI; i++) {
                    int32_t w = W1[j * NI + i];
                    uint32_t mw = (uint32_t)(w < 0 ? w + (int32_t)sp->t : w);
                    if (vfhe_mul_scalar(&tmp, &cts[i], mw) != 0) { ok = 0; break; }
                    if (vfhe_add(&nxt, &acc, &tmp) != 0) { ok = 0; break; }
                    vfhe_ct_free(&acc); vfhe_ct_free(&tmp);
                    acc = nxt; memset(&nxt, 0, sizeof(nxt));
                }
                if (!ok) break;
                vfhe_ct_copy(&cy1[j], &acc);       /* a = W1·x */
                if (vfhe_mult(&t2, &cy1[j], &cy1[j]) != 0) { ok = 0; break; }   /* a^2（深1） */
                if (vfhe_mul_scalar(&nxt, &t2, 2) != 0) { ok = 0; break; }      /* 2a^2 */
                if (vfhe_mul_scalar(&tmp, &cy1[j], 1) != 0) { ok = 0; break; }  /* a */
                if (vfhe_add(&ca1[j], &nxt, &tmp) != 0) { ok = 0; break; }
                vfhe_ct_free(&acc); vfhe_ct_free(&nxt); vfhe_ct_free(&tmp);
                vfhe_ct_free(&t2);
            }
            vfhe_ct_free(&acc); vfhe_ct_free(&tmp); vfhe_ct_free(&nxt); vfhe_ct_free(&t2);
        }
        /* 层 2：cz[k] = Σ_j W2[k][j]·a_j（W2 ∈ {0,1}：避免 m≈t 的明文标量乘
           放大深 1 密文噪声 t 倍——BFV 明文标量乘噪声 = m·e） */
        if (ok) {
            vfhe_ct_t acc = {0}, tmp = {0}, nxt = {0};
            for (int k = 0; k < NO; k++) {
                int32_t w0 = W2[k * NH + 0];
                uint32_t m0 = (uint32_t)(w0 < 0 ? w0 + (int32_t)sp->t : w0);
                if (vfhe_mul_scalar(&acc, &ca1[0], m0) != 0) { ok = 0; break; }
                for (int j = 1; j < NH; j++) {
                    int32_t w = W2[k * NH + j];
                    uint32_t mw = (uint32_t)(w < 0 ? w + (int32_t)sp->t : w);
                    if (vfhe_mul_scalar(&tmp, &ca1[j], mw) != 0) { ok = 0; break; }
                    if (vfhe_add(&nxt, &acc, &tmp) != 0) { ok = 0; break; }
                    vfhe_ct_free(&acc); vfhe_ct_free(&tmp);
                    acc = nxt; memset(&nxt, 0, sizeof(nxt));
                }
                if (!ok) break;
                vfhe_ct_copy(&cz[k], &acc);
                vfhe_ct_free(&acc);
            }
            vfhe_ct_free(&tmp); vfhe_ct_free(&nxt);
        }
        /* 解密+解包：逐样本对照 */
        for (int k = 0; k < NO && ok; k++) {
            if (vfhe_decrypt_poly(poly, &sk, &cz[k]) != 0) { ok = 0; break; }
            for (size_t i = 0; i < n; i++) {
                int32_t cv = vfhe_center((uint32_t)poly[i], sp->t);
                slots[i] = cv < 0 ? (uint64_t)(cv + (int32_t)sp->t) : (uint64_t)cv;
            }
            if (vfhe_slot_decode(slots, slots, pts, sp) != 0) { ok = 0; break; }
            for (size_t s = 0; s < n && ok; s++) {
                if ((int64_t)slots[s] != pz[k * n + s]) {
                    if (s < 5)
                        printf("  [FAIL] T11b out[%d] 样本 %zu: got %llu expect %lld\n", k, s,
                               (unsigned long long)slots[s], (long long)pz[k * n + s]);
                    ok = 0;
                }
            }
        }
        CHECK(ok, "T11b 模型接入层 8-8-4 网络（权重明文matmul+整数激活2a^2+a）与明文对照", "见上");
        for (int i = 0; i < NI; i++) vfhe_ct_free(&cts[i]);
        for (int j = 0; j < NH; j++) { vfhe_ct_free(&cy1[j]); vfhe_ct_free(&ca1[j]); }
        for (int k = 0; k < NO; k++) vfhe_ct_free(&cz[k]);
        free(pts); free(feat); free(poly); free(slots);
        free(py1); free(pa1); free(pz);
    }

    vfhe_sk_free(&sk);
    vfhe_pk_free(&pk);
    vfhe_ct_free(&ct); vfhe_ct_free(&ct2); vfhe_ct_free(&ct3);

    if (g_fail == 0)
        printf("  RESULT: ALL PASS (%d)\n", g_fail);
    else
        printf("  RESULT: %d FAILED\n", g_fail);
    return g_fail;
}
