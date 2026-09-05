/**
 * vllm_ntt.c - 负循环 NTT 实现（见 vllm_ntt.h）
 *
 * 数学（n=2^k，q 素数且 2n | q-1）：
 *   1. 找 primitive 2n-th root ψ：ψ^n ≡ -1 (mod q)，等价 ψ = g^((q-1)/(2n))，
 *      其中 g 是 q 的 primitive root（需 q-1 完整因子分解验证）。
 *   2. fwd：A[i] = Σ_j a[j]·ψ^j·(ψ²)^(ij)。先逐系数乘 ψ^j，再对 w=ψ² 做
 *      标准 DIT radix-2 NTT（bit-reversal + 蝶形）。
 *   3. inv：同一蝶形结构用 w^{-1}，结果逐系数乘 ψ^{-j}·n^{-1}。
 *   4. negacyclic 卷积：ntt(fwd(a)·fwd(b)) 逐点乘后 iNTT = a*b mod (x^n+1)。
 *
 * 验证（axiom_modal_transition_conservation_005）：自测做 iNTT(NTT(x))=x
 * 位级往返，并与教科书 O(n²) negacyclic 逐系数对照。
 */
#include "vllm_ntt.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

/* ================================================================
 * 模算术（60-bit 素数：a+b < 2^61 无溢出，用条件减法；Goldilocks 64-bit
 * a+b 可溢出 2^64，用 __int128）。mulmod 统一 __int128 % q（位级一致）。
 * ================================================================ */
static inline uint64_t ntt_addmod(uint64_t a, uint64_t b, uint64_t q) {
    if (q < (1ull << 63)) {          /* 60-bit 素数：无 2^64 溢出 */
        uint64_t r = a + b;
        if (r >= q) r -= q;
        return r;
    }
    return (uint64_t)(((unsigned __int128)a + b) % q);   /* Goldilocks */
}
static inline uint64_t ntt_submod(uint64_t a, uint64_t b, uint64_t q) {
    if (q < (1ull << 63)) {          /* 60-bit 素数：无 2^64 溢出 */
        return a >= b ? a - b : a + q - b;
    }
    __int128 r = (__int128)a - b;
    if (r < 0) r += q;
    return (uint64_t)r;
}
static inline uint64_t ntt_mulmod(uint64_t a, uint64_t b, uint64_t q) {
    return (uint64_t)((unsigned __int128)a * b % q);
}
/* Montgomery（R=2^64）：蝶形内 mulmod 的常数加速（~5 周期 vs __int128 除法 ~25-40）。
 * 整个 NTT 在 Montgomery 域运行：toMont（乘 r2）→ 蝶形 mont_mul → fromMont（乘 1）。
 * 表（wpow/psi_pow/n_inv）在 ntt_ctx_init 时转 Montgomery 域（乘 r2）。位级一致：
 * mont_mul 是精确整数运算，结果与 __int128 % q 逐位相同。 */
static inline uint64_t ntt_mont_mul(uint64_t a, uint64_t b, uint64_t q,
                                    uint64_t qinv) {
    unsigned __int128 t = (unsigned __int128)a * b;
    uint64_t m = (uint64_t)t * qinv;                   /* 低 64 位（回绕即 mod 2^64） */
    unsigned __int128 u = t + (unsigned __int128)m * q;
    uint64_t hi = (uint64_t)(u >> 64);
    if (hi >= q) hi -= q;
    return hi;
}
/* -q^{-1} mod 2^64（Newton 迭代，6 轮收敛到 2^64 精度） */
static uint64_t ntt_qinv_u64(uint64_t q) {
    uint64_t x = 1;
    for (int i = 0; i < 6; i++) x *= 2 - q * x;
    return (uint64_t)(0 - x);
}
static uint64_t ntt_powmod(uint64_t b, uint64_t e, uint64_t m) {
    uint64_t r = 1;
    b %= m;
    while (e) {
        if (e & 1) r = ntt_mulmod(r, b, m);
        b = ntt_mulmod(b, b, m);
        e >>= 1;
    }
    return r;
}

/* ================================================================
 * Miller-Rabin（< 2^64 确定性基组 {2,3,5,7,11,13,17}）
 * ================================================================ */
static int ntt_is_prime(uint64_t n) {
    if (n < 2) return 0;
    for (uint64_t p = 2; p * p <= n && p < 1000; p++)
        if (n % p == 0) return 0;
    uint64_t d = n - 1, s = 0;
    while ((d & 1) == 0) { d >>= 1; s++; }
    static const uint64_t bases[] = {2, 3, 5, 7, 11, 13, 17};
    for (size_t i = 0; i < sizeof(bases) / sizeof(bases[0]); i++) {
        uint64_t a = bases[i] % n;
        if (a == 0) continue;
        uint64_t x = ntt_powmod(a, d, n);
        if (x == 1 || x == n - 1) continue;
        int comp = 1;
        for (uint64_t r = 1; r < s; r++) {
            x = ntt_mulmod(x, x, n);
            if (x == n - 1) { comp = 0; break; }
        }
        if (comp) return 0;
    }
    return 1;
}

/* q-1 试除分解到 bound，返回质因子表。剩余 >1 的部分若为素数则计入；
 * 若为合数（bound 内不可分解）返回 0 表示失败（primitive root 判定不可靠）。
 * 优化：每步除尽后立即对剩余做素性判定，素数则提前返回（避免大素数
 * 或无小因子合数走满 2^25 试除）。 */
static uint32_t ntt_factor_distinct(uint64_t m, uint64_t *facs, uint32_t maxf,
                                    uint64_t bound) {
    uint32_t nf = 0;
    if (m == 1) return 0;
    if (ntt_is_prime(m)) { facs[nf++] = m; return nf; }
    if ((m & 1) == 0) {                    /* 先分离 2 因子（q-1 恒为偶数） */
        facs[nf++] = 2;
        while ((m & 1) == 0) m >>= 1;
        if (m > 1 && ntt_is_prime(m)) { facs[nf++] = m; return nf; }
    }
    for (uint64_t d = 3; d <= bound && d * d <= m; d += 2) {
        if (m % d == 0) {
            facs[nf++] = d;
            while (m % d == 0) m /= d;
            if (m > 1 && ntt_is_prime(m)) { facs[nf++] = m; return nf; }
            if (nf >= maxf) break;
        }
    }
    if (m > 1) {
        if (!ntt_is_prime(m)) return 0;   /* 大合数剩余：分解失败 */
        facs[nf++] = m;
    }
    return nf;
}

/* 找 primitive root g（g^((q-1)/f) != 1 对所有 prime f | q-1） */
static uint64_t ntt_primitive_root(uint64_t q) {
    uint64_t facs[16];
    uint32_t nf = ntt_factor_distinct(q - 1, facs, 16, 1ull << 25);
    if (nf == 0) return 0;   /* q-1 分解失败（含 bound 外大合数因子） */
    for (uint64_t g = 2; g < q; g++) {
        int ok = 1;
        for (uint32_t i = 0; i < nf; i++) {
            if (ntt_powmod(g, (q - 1) / facs[i], q) == 1) { ok = 0; break; }
        }
        if (ok) return g;
    }
    return 0;
}

/* ================================================================
 * 初始化
 * ================================================================ */
int ntt_ctx_init(ntt_ctx_t *ctx, uint64_t q, uint32_t n) {
    if (!ctx || n < 2 || (n & (n - 1)) != 0) return -1;
    if ((q - 1) % (2 * (uint64_t)n) != 0) return -1;   /* 2n-th root 存在性 */
    if (!ntt_is_prime(q)) return -1;
    memset(ctx, 0, sizeof(*ctx));
    ctx->q = q;
    ctx->n = n;
    uint64_t g = ntt_primitive_root(q);
    if (!g) return -1;
    ctx->psi = ntt_powmod(g, (q - 1) / (2 * (uint64_t)n), q);   /* ψ^2048 = 1 */
    if (ntt_powmod(ctx->psi, n, q) != q - 1) return -1;          /* ψ^n = -1 */
    ctx->n_inv = ntt_powmod(n, q - 2, q);
    ctx->qinv = ntt_qinv_u64(q);
    ctx->r2 = ntt_powmod(2, 128, q);   /* R² mod q = 2^128 mod q（toMont 因子） */
    ctx->use_mont = (q < (1ull << 63)); /* 64-bit Goldilocks 的 t+m·q 可溢 2^128 */

    uint32_t lg = 0;
    for (uint32_t t = n; t > 1; t >>= 1) lg++;

    ctx->wlen_fwd = (uint64_t *)calloc(lg + 1, sizeof(uint64_t));
    ctx->wlen_inv = (uint64_t *)calloc(lg + 1, sizeof(uint64_t));
    ctx->psi_pow = (uint64_t *)calloc(n, sizeof(uint64_t));
    ctx->inv_psi_pow = (uint64_t *)calloc(n, sizeof(uint64_t));
    ctx->rev = (uint32_t *)calloc(n, sizeof(uint32_t));
    ctx->wpow_fwd = (uint64_t *)calloc((size_t)(lg + 1) * (n / 2), sizeof(uint64_t));
    ctx->wpow_inv = (uint64_t *)calloc((size_t)(lg + 1) * (n / 2), sizeof(uint64_t));
    ctx->bufA = (uint64_t *)calloc(n, sizeof(uint64_t));
    ctx->bufB = (uint64_t *)calloc(n, sizeof(uint64_t));
    if (!ctx->wlen_fwd || !ctx->wlen_inv || !ctx->psi_pow ||
        !ctx->inv_psi_pow || !ctx->rev || !ctx->wpow_fwd || !ctx->wpow_inv ||
        !ctx->bufA || !ctx->bufB) {
        ntt_ctx_free(ctx);
        return -1;
    }

    uint64_t w = ntt_mulmod(ctx->psi, ctx->psi, q);   /* w = ψ²（n-th root） */
    uint64_t w_inv = ntt_powmod(w, q - 2, q);
    for (uint32_t len = 2, s = 1; len <= n; len <<= 1, s++) {
        ctx->wlen_fwd[s] = ntt_powmod(w, n / len, q);
        ctx->wlen_inv[s] = ntt_powmod(w_inv, n / len, q);
        /* 预计算每 stage 蝶形旋转因子 w^j（查表替代蝶形内连乘） */
        uint64_t *wf = &ctx->wpow_fwd[(size_t)(s - 1) * (n / 2)];
        uint64_t *wi = &ctx->wpow_inv[(size_t)(s - 1) * (n / 2)];
        uint64_t tf = 1, ti = 1;
        uint32_t half = len >> 1;
        for (uint32_t j = 0; j < half; j++) {
            wf[j] = tf; wi[j] = ti;
            tf = ntt_mulmod(tf, ctx->wlen_fwd[s], q);
            ti = ntt_mulmod(ti, ctx->wlen_inv[s], q);
        }
    }
    uint64_t p = 1, ip = 1, psi_inv = ntt_powmod(ctx->psi, q - 2, q);
    for (uint32_t j = 0; j < n; j++) {
        ctx->psi_pow[j] = p;
        ctx->inv_psi_pow[j] = ip;
        p = ntt_mulmod(p, ctx->psi, q);
        ip = ntt_mulmod(ip, psi_inv, q);
    }
    /* 蝶形旋转因子/预乘因子表转 Montgomery 域：mont_mul(x, r2) = x·R²·R⁻¹ = x·R
     * （必须用 mont_mul 而非普通乘法：普通 mulmod 得 x·R² 在"R² 域"）。
     * 仅 60-bit 素数（use_mont）转换；Goldilocks 走 __int128 路径表保持普通域。 */
    if (ctx->use_mont) {
        uint64_t r2 = ctx->r2, qi = ctx->qinv;
        for (uint32_t len = 2, s = 1; len <= n; len <<= 1, s++) {
            uint64_t *wf = &ctx->wpow_fwd[(size_t)(s - 1) * (n / 2)];
            uint64_t *wi = &ctx->wpow_inv[(size_t)(s - 1) * (n / 2)];
            for (uint32_t j = 0; j < (len >> 1); j++) {
                wf[j] = ntt_mont_mul(wf[j], r2, q, qi);
                wi[j] = ntt_mont_mul(wi[j], r2, q, qi);
            }
        }
        for (uint32_t j = 0; j < n; j++) {
            ctx->psi_pow[j] = ntt_mont_mul(ctx->psi_pow[j], r2, q, qi);
            ctx->inv_psi_pow[j] = ntt_mont_mul(ctx->inv_psi_pow[j], r2, q, qi);
        }
        ctx->n_inv = ntt_mont_mul(ctx->n_inv, r2, q, qi);
    }
    /* 位反转表 */
    for (uint32_t i = 0; i < n; i++) {
        uint32_t r = 0, x = i;
        for (uint32_t b = 0; b < lg; b++) { r = (r << 1) | (x & 1); x >>= 1; }
        ctx->rev[i] = r;
    }
    return 0;
}

void ntt_ctx_free(ntt_ctx_t *ctx) {
    if (!ctx) return;
    free(ctx->wlen_fwd); free(ctx->wlen_inv);
    free(ctx->psi_pow); free(ctx->inv_psi_pow);
    free(ctx->rev); free(ctx->wpow_fwd); free(ctx->wpow_inv);
    free(ctx->bufA); free(ctx->bufB);
    memset(ctx, 0, sizeof(*ctx));
}

/* ================================================================
 * 正/逆变换（就地，DIT radix-2，旋转因子查表 + OpenMP 并行）
 * ================================================================ */
void ntt_negacyclic_fwd(uint64_t *a, const ntt_ctx_t *ctx) {
    uint32_t n = ctx->n;
    uint64_t q = ctx->q, qi = ctx->qinv, r2 = ctx->r2;
    int um = ctx->use_mont;
    /* toMont + 预乘 ψ^j（ψ^j 表在 Montgomery 域：mont_mul 保持 Mont 域；
     * Goldilocks 走普通 __int128 路径，表为普通域） */
    for (uint32_t j = 0; j < n; j++) {
        a[j] = um ? ntt_mont_mul(a[j], r2, q, qi) : ntt_mulmod(a[j], ctx->psi_pow[j], q);
        if (um) a[j] = ntt_mont_mul(a[j], ctx->psi_pow[j], q, qi);
    }
    /* bit-reversal */
    for (uint32_t i = 0; i < n; i++) {
        uint32_t r = ctx->rev[i];
        if (r > i) { uint64_t t = a[i]; a[i] = a[r]; a[r] = t; }
    }
    /* 蝶形（Montgomery 域表 w^j·R 或普通域；串行最优，OpenMP 已回退） */
    for (uint32_t len = 2, s = 1; len <= n; len <<= 1, s++) {
        const uint64_t *wpow = &ctx->wpow_fwd[(size_t)(s - 1) * (n / 2)];
        uint32_t half = len >> 1;
        for (uint32_t i = 0; i < n; i += len) {
            for (uint32_t j = 0; j < half; j++) {
                uint64_t u = a[i + j];
                uint64_t v = um ? ntt_mont_mul(a[i + j + half], wpow[j], q, qi)
                                : ntt_mulmod(a[i + j + half], wpow[j], q);
                a[i + j] = ntt_addmod(u, v, q);
                a[i + j + half] = ntt_submod(u, v, q);
            }
        }
    }
    /* fromMont（乘 1）→ 普通域输出 */
    if (um)
        for (uint32_t j = 0; j < n; j++) a[j] = ntt_mont_mul(a[j], 1, q, qi);
}

void ntt_negacyclic_inv(uint64_t *a, const ntt_ctx_t *ctx) {
    uint32_t n = ctx->n;
    uint64_t q = ctx->q, qi = ctx->qinv, r2 = ctx->r2;
    int um = ctx->use_mont;
    /* toMont（输入普通域） */
    if (um)
        for (uint32_t j = 0; j < n; j++) a[j] = ntt_mont_mul(a[j], r2, q, qi);
    /* bit-reversal */
    for (uint32_t i = 0; i < n; i++) {
        uint32_t r = ctx->rev[i];
        if (r > i) { uint64_t t = a[i]; a[i] = a[r]; a[r] = t; }
    }
    /* 蝶形（w^{-1} 表；串行最优） */
    for (uint32_t len = 2, s = 1; len <= n; len <<= 1, s++) {
        const uint64_t *wpow = &ctx->wpow_inv[(size_t)(s - 1) * (n / 2)];
        uint32_t half = len >> 1;
        for (uint32_t i = 0; i < n; i += len) {
            for (uint32_t j = 0; j < half; j++) {
                uint64_t u = a[i + j];
                uint64_t v = um ? ntt_mont_mul(a[i + j + half], wpow[j], q, qi)
                                : ntt_mulmod(a[i + j + half], wpow[j], q);
                a[i + j] = ntt_addmod(u, v, q);
                a[i + j + half] = ntt_submod(u, v, q);
            }
        }
    }
    /* 后乘 ψ^{-j}·n^{-1}（Montgomery 域表），最后 fromMont */
    for (uint32_t j = 0; j < n; j++) {
        if (um) {
            a[j] = ntt_mont_mul(a[j], ctx->inv_psi_pow[j], q, qi);
            a[j] = ntt_mont_mul(a[j], ctx->n_inv, q, qi);
            a[j] = ntt_mont_mul(a[j], 1, q, qi);
        } else {
            a[j] = ntt_mulmod(a[j], ctx->inv_psi_pow[j], q);
            a[j] = ntt_mulmod(a[j], ctx->n_inv, q);
        }
    }
}

void ntt_negacyclic_mul(uint64_t *c, const uint64_t *a, const uint64_t *b,
                        const ntt_ctx_t *ctx) {
    uint32_t n = ctx->n;
    uint64_t q = ctx->q;
    /* 调用级局部 scratch（VLA）：ctx->bufA/bufB 是共享工作区，
       多线程并发调用 ntt_negacyclic_mul（如块级并行下的 key-switch）会相互踩踏。
       单线程位级等价。 */
    uint64_t ba[2 * n], *bb = ba + n;
    memcpy(ba, a, n * sizeof(uint64_t));
    memcpy(bb, b, n * sizeof(uint64_t));
    ntt_negacyclic_fwd(ba, ctx);
    ntt_negacyclic_fwd(bb, ctx);
    for (uint32_t i = 0; i < n; i++)
        ba[i] = ntt_mulmod(ba[i], bb[i], q);
    ntt_negacyclic_inv(ba, ctx);
    memcpy(c, ba, n * sizeof(uint64_t));
}

/* ================================================================
 * 自测（T13）：往返 + 位级对照 O(n²) + 性能计时
 * ================================================================ */
static void ntt_naive_mul(uint64_t *c, const uint64_t *a, const uint64_t *b,
                          uint32_t n, uint64_t q) {
    memset(c, 0, n * sizeof(uint64_t));
    for (uint32_t i = 0; i < n; i++) {
        if (!a[i]) continue;
        for (uint32_t j = 0; j < n; j++) {
            if (!b[j]) continue;
            uint64_t t = ntt_mulmod(a[i], b[j], q);
            uint32_t k = i + j;
            if (k < n) c[k] = ntt_addmod(c[k], t, q);
            else       c[k - n] = ntt_submod(c[k - n], t, q);
        }
    }
}

static uint64_t ntt_rng_state = 0x123456789ABCDEFull;
static uint64_t ntt_rng_u64(void) {
    uint64_t x = ntt_rng_state;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    ntt_rng_state = x;
    return x * 0x2545F4914F6CDD1Dull;
}

int ntt_self_test(void) {
    int fails = 0;
    printf("=== vLLM-NTT Self Test ===\n");
    /* 测试素数：Goldilocks 与 CKKS 首个 60-bit 素数 */
    static const uint64_t test_qs[] = {
        0xFFFFFFFF00000001ull,                /* Goldilocks 2^64-2^32+1 */
        1152921504606830593ull,               /* CKKS q0 */
    };
    uint32_t n = 1024;
    for (size_t tq = 0; tq < sizeof(test_qs) / sizeof(test_qs[0]); tq++) {
        uint64_t q = test_qs[tq];
        ntt_ctx_t ctx;
        if (ntt_ctx_init(&ctx, q, n) != 0) {
            printf("  [FAIL] ctx init q=%llu\n", (unsigned long long)q);
            fails++;
            continue;
        }
        printf("  q=%llu n=%u psi=%llu\n", (unsigned long long)q, n,
               (unsigned long long)ctx.psi);

        /* T13a：iNTT(NTT(x)) = x 位级往返（公理 ntt_bijection） */
        uint64_t *x = (uint64_t *)malloc(n * 8);
        for (uint32_t i = 0; i < n; i++) x[i] = ntt_rng_u64() % q;
        uint64_t *orig = (uint64_t *)malloc(n * 8);
        memcpy(orig, x, n * 8);
        ntt_negacyclic_fwd(x, &ctx);
        ntt_negacyclic_inv(x, &ctx);
        int bad = 0;
        for (uint32_t i = 0; i < n && bad < 4; i++)
            if (x[i] != orig[i]) { printf("  [roundtrip diff] i=%u\n", i); bad++; }
        printf("  T13a roundtrip: %s\n", bad ? "FAIL" : "PASS");
        fails += bad ? 1 : 0;

        /* T13b：NTT 乘 vs 教科书 negacyclic 位级对照 */
        uint64_t *a = (uint64_t *)malloc(n * 8), *b = (uint64_t *)malloc(n * 8);
        uint64_t *c1 = (uint64_t *)malloc(n * 8), *c2 = (uint64_t *)malloc(n * 8);
        for (uint32_t i = 0; i < n; i++) { a[i] = ntt_rng_u64() % q; b[i] = ntt_rng_u64() % q; }
        ntt_negacyclic_mul(c1, a, b, &ctx);
        ntt_naive_mul(c2, a, b, n, q);
        bad = 0;
        for (uint32_t i = 0; i < n && bad < 4; i++)
            if (c1[i] != c2[i]) { printf("  [ntt vs naive diff] i=%u\n", i); bad++; }
        printf("  T13b ntt vs O(n^2): %s\n", bad ? "FAIL" : "PASS");
        fails += bad ? 1 : 0;

        /* 性能计时：NTT vs O(n²)（各 20 次取均值） */
        clock_t t0 = clock();
        for (int r = 0; r < 20; r++) ntt_negacyclic_mul(c1, a, b, &ctx);
        clock_t t1 = clock();
        for (int r = 0; r < 20; r++) ntt_naive_mul(c2, a, b, n, q);
        clock_t t2 = clock();
        double t_ntt = (double)(t1 - t0) / CLOCKS_PER_SEC / 20 * 1e6;
        double t_nai = (double)(t2 - t1) / CLOCKS_PER_SEC / 20 * 1e6;
        printf("  perf: NTT=%.1fus  O(n^2)=%.1fus  加速=%.0fx\n",
               t_ntt, t_nai, t_nai / (t_ntt > 0 ? t_ntt : 1e-9));

        free(x); free(orig); free(a); free(b); free(c1); free(c2);
        ntt_ctx_free(&ctx);
    }

    /* 大 n 档（n=2048）：OpenMP 并行路径（n>=2048 才启用）的正确性 + 性能。
     * 动态找 60-bit 素数 q ≡ 1 mod 2n=4096，且 q-1 = 2^12 · p（p 素数）：
     * 保证 q-1 在 bound 内可瞬间分解（先除 2^12 后剩余 p 为素数），
     * 避免大合数剩余导致 primitive root 判定不可靠。 */
    {
        uint32_t n2 = 2048;
        uint64_t mod = 2 * (uint64_t)n2;                 /* 4096 = 2^12 */
        ntt_ctx_t ctx2;
        int inited = 0;
        /* 起点对齐到 x ≡ 1 mod 4096（2^60 ≡ 0 mod 4096，故 2^60-4095 ≡ 1） */
        for (uint64_t x = (1ull << 60) - 4095u; x >= (1ull << 59); x -= mod) {
            if (!ntt_is_prime(x)) continue;
            uint64_t p = (x - 1) >> 12;                  /* q-1 = 2^12·p */
            if ((p & 1) == 0 || !ntt_is_prime(p)) continue;
            if (ntt_ctx_init(&ctx2, x, n2) == 0) { inited = 1; break; }
        }
        if (!inited) { printf("  [FAIL] big-n prime search n=%u\n", n2); fails++; }
        else {
            printf("  q=%llu n=%u psi=%llu (OpenMP 并行路径)\n",
                   (unsigned long long)ctx2.q, n2,
                   (unsigned long long)ctx2.psi);
            /* T13a：位级往返 */
            uint64_t *x2 = (uint64_t *)malloc(n2 * 8);
            uint64_t *o2 = (uint64_t *)malloc(n2 * 8);
            for (uint32_t i = 0; i < n2; i++) x2[i] = ntt_rng_u64() % ctx2.q;
            memcpy(o2, x2, n2 * 8);
            ntt_negacyclic_fwd(x2, &ctx2);
            ntt_negacyclic_inv(x2, &ctx2);
            int bad = 0;
            for (uint32_t i = 0; i < n2 && bad < 4; i++)
                if (x2[i] != o2[i]) { printf("  [big-n roundtrip diff] i=%u\n", i); bad++; }
            printf("  T13a roundtrip (n=%u): %s\n", n2, bad ? "FAIL" : "PASS");
            fails += bad ? 1 : 0;
            /* 性能：NTT 乘 20 次取均值（n=2048） */
            uint64_t *a2 = (uint64_t *)malloc(n2 * 8), *b2 = (uint64_t *)malloc(n2 * 8);
            uint64_t *c2o = (uint64_t *)malloc(n2 * 8);
            for (uint32_t i = 0; i < n2; i++) { a2[i] = ntt_rng_u64() % ctx2.q; b2[i] = ntt_rng_u64() % ctx2.q; }
            clock_t t0 = clock();
            for (int r = 0; r < 20; r++) ntt_negacyclic_mul(c2o, a2, b2, &ctx2);
            clock_t t1 = clock();
            double t_ntt2 = (double)(t1 - t0) / CLOCKS_PER_SEC / 20 * 1e6;
            printf("  perf: NTT(n=%u)=%.1fus\n", n2, t_ntt2);
            free(x2); free(o2); free(a2); free(b2); free(c2o);
            ntt_ctx_free(&ctx2);
        }
    }
    printf("  RESULT: %s (%d)\n", fails ? "FAILED" : "ALL PASS", fails);
    return fails;
}
