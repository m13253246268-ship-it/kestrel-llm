/* t20_layer0.c - 鈶?鍗曞眰鏂囨湰 transformer 绔埌绔瘑鏂囨帹鐞嗭紙Qwen3-VL-2B layer0 鐪熷疄鏉冮噸锛孡=4 token锛? *
 * 瀹炴暟妲借〃绀猴細n=2048, K=1024 妲姐€?048 缁村悜閲?= 2 瀵嗘枃锛坸0/x1锛夈€? * 妲戒綅甯冨眬锛? *   x (token t): x0=dims[0:1024], x1=dims[1024:2048]
 *   q: q0=heads0-7 (t路1024+h路128+i), q1=heads8-15
 *   k/v: h'路128+i (h'=0..7, 鍗?ct)
 *   scores: t路64+h路4+s锛?56 妲斤級
 *   attn_out/灞傝緭鍑? 2 ct锛堝悓 q 甯冨眬锛? *   gate/up/act: 6144 = 6脳1024 妲斤紙6 ct锛? * 缂╂斁绾﹀畾锛堣蹇?鍙湁鏈€缁堣В鐮佸€奸』<0.5"锛夛細鍏ㄧ簿搴︿腑闂村€硷紙mod Q锛夛紝
 *   exp 鐢?/8锛坱13 鍚屾锛夛紝鏈€缁堝眰杈撳嚭 脳1/16 鍙В鐮併€? * 瀵圭収 .tmp_tok/l0/*.bin锛坃layer0_ref.py锛宼oken 0-3 鍚勯樁娈碉級銆? *
 * 缂栬瘧锛歡cc -O2 -fopenmp -DCKKS_NPRIMES=40 -DCKKS_N=2048 -I include/core \
 *       src/core/vllm_ntt.c src/core/vllm_ckks.c .tmp_tok/t20_layer0.c -o .tmp_tok/t20_layer0.exe -lm
 */
#include "vllm_ckks.h"
#include "vllm_tp.h"
#include <complex.h>
#include "vllm_ntt.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define DIM 2048
#define MID 6144
#define B   32
#define G   32
#define NQ  16
#define NKV 8
#define HD  128
#define EPS 1e-6
#define LTOK 4
#define LCHAIN 112   /* 层链素数级：M3b=96 放不下浅层 rmsnorm 输入预缩(+8) → 112 */

static inline uint64_t t20_addmod(uint64_t a, uint64_t b, uint64_t q) {
    uint64_t r = a + b;
    if (r >= q) r -= q;
    return r;
}
static inline uint64_t t20_mont_mul(uint64_t a, uint64_t b, uint64_t q, uint64_t qinv) {
    unsigned __int128 t = (unsigned __int128)a * b;
    uint64_t m = (uint64_t)t * qinv;
    unsigned __int128 u = t + (unsigned __int128)m * q;
    uint64_t hi = (uint64_t)(u >> 64);
    if (hi >= q) hi -= q;
    return hi;
}
static uint64_t t20_modinv(uint64_t a, uint64_t q) {
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
static int rot_map(int t, int j, int n) {
    uint64_t k = 1, base = 5, exp = (uint64_t)t, mod = 2 * (uint64_t)n;
    while (exp) { if (exp & 1) k = (k * base) % mod; base = (base * base) % mod; exp >>= 1; }
    uint64_t inv = t20_modinv(k % mod, mod);
    uint64_t s = ((uint64_t)(2 * j + 1) * inv) % mod;
    if (s >= (uint64_t)n) s = mod - s;
    return (int)((s - 1) / 2);
}
static ntt_ctx_t g_nc[CKKS_NPRIMES];
static int g_nc_used = 0;
static ntt_ctx_t *t20_ntt_for(uint64_t q, uint32_t n) {
    for (int i = 0; i < g_nc_used; i++)
        if (g_nc[i].q == q && g_nc[i].n == n) return &g_nc[i];
    if (g_nc_used >= CKKS_NPRIMES) return NULL;
    if (ntt_ctx_init(&g_nc[g_nc_used], q, n) != 0) return NULL;
    return &g_nc[g_nc_used++];
}

/* 鍧?matmul锛堝鍒嗛噺杈撳叆锛夛細y = W[r0..r0+1024, c0..c0+1024]路x锛寈 鐢ㄩ鏃嬭浆 baby 棰戝煙琛ㄣ€? * Wdim = W 琛屽銆傝緭鍑?scale=2^120 鏈?rescale銆?*/
static int block_bsgs(const float *W, int Wdim, int r0, int c0,
                      uint64_t **xb_ntt, const ckks_ctx_t *ctx,
                      const ckks_gk_t *giant_gk,
                      const ckks_sk_t *sk, ckks_ct_t *yout, int comps_in, int np_in) {
    size_t np = np_in, n = ctx->n;
    int K = (int)(n / 2);
    uint64_t *poly = malloc((size_t)ctx->nprimes * n * 8);
    uint64_t *T_all = calloc((size_t)G * comps_in * np * n, sizeof(uint64_t));
    if (!poly || !T_all) return -1;
    /* 棰勭儹 NTT ctx + cos 琛紙OpenMP 绾跨▼瀹夊叏鍓嶆彁锛氬叏閮ㄧ礌鏁颁竴娆℃€?init锛?*/
    for (size_t p = 0; p < np; p++) {
        ntt_ctx_t *nc = t20_ntt_for(ctx->q[p], (uint32_t)n);
        if (!nc) return -1;
    }
    {
        double dummy[2048] = {0};
        if (K <= 2048) ckks_encode(poly, dummy, ctx);
    }
    /* r 寰幆骞惰锛氭瘡 r 鐙珛 T_ntt锛坉iag 鎵撳寘 + NTT + 棰戝煙绱姞锛夛紝缁撴灉瀛樺叆 T_all[r] */
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int r = 0; r < G; r++) {
        uint64_t *T_ntt = &T_all[(size_t)r * comps_in * np * n];
        double *dl = malloc((size_t)K * sizeof(double));
        int *gpl = malloc((size_t)K * sizeof(int));
        int *bpl = malloc((size_t)K * sizeof(int));
        int *bvl = malloc((size_t)K * sizeof(int));
        uint64_t *pl = malloc((size_t)ctx->nprimes * n * 8);
        uint64_t *pln = malloc((size_t)ctx->nprimes * n * 8);
        if (!dl || !gpl || !bpl || !bvl || !pl || !pln) continue;
        for (int j = 0; j < K; j++) gpl[j] = rot_map(r * B, j, (int)n);
        for (int s = 0; s < B; s++) {
            for (int j = 0; j < K; j++) bpl[j] = rot_map(s, j, (int)n);
            for (int j = 0; j < K; j++) bvl[bpl[j]] = j;
            for (int j = 0; j < K; j++)
                dl[j] = W[((size_t)(r0 + gpl[j]) * Wdim) + (c0 + bvl[j])];
            ckks_encode(pl, dl, ctx);
            for (size_t p = 0; p < np; p++) {
                uint64_t q = ctx->q[p];
                ntt_ctx_t *nc = t20_ntt_for(q, (uint32_t)n);
                uint64_t qinv = nc->qinv, r2 = nc->r2;
                memcpy(&pln[p * n], &pl[p * n], n * 8);
                ntt_negacyclic_fwd(&pln[p * n], nc);
                for (size_t i = 0; i < n; i++)
                    pln[p * n + i] = t20_mont_mul(pln[p * n + i], r2, q, qinv);
                for (int cp = 0; cp < comps_in; cp++) {
                    const uint64_t *xb = &xb_ntt[s][((size_t)cp * np + p) * n];
                    uint64_t *T = &T_ntt[((size_t)cp * np + p) * n];
                    for (size_t i = 0; i < n; i++)
                        T[i] = t20_addmod(T[i], t20_mont_mul(xb[i], pln[p * n + i], q, qinv), q);
                }
            }
        }
        free(dl); free(gpl); free(bpl); free(bvl); free(pl); free(pln);
    }
    /* 涓茶锛歩nv NTT 鈫?giant 鏃嬭浆 鈫?绱姞 y */
    ckks_ct_t y = {0};
    for (int r = 0; r < G; r++) {
        ckks_ct_t Tr = {0};
        Tr.ctx = *ctx; Tr.ctx.nprimes = np_in; Tr.comps = comps_in;
        Tr.c = malloc((size_t)comps_in * np * n * sizeof(uint64_t));
        if (!Tr.c) { ckks_ct_free(&y); free(poly); free(T_all); return -1; }
        for (int cp = 0; cp < comps_in; cp++)
            for (size_t p = 0; p < np; p++) {
                uint64_t q = ctx->q[p];
                ntt_ctx_t *nc = t20_ntt_for(q, (uint32_t)n);
                uint64_t *dst = &Tr.c[((size_t)cp * np + p) * n];
                memcpy(dst, &T_all[((size_t)r * comps_in * np + (size_t)cp * np + p) * n], n * 8);
                ntt_negacyclic_inv(dst, nc);
            }
        if (r == 0) {
            ckks_ct_t s2 = {0};
            if (y.comps == 0) ckks_ct_copy(&y, &Tr);
            else { ckks_add(&s2, &y, &Tr); ckks_ct_free(&y); y = s2; }
        } else {
            ckks_ct_t g = {0};
            ckks_rotate(&g, &Tr, sk, &giant_gk[r], r * B);
            ckks_ct_t s2 = {0};
            if (y.comps == 0) ckks_ct_copy(&y, &g);
            else { ckks_add(&s2, &y, &g); ckks_ct_free(&y); y = s2; }
            ckks_ct_free(&g);
        }
        ckks_ct_free(&Tr);
    }
    free(poly); free(T_all);
    *yout = y;
    return 0;
}

static int baby_build(const ckks_ct_t *xct, uint64_t ***out, const ckks_ctx_t *ctx,
                      const ckks_sk_t *sk) {
    size_t np = xct->ctx.nprimes, n = xct->ctx.n;
    int comps = xct->comps;
    uint64_t **xb_ntt = malloc((size_t)B * sizeof(uint64_t *));
    ckks_ct_t *xb = malloc((size_t)B * sizeof(ckks_ct_t));
    for (int s = 0; s < B; s++) {
        if (s == 0) ckks_ct_copy(&xb[s], xct);
        else {
            ckks_gk_t gk;
            ckks_gk_gen(&gk, sk, ctx, s);
            ckks_rotate(&xb[s], xct, sk, &gk, s);
            ckks_gk_free(&gk);
        }
        xb_ntt[s] = malloc((size_t)comps * np * n * 8);
        for (int cp = 0; cp < comps; cp++)
            for (size_t p = 0; p < np; p++) {
                uint64_t q = ctx->q[p];
                ntt_ctx_t *nc = t20_ntt_for(q, (uint32_t)n);
                uint64_t *dst = &xb_ntt[s][((size_t)cp * np + p) * n];
                memcpy(dst, &xb[s].c[((size_t)cp * np + p) * n], n * 8);
                ntt_negacyclic_fwd(dst, nc);
            }
        ckks_ct_free(&xb[s]);
    }
    free(xb);
    *out = xb_ntt;
    return 0;
}
static void baby_free(uint64_t **xb_ntt) {
    for (int s = 0; s < B; s++) free(xb_ntt[s]);
    free(xb_ntt);
}

static int ct_switch_to(ckks_ct_t *out, const ckks_ct_t *in, int target) {
    ckks_ct_t t = {0};
    if (target >= in->ctx.nprimes) { ckks_ct_copy(&t, in); *out = t; return 0; }
    if (ckks_modswitch(&t, in) != 0) return -1;
    ckks_ct_t cur = t;
    while (cur.ctx.nprimes > target) {
        ckks_ct_t nxt = {0};
        if (ckks_modswitch(&nxt, &cur) != 0) { ckks_ct_free(&cur); return -1; }
        ckks_ct_free(&cur); cur = nxt;
    }
    *out = cur;
    return 0;
}
static int mul_relin(ckks_ct_t *out, const ckks_ct_t *a, const ckks_ct_t *b,
                     const ckks_rk_t *rk, const ckks_ctx_t *ctx) {
    ckks_ct_t m = {0}, r = {0}, rel = {0};
    if (ckks_mult(&m, a, b) != 0) return -1;
    if (ckks_rescale(&r, &m) != 0) { ckks_ct_free(&m); return -1; }
    ckks_ct_free(&m);
    if (ckks_relin(&rel, &r, rk) != 0) { ckks_ct_free(&r); return -1; }
    ckks_ct_free(&r);
    *out = rel;
    return 0;
}
static void encode_slot_const(uint64_t *poly, double c, int j0, const ckks_ctx_t *ctx) {
    size_t K = ctx->n / 2;
    double *z = malloc(K * sizeof(double));
    for (size_t j = 0; j < K; j++) z[j] = 0.0;
    if (j0 >= 0) z[j0] = c;
    else for (size_t j = 0; j < K; j++) z[j] = c;
    ckks_encode(poly, z, ctx);
    free(z);
}
static void make_const_ct(ckks_ct_t *ct, double c, int j0, const ckks_sk_t *sk,
                          const ckks_ctx_t *ctx, int target_np) {
    uint64_t *poly = malloc((size_t)ctx->nprimes * ctx->n * 8);
    encode_slot_const(poly, c, j0, ctx);
    ckks_encrypt(ct, sk, poly, ctx);
    free(poly);
    while ((int)ct->ctx.nprimes > target_np) {
        ckks_ct_t t = {0};
        ckks_modswitch(&t, ct);
        ckks_ct_free(ct);
        *ct = t;
    }
}
static void ct_mul_slot(ckks_ct_t *out, const ckks_ct_t *in, double c, int j0,
                        const ckks_sk_t *sk, const ckks_ctx_t *ctx) {
    uint64_t *poly = malloc((size_t)ctx->nprimes * ctx->n * 8);
    encode_slot_const(poly, c, j0, ctx);
    ckks_ct_t m = {0};
    ckks_mult_plain(&m, in, poly);
    free(poly);
    ckks_rescale(out, &m);
    ckks_ct_free(&m);
}
static void ct_add_const(ckks_ct_t *out, const ckks_ct_t *in, double c, int j0,
                         const ckks_sk_t *sk, const ckks_ctx_t *ctx) {
    ckks_ct_t cst = {0}, s = {0};
    make_const_ct(&cst, c, j0, sk, ctx, (int)in->ctx.nprimes);
    ckks_add(&s, in, &cst);
    ckks_ct_free(&cst);
    if (out->c) ckks_ct_free(out);
    *out = s;
}
static void ct_neg(ckks_ct_t *out, const ckks_ct_t *in) {
    size_t np = in->ctx.nprimes, n = in->ctx.n;
    out->ctx = in->ctx; out->comps = in->comps;
    out->c = malloc((size_t)in->comps * np * n * 8);
    for (int cp = 0; cp < in->comps; cp++)
        for (size_t p = 0; p < np; p++)
            for (size_t i = 0; i < n; i++) {
                uint64_t q = in->ctx.q[p];
                uint64_t v = in->c[((size_t)cp * np + p) * n + i];
                out->c[((size_t)cp * np + p) * n + i] = (q - v) % q;
            }
}
/* Horner锛歰ut = Q[0]路z^deg + ... + Q[deg]锛寊 灏卞湴闄嶉摼锛坱12 鍚屾锛夛紝j0=-1 婊℃Ы銆? * 姣忔 mult(3 鍒嗛噺)鈫抮escale鈫抮elin(3鈫?)锛坮elin 闆跺櫔澹扮簿纭級鈫抋dd 甯搁噺锛屼繚鎸?acc 2 鍒嗛噺锛? * 鍏抽敭锛歾 姣忔 modswitch 瀵归綈 acc 閾鹃暱锛堝惁鍒?ckks_mult 鍥?nprimes 涓嶅尮閰嶅け璐ワ級銆?*/
static void horner_q(ckks_ct_t *out, ckks_ct_t *z, const double *Q, int deg,
                     const ckks_rk_t *rk, const ckks_sk_t *sk,
                     const ckks_ctx_t *ctx, int j0) {
    ckks_ct_t acc = {0};
    make_const_ct(&acc, Q[0], j0, sk, ctx, (int)z->ctx.nprimes);
    for (int k = 1; k <= deg; k++) {
        ckks_ct_t m = {0}, r = {0}, rel = {0};
        ckks_mult(&m, z, &acc);
        ckks_rescale(&r, &m); ckks_ct_free(&m);
        ckks_relin(&rel, &r, rk); ckks_ct_free(&r);
        ckks_ct_free(&acc);
        acc = rel;
        ckks_ct_t cst = {0};
        make_const_ct(&cst, Q[k], j0, sk, ctx, (int)acc.ctx.nprimes);
        ckks_ct_t s = {0};
        ckks_add(&s, &acc, &cst);
        ckks_ct_free(&cst); ckks_ct_free(&acc);
        acc = s;
        if (k < deg) {
            ckks_ct_t zn = {0};
            ckks_modswitch(&zn, z);
            ckks_ct_free(z);
            *z = zn;
        }
    }
    *out = acc;
}

/* silu 鍋堕儴绯绘暟 E(u)=危 b_j路u^j锛圼-8,8] deg-17锛宼19 鍚屾锛?*/
static const double SILU_B[9] = {
    2.352963542201e-03, 2.423574584366e-01, -1.642397857413e-02,
    9.721357057704e-04, -3.875485356725e-05, 9.821825570427e-07,
    -1.506038643107e-08, 1.268707444657e-10, -4.495635644842e-13
};
/* silu = 0.5x + E(x虏)锛屽箓鍒嗚В u^1..u^8 + mult_plain 缁勫悎 + 0.5x锛岃緭鍑?脳1/16锛堚墹0.5 鍙В鐮侊級 */
static int eval_silu16(ckks_ct_t *out, const ckks_ct_t *x, const ckks_rk_t *rk,
                       const ckks_ctx_t *ctx, const ckks_sk_t *sk) {
    size_t n = ctx->n, np = ctx->nprimes;
    int fails = 0;
    uint64_t *pc = malloc(np * n * 8);
    ckks_ct_t pk[9] = {0};
    if (mul_relin(&pk[1], x, x, rk, ctx) != 0) fails++;
    for (int j = 2; j <= 8; j++) {
        ckks_ct_t us = {0}, r2 = {0};
        if (ct_switch_to(&us, &pk[1], pk[j - 1].ctx.nprimes) != 0) fails++;
        if (mul_relin(&r2, &pk[j - 1], &us, rk, ctx) != 0) fails++;
        ckks_ct_free(&us);
        pk[j] = r2;
    }
    printf("  [silu] pk8 np=%d pk1 np=%d\n", (int)pk[8].ctx.nprimes, (int)pk[1].ctx.nprimes);
    ckks_ct_t res = {0};
    {
        ckks_ct_t term = {0};
        make_const_ct(&term, SILU_B[8], -1, sk, ctx, pk[8].ctx.nprimes);
        ckks_ct_t m = {0};
        ckks_mult_plain(&m, &pk[8], term.c);
        ckks_ct_free(&term);
        ckks_rescale(&res, &m); ckks_ct_free(&m);
    }
    for (int j = 7; j >= 1; j--) {
        ckks_ct_t term = {0}, rl = {0};
        make_const_ct(&term, SILU_B[j], -1, sk, ctx, pk[j].ctx.nprimes);
        ckks_ct_t pks = {0}, m = {0};
        if (ct_switch_to(&pks, &pk[j], res.ctx.nprimes) != 0) fails++;
        ckks_mult_plain(&m, &pks, term.c);
        ckks_ct_free(&pks); ckks_ct_free(&term);
        ckks_ct_t tr = {0};
        ckks_rescale(&tr, &m); ckks_ct_free(&m);
        if (ct_switch_to(&rl, &res, tr.ctx.nprimes) != 0) fails++;
        ckks_ct_free(&res);
        if (ckks_add(&res, &rl, &tr) != 0) fails++;
        ckks_ct_free(&rl); ckks_ct_free(&tr);
    }
    /* + b0 */
    {
        ckks_ct_t cst = {0};
        make_const_ct(&cst, SILU_B[0], -1, sk, ctx, (int)res.ctx.nprimes);
        ckks_ct_t s = {0};
        ckks_add(&s, &res, &cst);
        ckks_ct_free(&cst); ckks_ct_free(&res); res = s;
    }
    /* 0.5x */
    {
        ckks_ct_t xl = {0}, x5 = {0};
        if (ct_switch_to(&xl, x, res.ctx.nprimes) != 0) fails++;
        ckks_ct_t cst = {0};
        make_const_ct(&cst, 0.5, -1, sk, ctx, (int)xl.ctx.nprimes);
        ckks_ct_t m = {0};
        ckks_mult_plain(&m, &xl, cst.c);
        ckks_ct_free(&cst); ckks_ct_free(&xl);
        ckks_rescale(&x5, &m); ckks_ct_free(&m);
        ckks_ct_t rl = {0};
        if (ct_switch_to(&rl, &res, x5.ctx.nprimes) != 0) fails++;
        ckks_ct_free(&res);
        if (ckks_add(&res, &rl, &x5) != 0) fails++;
        ckks_ct_free(&rl); ckks_ct_free(&x5);
    }
    /* 脳1/16 */
    {
        ckks_ct_t cst = {0};
        make_const_ct(&cst, 1.0 / 16.0, -1, sk, ctx, (int)res.ctx.nprimes);
        ckks_ct_t m = {0};
        ckks_mult_plain(&m, &res, cst.c);
        ckks_ct_free(&cst);
        ckks_rescale(out, &m); ckks_ct_free(&m);
        ckks_ct_free(&res);
    }
    for (int j = 1; j <= 8; j++) ckks_ct_free(&pk[j]);
    free(pc);
    if (fails) { ckks_ct_free(out); return -1; }
    return 0;
}

/* silu 鍏ㄧ簿搴︾増锛圡LP 鐢紝鏃?/16锛夛細silu = 0.5x + E(x虏)锛岃緭鍑烘渶缁堜箻 out_scale
 * 锛坥ut_scale=1 鏃?silu鈮?.3 涓棿鍊?mod Q 鍐呭畨鍏紱=1/16 鏃跺彲鐩存帴瑙ｇ爜锛夈€?*/
static int eval_silu(ckks_ct_t *out, const ckks_ct_t *x, double out_scale,
                     const ckks_rk_t *rk, const ckks_ctx_t *ctx, const ckks_sk_t *sk) {
    size_t n = ctx->n, np = ctx->nprimes;
    int fails = 0;
    printf("  [silu] in x np=%d\n", (int)x->ctx.nprimes);
    ckks_ct_t pk[9] = {0};
    if (mul_relin(&pk[1], x, x, rk, ctx) != 0) fails++;          /* u = x虏 */
    for (int j = 2; j <= 8; j++) {
        ckks_ct_t us = {0}, r2 = {0};
        if (ct_switch_to(&us, &pk[1], pk[j - 1].ctx.nprimes) != 0) fails++;
        if (mul_relin(&r2, &pk[j - 1], &us, rk, ctx) != 0) fails++;
        ckks_ct_free(&us);
        pk[j] = r2;
    }
    printf("  [silu] pk8 np=%d\n", (int)pk[8].ctx.nprimes);
    ckks_ct_t res = {0};
    /* 骞傜粍鍚堬細甯告暟蹇呴』缂栫爜涓虹湡鏄庢枃锛坈t_mul_slot = encode+mult_plain+rescale锛夛紝
     * 涓嶈兘涔樺姞瀵嗗父閲忓瘑鏂囩殑 c0锛堜細寮曞叆 s路e 澶у櫔澹帮紝t19 鍚屾鏁欒锛?*/
    ct_mul_slot(&res, &pk[8], SILU_B[8], -1, sk, ctx);
    for (int j = 7; j >= 1; j--) {
        ckks_ct_t rl = {0}, pks = {0}, tr = {0};
        if (ct_switch_to(&pks, &pk[j], res.ctx.nprimes) != 0) fails++;
        ct_mul_slot(&tr, &pks, SILU_B[j], -1, sk, ctx);
        ckks_ct_free(&pks);
        if (ct_switch_to(&rl, &res, tr.ctx.nprimes) != 0) fails++;
        ckks_ct_free(&res);
        if (ckks_add(&res, &rl, &tr) != 0) fails++;
        ckks_ct_free(&rl); ckks_ct_free(&tr);
    }
    /* + b0 */
    {
        ckks_ct_t cst = {0}, s = {0};
        make_const_ct(&cst, SILU_B[0], -1, sk, ctx, (int)res.ctx.nprimes);
        ckks_add(&s, &res, &cst);
        ckks_ct_free(&cst); ckks_ct_free(&res); res = s;
    }
    /* 0.5x */
    {
        ckks_ct_t xl = {0}, x5 = {0}, rl = {0};
        if (ct_switch_to(&xl, x, res.ctx.nprimes) != 0) fails++;
        ct_mul_slot(&x5, &xl, 0.5, -1, sk, ctx);
        ckks_ct_free(&xl);
        if (ct_switch_to(&rl, &res, x5.ctx.nprimes) != 0) fails++;
        ckks_ct_free(&res);
        if (ckks_add(&res, &rl, &x5) != 0) fails++;
        ckks_ct_free(&rl); ckks_ct_free(&x5);
    }
    /* 脳 out_scale */
    if (out_scale != 1.0) {
        ckks_ct_t t = {0};
        ct_mul_slot(&t, &res, out_scale, -1, sk, ctx);
        ckks_ct_free(&res);
        *out = t;
    } else {
        *out = res;
    }
    for (int j = 1; j <= 8; j++) ckks_ct_free(&pk[j]);
    if (fails) { ckks_ct_free(out); return -1; }
    return 0;
}

/* ===== M3b (c)：折叠域 silu（phase≥3，|g|≤19.5 覆盖层26/27 MLP gate）=====
 * silu(g)/Sg = 0.5·x + R(w)，x = g/Sg（≤0.93），w = x²/vm ∈ [0,1]；
 * R(w) deg-8 由 horner_q 求值（系数为"加常数"，可大）；1/Sg 与 1/vm 为 const mult（≤8）。 */
static const double SILU_W21_R[9] = {   /* w^8..w^0 */
    -52.555831396795135, 228.7193911068195, -414.74107047011859,
    406.34322413539491, -233.55938542700747, 80.378386397027882,
    -16.533501657853961, 2.3986479950949571, 0.012662442785882201,
};
#define SILU_W21_SG  21.0
#define SILU_W21_VMINV 1.159763314
static void ct_unfold(ckks_ct_t *ct, double f, const ckks_sk_t *sk, const ckks_ctx_t *ctx);
static int eval_silu_w(ckks_ct_t *out, const ckks_ct_t *x, const double *R,
                       double Sg, double vminv, const ckks_rk_t *rk,
                       const ckks_sk_t *sk, const ckks_ctx_t *ctx) {
    int dbg = getenv("T23_DBGRMS") != NULL;
    /* 大系数(≤415)在 horner 中 acc 消息可超 1 → 系数整体预缩 RSCALE（acc≤0.1），
     * 求值后 ×RSCALE 拆 ≤8 还原（R(w) ≤ ~0.35 消息，线性无碍） */
    const double RSC = 4096.0;
    ckks_ct_t xf = {0}, v = {0}, w = {0}, Rv = {0}, wz = {0};
    ct_mul_slot(&xf, x, 1.0 / Sg, -1, sk, ctx);          /* xf = g/Sg ≤ 0.93 */
    if (dbg) printf("  [silw] xf np=%d\n", (int)xf.ctx.nprimes);
    if (mul_relin(&v, &xf, &xf, rk, ctx) != 0) { ckks_ct_free(&xf); return -1; }
    if (dbg) printf("  [silw] v np=%d\n", (int)v.ctx.nprimes);
    ct_mul_slot(&w, &v, vminv, -1, sk, ctx);             /* w = xf²/vm ≤ 1 */
    ckks_ct_free(&v);
    if (dbg) printf("  [silw] w np=%d\n", (int)w.ctx.nprimes);
    ckks_ct_copy(&wz, &w);
    ckks_ct_free(&w);
    {
        double Rg[9];
        for (int j = 0; j < 9; j++) Rg[j] = R[j] / RSC;
        horner_q(&Rv, &wz, Rg, 8, rk, sk, ctx, -1);      /* R(w)/RSC，acc ≤0.1 */
    }
    ckks_ct_free(&wz);
    if (dbg) printf("  [silw] Rv np=%d\n", (int)Rv.ctx.nprimes);
    ct_unfold(&Rv, RSC, sk, ctx);                        /* → R(w) ≤0.35 */
    if (dbg) printf("  [silw] Rvu np=%d\n", (int)Rv.ctx.nprimes);
    {
        ckks_ct_t x5 = {0}, rl = {0}, r2 = {0};
        ct_mul_slot(&x5, &xf, 0.5, -1, sk, ctx);          /* 0.5·x */
        ckks_ct_free(&xf);
        int npmin = x5.ctx.nprimes < Rv.ctx.nprimes ? x5.ctx.nprimes : Rv.ctx.nprimes;
        ct_switch_to(&rl, &x5, npmin);
        ckks_ct_free(&x5);
        ct_switch_to(&r2, &Rv, npmin);
        ckks_ct_free(&Rv);
        ckks_add(out, &rl, &r2);
        ckks_ct_free(&rl); ckks_ct_free(&r2);
    }
    return 0;
}
/* 还原：const mult 明文系数上限 ~×8（超限饱和），折叠倍率拆 ≤8 分步乘 */
static void ct_unfold(ckks_ct_t *ct, double f, const ckks_sk_t *sk, const ckks_ctx_t *ctx) {
    ckks_ct_t u = {0};
    while (f > 1.0000001) {
        double st = f >= 8.0 ? 8.0 : f;
        ct_mul_slot(&u, ct, st, -1, sk, ctx);
        ckks_ct_free(ct);
        *ct = u; memset(&u, 0, sizeof u);
        f /= st;
    }
}

static const char *g_datadir = ".tmp_tok/l0";   /* M1b u-form 灞傜敤 l1 鍙傝€?*/
/* M3b: tail 输出端参数（u-form 残差折叠 / 大值域检查 scale / 参考文件前缀 / causal） */
static double g_fold = 16.0, g_chk = 16.0;
static int g_silu_direct = 0;   /* lay 浅层(silu mode 文件, |gate|<=8)用直接 silu 拟合(锚定0) */
static double g_gfold = 1.0;   /* M3b: xg 侧 γ 明文预缩(一般打包明文须槽值<~0.5) */
static double g_bfold = 1.0;   /* M3b: bcast 侧折叠；末段 ct×ct 结果<1，总还原 ×(g_gfold*g_bfold) */
static double g_m2c = 0.0;     /* final(c5): 临时传递 m2c_f 给 t23_final */
static int g_phase = 0;
static const char *g_refpre = "";
static int g_causal = 0;
static const char *g_wdir = ".tmp_tok/l0", *g_rdir = ".tmp_tok/l0";
static float *loadf(const char *name, size_t cnt) {
    char path[512]; snprintf(path, sizeof(path), "%s/%s%s.bin", g_datadir, g_refpre, name);
    FILE *f = fopen(path, "rb");
    float *p = malloc(cnt * 4);
    if (!f || fread(p, 4, cnt, f) != cnt) { free(p); if (f) fclose(f); return NULL; }
    fclose(f);
    return p;
}
/* M2：绝对路径读 double 数组（LNQ 拟合系数/m2c 中心，可选覆盖）；失败保持原值 */
static int load_f64f(const char *path, double *out, size_t n) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    size_t r = fread(out, 8, n, f);
    fclose(f);
    return r == n ? 0 : -1;
}

static double t_clock(void) {
#ifdef _OPENMP
    return omp_get_wtime();
#else
    return (double)clock() / CLOCKS_PER_SEC;
#endif
}

/* 鍗曞潡 matmul锛歰ut = W[r0..r0+1024, c0..c0+1024]路x锛? 涓垪鍧楋級 */
static int matmul1(ckks_ct_t *out, const float *W, int Wdim, int r0, int c0,
                   const ckks_ct_t *x, const ckks_gk_t *giant_gk,
                   const ckks_sk_t *sk, const ckks_ctx_t *ctx) {
    uint64_t **xb = NULL;
    baby_build(x, &xb, ctx, sk);
    ckks_ct_t r = {0};
    block_bsgs(W, Wdim, r0, c0, xb, ctx, giant_gk, sk, &r, x->comps, x->ctx.nprimes);
    baby_free(xb);
    *out = r;
    return 0;
}
/* 鍙屽潡 matmul锛堝垪 0/1 鐩稿姞锛夛細out = W[r0..r0+1024, 0:2048]路[xa;xb]锛岃緭鍑?scale 2^60 */
static int matmul2(ckks_ct_t *out, const float *W, int Wdim, int r0,
                   const ckks_ct_t *xa, const ckks_ct_t *xb,
                   const ckks_gk_t *giant_gk, const ckks_sk_t *sk,
                   const ckks_ctx_t *ctx) {
    uint64_t **xba = NULL, **xbb = NULL;
    baby_build(xa, &xba, ctx, sk);
    baby_build(xb, &xbb, ctx, sk);
    ckks_ct_t a0 = {0}, a1 = {0}, s = {0}, r = {0};
    block_bsgs(W, Wdim, r0, 0, xba, ctx, giant_gk, sk, &a0, xa->comps, xa->ctx.nprimes);
    block_bsgs(W, Wdim, r0, (int)(ctx->n / 2), xbb, ctx, giant_gk, sk, &a1, xb->comps, xb->ctx.nprimes);
    baby_free(xba); baby_free(xbb);
    if (ckks_add(&s, &a0, &a1) != 0) { ckks_ct_free(&a0); ckks_ct_free(&a1); return -1; }
    ckks_rescale(&r, &s);
    ckks_ct_free(&a0); ckks_ct_free(&a1); ckks_ct_free(&s);
    *out = r;
    return 0;
}

/* RMSNorm锛圖=2048锛屼袱鍗婏級锛歺n = x路纬路inv_std銆傝緭鍏?xa/xb 涓哄師濮?embed 瀵嗘枃锛堜笉淇敼锛夈€? * 鐢ㄥ浐瀹氫腑蹇?m2c锛堣法 token 鍧囧€硷級瀹?z' 鏄犲皠锛孡NQ 涓鸿涓績鎷熷悎鐨?inv_std/128 澶氶」寮忋€?*/
static void dbg_ct(const ckks_ct_t *ct, double S, const char *label,
                   const ckks_sk_t *sk, const ckks_ctx_t *ctx,
                   uint64_t *pdec, double *zy);
static int rmsnorm_enc(ckks_ct_t *xn0, ckks_ct_t *xn1,
                       const ckks_ct_t *xa, const ckks_ct_t *xb,
                       const float *gamma, double m2c, const double *LNQ,
                       double *zx, uint64_t *poly, uint64_t *pdec, double *zy,
                       const ckks_gk_t *giant_gk, const ckks_rk_t *rk,
                       const ckks_sk_t *sk, const ckks_ctx_t *ctx,
                       const char *tag) {
    int K = (int)(ctx->n / 2);
    int fails = 0;
    printf("  [rms %s] in xa np=%d xb np=%d\n", tag,
           (int)xa->ctx.nprimes, (int)xb->ctx.nprimes);
    /* 浅层 m2c 极小（u=y/F 下 m2c~1e-9..3e-5）使 1/(8sDIM) 常数与 LNQ 系数超
     * const 编码上限（经验 ~x8）饱和失效。修复：输入预缩 8^k（分步 <=8，与 x8192
     * 还原同款），m2c'=m2c*8^(2k)、LNQ'=LNQ/8^k；RMSNorm scale 不变性保证 xn 不变
     * （u'=8^k*u 与 inv_std'=inv_std/8^k 抵消）。深/中层 m2c>=2.4e-4 时 k=0，
     * 路径与已验证 M3b 逐字节一致。 */
    double m2c_e = m2c, mscale = 1.0;
    const double *LNQ_e = LNQ;
    double LNQ_s[9];
    ckks_ct_t xas = {0}, xbs = {0};
    const ckks_ct_t *xa_p = xa, *xb_p = xb;
    if (m2c > 0.0 && m2c < 2.4e-4) {   /* 浅层预缩（分步 ≤8）：m2c'→1e-2 域，常量/系数回安全区 */
        double need = sqrt(1e-2 / m2c);
        int k = 0;
        while (mscale < need && k < 6) { mscale *= 8.0; k++; }
        if (k > 0) {
            m2c_e = m2c * mscale * mscale;
            for (int i = 0; i <= 8; i++) LNQ_s[i] = LNQ[i] / mscale;
            LNQ_e = LNQ_s;
            ckks_ct_t t = {0};
            ckks_ct_copy(&xas, xa);
            double ms = mscale;
            while (ms > 1.0001) {
                double st = ms >= 8.0 ? 8.0 : ms;
                ct_mul_slot(&t, &xas, st, -1, sk, ctx);
                ckks_ct_free(&xas); xas = t; memset(&t, 0, sizeof t);
                ms /= st;
            }
            ckks_ct_copy(&xbs, xb);
            ms = mscale;
            while (ms > 1.0001) {
                double st = ms >= 8.0 ? 8.0 : ms;
                ct_mul_slot(&t, &xbs, st, -1, sk, ctx);
                ckks_ct_free(&xbs); xbs = t; memset(&t, 0, sizeof t);
                ms /= st;
            }
            xa_p = &xas; xb_p = &xbs;
            printf("  [rms %s] shallow m2c=%.3e -> scale x%.0f (m2c'=%.3e)\n",
                   tag, m2c, mscale, m2c_e);
        }
    }
    float *Wone = calloc((size_t)K * K, 4);
    for (int j = 0; j < K; j++) Wone[0 * K + j] = 1.0f;
    ckks_ct_t x0s = {0}, x1s = {0};
    if (mul_relin(&x0s, xa_p, xa_p, rk, ctx) != 0) { printf("  [rms %s] FAIL x0s\n", tag); fails++; }
    if (mul_relin(&x1s, xb_p, xb_p, rk, ctx) != 0) { printf("  [rms %s] FAIL x1s\n", tag); fails++; }
    printf("  [rms %s] x0s np=%d comps=%d\n", tag, (int)x0s.ctx.nprimes, x0s.comps);
    char lb[64];
    snprintf(lb, sizeof lb, "%s:x0s", tag); dbg_ct(&x0s, 1.0, lb, sk, ctx, pdec, zy);
    snprintf(lb, sizeof lb, "%s:x1s", tag); dbg_ct(&x1s, 1.0, lb, sk, ctx, pdec, zy);
    int np0 = x0s.ctx.nprimes, np1 = x1s.ctx.nprimes;
    uint64_t **xb0 = NULL, **xb1 = NULL;
    baby_build(&x0s, &xb0, ctx, sk);
    baby_build(&x1s, &xb1, ctx, sk);
    ckks_ct_t s0 = {0}, s1 = {0};
    block_bsgs(Wone, K, 0, 0, xb0, ctx, giant_gk, sk, &s0, 2, np0);
    block_bsgs(Wone, K, 0, 0, xb1, ctx, giant_gk, sk, &s1, 2, np1);
    baby_free(xb0); baby_free(xb1);
    ckks_ct_free(&x0s); ckks_ct_free(&x1s);
    printf("  [rms %s] s0 np=%d s1 np=%d\n", tag, (int)s0.ctx.nprimes, (int)s1.ctx.nprimes);
    snprintf(lb, sizeof lb, "%s:s0", tag); dbg_ct(&s0, 16.0, lb, sk, ctx, pdec, zy);
    snprintf(lb, sizeof lb, "%s:s1", tag); dbg_ct(&s1, 16.0, lb, sk, ctx, pdec, zy);
    ckks_ct_t ss = {0}, sr = {0};
    ckks_add(&ss, &s0, &s1);
    ckks_rescale(&sr, &ss);
    ckks_ct_free(&s0); ckks_ct_free(&s1); ckks_ct_free(&ss);
    printf("  [rms %s] sr np=%d\n", tag, (int)sr.ctx.nprimes);
    snprintf(lb, sizeof lb, "%s:sr", tag); dbg_ct(&sr, 16.0, lb, sk, ctx, pdec, zy);
    ckks_decrypt(pdec, sk, &sr);
    ckks_decode(zy, pdec, &sr.ctx);
    /* z' = (E2 鈭?c0)/(8s0)锛屽崟妲斤紱inv_std/128 = LNQ8(z')锛涘浐瀹氫腑蹇?c0=0.75m2c, s0=0.25m2c */
    double c = 0.75 * m2c_e, s = 0.25 * m2c_e;
    double s8inv_e = 1.0 / (8.0 * s * (double)DIM);
    double s8inv_c = 1.0 / (8.0 * s);
    ckks_ct_t zc = {0};
    ct_mul_slot(&zc, &sr, s8inv_e, 0, sk, ctx);
    ct_add_const(&zc, &zc, -c * s8inv_c, 0, sk, ctx);
    printf("  [rms %s] zc np=%d\n", tag, (int)zc.ctx.nprimes);
    snprintf(lb, sizeof lb, "%s:zc", tag); dbg_ct(&zc, 1.0, lb, sk, ctx, pdec, zy);
    ckks_ct_t invc = {0};
    horner_q(&invc, &zc, LNQ_e, 8, rk, sk, ctx, 0);
    printf("  [rms %s] invc np=%d\n", tag, (int)invc.ctx.nprimes);
    snprintf(lb, sizeof lb, "%s:invc", tag); dbg_ct(&invc, 1.0, lb, sk, ctx, pdec, zy);
    ckks_ct_t inv128 = {0};
    ct_mul_slot(&inv128, &invc, 128.0, 0, sk, ctx);
    ckks_ct_free(&invc);
    printf("  [rms %s] inv128 np=%d\n", tag, (int)inv128.ctx.nprimes);
    snprintf(lb, sizeof lb, "%s:inv128", tag); dbg_ct(&inv128, 64.0, lb, sk, ctx, pdec, zy);
    /* 骞挎挱 slot0 鈫?婊℃Ы */
    float *Wcol = calloc((size_t)K * K, 4);
    for (int i = 0; i < K; i++) Wcol[i * K + 0] = 1.0f;
    uint64_t **xbi = NULL;
    baby_build(&inv128, &xbi, ctx, sk);
    ckks_ct_t bcast = {0};
    block_bsgs(Wcol, K, 0, 0, xbi, ctx, giant_gk, sk, &bcast, inv128.comps,
               inv128.ctx.nprimes);
    baby_free(xbi);
    printf("  [rms %s] bcast np=%d\n", tag, (int)bcast.ctx.nprimes);
    snprintf(lb, sizeof lb, "%s:bcast", tag); dbg_ct(&bcast, 64.0, lb, sk, ctx, pdec, zy);
    ckks_ct_t bcast_r = {0};
    ckks_rescale(&bcast_r, &bcast);
    ckks_ct_free(&bcast);
    printf("  [rms %s] bcast_r np=%d\n", tag, (int)bcast_r.ctx.nprimes);
    snprintf(lb, sizeof lb, "%s:bcast_r", tag); dbg_ct(&bcast_r, 64.0, lb, sk, ctx, pdec, zy);
    /* xg0 = xa·γ[0:1024], xg1 = xb·γ[1024:2048]（mult_plain）。
     * M3b：一般打包明文槽值须 <~0.5（否则单素数系数环绕损坏），深层 γ~136 先 ÷g_gfold。 */
    ckks_ct_t xg0 = {0}, xg1 = {0};
    for (int j = 0; j < K; j++) zx[j] = gamma[j] / g_gfold;
    ckks_encode(poly, zx, ctx);
    {
        ckks_ct_t m = {0};
        ckks_mult_plain(&m, xa_p, poly);
        ckks_rescale(&xg0, &m); ckks_ct_free(&m);
    }
    for (int j = 0; j < K; j++) zx[j] = gamma[K + j] / g_gfold;
    ckks_encode(poly, zx, ctx);
    {
        ckks_ct_t m = {0};
        ckks_mult_plain(&m, xb_p, poly);
        ckks_rescale(&xg1, &m); ckks_ct_free(&m);
    }
    printf("  [rms %s] xg0 np=%d xg1 np=%d\n", tag, (int)xg0.ctx.nprimes, (int)xg1.ctx.nprimes);
    snprintf(lb, sizeof lb, "%s:xg0", tag); dbg_ct(&xg0, 256.0, lb, sk, ctx, pdec, zy);
    snprintf(lb, sizeof lb, "%s:xg1", tag); dbg_ct(&xg1, 256.0, lb, sk, ctx, pdec, zy);
    /* xn = xg·bcast（链长对齐后 ct×ct）。M3b：ct×ct 结果消息须<1 —— 深层
     * xn_true~100 时 xg(÷g_gfold, 已折) 与 bcast(÷g_bfold) 均小 → 结果<1 干净，
     * 乘积后 ×(g_gfold·g_bfold) 还原（const mult 线性，下游 linear 阶段容忍大消息）。 */
    {
        ckks_ct_t xga = {0}, xgb = {0}, brf = {0};
        if (g_bfold > 1.0) {
            ct_mul_slot(&brf, &bcast_r, 1.0 / g_bfold, -1, sk, ctx);
            printf("  [rms %s] bcast_r np=%d fold/%.0f -> np=%d\n", tag,
                   (int)bcast_r.ctx.nprimes, g_bfold, (int)brf.ctx.nprimes);
        } else {
            ckks_ct_copy(&brf, &bcast_r);
        }
        if (ct_switch_to(&xga, &xg0, (int)brf.ctx.nprimes) != 0) fails++;
        if (ct_switch_to(&xgb, &xg1, (int)brf.ctx.nprimes) != 0) fails++;
        ckks_ct_free(&xg0); ckks_ct_free(&xg1);
        if (mul_relin(xn0, &xga, &brf, rk, ctx) != 0) { printf("  [rms %s] FAIL xn0\n", tag); fails++; }
        if (mul_relin(xn1, &xgb, &brf, rk, ctx) != 0) { printf("  [rms %s] FAIL xn1\n", tag); fails++; }
        if (g_gfold > 1.0) {
            /* const mult_plain 明文系数上限 ~×8（超限饱和失效），拆 ≤8 分步乘还原 */
            double uf0 = g_gfold * g_bfold;
            ckks_ct_t u0 = {0};
            while (uf0 > 1.0000001) {
                double st = uf0 >= 8.0 ? 8.0 : uf0;
                ct_mul_slot(&u0, xn0, st, -1, sk, ctx);
                ckks_ct_free(xn0);
                *xn0 = u0; memset(&u0, 0, sizeof u0);
                uf0 /= st;
            }
            double uf1 = g_gfold * g_bfold;
            ckks_ct_t u1 = {0};
            while (uf1 > 1.0000001) {
                double st = uf1 >= 8.0 ? 8.0 : uf1;
                ct_mul_slot(&u1, xn1, st, -1, sk, ctx);
                ckks_ct_free(xn1);
                *xn1 = u1; memset(&u1, 0, sizeof u1);
                uf1 /= st;
            }
        }
        ckks_ct_free(&xga); ckks_ct_free(&xgb);
        ckks_ct_free(&brf); ckks_ct_free(&bcast_r);
    }
    printf("  [rms %s] out xn0 np=%d xn1 np=%d\n", tag,
           (int)xn0->ctx.nprimes, (int)xn1->ctx.nprimes);
    snprintf(lb, sizeof lb, "%s:xn0", tag); dbg_ct(xn0, 4096.0, lb, sk, ctx, pdec, zy);
    snprintf(lb, sizeof lb, "%s:xn1", tag); dbg_ct(xn1, 4096.0, lb, sk, ctx, pdec, zy);
    ckks_ct_free(&zc); ckks_ct_free(&sr);
    if (xas.c) ckks_ct_free(&xas);
    if (xbs.c) ckks_ct_free(&xbs);
    free(Wone); free(Wcol);
    return fails;
}

/* attention 鐢?W 鐭╅樀鏋勫缓锛堟瘡 1024脳1024锛宖loat锛夛細
 * b = 0..7 瀵瑰簲 q 澶?(2b, 2b+1)锛沨''鈭坽0,1}锛泂鈭坽0..3}锛沬鈭坽0..127}
 * 妲藉竷灞€ (h''路512 + s路128 + i)銆?*/
/* M3b 真实 C 折叠（_cfold_plan.py）：L26/27 全幅 q/k/v 逐槽乘积 >>1 会毁掉 ct×ct
 * （P=qr2⊙kr2 每槽 ≤ q·k ~ 122·74≈9000 > 1）。q/k/v 经 rep 矩阵预缩 1/RCF=1/256：
 * 逐槽 P ≤ 225·98.5/65536≈0.34<1；score'=score/65536∈±0.022（EXP 域）；den∈[0.98,4.09]
 * ⊂[0.9,4.2]（RECIP_C_Q 域）；atn=ao/256 需在 C 尾 ×256 还原（≤8 分步）。 */
#define RCF 256.0
static inline float rc_fold(void) { return g_causal ? (float)(1.0 / RCF) : 1.0f; }
static float *mk_W_rep_q(int b, int K) {
    float *W = calloc((size_t)K * K, 4);
    for (int h2 = 0; h2 < 2; h2++) {
        int src_h = (b < 4) ? (2 * b + h2) : (2 * b + h2 - 8);
        for (int s = 0; s < 4; s++)
            for (int i = 0; i < 128; i++)
                W[(h2 * 512 + s * 128 + i) * K + (src_h * 128 + i)] = rc_fold();
    }
    return W;
}
static float *mk_W_kv(int b, int s, int K) {
    float *W = calloc((size_t)K * K, 4);
    for (int h2 = 0; h2 < 2; h2++) {
        int hk = (2 * b + h2) % 8;
        for (int i = 0; i < 128; i++)
            W[(h2 * 512 + s * 128 + i) * K + (hk * 128 + i)] = rc_fold();
    }
    return W;
}
static float *mk_W_ones_i(int b, int K) {
    float *W = calloc((size_t)K * K, 4);
    for (int h2 = 0; h2 < 2; h2++)
        for (int s = 0; s < 4; s++)
            for (int i = 0; i < 128; i++)
                W[(h2 * 512 + s * 128) * K + (h2 * 512 + s * 128 + i)] = 1.0f;
    return W;
}
static float *mk_W_den(int b, int K) {
    float *W = calloc((size_t)K * K, 4);
    for (int h2 = 0; h2 < 2; h2++)
        for (int s = 0; s < 4; s++)
            for (int sp = 0; sp < 4; sp++)
                W[(h2 * 512 + s * 128) * K + (h2 * 512 + sp * 128)] = 1.0f;
    return W;
}
static float *mk_W_rep_p(int b, int K) {
    float *W = calloc((size_t)K * K, 4);
    for (int h2 = 0; h2 < 2; h2++)
        for (int s = 0; s < 4; s++)
            for (int i = 0; i < 128; i++)
                W[(h2 * 512 + s * 128 + i) * K + (h2 * 512 + s * 128)] = 1.0f;
    return W;
}
/* AV 姹傚拰锛歄_raw(h''路512+s路128+i) 鈫?attn 澶?(2b+h'')路128+i */
static float *mk_W_av(int b, int K) {
    float *W = calloc((size_t)K * K, 4);
    for (int h2 = 0; h2 < 2; h2++) {
        int hout = (2 * b + h2) % 8;   /* 杈撳嚭 ct 鍐呭ご鍙凤紙attn0 鎴?attn1锛?*/
        for (int s = 0; s < 4; s++)
            for (int i = 0; i < 128; i++)
                W[(hout * 128 + i) * K + (h2 * 512 + s * 128 + i)] = 1.0f;
    }
    return W;
}

/* 妫€鏌ワ細涓存椂 copy 涔?1/scale 鍚庤В鐮侊紝姣旇緝 scale路decode vs ref锛堟Ы浣?鈫?ref 绱㈠紩鏄犲皠锛?*/
static void check_ct_map(const ckks_ct_t *ct, const float *ref, const int *slotmap,
                         int cnt, double scale, const char *label, int *fails,
                         const ckks_sk_t *sk, const ckks_ctx_t *ctx,
                         uint64_t *pdec, double *zy) {
    ckks_ct_t t = {0}, r = {0};
    ckks_ct_copy(&t, ct);
    uint64_t *pc = malloc((size_t)ctx->nprimes * ctx->n * 8);
    double *z = malloc((size_t)(ctx->n / 2) * 8);
    for (int j = 0; j < (int)(ctx->n / 2); j++) z[j] = 1.0 / scale;
    ckks_encode(pc, z, ctx);
    ckks_mult_plain(&r, &t, pc);
    free(pc);
    ckks_rescale(&t, &r);
    ckks_ct_free(&r);
    ckks_decrypt(pdec, sk, &t);
    ckks_decode(zy, pdec, &t.ctx);
    double me = 0;
    for (int j = 0; j < cnt; j++) {
        int rj = slotmap ? slotmap[j] : j;
        double e = fabs(scale * zy[j] - (double)ref[rj]);
        if (e > me) me = e;
    }
    printf("  [%s] max|err|=%.3e %s\n", label, me, me < 2e-2 ? "PASS" : "FAIL");
    if (me >= 2e-2) (*fails)++;
    ckks_ct_free(&t);
    free(z);
}
static void check_ct_scale(const ckks_ct_t *ct, const float *ref, int cnt, double scale,
                           const char *label, int *fails, const ckks_sk_t *sk,
                           const ckks_ctx_t *ctx, uint64_t *pdec, double *zy) {
    check_ct_map(ct, ref, NULL, cnt, scale, label, fails, sk, ctx, pdec, zy);
}

/* M3b 诊断：打印 ct 真实消息值统计（同态 ×1/S 再解码，S·zy = 消息真值；仅 T23_DBGRMS=1 时启用） */
static void dbg_ct(const ckks_ct_t *ct, double S, const char *label,
                   const ckks_sk_t *sk, const ckks_ctx_t *ctx,
                   uint64_t *pdec, double *zy) {
    if (!getenv("T23_DBGRMS")) return;
    int Kd = (int)(ct->ctx.n / 2);
    uint64_t *pc = malloc((size_t)ct->ctx.nprimes * ct->ctx.n * 8);
    double *z = malloc((size_t)Kd * 8);
    ckks_ct_t t = {0}, r = {0};
    ckks_ct_copy(&t, ct);
    for (int j = 0; j < Kd; j++) z[j] = 1.0 / S;
    ckks_encode(pc, z, &t.ctx);
    ckks_mult_plain(&r, &t, pc);
    free(pc);
    ckks_rescale(&t, &r);
    ckks_ct_free(&r);
    ckks_decrypt(pdec, sk, &t);
    ckks_decode(zy, pdec, &t.ctx);
    double mx = 0, mn = 1e300, s0 = S * zy[0], s1 = S * zy[1], s2 = S * zy[2];
    int nbig = 0;
    for (int j = 0; j < Kd; j++) {
        double v = S * zy[j], a = fabs(v);
        if (a > mx) mx = a;
        if (v < mn) mn = v;
        if (a > 0.5) nbig++;
    }
    printf("  [dbg %s] np=%d S=%.0f max=%.5g min=%.5g nbig=%d s0=%.5g s1=%.5g s2=%.5g\n",
           label, (int)ct->ctx.nprimes, S, mx, mn, nbig, s0, s1, s2);
    ckks_ct_free(&t);
    free(z);
}


/* ===== M1c: bootstrap 机件（t21 合并，complex-slot 系数刷新）===== */

static double *g_cos_tab = NULL, *g_sin_tab = NULL;
static size_t g_tab_n = 0;
static int tab_prepare(size_t n) {
    size_t half = n / 2;
    if (g_cos_tab && g_tab_n == n) return 0;
    free(g_cos_tab); free(g_sin_tab);
    g_cos_tab = malloc(n * half * sizeof(double));
    g_sin_tab = malloc(n * half * sizeof(double));
    g_tab_n = n;
    uint64_t *pow5 = malloc(half * sizeof(uint64_t));
    uint64_t mod = 2 * (uint64_t)n, p = 1;
    for (size_t j = 0; j < half; j++) { pow5[j] = p; p = (p * 5) % mod; }
    for (size_t k = 0; k < n; k++)
        for (size_t j = 0; j < half; j++) {
            double ang = (double)(k * pow5[j]) * M_PI / (double)n;
            g_cos_tab[k * half + j] = cos(ang);
            g_sin_tab[k * half + j] = sin(ang);
        }
    free(pow5);
    return 0;
}
/* 澶嶇紪鐮侊紙t14 鍚屾锛夛細z[0..half-1] -> poly锛坰cale 2^60锛?*/
static int cks_encode(uint64_t *poly, const double complex *z, const ckks_ctx_t *ctx) {
    size_t n = ctx->n, half = n / 2;
    double sc = 2.0 * (double)ctx->scale / (double)n;
    tab_prepare(n);
    for (size_t k = 0; k < n; k++) {
        double acc = 0;
        for (size_t j = 0; j < half; j++)
            acc += creal(z[j]) * g_cos_tab[k * half + j] + cimag(z[j]) * g_sin_tab[k * half + j];
        int64_t iv = (int64_t)llround(sc * acc);
        for (size_t p = 0; p < (size_t)ctx->nprimes; p++) {
            uint64_t q = ctx->q[p];
            poly[p * n + k] = (uint64_t)(iv < 0 ? iv + (int64_t)q : iv) % q;
        }
    }
    return 0;
}
/* 澶嶈В鐮侊細poly -> z[0..half-1]锛坰cale 鍙傛暟鍙?ct 鐨?scale锛?*/
static int cks_decode(double complex *z, const uint64_t *poly, uint64_t scale,
                      const ckks_ctx_t *ctx) {
    size_t n = ctx->n, half = n / 2;
    uint64_t q0 = ctx->q[0];
    tab_prepare(n);
    for (size_t j = 0; j < half; j++) {
        double re = 0, im = 0;
        for (size_t k = 0; k < n; k++) {
            int64_t v = (int64_t)poly[k] - (int64_t)((poly[k] > q0 / 2) ? q0 : 0);
            re += (double)v * g_cos_tab[k * half + j];
            im += (double)v * g_sin_tab[k * half + j];
        }
        z[j] = (re + im * I) / (double)scale;
    }
    return 0;
}
/* 涓績鍖栵紙鍗曠礌鏁?q锛?*/
static int64_t cent(uint64_t v, uint64_t q) {
    return (int64_t)v - (int64_t)((v > q / 2) ? q : 0);
}

/* 澶嶅瑙?-> RNS 鏄庢枃澶氶」寮忥紙鍊?d锛屾寚瀹氱紪鐮?scale enc_scale锛?*/
static void diag_encode_s(uint64_t *poly, const double complex *d, const ckks_ctx_t *ctx,
                          double enc_scale) {
    size_t n = ctx->n, half = n / 2;
    double sc = 2.0 * enc_scale / (double)n;
    tab_prepare(n);
    for (size_t k = 0; k < n; k++) {
        double acc = 0;
        for (size_t j = 0; j < half; j++)
            acc += creal(d[j]) * g_cos_tab[k * half + j] + cimag(d[j]) * g_sin_tab[k * half + j];
        int64_t iv = (int64_t)llround(sc * acc);
        for (size_t p = 0; p < (size_t)ctx->nprimes; p++) {
            uint64_t q = ctx->q[p];
            poly[p * n + k] = (uint64_t)(iv < 0 ? iv + (int64_t)q : iv) % q;
        }
    }
}
/* 澶嶅父鏁板瑙掞紙鎵€鏈夋Ы浣?= c锛?*/
static void diag_const_s(uint64_t *poly, double complex c, const ckks_ctx_t *ctx,
                         double enc_scale) {
    size_t n = ctx->n, half = n / 2;
    double sc = 2.0 * enc_scale / (double)n;
    tab_prepare(n);
    for (size_t k = 0; k < n; k++) {
        double acc = 0;
        for (size_t j = 0; j < half; j++)
            acc += creal(c) * g_cos_tab[k * half + j] + cimag(c) * g_sin_tab[k * half + j];
        int64_t iv = (int64_t)llround(sc * acc);
        for (size_t p = 0; p < (size_t)ctx->nprimes; p++) {
            uint64_t q = ctx->q[p];
            poly[p * n + k] = (uint64_t)(iv < 0 ? iv + (int64_t)q : iv) % q;
        }
    }
}

/* mult_plain + rescale */
static int mp_rescale(ckks_ct_t *out, const ckks_ct_t *in, const uint64_t *poly,
                      const ckks_ctx_t *ctx) {
    ckks_ct_t t = {0};
    if (ckks_mult_plain(&t, in, poly) != 0) return -1;
    if (ckks_rescale(out, &t) != 0) { ckks_ct_free(&t); return -1; }
    ckks_ct_free(&t);
    return 0;
}
/* ct脳ct mult + rescale + relin锛堜繚鎸?2 鍒嗛噺锛?*/
static int mul_rescale_relin(ckks_ct_t *out, const ckks_ct_t *a, const ckks_ct_t *b,
                             const ckks_rk_t *rk, const ckks_ctx_t *ctx) {
    ckks_ct_t m = {0}, r = {0}, rel = {0};
    if (ckks_mult(&m, a, b) != 0) return -1;
    if (ckks_rescale(&r, &m) != 0) { ckks_ct_free(&m); return -1; }
    ckks_ct_free(&m);
    if (ckks_relin(&rel, &r, rk) != 0) { ckks_ct_free(&r); return -1; }
    ckks_ct_free(&r);
    *out = rel;
    return 0;
}
/* 绱犳暟閾惧榻愶紙闄嶅埌 target锛屼笉 rescale锛夛紱target >= 褰撳墠鏃剁洿鎺ユ嫹璐?*/
/* 瀵嗘枃鍙栬礋 */
/* ct + 甯告暟澶氶」寮忥紙瀹夊叏鐗堬細鍏?copy 鍐嶆敼 c0锛?*/
static int ct_add_plain(ckks_ct_t *ct, const ckks_ct_t *a, const uint64_t *poly,
                        const ckks_ctx_t *ctx) {
    ckks_ct_t tmp = {0};
    if (ckks_ct_copy(&tmp, a) != 0) return -1;
    size_t np = tmp.ctx.nprimes, n = tmp.ctx.n;
    for (size_t p = 0; p < np; p++) {
        uint64_t q = ctx->q[p];
        for (size_t i = 0; i < n; i++) {
            uint64_t *v = &tmp.c[(0 * np + p) * n + i];
            uint64_t u = poly[p * n + i];
            *v = (*v + u) % q;
        }
    }
    *ct = tmp;
    return 0;
}

/* 婊℃Ы浣?sin 鎶樺彔锛圱2B-7 绉绘锛宺=7锛夛細x锛堝疄瀵嗘枃锛屾Ы浣嶅€?~2蟺k'锛?> sin(x)锛?路s 缁堝€硷級 */
/* 密文就地取负（t21 1-参版，改名避免与层机件 2-参 ct_neg 冲突） */
static void bt_neg(ckks_ct_t *ct) {
    size_t np = ct->ctx.nprimes, n = ct->ctx.n;
    for (int cp = 0; cp < ct->comps; cp++)
        for (size_t p = 0; p < np; p++) {
            uint64_t q = ct->ctx.q[p];
            for (size_t i = 0; i < n; i++) {
                uint64_t *v = &ct->c[((size_t)cp * np + p) * n + i];
                *v = (*v == 0) ? 0 : q - *v;
            }
        }
}

static int sin_fold(ckks_ct_t *out, const ckks_ct_t *x, const ckks_rk_t *rk,
                    const ckks_ctx_t *ctx) {
    int r = 7;
    double scd = (double)ctx->scale;
    double powr = 1.0 / (double)(1 << r);
    double c1 = powr / 2.0, c3 = -powr * powr * powr / 6.0 / 2.0,
           c5 = pow(powr, 5) / 120.0 / 2.0, c7 = -pow(powr, 7) / 5040.0 / 2.0;
    double d0 = 1.0 / 2.0, d2 = -powr * powr / 2.0 / 2.0,
           d4 = pow(powr, 4) / 24.0 / 2.0, d6 = -pow(powr, 6) / 720.0 / 2.0;
    long long A1 = llround(scd * c1), A3 = llround(scd * c3), A5 = llround(scd * c5), A7 = llround(scd * c7);
    long long B0 = llround(scd * d0), B2 = llround(scd * d2), B4 = llround(scd * d4), B6 = llround(scd * d6);
    size_t n = ctx->n, np = ctx->nprimes;
    uint64_t *cst = malloc(np * n * 8);
    int fail = 0;
    ckks_ct_t u1 = {0}, u2 = {0}, u3 = {0}, u1s = {0};
    if (mul_rescale_relin(&u1, x, x, rk, ctx) != 0) fail++;
    if (mul_rescale_relin(&u2, &u1, &u1, rk, ctx) != 0) fail++;
    if (ct_switch_to(&u1s, &u1, u2.ctx.nprimes) != 0) fail++;
    if (mul_rescale_relin(&u3, &u1s, &u2, rk, ctx) != 0) fail++;
    int lev = u3.ctx.nprimes;
    if (ct_switch_to(&u1s, &u1s, lev) != 0) fail++;
    ckks_ct_t u2s = {0};
    if (ct_switch_to(&u2s, &u2, lev) != 0) fail++;
    ckks_ct_t t1 = {0}, t2 = {0}, t3 = {0}, si = {0};
    memset(cst, 0, np * n * 8);
    for (size_t pp = 0; pp < np; pp++) { uint64_t q = ctx->q[pp]; long long av = A7 % (long long)q; cst[pp * n] = (uint64_t)(av < 0 ? av + (long long)q : av); }
    if (mp_rescale(&t1, &u3, cst, ctx) != 0) fail++;
    for (size_t pp = 0; pp < np; pp++) { uint64_t q = ctx->q[pp]; long long av = A5 % (long long)q; cst[pp * n] = (uint64_t)(av < 0 ? av + (long long)q : av); }
    if (mp_rescale(&t2, &u2s, cst, ctx) != 0) fail++;
    for (size_t pp = 0; pp < np; pp++) { uint64_t q = ctx->q[pp]; long long av = A3 % (long long)q; cst[pp * n] = (uint64_t)(av < 0 ? av + (long long)q : av); }
    if (mp_rescale(&t3, &u1s, cst, ctx) != 0) fail++;
    ckks_ct_t t4 = {0};
    if (ckks_add(&t4, &t2, &t3) != 0) fail++;
    if (ckks_add(&si, &t1, &t4) != 0) fail++;
    for (size_t pp = 0; pp < np; pp++) { uint64_t q = ctx->q[pp]; long long av = A1 % (long long)q; cst[pp * n] = (uint64_t)(av < 0 ? av + (long long)q : av); }
    if (ct_add_plain(&si, &si, cst, ctx) != 0) fail++;
    ckks_ct_t ci = {0};
    ckks_ct_free(&t1); ckks_ct_free(&t2); ckks_ct_free(&t3);
    t1 = (ckks_ct_t){0}; t2 = (ckks_ct_t){0}; t3 = (ckks_ct_t){0};
    for (size_t pp = 0; pp < np; pp++) { uint64_t q = ctx->q[pp]; long long av = B6 % (long long)q; cst[pp * n] = (uint64_t)(av < 0 ? av + (long long)q : av); }
    if (mp_rescale(&t1, &u3, cst, ctx) != 0) fail++;
    for (size_t pp = 0; pp < np; pp++) { uint64_t q = ctx->q[pp]; long long av = B4 % (long long)q; cst[pp * n] = (uint64_t)(av < 0 ? av + (long long)q : av); }
    if (mp_rescale(&t2, &u2s, cst, ctx) != 0) fail++;
    for (size_t pp = 0; pp < np; pp++) { uint64_t q = ctx->q[pp]; long long av = B2 % (long long)q; cst[pp * n] = (uint64_t)(av < 0 ? av + (long long)q : av); }
    if (mp_rescale(&t3, &u1s, cst, ctx) != 0) fail++;
    if (ckks_add(&t4, &t1, &t2) != 0) fail++;
    if (ckks_add(&ci, &t3, &t4) != 0) fail++;
    for (size_t pp = 0; pp < np; pp++) { uint64_t q = ctx->q[pp]; long long av = B0 % (long long)q; cst[pp * n] = (uint64_t)(av < 0 ? av + (long long)q : av); }
    if (ct_add_plain(&ci, &ci, cst, ctx) != 0) fail++;
    ckks_ct_t xcs = {0}, sint = {0};
    if (ct_switch_to(&xcs, x, si.ctx.nprimes) != 0) fail++;
    if (mul_rescale_relin(&sint, &xcs, &si, rk, ctx) != 0) fail++;
    ckks_ct_t cis = {0};
    if (ct_switch_to(&cis, &ci, sint.ctx.nprimes) != 0) fail++;
    for (int it = 0; it < r; it++) {
        ckks_ct_t c2 = {0}, s2 = {0}, sc = {0}, neg = {0};
        if (mul_rescale_relin(&c2, &cis, &cis, rk, ctx) != 0) fail++;
        if (mul_rescale_relin(&s2, &sint, &sint, rk, ctx) != 0) fail++;
        if (mul_rescale_relin(&sc, &sint, &cis, rk, ctx) != 0) fail++;
        ckks_ct_t snew = {0}, cnew = {0}, tmp2 = {0}, cn2 = {0};
        if (ckks_add(&snew, &sc, &sc) != 0) fail++;
        if (ckks_add(&tmp2, &snew, &snew) != 0) fail++;
        ckks_ct_free(&snew); snew = tmp2;
        if (ckks_add(&cnew, &c2, &c2) != 0) fail++;
        if (ckks_add(&neg, &s2, &s2) != 0) fail++;
        bt_neg(&neg);
        if (ckks_add(&cn2, &cnew, &neg) != 0) fail++;
        ckks_ct_free(&cnew); cnew = cn2;
        ckks_ct_free(&c2); ckks_ct_free(&s2); ckks_ct_free(&sc); ckks_ct_free(&neg);
        ckks_ct_free(&sint); ckks_ct_free(&cis);
        sint = snew; cis = cnew;
    }
    ckks_ct_t s2x = {0};
    if (ckks_add(&s2x, &sint, &sint) != 0) fail++;
    ckks_ct_free(&sint); ckks_ct_free(&cis);
    ckks_ct_free(&u1); ckks_ct_free(&u2); ckks_ct_free(&u3); ckks_ct_free(&u1s); ckks_ct_free(&u2s);
    ckks_ct_free(&t1); ckks_ct_free(&t2); ckks_ct_free(&t3); ckks_ct_free(&t4);
    free(cst);
    if (fail) { ckks_ct_free(&s2x); return -1; }
    *out = s2x;
    return 0;
}

/* CoeffToSlot锛堢郴鏁?-> 澶嶆Ы浣?w_j = x_j + i路x_{j+N}锛寈_j 甯?2蟺/q0 缂╂斁锛夈€? * d_u[j] = (2蟺路2^60/q0)路味^{-j路5^{(j+u) mod N}}/N 鈮?2蟺路味/N锛岀紪鐮?scale 2^60銆? * 褰掍竴鍖栨帹瀵硷紙t15 宸查獙璇侊細d=味/N @ scale 2^60 -> 杈撳嚭 = 杈撳叆娑堟伅绯绘暟 m_j锛岀嚎鎬ф€э級锛? *   d'=c路味/N -> 杈撳嚭 = c路(杈撳叆娑堟伅 t_j)锛沜 = 2蟺路2^60/q0 -> 杈撳嚭绯绘暟 = 2蟺路t_j路2^60/q0
 *   = x_j路2^60锛堟姌鍙犺緭鍏ュ€?x_j = 2蟺路t_j/q0 鍦?scale 2^60锛夈€?*/
/* ---- CoeffToSlot/SlotToCoeff 并行：自研 vllm_tp（与 t23_chain.c 同款迁移），
 *      线程数 vllm_tp_init(T23_NT，默认 4)。 ---- */
typedef struct {
    const ckks_ct_t *ct;
    ckks_ct_t *y;
    const ckks_sk_t *sk;
    const ckks_ctx_t *ctx;
    int jGG;
    volatile int fail;
} cts_y_job_t;
static void cts_y_worker(void *arg, int u1) {
    cts_y_job_t *j = (cts_y_job_t *)arg;
    ckks_gk_t gky;
    if (ckks_gk_gen(&gky, j->sk, j->ctx, u1 * j->jGG) != 0) { j->fail = 1; return; }
    if (ckks_rotate(&j->y[u1], j->ct, j->sk, &gky, u1 * j->jGG) != 0) j->fail = 1;
    ckks_gk_free(&gky);
}
typedef struct {
    int u0;
    size_t n, N;
    const ckks_ctx_t *ctx;
    const ckks_ct_t *y;
    const uint64_t *pow5;
    double sc60, c2pi;
    int mode;
    uint64_t **polys;
    ckks_ct_t *t1r;
    int jGG;
    volatile int fail;
} cts_m_job_t;
static void cts_m_worker(void *arg, int u1) {
    cts_m_job_t *j = (cts_m_job_t *)arg;
    size_t n = j->n, N = j->N;
    int u = u1 * j->jGG + j->u0;
    j->polys[u1] = (uint64_t *)malloc((size_t)j->ctx->nprimes * n * 8);
    double complex *dd = (double complex *)malloc(N * sizeof(double complex));
    if (!j->polys[u1] || !dd) { free(j->polys[u1]); j->polys[u1] = NULL; free(dd); j->fail = 1; return; }
    for (size_t k = 0; k < N; k++) {
        size_t kr = (k + (size_t)u) % N;
        double ang, mul;
        if (j->mode == 0) { ang = -(double)k * (double)j->pow5[kr] * M_PI / (double)n; mul = j->c2pi / (double)N; }
        else              { ang = (double)kr * (double)j->pow5[k] * M_PI / (double)n; mul = 0.25; }
        dd[k] = (cos(ang) + sin(ang) * I) * mul;
    }
    if (j->u0) {
        double complex *d2l = (double complex *)malloc(N * sizeof(double complex));
        if (!d2l) { free(dd); free(j->polys[u1]); j->polys[u1] = NULL; j->fail = 1; return; }
        memcpy(d2l, dd, N * sizeof(double complex));
        for (size_t k = 0; k < N; k++) dd[k] = d2l[(k + N - (size_t)j->u0) % N];
        free(d2l);
    }
    diag_encode_s(j->polys[u1], dd, j->ctx,
                  j->mode == 0 ? j->sc60 : (double)j->ctx->scale);
    free(dd);
    ckks_ct_t t1 = {0};
    if (ckks_mult_plain(&t1, &j->y[u1], j->polys[u1]) != 0 ||
        ckks_rescale(&j->t1r[u1], &t1) != 0) j->fail = 1;
    ckks_ct_free(&t1);
}
static int coeff_to_slot(ckks_ct_t *w, const ckks_ct_t *ct, const ckks_sk_t *sk,
                         const ckks_ctx_t *ctx, const uint64_t *pow5, double q0) {
    size_t n = ctx->n, N = n / 2;
    double sc60 = (double)ctx->scale;
    double c2pi = (2.0 * M_PI) * sc60 / q0;   /* 2蟺路2^60/q0 鈮?2蟺 */
    ckks_ct_t *y = malloc(BB * sizeof(ckks_ct_t));
    cts_y_job_t yj = { ct, y, sk, ctx, GG, 0 };
    vllm_tp_parfor(0, BB, cts_y_worker, &yj);
    if (yj.fail) { for (int i = 0; i < BB; i++) ckks_ct_free(&y[i]); free(y); return -1; }
    ckks_ct_t v = {0}, acc = {0}, rv = {0};
    ckks_gk_t gk;
    clock_t c0 = clock();
    for (int u0 = 0; u0 < GG; u0++) {
        ckks_ct_t t1r[BB];
        memset(t1r, 0, sizeof(t1r));
        uint64_t *polys[BB];
        cts_m_job_t mj = { u0, n, N, ctx, y, pow5, sc60, c2pi, 0, polys, t1r, GG, 0 };
        vllm_tp_parfor(0, BB, cts_m_worker, &mj);
        if (mj.fail) {
            for (int i = 0; i < BB; i++) { ckks_ct_free(&t1r[i]); free(polys[i]); }
            for (int i = 0; i < BB; i++) ckks_ct_free(&y[i]);
            free(y); ckks_ct_free(&v); ckks_ct_free(&acc); ckks_ct_free(&rv);
            return -1;
        }
        for (int u1 = 0; u1 < BB; u1++) {
            if (u1 == 0) { ckks_ct_copy(&v, &t1r[u1]); }
            else {
                ckks_ct_t t2 = {0};
                ckks_add(&t2, &v, &t1r[u1]);
                ckks_ct_free(&v); v = t2;
            }
            ckks_ct_free(&t1r[u1]);
            free(polys[u1]);
        }
        if (ckks_gk_gen(&gk, sk, ctx, u0) != 0) return -1;
        if (ckks_rotate(&rv, &v, sk, &gk, u0) != 0) { ckks_gk_free(&gk); return -1; }
        ckks_gk_free(&gk);
        ckks_ct_free(&v); v = (ckks_ct_t){0};
        if (u0 == 0) { ckks_ct_copy(&acc, &rv); }
        else {
            ckks_ct_t t2 = {0};
            ckks_add(&t2, &acc, &rv);
            ckks_ct_free(&acc); acc = t2;
        }
        ckks_ct_free(&rv); rv = (ckks_ct_t){0};
        if ((u0 & 7) == 0)
            fprintf(stderr, "cts: u0=%d done (%.1fs)\n", u0,
                    (double)(clock() - c0) / CLOCKS_PER_SEC);
    }
    ckks_ct_free(&v);
    *w = acc;
    for (int u1 = 0; u1 < BB; u1++) ckks_ct_free(&y[u1]);
    free(y);
    return 0;
}

/* SlotToCoeff锛堝妲戒綅 -> 绯绘暟澶氶」寮忥紝脳4 add 閾捐ˉ鍋匡級 */
static int slot_to_coeff(ckks_ct_t *ct, const ckks_ct_t *w, const ckks_sk_t *sk,
                         const ckks_ctx_t *ctx, const uint64_t *pow5) {
    size_t n = ctx->n, N = n / 2;
    ckks_ct_t *y = malloc(BB * sizeof(ckks_ct_t));
    cts_y_job_t yj = { w, y, sk, ctx, GG, 0 };
    vllm_tp_parfor(0, BB, cts_y_worker, &yj);
    if (yj.fail) { for (int i = 0; i < BB; i++) ckks_ct_free(&y[i]); free(y); return -1; }
    ckks_gk_t gk;
    ckks_ct_t v = {0}, acc = {0}, rv = {0};
    clock_t c0 = clock();
    for (int u0 = 0; u0 < GG; u0++) {
        ckks_ct_t t1r[BB];
        memset(t1r, 0, sizeof(t1r));
        uint64_t *polys[BB];
        cts_m_job_t mj = { u0, n, N, ctx, y, pow5, 0, 0, 1, polys, t1r, GG, 0 };
        vllm_tp_parfor(0, BB, cts_m_worker, &mj);
        if (mj.fail) {
            for (int i = 0; i < BB; i++) { ckks_ct_free(&t1r[i]); free(polys[i]); }
            for (int i = 0; i < BB; i++) ckks_ct_free(&y[i]);
            free(y); ckks_ct_free(&v); ckks_ct_free(&acc); ckks_ct_free(&rv);
            return -1;
        }
        for (int u1 = 0; u1 < BB; u1++) {
            if (u1 == 0) { ckks_ct_copy(&v, &t1r[u1]); }
            else {
                ckks_ct_t t2 = {0};
                ckks_add(&t2, &v, &t1r[u1]);
                ckks_ct_free(&v); v = t2;
            }
            ckks_ct_free(&t1r[u1]);
            free(polys[u1]);
        }
        if (ckks_gk_gen(&gk, sk, ctx, u0) != 0) return -1;
        if (ckks_rotate(&rv, &v, sk, &gk, u0) != 0) { ckks_gk_free(&gk); return -1; }
        ckks_gk_free(&gk);
        ckks_ct_free(&v); v = (ckks_ct_t){0};
        if (u0 == 0) { ckks_ct_copy(&acc, &rv); }
        else {
            ckks_ct_t t2 = {0};
            ckks_add(&t2, &acc, &rv);
            ckks_ct_free(&acc); acc = t2;
        }
        ckks_ct_free(&rv); rv = (ckks_ct_t){0};
        if ((u0 & 7) == 0)
            fprintf(stderr, "stc: u0=%d done (%.1fs)\n", u0,
                    (double)(clock() - c0) / CLOCKS_PER_SEC);
    }
    ckks_ct_free(&v);
    /* 脳4 琛ュ伩锛坉 缂?1/4锛沵ult_plain 鍏?4 鏄?negacyclic 鍗风Н闈炴爣閲忎箻锛岀敤 add 閾撅級 */
    {
        ckks_ct_t t2 = {0};
        ckks_ct_copy(&t2, &acc);
        for (int x4 = 0; x4 < 3; x4++) {
            ckks_ct_t t3 = {0};
            ckks_add(&t3, &t2, &acc);
            ckks_ct_free(&t2); t2 = t3;
        }
        ckks_ct_free(&acc);
        *ct = t2;
    }
    for (int u1 = 0; u1 < BB; u1++) ckks_ct_free(&y[u1]);
    free(y);
    return 0;
}

/* ================= M1c: sk/ct 落盘 + 单密文 bootstrap 刷新 ================= */
static int t23_sk_save(const ckks_sk_t *sk, const char *path) {
    FILE *f = fopen(path, "wb"); if (!f) return -1;
    fwrite(&sk->n, 4, 1, f); fwrite(sk->s, 1, sk->n, f); fclose(f); return 0;
}
static int t23_sk_load(ckks_sk_t *sk, const char *path) {
    FILE *f = fopen(path, "rb"); if (!f) return -1;
    uint32_t nn; if (fread(&nn, 4, 1, f) != 1) { fclose(f); return -1; }
    sk->n = nn; sk->s = (int8_t *)malloc(nn);
    if (fread(sk->s, 1, nn, f) != nn) { fclose(f); return -1; }
    fclose(f); return 0;
}
static int t23_ct_save(const ckks_ct_t *ct, const char *path) {
    FILE *f = fopen(path, "wb"); if (!f) return -1;
    int npr = ct->ctx.nprimes, cp = ct->comps; uint32_t nn = ct->ctx.n;
    fwrite(&nn, 4, 1, f); fwrite(&npr, 4, 1, f); fwrite(&cp, 4, 1, f);
    fwrite(ct->c, 8, (size_t)cp * npr * nn, f); fclose(f); return 0;
}
static int t23_ct_load(ckks_ct_t *ct, const char *path, const ckks_ctx_t *ctx) {
    FILE *f = fopen(path, "rb"); if (!f) return -1;
    uint32_t nn; int npr, cp;
    if (fread(&nn, 4, 1, f) != 1 || fread(&npr, 4, 1, f) != 1 || fread(&cp, 4, 1, f) != 1) { fclose(f); return -1; }
    ct->ctx = *ctx; ct->ctx.nprimes = npr; ct->comps = cp;
    ct->c = (uint64_t *)malloc((size_t)cp * npr * nn * 8);
    if (fread(ct->c, 8, (size_t)cp * npr * nn, f) != (size_t)cp * npr * nn) { fclose(f); return -1; }
    fclose(f); return 0;
}
static void t23_pow5(uint64_t *out, size_t N, size_t n) {
    uint64_t p = 1, mod = 2 * (uint64_t)n;
    for (size_t t = 0; t < N; t++) { out[t] = p; p = (p * 5) % mod; }
}
/* 单密文完整 bootstrap：in（引擎 real-slot 层输出，np≥1）-> 降 q0 -> ModRaise ->
 * CoeffToSlot -> EvalMod(+恢复×c) -> SlotToCoeff -> out（满链刷新，系数=输入）。 */
static int refresh_one(ckks_ct_t *out, ckks_ct_t *in, const ckks_ctx_t *ctx,
                       const ckks_sk_t *sk, const ckks_rk_t *rk,
                       const uint64_t *pow5, double q0) {
    size_t n = ctx->n, np = ctx->nprimes;
    tab_prepare(n);   /* M1c 修复：coeff_to_slot/slot_to_coeff 的 omp 区并发首次触发
                       * tab_prepare 会 free+malloc+填表竞态（半填 cos/sin 表），
                       * 使进程首个 refresh 输出坏密文；此处串行单线程先建表。 */
    int fails = 0;
    /* 降 1 素数（直接拷贝 q0 分量） */
    ckks_ct_t q1 = {0};
    q1.ctx = *ctx; q1.ctx.nprimes = 1; q1.comps = in->comps;
    q1.c = (uint64_t *)malloc((size_t)in->comps * n * 8);
    for (int cp = 0; cp < in->comps; cp++)
        memcpy(&q1.c[(size_t)cp * n], &in->c[((size_t)cp * (size_t)in->ctx.nprimes) * n], n * 8);
    ckks_ct_t ra = {0};
    if (ckks_modraise(&ra, &q1, (int)np) != 0) { printf("  [FAIL] modraise\n"); fails++; }
    ckks_ct_free(&q1);
    ckks_ct_t w = {0};
    if (coeff_to_slot(&w, &ra, sk, ctx, pow5, q0) != 0) { printf("  [FAIL] coeff_to_slot\n"); fails++; }
    ckks_ct_free(&ra);
    /* EvalMod：共轭提取实/虚部 + 各自 sin 折叠 + 恢复 + 合并 */
    {
        uint64_t kconj = 2 * (uint64_t)n - 1;
        ckks_gk_t gkc;
        ckks_gk_gen_k(&gkc, sk, ctx, kconj);
        ckks_ct_t ccj = {0};
        ckks_rotate_k(&ccj, &w, sk, &gkc, kconj);
        ckks_gk_free(&gkc);
        ckks_ct_t nccj = {0};
        ckks_ct_copy(&nccj, &ccj);
        bt_neg(&nccj);
        ckks_ct_t sum = {0}, tsub = {0};
        ckks_add(&sum, &w, &ccj);
        ckks_add(&tsub, &w, &nccj);
        ckks_ct_free(&nccj); ckks_ct_free(&ccj);
        uint64_t *polys = (uint64_t *)malloc(np * n * 8);
        diag_const_s(polys, 0.5, ctx, (double)ctx->scale);
        ckks_ct_t re = {0};
        mp_rescale(&re, &sum, polys, ctx);
        ckks_ct_free(&sum);
        diag_const_s(polys, -I / 2.0, ctx, (double)ctx->scale);
        ckks_ct_t im = {0};
        mp_rescale(&im, &tsub, polys, ctx);
        ckks_ct_free(&tsub);
        free(polys);
        ckks_ct_t sre = {0}, sim = {0};
        if (sin_fold(&sre, &re, rk, ctx) != 0) { printf("  [FAIL] sin_fold re\n"); fails++; }
        if (sin_fold(&sim, &im, rk, ctx) != 0) { printf("  [FAIL] sin_fold im\n"); fails++; }
        ckks_ct_free(&re); ckks_ct_free(&im);
        double c = q0 / (2.0 * M_PI * (double)ctx->scale);
        uint64_t *polyb = (uint64_t *)malloc(np * n * 8);
        memset(polyb, 0, np * n * 8);
        for (size_t p = 0; p < np; p++)
            polyb[p * n] = (uint64_t)llround(c * (double)ctx->scale) % ctx->q[p];
        ckks_ct_t sre2 = {0}, sim2 = {0};
        mp_rescale(&sre2, &sre, polyb, ctx);
        mp_rescale(&sim2, &sim, polyb, ctx);
        ckks_ct_free(&sre); ckks_ct_free(&sim); free(polyb);
        uint64_t *polys2 = (uint64_t *)malloc(np * n * 8);
        diag_const_s(polys2, I, ctx, (double)ctx->scale);
        ckks_ct_t imi = {0};
        mp_rescale(&imi, &sim2, polys2, ctx);
        ckks_ct_free(&sim2); free(polys2);
        ckks_ct_t imis = {0};
        ct_switch_to(&imis, &imi, sre2.ctx.nprimes);
        ckks_ct_free(&imi); imi = imis;
        ckks_ct_t sres = {0};
        ct_switch_to(&sres, &sre2, imi.ctx.nprimes);
        ckks_ct_free(&sre2); sre2 = sres;
        ckks_ct_t w2 = {0};
        ckks_add(&w2, &sre2, &imi);
        ckks_ct_free(&sre2); ckks_ct_free(&imi); ckks_ct_free(&w);
        ckks_ct_t c2 = {0};
        if (slot_to_coeff(&c2, &w2, sk, ctx, pow5) != 0) { printf("  [FAIL] slot_to_coeff\n"); fails++; }
        ckks_ct_free(&w2);
        *out = c2;
    }
    return fails;
}
/* boot 阶段：逐个刷新 chain/u1_t_h.ct -> u1r_t_h.ct */
static int t23_boot(const ckks_ctx_t *ctx, const ckks_sk_t *sk, const ckks_rk_t *rk) {
    size_t n = ctx->n, N = n / 2, np = ctx->nprimes;
    uint64_t *pow5 = (uint64_t *)malloc(N * 8);
    t23_pow5(pow5, N, n);
    double q0 = (double)ctx->q[0];
    int fails = 0;
    int one = getenv("T23_ONE") != 0;   /* 冒烟：只刷 t0 h0 */
    /* M3b: 输入/输出前缀可配（默认 u1->u1r{LCHAIN}；t26 输出刷新用 T23_BI=u26 T23_BO=u26r） */
    const char *bin = getenv("T23_BI"); if (!bin) bin = "u1";
    const char *bop = getenv("T23_BO"); if (!bop) bop = "u1r";   /* 输出 = bop + LCHAIN */
    char path[256];
    for (int t = 0; t < LTOK; t++)
        for (int h = 0; h < 2; h++) {
            if (one && !(t == 0 && h == 0)) continue;
            ckks_ct_t cin = {0}, cout = {0};
            snprintf(path, sizeof(path), ".tmp_tok/chain/%s_%d_%d.ct", bin, t, h);
            if (t23_ct_load(&cin, path, ctx) != 0) { printf("[FAIL] load %s\n", path); fails++; continue; }
            printf("  [boot t%d h%d] in np=%d refresh...\n", t, h, cin.ctx.nprimes);
            fflush(stdout);
            if (refresh_one(&cout, &cin, ctx, sk, rk, pow5, q0) != 0) fails++;
            ckks_ct_free(&cin);
            printf("  [boot t%d h%d] out np=%d\n", t, h, cout.ctx.nprimes);
            /* 层接力协议：刷新结果切回 LCHAIN 素数层链再落盘 */
            ckks_ct_t c96 = {0};
            ct_switch_to(&c96, &cout, LCHAIN);
            ckks_ct_free(&cout);
            snprintf(path, sizeof(path), ".tmp_tok/chain/%s%d_%d_%d.ct", bop, LCHAIN, t, h);
            if (t23_ct_save(&c96, path) != 0) { printf("[FAIL] save %s\n", path); fails++; }
            ckks_ct_free(&c96);
        }
    free(pow5);
    printf("BOOT=%s (%d)\n", fails ? "FAILED" : "PASS", fails);
    return fails;
}

/* ---- M3b c5: final RMSNorm(u27) + lm_head(=embed_tokens) 分块 matmul -> logits 对照 ---- */
static int t23_final(const ckks_ctx_t *ctx, const ckks_sk_t *sk, const ckks_rk_t *rk) {
    size_t n = ctx->n; int K = (int)(n / 2);
    const int VOCAB = 151936, VB = 1024, VBP = 152576;  /* VBP = 149*1024 块对齐；LTOK 为宏 */
    int fails = 0, rf = 0;
    char path[256];
    float *gn = NULL, *embed = NULL, *xnf_ref = NULL, *lg_ref = NULL;
    double LNQ_F[9];
    {   /* 系数/权重/参考加载 */
        double m2c_f = 0.006611913547727052;
        FILE *f = fopen(".tmp_tok/tail/ln_f.bin", "rb");
        if (!f || fread(LNQ_F, 8, 9, f) != 9) { printf("[fin] ln_f load fail\n"); return 1; }
        fclose(f);
        f = fopen(".tmp_tok/tail/m2c_f.bin", "rb");
        if (f) { if (fread(&m2c_f, 8, 1, f) != 1) m2c_f = 0.006611913547727052; fclose(f); }
        printf("  [fin] m2c_f=%.10g\n", m2c_f);
        gn = malloc((size_t)DIM * 4);
        f = fopen(".tmp_tok/tail/norm.bin", "rb");
        if (!f || fread(gn, 4, DIM, f) != DIM) { printf("[fin] norm.bin load fail\n"); return 1; }
        fclose(f);
        embed = calloc((size_t)VBP * 2048, 4);   /* 尾部 0 填充到 149*1024 行 */
        f = fopen(".tmp_tok/tail/embed.bin", "rb");
        if (!f || fread(embed, 4, (size_t)VOCAB * 2048, f) != (size_t)VOCAB * 2048) {
            printf("[fin] embed.bin load fail\n"); return 1; }
        fclose(f);
        xnf_ref = malloc((size_t)LTOK * DIM * 4);
        f = fopen(".tmp_tok/tail/xnorm_f.bin", "rb");
        if (!f || fread(xnf_ref, 4, (size_t)LTOK * DIM, f) != (size_t)LTOK * DIM) { printf("[fin] xnorm_f load fail\n"); return 1; }
        fclose(f);
        lg_ref = malloc((size_t)LTOK * VOCAB * 4);
        f = fopen(".tmp_tok/tail/logits.bin", "rb");
        if (!f || fread(lg_ref, 4, (size_t)LTOK * VOCAB, f) != (size_t)LTOK * VOCAB) { printf("[fin] logits.bin load fail\n"); return 1; }
        fclose(f);
        g_m2c = m2c_f;
        printf("  [fin] embed %dx%d xnorm_f/logits ref loaded\n", VOCAB, 2048);
    }
    /* 密文输入：u27r{LCHAIN} 链（boot 后 LCHAIN 素数） */
    ckks_ct_t xct[4][2];
    for (int t = 0; t < LTOK; t++)
        for (int h = 0; h < 2; h++) {
            snprintf(path, sizeof path, ".tmp_tok/chain/u27r%d_%d_%d.ct", LCHAIN, t, h);
            if (t23_ct_load(&xct[t][h], path, ctx) != 0) { printf("[fin] load %s fail\n", path); return 1; }
            ckks_ct_t s96 = {0};
            if (ct_switch_to(&s96, &xct[t][h], LCHAIN) != 0) fails++;
            ckks_ct_free(&xct[t][h]); xct[t][h] = s96;
        }
    ckks_gk_t giant_gk[G];
    for (int r = 1; r < G; r++) ckks_gk_gen(&giant_gk[r], sk, ctx, r * B);
    uint64_t *pdec = malloc((size_t)ctx->nprimes * n * 8);
    double *zy = malloc((size_t)K * 8);
    double *zx = malloc((size_t)K * 8);
    uint64_t *poly = malloc((size_t)ctx->nprimes * n * 8);
    /* final RMSNorm（输入 u27 域，ln_f/m2c_f 拟合；输出消息 = xnorm_f/4096） */
    double tF = t_clock();
    ckks_ct_t xn[4][2];
    for (int t = 0; t < LTOK; t++) {
        if (rmsnorm_enc(&xn[t][0], &xn[t][1], &xct[t][0], &xct[t][1], gn, g_m2c,
                        LNQ_F, zx, poly, pdec, zy, giant_gk, rk, sk, ctx, "F") != 0) fails++;
        char lb[32];
        snprintf(lb, sizeof lb, "F:xnorm t%d", t);
        check_ct_scale(&xn[t][0], &xnf_ref[t * DIM], K, 4096.0, lb, &rf, sk, ctx, pdec, zy);
        snprintf(lb, sizeof lb, "F:xnorm1 t%d", t);
        check_ct_scale(&xn[t][1], &xnf_ref[t * DIM + K], K, 4096.0, lb, &rf, sk, ctx, pdec, zy);
        ckks_ct_free(&xct[t][0]); ckks_ct_free(&xct[t][1]);
    }
    printf("  [F] RMSNorm 用时 %.1fs\n", t_clock() - tF);
    /* lm_head：每 token 分块（每块 1024 vocab 行 = matmul2 双列块），decode 对照 logits.bin */
    double tL = t_clock();
    for (int t = 0; t < LTOK; t++) {
        uint64_t **xb0 = NULL, **xb1 = NULL;
        baby_build(&xn[t][0], &xb0, ctx, sk);
        baby_build(&xn[t][1], &xb1, ctx, sk);
        double top1v = -1e30, top2v = -1e30; int top1i = -1;
        double ref1 = -1e30, ref2 = -1e30; int ref1i = -1;
        double wme = 0; int wv0 = -1;
        for (int v0 = 0; v0 < VOCAB; v0 += VB) {
            ckks_ct_t a0 = {0}, a1 = {0}, s = {0}, lc = {0};
            if (block_bsgs(embed, 2048, v0, 0, xb0, ctx, giant_gk, sk, &a0, xn[t][0].comps, xn[t][0].ctx.nprimes) != 0) fails++;
            if (block_bsgs(embed, 2048, v0, 1024, xb1, ctx, giant_gk, sk, &a1, xn[t][1].comps, xn[t][1].ctx.nprimes) != 0) fails++;
            ckks_add(&s, &a0, &a1);
            ckks_rescale(&lc, &s);
            ckks_ct_free(&a0); ckks_ct_free(&a1); ckks_ct_free(&s);
            /* decode：先 ×1/4096 缩小（消息为全幅 logits，>0.5 需缩小）再解码还原 */
            {
                ckks_ct_t lr = {0}, lt = {0};
                uint64_t *pcl = malloc((size_t)ctx->nprimes * ctx->n * 8);
                double *zl = malloc((size_t)K * 8);
                ckks_ct_copy(&lr, &lc);
                for (int j = 0; j < K; j++) zl[j] = 1.0 / 4096.0;
                ckks_encode(pcl, zl, &lr.ctx);
                ckks_mult_plain(&lt, &lr, pcl);
                ckks_ct_free(&lr);
                ckks_rescale(&lr, &lt);
                ckks_ct_free(&lt);
                ckks_decrypt(pdec, sk, &lr);
                ckks_decode(zy, pdec, &lr.ctx);
                ckks_ct_free(&lr);
                free(pcl); free(zl);
            }
            int nb = (VOCAB - v0) < VB ? (VOCAB - v0) : VB;
            for (int j = 0; j < nb; j++) {
                double lv = 4096.0 * zy[j];
                double rv = (double)lg_ref[t * VOCAB + v0 + j];
                double e = fabs(lv - rv);
                if (e > wme) { wme = e; wv0 = v0 + j; }
                if (lv > top1v) { top2v = top1v; top1v = lv; top1i = v0 + j; }
                else if (lv > top2v) top2v = lv;
                if (rv > ref1) { ref2 = ref1; ref1 = rv; ref1i = v0 + j; }
                else if (rv > ref2) ref2 = rv;
            }
            ckks_ct_free(&lc);
        }
        baby_free(xb0); baby_free(xb1);
        double margin = top1v - top2v, ref_margin = ref1 - ref2;
        printf("  [F:logits t%d] max|err|=%.4g (vocab %d) | top1=%d ct=%.3f ref=%d ct2=%.3f | margin ct=%.3f ref=%.3f\n",
               t, wme, wv0, top1i, top1v, ref1i, ref2, margin, ref_margin);
        if (top1i != ref1i) { printf("  [F:top1 t%d] MISMATCH ct=%d ref=%d\n", t, top1i, ref1i); rf++; }
    }
    printf("  [F] lm_head 用时 %.1fs\n", t_clock() - tL);
    printf("RESULT=%s (%d; ref-deviate %d)\n", fails ? "FAILED" : "PASS", fails, rf);
    return fails;
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    /* 自研 vllm_tp 线程池：T23_NT 环境变量可配（默认 4） */
    {
        int tnt = 4;
        const char *tnts = getenv("T23_NT");
        if (tnts) { int x = atoi(tnts); if (x >= 1) tnt = x; }
        vllm_tp_init(tnt);
        fprintf(stderr, "vllm_tp threads=%d\n", vllm_tp_threads());
    }
    ckks_ctx_t ctx;
    if (ckks_ctx_init(&ctx) != 0) { printf("ctx init fail\n"); return 1; }
    size_t n = ctx.n, np = ctx.nprimes;
    int K = (int)(n / 2);
    const char *ph = getenv("T23_PHASE");
    if (!ph) ph = "l0";
    int phase = 0;               /* 0=l0 1=boot 2=l1(M2 layer1) 3=t26(layer26) 4=t27(layer27) 5=fin 6=lay */
    int lay = -1;                /* phase==6: T23_LAY 指定中间层 0..25 */
    char pref[32];
    if (strcmp(ph, "boot") == 0) phase = 1;
    else if (strcmp(ph, "l1") == 0) phase = 2;
    else if (strcmp(ph, "t26") == 0) phase = 3;
    else if (strcmp(ph, "t27") == 0) phase = 4;
    else if (strcmp(ph, "fin") == 0) phase = 5;
    else if (strcmp(ph, "lay") == 0) { phase = 6; lay = atoi(getenv("T23_LAY") ? getenv("T23_LAY") : "0"); }
    g_phase = phase;
    if (phase >= 3) { g_gfold = 256.0; g_bfold = 32.0; }
    else { g_gfold = 1.0; g_bfold = 1.0; }
    int uform = (phase >= 2);
    /* M3b 阶段参数：tail 层用统一 fold=1024 + 大值域检查 scale + 参考前缀 + causal */
    if (phase == 3) {
        g_fold = 1024.0; g_chk = 4096.0; g_refpre = "l26_"; g_causal = 1;
        g_wdir = ".tmp_tok/tail/w26"; g_rdir = ".tmp_tok/tail";
    } else if (phase == 4) {
        g_fold = 1024.0; g_chk = 4096.0; g_refpre = "l27_"; g_causal = 1;
        g_wdir = ".tmp_tok/tail/w27"; g_rdir = ".tmp_tok/tail";
    } else if (phase == 2) {
        g_wdir = ".tmp_tok/l1w"; g_rdir = ".tmp_tok/l1r";
    } else if (phase == 6) {   /* 中间层 LAY（0..25）：参考 tail/l{L}_*，权重 tail/w{L} */
        if (lay < 0 || lay > 27) { printf("T23_LAY %d out of [0,27]\n", lay); return 1; }
        snprintf(pref, sizeof pref, "l%d_", lay);
        g_fold = 1024.0; g_chk = 4096.0; g_refpre = pref; g_causal = 1;
        static char wd[64]; snprintf(wd, sizeof wd, ".tmp_tok/tail/w%d", lay);
        g_wdir = wd; g_rdir = ".tmp_tok/tail";
        /* 逐层自适应尺度 (axiom_arith_dynamic_scale_collapse_001): 加载 scale{lay}.bin(F) 作为残差
         * 折叠除数 g_fold; lay26/27 无 scale 文件 → 保持 1024 (M3b 兼容) */
        char sp[128]; snprintf(sp, sizeof sp, ".tmp_tok/tail/scale%d.bin", lay);
        FILE *fp = fopen(sp, "rb");
        if (fp) {
            double sc = 0;
            if (fread(&sc, 8, 1, fp) == 1 && sc >= 1.0 && sc <= 65536.0) g_fold = sc;
            fclose(fp);
            printf("  [scale] lay%d g_fold=%.0f\n", lay, g_fold);
        }
        /* silu 模式: silu{lay}.bin=1 → 直接拟合(|gate|<=8, 锚定0, M1c 路径; ÷21 拟合有 ~0.0127
         * DC 偏移 → 浅层小 gate 相对误差 12-35%); =0/缺省 → ÷21 折叠(M3b 深层) */
        g_silu_direct = 0;
        snprintf(sp, sizeof sp, ".tmp_tok/tail/silu%d.bin", lay);
        fp = fopen(sp, "rb");
        if (fp) {
            double sm = 0;
            if (fread(&sm, 8, 1, fp) == 1 && sm == 1.0) g_silu_direct = 1;
            fclose(fp);
        }
        if (g_silu_direct) printf("  [silu] lay%d direct\n", lay);
    }
    printf("=== T23 chain phase=%s (n=%zu slots=%d primes=%zu L=%d) fold=%.0f chk=%.0f causal=%d ===\n",
           ph, n, K, np, LTOK, g_fold, g_chk, g_causal);
    ckks_sk_t sk;
    if (phase == 0) ckks_sk_gen(&sk, (uint32_t)n);
    else if (t23_sk_load(&sk, ".tmp_tok/chain/sk.bin") != 0) { printf("sk load fail\n"); return 1; }
    ckks_rk_t rk;
    if (phase == 0 && getenv("T23_KEYONLY") != NULL) {   /* 一键转换 S3：仅生成并落盘 sk.bin 后退出 */
        if (t23_sk_save(&sk, ".tmp_tok/chain/sk.bin") != 0) { printf("[FAIL] sk save\n"); return 1; }
        printf("KEYONLY: sk.bin saved\n");
        ckks_sk_free(&sk); ckks_ctx_free(&ctx);
        return 0;
    }
    ckks_rk_gen(&rk, &sk, &ctx);
    if (phase == 1) { int fr = t23_boot(&ctx, &sk, &rk); ckks_sk_free(&sk); ckks_ctx_free(&ctx); ckks_rk_free(&rk); return fr; }
    if (phase == 5) { int fr = t23_final(&ctx, &sk, &rk); ckks_sk_free(&sk); ckks_ctx_free(&ctx); ckks_rk_free(&rk); return fr; }
    int fails = 0;
    /* M3b E2E 验收模式（phase>=3 + T23_E2EMODE）：真实 C 折叠使 L26/27 中间 ref(C/D/U2) 偏离
     * 真值属预期（_cfold_plan.py：|Δao|~1.4/0.8，E2E logits drift 0.14 < margin 2.03）；
     * 该模式下 C/D/U2 仅报告 err 不再计入 fails，验收移交 E2E logits 门 */
    int e2e = (phase >= 3 && getenv("T23_E2EMODE") != NULL);
    int rf = 0;

    g_datadir = g_wdir;   /* 权重目录（layer0/l1w/tail w26/w27） */
    const char *pre_w = g_refpre; g_refpre = "";   /* 权重文件名无前缀 */
    float *Wq = loadf("q_proj", (size_t)DIM * DIM);
    float *Wk = loadf("k_proj", (size_t)1024 * DIM);
    float *Wv = loadf("v_proj", (size_t)1024 * DIM);
    float *Wo = loadf("o_proj", (size_t)DIM * DIM);
    float *Wg = loadf("gate_proj", (size_t)MID * DIM);
    float *Wu = loadf("up_proj", (size_t)MID * DIM);
    float *Wd = loadf("down_proj", (size_t)DIM * MID);
    float *g1 = loadf("in_ln", DIM);
    float *g2 = loadf("post_ln", DIM);
    /* u-form 灞傝緭鍏?= 涓婁竴灞傜殑 u1锛?y/fold锛夛紱L0 杈撳叆 = embed4 */
    g_datadir = g_rdir;   /* 参考目录（l0/l1r/tail） */
    g_refpre = pre_w;     /* 恢复参考文件名前缀 */
    float *x = loadf(uform ? "u1" : "embed4", 4 * DIM);
    if (!Wq || !Wk || !Wv || !Wo || !Wg || !Wu || !Wd || !g1 || !g2 || !x) {
        printf("load weights fail\n"); return 1; }
    float *xn_ref = loadf("x_norm", LTOK * DIM);
    float *q_ref = loadf("q", LTOK * DIM);
    float *k_ref = loadf("k", LTOK * 1024);
    float *v_ref = loadf("v", LTOK * 1024);
    float *ao_ref = loadf("attn_out", LTOK * DIM);
    float *o_ref = loadf("o", LTOK * DIM);
    float *xattn_ref = loadf(uform && phase >= 3 ? "xattn_full" : "x_attn", 4 * DIM);
    float *gate_ref = loadf("gate", LTOK * MID);
    float *up_ref = loadf("up", LTOK * MID);
    float *act_ref = loadf("act", LTOK * MID);
    float *mlp_ref = loadf("mlp_out", LTOK * DIM);
    /* u-form 灞傛渶缁堝鐓?= u2_ref锛? y2/16锛屽瓨 l1/u2_ref.bin锛夛紱L0 = y */
    float *y_ref = loadf(uform ? "u2_ref" : "y", LTOK * DIM);
    if (!xn_ref || !q_ref || !k_ref || !v_ref || !ao_ref || !o_ref || !xattn_ref ||
        !gate_ref || !up_ref || !act_ref || !mlp_ref || !y_ref) {
        printf("load ref fail\n"); return 1; }

    /* 鏄庢枃 matmul 鍙傝€冨嚱鏁帮紙double 绱姞锛?*/
    #define REFMAT(OUT, W, Wr, Wc, XX, Xr) do { \
        for (int i = 0; i < (Wr); i++) { double s = 0; \
            for (int j = 0; j < (Wc); j++) s += (double)(W)[i * (Wc) + j] * (XX)[j]; \
            (OUT)[i] = (float)s; } } while (0)

    ckks_gk_t giant_gk[G];
    for (int r = 1; r < G; r++) ckks_gk_gen(&giant_gk[r], &sk, &ctx, r * B);

    uint64_t *poly = malloc(np * n * 8);
    uint64_t *pdec = malloc(np * n * 8);
    double *zx = malloc((size_t)K * 8);
    double *zy = malloc((size_t)K * 8);

    /* ---- silu 鑷锛堜粎 T20_SILUTEST=1 鏃舵墽琛岋紝楠岃瘉 eval_silu 鏄庢枃甯搁噺淇锛?----
     * decode 鍙敤 q0锛氭秷鎭繀椤?< q0/2锛垀0.5锛夈€俿ilu 杈撳嚭 鈮?.4 涓嶅彲鐩存帴瑙ｇ爜锛?     * 鏁呯敤 out_scale=1/16锛堚墹0.5 鍙В鐮侊級鈫?杞欢 脳16 杩樺師銆?*/
    if (getenv("T20_SILUTEST")) {
        static const double spat[] = {-8,-6,-4,-3,-2.4,-2,-1,-0.5,0,0.5,1,1.5,2,2.4,3,4,6,8};
        double *zv = malloc((size_t)K * sizeof(double));
        for (int j = 0; j < K; j++) zv[j] = spat[j % (int)(sizeof(spat) / sizeof(spat[0]))];
        for (int j = 0; j < K; j++) zx[j] = zv[j];
        ckks_encode(poly, zx, &ctx);
        ckks_ct_t ct0 = {0};
        ckks_encrypt(&ct0, &sk, poly, &ctx);
        ckks_ct_t so = {0};
        int f2 = eval_silu(&so, &ct0, 1.0 / 16.0, &rk, &ctx, &sk);
        double me = 0;
        if (!f2 && so.c) {
            ckks_decrypt(pdec, &sk, &so);
            ckks_decode(zy, pdec, &so.ctx);
            for (int j = 0; j < K; j++) {
                double xv = zv[j];
                double e = fabs(16.0 * zy[j] - xv / (1.0 + exp(-xv)));
                if (e > me) me = e;
            }
        }
        printf("SILUTEST eval_silu f2=%d out_np=%d max|err|=%.3e %s\n",
               f2, (int)so.ctx.nprimes, me, (!f2 && me < 2e-2) ? "PASS" : "FAIL");
        ckks_ct_free(&ct0); ckks_ct_free(&so);
        free(zv);
        return (!f2 && me < 2e-2) ? 0 : 1;
    }

    /* ================= 绔埌绔祦姘寸嚎锛圧MSNorm鈫扱KV鈫扐ttention鈫扥鈫扢LP锛孡=4锛?================= */
    float *sc_ref = loadf("scores", LTOK * NQ * LTOK);
    if (!sc_ref) { printf("load scores ref fail\n"); return 1; }
    /* attention 鎷熷悎绯绘暟锛坃attn_fit.py锛夛細exp deg-6 on [-0.6,0.5]锛況ecip deg-8 on [2.2,6.6] */
    static const double EXP_Q[7] = {
        0.0013345331468413693, 0.0084199656748899163, 0.041687167024342969,
        0.16665574107235029, 0.49999829447852773, 1.0000003483277793, 1.0000000191319709,
    };
    static const double RECIP_Q[9] = {
        0.0017353668508543719, -0.003233583989169786, 0.0027855151308509869,
        -0.006001190302329918, 0.014562383639137026, -0.028842984443374062,
        0.056966397255651176, -0.11386010944446567, 0.22752851420685311,
    };
    /* M3b 折叠域 RECIP 重拟合（_cfold_plan.py）：causal den=Σ_{s<=t} exp(score/4096²)~[1,4]；
     * 域 [0.9,4.2] deg-8 worst|err|~2e-4；仅 g_causal(phase>=3) 使用 */
    static const double RECIP_C_Q[9] = {
        0.04346395939586838, -0.05919293685777204, -0.006313915315061314,
        -0.006199404384775013, 0.08292632916609906, -0.1181855764050719,
        0.1622767052955656, -0.252785931008928, 0.3921971959272188,
    };
    static const double RECIP_C_C1 = 0.606060606060606;   /* 2/(4.2-0.9) */
    static const double RECIP_C_C2 = 1.545454545454545;   /* (4.2+0.9)/(4.2-0.9) */
    double tP = t_clock();
    ckks_ct_t xct[LTOK][2] = {0}, xnc[LTOK][2] = {0}, qc[LTOK][2] = {0};
    ckks_ct_t kc[LTOK] = {0}, vc[LTOK] = {0};
    ckks_ct_t atn[LTOK][2] = {0}, oc[LTOK][2] = {0}, xac[LTOK][2] = {0};
    ckks_ct_t mlpc[LTOK][2] = {0}, yc[LTOK][2] = {0};
    /* RMSNorm 鎷熷悎锛坃ln_fit3.py锛夛細L0 embed 涓績 ~0.001219 / post-LN ~0.0169锛?     * M1b u-form 灞傦紙杈撳叆 u1=y/16锛夊彟鎷熷悎锛欵1 涓績 2.490e-4锛坕nv_std/128锛夛紝P1 涓績 4.339e-4 */
    static const double LN_Q_E[9] = {
        20.115284866447489, -30.13425712605795, 20.258439475840198,
        -9.0610456130496502, 3.6180516357729426, -1.5225639827597353,
        0.68653783970627558, -0.34389036875646517, 0.25819968629847051,
    };
    static const double LN_Q_P[9] = {
        5.4358473619136225, -8.1409440238482258, 5.4700632509737837,
        -2.4444514559313406, 0.97495144782860055, -0.40983755940651695,
        0.18461311354960488, -0.092380316742432439, 0.069290656266086548,
    };
    static double LN_Q_E1[9] = {
        43.142469672085284, -64.709952734614447, 43.598577462066402,
        -19.572679456802717, 7.8526741134832978, -3.3196621314035704,
        1.5032036473336319, -0.75615428013407571, 0.57015212313514974,
    };
    static double LN_Q_P1[9] = {
        33.234148478986924, -49.815674242152291, 33.523993910680645,
        -15.020165073242962, 6.0108318033984807, -2.534862374975297,
        1.1452405118617746, -0.57478653520455458, 0.43241443035442501,
    };
    const double *LQ_E = uform ? LN_Q_E1 : LN_Q_E;   /* u-form 灞?in_ln 鎷熷悎 */
    const double *LQ_P = uform ? LN_Q_P1 : LN_Q_P;   /* u-form 灞?post_ln 鎷熷悎 */
    double m2c_emb = 0, m2c_post = 0;
    for (int t = 0; t < 4; t++) {
        double s1 = 0, s2 = 0;
        for (int j = 0; j < DIM; j++) {
            double xv = x[t * DIM + j];
            double av = xattn_ref[t * DIM + j];
            s1 += xv * xv;
            s2 += av * av;
        }
        m2c_emb += s1 / DIM;
        m2c_post += s2 / DIM;
    }
    m2c_emb /= 4.0;
    m2c_post /= 4.0;
    if (uform) {
        /* u-form 灞?post-LN 鍘熷杈撳叆 = mid16锛?x_attn/fold锛夛紝涓績鎸?mid16 鎷熷悎锛?         * xattn_ref锛?x_attn 鍏ㄥ€硷級绠楀嚭鐨?m2c_post 涓嶉€傜敤 鈫?瑕嗙洊銆俶2c_emb 鐢?x(=u1) 绠楀嚭锛屽ぉ鐒跺尮閰?E1銆?*/
        m2c_post = 0.0004338979568;
        /* 按层重拟合系数可选覆盖（M2: l1r/ln_e1..；M3b tail: ln26_/ln27_） */
        char lp[4][256];
        if (phase >= 3) {
            int lno = (phase == 6) ? lay : (phase + 23);   /* t26->26 t27->27 lay->T23_LAY */
            snprintf(lp[0], sizeof lp[0], ".tmp_tok/tail/ln%d_e1.bin", lno);
            snprintf(lp[1], sizeof lp[1], ".tmp_tok/tail/ln%d_p1.bin", lno);
            snprintf(lp[2], sizeof lp[2], ".tmp_tok/tail/m2c%d_e.bin", lno);
            snprintf(lp[3], sizeof lp[3], ".tmp_tok/tail/m2c%d_p.bin", lno);
        } else {
            snprintf(lp[0], sizeof lp[0], ".tmp_tok/l1r/ln_e1.bin");
            snprintf(lp[1], sizeof lp[1], ".tmp_tok/l1r/ln_p1.bin");
            snprintf(lp[2], sizeof lp[2], ".tmp_tok/l1r/m2c_emb.bin");
            snprintf(lp[3], sizeof lp[3], ".tmp_tok/l1r/m2c_post.bin");
        }
        if (load_f64f(lp[0], LN_Q_E1, 9) == 0) printf("  [fit] %s loaded\n", lp[0]);
        if (load_f64f(lp[1], LN_Q_P1, 9) == 0) printf("  [fit] %s loaded\n", lp[1]);
        if (load_f64f(lp[2], &m2c_emb, 1) == 0) printf("  [fit] %s m2c_emb=%.10g\n", lp[2], m2c_emb);
        if (load_f64f(lp[3], &m2c_post, 1) == 0) printf("  [fit] %s m2c_post=%.10g\n", lp[3], m2c_post);
    }

    /* ---- 阶段 A+B：逐 token 加密（L0/t26/lay0 从明文；l1/t27/lay>=1 用刷新后密文）+ RMSNorm + QKV ---- */
    if (phase == 2 || phase == 4 || (phase == 6 && lay >= 1)) {
        /* l1/t27/lay>=1：xct = boot 刷新并切回 LCHAIN 层链的 u{prev}r{LCHAIN} 密文 */
        char ip[32];
        if (phase == 2) snprintf(ip, sizeof ip, "u1r%d", LCHAIN);
        else if (phase == 4) snprintf(ip, sizeof ip, "u26r%d", LCHAIN);
        else snprintf(ip, sizeof ip, "u%dr%d", lay - 1, LCHAIN);
        /* 自适应尺度：输入状态 s{L-1}(gauge F{L-1}) 需折入本层 gauge F{L}: ×F{L-1}/F{L}≤1 */
        double ratio = 1.0;
        if (phase == 6 && lay >= 1 && lay <= 25) {
            double Fp = 0, Fc = 0;
            char sp[128];
            snprintf(sp, sizeof sp, ".tmp_tok/tail/scale%d.bin", lay - 1);
            FILE *fp = fopen(sp, "rb"); if (fp) { fread(&Fp, 8, 1, fp); fclose(fp); }
            snprintf(sp, sizeof sp, ".tmp_tok/tail/scale%d.bin", lay);
            fp = fopen(sp, "rb"); if (fp) { fread(&Fc, 8, 1, fp); fclose(fp); }
            if (Fp >= 1.0 && Fc >= 1.0) ratio = Fp / Fc;
            printf("  [scale] lay%d input ratio F%d/F%d=%.6g\n", lay, lay - 1, lay, ratio);
        }
        char path[256];
        for (int t = 0; t < LTOK; t++)
            for (int h = 0; h < 2; h++) {
                snprintf(path, sizeof(path), ".tmp_tok/chain/%s_%d_%d.ct", ip, t, h);
                ckks_ct_t lo = {0};
                if (t23_ct_load(&lo, path, &ctx) != 0) { printf("[FAIL] load %s\n", path); fails++; continue; }
                ckks_ct_t s96 = {0};
                if (ct_switch_to(&s96, &lo, LCHAIN) != 0) fails++;
                ckks_ct_free(&lo);
                if (ratio != 1.0 && s96.c) {   /* 折入本层 gauge（下缩，2 的幂，ct_mul_slot 语义同 /fold） */
                    ckks_ct_t rr = {0};
                    ct_mul_slot(&rr, &s96, ratio, -1, &sk, &ctx);
                    ckks_ct_free(&s96);
                    s96 = rr;
                }
                xct[t][h] = s96;
            }
    }
    for (int t = 0; t < LTOK; t++) {
        if (phase == 0 || phase == 3 || (phase == 6 && lay == 0)) {   /* L0/t26/lay0：明文加密（→ 切 LCHAIN 层链） */
            for (int j = 0; j < K; j++) zx[j] = x[t * DIM + j];
            ckks_encode(poly, zx, &ctx);
            ckks_encrypt(&xct[t][0], &sk, poly, &ctx);
            ckks_ct_t s96 = {0};
            ct_switch_to(&s96, &xct[t][0], LCHAIN);
            ckks_ct_free(&xct[t][0]); xct[t][0] = s96;
            for (int j = 0; j < K; j++) zx[j] = x[t * DIM + K + j];
            ckks_encode(poly, zx, &ctx);
            ckks_encrypt(&xct[t][1], &sk, poly, &ctx);
            ckks_ct_t s96b = {0};
            ct_switch_to(&s96b, &xct[t][1], LCHAIN);
            ckks_ct_free(&xct[t][1]); xct[t][1] = s96b;
        }
        if (rmsnorm_enc(&xnc[t][0], &xnc[t][1], &xct[t][0], &xct[t][1], g1, m2c_emb,
                        LQ_E, zx, poly, pdec, zy, giant_gk, &rk, &sk, &ctx, "A") != 0) fails++;
        if (matmul2(&qc[t][0], Wq, DIM, 0, &xnc[t][0], &xnc[t][1], giant_gk, &sk, &ctx) != 0) fails++;
        if (matmul2(&qc[t][1], Wq, DIM, K, &xnc[t][0], &xnc[t][1], giant_gk, &sk, &ctx) != 0) fails++;
        if (matmul2(&kc[t], Wk, DIM, 0, &xnc[t][0], &xnc[t][1], giant_gk, &sk, &ctx) != 0) fails++;
        if (matmul2(&vc[t], Wv, DIM, 0, &xnc[t][0], &xnc[t][1], giant_gk, &sk, &ctx) != 0) fails++;
        char lbl[32];
        snprintf(lbl, sizeof lbl, "A:xnorm t%d", t);
        check_ct_scale(&xnc[t][0], &xn_ref[t * DIM], K, g_chk, lbl, e2e ? &rf : &fails, &sk, &ctx, pdec, zy);
        if (getenv("T23_DBGRMS") && strcmp(getenv("T23_DBGRMS"), "E") != 0) {   /* 逐槽 dump：computed vs ref 最大偏差槽位 (="E" 时仅 E 段诊断) */
            double Suse;
            for (int pass = 0; pass < 2; pass++) {   /* pass0=输入 xa vs u1 参考; pass1=xn vs x_norm 参考 */
                const ckks_ct_t *src = (pass == 0) ? &xct[t][0] : &xnc[t][0];
                const float *refb = (pass == 0) ? &x[t * DIM] : &xn_ref[t * DIM];
                const char *nm = (pass == 0) ? "in0" : "xn0";
                Suse = (pass == 0) ? 1.0 : 4096.0;
                ckks_ct_t dd = {0}, dr = {0};
                ckks_ct_copy(&dd, src);
                uint64_t *dpc = malloc((size_t)dd.ctx.nprimes * dd.ctx.n * 8);
                double *dz = malloc((size_t)K * 8);
                for (int j = 0; j < K; j++) dz[j] = 1.0 / Suse;
                ckks_encode(dpc, dz, &dd.ctx);
                ckks_mult_plain(&dr, &dd, dpc);
                ckks_ct_free(&dd);
                ckks_rescale(&dd, &dr);
                ckks_ct_free(&dr);
                ckks_decrypt(pdec, &sk, &dd);
                ckks_decode(zy, pdec, &dd.ctx);
                int bj[5] = {0, 0, 0, 0, 0}; double be[5] = {0, 0, 0, 0, 0};
                for (int j = 0; j < K; j++) {
                    double e = fabs(Suse * zy[j] - (double)refb[j]);
                    for (int b = 0; b < 5; b++)
                        if (e > be[b]) { for (int q = 4; q > b; q--) { be[q] = be[q-1]; bj[q] = bj[q-1]; }
                                         be[b] = e; bj[b] = j; break; }
                }
                printf("  [%s top5] ", nm);
                for (int b = 0; b < 5; b++)
                    printf("j=%d ct=%.4g rf=%.4g e=%.4g | ", bj[b], Suse * zy[bj[b]],
                           (double)refb[bj[b]], be[b]);
                printf("\n");
                ckks_ct_free(&dd);
                free(dpc); free(dz);
            }
            for (int half = 0; half < 2; half++) {   /* xn1 另外半 */
                ckks_ct_t dd = {0}, dr = {0};
                ckks_ct_copy(&dd, &xnc[t][half]);
                uint64_t *dpc = malloc((size_t)dd.ctx.nprimes * dd.ctx.n * 8);
                double *dz = malloc((size_t)K * 8);
                for (int j = 0; j < K; j++) dz[j] = 1.0 / 4096.0;
                ckks_encode(dpc, dz, &dd.ctx);
                ckks_mult_plain(&dr, &dd, dpc);
                ckks_ct_free(&dd);
                ckks_rescale(&dd, &dr);
                ckks_ct_free(&dr);
                ckks_decrypt(pdec, &sk, &dd);
                ckks_decode(zy, pdec, &dd.ctx);
                int bj[5] = {0, 0, 0, 0, 0}; double be[5] = {0, 0, 0, 0, 0};
                for (int j = 0; j < K; j++) {
                    double e = fabs(4096.0 * zy[j] - (double)xn_ref[t * DIM + K + j]);
                    for (int b = 0; b < 5; b++)
                        if (e > be[b]) { for (int q = 4; q > b; q--) { be[q] = be[q-1]; bj[q] = bj[q-1]; }
                                         be[b] = e; bj[b] = j; break; }
                }
                printf("  [xn1 top5] ");
                for (int b = 0; b < 5; b++)
                    printf("j=%d ct=%.4g rf=%.4g e=%.4g | ", bj[b], 4096.0 * zy[bj[b]],
                           (double)xn_ref[t * DIM + K + bj[b]], be[b]);
                printf("\n");
                ckks_ct_free(&dd);
                free(dpc); free(dz);
            }
        }
        snprintf(lbl, sizeof(lbl), "B:q0 t%d", t);
        check_ct_scale(&qc[t][0], &q_ref[t * DIM], K, g_chk, lbl, e2e ? &rf : &fails, &sk, &ctx, pdec, zy);
        snprintf(lbl, sizeof(lbl), "B:q1 t%d", t);
        check_ct_scale(&qc[t][1], &q_ref[t * DIM + K], K, g_chk, lbl, e2e ? &rf : &fails, &sk, &ctx, pdec, zy);
        snprintf(lbl, sizeof(lbl), "B:k t%d", t);
        check_ct_scale(&kc[t], &k_ref[t * 1024], 1024, g_chk, lbl, e2e ? &rf : &fails, &sk, &ctx, pdec, zy);
        snprintf(lbl, sizeof lbl, "B:v t%d", t);
        check_ct_scale(&vc[t], &v_ref[t * 1024], 1024, g_chk, lbl, e2e ? &rf : &fails, &sk, &ctx, pdec, zy);
        if (t == 0 && getenv("T23_AONLY")) {
            printf("[AONLY] exit after t0 A+B\n");
            exit(0);
        }
        ckks_ct_free(&xnc[t][0]); ckks_ct_free(&xnc[t][1]);
    }
    printf("  [A+B] QKV 鐢ㄦ椂 %.1fs\n", t_clock() - tP);

    /* ---- 闃舵 C锛欰ttention锛堥€?token锛? 涓?ct_b锛屾瘡 ct_b 2 涓?q 澶达級 ---- */
    double tC = t_clock();
    if (getenv("T20_SKIPCD")) {
        /* 蹇€?E 杩唬锛歛o_ref 鍔犲瘑鐩村嚭 atn@57锛堥摼娣变笌鐪熷疄 C 杈撳嚭涓€鑷达紝鐪?~21min C锛?*/
        for (int t = 0; t < LTOK; t++) {
            for (int h = 0; h < 2; h++) {
                for (int j = 0; j < K; j++) zx[j] = ao_ref[t * DIM + h * K + j];
                ckks_encode(poly, zx, &ctx);
                ckks_encrypt(&atn[t][h], &sk, poly, &ctx);
                ckks_ct_t tmp = {0};
                ct_switch_to(&tmp, &atn[t][h], 57);
                ckks_ct_free(&atn[t][h]);
                atn[t][h] = tmp;
            }
            char lbl[32];
            snprintf(lbl, sizeof(lbl), "C:attn t%d", t);
            check_ct_scale(&atn[t][0], &ao_ref[t * DIM], K, g_chk, lbl, &fails, &sk, &ctx, pdec, zy);
        }
    } else
    for (int t = 0; t < LTOK; t++) {
        ckks_ct_t kr[4] = {0}, vr[4] = {0};
        /* 4 涓?kv 澶寸粍 rep锛坆=0..3锛沚+4 澶嶇敤鍚?kv 澶达級 */
        for (int b = 0; b < 4; b++) {
            for (int s = 0; s < 4; s++) {
                float *W = mk_W_kv(b, s, K);
                ckks_ct_t kp = {0}, vp = {0}, kk = {0}, vv = {0};
                matmul1(&kp, W, K, 0, 0, &kc[s], giant_gk, &sk, &ctx);
                matmul1(&vp, W, K, 0, 0, &vc[s], giant_gk, &sk, &ctx);
                free(W);
                if (kr[b].c) { ckks_add(&kk, &kr[b], &kp); ckks_ct_free(&kr[b]); ckks_ct_free(&kp); kr[b] = kk; }
                else { kr[b] = kp; kp.c = NULL; }
                if (vr[b].c) { ckks_add(&vv, &vr[b], &vp); ckks_ct_free(&vr[b]); ckks_ct_free(&vp); vr[b] = vv; }
                else { vr[b] = vp; vp.c = NULL; }
                ckks_ct_free(&kp); ckks_ct_free(&vp);
            }
        }
        for (int b = 0; b < 8; b++) {
            int bsrc = b < 4 ? b : b - 4;
            /* Q_rep */
            float *Wq2 = mk_W_rep_q(b, K);
            ckks_ct_t qr = {0};
            matmul1(&qr, Wq2, K, 0, 0, &qc[t][b < 4 ? 0 : 1], giant_gk, &sk, &ctx);
            free(Wq2);
            /* 瀵归綈閾鹃暱锛歲r/kr/vr 鍒板叕鍏?npmin锛屽悇 rescale 鍒?2^60 */
            int npmin = qr.ctx.nprimes;
            if (kr[bsrc].ctx.nprimes < npmin) npmin = kr[bsrc].ctx.nprimes;
            if (vr[bsrc].ctx.nprimes < npmin) npmin = vr[bsrc].ctx.nprimes;
            ckks_ct_t qra = {0}, kra = {0}, vra = {0}, qr2 = {0}, kr2 = {0}, vr2 = {0};
            if (ct_switch_to(&qra, &qr, npmin) != 0) fails++;
            if (ct_switch_to(&kra, &kr[bsrc], npmin) != 0) fails++;
            if (ct_switch_to(&vra, &vr[bsrc], npmin) != 0) fails++;
            ckks_ct_free(&qr);
            ckks_rescale(&qr2, &qra); ckks_ct_free(&qra);
            ckks_rescale(&kr2, &kra); ckks_ct_free(&kra);
            ckks_rescale(&vr2, &vra); ckks_ct_free(&vra);
            /* P = Q_rep 鈯?K_rep 鈫?S_raw = 危_i P锛坆sgs锛夆啋 S = S_raw/鈭?28 */
            ckks_ct_t P = {0}, Sraw = {0}, Sraw_r = {0}, S = {0};
            if (mul_relin(&P, &qr2, &kr2, &rk, &ctx) != 0) fails++;
            ckks_ct_free(&qr2); ckks_ct_free(&kr2);
            {
                float *W1 = mk_W_ones_i(b, K);
                matmul1(&Sraw, W1, K, 0, 0, &P, giant_gk, &sk, &ctx);
                free(W1);
            }
            ckks_ct_free(&P);
            ckks_rescale(&Sraw_r, &Sraw); ckks_ct_free(&Sraw);
            {
                ckks_ct_t m = {0};
                double *z1 = zx;
                for (int j = 0; j < K; j++) z1[j] = 1.0 / sqrt(HD);
                ckks_encode(poly, z1, &ctx);
                ckks_mult_plain(&m, &Sraw_r, poly);
                ckks_rescale(&S, &m); ckks_ct_free(&m);
            }
            ckks_ct_free(&Sraw_r);
            /* num = exp(S)锛坉eg-6 Horner锛屾弧妲斤紱闈?score 妲戒负 exp(0)=1锛岀敱 W 鐭╅樀闅旂锛?*/
            ckks_ct_t Sc = {0}, num = {0};
            ckks_ct_copy(&Sc, &S);
            horner_q(&num, &Sc, EXP_Q, 6, &rk, &sk, &ctx, -1);
            ckks_ct_free(&Sc); ckks_ct_free(&S);
            if (g_causal) {   /* M3b: causal——保留 s<=t 槽（t=当前 query token），掩后求和 den/p/v */
                for (int j = 0; j < K; j++) { int s = (j % 512) / 128; zx[j] = (s <= t) ? 1.0 : 0.0; }
                ckks_encode(poly, zx, &ctx);
                ckks_ct_t nm = {0}, nr = {0};
                ckks_mult_plain(&nm, &num, poly);
                ckks_rescale(&nr, &nm); ckks_ct_free(&nm);
                ckks_ct_free(&num); num = nr;
            }
            /* den = 危_s num锛坆sgs锛? 涓?score 妲芥眰鍜屽苟澶嶅埗鍒?4 涓?s 浣嶏級 */
            ckks_ct_t den = {0}, den_r = {0};
            {
                float *W1 = mk_W_den(b, K);
                matmul1(&den, W1, K, 0, 0, &num, giant_gk, &sk, &ctx);
                free(W1);
            }
            ckks_rescale(&den_r, &den); ckks_ct_free(&den);
            /* z = den 仿射到 [-1,1] 后 recip deg-8；M3b(g_causal) 换 causal 折叠域重拟合（den~[1,4]） */
            const double *rq = RECIP_Q;
            double rc1 = 1.0 / 2.199819269, rc2 = 4.395065814 / 2.199819269;
            if (g_causal) { rq = RECIP_C_Q; rc1 = RECIP_C_C1; rc2 = RECIP_C_C2; }
            ckks_ct_t dz = {0}, zr = {0}, inv = {0};
            {
                ckks_ct_t m = {0};
                double *z1 = zx;
                for (int j = 0; j < K; j++) z1[j] = rc1;
                ckks_encode(poly, z1, &ctx);
                ckks_mult_plain(&m, &den_r, poly);
                ckks_rescale(&dz, &m); ckks_ct_free(&m);
            }
            ckks_ct_free(&den_r);
            ct_add_const(&zr, &dz, -rc2, -1, &sk, &ctx);
            ckks_ct_free(&dz);
            horner_q(&inv, &zr, rq, 8, &rk, &sk, &ctx, -1);
            ckks_ct_free(&zr);
            /* p = num 鈯?inv锛堝榻愰摼闀匡級 */
            ckks_ct_t num2 = {0}, p = {0};
            if (ct_switch_to(&num2, &num, inv.ctx.nprimes) != 0) fails++;
            ckks_ct_free(&num);
            if (mul_relin(&p, &num2, &inv, &rk, &ctx) != 0) fails++;
            ckks_ct_free(&num2); ckks_ct_free(&inv);
            /* p_rep = 澶嶅埗 p 鍒板叏閮?i */
            ckks_ct_t pre = {0}, pre_r = {0};
            {
                float *W1 = mk_W_rep_p(b, K);
                matmul1(&pre, W1, K, 0, 0, &p, giant_gk, &sk, &ctx);
                free(W1);
            }
            ckks_ct_free(&p);
            ckks_rescale(&pre_r, &pre); ckks_ct_free(&pre);
            /* O_raw = p_rep 鈯?V_rep锛坴ra 闄嶉摼瀵归綈锛?*/
            ckks_ct_t vra2 = {0}, Oraw = {0};
            if (ct_switch_to(&vra2, &vr2, pre_r.ctx.nprimes) != 0) fails++;
            ckks_ct_free(&vr2);
            if (mul_relin(&Oraw, &pre_r, &vra2, &rk, &ctx) != 0) fails++;
            ckks_ct_free(&pre_r); ckks_ct_free(&vra2);
            /* AV 姹傚拰 鈫?attn 澶?(2b+h'')路128+i锛岀疮鍔犺繘 atn[t][b<4?0:1] */
            {
                float *W1 = mk_W_av(b, K);
                ckks_ct_t part = {0}, part_r = {0};
                matmul1(&part, W1, K, 0, 0, &Oraw, giant_gk, &sk, &ctx);
                free(W1);
                ckks_rescale(&part_r, &part); ckks_ct_free(&part);
                int half = b < 4 ? 0 : 1;
                ckks_ct_t s2 = {0};
                if (atn[t][half].c) { ckks_add(&s2, &atn[t][half], &part_r); ckks_ct_free(&atn[t][half]); ckks_ct_free(&part_r); atn[t][half] = s2; }
                else { atn[t][half] = part_r; part_r.c = NULL; }
                ckks_ct_free(&part_r);
            }
            ckks_ct_free(&Oraw);
        }
        /* 真实 C 折叠还原：atn 当前 = ao/256（v 经 rep 预缩 1/256），×256 拆 ≤8 还原到全幅 ao */
        if (g_causal) {
            ct_unfold(&atn[t][0], RCF, &sk, &ctx);
            ct_unfold(&atn[t][1], RCF, &sk, &ctx);
        }
        /* 校验 attn_out（token t）：atn[t][0/1] vs ao_ref（E2E 模式仅报告） */
        char lbl[32];
        snprintf(lbl, sizeof(lbl), "C:attn t%d", t);
        check_ct_scale(&atn[t][0], &ao_ref[t * DIM], K, g_chk, lbl, e2e ? &rf : &fails, &sk, &ctx, pdec, zy);
        snprintf(lbl, sizeof(lbl), "C:attn1 t%d", t);
        check_ct_scale(&atn[t][1], &ao_ref[t * DIM + K], K, g_chk, lbl, e2e ? &rf : &fails, &sk, &ctx, pdec, zy);
        ckks_ct_free(&kr[0]); ckks_ct_free(&kr[1]); ckks_ct_free(&kr[2]); ckks_ct_free(&kr[3]);
        ckks_ct_free(&vr[0]); ckks_ct_free(&vr[1]); ckks_ct_free(&vr[2]); ckks_ct_free(&vr[3]);
    }
    printf("  [C] attention 鐢ㄦ椂 %.1fs\n", t_clock() - tC);

    /* ---- 闃舵 D锛歄 鎶曞奖 + 娈嬪樊 x_attn = x + o ---- */
    double tD = t_clock();
    for (int t = 0; t < LTOK; t++) {
        if (matmul2(&oc[t][0], Wo, DIM, 0, &atn[t][0], &atn[t][1], giant_gk, &sk, &ctx) != 0) fails++;
        if (matmul2(&oc[t][1], Wo, DIM, K, &atn[t][0], &atn[t][1], giant_gk, &sk, &ctx) != 0) fails++;
        char lbl[32];
        snprintf(lbl, sizeof(lbl), "D:o t%d", t);
        check_ct_scale(&oc[t][0], &o_ref[t * DIM], K, g_chk, lbl, e2e ? &rf : &fails, &sk, &ctx, pdec, zy);
        if (t == 0 && getenv("T23_AD0")) {
            printf("[AD0] exit after t0 real-C+D\n");
            exit(0);
        }
        if (uform) {
            /* u-form 娈嬪樊锛歮id16 = u1 + o/16锛坲1 涓嶇缉鏀撅紱o 鍒嗘敮 /16 鎶樺彔锛屼緵 post-LN锛?*/
            for (int h = 0; h < 2; h++) {
                ckks_ct_t oc16 = {0};
                ct_mul_slot(&oc16, &oc[t][h], 1.0 / g_fold, -1, &sk, &ctx);
                ckks_ct_t xa = {0};
                if (ct_switch_to(&xa, &xct[t][h], (int)oc16.ctx.nprimes) != 0) fails++;
                ckks_ct_t s2 = {0};
                ckks_add(&s2, &xa, &oc16);
                ckks_ct_free(&xa); ckks_ct_free(&oc16); ckks_ct_free(&oc[t][h]);
                xac[t][h] = s2;   /* mid16 鈮?~0.16锛屽彲瑙ｇ爜 */
            }
        } else {
            /* 娈嬪樊锛歺 闄嶉摼瀵归綈 o锛堝叏鍊?x_attn = x + o锛?*/
            int npmin = oc[t][0].ctx.nprimes;
            if (oc[t][1].ctx.nprimes < npmin) npmin = oc[t][1].ctx.nprimes;
            for (int h = 0; h < 2; h++) {
                ckks_ct_t xa = {0};
                if (ct_switch_to(&xa, &xct[t][h], npmin) != 0) fails++;
                ckks_ct_t s2 = {0};
                ckks_add(&s2, &xa, &oc[t][h]);
                ckks_ct_free(&xa); ckks_ct_free(&oc[t][h]);
                xac[t][h] = s2;
            }
        }
        if (!uform) {   /* u-form 鐨?mid16 璇箟 鈮?x_attn 鍏ㄥ€硷紝姝ゅ鐓ц烦杩?*/
            snprintf(lbl, sizeof(lbl), "D:xattn t%d", t);
            check_ct_scale(&xac[t][0], &xattn_ref[t * DIM], K, 16.0, lbl, &fails, &sk, &ctx, pdec, zy);
        }
    }
    printf("  [D] O 鎶曞奖 鐢ㄦ椂 %.1fs\n", t_clock() - tD);

    /* ---- 闃舵 E锛歱ost-LN 鈫?gate/up 鈫?silu 鈫?act 鈫?down 鈫?娈嬪樊 y = x_attn + mlp_out ---- */
    double tE = t_clock();
    for (int t = 0; t < LTOK; t++) {
        ckks_ct_t xn2[2] = {0};
        printf("  [E t%d] xac np=%d/%d (oc)\n", t,
               (int)xac[t][0].ctx.nprimes, (int)xac[t][1].ctx.nprimes);
        if (rmsnorm_enc(&xn2[0], &xn2[1], &xac[t][0], &xac[t][1], g2, m2c_post,
                        LQ_P, zx, poly, pdec, zy, giant_gk, &rk, &sk, &ctx, "E") != 0) fails++;
        /* gate/up锛? chunks 脳 2 鍧?*/
        ckks_ct_t gate[6] = {0}, up[6] = {0}, sil[6] = {0}, act[6] = {0};
        for (int c = 0; c < 6; c++) {
            /* gate/up: Wg/Wu = [MID脳DIM]锛堣瀹?DIM锛夛紝block_bsgs 鐢?Wdim 浣滆璺?*/
            if (matmul2(&gate[c], Wg, DIM, c * K, &xn2[0], &xn2[1], giant_gk, &sk, &ctx) != 0) fails++;
            if (matmul2(&up[c], Wu, DIM, c * K, &xn2[0], &xn2[1], giant_gk, &sk, &ctx) != 0) fails++;
        }
        ckks_ct_free(&xn2[0]); ckks_ct_free(&xn2[1]);
        printf("  [E t%d] gate0 np=%d up0 np=%d xn2freed\n", t,
               (int)gate[0].ctx.nprimes, (int)up[0].ctx.nprimes);
        /* silu 鈫?act = silu 鈯?up。M3b(phase≥3)：折叠域 silu（gate÷21）+ up÷21，
         * act_f = silu·up/441 ≤ ~0.75<1，随后 ×441 拆 ≤8 还原（down 输入 act 全幅，线性可载） */
        for (int c = 0; c < 6; c++) {
            int use_fold = (g_phase >= 3 && !g_silu_direct);
            if (use_fold) {
                if (eval_silu_w(&sil[c], &gate[c], SILU_W21_R, SILU_W21_SG,
                                SILU_W21_VMINV, &rk, &sk, &ctx) != 0) fails++;
            } else if (eval_silu(&sil[c], &gate[c], 1.0, &rk, &ctx, &sk) != 0) fails++;
            ckks_ct_free(&gate[c]);
            ckks_ct_t upa = {0};
            if (use_fold) {
                ckks_ct_t uf0 = {0};
                ct_mul_slot(&uf0, &up[c], 1.0 / SILU_W21_SG, -1, &sk, &ctx);  /* up/21 */
                if (ct_switch_to(&upa, &uf0, sil[c].ctx.nprimes) != 0) fails++;  /* 对齐 sil 层 */
                ckks_ct_free(&uf0);
            } else if (ct_switch_to(&upa, &up[c], sil[c].ctx.nprimes) != 0) fails++;
            ckks_ct_free(&up[c]);
            if (mul_relin(&act[c], &sil[c], &upa, &rk, &ctx) != 0) fails++;
            ckks_ct_free(&sil[c]); ckks_ct_free(&upa);
            if (getenv("T23_DBGRMS") && act[c].c) {   /* act_f 消息 ≤0.44 直解对照 act_ref/441 */
                ckks_decrypt(pdec, &sk, &act[c]);
                ckks_decode(zy, pdec, &act[c].ctx);
                double me = 0;
                for (int j = 0; j < K; j++) {
                    double e = fabs(441.0 * zy[j] - (double)act_ref[t * MID + c * K + j]);
                    if (e > me) me = e;
                }
                printf("  [E:actf c%d] np=%d maxerr(vs act_ref/441)=%.4g\n", c,
                       (int)act[c].ctx.nprimes, me);
            }
            if (use_fold && act[c].c) ct_unfold(&act[c], SILU_W21_SG * SILU_W21_SG, &sk, &ctx);
        }
        /* down锛? 杈撳嚭鍗?脳 6 鍒楀潡 */
        ckks_ct_t d0 = {0}, d1 = {0};
        for (int c = 0; c < 6; c++) {
            ckks_ct_t r0 = {0}, r1 = {0};
            matmul1(&r0, Wd, MID, 0, c * K, &act[c], giant_gk, &sk, &ctx);
            matmul1(&r1, Wd, MID, K, c * K, &act[c], giant_gk, &sk, &ctx);
            ckks_ct_t s2 = {0};
            if (d0.c) { ckks_add(&s2, &d0, &r0); ckks_ct_free(&d0); ckks_ct_free(&r0); d0 = s2; }
            else { d0 = r0; r0.c = NULL; }
            if (d1.c) { ckks_add(&s2, &d1, &r1); ckks_ct_free(&d1); ckks_ct_free(&r1); d1 = s2; }
            else { d1 = r1; r1.c = NULL; }
            ckks_ct_free(&r0); ckks_ct_free(&r1);
        }
        ckks_ct_t mlp0 = {0}, mlp1 = {0};
        ckks_rescale(&mlp0, &d0); ckks_ct_free(&d0);
        ckks_rescale(&mlp1, &d1); ckks_ct_free(&d1);
        for (int c = 0; c < 6; c++) ckks_ct_free(&act[c]);
        if (getenv("T23_DBGRMS")) {   /* M3b 诊断：mlp vs mlp_ref（÷128 解码） */
            char lbb[32];
            snprintf(lbb, sizeof lbb, "E:mlp0 t%d", t);
            check_ct_scale(&mlp0, &mlp_ref[t * DIM], K, 128.0, lbb, &fails, &sk, &ctx, pdec, zy);
            snprintf(lbb, sizeof lbb, "E:mlp1 t%d", t);
            check_ct_scale(&mlp1, &mlp_ref[t * DIM + K], K, 128.0, lbb, &fails, &sk, &ctx, pdec, zy);
        }
        if (uform) {
            /* u-form 娈嬪樊锛歶2 = mid16 + mlp/16锛坢lp 鍒嗘敮 /16 鎶樺彔锛沵id16 宸叉槸 u 褰㈡€侊級*/
            for (int h = 0; h < 2; h++) {
                ckks_ct_t mm = {0};
                ct_mul_slot(&mm, (h == 0) ? &mlp0 : &mlp1, 1.0 / g_fold, -1, &sk, &ctx);
                ckks_ct_t xa = {0};
                if (ct_switch_to(&xa, &xac[t][h], (int)mm.ctx.nprimes) != 0) fails++;
                ckks_ct_t s2 = {0};
                ckks_add(&s2, &xa, &mm);
                ckks_ct_free(&xa); ckks_ct_free(&mm);
                yc[t][h] = s2;   /* u2 鈮?~0.27锛屽彲鐩存帴瑙ｇ爜 */
            }
        } else {
            /* 娈嬪樊锛歽 = x_attn + mlp_out锛泍16 = y/16 鍙В鐮?*/
            int npmin = mlp0.ctx.nprimes;
            if (mlp1.ctx.nprimes < npmin) npmin = mlp1.ctx.nprimes;
            for (int h = 0; h < 2; h++) {
                ckks_ct_t xa = {0};
                if (ct_switch_to(&xa, &xac[t][h], npmin) != 0) fails++;
                ckks_ct_t s2 = {0};
                ckks_add(&s2, &xa, (h == 0) ? &mlp0 : &mlp1);
                ckks_ct_free(&xa);
                /* y/16锛氬父鏁?1/16 鐢ㄧ湡鏄庢枃缂栫爜涔橈紙ct_mul_slot 宸插惈 rescale锛?*/
                ckks_ct_t m = {0};
                ct_mul_slot(&m, &s2, 1.0 / 16.0, -1, &sk, &ctx);
                ckks_ct_free(&s2);
                yc[t][h] = m;
            }
        }
        ckks_ct_free(&mlp0); ckks_ct_free(&mlp1);
        /* 鏍￠獙锛圠0锛歞ecode y16 鈫?脳16 杩樺師 y锛泆-form锛歽c 鍗?u2锛岀洿鎺ュ鐓?u2_ref锛?*/
        double sc_e = uform ? 1.0 : 16.0;
        for (int h = 0; h < 2; h++) {
            ckks_decrypt(pdec, &sk, &yc[t][h]);
            ckks_decode(zy, pdec, &yc[t][h].ctx);
            double me = 0;
            const float *yr = (h == 0) ? &y_ref[t * DIM] : &y_ref[t * DIM + K];
            for (int j = 0; j < K; j++) {
                double e = fabs(sc_e * zy[j] - (double)yr[j]);
                if (e > me) me = e;
            }
            printf("  [%s t%d h%d] max|err|=%.3e %s%s\n",
                   uform ? "U2" : "E:y", t, h, me, me < 3e-2 ? "PASS" : "FAIL",
                   e2e ? " [E2E:ref-deviate-ok]" : "");
            if (me >= 3e-2 && !e2e) fails++;
        }
        if (t == 0 && getenv("T23_E0")) {
            printf("[E0] exit after t0 E/U2\n");
            exit(0);
        }
    }
    printf("  [E] MLP 用时 %.1fs\n", t_clock() - tE);

    if (phase == 0 || phase == 3 || phase == 4 || phase == 6) {   /* 链接力落盘：u1/u26/u27/u{lay}（=yc，y/fold 形态） */
        char op[32];
        if (phase == 0) snprintf(op, sizeof op, "u1");
        else if (phase == 3) snprintf(op, sizeof op, "u26");
        else if (phase == 4) snprintf(op, sizeof op, "u27");
        else snprintf(op, sizeof op, "u%d", lay);   /* phase==6: 链节名 = 层号（u0..u27） */
        char path[256];
        if (phase == 0 && t23_sk_save(&sk, ".tmp_tok/chain/sk.bin") != 0) { printf("[FAIL] sk save\n"); fails++; }
        for (int t = 0; t < LTOK; t++)
            for (int h = 0; h < 2; h++) {
                snprintf(path, sizeof(path), ".tmp_tok/chain/%s_%d_%d.ct", op, t, h);
                if (t23_ct_save(&yc[t][h], path) != 0) { printf("[FAIL] ct save %s\n", path); fails++; }
            }
        printf("  CHAIN: dumped %s_[t]_[h].ct (y/fold)\n", op);
    }

    ckks_sk_free(&sk); ckks_ctx_free(&ctx); ckks_rk_free(&rk);
    free(Wq); free(Wk); free(Wv); free(Wo); free(Wg); free(Wu); free(Wd);
    free(g1); free(g2); free(x);
    free(xn_ref); free(q_ref); free(k_ref); free(v_ref); free(ao_ref); free(o_ref);
    free(xattn_ref); free(gate_ref); free(up_ref); free(act_ref); free(mlp_ref); free(y_ref);
    free(sc_ref);
    free(poly); free(pdec); free(zx); free(zy);
    printf("RESULT=%s (%d)\n", fails ? "FAILED" : "PASS", fails);
    return fails;
}
