/* verify_layer.c — 独立验证某层密文"算得对不对"（不依赖驱动的判据代码路径）
 *
 * 原理：用 sk.bin 解密该层的 .ct，解码成实数槽位，与**明文参考**逐元素对照，算 max|err|。
 *       参考来自 float32 明文前向（tail/l{L}_u2_ref.bin 等），与密文链完全独立。
 *
 * 两种模式：
 *   A) 准确度校验（默认）
 *      verify_layer -L 4
 *        → 读 chain/u4_{t}_{h}.ct（8 个），对照 tail/l4_u2_ref.bin，容差 3e-2
 *      verify_layer -Ct u4r -L 4          # 也可校验 boot 输出（前缀 u4r112）
 *      verify_layer -Ct u4 -Ref tail/l4_u2_ref.bin -Tol 3e-2 -Scale 1.0
 *   B) 位级交叉复算（防伪造）
 *      verify_layer -BitA u5 -BitB u5b -L 5
 *        → 两组密文逐字节比对；确定性流程下应完全相同
 *
 * 编译（与驱动同一套引擎源码）：
 *   gcc -O2 -fopenmp -Wno-implicit-function-declaration \
 *       -I include -I include/core -I include/common \
 *       -DCKKS_N=2048 -DCKKS_NPRIMES=112 -DBB=32 -DGG=32 \
 *       src/core/vllm_ntt.c src/core/vllm_ckks.c src/core/vllm_tp.c \
 *       tools/relay/verify_layer.c -o .tmp_tok/verify_layer -lm
 *
 * 退出码：0 = 全部 PASS；非 0 = 失败项数
 */
#include "vllm_ckks.h"
#include "vllm_ntt.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define LTOK 4
#define DIM  2048

/* ---------------- sk / ct 读写（与驱动一致的落盘格式） ---------------- */
static int sk_load(ckks_sk_t *sk, const char *path) {
    FILE *f = fopen(path, "rb"); if (!f) return -1;
    uint32_t nn;
    if (fread(&nn, 4, 1, f) != 1) { fclose(f); return -1; }
    sk->n = nn; sk->s = (int8_t *)malloc(nn);
    if (fread(sk->s, 1, nn, f) != nn) { fclose(f); return -1; }
    fclose(f); return 0;
}
static int ct_load(ckks_ct_t *ct, const char *path, const ckks_ctx_t *ctx) {
    FILE *f = fopen(path, "rb"); if (!f) return -1;
    uint32_t nn; int npr, cp;
    if (fread(&nn, 4, 1, f) != 1 || fread(&npr, 4, 1, f) != 1 || fread(&cp, 4, 1, f) != 1) { fclose(f); return -1; }
    ct->ctx = *ctx; ct->ctx.nprimes = npr; ct->comps = cp;
    ct->c = (uint64_t *)malloc((size_t)cp * npr * nn * 8);
    if (!ct->c) { fclose(f); return -1; }
    if (fread(ct->c, 8, (size_t)cp * npr * nn, f) != (size_t)cp * npr * nn) {
        free(ct->c); ct->c = NULL; fclose(f); return -1;
    }
    fclose(f); return 0;
}
static float *load_float_bin(const char *path, size_t cnt) {
    FILE *f = fopen(path, "rb"); if (!f) return NULL;
    float *v = (float *)malloc(cnt * 4);
    if (!v) { fclose(f); return NULL; }
    size_t got = fread(v, 4, cnt, f);
    fclose(f);
    if (got != cnt) { free(v); return NULL; }
    return v;
}

/* ---------------- 模式 B：两组密文位级比对 ---------------- */
static int bit_compare(const char *preA, const char *preB, int L, const char *chainDir) {
    int fails = 0;
    printf("[bit] cross-run bit-level comparison: %s vs %s (layer %d)\n", preA, preB, L);
    for (int t = 0; t < LTOK; t++) for (int h = 0; h < 2; h++) {
        char pa[256], pb[256];
        snprintf(pa, sizeof pa, "%s/%s_%d_%d.ct", chainDir, preA, t, h);
        snprintf(pb, sizeof pb, "%s/%s_%d_%d.ct", chainDir, preB, t, h);
        FILE *fa = fopen(pa, "rb"), *fb = fopen(pb, "rb");
        if (!fa || !fb) { printf("  [%s t%d h%d] LOAD_FAIL\n", preA, t, h); fails++; if (fa) fclose(fa); if (fb) fclose(fb); continue; }
        fseek(fa, 0, SEEK_END); long sa = ftell(fa); fseek(fa, 0, SEEK_SET);
        fseek(fb, 0, SEEK_END); long sb = ftell(fb); fseek(fb, 0, SEEK_SET);
        if (sa != sb) { printf("  [t%d h%d] SIZE_DIFF %ld vs %ld FAIL\n", t, h, sa, sb); fails++; fclose(fa); fclose(fb); continue; }
        unsigned char *ba = malloc(sa), *bb = malloc(sb);
        if (fread(ba, 1, sa, fa) != (size_t)sa || fread(bb, 1, sb, fb) != (size_t)sb) {
            printf("  [t%d h%d] READ_FAIL\n", t, h); fails++; free(ba); free(bb); fclose(fa); fclose(fb); continue;
        }
        long nd = 0; long first = -1;
        for (long i = 0; i < sa; i++) if (ba[i] != bb[i]) { nd++; if (first < 0) first = i; }
        printf("  [t%d h%d] bytes=%ld diff=%ld %s\n", t, h, sa, nd, nd ? "FAIL" : "PASS");
        if (nd) { printf("         first_diff_offset=%ld\n", first); fails++; }
        free(ba); free(bb); fclose(fa); fclose(fb);
    }
    return fails;
}

int main(int argc, char **argv) {
    int   L     = 4;
    const char *ctpre = NULL;
    const char *refpath = NULL;
    double tol  = 3e-2;
    double sc   = 1.0;
    const char *bitA = NULL, *bitB = NULL;
    const char *chainDir = ".tmp_tok/chain";
    const char *skpath   = ".tmp_tok/chain/sk.bin";

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "-L")   && i + 1 < argc) L = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-Ct")  && i + 1 < argc) ctpre = argv[++i];
        else if (!strcmp(argv[i], "-Ref") && i + 1 < argc) refpath = argv[++i];
        else if (!strcmp(argv[i], "-Tol") && i + 1 < argc) tol = atof(argv[++i]);
        else if (!strcmp(argv[i], "-Scale") && i + 1 < argc) sc = atof(argv[++i]);
        else if (!strcmp(argv[i], "-BitA") && i + 1 < argc) bitA = argv[++i];
        else if (!strcmp(argv[i], "-BitB") && i + 1 < argc) bitB = argv[++i];
        else if (!strcmp(argv[i], "-Dir") && i + 1 < argc) chainDir = argv[++i];
        else if (!strcmp(argv[i], "-Sk")  && i + 1 < argc) skpath = argv[++i];
        else { printf("unknown arg: %s\n", argv[i]); return -1; }
    }
    if (bitA || bitB) {
        if (!bitA || !bitB) { printf("-BitA and -BitB must be given together\n"); return -1; }
        int f = bit_compare(bitA, bitB, L, chainDir);
        printf("RESULT: %s (%d)\n", f ? "BIT_DIFF" : "BIT_IDENTICAL", f);
        return f;
    }

    char buf[256];
    if (!ctpre) { snprintf(buf, sizeof buf, "u%d", L); ctpre = strdup(buf); }
    char refbuf[256];
    if (!refpath) { snprintf(refbuf, sizeof refbuf, ".tmp_tok/tail/l%d_u2_ref.bin", L); refpath = refbuf; }

    ckks_ctx_t ctx;
    if (ckks_ctx_init(&ctx) != 0) { printf("ctx init fail\n"); return -1; }
    ckks_sk_t sk;
    if (sk_load(&sk, ".tmp_tok/chain/sk.bin") != 0) { printf("sk load fail\n"); return -1; }
    if (sk.n != ctx.n) { printf("sk.n=%u != ctx.n=%u\n", sk.n, ctx.n); return -1; }

    size_t K = ctx.n / 2;                 /* 槽位数 1024 */
    float *ref = load_float_bin(refpath, (size_t)LTOK * DIM);
    if (!ref) { printf("ref load fail: %s (expect %d floats)\n", refpath, LTOK * DIM); return -1; }

    printf("=== verify_layer: decrypt & compare vs plaintext reference ===\n");
    printf("n=%u  nprimes=%d  ct=%s/%s  ref=%s  tol=%.1e  scale=%g\n",
           ctx.n, ctx.nprimes, chainDir, ctpre, refpath, tol, sc);

    uint64_t *pdec = (uint64_t *)malloc((size_t)ctx.nprimes * ctx.n * 8);
    double   *zy   = (double   *)malloc(K * 8);
    int fails = 0, nok = 0;
    double gmax = 0;

    for (int t = 0; t < LTOK; t++) for (int h = 0; h < 2; h++) {
        char path[256];
        snprintf(path, sizeof path, "%s/%s_%d_%d.ct", chainDir, ctpre, t, h);
        ckks_ct_t ct = {0};
        if (ct_load(&ct, path, &ctx) != 0) { printf("  [%s t%d h%d] LOAD_FAIL %s\n", ctpre, t, h, path); fails++; continue; }
        if (ckks_decrypt(pdec, &sk, &ct) != 0) { printf("  [%s t%d h%d] DECRYPT_FAIL\n", ctpre, t, h); fails++; ckks_ct_free(&ct); continue; }
        ckks_decode(zy, pdec, &ct.ctx);
        /* 布局：ref 行宽 DIM=2048，h=0 -> 前 1024，h=1 -> 后 1024 */
        const float *rr = ref + (size_t)t * DIM + (h ? K : 0);
        double me = 0;
        for (size_t j = 0; j < K; j++) {
            double e = fabs(sc * zy[j] - (double)rr[j]);
            if (e > me) me = e;
        }
        if (me > gmax) gmax = me;
        int ok = (me < tol);
        printf("  [%s t%d h%d] max|err|=%.4e %s  (np=%d comps=%d)\n",
               ctpre, t, h, me, ok ? "PASS" : "FAIL", ct.ctx.nprimes, ct.comps);
        if (ok) nok++; else fails++;
        ckks_ct_free(&ct);
    }
    printf("RESULT: %s (%d/%d)  max_err=%.4e  tol=%.1e\n",
           fails ? "VERIFY_FAIL" : "VERIFY_PASS", nok, nok + fails, gmax, tol);
    free(pdec); free(zy); free(ref);
    ckks_sk_free(&sk); ckks_ctx_free(&ctx);
    return fails;
}
