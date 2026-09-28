/* 板端诊断专用：真实候选源码与冻结原 worker 在同一输入上逐字节对拍。 */
#include "../../src/model/vllm_safetensors.c"
#include "baseline_attn_worker.h"

typedef struct {
    float *raw, *p;
    size_t n;
} probe_buffer;

static uint32_t probe_rng = 0x631dbeefu;
static uint32_t probe_next(void) {
    probe_rng ^= probe_rng << 13;
    probe_rng ^= probe_rng >> 17;
    probe_rng ^= probe_rng << 5;
    return probe_rng;
}

static probe_buffer probe_alloc(size_t n) {
    probe_buffer b;
    b.raw = malloc((n + 32) * sizeof(float));
    if (!b.raw) exit(2);
    b.p = b.raw + 16;
    b.n = n;
    memset(b.raw, 0xa5, (n + 32) * sizeof(float));
    return b;
}

static void probe_fill(probe_buffer b, int style) {
    for (size_t i = 0; i < b.n; i++) {
        float x = ((int)(probe_next() & 65535) - 32768) / 32768.0f;
        b.p[i] = style == 1 ? 0.0f : style == 2 ? x * 8.0f : x;
    }
}

static void probe_guard(probe_buffer b) {
    for (int i = 0; i < 16; i++) {
        uint32_t a, z;
        memcpy(&a, b.raw + i, 4);
        memcpy(&z, b.p + b.n + i, 4);
        if (a != 0xa5a5a5a5u || z != 0xa5a5a5a5u) {
            fprintf(stderr, "GUARD FAIL\n");
            exit(3);
        }
    }
}

static void probe_equal(probe_buffer a, probe_buffer b, const char *name,
                        int nb, int prev, int hd, int mode) {
    if (!memcmp(a.p, b.p, a.n * sizeof(float))) return;
    for (size_t i = 0; i < a.n; i++) {
        uint32_t x, y;
        memcpy(&x, a.p + i, 4);
        memcpy(&y, b.p + i, 4);
        if (x != y) {
            fprintf(stderr, "FAIL %s index=%zu nb=%d prev=%d hd=%d mode=%d %08x != %08x\n",
                    name, i, nb, prev, hd, mode, x, y);
            exit(4);
        }
    }
}

static void probe_case(int nb, int prev, int nh, int nkv, int hd,
                       int has_imp, int style, int bench) {
    int stride = nb + prev + 7;
    probe_buffer q = probe_alloc((size_t)nb * nh * hd);
    probe_buffer k = probe_alloc((size_t)nkv * stride * hd);
    probe_buffer v = probe_alloc(k.n);
    probe_buffer out[2], scores[2], imp[2];
    for (int j = 0; j < 2; j++) {
        out[j] = probe_alloc(q.n);
        scores[j] = probe_alloc((size_t)nh * ST_ATTN_QB * stride);
        imp[j] = probe_alloc((size_t)nh * stride);
    }
    probe_fill(q, style);
    probe_fill(k, style);
    probe_fill(v, style);
    probe_fill(imp[0], 0);
    probe_buffer initial = probe_alloc(imp[0].n);
    memcpy(initial.p, imp[0].p, initial.n * sizeof(float));
    vllm_attn_batched_ctx c = {out[0].p, q.p, k.p, v.p, scores[0].p,
        has_imp ? imp[0].p : NULL, nb, prev, stride, nh, nkv, hd, stride,
        hd & ~3, 1.0f / sqrtf((float)hd), 0};
    vllm_tp_parfor(0, nh, baseline_attn_worker, &c);
    for (int mode = 0; mode <= 2; mode++) {
        memset(out[1].p, 0xa5, out[1].n * sizeof(float));
        memset(scores[1].p, 0xa5, scores[1].n * sizeof(float));
        memcpy(imp[1].p, initial.p, initial.n * sizeof(float));
        c.attn_out = out[1].p;
        c.scores = scores[1].p;
        c.imp_head = has_imp ? imp[1].p : NULL;
        c.pv_tile = mode;
        vllm_tp_parfor(0, nh, vllm_attn_batched_worker, &c);
        probe_equal(out[0], out[1], "out", nb, prev, hd, mode);
        probe_equal(scores[0], scores[1], "scores", nb, prev, hd, mode);
        probe_equal(imp[0], imp[1], "imp", nb, prev, hd, mode);
    }
    if (bench) {
        /* 三轮换序；计时不包含分配、比较或日志。输出在所有腿结束后打印。 */
        const int order[3][3] = {{0,1,2}, {2,1,0}, {0,2,1}};
        double times[3][3];
        for (int r = 0; r < 3; r++) {
            for (int leg = 0; leg < 3; leg++) {
                int mode = order[r][leg];
                c.pv_tile = mode;
                memcpy(imp[1].p, initial.p, initial.n * sizeof(float));
                double t = vllm_tp_wtime();
                vllm_tp_parfor(0, nh, vllm_attn_batched_worker, &c);
                times[r][mode] = (vllm_tp_wtime() - t) * 1000;
                probe_equal(out[0], out[1], "timed_out", nb, prev, hd, mode);
                probe_equal(imp[0], imp[1], "timed_imp", nb, prev, hd, mode);
            }
        }
        for (int r = 0; r < 3; r++)
            printf("BENCH nb=%d prev=%d nh=%d nkv=%d hd=%d r=%d ms0=%.6f ms1=%.6f ms2=%.6f\n",
                   nb, prev, nh, nkv, hd, r, times[r][0], times[r][1], times[r][2]);
    }
    probe_guard(q); probe_guard(k); probe_guard(v); probe_guard(initial);
    free(q.raw); free(k.raw); free(v.raw); free(initial.raw);
    for (int j = 0; j < 2; j++) {
        probe_guard(out[j]); probe_guard(scores[j]); probe_guard(imp[j]);
        free(out[j].raw); free(scores[j].raw); free(imp[j].raw);
    }
}

int main(int argc, char **argv) {
    int bench = argc > 1 && !strcmp(argv[1], "--bench");
    int threads = argc > 2 ? atoi(argv[2]) : 4;
    if (threads != 1 && threads != 4) return 2;
    vllm_tp_init(threads);
    printf("compute_threads=%d QB=%d\n", vllm_tp_threads(), ST_ATTN_QB);
    int cases = 0;
    if (bench) {
        for (int prev = 0; prev <= 1792; prev += 896) {
            probe_case(256, prev, 16, 8, 128, 1, 0, 1);
            cases++;
        }
    } else {
        const int batches[] = {1,2,3,4,5,7,8,9,16,33};
        const int histories[] = {0,1,124,125,126,127,128,129,253};
        /* GQA 比例都要覆盖：8:2、16:8（2B，kh=ha/2）、32:8（8B，kh=ha/4）。 */
        const int gqa_nh[] = {8, 16, 32};
        const int gqa_nkv[] = {2, 8, 8};
        for (size_t g = 0; g < 3; g++)
            for (size_t b = 0; b < sizeof(batches)/sizeof(*batches); b++)
                for (size_t h = 0; h < sizeof(histories)/sizeof(*histories); h++)
                    for (int imp = 0; imp <= 1; imp++)
                        for (int style = 0; style < 3; style++) {
                            probe_case(batches[b], histories[h], gqa_nh[g],
                                       gqa_nkv[g], 128, imp, style, 0);
                            cases++;
                        }
        const int dims[] = {16,32,64,80,128,256};
        for (size_t d = 0; d < sizeof(dims)/sizeof(*dims); d++) {
            probe_case(9, 129, 8, 8, dims[d], 1, 0, 0);
            cases++;
        }
        probe_case(256, 1792, 16, 8, 128, 1, 0, 0);
        cases++;
        probe_case(256, 1792, 32, 8, 128, 1, 0, 0);
        cases++;
    }
    printf("PASS cases=%d modes=3 out_scores_imp_bitwise=1 guards=1\n", cases);
    vllm_tp_shutdown();
    return 0;
}
