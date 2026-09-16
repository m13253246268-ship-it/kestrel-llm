/**
 * vllm_npu.c - Transparent RK3588 NPU acceleration backend
 *
 * See vllm_npu.h for the design contract. Implementation notes:
 *   - librknnrt.so is discovered with dlopen() and every rknn_* entry is
 *     resolved into a function-pointer table, so the binary runs unchanged
 *     on hosts without the NPU runtime (all entry points return "CPU path").
 *   - RKNN ABI types are declared here (opaque, field-order compatible with
 *     rknn_api.h) so this file compiles without the RKNN headers.
 *   - Operator-level models are loaded lazily per (M, N, K, precision) and
 *     cached in a small registry; a missing model or a shape below the FLOPs
 *     threshold transparently falls back to the CPU (NEON) path.
 */

#ifdef _WIN32
/* ================================================================
 * x86/Windows stub：NPU 为 RK3588 专属（/dev/rknpu + librknnrt）。
 * 所有入口透明降级 CPU 路径（available()==0），语义同"无 NPU 宿主"。
 * ================================================================ */
#include "vllm_npu.h"

int vllm_npu_init(vllm_npu_t **npu, const vllm_npu_cfg_t *cfg) {
    (void)npu; (void)cfg;
    return 0;   /* 成功但不可用：透明回退 */
}
void vllm_npu_destroy(vllm_npu_t *npu) { (void)npu; }
int vllm_npu_available(const vllm_npu_t *npu) { (void)npu; return 0; }
int vllm_npu_sdk_version(const vllm_npu_t *npu, char *buf, int len) {
    (void)npu;
    if (buf && len > 0) buf[0] = '\0';
    return -1;
}
int vllm_npu_gemm_f32(vllm_npu_t *npu, float *out, const float *A, const float *B,
                      int M, int N, int K) {
    (void)npu; (void)out; (void)A; (void)B; (void)M; (void)N; (void)K;
    return 0;   /* CPU 路径 */
}
int vllm_npu_gemm_i8(vllm_npu_t *npu, float *out,
                     const int8_t *Aq, const float *aw,
                     int M, int N, int K, int layer, int proj) {
    (void)npu; (void)out; (void)Aq; (void)aw;
    (void)M; (void)N; (void)K; (void)layer; (void)proj;
    return 0;
}
vllm_npu_backend_t vllm_npu_backend(const vllm_npu_t *npu) {
    (void)npu;
    return VLLM_NPU_BACKEND_AUTO;
}
int vllm_npu_gemm_gw(vllm_npu_t *npu, float *out,
                     const int8_t *Aq, const float *a_scale,
                     const int8_t *Wq, const float *b_scale,
                     int M, int N, int K, int G, int prec,
                     uint64_t wkey) {
    (void)npu; (void)out; (void)Aq; (void)a_scale;
    (void)Wq; (void)b_scale; (void)M; (void)N; (void)K;
    (void)G; (void)prec; (void)wkey;
    return 0;
}
int vllm_npu_selftest(vllm_npu_t *npu) { (void)npu; return 0; }
#else

#include "vllm_npu.h"
#include "vllm_npu_direct.h"
#include "vllm_platform.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>

/* ================================================================
 * RKNN ABI (field-order compatible with rknn_api.h - do not reorder)
 * ================================================================ */
typedef int32_t rknn_context_t;

typedef enum {
    RKNN_QUERY_SDK_VERSION = 0,
    RKNN_QUERY_CORE_COUNT   = 1
} rknn_query_cmd_t;

typedef enum {
    RKNN_TENSOR_FLOAT32 = 0,
    RKNN_TENSOR_FLOAT16 = 1,
    RKNN_TENSOR_INT8    = 2,
    RKNN_TENSOR_INT16   = 3,
    RKNN_TENSOR_UINT8   = 4,
    RKNN_TENSOR_UINT16  = 5,
    RKNN_TENSOR_BOOL    = 6,
    RKNN_TENSOR_INT64   = 7
} rknn_tensor_type_t;

typedef enum {
    RKNN_TENSOR_NHWC = 0,
    RKNN_TENSOR_NCHW = 1
} rknn_tensor_format_t;

typedef struct {                /* rknn_input */
    uint32_t index;
    void     *buf;
    uint32_t size;
    rknn_tensor_format_t fmt;
    rknn_tensor_type_t  type;
    void     *attr;             /* rknn_tensor_attr* (unused, may be NULL) */
    uint32_t pass_through;
} rknn_input_t;

typedef struct {                /* rknn_output */
    uint8_t  want_float;
    uint8_t  is_prealloc;
    uint8_t  pass_through;
    uint8_t  align_size;
    uint32_t index;
    uint32_t size;
    void    *buf;
    void    *attr;
} rknn_output_t;

typedef struct {                /* rknn_sdk_version (query SDK_VERSION) */
    int    major;
    int    minor;
    int    micro;
    char   api_version[256];
    char   driver_version[256];
} rknn_sdk_version_t;

typedef int32_t rknn_npu_core_mask_t;

/* Function-pointer table for the resolved librknnrt API. */
typedef struct {
    int (*rknn_init)(rknn_context_t *, void *, uint32_t, uint32_t, void *);
    int (*rknn_query)(rknn_context_t, rknn_query_cmd_t, void *, uint32_t);
    int (*rknn_inputs_set)(rknn_context_t, uint32_t, rknn_input_t *);
    int (*rknn_run)(rknn_context_t, void *);
    int (*rknn_outputs_get)(rknn_context_t, uint32_t, rknn_output_t *, void *);
    int (*rknn_outputs_release)(rknn_context_t, uint32_t, rknn_output_t *);
    int (*rknn_destroy)(rknn_context_t);
    int (*rknn_set_core_mask)(rknn_context_t, rknn_npu_core_mask_t);
} rknn_api_t;

/* ================================================================
 * DIRECT backend - the engine's own minimal rknpu userspace driver
 * (zero third-party dependencies; see vllm_npu_direct.h/.c).
 * Programs the NPU through the stock Rockchip rknpu kernel driver
 * (/dev/rknpu). The engine's Q8_0/Q4_0 block weights feed the hardware
 * as-is - no .rknn/.rkllm conversion, no external NPU library.
 * ================================================================ */

/* Cached operator model. */
#define VLLM_NPU_MAX_MODELS 32
typedef struct {
    int  M, N, K, prec;         /* prec: 0 = fp32 GEMM, 1 = int8 GEMM */
    int  layer, proj;           /* int8: (layer, projection) key */
    int  state;                 /* 0 = not attempted, 1 = loaded, -1 = failed */
    rknn_context_t ctx;
} npu_model_t;

struct vllm_npu_s {
    int         ok;             /* selected backend usable */
    vllm_npu_backend_t backend; /* live backend */
    int         verbosity;
    int         core_mask;
    double      flops_threshold;
    char        model_dir[512];
    char        sdk_version[128];
    int         core_count;
    /* RKNN backend */
    rknn_api_t  api;
    void       *dl;             /* dlopen handle to librknnrt.so */
    npu_model_t models[VLLM_NPU_MAX_MODELS];
    int         n_models;
    /* DIRECT backend (in-tree zero-dependency driver) */
    vllm_npu_direct_t *direct;
};

/* ================================================================
 * Runtime discovery
 * ================================================================ */
static void npu_resolve(vllm_npu_t *n, void *h) {
    /* rknn functions use the C calling convention; dlsym handles it. */
    *(void **)(&n->api.rknn_init)           = dlsym(h, "rknn_init");
    *(void **)(&n->api.rknn_query)          = dlsym(h, "rknn_query");
    *(void **)(&n->api.rknn_inputs_set)     = dlsym(h, "rknn_inputs_set");
    *(void **)(&n->api.rknn_run)            = dlsym(h, "rknn_run");
    *(void **)(&n->api.rknn_outputs_get)    = dlsym(h, "rknn_outputs_get");
    *(void **)(&n->api.rknn_outputs_release)= dlsym(h, "rknn_outputs_release");
    *(void **)(&n->api.rknn_destroy)        = dlsym(h, "rknn_destroy");
    *(void **)(&n->api.rknn_set_core_mask)  = dlsym(h, "rknn_set_core_mask");
    n->ok = (n->api.rknn_init && n->api.rknn_query && n->api.rknn_inputs_set &&
             n->api.rknn_run && n->api.rknn_outputs_get && n->api.rknn_destroy);
}

/* DIRECT backend discovery: open the in-tree zero-dependency driver. */
static void npu_direct_discover(vllm_npu_t *n, int verbosity) {
    n->direct = vllm_npu_direct_open(verbosity, n->core_mask);
}

int vllm_npu_init(vllm_npu_t **out, const vllm_npu_cfg_t *cfg) {
    if (!out) return -1;
    *out = NULL;
    vllm_npu_t *n = (vllm_npu_t *)calloc(1, sizeof(vllm_npu_t));
    if (!n) return -1;

    vllm_npu_backend_t want = cfg ? cfg->backend : VLLM_NPU_BACKEND_AUTO;
    n->verbosity = cfg ? cfg->verbosity : 0;
    n->core_mask = cfg ? cfg->core_mask : 0;
    n->flops_threshold = cfg ? cfg->flops_threshold : 1e8;
    if (cfg && cfg->model_dir)
        snprintf(n->model_dir, sizeof(n->model_dir), "%s", cfg->model_dir);

    if (want == VLLM_NPU_BACKEND_AUTO || want == VLLM_NPU_BACKEND_DIRECT) {
        /* Preferred: the engine's own minimal rknpu driver - zero deps,
         * no model conversion. Falls through to RKNN only when explicitly
         * requested or no /dev/rknpu node exists. */
        npu_direct_discover(n, n->verbosity);
        if (n->direct) {
            n->ok = 1;
            n->backend = VLLM_NPU_BACKEND_DIRECT;
            snprintf(n->sdk_version, sizeof(n->sdk_version),
                     "in-tree direct rknpu driver (/dev/rknpu)");
        } else if (want == VLLM_NPU_BACKEND_DIRECT) {
            fprintf(stderr, "[NPU] direct backend unavailable: no /dev/rknpu "
                    "node (is the rknpu kernel driver loaded?). CPU path.\n");
        }
    }
    if (n->ok == 0 && (want == VLLM_NPU_BACKEND_AUTO || want == VLLM_NPU_BACKEND_RKNN)) {
        n->dl = dlopen(cfg && cfg->rknn_path ? cfg->rknn_path : "librknnrt.so",
                       RTLD_NOW | RTLD_LOCAL);
        if (n->dl) {
            npu_resolve(n, n->dl);
            if (n->ok && n->api.rknn_query) {
                rknn_sdk_version_t ver;
                memset(&ver, 0, sizeof(ver));
                if (n->api.rknn_query(0, RKNN_QUERY_SDK_VERSION, &ver, sizeof(ver)) == 0) {
                    snprintf(n->sdk_version, sizeof(n->sdk_version), "%d.%d.%d %s",
                             ver.major, ver.minor, ver.micro, ver.api_version);
                } else {
                    snprintf(n->sdk_version, sizeof(n->sdk_version), "queried");
                }
            }
            if (n->ok) n->backend = VLLM_NPU_BACKEND_RKNN;
        }
    }
    if (n->ok && n->verbosity > 0)
        fprintf(stderr, "[NPU] backend=%s (%s)\n",
                n->backend == VLLM_NPU_BACKEND_DIRECT ? "direct" : "rknn",
                n->sdk_version[0] ? n->sdk_version : "?");
    *out = n;
    return 0;
}

void vllm_npu_destroy(vllm_npu_t *n) {
    if (!n) return;
    for (int i = 0; i < n->n_models; i++) {
        if (n->models[i].state == 1 && n->api.rknn_destroy)
            n->api.rknn_destroy(n->models[i].ctx);
    }
    if (n->dl) dlclose(n->dl);
    vllm_npu_direct_close(n->direct);
    free(n);
}

int vllm_npu_available(const vllm_npu_t *n) {
    return (n && n->ok);
}

int vllm_npu_sdk_version(const vllm_npu_t *n, char *buf, int len) {
    if (!n || !n->ok || !buf || len <= 0) return -1;
    int l = (int)strlen(n->sdk_version);
    if (l >= len) l = len - 1;
    memcpy(buf, n->sdk_version, (size_t)l);
    buf[l] = 0;
    return l;
}

vllm_npu_backend_t vllm_npu_backend(const vllm_npu_t *n) {
    return (n && n->ok) ? n->backend : VLLM_NPU_BACKEND_AUTO;
}

/* ================================================================
 * Group-wise quantized GEMM (DIRECT backend) - no model conversion.
 *
 * The engine's Q8_0 (prec=0) / Q4_0 (prec=1) block weights map 1:1 onto the
 * NPU group-wise matmul: group G = 32 == one block, scales per block.
 *   C_f[m,n] = Σ_g a_scale[m,g]·b_scale[n,g]·(int32 partial of K-group g)
 * ================================================================ */
int vllm_npu_gemm_gw(vllm_npu_t *n, float *out,
                     const int8_t *Aq, const float *a_scale,
                     const int8_t *Wq, const float *b_scale,
                     int M, int N, int K, int G, int prec,
                     uint64_t wkey) {
    /* Diagnostic: VLLM_NPU_FORCE_CPU=1 keeps the NPU initialized but makes
     * every offload point fall back to the CPU - separates "NPU init side
     * effect" from "NPU output used" as the source of a wrong result. */
    if (getenv("VLLM_NPU_FORCE_CPU")) return 0;
    if (!n || !n->ok || n->backend != VLLM_NPU_BACKEND_DIRECT)
        return 0;
    if (!n->direct) return 0;
    if (!out || !Aq || !a_scale || !Wq || !b_scale) return 0;
    if (M <= 0 || N <= 0 || K <= 0 || G <= 0) return 0;
    if (M % 4 != 0) return 0;         /* HW M%4==0 contract; M==1 stays CPU */
    if (K % G != 0) return 0;
    if (2.0 * (double)M * (double)N * (double)K < n->flops_threshold)
        return 0;

    int rc = vllm_npu_direct_gemm_gw(n->direct, out, Aq, a_scale, Wq, b_scale,
                                     M, N, K, G, prec, wkey);
    return (rc == 0) ? 1 : 0;   /* -2 (uncalibrated regcmd) -> CPU fallback */
}

/* ================================================================
 * Model registry (lazy per-shape load of exported operator .rknn)
 * ================================================================ */
static npu_model_t *npu_find_model(vllm_npu_t *n, int M, int N, int K, int prec,
                                   int layer, int proj) {
    for (int i = 0; i < n->n_models; i++)
        if (n->models[i].M == M && n->models[i].N == N &&
            n->models[i].K == K && n->models[i].prec == prec &&
            n->models[i].layer == layer && n->models[i].proj == proj)
            return &n->models[i];
    return NULL;
}

static npu_model_t *npu_load_model(vllm_npu_t *n, int M, int N, int K, int prec,
                                   int layer, int proj) {
    npu_model_t *m = npu_find_model(n, M, N, K, prec, layer, proj);
    if (m) return (m->state == 1) ? m : NULL;

    if (n->n_models >= VLLM_NPU_MAX_MODELS) return NULL;   /* registry full */
    if (!n->model_dir[0]) return NULL;                     /* no model dir */

    char path[640];
    if (prec == 0) {
        /* generic fp32 MatMul (A + B as runtime inputs), one file per shape */
        snprintf(path, sizeof(path), "%s/gemm_%dx%dx%d.rknn", n->model_dir, M, N, K);
    } else {
        /* per-(layer, projection) int8 GEMM with baked weights */
        snprintf(path, sizeof(path), "%s/gemm_i8_p%d_l%d.rknn", n->model_dir, proj, layer);
    }

    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0) { fclose(f); return NULL; }
    void *buf = malloc((size_t)sz);
    if (!buf) { fclose(f); return NULL; }
    size_t rd = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    if (rd != (size_t)sz) { free(buf); return NULL; }

    m = &n->models[n->n_models];
    m->M = M; m->N = N; m->K = K; m->prec = prec;
    m->layer = layer; m->proj = proj;
    m->ctx = 0; m->state = -1;
    n->n_models++;

    if (n->api.rknn_init(&m->ctx, buf, (uint32_t)sz, 0, NULL) != 0) {
        if (n->verbosity > 1)
            fprintf(stderr, "[NPU] rknn_init failed: %s\n", path);
        free(buf);
        return NULL;
    }
    free(buf);   /* rknn_init copies the model */
    m->state = 1;

    if (n->api.rknn_set_core_mask && n->core_mask != VLLM_NPU_CORE_AUTO)
        n->api.rknn_set_core_mask(m->ctx, (rknn_npu_core_mask_t)n->core_mask);

    if (n->verbosity > 1)
        fprintf(stderr, "[NPU] model loaded: %s\n", path);
    return m;
}

/* ================================================================
 * GEMM offload entry points (transparent: return 0 = run CPU path)
 * ================================================================ */
int vllm_npu_gemm_f32(vllm_npu_t *n, float *out,
                      const float *A, const float *B,
                      int M, int N, int K) {
    if (!n || !n->ok || !out || !A || !B || M <= 0 || N <= 0 || K <= 0)
        return 0;
    if (2.0 * (double)M * (double)N * (double)K < n->flops_threshold)
        return 0;

    npu_model_t *m = npu_load_model(n, M, N, K, 0, 0, 0);
    if (!m) return 0;

    rknn_input_t in[2];
    memset(in, 0, sizeof(in));
    in[0].index = 0; in[0].buf = (void *)A; in[0].size = (uint32_t)((size_t)M * K * 4);
    in[0].fmt = RKNN_TENSOR_NCHW; in[0].type = RKNN_TENSOR_FLOAT32;
    in[1].index = 1; in[1].buf = (void *)B; in[1].size = (uint32_t)((size_t)K * N * 4);
    in[1].fmt = RKNN_TENSOR_NCHW; in[1].type = RKNN_TENSOR_FLOAT32;

    rknn_output_t outd[1];
    memset(outd, 0, sizeof(outd));
    outd[0].want_float = 1;
    outd[0].buf = out;
    outd[0].size = (uint32_t)((size_t)M * N * 4);
    outd[0].is_prealloc = 1;

    if (n->api.rknn_inputs_set(m->ctx, 2, in) != 0) return 0;
    if (n->api.rknn_run(m->ctx, NULL) != 0) return 0;
    if (n->api.rknn_outputs_get(m->ctx, 1, outd, NULL) != 0) return 0;
    n->api.rknn_outputs_release(m->ctx, 1, outd);
    return 1;
}

int vllm_npu_gemm_i8(vllm_npu_t *n, float *out,
                     const int8_t *Aq, const float *aw,
                     int M, int N, int K, int layer, int proj) {
    if (!n || !n->ok || !out || !Aq || !aw || M <= 0 || N <= 0 || K <= 0)
        return 0;
    if (2.0 * (double)M * (double)N * (double)K < n->flops_threshold)
        return 0;

    npu_model_t *m = npu_load_model(n, M, N, K, 1, layer, proj);
    if (!m) return 0;

    /* Inputs: Aq int8 [M*K], aw fp32 [M]. Output: out fp32 [M*N].
     * The exported graph (tools/npu/npu_export_ops.py) has the int8 weights baked
     * as constants and dequantizes with per-channel scales internally. */
    rknn_input_t in[2];
    memset(in, 0, sizeof(in));
    in[0].index = 0; in[0].buf = (void *)Aq; in[0].size = (uint32_t)((size_t)M * K);
    in[0].fmt = RKNN_TENSOR_NCHW; in[0].type = RKNN_TENSOR_INT8;
    in[1].index = 1; in[1].buf = (void *)aw; in[1].size = (uint32_t)((size_t)M * 4);
    in[1].fmt = RKNN_TENSOR_NCHW; in[1].type = RKNN_TENSOR_FLOAT32;

    rknn_output_t outd[1];
    memset(outd, 0, sizeof(outd));
    outd[0].want_float = 1;
    outd[0].buf = out;
    outd[0].size = (uint32_t)((size_t)M * N * 4);
    outd[0].is_prealloc = 1;

    if (n->api.rknn_inputs_set(m->ctx, 2, in) != 0) return 0;
    if (n->api.rknn_run(m->ctx, NULL) != 0) return 0;
    if (n->api.rknn_outputs_get(m->ctx, 1, outd, NULL) != 0) return 0;
    n->api.rknn_outputs_release(m->ctx, 1, outd);
    return 1;
}

/* ================================================================
 * Self test (PASS/FAIL, honest failure reporting)
 * ================================================================ */
int vllm_npu_selftest(vllm_npu_t *n) {
    if (!n) { fprintf(stderr, "[NPU-SELFTEST] FAIL no backend\n"); return 1; }
    if (!n->ok) { fprintf(stderr, "[NPU-SELFTEST] SKIP runtime absent\n"); return 0; }

    char ver[128] = "?";
    vllm_npu_sdk_version(n, ver, sizeof(ver));
    int failures = 0;

    if (n->backend == VLLM_NPU_BACKEND_DIRECT) {
        /* The in-tree direct driver has its own staged self-test
         * (BO round-trip + group-wise matmul vs CPU reference). */
        return vllm_npu_direct_selftest(n->direct, ver);
    }

    /* RKNN: fp32 GEMM smoke test: 8x8 identity-like check */
    float A[64], B[64], C[64], ref[64];
    for (int i = 0; i < 64; i++) {
        A[i] = (float)(i % 7) * 0.25f;
        B[i] = (float)((i / 8) == (i % 8) ? 1.0f : 0.0f);  /* identity */
    }
    for (int r = 0; r < 8; r++)
        for (int c = 0; c < 8; c++) {
            double acc = 0.0;
            for (int k = 0; k < 8; k++) acc += (double)A[r * 8 + k] * B[k * 8 + c];
            ref[r * 8 + c] = (float)acc;
        }
    int ok = vllm_npu_gemm_f32(n, C, A, B, 8, 8, 8);
    if (ok) {
        double max_err = 0.0;
        for (int i = 0; i < 64; i++)
            max_err = fmax(max_err, fabs((double)C[i] - (double)ref[i]));
        int pass = (max_err < 1e-3);
        printf("[NPU-SELFTEST] gemm_f32 8x8 %s (max_err=%.2e) [sdk=%s]\n",
               pass ? "PASS" : "FAIL", max_err, ver);
        if (!pass) failures++;
    } else {
        printf("[NPU-SELFTEST] gemm_f32 8x8 SKIP (no model / below threshold)\n");
    }
    return failures;
}
#endif /* !_WIN32 */
