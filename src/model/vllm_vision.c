/**
 * vllm_vision.c - Vision Encoder Implementation for Qwen3-VL
 *
 * Implements the full ViT-based vision encoder including:
 *   - Patch embedding (3D for video, 2D for images via 3D with T=1)
 *   - 27-layer ViT with GELU activation and DeepStack features
 *   - Bidirectional attention (no causal mask)
 *   - Spatial 2x2 merge
 *   - Merger + DeepStack mergers → LLM embedding space
 */

#include "vllm_vision.h"
#include "vllm_safetensors.h"
#include "vllm_platform.h"
#include "vllm_tp.h"   /* self-contained thread pool (replaces OpenMP) */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

/* 单平台（RK3588/aarch64）：ViT 注意力内核走标量实现（位级语义与
 * 原 AVX2 逐 lane 一致），不再依赖 immintrin / vllm_avx_emul.h。 */
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

/* Q8_0 compact block bytes (34B per 32 floats: f16 scale + 32 int8).
 * Mirrors Q8_BYTES in vllm_safetensors.c; fc2 cols pad 4304 → 4320. */
#define VQ8_BYTES(n)  ((size_t)((n) + 31) / 32 * 34)
#define VF_PAD(vf)    (((vf) + 31) / 32 * 32)

/* ================================================================
 * Internal helpers
 * ================================================================ */

static inline float gelu_approx(float x) {
    /* tanh approximation of GELU: 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3))) */
    float x3 = x * x * x;
    float inner = 0.7978845608f * (x + 0.044715f * x3); /* sqrt(2/pi) = 0.7978845608 */
    return 0.5f * x * (1.0f + tanhf(inner));
}

static inline void rms_norm_f32(float *__restrict out, const float *__restrict x,
                                 const float *__restrict weight,
                                 int n, float eps) {
    double ss = 0.0;
    for (int i = 0; i < n; i++) {
        float v = x[i];
        ss += (double)v * v;
    }
    float rms = 1.0f / sqrtf((float)(ss / n) + eps);
    for (int i = 0; i < n; i++) {
        out[i] = x[i] * rms * weight[i];
    }
}

/* Layer norm with bias */
static inline void layer_norm_f32(float *__restrict out, const float *__restrict x,
                                   const float *__restrict weight,
                                   const float *__restrict bias,
                                   int n, float eps) {
    double sum = 0.0, ss = 0.0;
    for (int i = 0; i < n; i++) {
        sum += (double)x[i];
        ss  += (double)x[i] * x[i];
    }
    float mean = (float)(sum / n);
    float var  = (float)(ss / n) - mean * mean;
    float inv_std = 1.0f / sqrtf(var + eps);
    for (int i = 0; i < n; i++) {
        float xn = (x[i] - mean) * inv_std;
        out[i] = xn * weight[i] + bias[i];
    }
}

/* Matrix-vector multiply: out = W * x  (no bias)
 * W layout: [out_rows, in_cols] row-major */
typedef struct {
    float *out; const float *W; const float *x;
    int out_rows, in_cols; const float *bias;   /* bias != NULL => +bias */
} vvis_matvec_ctx;

static void vvis_matvec_worker(void *ctx_, int r) {
    vvis_matvec_ctx *c = (vvis_matvec_ctx *)ctx_;
    double sum = 0.0;
    const float *wr = c->W + (size_t)r * c->in_cols;
    for (int cc = 0; cc < c->in_cols; cc++)
        sum += (double)wr[cc] * c->x[cc];
    c->out[r] = (float)sum + (c->bias ? c->bias[r] : 0.0f);
}

static void matvec_f32(float *__restrict out, const float *__restrict W,
                        const float *__restrict x,
                        int out_rows, int in_cols) {
    vvis_matvec_ctx c = { out, W, x, out_rows, in_cols, NULL };
    vllm_tp_parfor(0, out_rows, vvis_matvec_worker, &c);
}

/* Matrix-vector with bias: out = W * x + bias */
static void matvec_f32_bias(float *__restrict out, const float *__restrict W,
                              const float *__restrict x, const float *__restrict bias,
                              int out_rows, int in_cols) {
    vvis_matvec_ctx c = { out, W, x, out_rows, in_cols, bias };
    vllm_tp_parfor(0, out_rows, vvis_matvec_worker, &c);
}

/* Softmax on a vector (NEON max + vector expf + vector normalize; the sum is
 * still accumulated in double like the scalar path, so the normalization
 * scale differs from it only by fp rounding). */
static void softmax_f32(float *x, int n) {
#if defined(__aarch64__)
    float max_val = x[0];
    if (n >= 4) {
        float32x4_t vmax = vld1q_f32(x);
        int i = 4;
        for (; i + 4 <= n; i += 4)
            vmax = vmaxq_f32(vmax, vld1q_f32(x + i));
        max_val = vmaxvq_f32(vmax);
        for (; i < n; i++) if (x[i] > max_val) max_val = x[i];
    } else {
        for (int i = 1; i < n; i++) if (x[i] > max_val) max_val = x[i];
    }
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += (double)x[i];
    }
    float inv_sum = (float)(1.0 / sum);
    int i = 0;
    if (n >= 4) {
        float32x4_t vinv = vdupq_n_f32(inv_sum);
        for (; i + 4 <= n; i += 4)
            vst1q_f32(x + i, vmulq_f32(vld1q_f32(x + i), vinv));
    }
    for (; i < n; i++) x[i] *= inv_sum;
#else
    float max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += (double)x[i];
    }
    float inv_sum = (float)(1.0 / sum);
    for (int i = 0; i < n; i++) {
        x[i] *= inv_sum;
    }
#endif
}

/* Horizontal sum of an 8-wide vector (scalar, mirrors hsum_avx2 tree). */
static inline float hsum8_ps(const float v[8]) {
    float lo[4], hi[4];
    for (int i = 0; i < 4; i++) { lo[i] = v[i]; hi[i] = v[4 + i]; }
    for (int i = 0; i < 4; i++) lo[i] += hi[i];
    float s01 = lo[0] + lo[1], s23 = lo[2] + lo[3];
    return s01 + s23;
}

/* ================================================================
 * Weight Management
 * ================================================================ */

int st_vision_weights_alloc(STVisionWeights *w, const STModelConfig *cfg) {
    if (!cfg->has_vision) return -1;
    memset(w, 0, sizeof(*w));

    int vh  = cfg->vis_hidden;   /* 1152 */
    int vd  = cfg->vis_depth;    /* 27 */
    int vf  = cfg->vis_ffn;      /* 4304 */
    int vp  = cfg->vis_patch;    /* 16 */
    int vt  = cfg->vis_temporal;  /* 2 */
    int vo  = cfg->vis_out_dim;  /* 4096 */
    int vm  = cfg->vis_merge;    /* 2 */
    int ds  = cfg->vis_ds_count; /* 3 */

    /* Patch embedding: Conv3d */
    w->patch_embed_weight = calloc((size_t)vh * 3 * vt * vp * vp, sizeof(float));
    w->patch_embed_bias   = calloc(vh, sizeof(float));

    /* Position embedding */
    w->pos_embed = calloc((size_t)cfg->vis_max_pos * vh, sizeof(float));

    /* ViT block weights */
    w->attn_qkv_weight  = calloc((size_t)vd * vh * 3 * vh, sizeof(float));
    w->attn_qkv_bias    = calloc((size_t)vd * 3 * vh, sizeof(float));
    w->attn_proj_weight = calloc((size_t)vd * vh * vh, sizeof(float));
    w->attn_proj_bias   = calloc((size_t)vd * vh, sizeof(float));
    w->norm1_weight     = calloc((size_t)vd * vh, sizeof(float));
    w->norm1_bias       = calloc((size_t)vd * vh, sizeof(float));
    w->norm2_weight     = calloc((size_t)vd * vh, sizeof(float));
    w->norm2_bias       = calloc((size_t)vd * vh, sizeof(float));
    w->mlp_fc1_weight   = calloc((size_t)vd * vf * vh, sizeof(float));
    w->mlp_fc1_bias     = calloc((size_t)vd * vf, sizeof(float));
    /* fc2: down projection, standard GELU (no gating) */
    w->mlp_fc2_weight   = calloc((size_t)vd * vh * vf, sizeof(float));
    w->mlp_fc2_bias     = calloc((size_t)vd * vh, sizeof(float));

    /* Q8_0 ViT weights (weight-only; f32 allocs above are kept for A/B).
     * fc2 col dim padded to a multiple of 32 (4304 → 4320). */
    int vf_pad = VF_PAD(vf);
    w->q8_attn_qkv_weight  = malloc((size_t)vd * VQ8_BYTES((size_t)vh * 3 * vh));
    w->q8_attn_proj_weight = malloc((size_t)vd * VQ8_BYTES((size_t)vh * vh));
    w->q8_mlp_fc1_weight   = malloc((size_t)vd * VQ8_BYTES((size_t)vf * vh));
    w->q8_mlp_fc2_weight   = malloc((size_t)vd * VQ8_BYTES((size_t)vh * vf_pad));
    if (!w->q8_attn_qkv_weight || !w->q8_attn_proj_weight ||
        !w->q8_mlp_fc1_weight || !w->q8_mlp_fc2_weight) {
        fprintf(stderr, "[VIS] Q8 weight allocation failed\n");
        return -1;
    }

    /* Main merger: after spatial merge, input_dim = vh * vm * vm = vh * 4 */
    int merge_in  = vh * vm * vm;
    int merge_mid = merge_in;   /* Qwen3-VL uses same dim for fc1 */
    w->merger_norm_weight = calloc(merge_in, sizeof(float));
    w->merger_norm_bias   = calloc(merge_in, sizeof(float));
    w->merger_fc1_weight  = calloc((size_t)merge_in * merge_mid, sizeof(float));
    w->merger_fc1_bias    = calloc(merge_mid, sizeof(float));
    w->merger_fc2_weight  = calloc((size_t)merge_mid * vo, sizeof(float));
    w->merger_fc2_bias    = calloc(vo, sizeof(float));

    /* DeepStack mergers (input = merged features, dim = vh * 4 = 4608) */
    int ds_mid = merge_in;  /* deepstack intermediate dim (= merge_in) */
    w->ds_norm_weight  = calloc((size_t)ds * merge_in, sizeof(float));
    w->ds_norm_bias    = calloc((size_t)ds * merge_in, sizeof(float));
    w->ds_fc1_weight   = calloc((size_t)ds * merge_in * ds_mid, sizeof(float));
    w->ds_fc1_bias     = calloc((size_t)ds * ds_mid, sizeof(float));
    w->ds_fc2_weight   = calloc((size_t)ds * ds_mid * vo, sizeof(float));
    w->ds_fc2_bias     = calloc((size_t)ds * vo, sizeof(float));

    w->is_allocated = 1;
    return 0;
}

void st_vision_weights_free(STVisionWeights *w) {
    free(w->patch_embed_weight); free(w->patch_embed_bias);
    free(w->pos_embed);
    free(w->attn_qkv_weight); free(w->attn_qkv_bias);
    free(w->attn_proj_weight); free(w->attn_proj_bias);
    free(w->norm1_weight); free(w->norm1_bias);
    free(w->norm2_weight); free(w->norm2_bias);
    free(w->mlp_fc1_weight); free(w->mlp_fc1_bias);
    free(w->mlp_fc2_weight); free(w->mlp_fc2_bias);
    free(w->q8_attn_qkv_weight); free(w->q8_attn_proj_weight);
    free(w->q8_mlp_fc1_weight);  free(w->q8_mlp_fc2_weight);
    free(w->merger_norm_weight); free(w->merger_norm_bias);
    free(w->merger_fc1_weight); free(w->merger_fc1_bias);
    free(w->merger_fc2_weight); free(w->merger_fc2_bias);
    free(w->ds_norm_weight); free(w->ds_norm_bias);
    free(w->ds_fc1_weight); free(w->ds_fc1_bias);
    free(w->ds_fc2_weight); free(w->ds_fc2_bias);
    memset(w, 0, sizeof(*w));
}

/* Free only the F32 duplicates of the Q8-quantized ViT matrices (inference
 * reads the Q8 copies). The F32-only small tensors (patch embed, pos embed,
 * norms, biases, merger, deepstack) stay. */
void st_vision_weights_free_f32(STVisionWeights *w) {
    if (!w) return;
    free(w->attn_qkv_weight);  w->attn_qkv_weight  = NULL;
    free(w->attn_proj_weight); w->attn_proj_weight = NULL;
    free(w->mlp_fc1_weight);   w->mlp_fc1_weight   = NULL;
    free(w->mlp_fc2_weight);   w->mlp_fc2_weight   = NULL;
}

/* Quantize ViT weights F32 → Q8_0 (weight-only). See definition below. */
static void st_vision_quantize_q8(STVisionWeights *w, const STModelConfig *cfg);

int st_vision_load_weights(STVisionWeights *w, const STModelConfig *cfg) {
    /* This function loads weights from safetensors.
     * We use the generic st_load_tensor to load named tensors.
     * The visual tensors are named like:
     *   model.visual.patch_embed.proj.weight
     *   model.visual.patch_embed.proj.bias
     *   model.visual.pos_embed.weight (or just pos_embed)
     *   model.visual.blocks.{N}.attn.qkv.weight
     *   model.visual.blocks.{N}.attn.qkv.bias
     *   model.visual.blocks.{N}.attn.proj.weight
     *   model.visual.blocks.{N}.attn.proj.bias
     *   model.visual.blocks.{N}.norm1.weight
     *   model.visual.blocks.{N}.norm1.bias
     *   model.visual.blocks.{N}.norm2.weight
     *   model.visual.blocks.{N}.norm2.bias
     *   model.visual.blocks.{N}.mlp.linear_fc1.weight
     *   model.visual.blocks.{N}.mlp.linear_fc1.bias
     *   model.visual.blocks.{N}.mlp.linear_fc2.weight
     *   model.visual.blocks.{N}.mlp.linear_fc2.bias
     *   model.visual.merger.norm.weight
     *   model.visual.merger.norm.bias
     *   model.visual.merger.linear_fc1.weight
     *   model.visual.merger.linear_fc1.bias
     *   model.visual.merger.linear_fc2.weight
     *   model.visual.merger.linear_fc2.bias
     *   model.visual.deepstack_merger_list.{M}.*
     */
    int vh = cfg->vis_hidden;
    int vd = cfg->vis_depth;
    int vf = cfg->vis_ffn;
    char name[256];
    int expected;

    /* --- Patch embedding --- */
    /* Transposed: Conv weight [out, in, kt, kh, kw] → need to load correctly */
    /* For now, load into flat buffer; the actual conv operation handles layout */
    expected = (int)((size_t)vh * 3 * cfg->vis_temporal * cfg->vis_patch * cfg->vis_patch);
    snprintf(name, sizeof(name), "model.visual.patch_embed.proj.weight");
    if (st_load_tensor(cfg, name, w->patch_embed_weight, expected) != 0) {
        fprintf(stderr, "[VIS] Failed to load %s\n", name);
        return -1;
    }
    snprintf(name, sizeof(name), "model.visual.patch_embed.proj.bias");
    if (st_load_tensor(cfg, name, w->patch_embed_bias, vh) != 0) {
        fprintf(stderr, "[VIS] Warning: no patch_embed bias\n");
    }

    /* --- Position embedding (try both names) --- */
    expected = cfg->vis_max_pos * vh;
    snprintf(name, sizeof(name), "model.visual.pos_embed.weight");
    if (st_load_tensor(cfg, name, w->pos_embed, expected) != 0) {
        /* Try alternate name */
        snprintf(name, sizeof(name), "model.visual.position_embedding.weight");
        if (st_load_tensor(cfg, name, w->pos_embed, expected) != 0) {
            fprintf(stderr, "[VIS] Warning: no position embedding found\n");
        }
    }

    /* --- ViT blocks --- */
    for (int l = 0; l < vd; l++) {
        /* norm1 */
        snprintf(name, sizeof(name), "model.visual.blocks.%d.norm1.weight", l);
        st_load_tensor(cfg, name, w->norm1_weight + (size_t)l * vh, vh);
        snprintf(name, sizeof(name), "model.visual.blocks.%d.norm1.bias", l);
        st_load_tensor(cfg, name, w->norm1_bias + (size_t)l * vh, vh);

        /* norm2 */
        snprintf(name, sizeof(name), "model.visual.blocks.%d.norm2.weight", l);
        st_load_tensor(cfg, name, w->norm2_weight + (size_t)l * vh, vh);
        snprintf(name, sizeof(name), "model.visual.blocks.%d.norm2.bias", l);
        st_load_tensor(cfg, name, w->norm2_bias + (size_t)l * vh, vh);

        /* attn.qkv */
        expected = (int)((size_t)vh * 3 * vh);
        snprintf(name, sizeof(name), "model.visual.blocks.%d.attn.qkv.weight", l);
        st_load_tensor(cfg, name, w->attn_qkv_weight + (size_t)l * vh * 3 * vh, expected);

        snprintf(name, sizeof(name), "model.visual.blocks.%d.attn.qkv.bias", l);
        st_load_tensor(cfg, name, w->attn_qkv_bias + (size_t)l * 3 * vh, 3 * vh);

        /* attn.proj */
        expected = vh * vh;
        snprintf(name, sizeof(name), "model.visual.blocks.%d.attn.proj.weight", l);
        st_load_tensor(cfg, name, w->attn_proj_weight + (size_t)l * vh * vh, expected);
        snprintf(name, sizeof(name), "model.visual.blocks.%d.attn.proj.bias", l);
        st_load_tensor(cfg, name, w->attn_proj_bias + (size_t)l * vh, vh);

        /* mlp linear_fc1 (gate+up fused for SwiGLU-style, but ViT uses GELU) */
        expected = (int)((size_t)vf * vh);
        snprintf(name, sizeof(name), "model.visual.blocks.%d.mlp.linear_fc1.weight", l);
        st_load_tensor(cfg, name, w->mlp_fc1_weight + (size_t)l * vf * vh, expected);
        snprintf(name, sizeof(name), "model.visual.blocks.%d.mlp.linear_fc1.bias", l);
        st_load_tensor(cfg, name, w->mlp_fc1_bias + (size_t)l * vf, vf);

        /* mlp linear_fc2 (standard GELU, fc2: vf → vh) */
        expected = (int)((size_t)vh * vf);
        snprintf(name, sizeof(name), "model.visual.blocks.%d.mlp.linear_fc2.weight", l);
        st_load_tensor(cfg, name, w->mlp_fc2_weight + (size_t)l * vh * vf, expected);
        snprintf(name, sizeof(name), "model.visual.blocks.%d.mlp.linear_fc2.bias", l);
        st_load_tensor(cfg, name, w->mlp_fc2_bias + (size_t)l * vh, vh);
    }

    /* --- Main merger --- */
    int merge_in = vh * cfg->vis_merge * cfg->vis_merge;
    int merge_mid = merge_in;
    /* Per-chunk LayerNorm: weight/bias size is vh (applied per-patch, not to concat) */
    snprintf(name, sizeof(name), "model.visual.merger.norm.weight");
    st_load_tensor(cfg, name, w->merger_norm_weight, vh);
    snprintf(name, sizeof(name), "model.visual.merger.norm.bias");
    st_load_tensor(cfg, name, w->merger_norm_bias, vh);
    snprintf(name, sizeof(name), "model.visual.merger.linear_fc1.weight");
    st_load_tensor(cfg, name, w->merger_fc1_weight, merge_in * merge_mid);
    snprintf(name, sizeof(name), "model.visual.merger.linear_fc1.bias");
    st_load_tensor(cfg, name, w->merger_fc1_bias, merge_mid);
    snprintf(name, sizeof(name), "model.visual.merger.linear_fc2.weight");
    st_load_tensor(cfg, name, w->merger_fc2_weight, merge_mid * cfg->vis_out_dim);
    snprintf(name, sizeof(name), "model.visual.merger.linear_fc2.bias");
    st_load_tensor(cfg, name, w->merger_fc2_bias, cfg->vis_out_dim);

    /* --- DeepStack mergers (input = merged features, dim = merge_in = 4608) --- */
    int ds_mid = merge_in;  /* deepstack intermediate dim */
    for (int m = 0; m < cfg->vis_ds_count; m++) {
        snprintf(name, sizeof(name), "model.visual.deepstack_merger_list.%d.norm.weight", m);
        st_load_tensor(cfg, name, w->ds_norm_weight + (size_t)m * merge_in, merge_in);
        snprintf(name, sizeof(name), "model.visual.deepstack_merger_list.%d.norm.bias", m);
        st_load_tensor(cfg, name, w->ds_norm_bias + (size_t)m * merge_in, merge_in);
        snprintf(name, sizeof(name), "model.visual.deepstack_merger_list.%d.linear_fc1.weight", m);
        st_load_tensor(cfg, name, w->ds_fc1_weight + (size_t)m * merge_in * ds_mid, merge_in * ds_mid);
        snprintf(name, sizeof(name), "model.visual.deepstack_merger_list.%d.linear_fc1.bias", m);
        st_load_tensor(cfg, name, w->ds_fc1_bias + (size_t)m * ds_mid, ds_mid);
        snprintf(name, sizeof(name), "model.visual.deepstack_merger_list.%d.linear_fc2.weight", m);
        st_load_tensor(cfg, name, w->ds_fc2_weight + (size_t)m * ds_mid * cfg->vis_out_dim,
                       ds_mid * cfg->vis_out_dim);
        snprintf(name, sizeof(name), "model.visual.deepstack_merger_list.%d.linear_fc2.bias", m);
        st_load_tensor(cfg, name, w->ds_fc2_bias + (size_t)m * cfg->vis_out_dim, cfg->vis_out_dim);
    }

    printf("[VIS] Vision weights loaded successfully.\n");
    st_vision_quantize_q8(w, cfg);
    return 0;
}

/* Quantize the ViT weight matrices F32 → Q8_0 (weight-only, axiom:
 * blas_precision_efficiency_tradeoff). Call after st_vision_load_weights.
 * fc2's col dim (vis_ffn=4304) is not a multiple of 32, so each row is
 * padded with zeros to vf_pad before quantization (scale unaffected). */
static void st_vision_quantize_q8(STVisionWeights *w, const STModelConfig *cfg) {
    int vh = cfg->vis_hidden;
    int vd = cfg->vis_depth;
    int vf = cfg->vis_ffn;
    int vf_pad = VF_PAD(vf);

    for (int l = 0; l < vd; l++) {
        /* QKV [3*vh, vh] — cols 32-aligned */
        f32_to_q8_0(w->q8_attn_qkv_weight + (size_t)l * VQ8_BYTES((size_t)vh * 3 * vh),
                    w->attn_qkv_weight + (size_t)l * vh * 3 * vh,
                    vh * 3 * vh);
        /* proj [vh, vh] */
        f32_to_q8_0(w->q8_attn_proj_weight + (size_t)l * VQ8_BYTES((size_t)vh * vh),
                    w->attn_proj_weight + (size_t)l * vh * vh,
                    vh * vh);
        /* fc1 [vf, vh] — cols 32-aligned */
        f32_to_q8_0(w->q8_mlp_fc1_weight + (size_t)l * VQ8_BYTES((size_t)vf * vh),
                    w->mlp_fc1_weight + (size_t)l * vf * vh,
                    vf * vh);
        /* fc2 [vh, vf] — pad each row to vf_pad */
        for (int r = 0; r < vh; r++) {
            float tmp[4352];  /* vf_pad=4320 max for vis_ffn <= 4352 */
            memcpy(tmp, w->mlp_fc2_weight + (size_t)l * vh * vf + (size_t)r * vf,
                   (size_t)vf * sizeof(float));
            memset(tmp + vf, 0, (size_t)(vf_pad - vf) * sizeof(float));
            f32_to_q8_0(w->q8_mlp_fc2_weight + (size_t)l * VQ8_BYTES((size_t)vh * vf_pad)
                            + (size_t)r * VQ8_BYTES(vf_pad),
                        tmp, vf_pad);
        }
    }
    fprintf(stderr, "[VIS] ViT weights quantized to Q8_0 (fc2 pad %d → %d)\n",
            vf, vf_pad);
}

/* ================================================================
 * Runtime State
 * ================================================================ */

int st_vision_init(STVisionState *vis, const STModelConfig *cfg) {
    memset(vis, 0, sizeof(*vis));
    vis->cfg = cfg;

    int vh = cfg->vis_hidden;
    int vf = cfg->vis_ffn;
    int vo = cfg->vis_out_dim;
    int max_patches = cfg->vis_max_pos;

    vis->hidden_states = calloc((size_t)max_patches * vh, sizeof(float));
    vis->attn_buf      = calloc((size_t)max_patches * vh, sizeof(float));
    vis->mlp_buf       = calloc((size_t)max_patches * vh, sizeof(float));
    vis->qkv_buf       = calloc((size_t)max_patches * 3 * vh, sizeof(float));
    vis->fc1_buf       = calloc((size_t)max_patches * VF_PAD(vf), sizeof(float));

    /* Patch features buffer */
    vis->patch_features = calloc((size_t)max_patches * vh, sizeof(float));

    /* Vision RoPE cos/sin tables ([max_patches, head_dim]) */
    vis->rot_cos = calloc((size_t)max_patches * (vh / cfg->vis_heads), sizeof(float));
    vis->rot_sin = calloc((size_t)max_patches * (vh / cfg->vis_heads), sizeof(float));

    /* DeepStack features: post-merger output [max_grid_tokens, vis_out_dim].
     * Each deepstack merger (full-norm) runs over the spatially-merged grid,
     * so the row dimension is the merged token count (max_patches/4). */
    if (cfg->vis_ds_count > 0) {
        vis->ds_features = calloc(cfg->vis_ds_count, sizeof(float*));
        int max_ds_tokens = max_patches / (cfg->vis_merge * cfg->vis_merge);
        for (int i = 0; i < cfg->vis_ds_count; i++) {
            vis->ds_features[i] = calloc((size_t)max_ds_tokens * cfg->vis_out_dim, sizeof(float));
        }
    }

    /* Visual tokens output: main grid only (deepstack features live in
     * ds_features, injected into the LLM — not part of the token sequence). */
    int max_grid_tokens = max_patches / (cfg->vis_merge * cfg->vis_merge);
    vis->visual_tokens = calloc((size_t)max_grid_tokens * vo, sizeof(float));
    vis->n_vis_tokens = 0;

    vis->pixel_values = NULL;
    vis->is_allocated = 1;
    return 0;
}

void st_vision_free(STVisionState *vis) {
    free(vis->hidden_states);
    free(vis->attn_buf);
    free(vis->mlp_buf);
    free(vis->qkv_buf);
    free(vis->fc1_buf);
    free(vis->patch_features);
    free(vis->rot_cos);
    free(vis->rot_sin);
    free(vis->visual_tokens);
    free(vis->pixel_values);
    if (vis->ds_features) {
        for (int i = 0; i < vis->cfg->vis_ds_count; i++) {
            free(vis->ds_features[i]);
        }
        free(vis->ds_features);
    }
    memset(vis, 0, sizeof(*vis));
}

/* ================================================================
 * ViT Block Forward Pass
 * ================================================================ */

/**
 * Single ViT block (fused): attention + MLP with residual connections.
 * x: [n_patches, hidden_dim] input/output
 * layer_idx: which ViT layer to use
 *
 * Fusion: the whole block runs in just TWO OMP parallel domains per layer
 * (vs ~10 before), so the per-domain fork/join and re-sweeps vanish:
 *   domain 1: LN1 → QKV (single-patch Q8 matvec) → RoPE
 *   domain 2: AVX2 attention → proj → residual → LN2 → fc1 → GELU → fc2 → residual
 * Each thread owns a patch and runs its whole pipeline serially, keeping the
 * patch's activations in L1/L2 while the Q8 weight rows stream from L3.
 */
typedef struct {
    STVisionState *vis; const STModelConfig *cfg;
    float *x, *attn_buf, *mlp_buf, *qkv_buf, *fc1_buf;
    float *score_scratch;   /* [nthreads * n_patches] attention score staging */
    const float *norm1_w, *norm1_b, *norm2_w, *norm2_b;
    const float *fc1_b, *fc2_b;
    const uint8_t *q8_qkv, *q8_proj, *q8_fc1, *q8_fc2;
    int vh, nh, hd, vf, vf_pad, layer_idx, n_patches;
    float scale;
} vvis_vit_ctx;

/* Domain 1a: LN1 -> attn_buf (patch-parallel). */
static void vvis_vit_ln1(void *ctx_, int p) {
    vvis_vit_ctx *c = (vvis_vit_ctx *)ctx_;
    layer_norm_f32(c->attn_buf + (size_t)p * c->vh,
                   c->x + (size_t)p * c->vh,
                   c->norm1_w, c->norm1_b, c->vh, c->cfg->norm_eps);
}

/* Domain 1b: RoPE on qkv (patch-parallel). qkv_buf layout is the
 * head-contiguous [3][nh][n_patches][hd] from st_q8_matvec_batch_qkv:
 * q = (0*nh+h), k = (1*nh+h), v = (2*nh+h) row slabs. */
static void vvis_vit_rope(void *ctx_, int p) {
    vvis_vit_ctx *c = (vvis_vit_ctx *)ctx_;
    const int half = c->hd / 2;
    const float *rc = c->vis->rot_cos + (size_t)p * c->hd;
    const float *rs = c->vis->rot_sin + (size_t)p * c->hd;
    const size_t slab = (size_t)c->n_patches * c->hd;
    for (int h = 0; h < c->nh; h++) {
        float *q = c->qkv_buf + ((size_t)h) * slab + (size_t)p * c->hd;
        float *k = c->qkv_buf + ((size_t)(c->nh + h)) * slab + (size_t)p * c->hd;
        for (int d = 0; d < half; d++) {
            float cc = rc[d], ss = rs[d];
            float q0 = q[d], q1 = q[d + half];
            q[d] = q0 * cc - q1 * ss;
            q[d + half] = q1 * cc + q0 * ss;
            float k0 = k[d], k1 = k[d + half];
            k[d] = k0 * cc - k1 * ss;
            k[d + half] = k1 * cc + k0 * ss;
        }
    }
}

/* Domain 2a: attention over a PATCH BLOCK (block-parallel). Each worker
 * processes VVIS_ATTN_PB query patches; a K/V row (4 cache lines) is then
 * reused by all PB queries, instead of the per-patch kernel where every
 * query re-scans the whole key set alone (qkv's head-major strided layout
 * made each scan touch the full 8.4 MB K/V at cache-line granularity ->
 * ~8.6 GB DRAM/layer -> ~2 s/layer on RK3588). Blocking cuts that to
 * 8.6 GB/PB per layer. scores[] = per-thread [PB * n_patches] scratch. */
#define VVIS_ATTN_PB 32

static void vvis_vit_attn_block(void *ctx_, int pb) {
    vvis_vit_ctx *c = (vvis_vit_ctx *)ctx_;
    int tid = vllm_tp_worker_id();
    if (tid < 0) tid = 0;
    const int vh = c->vh, hd = c->hd, nh = c->nh, n_patches = c->n_patches;
    const int v4n = hd / 4;
    const int p0 = pb * VVIS_ATTN_PB;
    const int pb_n = (p0 + VVIS_ATTN_PB <= n_patches) ? VVIS_ATTN_PB
                                                      : (n_patches - p0);
    float *scores = c->score_scratch + (size_t)tid * VVIS_ATTN_PB * n_patches;
    const size_t slab = (size_t)n_patches * hd;   /* per (seg, head) slab */

    for (int h = 0; h < nh; h++) {
        /* score: stream keys once (contiguous per head), dot all PB queries. */
        const float *k_h = c->qkv_buf + ((size_t)(nh + h)) * slab;   /* seg 1 */
        const float *q_h0 = c->qkv_buf + ((size_t)h) * slab;         /* seg 0 */
        for (int s = 0; s < n_patches; s++) {
            const float *k_s = k_h + (size_t)s * hd;
            for (int q = 0; q < pb_n; q++) {
                const float *q_h = q_h0 + (size_t)(p0 + q) * hd;
                float32x4_t acc[16];
                for (int i = 0; i < v4n; i++)
                    acc[i] = vmulq_f32(vld1q_f32(q_h + i * 4), vld1q_f32(k_s + i * 4));
                for (int w = v4n / 2; w >= 1; w >>= 1)
                    for (int i = 0; i < w; i++)
                        acc[i] = vaddq_f32(acc[i], acc[i + w]);
                float32x2_t lo = vget_low_f32(acc[0]);
                float32x2_t hi = vget_high_f32(acc[0]);
                scores[(size_t)q * n_patches + s] =
                    (vget_lane_f32(lo, 0) + vget_lane_f32(lo, 1) +
                     vget_lane_f32(hi, 0) + vget_lane_f32(hi, 1)) * c->scale;
            }
        }
        for (int q = 0; q < pb_n; q++)
            softmax_f32(scores + (size_t)q * n_patches, n_patches);

        /* attn@V: stream values once (contiguous), accumulate all PB queries. */
        const float *v_h = c->qkv_buf + ((size_t)(2 * nh + h)) * slab;  /* seg 2 */
        for (int q = 0; q < pb_n; q++) {
            float32x4_t accv[16];
            for (int i = 0; i < v4n; i++) accv[i] = vdupq_n_f32(0.0f);
            const float *sq = scores + (size_t)q * n_patches;
            for (int s = 0; s < n_patches; s++) {
                const float *v_s = v_h + (size_t)s * hd;
                float32x4_t sc = vdupq_n_f32(sq[s]);
                for (int i = 0; i < v4n; i++)
                    accv[i] = vfmaq_f32(accv[i], sc, vld1q_f32(v_s + i * 4));
            }
            float *out_h = c->attn_buf + (size_t)(p0 + q) * vh + h * hd;
            for (int i = 0; i < v4n; i++) vst1q_f32(out_h + i * 4, accv[i]);
        }
    }
}

/* Domain 2b: residual add after the attention projection (patch-parallel). */
static void vvis_vit_add1(void *ctx_, int p) {
    vvis_vit_ctx *c = (vvis_vit_ctx *)ctx_;
    float *xp = c->x + (size_t)p * c->vh;
    const float *mp = c->mlp_buf + (size_t)p * c->vh;
    for (int i = 0; i < c->vh; i++) xp[i] += mp[i];
}

/* Domain 2c: LN2 -> mlp_buf (patch-parallel). */
static void vvis_vit_ln2(void *ctx_, int p) {
    vvis_vit_ctx *c = (vvis_vit_ctx *)ctx_;
    layer_norm_f32(c->mlp_buf + (size_t)p * c->vh,
                   c->x + (size_t)p * c->vh,
                   c->norm2_w, c->norm2_b, c->vh, c->cfg->norm_eps);
}

/* Domain 2d: GELU on fc1_buf, zero the pad tail (patch-parallel). */
static void vvis_vit_gelu(void *ctx_, int p) {
    vvis_vit_ctx *c = (vvis_vit_ctx *)ctx_;
    float *fp = c->fc1_buf + (size_t)p * c->vf_pad;
    for (int i = 0; i < c->vf; i++) fp[i] = gelu_approx(fp[i]);
    for (int i = c->vf; i < c->vf_pad; i++) fp[i] = 0.0f;
}

/* Domain 2e: residual add after FC2 (patch-parallel). */
static void vvis_vit_add2(void *ctx_, int p) {
    vvis_vit_ctx *c = (vvis_vit_ctx *)ctx_;
    float *xp = c->x + (size_t)p * c->vh;
    const float *ap = c->attn_buf + (size_t)p * c->vh;
    for (int i = 0; i < c->vh; i++) xp[i] += ap[i];
}

static void vit_block_forward(STVisionState *vis, int layer_idx, int n_patches) {
    const STModelConfig *cfg = vis->cfg;
    static int vis_dbg = -1;
    if (vis_dbg < 0) vis_dbg = getenv("VLLM_MMDBG") != NULL;
    double t0 = vis_dbg ? st_now_sec() : 0.0;
    int vh = cfg->vis_hidden;
    int nh = cfg->vis_heads;
    int hd = vh / nh;  /* head_dim = 72 */
    int vf = cfg->vis_ffn;  /* 4304 */
    int vf_pad = VF_PAD(vf);

    float *x = vis->hidden_states;
    float *attn_buf = vis->attn_buf;
    float *mlp_buf = vis->mlp_buf;
    float *qkv_buf = vis->qkv_buf;
    float *fc1_buf = vis->fc1_buf;
    const float *norm1_w = vis->w.norm1_weight + (size_t)layer_idx * vh;
    const float *norm1_b = vis->w.norm1_bias + (size_t)layer_idx * vh;
    const float *norm2_w = vis->w.norm2_weight + (size_t)layer_idx * vh;
    const float *norm2_b = vis->w.norm2_bias + (size_t)layer_idx * vh;
    const float *fc1_b = vis->w.mlp_fc1_bias + (size_t)layer_idx * vf;
    const float *fc2_b = vis->w.mlp_fc2_bias + (size_t)layer_idx * vh;

    const uint8_t *q8_qkv = vis->w.q8_attn_qkv_weight + (size_t)layer_idx * VQ8_BYTES((size_t)vh * 3 * vh);
    const uint8_t *q8_proj = vis->w.q8_attn_proj_weight + (size_t)layer_idx * VQ8_BYTES((size_t)vh * vh);
    const uint8_t *q8_fc1  = vis->w.q8_mlp_fc1_weight  + (size_t)layer_idx * VQ8_BYTES((size_t)vf * vh);
    const uint8_t *q8_fc2  = vis->w.q8_mlp_fc2_weight  + (size_t)layer_idx * VQ8_BYTES((size_t)vh * vf_pad);

    vvis_vit_ctx vc;
    vc.vis = vis; vc.cfg = cfg;
    vc.x = x; vc.attn_buf = attn_buf; vc.mlp_buf = mlp_buf; vc.qkv_buf = qkv_buf; vc.fc1_buf = fc1_buf;
    vc.norm1_w = norm1_w; vc.norm1_b = norm1_b; vc.norm2_w = norm2_w; vc.norm2_b = norm2_b;
    vc.fc1_b = fc1_b; vc.fc2_b = fc2_b;
    /* Attention score staging: mlp_buf is idle until the proj matvec below,
     * so [nthreads * VVIS_ATTN_PB * n_patches] floats of it can stage the
     * per-thread softmax inputs of the blocked attention kernel. */
    vc.score_scratch = mlp_buf;
    vc.q8_qkv = q8_qkv; vc.q8_proj = q8_proj; vc.q8_fc1 = q8_fc1; vc.q8_fc2 = q8_fc2;
    vc.vh = vh; vc.nh = nh; vc.hd = hd; vc.vf = vf; vc.vf_pad = vf_pad;
    vc.layer_idx = layer_idx; vc.n_patches = n_patches;
    vc.scale = 1.0f / sqrtf((float)hd);

    /* ===== Domain 1: LN1 (patch-parallel) -> batch QKV (weights streamed
     * once, reused across all patches) -> RoPE (patch-parallel).
     * The batched matvec eliminates the per-patch full-weight re-read that
     * made large images (many patches) take minutes on RK3588. ===== */
    vllm_tp_parfor(0, n_patches, vvis_vit_ln1, &vc);
    st_q8_matvec_batch_qkv(qkv_buf, q8_qkv,
                           vis->w.attn_qkv_bias + (size_t)layer_idx * 3 * vh,
                           attn_buf, n_patches, 3 * vh, vh, nh, hd);
    vllm_tp_parfor(0, n_patches, vvis_vit_rope, &vc);
    if (vis_dbg)
        fprintf(stderr, "[VIS]   l%d d1_qkv %.3fs\n", layer_idx, st_now_sec() - t0);

    /* ===== Domain 2: attention (patch-block parallel, K/V rows reused by
     * VVIS_ATTN_PB queries per worker) -> batch proj -> residual -> LN2 ->
     * batch fc1 -> GELU -> batch fc2 -> residual ===== */
    vllm_tp_parfor(0, (n_patches + VVIS_ATTN_PB - 1) / VVIS_ATTN_PB,
                   vvis_vit_attn_block, &vc);
    if (vis_dbg)
        fprintf(stderr, "[VIS]   l%d attn   %.3fs\n", layer_idx, st_now_sec() - t0);
    st_q8_matvec_batch(mlp_buf, q8_proj,
                       vis->w.attn_proj_bias + (size_t)layer_idx * vh,
                       attn_buf, n_patches, vh, vh);
    vllm_tp_parfor(0, n_patches, vvis_vit_add1, &vc);
    vllm_tp_parfor(0, n_patches, vvis_vit_ln2, &vc);
    if (vis_dbg)
        fprintf(stderr, "[VIS]   l%d d2a     %.3fs\n", layer_idx, st_now_sec() - t0);
    st_q8_matvec_batch(fc1_buf, q8_fc1, fc1_b, mlp_buf, n_patches, vf, vh);
    vllm_tp_parfor(0, n_patches, vvis_vit_gelu, &vc);
    st_q8_matvec_batch(attn_buf, q8_fc2, fc2_b, fc1_buf, n_patches, vh, vf_pad);
    vllm_tp_parfor(0, n_patches, vvis_vit_add2, &vc);
    if (vis_dbg) {
        double _dt = st_now_sec() - t0;
        fprintf(stderr, "[VIS]   l%d d2b     %.3fs\n", layer_idx, _dt);
        fflush(stderr);
    }
}

/* ================================================================
 * Image Preprocessing & Patch Embedding
 * ================================================================ */

/**
 * Preprocess image: resize to compatible dimensions and normalize.
 * Qwen3-VL checkpoint normalization (preprocessor_config.json):
 *   image_mean = image_std = [0.5, 0.5, 0.5]
 *   pixel = (p/255 - 0.5) / 0.5 = p/127.5 - 1  →  [-1, 1]
 * (Qwen2.5-VL used ImageNet stats; Qwen3-VL overrides them to 0.5/0.5.
 *  Using the wrong stats shifts every ViT input by ~1.86x scale + offset.)
 */
static int preprocess_image(float *__restrict dst,
                             const uint8_t *__restrict src,
                             int src_w, int src_h,
                             int *out_w, int *out_h,
                             int patch_size, int max_grid) {
    /* Choose the resize target. The vision encoder's buffers are sized for
     * `max_grid` patches (vis_max_pos, e.g. 48x48=2304), so a photo that
     * would exceed that grid must be downscaled (aspect-preserving) instead
     * of rejected/crashing. */
    int gw = (src_w + patch_size - 1) / patch_size;
    int gh = (src_h + patch_size - 1) / patch_size;
    int target_w, target_h;
    if (gw * gh > max_grid) {
        double scale = sqrt((double)max_grid / ((double)gw * (double)gh));
        target_w = (int)((double)src_w * scale / patch_size) * patch_size;
        target_h = (int)((double)src_h * scale / patch_size) * patch_size;
        if (target_w < patch_size) target_w = patch_size;
        if (target_h < patch_size) target_h = patch_size;
        /* The snap can overshoot the grid for extreme aspect ratios; shrink
         * the longer side until it fits. */
        while ((target_w / patch_size) * (target_h / patch_size) > max_grid) {
            if (target_w >= target_h) target_w -= patch_size;
            else                     target_h -= patch_size;
            if (target_w < patch_size || target_h < patch_size) break;
        }
    } else {
        target_w = gw * patch_size;
        target_h = gh * patch_size;
    }

    /* Clamp to reasonable size */
    if (target_w > 2048) target_w = 2048;
    if (target_h > 2048) target_h = 2048;

    *out_w = target_w;
    *out_h = target_h;

    for (int y = 0; y < target_h; y++) {
        int sy = y * src_h / target_h;
        for (int x = 0; x < target_w; x++) {
            int sx = x * src_w / target_w;
            int src_idx = (sy * src_w + sx) * 3;
            int dst_idx = (y * target_w + x) * 3;
            for (int c = 0; c < 3; c++) {
                float v = (float)src[src_idx + c];
                dst[dst_idx + c] = v / 127.5f - 1.0f;
            }
        }
    }
    return 0;
}

/**
 * Patch embedding: split image into patches and project to hidden_dim.
 * Input: pixel_values [height * width * 3], normalized to [-1, 1]
 * Output: patch_features [grid_h * grid_w, vis_hidden]
 *
 * The patch_embed is a Conv3d: weight [vis_hidden, 3, temporal_patch, patch, patch]
 * For images: temporal=1 (single frame)
 * For video: temporal=temporal_patch_size
 *
 * We implement this as direct patch extraction + matvec, not actual convolution.
 */
typedef struct {
    STVisionState *vis; const float *pw, *pb, *pixels;
    int grid_h, grid_w, vh, vt, ps, img_w;
} vvis_conv_ctx;

static void vvis_patch_conv_worker(void *ctx_, int gy) {
    vvis_conv_ctx *c = (vvis_conv_ctx *)ctx_;
#if defined(__aarch64__)
    const int npx = 3 * c->vt * c->ps * c->ps;   /* 1536 for ps=16, vt=2 */
    /* stack staging in the contiguous [cc][t][py][px] layout the conv weight
     * uses; images are embedded as vt identical frames, so t just repeats. */
    float patch_px[3 * 2 * 16 * 16];
    for (int gx = 0; gx < c->grid_w; gx++) {
        int p_idx = gy * c->grid_w + gx;
        for (int cc = 0; cc < 3; cc++)
            for (int t = 0; t < c->vt; t++)
                for (int py = 0; py < c->ps; py++) {
                    const float *row = c->pixels +
                        (((size_t)(gy * c->ps + py) * c->img_w + gx * c->ps) * 3 + cc);
                    float *dst = patch_px + ((cc * c->vt + t) * c->ps + py) * c->ps;
                    for (int px = 0; px < c->ps; px++)
                        dst[px] = row[px * 3];
                }
        float *feat = c->vis->patch_features + (size_t)p_idx * c->vh;
        for (int oc = 0; oc < c->vh; oc++) {
            const float *w_oc = c->pw + (size_t)oc * npx;
            float32x4_t acc = vdupq_n_f32(0.0f);
            int i = 0;
            for (; i + 4 <= npx; i += 4)
                acc = vfmaq_f32(acc, vld1q_f32(w_oc + i), vld1q_f32(patch_px + i));
            float sum = vaddvq_f32(acc);
            for (; i < npx; i++) sum += w_oc[i] * patch_px[i];
            feat[oc] = sum + c->pb[oc];
        }
    }
#else
    for (int gx = 0; gx < c->grid_w; gx++) {
        int p_idx = gy * c->grid_w + gx;
        float *feat = c->vis->patch_features + (size_t)p_idx * c->vh;
        for (int oc = 0; oc < c->vh; oc++) {
            double sum = 0.0;
            const float *w_oc = c->pw + (size_t)oc * 3 * c->vt * c->ps * c->ps;
            for (int cc = 0; cc < 3; cc++) {
                for (int t = 0; t < c->vt; t++) {
                    for (int py = 0; py < c->ps; py++) {
                        for (int px = 0; px < c->ps; px++) {
                            int pixel_y = gy * c->ps + py;
                            int pixel_x = gx * c->ps + px;
                            int pixel_idx = (pixel_y * c->img_w + pixel_x) * 3 + cc;
                            float pixel = c->pixels[pixel_idx];
                            int w_idx = cc * c->vt * c->ps * c->ps + t * c->ps * c->ps + py * c->ps + px;
                            sum += (double)w_oc[w_idx] * pixel;
                        }
                    }
                }
            }
            feat[oc] = (float)sum + c->pb[oc];
        }
    }
#endif
}

static int patch_embed_image(STVisionState *vis,
                              const float *pixels, int img_h, int img_w) {
    const STModelConfig *cfg = vis->cfg;
    int vh = cfg->vis_hidden;
    int ps = cfg->vis_patch;   /* 16 */
    int vt = cfg->vis_temporal; /* 2: images are embedded as 2 identical frames */

    int grid_h = img_h / ps;
    int grid_w = img_w / ps;
    int n_patches = grid_h * grid_w;

    if (n_patches > cfg->vis_max_pos) {
        fprintf(stderr, "[VIS] Too many patches: %d > max %d\n", n_patches, cfg->vis_max_pos);
        return -1;
    }

    vis->grid_h = grid_h;
    vis->grid_w = grid_w;
    vis->n_frames_eff = 1;

    /* Extract patches and project.
     * Conv weight layout: [vh, 3, vt, ps, ps] (temporal kernel first for
     * each channel). Qwen treats a still image as vt identical frames, so
     * the temporal kernel is summed over vt duplicate frames. */
    const float *pw = vis->w.patch_embed_weight;
    const float *pb = vis->w.patch_embed_bias;

    vvis_conv_ctx vc = { vis, pw, pb, pixels, grid_h, grid_w, vh, vt, ps, img_w };
    vllm_tp_parfor(0, grid_h, vvis_patch_conv_worker, &vc);

    return n_patches;
}

/* ================================================================
 * Vision RoPE + interpolated position embedding
 * ================================================================ */

/**
 * Vision RoPE (Qwen3-VL reference: Qwen3VLVisionRotaryEmbedding +
 * get_vision_position_ids + apply_rotary_pos_emb_vision).
 *
 * Each patch token carries a 2D (h, w) grid position. The rotary table has
 *   inv_freq[i] = 1 / theta^(2*i / (head_dim/2))   theta=10000, i in [0, hd/4)
 * and the flattened rotary embedding for a patch is
 *   [h * inv_freq, w * inv_freq]  (hd/2 entries), repeated once (hd entries).
 *
 * Also adds the interpolated position embedding here (bilinear resample of
 * the square vis_max_pos table — 48x48 grid — onto the actual grid), which
 * replaces the old direct row-major index.
 *
 * Patch layout is raster order: images [row, col]; videos [t, row, col]
 * (each frame shares the same spatial positions). This matches the official
 * processor: sequence order is arbitrary for the (non-causal) ViT attention,
 * what matters is that each token's position code matches its grid location.
 */
typedef struct {
    STVisionState *vis; const float *pe, *inv_freq;
    int n_patches, n_spatial, grid_h, grid_w, side, vh, hd, half, n_freq;
} vvis_pos_ctx;

static void vvis_pos_worker(void *ctx_, int p) {
    vvis_pos_ctx *c = (vvis_pos_ctx *)ctx_;
    int within = p % c->n_spatial;
    int row = within / c->grid_w;
    int col = within % c->grid_w;

    float sh = (c->grid_h > 1) ? (float)row * (c->side - 1) / (float)(c->grid_h - 1) : 0.0f;
    float sw = (c->grid_w > 1) ? (float)col * (c->side - 1) / (float)(c->grid_w - 1) : 0.0f;
    int h0 = (int)floorf(sh);
    if (h0 > c->side - 1) h0 = c->side - 1;
    int h1 = (h0 + 1 < c->side) ? h0 + 1 : h0;
    int w0 = (int)floorf(sw);
    if (w0 > c->side - 1) w0 = c->side - 1;
    int w1 = (w0 + 1 < c->side) ? w0 + 1 : w0;
    float wh = sh - h0;
    float ww = sw - w0;

    float *feat = c->vis->patch_features + (size_t)p * c->vh;
    const float *p00 = c->pe + ((size_t)h0 * c->side + w0) * c->vh;
    const float *p01 = c->pe + ((size_t)h0 * c->side + w1) * c->vh;
    const float *p10 = c->pe + ((size_t)h1 * c->side + w0) * c->vh;
    const float *p11 = c->pe + ((size_t)h1 * c->side + w1) * c->vh;
    for (int d = 0; d < c->vh; d++) {
        float v = (1.0f - wh) * (1.0f - ww) * p00[d]
                + (1.0f - wh) * ww       * p01[d]
                + wh       * (1.0f - ww) * p10[d]
                + wh       * ww          * p11[d];
        feat[d] += v;
    }

    /* RoPE cos/sin table ([p, hd]) */
    float *rc = c->vis->rot_cos + (size_t)p * c->hd;
    float *rs = c->vis->rot_sin + (size_t)p * c->hd;
    for (int d = 0; d < c->half; d++) {
        int fi = (d < c->n_freq) ? d : d - c->n_freq;
        float ang = ((d < c->n_freq) ? (float)row : (float)col) * c->inv_freq[fi];
        float cc = cosf(ang), ss = sinf(ang);
        rc[d] = cc; rs[d] = ss;
        rc[d + c->half] = cc; rs[d + c->half] = ss;
    }
}

static void vision_pos_encode(STVisionState *vis, int n_patches,
                              int grid_t, int grid_h, int grid_w) {
    const STModelConfig *cfg = vis->cfg;
    int vh = cfg->vis_hidden;
    int hd = vh / cfg->vis_heads;          /* 72 */
    int half = hd / 2;                     /* 36 */
    int n_freq = half / 2;                 /* 18 */
    int side = (int)sqrtf((float)cfg->vis_max_pos);  /* 48 */

    float inv_freq[64];
    for (int i = 0; i < n_freq; i++) {
        /* arange(0, dim, 2)/dim with dim = half -> exponent 2*i/half */
        inv_freq[i] = 1.0f / powf(10000.0f, (float)(2 * i) / (float)half);
    }

    const float *pe = vis->w.pos_embed;    /* [side*side, vh] */
    int n_spatial = grid_h * grid_w;
    if (n_spatial <= 0) n_spatial = 1;

    vvis_pos_ctx vc = { vis, pe, inv_freq, n_patches, n_spatial, grid_h, grid_w,
                        side, vh, hd, half, n_freq };
    vllm_tp_parfor(0, n_patches, vvis_pos_worker, &vc);
}

/* ================================================================
 * Spatial Merge & Merger
 * ================================================================ */

/**
 * Spatial merge: group 2x2 patches and concatenate their features.
 * Input: [grid_h, grid_w, vh]
 * Output: [grid_h/2, grid_w/2, vh*4]
 */
typedef struct { float *merged; const float *patches; int grid_h, grid_w, vh; } vvis_merge_ctx;

static void vvis_merge_worker(void *ctx_, int my) {
    vvis_merge_ctx *c = (vvis_merge_ctx *)ctx_;
    const int merge = 2;
    int mg_w = c->grid_w / merge;
    int out_dim = c->vh * merge * merge;  /* vh * 4 */
    for (int mx = 0; mx < mg_w; mx++) {
        float *out = c->merged + ((size_t)my * mg_w + mx) * out_dim;
        for (int dy = 0; dy < merge; dy++) {
            for (int dx = 0; dx < merge; dx++) {
                int py = my * merge + dy;
                int px = mx * merge + dx;
                const float *src = c->patches + ((size_t)py * c->grid_w + px) * c->vh;
                float *dst_slot = out + (dy * merge + dx) * c->vh;
                memcpy(dst_slot, src, c->vh * sizeof(float));
            }
        }
    }
}

static void spatial_merge(float *__restrict merged,
                           const float *__restrict patches,
                           int grid_h, int grid_w, int vh) {
    int merge = 2;
    int mg_h = grid_h / merge;
    int mg_w = grid_w / merge;
    int out_dim = vh * merge * merge;  /* vh * 4 */

    vvis_merge_ctx c = { merged, patches, grid_h, grid_w, vh };
    vllm_tp_parfor(0, mg_h, vvis_merge_worker, &c);
}

/**
 * Merger forward: normalized → fc1 → GELU → fc2
 */
typedef struct {
    float *output; const float *input;
    int n_tokens, in_dim, out_dim, chunk_dim, full_norm, n_chunks, mid_dim;
    const float *norm_w, *norm_b, *fc1_w, *fc1_b, *fc2_w, *fc2_b;
    float norm_eps;
} vvis_merger_ctx;

static void vvis_merger_worker(void *ctx_, int i) {
    vvis_merger_ctx *c = (vvis_merger_ctx *)ctx_;
    const float *inp = c->input + (size_t)i * c->in_dim;
    float *outp = c->output + (size_t)i * c->out_dim;

    float *normed = (float*)alloca((size_t)c->in_dim * sizeof(float));
    if (c->full_norm) {
        layer_norm_f32(normed, inp, c->norm_w, c->norm_b, c->in_dim, c->norm_eps);
    } else {
        for (int chunk = 0; chunk < c->n_chunks; chunk++) {
            layer_norm_f32(normed + (size_t)chunk * c->chunk_dim,
                           inp + (size_t)chunk * c->chunk_dim,
                           c->norm_w, c->norm_b, c->chunk_dim, c->norm_eps);
        }
    }

    float *mid = (float*)alloca((size_t)c->mid_dim * sizeof(float));
    matvec_f32_bias(mid, c->fc1_w, normed, c->fc1_b, c->mid_dim, c->in_dim);
    for (int j = 0; j < c->mid_dim; j++) mid[j] = gelu_approx(mid[j]);
    matvec_f32_bias(outp, c->fc2_w, mid, c->fc2_b, c->out_dim, c->mid_dim);
}

static void merger_forward(float *__restrict output,
                            const float *__restrict input,
                            int n_tokens, int in_dim, int out_dim,
                            const float *norm_w, const float *norm_b,
                            const float *fc1_w, const float *fc1_b,
                            const float *fc2_w, const float *fc2_b,
                            int mid_dim, float norm_eps,
                            int chunk_dim, int full_norm) {
    int n_chunks = in_dim / chunk_dim;  /* 4 for 2x2 spatial merge */
    vvis_merger_ctx c = { output, input, n_tokens, in_dim, out_dim, chunk_dim,
                          full_norm, n_chunks, mid_dim, norm_w, norm_b,
                          fc1_w, fc1_b, fc2_w, fc2_b, norm_eps };
    vllm_tp_parfor(0, n_tokens, vvis_merger_worker, &c);
}

/**
 * DeepStack merger: run spatial merge + the deepstack layer's own merger
 * (full-norm, matching the official VisionSpatialMerger with
 * use_postshuffle_norm=True) on the BLOCK OUTPUT of a deepstack layer.
 * Produces ds_features[d] = [n_merged, vis_out_dim] for later injection
 * into the LLM's first layers (official _deepstack_process).
 */
static void vision_ds_merge(STVisionState *vis, int d,
                            int grid_t, int grid_h, int grid_w, int n_merged) {
    const STModelConfig *cfg = vis->cfg;
    int vh = cfg->vis_hidden;
    int merge = cfg->vis_merge;
    int merge_in = vh * merge * merge;
    int mg_h = grid_h / merge;
    int mg_w = grid_w / merge;
    int n_spatial = grid_h * grid_w;
    int out_dim = cfg->vis_out_dim;
    int ds_mid = merge_in;

    float *merged_buf = (float *)calloc((size_t)n_merged * merge_in, sizeof(float));
    for (int t = 0; t < grid_t; t++) {
        spatial_merge(merged_buf + (size_t)(t * mg_h * mg_w) * merge_in,
                      vis->hidden_states + (size_t)(t * n_spatial) * vh,
                      grid_h, grid_w, vh);
    }
    merger_forward(vis->ds_features[d], merged_buf, n_merged, merge_in, out_dim,
                   vis->w.ds_norm_weight + (size_t)d * merge_in,
                   vis->w.ds_norm_bias   + (size_t)d * merge_in,
                   vis->w.ds_fc1_weight  + (size_t)d * merge_in * ds_mid,
                   vis->w.ds_fc1_bias    + (size_t)d * ds_mid,
                   vis->w.ds_fc2_weight  + (size_t)d * ds_mid * out_dim,
                   vis->w.ds_fc2_bias    + (size_t)d * out_dim,
                   ds_mid, cfg->norm_eps, merge_in, 1);
    free(merged_buf);
}

/* ================================================================
 * Full Image Encoding Pipeline
 * ================================================================ */

int st_vision_encode_image(STVisionState *vis,
                            const uint8_t *image_data,
                            int width, int height) {
    const STModelConfig *cfg = vis->cfg;
    if (!cfg->has_vision) return -1;

    const int vis_dbg = getenv("VLLM_MMDBG") != NULL;
    double t_v = vis_dbg ? st_now_sec() : 0.0;
#define VIS_STEP(tag) do { if (vis_dbg) { \
        double _now = st_now_sec(); \
        fprintf(stderr, "[VIS] %-10s %.3fs (cum %.3fs)\n", tag, \
                _now - t_v, _now - t_v); \
        fflush(stderr); t_v = _now; } } while (0)

    int patch_size = cfg->vis_patch;
    int vh = cfg->vis_hidden;

    /* Step 1: Preprocess (resize + normalize) */
    int img_w, img_h;
    /* Allocate pixel buffer */
    free(vis->pixel_values);
    int max_pixels = (2048 / patch_size) * patch_size;
    max_pixels = max_pixels * max_pixels * 3;
    vis->pixel_values = calloc(max_pixels, sizeof(float));

    preprocess_image(vis->pixel_values, image_data, width, height,
                     &img_w, &img_h, patch_size, cfg->vis_max_pos);
    vis->img_width = img_w;
    vis->img_height = img_h;
    vis->n_frames = 1;
    VIS_STEP("preprocess");

    /* Step 2: Patch embedding */
    int n_patches = patch_embed_image(vis, vis->pixel_values, img_h, img_w);
    if (n_patches < 0) return -1;
    VIS_STEP("patch_embed");

    /* Step 3: Interpolated position embedding + vision RoPE table */
    vision_pos_encode(vis, n_patches, 1, vis->grid_h, vis->grid_w);
    VIS_STEP("pos_embed");

    /* Step 4: Copy to hidden_states for ViT processing */
    memcpy(vis->hidden_states, vis->patch_features,
           (size_t)n_patches * vh * sizeof(float));

    /* Step 5: ViT forward with DeepStack (features taken from the block
     * OUTPUT, matching the official forward: blk() then deepstack merger) */
    int grid_h = vis->grid_h;
    int grid_w = vis->grid_w;

    for (int l = 0; l < cfg->vis_depth; l++) {
        vit_block_forward(vis, l, n_patches);
        for (int d = 0; d < cfg->vis_ds_count; d++) {
            if (cfg->vis_ds_idx[d] == l) {
                /* Run the deepstack layer's own merger (full-norm) over the
                 * merged grid → ds_features[d] [n_merged, out_dim]. */
                int mg_h = grid_h / cfg->vis_merge;
                int mg_w = grid_w / cfg->vis_merge;
                vision_ds_merge(vis, d, 1, grid_h, grid_w, mg_h * mg_w);
                break;
            }
        }
        if (vis_dbg && (l == 0 || l == 5 || l == 10 || l == 15 || l == 20 ||
                        l == cfg->vis_depth - 1)) {
            double _now = st_now_sec();
            fprintf(stderr, "[VIS] vit l=%d/%d  %.3fs (cum %.3fs)\n", l,
                    cfg->vis_depth, _now - t_v, _now - t_v);
            fflush(stderr);
            t_v = _now;
        }
    }
    VIS_STEP("vit_all");

    /* Step 6: Spatial merge of final ViT output */
    int merge = cfg->vis_merge;
    int mg_h = grid_h / merge;
    int mg_w = grid_w / merge;
    int merge_in = vh * merge * merge;
    int n_merged = mg_h * mg_w;

    float *merged_buf = (float*)calloc((size_t)n_merged * merge_in, sizeof(float));
    spatial_merge(merged_buf, vis->hidden_states, grid_h, grid_w, vh);
    VIS_STEP("merge");

    /* Step 7: Main merger */
    int out_dim = cfg->vis_out_dim;  /* 4096 */
    float *main_tokens = (float*)calloc((size_t)n_merged * out_dim, sizeof(float));
    int merge_mid = merge_in;
    merger_forward(main_tokens, merged_buf, n_merged, merge_in, out_dim,
                   vis->w.merger_norm_weight, vis->w.merger_norm_bias,
                   vis->w.merger_fc1_weight, vis->w.merger_fc1_bias,
                   vis->w.merger_fc2_weight, vis->w.merger_fc2_bias,
                   merge_mid, cfg->norm_eps, vh, 0);

    /* Step 8: Visual token output = main merged tokens only.
     * DeepStack features (vis->ds_features) are kept for future layer-wise
     * injection into the LLM (official Qwen3-VL _deepstack_process) — they
     * are NOT part of the visual-token sequence. */
    vis->n_vis_tokens = n_merged;

    /* Reallocate visual tokens buffer if needed */
    free(vis->visual_tokens);
    vis->visual_tokens = calloc((size_t)vis->n_vis_tokens * out_dim, sizeof(float));

    /* Copy main merged tokens */
    memcpy(vis->visual_tokens, main_tokens,
           (size_t)n_merged * out_dim * sizeof(float));

    free(merged_buf);
    free(main_tokens);

    printf("[VIS] Image encoded: %dx%d → %d patches → %d visual tokens\n",
           img_w, img_h, n_patches, vis->n_vis_tokens);
    return vis->n_vis_tokens;
}

/* ================================================================
 * Video Encoding Pipeline
 * ================================================================ */

typedef struct {
    STVisionState *vis; const float *pw, *pb;
    int grid_h, grid_w, vh, temporal, patch_size, img_w, img_h, n_frames;
} vvis_vconv_ctx;

static void vvis_vconv_worker(void *ctx_, int t) {
    vvis_vconv_ctx *c = (vvis_vconv_ctx *)ctx_;
    for (int gy_v = 0; gy_v < c->grid_h; gy_v++) {
        for (int gx_v = 0; gx_v < c->grid_w; gx_v++) {
            int p_idx = (t * c->grid_h + gy_v) * c->grid_w + gx_v;
            float *feat = c->vis->patch_features + (size_t)p_idx * c->vh;

            for (int oc = 0; oc < c->vh; oc++) {
                double sum = 0.0;
                const float *w_oc = c->pw + (size_t)oc * 3 * c->temporal * c->patch_size * c->patch_size;

                int f_start = t * c->temporal;
                int f_end = (t + 1) * c->temporal;
                if (f_end > c->n_frames) f_end = c->n_frames;
                int frames_in_patch = f_end - f_start;

                for (int df = 0; df < frames_in_patch; df++) {
                    int frame_idx = f_start + df;
                    const float *frame = c->vis->pixel_values +
                        (size_t)frame_idx * c->img_w * c->img_h * 3;

                    for (int cc = 0; cc < 3; cc++) {
                        for (int py = 0; py < c->patch_size; py++) {
                            for (int px = 0; px < c->patch_size; px++) {
                                int pixel_y = gy_v * c->patch_size + py;
                                int pixel_x = gx_v * c->patch_size + px;
                                float pixel = frame[(pixel_y * c->img_w + pixel_x) * 3 + cc];
                                int w_idx = cc * c->temporal * c->patch_size * c->patch_size +
                                            df * c->patch_size * c->patch_size +
                                            py * c->patch_size + px;
                                sum += (double)w_oc[w_idx] * pixel;
                            }
                        }
                    }
                }
                feat[oc] = (float)sum + c->pb[oc];
            }
        }
    }
}

int st_vision_encode_video(STVisionState *vis,
                            const uint8_t **frame_data,
                            int width, int height, int n_frames) {
    const STModelConfig *cfg = vis->cfg;
    if (!cfg->has_vision) return -1;

    int patch_size = cfg->vis_patch;
    int temporal = cfg->vis_temporal;
    int vh = cfg->vis_hidden;

    /* Step 1: Preprocess each frame. grid_t is known up front (depends only
     * on n_frames and temporal), so each frame is downscaled to fit
     * vis_max_pos / grid_t spatial patches — the total grid then always
     * fits vis_max_pos. */
    int grid_t = (n_frames + temporal - 1) / temporal;
    int frame_max_grid = cfg->vis_max_pos / grid_t;
    if (frame_max_grid < 1) frame_max_grid = 1;

    int img_w, img_h;
    free(vis->pixel_values);
    int max_pixels = (2048 / patch_size) * patch_size;
    max_pixels = max_pixels * max_pixels * 3 * n_frames;
    vis->pixel_values = calloc(max_pixels, sizeof(float));

    vis->n_frames = n_frames;
    vis->img_width = width;
    vis->img_height = height;

    int frame_elems = 0;
    for (int f = 0; f < n_frames; f++) {
        int fw, fh;
        float *frame_pixels = vis->pixel_values + frame_elems;
        preprocess_image(frame_pixels, frame_data[f], width, height,
                         &fw, &fh, patch_size, frame_max_grid);
        if (f == 0) { img_w = fw; img_h = fh; }
        frame_elems += img_w * img_h * 3;
    }

    /* Step 2: Temporal patch embedding
     * Group frames into temporal patches of size `temporal`.
     * For each temporal patch, do 3D patch embedding. */
    int grid_h = img_h / patch_size;
    int grid_w = img_w / patch_size;
    int n_spatial = grid_h * grid_w;
    int n_patches = grid_t * n_spatial;

    vis->grid_h = grid_h;
    vis->grid_w = grid_w;
    vis->n_frames_eff = grid_t;

    if (n_patches > cfg->vis_max_pos) {
        fprintf(stderr, "[VIS] Too many video patches: %d > max %d\n",
                n_patches, cfg->vis_max_pos);
        return -1;
    }

    const float *pw = vis->w.patch_embed_weight;
    const float *pb = vis->w.patch_embed_bias;

    vvis_vconv_ctx vc = { vis, pw, pb, grid_h, grid_w, vh, temporal,
                          patch_size, img_w, img_h, n_frames };
    vllm_tp_parfor(0, grid_t, vvis_vconv_worker, &vc);

    /* Step 3: Interpolated position embedding + vision RoPE table
     * (each temporal patch reuses the same spatial positions) */
    vision_pos_encode(vis, n_patches, grid_t, grid_h, grid_w);

    /* Step 4: ViT forward (same as image; DeepStack features from block
     * output, matching the official forward) */
    memcpy(vis->hidden_states, vis->patch_features,
           (size_t)n_patches * vh * sizeof(float));

    for (int l = 0; l < cfg->vis_depth; l++) {
        vit_block_forward(vis, l, n_patches);
        for (int d = 0; d < cfg->vis_ds_count; d++) {
            if (cfg->vis_ds_idx[d] == l) {
                /* DeepStack merger per temporal group → [n_merged, out_dim] */
                int mg_h = grid_h / cfg->vis_merge;
                int mg_w = grid_w / cfg->vis_merge;
                vision_ds_merge(vis, d, grid_t, grid_h, grid_w, grid_t * mg_h * mg_w);
                break;
            }
        }
    }

    /* Step 5-8: Same merge + merger pipeline as image */
    int merge = cfg->vis_merge;
    int mg_h = grid_h / merge;
    int mg_w = grid_w / merge;
    int merge_in = vh * merge * merge;
    int n_merged = grid_t * mg_h * mg_w;

    float *merged_buf = (float*)calloc((size_t)n_merged * merge_in, sizeof(float));
    /* Spatial merge per temporal group */
    for (int t2 = 0; t2 < grid_t; t2++) {
        int src_off = t2 * n_spatial;
        int dst_off = t2 * mg_h * mg_w;
        spatial_merge(merged_buf + (size_t)dst_off * merge_in,
                      vis->hidden_states + (size_t)src_off * vh,
                      grid_h, grid_w, vh);
    }

    int out_dim = cfg->vis_out_dim;
    float *main_tokens = (float*)calloc((size_t)n_merged * out_dim, sizeof(float));
    int merge_mid = merge_in;
    merger_forward(main_tokens, merged_buf, n_merged, merge_in, out_dim,
                   vis->w.merger_norm_weight, vis->w.merger_norm_bias,
                   vis->w.merger_fc1_weight, vis->w.merger_fc1_bias,
                   vis->w.merger_fc2_weight, vis->w.merger_fc2_bias,
                   merge_mid, cfg->norm_eps, vh, 0);

    /* Visual token output = main merged tokens only (DeepStack features are
     * kept for future layer-wise injection, NOT part of the sequence). */
    vis->n_vis_tokens = n_merged;

    free(vis->visual_tokens);
    vis->visual_tokens = calloc((size_t)vis->n_vis_tokens * out_dim, sizeof(float));
    memcpy(vis->visual_tokens, main_tokens,
           (size_t)n_merged * out_dim * sizeof(float));

    free(merged_buf);
    free(main_tokens);

    printf("[VIS] Video encoded: %d frames %dx%d → %d patches → %d visual tokens\n",
           n_frames, img_w, img_h, n_patches, vis->n_vis_tokens);
    return vis->n_vis_tokens;
}

/* ================================================================
 * Accessor Functions
 * ================================================================ */

const float* st_vision_get_tokens(const STVisionState *vis) {
    return vis->visual_tokens;
}

int st_vision_get_num_tokens(const STVisionState *vis) {
    return vis->n_vis_tokens;
}
