/* Auto-extracted from src/model/vllm_safetensors.c (sha256 da768fbff3b10063a3815f8786b536892e0b7ec68ef6efe8a10180483bd27cc3)
 * Re-run tools/extract_quant.py after engine edits; verify sha256.
 * ================================================================ */

#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include "conv_platform.h"
#include "vqf_st.h"
#define Q8_BYTES(n) ((size_t)((n) + 31) / 32 * 34)
#define Q4_BYTES(n) ((size_t)((n) + 31) / 32 * 18)
#define Q2_BYTES(n) ((size_t)((n) + 31) / 32 * 16)
/* external: cfgio + quant + repack + wmode */
const STTensorInfo *find_tensor(const STModelConfig *cfg, const char *name);
int st_load_tensor(const STModelConfig *cfg, const char *name, float *dst, int n);
void f32_to_q8_0(uint8_t *o, const float *i, int n);
void f32_to_q4_0(uint8_t *o, const float *i, int n);
void f32_to_q4i8(uint8_t *o, const float *i, int n);
void f32_to_g256q8(uint8_t *o, const float *i, int n);
void f32_to_q2_1(uint8_t *o, const float *i, int n);
int repack_q4_0_4x4_inplace(uint8_t *b, int r, int c);
int repack_q8_0_8x8_inplace(uint8_t *b, int r, int c);
int repack_q8_0_4x4_inplace(uint8_t *b, int r, int c);
int repack_q8_0_tiled_inplace(uint8_t *b, int r, int c);
int repack_q4_0_8x8l(uint8_t *d, const uint8_t *s, int r, int c);
int st_wmode_effective(void);
extern int g_st_q8_repack, g_st_q4_repack, g_q2mix_tail;
extern int g_model_load_layer, g_model_load_total;
extern int g_st_no_q4;

/* load_layer_tensor (src 1889-1904, 归属本文件：loader 专用) */
static int load_layer_tensor(const STModelConfig *cfg, STModelWeights *w,
                              int layer, const char *suffix,
                              float *base_ptr, int elems_per_layer,
                              int expected_elems) {
    char name[256];
    snprintf(name, sizeof(name), "model.language_model.layers.%d.%s", layer, suffix);
    if (!find_tensor(cfg, name)) {
        /* Try alternate naming without language_model prefix */
        snprintf(name, sizeof(name), "model.layers.%d.%s", layer, suffix);
    }
    if (!find_tensor(cfg, name)) {
        fprintf(stderr, "[ST] WARNING: tensor not found for layer %d: %s\n", layer, suffix);
        return -1;
    }
    return st_load_tensor(cfg, name, base_ptr + layer * elems_per_layer, expected_elems);
}

/* ---- qwen3_moe 专家拼接助手（与稠密 ffn 整层缓冲同构）----
 * 每层 128 专家行连续堆叠后 gate/up: [n_experts*moe_ffn, dim]；down: [dim, n_experts*moe_ffn]
 * （HF expert gate/up shape [moe_ffn, dim] 行主序连续；down [dim, moe_ffn] 按列区段散开）。 */
static int moe_find(const STModelConfig *cfg, int layer, const char *suffix,
                    char *name, size_t name_sz) {
    snprintf(name, name_sz, "model.language_model.layers.%d.%s", layer, suffix);
    if (find_tensor(cfg, name)) return 1;
    snprintf(name, name_sz, "model.layers.%d.%s", layer, suffix);
    return find_tensor(cfg, name) ? 1 : 0;
}

static int load_moe_experts_rowband(const STModelConfig *cfg, int layer,
                                    const char *proj, float *dst) {
    /* dst: [n_experts*moe_ffn, dim]；专家 e 区段 = 行 [e*moe_ffn, (e+1)*moe_ffn)，连续 */
    int dim = cfg->dim, ne = cfg->n_experts, mf = cfg->moe_ffn;
    char name[256], suf[96];
    for (int e = 0; e < ne; e++) {
        snprintf(suf, sizeof(suf), "mlp.experts.%d.%s.weight", e, proj);
        if (!moe_find(cfg, layer, suf, name, sizeof(name))) {
            fprintf(stderr, "[ST] WARNING: MoE expert not found (layer %d): %s\n", layer, suf);
            return -1;
        }
        if (st_load_tensor(cfg, name, dst + (size_t)e * mf * dim, mf * dim) != 0)
            return -1;
    }
    return 0;
}

static int load_moe_experts_down(const STModelConfig *cfg, int layer, float *dst) {
    /* dst: [dim, n_experts*moe_ffn]；专家 e 列区段 = cols [e*moe_ffn, (e+1)*moe_ffn) */
    int dim = cfg->dim, ne = cfg->n_experts, mf = cfg->moe_ffn;
    float *tmp = malloc((size_t)dim * mf * sizeof(float));
    if (!tmp) { fprintf(stderr, "[ST] OOM MoE down tmp\n"); return -1; }
    char name[256], suf[96];
    for (int e = 0; e < ne; e++) {
        snprintf(suf, sizeof(suf), "mlp.experts.%d.down_proj.weight", e);
        if (!moe_find(cfg, layer, suf, name, sizeof(name))) {
            fprintf(stderr, "[ST] WARNING: MoE expert not found (layer %d): %s\n", layer, suf);
            free(tmp); return -1;
        }
        if (st_load_tensor(cfg, name, tmp, dim * mf) != 0) { free(tmp); return -1; }
        for (int r = 0; r < dim; r++)
            memcpy(dst + (size_t)r * (ne * mf) + (size_t)e * mf,
                   tmp + (size_t)r * mf, (size_t)mf * sizeof(float));
    }
    free(tmp);
    return 0;
}
/* ---- end MoE helpers ---- */
/* f16_to_f32_bits (src 2851-2869, Q8CHK 诊断用，同源副本) */
/* f16 (IEEE 754 half-precision) → f32 bit pattern */
static inline uint32_t f16_to_f32_bits(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    if (exp == 0) {
        if (mant == 0) return sign << 31;
        uint32_t m_norm = mant;
        int shift = 0;
        while (m_norm < 0x400) { m_norm <<= 1; shift++; }
        uint32_t f32_exp  = 113 - (uint32_t)shift;
        uint32_t f32_mant = (m_norm & 0x3FF) << 13;
        return (sign << 31) | (f32_exp << 23) | f32_mant;
    } else if (exp == 0x1F) {
        return (sign << 31) | (0xFF << 23) | (mant << 13);
    } else {
        return (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    }
}
/* st_load_layer_weights (src 1906-2290) */
int st_load_layer_weights(const STModelConfig *cfg, STModelWeights *w,
                           int layer_start, int layer_end) {
    int dim = cfg->dim;
    int nh = cfg->n_heads;
    int nkv = cfg->n_kv_heads;
    int hd = cfg->head_dim;
    int ff = cfg->ffn_dim;
    int q_out = nh * hd;   /* 4096 */
    int k_out = nkv * hd;  /* 1024 */

    /* Pre-allocate temp buffer for per-layer attention loading (64 MB).
     * Reused across all layers to avoid malloc/free fragmentation. */
    int max_attn_elems = dim * q_out;
    float *tmp_attn = NULL;
    if (!w->q_weight && (w->q8_q_weight || w->q4_q_weight)) {
        tmp_attn = malloc((size_t)max_attn_elems * sizeof(float));
        if (!tmp_attn) {
            fprintf(stderr, "[ST] OOM: cannot allocate temp attention buffer\n");
            return -1;
        }
    }

    for (int l = layer_start; l < layer_end; l++) {
        fprintf(stderr, "[ST] Loading layer %d/%d...\n", l, layer_end - 1);
        fflush(stderr);
        g_model_load_layer = l;
        g_model_load_total = layer_end - 1;

        load_layer_tensor(cfg, w, l, "input_layernorm.weight",
                           w->attn_norm, dim, dim);
        load_layer_tensor(cfg, w, l, "post_attention_layernorm.weight",
                           w->ffn_norm, dim, dim);
        /* Attention weights: load into temp buffer, quantize to Q8_0, free temp.
         * When F32 w->q_weight is NULL, we use per-layer temp buffers to avoid
         * allocating ~5.6 GB of F32 attention at peak. */
        if (w->q_weight) {
            /* Legacy: F32 attention pre-allocated */
            load_layer_tensor(cfg, w, l, "self_attn.q_proj.weight",
                               w->q_weight, dim * q_out, dim * q_out);
            load_layer_tensor(cfg, w, l, "self_attn.k_proj.weight",
                               w->k_weight, dim * k_out, dim * k_out);
            load_layer_tensor(cfg, w, l, "self_attn.v_proj.weight",
                               w->v_weight, dim * k_out, dim * k_out);
            load_layer_tensor(cfg, w, l, "self_attn.o_proj.weight",
                               w->o_weight, q_out * dim, q_out * dim);

            /* Quantize attention weights to Q8_0 */
            if (w->q8_q_weight) {
                f32_to_q8_0(w->q8_q_weight + Q8_BYTES((size_t)l * dim * q_out),
                            w->q_weight + (size_t)l * dim * q_out, dim * q_out);
                f32_to_q8_0(w->q8_k_weight + Q8_BYTES((size_t)l * dim * k_out),
                            w->k_weight + (size_t)l * dim * k_out, dim * k_out);
                f32_to_q8_0(w->q8_v_weight + Q8_BYTES((size_t)l * dim * k_out),
                            w->v_weight + (size_t)l * dim * k_out, dim * k_out);
                f32_to_q8_0(w->q8_o_weight + Q8_BYTES((size_t)l * q_out * dim),
                            w->o_weight + (size_t)l * q_out * dim, q_out * dim);
            }
        } else if (w->q8_q_weight || w->q4_q_weight) {
            /* Q8_0/Q4_0-only: load into pre-allocated tmp_attn, quantize, reuse */

            char name[256];

            /* Q projection */
            snprintf(name, sizeof(name), "model.language_model.layers.%d.self_attn.q_proj.weight", l);
            if (!find_tensor(cfg, name))
                snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.weight", l);
            st_load_tensor(cfg, name, tmp_attn, dim * q_out);
            if (w->q8_q_weight) {
                if (w->q8_buf_q4)
                    f32_to_q4i8(w->q8_q_weight + Q8_BYTES((size_t)l * dim * q_out),
                                tmp_attn, dim * q_out);
                else if (st_wmode_effective() == 4)
                    f32_to_g256q8(w->q8_q_weight + Q8_BYTES((size_t)l * dim * q_out),
                                  tmp_attn, dim * q_out);
                else
                    f32_to_q8_0(w->q8_q_weight + Q8_BYTES((size_t)l * dim * q_out),
                                tmp_attn, dim * q_out);
            }
            if (w->q4_q_weight)
                f32_to_q4_0(w->q4_q_weight + Q4_BYTES((size_t)l * dim * q_out),
                            tmp_attn, dim * q_out);

            /* K projection */
            snprintf(name, sizeof(name), "model.language_model.layers.%d.self_attn.k_proj.weight", l);
            if (!find_tensor(cfg, name))
                snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_proj.weight", l);
            st_load_tensor(cfg, name, tmp_attn, dim * k_out);
            if (w->q8_k_weight) {
                if (w->q8_buf_q4)
                    f32_to_q4i8(w->q8_k_weight + Q8_BYTES((size_t)l * dim * k_out),
                                tmp_attn, dim * k_out);
                else if (st_wmode_effective() == 4)
                    f32_to_g256q8(w->q8_k_weight + Q8_BYTES((size_t)l * dim * k_out),
                                  tmp_attn, dim * k_out);
                else
                    f32_to_q8_0(w->q8_k_weight + Q8_BYTES((size_t)l * dim * k_out),
                                tmp_attn, dim * k_out);
            }
            if (w->q4_k_weight)
                f32_to_q4_0(w->q4_k_weight + Q4_BYTES((size_t)l * dim * k_out),
                            tmp_attn, dim * k_out);

            /* V projection */
            snprintf(name, sizeof(name), "model.language_model.layers.%d.self_attn.v_proj.weight", l);
            if (!find_tensor(cfg, name))
                snprintf(name, sizeof(name), "model.layers.%d.self_attn.v_proj.weight", l);
            st_load_tensor(cfg, name, tmp_attn, dim * k_out);
            if (w->q8_v_weight) {
                if (w->q8_buf_q4)
                    f32_to_q4i8(w->q8_v_weight + Q8_BYTES((size_t)l * dim * k_out),
                                tmp_attn, dim * k_out);
                else if (st_wmode_effective() == 4)
                    f32_to_g256q8(w->q8_v_weight + Q8_BYTES((size_t)l * dim * k_out),
                                  tmp_attn, dim * k_out);
                else
                    f32_to_q8_0(w->q8_v_weight + Q8_BYTES((size_t)l * dim * k_out),
                                tmp_attn, dim * k_out);
            }
            if (w->q4_v_weight)
                f32_to_q4_0(w->q4_v_weight + Q4_BYTES((size_t)l * dim * k_out),
                            tmp_attn, dim * k_out);

            /* O projection */
            snprintf(name, sizeof(name), "model.language_model.layers.%d.self_attn.o_proj.weight", l);
            if (!find_tensor(cfg, name))
                snprintf(name, sizeof(name), "model.layers.%d.self_attn.o_proj.weight", l);
            st_load_tensor(cfg, name, tmp_attn, q_out * dim);
            if (w->q8_o_weight) {
                if (w->q8_buf_q4)
                    f32_to_q4i8(w->q8_o_weight + Q8_BYTES((size_t)l * q_out * dim),
                                tmp_attn, q_out * dim);
                else if (st_wmode_effective() == 4)
                    f32_to_g256q8(w->q8_o_weight + Q8_BYTES((size_t)l * q_out * dim),
                                  tmp_attn, q_out * dim);
                else
                    f32_to_q8_0(w->q8_o_weight + Q8_BYTES((size_t)l * q_out * dim),
                                tmp_attn, q_out * dim);
            }
            /* 权重 dump 自检（VLLM_Q8CHK=4）：O 第 0 层 F32 vs Q8 解包 */
            if (getenv("VLLM_Q8CHK") && getenv("VLLM_Q8CHK")[0] == '4' && l == 0 && w->q8_o_weight) {
                const uint8_t *o0 = w->q8_o_weight;
                fprintf(stderr, "[Q8CHK4] O l0 F32[0..8]: ");
                for (int i = 0; i < 8; i++) fprintf(stderr, "%.4f ", tmp_attn[i]);
                fprintf(stderr, "\n");
                for (int b = 0; b < 2; b++) {
                    uint32_t fb = f16_to_f32_bits(*(const uint16_t *)(o0 + (size_t)b * 34));
                    float d;
                    memcpy(&d, &fb, 4);
                    fprintf(stderr, "[Q8CHK4] O l0 block%d scale=%.6f qs:", b, d);
                    for (int i = 0; i < 8; i++)
                        fprintf(stderr, " %d", (int8_t)o0[(size_t)b * 34 + 2 + i]);
                    fprintf(stderr, "\n");
                    for (int i = 0; i < 8; i++) {
                        double dv = (double)(int8_t)o0[(size_t)b * 34 + 2 + i] * (double)d;
                        fprintf(stderr, "  [%d] F32=%.4f deq=%.4f\n", i, tmp_attn[(size_t)b * 32 + i], dv);
                    }
                }
                fflush(stderr);
            }
            if (w->q4_o_weight)
                f32_to_q4_0(w->q4_o_weight + Q4_BYTES((size_t)l * q_out * dim),
                            tmp_attn, q_out * dim);
        } else {
            /* No Q8_0: fallback to legacy scalar loading (needs F32 weights) */
            load_layer_tensor(cfg, w, l, "self_attn.q_proj.weight",
                               w->q_weight, dim * q_out, dim * q_out);
            load_layer_tensor(cfg, w, l, "self_attn.k_proj.weight",
                               w->k_weight, dim * k_out, dim * k_out);
            load_layer_tensor(cfg, w, l, "self_attn.v_proj.weight",
                               w->v_weight, dim * k_out, dim * k_out);
            load_layer_tensor(cfg, w, l, "self_attn.o_proj.weight",
                               w->o_weight, q_out * dim, q_out * dim);
        }

        if (cfg->has_q_norm) {
            load_layer_tensor(cfg, w, l, "self_attn.q_norm.weight",
                               w->q_norm, hd, hd);
            load_layer_tensor(cfg, w, l, "self_attn.k_norm.weight",
                               w->k_norm, hd, hd);
        }

        /* MoE router（qwen3_moe）：mlp.gate.weight [n_experts, dim] → f32 常驻
         * 堆叠 [n_layers*n_experts, dim]，top-k 选路边界精度敏感不量化。 */
        if (cfg->is_moe && w->moe_router) {
            load_layer_tensor(cfg, w, l, "mlp.gate.weight",
                               w->moe_router, cfg->n_experts * dim,
                               cfg->n_experts * dim);
        }

        /* FFN weights: load, then quantize to Q8_0 */
        if (w->gate_weight) {
            /* F32 FFN buffers exist: load normally, quantize after */
            if (cfg->is_moe) {
                load_moe_experts_rowband(cfg, l, "gate_proj",
                                         w->gate_weight + (size_t)l * ff * dim);
                load_moe_experts_rowband(cfg, l, "up_proj",
                                         w->up_weight + (size_t)l * ff * dim);
                load_moe_experts_down(cfg, l,
                                      w->down_weight + (size_t)l * dim * ff);
            } else {
                load_layer_tensor(cfg, w, l, "mlp.gate_proj.weight",
                                   w->gate_weight, ff * dim, ff * dim);
                load_layer_tensor(cfg, w, l, "mlp.up_proj.weight",
                                   w->up_weight, ff * dim, ff * dim);
                load_layer_tensor(cfg, w, l, "mlp.down_proj.weight",
                                   w->down_weight, dim * ff, dim * ff);
            }
        } else if (w->q8_gate_weight || w->q4_gate_weight || w->q2_gate_weight) {
            /* Q8_0/Q4_0/Q2_1-only: load into temp F32, quantize, free temp */
            float *tmp_ffn = malloc((size_t)(ff * dim) * sizeof(float));
            if (!tmp_ffn) { fprintf(stderr, "[ST] OOM temp FFN buf\n"); return -1; }

            /* q2mix layered precision: the first (nl - tail) layers pack Q2_1
             * (offset l), the last `tail` layers pack Q4_0 into the tail-only
             * segment (offset l - (nl - tail)). Non-q2mix keeps Q4 full-layer. */
            int qm_tail = g_q2mix_tail;
            if (qm_tail > w->n_layers_allocated) qm_tail = w->n_layers_allocated;
            const int q2_ffn = (w->q2_gate_weight) && (l < w->n_layers_allocated - qm_tail);
            const int q4_ffn_off = (w->q2_gate_weight && !q2_ffn) ? (l - (w->n_layers_allocated - qm_tail)) : l;

            if (cfg->is_moe)
                load_moe_experts_rowband(cfg, l, "gate_proj", tmp_ffn);
            else
                load_layer_tensor(cfg, w, l, "mlp.gate_proj.weight",
                                   tmp_ffn, 0, ff * dim);
            if (w->q8_gate_weight) {
                if (w->q8_buf_q4)
                    f32_to_q4i8(w->q8_gate_weight + (size_t)l * Q8_BYTES(ff * dim),
                                tmp_ffn, ff * dim);
                else if (st_wmode_effective() == 4)
                    f32_to_g256q8(w->q8_gate_weight + (size_t)l * Q8_BYTES(ff * dim),
                                  tmp_ffn, ff * dim);
                else
                    f32_to_q8_0(w->q8_gate_weight + (size_t)l * Q8_BYTES(ff * dim),
                                tmp_ffn, ff * dim);
            }
            if (w->q4_gate_weight && !q2_ffn)
                f32_to_q4_0(w->q4_gate_weight + (size_t)q4_ffn_off * Q4_BYTES(ff * dim),
                            tmp_ffn, ff * dim);
            if (w->q2_gate_weight && q2_ffn)
                f32_to_q2_1(w->q2_gate_weight + (size_t)l * Q2_BYTES(ff * dim),
                            tmp_ffn, ff * dim);

            if (cfg->is_moe)
                load_moe_experts_rowband(cfg, l, "up_proj", tmp_ffn);
            else
                load_layer_tensor(cfg, w, l, "mlp.up_proj.weight",
                                   tmp_ffn, 0, ff * dim);
            if (w->q8_up_weight) {
                if (w->q8_buf_q4)
                    f32_to_q4i8(w->q8_up_weight + (size_t)l * Q8_BYTES(ff * dim),
                                tmp_ffn, ff * dim);
                else if (st_wmode_effective() == 4)
                    f32_to_g256q8(w->q8_up_weight + (size_t)l * Q8_BYTES(ff * dim),
                                  tmp_ffn, ff * dim);
                else
                    f32_to_q8_0(w->q8_up_weight + (size_t)l * Q8_BYTES(ff * dim),
                                tmp_ffn, ff * dim);
            }
            if (w->q4_up_weight && !q2_ffn)
                f32_to_q4_0(w->q4_up_weight + (size_t)q4_ffn_off * Q4_BYTES(ff * dim),
                            tmp_ffn, ff * dim);
            if (w->q2_up_weight && q2_ffn)
                f32_to_q2_1(w->q2_up_weight + (size_t)l * Q2_BYTES(ff * dim),
                            tmp_ffn, ff * dim);

            if (cfg->is_moe)
                load_moe_experts_down(cfg, l, tmp_ffn);
            else
                load_layer_tensor(cfg, w, l, "mlp.down_proj.weight",
                                   tmp_ffn, 0, dim * ff);
            if (w->q8_down_weight) {
                if (w->q8_buf_q4)
                    f32_to_q4i8(w->q8_down_weight + (size_t)l * Q8_BYTES(dim * ff),
                                tmp_ffn, dim * ff);
                else if (st_wmode_effective() == 4)
                    f32_to_g256q8(w->q8_down_weight + (size_t)l * Q8_BYTES(dim * ff),
                                  tmp_ffn, dim * ff);
                else
                    f32_to_q8_0(w->q8_down_weight + (size_t)l * Q8_BYTES(dim * ff),
                                tmp_ffn, dim * ff);
            }
            if (w->q4_down_weight && !q2_ffn)
                f32_to_q4_0(w->q4_down_weight + (size_t)q4_ffn_off * Q4_BYTES(dim * ff),
                            tmp_ffn, dim * ff);
            if (w->q2_down_weight && q2_ffn)
                f32_to_q2_1(w->q2_down_weight + (size_t)l * Q2_BYTES(dim * ff),
                            tmp_ffn, dim * ff);

            free(tmp_ffn);
        }
    }

    w->has_q8 = (w->q8_gate_weight && w->q8_up_weight && w->q8_down_weight) ? 1 : 0;
    /* has_q4 covers the attention projections (q/k/v/o); the FFN Q4 buffers are
     * required only in dual/q4 modes — q2mix keeps attention Q4 + FFN Q2_1. */
    w->has_q4 = (w->q4_q_weight && w->q4_k_weight && w->q4_v_weight && w->q4_o_weight) ? 1 : 0;
    w->has_q2 = (w->q2_gate_weight && w->q2_up_weight && w->q2_down_weight) ? 1 : 0;
    if (w->q8_buf_q4) {
        fprintf(stderr, "[ST] Q8_0-layout buffers: pre-unpacked Q4_0 int8 (wmode=q4, nibble-free)\n");
    } else {
        fprintf(stderr, "[ST] FFN Q8_0 quantization: %s\n", w->has_q8 ? "enabled" : "NOT available");
    }
    fprintf(stderr, "[ST] FFN Q4_0 quantization: %s\n", w->has_q4 ? "enabled" : "NOT available");
    if (w->has_q2)
        fprintf(stderr, "[ST] FFN Q2_1 quantization (q2mix): enabled, %.0f%% weight-DRAM savings\n",
                100.0 * (1.0 - 16.0 / 18.0));
    if (g_st_no_q4) {
        w->has_q4 = 0;
        fprintf(stderr, "[ST] FFN Q4_0 quantization: disabled by --no-q4 (using Q8_0)\n");
    }
    /* M4: repack the Q4_0 layer weights into the 4x4 nibble-unpack-free
     * layout (same byte size, in-place per 4-row group). Matrices whose
     * shape isn't 4-row aligned are left in the legacy layout and the
     * kernels fall back to the unpack path automatically. */
    if (w->has_q4) {
        int l;
        for (l = layer_start; l < layer_end; l++) {
            /* q2mix allocates Q4 attention only; guard each projection so
             * the absent FFN Q4 buffers (NULL) are skipped. */
            if (w->q4_q_weight)
                repack_q4_0_4x4_inplace(w->q4_q_weight + (size_t)l * Q4_BYTES((size_t)dim * q_out),
                                        q_out, dim);
            if (w->q4_k_weight)
                repack_q4_0_4x4_inplace(w->q4_k_weight + (size_t)l * Q4_BYTES((size_t)dim * k_out),
                                        k_out, dim);
            if (w->q4_v_weight)
                repack_q4_0_4x4_inplace(w->q4_v_weight + (size_t)l * Q4_BYTES((size_t)dim * k_out),
                                        k_out, dim);
            if (w->q4_o_weight)
                repack_q4_0_4x4_inplace(w->q4_o_weight + (size_t)l * Q4_BYTES((size_t)q_out * dim),
                                        dim, q_out);
            /* q2mix uses a tail-only FFN Q4 segment (different stride), so the
             * full-layer 4x4 repack below is skipped for it. */
            if (w->q4_gate_weight && !w->has_q2)
                repack_q4_0_4x4_inplace(w->q4_gate_weight + (size_t)l * Q4_BYTES((size_t)ff * dim),
                                        ff, dim);
            if (w->q4_up_weight && !w->has_q2)
                repack_q4_0_4x4_inplace(w->q4_up_weight + (size_t)l * Q4_BYTES((size_t)ff * dim),
                                        ff, dim);
            if (w->q4_down_weight && !w->has_q2)
                repack_q4_0_4x4_inplace(w->q4_down_weight + (size_t)l * Q4_BYTES((size_t)dim * ff),
                                        dim, ff);
        }
    }
    /* M4e: repack the Q8_0 layer weights into llama block_q8_0x4 (136 B/block,
     * same layout family as Q4's M4d above). 2026-08-22: prefill Q8 差距
     * 3.18x 的根因与 Q4 相同（batched GEMV 每 token tile 重读权重）; repack 后
     * prefill 走 q8x4_gemm_batched、decode 走 4x4 解交织 GEMV。门控：Q8 缓冲区
     * 存在（has_q8）、非 g256/q4i 布局（st_weights_alloc 已清 g_st_q8_repack）、
     * 几何全对齐（几何门控已在分配期执行）。任一矩阵 repack 失败会清零
     * g_st_q8_repack，之后的矩阵不再重排（保持全局统一 legacy 布局）。 */
    if (w->has_q8 && g_st_q8_repack) {
        int l;
        for (l = layer_start; l < layer_end; l++) {
            repack_q8_0_tiled_inplace(w->q8_q_weight + (size_t)l * Q8_BYTES((size_t)dim * q_out),
                                      q_out, dim);
            repack_q8_0_tiled_inplace(w->q8_k_weight + (size_t)l * Q8_BYTES((size_t)dim * k_out),
                                      k_out, dim);
            repack_q8_0_tiled_inplace(w->q8_v_weight + (size_t)l * Q8_BYTES((size_t)dim * k_out),
                                      k_out, dim);
            repack_q8_0_tiled_inplace(w->q8_o_weight + (size_t)l * Q8_BYTES((size_t)q_out * dim),
                                      dim, q_out);
            repack_q8_0_tiled_inplace(w->q8_gate_weight + (size_t)l * Q8_BYTES((size_t)ff * dim),
                                      ff, dim);
            repack_q8_0_tiled_inplace(w->q8_up_weight + (size_t)l * Q8_BYTES((size_t)ff * dim),
                                      ff, dim);
            repack_q8_0_tiled_inplace(w->q8_down_weight + (size_t)l * Q8_BYTES((size_t)dim * ff),
                                      dim, ff);
        }
    }
    /* P4: fill the 16x8 prefill/decode Q4 copies (16 rows x 256 cols tile,
     * same byte size as Q4_0). All-or-nothing: any geometry mismatch clears
     * has_x8 and prefill keeps the legacy path. */
    if (w->has_x8 && w->x8_q_weight && w->x8_gate_weight) {
        int l, ok = 1;
        struct { uint8_t *dst, *src; int rows, cols; } m7[7];
        for (l = layer_start; l < layer_end; l++) {
            m7[0].dst = w->x8_q_weight + (size_t)l * Q4_BYTES((size_t)dim * q_out);
            m7[0].src = w->q4_q_weight + (size_t)l * Q4_BYTES((size_t)dim * q_out);
            m7[0].rows = q_out; m7[0].cols = dim;
            m7[1].dst = w->x8_k_weight + (size_t)l * Q4_BYTES((size_t)dim * k_out);
            m7[1].src = w->q4_k_weight + (size_t)l * Q4_BYTES((size_t)dim * k_out);
            m7[1].rows = k_out; m7[1].cols = dim;
            m7[2].dst = w->x8_v_weight + (size_t)l * Q4_BYTES((size_t)dim * k_out);
            m7[2].src = w->q4_v_weight + (size_t)l * Q4_BYTES((size_t)dim * k_out);
            m7[2].rows = k_out; m7[2].cols = dim;
            m7[3].dst = w->x8_o_weight + (size_t)l * Q4_BYTES((size_t)q_out * dim);
            m7[3].src = w->q4_o_weight + (size_t)l * Q4_BYTES((size_t)q_out * dim);
            m7[3].rows = dim; m7[3].cols = q_out;
            m7[4].dst = w->x8_gate_weight + (size_t)l * Q4_BYTES((size_t)ff * dim);
            m7[4].src = w->q4_gate_weight + (size_t)l * Q4_BYTES((size_t)ff * dim);
            m7[4].rows = ff; m7[4].cols = dim;
            m7[5].dst = w->x8_up_weight + (size_t)l * Q4_BYTES((size_t)ff * dim);
            m7[5].src = w->q4_up_weight + (size_t)l * Q4_BYTES((size_t)ff * dim);
            m7[5].rows = ff; m7[5].cols = dim;
            m7[6].dst = w->x8_down_weight + (size_t)l * Q4_BYTES((size_t)dim * ff);
            m7[6].src = w->q4_down_weight + (size_t)l * Q4_BYTES((size_t)dim * ff);
            m7[6].rows = dim; m7[6].cols = ff;
            for (int m = 0; m < 7; m++)
                if (repack_q4_0_8x8l(m7[m].dst, m7[m].src, m7[m].rows, m7[m].cols) != 0) ok = 0;
        }
        w->has_x8 = ok;
        if (ok)
            fprintf(stderr, "[ST] Q4 16x8 repack: enabled (7 proj x %d layers)\n", layer_end - layer_start);
        else
            fprintf(stderr, "[ST] Q4 16x8 repack: geometry mismatch, legacy path kept\n");
    }
    free(tmp_attn);
    return 0;
}
