/* ================================================================
 * conv_main.c - vqf_convert 主程序（跨平台）：safetensors -> VQF。
 *
 * 流程与引擎 --convert-vqf 一致（同一 loader / 量化 / repack 内核，
 * 见 quant_kernels.c + model_layers.c），保证产物与引擎逐字节一致：
 *   parse -> alloc(dual 布局) -> embed/final_norm -> lm_head chunked
 *   -> st_load_layer_weights(逐层量化+repack) -> 明文 VQF 写出 + FNV。
 *
 * v1 范围：文本模型；wmode=dual(0)/q4(1)/q8(2)/q4i(3)/g256(4)；
 * vision / VQF-Enc / SM2 内嵌签名不在 v1（后续补）。
 *
 * 用法：
 *   vqf_convert --model <safetensors dir> --convert-vqf <out.vqf>
 *              [--wmode dual|q4|q8|q4i|g256]
 * ================================================================ */
#include "conv_platform.h"
#include "vqf_st.h"
#include "vqf_format.h"
#include "vqf_vision.h"
#include <stdio.h>
#include <string.h>

/* ---- 量化 / repack（外部模块） ---- */
void f32_to_q8_0(uint8_t *o, const float *i, int n);
void f32_to_q4_0(uint8_t *o, const float *i, int n);
void f32_to_q4i8(uint8_t *o, const float *i, int n);
void f32_to_g256q8(uint8_t *o, const float *i, int n);
int repack_q8_0_tiled_inplace(uint8_t *b, int r, int c);
int repack_q4_0_4x4_inplace(uint8_t *b, int r, int c);
int repack_q4_0_8x8l(uint8_t *d, const uint8_t *s, int r, int c);

/* ---- cfgio / layers ---- */
int st_parse_config(const char *dir, STModelConfig *cfg);
int st_load_tensor(const STModelConfig *cfg, const char *name, float *dst, int n);
int st_load_tensor_slice(const STModelConfig *cfg, const char *name, int64_t off, float *dst, int64_t n);
void st_config_free(STModelConfig *cfg);
int st_load_layer_weights(const STModelConfig *cfg, STModelWeights *w, int a, int b);

/* 引擎抽取层引用的全局（默认值 = 引擎状态：repack 开）。*/
int g_st_q8_repack = 1;
int g_st_q4_repack = 1;
int g_st_no_q4 = 0;
int g_q2mix_tail = 0;
int g_model_load_layer = -1;
int g_model_load_total = 0;

/* wmode（dual=0/q4=1/q8=2/q4i=3/g256=4；q2mix=5 v1 拒） */
static int g_wmode = 0;
int st_wmode_effective(void) { return g_wmode; }

#define Q8B(n) (((size_t)(n) + 31) / 32 * 34)
#define Q4B(n) (((size_t)(n) + 31) / 32 * 18)

static void w_free(STModelWeights *w) {
    /* full calloc layout: free in alloc order (NULL-safe) */
    free(w->token_embed); free(w->final_norm); free(w->moe_router);
    free(w->attn_norm); free(w->ffn_norm);
    free(w->q_norm); free(w->k_norm);
    free(w->q8_lm_weight); free(w->q8_q_weight); free(w->q8_k_weight);
    free(w->q8_v_weight); free(w->q8_o_weight); free(w->q8_gate_weight);
    free(w->q8_up_weight); free(w->q8_down_weight);
    free(w->q4_lm_weight); free(w->q4_q_weight); free(w->q4_k_weight);
    free(w->q4_v_weight); free(w->q4_o_weight); free(w->q4_gate_weight);
    free(w->q4_up_weight); free(w->q4_down_weight);
    free(w->x8_q_weight); free(w->x8_k_weight); free(w->x8_v_weight);
    free(w->x8_o_weight); free(w->x8_gate_weight); free(w->x8_up_weight);
    free(w->x8_down_weight);
    if (w->vision) { st_vision_weights_free(w->vision); free(w->vision); w->vision = NULL; }
    memset(w, 0, sizeof(*w));
}

/* alloc：与引擎 st_weights_alloc_layers_q8ffn 同布局（per-layer Q8B/Q4B
 * 对齐，loader 按绝对层偏移写）。q2mix(vm5) 不支持。*/
static int w_alloc(STModelWeights *w, const STModelConfig *cfg) {
    memset(w, 0, sizeof(*w));
    int dim = cfg->dim, ff = cfg->ffn_dim, vc = cfg->vocab_size;
    int nl = cfg->n_layers;
    int q_out = cfg->n_heads * cfg->head_dim;
    int k_out = cfg->n_kv_heads * cfg->head_dim;
    int wm = g_wmode;

#define C(n, f) do { f = calloc(1, (size_t)(n)); if (!f) { w_free(w); return -1; } } while (0)
    C((size_t)vc * dim * 4, w->token_embed);
    C((size_t)dim * 4, w->final_norm);
    if (cfg->is_moe)   /* MoE router f32 resident [nl*n_experts, dim] */
        C((size_t)nl * cfg->n_experts * dim * 4, w->moe_router);
    C((size_t)nl * dim * 4, w->attn_norm);
    C((size_t)nl * dim * 4, w->ffn_norm);
    if (cfg->has_q_norm) {
        C((size_t)nl * cfg->head_dim * 4, w->q_norm);
        C((size_t)nl * cfg->head_dim * 4, w->k_norm);
    }
    if (wm != 1 && wm != 5) {   /* q8 鍖?*/
        C(Q8B((size_t)vc * dim), w->q8_lm_weight);
        C((size_t)nl * Q8B((size_t)dim * q_out), w->q8_q_weight);
        C((size_t)nl * Q8B((size_t)dim * k_out), w->q8_k_weight);
        C((size_t)nl * Q8B((size_t)dim * k_out), w->q8_v_weight);
        C((size_t)nl * Q8B((size_t)q_out * dim), w->q8_o_weight);
        C((size_t)nl * Q8B((size_t)ff * dim), w->q8_gate_weight);
        C((size_t)nl * Q8B((size_t)ff * dim), w->q8_up_weight);
        C((size_t)nl * Q8B((size_t)dim * ff), w->q8_down_weight);
    }
    if (wm == 0 || wm == 1) {   /* q4 + x8 鍖?*/
        C(Q4B((size_t)vc * dim), w->q4_lm_weight);
        C((size_t)nl * Q4B((size_t)dim * q_out), w->q4_q_weight);
        C((size_t)nl * Q4B((size_t)dim * k_out), w->q4_k_weight);
        C((size_t)nl * Q4B((size_t)dim * k_out), w->q4_v_weight);
        C((size_t)nl * Q4B((size_t)q_out * dim), w->q4_o_weight);
        C((size_t)nl * Q4B((size_t)ff * dim), w->q4_gate_weight);
        C((size_t)nl * Q4B((size_t)ff * dim), w->q4_up_weight);
        C((size_t)nl * Q4B((size_t)dim * ff), w->q4_down_weight);
    }
    /* x8（8x8l decode 副本）：与引擎 alloc 同 gate —— 仅 wm==1 且
     * VLLM_ENABLE_8X8L=1 时分配（dual/q4i/q8/g256 不分配）。 */
    if (wm == 1) {
        const char *e8 = getenv("VLLM_ENABLE_8X8L");
        if (e8 && e8[0] && e8[0] != '0') {
            C((size_t)nl * Q4B((size_t)ff * dim), w->x8_gate_weight);
            C((size_t)nl * Q4B((size_t)ff * dim), w->x8_up_weight);
            C((size_t)nl * Q4B((size_t)dim * ff), w->x8_down_weight);
            C((size_t)nl * Q4B((size_t)dim * q_out), w->x8_q_weight);
            C((size_t)nl * Q4B((size_t)dim * k_out), w->x8_k_weight);
            C((size_t)nl * Q4B((size_t)dim * k_out), w->x8_v_weight);
            C((size_t)nl * Q4B((size_t)q_out * dim), w->x8_o_weight);
            w->has_x8 = 1;
        }
    }
#undef C
    w->n_layers_allocated = nl;
    w->is_allocated = 1;
    if (wm == 3) w->q8_buf_q4 = 1;
    return 0;
}

/* ---- 明文 VQF 写出（文本模型；布局/顺序 = 引擎 vqf_collect 文本区） ---- */
typedef struct {
    const char *name;
    uint32_t qtype; uint32_t rows, cols;
    const void *ptr;
    uint64_t off, bytes;
} VQ_ENT;

#define ADDQ(nm, qt, p, r, c)                                                \
    do { if (n < 96 && (p)) {                                                \
        d[n].name = (nm); d[n].qtype = (qt); d[n].ptr = (p);                    \
        d[n].rows = (r); d[n].cols = (c);                                    \
        d[n].bytes = vq_bytes((qt), (uint64_t)(r) * (c)); n++; } } while (0)

static size_t vq_bytes(uint32_t qt, uint64_t n) {
    switch (qt) {
    case 0x01: return (size_t)(n * 4);
    case 0x05: return (size_t)(n * 2);
    case 0x02: return (size_t)((n + 31) / 32 * 34);
    default:   return (size_t)((n + 31) / 32 * 18);
    }
}

static int collect_text(const STModelWeights *w, const STModelConfig *cfg,
                        VQ_ENT *d) {
    int n = 0;
    int dim = cfg->dim, ff = cfg->ffn_dim, vc = cfg->vocab_size, nl = cfg->n_layers;
    int q_out = cfg->n_heads * cfg->head_dim;
    int k_out = cfg->n_kv_heads * cfg->head_dim;
    ADDQ("token_embed", 0x05 /* F16 */, w->token_embed, vc, dim);
    ADDQ("final_norm",  0x01, w->final_norm, dim, 1);
    ADDQ("attn_norm",   0x01, w->attn_norm, nl * dim, 1);
    ADDQ("ffn_norm",    0x01, w->ffn_norm, nl * dim, 1);
    if (cfg->has_q_norm) {
        ADDQ("q_norm", 0x01, w->q_norm, nl * cfg->head_dim, 1);
        ADDQ("k_norm", 0x01, w->k_norm, nl * cfg->head_dim, 1);
    }
    /* MoE router: f32 resident [nl*n_experts, dim]; dense -> NULL skipped.
     * Order anchored AFTER norms / BEFORE q8_* (engine collect mirror). */
    if (cfg->is_moe)
        ADDQ("moe_router", 0x01, w->moe_router, nl * cfg->n_experts, dim);
    ADDQ("q8_q", 0x02, w->q8_q_weight, nl * dim, q_out);
    ADDQ("q8_k", 0x02, w->q8_k_weight, nl * dim, k_out);
    ADDQ("q8_v", 0x02, w->q8_v_weight, nl * dim, k_out);
    ADDQ("q8_o", 0x02, w->q8_o_weight, nl * q_out, dim);
    ADDQ("q8_gate", 0x02, w->q8_gate_weight, nl * ff, dim);
    ADDQ("q8_up",   0x02, w->q8_up_weight, nl * ff, dim);
    ADDQ("q8_down", 0x02, w->q8_down_weight, nl * dim, ff);
    ADDQ("q8_lm",   0x02, w->q8_lm_weight, vc, dim);
    ADDQ("q4_q", 0x03, w->q4_q_weight, nl * dim, q_out);
    ADDQ("q4_k", 0x03, w->q4_k_weight, nl * dim, k_out);
    ADDQ("q4_v", 0x03, w->q4_v_weight, nl * dim, k_out);
    ADDQ("q4_o", 0x03, w->q4_o_weight, nl * q_out, dim);
    ADDQ("q4_gate", 0x03, w->q4_gate_weight, nl * ff, dim);
    ADDQ("q4_up",   0x03, w->q4_up_weight, nl * ff, dim);
    ADDQ("q4_down", 0x03, w->q4_down_weight, nl * dim, ff);
    ADDQ("q4_lm",   0x03, w->q4_lm_weight, vc, dim);
    ADDQ("x8_q", 0x04, w->x8_q_weight, nl * dim, q_out);
    ADDQ("x8_k", 0x04, w->x8_k_weight, nl * dim, k_out);
    ADDQ("x8_v", 0x04, w->x8_v_weight, nl * dim, k_out);
    ADDQ("x8_o", 0x04, w->x8_o_weight, nl * q_out, dim);
    ADDQ("x8_gate", 0x04, w->x8_gate_weight, nl * ff, dim);
    ADDQ("x8_up",   0x04, w->x8_up_weight, nl * ff, dim);
    ADDQ("x8_down", 0x04, w->x8_down_weight, nl * dim, ff);
    /* Vision 区（多模态，镜像引擎 vqf_collect 193-231 顺序）：
     * Q8 大矩阵 + F32 小张量；fc2 col 按 vf_pad 存。 */
    const STVisionWeights *vw = w->vision;
    if (vw && vw->is_allocated) {
        int vh = cfg->vis_hidden, vd = cfg->vis_depth, vf = cfg->vis_ffn;
        int vp = cfg->vis_patch, vt = cfg->vis_temporal;
        int vf_pad = ((vf + 31) / 32) * 32;
        int merge_in = vh * cfg->vis_merge * cfg->vis_merge;
        int ds_mid = merge_in, out_dim = cfg->vis_out_dim, ds = cfg->vis_ds_count;
        ADDQ("v_q8_qkv", 0x02, vw->q8_attn_qkv_weight, vd, 3 * vh * vh);
        ADDQ("v_q8_proj", 0x02, vw->q8_attn_proj_weight, vd, vh * vh);
        ADDQ("v_q8_fc1", 0x02, vw->q8_mlp_fc1_weight, vd, vf * vh);
        ADDQ("v_q8_fc2", 0x02, vw->q8_mlp_fc2_weight, vd, vh * vf_pad);
        ADDQ("v_patch_embed", 0x01, vw->patch_embed_weight, 1, vh * 3 * vt * vp * vp);
        ADDQ("v_patch_bias",  0x01, vw->patch_embed_bias,   1, vh);
        ADDQ("v_pos_embed",   0x01, vw->pos_embed,          1, cfg->vis_max_pos * vh);
        ADDQ("v_qkv_bias",    0x01, vw->attn_qkv_bias,      vd, 3 * vh);
        ADDQ("v_proj_bias",   0x01, vw->attn_proj_bias,     vd, vh);
        ADDQ("v_norm1_w",     0x01, vw->norm1_weight,       vd, vh);
        ADDQ("v_norm1_b",     0x01, vw->norm1_bias,         vd, vh);
        ADDQ("v_norm2_w",     0x01, vw->norm2_weight,       vd, vh);
        ADDQ("v_norm2_b",     0x01, vw->norm2_bias,         vd, vh);
        ADDQ("v_fc1_bias",    0x01, vw->mlp_fc1_bias,       vd, vf);
        ADDQ("v_fc2_bias",    0x01, vw->mlp_fc2_bias,       vd, vh);
        ADDQ("v_merger_nw",   0x01, vw->merger_norm_weight, 1, vh);
        ADDQ("v_merger_nb",   0x01, vw->merger_norm_bias,   1, vh);
        ADDQ("v_merger_f1w",  0x01, vw->merger_fc1_weight,  1, merge_in * merge_in);
        ADDQ("v_merger_f1b",  0x01, vw->merger_fc1_bias,    1, merge_in);
        ADDQ("v_merger_f2w",  0x01, vw->merger_fc2_weight,  1, merge_in * out_dim);
        ADDQ("v_merger_f2b",  0x01, vw->merger_fc2_bias,    1, out_dim);
        ADDQ("v_ds_nw",  0x01, vw->ds_norm_weight, ds, merge_in);
        ADDQ("v_ds_nb",  0x01, vw->ds_norm_bias,   ds, merge_in);
        ADDQ("v_ds_f1w", 0x01, vw->ds_fc1_weight,  ds, merge_in * ds_mid);
        ADDQ("v_ds_f1b", 0x01, vw->ds_fc1_bias,    ds, ds_mid);
        ADDQ("v_ds_f2w", 0x01, vw->ds_fc2_weight,  ds, ds_mid * out_dim);
        ADDQ("v_ds_f2b", 0x01, vw->ds_fc2_bias,    ds, out_dim);
    }
    return n;
}

/* f32 -> f16（与 vqf.c vqf_f32_to_f16 位级一致） */
static uint16_t f2h(float x) {
    uint32_t u; memcpy(&u, &x, 4);
    uint32_t s = (u >> 16) & 0x8000;
    int32_t e = (int32_t)((u >> 23) & 0xFF) - 127 + 15;
    uint32_t m = (u >> 13) & 0x3FF;
    if (e <= 0) return (uint16_t)(s | 0);
    if (e >= 0x1F) return (uint16_t)(s | 0x7C00);
    return (uint16_t)(s | (e << 10) | (m & 0x3FF));
}

static int write_vqf(const char *path, const STModelWeights *w,
                     const STModelConfig *cfg, uint32_t flags) {
    VQ_ENT d[96];
    int n = collect_text(w, cfg, d);
    if (n <= 0) { fprintf(stderr, "[conv] no tensors\n"); return -1; }
    size_t dir_off = (sizeof(VQFHeader) + 63) & ~(size_t)63;
    size_t data_off = (dir_off + (size_t)n * sizeof(VQFTensor) + VQF_HEADER_BYTES - 1)
                      & ~(size_t)(VQF_HEADER_BYTES - 1);
    uint64_t off = data_off;
    for (int i = 0; i < n; i++) { d[i].off = off; off = (off + d[i].bytes + 63) & ~(uint64_t)63; }
    uint64_t data_end = off;

    FILE *f = fopen(path, "wb+");   /* wb+：写后需 fread 回读重算尾部 FNV */
    if (!f) return -1;
    VQFHeader h; memset(&h, 0, sizeof(h));
    h.magic = VQF_MAGIC;
    h.version = cfg->is_moe ? VQF_VERSION_MOE : VQF_VERSION;
    h.n_tensors = (uint32_t)n;
    h.arch.dim = (uint32_t)cfg->dim; h.arch.n_layers = (uint32_t)cfg->n_layers;
    h.arch.n_heads = (uint32_t)cfg->n_heads; h.arch.n_kv_heads = (uint32_t)cfg->n_kv_heads;
    h.arch.head_dim = (uint32_t)cfg->head_dim; h.arch.ffn_dim = (uint32_t)cfg->ffn_dim;
    h.arch.vocab_size = (uint32_t)cfg->vocab_size; h.arch.max_seq_len = (uint32_t)cfg->max_seq_len;
    h.arch.rope_theta = cfg->rope_theta; h.arch.norm_eps = cfg->norm_eps;
    h.arch.bos_id = (uint32_t)cfg->bos_id; h.arch.eos_id = (uint32_t)cfg->eos_id;
    h.arch.has_q_norm = (uint32_t)cfg->has_q_norm; h.arch.has_mrope = (uint32_t)cfg->has_mrope;
    h.arch.head_dim_full = (uint32_t)cfg->head_dim_full;
    h.arch.kv_lora_rank = (uint32_t)cfg->kv_lora_rank;
    /* arch 全字段镜像引擎 vqf.c vqf_write（行 302-316）——mrope/image/video
     * 等必须逐字节一致，否则尾部 FNV 与 sha256 对不上。v1 仅文本，vision 相关
     * 字段按 cfg 原值写入（合成/文本模型均为 0）。 */
    h.arch.has_vision = (uint32_t)cfg->has_vision; h.arch.vis_depth = (uint32_t)cfg->vis_depth;
    h.arch.vis_hidden = (uint32_t)cfg->vis_hidden; h.arch.vis_heads = (uint32_t)cfg->vis_heads;
    h.arch.vis_ffn = (uint32_t)cfg->vis_ffn; h.arch.vis_patch = (uint32_t)cfg->vis_patch;
    h.arch.vis_temporal = (uint32_t)cfg->vis_temporal; h.arch.vis_merge = (uint32_t)cfg->vis_merge;
    h.arch.vis_out_dim = (uint32_t)cfg->vis_out_dim; h.arch.vis_in_chan = (uint32_t)cfg->vis_in_chan;
    h.arch.vis_max_pos = (uint32_t)cfg->vis_max_pos; h.arch.vis_ds_count = (uint32_t)cfg->vis_ds_count;
    h.arch.mrope_n_sec = (uint32_t)cfg->mrope_n_sec;
    for (int i = 0; i < 4; i++) {
        h.arch.vis_ds_idx[i] = (int32_t)cfg->vis_ds_idx[i];
        h.arch.mrope_sections[i] = (int32_t)cfg->mrope_sections[i];
    }
    h.arch.vision_start_id = (uint32_t)cfg->vision_start_id;
    h.arch.vision_end_id = (uint32_t)cfg->vision_end_id;
    h.arch.image_token_id = (uint32_t)cfg->image_token_id;
    h.arch.video_token_id = (uint32_t)cfg->video_token_id;
    /* MoE arch tail (v3; dense 全 0 保持 v2 canonical 不变) */
    h.arch.n_experts = (uint32_t)cfg->n_experts;
    h.arch.moe_ffn = (uint32_t)cfg->moe_ffn;
    h.arch.top_k = (uint32_t)cfg->top_k;
    h.arch.shared_experts = (uint32_t)cfg->shared_experts;
    h.flags = flags; h.data_offset = data_off;

    size_t head_sz = dir_off + (size_t)n * sizeof(VQFTensor);
    uint8_t *buf = (uint8_t *)calloc(1, head_sz);
    memcpy(buf, &h, sizeof(h));
    for (int i = 0; i < n; i++) {
        VQFTensor *t = (VQFTensor *)(buf + dir_off + (size_t)i * sizeof(VQFTensor));
        snprintf(t->name, sizeof(t->name), "%s", d[i].name);
        t->qtype = d[i].qtype; t->rows = d[i].rows; t->cols = d[i].cols;
        t->offset = d[i].off; t->bytes = d[i].bytes;
    }
    if (fwrite(buf, 1, head_sz, f) != head_sz) { free(buf); fclose(f); return -1; }
    free(buf);

    /* 数据区补齐：head_sz..data_off 的零 pad（不参与 FNV，与引擎一致） */
    {
        uint8_t pad[64] = {0};
        long pos = (long)head_sz;
        while (pos < (long)data_off) {
            size_t wb = ((long)data_off - pos) > 64 ? 64 : (size_t)((long)data_off - pos);
            if (fwrite(pad, 1, wb, f) != wb) { fclose(f); return -1; }
            pos += (long)wb;
        }
    }
    /* 逐张量写数据（F16：f32 -> f16 分块转换）；张量间零 pad 补齐到 64B */
    {
        uint8_t chunk[1 << 20];
        for (int i = 0; i < n; i++) {
            const VQ_ENT *e = &d[i];
            if (e->qtype == 0x05) {
                const float *src = (const float *)e->ptr;
                size_t nel = (size_t)e->rows * e->cols;
                size_t cap = sizeof(chunk) / 2;
                size_t done = 0;
                while (done < nel) {
                    size_t k = nel - done; if (k > cap) k = cap;
                    uint16_t *d16 = (uint16_t *)chunk;
                    for (size_t j = 0; j < k; j++) d16[j] = f2h(src[done + j]);
                    size_t wb = k * 2;
                    if (fwrite(chunk, 1, wb, f) != wb) { fclose(f); return -1; }
                    done += k;
                }
            } else {
                const uint8_t *p = (const uint8_t *)e->ptr;
                size_t left = (size_t)e->bytes;
                while (left) {
                    size_t k = left > sizeof(chunk) ? sizeof(chunk) : left;
                    if (fwrite(p, 1, k, f) != k) { fclose(f); return -1; }
                    p += k; left -= k;
                }
            }
            size_t rem = (size_t)((e->off + e->bytes + 63) & ~(uint64_t)63) - e->off - e->bytes;
            while (rem) {
                uint8_t zp[64] = {0};
                size_t wb = rem > 64 ? 64 : rem;
                if (fwrite(zp, 1, wb, f) != wb) { fclose(f); return -1; }
                rem -= wb;
            }
        }
    }
    h.file_len = data_end;
    if (fseek(f, 0, SEEK_SET) == 0) {
        uint8_t hdrb[sizeof(VQFHeader)];
        memcpy(hdrb, &h, sizeof(h));
        if (fwrite(hdrb, 1, sizeof(h), f) != sizeof(h)) { fclose(f); return -1; }
    }
    uint64_t wsum = 0;
    /* 尾部 FNV-1a（8B）：引擎规则 = FNV(dir) 续 FNV(data[data_off..file_len))，
     * 数据写完后从磁盘流式重算（绑定实际字节，与引擎 vqf_load 自检同规）。 */
    {
        uint64_t sum = 0xcbf29ce484222325ull;
        uint8_t tmp[256];
        size_t dn = (size_t)n * sizeof(VQFTensor);
        if (fseek(f, (long)dir_off, SEEK_SET) == 0) {
            while (dn) {
                size_t k = dn > sizeof(tmp) ? sizeof(tmp) : dn;
                if (fread(tmp, 1, k, f) != k) break;
                for (size_t j = 0; j < k; j++) { sum ^= tmp[j]; sum *= 0x100000001b3ull; }
                dn -= k;
            }
        }
        int64_t dlen = (int64_t)data_end - (int64_t)data_off;
        if (fseek(f, (long)data_off, SEEK_SET) == 0) {
            while (dlen > 0) {
                size_t k = dlen > (int64_t)sizeof(tmp) ? sizeof(tmp) : (size_t)dlen;
                if (fread(tmp, 1, k, f) != k) break;
                for (size_t j = 0; j < k; j++) { sum ^= tmp[j]; sum *= 0x100000001b3ull; }
                dlen -= (int64_t)k;
            }
        }
        if (fseek(f, 0, SEEK_END) == 0) {
            wsum = sum;
            if (fwrite(&sum, 1, 8, f) != 8) { fclose(f); return -1; }
        }
    }
    fclose(f);
    fclose(f);
    fprintf(stderr, "[conv] wrote %s: %d tensors, %.1f MB flags=0x%x sum=%016llx\n",
            path, n, (double)data_end / 1048576.0, flags,
            (unsigned long long)wsum);
    return 0;
}

/* ---- 主转换（文本，同引擎 load_quant_weights 序列） ---- */
static int run_convert(const char *model_dir, const char *out) {
    STModelConfig cfg; memset(&cfg, 0, sizeof(cfg));
    if (st_parse_config(model_dir, &cfg) != 0) {
        fprintf(stderr, "[conv] config parse failed\n");
        return 1;
    }
    if (cfg.max_seq_len > 8192) cfg.max_seq_len = 8192;
    STModelWeights w; memset(&w, 0, sizeof(w));
    if (w_alloc(&w, &cfg) != 0) {
        fprintf(stderr, "[conv] alloc failed\n");
        st_config_free(&cfg);
        return 1;
    }
    /* ---- repack 门控镜像（引擎 st_weights_alloc 期判定；须在 loader/lm
     *       repack 之前完成）---- */
    {
        int dim = cfg.dim, ff = cfg.ffn_dim, vc = cfg.vocab_size;
        int q_out = cfg.n_heads * cfg.head_dim;
        int k_out = cfg.n_kv_heads * cfg.head_dim;
        int wm = g_wmode;
        /* M4 (Q4 4x4)：任一 Q4 矩阵几何不满足 vc/q_out/k_out%4 且 dim/ff%32
         * 则全局禁用（引擎 2669-2672）。 */
        if (!(vc % 4 == 0 && q_out % 4 == 0 && k_out % 4 == 0 &&
              dim % 32 == 0 && ff % 32 == 0))
            g_st_q4_repack = 0;
        /* M4e (Q8 tiled)：wm3(q4i) 预解包 int8，repack 永不适用（引擎 2700）；
         * 否则 VLLM_DISABLE_Q8_REPACK 显式关闭；几何按 8x8(默认) 或 4x4
         * (VLLM_Q8_8X8=0) 门控（引擎 2702-2713）。 */
        if (wm == 3) {
            g_st_q8_repack = 0;
        } else if (g_st_q8_repack) {
            const char *dq = getenv("VLLM_DISABLE_Q8_REPACK");
            if (dq && dq[0] && dq[0] != '0') g_st_q8_repack = 0;
            if (g_st_q8_repack) {
                const char *q8e = getenv("VLLM_Q8_8X8");
                int eight = !(q8e && q8e[0] == '0');
                if (eight) {
                    if (!(q_out % 8 == 0 && k_out % 8 == 0 && ff % 8 == 0 &&
                          dim % 8 == 0 && dim % 32 == 0 && ff % 32 == 0))
                        g_st_q8_repack = 0;
                } else {
                    if (!(q_out % 4 == 0 && k_out % 4 == 0 && ff % 4 == 0 &&
                          dim % 4 == 0 && dim % 32 == 0 && ff % 32 == 0))
                        g_st_q8_repack = 0;
                }
            }
        }
        fprintf(stderr, "[conv] repack gates: q8=%d q4=%d (wmode=%d)\n",
                g_st_q8_repack, g_st_q4_repack, wm);
    }
    int dim = cfg.dim;
    /* embed + final_norm（Qwen3-VL 有 language_model 包裹层；qwen3_moe 无） */
    if (st_load_tensor(&cfg, "model.language_model.embed_tokens.weight",
                       w.token_embed, cfg.vocab_size * dim) != 0) {
        if (st_load_tensor(&cfg, "model.embed_tokens.weight",
                           w.token_embed, cfg.vocab_size * dim) != 0) {
            fprintf(stderr, "[conv] embed_tokens not found\n");
            goto fail;
        }
    }
    if (st_load_tensor(&cfg, "model.language_model.norm.weight",
                       w.final_norm, dim) != 0) {
        if (st_load_tensor(&cfg, "model.norm.weight",
                           w.final_norm, dim) != 0) {
            fprintf(stderr, "[conv] final_norm missing\n");
        }
    }
    /* lm_head（chunked；tie 时 slice 失败回退到 embed） */
    {
        int64_t lm_elems = (int64_t)cfg.vocab_size * dim;
        int chunk_rows = 16384;
        int64_t chunk_elems = (int64_t)chunk_rows * dim;
        float *tmp = (float *)malloc((size_t)chunk_elems * 4);
        if (!tmp) { fprintf(stderr, "[conv] lm chunk OOM\n"); goto fail; }
        int wm = g_wmode;
        for (int64_t off = 0; off < lm_elems; off += chunk_elems) {
            int64_t n = lm_elems - off; if (n > chunk_elems) n = chunk_elems;
            if (st_load_tensor_slice(&cfg, "lm_head.weight", off, tmp, n) != 0) {
                memcpy(tmp, w.token_embed + off, (size_t)n * 4); /* tied */
            }
            if (w.q8_lm_weight) {
                uint8_t *dst = w.q8_lm_weight + (size_t)(off / 32) * 34;
                if (wm == 3) f32_to_q4i8(dst, tmp, (int)n);
                else if (wm == 4) f32_to_g256q8(dst, tmp, (int)n);
                else f32_to_q8_0(dst, tmp, (int)n);
                if (g_st_q8_repack)
                    repack_q8_0_tiled_inplace(dst, (int)(n / dim), dim);
            }
            if (w.q4_lm_weight) {
                uint8_t *dst = w.q4_lm_weight + (size_t)(off / 32) * 18;
                f32_to_q4_0(dst, tmp, (int)n);
                if (g_st_q4_repack)
                    repack_q4_0_4x4_inplace(
                        w.q4_lm_weight + (size_t)(off / dim) * (((dim + 31) / 32) * 18),
                        (int)(n / dim), dim);
            }
        }
        free(tmp);
    }
    /* 层（loader 内部逐层量化 + repack + x8 复制） */
    fprintf(stderr, "[conv] loading %d layers (wmode=%d)...\n", cfg.n_layers, g_wmode);
    if (st_load_layer_weights(&cfg, &w, 0, cfg.n_layers) != 0) {
        fprintf(stderr, "[conv] layer load failed\n");
        goto fail;
    }
    /* Vision 权重（多模态；镜像引擎 run_convert_vqf：加载失败仅告警 →
     * 纯文本 VQF，成功则 w.vision 供 collect + VISION flag） */
    if (cfg.has_vision) {
        STVisionWeights *vvis = (STVisionWeights *)calloc(1, sizeof(STVisionWeights));
        if (vvis && st_vision_weights_alloc(vvis, &cfg) == 0 &&
            st_vision_load_weights(vvis, &cfg) == 0) {
            st_vision_weights_free_f32(vvis);
            w.vision = vvis;
            fprintf(stderr, "[VIS] vision weights loaded (ViT %d layers, hidden=%d)\n",
                    cfg.vis_depth, cfg.vis_hidden);
        } else {
            fprintf(stderr, "[conv] WARNING: vision weight load failed -> text-only VQF\n");
            if (vvis) { st_vision_weights_free(vvis); free(vvis); }
        }
    }
    /* flags（与引擎转换同口径） */
    uint32_t flags = 0;
    flags |= VQF_FLAG_EMB_F16;             /* token_embed 浠?F16 瀛?*/
    if (w.q8_buf_q4) flags |= VQF_FLAG_Q8BUF_Q4;
    if (w.has_x8)    flags |= VQF_FLAG_X8;
    if (g_wmode == 4) flags |= VQF_FLAG_G256;
    if (w.vision && w.vision->is_allocated) flags |= VQF_FLAG_VISION;
    if (cfg.is_moe)  flags |= VQF_FLAG_MOE;
    if (g_st_q8_repack) {
        const char *e = getenv("VLLM_Q8_8X8");
        if (!(e && e[0] == '0')) flags |= VQF_FLAG_Q8_8X8;
    }
    /* 引擎 vqf_write 以 g_st_q4_repack 全局为准（与是否有 q4 矩阵无关） */
    if (g_st_q4_repack) flags |= VQF_FLAG_Q4_4X4;
    int rc = write_vqf(out, &w, &cfg, flags);
    w_free(&w);
    st_config_free(&cfg);
    return rc == 0 ? 0 : 1;
fail:
    w_free(&w);
    st_config_free(&cfg);
    return 1;
}

/* ================================================================
 * --stream：低峰值内存转换（镜像引擎 --convert-vqf-stream 语义）
 *
 * 与全量路径产出逐字节一致（同 loader/量化/repack/头部/尾部 FNV），
 * 但量化缓冲不驻留 RAM —— 输出文件先 ftruncate + mmap(MAP_SHARED)，
 * STModelWeights 的层投影缓冲直接指向文件映射段，loader 逐层写入；
 * 每层 msync 落盘 + madvise(MADV_DONTNEED) 交回驻留页。embed/lm_head
 * 走分块 slice 直写；vision 全量加载后 memcpy 到文件段。
 * 峰值 ≈ 单层 f32 临时 + embed 分块 + vision(~1GB) ，与层数无关。
 * q2mix 拒绝（与引擎 stream 一致）。Windows 无 MADV_DONTNEED（no-op，
 * 语义仍正确；Linux 才有驻留回收）。
 * ================================================================ */
static void s_plan(const STModelConfig *cfg, int wm, int vis_ok,
                   VQ_ENT *e, int cap, int *n_out) {
    int n = 0;
    int dim = cfg->dim, ff = cfg->ffn_dim, vc = cfg->vocab_size, nl = cfg->n_layers;
    int q_out = cfg->n_heads * cfg->head_dim;
    int k_out = cfg->n_kv_heads * cfg->head_dim;
#define P(nm, qt, r, c) do { if (n < cap) {                      \
        e[n].name = (nm); e[n].qtype = (qt); e[n].ptr = NULL;    \
        e[n].rows = (r); e[n].cols = (c);                        \
        e[n].bytes = vq_bytes((qt), (uint64_t)(r) * (c)); n++; } } while (0)
    P("token_embed", 0x05, vc, dim);
    P("final_norm",  0x01, dim, 1);
    P("attn_norm",   0x01, nl * dim, 1);
    P("ffn_norm",    0x01, nl * dim, 1);
    if (cfg->has_q_norm) {
        P("q_norm", 0x01, nl * cfg->head_dim, 1);
        P("k_norm", 0x01, nl * cfg->head_dim, 1);
    }
    if (cfg->is_moe)   /* router f32 resident [nl*n_experts, dim] (mirror collect_text) */
        P("moe_router", 0x01, nl * cfg->n_experts, dim);
    if (wm != 1) {
        P("q8_q", 0x02, nl * dim, q_out);
        P("q8_k", 0x02, nl * dim, k_out);
        P("q8_v", 0x02, nl * dim, k_out);
        P("q8_o", 0x02, nl * q_out, dim);
        P("q8_gate", 0x02, nl * ff, dim);
        P("q8_up",   0x02, nl * ff, dim);
        P("q8_down", 0x02, nl * dim, ff);
        P("q8_lm",   0x02, vc, dim);
    }
    if (wm == 0 || wm == 1) {
        P("q4_q", 0x03, nl * dim, q_out);
        P("q4_k", 0x03, nl * dim, k_out);
        P("q4_v", 0x03, nl * dim, k_out);
        P("q4_o", 0x03, nl * q_out, dim);
        P("q4_gate", 0x03, nl * ff, dim);
        P("q4_up",   0x03, nl * ff, dim);
        P("q4_down", 0x03, nl * dim, ff);
        P("q4_lm",   0x03, vc, dim);
    }
    if (wm == 1) {
        const char *e8 = getenv("VLLM_ENABLE_8X8L");
        if (e8 && e8[0] && e8[0] != '0') {
            P("x8_q", 0x04, nl * dim, q_out);
            P("x8_k", 0x04, nl * dim, k_out);
            P("x8_v", 0x04, nl * dim, k_out);
            P("x8_o", 0x04, nl * q_out, dim);
            P("x8_gate", 0x04, nl * ff, dim);
            P("x8_up",   0x04, nl * ff, dim);
            P("x8_down", 0x04, nl * dim, ff);
        }
    }
    if (vis_ok) {
        int vh = cfg->vis_hidden, vd = cfg->vis_depth, vf = cfg->vis_ffn;
        int vp = cfg->vis_patch, vt = cfg->vis_temporal;
        int vf_pad = ((vf + 31) / 32) * 32;
        int merge_in = vh * cfg->vis_merge * cfg->vis_merge;
        int ds_mid = merge_in, out_dim = cfg->vis_out_dim, ds = cfg->vis_ds_count;
        P("v_q8_qkv", 0x02, vd, 3 * vh * vh);
        P("v_q8_proj", 0x02, vd, vh * vh);
        P("v_q8_fc1", 0x02, vd, vf * vh);
        P("v_q8_fc2", 0x02, vd, vh * vf_pad);
        P("v_patch_embed", 0x01, 1, vh * 3 * vt * vp * vp);
        P("v_patch_bias",  0x01, 1, vh);
        P("v_pos_embed",   0x01, 1, cfg->vis_max_pos * vh);
        P("v_qkv_bias",    0x01, vd, 3 * vh);
        P("v_proj_bias",   0x01, vd, vh);
        P("v_norm1_w",     0x01, vd, vh);
        P("v_norm1_b",     0x01, vd, vh);
        P("v_norm2_w",     0x01, vd, vh);
        P("v_norm2_b",     0x01, vd, vh);
        P("v_fc1_bias",    0x01, vd, vf);
        P("v_fc2_bias",    0x01, vd, vh);
        P("v_merger_nw",   0x01, 1, vh);
        P("v_merger_nb",   0x01, 1, vh);
        P("v_merger_f1w",  0x01, 1, merge_in * merge_in);
        P("v_merger_f1b",  0x01, 1, merge_in);
        P("v_merger_f2w",  0x01, 1, merge_in * out_dim);
        P("v_merger_f2b",  0x01, 1, out_dim);
        P("v_ds_nw",  0x01, ds, merge_in);
        P("v_ds_nb",  0x01, ds, merge_in);
        P("v_ds_f1w", 0x01, ds, merge_in * ds_mid);
        P("v_ds_f1b", 0x01, ds, ds_mid);
        P("v_ds_f2w", 0x01, ds, ds_mid * out_dim);
        P("v_ds_f2b", 0x01, ds, out_dim);
    }
#undef P
    *n_out = n;
}

/* STModelWeights 层投影/规范 字段名 → 指针（loader 直写文件映射段用） */
static void **s_w_field(STModelWeights *w, const char *nm) {
#define M(K, F) if (!strcmp(nm, (K))) return (void **)&(w->F)
    M("attn_norm", attn_norm);  M("ffn_norm", ffn_norm);
    M("q_norm", q_norm);        M("k_norm", k_norm);
    M("moe_router", moe_router);
    M("q8_q", q8_q_weight);     M("q8_k", q8_k_weight);
    M("q8_v", q8_v_weight);     M("q8_o", q8_o_weight);
    M("q8_gate", q8_gate_weight); M("q8_up", q8_up_weight);
    M("q8_down", q8_down_weight);
    M("q4_q", q4_q_weight);     M("q4_k", q4_k_weight);
    M("q4_v", q4_v_weight);     M("q4_o", q4_o_weight);
    M("q4_gate", q4_gate_weight); M("q4_up", q4_up_weight);
    M("q4_down", q4_down_weight);
    M("x8_q", x8_q_weight);     M("x8_k", x8_k_weight);
    M("x8_v", x8_v_weight);     M("x8_o", x8_o_weight);
    M("x8_gate", x8_gate_weight); M("x8_up", x8_up_weight);
    M("x8_down", x8_down_weight);
#undef M
    return NULL;
}

/* vision 字段名 → 源指针（memcpy 到文件段用） */
static void *s_v_field(STVisionWeights *v, const char *nm) {
    if (!v) return NULL;
#define M(K, F) if (!strcmp(nm, (K))) return (void *)(v->F)
    M("v_q8_qkv", q8_attn_qkv_weight); M("v_q8_proj", q8_attn_proj_weight);
    M("v_q8_fc1", q8_mlp_fc1_weight);  M("v_q8_fc2", q8_mlp_fc2_weight);
    M("v_patch_embed", patch_embed_weight); M("v_patch_bias", patch_embed_bias);
    M("v_pos_embed", pos_embed);
    M("v_qkv_bias", attn_qkv_bias);   M("v_proj_bias", attn_proj_bias);
    M("v_norm1_w", norm1_weight);     M("v_norm1_b", norm1_bias);
    M("v_norm2_w", norm2_weight);     M("v_norm2_b", norm2_bias);
    M("v_fc1_bias", mlp_fc1_bias);    M("v_fc2_bias", mlp_fc2_bias);
    M("v_merger_nw", merger_norm_weight); M("v_merger_nb", merger_norm_bias);
    M("v_merger_f1w", merger_fc1_weight); M("v_merger_f1b", merger_fc1_bias);
    M("v_merger_f2w", merger_fc2_weight); M("v_merger_f2b", merger_fc2_bias);
    M("v_ds_nw", ds_norm_weight); M("v_ds_nb", ds_norm_bias);
    M("v_ds_f1w", ds_fc1_weight); M("v_ds_f1b", ds_fc1_bias);
    M("v_ds_f2w", ds_fc2_weight); M("v_ds_f2b", ds_fc2_bias);
#undef M
    return NULL;
}

static int run_convert_stream(const char *model_dir, const char *out) {
    STModelConfig cfg; memset(&cfg, 0, sizeof(cfg));
    if (st_parse_config(model_dir, &cfg) != 0) {
        fprintf(stderr, "[conv-stream] config parse failed\n");
        return 1;
    }
    if (cfg.max_seq_len > 8192) cfg.max_seq_len = 8192;
    int wm = g_wmode;
    if (wm == 5) {
        fprintf(stderr, "[conv-stream] q2mix unsupported in stream mode\n");
        st_config_free(&cfg);
        return 1;
    }
    /* vision 先全量加载以决定是否含 vision 段（失败降级纯文本，镜像引擎） */
    STVisionWeights *vvis = NULL;
    int vis_ok = 0;
    if (cfg.has_vision) {
        vvis = (STVisionWeights *)calloc(1, sizeof(STVisionWeights));
        if (vvis && st_vision_weights_alloc(vvis, &cfg) == 0 &&
            st_vision_load_weights(vvis, &cfg) == 0) {
            st_vision_weights_free_f32(vvis);
            vis_ok = 1;
            fprintf(stderr, "[conv-stream] vision weights resident (ViT %d l)\n",
                    cfg.vis_depth);
        } else {
            fprintf(stderr, "[conv-stream] WARNING: vision load failed -> text-only\n");
            if (vvis) { st_vision_weights_free(vvis); free(vvis); vvis = NULL; }
        }
    }

    VQ_ENT e[96];
    int n = 0;
    s_plan(&cfg, wm, vis_ok, e, 96, &n);
    if (n <= 0) { fprintf(stderr, "[conv-stream] no tensors planned\n"); return 1; }

    size_t dir_off = (sizeof(VQFHeader) + 63) & ~(size_t)63;
    size_t data_off = (dir_off + (size_t)n * sizeof(VQFTensor) + VQF_HEADER_BYTES - 1)
                      & ~(size_t)(VQF_HEADER_BYTES - 1);
    uint64_t off = data_off;
    for (int i = 0; i < n; i++) { e[i].off = off; off = (off + e[i].bytes + 63) & ~(uint64_t)63; }
    uint64_t data_end = off;
    uint64_t fsize = data_end + 8;   /* +8：尾部 FNV 一并映射 */

    st_wmap_t m;
    if (st_wmap_create(&m, out, fsize) != 0) {
        fprintf(stderr, "[conv-stream] cannot create+map %s\n", out);
        st_config_free(&cfg);
        return 1;
    }
    uint8_t *base = (uint8_t *)m.data;

    /* 头部 + 目录（arch 全字段镜像 vqf_write） */
    VQFHeader h; memset(&h, 0, sizeof(h));
    h.magic = VQF_MAGIC;
    h.version = cfg.is_moe ? VQF_VERSION_MOE : VQF_VERSION;
    h.n_tensors = (uint32_t)n;
    h.arch.dim = (uint32_t)cfg.dim; h.arch.n_layers = (uint32_t)cfg.n_layers;
    h.arch.n_heads = (uint32_t)cfg.n_heads; h.arch.n_kv_heads = (uint32_t)cfg.n_kv_heads;
    h.arch.head_dim = (uint32_t)cfg.head_dim; h.arch.ffn_dim = (uint32_t)cfg.ffn_dim;
    h.arch.vocab_size = (uint32_t)cfg.vocab_size; h.arch.max_seq_len = (uint32_t)cfg.max_seq_len;
    h.arch.rope_theta = cfg.rope_theta; h.arch.norm_eps = cfg.norm_eps;
    h.arch.bos_id = (uint32_t)cfg.bos_id; h.arch.eos_id = (uint32_t)cfg.eos_id;
    h.arch.has_q_norm = (uint32_t)cfg.has_q_norm; h.arch.has_mrope = (uint32_t)cfg.has_mrope;
    h.arch.head_dim_full = (uint32_t)cfg.head_dim_full; h.arch.kv_lora_rank = (uint32_t)cfg.kv_lora_rank;
    h.arch.has_vision = (uint32_t)cfg.has_vision; h.arch.vis_depth = (uint32_t)cfg.vis_depth;
    h.arch.vis_hidden = (uint32_t)cfg.vis_hidden; h.arch.vis_heads = (uint32_t)cfg.vis_heads;
    h.arch.vis_ffn = (uint32_t)cfg.vis_ffn; h.arch.vis_patch = (uint32_t)cfg.vis_patch;
    h.arch.vis_temporal = (uint32_t)cfg.vis_temporal; h.arch.vis_merge = (uint32_t)cfg.vis_merge;
    h.arch.vis_out_dim = (uint32_t)cfg.vis_out_dim; h.arch.vis_in_chan = (uint32_t)cfg.vis_in_chan;
    h.arch.vis_max_pos = (uint32_t)cfg.vis_max_pos; h.arch.vis_ds_count = (uint32_t)cfg.vis_ds_count;
    h.arch.mrope_n_sec = (uint32_t)cfg.mrope_n_sec;
    for (int i = 0; i < 4; i++) {
        h.arch.vis_ds_idx[i] = (int32_t)cfg.vis_ds_idx[i];
        h.arch.mrope_sections[i] = (int32_t)cfg.mrope_sections[i];
    }
    h.arch.vision_start_id = (uint32_t)cfg.vision_start_id;
    h.arch.vision_end_id = (uint32_t)cfg.vision_end_id;
    h.arch.image_token_id = (uint32_t)cfg.image_token_id;
    h.arch.video_token_id = (uint32_t)cfg.video_token_id;
    /* MoE arch tail (v3; dense 全 0 保持 v2 canonical 不变) */
    h.arch.n_experts = (uint32_t)cfg.n_experts;
    h.arch.moe_ffn = (uint32_t)cfg.moe_ffn;
    h.arch.top_k = (uint32_t)cfg.top_k;
    h.arch.shared_experts = (uint32_t)cfg.shared_experts;
    h.data_offset = data_off;
    memcpy(base, &h, sizeof(h));
    for (int i = 0; i < n; i++) {
        VQFTensor *t = (VQFTensor *)(base + dir_off + (size_t)i * sizeof(VQFTensor));
        memset(t, 0, sizeof(*t));
        snprintf(t->name, sizeof(t->name), "%s", e[i].name);
        t->qtype = e[i].qtype; t->rows = e[i].rows; t->cols = e[i].cols;
        t->offset = e[i].off; t->bytes = e[i].bytes;
    }

    /* repack 门控（同全量路径；须在 lm/loader repack 前） */
    {
        int dim = cfg.dim, ff = cfg.ffn_dim, vc = cfg.vocab_size;
        int q_out = cfg.n_heads * cfg.head_dim;
        int k_out = cfg.n_kv_heads * cfg.head_dim;
        if (!(vc % 4 == 0 && q_out % 4 == 0 && k_out % 4 == 0 &&
              dim % 32 == 0 && ff % 32 == 0))
            g_st_q4_repack = 0;
        if (wm == 3) {
            g_st_q8_repack = 0;
        } else if (g_st_q8_repack) {
            const char *dq = getenv("VLLM_DISABLE_Q8_REPACK");
            if (dq && dq[0] && dq[0] != '0') g_st_q8_repack = 0;
            if (g_st_q8_repack) {
                const char *q8e = getenv("VLLM_Q8_8X8");
                int eight = !(q8e && q8e[0] == '0');
                if (eight) {
                    if (!(q_out % 8 == 0 && k_out % 8 == 0 && ff % 8 == 0 &&
                          dim % 8 == 0 && dim % 32 == 0 && ff % 32 == 0))
                        g_st_q8_repack = 0;
                } else {
                    if (!(q_out % 4 == 0 && k_out % 4 == 0 && ff % 4 == 0 &&
                          dim % 4 == 0 && dim % 32 == 0 && ff % 32 == 0))
                        g_st_q8_repack = 0;
                }
            }
        }
    }

    STModelWeights w; memset(&w, 0, sizeof(w));
    w.n_layers_allocated = cfg.n_layers;
    if (wm == 3) w.q8_buf_q4 = 1;
    /* 层投影/norm 缓冲 → 文件映射段（loader 直写；embed/final_norm/lm/v 排除） */
    for (int i = 0; i < n; i++) {
        const char *nm = e[i].name;
        if (nm[0] == 'v') continue;
        if (!strcmp(nm, "token_embed") || !strcmp(nm, "final_norm") ||
            !strcmp(nm, "q8_lm") || !strcmp(nm, "q4_lm")) continue;
        void **fp = s_w_field(&w, nm);
        if (!fp) { fprintf(stderr, "[conv-stream] no field for %s\n", nm); goto fail; }
        *fp = base + e[i].off;
    }
    w.has_x8 = 0;
    for (int i = 0; i < n; i++)
        if (e[i].name[0] == 'x' && e[i].name[1] == '8' && e[i].name[2] == '_') w.has_x8 = 1;

    /* ---- embed（分块 slice → F16/F32 直写映射段） ---- */
    {
        const char *emb_a = "model.language_model.embed_tokens.weight";
        const char *emb_b = "model.embed_tokens.weight";   /* qwen3_moe 无包裹层 */
        float probe8[8];
        const char *emb_src = emb_a;
        if (st_load_tensor_slice(&cfg, emb_a, 0, probe8, 8) != 0) emb_src = emb_b;
        int64_t vc_dim = (int64_t)cfg.vocab_size * cfg.dim;
        uint8_t *emb_map = base + e[0].off;   /* e[0] = token_embed */
        float chunk[1 << 18];
        int64_t nch = sizeof(chunk) / sizeof(float);
        for (int64_t i = 0; i < vc_dim; i += nch) {
            int64_t k = vc_dim - i; if (k > nch) k = nch;
            if (st_load_tensor_slice(&cfg, emb_src, i, chunk, k) != 0) {
                fprintf(stderr, "[conv-stream] embed slice load failed\n");
                goto fail;
            }
            if (e[0].qtype == 0x05) {
                uint16_t *d16 = (uint16_t *)emb_map + i;
                for (int64_t j = 0; j < k; j++) d16[j] = f2h(chunk[j]);
            } else {
                memcpy(emb_map + (size_t)i * 4, chunk, (size_t)k * 4);
            }
        }
    }

    /* ---- final_norm ---- */
    for (int i = 0; i < n; i++) {
        if (!strcmp(e[i].name, "final_norm")) {
            float *fn = (float *)(base + e[i].off);
            if (st_load_tensor(&cfg, "model.language_model.norm.weight", fn,
                               cfg.dim) != 0) {
                if (st_load_tensor(&cfg, "model.norm.weight", fn, cfg.dim) != 0)
                    fprintf(stderr, "[conv-stream] final_norm missing\n");
            }
            break;
        }
    }

    /* ---- lm_head（源 = lm_head.weight；tied 缺省时用 embed_tokens slice） ---- */
    {
        const char *lm_src = "lm_head.weight";
        float probe[8];
        int tie = (st_load_tensor_slice(&cfg, lm_src, 0, probe, 8) != 0);
        if (tie) lm_src = "model.language_model.embed_tokens.weight";
        uint8_t *lm_map8 = NULL, *lm_map4 = NULL;
        for (int i = 0; i < n; i++) {
            if (!strcmp(e[i].name, "q8_lm")) lm_map8 = base + e[i].off;
            if (!strcmp(e[i].name, "q4_lm")) lm_map4 = base + e[i].off;
        }
        if (lm_map8 || lm_map4) {
            int64_t lm_elems = (int64_t)cfg.vocab_size * cfg.dim;
            int chunk_rows = 16384;
            int64_t chunk_elems = (int64_t)chunk_rows * cfg.dim;
            float *tmp_lm = (float *)malloc((size_t)chunk_elems * sizeof(float));
            if (!tmp_lm) { fprintf(stderr, "[conv-stream] lm chunk OOM\n"); goto fail; }
            for (int64_t off = 0; off < lm_elems; off += chunk_elems) {
                int64_t nlm = lm_elems - off; if (nlm > chunk_elems) nlm = chunk_elems;
                if (st_load_tensor_slice(&cfg, lm_src, off, tmp_lm, nlm) != 0) {
                    fprintf(stderr, "[conv-stream] %s slice failed\n", lm_src);
                    free(tmp_lm);
                    goto fail;
                }
                if (lm_map8) {
                    uint8_t *dst = lm_map8 + (size_t)(off / 32) * 34;
                    if (wm == 3) f32_to_q4i8(dst, tmp_lm, (int)nlm);
                    else if (wm == 4) f32_to_g256q8(dst, tmp_lm, (int)nlm);
                    else f32_to_q8_0(dst, tmp_lm, (int)nlm);
                    if (g_st_q8_repack)
                        repack_q8_0_tiled_inplace(dst, (int)(nlm / cfg.dim), cfg.dim);
                }
                if (lm_map4) {
                    uint8_t *dst = lm_map4 + (size_t)(off / 32) * 18;
                    f32_to_q4_0(dst, tmp_lm, (int)nlm);
                    if (g_st_q4_repack)
                        repack_q4_0_4x4_inplace(
                            lm_map4 + (size_t)(off / cfg.dim) *
                                (((cfg.dim + 31) / 32) * 18),
                            (int)(nlm / cfg.dim), cfg.dim);
                }
            }
            free(tmp_lm);
        }
    }

    /* ---- 层循环：逐层量化直写映射段 + 落盘/交回 ---- */
    fprintf(stderr, "[conv-stream] streaming %d layers...\n", cfg.n_layers);
    for (int l = 0; l < cfg.n_layers; l++) {
        if (st_load_layer_weights(&cfg, &w, l, l + 1) != 0) {
            fprintf(stderr, "[conv-stream] layer %d load failed\n", l);
            goto fail;
        }
        for (int i = 0; i < n; i++) {
            const char *nm = e[i].name;
            if (nm[0] == 'v') continue;
            if (!strcmp(nm, "token_embed") || !strcmp(nm, "final_norm")) continue;
            int has_lm = (strstr(nm, "_lm") != NULL);
            size_t nl = has_lm ? 0 : (size_t)cfg.n_layers;
            if (!nl || e[i].rows < nl || e[i].rows % nl != 0) continue;
            uint64_t rb = e[i].bytes / nl;
            st_wmap_flush(&m, (size_t)(e[i].off + (uint64_t)l * rb), (size_t)rb);
            st_wmap_discard(&m, (size_t)(e[i].off + (uint64_t)l * rb), (size_t)rb);
        }
    }

    /* ---- vision：驻留缓冲 memcpy 到文件段 ---- */
    if (vis_ok && vvis) {
        for (int i = 0; i < n; i++) {
            if (e[i].name[0] != 'v') continue;
            void *src = s_v_field(vvis, e[i].name);
            if (!src) continue;
            memcpy(base + e[i].off, src, (size_t)e[i].bytes);
        }
    }

    /* ---- flags 回填（与全量路径同规则） + 尾部 FNV ---- */
    {
        uint32_t fflags = 0;
        fflags |= VQF_FLAG_EMB_F16;
        if (w.q8_buf_q4) fflags |= VQF_FLAG_Q8BUF_Q4;
        if (w.has_x8)    fflags |= VQF_FLAG_X8;
        if (wm == 4)     fflags |= VQF_FLAG_G256;
        if (vis_ok)      fflags |= VQF_FLAG_VISION;
        if (cfg.is_moe)  fflags |= VQF_FLAG_MOE;
        if (g_st_q8_repack) {
            const char *e8 = getenv("VLLM_Q8_8X8");
            if (!(e8 && e8[0] == '0')) fflags |= VQF_FLAG_Q8_8X8;
        }
        if (g_st_q4_repack) fflags |= VQF_FLAG_Q4_4X4;

        uint64_t sum = 0xcbf29ce484222325ull;
        for (size_t j = 0; j < (size_t)n * sizeof(VQFTensor); j++) {
            sum ^= base[dir_off + j]; sum *= 0x100000001b3ull;
        }
        for (uint64_t j = 0; j < data_end - data_off; j++) {
            sum ^= base[data_off + j]; sum *= 0x100000001b3ull;
        }
        memcpy(base + data_end, &sum, 8);
        h.flags = fflags; h.file_len = data_end;
        memcpy(base, &h, sizeof(h));
        fprintf(stderr, "[conv-stream] wrote %s: %d tensors, %.1f MB, flags=0x%x "
                        "file_sum=%016llx\n",
                out, n, (double)data_end / 1048576.0, fflags,
                (unsigned long long)sum);
    }
    st_wmap_close(&m);
    if (vvis) { st_vision_weights_free(vvis); free(vvis); }
    st_config_free(&cfg);
    return 0;
fail:
    st_wmap_close(&m);
    remove(out);
    if (vvis) { st_vision_weights_free(vvis); free(vvis); }
    st_config_free(&cfg);
    return 1;
}

int main(int argc, char **argv) {
    const char *model_dir = NULL, *out = NULL;
    int stream = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) model_dir = argv[++i];
        else if (!strcmp(argv[i], "--convert-vqf") && i + 1 < argc) out = argv[++i];
        else if (!strcmp(argv[i], "--stream")) stream = 1;
        else if (!strcmp(argv[i], "--wmode") && i + 1 < argc) {
            const char *m = argv[++i];
            if (!strcmp(m, "dual")) g_wmode = 0;
            else if (!strcmp(m, "q4")) g_wmode = 1;
            else if (!strcmp(m, "q8")) g_wmode = 2;
            else if (!strcmp(m, "q4i")) g_wmode = 3;
            else if (!strcmp(m, "g256")) g_wmode = 4;
            else { fprintf(stderr, "[conv] bad --wmode %s\n", m); return 2; }
        } else if (!strcmp(argv[i], "--help")) {
            printf("vqf_convert --model <dir> --convert-vqf <out.vqf> "
                   "[--stream] [--wmode dual|q4|q8|q4i|g256]\n");
            return 0;
        }
    }
    const char *envw = getenv("VLLM_WMODE");
    if (envw && envw[0]) {
        if (!strcmp(envw, "q4")) g_wmode = 1;
        else if (!strcmp(envw, "q8")) g_wmode = 2;
        else if (!strcmp(envw, "q4i")) g_wmode = 3;
        else if (!strcmp(envw, "g256")) g_wmode = 4;
    }
    if (!model_dir || !out) {
        fprintf(stderr, "usage: vqf_convert --model <dir> --convert-vqf <out.vqf> [--stream] [--wmode ...]\n");
        return 2;
    }
    return stream ? run_convert_stream(model_dir, out) : run_convert(model_dir, out);
}
