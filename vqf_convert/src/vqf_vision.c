/* vqf_vision.c - converter 的 ViT 权重加载/量化（逐字节镜像引擎 vllm_vision.c
 * 的 Weight Management 段，src 166-449；同源红线：只允许通过 st_load_tensor /
 * f32_to_q8_0 两处外部依赖，禁止漂移）。 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "conv_platform.h"
#include "vqf_st.h"
#include "vqf_vision.h"

#define VQ8_BYTES(n) ((size_t)((n) + 31) / 32 * 34)
#define VF_PAD(vf)   (((vf) + 31) / 32 * 32)

extern int st_load_tensor(const STModelConfig *cfg, const char *name, float *dst, int n);
extern void f32_to_q8_0(uint8_t *o, const float *i, int n);

int st_vision_weights_alloc(STVisionWeights *w, const STModelConfig *cfg) {
    if (!cfg->has_vision) return -1;
    memset(w, 0, sizeof(*w));

    int vh = cfg->vis_hidden;
    int vd = cfg->vis_depth;
    int vf = cfg->vis_ffn;
    int vp = cfg->vis_patch;
    int vt = cfg->vis_temporal;
    int vo = cfg->vis_out_dim;
    int vm = cfg->vis_merge;
    int ds = cfg->vis_ds_count;

    w->patch_embed_weight = calloc((size_t)vh * 3 * vt * vp * vp, sizeof(float));
    w->patch_embed_bias   = calloc(vh, sizeof(float));
    w->pos_embed = calloc((size_t)cfg->vis_max_pos * vh, sizeof(float));

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
    w->mlp_fc2_weight   = calloc((size_t)vd * vh * vf, sizeof(float));
    w->mlp_fc2_bias     = calloc((size_t)vd * vh, sizeof(float));

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

    int merge_in  = vh * vm * vm;
    int merge_mid = merge_in;
    w->merger_norm_weight = calloc(merge_in, sizeof(float));
    w->merger_norm_bias   = calloc(merge_in, sizeof(float));
    w->merger_fc1_weight  = calloc((size_t)merge_in * merge_mid, sizeof(float));
    w->merger_fc1_bias    = calloc(merge_mid, sizeof(float));
    w->merger_fc2_weight  = calloc((size_t)merge_mid * vo, sizeof(float));
    w->merger_fc2_bias    = calloc(vo, sizeof(float));

    int ds_mid = merge_in;
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
    if (!w) return;
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

void st_vision_weights_free_f32(STVisionWeights *w) {
    if (!w) return;
    free(w->attn_qkv_weight);  w->attn_qkv_weight  = NULL;
    free(w->attn_proj_weight); w->attn_proj_weight = NULL;
    free(w->mlp_fc1_weight);   w->mlp_fc1_weight   = NULL;
    free(w->mlp_fc2_weight);   w->mlp_fc2_weight   = NULL;
}

static void st_vision_quantize_q8(STVisionWeights *w, const STModelConfig *cfg) {
    int vh = cfg->vis_hidden;
    int vd = cfg->vis_depth;
    int vf = cfg->vis_ffn;
    int vf_pad = VF_PAD(vf);

    for (int l = 0; l < vd; l++) {
        f32_to_q8_0((uint8_t *)w->q8_attn_qkv_weight + (size_t)l * VQ8_BYTES((size_t)vh * 3 * vh),
                    w->attn_qkv_weight + (size_t)l * vh * 3 * vh, vh * 3 * vh);
        f32_to_q8_0((uint8_t *)w->q8_attn_proj_weight + (size_t)l * VQ8_BYTES((size_t)vh * vh),
                    w->attn_proj_weight + (size_t)l * vh * vh, vh * vh);
        f32_to_q8_0((uint8_t *)w->q8_mlp_fc1_weight + (size_t)l * VQ8_BYTES((size_t)vf * vh),
                    w->mlp_fc1_weight + (size_t)l * vf * vh, vf * vh);
        for (int r = 0; r < vh; r++) {
            float tmp[4352];
            memcpy(tmp, w->mlp_fc2_weight + (size_t)l * vh * vf + (size_t)r * vf,
                   (size_t)vf * sizeof(float));
            memset(tmp + vf, 0, (size_t)(vf_pad - vf) * sizeof(float));
            f32_to_q8_0((uint8_t *)w->q8_mlp_fc2_weight + (size_t)l * VQ8_BYTES((size_t)vh * vf_pad)
                            + (size_t)r * VQ8_BYTES(vf_pad),
                        tmp, vf_pad);
        }
    }
    fprintf(stderr, "[VIS] ViT weights quantized to Q8_0 (fc2 pad %d → %d)\n",
            vf, vf_pad);
}

int st_vision_load_weights(STVisionWeights *w, const STModelConfig *cfg) {
    int vh = cfg->vis_hidden;
    int vd = cfg->vis_depth;
    int vf = cfg->vis_ffn;
    char name[256];
    int expected;

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

    expected = cfg->vis_max_pos * vh;
    snprintf(name, sizeof(name), "model.visual.pos_embed.weight");
    if (st_load_tensor(cfg, name, w->pos_embed, expected) != 0) {
        snprintf(name, sizeof(name), "model.visual.position_embedding.weight");
        if (st_load_tensor(cfg, name, w->pos_embed, expected) != 0) {
            fprintf(stderr, "[VIS] Warning: no position embedding found\n");
        }
    }

    for (int l = 0; l < vd; l++) {
        snprintf(name, sizeof(name), "model.visual.blocks.%d.norm1.weight", l);
        st_load_tensor(cfg, name, w->norm1_weight + (size_t)l * vh, vh);
        snprintf(name, sizeof(name), "model.visual.blocks.%d.norm1.bias", l);
        st_load_tensor(cfg, name, w->norm1_bias + (size_t)l * vh, vh);
        snprintf(name, sizeof(name), "model.visual.blocks.%d.norm2.weight", l);
        st_load_tensor(cfg, name, w->norm2_weight + (size_t)l * vh, vh);
        snprintf(name, sizeof(name), "model.visual.blocks.%d.norm2.bias", l);
        st_load_tensor(cfg, name, w->norm2_bias + (size_t)l * vh, vh);
        expected = (int)((size_t)vh * 3 * vh);
        snprintf(name, sizeof(name), "model.visual.blocks.%d.attn.qkv.weight", l);
        st_load_tensor(cfg, name, w->attn_qkv_weight + (size_t)l * vh * 3 * vh, expected);
        snprintf(name, sizeof(name), "model.visual.blocks.%d.attn.qkv.bias", l);
        st_load_tensor(cfg, name, w->attn_qkv_bias + (size_t)l * 3 * vh, 3 * vh);
        expected = vh * vh;
        snprintf(name, sizeof(name), "model.visual.blocks.%d.attn.proj.weight", l);
        st_load_tensor(cfg, name, w->attn_proj_weight + (size_t)l * vh * vh, expected);
        snprintf(name, sizeof(name), "model.visual.blocks.%d.attn.proj.bias", l);
        st_load_tensor(cfg, name, w->attn_proj_bias + (size_t)l * vh, vh);
        expected = (int)((size_t)vf * vh);
        snprintf(name, sizeof(name), "model.visual.blocks.%d.mlp.linear_fc1.weight", l);
        st_load_tensor(cfg, name, w->mlp_fc1_weight + (size_t)l * vf * vh, expected);
        snprintf(name, sizeof(name), "model.visual.blocks.%d.mlp.linear_fc1.bias", l);
        st_load_tensor(cfg, name, w->mlp_fc1_bias + (size_t)l * vf, vf);
        expected = (int)((size_t)vh * vf);
        snprintf(name, sizeof(name), "model.visual.blocks.%d.mlp.linear_fc2.weight", l);
        st_load_tensor(cfg, name, w->mlp_fc2_weight + (size_t)l * vh * vf, expected);
        snprintf(name, sizeof(name), "model.visual.blocks.%d.mlp.linear_fc2.bias", l);
        st_load_tensor(cfg, name, w->mlp_fc2_bias + (size_t)l * vh, vh);
    }

    int merge_in = vh * cfg->vis_merge * cfg->vis_merge;
    int merge_mid = merge_in;
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

    int ds_mid = merge_in;
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
