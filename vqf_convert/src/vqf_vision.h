/* vqf_vision.h - converter 的 ViT 权重结构（从引擎 include/model/vllm_vision.h
 * 派生：仅权重字段，无推理状态）。vqf_st.h 前向声明的 STVisionWeights 由此补全。 */
#ifndef VQF_VISION_H
#define VQF_VISION_H

#include "vqf_st.h"

struct STVisionWeights {
    /* Patch embedding (Conv3d: out=vh, in=3*vt*vp*vp) */
    float *patch_embed_weight;
    float *patch_embed_bias;
    /* Position embedding (vis_max_pos * vh) */
    float *pos_embed;
    /* ViT block F32 (load 后由 free_f32 释放的大矩阵；仅 q8 副本固化) */
    float *attn_qkv_weight, *attn_qkv_bias;
    float *attn_proj_weight, *attn_proj_bias;
    float *norm1_weight, *norm1_bias;
    float *norm2_weight, *norm2_bias;
    float *mlp_fc1_weight, *mlp_fc1_bias;
    float *mlp_fc2_weight, *mlp_fc2_bias;
    /* Q8_0 固化副本（推理/转换只读；fc2 cols pad 到 %32） */
    float *q8_attn_qkv_weight;
    float *q8_attn_proj_weight;
    float *q8_mlp_fc1_weight;
    float *q8_mlp_fc2_weight;
    /* Merger (spatial merge -> LLM dim) */
    float *merger_norm_weight, *merger_norm_bias;
    float *merger_fc1_weight, *merger_fc1_bias;
    float *merger_fc2_weight, *merger_fc2_bias;
    /* DeepStack mergers */
    float *ds_norm_weight, *ds_norm_bias;
    float *ds_fc1_weight, *ds_fc1_bias;
    float *ds_fc2_weight, *ds_fc2_bias;
    int is_allocated;
};
typedef struct STVisionWeights STVisionWeights;

/* 镜像引擎 vllm_vision.c：alloc / load(+q8 量化) / free_f32 / free。
 * load 内部调用 cfgio 的 st_load_tensor 与 quant_kernels 的 f32_to_q8_0。 */
int  st_vision_weights_alloc(STVisionWeights *w, const STModelConfig *cfg);
int  st_vision_load_weights(STVisionWeights *w, const STModelConfig *cfg);
void st_vision_weights_free_f32(STVisionWeights *w);
void st_vision_weights_free(STVisionWeights *w);

#endif /* VQF_VISION_H */
