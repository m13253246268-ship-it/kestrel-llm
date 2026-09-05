/**
 * vllm_vision.h - Vision Encoder for Qwen3-VL Multimodal Model
 *
 * Implements the ViT-based vision encoder for image and video processing.
 * Supports:
 *   - Patch embedding (3D conv for video, 2D for images)
 *   - 27-layer ViT with DeepStack intermediate features
 *   - Spatial merge (2x2 window)
 *   - Merger projections to LLM embedding space
 *   - Image and video input preprocessing
 *
 * Qwen3-VL Vision Architecture:
 *   Input → PatchEmbed → ViT(27 layers) → SpatialMerge → Merger → visual tokens
 *                               ↓ (layers 8,16,24)
 *                          DeepStack Mergers → additional visual tokens
 *
 * Visual token format: <|vision_start|> [visual_tokens] <|vision_end|>
 *   - Image: <|image_pad|> tokens
 *   - Video: <|video_pad|> tokens
 */

#ifndef VLLM_VISION_H
#define VLLM_VISION_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#include "vllm_safetensors.h"

/* ================================================================
 * Vision Weight Buffers (loaded from safetensors)
 * ================================================================ */

/* 具名 struct（vllm_safetensors.h 前向声明 STModelWeights.vision 用） */
typedef struct STVisionWeights {
    /* Patch embedding: Conv3d weight [vis_hidden, 3, temporal, patch, patch] */
    float *patch_embed_weight;  /* [vis_hidden * 3 * vis_temporal * vis_patch * vis_patch] */
    float *patch_embed_bias;    /* [vis_hidden] */

    /* Position embedding */
    float *pos_embed;           /* [vis_max_pos, vis_hidden] */

    /* ViT blocks: per-layer weights (indexed by layer) */
    float *attn_qkv_weight;     /* [vis_depth, vis_hidden * 3 * vis_hidden] */
    float *attn_qkv_bias;       /* [vis_depth, 3 * vis_hidden] */
    float *attn_proj_weight;    /* [vis_depth, vis_hidden * vis_hidden] */
    float *attn_proj_bias;      /* [vis_depth, vis_hidden] */
    float *norm1_weight;        /* [vis_depth, vis_hidden] */
    float *norm1_bias;          /* [vis_depth, vis_hidden] */
    float *norm2_weight;        /* [vis_depth, vis_hidden] */
    float *norm2_bias;          /* [vis_depth, vis_hidden] */
    float *mlp_fc1_weight;      /* [vis_depth, vis_ffn * vis_hidden] gate+up fused */
    float *mlp_fc1_bias;        /* [vis_depth, vis_ffn] */
    float *mlp_fc2_weight;      /* [vis_depth, vis_hidden * (vis_ffn/2)] down */
    float *mlp_fc2_bias;        /* [vis_depth, vis_hidden] */

    /* Q8_0 quantized ViT weights (axiom: blas_precision_efficiency_tradeoff).
     * Weight-only quantization: activations stay F32, matvecs run on the
     * shared-input-fusion Q8 kernel (st_q8_matvec_batched). fc2 cols are
     * padded from vis_ffn to the next multiple of 32 (vis_ffn=4304 → 4320). */
    uint8_t *q8_attn_qkv_weight;  /* [vis_depth, Q8_BYTES(3*vh*vh)] */
    uint8_t *q8_attn_proj_weight; /* [vis_depth, Q8_BYTES(vh*vh)] */
    uint8_t *q8_mlp_fc1_weight;   /* [vis_depth, Q8_BYTES(vf*vh)] */
    uint8_t *q8_mlp_fc2_weight;   /* [vis_depth, Q8_BYTES(vh*vf_pad)] */

    /* Main merger: projects merged ViT output to LLM hidden_dim */
    float *merger_norm_weight;  /* [vis_hidden * 4]  (after 2x2 spatial merge) */
    float *merger_norm_bias;    /* [vis_hidden * 4] */
    float *merger_fc1_weight;   /* [vis_hidden * 4, merger_intermediate] */
    float *merger_fc1_bias;
    float *merger_fc2_weight;   /* [merger_intermediate, vis_out_dim] */
    float *merger_fc2_bias;     /* [vis_out_dim] */

    /* DeepStack mergers: for each deepstack layer [0..vis_ds_count-1] */
    float *ds_norm_weight;      /* [vis_ds_count, vis_hidden] */
    float *ds_norm_bias;        /* [vis_ds_count, vis_hidden] */
    float *ds_fc1_weight;       /* [vis_ds_count, vis_hidden, ds_intermediate] */
    float *ds_fc1_bias;
    float *ds_fc2_weight;       /* [vis_ds_count, ds_intermediate, vis_out_dim] */
    float *ds_fc2_bias;         /* [vis_ds_count, vis_out_dim] */

    int is_allocated;
} STVisionWeights;

/* ================================================================
 * Runtime Buffers for Vision Encoding
 * ================================================================ */

typedef struct {
    STVisionWeights  w;
    const STModelConfig *cfg;   /* pointer to shared model config */

    /* Input preprocessing */
    float *pixel_values;        /* preprocessed pixel buffer */
    int    img_height;          /* preprocessed image height */
    int    img_width;           /* preprocessed image width */
    int    n_frames;            /* 1 for image, N for video */

    /* Patch embedding output */
    float *patch_features;      /* [n_grid_h, n_grid_w, vis_hidden] */

    /* ViT forward buffers */
    float *hidden_states;       /* current hidden states */
    float *attn_buf;            /* attention output buffer */
    float *mlp_buf;             /* MLP output buffer */
    float *qkv_buf;             /* QKV projection buffer */
    float *fc1_buf;             /* fc1 output [max_patches, vis_ffn_pad] (pad to 32) */

    /* Vision RoPE (Qwen3-VL applies RoPE inside the ViT attention).
     * [vis_max_pos, head_dim] cos/sin tables per patch (grid position). */
    float *rot_cos;
    float *rot_sin;

    /* DeepStack intermediate features */
    /* DeepStack features: post-merger [n_vis_tokens, vis_out_dim] per
     * deepstack layer, ready for injection into the LLM's first layers
     * (official _deepstack_process: hidden[visual_pos] += ds_features). */
    float **ds_features;        /* [vis_ds_count][max_grid_tokens, vis_out_dim] */

    /* Merged output */
    float *visual_tokens;       /* final visual tokens [n_vis_tokens, vis_out_dim] */
    int    n_vis_tokens;        /* number of visual tokens */

    /* Temporary state */
    int    grid_h, grid_w;      /* patch grid dimensions after patch_embed */
    int    n_frames_eff;        /* effective frames (grid_t after temporal patches) */
    int    is_allocated;
} STVisionState;

/* ================================================================
 * API Functions
 * ================================================================ */

/**
 * Allocate vision encoder weights. Call after config is parsed.
 * weights: output, uninitialized
 * cfg: must have has_vision=1
 * Returns 0 on success, -1 on error.
 */
int st_vision_weights_alloc(STVisionWeights *w, const STModelConfig *cfg);

/**
 * Free vision encoder weights.
 */
void st_vision_weights_free(STVisionWeights *w);

/**
 * Free the F32 duplicates of the Q8-quantized ViT weight matrices
 * (attn_qkv/attn_proj/mlp_fc1/mlp_fc2) after Q8 quantization. Inference
 * reads the Q8 copies only, so on memory-constrained boards (RK3588, 16 GB)
 * the ~1.65 GB of F32 duplicates can be released right after load.
 */
void st_vision_weights_free_f32(STVisionWeights *w);

/**
 * Load vision encoder weights from safetensors files.
 * vis: output, must have weights pre-allocated
 * cfg: model config with tensor metadata
 * Returns 0 on success, -1 on error.
 */
int st_vision_load_weights(STVisionWeights *w, const STModelConfig *cfg);

/**
 * Initialize vision encoder runtime state.
 * vis: output, uninitialized
 * cfg: model config
 * Returns 0 on success.
 */
int st_vision_init(STVisionState *vis, const STModelConfig *cfg);

/**
 * Free vision encoder runtime state (keeps weights).
 */
void st_vision_free(STVisionState *vis);

/**
 * Encode an image into visual tokens.
 * vis: initialized vision state with loaded weights
 * image_data: RGB image data [height * width * 3], uint8 values [0,255]
 * width, height: image dimensions in pixels
 * Returns number of visual tokens generated, or -1 on error.
 */
int st_vision_encode_image(STVisionState *vis,
                            const uint8_t *image_data,
                            int width, int height);

/**
 * Encode video frames into visual tokens.
 * vis: initialized vision state with loaded weights
 * frame_data: array of frame pointers, each [height * width * 3], uint8
 * width, height: frame dimensions in pixels
 * n_frames: number of video frames
 * Returns number of visual tokens generated, or -1 on error.
 */
int st_vision_encode_video(STVisionState *vis,
                            const uint8_t **frame_data,
                            int width, int height, int n_frames);

/**
 * Get the visual tokens for integration into LLM input.
 * Returns pointer to [n_vis_tokens, vis_out_dim] float array.
 * Only valid after a successful encode call.
 */
const float* st_vision_get_tokens(const STVisionState *vis);

/**
 * Get the number of visual tokens from the last encode call.
 */
int st_vision_get_num_tokens(const STVisionState *vis);

#ifdef __cplusplus
}
#endif

#endif /* VLLM_VISION_H */
