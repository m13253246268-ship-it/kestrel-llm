/* vqf_st.h - converter 自足结构头，从引擎 vllm_safetensors.h 派生。
 * Auto-extracted (lines 30,38-215); re-run tools/extract_quant.py.
 * Engine header sha256: 45fa38bc7f8f64e6de432c58a9e4e50c3b478dda54b23b8ae0ab581dcb5bd980
 * ================================================================ */
#ifndef VQF_ST_H
#define VQF_ST_H
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

struct STVisionWeights;

/* Safetensors dtype codes */
#define ST_DTYPE_F32  0
#define ST_DTYPE_F16  1
#define ST_DTYPE_BF16 2

/**
 * Single tensor entry from the safetensors JSON header.
 */
typedef struct {
    char    *name;         /* tensor name, e.g. "model.language_model.layers.0.self_attn.q_proj.weight" */
    int      dtype;        /* ST_DTYPE_F32 / F16 / BF16 */
    int64_t  n_dims;       /* number of dimensions (up to 8 for Conv3d weights) */
    int64_t  ne[8];        /* dimensions: ne[0]=fastest, ne[7]=slowest */
    int64_t  n_elems;      /* total number of elements (ne[0] * ne[1] * ...) */
    int64_t  file_offsets[2]; /* start and end offset in data section */
    int64_t  n_bytes;       /* raw bytes in file */
} STTensorInfo;

/**
 * Parsed safetensors file header.
 */
typedef struct {
    int         n_files;        /* number of shard files */
    char      **file_paths;     /* [n_files] file paths */
    int         n_tensors;      /* total tensor count across all shards */
    STTensorInfo *tensors;      /* [n_tensors] tensor info */
    /* Model config extracted from config.json */
    int         dim;            /* hidden_size = 4096 */
    int         n_layers;       /* num_hidden_layers = 36 */
    int         n_heads;        /* num_attention_heads = 32 */
    int         n_kv_heads;     /* num_key_value_heads = 8 */
    int         head_dim;       /* head_dim = 128 */
    int         ffn_dim;        /* intermediate_size = 12288 */
    int         vocab_size;     /* 151936 */
    int         max_seq_len;    /* max_position_embeddings = 262144 */
    float       rope_theta;     /* 5000000.0 */
    float       norm_eps;       /* 1e-6 */
    int         bos_id;         /* 151643 */
    int         eos_id;         /* 151645 */
    bool        has_q_norm;     /* Qwen3-VL has per-layer q_norm/k_norm */
    bool        has_mrope;      /* Qwen3-VL uses MRoPE (3D RoPE) */
    int         head_dim_full;  /* 128 for Qwen3-VL */
    int         kv_lora_rank;   /* 0 means no KV lora */

    /* ---- MoE (qwen3_moe) ---- */
    int         is_moe;         /* 1 = qwen3_moe：每层 router + n_experts 专家 */
    int         n_experts;      /* num_experts (30B-A3B: 128) */
    int         moe_ffn;        /* moe_intermediate_size (30B-A3B: 768) 单专家中间宽 */
    int         top_k;          /* num_experts_per_tok (30B-A3B: 8) */
    int         shared_experts; /* num_shared_experts（Qwen3 MoE 无 → 0） */
    /* is_moe=1 时 ffn_dim 被覆写为 n_experts*moe_ffn（每层专家堆叠行数/列数），
     * 使 gate/up/down 的"整层堆叠→量化→repack"与稠密逐字节同构。 */

    /* ---- Vision config (Qwen3-VL) ---- */
    int         has_vision;     /* 1 = vision encoder present */
    int         vis_depth;      /* ViT depth (27) */
    int         vis_hidden;     /* ViT hidden_size (1152) */
    int         vis_heads;      /* ViT num_heads (16) */
    int         vis_ffn;        /* ViT intermediate_size (4304) */
    int         vis_patch;      /* patch_size (16) */
    int         vis_temporal;   /* temporal_patch_size (2) */
    int         vis_merge;      /* spatial_merge_size (2) */
    int         vis_out_dim;    /* out_hidden_size = 4096 */
    int         vis_in_chan;    /* in_channels (3) */
    int         vis_max_pos;    /* num_position_embeddings (2304) */
    int         vis_ds_idx[4];  /* deepstack_visual_indexes [8,16,24] */
    int         vis_ds_count;   /* number of deepstack layers (3) */
    int         mrope_sections[4]; /* MRoPE sections [24,20,20] */
    int         mrope_n_sec;    /* number of MRoPE sections (3) */

    /* Vision special token IDs */
    int         vision_start_id;   /* 151652 */
    int         vision_end_id;     /* 151653 */
    int         image_token_id;    /* 151655 */
    int         video_token_id;    /* 151656 */
} STModelConfig;

/**
 * Dynamically allocated weight buffers for Qwen3-VL model.
 * All weights are stored in float32 after conversion from safetensors.
 *
 * Memory estimate (f32): ~32 GB total. This is impractical for most systems.
 * For memory-constrained environments, consider:
 *   1. Memory-mapped loading (load per-layer on demand)
 *   2. Converting to GGUF Q8_0 first (~8.5 GB)
 */
typedef struct {
    STModelConfig cfg;

    /* --- Permanent tensors (always loaded) --- */
    float *token_embed;       /* [vocab_size, dim]  embedded_tokens.weight */
    float *lm_head;           /* [vocab_size, dim]  lm_head.weight (untied!) */
    float *final_norm;        /* [dim]              model.language_model.norm.weight */
    float *moe_router;        /* [n_layers*n_experts, dim] f32 常驻（qwen3_moe
                               * mlp.gate.weight：全模仅 ~50MB，top-k 边界需高精度） */

    /* --- Per-layer weights (can be loaded/freed dynamically) --- */
    /* All arrays are indexed by layer: weights[layer * dim_size] */
    float *attn_norm;       /* [n_layers, dim]    input_layernorm.weight */
    float *ffn_norm;        /* [n_layers, dim]    post_attention_layernorm.weight */
    float *q_weight;        /* [n_layers, dim * q_out]  q_proj.weight → [dim, n_heads*head_dim] */
    float *k_weight;        /* [n_layers, dim * k_out]  k_proj.weight → [dim, n_kv_heads*head_dim] */
    float *v_weight;        /* [n_layers, dim * k_out]  v_proj.weight */
    float *o_weight;        /* [n_layers, q_out * dim]  o_proj.weight → [n_heads*head_dim, dim] */
    float *q_norm;          /* [n_layers, q_out]        q_norm.weight */
    float *k_norm;          /* [n_layers, k_out]        k_norm.weight */
    float *gate_weight;     /* [n_layers, ffn_dim * dim] gate_proj.weight */
    float *up_weight;       /* [n_layers, ffn_dim * dim] up_proj.weight */
    float *down_weight;     /* [n_layers, dim * ffn_dim] down_proj.weight */

    /* --- Q8_0 compact storage for FFN + LM Head (fixedpoint_quantize) ---
     * Layout: 34 bytes per 32 elements = { f16 scale d; int8 qs[32]; }
     * When has_q8=1, FFN uses AVX2 dequantizing matvec (3-4x mem BW reduction) */
    uint8_t *q8_gate_weight;  /* [n_layers * (ffn_dim * dim + 31)/32 * 34] */
    uint8_t *q8_up_weight;
    uint8_t *q8_down_weight;
    uint8_t *q8_lm_weight;    /* [(vocab * dim + 31)/32 * 34] — LM head Q8_0 */
    uint8_t *q8_q_weight;     /* [n_layers * (q_out * dim + 31)/32 * 34] */
    uint8_t *q8_k_weight;     /* [n_layers * (k_out * dim + 31)/32 * 34] */
    uint8_t *q8_v_weight;
    uint8_t *q8_o_weight;     /* [n_layers * (dim * q_out + 31)/32 * 34] */
    int      has_q8;          /* 1 = Q8_0 compact weights available */

    /* --- Q4_0 compact storage for decode (fixedpoint_quantize_saturate B=4) ---
     * Layout: 18 bytes per 32 elements = { f16 scale d; uint8 nibbles qs[16]; }
     * Decode uses Q4_0 weights × Q8_0 activations (integer-domain dot) to halve
     * weight DRAM bandwidth vs Q8_0 (18B vs 34B per 32-element block). */
    uint8_t *q4_gate_weight;  /* [n_layers * (ffn_dim * dim + 31)/32 * 18] */
    uint8_t *q4_up_weight;
    uint8_t *q4_down_weight;
    uint8_t *q4_lm_weight;    /* [(vocab * dim + 31)/32 * 18] — LM head Q4_0 */
    uint8_t *q4_q_weight;     /* [n_layers * (q_out * dim + 31)/32 * 18] */
    uint8_t *q4_k_weight;     /* [n_layers * (k_out * dim + 31)/32 * 18] */
    uint8_t *q4_v_weight;
    uint8_t *q4_o_weight;     /* [n_layers * (dim * q_out + 31)/32 * 18] */
    int      has_q4;          /* 1 = Q4_0 compact weights available */

    /* --- Q4_0 8x8 GEMM 预打包副本（prefill 专用, 仅 wmode=q4 分配）---
     * 每个矩阵与 legacy 同大小（144B/block = 8 行 legacy 18B/block），布局为
     * llama.cpp block_q4_0x8：{ f16 d[8]; uint8 qs[128] }，qs 按 4B 组 8 行交织、
     * 每字节 ^0x88（nibble v 存为 v^0x88, 即 [-8,+7] 有符号）。点积无需 bias
     * 校正（与 llama 8x8 LUT 内核同约定）。decode/GEMV 仍用 legacy 副本。 */
    uint8_t *x8_gate_weight;
    uint8_t *x8_up_weight;
    uint8_t *x8_down_weight;
    uint8_t *x8_q_weight;
    uint8_t *x8_k_weight;
    uint8_t *x8_v_weight;
    uint8_t *x8_o_weight;
    int      has_x8;          /* 1 = 8x8 预打包副本就绪 */

    /* --- Q2_1 compact storage for mixed-precision decode (wmode=q2mix) ---
     * Layout: 16 bytes per 32 elements =
     *   { f16 d0; f16 m0; f16 d1; f16 m1; uint8 qs[8]; }
     * two 16-elem sub-blocks with own (scale d, min m); 2-bit codes q∈[0,3].
     * dequant w_i = m_h + d_h * q_i → 0.5 B/element (÷2.1 vs Q4_0).
     * Allocated for the FFN projections (gate/up/down) only; attention stays
     * Q4_0 (mixed precision: sensitive projections at 4-bit, FFN at 2-bit,
     * axiom blas_precision_efficiency_tradeoff). */
    uint8_t *q2_gate_weight;  /* [n_layers * (ffn_dim * dim + 31)/32 * 16] */
    uint8_t *q2_up_weight;
    uint8_t *q2_down_weight;  /* [n_layers * (dim * ffn_dim + 31)/32 * 16] */
    int      has_q2;          /* 1 = Q2_1 FFN weights available */

    /* Pre-unpacked Q4_0 → int8 (wmode=q4): the q8_* buffers above hold
     * Q4_0 values expanded to 34 B/block { f16 scale d; int8 qs[32]; } —
     * byte-layout identical to Q8_0, so the tuned Q8 kernels consume them
     * directly (no nibble unpack at runtime). qs[i] = (nibble-8) ∈ [-8,+7]
     * with the same scale bits as the nibble format, so results are bit-identical
     * to the nibble Q4_0 path. q4_* buffers are NOT allocated in this mode. */
    int      q8_buf_q4;        /* 1 = q8_* buffers store pre-unpacked Q4 int8 */

    int      emb_f16;          /* VQF: 1 = token_embed 存 F16，读取时转 F32 */

    int   weights_loaded;   /* bitmask: which weight groups are loaded */
    int   n_layers_allocated; /* actual number of layers with allocated buffers */
    int   is_allocated;

    /* VQF (vLLM Quantized Format) mmap ownership: when non-NULL all weight
     * pointers above alias the mmap'd VQF file (read-only, zero-copy).
     * st_weights_free() munmap()s it instead of free()ing each pointer. */
    void  *vqf_map;
    size_t vqf_map_len;

    /* VQF: 多模态 vision 权重（mmap 直挂，alias vqf_map，见 vllm_vision.h）。
     * vqf_load 在文件含 vision 张量（VQF_FLAG_VISION）时 calloc 填充，
     * st_weights_free() 的 vqf_map 分支一并释放该结构体。 */
    struct STVisionWeights *vision;
} STModelWeights;

#endif /* VQF_ST_H */
