/**
 * vllm_safetensors.h - Safetensors format parser for model weight loading
 *
 * Supports loading HuggingFace safetensors files with bfloat16/float16/float32
 * weight conversion to float32 for the Qwen3-VL-8B-Instruct model.
 *
 * Safetensors format:
 *   [0..7]   Header size (uint64 LE)
 *   [8..N-1] JSON header string
 *   [N..]    Tensor data (aligned)
 */

#ifndef VLLM_SAFETENSORS_H
#define VLLM_SAFETENSORS_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <string.h>

#include "vllm_l3.h"   /* STL3State: Phase-2 L3 disk eviction state */
#include "vllm_platform.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Vision weight buffers 的具名类型（定义见 vllm_vision.h）；供 STModelWeights
 * 前向引用 VQF mmap 挂载的 vision 张量（STModelWeights.vision）。 */
struct STVisionWeights;

/* Debug: front/back guard bytes around each KV block and trailing guard on
 * per-state scratch buffers. Data pointers are offset +KV_GUARD into the raw
 * allocation; every free must use (uint8_t*)ptr - KV_GUARD. Set to 0 to
 * disable (plain allocations, data pointer == raw pointer). */
#define KV_GUARD 4096

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

/**
 * Inference state for Qwen3-VL model.
 */
typedef struct {
    STModelWeights    weights;
    STModelConfig     cfg;
    float  *hidden;         /* [dim] current hidden state */
    float ***k_cache;        /* [n_layers][n_blocks] per-block KV cache */
    float ***v_cache;
    int    *cache_len;
    int     kv_bs;           /* KV block size in positions (= g_sparse_block) */
    int     kv_n_blocks;     /* allocated blocks per layer */
    int8_t ***k_cache_q8;    /* [n_layers][n_blocks] per-block INT8 K cache */
    float  *logits;         /* [vocab_size] */
    /* Temp buffers (maximum size across uses) */
    float  *q_buf;          /* max(n_heads * head_dim, ffn_dim) = 12288 */
    float  *k_buf;          /* max(n_kv_heads * head_dim, ffn_dim) = 12288 */
    float  *v_buf;          /* n_kv_heads * head_dim = 1024 */
    float  *attn_buf;       /* [n_heads * head_dim] = 4096 */
    float  *ffn_buf;        /* [ffn_dim] = 12288 */
    float  *ffn_out_buf;    /* [dim] = 4096 */
    int     seq_len;
    int     mrope_pos;      /* MRoPE text-position cursor. For multimodal
                             * prompts the text MRoPE position advances by
                             * max(grid_h,grid_w) per visual region, NOT by the
                             * visual-token count, so it diverges from seq_len.
                             * prefill_ex stores the final text_pos here and
                             * decode continues from it (0 = text-only: fall
                             * back to seq_len). */
    int     is_allocated;

    float  *scores_buf;        /* [max_kv_slots * nh] per-head rows for parallel attention */
    int     max_kv_slots;      /* KV cache slot count per layer (= max_seq) */

    /* Phase 1.5: prefill-accumulated attention importance (message-passing
     * potential, blas_sparse_message_passing_schedule). prefill_importance[s]
     * = total attention mass position s received during the exact prefill
     * pass (aggregated across heads and query tokens). The sparse decode
     * selection ranks blocks by this importance + a fresh probe, so blocks
     * actually attended during prefill (e.g. a needle fact) are retained even
     * at aggressive k. imp_head is the per-head accumulation scratch. */
    float  *prefill_importance;  /* [max_kv_slots] per-token attention mass */
    float  *imp_head;            /* [nh * max_kv_slots] per-head scratch */
    STL3State l3;                /* Phase-2 L3 disk eviction state (fp NULL = off) */

    /* KV cache INT8 quantization (~4x memory bandwidth reduction).
     * Uses fixed scale 1/127: K-norm already ensures K ≈ [-1, 1],
     * so round(K_f32 * 127) → INT8 is near-lossless. Same factor for V. */
    int      use_kv_q8;        /* 1 = use INT8 KV cache instead of F32 */
    int8_t ***v_cache_q8;      /* [n_layers][n_blocks] per-block INT8 V cache */
    float  **k_scale;          /* [n_layers][max_seq * nkv] per-token per-head K max-abs scale */
    float  **v_scale;          /* [n_layers][max_seq * nkv] per-token per-head V max-abs scale */

    /* KV cache Q4 compression (--kv-q4): 40-byte Q4_0 payload per 64 floats
     * (l3_q4_pack64/dot64/vacc64), ~1.65x denser than the INT8 form. Decode
     * attention scores/accumulates straight from the payloads (no dequant).
     * f32 cache is retained (exact prefill attention, L3 eviction, ppl-f32
     * A/B reference). Mutually exclusive with use_kv_q8. */
    int      use_kv_q4;        /* 1 = use Q4 KV payload cache */
    uint8_t ***k_cache_q4;     /* [n_layers][n_blocks] Q4 K payload blocks */
    uint8_t ***v_cache_q4;     /* [n_layers][n_blocks] Q4 V payload blocks */

    int     profile_decode;
    double  dec_t_qkv;
    double  dec_t_attn;
    double  dec_t_o;
    double  dec_t_gu;
    double  dec_t_down;
    double  dec_t_lm;
    double  dec_t_other;
} STQwenInferenceState;

/* ---- Safetensors Parser API ---- */

/**
 * Parse safetensors index file (model.safetensors.index.json) and config.json.
 * Populates STModelConfig with architecture parameters and tensor metadata.
 * Returns 0 on success, -1 on error.
 */
int st_parse_config(const char *model_dir, STModelConfig *cfg);

/* Transparent NPU backend hook (see vllm_npu.h). st_npu_set installs the
 * backend handle from main; st_npu_enabled reports whether offload is live. */
struct vllm_npu_s;
void st_npu_set(struct vllm_npu_s *npu);
int  st_npu_enabled(void);

/* Drop the cached NPU (layer, projection) weight strips. Called when a
 * DIFFERENT model is loaded; same-model reloads keep the cache. */
void st_npu_gw_clear_all(void);

/**
 * Load a specific tensor from safetensors files into float32 buffer.
 * Converts bfloat16/float16 to float32 automatically.
 * dst must have (n_elems * sizeof(float)) bytes allocated.
 * Returns 0 on success, -1 if tensor not found.
 */
int st_load_tensor(const STModelConfig *cfg, const char *tensor_name,
                    float *dst, int expected_elems);

int st_load_tensor_slice(const STModelConfig *cfg, const char *tensor_name,
                         int64_t elem_offset, float *dst, int64_t elem_count);

/**
 * Load all tensors for a range of layers from safetensors files.
 * layer_start: first layer index (inclusive)
 * layer_end:   last layer index (exclusive)
 * weights:     destination weight buffer (must be pre-allocated)
 * Returns 0 on success, -1 on error.
 */
int st_load_layer_weights(const STModelConfig *cfg, STModelWeights *w,
                           int layer_start, int layer_end);

/**
 * Allocate all weight buffers for the model (calloc).
 * No tensors are loaded yet; use st_load_tensor / st_load_layer_weights.
 */
void st_weights_alloc(STModelWeights *w, const STModelConfig *cfg);

/**
 * Allocate weights for a specific number of layers (memory-efficient).
 * Set n_layers_to_alloc = -1 for all layers.
 */
void st_weights_alloc_layers(STModelWeights *w, const STModelConfig *cfg, int n_layers);
/* Same as above but F32 FFN weights skipped — only Q8_0 compact FFN allocated.
 * Saves ~22 GB for Qwen3-VL-8B (can load all 36 layers in ~17 GB RAM). */
void st_weights_alloc_layers_q8ffn(STModelWeights *w, const STModelConfig *cfg, int n_layers);

/**
 * Free all weight buffers.
 */
void st_weights_free(STModelWeights *w);

/* ================================================================
 * Q8_0 quantization / Free
 * ================================================================ */

/**
 * Free STModelConfig (releases tensor metadata arrays).
 */
void st_config_free(STModelConfig *cfg);

/* Q8_0 quantization: F32 → block-compressed (axiom: fixedpoint_quantize_saturate) */
void f32_to_q8_0(uint8_t *q8_out, const float *f32_in, int n_elements);

/* Q4_0 quantization: F32 → block-compressed (axiom: fixedpoint_quantize_saturate B=4) */
void f32_to_q4_0(uint8_t *q4_out, const float *f32_in, int n_elements);

/* Q4_0 pre-unpacked-to-int8: F32 → Q8_0-layout { f16 scale d; int8 qs[32]; }
 * (34 B/block, same rounding/scale as f32_to_q4_0, qs[i] = nibble-8). */
void f32_to_q4i8(uint8_t *q8_out, const float *f32_in, int n_elements);

/* Q2_1 quantization: F32 → block-compressed 2-bit, 16 B/32 elements
 * ({ d0,m0,d1,m1 } f16 + 8B codes; dequant w = m + d*q, q∈[0,3]).
 * Mixed-precision FFN (axiom: blas_precision_efficiency_tradeoff). */
void f32_to_q2_1(uint8_t *q2_out, const float *f32_in, int n_elements);

/* M4: Q4_0 4x4 row-interleaved repack (nibble-unpack-free NEON kernels).
 * In-place re-arrangement of one [rows][cols] Q4_0 weight matrix into
 * 4-row x 72B groups with every nibble byte XOR 0x88 (two's-complement
 * bias flip); total bytes unchanged. Returns 0 on success, -1 when the
 * matrix can't be repacked (rows%4, cols%32, non-NEON-dotprod target, or
 * VLLM_DISABLE_Q4_REPACK set) - buffer untouched in that case. */
int repack_q4_0_4x4_inplace(uint8_t *buf, int rows, int cols);
/* Reads VLLM_DISABLE_Q4_REPACK once; safe to call repeatedly. */
void st_q4_repack_init(void);

/* M4e: Q8_0 4x4 row-interleaved repack (llama.cpp block_q8_0x4, 136 B/block).
 * In-place re-arrangement of one [rows][cols] Q8_0 weight matrix (legacy
 * row-major 34 B/block) into 4-row x 136 B groups: { d[4] f16, qs[128] }
 * with qs[k*16 + m*4 + i] = row m value (k*4+i). Total bytes unchanged.
 * Returns 0 on success, -1 when the matrix can't be repacked (rows%4,
 * cols%32, or g_st_q8_repack cleared) - buffer untouched in that case. */
int repack_q8_0_4x4_inplace(uint8_t *buf, int rows, int cols);

/* P0: Q8_0 8x8 row-interleaved repack (block_q8_0x8, 272 B/block), same
 * in-place semantics as repack_q8_0_4x4_inplace. { d[8] f16, qs[256] } with
 * qs[k*32 + m*4 + i] = row m value (k*4+i). Requires rows%8, cols%32. */
int repack_q8_0_8x8_inplace(uint8_t *buf, int rows, int cols);

/* Dispatch: 8x8 (VLLM_Q8_8X8 default) or legacy 4x4. Used by the per-layer
 * load repack AND by main.c's lm_head repack so all Q8 matrices share ONE
 * layout. */
int repack_q8_0_tiled_inplace(uint8_t *buf, int rows, int cols);

/* G=256 group quantize stored in the Q8_0 34 B/block layout (one f32 scale
 * per 256 columns, repeated as f16 across the 8 inner sub-blocks). CPU Q8_0
 * kernels consume it unchanged; the NPU DIRECT backend reads it as K=256
 * blocks (/8 ioctls - the NPU-vs-CPU prefill crossover). */
void f32_to_g256q8(uint8_t *q8_out, const float *f32_in, int n_elements);

/* Batched Q8_0 matvec with row-tiled shared-input fusion (axiom:
 * block_matrix_assoc_natural + partition_alignment):
 *   out[p*out_stride + r] = dot(q8_w[r], x[p]) + bias[r]
 * x is PATCH-MAJOR [batch][cols]; patch-parallel with 4 rows accumulated per
 * pass (independent FMA chains sharing one x vector). cols MUST be a multiple
 * of 32; out_stride lets the caller pad rows. */
void st_q8_matvec_batched(float *__restrict out, const uint8_t *__restrict q8_w,
                          const float *__restrict x, const float *__restrict bias,
                          int batch, int rows, int cols, int out_stride);

/* Single-vector Q8_0 matvec: out[r] = dot(q8_w[r], x) + bias[r].
 * NO internal OMP (caller owns the parallelism — used inside fused ViT
 * layers where each thread processes one patch serially). Rows are unrolled
 * 4-way sharing one x vector (independent FMA chains). cols multiple of 32. */
void st_q8_matvec_single(float *__restrict out, const uint8_t *__restrict q8_w,
                         const float *__restrict x, const float *__restrict bias,
                         int rows, int cols);

/* Batch Q8_0 matvec: out[B][rows] = W[rows][cols] · x[B][cols] (+bias).
 * The weight matrix is streamed ONCE per row (row-parallel inside); each
 * patch accumulates in the same block order as st_q8_matvec_single, so the
 * numerics are bit-identical to calling it per patch — only the weight
 * re-read traffic is eliminated (visual ViT encode hot path). */
void st_q8_matvec_batch(float *__restrict out, const uint8_t *__restrict q8_w,
                        const float *__restrict bias,
                        const float *__restrict x,
                        int B, int rows, int cols);

/* ViT QKV projection with head-contiguous output: out[3][nh][B][hd] instead
 * of [B][3*vh], so the block attention streams each head's K/V sequentially.
 * rows = 3*vh; vh = vis_hidden, nh = vis_heads, hd = vh/nh. */
void st_q8_matvec_batch_qkv(float *__restrict out, const uint8_t *__restrict q8_w,
                            const float *__restrict bias,
                            const float *__restrict x,
                            int B, int rows, int vh, int nh, int hd);

/* ---- Qwen3-VL Inference API ---- */

/* Runtime switch to disable Q4_0 weights (default 0 = Q4_0 ON when available).
 * Set to 1 via CLI --no-q4 to force the Q8_0 path. */
extern int g_st_no_q4;

/* Weight mode: 0 = dual Q8_0+Q4_0 (default), 1 = Q4_0 nibble-only,
 * 2 = Q8_0-only, 3 = Q4_0 pre-unpacked int8 (q4i, ~9.8 GB, decode ~3x slower
 * — offline prefill workloads only). Set via CLI --wmode q4|q4i|q8|dual or
 * env VLLM_WMODE. Q4_0 nibble-only halves resident weight memory (~13.8 GB ->
 * ~6.3 GB) to remove swap pressure while keeping the fastest decode path. */
extern int g_st_wmode;
/* Resolve the effective weight mode (env VLLM_WMODE > --no-q4 > CLI --wmode);
 * returns 0 dual, 1 q4, 2 q8, 3 q4i, 4 g256. */
int st_wmode_effective(void);

/* M4e: Q8_0 4x4 repack master switch (default ON on NEON-dotprod builds).
 * Cleared by the geometric gate in st_weights_alloc_layers (rows%4 / cols%32
 * mismatch, wmode=g256/q4i, or VLLM_NPU_INFER). main.c reads it to keep the
 * lm_head repack consistent with the layer matrices. */
extern int g_st_q8_repack;

/* Runtime switch for mixed precision: force prefill GEMM (QKV/O/FFN) onto the
 * Q8_0 path while keeping Q4_0 for decode. Default 1 (P1, 2026-08-29 A/B on
 * Qwen3-VL-2B: TPOT 102.5 -> 76.2ms, -25.6%); set to 0 for Q4_0-everywhere.
 * prefill is unpack-bound (Q8_0 wins) while decode is weight-DRAM-bound
 * (Q4_0 wins), so this takes the best of both. */
extern int g_st_prefill_q8;

/* q2mix layered precision (--q2mix-tail N, default 0): in wmode=q2mix the LAST
 * N transformer layers keep Q4_0 FFN weights while the earlier layers use Q2_1.
 * 2-bit error hurts most in the final layers (ds4/DwarfStar observation).
 * Measured on Qwen3-VL-2B the tail does NOT recover 2-bit quality on dense
 * models (the FFN is the main residual path); mechanism verified via N=all. */
extern int g_q2mix_tail;

/* Mixed-precision (Q4_0 vs Q8_0) kernel benchmark on synthetic weights at the
 * real Qwen3-VL-8B shapes. No model load required. Triggered by --bench-mixed. */
void st_bench_mixed_precision(void);

/* Isolated "OTHER"-bucket probe (MRoPE + KV-store + INT8 quantize) at
 * S=1024/2048/4096, no model load. Triggered by --probe-other. */
void st_probe_other(void);

/* Sparse attention Phase-1 self-test (synthetic KV, no model load).
 * Triggered by --test-sparse. */
void st_test_sparse_attn(void);

/* Prefill mini-batch size (default 32). Set via CLI --prefill-batch for A/B. */
extern int g_st_prefill_batch;

/* Phase 1: block-sparse decode attention (default OFF = exact).
 * --sparse-attn enables; --sparse-k sets top blocks per head; --sparse-block
 * sets positions per block; --sparse-probe sets probe samples per block.
 * Axiom: probabilistic_selection_nc + sparse schedule. */
extern int g_sparse_attn;
extern int g_sparse_k;
extern int g_sparse_block;
extern int g_sparse_probe;
extern int g_l3_evict;
extern float g_l3_ratio;
extern int g_l3_min_seq;          /* --l3-min-seq: only evict when seq_len >= N */
extern char *g_l3_path;           /* --l3-path: L3 cache file path (NULL = kv_l3.bin) */
extern long long g_l3_max_size;   /* --l3-max-size: L3 file size cap in MB (0 = unlimited) */
extern char *g_l3_cur_path;       /* serve: per-user L3 file for the current request
                                   * (NULL = use g_l3_path, e.g. bench mode) */
extern long g_l3_user_ttl;        /* --l3-user-ttl: minutes a per-user L3 file is kept
                                   * after the user's last request (0 = never auto-clean) */
extern int g_kv_q4;              /* --kv-q4: in-memory Q4 KV payload cache */
extern long long g_l3_disk_hits;   /* decode tokens served from L3 disk (obsv.) */

/* Speculative decode (--spec, n-gram draft + parallel prefill verification).
 * When g_verify_logits is non-NULL, st_qwen_model_prefill_batch computes lm_head
 * logits for EVERY position of the (single, K<=SPEC_DRAFT_MAX) draft
 * mini-batch into g_verify_logits[pos*vocab] and records each position's
 * top-1 token in g_verify_pred. Owned by the serve layer; must be cleared
 * (NULL) after each verify so normal prefills keep the last-token-only
 * contract. */
#define SPEC_DRAFT_MAX 16
extern float *g_verify_logits;
extern int    g_verify_draft[SPEC_DRAFT_MAX];
extern int    g_verify_pred[SPEC_DRAFT_MAX];
extern float  g_verify_margin[SPEC_DRAFT_MAX];  /* top-1 - top-2 logit gap per position */
extern int    g_verify_invalid;
extern int    g_verify_pos;

/* Model-load progress (updated by st_load_layer_weights so the web admin
 * page can show a live "Loading layer X/Y" bar). 0/0 when idle. */
extern int g_model_load_layer;
extern int g_model_load_total;

/**
 * Initialize inference state from loaded weights.
 * Allocates KV caches and temp buffers.
 * Returns 0 on success, -1 on error.
 */
int st_qwen_inference_init(STQwenInferenceState *st, const STModelWeights *weights);

/**
 * Run one forward pass through Qwen3-VL language model.
 * token_id: input token to process
 * Logits stored in st->logits.
 */
void st_qwen_model_forward(STQwenInferenceState *st, int token_id);

/**
 * Mini-batch prefill (axiom: blas_qkv_fusion_categorical).
 * Processes n_tokens prompt tokens in mini-batches of 8, sharing weight
 * dequantization across tokens within each batch.
 * Reduces memory bandwidth ~8x for weight reads during prefill.
 * Returns 0 on success, -1 on error.
 */
int st_qwen_model_prefill_batch(STQwenInferenceState *st,
                                const int *token_ids, int n_tokens);

/**
 * Phase-2 L3 cold-block Q4 disk eviction (--l3-evict). Called after prefill by
 * both the bench path and the HTTP serve path. Packs cold KV blocks into the
 * L3 cache file (--l3-path, default kv_l3.bin) and frees them from RAM.
 */
void l3_evict_after_prefill(STQwenInferenceState *st);

/**
 * Copy the KV cache for the first `prefix_len` tokens
 * from `src` into `dst` (exact copy, zero quality loss). Enables prefix reuse:
 * a shared prompt prefix is prefilled once, then cheaply forked into multiple
 * request states so only the unique suffix needs prefill. `dst` must already be
 * initialized with the same model config and a max_seq >= src's max_seq.
 */
void st_qwen_copy_kv_prefix(STQwenInferenceState *dst,
                            const STQwenInferenceState *src,
                            int prefix_len);

/**
 * Continuous batching: decode ONE next token for `n_batch` independent
 * requests in a single forward pass. Each layer's weight matrices are read
 * once and reused across all requests (amortizing the decode memory-bound
 * weight traffic, MA5-PAR independent-session composition). Each request
 * attends only its own KV prefix (per-request cache_len), so results are
 * bit-identical to decoding each request separately with st_qwen_model_forward.
 * `sts[i]` must be distinct initialized states with the same model config;
 * `tokens[i]` is the next token for request i.
 */
void st_qwen_model_forward_batch(STQwenInferenceState *const *sts,
                                 int n_batch, const int *tokens);

/**
 * Multimodal prefill: image/video + text inference.
 * Replaces <image_pad>/<video_pad> placeholder tokens with visual embeddings
 * from the vision encoder and runs the full LLM forward pass.
 *
 * @param st             Initialized inference state
 * @param token_ids      Tokenized prompt (with vision placeholder tokens)
 * @param n_tokens       Total number of tokens in the prompt
 * @param visual_tokens  Visual encoder output [n_vis_tokens, dim]
 * @param n_vis_tokens   Number of visual tokens
 * @param grid_thw       [grid_t, grid_h, grid_w] for visual region 3D positions
 *                       Pass NULL for images (defaults to [1, gh, gw])
 * @return 0 on success, -1 on error
 */
int st_qwen_model_multimodal_prefill(STQwenInferenceState *st,
                                      const int *token_ids, int n_tokens,
                                      const float *visual_tokens, int n_vis_tokens,
                                      int grid_thw[3]);

/**
 * Multimodal prefill with multiple visual regions (multi-image / image+video).
 * grids: [n_regions][3] = {grid_t, grid_h, grid_w} per region, in the same
 * order the visual tokens were concatenated. Each region contributes
 * grid_t*grid_h*grid_w visual tokens (main grid only).
 * ds_features: optional DeepStack features [n_ds][n_vis_tokens, dim], injected
 * into the LLM's first n_ds layers at the visual-token positions (official
 * _deepstack_process). Pass NULL/0 to disable.
 * @return 0 on success, -1 on error
 */
int st_qwen_model_multimodal_prefill_ex(STQwenInferenceState *st,
                                         const int *token_ids, int n_tokens,
                                         const float *visual_tokens, int n_vis_tokens,
                                         const int *grids, int n_regions,
                                         const float *const *ds_features, int n_ds);

/**
 * Free inference state.
 */
void st_qwen_inference_free(STQwenInferenceState *st);

/* Free a KV cache block base (debug page-guarded allocations must be released
 * with the matching path; see vllm_safetensors.c). */
void st_qwen_kv_free_raw(void *raw);

/* Re-allocate the fp32/INT8 KV blocks freed by Phase-2 L3 eviction (they were
 * set NULL). Call before the next prefill so it can write the new request's
 * KV. Returns bytes allocated (0 = none freed previously). */
size_t st_qwen_kv_rebuild_freed(STQwenInferenceState *st);

/* ---- Disk KV persistence (serve --disk-kv) ----
 * Snapshot the first n_tokens rows of the F32 KV cache to `path` together
 * with the model geometry, so a later process (after a restart) can restore
 * the same prefix and only prefill the suffix. F32 keeps the restore bit-
 * exact with a fresh prefill (same guarantee as the in-RAM prefix reuse);
 * the checkpoint is invalid when the model geometry changes (checked on
 * load). Saves st->cache_len[0] rows max; call with n_tokens <= cache_len[0].
 * Returns 0 on success, nonzero on failure (corrupt/geometry mismatch...). */
int st_kv_disk_save(STQwenInferenceState *st, const char *path,
                    const int *tokens, int n_tokens);
int st_kv_disk_load(STQwenInferenceState *st, const char *path, int n_tokens);

/* ---- Utility: bfloat16 to float32 ---- */

/** Convert a single bfloat16 value to float32. */
static inline float bf16_to_f32(uint16_t bf16) {
    uint32_t bits = (uint32_t)bf16 << 16;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

/** Convert float32 to bfloat16. */
static inline uint16_t f32_to_bf16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    return (uint16_t)(bits >> 16);
}

#ifdef __cplusplus
}
#endif

#endif /* VLLM_SAFETENSORS_H */
