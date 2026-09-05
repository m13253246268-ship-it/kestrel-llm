/**
 * vLLM-Kestrel: Superposition Hybrid System vLLM Inference Engine
 * 
 * Axiom-Driven Design (from axiom_registry.json):
 *   - exponential_jump_set:  batch attention key expansion via window method
 *   - superposition_addition_clear: fused score accumulation
 *   - collapse_priority_rule: softmax selection with interference
 *   - odd_even_interference_logic: hierarchical partial result merge
 *   - ntt_isomorphism: transform-domain convolution acceleration
 */

#ifndef VLLM_SUPERPOS_H
#define VLLM_SUPERPOS_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ================================================================
 * Axiom Parameter Constants (from parameter_declaration axioms)
 * ================================================================ */
#define SHS_MODULUS_N        0xFFFFFFFBu    /* sha256_modulus: near-2^32 prime */
#define SHS_MODULUS_Q        12289          /* ntt modulus, typical */
#define SHS_MODULUS_M        65537          /* auxiliary modulus */
#define SHS_ORDER_R          1024           /* ntt primitive root order */
#define SHS_BITWIDTH         32             /* sha256_bitwidth */

/* vLLM architecture constants */
#define VLLM_MAX_SEQ_LEN     1024
#define VLLM_MAX_BATCH_SIZE  32
#define VLLM_NUM_HEADS       8
#define VLLM_HEAD_DIM        64
#define VLLM_HIDDEN_DIM      (VLLM_NUM_HEADS * VLLM_HEAD_DIM)
#define VLLM_NUM_LAYERS      4
#define VLLM_KV_BLOCK_SIZE   8              /* tokens per KV-cache block */
#define VLLM_MAX_BLOCKS      256

/* Tiny model constants for real inference */
#define TINY_HIDDEN_DIM      128
#define TINY_NUM_HEADS       4
#define TINY_HEAD_DIM        (TINY_HIDDEN_DIM / TINY_NUM_HEADS)
#define TINY_NUM_LAYERS      2
#define TINY_VOCAB_SIZE      2048
#define TINY_MAX_SEQ_LEN     256
#define TINY_FFN_DIM         352   /* 128 * 8/3 ≈ 341 → 352 for SwiGLU */
#define TINY_ROPE_THETA       10000.0f

/* Limit superposition set size to prevent exponential blow-up */
#define SHS_MAX_SUPERPOSITION_SIZE  4096

/* ================================================================
 * Core SHS Type Definitions (from state_as_tuple axiom)
 * ================================================================ */

/** 
 * SHS Superposition State: (value, exponent) tuple.
 * Represents a single branch in superposition.
 * Maps to: state_as_tuple axiom (id: state_as_tuple_0407a8c2)
 */
typedef struct {
    uint32_t value;      /* v: value in Z_N */
    int64_t  exponent;   /* e: exponent / energy level */
    float    amplitude;  /* probability amplitude for collapse */
} SHS_State;

/**
 * SHS Set State: superposition of multiple (v, e) tuples.
 * Maps to: set_operations axiom (id: set_operations_91154c82)
 */
typedef struct {
    SHS_State *states;
    size_t     count;
    size_t     capacity;
    uint32_t   modulus;   /* N for this set */
} SHS_SetState;

/* ================================================================
 * KV-Cache (PagedAttention) Structures
 * ================================================================ */

typedef struct {
    uint32_t key[VLLM_HEAD_DIM];   /* key vector */
    uint32_t value[VLLM_HEAD_DIM]; /* value vector */
} KVSlot;

typedef struct {
    KVSlot  slots[VLLM_KV_BLOCK_SIZE]; /* block of KVs */
    int32_t ref_count;
    int32_t block_id;
    bool    is_free;
} KVBlock;

typedef struct {
    KVBlock  blocks[VLLM_MAX_BLOCKS];
    int32_t  free_list[VLLM_MAX_BLOCKS];
    int32_t  free_count;
    uint32_t block_size;
} KVCacheManager;

/* ================================================================
 * Attention Structures
 * ================================================================ */

typedef struct {
    uint32_t data[VLLM_HEAD_DIM];
} HeadVector;

typedef struct {
    HeadVector q_heads[VLLM_NUM_HEADS]; /* query projections */
    HeadVector k_heads[VLLM_NUM_HEADS]; /* key projections */
    HeadVector v_heads[VLLM_NUM_HEADS]; /* value projections */
    HeadVector o_heads[VLLM_NUM_HEADS]; /* output */
} AttentionLayer;

/* ================================================================
 * Request / Scheduler Structures
 * ================================================================ */

typedef enum {
    REQ_WAITING,
    REQ_RUNNING,
    REQ_PREEMPTED,
    REQ_FINISHED
} RequestStatus;

typedef struct {
    uint32_t      request_id;
    uint32_t      token_ids[VLLM_MAX_SEQ_LEN];
    uint32_t      seq_len;
    uint32_t      current_pos;  /* next token position to generate */
    int32_t       block_table[VLLM_MAX_SEQ_LEN / VLLM_KV_BLOCK_SIZE + 1];
    int32_t       num_blocks;
    RequestStatus status;
    float         priority;     /* for probabilistic_selection_nc */
} InferenceRequest;

typedef struct {
    InferenceRequest requests[VLLM_MAX_BATCH_SIZE];
    int32_t           num_requests;
    int32_t           running_mask;  /* bitmask of running requests */
    int32_t           max_batch_size;
} RequestScheduler;

/* ================================================================
 * Model Configuration
 * ================================================================ */

typedef struct {
    uint32_t num_layers;
    uint32_t num_heads;
    uint32_t head_dim;
    uint32_t hidden_dim;
    uint32_t vocab_size;
    uint32_t max_seq_len;
    uint32_t kv_block_size;
    /* NTT parameters (from axiom: parameter_declaration for modulus q) */
    uint32_t ntt_modulus;
    uint32_t ntt_order;
} ModelConfig;

/* ================================================================
 * Core API: SHS Superposition Operations (Axiom-aligned)
 * ================================================================ */

/**
 * existence_axiom: Create initial state set S^(1) = {a mod N}
 * (id: existence_axiom_a03280c0)
 */
SHS_SetState shs_existence(uint32_t a, uint32_t modulus);

/**
 * exponential_jump_set: Apply base^jump to state set
 * S' = { (v * base^k) mod N for v in S, k in K }
 * (id: exponential_jump_set_259264ad)
 */
void shs_exponential_jump_set(SHS_SetState *S, uint32_t base,
                              const int64_t *jumps, size_t num_jumps);

/**
 * superposition_addition_clear: Add two values with superposition rule
 * (id: superposition_addition_clear_e6cf6fd3)
 * Special: (1,1) -> {2,3}
 */
void shs_superposition_add(SHS_SetState *result, uint32_t x, uint32_t y,
                           uint32_t modulus);

/**
 * general_superposition_addition: Add two set states element-wise
 * (id: general_superposition_addition_001)
 */
void shs_general_superposition_add(SHS_SetState *result,
                                   const SHS_SetState *X,
                                   const SHS_SetState *Y);

/**
 * odd_even_interference_logic: Interference based on parity
 * (id: odd_even_interference_logic_85b52caf)
 * If (e1%2) != (e2%2): keep state with smaller exponent
 * If (e1%2) == (e2%2): maintain both
 */
void shs_odd_even_interference(SHS_SetState *S);

/**
 * collapse_priority_rule: Collapse set to classical value
 * (id: collapse_priority_rule_e16488e2)
 * Priority: singleton > contains_one > size_criteria
 */
bool shs_collapse_priority(SHS_SetState *S, uint32_t *result);

/**
 * value_mod_interference: Interference when values share same residue
 * (id: axiom_value_mod_interf_001)
 */
void shs_value_mod_interference(SHS_SetState *S, uint32_t modulus_m);

/**
 * tensor_iterated_addition: Apply element to all states in set
 * (id: tensor_iterated_addition_e02cfff4)
 */
void shs_tensor_iterated_add(SHS_SetState *S, uint32_t element,
                             uint32_t modulus);

/**
 * nonclassical_multiplication: ⊗_nc operator
 * (id: _2297_nc_6fec81b9)
 */
void shs_nc_multiplication(SHS_SetState *result,
                           const SHS_SetState *X,
                           const SHS_SetState *Y);

/**
 * selective_interference_for_compression: Reduce superposition size
 * (id: selective_interference_for_compression_8ecadfb1)
 */
void shs_selective_interference_compress(SHS_SetState *S);

/* ================================================================
 * KV-Cache Operations
 * ================================================================ */

void kvcache_init(KVCacheManager *mgr, uint32_t block_size);
int32_t kvcache_alloc_block(KVCacheManager *mgr);
void kvcache_free_block(KVCacheManager *mgr, int32_t block_id);
void kvcache_write(KVCacheManager *mgr, int32_t block_id,
                   int slot_idx, const KVSlot *kv);
const KVSlot* kvcache_read(const KVCacheManager *mgr,
                           int32_t block_id, int slot_idx);

/* ================================================================
 * Attention Engine
 * ================================================================ */

/**
 * Exact multi-head scaled dot-product attention over the paged KV cache.
 * Scores are computed as Q·K^T, softmax-normalized, then used to form a
 * weighted sum of the cached value vectors into o_heads.
 */
void exact_attention(AttentionLayer *layer,
                     KVCacheManager *kvcache,
                     const int32_t *block_table,
                     int32_t num_blocks,
                     int32_t seq_pos);

/* ================================================================
 * Scheduler Operations
 * ================================================================ */

/**
 * probabilistic_selection_nc: Select next batch using superposition logic
 * (id: probabilistic_selection_nc_38aaa102)
 */
void scheduler_select_batch(RequestScheduler *sched);

/**
 * adaptive_selection_based_on_features: Dynamic routing
 * (id: adaptive_selection_based_on_features_07597cca)
 */
void scheduler_adaptive_route(RequestScheduler *sched);

/* ================================================================
 * Full Inference Pipeline
 * ================================================================ */

/**
 * Initialize vLLM-Kestrel engine with model configuration.
 */
void vllm_kestrel_init(const ModelConfig *config);

/**
 * Run one forward pass through all layers for a batch.
 * This is the main inference entry point.
 */
void vllm_kestrel_forward(KVCacheManager *kvcache,
                      RequestScheduler *sched);

/**
 * Generate one token per running request (autoregressive step).
 */
void vllm_kestrel_generate(KVCacheManager *kvcache,
                       RequestScheduler *sched,
                       uint32_t *output_tokens,
                       int32_t *num_outputs);

/**
 * Free all resources.
 */
void vllm_kestrel_cleanup(void);

/* ================================================================
 * Internal Set State Helpers (exposed for cross-file use)
 * ================================================================ */
void setstate_init(SHS_SetState *S, uint32_t modulus);
void setstate_free(SHS_SetState *S);
void setstate_add_element(SHS_SetState *S, uint32_t value,
                           int64_t exponent, float amplitude);
void setstate_copy(SHS_SetState *dst, const SHS_SetState *src);
void setstate_dedup(SHS_SetState *S);

/* ================================================================
 * NTT Accelerated Convolution (from ntt_isomorphism axiom)
 * ================================================================ */

/**
 * NTT forward transform.
 * Maps to: axiom_isomorphic_superposition_004
 */
void shs_ntt_forward(uint32_t *data, size_t n, uint32_t modulus,
                     uint32_t primitive_root);

/**
 * NTT inverse transform.
 * Maps to: axiom_modal_transition_conservation_005
 */
void shs_ntt_inverse(uint32_t *data, size_t n, uint32_t modulus,
                     uint32_t primitive_root);

/**
 * NTT-domain pointwise multiplication.
 * Equivalent to polynomial convolution.
 */
void shs_ntt_pointwise_multiply(uint32_t *a, const uint32_t *b,
                                size_t n, uint32_t modulus);

#ifdef __cplusplus
}
#endif

/* ================================================================
 * Real Transformer Inference API
 * ================================================================ */

/**
 * Transformer weight matrices for one decoder layer.
 * Uses standard float32 for real matrix operations.
 */
typedef struct {
    /* Attention weights */
    float q_weight[TINY_HIDDEN_DIM * TINY_HIDDEN_DIM];   /* W_q */
    float k_weight[TINY_HIDDEN_DIM * TINY_HIDDEN_DIM];   /* W_k */
    float v_weight[TINY_HIDDEN_DIM * TINY_HIDDEN_DIM];   /* W_v */
    float o_weight[TINY_HIDDEN_DIM * TINY_HIDDEN_DIM];   /* W_o */
    /* RMS Norm weights (attention + ffn) */
    float attn_norm[TINY_HIDDEN_DIM];   /* attention input norm */
    float ffn_norm[TINY_HIDDEN_DIM];    /* ffn input norm */
    /* FFN SwiGLU weights */
    float gate_weight[TINY_HIDDEN_DIM * TINY_FFN_DIM];   /* W_gate */
    float up_weight[TINY_HIDDEN_DIM * TINY_FFN_DIM];     /* W_up */
    float down_weight[TINY_FFN_DIM * TINY_HIDDEN_DIM];   /* W_down */
} LayerWeights;

/**
 * Full model weights.
 */
typedef struct {
    float token_embed[TINY_VOCAB_SIZE * TINY_HIDDEN_DIM];  /* embedding table */
    float final_norm[TINY_HIDDEN_DIM];                      /* final RMS norm */
    float lm_head[TINY_HIDDEN_DIM * TINY_VOCAB_SIZE];       /* output projection */
    LayerWeights layers[TINY_NUM_LAYERS];
} ModelWeights;

/**
 * Runtime state for one layer during inference.
 */
typedef struct {
    float hidden[TINY_HIDDEN_DIM];      /* current hidden state */
    float q[TINY_NUM_HEADS * TINY_HEAD_DIM]; /* Q projection */
    float k[TINY_NUM_HEADS * TINY_HEAD_DIM]; /* K projection */
    float v[TINY_NUM_HEADS * TINY_HEAD_DIM]; /* V projection */
    float attn_out[TINY_HIDDEN_DIM];    /* attention output */
    float ffn_gate[TINY_FFN_DIM];       /* SwiGLU gate */
    float ffn_up[TINY_FFN_DIM];         /* SwiGLU up */
    float ffn_down[TINY_HIDDEN_DIM];    /* SwiGLU down */
    /* KV cache for this layer */
    float k_cache[TINY_MAX_SEQ_LEN * TINY_NUM_HEADS * TINY_HEAD_DIM];
    float v_cache[TINY_MAX_SEQ_LEN * TINY_NUM_HEADS * TINY_HEAD_DIM];
    int   cache_len;
} LayerState;

/**
 * Full inference runtime state.
 */
typedef struct {
    ModelWeights weights;
    LayerState   layers[TINY_NUM_LAYERS];
    float        logits[TINY_VOCAB_SIZE];
    int          seq_len;
} InferenceState;

/**
 * Tokenizer: maps between strings and token IDs.
 */
typedef struct {
    char    **tokens;        /* token strings */
    int       vocab_size;
    int       max_token_len;
    int       bos_id;        /* BOS token ID */
    int       eos_id;        /* EOS token ID */
} Tokenizer;

/* ---- Transformer Operations ---- */

/** RMS LayerNorm: y = x * w / sqrt(mean(x^2) + eps) */
void rms_norm(float *out, const float *x, const float *weight,
              int dim, float eps);

/** Matrix-vector multiply: y = W @ x  (W: [cols][rows] or [out_dim][in_dim]) */
void matvec(float *out, const float *W, const float *x, int rows, int cols);

/** RoPE: apply rotary position embedding to Q and K */
void rope_apply(float *q, float *k, int head_dim, int pos, float theta);

/** Scaled dot-product attention with causal mask and KV cache */
void causal_attention(float *output, const float *q, float *k_cache,
                      float *v_cache, int cache_len, int cur_pos,
                      int num_heads, int head_dim);

/** SwiGLU FFN: down @ (silu(gate @ x) * (up @ x)) */
void swiglu_ffn(float *out, const float *x,
                const float *gate_W, const float *up_W, const float *down_W,
                int hidden_dim, int ffn_dim);

/** Forward one transformer layer */
void transformer_layer_forward(LayerState *st, LayerWeights *w,
                               int pos, int num_heads, int head_dim,
                               int hidden_dim, int ffn_dim);

/** Forward entire model for next token */
void model_forward(InferenceState *st, int token_id);

/** Sample from logits using temperature and top-p */
int sample_token(const float *logits, int vocab_size,
                  float temperature, float top_p);

/** Sample with optional min-p filtering (min_p > 0): keep tokens with
 * prob >= min_p * max_prob (ds4/DwarfStar default sampler; min_p <= 0
 * behaves exactly like sample_token). top_p applies on top when < 1. */
int sample_token_p(const float *logits, int vocab_size,
                   float temperature, float top_p, float min_p);

/* ---- Weight Initialization ---- */

/** Initialize model weights with deterministic but meaningful values */
void weights_init(ModelWeights *weights);

/** Save weights to binary file */
int weights_save(const ModelWeights *weights, const char *filename);

/** Load weights from binary file */
int weights_load(ModelWeights *weights, const char *filename);

/* ---- Tokenizer ---- */
/* Forward declaration: GGUFModelConfig used by tokenizer_init_from_gguf */
typedef struct GGUFModelConfig_s GGUFModelConfig;

/** Initialize tokenizer with built-in 913-token vocabulary */
void tokenizer_init(Tokenizer *tok);
/** Initialize tokenizer from GGUF model config (e.g. SmolLM2 49K vocab) */
void tokenizer_init_from_gguf(Tokenizer *tok, const GGUFModelConfig *cfg);
/** Free tokenizer resources */
void tokenizer_free(Tokenizer *tok);

/** Encode text to token IDs. Returns number of tokens. */
int tokenizer_encode(Tokenizer *tok, const char *text, int *token_ids, int max_len);

/** Decode a single token ID to string */
const char* tokenizer_decode(Tokenizer *tok, int token_id);

/** Direct token decode using built-in vocab (for embedding init, no Tokenizer needed) */
const char* tokenizer_decode_direct(int token_id);

/* ---- Full Inference Pipeline ---- */

/** Initialize inference state with weights and tokenizer */
InferenceState* inference_create(void);

/** Generate text from a prompt */
char* inference_generate(InferenceState *st, Tokenizer *tok,
                          const char *prompt, int max_new_tokens,
                          float temperature, float top_p);

/** Free inference state */
void inference_free(InferenceState *st);

/* ================================================================
 * GGUF Model Loading API
 * ================================================================ */

/** GGUF tensor type constants */
enum {
    GGML_TYPE_F32  = 0,
    GGML_TYPE_F16  = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    GGML_TYPE_Q5_0 = 6,
    GGML_TYPE_Q5_1 = 7,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q8_1 = 9,
};

/**
 * Dynamic model configuration - populated from GGUF metadata.
 * Allows loading models of any size, not just the tiny defaults.
 */
typedef struct GGUFModelConfig_s {
    int dim;           /* hidden dimension */
    int hidden_dim;
    int n_layers;
    int n_heads;
    int head_dim;
    int n_kv_heads;    /* GQA: number of KV heads */
    int ffn_dim;       /* SwiGLU intermediate dim */
    int vocab_size;
    int max_seq_len;
    float rope_theta;
    float norm_eps;
    int   arch_type;   /* 0=llama, 1=qwen, etc. */
    /* Tokenizer data (parsed from GGUF metadata) */
    int    tok_count;      /* number of tokens in vocabulary */
    char **tok_strings;    /* [tok_count] token strings (owned, free on cleanup) */
    int    bos_id;
    int    eos_id;
} GGUFModelConfig;

/**
 * Dynamically allocated model weights (for GGUF loading).
 * Unlike the fixed-size ModelWeights, this supports arbitrary dimensions.
 */
typedef struct {
    GGUFModelConfig cfg;
    /* Token embedding: [vocab_size][dim] */
    float *token_embed;
    /* Final RMS norm: [dim] */
    float *final_norm;
    /* LM head: [dim][vocab_size] */
    float *lm_head;
    /* Per-layer weights: pointer to array of LayerWeightsGGUF */
    float *attn_norm;     /* [n_layers][dim] */
    float *ffn_norm;      /* [n_layers][dim] */
    float *q_weight;      /* [n_layers][dim * n_heads * head_dim] */
    float *k_weight;      /* [n_layers][dim * n_kv_heads * head_dim] */
    float *v_weight;      /* [n_layers][dim * n_kv_heads * head_dim] */
    float *o_weight;      /* [n_layers][dim * n_heads * head_dim] */
    float *gate_weight;   /* [n_layers][ffn_dim * dim] */
    float *up_weight;     /* [n_layers][ffn_dim * dim] */
    float *down_weight;   /* [n_layers][dim * ffn_dim] */
    int   is_allocated;

    /* Q8_0 compact storage: raw GGUF type-8 tensor data (34 bytes/32 elements).
     * When present, dyn_matvec_q8() is used instead of dyn_matvec() for 3.7x
     * memory bandwidth reduction. NULL if tensor was F32/F16. */
    uint8_t *q8_token_embed;  /* [vocab_size * dim / 32 * 34] */
    uint8_t *q8_lm_head;      /* same as token_embed if tied */
    uint8_t *q8_q_weight;     /* [n_layers][dim * nh * hd / 32 * 34] */
    uint8_t *q8_k_weight;
    uint8_t *q8_v_weight;
    uint8_t *q8_o_weight;
    uint8_t *q8_gate_weight;
    uint8_t *q8_up_weight;
    uint8_t *q8_down_weight;
    int     has_q8;           /* 1 if Q8_0 compact weights are populated */
} GGUFWeights;

/**
 * Dynamic inference state for GGUF-loaded models.
 */
typedef struct {
    GGUFWeights        weights;
    GGUFModelConfig    cfg;
    /* Per-layer hidden states and KV caches (dynamically allocated) */
    float *hidden;         /* [n_layers][dim] */
    float **k_cache;       /* [n_layers] pointer to per-layer cache */
    float **v_cache;       /* [n_layers] pointer to per-layer cache */
    int   *cache_len;      /* [n_layers] */
    float *logits;         /* [vocab_size] */
    float *q_buf;          /* temp: [n_heads * head_dim] */
    float *k_buf;          /* temp: [n_kv_heads * head_dim] */
    float *v_buf;          /* temp: [n_kv_heads * head_dim] */
    float *attn_buf;       /* temp: [n_heads * head_dim] */
    float *ffn_buf;        /* temp: [ffn_dim] */
    float *ffn_out_buf;    /* temp: [dim] */
    float *scores_buf;     /* temp: [max_seq_len] shared across heads, avoids stack alloc */
    float *rope_freq;      /* precomputed: [head_dim/2] inv freqs, avoids powf per token */
    int    seq_len;
    int    is_allocated;
} GGUFInferenceState;

/**
 * Parse a GGUF file and extract model configuration from metadata.
 * Returns 0 on success, -1 on error.
 * Populates cfg with detected architecture parameters.
 */
int gguf_parse_config(const char *filename, GGUFModelConfig *cfg);

/**
 * Allocate and initialize GGUFWeights with synthetic random weights.
 * Designed for algorithm validation.
 *
 * @param w      Output: allocated and initialized weights
 * @param dim     Hidden dimension
 * @param heads   Attention heads
 * @param layers  Number of transformer layers
 * @param ffn     FFN intermediate dimension (SwiGLU)
 * @param hd      Head dimension per attention head
 * @param kv_heads  KV heads (GQA, use heads for MHA)
 * @param vocab   Vocabulary size
 * @param max_seq  Maximum sequence length for KV cache
 * @return 0 on success, -1 on allocation failure
 */
int gguf_weights_init_synthetic(GGUFWeights *w,
    int dim, int heads, int layers, int ffn, int hd, int kv_heads,
    int vocab, int max_seq);

/**
 * Load all tensor weights from a GGUF file into dynamically allocated
 * GGUFWeights structure. The caller must call gguf_weights_free() to release.
 * Returns 0 on success, -1 on error.
 */
int gguf_load_weights(const char *filename, GGUFWeights *weights);

/**
 * Save current weights to a GGUF file with proper metadata.
 * Returns 0 on success, -1 on error.
 */
int gguf_save_weights(const char *filename, const GGUFWeights *weights);

/**
 * Export fixed-size ModelWeights to a GGUF file.
 */
int gguf_export_tiny(const ModelWeights *w, const char *filename,
                      int dim, int n_layers, int n_heads, int ffn_dim,
                      int vocab_size);

/**
 * Free dynamically allocated GGUF weights.
 */
void gguf_weights_free(GGUFWeights *weights);

/**
 * Initialize inference state from GGUF-loaded weights.
 * Dynamically allocates all buffers and KV caches.
 * Returns 0 on success, -1 on error.
 */
int gguf_inference_init(GGUFInferenceState *st, const GGUFWeights *weights);

/**
 * Run one forward pass through the dynamic model.
 * token_id: input token to process
 * Returns: logits are stored in st->logits
 */
void gguf_model_forward(GGUFInferenceState *st, int token_id);

/**
 * Generate text from a prompt using GGUF-loaded model.
 * Returns heap-allocated string (caller frees).
 */
char* gguf_inference_generate(GGUFInferenceState *st, Tokenizer *tok,
                               const char *prompt, int max_new_tokens,
                               float temperature, float top_p);

/**
 * Free GGUF inference state.
 */
void gguf_inference_free(GGUFInferenceState *st);

#endif /* VLLM_SUPERPOS_H */
