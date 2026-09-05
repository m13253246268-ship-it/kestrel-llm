/**
 * vllm_matmul.h - Axiom-Guided Matrix Computation Utility
 *
 * A standalone matrix-optimization tool class built under the constraints of
 * axiom_registry.json. Each entry maps to a concrete axiom:
 *
 *   - vmm_gemm_blocked : blas_matrix_block_natural_isomorphism
 *       (1.15x~1.45x speedup, 20-30% IO reduction via shared input fusion)
 *   - vmm_quantize_rows / vmm_gemm_q8 : blas_precision_efficiency_tradeoff
 *       + axiom_arith_fixedpoint_quantize_001 (INT8, saturation clamp,
 *       closed-form error bound)
 *   - vmm_lut_* : axiom_arith_range_lookup_001 (deterministic lookup table,
 *       INT16 compressed, forward-only)
 *   - vmm_gemm_mask32 : blas_sparse_message_passing_schedule (block-level
 *       sparsity skip, deterministic; factor-graph scheduling NOT adopted)
 *   - vmm_norm_consts / vmm_norm_cache : normalization hint caching
 *       (axiom_arith_dynamic_scale_collapse_001 for constant folding;
 *        axiom_arith_mod_inverse_hint_verify_001 / mod_sqrt_hint_verify_001
 *        for the finite-field hint pattern; axiom_arith_fixedpoint_encode_001
 *        + fixedpoint_quantize_001 for the bounded deterministic cache key)
 *
 * Red lines enforced from the axiom registry:
 *   1. Determinism - identical input always maps to identical output
 *      (axiom_arith_gumbel_argmax_001: no input-dependent randomness).
 *   2. Bounded table size - activation tables are INT16 fixed-point
 *      compressed (axiom_arith_range_lookup_001 cost guidance).
 *   3. Forward-only - no backprop / reversible replay (irreversibility_penalty).
 *   4. Saturation, no wrap-around (axiom_arith_fixedpoint_quantize_001).
 */

#ifndef VLLM_MATMUL_H
#define VLLM_MATMUL_H

#include <stdint.h>
#include <stddef.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ================================================================
 * 1. Blocked GEMM
 *    (blas_matrix_block_natural_isomorphism, conf 0.85)
 * ================================================================ */

/**
 * Cache-tiled row-major GEMM: C = beta*C + A @ B
 *   A: [M][K] row-major, B: [K][N] row-major, C: [M][N] row-major.
 * Uses M/K/N three-level blocking with a 4x4 register micro-kernel.
 * The A-block (BM x BK) is loaded once per (M-tile, K-tile) and reused
 * across every N-tile = "shared input fusion" (20-30% IO reduction).
 *
 * @param beta     0.0f overwrites C; any other value accumulates beta*C.
 * @param threads  <=0 uses OpenMP default; 1 forces single-threaded.
 */
void vmm_gemm_blocked(float *C, const float *A, const float *B,
                      int M, int N, int K, float beta, int threads);

/* ================================================================
 * 2. INT8 quantized GEMM
 *    (blas_precision_efficiency_tradeoff + axiom_arith_fixedpoint_quantize_001)
 * ================================================================ */

/**
 * Per-row INT8 quantization of a weight matrix W[rows][cols].
 *   scale[r] = max_abs(row) / 127
 *   q[r][c]  = clamp(round(w / scale[r]), -127, 127)   (saturation, no wrap)
 * The output of the full pipeline C = dequant(Wq) @ A satisfies the axiom
 * error bound: err <= max(scale) * 0.5 * max_m(sum_k |A[m][k]|).
 */
void vmm_quantize_rows(int8_t *q, float *scale,
                       const float *W, int rows, int cols);

/**
 * Weight-quantized GEMM: C[m][n] = scale[n] * sum_k A[m][k] * Wq[n][k]
 *   A kept fp32 (hybrid precision), accumulation in fp32.
 * @param err_bound_out optional: receives the closed-form absolute error
 *        bound (axiom: eps <= eps1*||Y|| + eps2*||X|| + eps1*eps2 with eps2=0).
 */
void vmm_gemm_q8(float *C, const float *A, const int8_t *Wq,
                 const float *Wscale, int M, int N, int K,
                 float *err_bound_out);

/* ================================================================
 * 3. Deterministic activation lookup tables
 *    (axiom_arith_range_lookup_001, conf 0.88; INT16 compression)
 * ================================================================ */

#define VMM_LUT_Q16              2048   /* fixed-point scale: real = q / Q16 */
#define VMM_LUT_DEFAULT_RANGE    8.0f   /* symmetric coverage [-8, 8] */
#define VMM_LUT_DEFAULT_ENTRIES  8192   /* 8192 * 2B = 16KB per table */

typedef enum {
    VMM_LUT_SILU = 0,   /* x * sigmoid(x) */
    VMM_LUT_GELU,       /* tanh approximation */
    VMM_LUT_SIGMOID     /* 1 / (1 + exp(-x)) */
} VMM_LUTKind;

typedef struct {
    VMM_LUTKind kind;
    int         n_entries;
    float       range;   /* symmetric: covers [-range, +range] */
    float       step;    /* 2*range / (n_entries - 1) */
    int16_t    *table;   /* INT16 fixed-point, deterministic, forward-only */
} vmm_lut_t;

/** Build the lookup table. Returns 0 on success, -1 on invalid args/OOM. */
int vmm_lut_init(vmm_lut_t *lut, VMM_LUTKind kind, float range, int n_entries);

/** Release table memory. */
void vmm_lut_free(vmm_lut_t *lut);

/** Evaluate f(x) via clamped linear interpolation (O(1), deterministic). */
float vmm_lut_apply(const vmm_lut_t *lut, float x);

/** In-place element-wise LUT application over a vector. */
void vmm_lut_apply_vec(const vmm_lut_t *lut, float *x, int n);

#define VMM_LUT_SILU_RANGE 12.0f   /* beyond |x|>range, silu is linear:
                                    *   x>range -> x, x<-range -> 0,
                                    *   err <= range*e^-range ~ 7e-5 */

/**
 * Deterministic SiLU via the default INT16 LUT with asymptotic extension.
 * Matches fast_silu's asymptotes exactly (x>20->x, x<-20->0) and extends
 * them from |x|>12 with error <= 12*e^-12 ~ 7.4e-5. Mid-range error is
 * bounded by Q16 quantization + linear interpolation (~6e-4), ~100x below
 * the Q8_0 per-block quantization noise - safe inside the quantized paths.
 * Same input always yields the same bits (determinism red line).
 */
float vmm_lut_silu(float x);

/** Lazily-initialized default silu table (8192 x INT16, range 12). */
const vmm_lut_t *vmm_lut_silu_default(void);

/* ================================================================
 * 4. Block-level sparse GEMM
 *    (blas_sparse_message_passing_schedule subset, conf 0.8)
 * ================================================================ */

#define VMM_SPARSE_BLOCK 32   /* K-block granularity (aligned with Q8_0) */

/**
 * C = A @ W with block-level sparsity skipping.
 *   mask: [N][n_kb] bytes, n_kb = ceil(K / VMM_SPARSE_BLOCK).
 *   mask[n * n_kb + kb] != 0  -> compute the 32-wide K-block of row n.
 * Deterministic: result pattern follows the axiom result_set
 *   S_C = { (i,j) | exists k: (i,k) in S_A and (k,j) in S_B }.
 */
void vmm_gemm_mask32(float *C, const float *A, const float *W,
                     const uint8_t *mask, int M, int N, int K);

/* ================================================================
 * 5. Normalization hint cache (方案4 - normalization caching)
 *    axiom_arith_dynamic_scale_collapse_001: per-layer constant folding.
 *    axiom_arith_mod_inverse_hint_verify_001 / mod_sqrt_hint_verify_001:
 *      hint pattern for 1/sqrt(mean) - verifier O(1), prover offline.
 *    axiom_arith_fixedpoint_encode_001 + fixedpoint_quantize_001:
 *      bounded deterministic cache key (saturation, no wrap).
 * ================================================================ */

/**
 * Per-layer collapsed normalization constants.
 * (dynamic_scale_collapse_001: at inference all per-step float scales fold
 * into a single per-layer constant - collapse_correctness PASS.)
 */
typedef struct {
    float inv_dim;          /* 1 / dim, precomputed */
    float head_dim_scale;   /* 1 / sqrt(head_dim), precomputed */
    float eps;              /* norm epsilon */
} vmm_norm_consts_t;

/** Precompute all input-independent norm constants for a layer. */
static inline void vmm_norm_consts_init(vmm_norm_consts_t *nc,
                                        int dim, int head_dim, float eps) {
    nc->inv_dim = 1.0f / (float)dim;
    nc->head_dim_scale = 1.0f / sqrtf((float)head_dim);
    nc->eps = eps;
}

/** mean = ss / dim + eps (collapse: 1/d precomputed into inv_dim). */
static inline float vmm_norm_mean(const vmm_norm_consts_t *nc, float ss) {
    return ss * nc->inv_dim + nc->eps;
}

/**
 * Bounded deterministic cache of mean -> 1/sqrt(mean).
 * Key is fixed-point quantized (encode_001), saturated (quantize_001),
 * INT16-stored (bounded memory red line), direct-mapped (deterministic -
 * no random eviction, gumbel determinism red line).
 * Float path computes rsqrtf on miss; a finite-field prover can instead
 * insert offline-computed hints via vmm_norm_hint_put (hint_verify pattern).
 */
typedef struct {
    int      n_slots;     /* power of 2, bounded */
    int      key_shift;   /* key = round(mean * 2^key_shift) */
    int      key_mask;
    int16_t *keys;        /* stored INT16 keys */
    float   *invs;        /* cached 1/sqrt(mean) values */
    long     hits;
    long     misses;
} vmm_norm_cache_t;

#define VMM_NORM_CACHE_DEFAULT_LOG2  10   /* 1024 slots */
#define VMM_NORM_CACHE_DEFAULT_SHIFT 10   /* key resolution 2^-10 */

/** Allocate a direct-mapped hint cache. Returns 0 on success, -1 on OOM. */
int vmm_norm_cache_init(vmm_norm_cache_t *c, int log2_slots, int key_shift);

/** Release cache memory. */
void vmm_norm_cache_free(vmm_norm_cache_t *c);

/**
 * 1/sqrt(mean) with hint-cache: hit returns the cached value (bitwise
 * identical), miss computes rsqrtf and stores it. Deterministic.
 */
float vmm_norm_rsqrt(vmm_norm_cache_t *c, float mean);

/**
 * Insert an externally computed hint (finite-field prover path:
 * mean_key -> inv_sqrt_mean precomputed off-circuit via Fermat +
 * Tonelli-Shanks, then verified on-chip in 1 mul gate). The hint value is
 * served verbatim on subsequent hits.
 */
void vmm_norm_hint_put(vmm_norm_cache_t *c, float mean, float hint);

/* ================================================================
 * 6. Self test (positive + negative)
 * ================================================================ */

/**
 * Validates all kernels against a naive reference, checks the quantized
 * error bound, LUT interpolation error, and negative cases (forged sparse
 * mask must be honored; out-of-range quantization must saturate, not wrap).
 * Prints per-case PASS/FAIL lines. Returns the number of failures (0 = OK).
 */
int vmm_self_test(void);

#ifdef __cplusplus
}
#endif

#endif /* VLLM_MATMUL_H */
