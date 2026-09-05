/**
 * vllm_safetensors.c - Safetensors parser + Qwen3-VL inference engine
 *
 * Features:
 *   - Parse HuggingFace safetensors shard files
 *   - bfloat16 → float32 weight conversion
 *   - Qwen3-VL language model forward pass (Q/K norms, MRoPE, GQA)
 *   - Memory-efficient: loads weights per-layer or all-at-once
 *
 * Qwen3-VL-8B-Instruct architecture:
 *   dim=4096, layers=36, heads=32, kv_heads=8, head_dim=128
 *   ffn=12288, vocab=151936
 *   MRoPE: 3D rotary with sections [24, 20, 20]
 *   Q/K RMS norms before RoPE
 *   Untied lm_head
 *
 * Third-party attribution:
 *   Several NEON quantized GEMM/GEMV kernels below derive from llama.cpp
 *   (https://github.com/ggerganov/llama.cpp), redistributed under the MIT
 *   License, Copyright (c) 2023-2026 The ggml authors. The MIT license text
 *   is reproduced at the top of tools/llama_gemm_q4_0_4x4_asm.c.
 *   Derivations (adapted and/or word-for-word ported in-place):
 *     - ggml_gemm_q4_0_4x4_q8_0  -> tools/llama_gemm_q4_0_4x4_asm.c (verbatim
 *       mechanical extraction, included at the M4d fast path)
 *     - ggml_gemm_q8_0_4x4_q8_0 NEON segment -> llama_gemm_q8_0_4x4_q8_0_neon()
 *     - ggml_gemv_q4_0_4x4_q8_0 -> q4x4_dot1_group16_gemv() (M4f)
 *     - ggml_gemv_q8_0_4x4_q8_0 -> q8_0 4x4 decode GEMV (M4e fast path)
 *     - ggml_vec_dot_q8_0_q8_0 / ggml_vec_dot_q4_0_q8_0 (arm/quants.c) ->
 *       SDOT batch-matvec / Q4_0 decode kernels referenced below
 *   Quantization block layouts (block_q4_0x4/q8_0x4, ggml block_q4_0/q8_0)
 *   are upstream-defined data formats used for wire/interchange compat; they
 *   are implemented independently here.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "vllm_safetensors.h"
#include "vllm_util.h"
#include "vllm_platform.h"
#include "vllm_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <sys/stat.h>
#include <errno.h>
#include <unistd.h>
#include <time.h>

/* 单平台（RK3588/aarch64）：SIMD 层统一走 NEON（vllm_platform.h），
 * 不再保留 x86 intrinsics 或标量仿真层。 */

/* Forward declaration: f16_to_f32_bits is defined in the Q8_0 helper
 * section below but used earlier (dequant on load); C99 implicit declaration
 * would otherwise clash with the static inline definition on GCC. */
static inline uint32_t f16_to_f32_bits(uint16_t h);

/* High-resolution wall-clock (seconds) for phase-level prefill profiling.
 * Provided by vllm_platform.h (st_now_sec) - portable across MSVC/aarch64. */

#include "vllm_tp.h"   /* self-contained thread pool (replaces OpenMP) */

/* NUMA/affinity binding is done ONCE inside the pool when workers are created
 * (vllm_tp.c: physical cores on x86 skipping SMT, A76 cluster on RK3588).
 * Kept as a no-op shim for call sites that used to bind via a parallel region
 * (Axiom: memory_bandwidth_reduction via core pinning). */
static void numa_bind_thread(void) { (void)0; }

/* Worker Pool abstraction — persistent thread pool, static scheduling.
 * Axiom: memory_bandwidth_reduction — persistent threads, reduced fork-join.
 * Usage: wp_parallel_for(start, end, fn, ctx) dispatches [start,end) to workers.
 * The fn signature is: void fn(void *ctx, int idx). */
static void wp_parallel_for(int start, int end, void (*fn)(void *ctx, int idx), void *ctx) {
    vllm_tp_parfor(start, end, fn, ctx);
}
/* wp_parallel_call: run the same function on all worker threads (no loop). */
static void wp_parallel_call(void (*fn)(void *ctx), void *ctx) {
    vllm_tp_parcall(fn, ctx);
}

/* Thread-count default (OMP_NUM_THREADS / VLLM_THREADS env, else 4 threads)
 * is handled by vllm_tp_init(0). RK3588 A76-only default: 4 (the A55 cluster
 * degrades the batched GEMM — measured 34 GFLOPS with 8 threads vs 72 GFLOPS
 * with 4 A76-only). Explicit OMP_NUM_THREADS wins. */
static void st_default_threads(void) {
    vllm_tp_init(0);
}

/* Q8_0 block-compressed storage: 34B per 32 floats (2B f16 scale + 32B int8) */
#define Q8_BYTES(n)  ((size_t)((n) + 31) / 32 * 34)
/* Q4_0 block-compressed storage: 18B per 32 floats (2B f16 scale + 16B nibbles) */
#define Q4_BYTES(n)  ((size_t)((n) + 31) / 32 * 18)
/* Q2_1 block-compressed storage: 16B per 32 floats
 * ({ d0,m0,d1,m1 } f16 + 8B 2-bit codes, two 16-elem sub-blocks) */
#define Q2_BYTES(n)  ((size_t)((n) + 31) / 32 * 16)

/* q2mix layered FFN precision (--q2mix-tail N): the FIRST (nl - N) layers use
 * the Q2_1 segment, the LAST N layers use the Q4_0 tail segment (2-bit error
 * hurts most near the output; ds4/DwarfStar). Both helpers are no-ops for
 * non-q2mix weights (has_q2 == 0 → Q4 stays full-layer). */
static inline int st_ffn_q2_layer(const STModelWeights *w, int l) {
    int tail = g_q2mix_tail;
    if (tail > w->n_layers_allocated) tail = w->n_layers_allocated;
    return w->has_q2 && w->q2_gate_weight && l < w->n_layers_allocated - tail;
}
static inline int st_ffn_q4_layer_off(const STModelWeights *w, int l) {
    int tail = g_q2mix_tail;
    if (tail > w->n_layers_allocated) tail = w->n_layers_allocated;
    if (w->has_q2 && w->q2_gate_weight && l >= w->n_layers_allocated - tail)
        return l - (w->n_layers_allocated - tail);   /* q2mix tail segment */
    return l;                                        /* dual/q4 full layer */
}

/* Default prefill mini-batch size (overridable via CLI --prefill-batch).
 * 256 amortizes the per-batch weight re-read: each mini-batch re-streams the
 * full layer weights from DRAM, so batch=32 re-reads weights ~8x more than
 * batch=256 for a 2K prompt (measured RK3588 2026-08-20: weight traffic is
 * the prefill GEMM bottleneck, not FLOPs). */
#define PREFILL_BATCH_SIZE 256

/* Runtime switch to disable Q4_0 weights (default 0 = Q4_0 ON when available).
 * Set to 1 via CLI --no-q4 to force the Q8_0 path for A/B benchmarking the
 * Q4_0 nibble-unpack cost vs Q8_0's doubled weight DRAM traffic. */
int g_st_no_q4 = 0;

/* M4: 4x4 Q4_0 repack (default ON on NEON dotprod targets). The weight
 * matrices are re-arranged at load time so the NEON kernels can consume the
 * nibbles directly (b<<4 / b&0xf0, /16 at the end) with zero per-nibble
 * unpack instructions - the unpack overhead that ate the Q4 bandwidth win.
 * Set VLLM_DISABLE_Q4_REPACK=1 to force the legacy per-block nibble-unpack
 * kernels (A/B). Only affects NEON dotprod builds; other targets no-op. */
int g_st_q4_repack = 1;

/* M4e: Q8_0 4x4 repack flag (llama.cpp block_q8_0x4 layout, 136 B/block).
 * 2026-08-22: prefill Q8 差距 3.18x 的根因与 Q4 相同（batched GEMV 结构 +
 * 逐 token 激活重读 vs llama 的 4x4 交织 GEMM）。启用后 Q8_0 权重在加载时
 * repack 为 block_q8_0x4（与 Q4 的 M4d 同一套布局家族），prefill 走
 * llama 移植的 q8x4_gemm_batched、decode 走 4x4 解交织 GEMV（f32 激活保持，
 * 保证 decode 数值路径不变）。解码/前缀缓存等其它 Q8 读取点同时切换。 */
int g_st_q8_repack = 1;

/* P0 8x8 tile gate (defined later in the M4e section) - forward decls so the
 * NPU gather (vllm_gw_row_worker) and the load-time repack dispatch can use
 * them before the definitions. */
static int q8_8x8_enabled(void);
int repack_q8_0_tiled_inplace(uint8_t *buf, int rows, int cols);

/* M4 repack scratch: one 4-row group (4*row_stride, at most 27648 B for a
 * 12288-wide down_proj). Pre-allocated in st_weights_alloc so the in-place
 * repack can never fail part-way - a mid-run allocation failure would leave
 * some matrices repacked and others not, while the kernels must see ONE
 * layout. If pre-allocation failed, g_st_q4_repack is cleared and every
 * matrix stays in the legacy layout (kernels fall back automatically). */
static uint8_t *g_q4r_scratch = NULL;
static size_t   g_q4r_scratch_cap = 0;

/* M4d (llama.cpp 4x4 asm GEMM) activation repack scratch: one repacked
 * block_q8_0x4 buffer of (n_batch/4) * (cols/32) * 136 B (e.g. S=512, cols
 * =4096 -> 2.23 MB). Grown on demand in q4x4_gemm_batched before the asm
 * path runs; a failed growth just falls back to the C TILE=3 path. */
static uint8_t *g_q8r_scratch = NULL;
static size_t   g_q8r_scratch_cap = 0;

/* Weight mode (default 0 = dual Q8_0+Q4_0):
 *   0 = dual  : keep BOTH copies (Qwen3-VL-8B ~13.8 GB resident).
 *   1 = q4    : Q4_0 nibble only (~6.3 GB) - halves resident weight memory,
 *               removes swap pressure on the 15 GB RK3588; default inference
 *               is already Q4_0 so this is lossless for the default path.
 *               Nibble 18 B/block is optimal for decode (DRAM-bandwidth-bound).
 *   2 = q8    : Q8_0 only (legacy --no-q4 behavior).
 *   3 = q4i   : Q4_0 pre-unpacked to int8 in Q8_0-layout buffers (~9.8 GB,
 *               q8_buf_q4=1) - decode is ~3x SLOWER (34 B/block vs 18 B),
 *               only useful for offline compute-bound prefill workloads.
 *   4 = g256  : G=256 group quantized (Q8_0-layout, 34 B/block with the SAME
 *               f16 scale repeated across 8 sub-blocks; ~8.2 GB). The coarse
 *               256-wide scale lets the NPU DIRECT backend run K=256 per
 *               submit (/8 ioctls) - the only measured path where NPU prefill
 *               beats the CPU NEON GEMM (K-block 32 vs 256). CPU Q8_0 kernels
 *               consume the Q8_0 layout unchanged; bit-exact CPU/NPU (same
 *               int8 codes + same f16 scale).
 * Override via env VLLM_WMODE=q4|q8|dual|q4i|g256 or CLI --wmode q4|q8|dual|q4i|g256. */
int g_st_wmode = 0;

/* Resolve the effective weight mode at allocation time (env wins, then the
 * legacy --no-q4 switch, then the CLI --wmode value). Exported so main.c's
 * lm_head quantization dispatch and the NPU paths agree on the mode. */
int st_wmode_effective(void) {
    const char *e = getenv("VLLM_WMODE");
    if (e && e[0]) {
        if (e[0] == 'q' || e[0] == 'Q') {
            if (e[1] == '2') return 5;   /* "q2mix": attention Q4_0 + FFN Q2_1 */
            if (e[1] == '4' && e[2] == 'i') return 3;   /* "q4i" before "q4" */
            if (e[1] == '4') return 1;
            if (e[1] == '8') return 2;
        }
        if (e[0] == 'g' || e[0] == 'G') return 4;       /* "g256" */
        if (e[0] == 'd' || e[0] == 'D') return 0;
    }
    if (g_st_no_q4) return 2;
    return g_st_wmode;
}

/* Mixed precision: prefill GEMM on Q8_0, decode on Q4_0.
 * Default 1 = decode Q4_0 + prefill Q8_0 (P1, 2026-08-29 A/B: TPOT 102.5 ->
 * 76.2ms, -25.6%, prefill unchanged; decode is DRAM-bound at Q8 reading
 * 1.83GB/token @27GB/s, halving to Q4_0 cuts the traffic). wmode=q4/q4i/
 * g256/q2mix have no Q8 layer weights and ignore this. --prefill-q8 (main.c)
 * forces it on; keep the flag for explicit config. See the extern in
 * vllm_safetensors.h. */
int g_st_prefill_q8 = 1;

/* q2mix layered precision: last N layers keep Q4_0 FFN. Default 0 = all-Q2
 * (measured on Qwen3-VL-2B: even tail=12 leaves 2-bit error accumulated in the
 * dense residual path garbled; tail=28 (=all-Q4) is clean, so the mechanism is
 * correct but 2-bit FFN quality is not recoverable on dense models). */
int g_q2mix_tail = 0;

/* Prefill mini-batch size (default 32, matches PREFILL_BATCH_SIZE). Larger
 * batches read the Q8_0 weight matrix once and reuse it across more tokens,
 * cutting weight DRAM traffic per token. Set via CLI --prefill-batch for A/B
 * benchmarking. */
int g_st_prefill_batch = PREFILL_BATCH_SIZE;

/* ================================================================
 * Phase 1: Block-sparse attention (opt-in, default OFF = exact).
 * Axiom: probabilistic_selection_nc ("top k by weight", deterministic),
 *        blas_sparse_message_passing_schedule (block-level sparse schedule).
 * Bounds decode attention to O(seq/BS + k*BS*head_dim) per head instead of
 * O(seq*head_dim). Softmax is renormalized over the selected blocks only.
 * Determinism red line (gumbel_argmax_001): same query+KV -> same selection.
 * ================================================================ */
int g_sparse_attn  = 0;   /* 0 = exact attention (default) */
int g_sparse_k     = 32;  /* top KV blocks kept per head (Phase-1 baseline: k=32, probe=8 keeps S=1024..4096 needle recall) */
int g_sparse_block = 32;  /* positions per KV block */
int g_sparse_probe = 8;   /* probe samples per block (max-dot fusion) */

/* Phase 2: L3 cold-block Q4 disk eviction (see vllm_l3.h). g_l3_evict is the
 * master switch; g_l3_ratio = fraction of cold blocks evicted per layer. */
int   g_l3_evict = 0;     /* 0 = L3 eviction disabled (default) */
float g_l3_ratio = 0.75f; /* evict the coldest 75% of blocks per layer */
int   g_l3_min_seq = 0;   /* 0 = always evict; else only when seq_len >= N */
char *g_l3_path = NULL;   /* --l3-path: L3 cache file path (NULL = kv_l3.bin) */
long long g_l3_max_size = 0; /* --l3-max-size: L3 file size cap in MB (0 = unlimited) */
char *g_l3_cur_path = NULL;  /* serve: per-user L3 file for the current request
                              * (NULL = use g_l3_path, e.g. bench mode) */
long g_l3_user_ttl = 30;  /* --l3-user-ttl: minutes a per-user L3 file is kept
                           * after the user's last request (0 = never auto-clean) */
long long g_l3_disk_hits = 0;   /* decode tokens served from L3 disk (observability) */
/* --kv-q4: in-memory Q4_0 KV payload cache (l3_q4 kernels, no disk). */
int g_kv_q4 = 0;
/* Web-admin model-load progress (see vllm_safetensors.h). */
int g_model_load_layer = 0;
int g_model_load_total = 0;

/* Speculative decode (--spec): n-gram draft + parallel prefill verification.
 * See the extern block in vllm_safetensors.h. */
float *g_verify_logits = NULL;
int    g_verify_draft[SPEC_DRAFT_MAX];
int    g_verify_pred[SPEC_DRAFT_MAX];
float  g_verify_margin[SPEC_DRAFT_MAX];
int    g_verify_invalid = 0;  /* set when the draft cannot be fully verified
                               * (e.g. draft longer than the prefill mini-batch);
                               * serve must treat the round as fully rejected */
int    g_verify_pos = -1;     /* MRoPE position base for verify-mode prefill:
                               * decode uses mrope_pos (multimodal) or seq_len
                               * (text); the verify prefill must use the same
                               * base or the draft KV is rotated wrong. -1 = no
                               * override (normal prefill uses seq_len). */

/* ================================================================
 * Transparent NPU acceleration (RK3588, opt-in via --npu).
 *
 * The NPU backend (vllm_npu.h) executes per-(layer, projection) int8 GEMM
 * operator models exported by tools/npu_export_ops.py; the safetensors model
 * format and the Q8_0/Q4_0 weights are never converted. Dispatch is a pure
 * function of (backend available, model present, FLOPs threshold) - when any
 * condition fails, the CPU (NEON/AVX) kernel runs exactly as before.
 * ================================================================ */
static vllm_npu_t *g_npu = NULL;   /* set by main via st_npu_set() */
static int g_npu_layer = -1;       /* current prefill layer (set per layer) */

/* Canary-scratch allocators are defined later in this file; declared here
 * because the NPU helper below uses them. */
static void *xq_alloc_canary(size_t bytes);
static void xq_free_canary(void *p, size_t bytes, const char *tag);

void st_npu_set(vllm_npu_t *npu) { g_npu = npu; }
int  st_npu_enabled(void) { return g_npu && vllm_npu_available(g_npu); }

/* Per-token Q8 quantization with per-G-block scales (token-major):
 *   q[m][k]   = clamp(round(x/scale), -127, 127)   per G-block
 *   scales[m][g] = max_abs(block)/127
 * This is the group-wise a_scale[g] the DIRECT NPU backend consumes directly.
 * G is the group width: 32 (Q8_0) or 256 (g256 wmode).
 * Saturation, no wrap - fixedpoint_quantize_saturate red line. */
/* Bit-level NaN/Inf detection. The build uses -ffast-math (assumes no
 * NaN), which makes the standard isnan()/isinf() compile to a constant
 * FALSE - decontamination checks would be silently dropped and a padding
 * NaN would flow into the quantizer. These inspect the raw bit pattern, so
 * -ffast-math cannot optimise them away. */
static inline int npu_isnan_f32(float x) {
    uint32_t u; memcpy(&u, &x, 4);
    return ((u & 0x7F800000u) == 0x7F800000u) && (u & 0x007FFFFFu);
}
static inline int npu_isnan_f64(double x) {
    uint64_t u; memcpy(&u, &x, 8);
    return ((u & 0x7FF0000000000000ULL) == 0x7FF0000000000000ULL) &&
           (u & 0x000FFFFFFFFFFFFFULL);
}
static inline int npu_isinf_f32(float x) {
    uint32_t u; memcpy(&u, &x, 4);
    return (u & 0x7FFFFFFFu) == 0x7F800000u;
}
static inline int npu_isinf_f64(double x) {
    uint64_t u; memcpy(&u, &x, 8);
    return (u & 0x7FFFFFFFFFFFFFFFULL) == 0x7FF0000000000000ULL;
}

static void st_npu_quantize_blocks(const float *__restrict x, int8_t *__restrict q,
                                   float *__restrict scales, int rows, int cols,
                                   int G) {
    int n_blocks = cols / G;
    for (int r = 0; r < rows; r++) {
        const float *xr = x + (size_t)r * cols;
        int8_t *qr = q + (size_t)r * cols;
        for (int b = 0; b < n_blocks; b++) {
            const float *xb = xr + (size_t)b * G;
            float amax = 1e-10f;
            for (int i = 0; i < G; i++) {
                float v = xb[i];
                if (npu_isnan_f32(v) || npu_isinf_f32(v)) v = 0.0f; /* padding decontaminate */
                float av = fabsf(v);
                if (av > amax) amax = av;
            }
            float s = amax / 127.0f;
            scales[(size_t)r * n_blocks + b] = s;
            int8_t *qb = qr + (size_t)b * G;
            for (int i = 0; i < G; i++) {
                float v = xb[i];
                /* Padding/edge rows can carry NaN (the CPU path masks them
                 * out via attention; the NPU path would otherwise propagate
                 * NaN through quantize -> int32 partial -> dequant, corrupting
                 * the whole output row). NaN -> 0 keeps the NPU GEMM
                 * numerically identical to the masked CPU path. */
                if (npu_isnan_f32(v) || npu_isinf_f32(v)) v = 0.0f;
                int iv = (int)(v / s + 0.5f);
                if (iv > 127) iv = 127;
                if (iv < -127) iv = -127;
                qb[i] = (int8_t)iv;
            }
        }
    }
}

/* INT4 activation quantization (B=4 branch of fixedpoint_quantize_saturate):
 *   q[m][k]   = clamp(round(x/scale), -7, 7)   per G-block
 *   scales[m][g] = max_abs(block)/7
 * The RK3588 direct int4 engine consumes 4-bit features ([-8,7]); feeding it
 * the int8-range codes from st_npu_quantize_blocks truncates the low nibble
 * (127->15, -127->1) and the output is garbage (~20-30x too small, wrong
 * signs) - exactly the pre-fix Q4 NPU symptom. The scale is written back so
 * the dequant (raw int dot * a_scale * b_scale) restores x*w exactly. */
static void st_npu_quantize_blocks_b4(const float *__restrict x, int8_t *__restrict q,
                                      float *__restrict scales, int rows, int cols,
                                      int G) {
    int n_blocks = cols / G;
    for (int r = 0; r < rows; r++) {
        const float *xr = x + (size_t)r * cols;
        int8_t *qr = q + (size_t)r * cols;
        for (int b = 0; b < n_blocks; b++) {
            const float *xb = xr + (size_t)b * G;
            float amax = 1e-10f;
            for (int i = 0; i < G; i++) {
                float av = fabsf(xb[i]);
                if (av > amax) amax = av;
            }
            float s = amax / 7.0f;
            scales[(size_t)r * n_blocks + b] = s;
            int8_t *qb = qr + (size_t)b * G;
            for (int i = 0; i < G; i++) {
                int v = (int)(xb[i] / s + 0.5f);
                if (v > 7) v = 7;
                if (v < -7) v = -7;
                qb[i] = (int8_t)v;
            }
        }
    }
}

/* Per-token Q8 quantization with ONE scale per token row (RKNN backend:
 * the exported int8 op models consume a per-token scale). Saturation, no
 * wrap - fixedpoint_quantize_saturate red line. */
static void st_npu_quantize_rows(const float *__restrict x, int8_t *__restrict q,
                                 float *__restrict scales, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        const float *xr = x + (size_t)r * cols;
        float amax = 1e-10f;
        for (int c = 0; c < cols; c++) {
            float av = fabsf(xr[c]);
            if (av > amax) amax = av;
        }
        float s = amax / 127.0f;
        scales[r] = s;
        int8_t *qr = q + (size_t)r * cols;
        for (int c = 0; c < cols; c++) {
            int v = (int)(xr[c] / s + 0.5f);
            if (v > 127) v = 127;
            if (v < -127) v = -127;
            qr[c] = (int8_t)v;
        }
    }
}

/* Q8_0 weight strip-cache for the DIRECT backend: the NPU group-wise matmul
 * needs contiguous int8 codes [N][K] + per-block fp32 scales [N][G], while
 * the engine's Q8_0 layout interleaves a 2-byte f16 scale header per 32-block.
 * Stripped once per (layer, projection) and reused across mini-batches.
 *
 * BOUNDED CACHE: 64 slots so a full 28-layer x (O+DOWN) prefill fits with
 * zero rebuild across reloads (2B: ~470MB), with a byte BUDGET gate so a
 * larger model (8B: O 25MB + DOWN 68MB/layer) cannot OOM the board (the
 * pre-256-slot cache OOM-killed at 9.8GB model + 8GB strip-cache). On
 * overflow (slots or bytes) the OLDEST entry is evicted (ring) and its
 * buffers freed - prefill/decode walk layers sequentially, so an evicted
 * entry always belongs to a completed layer. */
#define VLLM_NPU_GW_CACHE   64
#define VLLM_NPU_GW_BUDGET  (768u * 1024u * 1024u)   /* strip-cache byte cap */
typedef struct {
    int    layer, proj;
    int    N, K, G;
    int    q4;        /* 0 = Q8_0/G256 int8 codes, 1 = Q4_0 unpacked int4 values */
    int8_t *Wq;     /* [N][K] contiguous codes (headers stripped) */
    float  *bsc;    /* [N][G] block scales */
} st_npu_gw_t;
static st_npu_gw_t g_gw_cache[VLLM_NPU_GW_CACHE];
static int g_gw_n = 0;
static int g_gw_evict = 0;   /* ring cursor: next slot to evict on overflow */
static size_t g_gw_bytes = 0; /* live strip-cache bytes (budget gate) */

/* Free every cached (layer, projection) strip. Called when a DIFFERENT model
 * is loaded; same-model reloads keep the cache so a reloaded request skips
 * the rebuild storm. */
void st_npu_gw_clear_all(void) {
    for (int i = 0; i < g_gw_n; i++) {
        st_npu_gw_t *e = &g_gw_cache[i];
        if (e->Wq) xq_free_canary(e->Wq, (size_t)e->N * (size_t)e->K, "gw:wq");
        if (e->bsc) xq_free_canary(e->bsc, (size_t)e->N * (size_t)e->G * sizeof(float), "gw:bsc");
        memset(e, 0, sizeof(*e));
    }
    g_gw_n = 0; g_gw_evict = 0; g_gw_bytes = 0;
}

static st_npu_gw_t *st_npu_gw_get(int layer, int proj, int N, int K, int q4) {
    for (int i = 0; i < g_gw_n; i++)
        if (g_gw_cache[i].layer == layer && g_gw_cache[i].proj == proj &&
            g_gw_cache[i].N == N && g_gw_cache[i].K == K &&
            g_gw_cache[i].q4 == q4)
            return &g_gw_cache[i];
    return NULL;
}

static int g_xq_stagger;   /* fwd decl; real init (=1) at xq_alloc_init() */

typedef struct {
    st_npu_gw_t *e; const uint8_t *q8_w;
    int K, N, G, blk, layer, proj, row_stride;
} vllm_gw_ctx;

static void vllm_gw_row_worker(void *ctx_, int r) {
    vllm_gw_ctx *c = (vllm_gw_ctx *)ctx_;
    int8_t *wr = c->e->Wq + (size_t)r * c->K;
    float *bs = c->e->bsc + (size_t)r * c->G;
    /* g256 layout: blk/32 sub-blocks share ONE f16 scale (the first
     * sub-block's) and each 34 B sub-block stores [f16 scale][32 codes].
     * Gather the per-block scale + the (possibly strided) code bytes. */
    const int subper = c->blk / 32;
    const int nb = c->K / 32;
    if (g_st_q8_repack) {
        if (q8_8x8_enabled()) {
            const int r0 = r & ~7;
            const int m = r & 7;
            const uint8_t *gp = c->q8_w + (size_t)(r0 >> 3) * (size_t)nb * 272;
            for (int b = 0; b < nb; b++) {
                const uint8_t *bb = gp + (size_t)b * 272;
                const int g = b / subper;
                const int sub = b % subper;
                if (sub == 0) {
                    uint16_t h; memcpy(&h, bb + (size_t)m * 2, 2);
                    uint32_t fb = f16_to_f32_bits(h);
                    float d; memcpy(&d, &fb, 4);
                    if (npu_isnan_f32(d))
                        fprintf(stderr,
                                "[NPU-BSC-NAN] layer=%d proj=%d r=%d g=%d blk=%d "
                                "off=%lld f16=0x%04x row_stride=%d K=%d N=%d\n",
                                c->layer, c->proj, r, g, c->blk,
                                (long long)(b / subper) * subper * 34,
                                h, c->row_stride, c->K, c->N);
                    bs[g] = d;
                }
                for (int k = 0; k < 8; k++)
                    for (int i = 0; i < 4; i++)
                        wr[(size_t)g * c->blk + (size_t)sub * 32 + (k * 4 + i)] =
                            bb[16 + (size_t)k * 32 + (size_t)m * 4 + i];
            }
            return;
        }
        const int r0 = r & ~3;
        const int m = r & 3;
        const uint8_t *gp = c->q8_w + (size_t)(r0 >> 2) * (size_t)nb * 136;
        for (int b = 0; b < nb; b++) {
            const uint8_t *bb = gp + (size_t)b * 136;
            const int g = b / subper;
            const int sub = b % subper;
            if (sub == 0) {
                uint16_t h; memcpy(&h, bb + (size_t)m * 2, 2);
                uint32_t fb = f16_to_f32_bits(h);
                float d; memcpy(&d, &fb, 4);
                if (npu_isnan_f32(d))
                    fprintf(stderr,
                            "[NPU-BSC-NAN] layer=%d proj=%d r=%d g=%d blk=%d "
                            "off=%lld f16=0x%04x row_stride=%d K=%d N=%d\n",
                            c->layer, c->proj, r, g, c->blk,
                            (long long)(b / subper) * subper * 34,
                            h, c->row_stride, c->K, c->N);
                bs[g] = d;
            }
            for (int k = 0; k < 8; k++)
                for (int i = 0; i < 4; i++)
                    wr[(size_t)g * c->blk + (size_t)sub * 32 + (k * 4 + i)] =
                        bb[8 + (size_t)k * 16 + (size_t)m * 4 + i];
        }
        return;
    }
    const uint8_t *pr = c->q8_w + (size_t)r * c->row_stride;
    for (int g = 0; g < c->G; g++) {
        uint16_t h; memcpy(&h, pr + (size_t)g * subper * 34, 2);
        uint32_t fb = f16_to_f32_bits(h);
        float d; memcpy(&d, &fb, 4);
        if (npu_isnan_f32(d))
            fprintf(stderr,
                    "[NPU-BSC-NAN] layer=%d proj=%d r=%d g=%d blk=%d off=%lld "
                    "f16=0x%04x row_stride=%d K=%d N=%d\n",
                    c->layer, c->proj, r, g, c->blk, (long long)g * subper * 34,
                    h, c->row_stride, c->K, c->N);
        bs[g] = d;
        for (int sub = 0; sub < subper; sub++)
            memcpy(wr + (size_t)g * c->blk + (size_t)sub * 32,
                   pr + (size_t)(g * subper + sub) * 34 + 2, 32);
    }
}

static void st_npu_gw_build(const uint8_t *__restrict q8_w, int layer, int proj,
                            int N, int K) {
    /* wm==4 (g256): the Q8_0-layout weight repeats ONE f16 scale across 8
     * sub-blocks; the NPU consumes it as K=256 per submit (G=K/256). All
     * other modes keep the per-32 Q8_0 scale (G=K/32). VLLM_NPU_RESID_G
     * overrides the block width and MUST agree with st_npu_try_gemm_batched
     * (bsc array layout widens as the block shrinks: K/blk entries). */
    int wm = st_wmode_effective();
    int blk = (wm == 4) ? 256 : 32;   /* gw block width (== activation G) */
    {
        const char *rg = getenv("VLLM_NPU_RESID_G");
        if (rg && rg[0]) {
            int v = atoi(rg);
            if (v >= 32 && v <= 256 && (v % 32) == 0 && (K % v) == 0)
                blk = v;
        }
    }
    int G = K / blk;
    int row_stride = (K / 32) * 34;            /* storage is always 34 B/block */
    /* gw cache allocations outlive the calling layer: xq_alloc_canary sizes
     * itself with a per-layer stagger (g_npu_layer&3)*64, so an alloc here and
     * a later evict/free at a DIFFERENT layer would disagree on size and trip
     * a bogus canary check. Pin stagger off for the cache (sizes are exact). */
    int save_stagger = g_xq_stagger;
    g_xq_stagger = 0;
    st_npu_gw_t *e;
    const size_t need = (size_t)N * (size_t)K + (size_t)N * (size_t)G * sizeof(float);
    if (g_gw_n >= VLLM_NPU_GW_CACHE || g_gw_bytes + need > VLLM_NPU_GW_BUDGET) {
        /* Cache full OR byte budget exceeded: evict the oldest slot (ring)
         * and reuse it. The oldest entry always belongs to a completed layer
         * (sequential walks), so dropping it keeps peak memory bounded. */
        e = &g_gw_cache[g_gw_evict];
        if (e->Wq) xq_free_canary(e->Wq, (size_t)e->N * (size_t)e->K, "gw:wq");
        if (e->bsc) xq_free_canary(e->bsc, (size_t)e->N * (size_t)e->G * sizeof(float), "gw:bsc");
        g_gw_bytes -= (size_t)e->N * (size_t)e->K +
                      (size_t)e->N * (size_t)e->G * sizeof(float);
        memset(e, 0, sizeof(*e));
        g_gw_evict = (g_gw_evict + 1) % VLLM_NPU_GW_CACHE;
    } else {
        e = &g_gw_cache[g_gw_n++];
    }
    e->Wq = (int8_t *)xq_alloc_canary((size_t)N * K);
    e->bsc = (float *)xq_alloc_canary((size_t)N * G * sizeof(float));
    g_xq_stagger = save_stagger;
    if (!e->Wq || !e->bsc) {
        xq_free_canary(e->Wq, (size_t)N * K, "gw:wq");
        xq_free_canary(e->bsc, (size_t)N * G * sizeof(float), "gw:bsc");
        return;
    }
    g_gw_bytes += need;
    /* G256 conversion is row-independent: the per-row gather from the strided
     * Q8_0 layout is thread-pool-parallelized (measured ~10s of the NPU prefill
     * wall time, dominated by strided 34B-block reads; 4 threads cut it to
     * ~3s). */
    if (getenv("VLLM_NPU_GW_DBG")) {
        fprintf(stderr, "[GW-DBG] BUILD q8 layer=%d proj=%d N=%d K=%d G=%d "
                        "(cache_n=%d/%d bytes=%zuMB evict=%d)\n",
                layer, proj, N, K, G, g_gw_n, VLLM_NPU_GW_CACHE,
                g_gw_bytes >> 20, g_gw_evict);
        fflush(stderr);
    }
    vllm_gw_ctx gw = { e, q8_w, K, N, G, blk, layer, proj, row_stride };
    vllm_tp_parfor(0, N, vllm_gw_row_worker, &gw);
    e->layer = layer; e->proj = proj; e->N = N; e->K = K; e->G = G;
    e->q4 = 0;
}

/* Q4_0 nibble weight strip-cache for the DIRECT backend int4 path. The
 * engine's Q4_0 layout is [2B f16 scale][16B nibble] per 32 elements; the NPU
 * int4 GEMM wants contiguous int8 codes in [-8,7] ([N][K]) + per-32-block
 * fp32 scales ([N][K/32]). Nibble decode: low nibble = element j, high nibble
 * = element 16+j (see f32_to_q4_0); value = nibble - 8.
 *
 * The int16 NPU accumulate bounds the K-block to 32: 32 * 127 * 8 = 32512 <
 * 32767, so the Q4_0 32-wide group is both the natural and the safe block. */
typedef struct {
    st_npu_gw_t *e; const uint8_t *q4_w;
    int K, N, G, row_stride;
} vllm_gw4_ctx;

static void vllm_gw4_row_worker(void *ctx_, int r) {
    vllm_gw4_ctx *c = (vllm_gw4_ctx *)ctx_;
    const uint8_t *pr = c->q4_w + (size_t)r * c->row_stride;
    int8_t *wr = c->e->Wq + (size_t)r * c->K;
    float *bs = c->e->bsc + (size_t)r * c->G;
    for (int g = 0; g < c->G; g++) {
        const uint8_t *pb = pr + (size_t)g * 18;
        uint16_t h; memcpy(&h, pb, 2);
        uint32_t fb = f16_to_f32_bits(h);
        float d; memcpy(&d, &fb, 4);
        bs[g] = d;
        const uint8_t *nb = pb + 2;              /* 16 nibbles */
        int8_t *wq = wr + (size_t)g * 32;
#if ST_HAVE_NEON
        /* Q4_0: low nibble = element j, high nibble = element 16+j;
         * value = nibble - 8. Two 16-lane ops decode the whole block. */
        {
            uint8x16_t v = vld1q_u8(nb);
            uint8x16_t lo = vsubq_u8(vandq_u8(v, vdupq_n_u8(0x0F)), vdupq_n_u8(8));
            uint8x16_t hi = vsubq_u8(vshrq_n_u8(v, 4), vdupq_n_u8(8));
            vst1q_s8(wq,     vreinterpretq_s8_u8(lo));   /* elements 0..15  */
            vst1q_s8(wq + 16, vreinterpretq_s8_u8(hi));  /* elements 16..31 */
        }
#else
        for (int j = 0; j < 16; j++) {
            wq[j]      = (int8_t)((nb[j] & 0x0F) - 8);   /* element j     */
            wq[16 + j] = (int8_t)((nb[j] >> 4) - 8);     /* element 16+j  */
        }
#endif
    }
}

static void st_npu_gw_build_q4(const uint8_t *__restrict q4_w, int layer, int proj,
                               int N, int K) {
    int G = K / 32;
    int row_stride = (K / 32) * 18;          /* Q4_0 storage: 18 B per 32-block */
    int save_stagger = g_xq_stagger;
    g_xq_stagger = 0;
    st_npu_gw_t *e;
    const size_t need = (size_t)N * (size_t)K + (size_t)N * (size_t)G * sizeof(float);
    if (g_gw_n >= VLLM_NPU_GW_CACHE || g_gw_bytes + need > VLLM_NPU_GW_BUDGET) {
        e = &g_gw_cache[g_gw_evict];
        if (e->Wq) xq_free_canary(e->Wq, (size_t)e->N * (size_t)e->K, "gw:wq");
        if (e->bsc) xq_free_canary(e->bsc, (size_t)e->N * (size_t)e->G * sizeof(float), "gw:bsc");
        g_gw_bytes -= (size_t)e->N * (size_t)e->K +
                      (size_t)e->N * (size_t)e->G * sizeof(float);
        memset(e, 0, sizeof(*e));
        g_gw_evict = (g_gw_evict + 1) % VLLM_NPU_GW_CACHE;
    } else {
        e = &g_gw_cache[g_gw_n++];
    }
    e->Wq = (int8_t *)xq_alloc_canary((size_t)N * K);
    e->bsc = (float *)xq_alloc_canary((size_t)N * G * sizeof(float));
    g_xq_stagger = save_stagger;
    if (!e->Wq || !e->bsc) {
        xq_free_canary(e->Wq, (size_t)N * K, "gw:wq");
        xq_free_canary(e->bsc, (size_t)N * G * sizeof(float), "gw:bsc");
        return;
    }
    g_gw_bytes += need;
    if (getenv("VLLM_NPU_GW_DBG")) {
        fprintf(stderr, "[GW-DBG] BUILD q4 layer=%d proj=%d N=%d K=%d G=%d "
                        "(cache_n=%d/%d bytes=%zuMB evict=%d)\n",
                layer, proj, N, K, G, g_gw_n, VLLM_NPU_GW_CACHE,
                g_gw_bytes >> 20, g_gw_evict);
        fflush(stderr);
    }
    vllm_gw4_ctx gw4 = { e, q4_w, K, N, G, row_stride };
    vllm_tp_parfor(0, N, vllm_gw4_row_worker, &gw4);
    e->layer = layer; e->proj = proj; e->N = N; e->K = K; e->G = G;
    e->q4 = 1;
}

/* Transparent offload of one Q8 batched GEMM to the NPU. q8_w is the Q8_0
 * weight matrix (with headers) for the DIRECT backend; the RKNN backend uses
 * baked .rknn weights and ignores it. Returns 1 when the whole op ran on the
 * NPU, 0 = caller must run the CPU kernel.
 *
 * OFFLOAD POLICY (2026-08-20, measured): the K-chunked direct path costs a
 * fixed ~1ms per 32-wide K-block submit (NPU wait + CPU dequant), so a real
 * projection (e.g. Q 4096x4096 = 2048 submits) is ~7x SLOWER than the CPU
 * NEON path on prefill (M=260) and even slower on decode. Inference offload
 * is therefore DISABLED by default; set VLLM_NPU_INFER=1 to opt in. Even then
 * only a "single large GEMM" with few submits qualifies: flops >=
 * VLLM_NPU_INFER_MINFLOP (1e9) and submits <= VLLM_NPU_INFER_MAXSUB (256).
 * The NPU path itself stays fully wired for --npu-calib / --npu-selftest. */
static int st_npu_try_gemm_batched(float *__restrict out,
                                   const float *__restrict x_batch,
                                   const uint8_t *__restrict q8_w,
                                   int n_rows, int cols, int n_batch,
                                   int layer, int proj) {
    if (!g_npu || !vllm_npu_available(g_npu) || layer < 0) return 0;
    if (n_rows <= 0 || cols <= 0 || n_batch <= 0) return 0;
    /* VLLM_NPU_LOAD=<0|1|2> offload aggressiveness (default 0):
     *   0 = M5b baseline: prefill O(3)+DOWN(6) only, decode stays CPU.
     *   1 = +Q/K/V(0/1/2) prefill offload -> lower prefill CPU load
     *       (measured 2026-08-23: prefill CPU 322%->254%, TTFT +17%).
     *   2 = +decode offload (M=1 zero-padded to the %4 cube) -> lowest
     *       CPU load, slowest. On the 8B model this costs TPOT 318ms->
     *       3971ms (12.5x) for decode CPU 384%->251% - not worthwhile
     *       THERE, but kept for SMALL models where per-token NPU submits
     *       are few (small N/K -> few N-tiles) and the trade is viable.
     *   GATE/UP(4/5) stay CPU-only at every level (M5a silu-propagation
     *   correctness: gate|up together collapse the model). */
    int load = 0;
    {
        const char *le = getenv("VLLM_NPU_LOAD");
        if (le && le[0]) { int v = atoi(le); if (v > 0 && v <= 2) load = v; }
    }
    {
        const char *e = getenv("VLLM_NPU_INFER");
        if (!e || !e[0] || e[0] == '0') return 0;   /* off by default */
        /* LOAD=2 (decode) overrides the flops/submit gate: M=1 GEMMs are
         * far below minflop, but decode offload exists to free CPU cycles,
         * not to be fast. */
        if (load < 2) {
            double minflop = 1e9;
            const char *mf = getenv("VLLM_NPU_INFER_MINFLOP");
            if (mf && mf[0]) { double v = atof(mf); if (v > 0) minflop = v; }
            int maxsub = 256;
            const char *ms = getenv("VLLM_NPU_INFER_MAXSUB");
            if (ms && ms[0]) { int v = atoi(ms); if (v > 0) maxsub = v; }
            /* "single large GEMM" heuristic: big flops AND few submits. Real
             * projections fail the submit bound (Q/O 2048, gate/up 6144), so
             * prefill/decode stay on CPU; only small-K / small-N shapes qualify.
             * wmode g256 (G=256) cuts the submit count /8 and is the measured
             * path where NPU prefill beats the CPU NEON GEMM. */
            double flops = 2.0 * (double)n_batch * (double)n_rows * (double)cols;
            int wm = st_wmode_effective();
            int n_submits = (wm == 4)
                ? ((n_rows + 255) / 256) * ((cols + 1023) / 1024)   /* actual geometry */
                : ((n_rows + 255) / 256) * (cols / 32);             /* legacy bound */
            if (flops < minflop || n_submits > maxsub) return 0;
        }
    }
    if (vllm_npu_backend(g_npu) == VLLM_NPU_BACKEND_DIRECT && q8_w) {
        /* DIRECT: in-tree rknpu driver feeds our Q8_0 weights - zero deps. */
        /* M%4 (HW cube contract) is satisfied by the caller padding n_batch
         * up to a multiple of 4 below (all LOAD levels, same as the Q4 path);
         * vllm_npu_gemm_gw re-checks M%4 and falls back to CPU if unpadded. */
        if (cols % 32 != 0) return 0;
        /* Diagnostic: VLLM_NPU_PROJ_ONLY=<proj> (0=Q..6=DOWN) restricts the
         * int8 offload to one projection so a broken offload can be isolated
         * with a single CPU-comparison run. VLLM_NPU_PROJ_MASK=<bits> is the
         * multi-projection variant (bit p set = allow offload of proj p).
         * DEFAULT (no env): O(3) + DOWN(6) only - the M5a-qualified set. */
        {
            const char *po = getenv("VLLM_NPU_PROJ_ONLY");
            if (po && po[0]) {
                int only = atoi(po);
                if (proj != only) return 0;
            }
            const char *pm = getenv("VLLM_NPU_PROJ_MASK");
            if (pm && pm[0]) {
                unsigned mask = (unsigned)strtoul(pm, NULL, 0);
                if (proj < 0 || proj >= 32 || !(mask & (1u << proj))) return 0;
            } else if (load >= 1) {
                /* LOAD>=1 adds the remaining residual-stream projections
                 * Q/K/V(0/1/2) to O(3)+DOWN(6). GATE/UP(4/5) stay CPU-only
                 * at every level (silu-propagation collapse, M5a). */
                if (proj != 0 && proj != 1 && proj != 2 &&
                    proj != 3 && proj != 6) return 0;
            } else {
                /* M5a default: only the residual-stream projections O(3) and
                 * DOWN(6) are NPU-qualified. GATE/UP at G=256 collapse the
                 * model through silu propagation (expE8: VERIFY 0 FAIL but
                 * garbled) and are SLOWER on the NPU anyway (N=12288 -> 12
                 * N-tiles -> 75% of the submits). */
                if (proj != 3 && proj != 6) return 0;
            }
        }
        int wm = st_wmode_effective();
        /* Residual-error gate (M5a): the per-G activation quantize error
         * accumulates into the layer residual stream via the FFN (silu(gate)*up
         * -> down) and the attention O-add. Measured: ONE G=256 offload per
         * layer is tolerated (gate-only/up-only normal), but TWO (gate|up:
         * ~8% combined on the activated buffer) or the residual-stream
         * projections (O, DOWN) collapse the model (deterministic mojibake).
         * G=32 activation quantize is bit-exact vs the CPU G32 reference
         * (link test worst=0.0000); G=64/128 are the perf/precision A/B
         * options via VLLM_NPU_RESID_G. The g256 weight layout (per-256 scale
         * repeated across the 8 sub-blocks) makes the per-32 bsc values
         * identical to per-256 - only the bsc array layout widens. */
        int G = (wm == 4) ? 256 : 32;  /* activation quantize block width;
                                        * g256 wmode uses the per-256 protocol,
                                        * other modes the per-32 Q8_0 one */
        {
            const char *rg = getenv("VLLM_NPU_RESID_G");
            if (rg && rg[0]) {
                int v = atoi(rg);
                if (v >= 32 && v <= 256 && (v % 32) == 0 && (cols % v) == 0)
                    G = v;
            }
        }
        if (st_npu_gw_get(layer, proj, n_rows, cols, 0) == NULL)
            st_npu_gw_build(q8_w, layer, proj, n_rows, cols);
        st_npu_gw_t *e = st_npu_gw_get(layer, proj, n_rows, cols, 0);
        if (!e || !e->Wq || !e->bsc) return 0;
        /* NaN forensics: count NaN in the incoming activation (padding rows). */
        if (layer < 2 && (proj == 3 || proj == 6)) {
            int xnn = 0, lr_nan = 0;
            for (int i = 0; i < n_batch * cols; i++)
                if (npu_isnan_f32(x_batch[i])) xnn++;
            for (int k = 0; k < cols; k++)
                if (npu_isnan_f32(x_batch[(size_t)(n_batch - 1) * cols + k])) lr_nan++;
            if (xnn || lr_nan)
                fprintf(stderr,
                        "[NPU-XNAN] layer=%d proj=%d M=%d K=%d xnan=%d lastrow_nan=%d\n",
                        layer, proj, n_batch, cols, xnn, lr_nan);
        }

        /* Activations are [n_batch][cols] - cols is the INPUT (K) dimension
         * here (the weight matrix is [n_rows][cols]); n_rows is the output
         * width. K/V projections and down have K != N, so the quantization
         * width must be cols, not n_rows. */
        int q_blocks = cols / G;
        /* Pad M up to a multiple of 4 (HW cube contract): rows beyond the
         * real batch are zero activations so they contribute nothing, and the
         * output rows past n_batch are discarded - same pattern as the Q4
         * offload path. Without this, a prefill whose token count is not
         * %4==0 (e.g. 255) silently stays on the CPU for every LOAD level. */
        const int mb = (n_batch % 4) != 0 ? ((n_batch + 3) & ~3) : n_batch;
        int8_t *aq = (int8_t *)xq_alloc_canary((size_t)mb * cols);
        float  *asc = (float  *)xq_alloc_canary((size_t)mb * q_blocks * sizeof(float));
        float  *otmp = (mb != n_batch)
            ? (float *)malloc((size_t)mb * (size_t)n_rows * sizeof(float)) : NULL;
        if (!aq || !asc || (mb != n_batch && !otmp)) {
            xq_free_canary(aq, (size_t)mb * cols, "npu:aq");
            xq_free_canary(asc, (size_t)mb * q_blocks * sizeof(float), "npu:asc");
            free(otmp);
            return 0;
        }
        if (mb != n_batch) {
            memset(aq, 0, (size_t)mb * cols);
            for (int m = n_batch; m < mb; m++)
                for (int b = 0; b < q_blocks; b++)
                    asc[(size_t)m * q_blocks + b] = 1.0f;   /* 0 * scale = 0 */
            memset(otmp, 0, (size_t)mb * (size_t)n_rows * sizeof(float));
        }
        st_npu_quantize_blocks(x_batch, aq, asc, n_batch, cols, G);
        float *dst = otmp ? otmp : out;
        int ok = vllm_npu_gemm_gw(g_npu, dst, aq, asc, e->Wq, e->bsc,
                                  mb, n_rows, cols, G, 0,
                                  ((uint64_t)(uint32_t)layer << 32) | (uint32_t)proj);
        if (otmp) {
            /* Copy back only the real rows; padded rows are garbage. */
            for (int m = 0; m < n_batch; m++)
                memcpy(out + (size_t)m * n_rows, otmp + (size_t)m * n_rows,
                       (size_t)n_rows * sizeof(float));
        }
        /* VLLM_NPU_VERIFY=1: sample the NPU output against a CPU reference
         * computed from the SAME quantized artifacts (aq/asc + gw-cache
         * Wq/bsc) - pins down exactly which (layer, proj) offload is wrong.
         * On mismatch the GEMM falls back to the CPU kernel (correctness
         * safety net for the whole offload). Every 4th row x 3 tokens is
         * checked (sparser sampling can miss a localized row-range error
         * that still garbles the model output). */
        {
            const char *vf = getenv("VLLM_NPU_VERIFY");
            if (ok && vf && vf[0] && vf[0] != '0') {
                const int n_blk = cols / G;
                const int pm[3] = { 0, n_batch / 2, n_batch - 1 };
                double worst = 0.0; int wr = -1, wm2 = -1;
                double wgot = 0.0, wref = 0.0; int wg = -1;
                int n_checked = 0;
                for (int pi = 0; pi < 3; pi++) {
                    const int m = pm[pi];
                    if (m < 0) continue;
                    for (int nn = 0; nn < n_rows; nn += 4) {
                        double acc = 0.0;
                        for (int g = 0; g < n_blk; g++) {
                            int64_t p = 0;
                            const int8_t *a = aq + (size_t)m * cols + (size_t)g * G;
                            const int8_t *w = e->Wq + (size_t)nn * cols + (size_t)g * G;
                            for (int k = 0; k < G; k++)
                                p += (int64_t)a[k] * (int64_t)w[k];
                            acc += (double)p * asc[(size_t)m * n_blk + g] *
                                           e->bsc[(size_t)nn * n_blk + g];
                        }
                        const double got = out[(size_t)m * n_rows + nn];
                        const double err = fabs(acc - got);
                        n_checked++;
                        /* Bit-level NaN/Inf skip (-ffast-math kills isnan()). */
                        if (npu_isnan_f64(err) || npu_isinf_f64(err)) continue;
                        if (err > worst) { worst = err; wr = nn; wm2 = m;
                                           wgot = got; wref = acc; wg = 0; }
                    }
                }
                if (worst > 1e-3 || npu_isnan_f64(worst)) {
                    fprintf(stderr,
                            "[NPU-VERIFY] FAIL layer=%d proj=%d M=%d N=%d K=%d "
                            "G=%d worst_err=%.4f (m=%d n=%d) got=%.6f ref=%.6f "
                            "rel=%.4f asc[m,0]=%.3e bsc[n,0]=%.3e checked=%d -> CPU fallback\n",
                            layer, proj, n_batch, n_rows, cols, G, worst, wm2, wr,
                            wgot, wref, (wref != 0) ? worst / fabs(wref) : 0.0,
                            asc[(size_t)wm2 * n_blk], e->bsc[(size_t)wr * n_blk],
                            n_checked);
                    /* NaN forensics: scan the worst point's asc/bsc rows. */
                    int asc_nan = 0, bsc_nan = 0, first_an = -1, first_bn = -1;
                    for (int g = 0; g < n_blk; g++) {
                        if (npu_isnan_f32(asc[(size_t)wm2 * n_blk + g])) {
                            if (asc_nan == 0) first_an = g; asc_nan++;
                        }
                        if (npu_isnan_f32(e->bsc[(size_t)wr * n_blk + g])) {
                            if (bsc_nan == 0) first_bn = g; bsc_nan++;
                        }
                    }
                    if (asc_nan || bsc_nan)
                        fprintf(stderr,
                                "  [NPU-NAN-SCAN] m=%d n=%d asc_nan=%d(1st g=%d) "
                                "bsc_nan=%d(1st g=%d) n_blk=%d\n",
                                wm2, wr, asc_nan, first_an, bsc_nan, first_bn, n_blk);
                    xq_free_canary(aq, (size_t)mb * cols, "npu:aq");
                    xq_free_canary(asc, (size_t)mb * q_blocks * sizeof(float), "npu:asc");
                    free(otmp);
                    return 0;
                }
                /* The G256-protocol check above only proves the NPU faithfully
                 * executes the (aq/asc/Wq/bsc) contract. VLLM_NPU_VERIFY_CPU=1
                 * additionally rebuilds the CPU G32 semantic reference straight
                 * from the raw Q8_0-layout weights (34 B/block, per-32 f16
                 * scale) + per-32 activation quantize, and compares it against
                 * the NPU output - a big mismatch here means the activation or
                 * weight quantization protocol itself is broken, not just the
                 * NPU execution. */
                const char *vc = getenv("VLLM_NPU_VERIFY_CPU");
                if (vc && vc[0] && vc[0] != '0') {
                    const int rs = (cols / 32) * 34;   /* row stride, 34 B/block */
                    const int n_b32 = cols / 32;
                    double cworst = 0.0; int cwr = -1, cwm = -1;
                    double cref = 0.0;
                    for (int pi = 0; pi < 3; pi++) {
                        const int m = pm[pi];
                        if (m < 0) continue;
                        for (int nn = 0; nn < n_rows; nn += 4) {
                            double acc = 0.0;
                            for (int b = 0; b < n_b32; b++) {
                                const uint8_t *blk = q8_w + (size_t)nn * rs + (size_t)b * 34;
                                uint16_t h; memcpy(&h, blk, 2);
                                uint32_t fb = f16_to_f32_bits(h);
                                float dw; memcpy(&dw, &fb, 4);
                                float amax = 1e-10f;
                                for (int k = 0; k < 32; k++) {
                                    float av = fabsf(x_batch[(size_t)m * cols + (size_t)b * 32 + k]);
                                    if (av > amax) amax = av;
                                }
                                float da = amax / 127.0f;
                                int64_t p = 0;
                                for (int k = 0; k < 32; k++) {
                                    float xv = x_batch[(size_t)m * cols + (size_t)b * 32 + k];
                                    int v = (int)(xv / da + 0.5f);
                                    if (v > 127) v = 127;
                                    if (v < -127) v = -127;
                                    p += (int64_t)v * (int64_t)((const int8_t *)(blk + 2))[k];
                                }
                                acc += (double)p * da * dw;
                            }
                            const double got = out[(size_t)m * n_rows + nn];
                            const double err = fabs(acc - got);
                            if (err > cworst) { cworst = err; cwr = nn; cwm = m; cref = acc; }
                        }
                    }
                    if (cworst > 1.0) {
                        fprintf(stderr,
                                "[NPU-CPU-REF] FAIL layer=%d proj=%d M=%d N=%d K=%d "
                                "worst=%.4f (m=%d n=%d) got=%.3f cpu_ref=%.3f\n",
                                layer, proj, n_batch, n_rows, cols, cworst, cwm, cwr,
                                out[(size_t)cwm * n_rows + cwr], cref);
                    }
                }
            }
        }
        xq_free_canary(aq, (size_t)mb * cols, "npu:aq");
        xq_free_canary(asc, (size_t)mb * q_blocks * sizeof(float), "npu:asc");
        free(otmp);
        return ok;
    }
    /* RKNN backend: baked .rknn weights, per-token scale activation. The
     * activation width is cols (the INPUT dimension here, see DIRECT). */
    int8_t *aq = (int8_t *)xq_alloc_canary((size_t)n_batch * cols);
    float  *aw = (float  *)xq_alloc_canary((size_t)n_batch * sizeof(float));
    if (!aq || !aw) {
        xq_free_canary(aq, (size_t)n_batch * cols, "npu:aq");
        xq_free_canary(aw, (size_t)n_batch * sizeof(float), "npu:aw");
        return 0;
    }
    st_npu_quantize_rows(x_batch, aq, aw, n_batch, cols);
    int ok = vllm_npu_gemm_i8(g_npu, out, aq, aw, n_batch, n_rows, cols,
                              layer, proj);
    xq_free_canary(aq, (size_t)n_batch * cols, "npu:aq");
    xq_free_canary(aw, (size_t)n_batch * sizeof(float), "npu:aw");
    return ok;
}

/* Transparent offload of one Q4_0 batched GEMM to the NPU int4 path. q4_w is
 * the engine's Q4_0 nibble layout ([2B f16 scale][16B nibble] per 32). The
 * DIRECT backend consumes it as int4 weights x int8 activations -> int16
 * partials (prec=1). G is fixed to 32: the Q4_0 quantization group, which
 * also keeps the int16 accumulate inside 32*127*8 = 32512 < 32767.
 *
 * Same offload policy as the Q8 path (flops/minflop, submits/maxsub, M%4,
 * cols%32) - a too-small op stays on the CPU Q4 kernel. */
static int st_npu_try_gemm_batched_q4(float *__restrict out,
                                      const float *__restrict x_batch,
                                      const uint8_t *__restrict q4_w,
                                      int n_rows, int cols, int n_batch,
                                      int layer, int proj) {
    /* M4: the NPU int4 packer reads the legacy per-row Q4_0 layout; the 4x4
     * repack re-arranges it, so the two are incompatible. Q4 NPU offload is
     * already a dangerous opt-in (int4 x int4 feature truncation), so with
     * repack enabled it simply stays on the CPU 4x4 kernels. */
    if (g_st_q4_repack) return 0;
    /* DISABLED unless explicitly opted in: the RK3588 direct int4 engine
     * forces the FEATURE (activation) to 4-bit too (int4 x int4). The ±7
     * calibration data passes, but real ±127 activations truncate and the
     * output is garbage (~20-30x too small, wrong signs). int4 weight x
     * int8 feature is NOT supported by the HW (job hangs). Q4 therefore
     * stays on the CPU kernel; VLLM_NPU_Q4=1 is the dangerous opt-in. */
    {
        const char *q = getenv("VLLM_NPU_Q4");
        if (!q || !q[0] || q[0] == '0') return 0;
    }
    if (!g_npu || !vllm_npu_available(g_npu) || layer < 0) return 0;
    if (n_rows <= 0 || cols <= 0 || n_batch <= 0) return 0;
    {
        const char *e = getenv("VLLM_NPU_INFER");
        if (!e || !e[0] || e[0] == '0') return 0;   /* off by default */
        double minflop = 1e9;
        const char *mf = getenv("VLLM_NPU_INFER_MINFLOP");
        if (mf && mf[0]) { double v = atof(mf); if (v > 0) minflop = v; }
        /* G=32 int4 geometry: gate/up (12288x12288) estimates 18432 submits.
         * The default 256 blocked every G=32 Q4 GEMM, so Q4 prefill never
         * left the CPU; with the submit-retry in drm_submit_pc the storm is
         * tolerated, so default generously covers all projections. */
        int maxsub = 20000;
        const char *ms = getenv("VLLM_NPU_INFER_MAXSUB");
        if (ms && ms[0]) { int v = atoi(ms); if (v > 0) maxsub = v; }
        double flops = 2.0 * (double)n_batch * (double)n_rows * (double)cols;
        int n_submits = ((n_rows + 255) / 256) * (cols / 32);  /* G=32 geometry */
        if (flops < minflop || n_submits > maxsub) return 0;
    }
    if (vllm_npu_backend(g_npu) == VLLM_NPU_BACKEND_DIRECT && q4_w) {
        /* Diagnostic: VLLM_NPU_PROJ_ONLY=<proj> restricts offload to that
         * projection (0=Q,1=K,2=V,3=O,4=GATE,5=UP,6=DOWN) so a broken GEMM
         * can be isolated with a single CPU comparison. */
        {
            const char *po = getenv("VLLM_NPU_PROJ_ONLY");
            if (po && po[0]) {
                int only = atoi(po);
                if (proj != only) return 0;
            }
        }
        if (cols % 32 != 0) return 0;
        /* Pad M up to a multiple of 4 (HW cube contract): rows beyond the
         * real batch are zero activations so they contribute nothing, and the
         * output rows past n_batch are discarded. Without this, a prefill
         * whose token count is not %4==0 (e.g. 202) silently stays on CPU. */
        const int mp = (n_batch + 3) & ~3;
        int G = 32;                            /* Q4_0 group / int16-safe K-block */
        if (st_npu_gw_get(layer, proj, n_rows, cols, 1) == NULL)
            st_npu_gw_build_q4(q4_w, layer, proj, n_rows, cols);
        st_npu_gw_t *e = st_npu_gw_get(layer, proj, n_rows, cols, 1);
        if (!e || !e->Wq || !e->bsc) return 0;
        int q_blocks = cols / G;
        int8_t *aq = (int8_t *)xq_alloc_canary((size_t)mp * cols);
        float  *asc = (float  *)xq_alloc_canary((size_t)mp * q_blocks * sizeof(float));
        float  *otmp = (float *)malloc((size_t)mp * (size_t)n_rows * sizeof(float));
        if (!aq || !asc || !otmp) {
            xq_free_canary(aq, (size_t)mp * cols, "npu:aq4");
            xq_free_canary(asc, (size_t)mp * q_blocks * sizeof(float), "npu:asc4");
            free(otmp);
            return 0;
        }
        memset(aq, 0, (size_t)mp * cols);
        for (int m = n_batch; m < mp; m++)
            for (int b = 0; b < q_blocks; b++)
                asc[(size_t)m * q_blocks + b] = 1.0f;   /* pad rows: 0*scale=0 */
        st_npu_quantize_blocks_b4(x_batch, aq, asc, n_batch, cols, G);
        memset(otmp, 0, (size_t)mp * (size_t)n_rows * sizeof(float));
        int ok = vllm_npu_gemm_gw(g_npu, otmp, aq, asc, e->Wq, e->bsc,
                                  mp, n_rows, cols, G, 1,
                                  ((uint64_t)(uint32_t)layer << 32) | (uint32_t)proj);
        /* Diagnostic: VLLM_NPU_DUMP=1 recomputes this GEMM with a scalar
         * CPU oracle and prints the max |NPU - CPU| per projection - one
         * request then reveals exactly which projection is wrong. */
        if (ok == 1 && getenv("VLLM_NPU_DUMP")) {
            const int nblk = cols / G;
            double mx = 0.0, mx_sat = 0.0;
            int64_t mx_i = 0;
            for (int m = 0; m < n_batch; m++) {
                for (int nn = 0; nn < n_rows; nn++) {
                    double acc = 0.0, acc_sat = 0.0;
                    int64_t pmax = 0;
                    for (int g = 0; g < nblk; g++) {
                        int64_t p = 0;
                        int32_t p16 = 0;   /* int16-saturating accumulate */
                        for (int k = 0; k < G; k++) {
                            int64_t prod = (int64_t)aq[(size_t)m * cols + (size_t)g * G + k] *
                                           (int64_t)e->Wq[(size_t)nn * cols + (size_t)g * G + k];
                            p += prod;
                            p16 += (int32_t)prod;
                            if (p16 > 32767) p16 = 32767;
                            if (p16 < -32768) p16 = -32768;
                        }
                        acc += (double)p * asc[(size_t)m * nblk + g] *
                                           e->bsc[(size_t)nn * nblk + g];
                        acc_sat += (double)p16 * asc[(size_t)m * nblk + g] *
                                               e->bsc[(size_t)nn * nblk + g];
                        if (llabs(p) > pmax) pmax = llabs(p);
                    }
                    double d = fabs((double)otmp[(size_t)m * n_rows + nn] - acc);
                    double ds = fabs((double)otmp[(size_t)m * n_rows + nn] - acc_sat);
                    if (d > mx) mx = d;
                    if (ds > mx_sat) mx_sat = ds;
                    if (pmax > mx_i) mx_i = pmax;
                }
            }
            fprintf(stderr,
                    "[NPU-Q4] layer=%d proj=%d M=%d N=%d K=%d "
                    "maxdiff=%.3e satdiff=%.3e pmax=%lld\n",
                    layer, proj, n_batch, n_rows, cols, mx, mx_sat,
                    (long long)mx_i);
            fflush(stderr);
        }
        if (ok == 1) {
            memcpy(out, otmp, (size_t)n_batch * (size_t)n_rows * sizeof(float));
            xq_free_canary(aq, (size_t)mp * cols, "npu:aq4");
            xq_free_canary(asc, (size_t)mp * q_blocks * sizeof(float), "npu:asc4");
            free(otmp);
            return 1;
        }
        xq_free_canary(aq, (size_t)mp * cols, "npu:aq4");
        xq_free_canary(asc, (size_t)mp * q_blocks * sizeof(float), "npu:asc4");
        free(otmp);
        return 0;   /* NPU failed: caller falls back to the CPU kernel */
    }
    return 0;
}

/* ================================================================
 * Minimal JSON parser: extracts specific fields from flat JSON
 * (No external dependencies — hand-rolled for config parsing)
 * ================================================================ */

/* Find a JSON key and return pointer to its value start.
 * Returns NULL if not found. Supports nested objects limited to 4 levels. */
static const char* json_find_key(const char *json, const char *key) {
    char search[256];
    snprintf(search, sizeof(search), "\"%s\"", key);
    const char *pos = strstr(json, search);
    if (!pos) return NULL;
    pos += strlen(search);
    while (*pos == ' ' || *pos == ':') pos++;
    return pos;
}

/* Like json_find_key but limited to the range [start, end) */
static const char* json_find_key_in_range(const char *start, const char *end, const char *key) {
    char search[256];
    snprintf(search, sizeof(search), "\"%s\"", key);
    size_t key_len = strlen(search);
    size_t range_len = end - start;
    const char *pos = start;
    while (pos + key_len <= end) {
        pos = memchr(pos, '"', end - pos);
        if (!pos) return NULL;
        if (pos + key_len > end) return NULL;
        if (memcmp(pos, search, key_len) == 0) {
            pos += key_len;
            while (pos < end && (*pos == ' ' || *pos == ':')) pos++;
            if (pos >= end) return NULL;
            return pos;
        }
        pos++;
    }
    return NULL;
}

/* Parse an integer value at pos. Sets *out and returns
 * pointer one past the integer, or pos if failed. */
static const char* json_parse_int(const char *pos, int *out) {
    char *end;
    long v = strtol(pos, &end, 10);
    if (end == pos) { *out = 0; return pos; }
    *out = (int)v;
    return end;
}

/* Parse a float value at pos. */
static const char* json_parse_float(const char *pos, float *out) {
    char *end;
    float v = strtof(pos, &end);
    if (end == pos) { *out = 0.0f; return pos; }
    *out = v;
    return end;
}

/* Parse a string value (quoted) at pos. Allocates and returns new string.
 * Returns NULL if not a valid quoted string. */
static char* json_parse_string(const char *pos) {
    while (*pos && *pos != '"') pos++;
    if (!*pos) return NULL;
    pos++; /* skip opening " */
    const char *start = pos;
    while (*pos && *pos != '"') pos++;
    if (!*pos) return NULL;
    size_t len = pos - start;
    char *s = malloc(len + 1);
    if (!s) return NULL;
    memcpy(s, start, len);
    s[len] = '\0';
    return s;
}

/* ================================================================
 * bfloat16 conversion
 * ================================================================ */

static float bf16_to_f32_impl(uint16_t bf16) {
    /* bf16: [b15..b0] = [sign][exp8][mant7]
     * f32:  [b31..b0] = [sign][exp8][mant23]
     * Just shift left by 16 bits */
    uint32_t bits = (uint32_t)bf16 << 16;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static void bf16_to_f32_array(float *dst, const uint8_t *src, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        uint16_t bf16;
        memcpy(&bf16, src + i * 2, 2);
        dst[i] = bf16_to_f32_impl(bf16);
    }
}

static void f16_to_f32_array(float *dst, const uint8_t *src, int64_t n) {
    /* IEEE 754 f16 → f32 */
    for (int64_t i = 0; i < n; i++) {
        uint16_t h;
        memcpy(&h, src + i * 2, 2);
        int sign = (h >> 15) & 1;
        int exp  = (h >> 10) & 0x1F;
        int mant = h & 0x3FF;
        if (exp == 0) {
            if (mant == 0) { dst[i] = sign ? -0.0f : 0.0f; continue; }
            while (mant < 0x400) { mant <<= 1; exp--; }
            exp++; mant &= 0x3FF;
        } else if (exp == 0x1F) {
            dst[i] = NAN; continue;
        }
        int exp_f32 = exp + 112;
        uint32_t bits = ((uint32_t)sign << 31) | ((uint32_t)exp_f32 << 23) | ((uint32_t)mant << 13);
        memcpy(&dst[i], &bits, sizeof(float));
    }
}

/* ================================================================
 * Read entire file into memory
 * ================================================================ */

static uint8_t *read_file(const char *path, size_t *out_size) {
    FILE *f = st_fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[ST] Cannot open: %s (errno=%d)\n", path, errno);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long fsize64 = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (fsize64 <= 0) { fclose(f); return NULL; }
    /* Only read up to 1GB for full file reads; larger files need per-tensor loading */
    if (fsize64 > 1024 * 1024 * 1024) {
        fprintf(stderr, "[ST] File too large for full read: %s (%lld bytes)\n", path, fsize64);
        fclose(f);
        return NULL;
    }
    size_t fsize = (size_t)fsize64;
    uint8_t *buf = malloc(fsize + 1);
    if (!buf) { fclose(f); fprintf(stderr, "[ST] OOM reading %s (%zu bytes)\n", path, fsize); return NULL; }
    size_t nread = fread(buf, 1, fsize, f);
    fclose(f);
    if (nread != fsize) { free(buf); return NULL; }
    buf[fsize] = '\0';   /* null-terminate: callers pass buf to strstr/strchr */
    *out_size = fsize;
    return buf;
}

/**
 * Read only the JSON header portion of a safetensors file.
 * This avoids loading the multi-GB tensor data into memory.
 * Returns the header string (null-terminated) and sets data_offset.
 */
static char *read_st_header(const char *path) {
    FILE *f = st_fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[ST] Cannot open: %s (errno=%d)\n", path, errno);
        return NULL;
    }
    /* Read 8-byte header size (uint64 LE) */
    uint64_t header_size = 0;
    if (fread(&header_size, 8, 1, f) != 1) {
        fprintf(stderr, "[ST] Cannot read header from: %s\n", path);
        fclose(f);
        return NULL;
    }
    if (header_size > 100 * 1024 * 1024) {  /* Sanity: max 100MB header */
        fprintf(stderr, "[ST] Header too large: %llu bytes in %s\n",
                (unsigned long long)header_size, path);
        fclose(f);
        return NULL;
    }
    char *header = malloc((size_t)header_size + 1);
    if (!header) {
        fprintf(stderr, "[ST] OOM for header %zu bytes\n", (size_t)header_size);
        fclose(f);
        return NULL;
    }
    size_t nread = fread(header, 1, (size_t)header_size, f);
    fclose(f);
    if (nread != (size_t)header_size) {
        free(header);
        return NULL;
    }
    header[header_size] = '\0';
    return header;
}

/* ================================================================
 * st_parse_config: Parse model config from HuggingFace JSON files
 * ================================================================ */

/* Count tensors and deduce file layout from index JSON */
#define MAX_SHARDS 32
#define MAX_TENSORS 1024

int st_parse_config(const char *model_dir, STModelConfig *cfg) {
    memset(cfg, 0, sizeof(*cfg));

    /* --- Parse config.json --- */
    char path_cfg[1024];
    snprintf(path_cfg, sizeof(path_cfg), "%s/config.json", model_dir);
    size_t cfg_size;
    uint8_t *cfg_data = read_file(path_cfg, &cfg_size);
    if (!cfg_data) return -1;

    char *json = (char*)cfg_data;
    const char *pos;

    #define GET_INT(key, field) \
        pos = json_find_key(json, key); \
        if (pos && *pos != 'n') json_parse_int(pos, &cfg->field)

    #define GET_FLOAT(key, field) \
        pos = json_find_key(json, key); \
        if (pos) json_parse_float(pos, &cfg->field)

    GET_INT("hidden_size", dim);
    GET_INT("num_hidden_layers", n_layers);
    GET_INT("num_attention_heads", n_heads);
    GET_INT("num_key_value_heads", n_kv_heads);
    GET_INT("intermediate_size", ffn_dim);
    GET_INT("vocab_size", vocab_size);
    GET_INT("max_position_embeddings", max_seq_len);
    GET_INT("bos_token_id", bos_id);
    GET_INT("eos_token_id", eos_id);
    GET_FLOAT("rope_theta", rope_theta);
    GET_FLOAT("rms_norm_eps", norm_eps);

    /* Detect head_dim, q_norm, mrope from config */
    /* Prefer the explicit "head_dim" field: Qwen3 small models set it
     * independently of hidden_size/n_heads (e.g. Qwen3-0.6B: dim=1024,
     * heads=16, head_dim=128, so dim/n_heads=64 would be WRONG). The 8B
     * config also carries head_dim=128 == dim/n_heads, so this is a no-op
     * for the production model. Fall back to dim/n_heads when absent. */
    GET_INT("head_dim", head_dim);
    if (cfg->head_dim <= 0) {
        if (cfg->n_heads > 0 && cfg->dim > 0)
            cfg->head_dim = cfg->dim / cfg->n_heads;
        else
            cfg->head_dim = 128;
    }

    cfg->has_q_norm = (strstr(json, "q_norm") != NULL);
    cfg->has_mrope = (strstr(json, "mrope") != NULL);
    cfg->head_dim_full = cfg->head_dim;
    cfg->kv_lora_rank = 0;

    /* ---- Parse vision config ---- */
    /* Vision special tokens */
    GET_INT("vision_start_token_id", vision_start_id);
    GET_INT("vision_end_token_id", vision_end_id);
    GET_INT("image_token_id", image_token_id);
    GET_INT("video_token_id", video_token_id);

    /* Extract vision_config sub-object scope */
    {
        const char *vis_start = strstr(json, "\"vision_config\"");
        if (vis_start) {
            vis_start = strchr(vis_start, '{');
            if (vis_start) {
                /* Find matching closing brace */
                const char *vis_end = vis_start;
                int depth = 0;
                while (*vis_end) {
                    if (*vis_end == '{') depth++;
                    else if (*vis_end == '}') { depth--; if (depth == 0) break; }
                    vis_end++;
                }
                /* Parse vision_config fields within scope */
                #define VIS_GET_INT(key, field) \
                    do { const char *p2 = json_find_key_in_range(vis_start, vis_end, key); \
                         if (p2 && *p2 != 'n') json_parse_int(p2, &cfg->field); } while(0)

                VIS_GET_INT("depth", vis_depth);
                VIS_GET_INT("hidden_size", vis_hidden);
                VIS_GET_INT("num_heads", vis_heads);
                VIS_GET_INT("intermediate_size", vis_ffn);
                VIS_GET_INT("patch_size", vis_patch);
                VIS_GET_INT("temporal_patch_size", vis_temporal);
                VIS_GET_INT("spatial_merge_size", vis_merge);
                VIS_GET_INT("out_hidden_size", vis_out_dim);
                VIS_GET_INT("in_channels", vis_in_chan);
                VIS_GET_INT("num_position_embeddings", vis_max_pos);

                cfg->has_vision = (cfg->vis_depth > 0);

                /* Parse deepstack_visual_indexes array */
                cfg->vis_ds_count = 0;
                {
                    const char *ds = json_find_key_in_range(vis_start, vis_end, "deepstack_visual_indexes");
                    if (ds && *ds == '[') {
                        ds++; /* skip [ */
                        while (*ds && cfg->vis_ds_count < 4) {
                            while (*ds == ' ' || *ds == ',') ds++;
                            if (*ds == ']') break;
                            char *end2;
                            int v = (int)strtol(ds, &end2, 10);
                            if (end2 == ds) break;
                            cfg->vis_ds_idx[cfg->vis_ds_count++] = v;
                            ds = end2;
                        }
                    }
                }
                #undef VIS_GET_INT
            }
        }
    }

    /* Parse MRoPE sections from rope_scaling */
    cfg->mrope_n_sec = 0;
    if (cfg->has_mrope) {
        const char *mrope_pos = strstr(json, "\"mrope_section\"");
        if (mrope_pos) {
            mrope_pos = strchr(mrope_pos, '[');
            if (mrope_pos) {
                mrope_pos++; /* skip [ */
                while (*mrope_pos && cfg->mrope_n_sec < 4) {
                    while (*mrope_pos == ' ' || *mrope_pos == ',') mrope_pos++;
                    if (*mrope_pos == ']') break;
                    char *end2;
                    int v = (int)strtol(mrope_pos, &end2, 10);
                    if (end2 == mrope_pos) break;
                    cfg->mrope_sections[cfg->mrope_n_sec++] = v;
                    mrope_pos = end2;
                }
            }
        }
    }

    if (cfg->has_vision) {
        printf("[ST] Vision: depth=%d hidden=%d heads=%d ffn=%d patch=%d temporal=%d merge=%d out=%d\n",
               cfg->vis_depth, cfg->vis_hidden, cfg->vis_heads, cfg->vis_ffn,
               cfg->vis_patch, cfg->vis_temporal, cfg->vis_merge, cfg->vis_out_dim);
        if (cfg->vis_ds_count > 0) {
            printf("[ST]   DeepStack layers:");
            for (int i = 0; i < cfg->vis_ds_count; i++)
                printf(" %d", cfg->vis_ds_idx[i]);
            printf("\n");
        }
        if (cfg->mrope_n_sec > 0) {
            printf("[ST]   MRoPE sections:");
            for (int i = 0; i < cfg->mrope_n_sec; i++)
                printf(" %d", cfg->mrope_sections[i]);
            printf("\n");
        }
    }

    free(cfg_data);

    if (cfg->dim == 0 || cfg->n_layers == 0) {
        fprintf(stderr, "[ST] Failed to parse config.json\n");
        return -1;
    }

    printf("[ST] Config: dim=%d layers=%d heads=%d kv_heads=%d hd=%d ffn=%d vocab=%d\n",
           cfg->dim, cfg->n_layers, cfg->n_heads, cfg->n_kv_heads,
           cfg->head_dim, cfg->ffn_dim, cfg->vocab_size);
    printf("[ST]   rope_theta=%.1f norm_eps=%.1e q_norm=%d mrope=%d\n",
           cfg->rope_theta, cfg->norm_eps, cfg->has_q_norm, cfg->has_mrope);

    /* --- Parse safetensors index --- */
    char path_idx[1024];
    snprintf(path_idx, sizeof(path_idx), "%s/model.safetensors.index.json", model_dir);
    size_t idx_size;
    uint8_t *idx_data = read_file(path_idx, &idx_size);
    if (!idx_data) {
        /* Maybe single file: model.safetensors */
        snprintf(path_idx, sizeof(path_idx), "%s/model.safetensors", model_dir);
        if (st_access(path_idx, 0) != 0) {
            fprintf(stderr, "[ST] No model file found\n");
            return -1;
        }
        /* Single file mode */
        cfg->n_files = 1;
        cfg->file_paths = calloc(1, sizeof(char*));
        cfg->file_paths[0] = strdup(path_idx);
        fprintf(stderr, "[ST] Single safetensors file: %s\n", path_idx);
    } else {
        /* Multi-shard mode: parse weight_map from index */
        char *idx_json = (char*)idx_data;

        /* Count unique files in weight_map */
        const char *wm_start = strstr(idx_json, "\"weight_map\"");
        if (!wm_start) { free(idx_data); return -1; }

        /* Collect unique file names */
        char *files[MAX_SHARDS] = {0};
        int nf = 0;

        const char *p = wm_start;
        while ((p = strstr(p, "\": \"")) != NULL && nf < MAX_SHARDS) {
            p += 4; /* skip ":" " " "\"" to point at value content */
            const char *end = strchr(p, '"');
            if (!end) break;
            size_t flen = end - p;
            char *fname = malloc(flen + 1);
            memcpy(fname, p, flen);
            fname[flen] = '\0';

            /* Deduplicate */
            int found = 0;
            for (int i = 0; i < nf; i++) {
                if (strcmp(files[i], fname) == 0) { found = 1; break; }
            }
            if (!found) files[nf++] = fname;
            else free(fname);
            p = end + 1;
        }

        cfg->n_files = nf;
        cfg->file_paths = calloc(nf, sizeof(char*));
        for (int i = 0; i < nf; i++) {
            size_t plen = strlen(model_dir) + 1 + strlen(files[i]) + 1;
            cfg->file_paths[i] = malloc(plen);
            snprintf(cfg->file_paths[i], plen, "%s/%s", model_dir, files[i]);
            free(files[i]);
        }

        fprintf(stderr, "[ST] Found %d safetensors shard(s)\n", nf);
        free(idx_data);
    }

    /* --- Build tensor info: open each file, parse header, collect entries --- */
    int n_tensors_total = 0;
    cfg->tensors = calloc(MAX_TENSORS, sizeof(STTensorInfo));

    for (int fi = 0; fi < cfg->n_files; fi++) {
        char *header = read_st_header(cfg->file_paths[fi]);
        if (!header) {
            fprintf(stderr, "[ST] WARN: cannot read header[%d]: %s\n", fi, cfg->file_paths[fi]);
            continue;
        }

        size_t hdr_size = strlen(header);

        /* Parse header JSON: {"weight1": {"dtype":"BF16","shape":[4096,4096],"data_offsets":[0,33554432]}, ...} */
        const char *hp = header;
        int64_t data_offset = 8 + (int64_t)hdr_size;  /* 8-byte header_size field + JSON */

        while (*hp) {
            /* Find next key (tensor name) */
            hp = strchr(hp, '"');
            if (!hp) break;
            hp++; /* skip opening " */
            const char *name_start = hp;
            hp = strchr(hp, '"');
            if (!hp) break;
            size_t name_len = hp - name_start;
            hp++; /* skip closing " */
            hp = strchr(hp, '{'); /* find value object */
            if (!hp) break;
            hp++;

            STTensorInfo *ti = &cfg->tensors[n_tensors_total];

            /* name */
            ti->name = malloc(name_len + 1);
            memcpy(ti->name, name_start, name_len);
            ti->name[name_len] = '\0';

            /* dtype */
            const char *dp = strstr(hp, "\"dtype\"");
            if (dp) {
                dp = strchr(dp + 7, '"');
                if (dp) {
                    dp++;
                    if (strncmp(dp, "F32", 3) == 0) ti->dtype = ST_DTYPE_F32;
                    else if (strncmp(dp, "F16", 3) == 0) ti->dtype = ST_DTYPE_F16;
                    else if (strncmp(dp, "BF16", 4) == 0) ti->dtype = ST_DTYPE_BF16;
                }
            }

            /* shape */
            const char *sp = strstr(hp, "\"shape\"");
            if (sp) {
                sp = strchr(sp, '[');
                if (sp) {
                    sp++;
                    const char *sep;
                    int di = 0;
                    while (*sp != ']' && di < 8) {
                        ti->ne[di] = strtoll(sp, (char**)&sep, 10);
                        di++; sp = sep;
                        while (*sp == ',' || *sp == ' ') sp++;
                    }
                    /* Skip any extra dimensions beyond 4 */
                    while (*sp != ']') {
                        strtoll(sp, (char**)&sep, 10);
                        sp = sep;
                        while (*sp == ',' || *sp == ' ') sp++;
                    }
                    ti->n_dims = di;
                }
            }

            ti->n_elems = 1;
            for (int d = 0; d < ti->n_dims; d++) ti->n_elems *= ti->ne[d];

            /* data_offsets */
            const char *op = strstr(hp, "\"data_offsets\"");
            if (op) {
                op = strchr(op, '[');
                if (op) {
                    op++;
                    char *oe;
                    ti->file_offsets[0] = strtoll(op, &oe, 10);
                    ti->file_offsets[1] = strtoll(oe + 1, NULL, 10);
                }
            }

            ti->n_bytes = ti->file_offsets[1] - ti->file_offsets[0];

            /* Store which file this tensor is in */
            /* We encode file index in the upper bits of file_offsets[0] */
            /* Actually, we need to save file index. Let's use a separate field or encode it. */
            /* For now, store file index in n_bytes's negative range... no. */
            /* Append a special field: we'll look up by name and scan files later. */
            /* Easiest: store file index as offset bias. We'll compute absolute offset later. */
            /* For now, save the file path directly. */
            /* Actually, we already have cfg->file_paths[fi]. Just need to remember fi. */
            /* Let's encode it: file_offsets[0] is relative, add 1e18 * fi. */
            /* Simpler: add a file_index field to STTensorInfo. */
            /* But we don't have one. Let me encode it: */
            ti->file_offsets[0] += (int64_t)fi * 1000000000000000000LL; /* 1e18 * fi */

            n_tensors_total++;
            if (n_tensors_total >= MAX_TENSORS) break;

            /* Find closing "}" */
            hp = strchr(hp, '}');
            if (hp) hp++;
        }

        free(header);
    }

    cfg->n_tensors = n_tensors_total;
    fprintf(stderr, "[ST] Total %d tensors across %d file(s)\n", n_tensors_total, cfg->n_files);

    /* Detect architecture features from tensor names */
    for (int i = 0; i < n_tensors_total; i++) {
        if (strstr(cfg->tensors[i].name, "q_norm")) cfg->has_q_norm = 1;
        if (strstr(cfg->tensors[i].name, "mrope")) cfg->has_mrope = 1;
    }

    return 0;
}

static const STTensorInfo* find_tensor(const STModelConfig *cfg, const char *name) {
    for (int i = 0; i < cfg->n_tensors; i++) {
        if (strcmp(cfg->tensors[i].name, name) == 0) return &cfg->tensors[i];
    }
    return NULL;
}

/* Extract file index encoded in file_offsets[0] */
static int get_file_index(const STTensorInfo *ti) {
    return (int)(ti->file_offsets[0] / 1000000000000000000LL);
}

static int64_t get_file_offset(const STTensorInfo *ti) {
    return ti->file_offsets[0] % 1000000000000000000LL;
}

/* ================================================================
 * st_load_tensor: Load a single tensor from safetensors files
 * ================================================================ */

int st_load_tensor(const STModelConfig *cfg, const char *tensor_name,
                    float *dst, int expected_elems) {
    const STTensorInfo *ti = find_tensor(cfg, tensor_name);
    if (!ti) {
        fprintf(stderr, "[ST] Tensor not found: %s\n", tensor_name);
        return -1;
    }
    if (ti->n_elems != expected_elems && expected_elems > 0) {
        fprintf(stderr, "[ST] Size mismatch for %s: expected %d, got %lld\n",
                tensor_name, expected_elems, (long long)ti->n_elems);
    }

    int fi = get_file_index(ti);
    int64_t file_off = get_file_offset(ti);

    if (fi < 0 || fi >= cfg->n_files) {
        fprintf(stderr, "[ST] Invalid file index %d for %s\n", fi, tensor_name);
        return -1;
    }

    /* ====== mmap fast-path: map file once, reuse across tensor reads ======
     * Portable implementation (vllm_platform.h): CreateFileW+MapViewOfFile on
     * Windows, open+mmap on Linux/RK3588. */
    static st_mmap_t mc = {0};

    if (mc.fi != fi || !mc.data) {
        st_mmap_close(&mc);
        if (st_mmap_open(&mc, fi, cfg->file_paths[fi]) != 0) {
            fprintf(stderr, "[ST] mmap: Cannot open %s\n", cfg->file_paths[fi]);
            return -1;
        }
    }

    /* Use fread to get header_size first (8 bytes at file start) */
    int64_t data_offset = 0;
    {
        FILE *fh = st_fopen(cfg->file_paths[fi], "rb");
        if (!fh) return -1;
        uint64_t header_size = 0;
        int ok = (fread(&header_size, 8, 1, fh) == 1);
        fclose(fh);
        if (!ok) return -1;
        data_offset = 8 + (int64_t)header_size;
    }

    int64_t abs_off = data_offset + file_off;
    int64_t n_elems = ti->n_elems;
    const uint8_t *raw = (const uint8_t *)mc.data + abs_off;

    /* Convert and copy to destination */
    switch (ti->dtype) {
    case ST_DTYPE_F32:
        memcpy(dst, raw, (size_t)n_elems * 4);
        break;
    case ST_DTYPE_F16:
        f16_to_f32_array(dst, raw, n_elems);
        break;
    case ST_DTYPE_BF16:
        bf16_to_f32_array(dst, raw, n_elems);
        break;
    default:
        memset(dst, 0, (size_t)n_elems * 4);
        break;
    }

    return 0;
}

int st_load_tensor_slice(const STModelConfig *cfg, const char *tensor_name,
                         int64_t elem_offset, float *dst, int64_t elem_count) {
    const STTensorInfo *ti = find_tensor(cfg, tensor_name);
    if (!ti) {
        fprintf(stderr, "[ST] Tensor not found: %s\n", tensor_name);
        return -1;
    }
    if (elem_offset < 0) elem_offset = 0;
    if (elem_count < 0) elem_count = 0;
    if (elem_offset > ti->n_elems) elem_offset = ti->n_elems;
    if (elem_offset + elem_count > ti->n_elems) elem_count = ti->n_elems - elem_offset;

    int fi = get_file_index(ti);
    int64_t file_off = get_file_offset(ti);

    if (fi < 0 || fi >= cfg->n_files) {
        fprintf(stderr, "[ST] Invalid file index %d for %s\n", fi, tensor_name);
        return -1;
    }

    static st_mmap_t mc = {0};

    if (mc.fi != fi || !mc.data) {
        st_mmap_close(&mc);
        if (st_mmap_open(&mc, fi, cfg->file_paths[fi]) != 0) {
            fprintf(stderr, "[ST] mmap: Cannot open %s\n", cfg->file_paths[fi]);
            return -1;
        }
    }

    int64_t data_offset = 0;
    {
        FILE *fh = st_fopen(cfg->file_paths[fi], "rb");
        if (!fh) return -1;
        uint64_t header_size = 0;
        int ok = (fread(&header_size, 8, 1, fh) == 1);
        fclose(fh);
        if (!ok) return -1;
        data_offset = 8 + (int64_t)header_size;
    }

    int dtype_bytes = 4;
    if (ti->dtype == ST_DTYPE_F16 || ti->dtype == ST_DTYPE_BF16) dtype_bytes = 2;
    int64_t abs_off = data_offset + file_off + elem_offset * (int64_t)dtype_bytes;
    const uint8_t *raw = (const uint8_t *)mc.data + abs_off;

    switch (ti->dtype) {
    case ST_DTYPE_F32:
        memcpy(dst, raw, (size_t)elem_count * 4);
        break;
    case ST_DTYPE_F16:
        f16_to_f32_array(dst, raw, elem_count);
        break;
    case ST_DTYPE_BF16:
        bf16_to_f32_array(dst, raw, elem_count);
        break;
    default:
        memset(dst, 0, (size_t)elem_count * 4);
        break;
    }

    return 0;
}

/* ================================================================
 * st_load_layer_weights: Load weights for a specific layer
 * ================================================================ */

/* Forward declarations for Q8_0 functions (defined later) */
void f32_to_q8_0(uint8_t *q8_out, const float *f32_in, int n_elements);
void f32_to_q4_0(uint8_t *q4_out, const float *f32_in, int n_elements);
void f32_to_q2_1(uint8_t *q2_out, const float *f32_in, int n_elements);
static void quantize_row_q8_0_act(const float *__restrict x,
                                  int8_t *__restrict q, float *__restrict d, int cols);

/* Helper: load a layer tensor into weights array at the right offset */
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

        /* FFN weights: load, then quantize to Q8_0 */
        if (w->gate_weight) {
            /* F32 FFN buffers exist: load normally, quantize after */
            load_layer_tensor(cfg, w, l, "mlp.gate_proj.weight",
                               w->gate_weight, ff * dim, ff * dim);
            load_layer_tensor(cfg, w, l, "mlp.up_proj.weight",
                               w->up_weight, ff * dim, ff * dim);
            load_layer_tensor(cfg, w, l, "mlp.down_proj.weight",
                               w->down_weight, dim * ff, dim * ff);
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

/* ================================================================
 * st_weights_alloc / st_weights_free
 * ================================================================ */

void st_weights_alloc(STModelWeights *w, const STModelConfig *cfg) {
    st_weights_alloc_layers(w, cfg, -1);
}

void st_weights_alloc_layers(STModelWeights *w, const STModelConfig *cfg, int n_layers) {
    memset(w, 0, sizeof(*w));
    memcpy(&w->cfg, cfg, sizeof(STModelConfig));

    int dim = cfg->dim;
    int nl = (n_layers < 0 || n_layers > cfg->n_layers) ? cfg->n_layers : n_layers;
    int nh = cfg->n_heads;
    int nkv = cfg->n_kv_heads;
    int hd = cfg->head_dim;
    int ff = cfg->ffn_dim;
    int vc = cfg->vocab_size;
    int q_out = nh * hd;
    int k_out = nkv * hd;

    /* Estimate memory */
    size_t total = 0;
    total += (size_t)vc * dim * 4;       /* token_embed */
    total += (size_t)vc * dim * 4;       /* lm_head */
    total += (size_t)dim * 4;            /* final_norm */
    total += (size_t)nl * dim * 4;       /* attn_norm */
    total += (size_t)nl * dim * 4;       /* ffn_norm */
    total += (size_t)nl * dim * q_out * 4;  /* q_weight */
    total += (size_t)nl * dim * k_out * 4;  /* k_weight */
    total += (size_t)nl * dim * k_out * 4;  /* v_weight */
    total += (size_t)nl * q_out * dim * 4;  /* o_weight */
    total += (size_t)nl * ff * dim * 4;    /* gate_weight */
    total += (size_t)nl * ff * dim * 4;    /* up_weight */
    total += (size_t)nl * dim * ff * 4;    /* down_weight */
    if (cfg->has_q_norm) {
        total += (size_t)nl * q_out * 4;
        total += (size_t)nl * k_out * 4;
    }

    fprintf(stderr, "[ST] Estimated f32 weight memory: %.1f GB\n",
            total / (1024.0 * 1024.0 * 1024.0));
    fprintf(stderr, "[ST] Allocating weight buffers...\n");
    fflush(stderr);

    w->token_embed = calloc((size_t)vc * dim, sizeof(float));
    w->lm_head     = calloc((size_t)vc * dim, sizeof(float));
    w->final_norm  = calloc(dim, sizeof(float));
    w->attn_norm   = calloc((size_t)nl * dim, sizeof(float));
    w->ffn_norm    = calloc((size_t)nl * dim, sizeof(float));
    w->q_weight    = calloc((size_t)nl * dim * q_out, sizeof(float));
    w->k_weight    = calloc((size_t)nl * dim * k_out, sizeof(float));
    w->v_weight    = calloc((size_t)nl * dim * k_out, sizeof(float));
    w->o_weight    = calloc((size_t)nl * q_out * dim, sizeof(float));
    w->gate_weight = calloc((size_t)nl * ff * dim, sizeof(float));
    w->up_weight   = calloc((size_t)nl * ff * dim, sizeof(float));
    w->down_weight = calloc((size_t)nl * dim * ff, sizeof(float));

    /* Q8_0 compact storage for FFN weights (34 bytes per 32 elements)
     * axiom: fixedpoint_quantize_saturate */
    w->q8_gate_weight = calloc(Q8_BYTES((size_t)nl * ff * dim), 1);
    w->q8_up_weight   = calloc(Q8_BYTES((size_t)nl * ff * dim), 1);
    w->q8_down_weight = calloc(Q8_BYTES((size_t)nl * dim * ff), 1);
    w->has_q8 = 1;  /* will be set to 1 after quantization succeeds */

    if (cfg->has_q_norm) {
        w->q_norm = calloc((size_t)nl * hd, sizeof(float));
        w->k_norm = calloc((size_t)nl * hd, sizeof(float));
    }

    /* Check allocations */
    int failed = 0;
    #define CHECK(p, name) if (!(p)) { fprintf(stderr, "[ST] OOM allocating %s\n", name); failed = 1; }
    CHECK(w->token_embed, "token_embed");
    CHECK(w->lm_head,     "lm_head");
    CHECK(w->final_norm,  "final_norm");
    CHECK(w->attn_norm,   "attn_norm");
    CHECK(w->ffn_norm,    "ffn_norm");
    CHECK(w->q_weight,    "q_weight");
    CHECK(w->k_weight,    "k_weight");
    CHECK(w->v_weight,    "v_weight");
    CHECK(w->o_weight,    "o_weight");
    CHECK(w->gate_weight, "gate_weight");
    CHECK(w->up_weight,   "up_weight");
    CHECK(w->down_weight, "down_weight");
    if (cfg->has_q_norm) {
        CHECK(w->q_norm,  "q_norm");
        CHECK(w->k_norm,  "k_norm");
    }
    #undef CHECK

    if (failed) {
        st_weights_free(w);
        return;
    }

    fprintf(stderr, "[ST] Weight buffers allocated successfully.\n");

    w->is_allocated = 1;
    w->n_layers_allocated = nl;
}

/* Q8_0-only FFN variant: skips F32 FFN allocation (~22 GB savings for Qwen3-VL-8B) */
void st_weights_alloc_layers_q8ffn(STModelWeights *w, const STModelConfig *cfg, int n_layers) {
    memset(w, 0, sizeof(*w));
    memcpy(&w->cfg, cfg, sizeof(STModelConfig));

    int dim = cfg->dim;
    int nl = (n_layers < 0 || n_layers > cfg->n_layers) ? cfg->n_layers : n_layers;
    int nh = cfg->n_heads;
    int nkv = cfg->n_kv_heads;
    int hd = cfg->head_dim;
    int ff = cfg->ffn_dim;
    int vc = cfg->vocab_size;
    int q_out = nh * hd;
    int k_out = nkv * hd;
    int wm = st_wmode_effective();

    /* Estimate memory: F32 token_embed + norms + Q8_0/Q4_0 everything else */
    /* Use per-layer Q8B to avoid aggregate rounding under-allocation.
     * Q8B(nl*E) can be < sum of Q8_BYTES(l*E)+Q8_BYTES(E) for the last layer
     * due to ceiling properties: ceil((nl-1)E/32)+ceil(E/32) may > ceil(nl*E/32) */
    size_t total = 0;
    total += (size_t)vc * dim * 4;       /* token_embed (F32) */
    total += (size_t)dim * 4;            /* final_norm (F32) */
    total += (size_t)nl * dim * 4;       /* attn_norm (F32) */
    total += (size_t)nl * dim * 4;       /* ffn_norm (F32) */
    /* Q8_0-layout weights (34 B/block). Allocated when wm != 1 (dual, q8-only,
     * and q4i): wmode=q4i pre-unpacks Q4_0 to int8 into these buffers
     * (q8_buf_q4=1), so the tuned Q8 kernels consume Q4 directly with zero
     * dispatch changes. wmode=q4 keeps the compact nibble copy instead.
     * Per-layer aligned (nl * Q8B(layer) instead of Q8B(nl * layer)). */
    #define Q8B(n)  ((size_t)((n) + 31) / 32 * 34)
    if (wm != 1 && wm != 5) {
    total += Q8B((size_t)vc * dim);                     /* q8_lm_head (single tensor, not per-layer) */
    total += (size_t)nl * Q8B((size_t)dim * q_out);      /* q8_q */
    total += (size_t)nl * Q8B((size_t)dim * k_out);      /* q8_k */
    total += (size_t)nl * Q8B((size_t)dim * k_out);      /* q8_v */
    total += (size_t)nl * Q8B((size_t)q_out * dim);      /* q8_o */
    /* Q8_0 FFN */
    total += (size_t)nl * Q8B((size_t)ff * dim);         /* q8_gate */
    total += (size_t)nl * Q8B((size_t)ff * dim);         /* q8_up */
    total += (size_t)nl * Q8B((size_t)dim * ff);         /* q8_down */
    }
    #undef Q8B
    /* Q4_0 nibble weights (compact; ~halve resident DRAM vs the int8 form).
     * Allocated in dual (Q8 + Q4) and q4 (nibble-only) modes; skipped by
     * q8-only / --no-q4 and q4i (pre-unpacked) to save ~4.5 GB (see
     * estimate). q2mix allocates Q4 attention + lm_head only. */
    #define Q4B(n)  ((size_t)((n) + 31) / 32 * 18)
    if (wm == 0 || wm == 1) {            /* dual or q4-nibble */
        total += Q4B((size_t)vc * dim);                     /* q4_lm_head (single tensor) */
        total += (size_t)nl * Q4B((size_t)dim * q_out);      /* q4_q */
        total += (size_t)nl * Q4B((size_t)dim * k_out);      /* q4_k */
        total += (size_t)nl * Q4B((size_t)dim * k_out);      /* q4_v */
        total += (size_t)nl * Q4B((size_t)q_out * dim);      /* q4_o */
        total += (size_t)nl * Q4B((size_t)ff * dim);         /* q4_gate */
        total += (size_t)nl * Q4B((size_t)ff * dim);         /* q4_up */
        total += (size_t)nl * Q4B((size_t)dim * ff);         /* q4_down */
        /* P4: 8x8 prefill-only repack copy (same byte size per matrix) */
        total += (size_t)nl * Q4B((size_t)dim * q_out);      /* x8_q */
        total += (size_t)nl * Q4B((size_t)dim * k_out);      /* x8_k */
        total += (size_t)nl * Q4B((size_t)dim * k_out);      /* x8_v */
        total += (size_t)nl * Q4B((size_t)q_out * dim);      /* x8_o */
        total += (size_t)nl * Q4B((size_t)ff * dim);         /* x8_gate */
        total += (size_t)nl * Q4B((size_t)ff * dim);         /* x8_up */
        total += (size_t)nl * Q4B((size_t)dim * ff);         /* x8_down */
    } else if (wm == 5) {                /* q2mix: attention Q4 + lm Q4 */
        total += Q4B((size_t)vc * dim);                     /* q4_lm_head (single tensor) */
        total += (size_t)nl * Q4B((size_t)dim * q_out);      /* q4_q */
        total += (size_t)nl * Q4B((size_t)dim * k_out);      /* q4_k */
        total += (size_t)nl * Q4B((size_t)dim * k_out);      /* q4_v */
        total += (size_t)nl * Q4B((size_t)q_out * dim);      /* q4_o */
    }
    #undef Q4B
    /* Q2_1 2-bit FFN weights (16 B/32 elements, wmode=q2mix only). */
    #define Q2B(n)  ((size_t)((n) + 31) / 32 * 16)
    if (wm == 5) {
        total += (size_t)nl * Q2B((size_t)ff * dim);         /* q2_gate */
        total += (size_t)nl * Q2B((size_t)ff * dim);         /* q2_up */
        total += (size_t)nl * Q2B((size_t)dim * ff);         /* q2_down */
    }
    #undef Q2B
    if (cfg->has_q_norm) {
        total += (size_t)nl * q_out * 4;
        total += (size_t)nl * k_out * 4;
    }

    fprintf(stderr, "[ST-Q8] Estimated memory: %.1f GB (F32 token + %s weights)\n",
            total / (1024.0 * 1024.0 * 1024.0),
            wm == 1 ? "Q4_0 nibble" : (wm == 2 ? "Q8_0-only" :
            (wm == 3 ? "Q4_0 int8 (pre-unpacked)" :
            (wm == 4 ? "G256 Q8_0-layout" :
            (wm == 5 ? "Q4_0 attention + Q2_1 FFN (q2mix)" : "Q8_0 + Q4_0 nibble")))));
    fprintf(stderr, "[ST-Q8] Allocating weight buffers...\n");
    fflush(stderr);

    /* Allocate F32 attention + Q8_0 FFN (NO F32 FFN)
     * When Q8_0 attention is allocated, skip F32 attention & lm_head to save ~8 GB
     * peak memory. Loading uses per-layer temp buffers to quantize directly. */
    w->token_embed = calloc((size_t)vc * dim, sizeof(float));
    w->lm_head     = NULL;  /* Q8_0-only: temp-loaded and quantized */
    w->final_norm  = calloc(dim, sizeof(float));
    w->attn_norm   = calloc((size_t)nl * dim, sizeof(float));
    w->ffn_norm    = calloc((size_t)nl * dim, sizeof(float));
    /* F32 attention: SKIPPED — temp-loaded per layer, quantized to Q8_0, freed */
    w->q_weight    = NULL;
    w->k_weight    = NULL;
    w->v_weight    = NULL;
    w->o_weight    = NULL;
    /* F32 FFN: SKIPPED — gate/up/down_weight = NULL */
    w->gate_weight = NULL;
    w->up_weight   = NULL;
    w->down_weight = NULL;

    /* Q8_0-layout compact weights (FFN + LM head + Attention projections).
     * Allocated when wm != 1 (dual, q8-only, q4i): wmode=q4i stores
     * pre-unpacked Q4 int8 here (q8_buf_q4=1) so the tuned Q8 kernels run on
     * Q4 data unchanged; wmode=q4 keeps the compact nibble copy instead.
     * Per-layer aligned allocation: nl * Q8B(layer_elems) instead of
     * Q8B(nl * layer_elems) to prevent buffer overflow from ceiling rounding
     * gap in last layer. */
    #define Q8B(n)  ((size_t)((n) + 31) / 32 * 34)
    if (wm != 1 && wm != 5) {
    w->q8_gate_weight = calloc((size_t)nl * Q8B((size_t)ff * dim), 1);
    w->q8_up_weight   = calloc((size_t)nl * Q8B((size_t)ff * dim), 1);
    w->q8_down_weight = calloc((size_t)nl * Q8B((size_t)dim * ff), 1);
    w->q8_lm_weight   = calloc(Q8B((size_t)vc * dim), 1);
    w->q8_q_weight    = calloc((size_t)nl * Q8B((size_t)dim * q_out), 1);
    w->q8_k_weight    = calloc((size_t)nl * Q8B((size_t)dim * k_out), 1);
    w->q8_v_weight    = calloc((size_t)nl * Q8B((size_t)dim * k_out), 1);
    w->q8_o_weight    = calloc((size_t)nl * Q8B((size_t)q_out * dim), 1);
    }
    #undef Q8B
    w->has_q8 = (wm != 1 && wm != 5);
    w->q8_buf_q4 = (wm == 3);   /* q4i: q8_* buffers hold pre-unpacked Q4 */

    /* Q4_0 nibble compact weights (per-layer aligned, same ceiling-safety).
     * Dual (Q8 + Q4) and q4 (nibble-only) modes; q4i uses pre-unpacked int8
     * in the q8_* buffers and q8-only / --no-q4 skips Q4 entirely (see
     * estimate). q2mix allocates Q4 attention + lm_head only. */
    #define Q4B(n)  ((size_t)((n) + 31) / 32 * 18)
    if (wm == 0 || wm == 1) {            /* dual or q4-nibble */
        w->q4_gate_weight = calloc((size_t)nl * Q4B((size_t)ff * dim), 1);
        w->q4_up_weight   = calloc((size_t)nl * Q4B((size_t)ff * dim), 1);
        w->q4_down_weight = calloc((size_t)nl * Q4B((size_t)dim * ff), 1);
        w->q4_lm_weight   = calloc(Q4B((size_t)vc * dim), 1);
        w->q4_q_weight    = calloc((size_t)nl * Q4B((size_t)dim * q_out), 1);
        w->q4_k_weight    = calloc((size_t)nl * Q4B((size_t)dim * k_out), 1);
        w->q4_v_weight    = calloc((size_t)nl * Q4B((size_t)dim * k_out), 1);
        w->q4_o_weight    = calloc((size_t)nl * Q4B((size_t)q_out * dim), 1);
        /* P4: 16x8 repack copies (same byte size; filled after layers load).
         * Only in pure-Q4 (wm==1) and only when enabled via VLLM_ENABLE_8X8L=1
         * — 16x8 tile (16 rows x 256 cols, 36 cache-line aligned) replaces the
         * failed 8x8 kernel (register spills + 144B prefetcher-hostile reads).
         * dual (wm==0) 不分配。 */
        if (wm == 1) {
            const char *e8 = getenv("VLLM_ENABLE_8X8L");
            if (e8 && e8[0] && e8[0] != '0') {
                w->x8_gate_weight = calloc((size_t)nl * Q4B((size_t)ff * dim), 1);
                w->x8_up_weight   = calloc((size_t)nl * Q4B((size_t)ff * dim), 1);
                w->x8_down_weight = calloc((size_t)nl * Q4B((size_t)dim * ff), 1);
                w->x8_q_weight    = calloc((size_t)nl * Q4B((size_t)dim * q_out), 1);
                w->x8_k_weight    = calloc((size_t)nl * Q4B((size_t)dim * k_out), 1);
                w->x8_v_weight    = calloc((size_t)nl * Q4B((size_t)dim * k_out), 1);
                w->x8_o_weight    = calloc((size_t)nl * Q4B((size_t)q_out * dim), 1);
                w->has_x8 = (w->x8_gate_weight && w->x8_up_weight && w->x8_down_weight &&
                             w->x8_q_weight && w->x8_k_weight && w->x8_v_weight &&
                             w->x8_o_weight) ? 1 : 0;
            }
        }
    } else if (wm == 5) {                /* q2mix: Q4 attention + lm_head + tail FFN */
        int tail = g_q2mix_tail;
        if (tail > nl) tail = nl;
        w->q4_lm_weight   = calloc(Q4B((size_t)vc * dim), 1);
        w->q4_q_weight    = calloc((size_t)nl * Q4B((size_t)dim * q_out), 1);
        w->q4_k_weight    = calloc((size_t)nl * Q4B((size_t)dim * k_out), 1);
        w->q4_v_weight    = calloc((size_t)nl * Q4B((size_t)dim * k_out), 1);
        w->q4_o_weight    = calloc((size_t)nl * Q4B((size_t)q_out * dim), 1);
        /* Layered precision: the last `tail` layers keep Q4_0 FFN (2-bit error
         * hurts most near the output; ds4/DwarfStar). Earlier layers are Q2. */
        if (tail > 0) {
            w->q4_gate_weight = calloc((size_t)tail * Q4B((size_t)ff * dim), 1);
            w->q4_up_weight   = calloc((size_t)tail * Q4B((size_t)ff * dim), 1);
            w->q4_down_weight = calloc((size_t)tail * Q4B((size_t)dim * ff), 1);
        }
    }
    #undef Q4B
    w->has_q4 = (wm == 0 || wm == 1 || wm == 5);

    /* Q2_1 2-bit FFN weights (16 B/32 elements, wmode=q2mix only, first
     * (nl - tail) layers). */
    #define Q2B(n)  ((size_t)((n) + 31) / 32 * 16)
    if (wm == 5) {
        int tail = g_q2mix_tail;
        if (tail > nl) tail = nl;
        int q2_layers = nl - tail;
        if (q2_layers > 0) {
            w->q2_gate_weight = calloc((size_t)q2_layers * Q2B((size_t)ff * dim), 1);
            w->q2_up_weight   = calloc((size_t)q2_layers * Q2B((size_t)ff * dim), 1);
            w->q2_down_weight = calloc((size_t)q2_layers * Q2B((size_t)dim * ff), 1);
        }
    }
    #undef Q2B
    w->has_q2 = (wm == 5);

    if (cfg->has_q_norm) {
        w->q_norm = calloc((size_t)nl * hd, sizeof(float));
        w->k_norm = calloc((size_t)nl * hd, sizeof(float));
    }

    /* Check allocations */
    int failed = 0;
    #define CHECK(p, name) if (!(p)) { fprintf(stderr, "[ST-Q8] OOM allocating %s\n", name); failed = 1; }
    CHECK(w->token_embed, "token_embed");
    /* F32 lm_head: SKIPPED — temp-loaded per layer */
    CHECK(w->final_norm,  "final_norm");
    CHECK(w->attn_norm,   "attn_norm");
    CHECK(w->ffn_norm,    "ffn_norm");
    /* F32 Q/K/V/O: SKIPPED — temp-loaded per layer */
    if (wm != 1 && wm != 5) {
        CHECK(w->q8_gate_weight, "q8_gate");
        CHECK(w->q8_up_weight,   "q8_up");
        CHECK(w->q8_down_weight, "q8_down");
        CHECK(w->q8_lm_weight,   "q8_lm");
        CHECK(w->q8_q_weight,    "q8_q");
        CHECK(w->q8_k_weight,    "q8_k");
        CHECK(w->q8_v_weight,    "q8_v");
        CHECK(w->q8_o_weight,    "q8_o");
    }
    if (wm == 0 || wm == 1) {
        CHECK(w->q4_gate_weight, "q4_gate");
        CHECK(w->q4_up_weight,   "q4_up");
        CHECK(w->q4_down_weight, "q4_down");
        CHECK(w->q4_lm_weight,   "q4_lm");
        CHECK(w->q4_q_weight,    "q4_q");
        CHECK(w->q4_k_weight,    "q4_k");
        CHECK(w->q4_v_weight,    "q4_v");
        CHECK(w->q4_o_weight,    "q4_o");
    } else if (wm == 5) {
        int chk_tail = g_q2mix_tail;
        if (chk_tail > nl) chk_tail = nl;
        CHECK(w->q4_lm_weight,   "q4_lm");
        CHECK(w->q4_q_weight,    "q4_q");
        CHECK(w->q4_k_weight,    "q4_k");
        CHECK(w->q4_v_weight,    "q4_v");
        CHECK(w->q4_o_weight,    "q4_o");
        if (nl - chk_tail > 0) {
            CHECK(w->q2_gate_weight, "q2_gate");
            CHECK(w->q2_up_weight,   "q2_up");
            CHECK(w->q2_down_weight, "q2_down");
        }
        if (chk_tail > 0) {
            CHECK(w->q4_gate_weight, "q4_gate(tail)");
            CHECK(w->q4_up_weight,   "q4_up(tail)");
            CHECK(w->q4_down_weight, "q4_down(tail)");
        }
    }
    if (cfg->has_q_norm) {
        CHECK(w->q_norm,  "q_norm");
        CHECK(w->k_norm,  "k_norm");
    }
    #undef CHECK

    if (failed) {
        st_weights_free(w);
        return;
    }

    fprintf(stderr, "[ST-Q8] Weight buffers allocated successfully.\n");

    /* M4: gate + pre-allocate for the 4x4 repack.
     * Geometric gate: EVERY Q4 matrix (layer + lm_head) must be 4-row and
     * 32-col aligned, or repack is disabled globally BEFORE any matrix is
     * touched - the kernels must never see a mixed legacy/4x4 layout (a
     * fused QKV/GU/Down kernel reads all its matrices in the same mode). */
    {
        if (!(vc % 4 == 0 && q_out % 4 == 0 && k_out % 4 == 0 &&
              dim % 32 == 0 && ff % 32 == 0))
            g_st_q4_repack = 0;
        /* Pre-allocate the repack scratch (max 4-row group width). */
        size_t q4r_max = 4 * (size_t)(((ff > dim ? ff : dim) + 31) / 32) * 18;
        if (q4r_max > g_q4r_scratch_cap) {
            uint8_t *np = (uint8_t *)realloc(g_q4r_scratch, q4r_max);
            if (!np) {
                fprintf(stderr, "[ST] M4: repack scratch OOM (%zu B); 4x4 repack disabled\n",
                        q4r_max);
                g_st_q4_repack = 0;
            } else {
                g_q4r_scratch = np;
                g_q4r_scratch_cap = q4r_max;
            }
        }
    }

    /* M4e: geometric gate for the Q8_0 4x4 repack (llama block_q8_0x4).
     * Same rule as Q4: EVERY Q8 matrix must be 4-row / 32-col aligned or the
     * repack is disabled globally BEFORE any matrix is touched - the kernels
     * must never see a mixed legacy/4x4 layout. Q8 matrix shapes:
     * q/k/v/o/gate/up = (rows%4, cols%32), down = (dim%4, ff%32).
     * wm==3 (q4i) stores pre-unpacked Q4 in the q8 buffers - repack never
     * applies. wm==4 (g256) KEEPS the repack so the CPU prefill stays on the
     * fast 4x4 kernels; the NPU DIRECT gw_build unpacks block_q8_0x4 back to
     * the group layout (bit-identical gather, M5b), so NPU inference is
     * unaffected. The earlier M5a blanket `wm==4/3 -> repack=0` fixed the NPU
     * NaN but silently degraded CPU-g256 prefill 19.6x (legacy non-vdot
     * path, ~78s vs ~4s) - the M5b unpack restores both. */
    if (wm == 3) {
        g_st_q8_repack = 0;   /* q4i: pre-unpacked Q4 - N/A */
    } else if (g_st_q8_repack) {
        const char *dq = getenv("VLLM_DISABLE_Q8_REPACK");
        if (dq && dq[0] && dq[0] != '0')
            g_st_q8_repack = 0;
        if (!g_st_q8_repack) { /* already disabled */ }
        else if (q8_8x8_enabled()) {
            if (!(q_out % 8 == 0 && k_out % 8 == 0 && ff % 8 == 0 && dim % 8 == 0 &&
                  dim % 32 == 0 && ff % 32 == 0))
                g_st_q8_repack = 0;
        } else if (!(q_out % 4 == 0 && k_out % 4 == 0 && ff % 4 == 0 && dim % 4 == 0 &&
                     dim % 32 == 0 && ff % 32 == 0))
            g_st_q8_repack = 0;
        /* M5b: st_npu_gw_build now handles the repacked layout, so the old
         * VLLM_NPU_INFER-disable rule is removed (it forced the slow legacy
         * CPU path whenever NPU was on). */
        /* Pre-allocate the tile repack scratch (one 8x8 group or one 4x4
         * group, max row width dim or ff) so the load-time repack can never
         * fail part-way - the kernels must see ONE layout. Shares
         * g_q8r_scratch with the M4d activation repack (load-time vs
         * inference-time, never overlap). */
        if (g_st_q8_repack) {
            size_t q8r_max;
            if (q8_8x8_enabled())
                q8r_max = 8 * (size_t)(((ff > dim ? ff : dim) + 31) / 32) * 34;
            else
                q8r_max = 4 * (size_t)(((ff > dim ? ff : dim) + 31) / 32) * 34;
            if (q8r_max > g_q8r_scratch_cap) {
                uint8_t *np = (uint8_t *)realloc(g_q8r_scratch, q8r_max);
                if (!np) {
                    fprintf(stderr, "[ST] M4e: Q8 repack scratch OOM (%zu B); repack disabled\n",
                            q8r_max);
                    g_st_q8_repack = 0;
                } else {
                    g_q8r_scratch = np;
                    g_q8r_scratch_cap = q8r_max;
                }
            }
        }
    }
    w->is_allocated = 1;
    w->n_layers_allocated = nl;
}

void st_weights_free(STModelWeights *w) {
    if (!w->is_allocated) return;
    /* VQF mmap-backed weights: all pointers alias one read-only mapping. */
    if (w->vqf_map) {
        if (w->vqf_map && w->vqf_map_len)
            munmap(w->vqf_map, w->vqf_map_len);
        w->vqf_map = NULL; w->vqf_map_len = 0;
        /* VQF vision 结构体（vqf_load calloc；其内部指针 alias vqf_map，
         * munmap 后全部失效，仅释放结构体本身） */
        if (w->vision) { free(w->vision); w->vision = NULL; }
        memset(w, 0, sizeof(*w));
        return;
    }
    fprintf(stderr, "[FR] t_emb=%p lm=%p fn=%p an=%p fn2=%p\n",
            (void *)w->token_embed, (void *)w->lm_head, (void *)w->final_norm,
            (void *)w->attn_norm, (void *)w->ffn_norm);
    fprintf(stderr, "[FR] q=%p k=%p v=%p o=%p qn=%p kn=%p\n",
            (void *)w->q_weight, (void *)w->k_weight, (void *)w->v_weight,
            (void *)w->o_weight, (void *)w->q_norm, (void *)w->k_norm);
    fprintf(stderr, "[FR] g=%p u=%p d=%p q8g=%p q8u=%p q8d=%p q8lm=%p\n",
            (void *)w->gate_weight, (void *)w->up_weight, (void *)w->down_weight,
            (void *)w->q8_gate_weight, (void *)w->q8_up_weight,
            (void *)w->q8_down_weight, (void *)w->q8_lm_weight);
    fprintf(stderr, "[FR] q8q=%p q8k=%p q8v=%p q8o=%p\n",
            (void *)w->q8_q_weight, (void *)w->q8_k_weight,
            (void *)w->q8_v_weight, (void *)w->q8_o_weight);
    fprintf(stderr, "[FR] q4lm=%p q4q=%p q4k=%p q4v=%p q4o=%p q4g=%p q4u=%p q4d=%p\n",
            (void *)w->q4_lm_weight, (void *)w->q4_q_weight, (void *)w->q4_k_weight,
            (void *)w->q4_v_weight, (void *)w->q4_o_weight, (void *)w->q4_gate_weight,
            (void *)w->q4_up_weight, (void *)w->q4_down_weight);
    fflush(stderr);
    free(w->token_embed);  fprintf(stderr, "[FR] free te OK\n");  fflush(stderr);
    free(w->lm_head);
    fprintf(stderr, "[FR] free fn\n"); fflush(stderr);
    free(w->final_norm);
    fprintf(stderr, "[FR] free an\n"); fflush(stderr);
    free(w->attn_norm);
    fprintf(stderr, "[FR] free fn2\n"); fflush(stderr);
    free(w->ffn_norm);
    fprintf(stderr, "[FR] free q/k/v/o\n"); fflush(stderr);
    free(w->q_weight);
    free(w->k_weight);
    free(w->v_weight);
    free(w->o_weight);
    fprintf(stderr, "[FR] free qn/kn\n"); fflush(stderr);
    free(w->q_norm);
    free(w->k_norm);
    free(w->gate_weight);
    free(w->up_weight);
    free(w->down_weight);
    fprintf(stderr, "[FR] q8g\n"); fflush(stderr);
    free(w->q8_gate_weight);
    free(w->q8_up_weight);
    free(w->q8_down_weight);
    fprintf(stderr, "[FR] q8lm\n"); fflush(stderr);
    free(w->q8_lm_weight);
    fprintf(stderr, "[FR] q8q/k/v/o\n"); fflush(stderr);
    free(w->q8_q_weight);
    free(w->q8_k_weight);
    free(w->q8_v_weight);
    free(w->q8_o_weight);
    fprintf(stderr, "[FR] q4...\n"); fflush(stderr);
    free(w->q4_gate_weight);
    free(w->q4_up_weight);
    free(w->q4_down_weight);
    free(w->q4_lm_weight);
    free(w->q4_q_weight);
    free(w->q4_k_weight);
    free(w->q4_v_weight);
    free(w->q4_o_weight);
    free(w->x8_gate_weight);
    free(w->x8_up_weight);
    free(w->x8_down_weight);
    free(w->x8_q_weight);
    free(w->x8_k_weight);
    free(w->x8_v_weight);
    free(w->x8_o_weight);
    free(w->q2_gate_weight);
    free(w->q2_up_weight);
    free(w->q2_down_weight);
    fprintf(stderr, "[FR] all free OK\n"); fflush(stderr);
    memset(w, 0, sizeof(*w));
}

void st_config_free(STModelConfig *cfg) {
    for (int i = 0; i < cfg->n_tensors; i++) free(cfg->tensors[i].name);
    free(cfg->tensors);
    for (int i = 0; i < cfg->n_files; i++) free(cfg->file_paths[i]);
    free(cfg->file_paths);
    memset(cfg, 0, sizeof(*cfg));
}

/* ================================================================
 * Qwen3-VL Forward Pass
 * ================================================================ */

/* Inline helpers */

static inline float silu_f(float x) {
    return x / (1.0f + expf(-x));
}

/* ================================================================
 * AVX2 Q8_0 Helper Functions (axiom: fixedpoint_quantize_saturate)
 * ================================================================ */

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


/* f32 → f16 (simple round-to-nearest-even) */
/* f32 → f16 (simple round-to-nearest-even) */
static inline uint16_t f32_to_f16_bits(float x) {
    uint32_t u; memcpy(&u, &x, 4);
    uint32_t sign = (u >> 16) & 0x8000;
    int32_t  exp  = (int32_t)((u >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (u >> 13) & 0x3FF;
    if (exp <= 0)      return (uint16_t)(sign | 0);
    if (exp >= 0x1F)   return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | (exp << 10) | (mant & 0x3FF));
}

/* VQF embedding lookup：F16 存储时即时转 F32（dim = hidden 维） */
static inline float st_emb_get(const STModelWeights *w, size_t row, size_t col, int dim) {
    if (w->emb_f16) {
        const uint16_t *e16 = (const uint16_t *)w->token_embed;
        uint32_t bits = f16_to_f32_bits(e16[row * (size_t)dim + col]);
        float v; memcpy(&v, &bits, 4); return v;
    }
    return w->token_embed[row * (size_t)dim + col];
}

/* SiLU activation: x * sigmoid(x).
 * Numerically stable: clamped to [-88, 88] to prevent expf overflow.
 * SiLU(x) ≈ x for x > 20, and ≈ 0 for x < -20. Clamp output to [0, 100]. */
static float fast_silu(float x) {
    if (x <= 0.0f) {
        if (x < -20.0f) return 0.0f;
        float ex = expf(x);  /* x is negative, ex in (0, 1] */
        return x * ex / (1.0f + ex);
    } else {
        if (x > 20.0f) return x;  /* sigmoid(20) ≈ 1.0 */
        return x / (1.0f + expf(-x));
    }
}

/* ================================================================
 * Q8_0 Quantization: f32 weights → compact Q8_0 format
 * (axiom: fixedpoint_quantize_saturate + dynamic_scale_layer_collapse)
 *
 * Each block of 32 elements → 34 bytes: { f16 scale; int8 qs[32]; }
 *   scale = max_abs(block) / 127
 *   qs[i] = round(w[i] / scale),  clamped to [-127, 127]
 * ================================================================ */

void f32_to_q8_0(uint8_t *q8_out, const float *f32_in, int n_elements) {
    int block_size = 32;
    int n_blocks = n_elements / block_size;

    for (int b = 0; b < n_blocks; b++) {
        /* Find block max-abs for scale */
        float max_abs = 1e-10f;
        for (int i = 0; i < block_size; i++) {
            float v = f32_in[b * block_size + i];
            float av = fabsf(v);
            if (av > max_abs) max_abs = av;
        }
        float scale = max_abs / 127.0f;
        if (scale < 1e-10f) scale = 1e-10f;

        /* Store f16 scale */
        uint16_t h = f32_to_f16_bits(scale);
        memcpy(q8_out + (size_t)b * 34, &h, 2);

        /* Store int8 quants */
        int8_t *qs = (int8_t *)(q8_out + (size_t)b * 34 + 2);
        for (int i = 0; i < block_size; i++) {
            float v = f32_in[b * block_size + i];
            int q = (int)(v / scale + 0.5f);
            if (q > 127) q = 127;
            if (q < -127) q = -127;
            qs[i] = (int8_t)q;
        }
    }
}

/* ================================================================
 * Q4_0 Quantization: f32 weights → compact Q4_0 format
 * (axiom: fixedpoint_quantize_saturate, B=4 path; int4_ready)
 *
 * Each block of 32 elements → 18 bytes: { f16 scale d; uint8 nibbles qs[16]; }
 *   d = max_signed / -8   (matches llama.cpp q4_0 asymmetric range [-8,+7])
 *   qs[j] = low nibble  = round(elem[j]    / d) clamped to [0,15]
 *           high nibble = round(elem[16+j] / d) clamped to [0,15]
 * Dequantized value = (nibble - 8) * d
 * ================================================================ */
void f32_to_q4_0(uint8_t *q4_out, const float *f32_in, int n_elements) {
    int block_size = 32;
    int n_blocks = n_elements / block_size;

    for (int b = 0; b < n_blocks; b++) {
        const float *xb = f32_in + b * block_size;
        float amax = 0.0f;
        float maxv = 0.0f;
        for (int i = 0; i < block_size; i++) {
            float v = xb[i];
            float av = fabsf(v);
            if (av > amax) { amax = av; maxv = v; }
        }
        float d = maxv / -8.0f;
        if (d == 0.0f) d = 1.0f;   /* guard: zero block (should not happen for real weights) */
        float id = 1.0f / d;

        uint16_t h = f32_to_f16_bits(d);
        memcpy(q4_out + (size_t)b * 18, &h, 2);

        uint8_t *qs = q4_out + (size_t)b * 18 + 2;
        for (int j = 0; j < 16; j++) {
            float x0 = xb[j] * id;
            float x1 = xb[16 + j] * id;
            int xi0 = (int)(x0 + 8.5f);
            int xi1 = (int)(x1 + 8.5f);
            if (xi0 > 15) xi0 = 15;
            if (xi0 < 0)  xi0 = 0;
            if (xi1 > 15) xi1 = 15;
            if (xi1 < 0)  xi1 = 0;
            qs[j] = (uint8_t)(xi0 | (xi1 << 4));
        }
    }
}

/* ================================================================
 * Q4_0 pre-unpacked-to-int8: F32 → { f16 scale d; int8 qs[32]; } (34 B/block)
 *
 * Bit-identical to the nibble Q4_0 path: same d (maxv/-8), same f16 scale
 * bits, same (int)(x + 8.5f) rounding clamped to [0,15]; the only difference
 * is qs[i] is stored already expanded as int8 (xi - 8) instead of two
 * 4-bit nibbles. The output layout is byte-for-byte the Q8_0 layout, so the
 * Q8 kernels can consume it directly with zero dispatch changes.
 * ================================================================ */
void f32_to_q4i8(uint8_t *q8_out, const float *f32_in, int n_elements) {
    int block_size = 32;
    int n_blocks = n_elements / block_size;

    for (int b = 0; b < n_blocks; b++) {
        const float *xb = f32_in + b * block_size;
        float amax = 0.0f;
        float maxv = 0.0f;
        for (int i = 0; i < block_size; i++) {
            float v = xb[i];
            float av = fabsf(v);
            if (av > amax) { amax = av; maxv = v; }
        }
        float d = maxv / -8.0f;
        if (d == 0.0f) d = 1.0f;   /* guard: zero block (mirrors f32_to_q4_0) */
        float id = 1.0f / d;

        uint16_t h = f32_to_f16_bits(d);
        memcpy(q8_out + (size_t)b * 34, &h, 2);

        int8_t *qs = (int8_t *)(q8_out + (size_t)b * 34 + 2);
        for (int j = 0; j < 32; j++) {
            float x0 = xb[j] * id;
            int xi = (int)(x0 + 8.5f);   /* same rounding as nibble path */
            if (xi > 15) xi = 15;
            if (xi < 0)  xi = 0;
            qs[j] = (int8_t)(xi - 8);    /* ∈ [-8,+7] */
        }
    }
}

/* ================================================================
 * Q2_1 Quantization: f32 weights → compact Q2_1 format
 * (axiom: blas_precision_efficiency_tradeoff, 2-bit weight compression)
 *
 * Each block of 32 elements → 16 bytes:
 *   [0:2) f16 d0   [2:4) f16 m0   [4:6) f16 d1   [6:8) f16 m1   [8:16) qs[8]
 * Sub-block h (h=0,1) covers elements 16h..16h+15 with its own
 *   d_h = (max - min) / 3,  m_h = min
 *   code q_i = clamp(round((x_i - m_h) / d_h), 0, 3)  (2 bits, qs packed)
 * Dequantized value = m_h + d_h * q_i
 * ================================================================ */
void f32_to_q2_1(uint8_t *q2_out, const float *f32_in, int n_elements) {
    int n_blocks = n_elements / 32;
    for (int b = 0; b < n_blocks; b++) {
        const float *xb = f32_in + (size_t)b * 32;
        uint8_t *out = q2_out + (size_t)b * 16;
        memset(out, 0, 16);   /* clear codes; we OR into the packed bytes */
        for (int h = 0; h < 2; h++) {
            const float *xs = xb + h * 16;
            float minv = xs[0], maxv = xs[0];
            for (int i = 1; i < 16; i++) {
                if (xs[i] < minv) minv = xs[i];
                if (xs[i] > maxv) maxv = xs[i];
            }
            float d = (maxv - minv) / 3.0f;
            if (!(d > 0.0f)) d = 1.0f;   /* guard: flat/zero block → q=0 */
            float m = minv;
            float id = 1.0f / d;
            uint16_t dh = f32_to_f16_bits(d);
            uint16_t mh = f32_to_f16_bits(m);
            memcpy(out + h * 4, &dh, 2);
            memcpy(out + h * 4 + 2, &mh, 2);
            uint8_t *qs = out + 8 + h * 4;
            for (int i = 0; i < 16; i++) {
                int q = (int)((xs[i] - m) * id + 0.5f);
                if (q > 3) q = 3;
                if (q < 0) q = 0;
                qs[i / 4] |= (uint8_t)(q << (2 * (i % 4)));
            }
        }
    }
}

/* ================================================================
 * Q2_1 Matvec Kernels (mixed-precision decode: FFN at 2-bit)
 * (axiom: blas_precision_efficiency_tradeoff)
 *
 * Dot with Q8_0 activation (a_i = xq_i * d_a), 32-elem block:
 *   Σ w_i a_i = d_a * [ (m0·Σxq0 + d0·Σ(q·xq)_0)
 *                       + (m1·Σxq1 + d1·Σ(q·xq)_1) ]
 * 0.5 B/element vs Q4_0 1.0625 B/element → weight DRAM ÷2.1.
 * ================================================================ */

/* f16 at block offset p → f32. */
static inline float q2_f16(const uint8_t *p) {
    uint16_t h; memcpy(&h, p, 2);
    uint32_t fb = f16_to_f32_bits(h);
    float f; memcpy(&f, &fb, 4);
    return f;
}

/* 单平台（RK3588）：Q2_1 × Q8_0 块点积走标量实现（32 元素块）。 */
static inline float q2_q8_block_dot(const uint8_t *blk, const int8_t *xq,
                                    float a_scale) {
    float d0 = q2_f16(blk),      m0 = q2_f16(blk + 2);
    float d1 = q2_f16(blk + 4),  m1 = q2_f16(blk + 6);
    const uint8_t *qs = blk + 8;
    int32_t S0 = 0, Q0 = 0, S1 = 0, Q1 = 0;
    for (int i = 0; i < 16; i++) {
        int q0 = (qs[i / 4] >> (2 * (i % 4))) & 3;
        S0 += xq[i];
        Q0 += q0 * xq[i];
        int q1 = (qs[4 + i / 4] >> (2 * (i % 4))) & 3;
        S1 += xq[16 + i];
        Q1 += q1 * xq[16 + i];
    }
    return a_scale * ((m0 * (float)S0 + d0 * (float)Q0) +
                      (m1 * (float)S1 + d1 * (float)Q1));
}

/* ---- Fused gate+up Q2_1 matvec (single token) ---- */
typedef struct {
    float *gate_out, *up_out; const uint8_t *q2_gate, *q2_up;
    const int8_t *xq; const float *xd;
    int n_blocks, row_stride, rows;
} vllm_dyn_matvec_q2_gu_x86_ctx;

static void vllm_dyn_matvec_q2_gu_x86_worker(void *ctx_, int r) {
    vllm_dyn_matvec_q2_gu_x86_ctx *c = ctx_;
    const uint8_t *__restrict pg = c->q2_gate + (size_t)r * c->row_stride;
    const uint8_t *__restrict pu = c->q2_up   + (size_t)r * c->row_stride;
    float ag0 = 0.f, ag1 = 0.f, ag2 = 0.f, ag3 = 0.f;
    float au0 = 0.f, au1 = 0.f, au2 = 0.f, au3 = 0.f;
    int b;
    for (b = 0; b + 3 < c->n_blocks; b += 4) {
        ag0 += q2_q8_block_dot(pg + (size_t)(b + 0) * 16, c->xq + (size_t)(b + 0) * 32, c->xd[b + 0]);
        ag1 += q2_q8_block_dot(pg + (size_t)(b + 1) * 16, c->xq + (size_t)(b + 1) * 32, c->xd[b + 1]);
        ag2 += q2_q8_block_dot(pg + (size_t)(b + 2) * 16, c->xq + (size_t)(b + 2) * 32, c->xd[b + 2]);
        ag3 += q2_q8_block_dot(pg + (size_t)(b + 3) * 16, c->xq + (size_t)(b + 3) * 32, c->xd[b + 3]);
        au0 += q2_q8_block_dot(pu + (size_t)(b + 0) * 16, c->xq + (size_t)(b + 0) * 32, c->xd[b + 0]);
        au1 += q2_q8_block_dot(pu + (size_t)(b + 1) * 16, c->xq + (size_t)(b + 1) * 32, c->xd[b + 1]);
        au2 += q2_q8_block_dot(pu + (size_t)(b + 2) * 16, c->xq + (size_t)(b + 2) * 32, c->xd[b + 2]);
        au3 += q2_q8_block_dot(pu + (size_t)(b + 3) * 16, c->xq + (size_t)(b + 3) * 32, c->xd[b + 3]);
        pg += 64; pu += 64;
    }
    for (; b < c->n_blocks; b++) {
        ag0 += q2_q8_block_dot(pg, c->xq + (size_t)b * 32, c->xd[b]);
        au0 += q2_q8_block_dot(pu, c->xq + (size_t)b * 32, c->xd[b]);
        pg += 16; pu += 16;
    }
    c->gate_out[r] = (ag0 + ag1) + (ag2 + ag3);
    c->up_out[r]   = (au0 + au1) + (au2 + au3);
}

static void dyn_matvec_q2_q8_fused_gate_up(
    float *__restrict gate_out, float *__restrict up_out,
    const uint8_t *__restrict q2_gate, const uint8_t *__restrict q2_up,
    const float *__restrict x, int rows, int cols)
{
    int n_blocks = cols / 32;
    int row_stride = n_blocks * 16;

    int8_t *xq = (int8_t *)xq_alloc_canary((size_t)cols);
    float *xd = (float *)xq_alloc_canary((size_t)n_blocks * sizeof(float));
    if (!xq || !xd) {
        xq_free_canary(xq, (size_t)cols, "q2_gu:xq");
        xq_free_canary(xd, (size_t)n_blocks * sizeof(float), "q2_gu:xd");
        return;
    }
    quantize_row_q8_0_act(x, xq, xd, cols);

    vllm_dyn_matvec_q2_gu_x86_ctx vc = { gate_out, up_out, q2_gate, q2_up, xq, xd, n_blocks, row_stride, rows };
    vllm_tp_parfor(0, rows, vllm_dyn_matvec_q2_gu_x86_worker, &vc);
    xq_free_canary(xq, (size_t)cols, "q2_gu:xq");
    xq_free_canary(xd, (size_t)n_blocks * sizeof(float), "q2_gu:xd");
}

/* ---- Fused down + FFN residual Q2_1 matvec (single token) ---- */
typedef struct {
    float *x; const float *residual; const uint8_t *q2_down;
    const int8_t *xq; const float *xd;
    int n_blocks, row_stride, hidden_dim;
} vllm_dyn_matvec_q2_dn_x86_ctx;

static void vllm_dyn_matvec_q2_dn_x86_worker(void *ctx_, int j) {
    vllm_dyn_matvec_q2_dn_x86_ctx *c = ctx_;
    const uint8_t *__restrict pr = c->q2_down + (size_t)j * c->row_stride;
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    int b;
    for (b = 0; b + 3 < c->n_blocks; b += 4) {
        a0 += q2_q8_block_dot(pr + (size_t)(b + 0) * 16, c->xq + (size_t)(b + 0) * 32, c->xd[b + 0]);
        a1 += q2_q8_block_dot(pr + (size_t)(b + 1) * 16, c->xq + (size_t)(b + 1) * 32, c->xd[b + 1]);
        a2 += q2_q8_block_dot(pr + (size_t)(b + 2) * 16, c->xq + (size_t)(b + 2) * 32, c->xd[b + 2]);
        a3 += q2_q8_block_dot(pr + (size_t)(b + 3) * 16, c->xq + (size_t)(b + 3) * 32, c->xd[b + 3]);
        pr += 64;
    }
    for (; b < c->n_blocks; b++) {
        a0 += q2_q8_block_dot(pr, c->xq + (size_t)b * 32, c->xd[b]);
        pr += 16;
    }
    c->x[j] = c->residual[j] + ((a0 + a1) + (a2 + a3));
}

static void dyn_matvec_q2_q8_fused_down_residual(
    float *__restrict x, const float *__restrict residual,
    const uint8_t *__restrict q2_down, const float *__restrict activated,
    int hidden_dim, int ffn_dim)
{
    int n_blocks = ffn_dim / 32;
    int row_stride = n_blocks * 16;

    int8_t *xq = (int8_t *)xq_alloc_canary((size_t)ffn_dim);
    float *xd = (float *)xq_alloc_canary((size_t)n_blocks * sizeof(float));
    if (!xq || !xd) {
        xq_free_canary(xq, (size_t)ffn_dim, "q2_dn:xq");
        xq_free_canary(xd, (size_t)n_blocks * sizeof(float), "q2_dn:xd");
        return;
    }
    quantize_row_q8_0_act(activated, xq, xd, ffn_dim);

    vllm_dyn_matvec_q2_dn_x86_ctx vc = { x, residual, q2_down, xq, xd, n_blocks, row_stride, hidden_dim };
    vllm_tp_parfor(0, hidden_dim, vllm_dyn_matvec_q2_dn_x86_worker, &vc);
    xq_free_canary(xq, (size_t)ffn_dim, "q2_dn:xq");
    xq_free_canary(xd, (size_t)n_blocks * sizeof(float), "q2_dn:xd");
}

/* ---- Batched Q2_1 matvec kernels (batch decode / prefill) ----
 * A single worker sweeps every block once per (row, token); the Q8_0
 * activation for each token is quantized once up front (block_matrix_assoc_natural:
 * weights read once per row, activations shared across the row sweep). */

typedef struct {
    float *gate_out, *up_out; const uint8_t *q2_gate, *q2_up;
    const int8_t *xq; const float *xd;
    int n_blocks, row_stride, cols, rows, n_batch;
} vllm_q2_gu_b_x86_ctx;

static void vllm_q2_gu_b_x86_worker(void *ctx_, int r) {
    vllm_q2_gu_b_x86_ctx *c = ctx_;
    for (int t = 0; t < c->n_batch; t++) {
        const int8_t *xt = c->xq + (size_t)t * c->cols;
        const float *xdt = c->xd + (size_t)t * c->n_blocks;
        const uint8_t *pg = c->q2_gate + (size_t)r * c->row_stride;
        const uint8_t *pu = c->q2_up   + (size_t)r * c->row_stride;
        float ag = 0.f, au = 0.f;
        for (int b = 0; b < c->n_blocks; b++) {
            ag += q2_q8_block_dot(pg, xt + (size_t)b * 32, xdt[b]);
            au += q2_q8_block_dot(pu, xt + (size_t)b * 32, xdt[b]);
            pg += 16; pu += 16;
        }
        c->gate_out[(size_t)t * c->rows + r] = ag;
        c->up_out[(size_t)t * c->rows + r]   = au;
    }
}

typedef struct {
    const float *x_batch; int8_t *xq; float *xd;
    int cols, n_blocks, n_batch;
} vllm_q2_gu_b_quant_ctx;

static void vllm_q2_gu_b_quant_worker(void *ctx_, int t) {
    vllm_q2_gu_b_quant_ctx *c = ctx_;
    quantize_row_q8_0_act(c->x_batch + (size_t)t * c->cols,
                          c->xq + (size_t)t * c->cols,
                          c->xd + (size_t)t * c->n_blocks, c->cols);
}

static void dyn_matvec_q2_q8_fused_gate_up_batched(
    float *__restrict gate_out, float *__restrict up_out,
    const uint8_t *__restrict q2_gate, const uint8_t *__restrict q2_up,
    const float *__restrict x_batch, int rows, int cols, int n_batch)
{
    if (n_batch <= 0) return;
    int n_blocks = cols / 32;
    int row_stride = n_blocks * 16;

    int8_t *xq = (int8_t *)xq_alloc_canary((size_t)n_batch * cols);
    float *xd = (float *)xq_alloc_canary((size_t)n_batch * n_blocks * sizeof(float));
    if (!xq || !xd) {
        xq_free_canary(xq, (size_t)n_batch * cols, "q2_gate_up:xq");
        xq_free_canary(xd, (size_t)n_batch * n_blocks * sizeof(float), "q2_gate_up:xd");
        return;
    }
    {
        vllm_q2_gu_b_quant_ctx vc = { x_batch, xq, xd, cols, n_blocks, n_batch };
        vllm_tp_parfor(0, n_batch, vllm_q2_gu_b_quant_worker, &vc);
    }

    vllm_q2_gu_b_x86_ctx vg = { gate_out, up_out, q2_gate, q2_up, xq, xd, n_blocks, row_stride, cols, rows, n_batch };
    vllm_tp_parfor(0, rows, vllm_q2_gu_b_x86_worker, &vg);

    xq_free_canary(xq, (size_t)n_batch * cols, "q2_gate_up:xq");
    xq_free_canary(xd, (size_t)n_batch * n_blocks * sizeof(float), "q2_gate_up:xd");
}

typedef struct {
    float *x_batch; const float *residual_batch; const uint8_t *q2_down;
    const int8_t *xq; const float *xd;
    int n_blocks, row_stride, ffn_dim, hidden_dim, n_batch;
} vllm_q2_dn_b_x86_ctx;

static void vllm_q2_dn_b_x86_worker(void *ctx_, int j) {
    vllm_q2_dn_b_x86_ctx *c = ctx_;
    for (int t = 0; t < c->n_batch; t++) {
        const int8_t *xt = c->xq + (size_t)t * c->ffn_dim;
        const float *xdt = c->xd + (size_t)t * c->n_blocks;
        const uint8_t *pr = c->q2_down + (size_t)j * c->row_stride;
        float acc = 0.f;
        for (int b = 0; b < c->n_blocks; b++) {
            acc += q2_q8_block_dot(pr, xt + (size_t)b * 32, xdt[b]);
            pr += 16;
        }
        c->x_batch[(size_t)t * c->hidden_dim + j] =
            c->residual_batch[(size_t)t * c->hidden_dim + j] + acc;
    }
}

static void dyn_matvec_q2_q8_fused_down_residual_batched(
    float *__restrict x_batch, const float *__restrict residual_batch,
    const uint8_t *__restrict q2_down, const float *__restrict activated_batch,
    int hidden_dim, int ffn_dim, int n_batch)
{
    if (n_batch <= 0) return;
    int n_blocks = ffn_dim / 32;
    int row_stride = n_blocks * 16;

    int8_t *xq = (int8_t *)xq_alloc_canary((size_t)n_batch * ffn_dim);
    float *xd = (float *)xq_alloc_canary((size_t)n_batch * n_blocks * sizeof(float));
    if (!xq || !xd) {
        xq_free_canary(xq, (size_t)n_batch * ffn_dim, "q2_dn_b:xq");
        xq_free_canary(xd, (size_t)n_batch * n_blocks * sizeof(float), "q2_dn_b:xd");
        return;
    }
    {
        vllm_q2_gu_b_quant_ctx vc = { activated_batch, xq, xd, ffn_dim, n_blocks, n_batch };
        vllm_tp_parfor(0, n_batch, vllm_q2_gu_b_quant_worker, &vc);
    }

    vllm_q2_dn_b_x86_ctx vd = { x_batch, residual_batch, q2_down, xq, xd, n_blocks, row_stride, ffn_dim, hidden_dim, n_batch };
    vllm_tp_parfor(0, hidden_dim, vllm_q2_dn_b_x86_worker, &vd);

    xq_free_canary(xq, (size_t)n_batch * ffn_dim, "q2_dn_b:xq");
    xq_free_canary(xd, (size_t)n_batch * n_blocks * sizeof(float), "q2_dn_b:xd");
}

/* ================================================================
 * M4: Q4_0 4x4 row-interleaved repack (nibble-unpack-free kernels)
 *
 * Legacy layout, 18 B per 32-col block (row-major):   { f16 d; qs[16] }
 * Repacked layout, 72 B per 4-row x 32-col group:     { f16 d0..d3; qs[64] }
 *
 * 4 consecutive rows are grouped (llama.cpp block_q4_0x4 byte layout):
 *   qs[i*4 .. i*4+4) = rows[i&3].qs[(i>>2)*4 .. +4) ^ 0x88888888   (i=0..15)
 * so each 16B slice qs[k*16..] holds the SAME 4-elem group k of all 4 rows,
 * interleaved 4 bytes per row. XOR 0x88 flips the +8 nibble bias into
 * two's-complement:  (b<<4)  sign-extends the low  nibble as (nibble-8)*16,
 * (b&0xf0) sign-extends the high nibble the same way, so an int8 dot needs
 * only a /16 fix-up at the end - zero per-nibble unpack instructions.
 * Total bytes are unchanged (72 == 4*18), rows are always 32-aligned, so the
 * repack is in-place per 4-row group.
 *
 * The 4x4 kernels accumulate per (row, 4-elem group) exactly like the legacy
 * q4x16_to_i8x32 kernels: each vdotq_laneq_s32 yields one group of all 4 rows
 * (lane i = row r4+i), folded straight into the same cross-block FMA chain,
 * so the int8->f32->FMA order is identical -> results are bit-exact vs the
 * legacy path (verified PPL/Needle).
 * ================================================================ */
void st_q4_repack_init(void) {
    const char *e = getenv("VLLM_DISABLE_Q4_REPACK");
    if (e && e[0] && e[0] != '0') g_st_q4_repack = 0;
}

/* In-place repack of one Q4_0 weight matrix [rows][cols] (row-major, legacy
 * 18 B/block layout). Returns 0 on success; -1 when the matrix can't be
 * repacked (rows%4, cols%32, non-NEON-dotprod target, or repack disabled)
 * and the buffer is left untouched. */
int repack_q4_0_4x4_inplace(uint8_t *buf, int rows, int cols) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    st_q4_repack_init();
    if (!g_st_q4_repack) return -1;
    if (rows % 4 != 0 || cols % 32 != 0) return -1;
    int nb = cols / 32;
    size_t row_stride = (size_t)nb * 18;
    size_t group_bytes = 4 * row_stride;              /* 4 source rows */
    if (g_q4r_scratch_cap < group_bytes) {            /* pre-alloc should cover */
        fprintf(stderr, "[ST] M4 repack scratch too small (%zu < %zu); repack disabled\n",
                g_q4r_scratch_cap, group_bytes);
        g_st_q4_repack = 0;
        return -1;
    }
    uint8_t *tmp = g_q4r_scratch;
    for (int r0 = 0; r0 < rows; r0 += 4) {
        uint8_t *dst = buf + (size_t)(r0 >> 2) * group_bytes;
        memcpy(tmp, dst, group_bytes);
        for (int b = 0; b < nb; b++) {
            uint8_t *out = dst + (size_t)b * 72;
            for (int i = 0; i < 4; i++)               /* 4 f16 scales in order */
                memcpy(out + (size_t)i * 2,
                       tmp + (size_t)i * row_stride + (size_t)b * 18, 2);
            const uint32_t xor_mask = 0x88888888u;
            for (int i = 0; i < 16; i++) {            /* 4-byte interleave ^0x88 */
                uint32_t e;
                memcpy(&e, tmp + (size_t)(i & 3) * row_stride + (size_t)b * 18 + 2
                             + (size_t)(i >> 2) * 4, 4);
                e ^= xor_mask;
                memcpy(out + 8 + (size_t)i * 4, &e, 4);
            }
        }
    }
    return 0;
#else
    (void)buf; (void)rows; (void)cols;
    return -1;
#endif
}

/* ================================================================
 * Q8_0 8x8 tiled layout (block_q8_0x8, 272 B/block): P0 tile upgrade.
 *   { d[8] f16; qs[256] }, qs[k*32 + m*4 + i] = row m's qs[k*4+i].
 * Same per-output int32/fp32 accumulation order as the 4x4 kernel =>
 * results are bit-identical (verified on board 2026-08-29, 1.6x GEMM /
 * 1.17x GEMV). Gate: VLLM_Q8_8X8=0 falls back to the 4x4 layout.
 * ================================================================ */
static int g_q8_8x8 = -1;
static int q8_8x8_enabled(void) {
    if (g_q8_8x8 < 0) {
        const char *e = getenv("VLLM_Q8_8X8");
        g_q8_8x8 = (e && e[0] == '0') ? 0 : 1;   /* default 8x8 */
    }
    return g_q8_8x8;
}

/* legacy [rows][nb*34] -> 8x8 (272 B/block), in-place per 8-row group via
 * g_q8r_scratch (same pattern as repack_q8_0_4x4_inplace). */
int repack_q8_0_8x8_inplace(uint8_t *buf, int rows, int cols) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    if (!g_st_q8_repack) return -1;
    if (rows % 8 != 0 || cols % 32 != 0) return -1;
    int nb = cols / 32;
    size_t group_bytes = (size_t)nb * 272;   /* 8x8 group == 8 legacy rows */
    if (g_q8r_scratch_cap < group_bytes) {
        uint8_t *np = (uint8_t *)realloc(g_q8r_scratch, group_bytes);
        if (!np) return -1;
        g_q8r_scratch = np;
        g_q8r_scratch_cap = group_bytes;
    }
    uint8_t *tmp = g_q8r_scratch;
    int q8chk = (getenv("VLLM_Q8CHK") && getenv("VLLM_Q8CHK")[0] == '1');
    for (int r0 = 0; r0 < rows; r0 += 8) {
        uint8_t *dst = buf + (size_t)(r0 >> 3) * group_bytes;
        memcpy(tmp, dst, group_bytes);
        for (int b = 0; b < nb; b++) {
            uint8_t *out = dst + (size_t)b * 272;
            for (int m = 0; m < 8; m++)        /* 8 f16 scales, row order */
                memcpy(out + (size_t)m * 2,
                       tmp + (size_t)m * nb * 34 + (size_t)b * 34, 2);
            for (int k = 0; k < 8; k++)        /* 8 x 32 B interleaved qs */
                for (int m = 0; m < 8; m++)
                    for (int i = 0; i < 4; i++)
                        out[16 + k * 32 + m * 4 + i] =
                            tmp[(size_t)m * nb * 34 + (size_t)b * 34 + 2 + k * 4 + i];
        }
        /* 自检：tmp(legacy 原数据) vs dst(8x8 新布局) 逐字节一致（VLLM_Q8CHK=1） */
        if (q8chk) {
            int bad = 0;
            for (int b = 0; b < nb; b++)
                for (int m = 0; m < 8; m++) {
                    if (memcmp(dst + (size_t)b * 272 + (size_t)m * 2,
                               tmp + (size_t)m * nb * 34 + (size_t)b * 34, 2) != 0) bad++;
                    for (int k = 0; k < 8; k++)
                        for (int i = 0; i < 4; i++)
                            if (dst[(size_t)b * 272 + 16 + k * 32 + m * 4 + i] !=
                                tmp[(size_t)m * nb * 34 + (size_t)b * 34 + 2 + k * 4 + i]) bad++;
                }
            if (bad)
                fprintf(stderr, "[Q8CHK] 8x8 repack MISMATCH rows=%d cols=%d r0=%d bad=%d\n",
                        rows, cols, r0, bad);
        }
    }
    return 0;
#else
    (void)buf; (void)rows; (void)cols;
    return -1;
#endif
}

/* dispatch: 8x8 (default) or legacy 4x4 weight repack (exported for main.c's
 * lm_head repack, which must match the per-layer layout) */
int repack_q8_0_tiled_inplace(uint8_t *buf, int rows, int cols) {
    if (q8_8x8_enabled()) return repack_q8_0_8x8_inplace(buf, rows, cols);
    return repack_q8_0_4x4_inplace(buf, rows, cols);
}

/* P4: repack one Q4_0 matrix [rows][cols] from legacy 18 B/block into the
 * 8x8 GEMM layout (144 B/block = 8 rows), written to a separate prefill-only
 * copy `dst`. Same byte size as the 8 legacy rows it replaces. Layout:
 *   d[8] f16 scales first (16 B), then each of the 8 rows' 32 nibbles stored
 *   contiguously (16 B/row, nibble order preserved). No ^0x88 - rows are
 *   unpacked at runtime with the standard nibble LUT (bytes_from_nibbles_32
 *   minus 8 -> signed). Rows/cols must be 8/32 aligned; returns -1 otherwise
 *   (legacy layout kept, has_x8 stays 0 for that matrix's consumer). */
int repack_q4_0_8x8l(uint8_t *__restrict dst, const uint8_t *__restrict src,
                     int rows, int cols) {
    /* llama.cpp 式 8x8 聚簇布局（block_q4_0x8, 144B = 8×18B）:
     *   [0..16)  8 个 f16 scale（行主序）
     *   [16..144) 128B qs：16 个 8B 块，块 i 来自行 (i%8) 的偏移 (i/8)*8，
     *              每字节 XOR 0x88（nibble 偏置形式 → 符号形式，省运行时减法）。
     * K-chunk 内 8 行权重连续（128B）→ GEMM 4 条 32B load 覆盖，行分离靠
     * 寄存器 blend/permute。Rows/cols 须 8/32 对齐，否则返回 -1。 */
    if (rows % 8 != 0 || cols % 32 != 0) return -1;
    int nb = cols / 32;
    size_t src_row = (size_t)nb * 18;
    for (int r0 = 0; r0 < rows; r0 += 8) {
        for (int b = 0; b < nb; b++) {
            uint8_t *out = dst + (size_t)(r0 >> 3) * nb * 144 + (size_t)b * 144;
            const uint8_t *sb = src + (size_t)r0 * src_row + (size_t)b * 18;
            for (int r = 0; r < 8; r++)           /* f16 scales, row-major */
                memcpy(out + (size_t)r * 2, sb + (size_t)r * src_row, 2);
            for (int i = 0; i < 16; i++) {        /* 8B interleave + XOR 0x88 */
                uint64_t e;
                memcpy(&e, sb + (size_t)(i & 7) * src_row + 2 + (size_t)(i >> 3) * 8, 8);
                e ^= 0x8888888888888888ULL;
                memcpy(out + 16 + (size_t)i * 8, &e, 8);
            }
        }
    }
    return 0;
}


/* M4e: in-place repack of one Q8_0 weight matrix [rows][cols] (legacy
 * row-major 34 B/block) into llama.cpp block_q8_0x4 (136 B/block):
 *   per 4-row group: d[4] f16 scales first (8 B), then 128 B qs with
 *   qs[k*16 + m*4 + i] = row m, block b, value (k*4+i). Same byte size
 *   (4*34 = 136), in-place per 4-row group via g_q8r_scratch. Returns 0 on
 *   success; -1 if not 4-row/32-col aligned (legacy layout kept). */
int repack_q8_0_4x4_inplace(uint8_t *buf, int rows, int cols) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    if (!g_st_q8_repack) return -1;
    if (rows % 4 != 0 || cols % 32 != 0) return -1;
    int nb = cols / 32;
    size_t group_bytes = (size_t)nb * 136;   /* 4x4 group == 4 legacy rows */
    if (g_q8r_scratch_cap < group_bytes) {
        uint8_t *np = (uint8_t *)realloc(g_q8r_scratch, group_bytes);
        if (!np) return -1;
        g_q8r_scratch = np;
        g_q8r_scratch_cap = group_bytes;
    }
    uint8_t *tmp = g_q8r_scratch;
    for (int r0 = 0; r0 < rows; r0 += 4) {
        uint8_t *dst = buf + (size_t)(r0 >> 2) * group_bytes;
        memcpy(tmp, dst, group_bytes);
        for (int b = 0; b < nb; b++) {
            uint8_t *out = dst + (size_t)b * 136;
            for (int m = 0; m < 4; m++)        /* 4 f16 scales, row order */
                memcpy(out + (size_t)m * 2,
                       tmp + (size_t)m * nb * 34 + (size_t)b * 34, 2);
            for (int k = 0; k < 8; k++)        /* 8 x 16 B interleaved qs */
                for (int m = 0; m < 4; m++)
                    for (int i = 0; i < 4; i++)
                        out[8 + k * 16 + m * 4 + i] =
                            tmp[(size_t)m * nb * 34 + (size_t)b * 34 + 2 + k * 4 + i];
        }
    }
    return 0;
#else
    (void)buf; (void)rows; (void)cols;
    return -1;
#endif
}

/* ================================================================
 * G=256 group quantization, stored in the Q8_0 34 B/block layout.
 *
 * ONE f32 scale covers 256 columns (max-abs/127, standard Q8 saturating
 * quantize); the int8 codes are rounded against that f32 scale, and each of
 * the 8 inner 32-wide sub-blocks repeats the SAME f16(scale). This keeps the
 * CPU Q8_0 kernels bit-compatible (they dequantize 34 B/block as usual), and
 * lets the NPU DIRECT backend run K=256 per submit - the block count for a
 * 4096-wide K drops 128 -> 16, /8 ioctls (the measured NPU-vs-CPU crossover).
 * Dequant = int8 * f16(scale); CPU and NPU consume identical int8 codes and
 * the same f16 scale bits, so both paths agree to the bit (given equal
 * accumulation order). Precision: coarser than per-32 Q8_0 (~0.5-1% PPL),
 * acceptable trade for real NPU prefill speedup.
 * ================================================================ */
void f32_to_g256q8(uint8_t *q8_out, const float *f32_in, int n_elements) {
    const int super = 256;                     /* G: columns per scale */
    const int n_sup = n_elements / super;
    const int dbg = (getenv("VLLM_G256_DBG") != NULL);
    for (int s = 0; s < n_sup; s++) {
        const float *xs = f32_in + (size_t)s * super;
        float max_abs = 1e-10f;
        for (int i = 0; i < super; i++) {
            float av = fabsf(xs[i]);
            if (av > max_abs) max_abs = av;
        }
        float scale = max_abs / 127.0f;
        if (scale < 1e-10f) scale = 1e-10f;
        uint16_t h = f32_to_f16_bits(scale);   /* repeated across sub-blocks */
        if (dbg && (s == 0 || s == 1000 || s == n_sup / 2)) {
            printf("[G256DBG] n=%d s=%d max_abs=%.8g scale=%.8g h16=0x%04x x[0]=%.8g\n",
                   n_elements, s, max_abs, scale, (unsigned)h, xs[0]);
            fflush(stdout);
        }
        for (int sub = 0; sub < 8; sub++) {    /* 8 x 32-wide sub-blocks */
            uint8_t *qb = q8_out + (size_t)(s * 8 + sub) * 34;
            memcpy(qb, &h, 2);                 /* same scale in every block */
            int8_t *qs = (int8_t *)(qb + 2);
            const float *xb = xs + (size_t)sub * 32;
            for (int j = 0; j < 32; j++) {
                float v = xb[j];
                int q = (int)(v / scale + 0.5f);
                if (q > 127) q = 127;
                if (q < -127) q = -127;
                qs[j] = (int8_t)q;
            }
        }
    }
}

/* Q8_0 element offset helper: element index → byte offset in compact storage */
#define Q8O(e)  ((size_t)(e) / 32 * 34)

/* ================================================================
 * NEON (aarch64) kernel section - RK3588 migration
 * (axiom: blas_precision_efficiency_tradeoff + fixedpoint_quantize_001)
 *
 * The Q8_0/Q4_0 integer-domain dots use vdotq_s32 (ARMv8.2 dotprod,
 * Cortex-A76/A55) where available, else a widening-mla fallback. The
 * accumulation order mirrors the AVX2/AVX512 paths, so results are
 * reproducible per-platform (determinism red line). Cross-arch
 * bit-exactness is NOT guaranteed (different fp32 reduction order -
 * documented; consistency is verified at the PPL/greedy level).
 *
 * Layout conventions (identical to the x86 kernels):
 *   Q8 single-token : int8 weight x fp32 activation (fp32 accumulate).
 *   Q8 batched      : block-major quantized act xq[b][n_batch][32],
 *                     xd[b][n_batch]; output token-major [t * rows + r].
 *   Q4              : nibble weight x Q8_0 quantized activation.
 * ================================================================ */
#if ST_HAVE_NEON

#if defined(__ARM_FEATURE_DOTPROD)
#define ST_NEON_DOTPROD 1
#else
#define ST_NEON_DOTPROD 0
#endif

/* NEON register tiles (smaller than x86: 2 fp32x4 accumulators per output
 * vs one __m256; keeps the 32-register NEON budget from spilling).
 * Tuned 2026-08-20 on RK3588 (4x A76): larger tiles cut the weight re-read
 * count per row (n_batch/tile) at the cost of accumulator registers. */
#define NEON_TILE_QKV 3   /* acc = tile*2*3 (Q/K/V) = 18 regs */
#define NEON_TILE_QKV_Q 6  /* Q-only half of QKV: 3072/4096 rows, acc = tile*2 = 12 regs */
#define NEON_TILE_GU  5   /* acc = tile*2*2 (gate/up) = 20 regs */
#define NEON_TILE_1   8   /* acc = tile*2 (single) = 16 regs */

/* Horizontal sum of a float32x4_t. */
static inline float hsum_neon4(float32x4_t v) {
    float32x2_t s = vadd_f32(vget_low_f32(v), vget_high_f32(v));
    return vget_lane_f32(vpadd_f32(s, s), 0);
}

/* 16 int8 x 16 int8 -> int32x4 (4 lanes, each = 4 products). */
static inline int32x4_t i8x16_dot_s32(int8x16_t a, int8x16_t b) {
#if ST_NEON_DOTPROD
    return vdotq_s32(vdupq_n_s32(0), a, b);
#else
    int16x8_t la = vmovl_s8(vget_low_s8(a)),  ha = vmovl_s8(vget_high_s8(a));
    int16x8_t lb = vmovl_s8(vget_low_s8(b)),  hb = vmovl_s8(vget_high_s8(b));
    int32x4_t acc = vmull_s16(vget_low_s16(la), vget_low_s16(lb));  /* 0-3 */
    acc = vmlal_s16(acc, vget_high_s16(la), vget_high_s16(lb));     /* 4-7 */
    acc = vmlal_s16(acc, vget_low_s16(ha), vget_low_s16(hb));       /* 8-11 */
    acc = vmlal_s16(acc, vget_high_s16(ha), vget_high_s16(hb));     /* 12-15 */
    return acc;
#endif
}

/* Fast vectorized exp(x) - mirrors exp_ps256 instruction-for-instruction
 * (round-nearest-even, fma, 2^n scale) so softmax matches the x86 build. */
static inline float32x4_t exp_neon4(float32x4_t x) {
    const float32x4_t log2e = vdupq_n_f32(1.4426950408889634f);
    const float32x4_t ln2   = vdupq_n_f32(0.6931471805599453f);
    const float32x4_t c6 = vdupq_n_f32(0.001388888888888889f);
    const float32x4_t c5 = vdupq_n_f32(0.008333333333333333f);
    const float32x4_t c4 = vdupq_n_f32(0.041666666666666664f);
    const float32x4_t c3 = vdupq_n_f32(0.16666666666666666f);
    const float32x4_t c2 = vdupq_n_f32(0.5f);
    const float32x4_t c1 = vdupq_n_f32(1.0f);

    x = vmaxq_f32(x, vdupq_n_f32(-87.0f));
    x = vminq_f32(x, vdupq_n_f32(87.0f));

    float32x4_t n = vrndnq_f32(vmulq_f32(x, log2e));
    float32x4_t r = vfmsq_f32(x, n, ln2);   /* x - n*ln2 */

    float32x4_t p = c6;
    p = vfmaq_f32(c5, p, r);
    p = vfmaq_f32(c4, p, r);
    p = vfmaq_f32(c3, p, r);
    p = vfmaq_f32(c2, p, r);
    p = vfmaq_f32(c1, p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f), p, r);

    int32x4_t ni = vcvtq_s32_f32(n);
    int32x4_t bi = vshlq_n_s32(ni, 23);
    return vreinterpretq_f32_s32(vaddq_s32(bi, vreinterpretq_s32_f32(p)));
}

/* 16 bytes (Q4_0 block) -> 32 int8 in [-8,7]. Natural order: elements 0-15
 * = low nibbles, 16-31 = high nibbles (matches bytes_from_nibbles_32). */
static inline void q4x16_to_i8x32(const uint8_t *p, int8x16_t *lo, int8x16_t *hi) {
    uint8x16_t v = vld1q_u8(p);
    uint8x16_t mask = vdupq_n_u8(0x0F);
    *lo = vreinterpretq_s8_u8(vandq_u8(v, mask));
    *hi = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(v, 4), mask));
    *lo = vsubq_s8(*lo, vdupq_n_s8(8));
    *hi = vsubq_s8(*hi, vdupq_n_s8(8));
}

/* Repack token-major q8_0 激活（xq[S*cols] int8 + xd[S*nb] f32）为 llama.cpp
 * block_q8_0x4（136 B/块）：{ d[4] f16 缩放, qs[128] }，
 * qs[k*16 + m*4 + i] = act[m][4k+i]（低 16），qs[64 + ...] = act[m][16+4k+i]。
 * 写入 g_q8r_scratch（按需增长）。返回 0 成功 / -1 分配失败。
 * 纯数据搬运（无 NEON 内建），x86/ARM 共用。 */
static int repack_q8_0_4x4(const int8_t *__restrict xq, const float *__restrict xd,
                           int n_batch, int cols) {
    int nb = cols / 32;
    size_t nbytes = (size_t)(n_batch / 4) * nb * 136;
    if (g_q8r_scratch_cap < nbytes) {
        uint8_t *np = (uint8_t *)realloc(g_q8r_scratch, nbytes);
        if (!np) return -1;
        g_q8r_scratch = np;
        g_q8r_scratch_cap = nbytes;
    }
    uint8_t *vy = g_q8r_scratch;
    for (int tile = 0; tile < n_batch / 4; tile++) {
        const int8_t *xm[4];
        const float *dm[4];
        for (int m = 0; m < 4; m++) {
            xm[m] = xq + (size_t)(tile * 4 + m) * cols;
            dm[m] = xd + (size_t)(tile * 4 + m) * nb;
        }
        uint8_t *yb = vy + (size_t)tile * nb * 136;
        for (int b = 0; b < nb; b++) {
            uint16_t d[4];
            for (int m = 0; m < 4; m++) d[m] = f32_to_f16_bits(dm[m][b]);
            memcpy(yb + (size_t)b * 136, d, 8);
            uint8_t *qs = yb + (size_t)b * 136 + 8;
            for (int k = 0; k < 4; k++)
                for (int m = 0; m < 4; m++)
                    for (int i = 0; i < 4; i++) {
                        qs[k * 16 + m * 4 + i]      = (uint8_t)xm[m][b * 32 + k * 4 + i];
                        qs[64 + k * 16 + m * 4 + i] = (uint8_t)xm[m][b * 32 + 16 + k * 4 + i];
                    }
        }
    }
    return 0;
}

#if ST_NEON_DOTPROD
/* ---- M4: 4x4-repacked Q4_0 block dot ----
 * bq points at a 72B group { 4xf16 scale; qs[64] } in the repacked layout
 * (see repack_q4_0_4x4_inplace): qs[k*16 + r*4 .. +4) is 4-elem group k of
 * row r4+r, XOR 0x88. Each vdotq_laneq_s32 below computes one group across
 * all 4 rows (lane i = row r4+i):
 *   (b<<4)   sign-extends low  nibbles (elems 4k..4k+3) as (nibble-8)*16
 *   (b&0xf0) sign-extends high nibbles (elems 16+4k..) the same way
 * vcvtq_n_f32_s32(x, 4) undoes the /16 exactly (powers of two are exact in
 * f32), so each FMA equals the legacy unpack path's (float)dot * (wd*act_scale)
 * per (row, group) - bit-identical accumulation order.
 * lo[k]/hi[k] pack the 4 rows of group k (lane i = row r4+i). */
static inline void q4x4_dot1_group(const uint8_t *__restrict bq,
                                   const int8_t *__restrict xq_b,
                                   float act_scale,
                                   float32x4_t lo[4], float32x4_t hi[4]) {
    int8x16_t b0 = vld1q_s8((const int8_t *)(bq + 8));
    int8x16_t b1 = vld1q_s8((const int8_t *)(bq + 24));
    int8x16_t b2 = vld1q_s8((const int8_t *)(bq + 40));
    int8x16_t b3 = vld1q_s8((const int8_t *)(bq + 56));
    int8x16_t a0 = vld1q_s8(xq_b);
    int8x16_t a1 = vld1q_s8(xq_b + 16);
    int32x4_t r0 = vdotq_laneq_s32(vdupq_n_s32(0), vshlq_n_s8(b0, 4), a0, 0);
    int32x4_t r1 = vdotq_laneq_s32(vdupq_n_s32(0), vshlq_n_s8(b1, 4), a0, 1);
    int32x4_t r2 = vdotq_laneq_s32(vdupq_n_s32(0), vshlq_n_s8(b2, 4), a0, 2);
    int32x4_t r3 = vdotq_laneq_s32(vdupq_n_s32(0), vshlq_n_s8(b3, 4), a0, 3);
    const int8x16_t msk = vdupq_n_s8((int8_t)0xf0);
    int32x4_t s0 = vdotq_laneq_s32(vdupq_n_s32(0), vandq_s8(b0, msk), a1, 0);
    int32x4_t s1 = vdotq_laneq_s32(vdupq_n_s32(0), vandq_s8(b1, msk), a1, 1);
    int32x4_t s2 = vdotq_laneq_s32(vdupq_n_s32(0), vandq_s8(b2, msk), a1, 2);
    int32x4_t s3 = vdotq_laneq_s32(vdupq_n_s32(0), vandq_s8(b3, msk), a1, 3);
    float32x4_t wd = vcvt_f32_f16(vld1_f16((const __fp16 *)bq));
    float32x4_t dsv = vmulq_f32(wd, vdupq_n_f32(act_scale));
    lo[0] = vfmaq_f32(lo[0], vcvtq_n_f32_s32(r0, 4), dsv);
    lo[1] = vfmaq_f32(lo[1], vcvtq_n_f32_s32(r1, 4), dsv);
    lo[2] = vfmaq_f32(lo[2], vcvtq_n_f32_s32(r2, 4), dsv);
    lo[3] = vfmaq_f32(lo[3], vcvtq_n_f32_s32(r3, 4), dsv);
    hi[0] = vfmaq_f32(hi[0], vcvtq_n_f32_s32(s0, 4), dsv);
    hi[1] = vfmaq_f32(hi[1], vcvtq_n_f32_s32(s1, 4), dsv);
    hi[2] = vfmaq_f32(hi[2], vcvtq_n_f32_s32(s2, 4), dsv);
    hi[3] = vfmaq_f32(hi[3], vcvtq_n_f32_s32(s3, 4), dsv);
}

/* Per-row horizontal sum of a 4x4 group, mirroring the legacy kernels'
 * hsum_neon4(vaddq_f32(acc0, acc1)) tree exactly: t_k = lo_k+hi_k (lane add),
 * then s = (t0+t2) + (t1+t3). Built from NEON intrinsics so the -ffast-math
 * on-board build cannot re-associate the f32 adds; the final lane extract +
 * single add of two operands has no reassociation freedom. */
static inline float q4x4_row_sum(const float32x4_t lo[4], const float32x4_t hi[4],
                                 int row_lane) {
    float32x4_t t0 = vaddq_f32(lo[0], hi[0]);
    float32x4_t t1 = vaddq_f32(lo[1], hi[1]);
    float32x4_t t2 = vaddq_f32(lo[2], hi[2]);
    float32x4_t t3 = vaddq_f32(lo[3], hi[3]);
    float32x4_t s02 = vaddq_f32(t0, t2);   /* lane i = t0[i] + t2[i] */
    float32x4_t s13 = vaddq_f32(t1, t3);   /* lane i = t1[i] + t3[i] */
    return vgetq_lane_f32(s02, row_lane) + vgetq_lane_f32(s13, row_lane);
}

/* ---- M4c: 16 元素一次累加（llama.cpp ggml_gemm_q4_0_4x4_q8_0 风格）----
 * 逐组路径（q4x4_dot1_group）每 block 每 4 行做 8 次 vcvt+vfma，是 prefill
 * 计算指令的瓶颈（vcvt/vfma 为 llama 的 4 倍）。这里把 4 个 vdot 的 int32
 * 结果先两两相加成 2 个 16 元素组和（整数加法精确、无溢出：每元素点积
 * ≤ 4*112*127 ≈ 5.7e4，16 元素和 ≤ 2.3e5 << 2^31），再做 2 次 vcvt(/16,
 * 因和恒为 16 的倍数而精确) + 2 次 vfma -> FP 指令 -75%，每 token 累加器
 * 从 8 个减到 2 个（tile 可到 12）。
 * 代价：f32 舍入顺序与逐组路径不同，位级一致放宽 -> PPL 需重新验收。
 * 仅 M4b 真 GEMM（q4x4_gemm_batched）使用；decode GEMV 继续走逐组路径，
 * 保持位级一致。 */
static inline void q4x4_dot1_group16(const uint8_t *__restrict bq,
                                     const int8_t *__restrict xq_b,
                                     float act_scale,
                                     float32x4_t *lo, float32x4_t *hi) {
    int8x16_t b0 = vld1q_s8((const int8_t *)(bq + 8));
    int8x16_t b1 = vld1q_s8((const int8_t *)(bq + 24));
    int8x16_t b2 = vld1q_s8((const int8_t *)(bq + 40));
    int8x16_t b3 = vld1q_s8((const int8_t *)(bq + 56));
    int8x16_t a0 = vld1q_s8(xq_b);
    int8x16_t a1 = vld1q_s8(xq_b + 16);
    const int8x16_t msk = vdupq_n_s8((int8_t)0xf0);
    int32x4_t r0 = vdotq_laneq_s32(vdupq_n_s32(0), vshlq_n_s8(b0, 4), a0, 0);
    int32x4_t r1 = vdotq_laneq_s32(vdupq_n_s32(0), vshlq_n_s8(b1, 4), a0, 1);
    int32x4_t r2 = vdotq_laneq_s32(vdupq_n_s32(0), vshlq_n_s8(b2, 4), a0, 2);
    int32x4_t r3 = vdotq_laneq_s32(vdupq_n_s32(0), vshlq_n_s8(b3, 4), a0, 3);
    int32x4_t s0 = vdotq_laneq_s32(vdupq_n_s32(0), vandq_s8(b0, msk), a1, 0);
    int32x4_t s1 = vdotq_laneq_s32(vdupq_n_s32(0), vandq_s8(b1, msk), a1, 1);
    int32x4_t s2 = vdotq_laneq_s32(vdupq_n_s32(0), vandq_s8(b2, msk), a1, 2);
    int32x4_t s3 = vdotq_laneq_s32(vdupq_n_s32(0), vandq_s8(b3, msk), a1, 3);
    /* 16 元素一次：4 个 vdot 结果 int32 相加（顺序任意，int 加法精确） */
    int32x4_t rs = vaddq_s32(vaddq_s32(r0, r1), vaddq_s32(r2, r3));
    int32x4_t ss = vaddq_s32(vaddq_s32(s0, s1), vaddq_s32(s2, s3));
    float32x4_t wd = vcvt_f32_f16(vld1_f16((const __fp16 *)bq));
    float32x4_t dsv = vmulq_f32(wd, vdupq_n_f32(act_scale));
    *lo = vfmaq_f32(*lo, vcvtq_n_f32_s32(rs, 4), dsv);
    *hi = vfmaq_f32(*hi, vcvtq_n_f32_s32(ss, 4), dsv);
}

/* ---- M4f: llama.cpp ggml_gemv_q4_0_4x4_q8_0 移植（decode 单 token GEMV）----
 * 源自 llama.cpp，MIT (c) 2023-2026 The ggml authors（全文见本文件头/tools 提取文件）。
 * 2026-08-22。比 q4x4_dot1_group16 更进一步：8 个 vdot 全部累加进单个
 * int32x4（int 加法精确、无舍入），每 block 仅 1 次 vcvt(/16, 因点积恒为
 * 16 的倍数而精确) + 1 次 vfma - 对比逐组路径（q4x4_dot1_group）每 block
 * 8 次 vcvt + 8 次 vfma，FP 指令 -87.5%；与 llama gemv 完全同构，只是
 * 适配我方预 XOR 0x88 的 qs 布局（vshl/vand 直接取符号）与 f32 激活 scale。
 * int32 无溢出：每元素 |w|≤7, |a|≤127 -> 组点积 ≤ 4*7*127 ≈ 3.6e3，
 * 32 元素点积 ≤ 2.8e4 << 2^31。
 * 代价：f32 舍入顺序与逐组路径不同（每 block 一次舍入 vs 8 次）-> 位级
 * 一致放宽，PPL 重新验收（与 Q8 decode M4e 同口径）。 */
static inline void q4x4_dot1_group16_gemv(const uint8_t *__restrict bq,
                                          const int8_t *__restrict xq_b,
                                          float act_scale, float32x4_t *acc) {
    int8x16_t b0 = vld1q_s8((const int8_t *)(bq + 8));
    int8x16_t b1 = vld1q_s8((const int8_t *)(bq + 24));
    int8x16_t b2 = vld1q_s8((const int8_t *)(bq + 40));
    int8x16_t b3 = vld1q_s8((const int8_t *)(bq + 56));
    int8x16_t a0 = vld1q_s8(xq_b);
    int8x16_t a1 = vld1q_s8(xq_b + 16);
    const int8x16_t msk = vdupq_n_s8((int8_t)0xf0);
    int32x4_t ret = vdupq_n_s32(0);
    ret = vdotq_laneq_s32(ret, vshlq_n_s8(b0, 4), a0, 0);
    ret = vdotq_laneq_s32(ret, vshlq_n_s8(b1, 4), a0, 1);
    ret = vdotq_laneq_s32(ret, vshlq_n_s8(b2, 4), a0, 2);
    ret = vdotq_laneq_s32(ret, vshlq_n_s8(b3, 4), a0, 3);
    ret = vdotq_laneq_s32(ret, vandq_s8(b0, msk), a1, 0);
    ret = vdotq_laneq_s32(ret, vandq_s8(b1, msk), a1, 1);
    ret = vdotq_laneq_s32(ret, vandq_s8(b2, msk), a1, 2);
    ret = vdotq_laneq_s32(ret, vandq_s8(b3, msk), a1, 3);
    float32x4_t wd = vcvt_f32_f16(vld1_f16((const __fp16 *)bq));
    float32x4_t dsv = vmulq_f32(wd, vdupq_n_f32(act_scale));
    *acc = vfmaq_f32(*acc, vcvtq_n_f32_s32(ret, 4), dsv);
}

/* P2a (2026-08-29): a 2-accumulator variant of the above was A/B'd on the
 * board (VLLM_Q4_2ACC, 64-token means): qkv 6.360 vs 6.362ms, gu 17.862 vs
 * 17.865, down 9.324 vs 9.353 — no gain on any Q4 phase (differences <0.03ms,
 * i.e. DRAM-bound, chain length irrelevant). Reverted; keep the single chain. */

/* ---- M4d: llama.cpp 4x4 asm GEMM (ported verbatim) ----
 * 2026-08-22 板端微基准实测（S=512, cols=4096）：M5 = 7.74 cyc/block（152
 * GMAC/s）vs M4c TILE=3 = 27.8 cyc/block（42 GMAC/s）-> 内核 3.6x，接近 A76
 * vdot 峰值利用率（54% vs 14.7%）。
 *
 * 移植方式：llama.cpp ggml/src/ggml-cpu/arch/arm/repack.cpp 的
 * ggml_gemm_q4_0_4x4_q8_0() 手写 NEON asm（.inst sdot、16 累加器、软件流水
 * 载荷）由 tools/extract_llama_asm.py 机械提取（零转录风险），语义与 generic
 * 完全一致：
 *   s[row*bs + col] = sum_k W(row,k) * A(col,k)   （nr rows x nc cols）
 * 我方映射（输出 out[token*rows + row]）：
 *   nr = n_batch（tokens），nc = rows（权重行），bs = rows，
 *   vx = 现有 72B 4x4 repack（[rows/4][nb]，与 block_q4_0x4 字节布局一致），
 *   vy = 新增 block_q8_0x4 激活 repack（136B/块，[n_batch/4][nb]）。
 * 约束：n_batch % 4 == 0（否则退回 C TILE=3 路径）；f16 激活缩放 + fma 顺序
 * 与 M4c 不同 -> PPL 需重新验收（流程与 M4c 相同）。residual 在 asm 之后
 * 向量化后加（与 C 路径的 out = residual + matmul 语义一致）。
 * 依据公理：blas_matrix_block_natural_isomorphism（block_matrix_assoc_natural）。 */
#include "../../tools/llama_gemm_q4_0_4x4_asm.c"

typedef struct {
    float *out; const uint8_t *q4_w;
    int nb, cols, rows, n_batch;
    int row_slice, wblock_bytes, ntiles;
} vllm_q4x4_b0_ctx;

static void vllm_q4x4_b0_worker(void *ctx_, int it) {
    vllm_q4x4_b0_ctx *c = ctx_;
    /* M4h: llama.cpp 行分片调度。任务 = (行分片 rs) x (16-token tile)。
     * 4 线程同时处理同一行分片（索引行优先）-> 分片驻留 L3，权重 DRAM
     * 流量 ÷8（每片仅首次 token-group 读 DRAM）；任务粒度细 -> 负载均衡
     * 优于 9 个大 token-tile 任务。逐元素累加顺序不变 -> 位级一致。 */
    int tile = it % c->ntiles;
    int rs   = it / c->ntiles;
    int t0 = tile * 16;
    int nt = c->n_batch - t0;
    if (nt > 16) nt = 16;
    int r0 = rs * c->row_slice;
    int nr = c->rows - r0;
    if (nr > c->row_slice) nr = c->row_slice;
    st_gemm_q4_0_4x4_q8_0_neon(c->cols,
                               c->out + (size_t)t0 * c->rows + r0,
                               (size_t)c->rows,
                               c->q4_w + (size_t)rs * (c->row_slice >> 2) * c->wblock_bytes,
                               g_q8r_scratch + (size_t)(t0 >> 2) * c->nb * 136,
                               nt, nr);
}

typedef struct {
    float *out; const float *residual;
    int rows, n_batch;
} vllm_q4x4_res_ctx;

static void vllm_q4x4_res_worker(void *ctx_, int t) {
    vllm_q4x4_res_ctx *c = ctx_;
    float *o = c->out + (size_t)t * c->rows;
    const float *r = c->residual + (size_t)t * c->rows;
    for (int r4 = 0; r4 < c->rows; r4 += 4) {
        float32x4_t ov = vld1q_f32(o + r4);
        float32x4_t rv = vld1q_f32(r + r4);
        vst1q_f32(o + r4, vaddq_f32(rv, ov));
    }
}

static void st_gemm_q4_0_4x4_batched(float *__restrict out,
                                     const uint8_t *__restrict q4_w,
                                     int rows, int cols, int n_batch,
                                     const float *__restrict residual) {
    int nb = cols / 32;
    /* M4h: llama.cpp 行分片（nr0/(nth*4)，min 4 行，cap 512 保 L3 驻留）。
     * nr=16 大 token-tile 已被实测否定（权重重读不减、负载更差）；行分片
     * 把"每 token-group 重读全权重"变为"每分片只从 DRAM 读一次"。
     * 2026-08-29 板端 A/B（2B Q4_0，132-token prefill）：
     *   auto 1976-1998ms vs 不分片 2191ms（-10%）；GATEUP -28%、QKV/O -5~15%。
     *   初版 nb>128 宽收缩豁免经多轮复测被推翻（分片 down 494-517 vs 不分片
     *   551-582，噪声掩盖了 9 任务 3/2/2/2 负载不均的 -25% 损失）-> 全矩阵分片。
     * 调参：VLLM_ROW_SLICE=n 覆盖（0=不分片；>0=固定分片行数）。 */
    static int rs_env = -1;
    if (rs_env < 0) {
        const char *e = getenv("VLLM_ROW_SLICE");
        rs_env = e ? atoi(e) : -1;
    }
    int ntiles = (n_batch + 15) / 16;
    int row_slice;
    int no_split = 0;
    if (rs_env == 0) {
        row_slice = rows;                       /* 0 = 不分片（旧 9-tile 行为） */
        no_split = 1;
    } else if (rs_env > 0) {
        row_slice = rs_env;
    } else {
        row_slice = (rows + 15) / 16;           /* nth=4 -> rows/16，同 llama chunk_size */
    }
    row_slice = (row_slice + 3) & ~3;
    if (row_slice < 4) row_slice = 4;
    if (!no_split && row_slice > 512) row_slice = 512;  /* cap 仅对分片模式 */
    if (row_slice > rows) row_slice = rows;
    int nslices = (rows + row_slice - 1) / row_slice;
    vllm_q4x4_b0_ctx vc = { out, q4_w, nb, cols, rows, n_batch, row_slice, nb * 72, ntiles };
    vllm_tp_parfor(0, nslices * ntiles, vllm_q4x4_b0_worker, &vc);
    if (residual) {
        vllm_q4x4_res_ctx vr = { out, residual, rows, n_batch };
        vllm_tp_parfor(0, n_batch, vllm_q4x4_res_worker, &vr);
    }
}

/* ---- M4b: 真 GEMM batched (axiom block_matrix_assoc_natural) ----
 * 自研 prefill batched 内核原来是"batched GEMV"：外层 (4行组) × 中层 (token) ×
 * 内层 (权重 block)，每个 token 把同一份权重 block 重新读一遍（N 倍权重 DRAM
 * 流量），在 RK3588 低带宽平台上把 prefill 利用率压到 ~13% vs llama.cpp 的
 * ~52%（llama 的 ggml_gemm_q4_0_4x4_q8_0 用 4行×4 token tile，权重读一次、
 * token 内循环共享，依据公理"重排等价计算图 / shared input fusion"）。
 *
 * 这里把 batched 内核重排为真 GEMM 结构：
 *   for (4行组)  for (token tile)  for (权重 block)  for (tile 内 token)
 * 权重 block 在每个 token tile 内只读一次 -> 权重 DRAM 流量 ÷tile；
 * 单 token 的 block 累加顺序不变 -> 与原实现位级一致（PPL/Needle 回归验证）。
 * token tile = 3（M4c 16 元素一次累加，每 token 仅 2 个 f32x4 累加器；
 * 实测（2026-08-22 微基准）TILE=3 在 nb>=256 时 26 cyc/block，TILE=12 因
 * 24 累加器寄存器压力慢 50%（40 cyc/block）-> TILE 必须保持 3）。residual
 * 非空时输出 out = residual + matmul（fused 语义）。
 * 注意：16 元素一次累加放宽了位级一致（f32 舍入顺序变化）-> PPL 需重新
 * 验收；本内核仅在 4x4 repack 布局下使用（调用处已由 g_st_q4_repack 门控）。 */
typedef struct {
    const uint8_t *q4_w; const int8_t *xq; const float *xd;
    float *out; const float *residual;
    int rows, cols, n_blocks, n_batch, row_stride;
} vllm_q4x4_gemm_ctx;

static void vllm_q4x4_gemm_worker(void *ctx_, int it) {
    vllm_q4x4_gemm_ctx *c = ctx_;
    const int TILE = 3;
    int r4 = it * 4;
    const uint8_t *__restrict pr = c->q4_w + (size_t)(r4 >> 2) * c->row_stride * 4;
    for (int t0 = 0; t0 < c->n_batch; t0 += TILE) {
        int nt = c->n_batch - t0;
        if (nt > TILE) nt = TILE;
        float32x4_t lo[TILE], hi[TILE];
        for (int tt = 0; tt < nt; tt++) {
            lo[tt] = vdupq_n_f32(0.0f);
            hi[tt] = vdupq_n_f32(0.0f);
        }
        for (int b = 0; b < c->n_blocks; b++) {
            const uint8_t *bq = pr + (size_t)b * 72;
            ST_PREFETCH(pr + (size_t)(b + 4) * 72);
            for (int tt = 0; tt < nt; tt++) {
                int t = t0 + tt;
                q4x4_dot1_group16(bq, c->xq + (size_t)t * c->cols + (size_t)b * 32,
                                  c->xd[(size_t)t * c->n_blocks + b], &lo[tt], &hi[tt]);
            }
        }
        for (int tt = 0; tt < nt; tt++) {
            int t = t0 + tt;
            float *op = c->out + (size_t)t * c->rows + r4;
            /* 16 元素一次：行和 = 低16元素组 + 高16元素组 */
            float32x4_t s = vaddq_f32(lo[tt], hi[tt]);
            if (c->residual) {
                const float *rp = c->residual + (size_t)t * c->rows + r4;
                op[0] = rp[0] + vgetq_lane_f32(s, 0);
                op[1] = rp[1] + vgetq_lane_f32(s, 1);
                op[2] = rp[2] + vgetq_lane_f32(s, 2);
                op[3] = rp[3] + vgetq_lane_f32(s, 3);
            } else {
                op[0] = vgetq_lane_f32(s, 0);
                op[1] = vgetq_lane_f32(s, 1);
                op[2] = vgetq_lane_f32(s, 2);
                op[3] = vgetq_lane_f32(s, 3);
            }
        }
    }
}

static void q4x4_gemm_batched(float *__restrict out, const uint8_t *__restrict q4_w,
                              const int8_t *__restrict xq, const float *__restrict xd,
                              int rows, int cols, int n_batch,
                              const float *__restrict residual) {
    const int TILE = 3;
    /* M4d: llama.cpp 4x4 asm GEMM fast path. 条件：4x4 repack 开启、权重行
     * %4（nc 侧）、token 数 %4（nr 侧）。激活 repack 失败则回退 C 路径。 */
#if ST_NEON_DOTPROD
    if (g_st_q4_repack && (rows & 3) == 0 && (n_batch & 3) == 0 &&
        repack_q8_0_4x4(xq, xd, n_batch, cols) == 0) {
        st_gemm_q4_0_4x4_batched(out, q4_w, rows, cols, n_batch, residual);
        return;
    }
#endif
    int n_blocks = cols / 32;
    size_t row_stride = n_blocks * 18;
    vllm_q4x4_gemm_ctx vc = { q4_w, xq, xd, out, residual, rows, cols, n_blocks, n_batch, (int)row_stride };
    vllm_tp_parfor(0, (rows + 3) / 4, vllm_q4x4_gemm_worker, &vc);
}
#endif /* ST_NEON_DOTPROD */

/* 32 int8 (Q8_0 block) -> 8 float32x4 lanes (elements 0-3, 4-7, ... 28-31). */
static inline void i8x32_to_f32x8(const int8_t *qs, float32x4_t f[8]) {
    int8x16_t a = vld1q_s8(qs);
    int8x16_t b = vld1q_s8(qs + 16);
    int16x8_t al = vmovl_s8(vget_low_s8(a));
    int16x8_t ah = vmovl_s8(vget_high_s8(a));
    int16x8_t bl = vmovl_s8(vget_low_s8(b));
    int16x8_t bh = vmovl_s8(vget_high_s8(b));
    f[0] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(al)));
    f[1] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(al)));
    f[2] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(ah)));
    f[3] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(ah)));
    f[4] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(bl)));
    f[5] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(bl)));
    f[6] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(bh)));
    f[7] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(bh)));
}

/* Sum 8 fp32 lanes -> scalar (pairwise, deterministic order). */
static inline float f32x8_hsum(const float32x4_t f[8]) {
    float32x4_t a = vaddq_f32(vaddq_f32(f[0], f[1]), vaddq_f32(f[2], f[3]));
    float32x4_t b = vaddq_f32(vaddq_f32(f[4], f[5]), vaddq_f32(f[6], f[7]));
    return hsum_neon4(vaddq_f32(a, b));
}

/* Forward declarations of helpers defined below this section. */
static inline float q8_block_scale(const uint8_t *b);
static inline float q4_block_scale(const uint8_t *b);
static void quantize_row_q8_0_act(const float *__restrict x,
                                  int8_t *__restrict q, float *__restrict d, int cols);
static void quantize_row_q8_0_act_bm(const float *__restrict x,
                                     int8_t *__restrict q, float *__restrict d,
                                     int cols, int n_batch);
static void *xq_alloc_canary(size_t bytes);
static void xq_free_canary(void *p, size_t bytes, const char *tag);

/* Reusable kernel scratch (xq/xd). Grows on demand, never freed mid-run;
 * safe because GEMM kernels are entered serially (layer loop) and allocate
 * before their omp region. Mirrors llama.cpp's preallocated workspaces. */
static int8_t *g_xq_scratch = NULL;
static float  *g_xd_scratch = NULL;
static size_t  g_xq_scratch_cap = 0;
static size_t  g_xd_scratch_cap = 0;

static int xq_scratch_ensure(int8_t **xq, float **xd,
                             size_t nq_bytes, size_t nd_bytes) {
    if (g_xq_scratch_cap < nq_bytes) {
        int8_t *p = (int8_t *)realloc(g_xq_scratch, nq_bytes);
        if (!p) return -1;
        g_xq_scratch = p;
        g_xq_scratch_cap = nq_bytes;
    }
    if (g_xd_scratch_cap < nd_bytes) {
        float *p = (float *)realloc(g_xd_scratch, nd_bytes);
        if (!p) return -1;
        g_xd_scratch = p;
        g_xd_scratch_cap = nd_bytes;
    }
    *xq = g_xq_scratch;
    *xd = g_xd_scratch;
    return 0;
}

/* ================================================================
 * M4e: Q8_0 4x4 kernels (llama.cpp block_q8_0x4, 136 B/block)
 * ================================================================
 * 2026-08-22: prefill Q8 差距 3.18x 的根因与 Q4 相同（batched GEMV 每 token
 * tile 重读权重 vs llama 的 4x4 交织 GEMM 权重读一次）。权重在加载期由
 * repack_q8_0_4x4_inplace 重排为 block_q8_0x4（{ d[4] f16, qs[128] }，
 * qs[k*16 + m*4 + i] = 行 m 的 (k*4+i)）；激活在 prefill 由 repack_q8_0_4x4
 * 重排（M4d 已实现）。以下两个原语为 llama.cpp ggml_gemv/gemm_q8_0_4x4_q8_0
 * NEON 段逐字移植。
 *
 * 数值语义：vdot 的 int32 精确累加 + 单次 fma（scale 乘积）与 legacy 逐块
 * 路径舍入顺序不同 -> 位级一致放宽，PPL 需重新验收（流程与 M4d 相同）。 */

/* decode 原语：1 个 136B 权重组（4 行）x 1 个 32-int8 激活块。交织布局下
 * vdotq_laneq_s32 的 lane k 恰好命中行 k：bl.val[k] 的 4 个 int8 段 = 行
 * 0..3 的 4k..4k+3，故 ret.lane[m] = 行 m 的 32 元素 int32 点积；
 * acc.lane[m] += ret.lane[m] * d[m] * act_scale（4 行并行、无 hsum）。 */
static inline void q8x4_dot1_group(const uint8_t *__restrict bq,
                                   const int8_t *__restrict xq_b,
                                   float act_scale,
                                   float32x4_t *acc) {
    int8x16x4_t bl = vld1q_s8_x4((const int8_t *)(bq + 8));
    int8x16x4_t bh = vld1q_s8_x4((const int8_t *)(bq + 8 + 64));
    int8x16_t a0 = vld1q_s8(xq_b);
    int8x16_t a1 = vld1q_s8(xq_b + 16);
    int32x4_t ret = vdupq_n_s32(0);
    ret = vdotq_laneq_s32(ret, bl.val[0], a0, 0);
    ret = vdotq_laneq_s32(ret, bl.val[1], a0, 1);
    ret = vdotq_laneq_s32(ret, bl.val[2], a0, 2);
    ret = vdotq_laneq_s32(ret, bl.val[3], a0, 3);
    ret = vdotq_laneq_s32(ret, bh.val[0], a1, 0);
    ret = vdotq_laneq_s32(ret, bh.val[1], a1, 1);
    ret = vdotq_laneq_s32(ret, bh.val[2], a1, 2);
    ret = vdotq_laneq_s32(ret, bh.val[3], a1, 3);
    float32x4_t wd = vcvt_f32_f16(vld1_f16((const __fp16 *)bq));
    float32x4_t dsv = vmulq_f32(wd, vdupq_n_f32(act_scale));
    *acc = vfmaq_f32(*acc, vcvtq_f32_s32(ret), dsv);
}

/* prefill GEMM：llama ggml_gemm_q8_0_4x4_q8_0 NEON 段逐字移植
 * （源自 llama.cpp，MIT (c) 2023-2026 The ggml authors，全文见本文件头）。
 *   s[(y*4+m)*bs + x*4 + c] = sum_k W(x*4+c, k) * A(y*4+m, k)
 * 我方映射（与 M4d asm 一致）：nr = n_batch（y = token 组）、nc = rows（x =
 * 权重行组）、bs = rows、vx = 权重 4x4 repack、vy = 激活 4x4 repack
 * （g_q8r_scratch）。输出 out[token*rows + row]。 */
static void llama_gemm_q8_0_4x4_q8_0_neon(int n, float *__restrict s, size_t bs,
                                          const uint8_t *__restrict vx,
                                          const uint8_t *__restrict vy,
                                          int nr, int nc) {
    const int nb = n / 32;
    for (int y = 0; y < nr / 4; y++) {
        const uint8_t *__restrict a_ptr = vy + (size_t)y * nb * 136;
        for (int x = 0; x < nc / 4; x++) {
            const uint8_t *__restrict b_ptr = vx + (size_t)x * nb * 136;
            float32x4_t sumf[4];
            for (int m = 0; m < 4; m++) sumf[m] = vdupq_n_f32(0);
            for (int l = 0; l < nb; l++) {
                float32x4_t a_d = vcvt_f32_f16(vld1_f16((const float16_t *)(a_ptr + (size_t)l * 136)));
                float32x4_t b_d = vcvt_f32_f16(vld1_f16((const float16_t *)(b_ptr + (size_t)l * 136)));
                int32x4_t sumi_0 = vdupq_n_s32(0);
                int32x4_t sumi_1 = vdupq_n_s32(0);
                int32x4_t sumi_2 = vdupq_n_s32(0);
                int32x4_t sumi_3 = vdupq_n_s32(0);
                const int8_t *__restrict aqs = (const int8_t *)(a_ptr + (size_t)l * 136 + 8);
                const int8_t *__restrict bqs = (const int8_t *)(b_ptr + (size_t)l * 136 + 8);
                for (int k_group = 0; k_group < 8; k_group += 4) {
                    int8x16x4_t a = vld1q_s8_x4(aqs + 16 * k_group);
                    int8x16x4_t b = vld1q_s8_x4(bqs + 16 * k_group);
                    for (int k = 0; k < 4; k++) {
                        sumi_0 = vdotq_laneq_s32(sumi_0, b.val[k], a.val[k], 0);
                        sumi_1 = vdotq_laneq_s32(sumi_1, b.val[k], a.val[k], 1);
                        sumi_2 = vdotq_laneq_s32(sumi_2, b.val[k], a.val[k], 2);
                        sumi_3 = vdotq_laneq_s32(sumi_3, b.val[k], a.val[k], 3);
                    }
                }
                sumf[0] = vmlaq_f32(sumf[0], vmulq_laneq_f32(b_d, a_d, 0), vcvtq_f32_s32(sumi_0));
                sumf[1] = vmlaq_f32(sumf[1], vmulq_laneq_f32(b_d, a_d, 1), vcvtq_f32_s32(sumi_1));
                sumf[2] = vmlaq_f32(sumf[2], vmulq_laneq_f32(b_d, a_d, 2), vcvtq_f32_s32(sumi_2));
                sumf[3] = vmlaq_f32(sumf[3], vmulq_laneq_f32(b_d, a_d, 3), vcvtq_f32_s32(sumi_3));
            }
            for (int m = 0; m < 4; m++)
                vst1q_f32(s + (size_t)(y * 4 + m) * bs + x * 4, sumf[m]);
        }
    }
}

/* omp 包装：16-token 分块调 llama GEMM（每块 nr<=16），residual 在 GEMM 后
 * 向量化后加（与 C 路径 out = residual + matmul 语义一致）。调用方须保证
 * n_batch % 4 == 0 且 g_q8r_scratch 已 repack 本批激活。 */
typedef struct {
    float *out; const uint8_t *q8_w;
    int nb, cols, rows, n_batch;
} vllm_q8x4_b0_ctx;

static void vllm_q8x4_b0_worker(void *ctx_, int it) {
    vllm_q8x4_b0_ctx *c = ctx_;
    int t0 = it * 16;
    int nt = c->n_batch - t0;
    if (nt > 16) nt = 16;
    llama_gemm_q8_0_4x4_q8_0_neon(c->cols, c->out + (size_t)t0 * c->rows, (size_t)c->rows,
                                  c->q8_w, g_q8r_scratch + (size_t)(t0 >> 2) * c->nb * 136,
                                  nt, c->rows);
}

typedef struct {
    float *out; const float *residual;
    int rows, n_batch;
} vllm_q8x4_res_ctx;

static void vllm_q8x4_res_worker(void *ctx_, int t) {
    vllm_q8x4_res_ctx *c = ctx_;
    float *o = c->out + (size_t)t * c->rows;
    const float *r = c->residual + (size_t)t * c->rows;
    for (int r4 = 0; r4 < c->rows; r4 += 4) {
        float32x4_t ov = vld1q_f32(o + r4);
        float32x4_t rv = vld1q_f32(r + r4);
        vst1q_f32(o + r4, vaddq_f32(rv, ov));
    }
}

static void st_gemm_q8_0_4x4_batched(float *__restrict out,
                                     const uint8_t *__restrict q8_w,
                                     int rows, int cols, int n_batch,
                                     const float *__restrict residual) {
    int nb = cols / 32;
    vllm_q8x4_b0_ctx vc = { out, q8_w, nb, cols, rows, n_batch };
    vllm_tp_parfor(0, (n_batch + 15) / 16, vllm_q8x4_b0_worker, &vc);
    if (residual) {
        vllm_q8x4_res_ctx vr = { out, residual, rows, n_batch };
        vllm_tp_parfor(0, n_batch, vllm_q8x4_res_worker, &vr);
    }
}

/* q8x4 GEMM 包装：token-major 量化激活（xq[t*cols] int8 + xd[t*nb] f32）→
 * 激活 repack 为 block_q8_0x4（g_q8r_scratch）→ 16-token 分块调 llama GEMM。
 * 约束：n_batch % 4 == 0（否则调用方走 legacy C 路径）。residual 非空时
 * 输出 out = residual + matmul（fused 语义）。激活 repack 失败（realloc
 * OOM）时回退逐 token 4x4 GEMV 兜底 - 正确性优先，永不向调用方暴露失败。 */
typedef struct {
    float *out; const uint8_t *q8_w; const int8_t *xq; const float *xd;
    const float *residual; int nb, rows, cols, n_batch;
} vllm_q8x4_gemv_ctx;

static void vllm_q8x4_gemv_worker(void *ctx_, int t) {
    vllm_q8x4_gemv_ctx *c = ctx_;
    float *o = c->out + (size_t)t * c->rows;
    const int8_t *__restrict xqt = c->xq + (size_t)t * c->cols;
    const float *__restrict xdt = c->xd + (size_t)t * c->nb;
    for (int r4 = 0; r4 < c->rows; r4 += 4) {
        const uint8_t *__restrict pr = c->q8_w + (size_t)(r4 >> 2) * c->nb * 136;
        float32x4_t acc = vdupq_n_f32(0.0f);
        for (int b = 0; b < c->nb; b++)
            q8x4_dot1_group(pr + (size_t)b * 136, xqt + (size_t)b * 32, xdt[b], &acc);
        vst1q_f32(o + r4, acc);
    }
    if (c->residual) {
        const float *__restrict r = c->residual + (size_t)t * c->rows;
        for (int r4 = 0; r4 < c->rows; r4 += 4) {
            float32x4_t ov = vld1q_f32(o + r4);
            float32x4_t rv = vld1q_f32(r + r4);
            vst1q_f32(o + r4, vaddq_f32(rv, ov));
        }
    }
}

static void q8x8_gemm_batched(float *__restrict out, const uint8_t *__restrict q8_w,
                              const int8_t *__restrict xq, const float *__restrict xd,
                              int rows, int cols, int n_batch,
                              const float *__restrict residual);

static void q8x4_gemm_batched(float *__restrict out, const uint8_t *__restrict q8_w,
                              const int8_t *__restrict xq, const float *__restrict xd,
                              int rows, int cols, int n_batch,
                              const float *__restrict residual) {
    if (q8_8x8_enabled()) {
        q8x8_gemm_batched(out, q8_w, xq, xd, rows, cols, n_batch, residual);
        return;
    }
    if ((n_batch & 3) == 0 && repack_q8_0_4x4(xq, xd, n_batch, cols) == 0) {
        st_gemm_q8_0_4x4_batched(out, q8_w, rows, cols, n_batch, residual);
        return;
    }
    /* 逐 token 4x4 GEMV 回退（n_batch%4!=0 或激活 repack OOM）。正确性优先；
     * vdot 语义与 legacy 舍入顺序不同（非主路径，主路径为真 GEMM）。 */
    {
        int nb = cols / 32;
        vllm_q8x4_gemv_ctx vc = { out, q8_w, xq, xd, residual, nb, rows, cols, n_batch };
        vllm_tp_parfor(0, n_batch, vllm_q8x4_gemv_worker, &vc);
        return;
    }
}

/* decode 快速路径：单矩阵 4x4 vdot GEMV（llama ggml_gemv_q8_0_4x4_q8_0
 * 语义）。激活先量化成 legacy Q8_0（34B/block int8 + f32 scale），权重组
 * q8x4_dot1_group 用 vdotq_laneq 4 行并行——无 gather、无 spill（单 f32x4
 * 累加器）、权重只读 1 遍。
 * 2026-08-22 实测取舍：f32 激活的解交织版本（acc[4][8]=32 累加器）在 A76
 * 上完全 spill（TPOT 1.73s/tok），串行 4 行版把权重读 4 遍（TPOT 924ms）——
 * 均比 legacy 慢 5x+。vdot 版以激活量化（Q8_0 8bit，llama 同语义）换取
 * 性能，PPL 需重新验收（流程与 M4d 相同）。residual 非空时 out = res+matmul。 */
typedef struct {
    float *out; const uint8_t *q8_w; const int8_t *xq; const float *xd;
    const float *residual; int n_blocks, rows;
} vllm_q8x4_vdot_ctx;

static void vllm_q8x4_vdot_worker(void *ctx_, int it) {
    vllm_q8x4_vdot_ctx *c = ctx_;
    int r4 = it * 4;
    const uint8_t *__restrict pr = c->q8_w + (size_t)(r4 >> 2) * c->n_blocks * 136;
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (int b = 0; b < c->n_blocks; b++)
        q8x4_dot1_group(pr + (size_t)b * 136, c->xq + (size_t)b * 32, c->xd[b], &acc);
    const float rv[4] = {
        vgetq_lane_f32(acc, 0), vgetq_lane_f32(acc, 1),
        vgetq_lane_f32(acc, 2), vgetq_lane_f32(acc, 3) };
    for (int m = 0; m < 4; m++)
        c->out[r4 + m] = c->residual ? c->residual[r4 + m] + rv[m] : rv[m];
}

static void q8x8_matvec_vdot(float *__restrict out, const uint8_t *__restrict q8_w,
                             const int8_t *__restrict xq, const float *__restrict xd,
                             int rows, int cols, const float *__restrict residual);

static void q8x4_matvec_vdot(float *__restrict out, const uint8_t *__restrict q8_w,
                             const int8_t *__restrict xq, const float *__restrict xd,
                             int rows, int cols, const float *__restrict residual) {
    if (q8_8x8_enabled()) {
        q8x8_matvec_vdot(out, q8_w, xq, xd, rows, cols, residual);
        return;
    }
    int n_blocks = cols / 32;
    vllm_q8x4_vdot_ctx vc = { out, q8_w, xq, xd, residual, n_blocks, rows };
    vllm_tp_parfor(0, (rows + 3) / 4, vllm_q8x4_vdot_worker, &vc);
}

/* ================================================================
 * Q8_0 8x8 tile kernels (block_q8_0x8, 272 B/block) - P0 tile upgrade.
 * Activation repack: token-major xq[t][cols] + xd[t][nb] -> 8x8 groups in
 * g_q8r_scratch (zero-padded partial 8-token group; d=0 -> padded outputs 0).
 * ================================================================ */
static int repack_q8_0_8x8_act(const int8_t *__restrict xq, const float *__restrict xd,
                               int n_batch, int cols) {
    int nb = cols / 32;
    size_t nbytes = (size_t)((n_batch + 7) / 8) * nb * 272;
    if (g_q8r_scratch_cap < nbytes) {
        uint8_t *np = (uint8_t *)realloc(g_q8r_scratch, nbytes);
        if (!np) return -1;
        g_q8r_scratch = np;
        g_q8r_scratch_cap = nbytes;
    }
    uint8_t *vy = g_q8r_scratch;
    for (int t0 = 0; t0 < n_batch; t0 += 8) {
        int nt = n_batch - t0; if (nt > 8) nt = 8;
        for (int b = 0; b < nb; b++) {
            uint8_t *out = vy + (size_t)(t0 >> 3) * nb * 272 + (size_t)b * 272;
            memset(out, 0, 272);   /* zero-pad partial group (d=0 -> outputs 0) */
            for (int m = 0; m < nt; m++) {
                uint16_t d = f32_to_f16_bits(xd[(size_t)(t0 + m) * nb + b]);
                memcpy(out + (size_t)m * 2, &d, 2);
            }
            for (int k = 0; k < 8; k++)
                for (int m = 0; m < nt; m++)
                    memcpy(out + 16 + k * 32 + m * 4,
                           xq + (size_t)(t0 + m) * cols + (size_t)b * 32 + k * 4, 4);
        }
    }
    return 0;
}

/* 8x8 tile: 8 tokens x 8 rows. 16 int32 accs in registers, fp32 tail in the
 * L1-resident sumf[2][8]. Weight block read ONCE per 8-token group (4x4 reads
 * once per 4) -> weight DRAM traffic halved. Per-output order identical to the
 * 4x4 kernel => bit-identical results (verified on board 2026-08-29). */
static void gemm_q8_0_8x8_neon(int n, float *__restrict s, size_t bs,
                               const uint8_t *__restrict vx,
                               const uint8_t *__restrict vy,
                               int nr, int nc) {
    const int nb = n / 32;
    for (int y = 0; y < (nr + 7) / 8; y++) {
        const uint8_t *a_ptr = vy + (size_t)y * nb * 272;
        for (int x = 0; x < nc / 8; x++) {
            const uint8_t *b_ptr = vx + (size_t)x * nb * 272;
            float32x4_t sumf[2][8];
            for (int m = 0; m < 2; m++)
                for (int t = 0; t < 8; t++) sumf[m][t] = vdupq_n_f32(0.0f);
            for (int l = 0; l < nb; l++) {
                const uint8_t *ablk = a_ptr + (size_t)l * 272;
                const uint8_t *bblk = b_ptr + (size_t)l * 272;
                float32x4_t ad0 = vcvt_f32_f16(vld1_f16((const float16_t *)ablk));
                float32x4_t ad1 = vcvt_f32_f16(vld1_f16((const float16_t *)(ablk + 8)));
                float32x4_t bd0 = vcvt_f32_f16(vld1_f16((const float16_t *)bblk));
                float32x4_t bd1 = vcvt_f32_f16(vld1_f16((const float16_t *)(bblk + 8)));
                const int8_t *aqs = (const int8_t *)(ablk + 16);
                const int8_t *bqs = (const int8_t *)(bblk + 16);
                int32x4_t acc00 = vdupq_n_s32(0), acc01 = vdupq_n_s32(0), acc02 = vdupq_n_s32(0), acc03 = vdupq_n_s32(0);
                int32x4_t acc04 = vdupq_n_s32(0), acc05 = vdupq_n_s32(0), acc06 = vdupq_n_s32(0), acc07 = vdupq_n_s32(0);
                int32x4_t acc10 = vdupq_n_s32(0), acc11 = vdupq_n_s32(0), acc12 = vdupq_n_s32(0), acc13 = vdupq_n_s32(0);
                int32x4_t acc14 = vdupq_n_s32(0), acc15 = vdupq_n_s32(0), acc16 = vdupq_n_s32(0), acc17 = vdupq_n_s32(0);
                for (int k = 0; k < 8; k++) {
                    int8x16_t a0 = vld1q_s8(aqs + k * 32);
                    int8x16_t a1 = vld1q_s8(aqs + k * 32 + 16);
                    int8x16_t b0 = vld1q_s8(bqs + k * 32);
                    int8x16_t b1 = vld1q_s8(bqs + k * 32 + 16);
                    acc00 = vdotq_laneq_s32(acc00, b0, a0, 0);
                    acc01 = vdotq_laneq_s32(acc01, b0, a0, 1);
                    acc02 = vdotq_laneq_s32(acc02, b0, a0, 2);
                    acc03 = vdotq_laneq_s32(acc03, b0, a0, 3);
                    acc04 = vdotq_laneq_s32(acc04, b0, a1, 0);
                    acc05 = vdotq_laneq_s32(acc05, b0, a1, 1);
                    acc06 = vdotq_laneq_s32(acc06, b0, a1, 2);
                    acc07 = vdotq_laneq_s32(acc07, b0, a1, 3);
                    acc10 = vdotq_laneq_s32(acc10, b1, a0, 0);
                    acc11 = vdotq_laneq_s32(acc11, b1, a0, 1);
                    acc12 = vdotq_laneq_s32(acc12, b1, a0, 2);
                    acc13 = vdotq_laneq_s32(acc13, b1, a0, 3);
                    acc14 = vdotq_laneq_s32(acc14, b1, a1, 0);
                    acc15 = vdotq_laneq_s32(acc15, b1, a1, 1);
                    acc16 = vdotq_laneq_s32(acc16, b1, a1, 2);
                    acc17 = vdotq_laneq_s32(acc17, b1, a1, 3);
                }
                sumf[0][0] = vmlaq_f32(sumf[0][0], vmulq_laneq_f32(bd0, ad0, 0), vcvtq_f32_s32(acc00));
                sumf[0][1] = vmlaq_f32(sumf[0][1], vmulq_laneq_f32(bd0, ad0, 1), vcvtq_f32_s32(acc01));
                sumf[0][2] = vmlaq_f32(sumf[0][2], vmulq_laneq_f32(bd0, ad0, 2), vcvtq_f32_s32(acc02));
                sumf[0][3] = vmlaq_f32(sumf[0][3], vmulq_laneq_f32(bd0, ad0, 3), vcvtq_f32_s32(acc03));
                sumf[0][4] = vmlaq_f32(sumf[0][4], vmulq_laneq_f32(bd0, ad1, 0), vcvtq_f32_s32(acc04));
                sumf[0][5] = vmlaq_f32(sumf[0][5], vmulq_laneq_f32(bd0, ad1, 1), vcvtq_f32_s32(acc05));
                sumf[0][6] = vmlaq_f32(sumf[0][6], vmulq_laneq_f32(bd0, ad1, 2), vcvtq_f32_s32(acc06));
                sumf[0][7] = vmlaq_f32(sumf[0][7], vmulq_laneq_f32(bd0, ad1, 3), vcvtq_f32_s32(acc07));
                sumf[1][0] = vmlaq_f32(sumf[1][0], vmulq_laneq_f32(bd1, ad0, 0), vcvtq_f32_s32(acc10));
                sumf[1][1] = vmlaq_f32(sumf[1][1], vmulq_laneq_f32(bd1, ad0, 1), vcvtq_f32_s32(acc11));
                sumf[1][2] = vmlaq_f32(sumf[1][2], vmulq_laneq_f32(bd1, ad0, 2), vcvtq_f32_s32(acc12));
                sumf[1][3] = vmlaq_f32(sumf[1][3], vmulq_laneq_f32(bd1, ad0, 3), vcvtq_f32_s32(acc13));
                sumf[1][4] = vmlaq_f32(sumf[1][4], vmulq_laneq_f32(bd1, ad1, 0), vcvtq_f32_s32(acc14));
                sumf[1][5] = vmlaq_f32(sumf[1][5], vmulq_laneq_f32(bd1, ad1, 1), vcvtq_f32_s32(acc15));
                sumf[1][6] = vmlaq_f32(sumf[1][6], vmulq_laneq_f32(bd1, ad1, 2), vcvtq_f32_s32(acc16));
                sumf[1][7] = vmlaq_f32(sumf[1][7], vmulq_laneq_f32(bd1, ad1, 3), vcvtq_f32_s32(acc17));
            }
            for (int t = 0; t < 8; t++) {
                if (y * 8 + t >= nr) break;   /* partial group: 只写真实 token 行 */
                float *op = s + (size_t)(y * 8 + t) * bs + x * 8;
                vst1q_f32(op, sumf[0][t]);
                vst1q_f32(op + 4, sumf[1][t]);
            }
        }
    }
}

/* 8x4 tail tile: 8 rows x 4 tokens (n_batch % 8 == 4 last group). */
static void gemm_q8_0_8x4_neon(int n, float *__restrict s, size_t bs,
                               const uint8_t *__restrict vx,
                               const uint8_t *__restrict vy,
                               int nr, int nc) {
    const int nb = n / 32;
    for (int y = 0; y < nr / 4; y++) {
        const uint8_t *a_ptr = vy + (size_t)y * nb * 272;
        for (int x = 0; x < nc / 8; x++) {
            const uint8_t *b_ptr = vx + (size_t)x * nb * 272;
            float32x4_t sumf[2][4];
            for (int m = 0; m < 2; m++)
                for (int t = 0; t < 4; t++) sumf[m][t] = vdupq_n_f32(0.0f);
            for (int l = 0; l < nb; l++) {
                const uint8_t *ablk = a_ptr + (size_t)l * 272;
                const uint8_t *bblk = b_ptr + (size_t)l * 272;
                float32x4_t ad0 = vcvt_f32_f16(vld1_f16((const float16_t *)ablk));
                float32x4_t bd0 = vcvt_f32_f16(vld1_f16((const float16_t *)bblk));
                float32x4_t bd1 = vcvt_f32_f16(vld1_f16((const float16_t *)(bblk + 8)));
                const int8_t *aqs = (const int8_t *)(ablk + 16);
                const int8_t *bqs = (const int8_t *)(bblk + 16);
                int32x4_t acc00 = vdupq_n_s32(0), acc01 = vdupq_n_s32(0), acc02 = vdupq_n_s32(0), acc03 = vdupq_n_s32(0);
                int32x4_t acc10 = vdupq_n_s32(0), acc11 = vdupq_n_s32(0), acc12 = vdupq_n_s32(0), acc13 = vdupq_n_s32(0);
                for (int k = 0; k < 8; k++) {
                    int8x16_t a0 = vld1q_s8(aqs + k * 32);
                    int8x16_t b0 = vld1q_s8(bqs + k * 32);
                    int8x16_t b1 = vld1q_s8(bqs + k * 32 + 16);
                    acc00 = vdotq_laneq_s32(acc00, b0, a0, 0);
                    acc01 = vdotq_laneq_s32(acc01, b0, a0, 1);
                    acc02 = vdotq_laneq_s32(acc02, b0, a0, 2);
                    acc03 = vdotq_laneq_s32(acc03, b0, a0, 3);
                    acc10 = vdotq_laneq_s32(acc10, b1, a0, 0);
                    acc11 = vdotq_laneq_s32(acc11, b1, a0, 1);
                    acc12 = vdotq_laneq_s32(acc12, b1, a0, 2);
                    acc13 = vdotq_laneq_s32(acc13, b1, a0, 3);
                }
                sumf[0][0] = vmlaq_f32(sumf[0][0], vmulq_laneq_f32(bd0, ad0, 0), vcvtq_f32_s32(acc00));
                sumf[0][1] = vmlaq_f32(sumf[0][1], vmulq_laneq_f32(bd0, ad0, 1), vcvtq_f32_s32(acc01));
                sumf[0][2] = vmlaq_f32(sumf[0][2], vmulq_laneq_f32(bd0, ad0, 2), vcvtq_f32_s32(acc02));
                sumf[0][3] = vmlaq_f32(sumf[0][3], vmulq_laneq_f32(bd0, ad0, 3), vcvtq_f32_s32(acc03));
                sumf[1][0] = vmlaq_f32(sumf[1][0], vmulq_laneq_f32(bd1, ad0, 0), vcvtq_f32_s32(acc10));
                sumf[1][1] = vmlaq_f32(sumf[1][1], vmulq_laneq_f32(bd1, ad0, 1), vcvtq_f32_s32(acc11));
                sumf[1][2] = vmlaq_f32(sumf[1][2], vmulq_laneq_f32(bd1, ad0, 2), vcvtq_f32_s32(acc12));
                sumf[1][3] = vmlaq_f32(sumf[1][3], vmulq_laneq_f32(bd1, ad0, 3), vcvtq_f32_s32(acc13));
            }
            for (int t = 0; t < 4; t++) {
                float *op = s + (size_t)(y * 4 + t) * bs + x * 8;
                vst1q_f32(op, sumf[0][t]);
                vst1q_f32(op + 4, sumf[1][t]);
            }
        }
    }
}

/* 8x8 batched wrapper: 8-token chunks, 8x8 GEMM (8x4 for the 4-token tail).
 * Row-slicing was A/B'd 2026-08-29 (VLLM_ROW_SLICE=0/384/2048, 3-sample means):
 * no net gain over the plain 8-token chunking (~-0.6% within noise) - the
 * 8x8 tile already halves the weight re-reads vs 4x4, so the Q4 4x4 asm's
 * row-slice L3-residency win does NOT transfer here. Keep it simple. */
typedef struct {
    float *out; const uint8_t *q8_w;
    int nb, cols, rows, n_batch;
} vllm_q8x8_b0_ctx;

static void vllm_q8x8_b0_worker(void *ctx_, int it) {
    vllm_q8x8_b0_ctx *c = ctx_;
    int t0 = it * 8;
    int nt = c->n_batch - t0;
    if (nt > 8) nt = 8;
    const uint8_t *act = g_q8r_scratch + (size_t)it * c->nb * 272;
    if (nt == 4) {
        gemm_q8_0_8x4_neon(c->cols, c->out + (size_t)t0 * c->rows, (size_t)c->rows,
                           c->q8_w, act, nt, c->rows);
    } else {
        gemm_q8_0_8x8_neon(c->cols, c->out + (size_t)t0 * c->rows, (size_t)c->rows,
                           c->q8_w, act, nt, c->rows);
    }
}

static void st_gemm_q8_0_8x8_batched(float *__restrict out,
                                     const uint8_t *__restrict q8_w,
                                     int rows, int cols, int n_batch,
                                     const float *__restrict residual) {
    int nb = cols / 32;
    vllm_q8x8_b0_ctx vc = { out, q8_w, nb, cols, rows, n_batch };
    vllm_tp_parfor(0, (n_batch + 7) / 8, vllm_q8x8_b0_worker, &vc);
    if (residual) {
        vllm_q8x4_res_ctx vr = { out, residual, rows, n_batch };
        vllm_tp_parfor(0, n_batch, vllm_q8x4_res_worker, &vr);
    }
    /* 8x8 batched GEMM 自检（VLLM_Q8CHK=5）：token 0 前 32 行 vs 参考解包 */
    if (getenv("VLLM_Q8CHK") && getenv("VLLM_Q8CHK")[0] == '5') {
        static int chk_cnt = 0;
        if (chk_cnt < 60) {
            chk_cnt++;
            int bad = 0;
            double maxe = 0;
            const uint8_t *act = g_q8r_scratch;   /* it=0 的 8-token 激活组 */
            for (int r = 0; r < rows && r < 64; r++) {
                const uint8_t *grp = q8_w + (size_t)(r >> 3) * nb * 272;
                int mm = r & 7;
                double acc = 0;
                for (int b = 0; b < nb; b++) {
                    const uint8_t *ablk = act + (size_t)b * 272;
                    const uint8_t *wblk = grp + (size_t)b * 272;
                    uint32_t fba = f16_to_f32_bits(*(const uint16_t *)ablk);
                    uint32_t fbw = f16_to_f32_bits(*(const uint16_t *)(wblk + (size_t)mm * 2));
                    float da, dw;
                    memcpy(&da, &fba, 4);
                    memcpy(&dw, &fbw, 4);
                    double s = 0;
                    for (int k = 0; k < 8; k++)
                        for (int i = 0; i < 4; i++)
                            s += (double)(int8_t)wblk[16 + k * 32 + mm * 4 + i] *
                                 (double)(int8_t)ablk[16 + k * 32 + i];
                    acc += s * dw * da;
                }
                double got = out[r] - (residual ? residual[r] : 0.0f);
                double e = fabs(got - acc);
                if (e > 1e-3 && fabs(e / acc) > 1e-4) bad++;
                if (e > maxe) maxe = e;
            }
            if (bad || chk_cnt <= 2)
                fprintf(stderr, "[Q8CHK5#%d] batched 8x8 rows=%d cols=%d n_batch=%d nb=%d bad=%d maxe=%.3f out[0]=%.4f\n",
                        chk_cnt, rows, cols, n_batch, nb, bad, maxe, out[0]);
            fflush(stderr);
        }
    }
}

/* decode GEMV (8 rows/iter) + per-token GEMV fallback, 8x8 layout. */
typedef struct {
    float *out; const uint8_t *q8_w; const int8_t *xq; const float *xd;
    const float *residual; int n_blocks, rows;
} vllm_q8x8_vdot_ctx;

static inline void q8x8_dot_rows(float *__restrict out, int r8,
                                 const uint8_t *__restrict pr,
                                 const int8_t *__restrict xq, const float *__restrict xd,
                                 int n_blocks, const float *__restrict residual) {
    float32x4_t sumf0 = vdupq_n_f32(0.0f), sumf1 = vdupq_n_f32(0.0f);
    for (int b = 0; b < n_blocks; b++) {
        const uint8_t *bq = pr + (size_t)b * 272;
        const int8_t *qs = (const int8_t *)(bq + 16);
        int8x16_t a0v = vld1q_s8(xq + (size_t)b * 32);
        int8x16_t a1v = vld1q_s8(xq + (size_t)b * 32 + 16);
        int32x4_t acc0 = vdupq_n_s32(0), acc1 = vdupq_n_s32(0);
        int8x16_t b0 = vld1q_s8(qs + 0),      b1 = vld1q_s8(qs + 16);
        acc0 = vdotq_laneq_s32(acc0, b0, a0v, 0); acc1 = vdotq_laneq_s32(acc1, b1, a0v, 0);
        b0 = vld1q_s8(qs + 32);       b1 = vld1q_s8(qs + 48);
        acc0 = vdotq_laneq_s32(acc0, b0, a0v, 1); acc1 = vdotq_laneq_s32(acc1, b1, a0v, 1);
        b0 = vld1q_s8(qs + 64);       b1 = vld1q_s8(qs + 80);
        acc0 = vdotq_laneq_s32(acc0, b0, a0v, 2); acc1 = vdotq_laneq_s32(acc1, b1, a0v, 2);
        b0 = vld1q_s8(qs + 96);       b1 = vld1q_s8(qs + 112);
        acc0 = vdotq_laneq_s32(acc0, b0, a0v, 3); acc1 = vdotq_laneq_s32(acc1, b1, a0v, 3);
        b0 = vld1q_s8(qs + 128);      b1 = vld1q_s8(qs + 144);
        acc0 = vdotq_laneq_s32(acc0, b0, a1v, 0); acc1 = vdotq_laneq_s32(acc1, b1, a1v, 0);
        b0 = vld1q_s8(qs + 160);      b1 = vld1q_s8(qs + 176);
        acc0 = vdotq_laneq_s32(acc0, b0, a1v, 1); acc1 = vdotq_laneq_s32(acc1, b1, a1v, 1);
        b0 = vld1q_s8(qs + 192);      b1 = vld1q_s8(qs + 208);
        acc0 = vdotq_laneq_s32(acc0, b0, a1v, 2); acc1 = vdotq_laneq_s32(acc1, b1, a1v, 2);
        b0 = vld1q_s8(qs + 224);      b1 = vld1q_s8(qs + 240);
        acc0 = vdotq_laneq_s32(acc0, b0, a1v, 3); acc1 = vdotq_laneq_s32(acc1, b1, a1v, 3);
        float32x4_t wd0 = vcvt_f32_f16(vld1_f16((const float16_t *)bq));
        float32x4_t wd1 = vcvt_f32_f16(vld1_f16((const float16_t *)(bq + 8)));
        float32x4_t dsv0 = vmulq_f32(wd0, vdupq_n_f32(xd[b]));
        float32x4_t dsv1 = vmulq_f32(wd1, vdupq_n_f32(xd[b]));
        sumf0 = vfmaq_f32(sumf0, vcvtq_f32_s32(acc0), dsv0);
        sumf1 = vfmaq_f32(sumf1, vcvtq_f32_s32(acc1), dsv1);
    }
    const float rv0[4] = { vgetq_lane_f32(sumf0, 0), vgetq_lane_f32(sumf0, 1),
                           vgetq_lane_f32(sumf0, 2), vgetq_lane_f32(sumf0, 3) };
    const float rv1[4] = { vgetq_lane_f32(sumf1, 0), vgetq_lane_f32(sumf1, 1),
                           vgetq_lane_f32(sumf1, 2), vgetq_lane_f32(sumf1, 3) };
    for (int m = 0; m < 4; m++)
        out[r8 + m] = residual ? residual[r8 + m] + rv0[m] : rv0[m];
    for (int m = 0; m < 4; m++)
        out[r8 + 4 + m] = residual ? residual[r8 + 4 + m] + rv1[m] : rv1[m];
}

static void vllm_q8x8_vdot_worker(void *ctx_, int it) {
    vllm_q8x8_vdot_ctx *c = ctx_;
    int r8 = it * 8;
    if (r8 >= c->rows) return;
    const uint8_t *__restrict pr = c->q8_w + (size_t)(r8 >> 3) * c->n_blocks * 272;
    if (c->rows - r8 >= 8) {
        q8x8_dot_rows(c->out, r8, pr, c->xq, c->xd, c->n_blocks, c->residual);
    } else {
        float tmp[8];   /* partial last group: out buffer is rows-sized */
        q8x8_dot_rows(tmp, 0, pr, c->xq, c->xd, c->n_blocks, NULL);
        for (int m = 0; m < c->rows - r8; m++)
            c->out[r8 + m] = c->residual ? c->residual[r8 + m] + tmp[m] : tmp[m];
    }
}

static void q8x8_matvec_vdot(float *__restrict out, const uint8_t *__restrict q8_w,
                             const int8_t *__restrict xq, const float *__restrict xd,
                             int rows, int cols, const float *__restrict residual) {
    int n_blocks = cols / 32;
    vllm_q8x8_vdot_ctx vc = { out, q8_w, xq, xd, residual, n_blocks, rows };
    vllm_tp_parfor(0, (rows + 7) / 8, vllm_q8x8_vdot_worker, &vc);
    /* 推理期自检（VLLM_Q8CHK=3）：decode 8x8 GEMV vs 朴素参考解包 */
    if (getenv("VLLM_Q8CHK") && getenv("VLLM_Q8CHK")[0] == '3') {
        static int chk_cnt = 0;
        if (chk_cnt < 500) {
            chk_cnt++;
            int bad = 0;
            double maxe = 0;
            for (int r = 0; r < rows; r++) {
                const uint8_t *grp = q8_w + (size_t)(r >> 3) * n_blocks * 272;
                int mm = r & 7;
                double acc = 0;
                for (int b = 0; b < n_blocks; b++) {
                    const uint8_t *blk = grp + (size_t)b * 272;
                    uint32_t fb = f16_to_f32_bits(*(const uint16_t *)(blk + (size_t)mm * 2));
                    float d;
                    memcpy(&d, &fb, 4);
                    double s = 0;
                    for (int k = 0; k < 8; k++)
                        for (int i = 0; i < 4; i++)
                            s += (double)(int8_t)blk[16 + k * 32 + mm * 4 + i] *
                                 xq[(size_t)b * 32 + k * 4 + i];
                    acc += s * d * xd[b];
                }
                double e = fabs((residual ? residual[r] + acc : acc) - out[r]);
                if (e > 1e-3 && fabs(e / (residual ? residual[r] + acc : acc)) > 1e-4) bad++;
                if (e > maxe) maxe = e;
            }
            fprintf(stderr, "[Q8CHK3#%d] decode GEMV rows=%d cols=%d nb=%d bad=%d maxe=%.3f out[0]=%.4f\n",
                    chk_cnt, rows, cols, n_blocks, bad, maxe, out[0]);
            fflush(stderr);
        }
    }
}

/* decode 单 token GEMV 入口：按当前 Q8 布局 dispatch（8x8 tile 或 4x4）。
 * 权重 repack 布局由 VLLM_Q8_8X8 决定（8x8 需 rows%8==0，repack 门控保证）；
 * 单 token 路径必须与权重布局一致，否则按错布局解包产生垃圾输出
 * （2026-08-30 定案：8x8 ON 时 decode QKV/O/GATEUP/DOWN 曾硬编码 4x4）。 */
static void q8_vdot_dispatch(float *__restrict out, const uint8_t *__restrict w,
                             const int8_t *__restrict xq, const float *__restrict xd,
                             int rows, int cols, const float *__restrict residual) {
    if (q8_8x8_enabled() && (rows & 7) == 0)
        q8x8_matvec_vdot(out, w, xq, xd, rows, cols, residual);
    else
        q8x4_matvec_vdot(out, w, xq, xd, rows, cols, residual);
}

typedef struct {
    float *out; const uint8_t *q8_w; const int8_t *xq; const float *xd;
    const float *residual; int nb, rows, cols, n_batch;
} vllm_q8x8_gemv_ctx;

static void vllm_q8x8_gemv_worker(void *ctx_, int t) {
    vllm_q8x8_gemv_ctx *c = ctx_;
    float *o = c->out + (size_t)t * c->rows;
    const int8_t *xqt = c->xq + (size_t)t * c->cols;
    const float *xdt = c->xd + (size_t)t * c->nb;
    /* 残差按 token 偏移（token-major [n_batch][rows]）；q8x8_dot_rows 内部用
     * residual[r8+m] 绝对索引，故传入本 token 的残差基址。
     * 2026-08-30 修复：此前直接传 c->residual，token>0 误用 token0 残差，
     * 导致 8x8 模式 prefill O/down 投影在 nb>1 时输出错误（与 4x4 不一致）。 */
    const float *res_t = c->residual ? c->residual + (size_t)t * c->rows : NULL;
    int r8 = 0;
    for (; r8 + 8 <= c->rows; r8 += 8) {
        const uint8_t *__restrict pr = c->q8_w + (size_t)(r8 >> 3) * c->nb * 272;
        q8x8_dot_rows(o, r8, pr, xqt, xdt, c->nb, res_t);
    }
    if (r8 < c->rows) {   /* partial last 8-row group */
        float tmp[8];
        const uint8_t *__restrict pr = c->q8_w + (size_t)(r8 >> 3) * c->nb * 272;
        q8x8_dot_rows(tmp, 0, pr, xqt, xdt, c->nb, NULL);
        for (int m = 0; m < c->rows - r8; m++)
            o[r8 + m] = res_t ? res_t[r8 + m] + tmp[m] : tmp[m];
    }
}

/* 8x8 GEMM entry: tiled 8x8/8x4 when n_batch % 4 == 0 (repack OOM falls back
 * to per-token 8x8 GEMV). Zero-padded tail keeps padded outputs at exactly 0. */
static void q8x8_gemm_batched(float *__restrict out, const uint8_t *__restrict q8_w,
                              const int8_t *__restrict xq, const float *__restrict xd,
                              int rows, int cols, int n_batch,
                              const float *__restrict residual) {
    /* 8x8 GEMM 按 (n_batch+7)/8 组遍历，repack 对 partial group 补零（d=0 -> 贡献 0），
     * worker 支持任意尾部 token 数 -> 任意 n_batch 都可走真 GEMM（无需 %4）。 */
    if (repack_q8_0_8x8_act(xq, xd, n_batch, cols) == 0) {
        st_gemm_q8_0_8x8_batched(out, q8_w, rows, cols, n_batch, residual);
        return;
    }
    {
        int nb = cols / 32;
        vllm_q8x8_gemv_ctx vc = { out, q8_w, xq, xd, residual, nb, rows, cols, n_batch };
        vllm_tp_parfor(0, n_batch, vllm_q8x8_gemv_worker, &vc);
        /* 推理期自检（VLLM_Q8CHK=2）：token 0 的 8x8 per-token GEMV 输出
         * vs 8x8 布局朴素参考解包。bad=0 表示内核在真实数据下正确。 */
        if (getenv("VLLM_Q8CHK") && getenv("VLLM_Q8CHK")[0] == '2') {
            static int chk_cnt = 0;
            if (chk_cnt < 400) {
                chk_cnt++;
                int bad = 0, first_bad = -1;
                double maxe = 0, v_ref = 0, v_out = 0;
                for (int r = 0; r < rows; r++) {
                    const uint8_t *grp = q8_w + (size_t)(r >> 3) * nb * 272;
                    int mm = r & 7;
                    double acc = 0;
                    for (int b = 0; b < nb; b++) {
                        const uint8_t *blk = grp + (size_t)b * 272;
                        uint32_t fb = f16_to_f32_bits(*(const uint16_t *)(blk + (size_t)mm * 2));
                        float d;
                        memcpy(&d, &fb, 4);
                        double s = 0;
                        for (int k = 0; k < 8; k++)
                            for (int i = 0; i < 4; i++)
                                s += (double)(int8_t)blk[16 + k * 32 + mm * 4 + i] *
                                     xq[(size_t)b * 32 + k * 4 + i];
                        acc += s * d * xd[b];
                    }
                    double e = fabs((residual ? residual[r] + acc : acc) - out[r]);
                    if (e > 1e-3 && fabs(e / (residual ? residual[r] + acc : acc)) > 1e-4) {
                        bad++;
                        if (first_bad < 0) { first_bad = r; v_ref = acc; v_out = out[r]; }
                    }
                    if (e > maxe) maxe = e;
                }
                if (bad)
                    fprintf(stderr, "[Q8CHK2-BAD#%d] rows=%d cols=%d nb=%d bad=%d maxe=%.3f first_row=%d ref=%.4f out=%.4f\n",
                            chk_cnt, rows, cols, nb, bad, maxe, first_bad, v_ref, v_out);
                if (chk_cnt == 1)
                    fprintf(stderr, "[Q8CHK2] prefill per-token GEMV 检查开始（只打印 bad 调用）\n");
                fflush(stderr);
            }
        }
        return;
    }
}

/* Q8_0 single-token kernels (int8 weight x fp32 activation) */

typedef struct {
    float *out; const uint8_t *q8_w; const float *x;
    int n_blocks, row_stride, rows;
} vllm_dyn_matvec_q8_ctx;

static void vllm_dyn_matvec_q8_worker(void *ctx_, int r) {
    vllm_dyn_matvec_q8_ctx *c = ctx_;
    const uint8_t *__restrict pr = c->q8_w + (size_t)r * c->row_stride;
    float32x4_t acc[8];
    for (int i = 0; i < 8; i++) acc[i] = vdupq_n_f32(0.0f);
    for (int b = 0; b < c->n_blocks; b++) {
        float d = q8_block_scale(pr);
        float32x4_t f[8];
        i8x32_to_f32x8((const int8_t *)(pr + 2), f);
        float32x4_t dq = vdupq_n_f32(d);
        const float *xs = c->x + (size_t)b * 32;
        ST_PREFETCH(pr + 34 * 4);
        for (int i = 0; i < 8; i++)
            acc[i] = vfmaq_f32(acc[i], vmulq_f32(f[i], dq), vld1q_f32(xs + (size_t)i * 4));
        pr += 34;
    }
    c->out[r] = f32x8_hsum(acc);
}

static void dyn_matvec_q8_neon(float *__restrict out, const uint8_t *__restrict q8_w,
                               const float *__restrict x, int rows, int cols) {
    int n_blocks = cols / 32;
    int row_stride = n_blocks * 34;
#if ST_NEON_DOTPROD
    if (g_st_q8_repack) {
        if (q8_8x8_enabled() && (rows & 7) == 0) {
            /* P0: 8x8-repacked 权重（block_q8_0x8）+ 量化激活 vdot GEMV
             * （8 行/组，与 4x4 GEMV 逐行位级一致，板端实测 1.15-1.19x）。 */
            int8_t *xq = (int8_t *)xq_alloc_canary((size_t)cols);
            float *xd = (float *)xq_alloc_canary((size_t)n_blocks * sizeof(float));
            if (xq && xd) {
                quantize_row_q8_0_act(x, xq, xd, cols);
                q8x8_matvec_vdot(out, q8_w, xq, xd, rows, cols, NULL);
            }
            xq_free_canary(xq, (size_t)cols, "q8:xq");
            xq_free_canary(xd, (size_t)n_blocks * sizeof(float), "q8:xd");
            return;
        }
        if ((rows & 3) == 0) {
            /* M4e: 4x4-repacked 权重（block_q8_0x4）+ 量化激活 vdot GEMV。 */
            int8_t *xq = (int8_t *)xq_alloc_canary((size_t)cols);
            float *xd = (float *)xq_alloc_canary((size_t)n_blocks * sizeof(float));
            if (xq && xd) {
                quantize_row_q8_0_act(x, xq, xd, cols);
                q8x4_matvec_vdot(out, q8_w, xq, xd, rows, cols, NULL);
            }
            xq_free_canary(xq, (size_t)cols, "q8:xq");
            xq_free_canary(xd, (size_t)n_blocks * sizeof(float), "q8:xd");
            return;
        }
    }
#endif
    vllm_dyn_matvec_q8_ctx vc = { out, q8_w, x, n_blocks, row_stride, rows };
    vllm_tp_parfor(0, rows, vllm_dyn_matvec_q8_worker, &vc);
}

typedef struct {
    float *gate_out, *up_out; const uint8_t *q8_gate, *q8_up; const float *x;
    int n_blocks, row_stride, rows;
} vllm_dyn_matvec_q8_gu_ctx;

static void vllm_dyn_matvec_q8_gu_worker(void *ctx_, int r) {
    vllm_dyn_matvec_q8_gu_ctx *c = ctx_;
    const uint8_t *__restrict pg = c->q8_gate + (size_t)r * c->row_stride;
    const uint8_t *__restrict pu = c->q8_up   + (size_t)r * c->row_stride;
    float32x4_t ag[8], au[8];
    for (int i = 0; i < 8; i++) { ag[i] = vdupq_n_f32(0.0f); au[i] = vdupq_n_f32(0.0f); }
    for (int b = 0; b < c->n_blocks; b++) {
        float dg = q8_block_scale(pg);
        float du = q8_block_scale(pu);
        float32x4_t f[8];
        i8x32_to_f32x8((const int8_t *)(pg + 2), f);
        const float *xs = c->x + (size_t)b * 32;
        ST_PREFETCH(pg + 34 * 4); ST_PREFETCH(pu + 34 * 4);
        for (int i = 0; i < 8; i++) {
            float32x4_t xv = vld1q_f32(xs + (size_t)i * 4);
            ag[i] = vfmaq_f32(ag[i], vmulq_f32(f[i], vdupq_n_f32(dg)), xv);
            au[i] = vfmaq_f32(au[i], vmulq_f32(f[i], vdupq_n_f32(du)), xv);
        }
        pg += 34; pu += 34;
    }
    c->gate_out[r] = f32x8_hsum(ag);
    c->up_out[r]   = f32x8_hsum(au);
}

static void dyn_matvec_q8_fused_gate_up_neon(
    float *__restrict gate_out, float *__restrict up_out,
    const uint8_t *__restrict q8_gate, const uint8_t *__restrict q8_up,
    const float *__restrict x, int rows, int cols) {
    int n_blocks = cols / 32;
    int row_stride = n_blocks * 34;
#if ST_NEON_DOTPROD
    if (g_st_q8_repack && (rows & 3) == 0) {
        /* M4e: gate/up 各走 4x4 vdot GEMV（共享量化激活）。 */
        int8_t *xq = (int8_t *)xq_alloc_canary((size_t)cols);
        float *xd = (float *)xq_alloc_canary((size_t)n_blocks * sizeof(float));
        if (xq && xd) {
            quantize_row_q8_0_act(x, xq, xd, cols);
            q8_vdot_dispatch(gate_out, q8_gate, xq, xd, rows, cols, NULL);
            q8_vdot_dispatch(up_out, q8_up, xq, xd, rows, cols, NULL);
        }
        xq_free_canary(xq, (size_t)cols, "q8_gu:xq");
        xq_free_canary(xd, (size_t)n_blocks * sizeof(float), "q8_gu:xd");
        return;
    }
#endif
    vllm_dyn_matvec_q8_gu_ctx vc = { gate_out, up_out, q8_gate, q8_up, x, n_blocks, row_stride, rows };
    vllm_tp_parfor(0, rows, vllm_dyn_matvec_q8_gu_worker, &vc);
}

typedef struct {
    float *q_out, *k_out, *v_out; const uint8_t *q8_q, *q8_k, *q8_v; const float *x;
    int n_blocks, row_stride, kv_rows, q_rows;
} vllm_dyn_q8_qkv_ctx;

static void vllm_dyn_q8_qkv_worker(void *ctx_, int r) {
    vllm_dyn_q8_qkv_ctx *c = ctx_;
    const uint8_t *__restrict pq = c->q8_q + (size_t)r * c->row_stride;
    const uint8_t *__restrict pk = NULL;
    const uint8_t *__restrict pv = NULL;
    int is_kv = (r < c->kv_rows);
    if (is_kv) { pk = c->q8_k + (size_t)r * c->row_stride; pv = c->q8_v + (size_t)r * c->row_stride; }
    float32x4_t aq[8], ak[8], av[8];
    for (int i = 0; i < 8; i++) {
        aq[i] = vdupq_n_f32(0.0f);
        ak[i] = vdupq_n_f32(0.0f);
        av[i] = vdupq_n_f32(0.0f);
    }
    for (int b = 0; b < c->n_blocks; b++) {
        float dq = q8_block_scale(pq);
        float dk = 0.0f, dv = 0.0f;
        if (is_kv) { dk = q8_block_scale(pk); dv = q8_block_scale(pv); }
        float32x4_t fq[8], fk[8], fv[8];
        i8x32_to_f32x8((const int8_t *)(pq + 2), fq);
        if (is_kv) {
            i8x32_to_f32x8((const int8_t *)(pk + 2), fk);
            i8x32_to_f32x8((const int8_t *)(pv + 2), fv);
        }
        const float *xs = c->x + (size_t)b * 32;
        ST_PREFETCH(pq + 34 * 4);
        for (int i = 0; i < 8; i++) {
            float32x4_t xv = vld1q_f32(xs + (size_t)i * 4);
            aq[i] = vfmaq_f32(aq[i], vmulq_f32(fq[i], vdupq_n_f32(dq)), xv);
            if (is_kv) {
                ak[i] = vfmaq_f32(ak[i], vmulq_f32(fk[i], vdupq_n_f32(dk)), xv);
                av[i] = vfmaq_f32(av[i], vmulq_f32(fv[i], vdupq_n_f32(dv)), xv);
            }
        }
        pq += 34;
        if (is_kv) { pk += 34; pv += 34; }
    }
    c->q_out[r] = f32x8_hsum(aq);
    if (is_kv) {
        c->k_out[r] = f32x8_hsum(ak);
        c->v_out[r] = f32x8_hsum(av);
    }
}

static void dyn_matvec_q8_fused_qkv_neon(
    float *__restrict q_out, float *__restrict k_out, float *__restrict v_out,
    const uint8_t *__restrict q8_q, const uint8_t *__restrict q8_k,
    const uint8_t *__restrict q8_v, const float *__restrict x,
    int q_rows, int kv_rows, int cols) {
    int n_blocks = cols / 32;
    int row_stride = n_blocks * 34;
#if ST_NEON_DOTPROD
    if (g_st_q8_repack && (q_rows & 3) == 0 && (kv_rows & 3) == 0) {
        /* M4e: q/k/v 各走 4x4 vdot GEMV（共享量化激活）。 */
        int8_t *xq = (int8_t *)xq_alloc_canary((size_t)cols);
        float *xd = (float *)xq_alloc_canary((size_t)n_blocks * sizeof(float));
        if (xq && xd) {
            quantize_row_q8_0_act(x, xq, xd, cols);
            q8_vdot_dispatch(q_out, q8_q, xq, xd, q_rows, cols, NULL);
            q8_vdot_dispatch(k_out, q8_k, xq, xd, kv_rows, cols, NULL);
            q8_vdot_dispatch(v_out, q8_v, xq, xd, kv_rows, cols, NULL);
        }
        xq_free_canary(xq, (size_t)cols, "q8_qkv:xq");
        xq_free_canary(xd, (size_t)n_blocks * sizeof(float), "q8_qkv:xd");
        return;
    }
#endif
    vllm_dyn_q8_qkv_ctx vc = { q_out, k_out, v_out, q8_q, q8_k, q8_v, x, n_blocks, row_stride, kv_rows, q_rows };
    vllm_tp_parfor(0, q_rows, vllm_dyn_q8_qkv_worker, &vc);
}

typedef struct {
    float *x; const float *residual; const uint8_t *q8_o; const float *attn_in;
    int n_blocks, row_stride, rows;
} vllm_dyn_q8_o_ctx;

static void vllm_dyn_q8_o_worker(void *ctx_, int r) {
    vllm_dyn_q8_o_ctx *c = ctx_;
    const uint8_t *__restrict pr = c->q8_o + (size_t)r * c->row_stride;
    float32x4_t acc[8];
    for (int i = 0; i < 8; i++) acc[i] = vdupq_n_f32(0.0f);
    for (int b = 0; b < c->n_blocks; b++) {
        float d = q8_block_scale(pr);
        float32x4_t f[8];
        i8x32_to_f32x8((const int8_t *)(pr + 2), f);
        float32x4_t dq = vdupq_n_f32(d);
        const float *xs = c->attn_in + (size_t)b * 32;
        ST_PREFETCH(pr + 34 * 4);
        for (int i = 0; i < 8; i++)
            acc[i] = vfmaq_f32(acc[i], vmulq_f32(f[i], dq), vld1q_f32(xs + (size_t)i * 4));
        pr += 34;
    }
    c->x[r] = c->residual[r] + f32x8_hsum(acc);
}

static void dyn_matvec_q8_fused_o_residual_neon(
    float *__restrict x, const float *__restrict residual,
    const uint8_t *__restrict q8_o, const float *__restrict attn_in,
    int rows, int cols) {
    int n_blocks = cols / 32;
    int row_stride = n_blocks * 34;
#if ST_NEON_DOTPROD
    if (g_st_q8_repack && (rows & 3) == 0) {
        /* M4e: O projection 4x4 vdot GEMV + residual。 */
        int8_t *xq = (int8_t *)xq_alloc_canary((size_t)cols);
        float *xd = (float *)xq_alloc_canary((size_t)n_blocks * sizeof(float));
        if (xq && xd) {
            quantize_row_q8_0_act(attn_in, xq, xd, cols);
            q8_vdot_dispatch(x, q8_o, xq, xd, rows, cols, residual);
        }
        xq_free_canary(xq, (size_t)cols, "q8_o:xq");
        xq_free_canary(xd, (size_t)n_blocks * sizeof(float), "q8_o:xd");
        return;
    }
#endif
    vllm_dyn_q8_o_ctx vc = { x, residual, q8_o, attn_in, n_blocks, row_stride, rows };
    vllm_tp_parfor(0, rows, vllm_dyn_q8_o_worker, &vc);
}

typedef struct {
    float *x; const float *residual; const uint8_t *q8_down; const float *activated;
    int n_blocks, row_stride, hidden_dim;
} vllm_dyn_q8_dn_ctx;

static void vllm_dyn_q8_dn_worker(void *ctx_, int j) {
    vllm_dyn_q8_dn_ctx *c = ctx_;
    const uint8_t *__restrict pr = c->q8_down + (size_t)j * c->row_stride;
    float32x4_t acc[8];
    for (int i = 0; i < 8; i++) acc[i] = vdupq_n_f32(0.0f);
    for (int b = 0; b < c->n_blocks; b++) {
        float d = q8_block_scale(pr);
        float32x4_t f[8];
        i8x32_to_f32x8((const int8_t *)(pr + 2), f);
        float32x4_t dq = vdupq_n_f32(d);
        const float *xs = c->activated + (size_t)b * 32;
        ST_PREFETCH(pr + 34 * 4);
        for (int i = 0; i < 8; i++)
            acc[i] = vfmaq_f32(acc[i], vmulq_f32(f[i], dq), vld1q_f32(xs + (size_t)i * 4));
        pr += 34;
    }
    c->x[j] = c->residual[j] + f32x8_hsum(acc);
}

static void dyn_matvec_q8_fused_down_residual_neon(
    float *__restrict x, const float *__restrict residual,
    const uint8_t *__restrict q8_down, const float *__restrict activated,
    int hidden_dim, int ffn_dim) {
    int n_blocks = ffn_dim / 32;
    int row_stride = n_blocks * 34;
#if ST_NEON_DOTPROD
    if (g_st_q8_repack && (hidden_dim & 3) == 0) {
        /* M4e: down projection 4x4 vdot GEMV + residual。 */
        int8_t *xq = (int8_t *)xq_alloc_canary((size_t)ffn_dim);
        float *xd = (float *)xq_alloc_canary((size_t)n_blocks * sizeof(float));
        if (xq && xd) {
            quantize_row_q8_0_act(activated, xq, xd, ffn_dim);
            q8_vdot_dispatch(x, q8_down, xq, xd, hidden_dim, ffn_dim, residual);
        }
        xq_free_canary(xq, (size_t)ffn_dim, "q8_dn:xq");
        xq_free_canary(xd, (size_t)n_blocks * sizeof(float), "q8_dn:xd");
        return;
    }
#endif
    vllm_dyn_q8_dn_ctx vc = { x, residual, q8_down, activated, n_blocks, row_stride, hidden_dim };
    vllm_tp_parfor(0, hidden_dim, vllm_dyn_q8_dn_worker, &vc);
}

/* ================================================================
 * Q8_0 batched kernels (int8 weight x block-major quantized activation)
 * ================================================================ */

typedef struct {
    const float *x_batch; int8_t *xq; float *xd;
    int cols, n_blocks, n_batch;
} vllm_q8_quant_ctx;

static void vllm_q8_quant_worker(void *ctx_, int t) {
    vllm_q8_quant_ctx *c = ctx_;
    quantize_row_q8_0_act(c->x_batch + (size_t)t * c->cols,
                          c->xq + (size_t)t * c->cols,
                          c->xd + (size_t)t * c->n_blocks, c->cols);
}

typedef struct {
    float *q_out, *k_out, *v_out; const uint8_t *q8_q, *q8_k, *q8_v;
    const int8_t *xq; const float *xd;
    int n_blocks, row_stride, q_rows, kv_rows, n_batch;
} vllm_q8_qkvb_kv_ctx;

static void vllm_q8_qkvb_kv_worker(void *ctx_, int r) {
    vllm_q8_qkvb_kv_ctx *c = ctx_;
    const uint8_t *__restrict pq = c->q8_q + (size_t)r * c->row_stride;
    const uint8_t *__restrict pk = c->q8_k + (size_t)r * c->row_stride;
    const uint8_t *__restrict pv = c->q8_v + (size_t)r * c->row_stride;

    for (int t0 = 0; t0 < c->n_batch; t0 += NEON_TILE_QKV) {
        int nt = c->n_batch - t0;
        if (nt > NEON_TILE_QKV) nt = NEON_TILE_QKV;
        float32x4_t aq[NEON_TILE_QKV][2], ak[NEON_TILE_QKV][2], av[NEON_TILE_QKV][2];
        for (int tt = 0; tt < nt; tt++) {
            aq[tt][0] = vdupq_n_f32(0.0f); aq[tt][1] = vdupq_n_f32(0.0f);
            ak[tt][0] = vdupq_n_f32(0.0f); ak[tt][1] = vdupq_n_f32(0.0f);
            av[tt][0] = vdupq_n_f32(0.0f); av[tt][1] = vdupq_n_f32(0.0f);
        }
        for (int b = 0; b < c->n_blocks; b++) {
            int8x16_t wq0 = vld1q_s8((const int8_t *)(pq + (size_t)b * 34 + 2));
            int8x16_t wq1 = vld1q_s8((const int8_t *)(pq + (size_t)b * 34 + 2 + 16));
            float dqw = q8_block_scale(pq + (size_t)b * 34);
            int8x16_t wk0 = vld1q_s8((const int8_t *)(pk + (size_t)b * 34 + 2));
            int8x16_t wk1 = vld1q_s8((const int8_t *)(pk + (size_t)b * 34 + 2 + 16));
            int8x16_t wv0 = vld1q_s8((const int8_t *)(pv + (size_t)b * 34 + 2));
            int8x16_t wv1 = vld1q_s8((const int8_t *)(pv + (size_t)b * 34 + 2 + 16));
            float dkw = q8_block_scale(pk + (size_t)b * 34);
            float dvw = q8_block_scale(pv + (size_t)b * 34);
            const int8_t *xq_b = c->xq + (size_t)b * c->n_batch * 32;
            const float *xd_b = c->xd + (size_t)b * c->n_batch;
            ST_PREFETCH(pq + (size_t)(b + 4) * 34);
            for (int tt = 0; tt < nt; tt++) {
                int t = t0 + tt;
                int8x16_t x0 = vld1q_s8(xq_b + (size_t)t * 32);
                int8x16_t x1 = vld1q_s8(xq_b + (size_t)t * 32 + 16);
                float32x4_t dsv = vdupq_n_f32(dqw * xd_b[t]);
                aq[tt][0] = vfmaq_f32(aq[tt][0], vcvtq_f32_s32(i8x16_dot_s32(wq0, x0)), dsv);
                aq[tt][1] = vfmaq_f32(aq[tt][1], vcvtq_f32_s32(i8x16_dot_s32(wq1, x1)), dsv);
                float32x4_t dsv2 = vdupq_n_f32(dkw * xd_b[t]);
                ak[tt][0] = vfmaq_f32(ak[tt][0], vcvtq_f32_s32(i8x16_dot_s32(wk0, x0)), dsv2);
                ak[tt][1] = vfmaq_f32(ak[tt][1], vcvtq_f32_s32(i8x16_dot_s32(wk1, x1)), dsv2);
                float32x4_t dsv3 = vdupq_n_f32(dvw * xd_b[t]);
                av[tt][0] = vfmaq_f32(av[tt][0], vcvtq_f32_s32(i8x16_dot_s32(wv0, x0)), dsv3);
                av[tt][1] = vfmaq_f32(av[tt][1], vcvtq_f32_s32(i8x16_dot_s32(wv1, x1)), dsv3);
            }
        }
        for (int tt = 0; tt < nt; tt++) {
            int t = t0 + tt;
            c->q_out[(size_t)t * c->q_rows + r] =
                hsum_neon4(vaddq_f32(aq[tt][0], aq[tt][1]));
            c->k_out[(size_t)t * c->kv_rows + r] =
                hsum_neon4(vaddq_f32(ak[tt][0], ak[tt][1]));
            c->v_out[(size_t)t * c->kv_rows + r] =
                hsum_neon4(vaddq_f32(av[tt][0], av[tt][1]));
        }
    }
}

typedef struct {
    float *q_out; const uint8_t *q8_q;
    const int8_t *xq; const float *xd;
    int n_blocks, row_stride, q_rows, n_batch;
} vllm_q8_qkvb_q_ctx;

static void vllm_q8_qkvb_q_worker(void *ctx_, int r) {
    vllm_q8_qkvb_q_ctx *c = ctx_;
    const uint8_t *__restrict pq = c->q8_q + (size_t)r * c->row_stride;
    for (int t0 = 0; t0 < c->n_batch; t0 += NEON_TILE_QKV_Q) {
        int nt = c->n_batch - t0;
        if (nt > NEON_TILE_QKV_Q) nt = NEON_TILE_QKV_Q;
        float32x4_t aq[NEON_TILE_QKV_Q][2];
        for (int tt = 0; tt < nt; tt++) {
            aq[tt][0] = vdupq_n_f32(0.0f); aq[tt][1] = vdupq_n_f32(0.0f);
        }
        for (int b = 0; b < c->n_blocks; b++) {
            int8x16_t wq0 = vld1q_s8((const int8_t *)(pq + (size_t)b * 34 + 2));
            int8x16_t wq1 = vld1q_s8((const int8_t *)(pq + (size_t)b * 34 + 2 + 16));
            float dqw = q8_block_scale(pq + (size_t)b * 34);
            const int8_t *xq_b = c->xq + (size_t)b * c->n_batch * 32;
            const float *xd_b = c->xd + (size_t)b * c->n_batch;
            ST_PREFETCH(pq + (size_t)(b + 4) * 34);
            for (int tt = 0; tt < nt; tt++) {
                int t = t0 + tt;
                int8x16_t x0 = vld1q_s8(xq_b + (size_t)t * 32);
                int8x16_t x1 = vld1q_s8(xq_b + (size_t)t * 32 + 16);
                float32x4_t dsv = vdupq_n_f32(dqw * xd_b[t]);
                aq[tt][0] = vfmaq_f32(aq[tt][0], vcvtq_f32_s32(i8x16_dot_s32(wq0, x0)), dsv);
                aq[tt][1] = vfmaq_f32(aq[tt][1], vcvtq_f32_s32(i8x16_dot_s32(wq1, x1)), dsv);
            }
        }
        for (int tt = 0; tt < nt; tt++) {
            int t = t0 + tt;
            c->q_out[(size_t)t * c->q_rows + r] =
                hsum_neon4(vaddq_f32(aq[tt][0], aq[tt][1]));
        }
    }
}

static void dyn_matvec_q8_fused_qkv_batched_neon(
    float *__restrict q_out, float *__restrict k_out, float *__restrict v_out,
    const uint8_t *__restrict q8_q, const uint8_t *__restrict q8_k,
    const uint8_t *__restrict q8_v, const float *__restrict x_batch,
    int q_rows, int kv_rows, int cols, int n_batch) {
    int n_blocks = cols / 32;
    int row_stride = n_blocks * 34;
    int8_t *xq; float *xd;
    if (xq_scratch_ensure(&xq, &xd,
                          (size_t)n_batch * cols,
                          (size_t)n_batch * n_blocks * sizeof(float)) != 0)
        return;
#if ST_NEON_DOTPROD
    if (g_st_q8_repack && (q_rows & 3) == 0 && (kv_rows & 3) == 0) {
        if (n_batch == 1) {
            /* M4e: 单 token decode - 走单 token 4x4 解交织 GEMV（f32 激活，
             * 与 legacy 逐位一致）。 */
            dyn_matvec_q8_fused_qkv_neon(q_out, k_out, v_out, q8_q, q8_k, q8_v,
                                         x_batch, q_rows, kv_rows, cols);
            return;
        }
        /* M4e: token-major 量化 + q8x4 真 GEMM（llama block_q8_0x4）。权重
         * 每个 16-token 块读一次（vs legacy batched GEMV 每 token tile 重读），
         * 消除 Q8 prefill 3.18x 差距的根因。f32 舍入顺序放宽 -> PPL 验收。
         * n_batch%4!=0 时 q8x4_gemm_batched 内部逐 token 回退。 */
        vllm_q8_quant_ctx vc = { x_batch, xq, xd, cols, n_blocks, n_batch };
        vllm_tp_parfor(0, n_batch, vllm_q8_quant_worker, &vc);
        q8x4_gemm_batched(q_out, q8_q, xq, xd, q_rows, cols, n_batch, NULL);
        q8x4_gemm_batched(k_out, q8_k, xq, xd, kv_rows, cols, n_batch, NULL);
        q8x4_gemm_batched(v_out, q8_v, xq, xd, kv_rows, cols, n_batch, NULL);
        return;
    }
#endif
    quantize_row_q8_0_act_bm(x_batch, xq, xd, cols, n_batch);

    /* Row-split branchless loops (bit-identical accumulation order): rows
     * [0,kv_rows) compute Q+K+V, rows [kv_rows,q_rows) compute Q only. */
    vllm_q8_qkvb_kv_ctx vk = { q_out, k_out, v_out, q8_q, q8_k, q8_v, xq, xd, n_blocks, row_stride, q_rows, kv_rows, n_batch };
    vllm_tp_parfor(0, kv_rows, vllm_q8_qkvb_kv_worker, &vk);
    vllm_q8_qkvb_q_ctx vq = { q_out, q8_q, xq, xd, n_blocks, row_stride, q_rows, n_batch };
    vllm_tp_parfor(kv_rows, q_rows, vllm_q8_qkvb_q_worker, &vq);
}

typedef struct {
    const float *attn_batch; int8_t *xq; float *xd;
    int cols, n_blocks, n_batch;
} vllm_q8_o_quant_ctx;

static void vllm_q8_o_quant_worker(void *ctx_, int t) {
    vllm_q8_o_quant_ctx *c = ctx_;
    quantize_row_q8_0_act(c->attn_batch + (size_t)t * c->cols,
                          c->xq + (size_t)t * c->cols,
                          c->xd + (size_t)t * c->n_blocks, c->cols);
}

typedef struct {
    float *x_batch; const float *residual_batch; const uint8_t *q8_o;
    const int8_t *xq; const float *xd;
    int n_blocks, row_stride, rows, n_batch;
} vllm_q8_o_b_ctx;

static void vllm_q8_o_b_worker(void *ctx_, int r) {
    vllm_q8_o_b_ctx *c = ctx_;
    const uint8_t *__restrict pr = c->q8_o + (size_t)r * c->row_stride;
    for (int t0 = 0; t0 < c->n_batch; t0 += NEON_TILE_1) {
        int nt = c->n_batch - t0;
        if (nt > NEON_TILE_1) nt = NEON_TILE_1;
        float32x4_t acc[NEON_TILE_1][2];
        for (int tt = 0; tt < nt; tt++) {
            acc[tt][0] = vdupq_n_f32(0.0f);
            acc[tt][1] = vdupq_n_f32(0.0f);
        }
        for (int b = 0; b < c->n_blocks; b++) {
            int8x16_t w0 = vld1q_s8((const int8_t *)(pr + (size_t)b * 34 + 2));
            int8x16_t w1 = vld1q_s8((const int8_t *)(pr + (size_t)b * 34 + 2 + 16));
            float dw = q8_block_scale(pr + (size_t)b * 34);
            const int8_t *xq_b = c->xq + (size_t)b * c->n_batch * 32;
            const float *xd_b = c->xd + (size_t)b * c->n_batch;
            ST_PREFETCH(pr + (size_t)(b + 4) * 34);
            for (int tt = 0; tt < nt; tt++) {
                int t = t0 + tt;
                int8x16_t x0 = vld1q_s8(xq_b + (size_t)t * 32);
                int8x16_t x1 = vld1q_s8(xq_b + (size_t)t * 32 + 16);
                float32x4_t dsv = vdupq_n_f32(dw * xd_b[t]);
                acc[tt][0] = vfmaq_f32(acc[tt][0], vcvtq_f32_s32(i8x16_dot_s32(w0, x0)), dsv);
                acc[tt][1] = vfmaq_f32(acc[tt][1], vcvtq_f32_s32(i8x16_dot_s32(w1, x1)), dsv);
            }
        }
        for (int tt = 0; tt < nt; tt++) {
            int t = t0 + tt;
            c->x_batch[(size_t)t * c->rows + r] =
                c->residual_batch[(size_t)t * c->rows + r] +
                hsum_neon4(vaddq_f32(acc[tt][0], acc[tt][1]));
        }
    }
}

static void dyn_matvec_q8_fused_o_residual_batched_neon(
    float *__restrict x_batch, const float *__restrict residual_batch,
    const uint8_t *__restrict q8_o, const float *__restrict attn_batch,
    int rows, int cols, int n_batch) {
    int n_blocks = cols / 32;
    int row_stride = n_blocks * 34;
    int8_t *xq; float *xd;
    if (xq_scratch_ensure(&xq, &xd,
                          (size_t)n_batch * cols,
                          (size_t)n_batch * n_blocks * sizeof(float)) != 0)
        return;
#if ST_NEON_DOTPROD
    if (g_st_q8_repack && (rows & 3) == 0) {
        if (n_batch == 1) {
            /* M4e: 单 token decode - 单 token 4x4 解交织 GEMV（位级一致）。 */
            dyn_matvec_q8_fused_o_residual_neon(x_batch, residual_batch, q8_o,
                                                attn_batch, rows, cols);
            return;
        }
        /* M4e: O projection 真 GEMM + residual 后加（语义 = residual + matmul）。 */
        vllm_q8_o_quant_ctx vc = { attn_batch, xq, xd, cols, n_blocks, n_batch };
        vllm_tp_parfor(0, n_batch, vllm_q8_o_quant_worker, &vc);
        q8x4_gemm_batched(x_batch, q8_o, xq, xd, rows, cols, n_batch, residual_batch);
        return;
    }
#endif
    quantize_row_q8_0_act_bm(attn_batch, xq, xd, cols, n_batch);

    vllm_q8_o_b_ctx vb = { x_batch, residual_batch, q8_o, xq, xd, n_blocks, row_stride, rows, n_batch };
    vllm_tp_parfor(0, rows, vllm_q8_o_b_worker, &vb);
}

typedef struct {
    const float *x_batch; int8_t *xq; float *xd;
    int cols, n_blocks, n_batch;
} vllm_q8_gu_quant_ctx;

static void vllm_q8_gu_quant_worker(void *ctx_, int t) {
    vllm_q8_gu_quant_ctx *c = ctx_;
    quantize_row_q8_0_act(c->x_batch + (size_t)t * c->cols,
                          c->xq + (size_t)t * c->cols,
                          c->xd + (size_t)t * c->n_blocks, c->cols);
}

typedef struct {
    float *gate_out, *up_out; const uint8_t *q8_gate, *q8_up;
    const int8_t *xq; const float *xd;
    int n_blocks, row_stride, rows, n_batch;
} vllm_q8_gu_b_ctx;

static void vllm_q8_gu_b_worker(void *ctx_, int r) {
    vllm_q8_gu_b_ctx *c = ctx_;
    const uint8_t *__restrict pg = c->q8_gate + (size_t)r * c->row_stride;
    const uint8_t *__restrict pu = c->q8_up   + (size_t)r * c->row_stride;
    for (int t0 = 0; t0 < c->n_batch; t0 += NEON_TILE_GU) {
        int nt = c->n_batch - t0;
        if (nt > NEON_TILE_GU) nt = NEON_TILE_GU;
        float32x4_t ag[NEON_TILE_GU][2], au[NEON_TILE_GU][2];
        for (int tt = 0; tt < nt; tt++) {
            ag[tt][0] = vdupq_n_f32(0.0f); ag[tt][1] = vdupq_n_f32(0.0f);
            au[tt][0] = vdupq_n_f32(0.0f); au[tt][1] = vdupq_n_f32(0.0f);
        }
        for (int b = 0; b < c->n_blocks; b++) {
            int8x16_t wg0 = vld1q_s8((const int8_t *)(pg + (size_t)b * 34 + 2));
            int8x16_t wg1 = vld1q_s8((const int8_t *)(pg + (size_t)b * 34 + 2 + 16));
            int8x16_t wu0 = vld1q_s8((const int8_t *)(pu + (size_t)b * 34 + 2));
            int8x16_t wu1 = vld1q_s8((const int8_t *)(pu + (size_t)b * 34 + 2 + 16));
            float dg = q8_block_scale(pg + (size_t)b * 34);
            float du = q8_block_scale(pu + (size_t)b * 34);
            const int8_t *xq_b = c->xq + (size_t)b * c->n_batch * 32;
            const float *xd_b = c->xd + (size_t)b * c->n_batch;
            ST_PREFETCH(pg + (size_t)(b + 4) * 34);
            ST_PREFETCH(pu + (size_t)(b + 4) * 34);
            for (int tt = 0; tt < nt; tt++) {
                int t = t0 + tt;
                int8x16_t x0 = vld1q_s8(xq_b + (size_t)t * 32);
                int8x16_t x1 = vld1q_s8(xq_b + (size_t)t * 32 + 16);
                float32x4_t dgv = vdupq_n_f32(dg * xd_b[t]);
                float32x4_t duv = vdupq_n_f32(du * xd_b[t]);
                ag[tt][0] = vfmaq_f32(ag[tt][0], vcvtq_f32_s32(i8x16_dot_s32(wg0, x0)), dgv);
                ag[tt][1] = vfmaq_f32(ag[tt][1], vcvtq_f32_s32(i8x16_dot_s32(wg1, x1)), dgv);
                au[tt][0] = vfmaq_f32(au[tt][0], vcvtq_f32_s32(i8x16_dot_s32(wu0, x0)), duv);
                au[tt][1] = vfmaq_f32(au[tt][1], vcvtq_f32_s32(i8x16_dot_s32(wu1, x1)), duv);
            }
        }
        for (int tt = 0; tt < nt; tt++) {
            int t = t0 + tt;
            c->gate_out[(size_t)t * c->rows + r] =
                hsum_neon4(vaddq_f32(ag[tt][0], ag[tt][1]));
            c->up_out[(size_t)t * c->rows + r] =
                hsum_neon4(vaddq_f32(au[tt][0], au[tt][1]));
        }
    }
}

static void dyn_matvec_q8_fused_gate_up_batched_neon(
    float *__restrict gate_out, float *__restrict up_out,
    const uint8_t *__restrict q8_gate, const uint8_t *__restrict q8_up,
    const float *__restrict x_batch, int rows, int cols, int n_batch) {
    int n_blocks = cols / 32;
    int row_stride = n_blocks * 34;
    int8_t *xq; float *xd;
    if (xq_scratch_ensure(&xq, &xd,
                          (size_t)n_batch * cols,
                          (size_t)n_batch * n_blocks * sizeof(float)) != 0)
        return;
#if ST_NEON_DOTPROD
    if (g_st_q8_repack && (rows & 3) == 0) {
        if (n_batch == 1) {
            /* M4e: 单 token decode - 单 token 4x4 解交织 GEMV（位级一致）。 */
            dyn_matvec_q8_fused_gate_up_neon(gate_out, up_out, q8_gate, q8_up,
                                             x_batch, rows, cols);
            return;
        }
        /* M4e: gate/up 真 GEMM（共享激活 repack）。 */
        vllm_q8_gu_quant_ctx vc = { x_batch, xq, xd, cols, n_blocks, n_batch };
        vllm_tp_parfor(0, n_batch, vllm_q8_gu_quant_worker, &vc);
        q8x4_gemm_batched(gate_out, q8_gate, xq, xd, rows, cols, n_batch, NULL);
        q8x4_gemm_batched(up_out, q8_up, xq, xd, rows, cols, n_batch, NULL);
        return;
    }
#endif
    quantize_row_q8_0_act_bm(x_batch, xq, xd, cols, n_batch);

    vllm_q8_gu_b_ctx vb = { gate_out, up_out, q8_gate, q8_up, xq, xd, n_blocks, row_stride, rows, n_batch };
    vllm_tp_parfor(0, rows, vllm_q8_gu_b_worker, &vb);
}

typedef struct {
    const float *activated_batch; int8_t *xq; float *xd;
    int ffn_dim, n_blocks, n_batch;
} vllm_q8_dn_quant_ctx;

static void vllm_q8_dn_quant_worker(void *ctx_, int t) {
    vllm_q8_dn_quant_ctx *c = ctx_;
    quantize_row_q8_0_act(c->activated_batch + (size_t)t * c->ffn_dim,
                          c->xq + (size_t)t * c->ffn_dim,
                          c->xd + (size_t)t * c->n_blocks, c->ffn_dim);
}

typedef struct {
    float *x_batch; const float *residual_batch; const uint8_t *q8_down;
    const int8_t *xq; const float *xd;
    int n_blocks, row_stride, hidden_dim, n_batch;
} vllm_q8_dn_b_ctx;

static void vllm_q8_dn_b_worker(void *ctx_, int j) {
    vllm_q8_dn_b_ctx *c = ctx_;
    const uint8_t *__restrict pr = c->q8_down + (size_t)j * c->row_stride;
    for (int t0 = 0; t0 < c->n_batch; t0 += NEON_TILE_1) {
        int nt = c->n_batch - t0;
        if (nt > NEON_TILE_1) nt = NEON_TILE_1;
        float32x4_t acc[NEON_TILE_1][2];
        for (int tt = 0; tt < nt; tt++) {
            acc[tt][0] = vdupq_n_f32(0.0f);
            acc[tt][1] = vdupq_n_f32(0.0f);
        }
        for (int b = 0; b < c->n_blocks; b++) {
            int8x16_t w0 = vld1q_s8((const int8_t *)(pr + (size_t)b * 34 + 2));
            int8x16_t w1 = vld1q_s8((const int8_t *)(pr + (size_t)b * 34 + 2 + 16));
            float dw = q8_block_scale(pr + (size_t)b * 34);
            const int8_t *xq_b = c->xq + (size_t)b * c->n_batch * 32;
            const float *xd_b = c->xd + (size_t)b * c->n_batch;
            ST_PREFETCH(pr + (size_t)(b + 4) * 34);
            for (int tt = 0; tt < nt; tt++) {
                int t = t0 + tt;
                int8x16_t x0 = vld1q_s8(xq_b + (size_t)t * 32);
                int8x16_t x1 = vld1q_s8(xq_b + (size_t)t * 32 + 16);
                float32x4_t dsv = vdupq_n_f32(dw * xd_b[t]);
                acc[tt][0] = vfmaq_f32(acc[tt][0], vcvtq_f32_s32(i8x16_dot_s32(w0, x0)), dsv);
                acc[tt][1] = vfmaq_f32(acc[tt][1], vcvtq_f32_s32(i8x16_dot_s32(w1, x1)), dsv);
            }
        }
        for (int tt = 0; tt < nt; tt++) {
            int t = t0 + tt;
            c->x_batch[(size_t)t * c->hidden_dim + j] =
                c->residual_batch[(size_t)t * c->hidden_dim + j] +
                hsum_neon4(vaddq_f32(acc[tt][0], acc[tt][1]));
        }
    }
}

static void dyn_matvec_q8_fused_down_residual_batched_neon(
    float *__restrict x_batch, const float *__restrict residual_batch,
    const uint8_t *__restrict q8_down, const float *__restrict activated_batch,
    int hidden_dim, int ffn_dim, int n_batch) {
    int n_blocks = ffn_dim / 32;
    int row_stride = n_blocks * 34;
    int8_t *xq; float *xd;
    if (xq_scratch_ensure(&xq, &xd,
                          (size_t)n_batch * ffn_dim,
                          (size_t)n_batch * n_blocks * sizeof(float)) != 0)
        return;
#if ST_NEON_DOTPROD
    if (g_st_q8_repack && (hidden_dim & 3) == 0) {
        if (n_batch == 1) {
            /* M4e: 单 token decode - 单 token 4x4 解交织 GEMV（位级一致）。 */
            dyn_matvec_q8_fused_down_residual_neon(x_batch, residual_batch,
                                                   q8_down, activated_batch,
                                                   hidden_dim, ffn_dim);
            return;
        }
        /* M4e: down projection 真 GEMM + residual 后加。 */
        vllm_q8_dn_quant_ctx vc = { activated_batch, xq, xd, ffn_dim, n_blocks, n_batch };
        vllm_tp_parfor(0, n_batch, vllm_q8_dn_quant_worker, &vc);
        q8x4_gemm_batched(x_batch, q8_down, xq, xd, hidden_dim, ffn_dim, n_batch,
                          residual_batch);
        return;
    }
#endif
    quantize_row_q8_0_act_bm(activated_batch, xq, xd, ffn_dim, n_batch);

    vllm_q8_dn_b_ctx vb = { x_batch, residual_batch, q8_down, xq, xd, n_blocks, row_stride, hidden_dim, n_batch };
    vllm_tp_parfor(0, hidden_dim, vllm_q8_dn_b_worker, &vb);
}

/* ================================================================
 * Q4_0 kernels (nibble weight x Q8_0 activation)
 * ================================================================ */

typedef struct {
    float *out; const uint8_t *q4_w; const int8_t *xq; const float *xd;
    int n_blocks, row_stride, rows;
} vllm_q4_gemv_ctx;

static void vllm_q4_gemv_worker(void *ctx_, int it) {
    vllm_q4_gemv_ctx *c = ctx_;
    int r4 = it * 4;
    const uint8_t *__restrict pr = c->q4_w + (size_t)(r4 >> 2) * c->n_blocks * 72;
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (int b = 0; b < c->n_blocks; b++) {
        q4x4_dot1_group16_gemv(pr + (size_t)b * 72, c->xq + (size_t)b * 32,
                               c->xd[b], &acc);
        ST_PREFETCH(pr + (size_t)(b + 4) * 72);
    }
    c->out[r4]     = vgetq_lane_f32(acc, 0);
    c->out[r4 + 1] = vgetq_lane_f32(acc, 1);
    c->out[r4 + 2] = vgetq_lane_f32(acc, 2);
    c->out[r4 + 3] = vgetq_lane_f32(acc, 3);
}

typedef struct {
    float *out; const uint8_t *q4_w; const int8_t *xq; const float *xd;
    int n_blocks, row_stride, rows;
} vllm_q4_gemv_s_ctx;

static void vllm_q4_gemv_s_worker(void *ctx_, int r) {
    vllm_q4_gemv_s_ctx *c = ctx_;
    const uint8_t *__restrict pr = c->q4_w + (size_t)r * c->row_stride;
    float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = vdupq_n_f32(0.0f);
    for (int b = 0; b < c->n_blocks; b++) {
        float wd = q4_block_scale(pr);
        int8x16_t wlo, whi;
        q4x16_to_i8x32(pr + 2, &wlo, &whi);
        int8x16_t alo = vld1q_s8(c->xq + (size_t)b * 32);
        int8x16_t ahi = vld1q_s8(c->xq + (size_t)b * 32 + 16);
        float32x4_t dsv = vdupq_n_f32(wd * c->xd[b]);
        acc0 = vfmaq_f32(acc0, vcvtq_f32_s32(i8x16_dot_s32(wlo, alo)), dsv);
        acc1 = vfmaq_f32(acc1, vcvtq_f32_s32(i8x16_dot_s32(whi, ahi)), dsv);
        ST_PREFETCH(pr + 18 * 4);
        pr += 18;
    }
    c->out[r] = hsum_neon4(vaddq_f32(acc0, acc1));
}

static void dyn_matvec_q4_q8_neon(float *__restrict out, const uint8_t *__restrict q4_w,
                                  const float *__restrict x, int rows, int cols) {
    int n_blocks = cols / 32;
    int row_stride = n_blocks * 18;
    int8_t *xq = (int8_t *)xq_alloc_canary((size_t)cols);
    float *xd = (float *)xq_alloc_canary((size_t)n_blocks * sizeof(float));
    if (!xq || !xd) {
        xq_free_canary(xq, (size_t)cols, "q4:xq");
        xq_free_canary(xd, (size_t)n_blocks * sizeof(float), "q4:xd");
        return;
    }
    quantize_row_q8_0_act(x, xq, xd, cols);
#if ST_NEON_DOTPROD
    if (g_st_q4_repack && (rows & 3) == 0) {
        /* M4f: 4x4-repacked rows, 16-element decode GEMV（llama gemv 移植）。
         * 单 int32x4 累加 8 个 vdot，每 block 一次 fma。 */
        vllm_q4_gemv_ctx vc = { out, q4_w, xq, xd, n_blocks, row_stride, rows };
        vllm_tp_parfor(0, (rows + 3) / 4, vllm_q4_gemv_worker, &vc);
        xq_free_canary(xq, (size_t)cols, "q4:xq");
        xq_free_canary(xd, (size_t)n_blocks * sizeof(float), "q4:xd");
        return;
    }
#endif
    vllm_q4_gemv_s_ctx vs = { out, q4_w, xq, xd, n_blocks, row_stride, rows };
    vllm_tp_parfor(0, rows, vllm_q4_gemv_s_worker, &vs);
    xq_free_canary(xq, (size_t)cols, "q4:xq");
    xq_free_canary(xd, (size_t)n_blocks * sizeof(float), "q4:xd");
}

typedef struct {
    float *gate_out, *up_out; const uint8_t *q4_gate, *q4_up;
    const int8_t *xq; const float *xd;
    int n_blocks, row_stride, rows;
} vllm_q4_gu_ctx;

static void vllm_q4_gu_worker(void *ctx_, int it) {
    vllm_q4_gu_ctx *c = ctx_;
    int r4 = it * 4;
    const uint8_t *__restrict pg = c->q4_gate + (size_t)(r4 >> 2) * c->n_blocks * 72;
    const uint8_t *__restrict pu = c->q4_up   + (size_t)(r4 >> 2) * c->n_blocks * 72;
    float32x4_t gacc = vdupq_n_f32(0.0f), uacc = vdupq_n_f32(0.0f);
    for (int b = 0; b < c->n_blocks; b++) {
        q4x4_dot1_group16_gemv(pg + (size_t)b * 72, c->xq + (size_t)b * 32,
                               c->xd[b], &gacc);
        q4x4_dot1_group16_gemv(pu + (size_t)b * 72, c->xq + (size_t)b * 32,
                               c->xd[b], &uacc);
        ST_PREFETCH(pg + (size_t)(b + 4) * 72);
        ST_PREFETCH(pu + (size_t)(b + 4) * 72);
    }
    c->gate_out[r4]     = vgetq_lane_f32(gacc, 0);
    c->gate_out[r4 + 1] = vgetq_lane_f32(gacc, 1);
    c->gate_out[r4 + 2] = vgetq_lane_f32(gacc, 2);
    c->gate_out[r4 + 3] = vgetq_lane_f32(gacc, 3);
    c->up_out[r4]     = vgetq_lane_f32(uacc, 0);
    c->up_out[r4 + 1] = vgetq_lane_f32(uacc, 1);
    c->up_out[r4 + 2] = vgetq_lane_f32(uacc, 2);
    c->up_out[r4 + 3] = vgetq_lane_f32(uacc, 3);
}

typedef struct {
    float *gate_out, *up_out; const uint8_t *q4_gate, *q4_up;
    const int8_t *xq; const float *xd;
    int n_blocks, row_stride, rows;
} vllm_q4_gu_s_ctx;

static void vllm_q4_gu_s_worker(void *ctx_, int r) {
    vllm_q4_gu_s_ctx *c = ctx_;
    const uint8_t *__restrict pg = c->q4_gate + (size_t)r * c->row_stride;
    const uint8_t *__restrict pu = c->q4_up   + (size_t)r * c->row_stride;
    float32x4_t ag0 = vdupq_n_f32(0.0f), ag1 = vdupq_n_f32(0.0f);
    float32x4_t au0 = vdupq_n_f32(0.0f), au1 = vdupq_n_f32(0.0f);
    for (int b = 0; b < c->n_blocks; b++) {
        int8x16_t gl, gh, ul, uh;
        q4x16_to_i8x32(pg + 2, &gl, &gh);
        q4x16_to_i8x32(pu + 2, &ul, &uh);
        int8x16_t alo = vld1q_s8(c->xq + (size_t)b * 32);
        int8x16_t ahi = vld1q_s8(c->xq + (size_t)b * 32 + 16);
        float32x4_t dgv = vdupq_n_f32(q4_block_scale(pg) * c->xd[b]);
        float32x4_t duv = vdupq_n_f32(q4_block_scale(pu) * c->xd[b]);
        ag0 = vfmaq_f32(ag0, vcvtq_f32_s32(i8x16_dot_s32(gl, alo)), dgv);
        ag1 = vfmaq_f32(ag1, vcvtq_f32_s32(i8x16_dot_s32(gh, ahi)), dgv);
        au0 = vfmaq_f32(au0, vcvtq_f32_s32(i8x16_dot_s32(ul, alo)), duv);
        au1 = vfmaq_f32(au1, vcvtq_f32_s32(i8x16_dot_s32(uh, ahi)), duv);
        pg += 18; pu += 18;
    }
    c->gate_out[r] = hsum_neon4(vaddq_f32(ag0, ag1));
    c->up_out[r]   = hsum_neon4(vaddq_f32(au0, au1));
}

static void dyn_matvec_q4_q8_fused_gate_up_neon(
    float *__restrict gate_out, float *__restrict up_out,
    const uint8_t *__restrict q4_gate, const uint8_t *__restrict q4_up,
    const float *__restrict x, int rows, int cols) {
    int n_blocks = cols / 32;
    int row_stride = n_blocks * 18;
    int8_t *xq = (int8_t *)xq_alloc_canary((size_t)cols);
    float *xd = (float *)xq_alloc_canary((size_t)n_blocks * sizeof(float));
    if (!xq || !xd) {
        xq_free_canary(xq, (size_t)cols, "q4_gu:xq");
        xq_free_canary(xd, (size_t)n_blocks * sizeof(float), "q4_gu:xd");
        return;
    }
    quantize_row_q8_0_act(x, xq, xd, cols);
#if ST_NEON_DOTPROD
    if (g_st_q4_repack && (rows & 3) == 0) {
        /* M4f: 4x4-repacked gate+up, shared activation, 16-element GEMV. */
        vllm_q4_gu_ctx vc = { gate_out, up_out, q4_gate, q4_up, xq, xd, n_blocks, row_stride, rows };
        vllm_tp_parfor(0, (rows + 3) / 4, vllm_q4_gu_worker, &vc);
        xq_free_canary(xq, (size_t)cols, "q4_gu:xq");
        xq_free_canary(xd, (size_t)n_blocks * sizeof(float), "q4_gu:xd");
        return;
    }
#endif
    vllm_q4_gu_s_ctx vs = { gate_out, up_out, q4_gate, q4_up, xq, xd, n_blocks, row_stride, rows };
    vllm_tp_parfor(0, rows, vllm_q4_gu_s_worker, &vs);
    xq_free_canary(xq, (size_t)cols, "q4_gu:xq");
    xq_free_canary(xd, (size_t)n_blocks * sizeof(float), "q4_gu:xd");
}

typedef struct {
    float *q_out, *k_out, *v_out; const uint8_t *q4_q, *q4_k, *q4_v;
    const int8_t *xq; const float *xd;
    int n_blocks, row_stride, q_rows, kv_rows;
} vllm_q4_qkv_ctx;

static void vllm_q4_qkv_worker(void *ctx_, int it) {
    vllm_q4_qkv_ctx *c = ctx_;
    int r4 = it * 4;
    const uint8_t *__restrict pq = c->q4_q + (size_t)(r4 >> 2) * c->n_blocks * 72;
    const uint8_t *__restrict pk = c->q4_k + (size_t)(r4 >> 2) * c->n_blocks * 72;
    const uint8_t *__restrict pv = c->q4_v + (size_t)(r4 >> 2) * c->n_blocks * 72;
    float32x4_t qacc = vdupq_n_f32(0.0f), kacc = vdupq_n_f32(0.0f),
                vacc = vdupq_n_f32(0.0f);
    for (int b = 0; b < c->n_blocks; b++) {
        q4x4_dot1_group16_gemv(pq + (size_t)b * 72, c->xq + (size_t)b * 32,
                               c->xd[b], &qacc);
        q4x4_dot1_group16_gemv(pk + (size_t)b * 72, c->xq + (size_t)b * 32,
                               c->xd[b], &kacc);
        q4x4_dot1_group16_gemv(pv + (size_t)b * 72, c->xq + (size_t)b * 32,
                               c->xd[b], &vacc);
        ST_PREFETCH(pq + (size_t)(b + 4) * 72);
    }
    c->q_out[r4]     = vgetq_lane_f32(qacc, 0);
    c->q_out[r4 + 1] = vgetq_lane_f32(qacc, 1);
    c->q_out[r4 + 2] = vgetq_lane_f32(qacc, 2);
    c->q_out[r4 + 3] = vgetq_lane_f32(qacc, 3);
    c->k_out[r4]     = vgetq_lane_f32(kacc, 0);
    c->k_out[r4 + 1] = vgetq_lane_f32(kacc, 1);
    c->k_out[r4 + 2] = vgetq_lane_f32(kacc, 2);
    c->k_out[r4 + 3] = vgetq_lane_f32(kacc, 3);
    c->v_out[r4]     = vgetq_lane_f32(vacc, 0);
    c->v_out[r4 + 1] = vgetq_lane_f32(vacc, 1);
    c->v_out[r4 + 2] = vgetq_lane_f32(vacc, 2);
    c->v_out[r4 + 3] = vgetq_lane_f32(vacc, 3);
}

typedef struct {
    float *q_out; const uint8_t *q4_q;
    const int8_t *xq; const float *xd;
    int n_blocks, q_rows;
} vllm_q4_qkv_q_ctx;

static void vllm_q4_qkv_q_worker(void *ctx_, int it) {
    vllm_q4_qkv_q_ctx *c = ctx_;
    int r4 = it * 4;
    const uint8_t *__restrict pq = c->q4_q + (size_t)(r4 >> 2) * c->n_blocks * 72;
    float32x4_t qacc = vdupq_n_f32(0.0f);
    for (int b = 0; b < c->n_blocks; b++) {
        q4x4_dot1_group16_gemv(pq + (size_t)b * 72, c->xq + (size_t)b * 32,
                               c->xd[b], &qacc);
        ST_PREFETCH(pq + (size_t)(b + 4) * 72);
    }
    c->q_out[r4]     = vgetq_lane_f32(qacc, 0);
    c->q_out[r4 + 1] = vgetq_lane_f32(qacc, 1);
    c->q_out[r4 + 2] = vgetq_lane_f32(qacc, 2);
    c->q_out[r4 + 3] = vgetq_lane_f32(qacc, 3);
}

typedef struct {
    float *q_out, *k_out, *v_out; const uint8_t *q4_q, *q4_k, *q4_v;
    const int8_t *xq; const float *xd;
    int n_blocks, row_stride, kv_rows, q_rows;
} vllm_q4_qkv_s_ctx;

static void vllm_q4_qkv_s_worker(void *ctx_, int r) {
    vllm_q4_qkv_s_ctx *c = ctx_;
    const uint8_t *__restrict pq = c->q4_q + (size_t)r * c->row_stride;
    const uint8_t *__restrict pk = NULL;
    const uint8_t *__restrict pv = NULL;
    int is_kv = (r < c->kv_rows);
    if (is_kv) { pk = c->q4_k + (size_t)r * c->row_stride; pv = c->q4_v + (size_t)r * c->row_stride; }
    float32x4_t aq0 = vdupq_n_f32(0.0f), aq1 = vdupq_n_f32(0.0f);
    float32x4_t ak0 = vdupq_n_f32(0.0f), ak1 = vdupq_n_f32(0.0f);
    float32x4_t av0 = vdupq_n_f32(0.0f), av1 = vdupq_n_f32(0.0f);
    for (int b = 0; b < c->n_blocks; b++) {
        int8x16_t ql, qh, kl, kh, vl, vh;
        q4x16_to_i8x32(pq + 2, &ql, &qh);
        if (is_kv) {
            q4x16_to_i8x32(pk + 2, &kl, &kh);
            q4x16_to_i8x32(pv + 2, &vl, &vh);
        }
        int8x16_t alo = vld1q_s8(c->xq + (size_t)b * 32);
        int8x16_t ahi = vld1q_s8(c->xq + (size_t)b * 32 + 16);
        float32x4_t dqv = vdupq_n_f32(q4_block_scale(pq) * c->xd[b]);
        aq0 = vfmaq_f32(aq0, vcvtq_f32_s32(i8x16_dot_s32(ql, alo)), dqv);
        aq1 = vfmaq_f32(aq1, vcvtq_f32_s32(i8x16_dot_s32(qh, ahi)), dqv);
        if (is_kv) {
            float32x4_t dkv = vdupq_n_f32(q4_block_scale(pk) * c->xd[b]);
            ak0 = vfmaq_f32(ak0, vcvtq_f32_s32(i8x16_dot_s32(kl, alo)), dkv);
            ak1 = vfmaq_f32(ak1, vcvtq_f32_s32(i8x16_dot_s32(kh, ahi)), dkv);
            float32x4_t dvv = vdupq_n_f32(q4_block_scale(pv) * c->xd[b]);
            av0 = vfmaq_f32(av0, vcvtq_f32_s32(i8x16_dot_s32(vl, alo)), dvv);
            av1 = vfmaq_f32(av1, vcvtq_f32_s32(i8x16_dot_s32(vh, ahi)), dvv);
        }
        pq += 18;
        if (is_kv) { pk += 18; pv += 18; }
    }
    c->q_out[r] = hsum_neon4(vaddq_f32(aq0, aq1));
    if (is_kv) {
        c->k_out[r] = hsum_neon4(vaddq_f32(ak0, ak1));
        c->v_out[r] = hsum_neon4(vaddq_f32(av0, av1));
    }
}

static void dyn_matvec_q4_q8_fused_qkv_neon(
    float *__restrict q_out, float *__restrict k_out, float *__restrict v_out,
    const uint8_t *__restrict q4_q, const uint8_t *__restrict q4_k,
    const uint8_t *__restrict q4_v, const float *__restrict x,
    int q_rows, int kv_rows, int cols) {
    int n_blocks = cols / 32;
    int row_stride = n_blocks * 18;
    int8_t *xq = (int8_t *)xq_alloc_canary((size_t)cols);
    float *xd = (float *)xq_alloc_canary((size_t)n_blocks * sizeof(float));
    if (!xq || !xd) {
        xq_free_canary(xq, (size_t)cols, "q4_qkv:xq");
        xq_free_canary(xd, (size_t)n_blocks * sizeof(float), "q4_qkv:xd");
        return;
    }
    quantize_row_q8_0_act(x, xq, xd, cols);
#if ST_NEON_DOTPROD
    if (g_st_q4_repack && (q_rows & 3) == 0 && (kv_rows & 3) == 0) {
        /* M4f: 4x4-repacked Q/K/V, 16-element GEMV. Two row blocks so a
         * 4-row group never straddles the kv boundary. */
        /* P3 (2026-08-29): llama.cpp-style split q/k/v row-sliced GEMVs were
         * A/B'd (VLLM_QKV_SPLIT, 64-token means): qkv 6.383 vs 6.378ms — no
         * gain. The 2B qkv stream is only ~4.7MB; the fused q/k/v interleave
         * costs nothing in DRAM row locality. Keep the fused worker. */
        vllm_q4_qkv_ctx vc = { q_out, k_out, v_out, q4_q, q4_k, q4_v, xq, xd, n_blocks, row_stride, q_rows, kv_rows };
        /* P3b (2026-08-29): a single 512-task q/k/v parfor (merged worker) was
         * A/B'd (VLLM_QKV_1PARFOR): other 1.495 vs 1.521ms — no gain, within
         * run noise (±1ms). Keep the two-launch fused path. */
        vllm_tp_parfor(0, kv_rows / 4, vllm_q4_qkv_worker, &vc);
        vllm_q4_qkv_q_ctx vq = { q_out, q4_q, xq, xd, n_blocks, q_rows };
        vllm_tp_parfor(kv_rows / 4, q_rows / 4, vllm_q4_qkv_q_worker, &vq);
        xq_free_canary(xq, (size_t)cols, "q4_qkv:xq");
        xq_free_canary(xd, (size_t)n_blocks * sizeof(float), "q4_qkv:xd");
        return;
    }
#endif
    vllm_q4_qkv_s_ctx vs = { q_out, k_out, v_out, q4_q, q4_k, q4_v, xq, xd, n_blocks, row_stride, kv_rows, q_rows };
    vllm_tp_parfor(0, q_rows, vllm_q4_qkv_s_worker, &vs);
    xq_free_canary(xq, (size_t)cols, "q4_qkv:xq");
    xq_free_canary(xd, (size_t)n_blocks * sizeof(float), "q4_qkv:xd");
}

typedef struct {
    float *x; const float *residual; const uint8_t *q4_o;
    const int8_t *xq; const float *xd;
    int n_blocks, row_stride, rows;
} vllm_q4_o_ctx;

static void vllm_q4_o_worker(void *ctx_, int it) {
    vllm_q4_o_ctx *c = ctx_;
    int r4 = it * 4;
    const uint8_t *__restrict pr = c->q4_o + (size_t)(r4 >> 2) * c->n_blocks * 72;
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (int b = 0; b < c->n_blocks; b++) {
        q4x4_dot1_group16_gemv(pr + (size_t)b * 72, c->xq + (size_t)b * 32,
                               c->xd[b], &acc);
        ST_PREFETCH(pr + (size_t)(b + 4) * 72);
    }
    c->x[r4]     = c->residual[r4]     + vgetq_lane_f32(acc, 0);
    c->x[r4 + 1] = c->residual[r4 + 1] + vgetq_lane_f32(acc, 1);
    c->x[r4 + 2] = c->residual[r4 + 2] + vgetq_lane_f32(acc, 2);
    c->x[r4 + 3] = c->residual[r4 + 3] + vgetq_lane_f32(acc, 3);
}

typedef struct {
    float *x; const float *residual; const uint8_t *q4_o;
    const int8_t *xq; const float *xd;
    int n_blocks, row_stride, rows;
} vllm_q4_o_s_ctx;

static void vllm_q4_o_s_worker(void *ctx_, int r) {
    vllm_q4_o_s_ctx *c = ctx_;
    const uint8_t *__restrict pr = c->q4_o + (size_t)r * c->row_stride;
    float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = vdupq_n_f32(0.0f);
    for (int b = 0; b < c->n_blocks; b++) {
        int8x16_t wlo, whi;
        q4x16_to_i8x32(pr + 2, &wlo, &whi);
        int8x16_t alo = vld1q_s8(c->xq + (size_t)b * 32);
        int8x16_t ahi = vld1q_s8(c->xq + (size_t)b * 32 + 16);
        float32x4_t dsv = vdupq_n_f32(q4_block_scale(pr) * c->xd[b]);
        acc0 = vfmaq_f32(acc0, vcvtq_f32_s32(i8x16_dot_s32(wlo, alo)), dsv);
        acc1 = vfmaq_f32(acc1, vcvtq_f32_s32(i8x16_dot_s32(whi, ahi)), dsv);
        pr += 18;
    }
    c->x[r] = c->residual[r] + hsum_neon4(vaddq_f32(acc0, acc1));
}

static void dyn_matvec_q4_q8_fused_o_residual_neon(
    float *__restrict x, const float *__restrict residual,
    const uint8_t *__restrict q4_o, const float *__restrict attn_in,
    int rows, int cols) {
    int n_blocks = cols / 32;
    int row_stride = n_blocks * 18;
    int8_t *xq = (int8_t *)xq_alloc_canary((size_t)cols);
    float *xd = (float *)xq_alloc_canary((size_t)n_blocks * sizeof(float));
    if (!xq || !xd) {
        xq_free_canary(xq, (size_t)cols, "q4_o:xq");
        xq_free_canary(xd, (size_t)n_blocks * sizeof(float), "q4_o:xd");
        return;
    }
    quantize_row_q8_0_act(attn_in, xq, xd, cols);
#if ST_NEON_DOTPROD
    if (g_st_q4_repack && (rows & 3) == 0) {
        /* M4f: 4x4-repacked O projection + residual, 16-element GEMV. */
        vllm_q4_o_ctx vc = { x, residual, q4_o, xq, xd, n_blocks, row_stride, rows };
        vllm_tp_parfor(0, (rows + 3) / 4, vllm_q4_o_worker, &vc);
        xq_free_canary(xq, (size_t)cols, "q4_o:xq");
        xq_free_canary(xd, (size_t)n_blocks * sizeof(float), "q4_o:xd");
        return;
    }
#endif
    vllm_q4_o_s_ctx vs = { x, residual, q4_o, xq, xd, n_blocks, row_stride, rows };
    vllm_tp_parfor(0, rows, vllm_q4_o_s_worker, &vs);
    xq_free_canary(xq, (size_t)cols, "q4_o:xq");
    xq_free_canary(xd, (size_t)n_blocks * sizeof(float), "q4_o:xd");
}

typedef struct {
    float *x; const float *residual; const uint8_t *q4_down;
    const int8_t *xq; const float *xd;
    int n_blocks, row_stride, hidden_dim;
} vllm_q4_dn_ctx;

static void vllm_q4_dn_worker(void *ctx_, int it) {
    vllm_q4_dn_ctx *c = ctx_;
    int j4 = it * 4;
    const uint8_t *__restrict pr = c->q4_down + (size_t)(j4 >> 2) * c->n_blocks * 72;
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (int b = 0; b < c->n_blocks; b++) {
        q4x4_dot1_group16_gemv(pr + (size_t)b * 72, c->xq + (size_t)b * 32,
                               c->xd[b], &acc);
        ST_PREFETCH(pr + (size_t)(b + 4) * 72);
    }
    c->x[j4]     = c->residual[j4]     + vgetq_lane_f32(acc, 0);
    c->x[j4 + 1] = c->residual[j4 + 1] + vgetq_lane_f32(acc, 1);
    c->x[j4 + 2] = c->residual[j4 + 2] + vgetq_lane_f32(acc, 2);
    c->x[j4 + 3] = c->residual[j4 + 3] + vgetq_lane_f32(acc, 3);
}

typedef struct {
    float *x; const float *residual; const uint8_t *q4_down;
    const int8_t *xq; const float *xd;
    int n_blocks, row_stride, hidden_dim;
} vllm_q4_dn_s_ctx;

static void vllm_q4_dn_s_worker(void *ctx_, int j) {
    vllm_q4_dn_s_ctx *c = ctx_;
    const uint8_t *__restrict pr = c->q4_down + (size_t)j * c->row_stride;
    float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = vdupq_n_f32(0.0f);
    for (int b = 0; b < c->n_blocks; b++) {
        int8x16_t wlo, whi;
        q4x16_to_i8x32(pr + 2, &wlo, &whi);
        int8x16_t alo = vld1q_s8(c->xq + (size_t)b * 32);
        int8x16_t ahi = vld1q_s8(c->xq + (size_t)b * 32 + 16);
        float32x4_t dsv = vdupq_n_f32(q4_block_scale(pr) * c->xd[b]);
        acc0 = vfmaq_f32(acc0, vcvtq_f32_s32(i8x16_dot_s32(wlo, alo)), dsv);
        acc1 = vfmaq_f32(acc1, vcvtq_f32_s32(i8x16_dot_s32(whi, ahi)), dsv);
        pr += 18;
    }
    c->x[j] = c->residual[j] + hsum_neon4(vaddq_f32(acc0, acc1));
}

static void dyn_matvec_q4_q8_fused_down_residual_neon(
    float *__restrict x, const float *__restrict residual,
    const uint8_t *__restrict q4_down, const float *__restrict activated,
    int hidden_dim, int ffn_dim) {
    int n_blocks = ffn_dim / 32;
    int row_stride = n_blocks * 18;
    int8_t *xq = (int8_t *)xq_alloc_canary((size_t)ffn_dim);
    float *xd = (float *)xq_alloc_canary((size_t)n_blocks * sizeof(float));
    if (!xq || !xd) {
        xq_free_canary(xq, (size_t)ffn_dim, "q4_dn:xq");
        xq_free_canary(xd, (size_t)n_blocks * sizeof(float), "q4_dn:xd");
        return;
    }
    quantize_row_q8_0_act(activated, xq, xd, ffn_dim);
#if ST_NEON_DOTPROD
    if (g_st_q4_repack && (hidden_dim & 3) == 0) {
        /* M4f: 4x4-repacked down projection + residual, 16-element GEMV. */
        vllm_q4_dn_ctx vc = { x, residual, q4_down, xq, xd, n_blocks, row_stride, hidden_dim };
        vllm_tp_parfor(0, (hidden_dim + 3) / 4, vllm_q4_dn_worker, &vc);
        xq_free_canary(xq, (size_t)ffn_dim, "q4_dn:xq");
        xq_free_canary(xd, (size_t)n_blocks * sizeof(float), "q4_dn:xd");
        return;
    }
#endif
    vllm_q4_dn_s_ctx vs = { x, residual, q4_down, xq, xd, n_blocks, row_stride, hidden_dim };
    vllm_tp_parfor(0, hidden_dim, vllm_q4_dn_s_worker, &vs);
    xq_free_canary(xq, (size_t)ffn_dim, "q4_dn:xq");
    xq_free_canary(xd, (size_t)n_blocks * sizeof(float), "q4_dn:xd");
}

/* ---- Q4_0 batched kernels (token-major quantized activation) ---- */

typedef struct {
    const float *x_batch; int8_t *xq; float *xd;
    int cols, n_blocks, n_batch;
} vllm_q4_qkv_quant_ctx;

static void vllm_q4_qkv_quant_worker(void *ctx_, int t) {
    vllm_q4_qkv_quant_ctx *c = ctx_;
    quantize_row_q8_0_act(c->x_batch + (size_t)t * c->cols,
                          c->xq + (size_t)t * c->cols,
                          c->xd + (size_t)t * c->n_blocks, c->cols);
}

typedef struct {
    float *q_out, *k_out, *v_out; const uint8_t *q4_q, *q4_k, *q4_v;
    const int8_t *xq; const float *xd;
    int n_blocks, row_stride, cols, q_rows, kv_rows, n_batch;
} vllm_q4_qkvb_kv_ctx;

static void vllm_q4_qkvb_kv_worker(void *ctx_, int r) {
    vllm_q4_qkvb_kv_ctx *c = ctx_;
    const uint8_t *__restrict pq = c->q4_q + (size_t)r * c->row_stride;
    const uint8_t *__restrict pk = c->q4_k + (size_t)r * c->row_stride;
    const uint8_t *__restrict pv = c->q4_v + (size_t)r * c->row_stride;
    for (int t0 = 0; t0 < c->n_batch; t0 += NEON_TILE_QKV) {
        int nt = c->n_batch - t0;
        if (nt > NEON_TILE_QKV) nt = NEON_TILE_QKV;
        float32x4_t aq[NEON_TILE_QKV][2], ak[NEON_TILE_QKV][2], av[NEON_TILE_QKV][2];
        for (int tt = 0; tt < nt; tt++) {
            aq[tt][0] = vdupq_n_f32(0.0f); aq[tt][1] = vdupq_n_f32(0.0f);
            ak[tt][0] = vdupq_n_f32(0.0f); ak[tt][1] = vdupq_n_f32(0.0f);
            av[tt][0] = vdupq_n_f32(0.0f); av[tt][1] = vdupq_n_f32(0.0f);
        }
        for (int b = 0; b < c->n_blocks; b++) {
            int8x16_t wq0, wq1, wk0, wk1, wv0, wv1;
            q4x16_to_i8x32(pq + (size_t)b * 18 + 2, &wq0, &wq1);
            q4x16_to_i8x32(pk + (size_t)b * 18 + 2, &wk0, &wk1);
            q4x16_to_i8x32(pv + (size_t)b * 18 + 2, &wv0, &wv1);
            float dqw = q4_block_scale(pq + (size_t)b * 18);
            float dkw = q4_block_scale(pk + (size_t)b * 18);
            float dvw = q4_block_scale(pv + (size_t)b * 18);
            for (int tt = 0; tt < nt; tt++) {
                int t = t0 + tt;
                int8x16_t x0 = vld1q_s8(c->xq + (size_t)t * c->cols + (size_t)b * 32);
                int8x16_t x1 = vld1q_s8(c->xq + (size_t)t * c->cols + (size_t)b * 32 + 16);
                float dx = c->xd[(size_t)t * c->n_blocks + b];
                float32x4_t dsv = vdupq_n_f32(dqw * dx);
                aq[tt][0] = vfmaq_f32(aq[tt][0], vcvtq_f32_s32(i8x16_dot_s32(wq0, x0)), dsv);
                aq[tt][1] = vfmaq_f32(aq[tt][1], vcvtq_f32_s32(i8x16_dot_s32(wq1, x1)), dsv);
                float32x4_t dsv2 = vdupq_n_f32(dkw * dx);
                ak[tt][0] = vfmaq_f32(ak[tt][0], vcvtq_f32_s32(i8x16_dot_s32(wk0, x0)), dsv2);
                ak[tt][1] = vfmaq_f32(ak[tt][1], vcvtq_f32_s32(i8x16_dot_s32(wk1, x1)), dsv2);
                float32x4_t dsv3 = vdupq_n_f32(dvw * dx);
                av[tt][0] = vfmaq_f32(av[tt][0], vcvtq_f32_s32(i8x16_dot_s32(wv0, x0)), dsv3);
                av[tt][1] = vfmaq_f32(av[tt][1], vcvtq_f32_s32(i8x16_dot_s32(wv1, x1)), dsv3);
            }
        }
        for (int tt = 0; tt < nt; tt++) {
            int t = t0 + tt;
            c->q_out[(size_t)t * c->q_rows + r] =
                hsum_neon4(vaddq_f32(aq[tt][0], aq[tt][1]));
            c->k_out[(size_t)t * c->kv_rows + r] =
                hsum_neon4(vaddq_f32(ak[tt][0], ak[tt][1]));
            c->v_out[(size_t)t * c->kv_rows + r] =
                hsum_neon4(vaddq_f32(av[tt][0], av[tt][1]));
        }
    }
}

typedef struct {
    float *q_out; const uint8_t *q4_q;
    const int8_t *xq; const float *xd;
    int n_blocks, row_stride, cols, q_rows, n_batch;
} vllm_q4_qkvb_q_ctx;

static void vllm_q4_qkvb_q_worker(void *ctx_, int r) {
    vllm_q4_qkvb_q_ctx *c = ctx_;
    const uint8_t *__restrict pq = c->q4_q + (size_t)r * c->row_stride;
    for (int t0 = 0; t0 < c->n_batch; t0 += NEON_TILE_QKV_Q) {
        int nt = c->n_batch - t0;
        if (nt > NEON_TILE_QKV_Q) nt = NEON_TILE_QKV_Q;
        float32x4_t aq[NEON_TILE_QKV_Q][2];
        for (int tt = 0; tt < nt; tt++) {
            aq[tt][0] = vdupq_n_f32(0.0f); aq[tt][1] = vdupq_n_f32(0.0f);
        }
        for (int b = 0; b < c->n_blocks; b++) {
            int8x16_t wq0, wq1;
            q4x16_to_i8x32(pq + (size_t)b * 18 + 2, &wq0, &wq1);
            float dqw = q4_block_scale(pq + (size_t)b * 18);
            for (int tt = 0; tt < nt; tt++) {
                int t = t0 + tt;
                int8x16_t x0 = vld1q_s8(c->xq + (size_t)t * c->cols + (size_t)b * 32);
                int8x16_t x1 = vld1q_s8(c->xq + (size_t)t * c->cols + (size_t)b * 32 + 16);
                float dx = c->xd[(size_t)t * c->n_blocks + b];
                float32x4_t dsv = vdupq_n_f32(dqw * dx);
                aq[tt][0] = vfmaq_f32(aq[tt][0], vcvtq_f32_s32(i8x16_dot_s32(wq0, x0)), dsv);
                aq[tt][1] = vfmaq_f32(aq[tt][1], vcvtq_f32_s32(i8x16_dot_s32(wq1, x1)), dsv);
            }
        }
        for (int tt = 0; tt < nt; tt++) {
            int t = t0 + tt;
            c->q_out[(size_t)t * c->q_rows + r] =
                hsum_neon4(vaddq_f32(aq[tt][0], aq[tt][1]));
        }
    }
}

static void dyn_matvec_q4_q8_fused_qkv_batched_neon(
    float *__restrict q_out, float *__restrict k_out, float *__restrict v_out,
    const uint8_t *__restrict q4_q, const uint8_t *__restrict q4_k,
    const uint8_t *__restrict q4_v, const float *__restrict x_batch,
    int q_rows, int kv_rows, int cols, int n_batch) {
    int n_blocks = cols / 32;
    int row_stride = n_blocks * 18;
    int8_t *xq; float *xd;
    if (xq_scratch_ensure(&xq, &xd,
                          (size_t)n_batch * cols,
                          (size_t)n_batch * n_blocks * sizeof(float)) != 0)
        return;
    {
        vllm_q4_qkv_quant_ctx vc = { x_batch, xq, xd, cols, n_blocks, n_batch };
        vllm_tp_parfor(0, n_batch, vllm_q4_qkv_quant_worker, &vc);
    }
#if ST_NEON_DOTPROD
    if (g_st_q4_repack && (q_rows & 3) == 0 && (kv_rows & 3) == 0) {
        /* M4b: 真 GEMM batched Q/K/V. 单矩阵 pass（q 全行，k/v 各 kv 行），每个
         * pass 内权重 block 读一次、3-token tile 共享 - 权重 DRAM 流量 ÷3，
         * 且无 fused 多矩阵累加器的寄存器溢出。单 token 的 b 累加顺序与原
         * 实现一致 -> 位级一致。 */
        q4x4_gemm_batched(q_out, q4_q, xq, xd, q_rows, cols, n_batch, NULL);
        q4x4_gemm_batched(k_out, q4_k, xq, xd, kv_rows, cols, n_batch, NULL);
        q4x4_gemm_batched(v_out, q4_v, xq, xd, kv_rows, cols, n_batch, NULL);
        return;
    }
#endif
    /* Row-split branchless loops: rows [0,kv_rows) compute Q+K+V, rows
     * [kv_rows,q_rows) compute Q only. Both keep the same b-ascending dot
     * accumulation order, so results stay bit-identical to the old fused
     * is_kv loop, but the Q-only half drops the K/V unpack + acc entirely
     * (measured: QKV batched is unpack-bound on Q4_0). */
    vllm_q4_qkvb_kv_ctx vk = { q_out, k_out, v_out, q4_q, q4_k, q4_v, xq, xd, n_blocks, row_stride, cols, q_rows, kv_rows, n_batch };
    vllm_tp_parfor(0, kv_rows, vllm_q4_qkvb_kv_worker, &vk);
    vllm_q4_qkvb_q_ctx vq = { q_out, q4_q, xq, xd, n_blocks, row_stride, cols, q_rows, n_batch };
    vllm_tp_parfor(kv_rows, q_rows, vllm_q4_qkvb_q_worker, &vq);
}

typedef struct {
    const float *attn_batch; int8_t *xq; float *xd;
    int cols, n_blocks, n_batch;
} vllm_q4_o_quant_ctx;

static void vllm_q4_o_quant_worker(void *ctx_, int t) {
    vllm_q4_o_quant_ctx *c = ctx_;
    quantize_row_q8_0_act(c->attn_batch + (size_t)t * c->cols,
                          c->xq + (size_t)t * c->cols,
                          c->xd + (size_t)t * c->n_blocks, c->cols);
}

typedef struct {
    float *x_batch; const float *residual_batch; const uint8_t *q4_o;
    const int8_t *xq; const float *xd;
    int n_blocks, row_stride, cols, rows, n_batch;
} vllm_q4_o_b_ctx;

static void vllm_q4_o_b_worker(void *ctx_, int r) {
    vllm_q4_o_b_ctx *c = ctx_;
    const uint8_t *__restrict pr = c->q4_o + (size_t)r * c->row_stride;
    for (int t0 = 0; t0 < c->n_batch; t0 += NEON_TILE_1) {
        int nt = c->n_batch - t0;
        if (nt > NEON_TILE_1) nt = NEON_TILE_1;
        float32x4_t acc[NEON_TILE_1][2];
        for (int tt = 0; tt < nt; tt++) {
            acc[tt][0] = vdupq_n_f32(0.0f);
            acc[tt][1] = vdupq_n_f32(0.0f);
        }
        for (int b = 0; b < c->n_blocks; b++) {
            int8x16_t w0, w1;
            q4x16_to_i8x32(pr + (size_t)b * 18 + 2, &w0, &w1);
            float dw = q4_block_scale(pr + (size_t)b * 18);
            for (int tt = 0; tt < nt; tt++) {
                int t = t0 + tt;
                int8x16_t x0 = vld1q_s8(c->xq + (size_t)t * c->cols + (size_t)b * 32);
                int8x16_t x1 = vld1q_s8(c->xq + (size_t)t * c->cols + (size_t)b * 32 + 16);
                float dx = c->xd[(size_t)t * c->n_blocks + b];
                float32x4_t dsv = vdupq_n_f32(dw * dx);
                acc[tt][0] = vfmaq_f32(acc[tt][0], vcvtq_f32_s32(i8x16_dot_s32(w0, x0)), dsv);
                acc[tt][1] = vfmaq_f32(acc[tt][1], vcvtq_f32_s32(i8x16_dot_s32(w1, x1)), dsv);
            }
        }
        for (int tt = 0; tt < nt; tt++) {
            int t = t0 + tt;
            c->x_batch[(size_t)t * c->rows + r] =
                c->residual_batch[(size_t)t * c->rows + r] +
                hsum_neon4(vaddq_f32(acc[tt][0], acc[tt][1]));
        }
    }
}

static void dyn_matvec_q4_q8_fused_o_residual_batched_neon(
    float *__restrict x_batch, const float *__restrict residual_batch,
    const uint8_t *__restrict q4_o, const float *__restrict attn_batch,
    int rows, int cols, int n_batch) {
    int n_blocks = cols / 32;
    int row_stride = n_blocks * 18;
    int8_t *xq; float *xd;
    if (xq_scratch_ensure(&xq, &xd,
                          (size_t)n_batch * cols,
                          (size_t)n_batch * n_blocks * sizeof(float)) != 0)
        return;
    {
        vllm_q4_o_quant_ctx vc = { attn_batch, xq, xd, cols, n_blocks, n_batch };
        vllm_tp_parfor(0, n_batch, vllm_q4_o_quant_worker, &vc);
    }
#if ST_NEON_DOTPROD
    if (g_st_q4_repack && (rows & 3) == 0) {
        /* M4b: 真 GEMM batched O + residual（单矩阵 pass，3-token tile）。 */
        q4x4_gemm_batched(x_batch, q4_o, xq, xd, rows, cols, n_batch, residual_batch);
        return;
    }
#endif
    vllm_q4_o_b_ctx vb = { x_batch, residual_batch, q4_o, xq, xd, n_blocks, row_stride, cols, rows, n_batch };
    vllm_tp_parfor(0, rows, vllm_q4_o_b_worker, &vb);
}

typedef struct {
    const float *x_batch; int8_t *xq; float *xd;
    int cols, n_blocks, n_batch;
} vllm_q4_gu_quant_ctx;

static void vllm_q4_gu_quant_worker(void *ctx_, int t) {
    vllm_q4_gu_quant_ctx *c = ctx_;
    quantize_row_q8_0_act(c->x_batch + (size_t)t * c->cols,
                          c->xq + (size_t)t * c->cols,
                          c->xd + (size_t)t * c->n_blocks, c->cols);
}

typedef struct {
    float *gate_out, *up_out; const uint8_t *q4_gate, *q4_up;
    const int8_t *xq; const float *xd;
    int n_blocks, row_stride, cols, rows, n_batch;
} vllm_q4_gu_b_ctx;

static void vllm_q4_gu_b_worker(void *ctx_, int r) {
    vllm_q4_gu_b_ctx *c = ctx_;
    const uint8_t *__restrict pg = c->q4_gate + (size_t)r * c->row_stride;
    const uint8_t *__restrict pu = c->q4_up   + (size_t)r * c->row_stride;
    for (int t0 = 0; t0 < c->n_batch; t0 += NEON_TILE_GU) {
        int nt = c->n_batch - t0;
        if (nt > NEON_TILE_GU) nt = NEON_TILE_GU;
        float32x4_t ag[NEON_TILE_GU][2], au[NEON_TILE_GU][2];
        for (int tt = 0; tt < nt; tt++) {
            ag[tt][0] = vdupq_n_f32(0.0f); ag[tt][1] = vdupq_n_f32(0.0f);
            au[tt][0] = vdupq_n_f32(0.0f); au[tt][1] = vdupq_n_f32(0.0f);
        }
        for (int b = 0; b < c->n_blocks; b++) {
            int8x16_t wg0, wg1, wu0, wu1;
            q4x16_to_i8x32(pg + (size_t)b * 18 + 2, &wg0, &wg1);
            q4x16_to_i8x32(pu + (size_t)b * 18 + 2, &wu0, &wu1);
            float dg = q4_block_scale(pg + (size_t)b * 18);
            float du = q4_block_scale(pu + (size_t)b * 18);
            for (int tt = 0; tt < nt; tt++) {
                int t = t0 + tt;
                int8x16_t x0 = vld1q_s8(c->xq + (size_t)t * c->cols + (size_t)b * 32);
                int8x16_t x1 = vld1q_s8(c->xq + (size_t)t * c->cols + (size_t)b * 32 + 16);
                float dx = c->xd[(size_t)t * c->n_blocks + b];
                float32x4_t dgv = vdupq_n_f32(dg * dx);
                float32x4_t duv = vdupq_n_f32(du * dx);
                ag[tt][0] = vfmaq_f32(ag[tt][0], vcvtq_f32_s32(i8x16_dot_s32(wg0, x0)), dgv);
                ag[tt][1] = vfmaq_f32(ag[tt][1], vcvtq_f32_s32(i8x16_dot_s32(wg1, x1)), dgv);
                au[tt][0] = vfmaq_f32(au[tt][0], vcvtq_f32_s32(i8x16_dot_s32(wu0, x0)), duv);
                au[tt][1] = vfmaq_f32(au[tt][1], vcvtq_f32_s32(i8x16_dot_s32(wu1, x1)), duv);
            }
        }
        for (int tt = 0; tt < nt; tt++) {
            int t = t0 + tt;
            c->gate_out[(size_t)t * c->rows + r] =
                hsum_neon4(vaddq_f32(ag[tt][0], ag[tt][1]));
            c->up_out  [(size_t)t * c->rows + r] =
                hsum_neon4(vaddq_f32(au[tt][0], au[tt][1]));
        }
    }
}

static void dyn_matvec_q4_q8_fused_gate_up_batched_neon(
    float *__restrict gate_out, float *__restrict up_out,
    const uint8_t *__restrict q4_gate, const uint8_t *__restrict q4_up,
    const float *__restrict x_batch, int rows, int cols, int n_batch) {
    int n_blocks = cols / 32;
    int row_stride = n_blocks * 18;
    int8_t *xq; float *xd;
    if (xq_scratch_ensure(&xq, &xd,
                          (size_t)n_batch * cols,
                          (size_t)n_batch * n_blocks * sizeof(float)) != 0)
        return;
    {
        vllm_q4_gu_quant_ctx vc = { x_batch, xq, xd, cols, n_blocks, n_batch };
        vllm_tp_parfor(0, n_batch, vllm_q4_gu_quant_worker, &vc);
    }
#if ST_NEON_DOTPROD
    if (g_st_q4_repack && (rows & 3) == 0) {
        /* M4b: 真 GEMM batched gate+up（两个单矩阵 pass，各 3-token tile；
         * 共享已量化的 xq/xd，激活在 L1 重读代价可忽略）。 */
        q4x4_gemm_batched(gate_out, q4_gate, xq, xd, rows, cols, n_batch, NULL);
        q4x4_gemm_batched(up_out, q4_up, xq, xd, rows, cols, n_batch, NULL);
        return;
    }
#endif
    vllm_q4_gu_b_ctx vb = { gate_out, up_out, q4_gate, q4_up, xq, xd, n_blocks, row_stride, cols, rows, n_batch };
    vllm_tp_parfor(0, rows, vllm_q4_gu_b_worker, &vb);
}

typedef struct {
    const float *activated_batch; int8_t *xq; float *xd;
    int ffn_dim, n_blocks, n_batch;
} vllm_q4_dn_quant_ctx;

static void vllm_q4_dn_quant_worker(void *ctx_, int t) {
    vllm_q4_dn_quant_ctx *c = ctx_;
    quantize_row_q8_0_act(c->activated_batch + (size_t)t * c->ffn_dim,
                          c->xq + (size_t)t * c->ffn_dim,
                          c->xd + (size_t)t * c->n_blocks, c->ffn_dim);
}

typedef struct {
    float *x_batch; const float *residual_batch; const uint8_t *q4_down;
    const int8_t *xq; const float *xd;
    int n_blocks, row_stride, ffn_dim, hidden_dim, n_batch;
} vllm_q4_dn_b_ctx;

static void vllm_q4_dn_b_worker(void *ctx_, int j) {
    vllm_q4_dn_b_ctx *c = ctx_;
    const uint8_t *__restrict pr = c->q4_down + (size_t)j * c->row_stride;
    for (int t0 = 0; t0 < c->n_batch; t0 += NEON_TILE_1) {
        int nt = c->n_batch - t0;
        if (nt > NEON_TILE_1) nt = NEON_TILE_1;
        float32x4_t acc[NEON_TILE_1][2];
        for (int tt = 0; tt < nt; tt++) {
            acc[tt][0] = vdupq_n_f32(0.0f);
            acc[tt][1] = vdupq_n_f32(0.0f);
        }
        for (int b = 0; b < c->n_blocks; b++) {
            int8x16_t w0, w1;
            q4x16_to_i8x32(pr + (size_t)b * 18 + 2, &w0, &w1);
            float dw = q4_block_scale(pr + (size_t)b * 18);
            for (int tt = 0; tt < nt; tt++) {
                int t = t0 + tt;
                int8x16_t x0 = vld1q_s8(c->xq + (size_t)t * c->ffn_dim + (size_t)b * 32);
                int8x16_t x1 = vld1q_s8(c->xq + (size_t)t * c->ffn_dim + (size_t)b * 32 + 16);
                float dx = c->xd[(size_t)t * c->n_blocks + b];
                float32x4_t dsv = vdupq_n_f32(dw * dx);
                acc[tt][0] = vfmaq_f32(acc[tt][0], vcvtq_f32_s32(i8x16_dot_s32(w0, x0)), dsv);
                acc[tt][1] = vfmaq_f32(acc[tt][1], vcvtq_f32_s32(i8x16_dot_s32(w1, x1)), dsv);
            }
        }
        for (int tt = 0; tt < nt; tt++) {
            int t = t0 + tt;
            c->x_batch[(size_t)t * c->hidden_dim + j] =
                c->residual_batch[(size_t)t * c->hidden_dim + j] +
                hsum_neon4(vaddq_f32(acc[tt][0], acc[tt][1]));
        }
    }
}

static void dyn_matvec_q4_q8_fused_down_residual_batched_neon(
    float *__restrict x_batch, const float *__restrict residual_batch,
    const uint8_t *__restrict q4_down, const float *__restrict activated_batch,
    int hidden_dim, int ffn_dim, int n_batch) {
    int n_blocks = ffn_dim / 32;
    int row_stride = n_blocks * 18;
    int8_t *xq; float *xd;
    if (xq_scratch_ensure(&xq, &xd,
                          (size_t)n_batch * ffn_dim,
                          (size_t)n_batch * n_blocks * sizeof(float)) != 0)
        return;
    {
        vllm_q4_dn_quant_ctx vc = { activated_batch, xq, xd, ffn_dim, n_blocks, n_batch };
        vllm_tp_parfor(0, n_batch, vllm_q4_dn_quant_worker, &vc);
    }
#if ST_NEON_DOTPROD
    if (g_st_q4_repack && (hidden_dim & 3) == 0) {
        /* M4b: 真 GEMM batched down + residual（单矩阵 pass，3-token tile）。 */
        q4x4_gemm_batched(x_batch, q4_down, xq, xd, hidden_dim, ffn_dim, n_batch, residual_batch);
        return;
    }
#endif
    vllm_q4_dn_b_ctx vb = { x_batch, residual_batch, q4_down, xq, xd, n_blocks, row_stride, ffn_dim, hidden_dim, n_batch };
    vllm_tp_parfor(0, hidden_dim, vllm_q4_dn_b_worker, &vb);
}

/* ================================================================
 * Attention kernels (Q·K^T + fast-exp softmax + weighted-V)
 * NEON: 4-wide float32x4 dots, exp_neon4 mirrors exp_ps256.
 * ================================================================ */

typedef struct {
    float *attn_out; const float *q_buf; const float *k_pack; const float *v_pack;
    float *scores; float *imp_head;
    int nb, prev_len, seq_stride, nh, nkv, hd, score_stride, hdv;
    float scale;
} vllm_attn_batched_ctx;

static void vllm_attn_batched_worker(void *ctx_, int ha) {
    vllm_attn_batched_ctx *c = ctx_;
    int kh = (ha * c->nkv) / c->nh;
    const float *kp_head = c->k_pack + (size_t)kh * c->seq_stride * c->hd;
    const float *vp_head = c->v_pack + (size_t)kh * c->seq_stride * c->hd;
    /* M4j: 4-token 分块注意力（axiom: blas_sparse_message_passing_schedule —
     * shared input fusion）。4 个 query 共享一次 k 加载（QK 段）与 v 加载
     * （VKQ 段）。每个 token 的 QK 累加（i 升序 16 元素组）、softmax、VKQ
     * 累加（s 升序）顺序与串行版完全一致 -> 位级一致。
     * scores_buf 扩为 nh*4 行，本 head 用行 (ha*4+k)，stride 不变。 */
    float *scb = c->scores + (size_t)ha * 4 * c->score_stride;

    int t = 0;
    for (; t + 4 <= c->nb; t += 4) {
        const float *q[4]; float *o[4]; int n[4]; float maxs[4]; float *sc[4];
        for (int k = 0; k < 4; k++) {
            int tk = t + k;
            q[k] = c->q_buf + ((size_t)tk * c->nh + ha) * c->hd;
            o[k] = c->attn_out + ((size_t)tk * c->nh + ha) * c->hd;
            n[k] = c->prev_len + tk + 1;
            sc[k] = scb + (size_t)k * c->score_stride;
            maxs[k] = -1e9f;
        }

        /* Q·K^T：kp 每 16 元素加载一次，4 个 query 共享。n[k] 递增，
         * 位置 s 只对 n[k] > s 的 token 计算（kmax = 活跃 query 数）。 */
        int smax = n[3];
        for (int s = 0; s < smax; s++) {
            const float *kp = kp_head + (size_t)s * c->hd;
            float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = vdupq_n_f32(0.0f);
            float32x4_t acc2 = vdupq_n_f32(0.0f), acc3 = vdupq_n_f32(0.0f);
            int kmax = 4;
            while (kmax > 0 && s >= n[kmax - 1]) kmax--;
            for (int i = 0; i < c->hdv; i += 16) {
                float32x4_t k0 = vld1q_f32(kp + i);
                float32x4_t k1 = vld1q_f32(kp + i + 4);
                float32x4_t k2 = vld1q_f32(kp + i + 8);
                float32x4_t k3 = vld1q_f32(kp + i + 12);
                if (kmax >= 1) {
                    acc0 = vfmaq_f32(acc0, vld1q_f32(q[0] + i),     k0);
                    acc0 = vfmaq_f32(acc0, vld1q_f32(q[0] + i + 4), k1);
                    acc0 = vfmaq_f32(acc0, vld1q_f32(q[0] + i + 8), k2);
                    acc0 = vfmaq_f32(acc0, vld1q_f32(q[0] + i + 12), k3);
                }
                if (kmax >= 2) {
                    acc1 = vfmaq_f32(acc1, vld1q_f32(q[1] + i),     k0);
                    acc1 = vfmaq_f32(acc1, vld1q_f32(q[1] + i + 4), k1);
                    acc1 = vfmaq_f32(acc1, vld1q_f32(q[1] + i + 8), k2);
                    acc1 = vfmaq_f32(acc1, vld1q_f32(q[1] + i + 12), k3);
                }
                if (kmax >= 3) {
                    acc2 = vfmaq_f32(acc2, vld1q_f32(q[2] + i),     k0);
                    acc2 = vfmaq_f32(acc2, vld1q_f32(q[2] + i + 4), k1);
                    acc2 = vfmaq_f32(acc2, vld1q_f32(q[2] + i + 8), k2);
                    acc2 = vfmaq_f32(acc2, vld1q_f32(q[2] + i + 12), k3);
                }
                if (kmax >= 4) {
                    acc3 = vfmaq_f32(acc3, vld1q_f32(q[3] + i),     k0);
                    acc3 = vfmaq_f32(acc3, vld1q_f32(q[3] + i + 4), k1);
                    acc3 = vfmaq_f32(acc3, vld1q_f32(q[3] + i + 8), k2);
                    acc3 = vfmaq_f32(acc3, vld1q_f32(q[3] + i + 12), k3);
                }
            }
            float32x4_t acc[4] = { acc0, acc1, acc2, acc3 };
            for (int k = 0; k < kmax; k++) {
                float dot = hsum_neon4(acc[k]);
                for (int i = c->hdv; i < c->hd; i++) dot += q[k][i] * kp[i];
                sc[k][s] = dot * c->scale;
                if (sc[k][s] > maxs[k]) maxs[k] = sc[k][s];
            }
        }

        /* Softmax per token + VKQ（原串行结构，s 内层逐 token；
         * QK 段保留 4-token 共享 k 加载） */
        for (int k = 0; k < 4; k++) {
            int nk = n[k];
            float sum_exp = 0.0f;
            for (int s = 0; s + 4 <= nk; s += 4) {
                float32x4_t v = exp_neon4(vsubq_f32(vld1q_f32(sc[k] + s),
                                                    vdupq_n_f32(maxs[k])));
                vst1q_f32(sc[k] + s, v);
                sum_exp += hsum_neon4(v);
            }
            for (int s = nk & ~3; s < nk; s++) {
                sc[k][s] = expf(sc[k][s] - maxs[k]);
                sum_exp += sc[k][s];
            }
            float inv_sum = 1.0f / sum_exp;
            for (int i = 0; i < c->hdv; i += 4) vst1q_f32(o[k] + i, vdupq_n_f32(0.0f));
            for (int i = c->hdv; i < c->hd; i++) o[k][i] = 0.0f;
            for (int s = 0; s < nk; s++) {
                float wgt = sc[k][s] * inv_sum;
                if (c->imp_head) c->imp_head[(size_t)ha * c->score_stride + s] += wgt;
                const float *vp = vp_head + (size_t)s * c->hd;
                float32x4_t wv = vdupq_n_f32(wgt);
                for (int i = 0; i < c->hdv; i += 4)
                    vst1q_f32(o[k] + i, vfmaq_f32(vld1q_f32(o[k] + i), wv, vld1q_f32(vp + i)));
                for (int i = c->hdv; i < c->hd; i++) o[k][i] += wgt * vp[i];
            }
        }
    }

    /* --- 尾部（nb % 4）：原串行路径 --- */
    for (; t < c->nb; t++) {
        const float *q = c->q_buf + ((size_t)t * c->nh + ha) * c->hd;
        float *o = c->attn_out + ((size_t)t * c->nh + ha) * c->hd;
        /* tail 行必须与 M4j 4-token 分块一致（ha*4 行），否则 nb%4!=0 时
         * 不同 head 的 worker 会并发写同一 scores 行 → 数据竞争 → 偶发
         * attention 输出错误（多线程位级不确定的根因）。 */
        float *sc = c->scores + (size_t)ha * 4 * c->score_stride;
        int n = c->prev_len + t + 1;
        float max_score = -1e9f;

        for (int s = 0; s < n; s++) {
            const float *kp = kp_head + (size_t)s * c->hd;
            float32x4_t acc = vdupq_n_f32(0.0f);
            /* P3a (2026-08-29): a 2-accumulator split of this fp32 FMA chain
             * was A/B'd (VLLM_ATTN_2ACC): attn 6.08 vs 5.78-6.07 baseline
             * spread — no gain beyond run noise. Keep the single chain. */
            for (int i = 0; i < c->hdv; i += 4)
                acc = vfmaq_f32(acc, vld1q_f32(q + i), vld1q_f32(kp + i));
            float dot = hsum_neon4(acc);
            for (int i = c->hdv; i < c->hd; i++) dot += q[i] * kp[i];
            sc[s] = dot * c->scale;
            if (sc[s] > max_score) max_score = sc[s];
        }

        float sum_exp = 0.0f;
        for (int s = 0; s + 4 <= n; s += 4) {
            float32x4_t v = exp_neon4(vsubq_f32(vld1q_f32(sc + s),
                                                 vdupq_n_f32(max_score)));
            vst1q_f32(sc + s, v);
            sum_exp += hsum_neon4(v);
        }
        for (int s = n & ~3; s < n; s++) {
            sc[s] = expf(sc[s] - max_score);
            sum_exp += sc[s];
        }
        float inv_sum = 1.0f / sum_exp;

        for (int i = 0; i < c->hdv; i += 4) vst1q_f32(o + i, vdupq_n_f32(0.0f));
        for (int i = c->hdv; i < c->hd; i++) o[i] = 0.0f;

        for (int s = 0; s < n; s++) {
            float wgt = sc[s] * inv_sum;
            if (c->imp_head) c->imp_head[(size_t)ha * c->score_stride + s] += wgt;
            const float *vp = vp_head + (size_t)s * c->hd;
            float32x4_t wv = vdupq_n_f32(wgt);
            for (int i = 0; i < c->hdv; i += 4)
                vst1q_f32(o + i, vfmaq_f32(vld1q_f32(o + i), wv, vld1q_f32(vp + i)));
            for (int i = c->hdv; i < c->hd; i++) o[i] += wgt * vp[i];
        }
    }
}

static void st_attn_batched_packed_neon(
    float *restrict attn_out,        /* [nb * nh * hd] */
    const float *restrict q_buf,     /* [nb * nh * hd] */
    const float *restrict k_pack,    /* [nkv * seq_stride * hd] */
    const float *restrict v_pack,    /* [nkv * seq_stride * hd] */
    int nb, int prev_len, int seq_stride,
    int nh, int nkv, int hd, float scale,
    float *restrict scores,          /* [nh * score_stride] */
    int score_stride,
    float *restrict imp_head)        /* [nh * score_stride] or NULL */
{
    int hdv = hd & ~3;
    vllm_attn_batched_ctx vc = { attn_out, q_buf, k_pack, v_pack, scores, imp_head,
                                 nb, prev_len, seq_stride, nh, nkv, hd, score_stride, hdv, scale };
    vllm_tp_parfor(0, nh, vllm_attn_batched_worker, &vc);
}

/* Decode INT8-KV Flash Attention (single query, one head) — 长上下文 decode 加速。
 * Axiom: memory_bandwidth_reduction + online_softmax_single_pass：
 *   - 直接读 INT8 K/V 缓存（DRAM 流量为 float 的 1/4）；
 *   - Q·K^T 用 vdotq_s32（16 int8×int8 → int32x4）；
 *   - softmax 单遍 online（running max/sum + VKQ rescale），免去两遍扫描
 *     与 seq_len 大小的 scores 缓冲；
 *   - VKQ 累加器保持 F32（softmax 权重不量化）。
 * Q 激活按 Q8_0 每 32 元素 1 max-abs scale 量化（与 M4e decode 激活量化同一
 * 数值路线，PPL ratio≈0.994-0.998 已验证）。仅 NEON 平台使用；x86 走原
 * AVX2 路径（st_attn 内联循环）。输出已归一化。 */
static void flash_attn_single_q_q8_neon(
    float *restrict attn_out,           /* [hd] 归一化输出 */
    const float *restrict q,            /* [hd] query 激活 */
    int8_t *const *k_cache_q8,          /* 每层 INT8 K 块数组 */
    int8_t *const *v_cache_q8,
    const float *restrict kscale,       /* k_scale[l] + kh：per-token per-head */
    const float *restrict vscale,
    int seq_len, int kv_dim, int kv_bs, int nkv,
    int kh_off, int hd, float scale)
{
    const int nblk = hd / 32;           /* Q8_0 块数（Qwen hd=128 → 4） */
    const float KVQ_INV = 1.0f / 127.0f; /* = KVQ_SCALE（定义在下方，此处不可见） */
    float qsc[8];
    int8_t qi[256];

    /* 1. Q → INT8（每 32 元素 1 max-abs scale，Q8_0 同构） */
    for (int b = 0; b < nblk; b++) {
        float qm = 0.0f;
        const float *qb = q + b * 32;
        for (int i = 0; i < 32; i++) {
            float a = fabsf(qb[i]);
            if (a > qm) qm = a;
        }
        if (qm < 1e-6f) qm = 1.0f;
        qsc[b] = qm * KVQ_INV;          /* qm / 127 */
        float iq = 127.0f / qm;
        for (int i = 0; i < 32; i++) {
            float v = qb[i] * iq;
            int iv = (int)(v + (v >= 0.0f ? 0.5f : -0.5f));
            qi[b * 32 + i] = (int8_t)((iv > 127) ? 127 : ((iv < -128) ? -128 : iv));
        }
    }

    /* 2. VKQ 清零 + online softmax 运行态 */
    for (int i = 0; i < hd; i++) attn_out[i] = 0.0f;
    float M = -1e9f, S = 0.0f;

    /* 3. 单遍 KV 扫描（逐 token，online softmax 分块语义） */
    for (int t = 0; t < seq_len; t++) {
        const int8_t *kt = k_cache_q8[t / kv_bs] + (size_t)(t % kv_bs) * kv_dim + kh_off;
        const int8_t *vt = v_cache_q8[t / kv_bs] + (size_t)(t % kv_bs) * kv_dim + kh_off;
        float ks = kscale[(size_t)t * nkv] * KVQ_INV;
        float vs = vscale[(size_t)t * nkv] * KVQ_INV;

        /* Q·K^T：每 32 块一个 int32 累加，块 scale 分别乘后求和 */
        float s = 0.0f;
        for (int b = 0; b < nblk; b++) {
            int32x4_t acc = vdupq_n_s32(0);
            const int8_t *qa = qi + b * 32;
            const int8_t *kb = kt + b * 32;
            for (int i = 0; i < 32; i += 16)
                acc = vaddq_s32(acc, i8x16_dot_s32(
                        vld1q_s8(qa + i), vld1q_s8(kb + i)));
            int32_t dot32 = vgetq_lane_s32(acc, 0) + vgetq_lane_s32(acc, 1)
                          + vgetq_lane_s32(acc, 2) + vgetq_lane_s32(acc, 3);
            s += (float)dot32 * qsc[b];
        }
        s = s * ks * scale;

        /* online softmax：新 max → rescale 已累积 VKQ */
        float vsf;
        if (s > M) {
            float ms = expf(M - s);
            M = s;
            S *= ms;
            float32x4_t mv = vdupq_n_f32(ms);
            for (int i = 0; i < hd; i += 4)
                vst1q_f32(attn_out + i, vmulq_f32(vld1q_f32(attn_out + i), mv));
            vsf = 1.0f;
        } else {
            vsf = expf(s - M);
        }
        S += vsf;

        /* VKQ += INT8 V × (vsf × vs)，softmax 权重保持 F32 */
        float wv = vsf * vs;
        float32x4_t wv4 = vdupq_n_f32(wv);
        for (int i = 0; i < hd; i += 8) {
            int16x8_t v16 = vmovl_s8(vld1_s8(vt + i));
            float32x4_t va = vcvtq_f32_s32(vmovl_s16(vget_low_s16(v16)));
            float32x4_t vb = vcvtq_f32_s32(vmovl_s16(vget_high_s16(v16)));
            vst1q_f32(attn_out + i,     vfmaq_f32(vld1q_f32(attn_out + i),     va, wv4));
            vst1q_f32(attn_out + i + 4, vfmaq_f32(vld1q_f32(attn_out + i + 4), vb, wv4));
        }
    }

    /* 4. 归一化 */
    float inv = (S > 0.0f) ? 1.0f / S : 0.0f;
    float32x4_t iv = vdupq_n_f32(inv);
    for (int i = 0; i < hd; i += 4)
        vst1q_f32(attn_out + i, vmulq_f32(vld1q_f32(attn_out + i), iv));
}

typedef struct {
    float *attn_out; const float *q_buf; const float *k_pack; const float *v_pack;
    float *scores; float *imp_head;
    int nb, prev_len, seq_stride, nh, nkv, hd, score_stride, hdv, bs, k_blocks, n_probe;
    float scale;
} vllm_attn_sparse_ctx;

static void vllm_attn_sparse_worker(void *ctx_, int ha) {
    vllm_attn_sparse_ctx *c = ctx_;
    int kh = (ha * c->nkv) / c->nh;
    const float *kp_head = c->k_pack + (size_t)kh * c->seq_stride * c->hd;
    const float *vp_head = c->v_pack + (size_t)kh * c->seq_stride * c->hd;
    float *sc = c->scores + (size_t)ha * c->score_stride;

    int max_n = c->prev_len + c->nb;
    int n_blocks = (max_n + c->bs - 1) / c->bs;
    if (n_blocks <= 0) return;

    /* representative query: last token of the mini-batch */
    const float *q_rep = c->q_buf + ((size_t)(c->nb - 1) * c->nh + ha) * c->hd;

    float  *probe = (float *)malloc((size_t)n_blocks * sizeof(float));
    uint8_t *used = (uint8_t *)malloc((size_t)n_blocks);
    int    *sel = (int *)malloc((size_t)n_blocks * sizeof(int));
    if (!probe || !used || !sel) {
        free(probe); free(used); free(sel);
        return;   /* OOM: skip this head (outputs stay zeroed by caller) */
    }
    memset(used, 0, (size_t)n_blocks);

    /* probe each block: max dot over n_probe evenly-spaced K samples */
    int n_probe = c->n_probe;
    if (n_probe < 1) n_probe = 1;
    for (int b = 0; b < n_blocks; b++) {
        float best = -1e30f;
        int p0 = b * c->bs;
        int p_end = p0 + c->bs; if (p_end > max_n) p_end = max_n;
        int step = (p_end - p0) / n_probe; if (step < 1) step = 1;
        for (int ps = p0; ps < p_end; ps += step) {
            const float *kp = kp_head + (size_t)ps * c->hd;
            float32x4_t acc = vdupq_n_f32(0.0f);
            for (int i = 0; i < c->hdv; i += 4)
                acc = vfmaq_f32(acc, vld1q_f32(q_rep + i), vld1q_f32(kp + i));
            float dot = hsum_neon4(acc);
            for (int i = c->hdv; i < c->hd; i++) dot += q_rep[i] * kp[i];
            if (dot > best) best = dot;
        }
        probe[b] = best;
    }

    /* deterministic top-k (tie-break by block index) */
    int nsel = 0;
    for (int i = 0; i < n_blocks && nsel < c->k_blocks; i++) {
        int best = -1;
        for (int j = 0; j < n_blocks; j++) {
            if (used[j]) continue;
            if (best < 0 || probe[j] > probe[best] ||
                (probe[j] == probe[best] && j < best)) best = j;
        }
        if (best < 0) break;
        sel[nsel++] = best;
        used[best] = 1;
    }

    /* recency insurance: the most recent block is always kept */
    if (nsel < n_blocks && !used[n_blocks - 1]) {
        int low = 0;
        for (int i = 1; i < nsel; i++)
            if (probe[sel[i]] < probe[sel[low]]) low = i;
        used[sel[low]] = 0;
        sel[low] = n_blocks - 1;
        used[n_blocks - 1] = 1;
    }

    for (int t = 0; t < c->nb; t++) {
        const float *q = c->q_buf + ((size_t)t * c->nh + ha) * c->hd;
        float *o = c->attn_out + ((size_t)t * c->nh + ha) * c->hd;
        int n = c->prev_len + t + 1;
        float max_score = -1e9f;

        /* Q·K^T over selected blocks clipped to [0, n) */
        for (int si = 0; si < nsel; si++) {
            int p0 = sel[si] * c->bs;
            int p1 = p0 + c->bs; if (p1 > n) p1 = n;
            for (int s = p0; s < p1; s++) {
                const float *kp = kp_head + (size_t)s * c->hd;
                float32x4_t acc = vdupq_n_f32(0.0f);
                for (int i = 0; i < c->hdv; i += 4)
                    acc = vfmaq_f32(acc, vld1q_f32(q + i), vld1q_f32(kp + i));
                float dot = hsum_neon4(acc);
                for (int i = c->hdv; i < c->hd; i++) dot += q[i] * kp[i];
                sc[s] = dot * c->scale;
                if (sc[s] > max_score) max_score = sc[s];
            }
        }

        /* Softmax over the selected set */
        float sum_exp = 0.0f;
        for (int si = 0; si < nsel; si++) {
            int p0 = sel[si] * c->bs;
            int p1 = p0 + c->bs; if (p1 > n) p1 = n;
            for (int s = p0; s < p1; s++) {
                sc[s] = expf(sc[s] - max_score);
                sum_exp += sc[s];
            }
        }
        float inv_sum = 1.0f / sum_exp;

        for (int i = 0; i < c->hdv; i += 4)
            vst1q_f32(o + i, vdupq_n_f32(0.0f));
        for (int i = c->hdv; i < c->hd; i++) o[i] = 0.0f;

        /* Weighted V over selected set */
        for (int si = 0; si < nsel; si++) {
            int p0 = sel[si] * c->bs;
            int p1 = p0 + c->bs; if (p1 > n) p1 = n;
            for (int s = p0; s < p1; s++) {
                float wgt = sc[s] * inv_sum;
                if (c->imp_head) c->imp_head[(size_t)ha * c->score_stride + s] += wgt;
                const float *vp = vp_head + (size_t)s * c->hd;
                float32x4_t wv = vdupq_n_f32(wgt);
                for (int i = 0; i < c->hdv; i += 4)
                    vst1q_f32(o + i,
                              vfmaq_f32(vld1q_f32(o + i), wv, vld1q_f32(vp + i)));
                for (int i = c->hdv; i < c->hd; i++)
                    o[i] += wgt * vp[i];
            }
        }
    }
    free(probe); free(used); free(sel);
}

static void st_attn_batched_packed_sparse_neon(
    float *restrict attn_out,        /* [nb * nh * hd] */
    const float *restrict q_buf,     /* [nb * nh * hd] */
    const float *restrict k_pack,    /* [nkv * seq_stride * hd] */
    const float *restrict v_pack,    /* [nkv * seq_stride * hd] */
    int nb, int prev_len, int seq_stride,
    int nh, int nkv, int hd, float scale,
    float *restrict scores,          /* [nh * score_stride] */
    int score_stride, int bs, int k_blocks, int n_probe,
    float *restrict imp_head)        /* [nh * score_stride] or NULL */
{
    int hdv = hd & ~3;
    vllm_attn_sparse_ctx vc = { attn_out, q_buf, k_pack, v_pack, scores, imp_head,
                                nb, prev_len, seq_stride, nh, nkv, hd, score_stride, hdv, bs, k_blocks, n_probe, scale };
    vllm_tp_parfor(0, nh, vllm_attn_sparse_worker, &vc);
}

#endif /* ST_HAVE_NEON */

/* Decode f16 scale stored at the head of a Q8_0 block → f32 */
static inline float q8_block_scale(const uint8_t *b) {
    uint16_t h; memcpy(&h, b, 2);
    uint32_t fb = f16_to_f32_bits(h);
    float d; memcpy(&d, &fb, 4);
    return d;
}

/* Quantize a f32 activation row into int8 + per-block f32 scales (q8_0 layout,
 * symmetric scale = max_abs/127). Matches the weight quantizer semantics. */
static void quantize_row_q8_0_act(const float *__restrict x,
                                  int8_t *__restrict q, float *__restrict d, int cols) {
    int n_blocks = cols / 32;
    for (int b = 0; b < n_blocks; b++) {
        const float *xb = x + b * 32;
        float max_abs = 1e-10f;
        for (int i = 0; i < 32; i++) {
            float av = fabsf(xb[i]);
            if (av > max_abs) max_abs = av;
        }
        float scale = max_abs / 127.0f;
        d[b] = scale;
        int8_t *qb = q + b * 32;
        for (int i = 0; i < 32; i++) {
            int v = (int)(xb[i] / scale + 0.5f);
            if (v > 127) v = 127;
            if (v < -127) v = -127;
            qb[i] = (int8_t)v;
        }
    }
}

/* Block-major activation quantization: output xq[b][n_batch][32], xd[b][n_batch].
 * The inner GEMM then walks tokens with stride 32 (L1-friendly) instead of the
 * legacy [token][col] layout whose per-token stride is cols bytes (scattered
 * cache lines + TLB pressure). Parallelizes over blocks, which is much wider
 * than the legacy over-token loop (n_blocks = ffn_dim/32 = 384 vs n_batch=32). */
typedef struct {
    const float *x; int8_t *q; float *d;
    int cols, n_batch, n_blocks;
} vllm_quant_bm_ctx;

static void vllm_quant_bm_worker(void *ctx_, int b) {
    vllm_quant_bm_ctx *c = ctx_;
    for (int t = 0; t < c->n_batch; t++) {
        const float *xb = c->x + (size_t)t * c->cols + (size_t)b * 32;
        float max_abs = 1e-10f;
        for (int i = 0; i < 32; i++) {
            float av = fabsf(xb[i]);
            if (av > max_abs) max_abs = av;
        }
        float scale = max_abs / 127.0f;
        c->d[(size_t)b * c->n_batch + t] = scale;
        int8_t *qb = c->q + (size_t)b * c->n_batch * 32 + (size_t)t * 32;
        for (int i = 0; i < 32; i++) {
            int v = (int)(xb[i] / scale + 0.5f);
            if (v > 127) v = 127;
            if (v < -127) v = -127;
            qb[i] = (int8_t)v;
        }
    }
}

static void quantize_row_q8_0_act_bm(const float *__restrict x,
                                     int8_t *__restrict q, float *__restrict d,
                                     int cols, int n_batch) {
    int n_blocks = cols / 32;
    vllm_quant_bm_ctx vc = { x, q, d, cols, n_batch, n_blocks };
    vllm_tp_parfor(0, n_blocks, vllm_quant_bm_worker, &vc);
}

typedef struct {
    const float *x; int8_t *q; float *d;
    int cols, n_batch, n_blocks;
} vllm_quant_bm_b_ctx;

static void vllm_quant_bm_b_worker(void *ctx_, int b) {
    vllm_quant_bm_b_ctx *c = ctx_;
    for (int t = 0; t < c->n_batch; t++) {
        const float *xb = c->x + (size_t)t * c->cols + (size_t)b * 32;
        float max_abs = 1e-10f;
        for (int i = 0; i < 32; i++) {
            float av = fabsf(xb[i]);
            if (av > max_abs) max_abs = av;
        }
        float scale = max_abs / 127.0f;
        c->d[(size_t)b * c->n_batch + t] = scale;
        int8_t *qb = c->q + (size_t)b * c->n_batch * 32 + (size_t)t * 32;
        for (int i = 0; i < 32; i++) {
            int v = (int)(xb[i] / scale + 0.5f);
            if (v > 127) v = 127;
            if (v < -127) v = -127;
            qb[i] = (int8_t)(v ^ 0x80);
        }
    }
}

/* Block-major activation quantization that stores the +128-biased bytes
 * (v ^ 0x80) instead of the signed value, so the AVX-512 VNNI GATE/UP kernel
 * can feed VPDPBUSD directly without a per-token XOR. The per-block f32 scale
 * d[] is unchanged (identical dequant semantics); only the byte payload is
 * shifted into the unsigned domain for the unsigned×signed dot. */
static void quantize_row_q8_0_act_bm_biased(const float *__restrict x,
                                            int8_t *__restrict q, float *__restrict d,
                                            int cols, int n_batch) {
    int n_blocks = cols / 32;
    vllm_quant_bm_b_ctx vc = { x, q, d, cols, n_batch, n_blocks };
    vllm_tp_parfor(0, n_blocks, vllm_quant_bm_b_worker, &vc);
}

/* Batch-tile size: number of tokens processed per weight-block load. Larger
 * tiles reuse each loaded Q8_0 weight block across more tokens (fewer L1/L2
 * reloads and corr/scale recomputations). AVX-512 exposes 32 ZMM registers, so
 * these sizes fit comfortably; the AVX2 fallback (16 YMM) spills but is not the
 * active path in the /arch:AVX512 build. */
#define GEMM_TILE_QKV 5   /* 3 outputs → 3×5=15 accumulators */
#define GEMM_TILE_GU  12  /* 2 outputs → 2×12=24 accumulators */
#define GEMM_TILE_1   16  /* 1 output  → 16 accumulators */
#define GEMM_TILE_GU_Q8 8 /* 2 outputs (gate + up) × 8 tokens → 16 accumulators */

/* ================================================================
 * AVX2 Q8_0 Dequantizing Matvec Functions
 * (axiom: MI_fusion_rule + blas_precision_efficiency_tradeoff)
 * ================================================================ */


static void dyn_matvec_q8(float *__restrict out, const uint8_t *__restrict q8_w,
                            const float *__restrict x, int rows, int cols) {
    dyn_matvec_q8_neon(out, q8_w, x, rows, cols);
}

/* Batched Q8_0 matvec with row-tiled shared-input fusion
 * (axiom: block_matrix_assoc_natural + partition_alignment):
 *   out[p*out_stride + r] = dot(q8_w[r], x[p]) + bias[r]
 * x is PATCH-MAJOR [batch][cols]. Patch-parallel; inside each patch, 4 weight
 * rows are accumulated simultaneously (4 independent FMA chains) sharing one
 * x vector — hides FMA latency and cuts the activation stream 4x, while the
 * weight rows stream from L3 (resident across the patch tiles).
 * cols MUST be a multiple of 32; out_stride lets the caller pad rows. */
/* ================================================================
 * Batched Q8_0 matvec（NEON，RK3588 单平台）
 * ================================================================ */
typedef struct {
    float *out; const uint8_t *q8_w; const float *x; const float *bias;
    int batch, rows, cols, out_stride, n_blocks, row_stride;
} vllm_q8_matvec_b_neon_ctx;

static void vllm_q8_matvec_b_neon_worker(void *ctx_, int p) {
    vllm_q8_matvec_b_neon_ctx *c = ctx_;
    const float *xp = c->x + (size_t)p * c->cols;
    for (int r = 0; r < c->rows; r++) {
        const uint8_t *pr = c->q8_w + (size_t)r * c->row_stride;
        float32x4_t acc[8];
        for (int i = 0; i < 8; i++) acc[i] = vdupq_n_f32(0.0f);
        for (int b = 0; b < c->n_blocks; b++) {
            float d = q8_block_scale(pr);
            float32x4_t f[8];
            i8x32_to_f32x8((const int8_t *)(pr + 2), f);
            float32x4_t dq = vdupq_n_f32(d);
            const float *xs = xp + (size_t)b * 32;
            for (int i = 0; i < 8; i++)
                acc[i] = vfmaq_f32(acc[i], vmulq_f32(f[i], dq), vld1q_f32(xs + (size_t)i * 4));
            pr += 34;
        }
        c->out[(size_t)p * c->out_stride + r] = f32x8_hsum(acc) + (c->bias ? c->bias[r] : 0.0f);
    }
}

void st_q8_matvec_batched(float *__restrict out, const uint8_t *__restrict q8_w,
                          const float *__restrict x, const float *__restrict bias,
                          int batch, int rows, int cols, int out_stride) {
    int n_blocks = cols / 32;
    int row_stride = n_blocks * 34;
    /* NEON：patch 并行，每行一次扫描（patch-major 布局天然适配）。 */
    vllm_q8_matvec_b_neon_ctx vc = { out, q8_w, x, bias, batch, rows, cols, out_stride, n_blocks, row_stride };
    vllm_tp_parfor(0, batch, vllm_q8_matvec_b_neon_worker, &vc);
}

/* Single-vector Q8_0 matvec (no internal OMP; caller owns parallelism).
 * NEON (RK3588): 4-wide float32x4 dots, 8 accumulator lanes per row. */
void st_q8_matvec_single(float *__restrict out, const uint8_t *__restrict q8_w,
                         const float *__restrict x, const float *__restrict bias,
                         int rows, int cols) {
    int n_blocks = cols / 32;
    int row_stride = n_blocks * 34;
    for (int r = 0; r < rows; r++) {
        const uint8_t *pr = q8_w + (size_t)r * row_stride;
        float32x4_t acc[8];
        for (int i = 0; i < 8; i++) acc[i] = vdupq_n_f32(0.0f);
        for (int b = 0; b < n_blocks; b++) {
            float d = q8_block_scale(pr);
            float32x4_t f[8];
            i8x32_to_f32x8((const int8_t *)(pr + 2), f);
            float32x4_t dq = vdupq_n_f32(d);
            const float *xs = x + (size_t)b * 32;
            for (int i = 0; i < 8; i++)
                acc[i] = vfmaq_f32(acc[i], vmulq_f32(f[i], dq), vld1q_f32(xs + (size_t)i * 4));
            pr += 34;
        }
        out[r] = f32x8_hsum(acc) + (bias ? bias[r] : 0.0f);
    }
}

/* ---- Batch Q8_0 matvec (visual ViT encode hot path) ----
 * out[B][rows] = W[rows][cols] · x[B][cols] + bias.
 * Row-parallel: each weight row is streamed by exactly one thread and reused
 * across all B patches, so the matrix is read once per patch-block instead of
 * once per patch (per-patch single-row matvec re-read the whole matrix for
 * every patch — 1024 patches × 15MB/layer ≈ 370 GB of traffic on RK3588).
 * Per (row, patch) the block-accumulation order matches st_q8_matvec_single
 * (8-wide NEON FMA chains, block order), so results are bit-identical. */
#define Q8B_PATCH_BLK 16
#define Q8B_ROW_BLK   512

#if ST_NEON_DOTPROD
/* ---- Q8_0 x Q8_0 SDOT batch matvec (visual ViT, RK3588) ----
 * 2c: quantize the activation to Q8_0 blocks (one f16 scale + 32 int8 per
 * block, mirroring the weight layout) so the hot inner loop is int8 dot
 * products (vdotq_s32, 32 MAC/instr on A76) instead of fp32 dequant+FMA.
 * Reference: llama.cpp ggml_vec_dot_q8_0_q8_0 (arm/quants.c). ActQ8 changes
 * the numeric path (per-block symmetric activation quantization), the same
 * tradeoff the LLM GEMM path already accepts. */

/* Activation scratch: [n_blocks][nb_patches][34B] block-major so the SDOT
 * inner loop walks both weight and activation blocks sequentially. Grows on
 * demand; visual matvecs run serially, so a single scratch is safe. */
static uint8_t *g_q8a_scratch = NULL;
static size_t   g_q8a_cap = 0;

/* Quantize x[nb_p][cols] (patch-major fp32) -> xq8[b][p][34] blocks. */
typedef struct {
    const float *x; uint8_t *xq8;
    int n_patches, cols, n_blocks;
} vllm_q8blk_ctx;

static void vllm_q8blk_worker(void *ctx_, int b) {
    vllm_q8blk_ctx *c = ctx_;
    uint8_t *xb8 = c->xq8 + (size_t)b * c->n_patches * 34;
    const float *xc = c->x + (size_t)b * 32;
    for (int p = 0; p < c->n_patches; p++) {
        const float *xr = xc + (size_t)p * c->cols;
        float max_abs = 1e-10f;
        for (int i = 0; i < 32; i++) {
            float av = fabsf(xr[i]);
            if (av > max_abs) max_abs = av;
        }
        float scale = max_abs / 127.0f;
        uint16_t dh = f32_to_f16_bits(scale);
        memcpy(xb8, &dh, 2);
        for (int i = 0; i < 32; i++) {
            int v = (int)(xr[i] / scale + 0.5f);
            if (v > 127) v = 127;
            if (v < -127) v = -127;
            xb8[2 + i] = (uint8_t)(int8_t)v;
        }
        xb8 += 34;
    }
}

static int st_q8_quant_act_blk(uint8_t *xq8, const float *x, int n_patches, int cols) {
    int n_blocks = cols / 32;
    vllm_q8blk_ctx qc = { x, xq8, n_patches, cols, n_blocks };
    vllm_tp_parfor(0, n_blocks, vllm_q8blk_worker, &qc);
    return 0;
}
#endif /* ST_NEON_DOTPROD */

typedef struct {
    float *out; const uint8_t *q8_w; const float *bias; const float *x;
    const uint8_t *xq8;   /* non-NULL => Q8xQ8 SDOT path (activation pre-quantized) */
    int B, rows, cols;
    /* head-contiguous output mode (QKV -> [3][nh][B][hd] layout): when
     * nh > 0, out is addressed as ((seg*nh + h)*B + p)*hd + d for the
     * global row (row_base + r); row_base is the r0 row-block offset.
     * B_total is the FULL patch count and p_off the current patch-block
     * start, so head-contiguous rows stay [3][nh][B_total][hd] across
     * the Q8B_PATCH_BLK chunked writes (a per-chunk [3][nh][nb][hd]
     * would misalign every k/v slab for rope/attention). */
    int vh, nh, hd, row_base;
    int B_total, p_off;
} vllm_q8b_ctx;

static void vllm_q8b_worker(void *ctx_, int r) {
    vllm_q8b_ctx *c = ctx_;
    int n_blocks = c->cols / 32;
    int row_stride = n_blocks * 34;
    const uint8_t *pr = c->q8_w + (size_t)r * row_stride;
    float32x4_t acc[Q8B_PATCH_BLK][8];
    for (int p = 0; p < c->B; p++)
        for (int i = 0; i < 8; i++) acc[p][i] = vdupq_n_f32(0.0f);
    for (int b = 0; b < n_blocks; b++) {
        float d = q8_block_scale(pr);
        float32x4_t f[8];
        i8x32_to_f32x8((const int8_t *)(pr + 2), f);
        float32x4_t dq = vdupq_n_f32(d);
        for (int p = 0; p < c->B; p++) {
            const float *xs = c->x + (size_t)p * c->cols + (size_t)b * 32;
            for (int i = 0; i < 8; i++)
                acc[p][i] = vfmaq_f32(acc[p][i], vmulq_f32(f[i], dq),
                                      vld1q_f32(xs + (size_t)i * 4));
        }
        pr += 34;
    }
    if (c->nh > 0) {
        /* head-contiguous: [3][nh][B][hd] — attn reads K/V rows contiguously */
        int rr = c->row_base + r;
        int seg = rr / c->vh;
        int h = (rr % c->vh) / c->hd;
        int d = rr % c->hd;
        float *dst = c->out + ((size_t)(seg * c->nh + h) * c->B_total + c->p_off) * c->hd + d;
        for (int p = 0; p < c->B; p++)
            dst[(size_t)p * c->hd] =
                f32x8_hsum(acc[p]) + (c->bias ? c->bias[r] : 0.0f);
    } else {
        for (int p = 0; p < c->B; p++)
            c->out[(size_t)p * c->rows + r] =
                f32x8_hsum(acc[p]) + (c->bias ? c->bias[r] : 0.0f);
    }
}

#if ST_NEON_DOTPROD
/* Q8_0 weight x Q8_0 activation dot (SDOT): out[(p)*rows + r] =
 * sum_b dw_b * dx_b * dot(wqs_b, xqs_b) + bias[r]. Activation layout
 * xq8[b][p][34], so the inner patch loop walks 34B blocks sequentially. */
static void vllm_q8b_sdot_worker(void *ctx_, int r) {
    vllm_q8b_ctx *c = ctx_;
    int n_blocks = c->cols / 32;
    int row_stride = n_blocks * 34;
    const uint8_t *pr = c->q8_w + (size_t)r * row_stride;
    float32x4_t acc[Q8B_PATCH_BLK];
    for (int p = 0; p < c->B; p++) acc[p] = vdupq_n_f32(0.0f);
    for (int b = 0; b < n_blocks; b++) {
        const float dw = q8_block_scale(pr);
        const int8x16_t wl = vld1q_s8((const int8_t *)(pr + 2));
        const int8x16_t wh = vld1q_s8((const int8_t *)(pr + 18));
        const uint8_t *xblk = c->xq8 + (size_t)b * c->B * 34;
        for (int p = 0; p < c->B; p++) {
            const float dx = q8_block_scale(xblk);
            int32x4_t d = vaddq_s32(i8x16_dot_s32(wl, vld1q_s8((const int8_t *)(xblk + 2))),
                                    i8x16_dot_s32(wh, vld1q_s8((const int8_t *)(xblk + 18))));
            acc[p] = vmlaq_n_f32(acc[p], vcvtq_f32_s32(d), dw * dx);
            xblk += 34;
        }
        pr += 34;
    }
    if (c->nh > 0) {
        int rr = c->row_base + r;
        int seg = rr / c->vh;
        int h = (rr % c->vh) / c->hd;
        int d = rr % c->hd;
        float *dst = c->out + ((size_t)(seg * c->nh + h) * c->B_total + c->p_off) * c->hd + d;
        for (int p = 0; p < c->B; p++)
            dst[(size_t)p * c->hd] =
                hsum_neon4(acc[p]) + (c->bias ? c->bias[r] : 0.0f);
    } else {
        for (int p = 0; p < c->B; p++)
            c->out[(size_t)p * c->rows + r] =
                hsum_neon4(acc[p]) + (c->bias ? c->bias[r] : 0.0f);
    }
}

/* Quantize the patch block's activation into g_q8a_scratch, then run the
 * row-blocked SDOT matvec. Shared by the plain and head-contiguous entries. */
static void st_q8_sdot_patch_block(float *out, const uint8_t *q8_w,
                                   const float *bias, const float *x,
                                   int nb, int rows, int cols,
                                   int vh, int nh, int hd,
                                   int B_total, int p_off) {
    int n_blocks = cols / 32;
    size_t need = (size_t)n_blocks * nb * 34;
    if (g_q8a_cap < need) {
        uint8_t *np_ = (uint8_t *)realloc(g_q8a_scratch, need);
        if (!np_) return;
        g_q8a_scratch = np_;
        g_q8a_cap = need;
    }
    st_q8_quant_act_blk(g_q8a_scratch, x, nb, cols);
    int row_stride = n_blocks * 34;
    for (int r0 = 0; r0 < rows; r0 += Q8B_ROW_BLK) {
        int nr = rows - r0;
        if (nr > Q8B_ROW_BLK) nr = Q8B_ROW_BLK;
        vllm_q8b_ctx c = {
            nh > 0 ? out : out + (size_t)r0,
            q8_w + (size_t)r0 * row_stride,
            bias ? bias + r0 : NULL, NULL, g_q8a_scratch, nb, rows, cols
        };
        c.vh = vh; c.nh = nh; c.hd = hd; c.row_base = r0;
        c.B_total = B_total; c.p_off = p_off;
        vllm_tp_parfor(0, nr, vllm_q8b_sdot_worker, &c);
    }
}
#endif /* ST_NEON_DOTPROD */

void st_q8_matvec_batch(float *__restrict out, const uint8_t *__restrict q8_w,
                        const float *__restrict bias,
                        const float *__restrict x,
                        int B, int rows, int cols) {
    int row_stride = (cols / 32) * 34;
#if ST_NEON_DOTPROD
    /* A/B switch: VLLM_VIS_ACT_F32=1 keeps the fp32-activation kernel
     * (bit-identical to pre-2c); default is the Q8xQ8 SDOT path. */
    static int act_f32 = -1;
    if (act_f32 < 0) act_f32 = getenv("VLLM_VIS_ACT_F32") != NULL;
#endif
    while (B > 0) {
        int nb = B > Q8B_PATCH_BLK ? Q8B_PATCH_BLK : B;
        /* Row blocks of Q8B_ROW_BLK: the patch block's activations
         * (nb*cols floats, ~64 KB at cols=1024) stay resident in each core's
         * L2 across the whole row block, so x is fetched from memory once per
         * row block instead of once per row (a per-row re-read blew the
         * traffic back up to rows*nb*cols). A large row block also cuts the
         * parfor dispatch count ~8x vs the previous 64-row blocks. Weight rows
         * stream from L3 once each per patch block. */
#if ST_NEON_DOTPROD
        if (!act_f32) {
            st_q8_sdot_patch_block(out, q8_w, bias, x, nb, rows, cols, 0, 0, 0, 0, 0);
        } else
#endif
        {
            for (int r0 = 0; r0 < rows; r0 += Q8B_ROW_BLK) {
                int nr = rows - r0;
                if (nr > Q8B_ROW_BLK) nr = Q8B_ROW_BLK;
                vllm_q8b_ctx c = {
                    out + (size_t)r0, q8_w + (size_t)r0 * row_stride,
                    bias ? bias + r0 : NULL, x, NULL, nb, rows, cols
                };
                vllm_tp_parfor(0, nr, vllm_q8b_worker, &c);
            }
        }
        out += (size_t)nb * rows;
        x += (size_t)nb * cols;
        B -= nb;
    }
}

/* ViT QKV projection with head-contiguous output: out[3][nh][B][hd] (seg, head,
 * patch, dim) instead of [B][3*vh]. attn then streams each head's K/V rows
 * sequentially (no 12 KB strided scatter inside qkv), which is what the batched
 * block attention kernel wants. rows = 3*vh. */
void st_q8_matvec_batch_qkv(float *__restrict out, const uint8_t *__restrict q8_w,
                            const float *__restrict bias,
                            const float *__restrict x,
                            int B, int rows, int vh, int nh, int hd) {
    int cols = rows / 3;
    int row_stride = (cols / 32) * 34;
    int B_total = B;
    int p_off = 0;
#if ST_NEON_DOTPROD
    static int act_f32 = -1;
    if (act_f32 < 0) act_f32 = getenv("VLLM_VIS_ACT_F32") != NULL;
#endif
    while (B > 0) {
        int nb = B > Q8B_PATCH_BLK ? Q8B_PATCH_BLK : B;
#if ST_NEON_DOTPROD
        if (!act_f32) {
            st_q8_sdot_patch_block(out, q8_w, bias, x, nb, rows, cols,
                                   vh, nh, hd, B_total, p_off);
        } else
#endif
        {
            for (int r0 = 0; r0 < rows; r0 += Q8B_ROW_BLK) {
                int nr = rows - r0;
                if (nr > Q8B_ROW_BLK) nr = Q8B_ROW_BLK;
                vllm_q8b_ctx c = {
                    out, q8_w + (size_t)r0 * row_stride,
                    bias ? bias + r0 : NULL, x, NULL, nb, rows, cols
                };
                c.vh = vh; c.nh = nh; c.hd = hd; c.row_base = r0;
                c.B_total = B_total; c.p_off = p_off;
                vllm_tp_parfor(0, nr, vllm_q8b_worker, &c);
            }
        }
        x += (size_t)nb * cols;
        p_off += nb;
        B -= nb;
    }
}


static void dyn_matvec_q8_fused_gate_up(
    float *__restrict gate_out, float *__restrict up_out,
    const uint8_t *__restrict q8_gate, const uint8_t *__restrict q8_up,
    const float *__restrict x, int rows, int cols)
{
    dyn_matvec_q8_fused_gate_up_neon(gate_out, up_out, q8_gate, q8_up, x, rows, cols);
}


/* Fused Q/K/V Q8_0 matvec: single pass through input x produces Q,K,V together
 * (axiom: block_matrix_assoc_natural — shared input, triple FMAs).
 * For rows r < kv_rows: computes q_out[r], k_out[r], v_out[r].
 * For rows r >= kv_rows: computes only q_out[r] (K/V don't have this row). */
static void dyn_matvec_q8_fused_qkv(
    float *__restrict q_out, float *__restrict k_out, float *__restrict v_out,
    const uint8_t *__restrict q8_q, const uint8_t *__restrict q8_k,
    const uint8_t *__restrict q8_v, const float *__restrict x,
    int q_rows, int kv_rows, int cols)
{
    dyn_matvec_q8_fused_qkv_neon(q_out, k_out, v_out, q8_q, q8_k, q8_v, x, q_rows, kv_rows, cols);
}

/* ================================================================
 * Batched Q8_0 Matvec Kernels (Mini-Batch Prefill Optimization)
 * (axiom: blas_qkv_fusion_categorical + block_matrix_assoc_natural)
 *
 * Processes up to PREFILL_BATCH_SIZE tokens simultaneously through
 * the same Q8_0 weight matrix.  Dequantization cost is amortized:
 * each block's f16 scale and int8 values are loaded once and
 * reused across all tokens in the mini-batch.
 *
 * Memory bandwidth reduction: ~B× (weight read once, not B times).
 *
 * Performance
 * ================================================================ */

/* KV cache INT8 quantization: per-token per-head max-abs scale.
 * Unlike the old fixed 1/127 scale (which clipped V heads without Q/K norm),
 * each head's K/V is quantized against its own max |value| so it is near-lossless.
 * Dequantization multiplies INT8 by (scale / 127). */
#define KVQ_INV_SCALE 127.0f
#define KVQ_SCALE     (1.0f / 127.0f)

/* Quantize one token's K/V (kv_dim = nkv * hd, row-major per head) into INT8 with a
 * per-head max-abs scale. kscale/vscale receive nkv scale factors (max |value|). */
static void kv_quantize_per_head(int8_t *kdst, int8_t *vdst,
                                 float *kscale, float *vscale,
                                 const float *kf, const float *vf,
                                 int nkv, int hd) {
    for (int kh = 0; kh < nkv; kh++) {
        const float *kh_src = kf + (size_t)kh * hd;
        const float *vh_src = vf + (size_t)kh * hd;
        int8_t *kh_dst = kdst + (size_t)kh * hd;
        int8_t *vh_dst = vdst + (size_t)kh * hd;
        float maxk = 0.0f, maxv = 0.0f;
        for (int i = 0; i < hd; i++) {
            float ak = fabsf(kh_src[i]);
            float av = fabsf(vh_src[i]);
            if (ak > maxk) maxk = ak;
            if (av > maxv) maxv = av;
        }
        if (maxk < 1e-6f) maxk = 1.0f;
        if (maxv < 1e-6f) maxv = 1.0f;
        kscale[kh] = maxk;
        vscale[kh] = maxv;
        float ik = KVQ_INV_SCALE / maxk;
        float iv = KVQ_INV_SCALE / maxv;
        for (int i = 0; i < hd; i++) {
            int qk = (int)floorf(kh_src[i] * ik + 0.5f);
            int qv = (int)floorf(vh_src[i] * iv + 0.5f);
            kh_dst[i] = (int8_t)((qk > 127) ? 127 : ((qk < -128) ? -128 : qk));
            vh_dst[i] = (int8_t)((qv > 127) ? 127 : ((qv < -128) ? -128 : qv));
        }
    }
}

/* Flash Attention blocked kernel: tile KV cache into L1-friendly contiguous buffers.
 * Original KV layout: k_cache[seq_len][kv_dim] — stride=kv_dim=1024 between rows.
 * Each K[row] is 128 floats (512 bytes), spread over 8 cache lines at 1024-float gaps.
 * Packing FA_KV_TILE rows into K_packed[tile][hd] makes dot products cache-friendly.
 * Axiom: memory_bandwidth_reduction — eliminates stride-kv_dim cache misses. */
#define FA_KV_TILE   64
#define FA_THRESHOLD 64    /* minimum seq_len to use blocked FA; below this, scalar path */

static void *xq_alloc_canary(size_t bytes);
static void xq_free_canary(void *p, size_t bytes, const char *tag);
static size_t g_wbuf_pad;
static void dyn_matvec_q8_fused_qkv_batched(
    float *__restrict q_out, float *__restrict k_out, float *__restrict v_out,
    const uint8_t *__restrict q8_q, const uint8_t *__restrict q8_k,
    const uint8_t *__restrict q8_v, const float *__restrict x_batch,
    int q_rows, int kv_rows, int cols, int n_batch)
{
    /* Transparent NPU offload (prefill QKV). Partial success falls through to
     * the CPU kernel, which recomputes every output - correctness is safe. */
    if (st_npu_enabled() &&
        st_npu_try_gemm_batched(q_out, x_batch, q8_q, q_rows, cols, n_batch,
                                g_npu_layer, VLLM_NPU_PROJ_Q) &&
        st_npu_try_gemm_batched(k_out, x_batch, q8_k, kv_rows, cols, n_batch,
                                g_npu_layer, VLLM_NPU_PROJ_K) &&
        st_npu_try_gemm_batched(v_out, x_batch, q8_v, kv_rows, cols, n_batch,
                                g_npu_layer, VLLM_NPU_PROJ_V))
        return;
    dyn_matvec_q8_fused_qkv_batched_neon(q_out, k_out, v_out, q8_q, q8_k, q8_v, x_batch, q_rows, kv_rows, cols, n_batch);
}


static void dyn_matvec_q8_fused_o_residual_batched(
    float *__restrict x_batch, const float *__restrict residual_batch,
    const uint8_t *__restrict q8_o, const float *__restrict attn_batch,
    int rows, int cols, int n_batch)
{
    /* Transparent NPU offload: x_batch = residual + O(attn). */
    if (st_npu_enabled() &&
        st_npu_try_gemm_batched(x_batch, attn_batch, q8_o, rows, cols, n_batch,
                                g_npu_layer, VLLM_NPU_PROJ_O)) {
        for (int t = 0; t < n_batch; t++)
            for (int r = 0; r < rows; r++)
                x_batch[(size_t)t * rows + r] += residual_batch[(size_t)t * rows + r];
        return;
    }
    dyn_matvec_q8_fused_o_residual_batched_neon(x_batch, residual_batch, q8_o, attn_batch, rows, cols, n_batch);
}


static void dyn_matvec_q8_fused_gate_up_batched(
    float *__restrict gate_out, float *__restrict up_out,
    const uint8_t *__restrict q8_gate, const uint8_t *__restrict q8_up,
    const float *__restrict x_batch, int rows, int cols, int n_batch)
{
    /* Transparent NPU offload (prefill Gate+Up). */
    if (st_npu_enabled() &&
        st_npu_try_gemm_batched(gate_out, x_batch, q8_gate, rows, cols, n_batch,
                                g_npu_layer, VLLM_NPU_PROJ_GATE) &&
        st_npu_try_gemm_batched(up_out, x_batch, q8_up, rows, cols, n_batch,
                                g_npu_layer, VLLM_NPU_PROJ_UP))
        return;
    dyn_matvec_q8_fused_gate_up_batched_neon(gate_out, up_out, q8_gate, q8_up, x_batch, rows, cols, n_batch);
}


static void dyn_matvec_q8_fused_down_residual_batched(
    float *__restrict x_batch, const float *__restrict residual_batch,
    const uint8_t *__restrict q8_down, const float *__restrict activated_batch,
    int hidden_dim, int ffn_dim, int n_batch)
{
    /* Transparent NPU offload: x_batch = residual + down(activated). */
    if (st_npu_enabled() &&
        st_npu_try_gemm_batched(x_batch, activated_batch, q8_down, hidden_dim,
                                ffn_dim, n_batch, g_npu_layer,
                                VLLM_NPU_PROJ_DOWN)) {
        for (int t = 0; t < n_batch; t++)
            for (int j = 0; j < hidden_dim; j++)
                x_batch[(size_t)t * hidden_dim + j] +=
                    residual_batch[(size_t)t * hidden_dim + j];
        return;
    }
    dyn_matvec_q8_fused_down_residual_batched_neon(x_batch, residual_batch, q8_down, activated_batch, hidden_dim, ffn_dim, n_batch);
}


static void dyn_matvec_q8_fused_o_residual(
    float *__restrict x, const float *__restrict residual,
    const uint8_t *__restrict q8_o, const float *__restrict attn_in,
    int rows, int cols)
{
    dyn_matvec_q8_fused_o_residual_neon(x, residual, q8_o, attn_in, rows, cols);
}


/* Fused down projection + FFN residual: x[i] = residual[i] + down(act)[i]
 * (axiom: block_matrix_assoc_natural — skips ffn_out_buf)
 * Writes directly to x, saving one 16 KB write+read per layer. */
static void dyn_matvec_q8_fused_down_residual(
    float *__restrict x, const float *__restrict residual,
    const uint8_t *__restrict q8_down, const float *__restrict activated,
    int hidden_dim, int ffn_dim)
{
    dyn_matvec_q8_fused_down_residual_neon(x, residual, q8_down, activated, hidden_dim, ffn_dim);
}

/* ---- end Q8_0 functions ---- */

/* ================================================================
 * Q4_0 × Q8_0 Integer-Domain Decode Matvec Kernels
 * (axiom: blas_precision_efficiency_tradeoff + fixedpoint_quantize_saturate)
 *
 * Decode is DRAM-bandwidth bound: Q8_0 weights stream ~8 GB per token.
 * Q4_0 weights (18B/block vs 34B/block) nearly halve that stream while the
 * Q8_0 activation is quantized once per matvec (O(cols), negligible).
 *
 * Each block: unpack 16 nibbles → 32 signed int8 in [-8,+7], integer-dot with
 * the Q8_0 activation (32 int8), then scale by (w_scale × a_scale). Mirrors
 * llama.cpp ggml_vec_dot_q4_0_q8_0.
 * ================================================================ */


/* Decode f16 scale stored at the head of a Q4_0 block → f32. */
static inline float q4_block_scale(const uint8_t *b) {
    uint16_t h; memcpy(&h, b, 2);
    uint32_t fb = f16_to_f32_bits(h);
    float d; memcpy(&d, &fb, 4);
    return d;
}


static void dyn_matvec_q4_q8(float *__restrict out, const uint8_t *__restrict q4_w,
                             const float *__restrict x, int rows, int cols) {
    dyn_matvec_q4_q8_neon(out, q4_w, x, rows, cols);
}


static void dyn_matvec_q4_q8_fused_gate_up(
    float *__restrict gate_out, float *__restrict up_out,
    const uint8_t *__restrict q4_gate, const uint8_t *__restrict q4_up,
    const float *__restrict x, int rows, int cols)
{
    dyn_matvec_q4_q8_fused_gate_up_neon(gate_out, up_out, q4_gate, q4_up, x, rows, cols);
}


static void dyn_matvec_q4_q8_fused_qkv(
    float *__restrict q_out, float *__restrict k_out, float *__restrict v_out,
    const uint8_t *__restrict q4_q, const uint8_t *__restrict q4_k,
    const uint8_t *__restrict q4_v, const float *__restrict x,
    int q_rows, int kv_rows, int cols)
{
    dyn_matvec_q4_q8_fused_qkv_neon(q_out, k_out, v_out, q4_q, q4_k, q4_v, x, q_rows, kv_rows, cols);
}


static void dyn_matvec_q4_q8_fused_o_residual(
    float *__restrict x, const float *__restrict residual,
    const uint8_t *__restrict q4_o, const float *__restrict attn_in,
    int rows, int cols)
{
    dyn_matvec_q4_q8_fused_o_residual_neon(x, residual, q4_o, attn_in, rows, cols);
}


/* Fused down projection + FFN residual, Q4_0 weights. */
static void dyn_matvec_q4_q8_fused_down_residual(
    float *__restrict x, const float *__restrict residual,
    const uint8_t *__restrict q4_down, const float *__restrict activated,
    int hidden_dim, int ffn_dim)
{
    dyn_matvec_q4_q8_fused_down_residual_neon(x, residual, q4_down, activated, hidden_dim, ffn_dim);
}

/* ================================================================
 * Batched Q4_0 Matvec Kernels (Mini-Batch Prefill Optimization)
 * (axiom: blas_precision_efficiency_tradeoff + block_matrix_assoc_natural)
 *
 * Q4_0 weight dequantization (nibble unpack + scale) is amortized across
 * the mini-batch: each block's 16 packed bytes are unpacked ONCE and the
 * resulting signed int8 weights are reused for every token in the tile.
 *
 * Weight stream: 18B/block (vs 34B/block for Q8_0), nearly halving prefill
 * weight DRAM reads; Q8_0 activation quantized once per matvec.
 * ================================================================ */





static void dyn_matvec_q4_q8_fused_qkv_batched(
    float *__restrict q_out, float *__restrict k_out, float *__restrict v_out,
    const uint8_t *__restrict q4_q, const uint8_t *__restrict q4_k,
    const uint8_t *__restrict q4_v, const float *__restrict x_batch,
    int q_rows, int kv_rows, int cols, int n_batch,
    const uint8_t *x8_q, const uint8_t *x8_k, const uint8_t *x8_v)
{
    /* Transparent NPU int4 offload (prefill QKV). Partial success falls
     * through to the CPU kernel, which recomputes every output. */
    if (st_npu_enabled() &&
        st_npu_try_gemm_batched_q4(q_out, x_batch, q4_q, q_rows, cols, n_batch,
                                   g_npu_layer, VLLM_NPU_PROJ_Q) &&
        st_npu_try_gemm_batched_q4(k_out, x_batch, q4_k, kv_rows, cols, n_batch,
                                   g_npu_layer, VLLM_NPU_PROJ_K) &&
        st_npu_try_gemm_batched_q4(v_out, x_batch, q4_v, kv_rows, cols, n_batch,
                                   g_npu_layer, VLLM_NPU_PROJ_V))
        return;
    dyn_matvec_q4_q8_fused_qkv_batched_neon(q_out, k_out, v_out, q4_q, q4_k, q4_v, x_batch, q_rows, kv_rows, cols, n_batch);
}


static void dyn_matvec_q4_q8_fused_o_residual_batched(
    float *__restrict x_batch, const float *__restrict residual_batch,
    const uint8_t *__restrict q4_o, const float *__restrict attn_batch,
    int rows, int cols, int n_batch, const uint8_t *x8_o)
{
    /* Transparent NPU int4 offload: x_batch = residual + O(attn). */
    if (st_npu_enabled() &&
        st_npu_try_gemm_batched_q4(x_batch, attn_batch, q4_o, rows, cols, n_batch,
                                   g_npu_layer, VLLM_NPU_PROJ_O)) {
        for (int t = 0; t < n_batch; t++)
            for (int r = 0; r < rows; r++)
                x_batch[(size_t)t * rows + r] += residual_batch[(size_t)t * rows + r];
        return;
    }
    dyn_matvec_q4_q8_fused_o_residual_batched_neon(x_batch, residual_batch, q4_o, attn_batch, rows, cols, n_batch);
}


static void dyn_matvec_q4_q8_fused_gate_up_batched(
    float *__restrict gate_out, float *__restrict up_out,
    const uint8_t *__restrict q4_gate, const uint8_t *__restrict q4_up,
    const float *__restrict x_batch, int rows, int cols, int n_batch,
    const uint8_t *x8_gate, const uint8_t *x8_up)
{
    /* Transparent NPU int4 offload (prefill gate/up). */
    if (st_npu_enabled() &&
        st_npu_try_gemm_batched_q4(gate_out, x_batch, q4_gate, rows, cols, n_batch,
                                   g_npu_layer, VLLM_NPU_PROJ_GATE) &&
        st_npu_try_gemm_batched_q4(up_out, x_batch, q4_up, rows, cols, n_batch,
                                   g_npu_layer, VLLM_NPU_PROJ_UP))
        return;
    dyn_matvec_q4_q8_fused_gate_up_batched_neon(gate_out, up_out, q4_gate, q4_up, x_batch, rows, cols, n_batch);
}


static void dyn_matvec_q4_q8_fused_down_residual_batched(
    float *__restrict x_batch, const float *__restrict residual_batch,
    const uint8_t *__restrict q4_down, const float *__restrict activated_batch,
    int hidden_dim, int ffn_dim, int n_batch, const uint8_t *x8_down)
{
    /* Transparent NPU int4 offload: x_batch = residual + down(activated). */
    if (st_npu_enabled() &&
        st_npu_try_gemm_batched_q4(x_batch, activated_batch, q4_down,
                                   hidden_dim, ffn_dim, n_batch,
                                   g_npu_layer, VLLM_NPU_PROJ_DOWN)) {
        for (int t = 0; t < n_batch; t++)
            for (int r = 0; r < hidden_dim; r++)
                x_batch[(size_t)t * hidden_dim + r] +=
                    residual_batch[(size_t)t * hidden_dim + r];
        return;
    }
    dyn_matvec_q4_q8_fused_down_residual_batched_neon(x_batch, residual_batch, q4_down, activated_batch, hidden_dim, ffn_dim, n_batch);
}

/* ---- end Q4_0 functions ---- */

typedef struct {
    float *out; const float *x; const float *w;
    int d8; float rms;
} vllm_rms_scale_ctx;

static void vllm_rms_scale_worker(void *ctx_, int it) {
    vllm_rms_scale_ctx *c = ctx_;
    int i = it * 8;
    for (int k = 0; k < 8 && i + k < c->d8; k++)
        c->out[i + k] = c->x[i + k] * c->rms * c->w[i + k];
}

/* 8 元素水平归约（镜像原 AVX2 树：lo+hi → 两两 → 两半和）。 */
static inline float hsum8_f32(const float v[8]) {
    float lo[4], hi[4];
    for (int i = 0; i < 4; i++) { lo[i] = v[i]; hi[i] = v[4 + i]; }
    for (int i = 0; i < 4; i++) lo[i] += hi[i];
    float s01 = lo[0] + lo[1], s23 = lo[2] + lo[3];
    return s01 + s23;
}

static void dyn_rms_norm(float *out, const float *x, const float *w,
                          int d, float eps) {
    float ss = 0.0f;
    int i;
    int d8 = d & ~7;
    for (i = 0; i < d8; i += 8) {
        float v[8];
        for (int k = 0; k < 8; k++) v[k] = x[i + k];
        for (int k = 0; k < 8; k++) v[k] = v[k] * v[k];
        ss += hsum8_f32(v);
    }
    for (i = d8; i < d; i++) ss += x[i] * x[i];
    float rms = 1.0f / sqrtf(ss / (float)d + eps);
    vllm_rms_scale_ctx vc = { out, x, w, d8, rms };
    vllm_tp_parfor(0, (d8 + 7) / 8, vllm_rms_scale_worker, &vc);
    for (i = d8; i < d; i++) out[i] = x[i] * rms * w[i];
}

typedef struct {
    float *out; const float *x; const float *w;
    int nb, d; float eps;
} vllm_rms_norm_batch_ctx;

static void vllm_rms_norm_batch_worker(void *ctx_, int t) {
    vllm_rms_norm_batch_ctx *c = ctx_;
    const float *xr = c->x + (size_t)t * c->d;
    float *or = c->out + (size_t)t * c->d;
    float ss = 0.0f;
    for (int i = 0; i < c->d; i++) ss += xr[i] * xr[i];
    float rms = 1.0f / sqrtf(ss / (float)c->d + c->eps);
    for (int i = 0; i < c->d; i++) or[i] = xr[i] * rms * c->w[i];
}

static void dyn_rms_norm_batch(float *out, const float *x, const float *w,
                               int nb, int d, float eps) {
    vllm_rms_norm_batch_ctx vc = { out, x, w, nb, d, eps };
    vllm_tp_parfor(0, nb, vllm_rms_norm_batch_worker, &vc);
}

typedef struct {
    float *out; const float *W; const float *x;
    int rows, cols;
} vllm_dyn_matvec_ctx;

static void vllm_dyn_matvec_worker(void *ctx_, int r) {
    vllm_dyn_matvec_ctx *c = ctx_;
    float sum = 0.0f;
    for (int cc = 0; cc < c->cols; cc++)
        sum += c->W[(size_t)r * c->cols + cc] * c->x[cc];
    c->out[r] = sum;
}

static void dyn_matvec(float *out, const float *W, const float *x,
                        int rows, int cols) {
    vllm_dyn_matvec_ctx vc = { out, W, x, rows, cols };
    vllm_tp_parfor(0, rows, vllm_dyn_matvec_worker, &vc);
}

typedef struct {
    float *gate_buf, *up_buf;
    int ffn_dim;
} vllm_swiglu_act_ctx;

static void vllm_swiglu_act_worker(void *ctx_, int i) {
    vllm_swiglu_act_ctx *c = ctx_;
    c->gate_buf[i] = silu_f(c->gate_buf[i]) * c->up_buf[i];
}

static void dyn_swiglu(float *out, const float *x,
                        const float *gate_W, const float *up_W,
                        const float *down_W,
                        int hidden_dim, int ffn_dim,
                        float *gate_buf, float *up_buf) {
    dyn_matvec(gate_buf, gate_W, x, ffn_dim, hidden_dim);
    dyn_matvec(up_buf, up_W, x, ffn_dim, hidden_dim);
    vllm_swiglu_act_ctx vc = { gate_buf, up_buf, ffn_dim };
    vllm_tp_parfor(0, ffn_dim, vllm_swiglu_act_worker, &vc);
    dyn_matvec(out, down_W, gate_buf, hidden_dim, ffn_dim);
}

/* MRoPE: 3D rotary position embedding with interleaved sections.
 * Qwen3-VL uses mrope_sections = [24, 20, 20] for head_dim=128.
 * The interleaved layout means: dim positions cycle through 3 groups.
 * Group 0 (1D, pos): positions 0, 1,   3, 4,   6, 7,   ...  (every 3rd pair)
 * Group 1 (2D, pos): positions     2, 5, 8, ...           (remainder 2)
 * Group 2 (3D, pos): actually... let me think about this.
 *
 * In Qwen3-VL's mrope_interleaved implementation:
 * - The rotary_dim is 64 (first 64 of 128)
 * - It uses 3 "sections" with different position values
 * - Interleaved: [sec0_pair0, sec1_pair0, sec2_pair0, sec0_pair1, sec1_pair1, sec2_pair1, ...]
 *
 * For our purposes (text-only with pos=0 for all 3 dimensions),
 * the MRoPE degenerates to standard RoPE since all dimensions use the same position. */
/* Precomputed MRoPE inverse-frequency table (bit-identical to the per-token
 * powf sequence: same args, same libm call, so each cached value equals what
 * the old loop produced on every token). Rebuilt only when (half, theta) change. */
static float *g_rope_freq = NULL;
static int    g_rope_half = 0;
static float  g_rope_theta = 0.0f;

static const float *rope_freq_table(int half, float theta) {
    if (g_rope_freq && g_rope_half == half && g_rope_theta == theta)
        return g_rope_freq;
    float *f = (float *)realloc(g_rope_freq, (size_t)half * sizeof(float));
    if (!f) return NULL;
    g_rope_freq = f;
    for (int j = 0; j < half; j++)
        f[j] = 1.0f / powf(theta, (float)j / (float)half);
    g_rope_half = half;
    g_rope_theta = theta;
    return f;
}

static void dyn_mrope(float *q, float *k, int hd, int n_heads, int n_kv_heads,
                       int hd_full, int pos, float theta) {
    /* Qwen3-VL text RoPE (degenerate MRoPE).
     *
     * In transformers (Qwen3VLTextRotaryEmbedding), the text-only case computes
     * the standard RoPE inverse frequencies over the FULL head_dim, then builds
     * cos/sin as cat((freqs, freqs)) and applies them via Llama's
     * `rotate_half`, which pairs dim j with dim j + hd/2 (NOT the adjacent
     * (2j, 2j+1) pair).  The mrope_section interleaving is a no-op for text
     * because all three position dims are equal.
     *
     * So the correct text-only RoPE rotates all `hd` dims, with pair (j, j+hd/2)
     * and frequency 1/theta^(j/(hd/2)). */
    int half = hd / 2;
    (void)hd_full;
    const float *freqs = rope_freq_table(half, theta);
    if (!freqs) return;   /* OOM: leave unrotated rather than crash */
    for (int j = 0; j < half; j++) {
        float freq = freqs[j];
        float angle = (float)pos * freq;
        float ca = cosf(angle);
        float sa = sinf(angle);

        /* Apply to Q: all heads */
        for (int h = 0; h < n_heads; h++) {
            float *qh = q + (size_t)h * hd;
            float v0 = qh[j];
            float v1 = qh[j + half];
            qh[j]         = v0 * ca - v1 * sa;
            qh[j + half]  = v1 * ca + v0 * sa;
        }

        /* Apply to K: only kv heads */
        for (int h = 0; h < n_kv_heads; h++) {
            float *kh = k + (size_t)h * hd;
            float v0 = kh[j];
            float v1 = kh[j + half];
            kh[j]         = v0 * ca - v1 * sa;
            kh[j + half]  = v1 * ca + v0 * sa;
        }
    }
}




/* Phase 2b: per-block KV allocation helpers. A layer cache is [n_blocks]
 * pointers, each to a bs*kv_dim block, so evicted blocks can be physically
 * freed (set to NULL) while the rest stays randomly addressable. */

/* Row accessor for position t within a per-block cache array. */
/* In-path 4KB guard probe (once per buffer): a >64B over-write past a
 * guarded buffer's logical end is caught here, between the writer kernel
 * and the next malloc that would otherwise fail-fast. */
#define GUARD_CHK(TAIL, NAME, TAG, L, TOK) do { \
    if (TAIL) { \
        const uint8_t *_p = (const uint8_t *)(TAIL); \
        int _bad = -1; \
        for (int _i = 0; _i < KV_GUARD; _i += 64) { \
            if (_p[_i] != 0xA5) { _bad = _i; break; } \
        } \
        if (_bad >= 0) { \
            fprintf(stderr, "[GUARD] %s broken +%d l=%d tok=%d\n", \
                    (TAG), _bad, (L), (TOK)); \
            fflush(stderr); \
        } \
    } \
} while (0)
static inline float *kv_row_f32(float **blocks, int t, int kv_dim, int bs) {
    return blocks[t / bs] + (size_t)(t % bs) * (size_t)kv_dim;
}
static inline int8_t *kv_row_i8(int8_t **blocks, int t, int kv_dim, int bs) {
    return blocks[t / bs] + (size_t)(t % bs) * (size_t)kv_dim;
}

/* ---- In-memory Q4 KV cache (l3_q4 kernels, no disk) ----
 * Payload layout per block: [bs tokens][nkv heads][hd/64 payloads][40B].
 * K and V live in separate per-layer pointer arrays (like the f32/i8 caches). */
#define Q4_PAYLOAD_LEN 40   /* = L3_Q4_PAYLOAD64 */

static inline int q4_np(int hd) { return hd / 64; }
static inline uint8_t *q4_k_row(uint8_t *const *blocks, int t,
                                int kh, int hd, int bs, int nkv) {
    int np = q4_np(hd);
    return blocks[t / bs]
         + (size_t)(t % bs) * (size_t)nkv * (size_t)np * Q4_PAYLOAD_LEN
         + (size_t)kh * (size_t)np * Q4_PAYLOAD_LEN;
}
static inline uint8_t *q4_v_row(uint8_t *const *blocks, int t,
                                int kh, int hd, int bs, int nkv) {
    return q4_k_row(blocks, t, kh, hd, bs, nkv);   /* same geometry */
}
/* Pack one token's K/V into the per-block payload rows. */
static void q4_pack_token(uint8_t *kdst, uint8_t *vdst,
                          const float *kt, const float *vt, int nkv, int hd) {
    int np = q4_np(hd);
    for (int h = 0; h < nkv; h++) {
        uint8_t *kp = kdst + (size_t)h * (size_t)np * Q4_PAYLOAD_LEN;
        uint8_t *vp = vdst + (size_t)h * (size_t)np * Q4_PAYLOAD_LEN;
        for (int g = 0; g < np; g++) {
            l3_q4_pack64(kp + (size_t)g * Q4_PAYLOAD_LEN, kt + (size_t)h * hd + (size_t)g * 64);
            l3_q4_pack64(vp + (size_t)g * Q4_PAYLOAD_LEN, vt + (size_t)h * hd + (size_t)g * 64);
        }
    }
}
/* Q·K dot against one head's payloads, and wgt*V accumulation (no dequant). */
static float q4_head_dot(const uint8_t *p, const float *q, int hd) {
    int np = q4_np(hd);
    float sum = 0.0f;
    for (int g = 0; g < np; g++)
        sum += l3_q4_dot64(p + (size_t)g * Q4_PAYLOAD_LEN, q + (size_t)g * 64);
    return sum;
}
static void q4_head_vacc(float *acc, const uint8_t *p, float wgt, int hd) {
    int np = q4_np(hd);
    for (int g = 0; g < np; g++)
        l3_q4_vacc64(acc + (size_t)g * 64, p + (size_t)g * Q4_PAYLOAD_LEN, wgt);
}
/* Per-block payload allocation for one layer: [n_blocks] pointers, each
 * [bs * nkv * (hd/64) * 40] bytes. */
static void *cf_pg_alloc(size_t bytes);
static void cf_pg_free(void *p);
/* Debug page-guard allocator mode: 1 = VirtualAlloc+guard page, 0 = calloc.
 * MUST be 1 before st_qwen_inference_init so KV blocks are allocated with the
 * same allocator that st_qwen_inference_free releases (VirtualFree). */
static int g_xq_pages = 0;
/* VLLM_XQ_PAGES=0 forces the CRT heap allocators (calloc/_aligned_malloc) so
 * an AddressSanitizer build can cover the KV blocks and scratch buffers;
 * the default 1 keeps the VirtualAlloc+guard-page layout of normal x86 builds.
 * On non-x86 targets (Linux/aarch64, e.g. RK3588) the default is 0: the
 * Linux page-guard path (mmap+mprotect + a 256-entry registry) overflows at
 * inference init (KV cache alone makes ~37k guarded allocations), so the
 * unregistered per-prefill scratch allocations can never be munmap'd and the
 * prefill scratch allocs eventually fail ("wbuf OOM"), degrading to a broken
 * token-by-token fallback. Allocator mode must stay consistent within one
 * inference state lifetime. */
static int xq_pages_mode(void) {
    const char *e = getenv("VLLM_XQ_PAGES");
    if (e && e[0]) return (e[0] == '0') ? 0 : 1;
    /* 默认关闭: 页面守卫是调试工具(每次 GEMV 调 VirtualAlloc/Free), decode
     * 每步 ~180 次分配/释放, 实测拖慢 decode。需要排查越界写时用 VLLM_XQ_PAGES=1 开启。 */
    return 0;
}

static int pb_heap_mode(void) {
    const char *e = getenv("VLLM_PB_HEAP");
    if (!e || e[0] == '0') return 0;
    return 1;
}
static int prefill_debug_mode(void) {
    const char *e = getenv("VLLM_PREFILL_DEBUG");
    if (!e || e[0] == '0') return 0;
    return 1;
}
/* Debug-only canary gap (bytes inserted after each scratch/xq buffer so an
 * out-of-bounds write lands in the gap instead of the next allocation; the
 * gap size bounds how far the write overruns). 0 = exact sizes (normal). */
static size_t xq_pad_env(const char *name) {
    const char *e = getenv(name);
    if (!e) return 0;
    return (size_t)atol(e);
}
static uint8_t **alloc_q4_blocks(int n_blocks, int bs, int nkv, int hd) {
    int np = q4_np(hd);
    size_t blk = (size_t)bs * (size_t)nkv * (size_t)np * Q4_PAYLOAD_LEN;
    /* Pointer array on the guard allocator too: an out-of-bounds arr[b] write
     * trips the guard page as an AV instead of corrupting the CRT heap. */
    uint8_t **arr = (uint8_t **)cf_pg_alloc((size_t)n_blocks * sizeof(uint8_t *) + 64);
    if (!arr) return NULL;
    memset(arr, 0, (size_t)n_blocks * sizeof(uint8_t *));
    memset((uint8_t *)arr + (size_t)n_blocks * sizeof(uint8_t *), 0xA5, 64);
    for (int b = 0; b < n_blocks; b++) {
        uint8_t *raw = (uint8_t *)cf_pg_alloc(KV_GUARD + blk + KV_GUARD);
        if (!raw) {
            for (int j = 0; j < b; j++) cf_pg_free(arr[j] - KV_GUARD);
            cf_pg_free(arr);
            return NULL;
        }
        memset(raw, 0xA5, KV_GUARD);
        memset(raw + KV_GUARD + blk, 0xA5, KV_GUARD);
        arr[b] = raw + KV_GUARD;
    }
    return arr;
}

static float **alloc_kv_blocks_f32(int n_blocks, int bs, int kv_dim,
                                   uint8_t **arr_tail, uint8_t **last_tail) {
    float **arr = (float **)cf_pg_alloc((size_t)n_blocks * sizeof(float *) + 64);
    if (!arr) return NULL;
    memset(arr, 0, (size_t)n_blocks * sizeof(float *));
    if (arr_tail) {
        *arr_tail = (uint8_t *)arr + (size_t)n_blocks * sizeof(float *);
        memset(*arr_tail, 0xA5, 64);
    }
    size_t dat = (size_t)bs * (size_t)kv_dim * sizeof(float);
    for (int b = 0; b < n_blocks; b++) {
        uint8_t *raw = (uint8_t *)cf_pg_alloc(KV_GUARD + dat + KV_GUARD);
        if (!raw) {
            for (int j = 0; j < b; j++) cf_pg_free(arr[j] - KV_GUARD);
            cf_pg_free(arr);
            return NULL;
        }
        memset(raw, 0xA5, KV_GUARD);
        memset(raw + KV_GUARD + dat, 0xA5, KV_GUARD);
        arr[b] = (float *)(raw + KV_GUARD);
    }
    if (last_tail) {
        *last_tail = (uint8_t *)arr[n_blocks - 1] + dat;
        memset(*last_tail, 0xA5, 64);
    }
    return arr;
}
/* Canary/bounds-check helpers (defined near prefill; forward-declared for init).
 * NOTE: the +64B tail allocation is retained for heap-layout stability; the
 * check reads were removed (multi-state serving made the global tails dangle). */
static void *cf_calloc_canary(size_t n, size_t sz, uint8_t **tail);
static void *cf_calloc_guard(size_t n, size_t sz, uint8_t **tail);
static uint8_t *g_cn_scores = NULL, *g_cn_imphead = NULL, *g_cn_importance = NULL;
/* tails for the per-state scratch buffers (hidden/logits/matvec bufs/cache_len) */
static uint8_t *g_cn_hidden = NULL, *g_cn_logits = NULL, *g_cn_qbuf = NULL;
static uint8_t *g_cn_kbuf = NULL, *g_cn_vbuf = NULL, *g_cn_attnbuf = NULL;
static uint8_t *g_cn_ffnbuf = NULL, *g_cn_ffnout = NULL, *g_cn_cachelen = NULL;
/* Axiom arith_range_bound_001 (memory-bounds, KV edition): tails for the
 * per-layer KV pointer arrays, their last block, and the INT8 scale arrays. */
#define ST_CN_MAX_LAYERS 64
static uint8_t *g_cn_karr[ST_CN_MAX_LAYERS]  = {0}, *g_cn_varr[ST_CN_MAX_LAYERS]  = {0};
static uint8_t *g_cn_kq8arr[ST_CN_MAX_LAYERS] = {0}, *g_cn_vq8arr[ST_CN_MAX_LAYERS] = {0};
static uint8_t *g_cn_kblk[ST_CN_MAX_LAYERS]  = {0}, *g_cn_vblk[ST_CN_MAX_LAYERS]  = {0};
static uint8_t *g_cn_kq8blk[ST_CN_MAX_LAYERS] = {0}, *g_cn_vq8blk[ST_CN_MAX_LAYERS] = {0};
static uint8_t *g_cn_kscale[ST_CN_MAX_LAYERS] = {0}, *g_cn_vscale[ST_CN_MAX_LAYERS] = {0};

static int8_t **alloc_kv_blocks_i8(int n_blocks, int bs, int kv_dim,
                                   uint8_t **arr_tail, uint8_t **last_tail) {
    int8_t **arr = (int8_t **)cf_pg_alloc((size_t)n_blocks * sizeof(int8_t *) + 64);
    if (!arr) return NULL;
    memset(arr, 0, (size_t)n_blocks * sizeof(int8_t *));
    if (arr_tail) {
        *arr_tail = (uint8_t *)arr + (size_t)n_blocks * sizeof(int8_t *);
        memset(*arr_tail, 0xA5, 64);
    }
    size_t dat = (size_t)bs * (size_t)kv_dim;
    for (int b = 0; b < n_blocks; b++) {
        uint8_t *raw = (uint8_t *)cf_pg_alloc(KV_GUARD + dat + KV_GUARD);
        if (!raw) {
            for (int j = 0; j < b; j++) cf_pg_free(arr[j] - KV_GUARD);
            cf_pg_free(arr);
            return NULL;
        }
        memset(raw, 0xA5, KV_GUARD);
        memset(raw + KV_GUARD + dat, 0xA5, KV_GUARD);
        arr[b] = (int8_t *)(raw + KV_GUARD);
    }
    if (last_tail) {
        *last_tail = (uint8_t *)arr[n_blocks - 1] + dat;
        memset(*last_tail, 0xA5, 64);
    }
    return arr;
}

/* [NPU-WSCAN] weight-buffer NaN-scale forensics (VLLM_NPU_WSCAN=1).
 * Scans q8_o_weight / q8_down_weight scale slots to localize the M5a
 * corruption fence: G256DBG proves load-time scales are valid, yet they are
 * deterministically NaN by prefill entry. Call sites: init-begin (before KV
 * alloc) and prefill-entry (after load, before inference GEMMs). */
static void npu_wscan(const char *tag, const STModelWeights *w,
                      int d, int ff, int nl) {
    if (!getenv("VLLM_NPU_WSCAN") || !w->q8_o_weight) return;
    int o_nan = 0, o_blk = 0, d_nan = 0, d_blk = 0;
    int f_l = -1, f_r = -1, f_g = -1, f_dl = -1;
    for (int l = 0; l < nl; l++) {
        int lo_nan = 0;
        const uint8_t *ow = w->q8_o_weight + Q8_BYTES((size_t)l * d * d);
        for (int r = 0; r < d; r++) {
            const uint8_t *pr = ow + (size_t)r * ((d / 32) * 34);
            for (int g = 0; g < d / 32; g++) {
                uint16_t h; memcpy(&h, pr + (size_t)g * 34, 2);
                uint32_t fb = f16_to_f32_bits(h);
                float v; memcpy(&v, &fb, 4);
                o_blk++;
                if (npu_isnan_f32(v)) {
                    o_nan++; lo_nan++;
                    if (f_l < 0) { f_l = l; f_r = r; f_g = g; }
                }
            }
        }
        if (lo_nan > 0)
            fprintf(stderr, "[NPU-WSCAN]   %s l=%d o_nan=%d\n", tag, l, lo_nan);
        if (w->q8_down_weight) {
            int ld_nan = 0;
            const uint8_t *dw = w->q8_down_weight + Q8_BYTES((size_t)l * d * ff);
            for (int r = 0; r < d; r++) {
                const uint8_t *pr = dw + (size_t)r * ((ff / 32) * 34);
                for (int g = 0; g < ff / 32; g++) {
                    uint16_t h; memcpy(&h, pr + (size_t)g * 34, 2);
                    uint32_t fb = f16_to_f32_bits(h);
                    float v; memcpy(&v, &fb, 4);
                    d_blk++;
                    if (npu_isnan_f32(v)) {
                        d_nan++; ld_nan++;
                        if (f_dl < 0) f_dl = l;
                    }
                }
            }
            if (ld_nan > 0)
                fprintf(stderr, "[NPU-WSCAN]   %s l=%d down_nan=%d\n", tag, l, ld_nan);
        }
    }
    fprintf(stderr,
            "[NPU-WSCAN] %s q8_o nan=%d/%d q8_down nan=%d/%d first_o=layer%d_r%d_g%d first_d=layer%d\n",
            tag, o_nan, o_blk, d_nan, d_blk, f_l, f_r, f_g, f_dl);
    fflush(stderr);
}

typedef struct {
    float *q_buf; const float *qw; int hd, nh;
} vllm_qk_norm_q_ctx;

static void vllm_qk_norm_q_worker(void *ctx_, int hq) {
    vllm_qk_norm_q_ctx *c = ctx_;
    float *qh = c->q_buf + hq * c->hd;
    float ss = 0.0f;
    for (int i = 0; i < c->hd; i++) ss += qh[i] * qh[i];
    float rms = 1.0f / sqrtf(ss / (float)c->hd + 1e-6f);
    for (int i = 0; i < c->hd; i++) qh[i] = qh[i] * rms * c->qw[i];
}

typedef struct {
    float *k_buf; const float *kw; int hd, nkv;
} vllm_qk_norm_k_ctx;

static void vllm_qk_norm_k_worker(void *ctx_, int hk) {
    vllm_qk_norm_k_ctx *c = ctx_;
    float *kh = c->k_buf + hk * c->hd;
    float ss = 0.0f;
    for (int i = 0; i < c->hd; i++) ss += kh[i] * kh[i];
    float rms = 1.0f / sqrtf(ss / (float)c->hd + 1e-6f);
    for (int i = 0; i < c->hd; i++) kh[i] = kh[i] * rms * c->kw[i];
}

int st_qwen_inference_init(STQwenInferenceState *st, const STModelWeights *weights) {
    /* Allocator mode must match st_qwen_inference_free: the prefill path sets
     * g_xq_pages=1 (VirtualAlloc+guard-page blocks) and free() releases those
     * with VirtualFree. If init runs with g_xq_pages=0 the KV blocks are plain
     * calloc, and the later VirtualFree in cf_pg_free then frees a heap
     * segment (layout-sensitive heap corruption at the first state's free). */
    g_xq_pages = xq_pages_mode();
    memset(st, 0, sizeof(*st));
    if (getenv("VLLM_DEC_PROF")) st->profile_decode = 1; /* per-step decode profile */
    st->cfg = weights->cfg;
    memcpy(&st->weights, weights, sizeof(STModelWeights));
    npu_wscan("init-begin", weights, weights->cfg.dim, weights->cfg.ffn_dim,
              weights->n_layers_allocated);

    int d  = st->cfg.dim;
    int nl = weights->n_layers_allocated;
    int nh = st->cfg.n_heads;
    int nkv = st->cfg.n_kv_heads;
    int hd = st->cfg.head_dim;
    int kv_dim = nkv * hd;
    int ff = st->cfg.ffn_dim;
    int vc = st->cfg.vocab_size;
    int max_seq = st->cfg.max_seq_len;
    if (max_seq > 32768) max_seq = 2048; /* cap for memory */
    /* --bench-seqlen N: lift the cap so long-context benchmarks (N>2048)
     * have KV headroom. Exact prefill padding happens in the caller. */
    extern int g_bench_seqlen;
    if (g_bench_seqlen > 0) {
        int need = g_bench_seqlen + 128;
        if (need > max_seq) max_seq = need;
    }

    /* Q/K/V temp bufs: must be large enough for both attention (nh*hd / nkv*hd)
     * AND FFN swiglu gate/up projections (ffn_dim) */
    int q_buf_sz = (nh * hd > ff) ? nh * hd : ff;
    int k_buf_sz = (nkv * hd > ff) ? nkv * hd : ff;
    st->hidden     = cf_calloc_guard(d, sizeof(float), &g_cn_hidden);
    st->logits     = cf_calloc_guard(vc, sizeof(float), &g_cn_logits);
    st->q_buf      = cf_calloc_guard(q_buf_sz, sizeof(float), &g_cn_qbuf);
    st->k_buf      = cf_calloc_guard(k_buf_sz, sizeof(float), &g_cn_kbuf);
    st->v_buf      = cf_calloc_guard(nkv * hd, sizeof(float), &g_cn_vbuf);
    st->attn_buf   = cf_calloc_guard(nh * hd, sizeof(float), &g_cn_attnbuf);
    st->ffn_buf    = cf_calloc_guard(ff, sizeof(float), &g_cn_ffnbuf);
    st->ffn_out_buf = cf_calloc_guard(d, sizeof(float), &g_cn_ffnout);
    st->cache_len  = cf_calloc_guard(nl, sizeof(int), &g_cn_cachelen);
    st->k_cache    = (float ***)cf_pg_alloc((size_t)nl * sizeof(float **));
    st->v_cache    = (float ***)cf_pg_alloc((size_t)nl * sizeof(float **));
    memset(st->k_cache, 0, (size_t)nl * sizeof(float **));
    memset(st->v_cache, 0, (size_t)nl * sizeof(float **));

    /* Phase 2b: KV caches are per-block pointer arrays (block = bs positions),
     * aligned with the sparse-attention / L3 eviction block size. */
    int bs = g_sparse_block < 8 ? 8 : g_sparse_block;
    int n_blocks = (max_seq + bs - 1) / bs;
    if (n_blocks < 1) n_blocks = 1;
    st->kv_bs = bs;
    st->kv_n_blocks = n_blocks;

    int failed = 0;
    #define CHK(p, n) if (!(p)) { fprintf(stderr, "[QWEN] OOM allocating %s\n", n); failed = 1; }
    CHK(st->hidden, "hidden");
    CHK(st->logits, "logits");
    CHK(st->q_buf, "q_buf");
    CHK(st->k_buf, "k_buf");
    CHK(st->v_buf, "v_buf");
    CHK(st->attn_buf, "attn_buf");
    CHK(st->ffn_buf, "ffn_buf");
    CHK(st->ffn_out_buf, "ffn_out_buf");
    CHK(st->cache_len, "cache_len");
    CHK(st->k_cache, "k_cache");
    CHK(st->v_cache, "v_cache");
    #undef CHK

    if (failed) { st_qwen_inference_free(st); return -1; }

    /* Compressed KV cache: INT8 (near-lossless) by default; --kv-q4 switches
     * to the Q4_0 payload cache (~1.65x denser, decode dot without dequant).
     * DEBUG (W5): force float KV to test whether the INT8 path is corrupting
     * the K cache for Qwen3-0.6B (int8 K rows != float K rows observed).
     * The 0.6B corruption was root-caused to a head_dim parse bug (now fixed
     * above); the 8B INT8 KV is verified (M4e/M4f PPL≈0.994-0.998). Restore
     * the production default so decode uses the INT8 flash attention. */
    st->use_kv_q8 = g_kv_q4 ? 0 : 1;
    st->use_kv_q4 = g_kv_q4 ? 1 : 0;

    if (st->use_kv_q8) {
        st->k_cache_q8 = (int8_t ***)cf_pg_alloc((size_t)nl * sizeof(int8_t **));
        st->v_cache_q8 = (int8_t ***)cf_pg_alloc((size_t)nl * sizeof(int8_t **));
        st->k_scale = (float **)cf_pg_alloc((size_t)nl * sizeof(float *));
        st->v_scale = (float **)cf_pg_alloc((size_t)nl * sizeof(float *));
        if (!st->k_cache_q8 || !st->v_cache_q8 || !st->k_scale || !st->v_scale) {
            fprintf(stderr, "[QWEN] OOM allocating INT8 KV cache metadata\n");
            st_qwen_inference_free(st); return -1;
        }
        memset(st->k_cache_q8, 0, (size_t)nl * sizeof(int8_t **));
        memset(st->v_cache_q8, 0, (size_t)nl * sizeof(int8_t **));
        memset(st->k_scale, 0, (size_t)nl * sizeof(float *));
        memset(st->v_scale, 0, (size_t)nl * sizeof(float *));
        for (int l = 0; l < nl; l++) {
            st->k_cache_q8[l] = alloc_kv_blocks_i8(n_blocks, bs, kv_dim,
                                                    &g_cn_kq8arr[l], &g_cn_kq8blk[l]);
            st->v_cache_q8[l] = alloc_kv_blocks_i8(n_blocks, bs, kv_dim,
                                                    &g_cn_vq8arr[l], &g_cn_vq8blk[l]);
            st->k_scale[l] = (float *)cf_calloc_guard((size_t)max_seq * nkv, sizeof(float),
                                                      &g_cn_kscale[l]);
            st->v_scale[l] = (float *)cf_calloc_guard((size_t)max_seq * nkv, sizeof(float),
                                                      &g_cn_vscale[l]);
            if (!st->k_cache_q8[l] || !st->v_cache_q8[l] || !st->k_scale[l] || !st->v_scale[l]) {
                fprintf(stderr, "[QWEN] OOM allocating INT8 KV cache layer %d\n", l);
                st_qwen_inference_free(st); return -1;
            }
        }
    } else if (st->use_kv_q4) {
        st->k_cache_q4 = (uint8_t ***)cf_pg_alloc((size_t)nl * sizeof(uint8_t **));
        st->v_cache_q4 = (uint8_t ***)cf_pg_alloc((size_t)nl * sizeof(uint8_t **));
        if (!st->k_cache_q4 || !st->v_cache_q4) {
            fprintf(stderr, "[QWEN] OOM allocating Q4 KV cache metadata\n");
            st_qwen_inference_free(st); return -1;
        }
        memset(st->k_cache_q4, 0, (size_t)nl * sizeof(uint8_t **));
        memset(st->v_cache_q4, 0, (size_t)nl * sizeof(uint8_t **));
        for (int l = 0; l < nl; l++) {
            st->k_cache_q4[l] = alloc_q4_blocks(n_blocks, bs, nkv, hd);
            st->v_cache_q4[l] = alloc_q4_blocks(n_blocks, bs, nkv, hd);
            if (!st->k_cache_q4[l] || !st->v_cache_q4[l]) {
                fprintf(stderr, "[QWEN] OOM allocating Q4 KV cache layer %d\n", l);
                st_qwen_inference_free(st); return -1;
            }
        }
    }

    for (int l = 0; l < nl; l++) {
        st->k_cache[l] = alloc_kv_blocks_f32(n_blocks, bs, kv_dim, &g_cn_karr[l], &g_cn_kblk[l]);
        st->v_cache[l] = alloc_kv_blocks_f32(n_blocks, bs, kv_dim, &g_cn_varr[l], &g_cn_vblk[l]);
        if (!st->k_cache[l] || !st->v_cache[l]) {
            fprintf(stderr, "[QWEN] OOM allocating KV cache layer %d (seq=%d)\n", l, max_seq);
            st_qwen_inference_free(st);
            return -1;
        }
    }

    st->is_allocated = 1;

    /* Store max_seq for parallel attention stride */
    st->max_kv_slots = max_seq;

    /* Shared scores buffer: [max_seq * nh * 4] per-head rows for parallel
     * attention. 4x 行数：M4j 4-token 分块注意力每 head 同时用 4 行
     * （行 ha*4+k，stride = max_seq）。Axiom: block_parallel_injection. */
    st->scores_buf = (float *)cf_calloc_guard((size_t)max_seq * nh * 4, sizeof(float), &g_cn_scores);
    if (!st->scores_buf) {
        fprintf(stderr, "[QWEN] OOM allocating scores_buf\n");
        st_qwen_inference_free(st); return -1;
    }

    /* Phase 1.5: prefill-importance buffers (sparse decode selection signal) */
    st->prefill_importance = (float *)cf_calloc_guard((size_t)max_seq, sizeof(float), &g_cn_importance);
    st->imp_head = (float *)cf_calloc_guard((size_t)max_seq * nh, sizeof(float), &g_cn_imphead);
    if (!st->prefill_importance || !st->imp_head) {
        fprintf(stderr, "[QWEN] OOM allocating prefill_importance\n");
        st_qwen_inference_free(st); return -1;
    }

    fprintf(stderr, "[QWEN] Inference state ready (max_seq=%d).\n", max_seq);
    return 0;
}

/* Flash Attention blocked kernel — packs KV cache tiles into contiguous L1-friendly
 * buffers and uses online softmax across tiles. Optimized for single Q row (nq=1).
 *
 * Problem: K/V stored as [seq_len][kv_dim] with stride=kv_dim=1024.
 * Loading K[t][kh*hd..kh*hd+hd] for each t loads 8 cache lines at 4096-byte strides,
 * causing cache evictions between rows. For seq_len=2048 this is ~16K L1 misses.
 *
 * Solution: Pack FA_KV_TILE=64 rows into K_packed[64][hd=128] = 32 KB contiguous.
 * This fits in L1 cache (32 KB per core) and turns strided reads into linear.
 * Online softmax fuses across tiles: rescale old VKQ accumulator by exp(M_old - M_new).
 *
 * For nq=1, each head processes independently — ideal for OpenMP head parallelism.
 * Axiom: memory_bandwidth_reduction — tile packing eliminates stride cache misses. */
static void flash_attn_blocked_single_q(
    float *restrict attn_out,          /* [hd] output accumulator */
    const float *restrict q,           /* [hd] single Q row */
    float *const *k_cache,             /* [n_blocks] per-block K cache */
    float *const *v_cache,             /* [n_blocks] per-block V cache */
    int seq_len,
    int kh_off,                        /* KV head byte-offset = kh * hd */
    int kv_dim,                        /* full KV row stride (nkv * hd = 1024) */
    int hd,                            /* head dimension (128) */
    int bs,                            /* positions per block */
    float scale)                       /* 1/sqrt(hd) */
{
    int hd8 = hd & ~7;

    /* Online softmax running state */
    float M = -1e9f;          /* running max score */
    float S = 0.0f;           /* running sum of exp(score - M) */

    /* Zero VKQ accumulator */
    for (int i = 0; i < hd; i++)
        attn_out[i] = 0.0f;

    /* Process KV rows in tiles of FA_KV_TILE */
    for (int t_start = 0; t_start < seq_len; t_start += FA_KV_TILE) {
        int tile_sz = seq_len - t_start;
        if (tile_sz > FA_KV_TILE) tile_sz = FA_KV_TILE;

        /* Stage 1: Pack K tile into K_packed[tile_sz][hd] (contiguous, transposed view).
         * K_packed[t][i] = k_cache[(t_start+t)*kv_dim + kh_off + i] */
        float K_packed[FA_KV_TILE * 128];   /* max 64*128*4 = 32 KB on stack */
        for (int t = 0; t < tile_sz; t++) {
            const float *ks = k_cache[(t_start + t) / bs] + (size_t)((t_start + t) % bs) * (size_t)kv_dim + kh_off;
            float *kp = K_packed + (size_t)t * hd;
            for (int i = 0; i < hd; i++) kp[i] = ks[i];
        }

        /* Stage 2: Q·K^T for this tile — NEON 8-way FMA dot products */
        float scores[FA_KV_TILE];
        float M_tile = M;
        for (int t = 0; t < tile_sz; t++) {
            const float *kp = K_packed + (size_t)t * hd;
            float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = vdupq_n_f32(0.0f);
            for (int i = 0; i < hd8; i += 8) {
                acc0 = vfmaq_f32(acc0, vld1q_f32(q + i), vld1q_f32(kp + i));
                acc1 = vfmaq_f32(acc1, vld1q_f32(q + i + 4), vld1q_f32(kp + i + 4));
            }
            float32x4_t s = vaddq_f32(acc0, acc1);
            s = vpaddq_f32(s, s);
            s = vpaddq_f32(s, s);
            float dot = vgetq_lane_f32(s, 0);
            for (int i = hd8; i < hd; i++) dot += q[i] * kp[i];
            scores[t] = dot * scale;
            if (scores[t] > M_tile) M_tile = scores[t];
        }

        /* Stage 3: Online softmax — if new max exceeds running max, rescale old VKQ */
        if (M_tile > M) {
            float rescale = expf(M - M_tile);
            for (int i = 0; i < hd; i++) attn_out[i] *= rescale;
            S *= rescale;
            M = M_tile;
        }

        /* Accumulate exp(score - M) into S */
        for (int t = 0; t < tile_sz; t++) {
            scores[t] = expf(scores[t] - M);
            S += scores[t];
        }

        /* Stage 4: Pack V tile into V_packed[tile_sz][hd] (contiguous) */
        float V_packed[FA_KV_TILE * 128];   /* max 64*128*4 = 32 KB on stack */
        for (int t = 0; t < tile_sz; t++) {
            const float *vs = v_cache[(t_start + t) / bs] + (size_t)((t_start + t) % bs) * (size_t)kv_dim + kh_off;
            float *vp = V_packed + (size_t)t * hd;
            for (int i = 0; i < hd; i++) vp[i] = vs[i];
        }

        /* Stage 5: VKQ += softmax(scores) * V (unnormalized, divide by S at end) */
        for (int t = 0; t < tile_sz; t++) {
            float wgt = scores[t];
            const float *vp = V_packed + (size_t)t * hd;
            float32x4_t wv = vdupq_n_f32(wgt);
            for (int i = 0; i < hd8; i += 8) {
                float32x4_t ov0 = vld1q_f32(attn_out + i);
                float32x4_t ov1 = vld1q_f32(attn_out + i + 4);
                vst1q_f32(attn_out + i, vfmaq_f32(ov0, wv, vld1q_f32(vp + i)));
                vst1q_f32(attn_out + i + 4, vfmaq_f32(ov1, wv, vld1q_f32(vp + i + 4)));
            }
            for (int i = hd8; i < hd; i++)
                attn_out[i] += wgt * vp[i];
        }
    }

    /* Final normalization: attn_out /= S */
    float inv_S = 1.0f / S;
    for (int i = 0; i < hd; i++)
        attn_out[i] *= inv_S;
}

/* Copy the KV cache for the first `prefix_len`
 * tokens from `src` into `dst`, so `dst` can continue generation from that
 * shared prefix without re-prefilling it. This is an exact byte-for-byte copy
 * (axiom: linear_operator_on_set_state + partition_alignment), so downstream
 * output is bit-identical to a fresh prefill of the same prefix — zero quality
 * loss. Caller must ensure `dst` was init'd with the same model config. */
void st_qwen_copy_kv_prefix(STQwenInferenceState *dst,
                            const STQwenInferenceState *src,
                            int prefix_len) {
    int nl = src->weights.n_layers_allocated;
    int nkv = src->cfg.n_kv_heads;
    int hd  = src->cfg.head_dim;
    int kv_dim = nkv * hd;

    if (prefix_len <= 0) return;

    int bs = src->kv_bs > 0 ? src->kv_bs : 32;
    int nblocks = (prefix_len + bs - 1) / bs;

    for (int l = 0; l < nl; l++) {
        /* Phase 2b: per-block copy (blocks are individually allocated) */
        for (int b = 0; b < nblocks; b++) {
            if (src->k_cache[l][b] && dst->k_cache[l][b])
                memcpy(dst->k_cache[l][b], src->k_cache[l][b],
                       (size_t)bs * (size_t)kv_dim * sizeof(float));
            if (src->v_cache[l][b] && dst->v_cache[l][b])
                memcpy(dst->v_cache[l][b], src->v_cache[l][b],
                       (size_t)bs * (size_t)kv_dim * sizeof(float));
        }
        dst->cache_len[l] = prefix_len;

        if (dst->use_kv_q8 && src->k_cache_q8 && dst->k_cache_q8) {
            for (int b = 0; b < nblocks; b++) {
                if (src->k_cache_q8[l][b] && dst->k_cache_q8[l][b])
                    memcpy(dst->k_cache_q8[l][b], src->k_cache_q8[l][b],
                           (size_t)bs * (size_t)kv_dim * sizeof(int8_t));
                if (src->v_cache_q8[l][b] && dst->v_cache_q8[l][b])
                    memcpy(dst->v_cache_q8[l][b], src->v_cache_q8[l][b],
                           (size_t)bs * (size_t)kv_dim * sizeof(int8_t));
            }
            memcpy(dst->k_scale[l], src->k_scale[l],
                   (size_t)prefix_len * (size_t)nkv * sizeof(float));
            memcpy(dst->v_scale[l], src->v_scale[l],
                   (size_t)prefix_len * (size_t)nkv * sizeof(float));
        } else if (dst->use_kv_q4 && src->k_cache_q4 && dst->k_cache_q4) {
            int np = q4_np(hd);
            size_t blk = (size_t)bs * (size_t)nkv * (size_t)np * Q4_PAYLOAD_LEN;
            for (int b = 0; b < nblocks; b++) {
                if (src->k_cache_q4[l][b] && dst->k_cache_q4[l][b])
                    memcpy(dst->k_cache_q4[l][b], src->k_cache_q4[l][b], blk);
                if (src->v_cache_q4[l][b] && dst->v_cache_q4[l][b])
                    memcpy(dst->v_cache_q4[l][b], src->v_cache_q4[l][b], blk);
            }
        }
    }
    dst->seq_len = prefix_len;
}

/* Block-sparse decode attention for one query head (Phase 1).
 *
 * Pipeline: (1) probe each KV block via dot(query, K[block_first_pos]) - O(1)
 * per block; (2) deterministic top-k block selection (tie-break by index);
 * (3) exact scaled dot-product scores over selected blocks only; (4) softmax
 * renormalized over the selected set; (5) weighted V sum over selected blocks.
 * Bounds per-head work to O(seq/BS + k*BS*head_dim) vs O(seq*head_dim).
 *
 * Axiom: probabilistic_selection_nc ("top k by weight"),
 *        blas_sparse_message_passing_schedule (block-level sparse schedule),
 *        determinism red line (axiom_arith_gumbel_argmax_001).
 */
static void sparse_attn_head(float *__restrict attn_out, const float *__restrict qt,
                             float *const *__restrict k_cache,   /* [n_blocks] block rows, NULL = evicted */
                             float *const *__restrict v_cache,
                             int8_t *const *__restrict k_q8,
                             int8_t *const *__restrict v_q8,
                             const float *__restrict kscale,
                             const float *__restrict vscale,
                             int use_q8, int seq_len, int kv_dim, int nkv,
                             int kh, int hd, float scale,
                             int bs, int k_blocks, int n_probe,
                             const float *__restrict importance,
                             float *__restrict scores,
                             const STL3State *__restrict l3,
                             int layer) {
    int hd8 = hd & ~7;
    int n_blocks = (seq_len + bs - 1) / bs;
    if (n_blocks <= 0) return;

    /* Phase-2c: per-head L3 batch staging — one block's K or V area (single
     * head) staged with a single copy, then all token dots / V accumulations
     * run over the staged buffer instead of issuing per-token reads.
     * NULL on OOM falls back to the per-token read path below. */
    int n_payloads = hd / 64;
    size_t l3_chunk = (size_t)bs * (size_t)n_payloads * L3_Q4_PAYLOAD64;
    uint8_t *l3k_buf = l3 ? (uint8_t *)malloc(l3_chunk + 64) : NULL;
    uint8_t *l3v_buf = l3 ? (uint8_t *)malloc(l3_chunk + 64) : NULL;

    /* Phase 2b: block-array row accessor (blocks may live in RAM or on disk). */
#define BLK(rows, t) ((rows)[(t) / bs] + (size_t)((t) % bs) * (size_t)kv_dim)
#define L3BM(b) ((l3) ? &l3->blocks[(size_t)layer * l3->max_blocks + (b)] : NULL)

    float  *probe = (float *)malloc((size_t)n_blocks * sizeof(float) + 64);
    float  *imp_sum = importance ? (float *)calloc((size_t)n_blocks + 16, sizeof(float)) : NULL;
    uint8_t *used = (uint8_t *)malloc((size_t)n_blocks + 64);
    int    *sel   = (int *)malloc((size_t)n_blocks * sizeof(int) + 64);
    /* DEBUG: canary the small scratch so an over-write is caught right here
     * (the LFH small-bucket free list was being corrupted during sparse
     * decode with evicted blocks; every other allocation is canaried). */
    uint8_t *tail_probe = (uint8_t *)probe + (size_t)n_blocks * sizeof(float);
    uint8_t *tail_used  = used + (size_t)n_blocks;
    uint8_t *tail_sel   = (uint8_t *)sel + (size_t)n_blocks * sizeof(int);
    uint8_t *tail_imp   = imp_sum ? (uint8_t *)imp_sum + (size_t)n_blocks * sizeof(float) : NULL;
    if (probe) memset(tail_probe, 0xA5, 64);
    if (used)  memset(tail_used, 0xA5, 64);
    if (sel)   memset(tail_sel, 0xA5, 64);
    if (imp_sum) memset(tail_imp, 0xA5, 64);
    if (!probe || !used || !sel || (importance && !imp_sum)) {
        free(probe); free(used); free(sel); free(imp_sum);
        return;   /* on OOM, leave attn_out zeroed by caller */
    }
    memset(used, 0, (size_t)n_blocks);

    /* (1) probe each block: max dot over n_probe evenly-spaced K samples
     * (float cache always kept). Multi-sample fusion substantially improves
     * needle recall vs the single first-token probe: a high-attention token
     * anywhere inside the block is detected even if the block head is dull.
     * Probe and prefill-importance are kept SEPARATE: the importance channel
     * is consumed as a rank-based pre-selection below (scale-free), because
     * blending it into the dot score is dwarfed by the probe magnitude
     * (softmax mass ~1e-3/token vs raw Q·K dot ~1e1). */
    if (n_probe < 1) n_probe = 1;
    for (int b = 0; b < n_blocks; b++) {
        float best = -1e30f;
        int p0 = b * bs;
        int p_end = p0 + bs; if (p_end > seq_len) p_end = seq_len;
        int step = (p_end - p0) / n_probe; if (step < 1) step = 1;
        for (int ps = p0; ps < p_end; ps += step) {
            float dot;
            if (k_cache[ps / bs]) {
                const float *kt = BLK(k_cache, ps) + (size_t)kh * hd;
                float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = vdupq_n_f32(0.0f);
                for (int i = 0; i < hd8; i += 8) {
                    acc0 = vfmaq_f32(acc0, vld1q_f32(qt + i), vld1q_f32(kt + i));
                    acc1 = vfmaq_f32(acc1, vld1q_f32(qt + i + 4), vld1q_f32(kt + i + 4));
                }
                float32x4_t hs = vaddq_f32(acc0, acc1);
                hs = vpaddq_f32(hs, hs);
                hs = vpaddq_f32(hs, hs);
                dot = vgetq_lane_f32(hs, 0);
                for (int i = hd8; i < hd; i++) dot += qt[i] * kt[i];
            } else if (l3) {
                /* Evicted block: zero-copy Q4 probe straight from the RAM
                 * mirror (no per-sample memcpy); falls back to the chunk
                 * read when the mirror is absent or out of range. */
                const uint8_t *lp = l3_payload_at(l3, L3BM(ps / bs), kh, ps % bs, 0,
                                                  hd, bs, kv_dim);
                if (lp) {
                    dot = 0.0f;
                    for (int g = 0; g < n_payloads; g++)
                        dot += l3_q4_dot64(lp + (size_t)g * L3_Q4_PAYLOAD64,
                                           qt + (size_t)g * 64);
                } else {
                    dot = l3_read_k_dot(l3, L3BM(ps / bs), kh, ps % bs,
                                        qt, hd, bs, kv_dim);
                }
            } else {
                dot = -1e30f;   /* evicted with no L3 state: never selected */
            }
            if (dot > best) best = dot;
        }
        probe[b] = best;
        if (importance)
            for (int ps = p0; ps < p_end; ps++) imp_sum[b] += importance[ps];
    }

    /* (1b) prefill-importance pre-selection: reserve half the block budget for
     * the blocks the exact prefill pass actually attended (rank-based, so a
     * needle block with modest but real attention mass is retained regardless
     * of probe magnitude). Deterministic: tie-break by block index. */
    int n_imp = 0;
    if (importance) {
        int budget = (k_blocks + 1) / 2;   /* min 1 */
        for (int i = 0; i < n_blocks && n_imp < budget; i++) {
            int best = -1;
            for (int j = 0; j < n_blocks; j++) {
                if (used[j]) continue;
                if (best < 0 || imp_sum[j] > imp_sum[best] ||
                    (imp_sum[j] == imp_sum[best] && j < best)) best = j;
            }
            if (best < 0) break;
            sel[n_imp++] = best;
            used[best] = 1;
        }
    }
    int nsel = n_imp;

    /* (2) deterministic top-k over the remaining blocks, tie-break by index */
    for (int i = nsel; i < n_blocks && nsel < k_blocks; i++) {
        int best = -1;
        for (int j = 0; j < n_blocks; j++) {
            if (used[j]) continue;
            if (best < 0 || probe[j] > probe[best] ||
                (probe[j] == probe[best] && j < best)) best = j;
        }
        if (best < 0) break;
        sel[nsel++] = best;
        used[best] = 1;
    }

    /* (2b) recency insurance: the most recent block is always kept. It may
     * only evict a probe-chosen block (sel[n_imp..nsel-1]); importance-
     * pre-selected blocks are never displaced. */
    if (nsel < n_blocks && !used[n_blocks - 1] && nsel > n_imp) {
        int low = n_imp;
        for (int i = n_imp + 1; i < nsel; i++)
            if (probe[sel[i]] < probe[sel[low]]) low = i;
        used[sel[low]] = 0;
        sel[low] = n_blocks - 1;
        used[n_blocks - 1] = 1;
    }

    /* (3) exact scores over selected positions only */
    float max_score = -1e9f;
    for (int si = 0; si < nsel; si++) {
        int b = sel[si];
        int p0 = b * bs;
        int p1 = p0 + bs; if (p1 > seq_len) p1 = seq_len;
        /* Phase-2c: stage the whole evicted block's K area (this head) in one
         * copy, then score every token from the staged buffer. */
        const uint8_t *kbuf = NULL;
        if (k_cache[b] == NULL && l3k_buf)
            if (l3_fetch_block_area(l3, L3BM(b), kh, 0, l3k_buf, hd, bs, kv_dim) == 0)
                kbuf = l3k_buf;
        for (int t = p0; t < p1; t++) {
            float dot;
            if (kbuf) {
                /* Evicted block served from the batch-staged K area. */
                g_l3_disk_hits++;
                const uint8_t *lp = kbuf + (size_t)(t - p0) * (size_t)n_payloads
                                          * L3_Q4_PAYLOAD64;
                dot = 0.0f;
                for (int g = 0; g < n_payloads; g++)
                    dot += l3_q4_dot64(lp + (size_t)g * L3_Q4_PAYLOAD64,
                                       qt + (size_t)g * 64);
            } else if (k_cache[t / bs] == NULL) {
                /* Evicted block: compressed-state Q4 Q·K from disk. */
                if (!l3) continue;
                g_l3_disk_hits++;
                dot = l3_read_k_dot(l3, L3BM(t / bs), kh, t % bs,
                                    qt, hd, bs, kv_dim);
            } else if (use_q8) {
                const int8_t *kt8 = BLK(k_q8, t) + (size_t)kh * hd;
                float ks = kscale[(size_t)t * (size_t)nkv + (size_t)kh] * KVQ_SCALE;
                float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = vdupq_n_f32(0.0f);
                for (int i = 0; i < hd8; i += 8) {
                    int8x8_t kv = vld1_s8(kt8 + i);
                    int16x8_t kv16 = vmovl_s8(kv);
                    float32x4_t kf0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(kv16)));
                    float32x4_t kf1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(kv16)));
                    acc0 = vfmaq_f32(acc0, vld1q_f32(qt + i), kf0);
                    acc1 = vfmaq_f32(acc1, vld1q_f32(qt + i + 4), kf1);
                }
                float32x4_t hs = vaddq_f32(acc0, acc1);
                hs = vpaddq_f32(hs, hs);
                hs = vpaddq_f32(hs, hs);
                dot = vgetq_lane_f32(hs, 0) * ks;
                for (int i = hd8; i < hd; i++) dot += qt[i] * (float)kt8[i] * ks;
            } else {
                const float *kt = BLK(k_cache, t) + (size_t)kh * hd;
                float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = vdupq_n_f32(0.0f);
                for (int i = 0; i < hd8; i += 8) {
                    acc0 = vfmaq_f32(acc0, vld1q_f32(qt + i), vld1q_f32(kt + i));
                    acc1 = vfmaq_f32(acc1, vld1q_f32(qt + i + 4), vld1q_f32(kt + i + 4));
                }
                float32x4_t hs = vaddq_f32(acc0, acc1);
                hs = vpaddq_f32(hs, hs);
                hs = vpaddq_f32(hs, hs);
                dot = vgetq_lane_f32(hs, 0);
                for (int i = hd8; i < hd; i++) dot += qt[i] * kt[i];
            }
            scores[t] = dot * scale;
            if (scores[t] > max_score) max_score = scores[t];
        }
    }

    /* (4) softmax over the selected set */
    float sum_exp = 0.0f;
    for (int si = 0; si < nsel; si++) {
        int p0 = sel[si] * bs;
        int p1 = p0 + bs; if (p1 > seq_len) p1 = seq_len;
        for (int t = p0; t < p1; t++) {
            scores[t] = expf(scores[t] - max_score);
            sum_exp += scores[t];
        }
    }

    /* (5) weighted V over selected positions */
    for (int i = 0; i < hd; i++) attn_out[i] = 0.0f;

    for (int si = 0; si < nsel; si++) {
        int b = sel[si];
        int p0 = b * bs;
        int p1 = p0 + bs; if (p1 > seq_len) p1 = seq_len;
        /* Phase-2c: stage the whole evicted block's V area (this head) in one
         * copy, then accumulate every token from the staged buffer. */
        const uint8_t *vbuf = NULL;
        if (v_cache[b] == NULL && l3v_buf)
            if (l3_fetch_block_area(l3, L3BM(b), kh, 1, l3v_buf, hd, bs, kv_dim) == 0)
                vbuf = l3v_buf;
        for (int t = p0; t < p1; t++) {
            float wgt = scores[t] / sum_exp;
            if (vbuf) {
                /* Evicted block served from the batch-staged V area. */
                const uint8_t *lp = vbuf + (size_t)(t - p0) * (size_t)n_payloads
                                           * L3_Q4_PAYLOAD64;
                for (int g = 0; g < n_payloads; g++)
                    l3_q4_vacc64(attn_out + (size_t)g * 64,
                                 lp + (size_t)g * L3_Q4_PAYLOAD64, wgt);
            } else if (v_cache[t / bs] == NULL) {
                /* Evicted block: compressed-state V accumulation from disk. */
                if (l3)
                    l3_read_v_acc(l3, L3BM(t / bs), kh, t % bs, wgt,
                                  attn_out, hd, bs, kv_dim);
            } else if (use_q8) {
                const int8_t *vt8 = BLK(v_q8, t) + (size_t)kh * hd;
                float vs = vscale[(size_t)t * (size_t)nkv + (size_t)kh] * KVQ_SCALE;
                float32x4_t wv = vdupq_n_f32(wgt * vs);
                for (int i = 0; i < hd8; i += 8) {
                    int8x8_t vv = vld1_s8(vt8 + i);
                    int16x8_t vv16 = vmovl_s8(vv);
                    float32x4_t vf0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vv16)));
                    float32x4_t vf1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vv16)));
                    float32x4_t av0 = vld1q_f32(attn_out + i);
                    float32x4_t av1 = vld1q_f32(attn_out + i + 4);
                    vst1q_f32(attn_out + i, vfmaq_f32(av0, vf0, wv));
                    vst1q_f32(attn_out + i + 4, vfmaq_f32(av1, vf1, wv));
                }
                for (int i = hd8; i < hd; i++)
                    attn_out[i] += wgt * (float)vt8[i] * vs;
            } else {
                const float *vt = BLK(v_cache, t) + (size_t)kh * hd;
                float32x4_t wv = vdupq_n_f32(wgt);
                for (int i = 0; i < hd8; i += 8) {
                    float32x4_t av0 = vld1q_f32(attn_out + i);
                    float32x4_t av1 = vld1q_f32(attn_out + i + 4);
                    vst1q_f32(attn_out + i, vfmaq_f32(av0, vld1q_f32(vt + i), wv));
                    vst1q_f32(attn_out + i + 4, vfmaq_f32(av1, vld1q_f32(vt + i + 4), wv));
                }
                for (int i = hd8; i < hd; i++) attn_out[i] += wgt * vt[i];
            }
        }
    }

    {   /* DEBUG: verify the scratch canaries; a hit localizes the over-writer */
        for (int _cn_i = 0; _cn_i < 64; _cn_i++)
            if (tail_probe[_cn_i] != 0xA5) { fprintf(stderr, "[SP-CN] probe +%d l=%d\n", _cn_i, layer); break; }
        for (int _cn_i = 0; _cn_i < 64; _cn_i++)
            if (tail_used[_cn_i] != 0xA5) { fprintf(stderr, "[SP-CN] used +%d l=%d\n", _cn_i, layer); break; }
        for (int _cn_i = 0; _cn_i < 64; _cn_i++)
            if (tail_sel[_cn_i] != 0xA5) { fprintf(stderr, "[SP-CN] sel +%d l=%d\n", _cn_i, layer); break; }
        if (tail_imp)
            for (int _cn_i = 0; _cn_i < 64; _cn_i++)
                if (tail_imp[_cn_i] != 0xA5) { fprintf(stderr, "[SP-CN] imp_sum +%d l=%d\n", _cn_i, layer); break; }
    }

    free(probe); free(used); free(sel); free(imp_sum);
    free(l3k_buf); free(l3v_buf);
#undef L3BM
#undef BLK
}

/* Sparse attention Phase-1 self-test: verifies that selecting ALL blocks
 * reproduces exact attention (float + INT8-KV paths), selection is
 * deterministic (gumbel red line), and dropping blocks changes the result
 * (negative test). */

/* forward declarations (the packed kernels are defined later in this file) */
static void st_attn_batched_packed(
    float *restrict attn_out, const float *restrict q_buf,
    const float *restrict k_pack, const float *restrict v_pack,
    int nb, int prev_len, int seq_stride,
    int nh, int nkv, int hd, float scale,
    float *restrict scores, int score_stride,
    float *restrict imp_head);

static void st_attn_batched_packed_sparse(
    float *restrict attn_out, const float *restrict q_buf,
    const float *restrict k_pack, const float *restrict v_pack,
    int nb, int prev_len, int seq_stride,
    int nh, int nkv, int hd, float scale,
    float *restrict scores, int score_stride,
    int bs, int k_blocks, int n_probe,
    float *restrict imp_head);

void st_test_sparse_attn(void) {
    printf("\n=== Sparse Attention Phase-1 self-test ===\n");
    const int hd = 128, kv_dim = 1024, nkv = 8, nh = 8, seq_len = 256, bs = 32;
    float scale = 1.0f / sqrtf((float)hd);
    uint32_t rng = 0x12345678u;
    int fail = 0;

    float *k_cache = (float *)malloc((size_t)seq_len * kv_dim * sizeof(float));
    float *v_cache = (float *)malloc((size_t)seq_len * kv_dim * sizeof(float));
    float *q = (float *)malloc((size_t)nh * hd * sizeof(float));
    float *ref = (float *)malloc((size_t)nh * hd * sizeof(float));
    float *got = (float *)malloc((size_t)nh * hd * sizeof(float));
    float *got2 = (float *)malloc((size_t)nh * hd * sizeof(float));
    float *scores = (float *)malloc((size_t)seq_len * sizeof(float));
    int8_t *kq8 = (int8_t *)malloc((size_t)seq_len * kv_dim);
    int8_t *vq8 = (int8_t *)malloc((size_t)seq_len * kv_dim);
    float *ksc = (float *)malloc((size_t)seq_len * nkv * sizeof(float));
    float *vsc = (float *)malloc((size_t)seq_len * nkv * sizeof(float));
    if (!k_cache || !v_cache || !q || !ref || !got || !got2 || !scores || !kq8 || !vq8 || !ksc || !vsc) {
        printf("  [FAIL] OOM\n");
        goto test_free;
    }
    /* Phase 2b: block views over the contiguous test buffers so the
     * sparse kernel sees the same per-block layout as the real engine. */
    int n_blocks = seq_len / bs;
    float  **k_blocks  = (float  **)malloc((size_t)n_blocks * sizeof(float  *));
    float  **v_blocks  = (float  **)malloc((size_t)n_blocks * sizeof(float  *));
    int8_t **kq8_blocks = (int8_t **)malloc((size_t)n_blocks * sizeof(int8_t *));
    int8_t **vq8_blocks = (int8_t **)malloc((size_t)n_blocks * sizeof(int8_t *));
    if (!k_blocks || !v_blocks || !kq8_blocks || !vq8_blocks) {
        printf("  [FAIL] OOM (block views)\n");
        goto test_free;
    }
    for (int b = 0; b < n_blocks; b++) {
        k_blocks[b]  = k_cache + (size_t)b * bs * kv_dim;
        v_blocks[b]  = v_cache + (size_t)b * bs * kv_dim;
        kq8_blocks[b] = kq8 + (size_t)b * bs * kv_dim;
        vq8_blocks[b] = vq8 + (size_t)b * bs * kv_dim;
    }
    for (int i = 0; i < seq_len * kv_dim; i++) {
        rng = rng * 1664525u + 1013904223u;
        k_cache[i] = (((float)((rng >> 8) & 0xFFFF)) / 65535.0f - 0.5f) * 3.0f;
        v_cache[i] = k_cache[i] * 0.5f;
    }
    for (int i = 0; i < nh * hd; i++) {
        rng = rng * 1664525u + 1013904223u;
        q[i] = (((float)((rng >> 8) & 0xFFFF)) / 65535.0f - 0.5f) * 3.0f;
    }
    kv_quantize_per_head(kq8, vq8, ksc, vsc, k_cache, v_cache, nkv, hd);

    /* exact float reference per head */
    for (int h = 0; h < nh; h++) {
        int kh = (h * nkv) / nh;
        const float *qt = q + (size_t)h * hd;
        float maxs = -1e9f;
        for (int t = 0; t < seq_len; t++) {
            const float *kt = k_cache + (size_t)t * kv_dim + (size_t)kh * hd;
            float dot = 0.0f;
            for (int i = 0; i < hd; i++) dot += qt[i] * kt[i];
            scores[t] = dot * scale;
            if (scores[t] > maxs) maxs = scores[t];
        }
        float sume = 0.0f;
        for (int t = 0; t < seq_len; t++) { scores[t] = expf(scores[t] - maxs); sume += scores[t]; }
        float *ao = ref + (size_t)h * hd;
        for (int i = 0; i < hd; i++) ao[i] = 0.0f;
        for (int t = 0; t < seq_len; t++) {
            float w = scores[t] / sume;
            const float *vt = v_cache + (size_t)t * kv_dim + (size_t)kh * hd;
            for (int i = 0; i < hd; i++) ao[i] += w * vt[i];
        }
    }

    /* (1) all blocks selected (float path) must match exact */
    for (int h = 0; h < nh; h++) {
        int kh = (h * nkv) / nh;
        sparse_attn_head(got + (size_t)h * hd, q + (size_t)h * hd,
                         k_blocks, v_blocks, NULL, NULL, NULL, NULL,
                         0, seq_len, kv_dim, nkv, kh, hd, scale, bs, 9999, 4, NULL, scores, NULL, 0);
    }
    float maxd = 0.0f;
    for (int i = 0; i < nh * hd; i++) {
        float d = fabsf(got[i] - ref[i]);
        if (d > maxd) maxd = d;
    }
    printf("  [%s] sparse(all blocks) == exact float (max diff %.3e)\n",
           maxd < 1e-4f ? "PASS" : "FAIL", maxd);
    if (maxd >= 1e-4f) fail++;

    /* (2) deterministic: same input twice -> bitwise identical */
    for (int h = 0; h < nh; h++) {
        int kh = (h * nkv) / nh;
        sparse_attn_head(got + (size_t)h * hd, q + (size_t)h * hd,
                         k_blocks, v_blocks, NULL, NULL, NULL, NULL,
                         0, seq_len, kv_dim, nkv, kh, hd, scale, bs, 4, 4, NULL, scores, NULL, 0);
        sparse_attn_head(got2 + (size_t)h * hd, q + (size_t)h * hd,
                         k_blocks, v_blocks, NULL, NULL, NULL, NULL,
                         0, seq_len, kv_dim, nkv, kh, hd, scale, bs, 4, 4, NULL, scores, NULL, 0);
    }
    int det = 1;
    for (int i = 0; i < nh * hd; i++)
        if (got[i] != got2[i]) { det = 0; break; }
    printf("  [%s] sparse top-4 deterministic (bitwise)\n", det ? "PASS" : "FAIL");
    if (!det) fail++;

    /* (3) negative: dropping blocks must change the result */
    float maxd2 = 0.0f;
    for (int i = 0; i < nh * hd; i++) {
        float d = fabsf(got[i] - ref[i]);
        if (d > maxd2) maxd2 = d;
    }
    printf("  [%s] sparse top-4 differs from exact (negative, max diff %.3e)\n",
           maxd2 > 1e-3f ? "PASS" : "FAIL", maxd2);
    if (maxd2 <= 1e-3f) fail++;

    /* (4) INT8-KV path: all blocks selected must match the q8 exact reference */
    for (int h = 0; h < nh; h++) {
        int kh = (h * nkv) / nh;
        const float *qt = q + (size_t)h * hd;
        float maxs = -1e9f;
        for (int t = 0; t < seq_len; t++) {
            const int8_t *kt8 = kq8 + (size_t)t * kv_dim + (size_t)kh * hd;
            float ks = ksc[(size_t)t * nkv] * KVQ_SCALE;
            float dot = 0.0f;
            for (int i = 0; i < hd; i++) dot += qt[i] * (float)kt8[i] * ks;
            scores[t] = dot * scale;
            if (scores[t] > maxs) maxs = scores[t];
        }
        float sume = 0.0f;
        for (int t = 0; t < seq_len; t++) { scores[t] = expf(scores[t] - maxs); sume += scores[t]; }
        float *ao = ref + (size_t)h * hd;
        for (int i = 0; i < hd; i++) ao[i] = 0.0f;
        for (int t = 0; t < seq_len; t++) {
            float w = scores[t] / sume;
            const int8_t *vt8 = vq8 + (size_t)t * kv_dim + (size_t)kh * hd;
            float vs = vsc[(size_t)t * nkv] * KVQ_SCALE;
            for (int i = 0; i < hd; i++) ao[i] += w * (float)vt8[i] * vs;
        }
    }
    for (int h = 0; h < nh; h++) {
        int kh = (h * nkv) / nh;
        sparse_attn_head(got + (size_t)h * hd, q + (size_t)h * hd,
                         k_blocks, v_blocks, kq8_blocks, vq8_blocks, ksc, vsc,
                         1, seq_len, kv_dim, nkv, kh, hd, scale, bs, 9999, 4, NULL, scores, NULL, 0);
    }
    maxd = 0.0f;
    for (int i = 0; i < nh * hd; i++) {
        float d = fabsf(got[i] - ref[i]);
        if (d > maxd) maxd = d;
    }
    printf("  [%s] sparse(all blocks) == exact INT8-KV (max diff %.3e)\n",
           maxd < 1e-4f ? "PASS" : "FAIL", maxd);
    if (maxd >= 1e-4f) fail++;

    /* (5) prefill packed path: sparse(all blocks) == exact packed attention */
    {
        int nb = 8, prev_len = seq_len - nb;
        int seq_stride = seq_len;
        float *k_pack = (float *)malloc((size_t)nkv * seq_stride * hd * sizeof(float));
        float *v_pack = (float *)malloc((size_t)nkv * seq_stride * hd * sizeof(float));
        float *q_buf = (float *)malloc((size_t)nb * nh * hd * sizeof(float));
        float *att_ref = (float *)malloc((size_t)nb * nh * hd * sizeof(float));
        float *att_sp = (float *)malloc((size_t)nb * nh * hd * sizeof(float));
        float *att_sp2 = (float *)malloc((size_t)nb * nh * hd * sizeof(float));
        float *sc_buf = (float *)malloc((size_t)nh * seq_stride * sizeof(float));
        if (k_pack && v_pack && q_buf && att_ref && att_sp && att_sp2 && sc_buf) {
            for (int kh = 0; kh < nkv; kh++)
                for (int s = 0; s < seq_stride; s++)
                    for (int i = 0; i < hd; i++) {
                        k_pack[((size_t)kh * seq_stride + s) * hd + i] =
                            k_cache[(size_t)s * kv_dim + (size_t)kh * hd + i];
                        v_pack[((size_t)kh * seq_stride + s) * hd + i] =
                            v_cache[(size_t)s * kv_dim + (size_t)kh * hd + i];
                    }
            for (int t = 0; t < nb; t++)
                for (int h = 0; h < nh; h++)
                    for (int i = 0; i < hd; i++)
                        q_buf[((size_t)t * nh + h) * hd + i] =
                            q[(size_t)h * hd + i] * (1.0f + 0.01f * (float)t);

            st_attn_batched_packed(att_ref, q_buf, k_pack, v_pack, nb, prev_len,
                                   seq_stride, nh, nkv, hd, scale, sc_buf, seq_stride, NULL);
            st_attn_batched_packed_sparse(att_sp, q_buf, k_pack, v_pack, nb, prev_len,
                                          seq_stride, nh, nkv, hd, scale, sc_buf,
                                          seq_stride, bs, 9999, 4, NULL);
            maxd = 0.0f;
            for (int i = 0; i < nb * nh * hd; i++) {
                float d = fabsf(att_sp[i] - att_ref[i]);
                if (d > maxd) maxd = d;
            }
            printf("  [%s] prefill sparse(all blocks) == exact packed (max diff %.3e)\n",
                   maxd < 1e-4f ? "PASS" : "FAIL", maxd);
            if (maxd >= 1e-4f) fail++;

            /* deterministic + negative on top-4 */
            st_attn_batched_packed_sparse(att_sp, q_buf, k_pack, v_pack, nb, prev_len,
                                          seq_stride, nh, nkv, hd, scale, sc_buf,
                                          seq_stride, bs, 4, 4, NULL);
            st_attn_batched_packed_sparse(att_sp2, q_buf, k_pack, v_pack, nb, prev_len,
                                          seq_stride, nh, nkv, hd, scale, sc_buf,
                                          seq_stride, bs, 4, 4, NULL);
            det = 1;
            for (int i = 0; i < nb * nh * hd; i++)
                if (att_sp[i] != att_sp2[i]) { det = 0; break; }
            printf("  [%s] prefill sparse top-4 deterministic (bitwise)\n", det ? "PASS" : "FAIL");
            if (!det) fail++;

            maxd2 = 0.0f;
            for (int i = 0; i < nb * nh * hd; i++) {
                float d = fabsf(att_sp[i] - att_ref[i]);
                if (d > maxd2) maxd2 = d;
            }
            printf("  [%s] prefill sparse top-4 differs from exact (negative, max diff %.3e)\n",
                   maxd2 > 1e-3f ? "PASS" : "FAIL", maxd2);
            if (maxd2 <= 1e-3f) fail++;
        } else {
            printf("  [FAIL] prefill sparse test OOM\n");
            fail++;
        }
        free(k_pack); free(v_pack); free(q_buf);
        free(att_ref); free(att_sp); free(att_sp2); free(sc_buf);
    }

    /* (6) prefill-importance rescues a needle inside a low-probe block:
     * crafted K/V with the needle at pos 100 (block 3, bs=32) whose probed
     * offsets (96/104/112/120) are all dull. Without importance the needle
     * block is dropped -> output ~0; with an importance signal at the needle
     * the block is pre-selected -> output ~V[needle]=5. The spike is tiny
     * (0.01, far below any raw Q·K dot) to prove the selection is rank-based,
     * not magnitude-based. */
    {
        const int needle = 100;
        float *q2 = (float *)malloc((size_t)hd * sizeof(float));
        float *k2 = (float *)malloc((size_t)seq_len * kv_dim * sizeof(float));
        float *v2 = (float *)malloc((size_t)seq_len * kv_dim * sizeof(float));
        float *imp = (float *)calloc((size_t)seq_len, sizeof(float));
        float *o_no = (float *)malloc((size_t)hd * sizeof(float));
        float *o_imp = (float *)malloc((size_t)hd * sizeof(float));
        float  **k2_blocks = (float  **)malloc((size_t)n_blocks * sizeof(float  *));
        float  **v2_blocks = (float  **)malloc((size_t)n_blocks * sizeof(float  *));
        if (q2 && k2 && v2 && imp && o_no && o_imp && k2_blocks && v2_blocks) {
            for (int b = 0; b < n_blocks; b++) {
                k2_blocks[b] = k2 + (size_t)b * bs * kv_dim;
                v2_blocks[b] = v2 + (size_t)b * bs * kv_dim;
            }
            for (int i = 0; i < hd; i++) q2[i] = 1.0f;
            for (int t = 0; t < seq_len; t++)
                for (int i = 0; i < hd; i++) {
                    k2[(size_t)t * kv_dim + i] = (t == needle) ? 10.0f : -0.1f;
                    v2[(size_t)t * kv_dim + i] = (t == needle) ? 5.0f : 0.0f;
                }
            imp[needle] = 0.01f;

            sparse_attn_head(o_no, q2, k2_blocks, v2_blocks, NULL, NULL, NULL, NULL,
                             0, seq_len, kv_dim, nkv, 0, hd, scale, bs, 2, 4, NULL, scores, NULL, 0);
            sparse_attn_head(o_imp, q2, k2_blocks, v2_blocks, NULL, NULL, NULL, NULL,
                             0, seq_len, kv_dim, nkv, 0, hd, scale, bs, 2, 4, imp, scores, NULL, 0);

            float m_no = 0.0f, m_imp = 0.0f;
            for (int i = 0; i < hd; i++) {
                float a = fabsf(o_no[i]);
                float b = fabsf(o_imp[i]);
                if (a > m_no) m_no = a;
                if (b > m_imp) m_imp = b;
            }
            printf("  [%s] importance rescues needle in low-probe block "
                   "(out %.3f -> %.3f)\n",
                   (m_no < 0.1f && m_imp > 4.0f) ? "PASS" : "FAIL", m_no, m_imp);
            if (!(m_no < 0.1f && m_imp > 4.0f)) fail++;
        } else {
            printf("  [FAIL] importance-rescue test OOM\n");
            fail++;
        }
        free(q2); free(k2); free(v2); free(imp); free(o_no); free(o_imp);
        free(k2_blocks); free(v2_blocks);
    }

    /* (7) L3 disk path: evicted blocks (RAM set NULL) served from the Q4
     * disk state must match the all-in-RAM exact attention within the
     * quantization bound (Q4 scale = amax/7, err <= scale/2 ~ 0.22 here). */
    {
        const char *path = "kv_l3_st_sparse.bin";
        float *imp2 = (float *)calloc((size_t)seq_len, sizeof(float));
        float *ref2 = (float *)malloc((size_t)nh * hd * sizeof(float));
        float *got3 = (float *)malloc((size_t)nh * hd * sizeof(float));
        float **ke = (float **)malloc((size_t)n_blocks * sizeof(float *));
        float **ve = (float **)malloc((size_t)n_blocks * sizeof(float *));
        STL3State l3;
        int ok = imp2 && ref2 && got3 && ke && ve;
        if (ok) {
            memcpy(ke, k_blocks, (size_t)n_blocks * sizeof(float *));
            memcpy(ve, v_blocks, (size_t)n_blocks * sizeof(float *));
            /* reference: exact attention with all blocks in RAM */
            for (int h = 0; h < nh; h++) {
                int kh = (h * nkv) / nh;
                sparse_attn_head(ref2 + (size_t)h * hd, q + (size_t)h * hd,
                                 ke, ve, NULL, NULL, NULL, NULL,
                                 0, seq_len, kv_dim, nkv, kh, hd, scale,
                                 bs, 9999, 4, NULL, scores, NULL, 0);
            }
            if (l3_state_init(&l3, path, 1, nkv, hd, bs, n_blocks) == 0) {
                l3_evict_layer(&l3, 0, ke, ve, kv_dim, seq_len, imp2, 0.75f, bs);
                for (int b = 0; b < n_blocks; b++)
                    if (l3.blocks[b].on_disk) ke[b] = NULL;   /* simulate RAM free */
                for (int h = 0; h < nh; h++) {
                    int kh = (h * nkv) / nh;
                    sparse_attn_head(got3 + (size_t)h * hd, q + (size_t)h * hd,
                                     ke, ve, NULL, NULL, NULL, NULL,
                                     0, seq_len, kv_dim, nkv, kh, hd, scale,
                                     bs, 9999, 4, NULL, scores, &l3, 0);
                }
                float m = 0.0f;
                for (int i = 0; i < nh * hd; i++) {
                    float d = fabsf(got3[i] - ref2[i]);
                    if (d > m) m = d;
                }
                printf("  [%s] L3 disk-served evicted blocks match RAM exact (max diff %.4f)\n",
                       m < 0.5f ? "PASS" : "FAIL", m);
                if (m >= 0.5f) fail++;
                l3_state_free(&l3);
            } else {
                printf("  [FAIL] L3 disk path: cannot init l3 state\n");
                fail++;
            }
        } else {
            printf("  [FAIL] L3 disk path OOM\n");
            fail++;
        }
        free(imp2); free(ref2); free(got3); free(ke); free(ve);
        remove(path);
    }

    printf("=== Sparse self-test %s (%d checks) ===\n",
           fail == 0 ? "PASSED" : "FAILED", 9);

test_free:
    free(k_cache); free(v_cache); free(q); free(ref); free(got); free(got2);
    free(scores); free(kq8); free(vq8); free(ksc); free(vsc);
    free(k_blocks); free(v_blocks); free(kq8_blocks); free(vq8_blocks);
}

/* W4: capture head-0 kernel scores for the DUMP cross-check. */
static float g_dbg_scores0[64];
static float g_dbg_sumexp0 = 0.0f;
static int    g_dbg_seq0 = 0;

typedef struct {
    STQwenInferenceState *st;
    int nh, nkv, hd, l, seq_len, kv_dim, use_q8;
    float scale;
} vllm_attn_head_ctx;

static void vllm_attn_head_worker(void *ctx_, int h) {
    vllm_attn_head_ctx *c = ctx_;
    STQwenInferenceState *st = c->st;
    int kh = (h * c->nkv) / c->nh;
    /* Indexed by absolute token position [0, seq_len); with max_seq=8192
     * it must span the whole context. The old fixed 2048 cap overflowed the
     * stack ("stack smashing") as soon as a request exceeded 2K tokens (e.g.
     * 4K). 8K/16K contexts exceed the 8192 stack array too, so allocate
     * dynamically from the context length (bench-seqlen up to 16K). */
    float *scores = (float *)malloc((size_t)(c->seq_len > 8192 ? c->seq_len : 8192) * sizeof(float));
    if (!scores) {
        fprintf(stderr, "[ATTN] OOM: scores[%d]\n", c->seq_len);
        return;
    }
    float max_score = -1e9f;
    const float *qt = st->q_buf + (size_t)h * c->hd;
    int hd8 = c->hd & ~7;

    if (g_sparse_attn && !st->use_kv_q4 && c->seq_len > g_sparse_block * 2) {
        /* Phase 1.5/2b: block-probe + prefill-importance
         * potential + top-k selection + bounded attention.
         * Evicted blocks (RAM NULL) are served from the L3
         * disk state in compressed Q4 form. */
        sparse_attn_head(
            st->attn_buf + (size_t)h * c->hd, qt,
            st->k_cache[c->l], st->v_cache[c->l],
            st->k_cache_q8[c->l], st->v_cache_q8[c->l],
            st->k_scale[c->l], st->v_scale[c->l],
            c->use_q8, c->seq_len, c->kv_dim, c->nkv, kh, c->hd, c->scale,
            g_sparse_block, g_sparse_k, g_sparse_probe,
            st->prefill_importance, scores,
            &st->l3, c->l);
        free(scores);
        return;
    }

    if (c->use_q8) {
        /* INT8-KV flash attention（online softmax + vdot + KV 分块语义）。 */
        flash_attn_single_q_q8_neon(
            st->attn_buf + (size_t)h * c->hd, qt,
            st->k_cache_q8[c->l], st->v_cache_q8[c->l],
            st->k_scale[c->l] + kh, st->v_scale[c->l] + kh,
            c->seq_len, c->kv_dim, st->kv_bs, c->nkv, kh * c->hd, c->hd, c->scale);
        free(scores);
        return;
    } else if (st->use_kv_q4) {
        /* Q4 payloads: compressed dot, no dequant */
        for (int t = 0; t < c->seq_len; t++) {
            const uint8_t *kp = q4_k_row(st->k_cache_q4[c->l], t, kh, c->hd, st->kv_bs, c->nkv);
            float dot = q4_head_dot(kp, qt, c->hd);
            scores[t] = dot * c->scale;
            if (scores[t] > max_score) max_score = scores[t];
        }
    } else {
        for (int t = 0; t < c->seq_len; t++) {
            const float *kt = kv_row_f32(st->k_cache[c->l], t, c->kv_dim, st->kv_bs) + (size_t)kh * c->hd;
            float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = vdupq_n_f32(0.0f);
            for (int i = 0; i < hd8; i += 8) {
                acc0 = vfmaq_f32(acc0, vld1q_f32(qt + i), vld1q_f32(kt + i));
                acc1 = vfmaq_f32(acc1, vld1q_f32(qt + i + 4), vld1q_f32(kt + i + 4));
            }
            float32x4_t hs = vaddq_f32(acc0, acc1);
            hs = vpaddq_f32(hs, hs);
            hs = vpaddq_f32(hs, hs);
            float dot = vgetq_lane_f32(hs, 0);
            for (int i = hd8; i < c->hd; i++) dot += qt[i] * kt[i];
            scores[t] = dot * c->scale;
            if (scores[t] > max_score) max_score = scores[t];
        }
    }

    /* Softmax + weighted sum */
    float sum_exp = 0.0f;
    for (int t = 0; t < c->seq_len; t++) {
        scores[t] = expf(scores[t] - max_score);
        sum_exp += scores[t];
    }
    /* W4: capture head-0 post-softmax scores (kernel truth). */
    if (h == 0) {
        for (int _t = 0; _t < c->seq_len && _t < 64; _t++)
            g_dbg_scores0[_t] = scores[_t];
        g_dbg_sumexp0 = sum_exp;
        g_dbg_seq0 = c->seq_len;
    }

    /* Zero attention buffer */
    float *attn_out = st->attn_buf + (size_t)h * c->hd;
    for (int i = 0; i < c->hd; i++) attn_out[i] = 0.0f;

    if (c->use_q8) {
        const float *vsc = st->v_scale[c->l] + kh;
        for (int t = 0; t < c->seq_len; t++) {
            float wgt = scores[t] / sum_exp;
            const int8_t *vt = kv_row_i8(st->v_cache_q8[c->l], t, c->kv_dim, st->kv_bs) + (size_t)kh * c->hd;
            float vs = vsc[(size_t)t * c->nkv] * KVQ_SCALE;
            float32x4_t wv = vdupq_n_f32(wgt * vs);
            for (int i = 0; i < hd8; i += 8) {
                int8x8_t vv = vld1_s8(vt + i);
                int16x8_t vv16 = vmovl_s8(vv);
                float32x4_t vf0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vv16)));
                float32x4_t vf1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vv16)));
                float32x4_t av0 = vld1q_f32(attn_out + i);
                float32x4_t av1 = vld1q_f32(attn_out + i + 4);
                vst1q_f32(attn_out + i, vfmaq_f32(av0, vf0, wv));
                vst1q_f32(attn_out + i + 4, vfmaq_f32(av1, vf1, wv));
            }
            for (int i = hd8; i < c->hd; i++)
                attn_out[i] += wgt * (float)vt[i] * vs;
        }
    } else if (st->use_kv_q4) {
        /* Q4 payloads: wgt * dequant(V) accumulated in-place */
        for (int t = 0; t < c->seq_len; t++) {
            float wgt = scores[t] / sum_exp;
            const uint8_t *vp = q4_v_row(st->v_cache_q4[c->l], t, kh, c->hd, st->kv_bs, c->nkv);
            q4_head_vacc(attn_out, vp, wgt, c->hd);
        }
    } else {
        for (int t = 0; t < c->seq_len; t++) {
            float wgt = scores[t] / sum_exp;
            const float *vt = kv_row_f32(st->v_cache[c->l], t, c->kv_dim, st->kv_bs) + (size_t)kh * c->hd;
            float32x4_t wv = vdupq_n_f32(wgt);
            for (int i = 0; i < hd8; i += 8) {
                float32x4_t av0 = vld1q_f32(attn_out + i);
                float32x4_t av1 = vld1q_f32(attn_out + i + 4);
                vst1q_f32(attn_out + i, vfmaq_f32(av0, vld1q_f32(vt + i), wv));
                vst1q_f32(attn_out + i + 4, vfmaq_f32(av1, vld1q_f32(vt + i + 4), wv));
            }
            for (int i = hd8; i < c->hd; i++)
                attn_out[i] += wgt * vt[i];
        }
    }
    free(scores);
}

typedef struct {
    float *out; const float *a, *b; int n;
} vllm_add2_ctx;

static void vllm_add2_worker(void *ctx_, int i) {
    vllm_add2_ctx *c = ctx_;
    c->out[i] = c->a[i] + c->b[i];
}

typedef struct {
    float *gate, *up; int n;
} vllm_silu_mul_ctx;

/* M4i: SiLU NEON 向量化（2026-08-29）。原标量 fast_silu 每元素一次 expf，
 * prefill 28 层 × 811K 元素 ≈ 22.7M 次 expf 占 OTHER ~70ms。改为每任务 16
 * 元素、4 宽 exp_neon4（多项式 exp，|err|~1e-7）+ Newton 倒数 1 步。
 * 数值：±20 截断语义由 exp 夹紧（exp(±20)≈2e-9/4.8e9）自然保持，silu 值
 * 相对误差 ~1e-7 -> 位级放宽，PPL/文本口径验收（同 M4c/M4f 先例）。
 * 任务粒度 16：parfor 静态分区仍覆盖全部元素，尾部标量兜底。 */
static void vllm_silu_mul_worker(void *ctx_, int it) {
    vllm_silu_mul_ctx *c = ctx_;
    int i = it * 16;
    int n = c->n - i;
    if (n > 16) n = 16;
    float *g = c->gate + i;
    const float *u = c->up + i;
#if ST_HAVE_NEON
    int nv = n & ~3;
    for (int k = 0; k < nv; k += 4) {
        float32x4_t gx = vld1q_f32(g + k);
        float32x4_t ux = vld1q_f32(u + k);
        float32x4_t ex = exp_neon4(vmulq_n_f32(gx, -1.0f));
        float32x4_t den = vaddq_f32(ex, vdupq_n_f32(1.0f));
        float32x4_t rec = vrecpeq_f32(den);
        rec = vmulq_f32(rec, vrecpsq_f32(den, rec));   /* 2x Newton -> ~2^-25 */
        rec = vmulq_f32(rec, vrecpsq_f32(den, rec));
        vst1q_f32(g + k, vmulq_f32(vmulq_f32(gx, rec), ux));
    }
    for (int k = nv; k < n; k++) g[k] = fast_silu(g[k]) * u[k];
#else
    for (int k = 0; k < n; k++) g[k] = fast_silu(g[k]) * u[k];
#endif
}

void st_qwen_model_forward(STQwenInferenceState *st, int token_id) {
    STModelWeights *w = &st->weights;
    int d  = st->cfg.dim;
    int nl = w->n_layers_allocated;
    int nh = st->cfg.n_heads;
    int nkv = st->cfg.n_kv_heads;
    int hd = st->cfg.head_dim;
    int ff = st->cfg.ffn_dim;
    int vc = st->cfg.vocab_size;
    float eps = st->cfg.norm_eps;
    float theta = st->cfg.rope_theta;
    int kv_dim_q = nh * hd;  /* 4096 */
    int kv_dim   = nkv * hd; /* 1024 */
    /* MRoPE text position: multimodal prompts advance text positions by
     * max(grid_h,grid_w) per visual region (NOT the visual-token count), so
     * seq_len drifts from the MRoPE position. Continue from prefill_ex's
     * stored mrope_pos; text-only prompts keep 0 and use seq_len. */
    int pos = st->mrope_pos > 0 ? st->mrope_pos : st->seq_len;

    int prof = st->profile_decode;
    if (!prof && getenv("VLLM_DEC_PROF")) prof = 1;   /* serve 模式诊断开关 */
    double t_all0 = 0.0;
    double t_qkv = 0.0, t_attn = 0.0, t_o = 0.0, t_gu = 0.0, t_down = 0.0, t_lm = 0.0;
    if (prof) t_all0 = st_now_sec();

    float *residual = st->ffn_out_buf;
    float *normed  = st->ffn_buf;

    /* EMBEDDING */
    float *x = st->hidden;
    float *emb = w->token_embed;
    if (!emb) return;

    for (int i = 0; i < d; i++)
        x[i] = st_emb_get(w, (size_t)token_id, (size_t)i, d);

    for (int l = 0; l < nl; l++) {
        if (prof)
            fprintf(stderr, "[DDEC] fwd l=%d/%d seq=%d mrope=%d\n",
                    l, nl, st->seq_len, st->mrope_pos);
        g_npu_layer = l;   /* decode offload (VLLM_NPU_LOAD=2): per-layer model */
        /* --- Attention --- */
        memcpy(residual, x, d * sizeof(float));
        dyn_rms_norm(normed, x, w->attn_norm + l * d, d, eps);

        /* Q/K/V projections: fused matvec (axiom: block_matrix_assoc_natural)
         * Single pass through normed[] produces Q, K, V simultaneously.
         * Saves 2 re-reads of 16 KB input vector per layer (~1.1 MB total). */
        double t0 = 0.0;
        if (prof) t0 = st_now_sec();
        if (w->has_q4 && w->q4_q_weight) {
            dyn_matvec_q4_q8_fused_qkv_batched(st->q_buf, st->k_buf, st->v_buf,
                w->q4_q_weight + Q4_BYTES((size_t)l * d * kv_dim_q),
                w->q4_k_weight + Q4_BYTES((size_t)l * d * kv_dim),
                w->q4_v_weight + Q4_BYTES((size_t)l * d * kv_dim),
                normed, kv_dim_q, kv_dim, d, 1,
                w->has_x8 ? w->x8_q_weight + Q4_BYTES((size_t)l * d * kv_dim_q) : NULL,
                w->has_x8 ? w->x8_k_weight + Q4_BYTES((size_t)l * d * kv_dim) : NULL,
                w->has_x8 ? w->x8_v_weight + Q4_BYTES((size_t)l * d * kv_dim) : NULL);
        } else if (w->has_q8 && w->q8_q_weight) {
            dyn_matvec_q8_fused_qkv_batched(st->q_buf, st->k_buf, st->v_buf,
                w->q8_q_weight + Q8_BYTES((size_t)l * d * kv_dim_q),
                w->q8_k_weight + Q8_BYTES((size_t)l * d * kv_dim),
                w->q8_v_weight + Q8_BYTES((size_t)l * d * kv_dim),
                normed, kv_dim_q, kv_dim, d, 1);
        } else {
            dyn_matvec(st->q_buf, w->q_weight + l * d * kv_dim_q, normed, kv_dim_q, d);
            dyn_matvec(st->k_buf, w->k_weight + l * d * kv_dim, normed, kv_dim, d);
            dyn_matvec(st->v_buf, w->v_weight + l * d * kv_dim, normed, kv_dim, d);
        }
        if (prof) t_qkv += st_now_sec() - t0;
        if (prof) { fprintf(stderr, "[DDEC2] l=%d QKV done\n", l); fflush(stderr); }
        if (getenv("VLLM_PFDBG") && (l == 0 || l == 6 || l == 15 || l == 35)) {
            float qm = 0.0f, km = 0.0f, vm = 0.0f, xm = 0.0f;
            for (int i = 0; i < kv_dim_q; i++) { float a = fabsf(st->q_buf[i]); if (a > qm) qm = a; }
            for (int i = 0; i < kv_dim; i++) { float a = fabsf(st->k_buf[i]); if (a > km) km = a; }
            for (int i = 0; i < kv_dim; i++) { float a = fabsf(st->v_buf[i]); if (a > vm) vm = a; }
            for (int i = 0; i < d; i++) { float a = fabsf(x[i]); if (a > xm) xm = a; }
            printf("[DDBG] l=%d qmax=%.6g kmax=%.6g vmax=%.6g xmax=%.6g\n",
                   l, qm, km, vm, xm);
            fflush(stdout);
        }

        /* Q/K norms (Qwen3-VL specific) — per-head RMS normalization
         * Qwen3 uses Q/K norms: first RMS-normalize each head to unit length,
         * then multiply by learned weight vector. Applied before MRoPE. */
        if (w->q_norm) {
            const float *qw = w->q_norm + l * hd;
            vllm_qk_norm_q_ctx vq = { st->q_buf, qw, hd, nh };
            vllm_tp_parfor(0, nh, vllm_qk_norm_q_worker, &vq);
        }
        if (w->k_norm) {
            const float *kw = w->k_norm + l * hd;
            vllm_qk_norm_k_ctx vk = { st->k_buf, kw, hd, nkv };
            vllm_tp_parfor(0, nkv, vllm_qk_norm_k_worker, &vk);
        }

        /* MRoPE */
        if (st->cfg.has_mrope) {
            dyn_mrope(st->q_buf, st->k_buf, hd, nh, nkv, st->cfg.head_dim_full, pos, theta);
        } else {
            /* Standard RoPE (fallback for non-MRoPE models) */
            const float *freqs = rope_freq_table(hd / 2, theta);
            for (int i = 0; i < hd; i += 2) {
                float freq = freqs ? freqs[i / 2] : 1.0f / powf(theta, (float)i / (float)hd);
                float angle = (float)pos * freq;
                float ca = cosf(angle);
                float sa = sinf(angle);
                for (int h = 0; h < nh; h++) {
                    int base = h * hd;
                    float v0 = st->q_buf[base + i];
                    float v1 = st->q_buf[base + i + 1];
                    st->q_buf[base + i]     = v0 * ca - v1 * sa;
                    st->q_buf[base + i + 1] = v1 * ca + v0 * sa;
                }
                for (int h = 0; h < nkv; h++) {
                    int base = h * hd;
                    float v0 = st->k_buf[base + i];
                    float v1 = st->k_buf[base + i + 1];
                    st->k_buf[base + i]     = v0 * ca - v1 * sa;
                    st->k_buf[base + i + 1] = v1 * ca + v0 * sa;
                }
            }
        }

        /* Store K/V in cache (INT8 quantized if use_kv_q8, else float) */
        int cl = st->cache_len[l];
        if (cl >= st->kv_n_blocks * st->kv_bs)
            fprintf(stderr, "[KV-OOB] decode cl=%d cap=%d l=%d\n", cl,
                    st->kv_n_blocks * st->kv_bs, l);
        if (st->use_kv_q8) {
            /* Per-token per-head max-abs INT8 K/V quantization (near-lossless). */
            int8_t *kdst = kv_row_i8(st->k_cache_q8[l], cl, kv_dim, st->kv_bs);
            int8_t *vdst = kv_row_i8(st->v_cache_q8[l], cl, kv_dim, st->kv_bs);
            kv_quantize_per_head(kdst, vdst,
                                 st->k_scale[l] + (size_t)cl * nkv,
                                 st->v_scale[l] + (size_t)cl * nkv,
                                 st->k_buf, st->v_buf, nkv, hd);
            /* Also store in float cache */
            memcpy(kv_row_f32(st->k_cache[l], cl, kv_dim, st->kv_bs), st->k_buf, kv_dim * sizeof(float));
            memcpy(kv_row_f32(st->v_cache[l], cl, kv_dim, st->kv_bs), st->v_buf, kv_dim * sizeof(float));
        } else if (st->use_kv_q4) {
            q4_pack_token(q4_k_row(st->k_cache_q4[l], cl, 0, hd, st->kv_bs, nkv),
                          q4_v_row(st->v_cache_q4[l], cl, 0, hd, st->kv_bs, nkv),
                          st->k_buf, st->v_buf, nkv, hd);
            memcpy(kv_row_f32(st->k_cache[l], cl, kv_dim, st->kv_bs), st->k_buf, kv_dim * sizeof(float));
            memcpy(kv_row_f32(st->v_cache[l], cl, kv_dim, st->kv_bs), st->v_buf, kv_dim * sizeof(float));
        } else {
            memcpy(kv_row_f32(st->k_cache[l], cl, kv_dim, st->kv_bs), st->k_buf, kv_dim * sizeof(float));
            memcpy(kv_row_f32(st->v_cache[l], cl, kv_dim, st->kv_bs), st->v_buf, kv_dim * sizeof(float));
        }
        if (KV_GUARD) {
            int _blk = cl / st->kv_bs;
            const uint8_t *_bg = (const uint8_t *)st->k_cache[l][_blk]
                + (size_t)st->kv_bs * kv_dim * sizeof(float);
            for (int _i = 0; _i < KV_GUARD; _i += 64)
                if (_bg[_i] != 0xA5) {
                    fprintf(stderr, "[GUARD] decode-k-back l=%d cl=%d blk=%d off=%d\n",
                            l, cl, _blk, _i); fflush(stderr); break;
                }
            if (_blk + 1 < st->kv_n_blocks) {
                const uint8_t *_fg = (const uint8_t *)st->k_cache[l][_blk + 1] - KV_GUARD;
                for (int _i = 0; _i < KV_GUARD; _i += 64)
                    if (_fg[_i] != 0xA5) {
                        fprintf(stderr, "[GUARD] decode-k-front l=%d cl=%d blk=%d off=%d\n",
                                l, cl, _blk, _i); fflush(stderr); break;
                    }
            }
        }
        st->cache_len[l]++;

        /* Attention (GQA) */
        float scale = 1.0f / sqrtf((float)hd);
        int seq_len = st->cache_len[l];

        /* Standard / Blocked-Flash attention.
         * When seq_len >= FA_THRESHOLD and not using INT8 KV:
         * use tile-packed Flash Attention to eliminate stride-kv_dim cache misses.
         * Axiom: memory_bandwidth_reduction — tiled KV packing vs strided reads.
         * For short seq_len (< 64) or INT8 mode: fall back to scalar AVX2 path. */
        int use_q8 = st->use_kv_q8;

        if (prof) t0 = st_now_sec();
            if (!use_q8 && seq_len >= FA_THRESHOLD && !g_sparse_attn) {
                /* Blocked Flash Attention: pack K/V into L1-friendly tiles */
                for (int h = 0; h < nh; h++) {
                    int kh = (h * nkv) / nh;
                    flash_attn_blocked_single_q(
                        st->attn_buf + (size_t)h * hd,
                        st->q_buf + (size_t)h * hd,
                        st->k_cache[l], st->v_cache[l],
                        seq_len, kh * hd, kv_dim, hd, st->kv_bs, scale);
                }
            } else {
                /* Standard attention: per-head dot products (AVX2 accelerated).
                 * When use_kv_q8, reads INT8 K/V cache and dequantizes on-the-fly,
                 * trading 2 extra int8→float conversions for 4x memory bandwidth reduction.
                 * When g_sparse_attn, per-head top-k block sparse attention
                 * (Phase 1) replaces the full-position scan.
                 * M4h: head 循环 OMP 并行（32 heads 4 线程，每 head 独立写
                 * attn_buf，scores 用 per-iteration 局部数组避免共享竞争；并行
                 * 不改变每 head 的数值，位级一致保持）。 */
                vllm_attn_head_ctx vh = { st, nh, nkv, hd, l, seq_len, kv_dim, use_q8, scale };
                vllm_tp_parfor(0, nh, vllm_attn_head_worker, &vh);
            }
        if (prof) t_attn += st_now_sec() - t0;
        if (prof) { fprintf(stderr, "[DDEC2] l=%d ATT done\n", l); fflush(stderr); }

        /* W3: attnbuf IMMEDIATELY after attention (before O-proj), to detect
         * any corruption by the O-projection. */
        if (getenv("VLLM_DUMP_ATTN") && l == 0 && st->seq_len == 19)
            printf("[DUMP] l0-attnbuf-preO[0:8]=%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                   st->attn_buf[0], st->attn_buf[1], st->attn_buf[2], st->attn_buf[3],
                   st->attn_buf[4], st->attn_buf[5], st->attn_buf[6], st->attn_buf[7]);

        /* O projection + attention residual (fused — axiom: block_matrix_assoc_natural) */
        if (prof) t0 = st_now_sec();
        if (w->has_q4 && w->q4_o_weight) {
            dyn_matvec_q4_q8_fused_o_residual_batched(x, residual,
                w->q4_o_weight + Q4_BYTES((size_t)l * kv_dim_q * d),
                st->attn_buf, d, kv_dim_q, 1,
                w->has_x8 ? w->x8_o_weight + Q4_BYTES((size_t)l * kv_dim_q * d) : NULL);
        } else if (w->has_q8 && w->q8_o_weight) {
            dyn_matvec_q8_fused_o_residual_batched(x, residual,
                w->q8_o_weight + Q8_BYTES((size_t)l * kv_dim_q * d),
                st->attn_buf, d, kv_dim_q, 1);
        } else {
            dyn_matvec(normed, w->o_weight + l * kv_dim_q * d, st->attn_buf, d, kv_dim_q);
            vllm_add2_ctx vc = { x, residual, normed, d };
            vllm_tp_parfor(0, d, vllm_add2_worker, &vc);
        }
        if (prof) t_o += st_now_sec() - t0;
        if (prof) { fprintf(stderr, "[DDEC2] l=%d O done\n", l); fflush(stderr); }
        if (getenv("VLLM_DUMP_ATTN") && l == 0 && st->seq_len == 19) {
            printf("[DUMP] input-token=%d qpostrope[0:4]=%.6f %.6f %.6f %.6f\n",
                   token_id, st->q_buf[0], st->q_buf[1], st->q_buf[2], st->q_buf[3]);
            printf("[DUMP] l0-postattn x[0:8]=%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                   x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]);
            printf("[DUMP] l0-attnbuf[0:8]=%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                   st->attn_buf[0], st->attn_buf[1], st->attn_buf[2], st->attn_buf[3],
                   st->attn_buf[4], st->attn_buf[5], st->attn_buf[6], st->attn_buf[7]);
            {
                const float *k0 = kv_row_f32(st->k_cache[0], 0, kv_dim, st->kv_bs);
                const float *k18 = kv_row_f32(st->k_cache[0], 18, kv_dim, st->kv_bs);
                printf("[DUMP] l0-kcache[0][0:4]=%.6f %.6f %.6f %.6f\n", k0[0], k0[1], k0[2], k0[3]);
                printf("[DUMP] l0-kcache[18][0:4]=%.6f %.6f %.6f %.6f\n", k18[0], k18[1], k18[2], k18[3]);
                const float *v0 = kv_row_f32(st->v_cache[0], 0, kv_dim, st->kv_bs);
                printf("[DUMP] l0-vcache[0][0:4]=%.6f %.6f %.6f %.6f\n", v0[0], v0[1], v0[2], v0[3]);
                /* W2: recompute head-0 scores/weights from the same buffers the
                 * attention kernel reads; if wgt != ref then the score path is
                 * wrong, if wgt == ref but attnbuf != ref the V-acc is wrong. */
                int s_len = st->cache_len[0];
                printf("[DUMP] l0-seqlen=%d\n", s_len);
                const float *qt = st->q_buf;
                float scale2 = 1.0f / sqrtf((float)hd);
                float sc[64]; float mx = -1e9f;
                for (int t = 0; t < s_len && t < 64; t++) {
                    const float *kt = kv_row_f32(st->k_cache[0], t, kv_dim, st->kv_bs);
                    float dot = 0.0f;
                    for (int i = 0; i < hd; i++) dot += qt[i] * kt[i];
                    sc[t] = dot * scale2;
                    if (sc[t] > mx) mx = sc[t];
                }
                float se = 0.0f;
                for (int t = 0; t < s_len && t < 64; t++) { sc[t] = expf(sc[t] - mx); se += sc[t]; }
                printf("[DUMP] l0-h0-wgt=");
                for (int t = 0; t < s_len && t < 64; t++) printf("%.4f ", sc[t] / se);
                printf("\n");
                const float *k19 = kv_row_f32(st->k_cache[0], 19, kv_dim, st->kv_bs);
                const float *v19 = kv_row_f32(st->v_cache[0], 19, kv_dim, st->kv_bs);
                printf("[DUMP] l0-kcache19[0:4]=%.6f %.6f %.6f %.6f\n", k19[0], k19[1], k19[2], k19[3]);
                printf("[DUMP] l0-vcache19[0:4]=%.6f %.6f %.6f %.6f\n", v19[0], v19[1], v19[2], v19[3]);
                const float *k5 = kv_row_f32(st->k_cache[0], 5, kv_dim, st->kv_bs);
                const float *k10 = kv_row_f32(st->k_cache[0], 10, kv_dim, st->kv_bs);
                printf("[DUMP] l0-kcache5[0:4]=%.6f %.6f %.6f %.6f\n", k5[0], k5[1], k5[2], k5[3]);
                printf("[DUMP] l0-kcache10[0:4]=%.6f %.6f %.6f %.6f\n", k10[0], k10[1], k10[2], k10[3]);
                for (int _t = 0; _t < 19; _t++) {
                    const float *_vt = kv_row_f32(st->v_cache[0], _t, kv_dim, st->kv_bs);
                    if (_t == 5 || _t == 6 || _t == 9 || _t == 15 || _t == 18)
                        printf("[DUMP] l0-vcache%d[0:4]=%.6f %.6f %.6f %.6f\n",
                               _t, _vt[0], _vt[1], _vt[2], _vt[3]);
                }
                /* W4: kernel-truth head-0 weights + INT8 dequant rows. */
                printf("[DUMP] l0-kernel-wgt=");
                if (g_dbg_sumexp0 > 0.0f)
                    for (int _t = 0; _t < g_dbg_seq0 && _t < 64; _t++)
                        printf("%.4f ", g_dbg_scores0[_t] / g_dbg_sumexp0);
                printf("\n");
                if (st->use_kv_q8) {
                    for (int _t = 0; _t < 20; _t++) {
                        const int8_t *_ki = kv_row_i8(st->k_cache_q8[0], _t, kv_dim, st->kv_bs);
                        const int8_t *_vi = kv_row_i8(st->v_cache_q8[0], _t, kv_dim, st->kv_bs);
                        float _kks = st->k_scale[0][(size_t)_t * nkv] * KVQ_SCALE;
                        float _vvs = st->v_scale[0][(size_t)_t * nkv] * KVQ_SCALE;
                        printf("[DUMP] l0-q8row%d k=%.4f %.4f %.4f %.4f v=%.4f %.4f %.4f %.4f\n",
                               _t, _ki[0]*_kks, _ki[1]*_kks, _ki[2]*_kks, _ki[3]*_kks,
                               _vi[0]*_vvs, _vi[1]*_vvs, _vi[2]*_vvs, _vi[3]*_vvs);
                    }
                }
            }
            fflush(stdout);
        }

        /* --- FFN --- */
        if (prof) { fprintf(stderr, "[DDEC2] l=%d FFN in\n", l); fflush(stderr); }
        memcpy(residual, x, d * sizeof(float));
        dyn_rms_norm(normed, x, w->ffn_norm + l * d, d, eps);

        /* Save residual before FFN overwrites ffn_out_buf (alias!) */
        memcpy(st->attn_buf, residual, d * sizeof(float));

        if (st_ffn_q2_layer(w, l)) {
            /* Q2_1 FFN (layered q2mix: layers before the Q4 tail)
             * mixed-precision 2-bit (axiom: blas_precision_efficiency_tradeoff) */
            if (prof) t0 = st_now_sec();
            dyn_matvec_q2_q8_fused_gate_up(st->q_buf, st->k_buf,
                w->q2_gate_weight + Q2_BYTES((size_t)l * ff * d),
                w->q2_up_weight   + Q2_BYTES((size_t)l * ff * d),
                normed, ff, d);
            if (prof) t_gu += st_now_sec() - t0;
            vllm_silu_mul_ctx vg = { st->q_buf, st->k_buf, ff };
            vllm_tp_parfor(0, (ff + 15) / 16, vllm_silu_mul_worker, &vg);
            /* P4: fused down projection + FFN residual — writes directly to x */
            if (prof) t0 = st_now_sec();
            dyn_matvec_q2_q8_fused_down_residual(x, st->attn_buf,
                w->q2_down_weight + Q2_BYTES((size_t)l * d * ff),
                st->q_buf, d, ff);
            if (prof) t_down += st_now_sec() - t0;
            if (prof) { fprintf(stderr, "[DDEC2] l=%d FFN out (q2)\n", l); fflush(stderr); }
        } else if (w->has_q4) {
            /* Q4_0 path (incl. the q2mix tail layers): integer-domain
             * Q4×Q8 matvec (axiom: fixedpoint_quantize B=4) */
            if (prof) t0 = st_now_sec();
            dyn_matvec_q4_q8_fused_gate_up_batched(st->q_buf, st->k_buf,
                w->q4_gate_weight + Q4_BYTES((size_t)st_ffn_q4_layer_off(w, l) * ff * d),
                w->q4_up_weight   + Q4_BYTES((size_t)st_ffn_q4_layer_off(w, l) * ff * d),
                normed, ff, d, 1,
                w->has_x8 ? w->x8_gate_weight + Q4_BYTES((size_t)st_ffn_q4_layer_off(w, l) * ff * d) : NULL,
                w->has_x8 ? w->x8_up_weight   + Q4_BYTES((size_t)st_ffn_q4_layer_off(w, l) * ff * d) : NULL);
            if (prof) t_gu += st_now_sec() - t0;
            vllm_silu_mul_ctx vg = { st->q_buf, st->k_buf, ff };
            vllm_tp_parfor(0, (ff + 15) / 16, vllm_silu_mul_worker, &vg);
            /* P4: fused down projection + FFN residual — writes directly to x */
            if (prof) t0 = st_now_sec();
            dyn_matvec_q4_q8_fused_down_residual_batched(x, st->attn_buf,
                w->q4_down_weight + Q4_BYTES((size_t)st_ffn_q4_layer_off(w, l) * d * ff),
                st->q_buf, d, ff, 1,
                w->has_x8 ? w->x8_down_weight + Q4_BYTES((size_t)st_ffn_q4_layer_off(w, l) * d * ff) : NULL);
            if (prof) t_down += st_now_sec() - t0;
            if (prof) { fprintf(stderr, "[DDEC2] l=%d FFN out (q4)\n", l); fflush(stderr); }
        } else if (w->has_q8) {
            #define Q8O(e)  ((size_t)(e) / 32 * 34)
            if (prof) t0 = st_now_sec();
            dyn_matvec_q8_fused_gate_up_batched(st->q_buf, st->k_buf,
                w->q8_gate_weight + Q8O((size_t)l * ff * d),
                w->q8_up_weight   + Q8O((size_t)l * ff * d),
                normed, ff, d, 1);
            if (prof) t_gu += st_now_sec() - t0;
            vllm_silu_mul_ctx vg = { st->q_buf, st->k_buf, ff };
            vllm_tp_parfor(0, (ff + 15) / 16, vllm_silu_mul_worker, &vg);
            /* P4: fused down projection + FFN residual — writes directly to x */
            if (prof) t0 = st_now_sec();
            dyn_matvec_q8_fused_down_residual_batched(x, st->attn_buf,
                w->q8_down_weight + Q8O((size_t)l * d * ff),
                st->q_buf, d, ff, 1);
            if (prof) t_down += st_now_sec() - t0;
            if (prof) { fprintf(stderr, "[DDEC2] l=%d FFN out (q8)\n", l); fflush(stderr); }
            #undef Q8O
        } else {
            /* F32 path: standard float matvec */
            dyn_swiglu(st->ffn_out_buf, normed,
                        w->gate_weight + l * ff * d,
                        w->up_weight + l * ff * d,
                        w->down_weight + l * d * ff,
                        d, ff, st->q_buf, st->k_buf);
            /* FFN residual: use saved copy */
            vllm_add2_ctx va = { x, st->attn_buf, st->ffn_out_buf, d };
            vllm_tp_parfor(0, d, vllm_add2_worker, &va);
        }
        if (getenv("VLLM_DUMP_ATTN") && l == 0 && st->seq_len == 19) {
            printf("[DUMP] l0-postffn x[0:8]=%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                   x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]);
            printf("[DUMP] l0-logitsmax-todo\n");
            fflush(stdout);
        }
        if (getenv("VLLM_DUMP_LAYERS") && st->seq_len == 19) {
            float xm = 0.0f, nm = 0.0f, fom = 0.0f, am = 0.0f;
            for (int _i = 0; _i < d; _i++) { float a = fabsf(x[_i]); if (a > xm) xm = a; }
            for (int _i = 0; _i < d; _i++) { float a = fabsf(normed[_i]); if (a > nm) nm = a; }
            for (int _i = 0; _i < ff; _i++) { float a = fabsf(st->q_buf[_i]); if (a > fom) fom = a; }
            for (int _i = 0; _i < d; _i++) { float a = fabsf(st->attn_buf[_i]); if (a > am) am = a; }
            printf("[LAYER] l=%d x[0:4]=%.4f %.4f %.4f %.4f xmax=%.4f nrmmax=%.4f ffmax=%.4f attnresmax=%.4f\n",
                   l, x[0], x[1], x[2], x[3], xm, nm, fom, am);
            fflush(stdout);
        }
        GUARD_CHK(g_cn_hidden, hidden, "hidden", l, pos);
        GUARD_CHK(g_cn_qbuf, qbuf, "q_buf", l, pos);
        GUARD_CHK(g_cn_kbuf, kbuf, "k_buf", l, pos);
        GUARD_CHK(g_cn_vbuf, vbuf, "v_buf", l, pos);
        GUARD_CHK(g_cn_attnbuf, attnbuf, "attn_buf", l, pos);
        GUARD_CHK(g_cn_ffnbuf, ffnbuf, "ffn_buf", l, pos);
        GUARD_CHK(g_cn_ffnout, ffnout, "ffn_out_buf", l, pos);
        GUARD_CHK(g_cn_cachelen, cachelen, "cache_len", l, pos);
        GUARD_CHK(g_cn_scores, scores, "scores_buf", l, pos);

    }

    /* Final norm + LM head (Q8_0 when available — axiom: fixedpoint_quantize) */
    dyn_rms_norm(normed, x, w->final_norm, d, eps);
    if (prof) { fprintf(stderr, "[DDEC2] final_norm done\n"); fflush(stderr); }

    double t0 = 0.0;
    if (prof) t0 = st_now_sec();
    if (w->q4_lm_weight) {
        /* Q4_0 LM head: 331 MB weights vs 661 MB Q8_0 / 2.49 GB F32 */
        dyn_matvec_q4_q8(st->logits, w->q4_lm_weight, normed, vc, d);
    } else if (w->has_q8 && w->q8_lm_weight) {
        /* AVX2 Q8_0 dequantizing matvec: 660 MB weights vs 2.49 GB F32 */
        dyn_matvec_q8(st->logits, w->q8_lm_weight, normed, vc, d);
    } else {
        /* Scalar fallback */
        for (int t = 0; t < vc; t++) {
            float s = 0.0f;
            int base = t * d;
            for (int i = 0; i < d; i++) s += w->lm_head[base + i] * normed[i];
            st->logits[t] = s;
        }
    }
    if (prof) t_lm += st_now_sec() - t0;
    if (prof) { fprintf(stderr, "[DDEC2] lmhead done\n"); fflush(stderr); }
    if (getenv("VLLM_PFDBG") && st->seq_len < 4) {
        float lmax = 0.0f; int li = 0;
        for (int i = 0; i < vc; i++) { float a = fabsf(st->logits[i]); if (a > lmax) { lmax = a; li = i; } }
        printf("[DDBG] seq=%d logits max=%.6g@%d nlltok=%d\n",
               st->seq_len, lmax, li, token_id);
        fflush(stdout);
    }
    if (getenv("VLLM_DUMP_LOGITS")) {
        int topn = 8, ti[8]; float tv[8];
        for (int k = 0; k < topn; k++) { ti[k] = -1; tv[k] = -1e30f; }
        for (int i = 0; i < vc; i++) {
            float lv = st->logits[i];
            for (int k = 0; k < topn; k++) {
                if (lv > tv[k]) {
                    for (int j = topn - 1; j > k; j--) { ti[j] = ti[j-1]; tv[j] = tv[j-1]; }
                    ti[k] = i; tv[k] = lv;
                    break;
                }
            }
        }
        printf("[LTOP] seq=%d in=%d", st->seq_len, token_id);
        for (int k = 0; k < topn; k++) printf(" %d=%.3f", ti[k], tv[k]);
        printf("\n[LHID] x[0:6]=%.5f %.5f %.5f %.5f %.5f %.5f nrm[0:6]=%.5f %.5f %.5f %.5f %.5f %.5f\n",
               x[0], x[1], x[2], x[3], x[4], x[5],
               normed[0], normed[1], normed[2], normed[3], normed[4], normed[5]);
        fflush(stdout);
    }

    st->seq_len++;
    if (prof) { fprintf(stderr, "[DDEC2] seq_len done\n"); fflush(stderr); }
    if (st->mrope_pos > 0) st->mrope_pos++;

    if (prof) {
        double t_total = st_now_sec() - t_all0;
        double t_gemm = t_qkv + t_o + t_gu + t_down + t_lm;
        double t_other = t_total - t_gemm - t_attn;
        if (t_other < 0.0) t_other = 0.0;
        st->dec_t_qkv += t_qkv;
        st->dec_t_attn += t_attn;
        st->dec_t_o += t_o;
        st->dec_t_gu += t_gu;
        st->dec_t_down += t_down;
        st->dec_t_lm += t_lm;
        st->dec_t_other += t_other;
        if (getenv("VLLM_DEC_PROF")) {
            fprintf(stderr, "[DEC-SINGLE] seq=%d qkv=%.3fms attn=%.3f o=%.3f gu=%.3f down=%.3f lm=%.3f other=%.3f total=%.3f\n",
                    st->seq_len, t_qkv * 1e3, t_attn * 1e3, t_o * 1e3,
                    t_gu * 1e3, t_down * 1e3, t_lm * 1e3, t_other * 1e3,
                    (st_now_sec() - t_all0) * 1e3);
        }
    }
}

/* ================================================================
 * Continuous batching (MA5-PAR: independent sessions, batched forward)
 *
 * Decodes the next token for `nb` independent requests in one pass.
 * Per layer the weight matrices are read ONCE and reused across all nb
 * requests — this amortizes the memory-bound weight traffic that dominates
 * CPU decode. Each request keeps its own KV cache / cache_len and attends
 * only its own prefix, so every output is bit-identical to the sequential
 * st_qwen_model_forward path.
 * ================================================================ */
typedef struct {
    float *qb, *kb; const float *qw, *kw;
    int nq, nk, hd, q_rows, kv_dim;
} vllm_qk_norm_batch_ctx;

static void vllm_qk_norm_batch_worker(void *ctx_, int th2) {
    vllm_qk_norm_batch_ctx *c = ctx_;
    int r = th2 / (c->nq + c->nk);
    int hh = th2 % (c->nq + c->nk);
    if (hh < c->nq) {
        float *qh = c->qb + (size_t)r * c->q_rows + (size_t)hh * c->hd;
        float ss = 0.0f;
        for (int i = 0; i < c->hd; i++) ss += qh[i] * qh[i];
        float rms = 1.0f / sqrtf(ss / (float)c->hd + 1e-6f);
        for (int i = 0; i < c->hd; i++) qh[i] = qh[i] * rms * c->qw[i];
    } else {
        int hk = hh - c->nq;
        float *kh = c->kb + (size_t)r * c->kv_dim + (size_t)hk * c->hd;
        float ss = 0.0f;
        for (int i = 0; i < c->hd; i++) ss += kh[i] * kh[i];
        float rms = 1.0f / sqrtf(ss / (float)c->hd + 1e-6f);
        for (int i = 0; i < c->hd; i++) kh[i] = kh[i] * rms * c->kw[i];
    }
}

typedef struct {
    STQwenInferenceState *const *sts;
    const float *qb; float *att;
    int nb, nh, nkv, hd, hd8, l, kv_dim, q_rows;
    float scale;
} vllm_attn_batch_ctx;

static void vllm_attn_batch_worker(void *ctx_, int rt) {
    vllm_attn_batch_ctx *c = ctx_;
    int r = rt / c->nh, ha = rt - r * c->nh;
    STQwenInferenceState *st = c->sts[r];
    int kh = (ha * c->nkv) / c->nh;
    int seq_len = st->cache_len[c->l];          /* = pos+1 after KV store */
    int mkv = st->max_kv_slots;
    const float *qs = c->qb + (size_t)r * c->q_rows + (size_t)ha * c->hd;
    float *scores = st->scores_buf + (size_t)ha * mkv;
    float *o = c->att + (size_t)r * c->q_rows + (size_t)ha * c->hd;
    float max_score = -1e9f;

    if (st->use_kv_q8) {
        const float *ksc = st->k_scale[c->l] + kh;
        for (int s = 0; s < seq_len; s++) {
            const int8_t *ks = kv_row_i8(st->k_cache_q8[c->l], s, c->kv_dim, st->kv_bs) + (size_t)kh * c->hd;
            float ksc_s = ksc[(size_t)s * c->nkv] * KVQ_SCALE;
            float dot = 0.0f;
            for (int i = 0; i < c->hd; i++) dot += qs[i] * (float)ks[i];
            dot *= ksc_s;
            scores[s] = dot * c->scale;
            if (scores[s] > max_score) max_score = scores[s];
        }
    } else if (st->use_kv_q4) {
        for (int s = 0; s < seq_len; s++) {
            const uint8_t *kp = q4_k_row(st->k_cache_q4[c->l], s, kh, c->hd, st->kv_bs, c->nkv);
            float dot = q4_head_dot(kp, qs, c->hd);
            scores[s] = dot * c->scale;
            if (scores[s] > max_score) max_score = scores[s];
        }
    } else {
        for (int s = 0; s < seq_len; s++) {
            const float *ks = kv_row_f32(st->k_cache[c->l], s, c->kv_dim, st->kv_bs) + (size_t)kh * c->hd;
            float dot = 0.0f;
            for (int i = 0; i < c->hd; i++) dot += qs[i] * ks[i];
            scores[s] = dot * c->scale;
            if (scores[s] > max_score) max_score = scores[s];
        }
    }

    float sum_exp = 0.0f;
    for (int s = 0; s < seq_len; s++) {
        scores[s] = expf(scores[s] - max_score);
        sum_exp += scores[s];
    }
    float inv_sum = 1.0f / sum_exp;
    for (int i = 0; i < c->hd; i++) o[i] = 0.0f;

    if (st->use_kv_q8) {
        const float *vsc = st->v_scale[c->l] + kh;
        for (int s = 0; s < seq_len; s++) {
            float wgt = scores[s] * inv_sum;
            const int8_t *vs = kv_row_i8(st->v_cache_q8[c->l], s, c->kv_dim, st->kv_bs) + (size_t)kh * c->hd;
            float vsc_s = vsc[(size_t)s * c->nkv] * KVQ_SCALE;
            for (int i = 0; i < c->hd; i++)
                o[i] += wgt * (float)vs[i] * vsc_s;
        }
    } else if (st->use_kv_q4) {
        for (int s = 0; s < seq_len; s++) {
            float wgt = scores[s] * inv_sum;
            const uint8_t *vp = q4_v_row(st->v_cache_q4[c->l], s, kh, c->hd, st->kv_bs, c->nkv);
            q4_head_vacc(o, vp, wgt, c->hd);
        }
    } else {
        for (int s = 0; s < seq_len; s++) {
            float wgt = scores[s] * inv_sum;
            const float *vs = kv_row_f32(st->v_cache[c->l], s, c->kv_dim, st->kv_bs) + (size_t)kh * c->hd;
            for (int i = 0; i < c->hd; i++)
                o[i] += wgt * vs[i];
        }
    }
}

void st_qwen_model_forward_batch(STQwenInferenceState *const *sts,
                                 int nb, const int *tokens) {
    if (nb <= 0) return;
    STModelWeights *w = &sts[0]->weights;
    int d  = sts[0]->cfg.dim;
    int nl = w->n_layers_allocated;
    int nh = sts[0]->cfg.n_heads;
    int nkv = sts[0]->cfg.n_kv_heads;
    int hd = sts[0]->cfg.head_dim;
    int ff = sts[0]->cfg.ffn_dim;
    int vc = sts[0]->cfg.vocab_size;
    float eps = sts[0]->cfg.norm_eps;
    float theta = sts[0]->cfg.rope_theta;
    int q_rows = nh * hd;   /* 4096 */
    int kv_dim = nkv * hd;  /* 1024 */
    int use_q4 = w->has_q4 && w->q4_q_weight;
    st_default_threads();   /* RK3588: default 4x A76 (A55 degrades GEMM) */
    int hd8 = hd & ~7;

    /* Batch scratch: [nb][d] hidden/normed/residual, [nb][q_rows] q/attn,
     * [nb][kv_dim] k/v, [nb][ff] gate/up. One calloc block, freed on exit. */
    size_t nb_d   = (size_t)nb * (size_t)d;
    size_t nb_qr  = (size_t)nb * (size_t)q_rows;
    size_t nb_kv  = (size_t)nb * (size_t)kv_dim;
    size_t nb_ff  = (size_t)nb * (size_t)ff;
    float *h_batch = (float *)malloc(nb_d * sizeof(float));
    float *nrm     = (float *)malloc(nb_d * sizeof(float));
    float *res     = (float *)malloc(nb_d * sizeof(float));
    float *qb      = (float *)malloc(nb_qr * sizeof(float));
    float *att     = (float *)malloc(nb_qr * sizeof(float));
    float *kb      = (float *)malloc(nb_kv * sizeof(float));
    float *vb      = (float *)malloc(nb_kv * sizeof(float));
    float *gb      = (float *)malloc(nb_ff * sizeof(float));
    float *ub      = (float *)malloc(nb_ff * sizeof(float));
    if (!h_batch || !nrm || !res || !qb || !att || !kb || !vb || !gb || !ub) {
        free(h_batch); free(nrm); free(res); free(qb); free(att);
        free(kb); free(vb); free(gb); free(ub);
        fprintf(stderr, "[BATCH] OOM scratch\n");
        return;
    }
    float *normed = nrm;   /* reused as the per-request final-norm scratch */

    /* Embed each request's next token into its hidden row. */
    float *emb = w->token_embed;
    if (!emb) { free(h_batch); free(nrm); free(res); free(qb); free(att);
                free(kb); free(vb); free(gb); free(ub); return; }
    for (int r = 0; r < nb; r++) {
        float *x = h_batch + (size_t)r * d;
        for (int i = 0; i < d; i++) x[i] = st_emb_get(w, (size_t)tokens[r], (size_t)i, d);
    }

    for (int l = 0; l < nl; l++) {
        /* --- 1. Batched attention RMSNorm --- */
        dyn_rms_norm_batch(nrm, h_batch, w->attn_norm + (size_t)l * d, nb, d, eps);

        /* --- 2. Batched QKV projection (weights read once per layer) --- */
        if (use_q4 && w->q4_q_weight) {
            dyn_matvec_q4_q8_fused_qkv_batched(
                qb, kb, vb,
                w->q4_q_weight + Q4_BYTES((size_t)l * d * q_rows),
                w->q4_k_weight + Q4_BYTES((size_t)l * d * kv_dim),
                w->q4_v_weight + Q4_BYTES((size_t)l * d * kv_dim),
                nrm, q_rows, kv_dim, d, nb,
                w->has_x8 ? w->x8_q_weight + Q4_BYTES((size_t)l * d * q_rows) : NULL,
                w->has_x8 ? w->x8_k_weight + Q4_BYTES((size_t)l * d * kv_dim) : NULL,
                w->has_x8 ? w->x8_v_weight + Q4_BYTES((size_t)l * d * kv_dim) : NULL);
        } else {
            dyn_matvec_q8_fused_qkv_batched(
                qb, kb, vb,
                w->q8_q_weight + Q8_BYTES((size_t)l * d * q_rows),
                w->q8_k_weight + Q8_BYTES((size_t)l * d * kv_dim),
                w->q8_v_weight + Q8_BYTES((size_t)l * d * kv_dim),
                nrm, q_rows, kv_dim, d, nb);
        }

        /* --- 3. Per-request Q/K norms (batched, one omp region) + MRoPE + KV --- */
        {
            const float *qw = w->q_norm ? w->q_norm + (size_t)l * hd : NULL;
            const float *kw = w->k_norm ? w->k_norm + (size_t)l * hd : NULL;
            int nq = w->q_norm ? nh : 0;
            int nk = w->k_norm ? nkv : 0;
            int total = nb * (nq + nk);
            vllm_qk_norm_batch_ctx vc = { qb, kb, qw, kw, nq, nk, hd, q_rows, kv_dim };
            vllm_tp_parfor(0, total, vllm_qk_norm_batch_worker, &vc);
        }
        for (int r = 0; r < nb; r++) {
            STQwenInferenceState *st = sts[r];
            int pos = st->mrope_pos > 0 ? st->mrope_pos : st->cache_len[l];
            float *qt = qb + (size_t)r * q_rows;
            float *kt = kb + (size_t)r * kv_dim;
            float *vt = vb + (size_t)r * kv_dim;

            if (st->cfg.has_mrope) {
                dyn_mrope(qt, kt, hd, nh, nkv, st->cfg.head_dim_full, pos, theta);
            } else {
                const float *freqs = rope_freq_table(hd / 2, theta);
                for (int i = 0; i < hd; i += 2) {
                    float freq = freqs ? freqs[i / 2] : 1.0f / powf(theta, (float)i / (float)hd);
                    float angle = (float)pos * freq;
                    float ca = cosf(angle), sa = sinf(angle);
                    for (int h = 0; h < nh; h++) {
                        int base = h * hd;
                        float v0 = qt[base + i], v1 = qt[base + i + 1];
                        qt[base + i] = v0 * ca - v1 * sa;
                        qt[base + i + 1] = v1 * ca + v0 * sa;
                    }
                    for (int h = 0; h < nkv; h++) {
                        int base = h * hd;
                        float v0 = kt[base + i], v1 = kt[base + i + 1];
                        kt[base + i] = v0 * ca - v1 * sa;
                        kt[base + i + 1] = v1 * ca + v0 * sa;
                    }
                }
            }

            /* KV-cache store (float + compressed payload). */
            memcpy(kv_row_f32(st->k_cache[l], pos, kv_dim, st->kv_bs), kt, kv_dim * sizeof(float));
            memcpy(kv_row_f32(st->v_cache[l], pos, kv_dim, st->kv_bs), vt, kv_dim * sizeof(float));
            if (st->use_kv_q8) {
                kv_quantize_per_head(kv_row_i8(st->k_cache_q8[l], pos, kv_dim, st->kv_bs),
                                     kv_row_i8(st->v_cache_q8[l], pos, kv_dim, st->kv_bs),
                                     st->k_scale[l] + (size_t)pos * nkv,
                                     st->v_scale[l] + (size_t)pos * nkv,
                                     kt, vt, nkv, hd);
            } else if (st->use_kv_q4) {
                q4_pack_token(q4_k_row(st->k_cache_q4[l], pos, 0, hd, st->kv_bs, nkv),
                              q4_v_row(st->v_cache_q4[l], pos, 0, hd, st->kv_bs, nkv),
                              kt, vt, nkv, hd);
            }
            st->cache_len[l]++;
        }

        /* --- 4. Attention per (request x head), flattened for the pool --- */
        float scale = 1.0f / sqrtf((float)hd);
        int total_tasks = nb * nh;
        vllm_attn_batch_ctx va = { sts, qb, att, nb, nh, nkv, hd, hd8, l, kv_dim, q_rows, scale };
        vllm_tp_parfor(0, total_tasks, vllm_attn_batch_worker, &va);

        /* --- 5. Batched O projection + residual --- */
        for (int r = 0; r < nb; r++)
            memcpy(res + (size_t)r * d, h_batch + (size_t)r * d, (size_t)d * sizeof(float));
        if (use_q4 && w->q4_o_weight) {
            dyn_matvec_q4_q8_fused_o_residual_batched(
                h_batch, res, w->q4_o_weight + Q4_BYTES((size_t)l * q_rows * d),
                att, d, q_rows, nb,
                w->has_x8 ? w->x8_o_weight + Q4_BYTES((size_t)l * q_rows * d) : NULL);
        } else {
            dyn_matvec_q8_fused_o_residual_batched(
                h_batch, res, w->q8_o_weight + Q8_BYTES((size_t)l * q_rows * d),
                att, d, q_rows, nb);
        }

        /* --- 6. Batched FFN --- */
        dyn_rms_norm_batch(nrm, h_batch, w->ffn_norm + (size_t)l * d, nb, d, eps);
        if (st_ffn_q2_layer(w, l)) {
            dyn_matvec_q2_q8_fused_gate_up_batched(
                gb, ub,
                w->q2_gate_weight + Q2_BYTES((size_t)l * ff * d),
                w->q2_up_weight   + Q2_BYTES((size_t)l * ff * d),
                nrm, ff, d, nb);
        } else if (use_q4 && w->q4_gate_weight) {
            dyn_matvec_q4_q8_fused_gate_up_batched(
                gb, ub,
                w->q4_gate_weight + Q4_BYTES((size_t)st_ffn_q4_layer_off(w, l) * ff * d),
                w->q4_up_weight   + Q4_BYTES((size_t)st_ffn_q4_layer_off(w, l) * ff * d),
                nrm, ff, d, nb,
                w->has_x8 ? w->x8_gate_weight + Q4_BYTES((size_t)st_ffn_q4_layer_off(w, l) * ff * d) : NULL,
                w->has_x8 ? w->x8_up_weight   + Q4_BYTES((size_t)st_ffn_q4_layer_off(w, l) * ff * d) : NULL);
        } else {
            dyn_matvec_q8_fused_gate_up_batched(
                gb, ub,
                w->q8_gate_weight + Q8_BYTES((size_t)l * ff * d),
                w->q8_up_weight   + Q8_BYTES((size_t)l * ff * d),
                nrm, ff, d, nb);
        }
        vllm_silu_mul_ctx vg = { gb, ub, nb * ff };
        vllm_tp_parfor(0, (nb * ff + 15) / 16, vllm_silu_mul_worker, &vg);
        for (int r = 0; r < nb; r++)
            memcpy(res + (size_t)r * d, h_batch + (size_t)r * d, (size_t)d * sizeof(float));
        if (st_ffn_q2_layer(w, l)) {
            dyn_matvec_q2_q8_fused_down_residual_batched(
                h_batch, res, w->q2_down_weight + Q2_BYTES((size_t)l * d * ff),
                gb, d, ff, nb);
        } else if (use_q4 && w->q4_down_weight) {
            dyn_matvec_q4_q8_fused_down_residual_batched(
                h_batch, res,
                w->q4_down_weight + Q4_BYTES((size_t)st_ffn_q4_layer_off(w, l) * d * ff),
                gb, d, ff, nb,
                w->has_x8 ? w->x8_down_weight + Q4_BYTES((size_t)st_ffn_q4_layer_off(w, l) * d * ff) : NULL);
        } else {
            dyn_matvec_q8_fused_down_residual_batched(
                h_batch, res, w->q8_down_weight + Q8_BYTES((size_t)l * d * ff),
                gb, d, ff, nb);
        }
    }

    /* --- Scatter hidden + per-request final norm + LM head --- */
    for (int r = 0; r < nb; r++) {
        STQwenInferenceState *st = sts[r];
        st->seq_len++;
        if (st->mrope_pos > 0) st->mrope_pos++;
        memcpy(st->hidden, h_batch + (size_t)r * d, (size_t)d * sizeof(float));
        dyn_rms_norm(normed, st->hidden, w->final_norm, d, eps);
        if (w->q4_lm_weight) {
            dyn_matvec_q4_q8(st->logits, w->q4_lm_weight, normed, vc, d);
        } else if (w->has_q8 && w->q8_lm_weight) {
            dyn_matvec_q8(st->logits, w->q8_lm_weight, normed, vc, d);
        } else {
            for (int t = 0; t < vc; t++) {
                float s = 0.0f;
                int base = t * d;
                for (int i = 0; i < d; i++) s += w->lm_head[base + i] * normed[i];
                st->logits[t] = s;
            }
        }
    }

    free(h_batch); free(nrm); free(res); free(qb); free(att);
    free(kb); free(vb); free(gb); free(ub);
}

/* Pack a layer's strided K/V cache into contiguous per-head buffers:
 * k_pack[kh][s][i] = k_cache[s*kv_dim + kh*hd + i]. Packed once per layer and
 * reused across all mini-batch queries, eliminating the stride-kv_dim cache
 * misses in the Q·K and weighted-V loops (axiom: memory_bandwidth_reduction). */
typedef struct {
    float *const *k_cache; float *const *v_cache;
    float *k_pack, *v_pack;
    int seq_len, nkv, hd, kv_dim, seq_stride, bs;
} vllm_pack_kv_ctx;

static void vllm_pack_kv_worker(void *ctx_, int kh) {
    vllm_pack_kv_ctx *c = ctx_;
    float *kd = c->k_pack + (size_t)kh * c->seq_stride * c->hd;
    float *vd = c->v_pack + (size_t)kh * c->seq_stride * c->hd;
    for (int s = 0; s < c->seq_len; s++) {
        const float *ks = c->k_cache[s / c->bs] + (size_t)(s % c->bs) * (size_t)c->kv_dim + (size_t)kh * c->hd;
        const float *vs = c->v_cache[s / c->bs] + (size_t)(s % c->bs) * (size_t)c->kv_dim + (size_t)kh * c->hd;
        memcpy(kd + (size_t)s * c->hd, ks, c->hd * sizeof(float));
        memcpy(vd + (size_t)s * c->hd, vs, c->hd * sizeof(float));
    }
}

static void st_pack_kv_heads(float *const *k_cache,
                             float *const *v_cache,
                             float *restrict k_pack, float *restrict v_pack,
                             int seq_len, int nkv, int hd, int kv_dim,
                             int seq_stride, int bs)
{
    vllm_pack_kv_ctx vc = { k_cache, v_cache, k_pack, v_pack, seq_len, nkv, hd, kv_dim, seq_stride, bs };
    vllm_tp_parfor(0, nkv, vllm_pack_kv_worker, &vc);
}





static void st_attn_batched_packed(
    float *restrict attn_out,        /* [nb * nh * hd] */
    const float *restrict q_buf,     /* [nb * nh * hd] */
    const float *restrict k_pack,    /* [nkv * seq_stride * hd] */
    const float *restrict v_pack,    /* [nkv * seq_stride * hd] */
    int nb, int prev_len, int seq_stride,
    int nh, int nkv, int hd, float scale,
    float *restrict scores,          /* [nh * score_stride] */
    int score_stride,
    float *restrict imp_head)        /* [nh * score_stride] or NULL */
{
    /* P8: online-softmax single-pass batched prefill attention
     * (VLLM_ATTN_ONLINE=1) is only implemented on x86; the NEON path uses
     * the 2-pass kernel. */
    st_attn_batched_packed_neon(attn_out, q_buf, k_pack, v_pack, nb, prev_len, seq_stride, nh, nkv, hd, scale, scores, score_stride, imp_head);
}



static void st_attn_batched_packed_sparse(
    float *restrict attn_out,        /* [nb * nh * hd] */
    const float *restrict q_buf,     /* [nb * nh * hd] */
    const float *restrict k_pack,    /* [nkv * seq_stride * hd] */
    const float *restrict v_pack,    /* [nkv * seq_stride * hd] */
    int nb, int prev_len, int seq_stride,
    int nh, int nkv, int hd, float scale,
    float *restrict scores,          /* [nh * score_stride] */
    int score_stride, int bs, int k_blocks, int n_probe,
    float *restrict imp_head)        /* [nh * score_stride] or NULL */
{
    st_attn_batched_packed_sparse_neon(attn_out, q_buf, k_pack, v_pack, nb, prev_len, seq_stride, nh, nkv, hd, scale, scores, score_stride, bs, k_blocks, n_probe, imp_head);
}

/* ================================================================
 * Mini-Batch Prefill (axiom: blas_qkv_fusion_categorical)
 *
 * Processes n_tokens prompt tokens through all layers in mini-batches
 * of PREFILL_BATCH_SIZE.  Within each mini-batch, weight dequantization
 * in QKV/O/FFN projections is done once per block and reused across
 * all B tokens, reducing memory bandwidth ~B× for weight reads.
 *
 * Per-token operations (RMSNorm, Q/K norms, MRoPE, KV-cache, attention)
 * still run sequentially within the mini-batch loop.
 *
 * Work buffer: allocated once on heap, sized for B tokens.
 * Layout: hidden | normed | qbuf | kbuf | vbuf | attn_buf | residual
 * ================================================================ */
/* TEMP DEBUG: guard-page allocator for prefill's big scratch buffers.
 * (Kept disabled: guard layout did not capture the corruption and shifted
 *  the failure point, confirming a layout-sensitive small tail overrun.) */

/* Axiom arith_range_bound_001 (memory-bounds edition): allocate +64-byte tail
 * sentinel WITHOUT changing heap layout (allocation grows by 64B only), so the
 * layout-sensitive S=4096-after-prefix corruption keeps reproducing; any write
 * past the logical end is caught explicitly instead of as silent 0xC0000374.
 * g_wbuf_pad: debug-only padding to absorb suspected wbuf/k_pack over-writes. */
static size_t g_wbuf_pad = 0;
/* #region debug-point E:wbuf-pageguard
 * DEBUG: page-guard the prefill wbuf partitions + k_pack (via VirtualAlloc) so
 * any >64B over-write past a partition fails fast as an AV at the exact
 * kernel, instead of silently corrupting the heap free-list (which only trips
 * later at an unrelated malloc). g_xq_pages=1 enables. */
/* Portable page-guard helpers come from vllm_platform.h (st_pg_alloc /
 * st_pg_free): VirtualAlloc on Windows, mmap+mprotect on Linux/RK3588. */
#define DBG_MEM_COMMIT     0x1000u
#define DBG_MEM_RESERVE    0x2000u
#define DBG_MEM_RELEASE    0x8000u
#define DBG_PAGE_RW        0x04u
#define DBG_PAGE_NOACCESS  0x01u
/* Page-guarded alloc/free pair: guarded payload + trailing guard page
 * (g_xq_pages=1), else plain calloc/free. Used for the prefill KV cache
 * blocks, k_scale/v_scale, scores_buf, imp_head, prefill_importance and the
 * st scratch buffers so ANY over-write past them AVs at the exact kernel. */
static void *cf_pg_alloc(size_t bytes) {
    if (g_xq_pages) return st_pg_alloc(bytes);
    return calloc(1, bytes ? bytes : 1);
}
static void cf_pg_free(void *p) {
    if (!p) return;
    if (g_xq_pages) st_pg_free(p);
    else            free(p);
}
/* Exported for main.c (L3 eviction frees KV block bases with the same path). */
void st_qwen_kv_free_raw(void *raw) { cf_pg_free(raw); }

/* Re-allocate the fp32 (and INT8) KV blocks that Phase-2 L3 eviction
 * physically freed (pointers set NULL in st->k_cache[l][b] / v_cache[l][b]).
 * A later request's prefill writes those block rows again, so they must be
 * re-backed before use (the previous version crashed with a NULL deref on the
 * second prefill after an eviction). Returns bytes allocated (0 = none). */
size_t st_qwen_kv_rebuild_freed(STQwenInferenceState *st) {
    if (!st) return 0;
    int nl = st->weights.n_layers_allocated;
    int bs = st->kv_bs;
    int kv_dim = st->cfg.n_kv_heads * st->cfg.head_dim;
    size_t fdat = (size_t)bs * (size_t)kv_dim * sizeof(float);
    size_t idat = (size_t)bs * (size_t)kv_dim;
    size_t total = 0;
    for (int l = 0; l < nl; l++) {
        for (int b = 0; b < st->kv_n_blocks; b++) {
            if (!st->k_cache[l][b]) {
                uint8_t *raw = (uint8_t *)cf_pg_alloc(KV_GUARD + fdat + KV_GUARD);
                if (!raw) { fprintf(stderr, "[KV] OOM rebuilding K block l=%d b=%d\n", l, b); return total; }
                memset(raw, 0xA5, KV_GUARD);
                memset(raw + KV_GUARD + fdat, 0xA5, KV_GUARD);
                st->k_cache[l][b] = (float *)(raw + KV_GUARD);
                total += KV_GUARD + fdat + KV_GUARD;
            }
            if (!st->v_cache[l][b]) {
                uint8_t *raw = (uint8_t *)cf_pg_alloc(KV_GUARD + fdat + KV_GUARD);
                if (!raw) { fprintf(stderr, "[KV] OOM rebuilding V block l=%d b=%d\n", l, b); return total; }
                memset(raw, 0xA5, KV_GUARD);
                memset(raw + KV_GUARD + fdat, 0xA5, KV_GUARD);
                st->v_cache[l][b] = (float *)(raw + KV_GUARD);
                total += KV_GUARD + fdat + KV_GUARD;
            }
            if (st->use_kv_q8 && st->k_cache_q8 && st->k_cache_q8[l] && !st->k_cache_q8[l][b]) {
                uint8_t *raw = (uint8_t *)cf_pg_alloc(KV_GUARD + idat + KV_GUARD);
                if (!raw) { fprintf(stderr, "[KV] OOM rebuilding K8 block l=%d b=%d\n", l, b); return total; }
                memset(raw, 0xA5, KV_GUARD);
                memset(raw + KV_GUARD + idat, 0xA5, KV_GUARD);
                st->k_cache_q8[l][b] = (int8_t *)(raw + KV_GUARD);
                total += KV_GUARD + idat + KV_GUARD;
            }
            if (st->use_kv_q8 && st->v_cache_q8 && st->v_cache_q8[l] && !st->v_cache_q8[l][b]) {
                uint8_t *raw = (uint8_t *)cf_pg_alloc(KV_GUARD + idat + KV_GUARD);
                if (!raw) { fprintf(stderr, "[KV] OOM rebuilding V8 block l=%d b=%d\n", l, b); return total; }
                memset(raw, 0xA5, KV_GUARD);
                memset(raw + KV_GUARD + idat, 0xA5, KV_GUARD);
                st->v_cache_q8[l][b] = (int8_t *)(raw + KV_GUARD);
                total += KV_GUARD + idat + KV_GUARD;
            }
        }
    }
    return total;
}
static void *cf_aligned_canary(size_t nbytes, uint8_t **tail) {
    if (g_xq_pages) {
        void *p = st_pg_alloc(nbytes + g_wbuf_pad + 64);
        if (!p) return NULL;
        *tail = (uint8_t *)p + nbytes + g_wbuf_pad;
        memset(*tail, 0xA5, 64);
        return p;
    }
    void *p = st_aligned_alloc(nbytes + g_wbuf_pad + 64, 64);
    if (!p) return NULL;
    *tail = (uint8_t *)p + nbytes + g_wbuf_pad;
    memset(*tail, 0xA5, 64);
    return p;
}
static void cf_aligned_free_canary(void *p) {
    if (!p) return;
    if (g_xq_pages) st_pg_free(p);
    else            st_aligned_free(p);
}
/* #endregion */
static void *cf_calloc_canary(size_t n, size_t sz, uint8_t **tail) {
    void *p = calloc(1, n * sz + 64);
    if (!p) return NULL;
    *tail = (uint8_t *)p + n * sz;
    memset(*tail, 0xA5, 64);
    return p;
}
/* Guarded allocation: +KV_GUARD 0xA5 tail so an over-write past the logical
 * end (beyond 64B) is still caught by the [GUARD] probe after each writer. */
static void *cf_calloc_guard(size_t n, size_t sz, uint8_t **tail) {
    void *p = cf_pg_alloc(n * sz + KV_GUARD);
    if (!p) return NULL;
    *tail = (uint8_t *)p + n * sz;
    memset(*tail, 0xA5, KV_GUARD);
    return p;
}

/* ---- Activation-scratch canary (xq/xd) ----
 * The batched GEMM kernels malloc xq/xd, quantize into them, then free them.
 * Any over-write past the payload lands on the adjacent heap block / free-list
 * metadata and only fails fast LATER (layout-sensitive, cumulative). These
 * helpers append a 64-byte 0xA5 tail so the first over-write is caught at the
 * matching free().
 * g_xq_pad: debug-only padding absorbed into each allocation so a suspected
 * over-write (< pad) stays inside the block instead of corrupting the heap.
 * Set to 0 to restore exact sizes. */
static size_t g_xq_pad = 0;
/* #region debug-point F:xq-stagger
 * LFH/segment-heap allocation staggering (VLLM_XQ_STAGGER, default ON):
 * the batched prefill GEMMs allocate IDENTICAL xq/xd sizes on every layer;
 * on the CRT heap (_aligned_malloc, VLLM_XQ_PAGES=0) the allocator then
 * returns the SAME block for every iteration, so any timing-sensitive
 * free/alloc collision (or a delayed writer) repeatedly lands on the same
 * offset and can corrupt heap metadata -> 0xC0000374, probabilistic and
 * layout-sensitive. A layer-dependent 64B offset routes adjacent layers
 * into different heap buckets, breaking the reuse pattern. g_npu_layer is
 * constant within one dyn_matvec call (alloc and free agree); allocations
 * larger than XQ_STAGGER_MAX (NPU weights) are left untouched. */
#define XQ_STAGGER_MAX_BYTES (1u << 20)
static int g_xq_stagger = 1;
static int xq_stagger_env(void) {
    const char *e = getenv("VLLM_XQ_STAGGER");
    if (!e) return 0;              /* 默认关闭: 逐层错位尺寸破坏分配器复用, 调试用 */
    return (e[0] == '0') ? 0 : 1;
}
/* #endregion */
/* Step-granular heap validation (VLLM_HEAP_CHECK=1): HeapValidate after each
 * prefill kernel step. PB-HEAP (mini-batch granularity) proved the CRT heap is
 * intact at every mini-batch boundary yet 0xC0000374 still fires mid-batch, so
 * the corruption happens INSIDE a mini-batch. These checkpoints narrow the
 * window to the exact kernel step (qkv/kvstore/attn/o/gateup/down). HeapValidate
 * is cheap here: the CRT heap holds only a handful of blocks at this point. */
static int g_heap_check = 0;
static int heap_check_env(void) {
    const char *e = getenv("VLLM_HEAP_CHECK");
    if (!e) return 0;
    return (e[0] == '0') ? 0 : 1;
}
/* HeapWalk-based integrity probe: HeapValidate/_heapchk both miss the
 * corruption that trips CRT malloc with 0xC0000374 (8/8 instrumented runs
 * crashed with every checkpoint clean). HeapWalk enumerates every block and
 * fails with an error other than ERROR_NO_MORE_ITEMS when the free list or a
 * block header is damaged. Must hold the heap lock (no concurrent allocs at
 * the HCK checkpoints). Linux 下 CRT malloc 越界会直接 abort，探针为空实现。 */
static int heap_walk_ok(void) { return 1; }
#define HCK(STEP) do { \
    if (g_heap_check) { \
        if (!heap_walk_ok()) { \
            fprintf(stderr, "[HEAP-FAIL] walk step=%s l=%d tok=%d\n", \
                    (STEP), l, tok_offset + nb); \
            fflush(stderr); _exit(3); \
        } \
    } \
} while (0)
/* #region debug-point D:xq-pageguard
 * DEBUG: page-guard the xq/xd scratch. VirtualAlloc payload pages + one
 * PAGE_NOACCESS guard page; any over-write past the payload fails fast as an
 * AV at the exact kernel+offset (the 64B canary cannot catch >64B over-writes
 * that land in _aligned_malloc's alignment slack and then on heap metadata).
 * (VirtualAlloc/VirtualFree/DBG_* macros declared above with g_xq_pages.) */
static void *xq_alloc_canary(size_t bytes) {
    if (g_heap_check) {
        if (!heap_walk_ok()) {
            fprintf(stderr, "[HEAP-FAIL] walk xq_alloc pre l=%d\n", g_npu_layer);
            fflush(stderr); _exit(3);
        }
    }
    if (g_xq_stagger && bytes <= XQ_STAGGER_MAX_BYTES)
        bytes += (size_t)(g_npu_layer < 0 ? 0 : (g_npu_layer & 3)) * 64;
    if (g_xq_pages) {
        void *p = st_pg_alloc(bytes + g_xq_pad + 64);
        if (!p) return NULL;
        memset((uint8_t *)p + bytes + g_xq_pad, 0xA5, 64);
        return p;
    }
    void *p = st_aligned_alloc(bytes + g_xq_pad + 64, 64);
    if (p) memset((uint8_t *)p + bytes + g_xq_pad, 0xA5, 64);
    return p;
}
static void xq_free_canary(void *p, size_t bytes, const char *tag) {
    if (!p) return;
    if (g_heap_check) {
        if (!heap_walk_ok()) {
            fprintf(stderr, "[HEAP-FAIL] walk xq_free pre %s l=%d\n", tag, g_npu_layer);
            fflush(stderr); _exit(3);
        }
    }
    if (g_xq_stagger && bytes <= XQ_STAGGER_MAX_BYTES)
        bytes += (size_t)(g_npu_layer < 0 ? 0 : (g_npu_layer & 3)) * 64;
    const uint8_t *t = (const uint8_t *)p + bytes + g_xq_pad;
    for (int i = 0; i < 64; i++) {
        if (t[i] != 0xA5) {
            fprintf(stderr, "[XQ-CN] %s tail over-written +%d (pad=%zu)\n", tag, i, g_xq_pad);
            fflush(stderr);
            break;
        }
    }
    if (g_xq_pages) st_pg_free(p);
    else            st_aligned_free(p);
}
/* #endregion */

int st_qwen_model_prefill_batch(STQwenInferenceState *st,
                                  const int *token_ids, int n_tokens)
{
    g_xq_pad = xq_pad_env("VLLM_XQ_PAD");       /* DEBUG: canary gap (0 = exact) */
    g_wbuf_pad = xq_pad_env("VLLM_WBUF_PAD");   /* DEBUG: canary gap (0 = exact) */
    g_xq_pages = xq_pages_mode();     /* page-guard xq/xd (catch >64B over-writes) */
    g_xq_stagger = xq_stagger_env();  /* VLLM_XQ_STAGGER: LFH alloc staggering */
    g_heap_check = heap_check_env();  /* VLLM_HEAP_CHECK: step-granular heap probe */
    STModelWeights *w = &st->weights;
    /* Second WSCAN fence: after load + state init, before inference GEMMs. */
    npu_wscan("prefill-entry", w, st->cfg.dim, st->cfg.ffn_dim,
              w->n_layers_allocated);
    int d  = st->cfg.dim;
    int nl = w->n_layers_allocated;
    int nh = st->cfg.n_heads;
    int nkv = st->cfg.n_kv_heads;
    int hd = st->cfg.head_dim;
    int ff = st->cfg.ffn_dim;
    int vc = st->cfg.vocab_size;
    float eps = st->cfg.norm_eps;
    float theta = st->cfg.rope_theta;
    int q_rows = nh * hd;  /* 4096 */
    int kv_dim  = nkv * hd; /* 1024 */

    /* Mixed precision: --prefill-q8 forces this prefill GEMM onto Q8_0 while
     * decode keeps Q4_0 (see g_st_prefill_q8). */
    const int use_q4 = w->has_q4 && (!g_st_prefill_q8 || !w->q8_q_weight);
    st_default_threads();   /* RK3588: default 4x A76 (A55 degrades GEMM) */

    /* Phase timing accumulators (diagnostic): gemm = QKV+O+gate/up+down,
     * attn = attention, other = norms/MRoPE/KV-store/SiLU. */
    double t_gemm = 0.0, t_attn = 0.0, t_other = 0.0;
    double t_qkv = 0.0, t_o = 0.0, t_gu = 0.0, t_down = 0.0;
    double t0;
    double t_pf0 = st_now_sec();   /* prefill profile base (VLLM_DEC_PROF) */

    if (n_tokens <= 0 || !token_ids) return 0;
    if (!w->has_q8 && !w->has_q4) {
        /* Fallback: process token-by-token */
        fprintf(stderr, "[PREFILL] WARN: no quantized weights, token-by-token "
                        "fallback (%d tokens - this is slow)\n", n_tokens);
        for (int i = 0; i < n_tokens; i++)
            st_qwen_model_forward(st, token_ids[i]);
        return 0;
    }

    /* Phase 1.5: reset the prefill-importance accumulators. The exact packed
     * attention below records per-position attention mass (imp_head), reduced
     * into prefill_importance at the end. Sparse decode selection (when
     * --sparse-attn) ranks blocks by this real attention potential. */
    memset(st->imp_head, 0, (size_t)st->max_kv_slots * nh * sizeof(float));
    memset(st->prefill_importance, 0, (size_t)st->max_kv_slots * sizeof(float));

    /* Axiom: memory_bandwidth_reduction — workers are bound to physical
     * cores at pool creation (vllm_tp.c), so no per-region binding here. */

    int B = g_st_prefill_batch;
    int buf_hidden    = B * d;
    int buf_normed    = B * d;
    int buf_qk        = B * (q_rows > ff ? q_rows : ff); /* max of Q and gate/up */
    int buf_kv        = B * (kv_dim > ff ? kv_dim : ff);
    int buf_v         = B * kv_dim;
    int buf_attn      = B * q_rows;
    int buf_resid     = B * d;

    /* Partitioned activation scratch: each partition is its OWN canary-guarded
     * allocation so any intra-wbuf over-write (e.g. QKV writing past qbuf into
     * kbuf) trips a per-layer tail check instead of silently corrupting the
     * neighbor partition's data (which previously surfaced only as a delayed
     * heap Fastfail in an unrelated malloc). */
    uint8_t *tail_hidden = NULL, *tail_normed = NULL, *tail_qk = NULL, *tail_kv = NULL;
    uint8_t *tail_v = NULL, *tail_attn = NULL, *tail_resid = NULL;
    float *hidden_b = (float *)cf_aligned_canary((size_t)buf_hidden * sizeof(float), &tail_hidden);
    float *normed_b = (float *)cf_aligned_canary((size_t)buf_normed * sizeof(float), &tail_normed);
    float *qbuf_b   = (float *)cf_aligned_canary((size_t)buf_qk    * sizeof(float), &tail_qk);
    float *kbuf_b   = (float *)cf_aligned_canary((size_t)buf_kv    * sizeof(float), &tail_kv);
    float *vbuf_b   = (float *)cf_aligned_canary((size_t)buf_v     * sizeof(float), &tail_v);
    float *attn_b   = (float *)cf_aligned_canary((size_t)buf_attn  * sizeof(float), &tail_attn);
    float *resid_b  = (float *)cf_aligned_canary((size_t)buf_resid * sizeof(float), &tail_resid);
    if (!hidden_b || !normed_b || !qbuf_b || !kbuf_b || !vbuf_b || !attn_b || !resid_b) {
        /* OOM fallback: token-by-token */
        fprintf(stderr, "[PREFILL] WARN: wbuf OOM, token-by-token "
                        "fallback (%d tokens - very slow)\n", n_tokens);
        cf_aligned_free_canary(hidden_b); cf_aligned_free_canary(normed_b); cf_aligned_free_canary(qbuf_b);
        cf_aligned_free_canary(kbuf_b);   cf_aligned_free_canary(vbuf_b);   cf_aligned_free_canary(attn_b);
        cf_aligned_free_canary(resid_b);
        for (int i = 0; i < n_tokens; i++)
            st_qwen_model_forward(st, token_ids[i]);
        return 0;
    }
    memset(hidden_b, 0, (size_t)buf_hidden * sizeof(float));
    memset(normed_b, 0, (size_t)buf_normed * sizeof(float));
    memset(qbuf_b,   0, (size_t)buf_qk    * sizeof(float));
    memset(kbuf_b,   0, (size_t)buf_kv    * sizeof(float));
    memset(vbuf_b,   0, (size_t)buf_v     * sizeof(float));
    memset(attn_b,   0, (size_t)buf_attn  * sizeof(float));
    memset(resid_b,  0, (size_t)buf_resid * sizeof(float));

    /* Packed per-head K/V scratch, reused across layers. seq_stride is the
     * per-head row stride (= max_kv_slots) so the same buffer serves every
     * layer regardless of the current seq_len. */
    int seq_stride = st->max_kv_slots;
    size_t pack_elems = (size_t)nkv * (size_t)seq_stride * (size_t)hd;
    uint8_t *kpack_tail = NULL;
    float *k_pack = (float *)cf_aligned_canary(pack_elems * 2 * sizeof(float), &kpack_tail);
    if (!k_pack) {
        fprintf(stderr, "[PREFILL] WARN: k_pack OOM (need %zu MB), token-by-token "
                        "fallback (%d tokens - very slow)\n",
                pack_elems * 2 * sizeof(float) / 1048576, n_tokens);
        cf_aligned_free_canary(hidden_b); cf_aligned_free_canary(normed_b); cf_aligned_free_canary(qbuf_b);
        cf_aligned_free_canary(kbuf_b);   cf_aligned_free_canary(vbuf_b);   cf_aligned_free_canary(attn_b);
        cf_aligned_free_canary(resid_b);
        for (int i = 0; i < n_tokens; i++)
            st_qwen_model_forward(st, token_ids[i]);
        return 0;
    }
    float *v_pack = k_pack + pack_elems;

    /* Embed all tokens initially */
    float *emb = w->token_embed;
    if (!emb) {
        cf_aligned_free_canary(k_pack);
        cf_aligned_free_canary(hidden_b); cf_aligned_free_canary(normed_b); cf_aligned_free_canary(qbuf_b);
        cf_aligned_free_canary(kbuf_b);   cf_aligned_free_canary(vbuf_b);   cf_aligned_free_canary(attn_b);
        cf_aligned_free_canary(resid_b);
        return -1;
    }

    /* Process tokens in mini-batches through all layers */
    int tok_offset = 0;
    int last_nb = 0;
    while (tok_offset < n_tokens) {
        int nb = n_tokens - tok_offset;
        if (nb > B) nb = B;
        last_nb = nb;

        /* Embed this mini-batch's tokens */
        for (int t = 0; t < nb; t++) {
            float *h = hidden_b + (size_t)t * d;
            int tid = token_ids[tok_offset + t];
            for (int i = 0; i < d; i++)
                h[i] = st_emb_get(w, (size_t)tid, (size_t)i, d);
        }
        if (getenv("VLLM_PFDBG") && tok_offset == 0) {
            printf("[PFDBG] emb tid[0]=%d h[:8]=%.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g\n",
                   token_ids[0], hidden_b[0], hidden_b[1], hidden_b[2], hidden_b[3],
                   hidden_b[4], hidden_b[5], hidden_b[6], hidden_b[7]);
            fflush(stdout);
        }

        /* Pointers to this mini-batch's sub-regions */
        float *h_buf = hidden_b;
        float *nrm = normed_b;
        float *qb  = qbuf_b;
        float *kb  = kbuf_b;
        float *vb  = vbuf_b;
        float *att = attn_b;
        float *res = resid_b;

        /* Per-layer canary-tail checks: any write past a scratch partition's
         * logical end shows up here as [TAIL] (tail sits right after the data
         * when the debug pad env vars are 0). */
        #define CK_TAIL(TL, NAME, L, TOK) do { \
            if (TL) { \
                const uint8_t *_p = (const uint8_t *)(TL); \
                int _bad = -1; \
                for (int _i = 0; _i < 64; _i++) { \
                    if (_p[_i] != 0xA5) { _bad = _i; break; } \
                } \
                if (_bad >= 0) { \
                    fprintf(stderr, "[TAIL] %s broken +%d l=%d tok=%d\n", \
                            (NAME), _bad, (L), (TOK)); \
                    fflush(stderr); \
                } \
            } \
        } while (0)

        for (int l = 0; l < nl; l++) {
            g_npu_layer = l;   /* transparent NPU offload: per-layer op models */
            t0 = st_now_sec();
            /* --- Step 1: batched RMSNorm (ONE omp region, was nb x 2 regions) --- */
            dyn_rms_norm_batch(nrm, h_buf, w->attn_norm + (size_t)l * d, nb, d, eps);
            t_other += st_now_sec() - t0;

            t0 = st_now_sec();
            /* --- Step 2: Batched QKV projection --- */
            if (use_q4 && w->q4_q_weight) {
                const uint8_t *q4q = w->q4_q_weight + Q4_BYTES((size_t)l * d * q_rows);
                const uint8_t *q4k = w->q4_k_weight + Q4_BYTES((size_t)l * d * kv_dim);
                const uint8_t *q4v = w->q4_v_weight + Q4_BYTES((size_t)l * d * kv_dim);
                dyn_matvec_q4_q8_fused_qkv_batched(qb, kb, vb, q4q, q4k, q4v,
                                                    nrm, q_rows, kv_dim, d, nb,
                                                    w->has_x8 ? w->x8_q_weight + Q4_BYTES((size_t)l * d * q_rows) : NULL,
                                                    w->has_x8 ? w->x8_k_weight + Q4_BYTES((size_t)l * d * kv_dim) : NULL,
                                                    w->has_x8 ? w->x8_v_weight + Q4_BYTES((size_t)l * d * kv_dim) : NULL);
            } else {
                const uint8_t *q8q = w->q8_q_weight + Q8_BYTES((size_t)l * d * q_rows);
                const uint8_t *q8k = w->q8_k_weight + Q8_BYTES((size_t)l * d * kv_dim);
                const uint8_t *q8v = w->q8_v_weight + Q8_BYTES((size_t)l * d * kv_dim);
                dyn_matvec_q8_fused_qkv_batched(qb, kb, vb, q8q, q8k, q8v,
                                                 nrm, q_rows, kv_dim, d, nb);
            }
            t_qkv += st_now_sec() - t0;
            HCK("qkv");

            if (getenv("VLLM_PFDBG")) {
                float qm = 0.0f, km = 0.0f, vm = 0.0f;
                int qi = 0, ki = 0, vi = 0;
                for (int i = 0; i < nb * q_rows; i++) { float a = fabsf(qb[i]); if (a > qm) { qm = a; qi = i; } }
                for (int i = 0; i < nb * kv_dim; i++) { float a = fabsf(kb[i]); if (a > km) { km = a; ki = i; } }
                for (int i = 0; i < nb * kv_dim; i++) { float a = fabsf(vb[i]); if (a > vm) { vm = a; vi = i; } }
                printf("[PFDBG] l=%d QKV qmax=%.6g@%d kmax=%.6g@%d vmax=%.6g@%d\n",
                       l, qm, qi, km, ki, vm, vi);
                fflush(stdout);
            }

            /* --- Step 3-6: batched Q/K norms, per-token MRoPE + KV-cache --- */
            /* Batch both head-norms into ONE omp region (was nb sequential
             * fork-joins, 2 regions per token). Each (t,head) task runs the
             * identical math as before, so results stay bit-identical. */
            {
                const float *qw = w->q_norm ? w->q_norm + (size_t)l * hd : NULL;
                const float *kw = w->k_norm ? w->k_norm + (size_t)l * hd : NULL;
                int nq = w->q_norm ? nh : 0;
                int nk = w->k_norm ? nkv : 0;
                int total = nb * (nq + nk);
                vllm_qk_norm_batch_ctx vc = { qb, kb, qw, kw, nq, nk, hd, q_rows, kv_dim };
                vllm_tp_parfor(0, total, vllm_qk_norm_batch_worker, &vc);
            }
            for (int t = 0; t < nb; t++) {
                /* Speculative-verify mode positions the draft with the same
                 * base the decode step uses (mrope_pos for multimodal, seq_len
                 * for text-only) so the draft KV rotation matches what the
                 * accepted tokens would have produced in normal decode. */
                int pos = (g_verify_logits && g_verify_pos >= 0)
                          ? g_verify_pos + tok_offset + t
                          : st->seq_len + tok_offset + t;
                float *qt = qb + (size_t)t * q_rows;
                float *kt = kb + (size_t)t * kv_dim;
                float *vt = vb + (size_t)t * kv_dim;
                float *at = att + (size_t)t * q_rows;

                t0 = st_now_sec();

                /* MRoPE */
                if (st->cfg.has_mrope) {
                    dyn_mrope(qt, kt, hd, nh, nkv, st->cfg.head_dim_full, pos, theta);
                } else {
                    const float *freqs = rope_freq_table(hd / 2, theta);
                    for (int i = 0; i < hd; i += 2) {
                        float freq = freqs ? freqs[i / 2] : 1.0f / powf(theta, (float)i / (float)hd);
                        float angle = (float)pos * freq;
                        float ca = cosf(angle);
                        float sa = sinf(angle);
                        for (int h = 0; h < nh; h++) {
                            int base = h * hd;
                            float v0 = qt[base + i];
                            float v1 = qt[base + i + 1];
                            qt[base + i]     = v0 * ca - v1 * sa;
                            qt[base + i + 1] = v1 * ca + v0 * sa;
                        }
                        for (int h = 0; h < nkv; h++) {
                            int base = h * hd;
                            float v0 = kt[base + i];
                            float v1 = kt[base + i + 1];
                            kt[base + i]     = v0 * ca - v1 * sa;
                            kt[base + i + 1] = v1 * ca + v0 * sa;
                        }
                    }
                }

                /* KV-cache store (float + compressed payload for attention) */
                int cl = st->cache_len[l];
                if (cl >= st->kv_n_blocks * st->kv_bs)
                    fprintf(stderr, "[KV-OOB] prefill cl=%d cap=%d l=%d tok=%d\n", cl,
                            st->kv_n_blocks * st->kv_bs, l, tok_offset + t);
                memcpy(kv_row_f32(st->k_cache[l], cl, kv_dim, st->kv_bs), kt, kv_dim * sizeof(float));
                memcpy(kv_row_f32(st->v_cache[l], cl, kv_dim, st->kv_bs), vt, kv_dim * sizeof(float));
                if (KV_GUARD) {
                    static int _g1 = 0;
                    if (!_g1) {
                        int _blk = cl / st->kv_bs;
                        const uint8_t *_bg = (const uint8_t *)st->k_cache[l][_blk]
                            + (size_t)st->kv_bs * kv_dim * sizeof(float);
                        for (int _i = 0; _i < KV_GUARD; _i += 64)
                            if (_bg[_i] != 0xA5) {
                                fprintf(stderr, "[GUARD] prefill-k-back l=%d cl=%d blk=%d off=%d\n",
                                        l, cl, _blk, _i); fflush(stderr); _g1 = 1; break;
                            }
                        if (!_g1 && _blk + 1 < st->kv_n_blocks) {
                            const uint8_t *_fg = (const uint8_t *)st->k_cache[l][_blk + 1] - KV_GUARD;
                            for (int _i = 0; _i < KV_GUARD; _i += 64)
                                if (_fg[_i] != 0xA5) {
                                    fprintf(stderr, "[GUARD] prefill-k-front l=%d cl=%d blk=%d off=%d\n",
                                            l, cl, _blk, _i); fflush(stderr); _g1 = 1; break;
                                }
                        }
                    }
                }
                if (st->use_kv_q8) {
                    int8_t *kdst = kv_row_i8(st->k_cache_q8[l], cl, kv_dim, st->kv_bs);
                    int8_t *vdst = kv_row_i8(st->v_cache_q8[l], cl, kv_dim, st->kv_bs);
                    kv_quantize_per_head(kdst, vdst,
                                         st->k_scale[l] + (size_t)cl * nkv,
                                         st->v_scale[l] + (size_t)cl * nkv,
                                         kt, vt, nkv, hd);
                } else if (st->use_kv_q4) {
                    q4_pack_token(q4_k_row(st->k_cache_q4[l], cl, 0, hd, st->kv_bs, nkv),
                                  q4_v_row(st->v_cache_q4[l], cl, 0, hd, st->kv_bs, nkv),
                                  kt, vt, nkv, hd);
                }
                st->cache_len[l]++;

                t_other += st_now_sec() - t0;
            }
            HCK("kvstore");

            /* --- Step 6 (batched exact attention): pack K/V per head once, then
             * score all nb queries against the packed cache. --- */
            {
                t0 = st_now_sec();
                float scale = 1.0f / sqrtf((float)hd);
                int seq_len = st->cache_len[l];
                if (seq_len > seq_stride)
                    fprintf(stderr, "[KV-OOB] pack seq_len=%d > stride=%d l=%d\n",
                            seq_len, seq_stride, l);
                int prev_len = seq_len - nb;
                st_pack_kv_heads(st->k_cache[l], st->v_cache[l],
                                 k_pack, v_pack, seq_len, nkv, hd, kv_dim,
                                 seq_stride, st->kv_bs);
                /* Optimized path: with sparse decode attention enabled,
                 * prefill also runs sparse batched attention over the top-k
                 * KV blocks per head (probe + representative-query selection).
                 * It still records the per-token attention mass into imp_head
                 * (over the selected blocks only), keeping the sparse decode
                 * selection signal consistent. */
                if (g_sparse_attn && seq_len > g_sparse_block * 2) {
                    st_attn_batched_packed_sparse(att, qb, k_pack, v_pack, nb, prev_len,
                                                  seq_stride, nh, nkv, hd, scale,
                                                  st->scores_buf, st->max_kv_slots,
                                                  g_sparse_block, g_sparse_k, g_sparse_probe,
                                                  st->imp_head);
                } else {
                    st_attn_batched_packed(att, qb, k_pack, v_pack, nb, prev_len,
                                           seq_stride, nh, nkv, hd, scale,
                                           st->scores_buf, st->max_kv_slots,
                                           st->imp_head);
                }
                GUARD_CHK(g_cn_imphead, imphead, "imp_head", l, tok_offset + nb);
                GUARD_CHK(g_cn_scores, scores2, "scores_buf", l, tok_offset + nb);
                t_attn += st_now_sec() - t0;
            }
            HCK("attn");

            t0 = st_now_sec();
            /* --- Step 7: Batched O projection + residual --- */
            /* Save residuals first */
            for (int t = 0; t < nb; t++) {
                float *r  = res + (size_t)t * d;
                float *h  = h_buf + (size_t)t * d;
                memcpy(r, h, d * sizeof(float));
            }
            if (use_q4 && w->q4_o_weight) {
                const uint8_t *q4o = w->q4_o_weight + Q4_BYTES((size_t)l * q_rows * d);
                dyn_matvec_q4_q8_fused_o_residual_batched(h_buf, res, q4o, att, d, q_rows, nb,
                    w->has_x8 ? w->x8_o_weight + Q4_BYTES((size_t)l * q_rows * d) : NULL);
            } else {
                const uint8_t *q8o = w->q8_o_weight + Q8_BYTES((size_t)l * q_rows * d);
                dyn_matvec_q8_fused_o_residual_batched(h_buf, res, q8o, att, d, q_rows, nb);
            }
            t_o += st_now_sec() - t0;
            HCK("o");

            if (getenv("VLLM_PFDBG") && l == 6) {
                float hm = 0.0f;
                for (int i = 0; i < nb * d; i++) { float a = fabsf(h_buf[i]); if (a > hm) hm = a; }
                printf("[PFDBG] l=6 after-O h[:8]=%.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g hmmax=%.6g\n",
                       h_buf[0], h_buf[1], h_buf[2], h_buf[3],
                       h_buf[4], h_buf[5], h_buf[6], h_buf[7], hm);
                fflush(stdout);
            }

            t0 = st_now_sec();
            /* --- Step 8: batched RMSNorm (FFN) --- */
            dyn_rms_norm_batch(nrm, h_buf, w->ffn_norm + (size_t)l * d, nb, d, eps);
            t_other += st_now_sec() - t0;

            t0 = st_now_sec();
            /* --- Step 9: Batched Gate/Up projection --- */
            if (st_ffn_q2_layer(w, l)) {
                const uint8_t *q2g = w->q2_gate_weight + Q2_BYTES((size_t)l * ff * d);
                const uint8_t *q2u = w->q2_up_weight   + Q2_BYTES((size_t)l * ff * d);
                dyn_matvec_q2_q8_fused_gate_up_batched(qb, kb, q2g, q2u, nrm, ff, d, nb);
            } else if (use_q4 && w->q4_gate_weight) {
                const uint8_t *q4g = w->q4_gate_weight + Q4_BYTES((size_t)st_ffn_q4_layer_off(w, l) * ff * d);
                const uint8_t *q4u = w->q4_up_weight   + Q4_BYTES((size_t)st_ffn_q4_layer_off(w, l) * ff * d);
                const uint8_t *x8g = w->x8_gate_weight
                    ? w->x8_gate_weight + Q4_BYTES((size_t)st_ffn_q4_layer_off(w, l) * ff * d) : NULL;
                const uint8_t *x8u = w->x8_up_weight
                    ? w->x8_up_weight   + Q4_BYTES((size_t)st_ffn_q4_layer_off(w, l) * ff * d) : NULL;
                dyn_matvec_q4_q8_fused_gate_up_batched(qb, kb, q4g, q4u, nrm, ff, d, nb, x8g, x8u);
            } else {
                #define Q8O(el)  ((size_t)(el) / 32 * 34)
                const uint8_t *q8g = w->q8_gate_weight + Q8O((size_t)l * ff * d);
                const uint8_t *q8u = w->q8_up_weight   + Q8O((size_t)l * ff * d);
                dyn_matvec_q8_fused_gate_up_batched(qb, kb, q8g, q8u, nrm, ff, d, nb);
                #undef Q8O
            }
            t_gu += st_now_sec() - t0;
            HCK("gateup");

            if (getenv("VLLM_PFDBG") && l == 6) {
                float gm = 0.0f, um = 0.0f;
                for (int i = 0; i < nb * ff; i++) {
                    float a = fabsf(qb[i]); if (a > gm) gm = a;
                    float b = fabsf(kb[i]); if (b > um) um = b;
                }
                printf("[PFDBG] l=6 gate[:8]=%.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g gmax=%.6g\n",
                       qb[0], qb[1], qb[2], qb[3], qb[4], qb[5], qb[6], qb[7], gm);
                printf("[PFDBG] l=6 up[:8]=%.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g umax=%.6g\n",
                       kb[0], kb[1], kb[2], kb[3], kb[4], kb[5], kb[6], kb[7], um);
                fflush(stdout);
            }

            t0 = st_now_sec();
            /* --- Step 10: batched SiLU (ONE pool region over nb*ff) --- */
            vllm_silu_mul_ctx vg = { qb, kb, nb * ff };
            vllm_tp_parfor(0, (nb * ff + 15) / 16, vllm_silu_mul_worker, &vg);
            t_other += st_now_sec() - t0;

            if (getenv("VLLM_PFDBG") && l == 6) {
                float am = 0.0f; int ai = 0; float nm = 0.0f; int ni = 0;
                for (int i = 0; i < nb * ff; i++) { float a = fabsf(qb[i]); if (a > am) { am = a; ai = i; } }
                for (int i = 0; i < nb * d; i++)  { float a = fabsf(nrm[i]); if (a > nm) { nm = a; ni = i; } }
                printf("[PFDBG] l=6 activated max=%.6g@%d(t=%d,f=%d) nrm max=%.6g@%d nrm[:4]=%.6g %.6g %.6g %.6g\n",
                       am, ai, ai / ff, ai % ff, nm, ni, nrm[0], nrm[1], nrm[2], nrm[3]);
                fflush(stdout);
            }

            t0 = st_now_sec();
            /* --- Step 11: Batched Down projection + residual --- */
            /* Save residuals */
            for (int t = 0; t < nb; t++) {
                float *r = res + (size_t)t * d;
                float *h = h_buf + (size_t)t * d;
                memcpy(r, h, d * sizeof(float));
            }
            if (st_ffn_q2_layer(w, l)) {
                const uint8_t *q2d = w->q2_down_weight + Q2_BYTES((size_t)l * d * ff);
                dyn_matvec_q2_q8_fused_down_residual_batched(h_buf, res, q2d, qb, d, ff, nb);
            } else if (use_q4 && w->q4_down_weight) {
                const uint8_t *q4d = w->q4_down_weight + Q4_BYTES((size_t)st_ffn_q4_layer_off(w, l) * d * ff);
                dyn_matvec_q4_q8_fused_down_residual_batched(h_buf, res, q4d, qb, d, ff, nb,
                    w->has_x8 ? w->x8_down_weight + Q4_BYTES((size_t)st_ffn_q4_layer_off(w, l) * d * ff) : NULL);
            } else {
                #define Q8O(el)  ((size_t)(el) / 32 * 34)
                const uint8_t *q8d = w->q8_down_weight + Q8O((size_t)l * d * ff);
                dyn_matvec_q8_fused_down_residual_batched(h_buf, res, q8d, qb, d, ff, nb);
                #undef Q8O
            }
            t_down += st_now_sec() - t0;
            HCK("down");

            if (getenv("VLLM_PFDBG")) {
                float hm = 0.0f; int hn = 0; int hmi = 0;
                for (int i = 0; i < nb * d; i++) {
                    float a = fabsf(h_buf[i]);
                    if (a > hm) { hm = a; hmi = i; }
                    if (npu_isnan_f32(h_buf[i]) || npu_isinf_f32(h_buf[i])) hn++;
                }
                printf("[PFDBG] l=%d hidden max=%.6g@%d nan/inf=%d h0=%.4g h1=%.4g h2=%.4g h3=%.4g\n",
                       l, hm, hmi, hn, h_buf[0], h_buf[1], h_buf[2], h_buf[3]);
                fflush(stdout);
            }

            if (prefill_debug_mode()) {
                CK_TAIL(tail_hidden, "hidden", l, tok_offset + nb);
                CK_TAIL(tail_normed, "normed", l, tok_offset + nb);
                CK_TAIL(tail_qk, "qbuf", l, tok_offset + nb);
                CK_TAIL(tail_kv, "kbuf", l, tok_offset + nb);
                CK_TAIL(tail_v, "vbuf", l, tok_offset + nb);
                CK_TAIL(tail_attn, "attn", l, tok_offset + nb);
                CK_TAIL(tail_resid, "resid", l, tok_offset + nb);
                CK_TAIL(kpack_tail, "kpack", l, tok_offset + nb);

                {
                    int nblk = st->kv_n_blocks;
                    size_t kv_dim = st->cfg.n_kv_heads * st->cfg.head_dim;
                    uint8_t *at = (uint8_t *)st->k_cache[l] + (size_t)nblk * sizeof(float *);
                    int bad = -1;
                    for (int i = 0; i < 64; i++) if (at[i] != 0xA5) { bad = i; break; }
                    if (bad >= 0) fprintf(stderr, "[CANARY] k arr tail l=%d +%d tok=%d\n", l, bad, tok_offset + nb);
                    at = (uint8_t *)st->v_cache[l] + (size_t)nblk * sizeof(float *);
                    bad = -1;
                    for (int i = 0; i < 64; i++) if (at[i] != 0xA5) { bad = i; break; }
                    if (bad >= 0) fprintf(stderr, "[CANARY] v arr tail l=%d +%d tok=%d\n", l, bad, tok_offset + nb);
                    float *lk = st->k_cache[l][nblk - 1];
                    if (lk) {
                        uint8_t *bt = (uint8_t *)lk + (size_t)st->kv_bs * kv_dim * sizeof(float);
                        bad = -1;
                        for (int i = 0; i < 64; i++) if (bt[i] != 0xA5) { bad = i; break; }
                        if (bad >= 0) fprintf(stderr, "[CANARY] k last blk tail l=%d +%d tok=%d\n", l, bad, tok_offset + nb);
                    }
                    float *lv = st->v_cache[l][nblk - 1];
                    if (lv) {
                        uint8_t *bt = (uint8_t *)lv + (size_t)st->kv_bs * kv_dim * sizeof(float);
                        bad = -1;
                        for (int i = 0; i < 64; i++) if (bt[i] != 0xA5) { bad = i; break; }
                        if (bad >= 0) fprintf(stderr, "[CANARY] v last blk tail l=%d +%d tok=%d\n", l, bad, tok_offset + nb);
                    }
                }
            }
        }

        tok_offset += nb;
        /* Progress is always visible for long prefills: without it a slow
         * O(S^2) prefill looks like a hang (S=12288/16384 can take minutes). */
        fprintf(stderr, "[PREFILL] %d/%d tokens done\n", tok_offset, n_tokens);
        fflush(stderr);

        /* #region debug-point B:prefill-heapval
         * Whole-heap integrity at mini-batch granularity. Replaces the old
         * once-guarded _heapchk (which infinite-loops on a corrupted LFH free
         * list). HeapValidate returns FALSE instead; the first corrupt block
         * is printed and the run stops (_exit: no CRT teardown, so the
         * corrupted heap cannot fail-fast first). "enter" is printed before
         * each walk: if HeapValidate itself hangs, the last printed block is
         * the corruption window. On aarch64/Linux st_heapchk is a no-op. */
        {
            if (prefill_debug_mode() && pb_heap_mode() && n_tokens > 3000) {
                fprintf(stderr, "[PB-HEAP] check tok=%d\n", tok_offset); fflush(stderr);
                if (st_heapchk() != ST_HEAP_OK) {
                    fprintf(stderr, "[PB-HEAP] broken at tok=%d\n", tok_offset);
                    fflush(stderr);
                    _exit(3);
                }
            }
        }
        /* #endregion */
    }
    #undef CK_TAIL

    if (getenv("VLLM_DEC_PROF")) {
        double t_pf_total = (st_now_sec() - t_pf0) * 1e3;
        fprintf(stderr, "[PF-PROF] tokens=%d qkv=%.2fms/t attn=%.2f o=%.2f gu=%.2f down=%.2f other=%.2f total=%.2fms\n",
                n_tokens, t_qkv / n_tokens * 1e3, t_attn / n_tokens * 1e3,
                t_o / n_tokens * 1e3, t_gu / n_tokens * 1e3, t_down / n_tokens * 1e3,
                t_other / n_tokens * 1e3, t_pf_total);
        fflush(stderr);
    }

    /* Phase 1.5: reduce per-head attention mass into the per-token importance
     * potential consumed by sparse decode selection. O(nh * seq_len), done once
     * per prefill. Deterministic (serial, fixed order). */
    {
        const size_t stride = (size_t)st->max_kv_slots;
        for (int s = 0; s < st->seq_len + n_tokens; s++) {
            float acc = 0.0f;
            for (int ha = 0; ha < nh; ha++) acc += st->imp_head[(size_t)ha * stride + s];
            st->prefill_importance[s] = acc;
        }
    }

    /* Final norm + LM head for the LAST token of the last mini-batch.
     * Speculative-verify mode (g_verify_logits set): the draft is a single
     * mini-batch (K <= B), so compute lm_head logits for EVERY position into
     * g_verify_logits[pos*vocab] and record each position's top-1 token in
     * g_verify_pred; st->logits keeps the last position's logits (unchanged
     * contract). */
    if (g_verify_logits) {
        /* The verify hook only covers the LAST mini-batch; require the draft
         * to fit in one batch so every position's logits are captured. */
        g_verify_invalid = (last_nb != n_tokens) || (n_tokens > SPEC_DRAFT_MAX);
        if (g_verify_invalid) {
            float *last_hidden = hidden_b + (size_t)(last_nb - 1) * d;
            dyn_rms_norm(normed_b, last_hidden, w->final_norm, d, eps);
            if (use_q4 && w->q4_lm_weight)
                dyn_matvec_q4_q8(st->logits, w->q4_lm_weight, normed_b, vc, d);
            else
                dyn_matvec_q8(st->logits, w->q8_lm_weight, normed_b, vc, d);
        } else {
            for (int i = 0; i < last_nb && i < SPEC_DRAFT_MAX; i++) {
                float *vlog = g_verify_logits + (size_t)i * vc;
                float *vh = hidden_b + (size_t)i * d;
                dyn_rms_norm(normed_b, vh, w->final_norm, d, eps);
                if (use_q4 && w->q4_lm_weight)
                    dyn_matvec_q4_q8(vlog, w->q4_lm_weight, normed_b, vc, d);
                else
                    dyn_matvec_q8(vlog, w->q8_lm_weight, normed_b, vc, d);
                int best = 0, second = 0;
                float bv = -1.0e30f, sv = -1.0e30f;
                for (int t = 0; t < vc; t++)
                    if (vlog[t] > bv) { sv = bv; second = best; bv = vlog[t]; best = t; }
                    else if (vlog[t] > sv) { sv = vlog[t]; second = t; }
                g_verify_pred[i] = best;
                g_verify_margin[i] = bv - sv;
            }
            memcpy(st->logits, g_verify_logits + (size_t)(last_nb - 1) * vc,
                   (size_t)vc * sizeof(float));
        }
    } else {
        float *last_hidden = hidden_b + (size_t)(last_nb - 1) * d;
        dyn_rms_norm(normed_b, last_hidden, w->final_norm, d, eps);
        if (use_q4 && w->q4_lm_weight)
            dyn_matvec_q4_q8(st->logits, w->q4_lm_weight, normed_b, vc, d);
        else
            dyn_matvec_q8(st->logits, w->q8_lm_weight, normed_b, vc, d);
    }

    {
        t_gemm = t_qkv + t_o + t_gu + t_down;
        double t_total = t_gemm + t_attn + t_other;
        if (t_total > 0.0) {
            fprintf(stderr,
                "[PREFILL-TIMING] n=%d total=%.1fms | GEMM=%.1fms(%4.1f%%) "
                "ATTN=%.1fms(%4.1f%%) OTHER=%.1fms(%4.1f%%)\n",
                n_tokens, t_total * 1e3,
                t_gemm * 1e3, 100.0 * t_gemm / t_total,
                t_attn * 1e3, 100.0 * t_attn / t_total,
                t_other * 1e3, 100.0 * t_other / t_total);
            if (t_gemm > 0.0) {
                fprintf(stderr,
                    "[PREFILL-KERNELS] QKV=%.1fms(%4.1f%%) O=%.1fms(%4.1f%%) "
                    "GATEUP=%.1fms(%4.1f%%) DOWN=%.1fms(%4.1f%%)\n",
                    t_qkv * 1e3, 100.0 * t_qkv / t_gemm,
                    t_o   * 1e3, 100.0 * t_o   / t_gemm,
                    t_gu  * 1e3, 100.0 * t_gu  / t_gemm,
                    t_down* 1e3, 100.0 * t_down/ t_gemm);
                fflush(stderr);
            }
            fflush(stderr);
        }
    }

    st->seq_len += n_tokens;
    cf_aligned_free_canary(k_pack);
    cf_aligned_free_canary(hidden_b); cf_aligned_free_canary(normed_b); cf_aligned_free_canary(qbuf_b);
    cf_aligned_free_canary(kbuf_b);   cf_aligned_free_canary(vbuf_b);   cf_aligned_free_canary(attn_b);
    cf_aligned_free_canary(resid_b);
    return 0;
}

/* ================================================================
 * Multimodal Prefill: Image/Video + Text
 *
 * Integrates visual tokens from the vision encoder into the LLM
 * input by replacing <image_pad> / <video_pad> placeholder tokens
 * in the prompt with the corresponding visual embeddings.
 *
 * Visual token IDs:
 *   <image_pad> = 151655 → replaced with visual_tokens[i]
 *   <video_pad> = 151656 → replaced with visual_tokens[i]
 *
 * MRoPE: For visual tokens, we assign sequential 3D positions:
 *   - Image:  pos_t=0, pos_h=grid_y, pos_w=grid_x
 *   - Video:  pos_t=temporal_idx, pos_h=grid_y, pos_w=grid_x
 *   - Text:   pos_t=pos, pos_h=pos, pos_w=pos (degenerate to 1D RoPE)
 * ================================================================ */

typedef struct {
    float *qt, *kt; const float *cos_tab, *sin_tab;
    int half, hd, nh, nkv;
} vllm_mrope_heads_ctx;

static void vllm_mrope_heads_worker(void *ctx_, int qk) {
    vllm_mrope_heads_ctx *c = ctx_;
    if (qk < c->nh) {
        float *qh = c->qt + qk * c->hd;
        for (int j = 0; j < c->half; j++) {
            float co = c->cos_tab[j], si = c->sin_tab[j];
            float q0 = qh[j], q1 = qh[j + c->half];
            qh[j]         = q0 * co - q1 * si;
            qh[j + c->half]  = q1 * co + q0 * si;
        }
    } else {
        float *kh = c->kt + (qk - c->nh) * c->hd;
        for (int j = 0; j < c->half; j++) {
            float co = c->cos_tab[j], si = c->sin_tab[j];
            float k0 = kh[j], k1 = kh[j + c->half];
            kh[j]         = k0 * co - k1 * si;
            kh[j + c->half]  = k1 * co + k0 * si;
        }
    }
}

typedef struct {
    STQwenInferenceState *st;
    float *qt, *at;
    int nh, nkv, hd, hd8, l, seq_len, kv_dim, mkv;
    float scale;
} vllm_mm_attn_ctx;

static void vllm_mm_attn_i8_worker(void *ctx_, int ha) {
    vllm_mm_attn_ctx *c = ctx_;
    STQwenInferenceState *st = c->st;
    int kh = (ha * c->nkv) / c->nh;
    float max_score = -1e9f;
    const float *qs = c->qt + ha * c->hd;
    float *scores = st->scores_buf + (size_t)ha * c->mkv;
    const float *ksc = st->k_scale[c->l] + kh;
    const float *vsc = st->v_scale[c->l] + kh;

    for (int s = 0; s < c->seq_len; s++) {
        const int8_t *ks = kv_row_i8(st->k_cache_q8[c->l], s, c->kv_dim, st->kv_bs) + (size_t)kh * c->hd;
        float ksc_s = ksc[(size_t)s * c->nkv] * KVQ_SCALE;
        float dot = 0.0f;
        for (int i = 0; i < c->hd; i++) dot += qs[i] * (float)ks[i];
        dot *= ksc_s;
        scores[s] = dot * c->scale;
        if (scores[s] > max_score) max_score = scores[s];
    }

    float sum_exp = 0.0f;
    for (int s = 0; s < c->seq_len; s++) {
        scores[s] = expf(scores[s] - max_score);
        sum_exp += scores[s];
    }
    float inv_sum = 1.0f / sum_exp;

    float *o = c->at + ha * c->hd;
    for (int i = 0; i < c->hd; i++) o[i] = 0.0f;

    for (int s = 0; s < c->seq_len; s++) {
        float wgt = scores[s] * inv_sum;
        const int8_t *vs = kv_row_i8(st->v_cache_q8[c->l], s, c->kv_dim, st->kv_bs) + (size_t)kh * c->hd;
        float vsc_s = vsc[(size_t)s * c->nkv] * KVQ_SCALE;
        for (int i = 0; i < c->hd; i++)
            o[i] += wgt * (float)vs[i] * vsc_s;
    }
}

static void vllm_mm_attn_q4_worker(void *ctx_, int ha) {
    vllm_mm_attn_ctx *c = ctx_;
    STQwenInferenceState *st = c->st;
    int kh = (ha * c->nkv) / c->nh;
    float max_score = -1e9f;
    const float *qs = c->qt + ha * c->hd;
    float *scores = st->scores_buf + (size_t)ha * c->mkv;

    for (int s = 0; s < c->seq_len; s++) {
        const uint8_t *kp = q4_k_row(st->k_cache_q4[c->l], s, kh, c->hd, st->kv_bs, c->nkv);
        float dot = q4_head_dot(kp, qs, c->hd);
        scores[s] = dot * c->scale;
        if (scores[s] > max_score) max_score = scores[s];
    }

    float sum_exp = 0.0f;
    for (int s = 0; s < c->seq_len; s++) {
        scores[s] = expf(scores[s] - max_score);
        sum_exp += scores[s];
    }
    float inv_sum = 1.0f / sum_exp;

    float *o = c->at + ha * c->hd;
    for (int i = 0; i < c->hd; i++) o[i] = 0.0f;

    for (int s = 0; s < c->seq_len; s++) {
        float wgt = scores[s] * inv_sum;
        const uint8_t *vp = q4_v_row(st->v_cache_q4[c->l], s, kh, c->hd, st->kv_bs, c->nkv);
        q4_head_vacc(o, vp, wgt, c->hd);
    }
}

static void vllm_mm_attn_flash_worker(void *ctx_, int ha) {
    vllm_mm_attn_ctx *c = ctx_;
    STQwenInferenceState *st = c->st;
    int kh = (ha * c->nkv) / c->nh;
    flash_attn_blocked_single_q(
        c->at + ha * c->hd, c->qt + ha * c->hd,
        st->k_cache[c->l], st->v_cache[c->l],
        c->seq_len, kh * c->hd, c->kv_dim, c->hd, st->kv_bs, c->scale);
}

static void vllm_mm_attn_f32_worker(void *ctx_, int ha) {
    vllm_mm_attn_ctx *c = ctx_;
    STQwenInferenceState *st = c->st;
    int kh = (ha * c->nkv) / c->nh;
    float max_score = -1e9f;
    const float *qs = c->qt + ha * c->hd;
    float *scores = st->scores_buf + (size_t)ha * c->mkv;

    for (int s = 0; s < c->seq_len; s++) {
        const float *ks = kv_row_f32(st->k_cache[c->l], s, c->kv_dim, st->kv_bs) + (size_t)kh * c->hd;
        float dot = 0.0f;
        for (int i = 0; i < c->hd; i++) dot += qs[i] * ks[i];
        scores[s] = dot * c->scale;
        if (scores[s] > max_score) max_score = scores[s];
    }

    float sum_exp = 0.0f;
    for (int s = 0; s < c->seq_len; s++) {
        scores[s] = expf(scores[s] - max_score);
        sum_exp += scores[s];
    }
    float inv_sum = 1.0f / sum_exp;

    float *o = c->at + ha * c->hd;
    for (int i = 0; i < c->hd; i++) o[i] = 0.0f;

    for (int s = 0; s < c->seq_len; s++) {
        float wgt = scores[s] * inv_sum;
        const float *vs = kv_row_f32(st->v_cache[c->l], s, c->kv_dim, st->kv_bs) + (size_t)kh * c->hd;
        for (int i = 0; i < c->hd; i++)
            o[i] += wgt * vs[i];
    }
}

typedef struct {
    float *h_buf; const float *ds; const int *vis_row_map;
    int tok_offset, nb, d;
} vllm_ds_ctx;

static void vllm_ds_worker(void *ctx_, int t) {
    vllm_ds_ctx *c = ctx_;
    int vi = c->vis_row_map[c->tok_offset + t];
    if (vi >= 0) {
        float *h = c->h_buf + (size_t)t * c->d;
        const float *drow = c->ds + (size_t)vi * c->d;
        for (int j = 0; j < c->d; j++) h[j] += drow[j];
    }
}

int st_qwen_model_multimodal_prefill_ex(
    STQwenInferenceState *st,
    const int *token_ids, int n_tokens,
    const float *visual_tokens, int n_vis_tokens,
    const int *grids, int n_regions,  /* grids: [n_regions][3] = {grid_t, grid_h, grid_w} */
    const float *const *ds_features, int n_ds)  /* optional DeepStack [n_ds][n_vis_tokens, dim] */
{
    STModelWeights *w = &st->weights;
    int d  = st->cfg.dim;
    int nl = w->n_layers_allocated;
    int nh = st->cfg.n_heads;
    int nkv = st->cfg.n_kv_heads;
    int hd = st->cfg.head_dim;
    int ff = st->cfg.ffn_dim;
    int vc = st->cfg.vocab_size;
    float eps = st->cfg.norm_eps;
    float theta = st->cfg.rope_theta;
    int q_rows = nh * hd;
    int kv_dim  = nkv * hd;

    /* Mixed precision: --prefill-q8 forces this prefill GEMM onto Q8_0. */
    const int use_q4 = w->has_q4 && (!g_st_prefill_q8 || !w->q8_q_weight);

    if (n_tokens <= 0 || !token_ids) return 0;
    if (!w->has_q8 && !w->has_q4) {
        fprintf(stderr, "[MM] quantized weights required for multimodal prefill\n");
        return -1;
    }
    if (getenv("VLLM_MMDBG")) {
        int np = 0;
        for (int i = 0; i < n_tokens; i++)
            if (token_ids[i] == st->cfg.image_token_id || token_ids[i] == st->cfg.video_token_id) np++;
        fprintf(stderr, "[MMDBG] n_tokens=%d n_vis=%d placeholders=%d n_regions=%d seq_len_before=%d\n",
                n_tokens, n_vis_tokens, np, n_regions, st->seq_len);
    }

    /* Axiom: memory_bandwidth_reduction — workers are bound to physical
     * cores at pool creation (vllm_tp.c), so no per-region binding here. */

    int B = g_st_prefill_batch;
    int buf_hidden    = B * d;
    int buf_normed    = B * d;
    int buf_qk        = B * (q_rows > ff ? q_rows : ff);
    int buf_kv        = B * (kv_dim > ff ? kv_dim : ff);
    int buf_v         = B * kv_dim;
    int buf_attn      = B * q_rows;
    int buf_resid     = B * d;

    int total_floats = buf_hidden + buf_normed + buf_qk + buf_kv + buf_v + buf_attn + buf_resid;
    /* Axiom: partition_alignment — 64-byte aligned for vectorized parallel access */
    size_t wbuf_bytes = (size_t)total_floats * sizeof(float);
    wbuf_bytes = (wbuf_bytes + 63) & ~(size_t)63;  /* round up to 64B */
    float *wbuf = (float *)st_aligned_alloc(wbuf_bytes, 64);
    if (!wbuf) {
        for (int i = 0; i < n_tokens; i++)
            st_qwen_model_forward(st, token_ids[i]);
        return 0;
    }
    memset(wbuf, 0, wbuf_bytes);

    float *hidden_b  = wbuf;
    float *normed_b  = hidden_b + buf_hidden;
    float *qbuf_b    = normed_b + buf_normed;
    float *kbuf_b    = qbuf_b + buf_qk;
    float *vbuf_b    = kbuf_b + buf_kv;
    float *attn_b    = vbuf_b + buf_v;
    float *resid_b   = attn_b + buf_attn;

    /* Build per-token MRoPE position arrays:
     * pos_t[t], pos_h[t], pos_w[t] for each token in the full sequence */
    int *pos_t_arr = (int *)calloc((size_t)n_tokens, sizeof(int));
    int *pos_h_arr = (int *)calloc((size_t)n_tokens, sizeof(int));
    int *pos_w_arr = (int *)calloc((size_t)n_tokens, sizeof(int));

    /* vis_row_map[t] = index of the visual token (into visual_tokens /
     * ds_features rows) that replaced token t, or -1 for text tokens. Used by
     * the DeepStack injection (official _deepstack_process). */
    int *vis_row_map = (int *)malloc((size_t)n_tokens * sizeof(int));
    if (!vis_row_map) {
        free(pos_t_arr); free(pos_h_arr); free(pos_w_arr);
        return -1;
    }
    for (int i = 0; i < n_tokens; i++) vis_row_map[i] = -1;

    /* Assign visual positions. Each visual region (an image or a video) has
     * its own [grid_t, grid_h, grid_w] — already AFTER spatial merge (the
     * vision encoder outputs merged tokens). Region r occupies visual token
     * range [r_start[r], r_start[r+1]) of grid_t*grid_h*grid_w tokens. */
    if (!grids || n_regions < 1) n_regions = 1;
    int *r_start = (int *)calloc((size_t)n_regions + 1, sizeof(int));
    for (int r = 0; r < n_regions; r++)
        r_start[r + 1] = r_start[r] +
            grids[r * 3 + 0] * grids[r * 3 + 1] * grids[r * 3 + 2];
    int img_id  = st->cfg.image_token_id;   /* 151655 */
    int vid_id  = st->cfg.video_token_id;   /* 151656 */
    int text_pos = 0;  /* sequential position for text tokens */
    int vis_idx = 0;   /* which visual token we're on */

    for (int i = 0; i < n_tokens; i++) {
        int tid = token_ids[i];
        if (tid == img_id || tid == vid_id) {
            /* Visual token: assign 3D MRoPE position. Official reference
             * (get_vision_position_ids): position = meshgrid(pos_t, pos_h,
             * pos_w) + start_position, where start_position is the text
             * position at the region start and pos_h/pos_w run over the
             * merged grid. */
            if (vis_idx < n_vis_tokens) {
                /* Find the region this visual token belongs to. */
                int r = 0;
                while (r < n_regions - 1 && vis_idx >= r_start[r + 1]) r++;
                int g_t = grids[r * 3 + 0];
                int g_h = grids[r * 3 + 1];
                int g_w = grids[r * 3 + 2];
                /* Tokens are laid out frame-major then row-major (h, w) */
                int flat = vis_idx - r_start[r];
                int flat_cycle = flat % (g_h * g_w);
                int frame = g_t > 1 ? flat / (g_h * g_w) : 0;
                int ph = (flat_cycle / g_w) % g_h;
                int pw = flat_cycle % g_w;
                pos_t_arr[i] = text_pos + frame;
                pos_h_arr[i] = text_pos + ph;
                pos_w_arr[i] = text_pos + pw;
                vis_row_map[i] = vis_idx;   /* for DeepStack injection */
                vis_idx++;
                if (vis_idx == r_start[r + 1]) {
                    /* Region done: advance text position by the MRoPE span
                     * (official: max(grid_h, grid_w) on the merged grid). */
                    text_pos += (g_h > g_w ? g_h : g_w);
                }
            } else {
                pos_t_arr[i] = pos_h_arr[i] = pos_w_arr[i] = text_pos;
            }
        } else {
            /* Text token: all positions are the same (1D RoPE degenerate) */
            pos_t_arr[i] = pos_h_arr[i] = pos_w_arr[i] = text_pos;
            text_pos++;
        }
    }
    free(r_start);

    /* Process tokens in mini-batches through all layers */
    int tok_offset = 0;
    int last_nb = 0;
    while (tok_offset < n_tokens) {
        int nb = n_tokens - tok_offset;
        if (nb > B) nb = B;
        last_nb = nb;

        /* Embed this mini-batch's tokens, substituting visual tokens */
        int vis_used = 0;
        /* Pre-count visual tokens used before this batch */
        {
            int vi = 0;
            for (int p = 0; p < tok_offset; p++) {
                int tid = token_ids[p];
                if (tid == img_id || tid == vid_id) vi++;
            }
            vis_used = vi;
        }

        for (int t = 0; t < nb; t++) {
            float *h = hidden_b + (size_t)t * d;
            int tid = token_ids[tok_offset + t];

            if (tid == img_id || tid == vid_id) {
                /* Substitute with visual token */
                if (vis_used < n_vis_tokens) {
                    const float *vsrc = visual_tokens + (size_t)vis_used * d;
                    memcpy(h, vsrc, d * sizeof(float));
                    vis_used++;
                } else {
                    /* Fallback: use token embedding */
                    for (int i = 0; i < d; i++)
                        h[i] = st_emb_get(w, (size_t)tid, (size_t)i, d);
                }
            } else {
                for (int i = 0; i < d; i++)
                    h[i] = st_emb_get(w, (size_t)tid, (size_t)i, d);
            }
        }

        float *h_buf = hidden_b;
        float *nrm = normed_b;
        float *qb  = qbuf_b;
        float *kb  = kbuf_b;
        float *vb  = vbuf_b;
        float *att = attn_b;

        for (int l = 0; l < nl; l++) {
            g_npu_layer = l;   /* transparent NPU offload: per-layer op models */
            /* --- Step 1: RMSNorm per token --- */
            for (int t = 0; t < nb; t++) {
                float *h = h_buf + (size_t)t * d;
                float *nm = nrm + (size_t)t * d;
                dyn_rms_norm(nm, h, w->attn_norm + (size_t)l * d, d, eps);
            }

            if (use_q4 && w->q4_q_weight) {
                const uint8_t *q4q = w->q4_q_weight + Q4_BYTES((size_t)l * d * q_rows);
                const uint8_t *q4k = w->q4_k_weight + Q4_BYTES((size_t)l * d * kv_dim);
                const uint8_t *q4v = w->q4_v_weight + Q4_BYTES((size_t)l * d * kv_dim);
                dyn_matvec_q4_q8_fused_qkv_batched(qb, kb, vb, q4q, q4k, q4v,
                                                    nrm, q_rows, kv_dim, d, nb,
                                                    w->has_x8 ? w->x8_q_weight + Q4_BYTES((size_t)l * d * q_rows) : NULL,
                                                    w->has_x8 ? w->x8_k_weight + Q4_BYTES((size_t)l * d * kv_dim) : NULL,
                                                    w->has_x8 ? w->x8_v_weight + Q4_BYTES((size_t)l * d * kv_dim) : NULL);
            } else {
                const uint8_t *q8q = w->q8_q_weight + Q8_BYTES((size_t)l * d * q_rows);
                const uint8_t *q8k = w->q8_k_weight + Q8_BYTES((size_t)l * d * kv_dim);
                const uint8_t *q8v = w->q8_v_weight + Q8_BYTES((size_t)l * d * kv_dim);
                dyn_matvec_q8_fused_qkv_batched(qb, kb, vb, q8q, q8k, q8v,
                                                 nrm, q_rows, kv_dim, d, nb);
            }

            /* --- Step 3-5: batched Q/K norms + per-token MRoPE --- */
            {
                const float *qw = w->q_norm ? w->q_norm + (size_t)l * hd : NULL;
                const float *kw = w->k_norm ? w->k_norm + (size_t)l * hd : NULL;
                int nq = w->q_norm ? nh : 0;
                int nk = w->k_norm ? nkv : 0;
                int total = nb * (nq + nk);
                vllm_qk_norm_batch_ctx vc = { qb, kb, qw, kw, nq, nk, hd, q_rows, kv_dim };
                vllm_tp_parfor(0, total, vllm_qk_norm_batch_worker, &vc);
            }
            for (int t = 0; t < nb; t++) {
                int idx = tok_offset + t;
                int pos_t = pos_t_arr[idx];
                int pos_h = pos_h_arr[idx];
                int pos_w = pos_w_arr[idx];
                float *qt = qb + (size_t)t * q_rows;
                float *kt = kb + (size_t)t * kv_dim;

                /* MRoPE (matches Qwen3VLTextRotaryEmbedding + apply_interleaved_mrope):
                 *   inv_freq[j] = 1/theta^(j/half)  for j in [0, half), half = hd/2
                 *   pair (j, j+half) shares the angle (rotate_half pairing)
                 *   axis(j): j%3==1 && j<sec[1]*3 -> pos_h; j%3==2 && j<sec[2]*3 -> pos_w;
                 *            otherwise -> pos_t (T is the interleave base).
                 * cos/sin computed once per token, reused across all Q/K heads. */
                {
                    int n_mrope_sec = st->cfg.mrope_n_sec;
                    int sec[4];
                    if (n_mrope_sec == 3) {
                        sec[0] = st->cfg.mrope_sections[0]; /* 24 */
                        sec[1] = st->cfg.mrope_sections[1]; /* 20 */
                        sec[2] = st->cfg.mrope_sections[2]; /* 20 */
                    } else {
                        sec[0] = hd / 2; sec[1] = sec[2] = 0;
                    }
                    int half = hd / 2;
                    int bound_h = sec[1] * 3;
                    int bound_w = sec[2] * 3;

                    float cos_tab[64], sin_tab[64];
                    const float *freqs = rope_freq_table(half, theta);
                    for (int j = 0; j < half; j++) {
                        int axis = pos_t;
                        if ((j % 3) == 1 && j < bound_h) axis = pos_h;
                        else if ((j % 3) == 2 && j < bound_w) axis = pos_w;
                        float freq = freqs ? freqs[j] : 1.0f / powf(theta, (float)j / (float)half);
                        float angle = (float)axis * freq;
                        cos_tab[j] = cosf(angle);
                        sin_tab[j] = sinf(angle);
                    }

                    /* Apply MRoPE to Q and K (rotate_half pairing), one pool region */
                    vllm_mrope_heads_ctx vm = { qt, kt, cos_tab, sin_tab, half, hd, nh, nkv };
                    vllm_tp_parfor(0, nh + nkv, vllm_mrope_heads_worker, &vm);
                }
            }

            /* --- Step 5b: Batch KV-cache write — per-token into per-block rows
             * (blocks may not be contiguous in RAM after L3 eviction). */
            int cl = st->cache_len[l];
            for (int t = 0; t < nb; t++) {
                memcpy(kv_row_f32(st->k_cache[l], cl + t, kv_dim, st->kv_bs), kb + (size_t)t * kv_dim,
                       kv_dim * sizeof(float));
                memcpy(kv_row_f32(st->v_cache[l], cl + t, kv_dim, st->kv_bs), vb + (size_t)t * kv_dim,
                       kv_dim * sizeof(float));
            }
            if (st->use_kv_q8) {
                /* Quantize all nb tokens' K/V to INT8 with per-head max-abs scale */
                for (int t = 0; t < nb; t++) {
                    int8_t *kdst = kv_row_i8(st->k_cache_q8[l], cl + t, kv_dim, st->kv_bs);
                    int8_t *vdst = kv_row_i8(st->v_cache_q8[l], cl + t, kv_dim, st->kv_bs);
                    kv_quantize_per_head(kdst, vdst,
                                         st->k_scale[l] + (size_t)(cl + t) * nkv,
                                         st->v_scale[l] + (size_t)(cl + t) * nkv,
                                         kb + (size_t)t * kv_dim,
                                         vb + (size_t)t * kv_dim,
                                         nkv, hd);
                }
            } else if (st->use_kv_q4) {
                for (int t = 0; t < nb; t++) {
                    q4_pack_token(q4_k_row(st->k_cache_q4[l], cl + t, 0, hd, st->kv_bs, nkv),
                                  q4_v_row(st->v_cache_q4[l], cl + t, 0, hd, st->kv_bs, nkv),
                                  kb + (size_t)t * kv_dim,
                                  vb + (size_t)t * kv_dim,
                                  nkv, hd);
                }
            }

            /* --- Step 6: Attention: blocked Flash-Attn (seq≥64), INT8, or AVX2 scalar --- */
            /* CAUSAL: token t (global position cl+t) attends only keys in
             * [0, cl+t+1) — later tokens of the batch are future. (The packed
             * prefill path uses prev_len for this; here we must use cl+t+1.) */
            float scale = 1.0f / sqrtf((float)hd);
            int hd8 = hd & ~7;
            for (int t = 0; t < nb; t++) {
                int seq_len = cl + t + 1;
                float *qt = qb + (size_t)t * q_rows;
                float *at = att + (size_t)t * q_rows;

                if (st->use_kv_q8) {
                    /* INT8 KV cache: dequantize K/V on-the-fly */
                    int mkv = st->max_kv_slots;
                    vllm_mm_attn_ctx vc = { st, qt, at, nh, nkv, hd, hd8, l, seq_len, kv_dim, mkv, scale };
                    vllm_tp_parfor(0, nh, vllm_mm_attn_i8_worker, &vc);
                } else if (st->use_kv_q4) {
                    /* Q4 payload cache: compressed dot + in-place V accumulation */
                    int mkv = st->max_kv_slots;
                    vllm_mm_attn_ctx vc = { st, qt, at, nh, nkv, hd, hd8, l, seq_len, kv_dim, mkv, scale };
                    vllm_tp_parfor(0, nh, vllm_mm_attn_q4_worker, &vc);
                } else if (seq_len >= FA_THRESHOLD) {
                    /* Blocked Flash Attention: pack K/V tiles for L1 cache hit */
                    vllm_mm_attn_ctx vc = { st, qt, at, nh, nkv, hd, hd8, l, seq_len, kv_dim, 0, scale };
                    vllm_tp_parfor(0, nh, vllm_mm_attn_flash_worker, &vc);
                } else {
                    /* Standard AVX2 FMA float path */
                    int mkv = st->max_kv_slots;
                    vllm_mm_attn_ctx vc = { st, qt, at, nh, nkv, hd, hd8, l, seq_len, kv_dim, mkv, scale };
                    vllm_tp_parfor(0, nh, vllm_mm_attn_f32_worker, &vc);
                }

                /* KV-cache length increment */
                st->cache_len[l]++;
            }

            /* --- Step 7: Batched O projection with residual ---
             * Save the layer input as the attention residual (official:
             * h += attn(h)).  The prior code used the stale resid_b here,
             * which dropped layer-0's input residual and leaked the previous
             * batch's final layer output into the next batch's layer 0. */
            {
                for (int t = 0; t < nb; t++)
                    memcpy(resid_b + (size_t)t * d, h_buf + (size_t)t * d, d * sizeof(float));
                const uint8_t *q8o = w->q8_o_weight + Q8_BYTES((size_t)l * d * d);
                if (use_q4 && w->q4_o_weight) {
                    const uint8_t *q4o = w->q4_o_weight + Q4_BYTES((size_t)l * d * d);
                    dyn_matvec_q4_q8_fused_o_residual_batched(h_buf, resid_b, q4o, att, d, d, nb,
                        w->has_x8 ? w->x8_o_weight + Q4_BYTES((size_t)l * d * d) : NULL);
                } else {
                    dyn_matvec_q8_fused_o_residual_batched(h_buf, resid_b, q8o, att, d, d, nb);
                }
            }

            /* --- Step 8: RMSNorm per token --- */
            for (int t = 0; t < nb; t++) {
                float *h = h_buf + (size_t)t * d;
                float *nm = nrm + (size_t)t * d;
                dyn_rms_norm(nm, h, w->ffn_norm + (size_t)l * d, d, eps);
            }

            /* --- Step 9: Batched Gate/Up projection --- */
            if (st_ffn_q2_layer(w, l)) {
                const uint8_t *q2g = w->q2_gate_weight + Q2_BYTES((size_t)l * d * ff);
                const uint8_t *q2u = w->q2_up_weight   + Q2_BYTES((size_t)l * d * ff);
                dyn_matvec_q2_q8_fused_gate_up_batched(qb, kb, q2g, q2u, nrm, ff, d, nb);
            } else if (use_q4 && w->q4_gate_weight) {
                const uint8_t *q4g = w->q4_gate_weight + Q4_BYTES((size_t)st_ffn_q4_layer_off(w, l) * d * ff);
                const uint8_t *q4u = w->q4_up_weight   + Q4_BYTES((size_t)st_ffn_q4_layer_off(w, l) * d * ff);
                const uint8_t *x8g = w->x8_gate_weight
                    ? w->x8_gate_weight + Q4_BYTES((size_t)st_ffn_q4_layer_off(w, l) * d * ff) : NULL;
                const uint8_t *x8u = w->x8_up_weight
                    ? w->x8_up_weight   + Q4_BYTES((size_t)st_ffn_q4_layer_off(w, l) * d * ff) : NULL;
                dyn_matvec_q4_q8_fused_gate_up_batched(qb, kb, q4g, q4u, nrm, ff, d, nb, x8g, x8u);
            } else {
                const uint8_t *q8g = w->q8_gate_weight + Q8_BYTES((size_t)l * d * ff);
                const uint8_t *q8u = w->q8_up_weight + Q8_BYTES((size_t)l * d * ff);
                dyn_matvec_q8_fused_gate_up_batched(qb, kb, q8g, q8u, nrm, ff, d, nb);
            }

            /* Apply SiLU activation: gate = silu(gate) * up.
             * Use fast_silu (clamped to [-20,20]) — the inline 1/(1+expf(-g))
             * form overflows to inf for gate < ~-88 (visual tokens can hit
             * -90+), and /fp:fast then turns 0*g*u into NaN. */
            vllm_silu_mul_ctx vg = { qb, kb, nb * ff };
            vllm_tp_parfor(0, (nb * ff + 15) / 16, vllm_silu_mul_worker, &vg);

            /* --- Step 10: Batched Down projection with residual --- */
            {
                for (int t = 0; t < nb; t++)
                    memcpy(resid_b + (size_t)t * d, h_buf + (size_t)t * d, d * sizeof(float));
                const uint8_t *q8d = w->q8_down_weight + Q8_BYTES((size_t)l * d * ff);
                if (st_ffn_q2_layer(w, l)) {
                    const uint8_t *q2d = w->q2_down_weight + Q2_BYTES((size_t)l * d * ff);
                    dyn_matvec_q2_q8_fused_down_residual_batched(h_buf, resid_b, q2d, qb, d, ff, nb);
                } else if (use_q4 && w->q4_down_weight) {
                    const uint8_t *q4d = w->q4_down_weight + Q4_BYTES((size_t)st_ffn_q4_layer_off(w, l) * d * ff);
                    dyn_matvec_q4_q8_fused_down_residual_batched(h_buf, resid_b, q4d, qb, d, ff, nb,
                        w->has_x8 ? w->x8_down_weight + Q4_BYTES((size_t)st_ffn_q4_layer_off(w, l) * d * ff) : NULL);
                } else {
                    dyn_matvec_q8_fused_down_residual_batched(h_buf, resid_b, q8d, qb, d, ff, nb);
                }
            }

            /* DeepStack injection (official _deepstack_process): the first
             * n_ds layers add the layer's deepstack feature to each
             * visual-token row of the layer output (h_buf). */
            if (ds_features && n_ds > 0 && l < n_ds) {
                const float *ds = ds_features[l];
                vllm_ds_ctx vd = { h_buf, ds, vis_row_map, tok_offset, nb, d };
                vllm_tp_parfor(0, nb, vllm_ds_worker, &vd);
            }
        }

        tok_offset += nb;
    }

    /* Final norm + LM head for the LAST token of the last mini-batch */
    float *last_hidden = hidden_b + (size_t)(last_nb - 1) * d;
    dyn_rms_norm(normed_b, last_hidden, w->final_norm, d, eps);
    if (use_q4 && w->q4_lm_weight)
        dyn_matvec_q4_q8(st->logits, w->q4_lm_weight, normed_b, vc, d);
    else
        dyn_matvec_q8(st->logits, w->q8_lm_weight, normed_b, vc, d);

    st->seq_len += n_tokens;
    st->mrope_pos = text_pos;   /* decode continues the text MRoPE position from here */

    if (getenv("VLLM_MMDBG"))
        fprintf(stderr, "[MMDBG] prefill done: seq_len=%d mrope_pos=%d cache_len0=%d\n",
                st->seq_len, st->mrope_pos, st->cache_len[0]);

    st_aligned_free(wbuf);
    free(pos_t_arr);
    free(pos_h_arr);
    free(pos_w_arr);
    free(vis_row_map);
    return 0;
}

/* Single-region compatibility wrapper (image or video only). */
int st_qwen_model_multimodal_prefill(
    STQwenInferenceState *st,
    const int *token_ids, int n_tokens,
    const float *visual_tokens, int n_vis_tokens,
    int grid_thw[3])  /* [grid_t, grid_h, grid_w] for the single visual region */
{
    return st_qwen_model_multimodal_prefill_ex(st, token_ids, n_tokens,
                                               visual_tokens, n_vis_tokens,
                                               grid_thw, 1, NULL, 0);
}

/* ================================================================
 * Mixed-Precision (Q4_0 vs Q8_0) kernel benchmark - synthetic weights.
 *
 * Runs the real engine kernels on Qwen3-VL-8B shapes without loading the
 * 13.8 GB model, so it can run in constrained environments:
 *   decode (M=1): fused QKV / fused gate+up / fused down+residual
 *   prefill (M=B): the _batched variants
 * Both the Q4_0 and Q8_0 weight copies are quantized from the same
 * deterministic source, matching the engine's dual-quantization
 * (axiom: fixedpoint_quantize_saturate, B=8 and B=4).
 * ================================================================ */

/* Deterministic LCG row generator + dual quantization (Q8_0 + Q4_0). */
static void st_bench_quant_rows(uint8_t *q8, uint8_t *q4, int rows, int cols,
                                uint32_t *rng, float scl) {
    float *row = (float *)malloc((size_t)cols * sizeof(float));
    if (!row) return;
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            *rng = *rng * 1664525u + 1013904223u;
            row[c] = (((float)((*rng >> 8) & 0xFFFFu)) / 65535.0f - 0.5f) * scl;
        }
        if (q8) f32_to_q8_0(q8 + Q8_BYTES((size_t)r * cols), row, cols);
        if (q4) f32_to_q4_0(q4 + Q4_BYTES((size_t)r * cols), row, cols);
    }
    free(row);
}

#define ST_BENCH_MS(name, reps, flops, call)                              \
    do {                                                                  \
        for (int _w = 0; _w < 2; _w++) { call; }                          \
        double _best = 1e30;                                              \
        for (int _r = 0; _r < (reps); _r++) {                             \
            double _t0 = st_now_sec();                                    \
            call;                                                         \
            double _t1 = st_now_sec();                                    \
            double _ms = (_t1 - _t0) * 1000.0;                            \
            if (_ms < _best) _best = _ms;                                 \
        }                                                                 \
        printf("  %-40s %9.3f ms  %9.1f GFLOPS\n",                        \
               (name), _best, (double)(flops) / (_best * 1e-3) / 1e9);    \
    } while (0)

void st_bench_mixed_precision(void) {
    const int d = 4096, q_rows = 4096, kv_rows = 1024, ff = 12288, B = 32;
    uint32_t rng = 0x13579BDFu;

    st_default_threads();   /* RK3588: default 4x A76 */
    printf("\n=== Mixed-Precision Kernel Benchmark (Q4_0 vs Q8_0) ===\n");
    printf("  dim=%d q_rows=%d kv_rows=%d ffn=%d batch=%d | pool threads=%d\n",
           d, q_rows, kv_rows, ff, B, vllm_tp_threads());

    uint8_t *q8_q = (uint8_t *)malloc(Q8_BYTES((size_t)q_rows * d));
    uint8_t *q4_q = (uint8_t *)malloc(Q4_BYTES((size_t)q_rows * d));
    uint8_t *q8_k = (uint8_t *)malloc(Q8_BYTES((size_t)kv_rows * d));
    uint8_t *q4_k = (uint8_t *)malloc(Q4_BYTES((size_t)kv_rows * d));
    uint8_t *q8_v = (uint8_t *)malloc(Q8_BYTES((size_t)kv_rows * d));
    uint8_t *q4_v = (uint8_t *)malloc(Q4_BYTES((size_t)kv_rows * d));
    uint8_t *q8_g = (uint8_t *)malloc(Q8_BYTES((size_t)ff * d));
    uint8_t *q4_g = (uint8_t *)malloc(Q4_BYTES((size_t)ff * d));
    uint8_t *q8_u = (uint8_t *)malloc(Q8_BYTES((size_t)ff * d));
    uint8_t *q4_u = (uint8_t *)malloc(Q4_BYTES((size_t)ff * d));
    uint8_t *q8_dn = (uint8_t *)malloc(Q8_BYTES((size_t)d * ff));
    uint8_t *q4_dn = (uint8_t *)malloc(Q4_BYTES((size_t)d * ff));
    if (!q8_q || !q4_q || !q8_k || !q4_k || !q8_v || !q4_v ||
        !q8_g || !q4_g || !q8_u || !q4_u || !q8_dn || !q4_dn) {
        printf("  [FAIL] weight alloc OOM\n");
        goto bench_free;
    }
    printf("  Weights: dual Q8_0+Q4_0 copies (%.0f MB)\n",
           (Q8_BYTES((size_t)q_rows * d) + Q4_BYTES((size_t)q_rows * d) +
            Q8_BYTES((size_t)kv_rows * d) * 2 + Q4_BYTES((size_t)kv_rows * d) * 2 +
            Q8_BYTES((size_t)ff * d) * 3 + Q4_BYTES((size_t)ff * d) * 3) / 1048576.0);

    st_bench_quant_rows(q8_q, q4_q, q_rows, d, &rng, 0.01f);
    st_bench_quant_rows(q8_k, q4_k, kv_rows, d, &rng, 0.01f);
    st_bench_quant_rows(q8_v, q4_v, kv_rows, d, &rng, 0.01f);
    st_bench_quant_rows(q8_g, q4_g, ff, d, &rng, 0.05f);
    st_bench_quant_rows(q8_u, q4_u, ff, d, &rng, 0.05f);
    st_bench_quant_rows(q8_dn, q4_dn, d, ff, &rng, 0.05f);

    float *x = (float *)malloc((size_t)d * sizeof(float));
    float *q = (float *)malloc((size_t)q_rows * sizeof(float));
    float *k = (float *)malloc((size_t)kv_rows * sizeof(float));
    float *v = (float *)malloc((size_t)kv_rows * sizeof(float));
    float *gt = (float *)malloc((size_t)ff * sizeof(float));
    float *up = (float *)malloc((size_t)ff * sizeof(float));
    float *dn = (float *)malloc((size_t)d * sizeof(float));
    float *res = (float *)malloc((size_t)d * sizeof(float));
    float *xb = (float *)malloc((size_t)B * d * sizeof(float));
    float *qb = (float *)malloc((size_t)B * q_rows * sizeof(float));
    float *kb = (float *)malloc((size_t)B * kv_rows * sizeof(float));
    float *vb = (float *)malloc((size_t)B * kv_rows * sizeof(float));
    float *gtb = (float *)malloc((size_t)B * ff * sizeof(float));
    float *upb = (float *)malloc((size_t)B * ff * sizeof(float));
    float *dnb = (float *)malloc((size_t)B * d * sizeof(float));
    float *resb = (float *)malloc((size_t)B * d * sizeof(float));
    if (!x || !q || !k || !v || !gt || !up || !dn || !res ||
        !xb || !qb || !kb || !vb || !gtb || !upb || !dnb || !resb) {
        printf("  [FAIL] buffer alloc OOM\n");
        goto bench_free;
    }
    for (int i = 0; i < d; i++) {
        rng = rng * 1664525u + 1013904223u;
        x[i] = (((float)((rng >> 8) & 0xFFFFu)) / 65535.0f - 0.5f) * 3.0f;
        res[i] = x[i];
    }
    for (int i = 0; i < B * d; i++) {
        rng = rng * 1664525u + 1013904223u;
        xb[i] = (((float)((rng >> 8) & 0xFFFFu)) / 65535.0f - 0.5f) * 3.0f;
        resb[i] = xb[i];
    }

    printf("\n  --- Decode (M=1) ---\n");

    /* Real-DRAM bandwidth wall probe. Buffers are sized so every read misses
     * L3 (compact18 302 MB, compact34 570 MB, aligned64 1.07 GB) - decode
     * streams 4.5/8.7 GB of weights per token, so the wall that matters is
     * DRAM, not L3. compact18/34 = row-major Q4_0/Q8_0 layout (2 B scale +
     * 16/32 B data per block; the 34 B/block stride crosses cache lines);
     * aligned64 = padded 64 B/block layout. Reports effective payload read
     * bandwidth. A dual-channel DDR5-6000 box should show raw ~90 GB/s;
     * single-channel drops to ~45 GB/s. */
    {
        const size_t nrows = 131072, nblk = 128;
        uint8_t *pc18 = (uint8_t *)malloc(nrows * nblk * 18);
        uint8_t *pc34 = (uint8_t *)malloc(nrows * nblk * 34);
        uint8_t *pa   = (uint8_t *)malloc(nrows * nblk * 64);
        if (pc18 && pc34 && pa) {
            memset(pc18, 0xAB, nrows * nblk * 18);
            memset(pc34, 0xCD, nrows * nblk * 34);
            memset(pa,   0xEF, nrows * nblk * 64);
            uint64_t sink = 0;
            ptrdiff_t i;
            double best_c18 = 1e30, best_c34 = 1e30, best_a = 1e30;
            for (int rep = 0; rep < 7; rep++) {
                double t0 = st_now_sec();
                for (i = 0; i < nrows; i++) {
                    const uint8_t *row = pc18 + i * (nblk * 18);
                    uint64_t s = 0;
                    for (size_t b = 0; b < nblk; b++) {
                        s += *(const uint64_t *)(row + b * 18);
                        s += *(const uint64_t *)(row + b * 18 + 8);
                    }
                    sink += s;
                }
                double t1 = st_now_sec();
                if (t1 - t0 < best_c18) best_c18 = t1 - t0;
            }
            for (int rep = 0; rep < 7; rep++) {
                double t0 = st_now_sec();
                for (i = 0; i < nrows; i++) {
                    const uint8_t *row = pc34 + i * (nblk * 34);
                    uint64_t s = 0;
                    for (size_t b = 0; b < nblk; b++) {
                        s += *(const uint64_t *)(row + b * 34);
                        s += *(const uint64_t *)(row + b * 34 + 8);
                        s += *(const uint64_t *)(row + b * 34 + 16);
                        s += *(const uint64_t *)(row + b * 34 + 24);
                    }
                    sink += s;
                }
                double t1 = st_now_sec();
                if (t1 - t0 < best_c34) best_c34 = t1 - t0;
            }
            for (int rep = 0; rep < 7; rep++) {
                double t0 = st_now_sec();
                for (i = 0; i < nrows; i++) {
                    const uint8_t *row = pa + i * (nblk * 64);
                    uint64_t s = 0;
                    for (size_t b = 0; b < nblk; b++) {
                        s += *(const uint64_t *)(row + b * 64);
                        s += *(const uint64_t *)(row + b * 64 + 8);
                    }
                    sink += s;
                }
                double t1 = st_now_sec();
                if (t1 - t0 < best_a) best_a = t1 - t0;
            }
            printf("\n  --- Real-DRAM wall probe (%zu rows x %d blocks/row) ---\n",
                   nrows, (int)nblk);
            printf("  compact 18B/block  : %6.1f ms  -> %5.1f GB/s payload\n",
                   best_c18 * 1e3, (double)nrows * nblk * 18 / best_c18 / 1e9);
            printf("  compact 34B/block  : %6.1f ms  -> %5.1f GB/s payload\n",
                   best_c34 * 1e3, (double)nrows * nblk * 34 / best_c34 / 1e9);
            printf("  64B-aligned/block  : %6.1f ms  -> %5.1f GB/s payload (raw %.1f GB/s)\n",
                   best_a * 1e3, (double)nrows * nblk * 18 / best_a / 1e9,
                   (double)nrows * nblk * 64 / best_a / 1e9);
            printf("  (sink=%llu)\n", (unsigned long long)sink);
            fflush(stdout);
        }
        free(pc18); free(pc34); free(pa);
    }

    ST_BENCH_MS("Q4 fused QKV", 10, 2.0 * (q_rows + 2 * kv_rows) * d,
                (dyn_matvec_q4_q8_fused_qkv(q, k, v, q4_q, q4_k, q4_v, x, q_rows, kv_rows, d)));
    ST_BENCH_MS("Q8 fused QKV (dec)", 10, 2.0 * (q_rows + 2 * kv_rows) * d,
                (dyn_matvec_q8_fused_qkv_batched(q, k, v, q8_q, q8_k, q8_v, x, q_rows, kv_rows, d, 1)));
    ST_BENCH_MS("Q4 fused gate+up", 10, 2.0 * 2 * ff * d,
                (dyn_matvec_q4_q8_fused_gate_up(gt, up, q4_g, q4_u, x, ff, d)));
    ST_BENCH_MS("Q8 fused gate+up (dec)", 10, 2.0 * 2 * ff * d,
                (dyn_matvec_q8_fused_gate_up_batched(gt, up, q8_g, q8_u, x, ff, d, 1)));
    ST_BENCH_MS("Q4 fused down+res", 10, 2.0 * d * ff,
                (dyn_matvec_q4_q8_fused_down_residual(dn, res, q4_dn, gt, d, ff)));
    ST_BENCH_MS("Q8 fused down+res (dec)", 10, 2.0 * d * ff,
                (dyn_matvec_q8_fused_down_residual_batched(dn, res, q8_dn, gt, d, ff, 1)));

    printf("\n  --- Prefill (M=%d, batched) ---\n", B);
    ST_BENCH_MS("Q4 batched QKV", 5, 2.0 * (q_rows + 2 * kv_rows) * d * B,
                (dyn_matvec_q4_q8_fused_qkv_batched(qb, kb, vb, q4_q, q4_k, q4_v, xb, q_rows, kv_rows, d, B, NULL, NULL, NULL)));
    ST_BENCH_MS("Q8 batched QKV", 5, 2.0 * (q_rows + 2 * kv_rows) * d * B,
                (dyn_matvec_q8_fused_qkv_batched(qb, kb, vb, q8_q, q8_k, q8_v, xb, q_rows, kv_rows, d, B)));
    ST_BENCH_MS("Q4 batched gate+up", 5, 2.0 * 2 * ff * d * B,
                (dyn_matvec_q4_q8_fused_gate_up_batched(gtb, upb, q4_g, q4_u, xb, ff, d, B, NULL, NULL)));
    ST_BENCH_MS("Q8 batched gate+up", 5, 2.0 * 2 * ff * d * B,
                (dyn_matvec_q8_fused_gate_up_batched(gtb, upb, q8_g, q8_u, xb, ff, d, B)));
    ST_BENCH_MS("Q4 batched down+res", 5, 2.0 * d * ff * B,
                (dyn_matvec_q4_q8_fused_down_residual_batched(dnb, resb, q4_dn, gtb, d, ff, B, NULL)));
    ST_BENCH_MS("Q8 batched down+res", 5, 2.0 * d * ff * B,
                (dyn_matvec_q8_fused_down_residual_batched(dnb, resb, q8_dn, gtb, d, ff, B)));

    /* Honest accuracy trade-off (axiom: blas_precision_efficiency_tradeoff). */
    {
        float *gt4 = (float *)malloc((size_t)ff * sizeof(float));
        float *up4 = (float *)malloc((size_t)ff * sizeof(float));
        float *gt8 = (float *)malloc((size_t)ff * sizeof(float));
        float *up8 = (float *)malloc((size_t)ff * sizeof(float));
        if (gt4 && up4 && gt8 && up8) {
            dyn_matvec_q4_q8_fused_gate_up(gt4, up4, q4_g, q4_u, x, ff, d);
            dyn_matvec_q8_fused_gate_up(gt8, up8, q8_g, q8_u, x, ff, d);
            double max_abs = 0.0, max_diff = 0.0;
            for (int i = 0; i < ff; i++) {
                double a = fabs((double)gt8[i]);
                double dd = fabs((double)gt4[i] - (double)gt8[i]);
                if (a > max_abs) max_abs = a;
                if (dd > max_diff) max_diff = dd;
            }
            printf("\n  Q4 vs Q8 gate+up numerical gap (M=1): max |Q4-Q8| = %.4f "
                   "(max|Q8| = %.4f, %.2f%%)\n",
                   max_diff, max_abs, max_abs > 0 ? max_diff / max_abs * 100.0 : 0.0);
        }
        free(gt4); free(up4); free(gt8); free(up8);
    }

bench_free:
    free(q8_q); free(q4_q); free(q8_k); free(q4_k); free(q8_v); free(q4_v);
    free(q8_g); free(q4_g); free(q8_u); free(q4_u); free(q8_dn); free(q4_dn);
    free(x); free(q); free(k); free(v); free(gt); free(up); free(dn); free(res);
    free(xb); free(qb); free(kb); free(vb); free(gtb); free(upb); free(dnb); free(resb);
    printf("=== Mixed-Precision benchmark done ===\n");
}

/* ================================================================
 * OTHER-bucket isolation probe.
 *
 * Reproduces the prefill per-token "OTHER" workload (MRoPE + float KV-store
 * memcpy + INT8 kv_quantize) at the real shapes WITHOUT loading the model,
 * for S=1024/2048/4096. If per-token-layer cost stays flat, the S=4096
 * OTHER collapse (72.7% / 164x) is NOT in this pattern and needs the full
 * model-resident context; if it climbs, a memory-access pattern is isolated.
 * ================================================================ */
void st_probe_other(void) {
    const int nl = 36, nkv = 8, hd = 128, kv_dim = 1024, nh = 32, B = 32;
    const int d = 4096, ff = 12288;
    const int ctxs[] = { 1024, 2048, 4096 };
    double base_ptl = 0.0;   /* per-token-layer baseline from S=1024 */

    printf("\n=== OTHER-bucket isolation probe (MRoPE + KV-store + INT8 quant) ===\n");
    for (int ci = 0; ci < 3; ci++) {
        int S = ctxs[ci];
        int slots = S + 200;
        size_t csz = (size_t)nl * slots * kv_dim;
        size_t ssz = (size_t)nl * slots * nkv;

        float  *k_cache = (float *)malloc(csz * sizeof(float));
        float  *v_cache = (float *)malloc(csz * sizeof(float));
        int8_t *kq8 = (int8_t *)malloc(csz);
        int8_t *vq8 = (int8_t *)malloc(csz);
        float  *ksc = (float *)malloc(ssz * sizeof(float));
        float  *vsc = (float *)malloc(ssz * sizeof(float));
        float  *kt = (float *)malloc(kv_dim * sizeof(float));
        float  *vt = (float *)malloc(kv_dim * sizeof(float));
        float  *qt = (float *)malloc((size_t)nh * hd * sizeof(float));
        float  *h_buf = (float *)malloc((size_t)B * d * sizeof(float));
        float  *nrm = (float *)malloc((size_t)B * d * sizeof(float));
        float  *w_norm = (float *)malloc((size_t)d * sizeof(float));
        float  *qb = (float *)malloc((size_t)B * ff * sizeof(float));
        float  *kb = (float *)malloc((size_t)B * ff * sizeof(float));
        if (!k_cache || !v_cache || !kq8 || !vq8 || !ksc || !vsc || !kt || !vt || !qt ||
            !h_buf || !nrm || !w_norm || !qb || !kb) {
            printf("[PROBE] S=%d OOM, skip\n", S);
        } else {
            for (int i = 0; i < kv_dim; i++) {
                uint32_t u = (uint32_t)(i * 2654435761u);
                kt[i] = (((float)((u >> 8) & 0xFFFF)) / 65535.0f - 0.5f) * 3.0f;
                vt[i] = kt[i] * 0.5f;
            }
            for (int i = 0; i < d; i++) { w_norm[i] = 1.0f; h_buf[i] = 0.5f; nrm[i] = 0.0f; }
            for (int i = 0; i < B * ff; i++) { qb[i] = 0.5f; kb[i] = 0.5f; }
            double t0 = st_now_sec();
            for (int off = 0; off < S; off += B) {
                int nb = (S - off < B) ? (S - off) : B;
                for (int l = 0; l < nl; l++) {
                    /* Step 1 + Step 8 pattern: dyn_rms_norm per token, each call
                     * forks 2 OpenMP regions (the real prefill "OTHER" structure). */
                    for (int t = 0; t < nb; t++)
                        dyn_rms_norm(h_buf + (size_t)t * d, h_buf + (size_t)t * d,
                                     w_norm, d, 1e-6f);
                    for (int t = 0; t < nb; t++)
                        dyn_rms_norm(nrm + (size_t)t * d, h_buf + (size_t)t * d,
                                     w_norm, d, 1e-6f);
                    /* Steps 3-6 per-token: mrope + kv store + int8 quantize */
                    for (int t = 0; t < nb; t++) {
                        int cl = off + t;
                        dyn_mrope(qt, kt, hd, nh, nkv, hd, cl, 5000000.0f);
                        memcpy(k_cache + ((size_t)l * slots + cl) * kv_dim, kt,
                               kv_dim * sizeof(float));
                        memcpy(v_cache + ((size_t)l * slots + cl) * kv_dim, vt,
                               kv_dim * sizeof(float));
                        kv_quantize_per_head(
                            kq8 + ((size_t)l * slots + cl) * kv_dim,
                            vq8 + ((size_t)l * slots + cl) * kv_dim,
                            ksc + ((size_t)l * slots + cl) * nkv,
                            vsc + ((size_t)l * slots + cl) * nkv,
                            kt, vt, nkv, hd);
                    }
                    /* Step 10 pattern: per-token pool-parallel SiLU (nb regions). */
                    for (int t = 0; t < nb; t++) {
                        float *gt = qb + (size_t)t * ff;
                        float *up = kb + (size_t)t * ff;
                        vllm_silu_mul_ctx vg = { gt, up, ff };
                        vllm_tp_parfor(0, (ff + 15) / 16, vllm_silu_mul_worker, &vg);
                    }
                }
            }
            double dt = st_now_sec() - t0;
            double ptl = dt * 1e6 / ((double)S * nl);
            if (ci == 0) base_ptl = ptl;
            double kv_gb = (csz * 2 * 4.0 + csz * 2 + ssz * 2 * 4.0) / 1073741824.0;
            printf("[PROBE] S=%5d kv_buf=%5.2f GB | total=%8.3fs | "
                   "per-token-layer=%8.1f us | vs S=1024: %.2fx\n",
                   S, kv_gb, dt, ptl, ptl / base_ptl);

            /* NEW structure: batched RMSNorm + flattened SiLU (one omp region
             * per layer instead of ~160 per-token regions). */
            double t1 = st_now_sec();
            for (int off2 = 0; off2 < S; off2 += B) {
                int nb2 = (S - off2 < B) ? (S - off2) : B;
                for (int l = 0; l < nl; l++) {
                    dyn_rms_norm_batch(nrm, h_buf, w_norm, nb2, d, 1e-6f);
                    dyn_rms_norm_batch(nrm, h_buf, w_norm, nb2, d, 1e-6f);
                    for (int t = 0; t < nb2; t++) {
                        int cl = off2 + t;
                        dyn_mrope(qt, kt, hd, nh, nkv, hd, cl, 5000000.0f);
                        memcpy(k_cache + ((size_t)l * slots + cl) * kv_dim, kt,
                               kv_dim * sizeof(float));
                        memcpy(v_cache + ((size_t)l * slots + cl) * kv_dim, vt,
                               kv_dim * sizeof(float));
                        kv_quantize_per_head(
                            kq8 + ((size_t)l * slots + cl) * kv_dim,
                            vq8 + ((size_t)l * slots + cl) * kv_dim,
                            ksc + ((size_t)l * slots + cl) * nkv,
                            vsc + ((size_t)l * slots + cl) * nkv,
                            kt, vt, nkv, hd);
                    }
                    {
                        vllm_silu_mul_ctx vg = { qb, kb, nb2 * ff };
                        vllm_tp_parfor(0, (nb2 * ff + 15) / 16, vllm_silu_mul_worker, &vg);
                    }
                }
            }
            double dt_new = st_now_sec() - t1;
            double ptl_new = dt_new * 1e6 / ((double)S * nl);
            printf("[PROBE] S=%5d NEW-structure per-token-layer=%8.1f us | "
                   "%.2fx faster (fewer omp regions)\n",
                   S, ptl_new, ptl / ptl_new);
            fflush(stdout);
        }
        free(k_cache); free(v_cache); free(kq8); free(vq8);
        free(ksc); free(vsc); free(kt); free(vt); free(qt);
        free(h_buf); free(nrm); free(w_norm); free(qb); free(kb);
    }
    printf("=== OTHER probe done ===\n");
}

void st_qwen_inference_free(STQwenInferenceState *st) {
    if (!st->is_allocated) return;
    int nl = st->weights.n_layers_allocated;
    if (nl <= 0) nl = st->cfg.n_layers;
    /* Canary scan on the KV pointer arrays and their last block: any write
     * past the logical end (bs*kv_dim floats / n_blocks*8B) shows up here. */
    {
        int bs = st->kv_bs > 0 ? st->kv_bs : 32;
        int nblk = st->kv_n_blocks > 0 ? st->kv_n_blocks : 1;
        size_t kv_dim = (size_t)st->cfg.n_kv_heads * st->cfg.head_dim;
        for (int l = 0; l < nl; l++) {
            if (st->k_cache && st->k_cache[l]) {
                uint8_t *arr_tail = (uint8_t *)st->k_cache[l] + (size_t)nblk * sizeof(float *);
                int ok = 1;
                for (int i = 0; i < 64; i++) if (arr_tail[i] != 0xA5) { ok = 0; break; }
                if (!ok) fprintf(stderr, "[CANARY] k arr tail l=%d overwritten\n", l);
                uint8_t *blk_tail = (uint8_t *)st->k_cache[l][nblk - 1] + (size_t)bs * kv_dim * sizeof(float);
                ok = 1;
                for (int i = 0; i < 64; i++) if (blk_tail[i] != 0xA5) { ok = 0; break; }
                if (!ok) fprintf(stderr, "[CANARY] k last block tail l=%d overwritten\n", l);
            }
            if (st->v_cache && st->v_cache[l]) {
                uint8_t *arr_tail = (uint8_t *)st->v_cache[l] + (size_t)nblk * sizeof(float *);
                int ok = 1;
                for (int i = 0; i < 64; i++) if (arr_tail[i] != 0xA5) { ok = 0; break; }
                if (!ok) fprintf(stderr, "[CANARY] v arr tail l=%d overwritten\n", l);
                uint8_t *blk_tail = (uint8_t *)st->v_cache[l][nblk - 1] + (size_t)bs * kv_dim * sizeof(float);
                ok = 1;
                for (int i = 0; i < 64; i++) if (blk_tail[i] != 0xA5) { ok = 0; break; }
                if (!ok) fprintf(stderr, "[CANARY] v last block tail l=%d overwritten\n", l);
            }
        }
        /* scores_buf / imp_head / prefill_importance (cf_calloc_canary +64 A5) */
        int nh = st->cfg.n_heads;
        int mkv = st->max_kv_slots > 0 ? st->max_kv_slots : 1;
        if (st->scores_buf) {
            uint8_t *t = (uint8_t *)st->scores_buf + (size_t)mkv * nh * 4 * sizeof(float);
            int ok = 1;
            for (int i = 0; i < 64; i++) if (t[i] != 0xA5) { ok = 0; break; }
            if (!ok) fprintf(stderr, "[CANARY] scores_buf tail overwritten\n");
        }
        if (st->imp_head) {
            uint8_t *t = (uint8_t *)st->imp_head + (size_t)mkv * nh * sizeof(float);
            int ok = 1;
            for (int i = 0; i < 64; i++) if (t[i] != 0xA5) { ok = 0; break; }
            if (!ok) fprintf(stderr, "[CANARY] imp_head tail overwritten\n");
        }
        if (st->prefill_importance) {
            uint8_t *t = (uint8_t *)st->prefill_importance + (size_t)mkv * sizeof(float);
            int ok = 1;
            for (int i = 0; i < 64; i++) if (t[i] != 0xA5) { ok = 0; break; }
            if (!ok) fprintf(stderr, "[CANARY] prefill_importance tail overwritten\n");
        }
        /* Extended scan: INT8/Q4 KV blocks, scale arrays, and the per-state
         * scratch buffers (hidden/logits/qkv bufs/ffn bufs/cache_len). */
        {
            int nkv2 = st->cfg.n_kv_heads;
            int hd2 = st->cfg.head_dim;
            int ff2 = st->cfg.ffn_dim;
            int d2 = st->cfg.dim;
            int vc2 = st->cfg.vocab_size;
            int nh2 = st->cfg.n_heads;
            int np = q4_np(hd2);
            size_t q4_blk = (size_t)bs * (size_t)nkv2 * (size_t)np * Q4_PAYLOAD_LEN;
            #define CN_TAIL(PTR, NBYTES, TAG) do { \
                if (PTR) { \
                    const uint8_t *_tp = (const uint8_t *)(PTR) + (NBYTES); \
                    int _ok = 1; \
                    for (int _i = 0; _i < 64; _i++) if (_tp[_i] != 0xA5) { _ok = 0; break; } \
                    if (!_ok) fprintf(stderr, "[CANARY] %s tail overwritten\n", (TAG)); \
                } \
            } while (0)
            for (int l = 0; l < nl; l++) {
                if (st->k_cache_q8 && st->k_cache_q8[l])
                    CN_TAIL(st->k_cache_q8[l][nblk - 1], (size_t)bs * kv_dim, "k q8 last block");
                if (st->v_cache_q8 && st->v_cache_q8[l])
                    CN_TAIL(st->v_cache_q8[l][nblk - 1], (size_t)bs * kv_dim, "v q8 last block");
                if (st->k_cache_q4 && st->k_cache_q4[l])
                    CN_TAIL(st->k_cache_q4[l][nblk - 1], q4_blk, "k q4 last block");
                if (st->v_cache_q4 && st->v_cache_q4[l])
                    CN_TAIL(st->v_cache_q4[l][nblk - 1], q4_blk, "v q4 last block");
                if (st->k_scale && st->k_scale[l])
                    CN_TAIL(st->k_scale[l], (size_t)mkv * nkv2 * sizeof(float), "k_scale");
                if (st->v_scale && st->v_scale[l])
                    CN_TAIL(st->v_scale[l], (size_t)mkv * nkv2 * sizeof(float), "v_scale");
            }
            CN_TAIL(st->hidden, (size_t)d2 * sizeof(float), "hidden");
            CN_TAIL(st->logits, (size_t)vc2 * sizeof(float), "logits");
            CN_TAIL(st->q_buf, (size_t)((nh2 * hd2 > ff2) ? nh2 * hd2 : ff2) * sizeof(float), "q_buf");
            CN_TAIL(st->k_buf, (size_t)((nkv2 * hd2 > ff2) ? nkv2 * hd2 : ff2) * sizeof(float), "k_buf");
            CN_TAIL(st->v_buf, (size_t)nkv2 * hd2 * sizeof(float), "v_buf");
            CN_TAIL(st->attn_buf, (size_t)nh2 * hd2 * sizeof(float), "attn_buf");
            CN_TAIL(st->ffn_buf, (size_t)ff2 * sizeof(float), "ffn_buf");
            CN_TAIL(st->ffn_out_buf, (size_t)d2 * sizeof(float), "ffn_out_buf");
            CN_TAIL(st->cache_len, (size_t)nl * sizeof(int), "cache_len");
            #undef CN_TAIL
        }
        fflush(stderr);
    }
    cf_pg_free(st->hidden);
    cf_pg_free(st->logits);
    cf_pg_free(st->q_buf);
    cf_pg_free(st->k_buf);
    cf_pg_free(st->v_buf);
    cf_pg_free(st->attn_buf);
    cf_pg_free(st->ffn_buf);
    cf_pg_free(st->ffn_out_buf);
    cf_pg_free(st->cache_len);
    if (st->k_cache) {
        for (int l = 0; l < nl; l++) {
            if (st->k_cache[l]) {
                for (int b = 0; b < st->kv_n_blocks; b++)
                    if (st->k_cache[l][b]) cf_pg_free((uint8_t *)st->k_cache[l][b] - KV_GUARD);
                cf_pg_free(st->k_cache[l]);
            }
        }
        cf_pg_free(st->k_cache);
    }
    if (st->v_cache) {
        for (int l = 0; l < nl; l++) {
            if (st->v_cache[l]) {
                for (int b = 0; b < st->kv_n_blocks; b++)
                    if (st->v_cache[l][b]) cf_pg_free((uint8_t *)st->v_cache[l][b] - KV_GUARD);
                cf_pg_free(st->v_cache[l]);
            }
        }
        cf_pg_free(st->v_cache);
    }
    cf_pg_free(st->scores_buf);
    cf_pg_free(st->prefill_importance);
    cf_pg_free(st->imp_head);
    if (st->k_cache_q8) {
        for (int l = 0; l < nl; l++) {
            if (st->k_cache_q8[l]) {
                for (int b = 0; b < st->kv_n_blocks; b++)
                    if (st->k_cache_q8[l][b]) cf_pg_free((uint8_t *)st->k_cache_q8[l][b] - KV_GUARD);
                cf_pg_free(st->k_cache_q8[l]);
            }
        }
        cf_pg_free(st->k_cache_q8);
    }
    if (st->v_cache_q8) {
        for (int l = 0; l < nl; l++) {
            if (st->v_cache_q8[l]) {
                for (int b = 0; b < st->kv_n_blocks; b++)
                    if (st->v_cache_q8[l][b]) cf_pg_free((uint8_t *)st->v_cache_q8[l][b] - KV_GUARD);
                cf_pg_free(st->v_cache_q8[l]);
            }
        }
        cf_pg_free(st->v_cache_q8);
    }
    if (st->k_cache_q4) {
        for (int l = 0; l < nl; l++) {
            if (st->k_cache_q4[l]) {
                for (int b = 0; b < st->kv_n_blocks; b++)
                    if (st->k_cache_q4[l][b]) cf_pg_free((uint8_t *)st->k_cache_q4[l][b] - KV_GUARD);
                cf_pg_free(st->k_cache_q4[l]);
            }
        }
        cf_pg_free(st->k_cache_q4);
    }
    if (st->v_cache_q4) {
        for (int l = 0; l < nl; l++) {
            if (st->v_cache_q4[l]) {
                for (int b = 0; b < st->kv_n_blocks; b++)
                    if (st->v_cache_q4[l][b]) cf_pg_free((uint8_t *)st->v_cache_q4[l][b] - KV_GUARD);
                cf_pg_free(st->v_cache_q4[l]);
            }
        }
        cf_pg_free(st->v_cache_q4);
    }
    if (st->k_scale) {
        for (int l = 0; l < nl; l++) cf_pg_free(st->k_scale[l]);
        cf_pg_free(st->k_scale);
    }
    if (st->v_scale) {
        for (int l = 0; l < nl; l++) cf_pg_free(st->v_scale[l]);
        cf_pg_free(st->v_scale);
    }
    l3_state_free(&st->l3);
    memset(st, 0, sizeof(*st));
}

/* ---- Disk KV persistence (serve --disk-kv) ----
 * Snapshot the first n_tokens KV rows (F32) plus the token ids and the model
 * geometry into one file. The serve layer indexes these files by the FNV-1a
 * hash of the token sequence; on a later request (also after a process
 * restart) the file whose tokens share the longest prefix is loaded and only
 * the suffix is prefilled. F32 storage keeps the restore bit-exact with a
 * fresh prefill, matching the verified in-RAM prefix reuse.
 *
 * File layout (all little-endian, portable across x86/aarch64):
 *   [0..3]    magic  "VLKV" (0x564C4B56)
 *   [4..7]    version = 1
 *   [8..11]   n_layers
 *   [12..15]  nkv (KV heads)
 *   [16..19]  hd (head dim)
 *   [20..23]  kv_bs (KV block size in positions)
 *   [24..27]  file_n_tokens (rows saved)
 *   [28..31]  kv_dim (nkv * hd)
 *   [32..39]  token hash (FNV-1a 64 over the saved token ids)
 *   [40..43]  fmt = 0 (F32 K/V payload)
 *   [44..63]  reserved (zero)
 *   tokens:   file_n_tokens * int32
 *   payload:  per layer: per full kv_bs block: K block (kv_bs*kv_dim f32)
 *             then V block; a trailing partial block stores only its valid
 *             rows (same K-then-V order).                              */
#define DKV_MAGIC    0x564C4B56u
#define DKV_VERSION  1
#define DKV_HEADER   64

/* FNV-1a 64 over the token id bytes. Used for the checkpoint filename and as
 * a redundancy check in the header. The serve layer additionally verifies the
 * token sequences themselves (LCP is token-exact), so a hash collision can at
 * worst cause a wasted file lookup, never a wrong KV restore. */
static uint64_t fnv1a64(const int *tokens, int n) {
    uint64_t h = 1469598103934665603ULL;
    for (int i = 0; i < n; i++) {
        uint32_t x = (uint32_t)tokens[i];
        for (int b = 0; b < 4; b++) {
            h ^= (x & 0xFFu);
            h *= 1099511628211ULL;
            x >>= 8;
        }
    }
    return h;
}

int st_kv_disk_save(STQwenInferenceState *st, const char *path,
                    const int *tokens, int n_tokens) {
    if (!st || !path || !tokens || n_tokens <= 0) return -1;
    int avail = st->cache_len[0];
    if (n_tokens > avail) return -1;                 /* rows not computed yet */
    int nl = st->weights.n_layers_allocated;
    if (nl <= 0) nl = st->cfg.n_layers;
    int nkv = st->cfg.n_kv_heads;
    int hd  = st->cfg.head_dim;
    int bs  = st->kv_bs > 0 ? st->kv_bs : 32;
    int kv_dim = nkv * hd;
    size_t row = (size_t)kv_dim * sizeof(float);

    FILE *fp = fopen(path, "wb");
    if (!fp) return -1;
    uint8_t hdr[DKV_HEADER];
    memset(hdr, 0, sizeof(hdr));
    uint32_t m = DKV_MAGIC, v = DKV_VERSION;
    memcpy(hdr + 0,  &m, 4);
    memcpy(hdr + 4,  &v, 4);
    memcpy(hdr + 8,  &nl, 4);
    memcpy(hdr + 12, &nkv, 4);
    memcpy(hdr + 16, &hd, 4);
    memcpy(hdr + 20, &bs, 4);
    memcpy(hdr + 24, &n_tokens, 4);
    memcpy(hdr + 28, &kv_dim, 4);
    uint64_t hash = fnv1a64(tokens, n_tokens);
    memcpy(hdr + 32, &hash, 8);
    int rc = 0;
    if (fwrite(hdr, 1, DKV_HEADER, fp) != DKV_HEADER) rc = -1;
    if (rc == 0 && fwrite(tokens, sizeof(int), (size_t)n_tokens, fp)
                   != (size_t)n_tokens) rc = -1;
    for (int l = 0; rc == 0 && l < nl; l++) {
        int t = 0;
        while (t < n_tokens) {
            int nb = n_tokens - t;
            if (nb > bs) nb = bs;
            /* K rows [t, t+nb) then V rows (same geometry). */
            if (fwrite(kv_row_f32(st->k_cache[l], t, kv_dim, bs), row,
                       (size_t)nb, fp) != (size_t)nb) { rc = -1; break; }
            if (fwrite(kv_row_f32(st->v_cache[l], t, kv_dim, bs), row,
                       (size_t)nb, fp) != (size_t)nb) { rc = -1; break; }
            t += nb;
        }
    }
    if (fclose(fp) != 0 && rc == 0) rc = -1;
    return rc;
}

int st_kv_disk_load(STQwenInferenceState *st, const char *path, int n_tokens) {
    if (!st || !path || n_tokens <= 0) return -1;
    if (n_tokens > st->max_kv_slots) return -1;
    int nl = st->weights.n_layers_allocated;
    if (nl <= 0) nl = st->cfg.n_layers;
    int nkv = st->cfg.n_kv_heads;
    int hd  = st->cfg.head_dim;
    int bs  = st->kv_bs > 0 ? st->kv_bs : 32;
    int kv_dim = nkv * hd;
    size_t row = (size_t)kv_dim * sizeof(float);

    FILE *fp = fopen(path, "rb");
    if (!fp) return -1;
    uint8_t hdr[DKV_HEADER];
    int rc = 0;
    if (fread(hdr, 1, DKV_HEADER, fp) != DKV_HEADER) rc = -1;
    uint32_t m, v, fnl, fnkv, fhd, fbs, ftok, fdim;
    if (rc == 0) {
        memcpy(&m, hdr + 0, 4); memcpy(&v, hdr + 4, 4);
        memcpy(&fnl, hdr + 8, 4); memcpy(&fnkv, hdr + 12, 4);
        memcpy(&fhd, hdr + 16, 4); memcpy(&fbs, hdr + 20, 4);
        memcpy(&ftok, hdr + 24, 4); memcpy(&fdim, hdr + 28, 4);
        if (m != DKV_MAGIC || v != DKV_VERSION) rc = -1;
        if (fnl != (uint32_t)nl || fnkv != (uint32_t)nkv || fhd != (uint32_t)hd ||
            fbs != (uint32_t)bs || fdim != (uint32_t)kv_dim) rc = -1;
        if (n_tokens > (int)ftok) rc = -1;      /* requested prefix exceeds file */
        if ((uint32_t)n_tokens > (uint32_t)st->kv_n_blocks * (uint32_t)bs)
            rc = -1;
    }
    if (rc == 0) {
        /* Skip the token ids (they were verified against the live request by
         * the serve layer's LCP; only geometry needs re-checking). */
        if (fseek(fp, (long)((size_t)ftok * sizeof(int)), SEEK_CUR) != 0)
            rc = -1;
    }
    /* Payload layout mirrors the save: per layer, per kv_bs block: K rows
     * then V rows, blocks full (bs rows) except possibly the file's final
     * block. Restoring only the first n_tokens rows must therefore walk the
     * FILE's block boundaries (frows), not the requested count: after a
     * partial K read the rest of the file's K block must be skipped before V,
     * and each layer must be jumped to its exact end (the file stores ftok
     * rows per layer, not n_tokens). */
    const int64_t layer_bytes = (int64_t)ftok * 2 * kv_dim * (int64_t)sizeof(float);
    for (int l = 0; rc == 0 && l < nl; l++) {
        int t = 0;
        while (t < n_tokens) {
            int nb = n_tokens - t;
            if (nb > bs) nb = bs;
            int blk = t / bs;
            int frows = bs;
            if (blk * bs + bs > (int)ftok) frows = (int)ftok - blk * bs; /* file's last block */
            if (nb > frows) nb = frows;
            if (fread(kv_row_f32(st->k_cache[l], t, kv_dim, bs), row,
                      (size_t)nb, fp) != (size_t)nb) { rc = -1; break; }
            if (nb < frows && fseek(fp, (long)((frows - nb) * (int64_t)row),
                                    SEEK_CUR) != 0) { rc = -1; break; }
            if (fread(kv_row_f32(st->v_cache[l], t, kv_dim, bs), row,
                      (size_t)nb, fp) != (size_t)nb) { rc = -1; break; }
            if (nb < frows && fseek(fp, (long)((frows - nb) * (int64_t)row),
                                    SEEK_CUR) != 0) { rc = -1; break; }
            t += nb;
        }
        if (rc == 0) {
            /* Absolute jump to the next layer's data (immune to block math). */
            long pos = ftell(fp);
            long layer_end = (long)((int64_t)DKV_HEADER + (int64_t)ftok * sizeof(int)
                                    + (int64_t)(l + 1) * layer_bytes);
            if (pos < layer_end) {
                if (fseek(fp, layer_end - pos, SEEK_CUR) != 0) rc = -1;
            } else if (pos > layer_end) {
                rc = -1;   /* file shorter than expected */
            }
        }
    }
    fclose(fp);
    if (rc != 0) return rc;
    /* Rebuild the derived compressed caches exactly like the prefill store
     * path (vllm_safetensors.c kv store) so decode reads consistent data. */
    for (int l = 0; l < nl; l++) {
        for (int t = 0; t < n_tokens; t++) {
            float *kf = kv_row_f32(st->k_cache[l], t, kv_dim, bs);
            float *vf = kv_row_f32(st->v_cache[l], t, kv_dim, bs);
            if (st->use_kv_q8) {
                int8_t *kdst = kv_row_i8(st->k_cache_q8[l], t, kv_dim, bs);
                int8_t *vdst = kv_row_i8(st->v_cache_q8[l], t, kv_dim, bs);
                kv_quantize_per_head(kdst, vdst,
                                     st->k_scale[l] + (size_t)t * nkv,
                                     st->v_scale[l] + (size_t)t * nkv,
                                     kf, vf, nkv, hd);
            } else if (st->use_kv_q4) {
                q4_pack_token(q4_k_row(st->k_cache_q4[l], t, 0, hd, bs, nkv),
                              q4_v_row(st->v_cache_q4[l], t, 0, hd, bs, nkv),
                              kf, vf, nkv, hd);
            }
        }
    }
    for (int l = 0; l < nl; l++) st->cache_len[l] = n_tokens;
    st->seq_len = n_tokens;
    return 0;
}
