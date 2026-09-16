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
 *   is reproduced at the top of tools/kernels/llama_gemm_q4_0_4x4_asm.c.
 *   Derivations (adapted and/or word-for-word ported in-place):
 *     - ggml_gemm_q4_0_4x4_q8_0  -> tools/kernels/llama_gemm_q4_0_4x4_asm.c (verbatim
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
#include "vqf.h"          /* 分层驻留 vqf_stream_layer_advance（仅明文 VQF） */
#include "vllm_ep.h"      /* MoE 专家并行 transport 抽象层（阶段一/二） */
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

/* Forward declaration: f16_to_f32_bits is defined in the Q8_0 helper
 * section below but used earlier (dequant on load); C99 implicit declaration
 * would otherwise clash with the static inline definition on GCC. */
static inline uint32_t f16_to_f32_bits(uint16_t h);

/* 平台 SIMD 层：aarch64 走 NEON（vllm_platform.h）；x86-64 走 AVX2。
 * 本文件若干"共享函数体"内直接书写 NEON intrinsics（见下映射层），x86
 * 下由等价 __m128 映射补齐，保持同架构内位级一致、跨架构文档化近似。 */

/* ================================================================
 * x86-64 NEON-compat mapping layer (aarch64 编译时整段不参与)
 *
 * 仅为"未用 ST_HAVE_NEON 隔离、直接书写 NEON 的共享函数体"提供
 * __m128 等价（fp32 乘加经 FMA 保融合舍入语义；x86 构建需 -mavx2 -mfma）。
 * NEON 主内核区（#if ST_HAVE_NEON）不经过此层。
 * ================================================================ */
#if ST_ARCH_X86 && !ST_HAVE_NEON
#include <immintrin.h>

typedef __m128  float32x4_t;
typedef __m128i int8x16_t;
typedef __m128i uint8x16_t;
typedef __m128i int16x8_t;
typedef __m128i int32x4_t;
typedef __m128i uint16x8_t;
typedef __m128i int8x8_t;     /* 低 64 bit 承载 8×int8 */
typedef __m128i int16x4_t;    /* 低 64 bit 承载 4×int16 */

static inline float32x4_t vdupq_n_f32(float x)   { return _mm_set1_ps(x); }
static inline float32x4_t vld1q_f32(const float *p) { return _mm_loadu_ps(p); }
static inline void vst1q_f32(float *p, float32x4_t v) { _mm_storeu_ps(p, v); }
static inline float32x4_t vfmaq_f32(float32x4_t a, float32x4_t b, float32x4_t c) {
    return _mm_fmadd_ps(b, c, a);
}
static inline float32x4_t vmulq_f32(float32x4_t a, float32x4_t b) { return _mm_mul_ps(a, b); }
static inline float32x4_t vaddq_f32(float32x4_t a, float32x4_t b) { return _mm_add_ps(a, b); }
static inline float32x4_t vsubq_f32(float32x4_t a, float32x4_t b) { return _mm_sub_ps(a, b); }
static inline float32x4_t vmaxq_f32(float32x4_t a, float32x4_t b) { return _mm_max_ps(a, b); }
static inline float32x4_t vminq_f32(float32x4_t a, float32x4_t b) { return _mm_min_ps(a, b); }
static inline float32x4_t vmulq_n_f32(float32x4_t a, float x) {
    return _mm_mul_ps(a, _mm_set1_ps(x));
}
static inline float32x4_t vaddq_n_f32(float32x4_t a, float x) {
    return _mm_add_ps(a, _mm_set1_ps(x));
}
/* NEON vpaddq_f32: (a0+a1, a2+a3, b0+b1, b2+b3) == SSE3 hadd. */
static inline float32x4_t vpaddq_f32(float32x4_t a, float32x4_t b) {
    return _mm_hadd_ps(a, b);
}
static inline float vgetq_lane_f32(float32x4_t v, int i) {
    union { float32x4_t v; float f[4]; } u;
    u.v = v;
    return u.f[i];
}
/* 32 个 int8 → 8 组 float32x4（组内 4 个连续 int8 各扩为 float）。 */
static inline void i8x32_to_f32x8(const int8_t *qs, float32x4_t f[8]) {
    for (int i = 0; i < 8; i++) {
        int32_t v[4];
        for (int j = 0; j < 4; j++) v[j] = qs[i * 4 + j];
        f[i] = _mm_set_ps((float)v[3], (float)v[2], (float)v[1], (float)v[0]);
    }
}
/* 8×float32x4（32 元素）水平归约：两两向量相加后 4 宽折半。 */
static inline float f32x8_hsum(const float32x4_t f[8]) {
    __m128 s = _mm_add_ps(_mm_add_ps(f[0], f[1]), _mm_add_ps(f[2], f[3]));
    s = _mm_add_ps(s, _mm_add_ps(_mm_add_ps(f[4], f[5]), _mm_add_ps(f[6], f[7])));
    __m128 t = _mm_add_ps(s, _mm_shuffle_ps(s, s, 0xB1));
    t = _mm_add_ps(t, _mm_shuffle_ps(t, t, 0x4E));
    return _mm_cvtss_f32(t);
}
/* 4 宽水平归约（单个 float32x4）。 */
static inline float hsum_neon4(float32x4_t v) {
    __m128 t = _mm_add_ps(v, _mm_shuffle_ps(v, v, 0xB1));
    t = _mm_add_ps(t, _mm_shuffle_ps(t, t, 0x4E));
    return _mm_cvtss_f32(t);
}
/* ---- 8-bit KV Q8 dot 展开链（vld1_s8→vmovl_s8→vget_{low,high}_s16→vmovl_s16） ---- */
static inline int8x8_t vld1_s8(const int8_t *p) {
    return _mm_loadl_epi64((const __m128i *)p);
}
static inline int16x8_t vmovl_s8(int8x8_t a) {
    return _mm_cvtepi8_epi16(a);   /* SSE4.1: 低 8×int8 → 8×int16 */
}
/* 低/高 4×int16：cvtepi16_epi32 只读低 64bit，高组先右移 8B。 */
static inline int16x4_t vget_low_s16(int16x8_t a) { return a; }
static inline int16x4_t vget_high_s16(int16x8_t a) {
    return _mm_srli_si128(a, 8);
}
static inline int32x4_t vmovl_s16(int16x4_t a) {
    return _mm_cvtepi16_epi32(a);  /* SSE4.1: 低 4×int16 → 4×int32 */
}
static inline float32x4_t vcvtq_f32_s32(int32x4_t a) {
    return _mm_cvtepi32_ps(a);
}

/* ---- 供未隔离 wrapper 调用的 *_neon 同名 x86 实现（Q8_0/Q4_0 matvec）----
 * 语义与 NEON 同名函数一致（权重布局 Q8_0: 34B/32 元素；Q4_0: 18B/32 元素，
 * scale=fp16 头 + nibble 低/高半字节=元素 j/16+j），归约顺序与 ARM 不同属
 * 文档化跨架构近似。后续可换 AVX2 worker 提升吞吐。 */
static float st_f16scale(const uint8_t *b) {
    uint16_t h;
    memcpy(&h, b, 2);
    uint32_t fb = f16_to_f32_bits(h);
    float d;
    memcpy(&d, &fb, 4);
    return d;
}

/* ---- VQF 权重布局感知（x86）----
 * ARM（dotprod）默认把 Q8_0 权重 repack 为 8x8/4x4、Q4_0 repack 为 4x4
 * （repack_q8_0_8x8_inplace / repack_q4_0_4x4_inplace），转换产物经 VQF
 * 文件固化（flags Q8_8X8/Q4_4X4，loader 与 g_st_*_repack 一致性校验）。
 * x86 侧挂载不执行 repack，解码必须按"内存即文件布局"分派：
 *   q8 mode 0=legacy 34B/块 · 1=4x4 136B/组 · 2=8x8 272B/组
 *   q4 mode 0=legacy 18B/块 · 1=4x4 72B/组
 * 布局判定与 loader 的 vqf_cur_flags 同源（g_st_*_repack + VLLM_Q8_8X8）。
 * 数据布局规范（以 repack 函数为准）：
 *   q8 4x4 (136B, 4行/组)：{ d[m] f16, m=0..3 }@0，qs@8：
 *     qs[8+k*16+m*4+i] = 行 m 元素 (k*4+i)，k=0..7
 *   q8 8x8 (272B, 8行/组)：{ d[m] f16, m=0..7 }@0，qs@16：
 *     qs[16+k*32+m*4+i] = 行 m 元素 (k*4+i)，k=0..7
 *   q4 4x4 (72B, 4行/组)：{ d[m] f16, m=0..3 }@0，qs@8（每字节 XOR 0x88）：
 *     行 m 的 legacy nibble 字节（4 字节/块 k）位于 qs[k*16+m*4..+4)，
 *     解码前先 ^0x88 还原 nibble（与 ARM vdot 的符号语义逐位一致）。 */
extern int g_st_q8_repack;
extern int g_st_q4_repack;
static int q8_8x8_enabled(void);

static int st_q8_layout(void) {
    if (!g_st_q8_repack) return 0;
    return q8_8x8_enabled() ? 2 : 1;
}
static int st_q4_layout(void) {
    return g_st_q4_repack ? 1 : 0;
}

/* Q8_0 单行点积（矩阵基址 w + 行号 r 寻址；mode 见 st_q8_layout）。 */
static float st_q8_row_dot(const uint8_t *w, int r, const float *x, int n_blocks, int mode) {
    float acc = 0.0f;
    if (mode == 0) {
        const uint8_t *pr = w + (size_t)r * (size_t)n_blocks * 34;
        for (int b = 0; b < n_blocks; b++) {
            float d = st_f16scale(pr + (size_t)b * 34);
            const int8_t *qs = (const int8_t *)(pr + (size_t)b * 34 + 2);
            const float *xs = x + (size_t)b * 32;
            float s = 0.0f;
            for (int j = 0; j < 32; j++) s += (float)qs[j] * xs[j];
            acc += d * s;
        }
    } else if (mode == 1) {
        int m = r & 3;
        const uint8_t *g = w + (size_t)(r >> 2) * (size_t)n_blocks * 136;
        for (int b = 0; b < n_blocks; b++) {
            const uint8_t *out = g + (size_t)b * 136;
            float d = st_f16scale(out + 2 * m);
            const float *xs = x + (size_t)b * 32;
            float s = 0.0f;
            for (int k = 0; k < 8; k++)
                for (int i = 0; i < 4; i++)
                    s += (float)(int8_t)out[8 + k * 16 + m * 4 + i] * xs[k * 4 + i];
            acc += d * s;
        }
    } else {
        int m = r & 7;
        const uint8_t *g = w + (size_t)(r >> 3) * (size_t)n_blocks * 272;
        for (int b = 0; b < n_blocks; b++) {
            const uint8_t *out = g + (size_t)b * 272;
            float d = st_f16scale(out + 2 * m);
            const float *xs = x + (size_t)b * 32;
            float s = 0.0f;
            for (int k = 0; k < 8; k++)
                for (int i = 0; i < 4; i++)
                    s += (float)(int8_t)out[16 + k * 32 + m * 4 + i] * xs[k * 4 + i];
            acc += d * s;
        }
    }
    return acc;
}
/* Q4_0 单行点积（nibble 低/高半字节 = 元素 j / 16+j；mode 见 st_q4_layout）。 */
static float st_q4_row_dot(const uint8_t *w, int r, const float *x, int n_blocks, int mode) {
    float acc = 0.0f;
    if (mode == 0) {
        const uint8_t *pr = w + (size_t)r * (size_t)n_blocks * 18;
        for (int b = 0; b < n_blocks; b++) {
            float d = st_f16scale(pr + (size_t)b * 18);
            const uint8_t *nb = pr + (size_t)b * 18 + 2;
            const float *xs = x + (size_t)b * 32;
            float s = 0.0f;
            for (int j = 0; j < 16; j++)
                s += (float)((int)(nb[j] & 0x0F) - 8) * xs[j] +
                     (float)((int)(nb[j] >> 4) - 8) * xs[16 + j];
            acc += d * s;
        }
    } else {
        int m = r & 3;
        const uint8_t *g = w + (size_t)(r >> 2) * (size_t)n_blocks * 72;
        for (int b = 0; b < n_blocks; b++) {
            const uint8_t *out = g + (size_t)b * 72;
            float d = st_f16scale(out + 2 * m);
            const float *xs = x + (size_t)b * 32;
            const uint8_t *qs = out + 8;
            float s = 0.0f;
            for (int k = 0; k < 4; k++)           /* 块 k: 4 个 nibble 字节 */
                for (int j = 0; j < 4; j++) {
                    uint8_t nb = (uint8_t)(qs[k * 16 + m * 4 + j] ^ 0x88);
                    s += (float)((int)(nb & 0x0F) - 8) * xs[k * 4 + j] +
                         (float)((int)(nb >> 4) - 8) * xs[16 + k * 4 + j];
                }
            acc += d * s;
        }
    }
    return acc;
}

/* ---- q4 4x4（mode1）4 行组 AVX2 点积（dense qkv/o/lm 解码热内核）----
 * 与 st_q4_row_dot mode1 语义完全一致：每 tile 4 个 f16 scale（每行一个），
 * 每个 scale 应用在"整块 32 元素求和之后"（acc += d*s，1 次舍入），而非 MoE
 * 稀疏路径 vq4x4_w 的逐元素 scale（两者舍入链不同，勿混用）。
 * 一次处理 4 行 × 32 列：lane=行，k=0..31 逐元素 mul+add，链序 = 标量
 * （块外层 b → k 内层）→ A≡C 位级一致。VLLM_Q4_DENSE_SIMD=0 关闭。
 * 仅 x86 AVX2 且文件为 q4 4x4（g_st_q4_repack）。 */
static int q4_dense_simd_ok(void) {
#if defined(__AVX2__) && ST_ARCH_X86
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("VLLM_Q4_DENSE_SIMD");
        v = (e && e[0] == '0') ? 0 : 1;
    }
    return v && g_st_q4_repack;
#else
    return 0;
#endif
}

#if defined(__AVX2__) && ST_ARCH_X86
/* 重组索引：输出 lane 4s+ri ← chunk 内偏移 ri*4+s（mode1 字节布局 = chunk(16B) 内
 * 4 行 × 4 子列，与 MoE 区 q4x4_grp 同构）。 */
static const uint8_t q4d4_grp[16] = {
    0, 4, 8, 12,   1, 5, 9, 13,   2, 6, 10, 14,   3, 7, 11, 15,
};

/* mode1 单行组（4 行 × n_blocks×32 列）点积 → out4[4]。
 * 2026-09-13 重写：旧版**每个内层迭代都在栈上现场拼 16 字节 shuffle 掩码**
 * （`uint8_t mm[16]; memcpy; for(...)`，16 次/块），-O2 未展开时病态（§7.6/§9.19）。
 * 新版改为「整 chunk 预异或 + 单次 pshufb 重组 4 子列 × 4 行 + AVX2 一次加宽」，
 * 掩码全部 static const，**不依赖编译期展开**。
 * 数值与旧版逐位一致：块外层 b → (k 外层 0..3, j 内层 0..3) → s = s + (t1+t2)
 * → acc = acc + d*s；舍入链与操作顺序一字未动（只改取值方式的指令数）。 */
static void q4_dense4_x86(const uint8_t *w, int r4, const float *x,
                          int n_blocks, float out4[4]) {
    __m128 acc = _mm_setzero_ps();
    const uint8_t *g = w + (size_t)r4 * (size_t)n_blocks * 72;
    const __m128i grp   = _mm_loadu_si128((const __m128i *)q4d4_grp);
    const __m128i xor88 = _mm_set1_epi8((char)0x88);   /* 逐字节还原（≠ set1_epi32） */
    const __m128i m15   = _mm_set1_epi32(15);
    const __m128  m8    = _mm_set1_ps(8.0f);
    for (int b = 0; b < n_blocks; b++) {
        const uint8_t *out = g + (size_t)b * 72;
        __m128 d = _mm_setr_ps(st_f16scale(out + 0), st_f16scale(out + 2),
                               st_f16scale(out + 4), st_f16scale(out + 6));
        const uint8_t *qs = out + 8;
        const float *xs = x + (size_t)b * 32;
        __m128 s = _mm_setzero_ps();
        for (int k = 0; k < 4; k++) {          /* 外层 k 与标量一致 */
            __m128i C = _mm_xor_si128(
                _mm_loadu_si128((const __m128i *)(qs + (size_t)k * 16)), xor88);
            __m128i n = _mm_shuffle_epi8(C, grp);      /* lane 4s+ri = 行 ri 子列 s */
            __m256i y0 = _mm256_cvtepu8_epi32(n);              /* 子列 0,1 */
            __m256i y1 = _mm256_cvtepu8_epi32(_mm_srli_si128(n, 8)); /* 子列 2,3 */
            __m128i G0 = _mm256_castsi256_si128(y0);            /* 子列 0 */
            __m128i G1 = _mm256_extracti128_si256(y0, 1);       /* 子列 1 */
            __m128i G2 = _mm256_castsi256_si128(y1);            /* 子列 2 */
            __m128i G3 = _mm256_extracti128_si256(y1, 1);       /* 子列 3 */
            /* 一个 (k,j) 对：lo 半字节 → 元素 k*4+j，hi 半字节 → 元素 16+k*4+j；
             * 结合序 s = s + (t1 + t2) 必须保持。 */
#define Q4D4_PAIR(GQ, J)                                                       \
            do {                                                               \
                __m128 wlo_ = _mm_cvtepi32_ps(_mm_and_si128((GQ), m15));       \
                __m128 whi_ = _mm_cvtepi32_ps(                                 \
                    _mm_and_si128(_mm_srli_epi32((GQ), 4), m15));              \
                wlo_ = _mm_sub_ps(wlo_, m8);                                   \
                whi_ = _mm_sub_ps(whi_, m8);                                   \
                __m128 t1_ = _mm_mul_ps(wlo_,                                  \
                    _mm_set1_ps(xs[(size_t)k * 4 + (J)]));                     \
                __m128 t2_ = _mm_mul_ps(whi_,                                  \
                    _mm_set1_ps(xs[16 + (size_t)k * 4 + (J)]));                \
                s = _mm_add_ps(s, _mm_add_ps(t1_, t2_));                       \
            } while (0)
            Q4D4_PAIR(G0, 0);
            Q4D4_PAIR(G1, 1);
            Q4D4_PAIR(G2, 2);
            Q4D4_PAIR(G3, 3);
#undef Q4D4_PAIR
        }
        acc = _mm_add_ps(acc, _mm_mul_ps(d, s));   /* acc += d*s：与标量同 */
    }
    _mm_storeu_ps(out4, acc);
}
#endif /* __AVX2__ && ST_ARCH_X86 */

/* 线程池前向声明（vllm_tp.h 在文件后部 include；x86 区内核先行引用）。 */
extern void vllm_tp_parfor(int start, int end,
                           void (*fn)(void *ctx, int idx), void *ctx);

/* ---- down 执行结构重构 M2 插桩（VLLM_MOE_PROF=1，默认关；FP 零影响）----
 * 逐 token 累计 gu 并行区 / down j 循环 / 整 ffn 墙钟，打印 48 层合计，
 * 用于定位 decode ~375ms（T_down）的 stage 真分布。 */
static int moe_prof_enabled(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("VLLM_MOE_PROF"); v = (e && e[0] == '1') ? 1 : 0; }
    return v;
}
#if defined(_WIN32)
static double moe_now(void) {   /* vllm_tp_wtime（clock_gettime）在 MinGW 下返 0 */
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
    return f.QuadPart > 0 ? (double)c.QuadPart / (double)f.QuadPart : 0.0;
}
#else
static double moe_now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
#endif
static double mpt_ff = 0, mpt_gu = 0, mpt_dn = 0;
static int mpt_n = 0;
static double vllm_tp_wtime(void) { return moe_now(); }  /* 本文件内遮蔽：QPC 时钟 */

/* ---- wrapper 层入口追踪（VLLM_MOE_TRACE=1，默认关）----
 * 逐 token 统计 st_moe_ffn_sparse 实际分发到 q4/q8 的次数与总墙钟，
 * 用于厘清 DEC_PROF 中 gu 桶（≈100-170ms/token）与 q4 内部计时（0.1ms）的矛盾。 */
static int moe_ffn_trace_env(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("VLLM_MOE_TRACE"); v = (e && e[0] == '1') ? 1 : 0; }
    return v;
}
static double mft_ff = 0, mft_max = 0;
static int mft_nc = 0, mft_q4 = 0, mft_q8 = 0, mft_maxl = -1;

#if defined(__AVX2__) && ST_ARCH_X86
/* ---- Q8_0 8x8 (272B/块) 布局 dense decode AVX2 内核（A≡C 语义）----
 * st_q8_row_dot mode==2 的多行/多 tile 交错实现：一次推进 4 个 8 行 tile
 * （每 tile 两个 4 行半组独立 f32 累加链）以提升 ILP。
 * 链序与标量一致：块外层 b → k 外层 → i 内层，逐元素 mul+add（不融合）。
 * 重要：本内核依赖 fp 收缩关闭（build_x64.ps1 需 -ffp-contract=off），
 * 否则 gcc 会把 mul+add 收缩成 FMA 造成 ~1ulp 漂移（A≠C）。门控
 * VLLM_Q8_DENSE_SIMD=0 回退旧逐行标量。 */
static int q8_dense_simd_ok(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("VLLM_Q8_DENSE_SIMD");
        v = (e && e[0] == '0') ? 0 : 1;
    }
    return v && g_st_q8_repack && q8_8x8_enabled();
}

static const uint8_t q8x8_shuf4[4][16] = {
    {0, 4, 8, 12, 0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80},
    {1, 5, 9, 13, 0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80},
    {2, 6,10, 14, 0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80},
    {3, 7,11, 15, 0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80}
};

static inline __m128 q8x8_half_dot_add(__m128 p, __m128i bv, const float *xs, int i) {
    __m128i sh = _mm_shuffle_epi8(bv, _mm_loadu_si128((const __m128i *)q8x8_shuf4[i]));
    __m128i e32 = _mm_cvtepi8_epi32(sh);
    __m128 f = _mm_cvtepi32_ps(e32);
    return _mm_add_ps(p, _mm_mul_ps(f, _mm_set1_ps(xs[i])));
}

/* 绝对行段 [r0, r0+count) 的 mode2 点积写入 dst（add!=NULL 时 dst[r]=add[r]+dot）。
 * 每 32 行 chunk（4 tile×8 行）交错；chunk 尾部按 8 行 tile 标量路径补齐。 */
static void q8_dense_rows_range_avx2(const uint8_t *w, const float *x, int r0,
                                     int count, int n_blocks,
                                     float *__restrict dst,
                                     const float *__restrict add) {
    const int T = 4;             /* 交错 tile 数 */
    const int CH = T * 8;
    int base = 0;
    for (; base + CH <= count; base += CH) {
        __m128 acc[T][2];
        for (int t = 0; t < T; t++)
            for (int h = 0; h < 2; h++)
                acc[t][h] = _mm_setzero_ps();
        for (int b = 0; b < n_blocks; b++) {
            const float *xsb = x + (size_t)b * 32;
            for (int t = 0; t < T; t++) {
                const uint8_t *ob = w + (size_t)((r0 >> 3) + (base >> 3) + t)
                                        * (size_t)n_blocks * 272 + (size_t)b * 272;
                __m128 dlo = _mm_setr_ps(st_f16scale(ob + 0), st_f16scale(ob + 2),
                                          st_f16scale(ob + 4), st_f16scale(ob + 6));
                __m128 dhi = _mm_setr_ps(st_f16scale(ob + 8), st_f16scale(ob + 10),
                                          st_f16scale(ob + 12), st_f16scale(ob + 14));
                __m128 plo = _mm_setzero_ps(), phi = _mm_setzero_ps();
                for (int k = 0; k < 8; k++) {
                    __m256i s = _mm256_loadu_si256((const __m256i *)(ob + 16 + k * 32));
                    __m128i lo = _mm256_castsi256_si128(s);
                    __m128i hi = _mm256_extracti128_si256(s, 1);
                    for (int i = 0; i < 4; i++) {
                        plo = q8x8_half_dot_add(plo, lo, xsb + k * 4, i);
                        phi = q8x8_half_dot_add(phi, hi, xsb + k * 4, i);
                    }
                }
                acc[t][0] = _mm_add_ps(acc[t][0], _mm_mul_ps(dlo, plo));
                acc[t][1] = _mm_add_ps(acc[t][1], _mm_mul_ps(dhi, phi));
            }
        }
        for (int t = 0; t < T; t++) {
            float v[8];
            _mm_storeu_ps(v, acc[t][0]);
            _mm_storeu_ps(v + 4, acc[t][1]);
            for (int j = 0; j < 8; j++) {
                int r = r0 + base + t * 8 + j;
                dst[r] = add ? add[r] + v[j] : v[j];
            }
        }
    }
    for (int rr = base; rr < count; rr++) {
        int r = r0 + rr;
        float d = st_q8_row_dot(w, r, x, n_blocks, 2);
        dst[r] = add ? add[r] + d : d;
    }
}

/* 全矩阵入口（r0=0）。 */
static void q8_dense_rows_avx2(const uint8_t *w, const float *x, int rows,
                               int n_blocks, float *__restrict dst,
                               const float *__restrict add) {
    q8_dense_rows_range_avx2(w, x, 0, rows, n_blocks, dst, add);
}

/* ---- P1-2：行组分块线程池（vllm_tp_parfor）----
 * decode 单 token（n_batch==1）时把 rows 切成 Q8D_CHU 行/块并行；每块内部仍走
 * q8_dense_rows_range_avx2（A≡C 语义不变，各行写独立 dst 无竞争）。
 * Q8D_CHU 运行时可调（VLLM_Q8D_CHU，默认 128）→ P1-2 深挖用于扫描 128..1024。 */
static int q8d_chu(void) {
    static int c = 0;
    if (!c) {
        c = 128;
        const char *e = getenv("VLLM_Q8D_CHU");
        if (e && e[0] >= '1' && e[0] <= '9') {
            int v = atoi(e);
            if (v >= 32 && v <= 4096) c = v;
        }
    }
    return c;
}
#define Q8D_CHU q8d_chu()
typedef struct {
    const uint8_t *w; const float *x; int rows; int n_blocks;
    float *__restrict dst; const float *__restrict add;
} q8d_tp_ctx;
static void q8d_tp_worker(void *c, int it) {
    q8d_tp_ctx *v = (q8d_tp_ctx *)c;
    int r0 = it * Q8D_CHU;
    int cnt = r0 + Q8D_CHU <= v->rows ? Q8D_CHU : v->rows - r0;
    if (cnt > 0)
        q8_dense_rows_range_avx2(v->w, v->x, r0, cnt, v->n_blocks, v->dst, v->add);
}
static void q8_dense_rows_avx2_tp(const uint8_t *w, const float *x, int rows,
                                  int n_blocks, float *__restrict dst,
                                  const float *__restrict add) {
    if (rows <= Q8D_CHU) {
        q8_dense_rows_range_avx2(w, x, 0, rows, n_blocks, dst, add);
        return;
    }
    q8d_tp_ctx v = { w, x, rows, n_blocks, dst, add };
    vllm_tp_parfor(0, (rows + Q8D_CHU - 1) / Q8D_CHU, q8d_tp_worker, &v);
}

/* ---- P1-2 深挖：同层多矩阵合并为一次全层行分派 ----
 * decode 单 token 的 QKV（q/k/v 3 矩阵）与 gate/up（2 矩阵）原本各起一次
 * vllm_tp_parfor（每层多付 2~3 次发布/join 同步，且小矩阵分派粒度碎）。
 * 这里把共享同一输入 x 的多矩阵线性化为一条“全层行空间”，一次 parfor 派发
 * 全部行（段边界跨界时在段内切开）。每行点积链序与分开展开完全一致 →
 * A≡C 位级语义不变；行段起点均保持 8 对齐（rows 与 CHU 皆 8 的倍数）。 */
typedef struct {
    const uint8_t *const *w;   /* nseg 个矩阵 */
    float *const *dst;
    const float *const *add;   /* 可为 NULL（各段均无残差累加） */
    const int *rows;           /* 各段行数（合计 = total） */
    const float *x;
    int n_blocks, nseg, total;
} q8d_multi_ctx;
static void q8d_multi_worker(void *c, int it) {
    q8d_multi_ctx *v = (q8d_multi_ctx *)c;
    int r0 = it * Q8D_CHU;
    int cnt = r0 + Q8D_CHU <= v->total ? Q8D_CHU : v->total - r0;
    if (cnt <= 0) return;
    int seg = 0, base = 0;
    while (seg < v->nseg && base + v->rows[seg] <= r0) { base += v->rows[seg]; seg++; }
    int cur = r0, end = r0 + cnt;
    while (cur < end && seg < v->nseg) {
        int se = base + v->rows[seg];
        int n = end < se ? end - cur : se - cur;
        if (n > 0)
            q8_dense_rows_range_avx2(v->w[seg], v->x, cur - base, n, v->n_blocks,
                                     v->dst[seg], v->add ? v->add[seg] : NULL);
        cur += n; base = se; seg++;
    }
}
static void q8_dense_multi_avx2_tp(const uint8_t *const *w, const float *x,
                                   int n_blocks, float *const *dst,
                                   const float *const *add, const int *rows,
                                   int nseg, int total) {
    q8d_multi_ctx v = { w, dst, add, rows, x, n_blocks, nseg, total };
    if (total <= Q8D_CHU) { q8d_multi_worker(&v, 0); return; }
    vllm_tp_parfor(0, (total + Q8D_CHU - 1) / Q8D_CHU, q8d_multi_worker, &v);
}

/* ---- P2-1 M2：x86 prefill/批 q8 int8-激活 GEMM（llama 收口轨移植，M1b 内核）----
 * 精度=新轨：激活按 (token, 32 列块) q8_0 量化（s8），权重保持 mode2 8x8 s8，
 * 块内 s16 拓宽精确点积（vpmovsxbw→vpmaddwd→i32），逐块浮点 scale：
 *   out[r] = (resid? resid[r]:0) + Σ_b d_w·d_x·dot(r,b)   （b 升序链，确定性新锚）
 * 与引擎 f32 直算逐 token 轨声明分叉（prefill 收口方向，VLLM_GEMM_LEGACY=1 回退）。
 * 激活量化语义对齐引擎 ARM quantize_row_q8_0_act（max-abs/127，clip ±127）。
 * 几何门控：rows%8==0 && cols%32==0 && cols<=65536（重排缓冲上界）。 */
static int q8_gemm_legacy_env(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("VLLM_GEMM_LEGACY"); v = (e && e[0] == '1') ? 1 : 0; }
    return v;
}
static int q8_prefill_gemm_ok(int rows, int cols, int nb) {
    if (q8_gemm_legacy_env()) return 0;
    if (!(g_st_q8_repack && q8_8x8_enabled())) return 0;   /* mode2 8x8 */
    if (rows <= 0 || cols <= 0 || nb <= 0) return 0;
    if (rows & 7) return 0;
    if (cols & 31) return 0;
    if (cols > 65536) return 0;
    return 1;
}
static void q8g_quant(const float *x, int cols, int nb, int8_t *xq, float *dx, int nblk) {
    for (int p = 0; p < nb; p++) {
        const float *xp = x + (size_t)p * cols;
        for (int b = 0; b < nblk; b++) {
            float am = 0.0f;
            for (int j = 0; j < 32; j++) { float a = fabsf(xp[b * 32 + j]); if (a > am) am = a; }
            float s = am / 127.0f;
            dx[(size_t)p * nblk + b] = s;
            if (s <= 1e-30f) {
                memset(xq + (size_t)p * cols + b * 32, 0, 32);
                continue;
            }
            for (int j = 0; j < 32; j++) {
                float v = xp[b * 32 + j] / s;
                int qi = (int)lrintf(v);
                if (qi > 127) qi = 127; if (qi < -127) qi = -127;
                xq[(size_t)p * cols + b * 32 + j] = (int8_t)qi;
            }
        }
    }
}
static inline int q8g_reduce8(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i s = _mm_add_epi32(lo, hi);
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
    return _mm_cvtsi128_si32(s);
}
/* 重排 tile g 一行 m：vpgatherdd 把 mode2 块里行 m 的 32 s8 收成行连续 s16，
 * 存 ts[m*cols + b*32 .. +32)；dwg[b*8+m] 取块 f16 scale。 */
static void q8g_repack_tile(const uint8_t *w, int g, int nb, int cols,
                            int16_t *ts, float *dwg) {
    for (int b = 0; b < nb; b++) {
        const uint8_t *ob = w + ((size_t)g * nb + b) * 272;
        const int *qsb = (const int *)(ob + 16);
        for (int m = 0; m < 8; m++) {
            __m256i idx = _mm256_setr_epi32(m + 0 * 8, m + 1 * 8,
                                            m + 2 * 8, m + 3 * 8,
                                            m + 4 * 8, m + 5 * 8,
                                            m + 6 * 8, m + 7 * 8);
            __m256i g32 = _mm256_i32gather_epi32(qsb, idx, 4);
            __m128i lo = _mm256_castsi256_si128(g32);   /* k0..3 → j0..15 s8 */
            __m128i hi = _mm256_extracti128_si256(g32, 1);
            int16_t *dst = ts + (size_t)m * cols + (size_t)b * 32;
            _mm256_storeu_si256((__m256i *)dst, _mm256_cvtepi8_epi16(lo));
            _mm256_storeu_si256((__m256i *)(dst + 16), _mm256_cvtepi8_epi16(hi));
            dwg[(size_t)b * 8 + m] = st_f16scale(ob + 2 * m);
        }
    }
}
/* 单矩阵 int8 GEMM（tile 级 vllm_tp_parfor）。成功写 out 并返回 0；
 * 分配失败返回 -1（不动 out，调用方回退）。out 按 tile 分片写、无竞争。 */
typedef struct {
    const uint8_t *w; const float *x; const int8_t *xq; const float *dx;
    float *out; const float *resid;
    int rows, cols, nb, nblk, tiles;
    int16_t *pool_ts; float *pool_dw; int nslots;
} q8g_ctx;
/* 行阻塞（行末归约）tile worker —— q8g 默认内层（M2 换入）。
 * 背景：B-M1/P3/MoE-M1 微基准证明每 (行,32块) reduce8+标量 scale 结构开销是
 * 单行 ~40 GMAC/s 上限的根因；本实现把块级 i32→标量 改为 i32→f32 8-lane 向量
 * ×scale 直接累加，行末一次归约（M1 微基准 STRONG 1.73×，59.8 GMAC/s）。
 * 浮点求和序变化（块内 4 元素组部分和向量跨块累加 + 行末树归约）= 新精度锚；
 * 旧 int8 reduce 轨保留为 q8g_tile_worker_legacy（未接线：VLLM_GEMM_LEGACY=1
 * 语义为整体回退 f32 逐 token 轨，见 q8_prefill_gemm_ok）。 */
static inline float q8g_reduce8f(__m256 v) {   /* 行末归约树（序冻结，勿改） */
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_shuffle_ps(s, s, 0x4E));
    s = _mm_add_ps(s, _mm_shuffle_ps(s, s, 0xB1));
    return _mm_cvtss_f32(s);
}
__attribute__((optimize("O3","ffp-contract=off","no-fast-math")))   /* -O2 会 spill；须保 fp-contract=off 维持锚语义 */
static void q8g_tile_worker(void *c_, int it) {
    q8g_ctx *c = (q8g_ctx *)c_;
    int g = it;
    int slot = vllm_tp_worker_id();
    if (slot < 0) slot = 0;
    if (slot >= c->nslots) slot = c->nslots - 1;
    int16_t *ts = c->pool_ts + (size_t)slot * 8 * c->cols;
    float   *dw = c->pool_dw + (size_t)slot * 8 * c->nblk;
    q8g_repack_tile(c->w, g, c->nblk, c->cols, ts, dw);
    for (int p = 0; p < c->nb; p++) {
        const int8_t *xp = c->xq + (size_t)p * c->cols;
        const float  *dp = c->dx + (size_t)p * c->nblk;
        __m256 accF[8];
        for (int m = 0; m < 8; m++) accF[m] = _mm256_setzero_ps();
        for (int b = 0; b < c->nblk; b++) {
            const int8_t *xs = xp + (size_t)b * 32;
            __m256i a0 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)xs));
            __m256i a1 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(xs + 16)));
            float dxb = dp[b];
            const float *wd = dw + (size_t)b * 8;
            _Pragma("GCC unroll 8")   /* accF[8] 常量索引 → 寄存器化（-O2 亦防 spill） */
            for (int m = 0; m < 8; m++) {
                const int16_t *wr = ts + (size_t)m * c->cols + (size_t)b * 32;
                __m256i dotv = _mm256_add_epi32(
                    _mm256_madd_epi16(_mm256_loadu_si256((const __m256i *)(wr)), a0),
                    _mm256_madd_epi16(_mm256_loadu_si256((const __m256i *)(wr + 16)), a1));
                __m256 f = _mm256_mul_ps(_mm256_cvtepi32_ps(dotv), _mm256_set1_ps(wd[m] * dxb));
                accF[m] = _mm256_add_ps(accF[m], f);
            }
        }
        float *op = c->out + (size_t)p * c->rows + (size_t)g * 8;
        if (c->resid) {
            const float *rp = c->resid + (size_t)p * c->rows + (size_t)g * 8;
            _Pragma("GCC unroll 8")
            for (int m = 0; m < 8; m++) op[m] = rp[m] + q8g_reduce8f(accF[m]);
        } else {
            _Pragma("GCC unroll 8")
            for (int m = 0; m < 8; m++) op[m] = q8g_reduce8f(accF[m]);
        }
    }
}
static void q8g_tile_worker_legacy(void *c_, int it) {
    q8g_ctx *c = (q8g_ctx *)c_;
    int g = it;
    int slot = vllm_tp_worker_id();
    if (slot < 0) slot = 0;
    if (slot >= c->nslots) slot = c->nslots - 1;
    int16_t *ts = c->pool_ts + (size_t)slot * 8 * c->cols;
    float   *dw = c->pool_dw + (size_t)slot * 8 * c->nblk;
    q8g_repack_tile(c->w, g, c->nblk, c->cols, ts, dw);
    for (int p = 0; p < c->nb; p++) {
        const int8_t *xp = c->xq + (size_t)p * c->cols;
        const float  *dp = c->dx + (size_t)p * c->nblk;
        float acc8[8];
        for (int m = 0; m < 8; m++) acc8[m] = 0.0f;
        for (int b = 0; b < c->nblk; b++) {
            __m128i s0 = _mm_loadu_si128((const __m128i *)(xp + (size_t)b * 32));
            __m128i s1 = _mm_loadu_si128((const __m128i *)(xp + (size_t)b * 32 + 16));
            __m256i a0 = _mm256_cvtepi8_epi16(s0);
            __m256i a1 = _mm256_cvtepi8_epi16(s1);
            float dxb = dp[b];
            for (int m = 0; m < 8; m++) {
                const int16_t *wr = ts + (size_t)m * c->cols + (size_t)b * 32;
                __m256i w0 = _mm256_loadu_si256((const __m256i *)(wr));
                __m256i w1 = _mm256_loadu_si256((const __m256i *)(wr + 16));
                __m256i dv = _mm256_add_epi32(_mm256_madd_epi16(w0, a0),
                                              _mm256_madd_epi16(w1, a1));
                int dot = q8g_reduce8(dv);
                acc8[m] += dw[(size_t)b * 8 + m] * dxb * (float)dot;
            }
        }
        float *op = c->out + (size_t)p * c->rows + (size_t)g * 8;
        if (c->resid) {
            const float *rp = c->resid + (size_t)p * c->rows + (size_t)g * 8;
            for (int m = 0; m < 8; m++) op[m] = rp[m] + acc8[m];
        } else {
            for (int m = 0; m < 8; m++) op[m] = acc8[m];
        }
    }
}
static int q8g_gemm_mat(float *__restrict out, const float *__restrict resid,
                        const uint8_t *w, const float *x, int rows, int cols, int nb) {
    int nblk = cols / 32;
    int8_t  *xq = (int8_t  *)malloc((size_t)nb * cols);
    float   *dx = (float   *)malloc((size_t)nb * nblk * sizeof(float));
    if (!xq || !dx) { free(xq); free(dx); return -1; }
    q8g_quant(x, cols, nb, xq, dx, nblk);
    int tiles = rows / 8;
    int nt = vllm_tp_threads();
    if (nt < 1) nt = 1;
    if (nt > 64) nt = 64;
    if (tiles >= 2 && nb >= 4) {
        int16_t *p_ts = (int16_t *)malloc((size_t)nt * 8 * cols * sizeof(int16_t));
        float   *p_dw = (float   *)malloc((size_t)nt * 8 * nblk * sizeof(float));
        if (!p_ts || !p_dw) { free(p_ts); free(p_dw); free(xq); free(dx); return -1; }
        q8g_ctx c = { w, x, xq, dx, out, resid, rows, cols, nb, nblk, tiles,
                      p_ts, p_dw, nt };
        void (*wk)(void *, int) = q8_gemm_legacy_env() ? q8g_tile_worker_legacy
                                                       : q8g_tile_worker;
        vllm_tp_parfor(0, tiles, wk, &c);
        free(p_ts); free(p_dw);
    } else {
        /* 小矩阵/线程 1：串行 worker（slot 0 单份 scratch） */
        q8g_ctx c = { w, x, xq, dx, out, resid, rows, cols, nb, nblk, tiles,
                      (int16_t *)malloc((size_t)8 * cols * sizeof(int16_t)),
                      (float *)malloc((size_t)8 * nblk * sizeof(float)), 1 };
        if (!c.pool_ts || !c.pool_dw) { free(c.pool_ts); free(c.pool_dw); free(xq); free(dx); return -1; }
        void (*wk)(void *, int) = q8_gemm_legacy_env() ? q8g_tile_worker_legacy
                                                       : q8g_tile_worker;
        for (int g = 0; g < tiles; g++) wk(&c, g);
        free(c.pool_ts); free(c.pool_dw);
    }
    free(xq); free(dx);
    return 0;
}

/* ---- x86 q4 4x4 int16-激活 prefill GEMM（对齐上面 q8g_gemm_mat 骨架）----
 * 动机：x86 dense q4 prefill 原走 st1_*（逐 token × 4 行组 f32 点积，**无跨 token
 * 权重复用**，且 f32 反量化算力受限）：实测 30B QKV+O = 33 ms/tok，而 ARM 同工作量
 * 走 int8 SDOT + 3-token tile 只要 7.7 ms/tok（4.3× 反常，§9.20）。
 * 本核照 ARM / q8g_gemm_mat 的收口方向重做 x86 侧：
 *   激活按 (token, 32 列块) **s16** 量化（max-abs/32767、clip ±32767，见 q4g_quant16）；
 *   权重 nibble 展开为**精确** s8（Q4 本身无量化舍入），重排成行主序；
 *   块内 s16 拓宽精确点积（vpmovsxbw → vpmaddwd → i32），逐块浮点 scale：
 *     out[r] = (resid? resid[r]:0) + Σ_b d_w(b,r)·d_x(b)·dot(r,b)     （b 升序链）
 * **权重每 tile 重排一次、复用于 tile 内全部 nb 个 token → 权重 DRAM 流量 ÷nb。**
 * 与 f32 逐 token 轨**声明分叉**（prefill 收口方向；VLLM_GEMM_LEGACY=1 或
 * VLLM_Q4_DENSE_SIMD=0 回退原 f32 轨），正确性锚 = TOKIDS/PPL。
 * 几何门控：rows%8==0 && cols%32==0 && nb>1 && cols<=65536。 */
static int q4g_prefill_gemm_ok(int rows, int cols, int nb) {
    if (q8_gemm_legacy_env()) return 0;
    if (!(g_st_q4_repack && q4_dense_simd_ok())) return 0;
    if (rows <= 0 || cols <= 0 || nb <= 1) return 0;
    if (rows & 7) return 0;
    if (cols & 31) return 0;
    if (cols > 65536) return 0;
    return 1;
}
/* VLLM_Q4G_VERIFY=1：抽样对拍新轨 vs f32 参考 st_q4_row_dot，打印最大绝对/相对差，
 * 用来把「声明分叉」的量级（量化级 ~1e-2）与「内核 bug」（量级 ~1e0 或结构性）
 * 区分开。仅前 8 次调用打印（覆盖 layer0/1 的 q/k/v/o），默认关。 */
static int q4g_verify_env(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("VLLM_Q4G_VERIFY"); v = (e && e[0] == '1') ? 1 : 0; }
    return v;
}
/* 激活 s16 量化（max-abs/32767、clip ±32767），语义同 q8g_quant，只是位宽 16。
 * **为何不用 int8**：本核内层走 vpmaddwd(w_s16, a_s16) 而非 maddubs(u8×s8)，
 * 激活加宽到 16-bit **不增加内层指令数**（少两次 vpmovsxbw），而量化步长从
 * maxabs/127 → maxabs/32767（÷258）：实测 prefill 输出偏差自「输出量级的 0.36%」
 * 降到 ~1e-5（≈f32 累加序噪声），TOKIDS 锚得以保持（int8 版会把 MoE 模型第 2 个
 * token 的贪心选择翻转 198→271）。 */
static void q4g_quant16(const float *x, int cols, int nb, int16_t *xq,
                        float *dx, int nblk) {
    for (int p = 0; p < nb; p++) {
        const float *xp = x + (size_t)p * cols;
        for (int b = 0; b < nblk; b++) {
            float am = 0.0f;
            for (int j = 0; j < 32; j++) { float a = fabsf(xp[b * 32 + j]); if (a > am) am = a; }
            float s = am / 32767.0f;
            dx[(size_t)p * nblk + b] = s;
            if (s <= 1e-30f) {
                memset(xq + (size_t)p * cols + b * 32, 0, 32 * sizeof(int16_t));
                continue;
            }
            for (int j = 0; j < 32; j++) {
                float v = xp[b * 32 + j] / s;
                int qi = (int)lrintf(v);
                if (qi > 32767) qi = 32767; if (qi < -32767) qi = -32767;
                xq[(size_t)p * cols + b * 32 + j] = (int16_t)qi;
            }
        }
    }
}
/* mode1（4x4）权重块 → 行主序 s8：ts[m*cols + b*32 + c] = 行 m / 块 b / 列 c 的精确权重。
 * 布局（与 st_q4_row_dot mode1 一致）：块 72B = 4 行 f16 scale(8B) + 64B nibble；
 * nibble 字节 idx = k*16 + m*4 + j，低半字节 = 列 k*4+j、高半字节 = 列 16+k*4+j，
 * 半字节本身是 4-bit 二补码 → signed = (nib ^ 8) - 8。tile g 覆盖行 [8g, 8g+8)
 * = 两个 4 行组 2g / 2g+1。 */
static void q4g_repack_tile(const uint8_t *w, int g, int nb, int cols,
                            int8_t *ts, float *dwg) {
    for (int h = 0; h < 2; h++) {
        const uint8_t *gg = w + (size_t)(g * 2 + h) * (size_t)nb * 72;
        for (int b = 0; b < nb; b++) {
            const uint8_t *out = gg + (size_t)b * 72;
            const uint8_t *qs = out + 8;
            for (int m = 0; m < 4; m++)
                dwg[(size_t)b * 8 + h * 4 + m] = st_f16scale(out + 2 * m);
            for (int k = 0; k < 4; k++) {
                const uint8_t *src = qs + (size_t)k * 16;
                for (int m = 0; m < 4; m++) {
                    int8_t *dst = ts + (size_t)(h * 4 + m) * cols + (size_t)b * 32;
                    for (int j = 0; j < 4; j++) {
                        uint8_t byte = src[m * 4 + j];
                        dst[k * 4 + j]      = (int8_t)(((int)(byte & 0x0F) ^ 8) - 8);
                        dst[16 + k * 4 + j] = (int8_t)(((int)(byte >> 4)   ^ 8) - 8);
                    }
                }
            }
        }
    }
}
typedef struct {
    const uint8_t *w; const float *x; const int16_t *xq; const float *dx;
    float *out; const float *resid;
    int rows, cols, nb, nblk, tiles;
    int8_t *pool_ts; float *pool_dw; int nslots;
} q4g_ctx;
__attribute__((optimize("O3","no-fast-math")))
static void q4g_tile_worker(void *c_, int it) {
    q4g_ctx *c = (q4g_ctx *)c_;
    int g = it;
    int slot = vllm_tp_worker_id();
    if (slot < 0) slot = 0;
    if (slot >= c->nslots) slot = c->nslots - 1;
    int8_t *ts = c->pool_ts + (size_t)slot * 8 * c->cols;
    float  *dw = c->pool_dw + (size_t)slot * 8 * c->nblk;
    q4g_repack_tile(c->w, g, c->nblk, c->cols, ts, dw);
    for (int p = 0; p < c->nb; p++) {
        const int16_t *xp = c->xq + (size_t)p * c->cols;
        const float   *dp = c->dx + (size_t)p * c->nblk;
        __m256 accF[8];
        for (int m = 0; m < 8; m++) accF[m] = _mm256_setzero_ps();
        for (int b = 0; b < c->nblk; b++) {
            const int16_t *xs = xp + (size_t)b * 32;
            __m256i a0 = _mm256_loadu_si256((const __m256i *)xs);
            __m256i a1 = _mm256_loadu_si256((const __m256i *)(xs + 16));
            float dxb = dp[b];
            const float *wd = dw + (size_t)b * 8;
            _Pragma("GCC unroll 8")   /* accF[8] 常量索引 → 寄存器化 */
            for (int m = 0; m < 8; m++) {
                const int8_t *wr8 = ts + (size_t)m * c->cols + (size_t)b * 32;
                __m256i w0 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)wr8));
                __m256i w1 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(wr8 + 16)));
                __m256i dotv = _mm256_add_epi32(_mm256_madd_epi16(w0, a0),
                                                _mm256_madd_epi16(w1, a1));
                __m256 f = _mm256_mul_ps(_mm256_cvtepi32_ps(dotv),
                                         _mm256_set1_ps(wd[m] * dxb));
                accF[m] = _mm256_add_ps(accF[m], f);
            }
        }
        float *op = c->out + (size_t)p * c->rows + (size_t)g * 8;
        if (c->resid) {
            const float *rp = c->resid + (size_t)p * c->rows + (size_t)g * 8;
            _Pragma("GCC unroll 8")
            for (int m = 0; m < 8; m++) op[m] = rp[m] + q8g_reduce8f(accF[m]);
        } else {
            _Pragma("GCC unroll 8")
            for (int m = 0; m < 8; m++) op[m] = q8g_reduce8f(accF[m]);
        }
    }
}
/* 单矩阵 q4 GEMM（tile 级 vllm_tp_parfor）。成功写 out 并返回 0；
 * 分配失败返回 -1（不动 out，调用方回退原 f32 轨）。 */
static int q4g_gemm_mat(float *__restrict out, const float *__restrict resid,
                        const uint8_t *w, const float *x, int rows, int cols, int nb) {
    int nblk = cols / 32;
    int16_t *xq = (int16_t *)malloc((size_t)nb * cols * sizeof(int16_t));
    float   *dx = (float   *)malloc((size_t)nb * nblk * sizeof(float));
    if (!xq || !dx) { free(xq); free(dx); return -1; }
    q4g_quant16(x, cols, nb, xq, dx, nblk);
    int tiles = rows / 8;
    int nt = vllm_tp_threads();
    if (nt < 1) nt = 1;
    if (nt > 64) nt = 64;
    if (tiles >= 2 && nb >= 4) {
        int8_t *p_ts = (int8_t *)malloc((size_t)nt * 8 * cols);
        float  *p_dw = (float  *)malloc((size_t)nt * 8 * nblk * sizeof(float));
        if (!p_ts || !p_dw) { free(p_ts); free(p_dw); free(xq); free(dx); return -1; }
        q4g_ctx c = { w, x, xq, dx, out, resid, rows, cols, nb, nblk, tiles,
                      p_ts, p_dw, nt };
        vllm_tp_parfor(0, tiles, q4g_tile_worker, &c);
        free(p_ts); free(p_dw);
    } else {
        q4g_ctx c = { w, x, xq, dx, out, resid, rows, cols, nb, nblk, tiles,
                      (int8_t *)malloc((size_t)8 * cols),
                      (float *)malloc((size_t)8 * nblk * sizeof(float)), 1 };
        if (!c.pool_ts || !c.pool_dw) { free(c.pool_ts); free(c.pool_dw); free(xq); free(dx); return -1; }
        for (int g = 0; g < tiles; g++) q4g_tile_worker(&c, g);
        free(c.pool_ts); free(c.pool_dw);
    }
    if (q4g_verify_env()) {
        static int vprint = 0;
        if (vprint < 8) {
            vprint++;
            int step = rows / 16; if (step < 1) step = 1;
            float mx = 0.0f, am = 0.0f;
            int cnt = 0;
            for (int p = 0; p < nb; p++) {
                const float *xp = x + (size_t)p * cols;
                for (int r = 0; r < rows; r += step) {
                    float ref = st_q4_row_dot(w, r, xp, nblk, 1);
                    float got = out[(size_t)p * rows + r]
                              - (resid ? resid[(size_t)p * rows + r] : 0.0f);
                    float df = fabsf(got - ref);
                    float ar = fabsf(ref);
                    if (df > mx) mx = df;
                    if (ar > am) am = ar;
                    cnt++;
                }
            }
            /* 关键指标 = maxAbs 相对输出量级 |ref|max 的占比（逐点相对差会被近零 ref 放大，
             * 不用）。int8 激活量化（max-abs/127）的预期占比是 ~1e-3 量级。 */
            printf("[Q4G-VERIFY] rows=%d cols=%d nb=%d n=%d maxAbs=%.6g |ref|max=%.6g "
                   "=> %.4g%% of scale\n",
                   rows, cols, nb, cnt, mx, am, am > 0.0f ? 100.0f * mx / am : 0.0f);
            fflush(stdout);
        }
    }
    free(xq); free(dx);
    return 0;
}
#endif /* __AVX2__ && ST_ARCH_X86 */

/* Q8_0 单行点积的列带变体：只对 [cb0, cb0+cbe) 的 32-col 块做点积。
 * （MoE down 稀疏 decode：输出行 i 只需命中选中专家的列带，避免全行浪费。） */
static float st_q8_band_dot(const uint8_t *w, int r, const float *x,
                            int cb0, int cbe, int n_blocks, int mode) {
    float acc = 0.0f;
    if (mode == 0) {
        const uint8_t *pr = w + (size_t)r * (size_t)n_blocks * 34;
        for (int b = cb0; b < cb0 + cbe; b++) {
            const uint8_t *blk = pr + (size_t)b * 34;
            float d = st_f16scale(blk);
            const int8_t *qs = (const int8_t *)(blk + 2);
            const float *xs = x + (size_t)(b - cb0) * 32;
            float s = 0.0f;
            for (int j = 0; j < 32; j++) s += (float)qs[j] * xs[j];
            acc += d * s;
        }
    } else if (mode == 1) {
        int m = r & 3;
        const uint8_t *g = w + (size_t)(r >> 2) * (size_t)n_blocks * 136;
        for (int b = cb0; b < cb0 + cbe; b++) {
            const uint8_t *out = g + (size_t)b * 136;
            float d = st_f16scale(out + 2 * m);
            const float *xs = x + (size_t)(b - cb0) * 32;
            float s = 0.0f;
            for (int k = 0; k < 8; k++)
                for (int i = 0; i < 4; i++)
                    s += (float)(int8_t)out[8 + k * 16 + m * 4 + i] * xs[k * 4 + i];
            acc += d * s;
        }
    } else {
        int m = r & 7;
        const uint8_t *g = w + (size_t)(r >> 3) * (size_t)n_blocks * 272;
        for (int b = cb0; b < cb0 + cbe; b++) {
            const uint8_t *out = g + (size_t)b * 272;
            float d = st_f16scale(out + 2 * m);
            const float *xs = x + (size_t)(b - cb0) * 32;
            float s = 0.0f;
            for (int k = 0; k < 8; k++)
                for (int i = 0; i < 4; i++)
                    s += (float)(int8_t)out[16 + k * 32 + m * 4 + i] * xs[k * 4 + i];
            acc += d * s;
        }
    }
    return acc;
}

static void dyn_matvec_q8_neon(float *__restrict out, const uint8_t *__restrict q8_w,
                               const float *__restrict x, int rows, int cols) {
    int n_blocks = cols / 32;
    int mode = st_q8_layout();
#if defined(__AVX2__) && ST_ARCH_X86
    if (q8_dense_simd_ok() && mode == 2) {
        q8_dense_rows_avx2_tp(q8_w, x, rows, n_blocks, out, NULL);
        return;
    }
#endif
    for (int r = 0; r < rows; r++)
        out[r] = st_q8_row_dot(q8_w, r, x, n_blocks, mode);
}

/* ---- M4（down 执行结构重构 2026-09-10）：x86 q4 decode 行分块 TP ----
 * 镜像 q8_dense_rows_avx2_tp（P1-2）：行/4 行组按线程静态分块，行内点积
 * 累加序不变、各线程写不相交输出 → 位级一致（A≡C）。仅 decode n_batch==1
 * 路径与通用行点积（lm）使用；行串行序完全不改（预填充/小矩阵不受影响）。 */
#if defined(__AVX2__) && ST_ARCH_X86
typedef struct {
    const uint8_t *w; const float *x;
    float *out; const float *resid;
    int n_blocks; int mode;
} q4_rows_tp_ctx;
static void q4_rows_dot_worker(void *ctx_, int r) {   /* 通用行：st_q4_row_dot */
    q4_rows_tp_ctx *c = ctx_;
    float d = st_q4_row_dot(c->w, r, c->x, c->n_blocks, c->mode);
    c->out[r] = c->resid ? c->resid[r] + d : d;
}
static void q4_dense4_worker(void *ctx_, int r4) {    /* mode1 4x4 行组：q4_dense4_x86 */
    q4_rows_tp_ctx *c = ctx_;
    float o4[4];
    q4_dense4_x86(c->w, r4, c->x, c->n_blocks, o4);
    float *op = c->out + (size_t)r4 * 4;
    if (c->resid) {
        for (int i2 = 0; i2 < 4; i2++)
            op[i2] = c->resid[(size_t)r4 * 4 + i2] + o4[i2];
    } else {
        for (int i2 = 0; i2 < 4; i2++) op[i2] = o4[i2];
    }
}
typedef struct {
    const uint8_t *q, *k, *v; const float *x;
    float *qo, *ko, *vo; int n_blocks; int kv4;
} q4_qkv_tp_ctx;
static void q4_qkv_dense4_worker(void *ctx_, int r4) {
    q4_qkv_tp_ctx *c = ctx_;
    float o4[4];
    q4_dense4_x86(c->q, r4, c->x, c->n_blocks, o4);
    size_t base = (size_t)r4 * 4;
    if (r4 < c->kv4) {
        float k4[4], v4[4];
        q4_dense4_x86(c->k, r4, c->x, c->n_blocks, k4);
        q4_dense4_x86(c->v, r4, c->x, c->n_blocks, v4);
        for (int i2 = 0; i2 < 4; i2++) {
            c->qo[base + i2] = o4[i2];
            c->ko[base + i2] = k4[i2];
            c->vo[base + i2] = v4[i2];
        }
    } else {
        for (int i2 = 0; i2 < 4; i2++) c->qo[base + i2] = o4[i2];
    }
}
#endif /* __AVX2__ && ST_ARCH_X86 */
static void dyn_matvec_q4_q8_neon(float *__restrict out, const uint8_t *__restrict q4_w,
                                  const float *__restrict x, int rows, int cols) {
    int n_blocks = cols / 32;
    int mode = st_q4_layout();
#if defined(__AVX2__) && ST_ARCH_X86
    /* lm/dense-q4 单 matvec：mode1(4x4) 用 4 行组 q4_dense4_x86（同 qkv/o fast 路径；
     * 行并行 + 4 行组核内，替代逐行 st_q4_row_dot）。A≡C：行内点积序不变。 */
    if (q4_dense_simd_ok() && mode == 1 && (rows & 3) == 0 &&
        vllm_tp_threads() > 1 && (rows / 4) >= 2) {
        q4_rows_tp_ctx c = { q4_w, x, out, NULL, n_blocks, mode };
        vllm_tp_parfor(0, rows / 4, q4_dense4_worker, &c);
        return;
    }
    if (vllm_tp_threads() > 1 && rows >= 256) {
        q4_rows_tp_ctx c = { q4_w, x, out, NULL, n_blocks, mode };
        vllm_tp_parfor(0, rows, q4_rows_dot_worker, &c);
        return;
    }
#endif
    for (int r = 0; r < rows; r++)
        out[r] = st_q4_row_dot(q4_w, r, x, n_blocks, mode);
}

/* ---- 11 个 batched 推理热内核的 x86 实现（NEON 同名函数换核）----
 * 行点积见上（st_q8_row_dot / st_q4_row_dot：布局感知，按 st_q8_layout /
 * st_q4_layout 分派 legacy / 4x4 / 8x8）。
 * 输出一律 token-major：out[t*rows + r]（与 NEON worker 写入一致）。
 * f32 直算（权重反量化×f32 激活），与 ARM 的 int8-dot 归约顺序不同，
 * 属文档化跨架构近似；PPL/贪心文本口径验收。 */

static void dyn_matvec_q8_fused_qkv_batched_neon(
    float *__restrict q_out, float *__restrict k_out, float *__restrict v_out,
    const uint8_t *__restrict q8_q, const uint8_t *__restrict q8_k,
    const uint8_t *__restrict q8_v, const float *__restrict x_batch,
    int q_rows, int kv_rows, int cols, int n_batch) {
    int n_blocks = cols / 32;
    int mode = st_q8_layout();
#if defined(__AVX2__) && ST_ARCH_X86
    if (q8_dense_simd_ok() && mode == 2) {
        if (n_batch == 1) {
            /* P1-2 深挖：q/k/v 合并为一次全层行分派（行空间 = q+k+v，段跨界切开）。 */
            const uint8_t *w3[3] = { q8_q, q8_k, q8_v };
            float *d3[3] = { q_out, k_out, v_out };
            int r3[3] = { q_rows, kv_rows, kv_rows };
            q8_dense_multi_avx2_tp(w3, x_batch, n_blocks, d3, NULL, r3, 3,
                                   q_rows + 2 * kv_rows);
            return;
        }
        if (n_batch >= 8 &&
            q8_prefill_gemm_ok(q_rows > kv_rows ? q_rows : kv_rows, cols, n_batch)) {
            /* P2-1 M2：int8-激活 GEMM 新轨（激活量化有损，与 f32 逐 token 轨声明分叉；
             * VLLM_GEMM_LEGACY=1 回退下方逐 token。分配失败亦自动回退。） */
            if (q8g_gemm_mat(q_out, NULL, q8_q, x_batch, q_rows, cols, n_batch) == 0 &&
                q8g_gemm_mat(k_out, NULL, q8_k, x_batch, kv_rows, cols, n_batch) == 0 &&
                q8g_gemm_mat(v_out, NULL, q8_v, x_batch, kv_rows, cols, n_batch) == 0)
                return;
        }
        for (int t = 0; t < n_batch; t++) {
            const float *xt = x_batch + (size_t)t * cols;
            q8_dense_rows_avx2(q8_q, xt, q_rows, n_blocks,
                               q_out + (size_t)t * q_rows, NULL);
            q8_dense_rows_avx2(q8_k, xt, kv_rows, n_blocks,
                               k_out + (size_t)t * kv_rows, NULL);
            q8_dense_rows_avx2(q8_v, xt, kv_rows, n_blocks,
                               v_out + (size_t)t * kv_rows, NULL);
        }
        return;
    }
#endif
    for (int t = 0; t < n_batch; t++) {
        const float *xt = x_batch + (size_t)t * cols;
        for (int r = 0; r < kv_rows; r++) {
            q_out[(size_t)t * q_rows + r] = st_q8_row_dot(q8_q, r, xt, n_blocks, mode);
            k_out[(size_t)t * kv_rows + r] = st_q8_row_dot(q8_k, r, xt, n_blocks, mode);
            v_out[(size_t)t * kv_rows + r] = st_q8_row_dot(q8_v, r, xt, n_blocks, mode);
        }
        for (int r = kv_rows; r < q_rows; r++)
            q_out[(size_t)t * q_rows + r] = st_q8_row_dot(q8_q, r, xt, n_blocks, mode);
    }
}
static void dyn_matvec_q8_fused_o_residual_batched_neon(
    float *__restrict x_batch, const float *__restrict residual_batch,
    const uint8_t *__restrict q8_o, const float *__restrict attn_batch,
    int rows, int cols, int n_batch) {
    int n_blocks = cols / 32;
    int mode = st_q8_layout();
#if defined(__AVX2__) && ST_ARCH_X86
    if (q8_dense_simd_ok() && mode == 2) {
        if (n_batch == 1) {
            q8_dense_rows_avx2_tp(q8_o, attn_batch, rows, n_blocks,
                                  x_batch, residual_batch);
            return;
        }
        if (n_batch >= 8 && q8_prefill_gemm_ok(rows, cols, n_batch)) {
            if (q8g_gemm_mat(x_batch, residual_batch, q8_o, attn_batch,
                             rows, cols, n_batch) == 0)
                return;
        }
        for (int t = 0; t < n_batch; t++) {
            const float *at = attn_batch + (size_t)t * cols;
            q8_dense_rows_avx2(q8_o, at, rows, n_blocks,
                               x_batch + (size_t)t * rows,
                               residual_batch + (size_t)t * rows);
        }
        return;
    }
#endif
    for (int t = 0; t < n_batch; t++) {
        const float *at = attn_batch + (size_t)t * cols;
        for (int r = 0; r < rows; r++)
            x_batch[(size_t)t * rows + r] =
                residual_batch[(size_t)t * rows + r] +
                st_q8_row_dot(q8_o, r, at, n_blocks, mode);
    }
}
static void dyn_matvec_q8_fused_gate_up_batched_neon(
    float *__restrict gate_out, float *__restrict up_out,
    const uint8_t *__restrict q8_gate, const uint8_t *__restrict q8_up,
    const float *__restrict x_batch, int rows, int cols, int n_batch) {
    int n_blocks = cols / 32;
    int mode = st_q8_layout();
#if defined(__AVX2__) && ST_ARCH_X86
    if (q8_dense_simd_ok() && mode == 2) {
        if (n_batch == 1) {
            /* P1-2 深挖：gate/up 合并为一次全层行分派。 */
            const uint8_t *w2[2] = { q8_gate, q8_up };
            float *d2[2] = { gate_out, up_out };
            int r2[2] = { rows, rows };
            q8_dense_multi_avx2_tp(w2, x_batch, n_blocks, d2, NULL, r2, 2, 2 * rows);
            return;
        }
        if (n_batch >= 8 && q8_prefill_gemm_ok(rows, cols, n_batch)) {
            /* gate/up 共享同一量化激活缓冲（q8g_gemm_mat 内部各自量化一次）。 */
            if (q8g_gemm_mat(gate_out, NULL, q8_gate, x_batch, rows, cols, n_batch) == 0 &&
                q8g_gemm_mat(up_out, NULL, q8_up, x_batch, rows, cols, n_batch) == 0)
                return;
        }
        for (int t = 0; t < n_batch; t++) {
            const float *xt = x_batch + (size_t)t * cols;
            q8_dense_rows_avx2(q8_gate, xt, rows, n_blocks,
                               gate_out + (size_t)t * rows, NULL);
            q8_dense_rows_avx2(q8_up, xt, rows, n_blocks,
                               up_out + (size_t)t * rows, NULL);
        }
        return;
    }
#endif
    for (int t = 0; t < n_batch; t++) {
        const float *xt = x_batch + (size_t)t * cols;
        for (int r = 0; r < rows; r++) {
            gate_out[(size_t)t * rows + r] = st_q8_row_dot(q8_gate, r, xt, n_blocks, mode);
            up_out[(size_t)t * rows + r]   = st_q8_row_dot(q8_up,   r, xt, n_blocks, mode);
        }
    }
}
static void dyn_matvec_q8_fused_down_residual_batched_neon(
    float *__restrict x_batch, const float *__restrict residual_batch,
    const uint8_t *__restrict q8_down, const float *__restrict activated_batch,
    int hidden_dim, int ffn_dim, int n_batch) {
    int n_blocks = ffn_dim / 32;
    int mode = st_q8_layout();
#if defined(__AVX2__) && ST_ARCH_X86
    if (q8_dense_simd_ok() && mode == 2) {
        if (n_batch == 1) {
            q8_dense_rows_avx2_tp(q8_down, activated_batch, hidden_dim, n_blocks,
                                  x_batch, residual_batch);
            return;
        }
        if (n_batch >= 8 && q8_prefill_gemm_ok(hidden_dim, ffn_dim, n_batch)) {
            if (q8g_gemm_mat(x_batch, residual_batch, q8_down, activated_batch,
                             hidden_dim, ffn_dim, n_batch) == 0)
                return;
        }
        for (int t = 0; t < n_batch; t++) {
            const float *at = activated_batch + (size_t)t * ffn_dim;
            q8_dense_rows_avx2(q8_down, at, hidden_dim, n_blocks,
                               x_batch + (size_t)t * hidden_dim,
                               residual_batch + (size_t)t * hidden_dim);
        }
        return;
    }
#endif
    for (int t = 0; t < n_batch; t++) {
        const float *at = activated_batch + (size_t)t * ffn_dim;
        for (int j = 0; j < hidden_dim; j++)
            x_batch[(size_t)t * hidden_dim + j] =
                residual_batch[(size_t)t * hidden_dim + j] +
                st_q8_row_dot(q8_down, j, at, n_blocks, mode);
    }
}

/* ---- Step1（d 相位2b/a2，2026-09-10）：x86 prefill 批>1 曾单线程逐 token
 * （qkv/o），8 线程空转（QKV t4≈t8 实证）。改全池 vllm_tp_parfor 覆盖
 * (token × 4行组) 平面：每输出元素单 worker 同核同序 → A≡C 位级不变。 */
typedef struct {
    const uint8_t *q, *k, *v; const float *xb;
    float *qo, *ko, *vo;
    int n_blocks, q_rows, kv_rows, kv4, qr4;
} st1_qkv_ctx;
static void st1_qkv_worker(void *c_, int idx) {
    st1_qkv_ctx *c = c_;
    int t = idx / c->qr4, r4 = idx - t * c->qr4;
    const float *xt = c->xb + (size_t)t * (c->n_blocks * 32);
    if (r4 < c->kv4) {
        float o4[4], k4[4], v4[4];
        q4_dense4_x86(c->q, r4, xt, c->n_blocks, o4);
        q4_dense4_x86(c->k, r4, xt, c->n_blocks, k4);
        q4_dense4_x86(c->v, r4, xt, c->n_blocks, v4);
        for (int i2 = 0; i2 < 4; i2++) {
            c->qo[(size_t)t * c->q_rows + r4 * 4 + i2] = o4[i2];
            c->ko[(size_t)t * c->kv_rows + r4 * 4 + i2] = k4[i2];
            c->vo[(size_t)t * c->kv_rows + r4 * 4 + i2] = v4[i2];
        }
    } else {
        float o4[4];
        q4_dense4_x86(c->q, r4, xt, c->n_blocks, o4);
        for (int i2 = 0; i2 < 4; i2++)
            c->qo[(size_t)t * c->q_rows + r4 * 4 + i2] = o4[i2];
    }
}
typedef struct {
    const uint8_t *o; const float *attn, *resid;
    float *x;
    int n_blocks, rows;
} st1_o_ctx;
static void st1_o_worker(void *c_, int idx) {
    st1_o_ctx *c = c_;
    int qr4 = c->rows / 4;
    int t = idx / qr4, r4 = idx - t * qr4;
    const float *at = c->attn + (size_t)t * (c->n_blocks * 32);
    float *xt = c->x + (size_t)t * c->rows;
    const float *rt = c->resid + (size_t)t * c->rows;
    float o4[4];
    q4_dense4_x86(c->o, r4, at, c->n_blocks, o4);
    for (int i2 = 0; i2 < 4; i2++)
        xt[r4 * 4 + i2] = rt[r4 * 4 + i2] + o4[i2];
}
/* 稠密 FFN gate/up 4 行组 worker（decode n_batch==1 与 prefill 批>1 共用）：
 * 接线到 q4_dense4_x86，标量↔SIMD 位级一致（链序同：块外层 b → k 内层）。
 * idx 展平为 (token × 4行组)，每输出元素单 worker 同核同序 → A≡C。 */
typedef struct {
    const uint8_t *g, *u; const float *xb;
    float *go, *uo;
    int n_blocks, rows, rr4;
} st1_gateup_ctx;
static void st1_gateup_worker(void *c_, int idx) {
    st1_gateup_ctx *c = c_;
    int t = idx / c->rr4, r4 = idx - t * c->rr4;
    const float *xt = c->xb + (size_t)t * (c->n_blocks * 32);
    float g4[4], u4[4];
    q4_dense4_x86(c->g, r4, xt, c->n_blocks, g4);
    q4_dense4_x86(c->u, r4, xt, c->n_blocks, u4);
    float *gp = c->go + (size_t)t * c->rows + (size_t)r4 * 4;
    float *up = c->uo + (size_t)t * c->rows + (size_t)r4 * 4;
    for (int i2 = 0; i2 < 4; i2++) { gp[i2] = g4[i2]; up[i2] = u4[i2]; }
}
/* 稠密 FFN down(+残差) 4 行组 worker（decode n_batch==1 与 prefill 批>1 共用）：
 * 接线到 q4_dense4_x86，标量↔SIMD 位级一致（链序同：块外层 b → k 内层）。 */
typedef struct {
    const uint8_t *d; const float *ab, *resid;
    float *x; int n_blocks, hidden, ffn, hr4;
} st1_down_ctx;
static void st1_down_worker(void *c_, int idx) {
    st1_down_ctx *c = c_;
    int t = idx / c->hr4, r4 = idx - t * c->hr4;
    const float *at = c->ab + (size_t)t * c->ffn;
    const float *rt = c->resid + (size_t)t * c->hidden;
    float *xt = c->x + (size_t)t * c->hidden;
    float o4[4];
    q4_dense4_x86(c->d, r4, at, c->n_blocks, o4);
    for (int i2 = 0; i2 < 4; i2++)
        xt[(size_t)r4 * 4 + i2] = rt[(size_t)r4 * 4 + i2] + o4[i2];
}
static void dyn_matvec_q4_q8_fused_qkv_batched_neon(
    float *__restrict q_out, float *__restrict k_out, float *__restrict v_out,
    const uint8_t *__restrict q4_q, const uint8_t *__restrict q4_k,
    const uint8_t *__restrict q4_v, const float *__restrict x_batch,
    int q_rows, int kv_rows, int cols, int n_batch) {
    int n_blocks = cols / 32;
    int mode = st_q4_layout();
#if defined(__AVX2__) && ST_ARCH_X86
    if (q4_dense_simd_ok() && mode == 1 && (q_rows & 3) == 0 && (kv_rows & 3) == 0) {
        /* 新轨（§9.20）：q4 int16-激活 GEMM，权重每 tile 重排一次、复用全部 token。 */
        if (n_batch > 1 && q4g_prefill_gemm_ok(q_rows, cols, n_batch) &&
            q4g_prefill_gemm_ok(kv_rows, cols, n_batch) &&
            q4g_gemm_mat(q_out, NULL, q4_q, x_batch, q_rows, cols, n_batch) == 0 &&
            q4g_gemm_mat(k_out, NULL, q4_k, x_batch, kv_rows, cols, n_batch) == 0 &&
            q4g_gemm_mat(v_out, NULL, q4_v, x_batch, kv_rows, cols, n_batch) == 0)
            return;
        if (n_batch == 1 && vllm_tp_threads() > 1 && (q_rows / 4) >= 2) {
            /* M4：decode 单 token qkv 行分块（位级一致：行内点积序不变） */
            q4_qkv_tp_ctx c = { q4_q, q4_k, q4_v, x_batch, q_out, k_out, v_out,
                                n_blocks, kv_rows / 4 };
            vllm_tp_parfor(0, q_rows / 4, q4_qkv_dense4_worker, &c);
            return;
        }
        if (n_batch > 1 && vllm_tp_threads() > 1) {
            /* Step1：批>1 全池并行 (token×行组)，A≡C（每元素同核同序） */
            st1_qkv_ctx sc = { q4_q, q4_k, q4_v, x_batch, q_out, k_out, v_out,
                               n_blocks, q_rows, kv_rows, kv_rows / 4,
                               q_rows / 4 };
            vllm_tp_parfor(0, n_batch * (q_rows / 4), st1_qkv_worker, &sc);
            return;
        }
        for (int t = 0; t < n_batch; t++) {
            const float *xt = x_batch + (size_t)t * cols;
            int r4;
            for (r4 = 0; r4 < kv_rows / 4; r4++) {
                float o4[4], k4[4], v4[4];
                q4_dense4_x86(q4_q, r4, xt, n_blocks, o4);
                q4_dense4_x86(q4_k, r4, xt, n_blocks, k4);
                q4_dense4_x86(q4_v, r4, xt, n_blocks, v4);
                for (int i2 = 0; i2 < 4; i2++) {
                    q_out[(size_t)t * q_rows + r4 * 4 + i2] = o4[i2];
                    k_out[(size_t)t * kv_rows + r4 * 4 + i2] = k4[i2];
                    v_out[(size_t)t * kv_rows + r4 * 4 + i2] = v4[i2];
                }
            }
            for (; r4 < q_rows / 4; r4++) {
                float o4[4];
                q4_dense4_x86(q4_q, r4, xt, n_blocks, o4);
                for (int i2 = 0; i2 < 4; i2++)
                    q_out[(size_t)t * q_rows + r4 * 4 + i2] = o4[i2];
            }
        }
        return;
    }
#endif
    for (int t = 0; t < n_batch; t++) {
        const float *xt = x_batch + (size_t)t * cols;
        for (int r = 0; r < kv_rows; r++) {
            q_out[(size_t)t * q_rows + r] = st_q4_row_dot(q4_q, r, xt, n_blocks, mode);
            k_out[(size_t)t * kv_rows + r] = st_q4_row_dot(q4_k, r, xt, n_blocks, mode);
            v_out[(size_t)t * kv_rows + r] = st_q4_row_dot(q4_v, r, xt, n_blocks, mode);
        }
        for (int r = kv_rows; r < q_rows; r++)
            q_out[(size_t)t * q_rows + r] = st_q4_row_dot(q4_q, r, xt, n_blocks, mode);
    }
}
static void dyn_matvec_q4_q8_fused_o_residual_batched_neon(
    float *__restrict x_batch, const float *__restrict residual_batch,
    const uint8_t *__restrict q4_o, const float *__restrict attn_batch,
    int rows, int cols, int n_batch) {
    int n_blocks = cols / 32;
    int mode = st_q4_layout();
#if defined(__AVX2__) && ST_ARCH_X86
    if (q4_dense_simd_ok() && mode == 1 && (rows & 3) == 0) {
        /* 新轨（§9.20）：q4 int16-激活 GEMM（O + 残差）。 */
        if (n_batch > 1 && q4g_prefill_gemm_ok(rows, cols, n_batch) &&
            q4g_gemm_mat(x_batch, residual_batch, q4_o, attn_batch,
                         rows, cols, n_batch) == 0)
            return;
        if (n_batch == 1 && vllm_tp_threads() > 1 && (rows / 4) >= 2) {
            /* M4：decode 单 token o(+残差) 行分块（位级一致） */
            q4_rows_tp_ctx c = { q4_o, attn_batch, x_batch, residual_batch,
                                 n_blocks, mode };
            vllm_tp_parfor(0, rows / 4, q4_dense4_worker, &c);
            return;
        }
        if (n_batch > 1 && vllm_tp_threads() > 1) {
            /* Step1：批>1 全池并行 (token×行组)，A≡C（每元素同核同序） */
            st1_o_ctx sc = { q4_o, attn_batch, residual_batch, x_batch,
                             n_blocks, rows };
            vllm_tp_parfor(0, n_batch * (rows / 4), st1_o_worker, &sc);
            return;
        }
        for (int t = 0; t < n_batch; t++) {
            const float *at = attn_batch + (size_t)t * cols;
            const float *rt = residual_batch + (size_t)t * rows;
            float *xt = x_batch + (size_t)t * rows;
            for (int r4 = 0; r4 < rows / 4; r4++) {
                float o4[4];
                q4_dense4_x86(q4_o, r4, at, n_blocks, o4);
                for (int i2 = 0; i2 < 4; i2++)
                    xt[r4 * 4 + i2] = rt[r4 * 4 + i2] + o4[i2];
            }
        }
        return;
    }
#endif
    for (int t = 0; t < n_batch; t++) {
        const float *at = attn_batch + (size_t)t * cols;
        for (int r = 0; r < rows; r++)
            x_batch[(size_t)t * rows + r] =
                residual_batch[(size_t)t * rows + r] +
                st_q4_row_dot(q4_o, r, at, n_blocks, mode);
    }
}
static void dyn_matvec_q4_q8_fused_gate_up_batched_neon(
    float *__restrict gate_out, float *__restrict up_out,
    const uint8_t *__restrict q4_gate, const uint8_t *__restrict q4_up,
    const float *__restrict x_batch, int rows, int cols, int n_batch) {
    int n_blocks = cols / 32;
    int mode = st_q4_layout();
#if defined(__AVX2__) && ST_ARCH_X86
    /* 接线到 q4_dense4_x86（4 行组核，lane=行），替换标量逐行 st_q4_row_dot；
     * 标量↔SIMD 位级一致（链序同：块外层 b → k 内层）。rows 非 4 整除回落标量。 */
    if (q4_dense_simd_ok() && mode == 1 && (rows & 3) == 0) {
        /* 新轨（§9.20）：q4 int16-激活 GEMM（gate/up 各自量化一次激活，同 q8 轨行为）。 */
        if (n_batch > 1 && q4g_prefill_gemm_ok(rows, cols, n_batch) &&
            q4g_gemm_mat(gate_out, NULL, q4_gate, x_batch, rows, cols, n_batch) == 0 &&
            q4g_gemm_mat(up_out, NULL, q4_up, x_batch, rows, cols, n_batch) == 0)
            return;
        if (n_batch == 1 && vllm_tp_threads() > 1 && (rows / 4) >= 2) {
            /* decode 单 token：4 行组行分块（位级一致：行内点积序不变） */
            st1_gateup_ctx c = { q4_gate, q4_up, x_batch, gate_out, up_out,
                                 n_blocks, rows, rows / 4 };
            vllm_tp_parfor(0, rows / 4, st1_gateup_worker, &c);
            return;
        }
        if (n_batch > 1 && vllm_tp_threads() > 1) {
            /* prefill：全池并行 (token × 4行组)，A≡C（每元素同核同序） */
            st1_gateup_ctx c = { q4_gate, q4_up, x_batch, gate_out, up_out,
                                 n_blocks, rows, rows / 4 };
            vllm_tp_parfor(0, n_batch * (rows / 4), st1_gateup_worker, &c);
            return;
        }
        for (int t = 0; t < n_batch; t++) {
            const float *xt = x_batch + (size_t)t * cols;
            float *gt = gate_out + (size_t)t * rows;
            float *ut = up_out + (size_t)t * rows;
            for (int r4 = 0; r4 < rows / 4; r4++) {
                float g4[4], u4[4];
                q4_dense4_x86(q4_gate, r4, xt, n_blocks, g4);
                q4_dense4_x86(q4_up,   r4, xt, n_blocks, u4);
                for (int i2 = 0; i2 < 4; i2++) {
                    gt[r4 * 4 + i2] = g4[i2];
                    ut[r4 * 4 + i2] = u4[i2];
                }
            }
        }
        return;
    }
#endif
    for (int t = 0; t < n_batch; t++) {
        const float *xt = x_batch + (size_t)t * cols;
        for (int r = 0; r < rows; r++) {
            gate_out[(size_t)t * rows + r] = st_q4_row_dot(q4_gate, r, xt, n_blocks, mode);
            up_out[(size_t)t * rows + r]   = st_q4_row_dot(q4_up,   r, xt, n_blocks, mode);
        }
    }
}
static void dyn_matvec_q4_q8_fused_down_residual_batched_neon(
    float *__restrict x_batch, const float *__restrict residual_batch,
    const uint8_t *__restrict q4_down, const float *__restrict activated_batch,
    int hidden_dim, int ffn_dim, int n_batch) {
    int n_blocks = ffn_dim / 32;
    int mode = st_q4_layout();
#if defined(__AVX2__) && ST_ARCH_X86
    /* 接线到 q4_dense4_x86（4 行组核，lane=行），替换标量逐行 st_q4_row_dot；
     * 标量↔SIMD 位级一致（链序同：块外层 b → k 内层）。hidden 非 4 整除回落标量。 */
    if (q4_dense_simd_ok() && mode == 1 && (hidden_dim & 3) == 0) {
        /* 新轨（§9.20）：q4 int16-激活 GEMM（down + 残差）。 */
        if (n_batch > 1 && q4g_prefill_gemm_ok(hidden_dim, ffn_dim, n_batch) &&
            q4g_gemm_mat(x_batch, residual_batch, q4_down, activated_batch,
                         hidden_dim, ffn_dim, n_batch) == 0)
            return;
        if (n_batch == 1 && vllm_tp_threads() > 1 && (hidden_dim / 4) >= 2) {
            /* decode 单 token：4 行组行分块（位级一致：行内点积序不变） */
            st1_down_ctx c = { q4_down, activated_batch, residual_batch, x_batch,
                               n_blocks, hidden_dim, ffn_dim, hidden_dim / 4 };
            vllm_tp_parfor(0, hidden_dim / 4, st1_down_worker, &c);
            return;
        }
        if (n_batch > 1 && vllm_tp_threads() > 1) {
            /* prefill：全池并行 (token × 4行组)，A≡C（每元素同核同序） */
            st1_down_ctx c = { q4_down, activated_batch, residual_batch, x_batch,
                               n_blocks, hidden_dim, ffn_dim, hidden_dim / 4 };
            vllm_tp_parfor(0, n_batch * (hidden_dim / 4), st1_down_worker, &c);
            return;
        }
        for (int t = 0; t < n_batch; t++) {
            const float *at = activated_batch + (size_t)t * ffn_dim;
            const float *rt = residual_batch + (size_t)t * hidden_dim;
            float *xt = x_batch + (size_t)t * hidden_dim;
            for (int r4 = 0; r4 < hidden_dim / 4; r4++) {
                float o4[4];
                q4_dense4_x86(q4_down, r4, at, n_blocks, o4);
                for (int i2 = 0; i2 < 4; i2++)
                    xt[r4 * 4 + i2] = rt[r4 * 4 + i2] + o4[i2];
            }
        }
        return;
    }
#endif
    for (int t = 0; t < n_batch; t++) {
        const float *at = activated_batch + (size_t)t * ffn_dim;
        for (int j = 0; j < hidden_dim; j++)
            x_batch[(size_t)t * hidden_dim + j] =
                residual_batch[(size_t)t * hidden_dim + j] +
                st_q4_row_dot(q4_down, j, at, n_blocks, mode);
    }
}

/* ---- 多头 attention（全量 packed / sparse top-k / INT8-KV flash）x86 版 ----
 * 与 NEON 版共享缓冲布局契约：scores 每 head 占 4 行（ha*4*score_stride，
 * 避免并行 head 写冲突）；imp_head[ha*score_stride+s] 累加 softmax 权重；
 * 因果：token t 只见 s < prev_len+t+1。exp 用 expf（非 exp_neon4 多项式，
 * 跨架构近似；-ffast-math 语义下一致）。 */

/* 单 query × 单 head：对 0..n-1 写 sc[]（点积×scale）并回传 max。
 * restrict 承诺（调用方保证 q/kp_head/sc 互不重叠）——与原串行实现的形参
 * 处于同一别名假设下，恢复其代码生成质量（数值零改动）。 */
static float st_attn_qk_scalar(const float *restrict q, const float *restrict kp_head,
                               float *restrict sc, int n, int hd, float scale) {
    float mx = -1e9f;
    for (int s = 0; s < n; s++) {
        const float *kp = kp_head + (size_t)s * hd;
        float dot = 0.0f;
        for (int i = 0; i < hd; i++) dot += q[i] * kp[i];
        sc[s] = dot * scale;
        if (sc[s] > mx) mx = sc[s];
    }
    return mx;
}
/* softmax（sc[0..n) 原位变权重）+ VKQ 加权累加到 o[]。 */
static void st_attn_softmax_vkq_scalar(float *restrict sc, const float *restrict vp_head,
                                       float *restrict o, int n, int hd, float mx,
                                       float *restrict imp_row) {
    float sum_exp = 0.0f;
    for (int s = 0; s < n; s++) {
        sc[s] = expf(sc[s] - mx);
        sum_exp += sc[s];
    }
    float inv_sum = 1.0f / sum_exp;
    for (int i = 0; i < hd; i++) o[i] = 0.0f;
    for (int s = 0; s < n; s++) {
        float wgt = sc[s] * inv_sum;
        if (imp_row) imp_row[s] += wgt;
        const float *vp = vp_head + (size_t)s * hd;
        for (int i = 0; i < hd; i++) o[i] += wgt * vp[i];
    }
}

/* ==== §9.33 x86 QK「跨位置」SIMD（位级安全的向量化）====
 * 实测（微基准，与引擎内一致）：attention 里 **QK 占 79.8%**，且内层是
 * `dot += q[i]*kp[i]` 的**单链标量 FMA**，只有 ~1 FMA/cycle（9.4 GFLOP/s）；
 * SM+VKQ 占 20.2%（37 GFLOP/s，其 i 循环已由 GCC 向量化）。
 * 位级安全的做法：把**位置 s 铺到 SIMD 通道**，而不是把 i 铺进去做树形归约——
 * 每个通道内部仍是严格 i 升序、表达式与标量实现逐字同形 → 每条 score 逐位相同。
 * 代价：K 需按 [kv头][s/8 块][i][8] 重排一次（每层 O(n·hd·nkv)，约 2 MB，可忽略）。
 * 保守起见（不改数值），S=V 侧与 softmax 保持原样，只用 K 的重排。 */
static __thread float *g_kb = NULL;
static __thread size_t g_kb_cap = 0;

static float *st_attn_kb_reserve(size_t need) {
    if (g_kb_cap < need) {
        float *p = (float *)malloc(need * sizeof(float));
        if (!p) return NULL;
        free(g_kb); g_kb = p; g_kb_cap = need;
    }
    return g_kb;
}

/* K → 块 8 重排：kb[kh][b][i][j] = k_pack[kh][b*8+j][i]，末块不足 8 个位置补 0。 */
static void st_attn_k_block8(const float *restrict k_pack, float *restrict kb,
                            int n, int seq_stride, int nkv, int hd) {
    int nb8 = (n + 7) >> 3;
    for (int kh = 0; kh < nkv; kh++) {
        const float *src = k_pack + (size_t)kh * seq_stride * hd;
        float *dst = kb + (size_t)kh * nb8 * (size_t)hd * 8;
        for (int b = 0; b < nb8; b++) {
            int s0 = b * 8, m = n - s0;
            if (m > 8) m = 8;
            float *d = dst + (size_t)b * (size_t)hd * 8;
            for (int i = 0; i < hd; i++) {
                float *dr = d + (size_t)i * 8;
                for (int j = 0; j < 8; j++)
                    dr[j] = (j < m) ? src[(size_t)(s0 + j) * hd + i] : 0.0f;
            }
        }
    }
}

/* 单 query × 单 head（blocked K）：16 路主循环（2 条独立链）+ 8 路 + 标量尾。
 * 逐位等价于 st_attn_qk_scalar：每通道 i 升序、同样的乘/加两次舍入与 `*scale`；
 * mx 用升序扫描（max 精确且与顺序无关，保持同式）。
 * ▲ 必须用 mul+add 而**不是** FMA：本构建下标量参考核编译成 `vmulss`+`vaddss`
 *   （源码 `dot += q[i]*kp[i]` 未做 FMA 收缩），换成 FMA 会把两次舍入并成一次 →
 *   结果不再逐位相同（首版即踩此坑）。
 * ▲ 但**只写 mul+add 不够**：对向量内建 GCC 仍会把 `add(acc, mul(a,b))` 融合成
 *   `vfmadd`（标量那侧不融合、向量这侧融合——已用独立对照程序确认，n=1..1025
 *   逐元素比对：纯 mul+add 有 408504 处不符）。故对乘积累加空 asm 屏障
 *   `__asm__("" : "+x"(p))`，成本≈0，把乘的结果物化，彻底阻断融合；
 *   加屏障后不匹配数 **0**。 */
static float st_attn_qk_block8(const float *restrict q, const float *restrict kb_head,
                               float *restrict sc, int n, int hd, float scale) {
    const __m256 vscale = _mm256_set1_ps(scale);
    int s = 0, b = 0;
    for (; s + 16 <= n; s += 16, b += 2) {
        const float *k0 = kb_head + (size_t)b * (size_t)hd * 8;
        const float *k1 = k0 + (size_t)hd * 8;
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        for (int i = 0; i < hd; i++) {
            __m256 qi = _mm256_set1_ps(q[i]);
            __m256 p0 = _mm256_mul_ps(qi, _mm256_loadu_ps(k0 + (size_t)i * 8));
            __m256 p1 = _mm256_mul_ps(qi, _mm256_loadu_ps(k1 + (size_t)i * 8));
            __asm__ __volatile__("" : "+x"(p0));
            __asm__ __volatile__("" : "+x"(p1));
            a0 = _mm256_add_ps(a0, p0);
            a1 = _mm256_add_ps(a1, p1);
        }
        _mm256_storeu_ps(sc + s, _mm256_mul_ps(a0, vscale));
        _mm256_storeu_ps(sc + s + 8, _mm256_mul_ps(a1, vscale));
    }
    if (s + 8 <= n) {
        const float *k0 = kb_head + (size_t)b * (size_t)hd * 8;
        __m256 a0 = _mm256_setzero_ps();
        for (int i = 0; i < hd; i++) {
            __m256 qi = _mm256_set1_ps(q[i]);
            __m256 p = _mm256_mul_ps(qi, _mm256_loadu_ps(k0 + (size_t)i * 8));
            __asm__ __volatile__("" : "+x"(p));
            a0 = _mm256_add_ps(a0, p);
        }
        _mm256_storeu_ps(sc + s, _mm256_mul_ps(a0, vscale));
        s += 8;
    }
    for (; s < n; s++) {   /* 尾部位置：标量同形 */
        const float *kb = kb_head + (size_t)(s >> 3) * (size_t)hd * 8 + (s & 7);
        float dot = 0.0f;
        for (int i = 0; i < hd; i++) dot += q[i] * kb[(size_t)i * 8];
        sc[s] = dot * scale;
    }
    float mx = -1e9f;
    for (int t = 0; t < n; t++) if (sc[t] > mx) mx = sc[t];
    return mx;
}

/* ==== x86 prefill attention：按 head 并行（2026-09-13 续优化，见 §9.30）
 * 数值零改动：仅把原串行体的 `ha` 循环搬进 worker——每个 head 仍调用**同一组**
 * 标量参考核（st_attn_qk_scalar / st_attn_softmax_vkq_scalar），累加顺序一字未改；
 * 缓冲本就按 head 段隔离（scores / imp_head 各占 ha 段，attn_out 按 (t,ha) 索引），
 * 头与头之间无共享写点 → 结果与原串行实现**逐位相同**。
 * 依据：scores_buf 分配为 [max_seq * nh * 4]，注释即 "per-head rows for parallel"
 * （当初就是为并行 head 预留的布局）。
 * A/B：VLLM_ATTN_SERIAL=1 强制走原串行路径（同一二进制内对照）。 */
typedef struct {
    float *attn_out;
    const float *q_buf;
    const float *k_pack;
    const float *kb;            /* §9.33 块 8 重排的 K（NULL → 退回标量 QK） */
    int kb8;                    /* 每 kv 头的块数 = ceil(n/8) */
    const float *v_pack;
    int nb, prev_len, seq_stride, nh, nkv, hd;
    float scale;
    float *scores;
    int score_stride;
    float *imp_head;
} st_attn_x86_pctx;

/* §9.33 诊断（`VLLM_ATTN_WORK=1`）：统计本函数实际处理的 score 元素总数 Σ_ha Σ_t n，
 * 用来核对「因果稠密 attention」的 FLOP 模型——此前按该模型反推的 GFLOP/s 同时越过
 * 了单线程标量的下限与 8 核 AVX2 的上限，说明模型或真实工作量必有一错。 */
static __thread long long g_attn_work[256];

static void st_attn_x86_head_worker(void *c_, int ha) {
    st_attn_x86_pctx *c = (st_attn_x86_pctx *)c_;
    /* 字段→局部（原串行实现里这些是形参，恒在寄存器中）；并对 Q/K/V/out/
     * scores/imp 六类缓冲补回 restrict（调用方保证互不重叠）。两者都只为
     * 恢复原代码生成质量，不改变任何运算顺序。 */
    const int nb = c->nb, prev_len = c->prev_len, hd = c->hd, nh = c->nh;
    const float scale = c->scale;
    const int kh = (ha * c->nkv) / nh;
    const float *restrict kp_head = c->k_pack + (size_t)kh * c->seq_stride * hd;
    /* §9.33：有块 8 重排的 K 就走跨位置 SIMD（位级等价），否则退回标量。 */
    const float *restrict kb_head = c->kb
        ? c->kb + (size_t)kh * (size_t)c->kb8 * (size_t)hd * 8 : NULL;
    const float *restrict vp_head = c->v_pack + (size_t)kh * c->seq_stride * hd;
    float *restrict scb = c->scores + (size_t)ha * 4 * c->score_stride;
    float *restrict impr = c->imp_head ? c->imp_head + (size_t)ha * c->score_stride : NULL;
    long long wsum = 0;
    for (int t = 0; t < nb; t++) {
        const float *restrict q = c->q_buf + ((size_t)t * nh + ha) * hd;
        float *restrict o = c->attn_out + ((size_t)t * nh + ha) * hd;
        int n = prev_len + t + 1;
        wsum += n;
        float *restrict sc = scb;   /* 每个 head 只用自己那段的 k=0 行，隔离已满足 */
        float mx = kb_head ? st_attn_qk_block8(q, kb_head, sc, n, hd, scale)
                           : st_attn_qk_scalar(q, kp_head, sc, n, hd, scale);
        st_attn_softmax_vkq_scalar(sc, vp_head, o, n, hd, mx, impr);
    }
    if (ha < 256) g_attn_work[ha] = wsum;
}

static void st_attn_batched_packed_neon(
    float *restrict attn_out,        /* [nb * nh * hd] */
    const float *restrict q_buf,     /* [nb * nh * hd] */
    const float *restrict k_pack,    /* [nkv * seq_stride * hd] */
    const float *restrict v_pack,    /* [nkv * seq_stride * hd] */
    int nb, int prev_len, int seq_stride,
    int nh, int nkv, int hd, float scale,
    float *restrict scores,          /* [nh*4 * score_stride] */
    int score_stride,
    float *restrict imp_head)        /* [nh * score_stride] or NULL */
{
    st_attn_x86_pctx c;
    /* §9.33：先做 K 的块 8 重排（串行、在 parfor 之前完成），worker 只读。
     * VLLM_ATTN_QKSCALAR=1 可退回标量 QK，做同二进制同轮 A/B。 */
    const float *kb = NULL; int kb8 = 0;
    {
        static int qkscalar_env = -1;
        if (qkscalar_env < 0) {
            const char *e = getenv("VLLM_ATTN_QKSCALAR");
            qkscalar_env = (e && e[0] == '1') ? 1 : 0;
        }
        if (!qkscalar_env) {
            int nn = prev_len + nb;
            kb8 = (nn + 7) >> 3;
            float *buf = st_attn_kb_reserve((size_t)nkv * (size_t)kb8 * (size_t)hd * 8);
            if (buf) {
                st_attn_k_block8(k_pack, buf, nn, seq_stride, nkv, hd);
                kb = buf;
            }
        }
    }
    c.attn_out = attn_out; c.q_buf = q_buf; c.k_pack = k_pack; c.v_pack = v_pack;
    c.kb = kb; c.kb8 = kb8;
    c.nb = nb; c.prev_len = prev_len; c.seq_stride = seq_stride;
    c.nh = nh; c.nkv = nkv; c.hd = hd; c.scale = scale;
    c.scores = scores; c.score_stride = score_stride; c.imp_head = imp_head;

    /* A/B 开关：强制原串行路径（默认走并行）。 */
    static int serial_env = -1;
    if (serial_env < 0) {
        const char *e = getenv("VLLM_ATTN_SERIAL");
        serial_env = (e && e[0] == '1') ? 1 : 0;
    }
    if (serial_env) {
        for (int ha = 0; ha < nh; ha++) st_attn_x86_head_worker(&c, ha);
    } else {
        /* 单线程池时 vllm_tp_parfor 自动退化为串行 → 行为与旧实现一致 */
        vllm_tp_parfor(0, nh, st_attn_x86_head_worker, &c);
    }
    /* §9.33 工作量探针（默认关）。 */
    {
        static int work_env = -1;
        if (work_env < 0) {
            const char *e = getenv("VLLM_ATTN_WORK");
            work_env = (e && e[0] == '1') ? 1 : 0;
        }
        if (work_env) {
            long long s = 0;
            for (int ha = 0; ha < nh && ha < 256; ha++) s += g_attn_work[ha];
            fprintf(stderr, "[ATTNWORK] nh=%d nb=%d prev_len=%d stride=%d hd=%d sumN=%lld\n",
                    nh, nb, prev_len, seq_stride, hd, s);
        }
    }
}

static void flash_attn_single_q_q8_neon(
    float *restrict attn_out,           /* [hd] 归一化输出 */
    const float *restrict q,            /* [hd] query 激活 */
    int8_t *const *k_cache_q8,          /* 每层 INT8 K 块数组 */
    int8_t *const *v_cache_q8,
    const float *restrict kscale,       /* k_scale[l] + kh */
    const float *restrict vscale,
    int seq_len, int kv_dim, int kv_bs, int nkv,
    int kh_off, int hd, float scale)
{
    const int nblk = hd / 32;
    const float KVQ_INV = 1.0f / 127.0f;
    float qsc[8];
    int8_t qi[256];

    for (int b = 0; b < nblk; b++) {
        float qm = 0.0f;
        const float *qb = q + b * 32;
        for (int i = 0; i < 32; i++) {
            float a = fabsf(qb[i]);
            if (a > qm) qm = a;
        }
        if (qm < 1e-6f) qm = 1.0f;
        qsc[b] = qm * KVQ_INV;
        float iq = 127.0f / qm;
        for (int i = 0; i < 32; i++) {
            float v = qb[i] * iq;
            int iv = (int)(v + (v >= 0.0f ? 0.5f : -0.5f));
            qi[b * 32 + i] = (int8_t)((iv > 127) ? 127 : ((iv < -128) ? -128 : iv));
        }
    }
    for (int i = 0; i < hd; i++) attn_out[i] = 0.0f;
    float M = -1e9f, S = 0.0f;

    for (int t = 0; t < seq_len; t++) {
        const int8_t *kt = k_cache_q8[t / kv_bs] + (size_t)(t % kv_bs) * kv_dim + kh_off;
        const int8_t *vt = v_cache_q8[t / kv_bs] + (size_t)(t % kv_bs) * kv_dim + kh_off;
        float ks = kscale[(size_t)t * nkv] * KVQ_INV;
        float vs = vscale[(size_t)t * nkv] * KVQ_INV;

        float s = 0.0f;
        for (int b = 0; b < nblk; b++) {
            int32_t dot32 = 0;
            for (int i = 0; i < 32; i++) dot32 += (int32_t)qi[b * 32 + i] * (int32_t)kt[b * 32 + i];
            s += (float)dot32 * qsc[b];
        }
        s = s * ks * scale;

        float vsf;
        if (s > M) {
            float ms = expf(M - s);
            M = s;
            S *= ms;
            for (int i = 0; i < hd; i++) attn_out[i] *= ms;
            vsf = 1.0f;
        } else {
            vsf = expf(s - M);
        }
        S += vsf;

        float wv = vsf * vs;
        for (int i = 0; i < hd; i++)
            attn_out[i] += (float)vt[i] * wv;
    }
    float inv = (S > 0.0f) ? 1.0f / S : 0.0f;
    for (int i = 0; i < hd; i++) attn_out[i] *= inv;
}

static void st_attn_batched_packed_sparse_neon(
    float *restrict attn_out,        /* [nb * nh * hd] */
    const float *restrict q_buf,     /* [nb * nh * hd] */
    const float *restrict k_pack,    /* [nkv * seq_stride * hd] */
    const float *restrict v_pack,    /* [nkv * seq_stride * hd] */
    int nb, int prev_len, int seq_stride,
    int nh, int nkv, int hd, float scale,
    float *restrict scores,          /* [nh*4 * score_stride] */
    int score_stride, int bs, int k_blocks, int n_probe,
    float *restrict imp_head)        /* [nh * score_stride] or NULL */
{
    int max_n = prev_len + nb;
    int n_blocks = (max_n + bs - 1) / bs;
    if (n_blocks <= 0) return;
    if (k_blocks < 1) k_blocks = 1;
    if (n_probe < 1) n_probe = 1;

    for (int ha = 0; ha < nh; ha++) {
        int kh = (ha * nkv) / nh;
        const float *kp_head = k_pack + (size_t)kh * seq_stride * hd;
        const float *vp_head = v_pack + (size_t)kh * seq_stride * hd;
        float *scb = scores + (size_t)ha * 4 * score_stride;
        float *impr = imp_head ? imp_head + (size_t)ha * score_stride : NULL;
        const float *q_rep = q_buf + ((size_t)(nb - 1) * nh + ha) * hd;

        float *probe = (float *)malloc((size_t)n_blocks * sizeof(float));
        uint8_t *used = (uint8_t *)malloc((size_t)n_blocks);
        int *sel = (int *)malloc((size_t)n_blocks * sizeof(int));
        if (!probe || !used || !sel) {
            free(probe); free(used); free(sel);
            continue;
        }
        memset(used, 0, (size_t)n_blocks);

        for (int b = 0; b < n_blocks; b++) {
            float best = -1e30f;
            int p0 = b * bs;
            int p_end = p0 + bs; if (p_end > max_n) p_end = max_n;
            int step = (p_end - p0) / n_probe; if (step < 1) step = 1;
            for (int ps = p0; ps < p_end; ps += step) {
                const float *kp = kp_head + (size_t)ps * hd;
                float dot = 0.0f;
                for (int i = 0; i < hd; i++) dot += q_rep[i] * kp[i];
                if (dot > best) best = dot;
            }
            probe[b] = best;
        }

        int nsel = 0;
        for (int i = 0; i < n_blocks && nsel < k_blocks; i++) {
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
        if (nsel < n_blocks && !used[n_blocks - 1]) {
            int low = 0;
            for (int i = 1; i < nsel; i++)
                if (probe[sel[i]] < probe[sel[low]]) low = i;
            used[sel[low]] = 0;
            sel[low] = n_blocks - 1;
            used[n_blocks - 1] = 1;
        }

        for (int t = 0; t < nb; t++) {
            const float *q = q_buf + ((size_t)t * nh + ha) * hd;
            float *o = attn_out + ((size_t)t * nh + ha) * hd;
            int n = prev_len + t + 1;
            float *sc = scb;
            float mx = -1e9f;
            for (int si = 0; si < nsel; si++) {
                int p0 = sel[si] * bs;
                int p1 = p0 + bs; if (p1 > n) p1 = n;
                for (int s = p0; s < p1; s++) {
                    const float *kp = kp_head + (size_t)s * hd;
                    float dot = 0.0f;
                    for (int i = 0; i < hd; i++) dot += q[i] * kp[i];
                    sc[s] = dot * scale;
                    if (sc[s] > mx) mx = sc[s];
                }
            }
            float sum_exp = 0.0f;
            for (int si = 0; si < nsel; si++) {
                int p0 = sel[si] * bs;
                int p1 = p0 + bs; if (p1 > n) p1 = n;
                for (int s = p0; s < p1; s++) {
                    sc[s] = expf(sc[s] - mx);
                    sum_exp += sc[s];
                }
            }
            float inv_sum = 1.0f / sum_exp;
            for (int i = 0; i < hd; i++) o[i] = 0.0f;
            for (int si = 0; si < nsel; si++) {
                int p0 = sel[si] * bs;
                int p1 = p0 + bs; if (p1 > n) p1 = n;
                for (int s = p0; s < p1; s++) {
                    float wgt = sc[s] * inv_sum;
                    if (impr) impr[s] += wgt;
                    const float *vp = vp_head + (size_t)s * hd;
                    for (int i = 0; i < hd; i++) o[i] += wgt * vp[i];
                }
            }
        }
        free(probe); free(used); free(sel);
    }
}

/* ---- 单 token（decode）fused matvec 的 x86 实现（*_neon 换核）---- */
static void dyn_matvec_q8_fused_gate_up_neon(
    float *__restrict gate_out, float *__restrict up_out,
    const uint8_t *__restrict q8_gate, const uint8_t *__restrict q8_up,
    const float *__restrict x, int rows, int cols) {
    int n_blocks = cols / 32;
    int mode = st_q8_layout();
    for (int r = 0; r < rows; r++) {
        gate_out[r] = st_q8_row_dot(q8_gate, r, x, n_blocks, mode);
        up_out[r]   = st_q8_row_dot(q8_up,   r, x, n_blocks, mode);
    }
}
static void dyn_matvec_q4_q8_fused_qkv_neon(
    float *__restrict q_out, float *__restrict k_out, float *__restrict v_out,
    const uint8_t *__restrict q4_q, const uint8_t *__restrict q4_k,
    const uint8_t *__restrict q4_v, const float *__restrict x,
    int q_rows, int kv_rows, int cols) {
    int n_blocks = cols / 32;
    int mode = st_q4_layout();
    for (int r = 0; r < kv_rows; r++) {
        q_out[r] = st_q4_row_dot(q4_q, r, x, n_blocks, mode);
        k_out[r] = st_q4_row_dot(q4_k, r, x, n_blocks, mode);
        v_out[r] = st_q4_row_dot(q4_v, r, x, n_blocks, mode);
    }
    for (int r = kv_rows; r < q_rows; r++)
        q_out[r] = st_q4_row_dot(q4_q, r, x, n_blocks, mode);
}
static void dyn_matvec_q4_q8_fused_gate_up_neon(
    float *__restrict gate_out, float *__restrict up_out,
    const uint8_t *__restrict q4_gate, const uint8_t *__restrict q4_up,
    const float *__restrict x, int rows, int cols) {
    int n_blocks = cols / 32;
    int mode = st_q4_layout();
    for (int r = 0; r < rows; r++) {
        gate_out[r] = st_q4_row_dot(q4_gate, r, x, n_blocks, mode);
        up_out[r]   = st_q4_row_dot(q4_up,   r, x, n_blocks, mode);
    }
}
static void dyn_matvec_q4_q8_fused_down_residual_neon(
    float *__restrict x, const float *__restrict residual,
    const uint8_t *__restrict q4_down, const float *__restrict activated,
    int hidden_dim, int ffn_dim) {
    int n_blocks = ffn_dim / 32;
    int mode = st_q4_layout();
    for (int j = 0; j < hidden_dim; j++)
        x[j] = residual[j] +
               st_q4_row_dot(q4_down, j, activated, n_blocks, mode);
}
#endif /* ST_ARCH_X86 && !ST_HAVE_NEON */

/* ARM 版 MOEPROF 插桩（x86 版 moe_prof_enabled/mpt_* 定义在 x86 NEON-compat
 * 映射层内）。VLLM_MOE_PROF=1 逐 token 累计 gu/dn/ffn 墙钟，架构无关。 */
#if defined(__aarch64__) && defined(ST_HAVE_NEON) && !ST_ARCH_X86
static int moe_prof_enabled(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("VLLM_MOE_PROF"); v = (e && e[0] == '1') ? 1 : 0; }
    return v;
}
static double mpt_ff = 0, mpt_gu = 0, mpt_dn = 0;
static int mpt_n = 0;
#endif

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

/* Dense matvec 精度选择（decode 侧）：q4 是否优先。
 * - ARM(aarch64)：dense-Q4 有 NEON repack 内核，解码带宽省一半 → q4 优先。
 * - x86：dense-Q4 单 token 路径无 AVX2（P1-1 只落地了 dense-Q8 AVX2），
 *   让 decode 在有 Q8 张量时优先走 Q8（与 prefill 的 g_st_prefill_q8 语义对齐），
 *   否则回退 Q4。仅影响“q4/q8 同时存在”的模型（如 dual VQF）。 */
static inline int st_dense_q4_first(void) {
#if ST_ARCH_X86 && !ST_HAVE_NEON
    return 0;
#else
    return 1;
#endif
}

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
 * operator models exported by tools/npu/npu_export_ops.py; the safetensors model
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


static void quantize_row_q8_0_act(const float *__restrict x,
                                  int8_t *__restrict q, float *__restrict d, int cols);

/* ---- 纯 VQF 运行时：以下无原始权重加载代码 ---- */

void st_weights_free(STModelWeights *w) {
    if (!w->is_allocated) return;
    /* VQF mmap-backed weights: all pointers alias one read-only mapping. */
    if (w->vqf_map) {
        if (w->vqf_map && w->vqf_map_len)
            st_mmap_release_view(w->vqf_map, w->vqf_map_len);
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


/* f32 → f16：IEEE 754 binary16，round-to-nearest-even，含 subnormal。
 * 旧实现 exp<=0 直接返回 0（FTZ）且尾数只截断不舍入：|x| < 2^-14 的 scale
 * 会被抹成 0（权重转换侧曾因此把 q8_gate 2.21% 的块整块清零），尾数截断还会
 * 给每个 scale 引入 ~0.1% 相对误差。本实现与 ggml GGML_FP32_TO_FP16 一致，
 * 与 vqf_convert/src/quant_kernels.c 保持位级同源（KV/预填充 q8 scale 共用）。 */
static inline uint16_t f32_to_f16_bits(float x) {
    uint32_t u; memcpy(&u, &x, 4);
    uint32_t sign = (u >> 16) & 0x8000u;
    uint32_t ef   = (u >> 23) & 0xFFu;
    uint32_t mant = u & 0x7FFFFFu;

    if (ef == 0xFFu)                          /* inf / nan */
        return (uint16_t)(sign | 0x7C00u | (mant ? 0x200u : 0u));

    int32_t e = (int32_t)ef - 127 + 15;
    if (e >= 0x1F) return (uint16_t)(sign | 0x7C00u);

    if (e <= 0) {                             /* subnormal 或 0 */
        if (e < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        uint32_t sh = (uint32_t)(14 - e);
        uint32_t half = 1u << (sh - 1);
        uint32_t rem = mant & ((1u << sh) - 1u);
        mant >>= sh;
        if (rem > half || (rem == half && (mant & 1u))) mant++;
        return (uint16_t)(sign | mant);
    }
    uint32_t rem = mant & 0x1FFFu;
    mant >>= 13;
    if (rem > 0x1000u || (rem == 0x1000u && (mant & 1u))) {
        if (++mant == 0x400u) {
            mant = 0;
            if (++e >= 0x1F) return (uint16_t)(sign | 0x7C00u);
        }
    }
    return (uint16_t)(sign | ((uint32_t)e << 10) | mant);
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
 * 载荷）由 tools/kernels/extract_llama_asm.py 机械提取（零转录风险），语义与 generic
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
#include "../../tools/kernels/llama_gemm_q4_0_4x4_asm.c"

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

/* VLLM_GEMM_LEGACY=1：恢复 §10.32 之前的 q4 batched 分派——仅 n_batch%4==0
 * 才走真 GEMM，非 4 倍数整批走 C-tile worker（旧逐 token 舍入序）。默认 0 =
 * llama 收口语义（主段 n_batch&~3 真 GEMM + 余 1-3 token GEMV 尾）。对照用，
 * 避免跨版本锚点漂移时无回退手段。 */
static int vllm_gemm_legacy_env(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("VLLM_GEMM_LEGACY");
        v = (e && e[0] == '1') ? 1 : 0;
    }
    return v;
}

static void q4x4_gemm_batched(float *__restrict out, const uint8_t *__restrict q4_w,
                              const int8_t *__restrict xq, const float *__restrict xd,
                              int rows, int cols, int n_batch,
                              const float *__restrict residual) {
    const int TILE = 3;
    /* M4d: llama.cpp 4x4 asm GEMM fast path. 条件：4x4 repack 开启、权重行 %4。
     * n_batch 对齐 llama repack 分派（forward_mul_mat_one_chunk）：nrows>3 时
     * 主段 n_batch&~3 走真 GEMM、余 1-3 token 走 C tile worker（GEMV 尾），
     * 不再让非 4 倍数整批塌缩进 C worker —— 与 q8x8 零填充语义同一收口思路
     * （数值边界：主段为 M4h 舍入序、尾段为 C tile 序，均为既有文档化路径）。
     * VLLM_GEMM_LEGACY=1 时恢复旧 gate（n_batch%4==0 才进真 GEMM）。 */
#if ST_NEON_DOTPROD
    if (g_st_q4_repack && (rows & 3) == 0 &&
        (vllm_gemm_legacy_env() ? (n_batch & 3) == 0 : n_batch > 3) &&
        repack_q8_0_4x4(xq, xd, n_batch & ~3, cols) == 0) {
        int nb4 = n_batch & ~3;
        st_gemm_q4_0_4x4_batched(out, q4_w, rows, cols, nb4, residual);
        int tail = n_batch - nb4;
        if (tail > 0) {
            /* GEMV 尾：token 偏移 nb4，权重/量化激活指针同源后移。 */
            int n_blocks = cols / 32;
            vllm_q4x4_gemm_ctx vc = {
                q4_w,
                xq + (size_t)nb4 * cols,
                xd + (size_t)nb4 * n_blocks,
                out + (size_t)nb4 * rows,
                residual ? residual + (size_t)nb4 * rows : NULL,
                rows, cols, n_blocks, tail, n_blocks * 18
            };
            vllm_tp_parfor(0, (rows + 3) / 4, vllm_q4x4_gemm_worker, &vc);
        }
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

#if defined(__aarch64__) && defined(ST_HAVE_NEON) && !ST_ARCH_X86
/* 单 32 列块 q8_0 激活量化（NEON，位级等价）：max|·| → scale=max/127 → 逐元素
 * (int)(x/scale+0.5f)（vdivq IEEE 正确舍入除法、向零截断、clamp ±127）。
 * biased=1 时字节再 ^0x80（AVX-512 VNNI unsigned 域），d 语义不变。 */
static inline void q8q32_block_neon(const float *__restrict xb,
                                    int8_t *__restrict qb, float *dout, int biased) {
    const float32x4_t halfv = vdupq_n_f32(0.5f);
    const int32x4_t p127 = vdupq_n_s32(127), n127 = vdupq_n_s32(-127);
    float32x4_t mx = vdupq_n_f32(1e-10f);
    for (int j = 0; j < 32; j += 4)
        mx = vmaxq_f32(mx, vabsq_f32(vld1q_f32(xb + j)));
    float max_abs = fmaxf(fmaxf(vgetq_lane_f32(mx, 0), vgetq_lane_f32(mx, 1)),
                          fmaxf(vgetq_lane_f32(mx, 2), vgetq_lane_f32(mx, 3)));
    float scale = max_abs / 127.0f;
    *dout = scale;
    float32x4_t scv = vdupq_n_f32(scale);
    for (int i = 0; i < 32; i += 8) {
        int32x4_t a = vmaxq_s32(vminq_s32(
            vcvtq_s32_f32(vaddq_f32(vdivq_f32(vld1q_f32(xb + i), scv), halfv)), p127), n127);
        int32x4_t c = vmaxq_s32(vminq_s32(
            vcvtq_s32_f32(vaddq_f32(vdivq_f32(vld1q_f32(xb + i + 4), scv), halfv)), p127), n127);
        int8x8_t out = vmovn_s16(vcombine_s16(vmovn_s32(a), vmovn_s32(c)));
        if (biased) out = veor_s8(out, vdup_n_s8((int8_t)0x80));
        vst1_s8(qb + i, out);
    }
}
#endif /* ARM NEON */

/* Quantize a f32 activation row into int8 + per-block f32 scales (q8_0 layout,
 * symmetric scale = max_abs/127). Matches the weight quantizer semantics.
 *
 * ARM (aarch64+NEON) 走位级等价 NEON 分支：max|·|（vmax 精确无舍入）、每元素
 * IEEE 正确舍入除法（vdivq_f32 与标量 fdiv 同舍入）、+0.5 后 vcvtq_s32_f32
 * （向零截断 = (int) 强转）、clamp ±127 —— 与下方标量逐元素同序，位级一致。
 * x86 / 无 NEON 编译走原标量路径（语义不动）。 */
static void quantize_row_q8_0_act(const float *__restrict x,
                                  int8_t *__restrict q, float *__restrict d, int cols) {
#if defined(__aarch64__) && defined(ST_HAVE_NEON) && !ST_ARCH_X86
    const int n_blocks = cols / 32;
    for (int b = 0; b < n_blocks; b++) {
        q8q32_block_neon(x + b * 32, q + b * 32, &d[b], 0);
    }
#else
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
#endif
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
#if defined(__aarch64__) && defined(ST_HAVE_NEON) && !ST_ARCH_X86
        q8q32_block_neon(xb,
                         c->q + (size_t)b * c->n_batch * 32 + (size_t)t * 32,
                         &c->d[(size_t)b * c->n_batch + t], 0);
#else
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
#endif
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
#if defined(__aarch64__) && defined(ST_HAVE_NEON) && !ST_ARCH_X86
        q8q32_block_neon(xb,
                         c->q + (size_t)b * c->n_batch * 32 + (size_t)t * 32,
                         &c->d[(size_t)b * c->n_batch + t], 1);
#else
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
#endif
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
/* vllm_l3.c 的 l3_fill_block_from_disk 用 L3_Q8_INV_SCALE 做 q8 量化/反量化，
 * 必须与本文件的 KVQ_INV_SCALE 同值，否则前缀重建会另起口径（位级不一致）。 */
_Static_assert(L3_Q8_INV_SCALE == KVQ_INV_SCALE, "L3 q8 scale convention drift");

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

/* ================================================================
 * P0 原型（#2 去 f32 正典）：VLLM_PREFILL_Q8CACHE=1 质量门测量开关
 *
 * 动机：q8 档下 decode 注意力已纯 int8+scale；f32 正典唯一"不可替代"的读点
 * 是 prefill 历史段注意力（st_pack_kv_heads 打包 f32 行做精确注意）。删 f32
 * 必然把该段从"精确"变"q8 近似"，故先做原型测量（NLL/PPL 门禁）再决定删除。
 *
 * 本开关把写入 f32 正典行的值改为"q8 反量化值"（dequant = q*scale/127，与
 * kv_quantize_per_head 的 round(x/max*127) 同一量化路径，禁止另起口径）：
 *   env 关 = 现状（f32 精确历史 → prefill 精确）；
 *   env 开 = f32 行≈q8 反量化（模拟"只存 q8"后 prefill 历史段将看到的值）。
 * 用法：同 prompt 下关/开各跑一次 --stream-nll，对比 NLL_MEAN（质量门）。
 * ================================================================ */
static int vllm_pf_q8cache_env(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("VLLM_PREFILL_Q8CACHE"); v = (e && e[0] == '1'); }
    return v;
}
static void kv_dequant_roundtrip(int8_t *kdst, int8_t *vdst,
                                 const float *kscale, const float *vscale,
                                 float *kf, float *vf, int nkv, int hd) {
    for (int kh = 0; kh < nkv; kh++) {
        float sk = kscale[kh] * (1.0f / KVQ_INV_SCALE);
        float sv = vscale[kh] * (1.0f / KVQ_INV_SCALE);
        int8_t *kd = kdst + (size_t)kh * hd;
        int8_t *vd = vdst + (size_t)kh * hd;
        float *k0 = kf + (size_t)kh * hd;
        float *v0 = vf + (size_t)kh * hd;
        for (int i = 0; i < hd; i++) {
            k0[i] = (float)kd[i] * sk;
            v0[i] = (float)vd[i] * sv;
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
/* ================================================================
 * KV block 惰性分配（v2：斩断 init 的"全量触页"）
 *
 * 现状：KV payload 用 calloc（小块清零 → 每个 block 的每一页在 init 就被
 * memset 触碰进 RSS），max_seq=8192 时 f32+int8 双副本 init 即 ~2.35GB。
 *
 * 改法：默认（g_xq_pages=0）改用 mmap(MAP_ANONYMOUS) 惰性零页——
 *   1) 不在 init 写 payload（首写才自然 page-fault）；
 *   2) 不再预写 0xA5 守卫 canary（本引擎 KV 路径不读守卫，见 alloc 调用点
 *      CK_TAIL/CN_TAIL 仅在 debug env 下启用）；
 *   3) 未写 block 读到 0，与旧 calloc 语义逐字节等价 → 推理输出位级不变。
 * g_xq_pages=1（ASan/越界调试档）维持旧 cf_pg_alloc 布局与 canary 不变。
 * ================================================================ */
static int g_kv_lazy = 0;   /* st_qwen_inference_init 按 g_xq_pages 置位 */

/* P3（#2 去 f32 正典物理化）：VLLM_KV_NOF32=1 → q8 档不再分配 f32 正典数组。
 * 约束：仅 use_kv_q8 + 惰性分配（非 ASan 越界调试档 g_xq_pages=1）有效；
 * q4 / 纯 f32 档与越界调试档保留双份（诚实边界，见方案文档 §7.2 P3）。 */
static int g_kv_nof32 = 0;
static void kv_nof32_poll(STQwenInferenceState *st) {
    g_kv_nof32 = 0;
    const char *nf = getenv("VLLM_KV_NOF32");
    if (nf && nf[0] == '1' && st->use_kv_q8 && !g_xq_pages) {
        g_kv_nof32 = 1;
        fprintf(stderr, "[KV] P3 nof32 mode: f32 canonical KV arrays NOT allocated\n");
        fflush(stderr);
    }
}

static void *kv_block_alloc_raw(size_t payload_bytes) {
    if (!g_kv_lazy) return cf_pg_alloc(KV_GUARD + payload_bytes + KV_GUARD);
    size_t total = KV_GUARD + payload_bytes + KV_GUARD;
#ifdef _WIN32
    return VirtualAlloc(NULL, total, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
#else
    void *p = mmap(NULL, total, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return (p == MAP_FAILED) ? NULL : p;
#endif
}
static void kv_block_free_raw(void *base, size_t payload_bytes) {
    if (!base) return;
    if (!g_kv_lazy) { cf_pg_free(base); return; }
#ifdef _WIN32
    VirtualFree(base, 0, MEM_RELEASE);
#else
    munmap(base, KV_GUARD + payload_bytes + KV_GUARD);
#endif
}
/* 导出：main.c 的 L3 逐块驱逐（on_disk）释放用（payload → base 由内部换算） */
void st_qwen_kv_free_block(void *payload, size_t payload_bytes) {
    kv_block_free_raw((uint8_t *)payload - KV_GUARD, payload_bytes);
}
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
        uint8_t *raw = (uint8_t *)kv_block_alloc_raw(blk);
        if (!raw) {
            for (int j = 0; j < b; j++) kv_block_free_raw(arr[j] - KV_GUARD, blk);
            cf_pg_free(arr);
            return NULL;
        }
        if (!g_kv_lazy) {   /* 旧(调试)布局才写 0xA5；惰性 mmap 不预触守卫页 */
            memset(raw, 0xA5, KV_GUARD);
            memset(raw + KV_GUARD + blk, 0xA5, KV_GUARD);
        }
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
        uint8_t *raw = (uint8_t *)kv_block_alloc_raw(dat);
        if (!raw) {
            for (int j = 0; j < b; j++) kv_block_free_raw(arr[j] - KV_GUARD, dat);
            cf_pg_free(arr);
            return NULL;
        }
        if (!g_kv_lazy) {   /* 旧(调试)布局才写 0xA5；惰性 mmap 不预触守卫页 */
            memset(raw, 0xA5, KV_GUARD);
            memset(raw + KV_GUARD + dat, 0xA5, KV_GUARD);
        }
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
        uint8_t *raw = (uint8_t *)kv_block_alloc_raw(dat);
        if (!raw) {
            for (int j = 0; j < b; j++) kv_block_free_raw(arr[j] - KV_GUARD, dat);
            cf_pg_free(arr);
            return NULL;
        }
        if (!g_kv_lazy) {   /* 旧(调试)布局才写 0xA5；惰性 mmap 不预触守卫页 */
            memset(raw, 0xA5, KV_GUARD);
            memset(raw + KV_GUARD + dat, 0xA5, KV_GUARD);
        }
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
    g_kv_lazy = !g_xq_pages;   /* v2: KV block 惰性 mmap（ASan/越界调试档除外） */
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
    /* v2 (#3): VLLM_KV_MAXSEQ 动态收紧物理预分配上界（edge 档 1024~2048）。
     * 语义不变：KV block 现为惰性 mmap，预分配只占虚拟地址，RSS 按实际写
     * 入增长；该旋钮进一步限制"可寻址窗口"与 scores_buf 等一次性缓冲。 */
    {
        const char *kms = getenv("VLLM_KV_MAXSEQ");
        if (kms && kms[0]) {
            int cap = atoi(kms);
            if (cap < 64) cap = 64;
            if (max_seq > cap) max_seq = cap;
        }
    }
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
    {
        const char *kvf = getenv("VLLM_KV_F32");   /* A/B：强制 f32 KV（绕 INT8 缓存） */
        if (kvf && kvf[0] == '1') { st->use_kv_q8 = 0; st->use_kv_q4 = 0; }
    }
    kv_nof32_poll(st);   /* P3: VLLM_KV_NOF32=1 → 跳过 f32 正典数组（q8 档） */

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

    if (!g_kv_nof32) {   /* P3: nof32 档不分配 f32 正典 KV 块（省 max_seq×28 层双份） */
        for (int l = 0; l < nl; l++) {
            st->k_cache[l] = alloc_kv_blocks_f32(n_blocks, bs, kv_dim, &g_cn_karr[l], &g_cn_kblk[l]);
            st->v_cache[l] = alloc_kv_blocks_f32(n_blocks, bs, kv_dim, &g_cn_varr[l], &g_cn_vblk[l]);
            if (!st->k_cache[l] || !st->v_cache[l]) {
                fprintf(stderr, "[QWEN] OOM allocating KV cache layer %d (seq=%d)\n", l, max_seq);
                st_qwen_inference_free(st);
                return -1;
            }
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
        int q8sim = (dst->use_kv_q8 && dst->k_cache_q8 && dst->v_cache_q8 &&
                     vllm_pf_q8cache_env());   /* P1a：q8 世界（f32 由 q8 重建） */
        int src_f32 = src->k_cache && src->k_cache[l];   /* P3: nof32 源无 f32 正典 */
        int dst_f32 = dst->k_cache && dst->k_cache[l];
        if (!q8sim && src_f32 && dst_f32) {
            for (int b = 0; b < nblocks; b++) {
                if (src->k_cache[l][b] && dst->k_cache[l][b])
                    memcpy(dst->k_cache[l][b], src->k_cache[l][b],
                           (size_t)bs * (size_t)kv_dim * sizeof(float));
                if (src->v_cache[l][b] && dst->v_cache[l][b])
                    memcpy(dst->v_cache[l][b], src->v_cache[l][b],
                           (size_t)bs * (size_t)kv_dim * sizeof(float));
            }
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
            if (q8sim && dst_f32) {
                /* P1a：不复制 src 精确 f32 行，改为按 q8+scale 反量化重建 dst
                 * f32 正典行 → 复制前缀与未复制前缀在同一近似口径（q8 世界）。 */
                for (int b = 0; b < nblocks; b++) {
                    if (!dst->k_cache[l][b] || !dst->k_cache_q8[l][b] ||
                        !dst->v_cache_q8[l][b]) continue;
                    for (int p = b * bs; p < (b + 1) * bs && p < prefix_len; p++) {
                        kv_dequant_roundtrip(
                            kv_row_i8(dst->k_cache_q8[l], p, kv_dim, bs),
                            kv_row_i8(dst->v_cache_q8[l], p, kv_dim, bs),
                            dst->k_scale[l] + (size_t)p * nkv,
                            dst->v_scale[l] + (size_t)p * nkv,
                            kv_row_f32(dst->k_cache[l], p, kv_dim, bs),
                            kv_row_f32(dst->v_cache[l], p, kv_dim, bs),
                            nkv, hd);
                    }
                }
            }
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

/* ================================================================
 * P2 选块 scratch arena（per-thread）
 *
 * sparse_attn_head 是「每层每 query head」调用一次（vllm_attn_head_worker），
 * 2B 配置下 28 层 × 16 head = 448 次/token，每次 6~7 次 malloc/free 并重算一遍
 * 与 head 无关的 O(seq_len) 块级重要性归约 → 单 token 约 3.1k 次分配、16 倍冗余。
 *
 * arena 按线程缓存（线程池 worker 是持久线程），容量只增不减，上界由
 * max_seq / kv_bs 决定（8K 上下文约 40KB/线程）。块级重要性归约移交给调用方
 * 在层内算一次，经 ctx 传入（见 vllm_attn_head_ctx.imp_sum）。
 * 公理：npu_ntile_1024_submit_reduction_001（消重复固定开销）、
 *       shs_axiom_tropical_semiring_v1（调度与数据局部性）。
 * ================================================================ */
typedef struct {
    float   *probe;     size_t cap_probe;
    uint8_t *used;      size_t cap_used;
    int     *sel;       size_t cap_sel;
    float   *scores;    size_t cap_scores;
    float   *imp_sum;   size_t cap_imp;
    uint8_t *l3k;       size_t cap_l3k;
    uint8_t *l3v;       size_t cap_l3v;
} SparseScratch;

static __thread SparseScratch g_ss;

/* 只增不减地保证容量；realloc 失败时保留原指针并返回 NULL（调用方退化为
 * 输出全零，与旧 OOM 路径同语义）。 */
static void *ss_grow(void **p, size_t *cap, size_t need) {
    if (need == 0) return *p;
    if (*p && *cap >= need) return *p;
    void *n = realloc(*p, need);
    if (!n) return NULL;
    *p = n;
    *cap = need;
    return n;
}

/* n_blocks = 0 表示只要 scores（非稀疏路径）；l3_chunk = 0 表示不需要 L3 暂存。
 * 返回 0 成功、-1 OOM。 */
static int sparse_scratch_ensure(size_t n_blocks, size_t seq_len, size_t l3_chunk) {
    if (seq_len &&
        !ss_grow((void **)&g_ss.scores, &g_ss.cap_scores, seq_len * sizeof(float)))
        return -1;
    if (n_blocks) {
        if (!ss_grow((void **)&g_ss.probe, &g_ss.cap_probe,
                     n_blocks * sizeof(float) + 64)) return -1;
        if (!ss_grow((void **)&g_ss.used, &g_ss.cap_used, n_blocks + 64)) return -1;
        if (!ss_grow((void **)&g_ss.sel, &g_ss.cap_sel,
                     n_blocks * sizeof(int) + 64)) return -1;
        if (!ss_grow((void **)&g_ss.imp_sum, &g_ss.cap_imp,
                     n_blocks * sizeof(float) + 64)) return -1;
    }
    if (l3_chunk) {
        if (!ss_grow((void **)&g_ss.l3k, &g_ss.cap_l3k, l3_chunk + 64)) return -1;
        if (!ss_grow((void **)&g_ss.l3v, &g_ss.cap_l3v, l3_chunk + 64)) return -1;
    }
    return 0;
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
 *
 * P2: imp_sum_pre = 调用方在层内算好的块级重要性（长度 n_blocks，与 head 无关）。
 * 传入时直接采用（省掉每 head 重算 O(seq_len)）；为 NULL 时退回按 importance
 * 自算（自检与独立调用路径保持旧语义）。scratch 一律取自 per-thread arena。
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
                             const float *__restrict imp_sum_pre,
                             float *__restrict scores,
                             const STL3State *__restrict l3,
                             int layer) {
    int hd8 = hd & ~7;
    int has_f32 = k_cache != NULL;   /* P3/P2: nof32 档无 f32 正典块数组 */
    int n_blocks = (seq_len + bs - 1) / bs;
    if (n_blocks <= 0) return;

    /* Phase-2c: per-head L3 batch staging — one block's K or V area (single
     * head) staged with a single copy, then all token dots / V accumulations
     * run over the staged buffer instead of issuing per-token reads.
     * NULL on OOM falls back to the per-token read path below. */
    int n_payloads = hd / 64;
    size_t l3_chunk = (size_t)bs * (size_t)n_payloads * L3_Q4_PAYLOAD64;

    /* Phase 2b: block-array row accessor (blocks may live in RAM or on disk). */
#define BLK(rows, t) ((rows)[(t) / bs] + (size_t)((t) % bs) * (size_t)kv_dim)
#define L3BM(b) ((l3) ? &l3->blocks[(size_t)layer * l3->max_blocks + (b)] : NULL)

    /* P2: scratch 全走 per-thread arena，热路径不再分配（旧实现每 head 6 次
     * malloc/free）。OOM 语义与旧一致：输出留零，由调用方保证 attn_out 已清零。 */
    if (sparse_scratch_ensure((size_t)n_blocks, 0, l3 ? l3_chunk : 0) != 0) return;
    uint8_t *l3k_buf = l3 ? g_ss.l3k : NULL;
    uint8_t *l3v_buf = l3 ? g_ss.l3v : NULL;
    float   *probe = g_ss.probe;
    uint8_t *used  = g_ss.used;
    int     *sel   = g_ss.sel;
    /* 块级重要性：优先采用调用方层内算好的结果（与 head 无关）；未提供则自算。 */
    float       *imp_own = NULL;
    const float *imp_sum = imp_sum_pre;
    if (!imp_sum && importance) {
        imp_own = g_ss.imp_sum;
        memset(imp_own, 0, (size_t)n_blocks * sizeof(float));
        imp_sum = imp_own;
    }
    /* DEBUG: canary the small scratch so an over-write is caught right here
     * (the LFH small-bucket free list was being corrupted during sparse
     * decode with evicted blocks; every other allocation is canaried). */
    uint8_t *tail_probe = (uint8_t *)probe + (size_t)n_blocks * sizeof(float);
    uint8_t *tail_used  = used + (size_t)n_blocks;
    uint8_t *tail_sel   = (uint8_t *)sel + (size_t)n_blocks * sizeof(int);
    uint8_t *tail_imp   = imp_own ? (uint8_t *)imp_own + (size_t)n_blocks * sizeof(float) : NULL;
    memset(tail_probe, 0xA5, 64);
    memset(tail_used, 0xA5, 64);
    memset(tail_sel, 0xA5, 64);
    if (tail_imp) memset(tail_imp, 0xA5, 64);
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
            if (has_f32 && k_cache[ps / bs]) {
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
            } else if (use_q8 && k_q8 && k_q8[ps / bs]) {
                /* P2 nof32：f32 正典不存在 → 逐元素反量化（q*(scale/127)，与
                 * kv_dequant_roundtrip 写 f32 行位级一致）再走与 f32 相同的
                 * FMA → probe 值与 f32 行=dequant（ON 档）完全一致。 */
                const int8_t *k8t = BLK(k_q8, ps) + (size_t)kh * hd;
                float sk = kscale[(size_t)ps * (size_t)nkv + (size_t)kh]
                         * (1.0f / KVQ_INV_SCALE);
                float ktmp[128];
                for (int i = 0; i < hd && i < 128; i++)
                    ktmp[i] = (float)k8t[i] * sk;
                float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = vdupq_n_f32(0.0f);
                for (int i = 0; i < hd8; i += 8) {
                    acc0 = vfmaq_f32(acc0, vld1q_f32(qt + i), vld1q_f32(ktmp + i));
                    acc1 = vfmaq_f32(acc1, vld1q_f32(qt + i + 4), vld1q_f32(ktmp + i + 4));
                }
                float32x4_t hs = vaddq_f32(acc0, acc1);
                hs = vpaddq_f32(hs, hs);
                hs = vpaddq_f32(hs, hs);
                dot = vgetq_lane_f32(hs, 0);
                for (int i = hd8; i < hd; i++) dot += qt[i] * ktmp[i];
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
        if (imp_own)
            for (int ps = p0; ps < p_end; ps++) imp_own[b] += importance[ps];
    }

    /* (1b) prefill-importance pre-selection: reserve half the block budget for
     * the blocks the exact prefill pass actually attended (rank-based, so a
     * needle block with modest but real attention mass is retained regardless
     * of probe magnitude). Deterministic: tie-break by block index. */
    int n_imp = 0;
    if (imp_sum) {
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
        /* 块级 RAM 源判定：f32 世界看 k_cache[b]；nof32 世界看 q8（无 f32） */
        int blk_evd = has_f32 ? (k_cache[b] == NULL)
                              : (!k_q8 || !k_q8[b]);
        /* Phase-2c: stage the whole evicted block's K area (this head) in one
         * copy, then score every token from the staged buffer. */
        const uint8_t *kbuf = NULL;
        if (blk_evd && l3k_buf)
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
            } else if (blk_evd) {
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
        int blk_evd = has_f32 ? (v_cache[b] == NULL)
                              : (!v_q8 || !v_q8[b]);
        /* Phase-2c: stage the whole evicted block's V area (this head) in one
         * copy, then accumulate every token from the staged buffer. */
        const uint8_t *vbuf = NULL;
        if (blk_evd && l3v_buf)
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
            } else if (blk_evd) {
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

    /* P2: scratch 归 per-thread arena 所有，热路径不释放（跨 head/层复用）。 */
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
                         0, seq_len, kv_dim, nkv, kh, hd, scale, bs, 9999, 4,
                         NULL, NULL, scores, NULL, 0);
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
                         0, seq_len, kv_dim, nkv, kh, hd, scale, bs, 4, 4,
                         NULL, NULL, scores, NULL, 0);
        sparse_attn_head(got2 + (size_t)h * hd, q + (size_t)h * hd,
                         k_blocks, v_blocks, NULL, NULL, NULL, NULL,
                         0, seq_len, kv_dim, nkv, kh, hd, scale, bs, 4, 4,
                         NULL, NULL, scores, NULL, 0);
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
                         1, seq_len, kv_dim, nkv, kh, hd, scale, bs, 9999, 4,
                         NULL, NULL, scores, NULL, 0);
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
        /* M4j：dense packed 每 head 用 4 行 scores（ha*4+0..3，stride=
         * score_stride）→ sc_buf 需 nh*4*seq_stride 行（早期按 nh*seq_stride
         * 分配 → 堆越界，case5 合成测试 SIGBUS/SEGV；2026-09-07 P4 收口）。 */
        float *sc_buf = (float *)malloc((size_t)nh * 4 * seq_stride * sizeof(float));
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
                             0, seq_len, kv_dim, nkv, 0, hd, scale, bs, 2, 4,
                             NULL, NULL, scores, NULL, 0);
            sparse_attn_head(o_imp, q2, k2_blocks, v2_blocks, NULL, NULL, NULL, NULL,
                             0, seq_len, kv_dim, nkv, 0, hd, scale, bs, 2, 4,
                             imp, NULL, scores, NULL, 0);

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
                                 bs, 9999, 4, NULL, NULL, scores, NULL, 0);
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
                                     bs, 9999, 4, NULL, NULL, scores, &l3, 0);
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

/* ④ llama 对齐：decode 长序列（seq_len>=FA_THRESHOLD、f32 KV、非 sparse）走
 * tile-packed flash 注意力时按 head 并行（32 heads > 线程数，天然并行度充足；
 * llama split-KV 的 head 内再分片仅在 线程数>head 数时有意义，本机 4 线程不适用，
 * 故取 head 级并行——每 head 独立写 attn_buf 段、无共享可变状态，位级一致）。 */
typedef struct {
    STQwenInferenceState *st;
    int nh, nkv, hd, l, seq_len, kv_dim, bs;
    float scale;
} vllm_flash_head_ctx;

static void vllm_flash_head_worker(void *ctx_, int h) {
    vllm_flash_head_ctx *c = ctx_;
    STQwenInferenceState *st = c->st;
    int kh = (h * c->nkv) / c->nh;
    flash_attn_blocked_single_q(
        st->attn_buf + (size_t)h * c->hd,
        st->q_buf + (size_t)h * c->hd,
        st->k_cache[c->l], st->v_cache[c->l],
        c->seq_len, kh * c->hd, c->kv_dim, c->hd, c->bs, c->scale);
}

typedef struct {
    STQwenInferenceState *st;
    int nh, nkv, hd, l, seq_len, kv_dim, use_q8;
    float scale;
    /* P2: 层内预计算的块级 prefill 重要性（与 head 无关），各 head 只读共享 */
    const float *imp_sum;
} vllm_attn_head_ctx;

static void vllm_attn_head_worker(void *ctx_, int h) {
    vllm_attn_head_ctx *c = ctx_;
    STQwenInferenceState *st = c->st;
    int kh = (h * c->nkv) / c->nh;
    /* Indexed by absolute token position [0, seq_len); with max_seq=8192
     * it must span the whole context. The old fixed 2048 cap overflowed the
     * stack ("stack smashing") as soon as a request exceeded 2K tokens (e.g.
     * 4K). 8K/16K contexts exceed the 8192 stack array too, so take it from the
     * per-thread arena with the same 8192 lower bound (P2: 热路径不再 malloc)。 */
    if (sparse_scratch_ensure(0, (size_t)(c->seq_len > 8192 ? c->seq_len : 8192), 0) != 0) {
        fprintf(stderr, "[ATTN] OOM: scores[%d]\n", c->seq_len);
        return;
    }
    float *scores = g_ss.scores;
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
            st->prefill_importance, c->imp_sum, scores,
            &st->l3, c->l);
        return;
    }

    if (c->use_q8) {
        /* INT8-KV flash attention（online softmax + vdot + KV 分块语义）。 */
        flash_attn_single_q_q8_neon(
            st->attn_buf + (size_t)h * c->hd, qt,
            st->k_cache_q8[c->l], st->v_cache_q8[c->l],
            st->k_scale[c->l] + kh, st->v_scale[c->l] + kh,
            c->seq_len, c->kv_dim, st->kv_bs, c->nkv, kh * c->hd, c->hd, c->scale);
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

/* ================================================================
 * qwen3_moe（MoE）稀疏专家 FFN（单 token，标量确定核）
 *
 * 语义（Qwen3MoeSparseMoeBlock，norm_topk_prob=true）：
 *   router_logits = router(x_ffn)            [ne]
 *   topk → softmax(topk_logits) → p_e        (确定性：同分取小索引)
 *   y = Σ_{e∈topk} p_e · down_e(silu(gate_e(x))·up_e(x))
 *
 * 引擎 VQF 把每层 128 专家按 4x4-repack 巨矩阵连续堆叠：
 *   gate/up : 层矩阵 rows=n_experts*ef, cols=dim → 专家 e 输出行带 e*ef..
 *   down    : 层矩阵 rows=dim, cols=n_experts*ef → 专家 e 输入列带 e*ef..
 * ef=moe_ffn 满足 ef%4==0 且 ef%32==0 → 行带/列带都不跨界 4x4 tile
 * （4 行 × 32 列），故只读 top-k 专家自身的 tile 即可：
 *   gate/up ≈ 2×(k·ef/(ne·ef))·层权重，down 同比例；DRAM 从全宽
 *   ~340MB/层 降到 ~k/ne ≈ 6%（k=8/ne=128）。
 * 权重字节 = d_f16(每行) + XOR(0x88) 后的 4bit 码（signed4 = q-8），
 * 与 repack_q4_0_4x4_inplace 字节语义逐位一致。f32 标量累加，单线程
 * 确定性（无并行累加重排）。
 * ================================================================ */
static inline float vq_f16(uint16_t h) {
    uint32_t s = (uint32_t)(h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1fu, m = h & 0x3ffu, u;
    if (e == 0) {
        if (m == 0) { u = 0; }
        else {
            int sh = 0; uint32_t mm = m;
            while (!(mm & 0x400u)) { mm <<= 1; sh++; }
            /* §9.43 修正：f16 次正规值 = (1 + f/1024) × 2^(-14-sh) → f32 指数域 = 113-sh。
             * 原式写 `127-15-sh`（=112-sh）**少 1**，即次正规 scale 被解码成 **1/2**。
             * 实测本模型 48 层 MoE 权重里次正规 scale 出现 **0 次**（计数器 s=0）→ 修正后
             * 数值逐位不变（本模型零影响），仅修正潜在正确性缺陷。 */
            u = ((uint32_t)(127 - 14 - sh) << 23) | ((mm & 0x3ffu) << 13);
        }
    } else if (e == 31) {
        u = 0x7f800000u | (m << 13);
    } else {
        u = ((e - 15 + 127) << 23) | (m << 13);
    }
    uint32_t fbits = u | s;
    float f; memcpy(&f, &fbits, 4);
    return f;
}

/* 4x4 tile (72B) 内取值（llama/repack 约定）：
 * legacy block_q4_0 的 16 qs 字节：字节 bb 低半字节=元素 bb、高半字节=元素
 * bb+16（bb=0..15）。repack 把行 ri 的 legacy 字节 bb 放入 chunk i=(bb>>2)*4+ri
 * 的第 (bb&3) 字节（已 ^0x88，读取后还原）。 */
static inline float vq4x4_w(const uint8_t *tile, int ri, int k) {
    uint16_t h; memcpy(&h, tile + 2 * ri, 2);
    int bb = k & 15;
    int hi = k >= 16;
    uint8_t s = tile[8 + (((bb >> 2) * 4 + ri) << 2) + (bb & 3)] ^ 0x88;
    int nib = hi ? (s >> 4) : (s & 15);
    return vq_f16(h) * (float)(nib - 8);
}

/* ---- q4 4x4-tile AVX2 f32 decode（MoE FFN gate/up/down 内核，llama Q4_0_4x4 式）----
 * 目标：decode 热点 st_moe_ffn_sparse_q4 目前逐元素标量（vq4x4_w 每个 (ri,k) 一次
 * 字节拆解 + f16 转换）。本内核一次处理一个 72B tile 的全部 4 行 × 32 列：
 *   tile = 4×f16 scale（每行一个，tile[0..8)）+ 64B qs（4 chunk × 16B，chunk c
 *   内字节 j = 行 (j>>2) × 子列 (j&3)，每字节低半=元素 c*4+(j&3)、高半=+16）。
 * 数值语义与标量逐元素完全一致（mul+add 不收缩、f16→f32 用同 vq_f16 精确转换），
 * 逐行 FMA 链序 = 标量 cb 外层 / k 内层 → A≡C 位级一致。
 * 开关 VLLM_MOE_Q4SIMD=0 关闭（默认开，仅 x86 AVX2）。 */
static int moe_q4_simd_ok(void) {
#if defined(__AVX2__) && ST_ARCH_X86
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("VLLM_MOE_Q4SIMD");
        v = (e && e[0] == '0') ? 0 : 1;
    }
    return v;
#elif defined(__aarch64__) && defined(ST_HAVE_NEON) && !ST_ARCH_X86
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("VLLM_MOE_Q4SIMD");
        v = (e && e[0] == '0') ? 0 : 1;
    }
    return v;
#else
    return 0;
#endif
}

/* 每行 ri 的 legacy 字节 bb（bb=4q+s）在 chunk q 的 ri*4+s 处。
 * 2026-09-13 解包优化（与 ARM §9.15/§9.16 同法）：旧版每个 k 做一次 pshufb 取 4 行
 * 字节；新版改为「整 chunk 预异或 + 每 chunk 一次 pshufb 重组 16 通道 + AVX2 一次
 * 加宽 8 通道」，每 tile 的 pshufb 从 32 次降到 8 次。数值与旧版逐位一致
 * （同一 (nib-8)、同 cvt/sub/mul scale/×x[k] 的舍入序与 k 升序累加）。 */
#if defined(__AVX2__) && ST_ARCH_X86
/* 重组索引：输出 lane 4s+ri ← chunk 内偏移 ri*4+s 的字节（ri=行, s=子列） */
static const uint8_t q4x4_grp[16] = {
    0, 4, 8, 12,   1, 5, 9, 13,   2, 6, 10, 14,   3, 7, 11, 15,
};

/* 一个 chunk → 4 个 k 的累加（k = KB+0..3）。HALF=0 低半字节、HALF=1 高半字节。
 * 一次 pshufb 得 [s0 四行|s1 四行|s2 四行|s3 四行]，两次 vpmovzxbd 得 4 个 int32x4。 */
#define Q4X4_X86_ACC_CHUNK(CQ, KB, HALF)                                       \
    do {                                                                       \
        __m128i v_ = _mm_shuffle_epi8((CQ), grp);                              \
        __m128i n_ = (HALF) ? _mm_and_si128(_mm_srli_epi16(v_, 4), m15b)       \
                            : _mm_and_si128(v_, m15b);                         \
        __m256i y0_ = _mm256_cvtepu8_epi32(n_);              /* s0,s1 */        \
        __m256i y1_ = _mm256_cvtepu8_epi32(_mm_srli_si128(n_, 8)); /* s2,s3 */ \
        __m128 f_;                                                             \
        f_ = _mm_mul_ps(_mm_sub_ps(_mm_cvtepi32_ps(_mm256_castsi256_si128(y0_)), m8), scales); \
        ac = _mm_add_ps(ac, _mm_mul_ps(f_, _mm_set1_ps(x[(KB) + 0])));         \
        f_ = _mm_mul_ps(_mm_sub_ps(_mm_cvtepi32_ps(_mm256_extracti128_si256(y0_, 1)), m8), scales); \
        ac = _mm_add_ps(ac, _mm_mul_ps(f_, _mm_set1_ps(x[(KB) + 1])));         \
        f_ = _mm_mul_ps(_mm_sub_ps(_mm_cvtepi32_ps(_mm256_castsi256_si128(y1_)), m8), scales); \
        ac = _mm_add_ps(ac, _mm_mul_ps(f_, _mm_set1_ps(x[(KB) + 2])));         \
        f_ = _mm_mul_ps(_mm_sub_ps(_mm_cvtepi32_ps(_mm256_extracti128_si256(y1_, 1)), m8), scales); \
        ac = _mm_add_ps(ac, _mm_mul_ps(f_, _mm_set1_ps(x[(KB) + 3])));         \
    } while (0)

/* 单 tile：acc4（4 lanes=4 行）+= Σ_k scale_ri*(nib-8)*x[k]，逐元素 mul+add，
 * k=0..31 与标量同序（低半字节=元素 bb、高半=bb+16；存储端已 ^0x88，读取还原）。
 * 4 个 16B chunk 每 tile 各载入一次并预异或；累加序 0..15（低半）后 16..31（高半）。 */
static inline void q4x4_tile32_x86(const uint8_t *tile, const float *x, __m128 *acc) {
    uint16_t hs[4];
    memcpy(hs, tile, 8);
    __m128 scales = _mm_setr_ps(vq_f16(hs[0]), vq_f16(hs[1]),
                                vq_f16(hs[2]), vq_f16(hs[3]));  /* 与标量同 vq_f16 */
    const uint8_t *qbase = tile + 8;
    const __m128i xor88 = _mm_set1_epi8((char)0x88);
    const __m128i m15b  = _mm_set1_epi8(0x0f);
    const __m128i grp   = _mm_loadu_si128((const __m128i *)q4x4_grp);
    const __m128 m8     = _mm_set1_ps(8.0f);
    const __m128i C0 = _mm_xor_si128(_mm_loadu_si128((const __m128i *)(qbase +  0)), xor88);
    const __m128i C1 = _mm_xor_si128(_mm_loadu_si128((const __m128i *)(qbase + 16)), xor88);
    const __m128i C2 = _mm_xor_si128(_mm_loadu_si128((const __m128i *)(qbase + 32)), xor88);
    const __m128i C3 = _mm_xor_si128(_mm_loadu_si128((const __m128i *)(qbase + 48)), xor88);
    __m128 ac = *acc;

    Q4X4_X86_ACC_CHUNK(C0,  0, 0);
    Q4X4_X86_ACC_CHUNK(C1,  4, 0);
    Q4X4_X86_ACC_CHUNK(C2,  8, 0);
    Q4X4_X86_ACC_CHUNK(C3, 12, 0);
    Q4X4_X86_ACC_CHUNK(C0, 16, 1);
    Q4X4_X86_ACC_CHUNK(C1, 20, 1);
    Q4X4_X86_ACC_CHUNK(C2, 24, 1);
    Q4X4_X86_ACC_CHUNK(C3, 28, 1);

    *acc = ac;
}
#undef Q4X4_X86_ACC_CHUNK

/* 精确轨 wf 预计算：解包一个 72B tile（4 行 × 32 列 Q4）→ 32 个 k 的 wf
 * （4 行 f32，已 ×scale）。数值与 q4x4_tile32_x86 逐 k 的 wf 逐位一致
 * （同 pshufb 解包 + cvtepi32_ps + sub8 + mul scale），供 GroupGEMM 跨 token 复用。
 * 2026-09-13：与 q4x4_tile32_x86 同步改为 chunk 级解包（pshufb 32→8 次/tile）。 */
static inline void q4x4_tile32_wf_x86(const uint8_t *tile, __m128 wf[32]) {
    uint16_t hs[4];
    memcpy(hs, tile, 8);
    __m128 scales = _mm_setr_ps(vq_f16(hs[0]), vq_f16(hs[1]),
                                vq_f16(hs[2]), vq_f16(hs[3]));
    const uint8_t *qbase = tile + 8;
    const __m128i xor88 = _mm_set1_epi8((char)0x88);
    const __m128i m15b  = _mm_set1_epi8(0x0f);
    const __m128i grp   = _mm_loadu_si128((const __m128i *)q4x4_grp);
    const __m128 m8     = _mm_set1_ps(8.0f);
    const __m128i C0 = _mm_xor_si128(_mm_loadu_si128((const __m128i *)(qbase +  0)), xor88);
    const __m128i C1 = _mm_xor_si128(_mm_loadu_si128((const __m128i *)(qbase + 16)), xor88);
    const __m128i C2 = _mm_xor_si128(_mm_loadu_si128((const __m128i *)(qbase + 32)), xor88);
    const __m128i C3 = _mm_xor_si128(_mm_loadu_si128((const __m128i *)(qbase + 48)), xor88);

#define Q4X4_X86_WF_CHUNK(CQ, KB, HALF)                                        \
    do {                                                                       \
        __m128i v_ = _mm_shuffle_epi8((CQ), grp);                              \
        __m128i n_ = (HALF) ? _mm_and_si128(_mm_srli_epi16(v_, 4), m15b)       \
                            : _mm_and_si128(v_, m15b);                         \
        __m256i y0_ = _mm256_cvtepu8_epi32(n_);              /* s0,s1 */        \
        __m256i y1_ = _mm256_cvtepu8_epi32(_mm_srli_si128(n_, 8)); /* s2,s3 */ \
        wf[(KB) + 0] = _mm_mul_ps(_mm_sub_ps(_mm_cvtepi32_ps(_mm256_castsi256_si128(y0_)), m8), scales); \
        wf[(KB) + 1] = _mm_mul_ps(_mm_sub_ps(_mm_cvtepi32_ps(_mm256_extracti128_si256(y0_, 1)), m8), scales); \
        wf[(KB) + 2] = _mm_mul_ps(_mm_sub_ps(_mm_cvtepi32_ps(_mm256_castsi256_si128(y1_)), m8), scales); \
        wf[(KB) + 3] = _mm_mul_ps(_mm_sub_ps(_mm_cvtepi32_ps(_mm256_extracti128_si256(y1_, 1)), m8), scales); \
    } while (0)

    Q4X4_X86_WF_CHUNK(C0,  0, 0);
    Q4X4_X86_WF_CHUNK(C1,  4, 0);
    Q4X4_X86_WF_CHUNK(C2,  8, 0);
    Q4X4_X86_WF_CHUNK(C3, 12, 0);
    Q4X4_X86_WF_CHUNK(C0, 16, 1);
    Q4X4_X86_WF_CHUNK(C1, 20, 1);
    Q4X4_X86_WF_CHUNK(C2, 24, 1);
    Q4X4_X86_WF_CHUNK(C3, 28, 1);
#undef Q4X4_X86_WF_CHUNK
}

/* 每 4 行组跨 nbG 个 tile（沿 d 列）累加 gate/up：gv/uv[4]（单组行）。 */
static void moe_q4_gu_group_x86(const uint8_t *gb, const uint8_t *ub, int nbG,
                                const float *x, float gv[4], float uv[4]) {
    __m128 ga = _mm_setzero_ps(), ua = _mm_setzero_ps();
    for (int cb = 0; cb < nbG; cb++) {
        q4x4_tile32_x86(gb + (size_t)cb * 72, x + (size_t)cb * 32, &ga);
        q4x4_tile32_x86(ub + (size_t)cb * 72, x + (size_t)cb * 32, &ua);
        ST_PREFETCH(gb + (size_t)(cb + 4) * 72);
        ST_PREFETCH(ub + (size_t)(cb + 4) * 72);
    }
    _mm_storeu_ps(gv, ga);
    _mm_storeu_ps(uv, ua);
}

/* 每 4 输出行组跨 cbe 个 tile（专家 down 列带）累加：y[rg*4..+3] += p*acc。 */
static void moe_q4_down_group_x86(const uint8_t *db, int cbe,
                                  const float *av, float p, float *y) {
    __m128 acc = _mm_setzero_ps();
    for (int cbi = 0; cbi < cbe; cbi++) {
        q4x4_tile32_x86(db + (size_t)cbi * 72, av + (size_t)cbi * 32, &acc);
        ST_PREFETCH(db + (size_t)(cbi + 4) * 72);
    }
    __m128 pv = _mm_set1_ps(p);
    __m128 yo = _mm_loadu_ps(y);
    _mm_storeu_ps(y, _mm_add_ps(yo, _mm_mul_ps(pv, acc)));
}

/* EP 专用：只算原始点积（**不乘 p**），写入 acc_out。
 * 供 EP 贡献回传——由协调者按 j 序显式 `fmaf` 施加 p，使其与生产核
 * `_mm_add_ps(yo, _mm_mul_ps(pv, acc))`（-O2 -mfma 下被 GCC 收缩为 FMA → 1 次
 * 舍入）**同舍入**（§9.24）。不这样做的话，EP 侧的 `C_j = p·acc` 再归约天然是
 * 两次舍入，与单机逐项差 1 ULP，`--test-moe-ep` M2 失败。 */
static void moe_q4_down_group_raw_x86(const uint8_t *db, int cbe,
                                      const float *av, float *acc_out) {
    __m128 acc = _mm_setzero_ps();
    for (int cbi = 0; cbi < cbe; cbi++) {
        q4x4_tile32_x86(db + (size_t)cbi * 72, av + (size_t)cbi * 32, &acc);
        ST_PREFETCH(db + (size_t)(cbi + 4) * 72);
    }
    _mm_storeu_ps(acc_out, acc);
}

/* ============================================================
 * actq x86（AVX2）：q4 MoE 激活量化 int8 近似轨（ARM actq 同语义，VLLM_ACTQ=1）。
 * 整数域技巧（M1/M2 门禁）：权重 nib 取真值 0..15（存储 ^8 还原），点积用
 * _mm_maddubs_epi16(u8 nib × s8 激活) 精确配对，再减 8·Σact（块级校正）——
 * dot = Σ(nib-8)·act 整数精确（nib≤15×127 无 maddubs 饱和），f32 链
 * acc += (scale_ri*xd_cb)*dot 与 ARM actq 同构。
 * 行 legacy 重组：每 4 行×32 列 tile，pshufb 掩码 actq_msk[q][ri] 逐 chunk
 * 收集行字节（dst L 属于 chunk q → src ri*4+(L&3)，否则 0x80），4 OR 合一。
 * 默认关；开关 moe_q4_actq_ok（见上，x86 分支）。
 * ============================================================ */
static __m128i actq_msk[4][4];
/* 2a-c：256b 行收集掩码（由 actq_msk 派生）。行对 p=(r0,r1)=(2p,2p+1)：
 * MA=[msk1[r1]|msk0[r0]] MB=[msk0[r1]|msk1[r0]] MC=[msk3[r1]|msk2[r0]] MD=[msk2[r1]|msk3[r0]] */
static __m256i actq_msk2[2][4];
static int actq_msk_ready = 0;
static void actq_tab_ensure_x86(void) {
    if (actq_msk_ready) return;
    uint8_t raw[4][4][16];
    for (int q = 0; q < 4; q++)
        for (int ri = 0; ri < 4; ri++)
            for (int L = 0; L < 16; L++)
                raw[q][ri][L] = ((L >> 2) == q) ? (uint8_t)(ri * 4 + (L & 3)) : 0x80;
    for (int q = 0; q < 4; q++)
        for (int ri = 0; ri < 4; ri++)
            actq_msk[q][ri] = _mm_loadu_si128((const __m128i *)raw[q][ri]);
    for (int p = 0; p < 2; p++) {
        int r0 = p * 2, r1 = p * 2 + 1;
        actq_msk2[p][0] = _mm256_set_m128i(actq_msk[1][r1], actq_msk[0][r0]);
        actq_msk2[p][1] = _mm256_set_m128i(actq_msk[0][r1], actq_msk[1][r0]);
        actq_msk2[p][2] = _mm256_set_m128i(actq_msk[3][r1], actq_msk[2][r0]);
        actq_msk2[p][3] = _mm256_set_m128i(actq_msk[2][r1], actq_msk[3][r0]);
    }
    actq_msk_ready = 1;
}

static inline int actq_sum_s8_32(const int8_t *a) {
    int s = 0;
    for (int i = 0; i < 32; i++) s += a[i];
    return s;
}

/* 单块（32 列，gate+up 双矩阵）：4 行 int dot'（未减 8·Σa）。行 ri 的 legacy
 * 16B 由 4 chunk 各一次 pshufb 收集后 OR（与 ARM vqtbl1q 同布局语义）。 */
static void actq_block_gu_x86(const uint8_t *gb, const uint8_t *ub,
                              const int8_t *xq, int32_t pgs[4], int32_t pus[4]) {
    const uint8_t *gqs = gb + 8, *uqs = ub + 8;
    __m128i g0 = _mm_loadu_si128((const __m128i *)(gqs + 0));
    __m128i g1 = _mm_loadu_si128((const __m128i *)(gqs + 16));
    __m128i g2 = _mm_loadu_si128((const __m128i *)(gqs + 32));
    __m128i g3 = _mm_loadu_si128((const __m128i *)(gqs + 48));
    __m128i u0 = _mm_loadu_si128((const __m128i *)(uqs + 0));
    __m128i u1 = _mm_loadu_si128((const __m128i *)(uqs + 16));
    __m128i u2 = _mm_loadu_si128((const __m128i *)(uqs + 32));
    __m128i u3 = _mm_loadu_si128((const __m128i *)(uqs + 48));
    __m128i xl = _mm_loadu_si128((const __m128i *)xq);
    __m128i xh = _mm_loadu_si128((const __m128i *)(xq + 16));
    const __m128i m15   = _mm_set1_epi8(15);
    const __m128i m0F0F = _mm_set1_epi16(0x0F0F);
    const __m128i x8    = _mm_set1_epi8(8);
    const __m128i ones  = _mm_set1_epi16(1);
    for (int ri = 0; ri < 4; ri++) {
        __m128i Rg = _mm_or_si128(_mm_or_si128(
            _mm_shuffle_epi8(g0, actq_msk[0][ri]),
            _mm_shuffle_epi8(g1, actq_msk[1][ri])),
            _mm_or_si128(_mm_shuffle_epi8(g2, actq_msk[2][ri]),
                         _mm_shuffle_epi8(g3, actq_msk[3][ri])));
        __m128i Ru = _mm_or_si128(_mm_or_si128(
            _mm_shuffle_epi8(u0, actq_msk[0][ri]),
            _mm_shuffle_epi8(u1, actq_msk[1][ri])),
            _mm_or_si128(_mm_shuffle_epi8(u2, actq_msk[2][ri]),
                         _mm_shuffle_epi8(u3, actq_msk[3][ri])));
        __m128i gl = _mm_xor_si128(_mm_and_si128(Rg, m15), x8);
        __m128i gh = _mm_xor_si128(_mm_and_si128(_mm_srli_epi16(Rg, 4), m0F0F), x8);
        __m128i ul = _mm_xor_si128(_mm_and_si128(Ru, m15), x8);
        __m128i uh = _mm_xor_si128(_mm_and_si128(_mm_srli_epi16(Ru, 4), m0F0F), x8);
        __m128i gl16 = _mm_maddubs_epi16(gl, xl);
        __m128i gh16 = _mm_maddubs_epi16(gh, xh);
        __m128i ul16 = _mm_maddubs_epi16(ul, xl);
        __m128i uh16 = _mm_maddubs_epi16(uh, xh);
        __m128i gi32 = _mm_madd_epi16(_mm_add_epi16(gl16, gh16), ones);
        __m128i ui32 = _mm_madd_epi16(_mm_add_epi16(ul16, uh16), ones);
        gi32 = _mm_add_epi32(gi32, _mm_shuffle_epi32(gi32, 0x4E));
        ui32 = _mm_add_epi32(ui32, _mm_shuffle_epi32(ui32, 0x4E));
        gi32 = _mm_add_epi32(gi32, _mm_shuffle_epi32(gi32, 0xB1));
        ui32 = _mm_add_epi32(ui32, _mm_shuffle_epi32(ui32, 0xB1));
        pgs[ri] = _mm_cvtsi128_si32(gi32);
        pus[ri] = _mm_cvtsi128_si32(ui32);
    }
}

/* gate/up：每 4 行组跨 nbG 个 tile（沿 d 列）。gv/uv[4]。近似轨。 */
static void moe_q4_gu_group_actq_x86(const uint8_t *gb, const uint8_t *ub, int nbG,
                                     const int8_t *xq, const float *xd,
                                     float *gv, float *uv) {
    float gacc[4] = {0.0f, 0.0f, 0.0f, 0.0f}, uacc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int cb = 0; cb < nbG; cb++) {
        int32_t pgs[4], pus[4];
        actq_block_gu_x86(gb + (size_t)cb * 72, ub + (size_t)cb * 72,
                          xq + (size_t)cb * 32, pgs, pus);
        uint16_t hg[4], hu[4];
        memcpy(hg, gb + (size_t)cb * 72, 8);
        memcpy(hu, ub + (size_t)cb * 72, 8);
        int corr = 8 * actq_sum_s8_32(xq + (size_t)cb * 32);
        for (int ri = 0; ri < 4; ri++) {
            float scg = vq_f16(hg[ri]), scu = vq_f16(hu[ri]), xdc = xd[cb];
            gacc[ri] += (scg * xdc) * (float)(pgs[ri] - corr);
            uacc[ri] += (scu * xdc) * (float)(pus[ri] - corr);
        }
    }
    for (int i = 0; i < 4; i++) { gv[i] = gacc[i]; uv[i] = uacc[i]; }
}

/* ---- M3（行阻塞 GEMM 项目）：actq gate/up 行末归约变体 ----
 * 与 moe_q4_gu_group_actq_x86 数学同值但浮点序新：块内核保留 8-lane 部分和
 * （每 lane=4 元素组 Σ nib·act，取自 gl16+gh16 的 8 s16 lanes）→ 256b f32 向量
 * ×scale 跨块累加；corr（8Σact）为标量项按块 ×scale 单独 f32 累加；组末
 * reduce8f 归约再减 corrAcc。省每 (行,块,矩阵) 的 4-lane 归约 + 标量化（actq
 * 128b reduce ~4 inst × 8 组合/块）。VLLM_ACTQ_BLK=1 启用（A/B，默认关）。 */
static int actq_blk_env(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("VLLM_ACTQ_BLK"); v = (e && e[0] == '1') ? 1 : 0; }
    return v;
}
static void moe_q4_gu_group_actq_blk_x86(const uint8_t *gb, const uint8_t *ub, int nbG,
                                         const int8_t *xq, const float *xd,
                                         float *gv, float *uv) {
    actq_tab_ensure_x86();
    const __m128i m15 = _mm_set1_epi8(15), m0F0F = _mm_set1_epi16(0x0F0F);
    const __m128i x8 = _mm_set1_epi8(8);
    __m256 gacc[4], uacc[4];
    float gcorr[4] = {0.0f, 0.0f, 0.0f, 0.0f}, ucorr[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int m = 0; m < 4; m++) { gacc[m] = _mm256_setzero_ps(); uacc[m] = _mm256_setzero_ps(); }
    for (int cb = 0; cb < nbG; cb++) {
        const uint8_t *gt = gb + (size_t)cb * 72, *ut = ub + (size_t)cb * 72;
        const uint8_t *gqs = gt + 8, *uqs = ut + 8;
        __m128i g0 = _mm_loadu_si128((const __m128i *)(gqs + 0));
        __m128i g1 = _mm_loadu_si128((const __m128i *)(gqs + 16));
        __m128i g2 = _mm_loadu_si128((const __m128i *)(gqs + 32));
        __m128i g3 = _mm_loadu_si128((const __m128i *)(gqs + 48));
        __m128i u0 = _mm_loadu_si128((const __m128i *)(uqs + 0));
        __m128i u1 = _mm_loadu_si128((const __m128i *)(uqs + 16));
        __m128i u2 = _mm_loadu_si128((const __m128i *)(uqs + 32));
        __m128i u3 = _mm_loadu_si128((const __m128i *)(uqs + 48));
        const int8_t *xs = xq + (size_t)cb * 32;
        __m128i xl = _mm_loadu_si128((const __m128i *)xs);
        __m128i xh = _mm_loadu_si128((const __m128i *)(xs + 16));
        int corr = 8 * actq_sum_s8_32(xs);
        uint16_t hg[4], hu[4];
        memcpy(hg, gt, 8); memcpy(hu, ut, 8);
        for (int ri = 0; ri < 4; ri++) {
            __m128i Rg = _mm_or_si128(_mm_or_si128(
                _mm_shuffle_epi8(g0, actq_msk[0][ri]),
                _mm_shuffle_epi8(g1, actq_msk[1][ri])),
                _mm_or_si128(_mm_shuffle_epi8(g2, actq_msk[2][ri]),
                             _mm_shuffle_epi8(g3, actq_msk[3][ri])));
            __m128i Ru = _mm_or_si128(_mm_or_si128(
                _mm_shuffle_epi8(u0, actq_msk[0][ri]),
                _mm_shuffle_epi8(u1, actq_msk[1][ri])),
                _mm_or_si128(_mm_shuffle_epi8(u2, actq_msk[2][ri]),
                             _mm_shuffle_epi8(u3, actq_msk[3][ri])));
            __m128i gl = _mm_xor_si128(_mm_and_si128(Rg, m15), x8);
            __m128i gh = _mm_xor_si128(_mm_and_si128(_mm_srli_epi16(Rg, 4), m0F0F), x8);
            __m128i ul = _mm_xor_si128(_mm_and_si128(Ru, m15), x8);
            __m128i uh = _mm_xor_si128(_mm_and_si128(_mm_srli_epi16(Ru, 4), m0F0F), x8);
            __m128i gs16 = _mm_add_epi16(_mm_maddubs_epi16(gl, xl), _mm_maddubs_epi16(gh, xh));
            __m128i us16 = _mm_add_epi16(_mm_maddubs_epi16(ul, xl), _mm_maddubs_epi16(uh, xh));
            float scg = vq_f16(hg[ri]) * xd[cb], scu = vq_f16(hu[ri]) * xd[cb];
            __m256 fg = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(gs16)),
                                      _mm256_set1_ps(scg));
            __m256 fu = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(us16)),
                                      _mm256_set1_ps(scu));
            gacc[ri] = _mm256_add_ps(gacc[ri], fg);
            uacc[ri] = _mm256_add_ps(uacc[ri], fu);
            gcorr[ri] += scg * (float)corr;
            ucorr[ri] += scu * (float)corr;
        }
    }
    for (int ri = 0; ri < 4; ri++) {
        gv[ri] = q8g_reduce8f(gacc[ri]) - gcorr[ri];
        uv[ri] = q8g_reduce8f(uacc[ri]) - ucorr[ri];
    }
}

/* down：每 4 输出行组跨 cbe 个 tile（专家 down 列带）。y += p*acc。近似轨。
 * 2026-09-10（down 执行结构重构 M1）：补 tile 级预取（镜像 f32 轨 9057）+
 * unroll 提示。注意勿加 optimize("O3")——gcc16 下 ffp-contract 选项串被拒
 * （bad option 警告），O3 生效会引入 FMA 收缩致锚漂。FP 序零改动。 */
static void moe_q4_down_group_actq_x86(const uint8_t *db, int cbe,
                                       const int8_t *avq, const float *avd,
                                       float p, float *y) {
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    _Pragma("GCC unroll 8")
    for (int cbi = 0; cbi < cbe; cbi++) {
        const uint8_t *tile = db + (size_t)cbi * 72;
        ST_PREFETCH(db + (size_t)(cbi + 4) * 72);
        const uint8_t *qs_ = tile + 8;
        __m128i d0 = _mm_loadu_si128((const __m128i *)(qs_ + 0));
        __m128i d1 = _mm_loadu_si128((const __m128i *)(qs_ + 16));
        __m128i d2 = _mm_loadu_si128((const __m128i *)(qs_ + 32));
        __m128i d3 = _mm_loadu_si128((const __m128i *)(qs_ + 48));
        __m128i xl = _mm_loadu_si128((const __m128i *)(avq + (size_t)cbi * 32));
        __m128i xh = _mm_loadu_si128((const __m128i *)(avq + (size_t)cbi * 32 + 16));
        const __m128i m15 = _mm_set1_epi8(15), m0F0F = _mm_set1_epi16(0x0F0F);
        const __m128i x8 = _mm_set1_epi8(8), ones = _mm_set1_epi16(1);
        uint16_t hd[4];
        memcpy(hd, tile, 8);
        int corr = 8 * actq_sum_s8_32(avq + (size_t)cbi * 32);
        int32_t pas[4];
        for (int ri = 0; ri < 4; ri++) {
            __m128i Rd = _mm_or_si128(_mm_or_si128(
                _mm_shuffle_epi8(d0, actq_msk[0][ri]),
                _mm_shuffle_epi8(d1, actq_msk[1][ri])),
                _mm_or_si128(_mm_shuffle_epi8(d2, actq_msk[2][ri]),
                             _mm_shuffle_epi8(d3, actq_msk[3][ri])));
            __m128i l = _mm_xor_si128(_mm_and_si128(Rd, m15), x8);
            __m128i h = _mm_xor_si128(_mm_and_si128(_mm_srli_epi16(Rd, 4), m0F0F), x8);
            __m128i d16 = _mm_add_epi16(_mm_maddubs_epi16(l, xl),
                                        _mm_maddubs_epi16(h, xh));
            __m128i i32 = _mm_madd_epi16(d16, ones);
            i32 = _mm_add_epi32(i32, _mm_shuffle_epi32(i32, 0x4E));
            i32 = _mm_add_epi32(i32, _mm_shuffle_epi32(i32, 0xB1));
            pas[ri] = _mm_cvtsi128_si32(i32);
        }
        for (int ri = 0; ri < 4; ri++)
            acc[ri] += (vq_f16(hd[ri]) * avd[cbi]) * (float)(pas[ri] - corr);
    }
    for (int ri = 0; ri < 4; ri++) y[ri] += p * acc[ri];
}

/* ============================================================
 * s16 激活整数点积轨（VLLM_ACTQ16=1，opt-in，2026-09-13）
 * ============================================================
 * 动机：ACTQ（int8 激活）在 x86 实测 decode ~1.44×，但激活量化误差约「输出量级
 * 的 0.25–1.0%」，长文本 t≈24 起漂移。本轨把激活换成 **s16**（max-abs/32767）：
 *   · 行收集掩码（actq_msk）与 nibble 布局**完全复用**；
 *   · 权重取**有符号** nibble（(nib^8)-8），直接与 s16 激活做
 *     vpmaddwd(w_s16, a_s16) → i32 **精确**点积 —— 因此**不需要** ACTQ 那套
 *     「u8 偏移表示 + 8·Σact 标量校正」；本轨顺带省掉每块 32 次标量加；
 *   · 逐块 acc += (scale_ri·xdc)·dot，与 ACTQ / 精确轨同形。
 * 量化步长 ÷258 → 误差降到 ~1e-5（≈f32 累加序噪声），TOKIDS 锚可保持。
 * 默认**开**（2026-09-13 起；`VLLM_ACTQ16=0` 关闭，EP 场景必须关）。
 * 与 VLLM_ACTQ 互斥，本开关优先。 */
static int actq16_env(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("VLLM_ACTQ16"); v = (e && e[0] == '0') ? 0 : 1; }
    return v;
}
static void quantize_row_s16_act(const float *__restrict x, int16_t *__restrict q,
                                 float *__restrict d, int cols) {
    int nb = cols >> 5;
    for (int b = 0; b < nb; b++) {
        float am = 0.0f;
        for (int j = 0; j < 32; j++) { float a = fabsf(x[b * 32 + j]); if (a > am) am = a; }
        float s = am / 32767.0f;
        d[b] = s;
        if (s <= 1e-30f) { memset(q + b * 32, 0, 32 * sizeof(int16_t)); continue; }
        for (int j = 0; j < 32; j++) {
            float v = x[b * 32 + j] / s;
            int qi = (int)lrintf(v);
            if (qi > 32767) qi = 32767; if (qi < -32767) qi = -32767;
            q[b * 32 + j] = (int16_t)qi;
        }
    }
}
/* 单块（32 列）gate+up 双矩阵的 4 行整数点积（无偏移校正）。 */
static inline void actq16_block_gu_x86(const uint8_t *gb, const uint8_t *ub,
                                       const int16_t *xq, int32_t pgs[4], int32_t pus[4]) {
    const uint8_t *gqs = gb + 8, *uqs = ub + 8;
    __m128i g0 = _mm_loadu_si128((const __m128i *)(gqs + 0));
    __m128i g1 = _mm_loadu_si128((const __m128i *)(gqs + 16));
    __m128i g2 = _mm_loadu_si128((const __m128i *)(gqs + 32));
    __m128i g3 = _mm_loadu_si128((const __m128i *)(gqs + 48));
    __m128i u0 = _mm_loadu_si128((const __m128i *)(uqs + 0));
    __m128i u1 = _mm_loadu_si128((const __m128i *)(uqs + 16));
    __m128i u2 = _mm_loadu_si128((const __m128i *)(uqs + 32));
    __m128i u3 = _mm_loadu_si128((const __m128i *)(uqs + 48));
    __m256i xl = _mm256_loadu_si256((const __m256i *)xq);          /* act 0..15 */
    __m256i xh = _mm256_loadu_si256((const __m256i *)(xq + 16));   /* act 16..31 */
    const __m128i m15   = _mm_set1_epi8(15);
    const __m128i m0F0F = _mm_set1_epi16(0x0F0F);
    const __m128i x8    = _mm_set1_epi8(8);
    for (int ri = 0; ri < 4; ri++) {
        __m128i Rg = _mm_or_si128(_mm_or_si128(
            _mm_shuffle_epi8(g0, actq_msk[0][ri]),
            _mm_shuffle_epi8(g1, actq_msk[1][ri])),
            _mm_or_si128(_mm_shuffle_epi8(g2, actq_msk[2][ri]),
                         _mm_shuffle_epi8(g3, actq_msk[3][ri])));
        __m128i Ru = _mm_or_si128(_mm_or_si128(
            _mm_shuffle_epi8(u0, actq_msk[0][ri]),
            _mm_shuffle_epi8(u1, actq_msk[1][ri])),
            _mm_or_si128(_mm_shuffle_epi8(u2, actq_msk[2][ri]),
                         _mm_shuffle_epi8(u3, actq_msk[3][ri])));
        __m128i gl = _mm_sub_epi8(_mm_xor_si128(_mm_and_si128(Rg, m15), x8), x8);
        __m128i gh = _mm_sub_epi8(
            _mm_xor_si128(_mm_and_si128(_mm_srli_epi16(Rg, 4), m0F0F), x8), x8);
        __m128i ul = _mm_sub_epi8(_mm_xor_si128(_mm_and_si128(Ru, m15), x8), x8);
        __m128i uh = _mm_sub_epi8(
            _mm_xor_si128(_mm_and_si128(_mm_srli_epi16(Ru, 4), m0F0F), x8), x8);
        __m256i gi = _mm256_add_epi32(
            _mm256_madd_epi16(_mm256_cvtepi8_epi16(gl), xl),
            _mm256_madd_epi16(_mm256_cvtepi8_epi16(gh), xh));
        __m256i ui = _mm256_add_epi32(
            _mm256_madd_epi16(_mm256_cvtepi8_epi16(ul), xl),
            _mm256_madd_epi16(_mm256_cvtepi8_epi16(uh), xh));
        pgs[ri] = q8g_reduce8(gi);
        pus[ri] = q8g_reduce8(ui);
    }
}
static void moe_q4_gu_group_actq16_x86(const uint8_t *gb, const uint8_t *ub, int nbG,
                                       const int16_t *xq, const float *xd,
                                       float *gv, float *uv) {
    float gacc[4] = {0.0f, 0.0f, 0.0f, 0.0f}, uacc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int cb = 0; cb < nbG; cb++) {
        int32_t pgs[4], pus[4];
        actq16_block_gu_x86(gb + (size_t)cb * 72, ub + (size_t)cb * 72,
                            xq + (size_t)cb * 32, pgs, pus);
        uint16_t hg[4], hu[4];
        memcpy(hg, gb + (size_t)cb * 72, 8);
        memcpy(hu, ub + (size_t)cb * 72, 8);
        for (int ri = 0; ri < 4; ri++) {
            float scg = vq_f16(hg[ri]), scu = vq_f16(hu[ri]), xdc = xd[cb];
            gacc[ri] += (scg * xdc) * (float)pgs[ri];
            uacc[ri] += (scu * xdc) * (float)pus[ri];
        }
    }
    for (int i = 0; i < 4; i++) { gv[i] = gacc[i]; uv[i] = uacc[i]; }
}
static inline void actq16_block_dn_x86(const uint8_t *db, const int16_t *avq,
                                       int32_t pas[4]) {
    const uint8_t *qs_ = db + 8;
    __m128i d0 = _mm_loadu_si128((const __m128i *)(qs_ + 0));
    __m128i d1 = _mm_loadu_si128((const __m128i *)(qs_ + 16));
    __m128i d2 = _mm_loadu_si128((const __m128i *)(qs_ + 32));
    __m128i d3 = _mm_loadu_si128((const __m128i *)(qs_ + 48));
    __m256i xl = _mm256_loadu_si256((const __m256i *)avq);
    __m256i xh = _mm256_loadu_si256((const __m256i *)(avq + 16));
    const __m128i m15   = _mm_set1_epi8(15);
    const __m128i m0F0F = _mm_set1_epi16(0x0F0F);
    const __m128i x8    = _mm_set1_epi8(8);
    for (int ri = 0; ri < 4; ri++) {
        __m128i Rd = _mm_or_si128(_mm_or_si128(
            _mm_shuffle_epi8(d0, actq_msk[0][ri]),
            _mm_shuffle_epi8(d1, actq_msk[1][ri])),
            _mm_or_si128(_mm_shuffle_epi8(d2, actq_msk[2][ri]),
                         _mm_shuffle_epi8(d3, actq_msk[3][ri])));
        __m128i l = _mm_sub_epi8(_mm_xor_si128(_mm_and_si128(Rd, m15), x8), x8);
        __m128i h = _mm_sub_epi8(
            _mm_xor_si128(_mm_and_si128(_mm_srli_epi16(Rd, 4), m0F0F), x8), x8);
        __m256i i = _mm256_add_epi32(
            _mm256_madd_epi16(_mm256_cvtepi8_epi16(l), xl),
            _mm256_madd_epi16(_mm256_cvtepi8_epi16(h), xh));
        pas[ri] = q8g_reduce8(i);
    }
}
/* down：每 4 输出行组跨 cbe 个 tile（专家 down 列带）。y += p*acc。s16 轨。 */
static void moe_q4_down_group_actq16_x86(const uint8_t *db, int cbe,
                                         const int16_t *avq, const float *avd,
                                         float p, float *y) {
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int cbi = 0; cbi < cbe; cbi++) {
        const uint8_t *tile = db + (size_t)cbi * 72;
        uint16_t hd[4];
        int32_t pas[4];
        memcpy(hd, tile, 8);
        ST_PREFETCH(db + (size_t)(cbi + 4) * 72);
        actq16_block_dn_x86(tile, avq + (size_t)cbi * 32, pas);
        for (int ri = 0; ri < 4; ri++)
            acc[ri] += (vq_f16(hd[ri]) * avd[cbi]) * (float)pas[ri];
    }
    for (int ri = 0; ri < 4; ri++) y[ri] += p * acc[ri];
}
#endif /* __AVX2__ && ST_ARCH_X86 */

#if !(defined(__AVX2__) && ST_ARCH_X86)
/* §9.21/9.22 安全桩：ARM 侧暂无 s16 内核，但共享的 st_moe_ffn_sparse_q4 会引用这两个
 * 符号（x86 的实现在上面 x86-only 块内）。此处给出等价桩，保证 ARM 构建不破：
 * actq16 恒 0 → 走 ARM 原有精确 / ACTQ 轨；quantize 桩不会被真正调用。 */
static int actq16_env(void) { return 0; }
static void quantize_row_s16_act(const float *__restrict x, int16_t *__restrict q,
                                 float *__restrict d, int cols) {
    (void)x; (void)q; (void)d; (void)cols;
}
#endif

#if defined(__AVX2__) && ST_ARCH_X86
/* M5（down 执行结构重构，V1 重做 2026-09-10）：down 行组并行。
 * 512 个 4 行组按线程静态分块（vllm_tp_parfor）；每行组仅一个线程写
 * y[rg*4..+3]，专家 j 外层循环序不变 → 每输出行的浮点累加序与串行
 * 逐位一致（A≡C）。行内点积序不变。 */
typedef struct {
    const uint8_t *D; float *y;
    int cb0, cbe, gdB, rgmax;
    const int8_t *avq; const float *avd; float p;
} m4_dn_actq_ctx;
static void m4_dn_actq_worker(void *ctx_, int rg) {
    m4_dn_actq_ctx *c = ctx_;
    const uint8_t *db = c->D + (size_t)rg * (size_t)c->gdB + (size_t)c->cb0 * 72;
    if (rg + 8 < c->rgmax) ST_PREFETCH(db + (size_t)8 * (size_t)c->gdB);
    moe_q4_down_group_actq_x86(db, c->cbe, c->avq, c->avd, c->p,
                               c->y + (size_t)rg * 4);
}
/* s16 轨 down worker（VLLM_ACTQ16=1）：与 m4_dn_actq_worker 同调度、同写序。 */
typedef struct {
    const uint8_t *D; float *y;
    int cb0, cbe, gdB, rgmax;
    const int16_t *avq; const float *avd; float p;
} m4_dn_actq16_ctx;
static void m4_dn_actq16_worker(void *ctx_, int rg) {
    m4_dn_actq16_ctx *c = ctx_;
    const uint8_t *db = c->D + (size_t)rg * (size_t)c->gdB + (size_t)c->cb0 * 72;
    if (rg + 8 < c->rgmax) ST_PREFETCH(db + (size_t)8 * (size_t)c->gdB);
    moe_q4_down_group_actq16_x86(db, c->cbe, c->avq, c->avd, c->p,
                                 c->y + (size_t)rg * 4);
}
typedef struct {
    const uint8_t *D; float *y; const float *av;
    int cb0, cbe, gdB, rgmax; float p;
} m4_dn_f32_ctx;
static void m4_dn_f32_worker(void *ctx_, int rg) {
    m4_dn_f32_ctx *c = ctx_;
    const uint8_t *db = c->D + (size_t)rg * (size_t)c->gdB + (size_t)c->cb0 * 72;
    moe_q4_down_group_x86(db, c->cbe, c->av, c->p, c->y + (size_t)rg * 4);
}
/* EP 贡献用：原始点积（不乘 p）→ acc_out[rg*4..+3]。p 由协调者显式 fmaf 施加。 */
typedef struct {
    const uint8_t *D; float *acc_out; const float *av;
    int cb0, cbe, gdB, rgmax;
} m4_dn_raw_ctx;
static void m4_dn_raw_worker(void *ctx_, int rg) {
    m4_dn_raw_ctx *c = ctx_;
    const uint8_t *db = c->D + (size_t)rg * (size_t)c->gdB + (size_t)c->cb0 * 72;
    moe_q4_down_group_raw_x86(db, c->cbe, c->av, c->acc_out + (size_t)rg * 4);
}
#endif /* __AVX2__ && ST_ARCH_X86 */

#if defined(__aarch64__) && defined(ST_HAVE_NEON) && !ST_ARCH_X86
/* q4 4x4-tile NEON f32 decode（ARM 对等内核，语义与上面 AVX2 区逐位一致）。
 * 目标：decode 热点 st_moe_ffn_sparse_q4 在 ARM 此前只有逐元素标量（vq4x4_w）；
 * 本内核一次处理一个 72B tile 的 4 行 × 32 列，逐元素链序与标量/x86 完全相同：
 *   wf = scale_ri*(nib-8)（int→f32 精确、1 次乘舍入）→ acc += wf*x[k]
 *   （mul+add 各 1 次舍入，不收缩）→ 与标量、x86 AVX2 变体**逐位一致**。
 * 字节拾取（2026-09-13 解包优化，数值与旧逐 k vqtbl 版逐位一致）：
 * tile = 4×f16 scale@[0..8) + 64B qs（4 chunk × 16B）。元素 k：chunk
 * q=(k&15)>>2、子列 s=k&3，低半=元素 4q+s、高半=4q+s+16（k>=16）；字节在 chunk
 * q 偏移 ri*4+s（行 ri 子列 s）。旧版每个 k 做一次 vqtbl 拾取 4 行；新版改为
 * **按 chunk 整体处理**：4 个 chunk 各载入一次并**预异或** 0x88（旧版每 k 异或），
 * 再用**一次 vqtbl1q**（idx {0,4,8,12, 1,5,9,13, 2,6,10,14, 3,7,11,15}）把 16 字节
 * 重组成「lane 4s+ri = 行 ri 子列 s」→ 一次加宽（u8→u16→s32）即得 s=0..3 四个
 * 4 行组，最后按 k=4q+s 升序逐个乘加。指令数较旧版降约 1/3（§9.14 判决
 * decode 为解包指令受限），而 (nib-8)×scale 与 ×x[k] 的两次舍入序不变。
 * 开关同 x86：VLLM_MOE_Q4SIMD=0 关闭（moe_q4_simd_ok 见上）。 */
static inline void q4x4_tile32_neon(const uint8_t *tile, const float *x,
                                    float32x4_t *acc) {
    /* 重组索引：输出 lane 4s+ri ← chunk 内偏移 ri*4+s 的字节 */
    static const uint8_t GRP[16] = {
        0, 4, 8, 12,   1, 5, 9, 13,   2, 6, 10, 14,   3, 7, 11, 15,
    };
    const uint8x16_t grp   = vld1q_u8(GRP);
    const uint8x16_t xor88 = vdupq_n_u8(0x88);
    const uint8x16_t m15   = vdupq_n_u8(15);
    const int32x4_t  k8    = vdupq_n_s32(8);
    const uint8_t *qs = tile + 8;
    /* 预异或：整块 4 个 16B chunk 各一次（旧版每个 k 一次） */
    const uint8x16_t C0 = veorq_u8(vld1q_u8(qs +  0), xor88);
    const uint8x16_t C1 = veorq_u8(vld1q_u8(qs + 16), xor88);
    const uint8x16_t C2 = veorq_u8(vld1q_u8(qs + 32), xor88);
    const uint8x16_t C3 = veorq_u8(vld1q_u8(qs + 48), xor88);
    float sc[4];
    sc[0] = vq_f16((uint16_t)(tile[0] | ((uint16_t)tile[1] << 8)));
    sc[1] = vq_f16((uint16_t)(tile[2] | ((uint16_t)tile[3] << 8)));
    sc[2] = vq_f16((uint16_t)(tile[4] | ((uint16_t)tile[5] << 8)));
    sc[3] = vq_f16((uint16_t)(tile[6] | ((uint16_t)tile[7] << 8)));
    const float32x4_t scales = vld1q_f32(sc);
    float32x4_t ac = *acc;

    /* 一个 chunk → 4 个 k（k = KB+0..3）。HALF=0 取低半字节、HALF=1 取高半字节。
     * 累加严格按 k 升序：调用顺序 0..15（低半）后 16..31（高半）。 */
#define Q4X4_ACC_CHUNK(CQ, KB, HALF)                                          \
    do {                                                                      \
        uint8x16_t b_ = vqtbl1q_u8((CQ), grp);      /* lane 4s+ri = 行 ri 子列 s */ \
        uint8x16_t n_ = (HALF) ? vandq_u8(vshrq_n_u8(b_, 4), m15)             \
                               : vandq_u8(b_, m15);                           \
        uint16x8_t u0_ = vmovl_u8(vget_low_u8(n_));   /* s=0,1（各 4 行） */   \
        uint16x8_t u1_ = vmovl_u8(vget_high_u8(n_));  /* s=2,3 */             \
        int32x4_t s0_ = vmovl_s16(vget_low_s16(vreinterpretq_s16_u16(u0_)));  \
        int32x4_t s1_ = vmovl_s16(vget_high_s16(vreinterpretq_s16_u16(u0_))); \
        int32x4_t s2_ = vmovl_s16(vget_low_s16(vreinterpretq_s16_u16(u1_)));  \
        int32x4_t s3_ = vmovl_s16(vget_high_s16(vreinterpretq_s16_u16(u1_))); \
        ac = vaddq_f32(ac, vmulq_f32(vmulq_f32(                              \
                 vcvtq_f32_s32(vsubq_s32(s0_, k8)), scales),                  \
                 vdupq_n_f32(x[(KB) + 0])));                                  \
        ac = vaddq_f32(ac, vmulq_f32(vmulq_f32(                              \
                 vcvtq_f32_s32(vsubq_s32(s1_, k8)), scales),                  \
                 vdupq_n_f32(x[(KB) + 1])));                                  \
        ac = vaddq_f32(ac, vmulq_f32(vmulq_f32(                              \
                 vcvtq_f32_s32(vsubq_s32(s2_, k8)), scales),                  \
                 vdupq_n_f32(x[(KB) + 2])));                                  \
        ac = vaddq_f32(ac, vmulq_f32(vmulq_f32(                              \
                 vcvtq_f32_s32(vsubq_s32(s3_, k8)), scales),                  \
                 vdupq_n_f32(x[(KB) + 3])));                                  \
    } while (0)

    Q4X4_ACC_CHUNK(C0,  0, 0);
    Q4X4_ACC_CHUNK(C1,  4, 0);
    Q4X4_ACC_CHUNK(C2,  8, 0);
    Q4X4_ACC_CHUNK(C3, 12, 0);
    Q4X4_ACC_CHUNK(C0, 16, 1);
    Q4X4_ACC_CHUNK(C1, 20, 1);
    Q4X4_ACC_CHUNK(C2, 24, 1);
    Q4X4_ACC_CHUNK(C3, 28, 1);
#undef Q4X4_ACC_CHUNK

    *acc = ac;
}

/* 精确轨 wf 预计算：解包一个 72B tile（4 行 × 32 列 Q4）→ 32 个 k 的 wf
 * （4 行 f32，已 ×scale）。数值与 q4x4_tile32_neon 逐 k 的 wf 逐位一致
 * （同 vqtbl 解包 + vcvt + vmul scale），供 GroupGEMM 跨 token 复用。
 * 2026-09-13：与 q4x4_tile32_neon 同步改为「整 chunk 预异或 + 单次 vqtbl
 * 重组 16 通道 + chunk 内批量加宽」，把每 tile 的 vqtbl 从 32 次降到 8 次
 * （A76 permute 单元是 prefill 侧同样存在的瓶颈）。数值不变。 */
static inline void q4x4_tile32_wf_neon(const uint8_t *tile, float32x4_t wf[32]) {
    /* 重组索引：输出 lane 4s+ri ← chunk 内偏移 ri*4+s 的字节 */
    static const uint8_t GRP[16] = {
        0, 4, 8, 12,   1, 5, 9, 13,   2, 6, 10, 14,   3, 7, 11, 15,
    };
    const uint8x16_t grp   = vld1q_u8(GRP);
    const uint8x16_t m15   = vdupq_n_u8(15);
    const uint8x16_t xor88 = vdupq_n_u8(0x88);
    const int32x4_t  k8    = vdupq_n_s32(8);
    const uint8_t *qs = tile + 8;
    const uint8x16_t C0 = veorq_u8(vld1q_u8(qs +  0), xor88);
    const uint8x16_t C1 = veorq_u8(vld1q_u8(qs + 16), xor88);
    const uint8x16_t C2 = veorq_u8(vld1q_u8(qs + 32), xor88);
    const uint8x16_t C3 = veorq_u8(vld1q_u8(qs + 48), xor88);
    float sc[4];
    sc[0] = vq_f16((uint16_t)(tile[0] | ((uint16_t)tile[1] << 8)));
    sc[1] = vq_f16((uint16_t)(tile[2] | ((uint16_t)tile[3] << 8)));
    sc[2] = vq_f16((uint16_t)(tile[4] | ((uint16_t)tile[5] << 8)));
    sc[3] = vq_f16((uint16_t)(tile[6] | ((uint16_t)tile[7] << 8)));
    const float32x4_t scales = vld1q_f32(sc);

    /* 一个 chunk → wf[KB+0..3]。HALF=0 取低半字节（k=0..15）、HALF=1 取高半字节
     * （k=16..31），与 q4x4_tile32_neon 的 k 升序一一对应。 */
#define Q4X4_WF_CHUNK(CQ, KB, HALF)                                           \
    do {                                                                      \
        uint8x16_t b_ = vqtbl1q_u8((CQ), grp);      /* lane 4s+ri = 行 ri 子列 s */ \
        uint8x16_t n_ = (HALF) ? vandq_u8(vshrq_n_u8(b_, 4), m15)             \
                               : vandq_u8(b_, m15);                           \
        uint16x8_t u0_ = vmovl_u8(vget_low_u8(n_));   /* s=0,1（各 4 行） */   \
        uint16x8_t u1_ = vmovl_u8(vget_high_u8(n_));  /* s=2,3 */             \
        int32x4_t s0_ = vmovl_s16(vget_low_s16(vreinterpretq_s16_u16(u0_)));  \
        int32x4_t s1_ = vmovl_s16(vget_high_s16(vreinterpretq_s16_u16(u0_))); \
        int32x4_t s2_ = vmovl_s16(vget_low_s16(vreinterpretq_s16_u16(u1_)));  \
        int32x4_t s3_ = vmovl_s16(vget_high_s16(vreinterpretq_s16_u16(u1_))); \
        wf[(KB) + 0] = vmulq_f32(vcvtq_f32_s32(vsubq_s32(s0_, k8)), scales); \
        wf[(KB) + 1] = vmulq_f32(vcvtq_f32_s32(vsubq_s32(s1_, k8)), scales); \
        wf[(KB) + 2] = vmulq_f32(vcvtq_f32_s32(vsubq_s32(s2_, k8)), scales); \
        wf[(KB) + 3] = vmulq_f32(vcvtq_f32_s32(vsubq_s32(s3_, k8)), scales); \
    } while (0)

    Q4X4_WF_CHUNK(C0,  0, 0);
    Q4X4_WF_CHUNK(C1,  4, 0);
    Q4X4_WF_CHUNK(C2,  8, 0);
    Q4X4_WF_CHUNK(C3, 12, 0);
    Q4X4_WF_CHUNK(C0, 16, 1);
    Q4X4_WF_CHUNK(C1, 20, 1);
    Q4X4_WF_CHUNK(C2, 24, 1);
    Q4X4_WF_CHUNK(C3, 28, 1);
#undef Q4X4_WF_CHUNK
}

/* 每 4 行组跨 nbG 个 tile 累加 gate/up（NEON，语义同 moe_q4_gu_group_x86）。 */
static void moe_q4_gu_group_neon(const uint8_t *gb, const uint8_t *ub, int nbG,
                                 const float *x, float gv[4], float uv[4]) {
    float32x4_t ga = vdupq_n_f32(0.0f), ua = vdupq_n_f32(0.0f);
    for (int cb = 0; cb < nbG; cb++) {
        q4x4_tile32_neon(gb + (size_t)cb * 72, x + (size_t)cb * 32, &ga);
        q4x4_tile32_neon(ub + (size_t)cb * 72, x + (size_t)cb * 32, &ua);
    }
    vst1q_f32(gv, ga);
    vst1q_f32(uv, ua);
}

/* 每 4 输出行组跨 cbe 个 tile（专家 down 列带）：y[rg*4..+3] += p*acc（NEON）。 */
static void moe_q4_down_group_neon(const uint8_t *db, int cbe,
                                   const float *av, float p, float *y) {
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (int cbi = 0; cbi < cbe; cbi++)
        q4x4_tile32_neon(db + (size_t)cbi * 72, av + (size_t)cbi * 32, &acc);
    float32x4_t pv = vdupq_n_f32(p);
    float32x4_t yo = vld1q_f32(y);
    vst1q_f32(y, vaddq_f32(yo, vmulq_f32(pv, acc)));
}
/* EP 专用（ARM）：只算原始点积（**不乘 p**）。跨机协议要求两侧 contrib 语义一致
 * ——p 一律由协调者的 `moe_ep_reduce` 施加（§9.24：跨机 EP 若一侧乘 p、一侧不乘，
 * 协调者会对已乘 p 的贡献再乘一次）。 */
static void moe_q4_down_group_raw_neon(const uint8_t *db, int cbe,
                                       const float *av, float *acc_out) {
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (int cbi = 0; cbi < cbe; cbi++)
        q4x4_tile32_neon(db + (size_t)cbi * 72, av + (size_t)cbi * 32, &acc);
    vst1q_f32(acc_out, acc);
}
#endif /* __aarch64__ && ST_HAVE_NEON && !ST_ARCH_X86 */

/* ====================================================================
 * VLLM_ACTQ=1：q4 MoE 稀疏专家 激活量化 int8-dotprod 双轨（llama mmq 移植，
 * 语义参照 ggml ggml_vec_dot_q4_0_q8_0 / q8_0 激活量化；近似路径，默认关）
 *
 * 现状（精确轨，A≡C 位级）：moe_q4_gu_group_neon / moe_q4_down_group_neon
 * 对 f32 激活逐元素 s8×f32（每 (ri,k) 拆字节 + vq_f16 + mul/add 两次舍入）。
 * 近似轨（本区）：激活按 32 块对称量化为 q8_0（scale=max_abs/127、
 * round-nearest、clamp ±127，与 quantize_row_q8_0_act 完全同语义），
 * 权重 4bit 值 (nib-8) 与激活 int8 用 vdotq_s32 整型点积（i8x16_dot_s32），
 * 每 (行,块) 一次 f32 乘加：acc_ri += sc_ri[cb]*xd[cb]*dot_i32。
 *
 * 数值边界（诚实）：激活量化误差 ≤ ~0.4% + 块级缩放重排 → 与标量/x86
 * 精确轨**不逐位一致**，只保证锚点级 token/文本一致（实测 ~23 token 后贪婪
 * 解码漂移）；默认关闭，VLLM_ACTQ=1 显式开（aarch64 dotprod）。量化激活
 * x_ffn 每层每 token 一次（d=2048→64 块），全部专家/gate/up/down 行共享；
 * down 的激活 av（每专家 ef=768）每专家量化一次。
 * ==================================================================== */
static int moe_q4_actq_ok(void) {
#if defined(__aarch64__) && defined(ST_HAVE_NEON) && !ST_ARCH_X86 && ST_NEON_DOTPROD
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("VLLM_ACTQ");
        v = (e && e[0] == '1') ? 1 : 0;   /* 默认关；VLLM_ACTQ=1 显式开（近似轨，长文本漂移） */
    }
    return v;
#elif defined(__AVX2__) && ST_ARCH_X86
    /* x86 actq（AVX2 maddubs+nib 域）与 ARM vdot 同语义整数点积（M1/M2 门禁
     * 证实量化字节跨平台位级一致、整数校正精确）——平台内近似轨，默认关。 */
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("VLLM_ACTQ");
        v = (e && e[0] == '1') ? 1 : 0;
    }
    return v;
#else
    return 0;
#endif
}

#if defined(__aarch64__) && defined(ST_HAVE_NEON) && !ST_ARCH_X86 && ST_NEON_DOTPROD
/* 行 ri 的 legacy 字节 bb（bb=4q+s，q=chunk、s=子列）位于 chunk q 的
 * 偏移 ri*4+s。TAB[q][ri][L] = (L>>2)==q ? ri*4+(L&3) : 0x80 —— 对 4 个
 * chunk 各做一次 vqtbl1q 再 or，得到该行按 legacy bb 序的 16B 向量。 */
static uint8_t actq_tab[4][4][16];
static int actq_tab_ready = 0;
static void actq_tab_ensure(void) {
    if (actq_tab_ready) return;
    for (int q = 0; q < 4; q++)
        for (int ri = 0; ri < 4; ri++)
            for (int L = 0; L < 16; L++)
                actq_tab[q][ri][L] = ((L >> 2) == q) ? (uint8_t)(ri * 4 + (L & 3)) : 0x80;
    actq_tab_ready = 1;
}

/* R 为行 legacy 序 16B（含 0x88 掩码）；取低/高半字节还原真 nibble 后 -8，
 * 得 int8 权重 (nib-8) ∈ [-8,7]，与 vdotq_s32 激活侧直接配对。 */
static inline int8x16_t actq_w_from_r(uint8x16_t R, int ishi,
                                      uint8x16_t m15, uint8x16_t x8,
                                      int8x16_t s8) {
    uint8x16_t u = ishi ? vandq_u8(vshrq_n_u8(R, 4), m15) : vandq_u8(R, m15);
    u = veorq_u8(u, x8);                 /* 还原 0x88 翻转（真 nibble） */
    return vsubq_s8(vreinterpretq_s8_u8(u), s8);  /* (nib-8) */
}

/* 4 行 × 32 列 tile（gate 与 up 同列 cb）：点积部分。xv=xq+cb*32（int8）、
 * xd_cb=块 scale；acc 以 f32 累加（每 (行,块) 1 次乘加，llama 同款序）。 */
static void q4x4_gup_actq_neon(const uint8_t *gb, const uint8_t *ub,
                               const int8_t *xv, float xd_cb,
                               float32x4_t *ga, float32x4_t *ua,
                               const uint8x16_t tab[4][4],
                               uint8x16_t m15, uint8x16_t x8, int8x16_t s8) {
    const uint8_t *gqs = gb + 8, *uqs = ub + 8;
    uint8x16_t g0 = vld1q_u8(gqs),      g1 = vld1q_u8(gqs + 16);
    uint8x16_t g2 = vld1q_u8(gqs + 32), g3 = vld1q_u8(gqs + 48);
    uint8x16_t u0 = vld1q_u8(uqs),      u1 = vld1q_u8(uqs + 16);
    uint8x16_t u2 = vld1q_u8(uqs + 32), u3 = vld1q_u8(uqs + 48);
    int8x16_t xl = vld1q_s8(xv), xh = vld1q_s8(xv + 16);
    float32x4_t scg, scu;
    {
        uint16_t h[8]; float sg[4], su[4];
        memcpy(h, gb, 8); memcpy(h + 4, ub, 8);
        for (int i = 0; i < 4; i++) { sg[i] = vq_f16(h[i]); su[i] = vq_f16(h[i + 4]); }
        scg = vld1q_f32(sg); scu = vld1q_f32(su);
    }
    float32x4_t xdv = vdupq_n_f32(xd_cb);
    /* 逐行收集 int32 点积到数组，最后 vld1q_s32 一次载入（vsetq_lane_s32 的
     * lane 参数必须是编译期常量，循环变量 ri 在 gcc11 下编译不过）。 */
    int32_t pgs[4], pus[4];
    for (int ri = 0; ri < 4; ri++) {
        uint8x16_t Rg = vorrq_u8(
            vorrq_u8(vorrq_u8(vqtbl1q_u8(g0, tab[0][ri]), vqtbl1q_u8(g1, tab[1][ri])),
                     vqtbl1q_u8(g2, tab[2][ri])), vqtbl1q_u8(g3, tab[3][ri]));
        uint8x16_t Ru = vorrq_u8(
            vorrq_u8(vorrq_u8(vqtbl1q_u8(u0, tab[0][ri]), vqtbl1q_u8(u1, tab[1][ri])),
                     vqtbl1q_u8(u2, tab[2][ri])), vqtbl1q_u8(u3, tab[3][ri]));
        int8x16_t glo = actq_w_from_r(Rg, 0, m15, x8, s8);
        int8x16_t ghi = actq_w_from_r(Rg, 1, m15, x8, s8);
        int8x16_t ulo = actq_w_from_r(Ru, 0, m15, x8, s8);
        int8x16_t uhi = actq_w_from_r(Ru, 1, m15, x8, s8);
        pgs[ri] = vaddvq_s32(vaddq_s32(i8x16_dot_s32(glo, xl),
                                       i8x16_dot_s32(ghi, xh)));
        pus[ri] = vaddvq_s32(vaddq_s32(i8x16_dot_s32(ulo, xl),
                                       i8x16_dot_s32(uhi, xh)));
    }
    int32x4_t pg = vld1q_s32(pgs), pu = vld1q_s32(pus);
    /* f32 累加：acc += (sc_ri * xd_cb) * dot_i32（每块 1 次乘加） */
    *ga = vaddq_f32(*ga, vmulq_f32(vmulq_f32(scg, xdv), vcvtq_f32_s32(pg)));
    *ua = vaddq_f32(*ua, vmulq_f32(vmulq_f32(scu, xdv), vcvtq_f32_s32(pu)));
}

/* gate/up：每 4 行组跨 nbG 个 tile（沿 d 列）。gv/uv[4]。近似轨。 */
static void moe_q4_gu_group_actq_neon(const uint8_t *gb, const uint8_t *ub, int nbG,
                                      const int8_t *xq, const float *xd,
                                      float *gv, float *uv) {
    actq_tab_ensure();
    uint8x16_t tab[4][4];
    for (int q = 0; q < 4; q++)
        for (int ri = 0; ri < 4; ri++)
            tab[q][ri] = vld1q_u8(actq_tab[q][ri]);
    uint8x16_t m15 = vdupq_n_u8(15), x8 = vdupq_n_u8(8);
    int8x16_t s8 = vdupq_n_s8(8);
    float32x4_t ga = vdupq_n_f32(0.0f), ua = vdupq_n_f32(0.0f);
    for (int cb = 0; cb < nbG; cb++) {
        q4x4_gup_actq_neon(gb + (size_t)cb * 72, ub + (size_t)cb * 72,
                           xq + (size_t)cb * 32, xd[cb], &ga, &ua,
                           tab, m15, x8, s8);
    }
    vst1q_f32(gv, ga);
    vst1q_f32(uv, ua);
}

/* down：每 4 输出行组跨 cbe 个 tile（专家 down 列带）。y += p*acc。近似轨。 */
static void moe_q4_down_group_actq_neon(const uint8_t *db, int cbe,
                                        const int8_t *avq, const float *avd,
                                        float p, float *y) {
    actq_tab_ensure();
    uint8x16_t tab[4][4];
    for (int q = 0; q < 4; q++)
        for (int ri = 0; ri < 4; ri++)
            tab[q][ri] = vld1q_u8(actq_tab[q][ri]);
    uint8x16_t m15 = vdupq_n_u8(15), x8 = vdupq_n_u8(8);
    int8x16_t s8 = vdupq_n_s8(8);
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (int cbi = 0; cbi < cbe; cbi++) {
        /* 每 tile（4 输出行 × 32 av 列）：权重字节还原 (nib-8)、vdot avq，
         * f32 累加 acc += (sc_ri*avd[cbi])*dot_i32 */
        uint8x16_t d0 = vld1q_u8(db + (size_t)cbi * 72 + 8);
        uint8x16_t d1 = vld1q_u8(db + (size_t)cbi * 72 + 24);
        uint8x16_t d2 = vld1q_u8(db + (size_t)cbi * 72 + 40);
        uint8x16_t d3 = vld1q_u8(db + (size_t)cbi * 72 + 56);
        int8x16_t xl = vld1q_s8(avq + (size_t)cbi * 32);
        int8x16_t xh = vld1q_s8(avq + (size_t)cbi * 32 + 16);
        uint16_t h[4]; float sd[4];
        memcpy(h, db + (size_t)cbi * 72, 8);
        for (int i = 0; i < 4; i++) sd[i] = vq_f16(h[i]);
        float32x4_t sc = vld1q_f32(sd), xdv = vdupq_n_f32(avd[cbi]);
        /* 同 q4x4_gup_actq_neon：vsetq_lane_s32 需常量 lane，改用数组收集 */
        int32_t pas[4];
        for (int ri = 0; ri < 4; ri++) {
            uint8x16_t Rd = vorrq_u8(
                vorrq_u8(vorrq_u8(vqtbl1q_u8(d0, tab[0][ri]), vqtbl1q_u8(d1, tab[1][ri])),
                         vqtbl1q_u8(d2, tab[2][ri])), vqtbl1q_u8(d3, tab[3][ri]));
            int8x16_t lo = actq_w_from_r(Rd, 0, m15, x8, s8);
            int8x16_t hi = actq_w_from_r(Rd, 1, m15, x8, s8);
            pas[ri] = vaddvq_s32(vaddq_s32(i8x16_dot_s32(lo, xl),
                                           i8x16_dot_s32(hi, xh)));
        }
        int32x4_t pa = vld1q_s32(pas);
        acc = vaddq_f32(acc, vmulq_f32(vmulq_f32(sc, xdv), vcvtq_f32_s32(pa)));
    }
    float32x4_t pv = vdupq_n_f32(p);
    float32x4_t yo = vld1q_f32(y);
    vst1q_f32(y, vaddq_f32(yo, vmulq_f32(pv, acc)));
}
#endif /* __aarch64__ && ST_HAVE_NEON && !ST_ARCH_X86 && ST_NEON_DOTPROD */

/* MoE FFN q8_0 稀疏解码（A 路径：wmode=q8 专家，8x8 tiled / 与 q4 同几何）。
 * 行空间与 st_moe_ffn_sparse（q4）完全一致：gate/up = [ff×d]/层（专家 e 行
 * 段 [e*ef, e*ef+ef)），down = [d×ff]/层（专家 e 列带 [e*ef, e*ef+ef)）。
 * 默认逐行 st_q8_row_dot / st_q8_band_dot 标量解码；x86+8x8(mode2) 走
 * SIMD 8 行组块解码（VLLM_MOE_SIMD=0 关闭/1=SSE128/2=AVX2，见下），算术等价、
 * 同二进制确定（128 与 256 变体逐位一致，相对标量仅末位差异）。 */

/* VLLM_MOE_SIMD 等级：0=关闭（逐行标量原循环）；1=SSE128（8x8 单块 8 行点积，
 * 每 4 行一个 128-bit 累加器，历史路径）；2/缺省=AVX2 256-bit（8 行一个 256-bit
 * 累加器）。两 SIMD 内核的逐行 FMA 链序一致（k=0..7 外层、i=0..3 内层），
 * 输出 out8[] 逐位相同；与标量仅末位差异（文档化近似，fma 单舍入 vs mul+add）。 */
static int moe_q8_simd_lvl(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("VLLM_MOE_SIMD");
        if (e && e[0] == '0') v = 0;
        else if (e && e[0] == '1') v = 1;
        else v = 2;                      /* x86 8x8 默认 AVX2 */
    }
    return v;
}
static int moe_q8_simd_ok(void) { return moe_q8_simd_lvl() != 0; }

#if ST_ARCH_X86 && !ST_HAVE_NEON
/* 8x8(tiled) 单块（272B）8 行 int-dot：out8[m] = Σ_c qs(m,c)·x[c]（不含 scale）。
 * qs@16..272：8 chunk(k=0..7)×32B，chunk k=8 行×4 列（行 m 的 4 int8 在
 * 偏移 4m），列 c=4k+i → 块内 stride-4 抽取（同 st_q8_row_dot mode2 语义）。
 * 归约顺序固定（k 外层 0..7 → i 内层 0..3 逐 fma），两 SIMD 变体输出逐位一致。 */
/* SSE128 变体：半块（4 行）先字节转置为列向量 → ×x[4k+i] 广播 FMA → 每 4 行
 * 一个 128-bit 行累加器（lane=行）。 */
static void q8x8b_dot8_x86_sse(const uint8_t *blk, const float *x, float out8[8]) {
    static const uint8_t CM[4][16] = {
        { 0,  4,  8, 12, 0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80 },
        { 1,  5,  9, 13, 0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80 },
        { 2,  6, 10, 14, 0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80 },
        { 3,  7, 11, 15, 0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80 },
    };
    const __m128i msk[4] = {
        _mm_loadu_si128((const __m128i *)CM[0]),
        _mm_loadu_si128((const __m128i *)CM[1]),
        _mm_loadu_si128((const __m128i *)CM[2]),
        _mm_loadu_si128((const __m128i *)CM[3]),
    };
    const uint8_t *qs = blk + 16;
    __m128 a0 = _mm_setzero_ps(), a1 = _mm_setzero_ps();
    for (int k = 0; k < 8; k++) {
        const uint8_t *cp = qs + (size_t)k * 32;
        __m128i lo = _mm_loadu_si128((const __m128i *)cp);      /* rows 0-3 */
        __m128i hi = _mm_loadu_si128((const __m128i *)(cp + 16)); /* rows 4-7 */
        __m128 xq = _mm_loadu_ps(x + (size_t)k * 4);
        __m128 b0 = _mm_shuffle_ps(xq, xq, 0x00);
        __m128 b1 = _mm_shuffle_ps(xq, xq, 0x55);
        __m128 b2 = _mm_shuffle_ps(xq, xq, 0xAA);
        __m128 b3 = _mm_shuffle_ps(xq, xq, 0xFF);
        __m128 c0 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_shuffle_epi8(lo, msk[0])));
        __m128 c1 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_shuffle_epi8(lo, msk[1])));
        __m128 c2 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_shuffle_epi8(lo, msk[2])));
        __m128 c3 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_shuffle_epi8(lo, msk[3])));
        a0 = _mm_fmadd_ps(b0, c0, a0);
        a0 = _mm_fmadd_ps(b1, c1, a0);
        a0 = _mm_fmadd_ps(b2, c2, a0);
        a0 = _mm_fmadd_ps(b3, c3, a0);
        __m128 d0 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_shuffle_epi8(hi, msk[0])));
        __m128 d1 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_shuffle_epi8(hi, msk[1])));
        __m128 d2 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_shuffle_epi8(hi, msk[2])));
        __m128 d3 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_shuffle_epi8(hi, msk[3])));
        a1 = _mm_fmadd_ps(b0, d0, a1);
        a1 = _mm_fmadd_ps(b1, d1, a1);
        a1 = _mm_fmadd_ps(b2, d2, a1);
        a1 = _mm_fmadd_ps(b3, d3, a1);
    }
    _mm_storeu_ps(out8, a0);
    _mm_storeu_ps(out8 + 4, a1);
}

#if defined(__AVX2__)
/* AVX2 256-bit 变体：同 32B chunk 的 lo/hi 半块各自转置出列向量（4×128-bit），
 * 以 insertf128 拼成 8-lane（lane=行 0..7）单个 __m256 累加器；×x[4k+i] 广播
 * 256-bit FMA。逐行 FMA 链序与 SSE128 变体一致 → out8 逐位相同。 */
static void q8x8b_dot8_x86_avx2(const uint8_t *blk, const float *x, float out8[8]) {
    static const uint8_t CM[4][16] = {
        { 0,  4,  8, 12, 0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80 },
        { 1,  5,  9, 13, 0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80 },
        { 2,  6, 10, 14, 0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80 },
        { 3,  7, 11, 15, 0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80 },
    };
    const __m128i msk[4] = {
        _mm_loadu_si128((const __m128i *)CM[0]),
        _mm_loadu_si128((const __m128i *)CM[1]),
        _mm_loadu_si128((const __m128i *)CM[2]),
        _mm_loadu_si128((const __m128i *)CM[3]),
    };
    const uint8_t *qs = blk + 16;
    __m256 a = _mm256_setzero_ps();
    for (int k = 0; k < 8; k++) {
        const uint8_t *cp = qs + (size_t)k * 32;
        __m128i lo = _mm_loadu_si128((const __m128i *)cp);        /* rows 0-3 */
        __m128i hi = _mm_loadu_si128((const __m128i *)(cp + 16)); /* rows 4-7 */
        __m128 l0 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_shuffle_epi8(lo, msk[0])));
        __m128 l1 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_shuffle_epi8(lo, msk[1])));
        __m128 l2 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_shuffle_epi8(lo, msk[2])));
        __m128 l3 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_shuffle_epi8(lo, msk[3])));
        __m128 h0 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_shuffle_epi8(hi, msk[0])));
        __m128 h1 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_shuffle_epi8(hi, msk[1])));
        __m128 h2 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_shuffle_epi8(hi, msk[2])));
        __m128 h3 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_shuffle_epi8(hi, msk[3])));
        /* lane = 行 0..7：低 128 = rows 0-3，高 128 = rows 4-7（同列 i） */
        __m256 c0 = _mm256_insertf128_ps(_mm256_castps128_ps256(l0), h0, 1);
        __m256 c1 = _mm256_insertf128_ps(_mm256_castps128_ps256(l1), h1, 1);
        __m256 c2 = _mm256_insertf128_ps(_mm256_castps128_ps256(l2), h2, 1);
        __m256 c3 = _mm256_insertf128_ps(_mm256_castps128_ps256(l3), h3, 1);
        __m256 b0 = _mm256_broadcast_ss(x + (size_t)k * 4 + 0);
        __m256 b1 = _mm256_broadcast_ss(x + (size_t)k * 4 + 1);
        __m256 b2 = _mm256_broadcast_ss(x + (size_t)k * 4 + 2);
        __m256 b3 = _mm256_broadcast_ss(x + (size_t)k * 4 + 3);
        a = _mm256_fmadd_ps(b0, c0, a);
        a = _mm256_fmadd_ps(b1, c1, a);
        a = _mm256_fmadd_ps(b2, c2, a);
        a = _mm256_fmadd_ps(b3, c3, a);
    }
    _mm256_storeu_ps(out8, a);
}
#endif /* __AVX2__ */

/* 8x8 单块 8 行点积分派：VLLM_MOE_SIMD=2（默认）走 AVX2 256-bit；=1 走历史
 * SSE128。两内核逐位一致（同 FMA 链序）。 */
static void q8x8b_dot8_x86(const uint8_t *blk, const float *x, float out8[8]) {
#if defined(__AVX2__)
    if (moe_q8_simd_lvl() == 2) { q8x8b_dot8_x86_avx2(blk, x, out8); return; }
#endif
    q8x8b_dot8_x86_sse(blk, x, out8);
}
#endif

/* ARM/Linux 对等实现：st_q8_layout / st_q8_row_dot / st_q8_band_dot
 * （x86 映射区 #if ST_ARCH_X86 && !ST_HAVE_NEON 内的同名标量内核在 ARM 构建
 * 不参与编译，而 MoE 稀疏 q8 解码（moe_q8_gu/down 标量兜底）依赖它们）。
 * 布局感知语义与 x86 标量逐字节一致（纯 C、零 intrinsic）：q8 8x8 tiled =
 * 272B/8行组 { 8×f16 scale }@0 + qs@16，行 m 元素 (k*4+i) 在 qs[16+k*32+m*4+i]。
 * ARM（dotprod）VQF 恒为 8x8（g_st_q8_repack 默认开），mode = st_q8_layout()。 */
#if !(ST_ARCH_X86 && !ST_HAVE_NEON)
extern int g_st_q8_repack;
extern int g_st_q4_repack;
static float st_f16scale_arm(const uint8_t *b) {
    uint16_t h;
    memcpy(&h, b, 2);
    uint32_t fb = f16_to_f32_bits(h);
    float d;
    memcpy(&d, &fb, 4);
    return d;
}
static int st_q8_layout(void) {
    if (!g_st_q8_repack) return 0;
    return q8_8x8_enabled() ? 2 : 1;
}
static float st_q8_row_dot(const uint8_t *w, int r, const float *x, int n_blocks, int mode) {
    float acc = 0.0f;
    if (mode == 0) {
        const uint8_t *pr = w + (size_t)r * (size_t)n_blocks * 34;
        for (int b = 0; b < n_blocks; b++) {
            float d = st_f16scale_arm(pr + (size_t)b * 34);
            const int8_t *qs = (const int8_t *)(pr + (size_t)b * 34 + 2);
            const float *xs = x + (size_t)b * 32;
            float s = 0.0f;
            for (int j = 0; j < 32; j++) s += (float)qs[j] * xs[j];
            acc += d * s;
        }
    } else if (mode == 1) {
        int m = r & 3;
        const uint8_t *g = w + (size_t)(r >> 2) * (size_t)n_blocks * 136;
        for (int b = 0; b < n_blocks; b++) {
            const uint8_t *out = g + (size_t)b * 136;
            float d = st_f16scale_arm(out + 2 * m);
            const float *xs = x + (size_t)b * 32;
            float s = 0.0f;
            for (int k = 0; k < 8; k++)
                for (int i = 0; i < 4; i++)
                    s += (float)(int8_t)out[8 + k * 16 + m * 4 + i] * xs[k * 4 + i];
            acc += d * s;
        }
    } else {
        int m = r & 7;
        const uint8_t *g = w + (size_t)(r >> 3) * (size_t)n_blocks * 272;
        for (int b = 0; b < n_blocks; b++) {
            const uint8_t *out = g + (size_t)b * 272;
            float d = st_f16scale_arm(out + 2 * m);
            const float *xs = x + (size_t)b * 32;
            float s = 0.0f;
            for (int k = 0; k < 8; k++)
                for (int i = 0; i < 4; i++)
                    s += (float)(int8_t)out[16 + k * 32 + m * 4 + i] * xs[k * 4 + i];
            acc += d * s;
        }
    }
    return acc;
}
static float st_q8_band_dot(const uint8_t *w, int r, const float *x,
                            int cb0, int cbe, int n_blocks, int mode) {
    float acc = 0.0f;
    if (mode == 0) {
        const uint8_t *pr = w + (size_t)r * (size_t)n_blocks * 34;
        for (int b = cb0; b < cb0 + cbe; b++) {
            const uint8_t *blk = pr + (size_t)b * 34;
            float d = st_f16scale_arm(blk);
            const int8_t *qs = (const int8_t *)(blk + 2);
            const float *xs = x + (size_t)(b - cb0) * 32;
            float s = 0.0f;
            for (int j = 0; j < 32; j++) s += (float)qs[j] * xs[j];
            acc += d * s;
        }
    } else if (mode == 1) {
        int m = r & 3;
        const uint8_t *g = w + (size_t)(r >> 2) * (size_t)n_blocks * 136;
        for (int b = cb0; b < cb0 + cbe; b++) {
            const uint8_t *out = g + (size_t)b * 136;
            float d = st_f16scale_arm(out + 2 * m);
            const float *xs = x + (size_t)(b - cb0) * 32;
            float s = 0.0f;
            for (int k = 0; k < 8; k++)
                for (int i = 0; i < 4; i++)
                    s += (float)(int8_t)out[8 + k * 16 + m * 4 + i] * xs[k * 4 + i];
            acc += d * s;
        }
    } else {
        int m = r & 7;
        const uint8_t *g = w + (size_t)(r >> 3) * (size_t)n_blocks * 272;
        for (int b = cb0; b < cb0 + cbe; b++) {
            const uint8_t *out = g + (size_t)b * 272;
            float d = st_f16scale_arm(out + 2 * m);
            const float *xs = x + (size_t)(b - cb0) * 32;
            float s = 0.0f;
            for (int k = 0; k < 8; k++)
                for (int i = 0; i < 4; i++)
                    s += (float)(int8_t)out[16 + k * 32 + m * 4 + i] * xs[k * 4 + i];
            acc += d * s;
        }
    }
    return acc;
}
#endif /* !(ST_ARCH_X86 && !ST_HAVE_NEON) */

#if defined(__aarch64__) && defined(ST_HAVE_NEON) && !ST_ARCH_X86
/* 8x8(tiled) 单块（272B）8 行 int-dot（ARM NEON 变体）：out8[m] = Σ_c qs(m,c)·x[c]
 * （不含 scale，scale 由调用侧逐块乘加）。布局/归约语义与 x86 SSE128 内核一致：
 * qs@16，8 chunk(k=0..7)×32B；chunk k 内字节 = 行 m 的 4 int8 在偏移 4m（列
 * c=k*4+i），即全局字节 m*4+i；k 外层 0..7 → i 内层 0..3 逐列累加。实现：vqtbl2q
 * 一次从 32B chunk 拾取 8 行 × 列 i（索引 {i,4+i,…,28+i}），s8→s16→s32→f32 加宽
 * 后与 x[k*4+i] 广播 vfma；a0/a1 = 行 0-3 / 行 4-7 两组 128-bit 行累加器（链序与
 * x86 逐列一致；vs 标量 mul+add 仅末位差异，确定性保持）。 */
static void q8x8b_dot8_neon(const uint8_t *blk, const float *x, float out8[8]) {
    static const uint8_t CM[4][16] = {
        { 0, 4, 8, 12, 16, 20, 24, 28, 0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80 },
        { 1, 5, 9, 13, 17, 21, 25, 29, 0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80 },
        { 2, 6,10, 14, 18, 22, 26, 30, 0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80 },
        { 3, 7,11, 15, 19, 23, 27, 31, 0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80 },
    };
    const uint8x16_t ix[4] = {
        vld1q_u8(CM[0]), vld1q_u8(CM[1]), vld1q_u8(CM[2]), vld1q_u8(CM[3]),
    };
    const uint8_t *qs = blk + 16;
    float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
    for (int k = 0; k < 8; k++) {
        const uint8_t *cp = qs + (size_t)k * 32;
        uint8x16x2_t T;
        T.val[0] = vld1q_u8(cp);
        T.val[1] = vld1q_u8(cp + 16);
        float32x4_t b0 = vdupq_n_f32(x[(size_t)k * 4 + 0]);
        float32x4_t b1 = vdupq_n_f32(x[(size_t)k * 4 + 1]);
        float32x4_t b2 = vdupq_n_f32(x[(size_t)k * 4 + 2]);
        float32x4_t b3 = vdupq_n_f32(x[(size_t)k * 4 + 3]);
        uint8x16_t c0 = vqtbl2q_u8(T, ix[0]);
        int16x8_t w = vmovl_s8(vget_low_s8(vreinterpretq_s8_u8(c0)));
        int32x4_t s0 = vmovl_s16(vget_low_s16(w));
        int32x4_t s1 = vmovl_s16(vget_high_s16(w));
        a0 = vfmaq_f32(a0, b0, vcvtq_f32_s32(s0));
        a1 = vfmaq_f32(a1, b0, vcvtq_f32_s32(s1));
        uint8x16_t c1 = vqtbl2q_u8(T, ix[1]);
        w = vmovl_s8(vget_low_s8(vreinterpretq_s8_u8(c1)));
        s0 = vmovl_s16(vget_low_s16(w));
        s1 = vmovl_s16(vget_high_s16(w));
        a0 = vfmaq_f32(a0, b1, vcvtq_f32_s32(s0));
        a1 = vfmaq_f32(a1, b1, vcvtq_f32_s32(s1));
        uint8x16_t c2 = vqtbl2q_u8(T, ix[2]);
        w = vmovl_s8(vget_low_s8(vreinterpretq_s8_u8(c2)));
        s0 = vmovl_s16(vget_low_s16(w));
        s1 = vmovl_s16(vget_high_s16(w));
        a0 = vfmaq_f32(a0, b2, vcvtq_f32_s32(s0));
        a1 = vfmaq_f32(a1, b2, vcvtq_f32_s32(s1));
        uint8x16_t c3 = vqtbl2q_u8(T, ix[3]);
        w = vmovl_s8(vget_low_s8(vreinterpretq_s8_u8(c3)));
        s0 = vmovl_s16(vget_low_s16(w));
        s1 = vmovl_s16(vget_high_s16(w));
        a0 = vfmaq_f32(a0, b3, vcvtq_f32_s32(s0));
        a1 = vfmaq_f32(a1, b3, vcvtq_f32_s32(s1));
    }
    vst1q_f32(out8, a0);
    vst1q_f32(out8 + 4, a1);
}
#endif /* __aarch64__ && ST_HAVE_NEON && !ST_ARCH_X86 */

/* 单专家 gate/up：gv/uv[ef]。simd=1（mode2 8x8）→ 8 行组分块（x86 SSE/AVX2、
 * ARM NEON 变体）；否则逐行标量。两种 SIMD 归约链同构（k 外层 → i 内层逐列 fma），
 * vs 标量仅末位差异（文档化近似）。 */
static void moe_q8_gu(const uint8_t *G, const uint8_t *U, int rb, int ef,
                      int nbG, int mode8, int simd,
                      const float *x, float *gv, float *uv) {
#if defined(__aarch64__) && defined(ST_HAVE_NEON) && !ST_ARCH_X86
    if (simd && mode8 == 2) {
        const uint8_t *gB = G + (size_t)(rb >> 3) * (size_t)nbG * 272;
        const uint8_t *uB = U + (size_t)(rb >> 3) * (size_t)nbG * 272;
        for (int gg = 0; gg < ef / 8; gg++) {
            /* ga/ua 以两个 f32x4（行 0-3/4-7）作累加器，块循环内 vfma 收缩 =
             * -ffast-math 下既有标量 fma 单舍入语义，逐位一致。 */
            float32x4_t ga0 = vdupq_n_f32(0.0f), ga1 = vdupq_n_f32(0.0f);
            float32x4_t ua0 = vdupq_n_f32(0.0f), ua1 = vdupq_n_f32(0.0f);
            const uint8_t *gb = gB + (size_t)gg * (size_t)nbG * 272;
            const uint8_t *ub = uB + (size_t)gg * (size_t)nbG * 272;
            for (int b = 0; b < nbG; b++) {
                const uint8_t *gp = gb + (size_t)b * 272;
                const uint8_t *up = ub + (size_t)b * 272;
                float g8[8], u8[8];
                q8x8b_dot8_neon(gp, x + (size_t)b * 32, g8);
                q8x8b_dot8_neon(up, x + (size_t)b * 32, u8);
                /* scale 头一次 16B 载入解码（8×f16，与逐元素 st_f16scale_arm
                 * 同一 f16_to_f32_bits 语义），再按行 fma 进累加器。 */
                uint16_t gh[8], uh[8];
                memcpy(gh, gp, 16); memcpy(uh, up, 16);
                float gs[8], us[8];
                for (int m = 0; m < 8; m++) {
                    gs[m] = st_f16scale_arm((const uint8_t *)&gh[m]);
                    us[m] = st_f16scale_arm((const uint8_t *)&uh[m]);
                }
                float32x4_t gsv0 = vld1q_f32(gs), gsv1 = vld1q_f32(gs + 4);
                float32x4_t usv0 = vld1q_f32(us), usv1 = vld1q_f32(us + 4);
                float32x4_t go0 = vld1q_f32(g8), go1 = vld1q_f32(g8 + 4);
                float32x4_t uo0 = vld1q_f32(u8), uo1 = vld1q_f32(u8 + 4);
                ga0 = vfmaq_f32(ga0, gsv0, go0);
                ga1 = vfmaq_f32(ga1, gsv1, go1);
                ua0 = vfmaq_f32(ua0, usv0, uo0);
                ua1 = vfmaq_f32(ua1, usv1, uo1);
            }
            float ga[8], ua[8];
            vst1q_f32(ga, ga0); vst1q_f32(ga + 4, ga1);
            vst1q_f32(ua, ua0); vst1q_f32(ua + 4, ua1);
            for (int m = 0; m < 8; m++) { gv[gg * 8 + m] = ga[m]; uv[gg * 8 + m] = ua[m]; }
        }
        return;
    }
#elif ST_ARCH_X86 && !ST_HAVE_NEON
    if (simd && mode8 == 2) {
        const uint8_t *gB = G + (size_t)(rb >> 3) * (size_t)nbG * 272;
        const uint8_t *uB = U + (size_t)(rb >> 3) * (size_t)nbG * 272;
        for (int gg = 0; gg < ef / 8; gg++) {
            float g8[8], u8[8], ga[8], ua[8];
            for (int m = 0; m < 8; m++) { ga[m] = 0.0f; ua[m] = 0.0f; }
            const uint8_t *gb = gB + (size_t)gg * (size_t)nbG * 272;
            const uint8_t *ub = uB + (size_t)gg * (size_t)nbG * 272;
            for (int b = 0; b < nbG; b++) {
                const uint8_t *gp = gb + (size_t)b * 272;
                const uint8_t *up = ub + (size_t)b * 272;
                q8x8b_dot8_x86(gp, x + (size_t)b * 32, g8);
                q8x8b_dot8_x86(up, x + (size_t)b * 32, u8);
                for (int m = 0; m < 8; m++) {
                    ga[m] += st_f16scale(gp + 2 * m) * g8[m];
                    ua[m] += st_f16scale(up + 2 * m) * u8[m];
                }
            }
            for (int m = 0; m < 8; m++) { gv[gg * 8 + m] = ga[m]; uv[gg * 8 + m] = ua[m]; }
        }
        return;
    }
#endif
    (void)simd; (void)mode8;
    for (int rr = 0; rr < ef; rr++) {
        gv[rr] = st_q8_row_dot(G, rb + rr, x, nbG, mode8);
        uv[rr] = st_q8_row_dot(U, rb + rr, x, nbG, mode8);
    }
}

/* 单专家 down：y[d] += p·D 列带 [cb0*32, (cb0+cbe)*32) × av。simd 同 moe_q8_gu。 */
static void moe_q8_down(const uint8_t *D, int d, int nbD, int cb0, int cbe,
                        int mode8, int simd, const float *av, float p, float *y) {
#if defined(__aarch64__) && defined(ST_HAVE_NEON) && !ST_ARCH_X86
    if (simd && mode8 == 2) {
        for (int gg = 0; gg < d / 8; gg++) {
            float32x4_t ac0 = vdupq_n_f32(0.0f), ac1 = vdupq_n_f32(0.0f);
            const uint8_t *db = D + (size_t)gg * (size_t)nbD * 272;
            for (int cbi = 0; cbi < cbe; cbi++) {
                const uint8_t *bp = db + (size_t)(cb0 + cbi) * 272;
                float o8[8];
                q8x8b_dot8_neon(bp, av + (size_t)cbi * 32, o8);
                /* scale 头一次 16B 载入解码 + vfma（与 -ffast-math 标量 fma 同序） */
                uint16_t h[8];
                memcpy(h, bp, 16);
                float sd[8];
                for (int m = 0; m < 8; m++) sd[m] = st_f16scale_arm((const uint8_t *)&h[m]);
                float32x4_t sv0 = vld1q_f32(sd), sv1 = vld1q_f32(sd + 4);
                float32x4_t ov0 = vld1q_f32(o8), ov1 = vld1q_f32(o8 + 4);
                ac0 = vfmaq_f32(ac0, sv0, ov0);
                ac1 = vfmaq_f32(ac1, sv1, ov1);
            }
            float acc[8];
            vst1q_f32(acc, ac0); vst1q_f32(acc + 4, ac1);
            float pv = p;
            for (int m = 0; m < 8; m++) y[(size_t)gg * 8 + m] += pv * acc[m];
        }
        return;
    }
#elif ST_ARCH_X86 && !ST_HAVE_NEON
    if (simd && mode8 == 2) {
        for (int gg = 0; gg < d / 8; gg++) {
            float acc[8], o8[8];
            for (int m = 0; m < 8; m++) acc[m] = 0.0f;
            const uint8_t *db = D + (size_t)gg * (size_t)nbD * 272;
            for (int cbi = 0; cbi < cbe; cbi++) {
                const uint8_t *bp = db + (size_t)(cb0 + cbi) * 272;
                q8x8b_dot8_x86(bp, av + (size_t)cbi * 32, o8);
                for (int m = 0; m < 8; m++)
                    acc[m] += st_f16scale(bp + 2 * m) * o8[m];
            }
            for (int m = 0; m < 8; m++) y[(size_t)gg * 8 + m] += p * acc[m];
        }
        return;
    }
#endif
    (void)simd; (void)mode8;
    for (int r = 0; r < d; r++)
        y[r] += p * st_q8_band_dot(D, r, av, cb0, cbe, nbD, mode8);
}

/* ---- MoE 专家并行（q8 路径）----
 * A≡C 硬约束：MoE FFN 输出 y = Σ_j p_j·down_j(silu(gate_j(x))·up_j(x))。串行
 * 语义 = y 从 0 起按 j=0..tk-1 逐个 y += down_j（down 内部列带累加序固定）。若把
 * down 也并行到独立 slot 再归约，累加序与串行不完全一致（浮点非结合），会引入
 * 逐位差异——禁止。因此：gate/up→silu 的 av 计算每专家独立、无共享累加，可并行
 * （结果与串行逐位一致）；down 由调用线程按 j 升序逐个累加进 y，与串行完全同序。
 * VLLM_MOE_PAR=0 关闭（回退逐专家全串行原循环）。*/
static int moe_q8_par_ok(void) {
    const char *e = getenv("VLLM_MOE_PAR");
    if (e && e[0] == '0') return 0;
    return 1;
}

typedef struct {
    const uint8_t *G, *U;
    const float *x_ffn;
    float *av_accum;     /* [tk*ef]，每专家激活 av（silu(gate)·up） */
    int ef, nbG, mode8, simd;
    int ne, tk;
    const int   *sel;
} moe_q8_par_ctx;

static void moe_q8_par_worker(void *ctx_, int j) {
    moe_q8_par_ctx *c = (moe_q8_par_ctx *)ctx_;
    if (j < 0 || j >= c->tk) return;
    int e = c->sel[j];
    if (e < 0 || e >= c->ne) return;
    float gv[2048], uv[2048];
    if (c->ef > 2048) return;
    int rb = e * c->ef;
    moe_q8_gu(c->G, c->U, rb, c->ef, c->nbG, c->mode8, c->simd,
              c->x_ffn, gv, uv);
    float *av = c->av_accum + (size_t)j * c->ef;
    for (int i = 0; i < c->ef; i++) {
        float g = gv[i], u = uv[i];
        av[i] = (g / (1.0f + expf(-g))) * u;   /* silu(g)*u */
    }
}

/* ---- 真实 router 热度采集（EW 标定 H 用，VLLM_EW_HEAT 由 st_moe_heat_reset 开） ----
 * 在 MoE FFN 内按层累计被选中专家计数，落盘 "l:e:count"（计数>0 才写）。 */
static int g_moe_heat_on = 0;
static int g_moe_heat_nl = 0, g_moe_heat_ne = 0;
static uint32_t *g_moe_heat = NULL;   /* nl×ne */

void st_moe_heat_reset(const STModelWeights *w) {
    int nl = w ? w->cfg.n_layers : 0, ne = w ? w->cfg.n_experts : 0;
    if (nl <= 0 || ne <= 0) return;
    size_t n = (size_t)nl * ne;
    if (!g_moe_heat) g_moe_heat = (uint32_t *)calloc(n, sizeof(uint32_t));
    else memset(g_moe_heat, 0, n * sizeof(uint32_t));
    g_moe_heat_nl = nl; g_moe_heat_ne = ne;
    g_moe_heat_on = 1;
}

void st_moe_heat_save(const char *path) {
    g_moe_heat_on = 0;
    if (!g_moe_heat || g_moe_heat_nl <= 0 || !path) return;
    FILE *f = fopen(path, "w");
    if (!f) { fprintf(stderr, "[MOE-HEAT] save 失败: %s\n", path); return; }
    for (int l = 0; l < g_moe_heat_nl; l++)
        for (int e = 0; e < g_moe_heat_ne; e++) {
            uint32_t c = g_moe_heat[(size_t)l * (size_t)g_moe_heat_ne + e];
            if (c) fprintf(f, "%d:%d:%u\n", l, e, c);
        }
    fclose(f);
    fprintf(stderr, "[MOE-HEAT] saved %s\n", path);
}

static void st_moe_ffn_sparse_q8(const STModelWeights *w, int l,
                                 const float *x_ffn, float *y) {
    const STModelConfig *c = &w->cfg;
    int ne = c->n_experts, tk = c->top_k, d = c->dim, ff = c->ffn_dim;
    if (ne <= 0 || tk <= 0 || tk > 64 || ne > 4096 || d <= 0) return;
    int ef = ff / ne;
    memset(y, 0, (size_t)d * sizeof(float));   /* 先清零：守卫失败亦不留未初始化输出 */
    if (ef <= 0 || ne * ef != ff || (ef & 31) || (d & 31)) return;
    const float *router = w->moe_router + (size_t)l * ne * d;
    const uint8_t *G = w->q8_gate_weight + Q8_BYTES((size_t)l * (size_t)ff * d);
    const uint8_t *U = w->q8_up_weight   + Q8_BYTES((size_t)l * (size_t)ff * d);
    const uint8_t *D = w->q8_down_weight + Q8_BYTES((size_t)l * (size_t)d * ff);
    if (!G || !U || !D) return;
    int nbG = d >> 5;                 /* gate/up 行内 32-col 块数（=dim/32） */
    int nbD = ff >> 5;                /* down 行内块数（=ff/32） */
    int mode8 = st_q8_layout();
    int simd = mode8 == 2 && (d & 7) == 0 && (ef & 7) == 0 && (ff & 7) == 0
               && moe_q8_simd_ok();

    /* 1) router logits + 确定性 top-k（同分取小索引；与 q4 路径逐位一致） */
    float lg[4096];
    for (int e = 0; e < ne; e++) {
        const float *rw = router + (size_t)e * d;
        float s = 0.0f;
        for (int i = 0; i < d; i++) s += rw[i] * x_ffn[i];
        lg[e] = s;
    }
    int sel[64]; float sv[64];
    for (int k = 0; k < tk; k++) { sel[k] = -1; sv[k] = -INFINITY; }
    for (int e = 0; e < ne; e++) {
        for (int k = 0; k < tk; k++) {
            if (lg[e] > sv[k] || (lg[e] == sv[k] && (sel[k] < 0 || e < sel[k]))) {
                for (int j = tk - 1; j > k; j--) { sel[j] = sel[j - 1]; sv[j] = sv[j - 1]; }
                sel[k] = e; sv[k] = lg[e];
                break;
            }
        }
    }
    float mx = -INFINITY;
    for (int k = 0; k < tk; k++) if (sv[k] > mx) mx = sv[k];
    float sum = 0.0f, pr[64];
    for (int k = 0; k < tk; k++) { pr[k] = expf(sv[k] - mx); sum += pr[k]; }
    float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    for (int k = 0; k < tk; k++) pr[k] *= inv;
    if (l == 0 && getenv("VLLM_MOE_DUMP") && getenv("VLLM_MOE_DUMP")[0] == '1') {
        printf("[L0D] S9_router");
        for (int k = 0; k < tk; k++) printf(" %d:%.6e", sel[k], (double)pr[k]);
        printf("\n");
        fflush(stdout);
    }
    /* 真实 router 热度采集（EW 标定 H 用）：VLLM_EW_HEAT 打开后按层累计选中 */
    if (g_moe_heat_on && g_moe_heat_ne > 0) {
        uint32_t *h = g_moe_heat + (size_t)l * (size_t)g_moe_heat_ne;
        for (int j = 0; j < tk; j++)
            if (sel[j] >= 0 && sel[j] < g_moe_heat_ne) h[sel[j]]++;
    }

    /* 2) 每个选中专家：gate/up（SIMD 8 行组 / 逐行标量）→ silu → down 累加。
     * VLLM_MOE_PAR=1（默认）且 vllm_tp 线程数 >1 时并行：gate/up→silu 的 av
     * 计算每专家独立（无共享累加，与串行逐位一致）由线程池分块完成并写入
     * av_accum[j*ef..]；down 由调用线程按 j=0..tk-1 逐个 y += p_j·down(av_j)，
     * 与串行累加序完全一致（A≡C，不改变浮点结果）。malloc 失败回退原串行循环。 */
    if (ef > 2048) return;
    if (moe_q8_par_ok() && vllm_tp_threads() > 1 && tk > 1) {
        float *avacc = (float *)malloc((size_t)tk * (size_t)ef * sizeof(float));
        if (avacc) {
            moe_q8_par_ctx pc;
            pc.G = G; pc.U = U; pc.x_ffn = x_ffn; pc.av_accum = avacc;
            pc.ef = ef; pc.nbG = nbG; pc.mode8 = mode8; pc.simd = simd;
            pc.ne = ne; pc.tk = tk; pc.sel = sel;
            /* 并行算每专家 gate/up→silu 的 av（独立 slot，无共享累加，
             * 与串行逐位一致）。down 不在此并行——它必须按 j 升序累加进 y，
             * 否则浮点累加序与串行不同（A≡C 破坏）。 */
            vllm_tp_parfor(0, tk, moe_q8_par_worker, &pc);
            /* down：调用线程按 j=0..tk-1 逐个 y += p·down(av_j)（同串行原循环） */
            for (int j = 0; j < tk; j++) {
                int e = sel[j];
                if (e < 0 || e >= ne) continue;
                float p = pr[j];
                int rb = e * ef;
                int cb0 = rb >> 5, cbe = ef >> 5;
                moe_q8_down(D, d, nbD, cb0, cbe, mode8, simd,
                            avacc + (size_t)j * ef, p, y);
            }
            free(avacc);
            return;
        }
        /* malloc 失败：fallthrough 到串行原循环 */
    }
    {
    float gv[2048], uv[2048], av[2048];
    for (int j = 0; j < tk; j++) {
        int e = sel[j];
        if (e < 0 || e >= ne) continue;
        float p = pr[j];
        int rb = e * ef;                             /* 专家行/列带基（8 对齐） */
        moe_q8_gu(G, U, rb, ef, nbG, mode8, simd, x_ffn, gv, uv);
        for (int i = 0; i < ef; i++) {
            float g = gv[i], u = uv[i];
            av[i] = (g / (1.0f + expf(-g))) * u;     /* silu(g)*u */
        }
        int cb0 = rb >> 5;                           /* down 列带起始 32-col 块 */
        int cbe = ef >> 5;                           /* 每专家列块数（ef/32） */
        moe_q8_down(D, d, nbD, cb0, cbe, mode8, simd, av, p, y);
    }
    }
}

/* 计算 y[d] = MoE FFN（router→top-k→softmax→Σ p_e·down_e(silu(gate_e)·up_e)）。
 * x_ffn[d] 为该层 FFN 输入（已 RMSNorm）；不负责残差。 */
/* ---- MoE 专家并行（q4 路径）----
 * A≡C 硬约束同 q8：gate/up→silu 的 av 计算每专家独立（无共享累加），可并行；
 * down 必须由调用线程按 j=0..tk-1 逐个 y += p·down_j 累加（浮点序与串行一致）。
 * VLLM_MOE_PAR=0 关闭（回退逐专家全串行原循环）。gate/up 专家为行区（e*ge4 组，
 * 72B/4行 tile），down 专家为列带（cb0=(e*ef)>>5）。 */
typedef struct {
    const uint8_t *G, *U;
    const float *x_ffn;
    const int8_t *xq;        /* VLLM_ACTQ=1：x_ffn 的 q8_0 激活量化（NULL=关） */
    const float *xd;         /* 每 32 块 scale（nbG 项） */
    const int16_t *xq16;     /* VLLM_ACTQ16=1：s16 激活（NULL=关） */
    const float *xd16;
    float *av_accum;     /* [tk*ef]，每专家激活 av（silu(gate)·up） */
    int ef, nbG, ge4, ggB;
    int ne, tk;
    const int   *sel;
} moe_q4_par_ctx;

static void moe_q4_par_worker(void *ctx_, int j) {
    moe_q4_par_ctx *c = (moe_q4_par_ctx *)ctx_;
    if (j < 0 || j >= c->tk) return;
    int e = c->sel[j];
    if (e < 0 || e >= c->ne) return;
    float gv[2048], uv[2048];
    if (c->ef > 2048) return;
    const uint8_t *ge = c->G + (size_t)((size_t)e * c->ge4) * c->ggB;
    const uint8_t *ue = c->U + (size_t)((size_t)e * c->ge4) * c->ggB;
#if defined(__aarch64__) && defined(ST_HAVE_NEON) && !ST_ARCH_X86 && ST_NEON_DOTPROD
    if (c->xq && c->xd) {
        /* 激活量化 int8-dotprod 近似轨（llama mmq；VLLM_ACTQ=1） */
        for (int g = 0; g < c->ge4; g++)
            moe_q4_gu_group_actq_neon(ge + (size_t)g * c->ggB, ue + (size_t)g * c->ggB,
                                      c->nbG, c->xq, c->xd,
                                      gv + (size_t)g * 4, uv + (size_t)g * 4);
    } else
#endif
#if defined(__AVX2__) && ST_ARCH_X86
    if (c->xq16 && c->xd16) {
        /* s16 激活整数点积轨（VLLM_ACTQ16=1，2026-09-13）。 */
        for (int g = 0; g < c->ge4; g++)
            moe_q4_gu_group_actq16_x86(ge + (size_t)g * c->ggB,
                                       ue + (size_t)g * c->ggB,
                                       c->nbG, c->xq16, c->xd16,
                                       gv + (size_t)g * 4, uv + (size_t)g * 4);
    } else
    if (c->xq && c->xd) {
        /* 激活量化 int8 近似轨（AVX2 maddubs+nib 域；VLLM_ACTQ=1，ARM 同语义）。
         * VLLM_ACTQ_BLK=1：行阻塞（行末归约）变体（M3 项目，A/B）。 */
        void (*gu)(const uint8_t *, const uint8_t *, int, const int8_t *, const float *,
                   float *, float *) =
            actq_blk_env() ? moe_q4_gu_group_actq_blk_x86 : moe_q4_gu_group_actq_x86;
        for (int g = 0; g < c->ge4; g++)
            gu(ge + (size_t)g * c->ggB, ue + (size_t)g * c->ggB,
               c->nbG, c->xq, c->xd, gv + (size_t)g * 4, uv + (size_t)g * 4);
    } else
    if (moe_q4_simd_ok() && (c->ef & 3) == 0) {
        /* SIMD：每 4 行组一个 AVX2 累加（4 lanes=4 行），数值/舍入与标量逐位一致 */
        for (int g = 0; g < c->ge4; g++)
            moe_q4_gu_group_x86(ge + (size_t)g * c->ggB, ue + (size_t)g * c->ggB,
                                c->nbG, c->x_ffn,
                                gv + (size_t)g * 4, uv + (size_t)g * 4);
    } else
#elif defined(__aarch64__) && defined(ST_HAVE_NEON) && !ST_ARCH_X86
    if (moe_q4_simd_ok() && (c->ef & 3) == 0) {
        /* NEON：每 4 行组一个 128-bit 累加（4 lanes=4 行），链序同 x86 → 逐位一致 */
        for (int g = 0; g < c->ge4; g++)
            moe_q4_gu_group_neon(ge + (size_t)g * c->ggB, ue + (size_t)g * c->ggB,
                                 c->nbG, c->x_ffn,
                                 gv + (size_t)g * 4, uv + (size_t)g * 4);
    } else
#endif
    for (int g = 0; g < c->ge4; g++) {
        const uint8_t *tg = ge + (size_t)g * c->ggB;
        const uint8_t *tu = ue + (size_t)g * c->ggB;
        int r0 = g << 2;
        for (int ri = 0; ri < 4; ri++) {
            float ga = 0.0f, ua = 0.0f;
            for (int cb = 0; cb < c->nbG; cb++) {
                const uint8_t *tgb = tg + (size_t)cb * 72;
                const uint8_t *tub = tu + (size_t)cb * 72;
                const float *xb = c->x_ffn + (size_t)cb * 32;
                for (int k = 0; k < 32; k++) {
                    float xv = xb[k];
                    ga += vq4x4_w(tgb, ri, k) * xv;
                    ua += vq4x4_w(tub, ri, k) * xv;
                }
            }
            gv[r0 + ri] = ga;
            uv[r0 + ri] = ua;
        }
    }
    float *av = c->av_accum + (size_t)j * c->ef;
    for (int i = 0; i < c->ef; i++) {
        float g = gv[i], u = uv[i];
        av[i] = (g / (1.0f + expf(-g))) * u;   /* silu(g)*u */
    }
}

#if defined(__AVX2__) && ST_ARCH_X86
/* ---- M7（b-立项 相位1，2026-09-10）：q4 MoE ffn decode 调度融合（A/B）----
 * 现状 par 路径：gu = vllm_tp_parfor(0,tk) 仅 8 个专家级粗任务；down = 专家 j
 * 外层串行 × 每次全池 vllm_tp_parfor(512 行组) = 8 次全池 join，且每次仅流
 * ~786KB（dk 短流 + 8× 同步）。M7：gu 融成单池细分区（tk*ge4 行组一次 parfor，
 * worker 内 silu 立即写 avacc）；down 融成单池 512 行组一次 parfor，worker 内
 * 按专家序 j 升序逐专家 RMW y（与现逐专家 parfor 的行内浮点累加序逐位一致 →
 * f32 精确轨 A≡C 保留；actq 近似轨同序确定性）。行内点积核零改动。
 * VLLM_MOE_FUSE=1 启用（默认关）；注意 M7 gu 用非 blk actq 组核（隔离调度变量，
 * VLLM_ACTQ_BLK 属另一内核变量，两者不同时混用）。全 x86 门控。 */
static int moe_fuse_env(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("VLLM_MOE_FUSE"); v = (e && e[0] == '1') ? 1 : 0; }
    return v;
}

typedef struct {
    const uint8_t *G, *U;
    const float *xf; const int8_t *xq; const float *xd;
    int nbG, ge4, ggB, ne, tk, ef;
    const int *sel;
    float *avacc;
    int actq;
} m7_gu_ctx;
static void m7_gu_worker(void *ctx_, int n) {
    m7_gu_ctx *c = ctx_;
    int j = n / c->ge4;                       /* 行组全局序号 → 专家 j */
    int g = n - j * c->ge4;                   /* 组内序号 g（4 行） */
    if (j < 0 || j >= c->tk || g < 0 || g >= c->ge4) return;
    int e = c->sel[j];
    if (e < 0 || e >= c->ne) return;
    const uint8_t *ge = c->G + (size_t)((size_t)e * c->ge4) * c->ggB;
    const uint8_t *ue = c->U + (size_t)((size_t)e * c->ge4) * c->ggB;
    const uint8_t *tg = ge + (size_t)g * c->ggB;
    const uint8_t *tu = ue + (size_t)g * c->ggB;
    float gv[4], uv[4];
    if (c->actq)
        moe_q4_gu_group_actq_x86(tg, tu, c->nbG, c->xq, c->xd, gv, uv);
    else
        moe_q4_gu_group_x86(tg, tu, c->nbG, c->xf, gv, uv);
    float *av = c->avacc + (size_t)j * c->ef + (size_t)g * 4;
    for (int i = 0; i < 4; i++) {
        float ggv = gv[i], uuv = uv[i];
        av[i] = (ggv / (1.0f + expf(-ggv))) * uuv;   /* silu(g)·u：同原路径 */
    }
}

typedef struct {
    const uint8_t *D; float *y;
    const float *avf; const int8_t *avq; const float *avd;  /* f32/actq 轨 */
    const float *pr;
    int cbe, gdB, ef, ne, tk, actq;
    const int *sel;
} m7_dn_ctx;
static void m7_dn_worker(void *ctx_, int rg) {
    m7_dn_ctx *c = ctx_;
    for (int j = 0; j < c->tk; j++) {
        int e = c->sel[j];
        if (e < 0 || e >= c->ne) continue;
        const uint8_t *db = c->D + (size_t)rg * (size_t)c->gdB
                                  + (size_t)((e * c->ef) >> 5) * 72;
        float p = c->pr[j];
        float *yp = c->y + (size_t)rg * 4;
        if (c->actq)
            moe_q4_down_group_actq_x86(db, c->cbe,
                                       c->avq + (size_t)j * c->ef,
                                       c->avd + (size_t)j * c->cbe,
                                       p, yp);
        else
            moe_q4_down_group_x86(db, c->cbe, c->avf + (size_t)j * c->ef,
                                  p, yp);
    }
}
#endif /* __AVX2__ && ST_ARCH_X86 */

/* ---- M-D2（VLLM_DN_D2=1）：down 专家连续布局（架构无关搬移，ARM 亦启用）----
 * RDPRF 已证：dn 现"行组-major × 专家列带"(512 组 × 步长 221184B) 纯读仅
 * 25-31GB/s（~1.7× 布局税 vs gu 专家连续 45-53GB/s）。D2 = 运行期把每层 down
 * 重排为专家-major：D2[l][e][rg 连续 1728B×512]（每专家 884736B 连续），读取
 * 形态与 gu 相同 → 预期 dn 读回到 ~45GB/s 级。字节级 memcpy 复制、行内点积序
 * 不变 → f32/actq 双轨 A≡C 逐位不变。 */
static int dn_d2_env(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("VLLM_DN_D2"); v = (e && e[0] == '1') ? 1 : 0; }
    return v;
}
static uint8_t *g_dn_d2 = NULL;
static int g_dn_d2_n = 0;          /* 每 q4 层 D2 字节 */
static int g_dn_d2_off = 0;        /* 每专家字节（e 连续区） */
static int g_dn_d2_stride = 0;     /* 每行组 1728B（专家内 rg 步长） */
static int dn_d2_ready(void) { return g_dn_d2 != NULL; }
static void dn_d2_build(const STModelWeights *w, int off_l) {
    if (g_dn_d2) return;
    if (!w->q4_down_weight) return;
    if (w->has_q2 && w->q2_gate_weight) return;   /* 本 A/B 仅纯 q4 全层模型 */
    const STModelConfig *c = &w->cfg;
    int ne = c->n_experts, d = c->dim, ff = c->ffn_dim;
    if (ne <= 0 || d <= 0 || ff <= 0 || (ff % ne)) return;
    int ef = ff / ne;
    int S = (ef >> 5) * 72;                 /* 每行组每专家 1728B */
    int E = S * (d >> 2);                   /* 每专家 884736B */
    int per = (size_t)E * ne;               /* 每层 113MB */
    if (S <= 0 || E <= 0 || per <= 0) return;
    int nlay = w->n_layers_allocated;
    if (nlay <= 0 || nlay > 4096) return;
    size_t tot = (size_t)per * nlay;
    uint8_t *buf = (uint8_t *)malloc(tot);
    if (!buf) return;
    /* 逐层逐行组搬移：源 = 行组-major（带内 e 条带连续），目标 = e-major */
    for (int l = 0; l < nlay; l++) {
        const uint8_t *Dl = w->q4_down_weight +
                            Q4_BYTES((size_t)st_ffn_q4_layer_off(w, l) * (size_t)d * (size_t)ff);
        uint8_t *dst = buf + (size_t)l * per;
        for (int rg = 0; rg < (d >> 2); rg++) {
            const uint8_t *src = Dl + (size_t)rg * (size_t)((ff >> 5) * 72);
            for (int e = 0; e < ne; e++)
                memcpy(dst + (size_t)e * E + (size_t)rg * S, src + (size_t)e * S, (size_t)S);
        }
    }
    g_dn_d2 = buf; g_dn_d2_n = per; g_dn_d2_off = E; g_dn_d2_stride = S;
    printf("[D2] down expert-major repack ready: %zuMB (%d layers x %dMB)\n",
           tot / 1048576ULL, nlay, per / 1048576);
    fflush(stdout);
    (void)off_l;
}
#if defined(__AVX2__) && ST_ARCH_X86
typedef struct {
    const uint8_t *base;   /* D2[l] + e*E */
    float *y;
    int cbe, rgmax, S;
    const int8_t *avq; const float *avd; float p;
} m4_dn2_actq_ctx;
static void m4_dn2_actq_worker(void *ctx_, int rg) {
    m4_dn2_actq_ctx *c = ctx_;
    const uint8_t *db = c->base + (size_t)rg * (size_t)c->S;
    if (rg + 8 < c->rgmax) ST_PREFETCH(db + (size_t)8 * (size_t)c->S);
    moe_q4_down_group_actq_x86(db, c->cbe, c->avq, c->avd, c->p,
                               c->y + (size_t)rg * 4);
}
#endif /* __AVX2__ && ST_ARCH_X86 */

#if defined(__AVX2__) && ST_ARCH_X86
/* ---- M-RDPRF（b 相位2 判读探针，2026-09-10，VLLM_MOE_RDPRF=1）----
 * ffn ~40ms 在 M7(调度融合)/BLK(行末归约核) 双 null 后指向"访问模式/内存系统"。
 * 本探针以真实 sel 足迹做纯读（无 FP、无点积核）：gu = 每专家 G/U 连续条带
 * （行连续 884736B ×2），down = 现布局散列列带（512 行组 × 每专家 24×72B，
 * 步长 gdB=221184B）——两类可达 GB/s 之比即 down 布局散列税；与 llama 每专家
 * 连续 884KB 形态对照。测完 return（ffn 替换为纯读，不写 y）。全 x86 门控。 */
typedef struct {
    const uint8_t *G, *U;
    int ge4, ggB, ne, tk;
    const int *sel;
    volatile uint64_t *sink;
} rdp_gu_ctx;
static void rdp_gu_worker(void *c_, int j) {
    rdp_gu_ctx *c = c_;
    if (j < 0 || j >= c->tk) return;
    int e = c->sel[j];
    if (e < 0 || e >= c->ne) return;
    const uint8_t *ge = c->G + (size_t)((size_t)e * c->ge4) * c->ggB;
    const uint8_t *ue = c->U + (size_t)((size_t)e * c->ge4) * c->ggB;
    size_t n = (size_t)c->ge4 * (size_t)c->ggB / 8;
    uint64_t s[8] = {0, 0, 0, 0, 0, 0, 0, 0};   /* 8 独立累加器破 XOR 链，测真 MLP/带宽 */
    const uint64_t *pg = (const uint64_t *)ge, *pu = (const uint64_t *)ue;
    size_t i = 0;
    for (; i + 7 < n; i += 8) {
        s[0] ^= pg[i]; s[1] ^= pg[i + 1]; s[2] ^= pg[i + 2]; s[3] ^= pg[i + 3];
        s[4] ^= pg[i + 4]; s[5] ^= pg[i + 5]; s[6] ^= pg[i + 6]; s[7] ^= pg[i + 7];
    }
    for (; i < n; i++) s[0] ^= pg[i];
    i = 0;
    for (; i + 7 < n; i += 8) {
        s[0] ^= pu[i]; s[1] ^= pu[i + 1]; s[2] ^= pu[i + 2]; s[3] ^= pu[i + 3];
        s[4] ^= pu[i + 4]; s[5] ^= pu[i + 5]; s[6] ^= pu[i + 6]; s[7] ^= pu[i + 7];
    }
    for (; i < n; i++) s[0] ^= pu[i];
    c->sink[j] += s[0] ^ s[1] ^ s[2] ^ s[3] ^ s[4] ^ s[5] ^ s[6] ^ s[7];
}
typedef struct {
    const uint8_t *D;
    int gdB, cbe, ef, ne, tk;
    const int *sel;
    volatile uint64_t *sink;
} rdp_dn_ctx;
static void rdp_dn_worker(void *c_, int rg) {
    rdp_dn_ctx *c = c_;
    uint64_t s[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int t = 0;
    for (int j = 0; j < c->tk; j++) {
        int e = c->sel[j];
        if (e < 0 || e >= c->ne) continue;
        const uint8_t *p = c->D + (size_t)rg * (size_t)c->gdB
                                + (size_t)(((size_t)e * c->ef) >> 5) * 72;
        for (int b = 0; b < c->cbe; b++) {
            const uint64_t *q = (const uint64_t *)(p + (size_t)b * 72);
            for (int k = 0; k < 9; k++) { s[t & 7] ^= q[k]; t++; }
        }
    }
    c->sink[rg] += s[0] ^ s[1] ^ s[2] ^ s[3] ^ s[4] ^ s[5] ^ s[6] ^ s[7];
}
static int moe_rdprf_env(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("VLLM_MOE_RDPRF"); v = (e && e[0] == '1') ? 1 : 0; }
    return v;
}
#endif /* __AVX2__ && ST_ARCH_X86 */

#if defined(__aarch64__) && defined(ST_HAVE_NEON) && !ST_ARCH_X86
/* M5-ARM（2026-09-10，镜像 x86 M5）：q4 MoE down 行组并行。
 * 512 个 4 行组静态分块；每行组单线程写 y[rg*4..+3]，专家 j 外层序不变 →
 * 每行浮点累加序与串行逐位一致（A≡C）。行内点积序不变。 */
typedef struct {
    const uint8_t *D; float *y;
    int cb0, cbe, gdB, rgmax;
    const int8_t *avq; const float *avd; float p;
    const uint8_t *d2base; int d2_stride;   /* D2 专家连续（NULL=散列原布局） */
} arm_dn_actq_ctx;
static void arm_dn_actq_worker(void *ctx_, int rg) {
    arm_dn_actq_ctx *c = ctx_;
    const uint8_t *db = c->d2base
        ? c->d2base + (size_t)rg * (size_t)c->d2_stride
        : c->D + (size_t)rg * (size_t)c->gdB + (size_t)c->cb0 * 72;
    moe_q4_down_group_actq_neon(db, c->cbe, c->avq, c->avd, c->p,
                                c->y + (size_t)rg * 4);
}
/* f32 精确轨 down 行组并行 worker：与 arm_dn_actq_worker 同构，但走
 * moe_q4_down_group_neon（f32 乘加）。每个 rg 独立写 y[rg*4..+3]，
 * j 升序累加序由调用方 for j 保证 → A≡C 位级一致。 */
typedef struct {
    const uint8_t *D; float *y;
    int cb0, cbe, gdB, rgmax;
    const float *av; float p;
    const uint8_t *d2base; int d2_stride;   /* D2 专家连续（NULL=散列原布局） */
} arm_dn_f32_ctx;
static void arm_dn_f32_worker(void *ctx_, int rg) {
    arm_dn_f32_ctx *c = ctx_;
    const uint8_t *db = c->d2base
        ? c->d2base + (size_t)rg * (size_t)c->d2_stride
        : c->D + (size_t)rg * (size_t)c->gdB + (size_t)c->cb0 * 72;
    moe_q4_down_group_neon(db, c->cbe, c->av, c->p, c->y + (size_t)rg * 4);
}
#endif /* __aarch64__ && ST_HAVE_NEON && !ST_ARCH_X86 */

static void st_moe_ffn_sparse_q4(const STModelWeights *w, int l,
                                 const float *x_ffn, float *y) {
    const STModelConfig *c = &w->cfg;
    int prf = moe_prof_enabled();
    double tff0 = 0, td0 = 0;
    if (prf) {
        if (l == 0 && mpt_n > 0) {   /* 上一 token 完整 48 层后于新 token 起始打印 */
            printf("[MOEPROF] tok%d ffn=%.3fs gu=%.3fs dn=%.3fs\n",
                   mpt_n / 48, mpt_ff, mpt_gu, mpt_dn);
            fflush(stdout);
            mpt_ff = mpt_gu = mpt_dn = 0;
            mpt_n = 0;
        }
        mpt_n++;
        tff0 = vllm_tp_wtime();
    }
    int ne = c->n_experts, tk = c->top_k, d = c->dim, ff = c->ffn_dim;
    if (ne <= 0 || tk <= 0 || tk > 64 || ne > 4096 || d <= 0) return;
    int ef = ff / ne;
    if (ef <= 0 || ne * ef != ff || (ef & 3) || (d & 31)) return;
    const float *router = w->moe_router + (size_t)l * ne * d;
    int off_l = st_ffn_q4_layer_off(w, l);
    const uint8_t *G = w->q4_gate_weight +
                       Q4_BYTES((size_t)off_l * (size_t)ff * d);
    const uint8_t *U = w->q4_up_weight +
                       Q4_BYTES((size_t)off_l * (size_t)ff * d);
    const uint8_t *D = w->q4_down_weight +
                       Q4_BYTES((size_t)off_l * (size_t)d * ff);
    if (!G || !U || !D) return;
    if (dn_d2_env() && !dn_d2_ready()) dn_d2_build(w, off_l);   /* D2 一次性重排 */
    int nbG = d >> 5;
    int nbD = ff >> 5;
    int ggB = nbG * 72;
    int gdB = nbD * 72;
    int ge4 = ef >> 2;
    int sel[64]; float pr[64];
    (void)nbD;

    /* 1) router logits + 确定性 top-k（同分取小索引） */
    {
    float lg[4096];
    for (int e = 0; e < ne; e++) {
        const float *rw = router + (size_t)e * d;
        float s = 0.0f;
        for (int i = 0; i < d; i++) s += rw[i] * x_ffn[i];
        lg[e] = s;
    }
    float sv[64];
    for (int k = 0; k < tk; k++) { sel[k] = -1; sv[k] = -INFINITY; }
    for (int e = 0; e < ne; e++) {
        for (int k = 0; k < tk; k++) {
            if (lg[e] > sv[k] || (lg[e] == sv[k] && (sel[k] < 0 || e < sel[k]))) {
                for (int j = tk - 1; j > k; j--) { sel[j] = sel[j - 1]; sv[j] = sv[j - 1]; }
                sel[k] = e; sv[k] = lg[e];
                break;
            }
        }
    }
    float mx = -INFINITY;
    for (int k = 0; k < tk; k++) if (sv[k] > mx) mx = sv[k];
    float sum = 0.0f;
    for (int k = 0; k < tk; k++) { pr[k] = expf(sv[k] - mx); sum += pr[k]; }
    float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    for (int k = 0; k < tk; k++) pr[k] *= inv;
    }

#if defined(__AVX2__) && ST_ARCH_X86
    /* M-RDPRF 纯读探针：真实 sel 足迹分测 gu(连续)/dn(散列列带) 可达 GB/s。
     * 以真实每层 G/U/D 区 + router 为准，纯读无 FP；测完即 return（不写 y）。 */
    if (moe_rdprf_env() && ge4 > 0) {
        static double rs_gu = 0, rs_dn = 0;
        static unsigned long long rb_gu = 0, rb_dn = 0;
        static int rn = 0;
        uint64_t sink[2048];
        memset(sink, 0, sizeof(sink));
        (void)rb_gu; (void)rb_dn;
        double t0 = vllm_tp_wtime();
        uint64_t rsum = 0;
        for (int e = 0; e < ne; e++) {
            const float *rw = router + (size_t)e * d;
            for (int i = 0; i < d; i += 4)
                rsum += *(const volatile uint64_t *)&rw[i];   /* 防优化读 */
        }
        rs_gu += vllm_tp_wtime() - t0;
        rb_gu += (unsigned long long)ne * (unsigned long long)d * 4u;
        double tg = vllm_tp_wtime();
        rdp_gu_ctx gc2;
        gc2.G = G; gc2.U = U; gc2.ge4 = ge4; gc2.ggB = ggB;
        gc2.ne = ne; gc2.tk = tk; gc2.sel = sel; gc2.sink = sink;
        vllm_tp_parfor(0, tk, rdp_gu_worker, &gc2);
        rs_gu += vllm_tp_wtime() - tg;
        rb_gu += (unsigned long long)tk * 2u * (size_t)ge4 * (size_t)ggB;
        double td = vllm_tp_wtime();
        rdp_dn_ctx dc2;
        dc2.D = D; dc2.gdB = gdB; dc2.cbe = ef >> 5; dc2.ef = ef;
        dc2.ne = ne; dc2.tk = tk; dc2.sel = sel; dc2.sink = sink;
        vllm_tp_parfor(0, d >> 2, rdp_dn_worker, &dc2);
        rs_dn += vllm_tp_wtime() - td;
        rb_dn += (unsigned long long)tk * (size_t)(d >> 2) * (size_t)(ef >> 5) * 72u;
        rn++;
        if (l == 0 && rn >= 48) {
            printf("[RDPRF] tok gu=%.1fGB/s(%lluMB/%.1fms) dn=%.1fGB/s(%lluMB/%.1fms)\n",
                   (double)rb_gu / 1e6 / rs_gu, rb_gu / 1048576ULL, rs_gu * 1e3,
                   (double)rb_dn / 1e6 / rs_dn, rb_dn / 1048576ULL, rs_dn * 1e3);
            fflush(stdout);
            rs_gu = rs_dn = 0; rb_gu = rb_dn = 0; rn = 0;
        }
        (void)rsum;
        return;
    }
#endif

    /* 输出清零（并行 slot 归约后写入 y；串行路径也以清零 y 为累加起点） */
    memset(y, 0, (size_t)d * sizeof(float));

    if (ef <= 0 || ne * ef != ff || (ef & 3) || (d & 31) || ef > 2048) return;
    if (l == 0 && getenv("VLLM_MOE_DUMP") && getenv("VLLM_MOE_DUMP")[0] == '1') {
        printf("[L0D] S9_router");
        for (int k = 0; k < tk; k++) printf(" %d:%.6e", sel[k], (double)pr[k]);
        printf("\n");
        fflush(stdout);
    }
    /* 真实 router 热度采集（q4 路径，EW 标定 H 用） */
    if (g_moe_heat_on && g_moe_heat_ne > 0) {
        uint32_t *h = g_moe_heat + (size_t)l * (size_t)g_moe_heat_ne;
        for (int j = 0; j < tk; j++)
            if (sel[j] >= 0 && sel[j] < g_moe_heat_ne) h[sel[j]]++;
    }

    /* 选路后专家预取（VLLM_EW_PREFETCH=1，仅 VQF q4）：top-k 一确定就对本层
     * 选中专家的 gate/up 段 + 本层 down 列带发 WILLNEED，缺页重驻与下方并行
     * gu 计算重叠（§10.22：gu 已由计算型转为缺页/带宽主导）。只改页表/页缓存、
     * 浮点序零改动 → A≡C 位级一致；q8/非 VQF/几何不匹配时 vqf_ffn_prefetch
     * 内部自动 no-op。 */
    vqf_ffn_prefetch(w, l, sel, tk);

    /* 激活量化双轨（VLLM_ACTQ=1，近似 int8-dotprod，见内核区注释）：把
     * x_ffn（d=2048→64 块）量化为 q8_0，全部专家/gate/up 行共享。数值上
     * 与精确轨（moe_q4_gu_group_neon）不逐位一致 → 默认关、锚点级验证。 */
    int8_t xq[2048]; float xd[64];
    int16_t xq16[2048]; float xd16[64];
    int actq16 = actq16_env();
    int actq = moe_q4_actq_ok();
    if (actq16 && (d > 2048 || nbG > 64)) actq16 = 0;   /* 小栈缓冲上限 */
    if (actq16) actq = 0;                                /* 互斥：s16 轨优先 */
    if (actq && (d > 2048 || nbG > 64)) actq = 0;   /* 小栈缓冲上限 */
    if (actq16) {
        /* s16 激活整数点积轨（VLLM_ACTQ16=1）：与 ACTQ 共用 actq_msk 行收集掩码。 */
        static int actq16_logged = 0;
        if (!actq16_logged) {
            actq16_logged = 1;
            printf("[ACTQ16] VLLM_ACTQ16=1: q4 MoE s16-dot track ON "
                   "(l=0, d=%d nbG=%d)\n", d, nbG);
            fflush(stdout);
        }
        quantize_row_s16_act(x_ffn, xq16, xd16, d);
#if defined(__AVX2__) && ST_ARCH_X86
        actq_tab_ensure_x86();
#endif
    } else if (actq) {
        static int actq_logged = 0;
        if (!actq_logged) {
            actq_logged = 1;
            printf("[ACTQ] VLLM_ACTQ=1: q4 MoE int8-dot approximate track ON "
                   "(l=0, d=%d nbG=%d)\n", d, nbG);
            fflush(stdout);
        }
        quantize_row_q8_0_act(x_ffn, xq, xd, d);
#if defined(__aarch64__) && defined(ST_HAVE_NEON) && !ST_ARCH_X86 && ST_NEON_DOTPROD
        actq_tab_ensure();   /* 主线程先行建表，避免 worker 并发重复初始化 */
#elif defined(__AVX2__) && ST_ARCH_X86
        actq_tab_ensure_x86();
#endif
    }

    /* 2) 专家并行（默认开，VLLM_MOE_PAR=0 回退串行原循环）：
     *    gate/up→silu 的 av 并行算进独立 slot（无共享累加，与串行逐位一致）；
     *    down 由调用线程按 j 升序逐个累加进 y —— 与串行 y(0)→+=down_0→… 同序，
     *    A≡C 位级一致（down 不进并行，浮点累加序不变）。 */
    if (moe_q8_par_ok() && vllm_tp_threads() > 1 && tk > 1) {
        float *avacc = (float *)malloc((size_t)tk * (size_t)ef * sizeof(float));
        if (avacc) {
            moe_q4_par_ctx pc;
            pc.G = G; pc.U = U; pc.x_ffn = x_ffn; pc.av_accum = avacc;
            pc.xq = actq ? xq : NULL; pc.xd = actq ? xd : NULL;
            pc.xq16 = actq16 ? xq16 : NULL; pc.xd16 = actq16 ? xd16 : NULL;
            pc.ef = ef; pc.nbG = nbG; pc.ge4 = ge4; pc.ggB = ggB;
            pc.ne = ne; pc.tk = tk; pc.sel = sel;
#if defined(__AVX2__) && ST_ARCH_X86
            /* M7（b-立项 相位1）调度融合 A/B（VLLM_MOE_FUSE=1，默认关）：
             * 见上方 m7_* worker 注释。gu/down 各一次单池 parfor；行内浮点
             * 累加序与下方案件逐位一致 → f32/actq 轨均 A≡C（值不变，仅调度）。 */
            {
                int fuse_ok = 0;
                int8_t *avq_all = NULL; float *avd_all = NULL;
                if (moe_fuse_env() && !actq16 && ge4 > 0 && (ef & 31) == 0) {
                    if (actq) {
                        avq_all = (int8_t *)malloc((size_t)tk * (size_t)ef);
                        avd_all = (float *)malloc((size_t)tk * (size_t)(ef >> 5)
                                                  * sizeof(float));
                        if (avq_all && avd_all) fuse_ok = 1;
                        else { free(avq_all); free(avd_all); avq_all = NULL; avd_all = NULL; }
                    } else if (moe_q4_simd_ok()) {
                        fuse_ok = 1;   /* f32 精确轨：组核 gu_group_x86/down_group_x86 */
                    }
                }
                if (fuse_ok) {
                    double tg7 = prf ? vllm_tp_wtime() : 0;
                    m7_gu_ctx gc;
                    gc.G = G; gc.U = U; gc.xf = x_ffn;
                    gc.xq = actq ? xq : NULL; gc.xd = actq ? xd : NULL;
                    gc.nbG = nbG; gc.ge4 = ge4; gc.ggB = ggB; gc.ne = ne;
                    gc.tk = tk; gc.ef = ef; gc.sel = sel;
                    gc.avacc = avacc; gc.actq = actq;
                    vllm_tp_parfor(0, tk * ge4, m7_gu_worker, &gc);
                    if (prf) mpt_gu += vllm_tp_wtime() - tg7;
                    double tdn = prf ? vllm_tp_wtime() : 0;
                    int cbe = ef >> 5;
                    if (actq) {
                        for (int j = 0; j < tk; j++)
                            quantize_row_q8_0_act(avacc + (size_t)j * ef,
                                                  avq_all + (size_t)j * ef,
                                                  avd_all + (size_t)j * cbe, ef);
                        m7_dn_ctx dc;
                        dc.D = D; dc.y = y; dc.avf = NULL; dc.avq = avq_all;
                        dc.avd = avd_all; dc.pr = pr; dc.cbe = cbe;
                        dc.gdB = gdB; dc.ef = ef; dc.ne = ne; dc.tk = tk;
                        dc.actq = 1; dc.sel = sel;
                        vllm_tp_parfor(0, d >> 2, m7_dn_worker, &dc);
                    } else {
                        m7_dn_ctx dc;
                        dc.D = D; dc.y = y; dc.avf = avacc; dc.avq = NULL;
                        dc.avd = NULL; dc.pr = pr; dc.cbe = cbe;
                        dc.gdB = gdB; dc.ef = ef; dc.ne = ne; dc.tk = tk;
                        dc.actq = 0; dc.sel = sel;
                        vllm_tp_parfor(0, d >> 2, m7_dn_worker, &dc);
                    }
                    if (prf) { mpt_dn += vllm_tp_wtime() - tdn;
                               mpt_ff += vllm_tp_wtime() - tff0; }
                    free(avq_all); free(avd_all);
                    free(avacc);
                    return;
                }
                free(avq_all); free(avd_all);
            }
#endif
            double tg0 = prf ? vllm_tp_wtime() : 0;
            vllm_tp_parfor(0, tk, moe_q4_par_worker, &pc);
            if (prf) mpt_gu += vllm_tp_wtime() - tg0;
            /* down：调用线程按 j=0..tk-1 逐个 y += p·down(av_j)（同串行原循环） */
            td0 = prf ? vllm_tp_wtime() : 0;
            for (int j = 0; j < tk; j++) {
                int e = sel[j];
                if (e < 0 || e >= ne) continue;
                float p = pr[j];
                const float *av = avacc + (size_t)j * ef;
                int cb0 = (e * ef) >> 5;             /* down 列带起始 32-col 块 */
                int cbe = (ef >> 5);                 /* 每专家列块数（ef/32） */
                const uint8_t *d2base = NULL;        /* D2 专家连续（NULL=散列原布局） */
                if (dn_d2_ready() && g_dn_d2_stride == (ef >> 5) * 72
                    && e >= 0 && e < (g_dn_d2_n / g_dn_d2_off))
                    d2base = g_dn_d2 + (size_t)off_l * (size_t)g_dn_d2_n
                                        + (size_t)e * (size_t)g_dn_d2_off;
#if defined(__AVX2__) && ST_ARCH_X86
                if (actq16 && ef <= 2048) {
                    /* down s16 轨（VLLM_ACTQ16=1）：av 每专家量化一次。 */
                    int16_t avq16[2048]; float avd16[64];
                    quantize_row_s16_act(av, avq16, avd16, ef);
                    m4_dn_actq16_ctx dc16 = { D, y, cb0, cbe, gdB, d >> 2,
                                              avq16, avd16, p };
                    vllm_tp_parfor(0, d >> 2, m4_dn_actq16_worker, &dc16);
                } else
                if (actq && ef <= 2048) {
                    /* down 近似轨：av 每专家量化一次（ef≤2048 → ≤64 块）。
                     * D2（VLLM_DN_D2=1）：走专家连续副池（每专家 884736B 连续），
                     * 消除行组-major 列带散列读税（RDPRF: 25-31 → ~45GB/s 级）。 */
                    int8_t avq[2048]; float avd[64];
                    quantize_row_q8_0_act(av, avq, avd, ef);
                    if (dn_d2_ready() && g_dn_d2_stride == (ef >> 5) * 72
                        && e >= 0 && e < (g_dn_d2_n / g_dn_d2_off)) {
                        const uint8_t *b2 = g_dn_d2 + (size_t)off_l * (size_t)g_dn_d2_n
                                                    + (size_t)e * (size_t)g_dn_d2_off;
                        m4_dn2_actq_ctx dc2 = { b2, y, cbe, d >> 2, g_dn_d2_stride,
                                                avq, avd, p };
                        vllm_tp_parfor(0, d >> 2, m4_dn2_actq_worker, &dc2);
                    } else {
                        m4_dn_actq_ctx dc = { D, y, cb0, cbe, gdB, d >> 2, avq, avd, p };
                        vllm_tp_parfor(0, d >> 2, m4_dn_actq_worker, &dc);
                    }
                } else
                if (moe_q4_simd_ok() && (d & 3) == 0) {
                    /* down：每 4 输出行组一个 AVX2 累加（4 lanes=4 行） */
                    m4_dn_f32_ctx dc = { D, y, av, cb0, cbe, gdB, d >> 2, p };
                    vllm_tp_parfor(0, d >> 2, m4_dn_f32_worker, &dc);
                } else
#elif defined(__aarch64__) && defined(ST_HAVE_NEON) && !ST_ARCH_X86
                if (actq && ef <= 2048) {
                    /* down 近似轨：av 每专家量化一次（ef≤2048 → ≤64 块） */
                    int8_t avq[2048]; float avd[64];
                    quantize_row_q8_0_act(av, avq, avd, ef);
                    arm_dn_actq_ctx dc = { D, y, cb0, cbe, gdB, d >> 2, avq, avd, p,
                                           d2base, g_dn_d2_stride };
                    vllm_tp_parfor(0, d >> 2, arm_dn_actq_worker, &dc);
                } else
                if (moe_q4_simd_ok() && (d & 3) == 0) {
                    /* down：每 4 输出行组一个 NEON 128-bit 累加（链序同 x86）。
                     * 行组并行（parfor rg）：每 rg 独立写 y[rg*4..+3]，j 升序由
                     * 外层 for j 保证 → A≡C 位级一致。 */
                    arm_dn_f32_ctx dc = { D, y, cb0, cbe, gdB, d >> 2, av, p,
                                          d2base, g_dn_d2_stride };
                    vllm_tp_parfor(0, d >> 2, arm_dn_f32_worker, &dc);
                } else
#endif
                for (int rg = 0; rg < (d >> 2); rg++) {  /* down 输出行按 4 行组 */
                    const uint8_t *db = D + (size_t)rg * gdB + (size_t)cb0 * 72;
                    for (int ri = 0; ri < 4; ri++) {
                        float acc = 0.0f;
                        const float *ae = av;
                        for (int cbi = 0; cbi < cbe; cbi++) {
                            const uint8_t *tile = db + (size_t)cbi * 72;
                            for (int k = 0; k < 32; k++, ae++)
                                acc += vq4x4_w(tile, ri, k) * (*ae);
                        }
                        y[(size_t)rg * 4 + ri] += p * acc;
                    }
                }
            }
            if (prf) { mpt_dn += vllm_tp_wtime() - td0; mpt_ff += vllm_tp_wtime() - tff0; }
            free(avacc);
            return;
        }
        /* malloc 失败：fallthrough 到串行原循环 */
    }
    td0 = prf ? vllm_tp_wtime() : 0;
    for (int j = 0; j < tk; j++) {
        int e = sel[j];
        if (e < 0 || e >= ne) continue;
        float p = pr[j];
        const uint8_t *ge = G + (size_t)((size_t)e * ge4) * ggB;
        const uint8_t *ue = U + (size_t)((size_t)e * ge4) * ggB;
        float gv[2048], uv[2048], av[2048];   /* ef ≤ 2048（防御上限） */
        /* gate & up：逐 4 行组读 tile（4 行共享 72B tile） */
#if defined(__AVX2__) && ST_ARCH_X86
        if (actq16) {
            /* s16 整数点积轨（VLLM_ACTQ16=1）。 */
            for (int g = 0; g < ge4; g++)
                moe_q4_gu_group_actq16_x86(ge + (size_t)g * ggB,
                                           ue + (size_t)g * ggB,
                                           nbG, xq16, xd16,
                                           gv + (size_t)g * 4, uv + (size_t)g * 4);
        } else
        if (actq) {
            /* 激活量化 int8 近似轨（AVX2 maddubs+nib 域；VLLM_ACTQ=1）。
             * VLLM_ACTQ_BLK=1：行阻塞（行末归约）变体（M3 项目，A/B）。 */
            void (*gu)(const uint8_t *, const uint8_t *, int, const int8_t *, const float *,
                       float *, float *) =
                actq_blk_env() ? moe_q4_gu_group_actq_blk_x86 : moe_q4_gu_group_actq_x86;
            for (int g = 0; g < ge4; g++)
                gu(ge + (size_t)g * ggB, ue + (size_t)g * ggB,
                   nbG, xq, xd, gv + (size_t)g * 4, uv + (size_t)g * 4);
        } else
        if (moe_q4_simd_ok() && (ef & 3) == 0) {
            for (int g = 0; g < ge4; g++)
                moe_q4_gu_group_x86(ge + (size_t)g * ggB, ue + (size_t)g * ggB,
                                    nbG, x_ffn, gv + (size_t)g * 4, uv + (size_t)g * 4);
        } else
#elif defined(__aarch64__) && defined(ST_HAVE_NEON) && !ST_ARCH_X86
        if (actq) {
            /* 激活量化 int8-dotprod 近似轨（VLLM_ACTQ=1） */
            for (int g = 0; g < ge4; g++)
                moe_q4_gu_group_actq_neon(ge + (size_t)g * ggB, ue + (size_t)g * ggB,
                                          nbG, xq, xd,
                                          gv + (size_t)g * 4, uv + (size_t)g * 4);
        } else
        if (moe_q4_simd_ok() && (ef & 3) == 0) {
            for (int g = 0; g < ge4; g++)
                moe_q4_gu_group_neon(ge + (size_t)g * ggB, ue + (size_t)g * ggB,
                                     nbG, x_ffn, gv + (size_t)g * 4, uv + (size_t)g * 4);
        } else
#endif
        for (int g = 0; g < ge4; g++) {
            const uint8_t *tg = ge + (size_t)g * ggB;
            const uint8_t *tu = ue + (size_t)g * ggB;
            int r0 = g << 2;
            for (int ri = 0; ri < 4; ri++) {
                float ga = 0.0f, ua = 0.0f;
                for (int cb = 0; cb < nbG; cb++) {
                    const uint8_t *tgb = tg + (size_t)cb * 72;
                    const uint8_t *tub = tu + (size_t)cb * 72;
                    const float *xb = x_ffn + (size_t)cb * 32;
                    for (int k = 0; k < 32; k++) {
                        float xv = xb[k];
                        ga += vq4x4_w(tgb, ri, k) * xv;
                        ua += vq4x4_w(tub, ri, k) * xv;
                    }
                }
                gv[r0 + ri] = ga;
                uv[r0 + ri] = ua;
            }
        }
        /* 3) activated = silu(gate)*up，随后 down 加权累加 */
        for (int i = 0; i < ef; i++) {
            float g = gv[i], u = uv[i];
            av[i] = (g / (1.0f + expf(-g))) * u;   /* silu(g)*u */
        }
        int cb0 = (e * ef) >> 5;                 /* down 列带起始 32-col 块 */
        int cbe = (ef >> 5);                     /* 每专家列块数（ef/32） */
        const uint8_t *d2base = NULL;            /* D2 专家连续（NULL=散列原布局） */
        if (dn_d2_ready() && g_dn_d2_stride == (ef >> 5) * 72
            && e >= 0 && e < (g_dn_d2_n / g_dn_d2_off))
            d2base = g_dn_d2 + (size_t)off_l * (size_t)g_dn_d2_n
                                + (size_t)e * (size_t)g_dn_d2_off;
#if defined(__AVX2__) && ST_ARCH_X86
        if (actq16 && ef <= 2048) {
            /* down s16 轨（VLLM_ACTQ16=1）：av 每专家量化一次。 */
            int16_t avq16[2048]; float avd16[64];
            quantize_row_s16_act(av, avq16, avd16, ef);
            for (int rg = 0; rg < (d >> 2); rg++) {  /* down 输出行按 4 行组 */
                const uint8_t *db = D + (size_t)rg * gdB + (size_t)cb0 * 72;
                if (rg + 8 < (d >> 2)) ST_PREFETCH(db + (size_t)8 * gdB);
                moe_q4_down_group_actq16_x86(db, cbe, avq16, avd16, p,
                                             y + (size_t)rg * 4);
            }
        } else
        if (actq && ef <= 2048) {
            /* down 近似轨：av 每专家量化一次（ef≤2048 → ≤64 块） */
            int8_t avq[2048]; float avd[64];
            quantize_row_q8_0_act(av, avq, avd, ef);
            for (int rg = 0; rg < (d >> 2); rg++) {  /* down 输出行按 4 行组 */
                const uint8_t *db = D + (size_t)rg * gdB + (size_t)cb0 * 72;
                if (rg + 8 < (d >> 2)) ST_PREFETCH(db + (size_t)8 * gdB);
                moe_q4_down_group_actq_x86(db, cbe, avq, avd, p,
                                           y + (size_t)rg * 4);
            }
        } else
        if (moe_q4_simd_ok() && (d & 3) == 0) {
            for (int rg = 0; rg < (d >> 2); rg++) {  /* down 输出行按 4 行组 */
                const uint8_t *db = D + (size_t)rg * gdB + (size_t)cb0 * 72;
                moe_q4_down_group_x86(db, cbe, av, p, y + (size_t)rg * 4);
            }
        } else
#elif defined(__aarch64__) && defined(ST_HAVE_NEON) && !ST_ARCH_X86
        if (actq && ef <= 2048) {
            int8_t avq[2048]; float avd[64];
            quantize_row_q8_0_act(av, avq, avd, ef);
            for (int rg = 0; rg < (d >> 2); rg++) {  /* down 近似轨 */
                const uint8_t *db = d2base
                    ? d2base + (size_t)rg * (size_t)g_dn_d2_stride
                    : D + (size_t)rg * gdB + (size_t)cb0 * 72;
                moe_q4_down_group_actq_neon(db, cbe, avq, avd, p, y + (size_t)rg * 4);
            }
        } else
        if (moe_q4_simd_ok() && (d & 3) == 0) {
            arm_dn_f32_ctx dc = { D, y, cb0, cbe, gdB, d >> 2, av, p,
                                  d2base, g_dn_d2_stride };
            vllm_tp_parfor(0, d >> 2, arm_dn_f32_worker, &dc);
        } else
#endif
        for (int rg = 0; rg < (d >> 2); rg++) {  /* down 输出行按 4 行组 */
            const uint8_t *db = D + (size_t)rg * gdB + (size_t)cb0 * 72;
            for (int ri = 0; ri < 4; ri++) {
                float acc = 0.0f;
                const float *ae = av;
                for (int cbi = 0; cbi < cbe; cbi++) {
                    const uint8_t *tile = db + (size_t)cbi * 72;
                    for (int k = 0; k < 32; k++, ae++)
                        acc += vq4x4_w(tile, ri, k) * (*ae);
                }
                y[(size_t)rg * 4 + ri] += p * acc;
            }
        }
    }
    if (prf) { mpt_dn += vllm_tp_wtime() - td0; mpt_ff += vllm_tp_wtime() - tff0; }
}

#if defined(__AVX2__) && ST_ARCH_X86
/* ---- Step2 Phase0（d 立项，2026-09-10，VLLM_MOE_BATCH=1）：prefill MoE 跨 token
 * 融合调度（无 FP 改动，A≡C）----
 * 现状：批 forward 逐 token 调 st_moe_ffn_sparse → 每 token 内 1 次 gu parfor +
 * 逐专家 down parfor（~9 次全池 join/token，共 288 次/层批）。Phase0 把整批合到
 * 两个全池 parfor：(token×专家) 的 gu、(token×行组) 的 down；down worker 内按
 * 该 token 的 rank 序逐专家调用原核（acc→y+=p·acc 步序不变）→ 每行浮点累加序
 * 与单 token 路径逐位一致。权重仍按 token 复读（共享读属 Phase1）。
 * 仅 ACTQ=1 且 q4 专家几何成立时生效，否则调用方回落逐 token。 */
static int moe_batch_env(void) {   /* 2026-09-10 起默认开；VLLM_MOE_BATCH=0 显式关闭（A/B 用） */
    static int v = -1;
    if (v < 0) { const char *e = getenv("VLLM_MOE_BATCH"); v = (e && e[0] == '0') ? 0 : 1; }
    return v;
}
typedef struct {
    const uint8_t *G, *U, *D;
    const float *X;                 /* [nb][d] */
    float *Y;                       /* [nb][d] */
    int8_t *xq; float *xd;          /* 每 token 激活 q8 [nb][d] / [nb][nbG] */
    int16_t *xq16;                  /* §9.22：s16 激活轨 [nb][d]（NULL=关），scale 复用 xd */
    int16_t *avq16;                 /* §9.22：每 (t,j) silu 后 s16 [nsel][ef]，scale 复用 avd */
    int *xcorr;                     /* 2a-b：每 (token,块) 8·Σxq（消 192× 重算） */
    int *avcorr;                    /* 2a-b：每 (t,j,块) 8·Σavq */
    int8_t *avq; float *avd;        /* 每 (token,rank) silu 后 q8 */
    float *av;                      /* 每 (token,rank) silu 后 f32 [nsel][ef] */
    float *P;                       /* down 贡献 partial [(token,rank)][d] */
    int *ex_off;                    /* [ne+1] 专家→(t,j) 归属表偏移 */
    int *ex_t; int *ex_j;           /* 归属表（token, rank） */
    int *sel; float *pr;            /* [nb][tk] */
    int verify;                     /* 2a-b：与逐 token 参考核同场对拍（env 开） */
    int probe;                      /* 2a-c 诊断：1=保留 tile 访存，跳过 gather/解包/dot 的 ALU */
    volatile long sink;             /* 诊断探针防 DCE */
    int nb, tk, d, ef, ne, nbG, ggB, gdB, ge4, cbe, nsel;
    int actq;
    int actq16;                     /* §9.22：s16 轨（与 actq 互斥） */
    int fast;                       /* §9.42：s16 软件流水候选（VLLM_MOE_FAST=1） */
    int ilv;                        /* §9.46：k 方向交错深度 W（VLLM_MOE_ILV，2..4） */
    int wpf;                        /* §9.44：权重 tile「真载入式」预取距离（cb 为单位；0=关） */
} st2_ctx;
/* 2a 接入：把 actq 的"tile 行收集"从 per-token 提到 per-tile（同一 72B tile 的
 * 4 行收集结果对该 tile 关联的所有 token 复用）；后续定点/浮点步序与
 * actq_block_gu_x86 / moe_q4_down_group_actq_x86 内联版逐位相同 → A≡C。 */
static inline void actq_gather_tile(const uint8_t *tile, __m128i R[4]) {
    const uint8_t *qs = tile + 8;
    __m128i c0 = _mm_loadu_si128((const __m128i *)(qs + 0));
    __m128i c1 = _mm_loadu_si128((const __m128i *)(qs + 16));
    __m128i c2 = _mm_loadu_si128((const __m128i *)(qs + 32));
    __m128i c3 = _mm_loadu_si128((const __m128i *)(qs + 48));
    for (int ri = 0; ri < 4; ri++) {
        R[ri] = _mm_or_si128(_mm_or_si128(
                    _mm_shuffle_epi8(c0, actq_msk[0][ri]),
                    _mm_shuffle_epi8(c1, actq_msk[1][ri])),
                _mm_or_si128(_mm_shuffle_epi8(c2, actq_msk[2][ri]),
                             _mm_shuffle_epi8(c3, actq_msk[3][ri])));
    }
}
/* 2a-c：一次 256b shuffle 同时收集两行（lane0 = 行 2p、lane1 = 行 2p+1），
 * 掩码由 actq_msk 派生 → 与原 128b 收集逐位同序（仅指令数 ÷2）。 */
static inline void actq_gather_tile2(const uint8_t *tile, __m256i *R01, __m256i *R23) {
    const uint8_t *qs = tile + 8;
    __m256i A  = _mm256_loadu_si256((const __m256i *)(qs + 0));
    __m256i B  = _mm256_loadu_si256((const __m256i *)(qs + 32));
    __m256i As = _mm256_permute2x128_si256(A, A, 0x01);
    __m256i Bs = _mm256_permute2x128_si256(B, B, 0x01);
    *R01 = _mm256_or_si256(
        _mm256_or_si256(_mm256_shuffle_epi8(A, actq_msk2[0][0]),
                        _mm256_shuffle_epi8(As, actq_msk2[0][1])),
        _mm256_or_si256(_mm256_shuffle_epi8(B, actq_msk2[0][2]),
                        _mm256_shuffle_epi8(Bs, actq_msk2[0][3])));
    *R23 = _mm256_or_si256(
        _mm256_or_si256(_mm256_shuffle_epi8(A, actq_msk2[1][0]),
                        _mm256_shuffle_epi8(As, actq_msk2[1][1])),
        _mm256_or_si256(_mm256_shuffle_epi8(B, actq_msk2[1][2]),
                        _mm256_shuffle_epi8(Bs, actq_msk2[1][3])));
}
/* 2a-c：256b 解包（两 lane 各自低/高半字节 → int8 符号轨），替代 128b 逐行版本。 */
static inline void actq_unpack256(__m256i R, __m256i *L, __m256i *H) {
    const __m256i m15 = _mm256_set1_epi8(15), m0F0F = _mm256_set1_epi16(0x0F0F);
    const __m256i x8 = _mm256_set1_epi8(8);
    *L = _mm256_xor_si256(_mm256_and_si256(R, m15), x8);
    *H = _mm256_xor_si256(_mm256_and_si256(_mm256_srli_epi16(R, 4), m0F0F), x8);
}
/* 2a-b：解包（半字节→int8 符号轨）提升到 per-tile，并把两个 128b 行打包进同一
 * __m256i（低 lane = a 行、高 lane = b 行）。gu 用 (gate,up) 打包、down 用
 * (行0,行1)/(行2,行3) 打包，两条腿/两行共享一次 256b 点积链。
 * 整数部分与 actq_block_gu_x86 逐位同值（整数加可结合、归约顺序无关）。 */
static inline void actq_pack_lh2(__m128i Ra, __m128i Rb, __m256i *L, __m256i *H) {
    const __m128i m15 = _mm_set1_epi8(15), m0F0F = _mm_set1_epi16(0x0F0F);
    const __m128i x8 = _mm_set1_epi8(8);
    __m128i la = _mm_xor_si128(_mm_and_si128(Ra, m15), x8);
    __m128i ha = _mm_xor_si128(_mm_and_si128(_mm_srli_epi16(Ra, 4), m0F0F), x8);
    __m128i lb = _mm_xor_si128(_mm_and_si128(Rb, m15), x8);
    __m128i hb = _mm_xor_si128(_mm_and_si128(_mm_srli_epi16(Rb, 4), m0F0F), x8);
    *L = _mm256_set_m128i(lb, la);
    *H = _mm256_set_m128i(hb, ha);
}
/* 一条 256b 链出两个标量点积（低 lane→plo，高 lane→phi）；行末只归约一次。 */
static inline void actq_dot_lh2(__m256i L, __m256i H, __m256i Xl, __m256i Xh,
                                int *plo, int *phi) {
    const __m256i ones = _mm256_set1_epi16(1);
    __m256i d16 = _mm256_add_epi16(_mm256_maddubs_epi16(L, Xl),
                                   _mm256_maddubs_epi16(H, Xh));
    __m256i i32 = _mm256_madd_epi16(d16, ones);
    i32 = _mm256_hadd_epi32(i32, i32);
    i32 = _mm256_hadd_epi32(i32, i32);
    *plo = _mm_cvtsi128_si32(_mm256_castsi256_si128(i32));
    *phi = _mm_cvtsi128_si32(_mm256_extracti128_si256(i32, 1));
}
/* §9.22：s16 版本的一条 256b 链出两个标量点积（低 lane→plo，高 lane→phi）。
 * 与 actq_dot_lh2 同结构，但权重取**有符号** s8（(nib^8)-8）、激活为 s16，
 * 用 vpmaddwd(w_s16, a_s16) 做**精确**整数点积 → 无需 8·Σact 偏移校正。 */
static inline void actq16_dot_lh2(__m256i L, __m256i H, const int16_t *xq,
                                  int *plo, int *phi) {
    const __m256i c8 = _mm256_set1_epi8(8);
    __m256i Ls = _mm256_sub_epi8(L, c8);
    __m256i Hs = _mm256_sub_epi8(H, c8);
    __m256i xl = _mm256_loadu_si256((const __m256i *)xq);          /* act 0..15 */
    __m256i xh = _mm256_loadu_si256((const __m256i *)(xq + 16));   /* act 16..31 */
    __m256i da = _mm256_add_epi32(
        _mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm256_castsi256_si128(Ls)), xl),
        _mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm256_castsi256_si128(Hs)), xh));
    __m256i db = _mm256_add_epi32(
        _mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(Ls, 1)), xl),
        _mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(Hs, 1)), xh));
    *plo = q8g_reduce8(da);
    *phi = q8g_reduce8(db);
}
/* §9.39：把「乘加」与「水平归约」拆开——原 `actq16_dot_lh2` 每次调用 ≈23 条指令，
 * 其中 11 条花在两次 `q8g_reduce8`（行内归约）＝ ~48%。s16 轨是**整数**运算，
 * 整数加法精确且可结合 → **求和分组可任意重排而结果逐位相同**，故改为：
 *   actq16_madd2：只做乘加，产出 8×int32 的 da/db（不归约）
 *   red4        ：把 4 条 int32 向量一次性归约成一条 __m128i（4 个标量，
 *                 用 `_mm256_hadd_epi32` 树，6 条指令替掉 4×11=44 条）
 * ⚠ 只对 s16（整数）轨使用；int8 轨的 `corr` 与浮点收尾算式未变。 */
static inline void actq16_madd2(__m256i L, __m256i H, __m256i xl, __m256i xh,
                                __m256i *da, __m256i *db) {
    const __m256i c8 = _mm256_set1_epi8(8);
    __m256i Ls = _mm256_sub_epi8(L, c8);
    __m256i Hs = _mm256_sub_epi8(H, c8);
    *da = _mm256_add_epi32(
        _mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm256_castsi256_si128(Ls)), xl),
        _mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm256_castsi256_si128(Hs)), xh));
    *db = _mm256_add_epi32(
        _mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(Ls, 1)), xl),
        _mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(Hs, 1)), xh));
}
/* §9.40 诊断用编译期开关：置 0 可把 v4 路径**整段编译掉**（旧的逐行归约分支回到 §9.39
 * 之前的原文），用于排除「本次改动是否影响了旧路径的代码生成」。正常构建恒为 1。 */
#define ST_MOE_DOTV4_V4 1
/* §9.39 A/B 开关：VLLM_MOE_DOTV4=0 → 回退「逐行归约」旧路径（仅用于同二进制交错 A/B，
 * 消除跨二进制/跨轮次的机器状态漂移）。默认 1 = 4 行批量归约。两路数值逐位相同。 */
static int st2_dotv4_on(void) {
    static int v = -1;
    if (v < 0) {
        const char *s = getenv("VLLM_MOE_DOTV4");
        v = (s && s[0] == '0') ? 0 : 1;
        fprintf(stderr, "[DOTV4] path=%s\n", v ? "v4(4row-batch)" : "old(per-row)");
    }
    return v;
}
/* 4 条 8×int32 向量 → 一条 __m128i，lane r = 第 r 条的全和。整数部分和分组任意，等价。 */
static inline __m128i red4(__m256i v0, __m256i v1, __m256i v2, __m256i v3) {
    __m256i a = _mm256_hadd_epi32(_mm256_hadd_epi32(v0, v1),
                                  _mm256_hadd_epi32(v2, v3));
    return _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extracti128_si256(a, 1));
}
/* ============ §9.42 s16 轨「软件流水」候选实现（同二进制 A/B，默认关） ============
 * 动机（§9.41 插桩）：s16 轨内层每 (cb,k) ≈80 条指令、实测 IPC≈1.4 → 瓶颈在**停顿**
 * 而非指令数；且每条腿 `actq16_madd2` 在**逐 token** 内层里重复做 `sub_epi8(.,8)`
 * （对 k 不变，纯浪费 8 条/(cb,k)）。本候选做两件**位级安全**的事：
 *   ① 收集期 `actq_unpack256s` 直接产出**有符号** int8（nib^8-8），内层 `actq16_madd2s`
 *      免掉 sub —— 整数运算，值逐位相同；
 *   ② k 方向 2-token 交错：先发两条**独立**的激活 64B 载入，再做两条点积，把激活
 *      （L2/L3）延迟与当前 token 的 ALU 重叠。每 token 的 cb 升序累加链、浮点表达式
 *      `(scg*xdc)*(float)p` 一字未改 → 与逐 token 参考核逐位相同（A≡C 保持）。
 * 自 2026-09-14 起**默认开**（`VLLM_MOE_FAST=0` 回退；位级门：流式 ctx192/ctx512
 * text md5+TOKIDS 全同、服务端 continuous batching 38528 组 E2E 全 diffEl=0；
 * 收益 gu −9.0% / GATEUP −5.6% / prefill 总 −3.8%）。 */
static int st2_fast_on(void) {
    static int v = -1;
    if (v < 0) {
        const char *s = getenv("VLLM_MOE_FAST");
        v = (s && s[0] == '0') ? 0 : 1;
        fprintf(stderr, "[MOEFAST] swpipe(k2+sgn)=%s\n", v ? "on" : "off");
    }
    return v;
}
/* §9.46 A/B 开关：`VLLM_MOE_ILV=W`（2..4，**默认 4**）。
 * 动机：§9.44/§9.45 共同证实 gu 内层是**停顿/端口受限**（IPC≈1.4；两种预取均无效；
 * 只吃到内存屋顶 8.6%；且 SMT 逻辑核有 1.4× 级额外收益），故「提高 ILP」是对症的
 * 杠杆，而 §9.42 已在 W=2 上证明机制有效（gu −7.3%）。W 越大，同时在飞的独立
 * 激活载入 / vpmaddwd 链越多，越能遮盖延迟；代价是 k 方向的寄存器压力
 * （L[4]+H[4] 恒占用 8 个 YMM，激活 XL/XH 占 2W 个）。
 * **实测（同二进制交错 2/4/2/4，流式 ctx512 n=16 --warmup，QKV/O 作漂移标尺）**：
 *   gu  W2 均值 4238.4ms → W4 3938.5ms = **−7.1%**（标尺 +0.9%/+1.5% ⇒ 归一后 −7.9~−8.4%）；
 *   prefill 总  W2 6752.7ms → W4 6481.0ms = **−4.0%**（归一后 −5.4%）；
 *   标尺本身 W4 略慢 ⇒ 增益若被噪声影响也是被**低估**。故默认由 2 提升到 4。
 * **位级安全**：每个 token 仍是「按 cb 升序、按同一浮点表达式」累加，组间不共享任何
 * 累加器（只是把 W 个 token 的计算顺序交错）→ 与逐 token 参考核逐位相同（A≡C）。
 * 门已过：流式 4/4 轮 `text_md5`+`TOKIDS` 全同；服务端 `[2BVERIFY] E2E` ILV=2/4
 * 各 26208 行、**nonzero diffEl = 0**。解码段 ms/tok 66/64/70/66（m=1 ⇒ W 被钳为 1，
 * 与串行等价）→ 增益只落在 prefill。 */
static int st2_ilv_on(void) {
    static int v = -1;
    if (v < 0) {
        const char *s = getenv("VLLM_MOE_ILV");
        v = (s && s[0]) ? atoi(s) : 4;
        if (v < 2) v = 2;
        if (v > 8) v = 8;   /* §9.50：上限抬到 8，供 AVX-512 路径使用（AVX2 路径恒取 W=4，行为不变） */
        fprintf(stderr, "[MOEILV] k-interleave=%d\n", v);
    }
    return v;
}
/* §9.50 A/B 开关：`VLLM_MOE_ILV512=1` → gu 走 AVX-512 路径（**默认关**）。
 * 与 `VLLM_MOE_ILV` 组合使用：ILV512=1 & ILV=4 → AVX-512@W4；ILV512=1 & ILV=8 → AVX-512@W8；
 * ILV512=0 → 现有 AVX2 路径。三者都是**同一个二进制**，靠 env 切换做同会话交错 A/B。 */
static int st2_ilv512_on(void) {
    static int v = -1;
    if (v < 0) {
        const char *s = getenv("VLLM_MOE_ILV512");
        v = (s && s[0] == '1') ? 1 : 0;
        fprintf(stderr, "[MOEILV512] avx512-path=%s\n", v ? "on" : "off");
    }
    return v;
}
/* §9.51 A/B 开关：dn 的 k 方向交错深度。`VLLM_MOE_DNILV`：0=关（回到串行 k 循环，
 * 用于同二进制交错 A/B）、2/4/8=上限；**未设则跟随 `VLLM_MOE_ILV`（即默认开 W=4）**。
 * 依据 §9.50⑦ 的实测：dn 占 MoE FFN 段 39.7%（gu 50.1%），且 dn 内层原本**没有**
 * gu §9.46/§9.47 的 k 交错与常量展开。 */
static int st2_dnilv_on(void) {
    static int v = -1;
    if (v < 0) {
        const char *s = getenv("VLLM_MOE_DNILV");
        if (s && s[0]) {
            v = atoi(s);
            if (v < 0) v = 0;
            if (v > 8) v = 8;
        } else {
            v = st2_ilv_on();
        }
        fprintf(stderr, "[MOEDNILV] dn k-interleave=%d\n", v);
    }
    return v;
}
/* §9.44 A/B 开关：VLLM_MOE_WPF=N → 在 cb 体开头用**真载入**（而不是 `_mm_prefetch`）
 * 把前第 N 个块的权重 tile 拉进来。默认 0（关）。
 * ⚠ **实测结论：无效且有害，保持默认关**（同二进制交错 WPF=0/4/0/4/8，QKV 作漂移标尺）：
 *   gu +6.7%、dn +11.7%、GATEUP +7.6%、prefill 总 +5.0%（标尺校正后）。
 *   回退幅度≈新增 load 条数占比（+4/67≈+6%）→ 说明该流**不是延迟暴露**，加在途载入
 *   只多占发射槽；与 §9.36 的 `_mm_prefetch` 无效同因（HW 预取器已覆盖 distance-4 的顺序流）。
 * ❌ 并**一并更正 §9.43 的归因错误**：当时据 PROBE=4（2276ms）判「gu 的 ~2/3 花在权重
 *   tile 流」，那是**串行化伪影**——PROBE=4 删掉了 m 循环，原本用来遮盖 tile 流的并行
 *   工作也没了，故高估了该份额。反证：本机内存屋顶实测 46~55 GB/s（见 C:\vllm_xwork\membw.c），
 *   而 gu 的 tile 流仅 ~9.7GB/2.276s = **4.3 GB/s = 屋顶的 8.6%** ⇒ gu **不是带宽受限**。
 *   真正的瓶颈是**停顿/端口受限**（内层 IPC≈1.4 → 见 §9.42 注释；SMT 有效收益见 §9.45）。 */
static int st2_wpf_on(void) {
    static int v = -1;
    if (v < 0) {
        const char *s = getenv("VLLM_MOE_WPF");
        v = (s && s[0]) ? atoi(s) : 0;
        if (v < 0) v = 0;
        if (v > 8) v = 8;
        fprintf(stderr, "[MOEWPF] weight-prefetch dist=%d\n", v);
    }
    return v;
}
static inline void actq_unpack256s(__m256i R, __m256i *L, __m256i *H) {
    const __m256i m15 = _mm256_set1_epi8(15), m0F0F = _mm256_set1_epi16(0x0F0F);
    const __m256i x8 = _mm256_set1_epi8(8);
    *L = _mm256_sub_epi8(_mm256_xor_si256(_mm256_and_si256(R, m15), x8), x8);
    *H = _mm256_sub_epi8(_mm256_xor_si256(
            _mm256_and_si256(_mm256_srli_epi16(R, 4), m0F0F), x8), x8);
}
/* 与 actq16_madd2 同构，但入参已是**有符号** int8（免 sub_epi8）。 */
static inline void actq16_madd2s(__m256i L, __m256i H, __m256i xl, __m256i xh,
                                 __m256i *da, __m256i *db) {
    *da = _mm256_add_epi32(
        _mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm256_castsi256_si128(L)), xl),
        _mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm256_castsi256_si128(H)), xh));
    *db = _mm256_add_epi32(
        _mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(L, 1)), xl),
        _mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(H, 1)), xh));
}
/* §9.47：k 方向交错体的**编译期常量**展开版（WC 传字面量 → 两个 for 全展开、
 * 下标变常量 → XL/XH/XDC 全部寄存器化，不再经栈做 store-forwarding）。
 * 语义与 §9.46 的运行时 W 版**逐位相同**：每个 token 仍按 cb 升序、按同一浮点
 * 表达式累加，组间不共享任何累加器（只是把 WC 个 token 的计算顺序交错）。 */
#define ST2_ILV_BODY(WC)                                                        \
    do {                                                                        \
        __m256i XL[WC], XH[WC];                                                 \
        float XDC[WC];                                                          \
        for (int j = 0; j < (WC); j++) {                                        \
            int tj = c->ex_t[o0 + k + j];                                       \
            const int16_t *xsj = c->xq16 + (size_t)tj * c->d + (size_t)cb * 32; \
            XL[j] = _mm256_loadu_si256((const __m256i *)xsj);                   \
            XH[j] = _mm256_loadu_si256((const __m256i *)(xsj + 16));            \
            XDC[j] = c->xd[(size_t)tj * c->nbG + cb];                           \
        }                                                                       \
        for (int j = 0; j < (WC); j++) {                                        \
            __m256i a0, b0, a1, b1, a2, b2, a3, b3;                             \
            actq16_madd2s(L[0], H[0], XL[j], XH[j], &a0, &b0);                  \
            actq16_madd2s(L[1], H[1], XL[j], XH[j], &a1, &b1);                  \
            actq16_madd2s(L[2], H[2], XL[j], XH[j], &a2, &b2);                  \
            actq16_madd2s(L[3], H[3], XL[j], XH[j], &a3, &b3);                  \
            __m128i gs = red4(a0, a1, a2, a3);                                  \
            __m128i us = red4(b0, b1, b2, b3);                                  \
            __m128 xdcv = _mm_set1_ps(XDC[j]);                                  \
            __m128 gsc = _mm_mul_ps(_mm_loadu_ps(scgv), xdcv);                  \
            __m128 usc = _mm_mul_ps(_mm_loadu_ps(scuv), xdcv);                  \
            _mm_storeu_ps(&gacc[k + j][0],                                      \
                          _mm_add_ps(_mm_loadu_ps(&gacc[k + j][0]),             \
                                     _mm_mul_ps(gsc, _mm_cvtepi32_ps(gs))));    \
            _mm_storeu_ps(&uacc[k + j][0],                                      \
                          _mm_add_ps(_mm_loadu_ps(&uacc[k + j][0]),             \
                                     _mm_mul_ps(usc, _mm_cvtepi32_ps(us))));    \
        }                                                                       \
    } while (0)
/* ============ §9.50 AVX-512 路径（gu，opt-in `VLLM_MOE_ILV512=1`，默认关）============
 * 动机来自探针 `C:\vllm_xwork\avx512_probe.c`（2026-09-14）：
 *   ① 把内层**算术量翻倍**（同一段 body 跑两遍）耗时**不变** ⇒ 内层不是吞吐/端口受限，
 *      而是**停顿受限**。因此「AVX-512 减少指令数」本身**不产生收益**：实测 512b 重排
 *      只有 0.999×，且重排版的指令数反而**更多**——x86 没有 512-bit 水平归约
 *      （`vphaddd` 只到 256b），归约被迫用 `vextracti64x4` 拆开做，把 `vpmaddwd`
 *      省下的全吃掉。**这是一条重要的负面结论：不要为「更宽的码字」重写 gu。**
 *   ② 探针里 W 从 1 扫到 16（数据驻留 L1/L2）耗时**完全无差别** ⇒ 实机 W 的收益来自
 *      **掩盖真实 cache/DRAM 延迟（MLP）**，不是 ALU。
 *   ③ AVX2 只有 16 个 YMM：W=4 时权重 `L[4]+H[4]` 占 8 个、激活 `XL/XH` 占 2W=8 个
 *      = **正好占满**（§9.46 注释已指出"代价是寄存器压力"）。这就是 W 卡在 4 的
 *      结构原因；**AVX-512 的真正价值是寄存器富余度**：32 个 ZMM，且每个 token 的
 *      激活（偶数列 16 + 奇数列 16 = 64B）**只需 1 个 ZMM 一次载入**（AVX2 要 XL+XH
 *      两个 YMM）。于是每 cb 的占用从「8 权重 + 2W 激活」降到「8 权重 + W 激活」，
 *      W=8 只用 16/32 个寄存器 ⇒ 换来 2 倍**在途激活载入数**。
 * 位级安全：整数归约是精确加法、可任意结合 ⇒ 分组改变不影响结果；浮点表达式
 *   `gacc[k][ri] += (scg*xdc)*(float)p` 一字未改，且每个 token 仍按 **cb 升序**累加
 *   （cb 仍是外层循环、组间不共享累加器）⇒ 与逐 token 参考核逐位相同（A≡C）。
 * 编译方式：**不改全局编译 flags**，只在本组函数上用 `target` 属性局部启用 AVX-512，
 *   故 TU 内其余代码的代码生成完全不变（避免 §9.19「改 flags 破位级指纹」的老坑）。
 * ⚠ 适用性：本机 9800X3D 是 Zen5、512-bit 全宽数据通路。Zen4 及更早为「双泵」，
 *   512-bit 吞吐减半，需另行评估。 */
#define ST2_AVX512_TARGET __attribute__((target("avx512f,avx512bw,avx512vl,avx512dq")))

/* 4 条 512b 向量（各 16×int32）→ 一条 __m128i，lane r = 第 r 条的全和。
 * 先把每条的高低两个 256b 半区逐元素相加（整数加法精确，等价），再复用 red4 的 hadd 树。 */
ST2_AVX512_TARGET
static inline __m128i red512_4(__m512i v0, __m512i v1, __m512i v2, __m512i v3) {
    __m256i t0 = _mm256_add_epi32(_mm512_castsi512_si256(v0), _mm512_extracti64x4_epi64(v0, 1));
    __m256i t1 = _mm256_add_epi32(_mm512_castsi512_si256(v1), _mm512_extracti64x4_epi64(v1, 1));
    __m256i t2 = _mm256_add_epi32(_mm512_castsi512_si256(v2), _mm512_extracti64x4_epi64(v2, 1));
    __m256i t3 = _mm256_add_epi32(_mm512_castsi512_si256(v3), _mm512_extracti64x4_epi64(v3, 1));
    __m256i a = _mm256_hadd_epi32(_mm256_hadd_epi32(t0, t1), _mm256_hadd_epi32(t2, t3));
    return _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extracti128_si256(a, 1));
}

/* 一个 cb 的 AVX-512 全量处理：建权重寄存器 + 对 m 个归属 token 走 W 档位阶梯。
 * 权重 W[0..3] = gate 行 0..3、W[4..7] = up 行 0..3，每个 256b =
 *   [该行偶数列 16×int8 | 该行奇数列 16×int8]（取自 §9.47 的 L4/H4，即有符号 s8）。
 * 激活 X = 该 token 的 64B（32×int16，布局 [偶数列16 | 奇数列16]）一次载入；
 *   `cvtepi8_epi16(W[r])` 给出 int16 权重，与 X 的 `vpmaddwd` 即该行的完整点积。 */
ST2_AVX512_TARGET
static void st2_gu_cb512(const __m256i *Lv, const __m256i *Hv, const st2_ctx *c,
                         int o0, int m, int cb, __m128 scgv, __m128 scuv,
                         float (*gacc)[4], float (*uacc)[4], int ilv) {
    /* ⚠⚠ 两个**必须同时满足**的约束（2026-09-14 两次踩坑）：
     * ① 只能收**副本的指针**，绝不能把调用方的 `L`/`H` 数组直接传进来：`L[]` 一旦
     *    逃逸到这个不可内联的函数，GCC 就不再把 `L[ri]`/`H[ri]` 留在寄存器（原
     *    基线的 `&L[0]` 是传给 `static inline` 的 `actq_unpack256s`，GCC 看得穿），
     *    默认 AVX2 路径凭空慢 ~4.6%（`vllm_kestrel_x64` 对 `pre950` 交叉对照实测）。
     * ② 也**不能改成 8 个 `__m256i` 形参按值传**：本函数带 `target("avx512f")`，
     *    GCC 在 AVX-512 下把向量形参改走 ZMM 寄存器，而调用方没开 AVX-512、仍按
     *    基础 ABI 用隐藏指针传 → **ABI 不匹配，进函数即崩**（实测 prefill 首步崩）。
     * 故：调用方传**栈上副本** `Lc/Hc` 的地址。 */
    __m256i W[8];
    for (int r = 0; r < 4; r++) {
        W[r]     = _mm256_set_m128i(_mm256_castsi256_si128(Hv[r]),
                                    _mm256_castsi256_si128(Lv[r]));       /* gate 行 r */
        W[4 + r] = _mm256_set_m128i(_mm256_extracti128_si256(Hv[r], 1),
                                    _mm256_extracti128_si256(Lv[r], 1));  /* up   行 r */
    }
#define ST2_ILV512_BODY(WC)                                                       \
    do {                                                                          \
        __m512i X512[WC]; float XDC[WC];                                          \
        for (int j = 0; j < (WC); j++) {                                          \
            int tj = c->ex_t[o0 + k + j];                                         \
            X512[j] = _mm512_loadu_si512((const void *)(c->xq16                \
                        + (size_t)tj * c->d + (size_t)cb * 32));                  \
            XDC[j] = c->xd[(size_t)tj * c->nbG + cb];                             \
        }                                                                         \
        for (int j = 0; j < (WC); j++) {                                          \
            __m128i gs = red512_4(                                                \
                _mm512_madd_epi16(_mm512_cvtepi8_epi16(W[0]), X512[j]),           \
                _mm512_madd_epi16(_mm512_cvtepi8_epi16(W[1]), X512[j]),           \
                _mm512_madd_epi16(_mm512_cvtepi8_epi16(W[2]), X512[j]),           \
                _mm512_madd_epi16(_mm512_cvtepi8_epi16(W[3]), X512[j]));          \
            __m128i us = red512_4(                                                \
                _mm512_madd_epi16(_mm512_cvtepi8_epi16(W[4]), X512[j]),           \
                _mm512_madd_epi16(_mm512_cvtepi8_epi16(W[5]), X512[j]),           \
                _mm512_madd_epi16(_mm512_cvtepi8_epi16(W[6]), X512[j]),           \
                _mm512_madd_epi16(_mm512_cvtepi8_epi16(W[7]), X512[j]));          \
            __m128 xdcv = _mm_set1_ps(XDC[j]);                                    \
            __m128 gsc = _mm_mul_ps(scgv, xdcv);                                  \
            __m128 usc = _mm_mul_ps(scuv, xdcv);                                  \
            _mm_storeu_ps(&gacc[k + j][0],                                        \
                          _mm_add_ps(_mm_loadu_ps(&gacc[k + j][0]),               \
                                     _mm_mul_ps(gsc, _mm_cvtepi32_ps(gs))));      \
            _mm_storeu_ps(&uacc[k + j][0],                                        \
                          _mm_add_ps(_mm_loadu_ps(&uacc[k + j][0]),               \
                                     _mm_mul_ps(usc, _mm_cvtepi32_ps(us))));      \
        }                                                                         \
    } while (0)

    int k = 0;
    if (ilv >= 8) {
        for (; k + 8 <= m; k += 8) ST2_ILV512_BODY(8);
    }
    if (ilv >= 4) {
        for (; k + 4 <= m; k += 4) ST2_ILV512_BODY(4);
    }
    for (; k + 2 <= m; k += 2) ST2_ILV512_BODY(2);
    if (k < m) ST2_ILV512_BODY(1);
#undef ST2_ILV512_BODY
}

/* gu（相位1+2a+2a-b+2a-d）：task=行组主序 (g,e)。行组主序使每个线程的连续 chunk
 * 横跨全部专家 → 每 chunk 的 dot 总量恒为 Σ_e m_e（与 m_e 分布无关），消除
 * 专家主序下「撞上一串高 m_e 专家」的长尾（实测 GATEUP -17%，重复性 ±0.2%）。
 * 每 tile「行收集+解包+gate/up 打包」一次供 |T_e| 个 token 复用；corr 走预计算表。
 * 浮点仍按块序逐 (k,ri) 累加、表达式与逐 token 路径同形 → A≡C。
 * 注：2a-g 试过「块循环（每专家 GE 行组连续）」改善局部性 → GE∈{1,2,4,8} 扫描
 * 归一化后全在噪声内（gu 段中位 9.55→9.34ms，±2%）= null，已回退。 */
static void st2_gu_worker(void *c_, int idx) {
    st2_ctx *c = c_;
    int g = idx / c->ne, e = idx - g * c->ne;
    int o0 = c->ex_off[e], o1 = c->ex_off[e + 1];
    int m = o1 - o0;
    if (m <= 0) return;
    const uint8_t *ge = c->G + (size_t)((size_t)e * c->ge4) * c->ggB;
    const uint8_t *ue = c->U + (size_t)((size_t)e * c->ge4) * c->ggB;
    const uint8_t *gb = ge + (size_t)g * c->ggB;
    const uint8_t *ub = ue + (size_t)g * c->ggB;
    if (c->verify && idx == 0) {   /* 2a-b 临时：整数对拍（vs 逐行参考核） */
        int badi = 0;
        for (int cb2 = 0; cb2 < c->nbG; cb2++) {
            const uint8_t *gt2 = gb + (size_t)cb2 * 72, *ut2 = ub + (size_t)cb2 * 72;
            __m128i Rg2[4], Ru2[4];
            __m256i L2[4], H2[4];
            actq_gather_tile(gt2, Rg2);
            actq_gather_tile(ut2, Ru2);
            for (int ri = 0; ri < 4; ri++) actq_pack_lh2(Rg2[ri], Ru2[ri], &L2[ri], &H2[ri]);
            for (int k2 = 0; k2 < m; k2++) {
                int t2 = c->ex_t[o0 + k2];
                const int8_t *xs2 = c->xq + (size_t)t2 * c->d + (size_t)cb2 * 32;
                int32_t vpgs[4], vpus[4];
                actq_block_gu_x86(gt2, ut2, xs2, vpgs, vpus);
                __m256i Xl2 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)xs2));
                __m256i Xh2 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)(xs2 + 16)));
                for (int ri = 0; ri < 4; ri++) {
                    int pg2, pu2;
                    actq_dot_lh2(L2[ri], H2[ri], Xl2, Xh2, &pg2, &pu2);
                    if (pg2 != vpgs[ri] || pu2 != vpus[ri]) {
                        if (badi < 4)
                            printf("[2BVERIFY] gu int MISMATCH cb=%d k=%d ri=%d gate %d/%d up %d/%d\n",
                                   cb2, k2, ri, pg2, vpgs[ri], pu2, vpus[ri]);
                        badi++;
                    }
                }
            }
        }
        printf("[2BVERIFY] gu e=%d m=%d intBAD=%d\n", e, m, badi);
        fflush(stdout);
    }
    /* 修复（30B-MoE serve/NLL 首个请求段错误根因）：本 worker 处理「专家 e 的
     * 全部归属 token」，k 的上界是 m = 该专家的 (token,rank) 归属数，而 m 的上
     * 界是 prefill mini-batch 的 nb（g_st_prefill_batch 默认 256），并非 top_k=8
     * 或 32。原 `float gacc[32][4]` 是错把 m 当成 batch 上限 32；当 nb>32 且某专
     * 家被 >32 个 token 选中（30B 实测 nb=74 时 maxm=44）即写越界，砸穿 worker
     * 线程栈 → 0xC0000005。改为按 m 变长数组（VLA），容量恒等于实际归属数，
     * 无浮点步序改动 → A≡C 保持。 */
    float gacc[m][4], uacc[m][4];
    for (int k = 0; k < m; k++)
        for (int ri = 0; ri < 4; ri++) { gacc[k][ri] = 0.0f; uacc[k][ri] = 0.0f; }
    int local_sink = 0;   /* §9.33：probe 累加改局部，避免所有 worker 抢一个 int */
    for (int cb = 0; cb < c->nbG; cb++) {
        const uint8_t *gt = gb + (size_t)cb * 72, *ut = ub + (size_t)cb * 72;
        uint16_t hg[4], hu[4];
        float scgv[4], scuv[4];
        __m256i L[4], H[4];
        if (cb + 4 < c->nbG) ST_PREFETCH(gt + (size_t)4 * 72);
        if (cb + 4 < c->nbG) ST_PREFETCH(ut + (size_t)4 * 72);
        if (c->wpf) {   /* §9.44 真载入式预取：前第 wpf 个块的整块 tile（2×32B / 投影） */
            int pf = cb + c->wpf;
            if (pf < c->nbG) {
                const uint8_t *pg = gb + (size_t)pf * 72;
                const uint8_t *pu = ub + (size_t)pf * 72;
                __m256i v0 = _mm256_loadu_si256((const __m256i *)(pg + 8));
                __m256i v1 = _mm256_loadu_si256((const __m256i *)(pg + 40));
                __m256i v2 = _mm256_loadu_si256((const __m256i *)(pu + 8));
                __m256i v3 = _mm256_loadu_si256((const __m256i *)(pu + 40));
                local_sink += _mm256_extract_epi32(
                    _mm256_or_si256(_mm256_or_si256(v0, v1), _mm256_or_si256(v2, v3)), 0);
            }
        }
        memcpy(hg, gt, 8);
        memcpy(hu, ut, 8);
        if (c->probe == 3) {
            /* §9.43 定份额档：其余全走（tile 流 + 激活载入 + 点积 + 浮点收尾），
             * **只把 block scale 换成常数**（跳过 vq_f16，并保留 memcpy 的 8B 载入防 DCE）
             * → full 与 probe3 之差 = `vq_f16` 在 gu 里的真实份额。 */
            for (int ri = 0; ri < 4; ri++) { scgv[ri] = 1.0f; scuv[ri] = 1.0f; }
            local_sink += (int)hg[0] + (int)hu[0];
        } else {
            for (int ri = 0; ri < 4; ri++) {
                scgv[ri] = vq_f16(hg[ri]); scuv[ri] = vq_f16(hu[ri]);
            }
        }
        if (c->probe != 2) {   /* 2a-c：256b 两行/次收集 + 逐行 [up|gate] 打包 */
            __m256i Rg01, Rg23, Ru01, Ru23;
            actq_gather_tile2(gt, &Rg01, &Rg23);
            actq_gather_tile2(ut, &Ru01, &Ru23);
            if (c->fast) {   /* §9.42：直接产出有符号 int8（内层免 sub） */
                actq_unpack256s(_mm256_permute2x128_si256(Rg01, Ru01, 0x20), &L[0], &H[0]);
                actq_unpack256s(_mm256_permute2x128_si256(Rg01, Ru01, 0x31), &L[1], &H[1]);
                actq_unpack256s(_mm256_permute2x128_si256(Rg23, Ru23, 0x20), &L[2], &H[2]);
                actq_unpack256s(_mm256_permute2x128_si256(Rg23, Ru23, 0x31), &L[3], &H[3]);
            } else {
                actq_unpack256(_mm256_permute2x128_si256(Rg01, Ru01, 0x20), &L[0], &H[0]);
                actq_unpack256(_mm256_permute2x128_si256(Rg01, Ru01, 0x31), &L[1], &H[1]);
                actq_unpack256(_mm256_permute2x128_si256(Rg23, Ru23, 0x20), &L[2], &H[2]);
                actq_unpack256(_mm256_permute2x128_si256(Rg23, Ru23, 0x31), &L[3], &H[3]);
            }
        } else {   /* §9.41 probe=2：去 tile 流，常数权重喂 ALU（激活仍真读 → 不会整体外提） */
            for (int ri = 0; ri < 4; ri++) { L[ri] = _mm256_set1_epi8(3); H[ri] = _mm256_set1_epi8(5); }
        }
        /* §9.50：整段 cb 交给 AVX-512 路径（W 档位由 c->ilv 决定），跳过下面的逐 token
         * k 循环。只在正式路径（probe=0、wpf=0、s16 轨、fast 开）下启用；其余诊断档
         * 一律走原 AVX2 代码，保证 §9.41–§9.44 的份额诊断口径不被污染。 */
        if (c->xq16 && c->fast && c->probe == 0 && c->wpf == 0 && st2_ilv512_on()) {
            /* 传栈上**副本**：既不让 L/H 逃逸（保寄存器化），又避开 AVX-512 向量
             * 形参的 ABI 差异（详见 st2_gu_cb512 头部注释）。每 cb 8 次拷贝，
             * 摊销到 m 个 token 上可忽略。 */
            __m256i Lc[4], Hc[4];
            for (int r = 0; r < 4; r++) { Lc[r] = L[r]; Hc[r] = H[r]; }
            st2_gu_cb512(Lc, Hc, c, o0, m, cb,
                         _mm_loadu_ps(scgv), _mm_loadu_ps(scuv),
                         gacc, uacc, c->ilv);
            continue;
        }
        for (int k = 0; k < m; k++) {
            int t = c->ex_t[o0 + k];
            float xdc = c->xd[(size_t)t * c->nbG + cb];
            int corr = 0;
            int32_t pgs[4], pus[4];
            if (c->probe == 4) {
                /* §9.43 定份额档：**只留 tile 流**（gather 照走、每 cb 全量读 144B），
                 * 既**不读激活**也不做点积。于是与 probe=1（tile 流 + 激活读、无点积）
                 * 之差 = **激活载入**的成本，而 probe=4 本身 = **权重 tile 流 + gather** 成本。
                 * 用来判定该给哪条流做流水（寄存器预算只够一条）。 */
                local_sink += _mm256_extract_epi32(L[0], 0) + _mm256_extract_epi32(H[0], 0);
                for (int ri = 0; ri < 4; ri++) { pgs[ri] = 0; pus[ri] = 0; }
            } else if (c->probe == 1) {   /* §9.41 仅访存：tile 已全量 gather；此处真读激活 64B，
                                    * 跳过全部点积 ALU（pgs/pus=0 → 浮点项恰为 0）。 */
                const int16_t *xs = c->xq16 + (size_t)t * c->d + (size_t)cb * 32;
                int32_t v = (int32_t)xs[0] + (int32_t)xs[16]
                          + _mm256_extract_epi32(L[0], 0) + _mm256_extract_epi32(H[0], 0);
                local_sink += v;
                for (int ri = 0; ri < 4; ri++) { pgs[ri] = 0; pus[ri] = 0; }
            } else if (c->xq16) {
                /* §9.39 s16 轨：4 行批量归约 + 4 行向量化浮点收尾（数值逐位不变：
                 * 整数部分和分组任意等价；浮点仍是逐元素 (scg*xdc)*(float)p 再累加）。 */
                const int16_t *xs16 = c->xq16 + (size_t)t * c->d + (size_t)cb * 32;
#if ST_MOE_DOTV4_V4
                if (st2_dotv4_on()) {
                if (c->fast) {
                    /* §9.46/§9.47：k 方向 W-token 交错（W = c->ilv，默认 4）。
                     * ① 先把 W 个 token 的激活 64B（+各自 xdc）全部发出——这些载入彼此
                     *    独立，可同时在飞；② 再做 W 组「4×actq16_madd2s + red4 + 浮点收尾」，
                     *    组间只共享对 k 不变的 L[0..3]/H[0..3]，累加器互不依赖。
                     * §9.47：改成**编译期常量** WC=4/WC=2/WC=1 三次实例化（宏展开），
                     * 下标均为常量 ⇒ XL/XH/XDC 全进寄存器（§9.46 的运行时 W 版会把
                     * 这三个数组落栈、只能靠 store-forwarding 近似「在飞」）。
                     * ⚠ **尾部必须留在本体内**（WC=1 一档）：本体用 `actq16_madd2s`
                     * （有符号，对 `actq_unpack256s` 产出的 L/H 直接 madd），而下面
                     * 那条串行路径配的是**无符号**的 `actq_unpack256` + `actq16_madd2`
                     * （内层还要 `sub_epi8` 减 8）。若让尾 token「落下去」，就会对有符号
                     * 值再减一次 8 → 数值错（首版即踩此坑，被 `text_md5` 挡下）。
                     * **实测（同会话双二进制交错 A/B/A/B/A/B，QKV+O 双标尺）**：
                     * gu 常量展开均 3923.2ms vs 运行时 W 均 4009.7ms = **−2.2%**
                     * （5/5 相邻对全偏向展开版；标尺显示展开版那侧环境反而慢 0.5~0.7%
                     * ⇒ 归一后 −2.6~−2.8%），prefill 总 −1.1%（归一 ~−1.7%）。 */
                    if (c->ilv >= 4 && k + 4 <= m) { ST2_ILV_BODY(4); k += 3; continue; }
                    if (k + 2 <= m)                 { ST2_ILV_BODY(2); k += 1; continue; }
                    ST2_ILV_BODY(1); continue;      /* 尾 1 个 token（k<m 恒成立） */
                }
                __m256i xl = _mm256_loadu_si256((const __m256i *)xs16);
                __m256i xh = _mm256_loadu_si256((const __m256i *)(xs16 + 16));
                __m256i a0, b0, a1, b1, a2, b2, a3, b3;
                actq16_madd2(L[0], H[0], xl, xh, &a0, &b0);
                actq16_madd2(L[1], H[1], xl, xh, &a1, &b1);
                actq16_madd2(L[2], H[2], xl, xh, &a2, &b2);
                actq16_madd2(L[3], H[3], xl, xh, &a3, &b3);
                __m128i gs = red4(a0, a1, a2, a3);   /* gate：4 行 */
                __m128i us = red4(b0, b1, b2, b3);   /* up  ：4 行 */
                __m128 xdcv = _mm_set1_ps(xdc);
                __m128 gsc = _mm_mul_ps(_mm_loadu_ps(scgv), xdcv);
                __m128 usc = _mm_mul_ps(_mm_loadu_ps(scuv), xdcv);
                _mm_storeu_ps(&gacc[k][0],
                              _mm_add_ps(_mm_loadu_ps(&gacc[k][0]),
                                         _mm_mul_ps(gsc, _mm_cvtepi32_ps(gs))));
                _mm_storeu_ps(&uacc[k][0],
                              _mm_add_ps(_mm_loadu_ps(&uacc[k][0]),
                                         _mm_mul_ps(usc, _mm_cvtepi32_ps(us))));
                continue;
                }
#endif
                for (int ri = 0; ri < 4; ri++)   /* A/B 旧路径：逐行归约（= §9.39 之前的实现） */
                    actq16_dot_lh2(L[ri], H[ri], xs16, &pgs[ri], &pus[ri]);
            } else {
                const int8_t *xs = c->xq + (size_t)t * c->d + (size_t)cb * 32;
                __m256i Xl = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)xs));
                __m256i Xh = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)(xs + 16)));
                for (int ri = 0; ri < 4; ri++)
                    actq_dot_lh2(L[ri], H[ri], Xl, Xh, &pgs[ri], &pus[ri]);
                corr = c->xcorr[(size_t)t * c->nbG + cb];
            }
            for (int ri = 0; ri < 4; ri++) {   /* 与逐 token 参考核逐字同形（保 A≡C） */
                float scg = scgv[ri], scu = scuv[ri];
                gacc[k][ri] += (scg * xdc) * (float)(pgs[ri] - corr);
                uacc[k][ri] += (scu * xdc) * (float)(pus[ri] - corr);
            }
        }
    }
    if (c->probe || c->wpf) c->sink += local_sink;   /* §9.44：wpf 时也必须消费，否则真载入会被 DCE */
    if (c->verify && idx == 0) {   /* gu 全链对拍：vs moe_q4_gu_group_actq_x86（真参考核） */
        int badf = 0;
        for (int k2 = 0; k2 < m; k2++) {
            int t2 = c->ex_t[o0 + k2];
            float gv[4], uv[4];
            moe_q4_gu_group_actq_x86(gb, ub, c->nbG, c->xq + (size_t)t2 * c->d,
                                     c->xd + (size_t)t2 * c->nbG, gv, uv);
            for (int ri = 0; ri < 4; ri++) {
                if (gv[ri] != gacc[k2][ri] || uv[ri] != uacc[k2][ri]) {
                    if (badf < 3)
                        printf("[2BVERIFY] gu FLOAT k=%d ri=%d G %.9g/%.9g U %.9g/%.9g\n",
                               k2, ri, gacc[k2][ri], gv[ri], uacc[k2][ri], uv[ri]);
                    badf++;
                }
            }
        }
        printf("[2BVERIFY] gu vs REF e=%d m=%d floatBAD=%d / %d\n", e, m, badf, m * 4);
        fflush(stdout);
    }
    for (int k = 0; k < m; k++) {
        int t = c->ex_t[o0 + k], j = c->ex_j[o0 + k];
        float *av = c->av + ((size_t)t * c->tk + j) * c->ef + (size_t)g * 4;
        for (int ri = 0; ri < 4; ri++) {
            float gg = gacc[k][ri], uu = uacc[k][ri];
            av[ri] = (gg / (1.0f + expf(-gg))) * uu;
        }
    }
}
/* down：task = (行组, 专家)，行组主序（负载均衡，理由同 gu）。
 * 注：① 「每任务合并 rgs 个行组」实测 0.8%（噪声内）→ null 已回退；
 *     ② 「块循环 GE 局部性」实测 ±2% → null 已回退（dn 段中位 7.45→7.18ms）。 */
/* ============ §9.51 dn 的 k 方向交错体（编译期常量展开版，opt-in）============
 * 依据 §9.50⑦：dn 占 MoE FFN 段 **39.7%**（gu 50.1%），绝对量是 gu 的 79%，但 dn 内层
 * **完全没有** gu §9.46/§9.47 已经拿到的「k 方向交错（W）」与「编译期常量展开」。
 * 与 gu 的 ST2_ILV_BODY 同构，差异只在寄存器组：
 *   dn 每 cb 只用 **4 个权重寄存器**（`L01/H01` = 行对(1,0)、`L23/H23` = 行对(3,2)）
 *   + 每 token **2 个激活寄存器**（`xl/xh`，同一 `avq16` 供两组共用），
 *   故 WC=4 时仅占 4+8=12 个 YMM（gu 的同类交错是 8+8=**占满 16 个**）⇒ 压力更小。
 * 每 token：2 次 `actq16_madd2` + **1 次** `red4`（gu 是 4 次 madd + 2 次 red4）、
 *   一次 4 lane 浮点收尾。
 * 位级安全：整数归约是精确加法、可任意结合 ⇒ 分组改变不影响结果；每个 token 仍按
 *   **cbi 升序**、按同一浮点表达式 `acc[k][ri] += (sd[ri]*xdc) * (float)ps[ri]` 累加，
 *   组间不共享累加器（只是把 WC 个 token 的计算顺序交错）⇒ 与串行 k 路径逐位相同。
 * ⚠ 注意 dn 用的是**无符号** `actq16_madd2`（内含 `sub_epi8(.,8)`，因 `L01/H01` 来自
 *   `actq_unpack256` 的 xor8 轨）；这与 gu 的 `actq16_madd2s`（有符号、免 sub）不同，
 *   §9.47 记的「尾部 fall through 会被多减一次 8」正是这两种轨混用造成的——本处没有
 *   其它路径可落，故不存在该坑。 */
#define ST2_DN_ILV_BODY(WC)                                                     \
    do {                                                                        \
        __m256i XL[WC], XH[WC]; float XDC[WC];                                  \
        for (int j = 0; j < (WC); j++) {                                        \
            int tj = c->ex_t[o0 + k + j], jj = c->ex_j[o0 + k + j];             \
            size_t sj = (size_t)tj * c->tk + jj;                                \
            const int16_t *avj = c->avq16 + sj * c->ef + (size_t)cbi * 32;      \
            XL[j] = _mm256_loadu_si256((const __m256i *)avj);                   \
            XH[j] = _mm256_loadu_si256((const __m256i *)(avj + 16));            \
            XDC[j] = c->avd[sj * c->cbe + cbi];                                 \
        }                                                                       \
        for (int j = 0; j < (WC); j++) {                                        \
            __m256i a0, b0, a1, b1;                                             \
            actq16_madd2(L01, H01, XL[j], XH[j], &a0, &b0);                     \
            actq16_madd2(L23, H23, XL[j], XH[j], &a1, &b1);                     \
            __m128i ps = red4(a0, b0, a1, b1);                                  \
            __m128 scv = _mm_mul_ps(_mm_loadu_ps(sd), _mm_set1_ps(XDC[j]));     \
            _mm_storeu_ps(&acc[k + j][0],                                       \
                          _mm_add_ps(_mm_loadu_ps(&acc[k + j][0]),              \
                                     _mm_mul_ps(scv, _mm_cvtepi32_ps(ps))));    \
        }                                                                       \
    } while (0)

static void st2_dn_worker(void *c_, int idx) {
    st2_ctx *c = c_;
    int rg = idx / c->ne, e = idx - rg * c->ne;
    int o0 = c->ex_off[e], o1 = c->ex_off[e + 1];
    int m = o1 - o0;
    if (m <= 0) return;
    /* 该专家列带（4 行组）：tile 读一次 + 行收集一次，供其全部 token 复用 */
    const uint8_t *db = c->D + (size_t)rg * (size_t)c->gdB
                              + (size_t)((e * c->ef) >> 5) * 72;
    /* 同上：down worker 的 partial 累加按归属数 m 定界，原 [32] 在 nb>32 时越界。 */
    float acc[m][4];
    for (int k = 0; k < m; k++)
        for (int ri = 0; ri < 4; ri++) acc[k][ri] = 0.0f;
    /* §9.33：probe 用的累加器必须是**本 worker 局部**的。原先直接 `c->sink += v`
     * 是让所有 worker 抢同一个 int → 伪共享，实测 probe（跳过 ALU）反而比完整
     * 路径慢 1.9×，使该诊断失去意义。改为局部累加、函数末尾一次性回写。 */
    int local_sink = 0;
    for (int cbi = 0; cbi < c->cbe; cbi++) {
        const uint8_t *tile = db + (size_t)cbi * 72;
        uint16_t hd[4];
        float sd[4];
        __m256i L01, H01, L23, H23;
        if (cbi + 4 < c->cbe) ST_PREFETCH(db + (size_t)(cbi + 4) * 72);
        if (c->wpf) {   /* §9.44 真载入式预取（同 gu）：前第 wpf 个块的整块 tile */
            int pf = cbi + c->wpf;
            if (pf < c->cbe) {
                const uint8_t *pt = db + (size_t)pf * 72;
                __m256i v0 = _mm256_loadu_si256((const __m256i *)(pt + 8));
                __m256i v1 = _mm256_loadu_si256((const __m256i *)(pt + 40));
                local_sink += _mm256_extract_epi32(_mm256_or_si256(v0, v1), 0);
            }
        }
        memcpy(hd, tile, 8);
        if (c->probe == 3) {
            /* §9.43 定份额档（同 gu）：只跳过 scale 解码 vq_f16，其余全走 */
            for (int ri = 0; ri < 4; ri++) sd[ri] = 1.0f;
            local_sink += (int)hd[0];
        } else {
            for (int ri = 0; ri < 4; ri++) sd[ri] = vq_f16(hd[ri]);
        }
        if (c->probe != 2) {   /* 2a-c：一次 256b 收集即得 [行1|行0] / [行3|行2] */
            __m256i Rd01, Rd23;
            actq_gather_tile2(tile, &Rd01, &Rd23);
            actq_unpack256(Rd01, &L01, &H01);
            actq_unpack256(Rd23, &L23, &H23);
        } else {   /* §9.41 probe=2：去 tile 流，常数权重喂 ALU（激活仍真读） */
            L01 = _mm256_set1_epi8(3); H01 = _mm256_set1_epi8(5);
            L23 = _mm256_set1_epi8(3); H23 = _mm256_set1_epi8(5);
        }
        /* §9.51：本 cbi 的整段 k 交给交错体（W 档由 st2_dnilv_on() 决定），随后跳过下面的
         * 串行 k 循环。只在正式路径（probe=0、wpf=0、s16 轨、dotv4 开、DNILV>=2）启用；
         * `VLLM_MOE_DNILV=0` 回到原有串行路径，供**同二进制**交错 A/B。 */
        if (c->avq16 && c->probe == 0 && c->wpf == 0 && st2_dotv4_on()) {
            int dnw = st2_dnilv_on();
            if (dnw >= 2) {
                int k = 0;
                if (dnw >= 8) for (; k + 8 <= m; k += 8) ST2_DN_ILV_BODY(8);
                if (dnw >= 4) for (; k + 4 <= m; k += 4) ST2_DN_ILV_BODY(4);
                for (; k + 2 <= m; k += 2) ST2_DN_ILV_BODY(2);
                if (k < m) ST2_DN_ILV_BODY(1);
                continue;
            }
        }
#undef ST2_DN_ILV_BODY
        for (int k = 0; k < m; k++) {
            int t = c->ex_t[o0 + k], j = c->ex_j[o0 + k];
            size_t s = (size_t)t * c->tk + j;
            float xdc = c->avd[s * c->cbe + cbi];
            int corr = 0;
            int p0, p1, p2, p3;
            if (c->probe == 4) {   /* §9.43 定份额档（同 gu）：只留 tile 流，不读激活、不点积 */
                local_sink += _mm256_extract_epi32(L01, 0) + _mm256_extract_epi32(H01, 0);
                p0 = p1 = p2 = p3 = 0;
            } else if (c->probe == 1) {   /* §9.41 仅访存：tile 已全量 gather；真读激活 64B，跳过点积 */
                const int16_t *avq16r = c->avq16 + s * c->ef + (size_t)cbi * 32;
                int v = (int)avq16r[0] + (int)avq16r[16]
                      + _mm256_extract_epi32(L01, 0) + _mm256_extract_epi32(H01, 0);
                local_sink += v;
                p0 = p1 = p2 = p3 = 0;
            } else if (c->avq16) {
                /* §9.39 s16 轨：2 次调用 → 4 路一次归约（[p0,p1,p2,p3]）+ 向量化浮点收尾。
                 * 整数部分和分组任意等价；浮点仍逐元素 (sd*xdc)*(float)p 再累加 → 位级不变。 */
                const int16_t *avq16 = c->avq16 + s * c->ef + (size_t)cbi * 32;
#if ST_MOE_DOTV4_V4
                if (st2_dotv4_on()) {
                __m256i xl = _mm256_loadu_si256((const __m256i *)avq16);
                __m256i xh = _mm256_loadu_si256((const __m256i *)(avq16 + 16));
                __m256i a0, b0, a1, b1;
                actq16_madd2(L01, H01, xl, xh, &a0, &b0);
                actq16_madd2(L23, H23, xl, xh, &a1, &b1);
                __m128i ps = red4(a0, b0, a1, b1);
                __m128 scv = _mm_mul_ps(_mm_loadu_ps(sd), _mm_set1_ps(xdc));
                _mm_storeu_ps(&acc[k][0],
                              _mm_add_ps(_mm_loadu_ps(&acc[k][0]),
                                         _mm_mul_ps(scv, _mm_cvtepi32_ps(ps))));
                continue;
                }
#endif
                actq16_dot_lh2(L01, H01, avq16, &p0, &p1);   /* A/B 旧路径：逐行归约 */
                actq16_dot_lh2(L23, H23, avq16, &p2, &p3);
            } else {
                const int8_t *avq = c->avq + s * c->ef + (size_t)cbi * 32;
                __m256i Xl = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)avq));
                __m256i Xh = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)(avq + 16)));
                actq_dot_lh2(L01, H01, Xl, Xh, &p0, &p1);
                actq_dot_lh2(L23, H23, Xl, Xh, &p2, &p3);
                corr = c->avcorr[s * c->cbe + cbi];
            }
            acc[k][0] += (sd[0] * xdc) * (float)(p0 - corr);
            acc[k][1] += (sd[1] * xdc) * (float)(p1 - corr);
            acc[k][2] += (sd[2] * xdc) * (float)(p2 - corr);
            acc[k][3] += (sd[3] * xdc) * (float)(p3 - corr);
        }
    }
    if (c->probe || c->wpf) c->sink += local_sink;   /* §9.44：wpf 时也必须消费，否则真载入会被 DCE */
    if (c->verify && idx == 0) {   /* dn 全链对拍：vs moe_q4_down_group_actq_x86（p=1） */
        int badf = 0;
        float yref[4];
        for (int k2 = 0; k2 < m; k2++) {
            int t2 = c->ex_t[o0 + k2], j2 = c->ex_j[o0 + k2];
            size_t s2 = (size_t)t2 * c->tk + j2;
            yref[0] = yref[1] = yref[2] = yref[3] = 0.0f;
            moe_q4_down_group_actq_x86(db, c->cbe, c->avq + s2 * c->ef,
                                       c->avd + s2 * c->cbe, 1.0f, yref);
            for (int ri = 0; ri < 4; ri++) if (yref[ri] != acc[k2][ri]) badf++;
        }
        printf("[2BVERIFY] dn e=%d m=%d float mismatch=%d / %d\n", e, m, badf, m * 4);
        fflush(stdout);
    }
    for (int k = 0; k < m; k++) {
        int t = c->ex_t[o0 + k], j = c->ex_j[o0 + k];
        size_t s = (size_t)t * c->tk + j;
        float *P = c->P + s * c->d + (size_t)rg * 4;
        for (int ri = 0; ri < 4; ri++) P[ri] = acc[k][ri];   /* 存未加 p（p 留到 combine 内 FMA） */
    }
}
/* 合并：按 token 自身 rank 序把 partial 累加进 Y（Y 起点 0 → 与单 token 逐位同）。
 * 用显式 fmaf 复刻参考核 `y[ri] += p*acc[ri]` 的单次舍入（否则 P 预乘+加 = 两次舍入，
 * 会与逐 token 路径产生 1 ULP 偏差）。 */
static void st2_combine_worker(void *c_, int t) {
    st2_ctx *c = c_;
    float *y = c->Y + (size_t)t * c->d;
    for (int j = 0; j < c->tk; j++) {
        size_t s = (size_t)t * c->tk + j;
        if (c->sel[s] < 0) continue;
        const float *P = c->P + s * c->d;
        const float p = c->pr[s];
        for (int i = 0; i < c->d; i++) y[i] = fmaf(p, P[i], y[i]);
    }
}
/* av f32 → 每 (t,j) q8（与单 token 路径同式）+ 预计算每块 corr（2a-b） */
static void st2_quant_worker(void *c_, int k) {
    st2_ctx *c = c_;
    quantize_row_q8_0_act(c->av + (size_t)k * c->ef, c->avq + (size_t)k * c->ef,
                          c->avd + (size_t)k * c->cbe, c->ef);
    for (int cb = 0; cb < c->cbe; cb++)
        c->avcorr[(size_t)k * c->cbe + cb] =
            8 * actq_sum_s8_32(c->avq + (size_t)k * c->ef + (size_t)cb * 32);
}
/* §9.22：av f32 → 每 (t,j) s16（与逐 token 路径同式；scale 复用 avd）。无 corr。 */
static void st2_quant_worker16(void *c_, int k) {
    st2_ctx *c = c_;
    quantize_row_s16_act(c->av + (size_t)k * c->ef, c->avq16 + (size_t)k * c->ef,
                         c->avd + (size_t)k * c->cbe, c->ef);
}
/* 精确轨（actq=0）gu：wf 预计算（解包+×scale）每块一次、供 |T_e| 个 token 复用；
 * 逐 k f32 乘加序与 q4x4_tile32_x86 逐位一致（_mm_mul_ps+_mm_add_ps 不收缩）。 */
static void st2_gu_worker_exact(void *c_, int idx) {
    st2_ctx *c = c_;
    int g = idx / c->ne, e = idx - g * c->ne;
    int o0 = c->ex_off[e], o1 = c->ex_off[e + 1];
    int m = o1 - o0;
    if (m <= 0) return;
    const uint8_t *ge = c->G + (size_t)((size_t)e * c->ge4) * c->ggB;
    const uint8_t *ue = c->U + (size_t)((size_t)e * c->ge4) * c->ggB;
    const uint8_t *gb = ge + (size_t)g * c->ggB;
    const uint8_t *ub = ue + (size_t)g * c->ggB;
    __m128 gacc[m], uacc[m];
    for (int k = 0; k < m; k++) { gacc[k] = _mm_setzero_ps(); uacc[k] = _mm_setzero_ps(); }
    for (int cb = 0; cb < c->nbG; cb++) {
        const uint8_t *gt = gb + (size_t)cb * 72, *ut = ub + (size_t)cb * 72;
        if (cb + 4 < c->nbG) { ST_PREFETCH(gt + (size_t)4 * 72); ST_PREFETCH(ut + (size_t)4 * 72); }
        __m128 wfg[32], wfu[32];
        q4x4_tile32_wf_x86(gt, wfg);
        q4x4_tile32_wf_x86(ut, wfu);
        for (int k = 0; k < m; k++) {
            int t = c->ex_t[o0 + k];
            const float *x = c->X + (size_t)t * c->d + (size_t)cb * 32;
            for (int kk = 0; kk < 32; kk++) {
                __m128 xv = _mm_set1_ps(x[kk]);
                gacc[k] = _mm_add_ps(gacc[k], _mm_mul_ps(wfg[kk], xv));
                uacc[k] = _mm_add_ps(uacc[k], _mm_mul_ps(wfu[kk], xv));
            }
        }
    }
    for (int k = 0; k < m; k++) {
        int t = c->ex_t[o0 + k], j = c->ex_j[o0 + k];
        float gv[4], uv[4];
        _mm_storeu_ps(gv, gacc[k]);
        _mm_storeu_ps(uv, uacc[k]);
        float *av = c->av + ((size_t)t * c->tk + j) * c->ef + (size_t)g * 4;
        for (int ri = 0; ri < 4; ri++) {
            float gg = gv[ri], uu = uv[ri];
            av[ri] = (gg / (1.0f + expf(-gg))) * uu;   /* silu(g)*u，同逐 token 核 */
        }
    }
}
/* 精确轨（actq=0）dn：wf 预计算跨 token 复用；存未加权 acc → combine 内 mul+add。
 * 注意：dn 参考核 moe_q4_down_group_x86 的 acc 累加（q4x4_tile32_x86 内）被 GCC
 * 在 -mfma 下收缩成 FMA（单次舍入），故此处用显式 _mm_fmadd_ps 匹配；而 gu 参考核
 * moe_q4_gu_group_x86 未被收缩（保持 mul+add），故 gu worker 用 _mm_add_ps(_mm_mul_ps)。 */
static void st2_dn_worker_exact(void *c_, int idx) {
    st2_ctx *c = c_;
    int rg = idx / c->ne, e = idx - rg * c->ne;
    int o0 = c->ex_off[e], o1 = c->ex_off[e + 1];
    int m = o1 - o0;
    if (m <= 0) return;
    const uint8_t *db = c->D + (size_t)rg * (size_t)c->gdB
                              + (size_t)((e * c->ef) >> 5) * 72;
    __m128 acc[m];
    for (int k = 0; k < m; k++) acc[k] = _mm_setzero_ps();
    for (int cbi = 0; cbi < c->cbe; cbi++) {
        const uint8_t *tile = db + (size_t)cbi * 72;
        if (cbi + 4 < c->cbe) ST_PREFETCH(db + (size_t)(cbi + 4) * 72);
        __m128 wfd[32];
        q4x4_tile32_wf_x86(tile, wfd);
        for (int k = 0; k < m; k++) {
            int t = c->ex_t[o0 + k], j = c->ex_j[o0 + k];
            size_t s = (size_t)t * c->tk + j;
            const float *av = c->av + s * c->ef + (size_t)cbi * 32;
            for (int kk = 0; kk < 32; kk++) {
                __m128 avv = _mm_set1_ps(av[kk]);
                acc[k] = _mm_fmadd_ps(wfd[kk], avv, acc[k]);
            }
        }
    }
    for (int k = 0; k < m; k++) {
        int t = c->ex_t[o0 + k], j = c->ex_j[o0 + k];
        size_t s = (size_t)t * c->tk + j;
        _mm_storeu_ps(c->P + s * c->d + (size_t)rg * 4, acc[k]);
    }
}
/* 精确轨（actq=0）合并：按 token 自身 rank 序把 partial 累加进 Y。
 * 精确轨逐 token down 参考核 moe_q4_down_group_x86 末步是显式
 * _mm_add_ps(yo, _mm_mul_ps(pv, acc))（mul+add 两次舍入），故此处必须用同样
 * mul+add（不能像 actq 轨那样用 fmaf 单次舍入，否则与逐 token 差 1 ULP）。 */
static void st2_combine_worker_exact(void *c_, int t) {
    st2_ctx *c = c_;
    float *y = c->Y + (size_t)t * c->d;
    for (int j = 0; j < c->tk; j++) {
        size_t s = (size_t)t * c->tk + j;
        if (c->sel[s] < 0) continue;
        const float *P = c->P + s * c->d;
        __m128 pv = _mm_set1_ps(c->pr[s]);
        for (int i = 0; i < c->d; i += 4)
            _mm_storeu_ps(y + i, _mm_add_ps(_mm_loadu_ps(y + i),
                                            _mm_mul_ps(pv, _mm_loadu_ps(P + i))));
    }
}
/* §9.32：批式 router（logits = X · routerᵀ）按专家并行。
 * 原实现是**单线程标量**（4-token 展开），而同一层里的 gu/dn 都走 `vllm_tp_parfor`
 * + AVX2 → 实测 router 只有 12.6 GFLOP/s（gu 267 / dn 212 / 稠密 O 447），
 * 是整个 MoE FFN 桶里最大的效率离群点。每个专家只写 rlg[t*ne+e]（互不重叠），
 * 且每条累加仍是 i 升序的同一表达式 → **位级不变**。 */
typedef struct {
    const float *X; const float *router; float *rlg;
    int ne, d, nb;
} st2_router_ctx;

static void st2_router_worker(void *p_, int e) {
    st2_router_ctx *p = (st2_router_ctx *)p_;
    const int nb = p->nb, d = p->d, ne = p->ne;
    const float *rw = p->router + (size_t)e * d;
    const float *X = p->X;
    float *rlg = p->rlg;
    int t = 0;
    for (; t + 4 <= nb; t += 4) {
        const float *x0 = X + (size_t)(t + 0) * d;
        const float *x1 = X + (size_t)(t + 1) * d;
        const float *x2 = X + (size_t)(t + 2) * d;
        const float *x3 = X + (size_t)(t + 3) * d;
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        for (int i = 0; i < d; i++) {
            float r = rw[i];
            s0 += r * x0[i]; s1 += r * x1[i];
            s2 += r * x2[i]; s3 += r * x3[i];
        }
        rlg[(size_t)(t + 0) * ne + e] = s0;
        rlg[(size_t)(t + 1) * ne + e] = s1;
        rlg[(size_t)(t + 2) * ne + e] = s2;
        rlg[(size_t)(t + 3) * ne + e] = s3;
    }
    for (; t < nb; t++) {
        const float *xt = X + (size_t)t * d;
        float s = 0.0f;
        for (int i = 0; i < d; i++) s += rw[i] * xt[i];
        rlg[(size_t)t * ne + e] = s;
    }
}

/* §9.32：MoE 批式 FFN 的 scratch 复用（线程局部 arena）。
 * 原实现每次调用 malloc/free 约 30 MB（P 16.8 MB、av 6.3 MB、avq16 3.1 MB …，
 * 按 nb=256 → nsel=nb*tk=2048 计）。prefill 每层调一次，1014 token 约 4 个
 * mini-batch × 48 层 ≈ 192 次 → 约 5.8 GB 的「大块分配 + 首触缺页（Windows
 * 提交页由内核零填）+ 释放」，实测占整个 MoE FFN 桶约 26%（4.6 s）——这部分
 * 落在 [2BSEG] 插桩窗口之外，曾被误读成「MoE 内核效率只有稠密路径的 0.44×」。
 * 改为按形状惰性增长的 per-thread arena（容量够就复用）。纯缓冲管理，
 * 不触碰任何数值路径（位级不变）。 */
static __thread unsigned char *g_st2_arena = NULL;
static __thread size_t g_st2_arena_cap = 0;

static void *st2_arena_reserve(size_t need) {
    if (g_st2_arena_cap < need) {
        unsigned char *p = (unsigned char *)malloc(need);
        if (!p) return NULL;              /* 保留旧 arena；调用方回退逐 token */
        free(g_st2_arena);
        g_st2_arena = p;
        g_st2_arena_cap = need;
    }
    return g_st2_arena;
}
#define ST2_ARENA_OFF(acc, bytes) do { \
        (acc) = off; off += (((size_t)(bytes)) + 63u) & ~(size_t)63u; } while (0)

/* 返回 1=已处理，0=不适用（调用方回落逐 token）。 */
static int st_moe_ffn_sparse_q4_batch(const STModelWeights *w, int l, int nb,
                                      const float *X, float *Y) {
    const STModelConfig *c = &w->cfg;
    int ne = c->n_experts, tk = c->top_k, d = c->dim, ff = c->ffn_dim;
    if (ne <= 0 || tk <= 0 || tk > 32 || ne > 4096 || d <= 0 || nb < 2) return 0;
    int actq = moe_q4_actq_ok();   /* 0=精确轨 GroupGEMM；1=actq 近似轨（VLLM_ACTQ=1） */
    int actq16 = actq16_env();     /* §9.22：s16 激活整数轨（VLLM_ACTQ16=1），与 actq 互斥 */
    if (actq16) actq = 0;
    if (!moe_batch_env() || !w->q4_gate_weight) return 0;
    int ef = ff / ne;
    if (ef <= 0 || ne * ef != ff || (ef & 3) || (d & 31) || ef > 2048) return 0;
    int nbG = d >> 5, ggB = nbG * 72, gdB = (ff >> 5) * 72, ge4 = ef >> 2;
    int off_l = st_ffn_q4_layer_off(w, l);
    const uint8_t *G = w->q4_gate_weight + Q4_BYTES((size_t)off_l * (size_t)ff * d);
    const uint8_t *U = w->q4_up_weight   + Q4_BYTES((size_t)off_l * (size_t)ff * d);
    const uint8_t *D = w->q4_down_weight + Q4_BYTES((size_t)off_l * (size_t)d * ff);
    if (!G || !U || !D) return 0;
    actq_tab_ensure_x86();
    size_t nsel = (size_t)nb * tk;
    /* §9.32：一次性从 per-thread arena 切分，替代每次 malloc/free。 */
    size_t off = 0, o_xq, o_xd, o_xc, o_ac, o_rlg, o_avq, o_avd, o_av, o_P,
           o_sel, o_pr, o_eo, o_et, o_ej, o_us, o_x16, o_a16;
    ST2_ARENA_OFF(o_xq,  (size_t)nb * d);
    ST2_ARENA_OFF(o_xd,  (size_t)nb * nbG * sizeof(float));
    ST2_ARENA_OFF(o_xc,  (size_t)nb * nbG * sizeof(int));
    ST2_ARENA_OFF(o_ac,  nsel * (size_t)(ef >> 5) * sizeof(int));
    ST2_ARENA_OFF(o_rlg, (size_t)nb * ne * sizeof(float));
    ST2_ARENA_OFF(o_avq, nsel * (size_t)ef);
    ST2_ARENA_OFF(o_avd, nsel * (size_t)(ef >> 5) * sizeof(float));
    ST2_ARENA_OFF(o_av,  nsel * (size_t)ef * sizeof(float));
    ST2_ARENA_OFF(o_P,   nsel * (size_t)d * sizeof(float));
    ST2_ARENA_OFF(o_sel, nsel * sizeof(int));
    ST2_ARENA_OFF(o_pr,  nsel * sizeof(float));
    ST2_ARENA_OFF(o_eo,  ((size_t)ne + 1) * sizeof(int));
    ST2_ARENA_OFF(o_et,  nsel * sizeof(int));
    ST2_ARENA_OFF(o_ej,  nsel * sizeof(int));
    ST2_ARENA_OFF(o_us,  (size_t)ne * sizeof(int));
    if (actq16) {
        ST2_ARENA_OFF(o_x16, (size_t)nb * d * sizeof(int16_t));
        ST2_ARENA_OFF(o_a16, nsel * ef * sizeof(int16_t));
    }
    unsigned char *arena = (unsigned char *)st2_arena_reserve(off);
    if (!arena) return 0;
    int8_t *xq  = (int8_t *)(arena + o_xq);
    float  *xd  = (float  *)(arena + o_xd);
    int    *xcorr = (int *)(arena + o_xc);
    int    *avcorr = (int *)(arena + o_ac);
    float  *rlg = (float *)(arena + o_rlg);
    int8_t *avq = (int8_t *)(arena + o_avq);
    float  *avd = (float *)(arena + o_avd);
    float  *av  = (float *)(arena + o_av);
    float  *P   = (float *)(arena + o_P);
    int    *sel = (int *)(arena + o_sel);
    float  *pr  = (float *)(arena + o_pr);
    int    *ex_off = (int *)(arena + o_eo);
    int    *ex_t = (int *)(arena + o_et);
    int    *ex_j = (int *)(arena + o_ej);
    int    *used = (int *)(arena + o_us);
    /* §9.22：s16 激活缓冲（scale 复用 xd/avd，故只需两个 s16 值缓冲）。 */
    int16_t *xq16 = NULL, *avq16 = NULL;
    if (actq16) {
        xq16  = (int16_t *)(arena + o_x16);
        avq16 = (int16_t *)(arena + o_a16);
    }
    memset(Y, 0, (size_t)nb * d * sizeof(float));
    double t_entry = moe_now();
    const float *router = w->moe_router + (size_t)l * ne * d;
    /* 2a-e 批式 router：专家外层 → router 权重每层只读一次（原为每 token 复读 →
     * 32MB/层）；§9.32 起改为 `vllm_tp_parfor` 按专家并行，worker 内的
     * 4-token 展开与累加式逐字保留 → per-(t,e) 位级同值，A≡C 安全。 */
    st2_router_ctx rctx;
    rctx.X = X; rctx.router = router; rctx.rlg = rlg;
    rctx.ne = ne; rctx.d = d; rctx.nb = nb;
    vllm_tp_parfor(0, ne, st2_router_worker, &rctx);
    for (int t = 0; t < nb; t++) {
        const float *xt = X + (size_t)t * d;
        if (actq) {
            quantize_row_q8_0_act(xt, xq + (size_t)t * d, xd + (size_t)t * nbG, d);
            for (int cb = 0; cb < nbG; cb++)
                xcorr[(size_t)t * nbG + cb] =
                    8 * actq_sum_s8_32(xq + (size_t)t * d + (size_t)cb * 32);
        } else if (actq16) {
            quantize_row_s16_act(xt, xq16 + (size_t)t * d, xd + (size_t)t * nbG, d);
        }
        /* 确定性 top-k + softmax（与单 token 路径同式） */
        const float *lg = rlg + (size_t)t * ne;
        float sv[32];
        for (int k = 0; k < tk; k++) { sel[(size_t)t*tk+k] = -1; sv[k] = -INFINITY; }
        for (int e = 0; e < ne; e++) {
            for (int k = 0; k < tk; k++) {
                if (lg[e] > sv[k] || (lg[e] == sv[k] &&
                    (sel[(size_t)t*tk+k] < 0 || e < sel[(size_t)t*tk+k]))) {
                    for (int q = tk - 1; q > k; q--) {
                        sel[(size_t)t*tk+q] = sel[(size_t)t*tk+q-1];
                        sv[q] = sv[q-1];
                    }
                    sel[(size_t)t*tk+k] = e; sv[k] = lg[e];
                    break;
                }
            }
        }
        float mx = -INFINITY;
        for (int k = 0; k < tk; k++) if (sv[k] > mx) mx = sv[k];
        float sum = 0.0f;
        for (int k = 0; k < tk; k++) { pr[(size_t)t*tk+k] = expf(sv[k] - mx); sum += pr[(size_t)t*tk+k]; }
        float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
        for (int k = 0; k < tk; k++) pr[(size_t)t*tk+k] *= inv;
    }
    st2_ctx ctx;
    ctx.G = G; ctx.U = U; ctx.D = D; ctx.X = X; ctx.Y = Y;
    ctx.xq = xq; ctx.xd = xd; ctx.avq = avq; ctx.avd = avd;
    ctx.xcorr = xcorr; ctx.avcorr = avcorr;
    ctx.av = av; ctx.P = P;
    ctx.ex_off = ex_off; ctx.ex_t = ex_t; ctx.ex_j = ex_j;
    ctx.sel = sel; ctx.pr = pr; ctx.ne = ne; ctx.nsel = (int)nsel;
    ctx.nb = nb; ctx.tk = tk; ctx.d = d; ctx.ef = ef; ctx.nbG = nbG;
    ctx.ggB = ggB; ctx.gdB = gdB; ctx.ge4 = ge4; ctx.cbe = ef >> 5;
    ctx.actq = actq;
    ctx.actq16 = actq16;
    ctx.fast = (actq16 && st2_fast_on()) ? 1 : 0;
    ctx.ilv = st2_ilv_on();
    ctx.wpf = st2_wpf_on();
    ctx.xq16 = xq16; ctx.avq16 = avq16;
    ctx.verify = (getenv("VLLM_MOE_2B_VERIFY") && getenv("VLLM_MOE_2B_VERIFY")[0] == '1') ? 1 : 0;
    /* §9.41 诊断修正：原 probe 在 probe 模式下**根本不调 actq_gather_tile2**，只读 gt[0]/ut[0]
     * （且对 k 循环不变 → 会被 GCC 外提）⇒ 它测的其实**不是专家权重流**。改为两档：
     *   VLLM_MOE_2B_PROBE=1：**仅访存**——tile 全量 gather（真读 72B）+ 激活读 64B，跳过点积；
     *   VLLM_MOE_2B_PROBE=2：**去 tile 流**——不 gather（用常数权重），激活读 + 点积照走。
     * 于是：tile 流成本 ≈ full − probe2，ALU 成本 ≈ full − probe1。仅诊断，生产数值零改动。 */
    ctx.probe = 0;
    { const char *pv = getenv("VLLM_MOE_2B_PROBE"); if (pv && pv[0]) ctx.probe = atoi(pv); }
    ctx.sink = 0;
    /* 专家→(token,rank) 归属表（共享读；确定性：按 token/rank 升序填） */
    memset(ex_off, 0, ((size_t)ne + 1) * sizeof(int));
    for (size_t k = 0; k < nsel; k++) {
        int e = sel[k];
        if (e >= 0 && e < ne) ex_off[e + 1]++;
    }
    for (int e = 0; e < ne; e++) ex_off[e + 1] += ex_off[e];
    /* #region debug-point A:moe-batch-maxm */
    if (getenv("VLLM_MOE_BATCH_DBG") && getenv("VLLM_MOE_BATCH_DBG")[0] == '1') {
        int maxm = 0;
        for (int e = 0; e < ne; e++) {
            int mm = ex_off[e + 1] - ex_off[e];
            if (mm > maxm) maxm = mm;
        }
        printf("[MOEBATCH] l=%d nb=%d tk=%d ne=%d maxm=%d (per-worker stack arrays sized 32)\n",
               l, nb, tk, ne, maxm);
        fflush(stdout);
    }
    /* #endregion */
    memset(used, 0, (size_t)ne * sizeof(int));
    for (int t = 0; t < nb; t++)
        for (int j = 0; j < tk; j++) {
            int e = sel[(size_t)t * tk + j];
            if (e < 0 || e >= ne) continue;
            int pos = ex_off[e] + used[e]++;
            ex_t[pos] = t; ex_j[pos] = j;
        }
    /* 相位1：gu 共享读 → av 量化 → down 共享读派 partial → rank 序合并 */
    memset(av, 0, nsel * (size_t)ef * sizeof(float));
    int seg = (getenv("VLLM_MOE_2B_SEG") && getenv("VLLM_MOE_2B_SEG")[0] == '1') ? 1 : 0;
    double s0 = seg ? moe_now() : 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0;
    vllm_tp_parfor(0, ne * ge4, (ctx.actq || ctx.actq16) ? st2_gu_worker : st2_gu_worker_exact, &ctx);
    if (seg) s1 = moe_now();
    if (ctx.actq16) vllm_tp_parfor(0, (int)nsel, st2_quant_worker16, &ctx);
    else if (ctx.actq) vllm_tp_parfor(0, (int)nsel, st2_quant_worker, &ctx);
    if (seg) s2 = moe_now();
    vllm_tp_parfor(0, ne * (d >> 2), (ctx.actq || ctx.actq16) ? st2_dn_worker : st2_dn_worker_exact, &ctx);
    if (seg) s3 = moe_now();
    vllm_tp_parfor(0, nb, (ctx.actq || ctx.actq16) ? st2_combine_worker : st2_combine_worker_exact, &ctx);
    if (seg) {
        s4 = moe_now();
        printf("[2BSEG] l=%d tok=%.2f gu=%.2f q=%.2f dn=%.2f cmb=%.2f (ms)\n",
               l, (s0 - t_entry) * 1e3, (s1 - s0) * 1e3, (s2 - s1) * 1e3,
               (s3 - s2) * 1e3, (s4 - s3) * 1e3);
        fflush(stdout);
    }
    if (ctx.verify) {   /* 2a-b 临时端到端：批 MoE Y vs 逐 token 路径 */
        float *yt = (float *)malloc((size_t)d * sizeof(float));
        if (yt) {
            for (int t = 0; t < nb; t++) {
                int bad = 0; float maxd = 0.0f;
                memset(yt, 0, (size_t)d * sizeof(float));
                st_moe_ffn_sparse_q4(w, l, X + (size_t)t * d, yt);
                for (int i = 0; i < d; i++) {
                    float a = Y[(size_t)t * d + i], b = yt[i];
                    if (a != b) {
                        float df = a > b ? a - b : b - a;
                        if (df > maxd) maxd = df;
                        bad++;
                    }
                }
                printf("[2BVERIFY] E2E l=%d t=%d diffEl=%d/%d maxAbs=%.6g\n",
                       l, t, bad, d, maxd);
            }
            fflush(stdout);
            free(yt);
        }
    }
    /* §9.32：缓冲归 per-thread arena 所有，跨层/跨 mini-batch 复用，此处不释放。 */
    return 1;
}
#endif /* __AVX2__ && ST_ARCH_X86 */

#if defined(__aarch64__) && defined(ST_HAVE_NEON) && !ST_ARCH_X86 && ST_NEON_DOTPROD
/* ---- ARM 同构：Step2 批式 MoE（镜像 x86 st2_*，2026-09-10，VLLM_MOE_BATCH=1）----
 * 结构与 x86 完全一致：gu = (行组,专家) 行组主序（负载均衡）；down = (行组,专家)
 * 行组主序；av 走独立 pass 量化；down 存**未加权** acc（P）→ combine 按各 token
 * 自身 rank 序 FMA 合并进 Y（Y 起点 0）。批式 router：专家外层，权重每层只读一次。
 * 与 x86 的差异是平台语义决定、非实现取舍：
 *   ① ARM 权重侧已把 -8 偏置折进点积（actq_w_from_r 的 vsubq_s8）→ 整数精确，无 corr；
 *   ② tile 行收集用 vqtbl1q_u8（128-bit 单行/次 × 4 chunk 再 or）；
 *   ③ 点积用 i8x16_dot_s32（ST_NEON_DOTPROD 下即 vdotq_s32）。
 * 每 (行,块) 的 f32 乘加表达式与 cb 升序与逐 token 参考核
 * （moe_q4_gu_group_actq_neon / moe_q4_down_group_actq_neon）逐字同形 → A≡C。 */
static int arm2_batch_env(void) {   /* 2026-09-10 起默认开；VLLM_MOE_BATCH=0 显式关闭（A/B 用） */
    static int v = -1;
    if (v < 0) { const char *e = getenv("VLLM_MOE_BATCH"); v = (e && e[0] == '0') ? 0 : 1; }
    return v;
}
typedef struct {
    const uint8_t *G, *U, *D;
    const float *X;                 /* [nb][d] */
    float *Y;                       /* [nb][d] */
    int8_t *xq; float *xd;          /* 每 token 激活 q8 [nb][d] / [nb][nbG] */
    int8_t *avq; float *avd;        /* 每 (token,rank) silu 后 q8 */
    float *av;                      /* 每 (token,rank) silu 后 f32 [nsel][ef] */
    float *P;                       /* down 未加权 partial [(token,rank)][d] */
    int *ex_off;                    /* [ne+1] 专家→(t,j) 归属表偏移 */
    int *ex_t; int *ex_j;           /* 归属表（token, rank） */
    int *sel; float *pr;            /* [nb][tk] */
    int verify;
    int actq;                      /* 1=SDOT 近似轨；0=精确轨 f32 乘加 */
    int nb, tk, d, ef, ne, nbG, ggB, gdB, ge4, cbe, nsel;
} arm2_ctx;

/* tile 头 8B 的 4 个 f16 行 scale → f32[4] */
static inline void arm2_scales4(const uint8_t *t, float *s) {
    uint16_t h[4];
    memcpy(h, t, 8);
    for (int i = 0; i < 4; i++) s[i] = vq_f16(h[i]);
}
/* 半字节 → 符号扩展 int8（即 (nib-8)，等价 actq_w_from_r 但更少指令）：
 *   低半字节：左移 4 把符号位挪到 bit7、再算术右移 → 2 条；
 *   高半字节：对 int8 直接算术右移 4 位即完成符号扩展 → 1 条。
 * 纯整数位运算，输出与 actq_w_from_r 逐位相同（不改变任何浮点取值）。 */
static inline int8x16_t arm2_nib_lo(uint8x16_t R) {
    return vshrq_n_s8(vshlq_n_s8(vreinterpretq_s8_u8(R), 4), 4);
}
static inline int8x16_t arm2_nib_hi(uint8x16_t R) {
    return vshrq_n_s8(vreinterpretq_s8_u8(R), 4);
}
/* 72B tile 的 4 行各做一次 vqtbl 收集 + 半字节符号扩展 → 低/高半字节 int8。
 * 收集结果对该 tile 关联的全部 token 复用（x86 actq_gather_tile2 的 128-bit 版）。 */
static inline void arm2_rows4(const uint8_t *tile, const uint8x16_t tab[4][4],
                              int8x16_t lo[4], int8x16_t hi[4]) {
    const uint8_t *qs = tile + 8;
    uint8x16_t c0 = vld1q_u8(qs),      c1 = vld1q_u8(qs + 16);
    uint8x16_t c2 = vld1q_u8(qs + 32), c3 = vld1q_u8(qs + 48);
    for (int ri = 0; ri < 4; ri++) {
        uint8x16_t R = vorrq_u8(
            vorrq_u8(vorrq_u8(vqtbl1q_u8(c0, tab[0][ri]), vqtbl1q_u8(c1, tab[1][ri])),
                     vqtbl1q_u8(c2, tab[2][ri])), vqtbl1q_u8(c3, tab[3][ri]));
        lo[ri] = arm2_nib_lo(R);
        hi[ri] = arm2_nib_hi(R);
    }
}
/* gu：task = (行组 g, 专家 e)，行组主序（同 x86 2a-d）。tile 行收集每块一次，
 * 供该专家的 |T_e| 个 token 复用；f32 累加表达式与逐 token 核逐字同形。 */
static void arm2_gu_worker(void *c_, int idx) {
    arm2_ctx *c = c_;
    int g = idx / c->ne, e = idx - g * c->ne;
    int o0 = c->ex_off[e], o1 = c->ex_off[e + 1];
    int m = o1 - o0;
    if (m <= 0) return;
    const uint8_t *ge = c->G + (size_t)((size_t)e * c->ge4) * c->ggB
                             + (size_t)g * c->ggB;
    const uint8_t *ue = c->U + (size_t)((size_t)e * c->ge4) * c->ggB
                             + (size_t)g * c->ggB;
    actq_tab_ensure();
    uint8x16_t tab[4][4];
    for (int q = 0; q < 4; q++)
        for (int ri = 0; ri < 4; ri++) tab[q][ri] = vld1q_u8(actq_tab[q][ri]);
    /* 与 x86 st2_gu_worker 同因：k 的上界 m = 该专家的 (token,rank) 归属数，
     * 其上界是 prefill mini-batch 的 nb（g_st_prefill_batch 默认 256），不是 32。
     * 原 `float32x4_t gacc[32]` 在 nb>32 且专家热度集中时写越界（砸穿 worker 栈）；
     * 此前靠入口 `nb > 32` 护栏回避，代价是长 prefill 全部退回逐 token → 一并改为按 m 定界。 */
    float32x4_t gacc[m], uacc[m];
    for (int k = 0; k < m; k++) { gacc[k] = vdupq_n_f32(0.0f); uacc[k] = vdupq_n_f32(0.0f); }
    for (int cb = 0; cb < c->nbG; cb++) {
        const uint8_t *gt = ge + (size_t)cb * 72, *ut = ue + (size_t)cb * 72;
        float scg4[4], scu4[4];
        int8x16_t glo[4], ghi[4], ulo[4], uhi[4];
        arm2_scales4(gt, scg4); arm2_scales4(ut, scu4);
        arm2_rows4(gt, tab, glo, ghi);
        arm2_rows4(ut, tab, ulo, uhi);
        float32x4_t scg = vld1q_f32(scg4), scu = vld1q_f32(scu4);
        for (int k = 0; k < m; k++) {
            int t = c->ex_t[o0 + k];
            const int8_t *xs = c->xq + (size_t)t * c->d + (size_t)cb * 32;
            int8x16_t xl = vld1q_s8(xs), xh = vld1q_s8(xs + 16);
            float32x4_t xdv = vdupq_n_f32(c->xd[(size_t)t * c->nbG + cb]);
            /* 4 行点积用 3 条 vpaddq 归约成 [行0,行1,行2,行3] 的 int32x4，
             * 替代「4×vaddvq + 标量组回向量」；整数加结合律下结果逐位不变。 */
            int32x4_t gd0 = vaddq_s32(i8x16_dot_s32(glo[0], xl), i8x16_dot_s32(ghi[0], xh));
            int32x4_t gd1 = vaddq_s32(i8x16_dot_s32(glo[1], xl), i8x16_dot_s32(ghi[1], xh));
            int32x4_t gd2 = vaddq_s32(i8x16_dot_s32(glo[2], xl), i8x16_dot_s32(ghi[2], xh));
            int32x4_t gd3 = vaddq_s32(i8x16_dot_s32(glo[3], xl), i8x16_dot_s32(ghi[3], xh));
            int32x4_t ud0 = vaddq_s32(i8x16_dot_s32(ulo[0], xl), i8x16_dot_s32(uhi[0], xh));
            int32x4_t ud1 = vaddq_s32(i8x16_dot_s32(ulo[1], xl), i8x16_dot_s32(uhi[1], xh));
            int32x4_t ud2 = vaddq_s32(i8x16_dot_s32(ulo[2], xl), i8x16_dot_s32(uhi[2], xh));
            int32x4_t ud3 = vaddq_s32(i8x16_dot_s32(ulo[3], xl), i8x16_dot_s32(uhi[3], xh));
            int32x4_t pgv = vpaddq_s32(vpaddq_s32(gd0, gd1), vpaddq_s32(gd2, gd3));
            int32x4_t puv = vpaddq_s32(vpaddq_s32(ud0, ud1), vpaddq_s32(ud2, ud3));
            gacc[k] = vaddq_f32(gacc[k], vmulq_f32(vmulq_f32(scg, xdv), vcvtq_f32_s32(pgv)));
            uacc[k] = vaddq_f32(uacc[k], vmulq_f32(vmulq_f32(scu, xdv), vcvtq_f32_s32(puv)));
        }
    }
    for (int k = 0; k < m; k++) {
        int t = c->ex_t[o0 + k], j = c->ex_j[o0 + k];
        float gv[4], uv[4];
        vst1q_f32(gv, gacc[k]); vst1q_f32(uv, uacc[k]);
        float *av = c->av + ((size_t)t * c->tk + j) * c->ef + (size_t)g * 4;
        for (int ri = 0; ri < 4; ri++) {
            float gg = gv[ri], uu = uv[ri];
            av[ri] = (gg / (1.0f + expf(-gg))) * uu;   /* silu(g)*u，同逐 token 核 */
        }
    }
}
/* down：task = (行组 rg, 专家 e)，行组主序。列带 tile 读一次供 |T_e| 个 token 复用；
 * 存**未加权** acc → combine 内与 p 做 FMA（复刻参考核 vaddq(yo, vmulq(p, acc))）。 */
/* 精确轨（actq=0）gu：wf 预计算（解包+×scale）每块一次、供 |T_e| 个 token 复用；
 * 逐 k f32 乘加序与 q4x4_tile32_neon 逐位一致（vmul+vadd 不收缩）。 */
static void arm2_gu_worker_exact(void *c_, int idx) {
    arm2_ctx *c = c_;
    int g = idx / c->ne, e = idx - g * c->ne;
    int o0 = c->ex_off[e], o1 = c->ex_off[e + 1];
    int m = o1 - o0;
    if (m <= 0) return;
    const uint8_t *ge = c->G + (size_t)((size_t)e * c->ge4) * c->ggB
                             + (size_t)g * c->ggB;
    const uint8_t *ue = c->U + (size_t)((size_t)e * c->ge4) * c->ggB
                             + (size_t)g * c->ggB;
    float32x4_t gacc[m], uacc[m];
    for (int k = 0; k < m; k++) { gacc[k] = vdupq_n_f32(0.0f); uacc[k] = vdupq_n_f32(0.0f); }
    for (int cb = 0; cb < c->nbG; cb++) {
        const uint8_t *gt = ge + (size_t)cb * 72, *ut = ue + (size_t)cb * 72;
        float32x4_t wfg[32], wfu[32];
        q4x4_tile32_wf_neon(gt, wfg);
        q4x4_tile32_wf_neon(ut, wfu);
        for (int k = 0; k < m; k++) {
            int t = c->ex_t[o0 + k];
            const float *x = c->X + (size_t)t * c->d + (size_t)cb * 32;
            for (int kk = 0; kk < 32; kk++) {
                float32x4_t xv = vdupq_n_f32(x[kk]);
                gacc[k] = vaddq_f32(gacc[k], vmulq_f32(wfg[kk], xv));
                uacc[k] = vaddq_f32(uacc[k], vmulq_f32(wfu[kk], xv));
            }
        }
    }
    for (int k = 0; k < m; k++) {
        int t = c->ex_t[o0 + k], j = c->ex_j[o0 + k];
        float gv[4], uv[4];
        vst1q_f32(gv, gacc[k]); vst1q_f32(uv, uacc[k]);
        float *av = c->av + ((size_t)t * c->tk + j) * c->ef + (size_t)g * 4;
        for (int ri = 0; ri < 4; ri++) {
            float gg = gv[ri], uu = uv[ri];
            av[ri] = (gg / (1.0f + expf(-gg))) * uu;   /* silu(g)*u，同逐 token 核 */
        }
    }
}

static void arm2_dn_worker(void *c_, int idx) {
    arm2_ctx *c = c_;
    int rg = idx / c->ne, e = idx - rg * c->ne;
    int o0 = c->ex_off[e], o1 = c->ex_off[e + 1];
    int m = o1 - o0;
    if (m <= 0) return;
    const uint8_t *db = c->D + (size_t)rg * (size_t)c->gdB
                              + (size_t)((e * c->ef) >> 5) * 72;
    actq_tab_ensure();
    uint8x16_t tab[4][4];
    for (int q = 0; q < 4; q++)
        for (int ri = 0; ri < 4; ri++) tab[q][ri] = vld1q_u8(actq_tab[q][ri]);
    float32x4_t acc[m];   /* 同上：按归属数 m 定界（原 [32] 在 nb>32 时越界） */
    for (int k = 0; k < m; k++) acc[k] = vdupq_n_f32(0.0f);
    for (int cbi = 0; cbi < c->cbe; cbi++) {
        const uint8_t *tile = db + (size_t)cbi * 72;
        float sd4[4];
        int8x16_t lo[4], hi[4];
        arm2_scales4(tile, sd4);
        arm2_rows4(tile, tab, lo, hi);
        float32x4_t sd = vld1q_f32(sd4);
        for (int k = 0; k < m; k++) {
            int t = c->ex_t[o0 + k], j = c->ex_j[o0 + k];
            size_t s = (size_t)t * c->tk + j;
            const int8_t *avq = c->avq + s * c->ef + (size_t)cbi * 32;
            int8x16_t xl = vld1q_s8(avq), xh = vld1q_s8(avq + 16);
            float32x4_t xdv = vdupq_n_f32(c->avd[s * c->cbe + cbi]);
            /* 同 gu：vpaddq 归约 4 行点积（整数域逐位不变） */
            int32x4_t d0 = vaddq_s32(i8x16_dot_s32(lo[0], xl), i8x16_dot_s32(hi[0], xh));
            int32x4_t d1 = vaddq_s32(i8x16_dot_s32(lo[1], xl), i8x16_dot_s32(hi[1], xh));
            int32x4_t d2 = vaddq_s32(i8x16_dot_s32(lo[2], xl), i8x16_dot_s32(hi[2], xh));
            int32x4_t d3 = vaddq_s32(i8x16_dot_s32(lo[3], xl), i8x16_dot_s32(hi[3], xh));
            int32x4_t ps = vpaddq_s32(vpaddq_s32(d0, d1), vpaddq_s32(d2, d3));
            acc[k] = vaddq_f32(acc[k], vmulq_f32(vmulq_f32(sd, xdv), vcvtq_f32_s32(ps)));
        }
    }
    for (int k = 0; k < m; k++) {
        int t = c->ex_t[o0 + k], j = c->ex_j[o0 + k];
        size_t s = (size_t)t * c->tk + j;
        vst1q_f32(c->P + s * c->d + (size_t)rg * 4, acc[k]);
    }
}
/* 精确轨（actq=0）dn：wf 预计算跨 token 复用；存未加权 acc → combine 内 FMA。 */
static void arm2_dn_worker_exact(void *c_, int idx) {
    arm2_ctx *c = c_;
    int rg = idx / c->ne, e = idx - rg * c->ne;
    int o0 = c->ex_off[e], o1 = c->ex_off[e + 1];
    int m = o1 - o0;
    if (m <= 0) return;
    const uint8_t *db = c->D + (size_t)rg * (size_t)c->gdB
                              + (size_t)((e * c->ef) >> 5) * 72;
    float32x4_t acc[m];
    for (int k = 0; k < m; k++) acc[k] = vdupq_n_f32(0.0f);
    for (int cbi = 0; cbi < c->cbe; cbi++) {
        const uint8_t *tile = db + (size_t)cbi * 72;
        float32x4_t wfd[32];
        q4x4_tile32_wf_neon(tile, wfd);
        for (int k = 0; k < m; k++) {
            int t = c->ex_t[o0 + k], j = c->ex_j[o0 + k];
            size_t s = (size_t)t * c->tk + j;
            const float *av = c->av + s * c->ef + (size_t)cbi * 32;
            for (int kk = 0; kk < 32; kk++) {
                float32x4_t avv = vdupq_n_f32(av[kk]);
                acc[k] = vaddq_f32(acc[k], vmulq_f32(wfd[kk], avv));
            }
        }
    }
    for (int k = 0; k < m; k++) {
        int t = c->ex_t[o0 + k], j = c->ex_j[o0 + k];
        size_t s = (size_t)t * c->tk + j;
        vst1q_f32(c->P + s * c->d + (size_t)rg * 4, acc[k]);
    }
}

/* 合并：按 token 自身 rank 序把 partial FMA 进 Y（Y 起点 0 → 与逐 token 逐位同）。
 * 表达式与 moe_q4_down_group_actq_neon 末步 vaddq(yo, vmulq(p, acc)) 同形。 */
static void arm2_combine_worker(void *c_, int t) {
    arm2_ctx *c = c_;
    float *y = c->Y + (size_t)t * c->d;
    for (int j = 0; j < c->tk; j++) {
        size_t s = (size_t)t * c->tk + j;
        if (c->sel[s] < 0) continue;
        const float *P = c->P + s * c->d;
        float32x4_t pv = vdupq_n_f32(c->pr[s]);
        for (int i = 0; i < c->d; i += 4)
            vst1q_f32(y + i, vaddq_f32(vld1q_f32(y + i), vmulq_f32(pv, vld1q_f32(P + i))));
    }
}
/* av f32 → 每 (t,j) q8（与逐 token 路径同式） */
static void arm2_quant_worker(void *c_, int k) {
    arm2_ctx *c = c_;
    quantize_row_q8_0_act(c->av + (size_t)k * c->ef, c->avq + (size_t)k * c->ef,
                          c->avd + (size_t)k * c->cbe, c->ef);
}
/* 返回 1=已处理，0=不适用（调用方回落逐 token）。 */
static int st_moe_ffn_sparse_q4_batch_arm(const STModelWeights *w, int l, int nb,
                                          const float *X, float *Y) {
    const STModelConfig *c = &w->cfg;
    int ne = c->n_experts, tk = c->top_k, d = c->dim, ff = c->ffn_dim;
    if (ne <= 0 || tk <= 0 || tk > 32 || ne > 4096 || d <= 0 || nb < 2) return 0;
    /* 注：不再限制 nb<=32 —— 累加器已按归属数 m 的 VLA 定界（见 arm2_gu/dn_worker），
     * 长 prefill（nb 可达 256）同样走批式路径。 */
    if (!arm2_batch_env() || !w->q4_gate_weight) return 0;
    int ef = ff / ne;
    if (ef <= 0 || ne * ef != ff || (ef & 3) || (d & 31) || ef > 2048) return 0;
    int nbG = d >> 5, ggB = nbG * 72, gdB = (ff >> 5) * 72, ge4 = ef >> 2;
    int off_l = st_ffn_q4_layer_off(w, l);
    const uint8_t *G = w->q4_gate_weight + Q4_BYTES((size_t)off_l * (size_t)ff * d);
    const uint8_t *U = w->q4_up_weight   + Q4_BYTES((size_t)off_l * (size_t)ff * d);
    const uint8_t *D = w->q4_down_weight + Q4_BYTES((size_t)off_l * (size_t)d * ff);
    if (!G || !U || !D) return 0;
    actq_tab_ensure();
    size_t nsel = (size_t)nb * tk;
    int8_t *xq  = (int8_t *)malloc((size_t)nb * d);
    float  *xd  = (float *)malloc((size_t)nb * nbG * sizeof(float));
    int8_t *avq = (int8_t *)malloc(nsel * ef);
    float  *avd = (float *)malloc(nsel * (ef >> 5) * sizeof(float));
    float  *av  = (float *)malloc(nsel * (size_t)ef * sizeof(float));
    float  *P   = (float *)malloc(nsel * (size_t)d * sizeof(float));
    float  *rlg = (float *)malloc((size_t)nb * ne * sizeof(float));
    int    *sel = (int *)malloc(nsel * sizeof(int));
    float  *pr  = (float *)malloc(nsel * sizeof(float));
    int    *ex_off = (int *)malloc(((size_t)ne + 1) * sizeof(int));
    int    *ex_t = (int *)malloc(nsel * sizeof(int));
    int    *ex_j = (int *)malloc(nsel * sizeof(int));
    int    *used = (int *)malloc((size_t)ne * sizeof(int));
    if (!xq || !xd || !avq || !avd || !av || !P || !sel || !pr ||
        !ex_off || !ex_t || !ex_j || !used || !rlg) {
        free(xq); free(xd); free(avq); free(avd); free(av); free(P);
        free(sel); free(pr); free(ex_off); free(ex_t); free(ex_j); free(used);
        free(rlg);
        return 0;
    }
    memset(Y, 0, (size_t)nb * d * sizeof(float));
    double t_entry = st_now_sec();   /* ARM：moe_now 仅 x86 区定义，用平台头时钟 */
    const float *router = w->moe_router + (size_t)l * ne * d;
    /* 批式 router（同 x86 2a-e）：专家外层 → 权重每层只读一次；token 内层 4 路展开，
     * 每条链内部 i 升序、运算形式与逐 token 路径逐字同形 → per-(t,e) 位级同值。 */
    for (int e = 0; e < ne; e++) {
        const float *rw = router + (size_t)e * d;
        int t = 0;
        for (; t + 4 <= nb; t += 4) {
            const float *x0 = X + (size_t)(t + 0) * d;
            const float *x1 = X + (size_t)(t + 1) * d;
            const float *x2 = X + (size_t)(t + 2) * d;
            const float *x3 = X + (size_t)(t + 3) * d;
            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
            for (int i = 0; i < d; i++) {
                float r = rw[i];
                s0 += r * x0[i]; s1 += r * x1[i];
                s2 += r * x2[i]; s3 += r * x3[i];
            }
            rlg[(size_t)(t + 0) * ne + e] = s0;
            rlg[(size_t)(t + 1) * ne + e] = s1;
            rlg[(size_t)(t + 2) * ne + e] = s2;
            rlg[(size_t)(t + 3) * ne + e] = s3;
        }
        for (; t < nb; t++) {
            const float *xt = X + (size_t)t * d;
            float s = 0.0f;
            for (int i = 0; i < d; i++) s += rw[i] * xt[i];
            rlg[(size_t)t * ne + e] = s;
        }
    }
    for (int t = 0; t < nb; t++) {
        const float *xt = X + (size_t)t * d;
        quantize_row_q8_0_act(xt, xq + (size_t)t * d, xd + (size_t)t * nbG, d);
        /* 确定性 top-k + softmax（与逐 token 路径同式） */
        const float *lg = rlg + (size_t)t * ne;
        float sv[32];
        for (int k = 0; k < tk; k++) { sel[(size_t)t*tk+k] = -1; sv[k] = -INFINITY; }
        for (int e = 0; e < ne; e++) {
            for (int k = 0; k < tk; k++) {
                if (lg[e] > sv[k] || (lg[e] == sv[k] &&
                    (sel[(size_t)t*tk+k] < 0 || e < sel[(size_t)t*tk+k]))) {
                    for (int q = tk - 1; q > k; q--) {
                        sel[(size_t)t*tk+q] = sel[(size_t)t*tk+q-1];
                        sv[q] = sv[q-1];
                    }
                    sel[(size_t)t*tk+k] = e; sv[k] = lg[e];
                    break;
                }
            }
        }
        float mx = -INFINITY;
        for (int k = 0; k < tk; k++) if (sv[k] > mx) mx = sv[k];
        float sum = 0.0f;
        for (int k = 0; k < tk; k++) { pr[(size_t)t*tk+k] = expf(sv[k] - mx); sum += pr[(size_t)t*tk+k]; }
        float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
        for (int k = 0; k < tk; k++) pr[(size_t)t*tk+k] *= inv;
    }
    /* 与逐 token 路径同源的专家预取（只动页表/页缓存，浮点零改动） */
    for (int t = 0; t < nb; t++) vqf_ffn_prefetch(w, l, sel + (size_t)t * tk, tk);
    arm2_ctx ctx;
    ctx.G = G; ctx.U = U; ctx.D = D; ctx.X = X; ctx.Y = Y;
    ctx.xq = xq; ctx.xd = xd; ctx.avq = avq; ctx.avd = avd; ctx.av = av; ctx.P = P;
    ctx.ex_off = ex_off; ctx.ex_t = ex_t; ctx.ex_j = ex_j;
    ctx.sel = sel; ctx.pr = pr; ctx.ne = ne; ctx.nsel = (int)nsel;
    ctx.nb = nb; ctx.tk = tk; ctx.d = d; ctx.ef = ef; ctx.nbG = nbG;
    ctx.ggB = ggB; ctx.gdB = gdB; ctx.ge4 = ge4; ctx.cbe = ef >> 5;
    ctx.verify = (getenv("VLLM_MOE_2B_VERIFY") && getenv("VLLM_MOE_2B_VERIFY")[0] == '1') ? 1 : 0;
    ctx.actq = moe_q4_actq_ok();
    /* 专家→(token,rank) 归属表（确定性：按 token/rank 升序填） */
    memset(ex_off, 0, ((size_t)ne + 1) * sizeof(int));
    for (size_t k = 0; k < nsel; k++) {
        int e = sel[k];
        if (e >= 0 && e < ne) ex_off[e + 1]++;
    }
    for (int e = 0; e < ne; e++) ex_off[e + 1] += ex_off[e];
    memset(used, 0, (size_t)ne * sizeof(int));
    for (int t = 0; t < nb; t++)
        for (int j = 0; j < tk; j++) {
            int e = sel[(size_t)t * tk + j];
            if (e < 0 || e >= ne) continue;
            int pos = ex_off[e] + used[e]++;
            ex_t[pos] = t; ex_j[pos] = j;
        }
    memset(av, 0, nsel * (size_t)ef * sizeof(float));
    int seg = (getenv("VLLM_MOE_2B_SEG") && getenv("VLLM_MOE_2B_SEG")[0] == '1') ? 1 : 0;
    double s0 = seg ? st_now_sec() : 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0;
    vllm_tp_parfor(0, ne * ge4, ctx.actq ? arm2_gu_worker : arm2_gu_worker_exact, &ctx);
    if (seg) s1 = st_now_sec();
    if (ctx.actq) {
        vllm_tp_parfor(0, (int)nsel, arm2_quant_worker, &ctx);
    }
    if (seg) s2 = st_now_sec();
    vllm_tp_parfor(0, ne * (d >> 2), ctx.actq ? arm2_dn_worker : arm2_dn_worker_exact, &ctx);
    if (seg) s3 = st_now_sec();
    vllm_tp_parfor(0, nb, arm2_combine_worker, &ctx);
    if (seg) {
        s4 = st_now_sec();
        printf("[2BSEG-ARM] l=%d tok=%.2f gu=%.2f q=%.2f dn=%.2f cmb=%.2f (ms)\n",
               l, (s0 - t_entry) * 1e3, (s1 - s0) * 1e3, (s2 - s1) * 1e3,
               (s3 - s2) * 1e3, (s4 - s3) * 1e3);
        fflush(stdout);
    }
    if (ctx.verify) {   /* 端到端：批 MoE Y vs 逐 token 路径（同场对拍） */
        float *yt = (float *)malloc((size_t)d * sizeof(float));
        if (yt) {
            for (int t = 0; t < nb; t++) {
                int bad = 0; float maxd = 0.0f;
                memset(yt, 0, (size_t)d * sizeof(float));
                st_moe_ffn_sparse_q4(w, l, X + (size_t)t * d, yt);
                for (int i = 0; i < d; i++) {
                    float a = Y[(size_t)t * d + i], b = yt[i];
                    if (a != b) {
                        float df = a > b ? a - b : b - a;
                        if (df > maxd) maxd = df;
                        bad++;
                    }
                }
                printf("[2BVERIFY-ARM] E2E l=%d t=%d diffEl=%d/%d maxAbs=%.6g\n",
                       l, t, bad, d, maxd);
            }
            fflush(stdout);
            free(yt);
        }
    }
    free(xq); free(xd); free(avq); free(avd); free(av); free(P);
    free(sel); free(pr); free(ex_off); free(ex_t); free(ex_j); free(used);
    free(rlg);
    return 1;
}
#endif /* __aarch64__ && ST_HAVE_NEON && !ST_ARCH_X86 && ST_NEON_DOTPROD */

static void st_moe_ffn_sparse(const STModelWeights *w, int l,
                              const float *x_ffn, float *y) {
    int mtt = 0;
    double t0 = 0;
#if defined(__AVX2__) && ST_ARCH_X86
    mtt = moe_ffn_trace_env();
    t0 = mtt ? moe_now() : 0;
    if (mtt && l == 0 && mft_nc > 0) {   /* 上一 token 完整层后打印 */
        printf("[MOETOK] q4=%d q8=%d ffn=%.2fms maxL%d=%.3fms\n",
               mft_q4, mft_q8, mft_ff, mft_maxl, mft_max);
        fflush(stdout);
        mft_ff = 0; mft_max = 0; mft_nc = 0; mft_q4 = 0; mft_q8 = 0; mft_maxl = -1;
    }
#endif
    /* A 路径：无 q4 专家（wmode=q8 纯 q8_0 文件）→ q8 稀疏解码 */
    if (w->q4_gate_weight == NULL && w->q8_gate_weight != NULL) {
#if defined(__AVX2__) && ST_ARCH_X86
        if (mtt) mft_q8++;
#endif
        st_moe_ffn_sparse_q8(w, l, x_ffn, y);
    } else {
        /* q4 专家路径（或 q8+q4 双文件：q4_gate_weight 优先） */
#if defined(__AVX2__) && ST_ARCH_X86
        if (mtt) mft_q4++;
#endif
        st_moe_ffn_sparse_q4(w, l, x_ffn, y);
    }
#if defined(__AVX2__) && ST_ARCH_X86
    if (mtt) {
        double dt = (moe_now() - t0) * 1e3;   /* ms */
        mft_ff += dt; mft_nc++;
        if (dt > mft_max) { mft_max = dt; mft_maxl = l; }
    }
#endif
}

/* ---- M3 layer-0 对拍 dump（VLLM_MOE_DUMP=1，单 token decode，l==0） ----
 * VLLM_MOE_DUMP_L=<n|->1>：仅 n 层 / 全层打印（默认 0 保持原语义）。 */
static int moe_dump_l(void) {
    static int l = -1;
    if (l < 0) {
        const char *e = getenv("VLLM_MOE_DUMP_L");
        l = e ? atoi(e) : 0;
        if (l < -1) l = -1;
    }
    return l;
}
static int moe_dump_at(int l) {
    int L = moe_dump_l();
    return (L < 0) || (l == L);
}
static void moe_l0_dump(const char *tag, const float *v, int n) {
    static int on = -1;
    static int bits = -1;
    if (on < 0) on = (getenv("VLLM_MOE_DUMP") && getenv("VLLM_MOE_DUMP")[0] == '1') ? 1 : 0;
    if (bits < 0) bits = (getenv("VLLM_MOE_DUMP_BITS") && getenv("VLLM_MOE_DUMP_BITS")[0] == '1') ? 1 : 0;
    if (!on) return;
    printf("[L0D] %s", tag);
    for (int i = 0; i < n && i < 16; i++) {
        if (bits) {
            uint32_t u;
            memcpy(&u, &v[i], 4);
            printf(" %08x", u);
        } else {
            printf(" %.9e", (double)v[i]);
        }
    }
    printf("\n");
    fflush(stdout);
}

/* ================================================================
 * MoE 专家并行（EP）· 阶段一（同机验证）：专家段一等化（M1）+ 虚拟 rank（M2）
 * 方案锚点：资料/分布式专家提取_实施方案.md（2026-09-12）
 *
 * M1 —— 专家段一等化：把 (层 l, 专家 e) 映射为可独立寻址/分发的字节几何
 *   gate/up：连续段 [张量基址 + l*layerB + e*expB, +expB)，expB 整页对齐；
 *   down   ：列带（每 4 行组内 cbe*72 字节连续，行组步长 gdB）。
 *   校验用 VQF 目录元数据独立进行（目录 offset ↔ 引擎直挂指针、rows ↔ nl*ff、
 *   bytes ↔ nl*layerB、expB 整页），避免"用同一公式自证"。
 *
 * M2 —— 虚拟 rank 位级一致：专家按 rank 均分（连续块），各 rank 只算"自己
 *   拥有的被选专家"，产出**完整 d 维贡献** C_j = p_j·down_ej(av_j)；协调者再
 *   按 j 升序累加。A≡C 的三个不变量：
 *     ① C_j 复用与单机路径**完全相同的内核**（moe_q4_par_worker /
 *        m4_dn_f32_worker / moe_q4_down_group_neon），非重新实现；
 *     ② C_j 由 0 起累加（0 + p·acc），与单机 y(初值 0) += p·acc 逐位同值；
 *     ③ 最终按 j 升序 y += C_j，与单机 `for j: y += p·down_j` 逐步同序。
 *
 * 只读验证模块：不进入生产路径，st_moe_ffn_sparse_q4 零改动。
 * 边界：仅明文 VQF、仅 q4_4x4（本阶段）、仅 f32 精确轨。
 * ================================================================ */
typedef struct {
    int      ok;                 /* 1 = 几何成立且已交叉校验通过 */
    int      is_q8;              /* 1 = q8_8x8；0 = q4_4x4 */
    int      ne, me, d, ff, nl, tk;
    uint64_t layerB;             /* 每层字节（gate/up/down 同几何） */
    uint64_t expB;               /* 每专家 gate/up 连续段字节（整页对齐） */
    const uint8_t *g, *u, *dn;   /* 张量数据起始（mmap 直挂） */
    uint64_t g_off, u_off, d_off;/* 目录声明文件偏移（交叉校验用） */
    int      gdB;                /* down：4 行组的字节跨度 */
    int      cbe;                /* down：每专家列块数（me/32） */
} MoeEpGeom;

static uint64_t moe_ep_qt_bytes(int is_q8, uint64_t n) {
    return is_q8 ? (uint64_t)Q8_BYTES(n) : (uint64_t)Q4_BYTES(n);
}

/* VQF 目录查找（与 vqf.c 同布局：头 448B 后紧跟 VQFTensor 数组）。 */
static const VQFTensor *moe_ep_dir_find(const VQFHeader *h, const char *want) {
    size_t dir_off = (sizeof(VQFHeader) + 63) & ~(size_t)63;
    const VQFTensor *dir = (const VQFTensor *)((const uint8_t *)h + dir_off);
    for (uint32_t i = 0; i < h->n_tensors; i++)
        if (strcmp(dir[i].name, want) == 0) return &dir[i];
    return NULL;
}

/* M1：解析并交叉校验专家段几何。返回 1 = 可用。 */
static int moe_ep_geom_init(MoeEpGeom *G, const STModelWeights *w) {
    memset(G, 0, sizeof(*G));
    if (!w || !w->vqf_map || w->vqf_map_len < sizeof(VQFHeader)) return 0;
    if (!w->cfg.is_moe) return 0;
    const VQFHeader *h = (const VQFHeader *)w->vqf_map;
    if (h->flags & VQF_FLAG_ENC) return 0;       /* 加密 VQF：专家段不可寻址 */
    const VQFArch *a = &h->arch;
    int ne = (int)a->n_experts, me = (int)a->moe_ffn, d = (int)a->dim;
    int ff = (int)a->ffn_dim, nl = (int)a->n_layers, tk = (int)a->top_k;
    if (ne <= 0 || me <= 0 || d <= 0 || ff <= 0 || nl <= 0 || tk <= 0) return 0;
    if ((uint64_t)ne * (uint64_t)me != (uint64_t)ff) return 0;
    /* 行组（4 行）与 32 列块都不跨专家段 —— 切分对齐前提 */
    if ((me & 3) || (me & 31) || (d & 31)) return 0;

    const VQFTensor *TG = moe_ep_dir_find(h, "q4_gate");
    const VQFTensor *TU = moe_ep_dir_find(h, "q4_up");
    const VQFTensor *TD = moe_ep_dir_find(h, "q4_down");
    int is_q8 = 0;
    if (!TG || !TU || !TD) {
        TG = moe_ep_dir_find(h, "q8_gate");
        TU = moe_ep_dir_find(h, "q8_up");
        TD = moe_ep_dir_find(h, "q8_down");
        is_q8 = 1;
    }
    if (!TG || !TU || !TD) return 0;

    uint64_t layerB = moe_ep_qt_bytes(is_q8, (uint64_t)ff * (uint64_t)d);
    uint64_t expB   = moe_ep_qt_bytes(is_q8, (uint64_t)me * (uint64_t)d);
    if (expB == 0 || (expB & 4095) != 0) return 0;      /* 整页对齐 = 可分页/分发 */
    if ((uint64_t)TG->rows != (uint64_t)nl * (uint64_t)ff) return 0;
    if ((uint64_t)TD->rows != (uint64_t)nl * (uint64_t)d) return 0;
    if ((uint64_t)TG->bytes != layerB * (uint64_t)nl) return 0;
    if ((uint64_t)TU->bytes != layerB * (uint64_t)nl) return 0;
    if ((uint64_t)TD->bytes != layerB * (uint64_t)nl) return 0;

    /* 独立交叉校验：目录声明 offset ↔ 引擎 mmap 直挂指针 */
    const uint8_t *map = (const uint8_t *)w->vqf_map;
    if (!is_q8 && w->q4_gate_weight && w->q4_up_weight && w->q4_down_weight) {
        if ((const uint8_t *)w->q4_gate_weight != map + TG->offset) return 0;
        if ((const uint8_t *)w->q4_up_weight   != map + TU->offset) return 0;
        if ((const uint8_t *)w->q4_down_weight != map + TD->offset) return 0;
    } else if (is_q8 && w->q8_gate_weight && w->q8_up_weight && w->q8_down_weight) {
        if ((const uint8_t *)w->q8_gate_weight != map + TG->offset) return 0;
        if ((const uint8_t *)w->q8_up_weight   != map + TU->offset) return 0;
        if ((const uint8_t *)w->q8_down_weight != map + TD->offset) return 0;
    } else return 0;

    G->ok = 1; G->is_q8 = is_q8;
    G->ne = ne; G->me = me; G->d = d; G->ff = ff; G->nl = nl; G->tk = tk;
    G->layerB = layerB; G->expB = expB;
    G->g_off = TG->offset; G->u_off = TU->offset; G->d_off = TD->offset;
    G->g  = is_q8 ? w->q8_gate_weight : w->q4_gate_weight;
    G->u  = is_q8 ? w->q8_up_weight   : w->q4_up_weight;
    G->dn = is_q8 ? w->q8_down_weight : w->q4_down_weight;
    G->gdB = (ff >> 5) * 72;
    G->cbe = me >> 5;
    return 1;
}

/* M1 段视图：专家 (l,e) 的 gate / up 连续段指针。 */
static const uint8_t *moe_ep_gate_seg(const MoeEpGeom *G, int l, int e) {
    return G->g + (size_t)l * G->layerB + (size_t)e * G->expB;
}
static const uint8_t *moe_ep_up_seg(const MoeEpGeom *G, int l, int e) {
    return G->u + (size_t)l * G->layerB + (size_t)e * G->expB;
}
/* M1 段视图：专家 (l,e) 的 down 列带（非连续）在单个 4 行组内的起始偏移与
 * 连续长度；行组步长 = G->gdB，行组数 = d/4。 */
static void moe_ep_down_band(const MoeEpGeom *G, int l, int e,
                             size_t *rg_off, size_t *run) {
    int cb0 = (e * G->me) >> 5;
    *rg_off = (size_t)l * G->layerB + (size_t)cb0 * 72;
    *run    = (size_t)G->cbe * 72;
}

/* 确定性 router 选路（与生产路径 st_moe_ffn_sparse_q4 同码同序）：
 * logits = router·x → top-k（同分取小索引）→ 归一 softmax。 */
static void moe_ep_router(const STModelWeights *w, int l, const float *x_ffn,
                          int *sel, float *pr) {
    const STModelConfig *c = &w->cfg;
    int ne = c->n_experts, tk = c->top_k, d = c->dim;
    const float *router = w->moe_router + (size_t)l * ne * d;
    float lg[4096];
    for (int e = 0; e < ne; e++) {
        const float *rw = router + (size_t)e * d;
        float s = 0.0f;
        for (int i = 0; i < d; i++) s += rw[i] * x_ffn[i];
        lg[e] = s;
    }
    float sv[64];
    for (int k = 0; k < tk; k++) { sel[k] = -1; sv[k] = -INFINITY; }
    for (int e = 0; e < ne; e++) {
        for (int k = 0; k < tk; k++) {
            if (lg[e] > sv[k] || (lg[e] == sv[k] && (sel[k] < 0 || e < sel[k]))) {
                for (int j = tk - 1; j > k; j--) { sel[j] = sel[j - 1]; sv[j] = sv[j - 1]; }
                sel[k] = e; sv[k] = lg[e];
                break;
            }
        }
    }
    float mx = -INFINITY;
    for (int k = 0; k < tk; k++) if (sv[k] > mx) mx = sv[k];
    float sum = 0.0f;
    for (int k = 0; k < tk; k++) { pr[k] = expf(sv[k] - mx); sum += pr[k]; }
    float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    for (int k = 0; k < tk; k++) pr[k] *= inv;
}

/* 非对称专家分片（M5 实证）：split>0 且 N==2 时 rank0 拥 [0,split)、rank1 拥
 * [split,ne)；split<=0 回退均分。进程级设置，协调者与工作者两侧须设同一值。 */
static int g_moe_ep_split = 0;
void st_moe_ep_set_split(int split) { g_moe_ep_split = split; }

/* ================================================================
 * EPLB（Expert Parallelism Load Balancer，§9.25）
 *
 * 现状：专家→rank 是**静态**的（均分取模 / --ep-split 手工切分），既不按实测热度、
 * 也不计入各 rank 算力差异（跨机时 x86 协调者 vs ARM 工作者差 ~5.6×）→ 一侧空等。
 *
 * EPLB 改为按「热度 ÷ 算力」自动求**任意置换**：贪心 LPT（Longest Processing Time
 * first，按负载降序把每个专家分给当前归一化负载最小的 rank），使各 rank 的归一化
 * 完成时间趋同。
 *
 * A≡C 纪律：EPLB 只改「谁算」，不改「怎么算」——每个 C_j 仍由唯一 rank 用同一
 * 原始点积核产出，协调者仍按 j 升序归约 → 位级一致与静态分配时完全相同。
 * 跨机一致性：协调者在会话启动时建表并以 transport 广播，worker 原样安装。
 * ================================================================ */
#define MOE_EPLB_MAX_RANKS 16

static int      g_moe_eplb_on = 0;
static char     g_moe_eplb_heat[1024] = {0};
static double   g_moe_eplb_cap[MOE_EPLB_MAX_RANKS] = {0};
static int      g_moe_eplb_ncap = 0;
static uint8_t *g_moe_eplb_owner = NULL;   /* ne 项；非 NULL = 启用查表 */
static int      g_moe_eplb_ne = 0;

void st_moe_ep_eplb_config(int on, const char *heat_path, const double *cap, int ncap) {
    g_moe_eplb_on = on ? 1 : 0;
    g_moe_eplb_heat[0] = 0;
    if (heat_path && heat_path[0])
        snprintf(g_moe_eplb_heat, sizeof(g_moe_eplb_heat), "%s", heat_path);
    g_moe_eplb_ncap = 0;
    if (cap && ncap > 0) {
        if (ncap > MOE_EPLB_MAX_RANKS) ncap = MOE_EPLB_MAX_RANKS;
        for (int i = 0; i < ncap; i++) g_moe_eplb_cap[i] = (cap[i] > 0.0) ? cap[i] : 1.0;
        g_moe_eplb_ncap = ncap;
    }
}

int st_moe_ep_eplb_on(void) { return g_moe_eplb_on; }

void st_moe_ep_set_owner_tbl(const uint8_t *tbl, int ne) {
    free(g_moe_eplb_owner);
    g_moe_eplb_owner = NULL;
    g_moe_eplb_ne = 0;
    if (!tbl || ne <= 0) return;
    g_moe_eplb_owner = (uint8_t *)malloc((size_t)ne);
    if (!g_moe_eplb_owner) return;
    memcpy(g_moe_eplb_owner, tbl, (size_t)ne);
    g_moe_eplb_ne = ne;
}

/* 解析热度文件（"l:e:count" 每行，仅 count>0）→ 按专家跨层求和 load[ne]。
 * 缺文件/无有效项/总负载为 0 → 返回 -1（调用方退化为编号轮转）。 */
static int moe_eplb_load_heat(const char *path, double *load, int ne) {
    if (!path || !path[0]) return -1;
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "[MOE-EP-EPLB] 打不开热度文件 %s（退化编号轮转）\n", path);
        return -1;
    }
    for (int e = 0; e < ne; e++) load[e] = 0.0;
    char line[128];
    int n = 0;
    while (fgets(line, sizeof(line), f)) {
        int l, e; unsigned long long c;
        if (sscanf(line, "%d:%d:%llu", &l, &e, &c) != 3) continue;
        if (e < 0 || e >= ne || c == 0) continue;
        load[e] += (double)c;
        n++;
    }
    fclose(f);
    double tot = 0.0;
    for (int e = 0; e < ne; e++) tot += load[e];
    fprintf(stderr, "[MOE-EP-EPLB] 热度 %s：%d 项，总负载 %.0f\n", path, n, tot);
    return (n > 0 && tot > 0.0) ? 0 : -1;
}

/* 贪心 LPT 建表：按负载降序（并列按 e 升序，确定性），逐个专家分给
 * 「累计负载/算力」最小的 rank（并列取小 r）。返回 0 = 成功。 */
int st_moe_ep_eplb_build(int ne, int nranks, uint8_t *owner_out) {
    if (ne <= 0 || nranks <= 0 || nranks > MOE_EPLB_MAX_RANKS || !owner_out) return -1;
    if (nranks == 1) { for (int e = 0; e < ne; e++) owner_out[e] = 0; return 0; }
    double *load = (double *)malloc((size_t)ne * sizeof(double));
    int    *idx  = (int *)malloc((size_t)ne * sizeof(int));
    double *acc  = (double *)calloc((size_t)nranks, sizeof(double));
    double *tn   = (double *)calloc((size_t)nranks, sizeof(double));
    if (!load || !idx || !acc || !tn) { free(load); free(idx); free(acc); free(tn); return -1; }

    int have_heat = (moe_eplb_load_heat(g_moe_eplb_heat, load, ne) == 0);
    if (!have_heat) for (int e = 0; e < ne; e++) load[e] = 1.0;

    double cap[MOE_EPLB_MAX_RANKS];
    for (int r = 0; r < nranks; r++)
        cap[r] = (g_moe_eplb_ncap > 0 && r < g_moe_eplb_ncap) ? g_moe_eplb_cap[r] : 1.0;

    for (int e = 0; e < ne; e++) idx[e] = e;
    for (int i = 1; i < ne; i++) {                 /* 稳定插入排序，无库依赖 */
        int key = idx[i]; double kl = load[key];
        int j = i - 1;
        while (j >= 0 && (load[idx[j]] < kl || (load[idx[j]] == kl && idx[j] > key))) {
            idx[j + 1] = idx[j]; j--;
        }
        idx[j + 1] = key;
    }
    for (int i = 0; i < ne; i++) {
        int e = idx[i];
        int best = 0; double bestv = acc[0] / cap[0];
        for (int r = 1; r < nranks; r++) {
            double v = acc[r] / cap[r];
            if (v < bestv) { bestv = v; best = r; }
        }
        owner_out[e] = (uint8_t)best;
        acc[best] += load[e];
    }

    /* 均衡度报告：负载按算力归一化后的 max/mean 与 min/mean（1.000 = 完美均衡） */
    double sum_tn = 0.0;
    for (int r = 0; r < nranks; r++) { tn[r] = acc[r] / cap[r]; sum_tn += tn[r]; }
    double mean_tn = sum_tn / (double)nranks;
    double mx = 0.0, mn = 1e300;
    fprintf(stderr, "[MOE-EP-EPLB] 建表 ne=%d N=%d %s cap=", ne, nranks,
            have_heat ? "(按热度)" : "(无热度→编号轮转)");
    for (int r = 0; r < nranks; r++) fprintf(stderr, "%s%.3g", r ? "," : "", cap[r]);
    fprintf(stderr, " 各 rank 负载:");
    for (int r = 0; r < nranks; r++) {
        fprintf(stderr, " r%d=%.0f", r, acc[r]);
        if (tn[r] > mx) mx = tn[r];
        if (tn[r] < mn) mn = tn[r];
    }
    fprintf(stderr, "  归一化 max/mean=%.3f min/mean=%.3f\n",
            mean_tn > 0 ? mx / mean_tn : 0.0, mean_tn > 0 ? mn / mean_tn : 0.0);
    free(load); free(idx); free(acc); free(tn);
    return 0;
}

/* 专家 e 归属的 rank（-1 = 非法）。优先查 EPLB 表；未装表时回退静态
 * （--ep-split 非对称 / 均分取模），两者行为对旧路径零回归。 */
static int moe_ep_owner(int e, int ne, int nranks) {
    if (g_moe_eplb_owner && e >= 0 && e < g_moe_eplb_ne && e < ne) {
        int o = g_moe_eplb_owner[e];
        return (o < nranks) ? o : -1;
    }
    if (g_moe_ep_split > 0 && nranks == 2 && g_moe_ep_split < ne)
        return (e < g_moe_ep_split) ? 0 : 1;
    int per = ne / nranks;
    if (per <= 0) return -1;
    return e / per;
}

/* rank `rank`（共 nranks，ne % nranks == 0）拥有的被选专家 → 完整 d 维贡献
 * C_j = p_j·down_ej(av_j)，按 j 索引写入 Cbuf（未拥有的 j 不动）。
 * avacc 为 tk*ef 的暂存（worker 按 j 索引 → avacc[j*ef]）。返回本 rank 贡献个数。
 *
 * A≡C 关键：gu 走 moe_q4_par_worker、down 走 m4_dn_f32_worker /
 * moe_q4_down_group_neon —— 与单机路径**完全相同的内核**；C_j 由 0 起累加
 * （0 + p·acc），与单机 y(初值 0) += p·acc 逐位同值。 */
static int moe_ep_contrib(const STModelWeights *w, const MoeEpGeom *G, int l,
                          const float *x_ffn, const int *sel, const float *pr,
                          int rank, int nranks, float *avacc, float *Cbuf) {
    const STModelConfig *c = &w->cfg;
    int ne = c->n_experts, tk = c->top_k, d = c->dim, ff = c->ffn_dim;
    int ef = ff / ne;
    if (ef <= 0 || ne * ef != ff || (ef & 3) || (d & 31) || ef > 2048) return 0;
    if (nranks <= 0 || (ne % nranks) != 0) return 0;
    int off_l = st_ffn_q4_layer_off(w, l);
    const uint8_t *Gl = w->q4_gate_weight + Q4_BYTES((size_t)off_l * (size_t)ff * d);
    const uint8_t *Ul = w->q4_up_weight   + Q4_BYTES((size_t)off_l * (size_t)ff * d);
    const uint8_t *Dl = w->q4_down_weight + Q4_BYTES((size_t)off_l * (size_t)d * ff);
    if (!Gl || !Ul || !Dl) return 0;
    int nbG = d >> 5, ggB = nbG * 72, gdB = (ff >> 5) * 72, ge4 = ef >> 2;
    int cnt = 0;

    for (int j = 0; j < tk; j++) {
        int e = sel[j];
        if (e < 0 || e >= ne) continue;
        if (moe_ep_owner(e, ne, nranks) != rank) continue;   /* 非本 rank 的专家 */
        {
            moe_q4_par_ctx pc;
            pc.G = Gl; pc.U = Ul; pc.x_ffn = x_ffn;
            pc.xq = NULL; pc.xd = NULL;
            pc.xq16 = NULL; pc.xd16 = NULL;
            pc.av_accum = avacc;
            pc.ef = ef; pc.nbG = nbG; pc.ge4 = ge4; pc.ggB = ggB;
            pc.ne = ne; pc.tk = tk; pc.sel = sel;
            moe_q4_par_worker(&pc, j);
        }
        float *Cj = Cbuf + (size_t)j * d;
        memset(Cj, 0, (size_t)d * sizeof(float));    /* 贡献由 0 起累加 */
        int cb0 = (e * ef) >> 5;
        const float *av = avacc + (size_t)j * ef;
        (void)pr;   /* p 一律不在 contrib 施加（协议统一为「原始点积」），见 moe_ep_reduce */
#if defined(__AVX2__) && ST_ARCH_X86
        m4_dn_raw_ctx dc = { Dl, Cj, av, cb0, (ef >> 5), gdB, d >> 2 };
        vllm_tp_parfor(0, d >> 2, m4_dn_raw_worker, &dc);
#elif defined(__aarch64__) && defined(ST_HAVE_NEON) && !ST_ARCH_X86
        for (int rg = 0; rg < (d >> 2); rg++) {
            const uint8_t *db = Dl + (size_t)rg * gdB + (size_t)cb0 * 72;
            moe_q4_down_group_raw_neon(db, ef >> 5, av, Cj + (size_t)rg * 4);
        }
#else
        for (int rg = 0; rg < (d >> 2); rg++) {
            const uint8_t *db = Dl + (size_t)rg * gdB + (size_t)cb0 * 72;
            for (int ri = 0; ri < 4; ri++) {
                float acc = 0.0f; const float *ae = av;
                for (int cbi = 0; cbi < (ef >> 5); cbi++) {
                    const uint8_t *tile = db + (size_t)cbi * 72;
                    for (int k = 0; k < 32; k++, ae++) acc += vq4x4_w(tile, ri, k) * (*ae);
                }
                Cj[(size_t)rg * 4 + ri] = acc;   /* 原始点积（不含 p） */
            }
        }
#endif
        cnt++;
    }
    return cnt;
}

/* 贡献按 j 升序归约（与单机 `for j: y += p·down_j` 逐步同序；跳过非法 j）。
 * **协议**：`moe_ep_contrib` 在三平台上一律回传**原始点积**（不含 p），p 统一在此处
 * 施加——跨机时协调者与工作者是不同 ISA 的构建，若一侧乘 p、一侧不乘就会错。
 * 舍入按各 ISA 的生产核对齐：
 *   x86：生产核 `_mm_add_ps(yo,_mm_mul_ps(pv,acc))` 被 GCC 收缩成 FMA（**1 次舍入**）
 *        → 此处显式 `fmaf`。
 *   ARM：生产核是 `vaddq(yo, vmulq(pv,acc))`（**2 次舍入**），且板端构建带
 *        `-ffp-contract=off` → 此处 `y += p*Cj` 不会被收缩，同为 2 次舍入。 */
static void moe_ep_reduce(const float *Cbuf, const int *sel, const float *pr,
                          int tk, int d, float *y) {
    memset(y, 0, (size_t)d * sizeof(float));
    for (int j = 0; j < tk; j++) {
        if (sel[j] < 0) continue;
        const float *Cj = Cbuf + (size_t)j * d;
        float p = pr[j];
#if defined(__AVX2__) && ST_ARCH_X86
        for (int m = 0; m < d; m++) y[m] = fmaf(p, Cj[m], y[m]);
#else
        for (int m = 0; m < d; m++) y[m] += p * Cj[m];
#endif
    }
}

/* M2：EP 虚拟 rank 执行器（q4_4x4，f32 精确轨）。
 * nranks 均分 ne（要求 ne % nranks == 0）。返回 0 = 成功计算。 */
static int moe_ffn_sparse_q4_ep_vrank(const STModelWeights *w, const MoeEpGeom *G,
                                      int l, const float *x_ffn, float *y,
                                      int nranks) {
    const STModelConfig *c = &w->cfg;
    int ne = c->n_experts, tk = c->top_k, d = c->dim, ff = c->ffn_dim;
    if (G->is_q8) return 1;                       /* 本阶段仅 q4_4x4 */
    if (ne <= 0 || tk <= 0 || tk > 64 || ne > 4096 || d <= 0) return 1;
    int ef = ff / ne;
    if (ef <= 0 || ne * ef != ff || (ef & 3) || (d & 31) || ef > 2048) return 1;
    if (nranks <= 1 || nranks > ne || (ne % nranks) != 0) return 1;

    int sel[64]; float pr[64];
    moe_ep_router(w, l, x_ffn, sel, pr);

    float *avacc = (float *)malloc((size_t)tk * (size_t)ef * sizeof(float));
    float *Cbuf  = (float *)malloc((size_t)tk * (size_t)d * sizeof(float));
    if (!avacc || !Cbuf) { free(avacc); free(Cbuf); return 1; }

    /* 各 rank 只算自己拥有的被选专家（复用单机内核 → A≡C） */
    for (int r = 0; r < nranks; r++)
        moe_ep_contrib(w, G, l, x_ffn, sel, pr, r, nranks, avacc, Cbuf);

    /* 协调者按 j 升序归约（与单机 y += p·down_j 逐步同序） */
    moe_ep_reduce(Cbuf, sel, pr, tk, d, y);

    free(avacc); free(Cbuf);
    return 0;
}

/* 阶段一自检驱动：M1 几何交叉校验 + M2 虚拟 rank 位级一致。
 * 判据：M2 每个 (层, N) 的 EP 输出与 st_moe_ffn_sparse_q4 输出逐位相同。 */
int st_moe_ep_selftest(STModelWeights *w, int nranks_max) {
    printf("\n=== [MOE-EP] 阶段一：专家段一等化(M1) + 虚拟 rank 位级一致(M2) ===\n");
    printf("[MOE-EP] 方案锚点：资料/分布式专家提取_实施方案.md\n");
    if (!w || !w->cfg.is_moe) { printf("[MOE-EP] [FAIL] 非 MoE 模型\n"); return 1; }
    /* 仅验证 f32 精确轨：ACTQ / ACTQ16 两平台都有（ACTQ16 是 x86-only 内核，
     * 但 ARM 侧有恒 0 桩，符号可见）；FUSE/DN_D2 是 x86-only（ARM 无此
     * 符号，须守卫，否则板端链接失败）。
     * 注意：ACTQ16 默认已开，正常由 main.c 的 EP 强制关闭；此处兜底——
     * 若强制失效，宁可显式 FAIL 也不能让 M2 变成「s16 vs 精确轨」的假对拍。 */
    int env_blocked = moe_q4_actq_ok();
    if (actq16_env()) env_blocked = 1;
#if defined(__AVX2__) && ST_ARCH_X86
    if (moe_fuse_env() || dn_d2_env()) env_blocked = 1;
#endif
    if (env_blocked) {
        printf("[MOE-EP] [FAIL] 仅验证 f32 精确轨：请清除 "
               "VLLM_ACTQ / VLLM_ACTQ16 / VLLM_MOE_FUSE / VLLM_DN_D2\n");
        return 1;
    }

    /* ---- M1 ---- */
    MoeEpGeom G;
    if (!moe_ep_geom_init(&G, w)) {
        printf("[MOE-EP] M1 [FAIL] 几何不成立或目录/指针交叉校验失败\n");
        return 1;
    }
    printf("[MOE-EP] M1 geom: ne=%d me=%d d=%d nl=%d tk=%d layout=%s\n",
           G.ne, G.me, G.d, G.nl, G.tk, G.is_q8 ? "q8_8x8" : "q4_4x4");
    printf("[MOE-EP] M1 layerB=%llu B  expB=%llu B (%llu 页)  "
           "down_band run=%d B stride=%d B\n",
           (unsigned long long)G.layerB, (unsigned long long)G.expB,
           (unsigned long long)(G.expB / 4096),
           G.cbe * 72, G.gdB);
    /* 段视图 ↔ 目录声明 offset 的字节级对拍（抽样首/中/末专家各 4KiB） */
    {
        uint32_t s = 0x9E3779B9u; int diff = 0, nchk = 0;
        int samples[3] = { 0, G.ne / 2, G.ne - 1 };
        for (int si = 0; si < 3; si++) {
            int e = samples[si];
            const uint8_t *segv = moe_ep_gate_seg(&G, 0, e);
            const uint8_t *dirp = (const uint8_t *)w->vqf_map + G.g_off + (size_t)e * G.expB;
            for (int b = 0; b < 4096; b++) {
                s = s * 1664525u + 1013904223u;
                if (segv[(size_t)b + (s % 1024)] != dirp[(size_t)b + (s % 1024)]) diff++;
                nchk++;
            }
        }
        if (diff != 0) {
            printf("[MOE-EP] M1 [FAIL] 段视图与目录 offset 不一致 (%d/%d)\n", diff, nchk);
            return 1;
        }
        printf("[MOE-EP] M1 段视图/目录/rows/bytes/整页 交叉校验 [PASS] (%d 字节抽样)\n",
               nchk);
    }
    if (G.is_q8) {
        printf("[MOE-EP] M2 SKIP: 本阶段 EP 执行器仅 q4_4x4（本模型 q8_8x8）\n");
        printf("[MOE-EP] [PASS] M1 完成\n");
        return 0;
    }

    /* ---- M2 ---- */
    int d = G.d, nl = G.nl;
    int nlayers = nl;
    const char *le = getenv("VLLM_EP_LAYERS");
    if (le && le[0]) { int v = atoi(le); if (v > 0 && v < nlayers) nlayers = v; }
    float *x    = (float *)malloc((size_t)d * sizeof(float));
    float *yref = (float *)malloc((size_t)d * sizeof(float));
    float *yep  = (float *)malloc((size_t)d * sizeof(float));
    if (!x || !yref || !yep) {
        free(x); free(yref); free(yep);
        printf("[MOE-EP] M2 [FAIL] OOM\n");
        return 1;
    }
    /* 确定性探针输入（合成；只测 FFN 本身，与真实 hidden 无关） */
    {
        uint32_t s = 0x12345678u;
        for (int i = 0; i < d; i++) {
            s = s * 1664525u + 1013904223u;
            x[i] = (float)((int)((s >> 9) % 2001u) - 1000) / 1000.0f;
        }
    }

    int npass = 0, nfail = 0;
    /* EPLB（§9.25）：自检是单进程虚拟 rank，无 transport 可广播 → 直接本地按 N 建表
     * 安装。表只依赖 (ne,N,热度,算力)，与层无关，故在层循环外按 N 备好。 */
    uint8_t *eplb_tbl[17] = {0};
    if (st_moe_ep_eplb_on()) {
        for (int N = 2; N <= nranks_max && N <= 16; N *= 2) {
            if ((G.ne % N) != 0) continue;
            eplb_tbl[N] = (uint8_t *)malloc((size_t)G.ne);
            if (!eplb_tbl[N] || st_moe_ep_eplb_build(G.ne, N, eplb_tbl[N]) != 0) {
                free(eplb_tbl[N]); eplb_tbl[N] = NULL;
            }
        }
    }
    for (int l = 0; l < nlayers; l++) {
        st_moe_ffn_sparse_q4(w, l, x, yref);        /* 单机参考（生产路径） */
        for (int N = 2; N <= nranks_max; N *= 2) {
            if ((G.ne % N) != 0) continue;
            if (eplb_tbl[N]) st_moe_ep_set_owner_tbl(eplb_tbl[N], G.ne);  /* EPLB: 本 N 的表 */
            if (moe_ffn_sparse_q4_ep_vrank(w, &G, l, x, yep, N) != 0) {
                printf("[MOE-EP] M2 l=%d N=%d SKIP (前置不满足)\n", l, N);
                continue;
            }
            int diff = 0;
            for (int m = 0; m < d; m++) {
                uint32_t a, b;
                memcpy(&a, &yref[m], 4);
                memcpy(&b, &yep[m], 4);
                if (a != b) diff++;
            }
            if (diff == 0) {
                npass++;
                printf("[MOE-EP] M2 l=%2d N=%d  [PASS] 位级一致 (d=%d)\n", l, N, d);
            } else {
                nfail++;
                printf("[MOE-EP] M2 l=%2d N=%d  [FAIL] %d/%d 分量位不同\n",
                       l, N, diff, d);
            }
        }
    }
    free(x); free(yref); free(yep);
    for (int N = 0; N <= 16; N++) free(eplb_tbl[N]);
    printf("[MOE-EP] M2 SUMMARY: PASS=%d FAIL=%d (layers=%d, N<=%d)  -> %s\n",
           npass, nfail, nlayers, nranks_max,
           (nfail == 0 && npass > 0) ? "[PASS]" : "[FAIL]");
    return (nfail == 0 && npass > 0) ? 0 : 1;
}

/* ================================================================
 * M3：多进程 EP（阶段一：同机多进程；阶段二：跨机协同）
 *
 * 角色与协议：
 *   rank0（协调者）：持有 router；每层广播 {x_ffn, sel, pr}；收各 worker 的
 *                    完整 d 维贡献；按 j 升序归约 → 与单机参考位级对拍。
 *   rank r>0（工作者）：收广播 → 只算本 rank 拥有的被选专家 → 回传 {j, C_j}。
 *
 *   广播帧 = BcastHdr{int32 l,tk,d,ne} + x[d]f32 + sel[tk]i32 + pr[tk]f32
 *            （l < 0 = 结束哨兵，其余字段忽略）
 *   回传帧 = ReplyHdr{int32 n} + n × { int32 j; float C_j[d] }
 *
 * 传输层经 vllm_ep.h 抽象：阶段一 host=127.0.0.1（同机多进程）；阶段二把 host
 * 换成板 IP 即跨机协同，协议与上层逻辑零改动。
 *
 * 诚实边界：本阶段各进程 mmap 同一份完整 VQF（便于正确性验证），尚未做
 * "每节点仅驻 1/N 专家"的权重切分 —— 那属方案 A 的文件级提取，后续再做。
 * ================================================================ */
typedef struct { int32_t l, tk, d, ne; } MoeEpBcastHdr;
typedef struct { int32_t n; }           MoeEpReplyHdr;

/* ---- EPLB 会话启动握手（§9.25）----
 * 每个 EP 会话在 FFN 循环之前**必发/必收恰好一帧**（与是否启用 EPLB 无关），
 * 否则 worker 无法判断是否该等表 → 帧数错位。mode=0 = 静态分配（不传表）；
 * mode=1 = 帧后跟 ne 字节 owner 表，worker 原样安装。 */
#define MOE_EP_EPLB_MAGIC 0x45504C42   /* "EPLB" */
typedef struct { int32_t magic, mode, ne, nranks; } MoeEpEplbHdr;

/* 协调者：建表（EPLB 开）→ 本地安装 → 广播（含 mode=0 的静态帧）。0 = 成功。 */
static int moe_ep_eplb_handshake_root(VllmEpRoot *rt, int ne, int nranks) {
    MoeEpEplbHdr eh = { (int32_t)MOE_EP_EPLB_MAGIC, 0, (int32_t)ne, (int32_t)nranks };
    if (!st_moe_ep_eplb_on()) {
        if (vllm_ep_root_bcast(rt, &eh, sizeof(eh)) != 0) return -1;
        fprintf(stderr, "[MOE-EP-EPLB] 静态分配（未启用，N=%d ne=%d）\n", nranks, ne);
        return 0;
    }
    uint8_t *tbl = (uint8_t *)malloc((size_t)ne);
    if (!tbl || st_moe_ep_eplb_build(ne, nranks, tbl) != 0) {
        free(tbl);
        if (vllm_ep_root_bcast(rt, &eh, sizeof(eh)) != 0) return -1;  /* 仍发静态帧 */
        fprintf(stderr, "[MOE-EP-EPLB] 建表失败 → 回退静态分配\n");
        return 0;
    }
    st_moe_ep_set_owner_tbl(tbl, ne);
    eh.mode = 1;
    int rc = (vllm_ep_root_bcast(rt, &eh, sizeof(eh)) == 0 &&
              vllm_ep_root_bcast(rt, tbl, (size_t)ne) == 0) ? 0 : -1;
    free(tbl);
    return rc;
}

/* 工作者：收握手帧；mode=1 时再收 ne 字节表并安装（校验几何一致）。0 = 成功。 */
static int moe_ep_eplb_handshake_worker(VllmEpWorker *wk, int ne, int nranks) {
    MoeEpEplbHdr eh;
    if (vllm_ep_worker_recv(wk, &eh, sizeof(eh)) != 0) return -1;
    if (eh.magic != (int32_t)MOE_EP_EPLB_MAGIC) {
        fprintf(stderr, "[MOE-EP-EPLB] worker: 握手 magic 不符 (0x%08x)\n", (unsigned)eh.magic);
        return -1;
    }
    if (eh.mode == 0) {
        fprintf(stderr, "[MOE-EP-EPLB] worker: 静态分配（协调者未启用 EPLB）\n");
        return 0;
    }
    if (eh.mode != 1 || eh.ne != ne || eh.nranks != nranks) {
        fprintf(stderr, "[MOE-EP-EPLB] worker: 表几何不符 (ne=%d/%d N=%d/%d mode=%d)\n",
                (int)eh.ne, ne, (int)eh.nranks, nranks, (int)eh.mode);
        return -1;
    }
    uint8_t *tbl = (uint8_t *)malloc((size_t)ne);
    if (!tbl) return -1;
    if (vllm_ep_worker_recv(wk, tbl, (size_t)ne) != 0) { free(tbl); return -1; }
    st_moe_ep_set_owner_tbl(tbl, ne);
    free(tbl);
    int nbad = 0;
    for (int e = 0; e < ne; e++)
        if (g_moe_eplb_owner && g_moe_eplb_owner[e] >= nranks) nbad++;
    fprintf(stderr, "[MOE-EP-EPLB] worker: 已安装 EPLB 表 ne=%d N=%d%s\n",
            ne, nranks, nbad ? "（含非法项!）" : "");
    return 0;
}

/* 确定性探针输入（仅协调者生成，随广播下发；跨层可复现）。 */
static void moe_ep_probe_x(float *x, int d, int layer) {
    uint32_t s = 0x12345678u ^ (uint32_t)((uint32_t)layer * 2654435761u);
    for (int i = 0; i < d; i++) {
        s = s * 1664525u + 1013904223u;
        x[i] = (float)((int)((s >> 9) % 2001u) - 1000) / 1000.0f;
    }
}

/* 读 /proc/self/status 的 VmHWM（峰值常驻，kB）；非 Linux 返回 0。 */
static long moe_ep_peak_rss_kb(void) {
#ifdef _WIN32
    return 0;
#else
    FILE *f = fopen("/proc/self/status", "r");
    if (!f) return 0;
    char line[256];
    long v = 0;
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "VmHWM:", 6) == 0) { sscanf(line + 6, "%ld", &v); break; }
    }
    fclose(f);
    return v;
#endif
}

/* ================================================================
 * §9.26 单板串行法：单 rank「独占」FFN 基准
 *
 * 目的：在一块板上**串行**（一次只跑一个 rank）测出「某 rank 独占时」的每层
 * FFN 计算时延。跨机 EP 时每块板**独占**自己的 8 核 + DRAM 带宽，故该值 ≈ 跨机
 * 时该 rank 的时延 —— 而同机并发反而是错误代理（§9.11 实测 1.66–1.84× 更慢）。
 *
 * 内存副产物：本模式只读**本 rank 拥有的专家段**（走 moe_ep_contrib 的所有权
 * 过滤），故常驻集 ≈「加载该 rank 分片」的常驻集（两者都只 touch 7.59 GiB 专家
 * + 1.27 GiB 共享）→ 可在**零分片传输**下同时给出该 rank 的内存上界。
 * 注意：该等价性只在**内存**维度成立；磁盘占用/I/O 维度仍需真实分片（见 §9.26）。
 *
 * rank 语义：nranks=1 → rank0 拥有全部专家（= 单板全做；因 16.45 GiB > RAM，
 *           必然退化为流式，读数含 I/O，仅作参照）；
 *           nranks=2 → rank0/rank1 各拥一半（= 双板各半，8.86 GiB 可常驻）。
 * 边界：仅明文 q4_4x4 MoE、仅 f32 精确轨（与 M2 同）。返回 0 = 成功。
 * ================================================================ */
int st_moe_ep_rank_bench(STModelWeights *w, int nranks, int rank,
                         int layers_max, int reps) {
    if (!w || !w->cfg.is_moe) { printf("[EP-BENCH] [FAIL] 非 MoE 模型\n"); return 1; }
    MoeEpGeom G;
    if (!moe_ep_geom_init(&G, w)) { printf("[EP-BENCH] [FAIL] M1 几何/交叉校验失败\n"); return 1; }
    if (G.is_q8) { printf("[EP-BENCH] [FAIL] 本基准仅 q4_4x4\n"); return 1; }
    int ne = G.ne, tk = G.tk, d = G.d;
    if (nranks < 1 || nranks > 16 || (ne % nranks) != 0 || rank < 0 || rank >= nranks) {
        printf("[EP-BENCH] [FAIL] N=%d rank=%d 非法（须整除 ne=%d）\n", nranks, rank, ne);
        return 1;
    }
    if (reps < 1) reps = 1;
    int nl = G.nl;
    const char *le = getenv("VLLM_EP_LAYERS");
    if (le && le[0]) { int v = atoi(le); if (v > 0 && v < nl) nl = v; }
    if (layers_max > 0 && layers_max < nl) nl = layers_max;
    int verbose = 0;
    { const char *vv = getenv("VLLM_EP_BENCH_VERBOSE"); if (vv && vv[0] == '1') verbose = 1; }

    float *x     = (float *)malloc((size_t)d * sizeof(float));
    int   *sel   = (int *)malloc((size_t)tk * sizeof(int));
    float *pr    = (float *)malloc((size_t)tk * sizeof(float));
    float *avacc = (float *)malloc((size_t)tk * (size_t)(G.ff / ne) * sizeof(float));
    float *Cbuf  = (float *)malloc((size_t)tk * (size_t)d * sizeof(float));
    if (!x || !sel || !pr || !avacc || !Cbuf) {
        free(x); free(sel); free(pr); free(avacc); free(Cbuf);
        printf("[EP-BENCH] [FAIL] OOM\n"); return 1;
    }

    printf("\n=== [EP-BENCH] 单 rank 独占 FFN 基准：N=%d rank=%d layers=%d reps=%d ===\n",
           nranks, rank, nl, reps);
    printf("[EP-BENCH] owner=%s  ne=%d tk=%d d=%d（本 rank 拥有专家数=%d）\n",
           g_moe_eplb_owner ? "EPLB 表" :
           (g_moe_ep_split > 0 && nranks == 2 ? "--ep-split" : "均分/取模"),
           ne, tk, d, ne / nranks);
    fflush(stdout);

    /* 预热：把本 rank 的专家段拉进 DRAM（只读自己的段 → 常驻 ≈ 分片） */
    double t_w0 = st_now_sec();
    long warm_cnt = 0;
    for (int l = 0; l < nl; l++) {
        moe_ep_probe_x(x, d, l);
        moe_ep_router(w, l, x, sel, pr);
        warm_cnt += moe_ep_contrib(w, &G, l, x, sel, pr, rank, nranks, avacc, Cbuf);
    }
    double t_w1 = st_now_sec();
    long rss_warm = vqf_stream_rss_kb();

    /* 计时：每层跑 reps 次取最小值（抗调度噪声），并核对「真的算了东西」 */
    double tot_min = 0.0, tot_mean = 0.0;
    long owned = 0, timed_cnt = 0;
    for (int l = 0; l < nl; l++) {
        moe_ep_probe_x(x, d, l);
        moe_ep_router(w, l, x, sel, pr);
        int own = 0;
        for (int j = 0; j < tk; j++)
            if (sel[j] >= 0 && sel[j] < ne && moe_ep_owner(sel[j], ne, nranks) == rank) own++;
        owned += own;
        double best = 1e300, sum = 0.0;
        for (int r = 0; r < reps; r++) {
            double a = st_now_sec();
            int c = moe_ep_contrib(w, &G, l, x, sel, pr, rank, nranks, avacc, Cbuf);
            double b = st_now_sec();
            double ms = (b - a) * 1000.0;
            if (ms < best) best = ms;
            sum += ms;
            if (r == 0) timed_cnt += c;
        }
        tot_min += best;
        tot_mean += sum / (double)reps;
        if (verbose)
            printf("[EP-BENCH] l=%2d own=%d min=%.3f ms\n", l, own, best);
    }

    double avg_min = tot_min / (double)nl;
    double avg_mean = tot_mean / (double)nl;
    printf("[EP-BENCH] 预热 %.2fs（warm_cnt=%ld 次贡献）rss=%ld kB (%.2f GiB)；"
           "VmHWM=%ld kB (%.2f GiB)\n",
           t_w1 - t_w0, warm_cnt, rss_warm, (double)rss_warm / 1048576.0,
           moe_ep_peak_rss_kb(), (double)moe_ep_peak_rss_kb() / 1048576.0);
    printf("[EP-BENCH] 核对: 计时口径贡献数=%ld vs 拥有专家数合计=%ld %s\n",
           timed_cnt, owned, (timed_cnt == owned) ? "[一致]" : "[不一致! 计时无效]");
    printf("[EP-BENCH] 结果: 每层 FFN 本 rank 计算 min=%.3f ms mean=%.3f ms "
           "（平均拥有被选专家 %.2f/%d）\n",
           avg_min, avg_mean, (double)owned / (double)nl, tk);
    printf("[EP-BENCH] 合计: layers=%d 本 rank 纯计算=%.2f ms（min 口径）%.2f ms（mean 口径）\n",
           nl, tot_min, tot_mean);
    if (owned > 0)
        printf("[EP-BENCH] 折算: %.3f ms/被选专家（含每次调用固定开销）\n",
               tot_min / (double)owned);
    long rss_end = vqf_stream_rss_kb();
    printf("[EP-BENCH] rss_end=%ld kB (%.2f GiB)  VmHWM=%ld kB (%.2f GiB)\n",
           rss_end, (double)rss_end / 1048576.0,
           moe_ep_peak_rss_kb(), (double)moe_ep_peak_rss_kb() / 1048576.0);
    fflush(stdout);

    free(x); free(sel); free(pr); free(avacc); free(Cbuf);
    return (timed_cnt == owned) ? 0 : 1;
}

int st_moe_ep_coord_run(STModelWeights *w, int nranks, int port) {
    printf("\n=== [MOE-EP-MP] 协调者 rank0 / N=%d / port=%d ===\n", nranks, port);
    fflush(stdout);
    if (!w || !w->cfg.is_moe) { printf("[MOE-EP-MP] [FAIL] 非 MoE 模型\n"); return 1; }
    MoeEpGeom G;
    if (!moe_ep_geom_init(&G, w)) {
        printf("[MOE-EP-MP] [FAIL] M1 几何/交叉校验失败\n"); return 1;
    }
    if (G.is_q8) { printf("[MOE-EP-MP] [FAIL] 本阶段仅 q4_4x4\n"); return 1; }
    if (nranks < 2 || nranks > 16 || (G.ne % nranks) != 0) {
        printf("[MOE-EP-MP] [FAIL] nranks=%d 非法（需 2..16 且整除 ne=%d）\n",
               nranks, G.ne);
        return 1;
    }
    int d = G.d, ne = G.ne, tk = G.tk, nl = G.nl;
    int nlayers = nl;
    const char *le = getenv("VLLM_EP_LAYERS");
    if (le && le[0]) { int v = atoi(le); if (v > 0 && v < nlayers) nlayers = v; }

    VllmEpRoot *rt = NULL;
    float *x = NULL, *yref = NULL, *yep = NULL, *avacc = NULL, *Cbuf = NULL;
    int *sel = NULL; float *pr = NULL;
    int rc = 1, npass = 0, nfail = 0;

    rt = vllm_ep_root_open(port, nranks, VLLM_EP_BACKEND_TCP);
    if (!rt) { printf("[MOE-EP-MP] [FAIL] transport 建立失败\n"); goto done; }
    x     = (float *)malloc((size_t)d * sizeof(float));
    yref  = (float *)malloc((size_t)d * sizeof(float));
    yep   = (float *)malloc((size_t)d * sizeof(float));
    avacc = (float *)malloc((size_t)tk * (size_t)(G.ff / ne) * sizeof(float));
    Cbuf  = (float *)malloc((size_t)tk * (size_t)d * sizeof(float));
    sel   = (int *)malloc((size_t)tk * sizeof(int));
    pr    = (float *)malloc((size_t)tk * sizeof(float));
    if (!x || !yref || !yep || !avacc || !Cbuf || !sel || !pr) {
        printf("[MOE-EP-MP] [FAIL] OOM\n"); goto done;
    }
    /* EPLB（§9.25）：会话启动握手——建表 + 广播（未启用亦发静态帧，保证帧数配对） */
    if (moe_ep_eplb_handshake_root(rt, ne, nranks) != 0) {
        printf("[MOE-EP-MP] [FAIL] EPLB 握手失败\n"); goto done;
    }

    for (int l = 0; l < nlayers; l++) {
        moe_ep_probe_x(x, d, l);
        moe_ep_router(w, l, x, sel, pr);

        MoeEpBcastHdr h = { (int32_t)l, (int32_t)tk, (int32_t)d, (int32_t)ne };
        if (vllm_ep_root_bcast(rt, &h, sizeof(h)) != 0 ||
            vllm_ep_root_bcast(rt, x, (size_t)d * sizeof(float)) != 0 ||
            vllm_ep_root_bcast(rt, sel, (size_t)tk * sizeof(int)) != 0 ||
            vllm_ep_root_bcast(rt, pr, (size_t)tk * sizeof(float)) != 0) {
            printf("[MOE-EP-MP] [FAIL] 广播 l=%d\n", l); goto done;
        }

        /* rank0 自身贡献 + 各 worker 回传的贡献 → Cbuf[j] */
        moe_ep_contrib(w, &G, l, x, sel, pr, 0, nranks, avacc, Cbuf);
        for (int rk = 1; rk < nranks; rk++) {
            MoeEpReplyHdr rh;
            if (vllm_ep_root_recv(rt, rk, &rh, sizeof(rh)) != 0 ||
                rh.n < 0 || rh.n > tk) {
                printf("[MOE-EP-MP] [FAIL] 回传头 rank=%d l=%d\n", rk, l); goto done;
            }
            for (int i = 0; i < (int)rh.n; i++) {
                int32_t j = -1;
                if (vllm_ep_root_recv(rt, rk, &j, sizeof(j)) != 0 ||
                    j < 0 || j >= tk) {
                    printf("[MOE-EP-MP] [FAIL] 回传 j rank=%d l=%d\n", rk, l); goto done;
                }
                if (vllm_ep_root_recv(rt, rk, Cbuf + (size_t)j * d,
                                      (size_t)d * sizeof(float)) != 0) {
                    printf("[MOE-EP-MP] [FAIL] 回传 C_j rank=%d l=%d\n", rk, l); goto done;
                }
            }
        }

        moe_ep_reduce(Cbuf, sel, pr, tk, d, yep);
        st_moe_ffn_sparse_q4(w, l, x, yref);          /* 单机参考（生产路径） */
        int diff = 0;
        for (int m = 0; m < d; m++) {
            uint32_t a, b;
            memcpy(&a, &yref[m], 4);
            memcpy(&b, &yep[m], 4);
            if (a != b) diff++;
        }
        if (diff == 0) {
            npass++;
            printf("[MOE-EP-MP] l=%2d N=%d  [PASS] 位级一致 (d=%d)\n", l, nranks, d);
        } else {
            nfail++;
            printf("[MOE-EP-MP] l=%2d N=%d  [FAIL] %d/%d 分量位不同\n", l, nranks, diff, d);
        }
    }
    { MoeEpBcastHdr end = { -1, 0, 0, 0 };
      vllm_ep_root_bcast(rt, &end, sizeof(end)); }
    printf("[MOE-EP-MP] SUMMARY: PASS=%d FAIL=%d (layers=%d, N=%d) -> %s\n",
           npass, nfail, nlayers, nranks,
           (nfail == 0 && npass > 0) ? "[PASS]" : "[FAIL]");
    rc = (nfail == 0 && npass > 0) ? 0 : 1;

done:
    free(x); free(yref); free(yep); free(avacc); free(Cbuf); free(sel); free(pr);
    if (rt) vllm_ep_root_close(rt);
    return rc;
}

int st_moe_ep_worker_run(STModelWeights *w, int rank, const char *host,
                         int port, int nranks) {
    if (!w || !w->cfg.is_moe) return 1;
    MoeEpGeom G;
    if (!moe_ep_geom_init(&G, w) || G.is_q8) return 1;
    int d = G.d, ne = G.ne, tk = G.tk;
    if (rank < 1 || rank >= nranks || (ne % nranks) != 0) return 1;

    VllmEpWorker *wk = vllm_ep_worker_open(host, port, rank, nranks,
                                           VLLM_EP_BACKEND_TCP);
    if (!wk) return 1;
    /* EPLB（§9.25）：会话启动握手——安装协调者广播的分配表（mode=0 = 静态） */
    if (moe_ep_eplb_handshake_worker(wk, ne, nranks) != 0) {
        fprintf(stderr, "[MOE-EP-EPLB] worker(rank %d): 握手失败\n", rank);
        vllm_ep_worker_close(wk);
        return 1;
    }

    float *x = (float *)malloc((size_t)d * sizeof(float));
    float *avacc = (float *)malloc((size_t)tk * (size_t)(G.ff / ne) * sizeof(float));
    float *Cbuf  = (float *)malloc((size_t)tk * (size_t)d * sizeof(float));
    int *sel = (int *)malloc((size_t)tk * sizeof(int));
    float *pr = (float *)malloc((size_t)tk * sizeof(float));
    int rc = 0;
    if (!x || !avacc || !Cbuf || !sel || !pr) rc = 1;

    /* M4 验收项②：工作者侧分解 —— 等请求 / 本 rank 专家计算 / 回传 三段墙钟 */
    double acc_wait = 0.0, acc_comp = 0.0, acc_send = 0.0;
    uint64_t calls = 0;

    while (rc == 0) {
        MoeEpBcastHdr h;
        double t0 = st_now_sec();
        if (vllm_ep_worker_recv(wk, &h, sizeof(h)) != 0) { rc = 1; break; }
        if (h.l < 0) break;                                   /* 结束哨兵 */
        if (h.d != d || h.tk != tk) { rc = 1; break; }
        if (vllm_ep_worker_recv(wk, x, (size_t)d * sizeof(float)) != 0 ||
            vllm_ep_worker_recv(wk, sel, (size_t)tk * sizeof(int)) != 0 ||
            vllm_ep_worker_recv(wk, pr, (size_t)tk * sizeof(float)) != 0) {
            rc = 1; break;
        }
        double t1 = st_now_sec();
        int cnt = moe_ep_contrib(w, &G, (int)h.l, x, sel, pr, rank, nranks,
                                 avacc, Cbuf);
        double t2 = st_now_sec();
        MoeEpReplyHdr rh = { (int32_t)cnt };
        if (vllm_ep_worker_send(wk, &rh, sizeof(rh)) != 0) { rc = 1; break; }
        for (int j = 0; j < tk; j++) {
            int e = sel[j];
            if (e < 0 || e >= ne) continue;
            if (moe_ep_owner(e, ne, nranks) != rank) continue;
            int32_t jj = (int32_t)j;
            if (vllm_ep_worker_send(wk, &jj, sizeof(jj)) != 0) { rc = 1; break; }
            if (vllm_ep_worker_send(wk, Cbuf + (size_t)j * d,
                                    (size_t)d * sizeof(float)) != 0) { rc = 1; break; }
        }
        acc_wait += (t1 - t0);
        acc_comp += (t2 - t1);
        acc_send += (st_now_sec() - t2);
        calls++;
    }
    if (calls > 0) {
        uint64_t tx = 0, rx = 0, msgs = 0;
        vllm_ep_stats(&tx, &rx, &msgs);
        fprintf(stderr,
                "[MOE-EP-STATS] worker(rank %d): ffn_calls=%llu wait=%.3fs "
                "compute=%.3fs send=%.3fs (avg compute=%.3fms) tx=%.1fMB rx=%.1fMB msgs=%llu\n",
                rank, (unsigned long long)calls, acc_wait, acc_comp, acc_send,
                acc_comp * 1000.0 / (double)calls,
                (double)tx / 1048576.0, (double)rx / 1048576.0,
                (unsigned long long)msgs);
    }
    free(x); free(avacc); free(Cbuf); free(sel); free(pr);
    vllm_ep_worker_close(wk);
    return rc;
}

/* ================================================================
 * M3.5：EP 接入真实 forward（协调者侧）
 *
 * 目标：把 EP 从"合成探针"推进到"真实推理链路"，让注意力/KV/norm/LM 全部由
 * rank0 执行，只有 FFN 专家段走分布式。worker 侧无需任何改动 —— 它只按协议
 * 应答 FFN 请求（`st_moe_ep_worker_run` 的循环对"x 从哪来"无感）。
 *
 * 挂钩点：forward 内 3 处 `st_moe_ffn_sparse(w,l,x_ffn,y)` 调用改为
 * `moe_ffn_dispatch(...)`；EP 会话活跃时由本模块接管，否则回落单机路径。
 *
 * A≡C：协调者仍用 moe_ep_router/moe_ep_contrib/moe_ep_reduce（与单机同内核、
 * 同 j 升序归约）→ 与单机 st_moe_ffn_sparse_q4 位级一致。
 *
 * 惰性绑定：transport 由 main 在跑测试前 `st_moe_ep_root_begin()` 建立（不依赖
 * 模型）；几何/缓冲在首次 FFN 调用时绑定（那时模型已加载）。
 * ================================================================ */
static VllmEpRoot *g_moe_ep_rt = NULL;     /* 非 NULL = EP 会话活跃 */
static int         g_moe_ep_nranks = 0;
static int         g_moe_ep_bound = 0;
static MoeEpGeom   g_moe_ep_g;
static float      *g_moe_ep_avacc = NULL;
static float      *g_moe_ep_cbuf  = NULL;
static int        *g_moe_ep_sel   = NULL;
static float      *g_moe_ep_pr    = NULL;
/* M4 验收项②：EP 交换的墙钟累计（协调者视角，三段互不重叠）
 *   bcast_s = 广播写出耗时（≈ 网络推送）
 *   local_s = 协调者本 rank 专家贡献的计算耗时（与 worker 并发）
 *   wait_s  = 本地算完后，等各 rank 回传收齐的耗时（对端算力 + 往返） */
static double      g_moe_ep_bcast_s = 0.0;
static double      g_moe_ep_local_s = 0.0;
static double      g_moe_ep_wait_s  = 0.0;
static uint64_t    g_moe_ep_calls   = 0;
/* 负载均衡观测（§6.3 "M4 须报告分布"）：按 (rank, 该 rank 拥有的选中专家数) 计数。
 * 样本数 = ffn_calls × (nranks-1)；期望 = tk / nranks。 */
static uint64_t    g_moe_ep_hist[33] = {0};

/* 建立协调者 transport（等待 nranks-1 个 worker 接入）。0 = 成功。 */
int st_moe_ep_root_begin(int nranks, int port) {
    if (g_moe_ep_rt) return 0;
    if (nranks < 2) return -1;
    VllmEpRoot *rt = vllm_ep_root_open(port, nranks, VLLM_EP_BACKEND_TCP);
    if (!rt) return -1;
    g_moe_ep_rt = rt;
    g_moe_ep_nranks = nranks;
    g_moe_ep_bound = 0;
    fprintf(stderr, "[MOE-EP-FWD] root ready: N=%d port=%d\n", nranks, port);
    return 0;
}

/* 首次 FFN 调用时绑定几何/缓冲。0 = 成功。 */
static int moe_ep_bind(const STModelWeights *w) {
    if (g_moe_ep_bound) return 0;
    if (!moe_ep_geom_init(&g_moe_ep_g, w)) {
        fprintf(stderr, "[MOE-EP-FWD] 几何校验失败 → 回落单机\n");
        return -1;
    }
    if (g_moe_ep_g.is_q8) {
        fprintf(stderr, "[MOE-EP-FWD] 仅 q4_4x4 → 回落单机\n");
        return -1;
    }
    if ((g_moe_ep_g.ne % g_moe_ep_nranks) != 0) {
        fprintf(stderr, "[MOE-EP-FWD] ne=%d 不被 N=%d 整除 → 回落单机\n",
                g_moe_ep_g.ne, g_moe_ep_nranks);
        return -1;
    }
    /* EPLB（§9.25）：会话启动握手——建表 + 广播给 worker（未启用亦发静态帧配对）。
     * 放在缓冲分配之前：失败即回落，无需清理。此时端口已收妥 worker 连接。 */
    if (moe_ep_eplb_handshake_root(g_moe_ep_rt, g_moe_ep_g.ne, g_moe_ep_nranks) != 0) {
        fprintf(stderr, "[MOE-EP-FWD] EPLB 握手失败 → 回落单机\n");
        return -1;
    }
    int tk = g_moe_ep_g.tk, d = g_moe_ep_g.d;
    int ef = g_moe_ep_g.ff / g_moe_ep_g.ne;
    g_moe_ep_avacc = (float *)malloc((size_t)tk * (size_t)ef * sizeof(float));
    g_moe_ep_cbuf  = (float *)malloc((size_t)tk * (size_t)d * sizeof(float));
    g_moe_ep_sel   = (int *)malloc((size_t)tk * sizeof(int));
    g_moe_ep_pr    = (float *)malloc((size_t)tk * sizeof(float));
    if (!g_moe_ep_avacc || !g_moe_ep_cbuf || !g_moe_ep_sel || !g_moe_ep_pr) {
        free(g_moe_ep_avacc); free(g_moe_ep_cbuf); free(g_moe_ep_sel); free(g_moe_ep_pr);
        g_moe_ep_avacc = NULL; g_moe_ep_cbuf = NULL; g_moe_ep_sel = NULL; g_moe_ep_pr = NULL;
        return -1;
    }
    g_moe_ep_bound = 1;
    fprintf(stderr, "[MOE-EP-FWD] bound: ne=%d tk=%d d=%d ef=%d N=%d layout=%s\n",
            g_moe_ep_g.ne, tk, d, ef, g_moe_ep_nranks, "q4_4x4");
    return 0;
}

/* 结束 EP 会话：发结束哨兵 + 关闭 transport + 释放缓冲。幂等。 */
void st_moe_ep_root_end(void) {
    if (!g_moe_ep_rt) return;
    if (g_moe_ep_bound) {
        MoeEpBcastHdr end = { -1, 0, 0, 0 };
        vllm_ep_root_bcast(g_moe_ep_rt, &end, sizeof(end));
    }
    /* M4 验收项②：通信量/延迟实测报告（协调者视角） */
    if (g_moe_ep_calls > 0) {
        uint64_t tx = 0, rx = 0, msgs = 0;
        vllm_ep_stats(&tx, &rx, &msgs);
        double nc = (double)g_moe_ep_calls;
        fprintf(stderr,
                "[MOE-EP-STATS] coord: ffn_calls=%llu bcast=%.3fs local=%.3fs "
                "wait=%.3fs (avg bcast=%.3fms local=%.3fms wait=%.3fms) "
                "tx=%.1fMB rx=%.1fMB msgs=%llu\n",
                (unsigned long long)g_moe_ep_calls,
                g_moe_ep_bcast_s, g_moe_ep_local_s, g_moe_ep_wait_s,
                g_moe_ep_bcast_s * 1000.0 / nc,
                g_moe_ep_local_s * 1000.0 / nc,
                g_moe_ep_wait_s  * 1000.0 / nc,
                (double)tx / 1048576.0, (double)rx / 1048576.0,
                (unsigned long long)msgs);
        /* 负载均衡分布：横轴 = 某个 rank 在单次 FFN 中分到的选中专家数 */
        fprintf(stderr, "[MOE-EP-STATS] coord 每 rank 选中专家数分布:");
        for (int i = 0; i <= g_moe_ep_g.tk && i < 33; i++)
            fprintf(stderr, " %d:%llu", i, (unsigned long long)g_moe_ep_hist[i]);
        fprintf(stderr, "\n");
    }
    vllm_ep_root_close(g_moe_ep_rt);
    g_moe_ep_rt = NULL;
    g_moe_ep_bound = 0;
    free(g_moe_ep_avacc); free(g_moe_ep_cbuf); free(g_moe_ep_sel); free(g_moe_ep_pr);
    g_moe_ep_avacc = NULL; g_moe_ep_cbuf = NULL; g_moe_ep_sel = NULL; g_moe_ep_pr = NULL;
}

/* EP 接管一次 FFN：router → 广播 → 收各 rank 完整 d 维贡献 → j 升序归约。
 * 返回 0 = 已处理；非 0 = 未处理（调用方回落单机路径）。 */
static int st_moe_ep_forward_ffn(const STModelWeights *w, int l,
                                 const float *x_ffn, float *y) {
    if (!g_moe_ep_rt) return -1;
    if (!g_moe_ep_bound && moe_ep_bind(w) != 0) return -1;
    const MoeEpGeom *G = &g_moe_ep_g;
    int tk = G->tk, d = G->d, ne = G->ne;
    int *sel = g_moe_ep_sel;
    float *pr = g_moe_ep_pr;

    moe_ep_router(w, l, x_ffn, sel, pr);
    double t_x0 = st_now_sec();
    MoeEpBcastHdr h = { (int32_t)l, (int32_t)tk, (int32_t)d, (int32_t)ne };
    if (vllm_ep_root_bcast(g_moe_ep_rt, &h, sizeof(h)) != 0 ||
        vllm_ep_root_bcast(g_moe_ep_rt, x_ffn, (size_t)d * sizeof(float)) != 0 ||
        vllm_ep_root_bcast(g_moe_ep_rt, sel, (size_t)tk * sizeof(int)) != 0 ||
        vllm_ep_root_bcast(g_moe_ep_rt, pr, (size_t)tk * sizeof(float)) != 0) {
        fprintf(stderr, "[MOE-EP-FWD] 广播失败 l=%d → 回落单机\n", l);
        return -1;
    }
    double t_x1 = st_now_sec();
    moe_ep_contrib(w, G, l, x_ffn, sel, pr, 0, g_moe_ep_nranks,
                   g_moe_ep_avacc, g_moe_ep_cbuf);
    double t_l1 = st_now_sec();
    for (int rk = 1; rk < g_moe_ep_nranks; rk++) {
        MoeEpReplyHdr rh;
        if (vllm_ep_root_recv(g_moe_ep_rt, rk, &rh, sizeof(rh)) != 0 ||
            rh.n < 0 || rh.n > tk) {
            fprintf(stderr, "[MOE-EP-FWD] 回传头失败 rank=%d l=%d\n", rk, l);
            return -1;
        }
        if (rh.n < 33) g_moe_ep_hist[(int)rh.n]++;
        for (int i = 0; i < (int)rh.n; i++) {
            int32_t j = -1;
            if (vllm_ep_root_recv(g_moe_ep_rt, rk, &j, sizeof(j)) != 0 ||
                j < 0 || j >= tk) {
                fprintf(stderr, "[MOE-EP-FWD] 回传 j 失败 rank=%d l=%d\n", rk, l);
                return -1;
            }
            if (vllm_ep_root_recv(g_moe_ep_rt, rk, g_moe_ep_cbuf + (size_t)j * d,
                                  (size_t)d * sizeof(float)) != 0) {
                fprintf(stderr, "[MOE-EP-FWD] 回传 C_j 失败 rank=%d l=%d\n", rk, l);
                return -1;
            }
        }
    }
    double t_x2 = st_now_sec();
    g_moe_ep_bcast_s += (t_x1 - t_x0);
    g_moe_ep_local_s += (t_l1 - t_x1);
    g_moe_ep_wait_s  += (t_x2 - t_l1);
    g_moe_ep_calls++;
    moe_ep_reduce(g_moe_ep_cbuf, sel, pr, tk, d, y);
    return 0;
}

/* FFN 统一派发：EP 会话活跃则走 EP，否则原单机路径（行为完全不变）。 */
static void moe_ffn_dispatch(const STModelWeights *w, int l,
                             const float *x_ffn, float *y) {
    if (st_moe_ep_forward_ffn(w, l, x_ffn, y) == 0) return;
    st_moe_ffn_sparse(w, l, x_ffn, y);
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
    int trace = getenv("VLLM_DEC_TRACE") ? 1 : 0;     /* P1-2：轨迹打印与计时解耦 */
    double t_all0 = 0.0;
    double t_qkv = 0.0, t_attn = 0.0, t_o = 0.0, t_gu = 0.0, t_down = 0.0, t_lm = 0.0;
    double t_nrm = 0.0, t_rope = 0.0, t_kv = 0.0, t_silu = 0.0;   /* P1-2 深挖串行项细分 */
    double t_loop = 0.0, tw2 = 0.0;   /* 整层循环墙钟（残差归因） */
    if (prof) t_all0 = st_now_sec();

    float *residual = st->ffn_out_buf;
    float *normed  = st->ffn_buf;

    /* EMBEDDING */
    float *x = st->hidden;
    float *emb = w->token_embed;
    if (!emb) return;

    for (int i = 0; i < d; i++)
        x[i] = st_emb_get(w, (size_t)token_id, (size_t)i, d);
    if (getenv("VLLM_MOE_DUMP") && getenv("VLLM_MOE_DUMP")[0] == '1') {
        printf("[L0D] meta token=%d seq=%d dim=%d eps=%.3e theta=%.3e\n",
               token_id, st->seq_len, d, eps, theta);
    }

    for (int l = 0; l < nl; l++) {
        double tw = 0.0;             /* P1-2 深挖：串行项局部计时 */
        if (prof) tw2 = st_now_sec();
        vqf_stream_layer_advance(w, l);   /* 分层驻留：驱逐 l-N / 预读 l+1 */
        if (moe_dump_at(l)) moe_l0_dump("S0_embed", x, d);
        if (trace)
            fprintf(stderr, "[DDEC] fwd l=%d/%d seq=%d mrope=%d\n",
                    l, nl, st->seq_len, st->mrope_pos);
        g_npu_layer = l;   /* decode offload (VLLM_NPU_LOAD=2): per-layer model */
        /* --- Attention --- */
        if (prof) tw = st_now_sec();
        memcpy(residual, x, d * sizeof(float));
        dyn_rms_norm(normed, x, w->attn_norm + l * d, d, eps);
        if (prof) t_nrm += st_now_sec() - tw;
        if (moe_dump_at(l)) moe_l0_dump("S1_attn_norm", normed, d);

        /* Q/K/V projections: fused matvec (axiom: block_matrix_assoc_natural)
         * Single pass through normed[] produces Q, K, V simultaneously.
         * Saves 2 re-reads of 16 KB input vector per layer (~1.1 MB total). */
        double t0 = 0.0;
        if (prof) t0 = st_now_sec();
        if (w->has_q4 && w->q4_q_weight &&
            (st_dense_q4_first() || !w->has_q8 || !w->q8_q_weight)) {
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
        if (trace) { fprintf(stderr, "[DDEC2] l=%d QKV done\n", l); fflush(stderr); }
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
        if (moe_dump_at(l)) moe_l0_dump("S2_q_raw", st->q_buf, kv_dim_q);

        /* Q/K norms (Qwen3-VL specific) — per-head RMS normalization
         * Qwen3 uses Q/K norms: first RMS-normalize each head to unit length,
         * then multiply by learned weight vector. Applied before MRoPE. */
        if (prof) tw = st_now_sec();
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
        if (prof) t_nrm += st_now_sec() - tw;

        /* RoPE：统一走 dyn_mrope（半分裂配对 (j, j+hd/2)，即 HF rotate_half）。
         * 文本-only 时 MRoPE 三个位置分量相同 ⇒ 退化为标准 RoPE，故非 MRoPE 模型
         * （Qwen3 / Qwen3-MoE 纯文本）也用同一实现。
         * 2026-09-11 修：旧 fallback 用「相邻配对 (2j, 2j+1)」（GPT-J 风格），
         * 与 Qwen3 训练口径不符 → 位置信息被打乱 → 长尾贪心复读塌陷（30B-A3B 实测
         * 层 0 注意力输出偏差 39%、与 HF layer0 对拍 46%）。 */
        if (prof) tw = st_now_sec();
        dyn_mrope(st->q_buf, st->k_buf, hd, nh, nkv, st->cfg.head_dim_full, pos, theta);

        if (moe_dump_at(l)) moe_l0_dump("S3_q_postrope", st->q_buf, kv_dim_q);
        if (prof) t_rope += st_now_sec() - tw;

        /* Store K/V in cache (INT8 quantized if use_kv_q8, else float) */
        if (prof) tw = st_now_sec();
        int cl = st->cache_len[l];
        int have_f32 = st->k_cache[l] != NULL;   /* P3: nof32 档无 f32 正典行 */
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
            if (have_f32) {
                if (vllm_pf_q8cache_env())   /* P0 原型：f32 行 = q8 反量化（质量门测量） */
                    kv_dequant_roundtrip(kdst, vdst,
                                         st->k_scale[l] + (size_t)cl * nkv,
                                         st->v_scale[l] + (size_t)cl * nkv,
                                         st->k_buf, st->v_buf, nkv, hd);
                /* Also store in float cache */
                memcpy(kv_row_f32(st->k_cache[l], cl, kv_dim, st->kv_bs), st->k_buf, kv_dim * sizeof(float));
                memcpy(kv_row_f32(st->v_cache[l], cl, kv_dim, st->kv_bs), st->v_buf, kv_dim * sizeof(float));
            }
        } else if (st->use_kv_q4) {
            q4_pack_token(q4_k_row(st->k_cache_q4[l], cl, 0, hd, st->kv_bs, nkv),
                          q4_v_row(st->v_cache_q4[l], cl, 0, hd, st->kv_bs, nkv),
                          st->k_buf, st->v_buf, nkv, hd);
            if (have_f32) {
                memcpy(kv_row_f32(st->k_cache[l], cl, kv_dim, st->kv_bs), st->k_buf, kv_dim * sizeof(float));
                memcpy(kv_row_f32(st->v_cache[l], cl, kv_dim, st->kv_bs), st->v_buf, kv_dim * sizeof(float));
            }
        } else if (have_f32) {
            memcpy(kv_row_f32(st->k_cache[l], cl, kv_dim, st->kv_bs), st->k_buf, kv_dim * sizeof(float));
            memcpy(kv_row_f32(st->v_cache[l], cl, kv_dim, st->kv_bs), st->v_buf, kv_dim * sizeof(float));
        }
        if (KV_GUARD && !g_kv_lazy && have_f32) {
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
        if (prof) t_kv += st_now_sec() - tw;

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
                /* Blocked Flash Attention: pack K/V into L1-friendly tiles.
                 * ④ llama 对齐：按 head 并行（vllm_flash_head_worker）。每 head
                 * 独立写 attn_buf 段、scores/K_packed/V_packed 均在 worker 栈上、
                 * KV cache 只读共享 → 与串行 for 逐位一致（同 M4h 并行判据）。 */
                vllm_flash_head_ctx vf = { st, nh, nkv, hd, l, seq_len, kv_dim, st->kv_bs, scale };
                vllm_tp_parfor(0, nh, vllm_flash_head_worker, &vf);
            } else {
                /* Standard attention: per-head dot products (AVX2 accelerated).
                 * When use_kv_q8, reads INT8 K/V cache and dequantizes on-the-fly,
                 * trading 2 extra int8→float conversions for 4x memory bandwidth reduction.
                 * When g_sparse_attn, per-head top-k block sparse attention
                 * (Phase 1) replaces the full-position scan.
                 * M4h: head 循环 OMP 并行（32 heads 4 线程，每 head 独立写
                 * attn_buf，scores 用 per-iteration 局部数组避免共享竞争；并行
                 * 不改变每 head 的数值，位级一致保持）。 */
                /* P2: 块级 prefill 重要性在层内只算一次（与 head 无关），各 head
                 * 只读共享，省掉每 head 一遍 O(seq_len) 归约。累加顺序与原实现
                 * 逐位一致（按 pos 递增、初值 0），故不改变选块结果。 */
                const float *imp_layer = NULL;
                if (g_sparse_attn && !st->use_kv_q4 && g_sparse_block > 0 &&
                    seq_len > g_sparse_block * 2 && st->prefill_importance) {
                    int n_imp_blocks = (seq_len + g_sparse_block - 1) / g_sparse_block;
                    if (sparse_scratch_ensure((size_t)n_imp_blocks, 0, 0) == 0) {
                        float *acc = g_ss.imp_sum;
                        memset(acc, 0, (size_t)n_imp_blocks * sizeof(float));
                        for (int b = 0; b < n_imp_blocks; b++) {
                            int p0 = b * g_sparse_block;
                            int p1 = p0 + g_sparse_block; if (p1 > seq_len) p1 = seq_len;
                            for (int p = p0; p < p1; p++)
                                acc[b] += st->prefill_importance[p];
                        }
                        imp_layer = acc;
                    }
                }
                vllm_attn_head_ctx vh = { st, nh, nkv, hd, l, seq_len, kv_dim, use_q8,
                                          scale, imp_layer };
                vllm_tp_parfor(0, nh, vllm_attn_head_worker, &vh);
            }
        if (prof) t_attn += st_now_sec() - t0;
        if (trace) { fprintf(stderr, "[DDEC2] l=%d ATT done\n", l); fflush(stderr); }
        if (moe_dump_at(l)) moe_l0_dump("S4_attn_out", st->attn_buf, kv_dim_q);

        /* W3: attnbuf IMMEDIATELY after attention (before O-proj), to detect
         * any corruption by the O-projection. */
        if (getenv("VLLM_DUMP_ATTN") && l == 0 && st->seq_len == 19)
            printf("[DUMP] l0-attnbuf-preO[0:8]=%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                   st->attn_buf[0], st->attn_buf[1], st->attn_buf[2], st->attn_buf[3],
                   st->attn_buf[4], st->attn_buf[5], st->attn_buf[6], st->attn_buf[7]);

        /* O projection + attention residual (fused — axiom: block_matrix_assoc_natural) */
        if (prof) t0 = st_now_sec();
        if (w->has_q4 && w->q4_o_weight &&
            (st_dense_q4_first() || !w->has_q8 || !w->q8_o_weight)) {
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
        if (trace) { fprintf(stderr, "[DDEC2] l=%d O done\n", l); fflush(stderr); }
        if (getenv("VLLM_DUMP_ATTN") && !g_kv_nof32 && l == 0 && st->seq_len == 19) {
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

        if (moe_dump_at(l)) moe_l0_dump("S5_x_postattn", x, d);

        /* --- FFN --- */
        if (trace) { fprintf(stderr, "[DDEC2] l=%d FFN in\n", l); fflush(stderr); }
        if (prof) tw = st_now_sec();
        memcpy(residual, x, d * sizeof(float));
        dyn_rms_norm(normed, x, w->ffn_norm + l * d, d, eps);
        if (prof) t_nrm += st_now_sec() - tw;
        if (moe_dump_at(l)) moe_l0_dump("S6_ffn_norm", normed, d);

        /* Save residual before FFN overwrites ffn_out_buf (alias!) */
        memcpy(st->attn_buf, residual, d * sizeof(float));

        if (w->cfg.is_moe) {
            /* qwen3_moe 稀疏专家 FFN（router→top-k→softmax→加权聚合）。
             * 读取仅 top-k 专家的 q4 段（~6% 层权重）；残差在调用方加。 */
            if (prof) t0 = st_now_sec();
            moe_ffn_dispatch(w, l, normed, st->k_buf);
            if (prof) t_gu += st_now_sec() - t0;
            if (moe_dump_at(l)) moe_l0_dump("S7_ffn_out", st->k_buf, d);
            const float *rs = st->attn_buf;
            const float *yy = st->k_buf;
            for (int _i = 0; _i < d; _i++) x[_i] = rs[_i] + yy[_i];
            if (moe_dump_at(l)) moe_l0_dump("S8_x_postffn", x, d);
            if (trace) { fprintf(stderr, "[DDEC2] l=%d FFN out (moe)\n", l); fflush(stderr); }
        } else if (st_ffn_q2_layer(w, l)) {
            /* Q2_1 FFN (layered q2mix: layers before the Q4 tail)
             * mixed-precision 2-bit (axiom: blas_precision_efficiency_tradeoff) */
            if (prof) t0 = st_now_sec();
            dyn_matvec_q2_q8_fused_gate_up(st->q_buf, st->k_buf,
                w->q2_gate_weight + Q2_BYTES((size_t)l * ff * d),
                w->q2_up_weight   + Q2_BYTES((size_t)l * ff * d),
                normed, ff, d);
            if (prof) t_gu += st_now_sec() - t0;
            vllm_silu_mul_ctx vg = { st->q_buf, st->k_buf, ff };
            if (prof) tw = st_now_sec();
            vllm_tp_parfor(0, (ff + 15) / 16, vllm_silu_mul_worker, &vg);
            if (prof) t_silu += st_now_sec() - tw;
            /* P4: fused down projection + FFN residual — writes directly to x */
            if (prof) t0 = st_now_sec();
            dyn_matvec_q2_q8_fused_down_residual(x, st->attn_buf,
                w->q2_down_weight + Q2_BYTES((size_t)l * d * ff),
                st->q_buf, d, ff);
            if (prof) t_down += st_now_sec() - t0;
            if (trace) { fprintf(stderr, "[DDEC2] l=%d FFN out (q2)\n", l); fflush(stderr); }
        } else if (w->has_q4 &&
                   (st_dense_q4_first() || !w->has_q8 || !w->q8_gate_weight)) {
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
            if (prof) tw = st_now_sec();
            vllm_tp_parfor(0, (ff + 15) / 16, vllm_silu_mul_worker, &vg);
            if (prof) t_silu += st_now_sec() - tw;
            /* P4: fused down projection + FFN residual — writes directly to x */
            if (prof) t0 = st_now_sec();
            dyn_matvec_q4_q8_fused_down_residual_batched(x, st->attn_buf,
                w->q4_down_weight + Q4_BYTES((size_t)st_ffn_q4_layer_off(w, l) * d * ff),
                st->q_buf, d, ff, 1,
                w->has_x8 ? w->x8_down_weight + Q4_BYTES((size_t)st_ffn_q4_layer_off(w, l) * d * ff) : NULL);
            if (prof) t_down += st_now_sec() - t0;
            if (trace) { fprintf(stderr, "[DDEC2] l=%d FFN out (q4)\n", l); fflush(stderr); }
        } else if (w->has_q8) {
            #define Q8O(e)  ((size_t)(e) / 32 * 34)
            if (prof) t0 = st_now_sec();
            dyn_matvec_q8_fused_gate_up_batched(st->q_buf, st->k_buf,
                w->q8_gate_weight + Q8O((size_t)l * ff * d),
                w->q8_up_weight   + Q8O((size_t)l * ff * d),
                normed, ff, d, 1);
            if (prof) t_gu += st_now_sec() - t0;
            vllm_silu_mul_ctx vg = { st->q_buf, st->k_buf, ff };
            if (prof) tw = st_now_sec();
            vllm_tp_parfor(0, (ff + 15) / 16, vllm_silu_mul_worker, &vg);
            if (prof) t_silu += st_now_sec() - tw;
            /* P4: fused down projection + FFN residual — writes directly to x */
            if (prof) t0 = st_now_sec();
            dyn_matvec_q8_fused_down_residual_batched(x, st->attn_buf,
                w->q8_down_weight + Q8O((size_t)l * d * ff),
                st->q_buf, d, ff, 1);
            if (prof) t_down += st_now_sec() - t0;
            if (trace) { fprintf(stderr, "[DDEC2] l=%d FFN out (q8)\n", l); fflush(stderr); }
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
        if (getenv("VLLM_DUMP_ATTN") && !g_kv_nof32 && l == 0 && st->seq_len == 19) {
            printf("[DUMP] l0-postffn x[0:8]=%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                   x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]);
            printf("[DUMP] l0-logitsmax-todo\n");
            fflush(stdout);
        }
        if (getenv("VLLM_DUMP_LAYERS")) {
            float xm = 0.0f, nm = 0.0f, fom = 0.0f, am = 0.0f, ss = 0.0f;
            for (int _i = 0; _i < d; _i++) { float a = fabsf(x[_i]); if (a > xm) xm = a; ss += x[_i] * x[_i]; }
            for (int _i = 0; _i < d; _i++) { float a = fabsf(normed[_i]); if (a > nm) nm = a; }
            for (int _i = 0; _i < ff; _i++) { float a = fabsf(st->q_buf[_i]); if (a > fom) fom = a; }
            for (int _i = 0; _i < d; _i++) { float a = fabsf(st->attn_buf[_i]); if (a > am) am = a; }
            printf("[LAYER] l=%d pos=%d x[0:4]=%.4f %.4f %.4f %.4f xmax=%.4f xrms=%.4f nrmmax=%.4f ffmax=%.4f attnresmax=%.4f\n",
                   l, pos, x[0], x[1], x[2], x[3], xm, sqrtf(ss / (float)d), nm, fom, am);
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
        if (prof) t_loop += st_now_sec() - tw2;

    }

    /* Final norm + LM head (Q8_0 when available — axiom: fixedpoint_quantize) */
    /* 注意：final RMSNorm 是数值路径的一部分，绝不能放进 `if (prof)` —— 否则
     * 关闭 profiling 的 serve 会把 layer-35 的 FFN-norm 中间激活直接喂给 lm_head
     * （logits 扁平 + 固定吸引子 token，prefill/decode 结论不一致）。 */
    {
        double tn0 = prof ? st_now_sec() : 0.0;
        dyn_rms_norm(normed, x, w->final_norm, d, eps);
        if (prof) t_nrm += st_now_sec() - tn0;
    }
    if (trace) { fprintf(stderr, "[DDEC2] final_norm done\n"); fflush(stderr); }

    double t0 = 0.0;
    if (prof) t0 = st_now_sec();
    if (w->q4_lm_weight &&
        (st_dense_q4_first() || !w->has_q8 || !w->q8_lm_weight)) {
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
        double t_ser = t_nrm + t_rope + t_kv + t_silu;   /* P1-2：细分的串行项 */
        double t_other = t_total - t_gemm - t_attn - t_ser;
        if (t_other < 0.0) t_other = 0.0;
        st->dec_t_qkv += t_qkv;
        st->dec_t_attn += t_attn;
        st->dec_t_o += t_o;
        st->dec_t_gu += t_gu;
        st->dec_t_down += t_down;
        st->dec_t_lm += t_lm;
        st->dec_t_other += t_other;
        if (getenv("VLLM_DEC_PROF")) {
            double t_win = t_qkv + t_attn + t_o + t_gu + t_down + t_lm
                         + t_nrm + t_rope + t_kv + t_silu;
            double oth_loop = t_loop - (t_win - t_lm);   /* 层循环内窗口外残差（final-norm 含在 nrm 略有扣除） */
            if (oth_loop < 0.0) oth_loop = 0.0;
            fprintf(stderr, "[DEC-SINGLE] seq=%d qkv=%.2f attn=%.2f o=%.2f gu=%.2f down=%.2f lm=%.2f nrm=%.2f rope=%.2f kv=%.2f silu=%.2f othloop=%.2f other=%.2f total=%.2f\n",
                    st->seq_len, t_qkv * 1e3, t_attn * 1e3, t_o * 1e3,
                    t_gu * 1e3, t_down * 1e3, t_lm * 1e3,
                    t_nrm * 1e3, t_rope * 1e3, t_kv * 1e3, t_silu * 1e3,
                    oth_loop * 1e3, t_other * 1e3, (st_now_sec() - t_all0) * 1e3);
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
    int use_q4 = w->has_q4 && w->q4_q_weight &&
                  (st_dense_q4_first() || !w->has_q8 || !w->q8_q_weight);
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
        vqf_stream_layer_advance(w, l);   /* 分层驻留（仅明文 VQF） */
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

            /* RoPE：统一 dyn_mrope（半分裂配对；文本-only 退化为标准 RoPE） */
            dyn_mrope(qt, kt, hd, nh, nkv, st->cfg.head_dim_full, pos, theta);

            /* KV-cache store (float + compressed payload). */
            int have_f32 = st->k_cache[l] != NULL;   /* P3: nof32 档无 f32 正典行 */
            if (have_f32) {
                memcpy(kv_row_f32(st->k_cache[l], pos, kv_dim, st->kv_bs), kt, kv_dim * sizeof(float));
                memcpy(kv_row_f32(st->v_cache[l], pos, kv_dim, st->kv_bs), vt, kv_dim * sizeof(float));
            }
            if (st->use_kv_q8) {
                int8_t *kdst = kv_row_i8(st->k_cache_q8[l], pos, kv_dim, st->kv_bs);
                int8_t *vdst = kv_row_i8(st->v_cache_q8[l], pos, kv_dim, st->kv_bs);
                kv_quantize_per_head(kdst, vdst,
                                     st->k_scale[l] + (size_t)pos * nkv,
                                     st->v_scale[l] + (size_t)pos * nkv,
                                     kt, vt, nkv, hd);
                if (vllm_pf_q8cache_env() && have_f32)   /* P0 原型：f32 行原位改 q8 反量化 */
                    kv_dequant_roundtrip(kdst, vdst,
                                         st->k_scale[l] + (size_t)pos * nkv,
                                         st->v_scale[l] + (size_t)pos * nkv,
                                         kv_row_f32(st->k_cache[l], pos, kv_dim, st->kv_bs),
                                         kv_row_f32(st->v_cache[l], pos, kv_dim, st->kv_bs),
                                         nkv, hd);
            } else if (st->use_kv_q4) {
                q4_pack_token(q4_k_row(st->k_cache_q4[l], pos, 0, hd, st->kv_bs, nkv),
                              q4_v_row(st->v_cache_q4[l], pos, 0, hd, st->kv_bs, nkv),
                              kt, vt, nkv, hd);
                if (have_f32) {
                    memcpy(kv_row_f32(st->k_cache[l], pos, kv_dim, st->kv_bs), kt, kv_dim * sizeof(float));
                    memcpy(kv_row_f32(st->v_cache[l], pos, kv_dim, st->kv_bs), vt, kv_dim * sizeof(float));
                }
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
        if (w->cfg.is_moe) {
            /* qwen3_moe 稀疏专家 FFN：多请求合并 GroupGEMM（权重解包跨 token
             * 复用，位级一致于逐 token 路径）；nb<2 或非批式时回落逐 token。 */
            int mdone = 0;
#if defined(__AVX2__) && ST_ARCH_X86
            mdone = st_moe_ffn_sparse_q4_batch(w, l, nb, nrm, gb);
#elif defined(__aarch64__) && defined(ST_HAVE_NEON) && !ST_ARCH_X86 && ST_NEON_DOTPROD
            mdone = st_moe_ffn_sparse_q4_batch_arm(w, l, nb, nrm, gb);
#endif
            if (!mdone) {
                for (int r = 0; r < nb; r++)
                    moe_ffn_dispatch(w, l, nrm + (size_t)r * d, gb + (size_t)r * d);
            }
            for (int r = 0; r < nb; r++) {
                float *hb = h_batch + (size_t)r * d;
                const float *o = gb + (size_t)r * d;
                for (int i = 0; i < d; i++) hb[i] += o[i];
            }
        } else {
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
    }

    /* --- Scatter hidden + per-request final norm + LM head --- */
    for (int r = 0; r < nb; r++) {
        STQwenInferenceState *st = sts[r];
        st->seq_len++;
        if (st->mrope_pos > 0) st->mrope_pos++;
        memcpy(st->hidden, h_batch + (size_t)r * d, (size_t)d * sizeof(float));
        dyn_rms_norm(normed, st->hidden, w->final_norm, d, eps);
        if (w->q4_lm_weight &&
            (st_dense_q4_first() || !w->has_q8 || !w->q8_lm_weight)) {
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
 * misses in the Q·K and weighted-V loops (axiom: memory_bandwidth_reduction).
 *
 * P3 nof32: k_cache 可能为 NULL（f32 正典未分配）——此时改从 q8+scale 反量化
 * 填包（与 kv_dequant_roundtrip 同一口径：q*(scale/127)，位级一致）。 */
typedef struct {
    float *const *k_cache; float *const *v_cache;
    int8_t *const *k_q8; int8_t *const *v_q8;               /* nof32 源 */
    const float *kscale, *vscale;                            /* 逐 token 连续 [seq*nkv] */
    float *k_pack, *v_pack;
    int seq_len, nkv, hd, kv_dim, seq_stride, bs;
} vllm_pack_kv_ctx;

static void vllm_pack_kv_worker(void *ctx_, int kh) {
    vllm_pack_kv_ctx *c = ctx_;
    float *kd = c->k_pack + (size_t)kh * c->seq_stride * c->hd;
    float *vd = c->v_pack + (size_t)kh * c->seq_stride * c->hd;
    for (int s = 0; s < c->seq_len; s++) {
        if (c->k_cache) {
            const float *ks = c->k_cache[s / c->bs] + (size_t)(s % c->bs) * (size_t)c->kv_dim + (size_t)kh * c->hd;
            const float *vs = c->v_cache[s / c->bs] + (size_t)(s % c->bs) * (size_t)c->kv_dim + (size_t)kh * c->hd;
            memcpy(kd + (size_t)s * c->hd, ks, c->hd * sizeof(float));
            memcpy(vd + (size_t)s * c->hd, vs, c->hd * sizeof(float));
        } else if (c->k_q8 && c->kscale) {
            /* P3 nof32：无 f32 正典 → q8+scale 反量化（q*(scale/127)，位级 =
             * kv_dequant_roundtrip 写入 f32 行的结果，门禁见文档 §7.2 P3）。 */
            const int8_t *ks = c->k_q8[s / c->bs] + (size_t)(s % c->bs) * (size_t)c->kv_dim + (size_t)kh * c->hd;
            const int8_t *vs = c->v_q8[s / c->bs] + (size_t)(s % c->bs) * (size_t)c->kv_dim + (size_t)kh * c->hd;
            float sk = c->kscale[(size_t)s * c->nkv + kh] * (1.0f / KVQ_INV_SCALE);
            float sv = c->vscale[(size_t)s * c->nkv + kh] * (1.0f / KVQ_INV_SCALE);
            float *kdp = kd + (size_t)s * c->hd;
            float *vdp = vd + (size_t)s * c->hd;
            for (int i = 0; i < c->hd; i++) { kdp[i] = (float)ks[i] * sk; vdp[i] = (float)vs[i] * sv; }
        } else {
            /* 理论上不可达（nof32 仅 q8 档）；防御性清零避免越界语义漂移。 */
            memset(kd + (size_t)s * c->hd, 0, c->hd * sizeof(float));
            memset(vd + (size_t)s * c->hd, 0, c->hd * sizeof(float));
        }
    }
}

static void st_pack_kv_heads(float *const *k_cache, float *const *v_cache,
                             int8_t *const *k_q8, int8_t *const *v_q8,
                             const float *kscale, const float *vscale,
                             float *restrict k_pack, float *restrict v_pack,
                             int seq_len, int nkv, int hd, int kv_dim,
                             int seq_stride, int bs)
{
    vllm_pack_kv_ctx vc = { k_cache, v_cache, k_q8, v_q8, kscale, vscale,
                            k_pack, v_pack, seq_len, nkv, hd, kv_dim, seq_stride, bs };
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
        float **kfl = st->k_cache ? st->k_cache[l] : NULL;   /* P3: nof32 无 f32 块 */
        float **vfl = st->v_cache ? st->v_cache[l] : NULL;
        for (int b = 0; b < st->kv_n_blocks; b++) {
            if (kfl && !kfl[b]) {
                uint8_t *raw = (uint8_t *)kv_block_alloc_raw(fdat);
                if (!raw) { fprintf(stderr, "[KV] OOM rebuilding K block l=%d b=%d\n", l, b); return total; }
                if (!g_kv_lazy) {
                    memset(raw, 0xA5, KV_GUARD);
                    memset(raw + KV_GUARD + fdat, 0xA5, KV_GUARD);
                }
                kfl[b] = (float *)(raw + KV_GUARD);
                total += KV_GUARD + fdat + KV_GUARD;
            }
            if (vfl && !vfl[b]) {
                uint8_t *raw = (uint8_t *)kv_block_alloc_raw(fdat);
                if (!raw) { fprintf(stderr, "[KV] OOM rebuilding V block l=%d b=%d\n", l, b); return total; }
                if (!g_kv_lazy) {
                    memset(raw, 0xA5, KV_GUARD);
                    memset(raw + KV_GUARD + fdat, 0xA5, KV_GUARD);
                }
                vfl[b] = (float *)(raw + KV_GUARD);
                total += KV_GUARD + fdat + KV_GUARD;
            }
            if (st->use_kv_q8 && st->k_cache_q8 && st->k_cache_q8[l] && !st->k_cache_q8[l][b]) {
                uint8_t *raw = (uint8_t *)kv_block_alloc_raw(idat);
                if (!raw) { fprintf(stderr, "[KV] OOM rebuilding K8 block l=%d b=%d\n", l, b); return total; }
                if (!g_kv_lazy) {
                    memset(raw, 0xA5, KV_GUARD);
                    memset(raw + KV_GUARD + idat, 0xA5, KV_GUARD);
                }
                st->k_cache_q8[l][b] = (int8_t *)(raw + KV_GUARD);
                total += KV_GUARD + idat + KV_GUARD;
            }
            if (st->use_kv_q8 && st->v_cache_q8 && st->v_cache_q8[l] && !st->v_cache_q8[l][b]) {
                uint8_t *raw = (uint8_t *)kv_block_alloc_raw(idat);
                if (!raw) { fprintf(stderr, "[KV] OOM rebuilding V8 block l=%d b=%d\n", l, b); return total; }
                if (!g_kv_lazy) {
                    memset(raw, 0xA5, KV_GUARD);
                    memset(raw + KV_GUARD + idat, 0xA5, KV_GUARD);
                }
                st->v_cache_q8[l][b] = (int8_t *)(raw + KV_GUARD);
                total += KV_GUARD + idat + KV_GUARD;
            }
        }
    }
    return total;
}

/* P3: 把被 L3 驱逐的前缀块从 Q4 载荷重建回 RAM（纯解压，无任何重算）。
 * 见 include/model/vllm_safetensors.h 的语义说明。分配口径与
 * st_qwen_kv_rebuild_freed 一致（KV_GUARD 守卫 + f32/q8 两套表示），区别只在
 * 于本函数把 evict 前的 K/V 内容真正填回去，而 rebuild_freed 只"挂上空块"。 */
int l3_restore_prefix(STQwenInferenceState *st, int prefix_len) {
    if (!st || prefix_len <= 0) return 0;
    STL3State *l3 = &st->l3;
    if (!l3->fp || !l3->blocks || l3->evicted <= 0) return 0;
    /* VLLM_L3_PROF=1：rs_* 只统计本次回填（墙钟在调用方 ist_reset 里计）。 */
    int prof = l3_prof_on();

    int nl = st->weights.n_layers_allocated;
    int nkv = st->cfg.n_kv_heads;
    int hd = st->cfg.head_dim;
    int kv_dim = nkv * hd;
    int bs = st->kv_bs > 0 ? st->kv_bs : 32;
    if (nl <= 0 || kv_dim <= 0 || hd <= 0) return 0;

    size_t fdat = (size_t)bs * (size_t)kv_dim * sizeof(float);
    size_t idat = (size_t)bs * (size_t)kv_dim;
    int n_blocks = (prefix_len + bs - 1) / bs;
    if (n_blocks > l3->max_blocks) n_blocks = l3->max_blocks;
    if (n_blocks > st->kv_n_blocks) n_blocks = st->kv_n_blocks;

    int restored = 0;
    int failed = 0;
    for (int l = 0; l < nl; l++) {
        float  **kf = st->k_cache ? st->k_cache[l] : NULL;
        float  **vf = st->v_cache ? st->v_cache[l] : NULL;
        int8_t **k8 = (st->use_kv_q8 && st->k_cache_q8) ? st->k_cache_q8[l] : NULL;
        int8_t **v8 = (st->use_kv_q8 && st->v_cache_q8) ? st->v_cache_q8[l] : NULL;
        int has_f = (kf != NULL) && (vf != NULL);
        int has_i = (k8 != NULL) && (v8 != NULL);
        if (!has_f && !has_i) continue;
        for (int b = 0; b < n_blocks; b++) {
            STL3Block *bm = &l3->blocks[(size_t)l * (size_t)l3->max_blocks + (size_t)b];
            if (!bm->on_disk) continue;
            int need_f = has_f && (!kf[b] || !vf[b]);
            int need_i = has_i && (!k8[b] || !v8[b]);
            if (!need_f && !need_i) continue;   /* 该块仍在 RAM，无需重建 */
            if (need_f) {
                L3P_T0(t_al);
                uint8_t *rk = (uint8_t *)kv_block_alloc_raw(fdat);
                uint8_t *rv = (uint8_t *)kv_block_alloc_raw(fdat);
                L3P_ACC(t_al, rs_alloc_s);
                L3P_N(rs_alloc_calls, 2);
                if (!rk || !rv) {
                    if (rk) st_qwen_kv_free_block(rk + KV_GUARD, fdat);
                    if (rv) st_qwen_kv_free_block(rv + KV_GUARD, fdat);
                    fprintf(stderr, "[L3] OOM restoring f32 block l=%d b=%d\n", l, b);
                    return -1;   /* 前缀不完整：调用方不得复用 */
                }
                if (!g_kv_lazy) {
                    memset(rk, 0xA5, KV_GUARD);
                    memset(rk + KV_GUARD + fdat, 0xA5, KV_GUARD);
                    memset(rv, 0xA5, KV_GUARD);
                    memset(rv + KV_GUARD + fdat, 0xA5, KV_GUARD);
                }
                kf[b] = (float *)(rk + KV_GUARD);
                vf[b] = (float *)(rv + KV_GUARD);
            }
            if (need_i) {
                L3P_T0(t_ai);
                uint8_t *rk = (uint8_t *)kv_block_alloc_raw(idat);
                uint8_t *rv = (uint8_t *)kv_block_alloc_raw(idat);
                L3P_ACC(t_ai, rs_alloc_s);
                L3P_N(rs_alloc_calls, 2);
                if (!rk || !rv) {
                    if (rk) st_qwen_kv_free_block(rk + KV_GUARD, idat);
                    if (rv) st_qwen_kv_free_block(rv + KV_GUARD, idat);
                    fprintf(stderr, "[L3] OOM restoring q8 block l=%d b=%d\n", l, b);
                    return -1;   /* 前缀不完整：调用方不得复用 */
                }
                if (!g_kv_lazy) {
                    memset(rk, 0xA5, KV_GUARD);
                    memset(rk + KV_GUARD + idat, 0xA5, KV_GUARD);
                    memset(rv, 0xA5, KV_GUARD);
                    memset(rv + KV_GUARD + idat, 0xA5, KV_GUARD);
                }
                k8[b] = (int8_t *)(rk + KV_GUARD);
                v8[b] = (int8_t *)(rv + KV_GUARD);
            }
            const float *ksc = (need_i && st->k_scale)
                             ? st->k_scale[l] + (size_t)b * (size_t)bs * (size_t)nkv : NULL;
            const float *vsc = (need_i && st->v_scale)
                             ? st->v_scale[l] + (size_t)b * (size_t)bs * (size_t)nkv : NULL;
            L3P_T0(t_fl);
            int frc = l3_fill_block_from_disk(l3, bm, b,
                                              has_f ? kf : NULL, has_f ? vf : NULL,
                                              has_i ? k8 : NULL, has_i ? v8 : NULL,
                                              ksc, vsc, kv_dim, bs, hd);
            L3P_ACC(t_fl, rs_fill_s);
            if (frc != 0) {
                fprintf(stderr, "[L3] restore failed l=%d b=%d (payload unreadable?)\n",
                        l, b);
                failed = 1;   /* 继续重建其余块，但最终必须报 -1 */
                continue;
            }
            restored++;
            L3P_N(rs_blocks, 1);
        }
    }
    if (restored > 0) {
        fprintf(stderr, "[L3] restored %d prefix blocks from Q4 payload (prefix=%d)\n",
                restored, prefix_len);
        fflush(stderr);
    }
    /* 返回契约：>=0 = 前缀范围内已无缺失块（0 = 本来就都在 RAM）；
     * -1 = 有块重建失败，前缀不完整 —— 调用方必须放弃复用。 */
    return failed ? -1 : restored;
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

/* =====================================================================
 * VLLM_PFDUMP=<dir> [VLLM_PFDUMPTAG=<tag>]（serve 前缀复用分叉对拍临时工具）
 * prefill 完成后把本次写入的 KV 逐行哈希 + 末行 logits 写盘：
 *   <dir>/pf_<tag>_b<base>_n<n>.bin
 * 用于"全量 prefill (base=0,n=258)" vs "前缀复用部分 prefill (base=123,n=135)"
 * 的逐层逐行数值对拍（定位 serve 复用≠全量输出的首差异点，方案文档 §7.2 P4）。
 * 布局（小端）：
 *   "PFD2" + u32 mode(=1 逐行)
 *   8×u64: n, base, nl, kv_bs, kv_dim, nkv, hd, vc
 *   u64 flags                      bit0=f32 行存在, bit1=q8 行存在
 *   logits f32[vc]                 （本 prefill 末 token）
 *   per layer l in [0,nl):
 *     n × 6 u64 = [Kf32,Vf32,Kq8,Vq8,Kscale,Vscale]  第 i 行 = token(base+i)
 *   （不存在的视图 = 0）
 */
static uint64_t pf_hash64(const void *p, size_t len) {
    const unsigned char *b = (const unsigned char *)p;
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < len; i++) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}
/* VLLM_PFDBG2=<path>：layer0/首块逐阶段行哈希（embed→RMSNorm→QKV→MRoPE），
 * 用于二分"批宽非不变"首个差异进入点（A n258 vs B n123）。 */
static void pf_b2(const char *tag, const void *p, size_t bytes) {
    const char *path = getenv("VLLM_PFDBG2");
    if (!path || !path[0] || !p) return;
    FILE *f = fopen(path, "a");
    if (!f) return;
    fprintf(f, "%s %016llx\n", tag,
            (unsigned long long)pf_hash64(p, bytes));
    fclose(f);
}
static int pf_scratch_on(void) {
    const char *e = getenv("VLLM_PFSCRATCH");
    return e && e[0] == '1';
}
static void pf_dump_post(STQwenInferenceState *st, int base, int n_tokens) {
    const char *dir = getenv("VLLM_PFDUMP");
    if (!dir || !dir[0] || n_tokens <= 0) return;
    const char *tag = getenv("VLLM_PFDUMPTAG");
    if (!tag || !tag[0]) tag = "x";
    int nl = st->weights.n_layers_allocated;
    int kv_dim = st->cfg.n_kv_heads * st->cfg.head_dim;
    int kv_bs = st->kv_bs;
    int nkv = st->cfg.n_kv_heads;
    int hd = st->cfg.head_dim;
    int vc = st->cfg.vocab_size;
    char path[512];
    snprintf(path, sizeof(path), "%s/pf_%s_b%d_n%d.bin", dir, tag, base, n_tokens);
    FILE *f = fopen(path, "wb");
    if (!f) return;
    uint64_t u64[8];
    fwrite("PFD2", 1, 4, f);
    uint32_t mode = 1;
    fwrite(&mode, 4, 1, f);
    u64[0] = (uint64_t)n_tokens; u64[1] = (uint64_t)base; u64[2] = (uint64_t)nl;
    u64[3] = (uint64_t)kv_bs;    u64[4] = (uint64_t)kv_dim; u64[5] = (uint64_t)nkv;
    u64[6] = (uint64_t)hd;       u64[7] = (uint64_t)vc;
    fwrite(u64, 8, 8, f);
    int has_f32 = (st->k_cache != NULL);
    int has_q8  = (st->k_cache_q8 != NULL);
    uint64_t flags = (has_f32 ? 1u : 0u) | (has_q8 ? 2u : 0u);
    fwrite(&flags, 8, 1, f);
    if (st->logits) fwrite(st->logits, 4, (size_t)vc, f);
    /* per layer, per row (token = base+i), six 64-bit hashes */
    uint64_t h[6];
    for (int l = 0; l < nl; l++) {
        for (int i = 0; i < n_tokens; i++) {
            int pos = base + i;
            int b = pos / kv_bs, r = pos % kv_bs;
            const float *kf = NULL, *vf = NULL;
            const int8_t *kq = NULL, *vq = NULL;
            const float *ksc = NULL, *vsc = NULL;
            if (has_f32 && st->k_cache[l] && st->k_cache[l][b]) {
                kf = st->k_cache[l][b] + (size_t)r * kv_dim;
                if (st->v_cache[l] && st->v_cache[l][b])
                    vf = st->v_cache[l][b] + (size_t)r * kv_dim;
            }
            if (has_q8 && st->k_cache_q8[l] && st->k_cache_q8[l][b]) {
                kq = st->k_cache_q8[l][b] + (size_t)r * kv_dim;
                if (st->v_cache_q8[l] && st->v_cache_q8[l][b])
                    vq = st->v_cache_q8[l][b] + (size_t)r * kv_dim;
            }
            if (st->k_scale) ksc = st->k_scale[l] + (size_t)pos * nkv;
            if (st->v_scale) vsc = st->v_scale[l] + (size_t)pos * nkv;
            h[0] = kf ? pf_hash64(kf, (size_t)kv_dim * 4) : 0;
            h[1] = vf ? pf_hash64(vf, (size_t)kv_dim * 4) : 0;
            h[2] = kq ? pf_hash64(kq, (size_t)kv_dim) : 0;
            h[3] = vq ? pf_hash64(vq, (size_t)kv_dim) : 0;
            h[4] = ksc ? pf_hash64(ksc, (size_t)nkv * 4) : 0;
            h[5] = vsc ? pf_hash64(vsc, (size_t)nkv * 4) : 0;
            fwrite(h, 8, 6, f);
        }
    }
    fclose(f);
}

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
    int pf_base = st->seq_len;   /* VLLM_PFDUMP 对拍：本调用行的起点 */

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
     * --sparse-attn) ranks blocks by this real attention potential.
     * P3 前缀复用：pf_base 是本轮复用的历史行数，[0, pf_base) 的 KV 是上一轮
     * 原样留下的，其重要性图必须一并保留 —— 否则复用轮只重算最后 1 个 token，
     * decode 的稀疏选块预算（前一半由 prefill_importance 名次决定）会整体落空、
     * 退化成 probe-only。pf_base == 0 时清区段就是 [0, max)，与原实现逐位相同。
     * imp_head 只是本轮的累加器，整表清零即可。 */
    memset(st->imp_head, 0, (size_t)st->max_kv_slots * nh * sizeof(float));
    {
        size_t imp_off = (size_t)pf_base;
        if (imp_off > (size_t)st->max_kv_slots) imp_off = (size_t)st->max_kv_slots;
        memset(st->prefill_importance + imp_off, 0,
               ((size_t)st->max_kv_slots - imp_off) * sizeof(float));
    }

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
    if (pf_scratch_on()) {
        fprintf(stderr, "[KPACK] alloc %p (%zu B)\n",
                (void *)k_pack, pack_elems * 2 * sizeof(float));
        fflush(stderr);
    }

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
        if (tok_offset == 0) pf_b2("emb0", hidden_b, (size_t)d * 4);

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
            vqf_stream_layer_advance(w, l);   /* 分层驻留：驱逐 l-N / 预读 l+1 */
            g_npu_layer = l;   /* transparent NPU offload: per-layer op models */
            t0 = st_now_sec();
            /* --- Step 1: batched RMSNorm (ONE omp region, was nb x 2 regions) --- */
            dyn_rms_norm_batch(nrm, h_buf, w->attn_norm + (size_t)l * d, nb, d, eps);
            t_other += st_now_sec() - t0;
            if (l == 0 && tok_offset == 0) pf_b2("nrm0", nrm, (size_t)d * 4);

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
            if (l == 0 && tok_offset == 0) {
                pf_b2("qq0", qb, (size_t)q_rows * 4);
                pf_b2("qk0", kb, (size_t)kv_dim * 4);
            }
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

                /* RoPE：统一 dyn_mrope（半分裂配对；文本-only 退化为标准 RoPE） */
                dyn_mrope(qt, kt, hd, nh, nkv, st->cfg.head_dim_full, pos, theta);

                /* KV-cache store (float + compressed payload for attention) */
                int cl = st->cache_len[l];
                int have_f32 = st->k_cache[l] != NULL;   /* P3: nof32 档无 f32 正典行 */
                if (l == 0 && tok_offset == 0 && t == 0)
                    pf_b2("kt0", kt, (size_t)kv_dim * 4);
                if (cl >= st->kv_n_blocks * st->kv_bs)
                    fprintf(stderr, "[KV-OOB] prefill cl=%d cap=%d l=%d tok=%d\n", cl,
                            st->kv_n_blocks * st->kv_bs, l, tok_offset + t);
                if (have_f32) {
                    memcpy(kv_row_f32(st->k_cache[l], cl, kv_dim, st->kv_bs), kt, kv_dim * sizeof(float));
                    memcpy(kv_row_f32(st->v_cache[l], cl, kv_dim, st->kv_bs), vt, kv_dim * sizeof(float));
                }
                if (KV_GUARD && !g_kv_lazy && have_f32) {
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
                    /* P0 原型：f32 正典行改存 q8 反量化值（模拟"只存 q8"后
                     * prefill 历史段注意力将看到的值；质量门测量用）。 */
                    if (vllm_pf_q8cache_env() && have_f32) {
                        kv_dequant_roundtrip(kdst, vdst,
                                             st->k_scale[l] + (size_t)cl * nkv,
                                             st->v_scale[l] + (size_t)cl * nkv,
                                             kt, vt, nkv, hd);
                        memcpy(kv_row_f32(st->k_cache[l], cl, kv_dim, st->kv_bs), kt,
                               kv_dim * sizeof(float));
                        memcpy(kv_row_f32(st->v_cache[l], cl, kv_dim, st->kv_bs), vt,
                               kv_dim * sizeof(float));
                    }
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
                if (g_kv_nof32) {
                    /* P3：无 f32 正典 → pack 直接从 q8+scale 反量化填包 */
                    st_pack_kv_heads(NULL, NULL,
                                     st->k_cache_q8[l], st->v_cache_q8[l],
                                     st->k_scale[l], st->v_scale[l],
                                     k_pack, v_pack, seq_len, nkv, hd, kv_dim,
                                     seq_stride, st->kv_bs);
                } else {
                    st_pack_kv_heads(st->k_cache[l], st->v_cache[l],
                                     NULL, NULL, NULL, NULL,
                                     k_pack, v_pack, seq_len, nkv, hd, kv_dim,
                                     seq_stride, st->kv_bs);
                }
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
            if (w->cfg.is_moe) {
                /* qwen3_moe 稀疏专家 FFN：Step2 批融合（VLLM_MOE_BATCH=1，x86）
                 * 或逐 token 回落；残差并入不变。 */
                int mdone = 0;
#if defined(__AVX2__) && ST_ARCH_X86
                mdone = st_moe_ffn_sparse_q4_batch(w, l, nb, nrm, qb);
#elif defined(__aarch64__) && defined(ST_HAVE_NEON) && !ST_ARCH_X86 && ST_NEON_DOTPROD
                mdone = st_moe_ffn_sparse_q4_batch_arm(w, l, nb, nrm, qb);
#endif
                if (!mdone) {
                    for (int t = 0; t < nb; t++) {
                        float *o = qb + (size_t)t * d;
                        moe_ffn_dispatch(w, l, nrm + (size_t)t * d, o);
                    }
                }
                for (int t = 0; t < nb; t++) {
                    float *hb = h_buf + (size_t)t * d;
                    const float *o = qb + (size_t)t * d;
                    for (int i = 0; i < d; i++) hb[i] += o[i];
                }
                t_gu += st_now_sec() - t0;
                HCK("gateup");
                t_down += 0.0;
                HCK("down");
            } else {
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
            }

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

                if (st->k_cache[l] && st->v_cache[l]) {   /* P3: nof32 无 f32 正典 */
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

            /* 诊断（与 decode 的 [LAYER] 对称）：prefill 逐层 x 统计。
             * 对拍口径：prefill 位置 p 的 x ≡ decode [LAYER] pos=p（同前缀、因果）。
             * 每 mini-batch 末 token 一行；配合 --prefill-batch 1 得逐位置轨迹。 */
            if (getenv("VLLM_DUMP_LAYERS")) {
                const float *xh = h_buf + (size_t)(nb - 1) * d;
                float xm = 0.0f, ss = 0.0f;
                for (int _i = 0; _i < d; _i++) { float a = fabsf(xh[_i]); if (a > xm) xm = a; ss += xh[_i] * xh[_i]; }
                printf("[PLAYER] l=%d pos=%d x[0:4]=%.4f %.4f %.4f %.4f xmax=%.4f xrms=%.4f\n",
                       l, tok_offset + nb - 1, xh[0], xh[1], xh[2], xh[3], xm, sqrtf(ss / (float)d));
                fflush(stdout);
            }
        }

        /* 诊断：VLLM_DUMP_POS=<p[,p...]> → 打印指定绝对位置 p 的 top-8 原始 logits。
         * p 的含义与 decode 的 [LTOP] seq=p 一致（该位置的预测 = 上下文 0..p）。 */
        {
            const char *dpos = getenv("VLLM_DUMP_POS");
            if (dpos && dpos[0]) {
                const char *q = dpos;
                while (*q) {
                    int want = atoi(q);
                    while (*q && *q != ',') q++;
                    if (*q == ',') q++;
                    if (want < tok_offset || want >= tok_offset + nb) continue;
                    float *xh = h_buf + (size_t)(want - tok_offset) * d;
                    float *scratch = normed_b;   /* 层循环已结束，可复用 */
                    dyn_rms_norm(scratch, xh, w->final_norm, d, eps);
                    if (use_q4 && w->q4_lm_weight)
                        dyn_matvec_q4_q8(st->logits, w->q4_lm_weight, scratch, vc, d);
                    else
                        dyn_matvec_q8(st->logits, w->q8_lm_weight, scratch, vc, d);
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
                    printf("[PLOGIT] pos=%d", want);
                    for (int k = 0; k < topn; k++) printf(" %d=%.3f", ti[k], tv[k]);
                    printf("\n");
                    fflush(stdout);
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
        /* P3：下界取 pf_base。本轮只对 [pf_base, seq_len) 这几个 query 做了
         * attention，虽然 imp_head 对复用行也有累加，但那只是**极少几个 query**
         * 的观测；而 [0, pf_base) 现有的重要性图是上一轮那次「query 覆盖整段
         * 上下文」的 prefill 算出来的，更接近全量 prefill 的基准。用稀疏观测
         * 覆盖它，等于把复用轮的重要性图降级（复用轮 n_tokens=1 时尤其致命）。
         * pf_base == 0 时下界为 0，与原实现逐位相同。 */
        for (int s = pf_base; s < st->seq_len + n_tokens; s++) {
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

    /* 诊断（与 decode 的 [LTOP] 对称）：prefill 末位 top-8 原始 logits。
     * 对拍口径：prefill n=N 的末位 ≡ decode [LTOP] seq=N-1。 */
    if (getenv("VLLM_DUMP_LOGITS")) {
        float *lh = hidden_b + (size_t)(last_nb - 1) * d;
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
        printf("[PLTOP] n=%d pos=%d", n_tokens, n_tokens - 1);
        for (int k = 0; k < topn; k++) printf(" %d=%.3f", ti[k], tv[k]);
        printf("\n[PHID] x[0:6]=%.5f %.5f %.5f %.5f %.5f %.5f nrm[0:6]=%.5f %.5f %.5f %.5f %.5f %.5f\n",
               lh[0], lh[1], lh[2], lh[3], lh[4], lh[5],
               normed_b[0], normed_b[1], normed_b[2], normed_b[3], normed_b[4], normed_b[5]);
        fflush(stdout);
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
    pf_dump_post(st, pf_base, n_tokens);   /* VLLM_PFDUMP 对拍（env 门控） */
    if (pf_scratch_on()) {
        fprintf(stderr, "[KPACK] free  %p\n", (void *)k_pack);
        fflush(stderr);
    }
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
    const float *const *ds_features, int n_ds,  /* optional DeepStack [n_ds][n_vis_tokens, dim] */
    int start)  /* P4 prefix reuse: skip writing the first `start` token rows
                   (caller must have set cache_len[l]/seq_len = start first) */
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

    /* P4 prefix reuse: clamp `start` (skip rows) and refuse a full-coverage
     * call (0 tail rows would leave stale logits; the caller keeps
     * keep < n_tokens so this only fires on misuse). */
    if (start < 0) start = 0;
    if (start > n_tokens) start = n_tokens;
    if (start >= n_tokens) {
        st_aligned_free(wbuf);
        free(pos_t_arr); free(pos_h_arr); free(pos_w_arr);
        free(vis_row_map);
        return -1;
    }

    /* Process tokens in mini-batches through all layers; the first `start`
     * rows are already resident (prefix reuse), we only write the tail. The
     * MRoPE position arrays above are absolute over the whole sequence, so
     * tail rows land at exactly the positions a full prefill would use. */
    int tok_offset = start;
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
            vqf_stream_layer_advance(w, l);   /* 分层驻留：驱逐 l-N / 预读 l+1 */
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
            int have_f32 = st->k_cache[l] != NULL;   /* P3: nof32 档无 f32 正典行 */
            if (have_f32) {
                for (int t = 0; t < nb; t++) {
                    memcpy(kv_row_f32(st->k_cache[l], cl + t, kv_dim, st->kv_bs), kb + (size_t)t * kv_dim,
                           kv_dim * sizeof(float));
                    memcpy(kv_row_f32(st->v_cache[l], cl + t, kv_dim, st->kv_bs), vb + (size_t)t * kv_dim,
                           kv_dim * sizeof(float));
                }
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
                    if (vllm_pf_q8cache_env() && have_f32) {   /* P0 原型：f32 行原位改 q8 反量化 */
                        kv_dequant_roundtrip(kdst, vdst,
                                             st->k_scale[l] + (size_t)(cl + t) * nkv,
                                             st->v_scale[l] + (size_t)(cl + t) * nkv,
                                             kv_row_f32(st->k_cache[l], cl + t, kv_dim, st->kv_bs),
                                             kv_row_f32(st->v_cache[l], cl + t, kv_dim, st->kv_bs),
                                             nkv, hd);
                    }
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

            /* 诊断（与 decode 的 [LAYER] 对称）：prefill 逐层 x 统计。 */
            if (getenv("VLLM_DUMP_LAYERS")) {
                const float *xh = h_buf + (size_t)(nb - 1) * d;
                float xm = 0.0f, ss = 0.0f;
                for (int _i = 0; _i < d; _i++) { float a = fabsf(xh[_i]); if (a > xm) xm = a; ss += xh[_i] * xh[_i]; }
                printf("[PLAYER] l=%d pos=%d x[0:4]=%.4f %.4f %.4f %.4f xmax=%.4f xrms=%.4f\n",
                       l, start + tok_offset + nb - 1, xh[0], xh[1], xh[2], xh[3], xm, sqrtf(ss / (float)d));
                fflush(stdout);
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

    /* 诊断（与 decode 的 [LTOP] 对称）：prefill 最后一位的 top-8 原始 logits。
     * pos = 该 token 的绝对位置，可与 decode 的 [LTOP] seq=pos 直接对拍。 */
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
        printf("[PLTOP] n=%d pos=%d keep=%d", n_tokens, start + n_tokens - 1, start);
        for (int k = 0; k < topn; k++) printf(" %d=%.3f", ti[k], tv[k]);
        printf("\n[PHID] x[0:6]=%.5f %.5f %.5f %.5f %.5f %.5f nrm[0:6]=%.5f %.5f %.5f %.5f %.5f %.5f\n",
               last_hidden[0], last_hidden[1], last_hidden[2], last_hidden[3], last_hidden[4], last_hidden[5],
               normed_b[0], normed_b[1], normed_b[2], normed_b[3], normed_b[4], normed_b[5]);
        fflush(stdout);
    }

    st->seq_len += (n_tokens - start);   /* prefix rows pre-counted by ist_reset */
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
                                               grid_thw, 1, NULL, 0, 0);
}

/* ================================================================

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
        size_t fdat = (size_t)st->kv_bs *
                      (size_t)(st->cfg.n_kv_heads * st->cfg.head_dim) * sizeof(float);
        for (int l = 0; l < nl; l++) {
            if (st->k_cache[l]) {
                for (int b = 0; b < st->kv_n_blocks; b++)
                    if (st->k_cache[l][b]) kv_block_free_raw((uint8_t *)st->k_cache[l][b] - KV_GUARD, fdat);
                cf_pg_free(st->k_cache[l]);
            }
        }
        cf_pg_free(st->k_cache);
    }
    if (st->v_cache) {
        size_t fdat = (size_t)st->kv_bs *
                      (size_t)(st->cfg.n_kv_heads * st->cfg.head_dim) * sizeof(float);
        for (int l = 0; l < nl; l++) {
            if (st->v_cache[l]) {
                for (int b = 0; b < st->kv_n_blocks; b++)
                    if (st->v_cache[l][b]) kv_block_free_raw((uint8_t *)st->v_cache[l][b] - KV_GUARD, fdat);
                cf_pg_free(st->v_cache[l]);
            }
        }
        cf_pg_free(st->v_cache);
    }
    cf_pg_free(st->scores_buf);
    cf_pg_free(st->prefill_importance);
    cf_pg_free(st->imp_head);
    if (st->k_cache_q8) {
        size_t idat = (size_t)st->kv_bs *
                      (size_t)(st->cfg.n_kv_heads * st->cfg.head_dim);
        for (int l = 0; l < nl; l++) {
            if (st->k_cache_q8[l]) {
                for (int b = 0; b < st->kv_n_blocks; b++)
                    if (st->k_cache_q8[l][b]) kv_block_free_raw((uint8_t *)st->k_cache_q8[l][b] - KV_GUARD, idat);
                cf_pg_free(st->k_cache_q8[l]);
            }
        }
        cf_pg_free(st->k_cache_q8);
    }
    if (st->v_cache_q8) {
        size_t idat = (size_t)st->kv_bs *
                      (size_t)(st->cfg.n_kv_heads * st->cfg.head_dim);
        for (int l = 0; l < nl; l++) {
            if (st->v_cache_q8[l]) {
                for (int b = 0; b < st->kv_n_blocks; b++)
                    if (st->v_cache_q8[l][b]) kv_block_free_raw((uint8_t *)st->v_cache_q8[l][b] - KV_GUARD, idat);
                cf_pg_free(st->v_cache_q8[l]);
            }
        }
        cf_pg_free(st->v_cache_q8);
    }
    if (st->k_cache_q4) {
        size_t blk4 = (size_t)st->kv_bs * (size_t)st->cfg.n_kv_heads *
                      (size_t)(st->cfg.head_dim / 64) * Q4_PAYLOAD_LEN;
        for (int l = 0; l < nl; l++) {
            if (st->k_cache_q4[l]) {
                for (int b = 0; b < st->kv_n_blocks; b++)
                    if (st->k_cache_q4[l][b]) kv_block_free_raw((uint8_t *)st->k_cache_q4[l][b] - KV_GUARD, blk4);
                cf_pg_free(st->k_cache_q4[l]);
            }
        }
        cf_pg_free(st->k_cache_q4);
    }
    if (st->v_cache_q4) {
        size_t blk4 = (size_t)st->kv_bs * (size_t)st->cfg.n_kv_heads *
                      (size_t)(st->cfg.head_dim / 64) * Q4_PAYLOAD_LEN;
        for (int l = 0; l < nl; l++) {
            if (st->v_cache_q4[l]) {
                for (int b = 0; b < st->kv_n_blocks; b++)
                    if (st->v_cache_q4[l][b]) kv_block_free_raw((uint8_t *)st->v_cache_q4[l][b] - KV_GUARD, blk4);
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
 * Snapshot the first n_tokens KV rows plus the token ids and the model
 * geometry into one file. The serve layer indexes these files by the FNV-1a
 * hash of the token sequence; on a later request (also after a process
 * restart) the file whose tokens share the longest prefix is loaded and only
 * the suffix is prefilled.
 *
 * v2 (q8+scale, P1b): q8 档写出"q8 反量化可重建"快照——每 token 的 K/V i8 行
 * + per-head f32 scale，无 f32 依赖 → q8 语义下恢复与源逐位一致（DKVTEST 门禁）。
 * v1 (f32, legacy): f32 精确存储，恢复与"精确 f32 prefill"逐位一致；q8 档旧的
 * v1 快照仍可兼容读（仅在 f32 正典存在时可用；P3 nof32 档 load v1 拒绝 rc=-1）。
 *
 * File layout (all little-endian, portable across x86/aarch64):
 *   [0..3]    magic  "VLKV" (0x564C4B56)
 *   [4..7]    version = 1 | 2
 *   [8..11]   n_layers
 *   [12..15]  nkv (KV heads)
 *   [16..19]  hd (head dim)
 *   [20..23]  kv_bs (KV block size in positions)
 *   [24..27]  file_n_tokens (rows saved)
 *   [28..31]  kv_dim (nkv * hd)
 *   [32..39]  token hash (FNV-1a 64 over the saved token ids)
 *   [40..43]  reserved (zero)
 *   [44..63]  reserved (zero)
 *   tokens:   file_n_tokens * int32
 *   payload:
 *     v1: per layer: per full kv_bs block: K block (kv_bs*kv_dim f32)
 *         then V block; a trailing partial block stores only its valid
 *         rows (same K-then-V order).
 *     v2: per layer: per token: K i8 row (kv_dim), V i8 row (kv_dim),
 *         k_scale[nkv] f32, v_scale[nkv] f32（无块填充，逐 token 对齐）。 */
#define DKV_MAGIC    0x564C4B56u
#define DKV_VERSION  2                 /* 当前写出版本（v2=q8+scale）；load 兼容 v1 */
#define DKV_VERSION_V1 1
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
    /* v2（q8+scale，P1b）：q8 量化副本可用时写出，不再依赖 f32 正典；否则 v1。 */
    int ver = (st->use_kv_q8 && st->k_cache_q8 && st->k_scale) ? 2 : 1;
    uint32_t m = DKV_MAGIC, v = (uint32_t)ver;
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
        if (ver == 2) {
            /* v2 payload：逐 token K_i8/V_i8 行 + per-head f32 scale */
            for (int t = 0; t < n_tokens; t++) {
                if (fwrite(kv_row_i8(st->k_cache_q8[l], t, kv_dim, bs), 1,
                           (size_t)kv_dim, fp) != (size_t)kv_dim) { rc = -1; break; }
                if (fwrite(kv_row_i8(st->v_cache_q8[l], t, kv_dim, bs), 1,
                           (size_t)kv_dim, fp) != (size_t)kv_dim) { rc = -1; break; }
                if (fwrite(st->k_scale[l] + (size_t)t * nkv, sizeof(float),
                           (size_t)nkv, fp) != (size_t)nkv) { rc = -1; break; }
                if (fwrite(st->v_scale[l] + (size_t)t * nkv, sizeof(float),
                           (size_t)nkv, fp) != (size_t)nkv) { rc = -1; break; }
            }
        } else {
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
        if (m != DKV_MAGIC || (v != DKV_VERSION && v != DKV_VERSION_V1)) rc = -1;
        if (fnl != (uint32_t)nl || fnkv != (uint32_t)nkv || fhd != (uint32_t)hd ||
            fbs != (uint32_t)bs || fdim != (uint32_t)kv_dim) rc = -1;
        if (n_tokens > (int)ftok) rc = -1;      /* requested prefix exceeds file */
        if (v == DKV_VERSION_V1 &&
            (uint32_t)n_tokens > (uint32_t)st->kv_n_blocks * (uint32_t)bs)
            rc = -1;
        if (v == DKV_VERSION_V1 && (!st->k_cache || !st->k_cache[0]))
            rc = -1;   /* P3 nof32：无 f32 正典行，v1 f32 快照不可恢复 */
        if (v == DKV_VERSION &&
            (!st->use_kv_q8 || !st->k_cache_q8 || !st->k_scale)) rc = -1;
    }
    if (rc == 0) {
        /* Skip the token ids (they were verified against the live request by
         * the serve layer's LCP; only geometry needs re-checking). */
        if (fseek(fp, (long)((size_t)ftok * sizeof(int)), SEEK_CUR) != 0)
            rc = -1;
    }
    if (v == DKV_VERSION) {
        /* v2（P1b）：q8+scale 载荷。逐层逐 token：K_i8, V_i8, k_scale[nkv],
         * v_scale[nkv]；f32 正典若仍存在（P3 前）用 kv_dequant_roundtrip 回填，
         * 与 store 同一量化路径。 */
        const int64_t rec = (int64_t)kv_dim * 2 + (int64_t)nkv * 2;
        for (int l = 0; rc == 0 && l < nl; l++) {
            for (int t = 0; rc == 0 && t < n_tokens; t++) {
                int8_t *k8 = kv_row_i8(st->k_cache_q8[l], t, kv_dim, bs);
                int8_t *v8 = kv_row_i8(st->v_cache_q8[l], t, kv_dim, bs);
                float *ks = st->k_scale[l] + (size_t)t * nkv;
                float *vs = st->v_scale[l] + (size_t)t * nkv;
                if (fread(k8, 1, (size_t)kv_dim, fp) != (size_t)kv_dim ||
                    fread(v8, 1, (size_t)kv_dim, fp) != (size_t)kv_dim ||
                    fread(ks, sizeof(float), (size_t)nkv, fp) != (size_t)nkv ||
                    fread(vs, sizeof(float), (size_t)nkv, fp) != (size_t)nkv) {
                    rc = -1;
                    break;
                }
                if (st->k_cache && st->k_cache[l] && st->k_cache[l][t / bs] &&
                    st->v_cache && st->v_cache[l] && st->v_cache[l][t / bs]) {
                    kv_dequant_roundtrip(k8, v8, ks, vs,
                                         kv_row_f32(st->k_cache[l], t, kv_dim, bs),
                                         kv_row_f32(st->v_cache[l], t, kv_dim, bs),
                                         nkv, hd);
                }
            }
            if (rc == 0 && (int64_t)ftok > n_tokens) {
                if (fseek(fp, (long)(((int64_t)ftok - n_tokens) * rec),
                          SEEK_CUR) != 0) rc = -1;
            }
        }
    } else {
        /* v1：f32 载荷（历史快照，兼容读）。 */
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
        /* v1：重建派生压缩缓存，与 store 同路径（位级一致）。 */
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
    }
    fclose(fp);
    if (rc != 0) return rc;
    for (int l = 0; l < nl; l++) st->cache_len[l] = n_tokens;
    st->seq_len = n_tokens;
    return 0;
}
