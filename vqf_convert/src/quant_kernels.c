/* Auto-extracted from src/model/vllm_safetensors.c (sha256 da768fbff3b10063a3815f8786b536892e0b7ec68ef6efe8a10180483bd27cc3)
 * Re-run tools/extract_quant.py after engine edits; verify sha256.
 * ================================================================ */

#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

/* ---------- f16 helpers (f32_to_f16_bits, src lines 2873-2882) ---------- */
/* f32 → f16：IEEE 754 binary16，round-to-nearest-even，含 subnormal。
 *
 * 旧实现（与 ggml GGML_FP32_TO_FP16 不一致，实测造成两处真实精度损失）：
 *   1) exp<=0 直接返回 0 —— 下溢清零（FTZ）：任何 |x| < 2^-14 的 block scale
 *      都被抹成 0 → 该块 32 个权重全变 0。实测 q8_gate 有 2.21% 的块被清零
 *      （llama.cpp 的 Q8_0 用 subnormal 正常表示，d 最小 4.9e-05）。
 *   2) 尾数 `(u>>13) & 0x3FF` 只截断不舍入 → 每个 scale 最多 ~0.1% 相对误差。
 *   3) 溢出直接给 inf，未按 RNE 处理。
 * 本实现与 ggml 的 quantize_row_q8_0/GGML_FP32_TO_FP16 数值口径一致。 */
static inline uint16_t f32_to_f16_bits(float x) {
    uint32_t u; memcpy(&u, &x, 4);
    uint32_t sign = (u >> 16) & 0x8000u;
    uint32_t ef   = (u >> 23) & 0xFFu;
    uint32_t mant = u & 0x7FFFFFu;

    if (ef == 0xFFu)                          /* inf / nan */
        return (uint16_t)(sign | 0x7C00u | (mant ? 0x200u : 0u));

    int32_t e = (int32_t)ef - 127 + 15;       /* f16 偏置指数 */
    if (e >= 0x1F) return (uint16_t)(sign | 0x7C00u);   /* 溢出 */

    if (e <= 0) {                             /* subnormal 或 0 */
        if (e < -10) return (uint16_t)sign;   /* < 2^-25 → RNE 到 0 */
        mant |= 0x800000u;                    /* 补隐含 1 */
        uint32_t sh = (uint32_t)(14 - e);
        uint32_t half = 1u << (sh - 1);
        uint32_t rem = mant & ((1u << sh) - 1u);
        mant >>= sh;
        if (rem > half || (rem == half && (mant & 1u))) mant++;
        return (uint16_t)(sign | mant);
    }
    uint32_t rem = mant & 0x1FFFu;            /* 23 → 10 位尾数，RNE */
    mant >>= 13;
    if (rem > 0x1000u || (rem == 0x1000u && (mant & 1u))) {
        if (++mant == 0x400u) {
            mant = 0;
            if (++e >= 0x1F) return (uint16_t)(sign | 0x7C00u);
        }
    }
    return (uint16_t)(sign | ((uint32_t)e << 10) | mant);
}

/* ---------- f32_to_q8_0 / q4_0 / q4i8 / q2_1 (src lines 2908-3088) ---------- */
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
        float scale;
        {
            volatile float c127 = 127.0f;   /* volatile 除数：禁常量倒数改写（-ffast-math） */
            scale = max_abs / c127;
        }
        if (scale < 1e-10f) scale = 1e-10f;

        /* Store f16 scale */
        uint16_t h = f32_to_f16_bits(scale);
        memcpy(q8_out + (size_t)b * 34, &h, 2);

        /* Store int8 quants */
        int8_t *qs = (int8_t *)(q8_out + (size_t)b * 34 + 2);
        float rscale = 1.0f / scale;
        for (int i = 0; i < block_size; i++) {
            float v = f32_in[b * block_size + i];
            /* volatile 中间量：钉死单次 IEEE 乘加序列，杜绝 -ffast-math 的
             * FMA 融合 / reciprocal / 重结合 —— 两端编译器逐字节一致 */
            volatile float qv = v * rscale;
            /* round-half-away-from-zero（等价 ggml quantize_row_q8_0 的
             * roundf）。**不能用 (int)(qv + 0.5f)**：C 的 (int) 向零截断，
             * 负 qv 会整体 +1（qv<=-0.5 时），造成 P(q=0) 翻倍、非零 q 均值
             * +0.55 LSB 的系统性正偏（llama Q8_0 仅 +0.05），量化噪声 RMS
             * 0.29→0.76 LSB。 */
            int q = (int)(qv < 0.0f ? qv - 0.5f : qv + 0.5f);
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
            volatile float x0 = xb[j] * id;
            volatile float x1 = xb[16 + j] * id;
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
            volatile float x0 = xb[j] * id;
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
            float d;
            {
                volatile float c3 = 3.0f;   /* volatile 除数：禁常量倒数改写（-ffast-math） */
                d = (maxv - minv) / c3;
            }
            if (!(d > 0.0f)) d = 1.0f;   /* guard: flat/zero block → q=0 */
            float m = minv;
            float id = 1.0f / d;
            uint16_t dh = f32_to_f16_bits(d);
            uint16_t mh = f32_to_f16_bits(m);
            memcpy(out + h * 4, &dh, 2);
            memcpy(out + h * 4 + 2, &mh, 2);
            uint8_t *qs = out + 8 + h * 4;
            for (int i = 0; i < 16; i++) {
                volatile float tq = (xs[i] - m) * id;
                int q = (int)(tq + 0.5f);
                if (q > 3) q = 3;
                if (q < 0) q = 0;
                qs[i / 4] |= (uint8_t)(q << (2 * (i % 4)));
            }
        }
    }
}


/* ---------- f32_to_g256q8 (src lines 3591-3630) ---------- */
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
        float scale;
        {
            volatile float c127 = 127.0f;   /* volatile 除数：禁常量倒数改写（-ffast-math） */
            scale = max_abs / c127;
        }
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
            float rscale = 1.0f / scale;
            for (int j = 0; j < 32; j++) {
                float v = xb[j];
                volatile float qv = v * rscale;
                /* 同 f32_to_q8_0：负数必须按 round-half-away-from-zero，
                 * 不能用 (int)(qv + 0.5f)（向零截断 → 负权重系统性 +1）。 */
                int q = (int)(qv < 0.0f ? qv - 0.5f : qv + 0.5f);
                if (q > 127) q = 127;
                if (q < -127) q = -127;
                qs[j] = (int8_t)q;
            }
        }
    }
}
