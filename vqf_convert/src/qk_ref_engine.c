/* Auto-extracted from src/model/vllm_safetensors.c (sha256 da768fbff3b10063a3815f8786b536892e0b7ec68ef6efe8a10180483bd27cc3)
 * Re-run tools/extract_quant.py after engine edits; verify sha256.
 * ================================================================ */

#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

/* globals supplied by test/qk_globals.c (engine defaults: repack on) */
extern int      g_st_q8_repack;
extern int      g_st_q4_repack;
extern uint8_t *g_q4r_scratch;
extern size_t   g_q4r_scratch_cap;
extern uint8_t *g_q8r_scratch;
extern size_t   g_q8r_scratch_cap;
int repack_q4_0_4x4_inplace(uint8_t *buf, int rows, int cols);
int repack_q8_0_8x8_inplace(uint8_t *buf, int rows, int cols);
int repack_q8_0_4x4_inplace(uint8_t *buf, int rows, int cols);
int repack_q4_0_8x8l(uint8_t *dst, const uint8_t *src, int rows, int cols);

/* ---------- repack engine original (src lines 3360-3562) ---------- */
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
