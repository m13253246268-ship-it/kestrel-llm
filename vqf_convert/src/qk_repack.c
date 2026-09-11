/* ================================================================
 * qk_repack.c - 跨平台标量 repack（converter x86 路径）
 *
 * 布局与 src/model/vllm_safetensors.c 的 NEON 分支逐字节一致
 * （纯字节搬移，无浮点；q4_0_4x4/q8_0_8x8/q8_0_4x4 逻辑对应引擎
 * 行 3369-3407 / 3427-3477 / 3529-3562；q4_0_8x8l 对应行 3494-3520）。
 * 与引擎差异：临时组缓冲用 malloc（引擎用预分配 scratch 全局），
 * 几何/返回语义（不可 repack -> -1，buffer 不动）保持一致。
 * 对拍红线：板端 NEON 引擎同函数 vs 本文件标量，同输入哈希一致。
 * ================================================================ */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ---------------- Q4_0 legacy -> 4x4 (nibble) ----------------
 * per 4-row group: d[4] f16 scales (row order), then 16 interleaved
 * 4-byte chunks, each ^0x88888888. Output = 72 B per 32-col block. */
int repack_q4_0_4x4_inplace(uint8_t *buf, int rows, int cols) {
    if (rows % 4 != 0 || cols % 32 != 0) return -1;
    int nb = cols / 32;
    size_t row_stride = (size_t)nb * 18;
    size_t group_bytes = 4 * row_stride;
    uint8_t *tmp = (uint8_t *)malloc(group_bytes);
    if (!tmp) return -1;
    for (int r0 = 0; r0 < rows; r0 += 4) {
        uint8_t *dst = buf + (size_t)(r0 >> 2) * group_bytes;
        memcpy(tmp, dst, group_bytes);
        for (int b = 0; b < nb; b++) {
            uint8_t *out = dst + (size_t)b * 72;
            for (int i = 0; i < 4; i++)
                memcpy(out + (size_t)i * 2,
                       tmp + (size_t)i * row_stride + (size_t)b * 18, 2);
            const uint32_t xor_mask = 0x88888888u;
            for (int i = 0; i < 16; i++) {
                uint32_t e;
                memcpy(&e, tmp + (size_t)(i & 3) * row_stride + (size_t)b * 18 + 2
                             + (size_t)(i >> 2) * 4, 4);
                e ^= xor_mask;
                memcpy(out + 8 + (size_t)i * 4, &e, 4);
            }
        }
    }
    free(tmp);
    return 0;
}

/* ---------------- Q8_0 legacy -> 8x8 tiled ----------------
 * per 8-row group: d[8] f16 scales, then qs[k*32+m*4+i] = row m block b
 * elem (k*4+i). Output = 272 B per 32-col block. */
int repack_q8_0_8x8_inplace(uint8_t *buf, int rows, int cols) {
    if (rows % 8 != 0 || cols % 32 != 0) return -1;
    int nb = cols / 32;
    size_t group_bytes = (size_t)nb * 272;
    uint8_t *tmp = (uint8_t *)malloc(group_bytes);
    if (!tmp) return -1;
    for (int r0 = 0; r0 < rows; r0 += 8) {
        uint8_t *dst = buf + (size_t)(r0 >> 3) * group_bytes;
        memcpy(tmp, dst, group_bytes);
        for (int b = 0; b < nb; b++) {
            uint8_t *out = dst + (size_t)b * 272;
            for (int m = 0; m < 8; m++)
                memcpy(out + (size_t)m * 2,
                       tmp + (size_t)m * nb * 34 + (size_t)b * 34, 2);
            for (int k = 0; k < 8; k++)
                for (int m = 0; m < 8; m++)
                    for (int i = 0; i < 4; i++)
                        out[16 + k * 32 + m * 4 + i] =
                            tmp[(size_t)m * nb * 34 + (size_t)b * 34 + 2 + k * 4 + i];
        }
    }
    free(tmp);
    return 0;
}

/* ---------------- Q8_0 legacy -> 4x4 (block_q8_0x4) ----------------
 * per 4-row group: d[4] f16 scales, then qs[k*16+m*4+i]. 136 B/block. */
int repack_q8_0_4x4_inplace(uint8_t *buf, int rows, int cols) {
    if (rows % 4 != 0 || cols % 32 != 0) return -1;
    int nb = cols / 32;
    size_t group_bytes = (size_t)nb * 136;
    uint8_t *tmp = (uint8_t *)malloc(group_bytes);
    if (!tmp) return -1;
    for (int r0 = 0; r0 < rows; r0 += 4) {
        uint8_t *dst = buf + (size_t)(r0 >> 2) * group_bytes;
        memcpy(tmp, dst, group_bytes);
        for (int b = 0; b < nb; b++) {
            uint8_t *out = dst + (size_t)b * 136;
            for (int m = 0; m < 4; m++)
                memcpy(out + (size_t)m * 2,
                       tmp + (size_t)m * nb * 34 + (size_t)b * 34, 2);
            for (int k = 0; k < 8; k++)
                for (int m = 0; m < 4; m++)
                    for (int i = 0; i < 4; i++)
                        out[8 + k * 16 + m * 4 + i] =
                            tmp[(size_t)m * nb * 34 + (size_t)b * 34 + 2 + k * 4 + i];
        }
    }
    free(tmp);
    return 0;
}

/* dispatch: 默认 8x8（VLLM_Q8_8X8=0 回退 4x4），语义同引擎 */
static int g_q8_8x8 = -1;
static int q8_8x8_enabled(void) {
    if (g_q8_8x8 < 0) {
        const char *e = getenv("VLLM_Q8_8X8");
        g_q8_8x8 = (e && e[0] == '0') ? 0 : 1;
    }
    return g_q8_8x8;
}
int repack_q8_0_tiled_inplace(uint8_t *buf, int rows, int cols) {
    if (q8_8x8_enabled()) return repack_q8_0_8x8_inplace(buf, rows, cols);
    return repack_q8_0_4x4_inplace(buf, rows, cols);
}

/* ---------------- Q4_0 legacy -> 8x8l (x8 decode copy, pure scalar) ----------------
 * 引擎行 3494-3520 同源拷贝（无 NEON 差异）。 */
int repack_q4_0_8x8l(uint8_t *__restrict dst, const uint8_t *__restrict src,
                     int rows, int cols) {
    if (rows % 8 != 0 || cols % 32 != 0) return -1;
    int nb = cols / 32;
    size_t src_row = (size_t)nb * 18;
    for (int r0 = 0; r0 < rows; r0 += 8) {
        for (int b = 0; b < nb; b++) {
            uint8_t *out = dst + (size_t)(r0 >> 3) * nb * 144 + (size_t)b * 144;
            const uint8_t *sb = src + (size_t)r0 * src_row + (size_t)b * 18;
            for (int r = 0; r < 8; r++)
                memcpy(out + (size_t)r * 2, sb + (size_t)r * src_row, 2);
            for (int i = 0; i < 16; i++) {
                uint64_t e;
                memcpy(&e, sb + (size_t)(i & 7) * src_row + 2 + (size_t)(i >> 3) * 8, 8);
                e ^= 0x8888888888888888ULL;
                memcpy(out + 16 + (size_t)i * 8, &e, 8);
            }
        }
    }
    return 0;
}
