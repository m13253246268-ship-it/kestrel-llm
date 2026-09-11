/* ================================================================
 * qk_test.c - 量化内核一致性 harness（两端共用同一源码）
 *
 * 确定性 LCG 生成 f32 权重（含全零块与极端块，触发 scale guard），
 * 走量化 + 各 repack，逐段输出 FNV-1a 64 哈希。
 * 对拍：x86（qk_repack.c 标量）与 aarch64（qk_ref_engine.c NEON）
 * 两个编译产物打印的哈希必须逐行一致（位级一致红线）。
 * ================================================================ */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ---- 量化内核（quant_kernels.c）---- */
void f32_to_q8_0(uint8_t *out, const float *in, int n);
void f32_to_q4_0(uint8_t *out, const float *in, int n);
void f32_to_q4i8(uint8_t *out, const float *in, int n);
void f32_to_g256q8(uint8_t *out, const float *in, int n);

/* ---- repack（x86 用标量 qk_repack.c / 板端用 NEON qk_ref_engine.c）---- */
int repack_q4_0_4x4_inplace(uint8_t *buf, int rows, int cols);
int repack_q8_0_8x8_inplace(uint8_t *buf, int rows, int cols);
int repack_q8_0_4x4_inplace(uint8_t *buf, int rows, int cols);
int repack_q8_0_tiled_inplace(uint8_t *buf, int rows, int cols);
int repack_q4_0_8x8l(uint8_t *dst, const uint8_t *src, int rows, int cols);

#define Q8B(n) (((size_t)(n) + 31) / 32 * 34)
#define Q4B(n) (((size_t)(n) + 31) / 32 * 18)

static uint64_t fnv(const void *p, size_t n) {
    const uint8_t *b = (const uint8_t *)p;
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 0x100000001b3ull; }
    return h;
}

/* 确定性 LCG（xorshift32），序列固定，跨平台一致 */
static uint32_t g_seed = 0x9E3779B9u;
static uint32_t lcg(void) {
    uint32_t x = g_seed;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    g_seed = x;
    return x;
}

int main(void) {
    const int rows = 96;     /* %8==0 && %4==0 */
    const int cols = 4096;   /* %32==0 */
    size_t n = (size_t)rows * cols;

    float *w = (float *)malloc(n * sizeof(float));
    uint8_t *q8 = (uint8_t *)malloc(Q8B(n));
    uint8_t *q4 = (uint8_t *)malloc(Q4B(n));
    uint8_t *q4i = (uint8_t *)malloc(Q8B(n));
    uint8_t *x8 = (uint8_t *)malloc(Q4B(n));
    if (!w || !q8 || !q4 || !q4i || !x8) { fprintf(stderr, "OOM\n"); return 2; }

    /* 确定性权重：[-1,1] 混合；行 32-39 全零（scale guard）；行 80 置大值 */
    for (size_t i = 0; i < n; i++) {
        uint32_t u = lcg();
        float v = (float)(u >> 8) / 16777215.0f * 2.0f - 1.0f;
        w[i] = v;
    }
    for (int c = 0; c < cols; c++) w[(size_t)32 * cols + c] = 0.0f;
    for (int c = 0; c < cols; c++) w[(size_t)39 * cols + c] = 0.0f;
    for (int c = 0; c < cols; c++) w[(size_t)80 * cols + c] = (float)(c % 7) - 1000.0f;

    int fail = 0;
    /* 1) q8_0 量化 + tiled(默认8x8) repack */
    f32_to_q8_0(q8, w, (int)n);
    printf("q8_0         %016llx\n", (unsigned long long)fnv(q8, Q8B(n)));
    f32_to_q8_0(q8, w, (int)n);   /* repack in-place on copy */
    if (repack_q8_0_tiled_inplace(q8, rows, cols) != 0) {
        printf("q8 tiled     repack rc=-1 (geometry?)\n");
        fail = 1;
    } else {
        printf("q8 tiled     %016llx\n", (unsigned long long)fnv(q8, Q8B(n)));
    }
    /* 2) q8_0 4x4 repack */
    f32_to_q8_0(q8, w, (int)n);
    if (repack_q8_0_4x4_inplace(q8, rows, cols) != 0) {
        printf("q8 4x4       repack rc=-1\n");
        fail = 1;
    } else {
        printf("q8 4x4       %016llx\n", (unsigned long long)fnv(q8, Q8B(n)));
    }
    /* 3) q4_0 量化 + 4x4 repack */
    f32_to_q4_0(q4, w, (int)n);
    printf("q4_0         %016llx\n", (unsigned long long)fnv(q4, Q4B(n)));
    f32_to_q4_0(q4, w, (int)n);
    if (repack_q4_0_4x4_inplace(q4, rows, cols) != 0) {
        printf("q4 4x4       repack rc=-1\n");
        fail = 1;
    } else {
        printf("q4 4x4       %016llx\n", (unsigned long long)fnv(q4, Q4B(n)));
    }
    /* 4) q4_0 -> x8 8x8l 副本 */
    f32_to_q4_0(q4, w, (int)n);
    if (repack_q4_0_8x8l(x8, q4, rows, cols) != 0) {
        printf("x8 8x8l      repack rc=-1\n");
        fail = 1;
    } else {
        printf("x8 8x8l      %016llx\n", (unsigned long long)fnv(x8, Q4B(n)));
    }
    /* 5) q4i8 预解包 */
    f32_to_q4i8(q4i, w, (int)n);
    printf("q4i8         %016llx\n", (unsigned long long)fnv(q4i, Q8B(n)));
    /* 6) g256 */
    f32_to_g256q8(q8, w, (int)n);
    printf("g256         %016llx\n", (unsigned long long)fnv(q8, Q8B(n)));

    free(w); free(q8); free(q4); free(q4i); free(x8);
    printf(fail ? "QK FAIL\n" : "QK OK\n");
    return fail ? 1 : 0;
}
