/* ================================================================
 * qk_globals.c - 板端 NEON 参考（qk_ref_engine.c）链接所需的全局。
 *
 * 引擎源中这些是 vllm_safetensors.c 的 static/全局；抽取到独立
 * 文件后由本文件提供（初值 = 引擎默认：repack 开启，q4 4x4 预分配
 * scratch 与引擎 st_weights_alloc 一致）。
 * 仅测试用，不进 converter 正式产物。
 * ================================================================ */
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>

int      g_st_q8_repack = 1;
int      g_st_q4_repack = 1;
uint8_t *g_q4r_scratch = NULL;
size_t   g_q4r_scratch_cap = 0;
uint8_t *g_q8r_scratch = NULL;
size_t   g_q8r_scratch_cap = 0;

/* 预分配 4MB（覆盖 12288 宽 down_proj 的 4 行组 27648B 上限的百倍） */
__attribute__((constructor)) static void qk_scratch_init(void) {
    if (!g_q4r_scratch) {
        g_q4r_scratch = (uint8_t *)malloc(4 << 20);
        g_q4r_scratch_cap = g_q4r_scratch ? (4 << 20) : 0;
    }
}
