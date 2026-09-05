/* ================================================================
 * vqf.h - VQF (vLLM Quantized Format): engine-native pre-quantized,
 * pre-repacked model weight format (M0-M3 压缩态推理).
 *
 * 动机（公理 blas_precision_efficiency_tradeoff / block_matrix_assoc_natural）：
 *   当前加载 = safetensors mmap 读 F32/BF16 → 每层 f32_to_q8_0/f32_to_q4_0 量化
 *   → 层后 repack（Q8 8x8 tiled / Q4 4x4）→ 板端 2B 约 23s。
 *   VQF 把"量化 + 布局 repack"固化到文件：权重按内核直接消费的最终布局存储，
 *   加载 = mmap + 元数据校验 + 指针直挂（零转换零计算），并保留位级一致性
 *   （fixedpoint_quantize_saturate red line：转换复用引擎同一量化代码路径）。
 *
 * 与 GGUF 的差异：GGUF 存 legacy 线性布局，mmap 后仍需运行时 repack/解包；
 * VQF 直接存内核原生布局（8x8 tiled / 4x4 nibble / 8x8l），mmap 即用。
 *
 * 磁盘格式定义（VQFArch/VQFSig/VQFHeader/VQFTensor/flag 宏）在独立的
 * vqf_format.h（平台无关，离线工具与 KAT 测试共用，防止三方漂移）。
 * ================================================================ */
#ifndef VQF_H
#define VQF_H

#include "vqf_format.h"
#include "vllm_safetensors.h"

/* 转换：把已加载(量化+repack 完成)的权重 dump 成 VQF 文件。
 * flags 由调用方（main.c 转换入口）按引擎实际布局传入。 */
int vqf_write(const char *path, const STModelWeights *w,
              const STModelConfig *cfg, uint32_t flags);

/* 加载：mmap 直挂 VQF 权重到 w，填充 w->cfg（自包含）。
 * 返回 0 成功；-1 失败（格式/布局不匹配/IO），w 保持原样。
 * 成功时 w->vqf_map 记录 mmap 区域，st_weights_free 统一释放。 */
int vqf_load(STModelWeights *w, const char *path);

/* 探测文件是否为 VQF（读 magic）。 */
int vqf_is_file(const char *path);

/* 后台线程预热 mmap 权重页（madvise WILLNEED + 逐页 touch）：把 eMMC→RAM
 * 的页 fault 成本从首次推理挪到就绪后的空闲期。由 VLLM_VQF_PREWARM=1 触发；
 * detach 线程，不阻塞 serve。Windows 下为空操作。 */
void vqf_prewarm(const STModelWeights *w);

#endif /* VQF_H */
