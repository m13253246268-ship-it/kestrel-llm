/* ================================================================
 * vllm_gguf.h - GGUF (llama.cpp Unified Format) loader
 *
 * 让引擎直接消费 llama.cpp 生态的 .gguf 模型文件（Qwen2/Qwen3/Llama/
 * Mistral 等架构）。GGUF = 头 + 元数据 KV + 张量目录 + 数据区，权重按
 * 块量化存储（Q4_0/Q8_0/k-quants 等）。
 *
 * 与 safetensors 路径的差异：
 *   - safetensors 存 F32/BF16/F16，加载时逐层量化到 Q8_0/Q4_0；
 *   - GGUF 已量化（Q4_0/Q8_0 的块格式与引擎 legacy 布局字节一致，可
 *     零转换直挂；k-quants 需先反量化再量化到引擎布局）。
 * 加载后权重布局与 safetensors 路径完全一致（STModelWeights），后续
 * 推理/repack/VQF 复用同一套内核。
 *
 * 架构适配：GGUF 元数据 general.architecture → STModelConfig；张量名
 * llama 风格（token_embd / blk.N.attn_q / blk.N.ffn_gate / output）。
 * llama/mistral 无 q_norm/mrope，引擎内核按 has_q_norm=0/has_mrope=0
 * 条件分支已支持。
 * ================================================================ */
#ifndef VLLM_GGUF_H
#define VLLM_GGUF_H

#include "vllm_safetensors.h"
#include "vllm_tokenizer_qwen.h"

/* GGUF 张量类型（与 llama.cpp ggml.h 枚举对齐，勿改值） */
#define GGUF_T_F32   0
#define GGUF_T_F16   1
#define GGUF_T_Q4_0  2
#define GGUF_T_Q4_1  3
#define GGUF_T_Q5_0  6
#define GGUF_T_Q5_1  7
#define GGUF_T_Q8_0  8
#define GGUF_T_Q8_1  9
#define GGUF_T_Q2_K  10
#define GGUF_T_Q3_K  11
#define GGUF_T_Q4_K  12
#define GGUF_T_Q5_K  13
#define GGUF_T_Q6_K  14
#define GGUF_T_Q8_K  15
#define GGUF_T_BF16  30

/* 探测文件是否为 GGUF（读 magic）。 */
int gguf_is_file(const char *path);

/* 加载 GGUF 模型权重到 w（填 w->cfg，量化布局按当前 wmode / VLLM_Q8_8X8
 * 与 safetensors 路径一致，位级一致由同一 f32_to_q8_0/q4_0 + repack 保证）。
 * 返回 0 成功；-1 失败。 */
int gguf_load_model(const char *path, STModelWeights *w);

/* 从 GGUF 内嵌 vocab 构建 tokenizer（tokenizer.ggml.* KV）。 */
int gguf_load_tokenizer(const char *path, QwenTokenizer *tok);

/* 诊断：打印 GGUF 文件结构（头/KV/张量目录）到 stderr。返回 0 成功。 */
int gguf_dump_info(const char *path);

#endif /* VLLM_GGUF_H */
