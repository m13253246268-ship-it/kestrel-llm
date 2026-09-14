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

/* 加载：mmap 直挂 VQF 权重到 w，填充 w->cfg（自包含）。
 * 返回 0 成功；-1 失败（格式/布局不匹配/IO），w 保持原样。
 * 成功时 w->vqf_map 记录 mmap 区域，st_weights_free 统一释放。 */
int vqf_load(STModelWeights *w, const char *path);

/* 探测文件是否为 VQF（读 magic）。 */
int vqf_is_file(const char *path);

/* 已加载 VQF 的「文件内固化布局」（/admin 只读展示用）。返回 1 = 有已加载的
 * 模型并回填 flags（VQF_FLAG_*）/version（VQF_VERSION[_MOE]）；0 = 未加载。
 * 量化模式在转换期已固化进文件，加载侧不可更改，故这里只做只读回读。 */
int vqf_layout_state(uint32_t *flags, uint32_t *version);

/* 后台线程预热 mmap 权重页（madvise WILLNEED + 逐页 touch）：把 eMMC→RAM
 * 的页 fault 成本从首次推理挪到就绪后的空闲期。由 VLLM_VQF_PREWARM=1 触发；
 * detach 线程，不阻塞 serve。Windows 下为空操作。 */
void vqf_prewarm(const STModelWeights *w);

/* ================================================================
 * 分层驻留（layer streaming，AirLLM 型，CPU 内存预算实验）
 *
 * 前提：VQF 数据区张量按 rows=n_layers*R 整层堆叠连续存放、repack 不跨层
 * （tile 层内闭合，见 vllm_safetensors.c repack_q8_0_tiled_inplace 逐层原位），
 * 因此"层→文件字节段"可由目录几何现算，无需任何格式改动。
 * 本模块把 mmap 的按页懒加载升级为"受控逐层驻留"：
 *   VLLM_VQF_STREAM=N     启用（N = 允许常驻的落后层数；1 = 只剩当前层）
 *   VLLM_VQF_STREAM_LOG=1 在层边界打印 VmRSS 采样（/proc/self/status）
 *   VLLM_VQF_STREAM_COLD=1 驱逐时连带 posix_fadvise(DONTNEED) 丢页缓存：
 *     再访问必须从 eMMC 重读（度量"数据来源换盘"最坏代价；默认仅控制
 *     本进程 RSS，页缓存由内核按压力回收）
 *
 * 机制：层循环每进入一层调用 vqf_stream_layer_advance(w, l)：
 *   - 逐出 l-N 层全部权重段（madvise MADV_DONTNEED：丢弃干净文件页，再访问
 *     时从 eMMC 重新 page-fault，内容由文件重建 → 推理位级一致不变）；
 *   - 预读 l+1 层（MADV_WILLNEED：IO 与当前层计算重叠）。
 *
 * 诚实边界（红线，禁止跨过）：
 *   1) 仅明文 VQF（flags 无 VQF_FLAG_ENC）。加密模型整文件单遍 SM4-CTR
 *      在 MAP_PRIVATE COW 页原位解密 → 页为脏，DONTNEED 会丢解密数据；
 *      且认证（HMAC tag）与 CTR 流均为整文件粒度、无逐层 tag。自动拒绝。
 *   2) 逐层驱逐不减少每 token 的权重读取总量，只改变数据来源
 *      （RAM 命中 → eMMC 冷读）；模型 ≤ 预算时逐出反而自伤，须按需启用。
 *   3) embed/final_norm/lm_head/vision 为常驻段（与 AirLLM 同语义：仅整层
 *      权重参与流式；token_embed 的 F16 表在 2B 约 0.46GB 始终驻留）。
 * ================================================================ */
void vqf_stream_setup(STModelWeights *w);   /* vqf_load 成功后自动调用 */
void vqf_stream_layer_advance(STModelWeights *w, int layer);
long vqf_stream_rss_kb(void);               /* VmRSS(kB)，非 Linux 返回 0 */

/* VQF 分层驻留运行状态（/admin 状态展示用）。返回 1 = 已启用并回填字段；
 * 0 = 未启用（env 未开 / 非明文 VQF / 几何不匹配 / 未加载）。 */
int vqf_stream_state(int *keep, int *nl, int *nseg,
                     unsigned long long *per_layer, int *cold);

/* 运行时驻留控制（serve 内存驻留策略 vllm_res_policy 调用，依赖 setup 建表）：
 *   active     = 1 表示 seg 表已建（可执行"权重从高向低逐层释放"动作）；
 *   evict_above(keep) = 释放 layer > keep 的权重段物理页（保留 0..keep 层 +
 *                       embed/lm/final_norm/vision 常驻段；只改 PTE、可逆）；
 *   willneed_to(keep) = 预读 0..keep 层（WILLNEED，升档恢复的提示性加速）。 */
int  vqf_stream_active(void);
void vqf_stream_evict_above(int keep);
void vqf_stream_willneed_to(int keep);

/* 选路后专家预取（gate-first，VLLM_EW_PREFETCH=1，仅明文 VQF q4_4x4）：
 * MoE FFN top-k 选路完成后调用（st_moe_ffn_sparse_q4 内），默认对选中专家
 * 的 gate/up 段（0% 过取，100% 被 gu 消费）发 WILLNEED，把缺页重驻与并行
 * gu 计算重叠；VLLM_EW_PREFETCH_DOWN=1 额外连整层 down 带一起提示（≈3.4×
 * 过取，Windows 实测负收益，仅目标端 async 读入足够便宜时开）。
 * 原语：Linux madvise(WILLNEED) / Windows PrefetchVirtualMemory（缺特权静默
 * 降级）。只改页表/页缓存、浮点序零改动 → A≡C 位级一致。q8/非 VQF/几何
 * 不匹配 no-op。 */
void vqf_ffn_prefetch(const STModelWeights *w, int layer,
                      const int *sel, int tk);

/* EP 专家分片导出（方案 A，资料/分布式专家提取_实施方案.md）：
 * 把全量 VQF 导出为"本 rank 独占"的稀疏分片 —— 布局与源文件逐字节同构
 * （header/directory/data_offset/file_len/flags/arch 照抄），只物化本 rank 的
 * q4_gate/q4_up/q4_down 字节区间，其余留空洞。产物是**合法 VQF**，引擎
 * `vqf_load` 与 EP 路径无需任何改动即可加载。
 *   rank ∈ [0, nranks)，且须整除 arch.n_experts。
 * 返回 0 = 成功；非 0 = 失败（原因打印到 stderr）。仅明文 VQF。 */
int vqf_export_ep_shard(const char *src_path, int rank, int nranks,
                        const char *out_path);

#endif /* VQF_H */
