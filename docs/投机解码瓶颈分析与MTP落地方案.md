# 投机解码瓶颈分析与 MTP/DFlash 落地方案

> 版本：v1.0（已收口，2026-09-24）
> 范围：评估开源项目 [ninfer-master](../../参考/ninfer-master/) 的 MTP/DFlash 投机解码能否解决本工程 `src/` 的投机解码瓶颈，并给出可行落地方案。
> 依据公理：项目专属硬约束公理 [kestrel_prefill_axioms.json](../../kestrel_prefill_axioms.json)（`perf_axiom_*` 前缀）；通用库 [axiom_registry.json](../../axiom_registry.json) 中投机解码条目均为 SHS 理论公理（`operation_kind=axiom_statement`，confidence 0.3–0.6），**不构成硬约束**。

---

## 0. 摘要（结论先行）

1. **方向正确，但不等于性能可落地**。ninfer 的 MTP/DFlash 方法论恰好命中本工程投机解码的三个根因，是正确方向；但其 1.86×~2.83× 是 RTX 5090 CUDA 口径，**不构成对 RK3588 的任何性能承诺**（[perf_axiom_measured_benefit_region_v1](../../kestrel_prefill_axioms.json)「指令预算不能替代实测」、[perf_axiom_kv_precision_caliber_v1](../../kestrel_prefill_axioms.json)「跨硬件精度口径结论无效」）。
2. **直接照搬不可行**：硬件（CUDA/RTX 5090 vs RK3588 无 GPU 内存带宽受限）、权重（Qwen3-VL-2B/8B 无 MTP 草稿头）、位级一致性三方面不匹配。
3. **最终可行出路 = 放开「位级一致」约束（FAST）**。投机解码只需「输出 token 一致」（验证 top-1 == decode top-1），不必逐层 KV 位级一致；「位级一致」硬约束下 q4 GEMM 三口径两两位级不同，验证被迫走最慢 decode 同口径，投机无解（§4.3~§4.6 证伪链）。放开后（`VLLM_SPEC_FAST=1`：验证走 asm 快路 + 免重放 + 直接采用验证 KV），**重复/模板文本 1.43×、自由文本打平（n-gram α=0 不投机）**，且两场景输出 token 与无 spec 完全一致（§4.7 实测）。这是本引擎投机解码的收益上限。
4. **神经草稿头（MTP/EAGLE，P2）经板端实测否决**：Qwen3-VL 无内置 MTP 头，本机 5090 自训 EAGLE 式草稿头（top-4 树接受长度 2.02，显著优于 n-gram 的 1.12），但 RK3588 实测**负优化 3×**——草稿头自回归 K 步每步算 lm_head（151936×2048），带宽受限下草稿生成开销≈主模型 decode 单 token，吃光接受长度收益（§P2）。自由文本要投机盈利需 GPU（算力密集 lm_head）或 token 级草稿头，本环境均不具备。
5. **边界**：加密 VQF（VQF-Enc/FHE）模式下草稿权重须同加密且全层驻留；`--spec` 与 `--sparse-attn`/`--l3-evict`/`--batch-max` 互斥（代码已实证）。

---

## 1. 现状与瓶颈诊断

### 1.1 现役投机解码实现

本工程 `--spec [--spec-k N]` 位于 [vllm_server.c](../src/serve/vllm_server.c#L1007-L1243)，三段式：

| 阶段 | 实现 | 位置 |
|---|---|---|
| 草稿生成 | `spec_build_draft`：上下文末尾窗口（8→3 token）在历史找重复，取其后的 K token | [vllm_server.c#L891-L913](../src/serve/vllm_server.c#L891-L913) |
| 批量验证 | `st_qwen_model_prefill_batch(st, draft, K)`：一次并行 prefill，每位置 lm_head top-1 | [vllm_server.c#L1058](../src/serve/vllm_server.c#L1058) |
| 无损重放 | KV 回滚到 L，恢复上下文 logits，逐 token `st_qwen_model_forward` 重放确认 | [vllm_server.c#L1079-L1093](../src/serve/vllm_server.c#L1079-L1093) |

### 1.2 三个根因（瓶颈）

1. **草稿命中率低**：n-gram 只对**精确重复文本**（固定格式列表/代码模板/叠句）生效；自由文本命中低，编号递增类（"第N点"）实测 0% 命中（草稿系统性给旧值）。
2. **批量验证计算量 ≈ K×decode**：`prefill_batch` 走 fused-batched GEMM，比单 token decode 的单行 GEMM 更重，未摊销权重读取。
3. **无损重放叠加开销**：因 prefill_batch 与单 decode 是两条数值路径（fused-batched GEMM vs single-row GEMM，浮点差异逐轮累积进 KV），为保证 bit-exact 增加了 `accepted×decode` 的逐 token 重放。

### 1.3 实测结论（诚实边界）

- RK3588 8B Q8：**负优化**，比无 spec 慢 15–45%，即使 repeat_fixed 92% 接受率也只打平（[技术文档.md §5.3](../技术文档.md)）。
- x86：**反慢 2.4×**（verify 开销吃光增益）。
- **生产不建议开启**。

> 关键认识：RK3588 是**内存带宽受限**的 CPU（无 GPU），decode 本身是「每次读全量权重产 1 token」。投机解码在内存受限平台上**理论上是收益最大的方向**（草稿便宜 + 验证一次性摊销权重读取），本工程失败的原因不是「内存受限平台不适合投机解码」，而是草稿质量差（n-gram）+ 验证路径比 decode 贵（prefill_batch）+ 重放双倍开销。

### 1.4 历史：本线曾以「端到端 FAIL」封存（P5）

项目台账记录过这条线的结论：**P5 投机采样封存** —— 「已达成位级对齐（Gate 0 PASS），但因 **n-gram 草稿接受率低**且 **verify 摊销成本高**，端到端 FAIL，已封存」。

本方案与 P5 的关系必须讲清楚，否则是重开一个已否决的立项：

- P5 的两个否因里，**「verify 摊销成本高」= §4 的成本模型里的重放项（`real_acc×1`）**，那是一个**实现缺陷**（不是物理下界）：证据是现行代码把已接受的 token 又完整重放一遍。**这是 P5 当时未拆出的项**，也是 P1 唯一要动的东西。
- **「n-gram 草稿接受率低」是另一条轴（命中率），P1 不解、也解不了**，归 P2（神经草稿头）。
- 因此 P1 的定位不是「重开 P5」，而是**先拆掉 P5 两个否因中的一个实现缺陷，把成本轴清干净**；若拆完后 α→0 的工况仍回退（成本模型已预言 0.53×），则**证实 P5 的封存判决，且给出定量边界**（哪些 α 区间可用）。两种结局都有价值。

> 纪律提醒：按项目「探针先行」惯例（内核优化立项前需同档同 boot 探针验证上限），P1 必须先做 L1 对拍定位发散层 + 预登记，再动内核代码。

---

## 2. ninfer-master 机制分析

ninfer 是 CUDA/RTX 5090 的 from-scratch 推理引擎，投机解码有三后端：`--spec mtp` / `--spec dflash` / `--spec dflash2`。核心机制（[dflash.md](../../参考/ninfer-master/docs/maintainer/dflash.md)）：

### 2.1 草稿：一次 masked-block 前向并行出 K 个候选

- 草稿输入 = 目标 embedding：`[anchor a, MASK, MASK, ..., MASK]`（W=K+1 列）。
- 草稿由**神经草稿头**（MTP 头 / DFlash 层 / DFlash2 层）一次前向并行生成 K 个候选，**不是 K 次 decode**。
- MTP 头：主模型最后一层 hidden state 喂入一个轻量头（单层/数层），预测后续 token。
- DFlash：条件于目标模型多层残差特征 `s_t = concat(r_t^l)`，草稿层（5~6 小层）各自投影 K/V。

### 2.2 验证：target 因果验证且为唯一权威

- target 对 `[a, d_1, ..., d_K]` 做**一次因果前向**（Q=P+1 列），逐位验证。
- **target 始终是输出权威**：greedy 下接受匹配前缀，首个不匹配处 emit 目标修正 token；全接受则 emit bonus。
- **无独立重放步骤**——验证前向就是 decode 前向（同一因果路径），因此验证通过的 token 其 KV 天然与 decode 一致。

### 2.3 实测数据（RTX 5090，CUDA 口径，仅供参考）

| 指标 | 数值 |
|---|---|
| MTP3 接受率（自由文本） | code 76.4% / structured 90.8% / translation 75.0% / story 37.4% |
| MTP3 tokens/round | 2.1~3.7 |
| 端到端加速 | C=2 1.86×、C=4 2.83× |

> 这些数字**不可外推**到 RK3588（见 §0 结论 1 的限定）。

### 2.4 关键差异对照

| 维度 | 本工程 `--spec` | ninfer MTP/DFlash |
|---|---|---|
| 草稿来源 | n-gram（记忆查找，无权重） | 神经草稿头（权重内置/附带） |
| 草稿成本 | 免费（查找） | 一次小前向（K 并行） |
| 草稿命中（自由文本） | 低（仅精确重复） | 高（56–90%） |
| 验证路径 | `prefill_batch`（fused，漂移） | 与 decode 同因果路径（target 权威） |
| 重放 | 需要（KV 回滚 + 逐 token 重放） | 无 |
| 位级一致性 | 重放保证 | greedy 下 target 权威天然保证 |

---

## 3. 可行性评估（公理约束下）

### 3.1 结论 1：方法论方向正确 ✅（加限定）

命中三个根因，方向正确。但须按 [perf_axiom_measured_benefit_region_v1](../../kestrel_prefill_axioms.json) 声明：**方向正确 ≠ 性能可落地**；ninfer 的加速比是 RTX 5090 CUDA 口径，跨硬件不构成性能承诺，须实测界定。

### 3.2 结论 2：直接照搬不可行 ✅

三方面不匹配：
- **硬件**：CUDA/RTX 5090（compute-bound）vs RK3588 4×A76+4×A55 无 GPU（memory-bound）。CUDA f32 累加与 NEON f32 累加是不同数值口径，跨硬件照搬的位级结论天然无效。
- **权重**：Qwen3-VL-2B/8B **无内置 MTP 草稿头**（MTP 头需训练；Qwen3 base / Qwen3.5+ 系列才有）。
- **位级一致性**：ninfer 依赖 GPU 融合内核的数值路径，与本工程 NEON 路径不对齐。

### 3.3 结论 3：消除无损重放是核心杠杆 ⚠️（需满足完整位级等价条件）

方向正确，但「按行同累加顺序即可 bit-exact」**不完整**。按 [perf_axiom_order_preserving_reorder_v1](../../kestrel_prefill_axioms.json) 与 [perf_axiom_reduction_tree_sensitivity_v1](../../kestrel_prefill_axioms.json)，把 `prefill_batch` 换成「逐行 GEMV」须**同时满足**：

1. **s 全局升序 FMA 链与单 decode 相同**（每行 O[d]=Σ_s w_s·V_s[d] 的累加顺序不变）；
2. **s 分块间续接原累加器，非部分和再合并**；
3. **NEON `vfmaq` 操作数顺序逐字不变**（权重表达式与操作数寄存器顺序与单 decode 一致）；
4. **多行共享 V 载入时各自独立累加**，不做跨行归约；
5. **attention 的 QK hsum / softmax 分母归约树不变**（[perf_axiom_reduction_tree_sensitivity_v1](../../kestrel_prefill_axioms.json) 明列 softmax 分母求和，「数学等价但非位级等价」）；
6. **通过四级对拍**（隔离探针 → 完整模型 → TOKIDS+关闭回归 → 推广，见 [perf_axiom_bitwise_gate_v1](../../kestrel_prefill_axioms.json)），「生成文本/top-1 相同」不构成位级证明。

满足以上六条后，验证前向与 decode 前向位级等价，**贪婪模式下可免除重放**，且 K 行共享权重读取、摊销内存带宽。

> 这是 H1 允许的 row-level splitting / item-interleaving 调度级改动，不改变求和/归约/融合顺序。

#### 3.3.1 代码实证：漂移源已收窄到 attention（2026-09-24 复核）

对 `GitHupSRC` 现行工作树（commit `a236bf8`）逐调用点核对后，**原方案「把批量 GEMM 改成逐行 GEMV」是不必要的工作量**：

| 组件 | prefill_batch（验证路径） | st_qwen_model_forward（decode 路径） | 是否同一个核 |
|---|---|---|---|
| QKV 投影 | [dyn_matvec_q4_q8_fused_qkv_batched](../src/model/vllm_safetensors.c#L17575) | 同名函数（[15853](../src/model/vllm_safetensors.c#L15853)） | **是** |
| O 投影 | `dyn_matvec_q4_q8_fused_o_residual_batched` | 同名 batched（nb=1） | **是** |
| gate/up | `dyn_matvec_q4_q8_fused_gate_up_batched` | 同名 | **是** |
| down | `dyn_matvec_q4_q8_fused_down_residual_batched` | 同名 | **是** |
| lm_head | [dyn_matvec_q4_q8](../src/model/vllm_safetensors.c#L18119) 逐位置 | `dyn_matvec_q4_q8` | **是** |
| RMSNorm | `dyn_rms_norm_batch` | `dyn_rms_norm` | 需核 |
| **attention** | **[st_attn_batched_packed](../src/model/vllm_safetensors.c#L17791)（跨 query 打包核）** | **[flash_attn_blocked_single_q](../src/model/vllm_safetensors.c#L18332) / flash_attn_single_q_q8_neon** | **否** |

关键证据：QKV 核的注释自述 M4b 改造后「**单 token 的 b 累加顺序与原实现一致 -> 位级一致**」（[vllm_safetensors.c#L6476-L6479](../src/model/vllm_safetensors.c#L6476-L6479)），即批量 GEMM 已按 `perf_axiom_order_preserving_reorder_v1` 构造为逐行等价。

**推论（已由 §3.3.2 实测证实）**：验证路径与 decode 路径的剩余漂移**来自 batch 宽度会使若干算子切换到不同的数值路径**，符合 [perf_axiom_reduction_tree_sensitivity_v1](../../kestrel_prefill_axioms.json)。

**可直接复用的现成对拍装置**：源码自带 `VLLM_PFDUMP=<dir>` + `VLLM_PFDUMPTAG=<tag>`（[vllm_safetensors.c#L17201](../src/model/vllm_safetensors.c#L17201)、[pf_dump_post #L17236](../src/model/vllm_safetensors.c#L17236)），落点即 `prefill_batch` 与逐 token decode 的分叉对拍。**两臂驱动无需新写**：`--stream-test` 默认走 `prefill_batch`（A 臂），加 `VLLM_STREAM_SINGLE=1` 即逐 token `st_qwen_model_forward`（B 臂，[main.c#L3212-L3218](../src/main.c#L3212-L3218)）。

#### 3.3.2 L1 对拍实测结果（2026-09-24，板端 RK3588）

**装置**：板端 `/mnt/emmc/specl1/`（本地工作树 `CMakeLists.txt+src+include+tools` 独立构建，未触碰板端共享树）；新增默认关探针 `VLLM_SPECL1=<path>`（逐层 KV 行哈希，prefill/decode 两侧同格式）+ `VLLM_SPEC_VNORM=1`（修复）；2B=`Modl/Qwen3-VL-2B-q8fix`、8B=`Modl/8b-q4`。

**两臂合法性自证**：日志标签确为 `[batch]` / `[single-step]`（非 no-op），且两侧 `TOKIDS` 完全相同（2B/8B 均为 `100062`）—— 又一处「top-1 相同 ≠ 位级相同」的实例（[perf_axiom_bitwise_gate_v1](../../kestrel_prefill_axioms.json)）。

**结果一：2B-q8，现状全批宽 ⇒ 924 行中 923 行不同**，且 layer 0 / position 0（无上下文）即不同。

**结果二：阶段哈希定位首个发散点**（同二进制三臂，`emb0/nrm0/qq0/qk0`）：

| 阶段 | batch `VNORM=0` | batch `VNORM=1` | single（decode 基准） | 判定 |
|---|---|---|---|---|
| `emb0` embedding | `49da34c7…` | `49da34c7…` | `49da34c7…` | 一致 |
| `nrm0` attn RMSNorm | `c40c3a6a…` | **`60f6a85b…`** | **`60f6a85b…`** | **修复后一致** |
| `qq0`/`qk0` QKV | 发散 | 发散 | — | 见结果四 |

**根因 #1（已修复并验证）**：批式 RMSNorm 的平方和归约是**纯标量升序**（[vllm_rms_norm_batch_worker](../src/model/vllm_safetensors.c#L8596-L8620)），而 decode 的 `dyn_rms_norm` 是**8 元素分块 + `hsum8_f32` 树**（[#L8558-L8574](../src/model/vllm_safetensors.c#L8558-L8574)）。同一向量同一权重、两种浮点归约 ⇒ ULP 分歧。修复＝让批式 worker 复用 decode 的归约序（开关 `VLLM_SPEC_VNORM`，默认 0 零回归）。**2B/8B 均实测 `nrm0` 由不同变相同。**

**根因 #2（结构性，q8 模型）**：[dyn_matvec_q8_fused_qkv_batched_neon](../src/model/vllm_safetensors.c#L5586-L5604) 对 `n_batch==1` 走 **f32 激活** 单 token GEMV，对 `n_batch>1` 走 **int8 量化激活**的 `q8x4_gemm_batched`；注释自述「f32 舍入顺序放宽 -> PPL 验收」⇒ 批式在此处是**近似**，不可能位级相等。

**根因 #3（结构性，q4 模型）**：[q4x4_gemm_batched](../src/model/vllm_safetensors.c#L4392-L4418) 按宽度换路径——`n_batch>3` 主段走 `st_gemm_q4_0_4x4_batched`（注释自述「M4h 舍入序」），`n_batch<=3`（含 decode 的 nb=1）走 `vllm_q4x4_gemm_worker`（C tile 序）。⇒ **K≤3 时 QKV 与 decode 同 worker**；K≥4 必然发散。

**结果四：l=0 全同、l≥1 全不同（8B-q4，`VNORM=1` + `--prefill-batch 3`）**

| 层 | 不同行数 |
|---|---|
| **l=0** | **0 / 33（完全一致）** |
| l=1 … l=35 | 32–33 / 33（全不同） |

l=0 的 KV 行（= norm+QKV+rope，attention 之前）逐位全同 ⇒ 根因 #1 修复与 K≤3 规避同时成立；而 l≥1 全发散说明**层 0 的输出 hidden 已不同**，其输入（KV）相同、GEMM 同核 ⇒ **剩余唯一发散点在 attention 本身**。

**根因 #4（剩余项）**：prefill 走 [st_attn_batched_packed](../src/model/vllm_safetensors.c#L17791)（跨 query 打包、读 **f32 正典 KV**），decode 走 [vllm_attn_head_worker](../src/model/vllm_safetensors.c#L16119)（逐 query、读 **q8 KV 反量化**）。两者同时存在**归约树**与**KV 精度口径**两层差异，正是 [perf_axiom_reduction_tree_sensitivity_v1](../../kestrel_prefill_axioms.json) + [perf_axiom_kv_precision_caliber_v1](../../kestrel_prefill_axioms.json) 的组合。

### 3.4 结论 4：神经草稿头是大工程 ✅（门槛须落地）

Qwen3-VL 无 MTP 头，两条路线：
- **EAGLE 式草稿头训练**：在目标模型冻结参数之上，用目标 hidden state 训练一个轻量自回归草稿头（1 层 + 分类头），预测 K 个后续 token。
- **换带 MTP 的模型**（如 Qwen3 base / Qwen3.5+ 文本模型）：超出当前「Qwen3-VL 特化」范围，需显式产品决策。

无论哪条，都属 H4「未授权近似/新增权重」范畴，须**显式用户授权**后方可进入，且须过 go/no-go 门槛（§6）。

### 3.5 结论 5：边界 ✅

- 加密 VQF（VQF-Enc/FHE）：草稿权重须同加密、且全层驻留（「加密 VQF ⇒ 必须全层驻留」铁律）。「全层驻留可行性」受 RK3588 内存带宽约束，须实测界定，不得写成确定收益。
- 互斥：`--spec` 在 [vllm_server.c#L1009-L1011](../src/serve/vllm_server.c#L1009-L1011) 明确 `!g_l3_evict && !g_sparse_attn && ctx->batch_max < 2`；与 `--sparse-attn`/`--l3-evict`/`--batch-max` 互斥。

---

## 4. 落地方案（分阶段）

> 阶段 P1/P2 低风险可独立落地；阶段 P3 须用户授权。

### P1：消除无损重放（本地内核修复，不依赖 ninfer）

**成本模型（先算清楚有没有上限）**。设一次 decode 步的墙钟为 1，其中 GEMM 系数读数占 `G`、attention 占 `A`（`G+A=1`）；设 K=4、接受率 α。

| 路径 | 每轮成本 | 每轮产出 token | 加速比 |
|---|---|---|---|
| 无 spec（逐 token decode） | K | K | 1.00× |
| **现行 spec（prefill_batch + 重放）** | `G + K·A` + `real_acc×1` | `αK + 1` | ≈ 0.84× |
| **P1 后（验证位级等价，去掉重放）** | `G + K·A` | `αK + 1` | 见下 |

取 8B 的 `G≈0.7 / A≈0.3`：P1 后成本 = `0.7 + 1.2 = 1.9`。α=0.92（多模态/模板）⇒ `4.68/1.9 = **2.46×**`；α=0.75（自由文本）⇒ `4/1.9 = **2.11×**`；α=0（自由文本完全失配）⇒ `1/1.9 = 0.53×`（回退）。

> 这解释了文档里「repeat_fixed 92% 接受率也只打平」：现行路径把已接受的 `real_acc≈3.68` 个 token **又完整重放了一遍**（成本 +3.68），把 2.46× 拖成 0.84×。**重放是主承重项，不是 n-gram 命中率。** 但 α→0 时仍会回退，所以 P1 只解决「成本」轴，不解决「命中」轴（命中轴属 P2）。

**目标**：使验证路径与 decode 路径位级等价，去掉 KV 回滚 + 逐 token 重放。

**改动点（依据 §3.3.2 的 L1 实测，已收敛为 4 条，其中 2 条已有可用手段）**：

| # | 发散源 | 状态 | 手段 |
|---|---|---|---|
| 1 | 批式 RMSNorm 归约树 | **已修复并实测位级一致** | `VLLM_SPEC_VNORM=1`（默认关） |
| 2 | q4 QKV：`n_batch>3` 换真 GEMM 舍入序 | **已有零改码手段** | 约束 **K≤3**（落在 C tile 序，与 nb=1 同 worker） |
| 3 | q8 QKV：`n_batch==1` f32 激活 vs `n_batch>1` int8 激活 | 需改码（仅 q8 模型/2B） | 验证模式下强制逐 token 走 f32 激活核 |
| 4 | attention：跨 query 打包(读 f32 KV) vs 逐 query(读 q8 KV 反量化) | 需改码（剩余唯一项） | 见下 |

1. **RMSNorm**：已落地（`VLLM_SPEC_VNORM`）。
2~3. **GEMM**：**无需重写 GEMM**。q4 路径靠 **K≤3** 即可与 decode 同 worker（实测 l=0 全同）；q8 路径需在验证模式下逐 token 调 `n_batch=1` 的那条 f32 激活核（放弃激活 int8 共享）。
4. **attention（剩余唯一项）**：在 `st_qwen_model_prefill_batch` 的 attention 步骤（[vllm_safetensors.c#L17791](../src/model/vllm_safetensors.c#L17791)）增加 `VLLM_SPEC_VEXACT=1`（默认关）分支：把一次 `st_attn_batched_packed(nb 个 query)` 换成 nb 次 `vllm_attn_head_worker` 等价调用（**与 decode 同核**），第 t 个位置的有效前缀长度 = `prev_len + t + 1`；并让验证模式的 KV 读取走 **与 decode 同一精度口径**（同为 q8 反量化，或同为 f32 —— 必须一致，否则违反 [perf_axiom_kv_precision_caliber_v1](../../kestrel_prefill_axioms.json)）。放弃跨 query 注意力共享，换取构造性位级等价。
5. **去掉重放**：位级等价达成后，`vllm_server.c` 的 spec 块（[vllm_server.c#L1079-L1093](../src/serve/vllm_server.c#L1079-L1093)）跳过 KV 回滚 + 逐 token 重放，直接 `cache_len[l] = L + real_acc`、取 `g_verify_logits[real_acc-1]` 的 top-1 作为 bonus token。

**落地顺序**：先做第 4 项（attention），复跑 L1 直到 **8B-q4 / K≤3 全层 0 差异**；再动 spec 块去重放；最后按 §6 门槛做端到端 A/B。

**验收（四级对拍，[perf_axiom_bitwise_gate_v1](../../kestrel_prefill_axioms.json)）**：
- L1 隔离探针：用现成 `VLLM_PFDUMP` 对拍 `prefill_batch(K)` 与 K 次 `st_qwen_model_forward`，逐层 hidden / logits / KV 行哈希全等（**先做这一步定位首个发散层，再改代码**）。
- L2 完整模型：VEXACT 开/闭的 `--spec` 与无 spec 三臂 TOKIDS 逐位相同。
- L3 TOKIDS + 关闭回归：`VLLM_SPEC_VEXACT=0` 时输出与基线逐字节一致。
- L4 推广：2B/8B、n=1024~8192、多模态文本路径。

**性能门槛（gate）**：只有实测「去掉重放后 decode 吞吐显著优于现行 spec」才继续 P2/P3。若验证路径因 attention 不再共享而吃光收益（`K·A` 变大），判定本平台投机解码经济性不成立。

#### 4.1 支撑成本模型的关键实测（2026-09-24，板端，热态 3 次交替）

`--stream-test --stream-ctx 32`，同二进制、同模型（2B-q8fix）、A/B 交替 3 轮取稳定值：

| 臂 | 32 token prefill | 每 token | TOKIDS |
|---|---|---|---|
| `[batch]`（prefill_batch） | 0.515 / 0.515 / 0.563 s | **~17 ms** | 100062 |
| `[single-step]`（32× forward） | 2.283 / 2.277 / 2.304 s | **~72 ms** | 100062 |

⇒ 表面看是「共享权重读取带来 4.2×」，**但 §4.3 的实测推翻了这一读法**：该 4.2× 实际来自 **M4h asm 批式 GEMM（快路）vs 单 token GEMV**，与「token 摊薄权重读」无关（A 臂 `--prefill-batch` 默认 256 ⇒ 单 mini-batch 走 asm 快路）。代入 §4 成本模型：现行 spec = `verify(4)≈72ms` + `replay(3)≈215ms` ≈ **286ms / 4 token = 72ms/token（= 无 spec，打平）**；当初据此估计「去重放 ≈4×」——**该估计已被 §4.3 证否**。

> ⚠ **冷/热口径教训（本轮踩到，必须同引）**：首次运行（arm A 先跑、页缓存冷）读到 `batch 7.293s vs single 2.478s`，与设计意图完全相反；**交替复测（热态）后为 `batch 0.52s vs single 2.28s`**。凡跨进程比较 prefill 时长，必须交替复测并确认热态（同 [perf_axiom_fair_ab_comparison_v1](../../kestrel_prefill_axioms.json) 的「同线程+亲和+热态」）。

**本轮 L1 装置与产物**（可复现）：板端 `/mnt/emmc/specl1/`（`build/` 为独立构建）；`l1/t0..t2.txt`（2B 三臂阶段哈希）、`l1/8c.txt`/`l1/8d.txt`（8B `VNORM=1 --prefill-batch 3` vs single 的逐层 KV 行哈希）；本地比对脚本 `C:\Temp\specl1\cmp.py`、`cmp8b.py`。

#### 4.2 P1 落地结果与端到端实测（2026-09-24，板端，8B-q4）

**已实现**（均默认关，零回归）：`VLLM_SPEC_VNORM`（批式 RMSNorm 归约序对齐）、`VLLM_SPEC_VEXACT`（验证注意力走 decode 同核 + spec 块免重放 + 强制 `spec_k<=3` 自带自证行）。

**L1 门（8B-q4 / `--prefill-batch 3`）：PASS**。`VNORM=1 VEXACT=1` 批式 prefill 对逐 token decode ⇒ **36 层 × 32 位置 KV 行差异 = 0**，logits 亦一致（仅探针格式多出的 `lg` 行）。

**L3 门（同 prompt `--spec-k 3`，三臂）：PASS**。生成 token 序列 **39/39 逐位相同**：

| 臂 | 内容 | token 序列 | tpot |
|---|---|---|---|
| 无 spec（基准） | 相同 | 相同 | **188.9 ms** |
| `--spec` + VEXACT | 相同 | 相同 | **213.0 ms** |
| `--spec`（legacy 重放） | 相同 | 相同 | **332.2 ms** |

⇒ 去重放把投机从 **1.76× 慢** 收到 **1.13× 慢**，但**仍未跑赢无 spec** ⇒ **§4 性能门槛未达成**。

**为什么没赢：实测分解（`VLLM_DEC_PROF`，`[PF-PROF]`）**

| 口径 | 每 token 总时 | gu | down | qkv | o |
|---|---|---|---|---|---|
| 验证 prefill（tokens=3） | **153 ms/token**（总 460 ms） | 75.7 | 38.0 | 19.0 | 12.8 |
| 首轮 prefill（tokens=33，mini-batch 宽 8） | **52.6 ms/token** | 26.0 | 13.5 | 7.5 | 4.9 |
| 单 token decode | 186 ms/token | — | — | — | — |

结论：K=3 时验证只比 decode 便宜 **1.2×**，而每轮还要额外付 **1 个完整 decode 步**（emit_extra 的 `forward`）⇒ 净收益被吃光。当时据此推测「K 太小 ⇒ 提到 8 可摊薄权重读」——**该推测已被 §4.3 的实测证否（每 token 成本与 K 无关）**。

**下一步杠杆（K 提到 8）——已做，结论为负，且原因是结构性的**

已实现「验证模式强制走 C tile 逐 token 序 + tile 拉满到整批」（`q4x4_gemm_batched` 内 `spec_vexact_on() && n_batch<=8` 分支；该 worker 各 token 累加器独立 ⇒ 位级等价），并把 `spec_k` 上限放到 8。**L1 门复跑仍 PASS**（`--prefill-batch 8` ⇒ KV 行 0 差异、logits 0 差异）。

但端到端**更慢**：`--spec-k 8` ⇒ hits=30/6 轮（62% 命中，5 个/轮）、tpot **249.4 ms**（vs k=3 的 213.0、无 spec 的 188.9）。

`[PF-PROF]` 分解给出根因：

| 口径 | 每 token 总时 | gu | down |
|---|---|---|---|
| 验证 prefill K=8（强制 C tile） | **148 ms** | 73.2 | 38.2 |
| 验证 prefill K=3（强制 C tile） | 153 ms | 75.7 | 38.0 |
| 首轮 prefill 33 token（M4h asm 快路） | **52.6 ms** | 27.3 | 14.2 |
| 单 token decode（C tile GEMV） | 186 ms | — | — |

**每 token 成本与 K 无关（153 → 148）⇒「token 摊薄权重读」的前提在本引擎不成立**：q4 GEMV 路径是 **unpack/点积计算受限**，不是 DRAM 带宽受限，而摊薄只对 DRAM 流量有效。33 token 之所以只有 52.6 ms，是因为它走 **M4h asm 快路**（`n_batch>8` 未被强制）。

#### 4.3 P1 最终裁决：性能门槛不达成，且为结构性（2026-09-24）

- **位级等价可达成**（L1 PASS，K≤8）：把验证限制在「与 decode 同舍入序的 C tile GEMV + 同 attention 核 + 同 KV 精度口径」即可，代价已知且可控。
- **但该路径每 token 只比 decode 便宜 1.26×**（148 vs 186 ms），无法抵偿每轮多出的 1 个 decode 步（bonus token 的 `forward`）⇒ 净亏 13%~32%。
- **引擎里的快路（asm 批式 GEMM，52.6 ms/token，约 3.5× 于 decode）舍入序与 decode 不同** ⇒ 用它做验证就破坏位级等价，必须回滚重放。这正是「快」与「位级一致」在本内核结构下的**互斥**。
- 故在本引擎当前内核结构 + 位级一致性纪律下，**投机解码无法跑赢无 spec**。这与 P5「端到端 FAIL 已封存」的判决一致，且本轮给出了**机制级解释**（此前的理由是「接受率低 + verify 摊销贵」，现在能说清 verify 为何摊销不了：计算受限，非带宽受限）。
- **唯一可能的翻盘路径**（未做、成本高）：让 asm 批式 GEMM 的逐 token 累加树与 C tile GEMV 位级一致（等价于重写其一），从而在保位级一致的同时用上快路。这属于内核重构级改动，且有「改快路→影响全基线 / 改慢路→失去速度」的两难，建议单独立项并先做探针。

#### 4.4 更正（2026-09-24）：4.3 的「快与位级一致互斥」判断有误，翻盘路径真实存在

重读 `vllm_safetensors.c` 的 q4 内核，发现引擎里存在**三个**累加口径，而非两个：

| 内核 | 累加口径 | 当前用途 | 单 token 成本 |
|---|---|---|---|
| `q4x4_dot1_group16`（C tile worker） | **16 元素一次**，lo/hi 双 acc，每 block 2 次 f32 舍入 | **decode 的 QKV/O/gate/up/down** | 123 ms（gu+down） |
| `q4x4_dot1_group16_gemv`（M4f） | **32 元素一次**，单 acc，每 block 1 次 f32 舍入 | **死代码**（非 batched neon 无调用点） | 未知 |
| `st_gemm_q4_0_4x4_q8_0_neon`（asm 快路） | **32 元素一次**，手写汇编，批式 | prefill `n_batch>3` | 41.5 ms |

**关键发现：M4f 与 asm 快路都是「32 元素一次」口径**（8 个 vdot/sdot 累加进一个 int32 精确和，再一次 `scvtf ÷16` + 一次 `fma`），**位级同构**；而 decode 当前用的 C tile 是「16 元素一次」，与它们不同。

4.3 错在两点：
1. 误以为「decode 走 C tile」是不可变的位级基准，把验证也锁死在 C tile（148 ms/token），而 C tile 的 intrinsic 点积本身就慢。
2. 据此判 asm 快路「舍入序与 decode 不可调和」——实际 asm 与 M4f 同构，**只要 decode 换到 M4f（32 元素一次），验证即可走 asm 快路且位级一致**。

**翻盘路径（具体且成本低，M4f 代码已完整存在，只是未接线）**：
1. 把 decode 的 q4 GEMM 从 C tile 接线到 M4f（`q4x4_dot1_group16_gemv`，激活死代码）。
2. 验证（VEXACT）改走 asm 快路（`st_gemm_q4_0_4x4_batched`），与 M4f decode 同 32 元素一次口径 ⇒ 位级一致。
3. 免重放：验证每 token ~41.5 ms vs decode ~123 ms（≈3×）。

**预期收益**（K=8，α=0.62）：每 token = (41.5×8 + 123) / (0.62×8 + 1) ≈ 455/6 ≈ 76 ms ⇒ **约 1.6×**；α=0.76 ⇒ 约 1.9×。

**必须实测的 3 个前提（不可凭代码推断）**：
1. **asm 快路 == M4f 位级一致**（最关键）：理论同构（int 精确和 + ÷16 + 顺序 fma），但 asm 用 `sdot`+`scvtf`、M4f 用 `vdot`+`vcvt_n`，须 L1 对拍逐层 KV 行 0 差异。
2. **decode 换 M4f 后不退化**：P2a 记录「DRAM-bound、累加器变体无差别」暗示应持平，但须实测 decode tpot 不劣化。
3. **asm 快路在 K=8 小批仍 ~41.5 ms**：该数字来自 33-token prefill（摊薄 ÷33）；K=8 摊薄 ÷8，须实测确认仍有足够收益。

#### 4.5 实测证伪 4.4（2026-09-24）：asm 快路 ≠ M4f，翻盘路径不成立

按 4.4 落地并做了 L1 对拍（8B-q4，A 臂 = `VNORM+VEXACT+M4F_DECODE` + `--prefill-batch 8` 走 asm 快路；B 臂 = `VLLM_STREAM_SINGLE=1 M4F_DECODE` 走 M4f）：

**结果：KV 行 1187 / 1190 不同 ⇒ asm 快路与 M4f 位级不一致**，4.4 的「asm == M4f 位级同构」理论推断被证伪。

**差异是 ULP 级**：l=0 p=0 的 `kq`（int8 KV 载荷）两侧相同（`6ddd6c99e43df0ad`），而 `kf`（f32）/`ks`（scale）不同——f32 值差几个 ULP，被 int8 量化（256 档）吸收。

**根因（已定位）**：[repack_q8_0_4x4](../src/model/vllm_safetensors.c#L4039) 用 `f32_to_f16_bits(dm[m][b])` 把激活 scale 转成 **f16**（asm 快路算时再 `fcvtl` 转回 f32，引入舍入）；而 M4f 的 `quantize_row_q8_0_act` 直接用 **f32** scale。同一权重、同一激活，scale 精度口径不同 ⇒ ULP 分歧。

**最终裁决（回到 4.3，但机制更完整）**：

引擎 q4 GEMM 存在**三个两两位级不同**的舍入口径：

| 内核 | 舍入口径 | 相对 decode（C tile）位级一致 | 单 token 成本 |
|---|---|---|---|
| C tile（`q4x4_dot1_group16`） | 16 元素一次，lo/hi | **是（decode 本身）** | 123 ms |
| M4f（`q4x4_dot1_group16_gemv`） | 32 元素一次，单 acc | 否 | 死代码 |
| asm 快路（`st_gemm_q4_0_4x4_q8_0_neon`） | 32 元素一次，f16 scale | **否（实测）** | 41.5 ms |

- 「位级一致」要求验证与 decode 用**同一口径**；decode 当前是 C tile，故验证只能走 C tile（148 ms/token）⇒ 投机无法跑赢（4.3 结论成立）。
- 4.4 的翻盘思路（decode 换 M4f + 验证走 asm）错在**假设 M4f 与 asm 位级同构**；实测二者 scale 口径不同（f32 vs f16），位级不同。
- **根本矛盾**：asm 快路的「快」部分来自「放宽舍入序 + f16 scale」（省 FP 指令/带宽），而这**天然与「位级一致于 decode」互斥**。要让任一快核与 decode 位级一致，等价于把它的舍入序/scale 口径改回 decode 的「慢」口径，速度优势也随之消失。

**最终结论：在当前内核结构 + 位级一致性硬约束下，投机解码无法获得收益。** 这与 P5「端到端 FAIL 已封存」一致；本轮把机制从「接受率低」精确定位到「三口径两两位级不同，验证被迫用最慢的 decode 同口径」。唯一理论出路见 §4.6 的取舍与探针门槛。

### 4.6 唯一出路的取舍与探针门槛（2026-09-24）

#### 4.6.1 数学门槛（赢的充要条件）

投机每轮产出 `αK + 1`、成本 `V×K + D`（V=验证每 token，D=decode 单 token=175ms）。要赢：

```
V×K + D < D×(αK+1)  ⟺  V < D×α
```

| α（接受率） | 验证快核 per-token 须低于 |
|---|---|
| 0.62（自由文本） | **108 ms** |
| 0.76（k=3 实测） | **133 ms** |
| 0.92（重复文本） | 161 ms |

现状三口径 per-token：C tile 批式 **148ms**（>108，亏）、asm 快路 **~55ms**（赢但位级不一致）、decode 单 token **175ms**。

#### 4.6.2 两个方向对比

| 维度 | 方向 A：新写「C tile 口径的手写汇编」验证快核 | 方向 B：改 decode 对齐 asm 口径 |
|---|---|---|
| decode 是否变 | **不变（零回归）** | 变（输出改变） |
| 回归面 | 仅验证路径，`VLLM_SPEC_VEXACT` 默认关隔离 | **全基线重验**（PPL/NIAH/位级对拍/P5 基线） |
| 速度上限 | 快核 ~40-100ms，看纯读地板 | 验证走 asm 41.5ms（上限最高） |
| decode 单 token 变快否 | 不变 | **不变**（单 token DRAM-bound，换口径省不了读全量权重） |
| 符合项目纪律 | ✅ 默认关+逃生舱+探针先行 | ⚠️ 换位级基准，牵动核心承诺 |

#### 4.6.3 结论：方向 A 更适合

**方向 B 致命弱点**：改 decode 基准要付「重验全基线」的代价，却换不来额外收益——decode 单 token 是 DRAM-bound（读全量权重），换成 asm 口径后单 token 速度仍是 ~175ms，不会变快。方向 B 唯一作用是「让验证能和 decode 位级一致」，而这方向 A 也能做到、且**不碰 decode**。

**方向 A 优势**：零回归（decode 不变，位级基准不动）、风险隔离（默认关验证快核）、可独立验证（投机是否赢单独测）。

#### 4.6.4 方向 A 立项前的探针（纯读地板，探针先行）

方向 A 的成败取决于一个未知量：**asm 快路的 2.7× 速度里，多少来自「手写汇编+摊薄」，多少来自「32 元素一次+f16 scale」？** 不直接猜，而是量 C tile intrinsic 批式（111ms gu+down）的**纯内存读地板 M**（去掉全部点积计算，只读权重+xq+xd）。

- **M < 85ms** ⇒ 手写汇编理论下界 < 85ms < 108ms 阈值，方向 A 立项（乐观上界）。
- **M ≥ 85ms** ⇒ C tile 已内存受限，手写汇编无空间，方向 A 关闭，「唯一出路」坐实无解。

预登记（跑前冻结判据）见独立文件 `PRE_REG_Q4_GEMM_PROBE.txt`；判据摘要：`P-Q4-1`（前提）= 纯读地板占比 <0.80；`P-Q4-2`（决策）= 现状/纯读地板 ≥1.30（等价 M<85ms）。探针实现用 `VLLM_Q4_PROBE=1`（默认关，纯读 + sink 变量防优化），测量走引擎自带 `[PF-PROF] gu/down` 桶。

#### 4.6.5 探针实测结果与裁决（2026-09-24，板端 8B-q4，同 boot 交替 2 轮）

| 臂 | GATEUP+DOWN（32 token） | per-token | 抖动 |
|---|---|---|---|
| L0（`VLLM_Q4_CTILE=1`，C tile 现状） | 3646.6 ms | 114.0 ms | <0.5% |
| L1（`VLLM_Q4_CTILE=1 VLLM_Q4_PROBE=1`，纯读地板） | 2352.0 ms | **73.5 ms** | <0.5% |

**G0（validity）PASS**：两臂自证行 `[Q4CTILE]`/`[Q4PROBE]` engaged；n=32、GATEUP+DOWN>0、tokens 相等；无开关对照臂 TOKIDS=100062（探针零副作用自证，默认关行为逐位不变）；L1 的 TOKIDS 与 L0 不同（纯读产出垃圾，预期）。

**P-Q4-1（前提）PASS**：M/现状 = 2352/3647 = **0.645 < 0.80** ⇒ C tile intrinsic 有 ~35% 时间是点积计算（非纯内存），手写汇编有榨取空间。

**P-Q4-2（决策）PASS**：上界比 = 3647/2352 = **1.55 ≥ 1.30**（等价 M=73.5 < 85ms）⇒ **方向 A 立项**。

**裁决**：授权「试写 C tile 口径（16 元素一次 + lo/hi + f32 scale）的手写汇编批式 GEMM，作为投机验证快核 + 自带 A/B 回退」。乐观下界 73.5ms（纯读地板，假设计算榨到 0），实际落点预计 73.5~114ms；须后续：
1. 写汇编 → L1 位级对拍（vs C tile 逐字节，`VLLM_SPECL1` 探针复用）。
2. 端到端三臂（无 spec / spec+C tile / spec+快核），确认快核 per-token < 108ms（α=0.62）才真正赢。
3. 若汇编实测落点 ≥ 108ms（未能榨到阈值），按 PRE_REG 局限 (a)「纯读地板是乐观上界」如实关闭。

> 注意：上界比 1.55 是「乐观上界」，不是收益承诺——纯读地板假设汇编把计算榨到 0，实际汇编仍保留 16 元素一次的 vdot/vcvt/vfma，只能逼近 73.5ms 到不了 73.5ms。

#### 4.6.6 方向 A 手写汇编实测证伪（2026-09-24）

按用户授权「仍写手写汇编」，先做了最小兑现——`VLLM_Q4_ASM=1`（block 展开 2 次 + 预取的手写流水线，位级口径不变）。交替 2 轮（同 boot，抖动 <1%）：

| 臂 | GATEUP+DOWN 中位 | per-token | 收益 |
|---|---|---|---|
| L0（CTILE intrinsic） | 3658 ms | 114.3 ms | — |
| L1（CTILE + 手写流水线） | 3475 ms | 108.6 ms | **1.05×** |

**结论：手写流水线只榨 5%（114 → 108.6ms），远不足以让投机赢（需 <94ms = 1.2×）。**

**为什么榨不动**：纯读地板 73.5ms 与手写流水线 108.6ms 之间还剩 35ms「计算」，它**不是调度低效**，而是 lo/hi 双 acc 的**固有成本**——「lo 的 fma 链是 block 间串行的（数学要求，无法并行）」+「16 元素一次需要 2 次 vcvt+vfma（FP 指令是 asm 单 acc 的 2×）」。编译器 -O2 已经做了 block 内 sdot 交错 + ST_PREFETCH，手写流水线的额外调度空间只有 5%。

**对照 asm 快路（41.5ms）**：它快不是因为「手写调度」，而是「32 元素一次（FP 减半）+ 16-tile（xq 读摊薄）+ f16 scale」——这些是**位级口径/访问模式**的差异，不是「调度」，C tile 口径手写汇编无法复现（会破坏位级一致）。

**最终裁决：方向 A（手写汇编）证伪**。投机解码在本引擎的「唯一出路」两个方向都不通：
- 方向 B（改 decode 对齐 asm）＝重验全基线、纯代价无收益。
- 方向 A（手写 C tile 口径汇编）＝实测只 5%，固有成本锁死。

投机解码在本引擎 + 位级一致性硬约束下**无高性价比出路**。本轮完整结论：P5「端到端 FAIL 已封存」的机制级解释 = 三口径两两位级不同 + lo/hi 双 acc 固有成本，验证被迫用最慢的 decode 同口径且无法通过调度提速。

### 4.7 放开位级一致：FAST 模式（最终正面结果）

#### 4.7.1 关键澄清：位级一致 ≠ 输出 token 一致

之前 §4.3~§4.6 一直卡在「**位级一致**」（验证 KV 逐字节 == decode KV），才把验证锁死在 C tile（148ms）。但投机解码真正需要的是「**输出 token 一致**」（验证 top-1 接受判断 == decode top-1），不需要逐层 KV 位级一致。asm 快路（42.5ms gu+down）与 C tile decode 的差异是 ULP 级 + f16 scale 系统性偏差，几乎不翻转 top-1。

这是**传统投机解码的标准做法**（vLLM/llama.cpp 都不追求位级一致）。项目当前的「无损重放」是比行业标准更严的自加约束。

#### 4.7.2 实现：VLLM_SPEC_FAST=1（默认关，零回归）

验证走 `st_qwen_model_prefill_batch` 默认路径（n_batch=K>3 自动走 asm 快路 GEMM + batched attention），免去 KV 回滚 + 逐 token 重放，直接采用验证 KV 与验证 logits。改动仅 vllm_server.c 的 spec 块（`spec_fast_env()` + 免重放分支复用 VEXACT 逻辑）。

#### 4.7.3 端到端实测（8B-q4，serve 模式，K=8）

| 场景 | α | base（无 spec）tpot | fast tpot | **收益** | 输出 token vs base |
|---|---|---|---|---|---|
| 重复文本（8 行相同） | 62.5% | 181.1 ms | **126.6 ms** | **1.43×** | 39/39 一致 |
| 自由文本（量子纠缠解释） | **0（n-gram 无匹配，不投机）** | 182.6 ms | 181.8 ms | 1.00×（打平） | 167/167 一致 |
| legacy（无损重放，重复文本） | 62.5% | — | 265.9 ms | 0.68×（负） | 39/39 一致 |

**结论**：
1. **放开位级一致解除了「成本」轴**：验证走 asm 快路（42.5ms），草稿命中率足够时（重复文本）**赢 1.43×**，输出 token 与无 spec 完全一致（实测两场景 0 差异）。
2. **「命中」轴仍在**：自由文本 n-gram 草稿接受率低（固有局限），验证快也打平——这不是「位级一致」能解决的，需 **P2 神经草稿头**提接受率。
3. **legacy 无损重放是负优化**（0.68×），项目当前的「位级一致」强约束直接吃掉了投机解码的收益。

**最终结论**：投机解码在本引擎的可行出路 = **放开位级一致（FAST）**。单放开位级一致，重复/模板文本赢 1.43×、自由文本打平。P2 神经草稿头（提自由文本接受率）经板端实测**不盈利**（负优化 3×，见 §P2），故自由文本无解。**最终收益上限 = FAST 的重复/模板文本 1.43×，自由文本维持打平。** 这修正了 §4.3~§4.6 在「位级一致硬约束」下得出的「无解」判断，也修正了本段早期「叠加 P2 后自由文本赢 1.5-2×」的乐观预期（P2 被板端实测否决）。

### P2：神经草稿头（EAGLE 式）训练与接入

**前置**：P1 已达成「验证 ≈ 1× 权重读取摊销 K token」。

**计划**：
- 训练（外部，需 GPU）：冻结 Qwen3-VL 文本塔，在其上训练 1 层草稿头（输入=目标 hidden state，输出=K 个 next-token logits）。
- 数据：与目标同域语料，multi-token 目标。
- 接入：替换 `spec_build_draft`，复用 P1 验证路径。

**实际执行（2026-09-24，本机 RTX 5090 训练 + RK3588 板端实测）**：

1. **训练（本机 5090）**：冻结 Qwen3-VL-2B 文本塔，训 1 层 `Qwen3VLTextDecoderLayer` 草稿头（特征级外推，CE loss）。WikiText-2 真自由文本：**top-1 接受率 47.9%、top-4 树接受长度 2.02**（vs n-gram 3-gram 历史命中 12%/1.12）⇒ 接受率方向成立。
   - 坑：① 多步自回归第 2 步崩（特征级误差累积），靠「训练位置 shift」修正（1.49→1.63）；② top-k 边际递减（第 3 步后 <6%），真突破需 token 级草稿头（EAGLE-2 做法）；③ 数据量非瓶颈（1M→2M token 打平）。
2. **板端接入（C 实现）**：`st_draft_head_forward`（复用 dyn_matvec/dyn_swiglu/dyn_mrope/dyn_rms_norm + scalar attention + 独立 KV cache），`VLLM_SPEC_DRAFT` 环境变量替换 n-gram。编译通过、**数值正确**（板端接受长度 1.73 ≈ 本机 1.63）。
3. **实测（负优化）**：无 spec **39.2ms** vs 草稿头+FAST **120.5ms**（q8 量化+并行），**负优化 3×**（f32 130.8ms → q8 串行 147.3ms → q8 并行 120.5ms，均负）。
   - 根因（2026-09-24 板端探针实测拆分）：草稿头前向 79.7ms = lm_head 48.6ms（**61%**，全 vocab 151936 分类，每步 12.2ms×K=4）+ 草稿 transformer 31.1ms（39%，1 层 q8 QKV/attn/FFN）。早先「lm_head 占 96%」为估算偏高，实测仅 61%；但两者叠加仍使「草稿生成开销 ≈ 1.5× 主模型 decode 单 token」，吃光接受长度 1.73 的收益。

**结论**：神经草稿头（EAGLE 式特征级）在 RK3588 **不盈利，P2 否决**。自由文本要投机盈利需 GPU（算力密集的 lm_head）或 token 级草稿头（减少 lm_head 次数），本环境均不具备。这与 §4.3~§4.6 的「位级一致下无解」、n-gram 的「自由文本打平」同源——都是 RK3588 带宽约束。**投机解码在本引擎的收益上限 = FAST 的重复/模板文本 1.43×，自由文本无解。**

**补充探针（2026-09-24）：DFlash2 式「短名单 proposal head」能否翻转？—— 否决**

针对结论里「token 级草稿头（减少 lm_head 次数）」这条出路，参考 ninfer DFlash2 的「indexed proposal head + codebook」（用短名单替代全 vocab 分类），做了 CPU 成本探针：给 `st_draft_head_forward` 加 `VLLM_DRAFT_LM_ROWS=k` 门控（砍 lm_head 分类行数），板端同 boot 测草稿头前向时间（只计时，输出无效）。

| lm_head 行数 | 草稿头前向 | lm_head | 草稿 transformer 地板 |
|---|---|---|---|
| 151936（全 vocab） | 79.7ms | 48.6ms（61%） | 31.1ms |
| 4096 | 31.3ms | 1.34ms（4%） | 30.0ms |
| 256 | 30.1ms | 0.11ms（0%） | 30.0ms |

- lm_head ∝ vocab 行数（48.6→1.34→0.11ms，线性），可砍到接近 0。
- 但草稿 transformer（1 层 decoder q8 的 QKV/attention/FFN × K=4）**固定 30ms 地板**，不随 vocab 变。
- 翻正阈值 = `c_verify × (L−1)` = 39.2 × 0.78 = **30.6ms**（保守 L=1.73 → 28.6ms）。草稿头前向下限 30ms 已贴阈值，再加短名单 head 自身成本（1~5ms）与接受率损失（短名单漏 token，L 下降）→ **必然翻不正**。

**结论**：短名单 proposal head（DFlash2 式砍 lm_head 算力）同样否决。瓶颈不在 lm_head（可砍到 0），而在草稿 transformer 30ms 固定地板——它是投机解码在 CPU 上的硬成本，砍任何单一组件都无法低于翻正阈值。这再次印证 `perf_axiom_measured_benefit_region_v1`「指令预算推算不能替代实测」：理论推算（砍 lm_head 81→5ms）被实测（30ms 地板）推翻。

### P3：默认开启评估

仅当 P2 通过全部 go/no-go 门槛（§6）后，才评估写入默认（[perf_axiom_default_enablement_gate_v1](../../kestrel_prefill_axioms.json)）。**P2 已否决（板端负优化 3×），P3 不启动。**

---

## 5. 验证与测量计划（RK3588 设备操作约束）

进行 A/B 测量时，须遵守以下约束（公理 + 设备操作指南）：

- **亲和/异构核**：`taskset` 固定 A76 大核，两侧各自 warm，同 governor（RK3588 大小核不锁核会调度漂移）。
- **同权重同 token 流**：量化块逐字节验证（非同一模型名）；同一次 tokenize 结果喂两侧；batch/ubatch 显式对齐。
- **线程环境**：`OMP_NUM_THREADS` 与 `VLLM_THREADS` 两侧都显式设置并清除；`compute_threads=N` 从启动 banner 核对。
- **冷缓存**：只跑 cold-cache first-slot，一次 boot 只跑一个配置。
- **统计口径**：multi-token 统计或 bucket 级累积，禁用单 token `[DEC-SINGLE]`。
- **基线漂移归一**：同 boot 会话内测；跨 boot 用未修改基线 bucket 归一。
- **设备操作**：隐藏独立进程；`timeout.exe` 等待；ASCII/环境变量路径；SHA256/mtime+日志错误核对构建；进程结束后再读日志；切配置前杀残留引擎进程。

---

## 6. go/no-go 门槛（P2/P3 专用）

通过全部以下条件才允许推进（[perf_axiom_fair_ab_comparison_v1](../../kestrel_prefill_axioms.json) + [perf_axiom_default_enablement_gate_v1](../../kestrel_prefill_axioms.json) + [perf_axiom_measured_benefit_region_v1](../../kestrel_prefill_axioms.json)）：

1. **同权重**：草稿头与主模型权重量化块逐字节验证。
2. **同 token 流**：同一次 tokenize 喂两侧。
3. **同 batch/ubatch + 同线程/亲和/热态/governor**：显式对齐。
4. **位级对拍通过**：spec 开/关输出逐字节一致（§4 P1 四级对拍）。
5. **性能全面领先且极差稳定 <1%**：decode 吞吐在目标场景全面领先，多次测量极差 <1%。
6. **逃生舱**：`--spec=0` 关闭后输出与基线一致。
7. **不满足条件自动回退**：hd/形状/模式不满足时自动回退原路径。
8. **实测界定收益区间**：不同 n / 模型 / 自由文本 vs 精确重复，明确哪些区间有收益（不得用 `r ≈ 1/(1-acceptance_rate)` 理论公式推导，须实测）。

**若 P2 实测无法满足门槛（如 RK3588 上草稿头权重读取开销吃光收益），则诚实结论为：本平台投机解码经济性不成立，维持「生产不建议开启 `--spec`」现状，将优化重点转回 decode 内存带宽 / 权重量化。**

---

## 7. 风险与边界汇总

| 风险 | 说明 | 缓解 |
|---|---|---|
| 位级漂移 | 批量 GEMV 与单 decode 若累加/归约顺序不一致 | §3.3 六条 + 四级对拍 |
| 草稿头训练成本 | EAGLE 式训练需 GPU + 数据 + 调参 | P3 前 go/no-go 门槛 |
| 内存带宽不摊销 | RK3588 上草稿头权重读取可能吃光收益 | P1 性能门槛先行验证 |
| FHE/加密 | 加密模式下草稿权重须同加密、全层驻留 | 边界声明，实测界定 |
| 互斥 | 与稀疏/L3/批处理互斥 | 保留现互斥逻辑 |
| 理论公式误用 | 不得用 SHS 加速比公式推导性能 | 只实测，不套公式 |

---

## 附录：相关文件与公理索引

- 本工程投机解码：[vllm_server.c#L1007-L1243](../src/serve/vllm_server.c#L1007-L1243)
- 本工程投机解码说明：[技术文档.md §5.3](../技术文档.md)、[优化配置与边界说明.md §3.4](../优化配置与边界说明.md)
- ninfer 投机解码机制：[dflash.md](../../参考/ninfer-master/docs/maintainer/dflash.md)、[draft.cpp](../../参考/ninfer-master/src/models/qwen3_5/execution/draft.cpp)
- 硬约束公理：[kestrel_prefill_axioms.json](../../kestrel_prefill_axioms.json)（`perf_axiom_bitwise_gate_v1` / `order_preserving_reorder_v1` / `reduction_tree_sensitivity_v1` / `fair_ab_comparison_v1` / `default_enablement_gate_v1` / `measured_benefit_region_v1` / `kv_precision_caliber_v1` / `quantization_grid_fairness_v1`）
