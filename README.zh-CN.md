# Kestrel（红隼）— ARM(aarch64) 边缘 LLM 推理引擎（RK3588 开发基准）

[English](README.md) | **简体中文**

**纯 C11、零第三方运行时依赖的 ARM（aarch64）CPU LLM 推理引擎。**

> **版本：v1.0（测试版 / test release）**。相对上一个公开版本（v0）的主要变化见文末
> 「v1.0 变更摘要」。文中性能/体积数字按版本分区：**未标注版本的数字为 v1.0 板端实测
> （2026-09-12）**；标注 `（v0 测点）` 的取自文末「附录 A」，是在 v0 上测得、v1 未复测，
> **不得用于 v1 的任何结论**。换板复现请以 `./build_rk3588.sh --run-tests` 的当前自检与
> 产物为准。

**v0 / v1 能力对照（本 README 的数字按此分区）**

| 能力 | v0（上一公开版本） | **v1.0（本版本）** |
|---|---|---|
| 权重加载 | 支持 GGUF / safetensors 加载，且**引擎内置转换**（`vllm_gguf.c`、`convert.html`） | **纯 VQF 运行时：只认自研 VQF v2 单文件（mmap 直挂）**；内置 GGUF/safetensors 加载与引擎内转换已删除，转换职责移交随版工具 `vqf_convert/` |
| 权重驻留 | 全量常驻（全层） | 新增**逐层推理**（`VLLM_VQF_STREAM=1` 分层驻留）：权重常驻 RSS 与模型体积/层数解耦；但**只对明文 VQF 生效**——VQF-Enc / 内嵌 SM2 签名的权重会被显式拒绝（须全层驻留） |
| KV 与长上下文 | KV v1 | **KV v2 惰性分配** + L3 分层驻留 + P3（L3 驱逐 × 前缀复用共存） |
| 安全与合规 | VQF 存储态加密（SM4-CTR + HMAC-SM3）、SM2 供应链签名 | 追加**可验证推理 attestation（schema 3，含请求原文绑定）**，浏览器内 / 离线零依赖验签 |
| x86-64 | 无 | 随版提供 x86-64 移植层（`tools/build/build_x64.ps1` / `tools/build/check_x64.ps1`），**仅功能自检与位级一致性对照，不作性能基准** |
| 基准测点 | 2026-09-05：冷启动 / 8K 长上下文 / KV 恢复（含 llama.cpp 同机对照） | **2026-09-12**：逐层 vs 全层 × {2B / 8B / 30B-A3B} × {组合⑧基线, 组合①, 组合①+⑤}，14 个 serve 配置点 |

> 为什么不能混用：v1 移除了 GGUF/safetensors 加载路径并引入逐层推理与 KV v2，
> **冷启动、常驻内存、长上下文与多轮 prefill 的口径都已变化**。v0 的绝对值只作历史参考，
> 需要 v1 数字的场合一律以本正文（2026-09-12 测点）为准。

> 平台口径：引擎本质是**面向 ARM 架构 CPU 的推理**（ARMv8.2-A + NEON dotprod/fp16，
> 不依赖 GPU/NPU 与特定开发板）；**RK3588** 是当前开发、优化与基准测试平台，
> 并非唯一可运行设备——同类 aarch64 Linux 设备可尝试编译运行（跨设备验证状态见「启动流程」节平台约束）。

---

## 核心优势

一块 ARM 开发板（示例：RK3588 / Orange Pi 5 Plus）+ 一个约 **0.8 MB** 的单文件可执行程序
= 原生 LLM 推理服务：**自研 VQF v2 权重单文件 mmap 直挂、按需建页**，原生跑 Qwen3-VL
2B/8B 纯文本与图片/视频多模态，OpenAI 兼容 HTTP API 即刻可用。

| 核心优势 | 一句话指标 |
|---|---|
| **逐层推理**（v1 新增） | 权重常驻与模型体积解耦：**8B 权重常驻 4.19 GB → 0.48 GB（8.7×）**，serve 峰值 **0.82 GB**；远程复测热态 **TTFT 1.73 s / tpot 330 ms**（5 次同请求中位数，见「性能」§4） |
| **长文本与多轮** | 组合①（L3 驱逐 × 前缀复用）追问轮 prefill **−80% ~ −96%**；30B 再叠 MoE 组合⑤后 t2/t3 仅 **4.7 s / 3.2 s** |
| **纯自研格式与内核** | 只认自研 VQF v2 单文件；单产物 **≈0.8 MB**，零第三方运行时；手写 NEON 量化 GEMM/GEMV、自研线程池、自研国密 |
| **可验证推理** | SM2 供应链签名护权重 + 逐请求 attestation 凭证（schema 3），浏览器内 / 离线零依赖验签 |

### 逐层推理内存优势（举例）

`VLLM_VQF_STREAM=1` 让每层权重只在参与计算时建立文件页，算完立即 `MADV_DONTNEED`，
仅 keep 层（缺省 1 层）常驻——**权重常驻 RSS 不再随模型体积线性增长**。

以三个真实模型为例（`--stream-test` 冷页缓存，纯权重口径）：

| 模型（单文件 VQF） | 全层：权重常驻 / 峰值 | **逐层：权重常驻 / 峰值** | 常驻内存倍率 |
|---|---|---|---|
| 2B Qwen3-VL（4.16 GB） | 1.81 GB / 2.76 GB | **0.39 GB / 0.63 GB** | **4.6×** |
| 8B Qwen3-VL（6.60 GB） | 4.18 GB / 4.22 GB | **0.47 GB / 0.59 GB** | **8.9×** |
| 30B-A3B MoE（17.66 GB） | 11.73 GB / 12.70 GB | **0.51 GB / 0.71 GB** | **23.2×** |

**典型例子**：RK3588 板只有 16 GB RAM（MemTotal 15.6 GiB），Qwen3-30B-A3B-q4 的权重单文件
就有 17.66 GB —— 全层档虽能勉强跑起来，但 serve 峰值已到 14.2 GB，几乎不留 KV 与进程余量；
**逐层档把同一模型的 serve 峰值压到 1.11 GB，余量充足、可稳定服务**。2B 更直观：常驻权重只有
0.39 GB，比全层省 4.6×，代价是热态 tpot 从 113.0 ms 升到 156.7 ms（+38.7%）。

**推荐档是 8B**（以 2026-09-15 RK3588 远程复测为当前公开口径，见「性能」§4）：权重常驻
4.19 GB → **0.48 GB（8.7×）**，serve 峰值 **0.82 GB**；热态 **TTFT 中位 1.734 s、
tpot 中位 330.2 ms**（范围 `1.711~1.796 s` / `328.9~330.6 ms`，70 token 端到端约 24.5 s），
且 **5 次采样 tpot 波动仅约 ±0.26%**。它的权重 6.15 GiB **小于物理内存**，页缓存装得下，
所以热态稳定是有保障的——这一点 30B 做不到（详见「性能」§4 与 §5）。

**语义不变**：驱逐只丢弃干净文件页，内容由文件重建——`--stream-test` 的贪心 TOKIDS 序列
在两种驻留档下**逐位一致**（详见「性能」一节的 md5 对照）。

> 边界：逐层推理**只支持明文 VQF**（VQF-Enc / 内嵌 SM2 签名的权重会被显式拒绝，须全层驻留），
> 且与专家窗口 `VLLM_EW*` 不并存。完整口径、代价与矩阵见下文「性能」。
> **30B 是能力上限档，不是推荐档**：16.4 GiB 权重装不进 15.6 GiB 页缓存，且默认 thinking 下
> 一轮问答需 3~4 分钟（见「性能」§5）。

---

## 性能

### 性能数据分区说明

本章数字分两类，各小节标题会标明属于哪一类：

1. **受控 A/B（可复现到 1% 以内）** -- `--stream-test` 口径。同一块板、同一模型文件、**同一次会话内交错**运行**旧版**（Gitee 上一版源码 `ffd0b92` 于本机现编）与**新版**（本版源码），每个测点 2 轮，逐点记录温度与各核实时频率。**第 1、4 节的线程对照属于此类。**
2. **新测点（单次，非受控）** -- serve 类口径（第 2 / 3 / 5 节）。旧版对应点未复测，原因是旧版 30B serve 单点需 15-40 分钟。这些表**只反映本版表现，不构成新旧对比**。
3. **跨引擎对照（权重公平 A/B，第 7 节）** -- 同一块板、**同一份 4-bit 量化权重**、两侧各开各的优化，与 **llama.cpp** 逐档对照（2K / 4K 各 3 轮、8K 2 轮、16K 1 轮）。这是本章唯一的「对第三方引擎」口径，与前两类（本版 vs 上一版、本版与自身配置）**不可混引**。

**测量环境**：RK3588（Orange Pi 5 Plus，15.9 GB RAM，`governor=performance`）。旧版二进制 sha256 `76d5bffb8c6fb280...`，新版 `078349be598e13ef...`。温度全程 **30.5-47.2 度**，A76 / A55 频率恒定 **2,304,000 / 1,800,000 kHz** -- **全程无降频**，故新旧差异不来自温度或频率。

**正确性判据（`--stream-test` 贪心 TOKIDS 逐位一致）**：

- 30B-A3B：第 1 节共 **16 个测点**（旧/新 x 全层/逐层 x 2 轮）全部输出 32 token 且序列相同；首 8 个 id = `151667, 198, 99692, 3837, 20002, 104029, 11622, 104811`。
- 8B：共 **24 个测点**全部输出 29 token 且序列相同；首 4 个 id = `100062, 99371, 58814, 113272`。

> ### 为什么废弃 2026-09-12 的历史数字
>
> 那批数字出自**另一支二进制**（sha256 `e1484740...a8f8e8`）与**另一次会话**。用今天从同一源码提交现编的旧版二进制**无法复现**它们：
>
> | 历史记录 | 今天旧版实测 | 偏差 |
> |---|---|---|
> | 8B 全层 t8 热 decode 431 ms/tok | 468 ms/tok | +8.6% |
> | 8B 逐层 t4 热 decode 370 ms/tok | 319 ms/tok | -14% |
> | 8B 第 3 节的 P3 prefill（t2）1,885 ms | 2,730 ms（L3 清零后） | +45% |
> | 30B 全层 t8 冷 decode 2,442 ms/tok | 2,020 ms/tok | -17% |
>
> 差异来源不是代码，而是「不同二进制 + 不同会话状态（尤其 L3 / 页缓存）」。因此历史数字**已整段废弃**，本章不再引用；这也是本章改为「同会话交错 A/B」的原因。

### 结论速览

| 模型 | 相对上一版的进展 | 内存 |
|---|---|---|
| **30B-A3B（MoE，q4，17.66 GB）** | **decode 快 1.74-3.72 倍；prefill 快 1.21 倍（冷，I/O 重叠改善）到 5.44 倍（热）** | 持平（+0.1-1.0%） |
| **8B（稠密，q4，6.60 GB）** | **无变化** -- 24 个测点逐项一致，最大差异 1.0% | 持平 |
| **8B 对 llama.cpp（长上下文，第 7 节）** | **prefill：2K 慢 1.32 倍 -> 4K 慢 1.15 倍 -> 8K 反超 1.11 倍 -> 16K 反超 1.60 倍；decode：2K 全层反超 1.07 倍 -> 16K 反超 3.63 倍；multi-turn 复用轮：本版全面反超 1.33~3.4 倍（幅度随上下文单调递增）** | 峰值 16K 6.71~10.15 GB vs llama.cpp 10.77 GB（2K 档 1.46~4.90 GB vs 8.71 GB） |

---

### 1) 逐层推理 vs 全层（`--stream-test`，**受控 A/B**）

`VLLM_VQF_STREAM=1` 进入**分层驻留**模式：VQF 单文件 mmap 直挂，每层权重只在参与计算时建立文件页、算完立即 `MADV_DONTNEED`，仅 keep 层（缺省 1 层）常驻。配合 KV v2 惰性分配，**常驻内存与模型体积、层数解耦**：17.66 GB 的 30B-A3B 可在 16 GB 内存的板上服务。

**30B-A3B（MoE，q4，17.66 GB；`/mnt/VQF/qwen3-30B-A3B-q4/model.vqf`，位于 SanDisk microSD）**

| 驻留 | 线程 / 缓存 | 旧版 prefill | 新版 prefill | 比 | 旧版 decode | 新版 decode | 比 |
|---|---|---|---|---|---|---|---|
| 全层 | t8 / 冷页缓存 | 259.86 s | **213.70 s** | 1.22 倍 | 2,020 ms/tok | **986 ms/tok** | **2.05 倍** |
| 全层 | t4 / 热页缓存 | 36.28 s | **6.67 s** | **5.44 倍** | 1,205 ms/tok | **324 ms/tok** | **3.72 倍** |
| 逐层 | t8 / 冷页缓存 | 260.54 s | **215.15 s** | 1.21 倍 | 2,412 ms/tok | **1,387 ms/tok** | **1.74 倍** |
| 逐层 | t4 / 热页缓存 | 36.75 s | **6.93 s** | **5.30 倍** | 1,583 ms/tok | **564 ms/tok** | **2.81 倍** |

峰值 VmHWM：全层 12,705,354 -> **12,722,536 kB（+0.14%）**；逐层 708,162 -> **715,054 kB（+0.97%）**。两版**内存持平**；逐层相对全层省 **17.8 倍**（12,722,536 / 715,054）。

> **冷态 prefill 的 1.21-1.22 倍是间接收益**：该档由「从 SD 卡读入 17.66 GB」主导，等效读速旧版 68.0 MB/s、新版 82.6 MB/s。读的是同一个文件，唯一变量是 CPU，即旧版「解包太慢拖累了读盘流水」，新版把计算提速后与 I/O 重叠更好。**不要把它理解为「内核加速了读盘」。**

> **30B 的 decode 倍数依赖页缓存状态，请按区间理解**：模型 17.66 GB 装不进 15.9 GB RAM。新版 decode 实测区间为 **324-1,204 ms/tok**：全层缓存命中时约 324 ms/tok（即上表），逐层约 **570-580 ms/tok**（三组交错实测 571 / 576 / 581），一旦所需专家被驱逐就出现 1.6-2.5 s 的尖刺（同一二进制、同一配置两轮可差 3.7 倍，逐 token 明细 321 / 321 / 2511 / 1069 / ...）。旧版 1,205 ms/tok 反而**均匀**（计算受限，I/O 被完全掩盖）。所以 decode 的 3.72 倍是**缓存最佳态**，最差态约持平。

> **口径对齐（与项目文档）**：本表两侧都跑**精确轨**（不设 `VLLM_ACTQ`），以保证位级一致。本表的 prefill 5.44 倍即文档《分布式专家提取_实施方案》第 9.12 节的「精确轨 GroupGEMM 6.55 倍」，decode 2.81 倍即同节「dn 行组并行 2.88 倍」。**文档里另外那些「3-3.4 s / 约 200 ms」是近似轨（`VLLM_ACTQ=1`，激活 int8 SDOT，长文本 t 约 24 起贪婪分叉）的数字，不可与本表混比。** 参考量级（板端 30B、逐层、t4、热、32 token prefill）：精确轨逐 token 约 37-60 s，精确轨加批式约 7-9 s（本表新版 6.7 s 即此档），`ACTQ=1` 加批式约 2.7-3.4 s。即本版把**精确轨**从约 37 s 提到约 7 s（5.3 倍），成为位级一致路径上真正可用的速度；但它仍是近似轨（约 3.0 s）的 **2.3 倍**，并未追平。**另需注意：在推荐配置（`VLLM_ACTQ=1` 加批式，即组合 5）下，新旧基本持平（交错 2 轮实测 prefill 旧 3.022/3.027 s vs 新 3.078/2.994 s（1.00 倍）、decode 旧 442/446 ms vs 新 430/433 ms（1.03 倍）。本次优化的价值在于让精确轨快 5.3 倍，而不是提速推荐配置。**

**8B（Qwen3-VL-8B，q4，6.60 GB；`/mnt/VQF/8b/qwen3vl8b.q4.vqf`，位于 SanDisk microSD）**

| 驻留 | 线程 / 缓存 | 旧版 prefill | 新版 prefill | 旧版 decode | 新版 decode |
|---|---|---|---|---|---|
| 全层 | t8 / 冷页缓存 | 71.10 s | 71.08 s | 470 ms/tok | 469 ms/tok |
| 全层 | t4 / 热页缓存 | 1.594 s | 1.605 s | 185 | 185 |
| 全层 | t8 / 热页缓存 | 2.127 s | 2.135 s | 468 | 468 |
| 逐层 | t8 / 冷页缓存 | 71.05 s | 71.03 s | 590 | 588 |
| 逐层 | t4 / 热页缓存 | 1.707 s | 1.724 s | 320 | 318 |
| 逐层 | t8 / 热页缓存 | 2.220 s | 2.209 s | 581 | 582 |

**全部在 1.0% 以内 -- 8B 在本版中没有任何变化**（原因见第 8 节）。峰值 VmHWM：全层约 4,226.7 MB、逐层约 593.2 MB（两版相同）；逐层相对全层省 **7.12 倍**。

> 口径：两个模型都跑 `--stream-n 32`，但 8B 会提前 EOS，**实际输出 29 个 token**（30B 为 32 个）。「冷」= 每点前 `sync; echo 3 > /proc/sys/vm/drop_caches`；「热」= 不 drop_caches，且热点前先跑一次丢弃的预热（`--stream-n 16`），使两个二进制起跑缓存态一致。

**附：30B-A3B「ACTQ 轨 x 驻留」2x2（同版本表征，**受控 A/B**）**

上面回答「新旧差多少」，本表回答**同一二进制内部**两个正交维度的取舍：`VLLM_ACTQ`（精确轨 / 近似轨）与 `VLLM_VQF_STREAM`（全层 / 逐层）。命令沿用本节口径（`--stream-n 32 --threads 4`，每点前先跑一轮 `--stream-n 16` 预热并丢弃），**同一轮内 4 个配置交错**，共 2 轮。

| 轨 | 驻留 | prefill 纯计算 | prefill 端到端 | GATEUP | decode（稳态） | 峰值 VmHWM | TOKIDS |
|---|---|---|---|---|---|---|---|
| 精确 | 全层 | 6,528 / 6,530 ms | 6.670 / 6.658 s | 6,240 ms | **325 ms/tok** | 12,286 / 12,278 MB | `41eae062e4ee` |
| 精确 | 逐层 | 6,514 / 6,539 ms | 6.946 / 6.962 s | 6,212 / 6,230 ms | **572 / 573 ms/tok** | **695 MB** | `41eae062e4ee` |
| ACTQ | 全层 | 2,644 / 2,645 ms | 2.762 / 2.781 s | 2,379 ms | **197 ms/tok** | 12,354 / 12,346 MB | `1fd42e485639` |
| ACTQ | 逐层 | 2,665 / 2,654 ms | 3.057 / 3.032 s | 2,396 / 2,385 ms | **428 / 430 ms/tok** | **695 / 691 MB** | `1fd42e485639` |

（两轮各一个值；「prefill 纯计算」取引擎 `[PREFILL-TIMING] total`，「端到端」取 `[STREAM] prefill ... in`。）

**三个可直接引用的结论**

- **ACTQ 轨**：prefill 纯计算快 **2.47 倍**（全层）/ **2.45 倍**（逐层），端到端快 2.40 / 2.28 倍；decode 快 **1.65 倍**（全层 325 -> 197 ms/tok）/ **1.34 倍**（逐层 572 -> 428）。
- **逐层驻留几乎不付计算代价**：prefill 纯计算与全层**持平**（精确 6,514/6,528 = 1.000 倍；ACTQ 1.006 倍），端到端只多 4.4% / 9.9%（多出来的是逐层建页与调度）。**代价几乎全在 decode**：精确轨 +76%（325 -> 572），ACTQ 轨 +117%（197 -> 430）。
- **峰值内存**：全层 12,278-12,354 MB -> 逐层 691-695 MB，**17.7 倍**。

**正确性（这组数据同时给出了「轨」的判据）**

4 次精确轨运行（2 轮 x 2 驻留）TOKIDS 完全相同（`41eae062e4ee`），4 次 ACTQ 轨运行也彼此相同（`1fd42e485639`），但**两轨之间不同**。即：**驻留方式不改数值（逐层与全层位级一致），换轨才改数值**（ACTQ 是近似轨）。精确轨运行的日志不含任何 `[ACTQ]` / `[ACTQ16]` 行；ACTQ 轨运行每次都打印 `[ACTQ] VLLM_ACTQ=1: q4 MoE int8-dot approximate track ON`。

**与上表的交叉印证（两次独立会话互证）**：本表精确轨全层 t4 热 prefill **6.670 / 6.658 s**、逐层 **6.946 / 6.962 s**，与本节新版同档的 **6.67 s / 6.93 s** 一致；decode 全层 325 ms/tok 与本节 **324 ms/tok** 一致。ACTQ 轨全层 2.76-2.78 s 亦落在上文所述「`ACTQ=1` 加批式约 2.7-3.4 s」区间内。故这批数字可作基准。

> **两处读数纪律**
>
> 1. **第 1 轮两个「全层」点的 decode 平均值（445 / 275 ms/tok）不可引用**。逐 token 明细显示 `r1_exact_full` 前 16 个 token 稳定在 322-325 ms，随后跳到 355 -> 907 -> 883 -> 859 -> 607 -> ...；`r1_actq_full` 前 23 个稳定在 195-203 ms，t=24 起跳到 675 / 346 / 644 / 605 / ...。而第 2 轮同配置 32 个 token 全程 323-328 / 196-199（变异系数 0.4%），两个逐层点也全程稳定（0.7-1.5%）。故第 1 轮全层 decode 判为**外部干扰（页缓存回收 / I/O）污染**，本表 decode 一律取第 2 轮口径；prefill 不受影响（两轮偏差 0.02-0.8%，可直接引用）。
> 2. prefill「端到端」与「纯计算」的差额（精确轨约 140 ms、逐层约 430 ms）来自进程启动与逐层建页开销，**不计入计算口径**，故两个数都给出。

> 环境：全程 `governor=performance`，温度 30.5-45.3 度，8 核频率恒定（A55 1,800,000 kHz / A76 2,304,000 kHz），页缓存稳定 15.3-15.5 GB；日志确认每点都走批式路径（`[batch]`）。

---

### 2) 稳态口径（serve，**新测点，单次非受控**）

单请求 = 292 token 上文 + 32 token 生成（greedy，`--threads 8`）。冷 = `drop_caches` 后第一次；热 = 紧接着重发同一请求。

| 模型 | 驻留 | 峰值 VmHWM | 冷 prefill | 热 prefill | 热 TTFT | 热 tpot |
|---|---|---|---|---|---|---|
| 8B | 全层 | 4,351,464 kB | 84,044 ms | 15,292 ms | 15,293 ms | 499.2 ms |
| 8B | **逐层** | **741,216 kB** | 84,294 ms | 15,700 ms | 15,700 ms | 624.7 ms |
| 30B-A3B | 全层 | 14,208,808 kB | 272,175 ms | 76,673 ms | 76,674 ms | 554.3 ms |
| 30B-A3B | **逐层** | **1,116,948 kB** | 274,256 ms | 78,275 ms | 78,276 ms | 875.6 ms |

> **30B 这一档的「热」只是部分缓存**：模型 17.66 GB 大于板载 15.9 GB，物理上装不满页缓存，数值依赖此前跑过什么。同一次会话内实测 30B 逐层基线档的 t1/t2/t3 = 60.6 / 68.1 / 77.9 s **单调变慢**即为此故。因此 30B 的 serve 数字**只作量级参考，不可作为精确基准**。8B 装得进 RAM，不存在该问题。

---

### 3) 与「有效优化组合」叠加（**新测点，单次非受控**）

组合 1 = `--sparse-attn --sparse-k 32 --l3-evict --l3-ratio 0.75 --l3-min-seq 128` 加环境变量 `VLLM_L3_PREFIX_REUSE=1`；组合 5 = `VLLM_ACTQ=1 VLLM_MOE_BATCH=1`（仅对 MoE / q4 有意义）。

**8B（L3 目录每次清零后实测，给出两轮值）**

| 驻留 | 配置 | t2 prefill | t3 prefill | 相对基线 |
|---|---|---|---|---|
| 全层 | 基线（`--no-prefix-kv`） | 15,086 ms | 17,423 ms | -- |
| 全层 | 组合 1 P3 | **2,907 / 2,930 ms** | **2,732 / 2,742 ms** | **-81%** |
| 逐层 | 基线 | 15,376 ms | 17,829 ms | -- |
| 逐层 | 组合 1 P3 | **3,076 / 3,184 ms** | **2,874 / 2,871 ms** | **-80%** |

**30B-A3B**

| 驻留 | 配置 | t2 prefill | t3 prefill |
|---|---|---|---|
| 全层 | 基线 | 75,947 ms | 81,846 ms |
| 全层 | 组合 1 P3 | 8,599 ms | 7,005 ms |
| 全层 | 组合 1+5 | **4,382 ms** | **3,692 ms** |
| 逐层 | 基线 | 68,058 ms | 77,905 ms |
| 逐层 | 组合 1 P3 | 9,098 ms | 7,443 ms |
| 逐层 | 组合 1+5 | **4,860 ms** | **4,106 ms** |

- 8B 的 P3 收益 **-80% 到 -81%**，且**必须在 L3 清零的条件下解读**：L3 若已被上一次运行预热，会得到明显更乐观的数字（历史记录里的 1,885 ms 即属此列）。
- 30B 的组合 5 在组合 1 之上再降约 **1.9 倍**（8,599 -> 4,382 ms），两个轴仍可叠加。

---

### 4) 8B：推荐档的价值兑现（以 2026-09-15 RK3588 远程复测为当前公开口径）

**线程 A/B（热页缓存，受控 A/B，各 2 轮）**

| 档 | `--threads 4`（旧 / 新） | `--threads 8`（旧 / 新） | 结论 |
|---|---|---|---|
| 全层 decode | 185 / 185 ms/tok | 468 / 468 ms/tok | **t4 快 2.53 倍** |
| 逐层 decode | 320 / 318 ms/tok | 581 / 582 ms/tok | **t4 快 1.82 倍** |

两版在同一档位下一致；RK3588 是 4 个 A76 加 4 个 A55，`--threads 8` 会把 4 个 A55 小核拉进 GEMM 并行区，**decode 慢约 1.8-2.5 倍**。**8B 务必用 `--threads 4`。**

**serve 逐层档 + 短请求（`--no-think`、`max_tokens=96`、自然结束于 70 token；Gitee 树远程复测）**

| 档 | TTFT | 端到端 | tpot | 峰值 VmHWM |
|---|---|---|---|---|
| 冷（`drop_caches` 后首次） | 72.613 s | 95.505 s | 331.8 ms | 612,868 kB |
| **热（连续 5 次同请求）** | **中位 1.734 s**（1.711~1.796） | **约 24.5 s**（24.408~24.610） | **中位 330.2 ms**（328.9~330.6） | 824,152 kB |

热态 5 次采样的 **tpot 极差仅 0.5%**，TTFT 极差约 **4.9%**；这组数字来自同一远程板、同一
二进制、连续 5 次同请求，适合作为当前公开推荐档口径。

**8B 为什么是推荐档（三条，均可复现）**：

1. **省内存的收益拿满**：`--stream-test` 冷页缓存下，权重常驻 4.19 GB 降到 **0.48 GB（8.7×）**；
   serve 逐层峰值 **0.82 GB**。
2. **热态有物理保障**：权重 6.15 GiB **装得进 15.9 GB 页缓存**，因此热态不必赌运气；远程复测
   5 次同请求的 tpot 仅在 **328.9~330.6 ms** 间波动。
3. **冷启动一次性成本低**：冷态首轮 TTFT **72.613 s**，因为成本正比于权重体积除以介质带宽。

**一键复现**：`sh tools/bench/bench_value.sh`（参数可用环境变量覆盖，用法见脚本头部注释）。

---

### 5) 30B-A3B 短请求：能跑到什么程度（**新测点，单次非受控**）

配置：逐层 `VLLM_VQF_STREAM=1` 加组合 1 加组合 5，`OMP_NUM_THREADS=4 VLLM_THREADS=4`。请求 = 22 token 上文 + 32 token 生成（greedy，流式）。本例上文仅 22 token，日志 `[L3] skipped: seq=22 < l3-min-seq=128`，**L3 与前缀复用未介入**，故本表反映「逐层 + MoE 档」。

| 档 | TTFT | 端到端 | tpot | 峰值 VmHWM |
|---|---|---|---|---|
| 冷（`drop_caches` 后首次） | 202.33 s | 236.81 s | 1,110.6 ms | 664,484 kB |
| **热（紧接着重发同一请求）** | **2.483 s** | **16.17 s** | **439.4 ms** | 919,012 kB |

**关键前提：`enable_thinking` 默认为开**（与 HF `apply_chat_template` 一致）。上表那 32 个 token **全部落在思考段内，并未产出答案**，因此补测了开关对照：

| 档（`max_tokens=256`） | TTFT | 端到端 | 实际输出 | 结果 |
|---|---|---|---|---|
| thinking **开**（默认） | 2.417 s | 134.45 s | **256（打满预算）** | 文本仍以 `<think>` 开头，**无答案** |
| **thinking 关**（请求体 `"enable_thinking": false`） | **2.437 s** | **20.82 s** | **40（自然 EOS）** | **给出完整答案** |

**结论：30B-A3B 在这块 16 GB 板上的可用档 = 关 thinking + 短上下文。** 关思考后 20.8 s 拿到完整短答，峰值内存 0.92 GB；默认（thinking 开）下 256 token 走不完思考段，一次问答实际要预留 300 个以上 token。冷态 202 s 的代价来自首次从 SD 卡读入全部 17.66 GB 权重，是一次性成本。

---

### 6) 存储介质（同机 `dd iflag=direct`，1 GiB，绕过页缓存）

| 介质 | 用途 / 挂载点 | 设备 | 顺序读实测 |
|---|---|---|---|
| **SanDisk microSD** | 8B / 30B 权重（`/mnt/VQF`） | `/dev/mmcblk1` | **64.5 MB/s** |
| eMMC | 2B 权重 / L3 目录（`/mnt/emmc`） | `/dev/mmcblk0` | **267 MB/s** |

逐层档「用时间换内存」里的时间**正比于权重体积除以介质带宽**：30B 冷态约 202-215 s 对应 SD 卡的 64.5 MB/s；若把权重放到 eMMC（267 MB/s），按带宽比推算可降到约 50 s 量级，**此为换算推算，未实测**。

---

### 7) 长上下文：与 llama.cpp 的权重公平 A/B（**2026-09-16 复测**）

**这是本章唯一的「对第三方引擎」口径**，与前六节（本版 vs 上一版、本版与自身配置）不可混引。

**本轮与上一轮的差别（引数前必读）**：2026-09-15 那一轮用的是 `md5_exe=71f8f53e…`；本轮换用修复后的
`md5_exe=a1b6707d…`（tokenizer 词表索引 + L3 整块落盘 + 回填 NEON 内核）。**四个档的两侧数据都在本轮同一会话内
重新测过**（两侧仍交错、每 arm 前 `drop_caches`），因此可整体替换上一轮；上一轮的数留在 2026-09-15 归档里，
其「复用轮是唯一还输的一项」等表述**已作废**，见结论 4 / 5。轮数：2K / 4K 各 3 轮、7.2K 2 轮、15.6K 1 轮。

**公平性怎么保证（四条，均可核）**

| 维度 | 做法 |
|---|---|
| 权重同源同量化 | 两侧用**同一份 Qwen3-VL-8B 权重**。本版 `q4` VQF 的文本路径约 **4.25 bpw**（group-16 Q4）；llama.cpp 侧用 **Q4_0（4.50 bpw）**，**不是 Q4_K_M**（后者 4.9 bpw、反量化更贵，拿来当对照会放大本版优势） |
| 权重留证 | 本轮三方 md5：`md5_exe=a1b6707d…`（上一轮 `71f8f53e…`）、`md5_gguf=96fb8a5e…`（两侧对照期间未变）、`md5_vqf=b6d8d1d7…`（16K 档的历史快照为 `f8ca1002…`，差异**仅来自 VQF 头的 `max_seq` 元数据字段**，权重体逐字节未变）。注：`kv2_run2.sh` 原版有一处缺陷（未 `mkdir -p /tmp/kv2`，导致 md5 行写不进 `summary.txt`），本轮已修，三方 md5 记于归档 `MANIFEST.txt` |
| 各自开各自优化 | 本版：`--sparse-attn --sparse-k 32 --l3-evict --l3-ratio 0.75 --l3-min-seq 128 --prefill-batch 256` 加 `VLLM_L3_PREFIX_REUSE=1 VLLM_TP_SPIN=1`，逐层 / 全层两档；llama.cpp：`llama-server -t 4 -ngl 0`，**默认前缀缓存**（日志 `selected slot by LCP similarity`） |
| 同协议 | 两侧都 **4 线程**；每个 arm 启动前 `sync; echo 3 > /proc/sys/vm/drop_caches`，随后打一发**丢弃用的预热请求**，再进计时轮；**不交错** |

**请求形态**：同一篇中文长文（按档位重复到目标长度）+ 3 轮追问，turn2 / turn3 为前缀追加、**必然命中前缀复用**；`temperature=0`、`max_tokens=64`。本轮：2K / 4K 各 **3 轮**、8K **2 轮**（第 3 轮按需提前停止），表内取各轮 turn1 中位；16K 本轮未测。

**原始产物归档**：本轮 [`docs/bench/20260916-rk3588-llama-ab/`](docs/bench/20260916-rk3588-llama-ab/MANIFEST.txt)、上一轮 [`docs/bench/20260915-rk3588-llama-ab/`](docs/bench/20260915-rk3588-llama-ab/MANIFEST.txt) —— 本节全部数字的原始出处（各 arm 各轮 `json` / 引擎日志 / `summary.txt` / `REPORT.txt`，以及产生它们的板端脚本与 md5 留证清单）。本节只做解读，**引数请回原始产物核对**。

**prefill（增量 token 口径，t/s，越大越好）**

| 档（实际 prompt token） | llama.cpp Q4_0 | 本版逐层 | 本版全层 | 本版领先 |
|---|---|---|---|---|
| 2.0K（1,996~2,034） | **17.61** | 13.23 | 13.36 | 0.75 倍（慢 1.32~1.33 倍） |
| 4.0K（3,976~4,014） | **14.61** | 12.68 | 12.71 | 0.87 倍（慢 1.15 倍） |
| 7.2K（7,216~7,254） | 11.02~11.18 | 12.30~12.38 | **12.31~12.38** | **1.11~1.12 倍** |
| 16K（15,616~15,654） | 7.25 | 11.61 | **11.68** | **1.60 / 1.61 倍** |

**decode（t/s，越大越好）**

| 档 | llama.cpp Q4_0 | 本版逐层 | 本版全层 | 本版领先 |
|---|---|---|---|---|
| 2.0K | **2.61** | 2.28 | 2.79 | 全层 **1.07 倍**（逐层慢 1.14 倍） |
| 4.0K | 1.50 | 2.16 | **2.54** | **1.69 / 1.44 倍** |
| 7.2K | 0.94 | 1.94~1.97 | **2.27~2.28** | **2.41~2.43 / 2.02~2.10 倍** |
| 16K | 0.48 | 1.42 | **1.74** | **2.96 / 3.63 倍** |

**峰值内存（VmHWM，GB，各档取该档最大）**

| 档 | llama.cpp Q4_0 | 本版逐层 | 本版全层 |
|---|---|---|---|
| 2.0K | 8.71 | **1.46** | 4.90 |
| 4.0K | 9.00 | **2.22** | 5.66 |
| 7.2K | 9.49 | **3.47** | 6.91 |
| 16K | **10.77** | **6.71** | 10.15 |

**multi-turn 复用轮（turn2 / turn3 的 prefill 时延，越小越好；两侧都已命中前缀复用）**

| 档 | llama.cpp | 本版逐层 | 本版全层 | 本版领先 |
|---|---|---|---|---|
| 2.0K | 4.79~4.92 s | 3.34~3.60 s | **3.28~3.60 s** | **1.33~1.50 倍** |
| 4.0K | 8.40~8.55 s | 4.10~4.45 s | **4.05~4.35 s** | **1.92~2.11 倍** |
| 7.2K | 14.43~15.26 s | 5.52~5.76 s | **5.40~5.70 s** | **2.50~2.83 倍** |
| 16K | 29.52~29.56 s | 8.68~9.23 s | **8.85~9.01 s** | **3.2~3.4 倍** |

> **档位与二进制口径**：2K / 4K 各 3 轮、7.2K 2 轮、15.6K 1 轮，**各档两侧都测于 2026-09-16 同一会话**（逐层/全层两档都测）。
> 16K 档跑在 **`governor=ondemand`** 下（其余三档为 `performance`——板子在这中间重启过，本轮已把 governor 顶回 performance
> 供后续测试使用），故该档绝对值可能偏保守约 5~8%，**档内两侧对照不受影响**。

**五条结论**

1. **decode 的交叉点仍在 1K 之后**：稀疏剪枝阈值是 `seq_len > g_sparse_k * g_sparse_block = 1024`，阈值以下本版的 attention 白吃稠密成本。**全层** 2.0K 已反超 **1.07 倍** -> 4.0K **1.69 倍** -> 7.2K **2.41 倍** -> 16K **3.63 倍**；**逐层** 2.0K 仍慢 1.14 倍，**4.0K 起领先**（1.44 倍）-> 7.2K 2.02~2.10 倍 -> 16K **2.96 倍**。
2. **prefill 的交叉点在 4K~8K 之间**：2.0K 慢 1.32~1.33 倍 -> 4.0K 慢 1.15 倍 -> 7.2K 反超 **1.11~1.12 倍** -> 16K **1.60~1.61 倍**。
3. **差距的真正来源是衰减率**：2.0K -> 16K 全程，本版 prefill **-12.2%**（逐层 13.23 -> 11.61）、decode **-37.7%**（2.28 -> 1.42）；llama.cpp prefill **-58.8%**（17.61 -> 7.25）、decode **-81.6%**（2.61 -> 0.48）。llama.cpp 的长上下文衰减更陡：其 16K prefill 在**单次请求内部就分段衰减**（上一轮逐段实测 17.72 -> 14.61 -> 12.40 -> 10.79 -> 9.59 -> 8.62 t/s），本版全程平稳。
4. **multi-turn 复用轮本轮全面反超**（上一轮为落后 1.10~1.25 倍，**该结论已作废**）：2.0K **1.33~1.50 倍**、4.0K **1.92~2.11 倍**、7.2K **2.50~2.83 倍**、16K **3.2~3.4 倍** —— 领先幅度随上下文单调递增（16K 上一轮还是落后 1.07 倍）。归因于本版自身三处修复，均已单独实测：
   - **tokenizer 词表索引**（复用轮的主要残差）：位级回归 1548 用例 0 失配；扣掉后未归因残差 7.2K 从 **10.38 s -> 0.16 s**、2.0K 从 **2.74 s -> 0.03 s**。
   - **L3 驱逐整块落盘**：`write` **3.478 s -> 0.062 s（-98.2%）**，驱逐阶段合计 **4.742 s -> 1.271 s（-73%）**。
   - **L3 回填 NEON 内核**：`store` **0.665 s -> 0.483 s**、`dequant` **0.179 s -> 0.117 s**、`restore` 墙钟 **0.848 s -> 0.630 s**。
   - 三者叠加（2.0K 同会话 A/B）：turn2 TTFT **6.630 s -> 3.845 s（-42%）**、turn3 **6.522 s -> 3.588 s（-45%）**；且 turn1 prefill 计算差 0.2%、turn2 GEMM 差 0.4%，输出逐字一致。
5. **上一轮本节的两处归因错误，此处更正**（原始产物仍在 2026-09-15 归档中，未改动）：
   - 旧文「`[L3] restored 13176 prefix blocks` = **515 MB、约 28 s**（约 20 MB/s，解压受限）」——**515 MB 这个数本身是对的，错在把它与 28 s 绑定**。515 MB 就是盘上 Q4 载荷：13176 blocks × 40 KB ≈ 515 MB，与本轮实测斜率一致（1692 blocks = 66.09 MB、6120 blocks = 239 MB，约 39 KB/block）。而 **28 s 是那一轮复用轮的整轮量级，L3 回填当时也远没有占满它**：本轮 2.0K 实测回填段墙钟 **0.630 s**，按块数线性外推 16K 约 **5~6 s**（且受写带宽限制，不是"解压受限"），因此「约 20 MB/s」这个由 28 s 反推出来的数也随之作废。那一轮复用轮的真正大头是 tokenizer 词表线性扫描（2.0K ≈ 2.7 s、7.2K ≈ 10.4 s；16K 按 token 数外推约 21 s，与实测 30.2~32.3 s 的量级相称）。
   - 旧文「**唯一还输的一项是 multi-turn 复用轮时延**」——见结论 4，修复后该口径下本版领先。

**边界（引用本节时必须一并引用）**

- **轮数与二进制口径**：2K / 4K 各 3 轮、7.2K 2 轮（第 3 轮按需提前停止）、16K 1 轮；**四档都测于本轮二进制 `a1b6707d…`**，两侧同会话交错。逐轮极差：prefill ≤1.0%、decode ≤3.5%、复用轮时延 ≤5.1%（最大出现在 7.2K llama 的 turn3，且该档仅 2 轮）。**16K 档的 governor 是 `ondemand`**（其余三档 `performance`），绝对值可能偏保守约 5~8%；该档两侧同时同 governor，故对照关系有效。
- **跑 16K 需要改 VQF 头的 `max_seq` 字段**（8192 -> 20480，4 字节；引擎从 VQF 头读 `max_seq`，无运行时开关）。该字段是**元数据**，不改权重数值。本轮把改动全程自证了一遍：改前 md5 `b6d8d1d7…` -> 改后 `f8ca1002…`（与上一轮 16K 的历史快照**逐字节相同**，反证只动了这 4 字节）-> `--restore` 回滚后 md5 与 `max_seq` 均回到基线（`b6d8d1d7…` / 8192）。
- **端到端 turn 时延不可比**：两侧 EOS 时机不同（llama.cpp 常只生成 6~17 token，本版 40~64）。本节因此**只用 per-token 速率与 prefill 时延**，不引用端到端秒数——注意 `summary.txt` 里的 `turn* total_s` 含 decode，本节的「复用轮时延」取的是响应 `metrics.prefill_ms`。
- **本节是性能对照，不做位级一致对拍**：两侧输出文本前几轮相同、其后分叉（采样与模板差异），不影响本节结论。
- **32K / 64K 未测**。KV 实测约 **264 KB/token**（由 `[L3] freed … MB / 前缀 token` 反推）：16K 时本版全层峰值 10.15 GB、llama.cpp 10.77 GB，都逼近 15.9 GB 上限；更长的上下文需要先解决 KV 常驻（本版可用 `--l3-evict` 与逐层，llama.cpp 在本板无对应手段）。

---

### 8) 诚实边界

- **8B 在本版中无变化，这是预期内的**：8B 是**稠密**模型，每个 decode token 都要读全部 6.15 GiB 权重，属**带宽受限**；本版优化针对的是**解包计算量**，因此对 8B 无效。30B-A3B 是 **MoE**，每 token 只激活 top-8 专家（约 1.6 GB），属**计算受限**，故收益显著。
  > 保留意见：按 6.6 GB 除以 0.185 s 反推需要 35.7 GB/s，高于此前记录的 25.74 GB/s，因此「8B 带宽受限」这一解释**仍需一次带宽实测确认**，此处只作为当前最佳解释。
- **30B 的「热」不可当精确基准**：模型 17.66 GB 大于 15.9 GB RAM，装不满页缓存，数值依赖缓存史（证据见第 2 节注）。第 2 / 3 节的 30B 数字请按量级使用。
- **逐层只支持明文 VQF**：VQF-Enc / 内嵌 SM2 签名的权重会被显式拒绝（需全层驻留）。
- 逐层与专家窗口 `VLLM_EW*` 不并存（EW 接管层入口钩子）。
- 逐层压缩的是**权重驻留**；KV 底座另由 KV v2 惰性分配约束（配合 `VLLM_KV_NOF32=1` 可把常驻进一步压低，见 [docs/KV缓存v2-惰性分配与分层驻留方案.md](docs/KV缓存v2-惰性分配与分层驻留方案.md)）。
- **2B 档已从本章移除**：早先使用的 `Qwen3-VL-2B-Instruct` dual 权重（4.16 GB）已不在现役板上；现有的是 `Qwen3-VL-2B-q8fix`（3.19 GB，q8 且非 dual），量化口径不同，不可与旧数字混比。
- **x86-64 分支仅用于功能自检与位级一致性对照**，不作性能基准。

---

## 启动流程

从裸板到 HTTP 就绪共 4 步：**构建 → 权重就位 → 启动 → 验证**。

### 1) 构建（RK3588 / aarch64 Linux）

```bash
# 板上原生构建（需 gcc + cmake）
./build_rk3588.sh                    # 构建 + 复制到 ./vllm_kestrel
./build_rk3588.sh --run-tests        # 构建 + 运行 PASS/FAIL 自检
./build_rk3588.sh --static           # 全静态（零 .so 依赖）

# 或直接在板上：
cmake -B build-rk3588 && cmake --build build-rk3588 -j8

# x86_64 Linux 主机交叉编译（需要 gcc-aarch64-linux-gnu）
./build_rk3588.sh --cross
```

#### x86_64 原生构建（Windows / MinGW，用于功能与一致性自检）

引擎的**一级目标平台是 aarch64**；x86-64 分支（`vllm_platform.h`）仅用于**功能自检与
位级一致性对照**，**不作为性能基准**——x86 上跑的绝对吞吐/加速比不能外推到板端。

```powershell
# Windows / MinGW-w64（gcc 需在 PATH，或用 -Gcc 显式指定）
powershell -ExecutionPolicy Bypass -File tools\build\check_x64.ps1
#   → 编译 + 跑 --test-l3 / --test-sparse 自检，退出码 0/1
powershell -ExecutionPolicy Bypass -File tools\build\build_x64.ps1 -Gcc D:\tools\mingw64\bin\gcc.exe
#   → 仅构建，产物 build-x64\vllm_kestrel_x64.exe
```

环境变量口径：`VLLM_GCC`（gcc 路径）、`VLLM_X64_OUTDIR`（输出目录）可替代命令行参数。

> 平台约束与说明：引擎本质是 **ARM（aarch64，ARMv8.2-A + dotprod/fp16）CPU 推理引擎**，
> **RK3588（4×A76 + 4×A55）是开发与基准测试平台**，并非唯一可运行设备。同类 aarch64
> Linux 设备可尝试编译运行，但设备画像（如 A76 集群线程绑定、核心数）与性能档按 RK3588
> 验证——**换板运行请先跑 `--test-l3` / `--bench-mixed` 自检**并以自检结果为准。运行建议
> `export OMP_NUM_THREADS=8`（RK3588：4×A76 + 4×A55）。非 aarch64 / 非 x86-64 架构会在
> `vllm_platform.h` 编译期报错退出。

### 2) 权重就位

引擎**只加载 VQF v2 单文件**（见「模型转换工具」得到 `model.vqf`）。模型目录需含：

```bash
# config.json + model.vqf（单文件 VQF v2，mmap 直挂）
# + 可选 vocab.bin（由 tools/build/build_vocab_bin.py 从 tokenizer.json 生成；
#   缺失时引擎回落到内嵌 vocab，功能可用但体积/词表以模型自带为准）：
python tools/build/build_vocab_bin.py <tokenizer.json> <vocab.bin> <vocab.bin>
```

### 3) 启动服务

```bash
# 基础启动：OpenAI 兼容 HTTP 服务（默认端口 8080）
./vllm_kestrel --serve --port 8080 --model <model-dir> --auto-load --wmode q4

# 长上下文优化档（全部可选；prefix-kv 前缀复用默认开启）：
# 注：8K 长上下文基准是 v0 测点（见附录 A），v1 未复测；下列开关为 v1 口径。
# --l3-evict 需配合环境变量 VLLM_L3_PREFIX_REUSE=1，二者共存才有 P3 收益
# （否则 L3 驱逐会静默打掉 prefix-kv，多轮追问将全量重算 prefill）。
VLLM_L3_PREFIX_REUSE=1 ./vllm_kestrel --serve --port 8080 --model <model-dir> --auto-load \
    --wmode q4 --sparse-attn --sparse-k 32 --spec --spec-k 4 \
    --l3-evict --l3-ratio 0.75 --l3-min-seq 128 \
    --disk-kv <kv-dir> --threads 8

# 大模型省内存档：逐层驻留（只支持明文 VQF）
VLLM_VQF_STREAM=1 ./vllm_kestrel --serve --port 8080 --model <model-dir> --auto-load --wmode q4
```

### 4) 验证就绪

```bash
# 健康检查 / 模型列表
curl http://<board>:8080/health
curl http://<board>:8080/v1/models

# 对话（OpenAI 兼容）
curl http://<board>:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-vl","messages":[{"role":"user","content":"你好"}],"max_tokens":64}'
```

- 管理页 / 对话页：`http://<board>:8080/admin/` 、`http://<board>:8080/chat/`
  （管理页含「有效优化组合（实测台账）」：8 个组合可一键套用，并标注每项优化
  针对 RAM / x86 是否有效；保存配置后需重启引擎生效）
- 换板或换版本首次运行：先跑内置自检（`--test-l3` / `--test-sparse` / `--bench-mixed`）
  并以自检结果为准。
- 可选 NPU 加速：默认后端为**零第三方依赖直驱**（自研写 stock rknpu 内核驱动）。
  首次使用先跑板端校准：`./vllm_kestrel --npu --npu-selftest --perf-only`
  （寄存器命令表未校准通过前提交路径保持禁用，自动回退 CPU）。

### 5) 分布式专家并行（EP）——**仅限可信网络**

把 MoE 专家分片到多台机器上协同推理，让单板内存放不下的大模型也能跑起来。

```bash
# rank0（协调者）与 rank r>0（工作者）；跨机时 --ep-host 填对端的板 IP
# rank0:
./vllm_kestrel --moe-ep-coord  --ep-nranks 2 --ep-host 0.0.0.0 --ep-port 29500 \
    --model <model-dir> --auto-load --wmode q4
# rank r:
./vllm_kestrel --moe-ep-worker --ep-nranks 2 --ep-host <rank0-ip> --ep-port 29500 \
    --model <model-dir> --auto-load --wmode q4
```

- EP 的正确性口径是**各 rank（含跨 ISA）输出逐位一致**；引擎会自动把 `VLLM_ACTQ16` 置 0
  （近似轨不参与跨机一致性对拍），无需手工设置。
- ⚠️ **EP 的传输协议没有鉴权、没有完整性校验、也没有版本号**：`--ep-host` 默认
  `127.0.0.1`（只监听本机），**一旦为跨机而绑定板 IP，该端口就对整个局域网开放**——
  任何能连上它的主机都可参与张量交换。请只在可信网络内使用，必要时用网段隔离或隧道，
  **切勿暴露到公网**。详见 [SECURITY.md](SECURITY.md)「三、安全面」。

---

## 模型转换工具（vqf_convert/）

**v1 起引擎只加载 VQF v2（只认自研格式）**，不再内置 safetensors/GGUF 加载与引擎内转换
路径（v0 曾支持，见「v0 / v1 能力对照」）；转换职责由随版发布的独立工具 **`vqf_convert/`**
承担（safetensors / GGUF → 单文件 mmap 的 VQF）。

```bash
# 构建转换工具（主机/板端均可；Windows 用 build.bat）
cd vqf_convert && ./build.sh
# 转换（明文）
./vqf_conv --model <safetensors-dir> --convert-vqf <out.vqf> --wmode q4
# 加密（VQF-Enc：SM4-CTR + HMAC-SM3）
VLLM_VQF_KEY='<pass>' ./vqf_conv --model <safetensors-dir> --convert-vqf <out.vqf> --wmode q4
# 内嵌 SM2 供应链签名（可与加密叠加）
VLLM_VQF_SIGN_PRIV='<64hex>' ./vqf_conv --model <safetensors-dir> --convert-vqf <out.vqf> --wmode q4
```

---

## 同态加密（FHE）密文推理链驱动（tools/drivers/）

在引擎的 RNS-CKKS 核心（`src/core/vllm_ntt.c` / `vllm_ckks.c` / `vllm_tp.c`）之上，本仓库提供**层链驱动**，用于把模型逐层跑在密文上（每层 = `lay` 前向 + `boot` 自举刷新）：

```bash
# lay / fin（层链 112 个素数）
gcc -O2 -fopenmp -Wno-implicit-function-declaration \
    -I include -I include/core -I include/common \
    -DCKKS_N=2048 -DCKKS_NPRIMES=112 -DBB=32 -DGG=32 \
    src/core/vllm_ntt.c src/core/vllm_ckks.c src/core/vllm_tp.c \
    tools/drivers/t23_m3p.c -o t23lay -lm

# boot（2100 个素数；大栈必需）
gcc -O2 -fopenmp -Wno-implicit-function-declaration '-Wl,--stack,33554432' \
    -I include -I include/core -I include/common \
    -DCKKS_N=2048 -DCKKS_NPRIMES=2100 -DBB=32 -DGG=32 \
    src/core/vllm_ntt.c src/core/vllm_ckks.c src/core/vllm_tp.c \
    tools/drivers/t23_chain.c -o t23boot -lm
```

- 用法、判据口径、数据目录约定、线程数实测结论见 [`tools/drivers/README.md`](tools/drivers/README.md)（**分口径**：层内 `[A+B]` 段 4 线程最快；整跳 `boot` 8 线程快约 20%；且线程数是结果变量，`NT=4`/`NT=8` 密文位级不一致）；
- **数据自备**：仓库不含 ~7 GB 数据包，用 [`tools/preproc/`](tools/preproc/) 自行生成（权重导出已用 SHA256 验证与已发布结果逐字节一致）；
- 运行 / 打包 / 验证脚本与接力指南：[`tools/relay/`](tools/relay/)（逐工具用法、判据口径、数据目录约定）；
- 归档产物目录由 [`tools/relay/collect_results.ps1`](tools/relay/collect_results.ps1) 按需生成（默认输出 `results/L0-4/`，不入库）；
- **社区接力**：层 0–4 已跑通，剩余 23 层正在征集算力——接力帖、操作手册、实测数据附录与 **0–4 层交接物归档（Release 资产）** 见协调仓
  [`kestrel-fhe-relay`](https://gitee.com/pei-xiaoguang/kestrel-fhe-relay)。

---

## 权重保护

**产出侧**（由 `vqf_convert/` 完成，二者可叠加）：

| 防线 | 开关（产出侧） | 机制 |
|---|---|---|
| 存储态加密 | `VLLM_VQF_KEY` | VQF-Enc：SM4-CTR + HMAC-SM3 |
| 供应链签名 | `VLLM_VQF_SIGN_PRIV`（64 hex） | 内嵌 SM2 签名，绑定权重来源 |

**加载侧**：置 `VLLM_VQF_KEY` 解密、置 `VLLM_VQF_SIGN_PUB` 验签；错误口令 / 篡改数据字节
均被拒绝（加载日志可见 `decrypted (SM4-CTR, HMAC-SM3 ok)` + `SM2 verify ok`）。

> 边界：加密 / 签名文件需**全层驻留**（`VLLM_VQF_STREAM` 会对加密 VQF 显式拒绝）。
> 全量路径与 `--stream` 路径均支持加密 / 签名；明文下二者产物**逐字节一致**（已 sha256 对拍）。
> 可验证推理的密钥目录须放在**支持 POSIX 权限的文件系统**（ext4/f2fs 等）上——vfat/exfat
> 上 `chmod` 不生效，私钥会变为世界可读，启动日志会打印 `[ATTEST] WARN`。

---

## 推理验证（attestation）

置 `VLLM_ATTEST=1`（可选 `VLLM_ATTEST_DIR=<密钥目录>`）即开启**逐响应出证**：

- 引擎启动时生成 / 加载 SM2 密钥对，`GET /v1/attest` 下发设备公钥（`pub`，128 hex）；
- 每个响应携带 `attest` 凭证：schema=3，magic `VLLM-AT-3`，**含请求原文绑定**
  （`body_sha = SM3(客户端原始请求体)`）——只改 `top_k` / `thinking` 也必须改摘要；
- 验签两条路径：**浏览器内自验**（对话页 / 管理页内建 SM3 + SM2 验签，零依赖）与
  **离线复验**（`tools/security/verify_attest.py`，纯 Python 零依赖，退出码 0=PASS / 1=FAIL，
  含 `--selftest`）。

> 边界：凭证证明「该设备产出且内容未被篡改」，其可信度依赖设备私钥的保管
> （私钥保护边界见上文「权重保护」）。方案全文见
> [docs/权重保护与可验证推理方案.md](docs/权重保护与可验证推理方案.md)。

---

## 模型与复现

> **模型支持范围（诚实声明）**：本引擎针对并实测验证的是**两类 Qwen3 架构**：
>
> - **Qwen3-VL 系列（2B / 8B）**——纯文本与图片/视频多模态；tokenizer、mrope、
>   DeepStack 视觉塔等均为该架构特化实现。
> - **Qwen3-MoE 系列（如 Qwen3-30B-A3B，文本）**——路由 + 逐专家 FFN 已接入
>   （`--moe-batch` / `VLLM_ACTQ` 等加速，见管理页「MoE 模型服务」组合）。
>
> **其他架构（Llama、旧版 Qwen / Qwen2 纯文本等）未经适配与验证**：转换可能报错或
> 输出不可用，请勿据此推定为通用推理引擎。
>
> **文中性能数据的模型与测点**：v1.0（2026-09-12）为 Qwen3-VL-2B / Qwen3-VL-8B /
> Qwen3-30B-A3B（均 RK3588 板端，逐层 vs 全层 × 组合⑧/①/①+⑤）；v0（2026-09-05，
> 见附录 A）为 Qwen3-VL-2B 板端与 Qwen3-VL-8B x86 基准机，**不得用于 v1 结论**。

- 模型权重不随仓库分发。Qwen 系列权重遵循其原始开源许可（Qwen 社区许可），
  下载后可用 `vqf_convert/` 转换为 VQF 后加载（见上文「模型转换工具」）。
- 基准数据复现方法、语料与驱动位置见
  [docs/RK3588_性能基准报告.md](docs/RK3588_性能基准报告.md) 附录（**v0 测点**）；
  v1 测点的复现口径见上文「性能」一节。
- **v1 一键复现**：`sh tools/bench/bench_value.sh`——产出逐层 vs 全层 A/B 内存、热态 5 次采样
  稳定度与页缓存证据（参数可用环境变量覆盖，见脚本头部注释；依赖仅 Python 3 标准库）。

---

## 体量与依赖（小而全）

| 项 | 数值 / 口径 |
|---|---|
| 可执行文件 | `vllm_kestrel` ≈ **0.8 MB**（RK3588 Release, `-O2 -s`，板端实测 818,872 B，v0 测点）；`-DVLLM_STATIC=ON` 全静态 ≈ 1.5 MB，`ldd` 零 .so 依赖 |
| 源码 | **28 个 C 文件**（main + core 11 + common 2 + serve 6 + model 6 + npu 2），C11，单工程单产物 |
| 运行时依赖 | **无第三方运行时**——标准构建仅需 gcc + libm（`-fopenmp` 仅 NPU pack 并行区使用 libgomp，系 gcc 自带；全静态构建一并内联，零 .so） |
| 代码内第三方 | 仅 `stb_image.h`（MIT, Sean Barrett）与 llama.cpp 派生 4x4 asm 内核（MIT, The ggml authors），见 [LICENSING.md](LICENSING.md) 第三节 |
| 自研件 | NEON 量化 GEMM/GEMV、线程池 `vllm_tp`（替代 OpenMP）、国密 SM3/SM4/SM2、VQF v2 mmap 格式、NPU 直驱 `/dev/rknpu` |
| 部署 | 单文件 + 可选 `vocab.bin`，拷贝即运行；VQF mmap 冷启动 **2.0 s**（v0 测点） |

同机对照（**v0 测点**，v1 未复测）：冷启动 2.0 s vs llama.cpp 5.0 s、峰值 RSS 2.47 GB vs 3.03 GB、
长上下文 decode 与 KV 恢复优势，原始数据与口径见文末附录 A 与
[docs/RK3588_性能基准报告.md](docs/RK3588_性能基准报告.md)。
> 诚实边界：二进制体积仅列本引擎自身（llama.cpp 动态/静态构建口径不同，未做同口径对比，
> 不作跨框架体积比较）。

### 零第三方依赖（自证清单）

「零第三方依赖」不是说"没写什么"，而是说**下面这些常见组件栈都没有引入**：

| 常见依赖项 | 本项目实际情况 |
|---|---|
| 推理框架运行时（Python / PyTorch / vLLM 栈） | 无——单一 C11 可执行文件即完整 HTTP 服务 |
| GPU 计算栈（CUDA / ROCm） | 无——纯 CPU + NEON 量化内核；NPU 仅走系统内核公共 UAPI 直驱 |
| 第三方推理/矩阵库（ggml、OpenBLAS、oneDNN…） | 无——GEMM/GEMV 手写（llama.cpp 派生 4x4 asm 为 MIT 提取件，文件头署名，见 LICENSING.md 第三节） |
| 密码学库（OpenSSL / GmSSL / MbedTLS…） | 无——SM3 / SM4-CTR / HMAC-SM3 / SM2 全自研并经国密标准 KAT 验证 |
| 图像/视频解码库（OpenCV / FFmpeg…） | 无——图片解码用单头 `stb_image.h`（MIT）；H.264 模块独立且默认不启用 |
| Web/HTTP 框架与 JSON 库 | 无——自研 select 轮询 HTTP + SSE、自研最小 JSON 解析 |
| OpenMP 运行时 | 引擎核心并行用自研线程池 `vllm_tp`；`-fopenmp` 仅 NPU direct 后端 pack 并行（libgomp 为 gcc 自带） |

依赖上限一句话：**动态构建只碰系统工具链标准件（glibc / libgomp）；全静态构建（`-DVLLM_STATIC=ON`）产物 `ldd` 报 not a dynamic executable，可拷到任意 aarch64 Linux 直接运行**——没有任何第三方项目运行时随引擎分发。

---

## 目录结构

```
├── CMakeLists.txt             # 构建（Release / 静态 / NPU 直驱默认）
├── build_rk3588.sh            # RK3588 构建 + 自检入口
├── cmake/toolchain-aarch64-rk3588.cmake   # x86 主机交叉编译工具链
├── include/  src/             # C11 源码（common/core/media/model/npu/serve）
│   ├── core/                  # 推理内核（NTT/FHE/CKKS/tp/matmul/attention/l3…）
│   ├── model/                 # 权重加载（纯 VQF v2 mmap）、视觉、分词
│   ├── serve/                 # HTTP/管理页/批处理/可验证推理(attest)
│   └── media/                 # H.264/MP4 解码（独立模块，默认构建不启用）
├── tools/                     # 自研工具
│   ├── bench/                 # 基准与延迟探针
│   │   ├── bench_value.sh     # 逐层价值实测：A/B 内存 + 热态稳定度 + 页缓存证据（见性能 §4）
│   │   ├── bench_http_probe.py  # 零依赖 HTTP 流式延迟探针（TTFT/tpot/峰值 VmHWM），供上者调用
│   │   └── patch_vqf_max_seq.py # 读写 VQF 头 max_seq_len（--dump / --set N / --restore），见性能 §7
│   ├── build/                 # 构建与代码生成
│   │   ├── gen_embedded_web.py  # HTML → 内嵌字节数组生成器（改页面后须重跑）
│   │   ├── build_vocab_bin.py   # tokenizer.json → vocab.bin（字节解码修复版）
│   │   ├── build_x64.ps1        # x86_64(MinGW) 原生构建脚本（非基准，仅一致性自检）
│   │   └── check_x64.ps1        # x86 构建 + 自检一条命令（退出码 0/1）
│   ├── client/                # vllm_client.py：OpenAI 兼容 HTTP 客户端（零依赖）
│   ├── ops/                   # vllm_mgr.py：引擎进程守护（start/stop/restart/状态页）
│   ├── security/              # verify_attest.py（离线验签）+ vllm_vqf_sign.c（VQF SM2 签名）
│   ├── npu/                   # npu_export_ops.py：RK3588 算子级 NPU 模型导出
│   ├── kernels/               # extract_llama_asm.py + llama_gemm_q4_0_4x4_asm.c（MIT）
│   ├── preproc/               # FHE 密文链数据预处理脚本
│   ├── drivers/               # FHE 密文推理链驱动（t23_m3p.c / t23_chain.c）
│   └── relay/                 # 密文链接力脚本：跑层 / 打包交接 / 验包 / 准确度校验
├── vqf_convert/               # 独立权重转换工具（safetensors/GGUF → VQF v2）
└── docs/                      # 技术文档 / 基准报告 / 安全方案（中文）
    └── bench/                 # 原始基准归档：每套测点一目录，命名 YYYYMMDD-平台-主题
        └── 20260915-rk3588-llama-ab/   # 性能 §7 的原始产物（json / 日志 / summary / 板端脚本），见其 MANIFEST.txt
```

---

## 文档（docs/，中文）

| 文档 | 内容 |
|---|---|
| [docs/技术文档.md](docs/技术文档.md) | 架构、模块、权重格式 VQF、内核、服务层、多模态、上下文管理、位级确定性、NPU、性能、调试、版本演进 |
| [docs/RK3588_性能基准报告.md](docs/RK3588_性能基准报告.md) | vllm_kestrel vs llama.cpp 冷启动 / 长上下文 / KV 恢复全矩阵（**v0 测点，对应附录 A，不适用于 v1**） |
| [docs/优化配置与边界说明.md](docs/优化配置与边界说明.md) | 各优化档机制、收益与诚实边界（含有效组合与 x86 复核口径；**部分数字为 v0/热缓存口径**） |
| [docs/KV缓存v2-惰性分配与分层驻留方案.md](docs/KV缓存v2-惰性分配与分层驻留方案.md) | KV 惰性分配、L3 分层驻留与 P3 前缀复用共存（L3 驱逐 + 前缀复用） |
| [docs/权重保护与可验证推理方案.md](docs/权重保护与可验证推理方案.md) | 三道安全防线：VQF 存储态加密、SM2 供应链签名、推理出证（attestation schema=3，含请求原文绑定），含相互关系、端到端用法与统一安全边界 |
| [docs/bench/](docs/bench/20260915-rk3588-llama-ab/MANIFEST.txt) | **原始基准归档**：正文各测点的原始产物（各轮响应 json、引擎日志、`summary.txt`、`REPORT.txt`）与产生它们的板端脚本、md5 留证清单；每个测点一个目录，入口见目录内 `MANIFEST.txt` |

> 不想逐篇翻文档？直接看 **[项目 Wiki](https://gitee.com/pei-xiaoguang/kestrel-llm/wikis/Home)**——
> 按「上手 → 原理 → 数据与调优 → 安全」组织的导航页，并统一了**术语与数据口径**
> （版本分区、冷/热页缓存、纯权重 RSS vs serve 峰值）。Wiki 为中文，由上述 `docs/` 提炼，
> 细节以原文为准；同一套内容也随仓库分发在 [`wiki/`](wiki/Home.md)（克隆后离线可读）。

---

## v1.0 变更摘要（相对 v0）

**工程与形态**

- **纯 VQF 运行时**：引擎只加载单文件 VQF v2（mmap 直挂），删除内置的 GGUF /
  safetensors 加载与引擎内转换路径；转换职责移交随版发布的独立工具 **`vqf_convert/`**。
- **源码瘦身**：随纯 VQF 运行时移除 `vllm_gguf.c/.h` 与 `convert.html` 等遗留件；
  单产物仍约 0.8 MB；`CMakeLists.txt` 版本号提升至 `VERSION 1.0.0`。
- **x86_64 分支随版**：`vllm_platform.h` 提供 x86-64（MinGW/MSVC）移植层；新增
  `tools/build/build_x64.ps1` / `tools/build/check_x64.ps1`（已参数化，`VLLM_GCC` /
  `VLLM_X64_OUTDIR` 可覆盖）。**x86 仅供功能自检与位级一致性对照，不作性能基准。**

**性能与内存**

- **逐层推理可交付化**（`VLLM_VQF_STREAM=1` 分层驻留 + KV v2 惰性分配）——权重常驻 RSS
  实测省 **4.6× / 8.9× / 23.2×**（2B / 8B / 30B-A3B），使 17.66 GB 的 30B-A3B 能在
  16 GB 板上服务；TOKIDS 与全层档**逐位一致**（语义不变）。口径与完整对照见
  「性能」一节。
- **P3：L3 驱逐 × 前缀复用共存**（`--l3-evict` + `VLLM_L3_PREFIX_REUSE=1`）——多轮
  追问轮 prefill 实测 **−93%~−97%**（v0 时期口径，热缓存稳态）/ 2026-09-12 v1 同口径
  成矩阵复测 **−80% ~ −96%**（2B/8B/30B-A3B × 全层/逐层，300 token 上文），为 v1.0
  收益最大的单项优化；30B-A3B 再叠加 MoE 组合⑤后追问轮 prefill 降到 **3.2~5.3 s**
  （相对基线 −99.3%）。
- **L3 紧凑布局**：按驱逐顺序连续分配 `disk_off`（`wcursor`），文件 / RAM mirror
  尺寸跟踪真实载荷而非全 KV 窗。
- **P1/P2 内存分页**：mirror 跨轮复用 + `MADV_DONTNEED`；arena 化 + `imp_sum`
  层内共享。
- **管理页实测台账**：`/admin/` 新增「有效优化组合」表（8 组，可一键套用），并对
  每项优化标注**针对 RAM / x86 是否有效**；补齐 P3 开关与 `--l3-evict` 未开 P3 门
  时的联动告警。

**安全与正确性**

- **attestation schema 3**：新增**请求原文绑定**（`body_sha = SM3(客户端原始请求体)`），
  只改 `top_k` / `thinking` 也必须改摘要。
- **转换工具支持加密 / 签名产出**：`vqf_convert/` 设 `VLLM_VQF_KEY` 即输出 VQF-Enc 加密
  文件（SM4-CTR + HMAC-SM3），设 `VLLM_VQF_SIGN_PRIV` 即内嵌 SM2 供应链签名，二者可叠加；
  全量与 `--stream` 路径均可。引擎侧加载日志实测 `decrypted (SM4-CTR, HMAC-SM3 ok)` +
  `SM2 verify ok`，错口令 / 篡改数据字节均被拒绝。
- **VQF 离线补签口径修复**：离线签名工具先置 `VQF_FLAG_SIGNED` 再算摘要（与写侧
  口径一致），修复补签后永远 digest mismatch 的问题。
- **Debug 构建修复**：`CMAKE_C_FLAGS_DEBUG` 补 `-march`，避免 NEON dotprod 内联
  在 Debug 档编译失败。
- **注释编码修复**：修正历史遗留的若干源码注释乱码（`vqf_convert/src/conv_main.c`、
  `src/serve/vllm_server.c`、`src/serve/vllm_batch.c`）。

---

## 附录 A：v0（历史版本）实测数据（2026-09-05，**不适用于 v1**）

> **警告：以下数据全部在 v0 上测得，v1 未复测，不得用于 v1 的任何结论。**
> v1 已移除 GGUF / safetensors 加载与引擎内转换（改为只认自研 VQF v2 单文件），并引入
> **逐层推理**与 **KV v2**，因此**冷启动、常驻内存、长上下文与多轮 prefill 的口径都已变化**。
> 本附录只作版本演进对照与历史参考；v1 的性能主张一律以正文
> 「性能」（2026-09-12 测点）为准。

平台：Orange Pi 5 Plus（RK3588，8 核，15GB RAM，eMMC，无 GPU/NPU 参与）。
模型：Qwen3-VL-2B-Instruct（vllm_kestrel VQF q4 全优化档 vs llama.cpp GGUF Q4_0）。
完整方法学、口径与原始数据见 [docs/RK3588_性能基准报告.md](docs/RK3588_性能基准报告.md)。

### A.1 冷启动
| 引擎 | spawn→HTTP ready | 峰值 RSS (VmHWM) |
|---|---|---|
| vllm_kestrel（VQF mmap） | **2.01 s** | ~2465 MB |
| llama.cpp (GGUF mmap) | 5.02 s | ~3027 MB |

### A.2 长上下文（内容 token 1K/2K/4K/8K，生成 64 token）
| 档 | vllm decode (TPOT) | llama decode (TPOT) | decode 比 |
|---|---|---|---|
| 1K | 73.4 ms | 102 ms | vllm 1.4× |
| 2K | 84.6 ms | 126 ms | vllm 1.5× |
| 4K | 102 ms | 218 ms | vllm 2.1× |
| **8K** | **136.7 ms** | **411.6 ms** | **vllm 3.0×** |

- prefill：1K–4K llama 快 1.4–1.6×（ggml NEON Q4 更成熟），**8K 拉平**
  （vllm 40.0 vs llama 38.3 tok/s）；vllm 预填吞吐 4K→8K 不降，llama 每档下降 20%+
  （sparse-attn 的 O(n·k) 效果）。
- decode 随上下文放大是架构级差异：llama f16-KV 每词全扫 8K、2K→8K 劣化 3.3×；
  vllm q8-KV + 稀疏 decode 仅 1.6×。

### A.3 KV 缓存恢复（同进程前缀复用 vs 跨进程磁盘恢复）
- 同进程第二轮（29 token 增量 prefill，8K 上下文）：**vllm 2.0s vs llama 6.63s（3.3×）**。
- 跨进程磁盘 KV 恢复（`--disk-kv`，8K 会话 F32 快照 1.87GB，重启后 15.4s 恢复 vs
  全量 prefill 202s）：**vllm 13.1×**；llama.cpp 无等价物（重启即失）。

> 诚实口径：本报告为板内单引擎串行测量、权重同源（Qwen3-VL-2B safetensors）但量化
> 格与算子不同（非位级同一权重），数值为各自引擎原始字段对齐后的并列展示。
> 复现方法与数据文件见报告附录。

---

## 许可与合规

- **许可：双许可（AGPL-3.0-or-later 或 商业许可，二选一）**——本项目是**自由软件**：
  你可以依 **GNU Affero 通用公共许可证 v3.0 或更新版本**（SPDX：`AGPL-3.0-or-later`）
  自由使用、修改与分发；**若你不能或不愿承担 AGPL 的源码开放义务**
  （例如在闭源产品中集成、以闭源方式提供商业服务 / SaaS），则**须先取得商业授权**。
  完整条款见 [LICENSE](LICENSE)（AGPL-3.0 正文），双许可条款与商用授权见 [LICENSING.md](LICENSING.md)，贡献规则见 [CONTRIBUTING.md](CONTRIBUTING.md)。
- **第三方组件**按各自许可保留：`stb_image.h`（MIT, Sean Barrett）与源自 llama.cpp
  的 4x4 asm GEMM 提取文件及其派生内核（MIT, The ggml authors）——版权与许可
  文本见对应文件头，详见 [LICENSING.md](LICENSING.md) 第三节。两者均与 AGPL 兼容。
- **安全漏洞请私下报告**（不要开公开 Issue），流程与承诺见 [SECURITY.md](SECURITY.md)。
- 本引擎的 NPU 直驱后端仅与操作系统内核驱动（stock rknpu）的公共 UAPI 交互，
  不包含任何闭源库或第三方头文件。

## 联系

商业授权 / 研究合作 / 复现数据：**398152090@qq.com**
（也可通过 [Issues](https://gitee.com/pei-xiaoguang/kestrel-llm/issues) 或 Gitee 站内私信联系作者；
商业许可协议与双许可条款见 [LICENSING.md](LICENSING.md)）

For commercial licensing / research collaborations / data requests:
**398152090@qq.com**, or open an issue at https://gitee.com/pei-xiaoguang/kestrel-llm/issues.
