# 逐层推理与 KV 缓存

本页是 Kestrel（红隼，工程名 `vllm_kestrel`）v1 最核心的技术页面，讲清两条**正交**的内存轴：

- **(a) 逐层推理**（`VLLM_VQF_STREAM=1`）——管**权重驻留**：VQF 单文件 mmap 直挂，每层权重只在参与计算时建立文件页，算完立即 `MADV_DONTNEED` 释放，仅 keep 层（缺省 1 层）常驻，从而让**权重常驻内存与模型体积、层数解耦**。
- **(b) KV v2（惰性分配 + L3 分层驻留 + P3）**——管 **KV 底座**：KV 物理占用从「满窗预分配」变「按实际写入增长」；L3 驱逐把冷块落盘，`VLLM_L3_PREFIX_REUSE=1` 让 L3 驱逐与内存前缀复用共存。

两者互不替代，README 结论明确「**权重驻留轴与 KV 分页轴正交**」——同一优化组合在逐层/全层两种驻留档下收益同量级即为例证。

> 口径提示：标注 `（v0 测点）` 的数字取自旧版加载逻辑，不得用于 v1 任何结论（v0 = 2026-09-05）。本页逐层推理与组合①数字为 v1.0 板端实测测点 **2026-09-12**；KV v2 专项实测为 **2026-09-07** 板端测点，均单独标注。

## 一、逐层推理（`VLLM_VQF_STREAM=1`）

### 1.1 机理与作用域

VQF 单文件仍以 mmap 直挂，但不一次性把全部权重触入内存：每层只在参与计算时建立文件页，算完立即 `MADV_DONTNEED` 释放，仅 keep 层（缺省 1 层）常驻。实现位于 `src/model/vqf.c` 的 `vqf_stream_*`（层切片表 + madvise/fadvise + RSS 采样 + 加密拒绝），`src/main.c` 的 `--stream-test` 为验证 harness。效果是**权重常驻 RSS 不再随模型体积线性增长**：17.66 GB 的 Qwen3-30B-A3B-q4 可在 16 GB RAM 的 RK3588（MemTotal 15.6 GB）上服务。

生效自证（板端日志原文）：

```text
[VQF-STREAM] enabled keep=1 nl=48 segs=11 per-layer=334.1MB data=16847.2MB resident~808.4MB rss=988kB
```

### 1.2 实测：纯权重口径（`--stream-test`，冷页缓存，v1.0 2026-09-12）

每配置点先 `sync; echo 3 > /proc/sys/vm/drop_caches` 再跑，为冷页缓存首次前向：

| 模型（单文件 VQF） | 全层：权重常驻 / 峰值 | 逐层：权重常驻 / 峰值 | 常驻倍率 |
|---|---|---|---|
| 2B Qwen3-VL（4.16 GB） | 1,806,732 / 2,761,732 kB | **394,464 / 631,428 kB** | **4.6×** |
| 8B Qwen3-VL（6.60 GB） | 4,183,048 / 4,224,032 kB | **471,800 / 593,172 kB** | **8.9×** |
| 30B-A3B MoE（17.66 GB） | 11,728,860 / 12,703,020 kB | **506,368 / 713,240 kB** | **23.2×** |

### 1.3 实测：serve 峰值（含 KV，v1.0 2026-09-12）

单请求 = 300 token 上文 + 32 token 生成（greedy），峰值为 VmHWM，**含 KV 缓存**，故倍率小于纯权重口径：

| 模型 | 全层峰值 | 逐层峰值 | 倍率 |
|---|---|---|---|
| 2B | 2,968,920 kB | **854,476 kB** | 3.5× |
| 8B | 4,523,240 kB | **914,448 kB** | 4.9× |
| 30B-A3B | 14,228,868 kB | **1,112,500 kB** | 12.8× |

30B 全层档峰值 14.2 GB 已顶到 16 GB 板的上限；逐层档把同一模型压到 1.11 GB，留出全部 KV 与进程余量。**逐层真正解耦的是权重，KV 由 KV v2 惰性分配另管。**

### 1.4 语义不变与代价

驱逐只丢弃干净文件页，内容由文件重建，不改数值：`--stream-test` 的贪心 TOKIDS 序列在「全层 / 逐层」两档下**逐位一致**，序列 md5 前 12 位 2B `1a5d48906a4c`、8B `efb5a00c5803`、30B-A3B `d4996200fcf2`（同一模型两档取值相同）。

代价随「权重体积 ÷ 存储带宽」上升：

- `--stream-test` 冷页缓存：prefill **+12% / −1% / −0.1%**，decode **+16% / +37% / +16%**（2B / 8B / 30B）。
- serve 稳态：热档 tpot **+38.7% / +34.6% / +18.2%**，热档 prefill 只 **+3.3% / +2.7% / +0.4%**；2B 典型例热态 tpot 从 113.0 ms 升至 156.7 ms（+38.7%）。

逐层档每轮重新触碰权重页（热档仍受存储带宽约束）——这正是「用时间换内存」的边际成本。

### 1.5 边界（照实写）

- **只支持明文 VQF**：VQF-Enc / 内嵌 SM2 签名的权重会被**显式拒绝**，须全层驻留。
- 与专家窗口 `VLLM_EW*` **不并存**（EW 接管层入口钩子）。

## 二、KV v2 惰性分配（#1）

### 2.1 问题：`calloc` 全量触页

v1 原 KV 预分配是「init 全量触页」：`st_qwen_inference_init` 无条件为 **f32 + INT8(q8) 双副本 × 28 层 × max_seq 8192** 逐 block `calloc`（`src/model/vllm_safetensors.c`：`alloc_kv_blocks_f32 / alloc_kv_blocks_i8 / alloc_q4_blocks`）。`KV_GUARD=4096`（`include/model/vllm_safetensors.h`），`g_sparse_block=32` → 每层 256 个 block；block 小（f32 每块 ≤128KB），`calloc` 小块清零会 **memset 触碰每一个 payload 页** → init 结束 RSS 即 ~2.46 GB（2026-09-07 实测）。

### 2.2 机理：mmap 匿名惰性零页

新分配器 `kv_block_alloc_raw / kv_block_free_raw`（`src/model/vllm_safetensors.c`）：

- 默认（`g_kv_lazy=1`，init 按 `!g_xq_pages` 置位）：每个 block 用 `mmap(MAP_PRIVATE|MAP_ANONYMOUS)` 分配 `KV_GUARD+payload+KV_GUARD`，**不写 payload、不预写 0xA5 canary**；
- 未写入的 block 读回全 0，与旧 `calloc` 清零语义**逐字节等价** → 推理输出位级不变；
- `VLLM_XQ_PAGES=1`（ASan / 越界调试档）保持旧 `cf_pg_alloc + 0xA5 canary` 布局不变。

配套：三处 block 分配、L3 逐块驱逐后再分配（`st_qwen_kv_rebuild_freed`）、`st_qwen_inference_free` 内六组释放均改走新分配器；新增导出 `st_qwen_kv_free_block(payload, payload_bytes)`（`include/model/vllm_safetensors.h`），`src/main.c` 的 L3 on-disk 逐块释放改用它。

### 2.3 动态窗口 `VLLM_KV_MAXSEQ`

`VLLM_KV_MAXSEQ`（`st_qwen_inference_init`）在 `>32768→2048` 兜底之后、`--bench-seqlen` 抬升之前生效：缺省不改现状（2B=8192），edge 部署可收紧（`2048 / 512`）。惰性分配后它主要约束可寻址窗口与一次性缓冲，仍用于锁小虚拟上界。

### 2.4 实测（2026-09-07，2B / RK3588）

`--stream-test` harness，32-token prefill + 6-token greedy decode：

| 档位 | init RSS | prefill(32tok)后 | decode 稳态 |
|---|---|---|---|
| v1 baseline（calloc 全触页） | ~2.46GB | 3.41GB | 3.41GB |
| v1 + `VLLM_VQF_STREAM=1` | ~2.46GB | 2.66GB | 2.66GB |
| A：v2 懒分配（全驻留） | **13.9MB** | ~964MB | ~969MB |
| B：v2 + `VLLM_VQF_STREAM=1` | 13.8MB | **~222MB** | ~218–222MB |
| C：B + `VLLM_KV_MAXSEQ=512` | **7MB** | ~209MB | ~209MB |

- init 降幅：A 档 −99.4%；C 档 init 打印 max_seq=512 确认窗口旋钮生效。
- 六档 TOKIDS **全一致**：`100062 99371 58814 113272 101158 18493`，文本一致，验证语义等价。
- decode 时间：A ~214ms/tok；B/C ~266–267ms/tok（与 v1 分层档一致，惰性不引入计算开销）。
- KV 增长斜率 ≈280KB/pos（f32 K+V 229KB + i8 K+V 57KB）：1024 token 时 KV ≈0.29GB，相对 2.35GB 降幅 >90%。

### 2.5 诚实边界：canary 调试能力降级

惰性档（发布默认）不再预写块尾 0xA5 canary，`[GUARD]` 溢出探测会恒误报，故在惰性档下以 `!g_kv_lazy` 门控关闭。**代价**：发布默认档不再有「块尾 0xA5 溢出哨兵」，检测改由 mmap 隔离承担（单块独立映射，越界不污染相邻块/堆元数据）。需要哨兵调试或 ASan 覆盖 KV block 内部越界时用 `VLLM_XQ_PAGES=1`（走旧布局）。

## 三、去 f32 正典（`VLLM_KV_NOF32=1`，实验档）

f32 KV 是全引擎**读写正典**，不是冗余。`VLLM_KV_NOF32=1`（默认关，实验档）让 q8 档不再分配 f32 正典 KV 数组（init 跳过 `alloc_kv_blocks_f32`，`k_cache[l]/v_cache[l]` 保持 NULL）。迁移前须先理清 f32 的读/写点：

| 读/写点 | 源码符号（默认 `src/model/vllm_safetensors.c`，注明除外） | 用途 |
|---|---|---|
| 写 f32 行 | decode `st_qwen_model_forward`、prefill `st_qwen_model_prefill_batch`、multimodal `st_qwen_model_multimodal_prefill_ex` 的 `memcpy(kv_row_f32(...))` | 每一 token 的 K/V 都落 f32 |
| decode 注意力 | `use_q8 ? INT8-flash : f32` 两分支；f32 分支（非 q8、含 fallback）与 tile-packed flash 读 f32 | 主注意力路径之一 |
| 前缀复用 | `st_qwen_copy_kv_prefix` 逐 block `memcpy` f32 行（同时复制 q8） | 跨请求前缀 KV 复制 |
| L3 驱逐 | `src/main.c` `l3_evict_layer(... st->k_cache[l] ...)` 以 f32 块为单元扫描/评估 | 重要性评估读 f32 |
| disk-kv | `st_qwen_kv_disk_save/load` fwrite/fread f32 行 | 快照/恢复 |
| 稀疏注意力 | `sparse_attn_head` 体系基于 f32 块 + L3 q4 落盘镜像 | 长上下文 |

实测与语义（2026-09-07）：nof32 语义 ≡ q8 反量化世界（pack 从 q8+scale 反量化 `q*(scale/127)` 与 dequant 写入 f32 行**逐位相同**，COPYTEST/DKVTEST `diff=0 PASS`）；相对 f32 精确正典的质量差即门禁测得（4 段语料均值 ΔNLL **−0.079**，PASS ≤ +0.10 门槛）。serve 评估（2B）：同阶段 RSS 较 f32 双份档省 **227–306MB（−22%）**；KV 增量斜率 **316 → 92 KB/token（−71%）**；VmSize 由 ~5.4–5.5 GB 降至 ~3.4–3.6 GB。

边界（照实写）：nof32 仅 q8 档 + 惰性分配有效，`VLLM_XQ_PAGES=1` 调试档自动退回双份；DKV v1（f32）旧快照在 nof32 下 `load` 拒绝（rc=-1）；长上下文下 f32 与 nof32 贪心输出可发生**内容级分叉**（P4-D 于 ~char18 处；多模态 20 轮评估中 F 答「12」、NS 答「20」）——对外口径统一为「**确定性但近似**，语义级稳健，细节计数类长任务可能分档差异」。

## 四、L3 分层驻留与 P3（L3 驱逐 × 前缀复用共存）

### 4.1 L3 棘轮修复（2026-09-07）

sparse+L3 的 serve 多轮 RSS 曾出现「每轮 +309MB 无收益棘轮」。根因双因：① `l3_load_to_mem` 原按全容量 `nl×max_blocks×block_bytes`（8192 max_seq 下 ≈280MiB）malloc+read，且 serve 同用户多轮无清理点 → 每轮泄漏一个 280MiB 全容量镜像；② 固定偏移文件布局使 extent 化无法压缩。

修复（`src/core/vllm_l3.c` + `include/core/vllm_l3.h` + `src/main.c`）：**紧凑布局**（`STL3State.wcursor` 按驱逐顺序连续分配 `disk_off`）、**extent 镜像**、**轮间释放**（`l3_evict_after_prefill` 入口先 `l3_state_free`）。板端 4 轮复测：mirror 280MiB → 29.5MiB；每轮 +309MB → 前两轮平台级（+21.7MB），**r2 之后每轮 ~+52MB 残余为开放项**（与 k_pack/镜像泄漏无关，是独立 anon 页/映射现象，记录在案不掩盖）；`--test-l3` PASS，轮末绝对 RSS 低于 dense 档。

### 4.2 P3：L3 驱逐 × 前缀复用共存

L3 驱逐（`--l3-evict`）默认与内存前缀复用**互斥**（L3 释放前缀块后 prefill 无 NULL 检查，`ist_reset` 门）；v1.0 起置 `VLLM_L3_PREFIX_REUSE=1` 即让二者**共存**。

- README v1.0（2026-09-12）：组合① = `--sparse-attn --sparse-k 32 --l3-evict --l3-min-seq 128` + `VLLM_L3_PREFIX_REUSE=1`，追问轮 prefill 降幅 **−80% ~ −96%**（2B −80.4%/−83.1%、8B −89.5%/−89.8%、30B-A3B −95.9%/−96.0%，均逐层档）。
- 源文档记录（2026-09-07）：多轮追问轮 prefill 实测 **−93%~−97%**，为 v1.0 收益最大的单项优化。
- 同一组合在**两种驻留档下收益同量级**，即「权重驻留轴与 KV 分页轴正交」的直接证据。

生效自证（2B 全层档板端日志原文，L3 落盘 + 回填 + 前缀复用三段可见）：

```text
[L3] evicted 252 blocks -> /mnt/emmc/l3bench_u65bbfe41 (cursor=9.84 MB, seq=345, keep=332, ratio=0.75, ...), freed 78.8 MB from RAM
[L3] restored 252 prefix blocks from Q4 payload (prefix=377)
[KV-PREFIX] reuse 377-token KV prefix, prefill rest
```

## 五、可复现命令与环境变量

以下命令与环境变量均来自 README「性能」一节与 KV v2 源文档；README 复现口径为 `--stream-test` 用 `--threads 8`，serve 用 `--serve --port 18080 --device arm-rk3588-opi5 --auto-load`。

```bash
# serve 基线（组合⑧）
./vllm_kestrel --serve --port 18080 --device arm-rk3588-opi5 --auto-load --no-prefix-kv
# 组合①：L3 驱逐 × 内存前缀复用
./vllm_kestrel --serve --port 18080 --device arm-rk3588-opi5 --auto-load \
  --sparse-attn --sparse-k 32 --l3-evict --l3-ratio 0.75 --l3-min-seq 128 --l3-path /mnt/emmc/l3bench
# 关键环境变量
VLLM_L3_PREFIX_REUSE=1          # P3：L3 驱逐 × 内存前缀复用共存
VLLM_VQF_STREAM=1               # 逐层推理（权重驻留）
VLLM_KV_NOF32=1                 # 去 f32 正典（q8 单份）
VLLM_ACTQ=1 VLLM_MOE_BATCH=1    # 组合⑤：MoE 专家激活量化 + 专家批量（仅 q4 权重，30B-A3B）
```

KV v2 逐层档复现（源文档 §5 原文口径，模型 `/mnt/emmc/day25_fixed`）：

```bash
./vllm_kestrel --stream-test --model /mnt/emmc/day25_fixed --stream-n 6
VLLM_VQF_STREAM=1 ./vllm_kestrel --stream-test --model /mnt/emmc/day25_fixed --stream-n 6
VLLM_VQF_STREAM=1 VLLM_KV_MAXSEQ=512 ./vllm_kestrel --stream-test --model /mnt/emmc/day25_fixed --stream-n 6
```

其余调试/诊断开关（均 env 门控，默认关）：`VLLM_KV_MAXSEQ`（窗口上界）、`VLLM_XQ_PAGES=1`（旧 `calloc + 0xA5 canary` 布局）、`VLLM_L3_DIAG=1`（L3 evict RSS 采样）、`VLLM_PREFILL_Q8CACHE=1`（P0 原型：写入 f32 行的值改为 q8 反量化）、`VLLM_PFDUMP` / `VLLM_PFDUMPTAG`（prefill 逐行 KV 哈希 + logits 写盘）、`VLLM_PFDBG2` / `VLLM_PFSCRATCH=1`（逐阶段行哈希 / k_pack alloc-free 日志）。

## 六、诚实边界汇总

- **RSS ≠ 物理省到页缓存**：`madvise(DONTNEED)` / 惰性页只保证「进程 RSS 按写入增长」，Linux 页缓存仍可能按压力回收。
- **逐层只支持明文 VQF**（VQF-Enc / 内嵌 SM2 签名被显式拒绝），且与 `VLLM_EW*` 不并存。
- **惰性档无 0xA5 哨兵**，需 `VLLM_XQ_PAGES=1` 恢复调试能力。
- **KV 增长斜率按 q8 默认档**；`--kv-q4` 等改档后斜率不同，需另行测量。
- L3 修复后 r2 之后每轮 ~+52MB 残余为**开放项**。
- 前缀复用（部分 prefill）**数值上必然 ≠ 全量 prefill**：批宽 `%4` 决定走真 GEMM 还是 GEMV 回退，两内核浮点累加序不同；口径为「确定性但近似，同路径可复现」。
- 30B-A3B 在 16 GB 板的绝对速度很低（未优化基线一轮 300 token 上文要 620 s，≈0.45 tok/s），**这是硬件边界而非实现缺陷**。

> **口径说明**：v0（2026-09-05，旧版 GGUF/safetensors 加载逻辑）的冷启动 / 8K 长上下文 / KV 恢复数据 v1 未复测（v1 已移除旧加载路径并引入逐层推理与 KV v2），**不得用于 v1 的任何结论**，也不要与 v1 数字混比。本页数字均非 v0 测点。

## 相关页面

- [Home](Home.md)
- [快速上手](快速上手.md)
- [构建与复现](构建与复现.md)
- [架构总览](架构总览.md)
- [性能与基准](性能与基准.md)
- [优化配置与边界](优化配置与边界.md)
- [权重保护与可验证推理](权重保护与可验证推理.md)
- [常见问题](常见问题.md)
- [术语与数据口径](术语与数据口径.md)

> 源文档：[docs/KV缓存v2-惰性分配与分层驻留方案.md](../docs/KV缓存v2-惰性分配与分层驻留方案.md)
