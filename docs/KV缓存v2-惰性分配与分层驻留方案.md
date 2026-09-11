# KV 缓存 v2：惰性分配与分层驻留方案

> 基于 2026-09-07 板端实测（RK3588 / Qwen3-VL-2B，`--stream-test` harness，32-token prefill + 6-token greedy decode）。
> 模型：`/mnt/emmc/day25_fixed/model.vqf`（2.33GB，签名明文，28 层，max_seq=8192，KV q8 默认档）。
> 本文是 v2 内存方向的实现记录 + 二期迁移清单；纯文本与源码锚点见各节。
> （主副本：`GitHupSRC/docs/KV缓存v2-惰性分配与分层驻留方案.md`）
>
> **v1.0 补充（P3 门）**：L3 驱逐（`--l3-evict`）默认与内存前缀复用**互斥**（L3 释放前缀块后
> prefill 无 NULL 检查）；v1.0 起置环境变量 `VLLM_L3_PREFIX_REUSE=1` 即可让二者**共存**——
> 多轮追问轮 prefill 实测 **−93%~−97%**，为 v1.0 收益最大的单项优化。相关开关与实测台账
> 见 [优化配置与边界说明.md](优化配置与边界说明.md) §七。

---

## 一、背景与目标

v1 KV 缓存的物理预分配方式是"init 全量触页"：

- `st_qwen_inference_init` 无条件为 **f32 + INT8(q8) 双副本 × 28 层 × max_seq 8192** 逐 block `calloc`（`src/model/vllm_safetensors.c`，`alloc_kv_blocks_f32 / alloc_kv_blocks_i8 / alloc_q4_blocks`）。
- `KV_GUARD=4096`（`include/model/vllm_safetensors.h`），`g_sparse_block=32` → 每层 256 个 block；block 小（f32 每块 128KB 以下），`calloc` 的小块清零会 **memset 触碰每一个 payload 页** → init 结束 RSS 即 ~2.46GB（实测）。

同时 AirLLM 型"分层驻留"（`VLLM_VQF_STREAM=N`，见 `src/model/vqf.c` `vqf_stream_layer_advance`）只能约束 **权重文件页**（约 −0.75GB），压不到 KV —— 单层档 RSS 仍停在 ~2.66GB，大头是 KV 底座 ~2.35GB。

**v2 目标**：让 KV 物理占用从"满窗预分配"变成"按实际写入增长"，首 token 启动 KV≈0，1024 token 时 KV ≈ 0.29GB（相对 2.35GB 降幅 >90%），并与分层驻留叠加得到真正的"单层档"小内存下限。

---

## 二、三条杠杆与落地判定

| 杠杆 | 内容 | 判定 | 状态 |
|---|---|---|---|
| #1 惰性分配 | 放弃 init 对 KV payload 的清零触页，mmap 匿名零页、首写自然 fault | 零语义风险，可实现 | ✅ 已落地并验证 |
| #2 去 f32 双备份 | KV 物理只留 int8+scale，删除 1.88GB f32 正典 | 需先迁移注意力/前缀复制/disk-kv/L3，见 §7 | ⏳ 二期 |
| #3 动态窗口 | max_seq 预分配上界 8192 → 可配置（edge 档 1024~2048） | 可实现 | ✅ 已落地（`VLLM_KV_MAXSEQ`） |

---

## 三、#1 惰性 KV 分配（已实现）

### 3.1 为什么 calloc 会全量触页

`cf_pg_alloc → calloc`：Linux glibc 对低于 mmap 阈值（默认 128KB）的请求走堆内清零 —— 写一遍即 fault 一页。KV block 的 payload（f32：`bs×kv_dim×4 = 32×1024×4 = 128KB` 附近，i8：32KB）多数命中"必须显式清零"的分支，于是 init 就把 2.35GB 全部 fault 进 RSS。

### 3.2 机理：mmap 匿名惰性零页

新分配器 `kv_block_alloc_raw / kv_block_free_raw`（`src/model/vllm_safetensors.c`，位于 `g_xq_pages` 声明之后）：

- 默认（`g_kv_lazy=1`，由 init 按 `!g_xq_pages` 置位）：每个 block 用 `mmap(MAP_PRIVATE|MAP_ANONYMOUS)` 分配 `KV_GUARD+payload+KV_GUARD`，**不写 payload、不预写 0xA5 canary**；
- 未写入的 block 读回全 0，与旧 `calloc` 清零语义 **逐字节等价** → 推理输出位级不变（实测 token 全程一致）；
- `g_xq_pages=1`（ASan / 越界调试档）保持旧 `cf_pg_alloc + 0xA5 canary` 布局不变。

配套改动（全部走新分配器并携带 payload 字节数做 `munmap`）：

- 三处 block 分配：`alloc_q4_blocks / alloc_kv_blocks_f32 / alloc_kv_blocks_i8`；
- L3 逐块驱逐后的再分配：`st_qwen_kv_rebuild_freed`；
- 释放：`st_qwen_inference_free` 内 f32/q8/q4 六组循环；
- 新增导出 `st_qwen_kv_free_block(payload, payload_bytes)`（`include/model/vllm_safetensors.h`），main.c 的 L3 on-disk 逐块释放改用它。

### 3.3 canary 探测门控（诚实声明）

decode/prefill 的 KV 写后 `[GUARD]` 探测依赖 block 尾 0xA5 canary（`src/model/vllm_safetensors.c` `decode-k-back/front`、`prefill-k-back/front`）。惰性档不再预写 canary 后，这些探测会恒误报（首字节=0），故在惰性档下以 `!g_kv_lazy` 门控关闭。

**代价**：惰性档（发布默认）不再有"块尾 0xA5 溢出哨兵"；溢出检测改由 mmap 隔离承担（单块独立映射，越界不污染相邻块/堆元数据，落在映射内则越界写只破坏自身 payload，映射外则 SIGSEGV）。需要哨兵调试时用 `VLLM_XQ_PAGES=1`（走旧布局）。ASan 覆盖 KV block 内部越界也需 `VLLM_XQ_PAGES=1`。

### 3.4 为什么 RSS 会按写入增长（数值推导）

每写入 1 个位置：f32 K+V `2×28×1024×4 = 229KB`，i8 K+V `2×28×1024 = 57KB`，合计 **≈280KB/pos**：

- 32 token：≈9MB（实测 B 档 prefill 后总 RSS 222MB，大头是权重视图 ~0.2GB 常驻 + embed/lm）；
- 1024 token：≈0.29GB（相对 2.35GB 降幅 >90%）。

---

## 四、#3 动态窗口（已实现）

`VLLM_KV_MAXSEQ`（`src/model/vllm_safetensors.c` `st_qwen_inference_init`）在既有 `>32768→2048` 兜底之后、`--bench-seqlen` 抬升之前生效：

- 缺省不改变现状（仍随模型 arch，2B=8192），保证既有基准/长上下文回归口径不变；
- edge 部署显式收紧，例如 `VLLM_KV_MAXSEQ=2048 / 512`；
- 惰性分配后该旋钮主要约束可寻址窗口与 `scores_buf` 等一次性缓冲（物理页已按需增长），仍保留用于把"虚拟上界/哨兵容量"锁小。

---

## 五、板端实测

| 档位 | init RSS | prefill(32tok)后 | decode 稳态 | 说明 |
|---|---|---|---|---|
| v1 baseline（calloc 全触页） | ~2.46GB | 3.41GB | 3.41GB | 分层前 |
| v1 + `VLLM_VQF_STREAM=1` | ~2.46GB | 2.66GB | 2.66GB | 分层只压权重，KV 底座不动 |
| A：v2 懒分配（全驻留） | **13.9MB** | ~964MB | ~969MB | init −99.4% |
| B：v2 + `VLLM_VQF_STREAM=1` | 13.8MB | **~222MB** | ~218–222MB | 单层档下限 |
| C：B + `VLLM_KV_MAXSEQ=512` | **7MB** | ~209MB | ~209MB | 窗口旋钮生效（init 打印 max_seq=512） |

decode：A ~214ms/tok；B/C ~266–267ms/tok（与 v1 分层档一致，惰性不引入计算开销）。**六档 TOKIDS 全一致**（`100062 99371 58814 113272 101158 18493`），文本一致，验证语义等价。

复现：

```bash
cd /mnt/emmc/New_vLLM
./vllm_kestrel --stream-test --model /mnt/emmc/day25_fixed --stream-n 6
VLLM_VQF_STREAM=1 ./vllm_kestrel --stream-test --model /mnt/emmc/day25_fixed --stream-n 6
VLLM_VQF_STREAM=1 VLLM_KV_MAXSEQ=512 ./vllm_kestrel --stream-test --model /mnt/emmc/day25_fixed --stream-n 6
```

留档：板 `/tmp/kv2_{a,b,c}.log`、`/tmp/st{0,1,2}.log`、`/tmp/st1_cold.log`；本地 `_work_tmp/kv2_*.log`。

### 5.1 KV v2 方案 serve 评估（2026-09-07，2B·RK3588，f32/nof32/sparse+L3）

请求流（serve，text-only）：短(123tok, max_tokens=8) → 长 msgs_long(1159tok, max_tokens=8) → 三轮追问 msgs_t3(~1202tok, max_tokens=160)。档位：F=默认（f32 正典+q8 影子）、N=`VLLM_KV_NOF32=1`（q8 单份）、FS/NS=F/N+`--sparse-attn --l3-evict --l3-min-seq 128`。

**内存（VmRSS，MB）**

| 档 | load | +131tok | +1167tok | 追问后 | KV 增量 b−a（+1036tok） |
|---|---|---|---|---|---|
| F (f32+q8) | 14 | 1003 | 1331 | 1410 | 327MB ≈ 316KB/tok |
| **N (nof32)** | 14 | 974 | **1069** | **1104** | **95MB ≈ 92KB/tok** |
| FS (f32+sp+L3) | 14 | 1001 | 1375 | 1732 | —（见 L3 探针） |
| NS (nof32+sp+L3) | 14 | 972 | 1307 | 1627 | —（见 L3 探针） |

**推理（PREFILL-TIMING 与 SSE 计时）**

| 指标 | F | N | FS | NS |
|---|---|---|---|---|
| 长 prefill | 50.2 tok/s（1086，73 前缀复用） | 50.9 | 43.6（1159 全量，无复用） | 44.2 |
| 追问 TTFT | 5.78s（复用 1159 + 补 43） | 5.60s | 34.3s（全量 1202） | 34.3s |
| decode 吞吐（~1300 ctx） | 12.8 tok/s | 13.2 | 12.7 | 12.6 |

**L3 棘轮专项探针（同长 msgs_long ×2 连发，全量 prefill，RSS 每轮采样；修复前基线）**

| 档 | rss_r1 | rss_r2 | Δ |
|---|---|---|---|
| F2（dense f32） | 1330536 | 1352288 | **+22MB（平台）** |
| N2（dense nof32） | 1068852 | 1090340 | **+22MB（平台）** |
| FS2（f32+sp+L3） | 1375316 | 1683792 | **+309MB（棘轮）** |
| NS2（nof32+sp+L3） | 1307140 | 1615616 | **+308MB（棘轮）** |

**L3 棘轮修复（2026-09-07 本会话，根因已闭合 + 板端 4 轮复测）**

根因双因：
1. **mirror 按全容量分配 + 轮间不释放（主导，~280MiB/轮）**：`l3_load_to_mem` 原按 `nl×max_blocks×block_bytes`（8192 max_seq 下 293,601,344B ≈ 280MiB）malloc+memset+read——即使只 evict 了 756 块/29.5MB 载荷；且 serve 同用户多轮无任何清理点，下一轮 `l3_state_init` 的 `memset(s,0)` 直接把上轮 mirror 指针丢弃 → **每轮泄漏一个 280MiB 全容量镜像**（与实测 +309MB/轮吻合）。
2. 固定偏移文件布局：块偏移 = `(layer×max_blocks+block)×block_bytes`，layer27/block35 的写偏移推到 ~284MB → extent 化只按"最高写偏移"无法压缩（这是首版修复只省 9MB 的原因）。

修复（`src/core/vllm_l3.c` + `include/core/vllm_l3.h` + `src/main.c`）：
- **紧凑布局**：`STL3State.wcursor` 游标在 `l3_state_init` 置 `L3_HEADER_SIZE`，`l3_evict_block(_q8)` 改为**按驱逐顺序连续分配 disk_off**（文件/RAM mirror 尺寸跟踪真实 evict 载荷而非全 KV 窗）。decode 全部经 `blocks[i].disk_off` 定位（无索引算术消费者），自检 `--test-l3` 9 项 PASS。
- **extent 镜像**：`l3_load_to_mem` 改为扫描 `on_disk` 元数据求最高写端，无 on_disk 时不建镜像。
- **轮间释放**：`l3_evict_after_prefill` 入口先 `l3_state_free`（serve 同用户每轮唯一清理点）。
- `VLLM_L3_DIAG=1` 插桩：evict 入口/出口 + stale mirror 释放点 RSS 采样。

板端 4 轮复测（同 msgs_long，修复版二进制）：

| 档 | rss_r1 | rss_r2 | rss_r3 | rss_r4 | r1→r2 | r2→r4/轮 |
|---|---|---|---|---|---|---|
| F4b（dense 对照） | 1330540 | 1352028 | 1373516 | 1395268 | +21.5MB | **+21.5MB** |
| FS4b（f32+sp+L3） | 1118820 | 1140568 | 1192296 | 1244288 | +21.7MB | **+51.7MB** |
| NS4b（nof32+sp+L3） | 1050668 | 1072416 | 1124144 | 1176136 | +21.7MB | **+51.7MB** |

- mirror：`[L3-DIAG] mirror alloc'd=30965824 B (29.5 MiB)`（原 280MiB → 29.5MiB）；evict 段 RSS **实降** `delta −211676 kB`（enter 1,328,256 → exit 1,116,580）。
- 首轮绝对量较修复前 FS2 降 ~256MB（r1 1,118,820 vs 1,375,316）；棘轮 +309MB/轮 → 前两轮平台级（+21.7MB），**r2 之后每轮 ~+52MB 残余（开放项，见下）**。
- 无崩溃回归；`--test-l3` PASS；客户端 decode 正常（BENCHRES tokens=8）。

**残余 ~52MB/轮 归属（开放项，smaps + 运行时日志证据，k_pack 假设已否定）**：NS 档每轮末全量 `/proc/<pid>/smaps` 快照显示 VmSize 同步增长（r1 3,451,036 → r2 3,486,328 → r3 3,551,864 kB），每轮**新增一个 ~64MB 匿名区（~50MB 已驻留，地址单调上升、前轮区不回收到新复用）**。**但 `VLLM_PFSCRATCH=1` 运行时日志（本会话）否定 k_pack 泄漏**：`st_qwen_model_prefill_batch` 的 k_pack（64MiB）每轮 alloc 后**必在尾部配对 free**，地址跨轮复用（实测 `0x7ebbfff040`×2、`0x7e9ffff040`×2，全部 alloc/free 配对）；静态审计亦确认函数内唯一早退点（`!emb` @~10892）先释放 k_pack。残余收敛口径：r1→r2 = +21.7MB（平台级），r2→r4 每轮 +51.7MB；evict 段 RSS 实降逐轮不一致（同 freed 47MB 逻辑量，r2 降 −48MB、r3 仅降 −18MB）→ 候选机制收敛为 **anon 页返回不对称**（evict munmap 应返回的页与 prefill 重触页在交替轮次不配对；或 rebuild mmap 的零页被 kernel 页缓存/他处截流）。待办：evict 前对将释放块做 **pagemap resident 统计**（验证 munmap 时页是否真驻留）+ ist_reset rebuild 前后 RSS 采样 + 试 `MADV_DONTNEED` 对比。**本轮结论口径**：L3 修复使 serve 多轮 RSS 从"每轮 +309MB 无收益棘轮"降为"前两轮平台级 + 之后 +52MB/轮"，且轮末绝对 RSS 低于 dense 档（冷块真返 OS）；残余 52MB/轮与"逻辑 free ≠ RSS 实降"无关、与 k_pack/镜像泄漏无关，是独立 anon 页/映射现象，记录在案不掩盖。

结论与诚实边界：
- **nof32 达成设计目标**：同阶段 RSS 较 f32 双份档省 227–306MB（−22%）；KV 增量斜率 316→92KB/token（−71%，与 q8 单份理论 56KB/token+分配余量吻合）；prefill/decode 与 f32 档**性能等价**（噪声内）。
- **sparse+L3 的 serve 多轮 RSS 棘轮——已修复（根因与 4 轮复测见上方"L3 棘轮修复"段）**：根因 = L3 mirror 全容量分配（~280MiB/轮）+ 同用户多轮无清理点泄漏；修复后（紧凑布局 + extent 镜像 + 轮间 state_free）棘轮 +309MB/轮 → 前两轮平台级，残余 ~52MB/轮为独立映射泄漏（smaps 已证，开放项）。**对 P4-D 单请求"freed 236MB"口径的修正（保留）**：那是逻辑值；修复后轮末 RSS **低于 dense**（FS4b r1 1,118,820 vs F4b 1,330,540 kB，冷块真返 OS）。另：L3 与前缀复用互斥（`ist_reset` 门），追问 TTFT 34s vs dense 5.7s 即该取舍的直接体现。

### 5.2 多模态 20 轮图文对话综合评估（2026-09-07，F 全量基线 vs NS 组合档，3 轮取中位）

语料：20 轮 user（每轮带同一 64×64 蓝色 data-URL 图 + 属性/记忆问题）+ 19 轮 assistant 标答历史 + 末轮 user 触发生成（max_tokens=192）；上下文 772 tok/请求（`[L3] seq=772`）；同一请求 ×3 轮取中位；F=默认 dense 全量加载，NS=`VLLM_KV_NOF32=1`+sparse+L3。素材：`/tmp/msgs_20r.json`（板）、`_work_tmp/gen_20r.py`（本地生成器）。

| 维度 | F（dense 基线） | NS（nof32+sparse+L3） | 结论 |
|---|---|---|---|
| 功能 | 6/6 请求成功；20 图 VIS-CACHE hit（首图首轮 miss encode，余命中）；无 MEDIA/解码失败 | 同左 | 两档多模态图文 E2E 均通 |
| prefill/TTFT（中位） | 22.10 s | 23.36 s | NS +5.7% |
| wall（中位） | 30.44 s | 33.40 s | NS +9.7% |
| decode 吞吐（中位） | 15.72 tok/s | 13.46 tok/s | NS −14.4%（sparse+L3 镜像 Q4 反量化 + 选块真实代价，如实标注） |
| 内存 VmRSS r1/r2/r3（kB） | 1,944,396/1,967,972/1,991,548 | 1,728,088/1,751,664/1,775,240 | **NS 每轮 −216,308（−11.1%）** |
| 每轮 RSS 增量 | +23,576 | +23,576 | 同平台级（本 772-tok 场景 NS 未现 +52MB 残余） |
| VmSize | ~5.4–5.5 GB | ~3.4–3.6 GB | nof32 不分配 f32 正典数组（VA −1.9GB） |
| 输出确定性 | 3 轮逐字节一致 | 3 轮逐字节一致 | 两档各自确定 |
| 输出质量（7 点核验） | 6/7 命中（蓝/冷/无文字/中亮/64×64/科技风 ✓；**轮数答 12 ✗**） | 6/7 命中（同前 ✓；**轮数答 20 ✓**） | 属性主体两档一致正确；长上下文细节计数任务上档间数值差异产生**内容级分叉**（F 错 NS 对） |

关键记录：
- **多模态 serve 前缀复用不生效**：F 档 3 轮均全量 prefill（log 无 `KV-PREFIX`），故本评估 F 基线不含复用收益；F/NS 的 TTFT 差异纯来自 nof32+sparse+L3 计算开销（多模态路径下两者都全量，对照干净）。
- **档间"确定性但近似"在多模态同样成立**：F≠NS 输出逐字节不同（内核批宽/精度差，与 §7.2 P4-C② 同源），且本轮展示出内容级影响（轮数 12 vs 20）；建议对外口径统一为"确定性但近似，语义级稳健，细节计数类长任务可能分档差异"。
- NS 多模态（772 tok）r1→r3 每轮 +23.6MB = 平台级，未复现纯文本 1159+ 场景的 r2→r4 +52MB 残余（残余与上下文长度/轮次相关，记录不掩盖）。
- load 采样点为 health 后 3s（模型 mmap 惰性未触页，~14MB 不代表稳态），内存对比以 r1..r3 稳态为准。
- N/FS 单拆解档未纳入本轮（时长预算），由 F/NS 端到端推断贡献。
- 板端留档：`/tmp/eval20r/{results.txt,F.log,NS.log,F_r{1,2,3}.txt,NS_r{1,2,3}.txt}`；本地 `_work_tmp/ev_*.txt`、`msgs_20r.json`、`eval_client.py`、`eval20r.sh`。

---

## 六、诚实边界

1. **RSS ≠ 物理省到页缓存**：`madvise(DONTNEED)`/惰性页只保证"进程 RSS 按写入增长"；Linux 页缓存仍可能按压力回收（vfat 上 `drop_caches` 实测几乎无效，本环境 COLD 档仅 ~+40% 慢，不能当作 eMMC 冷读最坏值，见分层方案档）。
2. **f32 仍是正典存储**：惰性只改"何时物理占用"，未删任何 buffer（删 = 二期 §7）。
3. **canary 调试能力降级**：惰性档无 0xA5 哨兵（§3.3），需 `VLLM_XQ_PAGES=1` 恢复。
4. **KV 增长斜率按 q8 默认档**：`--kv-q4` 等改档后斜率不同，需另行测量。
5. **未写 block 保证为 0** 依赖 mmap 匿名零页语义；任何依赖"KV 初始垃圾值"的路径不存在（写后读），故等价成立，但仍按 §5 的 TOKIDS 全档一致作为回归红线。

---

## 七、二期：#2 "去 f32 正典"迁移动线（未实施，规划）

### 7.1 现状：为什么不能直接删

f32 KV 是全引擎**读写正典**，不是冗余：

| 读/写点 | 位置（vllm_safetensors.c 符号） | 用途 |
|---|---|---|
| 写 f32 行 | decode `st_qwen_model_forward`、prefill `st_qwen_model_prefill_batch`、multimodal `st_qwen_model_multimodal_prefill_ex` 的 `memcpy(kv_row_f32(...))` | 每一 token 的 K/V 都落 f32 |
| decode 注意力 | `use_q8 ? INT8-flash : f32` 两分支；f32 分支（非 q8、含 fallback）与 tile-packed flash 读 f32 | 主注意力路径之一 |
| 前缀复用 | `st_qwen_copy_kv_prefix` 逐 block `memcpy` f32 行（同时复制 q8） | 跨请求前缀 KV 复制 |
| L3 驱逐 | main.c `l3_evict_layer(... st->k_cache[l] ...)` 以 f32 块为单元扫描/评估 | 重要性评估读 f32 |
| disk-kv | `st_qwen_kv_disk_save/load` fwrite/fread f32 行 | 快照/恢复 |
| 稀疏注意力 | `sparse_attn_head` 体系基于 f32 块 + L3 q4 落盘镜像 | 长上下文 |

### 7.2 分步迁移（每步以 bit-identical / PPL 门禁收口）

**P0 原型实测（2026-09-07，2B，第二轮 15-token prefill + 34-token forced-NLL）**：

- 开关：`VLLM_PREFILL_Q8CACHE=1`（见 `src/model/vllm_safetensors.c` `vllm_pf_q8cache_env / kv_dequant_roundtrip`）——把写入 f32 正典行的值改为 q8 反量化（`q*scale/127`，与 `kv_quantize_per_head` 同路径），模拟"只存 q8"后 prefill 历史段将读到的值；`VLLM_STREAM_NLL=1` 在 harness 里做"第二轮 prefill + 固定参考句 forced-NLL"。
- 结果：f32 精确历史 **NLL=2.8212**（≈PPL 16.8）→ q8 反量化历史 **NLL=2.9376**（≈PPL 18.9），ΔNLL≈+0.116（+4.1%，PPL 口径 +12.5%）。greedy 前 6 token 文本两档一致（`逐层加载推理是一种在`）。
- 说明：该原型是**上界**——它连"本批自注意"也一并走了反量化；真实设计若只对"历史段"走 q8、本批保持精确，delta 会更小。单样本不足以定门槛，需固定多段语料 + 阈值后再进 P1。

**多段语料 NLL 基线（4 段中文本 × turn1+turn2 prefill × 89 token forced-NLL，同开关）**：

| 语料 | f32 精确历史 (OFF) | q8 反量化历史 (ON) | Δ |
|---|---|---|---|
| P0 | 3.7035 | 3.7932 | +0.090 |
| P1 | 3.8379 | 3.7231 | −0.115 |
| P2 | 4.9831 | —（见 CORPUS） | — |
| P3 | 2.5877 | 2.4213 | −0.166 |
| **CORPUS 均值** | **3.7267** | **3.6474** | **−0.079** |

- 判读：聚合 ΔNLL ≈ **−0.079（约 −2%，方向反而略优）**，逐段散度 ±~5%、无系统性劣化 → q8 历史段对 prefill 质量无可测系统代价（与 q8-KV"近无损"口径一致）。
- **门槛（P1/P3 放行标准）**：`ΔNLL_ON−OFF ≤ +0.10` 视为通过；本语料 −0.079 **PASS**。注意 n=89 偏小、参考句为人工编写，绝对值（PPL 偏高）不代表通用集；后续增容语料再复核。
- **P1a（已实现，板端构建过，serve 前缀回归待做）**：`st_qwen_copy_kv_prefix` 在 q8 世界（`VLLM_PREFILL_Q8CACHE=1`）下不复制 src 精确 f32 行，改为 q8+scale 反量化重建 dst f32 正典行 → 复制前缀与未复制前缀同口径。**COPYTEST 回归（2026-09-07 会话）**：OFF `A_NLL=B_NLL=3.703504563`、ON `=3.793169508`，diff=0 均 PASS（同一世界内逐位一致）。
- **P1b（已实现，2026-09-07 会话实测通过）**：`st_qwen_kv_disk_save/load` 增加 **v2 快照格式**（`DKV_VERSION 2`，q8 行 + per-head f32 scale，无 f32 依赖）；v1 f32 旧快照兼容读。DKVTEST 断言：恢复行的 q8 字节与 scale 与源**逐字节一致**（rows_equal），恢复后再 decode 一个 token 的 logits 与源**逐位一致**（logits_equal）。实测（2B，108 行快照 = m1+m2 前缀 + forced-NLL 追加的 ref token）：

  | 档 | COPYTEST (diff) | DKVTEST |
  |---|---|---|
  | OFF（f32 精确正典） | 0.000e+00 PASS | save=0 load=0 rows=108 rows_equal=PASS logits_equal=PASS |
  | ON（q8 世界） | 0.000e+00 PASS | 同上 PASS |

  说明：decode/prefill 历史注意力在 `use_kv_q8=1`（默认）时**只读 q8+scale**，f32 正典仅被批量 packed 注意力（`st_pack_kv_heads`）作为源；因此 v2 恢复在 q8 语义下逐位保真。harness 初版以 `tot=m1+m2` 行快照、忽略 NLL 已把状态前推 rn-1 行 → decode 对比起点不等恒 FAIL（已修正为 `cache_len` 实驻行数快照，见 `src/main.c` DKVTEST 区注释）。

- **P0 注意力主路径量化**：使 decode+prefill 注意力在 q8 档恒定只读 int8+per-head f32 scale（scale 已按 `[max_seq×nkv]` 存在 `k_scale/v_scale`），消灭"仍写/读 f32 行做注意力的路径"。验证：默认 q8 档 TOKIDS 与现版逐位一致（**注意：仅"单批/首轮、无历史缓存"可逐位；含历史段的 prefill 必然产生数值变化，须走上面 PPL/NLL 门禁**）。
- **P1 派生路径**：`st_qwen_copy_kv_prefix` 增加纯 q8 复制模式（f32 分支仅 fallback）；`st_qwen_kv_disk_save/load` 增加 q8+scale 快照格式（disk-kv 格式变更需兼容旧快照或标注格式 v2）。验证：多轮前缀复用与跨进程恢复文本一致。
- **P2（已实现，2026-09-07 会话实测）**：L3/sparse 的 f32 依赖收口（nof32 边界项关闭）：
  - decode/prefill 的 sparse 分派不再因 nof32 回退（去掉 `!g_kv_nof32` 门）；`sparse_attn_head` 在 f32 正典不存在（`k_cache==NULL`）时：probe 改为**逐元素 q8 反量化**（`q*(scale/127)`，与 f32 行=dequant 位级一致）再走与 f32 相同的 FMA；scoring/V 各分支按"块级 RAM 源"判定（f32 世界看 `k_cache[b]`，nof32 世界看 `k_q8[b]`），L3 取块路径 NULL 安全。
  - **L3 驱逐新增 q8 源变体** `l3_evict_layer_q8 / l3_evict_block_q8`（vllm_l3.c/.h）：无 f32 时按 q8+scale 逐 64 组反量化打包 Q4 镜像（镜像内容与 f32 行=dequant 时一致）；`l3_evict_after_prefill`（main.c）按 `k_cache[l]` 存在性分派 f32/q8 变体，释放循环对 NULL k_cache 免疫（nof32 只释放 q8 块）；排序逻辑抽公共 `l3_pick_evict_blocks`（原 f32 路径语义不变）。
  - **门禁实测（sparse ON，4 段语料）**：`PREFILL_Q8CACHE=1` vs `KV_NOF32=1` 各段 NLL 逐位一致——P0 3.738499 / P1 3.684517 / P2 4.748005 / P3 2.397819，**CORPUS_NLL_MEAN 均 = 3.586151**（nof32 无 f32 数组 + q8 probe/打包 ≡ f32 行载 q8 dequant 档）。
  - **边界（诚实标注，更新）**：L3 q8 变体与 nof32+sparse 组合的 **serve/长上下文端到端实测放 P4**（本会话以语料 harness 验证，未走 serve/`--l3-evict` 全链路）；DKV v1（f32）快照在 nof32 下 `load` 拒绝（rc=-1）不变；`--test-sparse` 合成 selftest 在 case5（`st_attn_batched_packed*`，P2 未触碰的代码）SIGBUS/SIGSEGV——**疑似既有问题**（今日快照因头文件 API 漂移未能直接复编对照，未闭合），记 P4 以符号构建收口。
- **P3（已实现，2026-09-07 会话实测）**：新开关 **`VLLM_KV_NOF32=1`**（默认关，实验档）→ q8 档不再分配 f32 正典 KV 数组（init 跳过 `alloc_kv_blocks_f32`，k_cache[l]/v_cache[l] 保持 NULL）；四个存储点（decode/forward_batch/prefill_batch/multimodal）跳过 f32 行写与 P0 dequant；prefill 批量注意力改为 **pack 直接从 q8+scale 反量化填 f32 scratch**（与 `kv_dequant_roundtrip` 同一口径 `q*(scale/127)`，位级一致）；copy/rebuild/DKV v1 读 v1 等按 f32 存在性门控。门禁矩阵（2B，COPYTEST+DKVTEST）：

  | 档 | COPYTEST NLL (n=20) | DKVTEST |
  |---|---|---|
  | OFF（基线 f32 精确） | 3.703504563 diff=0 PASS | rows=108 rows_equal=PASS logits_equal=PASS |
  | ON（q8 世界，PREFILL_Q8CACHE） | 3.793169508 diff=0 PASS | 同上 PASS |
  | **nof32（VLLM_KV_NOF32，无 f32 数组）** | **3.793169508 diff=0 PASS（与 ON 逐位一致）** | 同上 PASS |

  → **nof32 语义 ≡ ON（q8 反量化）世界**：pack 从 q8 反量化与 dequant 写入 f32 行的结果逐位相同；相对 OFF（精确 f32）的质量差即 P0 门禁已测的 ΔNLL≈+0.09（单参考句）/<1 语料均值 −0.079（PASS ≤ +0.10 门槛）。RSS 实测（同型单状态 run，含惰性页/噪声）：init 13776→13268kB（VA 窗口释放，RSS 惰性不触页）；32-token prefill 后 OFF 968,208 vs nof32 956,552kB（−11.7MB）；decode 至 ~70 行后 rss_end OFF 979,400 vs nof32 963,148kB（**−16.3MB**）→ 经验斜率 ≈230kB/行（与 f32 K+V 行 229kB/行理论吻合）→ 按此外推 2K 上下文 ≈−0.47GB、8K ≈−1.9GB（**计算外推，长上下文/serve 端到端实测放 P4**）。
  **边界（诚实标注）**：① nof32 仅 q8 档 + 惰性分配有效（`g_xq_pages=1` ASan 越界调试档自动退回双份）；② sparse 在 nof32 下已可用（P2 收口：decode probe 走 q8 反量化、scoring/V 走 q8/l3），L3 驱逐 q8 镜像变体已实现但 **serve/长上下文全链路实测留 P4**；③ DKV v1（f32）旧快照在 nof32 下 `load` 拒绝（rc=-1）；④ 多模态/text 存储点同构门控、注意力在 q8 档纯 int8（未受 nof32 影响）；**默认档多模态单图 E2E 已在 P4 复核通过（serve image_url 输出断言"蓝色"）；nof32 档多模态端到端仍未真机跑过**（边界收窄，见 P4 节）。
- **P4（已收口，2026-09-07 会话板端实测）**：serve 级全量回归 + 两处修复 + 一处位级精确收口。§5 数字无需更新（P4 未重测 RSS）。
  - **P4-A：`--test-sparse` case5 崩溃修复（符号构建定位）**：case5 合成打包路径的 packed worker（`vllm_attn_batched_worker`）每 head 写 **4 行** scores（`ha*4+0..3`，stride=`score_stride`），而 `sc_buf` 原按 `nh*seq_stride` 分配 → 堆越界（SIGBUS/SEGV）。改为 `(size_t)nh*4*seq_stride`（`src/model/vllm_safetensors.c` 9125 行注释）。修复后板端 `--test-sparse` → `Sparse self-test PASSED (9 checks)`；**P4-B**：ASan（`build-asan`，Debug + `-fsanitize=address -O1 -g -mcpu=cortex-a76 -march=armv8.2-a+dotprod+fp16`）同档 `--test-l3`/`--test-sparse` 均 rc=0，日志 0 条 AddressSanitizer/runtime error。
  - **P4-C①：serve disk-kv 索引拒 v2 快照（修复）**：P1b 后默认（`use_kv_q8`）快照写 **v2（q8+scale）** 头，而 `vllm_server.c` 的扫描 `dkv_read_entry` 仍硬编码 `ver==1` → v2 checkpoint 全被拒 → `--disk-kv` 在默认档**永不命中恢复**（日志 `index: 0 checkpoint(s)`，此前 C2/F2 首轮即此现象）。修复为 `ver==1 || ver==2`（v1/v2 头与 token 段布局一致，载荷差异由 `st_kv_disk_load` 按版本处理）。修复后跨进程恢复全链路：`index: 1 checkpoint(s)` → `[KV-DISK] loaded 139-token KV prefix` → `reuse 139 + prefill rest`。
  - **P4-C②：serve 前缀复用 ≠ 全量重算——根因已闭合（2026-09-07 会话对拍）**：三次确定性对照（temp=0，同输入 258-token 双轮）：全量 prefill 跨进程两次**逐字节一致**（Y1==Y2）；RAM 前缀复用跨进程两次**逐字节一致**（X1==X2）；但 **复用档 ≠ 全量档**（对抗用例分叉于生成第 ~60 token：复用档复述上轮回答 427B、全量档正确转进追问 436B；q8 disk 恢复档在此输入上更出现 `!`×96 退化）。曾假设差异来自"复用 decode 阶段写入的 KV 行 vs 重 prefill（批量 GEMM）"并把复用边界截断到纯 prefill 行（139→123）重测——输出逐字节不变、仍 ≠ 全量 → 机制证伪并回退（保留全 LCP 复用）。**逐层对拍（本会话，VLLM_PFDUMP 哈希 dump 钩子 + 跨进程复测）已定位机制**：
    - 同形 prefill **跨进程逐位一致**（A1==A2：258-token 全量两遍，28 层×258 行 6 视图 + 末行 logits 全零差异）。
    - 不同批宽下同一 (token,pos) 的 KV 行**全部不一致**：258 批 vs 123 批（token 0..122，0 差异起于 **pos 0**）、全量 vs 复用部分 prefill（base=131,n=127）均**逐行逐层**在 f32/q8/scale 三视图上哈希不同（123 批：3444/3444 行全异；部分批：3554/3554 行全异）——即**引擎批式 prefill 非批宽不变**（候选机制：激活量化/GEMM 归约按批宽变化），部分 prefill 与全量 prefill 对同一 token 无法位级一致。
    - 结论：前缀复用（部分 prefill）**在数值上必然 ≠ 全量 prefill**，分叉不是缓存/簿记 bug 而是批宽依赖数值差异经贪心放大的确定性结果；"确定性但近似"口径升级为**机制级**（宣传可引用：复用=确定性但数值近似，同路径可复现）。**内核分派级闭合（本会话二探，VLLM_PFDBG2 layer0 逐阶段哈希 + q8x4_gemm_batched 分派确认）**：
      - stage 哈希（n258 vs n123，layer0 首块 row0）：**emb0、nrm0 逐位一致**（嵌入与 RMSNorm 行局部性确认），**qq0/qk0/kt0 起分叉** → 差异进入点 = **批量 QKV GEMM**，不是嵌入/归一化。
      - 分派机制：`q8x4_gemm_batched`（safetensors.c ~4316）仅在 `(n_batch&3)==0 && repack 成功` 时走真 GEMM `st_gemm_q8_0_4x4_batched`；**否则整批回退 per-token q8x4 GEMV**（注释：舍入顺序与真 GEMM 不同）。A(nb=256) 走真 GEMM；B 部分 prefill（nb=123/127）`%4≠0` 整批走 GEMV 回退 → 同一 (token,pos) 位不同。即**批宽 %4 决定内核、两内核浮点累加序不同**——任何 %4≠0 的部分 prefill 与 %4==0 的全量必然位不同（含 row0，与"pos0 起全行差异"观测一致）。
    - 幅度：末行 logits idx0 Δ≈0.9（本输入，pf dump 直接对拍）；逐行 f32 值差未逐字节量化（哈希只证异/同）——归因于内核精度差经 28 层累放。**宣传口径（如实标注）**：前缀复用=确定性但近似（同路径可复现，非全量重算的位级克隆），语义级稳健，long-tail 边缘输入贪心可分叉。彻底位级收敛的前置 = 让 GEMV 回退与真 GEMM 同序（改内核而非复用语义），属性能数值工程，非正确性缺陷。
  - **P4-D：nof32+sparse+L3 serve 长上下文端到端（1159-token 中文，`--sparse-attn --l3-evict --l3-min-seq 128`）**：f32 档 prefill 26.90s，`[L3] evicted 756 blocks -> kv_l3_u65bbfe41.bin (29.5 MB, seq=1159, ratio=0.75), freed 236.2 MB`；nof32 档 26.48s，日志 `[KV] P3 nof32 mode: f32 canonical KV arrays NOT allocated`，L3 驱逐 freed 47.2 MB（q8 档无 f32 正典的直接省内存收益：q8 行 ~45KB/token vs f32 行 ~230KB/token）→ **nof32 q8 镜像在 L3 全链路可用**。**诚实边界**：f32 与 nof32 两档贪心输出在 ~char18 处分叉（sparse 分块在近并列时的跨档选择差异；P2/P3 的 NLL 门禁只覆盖 evict 前的 prefill logits，未覆盖 evict 后 decode 的分块选择）——记录为待逐层对拍项，非功能缺陷。
  - **P4-E：serve 单图 chat E2E 复核（multimodal）**：64×64 纯色（RGB8 有效 PNG）经 `image_url` data URL 上传 → decode → ViT encode → 模型回答"这张图片的主色调是蓝色。"。负面用例：首测素材曾用 filter 字节放错位置（每像素 4 字节行）的 PNG（zlib 可解、CRC 合法但行宽 256≠193 非法）→ stb_image 合法拒绝（`[MEDIA] Failed to decode image payload`）→ 修正素材后通过；此前的 image_url 实测以缓存/时延测量为主，本次对**输出文本显式断言**。

### 7.3 风险与红线

- 每次删 f32 读点都必须保留一个对照（同条件 TOKIDS 或 PPL）；不能"删了再修"。
- q8 是 per-head 量化（scale 逐 head），任何迁移必须保持与 `kv_quantize_per_head` 同一量化路径，禁止另起新量化（位级漂移红线）。
- `--kv-q4` / 稀疏互斥与 disk-kv 旧格式是兼容性雷区，须单独列回归用例。

---

## 八、本次已落变更清单

| 文件 | 变更 |
|---|---|
| `include/model/vllm_safetensors.h` | 新增导出 `st_qwen_kv_free_block` 声明 |
| `src/model/vllm_safetensors.c` | `g_kv_lazy` 置位；`kv_block_alloc_raw/free`；三处 block 分配、rebuild、六组释放改惰性；`[GUARD]` 探测门控；`VLLM_KV_MAXSEQ` 旋钮 |
| `src/main.c` | L3 on-disk 逐块释放改 `st_qwen_kv_free_block`（munmap 语义） |
| `src/model/vllm_safetensors.c` | （P1b/P3，本会话）DKV 存储 v2 格式（`DKV_VERSION 2`：逐 token K/V i8 行 + per-head f32 scale；v1 f32 兼容读）；`g_kv_nof32` + `VLLM_KV_NOF32`（P3）：init 跳过 f32 正典块分配、四存储点按 f32 存在性门控、`st_pack_kv_heads` 增 q8+scale 反量化源（pack worker 位级 = `kv_dequant_roundtrip`）、copy/rebuild/DKV-v1-load/sparse/CANARY/DUMP 门控；`st_qwen_kv_rebuild_freed` 对 NULL k_cache[l] 免疫 |
| `src/model/vllm_safetensors.c` | （P2，本会话）sparse 分派去 nof32 门（decode/prefill 均可）；`sparse_attn_head` f32 NULL 安全：probe 增 q8 逐元素反量化分支（位级 = dequant 行）、scoring/V 按块级 RAM 源判定（f32/q8/l3） |
| `src/core/vllm_l3.c` + `include/core/vllm_l3.h` | （P2，本会话）驱逐排序抽公共 `l3_pick_evict_blocks`；新增 q8 源变体 `l3_evict_block_q8/l3_evict_layer_q8`（无 f32 时按 q8+scale 逐 64 组反量化打包 Q4 镜像） |
| `src/main.c` | （P2，本会话）`l3_evict_after_prefill` 按 `k_cache[l]` 存在性分派 f32/q8 驱逐变体；释放循环 NULL 免疫（nof32 只释放 q8 块） |
| `src/main.c` | （P1b，本会话）DKVTEST harness：快照改为 `istA.cache_len` 实驻行数（含 forced-NLL 追加的 ref 行），rows_equal（q8 字节+scale）+ 同位置 decode logits_equal 双断言 |
| `src/model/vllm_safetensors.c` | （P4，本会话）`st_test_sparse_attn` case5 `sc_buf` 分配 `nh*seq_stride` → `nh*4*seq_stride`（packed worker 每 head 4 行 scores，堆越界修复，`--test-sparse` 9 checks PASS） |
| `src/serve/vllm_server.c` | （P4，本会话）`dkv_read_entry` 扫描接受 v2 快照（`ver==1||ver==2`，修 `--disk-kv` 默认档恢复永不命中；v1/v2 头与 token 段布局一致，载荷差异由 `st_kv_disk_load` 按版本处理） |
| `src/core/vllm_l3.c` + `include/core/vllm_l3.h` | （L3 棘轮修复，本会话）`STL3State.wcursor` 紧凑布局（disk_off 按驱逐顺序连续分配）；`l3_load_to_mem` 改按 on_disk 实际 extent 建镜像（280MiB→~实际载荷），无 on_disk 不建；`l3_state_init` 置 `wcursor=L3_HEADER_SIZE` |
| `src/main.c` | （L3 棘轮修复，本会话）`l3_evict_after_prefill` 入口先 `l3_state_free`（serve 同用户多轮唯一清理点，消除每轮泄漏一个全容量 mirror）；`VLLM_L3_DIAG=1` 插桩（evict 入口/出口 + stale 释放点 VmRSS 采样） |
| `src/model/vllm_safetensors.c` | （前缀复用对拍工具，本会话）`VLLM_PFDUMP=<dir>/VLLM_PFDUMPTAG=<tag>`：prefill 尾部逐行 KV 哈希（f32/q8/scale 六视图）+ 末行 logits 写盘（env 门控，默认关） |
| `src/model/vllm_safetensors.c` | （内核分派二分工具 + k_pack 追踪，本会话）`VLLM_PFDBG2=<path>`：layer0/首块逐阶段行哈希（emb0/nrm0/qq0/qk0/kt0）；`VLLM_PFSCRATCH=1`：prefill k_pack alloc/free 地址日志（均 env 门控，默认关） |
| `include/model/vqf.h` + `src/model/vqf.c` | （此前落地）分层驻留 `vqf_stream_*`：层切片表 + madvise/fadvise + RSS 采样 + 加密拒绝 |
| `src/main.c` | （此前落地）`--stream-test` 验证 harness |

> 说明：惰性分配只改变物理驻留时机，不改变任何数值语义；分层驻留见其独立实现注释与本文 §5 交叉引用。
