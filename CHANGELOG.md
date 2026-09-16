# 更新记录（CHANGELOG）

本文按「轮次」倒序记录**代码改动、优化项与验证证据**，供查阅"改了什么、优化了什么、凭什么说有效"。
数据口径与边界以 [README 第 7 节](README.md) / [wiki/性能与基准.md §6](wiki/性能与基准.md) 为准，本文不重复结论细节。

---

## 2026-09-16（下半场）— 30B-A3B MoE 8K 上下文可行性验证（组合⑤）

### 0. 这一轮要回答的问题

**15.9 GB 内存的 RK3588 板，能不能吃下 30B-A3B MoE 模型的 8K 上下文**，
并且逐层驻留 + 组合①（L3 驱逐 × 前缀复用）+ **组合⑤（MoE 专属优化开关）**同时开着。

**结论（先给答案）**：**能。** 峰值 VmHWM **2.92 GB**、RSS **1.09 GB**，对 15.9 GB 板余量充足；
L3 驱逐/回填与进程内前缀复用在 MoE 上同样生效，turn2 靠复用 7212 token 前缀把 852 s 压到 75 s。

### 1. 口径声明（**本档不是基准，不可与 §7 表混比**）

| # | 边界 | 说明 |
|---|---|---|
| 1 | 无对照、无 A/B | 只有 1 轮 × 2 turn，**没有 llama.cpp 对照，也没有同档交错 A/B** ⇒ 绝对值不与 §7 的 2K/4K/8K/16K 权重公平 A/B 同列 |
| 2 | 近似轨未对拍 | 组合⑤ 的 `VLLM_ACTQ=1` 走 q4 MoE int8-dot **近似轨**（日志 `[ACTQ] … approximate track ON`）；本轮**未做 ACTQ 开/关精度对拍** ⇒ 只证明"能跑通、能出内容"，**不构成数值质量结论** |
| 3 | 无 usage 字段 | 两轮响应 `usage={}`（`finish_reason=stop`，正文 126 / 127 字符），生成 token 数取不到 ⇒ **decode t/s 无法精确给出，本档不报 decode 速率** |
| 4 | 预热非算力 | 首次预热（3 段 ≈53 token）耗 **213 s**，是逐层冷读 16.8 GB 权重的代价 |
| 5 | 同进程两轮 | 两轮在同一引擎进程内完成（与 `kv2_run2.sh` 每轮 3 个 turn 的结构一致），turn2 靠**进程内**前缀复用，非重载 |

### 2. 配置

- **必须 `VLLM_VQF_STREAM=1`**：16.45 GB 权重 > 15.9 GB RAM，全层驻留放不下。
- 组合⑤ = `VLLM_ACTQ=1 VLLM_MOE_BATCH=1`（专家激活量化 + 专家批量，仅对 q4 MoE 权重有意义）。
- 其余与 §7 的 8B 档同口径：`OMP_NUM_THREADS=4 VLLM_THREADS=4 VLLM_NPU_FORCE_CPU=1 VLLM_L3_PREFIX_REUSE=1 VLLM_TP_SPIN=1`，
  `--threads 4 --sparse-attn --sparse-k 32 --l3-evict --l3-ratio 0.75 --l3-min-seq 128 --prefill-batch 256`。
- 提示词 600 段（≈7217 token）；生成 `max_tokens=64 temperature=0 top_p=1`。
- **VQF 头未改动**：`patched=0`（`max_seq` 本来就是 8192），`md5_vqf_before=d4e3c3ae…`。
  **不要**对 30B 沿用 16K 档那套 `--set 20480`。
- governor 跑前 `ondemand` → 脚本顶回 `performance`（与 8B 8K 档同口径）；温度 29.6 °C。

### 3. 结果

| 项 | turn1 | turn2（复用 7212 token 前缀） |
|---|---|---|
| 墙钟 | **852.45 s** | **74.69 s**（**11.4×**） |
| `prefill_ms` | 795.3 s（7217 tok） | 23.8 s（只 prefill 新增 **23** token） |
| `prompt` | 7217 | 7235 |
| 响应 | OK / 48098 B | OK / 48198 B |
| RSS / VmHWM | 1.15 GB / **2.67 GB** | 1.09 GB / **2.92 GB** |

prefill 分解（引擎侧 `[PREFILL-TIMING]`，turn1 整段）：
`n=7178 total=781829.9ms | GEMM=523834.2ms(67.0%) ATTN=251126.6ms(32.1%) OTHER=6869.1ms(0.9%)`，
其中 `[PREFILL-KERNELS] GATEUP=459707.5ms（占 GEMM 87.8%）` ⇒ **MoE 专家 GEMM 是主开销**。
（`DOWN=0.0ms` 是 MoE 路径的既有口径，不是缺测。）

### 4. 关键证据行

```
[VQF] MoE v3: experts=128 top_k=8 moe_ffn=768 shared=0 ffn_dim=98304 router=48.0MB (f32 resident) experts=q4
[VQF-STREAM] enabled keep=1 nl=48 segs=11 per-layer=334.1MB data=16847.2MB resident~808.4MB
[ACTQ] VLLM_ACTQ=1: q4 MoE int8-dot approximate track ON (l=0, d=2048 nbG=64)
[L3] evicted 8112 blocks -> /mnt/emmc/moe8k_l3_u65bbfe41 (cursor=158.44 MB, seq=7217, keep=39,
      ratio=0.75, packed_now=8112, total_packed=8112), freed 1267.5 MB from RAM
[L3] restored 8112 prefix blocks from Q4 payload (prefix=7212)
[KV-PREFIX] reuse 7212-token KV prefix, prefill rest
[PREFILL-TIMING] n=23 total=17057.7ms          ← turn2 只算新增 23 token
```

即：16.45 GB 权重在逐层下只常驻 **808 MB**；8K 的 KV 由 L3 驱逐出 **1267 MB** 到盘上
（盘上 cursor 158.44 MB）；turn2 回填 8112 块后直接复用 7212 token 前缀。

### 5. harness 的两处修复（本轮自查发现，已修在归档 harness 内）

| 位置 | 问题 | 处理 |
|---|---|---|
| `moe8k.sh` 头字段解析 | `awk '/dim /{print $3}'` **同时命中 `dim` 与 `head_dim`** ⇒ `DIM="2048\n128"`，随后 `[ "$DIM" -ge 512 ]` 报 `Illegal number` 而**恒假**，守卫失效 | 改为按行首字段名锚定。板端实测：旧式 `OLD_GUARD=fail / Illegal number: 2048`，新式 `NEW_GUARD=pass`。本档 `patched=0`，**结果未受影响** |
| `moe8k.sh` 收尾抽数 | 写的是 `python3 "$H/pk.sh"`，而 `pk.sh` 是 shell 脚本 ⇒ `SyntaxError`，抽数那步空跑 | 改为 `sh "$H/pk.sh"`；数据未丢，本档 `prefill_ms` 为事后单独抽取 |

### 6. 归档

新增 `docs/bench/20260916-rk3588-moe-8k/`：`run.log`（harness 主日志）、`moe_8k_serve.log`（引擎日志）、
`moe_8k_t1.json` / `moe_8k_t2.json`（两轮响应）、`harness/moe8k.sh`（已修版本）、`MANIFEST.txt`（口径 + 复现步骤 + 入库 md5）。
引擎二进制 md5 = `a1b6707de9c925536df980f35aae6a1e`，**与本轮 §7 复测用的是同一份**。

### 7. 已知遗留

- 未做 **ACTQ 开/关精度对拍** —— 这是本档最大的未闭合项（只有它才把"能跑"升级成"能用"）。
- 未测 30B MoE 的 16K（L3 与 KV 体量按 8K 外推风险未知），也未与 llama.cpp 做 30B 对照。
- `/mnt/emmc` 跑前已用 **96%（剩 1.2 GB）**：再跑大档前需先清盘（L3 目录本档占 159 MB）。

---

## 2026-09-16 — L3 长上下文链路：插桩定位 + 三处修复 + 全档重测

### 0. 这一轮要回答的问题

L3 长上下文（冷块驱逐/回填 + 前缀复用）在**复用轮**上明显慢于 llama.cpp。目标：插桩定位**哪个阶段耗时**，
并回答"**是不是压缩/解压缩**造成的"。

**结论（先给答案）**：不是压缩/解压缩。真凶是 **tokenizer 的词表线性扫描** —— 主循环每产一个 token 会调
1~3 次 `find_longest_match_qwen()`，而它对 151,936 条词表逐条 `memcmp`；7.2K prompt 约 **1.1e9 次比较 ≈ 10.4 s**，
且随 prompt 长度线性增长。L3 回填在同档只有 **亚秒~数秒**（2K 实测回填段墙钟 0.630 s）。

### 1. 新增：L3 链路插桩（`VLLM_L3_PROF=1`）

- **文件**：`src/core/vllm_l3.c`、`include/core/vllm_l3.h`
- **开关**：环境变量 `VLLM_L3_PROF=1`。不设时只付**一次分支判断**，零行为改变（同一输入下盘上产物与不插桩时逐字节相同）。
- **打点分三段**：
  - **驱逐**（`l3-evict`）：`init`（fopen/header/calloc）、`layer`→`pack`（Q4 量化）、`q8conv`（q8→f32）、
    `write`（`fseeko`+`fwrite`）、`flush`、`free`（munmap/madvise）
  - **镜像回读**：`scan`、`alloc`、`discard`、`read`、`zero`
  - **回填**（`l3-restore`）：`alloc`（`kv_block_alloc_raw` = mmap）、`fill`→`fetch_mem`、`dequant`、
    `store`（落 f32 + 顺带 q8）、`scratch`
- **每段出口打印**：`wall` / `acct` / `unacct` 三个口径，并给出单位成本（ns/payload、MB/s、块数、clock 读数次数）。
- **宏**：`L3P_T0` / `L3P_ACC` / `L3P_LAP` / `L3P_N` / `l3_prof_on/reset/scope/report`。
- **自检**：`--test-l3` 16 项（x86）/ 17 项（板端，多一项 NEON 对拍）全 PASS；插桩被真实请求触发（含 `[L3] skipped` 分支）。

### 2. 优化 P0：tokenizer 词表索引（复用轮主要残差）

- **文件**：`src/model/vllm_tokenizer_qwen.c`、`include/model/vllm_tokenizer_qwen.h`
- **实现**：开放寻址哈希词表索引，524,288 槽 / 2.0 MB / load factor 0.29。
- **语义保持**：严格复刻旧线性实现的语义 —— 等长匹配取**最小 id**、`lim` 截断、特殊 token 优先、`Ġ`/`Ċ` 变体处理。
- **回退与对照**：旧实现保留为 `find_longest_match_linear()`；`VLLM_TOK_LINEAR=1` 可强制走旧路径做回归对照；
  索引不可用时 `find_exact_token` 自动回退。
- **位级验证**：新增 `tools/bench/tok_ref_check.c`，**1548 用例 0 失配**（含 39 条特殊 token/变体/短边界 + 1500 条随机切片）；
  本地 x86 同段文本加速 **1268×**（索引 0.5 µs/token vs 线性 605.8 µs/token）。
- **收益**：同 prompt 下未归因残差 **7.2K：10.38 s → 0.16 s**、**2.0K：2.74 s → 0.03 s**；
  同会话 A/B 的 turn2 TTFT **6.630 s → 4.009 s（−39.5%）**、turn3 **6.522 s → 3.738 s（−42.7%）**（2K）。
- **零副作用**：turn1 prefill 计算差 0.2%、turn2 GEMM 差 0.4%；3/3 轮输出逐字一致；
  且同 prompt 下新旧二进制写出的 **L3 盘上文件逐字节相同**（md5 `d27d0867169610fa31b57de3a3f48397`）。

### 3. 优化 P1：L3 驱逐整块落盘

- **文件**：`src/core/vllm_l3.c`（`l3_evict_block` 的 f32 路径与 `l3_evict_block_q8`）
- **问题**：原实现按 payload（40 B）逐条 `fseeko`+`fwrite`，一次驱逐产生 347 万次 clock/IO 调用。
- **改法**：block 在文件中本就是**连续**的 `[K area][V area]`，各 `k_area`；改为整块攒满 `2*k_area`（40 KB）后
  **一次 `fseeko` + 一次 `fwrite`**。两条路径（f32 / q8）都改。
- **收益**：`write` **3.478 s → 0.062 s（−98.2%，56×）**；驱逐阶段合计 **4.742 s → 1.271 s（−73%）**；
  插桩 clock 调用 3,470,306 → 10,166（−99.7%）。
- **正确性**：盘上 L3 文件 md5 不变（`d27d0867…`）；`ev_payloads` 计数不变（1,732,608）。

### 4. 优化 P2：L3 回填 NEON 内核

- **文件**：`src/core/vllm_l3.c`（`l3_q4_dequant64`、`l3_store_64`）
- **改法**：新增 NEON 版 `l3_dequant64_neon` / `l3_store64_neon`；**标量版保留**为对照与 x86 路径。
  - 反量化：取高低半字节 → `vzip1q/vzip2q` 还原元素序 → `vsubq_s8(…,8)` → `vmovl_s8/s16` → `vcvtq_f32_s32` → 单次 `vmulq_f32(scale)`
  - 落行：`vmulq` + `vaddq` 两条**独立**指令做 `q = floor(v·ik + 0.5)`，再 clamp 到 `[-128,127]`，最后 `× sk` 落 f32
- **位级红线（关键）**：标量版是 `(int)floorf(v * ik + 0.5f)`，即**乘 + 加两次舍入**。
  NEON 侧**必须禁用 `vmlaq`（FMA）**，否则融合成一次舍入 → 与标量结果不同位。
  且落 f32 时必须用**已 clamp** 后的值乘 `sk`。
- **自检**：新增 `--test-l3` 第 (1b) 项 "P2 restore NEON vs scalar bitwise"：按真实形状构造输入
  （`row64 = q·sk`、`sk = maxk/127`、`ik = 127/maxk`），注入 `0 / ±maxk / ±1e3`（强制饱和）样本，
  对 dequant 与 store 两路各 200 轮 `memcmp`，**0 不一致**（板端 PASS；x86 无 NEON，该项 SKIP）。
- **收益**：`store` **0.665 s → 0.483 s（−27%）**、`dequant` **0.179 s → 0.117 s（−35%）**、
  `fill` 0.862 → 0.616、`restore` 墙钟 **0.848 s → 0.630 s（−26%）**。
- **为什么不是 3~4×**：此时已是**写带宽受限**而非 ALU 受限 —— `store` 每 payload 写 320 B（f32 256 + q8 64）
  = 1.15 GB/s；f32 那一份占写出字节的 **80%**。要再降须**减少写的字节数**（如 `VLLM_KV_NOF32=1`
  或改遍历顺序让目标写连续），而不是继续向量化。

### 5. 仓库与构建修复（与本轮优化无关的既有缺陷）

| 文件 | 问题 | 处理 |
|---|---|---|
| `tools/build/build_x64.ps1` | 源码清单**漏了 `src/core/vllm_ep.c`**，而 `vllm_safetensors.c` 已引用 14 处 `vllm_ep_root_bcast` → x86 必链拉失败（HEAD 即如此） | 补入源码清单 |
| `tools/build/build_x64.ps1`、`check_x64.ps1` | 工作区版本丢了自带 UTF-8 BOM，PowerShell 5.1 会按 ANSI 读中文而乱码 | 补回 BOM |
| 板端 `kv2_run2.sh`（见归档 harness） | 写 md5 缓存前缺 `mkdir -p /tmp/kv2`，导致**三方 md5 从未写进 `summary.txt`** | 补 `mkdir`（本轮已生效：md5 行正常落盘） |
| `docs/bench/20260915-.../MANIFEST.txt` | 复现步骤里 `kv2_run2.sh` 的位置参数顺序写错（曾写作 `full "165 330 600" 3 …`） | 更正为 `<SEGS> <ROUNDS> <TAG>`、`SEGS` 逗号分隔 |

注：本机仓库路径含中文，MinGW `ld` 无法在 `D:\项目\...` 下建输出文件，需用 `-OutDir` 指向纯 ASCII 路径；
这是环境限制，未改脚本默认值。

### 6. 文档与归档

- `README.md` §7 + 结论速览、`README.en.md` §7 + conclusions、`wiki/性能与基准.md` §6：全部换成 2026-09-16 复测数据，
  并**更正上一轮两处归因错误**：
  1. 旧文「`[L3] restored 13176 prefix blocks` = 515 MB、约 28 s（约 20 MB/s，解压受限）」——
     **515 MB 这个数本身没错**（13176 blocks × 40 KB ≈ 515 MB，与本轮斜率一致：1692 blocks = 66.09 MB、
     6120 blocks = 239 MB），**错在把它与 28 s 绑定**：28 s 是那一轮**整轮**的量级，回填远没占满它
     （本轮 2K 实测回填段墙钟 0.630 s，按块数外推 16K 约 5~6 s；写带宽受限，非"解压受限"）。
     由此反推的「约 20 MB/s」随之作废。当时复用轮的大头是 tokenizer 线性扫描（2K ≈2.7 s、7.2K ≈10.4 s、16K 外推 ≈21 s）。
  2. 旧文「唯一还输的一项 = multi-turn 复用轮时延」——修复后该口径下本版全面领先（见下）。
- 归档：新增 `docs/bench/20260916-rk3588-llama-ab/`（2K/4K/8K 原始产物 + `16k-tier/` + `harness/` + `MANIFEST.txt`）；
  `docs/bench/20260915-rk3588-llama-ab/` 为上一轮，原样保留（仅更正其 MANIFEST 的参数顺序笔误）。
- 工具：新增 `tools/bench/patch_vqf_max_seq.py`（VQF 头 `max_seq` 读写，偏移 44，改前备份前 4096 B）、
  `tools/bench/tok_ref_check.c`（tokenizer 位级回归）。

### 7. 本轮验收记录

| 平台 | 构建 | `--test-l3` | `--test-sparse` | 其他 |
|---|---|---|---|---|
| x86 MinGW | rc=0 | 16 PASS + 1 SKIP / **0 FAIL** | 9/9 | `check_x64.ps1` → `X64 CHECK PASSED`；`tok_ref_check` 1548 用例 0 失配 |
| RK3588 板端 | 全量重建 OK | **17/17 PASS** | 9/9 | `--bench-mixed`、`--npu-selftest` 全 PASS |

二进制 md5 留证（同会话 A/B）：原始 `71f8f53e` → tokenizer 索引 `d2ed5155` → P1 `592f5983` → P2 **`a1b6707d`**（本轮最终）。

### 8. 本轮 §7 数据（对 llama.cpp 权重公平 A/B，二进制 `a1b6707d`）

- 档位：2K / 4K 各 3 轮、8K 2 轮、16K 1 轮（后两档按需提前停止）；两侧同会话交错，每 arm 前 `drop_caches`。
- 复用轮时延（turn2/turn3 prefill，越小越好）：本版领先 **2K 1.33~1.50× → 4K 1.92~2.11× → 8K 2.50~2.83× → 16K 3.2~3.4×**，
  幅度随上下文**单调递增**（上一轮 16K 还是落后 1.07×）。
- prefill（增量 t/s）：2K 慢 1.32~1.33× → 4K 慢 1.15× → 8K 反超 1.11~1.12× → 16K 反超 1.60~1.61×。
- decode（t/s）：全层 2K 反超 1.07× → 4K 1.69× → 8K 2.41~2.43× → 16K 3.63×。
- 峰值内存（VmHWM）：16K 时 逐层 **6.71 GB** / 全层 10.15 GB vs llama.cpp 10.77 GB。
- 16K 档 VQF 头自证：改前 `b6d8d1d7…` → `--set 20480` 后 `f8ca1002…`（与上一轮 16K 历史快照逐字节相同，
  反证只动了那 4 字节）→ `--restore` 后 md5 与 `max_seq` 均回到基线。

### 9. 已知遗留 / 未做

- **16K 档跑在 `governor=ondemand` 下**（其余三档 `performance`；板子在这中间重启过），其绝对值可能偏保守约 5~8%。
  该档两侧同 governor，故对照关系有效；是否重跑待定。
- P3（可选）：`pack` 的 NEON 化 —— P1 之后它是驱逐侧唯一剩下的 ALU，但只占 turn1 的约 0.7%。
- 未测：`VLLM_KV_NOF32=1` 对回填耗时的影响（纯配置口径复核，不改代码）。
- 32K / 64K 未测。
- `_release_verify/{gitee,github}` 发布快照**未同步**：两份快照仍是 2026-09-12 世代（`src`+`include` 62 个文件里
  56 个与仓库不同，且**没有 §7 这一节**），重新生成需要发布命名/路径净化的完整规则，待确认后单独处理。
