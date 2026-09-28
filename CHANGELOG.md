# 更新记录（CHANGELOG）

本文按「轮次」倒序记录**代码改动、优化项与验证证据**，供查阅"改了什么、优化了什么、凭什么说有效"。
数据口径与边界以 [README 第 7 节](README.md) / [wiki/性能与基准.md §6](wiki/性能与基准.md) 为准，本文不重复结论细节。

---

## 2026-09-28（发布面收尾）— 双远端全新拉取复验（x86-64 + aarch64）、修掉 `check_x64` 假通过、补 `.sh` 执行位

**这一轮要回答的问题**：两个远端（Gitee `master` / GitHub 门面 `gh`）上现在这份代码，
**全新克隆之后能不能原样编译、能不能跑、是否与本机开发树一致**？

**结论（先给答案）**：

1. **能，且三方一致**。x86-64 与 aarch64 各自「全新拉取 → 原生构建 → 无模型确定性自检」全部通过：

   | 侧 | 源码身份 | 构建 | `--test-l3` | `--test-sparse` |
   |---|---|---|---|---|
   | 本机 master 工作树（x86-64） | `9453871` / tree `4999e05a…` | rc=0 | 16 PASS / 0 FAIL / 1 SKIP | 9 PASS / 0 FAIL |
   | Gitee 全新克隆（x86-64） | 同上（tree 逐位相同） | rc=0 | 同上 | 同上 |
   | GitHub 全新克隆（x86-64） | `f332e19` / tree `2f2003e5…` = 本机 `gh` | rc=0 | 同上 | 同上 |
   | Gitee 全新克隆（aarch64 板端） | `9453871` / tree `4999e05a…` | rc=0 | **17 PASS / 0 FAIL / 0 SKIP** | 9 PASS / 0 FAIL |
   | GitHub 归档（aarch64 板端，codeload） | `gh` 门面树 | rc=0 | 同上 | 同上 |

   自检输出 SHA256：x86 三方**逐字节相同**；aarch64 两侧**逐字节相同**。ARM 比 x86 多出的那一项是
   `P2 restore NEON vs scalar bitwise`（x86 无 NEON 打 SKIP，ARM 实跑 PASS）。

2. **x86 与 aarch64 的差异只有三类，全部可解释**：`[DEV]` 设备画像行、上述 NEON 项（SKIP↔PASS）、
   以及 NEON 与标量内核的 1e-7 量级浮点差（`q4 dot rel 0.00e+00` ↔ `3.99e-07`；
   `prefill sparse == exact` max diff `3.576e-07` ↔ `3.874e-07`）。**没有行为差异。**

3. **二进制哈希不能当跨平台 / 跨路径口径**。aarch64 两侧 exe 差 73,092 字节，按段定位后根因是
   `.rodata` 里内嵌了一条**绝对源码路径**（`…/stb_image.h`，两侧检出目录名长度差 19 字符）：
   `.rodata` 因此大 24 字节，其后所有段偏移后移（86.9% 的"差异"落在 `.eh_frame`）；
   `.got` / `.init_array` **逐字节相同**，`.text` 仅 367 字节变（引用 rodata 的地址位移）。
   **同路径重建两次 sha256 完全相同**，证明构建本身是确定性的。
   对照：x86 侧只差 **2 字节**（PE `TimeDateStamp` 与其派生 `CheckSum`），因为 `build_x64.ps1`
   用**相对路径**调 gcc，`__FILE__` 不含路径；板端走 CMake，传的是绝对路径。

### 1. 改动清单

| 文件 | 改动 |
|---|---|
| `tools/build/build_x64.ps1` | 编译前 `Remove-Item $out`；**补回 UTF-8 BOM**（文件头写着"必须以 UTF-8 BOM 保存"，实际缺） |
| `tools/build/check_x64.ps1` | 记录构建开始时刻 `$t0`，追加「产物 mtime ≥ `$t0`」断言 |
| 15 个 `.sh` | `git update-index --chmod=+x`（100644 → 100755） |
| `wiki/构建与复现.md` | 已知坑补三条：非 ASCII 路径（补"已加防线"）、`.sh` 无执行位（已修）、板端直连 GitHub 不可达（走 codeload） |

### 2. 两个必须记录的缺陷

**① `check_x64.ps1` 会假通过（已修）**。它原先只判 `Test-Path $exe`。本次复验时本机 `rc=1`
（非 ASCII 输出路径导致 `ld` 链接失败），脚本仍报 `X64 CHECK PASSED` 并跑完全套自检 ——
跑的是 **2026-09-25 19:08 的旧 exe**（1,369,853 B，比新代码小 2,643 B，早于 09-28 的 rulebook 提交）。
这与已记过的 `[OPT]` 假日志、`.ps1` 双 BOM 静默失败同源：**"看起来跑过了"**。
修法双保险：`build_x64.ps1` 编译前删产物（令"产物存在"⇒"本次产物"）+ `check_x64.ps1` 时间戳断言。

**② 仓库里 15 个 `.sh` 全是 `100644`（已修）**。板端全新克隆后执行脚本自带 Usage 里的
`./build_rk3588.sh` 直接 `Permission denied (rc=126)`。Windows 侧没有执行位概念所以从未暴露，
**Unix/aarch64 用户按文档跑必然失败**。

### 3. 验证方法（可照做）

```powershell
# x86-64（本机）
powershell -ExecutionPolicy Bypass -File tools\build\check_x64.ps1 -OutDir C:\vllm_x64_build
```

```bash
# aarch64（板端，全新克隆）
git clone https://gitee.com/pei-xiaoguang/kestrel-llm.git && cd kestrel-llm
./build_rk3588.sh
./build-rk3588/vllm_kestrel --test-l3
./build-rk3588/vllm_kestrel --test-sparse
```

### 4. 边界

- 本轮是**编译 + 无模型确定性自检**，与 x86 同口径；**未跑真实模型推理**（需权重与页缓存控制）。
- 板端 `--bench-mixed` / `--npu-selftest` 未跑（耗时长且需 NPU 校准）。
- 板端到 GitHub 只能走 codeload 归档（`github.com` 直连超时），故 GitHub 侧在板端是"归档校验"
  而非"git 克隆校验"；两侧编译输入已在 x86 侧证明逐条相同（`src/ include/ tools/` 清单无差异）。
- x86 侧仍有开放项：**x64 偶发整请求卡死**（本轮未复现，也未解决）。

---

## 2026-09-28（下半场）— 工业边缘「预置上下文」：规则包 / 会话存储 / 审计流水

**这一轮要回答的问题**：工业现场的设备上，模型每轮问答都要用到同一份**固定且反复使用**的
"命名上下文"——现场规程、设备清单、术语表、工艺参数表、安全红线。若把它当普通 prompt 每轮
重新 prefill，代价随对话轮数**线性叠加**。**能不能把它预置成一份可复用的前缀，让后续请求只付
"增量"的那部分？**

**结论（先给答案）**：

1. **能，且短手册的收益量级很大 —— 命中 ≈ 82×**。同机同模型（RK3588 / Qwen3-VL-2B），
   手册 5,791 token：

   | 路径 | 端到端 |
   |---|---|
   | 手册全文塞 prompt（每请求全量 prefill） | **194,275 ms** |
   | 命中规则包（复用落盘 KV 快照） | **2,361 ms**（进程内首次 3,174 ms，需载入快照） |
   | **收益** | **≈ 82×** |

   快照体积 **59.14 kB / token**（两条独立测量一致）。

2. **适用范围是「短而稳定的手册」，不是长文档**。硬边界来自源码 `DISKKV_MAX_TOKENS = 8192`：
   ≤ 8192 token 落盘复用；**> 8192 token 不落盘**，每请求退回全量 prefill。实测 200,000 B 手册
   （75,146 token）：`reused_prefix=false`、分块选段把**注入量**压到 19,467 token（−74%），
   但**每请求仍要 1,562 s** —— 分块检索解决的是"注入多少"，不解决"能不能复用"。

3. **配套两个模块**：`vllm_session`（会话历史落盘 + 上限）与 `vllm_audit`（追加式审计流水，
   带哈希链）；管理页补规则包注册中心与命中计数，并新增**审查台** `review.html`。

4. **默认关时与引入前一致**：不带 `rulebook_id` / `session_id` 时，新旧二进制的
   `history_tokens` 序列**逐 token 完全相同**（token-id 级证据）。

### 1. 改动清单

| 文件 | 改动 |
|---|---|
| `include/serve/vllm_rulebook.h` + `src/serve/vllm_rulebook.c` | **新增**：规则包构建 / 加载 / 注册中心；快照命名 `rbk_<id>_<ver>_<asm>.kv`，`<asm>` 只由**实际注入的 token 序列**决定（同包不同选段 ⇒ 不同快照，不互相污染） |
| `include/serve/vllm_session.h` + `src/serve/vllm_session.c` | **新增**：会话历史落盘与上限（`--session-dir` / `--session-limit`） |
| `include/serve/vllm_audit.h` + `src/serve/vllm_audit.c` | **新增**：追加式审计流水（`--audit-log`），带哈希链 |
| `src/serve/review.html` | **新增**：审查台页面 |
| `src/serve/admin.html` / `vllm_admin.c` | 管理面新增 `GET /admin/api/rulebooks`、`/admin/api/config`、`/admin/api/status`（含 `rulebook_hits`）与删除端点 |
| `src/main.c` | 新增 `--rulebook-build/-id/-name/-version/-tenant/-scope/-dir`、`--session-dir/-limit`、`--audit-log` 解析 |
| `src/serve/vllm_server.c` / `vllm_http.c` / `include/serve/vllm_server.h` | 请求体支持 `rulebook_id` / `session_id` / `tenant_id` / `user_id`；响应 metrics 增加 `reused_prefix` / `rulebook_tokens` |
| `src/serve/embedded_web.c` / `include/serve/embedded_web.h` / `tools/build/gen_embedded_web.py` | 内嵌页面集扩展（含 `review.html`） |
| `CMakeLists.txt` / `tools/build/build_x64.ps1` | 纳入 3 个新模块 |
| `.gitignore` | 排除运行期产物：`rulebooks/`、`sessions/`、`rbk_*.kv`、`sess_*.json` |
| `wiki/规则包与预置上下文.md`（新增页）、`wiki/Home.md`、`wiki/导航.md` | 机制、开关、实测与**诚实边界**成页，并接入导航 |

**落盘产物**（均在 `--rulebook-dir` 下）：`rb_<id>.json`（元信息）、`rb_<id>.txt`（注入文本）、
`rbk_<id>_<ver>_<asm>.kv`（预置前缀 KV 快照）。

### 2. 验证证据（RK3588 / Qwen3-VL-2B / 2026-09）

**功能正确性（全过）**

| 项 | 结果 |
|---|---|
| 离线构建 | 15,000 B → 5,791 token，`chunks=12`，`kv_snapshot=yes`，构建 204.7 s |
| 借用上下文端到端 | 连续 3 次全部 `reused_prefix=true`，`rulebook_hits` 1→3，答案正确 |
| 失效隔离 | 不存在的 id / 含 `/` / 含空格 / 超长 id → **404**；版本不符 → **409** |
| 删除 | 删后请求 404；审计流水记 `rulebook_delete` |
| 会话隔离 | 两个 session 各答自身内容，无串扰 |
| 分块检索 | 75,146 token 手册 + 事实置于**后半段** → **仍能检索到** |

**兼容性**

- **默认关时与引入前一致**：不带 `rulebook_id` / `session_id` 时 `history_tokens` 逐 token 相同。
- **可与稀疏档共存**：`--sparse-attn` + `VLLM_SPARSE_PREFILL=1` 下复用正常、召回正确；快照名的
  `<asm>` 与不开稀疏档时**相同**。
- **`--no-prefix-kv` 是逃生舱**：`reused_prefix=false`、每次全量 prefill、`rulebook_hits` 恒 0。
- **多包 / 多租户**：两个包交替请求，命中计数各自递增，无串扰。

### 3. 口径红线（必须随结论一起给出）

复用路径与"全文重算"的**浮点归约顺序不同**，因此这是 **「确定性但近似」**，
**不是**全量重算的位级克隆。要回到全量位级结果，用 `--no-prefix-kv`。

### 4. 诚实边界

1. **82× 依赖"手册短"**：在 5,791 token 手册上测得；超过 8,192 token 即失去复用。
2. **兼容性证据只到 token-id 级**（引擎不暴露 logits），**非位级**。
3. 性能表的 B 节为 **n=1 单次采样**，未做重复统计与噪声门槛。
4. `DISKKV_MAX_TOKENS = 8192` 来自源码，板端仅有"超限未落盘"这一条**间接**印证。
5. **未做并发 / 多租户压力测试**，全部测点为单请求串行。
6. `--no-prefix-kv` 的"回到位级结果"由 `reused_prefix=false` + 全量 prefill + 源码门条件**推断**，未做逐 token diff。
7. 体积外推（20K token → 1.18 GB、100K token → 5.91 GB）是**推算，非实测**。
8. 运行期产物（`rulebooks/`、`sessions/`、`rbk_*.kv`、`sess_*.json`）已进 `.gitignore`，**不随仓库分发**。

### 5. 同批归档的文档与预注册

- `docs/N1_64核_多线程性能基准报告.md`（64 核平台线程扩展性基准）
- `docs/性能优化方法论.md`、`docs/prefill注意力PV分块优化原理.md`
- `docs/投机解码瓶颈分析与MTP落地方案.md`、`docs/逐层驻留专家级读取方案.md`
- **预注册**：`docs/PRE_REG_MOE_LONGCTX_NIAH.txt`、`PRE_REG_Q4_GEMM_PROBE.txt`、
  `PRE_REG_STREAM_EXPERT_GRAIN_PROBE.txt`、`PRE_REG_STREAM_SA.txt`（先写判据再看数据）
- `docs/bench/20260924-n1-thread-scaling/`、`docs/bench/20260925-n1-moe-30b-a3b/`：原始产物
  （日志 / 计时 / SUMMARY.tsv）+ `MANIFEST.txt` md5 留证

---

## 2026-09-28 — 稀疏注意力的「上下文长度门控」：decode / prefill 两侧解耦

**这一轮要回答的问题**：`--sparse-attn` 与 `VLLM_SPARSE_PREFILL=1` 的启用门都是
`seq_len > g_sparse_block * 2`（=64）。64 远低于任何有意义的上下文长度，**等于没有门**。
而 probe 选块开销 ∝ ctx/block、省下的注意力 ∝ (ctx − k·block)，短上下文下净亏。
能否改成按上下文长度门控？

**结论（先给答案）**：

1. **确实是净亏，且只发生在短上下文**。2B 多轮实测（真实语料 + 严格判据，3 次重复一致），
   稀疏 prefill 相对精确 prefill 的代价：

   | ctx | 1067 | 2174 | 4374 | 8618 | 16482 |
   |---|---|---|---|---|---|
   | 变化 | **+18.4%** | **+12.0%** | −11.1% | −29.0% | −45.1% |

   交叉点在 **2K~4K** 之间。

2. 新增两个**独立**的门（都在旧门之上取「与」）：`--sparse-pf-min-ctx`（prefill，默认 **3072**）、
   `--sparse-min-ctx`（decode，默认 **1024**）。两侧交叉点不同：prefill 每个 prompt token 只付
   一次 probe，decode 每个生成 token 都要付；实测 4K 起 decode 稀疏已稳定快 **1.8×**
   （dense 409 → sp32 223 ms/tok），故 decode 门槛更低。

3. **默认路径逐位不变**：两个门只在显式开启 `--sparse-attn` / `VLLM_SPARSE_PREFILL=1` 时参与
   判定；`0` = 关闭本门，回到改动前行为。

4. 顺带修正一处**假日志**：`[OPT]` 原印 `sparse 剪枝阈值: seq_len > %d`（值取
   `g_sparse_k * g_sparse_block` = 1024），但该表达式**只出现在这行 printf 里，不对应任何代码
   路径** —— 两个真实门都是 `g_sparse_block * 2`（=64）。已改为印真值
   `decode>=%d prefill>=%d`。

### 1. 改动清单

| 文件 | 改动 |
|---|---|
| `src/model/vllm_safetensors.c` | 新增 `int g_sparse_min_ctx = 1024;` / `int g_sparse_pf_min_ctx = 3072;`；decode 门（`vllm_attn_head_worker`）与 prefill 门（`st_attn_batched_packed_sparse` 调用点）各加 `seq_len >= <对应门>` |
| `include/model/vllm_safetensors.h` | 两个 `extern` |
| `src/main.c` | `--sparse-min-ctx N` / `--sparse-pf-min-ctx N` 解析；`[OPT]` 回显新增两字段并把阈值文案改正 |

### 2. 验证证据

**本地 x64（受控对照：同 ctx、同参数，只差「门」或只差「filler」）**

| case | ctx | 门 | filler | `prefill-sparse` | `decode-sparse` calls | 召回 |
|---|---|---|---|---|---|---|
| A | 2049 | pf=3072 | 模板 | **0/0** | — | YES |
| C | 2049 | pf=0 | 模板 | 101020/143808 (70.25%) | — | YES |
| E | 2048 | pf=3072 | 真实语料 | **0/0** | — | YES |
| B | 4097 | pf=3072 | 模板 | 24629/143808 (17.13%) | — | **NO** |
| D | 4096 | pf=3072 | 真实语料 | 34652/143360 (24.17%) | — | **YES** |
| G | 768 | dec=1024 | 模板 | 0/0 | **0** | YES |
| H | 768 | dec=0 | 模板 | 0/0 | 14336 | YES |

- **A vs C**：同 ctx、同参数，只差 prefill 门 ⇒ 稀疏核从「0 次调用」变为「70.25% 针块命中」
  ⇒ 证明门真的在控制内核调用，不只是改日志。
- **B vs D**：同 ctx、同门，只差 filler ⇒ **B 的 `NO` 是模板 filler 伪影**。引擎自身注释即警告
  「模板 filler 是 32 个人名/8 种颜色的循环拼接，高度重复，用它测出的『稀疏注意力是否可用』
  对真实语料未必成立」。**此后判断稀疏注意力质量必须带 `--longctx-corpus` 真实语料。**
- **G vs H**：只差 decode 门 ⇒ `calls 0` vs `calls 14336`。

**板端 aarch64（`candidate-build/vllm_kestrel` sha `90c0d836…`，真实语料，`VLLM_NEEDLE_TRACE=1`）**

| case | ctx | 门 | 观测 | 判定 |
|---|---|---|---|---|
| `D768_minctx1024` | 768 | dec=1024 | `decode-sparse … calls 0` | 未调用 |
| `D768_minctx0` | 768 | dec=0 | `calls 14336` | 调用 |
| `P2048_pfminctx3072` | 2048 | pf=3072 | `prefill-sparse 0/0` | 未调用 |
| `P2048_pfminctx0` | 2048 | pf=0 | `1396/2240 (62.32%)` | 调用 |
| `P4096_default` | 4096 | pf=3072 | `692/2240 (30.89%)`，召回 YES | 调用 + 正确 |

`[OPT]` 回显实测：

```
[OPT] … sparse_min_ctx=1024 sparse_pf_group=4 sparse_pf_k=0 sparse_pf_min_ctx=3072 l3_evict=0
      | sparse 剪枝阈值: decode>=1024 prefill>=3072
```

**溯源闭环**：`candidate-build/vllm_kestrel` 的改动前备份 sha256 = `20ca8bf8…`，正是
2026-09-27 性能报告里钉的那枚引擎 ⇒ 「报告 → 源码 → 二进制」三者可对齐。

### 3. 边界与诚实标注

1. **decode 门 64→1024 的收紧未经实测**。它只影响 ctx 64..1023，位于本轮全部实测区间
   （≥1067）**之外**，故既有性能矩阵与召回结论不受影响；但「1024 是最优点」本身没有实测支撑 ——
   它只是把「远低于任何有意义的上下文长度」的 64 提到原始设计意图值。若要坐实，需在
   ctx 512~1024 区间补 decode dense/sparse 的 A/B。
2. **prefill 门 3072 有实测支撑，但只有 2B**。8B 的逐轮 prefill 分解未单独跑（矩阵只记了 16K 轮），
   8B 的交叉点是否同样在 2K~4K 未验证。
3. **ctx=768 档两个 arm（含走精确核的那个）召回都是 NO** ⇒ 该档失败与稀疏无关，是极短上下文下的
   答案格式/语料问题；它作为一条**阴性对照**支持「门未引入质量回归」。
4. **未上线**：本轮改动只在本地工作区与板端 `candidate/` + `candidate-build/`；`/NewVLLM/` 部署件
   仍为 `20ca8bf8…`（与已交付的性能报告严格对应），按决定暂不替换。
5. 板端 `candidate/` 的这三个文件是**更旧的修订**（例：prefill 门处无 `vexact` 前缀、decode 门在
   L10345 而本地在 L10795），故补丁按**板端原文**逐条断言唯一性后定点替换（9 处 `hits=1`），
   不是整文件覆盖。板端与本地在 `vllm_safetensors.{c,h}` 上的整体对齐仍待单独排期。

### 4. 受影响文件

| 文件 | 改动函数 / 位置 |
|---|---|
| `src/model/vllm_safetensors.c` | 新增 `g_sparse_min_ctx` / `g_sparse_pf_min_ctx`（全局区）；门分别位于 `vllm_attn_head_worker`（decode）与稀疏 prefill 调用点 |
| `include/model/vllm_safetensors.h` | 两个 `extern` |
| `src/main.c` | CLI 解析（`--sparse-min-ctx` / `--sparse-pf-min-ctx`）与 `[OPT]` 回显 |

---

## 2026-09-25（下半场）— 逐层驻留「专家级权重读取」方向 E：预注册探针 + 形态 A 设计稿 + 本地 MoE 闭环

**这一轮要回答的问题**：MoE 的逐层驻留档下，decode 的**计算**是稀疏的（只算 top_k=8），
但**权重读取/页驻留**的粒度是**整层**（含全部 128 专家）。能否把读取也降到专家级？

**结论（先给答案）**：
1. 结论分两轴 —— 计算**只算激活专家**（代码确认）；驻留**是整层**（`step = bytes/nl` 含全部
   128 专家，层入口对 l+1 整层 WILLNEED、对 l-keep 整层 DONTNEED）。
2. N1 实测 `stream − full = +277.5 ms/token`，可归为 **每 token ≈31.32 GiB 的 madvise 区间**
   （8.21M 页 ⇒ 33.8 ns/页）；**不 ∝ 激活权重**（跨模型旁证见下）。
3. 方向**已实现但未转正**：形态 A 代码已落地（`VLLM_SA`，默认关）；经板端替代口径判据
   （§7B）裁决 **P-A-1t 未达标 ⇒ 不转正**（详见 §6）。落地转正的原始门槛仍是
   **N1 的 P-EW-1 达标 + 用户显式批准**。

### 1. 事实与量化（两轴）

| 轴 | 现状 | 证据 |
|---|---|---|
| 计算 | 只算 top_k=8 个激活专家 | `st_moe_ffn_sparse_q4`：选路后只遍历 `sel[0..tk-1]`；批式 GroupGEMM 有 `nb < 2 return 0`，decode nb=1 进不去 |
| 驻留/读取 | **整层**（含全部 128 专家） | `vqf_stream_setup` 的 `step = te->bytes/nl`；`vqf_stream_layer_advance` 整层 `MADV_DONTNEED(l-keep)` + 整层 `MADV_WILLNEED(l+1)` |

代价模型（N1/aarch64，30B-A3B-q4，t32，热页缓存；归档 `docs/bench/20260925-n1-moe-30b-a3b/`）：

| 项 | 值 |
|---|---|
| `stream − full` decode | **+277.5 ms/token**（T=32/40/48/56/64 为 277.5/281.0/293.0/292.0/306.5，几乎不随算力变） |
| 每 token madvise 区间 | `334.1 MiB × 48 × 2 = 32,073.6 MiB` = **31.32 GiB ≈ 8.21M 页** |
| 单位成本 | **33.8 ns/页**（syscall 计数被排除：1056 次/token 仅 ≈263 µs/次） |
| 每层字节分解 | gate+up 216.0 MiB（64.7%）/ down 108.0 MiB（32.3%）/ attn+norm 10.1 MiB（3.0%） |
| 跨模型**旁证**（跨 regime，不作判据） | 8B（稠密 6.60GB）Δ=+157ms；30B-A3B（激活 ~1.6GB）Δ=+398ms ⇒ 比 2.54 ≈ 模型字节比 2.68，≠ 激活字节比 0.24 |

### 2. 本轮落地：预注册探针 `VLLM_STREAM_PROBE`（默认关、零回归）

| 档 | 语义 |
|---|---|
| 1 | 只保留整层 DONTNEED(l-keep)，关闭整层 WILLNEED(l+1) |
| 2 | 只保留整层 WILLNEED(l+1)，关闭整层 DONTNEED(l-keep) |
| 3 | **专家段（q4/q8_gate/up/down 且 MoE 几何成立）不做任何 madvise**；非专家段同现状 ⇒ 「区间→0」乐观地板（RSS 无界，仅大内存机可跑） |
| 4 | 区间同现状，但每次调用按 16 片等分发出（syscall 计数对照） |

实现：`src/model/vqf.c` —— `VQFStreamSeg` 增 `is_expert`（MoE 几何门 `arch.n_experts·moe_ffn == ffn_dim`
+ 张量名），新增 `vqf_stream_probe_mode/_advise_gran/_attest`，层入口钩子按档分派；
一次性自证行 `[STREAM-PROBE] mode=.. nl=.. segs=.. expert_segs=.. per-layer=..MB`
（未设开关时**无任何输出** ⇒ 零回归可核）。**只改 madvise 的目标区间与调用切分，浮点序零改动。**

预注册（判据跑前冻结、禁止事后回改）：
- 探针：[docs/PRE_REG_STREAM_EXPERT_GRAIN_PROBE.txt](docs/PRE_REG_STREAM_EXPERT_GRAIN_PROBE.txt)
  —— **P-EW-1 `Δ_L3/Δ_L0 ≤ 0.35`**（不达标即关闭方向）、**P-EW-2 `Δ_L4 ≤ 1.05×Δ_L0`**、
  G0（F 与 L0–L4 的 TOKIDS 逐位相同）。
- 形态 A：[docs/PRE_REG_STREAM_SA.txt](docs/PRE_REG_STREAM_SA.txt)
  —— F/A0/A1/A2 四臂、G0 与 P-A-1..P-A-4（比值 ≤0.35 / RSS / 板端 prefill 不退化 / 回归），
  开关 `VLLM_SA`、`VLLM_SA_DOWN=whole|keep`、`VLLM_SA_SELPROBE`；M6 豁免（热缓存 + 同 boot 多配置）
  须用户签署、M4 以 F 桶做漂移锚点。

设计稿（**已按此落地，见 §6**）：[docs/逐层驻留专家级读取方案.md](docs/逐层驻留专家级读取方案.md)
核心三点：① 驱逐参照系必须是**驻留集 R_l**（不是「层的全体」——保留 8/128 在页粒度上等于没保留，
这正是 EW 静态热窗的参照系问题）；② 时机必须从层入口移到**选路后**（层入口拿不到本 token 的真 A_l）；
③ **down 是硬缺口**：行组 221,184B（54 页）/ 专家列带 1,728B，页被邻专家共享 ⇒「保留 8、驱逐 120」
需 ~4000 区间/层（≈20 万 syscall/token）不可表达 ⇒ 三选一：**D1**（整层 DONTNEED + 取消 down 的
WILLNEED，RSS 有界，Δ≈64ms）/ D2（down 不驱逐，Δ≈19ms，RSS +5.06 GiB）/ D3（文件重排，另立项）。
用户已冻结：**默认 D1**；接受 down 的 ≈2.4× 页共享地板。

### 3. x86 功能/位级自检（只证零回归与 A≡C，**不出性能结论**）

（a）稠密 2B（`Modl/Qwen3-VL-2B-q8local`）—— 六档 TOKIDS 全 `100062`；零回归 + A≡C PASS；
`tools/build/check_x64.ps1` → `--test-l3` 16 PASS/0 FAIL、`--test-sparse` 9/9、X64 CHECK PASSED。
探针二进制 `build-x64-probe/vllm_kestrel_x64.exe` sha256 `2226F162…364F`（1,365,614 B）。
该档 `expert_segs=0`（稠密文件无 MoE 专家段）⇒ mode=3 按定义退化为 L0（WARN 已打）。

（b）**真实 MoE 闭环**（本轮新增）：本地用 `vqf_convert` 把 `Modl/Qwen3-30B-A3B` safetensors
转 q4 VQF ⇒ `Modl/Qwen3-30B-A3B-q4local/model.vqf` = **17,665,601,544 B**（与 N1 归档逐字节同尺寸；
`flags=0x223` = Q8_8X8|Q4_4X4|EMB_F16|MOE），补 `vocab.bin`（md5 `207fbf41…`，与 30B 同源）
+ `config.json` 后重跑六档：

```
F : TOKIDS 151667  rss=11,168,612kB
L0: TOKIDS 151667  rss=  510,736kB  [VQF-STREAM] enabled keep=1 nl=48 segs=11
                                    per-layer=334.1MB data=16847.2MB resident~808.4MB
L1: TOKIDS 151667  rss=  511,888kB  [STREAM-PROBE] mode=1 expert_segs=3
L2: TOKIDS 151667  rss=11,168,592kB mode=2 expert_segs=3
L3: TOKIDS 151667  rss=10,680,504kB mode=3 expert_segs=3
L4: TOKIDS 151667  rss=  511,860kB  mode=4 expert_segs=3
```

- 六档 TOKIDS **逐位相同** ⇒ A≡C 在真实 MoE 上成立；
- **`expert_segs=3`** ⇒ mode=3/4 的专家段分支在真实 MoE 上生效（原「本机无法覆盖」的空白闭环）；
- L3 的 RSS 10.68GB（专家段全不驱逐）符合设计预期（无界为故意）；
- `[VQF-STREAM]` 自证行与 N1 归档**逐字段相同**（`per-layer=334.1MB` / `data=16847.2MB` /
  `resident~808.4MB`）⇒ 跨机复现了转换链与驻留口径。
- 边界：x86 的 ms（F 107 / L0 1228 / L1 395 / L2 759 / L3 199 / L4 1255）**无性能含义**——
  Windows 用 `VirtualUnlock`/`PrefetchVirtualMemory` 代理，语义与 Linux `madvise` 不同；只作机制旁证。

### 4. 顺带发现（**x86 默认轨与 ARM 不同，位级对拍必须显式锁轨**）

x86 的 `actq16_env()` **默认返回 1**（除非显式 `VLLM_ACTQ16=0`，`vllm_safetensors.c` L11403-L11407），
即 **x86 上 MoE q4 的缺省轨是 s16-dot 近似轨**；ARM 侧是恒 0 的编译期桩（L11534-L11543）。
⇒ **x86 上做 MoE 位级对拍必须显式 `VLLM_ACTQ16=0`（并 `VLLM_ACTQ=0`）**，否则拿到的是近似轨输出；
**N1/RK3588（ARM）不受影响**，精确轨即缺省（与 README「精确轨日志不含 `[ACTQ]`/`[ACTQ16]` 行」一致）。
本轮 (b) 的六档即在显式锁轨下取得（输出无任何 `[ACTQ*]` 行）。

### 5. 板端旁证（RK3588）：一次被自查作废的轮次 + 受控轮结果

部署：把源码（含探针改动）打包 → scp 到板端**新目录** `/mnt/emmc/saprobe`（**不动**
`/mnt/emmc/kestrel_pull` 与 `kestrel_prod_*` 两棵既有树），用仓库自带
[tools/relay/fix_crlf.sh](tools/relay/fix_crlf.sh) 统一 LF（29 → 0），`build_rk3588.sh` 原生构建：
binary sha256 `b93893282288e09255a3aa568cefc946411fd65b9986f76129ae35f24d9829bf`（1,060,112 B）。
模型 `/mnt/VQF/qwen3-30B-A3B-q4`（`model.vqf` sha256 `DAE4CB26…FE93`，与其自带 `.sha256` 自洽）。

**教训（必须照实记）：第 1 轮（暖缓存、按 L0→L1→L2→L4 顺序）判为 VOID。**
表面结果极强——decode L0 2308 / L1 777 / L2 445 / L4 883 ms/tok——但**prefill 也随执行次序
单调下降**（96.3 / 20.3 / 8.3 / 16.7 s）。prefill 是另一相位却也降 11.5× ⇒ 该轮测到的主要是
**页缓存逐臂变暖**，不是臂语义。该轮数据仅留档为「未控制状态变量的反例」（`logs/L*.log`），
**禁止引用其排序**。

**受控轮**（每臂前 `sync; echo 3 > /proc/sys/vm/drop_caches`，使各臂从同一冷态开始；
ctx=32 / n=8 / threads=4 / `VLLM_ACTQ=0`，单轮）：

| 臂 | prefill 32tok | decode ms/tok | rss_after_prefill | TOKIDS |
|---|---|---|---|---|
| L0（现状：整层 DONTNEED + 整层 WILLNEED） | 228.889 s | 1992 | 513,056 kB | 151667 |
| L1（关整层 WILLNEED） | 229.248 s | 1979 | 517,076 kB | 151667 |
| L4（区间同 L0、切 16 片） | 232.951 s | 1841 | 513,496 kB | 151667 |
| L2（关整层 DONTNEED） | 227.991 s | 1500 | 11,750,152 kB | 151667 |
| L3*（专家段零 madvise） | 228.553 s | 1520 | 11,262,536 kB | 151667 |

- prefill 各臂 228~233 s（Δ<2%）⇒ **探针不改批式/prefill 路径**，与设计一致；
- **L1 ≈ L0（+0.7%）⇒ 整层 WILLNEED 不是板端 decode 的主导项**（第 1 轮的「L1 快 66%」是伪影）；
- L4 −7.6%（弱效应）；L2/L3（不驱逐）−24~25%，代价是 RSS 11.3~11.8GB（≈ 全模型）——
  与逐层档「常驻与模型体积解耦」的目的相悖，在 15.6GB 板上不可持续；
- 五臂（含 L3）TOKIDS 全 151667 ⇒ **位级一致在板端亦成立**；
- 冷 prefill 228 s / 17.67 GB ⇒ ≈77MB/s，与 microSD 级带宽同量级 ⇒ 板端是**介质受限**；
  臂间差异 ≤25%，**无法分离「页表成本」与「I/O 成本」** ⇒ **不能裁决 P-EW-1**（那必须 N1）。
- *L3 越出预注册臂集，按约定标「越界旁证、不作判据」。

**"只加载激活权重"能不能用现成开关拼出来 + 它到底有没有收益**（探索臂 xA，不在冻结臂集内）
组合 `VLLM_VQF_STREAM=1 VLLM_STREAM_PROBE=1 VLLM_EW_PREFETCH=1`：层入口不再发整层 WILLNEED，
预取只发到本 token 激活的 8 个专家的 gate/up 段（expB=0.844MB / 216 页，**0% 过取**），down 不预取。
自证：`[VQF-PF] enabled nl=48 ne=128 exp=0.844MB(216页) down=0` + `[STREAM-PROBE] mode=1 … expert_segs=3`。

**交错双轮 A/B**（每臂前 `drop_caches`；轮序 r1 = L0→xA，r2 = xA→L0）：

| 轮 | 臂 | prefill 32tok | decode ms/tok | TOKIDS |
|---|---|---|---|---|
| r1 | L0 | 229.062 s | 1963 | 151667 |
| r1 | xA | 198.447 s | 1840 | 151667 |
| r2 | xA | 198.624 s | 1835 | 151667 |
| r2 | L0 | 228.053 s | 1958 | 151667 |

- 重复性：L0 两条 0.26%、xA 两条 0.27%，**且轮序颠倒后方向不变** ⇒ 效应为真，非次序伪影；
- 中位：decode **−6.3%**（1960.5 → 1837.5 ms/tok）、prefill **−13.1%**（228.56 → 198.54 s）；
- ~~归因：收益来自"专家级预取"~~ —— **该归因已被下面的「归因判定轮」推翻，勿引用**；
- **归因判定轮**（四臂同轮受控、每臂前 `drop_caches`，单轮）：L0 1966 / **L1 1931**（真"少加载"：
  整层 WILLNEED 完全不发）/ **L4 1868**（语义为零：区间同 L0，仅把调用切 16 片）/ **xA 1840** ms/tok。
  合并各轮（L0 1992/1963/1958/1966；L1 1979/1931；L4 1841/1868；xA 1844/1840/1835/1840）：
  **L1 − L0 ≈ 0（无收益）；L4 ≈ xA ≈ −5.5%~−6.3%** ⇒
  **−6% 不来自「只加载激活专家」，也不是「少了建页量」**（L4 的建页量与 L0 完全相同），
  而是来自**调用形态**（把一次 ≤334MiB 的巨区间 madvise 换成多次小区间调用）。
- 对本方向的含义：**「只加载激活 → 降低建页成本」这一目标至今没有任何实测覆盖** ——
  它只能在 N1（页表主导、245GiB 全缓存）上测（P-EW-1/L3，**未采集**）；
  板端是介质受限端，测不到该轴。
- 附带线索（与"专家级加载"无关，须独立验证）：板端"把大 madvise 拆小"本身有 −5.5%（零语义改动），
  但 syscall 数放大 16× —— 是否在页表主导端也成立，由 P-EW-2 裁决。
- 边界：`down=0` ⇒ 本组合**不含 down 的专家级加载**（页共享，1728B 列带 < 4KB 页，地板 ≈2.4×）；
  **结论只对板端（介质受限）成立**；N1（页表主导端）是否有收益、形态 A 的「只驱逐激活」是否更优，
  仍须 P-EW-1 / P-A-1。

### 6. 形态 A 落地：`VLLM_SA`（decode 只加载/驻留激活专家 + 专家级驱逐）+ 四层验证 + **板端判据裁决（不转正）**

**实现了什么**（`src/model/vqf.c`，`include/model/vqf.h`，`src/model/vllm_safetensors.c` 一处守卫）：

| 开关 | 语义 |
|---|---|
| `VLLM_SA=1` | 主开关（**默认关**，未设 = 逐字节现状） |
| `VLLM_SA_DOWN=keep` | down 段不做任何 madvise（D2）；默认 `whole`（D1）= 层入口整层 DONTNEED、不预读 ⇒ RSS 有界 |
| `VLLM_SA_STAT=1` | 只读计量：每 token 打印 madvise 覆盖页数（建页/拆页工作量）与 minor fault 增量；**独立于 `VLLM_SA`**，L0 档也能测 |

机制：层入口**不再**对 gate/up 专家段做整层 DONTNEED/WILLNEED；改用「每层驻留集记账 R_l」
（768 B 全模型）在**选路后**（top-k 已知）收敛：驱逐 `R_l \ A_l`（每专家 gate/up 各 1 次调用，
区间 = 该专家段 216 页）→ `R_l := A_l` → 对 `A_l` 发 WILLNEED（0% 过取）。
批式/prefill 由 `vqf_sa_set_batch(1)` 守卫（批内激活集≈全体专家）；非 MoE / 加密 / 未开流式 /
几何或**页对齐**不成立时 `[VQF-SA] REFUSED`（新增 offset 页对齐门控——转换器只保 64B 对齐）。

**验证一：位级一致 + 零回归（x86 真 MoE，显式锁精确轨 `VLLM_ACTQ=0 VLLM_ACTQ16=0`）**

| 臂 | TOKIDS | decode ms/tok | 自证 |
|---|---|---|---|
| off（零回归） | 151667 | 1229 | 无 `[VQF-SA]` 行 |
| A1（D1） | 151667 | 336 | `enabled down=whole nl=48 ne=128 exp=0.844MB(216页) res=768B` |
| A2（D2） | 151667 | 202 | `enabled down=keep` |
| 稠密 2B + `VLLM_SA=1` | 100062 | 281 | `REFUSED: 非 MoE 或几何不成立`（行为不变） |

⇒ **A≡C 成立**（只调 madvise，浮点序零改动）；默认关逐字节现状；非 MoE 优雅拒绝。
x86 的 ms 无性能含义（Windows 用 `VirtualUnlock`/`PrefetchVirtualMemory` 代理）。

**验证二：建页量（设备无关的代码级计量，本地与板端数字一致）**

| 臂 | pages_w / token | pages_d / token | **pages_sum / token** | vs L0 |
|---|---|---|---|---|
| L0（现状） | 4,020,474 | 4,020,474 | **8,040,948** | 1.00× |
| **A1（D1）** | 166,170 | 1,465,626 | **1,631,796** | **−79.7%** |
| **A2（D2）** | 166,170 | 166,170 | **332,340** | **−95.9%** |

（板端判据臂 ctx=256 n=32；x86 ctx=32 n=8 交叉复现 A1 = 1,690,548 ⇒ 计数器设备无关。
解析式预测 L0 = 8,210,842 页/token，实测 8,040,948，偏差 2.1% ⇒ 自洽。）

**验证三：板端判据 A/B（§7B 冻结口径；binary sha256 `203aae7e…`，ctx=256 n=32，每臂前 `drop_caches`）**

| 轮 | 臂 | prefill | rss_after_prefill | **rss_end** | decode ms/tok | pages_sum/tok | minflt/tok |
|---|---|---|---|---|---|---|---|
| r1 | L0 | 276.773 s | 606,972 kB | 612,208 kB | 1140 | 8,040,948 | 95,127 |
| r1 | A1 | 278.447 s | 605,464 kB | **1,447,376 kB** | **1058** | **1,631,796** | 87,139 |
| r2 | A1 | 277.558 s | 605,380 kB | **1,448,084 kB** | **1032** | **1,631,796** | 87,150 |
| r2 | L0 | 276.699 s | 604,244 kB | 612,532 kB | 1147 | 8,040,948 | 95,115 |
| r3 | A2 | 278.745 s | 605,320 kB | **6,640,544 kB** | **786** | 332,340 | 10,793 |

整格 stdout 落盘：[docs/bench/20260925-rk3588-moe-sa-protocol/](docs/bench/20260925-rk3588-moe-sa-protocol/)；
逐条裁决：[docs/PRE_REG_STREAM_SA.txt](docs/PRE_REG_STREAM_SA.txt) §7B-结果。

**裁决（§7B 冻结阈值，未回改）**：G0 PASS（五臂 TOKIDS 全 151667；prefill Δ=+0.74%<3%）；
**P-A-1b PASS**（0.203 ≤0.35）；**P-A-2b PASS**（按冻结形式：prefill 时刻）；**P-A-1t FAIL**
（A1/L0 = 1045/1143.5 = **0.914 > 0.90**，实际仅 −8.6%，未达 ≥10% 门槛）。
⇒ 依 §5 第 3 条（P-A-3(b) 失败 ⇒ 关闭形态 A）：**形态 A 不转正，`VLLM_SA` 保持默认关**，
实现作为可选档保留、不改既有语义。

**归因与边界（必须同引）**：
- **设计目标只在「建页量」轴上达成**：区间页数 **−79.7%（D1）/ −95.9%（D2）**，但实际 minor fault
  仅 **−8.4%（D1）** ⇒ L0 那 4.02M 页/token 的区间里**绝大多数是本 token 并不访问的页（区间过取）**，
  在介质受限端驱逐它们几乎不产生时间代价。
- **内存维度未达成「只驻留激活专家」（本轮新发现）**：`R_l` 记账只驱逐 `R_l \ A_l`，故每层仍常驻
  「上一 token 的 8 个专家」（48 层 ≈648 MB）⇒ A1 的 `rss_end` 在 decode 期涨到 **1.38 GiB
  （L0 的 2.36×，增量 ≈0.80 GiB，恰卡在 §7B 的 0.8 GiB 门槛上）**；A2 的 down 完全不驱逐
  ⇒ 48 层 down 全量累积，`rss_end` **6.33 GiB（L0 的 10.8×）**。
  **§7B 的 `rss_after_prefill` 探针看不到这段增长**（缺口已留档，下轮须把 `rss_end` 写进判据）。
- **A2 的 decode 0.687（−31.3%）不可据此转正**：§7B 已冻结 A2 为**辅助/不作判据**，且其
  6.33 GiB 驻留在 15.6 GB 机型上无部署余量；要转正须**另立预注册**。
- **收益判据仍只待 N1（P-EW-1/P-A-1）**：板端由介质 I/O 主导，**测不到页表成本轴**；
  本档**不得外推**为「页表主导端有效」。

**勘误（本轮先修后测，必须与旧数据同引）**：本轮先定位并修复了一个实现层 bug —
① `vqf_sa_sync` 原挂在 `vqf_ffn_prefetch` 末尾，而后者在 `VLLM_EW_PREFETCH` 未开时**提前 return**
   ⇒ 专家级「驱逐 R_l\A + 预取 A_l」**从未执行**；
② 层入口「跳过 gate/up 整层驱逐」原为**无条件**，而 prefill 期同步被批式守卫抑制
   ⇒ prefill 全程无人驱逐 gate/up，驻留涨到 7.75 GB。
修复 = 新增 `warm` 标志（仅本 token 已完成专家级同步才跳过整层驱逐）+ 同步改名公开的
`vqf_sa_sync_token()` 并**独立挂在 top-k 选路点**（`vqf_ffn_prefetch()` 之后一行）。
⇒ 旧 binary `2ecd48af…`/`eed8f2d7…` 的 A1 数据（pages_sum 1,543,668、decode −4.8%、rss 7.75 GB）
**只反映「调用形态」，不代表形态 A，已作废**（详见预注册 §7B 跑前勘误）。

**过程教训**：本轮有一次 A/B 整轮作废——根目录 `./vllm_kestrel` 是旧二进制，而运行脚本调的是它
（新产物在 `build-rk3588/`）。`build_rk3588.sh` 本来会 `cp` 同步根目录，但我这次手工跑的
`cmake --build`，绕过了那步。**板端跑前必须核 `sha256sum` + `strings | grep` 自证行**（已写入
预注册的板端块）。另：`/proc/self/stat` 的 minflt 解析首版有 bug（state 字段非数字，
`strtol` 原地不动 ⇒ 恒 0），已修正为「跨过 state 字符后取第 7 个数字字段」。
再另两点：`VLLM_SA_STAT` 首版只在 `VLLM_SA=1` 时生效，无法在 L0 档测建页量 ⇒ 已改为独立开启；
**含中文路径的 `.ps1` 在 PS 5.1 下按 ANSI 解码会乱码**（本轮 x86 驱动脚本踩到，路径变 `椤圭洰`）⇒
须存为 UTF-8 **BOM** 或改内联命令执行。
再再一点（方法学）：**判据探针本身可能有盲区**——§7B 用 `rss_after_prefill` 量内存，
它看不到 decode 期驻留增长（A1 +0.80 GiB / A2 +5.75 GiB 全靠 `rss_end` 才暴露）。
下轮凡涉及"常驻/驻留"的判据，探针须覆盖**相位末端**而非仅 prefill 末。

### 7. 层尾驱逐（`VLLM_SA_EVL`）：目标 B「紧致驻留」的机制达成、判据未达成

**背景（术语纠正，先定死）**：此前把「只驻留激活专家」当成一个目标，实测表明必须拆成两个：
- **目标 A「稀疏」**：驻留集不含任何非激活专家 ⇒ **形态 A 已达成**（gate/up 过取 = 0%：
  pages_w = 166,170 页/token，解析值 8×2×216×48 = 165,888 ⇒ 1:1）。
- **目标 B「紧致」**：驻留量 ≈ **单层瞬时**（≈13.5 MiB/层），而非 48 层激活集之并 ⇒ **未达成**。
  `R_l` 在选路后即 `:= A_l`，**没有驱逐残留**；1.38 GiB 是 48 层激活集的**并**。
  另：**压 `pages_sum` 压不到时间**——L0 的 8.04M 页 vs 95k 缺页（84.5×）、A1 的 1.63M vs 87k（18.7×），
  差距收窄 4.5 倍而板端时间只从 −0% 走到 −8.6% ⇒ 过取的页本来就不产生缺页。

**实现**（`src/model/vqf.c` / `include/model/vqf.h`；默认关）：`VLLM_SA_EVL=1`（需 `VLLM_SA=1`）——
进入 layer l 时把「上一层已用完的激活集」`R_{l-keep}` 的 gate/up 段 DONTNEED 并清空记账
（`vqf_sa_evict_layer`），把驻留从「48 层激活集之并」压到「单层瞬时」。只改 madvise ⇒ A≡C。
`[VQF-SA] enabled …` 行追加 `evl=0|1` 自证字段。

**判据（预注册 §8，跑前冻结）与实测**（板端，ctx=256 n=32，binary `d1ea20ad…`，六臂交错）：

| 轮 | 臂 | prefill | rss_after_prefill | **rss_end** | decode ms/tok | pages_sum/tok | minflt/tok | TOKIDS |
|---|---|---|---|---|---|---|---|---|
| r1 | L0 | 277.262 s | 604,032 kB | 610,408 kB | 1154 | 8,040,948 | 95,126 | 151667 |
| r1 | A1 | 278.414 s | 602,712 kB | 1,445,596 kB | 1036 | 1,631,796 | 87,141 | 151667 |
| r1 | **A1E** | 278.395 s | 606,488 kB | **803,012 kB** | **1094** | 1,869,396 | 95,116 | 151667 |
| r3 | **A1E** | 275.710 s | 603,732 kB | **806,256 kB** | **1068** | 1,869,396 | 95,116 | 151667 |
| r3 | A1 | 276.823 s | 604,032 kB | 1,446,968 kB | 1029 | 1,631,796 | 87,148 | 151667 |
| r3 | L0 | 277.217 s | 604,680 kB | 611,368 kB | 1122 | 8,040,948 | 95,109 | 151667 |

- **G0 PASS**：六臂 TOKIDS 全 151667；A1E 两轮均打印 `evl=1`；prefill 最大劣化 +0.42%。
  另：本轮 L0/A1 复现了 §7B 的 `rss_end`（差 <0.4%）⇒ `evl=0` 是 no-op **获交叉证实**。
- **P-E-1 FAIL**：rss_end(A1E) 804,634 kB > 门槛 715,746 kB（= L0 + 0.1 GiB），超 +86.8 MiB。
- **P-E-2 PASS（勉强）**：decode(A1E)/decode(A1) = 1081/1032.5 = **1.047**（n=64 复测 1.040）。
- **P-E-3**：pages_sum **+14.6%**、minflt **+9.2%**（回到 L0 水平）。

**归因（判据外诊断，binary `90614b2e…`，n=64；`[SA-STAT]` 新增 `rss_anon/rss_file` 只读拆分）**

| 臂 | `rss_file` prefill | `rss_file` t=32 | **`rss_file` t=64** | **file − L0file** | `rss_anon` t=64 | decode |
|---|---|---|---|---|---|---|
| L0 | 499,692 kB | 500,856 | **501,608** | 0（基准） | 121,548 | 962 |
| A1E | 498,944 kB | 690,672 | **714,056** | **+207.5 MiB** | 121,560 | 937 |
| A1 | 499,244 kB | 1,334,692 | **1,354,448** | **+832.9 MiB** | 121,556 | 901 |

- **`rss_anon` 三臂逐点完全相同**（+15.4 MB/64 tok = 0.24 MB/token ≈ KV 预测 0.19）⇒
  **"分配器/KV 泄漏"被排除**，差异全在 **`rss_file`（mmap 权重页）**；**L0 的 file 全程平**（+1.9 MB）。
- **EVL 生效量可精确核对**：file(A1) − file(A1E) = **625.4 MiB**，预测应移除 648 MiB ⇒ **96.5%**。
- **残留 = `rss_file` 的 SA 专属增量 +207.5 MiB = 门槛的 2.03×** ⇒ P-E-1 的 FAIL 由它造成，
  **与 EVL 机制本身无关**。残留**减速收敛**（t=8→32 增 79.3 MiB；t=32→64 只增 22.6 MiB）
  但尾部仍有 ≈0.94 MiB/token（n=64 不足以下"已收敛"结论）。
- **类别已定位、具体来源未定位**。候选（**未验证**）：(a) 边界层——`evict_l = layer−keep`
  永不覆盖 `layer = nl−1`；(b) readahead 外溢——SA 每层发 8×2 个 216 页小区间 WILLNEED，
  内核预读越出被驱逐区间后**永不被 expert-granular DONTNEED 回收**。
  判别实验（下轮，另立预注册）：把驱逐区间页对齐后向外扩 N 页，残留随之下降 ⇒ (b)，不变 ⇒ (a)。

**结论**：目标 B 的**机制按设计生效**（96.5%），但**判据不达标**；
**形态 A 转正状态不变（仍不转正，理由仍是 §7B 的 P-A-1t FAIL）**。
预注册与逐条裁决：[PRE_REG_STREAM_SA.txt](docs/PRE_REG_STREAM_SA.txt) §8；
整格 stdout：[docs/bench/20260925-rk3588-moe-sa-protocol/](docs/bench/20260925-rk3588-moe-sa-protocol/)（`ev_*.log` 判据臂、`dg_*.log` 归因诊断）。

**过程记录**：① 诊断轮 build 出现 `cp: … Text file busy` ⇒ 仓库根副本未同步（根=旧 binary /
`build-rk3588/`=新 binary），**正是 §4 第 3 条禁止的陷阱**，靠 hash 核对发现并 `cp -f` 修正后才开跑；
② 批量 3 臂时 L0 臂在 prefill 起始被杀（无 OOM、SSH 挂住）⇒ 单独重跑成功，判据臂未受影响；
③ 含中文路径的 `.ps1` 须存 UTF-8 **BOM**（本轮 x86 驱动脚本再次踩到）。

### 8. 收口（2026-09-25，用户决定：停止推进，记录入档）

**收口宣告**：形态 A 家族（**A1 / A1E / A2**）冻结为**可选档、默认关**，**不再继续实现、不再在板端测量**。
唯一开放项 = **N1（页表主导端）上的 P-EW-1 / P-EW-2 / P-A-1**，而 N1 通道当前不可达
（`114.215.207.71` 22/443/ping 全不通、本机无凭据）。**账目与逐条证据见
[PRE_REG_STREAM_SA.txt](docs/PRE_REG_STREAM_SA.txt) §9（收口宣告）**，本节只摘结论。

**为什么停在这里（决策价值）**

| 可能的下一步 | 决策价值 | 理由 |
|---|---|---|
| **N1 三臂（L0/A1/A2）** | **高（唯一）** | 唯一能裁决「方向成立/放弃」的判据；预测可把逐层档的时间代价从 **3.53× 压到 1.51×（A1）/1.10×（A2）** |
| EVL 残留归因（驱逐区间外扩 N 页） | **零** | 修好也只是**追平 L0**（610 MB），不构成收益；且不触碰 P-A-1t |
| 板端再跑别的臂 | **零** | 板端已**两次自证**测不到页表成本轴（L1≈L0 复现、L4 零语义对照） |
| 再改 madvise 形态 | **零** | gate/up 过取已 0%；down 原理上做不到；区间已降到 20.3% |

**唯一开放判据的定量尺度（预测，非实测）**：N1 归档 ctx=256/t32 为 `full` 109.5、`stream` 387.0、
Δ 277.5 ms/token（T∈32..64 几乎不变）。若「成本 ∝ 覆盖页数」成立，A1（20.3% 覆盖）⇒
`stream ≈166 ms`（vs L0 **−57%**、vs full 1.51×）；A2（4.1%）⇒ `≈121 ms`（−69%、1.10×）。
**但方向不确定**：该假设只有同量级跨模型旁证（非定律）；A1 的覆盖 91% 是 down 整层驱逐、
**未减少 madvise 调用数**；而板端已实测「零语义、只把调用切小」值 −5.5~−6.3% ⇒ N1 上调用形态
可能**反向加价**——这正是未采集的 **P-EW-2**。

**必须披露的两点**：① **§7B 把 A2 冻结为「辅助/不作判据」，但事后数据指向 A2 可能最优**
（板端 decode −31.3%、缺页 −88.7%、`rss_end` 6.33 GiB 相对 `full` 13.6 GB 仍省 2.2×）——
不能在 §7B 事后改口径；若 N1 恢复，**须另立预注册**把 A2 作判据臂。
② **板端的最优策略是「少驱逐」**（见 §7 的缺页模型），与形态 A 的「激进驱逐」方向相反。

**收口时的仓库/装置状态**（便于复现与回滚）
- 改动文件（工作树，**未提交 git**）：`src/model/vqf.c`、`include/model/vqf.h`、
  `src/model/vllm_safetensors.c`；文档 `PRE_REG_STREAM_SA.txt`、`逐层驻留专家级读取方案.md`、本文件；
  证据 `docs/bench/20260925-rk3588-moe-sa-protocol/`。
- 所有开关**默认关**（`VLLM_SA` / `VLLM_SA_DOWN` / `VLLM_SA_STAT` / `VLLM_SA_EVL`）；
  未设 `VLLM_SA` 时逐字节现状；x86 六档 TOKIDS 逐位相同 + `X64 CHECK PASSED`（16/0、9/9）。
- 板端 `/mnt/emmc/saprobe`：binary `90614b2e…`（含 EVL 与 `rss_anon/rss_file` 只读拆分探针），
  `build-rk3588/` 与仓库根已同步；源码备份 `/tmp/bak_sa/`（`.orig` = 未引入 SA 的原始版）。
- 证据清单：`pb_*`（§7B 判据臂）、`gate_*`（功能门）、`ev_*`（§8 判据臂）、`dg_*`（归因诊断）。
- 本地 17.7 GB 产物 `Modl/Qwen3-30B-A3B-q4local/` 为自检所需，验证后可按需删除。

**方法学反思**：预注册原定「P-EW-1 达标 + 用户批准后才实现」，实际因 N1 不可达而改为
「板端替代口径 + 先实现后验证」。代价 = 在核心前提未测量时实现了两代机制；收益 = 7 条硬结论 + 1 条可复用模型。
**教训：替代口径能证否机制，但不能证成方向——方向必须回到原判据端。**

---

## 2026-09-25 — 稀疏两个旋钮解耦（`--sparse-pf-k`）+ 管理页接线（附一处「磁盘 HTML 覆盖内嵌」的坑）

需求口径：**把解码与 prefill 的稀疏块数分开设置，默认都是 32**。此前两者共用 `--sparse-k`，导致没法「decode 用大 k 保质量 + prefill 用小 k 省时间」这样分开调。

### 1. 引擎：`--sparse-pf-k`（只作用于稀疏 prefill 核）

| 项 | 内容 |
|---|---|
| 新全局 | `int g_sparse_pf_k = 0;`（0 = **沿用 `--sparse-k`**，默认） |
| 新口径 | `int st_sparse_pf_k_eff(int n_blocks)`：`g_sparse_pf_k <= 0` 时**原样转发** `st_sparse_k_eff()`（含 `--sparse-ratio`），否则 `clamp(g_sparse_pf_k, 1, n_blocks)` |
| 唯一调用点 | 稀疏 prefill 分支里 `st_attn_batched_packed_sparse(...)` 的 k 实参（第 17830 行附近）。decode 侧的选块点仍走 `st_sparse_k_eff()`，一行未动 |
| CLI | `--sparse-pf-k N` |
| 回显 | `[OPT] ... sparse_pf_group=%d sparse_pf_k=%d ...` |

**默认档逐位不变**：`g_sparse_pf_k <= 0` 时 `st_sparse_pf_k_eff ≡ st_sparse_k_eff`，所以不传新开关的输出与改动前同一路径。

### 2. 管理页：4 个控件（解码 k / prefill 开关 / prefill k / prefill 选块粒度）

- 「解码 sparse-k」= 原 `sparse-k`（改名以免歧义），默认 32
- 「prefill 也走稀疏」= **env `VLLM_SPARSE_PREFILL`**（默认关；只有 env 门，所以走 `collectEnv()` → `config.json` 的 `env`，由 `vllm_mgr.py` 原样注入子进程）
- 「prefill sparse-k」= `--sparse-pf-k`，默认 **32**
- 「prefill 选块粒度」= `--sparse-pf-group`，**UI 默认给 4**（CLI 默认仍是 0 = 旧行为；0 会丢针，见 09-24 节，故 UI 不给 0 当默认，标签里写明推荐 4~8）

组装链：`admin.html` → `/admin/api/config/save`（通用 JSON 落盘，**无键白名单**，后端不用改）→ [`tools/ops/vllm_mgr.py`](tools/ops/vllm_mgr.py) 组装 argv（新增分支：`sparse_attn && sparse_prefill` 时追加 `--sparse-pf-k` / `--sparse-pf-group`）。

### 3. 验证（板端 RK3588）

| 检查 | 结果 |
|---|---|
| `[OPT]` 默认档 | `sparse_pf_group=0 sparse_pf_k=0`（未传新开关） |
| `[OPT]` 显式档 | `--sparse-pf-k 24 --sparse-pf-group 8` → `sparse_pf_group=8 sparse_pf_k=24` |
| `/admin/` 真取页面 | 200，107180 B，含 `f_sparse_k`×4 / `f_sparse_prefill`×5 / `f_sparse_pf_k`×4 / `f_sparse_pf_group`×4 |
| 内嵌副本 | `embedded_web.c` 还原后 153181 B，含上述字段（重新生成过） |

### 4. 一处必须知道的坑：**磁盘上的 `admin.html` 会盖掉内嵌副本**

`admin_html_path()` 的优先级是 `VLLM_ADMIN_HTML` → `./admin.html` → `/NewVLLM/admin.html` → **最后**才用内嵌副本。板端 `/NewVLLM/admin.html` 是 9-14 的旧版，所以只重新编译二进制**看不到新字段**。
处置：已备份 `/NewVLLM/admin.html` → `.bak_20260925`，并把新版同步过去。**以后改 UI 必须同时更新磁盘副本**（或删掉它，让内嵌副本生效）。

### 5. 边界与未决

- 两个新开关**默认都关**，且**不进任何预设**；理由与 09-24 节相同（稀疏 prefill 的 8B 复验还没做）。
- UI 默认粒度 4 只影响"用户主动开 prefill 稀疏"时；不动则在 CLI 层仍是 0（旧行为）。
- 板端 `candidate/src/serve/embedded_web.c` 已重生成；本地工作区未重生成（本地无板端分叉的那套改动，需在各自工作区各跑一次 `tools/build/gen_embedded_web.py`）。

### 6. 受影响文件

| 文件 | 改动 |
|---|---|
| `src/model/vllm_safetensors.c` | 新增 `g_sparse_pf_k` + `st_sparse_pf_k_eff()`；稀疏 prefill 调用点换口径（decode 侧不动） |
| `include/model/vllm_safetensors.h` | 导出 `g_sparse_pf_k`、`st_sparse_pf_k_eff()` |
| `src/main.c` | `--sparse-pf-k` 解析；`[OPT]` 回显加 `sparse_pf_k=%d` |
| `src/serve/admin.html` | 新增 4 个控件 + `collectForm`/`collectEnv`/`applyConfig`/`applyPreset` 接线 |
| `tools/ops/vllm_mgr.py` | 组装 `--sparse-pf-k` / `--sparse-pf-group`；schema 注释补 3 个键 |

### 7. 补记：x86 标量稀疏 prefill 核同步 `--sparse-pf-group` 分组选块

09-24 节的修复只落在 NEON 路径，`ST_ARCH_X86 && !ST_HAVE_NEON` 的标量核仍是「整批一个代表 query」。本次把同一分组逻辑**逐字同构**地补到 x86 标量核，使该开关在 x86 上真正生效（逐 token 主体一行未动）。

| 项 | 内容 |
|---|---|
| 改动函数 | `st_attn_batched_packed_sparse_neon`（`src/model/vllm_safetensors.c`，`ST_ARCH_X86 && !ST_HAVE_NEON` 标量分支） |
| 改动内容 | head 循环外计算 `grp`（`g_sparse_pf_group>0` 取值、clamp `[1,nb]`）；`q_rep` / `memset(used)` / probe / top-k / 最近块保底 / `ndl_trace_prefill` 一并套进 `for (int t_grp = 0; t_grp < nb; t_grp += grp)`；逐 token 主体范围改 `[t_grp, t_end)`；组循环结束再 `free` |
| 默认档位级不变 | `g_sparse_pf_group==0 ⇒ grp==nb ⇒ 组循环只跑 1 次`，`t_end==nb`，`q_rep = q_buf + ((size_t)(nb-1)*nh+ha)*hd` 与改前逐字相同；`memset`/选块/`ndl_trace_prefill` 仍整头各 1 次，token 循环范围 `[0,nb)` 不变 ⇒ 默认档产出与改前**位级一致** |
| 编译验证 | `powershell -ExecutionPolicy Bypass -File tools\build\build_x64.ps1 -OutDir C:\vllm_kestrel_x64_build` ⇒ `rc=0`，产物 `vllm_kestrel_x64.exe` 1272537 B（2026-09-25 09:23:12）。全量 37 条既有告警、0 条 `error:`；`vllm_safetensors.c` 的 9 条告警全在既有位置（876/879/964/8373/8561/8573/8631/10678），改动区 2117~2239 无任何告警 |
| 说明 | ① 记账函数 `ndl_trace_prefill` 是纯累加账本（原子 `NDL_ADD`，**无**「只打印一次」语义），每组调用一次不破坏语义，只使 `pf_calls` 按组数增加（与 ARM 一致）。② 环境坑：默认 `-OutDir <repo>\build-x64` 时 `ld` 报 `cannot open output file ...: No such file or directory` —— 因把含非 ASCII（`项目`）的绝对 `-o` 路径交给 ld，控制台代码页将其弄花；换纯 ASCII 输出目录后即成功，源码编译本身无错 |
| **自检回归门** | `powershell -ExecutionPolicy Bypass -File tools\build\check_x64.ps1 -OutDir C:\vllm_kestrel_x64_build` ⇒ `X64 CHECK PASSED`：`--test-l3` **PASS=16 FAIL=0**、`--test-sparse` **PASS=9 FAIL=0**（含 prefill 稀疏「全块 == 精确」与「top-4 负向」两条）。产物 1272537 B，sha256 `E63F2EB41A1809B11812C7D30A407563EE85349EC20533E6515E7FD5D867DDA2` |
| **运行期证据（G 是否真的生效）** | 本地 x86 真模型单发捞针（`--longctx-ctx 1024`、2B 本地转 q8、`VLLM_SPARSE_PREFILL=1`），只改 `--sparse-pf-group`，读 `[NDLTRACE] prefill-sparse: needle blk selected X/Y` 的 **Y**（= `g_ndl_pf_calls`，**每 head、每选块分组各 +1**）：<br>G=0（默认）→ **1344**；**G=1 → 344064（256×）**；**G=8 → 43008（32×）**<br>与 `Y = nh × ceil(nb/grp) × calls`（`nh=16`）精确吻合 ⇒ 反推 **`nb=256`**、`ceil(256/8)=32`；在 2048 / 4096 / 16384 三个 ctx 上 `Y_G8/Y_G0` **恒为 32.000**。⇒ **组循环确实在跑，该开关在 x86 上不再是静默空开关** |
| 复现命令 | 见 §7.1 |
| 顺带观察（**不可与板端结论互换**） | 同批运行的 recall 只有内置 `ramen` **单发宽松**判据、每格 1 样本：G=0 在 1024/2048/4096/16384 全 YES；G=8 在 1024/2048/4096 YES、**16K NO**（与板端「G=8 的失效点也在 16K」同向）。1024 档两臂皆 NO 是**判定伪影**：输出为 `"Ramen."`（首字母大写），而判定键 `ramen` 走大小写敏感的 `strstr`，模型其实捞到了 |
| 边界 | x86 已证「分组逻辑生效」+「默认档位级不变」+「自检无回归」。**未做**与板端同口径的端到端质量验证 —— 板端结论出自多轮累加 + 稀有码的**严格**判据，上表是单发**宽松**判据，两者不可互换（见 09-24 节 §5 口径提醒）。x86 标量核串行、无 SIMD，**性能不可与板端横比**（16K 单档本地耗时 ~7.5 min）。另：11 次引擎运行均正常退出，**未撞上**已知开放项「x64 偶发整请求卡死」 |

#### 7.1 复现命令（本地 x86）

```powershell
# 0) 模型：本地引擎只加载 VQF v2（safetensors 目录会被拒：[LONGCTX] FAIL: … 无 model.vqf）
cd e:\项目\GPT_VLLM\GitHupSRC\vqf_convert
gcc -O2 -Wall -Wextra -I src src\conv_main.c src\model_cfgio.c src\model_layers.c `
    src\quant_kernels.c src\qk_repack.c src\vqf_vision.c src\vllm_crypto.c `
    -o vqf_conv.exe -lm -lbcrypt
.\vqf_conv.exe --model "e:\项目\GPT_VLLM\Modl\Qwen3-VL-2B-Instruct" `
    --convert-vqf "e:\项目\GPT_VLLM\Modl\Qwen3-VL-2B-q8local\model.vqf" --wmode q8
# 4.25 GB safetensors → model.vqf 3191541768 B，~21 s；同参重跑 sum 一致（转换确定性）

# 1) 自检回归门（构建 + 16/0 + 9/9）
powershell -ExecutionPolicy Bypass -File tools\build\check_x64.ps1 -OutDir C:\vllm_kestrel_x64_build

# 2) G 激活性：三档只改 --sparse-pf-group，看 X/Y 里的 Y（期望 1344 / 344064 / 43008）
$env:VLLM_NEEDLE_TRACE="1"; $env:VLLM_SPARSE_PREFILL="1"
$env:VLLM_THREADS="4";       $env:OMP_NUM_THREADS="4"
$BIN="C:\vllm_kestrel_x64_build\vllm_kestrel_x64.exe"
$M="e:\项目\GPT_VLLM\Modl\Qwen3-VL-2B-q8local"
$C="e:\项目\GPT_VLLM\prefill_lab_20260923\corpus_manual.txt"

& $BIN --model $M --longctx-ctx 1024 --needle-pos 50 --longctx-corpus $C --sparse-attn --sparse-k 32
& $BIN --model $M --longctx-ctx 1024 --needle-pos 50 --longctx-corpus $C --sparse-attn --sparse-k 32 --sparse-pf-group 1
& $BIN --model $M --longctx-ctx 1024 --needle-pos 50 --longctx-corpus $C --sparse-attn --sparse-k 32 --sparse-pf-group 8
```

注意事项（两条会让这一步白跑）：

- **别用 `--test-sparse` 自检来判别 G**。该用例里 `q_buf[t] = q[h] * (1 + 0.01*t)` 是**同一方向的标量缩放**，
  而 probe 选块是 argmax，对 query 正缩放不变 ⇒ 换哪个 token 当代表 query 都选出同一组块，
  `prefill sparse top-4 … max diff` 在 G=0/1/4 下**恒为 `1.713e-01`**，天然无法区分。必须用真 prefill + `VLLM_NEEDLE_TRACE`。
- `grp` 被 clamp 到 `nb`，所以要先确认 `nb`（prefill mini-batch 大小，本机为 256）。若 `nb <= G`，档位与 G=0 等价、Y 不变 —— **不是**没生效。

---

## 2026-09-24（长上下文）— 稀疏 prefill 丢针：根因在「写入侧」，修法是按 query 分组选块（`--sparse-pf-group`）

`VLLM_SPARSE_PREFILL=1`（让 prefill 也走稀疏核）在板端一直是「更快但答错」：2B 多轮真实手册语料检索在 **4K** 起丢针（5 轮 **2/5**），8B 到 **16K** 丢（5 轮 **4/5**）。§10.3 / §15 已据此把它默认关，但归因停在「稀疏近似本身损坏 hidden state」这一层，没有可修的抓手。本轮把它拆到**代码级**，定位到一个**具体的口径缺陷**，并给出 opt-in 修复：**质量回来了，代价 +13.8%（8K）**。

### 0. 观测口径

Qwen3-VL-2B / 板端 RK3588（4×A76@cpu4-7，`taskset -c 4-7`）/ 内置捞针任务
`--longctx-quality --longctx-ctx {4096,8192,16384} --needle-pos 50 --longctx-corpus corpus_manual.txt`
+ `--sparse-attn --sparse-k 32 --sparse-block 32` + `VLLM_SPARSE_PREFILL=1` + `VLLM_NEEDLE_TRACE=1`（**只读**插桩，不动计算路径）。

> 口径提醒：内置针是 `Zara likes the color teal and her favorite food is ramen.`（14 token），
> 判据要生成 `ramen` —— **高频、可预测**，比本项目的多轮 `KX2Bxxx` 稀有码检索**宽松得多**。
> 所以「内置任务判 YES」不等于「多轮检索也过」，两者必须分开记。

### 1. 三档实测：针块命中率与结果单调相关，decode 选块与结果**不相关**

| ctx | prefill 稀疏核「针块被选进 top-k」 | prefill 墙钟 | decode `MISSED` | decode `imp_rank`（预算 16） | 召回 |
|---|---|---|---|---|---|
| 4K | **42.93%** (1731/4032) | 86.6 s | 10811/14336 | 24/128 `NO` | YES |
| 8K | **32.17%** (2450/7616) | 184.4 s | **0/14336** | **15/256 `YES`（预算内）** | **NO** |
| 16K | **15.00%** (2218/14784) | 392.2 s | 12200/14336 | 24/512 `NO` | **NO** |

**8K 那一行是判别点**：`MISSED 0` + `imp-half 14336` + `imp_rank 15 < 预算 16`
⇒ **针块在每一次 decode 调用里都被 attend 到了，模型仍然答 `...her favorite food is not specified.`**

反向的一行同样关键：4K 档 decode **75% 的调用没看到针块，却答对了**。

⇒ 两个候选原因被分开：
- **decode 选块的好坏与结果不相关**（75% miss 答对 / 0% miss 答错）；
- **prefill 核的针块命中率与结果单调相关**（43%→32%→15% 对应 YES→NO→NO）。

这也独立复现了 §10.4 的判据（当时是在 16K 上用 `--force-blk` 强塞针块，`MISSED 0` 仍答错）。
顺带排除两条常见误判方向：`[OPT]` 全批次回显 `disk_kv=0 l3_evict=0 l3_prefix_reuse=0`
⇒ **与 L3 落盘无关**（丢针档反而**更快**：8K 稀疏 184.4 s，16K 392.2 s，没有读盘等待）；
`prefix_kv=1` 在通过档与失败档都开 ⇒ 与内存前缀复用无关。

### 2. 根因：稀疏 prefill 核让「一个 query 替整批 token 选块」，而 prefill 输出就是写进 KV 的 K/V 来源

代码位置：`st_attn_batched_packed_sparse_neon` → `vllm_attn_sparse_worker`（NEON 稀疏 prefill 核）。

```c
/* representative query: last token of the mini-batch */
const float *q_rep = c->q_buf + ((size_t)(c->nb - 1) * c->nh + ha) * c->hd;
```

- 用**批内最后一个 token** 作代表 query，给整个 mini-batch（板端实测 **nb = 256**，
  由 §4 的记账分母反推：G=1 时分母 1949696 = 7616 × 256）选一套 top-k 块；
- 随后每个 query 的 softmax **只在共享的选中集上重归一化**。

所以**批内 255 个 token 用别人的上下文写自己的表示**。而 prefill 的 attention 输出正是
**写进 KV 的 K/V 的来源** —— 污染发生在**写入侧**，不是读取侧。这解释了为什么
「decode 把针块塞进来（`MISSED 0`）也答不对」：读到的 K/V 从源头就是坏的。

命中率随上下文**单调下降**（43%→32%→15%）是因为共享窗口固定为 `k×bs = 1024` token，
上下文越长，越装不下「每个 token 各自需要的上下文」。

> 一处口径更正：`--sparse-ratio` 那一类「按比例保留」与本节无关；而
> 「只在选中集上重归一化」是 H2O / SnapKV 一类的**标准做法**，**不算缺陷**。
> 主嫌是**代表 query 太粗**（1 个 query 代表 256 个），不是归一化口径。

### 3. 修复：`--sparse-pf-group G`（默认 0 = 历史行为）

把「整批一个代表 query」改成「每 G 个 token 一组、组内最后一个 token 作代表」：

```c
/* 0（默认）= 整批一个代表 query（逐位保持旧行为）；G=1 = 每个 token 自选 */
int grp = g_sparse_pf_group > 0 ? g_sparse_pf_group : c->nb;
for (int t_grp = 0; t_grp < c->nb; t_grp += grp) {
    int t_end = t_grp + grp; ...
    const float *q_rep = c->q_buf + ((size_t)(t_end - 1) * c->nh + ha) * c->hd;
    /* probe → top-k → 最近块保底 → 逐 query 计算 */
}
```

代价模型：probe 成本 × `ceil(nb/G)`。`G=1` 时 QK ≈ 全量的 `n_probe/bs = 8/32 = 25%`，
PV 仍只有 `k*bs/seq = 6.2%` ⇒ 设计上仍有 3~4× 空间。

**7 处编辑**（3 文件）：`src/model/vllm_safetensors.c` 4 处（新全局 + worker 分组循环的开口/闭口/循环范围）、
`include/model/vllm_safetensors.h` 1 处（导出）、`src/main.c` 2 处（`--sparse-pf-group` 解析 + `[OPT]` 回显
新增 `sparse_pf_group=%d`，避免"诊断开关静默"）。

### 4. 实测：质量回来，代价 +13.8%

2B / 8K / 内置任务 / 稀疏 prefill（唯一变量 `--sparse-pf-group`）：

| G | 组数/批 | Prefill | 相对 G=0 | 记账分母（=7616×组数，证明生效） | 针块命中率 | 召回 |
|---|---|---|---|---|---|---|
| 0（旧行为） | 1 | 184.4 s | — | 7616 | 32.17% | **NO**（`is not specified`） |
| **8** | 32 | 209.8 s | **+13.8%** | 243712 | 33.34% | **YES** |
| 4 | 64 | 245.3 s | +33.0% | 487424 | 32.96% | **YES** |
| 1 | 256 | 456.7 s | +148% | 1949696 | 34.43% | **YES**（生成最干净） |

命中率几乎不变（32%~34%）而结果反转，**不是矛盾**：命中率是**全批所有 token** 的汇总，
文档里绝大多数 token 本就不需要看针；变的是**谁能看什么** —— 旧行为强迫整批 256 个 token
共用「批内最后一个 token（问句性质）」选出的块，现在每组自选，文档 token 拿回了自己的局部上下文。

又一次证伪「选块论」：G=0 时重要性表把针块排在 **15/256（预算内）** 却答错；
G=8/4 时针块**掉出** top-16（`imp-half 0`），靠 probe-half 命中 24% 反而答对。

**零回归**：同一新二进制上 G=0 逐项复现改前的数（`2450/7616 (32.17%)`、`MISSED 0`、
184.4 s vs 184.0 s，0.5% 噪声内）⇒ 默认档行为不变。

### 5. 代价与边界（照实标注）

- **收益区在长上下文，不在短上下文**（2026-09-25 补测：8K/16K 的精确 prefill 基线已补上）：

  | 2B / 内置捞针（needle@50%、真实语料） | 4K | 8K | 16K |
  |---|---|---|---|
  | 精确 prefill | 105.5 s（YES） | 345.1 s（YES） | **1111.6 s**（YES） |
  | 稀疏 **G=0**（旧行为） | 86.6 s（YES） | 184.4 s（**NO**） | 392.2 s（**NO**） |
  | 稀疏 **G=8** | — | **240.0 s**（**YES**） | **567.5 s**（**YES**） |
  | 稀疏 G=4 | — | 245.3 s（YES） | 722.1 s（YES） |

  ⇒ 修复后 **G=8 相对精确档：8K 1.44×、16K 1.96×**，质量同为 YES（旧的坏配置是 1.87×/2.83×，
  修质量花掉约 1/3 的加速）。4K 单发 A/B 只有 1.22×，G=8 的 +14% 基本吃光 ⇒ **甜点在 8K 以上**。
  G=4 在 16K 反而比 G=8 慢 27%（probe 成本 ∝ 组数），**G=8 是这几档里的甜点**。
- **严格判据也验了**（2B 多轮真实语料 + 稀有码 `KX2Bxxx` 逐字检索，5 轮累加到 16K）：

  | 档 | 1K | 2K | 4K | 8K | 16K | 合计 |
  |---|---|---|---|---|---|---|
  | 精确 prefill | Y | Y | Y | Y | Y | 5/5 |
  | 稀疏 G=0（旧） | Y | Y | N | N | N | 2/5 |
  | 稀疏 **G=8** | Y | Y | Y | Y | **N** | **4/5** |
  | 稀疏 **G=4** | Y | Y | Y | Y | **Y** | **5/5** |

  ⇒ **失效点从 4K 推到 16K（G=8）/ 完全消除（G=4）**。G=4 在 16K 轮 prefill 425.7 s
  vs 精确档同轮 734.3 s ⇒ 仍快 1.72×。代价（相对 G=0）：1K +23% → 16K +58%（probe 成本 ∝ n）。
- **x86 标量核已同步移植**：`ST_ARCH_X86 && !ST_HAVE_NEON` 的标量稀疏核
  （`st_attn_batched_packed_sparse_neon` 的同名 x86 分支）**已补上同一分组选块逻辑**（2026-09-25 补），
  在 x86 上 `--sparse-pf-group` 现在生效，且**已有运行期证据**：`VLLM_NEEDLE_TRACE` 打出的
  `prefill-sparse … X/Y` 里的 Y（= 每 head、每选块分组各 +1 的 `g_ndl_pf_calls`）在 G=0/1/8 下为
  **1344 / 344064 / 43008**，比值精确等于 `1 : nb : ceil(nb/8)`（反推 `nb=256`），且在
  2048/4096/16384 三个 ctx 上 `Y_G8/Y_G0` 恒为 32.000（见 2026-09-25 节 §7）。
  板端（aarch64+NEON）仍是唯一做过**与本节同口径的端到端质量验证**的平台；
  x86 只做了自检回归（`check_x64.ps1`：16/0 + 9/9）+ 分组激活性 + 默认档位级不变，
  **未做同口径质量验证**。
- **口径提醒（自证一遍）**：内置 `ramen` 任务在 16K 判 YES，而严格的多轮稀有码判 NO ——
  **判据难度不同，结论不能互换**。凡"可用"结论都必须过严格判据。

### 6. 尚未完成（下一步）

1. **8B 复验**（唯一剩下的大缺口：8B 的 `G>0` 稀疏 prefill 一档都没跑；另外 8B 纯 `sp32`
   的 16K 点也缺 —— 当时按"先不测精准基线"的口径在 turn5 前砍掉了）；
2. 四项都过之前，`VLLM_SPARSE_PREFILL` 仍**默认关**，`--sparse-pf-group` / `--sparse-pf-k`
   只是可选的实验开关（管理页已接线，但默认关、不进预设，见 09-25 节）。

### 7. 施工方法与留档（含一处必须知道的隐患）

- **本地 `GitHupSRC` 与板端 `candidate/` 已经分叉**：本地多 553 行、板端多 12 行
  （如 `const int TILE = 3;`、`g_st_q4_repack`）。⇒ 「改本地 → scp 整文件」这条老路会**互相抹掉**对方的改动。
- 本轮改用**带唯一性断言的定点替换脚本**（板端 `apply_pfgroup.py`，7 处编辑各断言"恰好命中 1 次"，
  行尾自适应 CRLF/LF：板端 `main.c`/`.h` 是 CRLF、`vllm_safetensors.c` 是 LF），
  先 `--check` 全绿再写入。**建议后续专门做一次本地↔板端双向对账**。
- 留档（板端 `prefill_lab_20260923/`）：改前二进制 `candidate-build/vllm_kestrel_pre_pfgroup`；
  改前源码 `backup_pre_pfgroup/`；日志 `ndlprobe/{precise,sparse,sparse_8192,sparse_16384}.log`、
  `pfgroup/G{0,8,4,1}.log`；脚本 `ndl_probe.sh`、`ndl_probe2.sh`、`pfgroup_sweep.sh`、`apply_pfgroup.py`。

### 8. 受影响文件

| 文件 | 改动 |
|---|---|
| `src/model/vllm_safetensors.c` | 新增 `int g_sparse_pf_group`（含依据注释）；`vllm_attn_sparse_worker` 选中集从「整批一个代表 query」改为「每 G 个 token 一组」（默认 0 = 旧行为，逐位不变） |
| `include/model/vllm_safetensors.h` | 导出 `g_sparse_pf_group` |
| `src/main.c` | 新增 `--sparse-pf-group G` 解析；`[OPT]` 回显新增 `sparse_pf_group=%d` |

---

## 2026-09-16（长上下文）— 「大海捞针」在 `--sparse-attn` 下失败的三个病因与一处零算力修复

Part F 的长上下文针检索在 `--sparse-attn` 下会失败（同一条提示，精确注意力判 YES、稀疏判 NO），且位置越靠中间越差。本轮把它拆到可逐条证伪：**三个独立病因** —— 两个引擎缺陷、一个判据缺陷。修复后 16K 默认档（不设任何 env）两处针位都能在 decode 内命中，且**无不可测量以外的代价**。

### 0. 观测口径

Qwen3-VL-2B / 16K / `--sparse-attn --sparse-k 32 --sparse-block 32 --sparse-probe 8`。先加 `VLLM_NEEDLE_TRACE=1` **只读**插桩（不触碰任何计算路径），把针块在**两处稀疏核**（prefill 核与 decode 核）的去向分别记账。名次用与选块**完全相同**的比较口径算（值大者靠前，同值块号小者靠前），所以打出来的名次就是选块当时看到的那个名次。

> 计数曾出现 `selected 14154 / calls 14139` 这类自相矛盾的值，根因是 decode 的 head 循环走 `vllm_tp_parfor` 并行、非原子的 `++`/`+=` 丢更新；账本全部改成 `__atomic_fetch_add`（min 用 `__atomic_compare_exchange_n`）后才可信。

### 1. 病因 1：稀疏档下 prefill 也走稀疏核，把「重要性表」喂成了 probe 的自证闭环

`--sparse-attn` 在 **prefill 阶段同样调用稀疏核**，而稀疏核只对**被选中的块**累加 `imp_head`。于是 `prefill_importance[s]` 记的不是「位置 s 的注意力质量」，而是「**probe 恰好也选中块 b(s)** 的那部分注意力质量」。decode 选块的 (1b) 步又按这张表发掉一半预算 ⇒ **用 probe 的结论再选一遍 probe 喜欢的块**。

16K / k=32 / pos=25%，单变量 A/B（其余参数完全相同）：

| prefill 内核 | 针块 imp 名次 | 针块占比 | `ratio_needle/thr` | decode 重要性半命中 | decode 探针半命中 | 召回 | Prefill | Decode |
|---|---|---|---|---|---|---|---|---|
| **稀疏**（旧） | 188/513 | 0.2882% | 0.514 | **0/14336** | 220/14336 | **NO** | 352,366 ms | 53.8 ms/tok |
| **精确**（新） | **12/513** | 0.6386% | 1.068 | **14336/14336** | 0/14336 | **YES** | 274,360 ms | 49.3 ms/tok |

pos=50% 同向：`303/513 → 148/513`、占比 `0.2442% → 0.3176%`、重要性半 `0 → 仍 0`（148 超过 16 的预算）、探针半 `4 → 35`、召回 `NO → YES*`（* 由 prefill 兜住第 0 个 token，非 decode 之功，见 §5）。复现腿逐位复现（连 `PPL 6.651/6.759` 与 greedy 串都相同）⇒ 该开关是干净单变量。

**顺带发现：稀疏 prefill 在 x86 上反而更慢。** 同二进制同提示的 ATTN 分解：

```
稀疏 prefill（旧）：total 352115 ms | GEMM  91145 (25.9%) | ATTN 256740 (72.9%)
精确 prefill（新）：total 274119 ms | GEMM  92550 (33.8%) | ATTN 177236 (64.7%)
```

稀疏 prefill 的 ATTN 多 45%、总 prefill 多 28%。结构性原因：**x86 走的是标量核**（`st_attn_batched_packed_sparse_neon` 在 x86 是标量路径），探针每个 head、每个 mini-batch 都要扫全 K 缓存，且只用 **1 个代表 query**（无 query 复用），是纯带宽型开销。

> **⚠ 这条只对 x86 成立，不要外推。** 板端 RK3588 的 NEON 稀疏核实测比精确 prefill **快 4.0×**（455 s vs 1816 s @16K，见 §10）。所以「稀疏 prefill 更慢」不是一个普遍事实，默认关它的**主理由必须是质量**（稀疏 prefill 污染重要性表 → decode 丢针 → 答错），而不是速度。

### 2. 病因 1b：`--sparse-k` 是「固定块数」而非比例，保留率随长度线性塌陷

`k=32`、`bs=32` 时：2K → 保留 49%、8K → 12.5%、16K → **6.2%**。长上下文下模型实际只看到极小一部分上下文。

新增 `--sparse-ratio R`（默认 **0 = 关闭**，`--sparse-k` 保持固定块数语义）：`k_eff = clamp(ceil(R * n_blocks), g_sparse_k, n_blocks)`，即 `--sparse-k` 退化为**下限**。decode 与稀疏 prefill 两处选块都改走同一个 `st_sparse_k_eff()`，保证与诊断口径一致。

### 3. 病因 0（本轮关键）：`prefill_importance` 的「整段上下文全部 query 求和」口径带强位置偏置

这张表的用途是**估计 decode 的 query 想 attend 哪些块**，而旧口径是**整段上下文所有 query 的注意力质量之和**。开头与近邻的块会被更多 query 顺带 attend 到，于是**同一个针块**的名次随位置单调劣化：

| 针位 | 针块名次 | 针块占比 | `ratio_needle/thr` | decode 重要性半 | Decode | Prefill | 召回 |
|---|---|---|---|---|---|---|---|
| pos 25% 旧口径 | 12/513 | 0.639% | 1.068 | 14336/14336 | 49.3 ms/tok | 274.1 s | YES |
| pos 25% **新口径** | **4/513** | **5.164%** | **12.46** | 14336/14336 | 49.1 ms/tok | 275.8 s | YES |
| pos 50% 旧口径 | **148/513** | 0.318% | 0.532 | **0/14336** | 48.5 ms/tok | 272.9 s | YES* |
| pos 50% **新口径** | **6/513** | **0.801%** | **2.298** | **14336/14336** | 49.3 ms/tok | 277.4 s | YES |

（*pos 50% 旧口径那个 `YES` 是 **prefill 兜住第 0 个 token**，不是 decode 选块之功 —— 旧的判据不分这两种情况，见 §5。就「decode 是否看见针块」而言，旧口径在 16K @50% 是**全丢**：重要性半 `0/14336`。）

修复：`VLLM_IMP_LAST_MB`（**默认开**，设 `=0` 回退旧口径）—— prefill 重要性**只统计最后一个 mini-batch** 的注意力质量。理由：最后一个 mini-batch 恰好装着问句与指令，它的注意力分布最接近紧接着要 decode 的那个 query。**代价 = 每个 prefill 多一次 `imp_head` 清零**（`nh * max_kv_slots` 个 float ≈ 1 MB），**decode 侧一行未改、算力零增加**。

### 4. 保留率扫描：`--sparse-attn` 的甜点是口径修复后才回来的

`--sparse-ratio` 关闭、`VLLM_SPARSE_PREFILL=0`，单引擎，16K：

| k | 保留率 | 针块名次 | 重要性半预算 | 重要性半命中 | Decode | greedy 首段 |
|---|---|---|---|---|---|---|
| 32 | 6.2% | 12/513 @25% | 16 | 14336/14336 | **49.3** ms/tok | `ramen Why is this answer incorrect?…` |
| 128 | 25% | 12/513 @25% | 64 | 14336/14336 | 69.3 ms/tok | `ramen I need to find…` |
| 256 | 50% | 12/513 @25% | 128 | 14336/14336 | 93.1 ms/tok | `ramen I need to find…` |
| 32 | 6.2% | 148/513 @50% | 16 | **0** | 48.5 ms/tok | — |
| 256 | 50% | 148/513 @50% | 128 | **0**（148 > 128） | 96.4 ms/tok | `ramen I need to determine…` |
| 无 sparse | 100% | 12 → 148 | — | — | **100.6** ms/tok | `ramen I'm sorry, but…` |

**在旧口径下**，中段针需 `k ≥ 295`（`(k+1)/2 ≥ 148`，即 ≥58% 保留）才够，而那时 decode 已 96.4 ms/tok ≈ 全量精确的 100.6 ms/tok ⇒ **旧口径在 16K 上不存在收益区间**。新口径（§3）把中段针的名次从 148 拉到 6，`k=32` 即够用 —— 甜点由口径修复夺回，**不是靠加 k**：16K 默认档 `k=32` 是 **49.3 ms/tok vs 全量精确 100.6 ms/tok，decode 快约 2×**。

同时修正一处我先前的归因错误：k=32 的输出退化**不是**保留率低造成的 —— k 从 32 提到 256 只是让模型「答出第一个 token 后开始找」，并没有变可用（输出质量见 §9）。

**8K 档（同口径、单引擎）**：

| 8K @50% | Prefill | Decode | 针块名次 | 重要性半 | 召回 |
|---|---|---|---|---|---|
| `--sparse-k 32`（默认，`--sparse-ratio 0`） | 108.7 s | **42.4** ms/tok | 6/257 | 28672/28672 | YES |
| `--sparse-ratio 0.25`（k_eff=65） | 106.4 s | 49.8 ms/tok | 6/257 | 28672/28672 | YES |
| `--sparse-ratio 0.50`（k_eff=129） | 103.9 s | 61.0 ms/tok | 6/257 | 28672/28672 | YES |
| 全量精确（无 `--sparse-attn`） | 92.1 s | 62.4 ms/tok | 6/257 | — | YES |

⇒ **8K 档 `k=32` 比全量精确 decode 快 32%**；`--sparse-ratio` 在 8K 与 16K 都**只把 decode 拖慢**（42.4 → 49.8 → 61.0 ms/tok），却换不到任何召回提升 —— 口径修复后针块在**最小预算**下就进得来。**故默认值保持 0**。四条腿的 prefill 一列散布 92–109 s，而它们全部走精确核、参数相同 —— 这个 15% 的散布本身就是机器噪声（见 §7）。

### 5. 病因 2：判据截断造成假失败

原判据 `got = strstr(buf_out, "ramen")`，而 `--longctx-gen` 默认 32 token —— **答对但没答完**会被判 NO。实测 2K @pos 0%：gen=32 判 NO；gen=64 输出完整正确推理（`…The text states: "Zara likes the color teal and her favorite food is ramen." From this information, we can directly infer that Zara's favorite food is ramen.`）。

判据改为记录**首次命中发生在第几个生成 token**（`got_at`）：第 0 个 token 的 logits 由 prefill 产出（最后一个 prompt token 的精确注意力），后续才是 decode 自己捞的，两者混在一起会把「prefill 兜住」误判成「decode 选块对」。据此输出 `(hit at token 0 = prefill 兜住…)` / `(hit inside decode)`，并在 `got_at == 0 && GEN_MAX <= 32` 时打 WARN 建议提到 gen=64。

### 6. 默认档端到端回归（`ndl_x64h`，**不设任何 env**）

| 场景 | 针块名次 | 重要性半 | decode 重要性半命中 | 召回 | Decode |
|---|---|---|---|---|---|
| 2K @0%，gen=64 | **1/65** | YES（占比 100%，`ratio/thr = 177.2`） | 28672/28672 | **YES (hit inside decode)** | 38.4 ms/tok |
| 16K @50%，gen=32 | **6/513** | YES（`ratio/thr = 2.298`） | 14336/14336 | **YES (hit inside decode)** | 52.7 ms/tok |
| 16K @25%，gen=64 | **4/513** | YES（占比 5.164%，`ratio/thr = 12.46`） | 28672/28672 | **YES (hit inside decode)** | 50.4 ms/tok |
| 16K @50%，gen=64 | **6/513** | YES（`ratio/thr = 2.298`） | 28672/28672 | **YES (hit inside decode)** | 51.2 ms/tok |

与显式开 env 的两腿一致 ⇒ 默认值确实生效。插桩同时报 `prefill-sparse: needle blk selected 0/0` —— 稀疏 prefill 已不再被调用，符合 §1 的默认关。

16K 两处针位在 gen=64 下**全部 64 个 decode 步都带着针块**（`MISSED 0`），但**输出仍未在 64 token 内落到最终答案**（`…The text provided is a list of facts about various people… However, there is no`）—— 详见 §9，这是模型行为而非选块问题。

### 7. 代价声明（照实标注）

- `VLLM_IMP_LAST_MB` 的 prefill 侧开销是「每 prefill 一次 1 MB 清零」。同配置重复运行 Prefill **274.1 / 275.8 / 277.4 / 300.8 s**、Decode **48.5 / 49.1 / 49.3 / 52.7 ms/tok** ⇒ **本机（AMD 9800X3D）噪声 7–10%**。改动前后的差异落在这个噪声内，正确说法是**「无可测量代价」，而不是「精确到 ±1%」**。
- 稀疏 decode 路径（选块与注意力核）本轮**未改动任一行**；`VLLM_SPARSE_PREFILL` 只切换 prefill 走哪个已存在的核。
- §1 的 ATTN 分解是 x86 标量核的数据。板端为 NEON 核、常数不同，**不可直接外推**。

### 8. 受影响文件

| 文件 | 改动 |
|---|---|
| `src/model/vllm_safetensors.c` | 新增 `VLLM_NEEDLE_TRACE` 只读插桩（`st_ndl_trace_reset/report`、`ndl_trace_prefill/decode`，原子账本）；新增 `g_sparse_ratio` / `st_sparse_k_eff()`；两处选块点改走 `st_sparse_k_eff()`；稀疏 prefill 默认关闭（`VLLM_SPARSE_PREFILL`）；prefill 重要性默认只统计最后一个 mini-batch（`VLLM_IMP_LAST_MB`） |
| `src/main.c` | `--sparse-ratio` CLI 与 `[OPT]` 回显；`[NDLTRACE]` 诊断块改走 `st_sparse_k_eff()` 同一口径；`got_at` 判据修复与 WARN；`st_ndl_trace_reset/report` 调用点 |
| `include/model/vllm_safetensors.h` | 导出 `g_sparse_ratio`、`st_sparse_k_eff()` |

### 9. 未决与边界

- **生成质量的复述行为（x86 Q4 档）**：新口径下 16K 输出仍是 `ramen I need to determine what Zarra's favorite food is based on the provided information. I will go through the given text to find the relevant information…`，饶一圈才落到答案，gen=64 也没收口。**精确注意力档的开头完全相同**（16K @25%、无 sparse：`ramen I need to find the information about Zarra's favorite food. I will go through the provided text and look for the name "Zara"`）⇒ 与选块无关。**注意这条是 Q4 权重档的现象**：同一提示、同一代码，板端 2B **Q8** 在 16K @50% 直接输出 `Based on the information provided, Zara's favorite food is ramen. Therefore, the answer is: ramen.` 干净收口（§10）⇒ 更可能是 Q4_0 量化下的行为差异，需单独立项，不要写成「2B 模型固有」。
- **`VLLM_IMP_LAST_MB` 是启发式变更**，同时影响 `l3_evict_layer` 的冷块淘汰。若某任务「最后一个 mini-batch 不是 query」（纯续写、无问句），可能不如旧口径。备选：**两表取并集**（新旧口径都进不了预算才丢，代价每 prefill 多 1 MB 表、仍零算力）。
- **板端复验已做（同日）**：见 §10。`VLLM_SPARSE_PREFILL=1/0` 的 ATTN 分解、以及重要性口径在真硬件上的复现都拿到了，且**结论与 x86 不同向**。
- **已分离（同日，见 §10.5）**：先前「稀疏 prefill 答错究竟是选块失败还是 hidden state 被损坏」这个二选一，已用 `--force-blk` 钩子判死：**损坏在 prefill 侧**。即便把针块强制保留、让它在全部 28672 次 head 调用里都被 attend，模型仍答「文本里没有这个信息」。所以那 4.0× 的板端 prefill 提速**不可能靠修重要性表/修选块换回**；也作废了先前设想的「只让最后一个 mini-batch 走精确核以清洁重要性表」那条路。
- **x86 8K/16K 的 `--sparse-ratio` 已量**（§4）：8K 下 42.4 → 49.8 → 61.0 ms/tok 单调变慢、召回零提升 ⇒ 默认 0。

### 10. 板端复验（RK3588 / Orange Pi 5 Plus，aarch64）

**同步与构建**：本地 x86 工作区生成 LF 补丁（`git diff --output=`，注意 PowerShell 的 `>` 会把内容写成 UTF-16LE，必须用 `--output`），`scp` 到板端 `kestrel_pull` 后 `git apply --check` → `git apply` 干净通过；板端 `git diff --stat` 与本地**逐字相同**（`499 insertions(+), 20 deletions(-)`）。`build_rk3588.sh` 构建成功（933,032 B，较旧 920,712 B +12,320 B），`--test-sparse` **9 PASS / 0 FAIL**、`--test-l3` **0 FAIL**。

模型口径差：板端只有 2B **Q8**（`Qwen3-VL-2B-q8fix`，日志 `Q4_0 quantization: NOT active`），x86 实验用 **Q4**。板端内部 A/B 自洽，但**与 x86 的绝对值不可直接比**。

**10.1 功能复验（不设任何 env）**

| 场景 | 针块名次 | `ratio_needle/thr` | decode 重要性半 | 召回 | Prefill | Decode |
|---|---|---|---|---|---|---|
| 2K @0% | **1/65** | 173.4 | 28672/28672 | **YES (inside decode)** | 56.2 s | 119.1 ms/tok |
| 16K @50% | **6/513** | 1.964 | 28672/28672 | **YES (inside decode)** | 1815.7 s | 216.9 ms/tok |

16K 的 greedy 输出是**干净的正确答案**：`Based on the information provided, Zara's favorite food is ramen. Therefore, the answer is: ramen.` —— 与 x86 Q4 档「饶一圈复述」的行为不同（§9 已据此改写归因）。

**10.2 反转：稀疏 prefill 在板端快 4.0×（与 x86 相反）**

| 16K @50% | Prefill | GEMM | ATTN | 针块 imp 名次 | 重要性半 | 召回 |
|---|---|---|---|---|---|---|
| **精确 prefill**（默认，`VLLM_SPARSE_PREFILL=0`） | 1815.7 s | 237.9 s (13.1%) | **1564.3 s (86.2%)** | **6/513** | 28672/28672 | **YES** |
| **稀疏 prefill**（`VLLM_SPARSE_PREFILL=1`） | **455.2 s** | 243.5 s (53.7%) | **197.8 s (43.6%)** | **510/513** | **0/28672** | **NO** |

稀疏档的 ATTN 只有精确档的 **12.6%**，总 prefill **快 4.0×**（GEMM 基本相同，差异全在 ATTN）。这直接推翻 §1 里从 x86 外推的那条：**「稀疏 prefill 更慢」是 x86 标量核的性质，不是普遍事实。** 板端瓶颈恰恰是那一项 —— 16K 精确 prefill 的 86.2% 花在 NEON attention 上（x86 同配置只有 64.7%）。

稀疏档的召回是 **NO**，且失败形态比 x86 更彻底：`imp: needle_blk=0 share=0.0000% rank=510/513`、`imp_rank avg=509.0`、探针半仅 `19/28672`，greedy 直接答 `The text provided does not contain any information about Zarra's favorite food.` —— 这是病因 1（稀疏 prefill 污染重要性表 → probe 自证闭环）在板端的复现。

**10.3 因此默认值怎么定**

`VLLM_SPARSE_PREFILL` 默认**仍是关**（= 精确 prefill），但**理由换成质量而非速度**：

- 想拿板端那 4.0× 的 prefill 提速，代价是 16K 中段针**彻底捞不到**（rank 510/513）。对长上下文检索类任务，这是「更快但答错」，不可接受。
- 也不是「k 调大就行」：稀疏 ATTN 工作量 ∝ k，`k=32`（6.2% 保留）已是 197.8 s，线性外推到 `k=256`（50%）约 1.58×10³ s，**与精确档的 1564.3 s 基本持平** —— 稀疏档的收益只存在于极小 k，而小 k 正是质量崩掉的那一档。
- **16K 在板端 prefill 要 30 分钟**这个事实，是「板端不适合原生 16K 长上下文」的结论，**不是本修复引入的代价**（默认档就是精确 prefill，与修复前同一路径）。

**10.4 旧口径失败是长度相关的（8K 旧口径腿）**

板端 8K @50%、`VLLM_IMP_LAST_MB=0`（旧口径）、精确 prefill：

| Prefill | ATTN | 针块 imp 名次 | `ratio_needle/thr` | 重要性半 | 探针半 | 召回 |
|---|---|---|---|---|---|---|
| 491.6 s | 363.3 s (74.0%) | **63/257** | **0.6796**（<1，进不来） | **0/28672** | 2392/28672 | **YES（勉强，靠探针半兜住）** |

⇒ 旧口径在 8K 时重要性半已经全丢（`0/28672`），但探针半还能以 8.3% 的命中率勉强把针捞回来；到 16K（x86：rank 148/513；板端：rank 510/513）探针半也掉到 0.03–0.07%，两半全丢 ⇒ 召回 NO。这解释了历史观测里「0% 中、25% 不中」那种非单调召回：**它不是随机，而是随长度/位置单调劣化，只是短上下文下被探针半兜住了**。

**10.5 归因分离：稀疏 prefill 的损坏在 prefill 侧，不在选块侧**

§10.2 那两条腿有一个二选一没答：稀疏 prefill 答错的根因，是「decode 没选中针块」，还是「稀疏近似注意力把针块自身的 hidden state/KV 改坏了」？本轮新增了一个**只作用于 decode 选块**的钩子 `--force-blk B[,B...]`（`g_force_blk`，无头默认关闭）来判它 —— 注意**不能**用 `--sparse-ratio` 做这个分离，因为它同时改 prefill 与 decode 的选块预算，会把 prefill 变成全选、实验直接退化。也不能只强制一个块：16K @50% 的针落在 token 8190..8203，而 `bs=32` 的分块是 8160–8191 / 8192–8223，**针横跨 block 255 与 256**，只留 255 就把 `ramen` 本身丢掉了（这也是钩子做成列表的原因）。

16K @50%，**同一二进制**（`acb1b17f55a05ecc4e86832a3fc6ddb3`）、同参数，唯一变量是钩子：

| 腿 | decode 选块集里的针块 | 选块调用 | 召回 | PPL(Q8) |
|---|---|---|---|---|
| g2 基线（无钩子） | **0/28672** | `selected 19`（探针半 19） | NO | 8.335 |
| g1 强制保留块 255,256 | **28672/28672**（`in_sel=1`，`nsel=34`） | `selected 28672`（`MISSED 0`） | **NO** | 8.019 |
| b2 精确 prefill（对照） | 28672/28672 | `selected 28672`（`MISSED 0`） | **YES** | **3.916** |

⇒ **针块在每一次 head 调用里都被 attend 了（`MISSED 0`），模型仍然答 `The text provided does not contain any information about Zara's favorite food.`** 而 PPL 只从 8.335 掉到 8.019（强制保留两块带来的 softmax 归一化扰动确实有，但量级远不够），换回精确 prefill 才是 3.916。**主因是 prefill 核，不是 decode 选块。**

结论落到三处：

1. §10.3 的默认值判断更硬了：`VLLM_SPARSE_PREFILL` 必须默认关，理由不是「重要性表被污染」这种可修的副作用，而是**稀疏近似注意力本身就损坏了上下文表示**，修选块救不回来。
2. 板端那 4.0× 的 prefill 提速，**只存在于「质量已经崩掉」的那一档**；要提速必须让 `k` 大到近似足够好，而那时 ATTN ∝ k 会涨到与精确档持平（§10.3）。
3. 先前设想的「只让最后一个 mini-batch 走精确核、其余走稀疏，从而拿到干净的重要性表」方案**作废** —— 它修的是表，坏的是 hidden state。

日志里当时没有 `[OPT] force_blk=...` 这一行：`[OPT]` 回显只在 serve 路径打印，`--longctx-*` 路径下钩子会静默生效。已在 `[NDLTRACE]` 块补了回显（诊断开关静默是个真隐患 —— 复跑实验的人不知道带了钩子）。当时可用的在日志证据是 `nsel=34`（= 32 + 2）与 `in_sel=1`。

### 11. 板端 attention 提速试点：QB 分块、线程数、寄存器墙（全部负结果，但边界被钉死）

板端 16K 精确 prefill 的 86.2% 在 ATTN（§10.2），所以这一轮试了三个方向。**三个都没拿到收益，但每个都给出了明确的边界**，并作废了一条我先前提的推理。

**11.1 QB（查询分块因子）A/B —— 收益 −5.9%，且不是位级中立**

`ST_ATTN_QB`（默认 4）参数化，8K @50% 精确 prefill，同轮背靠背（GEMM 作对照，几乎不动 ⇒ 机况一致）：

| 腿 | QB | ATTN | GEMM | PPL(Q8/F32) | greedy |
|---|---|---|---|---|---|
| h1 | 4 | 366.4 s | 120.8 s | 4.118 / 4.143 | `627306ab…` |
| h2 | 8 | **344.9 s（−5.9%）** | 121.4 s（+0.5%） | 4.127 / 4.103 | **分叉** |

- 带宽模型（流量 ∝ 1/QB ⇒ 应 −50%）**被自己的实验否掉**。
- 位级：greedy 前 20 token 相同、之后分叉；PPL 四个数全不同；**prefill 重要性指纹也不同**（`needle_blk` 8.7349 → 8.68818、`max` 1051.46 → 1051.36）⇒ 差异在 prefill 数值，不在 decode 选块。
- 2K 复现（nb=ctx+5=2053，两者尾部 1 vs 5）：PPL 3.720 → 3.770、imp 209.259 → 209.425、greedy 不同。而 `--test-sparse` 的合成用例（nb=8，两者都无尾部）在 QB=4/8 下**逐字节相同** ⇒ 分块本身是路径不变的，差异来自「**哪些 query 落进尾部串行路径**」。**结论：`ST_ATTN_QB` 不是位级中立的 A/B 旋钮**（默认 4 时行为不变，见 11.4）。
- **撤回一条我先前的推理**：拿「16K 流量 15.4 TB ÷ 10 GB/s ≈ 1564 s 与实测 1564.3 s 吻合」当 DRAM 带宽证据是无效的 —— 流量与时间都 ∝ nb²，两者简并；那个 10.5 GB/s 是**用流量除时间反推出来的**，不是独立测量。

**11.2 寄存器墙（直接证据，不再是推测）**

同一 TU 编译两份汇编（`cc -S`，flags 仅差 `-DST_ATTN_QB`），`vllm_attn_batched_worker` 函数体两边同为 726 行：

| 构建 | 栈帧 |
|---|---|
| `ST_ATTN_QB=4` | **无 `sub sp, sp`（0 字节）** |
| `ST_ATTN_QB=8` | **`sub sp, sp, #624`** |

8×float32x4 累加器 = 32 个 NEON 寄存器，必然溢出；内层循环因此多出栈读写，减半的 K/V 流量换不成时间。⇒ **分块类「减少 K/V 重复读」的优化在 NEON 上被寄存器墙封顶，QB≈4 已是上限。**

**11.3 线程数：用户结论被分解验证**

8K、同一二进制、同参数，`VLLM_THREADS=8`（含 4×A55，已核 `Threads: 8`、load 7.23）：

| 腿 | Prefill 总 | GEMM | ATTN | 输出 |
|---|---|---|---|---|
| t4 | 493.6 s | 122.0 s | 365.3 s | — |
| t8 | 494.3 s（+0.2%） | **103.8 s（−15%）** | **382.3 s（+4.4%）** | **与 t4 逐位相同** |

⇒ 板端只有 4 个 A76 大核（cpu4-7）值得用：A55 在 GEMM 上有正贡献、在 **attention 上是负贡献**，总时抵消为零。结论已写入项目记忆。

**11.4 确定性锚点（本轮新增，全部逐位相同）**

| 对照 | 结果 |
|---|---|
| k1（8K, QB=4）vs h1（同二进制同参数复跑） | PPL/imp/greedy 全等（ATTN 366.4 vs 365.3 s，−0.3%） |
| m1（2K, QB=4）vs h0 | 全等 |
| e2 vs g2（16K 稀疏 prefill，历史两腿） | 全等 |
| pristine 二进制的 `--test-sparse` / `--test-l3` vs 补丁二进制 | **逐字节相同** |

⇒ 精确 prefill 路径**不存在竞态或未初始化读**（否则复跑不会一致）；本轮的位级差异都是构建/参数引起的系统性差异。

**11.5 pristine（未打补丁）端到端锚点：做不到，原因是设计性的**

用 `git worktree add --detach HEAD` 重建未打补丁树（**不动工作树**，跑完 `pre.patch`/`post.patch` 两个 md5 相同即证树未被碰）后想跑同配置 8K 对照 —— **不可行**：pristine 里 `g_longctx_quality` 只有赋值、**从未被使用**（Part F 的调用点就是本补丁新增的），未打补丁二进制根本进不了 8K 捞针路径。且 pristine 收到未知参数（`--needle-pos` 是补丁新增的）会**静默退化成自检套件并返回 `rc=0`** —— 那条腿 2 秒「成功」是假成功。

⇒ 「QB=4 ≡ 历史行为」这条**目前只有论证 + 合成用例逐字节相同，没有 8K 端到端实证**，作为已知缺口记录在此。方法论教训：任何 `rc=0` 都要核对耗时与输出内容。

**缺口闭环（2026-09-17）**：pristine 那条路依旧不可达（设计性，见上），因此改用**跨一轮代码的回归锚点**：`vllm_kestrel_qb4`（md5 `4f6e0c0e…`，QB 参数化那一轮构建的二进制）对现役 `vllm_kestrel`（`1070a164…`，其后追加了 int8 快路径；本轮未设 `VLLM_ATTN_I8`，该路径不参与）。两腿同配置 8K 针检索（`--longctx-ctx 8192 --needle-pos 50`），去掉 rss/计时/进度/回显后比规范形：

| 项 | 现役 `1070a164…` | `qb4` `4f6e0c0e…` |
|---|---|---|
| 规范形比对 | **逐字节相同（EQ）** | 同上 |
| `needle_blk` / `rank` | 8.7349 / 6/257 | 8.7349 / 6/257 |
| `ratio_needle/thr` | 2.396 | 2.396 |
| `PPL Q8 / F32` | 4.118 / 4.143 | 4.118 / 4.143 |
| greedy 串 | 逐字相同 | 逐字相同 |
| Prefill 总时 / ATTN | 496.5 s / 367.5 s | 495.2 s / 368.1 s（0.25%，机况噪声） |

⇒ **「QB=4 ≡ 历史行为」现在有 8K 端到端实证**。口径边界照实写：这是**跨一轮代码**的锚点，不是跨 pristine 的锚点；pristine 不可用的原因是上面那条（`g_longctx_quality` 从未被使用 + 未知参数静默降级并返回 `rc=0`）。

> harness 缺陷留证：该腿第一次运行因驱动脚本 `run()` 参数错位（`tag` 与二进制名混用）导致第一腿执行 `./k22_cur`（不存在）而以 `rc=127` 秒退，`seq22.txt` 里因此先出现了一条**假阴性**的 `DIFF … NOT closed`。缺腿用 `ndl_board22b.sh` 补跑后得到上面的 EQ。任何「秒退 + 空输出」都要先怀疑腿没跑起来，而不是先怀疑结论。

### 12. 板端 16K prefill attention 提速：设计依据（载荷 int8 化已实现，见 §13；layout 重写未立项）

11.2 与 11.3 指向同一件事 —— 这个内核同时贴着**核内载入槽**与**共享内存侧（DRAM/互连）**两堵墙，t4 时两者都接近饱和，所以**只动其中一个都拿不到收益**：

> 注意：这两堵墙里，「核内载入槽」（寄存器墙）有汇编实证（11.2），「共享内存侧」这一侧的依据与下表第 3 行**同源**（即被 §11.1 撤回的反推带宽），见下方更正。

| 只动一处 | 结果 |
|---|---|
| 只减字节（QB 分块） | 撞寄存器墙（11.2） |
| 只加核（t8，含 A55） | 共享墙不动，ATTN 反而 +4.4%（11.3） |
| 只降指令（K 转置成 dim-major，lane=位置） | 原判「转置后 DRAM 成新墙：3.85 TB ÷ 当前有效带宽 ≈ 366 s」—— **该依据已失效，见下方更正** |

> **更正（本轮）**：上表第 3 行的定量依据**已失效**，留着不撤就等于保留一条建立在撤回结论上的判据。它用的「当前有效带宽」正是 §11.1 判定为**简并**的那个反推值（16K 流量与时间都 ∝ nb²，10.5 GB/s 是流量 ÷ 时间倒算出来的，不是独立测量）。⇒ 「转置后 DRAM 成新墙」目前是**未验证的推测**，转置后的流量敏感性**从未测过**。

⇒ 要提速必须**同时降字节与降指令**：`k_pack`/`v_pack` 改 **int8 载荷 + dim-major 转置打包**，QK 段用 `vdotq_s32`（一次 128-bit 载入出 16 个 int8 ⇒ 16 MAC/载入；现结构是 4 MAC/向量载入、且向量载入次数 ≈ 1.25×FMA 次数）。量化前的 Hadamard 旋转（压 outlier）与逐组 scale 是配套项。这条路线与公理层 `blas_precision_efficiency_tradeoff`（INT8 2–4×）、`npu_weight_prepacking_001`（一次性重排摊到多消费者）、以及 `资料/ninfer-master` 的 `int8_g64_codec.cuh` + `hadamard_d256.cuh` 是同一形态。

> **同时更正这条推论本身**：上面的「必须同时降字节与降指令」**没有被验证过**（左腿的依据已失效，右腿未做）。而且 layout 重写只作用于 QK 段，按 §13.2 的指令拆分，QK 只剩 34/131 —— **即使把 QK 段优化到 0，上限也只有 ~1.35×，不是 2–3×**。要 2–3× 必须连 VKQ 段一起重写（VKQ 每个 `s` 对 `o` 做读—改—写，是更贵的结构，且它有**位级不变**的改法：`o` 常驻寄存器 + `hd` 分趟，因为 VKQ 无跨 lane 归约）。⇒ 该路线**仍未立项**，需先用最小切片分离出真实的受限类型。

数值上必然不等价（量化 + 换求和树）⇒ 按既定规矩做**默认关闭的 fast path**，验收面必须是**召回**，不能只看 PPL。**状态：第一步（载荷 int8 化，不动 layout）已实现并实测，见 §13；layout 重写仍未立项。**

### 13. int8 KV 载荷快路：三条路径的实测判决与 16K 目标档 −25.7%（已实现，默认关闭）

§12 那条路线的**第一步**（先只换载荷与点积指令，**不动 K 的 layout**）本轮实现了三个版本并全部拿到板端实测。结论：**v1/v2 被否，v3 是长度相关的两边下注，最终采纳的形态是「K int8 必需 + V int8 可选开关」**，在 16K 目标档拿到 ATTN **−25.7%**、prefill 总时 **−21.5%**，且**召回保持 YES**。

**13.1 三条路径**

| 版本 | 形态 | 2K | 8K | 判决 |
|---|---|---|---|---|
| v1 | QK 段直读 `kv_cache_q8`（不拷包） | ATTN **+67%（慢）** | — | **否** |
| v2 | K、V 都拷 int8 连续包，QK 用 `vdotq_s32` | +17~18%（慢） | ATTN **−19.4%** | **否**（长度反转） |
| v3 | K int8 连续包 + **V 保持 float** | +2.1%（中性） | ATTN −6.9% | 采纳其一 |
| **v4** | K int8 必需 + **V int8 做成运行时开关** | — | — | **定稿** |

- **v1 为什么慢**：`kv_cache_q8` 按 head 分块存放，行距 1024 B，而单行只需要 128 B ⇒ 每取 128 B 跨一次缓存行，**载入条数比 float 路径还多**。教训：int8 的收益前提是**连续打包**，不是「数据类型更小」。
- **v2 为什么在 2K 反而慢**：V 也 int8 后 VKQ 段需要把 int8 V 逐元素转回 float 并加权归约，指令数**净涨**（见 13.3 指令预算表）⇒ 短序列下 int8 占不到便宜。

**13.2 长度反转与 16K 目标档（决定性正收益）**

8K（`--sparse-attn` 快路 A/B，同轮背靠背）：

| 腿 | 形态 | ATTN | 相对 | 针块 rank | 召回 |
|---|---|---|---|---|---|
| r3 | float（基线） | 366.7 s | — | 6/257 | YES |
| r4 | v2（K+V int8） | **295.8 s** | **−19.4%** | 6/257 | YES |
| s4 | v3（K int8, V float） | 341.1 s | −6.9% | 6/257 | YES |

16K 目标档三腿（精确 prefill，`--longctx-ctx 16384`）：

| 腿 | 形态 | Prefill 总 | GEMM | ATTN | 针块 rank | PPL(Q8/F32) | 召回 |
|---|---|---|---|---|---|---|---|
| f1 | float（基线） | 1822.7 s | 241.2 s | 1569.1 s（86.1%） | 6/513 | 4.110 / 4.077 | YES |
| f2 | V=float | 1715.5 s（−5.9%） | 244.8 s | 1458.3 s（−7.1%） | 7/513 | 4.036 / 4.032 | YES |
| **g1** | **V=int8** | **1430.5 s（−21.5%）** | 251.3 s（+4.2%） | **1166.4 s（−25.7%）** | **7/513** | **4.095 / 4.082**（ratio 1.0031） | **YES** |

g1 的 greedy 输出 `Based on the information provided, Zara's favorite food is ramen.` —— **答对**。decode 侧 ATTN 仍占 88.5%（`n=64 total=39169.0ms (612.02 ms/tok) | GEMM=4485.0ms(11.5%) ATTN=34666.2ms(88.5%)`）。

⇒ 16K 端到端：**ATTN 26.2 min → 19.4 min，prefill 总时 30.4 min → 23.8 min**；代价是 GEMM +4.2%（int8 包的额外拷贝），净赚。

**长度越长 V 走 int8 越划算** —— 8K 时 v2 相对 v3 差 13%，16K 时差 **20%**。这条与 2K 的结论相反，印证「短序列指令受限、长序列流量受限」：

| 每（query, s）指令数 | float 全档 | int8 全档（v2） | v3（K int8 + V float） |
|---|---|---|---|
| QK | 75 | **34** | **34** |
| VKQ | 97 | **145** | 97 |
| 合计 | 172 | **179** | **131** |

**13.3 载入条数模型（取代之前被否掉的 DRAM 带宽模型）**

每条 FMA 实际要付 `(4 K 载入 + 4·QB Q 载入) / (4·QB FMA)` = `1/QB + 1` 次向量载入 ⇒ QB=4 为 1.25、QB=8 为 1.125（预测 −10%，实测 −5.9%，见 §11.1）。**这个模型解释了 v2 的 2K 反例**：模型只数「条数」，而 int8 每条指令的真实代价高于 float（SDOT 吞吐、int8 载入、归约链依赖），所以**条数不是唯一变量**。

**13.4 跨平台构建损坏（本轮修，教训在此）**

`ST_ATTN_QB` 与新增的 `ST_ATTN_Q8_*` 常量**原先定义在 `#if ST_HAVE_NEON` 区内**，而它们有三个使用点在区外（`scores_buf` 分配、金丝雀检查、`sc_buf` 分配）⇒ **自上一轮 QB 参数化起，x86 构建就已经是坏的**。板端 ARM 编译看不见这个错，因为那个条件区照常展开。已把定义上移到文件顶部**无 arch 条件**区并加注释说明为何不能放回。

教训：**参数化一个原本只活在 arch 条件区内的宏，必须连同它的全部使用点一起看**；只在目标板验证会长期掩盖另一平台的构建失败。

**13.5 与 `--prefill-batch` 的关系（正交，天然叠加，但本身不贡献速度）**

代码耦合已定位在两处：`VLLM_IMP_LAST_MB`（重要性表只由**最后一个 mini-batch 窗口**的 query 决定 ⇒ nb 越大窗口越粗，极限 `nb >= n_tokens` 退化成旧的「整段求和」口径 = 病因 0 复发）与 `nb % ST_ATTN_QB`（≠0 时 batch 末尾 token 走串行尾路径、数值不同）。

性能上：16K 的 GEMM 已经跑到 int8 点积峰值的 ~95%（241 s 对应 15.2 MAC/核周期），抬 nb 最多省 ~6 s（**0.3%**）；**ATTN 与 nb 完全无关**。⇒ 与本轮 int8 快路**正交、可叠加**，但它自己**不带来速度**，不应作为提速项列入。

> **本段是推算，无实测背书。** 板端三腿 A/B（8K，pb=256 / pb=1024 / pb=1024+I8）跑到第一腿 `[PREFILL] 5376/8197` 时**被人工中止**，只留下了代码耦合的两处定位（可静态核对），性能数字取自 GEMM 峰值的算术推算。

**13.6 开关与边界**

- `VLLM_ATTN_I8=1`：K 的 int8 连续包（必需项），启用 int8-K 内核；生效回显只打一次（`g_attn_i8_logged`）。
- `VLLM_ATTN_V_I8=1`：V 也走 int8 连续包（**默认 0**，即 V 保持 float）；仅在 `VLLM_ATTN_I8=1` 时才有意义。做成开关是因为它是**长度相关的两边下注**（13.2 表）。
- 两者都**默认关闭**，是 fast path；数值必然不等价（量化 + 换求和树）⇒ 验收面是**召回**，不能只看 PPL。
- 回归闸门全部通过（`I8=off` 逐位相同）：s1(2K,off)=h0/m1、s3(8K,off)=k1/h1、r3=k1/h1、k1(8K/QB4)=h1、m1(2K)=h0、e2=g2(16K)。
- 一次重复 `free`（heap 双释放风险）已修，两个释放点各 2 处已复核。
- **剩余缺口**：−26% 只吃掉 §12 那堵墙的 **1/4**。要到 2–3× 仍需 §12 的**布局重写**（K 转置成 dim-major + 位置分块，把 Q 变成寄存器内标量广播）—— 较大立项，**未立项、等裁定**。

### 14. VKQ 段「o 常驻寄存器」改造：位级等价已证（含寄存器分配的三次实测教训）

**14.1 为什么这条改造位级中立 —— 结构性论证，不是经验判断**

VKQ 沿 `i` 方向**没有跨 lane 归约**：每个输出元素 `i` 只对 `s` 独立累加（与 QK 段的 `hsum_neon4` 正相反）。所以维度**分块**（各块维度互不相交）以及块内的遍历顺序，都不改变任一元素的数值；唯一要守住的是「对固定 `i`，`s` 从 0 升序、走同一条 `vfmaq_f32`」。⇒ 这是**逐位相同**，不是近似等价。顺带：编译侧也不会偷偷重排 —— 板端 Release 是 `-O2 -ffast-math -fno-unsafe-math-optimizations -ffp-contract=off`，且 fma 用显式 intrinsic。

**14.2 验证方法（板端八腿矩阵 + 控制构建）**

| 腿 | 内容 |
|---|---|
| 控制构建 | 用**未改动**的板端树重建 → `build-base/vllm_kestrel` md5 = **`1070a164…`，与线上二进制逐字节相同** ⇒ 任何差异只能归因于补丁 |
| a1 / a2 / a3 | 2K、V=int8：旧二进制 / 新二进制 OREG=0 / OREG=1 |
| a4 / a5 | 2K、V=float：OREG=0 / OREG=1 |
| b1 / b2 / b3 | 8K、V=int8：旧 / OREG=0 / OREG=1 |

判据：剔除 rss、计时、进度行与开关回显后取**规范形**（95 行，含 `needle_blk/rank/ratio`、PPL、greedy 串），再比 md5。

| 比对 | 结果 | 规范形 md5 |
|---|---|---|
| 旧 ↔ 新(OREG=0) @2K | **EQ** | `2E18C143…` |
| 新(OREG=0) ↔ 新(OREG=1) @2K | **EQ** | 同上 |
| V=float：OREG=0 ↔ OREG=1 @2K | **EQ** | `243FF551…` |
| 旧 ↔ 新(OREG=0) @8K | **EQ** | `FC232459…` |
| 新(OREG=0) ↔ 新(OREG=1) @8K | **EQ** | 同上 |

等效到具体数值（8K）：`needle_blk=8.42408`、`rank=6/257`、`ratio_needle/thr=2.45`、`PPL Q8=4.250 / F32=4.243`、greedy 串逐字相同。开关回显只出现在 OREG=1 的腿，且日志显示 `hd=128` ⇒ 新路径确实被执行（两趟 64 维分块），不是「开关没生效还返回 rc=0」的假成功。

**跨批交叉核对（本轮闭合）**：把两批（v1 版与 v3 版）的 `.out` 用**同一个过滤器 + 归一化换行**重算哈希 —— 2K V=int8 的 5 个文件（线上旧二进制、v1-off/on、v3-off/on）、2K V=float 的 4 个文件（v1-off/on、v3-off/on）、8K V=int8 的 7 个文件（旧二进制、v1-off/on、v3-off×2/on×2）**各自只得到 1 个哈希**（均 95 行）⇒ 线上二进制、v1、v3 在同配置下输出**逐位相同**。先前拿板端 `grep` 产出的 md5 去比本地 PowerShell 产出的 md5 是**错的**（前者 LF、后者 CRLF），与过滤器无关 —— 这是个容易误判成「结果不一致」的坑。

**14.3 但收益没兑现：三个实现的寄存器/栈帧实测（同 TU 编汇编对比）**

| 版本 | 写法 | 栈帧 | `[sp,` / `[x29,` 引用 |
|---|---|---|---|
| before（线上） | 逐 s 读—改—写 `o` | `sub sp, sp, #1456` | 46 |
| v1 | 累加器数组 + 内层 `v8_on` 分支 | `sub sp, sp, #1712`（**+256 B**） | 45 |
| v2 | 数组 + `_Pragma("GCC unroll N")` | `#1520` | **82** |
| v3 | **具名标量累加器**（宏展开，无下标） | `#1504` | 82 |

- v1 的 **+256 B 恰好 = 16 × 16 B**，即 16 个 `float32x4` 累加器**整块落栈** ⇒ 「省掉的 `o` 读—改—写」被原样换成「栈的读—写」，这正是它没提速的原因。
- v2 想让 `acc[j/4]` 的下标变常量索引而寄存器化（本文件 L896 的 q8g 内核正是用同一手法成功的），**结果反而更差**：函数从 730 行膨胀到 995 行、栈引用涨到 82。
- v3 改成 16 个具名标量（`oa0..oa15`，宏展开）后栈帧只 +48 B ⇒ **不再是整块溢出**，但栈引用仍是 82，说明仍有部分值走栈。
- 结论（实测、非推测）：**A76 上「累加器数组 → 寄存器」不可依赖**，而且同一种手法在不同体量的内核里结论相反（L896 成立、这里不成立）⇒ 这类改造必须看汇编，不能看源码意图。

**14.4 性能判决：位级等价成立，但收益为零到微负 ⇒ 该方向判否**

8K **同轮交错 A/B**（同一二进制、单变量 env，腿序 `off/on/off/on`；`--longctx-ctx 8192`、V=int8）：

| 腿 | Prefill | 相对 off 均值 |
|---|---|---|
| s1 off | 428 548.7 ms | — |
| s2 **on** | 431 040.3 ms | +0.58% |
| s3 off | 429 359.9 ms | — |
| s4 **on** | 433 027.1 ms | +0.85% |

off 均值 428 954 ms、on 均值 432 034 ms ⇒ **+0.72%**（off 腿自身离散 0.19%、on 腿 0.46%），且**两条 on 腿都高于两条 off 腿** ⇒ 方向一致，不是噪声内的中性，是**小幅负收益**。对照：v1 的 8K 单腿是 **+8.7%** —— 具名累加器把 v1 的严重退化压回到「−0.7%」，但**没有换来任何收益**。

原因与 §14.3 的汇编证据一致：`o` 的读—改—写本来就是 **L1 常驻**的（每个 query head 的 `o` 仅 `hd×4 B = 512 B`），**不是 DRAM 流量**；把累加器搬进寄存器只省下 LSU 发射槽，代价却是把 16 个值长期占住寄存器文件，一部分只好换到栈上（栈引用 46 → 82）。⇒ 在这堵墙上「省掉 o 的 RMW」不是可交换的收益。

**判决：本方向判否**（与 §11.2 的 QB=8 同类处置 —— 负结果 + 边界被钉死）。`VLLM_ATTN_OREG` 保留为默认关闭的开关作为记录，不再继续调参；要动 16K 的 ATTN，仍需 §12 那条「同时降字节与降指令」的路径，而它到目前为止**没有可信的定量依据**（§12 更正）。

**14.5 与 §11.2（QB=8）的对照**

两者撞的是同一堵 NEON 寄存器墙（QB=8 时 `vllm_attn_batched_worker` 出现 `sub sp, sp, #624`）。区别是：QB=8 拿寄存器换**流量**，本改造拿寄存器换掉 `o` 的**读—改—写**；两次都说明「寄存器够不够」只能由汇编回答。

**14.6 开关**

`VLLM_ATTN_OREG=1` 打开（**默认关**）。它与其他快路开关的正交性：只改 VKQ 内部循环结构，不改量化、不改 QK、不改 `imp_head` 的加序（重要性表改为独立一趟扫，`k` 升序、`s` 升序与原来一致，且只加一遍）。

### 15. 长上下文（大海捞针）收尾：默认档与开关台账

本节把长上下文相关开关一次性定稿。判据是**用户可见行为**（默认档能不能过针检索），不是单项微优化的账面收益：

| 开关 | 默认 | 作用 | 为什么是这个默认 |
|---|---|---|---|
| `VLLM_SPARSE_PREFILL` | **关** | 让 prefill 也走稀疏核（旧行为） | 稀疏 prefill 只对**被选中的块**累加 `imp_head` ⇒ 重要性表变成 probe 的自证闭环，且**损坏 prefill 的 hidden state**（§1、§10.3）。开它就丢针 |
| `VLLM_IMP_LAST_MB` | **开** | 重要性表只统计**最后一个 mini-batch** 的 query | 旧口径「整段全部 query 求和」带强位置偏置：16K@50% 的针块名次 6/513 → 148/513、decode 全丢（§3） |
| `VLLM_ATTN_I8` | **关** | 精确 prefill 改走 int8-K 载荷内核 | 属 fast path（换量化 + 换点积指令），验收面是**召回**；默认装在 float 精确档上（§13） |
| `VLLM_ATTN_V_I8` | **关** | V 也走 int8 连续包 | **长度相关的两边下注**：2K 指令受限用 float 更省、8K/16K 流量受限用 int8 更省（§13.2） |
| `VLLM_ATTN_OREG` | **关** | VKQ 走累加器常驻 + 维度分块 | 位级不变，但实测 **+0.72%（更慢）** ⇒ 方向判否，仅留作记录（§14.4） |
| `ST_ATTN_QB` | **4**（可用 `-D` 覆盖） | 查询分块因子 | 4 与历史行为一致；**>4 不位级中立**（改变「哪些 query 落进尾部串行路径」）且撞寄存器墙（§11.1/§11.2）⇒ 不是可以随便调的旋钮 |
| `VLLM_SPARSE_RATIO` | **0**（关） | 稀疏保留率改为按比例 | 默认不改变 `--sparse-k` 的固定块数语义（§2） |
| `--sparse-pf-group G` | **0**（关） | 稀疏 **prefill** 核的选块粒度：每 G 个 token 一组独立选块 | 0 = 历史行为（整批一个代表 query）。G=8 在 8K 内置捞针上把召回从 NO 翻回 YES、代价 +13.8% ⇒ 修的是**写入侧**污染（见 2026-09-24 节）。**但多轮稀有码检索 / 16K / 8B 尚未复验，故仍默认关** |

一句话口径：**默认档 = 稀疏 decode + 精确 prefill**。跑长上下文事实检索请用默认档。
2026-09-24 的修正：把 prefill 交给稀疏核**默认仍是质量倒退**，但它的根因已定位为
「整批共用一个代表 query 选块」这一**具体口径缺陷**，用 `--sparse-pf-group 8` 可在
8K 内置任务上把质量救回（代价 +13.8%）；该档在通过更严的验收前**不默认化**。

### 16. 大海捞针对标 llama.cpp（默认档）：同文本、同 token 数、同核数

把 §15 定稿的默认档拿去和 llama.cpp 在同一块 RK3588 上对打，两轮（`r1`/`r2`），每轮深度 25/50/75%，每轮冷启。

**16.1 口径与公平性（先说清不可比的地方）**

| 项 | 我方 | llama.cpp |
|---|---|---|
| 二进制 | `vllm_kestrel` `1070a164…` | `llama-server`（`/root/llama_build_rk3588`） |
| 权重 | `/mnt/VQF/8b/serve`（`model.vqf` `b6d8d1d7…`，6.6 GB，group-16 Q4 ≈ **4.25 bpw**） | `qwen3vl8b-q4_0.gguf`（`96fb8a5e…`，4.77 GB，Q4_0 **4.50 bpw**） |
| 线程 | `--threads 4` / `VLLM_THREADS=4` / `OMP_NUM_THREADS=4` | `-t 4` |
| 上下文上限 | 模型自带 `max_seq`（本次提示词 7249 token 全在窗内） | `-c 9848` |
| governor | `performance`（11:47:03 由 `ondemand` 切；两侧被计时的请求都在切换之后） | 同 |
| 提示词 | 600 段 `边缘计算与云计算的核心区别在于数据处理发生的位置。` + 一句针 + 提问 | **逐字相同的同一段文本** |
| token 数 | 7249（两侧 tokenizer 一致，实测两侧都是 7249 / 7271） | 7249 |
| 判据 | 答案数字里出现 `739152`（问题里从未出现该串） | 同 |

三处**不可比**，必须一起记住：

1. **权重不同**：4.25 bpw vs 4.50 bpw，我方每条权重少约 6% 字节。
2. **前缀缓存语义不同**：llama 复用它上下文里的**原始 f32 KV**（无损）；我方第二、三问走 `VLLM_L3_PREFIX_REUSE=1`，前缀 KV 是**从 Q4 载荷解压回来的**（有损，纯解压不重算 —— 见 [vllm_server.c](file:///d:/项目/New_vLLM/GitHupSRC/src/serve/vllm_server.c#L464-L469) 的语义说明）。⇒ **d50/d75 的时间不能与 llama 做等价比较**。
   但**复用的“量”是同一个数**：两侧在 d25→d50 都复用 **1803** token（我方日志 `reuse 1803-token KV prefix`；llama `cache_n=1803` / `prompt_n=5446`）。这个数由提示词的构造方式决定、与引擎无关，所以「要重算多少」这一侧是对齐的，差别只在恢复出来的 KV 是否无损。
3. **decode 两侧不是同一个算法**：我方默认档 decode 走 `--sparse-attn --sparse-k 32` 稀疏核，llama 是稠密 attention。

**16.2 r1 分相（7249 token 冷态，两侧同一请求）**

| 阶段 | 我方（默认档） | llama.cpp Q4_0 | 比值 |
|---|---|---|---|
| prompt 处理 | **1148.03 s**（ttft；`[PREFILL-TIMING]` 1141.56 s） | **638.33 s** | 我方慢 **1.80×** |
| ├ GEMM | 368.99 s（50.9 ms/tok） | 未拆 | — |
| ├ ATTN | 760.92 s（**105.0 ms/tok**） | 未拆 | — |
| └ OTHER | 11.65 s | — | — |
| decode 22 token | 9.95 s（**473.7 ms/tok**） | 22.71 s（1081.3 ms/tok） | 我方快 **2.28×** |
| 端到端 | 1158.04 s | 661.04 s | 我方慢 1.75× |
| 峰值内存 VmHWM | 7.10 GB | 10.10 GB | — |

llama 侧的数字取自它自己的 `slot print_timing`（`prompt eval time = 638334.65 ms / 7249 tokens`、`eval time = 22706.90 ms / 22 tokens`），不是我方从总时里减出来的。

**另有**一个用户可见的差别，一并记下：**每会话第一问**（3 段、48 token、`max_tokens=1`）我方 **72.45 s（r1）/ 72.73 s（r2）**，llama **2.38 s（两轮相同）**。我方那 72 s 里有 66.29 s 记在 `GEMM`（`[PREFILL-TIMING] n=48 total=66285.4ms | GEMM=65915.0ms(99.4%)`）—— 48 个 token 花 66 s，这不是算力，是**首问的一次性成本**（惰性权重/页落地），两轮近乎相同（72.45 / 72.73）说明它可复现。这条不影响后面的 prompt 处理与 decode 数字（那些是独立请求），但**它确实是用户会先撞到的**，故不做粉饰。

**一处值得记住的对比**：我方**单 ATTN** 的每 token 成本（105.0 ms）已经超过 llama **整个 prompt 处理**的每 token 成本（88.06 ms）。⇒ prefill 的差距不在 GEMM（我方 int8 点积已贴近峰值，§13.5），**全在 attention**。

**顺手排除一种受限类型（用同批已有数据，无需新实验）**：把 KV 载荷换成 int8（K 与 V 的字节都掉到 1/4）在 8K 只拿到 ATTN **−19.4%**（§13.2）。若 prefill attention 是 **DRAM 带宽受限**，4× 减字节应换来接近 4× 的提速 ⇒ **它不是带宽受限**。这条与 §11.1 撤回的那条反推带宽结论方向一致（那里同样不该用「流量 ÷ 时间」倒算带宽）。本轮 7.2K 也能复算出那个反推值：K+V 每 token ≈ `2 × 3625(因果均) × 1024(kv_dim) × 4 B × 36 层` ≈ **1.07 GB**，105 ms ⇒ ~10.2 GB/s —— **这个数同样不能当独立测量**，列在这里只说明它自我一致，不构成依据。

那受限的是什么？同一轮里我方 **GEMM** 跑到 int8 点积峰值附近（§13.5），而 **ATTN** 折合 `2(QK+PV) × 2(MAC) × 3625 × 32(head) × 128(hd) × 36 层 = 2.14 GFLOP/token ÷ 105 ms ≈ **20.4 GFLOP/s**`。同一批核、同一轮，GEMM 270 GFLOP/s vs ATTN 20.4 GFLOP/s，**差 ~13×** ⇒ 瓶颈在**指令/依赖**，既不是算力也不是带宽。这与上一轮测到的「每 (query,s) 对 175 cycles ÷ 179 条指令 ≈ IPC 1.0」是同一件事。

> **插值推算，非本轮实测**：§13.2 的 int8 载荷快路在 8K 是 ATTN −19.4%、16K −25.7%。若 7.2K 也能拿到约 −20%，则 ATTN 105.0 → ~84 ms/tok、prompt 处理 158.4 → ~137 ms/tok，差距从 1.80× 收窄到 ~1.56×。这条只是把已实测的两个点做插值，**本轮没有在 7.2K 上开 `VLLM_ATTN_I8` 实测**。

`r2` 与命中矩阵见 §16.5；`r1` 里我方 d50/d75 不是结果而是引擎退出，见 §16.3。

**16.3 r1 里我方 d50/d75 不是结果，是引擎退出（L3 部分前缀复用写 NULL 块）**

症状（`vllm_r1.log` 证据）：d25 跑完后，d50 的日志**停在**

```
[L3] evicted 6156 blocks -> /mnt/emmc/niah_l3_niah8b_u65bbfe41 (cursor=240.47 MB, seq=7249, keep=39, ratio=0.75, packed_now=6156, total_packed=6156), freed 1923.8 MB from RAM
[L3] restored 1692 prefix blocks from Q4 payload (prefix=1803)
[KV-PREFIX] L3 restored 1692 block(s) for 1803-token prefix
[KV-PREFIX] reuse 1803-token KV prefix, prefill rest
```

**之后一行 `[PREFILL]` 进度都没有**，HTTP 侧 0.86 s 就返回了（空 body）；d75 更短（0.0005 s，连接直接失败）⇒ 进程在 d50 那一刻就已经没了，d75 打的是一具尸体。所以这两行 `total_s=0.86 / 0.0005` 是**坏数据，不是结果**，不能进命中矩阵的分母。

根因（代码证据，与上面的日志逐句对得上）：

1. [vllm_server.c](file:///d:/项目/New_vLLM/GitHupSRC/src/serve/vllm_server.c#L476-L487) 的 `l3_restore_prefix(st, keep_lcp)` 只重建**前缀范围内**的块（[vllm_safetensors.c](file:///d:/项目/New_vLLM/GitHupSRC/src/model/vllm_safetensors.c#L16785-L16787)：`n_blocks = ceil(prefix_len / bs)`）。
2. 但本请求**余量**（`keep_lcp..n`）要写的块同样被 Phase-2 驱逐置成了 `NULL`（同一条日志：`evicted 6156 blocks ... freed 1923.8 MB`）。
3. reuse 分支原先直接 `return`，**没有**调 `st_qwen_kv_rebuild_freed()`；而全量 prefill 分支调了（[vllm_server.c](file:///d:/项目/New_vLLM/GitHupSRC/src/serve/vllm_server.c#L512-L520)）⇒ 走复用这条路时，prefill 直接往 NULL 块上写。

这不是新事故：[vllm_safetensors.c](file:///d:/项目/New_vLLM/GitHupSRC/src/model/vllm_safetensors.c#L16703-L16707) 里那个函数的注释早就记着同一类崩溃 —— *"the previous version crashed with a NULL deref on the second prefill after an eviction"* —— **上一轮只修了全量路径**。同一处 [vllm_server.c](file:///d:/项目/New_vLLM/GitHupSRC/src/serve/vllm_server.c#L413-L418) 的注释也写着「**危险死区**：步骤 3 接线完成前，门开会让 prefill 读 NULL 块 → 崩溃」。

修复（本轮）：reuse 分支补调 `st_qwen_kv_rebuild_freed(st)` 并回显重建字节数。该函数遍历**全部** `b < st->kv_n_blocks`、只看指针是否 NULL（[vllm_safetensors.c](file:///d:/项目/New_vLLM/GitHupSRC/src/model/vllm_safetensors.c#L16716-L16763)），与 `cache_len` 无关 ⇒ 恰好覆盖余量块，语义上是完备的，不是兜底。

安全性复核（改之前先做的）：

- **不会把 SIGSEGV 换成断言失败**：lazy 档下 KV 块走 `mmap` 且**不预写 0xA5 守卫** —— [vllm_safetensors.c](file:///d:/项目/New_vLLM/GitHupSRC/src/model/vllm_safetensors.c#L8620-L8628) 原文：*"不再预写 0xA5 守卫 canary（本引擎 KV 路径不读守卫，CK_TAIL/CN_TAIL 仅在 debug env 下启用）；未写 block 读到 0，与旧 calloc 语义逐字节等价"*。
- **无脏读**：余量 prefill 写第 `t` 个 token 前只读 `[0, t)`；其中 `< keep_lcp` 的行来自已恢复的前缀，`>= keep_lcp` 的行由本次 prefill 顺序写入。
- **代价**：它会把被驱逐的块**整批** re-back（本次 1923.8 MB），RAM 占用回到驱逐前水平（8B 上 ~7.1 GB HWM，紧但可承受）。「只补余量区间」是后续可做的优化，与本处正确性无关。

复现与修复验证见 §16.5。

**16.4 与公布口径的差：本轮默认档（精确 prefill）vs README §7（2026-09-16）的稀疏档**

结论一句话：**只差 prefill，差约 2×；decode 基本没动。** 这与 §16.2 的「对 llama 落后 1.80×」是同一件事的两面。

两轮同口径实测（取两侧各自 `metrics` 的**原始字段**，不是从总时里减出来的）：

| 指标（8B / 7249 token / 冷启 / 4 线程 / 全层） | r1 | r2 | 均值 | llama.cpp r1 / r2 |
|---|---|---|---|---|
| 我方 prompt（`ttft_ms`） | 1148.03 s | 1152.59 s | **6.30 t/s** | 638.34 / 641.10 s（11.36 / 11.31 t/s） |
| 我方 decode（`tpot_ms`） | 473.67 ms | 454.22 ms | **2.16 t/s** | 1081.28 / 1079.59 ms（0.926 t/s） |
| 领先关系 | prefill **落后 1.798×** | prefill **落后 1.798×** | — | decode **2.28× / 2.38×** |
| 峰值 VmHWM | 7.10 GB | 7.10 GB | — | 10.10 GB |

与 **README 正文 §7**（2026-09-16；同权重 8B、同 4 线程、7.2K 档）逐项对齐：

| 项 | 公布 §7（全层） | 本轮默认档 | 差 |
|---|---|---|---|
| **我方 prefill** | **12.31 ~ 12.38 t/s** | **6.30 t/s** | **慢 1.95 ~ 1.97×** |
| 我方 decode | 2.27 ~ 2.28 t/s | 2.16 t/s | −3.5 ~ −5% |
| llama prefill | 11.02 ~ 11.18 t/s | 11.31 ~ 11.36 t/s | +1.2 ~ +3.1%（复现） |
| llama decode | 0.94 t/s | 0.926 t/s | −1.6%（复现） |
| 领先关系 | prefill **领先 1.11~1.12×**；decode 2.41~2.43× | prefill **落后 1.80×**；decode 领先 2.33× | **prefill 翻转** |

同档性的两条旁证：公布 §7 全层 7.2K 的 VmHWM 是 6.91 GB，本轮 7.10 GB（+2.7%）；**llama 侧四项数字都在 ±3% 内复现** ⇒ 差异不是平台或口径漂移，**只在我方 prefill**。

归因（算术自洽 + 代码证据，不是猜测）：

1. 本轮 158.7 ms/tok（`ttft_ms` 口径）= GEMM 51.2 + **ATTN 105.5~106.2** + OTHER 1.6。
2. 公布档 81.0 ms/tok ⇒ 若 GEMM 相同（同一内核、同一权重），其 ATTN+other ≈ **30 ms/tok，仅为本轮的 28%**。
3. 稀疏档的覆盖比例恰是 `k=32 块 × 32 token/块 = 1024 / 7210 = 14.2%` ⇒ **量级相符**（差额还含 probe 与选块开销）。
4. 代码侧把这条钉死：[vllm_safetensors.c](file:///d:/项目/New_vLLM/GitHupSRC/src/model/vllm_safetensors.c#L17593-L17609) 的 prefill 内核选择链是 `VLLM_ATTN_I8 →（VLLM_SPARSE_PREFILL=1 才走稀疏核）→ 精确`，而 `VLLM_SPARSE_PREFILL` **默认关**；紧邻注释就是病因 1 的原始对照（x86 / 16K）：**稀疏 prefill 时针块名次 188/513 → 召回 NO；精确 prefill 12/513 → 召回 YES**。

⇒ **README §7 的 prefill 列，是在「长上下文检索会失败」的那个档位下测出来的**（该档 prefill 走稀疏核）。大海捞针要求 prefill 走精确核，两者在默认档下不能兼得。decode 的稀疏核本轮未改动，所以只差个位数百分比。

受影响的公布面：§7 的 prefill 表（4 行）与结论 2（「交叉点在 4K~8K，16K 领先 1.60×」）在默认档下**已不成立**；§7 的 multi-turn 复用轮时延表同样走 prefill 核，需重标。首页摘要里「追问轮 prefill −80% ~ −96%」是**比值**、分子分母同核，**大体仍成立**。decode 表与结论 1 / 4 / 5 不受影响。

---

## 2026-09-16（可移植性）— 去掉线程池「最多 4 核」硬编码，改为按内核 cpu_capacity 推导

### 0. 问题：两处 RK3588 专属假设被写进了引擎

`vllm_tp_init()` 的 ARM 分支把默认线程数上限写死：

```c
nthreads = (nc > 4) ? 4 : (int)nc;   /* RK3588 A76-only default */
```

这个 `4` 是 **RK3588 的 A76 集群规模**，不是通用事实 —— 换一块 big.LITTLE 板（大核数量、编号都可能不同）就错。同一类假设还有一处：worker 亲和里 `core = 4 + (slot % 4)`，即硬编码"大核是 cpu4-7"。为了后续适配其他 ARM 设备，两处一并去掉。

### 1. 改法：从内核自己的容量数推导「性能集群」

新增 `st_perf_cpus()`（`include/common/vllm_platform.h`）：读 `/sys/devices/system/cpu/cpuN/cpu_capacity`（EAS 用的同一组数），取最大值 `maxc`，凡 `capacity >= maxc/2` 的 CPU 计入性能集群。**默认线程数与 worker 亲和都取自这个集合**；推导不可用时退回"全部逻辑核轮转"，而不是猜一个固定簇。

**50% 这个阈值是实测逼出来的，不是拍的**：

| RK3588 CPU | `cpufreq/cpuinfo_max_freq` | `cpu_capacity` |
|---|---|---|
| cpu0-3（A55, part `0xd05`） | 1,800,000 | 414 |
| cpu4-5（A76, part `0xd0b`） | 2,256,000 | 1002 |
| cpu6-7（A76, part `0xd0b`） | 2,304,000 | 1024 |

同一个 A76 集群内部**并不同频**（1002 vs 1024）。所以「按容量相等分组、取最大组」只得 2 核，「按 cpufreq policy 取最高频簇」也只得 2 核（`policy6` = cpu6,7）；只有按 50% 切才干净地拿到全部 4 个 A76。同构机器（容量全相等）→ 全选，即用满所有核。

### 2. 板端实测（aarch64，RK3588）

```
st_perf_cpus()  →  n_cpus=8   perf_n=4   perf_cpus=4 5 6 7
```

| 场景 | 结果 | 与旧行为 |
|---|---|---|
| 不设 env | `tp_threads = 4` | **不变**（推导值恰等于旧硬编码值） |
| `OMP_NUM_THREADS=8` | `8` | 显式优先，不再被 4 卡住 |
| `VLLM_THREADS=6` | `6` | 别名照旧生效 |
| `VLLM_TP_BIND=1` | worker tid0/1/2 → cpu 4/5/6；caller tid3 → cpu 7 | **不变**（`{4,5,6}∪{7}` ≡ 旧 `4+(slot%4)`） |
| 不设 `VLLM_TP_BIND` | 全核掩码（绑定默认关闭） | 不变 |

构建与自检：`build_rc=0`；`--test-l3` **17 PASS / 0 SKIP / 0 FAIL**；`--test-sparse` 9 PASS / 0 FAIL；`--bench-mixed`、`--npu-selftest` 均 rc=0。x86 侧 `-fsyntax-only` rc=0，且 x86 分支的 `nc-2` 一行未动。

### 3. 边界（照实标注，不夸大）

- **回退不是 4**：`cpu_capacity` 读不到时（老内核 / 容器 / 裁剪过的 sysfs）回退为**全部逻辑核**。RK3588 上该文件存在、推导得 4，故板端零变化；但在没有该文件的板子上默认会吃满所有核 —— 那类设备必须用 `OMP_NUM_THREADS` 显式指定。
- 本次**只改默认值的来源，未触碰任何计算路径**（并行仍按 idx 静态切分）。端到端位级对照见下一节。
- 板端二进制随之变更为 `b3c03027f2b0747b7bf338f33656c083`（920,712 B；旧 `a1b6707de9c925536df980f35aae6a1e` / 916,608 B）。**这不是位级回归** —— 差 4,104 字节来自新增的推导代码本身。

### 4. 全新克隆 + 运行时复验（同日补做）

上面几条只证明了「编译过 + 自检过」。这一节回答两个更硬的问题：**从仓库重新拉取的代码是否自足**，以及**改线程后引擎在板端是否真的跑得对**。

**4.1 全新克隆（不复用任何已有检出）**

```
git clone <gitee> /mnt/emmc/kestrel_fresh
head = 54dd39b   tree = c6e75119   branch = master   dirty = 0
old_hardcode_gone=yes   st_perf_cpus_hits=2   tp_perf_cpu_list_hits=2
build_rc = 0   920,712 B   md5 = f196b4beada43d95d6b5ca3b3946d410
--test-l3 17 PASS / 0 SKIP / 0 FAIL   --test-sparse 9 PASS / 0 FAIL
--bench-mixed rc=0                    --npu-selftest rc=0
```

该 md5 与 `kestrel_pull` 的 `b3c03027` 不同是**预期**的：两份检出的绝对路径不同，而 `stb_image.h` 的 `__FILE__` 会被嵌进二进制。两者**字节数完全相同**（920,712）正说明差异只是路径内容 —— 与「47 字节 = 20 B build-id + 27 B `__FILE__`」那次定位一致。

**4.2 位级对照：新二进制 ≡ 旧被验证二进制**

同一模型（Qwen3-VL-2B）、同参数，新旧二进制**交错**跑：

| 负载 | 运行次数 | `distinct_ids_md5` |
|---|---|---|
| 32-token 提示 | 10（new×3 + old×3 + `t2`/`t8` 各 2 + 预热） | **1** |
| **502-token 提示** | **12**（new/old 各 4 轮，顺序平衡） | **1** |

`ids_md5 = 5f6f3007950b0f4ae43c3411ec4b3139`，`text_md5` 亦全同。即：**贪心解码的 token 序列与解码文本，在新旧二进制之间、以及 `t2`/`t4`/`t8` 各档之间，全部逐位相同** —— 与「并行按 idx 静态切分、不改浮点步序」的说法吻合。上一节那句"未做端到端对比"至此作废。

**4.3 性能：无显著影响（且此前的"变慢"是噪声）**

首次 32-token 对照曾显示 new 的 prefill 比 old 高约 14%。把预填放大到 502 token、并让 new/old 交替当"先跑的那个"（消除顺序效应）后：

```
new prefill_s: 8.372 8.448 8.560 8.625   中位 8.504
old prefill_s: 8.440 8.454 8.537 8.634   中位 8.496     → 差 0.1%
new decode ms/tok: 84 86 86 88   中位 86
old decode ms/tok: 83 83 84 87   中位 83.5              → 差约 3%（噪声量级）
```

机理上也对得上：默认路径下 `VLLM_TP_BIND` 关闭，`tp_bind_worker` 在调用 `st_perf_cpus()` **之前**就 return 了，故改动在默认路径只多出池初始化时的一次 8 次 `fopen`。**那条 14% 是单次 0.5 s + 顺序未平衡造成的噪声**，不主张也不承认它。

顺带实践验证了推导值：默认档（推导 = 4）prefill 0.62 s，`t2` 0.82 s、`t8` 0.79 s（新旧二进制皆然）—— 与文档"A55 拉进 GEMM 会变慢"一致，推导出的 4 确实是该板最优档；显式 `OMP_NUM_THREADS` 仍可覆盖。

**4.4 运行时（serve 端到端）**

```
模型加载 124 ms；/health 2 秒内就绪；/v1/models 正常返回
warmup ttft=0.721s
warm_1 ttft=0.081s total=2.383s tokens=32 tpot=74.3ms
warm_2 ttft=0.081s total=2.379s tokens=32 tpot=74.1ms
输出文本正常（英文完整句）；KV 前缀复用生效
退出后无残留进程；dmesg OOM = 0；板端三个检出 dirty=0
```

**4.5 过程中我自己的三个测量错误（照实记，避免别人再踩）**

1. 第一次 A/B 报出"TOKIDS 不一致"：我把 `t=1 ms=257 rss=…kB` 这些**计时字段**一起 hash 了。真实输出是**每 token 一行**，只有 ID 列确定 —— 这是一次**看起来像回归的假告警**，靠回看原始日志才发现。
2. `one_run warmup_new NEW default` 传了**字面量 `NEW`** 而非 `"$NEW"` → `rc=127`，10 次运行全部空转（日志里 `n_ids=0`）。
3. 板端脚本里又留了一处中文串（prompt），PowerShell 按 GBK 解码吞掉续行符 → `Syntax error near "("`，脚本中途死掉。**板端脚本必须纯 ASCII**，这条规则再次被违反并再次付出代价。

### 5. 受影响文件

| 文件 | 改动 |
|---|---|
| `include/common/vllm_platform.h` | 新增 `st_perf_cpus()` / `ST_PERF_CPU_MAX`（POSIX 读 sysfs；Windows 分支退化为"全部 CPU"） |
| `src/core/vllm_tp.c` | 线程数默认值去掉 `>4 → 4` 上限；`tp_bind_worker` 改为在推导集合内轮转（含一次性懒解析缓存） |
| `src/model/vllm_safetensors.c` | 两处 `st_default_threads()` 调用点注释 + `st_default_threads()` 头注释 |

---

## 2026-09-16（发布面）— 板端源码树归一到 `fb3008c`；发布面核实与 GitHub 门面同步

### 0. 板端 `kestrel_pull` 已归一（并且把性能结论保住了）

此前板端源码树停在 `4f1ad70`（tools 未归档），与仓库差一代。本轮：

1. **先可回退**：整树 `tar` 备份（`/mnt/emmc/kv2h2/kestrel_pull_4f1ad70.tgz`，5.9 MB）
   + 本地改动 patch（`kestrel_pull_local_edits.patch`，56 KB）
   + 未跟踪残留移出至 `kestrel_pull_untracked_bak/`（`.l3prof_bak/`、4 个 `*.bak_*`、`tools/bench/`，
   后者恰是 HEAD 的已跟踪路径，不移开就会阻塞 `reset --hard`）。
2. `git fetch` + `reset --hard origin/master` → `head=fb3008c`、`tree=139683d8…`、`dirty=0`；
   `tools/` 变为归档布局（`bench build client drivers kernels npu ops preproc relay security`）。
3. **同路径重建 → 逐字节复现被验证二进制 `a1b6707de9c925536df980f35aae6a1e`（`SAME_AS_VALIDATED`）**。
   因为源码路径未变，`stb_image.h` 经 `assert()` 嵌入的绝对路径也一致。
   ⇒ **§7 的性能结论仍然有效，不需要重跑基准**（这一点此前是"待定"，现已闭合）。
4. 自检：`--test-l3` **17 PASS / 0 FAIL**、`--test-sparse` 9/9、`--bench-mixed`、`--npu-selftest` 均 rc=0。

### 1. 先弄清 `_release_verify/` 是什么（**此前对它的判断是错的**）

`_release_verify/{gitee,github}` **不是"导出树"，而是两份验证克隆**：

- `gitee/`  → clone of `https://gitee.com/pei-xiaoguang/kestrel-llm.git`（branch master）
- `github/` → clone of `git@github.com:m13253246268-ship-it/kestrel-llm.git`（branch master）

用途是核对「两个站点上**真正发布出去**的内容」。而 GitHub 侧不是 master 的镜像：它由本仓库的
**`gh` 门面分支**承载（GitHub 以英文作门面），靠定期把 master 合并进 `gh` 来维护。

此前「快照是 09-12 世代、`src+include` 62 个文件里 56 个不同、没有 §7」的说法**主语搞错了**：
那是这两个克隆的**工作区**与仓库的差异，且主因是 `core.autocrlf=true`（克隆在 Windows 上检出
→ 整树 CRLF），不是任何"净化规则"；真正的落后是「克隆 HEAD 停在 `4f1ad70` / `2ed2e08b`」。

### 2. Gitee 侧：已同步并验证

`_release_verify/gitee` 原先停在 `4f1ad70`，且有 10 个未跟踪目录挡住 `pull`
（09-15 有人手工放入归档后的 tools 内容而未提交）。处理：把未跟踪项移到
`_release_verify/_untracked_bak_gitee/`（54 文件），再 `git pull --ff-only`。
结果 `head/tree = 81cd44b / 069bc4f9…`，与本地 master **逐字节相同**，`dirty=0`。

### 3. GitHub 侧：`gh` 门面分支落后 9 个提交且已分叉，本轮已合并并推送

`gh` 与 `master` 的 merge-base 是 `4f1ad70`：`gh` 落后 9 个、master 落后 6 个。
`git merge-tree` 预演显示冲突**只有 3 个文件**，且全部来自那次 i18n 互换本身：
`README.en.md`（gh 侧删除=改名）、`README.md`（内容冲突）、`wiki/Home.md`（自动合并成功）。

解析方式不是手工改，而是**用门面约定直接取内容**：新增 `tools/build/make_release.py`，
把门面约定固化成可执行定义（`git -c core.autocrlf=false archive <ref>` + GitHub 侧 i18n 互换），
用它产出目标内容来解析冲突。`gh` 经两轮合并：

| 轮次 | `gh` 起点 → 终点 | 合入的 master |
|---|---|---|
| 1 | `2ed2e08b` → `331800ce` | `81cd44b` |
| 2 | `331800ce` → **`f2cffd78`** | `4209982` |

**推送前先证是快进**：GitHub 侧 `master = 2ed2e08b`，而 `2ed2e08b` 是 `f2cffd78` 的祖先
（`git merge-base --is-ancestor` rc=0）⇒ 无覆盖、无 force。随即 `git push github gh:master`，
远端 `master` 回到 `f2cffd78`。注意 GitHub 上**没有** `gh` 分支：门面只以 `master` 的形式存在，
`gh` 是本仓库用来维护它的工作分支 —— 这是既定约定，不是遗漏。

### 4. 验证（git 层面，绕开文件系统）

用 `git ls-tree` / `cat-file` 逐文件比对。**刻意不经 tar 解包** —— Windows 侧 tar 会把部分
非 ASCII 文件名改坏，我据此一度误判出「gh 有 15 个乱码名重复文件」，随后被 `gh-only = 0` 证伪
（那 15 个是解包产物，GitHub 上并不存在）。

| 比对 | 结果 |
|---|---|
| `gh` 分支 ≡ `make_release.py --facade github`（425 文件） | **IDENTICAL**（ref-only 0 / facade-only 0 / content-diff 0） |
| `master` ≡ `make_release.py --facade gitee`（425 文件） | **IDENTICAL**（同上） |

即：门面约定被脚本 **完整** 刻画，`gh` 的合并结果与 master 只差那一层 i18n 互换。

推送/同步闭环上再核一遍**两个端点的 HEAD**（这才是"发布快照"的真正主语）：

| 端点 | HEAD | 与本地的关系 | 工作区 |
|---|---|---|---|
| `_release_verify/gitee` | `81cd44b` → 与 `master` 同步 | ≡ 本地 `master`（tree 逐字节相同） | `dirty=0` |
| `_release_verify/github` | `2ed2e08b` → `f2cffd78`，再由 `gh` 合并推进 | ≡ 本地 `gh`（tree 逐字节相同） | `dirty=0` |

github 克隆原先脏 26 项（3 个 `M` + 13 个 `D tools/*` + 10 个未跟踪 `tools/<子目录>/`）——
与 gitee 克隆当初同型（有人手工放入归档后的 tools 内容而未提交）。处理仍走"先备份再清"：
未跟踪项移到 `_release_verify/_untracked_bak_github/`，被跟踪文件的脏存成
`tracked_dirt.patch`（255 KB）后 `reset --hard HEAD`，再 `pull --ff-only`。

### 5. 顺带修掉脚本自身的两个 bug（都是实测踩出来的）

| bug | 现象 | 修法 |
|---|---|---|
| `git archive` 未关 `autocrlf` | 门面 425 个文件里 **251 个被转成 CRLF**，与仓库 blob 的 LF 不符 —— 这正是历史上"导出树与仓库逐文件全不同"的真因 | 显式 `-c core.autocrlf=false` |
| 隐式依赖 `HEAD` | 在 `gh` 分支上跑 `--facade github` 会因缺 `README.en.md` 直接失败，gitee 门面也会取错源 | 新增 `--ref`（默认 `master`） |

用法：

```bash
python3 tools/build/make_release.py --out <目录>/gitee  --facade gitee  --ref master --clean
python3 tools/build/make_release.py --out <目录>/github --facade github --ref master --clean
```

### 6. 一处「刻意保留」的澄清（避免误当成净化）

`vllm_shs` 在发布面残留 15 处是**刻意保留的历史证据**（`wiki/性能与基准.md` 6、
`wiki/优化配置与边界.md` 3、`tools/bench/bench_value.sh` 3、`tools/bench/bench_http_probe.py` 2、
`tools/preproc/_g256_conv.py` 1 —— v0 时期二进制名与板端旧路径）；另 2 处（`tools/drivers/README*.md`
的引擎名）已随 `0a77753` 改为 `vllm_kestrel`。GitHub 侧 wiki 链接**仍指向 Gitee**（未做 URL 改写）：
Gitee 是正式站点；若将来要以 GitHub 为主站，需单独一轮并同步改 `README` / `CONTRIBUTING` / `.github` 模板。

### 7. 闭环状态与后续规矩

**两个老问题都已闭合**：

1. **板端源码树**：`4f1ad70` → `fb3008c`，且同路径重建**逐字节复现被验证二进制**（§0），
   §7 性能结论无需重跑。
2. **发布快照**：`gh` 合并后 `push github gh:master`（GitHub 侧 `master` 由 `2ed2e08b` 快进而来），
   两份验证克隆各 `pull` 到与本地 **逐字节相同**的最新提交；旧克隆备份
   `_release_verify/_superseded_20260912_gitee_github.tgz`（7.0 MB）已删
   （两份克隆可随时重克隆，无信息损失）。

上表记录的是核对**当时**的哈希，刻意不追到"最后一次提交"：写这份文档这个动作本身也会让
`master` 前进，于是 `gh` 又落后一个提交 —— 追哈希追不完。收尾动作固定为以下三步，做完再各
`pull --ff-only` 一次，并在**最终头**上重跑两次 `IDENTICAL` 校验。

一个**不变量**取代了逐次追哈希：`gh` ≡ `make_release.py --facade github`（源自 `master`）。
每次 `master` 有新提交后重跑以下三步即可维持：

```bash
python3 tools/build/make_release.py --out <tmp>/github --facade github --ref master --clean
git checkout gh && git merge master          # 冲突只会落在 README*.md / wiki/Home.md
git push github gh:master && git checkout master
```

`gh` 相对 master 落后几个提交**本身不是问题**（它就是这样被维护的）；真正会出问题的是
"两侧内容不再满足上述不变量"，而那由 `verify_ref.py` 逐文件强制校验。

### 8. 「GitHub 上那份代码到底能不能编译、能不能跑」——真拉一份下来验（同日追加）

§4 只到 git 层面（`ls-tree` / `cat-file`）证明了"内容相同"，没回答**可编译性**。本轮补上：
真从 GitHub 克隆一份，两侧各自构建并跑起来。

**通道（先说清怎么拉的）**：板端到 github.com **不通** —— `curl https://github.com/` 逾时、
`git ls-remote https://…` 挂死、`Failed to connect to github.com:443 after 21052 ms`；只有
`api.github.com` 通（`code=200, time=0.45 s`）。所以改为**本地拉 GitHub、再把同一份树送板端**。

全量克隆卡在 `Receiving objects: 96% (944/983)`（同一条 SSH 通道，pack 才 ~2.5 MiB），
改 `--depth 1 --branch master` 后一次成功（1.44 MiB，rc=0）：

| 项 | 值 |
|---|---|
| remote | `git@github.com:m13253246268-ship-it/kestrel-llm.git` |
| HEAD | `8422e7311895f6831c696cd55e80844f282375d0` |
| TREE | `862b2c786b3228f1260109fae7b227c187767f75` |
| branch / dirty / tracked | `master` / 0 / **425** |
| 门面命名 | `README.md`（英文门面）+ `README.zh-CN.md` 在，**`README.en.md` 不在**，`wiki/Home.md` 在 |

**x86 侧**（用门面树自带的 `tools/build/check_x64.ps1`；`-OutDir` 必须指向纯 ASCII 路径，
以绕开 MinGW `ld` 打不开非 ASCII 输出路径的既有缺陷）：

```
=== [1/3] build x64 ===   rc=0
--test-l3       PASS=16 FAIL=0   -> OK
--test-sparse   PASS=9  FAIL=0   -> OK
sha256 = E50FB5685AA08B2A615E703942FB47F89367A11436DE04EFB9BFDFFCF9DB1AC0
X64 CHECK PASSED
```

**aarch64 侧**（`git archive` 打包 → scp → 解到全新目录 `/mnt/emmc/gh_verify_8422e73`；
tar 两侧 md5 `b490b1445cd0f6dc7fce113562bf3ef3` / 10,158,080 B 回验一致）：

```
build_rc=0   exe_size=920712   exe_md5=cf3b3e10bb175af602dd0843e2abbb51
--test-l3      rc=0 PASS=17 SKIP=0 FAIL=0   -> L3 self-test PASSED (0 failures)
--test-sparse  rc=0 PASS=9  FAIL=0          -> Sparse self-test PASSED (9 checks)
--bench-mixed  rc=0
--npu-selftest rc=0
```

x86 数是 16 PASS（(1b)「P2 restore NEON vs scalar」在 x86 打 `[SKIP]`），aarch64 是 17 ——
差的是那一项，不是回归。

**端到端生成**（默认线程档，不设 `OMP_NUM_THREADS` / `VLLM_THREADS`，`--stream-test` 502-token 上下文）：

```
stream_rc=0   prompt_tokens=502
prefill 502 tokens in 8.610s (58 tok/s) [batch]
decoded 23 tokens, 86 ms/tok (11.69 tok/s)
ids_md5 = 5f6f3007950b0f4ae43c3411ec4b3139
stray=0   oom=0
```

`ids_md5` 与既有 502-token 基线的 `5f6f3007…` **逐位相同** ⇒ 门面树（i18n 互换后）与 `master`
在**数值行为上不可区分**，而不只是 git 对象相同。

**踩到的两个坑（都是验证方的，不是仓库缺陷）**：

| 坑 | 现象 | 真因 / 修法 |
|---|---|---|
| 手搓 `git archive` 忘关 `autocrlf` | 板端 `build_rk3588.sh` 第 32 行 `set -euo pipefail` 报 `pipefail: invalid option name`（实际是 `pipefail\r`），`build_rc=2`、`NO_BINARY` | 与 §5 同根：本机 `core.autocrlf` 会让 `git archive` 把 LF 转 CRLF。**仓库内容本身是 LF**（克隆工作区 CR 计数 = 0）。显式 `git -c core.autocrlf=false archive` 后 tar 由 10,260,480 B 降到 10,158,080 B —— 正好是去掉的那 102,400 个 CR。**凡是对外发 tar / 做导出，必须带这个参数**；这是 §5 那条 bug 的第二次现身 |
| `pkill -f gh_verify_run.sh` | ssh 立刻断连（退出码 -1），脚本根本没起来 | `-f` 匹配**整条命令行**，把 ssh 自己也算进去了。探针与清理一律用自排除写法：`gh_verify_ru[n]` |

板上留存证据：`/mnt/emmc/kv2h2/gh_verify_run.log`、`ids.sh`，源码树 `/mnt/emmc/gh_verify_8422e73`（15 MB）。
本轮**未改仓库内容**（门面树的 x86/aarch64 两侧复验通过，无须修）。

---

## 2026-09-16（交叉验证）— 从 gitee 全新克隆到板端编译，暴露并修复 2 处只有 ARM 才显现的缺陷

### 0. 做法

在板端**独立克隆** gitee（`/mnt/emmc/kestrel_gitee`，**不触碰**被验证树 `/mnt/emmc/kestrel_pull`），
构建后跑自检，再与本地 x86 对照。脚本 `sh /mnt/emmc/kv2h2/board_build_verify.sh`
（内部先 `git fetch` + `reset --hard origin/master`，并自证 HEAD / tree / dirty）。

### 1. 暴露的缺陷（**都不是本轮引入，但都只有 ARM 侧才显现**）

| # | 位置 | 现象 | 根因 | 修复 |
|---|---|---|---|---|
| 1 | `src/model/vllm_safetensors.c` | 板端 `./build_rk3588.sh` **BUILD_RC=2**：`fatal error: ../../tools/llama_gemm_q4_0_4x4_asm.c: No such file or directory` | `0ac05cc` 把 `tools/` 归档进子目录时把该文件移到 `tools/kernels/`，**漏改了这条 include**。它位于 `#if ST_NEON_DOTPROD` 块内（3806 行起），x86 编不到 ⇒ **此前 x86 全绿把它掩盖了** | 改为 `../../tools/kernels/llama_gemm_q4_0_4x4_asm.c`（`4adeca8`） |
| 2 | `src/main.c` | 构建与自检都能过，但二进制里带着旧名 banner | `d91c18d` 夹带的旧世代命名里，我上一轮只 grep 了小写 `vllm_shs`，**漏了大写 `vLLM-SHS`**（文件头注释 + 启动 `printf` 共 2 处；banner 是字符串常量、会进二进制） | 改为 `vLLM-Kestrel`（`b3928f8`） |

全仓审计确认 #1 是唯一的功能性 stale 路径（`src/`+`include/` 的 `#include`、CMakeLists、`cmake/`、
构建脚本全部核对）。#2 修完后 `vLLM-SHS` 全仓为 0，且 `src/main.c` 与板端被验证树**逐字节相同**
（md5 `58c838f0e66536b34ba61e3e331056b0`）。注：同文件第 4 行 `SHS axiom constraints` 指算法名
（Superposition Hybrid System），板端同样保留，未改。

### 2. 验证结果

| 项 | 本地 x86（MinGW） | 板端 ARM（RK3588） |
|---|---|---|
| 源码自证 | HEAD `b3928f8` / tree `d575ae44…` | 克隆的 HEAD / tree 与本地**逐字节一致**，`dirty=0` |
| 构建 | `X64 CHECK PASSED` | **`BUILD_RC=0`**，`vllm_kestrel` 916,608 B aarch64 |
| `--test-l3` | 16 PASS + 1 SKIP(NEON) / **0 FAIL** | **17 PASS / 0 SKIP / 0 FAIL**（含 x86 上被 SKIP 的 NEON 对拍：dequant 0/200、store 0/200 不一致） |
| `--test-sparse` | 9/9 | 9/9 |
| 其它 | — | `--bench-mixed`、`--npu-selftest` 均 RC=0 |

本地产物 `vllm_kestrel_x64.exe` 1,236,786 B，md5 `7dc953221c083268e5dd51b5c0740a8b`。

### 3. 板端克隆构建 vs 被验证二进制 `a1b6707d…`：差异已定位到字节

`a1af6d78…`（克隆构建） vs `a1b6707d…`（被验证）：**同为 916,608 B，仅 47 字节不同**：

| 偏移 | 字节数 | 内容 |
|---|---|---|
| 668–687 | 20 | `.note.gnu.build-id` 的 ID |
| 827,258–827,284 | 27 | `.rodata` 中第三方头 `stb_image.h` 的 `assert()` 经 `__FILE__` 嵌入的**源码绝对路径**：`/mnt/emmc/kestrel_pull/…` vs `/mnt/emmc/kestrel_gitee/…` |

构建配置的唯一差异同样只是源码/构建目录的绝对路径（`CMakeCache.txt`、`flags.make`）。
**收口实验**：在板端**同一路径**原地重建 `kestrel_pull` → **逐字节复现 `a1b6707d…`（`REPRO_OK`）**。
⇒ 构建可复现；47 字节差异纯由「克隆到哪里」造成，**代码层面 gitee HEAD 与板端被验证源码等价**。
被验证二进制已备份至 `/mnt/emmc/kv2h2/vllm_kestrel_a1b6707d.bak`。

### 4. 遗留

- 板端 `kestrel_pull` 仍停在 `4f1ad70`（tools 未归档），**刻意不动**：它正是 `a1b6707d` 的来源，
  且已被证明与 gitee HEAD 代码等价。若要统一，需另起一轮并重跑 §7 基准。
- 板端 `kestrel_pull` 内有 `src/*.bak_*` 等未跟踪残留；`/mnt/emmc` 余量 975 MB。

---

## 2026-09-16（收尾）— 修复：`d91c18d` 夹带旧世代命名，导致 x64 断链

### 0. 问题

`d91c18d`（本轮 L3 改动那次提交）在提交 `src/main.c` / `src/serve/vllm_server.c` /
`src/model/vllm_safetensors.c` 时，把工作副本里**未提交的旧世代命名**一并带了进去 ——
这 3 个文件里的引擎符号被写成 `vllm_shs_*`，而定义方
（`include/core/vllm_superpos.h`、`src/core/vllm_scheduler.c`）仍是 `vllm_kestrel_*`。

**后果：`HEAD` 链不起来。** 实测（x86 MinGW，`tools/build/build_x64.ps1`）：

```
undefined reference to `vllm_shs_init'
undefined reference to `vllm_shs_generate'
undefined reference to `vllm_shs_cleanup'
collect2.exe: error: ld returned 1 exit status
```

该提交已推送 gitee，即远程源码一度处于不可链接状态。

### 1. 为什么回改到 `vllm_kestrel`（而不是把引擎改名为 `vllm_shs`）

| 判据 | 结论 |
|---|---|
| 仓库**首个提交** `63cba68`（2026-09-05） | 标题即 `feat: 发布 Kestrel (vllm_kestrel) …（双许可）`；仓库从第一天就用 kestrel |
| 板端被验证的源码树 / 二进制 | `/mnt/emmc/kestrel_pull`，二进制名 `vllm_kestrel`（md5 `a1b6707d…`），其 `main.c` 用 `vllm_kestrel_init` |
| 签名绑定 | `include/model/vqf_format.h` 的 `VQF_SM2_ID = "vllm-kestrel-vqf"` 是 SM2 的 Z_A 绑定串，已发布 VQF 权重按此 ID 签名 |
| 全部文档与 harness | README / README.en / wiki / 归档 harness（`kv2_run2.sh`、`moe8k.sh`）一律 kestrel |
| `vllm_shs` 的实际分布 | 只在**本地未提交的工作副本**（12 个文件）与 2026-09-07 的源码备份 `备份/ARM_SRC_20260907_V0`（该备份 0 处 kestrel），与仓库时间线无关 |

### 2. 改动：只回改行为性/对外可见的 9 处

| 文件 | 处数 | 性质 |
|---|---|---|
| `src/main.c` | 3 | `vllm_shs_init/_generate/_cleanup` → `vllm_kestrel_*`（**修链**） |
| `src/serve/vllm_server.c` | 5 | `owned_by`、响应 `id`、`/health` 的 `server` 三个对外串 + 2 处注释 |
| `src/model/vllm_safetensors.c` | 1 | 注释里的基准二进制名 `vllm_shs_x64` → `vllm_kestrel_x64`（与板端同一行一致） |

**刻意保留不改**（历史证据，改了反而不准）：`wiki/性能与基准.md`(6) 与 `wiki/优化配置与边界.md`(3)
里 v0 时期（2026-09-05 测点）的 `./vllm_shs` 命令；`tools/bench/bench_value.sh`(3)、
`tools/bench/bench_http_probe.py`(2)（板端同款，且探针已同时兼容两个名字）；
`tools/preproc/_g256_conv.py`(1)（板端旧路径的一次性转换命令）。

**未改动 `VQF_SM2_ID`**：工作副本把它改成了 `"vllm-shs-vqf"`，这会让已签名 VQF **验签失败**（功能性
改动，非改名）。它随同批旧世代命名一并丢弃。

### 3. 同时丢弃的旧世代改动（未提交）

`git checkout --` 回退 13 个文件：`include/common/vllm_platform.h`、`include/core/vllm_superpos.h`、
`include/model/vqf_format.h`、`include/npu/vllm_npu.h`、`include/serve/embedded_web.h`、
`include/serve/vllm_server.h`、`src/common/vllm_crypto.c`、`src/core/vllm_scheduler.c`、
`src/npu/vllm_npu.c`、`src/serve/admin.html`、`src/serve/chat.html`、`src/serve/embedded_web.c`、
`src/serve/vllm_admin.c`。

另 5 个文件（`include/core/vllm_ep.h`、`include/model/vqf.h`、`src/core/vllm_ep.c`、
`src/core/vllm_tp.c`、`src/model/vqf.c`）经 `git hash-object --no-filters` 比对与 `HEAD` **逐字节相同**，
`git status` 里的 ` M` 只是 `core.autocrlf=true` 的行尾口径噪声，无需处理。

回退前已把全部工作区改动导出为 patch：`.tmp_tok/l3prof/pre_tidy_worktree.patch`（2.94 MB，可回放）。

### 4. 验证

| 项 | 结果 |
|---|---|
| 反向验证（换回 HEAD 版 `main.c`） | x64 链接失败，报上述 3 条 `undefined reference`，无产物 |
| 修复后构建 | `vllm_kestrel_x64.exe` 1,236,786 B，md5 `742fb4e28de8ea33f34623343c102112`；日志中 `error:` / `undefined reference` **零条** |
| `check_x64.ps1` | **`X64 CHECK PASSED`**（构建 + `--test-l3` 16 PASS + 1 SKIP(NEON) / **0 FAIL** + `--test-sparse` 9/9） |

### 5. 已知遗留

- **板端源码树仍停在 `4f1ad70`**，未同步 `0ac05cc` 的 tools 目录归档（板端 `tools/` 仍是扁平结构）。
  本次修复不需要动板端（板端本就使用 kestrel），但这条「仓库 vs 板端」的世代差需单独处理 ——
  此前「板端树 = 仓库」的说法对 `src/main.c` 及 tools 路径并不成立。
- 仓库外的散落工作副本（`d:\项目\New_vLLM\src`、`d:\项目\New_vLLM\tools`）仍带旧世代命名，
  不在本仓库范围内，未处理。

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
