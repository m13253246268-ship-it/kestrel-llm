[English](README.en.md) | **简体中文**

# 密文链接力工具（lay / boot）

这套脚本用于 **RNS-CKKS 密文推理链的「跑层 → 交接 → 复核」**：逐层执行 `lay`（层内前向）+ `boot`
（自举刷新），把结果打包交给下一棒，并让第三方在不信任打包者的前提下完成校验。

驱动源码在 [`../drivers/`](../drivers/)（`t23_m3p.c` / `t23_chain.c`），引擎源码在 `src/core/`。

## 0. 环境前提

| 项 | 说明 |
|---|---|
| gcc | 支持 C11 + OpenMP（MinGW-W64 或 Linux gcc 均可）；只依赖 `libc + libm` |
| PowerShell | Windows 下用 PowerShell 5.1 及以上 |
| 数据包 | **仓库内不含数据**（约 7 GB）。引擎把数据路径硬编码到 `.tmp_tok/`，需自行生成，见 [`../preproc/README.md`](../preproc/README.md) |
| 运行目录 | **所有脚本都在仓库根目录运行**（脚本内部会自行定位仓库根，不必手动 `cd`） |

数据就位后，`.tmp_tok/` 里至少要有：`chain/sk.bin`、你这一棒的输入密文 `chain/u{L-1}r112_{t}_{h}.ct`、
`tail/w{L}/`、`tail/l{L}_*.bin`、`tail/scale{L}.bin`。缺一个会在加载阶段直接报错退出。

> **先建目录**：`New-Item -ItemType Directory -Force .tmp_tok, .tmp_tok\chain`。
> 驱动用裸 `fopen` 落盘、**不会自建目录**：缺 `.tmp_tok/chain/` 会报 `[FAIL] ct save` / `[FAIL] sk save`；
> 缺 `.tmp_tok/` 时 gcc 的 `-o` 也会报 `cannot open output file`。
> `rerun5_lay_boot.ps1` 会自己准备好这两个目录（以及 `relay_rerun/`），其余脚本请自行确保。

> `chain/sk.bin` 可由驱动自身生成：建好 `.tmp_tok/chain/` 后，`T23_PHASE=l0 T23_KEYONLY=1` 跑一次 `t23lay`，
> 会打印 `KEYONLY: sk.bin saved`。

## 1. 工具一览

| 工具 | 作用 | 关键参数 | 主要产物 |
|---|---|---|---|
| [rerun5_lay_boot.ps1](rerun5_lay_boot.ps1) | **跑层**：逐层 `lay`+`boot`，缺产物自动跳过（断点续跑） | `-From -To -LayExe -BootExe -LogDir` | `.tmp_tok/chain/*.ct`、`.tmp_tok/relay_rerun/{lay,boot}*.out`、`timing.tsv` |
| [verify_layer.ps1](verify_layer.ps1) | **验算**：解密 → 解码 → 对明文参考算 `max\|err\|`（独立于驱动 PASS 判据） | `-From -To -Mode lay\|boot -Tol` | `PASS/FAIL` + `max\|err\|`；退出码 = 失败层数 |
| [layver_run.ps1](layver_run.ps1) | **数值等价性**：只换输入密文、不改线程数，看下一层 `max\|err\|` 是否仍达标 | `-TargetLayer -AltDir -BaselineLog` | `.tmp_tok/relay_state/layver_result.txt` |
| [collect_results.ps1](collect_results.ps1) | **归档**：把某层范围的密文/日志/元数据收齐并生成 `manifest.sha256` | `-From -To -LogDir -Dest` | `<Dest>/chain`、`<Dest>/logs`、`manifest.sha256` |
| [pack_relay.ps1](pack_relay.ps1) | **打包**：把某层范围打成可上传的 zip 交接包 | `-From -To -Id [-NoKey]` | `.tmp_tok/relay_out/relay_L{a}-{b}_{id}_{时间戳}.zip` |
| [verify_relay.ps1](verify_relay.ps1) | **验包**：校验 zip 的完整性 + 判据 + 结构 | `-Zip [-Keep]` | 控制台明细；退出码 = 失败项数 |
| [watch_relay.ps1](watch_relay.ps1) | **无人值守**：每 15 分钟记一次状态，条件满足则自动开跑并在结束后归档+验证 | `-From -To -IntervalSec -RefDir` | `.tmp_tok/relay_state/` 下的 tick 日志与验收摘要 |
| [_monitor.ps1](_monitor.ps1) | 后台进度监控（辅助） | 无 | `.tmp_tok/relay_rerun/_status.txt`（每分钟刷新） |
| [fix_crlf.sh](fix_crlf.sh) | 把代码树的 `CRLF` 统一成 `LF`（跨平台导出后 `.sh` 无法执行时用） | `[目录]`（默认仓库根） | 仅转换文本文件，`.git/` 一律不动 |

三个二进制由脚本自动构建（缺失时）或按需手动构建：`t23lay`（np=112）、`t23boot`（np=2100）、
`verify_layer`（np=112），默认输出到 `.tmp_tok/`。

## 2. 手动构建（可选，脚本缺失时会自动做）

```bash
# lay（层内前向，112 素数）
gcc -O2 -fopenmp -Wno-implicit-function-declaration \
    -I include -I include/core -I include/common \
    -DCKKS_N=2048 -DCKKS_NPRIMES=112 -DBB=32 -DGG=32 \
    src/core/vllm_ntt.c src/core/vllm_ckks.c src/core/vllm_tp.c \
    tools/drivers/t23_m3p.c -o .tmp_tok/t23lay -lm

# boot（自举刷新，2100 素数；大栈必需）
gcc -O2 -fopenmp -Wno-implicit-function-declaration '-Wl,--stack,33554432' \
    -I include -I include/core -I include/common \
    -DCKKS_N=2048 -DCKKS_NPRIMES=2100 -DBB=32 -DGG=32 \
    src/core/vllm_ntt.c src/core/vllm_ckks.c src/core/vllm_tp.c \
    tools/drivers/t23_chain.c -o .tmp_tok/t23boot -lm

# verify_layer（独立复核，112 素数即可）
gcc -O2 -fopenmp -Wno-implicit-function-declaration \
    -I include -I include/core -I include/common \
    -DCKKS_N=2048 -DCKKS_NPRIMES=112 -DBB=32 -DGG=32 \
    src/core/vllm_ntt.c src/core/vllm_ckks.c src/core/vllm_tp.c \
    tools/relay/verify_layer.c -o .tmp_tok/verify_layer -lm
```

编译期坑：

1. `-I` 必须同时给 `include`、`include/core`、`include/common`（`vllm_tp.c` 依赖 `common/vllm_platform.h`）。
2. PowerShell 下 `-Wl,--stack,33554432` 的**逗号会被当成参数分隔符** → 整段加引号。
3. `boot` 用 np=2100，MinGW 默认栈会溢出（`0xC00000FD`）→ 必须加大栈。
4. 含中文注释的 `.ps1` 必须存为 **UTF-8 with BOM**，否则 PowerShell 5.1 按 ANSI 读会报「意外的标记 }」。

## 3. 典型工作流

**接力者（跑几层并提交）**

```powershell
# 1) 确认输入就位：chain\u{L-1}r112_*.ct 与 tail\w{L}\ 存在
# 2) 跑层（T23_NT 未设置时默认 4）
$env:T23_NT = "4"
powershell -NoProfile -ExecutionPolicy Bypass -File tools\relay\rerun5_lay_boot.ps1 -From 5 -To 7
# 3) 自检准确度（可选，需要数据包里的明文参考）
powershell -NoProfile -ExecutionPolicy Bypass -File tools\relay\verify_layer.ps1 -From 5 -To 7
# 4) 打包提交
powershell -NoProfile -ExecutionPolicy Bypass -File tools\relay\pack_relay.ps1 -From 5 -To 7 -Id yourname
```

**验证者（收包复核）**

```powershell
# 1) 先核对 zip 本身的 SHA256（发布方公布值）
Get-FileHash .tmp_tok\relay_out\relay_L5-7_yourname_<时间戳>.zip -Algorithm SHA256
# 2) 包内逐字节 + 判据 + 结构
powershell -NoProfile -ExecutionPolicy Bypass -File tools\relay\verify_relay.ps1 -Zip <包路径>
# 3) 有数据包时可进一步判「算得对不对」
powershell -NoProfile -ExecutionPolicy Bypass -File tools\relay\verify_layer.ps1 -From 5 -To 7
```

**复现者（从种子跑 L0–4）**

```powershell
# 前置：备份并清空 chain\ 的 *.ct（保留 sk.bin）
powershell -NoProfile -ExecutionPolicy Bypass -File tools\relay\rerun5_lay_boot.ps1 -From 0 -To 4
powershell -NoProfile -ExecutionPolicy Bypass -File tools\relay\verify_layer.ps1 -From 0 -To 4
```

## 4. 判据口径（最容易误判的地方）

跑起来后日志里会同时出现 `PASS` 和大量 `FAIL`，**这不是矛盾**：

- **只认层输出判据**：`lay` 看日志末尾 `RESULT=PASS`；`boot` 看 `BOOT=PASS` 且该层出现 8 次
  `out np=2083`（4 token × 2 分量，说明确实刷回满链）。
- **段级 A/B/C/D 的 `max|err| FAIL` 不是判据**：它对照的是「输入未偏离的明文参考」，而真实密文链
  上层间必然带偏离；归一化会把幅度拉回，最终段带 `[E2E:ref-deviate-ok]` 标记。

  **为什么"段级 FAIL、最终 PASS"不矛盾**（依据 `t23_m3p.c` 的 E2E 模式注释与规划脚本 `_cfold_plan.py`）：

  | 环节 | 说明 |
  |---|---|
  | 段级为何 FAIL | L26/27 使用**真实 C 折叠**（`g_causal` + 实 C 折叠），中间量 `C/D/U2` 偏离真值**属预期**，不是缺陷 |
  | 预期偏差的量级 | 规划脚本事先量化：`\|Δao\| ~ 1.4 / 0.8`（与实测 `[C:attn] 8.5~12`、`[D:o] 14~29` 同量级） |
  | 为何最终仍 PASS | 同一份规划给出**归一化判据**：`E2E logits drift 0.14 < margin 2.03`——中间偏差传到 logits 只剩 0.14，而 top1 的判决余量是 2.03 |
  | 判定移交到哪 | 该模式下 `C/D/U2` **仅报告 err、不计入 `fails`**，验收移交 **E2E logits 门** |

  **一句话**：段级 FAIL 是中间量的**口径差异**（可预期、已量化），最终 PASS 是**端到端 logits 门
  的判决**——两者不是同一把尺子，混着看就会得出"自相矛盾"的错觉。

  **注意：`fin` 阶段的门与上面不是同一套。** `C/D/U2` 由 `T23_E2EMODE` 控制（不设该变量时计入 `fails`），而 `fin`（phase 5）的 `[F:xnorm]` / `[F:top1]` 走的是**硬编码 `&rf`**——无论是否设置 `T23_E2EMODE`，都不计入 `fails`。**两套门不要混着读。**

  依据：驱动源码 `tools/drivers/t23_m3p.c` 中 `fin` 的两处检查传的计数器是 `&rf`（约 1533-1535 行），而 `C/D/U2` 传的是 `e2e ? &rf : &fails`（如约 1908、1973、2146、2161 行）；`[F:top1]` 不匹配也只是 `rf++`（约 1589 行）。

## 5. 边界（诚实说明）

- **`BOOT=PASS` ≠ 数值正确**：它只说明 `load / refresh / save` 没有失败。要判数值，必须另跑
  `verify_layer.ps1`（或 `layver_run.ps1`）。
- **`lay` 的 `RESULT=PASS` 也 ≠ 数值正确**：接力脚本设 `T23_E2EMODE=1`，把 `C/D/U2` 的偏差降级为"仅报告"，
  所以**跑错 SiLU 路径（`.tmp_tok/tail/silu{L}.bin` 缺失 → 浅层误用 ÷21 折叠）照样打 PASS**
  （实测层 0：`max|err|` 4.2e-2~8.5e-2，`verify_layer` FAIL，而驱动报 `RESULT=PASS (0)`）。
  这两条合起来意味着：**判据日志只能用来排障，验收必须看 `verify_layer` 的 `max|err|`。**
- **`verify_relay.ps1` 不能证明「计算是对的」**：它能证明包没被篡改/截断/夹带，判据日志理论上可伪造。
  防伪造靠**交叉复算**：流程确定性，两人独立重跑同一层应逐字节一致。
- **线程数是结果变量**：`layver_run.ps1` 存在的意义就是——换输入/换线程产出不同字节时，用
  `max|err|` 而非字节相等来判等价性。
- **`layver_run.ps1` 的 `-TargetLayer` 必须 ≥ 1**：`lay0` 走明文加密分支，不读任何链上密文，
  把替代 `u0r112` 注入后跑 `lay0` 等于什么都没比。
