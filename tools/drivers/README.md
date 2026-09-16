[English](README.en.md) | **简体中文**

# 层链驱动（lay / boot / fin）

这两个 C 文件是 **FHE 密文推理链的层驱动**，与 `src/core/` 里的 RNS-CKKS 引擎（`vllm_ntt.c` / `vllm_ckks.c` / `vllm_tp.c`）配套使用。
它们**不属于推理引擎本体**（`vllm_kestrel`），而是跑**同态加密测试 / 社区接力**用的驱动。

## 0. 能不能跑起来？

| 步骤 | 需要什么 | 状态 |
|---|---|---|
| **编译**（§4） | 只要 `gcc`（含 `-fopenmp`、`pthread`） | ✅ 仓库内即可完成 |
| **运行**（§5） | **~7 GB 数据包**（`sk.bin` + `tail/w{L}/` 权重 + `tail/l{L}_*` 明文参考） | ⚠️ **仓库内不含数据**，需自行生成：见 [`../preproc/README.md`](../preproc/README.md) |

> 实测（2026-09）：按 §4 两条命令原样编译均成功；在**没有数据包**的目录直接运行，会以
> `sk load fail` 退出。所以「编译没问题、运行要先备数据」是正常现象，不是环境坏。

三个容易踩的坑：

1. **`-o` 不要用含中文的绝对路径**（Windows/MinGW）：`-o D:\中文路径\x.exe` 会报
   `cannot open output file ...: No such file or directory`；用相对 ASCII 名（如 `-o t23lay`）即可。
2. **`lay0` 的"明文起点"是三个文件**：`chain/sk.bin`（可由 `T23_KEYONLY=1` 生成）+ `tail/l0_u1.bin`（输入参考）+ `tail/w0/`（权重）。排障先查这三样。
3. **`fin` 不能单独跑**：它读 `u27r112`，必须先跑完整条链（lay0→boot0→…→lay27→boot27）。

## 1. 是什么

| 文件 | 作用 | 阶段 |
|---|---|---|
| `t23_m3p.c` | `lay`（层内前向）+ `fin`（最终 logits） | `T23_PHASE=lay` / `fin` |
| `t23_chain.c` | `boot`（自举刷新，噪声预算复位） | `T23_PHASE=boot` |

## 2. 链协议（每层两跳）

```
lay{L}  : u{L-1}r112  ──▶  u{L}        判据 RESULT=PASS
boot{L} : u{L}        ──▶  u{L}r112    判据 BOOT=PASS（out np=2083）
fin     : u27r112     ──▶  logits      对照 tail/logits.bin 的 top1
```

每跳落 **8 个密文**（4 个 token × 2 个密文分量），即 `{prefix}_{t}_{h}.ct`，`t=0..3`、`h=0..1`。
层号从 0 起：`lay0` 从明文起点出发，`lay1` 读 `u0r112`，依此类推。

## 3. 数据目录约定（重要）

驱动内的数据路径**硬编码为 `.tmp_tok/`**，需在工作目录下准备：

```
<工作目录>/.tmp_tok/
├── chain/   sk.bin、u{L}_{t}_{h}.ct、u{L}r112_{t}_{h}.ct
├── tail/    w{L}/ 每层权重、l{L}_*.bin 明文参考、logits.bin、embed.bin、xnorm_f.bin 等
└── l0/ l1w/ l1r/   layer0 / layer1 专用权重与参考
```

数据包约 7 GB（权重 + 明文参考 + 私钥），需自行生成：见 [`../preproc/README.md`](../preproc/README.md)（权重大小/格式/生成顺序/校验都有说明）。
其中 `chain/sk.bin` 可由驱动自身生成：建好 `.tmp_tok/chain/` 后跑 `T23_PHASE=l0 T23_KEYONLY=1 ./t23lay`（会打印 `KEYONLY: sk.bin saved`）。**驱动不会自建目录**，`.tmp_tok/chain/` 缺了会报 `[FAIL] sk save`。

## 4. 编译

```bash
# lay / fin（层链 112 个 60-bit 素数）
gcc -O2 -fopenmp -Wno-implicit-function-declaration \
    -I include -I include/core -I include/common \
    -DCKKS_N=2048 -DCKKS_NPRIMES=112 -DBB=32 -DGG=32 \
    src/core/vllm_ntt.c src/core/vllm_ckks.c src/core/vllm_tp.c \
    tools/drivers/t23_m3p.c -o t23lay -lm

# boot（2100 素数；大栈必需，否则 MinGW 下 0xC00000FD 栈溢出）
gcc -O2 -fopenmp -Wno-implicit-function-declaration '-Wl,--stack,33554432' \
    -I include -I include/core -I include/common \
    -DCKKS_N=2048 -DCKKS_NPRIMES=2100 -DBB=32 -DGG=32 \
    src/core/vllm_ntt.c src/core/vllm_ckks.c src/core/vllm_tp.c \
    tools/drivers/t23_chain.c -o t23boot -lm
```

> PowerShell 下 `-Wl,--stack,33554432` 的**逗号会被当成参数分隔符** → 整段加引号。
> `-I` 必须同时给 `include`、`include/core`、`include/common`（`vllm_tp.c` 依赖 `common/vllm_platform.h`）。

## 5. 运行

| 环境变量 | 作用 | 例 |
|---|---|---|
| `T23_PHASE` | `lay` / `boot` / `fin` | `lay` |
| `T23_LAY` | 层号（仅 `lay`） | `5` |
| `T23_BI` / `T23_BO` | boot 的输入/输出前缀（仅 `boot`） | `u5` / `u5r` |
| `T23_E2EMODE` | 端到端模式（忽略参考偏离导致的段级 FAIL） | `1` |
| `T23_NT` | 自研线程池 `vllm_tp` 线程数 | `4` |

```bash
# 第 5 层的两跳
T23_PHASE=lay  T23_LAY=5 T23_E2EMODE=1 T23_NT=4 ./t23lay  > lay5.out  2> lay5.err
T23_PHASE=boot T23_BI=u5 T23_BO=u5r   T23_NT=4 ./t23boot > boot5.out 2> boot5.err
```

## 6. 判据（只认层输出）

- `lay` → 日志末尾 `RESULT=PASS`
- `boot` → `BOOT=PASS`，且该层出现 8 次 `out np=2083`（刷新回满链）
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

## 7. 线程数（实测结论：**必须分口径说**）

**加线程的效果取决于你看的是哪个阶段，两个口径结论相反。而且线程数本身会改变输出字节，
所以不能跨线程数混用同一条谱系。**

### 7.1 口径一：层内 `[A+B]` 段（OpenMP 主导）—— 4 线程最快

同机、`lay0` 的 `[A+B]` 段（矩阵乘），基准 225 s：

| 配置 | `[A+B]` | 结论 |
|---|---|---|
| 4 线程 / OpenMP 默认 | 225 s | **最快** |
| 8 线程 / OpenMP 默认 | 255 s | **+13%（更慢）** |
| 14 线程 / OpenMP 默认 | >384 s | **>+70%（更慢）** |
| 4 线程 / `OMP_NUM_THREADS=4` | 263 s | **+16%（更慢）** |

原因：`[A+B]` 的矩阵乘由 **OpenMP 段**主导（默认已占满 16 逻辑核），而 `vllm_tp` 的 worker 是
**spin-wait** 空转，两者叠加造成超订抢核；同时该阶段受**内存带宽**约束（中间量 ≈112 MB），
堆线程无法增加带宽。

### 7.2 口径二：整跳（单层 `boot` 全程）—— 8 线程快约 20%

同机、同二进制（`t23boot_gh.exe`）、同为 8 个密文：

| 跳 | 配置 | 墙钟 | `coeff_to_slot` | `slot_to_coeff` | profile total |
|---|---|---|---|---|---|
| 整跳 | `T23_NT=4` | **4507 s** | 1749.86 s | 1758.16 s | 4429.71 s |
| 整跳 | `T23_NT=8` | **3600 s** | 1264.22 s | 1270.38 s | 3516.84 s |
| 差值 | — | **−20.1%** | −27.8% | −27.7% | −20.6% |

原因：`boot` 的 ≈79% 时间在 `coeff_to_slot` / `slot_to_coeff`，这两段由**自研线程池**驱动、
吃线程红利，远超 `[A+B]` 那点超订损失。

单密文冒烟（`T23_ONE=1`）同趋势复现：`NT=4` 567.05 s → `NT=8` 440.64 s（−22.3%）；
独立一轮 551.83 s → 439.64 s（−20.3%）。

### 7.3 ⚠️ 关键限制：线程数是**结果变量**，不是单纯的性能旋钮

- `NT=4` 与 `NT=8` 的输出密文 **8/8 位级不同**（字节差异 ≈98.9%）；单密文最小场景同样复现 ≈98.87%。
- 根因：`src/core/vllm_ckks.c` 的 RNG 状态是 `_Thread_local`，`ckks_gk_gen()` 在并行 worker 内
  **按 item** 调用，掩码取值因此依赖"该 item 落到哪个线程、该线程此前已消费多少随机数"。
- 即 `output = f(input, T23_NT)`，而不是 `f(input)`。
- 因此**同一条谱系必须固定线程数**：本仓库的接力归档统一用 `T23_NT=4`。
- 但两者**数值等价**：同一把 `sk` 解密后逐项 `max|err|` 完全相同（`max_err=1.2706e-03`，8/8 PASS）；
  把 `NT=8` 的 `u0r112` 喂给 `lay1` 同样 `RESULT=PASS`、误差统计逐位一致。

### 7.4 一次未成功的尝试（如实记录）

`VLLM_TP_SPIN=1` + `NT=8` 整跳：wall=3719 s / total=3634.02 s，比 `NT=8` 默认（3600 s）更慢。
该轮与机器漂移混叠，**不能据此判断 SPIN 有益或有害**。

## 8. 已知瑕疵（诚实说明）

两个文件的**中文注释存在历史编码残留**：出现 `鈶?` 这类乱码，且部分行被合并（如 `锛? *`）。
- **不影响编译，也不影响数值** —— 代码本体已通过逐层位级复现验证；
- 该损坏**不可逆**（部分字节已丢失），清理时只能按语义重写注释，不能靠转码还原。

**本目录副本与产出结果的源码逐字节一致**（SHA256 前 16 位）：

| 文件 | SHA256 |
|---|---|
| `t23_m3p.c` | `F3BF25FC7C86EE69` |
| `t23_chain.c` | `9A078E03D6C1C1F0` |

## 9. 相关

- **运行 / 打包 / 验证脚本**：[`../relay/`](../relay/)（跑层、打包交接、验包、准确度校验）
- **逐工具用法与判据口径**：[`../relay/README.md`](../relay/README.md)
- **归档产物目录**：`results/L0-4/`（由 `../relay/collect_results.ps1` 生成）
