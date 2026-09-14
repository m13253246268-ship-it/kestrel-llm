[English](README.en.md) | **简体中文**

# 数据预处理脚本（tools/preproc/）

本目录的脚本用于**自行生成**层链驱动所需的 ~7 GB 数据包（权重 + 明文参考 + 拟合系数），
这样复现者不必依赖我们分发大文件。

> **前提**：先按根 README 的「同态加密（FHE）密文推理链驱动」一节编译出 `t23lay` / `t23boot`，
> 驱动用法与数据目录约定见 [`../drivers/README.md`](../drivers/README.md)。

## 1. 环境要求

| 项 | 要求 |
|---|---|
| Python | 3.10+（实测 3.12.9） |
| 依赖 | `numpy`、`safetensors`（`pip install numpy safetensors`） |
| 模型 | `Modl/Qwen3-VL-2B-Instruct/model.safetensors`（约 4.26 GB） |
| 工作目录 | **必须在仓库根目录执行**——脚本内路径是相对且硬编码的 |

## 2. 脚本清单与产物

| 脚本 | 作用 | 产物 |
|---|---|---|
| **`_export_weights.py`** | **权重量化无关导出**（safetensors → 驱动读取的 `.bin`） | `tail/w0..w27/`（每层 9 个）+ `tail/embed.bin` + `tail/norm.bin` |
| `_full_layers.py` | 28 层因果明文前向参考（L0..L25） | `tail/l{L}_*`、`tail/ln{L}_e1/p1.bin`、`tail/m2c{L}_e/p.bin` |
| `_tail_ref.py` | 末段明文参考（L26/L27 + final norm + lm_head） | `tail/l26_*`、`tail/l27_*`、`tail/xnorm_f.bin`、`tail/logits.bin`、`tail/ln_f.bin`、`tail/m2c_f.bin` |
| `_adap_scale.py` | 逐层标定折叠系数 `F[L]` 并重生成输入参考 | `tail/scale{L}.bin`、`tail/l0_u1.bin` |
| `_sin_fit*.py` / `_cos_fit.py` | SiLU/SwiGLU 激活的多项式拟合 | 拟合系数（配合 `_adap_scale` 使用） |
| `_recip_fit.py` / `_refit_lnq.py` / `_ln_fit*.py` | LNQ 求逆（`1/sqrt`）拟合与重拟合 | `tail/ln*_e1/p1.bin`、`tail/m2c*.bin` |
| `_attn_fit.py` | attention 段拟合 | — |
| `_fold_fit.py` / `_fold_sim.py` / `_cfold_plan.py` | fold/未折叠域的误差仿真与规划 | 统计输出（用于选参数） |
| `_g256_conv.py` / `_g256_chk.py` | 256 分组（`BB=32`/`GG=32`）的转换与校验 | 校验输出 |
| `_silu_err.py` / `_logits_tol.py` | 误差诊断（SiLU 误差、logits 容差） | 诊断输出 |

## 3. 建议执行顺序

```bash
# 0) 从仓库根目录执行
cd <repo-root>

# 1) 权重与顶层矩阵（必需，最先做）
python tools/preproc/_export_weights.py

# 2) 明文参考：全层 + 末段
python tools/preproc/_full_layers.py
python tools/preproc/_tail_ref.py

# 3) 拟合：激活 / LNQ 求逆
python tools/preproc/_sin_fit.py
python tools/preproc/_recip_fit.py

# 4) 逐层标定折叠系数 + 重生成输入参考（会刷新 l0_u1.bin）
python tools/preproc/_adap_scale.py
```

其余 `_*_fit*.py` / `_*_sim.py` / `_g256_*.py` / `_*_err.py` / `_*_tol.py` 是**选参数与诊断**用的，可按需运行。

## 4. 产物格式（驱动读取约定）

- **权重**：原始 `float32`、行主序 `[out_dim, in_dim]`、无 header、无转置。
  例如 `tail/w0/q_proj.bin` = 2048×2048×4 = 16,777,216 字节；`tail/w0/gate_proj.bin` = 6144×2048×4。
- **`tail/embed.bin`**：`[151936, 2048]` float32（词表 embedding）。
- **参考**（`tail/l{L}_*.bin`）：4 token × 2048 float32（`_u1`/`_u2_ref` 等），`logits.bin` 为 4 × 151936 float32。

## 5. `_export_weights.py` 的验证

对同一层导出到临时目录，与已发布数据里的 `tail/w{L}/` 逐字节比对（SHA256）：

```
层 0  : q/k/v/o/gate/up/down + in_ln + post_ln  → 9/9  IDENTICAL
层 26 : 9/9 IDENTICAL
层 27 : 9/9 IDENTICAL
embed.bin  IDENTICAL   (1,244,659,712 B)
norm.bin   IDENTICAL   (8,192 B)
```

即：**本脚本的导出结果与产出已发布前 5 层结果的权重完全一致**。

## 6. 已知缺口（诚实说明）

1. **`.tmp_tok/l0/embed4.bin` 无生成脚本**。它是本次测试所用的 4 个 token 经 embedding 查表后的
   明文输入（4×2048 float32），被 `_full_layers.py` / `_tail_ref.py` / `_adap_scale.py` 依赖。
   目前需由数据包分发，或按你的实际提示词自行生成（需 tokenizer + 位置编码）。
2. **`.tmp_tok/l1w/`、`.tmp_tok/l1r/`** 是早期单层（L1）调试流程的专用权重与参考，中间层链（`phase=6`）
   并不需要；本目录脚本未覆盖，也不影响 0..27 层链的运行。
3. 脚本内的模型路径（`Modl/...`）与输出路径（`.tmp_tok/...`）是**硬编码**的；换机器需改 `SRC`/`D0`/`OUT` 常量。

## 7. 相关

- **驱动用法 / 数据目录约定 / 判据**：[`../drivers/README.md`](../drivers/README.md)
- **运行 / 打包 / 验证脚本**：`arxiv/repo/tools/relay/`
- **前 5 层跑通结果（含 SHA256 清单与验证步骤）**：`arxiv/results/L0-4/`
