# 许可说明 · vllm_kestrel / Licensing

Copyright (C) 2026 裴晓光 (Pei Xiaoguang) and contributors
项目主页 / Project home: https://gitee.com/pei-xiaoguang/kestrel-llm

> 许可证**正文**见 [LICENSE](LICENSE)（Apache License 2.0 完整条款）。
> 本文件只承载**版权声明、历史版本说明、第三方组件清单与贡献规则**。

---

## 一、本项目的许可：Apache License 2.0

**SPDX-License-Identifier: Apache-2.0**

本项目以 **Apache License, Version 2.0** 发布。你可以自由地**商用（含闭源产品与闭源 SaaS）**、
修改、分发、再许可、销售——**无需另行取得任何授权、也无需开放你的源码**。

对应的义务只有三条（Apache-2.0 第 4 条）：

1. 随附一份许可证副本；
2. 你**修改过的文件**要保留"已修改"的显著标注；
3. 保留源码形态中的版权、专利、商标与归属声明。

**第 3 条含明示专利授权**：每位贡献者就其**必要专利**授予你永久的、全球的、免费的、不可撤销的许可。
对价是"专利报复条款"——**你若主动发起专利诉讼**指控本项目（或其贡献）构成专利侵权，
该专利授权自起诉之日起终止。

**第 6 条不含商标授权**：不得使用本项目的名称、商标或产品名（"Kestrel" / "vllm_kestrel"）
暗示背书，正常描述来源除外。

本程序按**"现状"**分发，不提供任何担保（第 7 条），作者不承担任何赔偿责任（第 8 条）。

---

## 二、与此前 "AGPL-3.0-or-later / 商业许可" 双许可的关系（**重要，请勿误读**）

本项目**此前**采用**双许可**：AGPL-3.0-or-later，或商业许可（二选一）。
**自本文件随附的版本起，改为 Apache-2.0 单许可。**

| 事项 | 结论 |
|---|---|
| 历史版本（此前的 commit / 已上传的 Release 资产 / 已分发的副本） | **仍为 AGPL-3.0-or-later**。任何已获得这些副本的人**永久保有**该授权 |
| 本次变更能否撤回那些授权 | **不能**，也不打算撤回。旧副本的权利不受影响 |
| 本次变更是否追溯 | **不追溯**。需要 Apache-2.0 条款，请使用本变更**之后**的版本 |
| 商业许可 | **不再提供、也不再需要**——Apache-2.0 本身已允许闭源商用。（此前已签署的商业许可协议按该协议自身条款继续有效） |
| 如何确认某个具体 commit 适用哪个许可 | 以**该 commit 内的** `LICENSE` 与 `LICENSING.md` 为准 |

> 一句话：**新版本宽松、旧版本照旧**。别把"现在是 Apache-2.0"读成"以前下载的那份也是 Apache-2.0"。

---

## 三、第三方组件（各自遵循其原许可）

本项目的**原创代码**适用 Apache-2.0；随附的第三方组件独立保留其原许可证：

| 组件 | 位置 | 许可 |
|---|---|---|
| `stb_image.h` | `src/model/stb_image.h` | MIT License, Copyright (c) 2017 Sean Barrett |
| llama.cpp 派生 4x4 asm GEMM 内核 | `tools/kernels/llama_gemm_q4_0_4x4_asm.c`，以及 `src/model/vllm_safetensors.c` 中标注的 ggml_gemm / gemv / vec_dot 派生内核 | MIT License, Copyright (c) 2023–2026 The ggml authors |

版权声明与许可文本见各文件头。上述 MIT 组件与 Apache-2.0 **兼容**（MIT 许可允许被纳入采用其他条款的作品），
其原始版权声明按 Apache-2.0 第 4(c) 条原样保留。

---

## 四、其他声明

1. 本许可**不转移**任何商标权；专利权按 Apache-2.0 第 3 条授予。
2. 对本项目的贡献（Issue / PR）按 **Apache-2.0 第 5 条**处理：除非你明确另行书面声明，
   你提交并被纳入的贡献即以 **Apache-2.0** 条款授权。详见 [CONTRIBUTING.md](CONTRIBUTING.md)。
3. **模型权重、数据集与转换产物不在本仓库内分发**；如需使用，请遵循其原始来源的开源许可
   （例如 Qwen 系列遵循 Qwen 社区许可）。
4. 安全漏洞请按 [SECURITY.md](SECURITY.md) 的流程私下报告，**不要**直接开公开 Issue。

---

## 附：源码文件建议附注（SPDX）

建议在源文件头部使用：

```
SPDX-License-Identifier: Apache-2.0
Copyright (C) 2026 裴晓光 and contributors
```

或使用简短声明：

```
This file is part of vllm_kestrel, licensed under the Apache License, Version 2.0.
See LICENSE for details.
```
