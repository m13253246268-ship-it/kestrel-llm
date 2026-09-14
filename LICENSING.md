# vllm_kestrel Dual Licensing / 双许可

Copyright (C) 2026 裴晓光 (Pei Xiaoguang) and contributors
项目主页 / Project home: https://gitee.com/pei-xiaoguang/kestrel-llm

> 本文件承载**双许可声明、第三方组件清单与商用授权条款**；许可证正文见 [LICENSE](LICENSE)（AGPL-3.0 完整条款）。

This project is **dual-licensed**. You may use it under **either** of the two options below,
at your choice. 本项目采用**双许可**模式，你可以**二选一**：

---

## 选项一（开源）：GNU Affero General Public License v3.0 or later

**SPDX-License-Identifier: AGPL-3.0-or-later**

本项目是自由软件：你可以依据自由软件基金会发布的 **GNU Affero 通用公共许可证**
（第 3 版或你选择的任何更新版本）的条款，再分发和/或修改它。

This program is free software: you can redistribute it and/or modify it under the terms of the
**GNU Affero General Public License** as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later version.

本程序按"现状"分发，不提供任何担保。详见许可证正文。

**AGPL-3.0 的完整条款文本（canonical text）请以自由软件基金会发布为准：**
- 官方纯文本：https://www.gnu.org/licenses/agpl-3.0.txt
- 官方说明页：https://www.gnu.org/licenses/agpl-3.0.html

> **重要**：AGPL 是**强 copyleft** 许可。如果你修改了本软件并通过**网络服务**向用户提供，
> 你必须依 AGPL 第 13 条向这些用户提供对应的完整源码。
> **如果你的使用方式无法接受这一条（例如需要闭源集成或闭源商用），请走「选项二」。**

---

## 选项二（商业）：Commercial License

若你不希望（或不能）遵守 AGPL-3.0 的条款——包括但不限于：在**闭源产品**中集成、
以**闭源方式**提供商业服务 / SaaS、转售，或在不承担 AGPL 源码开放义务的前提下
进行以盈利为目的的部署与二次开发——**必须**事先与版权所有者签订商业许可协议并取得书面授权。

If you cannot or do not wish to comply with AGPL-3.0 — including embedding in closed-source
products, offering closed-source commercial services / SaaS, reselling, or profit-oriented
deployment without the AGPL source-disclosure obligation — you **must** obtain a prior written
commercial licence from the copyright holder.

**商业授权联系 / Commercial licensing contact：398152090@qq.com**

---

## 三、第三方组件（各自遵循其原许可）

本项目的**原创代码**适用上述双许可；随附的第三方组件独立保留其原许可证：

| 组件 | 位置 | 许可 |
|---|---|---|
| `stb_image.h` | `src/model/stb_image.h` | MIT License, Copyright (c) 2017 Sean Barrett |
| llama.cpp 派生 4x4 asm GEMM 内核 | `tools/kernels/llama_gemm_q4_0_4x4_asm.c`，以及 `src/model/vllm_safetensors.c` 中标注的 ggml_gemm / gemv / vec_dot 派生内核 | MIT License, Copyright (c) 2023–2026 The ggml authors |

版权声明与许可文本见各文件头。上述 MIT 组件与 AGPL 兼容（MIT 许可允许被纳入 AGPL 作品）。

## 四、其他声明

1. 本许可**不转移**任何商标权、专利权。
2. 对本项目的贡献（Issue / PR）按"提交即同意以本双许可条款发布"处理，详见
   [CONTRIBUTING.md](CONTRIBUTING.md)。
3. **模型权重、数据集与转换产物不在本仓库内分发**；如需使用，请遵循其原始来源的开源许可
   （例如 Qwen 系列遵循 Qwen 社区许可）。
4. 安全漏洞请按 [SECURITY.md](SECURITY.md) 的流程私下报告，**不要**直接开公开 Issue。

---

## 附：源码文件建议附注（SPDX）

建议在源文件头部使用：

```
SPDX-License-Identifier: AGPL-3.0-or-later
Copyright (C) 2026 裴晓光 and contributors
```

或使用简短声明：

```
This file is part of vllm_kestrel, dual-licensed under AGPL-3.0-or-later
or a commercial licence. See LICENSE for details.
```

---

# GNU AFFERO GENERAL PUBLIC LICENSE

Version 3, 19 November 2007

（以下为 AGPL-3.0 完整条款正文；上方为双许可声明与第三方组件说明。）
