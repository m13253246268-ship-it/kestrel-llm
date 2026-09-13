# Kestrel（红隼）· Wiki

**纯 C11、零第三方运行时的 ARM（aarch64）CPU LLM 推理引擎。**

一块 ARM 开发板（示例：RK3588 / Orange Pi 5 Plus）+ 一个约 **0.8 MB** 的单文件可执行程序
= 原生 LLM 推理服务：自研 **VQF v2** 权重单文件 mmap 直挂、按需建页，原生跑 Qwen3-VL
2B/8B 纯文本与图片/视频多模态，OpenAI 兼容 HTTP API 即刻可用。

> 仓库：[kestrel-llm (Gitee)](https://gitee.com/pei-xiaoguang/kestrel-llm)
> 许可：源码可得双许可（学习与学术研究免费，商业与产品化部署需取得商业授权），详见 [LICENSING.md](../LICENSING.md)

---

## 本 Wiki 的版本纪律（先读这一条）

本 Wiki 对应 **v1.0（测试版 / test release）**。v0 与 v1 差别很大，性能数据**按版本分区，严禁混用**：

| 分区 | 测点 | 能说明什么 |
|---|---|---|
| **v1.0** | 2026-09-12 板端实测 | 逐层 vs 全层 × {2B / 8B / 30B-A3B} 的完整矩阵 |
| **v0** | 2026-09-05 | 冷启动 / 8K 长上下文 / KV 恢复 / vs llama.cpp 对照，**在 v0 上测得、v1 未复测** |

v1 已移除引擎内置的 GGUF / safetensors 加载与引擎内转换、引入逐层推理与 KV v2，
**冷启动、常驻内存、长上下文与多轮 prefill 的口径都已变化**。凡出现「（v0 测点）」标注的数字，
一律只作历史参考，**不得用于 v1 的任何结论**。

详见 [术语与数据口径](https://gitee.com/pei-xiaoguang/kestrel-llm/wikis/术语与数据口径)。

---

## 核心优势（一目了然）

| 核心优势 | 一句话指标 |
|---|---|
| **逐层推理**（v1 新增；**推荐档 8B**） | 权重常驻与模型体积解耦：**8B 权重常驻 4.19 GB → 0.48 GB（8.7×）**，serve 峰值 **0.82 GB**；热态 **TTFT 1.95 s / tpot 377 ms**，5 次采样波动仅 **±1.8% / ±0.13%**（详见 [性能与基准](https://gitee.com/pei-xiaoguang/kestrel-llm/wikis/性能与基准) §4） |
| **长文本与多轮** | 组合①（L3 驱逐 × 前缀复用）追问轮 prefill **−80% ~ −96%**；30B 再叠 MoE 组合⑤后 t2/t3 仅 **4.7 s / 3.2 s** |
| **纯自研格式与内核** | 只认自研 VQF v2 单文件；单产物 **≈0.8 MB**，零第三方运行时；手写 NEON 量化 GEMM/GEMV、自研线程池、自研国密 |
| **可验证推理** | SM2 供应链签名护权重 + 逐请求 attestation 凭证（schema 3），浏览器内 / 离线零依赖验签 |
| **能力上限：30B-A3B**（2026-09-12 实测） | 16.4 GiB 权重**装不进** 15.6 GiB 页缓存，可用档需**关 thinking** + 短上下文：**24.5 s 拿到完整一句话答案**、峰值 **0.91 GB**；默认 thinking 开时一次问答约需 3~4 分钟。**这是能力上限档，不是推荐档** |

**为什么推荐 8B**：它的权重 6.15 GiB **小于物理内存**，页缓存装得下，所以热态稳定是有物理
保障的（实测 5 次采样 tpot 波动仅 ±0.13%）——这一点 30B 做不到。同时它的省内存收益已拿满
（4.19 GB → 0.48 GB，8.7×），冷启动一次性成本也只有 73 s（30B 为 195~211 s）。

**逐层推理举例**（`VLLM_VQF_STREAM=1`，`--stream-test` 冷页缓存、纯权重口径）：

| 模型（单文件 VQF） | 全层：权重常驻 | **逐层：权重常驻** | 常驻内存倍率 |
|---|---|---|---|
| 2B Qwen3-VL（4.16 GB） | 1.81 GB | **0.39 GB** | **4.6×** |
| 8B Qwen3-VL（6.60 GB） | 4.18 GB | **0.47 GB** | **8.9×** |
| 30B-A3B MoE（17.66 GB） | 11.73 GB | **0.51 GB** | **23.2×** |

---

## 从这里开始

| 你现在的状态 | 建议入口 |
|---|---|
| 想尽快跑起来 | [快速上手](https://gitee.com/pei-xiaoguang/kestrel-llm/wikis/快速上手) —— 从裸板到 HTTP 就绪 4 步 |
| 手边没板子，只有一台 x86 开发机 | [构建与复现](https://gitee.com/pei-xiaoguang/kestrel-llm/wikis/构建与复现) —— x86 分支只作功能自检与位级一致性对照 |
| 想知道它为什么省内存 | [逐层推理与 KV 缓存](https://gitee.com/pei-xiaoguang/kestrel-llm/wikis/逐层推理与KV缓存) |
| 要核对/引用性能数字 | [性能与基准](https://gitee.com/pei-xiaoguang/kestrel-llm/wikis/性能与基准) —— 先读「测量纪律与口径」 |
| 要调优 | [优化配置与边界](https://gitee.com/pei-xiaoguang/kestrel-llm/wikis/优化配置与边界) —— 每个开关的机制、收益与代价 |
| 关心权重与推理可信 | [权重保护与可验证推理](https://gitee.com/pei-xiaoguang/kestrel-llm/wikis/权重保护与可验证推理) |
| 想读代码结构 | [架构总览](https://gitee.com/pei-xiaoguang/kestrel-llm/wikis/架构总览) |

---

## 平台口径（避免误读）

引擎本质是**面向 ARM 架构 CPU 的推理**（ARMv8.2-A + NEON dotprod/fp16，不依赖 GPU/NPU
与特定开发板）；**RK3588** 是当前开发、优化与基准测试平台，并非唯一可运行设备——
同类 aarch64 Linux 设备可尝试编译运行，但请先跑自检并以自检结果为准。

**模型支持范围（诚实声明）**：仅针对并实测验证 **Qwen3-VL 系列（2B / 8B）** 与
**Qwen3-MoE 系列（如 Qwen3-30B-A3B）**；Llama、旧版 Qwen / Qwen2 等**未经适配与验证**，
请勿据此推定为通用推理引擎。

---

## 原始文档

本 Wiki 由仓库 `docs/` 下的文档提炼而成，细节以原文为准：

- [docs/技术文档.md](../docs/技术文档.md)
- [docs/RK3588_性能基准报告.md](../docs/RK3588_性能基准报告.md)
- [docs/优化配置与边界说明.md](../docs/优化配置与边界说明.md)
- [docs/KV缓存v2-惰性分配与分层驻留方案.md](../docs/KV缓存v2-惰性分配与分层驻留方案.md)
- [docs/权重保护与可验证推理方案.md](../docs/权重保护与可验证推理方案.md)
- [README.md](../README.md) ｜ [README.en.md](../README.en.md)

---

## 相关页面

- [快速上手](https://gitee.com/pei-xiaoguang/kestrel-llm/wikis/快速上手)
- [构建与复现](https://gitee.com/pei-xiaoguang/kestrel-llm/wikis/构建与复现)
- [架构总览](https://gitee.com/pei-xiaoguang/kestrel-llm/wikis/架构总览)
- [逐层推理与 KV 缓存](https://gitee.com/pei-xiaoguang/kestrel-llm/wikis/逐层推理与KV缓存)
- [性能与基准](https://gitee.com/pei-xiaoguang/kestrel-llm/wikis/性能与基准)
- [优化配置与边界](https://gitee.com/pei-xiaoguang/kestrel-llm/wikis/优化配置与边界)
- [权重保护与可验证推理](https://gitee.com/pei-xiaoguang/kestrel-llm/wikis/权重保护与可验证推理)
- [常见问题](https://gitee.com/pei-xiaoguang/kestrel-llm/wikis/常见问题)
- [术语与数据口径](https://gitee.com/pei-xiaoguang/kestrel-llm/wikis/术语与数据口径)
