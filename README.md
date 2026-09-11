# Kestrel（红隼）— ARM(aarch64) 边缘 LLM 推理引擎（RK3588 开发基准）

**纯 C11、零第三方运行时依赖的 ARM（aarch64）CPU LLM 推理引擎。**

> **版本：v1.0（测试版 / test release）**。相对上一个公开版本（v0）的主要变化见文末
> 「v1.0 变更摘要」。文中性能/体积数字为**板端实测口径**（测点版本已就近标注）；
> 换板或换版本复现请以 `./build_rk3588.sh --run-tests` 的当前自检与产物为准。

> 平台口径：引擎本质是**面向 ARM 架构 CPU 的推理**（ARMv8.2-A + NEON dotprod/fp16，
> 不依赖 GPU/NPU 与特定开发板）；**RK3588** 是当前开发、优化与基准测试平台，
> 并非唯一可运行设备——同类 aarch64 Linux 设备可尝试编译运行（跨设备验证状态见「构建」节平台约束）。

一块 ARM 开发板（示例：RK3588 / Orange Pi 5 Plus）+ 一个约 **0.8 MB** 的单文件可执行程序 = 原生 LLM 推理服务：
**2 秒冷启动**，原生跑 Qwen3-VL 2B/8B 纯文本与图片/视频多模态，OpenAI 兼容 HTTP API 即刻可用
（以下数据均为板端实测，方法学见 [docs/RK3588_性能基准报告.md](docs/RK3588_性能基准报告.md)）。

**核心亮点**

| 亮点 | 说明 |
|---|---|
| 单文件约 0.8 MB | 28 个 C 文件 → 单产物 **818,872 B**（Release 实测）；全静态约 1.5 MB、零 .so，拷贝即跑 |
| 冷启动 2.0 s | spawn→HTTP ready **2.01 s**（对比 llama.cpp 5.0 s）；VQF v2 预量化 + mmap 直挂，加载 ≈1.2 s |
| 零第三方依赖 | 手写 NEON 量化 GEMM/GEMV + 自研线程池（替代 OpenMP）+ 自研国密 SM3/SM4/SM2；标准构建仅需 gcc + libm |
| 长上下文更快 | 8K decode **3.0×**（136.7 vs 411.6 ms/词）；磁盘 KV 跨进程恢复 **13.1×**（重启不丢会话） |
| Qwen3-VL 多模态 | 2B/8B 纯文本 + 图片/视频帧；纯 CPU 即可运行，NPU 直驱为可选透明加速 |
| 可验证推理 | SM2 供应链签名护权重 + 逐请求 attestation 密码学凭证、离线验签（医疗/法务级合规场景） |

> 品牌命名：**Kestrel（红隼）** 为品牌/对外名（取义：最小猛禽、俯冲精确——边缘小模型 + 可验证推理）；
> 工程名与可执行文件为 `vllm_kestrel`，下文命令、日志与代码中的 `vllm_kestrel` / `vllm` 均指本引擎。

引擎从零自研、面向边缘设备（RK3588 / 4×Cortex-A76 + 4×Cortex-A55），
**针对 Qwen3-VL 系列（2B/8B）开发并实测**（支持范围诚实声明见「模型与复现」一节）；
配套长上下文工程（sparse-attention / 前缀 KV 复用 / 磁盘 KV / 推测解码 / 连续批处理）与
**同态加密推理研究内核**（RNS-CKKS，`src/core/vllm_ckks.c` 等，与明文引擎共享工程基础设施，见 [docs](docs/)）。

---

## 体量与依赖（小而全）

| 项 | 数值 / 口径 |
|---|---|
| 可执行文件 | `vllm_kestrel` ≈ **0.8 MB**（RK3588 Release, `-O2 -s`，板端实测 818,872 B）；`-DVLLM_STATIC=ON` 全静态 ≈ 1.5 MB，`ldd` 零 .so 依赖 |
| 源码 | **28 个 C 文件**（main + core 11 + common 2 + serve 6 + model 6 + npu 2），C11，单工程单产物 |
| 运行时依赖 | **无第三方运行时**——标准构建仅需 gcc + libm（`-fopenmp` 仅 NPU pack 并行区使用 libgomp，系 gcc 自带；全静态构建一并内联，零 .so） |
| 代码内第三方 | 仅 `stb_image.h`（MIT, Sean Barrett）与 llama.cpp 派生 4x4 asm 内核（MIT, The ggml authors），见 [LICENSE](LICENSE) 第三节 |
| 自研件 | NEON 量化 GEMM/GEMV、线程池 `vllm_tp`（替代 OpenMP）、国密 SM3/SM4/SM2、VQF v2 mmap 格式、NPU 直驱 `/dev/rknpu` |
| 部署 | 单文件 + 可选 `vocab.bin`，拷贝即运行；VQF mmap 冷启动 **2.0 s** |

同机对照（板端实测口径）：冷启动 2.0 s vs llama.cpp 5.0 s、峰值 RSS 2.47 GB vs 3.03 GB、
长上下文 decode 与 KV 恢复优势，见 [RK3588_性能基准报告.md](docs/RK3588_性能基准报告.md)。
> 诚实边界：二进制体积仅列本引擎自身（llama.cpp 动态/静态构建口径不同，未做同口径对比，
> 不作跨框架体积比较）。

### 零第三方依赖（自证清单）

「零第三方依赖」不是说"没写什么"，而是说**下面这些常见组件栈都没有引入**：

| 常见依赖项 | 本项目实际情况 |
|---|---|
| 推理框架运行时（Python / PyTorch / vLLM 栈） | 无——单一 C11 可执行文件即完整 HTTP 服务 |
| GPU 计算栈（CUDA / ROCm） | 无——纯 CPU + NEON 量化内核；NPU 仅走系统内核公共 UAPI 直驱 |
| 第三方推理/矩阵库（ggml、OpenBLAS、oneDNN…） | 无——GEMM/GEMV 手写（llama.cpp 派生 4x4 asm 为 MIT 提取件，文件头署名，见 LICENSE 第三节） |
| 密码学库（OpenSSL / GmSSL / MbedTLS…） | 无——SM3 / SM4-CTR / HMAC-SM3 / SM2 全自研并经国密标准 KAT 验证 |
| 图像/视频解码库（OpenCV / FFmpeg…） | 无——图片解码用单头 `stb_image.h`（MIT）；H.264 模块独立且默认不启用 |
| Web/HTTP 框架与 JSON 库 | 无——自研 select 轮询 HTTP + SSE、自研最小 JSON 解析 |
| OpenMP 运行时 | 引擎核心并行用自研线程池 `vllm_tp`；`-fopenmp` 仅 NPU direct 后端 pack 并行（libgomp 为 gcc 自带） |

依赖上限一句话：**动态构建只碰系统工具链标准件（glibc / libgomp）；全静态构建（`-DVLLM_STATIC=ON`）产物 `ldd` 报 not a dynamic executable，可拷到任意 aarch64 Linux 直接运行**——没有任何第三方项目运行时随引擎分发。

---

## 最近实测数据（RK3588，2026-09-05）

平台：Orange Pi 5 Plus（RK3588，8 核，15GB RAM，eMMC，无 GPU/NPU 参与）。
模型：Qwen3-VL-2B-Instruct（vllm_kestrel VQF q4 全优化档 vs llama.cpp GGUF Q4_0）。
完整方法学、口径与原始数据见 [docs/RK3588_性能基准报告.md](docs/RK3588_性能基准报告.md)。

### 冷启动
| 引擎 | spawn→HTTP ready | 峰值 RSS (VmHWM) |
|---|---|---|
| vllm_kestrel（VQF mmap） | **2.01 s** | ~2465 MB |
| llama.cpp (GGUF mmap) | 5.02 s | ~3027 MB |

### 长上下文（内容 token 1K/2K/4K/8K，生成 64 token）
| 档 | vllm decode (TPOT) | llama decode (TPOT) | decode 比 |
|---|---|---|---|
| 1K | 73.4 ms | 102 ms | vllm 1.4× |
| 2K | 84.6 ms | 126 ms | vllm 1.5× |
| 4K | 102 ms | 218 ms | vllm 2.1× |
| **8K** | **136.7 ms** | **411.6 ms** | **vllm 3.0×** |

- prefill：1K–4K llama 快 1.4–1.6×（ggml NEON Q4 更成熟），**8K 拉平**
  （vllm 40.0 vs llama 38.3 tok/s）；vllm 预填吞吐 4K→8K 不降，llama 每档下降 20%+
  （sparse-attn 的 O(n·k) 效果）。
- decode 随上下文放大是架构级差异：llama f16-KV 每词全扫 8K、2K→8K 劣化 3.3×；
  vllm q8-KV + 稀疏 decode 仅 1.6×。

### KV 缓存恢复（同进程前缀复用 vs 跨进程磁盘恢复）
- 同进程第二轮（29 token 增量 prefill，8K 上下文）：**vllm 2.0s vs llama 6.63s（3.3×）**。
- 跨进程磁盘 KV 恢复（`--disk-kv`，8K 会话 F32 快照 1.87GB，重启后 15.4s 恢复 vs
  全量 prefill 202s）：**vllm 13.1×**；llama.cpp 无等价物（重启即失）。

> 诚实口径：本报告为板内单引擎串行测量、权重同源（Qwen3-VL-2B safetensors）但量化
> 格与算子不同（非位级同一权重），数值为各自引擎原始字段对齐后的并列展示。
> 复现方法与数据文件见报告附录。

---

## 目录结构

```
├── CMakeLists.txt             # 构建（Release / 静态 / NPU 直驱默认）
├── build_rk3588.sh            # RK3588 构建 + 自检入口
├── cmake/toolchain-aarch64-rk3588.cmake   # x86 主机交叉编译工具链
├── include/  src/             # C11 源码（common/core/media/model/npu/serve）
│   ├── core/                  # 推理内核（NTT/FHE/CKKS/tp/matmul/attention/l3…）
│   ├── model/                 # 权重加载（纯 VQF v2 mmap）、视觉、分词
│   ├── serve/                 # HTTP/管理页/批处理/可验证推理(attest)
│   └── media/                 # H.264/MP4 解码（独立模块，默认构建不启用）
├── tools/                     # 自研工具
│   ├── gen_embedded_web.py    # HTML → 内嵌字节数组生成器（改页面后须重跑）
│   ├── build_vocab_bin.py     # tokenizer.json → vocab.bin（字节解码修复版）
│   ├── extract_llama_asm.py   # 从 llama.cpp 提取 4x4 asm GEMM（MIT，见文件头）
│   ├── verify_attest.py       # 可验证推理凭证离线验签（零依赖）
│   ├── vllm_vqf_sign.c        # VQF SM2 供应链签名 / 密钥管理工具
│   ├── vllm_mgr.py            # 引擎进程守护（start/stop/restart/状态页）
│   ├── build_x64.ps1          # x86_64(MinGW) 原生构建脚本（非基准，仅一致性自检）
│   └── check_x64.ps1          # x86 构建 + 自检一条命令（退出码 0/1）
├── vqf_convert/               # 独立权重转换工具（safetensors/GGUF → VQF v2）
└── docs/                      # 技术文档 / 基准报告 / 安全方案（中文）
```

## 构建（RK3588 / aarch64 Linux）

```bash
# 板上原生构建（需 gcc + cmake）
./build_rk3588.sh                    # 构建 + 复制到 ./vllm_kestrel
./build_rk3588.sh --run-tests        # 构建 + 运行 PASS/FAIL 自检
./build_rk3588.sh --static           # 全静态（零 .so 依赖）

# 或直接在板上：
cmake -B build-rk3588 && cmake --build build-rk3588 -j8

# x86_64 Linux 主机交叉编译（需要 gcc-aarch64-linux-gnu）
./build_rk3588.sh --cross
```

### x86_64 原生构建（Windows / MinGW，用于功能与一致性自检）

引擎的**一级目标平台是 aarch64**；x86-64 分支（`vllm_platform.h`）仅用于**功能自检与
位级一致性对照**，**不作为性能基准**——x86 上跑的绝对吞吐/加速比不能外推到板端。

```powershell
# Windows / MinGW-w64（gcc 需在 PATH，或用 -Gcc 显式指定）
powershell -ExecutionPolicy Bypass -File tools\check_x64.ps1
#   → 编译 + 跑 --test-l3 / --test-sparse 自检，退出码 0/1
powershell -ExecutionPolicy Bypass -File tools\build_x64.ps1 -Gcc D:\tools\mingw64\bin\gcc.exe
#   → 仅构建，产物 build-x64\vllm_kestrel_x64.exe
```

环境变量口径：`VLLM_GCC`（gcc 路径）、`VLLM_X64_OUTDIR`（输出目录）可替代命令行参数。

> 平台约束与说明：引擎本质是 **ARM（aarch64，ARMv8.2-A + dotprod/fp16）CPU 推理引擎**，
> **RK3588（4×A76 + 4×A55）是开发与基准测试平台**，并非唯一可运行设备。同类 aarch64
> Linux 设备可尝试编译运行，但设备画像（如 A76 集群线程绑定、核心数）与性能档按 RK3588
> 验证——**换板运行请先跑 `--test-l3` / `--bench-mixed` 自检**并以自检结果为准。运行建议
> `export OMP_NUM_THREADS=8`（RK3588：4×A76 + 4×A55）。非 aarch64 / 非 x86-64 架构会在
> `vllm_platform.h` 编译期报错退出。

## 运行

```bash
# 1) 模型目录需含：config.json + model.vqf（单文件 VQF v2，mmap 直挂）
#    + 可选 vocab.bin（由 tools/build_vocab_bin.py 从 tokenizer.json 生成；
#      缺失时引擎回落到内嵌 vocab，功能可用但体积/词表以模型自带为准）：
#    python tools/build_vocab_bin.py <tokenizer.json> <vocab.bin> <vocab.bin>

# 2) 启动 OpenAI 兼容 HTTP 服务（默认端口 8080）
./vllm_kestrel --serve --port 8080 --model <model-dir> --auto-load --wmode q4

#    长上下文优化档（对应上文 8K 基准，全部可选；prefix-kv 前缀复用默认开启）：
#    --l3-evict 需配合环境变量 VLLM_L3_PREFIX_REUSE=1，二者共存才有 P3 收益
#    （否则 L3 驱逐会静默打掉 prefix-kv，多轮追问将全量重算 prefill）。
VLLM_L3_PREFIX_REUSE=1 ./vllm_kestrel --serve --port 8080 --model <model-dir> --auto-load \
    --wmode q4 --sparse-attn --sparse-k 32 --spec --spec-k 4 \
    --l3-evict --l3-ratio 0.75 --l3-min-seq 128 \
    --disk-kv <kv-dir> --threads 8

# 3) 对话（OpenAI 兼容）
curl http://<board>:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-vl","messages":[{"role":"user","content":"你好"}],"max_tokens":64}'

# 管理页 / 聊天页：http://<board>:8080/admin/  :8080/chat/
# （管理页含「有效优化组合（实测台账）」：8 个组合可一键套用，并标注每项优化
#   针对 RAM / x86 是否有效；保存配置后需重启引擎生效）
```

权重格式转换：本引擎**只加载 VQF v2**，不再内置 safetensors/GGUF 转换路径；
转换由随版发布的独立工具 **`vqf_convert/`** 完成（safetensors / GGUF → 单文件 mmap 的 VQF）。

```bash
# 构建转换工具（主机/板端均可；Windows 用 build.bat）
cd vqf_convert && ./build.sh
# 转换（明文）
./vqf_conv --model <safetensors-dir> --convert-vqf <out.vqf> --wmode q4
# 加密（VQF-Enc：SM4-CTR + HMAC-SM3）
VLLM_VQF_KEY='<pass>' ./vqf_conv --model <safetensors-dir> --convert-vqf <out.vqf> --wmode q4
# 内嵌 SM2 供应链签名（可与加密叠加）
VLLM_VQF_SIGN_PRIV='<64hex>' ./vqf_conv --model <safetensors-dir> --convert-vqf <out.vqf> --wmode q4
```

NPU 加速（可选）：默认后端为**零第三方依赖直驱**（自研写 stock rknpu 内核驱动）。
首次使用先跑板端校准：`./vllm_kestrel --npu --npu-selftest --perf-only`
（寄存器命令表未校准通过前提交路径保持禁用，自动回退 CPU）。

权重保护与可验证推理：**产出侧**由随版发布的 `vqf_convert/` 完成——设 `VLLM_VQF_KEY`
输出 VQF-Enc 加密文件（SM4-CTR + HMAC-SM3），设 `VLLM_VQF_SIGN_PRIV`（64 hex）内嵌 SM2
供应链签名，二者可叠加；**加载侧**置 `VLLM_VQF_KEY` / `VLLM_VQF_SIGN_PUB` 解密验签。
置 `VLLM_ATTEST=1`（详见 [docs/权重保护与可验证推理方案.md](docs/权重保护与可验证推理方案.md)）
即逐响应出证（attestation schema=3，含请求原文绑定），用 `tools/verify_attest.py` 离线验签。
> 边界：加密 / 签名文件需**全层驻留**（`VLLM_VQF_STREAM` 会对加密 VQF 显式拒绝）；
> 转换的 `--stream` 路径暂不支持加密 / 签名（去掉 `--stream` 走全量路径即可）。

## 模型与复现

> **模型支持范围（诚实声明）**：本引擎针对并实测验证的是**两类 Qwen3 架构**：
>
> - **Qwen3-VL 系列（2B / 8B）**——纯文本与图片/视频多模态；tokenizer、mrope、
>   DeepStack 视觉塔等均为该架构特化实现。
> - **Qwen3-MoE 系列（如 Qwen3-30B-A3B，文本）**——路由 + 逐专家 FFN 已接入
>   （`--moe-batch` / `VLLM_ACTQ` 等加速，见管理页「MoE 模型服务」组合）。
>
> **其他架构（Llama、旧版 Qwen / Qwen2 纯文本等）未经适配与验证**：转换可能报错或
> 输出不可用，请勿据此推定为通用推理引擎。文中性能与安全数据均基于 Qwen3-VL-2B
> （RK3588 板端）与 Qwen3-VL-8B（x86 基准机）实测，MoE 路线为功能打通与机制验证口径。

- 模型权重不随仓库分发。Qwen 系列权重遵循其原始开源许可（Qwen 社区许可），
  下载后可用 `vqf_convert/` 转换为 VQF 后加载（见上文「权重格式转换」）。
- 基准数据复现方法、语料与驱动位置见
  [docs/RK3588_性能基准报告.md](docs/RK3588_性能基准报告.md) 附录。

## 文档（docs/，中文）

| 文档 | 内容 |
|---|---|
| [docs/技术文档.md](docs/技术文档.md) | 架构、模块、权重格式 VQF、内核、服务层、多模态、上下文管理、位级确定性、NPU、性能、调试、版本演进 |
| [docs/RK3588_性能基准报告.md](docs/RK3588_性能基准报告.md) | vllm_kestrel vs llama.cpp 冷启动 / 长上下文 / KV 恢复全矩阵 |
| [docs/优化配置与边界说明.md](docs/优化配置与边界说明.md) | 各优化档机制、收益与诚实边界（含有效组合与 x86 复核口径） |
| [docs/KV缓存v2-惰性分配与分层驻留方案.md](docs/KV缓存v2-惰性分配与分层驻留方案.md) | KV 惰性分配、L3 分层驻留与 P3 前缀复用共存（L3 驱逐 + 前缀复用） |
| [docs/权重保护与可验证推理方案.md](docs/权重保护与可验证推理方案.md) | 三道安全防线：VQF 存储态加密、SM2 供应链签名、推理出证（attestation schema=3，含请求原文绑定），含相互关系、端到端用法与统一安全边界 |

## v1.0 变更摘要（相对 v0）

**工程与形态**

- **纯 VQF 运行时**：引擎只加载单文件 VQF v2（mmap 直挂），删除内置的 GGUF /
  safetensors 加载与引擎内转换路径；转换职责移交随版发布的独立工具 **`vqf_convert/`**。
- **源码瘦身**：随纯 VQF 运行时移除 `vllm_gguf.c/.h` 与 `convert.html` 等遗留件；
  单产物仍约 0.8 MB；`CMakeLists.txt` 版本号提升至 `VERSION 1.0.0`。
- **x86_64 分支随版**：`vllm_platform.h` 提供 x86-64（MinGW/MSVC）移植层；新增
  `tools/build_x64.ps1` / `tools/check_x64.ps1`（已参数化，`VLLM_GCC` /
  `VLLM_X64_OUTDIR` 可覆盖）。**x86 仅供功能自检与位级一致性对照，不作性能基准。**

**性能与内存**

- **P3：L3 驱逐 × 前缀复用共存**（`--l3-evict` + `VLLM_L3_PREFIX_REUSE=1`）——多轮
  追问轮 prefill 实测 **−93%~−97%**，为 v1.0 收益最大的单项优化。
- **L3 紧凑布局**：按驱逐顺序连续分配 `disk_off`（`wcursor`），文件 / RAM mirror
  尺寸跟踪真实载荷而非全 KV 窗。
- **P1/P2 内存分页**：mirror 跨轮复用 + `MADV_DONTNEED`；arena 化 + `imp_sum`
  层内共享。
- **管理页实测台账**：`/admin/` 新增「有效优化组合」表（8 组，可一键套用），并对
  每项优化标注**针对 RAM / x86 是否有效**；补齐 P3 开关与 `--l3-evict` 未开 P3 门
  时的联动告警。

**安全与正确性**

- **attestation schema 3**：新增**请求原文绑定**（`body_sha = SM3(客户端原始请求体)`），
  只改 `top_k` / `thinking` 也必须改摘要。
- **转换工具支持加密 / 签名产出**：`vqf_convert/` 设 `VLLM_VQF_KEY` 即输出 VQF-Enc 加密
  文件（SM4-CTR + HMAC-SM3），设 `VLLM_VQF_SIGN_PRIV` 即内嵌 SM2 供应链签名，二者可叠加；
  引擎侧加载日志实测 `decrypted (SM4-CTR, HMAC-SM3 ok)` + `SM2 verify ok`，错口令 / 篡改
  数据字节均被拒绝。
- **VQF 离线补签口径修复**：离线签名工具先置 `VQF_FLAG_SIGNED` 再算摘要（与写侧
  口径一致），修复补签后永远 digest mismatch 的问题。
- **Debug 构建修复**：`CMAKE_C_FLAGS_DEBUG` 补 `-march`，避免 NEON dotprod 内联
  在 Debug 档编译失败。
- **注释编码修复**：修正历史遗留的若干源码注释乱码（`vqf_convert/src/conv_main.c`、
  `src/serve/vllm_server.c`、`src/serve/vllm_batch.c`）。


## 许可与合规

- **许可：source-available 双许可（非 OSI 开源许可）**——学习 / 学术研究 / 论文复现
  完全免费；任何商业使用与企业内部生产部署须先取得商业授权。
  **请勿以 MIT/Apache 等标准开源协议理解本仓库**；完整条款见 [LICENSE](LICENSE)。
- 第三方组件按各自许可保留：`stb_image.h`（MIT, Sean Barrett）与源自 llama.cpp
  的 4x4 asm GEMM 提取文件及其派生内核（MIT, The ggml authors）——版权与许可
  文本见对应文件头，详见 [LICENSE](LICENSE) 第三节。
- 本引擎的 NPU 直驱后端仅与操作系统内核驱动（stock rknpu）的公共 UAPI 交互，
  不包含任何闭源库或第三方头文件。

## 联系

商业授权 / 研究合作 / 复现数据：**398152090@qq.com**
（也可通过 [Issues](https://gitee.com/pei-xiaoguang/kestrel-llm/issues) 或 Gitee 站内私信联系作者；
商业许可协议与双许可条款见 [LICENSE](LICENSE)）

For commercial licensing / research collaborations / data requests:
**398152090@qq.com**, or open an issue at https://gitee.com/pei-xiaoguang/kestrel-llm/issues.
