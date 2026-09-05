# Kestrel（红隼）— RK3588 边缘 LLM 推理引擎

**纯 C11、零第三方运行时依赖的 RK3588 (aarch64) 边缘 LLM 推理引擎。**

> **Kestrel（红隼）** 为本引擎的品牌 / 对外名（取义：最小猛禽、俯冲精确——边缘小模型 + 可验证推理）。
> 工程名与可执行文件为 `vllm_kestrel`：下文命令、日志与代码中的 `vllm_kestrel` / `vllm` 均指本引擎。

vllm_kestrel 是一个从零自研、面向边缘设备（RK3588 / 4×Cortex-A76 + 4×Cortex-A55）的
大语言模型推理引擎：单文件 `vllm_kestrel` 即可服务 OpenAI 兼容 HTTP API，支持
Qwen3-VL 系列（2B/8B）纯文本与多模态（图片/视频帧）输入。核心亮点：

- **纯 CPU 推理**：手写 NEON 量化 GEMM/GEMV（8x8 / 4x4 / SDOT），不依赖 GPU/NPU
  即可运行；NPU 直驱（自研 `/dev/rknpu` 驱动）为可选透明加速。
- **零第三方依赖**：自研 SM3/SM4/SM2 国密原语、自研线程池 `vllm_tp`（替代 OpenMP）、
  单文件 mmap 权重格式 VQF；标准构建仅需 gcc + libm。
- **长上下文工程优化**：sparse-attention、prefix-KV 前缀复用、磁盘 KV 持久化
  （跨进程恢复）、推测解码、连续批处理（详见 [docs](docs/)）。
- **权重保护与可验证推理**：VQF v2 存储态加密（SM4-CTR + HMAC-SM3）+ SM2 供应链签名
  保护权重文件；逐请求 SM3 转录摘要 + SM2 设备签名出证（attestation），
  `tools/verify_attest.py` 离线验签（统一方案见
  [docs/权重保护与可验证推理方案.md](docs/权重保护与可验证推理方案.md)）。

> 仓库亦包含自研的**同态加密推理**核心（`src/core/vllm_ckks.c` / `vllm_fhe.c` /
> `vllm_ntt.c` 等，RNS-CKKS 全同态加密 2B 级模型推理的独立研究实现），
> 与明文引擎共享同一工程基础设施。

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
│   ├── core/                  # 推理内核（NTT/FHE/CKKS/tp/matmul/attention…）
│   ├── model/                 # 权重加载（VQF/GGUF/safetensors）、视觉、分词
│   ├── serve/                 # HTTP/管理页/批处理/可验证推理(attest)
│   └── media/                 # H.264/MP4 解码（独立模块，默认构建不启用）
├── tools/                     # 自研工具
│   ├── gen_embedded_web.py    # HTML → 内嵌字节数组生成器
│   ├── build_vocab_bin.py     # tokenizer.json → vocab.bin（字节解码修复版）
│   ├── extract_llama_asm.py   # 从 llama.cpp 提取 4x4 asm GEMM（MIT，见文件头）
│   ├── verify_attest.py       # 可验证推理凭证离线验签（零依赖）
│   ├── vllm_vqf_sign.c        # VQF SM2 供应链签名 / 密钥管理工具
│   └── vllm_mgr.py            # 引擎进程守护（start/stop/restart/状态页）
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

> 平台约束：本引擎目标为 **RK3588 (aarch64)**，`vllm_platform.h` 在其他架构上
> 编译时报错退出。运行建议 `export OMP_NUM_THREADS=8`（4×A76 + 4×A55）。

## 运行

```bash
# 1) 模型目录需含：config.json + model.safetensors（或 model.vqf / model.gguf）+ vocab.bin
#    vocab.bin 可由 tools/build_vocab_bin.py 从 tokenizer.json 生成：
#    python tools/build_vocab_bin.py <tokenizer.json> <vocab.bin> <vocab.bin>

# 2) 启动 OpenAI 兼容 HTTP 服务（默认端口 8080）
./vllm_kestrel --serve --port 8080 --model <model-dir> --auto-load --wmode q4

#    长上下文优化档（对应上文 8K 基准，全部可选；prefix-kv 前缀复用默认开启）：
./vllm_kestrel --serve --port 8080 --model <model-dir> --auto-load \
    --wmode q4 --sparse-attn --sparse-k 32 --spec --spec-k 4 \
    --disk-kv <kv-dir> --threads 8

# 3) 对话（OpenAI 兼容）
curl http://<board>:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-vl","messages":[{"role":"user","content":"你好"}],"max_tokens":64}'

# 管理页 / 聊天页 / 转换页：http://<board>:8080/admin/  :8080/chat/  :8080/convert/
```

权重格式转换（一次性，均在本引擎内完成，输出单文件 mmap 的 VQF）：

```bash
./vllm_kestrel --convert-vqf <out.vqf> --model <safetensors-model-dir> --wmode q4
./vllm_kestrel --convert-gguf <out.vqf> --model <model.gguf> --wmode q4
```

NPU 加速（可选）：默认后端为**零第三方依赖直驱**（自研写 stock rknpu 内核驱动）。
首次使用先跑板端校准：`./vllm_kestrel --npu --npu-selftest --perf-only`
（寄存器命令表未校准通过前提交路径保持禁用，自动回退 CPU）。

权重保护与可验证推理：转换时可选口令加密（VQF-Enc）与 SM2 供应链签名；启动时置
`VLLM_VQF_KEY` / `VLLM_VQF_SIGN_PUB` 解密验签，置 `VLLM_ATTEST=1`（详见
[docs/权重保护与可验证推理方案.md](docs/权重保护与可验证推理方案.md)）即逐响应出证，
用 `tools/verify_attest.py` 离线验签。

## 模型与复现

- 模型权重不随仓库分发。Qwen3-VL 系列权重遵循其原始开源许可（Qwen 社区许可），
  下载后可用上文命令转换为 VQF/加载。
- 基准数据复现方法、语料与驱动位置见
  [docs/RK3588_性能基准报告.md](docs/RK3588_性能基准报告.md) 附录。

## 文档（docs/，中文）

| 文档 | 内容 |
|---|---|
| [docs/技术文档.md](docs/技术文档.md) | 架构、模块、权重格式 VQF、内核、服务层、多模态、上下文管理、位级确定性、NPU、性能、调试、版本演进 |
| [docs/RK3588_性能基准报告.md](docs/RK3588_性能基准报告.md) | vllm_kestrel vs llama.cpp 冷启动 / 长上下文 / KV 恢复全矩阵 |
| [docs/优化配置与边界说明.md](docs/优化配置与边界说明.md) | 各优化档机制、收益与诚实边界 |
| [docs/权重保护与可验证推理方案.md](docs/权重保护与可验证推理方案.md) | 三道安全防线：VQF 存储态加密、SM2 供应链签名、推理出证（schema=2），含相互关系、端到端用法与统一安全边界 |

## 第三方与合规

- 本项目为**双许可**：学习 / 学术研究 / 论文复现免费；任何商业使用须先取得
  商业授权。完整条款见 [LICENSE](LICENSE)。
- 第三方组件按各自许可保留：`stb_image.h`（MIT, Sean Barrett）与源自 llama.cpp
  的 4x4 asm GEMM 提取文件及其派生内核（MIT, The ggml authors）——版权与许可
  文本见对应文件头，详见 [LICENSE](LICENSE) 第三节。
- 本引擎的 NPU 直驱后端仅与操作系统内核驱动（stock rknpu）的公共 UAPI 交互，
  不包含任何闭源库或第三方头文件。

## 联系

For research collaborations / commercial licensing / data requests, please
open an issue in this repository or contact the authors.
