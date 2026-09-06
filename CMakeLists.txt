cmake_minimum_required(VERSION 3.16)
project(vllm_kestrel C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# ================================================================
# 单平台：RK3588 (aarch64, Cortex-A76/A55)
#   - NEON 恒可用；ARMv8.2 dotprod (vdotq_s32) 与 fp16 在 A76 可用。
#   - -mcpu=cortex-a76 选择完整 RK3588 ISA。
#   - 并行：OpenMP（板端 gcc 原生支持）。
#   - 优化档：-O2。2026-08-29 板端实测（2B Q4_0，132-token prefill）：-O2 比
#     -Os 快 ~5%（2306→2194ms，主要是 attention/norm/量化区段；GEMM 为手写
#     asm 不受影响），位级一致验证通过（同 prompt 输出文本逐 token 一致）。
#     -Os 的体积优势（~0.5MB vs ~0.6MB）在 57GB 板端存储下无意义。
#   - -s：strip 符号表。
# 覆盖 march：cmake -DVLLM_MARCH=armv8.2-a+dotprod+fp16
# ================================================================
if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
    if(NOT VLLM_MARCH)
        message(FATAL_ERROR "vllm_kestrel targets aarch64 (RK3588) only; found CMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR}")
    endif()
endif()

if(NOT VLLM_MARCH)
    set(VLLM_MARCH "armv8.2-a+dotprod+fp16")
endif()

# ================================================================
# 零依赖静态构建（可选）：-DVLLM_STATIC=ON 产出无任何 .so 依赖的二进制
#   （仅内核 vdso），可迁移到任意 aarch64 Linux 设备，与目标 glibc 版本无关。
#   静态 glibc 下 dlopen（NPU rknn 后端）仍可编译，运行时加载失败则回退 CPU；
#   NPU direct 后端（默认，零依赖直写 /dev/rknpu）不受影响。
# ================================================================
option(VLLM_STATIC "Build a fully static (zero .so dependency) binary" OFF)

set(CMAKE_C_FLAGS_RELEASE "-O2 -mcpu=cortex-a76 -march=${VLLM_MARCH} -ffast-math -fopenmp -D_GNU_SOURCE")
set(CMAKE_C_FLAGS_DEBUG "-O0 -g -march=${VLLM_MARCH} -fopenmp -D_GNU_SOURCE")
if(VLLM_STATIC)
    # glibc 全静态：libc.a/libm.a/libgomp.a 内联，pthread 已并入 glibc 2.35 libc.a。
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "-s -static")
else()
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "-s")
endif()

# Source files (functional layout: core engine / serve / model / npu)
set(VLLM_SOURCES
    src/main.c
    src/core/vllm_superpos.c
    src/core/vllm_attention.c
    src/core/vllm_scheduler.c
    src/core/vllm_transformer.c
    src/core/vllm_tokenizer.c
    src/core/vllm_matmul.c
    src/core/vllm_tp.c
    src/core/vllm_l3.c
    src/core/vllm_fhe.c
    src/core/vllm_ckks.c
    src/core/vllm_ntt.c
    src/common/vllm_device.c
    src/common/vllm_crypto.c
    src/serve/vllm_http.c
    src/serve/vllm_server.c
    src/serve/vllm_batch.c
    src/serve/vllm_admin.c
    src/serve/vllm_attest.c
    src/serve/embedded_web.c
    src/model/vllm_safetensors.c
    src/model/vqf.c
    src/model/vllm_gguf.c
    src/model/vllm_tokenizer_qwen.c
    src/model/vllm_vision.c
    src/model/vllm_media.c
    src/npu/vllm_npu.c
    src/npu/rk3588/vllm_npu_direct.c
)

# Headers (functional layout mirrors src/)
set(VLLM_HEADERS
    include/common/vllm_platform.h
    include/common/vllm_util.h
    include/common/vllm_utf8_utils.h
    include/common/vllm_device.h
    include/common/vllm_crypto.h
    include/common/vllm_sign_keys.h
    include/core/vllm_superpos.h
    include/core/vllm_matmul.h
    include/core/vllm_tp.h
    include/core/vllm_l3.h
    include/core/vllm_fhe.h
    include/core/vllm_ckks.h
    include/core/vllm_ntt.h
    include/serve/vllm_http.h
    include/serve/vllm_server.h
    include/serve/vllm_batch.h
    include/serve/vllm_attest.h
    include/serve/embedded_web.h
    include/model/vllm_safetensors.h
    include/model/vqf.h
    include/model/vqf_format.h
    include/model/vllm_tokenizer_qwen.h
    include/model/vllm_vision.h
    include/model/vllm_media.h
    include/npu/vllm_npu.h
    include/npu/rk3588/vllm_npu_direct.h
)

# Main executable
add_executable(vllm_kestrel ${VLLM_SOURCES} ${VLLM_HEADERS})

# Headers are included by flat names ("vllm_xxx.h") from every module, so
# expose each functional include/ subdirectory on the search path.
target_include_directories(vllm_kestrel PRIVATE
    include/common
    include/core
    include/serve
    include/model
    include/npu
    include/npu/rk3588
)

# Link math library (for expf, sqrtf)
target_link_libraries(vllm_kestrel PRIVATE m)

# Link RKNN NPU runtime when present (transparent interface; optional).
#   -DVLLM_RKNN_DIR=</opt/npu/rknn-toolkit2/rknpu2/runtime/Linux/librknn_api>
# If not given, vllm_npu.c dlopen()s librknnrt.so at runtime instead.
# NOTE: the DEFAULT backend is the in-tree zero-dependency driver
# (src/npu/rk3588/vllm_npu_direct.c, talks to the stock rknpu kernel node)
# - no external library is needed for NPU acceleration.
if(VLLM_RKNN_DIR)
    target_include_directories(vllm_kestrel PRIVATE ${VLLM_RKNN_DIR}/include)
    target_link_libraries(vllm_kestrel PRIVATE ${VLLM_RKNN_DIR}/lib/librknnrt.so)
    target_compile_definitions(vllm_kestrel PRIVATE VLLM_RKNN_STATIC_LINK=1)
endif()

# Increase stack size to handle large local arrays (scores[4096], etc.).
# Linux: main() calls st_raise_stack_limit() so the ELF link flag is NOT used.
# Optional: add test target
enable_testing()
add_test(NAME vllm_kestrel_test COMMAND vllm_kestrel)
