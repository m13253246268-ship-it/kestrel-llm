# Cross-compilation toolchain for RK3588 (aarch64 Linux).
#
# Two ways to use:
#   1) Cross-compile on an x86_64 Linux host with gcc-aarch64-linux-gnu:
#        cmake -B build-rk3588 \
#              -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-aarch64-rk3588.cmake \
#              -DVLLM_RKNN_DIR=/path/to/rknn-toolkit2/rknpu2/runtime/Linux/librknn_api
#        cmake --build build-rk3588 -j
#
#   2) Native build directly on the RK3588 board (Ubuntu aarch64):
#        cmake -B build-rk3588 && cmake --build build-rk3588 -j
#      (skip the toolchain file; the CMakeLists aarch64 branch applies
#       -mcpu=cortex-a76 -march=armv8.2-a+dotprod+fp16 automatically)
#
# NOTE: ASLR equivalent on Linux is irrelevant (no /DYNAMICBASE); the
# MSVC-only /DYNAMICBASE:NO constraint does not apply to aarch64 builds.
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_C_COMPILER_TARGET aarch64-linux-gnu)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

# RK3588: Cortex-A76 (big) + A55 (LITTLE), NEON + fp16 + dotprod.
set(VLLM_MARCH "armv8.2-a+dotprod+fp16" CACHE STRING "RK3588 ISA target")
