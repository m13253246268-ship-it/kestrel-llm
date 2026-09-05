#!/usr/bin/env bash
# build_rk3588.sh - Build + verify vllm_kestrel for the RK3588 (aarch64) target.
#
# Usage:
#   ./build_rk3588.sh                 # native build on the RK3588 board
#   ./build_rk3588.sh --cross         # cross-compile on an x86_64 host
#   ./build_rk3588.sh --static        # fully static build (zero .so dependency)
#   ./build_rk3588.sh --rknn <dir>    # also link librknnrt.so statically
#   ./build_rk3588.sh --run-tests     # run the PASS/FAIL self-tests after build
#
# Environment (RKNN NPU path on the board):
#   export VLLM_RKNN_LIB=/usr/lib/librknnrt.so   (or add to ldconfig)
#
# NPU offload (transparent; falls back to CPU when absent).
#
# DIRECT backend (default - zero third-party dependencies, no model
# conversion): the engine's OWN minimal driver talks straight to the stock
# Rockchip rknpu kernel driver (/dev/rknpu, part of the board BSP). The
# engine's Q8_0/Q4_0 block weights feed the NPU as-is.
#   1) Make sure the rknpu kernel driver is loaded:
#        ls /dev/rknpu        (or /dev/accel/accel0)
#   2) Build:  ./build_rk3588.sh --run-tests
#   3) Run:    ./vllm_kestrel --npu --npu-selftest --perf-only
#   NOTE: the matmul register-command table is calibrated on the target via
#   --npu-selftest (CPU-reference PASS/FAIL). Until it is verified, the NPU
#   submit path stays disabled and the engine runs the CPU (NEON) path.
#
# RKNN backend (optional; requires exporting operator .rknn models):
#   1) python3 tools/npu_export_ops.py --config <model-config-dir>/config.json \
#        --out npu_ops --mode int8 --m-list 1,16 --skip-lmhead
#   2) ./vllm_kestrel --npu --npu-backend rknn --npu-dir npu_ops --npu-selftest --perf-only
set -euo pipefail

cd "$(dirname "$0")"

CROSS=0
RKNN_DIR=""
RUN_TESTS=0
STATIC=0
BUILD_DIR="build-rk3588"

for a in "$@"; do
  case "$a" in
    --cross)    CROSS=1 ;;
    --static)   STATIC=1 ;;
    --rknn)     shift; RKNN_DIR="$1" ;;
    --run-tests) RUN_TESTS=1 ;;
    *) echo "unknown arg: $a" >&2; exit 1 ;;
  esac
done

CMAKE_ARGS=(-DCMAKE_BUILD_TYPE=Release)
if [ "$CROSS" = "1" ]; then
  CMAKE_ARGS+=(-DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-aarch64-rk3588.cmake)
fi
if [ "$STATIC" = "1" ]; then
  CMAKE_ARGS+=(-DVLLM_STATIC=ON)
  BUILD_DIR="build-rk3588-static"
fi
if [ -n "$RKNN_DIR" ]; then
  CMAKE_ARGS+=(-DVLLM_RKNN_DIR="$RKNN_DIR")
fi

cmake -B "$BUILD_DIR" "${CMAKE_ARGS[@]}"
cmake --build "$BUILD_DIR" -j"$(nproc 2>/dev/null || echo 4)"
cp src/serve/admin.html "$BUILD_DIR/admin.html"
cp src/serve/chat.html "$BUILD_DIR/chat.html"

echo
echo "=== vllm_kestrel built: $BUILD_DIR/vllm_kestrel ==="
echo "  arch      : $(file "$BUILD_DIR/vllm_kestrel" | sed 's/.*: //')"
echo "  stack     : raised to 16MB in main() (ELF has no /STACK flag)"
echo "  OpenMP    : export OMP_NUM_THREADS=8 (4x A76 + 4x A55)"

# Keep the repo-root ./vllm_kestrel in lockstep with the build artifact. Every
# ./vllm_kestrel invocation on the board then runs the LATEST kernel code -
# without this, stale binaries silently invalidate perf/quality runs
# (2026-08-22: a 00:05 M4b binary at the repo root invalidated all M4c runs).
cp "$BUILD_DIR/vllm_kestrel" ./vllm_kestrel && echo "  root sync : ./vllm_kestrel updated to $BUILD_DIR/vllm_kestrel"

if [ "$RUN_TESTS" = "1" ]; then
  BIN="$BUILD_DIR/vllm_kestrel"
  echo
  echo "=== self-tests (PASS/FAIL) ==="
  set +e
  "$BIN" --test-l3     > /tmp/l3.log 2>&1;  echo "test-l3      : $([ $? -eq 0 ] && echo PASS || echo FAIL)"
  "$BIN" --test-sparse > /tmp/sp.log 2>&1; echo "test-sparse  : $([ $? -eq 0 ] && echo PASS || echo FAIL)"
  "$BIN" --bench-mixed > /tmp/bm.log 2>&1; echo "bench-mixed  : $([ $? -eq 0 ] && echo PASS || echo FAIL)"
  "$BIN" --npu-selftest > /tmp/npu.log 2>&1
  echo "npu-selftest : $([ $? -eq 0 ] && echo PASS || echo FAIL) (SKIP = runtime absent)"
  set -e
fi

echo
echo "=== NPU offload quick check (zero-dependency direct driver) ==="
echo "  ./vllm_kestrel --npu --npu-selftest --perf-only"
echo
echo "=== long-context quality gate (G1: decode <= 1.5x A0 baseline) ==="
echo "  ./vllm_kestrel --longctx-quality --l3-evict --kv-q4 1> out 2> err"
