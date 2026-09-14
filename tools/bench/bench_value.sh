#!/bin/sh
# =============================================================================
# bench_value.sh —— 逐层推理「价值」实测：内存收益 + 热态稳定性（板端脚本）
#
# 回答两个问题：
#   A. 逐层推理到底省多少内存？—— 全层 vs 逐层 的权重常驻 RSS（冷页缓存同口径 A/B）
#   B. 逐层档的热态是否稳定？—— 连续 N 次同请求的 TTFT/tpot 波动 + 峰值 VmHWM
#
# 用法（板端执行；所有参数均可用环境变量覆盖）：
#   sh tools/bench/bench_value.sh
#   MODEL_DIR=/path/to/dir MODEL_FILE=/path/to/model.vqf sh tools/bench/bench_value.sh
#
# 环境变量：
#   EXE        引擎可执行文件        默认 /mnt/emmc/New_vLLM/vllm_shs
#   MODEL_DIR  serve 用模型目录（含 model.vqf + vocab.bin + config.json）
#   MODEL_FILE stream-test 用单文件 VQF（同目录需有 vocab.bin）
#   OUT        结果目录              默认 ./bench_value_out
#   PORT       serve 端口            默认 18092
#   N_WARM     热态样本数            默认 5
#   MAXTOK     单次最大生成 token    默认 96
#   THREADS    推理线程数            默认 4 —— RK3588 上务必用 4（绑 4×A76）。
#                                    实测：8 线程把 4×A55 拉进 GEMM，decode 慢约 2×。
#   DEVICE     设备档位              默认 arm-rk3588-opi5
#   NO_THINK   1 = 请求带 enable_thinking=false（短问答推荐）默认 1
#
# 产物：$OUT/{stream_full.log,stream_layer.log,serve.log,http.json,http.err}
# 依赖：curl、python3、sha256sum；探针 tools/bench/bench_http_probe.py 与本脚本同目录
# =============================================================================
set -u

EXE=${EXE:-/mnt/emmc/New_vLLM/vllm_shs}
MODEL_DIR=${MODEL_DIR:-/mnt/VQF/8b/serve}
MODEL_FILE=${MODEL_FILE:-/mnt/VQF/8b/qwen3vl8b.q4.vqf}
OUT=${OUT:-./bench_value_out}
PORT=${PORT:-18092}
N_WARM=${N_WARM:-5}
MAXTOK=${MAXTOK:-96}
THREADS=${THREADS:-4}
DEVICE=${DEVICE:-arm-rk3588-opi5}
NO_THINK=${NO_THINK:-1}

HERE=$(cd "$(dirname "$0")" && pwd)
PROBE="$HERE/bench_http_probe.py"

mkdir -p "$OUT"
pkill -9 vllm_shs 2>/dev/null
sleep 1

echo "=== 0) 环境 ==="
echo "exe        : $EXE"
echo "model_dir  : $MODEL_DIR"
echo "model_file : $MODEL_FILE"
echo "out        : $OUT"
echo "threads    : $THREADS"
if command -v sha256sum >/dev/null 2>&1; then
    echo "exe sha256 : $(sha256sum "$EXE" | cut -d' ' -f1)"
fi
echo "mem        :"; free -m | head -2
echo "model size : $(ls -l "$MODEL_FILE" 2>/dev/null | awk '{print $5}') bytes"

# -----------------------------------------------------------------------------
# A) 全层 vs 逐层：同口径 A/B（冷页缓存）
# -----------------------------------------------------------------------------
echo
echo "=== A1) --stream-test 全层（冷页缓存） ==="
sync; echo 3 > /proc/sys/vm/drop_caches; sleep 3
env OMP_NUM_THREADS=$THREADS VLLM_THREADS=$THREADS VLLM_NPU_FORCE_CPU=1 \
    "$EXE" --stream-test --model "$MODEL_FILE" --stream-n 32 --threads "$THREADS" \
    > "$OUT/stream_full.log" 2>&1
grep -E 'rss_after_load|rss_after_prefill|rss_end|decoded|TOKIDS' "$OUT/stream_full.log" | head -8

echo
echo "=== A2) --stream-test 逐层（冷页缓存） ==="
sync; echo 3 > /proc/sys/vm/drop_caches; sleep 3
env OMP_NUM_THREADS=$THREADS VLLM_THREADS=$THREADS VLLM_NPU_FORCE_CPU=1 VLLM_VQF_STREAM=1 \
    "$EXE" --stream-test --model "$MODEL_FILE" --stream-n 32 --threads "$THREADS" \
    > "$OUT/stream_layer.log" 2>&1
grep -E 'VQF-STREAM|rss_after_load|rss_after_prefill|rss_end|decoded|TOKIDS' "$OUT/stream_layer.log" | head -9

# -----------------------------------------------------------------------------
# B) 逐层档 serve：冷启动 + N 次热态同请求（稳定度）
# -----------------------------------------------------------------------------
echo
echo "=== B) serve 逐层 + 短请求（NO_THINK=$NO_THINK, N_WARM=$N_WARM） ==="
sync; echo 3 > /proc/sys/vm/drop_caches; sleep 3
env OMP_NUM_THREADS=$THREADS VLLM_THREADS=$THREADS VLLM_NPU_FORCE_CPU=1 VLLM_VQF_STREAM=1 \
    nohup "$EXE" --serve --port "$PORT" --device "$DEVICE" \
    --model "$MODEL_DIR" --auto-load > "$OUT/serve.log" 2>&1 &
ENGINE_PID=$!
echo "engine_pid=$ENGINE_PID"

for _ in $(seq 1 150); do
    if curl -s -o /dev/null "http://127.0.0.1:$PORT/health"; then
        echo "health ready"
        break
    fi
    sleep 2
done

if [ "$NO_THINK" = "1" ]; then THINK_ARG="--no-think"; else THINK_ARG=""; fi
python3 "$PROBE" --port "$PORT" --n "$N_WARM" --max-tokens "$MAXTOK" $THINK_ARG \
    > "$OUT/http.json" 2>"$OUT/http.err"
echo "--- probe 逐次结果 ---"
cat "$OUT/http.err"
echo "--- http.json ---"
cat "$OUT/http.json"

# -----------------------------------------------------------------------------
# C) 页缓存证据：权重能否被 RAM 整体容纳（决定逐层热态是否稳定）
# -----------------------------------------------------------------------------
echo
echo "=== C) 页缓存占用（权重是否装得进 RAM） ==="
free -m
grep -E '^Cached|^MemAvailable|^MemTotal' /proc/meminfo

kill $ENGINE_PID 2>/dev/null
echo
echo "=== 完成，产物在 $OUT/ ==="
