#!/bin/sh
# probe16k.sh —— 只验证引擎能否按新 max_seq(=20480) 启动，不做推理
EXE=/mnt/emmc/kestrel_pull/build-rk3588/vllm_kestrel
L3=/mnt/emmc/kv2_l3_probe
pkill -9 -f 'vllm_kestre[l]' 2>/dev/null
sleep 2
rm -rf "$L3"; mkdir -p "$L3"
env OMP_NUM_THREADS=4 VLLM_THREADS=4 VLLM_NPU_FORCE_CPU=1 VLLM_VQF_STREAM=0 \
    VLLM_L3_PREFIX_REUSE=1 VLLM_TP_SPIN=1 \
  nohup "$EXE" --serve --port 19200 --device arm-rk3588-opi5 --model /mnt/VQF/8b/serve \
    --auto-load --threads 4 --sparse-attn --sparse-k 32 --l3-evict --l3-ratio 0.75 \
    --l3-min-seq 128 --l3-path "$L3" --prefill-batch 256 \
  > /tmp/probe16k.log 2>&1 &
sleep 30
echo "=== SERVE/max_seq ==="
grep -m4 -E 'SERVE|max_seq|VQF-STREAM' /tmp/probe16k.log
echo "=== health ==="
curl -s -m 3 http://127.0.0.1:19200/health
echo
echo "=== rss ==="
pid=$(pgrep -f 'vllm_kestre[l]' | head -1)
[ -n "$pid" ] && awk '/VmRSS|VmHWM/{print $1, $2, $3}' /proc/$pid/status
pkill -9 -f 'vllm_kestre[l]' 2>/dev/null
echo PROBE_DONE
