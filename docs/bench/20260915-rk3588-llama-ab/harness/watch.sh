#!/bin/sh
# watch.sh <tag> —— 每 60 秒把测试进度追加到 /tmp/watch_<tag>.txt（后台常驻）
T=$1
W=/tmp/watch_$T.txt
: > "$W"
while :; do
  {
    date '+%F %T'
    echo "  -- procs --"
    ps -ef | grep -E 'kv2_run|llama-serve|vllm_kestrel' | grep -v grep | head -6
    echo "  -- summary tail --"
    tail -5 "/mnt/emmc/kv2_$T/summary.txt" 2>/dev/null
    echo "-----"
  } >> "$W" 2>&1
  sleep 60
done
