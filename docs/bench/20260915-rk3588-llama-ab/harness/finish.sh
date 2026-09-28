#!/bin/sh
# finish.sh <tag> —— 等测试写完全部即自动汇总，结果落在 /mnt/emmc/kv2_<tag>/REPORT.txt
T=$1
S=/mnt/emmc/kv2_$T/summary.txt
while :; do
  grep -q KV2_DONE "$S" 2>/dev/null && break
  sleep 60
done
python3 /tmp/kv2/kv2_sum2.py "/mnt/emmc/kv2_$T" > "/mnt/emmc/kv2_$T/REPORT.txt" 2>&1
echo ANALYSIS_DONE >> "/mnt/emmc/kv2_$T/REPORT.txt"
