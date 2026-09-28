#!/bin/sh
# kv2_stop_after_r2.sh — 按用户指示：8K(600) 跑完第 2 轮即停，不跑第 3 轮、不跑 16K。
#   1) 先杀 kv2_chain.sh（否则它会在 KV2_DONE 后自动去改 max_seq 跑 16K）
#   2) 等 8K r2 的最后一个 arm（llama n600 r2）打完 turn1..turn3
#   3) 杀掉 kv2_run2.sh 与两侧引擎
#   4) 记录口径偏差并生成 REPORT.txt
# 用法（板端）：setsid nohup sh /mnt/emmc/kv2h2/kv2_stop_after_r2.sh >/dev/null 2>&1 &
set -u

H=/mnt/emmc/kv2h2
LOG=$H/stop.log
SUM=/mnt/emmc/kv2_full_v2/summary.txt
: > "$LOG"
say() { echo "$1 [$(date '+%F %T')]" >> "$LOG"; }

say "STOPPER_START"

# ---- 1) 掐断自动链 ----
pkill -9 -f 'kv2_chai[n]' 2>/dev/null
sleep 2
say "CHAIN_KILLED alive=$(ps -ef | grep -c 'kv2_chai[n]')"

# ---- 2) 等 8K r2 打完（llama n600 r2 的 3 个 turn 行）----
i=0
while [ $i -lt 240 ]; do
  n=$(awk '/\[llama n600 r2\]/{f=1;next} f&&/^turn/{c++} END{print c+0}' "$SUM" 2>/dev/null)
  [ "${n:-0}" -ge 3 ] && break
  sleep 30; i=$((i+1))
done
say "R2_WAITED_MIN=$i turn_lines=$n"

# ---- 3) 停跑 ----
pkill -9 -f 'kv2_run2[.]sh' 2>/dev/null
pkill -9 -f 'vllm_kestre[l]' 2>/dev/null
pkill -9 -f 'llama-serve[r]' 2>/dev/null
sleep 3
say "PROCS_LEFT vllm=$(ps -ef | grep -c 'vllm_kestre[l]') llama=$(ps -ef | grep -c 'llama-serve[r]') run=$(ps -ef | grep -c 'kv2_run2[.]sh')"

# ---- 4) 记录口径偏差（不改数，只标注）----
{
  echo "NOTE_STOP=按用户指示：8K(segs=600) 只跑 2 轮（r3 未跑），16K 档未跑（链已掐断）。"
  echo "STOP_END=$(date '+%F %T')"
} >> "$SUM"

python3 "$H/kv2_sum2.py" /mnt/emmc/kv2_full_v2 > /mnt/emmc/kv2_full_v2/REPORT.txt 2>&1
say "SUM_DONE"

echo STOPPER_DONE >> "$LOG"
