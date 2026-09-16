#!/bin/sh
# kv2_chain.sh — §7 重测的自主收尾链（板端自持，跑完自动停）
#   1) 等 full_v2（2K/4K/8K）结束
#   2) 汇总 full_v2
#   3) 把 VQF 头 max_seq 8192 -> 20480（只改 4 字节元数据，自动备份）
#   4) 跑 16K（1300 段，1 轮，两侧交错）
#   5) 用 --restore 回滚 max_seq，并打印 md5 自证回到 b6d8d1d7…
#   6) 汇总 16K
# 用法（板端）：setsid nohup sh /mnt/emmc/kv2h2/kv2_chain.sh >/dev/null 2>&1 &
set -u

H=/mnt/emmc/kv2h2
LOG=$H/chain.log
VQF=/mnt/VQF/8b/serve/model.vqf
FULL=/mnt/emmc/kv2_full_v2
K16=/mnt/emmc/kv2_16k_v2
: > "$LOG"

say() { echo "$1 [$(date '+%F %T')]" >> "$LOG"; }

say "CHAIN_START"

# ---- 1) 等 full_v2 结束（最多 12 h 兜底）----
i=0
while [ $i -lt 720 ]; do
  grep -q KV2_DONE "$FULL/summary.txt" 2>/dev/null && break
  sleep 60; i=$((i+1))
done
say "FULL_FLAG=$(grep -c KV2_DONE "$FULL/summary.txt" 2>/dev/null) WAITED_MIN=$i"

# ---- 2) 汇总 full_v2 ----
python3 "$H/kv2_sum2.py" "$FULL" >> "$LOG" 2>&1
say "SUM_FULL_DONE"

# ---- 3) 抬 max_seq ----
cd /mnt/emmc/kestrel_pull || exit 1
python3 tools/bench/patch_vqf_max_seq.py --dump "$VQF" >> "$LOG" 2>&1
python3 tools/bench/patch_vqf_max_seq.py --set 20480 "$VQF" >> "$LOG" 2>&1
echo "md5_16k_patched=$(md5sum "$VQF" | cut -d' ' -f1)" >> "$LOG"
say "HDR_SET"

# ---- 4) 跑 16K ----
sh "$H/kv2_run2.sh" 1300 1 16k_v2 vllm,llama 1,0 >> "$LOG" 2>&1
say "RUN16K_FLAG=$(grep -c KV2_DONE "$K16/summary.txt" 2>/dev/null)"

# ---- 5) 回滚 max_seq ----
python3 tools/bench/patch_vqf_max_seq.py --restore "$VQF" >> "$LOG" 2>&1
echo "md5_restored=$(md5sum "$VQF" | cut -d' ' -f1)" >> "$LOG"
say "HDR_RESTORED"

# ---- 6) 汇总 16K ----
python3 "$H/kv2_sum2.py" "$K16" >> "$LOG" 2>&1
say "SUM16K_DONE"

echo "CHAIN_DONE" >> "$LOG"
