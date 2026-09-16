#!/bin/sh
# poll.sh <tag> [sleep_s] —— 汇总板端状态到 /tmp/poll.txt（供 scp 取回）
# 可选第二参数：先在板端 sleep 指定秒数（等测试推进），再采集。
[ -n "$2" ] && sleep "$2"
{
  date
  echo "== procs =="
  ps -ef | grep -E 'kv2_run|llama-serve|vllm_kestrel|md5sum' | grep -v grep
  echo "== kv2 dirs =="
  ls -l /mnt/emmc/ | grep kv2
  echo "== mem =="
  free -m
  echo "== run out =="
  tail -20 "/tmp/kv2/run_$1.out" 2>/dev/null
  echo "== summary =="
  tail -45 "/mnt/emmc/kv2_$1/summary.txt" 2>/dev/null
  echo "== files =="
  ls -l "/mnt/emmc/kv2_$1/" 2>/dev/null
} > /tmp/poll.txt 2>&1
echo POLL_DONE >> /tmp/poll.txt
