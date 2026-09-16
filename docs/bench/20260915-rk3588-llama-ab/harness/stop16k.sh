#!/bin/sh
# stop16k.sh —— 停掉第二轮 16K 及所有守护脚本
pkill -9 -f 'kv2_run[2]' 2>/dev/null
pkill -9 -f 'kv3_lon[g]' 2>/dev/null
pkill -9 -f 'chain16[k]' 2>/dev/null
pkill -9 -f 'finish.s[h]' 2>/dev/null
pkill -9 -f 'watch.s[h]' 2>/dev/null
pkill -9 -f 'vllm_kestre[l]' 2>/dev/null
pkill -9 -f 'llama-serve[r]' 2>/dev/null
pkill -9 -f 'md5su[m]' 2>/dev/null
sleep 3
echo "== 残留进程 =="
ps -ef | grep -E 'kv2_run|vllm_kestrel|llama-serve' | grep -v grep
echo "== 已跑到的位置 =="
grep -c . /mnt/emmc/kv2_16kR2/summary.txt 2>/dev/null
tail -5 /mnt/emmc/kv2_16kR2/summary.txt 2>/dev/null
echo STOPPED
