#!/bin/sh
# launch.sh <segs> <rounds> <tag> [engines] [streams] —— 彻底脱离 ssh 会话启动对照测试
cd /tmp/kv2 || exit 1
pkill -9 -f 'kv2_run[2]' 2>/dev/null
pkill -9 -f 'vllm_kestre[l]' 2>/dev/null
pkill -9 -f 'llama-serve[r]' 2>/dev/null
pkill -9 -f 'md5su[m]' 2>/dev/null
sleep 2
setsid nohup sh kv2_run2.sh "$1" "$2" "$3" "$4" "$5" > /tmp/kv2/run_$3.out 2>&1 < /dev/null &
echo LAUNCHED_$3
