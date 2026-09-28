#!/bin/sh
# kv2_run2.sh — 权重公平长上下文对照 v2（修正 pilot 暴露的两个问题）
#   * 修正1：丢弃用预热请求 max_tokens=1（pilot 里误用 64，白跑 60 秒）
#   * 修正2：llama 的 -c 按实际 token 估算（≈13 token/段）而非 30，避免 KV 预分配过大
#
# 用法: sh kv2_run2.sh <SEGS> <ROUNDS> <TAG> [ENGINES] [STREAMS]
set -u
SEGS="$1"; ROUNDS="$2"; TAG="$3"; ENGINES="${4:-vllm,llama}"; STREAMS="${5:-1,0}"

EXE=/mnt/emmc/kestrel_pull/build-rk3588/vllm_kestrel
VMODEL=/mnt/VQF/8b/serve
GGUF=/mnt/emmc/qwen3vl8b-q4_0.gguf
LLAMA=/root/llama_build_rk3588/bin/llama-server
OUT=/mnt/emmc/kv2_$TAG
L3DIR=/mnt/emmc/kv2_l3_$TAG
PV=19200; PL=19201; NTOK=64; TH=4
SEG="边缘计算与云计算的核心区别在于数据处理发生的位置。"

mkdir -p "$OUT"; SUM="$OUT/summary.txt"; : > "$SUM"
echo "TAG=$TAG SEGS=$SEGS ROUNDS=$ROUNDS ENGINES=$ENGINES STREAMS=$STREAMS TH=$TH NTOK=$NTOK" >> "$SUM"
echo "START=$(date '+%F %T')" >> "$SUM"
echo "gov=$(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_governor)" >> "$SUM"
MD5C=/tmp/kv2/md5_${TAG}.txt
if [ ! -s "$MD5C" ]; then
  { echo -n "md5_exe=";  md5sum $EXE | cut -d' ' -f1
    echo -n "md5_gguf="; md5sum $GGUF | cut -d' ' -f1
    echo -n "md5_vqf=";  md5sum /mnt/VQF/8b/serve/model.vqf | cut -d' ' -f1
  } > "$MD5C"
fi
cat "$MD5C" >> "$SUM"

killer() { pkill -9 -f 'vllm_kestre[l]' 2>/dev/null; pkill -9 -f 'llama-serve[r]' 2>/dev/null; sleep 3; }
dropc()  { sync; echo 3 > /proc/sys/vm/drop_caches 2>/dev/null; sleep 2; }
wait_http() { i=0; while [ $i -lt 300 ]; do
    if curl -s -m 3 "http://127.0.0.1:$1/health" 2>/dev/null | grep -q '"status":"ok"'; then return 0; fi
    sleep 2; i=$((i+1)); done; return 1; }
chk() { # $1=json path -> 成功返回 0，失败打印告警
  grep -q '"choices"' "$1" 2>/dev/null && return 0
  echo "  !! BAD_RESP $(head -c 160 "$1" 2>/dev/null)" >> "$SUM"; return 1; }
hwm() { [ -r "/proc/$1/status" ] && awk '/VmHWM/{print $2}' "/proc/$1/status"; }
rss() { [ -r "/proc/$1/status" ] && awk '/VmRSS/{print $2}' "/proc/$1/status"; }

mk_body() { # $1=out $2=segs $3=turn $4=maxtok
  {
    printf '{"model":"qwen3vl","max_tokens":%d,"temperature":0,"top_p":1,"messages":[{"role":"user","content":"' "$4"
    i=0; while [ $i -lt "$2" ]; do printf '%s' "$SEG"; i=$((i+1)); done
    printf '\\n请用一句话总结上面这段话。'
    [ "$3" -ge 2 ] && printf '\\n\\n追问一：请把这段话压缩成一个不超过 10 字的小标题。'
    [ "$3" -ge 3 ] && printf '\\n\\n追问二：请给出一个反例，说明这个区别在什么情况下不成立。'
    printf '"}]}'
  } > "$1"
}

warm() { # $1=port $2=path
  mk_body /tmp/kv2_warm.json 3 1 1
  curl -s -m 1800 -H 'Content-Type: application/json' -d @/tmp/kv2_warm.json \
    "http://127.0.0.1:$1$2" -o /dev/null -w 'warm_s=%{time_total}'
}

run_vllm() { # $1=stream $2=segs $3=round
  st="$1"; n="$2"; rd="$3"
  log="$OUT/vllm_s${st}_${n}_r${rd}.log"
  rm -rf "$L3DIR"; mkdir -p "$L3DIR"
  dropc
  env OMP_NUM_THREADS=$TH VLLM_THREADS=$TH VLLM_NPU_FORCE_CPU=1 VLLM_VQF_STREAM=$st \
      VLLM_L3_PREFIX_REUSE=1 VLLM_TP_SPIN=1 \
    nohup "$EXE" --serve --port $PV --device arm-rk3588-opi5 --model "$VMODEL" --auto-load \
      --threads $TH --sparse-attn --sparse-k 32 --l3-evict --l3-ratio 0.75 --l3-min-seq 128 \
      --l3-path "$L3DIR" --prefill-batch 256 > "$log" 2>&1 &
  pid=$!
  if ! wait_http $PV; then echo "[vllm s$st n$n r$rd] START_FAIL" >> "$SUM"; killer; return 1; fi
  echo "[vllm s$st n$n r$rd] pid=$pid $(warm $PV /v1/chat/completions)" >> "$SUM"
  t=1
  while [ $t -le 3 ]; do
    pj="$OUT/vllm_s${st}_${n}_r${rd}_t${t}.json"
    mk_body /tmp/kv2_req.json "$n" "$t" "$NTOK"
    tt=$(curl -s -m 5400 -H 'Content-Type: application/json' -d @/tmp/kv2_req.json \
          "http://127.0.0.1:$PV/v1/chat/completions" -o "$pj" -w '%{time_total}')
    echo "turn$t total_s=$tt rss_kb=$(rss $pid) hwm_kb=$(hwm $pid)" >> "$SUM"
    chk "$pj"
    t=$((t+1))
  done
  grep -aE 'PREFILL-TIMING|KV-PREFIX|\[L3\]' "$log" | tail -16 >> "$SUM"
  killer
}

run_llama() { # $1=segs $2=round
  n="$1"; rd="$2"
  log="$OUT/llama_${n}_r${rd}.log"
  ctx=$(( n * 13 + 1024 ))
  dropc
  nohup "$LLAMA" -m "$GGUF" -t $TH -c $ctx -ngl 0 --host 127.0.0.1 --port $PL \
      > "$log" 2>&1 &
  pid=$!
  if ! wait_http $PL; then echo "[llama n$n r$rd] START_FAIL" >> "$SUM"; killer; return 1; fi
  echo "[llama n$n r$rd] ctx=$ctx pid=$pid $(warm $PL /v1/chat/completions)" >> "$SUM"
  t=1
  while [ $t -le 3 ]; do
    pj="$OUT/llama_${n}_r${rd}_t${t}.json"
    mk_body /tmp/kv2_req.json "$n" "$t" "$NTOK"
    tt=$(curl -s -m 5400 -H 'Content-Type: application/json' -d @/tmp/kv2_req.json \
          "http://127.0.0.1:$PL/v1/chat/completions" -o "$pj" -w '%{time_total}')
    echo "turn$t total_s=$tt rss_kb=$(rss $pid) hwm_kb=$(hwm $pid)" >> "$SUM"
    chk "$pj"
    t=$((t+1))
  done
  grep -aE 'prompt eval time|eval time|n_past|slot' "$log" | tail -12 >> "$SUM"
  killer
}

killer
for n in $(echo "$SEGS" | tr ',' ' '); do
  rd=1
  while [ $rd -le "$ROUNDS" ]; do
    for ENG in $(echo "$ENGINES" | tr ',' ' '); do
      if [ "$ENG" = "vllm" ]; then
        for st in $(echo "$STREAMS" | tr ',' ' '); do run_vllm "$st" "$n" "$rd"; done
      else
        run_llama "$n" "$rd"
      fi
    done
    rd=$((rd+1))
  done
done
echo "END=$(date '+%F %T')" >> "$SUM"
echo KV2_DONE >> "$SUM"
