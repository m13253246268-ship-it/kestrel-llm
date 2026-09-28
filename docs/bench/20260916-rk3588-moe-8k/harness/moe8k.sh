#!/bin/sh
# moe8k.sh — 等 16K 收工后，跑 30B-A3B MoE + 组合⑤，验证能否吃下 8K 上下文
#   组合⑤ = VLLM_ACTQ=1 VLLM_MOE_BATCH=1（专家激活量化 + 专家批量，仅对 q4 MoE 权重）
#   30B-A3B 权重 17.66 GB > 15.9 GB RAM ⇒ 必须 VLLM_VQF_STREAM=1（逐层）
#   用法（板端）：setsid nohup sh /mnt/emmc/kv2h2/moe8k.sh >/dev/null 2>&1 &
set -u

H=/mnt/emmc/kv2h2
LOG=$H/moe8k.log
BOARD=/mnt/emmc/kestrel_pull
EXE=$BOARD/build-rk3588/vllm_kestrel
MDIR=/mnt/VQF/qwen3-30B-A3B-q4
VQF=$MDIR/model.vqf
L3=/mnt/emmc/moe8k_l3
OUT=/mnt/emmc/moe8k
PV=19200; TH=4; NTOK=64; SEGS=600
SEG="边缘计算与云计算的核心区别在于数据处理发生的位置。"
: > "$LOG"; mkdir -p "$OUT" "$L3"
say() { echo "$1 [$(date '+%F %T')]" >> "$LOG"; }
rss() { [ -r "/proc/$1/status" ] && awk '/VmRSS/{print $2}' "/proc/$1/status"; }
hwm() { [ -r "/proc/$1/status" ] && awk '/VmHWM/{print $2}' "/proc/$1/status"; }
killer() { pkill -9 -f 'vllm_kestre[l]' 2>/dev/null; sleep 3; }

say "MOE8K_START"
echo "gov_before=$(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_governor)" >> "$LOG"

# ---- 0) 等 16K 收工（最多 2 h）----
i=0
while [ $i -lt 240 ]; do
  grep -q 16K_DONE "$H/chain16k.log" 2>/dev/null && break
  sleep 30; i=$((i+1))
done
say "WAITED_16K_MIN=$i"
killer
df -h /mnt/emmc | tail -1 >> "$LOG"

# ---- 1) governor 顶到 performance ----
for p in /sys/devices/system/cpu/cpufreq/policy*; do
  echo performance > "$p/scaling_governor" 2>/dev/null
done
echo "gov_after=$(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_governor)" >> "$LOG"
say "GOV_SET"

# ---- 2) 头部自检：max_seq 是否够 8K ----
cd "$BOARD" || exit 1
python3 tools/bench/patch_vqf_max_seq.py --dump "$VQF" >> "$LOG" 2>&1
# 2026-09-16 修复：原用 /dim /{print $3} 会同时命中 "dim" 与 "head_dim"，
# 使 DIM="2048\n128"，下面的 [ "$DIM" -ge 512 ] 直接报 integer expression expected 而恒假
# （守卫失效）。改为按行首字段名精确锚定。
DIM=$(python3 tools/bench/patch_vqf_max_seq.py --dump "$VQF" | awk '/^[[:space:]]*dim[[:space:]]/{print $3}')
LAY=$(python3 tools/bench/patch_vqf_max_seq.py --dump "$VQF" | awk '/^[[:space:]]*layers[[:space:]]/{print $3}')
MS=$(python3 tools/bench/patch_vqf_max_seq.py --dump "$VQF" | awk '/^[[:space:]]*max_seq[[:space:]]/{print $3}')
echo "parsed dim=$DIM layers=$LAY max_seq=$MS" >> "$LOG"
echo "md5_vqf_before=$(md5sum "$VQF" | cut -d' ' -f1)" >> "$LOG"
PATCHED=0
# 只在头字段看起来合理时才动它（避免对不认识的布局误写）
if [ "${LAY:-0}" -ge 24 ] && [ "${LAY:-0}" -le 96 ] && [ "${DIM:-0}" -ge 512 ] && [ "$MS" -lt 8192 ]; then
  python3 tools/bench/patch_vqf_max_seq.py --set 8192 "$VQF" >> "$LOG" 2>&1
  PATCHED=1
  echo "md5_vqf_patched=$(md5sum "$VQF" | cut -d' ' -f1)" >> "$LOG"
fi
say "HDR_CHECK_DONE patched=$PATCHED"

# ---- 3) 启引擎：逐层 + 组合① + 组合⑤ ----
LOGONE=$OUT/moe_8k_serve.log
rm -rf "$L3"; mkdir -p "$L3"
sync; echo 3 > /proc/sys/vm/drop_caches 2>/dev/null; sleep 2
env OMP_NUM_THREADS=$TH VLLM_THREADS=$TH VLLM_NPU_FORCE_CPU=1 \
    VLLM_VQF_STREAM=1 VLLM_L3_PREFIX_REUSE=1 VLLM_TP_SPIN=1 \
    VLLM_ACTQ=1 VLLM_MOE_BATCH=1 \
  nohup "$EXE" --serve --port $PV --device arm-rk3588-opi5 --model "$MDIR" --auto-load \
    --threads $TH --sparse-attn --sparse-k 32 --l3-evict --l3-ratio 0.75 --l3-min-seq 128 \
    --l3-path "$L3" --prefill-batch 256 > "$LOGONE" 2>&1 &
PID=$!
say "ENGINE_PID=$PID"

# 30B 启动到就绪给足 15 min
i=0; OK=0
while [ $i -lt 450 ]; do
  if curl -s -m 3 "http://127.0.0.1:$PV/health" 2>/dev/null | grep -q '"status":"ok"'; then OK=1; break; fi
  kill -0 $PID 2>/dev/null || break
  sleep 2; i=$((i+1))
done
say "HEALTH_OK=$OK waited_s=$((i*2))"
if [ "$OK" = "0" ]; then
  echo "!! START_FAIL, 引擎日志尾：" >> "$LOG"
  tail -25 "$LOGONE" >> "$LOG"
  killer
  [ "$PATCHED" = "1" ] && python3 tools/bench/patch_vqf_max_seq.py --restore "$VQF" >> "$LOG" 2>&1
  echo "MOE8K_DONE rc=2" >> "$LOG"; exit 2
fi
echo "rss_after_load_kb=$(rss $PID) hwm_after_load_kb=$(hwm $PID)" >> "$LOG"
grep -aE 'VQF|moe|MoE|ACTQ|stream|STREAM' "$LOGONE" | head -12 >> "$LOG"

mk_body() { # $1=out $2=segs $3=turn $4=maxtok
  {
    printf '{"model":"qwen3","max_tokens":%d,"temperature":0,"top_p":1,"messages":[{"role":"user","content":"' "$4"
    i=0; while [ $i -lt "$2" ]; do printf '%s' "$SEG"; i=$((i+1)); done
    printf '\\n请用一句话总结上面这段话。'
    [ "$3" -ge 2 ] && printf '\\n\\n追问一：请把这段话压缩成一个不超过 10 字的小标题。'
    printf '"}]}'
  } > "$1"
}

# ---- 4) 预热（丢弃用）----
mk_body /tmp/moe_warm.json 3 1 1
echo "warm_s=$(curl -s -m 1800 -H 'Content-Type: application/json' -d @/tmp/moe_warm.json \
      "http://127.0.0.1:$PV/v1/chat/completions" -o /dev/null -w '%{time_total}')" >> "$LOG"
say "WARM_DONE"

# ---- 5) 8K 两轮（turn1 整段，turn2 追加追问，验前缀复用 + L3 回填）----
for t in 1 2; do
  pj="$OUT/moe_8k_t${t}.json"
  mk_body /tmp/moe_req.json "$SEGS" "$t" "$NTOK"
  echo "req_body_bytes=$(wc -c < /tmp/moe_req.json)" >> "$LOG"
  tt=$(curl -s -m 10800 -H 'Content-Type: application/json' -d @/tmp/moe_req.json \
        "http://127.0.0.1:$PV/v1/chat/completions" -o "$pj" -w '%{time_total}')
  echo "turn$t total_s=$tt rss_kb=$(rss $PID) hwm_kb=$(hwm $PID)" >> "$LOG"
  if grep -q '"choices"' "$pj" 2>/dev/null; then
    echo "turn$t OK resp_bytes=$(wc -c < "$pj")" >> "$LOG"
  else
    echo "turn$t BAD_RESP head=$(head -c 200 "$pj" 2>/dev/null)" >> "$LOG"
  fi
done
grep -aE 'PREFILL-TIMING|KV-PREFIX|\[L3\]' "$LOGONE" | tail -20 >> "$LOG"

# ---- 6) 收尾 ----
killer
if [ "$PATCHED" = "1" ]; then
  python3 tools/bench/patch_vqf_max_seq.py --restore "$VQF" >> "$LOG" 2>&1
  echo "md5_vqf_restored=$(md5sum "$VQF" | cut -d' ' -f1)" >> "$LOG"
fi
# 2026-09-16 修复：pk.sh 是 shell 脚本，原来误用 python3 调用 → SyntaxError，抽数那步空跑。
sh "$H/pk.sh" "$OUT" 'moe_8k_t*.json' >> "$LOG" 2>&1
echo "MOE8K_DONE rc=0 $(date '+%F %T')" >> "$LOG"
