#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""kv2_sum2.py — 权重公平长上下文对照汇总（按增量 token 口径）
用法: python3 kv2_sum2.py /mnt/emmc/kv2_<TAG>

口径说明：
  * 我方（vllm）：prefill 的真增量 token 数取自同轮日志的 [PREFILL-TIMING] n=
    （每个进程按顺序出现：预热1行 + turn1/2/3 各1行），避免用 prompt_tokens 总量冒充速率。
  * llama：直接用响应 timings 的 prompt_n / prompt_ms（本身就是增量）。
  * decode：我方 1000/tpot_ms；llama predicted_per_second。
  * 内存：取 summary.txt 里每轮的 VmHWM（kB）。
"""
import glob
import json
import os
import re
import sys

d = sys.argv[1] if len(sys.argv) > 1 else "."
rows = []
hwm = {}


def load_json(f):
    try:
        with open(f, encoding="utf-8", errors="replace") as fh:
            return json.load(fh)
    except Exception:
        return None


# ---- 1) 解析 summary：VmHWM + md5 ----
sumf = os.path.join(d, "summary.txt")
hwm_cur = None
if os.path.exists(sumf):
    eng_cur = None
    for ln in open(sumf, encoding="utf-8", errors="replace"):
        m = re.search(r"\[(vllm s(\d) n(\d+) r(\d+))\]", ln)
        if m:
            eng_cur = ("vllm s%s" % m.group(2), int(m.group(3)), int(m.group(4)))
            hwm[eng_cur] = []
            continue
        m = re.search(r"\[llama n(\d+) r(\d+)\]", ln)
        if m:
            eng_cur = ("llama", -1, int(m.group(1)), int(m.group(2)))
            hwm[eng_cur] = []
            continue
        m = re.search(r"turn(\d) total_s=([\d.]+) rss_kb=(\d+) hwm_kb=(\d+)", ln)
        if m and eng_cur:
            hwm[eng_cur].append((int(m.group(1)), float(m.group(2)), int(m.group(4))))

# ---- 2) 解析我方日志：[PREFILL-TIMING] n= 序列 ----
pf_log = {}
for f in glob.glob(os.path.join(d, "vllm_s*_*_r*.log")):
    b = os.path.basename(f)[:-4]
    m = re.match(r"vllm_s(\d)_(\d+)_r(\d+)$", b)
    if not m:
        continue
    key = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    ns = [int(x) for x in re.findall(r"\[PREFILL-TIMING\] n=(\d+)", open(f, encoding="utf-8", errors="replace").read())]
    pf_log[key] = ns[1:4] if len(ns) >= 4 else ns  # 丢掉预热那一行

# ---- 3) 汇总每条请求 ----
for f in sorted(glob.glob(os.path.join(d, "*.json"))):
    b = os.path.basename(f)[:-5]
    j = load_json(f)
    if j is None:
        continue
    m = re.match(r"vllm_s(\d)_(\d+)_r(\d+)_t(\d+)$", b)
    if m:
        st, n, rd, t = (int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)))
        met = j.get("metrics") or {}
        pf_ms = met.get("prefill_ms") or 0.0
        tpot = met.get("tpot_ms") or 0.0
        seq = pf_log.get((st, n, rd), [])
        inc = seq[t - 1] if t - 1 < len(seq) else 0
        h = hwm.get(("vllm s%d" % st, n, rd), [])
        hh = next((x[2] for x in h if x[0] == t), 0)
        rows.append(dict(eng="vllm", st=st, n=n, rd=rd, t=t, ptot=met.get("prompt_tokens") or 0,
                         inc=inc, pf_ms=pf_ms, pf_tps=(inc * 1000.0 / pf_ms) if pf_ms > 0 and inc else 0.0,
                         tpot=tpot, dc_tps=(1000.0 / tpot) if tpot > 0 else 0.0,
                         hwm_gb=hh / 1048576.0, nt=met.get("n_tokens") or 0,
                         txt=((j.get("choices") or [{}])[0].get("message") or {}).get("content", "")[:16]))
        continue
    m = re.match(r"llama_(\d+)_r(\d+)_t(\d+)$", b)
    if m:
        n, rd, t = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
        ti = j.get("timings") or {}
        inc = ti.get("prompt_n") or 0
        pf_ms = ti.get("prompt_ms") or 0.0
        if inc == 0 and pf_ms == 0.0:
            continue  # 失败/占位的空响应
        h = hwm.get(("llama", -1, n, rd), [])
        hh = next((x[2] for x in h if x[0] == t), 0)
        rows.append(dict(eng="llama", st=-1, n=n, rd=rd, t=t, ptot=inc,
                         inc=inc, pf_ms=pf_ms, pf_tps=ti.get("prompt_per_second") or 0.0,
                         tpot=ti.get("predicted_per_token_ms") or 0.0,
                         dc_tps=ti.get("predicted_per_second") or 0.0,
                         hwm_gb=hh / 1048576.0, nt=ti.get("predicted_n") or 0,
                         txt=((j.get("choices") or [{}])[0].get("message") or {}).get("content", "")[:16]))

if not rows:
    print("NO DATA in %s" % d)
    sys.exit(1)

rows.sort(key=lambda r: (r["n"], r["rd"], r["t"], r["eng"], -r["st"]))
lab = {("llama"): "llama", ("vllm", 1): "vllm逐层", ("vllm", 0): "vllm全层"}


def tag(r):
    return lab.get((r["eng"], r["st"]), r["eng"])


print("=== 明细（增量 token 口径；HWM=该轮进程峰值 RSS）===")
hdr = "%-9s %-6s %-3s %-3s %-8s %-7s %-9s %-10s %-9s %-10s %-8s" % (
    "arm", "segs", "rd", "turn", "prompt_T", "inc_tok", "prefill_ms", "prefill t/s", "tpot_ms", "decode t/s", "HWM_GB")
print(hdr)
print("-" * len(hdr))
for r in rows:
    print("%-9s %-6d %-3d %-3d %-8d %-7d %-9.1f %-10.2f %-9.1f %-10.2f %-8.2f" % (
        tag(r), r["n"], r["rd"], r["t"], r["ptot"], r["inc"], r["pf_ms"], r["pf_tps"], r["tpot"], r["dc_tps"], r["hwm_gb"]))

lut = {}
for r in rows:
    lut[(r["n"], r["rd"], r["t"], r["eng"], r["st"])] = r
keys = sorted({(r["n"], r["rd"], r["t"]) for r in rows})


def cell(x, f):
    return f % x if x else "   n/a"


print()
print("=== 成对：prefill t/s（增量）===")
print("%-6s %-3s %-3s | %-11s %-11s %-11s" % ("segs", "rd", "turn", "llama", "vllm逐层", "vllm全层"))
for k in keys:
    a = lut.get(k + ("llama", -1))
    b1 = lut.get(k + ("vllm", 1))
    b0 = lut.get(k + ("vllm", 0))
    print("%-6d %-3d %-3d | %-11s %-11s %-11s" % (k[0], k[1], k[2],
          cell(a and a["pf_tps"], "%8.2f"), cell(b1 and b1["pf_tps"], "%8.2f"), cell(b0 and b0["pf_tps"], "%8.2f")))

print()
print("=== 成对：decode t/s ===")
print("%-6s %-3s %-3s | %-11s %-11s %-11s" % ("segs", "rd", "turn", "llama", "vllm逐层", "vllm全层"))
for k in keys:
    a = lut.get(k + ("llama", -1))
    b1 = lut.get(k + ("vllm", 1))
    b0 = lut.get(k + ("vllm", 0))
    print("%-6d %-3d %-3d | %-11s %-11s %-11s" % (k[0], k[1], k[2],
          cell(a and a["dc_tps"], "%8.2f"), cell(b1 and b1["dc_tps"], "%8.2f"), cell(b0 and b0["dc_tps"], "%8.2f")))

print()
print("=== 复用轮时延（turn2/turn3 的 prefill_ms，越小越好）===")
print("%-6s %-3s %-3s | %-11s %-11s %-11s" % ("segs", "rd", "turn", "llama", "vllm逐层", "vllm全层"))
for k in keys:
    if k[2] == 1:
        continue
    a = lut.get(k + ("llama", -1))
    b1 = lut.get(k + ("vllm", 1))
    b0 = lut.get(k + ("vllm", 0))
    print("%-6d %-3d %-3d | %-11s %-11s %-11s" % (k[0], k[1], k[2],
          cell(a and a["pf_ms"], "%8.0f"), cell(b1 and b1["pf_ms"], "%8.0f"), cell(b0 and b0["pf_ms"], "%8.0f")))

print()
print("=== 峰值内存（各 arm 的最大 HWM，GB）===")
agg = {}
for r in rows:
    k = tag(r)
    agg[k] = max(agg.get(k, 0.0), r["hwm_gb"])
for k, v in sorted(agg.items(), key=lambda x: x[1]):
    print("  %-9s %.2f GB" % (k, v))

print()
print("=== 输出抽检（前 16 字）===")
for r in rows[:24]:
    print("[%s segs=%d r%d t%d] %s" % (tag(r), r["n"], r["rd"], r["t"], r["txt"]))
print("KV2_SUM2_DONE")
