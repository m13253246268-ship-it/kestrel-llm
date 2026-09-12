#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""bench_http_probe.py —— 对 vllm_shs 的 OpenAI 兼容接口做流式延迟探针（零第三方依赖）。

配合 tools/bench_value.sh 使用，也可单独运行。

用法:
  python3 bench_http_probe.py [--port 18092] [--n 5] [--max-tokens 96]
                             [--prompt "..."] [--no-think] [--timeout 3600]

输出: 单个 JSON 到 stdout，含每次请求的 ttft/total/tokens/tpot 与汇总（最小/中位/最大）。

口径:
  ttft = 首个 content chunk 到达时刻
  tpot = (total - ttft) / (tokens - 1)
  首个请求为 warmup（冷态，不计入汇总），其后 n 次为热态稳定度样本。
  峰值内存取引擎进程的 VmHWM（/proc/<pid>/status）。
"""
import argparse
import json
import os
import statistics
import sys
import time
import urllib.request

DEF_PROMPT = "请用一句话解释什么是边缘计算，以及它和云计算的区别。"


def engine_vmhwm():
    """读取引擎进程的 VmHWM（峰值物理内存），单位 kB。"""
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open("/proc/%s/cmdline" % pid, "rb") as f:
                cmd = f.read().decode("utf-8", "replace")
        except Exception:
            continue
        if "vllm_shs" not in cmd and "vllm_kestrel" not in cmd:
            continue
        try:
            with open("/proc/%s/status" % pid) as f:
                for line in f:
                    if line.startswith("VmHWM"):
                        return int(line.split()[1])
        except Exception:
            pass
    return None


def one_request(port, prompt, max_tokens, thinking, timeout):
    body = {
        "model": "bench",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }
    if thinking is not None:
        body["enable_thinking"] = thinking
    req = urllib.request.Request(
        "http://127.0.0.1:%d/v1/chat/completions" % port,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    ttft = None
    tokens = 0
    text = []
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                j = json.loads(payload)
            except Exception:
                continue
            ch = j.get("choices") or []
            if not ch:
                continue
            c = (ch[0].get("delta") or {}).get("content")
            if not c:
                continue
            if ttft is None:
                ttft = time.time() - t0
            tokens += 1
            text.append(c)
    total = time.time() - t0
    return {
        "ttft_s": round(ttft, 3) if ttft is not None else None,
        "total_s": round(total, 3),
        "tokens": tokens,
        "tpot_ms": round((total - ttft) * 1000.0 / (tokens - 1), 1) if ttft and tokens > 1 else None,
        "vmhwm_kb": engine_vmhwm(),
        "text_head": "".join(text)[:80],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18092)
    ap.add_argument("--n", type=int, default=5, help="热态样本数（不含 warmup）")
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--prompt", default=DEF_PROMPT)
    ap.add_argument("--no-think", action="store_true", help="传 enable_thinking=false")
    ap.add_argument("--timeout", type=int, default=3600)
    a = ap.parse_args()

    thinking = False if a.no_think else None
    runs = []
    for i in range(a.n + 1):
        tag = "warmup" if i == 0 else "warm_%d" % i
        r = one_request(a.port, a.prompt, a.max_tokens, thinking, a.timeout)
        r["tag"] = tag
        runs.append(r)
        print("[probe] %-8s ttft=%ss total=%ss tokens=%s tpot=%sms"
              % (tag, r["ttft_s"], r["total_s"], r["tokens"], r["tpot_ms"]),
              file=sys.stderr, flush=True)

    warm = runs[1:]

    def _vals(key):
        return [r[key] for r in warm if r[key] is not None]

    def med(key):
        v = _vals(key)
        return round(statistics.median(v), 3) if v else None

    out = {
        "port": a.port,
        "prompt": a.prompt,
        "max_tokens": a.max_tokens,
        "thinking": thinking,
        "cold_warmup": runs[0],
        "runs": warm,
        "summary_warm": {
            "samples": len(warm),
            "ttft_median_s": med("ttft_s"),
            "ttft_min_s": round(min(_vals("ttft_s")), 3) if _vals("ttft_s") else None,
            "ttft_max_s": round(max(_vals("ttft_s")), 3) if _vals("ttft_s") else None,
            "tpot_median_ms": med("tpot_ms"),
            "tpot_min_ms": round(min(_vals("tpot_ms")), 1) if _vals("tpot_ms") else None,
            "tpot_max_ms": round(max(_vals("tpot_ms")), 1) if _vals("tpot_ms") else None,
            "tokens_median": med("tokens"),
            "peak_vmhwm_kb": max([r["vmhwm_kb"] for r in runs if r["vmhwm_kb"]] or [0]) or None,
        },
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
