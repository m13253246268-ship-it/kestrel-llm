# Kestrel — ARM (aarch64) Edge LLM Inference Engine (RK3588 Development Baseline)

[简体中文](README.md) | **English**

**A pure-C11, zero-third-party-runtime LLM inference engine for ARM (aarch64) CPUs.**

> **Version: v1.0 (test release).** For the diff against the previous public version (v0), see
> "v1.0 Change Summary" at the end. Performance/size figures in this document are partitioned
> by version: **figures without a version tag are v1.0 board measurements (2026-09-12)**;
> figures tagged `(v0 measurement)` come from "Appendix A" at the end — they were measured on
> v0 and **have not been re-measured on v1, so they must not be used for any v1 conclusion**.
> When reproducing on another board, take `./build_rk3588.sh --run-tests` (current self-test
> and artifacts) as the source of truth.

**v0 / v1 capability comparison (the partition used by this README)**

| Capability | v0 (previous public release) | **v1.0 (this release)** |
|---|---|---|
| Weight loading | GGUF / safetensors loading, plus **in-engine conversion** (`vllm_gguf.c`, `convert.html`) | **Pure-VQF runtime: only the in-house VQF v2 single file (mmap-mounted)**; in-engine GGUF/safetensors loading and conversion removed — conversion moved to the shipped `vqf_convert/` tool |
| Weight residency | Fully resident (all layers) | New **per-layer residency** (`VLLM_VQF_STREAM=1`): resident weight RSS decoupled from model size / layer count; **plaintext VQF only** — VQF-Enc / SM2-signed weights are explicitly rejected (must stay fully resident) |
| KV & long context | KV v1 | **KV v2 lazy allocation** + L3 tiered residency + P3 (L3 eviction × prefix reuse coexisting) |
| Security & compliance | VQF at-rest encryption (SM4-CTR + HMAC-SM3), SM2 supply-chain signature | Adds **verifiable-inference attestation (schema 3, bound to the raw request body)**, verified in-browser or offline with zero dependencies |
| x86-64 | none | Shipped x86-64 portability layer (`tools/build_x64.ps1` / `check_x64.ps1`), **for functional self-test and bit-exactness comparison only — never a performance baseline** |
| Benchmark points | 2026-09-05: cold start / 8K long context / KV restore (with llama.cpp side-by-side) | **2026-09-12**: per-layer vs full residency × {2B / 8B / 30B-A3B} × {combo ⑧ baseline, combo ①, combo ①+⑤}, 14 serve configurations |

> Why they must not be mixed: v1 removed the GGUF/safetensors loading path and introduced
> per-layer residency and KV v2, so **the definitions behind cold start, resident memory,
> long context and multi-turn prefill have all changed**. v0 absolute values are historical
> reference only; anything requiring v1 numbers must use the main body of this document
> (2026-09-12 measurements).

> Platform scope: this engine is fundamentally an **ARM-CPU inference engine**
> (ARMv8.2-A + NEON dotprod/fp16, no GPU/NPU and no particular board required); **RK3588** is
> the current development, optimization and benchmarking platform, not the only supported
> device — comparable aarch64 Linux devices can be built and tried (cross-device validation
> status is in the "Quickstart" section).

---

## Core Strengths

One ARM board (e.g. RK3588 / Orange Pi 5 Plus) + one ~**0.8 MB** single-binary executable
= a native LLM inference service: **the in-house VQF v2 weight file is mmap-mounted and paged
in on demand**, Qwen3-VL 2B/8B run natively for text plus image/video input, and an
OpenAI-compatible HTTP API is available immediately.

| Strength | One-line metric |
|---|---|
| **Per-layer residency** (new in v1) | Resident memory decoupled from model size: **8B resident weights 4.19 GB → 0.48 GB (8.7×)**, serve peak **0.82 GB**; warm **TTFT 1.95 s / tpot 377 ms**, spread over 5 samples **±1.8% / ±0.13%** |
| **Long text and multi-turn** | Combo ① (L3 eviction × prefix reuse) cuts follow-up-turn prefill by **−80% ~ −96%**; with MoE combo ⑤ the 30B reaches **4.7 s / 3.2 s** for t2/t3 |
| **In-house format and kernels** | Only the in-house VQF v2 single file; single ~0.8 MB binary, zero third-party runtime; hand-written NEON quantized GEMM/GEMV, own thread pool, own SM2/SM3/SM4 |
| **Verifiable inference** | SM2 supply-chain signature protects weights + per-request attestation proof (schema 3), verified in-browser or offline with zero dependencies |

### The memory advantage of per-layer residency (worked example)

`VLLM_VQF_STREAM=1` makes each layer's weights establish file pages only while that layer
computes, then release them immediately with `MADV_DONTNEED`, keeping only the `keep` layers
(1 by default) resident — **resident weight RSS no longer grows linearly with model size**.

Three real models (`--stream-test`, cold page cache, weight-only figures):

| Model (single-file VQF) | Full: resident / peak | **Per-layer: resident / peak** | Resident-memory ratio |
|---|---|---|---|
| 2B Qwen3-VL (4.16 GB) | 1.81 GB / 2.76 GB | **0.39 GB / 0.63 GB** | **4.6×** |
| 8B Qwen3-VL (6.60 GB) | 4.18 GB / 4.22 GB | **0.47 GB / 0.59 GB** | **8.9×** |
| 30B-A3B MoE (17.66 GB) | 11.73 GB / 12.70 GB | **0.51 GB / 0.71 GB** | **23.2×** |

**The canonical example**: an RK3588 board has only 16 GB RAM (MemTotal 15.6 GiB), while the
Qwen3-30B-A3B-q4 weight file alone is 17.66 GB — the full-residency tier does barely run, but
its serve peak reaches 14.2 GB, leaving almost no headroom for KV or the process itself.
**Per-layer residency brings the same model down to a 1.11 GB serve peak, with ample headroom
and stable serving.** The 2B case is even more direct: resident weights of just 0.39 GB, 4.6×
less than full residency, at the cost of warm-state tpot rising from 113.0 ms to 156.7 ms
(+38.7%).

**The recommended tier is 8B** (measured 2026-09-12, see "Performance" §4): resident weights
4.19 GB → **0.48 GB (8.7×)**, serve peak **0.82 GB**, warm **TTFT 1.95 s / tpot 377 ms**
(27.9 s end-to-end for 70 tokens), with a **±0.13% tpot spread over 5 samples**. Its 6.15 GiB of
weights are **smaller than physical RAM**, so the page cache holds them and warm-state stability
is guaranteed by physics — something the 30B cannot claim (see "Performance" §4 and §5).

**Semantics unchanged**: eviction only drops clean file pages, and content is rebuilt from the
file — the greedy TOKIDS sequence produced by `--stream-test` is **bit-identical** between the
two residency tiers (see the md5 comparison under "Performance").

> Limits: per-layer residency **supports plaintext VQF only** (VQF-Enc / SM2-signed weights are
> explicitly rejected and must stay fully resident), and it does not coexist with expert windows
> (`VLLM_EW*`). Full definitions, costs and matrices are under "Performance".
> **The 30B is the capability-ceiling tier, not the recommended tier**: its 16.4 GiB of weights
> do not fit in the 15.6 GiB page cache, and with thinking on by default a single Q&A needs
> 3~4 minutes (see "Performance" §5).

---

## Performance

### How performance figures are partitioned

The main body uses **v1.0 (2026-09-12) measurements only**: 3 models (2B / 8B / 30B-A3B) ×
{full residency, per-layer residency} × {combo ⑧ baseline, combo ①, combo ①+⑤} as a full
matrix. **The v0 (2026-09-05) cold start / 8K long context / KV restore data was moved
wholesale to "Appendix A"** — it was measured on v0 and not re-measured on v1 (v1 removed the
GGUF/safetensors loading path and introduced per-layer residency and KV v2, so the definitions
changed). **It must not be used for any v1 conclusion**, and must not be mixed with the v1
numbers below.

### Per-layer vs full residency (RK3588, v1.0 measurement, 2026-09-12)

`VLLM_VQF_STREAM=1` puts the engine into **tiered (per-layer) residency** mode: the VQF single
file is still mmap-mounted, but each layer's weights establish file pages only while that layer
computes, then release them immediately with `MADV_DONTNEED`, keeping only the `keep` layers
(1 by default) resident. Together with v1.0's KV v2 lazy allocation (the KV base grows on
demand), **resident memory is decoupled from model size and layer count** — a 17.66 GB
Qwen3-30B-A3B-q4 can be served on a 16 GB RK3588, where the same model fully resident needs a
12.7 GB peak (`--stream-test`) / 14.2 GB peak (serve, including KV). This section covers two
measurement methods (`--stream-test` same-basis A/B, and serve cold/warm plus 3 follow-up
turns) over 14 serve configurations, with no OOM, no failure and no fallback throughout.

Self-evidence from the board log:

```
[VQF-STREAM] enabled keep=1 nl=48 segs=11 per-layer=334.1MB data=16847.2MB resident~808.4MB rss=988kB
```

**Semantics unchanged**: the greedy TOKIDS sequence from `--stream-test` is **bit-identical**
between the two residency tiers — the last column below is the first 12 hex digits of the md5
of that sequence, and the "full" / "per-layer" rows of the same model carry the same value
(each of the three models agrees internally); eviction only drops clean file pages and content
is rebuilt from the file, so no value changes.

Reproduction basis (board binary `vllm_shs`, sha256 `e1484740…a8f8e8`): models were
`/mnt/emmc/Modl/Qwen3-VL-2B-Instruct/qwen3vl2b.dual.vqf` (4.16 GB),
`/mnt/VQF/8b/qwen3vl8b.q4.vqf` (6.60 GB) and `/mnt/VQF/qwen3-30B-A3B-q4` (17.66 GB);
`--stream-test` used `--threads 8`; serve used `--serve --port 18080 --device arm-rk3588-opi5
--auto-load`, the baseline added `--no-prefix-kv`, combo ① added
`--sparse-attn --sparse-k 32 --l3-evict --l3-ratio 0.75 --l3-min-seq 128 --l3-path /mnt/emmc/l3bench`
plus the environment variable `VLLM_L3_PREFIX_REUSE=1`, combo ⑤ added
`VLLM_ACTQ=1 VLLM_MOE_BATCH=1`; per-layer mode is the environment variable `VLLM_VQF_STREAM=1`.

#### 1) Same-basis A/B (`--stream-test`: 32-token prefill + 32-token greedy decode, cold page cache)

| Model (single-file VQF) | Residency | Weight RSS (after prefill) | Peak VmHWM | prefill 32tok | decode | TOKIDS |
|---|---|---|---|---|---|---|
| 2B (Qwen3-VL-2B, dual, 4.16 GB) | full | 1,806,732 kB | 2,761,732 kB | 7.14 s | 221 ms/tok | `1a5d48906a4c` |
| 2B | **per-layer** | **394,464 kB (4.6×)** | **631,428 kB (4.4×)** | 8.02 s (+12%) | 257 ms/tok (+16%) | `1a5d48906a4c` |
| 8B (Qwen3-VL-8B, q4, 6.60 GB) | full | 4,183,048 kB | 4,224,032 kB | 73.39 s | 429 ms/tok | `efb5a00c5803` |
| 8B | **per-layer** | **471,800 kB (8.9×)** | **593,172 kB (7.1×)** | 72.48 s (−1%) | 586 ms/tok (+37%) | `efb5a00c5803` |
| 30B-A3B (MoE, q4, 17.66 GB) | full | 11,728,860 kB | 12,703,020 kB | 273.6 s | 2442 ms/tok | `d4996200fcf2` |
| 30B-A3B | **per-layer** | **506,368 kB (23.2×)** | **713,240 kB (17.8×)** | 273.3 s (−0.1%) | 2840 ms/tok (+16%) | `d4996200fcf2` |

> Method: each configuration is preceded by `sync; echo 3 > /proc/sys/vm/drop_caches`, so this
> is the **first forward pass on a cold page cache** (including the full cost of reading
> GB-scale weights from storage). The 8B/30B prefill times above are therefore dominated
> by weight reads and must not be compared with "warm-cache steady state" (see §2); the 30B's
> 17.66 GB of weights exceed 16 GB of RAM, so the full-residency tier sits right at the memory
> ceiling.
>
> **Storage medium (measured 2026-09-12)**: the v1 8B/30B weights live on a **SanDisk microSD
> card** (`/mnt/VQF` sits on the rootfs, and the rootfs is on the SD card), while the 2B weights
> and the L3 directory are on eMMC. The two differ by 3.8× in measured bandwidth (see §5), so
> the 8B/30B cold-read cost in this section is priced at **62.7 MB/s (SD card)**.
>
> The difference from "4.6×~21.8× saved, prefill +11~25% / decode +121~259%" in
> `docs/优化配置与边界说明.md` comes from the **measurement basis**: that set is **v0-era**
> warm-cache (weights already in RAM/page cache) steady state and **is not a v1 conclusion**.

#### 2) Steady-state basis (serve, "cold → warm" same question in one process; combo ⑧ baseline `--no-prefix-kv`)

| Model | Residency | Peak VmHWM | Cold prefill | Warm prefill | Warm TTFT | Warm tpot |
|---|---|---|---|---|---|---|
| 2B | full | 2,968,920 kB | 13,993 ms | 5,267 ms | 5,267 ms | 113.0 ms |
| 2B | **per-layer** | **854,476 kB (3.5×)** | 14,882 ms | 5,441 ms | 5,441 ms | 156.7 ms |
| 8B | full | 4,523,240 kB | 87,459 ms | 16,035 ms | 16,035 ms | 458.4 ms |
| 8B | **per-layer** | **914,448 kB (4.9×)** | 86,375 ms | 16,463 ms | 16,463 ms | 617.2 ms |
| 30B-A3B | full | 14,228,868 kB | 841,365 ms | 620,774 ms | 620,775 ms | 2,240.4 ms |
| 30B-A3B | **per-layer** | **1,112,500 kB (12.8×)** | 841,969 ms | 623,125 ms | 623,126 ms | 2,647.7 ms |

One request = 300-token context + 32 generated tokens (greedy). Cold = the first request after
`drop_caches`; warm = the same request sent immediately afterwards: the full tier already has
its weights resident in RAM (the warm pass pays compute only), while the per-layer tier touches
the weight pages again every pass (the warm pass is still bound by storage bandwidth) — this is
precisely the marginal cost of trading time for memory.

> Peaks in this table include the KV cache, so the ratios are smaller than the "weight-only
> RSS" ratios in §1 (30B: 12.8× vs 23.2×): what per-layer residency truly decouples is the
> **weights**, while KV is managed separately by KV v2 lazy allocation. The 30B full-residency
> peak of 14.2 GB already hits the ceiling of a 16 GB board (MemTotal 15.6 GiB), whereas the
> per-layer tier compresses the same model to 1.11 GB, leaving all the headroom for KV and the
> process itself.

#### 3) Stacking with the "effective optimization combos" (serve multi-turn follow-ups; combo ① = `--sparse-attn --sparse-k 32 --l3-evict --l3-min-seq 128` + `VLLM_L3_PREFIX_REUSE=1`)

| Model | Residency | Config | t2 prefill | t3 prefill | vs baseline |
|---|---|---|---|---|---|
| 2B | full | ⑧ base | 5,882 ms | 6,545 ms | — |
| 2B | full | ① P3 | **952 ms** | **972 ms** | **−83.8% / −85.1%** |
| 2B | per-layer | ⑧ base | 6,057 ms | 6,706 ms | — |
| 2B | per-layer | ① P3 | **1,189 ms** | **1,130 ms** | **−80.4% / −83.1%** |
| 8B | full | ⑧ base | 19,251 ms | 21,193 ms | — |
| 8B | full | ① P3 | **1,885 ms** | **2,008 ms** | **−90.2% / −90.5%** |
| 8B | per-layer | ⑧ base | 19,579 ms | 21,545 ms | — |
| 8B | per-layer | ① P3 | **2,048 ms** | **2,203 ms** | **−89.5% / −89.8%** |
| 30B-A3B | full | ⑧ base | 722,394 ms | 822,409 ms | — |
| 30B-A3B | full | ① P3 | **28,633 ms** | **32,140 ms** | **−96.0% / −96.1%** |
| 30B-A3B | full | ①+⑤ P3+MoE | **4,727 ms** | **3,157 ms** | **−99.3% / −99.6%** |
| 30B-A3B | per-layer | ⑧ base | 724,367 ms | 824,535 ms | — |
| 30B-A3B | per-layer | ① P3 | **29,368 ms** | **32,820 ms** | **−95.9% / −96.0%** |
| 30B-A3B | per-layer | ①+⑤ P3+MoE | **5,269 ms** | **3,695 ms** | **−99.3% / −99.6%** |

Combo ⑤ = `VLLM_ACTQ=1 VLLM_MOE_BATCH=1` (MoE expert activation quantization + expert
batching). It only applies to MoE weights (q4) and stacks orthogonally with combo ① and with
per-layer residency.

Self-evidence for combo ① (raw board log, 2B full tier; all three stages — L3 spill, restore,
prefix reuse — are visible):

```
[L3] evicted 252 blocks -> /mnt/emmc/l3bench_u65bbfe41 (cursor=9.84 MB, seq=345, keep=332, ratio=0.75, ...), freed 78.8 MB from RAM
[L3] restored 252 prefix blocks from Q4 payload (prefix=377)
[KV-PREFIX] reuse 377-token KV prefix, prefill rest
```

The same combo yields gains of the same order in both residency tiers (2B −80% ~ −83%,
8B −89% ~ −90%, 30B −96%), showing that the **weight-residency axis and the KV-paging axis are
orthogonal**. Adding combo ⑤ (MoE) on the 30B-A3B (`VLLM_ACTQ=1 VLLM_MOE_BATCH=1`, q4 weights
required) cuts the second/third follow-up prefill from the ⑧ baseline's 722 s / 822 s down to
**4.7 s / 3.2 s (full tier)**, and the first turn from 620 s to 23 s, with tpot going from
2,240 ms to 504 ms — all measured in this round; the same tier also holds under **per-layer
residency** (t2/t3 = 5.3 s / 3.7 s), i.e. "memory-efficient" and "fast" can be had at once.

#### 4) 8B: the recommended tier delivers (measured 2026-09-12)

§1/§2 above cover all three tiers (2B/8B/30B). This section answers one question: **where does
per-layer residency pay off best — the answer is 8B.**

Configuration: `--threads 4` (`OMP_NUM_THREADS=4 VLLM_THREADS=4`); **weights on a SanDisk
microSD card (62.7 MB/s)**. Memory basis as in §1 (`--stream-test`, 32-token prefill + 32-token
decode, cold page cache A/B). The prefill and decode below come **from the same cold-page-cache
run** (hence decode is slower than the warm-page-cache thread A/B table further down):

| Tier | Weight RSS (after prefill) | rss_end | prefill 32tok | decode |
|---|---|---|---|---|
| full | 4,189,912 kB | 4,202,988 kB | 67.9 s | 207 ms/tok (4.84 tok/s) |
| **per-layer** | **480,348 kB (8.7×)** | **487,912 kB** | 63.2 s | 385 ms/tok (2.60 tok/s) |

Serve in per-layer mode + short request (`enable_thinking=false`, `max_tokens=96`; finished
naturally at 70 tokens):

| Tier | TTFT | End-to-end | tpot | Peak VmHWM |
|---|---|---|---|---|
| Cold (first after `drop_caches`) | 73.1 s | 99.7 s | 385.3 ms | 611,880 kB |
| **Warm (5 identical consecutive requests)** | **1.95 s** (1.899~1.966) | **27.9 s** | **376.6 ms** (376.4~377.4) | 824,192 kB |

> The warm VmHWM is the **high-water mark accumulated over 5 runs** (656,144 → 698,228 → 740,088
> → 782,236 → 824,192 kB, about +42 MB per run); see the VmHWM note in §5 for the basis and the
> open item.

**Why 8B is the recommended tier (three reasons, all reproducible)**:

1. **You capture the full memory win at a controllable cost**: resident weights 4.19 GB →
   **0.48 GB (8.7×)**, serve peak **0.82 GB**.
2. **Warm-state stability is backed by physics**: the 6.15 GiB of weights **fit in the 15.6 GiB
   page cache**, so warm behaviour is not a gamble — the measured spread over 5 samples is
   **TTFT ±1.8%, tpot ±0.13%**.
3. **The one-off cold-start cost is low**: 73 s (vs 195~211 s for the 30B), because the cost is
   proportional to weight size ÷ medium bandwidth.

**The hidden 2×: you must use `--threads 4`** (same board, same model, A/B, 2 repeats each,
warm page cache):

| Tier | `--threads 4` | `--threads 8` | Ratio |
|---|---|---|---|
| full decode | **186 / 189 ms/tok** | 431 / 433 ms/tok | 4 threads **2.3× faster** |
| per-layer decode | **370 / 378 ms/tok** | 580 / 582 ms/tok | 4 threads **1.56× faster** |

> The RK3588 is 4×A76 + 4×A55, and `--threads 8` pulls the four A55 little cores into the GEMM
> parallel region. The 8B figures in §1 (full 429 / per-layer 586 ms/tok) are exactly the
> **`--threads 8`** basis and match the right-hand column above; **switching to 4 threads brings
> 8B full-residency decode to 5.4 tok/s**.

**One-command reproduction**: `sh tools/bench_value.sh` (parameters are overridable via
environment variables; see the header comment in the script) — it produces the A/B memory
comparison, warm-state stability and page-cache evidence, and writes `http.json`.

#### 5) 30B-A3B on a short request: how far it actually goes (thinking switch measured)

> Additional measurement (2026-09-12, same board, same engine sha256 `e1484740…a8f8e8`).
> The question it answers: is the 30B on a 16 GB board merely *barely runnable*, or genuinely
> usable? **The weights sit on a SanDisk microSD card.**

Configuration: per-layer `VLLM_VQF_STREAM=1` + combo ⑤ `VLLM_ACTQ=1 VLLM_MOE_BATCH=1` +
combo ① flags (with only a 22-token context the log shows `[L3] skipped: seq=22 <
l3-min-seq=128`, i.e. **L3 and prefix reuse did not engage**, so this table reflects the
"per-layer + MoE" tier without combo ①'s KV-side gain); `OMP_NUM_THREADS=4 VLLM_THREADS=4`.
Request = 22-token context + 32 generated tokens (greedy, streaming).

| Tier | TTFT | End-to-end | tpot | Peak VmHWM |
|---|---|---|---|---|
| Cold (first request after `drop_caches`) | 211.5 s | 258.8 s | 1,524.5 ms | 666,468 kB |
| **Warm (same request sent immediately after)** | **3.3 s** | **20.8 s** | **564.8 ms** | 918,300 kB |

Cross-checked with the same request in non-streaming mode: 20.7 s (consistent with 20.8 s
streaming); after that run the process high-water mark had accumulated to 933,656 kB.

> **On VmHWM**: it is the process's **historical high-water mark** (monotonically non-decreasing),
> not the steady-state footprint of a single request, so a rising value across runs is expected.
> The two rows above are the marks measured at their respective points; the same applies to the
> 8B five-sample warm run (611,880 → 656,144 → 698,228 → 740,088 → 782,236 → 824,192 kB, about
> +42 MB per run). Whether the per-request footprint falls back would require sampling `VmRSS`
> per run — **we have not done that sampling; it is recorded as an open item**.

**Critical precondition: `enable_thinking` defaults to on** (the engine matches HF
`apply_chat_template`; see `resolve_thinking` in `src/serve/vllm_server.c`). Those 32 tokens
**were all spent inside the thinking block and produced no answer** — measuring only "how long
does it take to emit 32 tokens" yields an over-optimistic usability verdict. Hence the switch
comparison below:

| Tier (max_tokens=256) | TTFT | End-to-end | Actual output | Result |
|---|---|---|---|---|
| thinking **on** (default) | 29.5 s | 221.9 s | **256 (budget exhausted)** | **`</think>` never appears; no answer** |
| **thinking off** (request body `"enable_thinking": false`) | **3.1 s** | **24.5 s** | **40 (natural EOS)** | **complete answer delivered** |

The thinking-off output *is* the answer:
「边缘计算是在数据产生地附近进行数据处理和分析的计算模式，而云计算则是在远程数据中心进行集中式数据处理，两者的主要区别在于数据处理的位置和实时性需求。」

> Note: both requests followed a cold warm-up request. The thinking-off run was the 3rd request
> with the warmest page cache (TTFT 3.1 s), while the thinking-on run was the 2nd (TTFT 29.5 s
> reflects a still-warming page cache) — **the TTFT gap comes mainly from page-cache state, not
> from the thinking switch itself**. The thinking-on run took longer because it never finished
> reasoning within its 256-token budget.

**Revised conclusion: the usable tier for the 30B-A3B on this 16 GB board is "thinking off +
short context".** With thinking disabled it returns a complete one-sentence answer in 24.5 s at
a 0.91 GB peak — that is the real "usable" figure. With thinking on (the default), 256 tokens
are not enough to finish the reasoning block; a single Q&A actually needs 300+ tokens, i.e.
roughly 3~4 minutes. The 258.8 s cold figure is the one-off cost of reading all 17.66 GB of
weights (16.8 GB of which is the weight data segments) from the SD card on first touch; once the
weights sit in the page cache it returns to
seconds — which is exactly how per-layer residency fits a 17.66 GB model onto a 16 GB board.

**Storage medium measured** (same board, `dd iflag=direct`, 1 GiB, bypassing the page cache):

| Medium | Role / mount | Device | Measured sequential read |
|---|---|---|---|
| **SanDisk microSD** | 8B/30B weights (`/mnt/VQF`) | `/dev/mmcblk1` (`name=SD64G`, `type=SD`, `manfid=0x000003`) | **62.7 MB/s** |
| eMMC | 2B weights, L3 directory (`/mnt/emmc`) | `/dev/mmcblk0` (`name=BJTD4R`, `type=MMC`) | **240 MB/s** |

> This explains the cold-read magnitude in §1: in the per-layer tier's "trade time for memory",
> **the time is proportional to weight size ÷ medium bandwidth**. The 211.5 s above is priced at
> the SD card's 62.7 MB/s; **moving the weights to eMMC (240 MB/s) would, by bandwidth ratio,
> bring first token down to roughly 56 s — this is an extrapolation, not a measurement.**

#### 6) Honest boundaries

- **Per-layer residency is plaintext-only**: VQF-Enc / SM2-signed weights are explicitly
  rejected (they must stay fully resident).
- It does not coexist with expert windows (`VLLM_EW*`; EW owns the layer entry hook).
- The speed cost grows with "weight size ÷ storage bandwidth": under `--stream-test` cold page
  cache, prefill +12% (2B) / −1% (8B) / −0.1% (30B) and decode +16% / +37% / +16%; in serve
  steady state (§2) warm tpot is +38.7% (2B) / +34.6% (8B) / +18.2% (30B) while warm prefill is
  only +3.3% / +2.7% / +0.4%. What you get in return is 4.6× / 8.9× / 23.2× resident-memory
  reduction (weight-only RSS).
- **The absolute speed of the 30B-A3B on a 16 GB board is low — that is a hardware boundary,
  not an implementation defect**: an unoptimized baseline needs 620 s for a 300-token context
  (tpot 2,240 ms, ≈0.45 tok/s), and the full tier's 14.2 GB peak already sits at the memory
  ceiling. The practical configuration is "per-layer residency (1.11 GB) + combo ① + combo ⑤"
  — combo ⑤ MoE brings the first turn down to 23 s and follow-ups to 3~5 s, which is what makes
  the 30B actually usable. **See §5 for the short-request measurement** — note that the usable
  tier requires **thinking off**: with it disabled, a complete short answer comes back in 24.5 s
  at a 0.91 GB peak; with the default thinking on, 256 tokens do not finish the reasoning block.
- Per-layer residency compresses **weight residency**; the KV base is constrained separately by
  v1.0's KV v2 lazy allocation (with `VLLM_KV_NOF32=1` the 2B can be pushed further to the
  ~222 MB range — see
  [docs/KV缓存v2-惰性分配与分层驻留方案.md](docs/KV缓存v2-惰性分配与分层驻留方案.md)).

---

## Quickstart

From a bare board to HTTP-ready in 4 steps: **build → weights in place → start → verify**.

### 1) Build (RK3588 / aarch64 Linux)

```bash
# Native build on the board (needs gcc + cmake)
./build_rk3588.sh                    # build + copy to ./vllm_kestrel
./build_rk3588.sh --run-tests        # build + run PASS/FAIL self-tests
./build_rk3588.sh --static           # fully static (zero .so dependencies)

# Or directly on the board:
cmake -B build-rk3588 && cmake --build build-rk3588 -j8

# Cross-compile from an x86_64 Linux host (needs gcc-aarch64-linux-gnu)
./build_rk3588.sh --cross
```

#### x86_64 native build (Windows / MinGW, for functional and consistency self-tests)

The engine's **primary target is aarch64**; the x86-64 branch (`vllm_platform.h`) exists only
for **functional self-tests and bit-exactness comparison** and is **never a performance
baseline** — absolute throughput or speedups measured on x86 must not be extrapolated to the
board.

```powershell
# Windows / MinGW-w64 (gcc must be on PATH, or pass -Gcc explicitly)
powershell -ExecutionPolicy Bypass -File tools\check_x64.ps1
#   → compiles + runs --test-l3 / --test-sparse self-tests, exit code 0/1
powershell -ExecutionPolicy Bypass -File tools\build_x64.ps1 -Gcc D:\tools\mingw64\bin\gcc.exe
#   → build only; artifact build-x64\vllm_kestrel_x64.exe
```

Environment overrides: `VLLM_GCC` (gcc path) and `VLLM_X64_OUTDIR` (output directory) replace
the corresponding command-line arguments.

> Platform constraints: the engine is fundamentally an **ARM (aarch64, ARMv8.2-A +
> dotprod/fp16) CPU inference engine**, and **RK3588 (4×A76 + 4×A55) is the development and
> benchmarking platform**, not the only supported device. Comparable aarch64 Linux devices can
> be built and tried, but the device profile (e.g. A76 cluster thread affinity, core count) and
> performance tiers are validated on RK3588 — **on a different board, run `--test-l3` /
> `--bench-mixed` first and trust the self-test results**. Threads are recommended as
> `export OMP_NUM_THREADS=8` (RK3588: 4×A76 + 4×A55). Non-aarch64 / non-x86-64 architectures
> fail at compile time in `vllm_platform.h`.

### 2) Weights in place

The engine **only loads a VQF v2 single file** (`model.vqf`, produced by the "Conversion Tool"
section). The model directory must contain:

```bash
# config.json + model.vqf (VQF v2 single file, mmap-mounted)
# + optional vocab.bin (generated from tokenizer.json by tools/build_vocab_bin.py;
#   if absent the engine falls back to the embedded vocab — functional, but size/vocab
#   then follow the model's own):
python tools/build_vocab_bin.py <tokenizer.json> <vocab.bin> <vocab.bin>
```

### 3) Start the service

```bash
# Basic start: OpenAI-compatible HTTP service (default port 8080)
./vllm_kestrel --serve --port 8080 --model <model-dir> --auto-load --wmode q4

# Long-context optimization tier (all optional; prefix-kv prefix reuse is on by default):
# Note: the 8K long-context benchmark is a v0 measurement (see Appendix A) and was not
# re-measured on v1; the switches below are the v1 basis.
# --l3-evict requires the environment variable VLLM_L3_PREFIX_REUSE=1; P3 gains only exist
# when both are present (otherwise L3 eviction silently disables prefix-kv and every
# follow-up turn recomputes the full prefill).
VLLM_L3_PREFIX_REUSE=1 ./vllm_kestrel --serve --port 8080 --model <model-dir> --auto-load \
    --wmode q4 --sparse-attn --sparse-k 32 --spec --spec-k 4 \
    --l3-evict --l3-ratio 0.75 --l3-min-seq 128 \
    --disk-kv <kv-dir> --threads 8

# Large-model memory-saving tier: per-layer residency (plaintext VQF only)
VLLM_VQF_STREAM=1 ./vllm_kestrel --serve --port 8080 --model <model-dir> --auto-load --wmode q4
```

### 4) Verify readiness

```bash
# Health check / model list
curl http://<board>:8080/health
curl http://<board>:8080/v1/models

# Chat completion (OpenAI-compatible)
curl http://<board>:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-vl","messages":[{"role":"user","content":"hello"}],"max_tokens":64}'
```

- Admin console / chat page: `http://<board>:8080/admin/` and `http://<board>:8080/chat/`
  (the admin console includes the "Effective optimization combos" ledger: 8 combos that can be
  applied in one click, each annotated with whether it helps for RAM / x86; restart the engine
  after saving a configuration)
- First run on a new board or a new version: run the built-in self-tests (`--test-l3` /
  `--test-sparse` / `--bench-mixed`) first and trust their results.
- Optional NPU acceleration: the default backend is **a zero-third-party direct driver**
  (our own code against the stock rknpu kernel driver). Calibrate on the board first:
  `./vllm_kestrel --npu --npu-selftest --perf-only` (until the register command table is
  calibrated the submission path stays disabled and falls back to CPU automatically).

---

## Conversion Tool (vqf_convert/)

**Since v1 the engine loads VQF v2 only (the in-house format)**, with no in-engine
safetensors/GGUF loading or conversion (v0 supported them — see the v0/v1 table above);
conversion is handled by the shipped standalone tool **`vqf_convert/`**
(safetensors / GGUF → a single mmap-able VQF file).

```bash
# Build the converter (host or board; use build.bat on Windows)
cd vqf_convert && ./build.sh
# Convert (plaintext)
./vqf_conv --model <safetensors-dir> --convert-vqf <out.vqf> --wmode q4
# Encrypt (VQF-Enc: SM4-CTR + HMAC-SM3)
VLLM_VQF_KEY='<pass>' ./vqf_conv --model <safetensors-dir> --convert-vqf <out.vqf> --wmode q4
# Embed an SM2 supply-chain signature (can be combined with encryption)
VLLM_VQF_SIGN_PRIV='<64hex>' ./vqf_conv --model <safetensors-dir> --convert-vqf <out.vqf> --wmode q4
```

---

## Weight Protection

**Producer side** (done by `vqf_convert/`; the two can be combined):

| Defense | Switch (producer side) | Mechanism |
|---|---|---|
| At-rest encryption | `VLLM_VQF_KEY` | VQF-Enc: SM4-CTR + HMAC-SM3 |
| Supply-chain signature | `VLLM_VQF_SIGN_PRIV` (64 hex) | Embedded SM2 signature binding the weight's origin |

**Loader side**: set `VLLM_VQF_KEY` to decrypt and `VLLM_VQF_SIGN_PUB` to verify; a wrong
passphrase or tampered data bytes are rejected (the load log shows
`decrypted (SM4-CTR, HMAC-SM3 ok)` + `SM2 verify ok`).

> Limits: encrypted / signed files must stay **fully resident** (`VLLM_VQF_STREAM` explicitly
> rejects encrypted VQF). Both the full path and the `--stream` path support encryption /
> signing; in plaintext the two paths produce **byte-identical** output (verified by sha256).
> The verifiable-inference key directory must live on a **file system with POSIX permission
> semantics** (ext4/f2fs, etc.) — on vfat/exfat `chmod` has no effect, the private key becomes
> world-readable, and the startup log prints `[ATTEST] WARN`.

---

## Inference Attestation

Set `VLLM_ATTEST=1` (optionally `VLLM_ATTEST_DIR=<key dir>`) to enable **per-response
attestation**:

- On startup the engine generates / loads an SM2 key pair, and `GET /v1/attest` publishes the
  device public key (`pub`, 128 hex);
- Every response carries an `attest` proof: schema=3, magic `VLLM-AT-3`, **bound to the raw
  request body** (`body_sha = SM3(client's original request body)`) — changing only `top_k` /
  `thinking` also changes the digest;
- Two verification paths: **in-browser self-verification** (the chat and admin pages embed
  SM3 + SM2 verification with zero dependencies) and **offline re-verification**
  (`tools/verify_attest.py`, pure Python with zero dependencies, exit code 0=PASS / 1=FAIL,
  including a `--selftest`).

> Limits: a proof shows that "this device produced it and the content was not tampered with",
> and its trustworthiness rests on how well the device private key is protected (see "Weight
> Protection" above). Full specification:
> [docs/权重保护与可验证推理方案.md](docs/权重保护与可验证推理方案.md).

---

## Models and Reproduction

> **Model support (honest disclosure)**: this engine is built and validated for **two Qwen3
> architecture families**:
>
> - **Qwen3-VL series (2B / 8B)** — text plus image/video multimodal; the tokenizer, mrope and
>   the DeepStack vision tower are specialized implementations for this architecture.
> - **Qwen3-MoE series (e.g. Qwen3-30B-A3B, text)** — routing and per-expert FFN are
>   integrated (accelerations such as `--moe-batch` / `VLLM_ACTQ`; see the "MoE model serving"
>   combo in the admin console).
>
> **Other architectures (Llama, older Qwen / Qwen2 text models, etc.) are neither adapted nor
> validated**: conversion may fail or produce unusable output — do not treat this as a
> general-purpose inference engine.
>
> **Models and measurement points behind the figures**: v1.0 (2026-09-12) covers
> Qwen3-VL-2B / Qwen3-VL-8B / Qwen3-30B-A3B (all on RK3588, per-layer vs full residency ×
> combos ⑧/①/①+⑤); v0 (2026-09-05, see Appendix A) covers Qwen3-VL-2B on the board and
> Qwen3-VL-8B on an x86 baseline machine and **must not be used for v1 conclusions**.

- Model weights are not distributed with this repository. Qwen weights follow their original
  open licences (Qwen community licence); after downloading, convert them to VQF with
  `vqf_convert/` and load them (see "Conversion Tool" above).
- Reproduction methods, corpora and driver locations for the benchmark data are in the appendix
  of [docs/RK3588_性能基准报告.md](docs/RK3588_性能基准报告.md) (**v0 measurements**); the
  reproduction basis for v1 measurements is under "Performance" above.
- **One-command v1 reproduction**: `sh tools/bench_value.sh` — produces the per-layer vs full
  A/B memory comparison, the warm-state 5-sample stability and the page-cache evidence
  (parameters are overridable via environment variables, see the script header; its only
  dependency is the Python 3 standard library).

---

## Size and Dependencies (small but complete)

| Item | Value / basis |
|---|---|
| Executable | `vllm_kestrel` ≈ **0.8 MB** (RK3588 Release, `-O2 -s`, board-measured 818,872 B, v0 measurement); `-DVLLM_STATIC=ON` fully static ≈ 1.5 MB, `ldd` reports zero .so dependencies |
| Source | **28 C files** (main + core 11 + common 2 + serve 6 + model 6 + npu 2), C11, one project one artifact |
| Runtime dependencies | **None** — a standard build needs only gcc + libm (`-fopenmp` is used only for the NPU pack parallel region via libgomp, which ships with gcc; fully static builds inline it too, zero .so) |
| Third-party code inside | Only `stb_image.h` (MIT, Sean Barrett) and the llama.cpp-derived 4x4 asm kernel (MIT, The ggml authors); see [LICENSE](LICENSE) section 3 |
| In-house components | NEON quantized GEMM/GEMV, thread pool `vllm_tp` (replacing OpenMP), SM2/SM3/SM4, the VQF v2 mmap format, the NPU direct driver `/dev/rknpu` |
| Deployment | Single binary + optional `vocab.bin`, copy and run; VQF mmap cold start **2.0 s** (v0 measurement) |

Side-by-side comparison (**v0 measurements**, not re-measured on v1): cold start 2.0 s vs
llama.cpp 5.0 s, peak RSS 2.47 GB vs 3.03 GB, plus long-context decode and KV restore
advantages; raw data and definitions are in Appendix A and
[docs/RK3588_性能基准报告.md](docs/RK3588_性能基准报告.md).
> Honest boundary: the binary size covers this engine alone (llama.cpp dynamic/static build
> bases differ and were not compared on equal footing), so no cross-framework size comparison
> is claimed.

### Zero third-party dependencies (self-proof checklist)

"Zero third-party dependencies" is not about what was *not written*, but about the fact that
**none of the following common stacks are pulled in**:

| Common dependency | Actual situation in this project |
|---|---|
| Inference-framework runtime (Python / PyTorch / vLLM stack) | None — a single C11 executable is the complete HTTP service |
| GPU compute stack (CUDA / ROCm) | None — pure CPU with NEON quantized kernels; the NPU is driven directly through the OS kernel's public UAPI |
| Third-party inference/matrix libraries (ggml, OpenBLAS, oneDNN…) | None — GEMM/GEMV are hand-written (the llama.cpp-derived 4x4 asm is an MIT extract, credited in its file header; see LICENSE section 3) |
| Cryptography libraries (OpenSSL / GmSSL / MbedTLS…) | None — SM3 / SM4-CTR / HMAC-SM3 / SM2 are all in-house and validated against the national-standard KAT vectors |
| Image/video decoding libraries (OpenCV / FFmpeg…) | None — image decoding uses the single header `stb_image.h` (MIT); the H.264 module is separate and disabled by default |
| Web/HTTP framework and JSON library | None — an in-house select-based HTTP + SSE loop and a minimal JSON parser |
| OpenMP runtime | Core parallelism uses the in-house thread pool `vllm_tp`; `-fopenmp` is only for the NPU direct backend's pack parallelism (libgomp ships with gcc) |

The dependency ceiling in one sentence: **a dynamic build touches only standard toolchain
components (glibc / libgomp); a fully static build (`-DVLLM_STATIC=ON`) reports "not a dynamic
executable" from `ldd` and can be copied to any aarch64 Linux and run directly** — no
third-party project runtime is ever shipped with the engine.

---

## Repository Layout

```
├── CMakeLists.txt             # build (Release / static / NPU direct by default)
├── build_rk3588.sh            # RK3588 build + self-test entry point
├── cmake/toolchain-aarch64-rk3588.cmake   # x86-host cross-compilation toolchain
├── include/  src/             # C11 sources (common/core/media/model/npu/serve)
│   ├── core/                  # inference kernels (NTT/FHE/CKKS/tp/matmul/attention/l3…)
│   ├── model/                 # weight loading (pure VQF v2 mmap), vision, tokenizer
│   ├── serve/                 # HTTP / admin console / batching / attestation
│   └── media/                 # H.264/MP4 decoding (separate module, off by default)
├── tools/                     # in-house tooling
│   ├── gen_embedded_web.py    # HTML → embedded byte-array generator (re-run after page edits)
│   ├── build_vocab_bin.py     # tokenizer.json → vocab.bin (byte-decoding fixed)
│   ├── extract_llama_asm.py   # extract the 4x4 asm GEMM from llama.cpp (MIT, see header)
│   ├── verify_attest.py       # offline attestation verification (zero dependencies)
│   ├── vllm_vqf_sign.c        # VQF SM2 supply-chain signing / key management tool
│   ├── vllm_mgr.py            # engine process supervisor (start/stop/restart/status page)
│   ├── build_x64.ps1          # x86_64 (MinGW) native build script (not a baseline, consistency only)
│   ├── check_x64.ps1          # x86 build + self-test in one command (exit code 0/1)
│   ├── bench_value.sh         # per-layer value bench: A/B memory + warm stability + page-cache evidence (see Performance §4)
│   └── bench_http_probe.py    # zero-dependency streaming HTTP latency probe (TTFT/tpot/peak VmHWM), called by the above
├── vqf_convert/               # standalone conversion tool (safetensors/GGUF → VQF v2)
└── docs/                      # technical docs / benchmark reports / security specs (Chinese)
```

---

## Documentation (docs/, Chinese)

| Document | Contents |
|---|---|
| [docs/技术文档.md](docs/技术文档.md) | Architecture, modules, the VQF weight format, kernels, serving layer, multimodal, context management, bit-exact determinism, NPU, performance, debugging, version history |
| [docs/RK3588_性能基准报告.md](docs/RK3588_性能基准报告.md) | Full vllm_kestrel vs llama.cpp matrix for cold start / long context / KV restore (**v0 measurements, see Appendix A, not applicable to v1**) |
| [docs/优化配置与边界说明.md](docs/优化配置与边界说明.md) | Mechanisms, gains and honest boundaries of each optimization tier (including effective combos and the x86 cross-check basis; **some figures are v0 / warm-cache**) |
| [docs/KV缓存v2-惰性分配与分层驻留方案.md](docs/KV缓存v2-惰性分配与分层驻留方案.md) | KV lazy allocation, L3 tiered residency and coexistence with P3 prefix reuse (L3 eviction + prefix reuse) |
| [docs/权重保护与可验证推理方案.md](docs/权重保护与可验证推理方案.md) | Three lines of defense: VQF at-rest encryption, SM2 supply-chain signature, inference attestation (schema 3, bound to the raw request body) — relationships, end-to-end usage and the unified security boundary |

---

## v1.0 Change Summary (vs v0)

**Engineering and shape**

- **Pure-VQF runtime**: the engine loads only the VQF v2 single file (mmap-mounted); in-engine
  GGUF / safetensors loading and conversion were removed and moved to the shipped standalone
  tool **`vqf_convert/`**.
- **Source slimming**: `vllm_gguf.c/.h`, `convert.html` and other leftovers were removed along
  with the pure-VQF runtime; the single artifact is still about 0.8 MB; `CMakeLists.txt` was
  bumped to `VERSION 1.0.0`.
- **x86_64 branch shipped**: `vllm_platform.h` provides an x86-64 (MinGW/MSVC) portability
  layer; `tools/build_x64.ps1` / `tools/check_x64.ps1` were added (parameterized, overridable
  via `VLLM_GCC` / `VLLM_X64_OUTDIR`). **x86 is for functional self-test and bit-exactness
  comparison only, never a performance baseline.**

**Performance and memory**

- **Per-layer residency made deliverable** (`VLLM_VQF_STREAM=1` tiered residency + KV v2 lazy
  allocation) — measured resident weight RSS is **4.6× / 8.9× / 23.2×** smaller
  (2B / 8B / 30B-A3B), letting a 17.66 GB 30B-A3B serve on a 16 GB board; TOKIDS stay
  **bit-identical** to the full tier (semantics unchanged). Definitions and the full comparison
  are under "Performance".
- **P3: L3 eviction × prefix reuse coexisting** (`--l3-evict` + `VLLM_L3_PREFIX_REUSE=1`) —
  follow-up-turn prefill measured at **−93%~−97%** (v0-era basis, warm-cache steady state) and
  re-measured on 2026-09-12 on the same basis as a full matrix at **−80% ~ −96%**
  (2B/8B/30B-A3B × full/per-layer, 300-token context), the single largest win in v1.0; adding
  MoE combo ⑤ on the 30B-A3B brings follow-up prefill down to **3.2~5.3 s** (−99.3% vs baseline).
- **Compact L3 layout**: `disk_off` (`wcursor`) is allocated contiguously in eviction order, and
  file / RAM mirror sizes track the real payload rather than the whole KV window.
- **P1/P2 memory paging**: mirror reuse across turns + `MADV_DONTNEED`; arenaization plus
  `imp_sum` sharing within a layer.
- **Admin-console ledger**: `/admin/` gained the "Effective optimization combos" table (8 combos,
  one-click apply) with each optimization annotated as effective or not **for RAM / x86**; the
  P3 switch and the interlock warning for `--l3-evict` without the P3 gate were completed.

**Security and correctness**

- **attestation schema 3**: adds **raw-request binding** (`body_sha = SM3(client's original
  request body)`), so changing only `top_k` / `thinking` also changes the digest.
- **Converter supports encrypted / signed output**: setting `VLLM_VQF_KEY` makes `vqf_convert/`
  emit a VQF-Enc file (SM4-CTR + HMAC-SM3), and `VLLM_VQF_SIGN_PRIV` embeds an SM2 supply-chain
  signature; the two can be combined. Both the full and `--stream` paths are supported. On the
  engine side the load log shows `decrypted (SM4-CTR, HMAC-SM3 ok)` + `SM2 verify ok`, and both
  a wrong passphrase and tampered data bytes are rejected.
- **VQF offline re-sign fix**: the offline signing tool now sets `VQF_FLAG_SIGNED` before
  computing the digest (matching the writer side), fixing the permanent digest mismatch after
  re-signing.
- **Debug build fix**: `CMAKE_C_FLAGS_DEBUG` now includes `-march`, so NEON dotprod inlining no
  longer fails to compile in Debug builds.
- **Comment encoding fix**: corrected mojibake in several historical source comments
  (`vqf_convert/src/conv_main.c`, `src/serve/vllm_server.c`, `src/serve/vllm_batch.c`).

---

## Appendix A: v0 (Historical) Measurements (2026-09-05, **not applicable to v1**)

> **Warning: all data below was measured on v0 and not re-measured on v1; it must not be used
> for any v1 conclusion.** v1 removed GGUF / safetensors loading and in-engine conversion
> (it only accepts the in-house VQF v2 single file) and introduced **per-layer residency** and
> **KV v2**, so **the definitions behind cold start, resident memory, long context and
> multi-turn prefill have all changed**. This appendix is a version-history reference only;
> v1 performance claims always follow "Performance" (2026-09-12 measurements) in the main body.

Platform: Orange Pi 5 Plus (RK3588, 8 cores, 15 GB RAM, eMMC, no GPU/NPU involved).
Model: Qwen3-VL-2B-Instruct (vllm_kestrel VQF q4 fully-optimized tier vs llama.cpp GGUF Q4_0).
Full methodology, definitions and raw data:
[docs/RK3588_性能基准报告.md](docs/RK3588_性能基准报告.md).

### A.1 Cold start
| Engine | spawn→HTTP ready | Peak RSS (VmHWM) |
|---|---|---|
| vllm_kestrel (VQF mmap) | **2.01 s** | ~2465 MB |
| llama.cpp (GGUF mmap) | 5.02 s | ~3027 MB |

### A.2 Long context (content tokens 1K/2K/4K/8K, 64 tokens generated)
| Tier | vllm decode (TPOT) | llama decode (TPOT) | decode ratio |
|---|---|---|---|
| 1K | 73.4 ms | 102 ms | vllm 1.4× |
| 2K | 84.6 ms | 126 ms | vllm 1.5× |
| 4K | 102 ms | 218 ms | vllm 2.1× |
| **8K** | **136.7 ms** | **411.6 ms** | **vllm 3.0×** |

- prefill: llama is 1.4–1.6× faster from 1K–4K (ggml's NEON Q4 is more mature), **evening out
  at 8K** (vllm 40.0 vs llama 38.3 tok/s); vllm's prefill throughput does not drop from 4K to
  8K, while llama loses 20%+ per step (the O(n·k) effect of sparse attention).
- decode scaling with context is an architectural difference: llama with f16-KV scans all 8K
  per token and degrades 3.3× from 2K to 8K; vllm with q8-KV plus sparse decode degrades only
  1.6×.

### A.3 KV cache restore (in-process prefix reuse vs cross-process disk restore)
- Second in-process turn (29-token incremental prefill, 8K context): **vllm 2.0 s vs llama
  6.63 s (3.3×)**.
- Cross-process disk KV restore (`--disk-kv`, 8K session, 1.87 GB F32 snapshot, 15.4 s restore
  after restart vs 202 s full prefill): **vllm 13.1×**; llama.cpp has no equivalent (state is
  lost on restart).

> Honest basis: this report is a single-engine serial measurement on the board; the weights are
> from the same source (Qwen3-VL-2B safetensors) but the quantization grids and operators differ
> (not bit-identical weights), so the numbers are side-by-side presentations of each engine's
> own fields. Reproduction methods and data files are in the report appendix.

---

## Licence and Compliance

- **Licence: dual licensing (AGPL-3.0-or-later OR commercial, your choice)** — this project is
  **free software**: you may use, modify and distribute it under the **GNU Affero General Public
  License v3.0 or later** (SPDX: `AGPL-3.0-or-later`). **If you cannot or do not wish to carry the
  AGPL source-disclosure obligation** (e.g. embedding in a closed-source product, or offering a
  closed-source service / SaaS), you **must obtain a commercial licence** first. Full terms are in
  [LICENSE](LICENSE); contribution rules are in [CONTRIBUTING.md](CONTRIBUTING.md).
- Third-party components are retained under their own licences: `stb_image.h` (MIT, Sean
  Barrett) and the 4x4 asm GEMM file extracted from llama.cpp plus its derived kernels (MIT,
  The ggml authors) — copyright and licence text are in the respective file headers, detailed
  in [LICENSE](LICENSE) section 3. Both are compatible with AGPL.
- **Please report security vulnerabilities privately** (do not open a public issue); see
  [SECURITY.md](SECURITY.md) for the process and our response commitments.
- The NPU direct backend of this engine interacts only with the OS kernel driver's (stock
  rknpu) public UAPI and contains no closed-source library or third-party header.

## Contact

Commercial licensing / research collaboration / reproduction data: **398152090@qq.com**
(also via [Issues](https://gitee.com/pei-xiaoguang/kestrel-llm/issues) or Gitee direct message;
see [LICENSE](LICENSE) for the commercial agreement and dual-licensing terms)

For commercial licensing / research collaborations / data requests:
**398152090@qq.com**, or open an issue at https://gitee.com/pei-xiaoguang/kestrel-llm/issues.
