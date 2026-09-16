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
| x86-64 | none | Shipped x86-64 portability layer (`tools/build/build_x64.ps1` / `tools/build/check_x64.ps1`), **for functional self-test and bit-exactness comparison only — never a performance baseline** |
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
| **Per-layer residency** (new in v1) | Resident memory decoupled from model size: **8B resident weights 4.19 GB → 0.48 GB (8.7×)**, serve peak **0.82 GB**; warm, remotely re-measured **TTFT 1.73 s / tpot 330 ms** (median of 5 identical requests, see "Performance" §4) |
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

**The recommended tier is 8B** (with the 2026-09-15 RK3588 remote re-test as the current public
baseline, see "Performance" §4): resident weights 4.19 GB → **0.48 GB (8.7×)**, serve peak
**0.82 GB**, warm **TTFT median 1.734 s / tpot median 330.2 ms** (range `1.711-1.796 s` /
`328.9-330.6 ms`, about 24.5 s end-to-end for 70 tokens), with a **tpot spread of about ±0.26% over
5 samples**. Its 6.15 GiB of
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

Numbers in this chapter come from two sources, and each subsection heading states which one it is:

1. **Controlled A/B (reproducible to within 1%)** -- the `--stream-test` basis. On the same board, with the same model file, the **old build** (compiled here from the previous Gitee source commit `ffd0b92`) and the **new build** (this release's source) were run **interleaved within a single session**, 2 rounds per configuration, with temperature and per-core frequency sampled at every point. **Sections 1 and 4 belong here.**
2. **New points (single-shot, not controlled)** -- serve-style bases (sections 2 / 3 / 5). The matching old-build points were not re-measured, because a single old-build 30B serve point takes 15-40 minutes. These tables **describe this release only and are not an old-vs-new comparison**.
3. **Cross-engine comparison (weight-fair A/B, section 7)** -- same board, **the same 4-bit quantized weights**, each side free to use its own optimizations, compared tier by tier against **llama.cpp** (3 rounds each at 2K / 4K, 2 at 8K, 1 at 16K). This is the only "against a third-party engine" basis in this chapter and **must not be mixed with the first two** (this release vs the previous release, or this release against itself).

**Measurement environment**: RK3588 (Orange Pi 5 Plus, 15.9 GB RAM, `governor=performance`). Old build sha256 `76d5bffb8c6fb280...`, new build `078349be598e13ef...`. Temperature stayed within **30.5-47.2 C** and the A76 / A55 cores held **2,304,000 / 1,800,000 kHz** throughout -- **there was no downclocking**, so the old-vs-new differences do not come from temperature or frequency.

**Correctness criterion (bit-identical greedy TOKIDS from `--stream-test`)**:

- 30B-A3B: all **16 points** in section 1 (old/new x full/per-layer x 2 rounds) emitted 32 tokens with the same sequence; first 8 ids = `151667, 198, 99692, 3837, 20002, 104029, 11622, 104811`.
- 8B: all **24 points** emitted 29 tokens with the same sequence; first 4 ids = `100062, 99371, 58814, 113272`.

> ### Why the 2026-09-12 figures were retired
>
> Those numbers came from a **different binary** (sha256 `e1484740...a8f8e8`) and a **different session**. They **cannot be reproduced** by the old build compiled here today from the same source commit:
>
> | Historical record | Old build measured today | Deviation |
> |---|---|---|
> | 8B full t8 warm decode 431 ms/tok | 468 ms/tok | +8.6% |
> | 8B per-layer t4 warm decode 370 ms/tok | 319 ms/tok | -14% |
> | 8B P3 prefill (t2), section 3: 1,885 ms | 2,730 ms (after clearing L3) | +45% |
> | 30B full t8 cold decode 2,442 ms/tok | 2,020 ms/tok | -17% |
>
> The cause is not code but "different binary + different session state (especially L3 / page cache)". The historical figures are therefore **retired wholesale** and are no longer cited; this is also why the chapter was rebuilt as an interleaved same-session A/B.

### Summary

| Model | Progress vs the previous release | Memory |
|---|---|---|
| **30B-A3B (MoE, q4, 17.66 GB)** | **decode 1.74-3.72x faster; prefill 1.21x (cold, better I/O overlap) to 5.44x (warm)** | unchanged (+0.1-1.0%) |
| **8B (dense, q4, 6.60 GB)** | **no change** -- 24 points agree point by point, max deviation 1.0% | unchanged |
| **8B vs llama.cpp (long context, section 7)** | **prefill: 1.32x slower at 2K -> 1.15x slower at 4K -> 1.11x faster at 8K -> 1.60x faster at 16K; decode: full 1.07x faster at 2K -> 3.63x faster at 16K; multi-turn reuse turns: this release 1.33-3.4x faster across the board (lead grows with context)** | peak at 16K 6.71-10.15 GB vs llama.cpp 10.77 GB (at 2K: 1.46-4.90 GB vs 8.71 GB) |

---

### 1) Per-layer vs full residency (`--stream-test`, **controlled A/B**)

`VLLM_VQF_STREAM=1` puts the engine into **tiered (per-layer) residency** mode: the VQF single file stays mmap-mounted, but each layer's weights establish file pages only while that layer computes and release them immediately with `MADV_DONTNEED`, keeping only the `keep` layers (1 by default) resident. Together with KV v2 lazy allocation, **resident memory is decoupled from model size and layer count**: a 17.66 GB Qwen3-30B-A3B-q4 can be served on a 16 GB board.

**30B-A3B (MoE, q4, 17.66 GB; `/mnt/VQF/qwen3-30B-A3B-q4/model.vqf`, on the SanDisk microSD card)**

| Residency | Threads / cache | Old prefill | New prefill | Ratio | Old decode | New decode | Ratio |
|---|---|---|---|---|---|---|---|
| full | t8 / cold | 259.86 s | **213.70 s** | 1.22x | 2,020 ms/tok | **986 ms/tok** | **2.05x** |
| full | t4 / warm | 36.28 s | **6.67 s** | **5.44x** | 1,205 ms/tok | **324 ms/tok** | **3.72x** |
| per-layer | t8 / cold | 260.54 s | **215.15 s** | 1.21x | 2,412 ms/tok | **1,387 ms/tok** | **1.74x** |
| per-layer | t4 / warm | 36.75 s | **6.93 s** | **5.30x** | 1,583 ms/tok | **564 ms/tok** | **2.81x** |

Peak VmHWM: full 12,705,354 -> **12,722,536 kB (+0.14%)**; per-layer 708,162 -> **715,054 kB (+0.97%)**. Memory is **unchanged** between the two builds; per-layer saves **17.8x** versus full (12,722,536 / 715,054).

> **The 1.21-1.22x on cold prefill is an indirect gain**: that basis is dominated by reading 17.66 GB from the SD card, with an effective read rate of 68.0 MB/s (old) versus 82.6 MB/s (new). The file is identical and the only variable is the CPU -- the old build's slow unpacking stalled the read pipeline, while the new build overlaps computation with I/O far better. **Do not read this as "the kernel accelerated the disk read".**

> **The 30B decode ratio depends on page-cache state -- read it as a range**: the model is 17.66 GB against 15.9 GB of RAM. The new build's decode was measured between **324 and 1,204 ms/tok**: about 324 ms/tok when the active experts are cached (full residency, the table above) and about **570-580 ms/tok** for per-layer residency (three interleaved runs: 571 / 576 / 581), but 1.6-2.5 s spikes once they are evicted (the same binary and the same config differed by 3.7x across two rounds; per-token detail 321 / 321 / 2511 / 1069 / ...). The old build's 1,205 ms/tok is, by contrast, **uniform** (compute-bound, with I/O fully hidden). So the 3.72x for decode is a **best-case-cache** figure; worst case it is roughly a wash.

> **Caliber alignment (with the project docs)**: both sides of the table above run the **exact track** (no `VLLM_ACTQ`), which is what preserves bit-exactness. The 5.44x prefill here is the "exact-track GroupGEMM 6.55x" of the project document (Distributed Expert Extraction plan, section 9.12), and the 2.81x decode is its "down row-group parallelism 2.88x". **The separate "3-3.4 s / about 200 ms" figures in that document are the approximate track (`VLLM_ACTQ=1`, int8 SDOT activations, greedy divergence from about token 24 on long text) and must not be mixed with this table.** Reference magnitudes (board 30B, per-layer, t4, warm, 32-token prefill): exact track per-token about 37-60 s; exact track with batching about 7-9 s (the new build's 6.7 s is this tier); `ACTQ=1` with batching about 2.7-3.4 s. In other words this release takes the **exact** track from about 37 s to about 7 s (5.3x), making bit-exact inference genuinely usable; but it is still **2.3x** away from the approximate track (about 3.0 s) and has not caught up. **Note also that under the recommended configuration (`VLLM_ACTQ=1` plus batching, i.e. combo 5) the two builds are essentially level -- across 2 interleaved rounds: prefill old 3.022/3.027 s vs new 3.078/2.994 s (1.00x), decode old 442/446 ms vs new 430/433 ms (1.03x). The value of this release is that the exact track is 5.3x faster, not that the recommended configuration is faster.**

**8B (Qwen3-VL-8B, q4, 6.60 GB; `/mnt/VQF/8b/qwen3vl8b.q4.vqf`, on the SanDisk microSD card)**

| Residency | Threads / cache | Old prefill | New prefill | Old decode | New decode |
|---|---|---|---|---|---|
| full | t8 / cold | 71.10 s | 71.08 s | 470 ms/tok | 469 ms/tok |
| full | t4 / warm | 1.594 s | 1.605 s | 185 | 185 |
| full | t8 / warm | 2.127 s | 2.135 s | 468 | 468 |
| per-layer | t8 / cold | 71.05 s | 71.03 s | 590 | 588 |
| per-layer | t4 / warm | 1.707 s | 1.724 s | 320 | 318 |
| per-layer | t8 / warm | 2.220 s | 2.209 s | 581 | 582 |

**Everything is within 1.0% -- 8B is completely unchanged in this release** (see section 8). Peak VmHWM: full about 4,226.7 MB, per-layer about 593.2 MB (identical for both builds); per-layer saves **7.12x** versus full.

> Basis: both models run `--stream-n 32`, but 8B reaches EOS early and **actually emits 29 tokens** (30B emits 32). "Cold" = `sync; echo 3 > /proc/sys/vm/drop_caches` before each point; "warm" = no `drop_caches`, and each warm point is preceded by a discarded warm-up run (`--stream-n 16`) so both builds start from the same cache state.

**Addendum: 30B-A3B "ACTQ track x residency" 2x2 (same-version characterization, **controlled A/B**)**
> The main table above answers "how much did the new build gain". This addendum answers a different question: the trade-off between two orthogonal dimensions **inside one binary** -- `VLLM_ACTQ` (exact track / approximate track) and `VLLM_VQF_STREAM` (full-layer / per-layer). It uses the same basis as this section (`--stream-n 32 --threads 4`, with a discarded `--stream-n 16` warm-up before each point) and runs **the 4 configurations interleaved within a round**, 2 rounds.

| Track | Residency | prefill (pure compute) | prefill (end to end) | GATEUP | decode (steady) | Peak VmHWM | TOKIDS |
|---|---|---|---|---|---|---|---|
| exact | full | 6,528 / 6,530 ms | 6.670 / 6.658 s | 6,240 ms | **325 ms/tok** | 12,286 / 12,278 MB | `41eae062e4ee` |
| exact | per-layer | 6,514 / 6,539 ms | 6.946 / 6.962 s | 6,212 / 6,230 ms | **572 / 573 ms/tok** | **695 MB** | `41eae062e4ee` |
| ACTQ | full | 2,644 / 2,645 ms | 2.762 / 2.781 s | 2,379 ms | **197 ms/tok** | 12,354 / 12,346 MB | `1fd42e485639` |
| ACTQ | per-layer | 2,665 / 2,654 ms | 3.057 / 3.032 s | 2,396 / 2,385 ms | **428 / 430 ms/tok** | **695 / 691 MB** | `1fd42e485639` |

(One value per round. "Pure compute" is the engine's `[PREFILL-TIMING] total`; "end to end" is `[STREAM] prefill ... in`.)

**Three quotable conclusions**

- **ACTQ track**: prefill pure compute is **2.47x** faster (full) / **2.45x** (per-layer); end to end 2.40x / 2.28x. Decode is **1.65x** faster (full, 325 -> 197 ms/tok) / **1.34x** (per-layer, 572 -> 428).
- **Per-layer residency costs almost no compute**: prefill pure compute **ties** with full-layer (exact 6,514/6,528 = 1.000x; ACTQ 1.006x); end to end is only 4.4% / 9.9% higher (that is per-layer page establishment and scheduling). **The cost sits almost entirely in decode**: exact +76% (325 -> 572), ACTQ +117% (197 -> 430).
- **Peak memory**: full 12,278-12,354 MB -> per-layer 691-695 MB, i.e. **17.7x**.

**Correctness (this dataset also yields the criterion for a "track")**

All 4 exact-track runs (2 rounds x 2 residencies) produce identical TOKIDS (`41eae062e4ee`), all 4 ACTQ-track runs are likewise identical to each other (`1fd42e485639`), but the **two tracks differ from each other**. So: **residency does not change the numbers (per-layer is bit-identical to full-layer); only switching the track does** (ACTQ is an approximate track). Exact-track logs contain no `[ACTQ]` / `[ACTQ16]` line; every ACTQ-track run prints `[ACTQ] VLLM_ACTQ=1: q4 MoE int8-dot approximate track ON`.

**Cross-check against the table above (two independent sessions)**: this addendum's exact-track full t4 warm prefill is **6.670 / 6.658 s** and per-layer is **6.946 / 6.962 s**, matching this section's new-build figures of **6.67 s / 6.93 s**; decode 325 ms/tok matches **324 ms/tok**. The ACTQ-track full-layer 2.76-2.78 s also falls inside the "`ACTQ=1` plus batching about 2.7-3.4 s" band stated above. The numbers are therefore usable as a baseline.

> **Two reading rules**
>
> 1. **The round-1 "full" decode averages (445 / 275 ms/tok) must not be quoted.** The per-token detail shows `r1_exact_full` steady at 322-325 ms for the first 16 tokens, then jumping to 355 -> 907 -> 883 -> 859 -> 607 -> ...; `r1_actq_full` steady at 195-203 ms for the first 23 tokens, then jumping at t=24 to 675 / 346 / 644 / 605 / .... The round-2 runs of the same configurations are steady across all 32 tokens (323-328 / 196-199, coefficient of variation 0.4%), and both per-layer points are steady too (0.7-1.5%). Round-1 full-layer decode is therefore judged **contaminated by external interference (page-cache reclaim / I/O)**; this table takes decode from round 2 only. Prefill is unaffected (0.02-0.8% spread across rounds, directly quotable).
> 2. The gap between "end to end" and "pure compute" prefill (about 140 ms exact, about 430 ms per-layer) comes from process startup and per-layer page establishment and is **not part of the compute basis**, which is why both numbers are given.

> Environment: `governor=performance` throughout; temperature 30.5-45.3 C; all 8 cores at constant frequency (A55 1,800,000 kHz / A76 2,304,000 kHz); page cache steady at 15.3-15.5 GB; logs confirm every point took the batched path (`[batch]`).

---

### 2) Steady-state basis (serve, **new points, single-shot**)

One request = 292-token context + 32 generated tokens (greedy, `--threads 8`). Cold = first request after `drop_caches`; warm = the same request sent immediately afterwards.

| Model | Residency | Peak VmHWM | Cold prefill | Warm prefill | Warm TTFT | Warm tpot |
|---|---|---|---|---|---|---|
| 8B | full | 4,351,464 kB | 84,044 ms | 15,292 ms | 15,293 ms | 499.2 ms |
| 8B | **per-layer** | **741,216 kB** | 84,294 ms | 15,700 ms | 15,700 ms | 624.7 ms |
| 30B-A3B | full | 14,208,808 kB | 272,175 ms | 76,673 ms | 76,674 ms | 554.3 ms |
| 30B-A3B | **per-layer** | **1,116,948 kB** | 274,256 ms | 78,275 ms | 78,276 ms | 875.6 ms |

> **The "warm" 30B row is only partially cached**: the model is 17.66 GB against 15.9 GB of board RAM, so it can never fill the page cache, and the figures depend on what ran before it. Measured within a single session, the 30B per-layer baseline showed t1/t2/t3 = 60.6 / 68.1 / 77.9 s, i.e. **monotonically slower** -- exactly this effect. Treat the 30B serve figures as **orders of magnitude only, not a precise baseline**. 8B fits in RAM and has no such problem.

---

### 3) Stacking with the "effective optimization combos" (**new points, single-shot**)

Combo 1 = `--sparse-attn --sparse-k 32 --l3-evict --l3-ratio 0.75 --l3-min-seq 128` plus the environment variable `VLLM_L3_PREFIX_REUSE=1`; combo 5 = `VLLM_ACTQ=1 VLLM_MOE_BATCH=1` (meaningful for MoE / q4 only).

**8B (measured after clearing the L3 directory each time; two rounds reported)**

| Residency | Config | t2 prefill | t3 prefill | vs baseline |
|---|---|---|---|---|
| full | baseline (`--no-prefix-kv`) | 15,086 ms | 17,423 ms | -- |
| full | combo 1 P3 | **2,907 / 2,930 ms** | **2,732 / 2,742 ms** | **-81%** |
| per-layer | baseline | 15,376 ms | 17,829 ms | -- |
| per-layer | combo 1 P3 | **3,076 / 3,184 ms** | **2,874 / 2,871 ms** | **-80%** |

**30B-A3B**

| Residency | Config | t2 prefill | t3 prefill |
|---|---|---|---|
| full | baseline | 75,947 ms | 81,846 ms |
| full | combo 1 P3 | 8,599 ms | 7,005 ms |
| full | combo 1+5 | **4,382 ms** | **3,692 ms** |
| per-layer | baseline | 68,058 ms | 77,905 ms |
| per-layer | combo 1 P3 | 9,098 ms | 7,443 ms |
| per-layer | combo 1+5 | **4,860 ms** | **4,106 ms** |

- The 8B P3 gain is **-80% to -81%** and **must be read with the L3 directory cleared**: if L3 was already warmed by a previous run you get a markedly more optimistic figure (the 1,885 ms in the historical record is exactly that case).
- On the 30B, combo 5 takes off a further about **1.9x** on top of combo 1 (8,599 -> 4,382 ms); the two axes still stack.

---

### 4) 8B: the recommended tier delivers (with the 2026-09-15 RK3588 remote re-test as the current public baseline)

**Thread A/B (warm page cache, controlled A/B, 2 rounds each)**

| Tier | `--threads 4` (old / new) | `--threads 8` (old / new) | Conclusion |
|---|---|---|---|
| full decode | 185 / 185 ms/tok | 468 / 468 ms/tok | **4 threads are 2.53x faster** |
| per-layer decode | 320 / 318 ms/tok | 581 / 582 ms/tok | **4 threads are 1.82x faster** |

Both builds agree at each tier. The RK3588 has 4x A76 + 4x A55, and `--threads 8` pulls the four A55 little cores into the GEMM parallel region, making **decode 1.8-2.5x slower**. **Always use `--threads 4` for 8B.**

**Per-layer serve + short request (`--no-think`, `max_tokens=96`, naturally ends at 70 tokens; remote re-test on the Gitee tree)**

| Tier | TTFT | Total | tpot | Peak VmHWM |
|---|---|---|---|---|
| cold (first request after `drop_caches`) | 72.613 s | 95.505 s | 331.8 ms | 612,868 kB |
| **warm (5 consecutive identical requests)** | **median 1.734 s** (1.711-1.796) | **about 24.5 s** (24.408-24.610) | **median 330.2 ms** (328.9-330.6) | 824,152 kB |

Across the 5 warm samples the **tpot spread is only 0.5%** and the TTFT spread about **4.9%**; these
figures come from the same remote board, the same binary and 5 consecutive identical requests, so
they serve as the current public baseline for the recommended tier.

**Why 8B is the recommended tier (three points, all reproducible)**:

1. **It collects the full memory saving**: under `--stream-test` with a cold page cache, resident
   weights drop from 4.19 GB to **0.48 GB (8.7×)**, with a serve per-layer peak of **0.82 GB**.
2. **Warm-state stability is physically guaranteed**: the 6.15 GiB of weights **fit in the 15.9 GB
   page cache**, so warm behaviour is not a gamble; across 5 consecutive identical requests in the
   remote re-test, tpot stayed within **328.9-330.6 ms**.
3. **The one-off cold-start cost is low**: cold-state first-request TTFT is **72.613 s**, because the
   cost scales with weight size divided by storage bandwidth.

**One-command reproduction**: `sh tools/bench/bench_value.sh` (parameters are overridable via environment variables; see the header comment in the script).

---

### 5) 30B-A3B on a short request (**new points, single-shot**)

Config: per-layer `VLLM_VQF_STREAM=1` + combo 1 + combo 5, `OMP_NUM_THREADS=4 VLLM_THREADS=4`. Request = 22-token context + 32 generated tokens (greedy, streaming). With only 22 tokens of context the log shows `[L3] skipped: seq=22 < l3-min-seq=128`, so **L3 and prefix reuse never engage**; this table therefore reflects the "per-layer + MoE tier".

| Tier | TTFT | Total | tpot | Peak VmHWM |
|---|---|---|---|---|
| cold (first request after `drop_caches`) | 202.33 s | 236.81 s | 1,110.6 ms | 664,484 kB |
| **warm (same request sent immediately again)** | **2.483 s** | **16.17 s** | **439.4 ms** | 919,012 kB |

**Key precondition: `enable_thinking` defaults to on** (matching HF `apply_chat_template`). The 32 tokens above **all fall inside the reasoning block and produce no answer**, so the switch was measured as well:

| Tier (`max_tokens=256`) | TTFT | Total | Actual output | Result |
|---|---|---|---|---|
| thinking **on** (default) | 2.417 s | 134.45 s | **256 (budget exhausted)** | text still starts with `<think>`, **no answer** |
| **thinking off** (`"enable_thinking": false`) | **2.437 s** | **20.82 s** | **40 (natural EOS)** | **complete answer** |

**Conclusion: on this 16 GB board the usable 30B-A3B tier is thinking off + short context.** With thinking off a complete short answer arrives in 20.8 s at a 0.92 GB peak; with the default (thinking on) 256 tokens do not finish the reasoning block, so a single exchange must budget 300+ tokens. The 202 s cold figure is the one-off cost of reading all 17.66 GB of weights from the SD card.

---

### 6) Storage media (same board, `dd iflag=direct`, 1 GiB, bypassing the page cache)

| Media | Use / mount | Device | Measured sequential read |
|---|---|---|---|
| **SanDisk microSD** | 8B / 30B weights (`/mnt/VQF`) | `/dev/mmcblk1` | **64.5 MB/s** |
| eMMC | 2B weights / L3 directory (`/mnt/emmc`) | `/dev/mmcblk0` | **267 MB/s** |

The time in "trade time for memory" scales with weight size divided by storage bandwidth: the 30B cold figure of about 202-215 s corresponds to the SD card's 64.5 MB/s; moving the weights to eMMC (267 MB/s) would, by simple bandwidth ratio, bring it to roughly the 50 s range -- **that is an extrapolation, not a measurement**.

---

### 7) Long context: weight-fair A/B against llama.cpp (full re-test on **2026-09-16**)

**This is the only "against a third-party engine" basis in this chapter** and must not be mixed with sections 1-6 (this release vs the previous release, or this release against itself).

**What changed since the previous round (read before citing)**: the 2026-09-15 round used `md5_exe=71f8f53e...`; this round uses the fixed build `md5_exe=a1b6707d...` (tokenizer vocab index + L3 whole-block write + NEON restore kernels). **Both sides of all four tiers were re-measured in the same session in this round** (still interleaved, still `drop_caches` before every arm), so they replace the previous round wholesale. The previous round's figures remain in the 2026-09-15 archive, and its statements such as "the reuse turn is the one item still lost" are **withdrawn** -- see conclusions 4 / 5. Rounds: 3 each at 2K / 4K, 2 at 7.2K, 1 at 16K.

**How fairness is enforced (four points, all verifiable)**

| Aspect | What was done |
|---|---|
| Same weight source, same quant class | Both sides use **the same Qwen3-VL-8B weights**. This release's `q4` VQF is about **4.25 bpw** on the text path (group-16 Q4); llama.cpp used **Q4_0 (4.50 bpw)**, **not Q4_K_M** (the latter is 4.9 bpw with more expensive dequantization and would flatter this release) |
| Weight evidence | This round's three md5s: `md5_exe=a1b6707d...` (previous round `71f8f53e...`), `md5_gguf=96fb8a5e...` (unchanged throughout), `md5_vqf=b6d8d1d7...` (the historical 16K snapshot is `f8ca1002...`; the difference is **only the `max_seq` metadata field in the VQF header** -- the weight body is byte-identical). Note: `kv2_run2.sh` had a defect (missing `mkdir -p /tmp/kv2`, so the md5 lines never reached `summary.txt`); fixed this round, and the three md5s are recorded in the archive's `MANIFEST.txt` |
| Each side runs its own optimizations | This release: `--sparse-attn --sparse-k 32 --l3-evict --l3-ratio 0.75 --l3-min-seq 128 --prefill-batch 256` plus `VLLM_L3_PREFIX_REUSE=1 VLLM_TP_SPIN=1`, in both per-layer and full-residency modes; llama.cpp: `llama-server -t 4 -ngl 0` with its **default prompt cache** (log: `selected slot by LCP similarity`) |
| Same protocol | **4 threads on both sides**; before every arm, `sync; echo 3 > /proc/sys/vm/drop_caches`, then one **throwaway warm-up request**, then the timed turns; **no interleaving** |

**Request shape**: one long Chinese passage (repeated to the target length per tier) plus 3 follow-up questions; turns 2 and 3 append to the prefix, so **prefix reuse always hits**; `temperature=0`, `max_tokens=64`. This round: **3 rounds** each at 2K / 4K and **2 rounds** at 8K (the third was stopped on request); tables use the median of the per-round turn-1 values. 16K was not re-measured this round.

**Raw artifacts**: this round [`docs/bench/20260916-rk3588-llama-ab/`](docs/bench/20260916-rk3588-llama-ab/MANIFEST.txt), previous round [`docs/bench/20260915-rk3588-llama-ab/`](docs/bench/20260915-rk3588-llama-ab/MANIFEST.txt) — the raw source of every number in this section (per-arm per-round `json`, engine logs, `summary.txt`, `REPORT.txt`, plus the board-side scripts that produced them and the md5 evidence list). This section only interprets; **cite figures by checking them against the raw artifacts**.

**prefill (incremental-token basis, t/s, higher is better)**

| Tier (actual prompt tokens) | llama.cpp Q4_0 | This release, per-layer | This release, full | Lead |
|---|---|---|---|---|
| 2.0K (1,996-2,034) | **17.61** | 13.23 | 13.36 | 0.75x (1.32-1.33x slower) |
| 4.0K (3,976-4,014) | **14.61** | 12.68 | 12.71 | 0.87x (1.15x slower) |
| 7.2K (7,216-7,254) | 11.02-11.18 | 12.30-12.38 | **12.31-12.38** | **1.11-1.12x** |
| 16K (15,616-15,654) | 7.25 | 11.61 | **11.68** | **1.60 / 1.61x** |

**decode (t/s, higher is better)**

| Tier | llama.cpp Q4_0 | This release, per-layer | This release, full | Lead |
|---|---|---|---|---|
| 2.0K | **2.61** | 2.28 | 2.79 | full **1.07x** (per-layer 1.14x slower) |
| 4.0K | 1.50 | 2.16 | **2.54** | **1.69 / 1.44x** |
| 7.2K | 0.94 | 1.94-1.97 | **2.27-2.28** | **2.41-2.43 / 2.02-2.10x** |
| 16K | 0.48 | 1.42 | **1.74** | **2.96 / 3.63x** |

**Peak memory (VmHWM, GB; the maximum within each tier)**

| Tier | llama.cpp Q4_0 | This release, per-layer | This release, full |
|---|---|---|---|
| 2.0K | 8.71 | **1.46** | 4.90 |
| 4.0K | 9.00 | **2.22** | 5.66 |
| 7.2K | 9.49 | **3.47** | 6.91 |
| 16K | **10.77** | **6.71** | 10.15 |

**multi-turn reuse turns (prefill latency of turns 2 / 3, lower is better; both sides hit prefix reuse)**

| Tier | llama.cpp | This release, per-layer | This release, full | Lead |
|---|---|---|---|---|
| 2.0K | 4.79-4.92 s | 3.34-3.60 s | **3.28-3.60 s** | **1.33-1.50x** |
| 4.0K | 8.40-8.55 s | 4.10-4.45 s | **4.05-4.35 s** | **1.92-2.11x** |
| 7.2K | 14.43-15.26 s | 5.52-5.76 s | **5.40-5.70 s** | **2.50-2.83x** |
| 16K | 29.52-29.56 s | 8.68-9.23 s | **8.85-9.01 s** | **3.2-3.4x** |

> **Tier and build basis**: 3 rounds each at 2K / 4K, 2 at 7.2K, 1 at 16K; **both sides of every tier were measured in the same 2026-09-16 session** (both per-layer and full residency). The 16K tier ran under **`governor=ondemand`** (the other three under `performance` -- the board was rebooted in between; the governor has since been set back to `performance`), so its absolute values may be ~5-8% conservative; **the within-tier two-sided comparison is unaffected**.

**Five conclusions**

1. **The decode crossover still sits just above 1K**: sparse pruning only kicks in past `seq_len > g_sparse_k * g_sparse_block = 1024`, so below that threshold this release pays full dense attention for nothing. **Full residency** already leads by **1.07x** at 2.0K -> **1.69x** at 4.0K -> **2.41x** at 7.2K -> **3.63x** at 16K; **per-layer residency** is still 1.14x slower at 2.0K but **ahead from 4.0K** (1.44x) -> 2.02-2.10x at 7.2K -> **2.96x** at 16K.
2. **The prefill crossover sits between 4K and 8K**: 1.32-1.33x slower at 2.0K -> 1.15x slower at 4.0K -> **1.11-1.12x** faster at 7.2K -> **1.60-1.61x** faster at 16K.
3. **The real source of the gap is the decay rate**: from 2.0K to 16K, this release loses **12.2%** on prefill (13.23 -> 11.61 per-layer) and **37.7%** on decode (2.28 -> 1.42); llama.cpp loses **58.8%** (17.61 -> 7.25) and **81.6%** (2.61 -> 0.48). llama.cpp decays more steeply (the previous round measured its 16K prefill decaying **within a single request**: 17.72 -> 14.61 -> 12.40 -> 10.79 -> 9.59 -> 8.62 t/s), while this release stays flat.
4. **The multi-turn reuse turn now goes to this release across the board** (the previous round had it 1.10-1.25x behind; **that conclusion is withdrawn**): **1.33-1.50x** at 2.0K, **1.92-2.11x** at 4.0K, **2.50-2.83x** at 7.2K, **3.2-3.4x** at 16K -- the lead grows monotonically with context (at 16K the previous round was still 1.07x behind). Attributable to three fixes in this release, each measured separately:
   - **tokenizer vocab index** (the dominant residual in reuse turns): bit-exact regression 1548 cases / 0 mismatches; with it removed, the unattributed residual drops from **10.38 s -> 0.16 s** at 7.2K and **2.74 s -> 0.03 s** at 2.0K.
   - **L3 eviction whole-block write**: `write` **3.478 s -> 0.062 s (-98.2%)**; whole eviction phase **4.742 s -> 1.271 s (-73%)**.
   - **L3 restore NEON kernels**: `store` **0.665 s -> 0.483 s**, `dequant` **0.179 s -> 0.117 s**, `restore` wall clock **0.848 s -> 0.630 s**.
   - Combined (2.0K, same-session A/B): turn2 TTFT **6.630 s -> 3.845 s (-42%)**, turn3 **6.522 s -> 3.588 s (-45%)**; turn-1 prefill compute differs by 0.2% and turn-2 GEMM by 0.4%, and the output text is identical.
5. **Two attributions in the previous round were wrong; corrected here** (the original artifacts remain untouched in the 2026-09-15 archive):
   - "`[L3] restored 13176 prefix blocks` = **515 MB in about 28 s** (roughly 20 MB/s, decompression-bound)" — **the 515 MB figure itself was right; the error was tying it to 28 s**. 515 MB *is* the on-disk Q4 payload: 13176 blocks x 40 KB ~= 515 MB, consistent with this round's measured slope (1692 blocks = 66.09 MB, 6120 blocks = 239 MB, about 39 KB/block). But **28 s was the order of magnitude of the whole reuse turn, and the L3 restore was nowhere near filling it**: this round measures a restore-stage wall clock of **0.630 s** at 2.0K, extrapolating linearly by block count to about **5-6 s** at 16K (bounded by write bandwidth, not "decompression"), so the "roughly 20 MB/s" derived from 28 s is void as well. The real dominant term in that reuse turn was the tokenizer's linear vocab scan (about 2.7 s at 2.0K, 10.4 s at 7.2K; about 21 s at 16K by token count, consistent with the measured 30.2-32.3 s).
   - "**The one item still lost is the multi-turn reuse turn**" — see conclusion 4; on that basis this release now leads.

**Boundaries (must be quoted together with this section)**

- **Rounds and build basis**: 3 rounds each at 2K / 4K, 2 at 7.2K (the third was stopped on request), 1 at 16K; **all four tiers were measured with this round's build `a1b6707d...`**, both sides interleaved in one session. Per-round spread: prefill <=1.0%, decode <=3.5%, reuse-turn latency <=5.1% (worst case is 7.2K llama turn 3, and that tier has only 2 rounds). **The 16K tier ran under `governor=ondemand`** (the other three under `performance`), so its absolute values may be ~5-8% conservative; both sides of that tier shared the same governor, so the comparison holds.
- **Running 16K requires editing the VQF header's `max_seq` field** (8192 -> 20480, 4 bytes; the engine reads `max_seq` from the VQF header and there is no runtime switch). The field is **metadata** and does not change weight values. This round proved the whole edit end to end: md5 `b6d8d1d7...` before -> `f8ca1002...` after (byte-identical to the previous round's historical 16K snapshot, which confirms only those 4 bytes moved) -> after `--restore`, both the md5 and `max_seq` are back to baseline (`b6d8d1d7...` / 8192).
- **End-to-end turn latency is not comparable**: the two sides stop at different token counts (llama.cpp often emits only 6-17 tokens, this release 40-64). This section therefore uses **per-token rates and prefill latency only** and does not quote end-to-end seconds — note that `turn* total_s` in `summary.txt` includes decode; the "reuse turn latency" here is the response's `metrics.prefill_ms`.
- **This section is a performance comparison only, with no bit-identical cross-check**: the two sides' output text agrees for the first few turns and diverges afterwards (sampling and template differences), which does not affect these conclusions.
- **32K / 64K were not measured.** KV costs about **264 KB/token** in practice (derived from `[L3] freed ... MB / prefix tokens`): at 16K this release's full-residency peak is 10.14 GB and llama.cpp's is 10.77 GB, both close to the 15.9 GB limit; going longer requires solving KV residency first (this release can use `--l3-evict` and per-layer residency; llama.cpp has no equivalent on this board).

---

### 8) Honest boundaries

- **8B is unchanged in this release, and that is expected**: 8B is a **dense** model, so every decode token reads all 6.15 GiB of weights and it is **bandwidth-bound**; this release optimizes **unpacking compute**, which does not help 8B. 30B-A3B is **MoE** and activates only the top-8 experts per token (about 1.6 GB), making it **compute-bound**, hence the large gain.
  > Caveat: 6.6 GB / 0.185 s implies 35.7 GB/s, above the 25.74 GB/s recorded earlier, so the "8B is bandwidth-bound" explanation **still needs a direct bandwidth measurement for confirmation**; treat it as the current best explanation.
- **The 30B "warm" numbers are not a precise baseline**: the model is 17.66 GB against 15.9 GB of RAM, so it cannot fill the page cache and the figures depend on cache history (evidence in the note under section 2). Use the section 2 / 3 30B figures as orders of magnitude only.
- **Per-layer residency is plaintext-only**: VQF-Enc / SM2-signed weights are explicitly rejected (they must stay fully resident).
- It does not coexist with expert windows (`VLLM_EW*`; EW owns the layer entry hook).
- Per-layer residency compresses **weight residency**; the KV base is constrained separately by KV v2 lazy allocation (with `VLLM_KV_NOF32=1` resident memory can be pushed lower -- see [docs/KV缓存v2-惰性分配与分层驻留方案.md](docs/KV缓存v2-惰性分配与分层驻留方案.md)).
- **The 2B tier has been removed from this chapter**: the `Qwen3-VL-2B-Instruct` dual weights (4.16 GB) used earlier are no longer on the active board; what exists now is `Qwen3-VL-2B-q8fix` (3.19 GB, q8 and not dual), a different quantization basis that must not be mixed with the old figures.
- **The x86-64 branch is for functional self-test and bit-exactness comparison only** and is never a performance baseline.

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
powershell -ExecutionPolicy Bypass -File tools\build\check_x64.ps1
#   → compiles + runs --test-l3 / --test-sparse self-tests, exit code 0/1
powershell -ExecutionPolicy Bypass -File tools\build\build_x64.ps1 -Gcc D:\tools\mingw64\bin\gcc.exe
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
# + optional vocab.bin (generated from tokenizer.json by tools/build/build_vocab_bin.py;
#   if absent the engine falls back to the embedded vocab — functional, but size/vocab
#   then follow the model's own):
python tools/build/build_vocab_bin.py <tokenizer.json> <vocab.bin> <vocab.bin>
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

## FHE Ciphertext Inference Chain Drivers (`tools/drivers/`)

On top of the engine's RNS-CKKS core (`src/core/vllm_ntt.c` / `vllm_ckks.c` / `vllm_tp.c`), this repository
ships **layer-chain drivers** that run a model layer by layer on ciphertexts
(each layer = one `lay` forward hop + one `boot` bootstrap refresh):

```bash
# lay / fin (layer chain, 112 primes)
gcc -O2 -fopenmp -Wno-implicit-function-declaration \
    -I include -I include/core -I include/common \
    -DCKKS_N=2048 -DCKKS_NPRIMES=112 -DBB=32 -DGG=32 \
    src/core/vllm_ntt.c src/core/vllm_ckks.c src/core/vllm_tp.c \
    tools/drivers/t23_m3p.c -o t23lay -lm

# boot (2100 primes; a large stack is required)
gcc -O2 -fopenmp -Wno-implicit-function-declaration '-Wl,--stack,33554432' \
    -I include -I include/core -I include/common \
    -DCKKS_N=2048 -DCKKS_NPRIMES=2100 -DBB=32 -DGG=32 \
    src/core/vllm_ntt.c src/core/vllm_ckks.c src/core/vllm_tp.c \
    tools/drivers/t23_chain.c -o t23boot -lm
```

- **Usage, judging criteria, data-directory convention and the measured thread-count conclusions**:
  see [`tools/drivers/README.md`](tools/drivers/README.md)
  (**scope matters**: within a layer, the `[A+B]` segment is fastest at 4 threads; over a full hop, `boot`
  is ~20% faster at 8 threads; and the thread count is a *result variable* — `NT=4` and `NT=8` ciphertexts
  are not bit-identical).
- **Data is not shipped**: the repo excludes the ~7 GB data package; generate it with
  [`tools/preproc/`](tools/preproc/) (the exported weights have been SHA256-verified byte-for-byte against
  the published results).
- **Run / pack / verify scripts and the relay guide**: [`tools/relay/`](tools/relay/)
  (per-tool usage, judging criteria, data-directory convention).
- **Archive output directory** is produced on demand by
  [`tools/relay/collect_results.ps1`](tools/relay/collect_results.ps1) (default `results/L0-4/`, not tracked);
- **Community relay**: layers 0–4 are done and the remaining 23 layers are open for community compute —
  the relay post, operating manual, measured-data appendix and the **layer 0–4 hand-off archive
  (Release asset)** live in the coordination repo
  [`kestrel-fhe-relay`](https://gitee.com/pei-xiaoguang/kestrel-fhe-relay).

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
  (`tools/security/verify_attest.py`, pure Python with zero dependencies, exit code 0=PASS / 1=FAIL,
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
- **One-command v1 reproduction**: `sh tools/bench/bench_value.sh` — produces the per-layer vs full
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
| Third-party code inside | Only `stb_image.h` (MIT, Sean Barrett) and the llama.cpp-derived 4x4 asm kernel (MIT, The ggml authors); see [LICENSING.md](LICENSING.md) section 3 |
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
| Third-party inference/matrix libraries (ggml, OpenBLAS, oneDNN…) | None — GEMM/GEMV are hand-written (the llama.cpp-derived 4x4 asm is an MIT extract, credited in its file header; see LICENSING.md section 3) |
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
│   ├── bench/                 # benchmarks + streaming latency probe
│   │   ├── bench_value.sh     # per-layer value bench: A/B memory + warm stability + page-cache evidence (see Performance §4)
│   │   ├── bench_http_probe.py  # zero-dependency streaming HTTP latency probe (TTFT/tpot/peak VmHWM), called by the above
│   │   └── patch_vqf_max_seq.py # read/write the VQF header max_seq_len (--dump / --set N / --restore), see Performance §7
│   ├── build/                 # build + code generation
│   │   ├── gen_embedded_web.py  # HTML → embedded byte-array generator (re-run after page edits)
│   │   ├── build_vocab_bin.py   # tokenizer.json → vocab.bin (byte-decoding fixed)
│   │   ├── build_x64.ps1        # x86_64 (MinGW) native build script (not a baseline, consistency only)
│   │   └── check_x64.ps1        # x86 build + self-test in one command (exit code 0/1)
│   ├── client/                # vllm_client.py: OpenAI-compatible HTTP client (zero deps)
│   ├── ops/                   # vllm_mgr.py: engine process supervisor (start/stop/restart/status page)
│   ├── security/              # verify_attest.py (offline verify) + vllm_vqf_sign.c (VQF SM2 signing)
│   ├── npu/                   # npu_export_ops.py: RK3588 operator-level NPU model export
│   ├── kernels/               # extract_llama_asm.py + llama_gemm_q4_0_4x4_asm.c (MIT)
│   ├── preproc/               # FHE ciphertext-chain data preprocessing scripts
│   ├── drivers/               # FHE ciphertext-chain drivers (t23_m3p.c / t23_chain.c)
│   └── relay/                 # ciphertext-chain relay scripts: run / pack / verify / accuracy check
├── vqf_convert/               # standalone conversion tool (safetensors/GGUF → VQF v2)
└── docs/                      # technical docs / benchmark reports / security specs (Chinese)
    └── bench/                 # raw benchmark archive: one directory per measurement set, named YYYYMMDD-platform-topic
        └── 20260915-rk3588-llama-ab/   # raw artifacts behind Performance §7 (json / logs / summary / board scripts), see its MANIFEST.txt
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
| [docs/bench/](docs/bench/20260915-rk3588-llama-ab/MANIFEST.txt) | **Raw benchmark archive**: the raw artifacts behind each measurement in the main text (per-round response json, engine logs, `summary.txt`, `REPORT.txt`) plus the board-side scripts that produced them and the md5 evidence list; one directory per measurement set, entry point is the `MANIFEST.txt` inside |

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
  layer; `tools/build/build_x64.ps1` / `tools/build/check_x64.ps1` were added (parameterized, overridable
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
  [LICENSE](LICENSE) (AGPL-3.0 text); dual-licensing terms and commercial authorisation are in [LICENSING.md](LICENSING.md); contribution rules are in [CONTRIBUTING.md](CONTRIBUTING.md).
- Third-party components are retained under their own licences: `stb_image.h` (MIT, Sean
  Barrett) and the 4x4 asm GEMM file extracted from llama.cpp plus its derived kernels (MIT,
  The ggml authors) — copyright and licence text are in the respective file headers, detailed
  in [LICENSING.md](LICENSING.md) section 3. Both are compatible with AGPL.
- **Please report security vulnerabilities privately** (do not open a public issue); see
  [SECURITY.md](SECURITY.md) for the process and our response commitments.
- The NPU direct backend of this engine interacts only with the OS kernel driver's (stock
  rknpu) public UAPI and contains no closed-source library or third-party header.

## Contact

Commercial licensing / research collaboration / reproduction data: **398152090@qq.com**
(also via [Issues](https://gitee.com/pei-xiaoguang/kestrel-llm/issues) or Gitee direct message;
see [LICENSING.md](LICENSING.md) for the commercial agreement and dual-licensing terms)

For commercial licensing / research collaborations / data requests:
**398152090@qq.com**, or open an issue at https://gitee.com/pei-xiaoguang/kestrel-llm/issues.
