**English** | [简体中文](README.md)

# Data preprocessing scripts (tools/preproc/)

The scripts here **self-generate** the ~7 GB data package required by the layer-chain drivers
(weights + plaintext references + fitted coefficients), so that anybody reproducing our results does
not depend on us shipping large files.

> **Prerequisite**: first build `t23lay` / `t23boot` following the "FHE ciphertext inference chain
> drivers" section of the root README. Driver usage and the data-directory convention are documented
> in [`../drivers/README.md`](../drivers/README.md).

## 1. Requirements

| Item | Requirement |
|---|---|
| Python | 3.10+ (verified on 3.12.9) |
| Packages | `numpy`, `safetensors` (`pip install numpy safetensors`) |
| Model | `Modl/Qwen3-VL-2B-Instruct/model.safetensors` (~4.26 GB) |
| Working dir | **Run from the repository root** — the paths inside the scripts are relative and hard-coded |

## 2. Script list and outputs

| Script | Purpose | Output |
|---|---|---|
| **`_export_weights.py`** | **Quantization-free weight export** (safetensors → the `.bin` the driver reads) | `tail/w0..w27/` (9 files each) + `tail/embed.bin` + `tail/norm.bin` |
| **`_embed4.py`** | **lay0's plaintext input**: gather rows of the embedding table by token id | `l0/embed4.bin` (4×2048 float32, 32768 B) |
| **`_silu_mode.py`** | **per-layer SiLU path switch**: `silu{L}.bin` = `1.0` (direct fit) / `0.0` (÷21 fold) | `tail/silu0..25.bin` (26 files, 8 B each) |
| `_full_layers.py` | Plaintext causal forward references for all 28 layers (L0..L25) | `tail/l{L}_*`, `tail/ln{L}_e1/p1.bin`, `tail/m2c{L}_e/p.bin` |
| `_tail_ref.py` | Tail-segment references (L26/L27 + final norm + lm_head) | `tail/l26_*`, `tail/l27_*`, `tail/xnorm_f.bin`, `tail/logits.bin`, `tail/ln_f.bin`, `tail/m2c_f.bin` |
| `_adap_scale.py` | Per-layer folding-scale calibration `F[L]`, regenerates input reference | `tail/scale{L}.bin`, `tail/l0_u1.bin` |
| `_sin_fit*.py` / `_cos_fit.py` | Polynomial fitting of the SiLU/SwiGLU activation | fitted coefficients (used with `_adap_scale`) |
| `_recip_fit.py` / `_refit_lnq.py` / `_ln_fit*.py` | LNQ inverse (`1/sqrt`) fitting and refitting | `tail/ln*_e1/p1.bin`, `tail/m2c*.bin` |
| `_attn_fit.py` | Attention-segment fitting | — |
| `_fold_fit.py` / `_fold_sim.py` / `_cfold_plan.py` | Error simulation / planning for folded vs unfolded domains | statistics (used to pick parameters) |
| `_g256_conv.py` / `_g256_chk.py` | 256-group (`BB=32`/`GG=32`) conversion and checking | check output |
| `_silu_err.py` / `_logits_tol.py` | Error diagnostics (SiLU error, logits tolerance) | diagnostic output |

## 3. Recommended order

```bash
# 0) run from the repository root
cd <repo-root>

# 1) weights and top-level matrices (mandatory, do this first)
python tools/preproc/_export_weights.py

# 1b) lay0's plaintext input l0/embed4.bin (mandatory: _full_layers.py / _tail_ref.py / _adap_scale.py need it)
python tools/preproc/_embed4.py

# 1c) per-layer SiLU path switch silu{0..25}.bin (mandatory: if missing the driver falls back to the
#     ÷21 fold path, which is simply wrong for shallow layers)
python tools/preproc/_silu_mode.py

# 2) plaintext references: all layers + tail segment
python tools/preproc/_full_layers.py
python tools/preproc/_tail_ref.py

# 3) fitting: activation / LNQ inverse
python tools/preproc/_sin_fit.py
python tools/preproc/_recip_fit.py

# 4) per-layer folding-scale calibration + regenerate input reference (rewrites l0_u1.bin)
python tools/preproc/_adap_scale.py
```

The remaining `_*_fit*.py` / `_*_sim.py` / `_g256_*.py` / `_*_err.py` / `_*_tol.py` scripts are for
**parameter selection and diagnostics**; run them as needed.

## 4. Output format (what the driver reads)

- **Weights**: raw `float32`, row-major `[out_dim, in_dim]`, no header, no transpose.
  E.g. `tail/w0/q_proj.bin` = 2048×2048×4 = 16,777,216 bytes; `tail/w0/gate_proj.bin` = 6144×2048×4.
- **`tail/embed.bin`**: `[151936, 2048]` float32 (vocabulary embedding matrix).
- **References** (`tail/l{L}_*.bin`): 4 tokens × 2048 float32 (e.g. `_u1` / `_u2_ref`);
  `logits.bin` is 4 × 151936 float32.

## 5. Verification of `_export_weights.py`

Export a layer into a temporary directory and compare byte-for-byte (SHA256) against the
`tail/w{L}/` shipped with the published data:

```
layer 0  : q/k/v/o/gate/up/down + in_ln + post_ln  ->  9/9  IDENTICAL
layer 26 : 9/9 IDENTICAL
layer 27 : 9/9 IDENTICAL
embed.bin  IDENTICAL   (1,244,659,712 B)
norm.bin   IDENTICAL   (8,192 B)
```

In other words: **this script's output is exactly the weights that produced the published
first-5-layer results.**

## 6. Known gaps (honest note)

1. **`l0/embed4.bin` (lay0's plaintext input) used to have no generator — that is now fixed**:
   generate it with [`_embed4.py`](_embed4.py). It gathers rows of the embedding table by token id
   and defaults to `--ids 100,101,102,103` (the four ids the published data uses); the output is
   **32768 B** with SHA256 prefix `c945c8fe92cc0595`, **byte-identical** to the `embed4.bin` in the
   data package (verified by re-running it). For your own prompt use `--ids <your 4 token ids>` —
   that gives you **your own chain**, and the references must be regenerated with `_full_layers.py` /
   `_tail_ref.py`; it will no longer be byte-identical to the published results.
2. **`tail/silu{L}.bin` (the per-layer SiLU path switch) also had no generator — that is now fixed**:
   use [`_silu_mode.py`](_silu_mode.py); it writes 26 8-byte files (layers 0..25) that are
   **byte-identical** to the published data (verified by re-running it).
   The driver (`t23_m3p.c:1650-1660`) reads it to pick the SiLU for lay{L}: `1.0` = direct fit (shallow
   layers, |gate| ≤ 8, anchored at 0); `0.0` **or a missing file** = the ÷21 folded path (M3b deep layers).
   **Missing this file is wrong for shallow layers** — the ÷21 fit has a ~0.0127 DC offset, and the driver
   comment itself quantifies "12–35% relative error for small shallow gates".
   > ⚠️ **The danger: with the file missing the driver reports no error.** `RESULT=PASS` is still printed,
   > but `verify_layer` fails on independent re-check (measured: layer 0 without the file gives
   > `max|err|` 4.2e-2–8.5e-2, above the 3e-2 tolerance; with it, back to 1.27e-3).
   > So **a criteria log is not a substitute for numeric re-checking** — see [`../relay/README.md`](../relay/README.md).
3. **`.tmp_tok/l1w/`, `.tmp_tok/l1r/`** belong to an early single-layer (L1) debug flow; the mid-layer
   chain (`phase=6`) does not need them. The scripts here do not cover them, and they are irrelevant
   to running the 0..27 layer chain.
4. **The model path (`Modl/...`) and output paths (`.tmp_tok/...`) inside the scripts are hard-coded**;
   on another machine you must edit the `SRC` / `D0` / `OUT` constants.

## 7. See also

- **Driver usage / data-directory convention / judging criteria**: [`../drivers/README.md`](../drivers/README.md)
- **Run / pack / verify scripts**: [`../relay/README.md`](../relay/README.md)
- **Archive output directory**: `results/L0-4/` (produced by `../relay/collect_results.ps1`)
