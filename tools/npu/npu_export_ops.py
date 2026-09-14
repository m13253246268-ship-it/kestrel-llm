#!/usr/bin/env python3
"""
npu_export_ops.py - Deploy-time export of transparent-NPU operator models.

RK3588 (rknn-toolkit2) reference: converts generic MatMul operator graphs
into .rknn files so the New_vLLM engine can offload GEMM stages to the NPU
WITHOUT converting the safetensors model format. The engine keeps loading
its original weights; these .rknn files are acceleration artifacts derived
from the projection shapes (and, in int8 mode, from the weights themselves).

Run on an x86_64 Linux host with rknn-toolkit2 installed:

    pip install rknn-toolkit2 numpy onnx

    python tools/npu/npu_export_ops.py \
        --config <model-config-dir>/config.json \
        --out npu_ops --mode fp32 --m-list 1,16

Output (mode fp32, one generic MatMul per shape; A and B are RUNTIME inputs
so the same file serves every layer):

    npu_ops/gemm_1x12288x4096.rknn   (QKV/Gate/Up:    dim -> 3*dim / ffn)
    npu_ops/gemm_16x12288x4096.rknn
    npu_ops/gemm_16x4096x12288.rknn  (Down: ffn -> dim)
    npu_ops/gemm_16x4096x4096.rknn   (O:    dim -> dim)
    ...

Output (mode int8, per-(layer, projection) baked-weight GEMM):

    npu_ops/gemm_i8_p0_l0.rknn ...   (QKV, layer 0)
    npu_ops/manifest.json            (shape/scale metadata for the engine)

The C side (vllm_npu.c) loads these lazily by shape (fp32) or by
(proj, layer) (int8) and transparently falls back to the CPU NEON path when
a model is missing or the NPU runtime is absent.

Axiom note: the int8 path implements blas_precision_efficiency_tradeoff
(INT8 weights, controlled error bound). The engine verifies consistency at
the PPL / greedy-text level, not bit-exactness (different reduction order).
"""

import argparse
import json
import os
import sys

import numpy as np


def read_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    hc = cfg.get("hidden_size") or cfg.get("d_model")
    ffn = cfg.get("intermediate_size") or cfg.get("ffn_dim")
    vocab = cfg.get("vocab_size")
    layers = cfg.get("num_hidden_layers") or cfg.get("num_layers")
    nkv = cfg.get("num_key_value_heads", 0)
    hd = cfg.get("head_dim", 0)
    if not hd:
        nq = cfg.get("num_attention_heads", 0)
        hd = hc // nq if nq else 128
    if not (hc and ffn and vocab and layers):
        sys.exit(f"config.json: missing hidden_size/intermediate_size/"
                 f"vocab_size/num_hidden_layers in {path}")
    return int(hc), int(ffn), int(vocab), int(layers), int(nkv), int(hd)


def projection_shapes(hc, ffn, vocab, nkv, hd):
    """(proj_id, name, K, N) for every distinct projection the engine runs.
    Proj ids MUST match VLLM_NPU_PROJ_* in include/vllm_npu.h."""
    kv = (nkv * hd) if nkv else hc   # GQA: kv_rows = n_kv_heads * head_dim
    return [
        (0, "q",     hc,  hc),
        (1, "k",     hc,  kv),
        (2, "v",     hc,  kv),
        (3, "o",     hc,  hc),
        (4, "gate",  hc,  ffn),
        (5, "up",    hc,  ffn),
        (6, "down",  ffn, hc),
        (7, "lmhead", hc, vocab),
    ]


def build_onnx_fp32(M, N, K, tmp_dir):
    """Generic MatMul: A[M,K] x B[K,N] -> C[M,N]. Both A and B are inputs."""
    import onnx
    from onnx import helper, TensorProto

    a = helper.make_tensor_value_info("A", TensorProto.FLOAT, [M, K])
    b = helper.make_tensor_value_info("B", TensorProto.FLOAT, [K, N])
    c = helper.make_tensor_value_info("C", TensorProto.FLOAT, [M, N])
    node = helper.make_node("MatMul", ["A", "B"], ["C"])
    graph = helper.make_graph([node], f"gemm_{M}x{N}x{K}", [a, b], [c])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    path = os.path.join(tmp_dir, f"gemm_{M}x{N}x{K}.onnx")
    onnx.save(model, path)
    return path


def build_onnx_i8(M, N, K, wq, ws, tmp_dir, tag):
    """
    Baked-weight int8 GEMM: MatMul(int8 A, int8 W) -> int32 -> fp32,
    then Mul by per-channel weight scale ws[N] and per-token scale aw[M].
    Inputs: A (int8, passthrough - engine feeds its own Q8 bytes),
            aw (fp32 [M]).
    Constants: W (int8 [K,N]), ws (fp32 [N]).
    """
    import onnx
    from onnx import helper, TensorProto

    a = helper.make_tensor_value_info("A", TensorProto.INT8, [M, K])
    aw = helper.make_tensor_value_info("aw", TensorProto.FLOAT, [M])
    c = helper.make_tensor_value_info("C", TensorProto.FLOAT, [M, N])

    w_init = helper.make_tensor("W", TensorProto.INT8, [K, N], wq.flatten().tolist())
    ws_init = helper.make_tensor("ws", TensorProto.FLOAT, [N], ws.flatten().tolist())

    dot = helper.make_node("MatMul", ["A", "W"], ["dot"])
    cast = helper.make_node("Cast", ["dot"], ["dot_f"], to=TensorProto.FLOAT)
    mul_w = helper.make_node("Mul", ["dot_f", "ws"], ["scaled"])
    aw2 = helper.make_node("Reshape", ["aw", "aw_shape"], ["aw_2d"])
    aw_shape = helper.make_tensor("aw_shape", TensorProto.INT64, [2], [M, 1])
    mul_a = helper.make_node("Mul", ["scaled", "aw_2d"], ["C"])

    graph = helper.make_graph(
        [dot, cast, mul_w, aw2, mul_a], f"gemm_i8_{tag}",
        [a, aw], [c],
        initializer=[w_init, ws_init, aw_shape])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    path = os.path.join(tmp_dir, f"gemm_i8_{tag}.onnx")
    onnx.save(model, path)
    return path


def quantize_per_channel(w):
    """Per-output-channel symmetric int8 quantization (saturation, no wrap;
    matches the engine's fixedpoint_quantize_saturate red line)."""
    amax = np.abs(w).max(axis=0, keepdims=True)
    amax = np.maximum(amax, 1e-10)
    scale = amax / 127.0
    q = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
    return q, scale.reshape(-1)


def load_safetensors_shard_weights(path):
    """Minimal safetensors reader: returns {tensor_name: np.ndarray fp32}.
    The engine itself already converts BF16/F16 -> fp32 at load time; this
    mirrors that so the baked int8 weights match the engine's fp32 view."""
    import struct

    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        data = f.read()
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        begin = meta["data_offsets"][0]
        end = meta["data_offsets"][1]
        shape = meta["shape"]
        dtype = meta["dtype"]
        raw = data[begin:end]
        if dtype == "F32":
            arr = np.frombuffer(raw, dtype=np.float32).reshape(shape)
        elif dtype == "BF16":
            raise NotImplementedError(
                "BF16 decode is not implemented in the tool - use the engine's "
                "--dump-weights (st_load_tensor already converts BF16->fp32) "
                "and point --weights-dir at the dumped .npy files")
        elif dtype == "F16":
            arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(shape)
        else:
            raise NotImplementedError(f"dtype {dtype}")
        out[name] = arr
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="safetensors config.json")
    ap.add_argument("--out", default="npu_ops", help="output dir for .rknn files")
    ap.add_argument("--mode", choices=["fp32", "int8"], default="fp32")
    ap.add_argument("--m-list", default="1,16",
                    help="comma-separated token batch sizes M to export")
    ap.add_argument("--weights-dir", default=None,
                    help="dir of per-layer fp32 .npy weight files (int8 mode; "
                         "see --dump-weights in the engine) - if omitted, "
                         "int8 mode tries the safetensors shards in the "
                         "config dir")
    ap.add_argument("--skip-lmhead", action="store_true",
                    help="skip the (large) lm_head export")
    args = ap.parse_args()

    try:
        from rknn.api import RKNN
    except ImportError:
        sys.exit("rknn-toolkit2 not importable; run on an x86_64 Linux host "
                 "with `pip install rknn-toolkit2 numpy onnx`")

    hc, ffn, vocab, n_layers, nkv, hd = read_config(args.config)
    m_list = [int(x) for x in args.m_list.split(",") if x.strip()]
    os.makedirs(args.out, exist_ok=True)
    os.makedirs("npu_tmp", exist_ok=True)

    manifest = {"target": "rk3588", "mode": args.mode,
                "hidden": hc, "ffn": ffn, "vocab": vocab, "layers": n_layers,
                "kv_dim": nkv * hd if nkv else hc,
                "models": {}}

    for proj_id, name, K, N in projection_shapes(hc, ffn, vocab, nkv, hd):
        if args.skip_lmhead and name == "lmhead":
            continue
        if args.mode == "fp32":
            for M in m_list:
                tag = f"{M}x{N}x{K}"
                onnx_path = build_onnx_fp32(M, N, K, "npu_tmp")
                rknn_path = os.path.join(args.out, f"gemm_{tag}.rknn")
                rknn = RKNN(verbose=False)
                rknn.config(mean_values=[[0]], std_values=[[1]],
                            target_platform="rk3588")
                rknn.load_onnx(model=onnx_path,
                               input_size_list=[[1, M, K], [1, K, N]])
                rknn.build(do_quantization=False, pre_compile=False)
                rknn.export_rknn(rknn_path)
                rknn.release()
                manifest["models"][f"gemm_{tag}"] = {"M": M, "N": N, "K": K}
                print(f"[npu-export] {rknn_path}")
        else:
            for layer in range(n_layers):
                w = load_proj_weights(args, layer, name, K, N)
                if w is None:
                    print(f"[npu-export] SKIP p{proj_id} l{layer} "
                          f"({name} {K}x{N}): weights unavailable")
                    continue
                wq, ws = quantize_per_channel(w)      # [K][N] -> int8 + [N]
                tag = f"p{proj_id}_l{layer}"
                onnx_path = build_onnx_i8(m_list[0], N, K, wq, ws,
                                          "npu_tmp", tag)
                rknn = RKNN(verbose=False)
                rknn.config(mean_values=[[0]], std_values=[[1]],
                            target_platform="rk3588",
                            quantized_dtype="asymmetric_quantized-8")
                rknn.load_onnx(model=onnx_path,
                               input_size_list=[[1, m_list[0], K], [1, m_list[0]]])
                # do_quantization=True quantizes A to int8; the baked W is
                # already int8 passthrough. Calibration uses random normal
                # activations; adjust with real activations for max fidelity.
                ds = os.path.join("npu_tmp", f"dataset_{tag}.txt")
                with open(ds, "w") as f:
                    f.write("calib.bin\n")
                calib = np.random.randn(m_list[0] * K).astype(np.float32)
                calib.tofile(os.path.join("npu_tmp", "calib.bin"))
                rknn.build(do_quantization=True, dataset=ds, pre_compile=False)
                rknn_path = os.path.join(args.out, f"gemm_i8_{tag}.rknn")
                rknn.export_rknn(rknn_path)
                rknn.release()
                manifest["models"][f"gemm_i8_{tag}"] = {
                    "M": m_list[0], "N": N, "K": K, "proj": proj_id,
                    "layer": layer}
                print(f"[npu-export] {rknn_path}")

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[npu-export] manifest: {os.path.join(args.out, 'manifest.json')}")


def load_proj_weights(args, layer, name, K, N):
    """Fetch the fp32 weight matrix for (layer, projection).
    Prefer pre-dumped .npy (engine --dump-weights), else read the safetensors
    shards next to --config. Weights are [out=N][in=K] in the engine layout;
    transpose to [K][N] for the baked MatMul B operand."""
    if args.weights_dir:
        p = os.path.join(args.weights_dir, f"l{layer}_{name}.npy")
        if os.path.exists(p):
            w = np.load(p)
            return np.ascontiguousarray(w.T) if w.shape == (N, K) else w
        return None
    # Fallback: read shards from the config directory (best effort).
    cfg_dir = os.path.dirname(args.config)
    suffix_map = {
        "q": "q_proj", "k": "k_proj", "v": "v_proj", "o": "o_proj",
        "gate": "gate_proj", "up": "up_proj", "down": "down_proj",
        "lmhead": "lm_head",
    }
    suffix = suffix_map[name]
    for shard in sorted(os.listdir(cfg_dir)):
        if not shard.endswith(".safetensors"):
            continue
        try:
            tensors = load_safetensors_shard_weights(os.path.join(cfg_dir, shard))
        except Exception:
            continue
        keys = [f"model.layers.{layer}.self_attn.{suffix}.weight",
                f"model.layers.{layer}.mlp.{suffix}.weight",
                f"model.layers.{layer}.{suffix}.weight",
                f"lm_head.weight"]
        for k in keys:
            if k in tensors:
                w = tensors[k]
                return np.ascontiguousarray(w.T) if w.shape == (N, K) else w
    return None


if __name__ == "__main__":
    main()
