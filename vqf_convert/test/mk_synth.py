#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""确定性 Qwen3-VL 合成小模型（safetensors bf16 + config + index），
用于 vqf_convert x86 与引擎 aarch64 转换产物 sha256 对拍。零第三方依赖。
"""
import json, os, struct, sys

OUT = sys.argv[1] if len(sys.argv) > 1 else r"./synth"
DIM, FF, NH, NKV, HD, NL, VOCAB = 256, 2304, 16, 8, 128, 2, 512


def lcg(seed):
    x = seed
    while True:
        x = (x * 1664525 + 1013904223) & 0xFFFFFFFF
        yield x


def bf16_from_f32(x):
    u = struct.unpack("<I", struct.pack("<f", x))[0]
    b = (u + 0x8000 + ((u >> 16) & 1)) >> 16
    return struct.pack("<H", b & 0xFFFF)


def tensor_bytes(rows, cols, rng):
    """确定性 bf16（含 0 / 极端值小块）。"""
    n = rows * cols
    out = bytearray(n * 2)
    for i in range(n):
        v = ((rng.__next__() % 65536) - 32768) / 32768.0
        if i % 77777 == 0:
            v = 0.0
        if i % 99991 == 0:
            v = -42.0
        out[i * 2:i * 2 + 2] = bf16_from_f32(v)
    return bytes(out)


os.makedirs(OUT, exist_ok=True)
config = {
    "architectures": ["Qwen3VLForConditionalGeneration"],
    "model_type": "qwen3_vl",
    "image_token_id": 151655,
    "text_config": {
        "attention_bias": False, "bos_token_id": 151643, "eos_token_id": 151645,
        "head_dim": HD, "hidden_act": "silu", "hidden_size": DIM,
        "intermediate_size": FF, "max_position_embeddings": 262144,
        "model_type": "qwen3_vl_text",
        "num_attention_heads": NH, "num_hidden_layers": NL,
        "num_key_value_heads": NKV, "rms_norm_eps": 1e-6,
        "rope_scaling": {"mrope_interleaved": True, "mrope_section": [24, 20, 20],
                         "rope_type": "default"},
        "rope_theta": 5000000.0, "tie_word_embeddings": False,
        "use_cache": True, "vocab_size": VOCAB,
    },
    "tie_word_embeddings": False,
    "video_token_id": 151656,
}
with open(os.path.join(OUT, "config.json"), "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)

header = {}
q_out, k_out = NH * HD, NKV * HD


def put(name, shape):
    header[name] = {"dtype": "BF16", "shape": shape}


put("model.language_model.embed_tokens.weight", [VOCAB, DIM])
put("model.language_model.norm.weight", [DIM, 1])
for l in range(NL):
    nm = "model.language_model.layers.%d." % l
    put(nm + "input_layernorm.weight", [DIM, 1])
    put(nm + "post_attention_layernorm.weight", [DIM, 1])
    put(nm + "self_attn.q_proj.weight", [q_out, DIM])
    put(nm + "self_attn.k_proj.weight", [k_out, DIM])
    put(nm + "self_attn.v_proj.weight", [k_out, DIM])
    put(nm + "self_attn.o_proj.weight", [DIM, q_out])
    put(nm + "self_attn.q_norm.weight", [HD, 1])
    put(nm + "self_attn.k_norm.weight", [HD, 1])
    put(nm + "mlp.gate_proj.weight", [FF, DIM])
    put(nm + "mlp.up_proj.weight", [FF, DIM])
    put(nm + "mlp.down_proj.weight", [DIM, FF])
put("lm_head.weight", [VOCAB, DIM])

rng = lcg(0x9E3779B9)
payload = bytearray()
for k in header:
    rows, cols = header[k]["shape"]
    payload += tensor_bytes(rows, cols, rng)

# 回填 data_offsets（引擎 parse 用）
off = 0
for k in header:
    rows, cols = header[k]["shape"]
    n = rows * cols * 2
    header[k]["data_offsets"] = [off, off + n]
    off += n

hdr_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
data = struct.pack("<Q", len(hdr_json)) + hdr_json + b"\x00" * 8 + bytes(payload)
with open(os.path.join(OUT, "model.safetensors"), "wb") as f:
    f.write(data)

index = {"metadata": {"total_size": len(data)}, "weight_map": {}}
for k in header:
    index["weight_map"][k] = "model.safetensors"
with open(os.path.join(OUT, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
    json.dump(index, f, indent=2)

print("synth ->", OUT, "bytes:", len(data))
