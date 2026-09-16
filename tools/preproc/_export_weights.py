#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# _export_weights.py — 把 Qwen3-VL-2B 的 safetensors 权重导出为层链驱动读取的 .bin
#
# 驱动（tools/drivers/t23_m3p.c / t23_chain.c）按 tail/w{L}/ 目录读取每层 9 个权重文件，
# 格式为「原始 float32、行主序 [out_dim, in_dim]、无转置」：
#     q_proj.bin  k_proj.bin  v_proj.bin  o_proj.bin
#     gate_proj.bin  up_proj.bin  down_proj.bin
#     in_ln.bin   post_ln.bin
# 另需 tail/embed.bin（词表 embedding，[vocab, 2048]）与 tail/norm.bin（final RMSNorm 权重）。
#
# 用法：
#   python tools/preproc/_export_weights.py                     # 全部 28 层 + embed + norm
#   python tools/preproc/_export_weights.py --layers 0,26,27     # 只导出指定层
#   python tools/preproc/_export_weights.py --out /tmp/wexp      # 换输出目录（默认 .tmp_tok/tail）
#   python tools/preproc/_export_weights.py --src <model.safetensors>
#
# 验证：对同一层用本脚本导出到临时目录，与已有 tail/w{L}/ 逐字节比对应完全一致。
import argparse, os
import numpy as np
from safetensors import safe_open

DEFAULT_SRC = 'Modl/Qwen3-VL-2B-Instruct/model.safetensors'
DEFAULT_OUT = '.tmp_tok/tail'
NLAYER, HID = 28, 2048

# safetensors 张量后缀 -> 驱动读取的文件名
WMAP = [
    ('self_attn.q_proj.weight', 'q_proj'),
    ('self_attn.k_proj.weight', 'k_proj'),
    ('self_attn.v_proj.weight', 'v_proj'),
    ('self_attn.o_proj.weight', 'o_proj'),
    ('mlp.gate_proj.weight', 'gate_proj'),
    ('mlp.up_proj.weight', 'up_proj'),
    ('mlp.down_proj.weight', 'down_proj'),
    ('input_layernorm.weight', 'in_ln'),
    ('post_attention_layernorm.weight', 'post_ln'),
]


def dump(path, arr):
    """按驱动约定写原始 float32（不写 header、不做转置）。"""
    arr.astype(np.float32).tofile(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=DEFAULT_SRC, help='safetensors 路径')
    ap.add_argument('--out', default=DEFAULT_OUT, help='输出目录（默认 .tmp_tok/tail）')
    ap.add_argument('--layers', default='all', help='all 或逗号分隔，如 0,26,27')
    ap.add_argument('--no-toplevel', action='store_true', help='跳过 embed.bin / norm.bin')
    a = ap.parse_args()

    layers = list(range(NLAYER)) if a.layers == 'all' else [int(x) for x in a.layers.split(',')]
    os.makedirs(a.out, exist_ok=True)
    f = safe_open(a.src, framework='pt')
    keys = set(f.keys())

    # 前缀探测：Qwen3-VL 为 model.language_model.，纯文本 Qwen3 可能是 model.
    P = 'model.language_model.'
    if (P + 'layers.0.self_attn.q_proj.weight') not in keys:
        P = 'model.'
    print('tensor prefix = %r' % P)

    for L in layers:
        d = os.path.join(a.out, 'w%d' % L)
        os.makedirs(d, exist_ok=True)
        base = '%slayers.%d.' % (P, L)
        for suffix, name in WMAP:
            t = f.get_tensor(base + suffix).float().numpy()
            dump(os.path.join(d, name + '.bin'), t)
        print('  w%-2d  %d files' % (L, len(WMAP)))

    if not a.no_toplevel:
        emb = P + 'embed_tokens.weight'
        if emb in keys:
            e = f.get_tensor(emb).float().numpy()
            dump(os.path.join(a.out, 'embed.bin'), e)
            print('  embed.bin  shape=%s' % (e.shape,))
        nrm = P + 'norm.weight'
        if nrm in keys:
            dump(os.path.join(a.out, 'norm.bin'), f.get_tensor(nrm).float().numpy())
            print('  norm.bin')

    print('done -> %s' % a.out)


if __name__ == '__main__':
    main()
