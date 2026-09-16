#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# _embed4.py — 生成 `.tmp_tok/l0/embed4.bin`：lay0 的明文输入（4 个 token 的 embedding 查表结果）
#
# 格式：float32、行主序 [4, 2048]、无 header（= 逐行 embed_tokens[id].float() 拼接，32768 字节）
# 这是 `_full_layers.py` / `_tail_ref.py` / `_adap_scale.py` 以及驱动 lay0（uform=false）的输入。
#
# 用法（在仓库根执行）：
#   python tools/preproc/_embed4.py                        # 默认 id 100,101,102,103 —— 与已发布数据逐字节相同
#   python tools/preproc/_embed4.py --ids 9707,1879,0,1    # 换成你自己提示词的 token id
#   python tools/preproc/_embed4.py --out /tmp/embed4.bin  # 换输出路径
#   python tools/preproc/_embed4.py --src <model.safetensors>
#
# 说明：本脚本不走 safetensors 的 numpy 读取路径，而是自己解析文件头后用 memmap 取行，
#       因为 numpy 版 safetensors 不认识 BF16 这个 dtype。
import argparse, json, os, struct, sys
import numpy as np

DEFAULT_SRC = 'Modl/Qwen3-VL-2B-Instruct/model.safetensors'
EMB_KEY     = 'model.language_model.embed_tokens.weight'
DEFAULT_IDS = [100, 101, 102, 103]
DEFAULT_OUT = '.tmp_tok/l0/embed4.bin'
HID         = 2048


def open_shard(path):
    """返回 (data_start, header_dict)。tensor 数据区从 data_start + data_offsets[0] 开始。"""
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        hdr = json.loads(f.read(n))
    return 8 + n, hdr


def read_rows(path, key, ids):
    """按 id 取 embedding 表的行，返回 float32 [len(ids), HID]。"""
    data_start, hdr = open_shard(path)
    if key not in hdr:
        cand = [k for k in hdr if 'embed_tokens' in k or k.endswith('embed_tokens.weight')]
        sys.exit('key %r not found in %s; candidates: %s' % (key, path, cand))
    info = hdr[key]
    dt, shape = info['dtype'], info['shape']
    off = data_start + info['data_offsets'][0]
    rows, cols = shape[0], shape[1]
    if cols != HID:
        sys.exit('unexpected embedding hidden size %d (expected %d)' % (cols, HID))
    for i in ids:
        if not (0 <= i < rows):
            sys.exit('token id %d out of range [0, %d)' % (i, rows))

    if dt == 'F32':
        tbl = np.memmap(path, dtype='<f4', mode='r', offset=off, shape=(rows, cols))
        out = np.asarray(tbl[ids], dtype=np.float32)
    elif dt == 'F16':
        tbl = np.memmap(path, dtype='<f2', mode='r', offset=off, shape=(rows, cols))
        out = np.asarray(tbl[ids]).astype(np.float32)
    elif dt == 'BF16':
        tbl = np.memmap(path, dtype='<u2', mode='r', offset=off, shape=(rows, cols))
        out = (np.asarray(tbl[ids]).astype(np.uint32) << 16).view(np.float32)
    else:
        sys.exit('unsupported dtype %s (expected F32 / F16 / BF16)' % dt)
    return np.ascontiguousarray(out, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=DEFAULT_SRC)
    ap.add_argument('--key', default=EMB_KEY)
    ap.add_argument('--ids', default=','.join(str(i) for i in DEFAULT_IDS),
                    help='4 个 token id，逗号分隔（默认 100,101,102,103）')
    ap.add_argument('--out', default=DEFAULT_OUT)
    a = ap.parse_args()

    ids = [int(x) for x in a.ids.split(',') if x.strip() != '']
    if len(ids) != 4:
        sys.exit('需要恰好 4 个 token id，收到 %d 个' % len(ids))

    x = read_rows(a.src, a.key, ids)
    d = os.path.dirname(a.out)
    if d:
        os.makedirs(d, exist_ok=True)
    x.tofile(a.out)
    print('embed4 |max|=%.4f  ids=%s  -> %s (%d B)' % (np.abs(x).max(), ids, a.out, x.size * 4))


if __name__ == '__main__':
    main()
