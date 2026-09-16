#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""patch_vqf_max_seq.py —— 读写 VQF 头的 max_seq_len（int32 LE @ offset 44）

为什么需要它：引擎的上下文窗口上限来自 VQF 头
（`src/model/vqf.c`: `c->max_seq_len = (int)a->max_seq_len`），**没有运行时开关**。
权重文件是按 8192 生成的，要跑 16K 就得把这 4 字节元数据抬上去。
本脚本**只改这 4 字节元数据，不触碰任何权重数值**；改前自动备份头部前 4096 字节。

用法：
    python3 patch_vqf_max_seq.py --dump   <model.vqf>
    python3 patch_vqf_max_seq.py --set N  <model.vqf>      # N 如 20480（自动备份到 <model>.hdr.bak）
    python3 patch_vqf_max_seq.py --restore <model.vqf>     # 从 <model>.hdr.bak 回滚

参考：README §7「与 llama.cpp 的权重公平 A/B」的 16K 档即用本脚本把 8192 -> 20480。
"""
import argparse
import os
import struct
import sys

OFF = 44          # max_seq_len 的字节偏移（int32 LE）
HDR_LEN = 4096    # 备份长度：整段头部
# 头布局（int32 LE，已用 od 核对）：0=magic 4=version 8=arch 12=? 16=dim 20=layers
# 24=heads 28=kv_heads 32=head_dim 36=ffn 40=vocab 44=max_seq
FIELDS = [(16, "dim"), (20, "layers"), (24, "heads"), (28, "kv_heads"),
          (32, "head_dim"), (36, "ffn"), (40, "vocab"), (OFF, "max_seq")]


def read_fields(path):
    out = {}
    with open(path, "rb") as f:
        for off, name in FIELDS:
            f.seek(off)
            out[name] = struct.unpack("<i", f.read(4))[0]
    return out


def dump(path):
    fl = read_fields(path)
    print("%s  大小 %.2f GB" % (path, os.path.getsize(path) / 1024 ** 3))
    for _, name in FIELDS:
        print("  %-9s = %d" % (name, fl[name]))
    bak = path + ".hdr.bak"
    print("  备份: %s" % (bak if os.path.exists(bak) else "（无）"))


def set_max_seq(path, new):
    bak = path + ".hdr.bak"
    if not os.path.exists(bak):
        with open(path, "rb") as f:
            head = f.read(HDR_LEN)
        with open(bak, "wb") as f:
            f.write(head)
        print("已备份头部 -> %s" % bak)
    with open(path, "r+b") as f:
        f.seek(OFF)
        old = struct.unpack("<i", f.read(4))[0]
        f.seek(OFF)
        f.write(struct.pack("<i", new))
        f.flush()
        os.fsync(f.fileno())
    print("max_seq: %d -> %d" % (old, new))
    dump(path)


def restore(path):
    bak = path + ".hdr.bak"
    if not os.path.exists(bak):
        sys.exit("找不到备份 %s，无法回滚" % bak)
    with open(bak, "rb") as f:
        head = f.read(HDR_LEN)
    with open(path, "r+b") as f:
        f.seek(0)
        f.write(head)
        f.flush()
        os.fsync(f.fileno())
    print("已从 %s 回滚头部" % bak)
    dump(path)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dump", action="store_true", help="打印头字段")
    g.add_argument("--set", type=int, metavar="N", help="设置 max_seq_len")
    g.add_argument("--restore", action="store_true", help="从 .hdr.bak 回滚")
    ap.add_argument("model")
    a = ap.parse_args()
    if a.dump:
        dump(a.model)
    elif a.restore:
        restore(a.model)
    else:
        set_max_seq(a.model, a.set)


if __name__ == "__main__":
    main()
