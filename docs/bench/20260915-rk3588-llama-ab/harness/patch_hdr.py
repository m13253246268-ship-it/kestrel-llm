#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""patch_hdr.py <new_max_seq> —— 改 VQF 头的 max_seq_len（int32 LE @ offset 44）
先备份前 4096 字节到 /mnt/emmc/kv2_16k/vqf_hdr_orig.bin（只备份一次），可原样回滚。
"""
import os
import struct
import sys

P = "/mnt/VQF/8b/serve/model.vqf"
BAK = "/mnt/emmc/kv2_16k/vqf_hdr_orig.bin"
OFF = 44
new = int(sys.argv[1]) if len(sys.argv) > 1 else 20480

os.makedirs(os.path.dirname(BAK), exist_ok=True)
if not os.path.exists(BAK):
    with open(P, "rb") as f:
        d = f.read(4096)
    with open(BAK, "wb") as f:
        f.write(d)
    print("backup -> %s (%d bytes)" % (BAK, len(d)))
else:
    print("backup exists: %s" % BAK)

with open(P, "r+b") as f:
    f.seek(OFF)
    old = struct.unpack("<i", f.read(4))[0]
    f.seek(OFF)
    f.write(struct.pack("<i", new))
    f.flush()
    os.fsync(f.fileno())
print("max_seq_len: %d -> %d (offset %d)" % (old, new, OFF))
with open(P, "rb") as f:
    f.seek(OFF)
    now = struct.unpack("<i", f.read(4))[0]
print("verify: %d" % now)
