#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# _silu_mode.py — 生成 `.tmp_tok/tail/silu{L}.bin`：逐层 SiLU 路径开关
#
# 驱动（tools/drivers/t23_m3p.c:1650-1660）读这个文件决定 lay{L} 用哪条 SiLU 路径：
#     文件内容（8 字节 float64） == 1.0   → 直接拟合（浅层；|gate| <= 8，锚定 0）
#     文件缺失 或 == 0.0                 → ÷21 折叠（M3b 深层路径）
#
# ⚠️ **文件缺失 = 走 ÷21 折叠**，对浅层是错的：驱动注释自己写了 ÷21 拟合有 ~0.0127 DC 偏移，
#    浅层小 gate 相对误差 12~35%。而且驱动**不会报错**——`RESULT=PASS` 仍然会打出来，
#    但 `verify_layer` 独立复核会 FAIL。所以这个文件是数据包的必要组成部分，不能省。
#
# 本脚本写出的就是**产出已发布结果所用的逐层选择**（层 0..25 共 26 个文件；
# 层 26/27 走 tail 专线，不读该文件）。
#
# 用法（在仓库根执行）：
#   python tools/preproc/_silu_mode.py                  # 写全部 26 层（与已发布数据逐字节相同）
#   python tools/preproc/_silu_mode.py --layers 0       # 只写层 0（排障用）
#   python tools/preproc/_silu_mode.py --out /tmp/tail  # 换输出目录
import argparse, os, struct, sys

DEFAULT_OUT = '.tmp_tok/tail'
LAY_MAX = 25

# 产出已发布结果的逐层选择：集合内 = 直接拟合(direct)，其余 = ÷21 折叠。
# 依据：驱动注释「silu{lay}.bin=1 → 直接拟合(|gate|<=8, 锚定0, M1c 路径)」。
DIRECT_LAYERS = {0, 1, 2, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16, 18}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=DEFAULT_OUT, help='输出目录（默认 .tmp_tok/tail）')
    ap.add_argument('--layers', default='', help='逗号分隔的层号；默认全部 0..%d' % LAY_MAX)
    a = ap.parse_args()

    if a.layers.strip():
        layers = [int(x) for x in a.layers.split(',') if x.strip() != '']
    else:
        layers = list(range(LAY_MAX + 1))

    os.makedirs(a.out, exist_ok=True)
    n_direct = 0
    for L in layers:
        if not (0 <= L <= LAY_MAX):
            sys.exit('layer %d out of range [0, %d]' % (L, LAY_MAX))
        v = 1.0 if L in DIRECT_LAYERS else 0.0
        n_direct += int(v == 1.0)
        with open(os.path.join(a.out, 'silu%d.bin' % L), 'wb') as f:
            f.write(struct.pack('<d', v))

    print('wrote %d file(s) -> %s  (direct=%d, fold=%d)'
          % (len(layers), a.out, n_direct, len(layers) - n_direct))


if __name__ == '__main__':
    main()
