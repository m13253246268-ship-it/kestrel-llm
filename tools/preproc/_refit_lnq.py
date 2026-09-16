#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# _refit_lnq.py - 修复浅层 LNQ 拟合退化
# 背景: fit_lnq 目标 y=1/sqrt(u+1e-6)/128 中 eps=1e-6 对 u~m2c<~1e-5 的浅层
#       输入域（u=y/F 极小）不可忽略 -> 目标退化为常量 ~1000/128 -> 引擎 xn~0。
# 本脚本: 对 m2c<2.4e-4（引擎将做 8^k 输入预缩，LNQ 除以同因子）的层/域
#       重新拟合无 eps 污染的准确多项式 P_orig(u), 存入 ln{L}_{e1|p1}.bin。
#       m2c{L}_{e|p}.bin 保持原值不动（引擎内部乘 M^2 用）。
import numpy as np, os
OUT = '.tmp_tok/tail'
T = 2.4e-4
for L in range(20):
    for tag, mcfile, lnfile in (('e1', 'm2c%d_e.bin', 'ln%d_e1.bin'),
                                ('p1', 'm2c%d_p.bin', 'ln%d_p1.bin')):
        mp = os.path.join(OUT, mcfile % L)
        lp = os.path.join(OUT, lnfile % L)
        if not os.path.exists(mp) or not os.path.exists(lp):
            continue
        m2c = float(np.fromfile(mp, dtype=np.float64)[0])
        if m2c >= T:
            continue
        c0, s0 = 0.75 * m2c, 0.25 * m2c
        u = np.linspace(0.5 * m2c, 1.5 * m2c, 200001)
        z = (u - c0) / (8.0 * s0)
        y = 1.0 / np.sqrt(u) / 128.0          # 无 eps 污染的精确目标
        q = np.polyfit(z, y, 8)
        e = np.abs(np.polyval(q, z) - y).max()
        rel = e / y.mean()
        print('L%02d %s m2c=%.3e refit err=%.3e (rel %.2e)' % (L, tag, m2c, e, rel))
        q.astype(np.float64).tofile(lp)
print('DONE')
