#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# _adap_scale.py - 逐层自适应尺度 (axiom_arith_dynamic_scale_collapse_001 落地)
# 目标: 每层链状态 s{L} = y{L}/F{L}, F{L} 按层输出 max 选取(2 的幂, 单调不减, 相邻比<=4),
#       s max <= ~0.35 (decode/boot 界), s rms >= ~0.02 (SNR>=20 vs ~1e-3 噪声地板)
# 事实基础: tail refs 明文恒等式 yL = y{L-1}+o+mlp 已验(1.5e-5); scale-free refs(x_norm/q/k/v/o/
#   gate/up/act/mlp_out/xattn_full) 不变; 仅残差路径 refs(u1/u2_ref/mid16) + ln/m2c 拟合重生成
# 用法: python .tmp_tok/_adap_scale.py   (会覆盖 tail/l{L}_u1|u2_ref|mid16 + ln/m2c fits, 先备份)
import numpy as np, os, shutil, math
T = '.tmp_tok/tail'
BAK = os.path.join(T, '_g1024_bak')
FOLD_OLD = 1024.0
EMB = np.fromfile('.tmp_tok/l0/embed4.bin', dtype=np.float32)      # 真值 embed4 (4x2048 flat)
TGT_MAX = 0.40        # s{L} max 上限 (decode/boot 兼容; 取幂2后实际 max ∈ (T/2, T])
SNR_MIN = 0.02        # s rms 目标下限
RATIO = 4             # F 相邻最大比(仅下缩: F 单调不减 → 输入比率乘恒 ≤1)

def ld(name, L):
    a = np.fromfile(os.path.join(T, name % L), dtype=np.float32)
    assert a.size > 0, name % L
    return a

# ---- 备份原始 /1024-gauge 残差 refs + fits (L0..25) ----
os.makedirs(BAK, exist_ok=True)
for L in range(26):
    for pat in ('l%d_u1.bin', 'l%d_u2_ref.bin', 'l%d_mid16.bin', 'ln%d_e1.bin', 'ln%d_p1.bin',
                'm2c%d_e.bin', 'm2c%d_p.bin'):
        src = os.path.join(T, pat % L)
        if os.path.exists(src):
            dst = os.path.join(BAK, pat % L)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)

def fit_lnq(uarr):
    m2s = [(uarr[t * 2048:(t + 1) * 2048] ** 2).mean() for t in range(4)]
    m2c = float(np.mean(m2s))
    c0, s0 = 0.75 * m2c, 0.25 * m2c
    u = np.linspace(0.5 * m2c, 1.5 * m2c, 200001)
    z = (u - c0) / (8 * s0)
    y = 1.0 / np.sqrt(u) / 128.0   # 无 eps: m2c~1e-6 时 +1e-6 会污染 ~20% (u>=0.5*m2c 永不触 0)
    q = np.polyfit(z, y, 8)
    e = np.abs(np.polyval(q, z) - y).max()
    return q, m2c, e

def dump(name, a):
    a.astype(np.float32).tofile(os.path.join(T, name))

# ---- 真值状态重建: 从 embed4 + o + mlp 前向积分 (o/mlp 永不被覆盖 → 幂等) ----
# 先恢复尾文件为原始 /1024-gauge(若备份存在), 保证可重复运行
if os.path.isdir(BAK):
    for fn in os.listdir(BAK):
        shutil.copy2(os.path.join(BAK, fn), os.path.join(T, fn))
y_true = {}
y_true[-1] = EMB
for L in range(26):
    oL = ld('l%d_o.bin', L)
    mlpL = ld('l%d_mlp_out.bin', L)
    y_true[L] = y_true[L - 1] + oL + mlpL   # 明文恒等式 yL = y{L-1}+o+mlp (已验 1.5e-5)

# ---- F 调度: 2 的幂, 单调不减(仅下缩比率), max(y)/F <= ~TGT_MAX ----
# L0 输入为明文(无前层稀释) → 仅按 max 定 F; 之后单调 clamp + F{L}<=4*F{L-1}
F = {}
for L in range(26):
    ymax = np.abs(y_true[L]).max()
    f = 2.0 ** math.ceil(max(math.log2(ymax / TGT_MAX), 0.0))     # max 界(0.40)
    if L >= 1:
        need50 = 2.0 ** math.ceil(max(math.log2(ymax / 0.5), 0.0))  # 硬界: 至少压到 ≤0.5(decode/boot)
        f = max(f, need50)
        if f > F[L - 1] * RATIO:
            f = max(F[L - 1] * RATIO, need50)  # 爆炸层(如真链 y2~55)跳升到 need50, 可超 4× 比
        if f < F[L - 1]: f = F[L - 1]                              # 单调不减 → 比率乘恒 ≤1
    if f < 1: f = 1.0
    F[L] = float(f)

# ---- 逐层重生成: u1(层输入 s{L-1}/embed4), u2_ref(s{L}), mid16(x_attn{L}/F{L}) + 拟合 ----
rows = []
for L in range(26):
    oL = ld('l%d_o.bin', L)          # 真值 o (scale-free, 不变)
    mlpL = ld('l%d_mlp_out.bin', L)  # 真值 mlp (不变)
    yLm1 = y_true[L - 1]             # 真值前层输出
    xa_true = yLm1 + oL              # 真值 x_attn{L}
    s = y_true[L] / F[L]             # 新链状态
    mid = xa_true / F[L]             # post-LN 输入域
    s_rms, s_max = math.sqrt((s ** 2).mean()), np.abs(s).max()
    # 恒等式(自适应 gauge): s{L} = yLm1/F[L] + (o+mlp)/F[L]
    id_err = np.abs((yLm1 + oL + mlpL) / F[L] - s).max()
    # E1 拟合域 = xct 经比率乘后的实际输入 = y{L-1}/F{L} (L0: embed4/F0) — 与 C 代码 post-ratio 一致
    in_state = yLm1 / F[L]
    qe, me, ee = fit_lnq(in_state)
    qp, mp, ep = fit_lnq(mid)
    dump('l%d_u2_ref.bin' % L, s)
    dump('l%d_mid16.bin' % L, mid)
    dump('l%d_u1.bin' % (L + 1), s)          # 下一层输入 = 本层状态
    np.float64(F[L]).tofile(os.path.join(T, 'scale%d.bin' % L))
    np.asarray(qe, np.float64).tofile(os.path.join(T, 'ln%d_e1.bin' % L))
    np.asarray(qp, np.float64).tofile(os.path.join(T, 'ln%d_p1.bin' % L))
    np.float64(me).tofile(os.path.join(T, 'm2c%d_e.bin' % L))
    np.float64(mp).tofile(os.path.join(T, 'm2c%d_p.bin' % L))
    rows.append((L, F[L], s_rms, s_max, me, ee, mp, ep, id_err))
    print('L%2d F=%6d s_rms=%.3e s_max=%.3e | m2c_e=%.2e err_e=%.1e | m2c_p=%.2e err_p=%.1e | id=%.1e'
          % (L, F[L], s_rms, s_max, me, ee, mp, ep, id_err))

# ---- 汇总校验 ----
dump('l0_u1.bin', EMB / F[0])            # lay0 C 代码输入 = embed4/F0 (残差 gauge F0)
sr = [r[2] for r in rows]; sm = [r[3] for r in rows]; idm = max(r[8] for r in rows)
me2 = min(r[4] for r in rows)
print('\nF0..25 =', [int(F[L]) for L in range(26)])
print('s rms: min=%.3e max=%.3e | s max: min=%.3e max=%.3e' % (min(sr), max(sr), min(sm), max(sm)))
print('identity(自适应 gauge) worst = %.1e  (须 ~1e-4 量级)' % idm)
print('E1 拟合域 m2c_e 最小 = %.2e  (>2.4e-4 则 A 预缩休眠; L0 输入 max=%.4f<0.5)'
      % (me2, np.abs(EMB / F[0]).max()))
