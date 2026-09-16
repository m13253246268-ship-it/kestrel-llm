#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# _full_layers.py - 推满 28 层：生成 L0..L25 每层 causal 明文参考 + LNQ/m2c 拟合 + 折叠域统计
# 语义与 _tail_ref.py 同款：F=1024、causal、4-token、输入 embed4；L26/27/final 参考已在 tail
# 输出 tail/l{L}_*（参考全套，命名同 l26/l27）+ tail/ln{L}_e1/p1.bin + m2c{L}_e/p.bin + 统计表
import numpy as np, math, os
from safetensors.torch import safe_open

SRC = r'Modl/Qwen3-VL-2B-Instruct/model.safetensors'
D0 = '.tmp_tok/l0'
OUT = '.tmp_tok/tail'
os.makedirs(OUT, exist_ok=True)
F = 1024.0
x = np.fromfile(D0 + '/embed4.bin', dtype=np.float32).reshape(4, 2048).astype(np.float64)
f = safe_open(SRC, framework='pt')
NQ, NKV, HD = 16, 8, 128
EPS = 1e-6

def rmsnorm(xrow, w):
    r = np.sqrt((xrow ** 2).mean() + EPS)
    return xrow / r * w

def dump(name, a): a.astype(np.float32).tofile(os.path.join(OUT, name + '.bin'))

def fit_lnq(uarr, tag):
    m2s = [(uarr[t] ** 2).mean() for t in range(4)]
    m2c = float(np.mean(m2s))
    c0, s0 = 0.75 * m2c, 0.25 * m2c
    u = np.linspace(0.5 * m2c, 1.5 * m2c, 200001)
    z = (u - c0) / (8 * s0)
    y = 1.0 / np.sqrt(u + 1e-6) / 128.0
    q = np.polyfit(z, y, 8)
    e = np.abs(np.polyval(q, z) - y).max()
    return q, m2c, e

def layer_weights(L):
    P = 'model.language_model.layers.%d.' % L
    return dict(
        Wq=f.get_tensor(P+'self_attn.q_proj.weight').float().numpy().astype(np.float64),
        Wk=f.get_tensor(P+'self_attn.k_proj.weight').float().numpy().astype(np.float64),
        Wv=f.get_tensor(P+'self_attn.v_proj.weight').float().numpy().astype(np.float64),
        Wo=f.get_tensor(P+'self_attn.o_proj.weight').float().numpy().astype(np.float64),
        Wg=f.get_tensor(P+'mlp.gate_proj.weight').float().numpy().astype(np.float64),
        Wu=f.get_tensor(P+'mlp.up_proj.weight').float().numpy().astype(np.float64),
        Wd=f.get_tensor(P+'mlp.down_proj.weight').float().numpy().astype(np.float64),
        ln1=f.get_tensor(P+'input_layernorm.weight').float().numpy().astype(np.float64),
        ln2=f.get_tensor(P+'post_attention_layernorm.weight').float().numpy().astype(np.float64))

def fwd(L, y, tag=None, dump_ref=False):
    W = layer_weights(L)
    xn = np.stack([rmsnorm(y[t], W['ln1']) for t in range(4)])
    q = xn @ W['Wq'].T; k = xn @ W['Wk'].T; v = xn @ W['Wv'].T
    qh = q.reshape(4, NQ, HD); kh = k.reshape(4, NKV, HD); vh = v.reshape(4, NKV, HD)
    sc = np.zeros((4, NQ, 4))
    for h in range(NQ):
        sc[:, h, :] = (qh[:, h, :] @ kh[:, h % NKV, :].T) / math.sqrt(HD)
    for t in range(4):
        for s in range(t + 1, 4): sc[t, :, s] = -1e9
    sm = sc - sc.max(axis=2, keepdims=True)
    e = np.exp(sm); p = e / e.sum(axis=2, keepdims=True)
    ao = np.zeros((4, NQ, HD))
    for h in range(NQ):
        ao[:, h, :] = p[:, h, :] @ vh[:, h % NKV, :]
    o = ao.reshape(4, 2048) @ W['Wo'].T
    xa = y + o
    x2 = np.stack([rmsnorm(xa[t], W['ln2']) for t in range(4)])
    gate = x2 @ W['Wg'].T; up = x2 @ W['Wu'].T
    act = gate * (1.0 / (1.0 + np.exp(-gate))) * up
    mlp = act @ W['Wd'].T
    y2 = xa + mlp
    if dump_ref and tag is not None:
        u1 = (y / F).astype(np.float32); mid = (xa / F).astype(np.float32)
        dump(tag + '_x_norm', xn); dump(tag + '_q', q); dump(tag + '_k', k); dump(tag + '_v', v)
        dump(tag + '_scores', sc.astype(np.float32)); dump(tag + '_attn_out', ao.reshape(4, 2048))
        dump(tag + '_o', o); dump(tag + '_mid16', mid); dump(tag + '_xattn_full', xa)
        dump(tag + '_gate', gate); dump(tag + '_up', up); dump(tag + '_act', act)
        dump(tag + '_mlp_out', mlp); dump(tag + '_u1', u1)
        dump(tag + '_u2_ref', (y2 / F).astype(np.float32))
    return y2, dict(xn=xn, xa=xa, mid=xa / F, u1=y / F, y2=y2,
                    q=q, k=k, sc=sc, gate=gate, up=up)

print('embed4 |max|=%.4f' % np.abs(x).max())
y = x.copy()
meta = {}
stats = []
for L in range(28):
    y, m = fwd(L, y, tag='l%d' % L if L < 26 else None, dump_ref=(L < 26))
    meta[L] = m
    if L < 26:
        stats.append((L, np.abs(m['q']).max(), np.abs(m['k']).max(),
                      np.abs(m['sc'][m['sc'] > -1e8]).max(),
                      np.abs(m['gate']).max(), np.abs(m['up']).max(),
                      np.abs(y).max()))
        print('  L%d dump ok |y|max=%.3f' % (L, np.abs(y).max()))
print('per-layer |y|max:', ' '.join('L%d=%.1f' % (L, np.abs(meta[L]['y2']).max()) for L in range(28)))
print('L%d stats (qmax kmax scmax gatemax upmax yout):' % 0)
for s in stats:
    print('  L%d q=%.1f k=%.1f sc=%.1f gate=%.1f up=%.1f y=%.1f' % s)
# LNQ/m2c 拟合：层 L 输入域 = y_{L-1}/F（L0 用 embed4 全幅）；post 域 = mid_L
inp = x.copy()
for L in range(28):
    m = meta[L]
    if L == 0:
        qe, me, ee = fit_lnq(x / F, 'ln0_e1(embed/F)') if np.abs(x).max() / F < 1 else fit_lnq(x, 'ln0_e1(embed)')
    else:
        qe, me, ee = fit_lnq(meta[L-1]['y2'] / F, 'ln%d_e1' % L)
    qp, mp, ep = fit_lnq(m['mid'], 'ln%d_p1' % L)
    if L < 26:
        qe.astype(np.float64).tofile(os.path.join(OUT, 'ln%d_e1.bin' % L))
        qp.astype(np.float64).tofile(os.path.join(OUT, 'ln%d_p1.bin' % L))
        np.float64(me).tofile(os.path.join(OUT, 'm2c%d_e.bin' % L))
        np.float64(mp).tofile(os.path.join(OUT, 'm2c%d_p.bin' % L))
        print('  L%d fit e: m2c=%.3e err=%.2e | p: m2c=%.3e err=%.2e' % (L, me, ee, mp, ep))
print('DONE')
