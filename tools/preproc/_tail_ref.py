#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# _tail_ref.py - M3a 明文基线：28 层真实 causal 前向 → 输出端参考（fold F=1024）
# 语义：因果 attention（token t 只见 0..t）；统一 fold F 使 u_n = y_n/F 且 F >= 2*max|y25..27|
# 密文窗口 = layer26（u-form L1，输入 y25/F）→ layer27 → final RMSNorm → lm_head logits
# 输出（.tmp_tok/tail/）：l26_* / l27_*（各阶段，x_norm..u2_ref/u1 命名同 t23 对照约定）
#   xnorm_f.bin logits.bin（final norm 输出与 lm_head logits 全 vocab）
#   ln*_e1/ln*_p1/ln_f.bin + m2c*.bin（u-form 域 LNQ 拟合，deg8 高->低，y=1/sqrt(u)/128）
import numpy as np, math, os
from safetensors.torch import safe_open

SRC = r'Modl/Qwen3-VL-2B-Instruct/model.safetensors'
D0 = '.tmp_tok/l0'
OUT = '.tmp_tok/tail'
os.makedirs(OUT, exist_ok=True)
F = 1024.0  # 统一 fold（y25 max~253 → u<0.25；y27 max~309 → u<0.31）
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
    print('  %s m2c=%.10g deg8 err=%.3e' % (tag, m2c, e))
    return q, m2c

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

def fwd(L, y, causal=True, tag=None, dump_ref=False):
    W = layer_weights(L)
    xn = np.stack([rmsnorm(y[t], W['ln1']) for t in range(4)])
    q = xn @ W['Wq'].T; k = xn @ W['Wk'].T; v = xn @ W['Wv'].T
    qh = q.reshape(4, NQ, HD); kh = k.reshape(4, NKV, HD); vh = v.reshape(4, NKV, HD)
    sc = np.zeros((4, NQ, 4))
    for h in range(NQ):
        sc[:, h, :] = (qh[:, h, :] @ kh[:, h % NKV, :].T) / math.sqrt(HD)
    if causal:
        for t in range(4):
            for s in range(t + 1, 4): sc[t, :, s] = -1e9
    sm = sc - sc.max(axis=2, keepdims=True)
    e = np.exp(sm)
    p = e / e.sum(axis=2, keepdims=True)
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
        u1 = (y / F).astype(np.float32)
        mid = (xa / F).astype(np.float32)
        dump(tag + '_x_norm', xn); dump(tag + '_q', q); dump(tag + '_k', k); dump(tag + '_v', v)
        dump(tag + '_scores', sc.astype(np.float32))
        dump(tag + '_attn_out', ao.reshape(4, 2048))
        dump(tag + '_o', o); dump(tag + '_mid16', mid)
        dump(tag + '_xattn_full', xa)
        dump(tag + '_gate', gate); dump(tag + '_up', up); dump(tag + '_act', act)
        dump(tag + '_mlp_out', mlp)
        dump(tag + '_u1', u1)
        dump(tag + '_u2_ref', (y2 / F).astype(np.float32))
        np.save(os.path.join(OUT, tag + '_u1.npy'), y / F)
        np.save(os.path.join(OUT, tag + '_mid16.npy'), xa / F)
        np.save(os.path.join(OUT, tag + '_u2_ref.npy'), y2 / F)
    return y2, dict(xn=xn, xa=xa, mid=xa / F, u1=y / F, y2=y2)

# ---- 逐层 causal 前向 0..27 ----
y = x.copy()
ymax = {}
for L in range(28):
    y, meta = fwd(L, y, causal=True)
    ymax[L] = np.abs(y).max()
print('per-layer |y|max:', ' '.join('L%d=%.1f' % (L, ymax[L]) for L in range(28)))
if ymax[27] / F > 0.5:
    print('!! F=%g too small: y27/F=%.3f > 0.5' % (F, ymax[27] / F))

# ---- 窗口 layer26 / layer27 参考（输入重新从 y25 起，保持 dump）----
# 直接重跑 26/27 以拿齐 dump（y25 已在上循环释放，重新快速前向到 25）
y = x.copy()
for L in range(26):
    y, _ = fwd(L, y, causal=True)
y25 = y.copy()
y26, m26 = fwd(26, y25, causal=True, tag='l26', dump_ref=True)
y27, m27 = fwd(27, y26, causal=True, tag='l27', dump_ref=True)
np.save(os.path.join(OUT, 'y25.npy'), y25)
print('window: |y25|max %.3f |y26|max %.3f |y27|max %.3f' % (
    np.abs(y25).max(), np.abs(y26).max(), np.abs(y27).max()))
print('u26 max %.4f u27 max %.4f (must <0.5)' % (
    np.abs(y26).max() / F, np.abs(y27).max() / F))

# ---- final norm + lm_head（绑定 embed_tokens）----
nw = f.get_tensor('model.language_model.norm.weight').float().numpy().astype(np.float64)
emb = f.get_tensor('model.language_model.embed_tokens.weight').float().numpy().astype(np.float64)
xf = np.stack([rmsnorm(y27[t], nw) for t in range(4)])
logits = xf @ emb.T
dump('xnorm_f', xf)
dump('logits', logits)
print('final xnorm |max| %.4f; logits [%.3f,%.3f]; top1 t0=%.3f' % (
    np.abs(xf).max(), logits.min(), logits.max(), np.sort(logits[0])[::-1][0]))
np.save(os.path.join(OUT, 'logits.npy'), logits)

# ---- u-form 域拟合：layer26 in(u25) post(mid26)；layer27 in(u26) post(mid27)；final in(u27) ----
u25 = y25 / F
qe, me = fit_lnq(u25, 'ln26_e1(u25)')
qp, mp = fit_lnq(m26['mid'], 'ln26_p1(mid26)')
qe.astype(np.float64).tofile(os.path.join(OUT, 'ln26_e1.bin'))
qp.astype(np.float64).tofile(os.path.join(OUT, 'ln26_p1.bin'))
np.float64(me).tofile(os.path.join(OUT, 'm2c26_e.bin'))
np.float64(mp).tofile(os.path.join(OUT, 'm2c26_p.bin'))
qe2, me2 = fit_lnq(m26['u1'], 'ln27_e1(u26)')
qp2, mp2 = fit_lnq(m27['mid'], 'ln27_p1(mid27)')
qe2.astype(np.float64).tofile(os.path.join(OUT, 'ln27_e1.bin'))
qp2.astype(np.float64).tofile(os.path.join(OUT, 'ln27_p1.bin'))
np.float64(me2).tofile(os.path.join(OUT, 'm2c27_e.bin'))
np.float64(mp2).tofile(os.path.join(OUT, 'm2c27_p.bin'))
qef, mef = fit_lnq(y27 / F, 'ln_f(u27)')
qef.astype(np.float64).tofile(os.path.join(OUT, 'ln_f.bin'))
np.float64(mef).tofile(os.path.join(OUT, 'm2c_f.bin'))
print('DONE')
