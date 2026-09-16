#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# _logits_tol.py - (b) E2E logits 容差分析：silu/act 相对误差 σ → 最终 logits/top1 稳定性
# 方法：真值 y25(.npy) 起，layer26 act 注入相对误差 → y26' → fwd(27) → final norm → lm_head
# 扫描 σ；输出每 token top1 变化 / top1 余量 / max|Δlogits|
import numpy as np, math, os
from safetensors.torch import safe_open

SRC = r'Modl/Qwen3-VL-2B-Instruct/model.safetensors'
OUT = '.tmp_tok/tail'
F = 1024.0
NQ, NKV, HD = 16, 8, 128
EPS = 1e-6
f = safe_open(SRC, framework='pt')

def rmsnorm(xrow, w):
    r = np.sqrt((xrow ** 2).mean() + EPS)
    return xrow / r * w

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

def fwd_layer(L, y, W, act26err=None, seed=0):
    xn = np.stack([rmsnorm(y[t], W['ln1']) for t in range(4)])
    q = xn @ W['Wq'].T; k = xn @ W['Wk'].T; v = xn @ W['Wv'].T
    qh = q.reshape(4, NQ, HD); kh = k.reshape(4, NKV, HD); vh = v.reshape(4, NKV, HD)
    sc = np.zeros((4, NQ, 4))
    for h in range(NQ):
        sc[:, h, :] = (qh[:, h, :] @ kh[:, h % NKV, :].T) / math.sqrt(HD)
    for t in range(4):
        for s in range(t + 1, 4):
            sc[t, :, s] = -1e9
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
    sil = gate * (1.0 / (1.0 + np.exp(-gate)))
    act = sil * up
    if act26err is not None:
        rng = np.random.RandomState(seed)
        act = act * (1.0 + act26err * rng.randn(*act.shape))
    mlp = act @ W['Wd'].T
    return xa + mlp

y25 = np.load(os.path.join(OUT, 'y25.npy'))
W26 = layer_weights(26); W27 = layer_weights(27)
nw = f.get_tensor('model.language_model.norm.weight').float().numpy().astype(np.float64)
emb = f.get_tensor('model.language_model.embed_tokens.weight').float().numpy().astype(np.float64)

def logits_of(y27):
    xf = np.stack([rmsnorm(y27[t], nw) for t in range(4)])
    return xf @ emb.T

baseline = logits_of(fwd_layer(27, fwd_layer(26, y25, W26), W27))
np.save(os.path.join(OUT, 'logits_chk.npy'), baseline)
top1 = baseline.argmax(axis=1)
for t in range(4):
    srt = np.sort(baseline[t])[::-1]
    print('  base t%d top1=%d margin=%.3f' % (t, top1[t], srt[0] - srt[1]))

print('sigma  token  top1_chg  max|dlogits|  rms|dlogits|')
for sig in [0.01, 0.03, 0.05, 0.1, 0.2, 0.3]:
    worst = 0
    for seed in range(3):
        y26p = fwd_layer(26, y25, W26, act26err=sig, seed=seed)
        lg = logits_of(fwd_layer(27, y26p, W27))
        d = np.abs(lg - baseline).max(axis=1)
        chg = (lg.argmax(axis=1) != top1)
        r = np.sqrt(((lg - baseline) ** 2).mean(axis=1))
        for t in range(4):
            if chg[t]:
                print('  %.2f  t%d   TOP1-CHANGE %d  maxd=%.4g rmsd=%.4g' % (sig, t, seed, d[t], r[t]))
                worst = 1
    if worst == 0:
        print('  %.2f  all   top1 stable (3 seeds); max|dlogits| over t = %.4g' %
              (sig, np.max([np.abs(logits_of(fwd_layer(27, fwd_layer(26, y25, W26, act26err=sig, seed=s), W27)) - baseline).max() for s in range(3)])))
