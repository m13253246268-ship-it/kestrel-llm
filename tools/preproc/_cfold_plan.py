#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# _cfold_plan.py - 真实C折叠定案：引擎同构折叠前向 + RECIP 重拟合 + E2E drift
# 折叠语义与 t23_m3p.c C 段一致：
#   score 槽 = (q·k/√d) [链内消息已隐式除 F=4096²]
#   num = EXP_Q(score/F)（引擎对全 s 求 exp，再 causal mask）
#   den = Σ_{s≤t} num；inv = recip(den)；p = num·inv
import numpy as np, math, os
from safetensors.torch import safe_open
SRC = r'Modl/Qwen3-VL-2B-Instruct/model.safetensors'
OUT = '.tmp_tok/tail'
NQ, NKV, HD = 16, 8, 128
EPS = 1e-6
f = safe_open(SRC, framework='pt')
def gt(n): return f.get_tensor(n).float().numpy()
def rmsnorm(xrow, w):
    return xrow / np.sqrt((xrow ** 2).mean() + EPS) * w
def LW(L):
    P = 'model.language_model.layers.%d.' % L
    return {k: gt(P + p) for k, p in dict(Wq='self_attn.q_proj.weight', Wk='self_attn.k_proj.weight',
        Wv='self_attn.v_proj.weight', Wo='self_attn.o_proj.weight', Wg='mlp.gate_proj.weight',
        Wu='mlp.up_proj.weight', Wd='mlp.down_proj.weight', ln1='input_layernorm.weight',
        ln2='post_attention_layernorm.weight').items()}

EXP_Q = np.array([0.0013345331468413693, 0.0084199656748899163, 0.041687167024342969,
                  0.16665574107235029, 0.49999829447852773, 1.0000003483277793, 1.0000000191319709])
def polyval_desc(c, x):
    return np.polynomial.polynomial.polyval(x, c[::-1])

def recip_fit(lo, hi, deg=8):
    from numpy.polynomial import Chebyshev, Polynomial
    n = 4096
    u = np.cos(np.pi * (np.arange(n) + 0.5) / n)
    den = (u + 1) / 2 * (hi - lo) + lo
    p = Chebyshev.fit(u, 1.0 / den, deg, domain=[-1, 1]).convert(kind=Polynomial)
    c_asc = p.coef
    v = np.linspace(-1, 1, 100001); d = (v + 1) / 2 * (hi - lo) + lo
    e = np.abs(np.polynomial.polynomial.polyval(v, c_asc) - 1.0 / d).max()
    return c_asc[::-1], 2.0 / (hi - lo), (hi + lo) / (hi - lo), e

def attn_p(qh, kh, F, RECIP_C=None, c1=None, c2=None, dbg=False):
    """引擎同构：score/√d 再 /F -> EXP_Q -> (全s) -> 对 t 行 causal mask -> den -> recip -> p"""
    p = np.zeros((4, NQ, 4)); num_all = np.zeros((4, NQ, 4))
    den_rows = np.zeros((4, NQ))
    for t in range(4):
        for h in range(NQ):
            sc = (qh[t, h].astype(np.float64) @ kh[:, h % NKV].astype(np.float64).T) / math.sqrt(HD)
            num_all[t, h, :] = polyval_desc(EXP_Q, sc / F)
    for t in range(4):
        for h in range(NQ):
            num = num_all[t, h, :].copy(); num[t + 1:] = 0.0
            den = num.sum(); den_rows[t, h] = den
            if RECIP_C is None:
                inv = 1.0 / den
            else:
                z = den * c1 - c2
                inv = polyval_desc(RECIP_C, z)
            p[t, h, :] = num * inv
    if dbg:
        print('   den rows: min=%.4f max=%.4f  (t0 avg=%.4f t3 avg=%.4f)' % (
            den_rows.min(), den_rows.max(), den_rows[0].mean(), den_rows[3].mean()))
    return p

def folded_fwd(y, W, F, RECIP_C, c1, c2, want_ao=False):
    xn = np.stack([rmsnorm(y[t], W['ln1']) for t in range(4)])
    q = xn @ W['Wq'].T; k = xn @ W['Wk'].T; v = xn @ W['Wv'].T
    qh = q.reshape(4, NQ, HD); kh = k.reshape(4, NKV, HD); vh = v.reshape(4, NKV, HD)
    p = attn_p(qh, kh, F, RECIP_C, c1, c2)
    ao = np.zeros((4, NQ, HD))
    for t in range(4):
        for h in range(NQ):
            ao[t, h, :] = p[t, h, :] @ vh[:, h % NKV, :].astype(np.float64)
    o = ao.reshape(4, 2048) @ W['Wo'].T
    xa = y + o
    x2 = np.stack([rmsnorm(xa[t], W['ln2']) for t in range(4)])
    gate = x2 @ W['Wg'].T; up = x2 @ W['Wu'].T
    act = gate * (1.0 / (1.0 + np.exp(-gate))) * up
    y2 = xa + act @ W['Wd'].T
    return (y2, ao) if want_ao else y2

def exact_fwd(y, W, want_ao=False):
    xn = np.stack([rmsnorm(y[t], W['ln1']) for t in range(4)])
    q = xn @ W['Wq'].T; k = xn @ W['Wk'].T; v = xn @ W['Wv'].T
    qh = q.reshape(4, NQ, HD); kh = k.reshape(4, NKV, HD); vh = v.reshape(4, NKV, HD)
    ao = np.zeros((4, NQ, HD))
    for t in range(4):
        for h in range(NQ):
            sc = (qh[t, h].astype(np.float64) @ kh[:t + 1, h % NKV].astype(np.float64).T) / math.sqrt(HD)
            e = np.exp(sc - sc.max()); p = e / e.sum()
            ao[t, h, :] = p @ vh[:t + 1, h % NKV, :].astype(np.float64)
    o = ao.reshape(4, 2048) @ W['Wo'].T
    xa = y + o
    x2 = np.stack([rmsnorm(xa[t], W['ln2']) for t in range(4)])
    gate = x2 @ W['Wg'].T; up = x2 @ W['Wu'].T
    act = gate * (1.0 / (1.0 + np.exp(-gate))) * up
    y2 = xa + act @ W['Wd'].T
    return (y2, ao) if want_ao else y2

def logits_of(y27, nw, emb64):
    xf = np.stack([rmsnorm(y27[t], nw) for t in range(4)])
    return xf @ emb64.T

y25 = np.load(os.path.join(OUT, 'y25.npy')).astype(np.float32)
W26 = LW(26); W27 = LW(27)
nw = gt('model.language_model.norm.weight')
emb64 = gt('model.language_model.embed_tokens.weight').astype(np.float64)

y26t, ao26t = exact_fwd(y25, W26, want_ao=True)
y27t, ao27t = exact_fwd(y26t, W27, want_ao=True)
base = logits_of(y27t, nw, emb64)
top1 = base.argmax(axis=1)
mm = np.sort(base[0])[::-1]; print('exact baseline top1 margin=%.3f' % (mm[0] - mm[1]))

# ref 文件偏差对照（协议记录）
ref26 = np.fromfile(os.path.join(OUT, 'l26_attn_out.bin'), dtype=np.float32).reshape(4, 2048)
ref27 = np.fromfile(os.path.join(OUT, 'l27_attn_out.bin'), dtype=np.float32).reshape(4, 2048)
print('true-ao vs ref bin: L26 max|d|=%.3f  L27 max|d|=%.3f' % (
    np.abs(ao26t.reshape(4, 2048) - ref26).max(), np.abs(ao27t.reshape(4, 2048) - ref27).max()))

# ---- RECIP 重拟合 ----
print('\n=== RECIP 重拟合（causal den 域, 折叠 F 参数） ===')
for F in (4096.0 * 4096.0, 22500.0, 4096.0):
    # 引擎折叠 p 之 den 域（用真实 l26 输入，EXP_Q 在折叠 score 上）
    xn = np.stack([rmsnorm(y25[t], W26['ln1']) for t in range(4)])
    qh = (xn @ W26['Wq'].T).reshape(4, NQ, HD); kh = (xn @ W26['Wk'].T).reshape(4, NKV, HD)
    denmin, denmax = 9e9, -9e9
    for t in range(4):
        for h in range(NQ):
            sc = (qh[t, h].astype(np.float64) @ kh[:, h % NKV].astype(np.float64).T) / math.sqrt(HD)
            num = polyval_desc(EXP_Q, sc / F).copy(); num[t + 1:] = 0
            d0 = num.sum(); denmin = min(denmin, d0); denmax = max(denmax, d0)
    print('F=%.4g: den range [%.4f, %.4f]' % (F, denmin, denmax))
    lo, hi = max(0.6, denmin * 0.9), denmax * 1.05
    c, c1, c2, e = recip_fit(lo, hi)
    print('  -> fit domain [%.3f, %.3f] c1=%.6f c2=%.6f worst|err|=%.2e' % (lo, hi, c1, c2, e))
    # 引擎同构折叠 p（用新 fit）E2E
    y26f, ao26f = folded_fwd(y25, W26, F, c, c1, c2, want_ao=True)
    y27f, ao27f = folded_fwd(y26f, W27, F, c, c1, c2, want_ao=True)
    lgf = logits_of(y27f, nw, emb64)
    d = np.abs(lgf - base).max(axis=1)
    print('  E2E fold: max|Δlogits|=%.3f top1chg=%s  margin0=%.2f' % (
        d.max(), lgf.argmax(axis=1) != top1, np.sort(lgf[0])[::-1][0] - np.sort(lgf[0])[::-1][1]))
    print('  protocol-note: |Δao26|max=%.2f (ao_ref bin=%.2f)  |Δao27|max=%.2f' % (
        np.abs(ao26f.reshape(4, 2048) - ref26).max(), np.abs(ao26f.reshape(4, 2048) - ao26t.reshape(4, 2048)).max(),
        np.abs(ao27f.reshape(4, 2048) - ref27).max()))
