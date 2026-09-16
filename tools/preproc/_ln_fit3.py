# 拟合 post-LN（x_attn）的 1/sqrt 多项式：z'=(E2-c0)/(8s0)，c0=0.75*m2c, s0=0.25*m2c
import numpy as np
def L(n): return np.fromfile(r'D:\项目\New_vLLM\.tmp_tok\l0\%s.bin' % n, dtype=np.float32)
DIM=2048; L4=4
x = L('embed4').reshape(L4,DIM)
xattn = L('x_attn').reshape(L4,DIM)
for tag, arr in (('embed', x), ('x_attn', xattn)):
    m2s = [(arr[t].astype(np.float64)**2).mean() for t in range(L4)]
    print('%s m2 per token: %s' % (tag, ['%.6f'%v for v in m2s]))
    m2c = float(np.mean(m2s))
    lo, hi = 0.5*m2c, 1.5*m2c
    eps = 1e-6
    u = np.linspace(lo, hi, 200001)
    c0, s0 = 0.75*m2c, 0.25*m2c
    z = (u - c0) / (8*s0)
    y = 1.0/np.sqrt(u+eps)/128.0
    q = np.polyfit(z, y, 8)
    e = np.abs(np.polyval(q, z) - y).max()
    print('  m2c=%.10g range[%.4g,%.4g] deg8 err=%.3e |Q|max=%.3f' % (m2c, lo, hi, e, np.abs(q).max()))
    # 验证各 token 位置
    for t in range(L4):
        zt = (m2s[t]-c0)/(8*s0)
        val = np.polyval(q, zt)*128
        print('    token%d z=%.4f poly*128=%.6f true=%.6f' % (t, zt, val, 1.0/np.sqrt(m2s[t]+eps)))
    if tag == 'embed':
        print('LN_Q_embed = [%s]' % ', '.join('%.17g'%v for v in q))
    else:
        print('LN_Q_post = [%s]' % ', '.join('%.17g'%v for v in q))
