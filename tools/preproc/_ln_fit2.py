# 拟合 layer0 RMSNorm 的 1/sqrt 多项式：z'=(E2-0.75*m2)/(2*m2)，目标 y=inv_std/128
import numpy as np

def loadf(name, cnt):
    with open(r'D:\项目\New_vLLM\.tmp_tok\l0\%s.bin' % name, 'rb') as f:
        return np.frombuffer(f.read(4*cnt), dtype='<f4').astype(np.float64)

DIM = 2048
L = 4
x = loadf('embed4', L*DIM)
g1 = loadf('in_ln', DIM)

# 每 token 的 m2 = E[x^2]（RMSNorm 无均值），与 C 代码一致（token0）
for t in range(L):
    seg = x[t*DIM:(t+1)*DIM]
    m2 = float((seg**2).sum() / DIM)
    print('token%d m2=%.10f inv_std=%.6f' % (t, m2, 1.0/np.sqrt(m2 + 1e-6)))

# 拟合中心用 token0 的 m2（C 代码用 sumx2_ref/DIM）
m2 = float((x[0:DIM]**2).sum() / DIM)
eps = 1e-6
# 覆盖范围：E2 in [0.5*m2, 1.5*m2]（跨 token 富余）
lo, hi = 0.5*m2, 1.5*m2
c, s = 0.75*m2, 0.25*m2           # 代码映射 z'=(E2-c)/(8s) 的分母 8s=2*m2
u = np.linspace(lo, hi, 200001)
z = (u - c) / (8*s)               # z' 映射（与代码一致）
y = 1.0 / np.sqrt(u + eps) / 128.0

for deg in (8, 10):
    q = np.polyfit(z, y, deg)
    e = np.abs(np.polyval(q, z) - y).max()
    print('deg=%d max_abs_err=%.3e (vs y~%.3f) |Q|max=%.3f' % (deg, e, y.mean(), np.abs(q).max()))

deg = 8
q = np.polyfit(z, y, deg)
print('z range: [%.4f, %.4f]' % (z.min(), z.max()))
print('// LN_Q for z=(E2-%.10g)/(%.10g), y=inv_std/128' % (c, 8*s))
print('static const double LN_Q[9] = {')
for v in q:
    print('    %.17g,' % v)
print('};')
acc = np.polyval(q, z)
e = np.abs(acc - y).max()
print('horner max_abs_err=%.3e' % e)
# 在 z'=0.125（token0 的位置）处验证
z0 = 0.125
val = np.polyval(q, z0)
print('at z=0.125 (token0): poly*128=%.6f true=%.6f' % (val*128, 1.0/np.sqrt(m2+eps)))
