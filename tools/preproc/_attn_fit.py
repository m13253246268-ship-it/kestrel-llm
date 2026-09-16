# 拟合 attention exp 与 recip（真实 score 范围，无 max-sub）
import numpy as np
import math
def L(n): return np.fromfile(r'D:\项目\New_vLLM\.tmp_tok\l0\%s.bin' % n, dtype=np.float32)
q = L('q').reshape(4,2048); k = L('k').reshape(4,1024); v = L('v').reshape(4,1024)
NQ, NKV, HD = 16, 8, 128
qh = q.reshape(4,NQ,HD); kh = k.reshape(4,NKV,HD)
scores = np.zeros((4,NQ,4), dtype=np.float64)
for h in range(NQ):
    hk = h % NKV
    scores[:,h,:] = (qh[:,h,:] @ kh[:,hk,:].T)/math.sqrt(HD)
# 无 max-sub 的 softmax 需要 exp 覆盖 [min_s, max_s]
smin, smax = scores.min(), scores.max()
print('score range [%.4f, %.4f]' % (smin, smax))

# --- exp 拟合：S in [lo, hi]，直接 fit（窄区间，deg-6 足够） ---
lo, hi = -0.6, 0.5
u = np.linspace(lo, hi, 200001)
z = u  # 直接对 S 拟合（无平移，系数小）
y = np.exp(z)
for deg in (5, 6):
    c = np.polyfit(z, y, deg)
    e = np.abs(np.polyval(c, z) - y).max()
    print('exp deg=%d max_abs_err=%.3e |c|max=%.3f (y range [%.4f,%.4f])' % (deg, e, np.abs(c).max(), y.min(), y.max()))
deg=6; c_exp = np.polyfit(z, y, deg)
# --- recip 拟合：den=sum_s exp(S(h,s)) in [4*exp(lo), 4*exp(hi)] ---
dlo, dhi = 4*math.exp(lo), 4*math.exp(hi)
print('den range [%.4f, %.4f]' % (dlo, dhi))
c0 = 0.5*(dlo+dhi); s0 = 0.5*(dhi-dlo)
w = np.linspace(dlo, dhi, 200001)
zw = (w - c0)/s0
yw = 1.0/w
for deg in (6, 8):
    cw = np.polyfit(zw, yw, deg)
    ew = np.abs(np.polyval(cw, zw) - yw).max()
    print('recip deg=%d max_abs_err=%.3e (rel %.3e) |c|max=%.3f' % (deg, ew, ew/yw.max(), np.abs(cw).max()))
degw=8; c_recip = np.polyfit(zw, yw, degw)
print('RECIP z=(den-%.10g)/%.10g' % (c0, s0))

print('const double EXP_Q[7] = {'); [print('    %.17g,' % v) for v in c_exp]; print('};')
print('const double RECIP_Q[9] = {'); [print('    %.17g,' % v) for v in c_recip]; print('};')
# 验证 horner
e1 = np.abs(np.polyval(c_exp, u) - np.exp(u)).max()
print('exp horner err %.3e' % e1)
e2 = np.abs(np.polyval(c_recip, zw) - 1.0/w).max()
print('recip horner err %.3e' % e2)
# p 值域与 attn_out
smax2 = scores.max(axis=2, keepdims=True)
e = np.exp(scores - smax2)
p = e/e.sum(axis=2, keepdims=True)
print('p range [%.4f, %.4f]' % (p.min(), p.max()))
