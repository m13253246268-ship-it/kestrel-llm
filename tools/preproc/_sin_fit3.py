# _sin_fit3.py - sin+cos 多项式 + 倍角（无 √）
import numpy as np

D = 1 << 60
q0 = 1152921504606830593
frac = 62
P_small = 1 << frac
mbit = 4

def fit(fn, lo, hi, deg, npts=400000):
    xs = np.linspace(lo, hi, npts)
    ys = fn(xs)
    V = np.vander(xs, deg + 1, increasing=True)
    coef, *_ = np.linalg.lstsq(V, ys, rcond=None)
    return np.max(np.abs(V @ coef - ys)), coef

lo, hi = -2.16, 1.77
es, cs = fit(np.sin, lo, hi, 9)
ec, cc = fit(np.cos, lo, hi, 9)
print(f"sin9次 err={es:.2e}  cos9次 err={ec:.2e}")

# 端到端
z = 0.0928
m = D * z
rng = np.random.default_rng(1)
Ks = rng.integers(-5, 5, size=500).astype(float)
plain = m + q0 * Ks

c1 = 2 * np.pi / q0
A1 = round(c1 * P_small)
t1 = plain * A1 / P_small / (1 << mbit)   # θ_in = 2π·明文/(q0·16)
print(f"θ_in 范围 [{t1.min():.2f}, {t1.max():.2f}]")

s = np.polynomial.polynomial.polyval(t1, cs)
c = np.polynomial.polynomial.polyval(t1, cc)
for _ in range(mbit):
    s, c = 2*s*c, 2*c*c - 1
ref = np.sin(2 * np.pi * plain / q0)
print(f"倍角后 sin(2π·明文/q0) max err={np.max(np.abs(s - ref)):.2e}")

A3 = round(q0 / (2 * np.pi))
out = s * A3
print(f"乘回后 out/Δ 均值 {np.mean(out/D):.4f} vs z={z}，rel err={np.max(np.abs(out-m)/np.abs(m)):.2e}")

print("\nsin9 系数:", [f"{x:.9e}" for x in cs])
print("cos9 系数:", [f"{x:.9e}" for x in cc])
