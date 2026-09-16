# _sin_fit2.py - 分步标量折叠：×2π/q0（定点）→ ×2^-m（精确）→ sin窄区间 → 倍角
import numpy as np

D = 1 << 60
q0 = 1152921504606830593
frac = 62          # P_small ~ 2^62
P_small = 1 << frac
mbit = 4

# --- sin 窄区间多项式：θ/2^m ∈ 2π·[-5.5,4.5]/16 = [-2.16, 1.77] ---
def fit(lo, hi, deg, npts=400000):
    xs = np.linspace(lo, hi, npts)
    ys = np.sin(xs)
    V = np.vander(xs, deg + 1, increasing=True)
    coef, *_ = np.linalg.lstsq(V, ys, rcond=None)
    return np.max(np.abs(V @ coef - ys)), coef

lo, hi = -2.16, 1.77
for deg in [5, 7, 9]:
    err, c = fit(lo, hi, deg)
    print(f"sin θ/16 次数 {deg} (θ∈[{lo},{hi}]): max err={err:.2e}")

err, csin = fit(lo, hi, 9)
print("sin 9 次系数:", [f"{x:.9e}" for x in csin])

# --- √(1-x²) ---
u = np.linspace(0, 1, 400000)
V2 = np.vander(u, 7, increasing=True)
cs, *_ = np.linalg.lstsq(V2, np.sqrt(1 - u), rcond=None)
print("√6次 err:", np.max(np.abs(V2 @ cs - np.sqrt(1 - u))))
print("√6次系数:", [f"{x:.9e}" for x in cs])

# --- 端到端明文模拟 ---
z = 0.0928
m = D * z
rng = np.random.default_rng(1)
Ks = rng.integers(-5, 5, size=500).astype(float)   # 槽位域混叠 K（代数嵌入整数）
plain = m + q0 * Ks    # 明文（槽位：z + q0·K）

# 1. ×c = 2π/q0（定点，5 位）
c1 = 2 * np.pi / q0
A1 = round(c1 * P_small)
t1 = plain * A1 / P_small
print(f"\n×2π/q0: A1={A1}（{A1.bit_length()} 位），θ=明文×c1 误差={np.max(np.abs(t1 - plain*c1)/np.abs(plain*c1)):.2e}")

# 2. ×2^-m（精确）
t2 = t1 / (1 << mbit)
theta_in = t2    # = 2π·明文/(q0·16)
print(f"×2^-4: θ_in 范围 [{theta_in.min():.2f}, {theta_in.max():.2f}]（应 ⊂ [-2.16,1.77]）")

# 3. sin 多项式 + cos + 倍角 mbit 次
s = np.polynomial.polynomial.polyval(theta_in, csin)
cval = np.polynomial.polynomial.polyval(1 - s*s, cs)
for _ in range(mbit):
    s2 = 2 * s * cval
    c2 = 2 * cval * cval - 1
    s, cval = s2, c2
# s = sin(16·θ_in) = sin(2π·明文/q0)
ref = np.sin(2 * np.pi * plain / q0)
print(f"倍角后 sin(2π·明文/q0)：max err={np.max(np.abs(s - ref)):.2e}")

# 4. 乘回 q0/2π（raw）：槽位 × q0/2π
A3 = round(q0 / (2 * np.pi))
out = s * A3
m_ref = np.sin(2 * np.pi * plain / q0) * q0 / (2 * np.pi)
print(f"乘回 q0/2π: A3={A3}，out vs Δz*...  rel={np.max(np.abs(out - m)/np.abs(m)):.2e}")
print(f"  out 均值 {np.mean(out/D):.4f} vs z={z}（out/Δ 应为 z）")
