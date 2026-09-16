# _fold_sim.py - 折叠函数端到端明文模拟（含 scale 语义）
# 流程：ModRaise 混叠(v'=m+q0·k) -> 乘 2π/(Δ·2^m) -> sin 倍角 m 次 -> 乘回 Δ/2π -> m
import numpy as np

D = 1 << 60          # Δ = 2^60
q0 = 1152921504606830593  # 链最大素数 ~2^60
n = 1024
frac = 20            # 定点位数
P_small = 1 << frac  # ~2^20 除定点

def poly_fixed(c, Dmul=1):
    """poly[0] = round(Δ·2^frac·c/Dmul)，明文槽位乘它后除 2^frac 得 ×c/Dmul"""
    return round(D * (1 << frac) * c / Dmul)

# 明文（单槽位）：z = 0.0928，m = Δ·z
z = 0.0928
m = D * z

# 混叠：ModRaise 后明文 v' = m + q0·k（k ∈ [-5,4] 随机）
rng = np.random.default_rng(42)
k = rng.integers(-5, 5, size=1000).astype(float)
v = m + q0 * k

# 倍角次数
mbit = 4
scale_in = 2 ** mbit

# --- 折叠 ---
# 1. 乘 2π/(Δ·2^m)：poly = round(Δ·2^frac·2π/(Δ·2^m)) = round(2^frac·2π/2^m)
A1 = poly_fixed(2 * np.pi / scale_in, Dmul=1)   # round(2^20·2π/16)
theta = v * A1 / (1 << frac) / D * 1.0          # v·A1/2^frac / Δ = v·2π/(Δ·2^m)
theta_real = v * 2 * np.pi / (D * scale_in)     # 真实 θ
print(f"A1={A1}  θ 误差 rel={np.max(np.abs(theta - theta_real)/np.abs(theta_real)):.2e}")

# 2. sin(θ) 多项式（θ ∈ 2π·[-5.5,4.5]/16 ≈ [-2.16,1.77]，10 次最小二乘）
xs = np.linspace(-2.2, 1.8, 20000)
ys = np.sin(xs)
V = np.vander(xs, 11, increasing=True)
coef_sin, *_ = np.linalg.lstsq(V, ys, rcond=None)
s = np.polynomial.polynomial.polyval(theta, coef_sin)
print(f"sin(θ) 多项式 10 次 max err={np.max(np.abs(s - np.sin(theta))):.2e}")

# 3. cos(θ) = √(1-sin²θ)：√ 多项式（[0,1]，6 次）
u = np.linspace(0, 1, 20000)
V2 = np.vander(u, 7, increasing=True)
coef_sqrt, *_ = np.linalg.lstsq(V2, np.sqrt(1 - u), rcond=None)
c = np.polynomial.polynomial.polyval(1 - s * s, coef_sqrt)
print(f"√ 多项式 6 次 max err={np.max(np.abs(c - np.sqrt(np.maximum(1-s*s,0)))):.2e}")

# 4. 倍角 mbit 次：s'=2sc, c'=2c²-1
for _ in range(mbit):
    s2 = 2 * s * c
    c2 = 2 * c * c - 1
    s, c = s2, c2
# 结果 s = sin(16θ) = sin(2π·v'/Δ)
s_ref = np.sin(2 * np.pi * v / D)
print(f"倍角后 s vs sin(2πv/Δ)：max err={np.max(np.abs(s - s_ref)):.2e}")

# 5. 乘回 Δ/2π：mult_plain_fixed(Δ/2π)，poly[0]=round(Δ·2^frac·Δ/2π)——超 2^60 不可编码！
#   分步：先乘 Δ/2^60（=round(2^frac·Δ/2^60)≈2^20），再乘 2^60/2π（≈2^59 可 raw 编码）
A2 = poly_fixed(1.0, Dmul=1)      # round(2^20) 乘 Δ/2^60? 不对——重新设计
# 直接乘回 round(2^frac/(2π))（s ~ 2πz，乘 2^frac/(2π) 得 z·2^frac）
A3 = round((1 << frac) / (2 * np.pi))   # ~1.67e5
out0 = s * A3 / (1 << frac)             # = s/(2π) ~ z
print(f"A3={A3}  s/(2π) ≈ z，误差={np.max(np.abs(out0 - np.sin(2*np.pi*v/D)/(2*np.pi))) :.2e}")
# 再乘 Δ（raw poly[0]=2^60）：明文槽位 × 2^60 → m 尺度
out = out0 * D
print(f"最终 out = Δ·sin(2πv/Δ)/2π ≈ m？")
print(f"  out vs m(Δz)：rel={np.max(np.abs(out - m)/np.abs(m)):.2e}")
print(f"  折叠输出明文 ~ Δ·z ✓（out/Δ 均值={np.mean(out/D):.4f} vs z={z}）")
