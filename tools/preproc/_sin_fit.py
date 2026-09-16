# _sin_fit.py - sin 折叠多项式：θ = 2π·明文/q_0（混叠 K∈[-5,4] 周期吸收）
# m=2 倍角：θ_in = 2π·明文/(q_0·4) ∈ 2π·[-5.5,4.5]/4 = [-8.64, 7.07]
import numpy as np

def fit_sin(lo, hi, deg, npts=200000):
    xs = np.linspace(lo, hi, npts)
    ys = np.sin(xs)
    V = np.vander(xs, deg + 1, increasing=True)
    coef, *_ = np.linalg.lstsq(V, ys, rcond=None)
    err = np.max(np.abs(V @ coef - ys))
    return err, coef

print("方案 A（m=2 倍角）：sin 输入 θ ∈ [-8.64, 7.07]（~2.5 周期）")
for deg in [7, 9, 11, 13, 15]:
    err, c = fit_sin(-8.64, 7.07, deg)
    print(f"  次数 {deg}: max err = {err:.2e}")

print("\n方案 B（m=0 无倍角）：sin 输入 θ ∈ [-34.6, 28.3]（5.5 周期）")
for deg in [17, 21, 25]:
    err, c = fit_sin(-34.6, 28.3, deg)
    print(f"  次数 {deg}: max err = {err:.2e}")

print("\n√(1-x²) 多项式（cos 用），x²∈[0,1]")
u = np.linspace(0, 1, 200000)
V2 = np.vander(u, 7, increasing=True)
coef_sqrt, *_ = np.linalg.lstsq(V2, np.sqrt(1 - u), rcond=None)
err = np.max(np.abs(V2 @ coef_sqrt - np.sqrt(1 - u)))
print(f"  √ 6 次: max err = {err:.2e}")

# 保存方案 A 的系数（sin 11 次 + √ 6 次）供 C 实现
err, cA = fit_sin(-8.64, 7.07, 11)
print("\n方案A sin 11 次系数（低→高）:", [f"{x:.9e}" for x in cA])
print("√ 6 次系数:", [f"{x:.9e}" for x in coef_sqrt])
