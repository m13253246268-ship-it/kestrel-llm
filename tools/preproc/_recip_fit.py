# 生成 1/x 幂基多项式拟合（密文域除法近似，Horner 深 6）
import numpy as np
from numpy.polynomial import chebyshev as C
from numpy.polynomial import polynomial as P

lo = 8 * np.exp(-1.0)
hi = 8 * np.exp(1.0)
print(f"denominator range: [{lo:.4f}, {hi:.4f}]")

# 用切比雪夫拟合后转幂基，或直接 np.polyfit 幂基
xs = np.linspace(lo, hi, 200001)
ys = 1.0 / xs
N = 8
coef = np.polyfit(xs, ys, N)  # 幂基系数，从高次到低次
approx = np.polyval(coef, xs)
err = np.abs(approx - ys).max()
print(f"1/x polyfit N={N} max abs err={err:.3e}")

# 输出 C 数组（p6..p0，高次到低次）
print("C array (high->low):")
for c in coef:
    print("  %.10e," % float(c))

# 也输出低次到高次（Horner 用 p6 先乘）
print("C array (low->high):")
for c in coef[::-1]:
    print("  %.10e," % float(c))
