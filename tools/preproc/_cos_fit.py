# 生成 bootstrapping 核心：cos(2*pi*x) 在 [-0.25, 0.25] 的切比雪夫拟合系数
import numpy as np

# Chebyshev fit via numpy.polynomial.chebyshev (numpy 2.5)
from numpy.polynomial import chebyshev as C

N = 12  # 次数
xs = np.linspace(-0.25, 0.25, 20001)
ys = np.cos(2 * np.pi * xs)
coef = C.chebfit(xs, ys, N)
print("coef:", [float(c) for c in coef])

# 验证误差
approx = C.chebval(xs, coef)
err = np.abs(approx - ys).max()
print("max err:", err)

# 输出 C 数组（chebyshev 基: T_0..T_N 系数）
print("C array:")
for c in coef:
    print("  %.10f," % float(c))
