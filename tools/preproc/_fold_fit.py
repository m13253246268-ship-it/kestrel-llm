# _fold_fit.py - 折叠函数 P(x)≈sin(2πx) 在混叠区间并集上的逼近次数估计
# 混叠：明文 v' = m+e+q0·k（k∈[-I,I]），x = v'/q0 = k + ε，ε∈[0, 0.5]（|z|<0.5）
# P 需在 ∪_{k=-I}^{I} [k, k+0.5] 上 ≈ sin(2πx) = sin(2πε)（周期吸收 k）
import numpy as np

def fit_and_err(I, deg, npts=40000):
    """在 ∪[k, k+0.5] 上用最小二乘拟合 sin(2πx)，返回 max abs err"""
    xs = np.concatenate([np.linspace(k, k + 0.5, npts // (2 * I + 1) + 2)
                         for k in range(-I, I + 1)])
    ys = np.sin(2 * np.pi * xs)
    # 幂基最小二乘（Vandermonde）
    V = np.vander(xs, deg + 1, increasing=True)
    coef, *_ = np.linalg.lstsq(V, ys, rcond=None)
    pred = V @ coef
    err = np.max(np.abs(pred - ys))
    return err, coef

for I in [2, 4, 6, 8]:
    print(f"混叠范围 I={I}（区间 {2*I+1} 个）:")
    for deg in [9, 13, 17, 21, 25, 31, 41]:
        err, _ = fit_and_err(I, deg)
        print(f"  次数 {deg}: max err = {err:.2e}")
    print()
