# LayerNorm 1/sqrt 拟合：平移归一化 z=(u-c)/s 上拟合（系数 O(10)，可密文编码）
import safetensors.torch
import numpy as np

path = r'D:\项目\New_vLLM\Modl\Qwen3-VL-2B-Instruct\model.safetensors'
with safetensors.torch.safe_open(path, framework='pt') as f:
    embed = f.get_tensor('model.language_model.embed_tokens.weight').float().numpy()
x = embed[100][0:1024].astype(np.float64)
var = x.var()

a = 0.5 * var
b = 2.0 * var
c = 0.5 * (a + b)
s = 0.5 * (b - a)          # z = (u - c)/s ∈ [-1,1]
u = np.linspace(a, b, 200001)
z = (u - c) / s
y = 1.0 / np.sqrt(u)

for deg in (6, 8):
    q = np.polyfit(z, y, deg)          # Q(z)，q[0] 为最高次
    y2 = np.polyval(q, z)
    e = np.abs(y2 - y)
    print('deg=%d max_abs_err=%.3e max_rel_err=%.3e  |Q|max=%.2f' % (deg, e.max(), (e / y).max(), np.abs(q).max()))

deg = 8
q = np.polyfit(z, y, deg)
print('\n// Q(z), z=(u-c)/s, deg=%d, c=%.17g s=%.17g' % (deg, c, s))
print('const double LN_Q[%d] = {' % (deg + 1))
for v in q:
    print('    %.17g,' % v)
print('};')
# 验证 Horner
zz = z
acc = q[0]
for k in range(1, deg + 1):
    acc = acc * zz + q[k]
e = np.abs(acc - y).max()
print('horner max_abs_err=%.3e' % e)
