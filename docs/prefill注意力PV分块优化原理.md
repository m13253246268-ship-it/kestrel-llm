# prefill 注意力 P×V 分块优化（m2）实现原理

> 本文单独解释 `VLLM_ATTN_PV_TILE=2`（m2，默认开）的**实现原理**，不含流程方法论
> （后者见 [性能优化方法论.md](性能优化方法论.md)）。目标读者：需要读懂、改动或复现
> 这段内核的人。所有行号对应 [../src/model/vllm_safetensors.c](../src/model/vllm_safetensors.c)。

---

## 一、这段代码算的是什么

prefill 的注意力（精确档，非稀疏）对每个 query 做：

```
O[q][d] = Σ_s  softmax(Q[q]·K[s])_s · V[s][d]
```

引擎把它拆成三段（在 `vllm_attn_batched_worker` 内）：

1. **QK**：算 `score[q][s] = Q[q]·K[s] / sqrt(hd)`，并取每 query 的 `max`；
2. **softmax**：`sc[q][s] = exp(score - max) / Σ exp(...)`；
3. **PV（VKQ）**：`O[q][d] += sc[q][s] · V[s][d]`。

m2 只改第 3 段 PV。前两段 QK 与 softmax **逐字不动**——这是它能做到位级不变的前提。

---

## 二、原始 PV 段的问题：逐 s 读-改-写

原始路径（[vllm_safetensors.c](../src/model/vllm_safetensors.c) 中非 PV 分支）对每个 query、每个位置 s：

```c
for (int s = 0; s < nk; s++) {
    float wgt = sc[k][s] * inv_sum;
    const float *vp = vp_head + (size_t)s * hd;
    float32x4_t wv = vdupq_n_f32(wgt);
    for (int i = 0; i < hdv; i += 4)
        vst1q_f32(o[k] + i, vfmaq_f32(vld1q_f32(o[k] + i), wv, vld1q_f32(vp + i)));
}
```

每 4 个维度、每个 s，都做 **1 次 load O + 1 次 load V + 1 次 store O**。O 只有
`hd×4 = 512 B`（每个 query head），本应 L1 常驻，但"读-改-写"让它每个 s 都进出
一次 LSU，白白占用发射槽。这是 PV 段的真正浪费——不是 DRAM 流量，是**每个 FMA
都拖着一次 O 的 load + store**。

历史上"把累加器搬进寄存器"（OREG）想解决它，但单独做是负收益：16 个 `float32x4`
累加器（64 维）加上 int8 V 的 `vmovl/vcvt` 临时量，把 32 个 NEON 寄存器挤爆，一部分
值被迫落栈（CHANGELOG §14 的汇编证据）。

---

## 三、m2 的核心思想：寄存器驻留 + V 载入共享

m2 一次解决两件事：

1. **累加器常驻寄存器**：把 O 的 32 维放进 8 个 `float32x4`，整段 s 循环只 FMA 到寄存器，
   段尾一次写回。彻底消除 O 的逐 s 读-改-写。
2. **双 query 共享 V 载入**：同一 kv head 下相邻两个 query 用**同一份 V 行**（GQA 语义），
   于是 V 载入一次、同时喂两个 query 的累加器，V 的 load 数减半。

关键洞察在于**这两件事互相成全**：

- 若单 query 用 16 个累加器（64 维，即 m1），寄存器被占满、无余量做 V 共享，还可能溢出；
- 若每个 query 只用 8 个累加器（32 维），两个 query 合计 16 个累加器，剩下 16 个寄存器
  留给 `wa/wb` 广播与 `vv` 临时量——**寄存器不再溢出，同时换来 V 载入减半**。

实测证明这个取舍是对的：V 载入减半的流量收益，大于维度块从 64 收到 32 的代价。
m2 在 n2048 上比 m1 快约 1.4%（35.37s vs 35.88s），且都远快于原始路径。

---

## 四、代码逐段解读

### 4.1 宏：把重复的展开写清楚

```c
#define PV_LOAD(j)  float32x4_t a##j = s0 ? vld1q_f32(oa + (j)*4) : vdupq_n_f32(0.0f);
#define PV_FMA(j)   a##j = vfmaq_f32(a##j, wa, vld1q_f32(vp + (j)*4));
#define PV_STORE(j) vst1q_f32(oa + (j)*4, a##j);
#define PV_EIGHT(M) M(0) M(1) M(2) M(3) M(4) M(5) M(6) M(7)
#define PV_SIXTEEN(M) PV_EIGHT(M) M(8) M(9) M(10) M(11) M(12) M(13) M(14) M(15)
#define PV_LOAD_B(j) float32x4_t b##j = s0 ? vld1q_f32(ob + (j)*4) : vdupq_n_f32(0.0f);
#define PV_PAIR(j) do { \
    float32x4_t vv = vld1q_f32(vp + (j)*4); \
    a##j = vfmaq_f32(a##j, wa, vv); \
    b##j = vfmaq_f32(b##j, wb, vv); \
} while (0);
#define PV_FMA_B(j)   b##j = vfmaq_f32(b##j, wb, vld1q_f32(vp + (j)*4));
#define PV_STORE_B(j) vst1q_f32(ob + (j)*4, b##j);
```

要点：

- `a##j` 是 query `k` 的第 `j` 个 4 维累加器，`b##j` 是 query `k+1` 的。
- `PV_LOAD` 的 `s0 ? 读 : 0` 是**跨 s 分块续接**的关键：`s0 == 0` 时从零开始（首块），
  否则把上一块已写回 O 的中间值重新载入累加器，继续同一条 FMA 链。
- `PV_PAIR` 是共享的核心：`vv` 只载入一次，`a` 和 `b` 各 fma 一次。

### 4.2 主循环：s 方向分块

```c
for (int s0 = 0; s0 < n[ST_ATTN_QB - 1]; s0 += ST_ATTN_PV_BLK) {
    ...
}
```

`s0` 是 KV 位置（s）方向的分块起点，块宽 `ST_ATTN_PV_BLK = 128`。128 行 × 4 B × 1 head
= 512 B 的 V 工作集，4 个 query 复用。分块让 V 工作集落在 L1，且为"续接累加器"提供边界。

### 4.3 m1：单 query / d64

```c
for (int k = 0; k < ST_ATTN_QB; k++) {
    int se = n[k] < s0 + ST_ATTN_PV_BLK ? n[k] : s0 + ST_ATTN_PV_BLK;
    if (se <= s0) continue;
    for (int d = 0; d < 128; d += 64) {
        float *oa = o[k] + d;
        PV_SIXTEEN(PV_LOAD)              // 16 个累加器 = 64 维
        for (int s = s0; s < se; s++) {
            float32x4_t wa = vdupq_n_f32(sc[k][s] * invs[k]);
            const float *vp = vp_head + (size_t)s * 128 + d;
            PV_SIXTEEN(PV_FMA)           // 每个 4 维 fma 一次，V 不共享
        }
        PV_SIXTEEN(PV_STORE)
    }
}
```

只做了"寄存器驻留"，V 不共享。是 m2 的对照基线。

### 4.4 m2：双 query / d32 共享 V

```c
for (int k = 0; k < ST_ATTN_QB; k += 2) {
    int ea = n[k]   < s0 + ST_ATTN_PV_BLK ? n[k]   : s0 + ST_ATTN_PV_BLK;
    int eb = n[k+1] < s0 + ST_ATTN_PV_BLK ? n[k+1] : s0 + ST_ATTN_PV_BLK;
    if (eb <= s0) continue;
    for (int d = 0; d < 128; d += 32) {
        float *oa = o[k] + d, *ob = o[k+1] + d;
        PV_EIGHT(PV_LOAD)                // a0..a7 = query k 的 32 维
        PV_EIGHT(PV_LOAD_B)              // b0..b7 = query k+1 的 32 维
        int s = s0;
        for (; s < ea; s++) {            // 两 query 都有效的段
            float32x4_t wa = vdupq_n_f32(sc[k][s] * invs[k]);
            float32x4_t wb = vdupq_n_f32(sc[k+1][s] * invs[k+1]);
            const float *vp = vp_head + (size_t)s * 128 + d;
            PV_EIGHT(PV_PAIR)            // V 载入一次，喂两个累加器
        }
        for (; s < eb; s++) {            // 仅 query k+1 有效（因果边界）
            float32x4_t wb = vdupq_n_f32(sc[k+1][s] * invs[k+1]);
            const float *vp = vp_head + (size_t)s * 128 + d;
            PV_EIGHT(PV_FMA_B)
        }
        PV_EIGHT(PV_STORE)
        PV_EIGHT(PV_STORE_B)
    }
}
```

对比 m1：d 块从 64 收到 32（每个 query 的累加器从 16 个减到 8 个），换来的是一段
`s ∈ [s0, ea)` 里 `PV_PAIR` 把 V 载入一次、同时喂 `a` 和 `b`——**V load 减半**。

---

## 五、位级不变论证（为什么能逐字节相同）

对任意固定输出元素 `O[k][d]`，累加是：

```
O[k][d] = 0; for s in 0..n[k]-1:  O[k][d] = fma(O[k][d], sc[k][s]·invs[k], V[s][d])
```

这是一条**沿 s 的一维 FMA 链，各 s 项之间无跨项依赖**。m2 对它的处理：

1. **s 分块**（`s0 += 128`）：块与块之间用 `PV_LOAD` 把上一块的中间值**续接**进累加器，
   不是"各块求部分和再相加"。所以对固定 `(k,d)`，`s` 仍是全局升序的同一条链。
2. **d 分块**（`d += 32/64`）：`d` 方向本就没有跨维归约，每个 `d` 独立。
3. **双 query 配对**：`a`（query k）与 `b`（query k+1）各自独立累加，共享的只是
   `vv` 这个中间载入值，不改变任何一条链的 FMA 顺序或操作数。

唯一需要逐字保证的是：`sc[k][s]·invs[k]` 这个权重表达式不变、`vfmaq_f32` 的操作数
顺序（`vfmaq_f32(a, wa, vv)` 的语义是 `a += wa*vv`）不变、标量尾不做 FMA 融合。这三条
m2 都满足。QK 与 softmax 完全没动，所以它们的归约树也不受影响。

结论：m2 与原始路径**位级等价**，而非"近似等价"。这是它能默认开启的根本依据。

---

## 六、因果边界处理

prefill 是因果注意力：query `k` 只能看位置 `s < prev_len + t + k`。在一个 4-query 组里
（`t, t+1, t+2, t+3`），`n[k]` 递增，所以存在一段 `s` 对前面的 query 已越界、对后面的
query 仍有效。

m2 里用 `ea`（query k 的边界）和 `eb`（query k+1 的边界）分开处理：

- `s ∈ [s0, ea)`：两个 query 都有效，走 `PV_PAIR`；
- `s ∈ [ea, eb)`：只有 query k+1 有效，只更新 `b`（`PV_FMA_B`），**不给 query k 补算
  零权重**——这是保持位级不变的关键，补一个 `wa=0` 的 FMA 会在 `O[k][d]` 上多加一次
  `0*vv` 的舍入，结果会变。

QK 段本身也遵守同一因果（`kmax` 收缩），m2 只是把 PV 段的边界显式写出来。

---

## 七、启用与回退条件

```c
if (c->pv_tile && c->hd == 128 && ST_ATTN_QB == 4) {
    ... vllm_attn_pv_tile128(o, sc, invs, n, vp_head, c->pv_tile);
    continue;
}
// 否则走原始串行路径
```

- **只对 `head_dim == 128` 且 `ST_ATTN_QB == 4` 生效**：`hd=128` 恰好被 d 方向切成
  4 段、每段 32 维 = 8 个 `float32x4`；`ST_ATTN_QB == 4` 保证 query 能两两配对。
  两条都是 m2 寄存器预算成立的前提，不满足自动回退，无需用户判断。
- **默认 `2`**：位级对拍 + A/B 领先后才默认化（见方法论文档 §六）。
- **`VLLM_ATTN_PV_TILE=0`**：显式关回原始路径，逃生舱。
- 尾部（`nb % 4`）走原串行路径，与主分块共用同一份实现，避免位级分叉。

---

## 八、性能数据（RK3588，4×A76，热态，3 次取中位）

| 档 | 原始 m0 | m1 | m2 | m2 相对 m0 |
|---|---|---|---|---|
| 2B · 2048 token | 39.70 s | 35.88 s | **35.37 s** | −10.9% |
| 2B · 256 token | 3.275 s | 3.209 s | **3.206 s** | −2.1% |
| 8B · 512 token | 27.02 s | — | **25.86 s** | −4.3% |
| 8B · 1024 token | 59.07 s | — | **55.45 s** | −6.1% |
| 8B · 2048 token | 139.10 s | — | **127.34 s** | −8.5% |

分桶（2B 2048 token）：ATTN 桶 15516 ms → **11435 ms（−27%）**，GEMM 桶 ±0.4% 不动；
8B 同样只动 ATTN（n2048：42432 → **30778 ms（−27%）**，GEMM 94.1 s → 93.3 s ±0.8%）——
坐实只加速了 PV 段。m2 比 m1 快约 1.4%，是"V 载入减半"相对"更宽维度块"的净收益。
收益随上下文单调放大（attention 占比随 n 增长）；8B 相对 2B 的绝对幅度略小，因为
8B 的 FFN/GEMM 是 2B 的 4 倍、attention 占总 prefill 比例更低。

---

## 九、汇编验证要点（复用/改动必查）

历史教训：源码把累加器写成 `float32x4_t acc[16]`，编译器可能整体落栈。改这段必须
`objdump` 看热循环，确认：

1. 累加器真的在 `v0..v31`，不是 `stp/ldp` 进出 `sp`；
2. `PV_PAIR` 段是"一次 V `ldr`/`ldp` + 两次 `fmla`"，V 没有重复载入；
3. 栈帧 `sub sp, sp, #N` 没有因寄存器溢出而异常变大。

本轮在 2B 上确认：m2 热循环里 16 个累加器（`v16..v31` 一带）+ 双字 V 载入（`ldp q`），
无 sp 写回，说明寄存器驻留成立。换编译选项或改块宽后必须重新核对。
