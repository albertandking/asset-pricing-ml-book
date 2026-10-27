---
short_title: 0.2（续）因子、特征、载荷的关系
kernelspec:
  name: python3
  display_name: Python 3
numbering:
  enumerator: '0.%s'
  exercise:
    template: 习题 %s
---

(sec-ch0-fcl)=
# 0.2 基本概念速览（续）：因子、特征、载荷的关系

**本页准备。** 本书每个页面的代码在独立的内核中运行，前面页面定义的变量在这里不可用。下面的单元导入本页用到的模块，并重建上一页“特征”一节中的默认载荷映射 `gamma0`。

```{code-cell} python
import numpy as np

from apsim import MarketParams, simulate_market

gamma0 = MarketParams().resolved_gamma()                    # 默认 Γ: (L+1, K) = (6, 3)，与上一页“特征”一节相同
```

**白话。** 一句话概括：**特征是股票的属性，因子是收益序列，载荷是敏感度；特征决定载荷，载荷乘以因子得到收益。** 三者的区别看下标最清楚：

| | 是什么 | 随什么变化（下标） | 例子 |
|---|---|---|---|
| 特征 $\mathbf z_{i,t-1}$ | 期初就能观察到的股票属性 | 随股票 $i$ 与时间 $t$ 变化 | 某只小盘股本月初标准化的对数市值为 $-1.8$ |
| 因子 $\mathbf f_t$ | 许多股票共同面对的收益来源，通常是一个组合的收益 | 只随时间 $t$ 变化，同一期所有股票面对同一个值 | 本月规模因子（小减大）的收益为 3% |
| 载荷 $\bv\beta_{i,t}$ | 股票收益对因子的敏感度 | 随股票 $i$ 与时间 $t$ 变化 | 这只小盘股对规模因子的载荷为 1.2 |

本章模型把三者连成两个式子：

```{math}
:enumerated: false
\underbrace{\bv\beta_{i,t}}_{K\times 1} = \underbrace{\bv\Gamma'}_{K\times(L+1)}\,\underbrace{\tilde{\mathbf z}_{i,t-1}}_{(L+1)\times 1}, \qquad
\underbrace{r_{i,t}}_{1\times 1} = \underbrace{\alpha_{i,t}}_{1\times 1} + \underbrace{\bv\beta_{i,t}'}_{1\times K}\,\underbrace{\mathbf f_t}_{K\times 1} + \underbrace{\varepsilon_{i,t}}_{1\times 1} .
```

第一个式子说“特征决定载荷”，第二个式子说“载荷乘以因子得到收益”。特征与载荷带下标 $i$，因子不带：因子是全市场共同的，股票之间收益的系统性差别全部来自载荷（以及定价误差 $\alpha_{i,t}$）。

**因子往往由特征排序构造。** 三者还有一层容易混淆的关系：实证中的因子，常常是按某个特征排序构造的组合收益。规模因子就是这样得到的：每期按市值把股票排序，买入市值小的一组、卖空市值大的一组，这个“小减大”组合的收益就是规模因子。于是同一个词“规模”可以指三样东西：

- **特征**：一只股票的（标准化）对数市值，例如上表的 $-1.8$；
- **因子**：“小减大”组合每期的收益，例如本月的 3%；
- **载荷**：一只股票的收益对规模因子的敏感度，例如上表的 1.2。

特征与载荷是两个不同的量。小市值股票的规模特征（标准化对数市值）是负的，它对规模因子的载荷却通常为正；不过这个符号差别只是因为规模因子的方向被定义为“小减大”，如果把因子定义为“大减小”，载荷就与特征同号。二者真正的区别有三点。第一，尺度与单位不同：特征的单位是“横截面标准差”，载荷的单位是“因子每变动 1 个百分点，股票收益变动多少个百分点”；本章模拟器中载荷是 $0.5\, z_1$ 而不是 $z_1$ 本身。第二，特征可以直接观察，载荷却需要估计（例如用前面的时间序列回归）。第三，“特征决定载荷”只是一个模型假设，$\bv\Gamma$ 刻画二者的关系；本章的模拟器中这个关系是精确的线性关系，真实数据中至多是近似。

**例子。** 在模拟器的默认 $\bv\Gamma$ 下，载荷为 $\bv\beta_{i,t} = (1,\ 0.5\, z_{1,i,t-1},\ 0.5\, z_{2,i,t-1})'$。设三只股票的 $z_1$ 分别为 $-1$、$0$、$2$，其他特征都为 0。按 $z_1$ 从低到高排序，它们对第 2 个因子的载荷依次为 $-0.5$、$0$、$1$，对第 1 个因子的载荷都是 1，对第 3 个因子的载荷都是 0。所以买入 $z_1$ 高的股票、卖空 $z_1$ 低的股票，得到的组合对第 1 个因子的载荷正好抵消，对第 2 个因子的载荷为正：“$z_1$ 高减低”组合的收益主要跟着 $f_{2,t}$ 走。如果把 $z_1$ 理解为“小市值程度”（负的标准化对数市值），这个组合就是“小减大”，$f_{2,t}$ 扮演规模因子的角色。

下面在模拟器中验证。用默认参数、120 个月的样本，每期买入 $z_{1,i,t-1}$ 最高的 30% 股票、卖空最低的 30%，两条腿各自等权。组合在第 $t$ 期的收益是 $\mathbf w_t'\mathbf r_t$，其中权重向量 $\mathbf w_t \in \Real^N$ 在高组取 $1/N_H$、在低组取 $-1/N_L$、其余为 0（$N_H$、$N_L$ 为两组的股票数），各分量之和为 0；组合的载荷是 $\mathbf B_t'\mathbf w_t$（$K\times N$ 乘 $N\times 1$，得 $K\times 1$）。代码中的 `np.einsum("tn,tnk->tk", w_ls, B_f)` 对每一期 $t$ 把 $n$ 求和，即逐期计算 $\mathbf B_t'\mathbf w_t$。

```{code-cell} python
# 三只股票的 z1 = -1、0、2（假设数字），其余特征为 0
z1_three = np.array([-1.0, 0.0, 2.0])                                   # (3,)
zt_three = np.column_stack([np.ones(3), z1_three, np.zeros((3, 4))])    # 每行 z̃_i'：(3, L+1) = (3, 6)
beta_three = zt_three @ gamma0                                           # (3, 6) @ (6, 3) -> (3, 3)：每行 β_i'
print("三只股票的载荷（每行一只股票）：\n", beta_three)
assert np.allclose(beta_three, [[1, -0.5, 0], [1, 0, 0], [1, 1, 0]])

# 模拟市场中的“z1 高减低”组合：每期买入 z1 最高的 30%、卖空最低的 30%，两条腿各自等权
fcl = simulate_market(MarketParams(n_periods=120), seed=7)
T_f, N_f = fcl.params.n_periods, fcl.params.n_stocks
Z1_f = fcl.panel["z1"].unstack("stock").to_numpy()                       # (T, N)：第 t 行为各股票的 z_{1,i,t-1}
exret_f = fcl.panel["exret"].unstack("stock").to_numpy()                 # (T, N)：第 t 行为 r_t'
B_f = fcl.panel[["beta1", "beta2", "beta3"]].to_numpy().reshape(T_f, N_f, 3)   # (TN, K) -> (T, N, K)
lo_q, hi_q = np.quantile(Z1_f, [0.3, 0.7], axis=1)                       # 各期的 30%、70% 分位数，各为 (T,)
high, low = Z1_f >= hi_q[:, None], Z1_f <= lo_q[:, None]                # (T, N) 的布尔数组
w_ls = high / high.sum(axis=1, keepdims=True) - low / low.sum(axis=1, keepdims=True)   # (T, N)：第 t 行为 w_t'
ls_ret = (w_ls * exret_f).sum(axis=1)                                    # (T,)：w_t'r_t
ls_beta = np.einsum("tn,tnk->tk", w_ls, B_f)                             # (T, N), (T, N, K) -> (T, K)：第 t 行为 (B_t'w_t)'
F_f = fcl.factors.to_numpy()                                             # (T, K)
corr_ls = np.array([np.corrcoef(ls_ret, F_f[:, k])[0, 1] for k in range(3)])
print("组合载荷的时间平均：", ls_beta.mean(axis=0).round(3) + 0.0)          # 加 0.0 把 -0.0 显示为 0.0
print("组合收益与 f1、f2、f3 的相关系数：", corr_ls.round(3))
assert np.allclose(w_ls.sum(axis=1), 0) and np.allclose(ls_beta[:, 0], 0)   # 零成本；对因子 1 的载荷恰好抵消
assert 1.0 < ls_beta[:, 1].mean() < 1.3 and abs(ls_beta[:, 2].mean()) < 0.1
assert corr_ls[1] > 0.85 and np.all(np.abs(corr_ls[[0, 2]]) < 0.3)
```

组合收益与 $f_{2,t}$ 的相关系数在 0.85 以上，与另两个因子的相关系数绝对值都在 0.3 以下；组合对因子 2 的平均载荷在 1.0～1.3 之间（均由 `assert` 保证）。原因可以从组合的载荷直接读出：$\mathbf w_t$ 各分量之和为 0，所以组合对第 1 个因子的载荷 $\mathbf w_t'\mathbf 1_N = 0$，恰好抵消；在默认 $\bv\Gamma$ 下，股票对因子 2 的载荷是 $0.5\, z_1$，组合对因子 2 的载荷就是 0.5 乘以高组与低组 $z_1$ 均值之差。标准正态分布最高 30% 的条件均值约为 1.16，两组之差约为 2.3 个标准差，所以组合对因子 2 的载荷约为 $0.5 \times 2.3 \approx 1.16$。两组股票的 $z_2$ 平均而言相同，组合对因子 3 的载荷接近 0。剩下的是两组股票异质性冲击的平均值之差，它随每组股票数的增加而变小，但在每组 150 只股票时仍不可忽略，所以相关系数没有达到 1。

**业界的口径：Barra 与横截面回归。** 国内量化业界说的“因子”，多数时候指的是这里的**特征**或**信号**：“市值因子”“动量因子”“换手率因子”通常指每只股票每期的一个数值，而不是一条收益序列。Barra 风险模型的做法与此一致：它直接把标准化后的特征当作载荷，称为“因子暴露”，再在每一期用横截面回归估计“因子收益”。

用本章的模型看，这相当于一个特例：$K = L+1$、$\bv\Gamma = \mathbf I_{L+1}$。常数项对应一个所有股票载荷都为 1 的因子（Barra 称为国家因子；它与行业哑变量完全共线，求解方法见本小节末尾），其余每个特征对应一个因子，载荷就是特征本身，即去掉常数行与常数列之后，$\bv\Gamma$ 的其余部分是单位阵 $\mathbf I_L$；因子收益事先未知，每期由回归得到，是估计值。为什么横截面回归能估计“因子收益”？把式 {eq}`eq-ch0-alpha` 与 {eq}`eq-ch0-beta` 代入收益方程，并利用 $\bv\beta_{i,t}'\mathbf f_t = \tilde{\mathbf z}_{i,t-1}'\bv\Gamma\mathbf f_t$：

```{math}
:enumerated: false
r_{i,t} = \bv\theta'\mathbf z_{i,t-1} + \tilde{\mathbf z}_{i,t-1}'\bv\Gamma\mathbf f_t + \varepsilon_{i,t} = \underbrace{\tilde{\mathbf z}_{i,t-1}'}_{1\times(L+1)}\,\underbrace{\mathbf g_t}_{(L+1)\times 1} + \varepsilon_{i,t}, \qquad \underbrace{\mathbf g_t}_{(L+1)\times 1} = \underbrace{\bv\Gamma}_{(L+1)\times K}\,\underbrace{\mathbf f_t}_{K\times 1} + \begin{pmatrix} 0 \\ \bv\theta \end{pmatrix} .
```

在第 $t$ 期的横截面上，$\mathbf f_t$ 是所有股票共同的一个已实现值，$\mathbf g_t \in \Real^{L+1}$ 因而是一个对所有股票相同的系数向量：收益是期初特征的精确线性函数，只多了异质性冲击。把 $N$ 只股票按行堆叠，$\mathbf r_t = \tilde{\mathbf Z}_{t-1}\mathbf g_t + \bv\varepsilon_t$，最小二乘估计为

```{math}
:enumerated: false
\underbrace{\hat{\mathbf g}_t}_{(L+1)\times 1} = (\tilde{\mathbf Z}_{t-1}'\tilde{\mathbf Z}_{t-1})^{-1}\tilde{\mathbf Z}_{t-1}'\mathbf r_t = \mathbf g_t + \underbrace{(\tilde{\mathbf Z}_{t-1}'\tilde{\mathbf Z}_{t-1})^{-1}}_{(L+1)\times(L+1)}\,\underbrace{\tilde{\mathbf Z}_{t-1}'}_{(L+1)\times N}\,\underbrace{\bv\varepsilon_t}_{N\times 1} .
```

$\bv\varepsilon_t$ 与期初特征和因子独立、均值为 0，所以给定 $\tilde{\mathbf Z}_{t-1}$ 与 $\mathbf f_t$，$\hat{\mathbf g}_t$ 是 $\mathbf g_t$ 的无偏估计，误差的条件协方差矩阵为 $\sigma_\varepsilon^2(\tilde{\mathbf Z}_{t-1}'\tilde{\mathbf Z}_{t-1})^{-1}$。模拟器中的特征相互独立、服从 $N(\mathbf 0, \mathbf I_L)$，$\tilde{\mathbf Z}_{t-1}'\tilde{\mathbf Z}_{t-1}/N$ 接近 $\mathbf I_{L+1}$，所以误差每个分量的标准差约为 $\sigma_\varepsilon/\sqrt N$：默认 $\sigma_\varepsilon = 0.10$、$N = 500$ 时约为 0.45%。这是同一期横截面上的噪声，随股票数增加而变小。默认 $\bv\Gamma$ 下 $\bv\Gamma\mathbf f_t = (f_{1,t},\ 0.5 f_{2,t},\ 0.5 f_{3,t},\ 0,\ 0,\ 0)'$，于是第 $t$ 期对常数与 $z_1$、$z_2$ 的系数估计为

```{math}
:enumerated: false
(\hat g_{0,t},\ \hat g_{1,t},\ \hat g_{2,t}) \approx (f_{1,t},\ 0.5 f_{2,t},\ 0.5 f_{3,t}) + (0,\ \theta_1,\ \theta_2) + \text{横截面噪声} .
```

$z_3$～$z_5$ 的系数真值为 0（假定 $\bv\theta$ 也只作用于 $z_1$、$z_2$），所以回归中只放常数与 $z_1$、$z_2$，结论不变。系数是 $0.5 f_{2,t}$ 而不是 $f_{2,t}$，因为每单位 $z_1$ 带来的载荷是 0.5：Barra 式回归估计的“因子收益”是 $\bv\Gamma\mathbf f_t$，即真实因子经过特征到载荷的映射换算之后的收益。$\bv\theta \neq \mathbf 0$ 时，系数还包含定价误差 $\bv\theta$：横截面回归本身无法区分特征带来的是风险补偿还是错误定价，这与“常见误区”一节的误区四是同一件事。

```{code-cell} python
# 逐期横截面回归：r_t 对 (1, z1, z2)，与 (f1, 0.5 f2, 0.5 f3) 比较（默认 Γ、θ = 0）
X_cs = np.concatenate([np.ones((T_f, N_f, 1)),
                       fcl.panel[["z1", "z2"]].to_numpy().reshape(T_f, N_f, 2)], axis=2)   # (T, N, 3)：第 t 期为 [1_N, z1, z2]
g_hat = np.array([np.linalg.lstsq(X_cs[t], exret_f[t], rcond=None)[0] for t in range(T_f)])  # (T, 3)：第 t 行为 ĝ_t'
g_true = F_f * np.array([1.0, 0.5, 0.5])                                  # 广播 (T, 3) * (3,) -> (T, 3)：(f1, 0.5 f2, 0.5 f3)
err_cs = g_hat - g_true                                                   # (T, 3)：横截面噪声
corr_cs = np.array([np.corrcoef(g_hat[:, k], g_true[:, k])[0, 1] for k in range(3)])
noise_sd = fcl.params.idio_vol / np.sqrt(N_f)                             # σ_ε / √N
print("系数与 (f1, 0.5 f2, 0.5 f3) 的相关系数：", corr_cs.round(3))
print("噪声的标准差：", err_cs.std(axis=0).round(4), "，理论值 σ_ε/√N ≈", round(noise_sd, 4))
assert np.all(corr_cs > 0.9)
assert np.all(np.abs(err_cs.std(axis=0) / noise_sd - 1) < 0.2)
assert np.all(np.abs(err_cs.mean(axis=0)) < 3 * err_cs.std(axis=0, ddof=1) / np.sqrt(T_f))   # 噪声均值为 0

# θ1 = 0.01：同一种子，只改定价误差；α = θ1 z1 在回归元张成的空间内，z1 的系数每期恰好多出 θ1
fcl_a = simulate_market(MarketParams(n_periods=120, alpha_coef=(0.01, 0, 0, 0, 0)), seed=7)
R_a = fcl_a.panel["exret"].unstack("stock").to_numpy()                   # (T, N)
g_hat_a = np.array([np.linalg.lstsq(X_cs[t], R_a[t], rcond=None)[0] for t in range(T_f)])  # (T, 3)
gap_a = g_hat_a - g_hat                                                   # (T, 3)：两次回归系数之差，每期应恰为 (0, θ1, 0)
print("θ1 = 0.01 与 θ = 0 的系数之差（前 3 期）：\n", gap_a[:3].round(6) + 0.0)
assert np.array_equal(fcl_a.factors.to_numpy(), F_f)                       # 两个市场的因子相同
assert np.allclose(gap_a, [0, 0.01, 0])                                    # 每期精确成立，不只是平均
```

三列系数与对应的 $(f_{1,t}, 0.5 f_{2,t}, 0.5 f_{3,t})$ 的相关系数都在 0.9 以上，噪声的标准差与 $\sigma_\varepsilon/\sqrt N$ 相差不到 20%，噪声的时间平均与 0 的距离在 3 倍标准误以内；加入 $\theta_1 = 0.01$ 后，$z_1$ 的系数每期恰好多出 $\theta_1$（均由 `assert` 保证）。同一种子下两个市场的因子、特征与异质性冲击完全相同，收益只多了 $\alpha_{i,t} = \theta_1 z_{1,i,t-1}$；它是回归元 $z_1$ 的倍数，而最小二乘系数是收益的线性函数，所以两次回归的系数之差每期精确等于 $(0, \theta_1, 0)$，没有噪声。如果 $\bv\theta$ 作用于回归中省略的 $z_3$～$z_5$，就不再精确：模拟器中特征相互独立，被省略的那部分 $\alpha_{i,t}$ 只增加噪声；真实数据中特征相关，它会使保留下来的系数产生偏误。实际的 Barra 模型还包含行业因子；国家因子与全部行业哑变量完全共线，Barra 通过约束“行业因子收益按市值加权之和为 0”来求解。回归通常还按市值的平方根加权。这些细节不改变上面的逻辑。

这些名词在学术论文、国内量化业界与本书中的不同口径，集中整理在[术语对照：学术与业界](../glossary.md)一页，例如[因子](#glossary-factor)、[载荷与暴露](#glossary-loading)、[因子收益](#glossary-factor-return)与[横截面回归与 Fama–MacBeth](#glossary-fm)。

