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

**拆开讲这两个式子。** 第一个式子的分量、维度与展开，见上一页“特征”一节的[拆解](#ch0-unpack-gamma)。第二个式子就是 0.4 节的式 {eq}`eq-ch0-ret`：它的四项的含义与单位、逐项展开，以及同一期 $N$ 只股票按行堆叠成 $\mathbf r_t = \bv\alpha_t + \mathbf B_t\mathbf f_t + \bv\varepsilon_t$、$\mathbf B_t = \tilde{\mathbf Z}_{t-1}\bv\Gamma$ 的结构，统一在 0.3～0.4 节页面的[拆解](#ch0-unpack-return)中讲解。本页只用到其中一点：$N\times K$ 的载荷矩阵 $\mathbf B_t$ 的第 $i$ 行是 $\bv\beta_{i,t}'$。下面代码中的 `B_f` 形状为 `(T, N, K)`，`B_f[t-1]` 就是 $\mathbf B_t$。

**因子往往由特征排序构造。** 三者还有一层容易混淆的关系：实证中的因子，常常是按某个特征排序构造的组合收益。规模因子就是这样得到的：每期按市值把股票排序，买入市值小的一组、卖空市值大的一组，这个“小减大”组合的收益就是规模因子。于是同一个词“规模”可以指三样东西：

- **特征**：一只股票的（标准化）对数市值，例如上表的 $-1.8$；
- **因子**：“小减大”组合每期的收益，例如本月的 3%；
- **载荷**：一只股票的收益对规模因子的敏感度，例如上表的 1.2。

特征与载荷是两个不同的量。小市值股票的规模特征（标准化对数市值）是负的，它对规模因子的载荷却通常为正；不过这个符号差别只是因为规模因子的方向被定义为“小减大”，如果把因子定义为“大减小”，载荷就与特征同号。二者真正的区别有三点。第一，尺度与单位不同：特征的单位是“横截面标准差”，载荷的单位是“因子每变动 1 个百分点，股票收益变动多少个百分点”；本章模拟器中载荷是 $0.5\, z_1$ 而不是 $z_1$ 本身。第二，特征可以直接观察，载荷却需要估计（例如用前面的时间序列回归）。第三，“特征决定载荷”只是一个模型假设，$\bv\Gamma$ 刻画二者的关系；本章的模拟器中这个关系是精确的线性关系，真实数据中至多是近似。

**例子。** 在模拟器的默认 $\bv\Gamma$ 下，载荷为 $\bv\beta_{i,t} = (1,\ 0.5\, z_{1,i,t-1},\ 0.5\, z_{2,i,t-1})'$。设三只股票的 $z_1$ 分别为 $-1$、$0$、$2$，其他特征都为 0。按 $z_1$ 从低到高排序，它们对第 2 个因子的载荷依次为 $-0.5$、$0$、$1$，对第 1 个因子的载荷都是 1，对第 3 个因子的载荷都是 0。所以买入 $z_1$ 高的股票、卖空 $z_1$ 低的股票，得到的组合对第 1 个因子的载荷正好抵消，对第 2 个因子的载荷为正：“$z_1$ 高减低”组合的收益主要跟着 $f_{2,t}$ 走。如果把 $z_1$ 理解为“小市值程度”（负的标准化对数市值），这个组合就是“小减大”，$f_{2,t}$ 扮演规模因子的角色。

下面在模拟器中验证。用默认参数、120 个月的样本，每期买入 $z_{1,i,t-1}$ 最高的 30% 股票、卖空最低的 30%，两条腿各自等权。组合在第 $t$ 期的收益是 $\mathbf w_t'\mathbf r_t$，其中权重向量 $\mathbf w_t \in \Real^N$ 在高组取 $1/N_H$、在低组取 $-1/N_L$、其余为 0（$N_H$、$N_L$ 为两组的股票数），各分量之和为 0；组合的载荷是 $\mathbf B_t'\mathbf w_t$（$K\times N$ 乘 $N\times 1$，得 $K\times 1$）。代码中的 `np.einsum("tn,tnk->tk", w_ls, B_f)` 对每一期 $t$ 把 $n$ 求和，即逐期计算 $\mathbf B_t'\mathbf w_t$。

(ch0-unpack-portfolio)=
**拆开讲组合收益 $\mathbf w_t'\mathbf r_t$ 与组合载荷 $\mathbf B_t'\mathbf w_t$。** 写出分量：

```{math}
:enumerated: false
\underbrace{\mathbf w_t}_{N\times 1} = \begin{pmatrix} w_{1,t} \\ w_{2,t} \\ \vdots \\ w_{N,t} \end{pmatrix} , \qquad
\underbrace{\mathbf r_t}_{N\times 1} = \begin{pmatrix} r_{1,t} \\ r_{2,t} \\ \vdots \\ r_{N,t} \end{pmatrix} ,
```

$w_{i,t}$ 是第 $t$ 期组合在股票 $i$ 上的权重，即每 1 元多头（或空头）中投在股票 $i$ 上的份额，正数为买入、负数为卖空，没有单位；$r_{i,t}$ 是股票 $i$ 的超额收益。**维度与展开**：$\mathbf w_t'$ 是 $1\times N$，乘 $N\times 1$ 的 $\mathbf r_t$ 得到一个数；$\mathbf B_t'$ 是 $K\times N$，第 $i$ 列是 $\bv\beta_{i,t}$，乘 $N\times 1$ 的 $\mathbf w_t$ 得到 $K\times 1$ 的向量：

```{math}
:enumerated: false
\mathbf w_t'\mathbf r_t = \sum_{i=1}^{N} w_{i,t} r_{i,t} , \qquad
\mathbf B_t'\mathbf w_t = (\bv\beta_{1,t}, \bv\beta_{2,t}, \dots, \bv\beta_{N,t}) \begin{pmatrix} w_{1,t} \\ w_{2,t} \\ \vdots \\ w_{N,t} \end{pmatrix} = \sum_{i=1}^{N} w_{i,t}\, \bv\beta_{i,t} , \qquad
\mathbf w_t'\mathbf 1_N = \sum_{i=1}^{N} w_{i,t} .
```

白话：组合收益是各只股票收益按权重加权求和，在这里就是高组的平均收益减低组的平均收益；组合载荷的第 $k$ 个分量 $\sum_i w_{i,t}\beta_{k,i,t}$ 是各只股票对因子 $k$ 的载荷按同样的权重加权求和；$\mathbf w_t'\mathbf 1_N = 0$ 表示买入与卖空的金额相等，期初不花钱。例如在上面三只股票中买入 1 元 $z_1 = 2$ 的股票、卖空 1 元 $z_1 = -1$ 的股票，$\mathbf w_t = (-1, 0, 1)'$，组合载荷为 $(1, 1, 0)' - (1, -0.5, 0)' = (0, 1.5, 0)'$。**特例**：$\mathbf w_t$ 只在股票 $i$ 上为 1、其余为 0 时，组合收益与组合载荷就是这只股票自己的 $r_{i,t}$ 与 $\bv\beta_{i,t}$；$\mathbf w_t = \mathbf 1_N/N$（等权买入全部股票）时，组合载荷是全部股票载荷的横截面平均。**数据矩阵**：代码中的 `w_ls` 形状为 `(T, N)`，第 $t$ 行是 $\mathbf w_t'$；权重每期重新计算，`ls_beta` 的第 $t$ 行是 $(\mathbf B_t'\mathbf w_t)'$。

```{code-cell} python
# 拆开讲组合载荷：买入 1 元 z1 = 2 的股票、卖空 1 元 z1 = -1 的股票，B'w 等于两只股票载荷之差
B_three = np.column_stack([np.ones(3), [-1.0, 0.0, 2.0], np.zeros((3, 4))]) @ gamma0   # (3, 6) @ (6, 3) -> (3, 3)
w_three = np.array([-1.0, 0.0, 1.0])                                     # w: (N,) = (3,)
print("组合载荷 B'w =", B_three.T @ w_three)                             # (3, 3) @ (3,) -> (3,)
assert np.allclose(B_three.T @ w_three, [0, 1.5, 0]) and np.isclose(w_three.sum(), 0)
assert np.allclose(B_three.T @ w_three, sum(w_three[i] * B_three[i] for i in range(3)))   # Σ_i w_i β_i
```

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

(ch0-unpack-g)=
**拆开讲 $\mathbf g_t$。** 第一个等号用了 $\bv\beta_{i,t}' = (\bv\Gamma'\tilde{\mathbf z}_{i,t-1})' = \tilde{\mathbf z}_{i,t-1}'\bv\Gamma$（转置一个乘积，要把次序倒过来）；$1\times(L+1)$ 乘 $(L+1)\times K$ 乘 $K\times 1$，得到一个数。第二个等号把 $\bv\theta'\mathbf z_{i,t-1}$ 也写成 $\tilde{\mathbf z}_{i,t-1}'(0, \bv\theta')'$：在 $\bv\theta$ 上面补一个 0，与 $\tilde{\mathbf z}_{i,t-1}$ 的常数 1 对齐。写出分量，

```{math}
:enumerated: false
\mathbf g_t = \begin{pmatrix} g_{0,t} \\ g_{1,t} \\ \vdots \\ g_{L,t} \end{pmatrix}
= \begin{pmatrix} \Gamma_{0,1} f_{1,t} + \dots + \Gamma_{0,K} f_{K,t} \\ \Gamma_{1,1} f_{1,t} + \dots + \Gamma_{1,K} f_{K,t} + \theta_1 \\ \vdots \\ \Gamma_{L,1} f_{1,t} + \dots + \Gamma_{L,K} f_{K,t} + \theta_L \end{pmatrix} , \qquad
\tilde{\mathbf z}_{i,t-1}'\mathbf g_t = g_{0,t} + z_{1,i,t-1}\, g_{1,t} + \dots + z_{L,i,t-1}\, g_{L,t} .
```

$g_{j,t}$（$j \geq 1$）是第 $t$ 期特征 $j$ 每高 1 个标准差、股票收益高出的部分，可以理解为特征 $j$ 在这一期的“回报”，单位与收益相同；$g_{0,t}$ 是全部特征处于平均水平的股票的收益（不计异质性冲击）。$g_{j,t}$ 由两部分组成：$\bv\Gamma$ 的第 $j$ 行与 $\mathbf f_t$ 的内积，即特征 $j$ 通过载荷带来的因子收益；以及定价误差系数 $\theta_j$。默认 $\bv\Gamma$ 下 $\mathbf g_t = (f_{1,t},\ 0.5 f_{2,t} + \theta_1,\ 0.5 f_{3,t} + \theta_2,\ \theta_3,\ \theta_4,\ \theta_5)'$。**特例**：Barra 的口径 $K = L+1$、$\bv\Gamma = \mathbf I_{L+1}$ 下，$\mathbf g_t = \mathbf f_t + (0, \bv\theta')'$，每期的回归系数就是“因子收益”（再加上定价误差系数）。**数据矩阵**：同一期 $N$ 只股票共用一个 $\mathbf g_t$，每只股票各有一行特征 $\tilde{\mathbf z}_{i,t-1}'$，堆叠后的设计矩阵就是 0.4 节写出的 $N\times(L+1)$ 的 $\tilde{\mathbf Z}_{t-1}$（[拆解](#ch0-unpack-return)）；不同时期的 $\mathbf g_t$ 不同，$T$ 期共有 $T$ 个。

在第 $t$ 期的横截面上，$\mathbf f_t$ 是所有股票共同的一个已实现值，$\mathbf g_t \in \Real^{L+1}$ 因而是一个对所有股票相同的系数向量：收益是期初特征的精确线性函数，只多了异质性冲击。把 $N$ 只股票按行堆叠，$\mathbf r_t = \tilde{\mathbf Z}_{t-1}\mathbf g_t + \bv\varepsilon_t$，最小二乘估计为

```{math}
:enumerated: false
\underbrace{\hat{\mathbf g}_t}_{(L+1)\times 1} = (\tilde{\mathbf Z}_{t-1}'\tilde{\mathbf Z}_{t-1})^{-1}\tilde{\mathbf Z}_{t-1}'\mathbf r_t = \mathbf g_t + \underbrace{(\tilde{\mathbf Z}_{t-1}'\tilde{\mathbf Z}_{t-1})^{-1}}_{(L+1)\times(L+1)}\,\underbrace{\tilde{\mathbf Z}_{t-1}'}_{(L+1)\times N}\,\underbrace{\bv\varepsilon_t}_{N\times 1} .
```

(ch0-unpack-ghat)=
**拆开讲 $\hat{\mathbf g}_t$。** 式中的两个矩阵乘积都是对 $N$ 只股票求和：$\tilde{\mathbf Z}_{t-1}'\tilde{\mathbf Z}_{t-1}$ 是 $(L+1)\times N$ 乘 $N\times(L+1)$，得 $(L+1)\times(L+1)$，$(j, l)$ 元素为 $\sum_{i=1}^{N} \tilde z_{j,i,t-1}\tilde z_{l,i,t-1}$；$\tilde{\mathbf Z}_{t-1}'\mathbf r_t$ 是 $(L+1)\times N$ 乘 $N\times 1$，得 $(L+1)\times 1$，第 $j$ 个元素为 $\sum_{i=1}^{N} \tilde z_{j,i,t-1} r_{i,t}$。只有一个特征（$L = 1$）时写出来是

```{math}
:enumerated: false
\underbrace{\tilde{\mathbf Z}_{t-1}'\tilde{\mathbf Z}_{t-1}}_{2\times 2} = \begin{pmatrix} N & \sum_i z_{1,i,t-1} \\ \sum_i z_{1,i,t-1} & \sum_i z_{1,i,t-1}^2 \end{pmatrix} , \qquad
\underbrace{\tilde{\mathbf Z}_{t-1}'\mathbf r_t}_{2\times 1} = \begin{pmatrix} \sum_i r_{i,t} \\ \sum_i z_{1,i,t-1} r_{i,t} \end{pmatrix} .
```

**特例**：$L = 0$（只有常数项）时 $\hat g_{0,t} = N^{-1}\sum_i r_{i,t}$，就是这一期全部股票收益的横截面平均；$L = 1$ 时，斜率 $\hat g_{1,t}$ 等于 $z_1$ 与收益的横截面样本协方差除以 $z_1$ 的横截面样本方差，截距等于收益的横截面均值减去斜率乘以 $z_1$ 的横截面均值。这与“用回归斜率估计载荷”中的单因子公式形式相同，只是求和对象从“同一只股票的 $T$ 个月”换成了“同一个月的 $N$ 只股票”。特征经过标准化、相互独立时 $\tilde{\mathbf Z}_{t-1}'\tilde{\mathbf Z}_{t-1}/N \approx \mathbf I_{L+1}$，于是 $\hat g_{j,t} \approx N^{-1}\sum_i z_{j,i,t-1} r_{i,t}$，近似于一个按特征 $j$ 加权的多空组合的收益。**数据矩阵**：与时间序列回归正好相反。时间序列回归中，所有股票共用一个 $T\times(1+K)$ 的设计矩阵 $[\mathbf 1_T, \mathbf F]$，每只股票各估一套 $\bv\beta_i$；这里每一期有自己的 $N\times(L+1)$ 设计矩阵 $\tilde{\mathbf Z}_{t-1}$（特征每期都在变），这一期的全部股票共用一个系数向量 $\hat{\mathbf g}_t$，每期各估一个。下面代码中的回归只放常数与 $z_1$、$z_2$（理由见下文）：`X_cs` 的形状为 `(T, N, 3)`，`X_cs[t-1]` 是第 $t$ 期 $\tilde{\mathbf Z}_{t-1}$ 的前三列（$N\times 3$）；`g_hat` 的形状为 `(T, 3)`，`g_hat[t-1]` 是这个缩减回归在第 $t$ 期的 3 个系数，估计的是 $\mathbf g_t$ 的前三个分量 $(g_{0,t}, g_{1,t}, g_{2,t})$，而不是完整的 $(L+1)$ 维 $\hat{\mathbf g}_t$。

```{code-cell} python
# 拆开讲 ĝ_t 的两个特例（第 1 期）：只有常数项时为横截面均值；只有 z1 时斜率为横截面协方差除以方差
r_p1, z1_p1 = exret_f[0], Z1_f[0]                                         # 第 1 期的 r_t 与 z1，各为 (N,)
g_const = np.linalg.lstsq(np.ones((N_f, 1)), r_p1, rcond=None)[0]        # (N, 1) 的设计矩阵 -> (1,)
g_uni = np.linalg.lstsq(np.column_stack([np.ones(N_f), z1_p1]), r_p1, rcond=None)[0]   # (N, 2) -> (2,)
print(f"只有常数项：{g_const[0]:.5f}，横截面均值 {r_p1.mean():.5f}；只有 z1：截距 {g_uni[0]:.5f}，斜率 {g_uni[1]:.5f}")
assert np.isclose(g_const[0], r_p1.mean())
assert np.isclose(g_uni[1], np.cov(z1_p1, r_p1)[0, 1] / np.var(z1_p1, ddof=1))
assert np.isclose(g_uni[0], r_p1.mean() - g_uni[1] * z1_p1.mean())
```

$\bv\varepsilon_t$ 与期初特征和因子独立、均值为 0，所以给定 $\tilde{\mathbf Z}_{t-1}$ 与 $\mathbf f_t$，$\hat{\mathbf g}_t$ 是 $\mathbf g_t$ 的无偏估计，误差的条件协方差矩阵为 $\sigma_\varepsilon^2(\tilde{\mathbf Z}_{t-1}'\tilde{\mathbf Z}_{t-1})^{-1}$（$(L+1)\times(L+1)$ 矩阵，第 $j$ 个对角元是系数估计 $\hat g_{j,t}$ 的条件方差，非对角元是两个系数估计之间的条件协方差）。模拟器中的特征相互独立、服从 $N(\mathbf 0, \mathbf I_L)$，$\tilde{\mathbf Z}_{t-1}'\tilde{\mathbf Z}_{t-1}/N$ 接近 $\mathbf I_{L+1}$，所以误差每个分量的标准差约为 $\sigma_\varepsilon/\sqrt N$：默认 $\sigma_\varepsilon = 0.10$、$N = 500$ 时约为 0.45%。这是同一期横截面上的噪声，随股票数增加而变小。默认 $\bv\Gamma$ 下 $\bv\Gamma\mathbf f_t = (f_{1,t},\ 0.5 f_{2,t},\ 0.5 f_{3,t},\ 0,\ 0,\ 0)'$，于是第 $t$ 期对常数与 $z_1$、$z_2$ 的系数估计为

```{math}
:enumerated: false
(\hat g_{0,t},\ \hat g_{1,t},\ \hat g_{2,t}) \approx (f_{1,t},\ 0.5 f_{2,t},\ 0.5 f_{3,t}) + (0,\ \theta_1,\ \theta_2) + \text{横截面噪声} .
```

$z_3$～$z_5$ 的系数真值为 0（假定 $\bv\theta$ 也只作用于 $z_1$、$z_2$），所以回归中只放常数与 $z_1$、$z_2$，结论不变。系数是 $0.5 f_{2,t}$ 而不是 $f_{2,t}$，因为每单位 $z_1$ 带来的载荷是 0.5：Barra 式回归估计的“因子收益”是 $\bv\Gamma\mathbf f_t$，即真实因子经过特征到载荷的映射换算之后的收益。$\bv\theta \neq \mathbf 0$ 时，系数还包含定价误差 $\bv\theta$：横截面回归本身无法区分特征带来的是风险补偿还是错误定价，这与“常见误区”一节的误区四是同一件事。

```{code-cell} python
# 逐期横截面回归：r_t 对 (1, z1, z2)，与 (f1, 0.5 f2, 0.5 f3) 比较（默认 Γ、θ = 0）
X_cs = np.concatenate([np.ones((T_f, N_f, 1)),
                       fcl.panel[["z1", "z2"]].to_numpy().reshape(T_f, N_f, 2)], axis=2)   # (T, N, 3)：第 t 期为 [1_N, z1, z2]
g_hat = np.array([np.linalg.lstsq(X_cs[t], exret_f[t], rcond=None)[0] for t in range(T_f)])  # (T, 3)：g_hat[t-1] 为第 t 期缩减回归的 3 个系数，估计 g_t 的前三个分量
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

