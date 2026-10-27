---
kernelspec:
  name: python3
  display_name: Python 3
numbering:
  enumerator: '0.%s'
  equation:
    continue: true
  exercise:
    template: 习题 %s
---

# A 股视角与模拟实验

**本页准备。** 本书每个页面的代码在独立的内核中运行，前面页面定义的变量在这里不可用。下面的单元导入本页用到的模块，并按与 0.4 节完全相同的参数和种子，重建本页用到的默认市场及由它算出的数组。

```{code-cell} python
import numpy as np
import pandas as pd

from apsim import MarketParams, simulate_market, stats

# 0.4 节的默认市场（种子 0）及其逐行的载荷 β_{i,t}' 与因子 f_t'
params = MarketParams()
mkt = simulate_market(params, seed=0)
L, K = params.n_chars, params.n_factors
beta = mkt.panel[[f"beta{k}" for k in range(1, K + 1)]].to_numpy()                  # (TN, K)，每行 β_{i,t}'
f_row = mkt.factors.reindex(mkt.panel.index.get_level_values("period")).to_numpy()  # (TN, K)，每行 f_t'
```

## A 股视角

本章的模拟市场刻意保持简单，以便先把定价方程、时点约定与“已知真值”的检验逻辑讲清楚。与真实的 A 股市场相比，它至少省略了以下几类特征：

1. **涨跌停。** A 股对单日涨跌幅设有限制，不同板块与 ST 股票的限制比例不同，并随制度改革调整。涨跌停会截断日收益、可能推迟价格对信息的反映，也让触及涨跌停的股票在当日实际上难以按收盘价成交。第 15 章在 `apsim.frictions` 中加入这一机制，第 17 章讨论它对资产定价检验的影响。
2. **停牌。** 停牌期间没有成交价，复牌后的收益集中体现了停牌期间积累的信息。月度数据中如何处理停牌股票的收益与特征，会直接影响组合收益，详见第 15 章。
3. **卖空限制。** 本章把因子当作可以无成本构造的多空组合。A 股融券规模小、成本高，许多“空头腿”在现实中难以实现，这既影响因子收益的可实现性（第 26 章），也是错误定价能够持续存在的一个原因（第 8 章）。
4. **壳价值。** 在发行上市受到严格管制的时期，小市值公司作为借壳上市的潜在对象而具有额外价值，这使市值最小的股票的收益不能用一般的风险补偿解释。Liu、Stambaugh 与 Yuan 构造中国市场的规模与价值因子时，因此剔除了市值最小的 30% 的股票 [@liu2019size]。第 16、17 章讨论这一处理。
5. **财报披露滞后。** 会计期末的财务数据要到数月之后才公开（例如年报通常在次年 4 月底前披露）。如果按会计期末而不是实际披露日对齐，特征 $\mathbf z_{i,t-1}$ 中就混入了第 $t-1$ 期末尚不可得的信息，即本章强调的时点错位。正式研究必须使用带有公告日期的时点数据，第 15 章给出对齐规则。

此外，模拟器中的因子收益是独立同分布的正态变量，特征服从相互独立的正态 AR(1)，异质性冲击在股票之间不相关，股票数量在样本期内不变。真实数据中的厚尾、波动聚集、行业内的同期相关、新股上市与退市，都会在相应章节逐步加入。中国股市中机器学习方法的实证表现，可先参阅 Leippold、Wang 与 Zhou 的研究 [@leippold2022machine]，第四部分会详细讨论。

(sec-ch0-interact)=
## 模拟实验：用交互回归恢复 Γ

### 方法

把式 {eq}`eq-ch0-beta` 代入式 {eq}`eq-ch0-ret`，在 $\bv\theta = \mathbf 0$ 时：

$$
r_{i,t} = \underbrace{\tilde{\mathbf z}_{i,t-1}'}_{1\times(L+1)} \underbrace{\bv\Gamma}_{(L+1)\times K} \underbrace{\mathbf f_t}_{K\times 1} + \varepsilon_{i,t} = \sum_{j=0}^{L} \sum_{k=1}^{K} \Gamma_{j,k}\, \left(\tilde z_{j,i,t-1} f_{k,t}\right) + \varepsilon_{i,t},
$$ (eq-ch0-interact)

其中 $\tilde z_{0,i,t-1} = 1$，$\tilde z_{j,i,t-1} = z_{j,i,t-1}$（$j \geq 1$）。这是一个以 $(L+1)K$ 个“特征 × 因子”交互项为解释变量、以 $\bv\Gamma$ 的各元素为系数的线性回归。

(ch0-unpack-interact)=
**拆开讲式 {eq}`eq-ch0-interact`。** 第一个等号右边的 $\tilde{\mathbf z}_{i,t-1}'\bv\Gamma\mathbf f_t$ 是 $1\times(L+1)$、$(L+1)\times K$、$K\times 1$ 三者之积，结果是一个数，就是因子部分 $\bv\beta_{i,t}'\mathbf f_t$（$\tilde{\mathbf z}_{i,t-1}$ 与 $\bv\Gamma$ 的分量见[拆解](#ch0-unpack-gamma)）。第二个等号把矩阵乘法逐项展开：先算 $\bv\Gamma\mathbf f_t$ 的第 $j$ 个分量 $\sum_k \Gamma_{j,k} f_{k,t}$，再乘以 $\tilde z_{j,i,t-1}$、对 $j$ 求和，一共 $(L+1)K$ 项，每一项是“系数 $\Gamma_{j,k}$ 乘以交互项 $\tilde z_{j,i,t-1} f_{k,t}$”。交互项是特征（以横截面标准差为单位）与因子收益之积，单位与收益相同；$j = 0$ 的交互项 $\tilde z_{0,i,t-1} f_{k,t} = f_{k,t}$ 就是因子本身。默认参数下 $L = 5$、$K = 3$，共 18 项，但默认 $\bv\Gamma$ 只有三个非零元素：

```{math}
:enumerated: false
r_{i,t} = \Gamma_{0,1} f_{1,t} + \Gamma_{1,2}\, z_{1,i,t-1} f_{2,t} + \Gamma_{2,3}\, z_{2,i,t-1} f_{3,t} + \varepsilon_{i,t} = f_{1,t} + 0.5\, z_{1,i,t-1} f_{2,t} + 0.5\, z_{2,i,t-1} f_{3,t} + \varepsilon_{i,t} ,
```

其余 15 个交互项的真实系数为 0，回归要从数据中发现这一点。**特例**：$L = 0$（没有特征）时只剩 $\sum_k \Gamma_{0,k} f_{k,t}$，所有股票的载荷相同，回归退化为所有股票共用一组斜率、截距为 0 的时间序列回归；$L = 1$、$K = 1$ 时 $r_{i,t} = \Gamma_{0,1} f_{1,t} + \Gamma_{1,1}\, z_{1,i,t-1} f_{1,t} + \varepsilon_{i,t}$，即载荷 $\beta_{1,i,t} = \Gamma_{0,1} + \Gamma_{1,1} z_{1,i,t-1}$ 随一个特征线性变化，$f_{1,t}$ 取市场因子时就是条件 CAPM 常用的“载荷随特征变化”的回归。**数据矩阵**：每个观测 $(i, t)$ 贡献一行 $(L+1)K$ 个交互项，$TN$ 个观测叠成下文的 $\mathbf X$；**$\bv\Gamma$ 由所有股票、所有时期共用**，所以只需估计 $(L+1)K$ 个系数，而不是每只股票一套。

**写成标准的回归形式。** 要把 $\bv\Gamma$ 的元素排成一个系数向量，需要用到 vec 算子与 Kronecker 积。$\operatorname{vec}(\mathbf A)$ 把矩阵 $\mathbf A$ 的各列从左到右依次叠成一个长的列向量（列优先）；Kronecker 积 $\mathbf a \otimes \mathbf c$ 把 $\mathbf a$ 的每个元素 $a_j$ 换成块 $a_j \mathbf c$。二者由恒等式

$$
\operatorname{vec}(\mathbf A \mathbf B \mathbf C) = (\mathbf C' \otimes \mathbf A)\operatorname{vec}(\mathbf B)
$$ (eq-ch0-vec)

联系起来（$\mathbf A$、$\mathbf B$、$\mathbf C$ 为任意可以相乘的矩阵）。标量 $\tilde{\mathbf z}_{i,t-1}'\bv\Gamma\mathbf f_t$ 等于它的转置 $\mathbf f_t'\bv\Gamma'\tilde{\mathbf z}_{i,t-1}$，取 $\mathbf A = \mathbf f_t'$（$1\times K$）、$\mathbf B = \bv\Gamma'$（$K\times(L+1)$）、$\mathbf C = \tilde{\mathbf z}_{i,t-1}$（$(L+1)\times 1$），一个标量的 vec 就是它自己，于是

$$
r_{i,t} = \underbrace{(\tilde{\mathbf z}_{i,t-1} \otimes \mathbf f_t)'}_{1\times(L+1)K}\; \underbrace{\operatorname{vec}(\bv\Gamma')}_{(L+1)K\times 1} + \varepsilon_{i,t} .
$$ (eq-ch0-kron)

这里用到 $(\tilde{\mathbf z}_{i,t-1}' \otimes \mathbf f_t') = (\tilde{\mathbf z}_{i,t-1} \otimes \mathbf f_t)'$。系数向量 $\operatorname{vec}(\bv\Gamma')$ 按列叠放 $\bv\Gamma'$，也就是按行叠放 $\bv\Gamma$：先是第 0 行 $(\Gamma_{0,1}, \dots, \Gamma_{0,K})$，再是第 1 行，依此类推，$j$ 在外层、$k$ 在内层；设计矩阵的第 $(i,t)$ 行 $(\tilde{\mathbf z}_{i,t-1} \otimes \mathbf f_t)'$ 的元素也按同样的顺序排列为 $\tilde z_{j,i,t-1} f_{k,t}$。把全部 $TN$ 个观测按行叠成 $\mathbf X \in \Real^{TN\times(L+1)K}$ 与 $\mathbf y \in \Real^{TN}$，因子可观测时，对 $\mathbf y$ 关于 $\mathbf X$ 做最小二乘，就能估计 $\operatorname{vec}(\bv\Gamma')$。

(ch0-unpack-vec)=
**拆开讲 vec、Kronecker 积与式 {eq}`eq-ch0-vec`。** 先看两个最小的例子：$2\times 2$ 矩阵按列叠放，以及 $2\times 1$ 与 $3\times 1$ 两个列向量的 Kronecker 积，

```{math}
:enumerated: false
\operatorname{vec}\begin{pmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{pmatrix} = \begin{pmatrix} a_{11} \\ a_{21} \\ a_{12} \\ a_{22} \end{pmatrix} , \qquad
\begin{pmatrix} a_1 \\ a_2 \end{pmatrix} \otimes \begin{pmatrix} c_1 \\ c_2 \\ c_3 \end{pmatrix} = \begin{pmatrix} a_1\mathbf c \\ a_2\mathbf c \end{pmatrix} = \begin{pmatrix} a_1 c_1 \\ a_1 c_2 \\ a_1 c_3 \\ a_2 c_1 \\ a_2 c_2 \\ a_2 c_3 \end{pmatrix} .
```

**维度**：$m\times n$ 矩阵的 vec 是 $mn\times 1$ 的列向量；$m\times 1$ 的 $\mathbf a$ 与 $n\times 1$ 的 $\mathbf c$ 的 Kronecker 积是 $mn\times 1$，$\mathbf a$ 的下标在外层（变化慢），$\mathbf c$ 的下标在内层（变化快）。一般地，$m\times n$ 的 $\mathbf A$ 与 $p\times q$ 的 $\mathbf B$ 的 Kronecker 积 $\mathbf A\otimes\mathbf B$ 是 $mp\times nq$ 的分块矩阵，第 $(j, l)$ 块为 $a_{jl}\mathbf B$。式 {eq}`eq-ch0-vec` 两边的维度：设 $\mathbf A$ 为 $m\times n$、$\mathbf B$ 为 $n\times p$、$\mathbf C$ 为 $p\times q$，左边 $\operatorname{vec}(\mathbf A\mathbf B\mathbf C)$ 是 $mq\times 1$；右边 $\mathbf C'\otimes\mathbf A$ 是 $qm\times pn$，乘 $np\times 1$ 的 $\operatorname{vec}(\mathbf B)$，也是 $mq\times 1$。**特例**：$\mathbf A = \mathbf a'$（$1\times n$）、$\mathbf C = \mathbf c$（$p\times 1$）时左边是标量 $\mathbf a'\mathbf B\mathbf c$，恒等式化为

```{math}
:enumerated: false
\mathbf a'\mathbf B\mathbf c = (\mathbf c'\otimes\mathbf a')\operatorname{vec}(\mathbf B) = \sum_{k=1}^{p}\sum_{j=1}^{n} c_k\, a_j\, B_{jk} ,
```

右边就是双重求和 $\sum_j\sum_k a_j B_{jk} c_k$ 换了一种排列：$\operatorname{vec}(\mathbf B)$ 中 $B_{jk}$ 的行下标 $j$ 变化快，$\mathbf c'\otimes\mathbf a'$ 中 $a_j$ 的下标 $j$ 也变化快，两者逐个对齐。式 {eq}`eq-ch0-kron` 正是取 $\mathbf a = \mathbf f_t$、$\mathbf B = \bv\Gamma'$、$\mathbf c = \tilde{\mathbf z}_{i,t-1}$ 的这一特例。

(ch0-unpack-kron)=
**拆开讲式 {eq}`eq-ch0-kron`。** 写出分量，设计矩阵一行（转置成列）与系数向量是

```{math}
:enumerated: false
\underbrace{\tilde{\mathbf z}_{i,t-1}\otimes\mathbf f_t}_{(L+1)K\times 1} = \begin{pmatrix} 1\cdot\mathbf f_t \\ z_{1,i,t-1}\,\mathbf f_t \\ \vdots \\ z_{L,i,t-1}\,\mathbf f_t \end{pmatrix} = \begin{pmatrix} f_{1,t} \\ \vdots \\ f_{K,t} \\ z_{1,i,t-1}f_{1,t} \\ \vdots \\ z_{L,i,t-1}f_{K,t} \end{pmatrix} , \qquad
\underbrace{\operatorname{vec}(\bv\Gamma')}_{(L+1)K\times 1} = \begin{pmatrix} \Gamma_{0,1} \\ \vdots \\ \Gamma_{0,K} \\ \Gamma_{1,1} \\ \vdots \\ \Gamma_{L,K} \end{pmatrix} .
```

两者都分成 $L+1$ 段、每段 $K$ 个元素：第 $j$ 段分别是 $\tilde z_{j,i,t-1}\mathbf f_t$ 与 $\bv\Gamma$ 的第 $j$ 行（转置成列）。**维度**：$1\times(L+1)K$ 的行乘 $(L+1)K\times 1$ 的列，得到一个数。**展开运算**：对应位置相乘再加总，就是式 {eq}`eq-ch0-interact` 的双重求和 $\sum_j\sum_k \Gamma_{j,k}\,\tilde z_{j,i,t-1}f_{k,t}$。以 0.2 节的股票 A 为例（$z_1 = 2$、其余特征为 0，$\mathbf f_t = (4\%, 2\%, -1\%)'$，见[拆解](#ch0-unpack-gamma)）：$\tilde{\mathbf z}_{A,t-1}\otimes\mathbf f_t$ 的前两段为 $(4\%, 2\%, -1\%)$ 与 $(8\%, 4\%, -2\%)$，其余 12 个元素为 0；默认 $\operatorname{vec}(\bv\Gamma')$ 的前两段为 $(1, 0, 0)$ 与 $(0, 0.5, 0)$，于是内积为 $1 \times 4\% + 0.5 \times 4\% = 6\%$，与 0.2 节算出的因子部分相同（下方代码核对）。**特例**：$L = 0$ 时 $\tilde{\mathbf z}_{i,t-1} = 1$，$\tilde{\mathbf z}_{i,t-1}\otimes\mathbf f_t = \mathbf f_t$，$\operatorname{vec}(\bv\Gamma')$ 就是所有股票共同的载荷向量。**数据矩阵**：$\mathbf X$ 与 $\mathbf y$ 的行按“期在外层、股票在内层”排列：先是第 1 期的 $N$ 只股票，再是第 2 期的 $N$ 只股票，依此类推，与面板的 `(period, stock)` 顺序相同；第 $(i, t)$ 行是 $(\tilde{\mathbf z}_{i,t-1}\otimes\mathbf f_t)'$，$\mathbf y$ 的对应元素是 $r_{i,t}$。同一期的 $N$ 行共用同一个 $\mathbf f_t$，各行的 $\tilde{\mathbf z}_{i,t-1}$ 不同；系数向量 $\operatorname{vec}(\bv\Gamma')$ 由全部 $TN$ 行共用。默认参数下 $\mathbf X$ 为 $120000\times 18$。

```{code-cell} python
# 拆开讲 vec 与 eq-ch0-kron：vec 按列叠放；股票 A（z1 = 2，其余为 0）在 f_t = (4%, 2%, -1%)' 下，(z̃ ⊗ f)' vec(Γ') = 6%
assert np.array_equal(np.array([[11, 12], [21, 22]]).ravel(order="F"), [11, 21, 12, 22])   # vec：列优先
zt_A = np.array([1.0, 2.0, 0.0, 0.0, 0.0, 0.0])                   # z̃_A: (L+1,) = (6,)
f_A = np.array([0.04, 0.02, -0.01])                               # f_t: (K,) = (3,)
x_A = np.kron(zt_A, f_A)                                          # z̃ ⊗ f: ((L+1)K,) = (18,)
print("z̃ ⊗ f 的前两段：", x_A[:6], "；与 vec(Γ') 的内积", round(x_A @ mkt.gamma.ravel(), 4))
assert np.allclose(x_A[:6], [0.04, 0.02, -0.01, 0.08, 0.04, -0.02]) and np.allclose(x_A[6:], 0)
assert np.isclose(x_A @ mkt.gamma.ravel(), 0.06) and np.isclose(x_A @ mkt.gamma.ravel(), zt_A @ mkt.gamma @ f_A)
```

**与代码的对应。** 下面的 `interaction_design` 用广播构造 $\mathbf X$：`zt[:, :, None] * ff[:, None, :]` 的形状为 `(TN, L+1, K)`，其 `[n, j, k]` 元素是 $\tilde z_j f_k$；NumPy 的 `reshape` 默认按行优先（C 顺序）展开，最后一维 `k` 变化最快，所以展开后每一行正是 $(\tilde{\mathbf z} \otimes \mathbf f)'$。同理，`gamma.ravel()`（行优先）就是 $\operatorname{vec}(\bv\Gamma')$，而列优先的 `gamma.ravel(order="F")` 才是 $\operatorname{vec}(\bv\Gamma)$；系数向量用 `coef.reshape(L+1, K)` 还原为 $\hat{\bv\Gamma}$。数学上的 vec 是列优先，NumPy 的默认展开是行优先，两者差一个转置，这是最容易出错的地方，下面的代码单元用 `np.kron` 与 `assert` 核对这一对应。

由于 $\varepsilon_{i,t}$ 与 $(\mathbf z_{i,t-1}, \mathbf f_t)$ 独立，解释变量严格外生，最小二乘估计量无偏，其条件协方差矩阵为 $\sigma_\varepsilon^2 (\mathbf X'\mathbf X)^{-1}$（$(L+1)K \times (L+1)K$）。函数 `ols` 用 `lstsq` 求系数、用 `solve` 求 $(\mathbf X'\mathbf X)^{-1}$，而不调用 `inv`。这两个函数也收在 `apsim.stats` 中，代码单元用 `assert` 核对二者与这里的定义结果相同，后面的页面直接调用 `apsim.stats`。下面用默认参数（$N = 500$、$T = 240$）检验这一点。

(ch0-unpack-xtx)=
**拆开讲 $\sigma_\varepsilon^2(\mathbf X'\mathbf X)^{-1}$。** 记 $\mathbf x_{i,t} = \tilde{\mathbf z}_{i,t-1}\otimes\mathbf f_t$，它是设计矩阵第 $(i, t)$ 行的转置，为 $(L+1)K\times 1$。$(L+1)K\times TN$ 的 $\mathbf X'$ 乘 $TN\times(L+1)K$ 的 $\mathbf X$，得到 $(L+1)K\times(L+1)K$ 的方阵，它等于全部观测的外积之和；$\mathbf X'\mathbf y$ 是 $(L+1)K\times 1$：

```{math}
:enumerated: false
\mathbf X'\mathbf X = \sum_{t=1}^{T}\sum_{i=1}^{N} \mathbf x_{i,t}\mathbf x_{i,t}' , \qquad \mathbf X'\mathbf y = \sum_{t=1}^{T}\sum_{i=1}^{N} \mathbf x_{i,t}\, r_{i,t} , \qquad \operatorname{vec}(\hat{\bv\Gamma}') = (\mathbf X'\mathbf X)^{-1}\mathbf X'\mathbf y .
```

$\mathbf X'\mathbf X$ 中对应 $\Gamma_{j,k}$ 与 $\Gamma_{l,m}$ 的元素是 $\sum_{t}\sum_{i}\tilde z_{j,i,t-1}\tilde z_{l,i,t-1}f_{k,t}f_{m,t}$，即两个交互项的乘积之和。$\sigma_\varepsilon^2(\mathbf X'\mathbf X)^{-1}$ 的第 $p$ 个对角元是第 $p$ 个系数估计的条件方差，开平方就是标准误（代码中的 `np.sqrt(np.diag(s2 * XtX_inv))`）；非对角元是两个系数估计之间的条件协方差。$\sigma_\varepsilon^2$ 未知，代码用残差平方和除以“观测数减系数个数”估计，即 `s2`。**特例**：只有一个解释变量 $x_n$ 时，$\mathbf X'\mathbf X = \sum_n x_n^2$ 是一个数，系数估计的方差为 $\sigma_\varepsilon^2/\sum_n x_n^2$：解释变量变化越大，系数估计越准，这也是下文“第一个因子一列的标准误更小”的原因。

```{code-cell} python
def interaction_design(market):
    """返回交互项设计矩阵 X（行：(t, i)；列：z̃_j f_k，j 在外层、k 在内层）与期初特征矩阵 Z。"""
    pnl = market.panel
    n_chars = market.params.n_chars
    Z = pnl[[f"z{j}" for j in range(1, n_chars + 1)]].to_numpy()               # (TN, L)
    zz = np.column_stack([np.ones(len(pnl)), Z])                               # (TN, L+1)，每行 z̃'
    ff = market.factors.reindex(pnl.index.get_level_values("period")).to_numpy()   # (TN, K)，每行 f_t'
    X = (zz[:, :, None] * ff[:, None, :]).reshape(len(pnl), -1)                # (TN, L+1, K) -> (TN, (L+1)K)
    return X, Z


def ols(X, y):
    """最小二乘系数与同方差标准误。X: (n, p)，y: (n,)。"""
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)                               # (p,)
    resid = y - X @ coef                                                       # (n, p) @ (p,) -> (n,)
    s2 = resid @ resid / (len(y) - X.shape[1])                                 # σ̂_ε²
    XtX_inv = np.linalg.solve(X.T @ X, np.eye(X.shape[1]))                     # (X'X)⁻¹：(p, p)
    return coef, np.sqrt(np.diag(s2 * XtX_inv))


X, _ = interaction_design(mkt)
# 核对 reshape 顺序：第 n 行等于 z̃ ⊗ f，且 X @ vec(Γ') 等于 β'f
zt_row = np.r_[1.0, mkt.panel.iloc[7][[f"z{j}" for j in range(1, L + 1)]].to_numpy()]
f_row7 = mkt.factors.loc[mkt.panel.index[7][0]].to_numpy()
assert np.allclose(X[7], np.kron(zt_row, f_row7))
assert np.allclose(X @ mkt.gamma.ravel(), np.sum(beta * f_row, axis=1))       # (TN, (L+1)K) @ ((L+1)K,)
assert not np.allclose(X @ mkt.gamma.ravel(order="F"), np.sum(beta * f_row, axis=1))

coef, se = ols(X, mkt.panel["exret"].to_numpy())
# 上面的内联定义与 apsim.stats 中的同名函数结果相同：比较设计矩阵 X、Z，以及系数与标准误
X_lib, Z_lib = stats.interaction_design(mkt)
coef_lib, se_lib = stats.ols(X, mkt.panel["exret"].to_numpy())
assert np.array_equal(X, X_lib)
assert np.array_equal(interaction_design(mkt)[1], Z_lib)
assert np.allclose(coef, coef_lib)
assert np.allclose(se, se_lib)
names = [f"Γ[{j},{k + 1}]" for j in range(params.n_chars + 1) for k in range(K)]
table = pd.DataFrame(
    {"真值": mkt.gamma.ravel(), "估计值": coef, "标准误": se, "标准化偏差": (coef - mkt.gamma.ravel()) / se},
    index=names,
)
print(table.round(4).to_string())
n_big = int(np.sum(np.abs(table["标准化偏差"]) > 2))
print(f"标准化偏差绝对值超过 2 的系数：{n_big} 个（标准正态分布下 18 个中平均约 {18 * 0.0455:.1f} 个）")
assert np.all(np.abs(table["标准化偏差"]) < 4)
```

表中“标准化偏差”为 $(\hat\Gamma_{j,k} - \Gamma_{j,k}) / \text{se}$。18 个系数的估计值都落在真值附近（`assert` 保证每个标准化偏差的绝对值小于 4）。单次模拟中，绝对值超过 2 的标准化偏差有几个，取决于种子（见上方输出）；若估计量无偏、标准误正确，18 个系数中平均约有 0.8 个。单次模拟无法区分“抽样误差”与“标准误算错了”，要靠重复模拟：如果估计量无偏、标准误正确，标准化偏差在多次重复中应当近似服从标准正态分布。

```{code-cell} python
zs = []
for r in range(30):
    rep = simulate_market(params, seed=1000 + r)
    Xr, _ = interaction_design(rep)
    cr, ser = ols(Xr, rep.panel["exret"].to_numpy())
    zs.append((cr - rep.gamma.ravel()) / ser)
zs = np.concatenate(zs)
frac_big = np.mean(np.abs(zs) > 2)
print(f"30 次重复、共 {zs.size} 个标准化偏差：均值 {zs.mean():.3f}，标准差 {zs.std():.3f}，"
      f"绝对值超过 2 的比例 {frac_big:.3f}（标准正态分布下为 0.046）")
# 540 个独立的标准正态抽样：均值、标准差与比例的抽样误差分别约为 0.043、0.030、0.009
assert abs(zs.mean()) < 0.13 and abs(zs.std() - 1) < 0.09 and abs(frac_big - 0.0455) < 0.027
se_grid = se.reshape(-1, K)                                                    # (L+1, K)，与 Γ 同形
assert np.all(se_grid[:, 0] < se_grid[:, 1:].min(axis=1))
```

重复模拟的结果与标准正态分布相符（代码中的 `assert` 要求均值、标准差与大偏差比例都落在约 3 倍抽样误差以内）：估计量无偏，同方差标准误也是正确的，单次模拟中大偏差的个数只反映抽样误差。另外，第一个因子一列（$k = 1$）的标准误小于另外两列：$\sigma_1 = 0.05$ 较大，交互项 $\tilde z_j f_{1,t}$ 的变化也更大，因而包含更多关于 $\Gamma_{j,1}$ 的信息。

### 当存在错误定价时

如果 $\bv\theta \neq \mathbf 0$，式 {eq}`eq-ch0-interact` 遗漏了 $\alpha_{i,t} = \bv\theta' \mathbf z_{i,t-1}$。遗漏项与解释变量相关吗？交互项 $z_{j,i,t-1} f_{k,t}$ 与 $z_{j,i,t-1}$ 的协方差为 $\E[z_{j,i,t-1}^2]\,\E[f_{k,t}] = \mu_k \neq 0$，所以答案是肯定的：只用交互项的回归有遗漏变量偏误。

逐项看这个协方差：$\Cov(z_{j,i,t-1}f_{k,t},\ z_{j,i,t-1}) = \E[z_{j,i,t-1}^2 f_{k,t}] - \E[z_{j,i,t-1}f_{k,t}]\,\E[z_{j,i,t-1}]$。特征与因子独立，第一项等于 $\E[z_{j,i,t-1}^2]\,\E[f_{k,t}] = 1 \times \mu_k$；第二项中 $\E[z_{j,i,t-1}] = 0$。所以协方差等于因子均值 $\mu_k$，单位与收益相同，例如默认参数下 $z_1 f_1$ 与 $z_1$ 的协方差为 $\mu_1 = 0.6\%$。遗漏项 $\theta_j z_{j,i,t-1}$ 因而有一部分会被交互项 $z_{j,i,t-1}f_{k,t}$“冒名顶替”。

(ch0-unpack-ovb)=
**先拆开下面推导要用到的几个矩阵。**

- 因子矩阵 $\mathbf F$ 是 $T\times K$，第 $t$ 行为 $\mathbf f_t'$（结构见[拆解](#ch0-unpack-f)）。$\mathbf F'\mathbf F$ 是 $K\times T$ 乘 $T\times K$，得 $K\times K$，等于 $\sum_t \mathbf f_t\mathbf f_t'$，$(k, l)$ 元素为 $\sum_t f_{k,t}f_{l,t}$；$\mathbf F'\mathbf 1_T$ 是 $K\times 1$，第 $k$ 个元素为 $\sum_t f_{k,t}$，即 $T$ 乘以第 $k$ 个因子的样本均值。
- $\tilde{\mathbf z}_{i,t-1}\tilde{\mathbf z}_{i,t-1}'$ 是 $(L+1)\times(L+1)$ 的外积，$(j, l)$ 元素为 $\tilde z_{j,i,t-1}\tilde z_{l,i,t-1}$。特征相互独立、均值为 0、方差为 1，所以横截面平均后对角元趋于 1，非对角元（包括常数项与特征的交叉，即特征的均值）趋于 0，整体趋于 $\mathbf I_{L+1}$。$\tilde{\mathbf z}_{i,t-1}\mathbf z_{i,t-1}'\bv\theta = \tilde{\mathbf z}_{i,t-1}\,\alpha_{i,t}$ 是 $(L+1)\times 1$，横截面平均后第 0 个分量趋于 $\E[\alpha_{i,t}] = 0$，第 $j$ 个分量趋于 $\E[z_{j,i,t-1}\,\bv\theta'\mathbf z_{i,t-1}] = \theta_j$，整体趋于 $(0, \bv\theta')'$。
- 混合乘积规则要求 $\mathbf A\mathbf C$ 与 $\mathbf B\mathbf D$ 都能相乘。用于设计矩阵的一行：$\mathbf x_{i,t}\mathbf x_{i,t}' = (\tilde{\mathbf z}_{i,t-1}\otimes\mathbf f_t)(\tilde{\mathbf z}_{i,t-1}'\otimes\mathbf f_t') = (\tilde{\mathbf z}_{i,t-1}\tilde{\mathbf z}_{i,t-1}')\otimes(\mathbf f_t\mathbf f_t')$，是 $(L+1)\times(L+1)$ 矩阵与 $K\times K$ 矩阵的 Kronecker 积，为 $(L+1)K\times(L+1)K$；同理 $\mathbf x_{i,t}\,\alpha_{i,t} = (\tilde{\mathbf z}_{i,t-1}\alpha_{i,t})\otimes\mathbf f_t$。同一期的 $\mathbf f_t$ 不随 $i$ 变化，所以对 $i$ 的平均只作用在特征一侧。
- $\bv\alpha$ 是 $TN\times 1$，元素按 $\mathbf y$ 的顺序排列，第 $(i, t)$ 个为 $\alpha_{i,t} = \bv\theta'\mathbf z_{i,t-1}$。

极限中的两个 Kronecker 积写出来是

```{math}
:enumerated: false
\mathbf I_{L+1}\otimes\mathbf F'\mathbf F = \begin{pmatrix} \mathbf F'\mathbf F & \mathbf 0 & \cdots & \mathbf 0 \\ \mathbf 0 & \mathbf F'\mathbf F & \cdots & \mathbf 0 \\ \vdots & \vdots & \ddots & \vdots \\ \mathbf 0 & \mathbf 0 & \cdots & \mathbf F'\mathbf F \end{pmatrix} , \qquad
\begin{pmatrix} 0 \\ \bv\theta \end{pmatrix}\otimes\mathbf F'\mathbf 1_T = \begin{pmatrix} 0\cdot\mathbf F'\mathbf 1_T \\ \theta_1\,\mathbf F'\mathbf 1_T \\ \vdots \\ \theta_L\,\mathbf F'\mathbf 1_T \end{pmatrix} .
```

前者是分块对角矩阵，对角线上是 $L+1$ 个相同的 $K\times K$ 块 $\mathbf F'\mathbf F$，其余为 0：不同特征的交互项在极限中互不相关，$L+1$ 段系数可以分开求解；分块对角矩阵的逆就是把每一块各自求逆，即 $\mathbf I_{L+1}\otimes(\mathbf F'\mathbf F)^{-1}$。后者是 $(L+1)K\times 1$，第 $j$ 段为 $\theta_j\mathbf F'\mathbf 1_T$。

偏误的大小可以算出来。记 $\mathbf F = (\mathbf f_1, \dots, \mathbf f_T)' \in \Real^{T\times K}$（第 $t$ 行为 $\mathbf f_t'$），$\mathbf 1_T$ 为 $T$ 维全 1 向量。以因子路径 $\mathbf F$ 为条件，令 $N \to \infty$。每一期的横截面平均满足 $N^{-1}\sum_i \tilde{\mathbf z}_{i,t-1} \tilde{\mathbf z}_{i,t-1}' \to \mathbf I_{L+1}$、$N^{-1}\sum_i \tilde{\mathbf z}_{i,t-1} \mathbf z_{i,t-1}' \bv\theta \to (0, \bv\theta')'$。由式 {eq}`eq-ch0-kron`，设计矩阵的行是 $(\tilde{\mathbf z}_{i,t-1} \otimes \mathbf f_t)'$；利用 Kronecker 积的混合乘积规则 $(\mathbf A \otimes \mathbf B)(\mathbf C \otimes \mathbf D) = \mathbf A\mathbf C \otimes \mathbf B\mathbf D$，

```{math}
:enumerated: false
\frac{1}{N}\mathbf X'\mathbf X = \sum_t \Bigl(\frac{1}{N}\sum_i \tilde{\mathbf z}_{i,t-1}\tilde{\mathbf z}_{i,t-1}'\Bigr) \otimes \mathbf f_t\mathbf f_t' \to \mathbf I_{L+1} \otimes \mathbf F'\mathbf F, \qquad
\frac{1}{N}\mathbf X'\bv\alpha \to \sum_t \begin{pmatrix} 0 \\ \bv\theta \end{pmatrix} \otimes \mathbf f_t = \begin{pmatrix} 0 \\ \bv\theta \end{pmatrix} \otimes \mathbf F'\mathbf 1_T ,
```

其中 $\bv\alpha$ 是全部 $TN$ 个 $\alpha_{i,t}$ 叠成的向量；前者为 $(L+1)K \times (L+1)K$，后者为 $(L+1)K \times 1$。于是

$$
\operatorname{plim}_{N\to\infty} \operatorname{vec}(\hat{\bv\Gamma}')
= \operatorname{vec}(\bv\Gamma') + \left[\mathbf I_{L+1} \otimes \mathbf F'\mathbf F\right]^{-1} \left[\begin{pmatrix} 0 \\ \bv\theta \end{pmatrix} \otimes \mathbf F'\mathbf 1_T\right]
= \operatorname{vec}(\bv\Gamma') + \begin{pmatrix} 0 \\ \bv\theta \end{pmatrix} \otimes \mathbf c ,
\qquad \underbrace{\mathbf c}_{K\times 1} = \underbrace{(\mathbf F'\mathbf F)^{-1}}_{K\times K} \underbrace{\mathbf F'\mathbf 1_T}_{K\times 1} ,
$$ (eq-ch0-bias)

第二个等号再次用到混合乘积规则，以及 $[\mathbf I_{L+1} \otimes \mathbf F'\mathbf F]^{-1} = \mathbf I_{L+1} \otimes (\mathbf F'\mathbf F)^{-1}$。按 $\operatorname{vec}(\bv\Gamma')$ 的排列顺序读，常数项一行没有偏误，第 $j$ 个特征一行的偏误是 $\theta_j \mathbf c'$；$\mathbf c$ 恰好是把常数 1 对因子做无截距回归的系数。当 $T \to \infty$ 时，$\mathbf F'\mathbf F/T \to \bv\Sigma_f + \bv\mu\bv\mu'$，$\mathbf F'\mathbf 1_T/T \to \bv\mu$，由 Sherman–Morrison 公式 $(\mathbf A + \mathbf u\mathbf v')^{-1} = \mathbf A^{-1} - \mathbf A^{-1}\mathbf u\mathbf v'\mathbf A^{-1}/(1 + \mathbf v'\mathbf A^{-1}\mathbf u)$（取 $\mathbf A = \bv\Sigma_f$，$\mathbf u = \mathbf v = \bv\mu$），

$$
\mathbf c \to (\bv\Sigma_f + \bv\mu\bv\mu')^{-1}\bv\mu = \bv\Sigma_f^{-1}\bv\mu - \frac{\bv\Sigma_f^{-1}\bv\mu\,(\bv\mu'\bv\Sigma_f^{-1}\bv\mu)}{1 + \bv\mu'\bv\Sigma_f^{-1}\bv\mu} = \frac{\bv\Sigma_f^{-1}\bv\mu}{1 + \bv\mu'\bv\Sigma_f^{-1}\bv\mu} = \frac{\mathbf b}{1 + \bv\mu'\bv\Sigma_f^{-1}\bv\mu} ,
$$ (eq-ch0-bias-limit)

分母中的 $\bv\mu'\bv\Sigma_f^{-1}\bv\mu$ 是标量。偏误的方向正是 SDF 系数 $\mathbf b$ 的方向。解决办法很简单：在回归中加入特征本身作为解释变量，其系数估计的就是 $\bv\theta$。IPCA 在载荷之外允许一个由特征驱动的截距，也是出于同样的考虑 [@kelly2019characteristics]。下面的代码在 $\bv\theta = (0.01, 0, 0, 0, 0)'$ 下比较两种回归。两者使用同一份数据，二者之差几乎不含异质性冲击带来的抽样噪声，可以直接与式 {eq}`eq-ch0-bias` 的 $\theta_1 \mathbf c'$ 比较。

(ch0-unpack-bias)=
**拆开讲式 {eq}`eq-ch0-bias`。** 第一个等号是把 $\mathbf y = \mathbf X\operatorname{vec}(\bv\Gamma') + \bv\alpha + \bv\varepsilon$ 代入最小二乘公式 $\operatorname{vec}(\hat{\bv\Gamma}') = (\mathbf X'\mathbf X)^{-1}\mathbf X'\mathbf y$（[拆解](#ch0-unpack-xtx)），再放入上面的两个极限（$\bv\varepsilon$ 一项在 $N\to\infty$ 时消失）。按 $\operatorname{vec}(\bv\Gamma')$ 的分段写出偏误向量：

```{math}
:enumerated: false
\underbrace{\begin{pmatrix} 0 \\ \bv\theta \end{pmatrix}}_{(L+1)\times 1}\otimes\underbrace{\mathbf c}_{K\times 1}
= \begin{pmatrix} 0\cdot\mathbf c \\ \theta_1\mathbf c \\ \vdots \\ \theta_L\mathbf c \end{pmatrix} , \qquad
\mathbf c = \begin{pmatrix} c_1 \\ c_2 \\ \vdots \\ c_K \end{pmatrix} ,
```

第 0 段（常数项一行）为 0，第 $j$ 段为 $\theta_j\mathbf c$，即 $\hat\Gamma_{j,k}$ 的偏误是 $\theta_j c_k$。**$\mathbf c$ 的含义**：$(\mathbf F'\mathbf F)^{-1}\mathbf F'\mathbf 1_T$ 是把 $T\times 1$ 的全 1 向量对 $T\times K$ 的 $\mathbf F$ 做不带截距的最小二乘回归所得的系数，$\mathbf c'\mathbf f_t$ 是用因子对常数 1 的最佳线性逼近，$c_k$ 的单位是“收益的倒数”。这给出偏误的直观：遗漏项 $\theta_j z_{j,i,t-1} = \theta_j z_{j,i,t-1}\cdot 1 \approx \theta_j z_{j,i,t-1}\,\mathbf c'\mathbf f_t = \sum_k \theta_j c_k\,(z_{j,i,t-1}f_{k,t})$，回归只能用交互项 $z_{j,i,t-1}f_{k,t}$ 去“模仿”它，于是这些交互项的系数多出 $\theta_j c_k$；常数项一行的交互项就是 $\mathbf f_t$ 本身，而 $\alpha_{i,t}$ 的横截面平均为 0，那一行没有偏误。

(ch0-unpack-c)=
**拆开讲式 {eq}`eq-ch0-bias-limit`。** $T\to\infty$ 时 $\mathbf F'\mathbf F/T = T^{-1}\sum_t\mathbf f_t\mathbf f_t' \to \E[\mathbf f_t\mathbf f_t'] = \bv\Sigma_f + \bv\mu\bv\mu'$，其 $(k, l)$ 元素为 $\E[f_{k,t}f_{l,t}] = \Sigma_{kl} + \mu_k\mu_l$（$\Sigma_{kl}$ 是 $\bv\Sigma_f$ 的 $(k, l)$ 元素）；$\mathbf F'\mathbf 1_T/T \to \bv\mu$。两个 $T$ 约掉，$\mathbf c \to (\bv\Sigma_f + \bv\mu\bv\mu')^{-1}\bv\mu$。Sherman–Morrison 公式中，$\mathbf A$ 是 $K\times K$ 的可逆矩阵，$\mathbf u$、$\mathbf v$ 是 $K\times 1$ 向量，$\mathbf u\mathbf v'$ 是 $K\times K$ 的外积（秩为 1），$\mathbf v'\mathbf A^{-1}\mathbf u$ 是标量：矩阵加上一个秩 1 的修正后，逆矩阵也只差一个秩 1 的修正。取 $\mathbf A = \bv\Sigma_f$、$\mathbf u = \mathbf v = \bv\mu$，并记标量 $s = \bv\mu'\bv\Sigma_f^{-1}\bv\mu$（即 0.5 节的 $\Var(m_t)$，[拆解](#ch0-unpack-varm)），式 {eq}`eq-ch0-bias-limit` 的各步是

```{math}
:enumerated: false
(\bv\Sigma_f + \bv\mu\bv\mu')^{-1}\bv\mu = \bv\Sigma_f^{-1}\bv\mu - \frac{\bv\Sigma_f^{-1}\bv\mu\,(\bv\mu'\bv\Sigma_f^{-1}\bv\mu)}{1 + \bv\mu'\bv\Sigma_f^{-1}\bv\mu} = \mathbf b - \frac{s}{1+s}\,\mathbf b = \frac{\mathbf b}{1+s} ,
```

第一步是 Sherman–Morrison 公式右乘 $\bv\mu$，分子中 $\mathbf v'\mathbf A^{-1}\bv\mu = \bv\mu'\bv\Sigma_f^{-1}\bv\mu = s$ 是一个数；第二步用到 $\bv\Sigma_f^{-1}\bv\mu = \mathbf b$，$K\times 1$ 的 $\mathbf b$ 乘标量 $s/(1+s)$。**特例**：$K = 1$ 时 $\mathbf c$ 就是数 $c = \sum_t f_t/\sum_t f_t^2$，极限为 $\mu/(\sigma^2 + \mu^2) = (\mu/\sigma^2)/(1 + \mu^2/\sigma^2)$；只取默认的第一个因子（$\mu_1 = 0.6\%$、$\sigma_1 = 5\%$）时 $c \approx 2.37$。默认三个因子下 $s \approx 0.0422$，$\mathbf b \approx (2.4, 3.33, 4.44)'$，于是 $\mathbf c$ 的极限约为 $(2.30, 3.20, 4.26)'$，$\theta_1 = 0.01$ 时 $z_1$ 一行的偏误约为 $(0.023, 0.032, 0.043)$：$\Gamma_{1,1}$、$\Gamma_{1,3}$ 的真值为 0，偏误让 $z_1$ 看起来也影响对第一、三个因子的载荷（下方代码核对）。**数据矩阵**：$\mathbf c$ 只依赖因子路径 $\mathbf F$，由所有股票共用，所以偏误对所有股票相同；有限 $T$ 下它取决于本样本实际出现的因子，而不是总体矩（下文的输出会看到这一点）。

```{code-cell} python
# 拆开讲 c 的极限：s = μ'Σ_f⁻¹μ，c → b / (1 + s)；只有第一个因子时 c → μ / (σ² + μ²)
mu_d, Sig_d = np.asarray(params.factor_mean), params.factor_cov            # μ: (K,)，Σ_f: (K, K)
b_d = np.linalg.solve(Sig_d, mu_d)                                         # b: (K,)
s_d = mu_d @ b_d                                                           # μ'Σ_f⁻¹μ：标量
c_lim = b_d / (1 + s_d)                                                    # (K,)
c_one = mu_d[0] / (Sig_d[0, 0] + mu_d[0] ** 2)                             # K = 1 时的极限
print(f"s = {s_d:.4f}；c 的极限 {c_lim.round(2)}；θ1·c {(0.01 * c_lim).round(3)}；只有第一个因子时 c = {c_one:.2f}")
assert np.allclose(c_lim, np.linalg.solve(Sig_d + np.outer(mu_d, mu_d), mu_d))
assert np.allclose(c_lim.round(2), [2.30, 3.20, 4.26]) and np.allclose((0.01 * c_lim).round(3), [0.023, 0.032, 0.043])
assert 0.0421 < s_d < 0.0422 and np.isclose(round(c_one, 2), 2.37)
```

下面的代码中，`c` 是本样本的 $\mathbf c$，`lhs` 与 `rhs` 是它的极限的两种算法。

```{code-cell} python
mis_params = MarketParams(alpha_coef=(0.01, 0.0, 0.0, 0.0, 0.0))
mis = simulate_market(mis_params, seed=0)
y = mis.panel["exret"].to_numpy()
X, Z = interaction_design(mis)                              # X: (TN, (L+1)K)，Z: (TN, L)
coef_short, _ = ols(X, y)                                  # 只有交互项
coef_long, se_long = ols(np.column_stack([X, Z]), y)       # 交互项 + 特征本身：(TN, (L+1)K + L)
G_short = coef_short.reshape(-1, K)                        # vec(Γ̂') -> Γ̂：((L+1)K,) -> (L+1, K)
G_long = coef_long[: X.shape[1]].reshape(-1, K)
theta_hat, theta_se = coef_long[X.shape[1]:], se_long[X.shape[1]:]

F_path = mis.factors.to_numpy()                            # F: (T, K)
c = np.linalg.solve(F_path.T @ F_path, F_path.sum(axis=0)) # 解 (F'F) c = F'1_T：(K, K) c = (K,)
print("z1 一行：只用交互项", G_short[1].round(4), "| 加入特征", G_long[1].round(4), "| 真值", mis.gamma[1])
print("常数项一行之差      ", (G_short[0] - G_long[0]).round(4))
print("z1 一行之差         ", (G_short[1] - G_long[1]).round(4))
print("偏误公式预测 θ1·c   ", (0.01 * c).round(4))
print("θ 的估计", theta_hat.round(4), "标准误", theta_se.round(4), "真值", mis.theta)

# Sherman–Morrison：T → ∞ 时 c 的极限
mu_v, Sig = np.asarray(mis_params.factor_mean), mis_params.factor_cov
lhs = np.linalg.solve(Sig + np.outer(mu_v, mu_v), mu_v)    # (Σ_f + μμ')⁻¹μ
rhs = mis.sdf_weights / (1 + mu_v @ mis.sdf_weights)       # b / (1 + μ'Σ_f⁻¹μ)
print("c 的极限", lhs.round(4), "=", rhs.round(4), "；本样本的 c", c.round(4))
assert np.allclose(lhs, rhs) and np.allclose(G_short[0] - G_long[0], 0, atol=0.003)   # 常数项一行几乎无偏误
rel_gap = (G_short[1] - G_long[1]) / (0.01 * c) - 1                # 实际之差相对公式预测的偏离
c_dev = c / lhs - 1                                        # 本样本 c 相对极限的偏离
print("之差相对公式预测的偏离", rel_gap.round(3), "；c 相对极限的偏离", c_dev.round(3))
assert np.all(np.abs(rel_gap) < 0.15)
f_bar_mis = F_path.mean(axis=0)
print("本样本的因子均值", f_bar_mis.round(4), "真值", mis_params.factor_mean)
assert np.all(c_dev > 0.15) and np.argmax(c_dev) == 1 and 0.3 < c_dev[1] < 0.5
assert f_bar_mis[1] > 1.2 * mu_v[1]
```

两种回归之差与式 {eq}`eq-ch0-bias` 的预测相符：$z_1$ 一行之差与 $\theta_1 \mathbf c'$ 的各分量相对差别在 15% 以内，常数项一行之差接近 0（均由 `assert` 保证）；有限 $N$ 下横截面平均与总体矩之间仍有小的差别，所以二者不完全相等。还要注意，本例中 $\mathbf c$ 与它的极限 {eq}`eq-ch0-bias-limit` 有明显差别：三个分量都比极限大，其中第二个分量偏离最大，约大四成（相对偏离见上方输出，由 `assert` 守护）。原因是 240 个月的样本中，因子样本均值本身就有可观的抽样误差——本样本中第二个因子的样本均值比真值高出两成以上——而偏误取决于样本中实际出现的因子路径，而不是总体矩。

四个常见误区见下一页[常见误区](#sec-ch0-pitfalls)。
