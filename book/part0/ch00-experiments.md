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

# A 股视角、模拟实验与常见误区

**本页准备。** 本书每个页面的代码在独立的内核中运行，前面页面定义的变量在这里不可用。下面的单元导入本页用到的模块，并按与 0.4 节、0.6 节完全相同的参数和种子，重建本页用到的两个模拟市场及由它们算出的数组。

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

# 0.6 节“模拟验证”的长样本（θ1 = 0.01，种子 2026），误区一用到
long_params = MarketParams(n_stocks=20, n_periods=20_000, alpha_coef=(0.01, 0.0, 0.0, 0.0, 0.0))
long_mkt = simulate_market(long_params, seed=2026)
MR = long_mkt.sdf().to_numpy()[:, None] * long_mkt.panel["exret"].unstack("stock").to_numpy()   # (T, N)：m_t r_{i,t}
Z1 = long_mkt.panel["z1"].unstack("stock").to_numpy()                                         # (T, N)
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

**写成标准的回归形式。** 要把 $\bv\Gamma$ 的元素排成一个系数向量，需要用到 vec 算子与 Kronecker 积。$\operatorname{vec}(\mathbf A)$ 把矩阵 $\mathbf A$ 的各列从左到右依次叠成一个长的列向量（列优先）；Kronecker 积 $\mathbf a \otimes \mathbf c$ 把 $\mathbf a$ 的每个元素 $a_j$ 换成块 $a_j \mathbf c$。二者由恒等式

$$
\operatorname{vec}(\mathbf A \mathbf B \mathbf C) = (\mathbf C' \otimes \mathbf A)\operatorname{vec}(\mathbf B)
$$ (eq-ch0-vec)

联系起来（$\mathbf A$、$\mathbf B$、$\mathbf C$ 为任意可以相乘的矩阵）。标量 $\tilde{\mathbf z}_{i,t-1}'\bv\Gamma\mathbf f_t$ 等于它的转置 $\mathbf f_t'\bv\Gamma'\tilde{\mathbf z}_{i,t-1}$，取 $\mathbf A = \mathbf f_t'$（$1\times K$）、$\mathbf B = \bv\Gamma'$（$K\times(L+1)$）、$\mathbf C = \tilde{\mathbf z}_{i,t-1}$（$(L+1)\times 1$），一个标量的 vec 就是它自己，于是

$$
r_{i,t} = \underbrace{(\tilde{\mathbf z}_{i,t-1} \otimes \mathbf f_t)'}_{1\times(L+1)K}\; \underbrace{\operatorname{vec}(\bv\Gamma')}_{(L+1)K\times 1} + \varepsilon_{i,t} .
$$ (eq-ch0-kron)

这里用到 $(\tilde{\mathbf z}_{i,t-1}' \otimes \mathbf f_t') = (\tilde{\mathbf z}_{i,t-1} \otimes \mathbf f_t)'$。系数向量 $\operatorname{vec}(\bv\Gamma')$ 按列叠放 $\bv\Gamma'$，也就是按行叠放 $\bv\Gamma$：先是第 0 行 $(\Gamma_{0,1}, \dots, \Gamma_{0,K})$，再是第 1 行，依此类推，$j$ 在外层、$k$ 在内层；设计矩阵的第 $(i,t)$ 行 $(\tilde{\mathbf z}_{i,t-1} \otimes \mathbf f_t)'$ 的元素也按同样的顺序排列为 $\tilde z_{j,i,t-1} f_{k,t}$。把全部 $TN$ 个观测按行叠成 $\mathbf X \in \Real^{TN\times(L+1)K}$ 与 $\mathbf y \in \Real^{TN}$，因子可观测时，对 $\mathbf y$ 关于 $\mathbf X$ 做最小二乘，就能估计 $\operatorname{vec}(\bv\Gamma')$。

**与代码的对应。** 下面的 `interaction_design` 用广播构造 $\mathbf X$：`zt[:, :, None] * ff[:, None, :]` 的形状为 `(TN, L+1, K)`，其 `[n, j, k]` 元素是 $\tilde z_j f_k$；NumPy 的 `reshape` 默认按行优先（C 顺序）展开，最后一维 `k` 变化最快，所以展开后每一行正是 $(\tilde{\mathbf z} \otimes \mathbf f)'$。同理，`gamma.ravel()`（行优先）就是 $\operatorname{vec}(\bv\Gamma')$，而列优先的 `gamma.ravel(order="F")` 才是 $\operatorname{vec}(\bv\Gamma)$；系数向量用 `coef.reshape(L+1, K)` 还原为 $\hat{\bv\Gamma}$。数学上的 vec 是列优先，NumPy 的默认展开是行优先，两者差一个转置，这是最容易出错的地方，下面的代码单元用 `np.kron` 与 `assert` 核对这一对应。

由于 $\varepsilon_{i,t}$ 与 $(\mathbf z_{i,t-1}, \mathbf f_t)$ 独立，解释变量严格外生，最小二乘估计量无偏，其条件协方差矩阵为 $\sigma_\varepsilon^2 (\mathbf X'\mathbf X)^{-1}$（$(L+1)K \times (L+1)K$）。函数 `ols` 用 `lstsq` 求系数、用 `solve` 求 $(\mathbf X'\mathbf X)^{-1}$，而不调用 `inv`。这两个函数也收在 `apsim.stats` 中，代码单元用 `assert` 核对二者与这里的定义结果相同，后面的页面直接调用 `apsim.stats`。下面用默认参数（$N = 500$、$T = 240$）检验这一点。

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

(sec-ch0-pitfalls)=
## 常见误区

**误区一：把同期特征当作期初特征。** 用 $\mathbf z_{i,t}$ 而不是 $\mathbf z_{i,t-1}$ 去解释 $r_{i,t}$，在真实数据中可能造成严重的前视偏差：许多特征（市值、动量、换手率）与当期收益机械相关，例如第 $t$ 期末的市值本身就包含了第 $t$ 期的收益。在本章的模拟器中，特征更新的冲击 $\mathbf e_{i,t}$ 与收益无关，所以错用同期特征不会产生虚假的预测力，只会削弱真实的关系：由式 {eq}`eq-ch0-char` 与 $\mathbf e_{i,t}$ 的独立性，

```{math}
:enumerated: false
\underbrace{\E[m_t r_{i,t} \mathbf z_{i,t}]}_{L\times 1} = \phi\,\E[m_t r_{i,t} \mathbf z_{i,t-1}] + \sqrt{1-\phi^2}\,\E[m_t r_{i,t} \mathbf e_{i,t}] = \phi\,\bv\theta + \mathbf 0 .
```

这个例子说明了模拟的局限：模拟器不会自动暴露所有错误，只能暴露我们放进模型的那些机制。下面用 $\phi = 0.9$ 的长样本验证。两个样本矩都带有抽样误差，而且误差高度相关；所以除了把它们分别与理论值比较，还检验二者之差是否满足推导给出的关系：用 $\overline{m r z_1}$ 表示 $m_t r_{i,t} z_{1,i,t-1}$ 对全部 $(i,t)$ 的样本均值、$\overline{m r z_1^{+}}$ 表示把 $z_{1,i,t-1}$ 换成同期 $z_{1,i,t}$ 后的样本均值，则 $\overline{m r z_1^{+}} - \phi\, \overline{m r z_1}$ 只剩下 $\sqrt{1-\phi^2}\, \overline{m r e_1}$（$e_{1,i,t}$ 为 $\mathbf e_{i,t}$ 的第一个分量），理论值为 0。

```{code-cell} python
Z1_now = long_mkt.panel["z1"].unstack("stock").shift(-1).to_numpy()   # (T, N)，第 t 行放 z_{1,i,t}
valid = ~np.isnan(Z1_now[:, 0])                                        # 最后一期没有 z_{i,T}
now = (MR[valid] * Z1_now[valid]).mean()
lag = (MR[valid] * Z1[valid]).mean()
phi = long_params.char_persistence
print(f"用期初特征 z_(t-1)：{lag:.5f}（理论值 θ1 = {long_params.alpha_coef[0]}）")
print(f"用同期特征 z_t    ：{now:.5f}（理论值 φθ1 = {phi * 0.01:.4f}）")
resid_term = (MR[valid] * (Z1_now[valid] - phi * Z1[valid])).mean(axis=1)   # 每期横截面平均：(T-1,)
se_resid = resid_term.std() / np.sqrt(resid_term.size)
print(f"now − φ·lag = {now - phi * lag:.5f}，标准误约 {se_resid:.5f}（理论值 0）")
assert abs(now - phi * lag) < 3 * se_resid
```

二者之差与 0 的距离在 3 倍标准误以内（由 `assert` 保证）。这里的标准误同样先对每期求横截面平均再计算。

**误区二：把样本均值当作总体值。** 默认参数下第一个因子的月均值 $\mu_1 = 0.6\%$、波动率 $5\%$，240 个月样本均值的标准误为 $0.05/\sqrt{240} \approx 0.32\%$，样本均值完全可能偏离真值一半以上。更隐蔽的错误是用样本矩构造 SDF，再在同一样本中“验证”它。若 $\hat{\mathbf b} = \hat{\bv\Sigma}_f^{-1}\bar{\mathbf f}$（$\hat{\bv\Sigma}_f$ 为除以 $T$ 的样本协方差矩阵，$\bar{\mathbf f}$ 为样本均值向量），$\hat m_t = 1 - \hat{\mathbf b}'(\mathbf f_t - \bar{\mathbf f})$，则把[定理 %s](#thm-ch0-sdf) 第 2 条证明中的总体矩逐一换成样本矩，可得样本内 $T^{-1}\sum_t \hat m_t \mathbf f_t = \bar{\mathbf f} - \hat{\bv\Sigma}_f \hat{\mathbf b} = \mathbf 0$ **恒成立**，无论模型对错。这样的“检验”什么也没有检验。

```{code-cell} python
F240 = mkt.factors.to_numpy()                              # (T, K)
f_bar = F240.mean(axis=0)                                  # (K,)
Sigma_hat = np.cov(F240, rowvar=False, ddof=0)             # (K, K)
b_hat = np.linalg.solve(Sigma_hat, f_bar)                  # 解 Σ̂_f b̂ = f̄
m_hat = 1 - (F240 - f_bar) @ b_hat                         # (T, K) @ (K,) -> (T,)
in_sample = (m_hat[:, None] * F240).mean(axis=0)
true_pricing = (mkt.sdf().to_numpy()[:, None] * F240).mean(axis=0)
print("因子样本均值", f_bar.round(4), "真值", params.factor_mean, "标准误", (np.asarray(params.factor_vol) / np.sqrt(len(F240))).round(4))
print("b 的样本估计", b_hat.round(2), "真值", mkt.sdf_weights.round(2))
print("样本内 mean(m̂ f) =", in_sample)
print("真实 m 的 mean(m f) =", true_pricing.round(5))
assert np.max(np.abs(in_sample)) < 1e-12 and np.max(np.abs(true_pricing)) > 1e-4
```

样本内的 $T^{-1}\sum_t \hat m_t \mathbf f_t$ 在机器精度内为 $\mathbf 0$，而真实 SDF 在同一样本中的样本定价误差并不为 0——后者才反映了抽样误差的真实大小。真正的检验需要样本外数据，或者把估计误差计入检验统计量，这是第二部分的主题。

**误区三：忽视随机种子。** 没有固定种子的随机数每次运行都不同，书中的数字因此无法复现。为了避免这种错误，`apsim.simulate_market` 要求必须以关键字传入 `seed`，省略它或写成 `seed=None` 都会报错；确实需要不可复现的随机数时，只能显式传入一个未设种子的 `np.random.default_rng()`。反过来的错误同样常见：在蒙特卡洛循环中每次都用同一个种子，得到的“独立重复”其实完全相同，估计量的抽样方差被算成 0。

```{code-cell} python
small = MarketParams(n_stocks=50, n_periods=60)
try:
    simulate_market(small)
except TypeError as err:
    print("省略 seed：TypeError：", err)

first = simulate_market(small, seed=np.random.default_rng()).panel["exret"].iloc[0]
second = simulate_market(small, seed=np.random.default_rng()).panel["exret"].iloc[0]
print(f"未设种子的 Generator 两次运行的第一个收益：{first:.4f} 与 {second:.4f}")

same = [simulate_market(small, seed=0).factors["f1"].mean() for _ in range(5)]
diff = [simulate_market(small, seed=100 + r).factors["f1"].mean() for r in range(5)]
print("同一种子重复 5 次，f1 样本均值的标准差：", np.std(same))
print("不同种子重复 5 次，f1 样本均值的标准差：", round(float(np.std(diff)), 4))
assert np.std(same) == 0 and np.std(diff) > 0
```

（第二行的两个数字每次重新执行都会变化，这正是问题所在。）本书的约定是：单次演示用固定的整数种子；蒙特卡洛重复用“基础种子 + 重复编号”，保证各次重复互不相同，而整体可以复现。另一种同样可复现的做法，是新建一个带种子的 `rng = np.random.default_rng(2026)`（2026 为基础种子），在循环中反复传入 `seed=rng`：`simulate_market` 每次都从它派生新的子流，各次重复互不相同。

**误区四：把可预测性等同于错误定价。** 即使 $\bv\theta = \mathbf 0$，特征也能预测收益。由式 {eq}`eq-ch0-ret` 与独立性，$\E[r_{i,t} \mid \mathbf z_{i,t-1}] = \alpha_{i,t} + \bv\beta_{i,t}'\bv\mu$，而 $\bv\beta_{i,t}$ 随特征变化。默认参数下 $\E[r_{i,t} \mid \mathbf z_{i,t-1}] = \mu_1 + 0.5\mu_2\, z_{1,i,t-1} + 0.5\mu_3\, z_{2,i,t-1}$：特征 $z_1$、$z_2$ 能预测收益，完全是因为它们决定了风险暴露，这正是“特征即协方差”的含义 [@kelly2019characteristics]。判断一个能预测收益的特征是风险补偿还是错误定价，要看它的预测力能否被载荷与因子风险溢价解释，而不能只看预测力本身（第 8、23 章）。下面用 $\bv\theta = \mathbf 0$ 的长样本做 $r_{i,t}$ 对 $\tilde{\mathbf z}_{i,t-1}$ 的混合回归。由于 $\E[r_{i,t} \mid \mathbf z_{i,t-1}] = \bv\beta_{i,t}'\bv\mu = \tilde{\mathbf z}_{i,t-1}'\bv\Gamma\bv\mu$，系数的理论值为 $\bv\Gamma\bv\mu$（$(L+1)\times K$ 乘 $K\times 1$，为 $L+1$ 维向量）。

逐期回归的代码用 `reshape(n_t, n_i, -1)` 把 $TN$ 行的设计矩阵还原为“期 × 股票 × 变量”的三维数组。面板按 `(period, stock)` 排序、期在外层，所以前 $N$ 行是第 1 期，接下来的 $N$ 行是第 2 期，行优先的 `reshape` 恰好按期切开。

```{code-cell} python
pred = simulate_market(MarketParams(n_stocks=100, n_periods=5_000), seed=11)
Xp = np.column_stack([np.ones(len(pred.panel)), pred.panel[[f"z{j}" for j in range(1, 6)]].to_numpy()])  # (TN, L+1)
coef_p, se_p = ols(Xp, pred.panel["exret"].to_numpy())
truth_p = pred.gamma @ np.asarray(pred.params.factor_mean)               # Γμ：(L+1, K) @ (K,) -> (L+1,)

# 逐期横截面回归，用各期系数的时间序列标准差计算标准误（Fama–MacBeth 的做法，第 10 章详述）
n_t, n_i = pred.params.n_periods, pred.params.n_stocks
Xp3 = Xp.reshape(n_t, n_i, -1)                                           # (TN, L+1) -> (T, N, L+1)
yp3 = pred.panel["exret"].to_numpy().reshape(n_t, n_i)                   # (TN,) -> (T, N)
per_period = np.array([np.linalg.lstsq(Xp3[t], yp3[t], rcond=None)[0] for t in range(n_t)])   # (T, L+1)
se_fm = per_period.std(axis=0, ddof=1) / np.sqrt(n_t)
fm_p = stats.fama_macbeth(Xp3, yp3)                                      # 同一程序的 apsim.stats 版本：比较 coefs、mean、se
assert all(np.allclose(a, b) for a, b in [(fm_p.coefs, per_period), (fm_p.mean, per_period.mean(axis=0)), (fm_p.se, se_fm)])
print(pd.DataFrame({"理论值 Γμ": truth_p, "混合回归估计": coef_p, "同方差标准误": se_p, "逐期回归标准误": se_fm},
                   index=["常数", "z1", "z2", "z3", "z4", "z5"]).round(5))
ratio = se_fm / se_p
print("逐期回归标准误 / 同方差标准误：", ratio.round(2))
assert np.all(np.abs(coef_p - truth_p) < 2 * se_fm)                    # 偏离在两倍逐期回归标准误以内
assert np.all(coef_p[1:3] / se_fm[1:3] > 3)                            # z1、z2 的系数显著为正
assert np.all(ratio[:3] > 1.3) and np.all(np.abs(ratio[3:] - 1) < 0.2)
```

系数的估计值与理论值 $\bv\Gamma\bv\mu$ 相符（偏离都在两倍逐期回归标准误以内）：$z_1$、$z_2$ 的系数显著为正（均由 `assert` 保证），而 $\bv\theta = \mathbf 0$，市场中没有任何错误定价。表中还有一个第二部分会反复出现的问题：“同方差标准误”低估了真实的不确定性。同一期的所有股票共同受到 $\mathbf f_t$ 的冲击，残差在横截面上相关，混合回归把 $NT$ 个观测当作相互独立，高估了信息量。按期做横截面回归、再看各期系数的时间序列波动，得到的标准误更接近真实的抽样误差，这就是 Fama–MacBeth 标准误（[导读：几种回归](#sec-ch0-regs)，第 10、12 章）：常数项与 $z_1$、$z_2$ 的逐期回归标准误明显大于同方差标准误（比值见上方输出），因为它们的系数随 $\mathbf f_t$ 逐期波动；$z_3$～$z_5$ 与因子无关，两种标准误相近。估计值与理论值的偏离，应当按这种更大的标准误来判断。

