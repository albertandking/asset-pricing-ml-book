# 记号表

本表在第 0 章确定，全书遵守。后续章节引入新记号时，会在该章首次出现处定义，并补充到本表。

## 向量与矩阵约定

### 字体

| 对象 | 字体 | 例子 |
|---|---|---|
| 标量 | 斜体，不加粗 | $r_{i,t}$、$\alpha_{i,t}$、$m_t$、$\phi$、$\sigma_\varepsilon$、$T$、$N$、$K$、$L$ |
| 向量（拉丁字母） | 小写粗体（`\mathbf`） | $\mathbf f_t$、$\mathbf z_{i,t-1}$、$\tilde{\mathbf z}_{i,t-1}$、$\mathbf b$、$\mathbf r_t$、$\mathbf 1_N$ |
| 向量（希腊字母） | 小写粗体（`\boldsymbol`，源文件中写作宏，如 `\bv\beta`） | $\bv\mu$、$\bv\beta_{i,t}$、$\bv\theta$、$\bv\sigma$、$\bv\alpha_t$、$\bv\varepsilon_t$、$\bv\lambda$ |
| 矩阵 | 大写粗体 | $\bv\Gamma$、$\bv\Sigma_f$、$\mathbf Z_{t-1}$、$\tilde{\mathbf Z}_{t-1}$、$\mathbf B_t$、$\mathbf I_K$ |

支付 $x_{t+1}$、价格 $P_t$ 与随机贴现因子 $m_{t+1}$ 是标量；需要多个资产的支付或价格组成的向量时，写作 $\mathbf x_{t+1}$、$\mathbf P_t$。

这样，一个元素和它所在的向量或矩阵在字形上直接对应：$r_{i,t}$ 是 $\mathbf r_t$ 的第 $i$ 个元素，$\bv\beta_{i,t}'$ 是 $\mathbf B_t$ 的第 $i$ 行，$\Gamma_{j,k}$ 是 $\bv\Gamma$ 的 $(j,k)$ 元素。许多文献（例如 Cochrane 的《Asset Pricing》）并不给向量加粗，读原始论文时要从上下文判断一个符号是标量还是向量；在代码中，可以直接查看数组的 `.shape`。

### 方向与维度

- 公式中的向量一律是**列向量**。行向量用转置表示，转置记号为撇号 $'$，例如 $\mathbf b'$、$\mathbf z_{i,t-1}'$。两个向量的内积写作 $\mathbf a'\mathbf b$。
- 每个向量或矩阵在首次出现时给出维度，例如 $\mathbf f_t \in \Real^{K}$、$\bv\Gamma \in \Real^{(L+1)\times K}$。矩阵公式之后会标出各项的维度，便于读者核对乘法是否相容。
- $\mathbf 1_N$ 是 $N$ 维全 1 向量，$\mathbf 0$ 是零向量或零矩阵（维度由上下文决定），$\mathbf I_K$ 是 $K$ 阶单位矩阵。

### 横截面堆叠

单只股票的模型写成列向量形式：$\bv\beta_{i,t} = \bv\Gamma'\tilde{\mathbf z}_{i,t-1}$，其中 $\tilde{\mathbf z}_{i,t-1} = (1, \mathbf z_{i,t-1}')' \in \Real^{L+1}$。同一期的 $N$ 只股票按**行**堆叠：

| 堆叠对象 | 定义 | 维度 |
|---|---|---|
| $\mathbf Z_{t-1}$ | 第 $i$ 行为 $\mathbf z_{i,t-1}'$ | $N \times L$ |
| $\tilde{\mathbf Z}_{t-1}$ | $[\mathbf 1_N, \mathbf Z_{t-1}]$，第 $i$ 行为 $\tilde{\mathbf z}_{i,t-1}'$ | $N \times (L+1)$ |
| $\mathbf B_t$ | $\tilde{\mathbf Z}_{t-1}\bv\Gamma$，第 $i$ 行为 $\bv\beta_{i,t}'$ | $N \times K$ |
| $\bv\alpha_t$ | $\mathbf Z_{t-1}\bv\theta$，第 $i$ 个元素为 $\alpha_{i,t}$ | $N \times 1$ |
| $\mathbf r_t$、$\bv\varepsilon_t$ | 第 $i$ 个元素为 $r_{i,t}$、$\varepsilon_{i,t}$ | $N \times 1$ |

于是全部股票的收益可以一次写出：

```{math}
:enumerated: false
\underbrace{\mathbf r_t}_{N\times 1} = \underbrace{\bv\alpha_t}_{N\times 1} + \underbrace{\mathbf B_t}_{N\times K}\,\underbrace{\mathbf f_t}_{K\times 1} + \underbrace{\bv\varepsilon_t}_{N\times 1} .
```

这与逐只股票的写法是同一个模型。$\mathbf B_t$ 的第 $i$ 行是

```{math}
:enumerated: false
\tilde{\mathbf z}_{i,t-1}'\,\bv\Gamma = \left(\bv\Gamma'\tilde{\mathbf z}_{i,t-1}\right)' = \bv\beta_{i,t}' ,
```

所以 $\mathbf B_t\mathbf f_t$ 的第 $i$ 个元素就是 $\bv\beta_{i,t}'\mathbf f_t$。

### 公式与代码的对应

- NumPy 的一维数组没有行、列之分。本书中一维数组一律代表公式中的列向量。两个一维数组的 `a @ b` 是内积 $\mathbf a'\mathbf b$（标量），矩阵与一维数组的 `M @ v` 是 $\mathbf M\mathbf v$。
- 数据数组遵循 pandas 与 NumPy 的“观测在行”习惯：因子数组的形状为 `(T, K)`，特征数组为 `(T, N, L)`。这类数组的一行，是公式中对应列向量的**转置**。
- 因此，形状为 `(N, L+1)` 的数组 `x` 与 `gamma` 相乘，`x @ gamma` 计算的是 $\tilde{\mathbf Z}_{t-1}\bv\Gamma = \mathbf B_t$；逐行看，第 $i$ 行是 $\tilde{\mathbf z}_{i,t-1}'\bv\Gamma = \bv\beta_{i,t}'$。

`apsim/market.py` 中生成收益的三行代码可以逐一对应到上面的公式（数组的第一维是期，`t` 从 0 起，`x[t-1]` 对应第 $t$ 期）：

| 代码 | 运算对象的形状 | 结果形状 | 对应公式 |
|---|---|---|---|
| `beta = x @ gamma` | `(T, N, L+1) @ (L+1, K)` | `(T, N, K)` | `@` 对第一维逐期做矩阵乘法：`beta[t-1] = x[t-1] @ gamma` 即 $\mathbf B_t = \tilde{\mathbf Z}_{t-1}\bv\Gamma$ |
| `alpha = z_lag @ theta` | `(T, N, L) @ (L,)` | `(T, N)` | `theta` 作列向量：`alpha[t-1] = z_lag[t-1] @ theta` 即 $\bv\alpha_t = \mathbf Z_{t-1}\bv\theta$，元素为 $\bv\theta'\mathbf z_{i,t-1}$ |
| `np.einsum("tnk,tk->tn", beta, f)` | `(T, N, K)` 与 `(T, K)` | `(T, N)` | 对每个 $(t, i)$ 求 $\sum_k \beta_{k,i,t} f_{k,t} = \bv\beta_{i,t}'\mathbf f_t$，即逐期计算 $\mathbf B_t\mathbf f_t$ |

`einsum` 的下标字符串按“输入下标 `->` 输出下标”读：出现在输入、却不出现在输出中的下标（这里是 `k`）被求和，其余下标保留。

| 公式对象 | 代码变量 | 形状 |
|---|---|---|
| $\mathbf f_t'$（第 $t$ 行） | `f`（`market.py`）；`mkt.factors.to_numpy()` | `(T, K)` |
| $\mathbf z_{i,t-1}'$（第 $t$ 期、第 $i$ 行） | `z_lag`；`panel[[f"z{j}" for j in range(1, L + 1)]]` | `(T, N, L)`；`(T·N, L)` |
| $\tilde{\mathbf Z}_{t-1}$ | `x[t-1]` | `(N, L+1)` |
| $\bv\Gamma$ | `gamma`；`mkt.gamma` | `(L+1, K)` |
| $\bv\theta$ | `theta`；`mkt.theta` | `(L,)` |
| $\mathbf B_t$ | `beta[t-1]` | `(N, K)` |
| $\bv\alpha_t$、$\mathbf r_t$ | `alpha[t-1]`、`exret[t-1]` | `(N,)` |
| $\bv\mu$ | `np.asarray(params.factor_mean)` | `(K,)` |
| $\bv\Sigma_f$ | `params.factor_cov` | `(K, K)` |
| $\mathbf b$ | `mkt.sdf_weights` | `(K,)` |

**解方程而不求逆。** 公式中的 $\bv\Sigma_f^{-1}\bv\mu$、$(\mathbf X'\mathbf X)^{-1}\mathbf X'\mathbf y$ 在代码中用 `np.linalg.solve(Sigma_f, mu)` 与 `np.linalg.lstsq(X, y)` 计算，而不是先用 `np.linalg.inv` 求出逆矩阵再相乘：直接解线性方程组的计算量更小，在矩阵接近奇异时舍入误差也小得多。只有需要逆矩阵本身时（例如最小二乘系数的协方差矩阵 $\hat\sigma_\varepsilon^2(\mathbf X'\mathbf X)^{-1}$），才通过 `solve(A, I)` 得到它。更多的矩阵代数与矩阵微积分工具见附录 A。

## 下标与维度

| 记号 | 含义 | 说明 |
|---|---|---|
| $t$ | 时间下标（期） | 本书模拟数据为月度，$t = 1, \dots, T$ |
| $i$ | 资产（股票）下标 | 理论推导中 $i = 1, \dots, N$；`apsim` 中股票编号从 0 开始，为 $0, \dots, N-1$ |
| $T$ | 期数 | `MarketParams.n_periods` |
| $N$ | 股票数 | `MarketParams.n_stocks` |
| $K$ | 因子数，下标 $k = 1, \dots, K$ | `MarketParams.n_factors`，等于 `factor_mean` 的长度 |
| $L$ | 特征数，下标 $j = 1, \dots, L$ | `MarketParams.n_chars` |

## 时点约定

- 下标 $t$ 表示“在第 $t$ 期末已知”。$\E_t[\cdot]$ 是以第 $t$ 期末信息为条件的期望。
- 收益带下标 $t$，表示从第 $t-1$ 期末持有到第 $t$ 期末实现的收益。
- 横截面面板中，第 $t$ 期的一行同时记录“期初已知的特征 $\mathbf z_{i,t-1}$”与“本期实现的超额收益 $r_{i,t}$”：**特征比收益早一期**。
- 理论部分沿用 $t \to t+1$ 的写法（如 $P_t = \E_t[m_{t+1} x_{t+1}]$）；面板与模拟部分沿用 $t-1 \to t$ 的写法（如 $\E_{t-1}[m_t r_{i,t}]$）。两种写法只差一个时间平移，含义相同。
- 在 $t-1 \to t$ 的写法下，适用于第 $t$ 期的无风险利率在第 $t-1$ 期末已知，记作 $R^f_{t-1}$，于是 $R^e_{i,t} = R_{i,t} - R^f_{t-1}$。

## 收益、价格与随机贴现因子

| 记号 | 含义 | 说明 |
|---|---|---|
| $P_t$ | 资产在第 $t$ 期末的价格（水平） | 大写；小写 $p_t$ 留给对数价格 |
| $x_{t+1}$ | 第 $t+1$ 期的支付（payoff，下一期价格加红利） | 只表示支付 |
| $R_{t+1}$ | 总收益（gross return） | $R_{t+1} = x_{t+1} / P_t$ |
| $R^f_t$ | 无风险总收益 | 第 $t$ 期末已知，适用于从 $t$ 到 $t+1$ |
| $R^e_{t+1}$ | 超额收益 | $R^e_{t+1} = R_{t+1} - R^f_t$ |
| $r_{i,t}$ | 股票 $i$ 在第 $t$ 期的超额收益（面板记号，水平值） | 即 $R^e_{i,t}$ 的面板写法；`panel["exret"]`；是下文对数约定的例外 |
| $p_t$、$d_t$、$c_t$、$r_t$ | 对数价格、对数红利、对数消费、对数收益 | 第 1、6、7 章的惯例，如 $p_t = \log P_t$ |
| $m_{t+1}$ | 随机贴现因子（stochastic discount factor，SDF） | 面板部分写作 $m_t$；`SimulatedMarket.sdf()` |
| $P_t = \E_t[m_{t+1} x_{t+1}]$ | 基本定价方程 | 对总收益为 $1 = \E_t[m_{t+1} R_{t+1}]$，对超额收益为 $0 = \E_t[m_{t+1} R^e_{t+1}]$ |

## 因子、载荷与定价误差

| 记号 | 含义 | 说明 |
|---|---|---|
| $\mathbf f_t \in \Real^K$ | 因子超额收益向量，分量 $f_{k,t}$ | `factors` 的列 `f1`……`fK` |
| $\bv\mu \in \Real^K$ | 因子均值 $\E[\mathbf f_t]$，分量 $\mu_k$ | `factor_mean` |
| $\bv\sigma \in \Real^K$ | 因子波动率向量，分量 $\sigma_k$ | `factor_vol` |
| $\bv\Sigma_f \in \Real^{K\times K}$ | 因子协方差矩阵 $\Var(\mathbf f_t)$ | 模拟器中因子相互独立，$\bv\Sigma_f = \operatorname{diag}(\sigma_1^2, \dots, \sigma_K^2)$；`factor_cov` |
| $\mathbf b \in \Real^K$ | SDF 系数，$m_t = 1 - \mathbf b'(\mathbf f_t - \bv\mu)$ | $\mathbf b = \bv\Sigma_f^{-1}\bv\mu$；模拟器中化为 $b_k = \mu_k / \sigma_k^2$；`sdf_weights` |
| $\bv\beta_{i,t} \in \Real^K$ | 股票 $i$ 在第 $t$ 期的因子载荷，分量 $\beta_{k,i,t}$ | 第 $t-1$ 期末已知；`panel["beta1"]`…… |
| $\bv\lambda \in \Real^K$ | 因子风险溢价 | 因子本身是超额收益时 $\bv\lambda = \bv\mu$ |
| $\alpha_{i,t}$ | 定价误差（错误定价） | $\alpha_{i,t} = \E_{t-1}[m_t r_{i,t}]$ 在本书模型中等于 $\bv\theta' \mathbf z_{i,t-1}$；`panel["alpha"]` |
| $\varepsilon_{i,t}$ | 异质性冲击 | $\varepsilon_{i,t} \sim N(0, \sigma_\varepsilon^2)$ |

## 特征与载荷矩阵

| 记号 | 含义 | 说明 |
|---|---|---|
| $\mathbf z_{i,t} \in \Real^L$ | 股票 $i$ 在第 $t$ 期末的特征向量，分量 $z_{j,i,t}$ | 面板第 $t$ 期的列 `zj` 存放 $z_{j,i,t-1}$ |
| $\tilde{\mathbf z}_{i,t-1} \in \Real^{L+1}$ | 带常数项的特征向量 $(1, \mathbf z_{i,t-1}')'$ | 分量记为 $\tilde z_{j,i,t-1}$，$j = 0, \dots, L$，$\tilde z_{0,i,t-1} = 1$ |
| $\bv\Gamma \in \Real^{(L+1)\times K}$ | 载荷矩阵，$\bv\beta_{i,t} = \bv\Gamma' \tilde{\mathbf z}_{i,t-1}$ | 第 0 行对应常数项，第 $j$ 行对应特征 $j$；`gamma` |
| $\bv\theta \in \Real^L$ | 定价误差系数，$\alpha_{i,t} = \bv\theta' \mathbf z_{i,t-1}$ | `alpha_coef`，默认全为 0 |
| $\phi$ | 特征的一阶自相关（持续性） | `char_persistence` |
| $\sigma_\varepsilon$ | 异质性波动率 | `idio_vol` |

## 运算符

| 记号 | 含义 |
|---|---|
| $\E[\cdot]$ | 无条件期望 |
| $\E_t[\cdot]$ | 以第 $t$ 期末信息为条件的期望 |
| $\E[\cdot \mid \mathbf z]$ | 以随机向量 $\mathbf z$ 为条件的期望 |
| $\Var(\cdot)$、$\Cov(\cdot, \cdot)$ | 方差（或协方差矩阵）与协方差 |
| $\operatorname{diag}(a_1, \dots, a_K)$ | 以 $a_1, \dots, a_K$ 为对角元的对角矩阵 |
| $\operatorname{vec}(\mathbf A)$ | 把矩阵 $\mathbf A$ 按列依次叠成一个列向量（列优先） |
| $\otimes$ | Kronecker 积 |
| $\bar{y}$ | 样本均值，例如 $\bar{y} = T^{-1}\sum_{t=1}^{T} y_t$；与总体期望 $\E[y]$ 严格区分 |
| $\Real$ | 实数集，$\Real^K$ 为 $K$ 维实向量空间 |

## 已预留与已知冲突

以下记号已为后续章节预留，或在不同场合有不同含义。第 0 章的记号已按这些约定选取。

| 记号 | 约定 | 说明 |
|---|---|---|
| $\delta$ | 主观贴现因子（Campbell 等文献的惯例） | 定价误差系数因此记为 $\bv\theta$，不用 $\delta$ |
| $\rho$ | Campbell–Shiller 对数线性化常数；相关系数 | 特征持续性因此记为 $\phi$ |
| $s_t$ | Campbell–Cochrane 模型中的（对数）盈余消费比 | 异质性波动率因此记为 $\sigma_\varepsilon$；最小二乘的残差方差估计记作 $\hat\sigma_\varepsilon^2$，不用 $s^2$ |
| $x_{t+1}$ | 支付 | 带常数项的特征向量记为 $\tilde{\mathbf z}_{i,t-1}$，不用 $x$ |
| $P_t$ 与 $p_t$ | 大写为价格水平，小写为对数价格 | 同理，小写 $d_t$、$c_t$、$r_t$ 为对数；面板中的 $r_{i,t}$ 是水平超额收益，是明确声明的例外 |
| $\bv\lambda$ | 因子风险溢价 | 第四部分的正则化（惩罚）系数记为 $\kappa$，或在该章另行声明，不用 $\lambda$ |
| $\mathbf u_t$、$u_{i,t}$ | $\mathbf u_t$ 为因子的标准化冲击向量；$u_{i,t}$ 为时间序列回归的残差（标量，见第 0 章 0.2 节与习题） | 两者由字体与下标区分，不改 |
| $\mathbf e_{i,t}$ 与 $\varepsilon_{i,t}$ | $\mathbf e_{i,t}$ 为特征的更新冲击；$\varepsilon_{i,t}$ 为收益的异质性冲击 | 注意区分 |
| $\mathbf c$ | 第 0 章中遗漏变量偏误的系数向量 | 只在第 0 章局部使用；粗体，与标量的对数消费 $c_t$ 不同 |

## 与 `apsim` 代码字段的对应

| 代码 | 记号 | 备注 |
|---|---|---|
| `MarketParams.n_stocks`、`n_periods`、`n_chars`、`n_factors` | $N$、$T$、$L$、$K$ | |
| `factor_mean`、`factor_vol` | $\bv\mu$、$\bv\sigma = (\sigma_1, \dots, \sigma_K)'$ | |
| `MarketParams.factor_cov` | $\bv\Sigma_f$ | 当前为 $\operatorname{diag}(\sigma_1^2, \dots, \sigma_K^2)$ |
| `char_persistence`、`idio_vol` | $\phi$、$\sigma_\varepsilon$ | |
| `gamma`、`alpha_coef` | $\bv\Gamma$、$\bv\theta$ | 为 `None` 时分别取默认载荷矩阵与零向量 |
| `SimulatedMarket.gamma`、`SimulatedMarket.theta` | $\bv\Gamma$、$\bv\theta$ | 只读数组 |
| `SimulatedMarket.factors` 第 $t$ 行 | $\mathbf f_t'$ | 行索引 `period` 取 $1, \dots, T$ |
| `panel` 的索引 `(period, stock)` | $(t, i)$ | `period` 从 1 开始，`stock` 从 0 开始 |
| `panel["z1"]`……`panel["zL"]` | $z_{1,i,t-1}, \dots, z_{L,i,t-1}$ | 期初特征，比收益早一期 |
| `panel["beta1"]`……`panel["betaK"]` | $\beta_{1,i,t}, \dots, \beta_{K,i,t}$ | |
| `panel["alpha"]` | $\alpha_{i,t}$ | |
| `panel["exret"]` | $r_{i,t}$ | |
| `SimulatedMarket.sdf_weights` | $\mathbf b$ | |
| `SimulatedMarket.sdf()` | $m_t$，$t = 1, \dots, T$ | |
| `simulate_market(params, seed=0)` | | `seed` 必须以关键字传入，为整数或 `np.random.Generator` |
