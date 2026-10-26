# 记号表

本表在第 0 章确定，全书遵守。后续章节引入新记号时，会在该章首次出现处定义，并补充到本表。向量一律为列向量，$'$ 表示转置；粗体不单独使用，向量与标量从上下文区分。

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
- 横截面面板中，第 $t$ 期的一行同时记录“期初已知的特征 $z_{i,t-1}$”与“本期实现的超额收益 $r_{i,t}$”：**特征比收益早一期**。
- 理论部分沿用 $t \to t+1$ 的写法（如 $p_t = \E_t[m_{t+1} x_{t+1}]$）；面板与模拟部分沿用 $t-1 \to t$ 的写法（如 $\E_{t-1}[m_t r_{i,t}]$）。两种写法只差一个时间平移，含义相同。

## 收益、价格与随机贴现因子

| 记号 | 含义 | 说明 |
|---|---|---|
| $p_t$ | 资产在第 $t$ 期末的价格 | |
| $x_{t+1}$ | 第 $t+1$ 期的支付（payoff，下一期价格加红利） | |
| $R_{t+1}$ | 总收益（gross return） | $R_{t+1} = x_{t+1} / p_t$ |
| $R^f_t$ | 无风险总收益 | 第 $t$ 期末已知，适用于从 $t$ 到 $t+1$ |
| $R^e_{t+1}$ | 超额收益 | $R^e_{t+1} = R_{t+1} - R^f_t$ |
| $r_{i,t}$ | 股票 $i$ 在第 $t$ 期的超额收益（面板记号） | 即 $R^e_{i,t}$ 的面板写法；`panel["exret"]` |
| $m_{t+1}$ | 随机贴现因子（stochastic discount factor，SDF） | 面板部分写作 $m_t$；`SimulatedMarket.sdf()` |
| $p_t = \E_t[m_{t+1} x_{t+1}]$ | 基本定价方程 | 对总收益为 $1 = \E_t[m_{t+1} R_{t+1}]$，对超额收益为 $0 = \E_t[m_{t+1} R^e_{t+1}]$ |

## 因子、载荷与定价误差

| 记号 | 含义 | 说明 |
|---|---|---|
| $f_t$ | $K$ 维因子超额收益向量，分量 $f_{k,t}$ | `factors` 的列 `f1`……`fK` |
| $\mu$ | 因子均值 $\E[f_t]$ | `factor_mean` |
| $\sigma_k$ | 第 $k$ 个因子的波动率 | `factor_vol` |
| $\Sigma_f$ | 因子协方差矩阵 $\Var(f_t)$ | 模拟器中因子相互独立，$\Sigma_f = \operatorname{diag}(\sigma_1^2, \dots, \sigma_K^2)$ |
| $b$ | SDF 系数，$m_t = 1 - b'(f_t - \mu)$ | 一般情形 $b = \Sigma_f^{-1}\mu$；模拟器中化为 $b_k = \mu_k / \sigma_k^2$；`sdf_weights` |
| $\beta_{i,t}$ | 股票 $i$ 在第 $t$ 期的 $K$ 维因子载荷，分量 $\beta_{k,i,t}$ | 第 $t-1$ 期末已知；`panel["beta1"]`…… |
| $\lambda$ | 因子风险溢价 | 因子本身是超额收益时 $\lambda = \mu$ |
| $\alpha_{i,t}$ | 定价误差（错误定价） | $\alpha_{i,t} = \E_{t-1}[m_t r_{i,t}]$ 在本书模型中等于 $\delta' z_{i,t-1}$；`panel["alpha"]` |
| $\varepsilon_{i,t}$ | 异质性冲击 | $\varepsilon_{i,t} \sim N(0, s^2)$ |

## 特征与载荷矩阵

| 记号 | 含义 | 说明 |
|---|---|---|
| $z_{i,t}$ | 股票 $i$ 在第 $t$ 期末的 $L$ 维特征向量，分量 $z_{j,i,t}$ | 面板第 $t$ 期的列 `zj` 存放 $z_{j,i,t-1}$ |
| $x_{i,t-1}$ | 带常数项的特征向量 $(1, z_{i,t-1}')'$ | $L+1$ 维 |
| $\Gamma$ | $(L+1) \times K$ 载荷矩阵，$\beta_{i,t} = \Gamma' x_{i,t-1}$ | 第 0 行对应常数项，第 $j$ 行对应特征 $j$；`gamma` |
| $\delta$ | $L$ 维定价误差系数，$\alpha_{i,t} = \delta' z_{i,t-1}$ | `alpha_delta`，默认全为 0 |
| $\rho$ | 特征的一阶自相关（持续性） | `char_persistence` |
| $s$ | 异质性波动率 | `idio_vol` |

## 运算符

| 记号 | 含义 |
|---|---|
| $\E[\cdot]$ | 无条件期望 |
| $\E_t[\cdot]$ | 以第 $t$ 期末信息为条件的期望 |
| $\E[\cdot \mid z]$ | 以随机变量 $z$ 为条件的期望 |
| $\Var(\cdot)$、$\Cov(\cdot, \cdot)$ | 方差（或协方差矩阵）与协方差 |
| $\operatorname{diag}(a_1, \dots, a_K)$ | 以 $a_1, \dots, a_K$ 为对角元的对角矩阵 |
| $\bar{y}$ | 样本均值，例如 $\bar{y} = T^{-1}\sum_{t=1}^{T} y_t$；与总体期望 $\E[y]$ 严格区分 |

## 与 `apsim` 代码字段的对应

| 代码 | 记号 | 备注 |
|---|---|---|
| `MarketParams.n_stocks`、`n_periods`、`n_chars`、`n_factors` | $N$、$T$、$L$、$K$ | |
| `factor_mean`、`factor_vol` | $\mu$、$(\sigma_1, \dots, \sigma_K)$ | |
| `char_persistence`、`idio_vol` | $\rho$、$s$ | |
| `gamma`、`alpha_delta` | $\Gamma$、$\delta$ | 为 `None` 时分别取默认载荷矩阵与零向量 |
| `SimulatedMarket.factors` 第 $t$ 行 | $f_t'$ | 行索引 `period` 取 $1, \dots, T$ |
| `panel` 的索引 `(period, stock)` | $(t, i)$ | `period` 从 1 开始，`stock` 从 0 开始 |
| `panel["z1"]`……`panel["zL"]` | $z_{1,i,t-1}, \dots, z_{L,i,t-1}$ | 期初特征，比收益早一期 |
| `panel["beta1"]`……`panel["betaK"]` | $\beta_{1,i,t}, \dots, \beta_{K,i,t}$ | |
| `panel["alpha"]` | $\alpha_{i,t}$ | |
| `panel["exret"]` | $r_{i,t}$ | |
| `SimulatedMarket.sdf_weights` | $b$ | |
| `SimulatedMarket.sdf()` | $m_t$，$t = 1, \dots, T$ | |
