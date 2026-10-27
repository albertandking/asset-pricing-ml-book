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

`apsim/market.py` 中生成收益的三行代码可以逐一对应到上面的公式（数组的第一维是期，数组下标从 0 起，`x[t-1]` 对应第 $t$ 期）：

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

(notation-unpack)=
## 范例：拆开讲 $\bv\beta_i'\mathbf f_t$

本书的公式都按同一个顺序“拆开讲”：每一个编号公式，以及每一个向量或矩阵表达式在全书（或本章）第一次出现时，依次**写出分量、标维度、展开运算、讲特例、写数据矩阵**；标量公式则逐项解释每个符号的含义和单位。同一个式子在后文再次出现时不再重复拆解，而是链接到第一次拆开讲的地方，例如“各分量的含义见[记号表的范例](#notation-unpack)”。下面以时间序列回归 $r_{i,t} = a_i + \bv\beta_i'\mathbf f_t + u_{i,t}$ 中的因子部分 $\bv\beta_i'\mathbf f_t$ 为例示范这五步，它是全书拆解公式的标准写法。

**第一步，写出分量。** $\mathbf f_t$ 是 $K$ 个因子在第 $t$ 期的值竖着排成的列向量：

```{math}
:enumerated: false
\mathbf f_t = \begin{pmatrix} f_{1,t} \\ f_{2,t} \\ \vdots \\ f_{K,t} \end{pmatrix} .
```

- $f_{1,t}$：第 1 个因子在第 $t$ 期的值，例如市场因子 MKT（市场组合的超额收益）
- $f_{2,t}$：第 2 个因子在第 $t$ 期的值，例如规模因子 SMB（小市值组合减大市值组合）
- ……
- $f_{K,t}$：第 $K$ 个因子在第 $t$ 期的值

股票 $i$ 对这 $K$ 个因子各有一个载荷，同样竖着排成列向量：

```{math}
:enumerated: false
\bv\beta_i = \begin{pmatrix} \beta_{1,i} \\ \beta_{2,i} \\ \vdots \\ \beta_{K,i} \end{pmatrix} ,
```

其中 $\beta_{k,i}$ 是股票 $i$ 对第 $k$ 个因子的载荷：其他因子不变时，第 $k$ 个因子每高 1 个百分点，股票 $i$ 的收益平均高 $\beta_{k,i}$ 个百分点。下标的顺序是“因子在前、股票在后”，与时变载荷 $\bv\beta_{i,t}$ 的分量 $\beta_{k,i,t}$ 一致；不少文献写作 $\beta_{ik}$，含义相同。

**第二步，标维度。** $\mathbf f_t$ 与 $\bv\beta_i$ 都是 $K\times 1$ 的列向量，即 $\mathbf f_t \in \Real^K$、$\bv\beta_i \in \Real^K$。转置把列向量横过来，$\bv\beta_i'$ 是 $1\times K$ 的行向量：$\bv\beta_i' = (\beta_{1,i}, \beta_{2,i}, \dots, \beta_{K,i})$。$1\times K$ 乘 $K\times 1$，结果是 $1\times 1$，即一个标量。

**第三步，展开运算。**

```{math}
:enumerated: false
\underbrace{\bv\beta_i'}_{1\times K}\,\underbrace{\mathbf f_t}_{K\times 1}
= (\beta_{1,i}, \beta_{2,i}, \dots, \beta_{K,i}) \begin{pmatrix} f_{1,t} \\ f_{2,t} \\ \vdots \\ f_{K,t} \end{pmatrix}
= \beta_{1,i} f_{1,t} + \beta_{2,i} f_{2,t} + \dots + \beta_{K,i} f_{K,t}
= \sum_{k=1}^{K} \beta_{k,i} f_{k,t} .
```

白话：$\bv\beta_i'\mathbf f_t$ 是 $K$ 个因子共同贡献给股票 $i$ 的那部分收益，即每个因子的收益乘以股票对它的敏感度，再加总。例如[第 0 章“载荷”](#sec-ch0-loading)一节中的大盘银行股，$\bv\beta_i = (0.7, -0.5, 0.8)'$，某个月 $\mathbf f_t = (3\%, 3\%, -1\%)'$，则 $\bv\beta_i'\mathbf f_t = 0.7 \times 3\% + (-0.5) \times 3\% + 0.8 \times (-1\%) = 2.1\% - 1.5\% - 0.8\% = -0.2\%$。

**第四步，特例。** $K = 1$ 时只有市场一个因子，$\bv\beta_i'\mathbf f_t = \beta_{1,i} f_{1,t}$，回归退化为单因子的时间序列回归 $r_{i,t} = a_i + \beta_{1,i} f_{1,t} + u_{i,t}$；当 $f_{1,t}$ 是市场组合的超额收益、并且 CAPM 成立（截距 $a_i = 0$）时，它就是经典 CAPM 的时间序列形式；$K = 3$ 就是 Fama–French 三因子（市场、规模 SMB、价值 HML）[@fama1993common]，中国市场的 CH3 也是三个因子（市场、规模、价值，价值按盈利价格比划分）[@liu2019size]；$K = 5$ 就是 Fama–French 五因子（再加盈利 RMW、投资 CMA）[@fama2015five]，以此类推。

**第五步，数据矩阵。** 估计 $\bv\beta_i$ 时，要把 $T$ 期的数据堆叠起来，和矩阵的维度对应一下（很关键）：

- 因子矩阵 $\mathbf F = (\mathbf f_1, \dots, \mathbf f_T)' \in \Real^{T\times K}$，第 $t$ 行是 $\mathbf f_t'$：一共 $T$ 个时间点，每个时间点有 $K$ 个因子，所以是 $T$ 行 $K$ 列。
- 设计矩阵 $[\mathbf 1_T, \mathbf F]$ 在 $\mathbf F$ 左边多了一列全 1 向量，用来估计截距 $a_i$，维度是 $T\times(1+K)$。
- 所有股票共用同一套因子时间序列 $\mathbf F$（因而共用同一个设计矩阵），但每只股票有自己的一套 $\bv\beta_i$（以及截距 $a_i$），也就是每只股票对这 $K$ 个因子的敏感程度不一样。

股票 $i$ 的 $T$ 期回归于是可以一次写出：

```{math}
:enumerated: false
\underbrace{\begin{pmatrix} r_{i,1} \\ r_{i,2} \\ \vdots \\ r_{i,T} \end{pmatrix}}_{T\times 1}
= \underbrace{\begin{pmatrix} 1 & f_{1,1} & \cdots & f_{K,1} \\ 1 & f_{1,2} & \cdots & f_{K,2} \\ \vdots & \vdots & & \vdots \\ 1 & f_{1,T} & \cdots & f_{K,T} \end{pmatrix}}_{[\mathbf 1_T,\ \mathbf F]:\ T\times(1+K)}
\underbrace{\begin{pmatrix} a_i \\ \beta_{1,i} \\ \vdots \\ \beta_{K,i} \end{pmatrix}}_{(1+K)\times 1}
+ \underbrace{\begin{pmatrix} u_{i,1} \\ u_{i,2} \\ \vdots \\ u_{i,T} \end{pmatrix}}_{T\times 1} .
```

设计矩阵的第 $t$ 行乘以系数向量，就是第 $t$ 期的 $a_i + \bv\beta_i'\mathbf f_t$。反过来，把 $N$ 只股票的 $\bv\beta_i'$ 按行堆叠，得到 $N\times K$ 的载荷矩阵 $\mathbf B$，第 $i$ 行是 $\bv\beta_i'$；$\mathbf B\mathbf f_t$（$N\times K$ 乘 $K\times 1$，得 $N\times 1$）的第 $i$ 个元素就是 $\bv\beta_i'\mathbf f_t$。代码中，$\mathbf F$ 是形状为 `(T, K)` 的 `mkt.factors.to_numpy()`，设计矩阵是 `np.column_stack([np.ones(T), F])`，形状为 `(T, 1 + K)`。

## 下标与维度

| 记号 | 含义 | 说明 |
|---|---|---|
| $t$ | 时间下标（期） | 第 0 章的模拟市场为月度，第 1 章的 `apsim.consumption` 为年度；$t = 1, \dots, T$ |
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
| $p_t$、$d_t$、$c_t$、$r_t$ | 对数价格、对数红利、对数消费、对数收益 | 如 $p_t = \log P_t$；第 1 章起使用 $c_t$、$d_t$，第 1 章不用 $r_t$，对数收益直接写作 $\log R_{t+1}$；第 6、7 章沿用 |
| $m_{t+1}$ | 随机贴现因子（stochastic discount factor，SDF） | 面板部分写作 $m_t$；`SimulatedMarket.sdf()` |
| $P_t = \E_t[m_{t+1} x_{t+1}]$ | 基本定价方程 | 对总收益为 $1 = \E_t[m_{t+1} R_{t+1}]$，对超额收益为 $0 = \E_t[m_{t+1} R^e_{t+1}]$ |

## 消费、偏好与 Lucas 树（第 1 章）

第 1 章的数值例子都使用年度的假设参数。下表“说明”一列给出对应的 `apsim.consumption` 字段：`ConsumptionParams` 的参数与只读属性，以及 `simulate_consumption` 返回值中 `data` 的列。

| 记号 | 含义 | 说明 |
|---|---|---|
| $C_t$ | 第 $t$ 期的消费（水平） | 大写为水平；Lucas 树中 $C_t = D_t$ |
| $c_t$、$\Delta c_{t+1}$ | 对数消费 $\log C_t$；对数消费增长 $c_{t+1} - c_t = \mu_c + \sigma_c\varepsilon_{t+1}$ | $\Delta c_t$ 为 `data["dc"]` |
| $D_t$ | 第 $t$ 期的红利（水平） | 红利索取权获得红利流 $D_{t+1}, D_{t+2}, \dots$ |
| $d_t$、$\Delta d_{t+1}$ | 对数红利 $\log D_t$；对数红利增长 $\mu_d + \sigma_d\eta_{t+1}$ | $\Delta d_t$ 为 `data["dd"]` |
| $\delta$ | 主观贴现因子，理论上 $0 < \delta < 1$，越小越没耐心 | `delta`，默认 0.99；代码只要求 $\delta > 0$，价格是否有限由 $a < 1$ 判断 |
| $\gamma$ | 风险厌恶系数（相对风险厌恶系数 $-Cu''(C)/u'(C)$）；幂效用下跨期替代弹性为 $1/\gamma$ | `gamma`，默认 2；与载荷矩阵 $\bv\Gamma$ 无关 |
| $u(\cdot)$ | 效用函数；幂效用 $u(C) = C^{1-\gamma}/(1-\gamma)$，$u'(C) = C^{-\gamma}$ | 撇号在这里表示导数，见“已预留与已知冲突” |
| $e_t$、$\xi$ | 1.1 节两期问题中的收入（禀赋）与买入资产的份数 | 只在 1.1 节局部使用 |
| $q_s$、$\mathbf q$、$\mathbf X$ | 状态价格 $q_s = \pi_s m_s$（只在状态 $s$ 支付 1 元的证券今天的价格）及其组成的列向量；支付矩阵，第 $i$ 行为资产 $i$ 在各状态下的支付 | 1.3 节两状态例子中 $\mathbf P = \mathbf X\mathbf q$（$3\times 2$ 乘 $2\times 1$）；$\mathbf X$ 与回归的设计矩阵同字母，由上下文区分 |
| $m_{t+1} = \delta e^{-\gamma\Delta c_{t+1}}$ | 幂效用下的 SDF | $m_t$ 为 `data["m"]`；$\E[m]$ 为 `sdf_mean` |
| $M_{t,t+j}$ | 从第 $t$ 期到第 $t+j$ 期的多期 SDF $m_{t+1}m_{t+2}\cdots m_{t+j}$，$M_{t,t} = 1$ | 为第 $t+j$ 期的支付定价 |
| $\mu_c$、$\sigma_c$ | 对数消费增长的均值、标准差 | `mu_c`、`sigma_c`，默认均为 0.02 |
| $\mu_d$、$\sigma_d$ | 对数红利增长的均值、标准差 | `mu_d`、`sigma_d`，默认 0.02、0.12 |
| $\rho_{cd}$ | 消费增长与红利增长的相关系数，$\Cov(\Delta c, \Delta d) = \rho_{cd}\sigma_c\sigma_d$ | `rho_cd`，默认 0.5 |
| $\varepsilon_{t+1}$、$\nu_{t+1}$、$\eta_{t+1}$ | 消费冲击、与之独立的红利冲击、红利增长的标准化冲击 $\eta_{t+1} = \rho_{cd}\varepsilon_{t+1} + \sqrt{1-\rho_{cd}^2}\,\nu_{t+1}$ | 都是标准正态的标量；$\varepsilon_{t+1}$ 不带资产下标，与异质性冲击 $\varepsilon_{i,t}$ 不同 |
| $a$ | $\E[m_{t+1}e^{\Delta d_{t+1}}]$：明年红利增长因子 $D_{t+1}/D_t$ 的索取权今天的价格；$a^j$ 对应第 $j$ 年 | `dividend_discount`；$a < 1$ 是价格为正且有限的条件 |
| $P_t/D_t$ | 价格—红利比（$P/D$），独立同分布时为常数 $a/(1-a)$ | `price_dividend_ratio`（$a \geq 1$ 时抛出 `ValueError`）；倒数 $D/P$ 为股息率；不是市盈率 $P/E$ |
| $R_{t+1}$、$R^f$ | 红利索取权的总收益 $e^{\Delta d_{t+1}}/a$；无风险总收益 $1/\E[m]$（常数） | `data["R"]`、`data["Rf"]`；$\E[R]$ 为 `expected_return`，$R^f$ 为 `risk_free` |
| $\log(\E[R]/R^f)$ | 对数溢价，对红利索取权等于 $\gamma\rho_{cd}\sigma_c\sigma_d$ | `log_equity_premium` |
| $\beta_{\Delta c}$、$\beta_{\Delta c,i}$ | 超额收益对消费增长做时间序列回归的斜率 $\Cov(R^e, \Delta c)/\Var(\Delta c)$；第 $i$ 项资产的记作 $\beta_{\Delta c,i}$，是标量，不加粗 | 消费 CAPM：$\E[R^e] \approx \beta_{\Delta c}\lambda_c$；$N$ 项资产的 $\beta_{\Delta c,i}$ 跨资产堆成的一列记作 $\mathbf B$（估计值 $\hat{\mathbf B}$），$N\times 1$，即 $K = 1$ 时的载荷矩阵（第 1 章实验三中 $N = 5$），与下文 $\hat{\mathbf B}$ 的约定一致 |
| $\lambda_c$ | 消费增长的风险溢价 $\gamma\sigma_c^2$，即每单位 $\beta_{\Delta c}$ 的补偿 | 不等于 $\E[\Delta c] = \mu_c$；无对应字段 |

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
| $\mathbf g_t \in \Real^{L+1}$ | 第 $t$ 期收益对 $\tilde{\mathbf z}_{i,t-1}$ 的横截面系数，$\mathbf g_t = \bv\Gamma\mathbf f_t + (0, \bv\theta')'$ | $\hat{\mathbf g}_t$ 为第 $t$ 期的最小二乘估计，分量 $\hat g_{j,t}$，$j = 0, \dots, L$；见第 0 章 0.2 节“因子、特征、载荷的关系” |
| $\bv\beta_i \in \Real^K$ | 时间序列回归 $r_{i,t} = a_i + \bv\beta_i'\mathbf f_t + u_{i,t}$ 中股票 $i$ 的常数载荷（不带下标 $t$） | 即第 0 章习题“时间序列回归的截距”中的 $\mathbf h_i$；本书模型中总体值为 $\E[\bv\beta_{i,t}]$ |
| $a_i$、$\hat a_i$；$\hat{\bv\beta}_i \in \Real^K$ | 股票 $i$ 时间序列回归的截距（标量）及其估计，常数载荷 $\bv\beta_i$ 的估计 | 载荷时变、且载荷与因子的实现值独立时（本模拟器满足，真实数据中条件 CAPM 一类的设定会破坏它），给定特征路径，$\hat{\bv\beta}_i$ 估计样本期内的平均载荷；$\hat{\mathbf B} \in \Real^{N\times K}$ 的第 $i$ 行为 $\hat{\bv\beta}_i'$（[拆解](#primer-unpack-cs)）；见[导读：几种回归](#sec-ch0-regs)，模拟验证见[续页](#sec-ch0-regs-sim) |
| $\mathbf F \in \Real^{T\times K}$ | 因子按期堆叠，第 $t$ 行为 $\mathbf f_t'$ | 时间序列回归的设计矩阵为 $[\mathbf 1_T, \mathbf F]$；`factors.to_numpy()` |
| $\bar{\mathbf g} \in \Real^{L+1}$ | Fama–MacBeth 点估计 $T^{-1}\sum_t\hat{\mathbf g}_t$ | 标准误逐分量为 $\hat g_{j,t}$ 的时间序列标准差除以 $\sqrt T$；见[导读：几种回归](#sec-ch0-regs)（[拆解](#primer-unpack-fm)） |
| $\lambda_{0,t}$、$\hat{\bv\lambda}_t \in \Real^K$、$\bar{\bv\lambda}$ | 第 $t$ 期收益对载荷估计值横截面回归的截距与斜率（每单位载荷的实现补偿），及斜率的时间平均 | 两步法的第二步；$\bar{\bv\lambda}$ 估计风险溢价 $\bv\lambda$（[拆解](#primer-unpack-cs)） |
| $\hat{\mathbf g}^{\mathrm P} \in \Real^{L+1}$ | 混合回归（pooled）的系数估计 | $\hat{\mathbf g}^{\mathrm P} = \sum_t\mathbf W_t\hat{\mathbf g}_t$，$\mathbf W_t = (\sum_s\tilde{\mathbf Z}_{s-1}'\tilde{\mathbf Z}_{s-1})^{-1}\tilde{\mathbf Z}_{t-1}'\tilde{\mathbf Z}_{t-1}$ 为 $(L+1)\times(L+1)$ 权重矩阵（[拆解](#primer-unpack-w)） |
| $\mathbf w_t \in \Real^N$ | 第 $t$ 期的组合权重向量 | 组合收益 $\mathbf w_t'\mathbf r_t$，组合载荷 $\mathbf B_t'\mathbf w_t$；零成本多空组合满足 $\mathbf w_t'\mathbf 1_N = 0$ |

## 特征与载荷矩阵

| 记号 | 含义 | 说明 |
|---|---|---|
| $\mathbf z_{i,t} \in \Real^L$ | 股票 $i$ 在第 $t$ 期末的特征向量，分量 $z_{j,i,t}$ | 面板第 $t$ 期的列 `zj` 存放 $z_{j,i,t-1}$ |
| $\tilde{\mathbf z}_{i,t-1} \in \Real^{L+1}$ | 带常数项的特征向量 $(1, \mathbf z_{i,t-1}')'$ | 分量记为 $\tilde z_{j,i,t-1}$，$j = 0, \dots, L$，$\tilde z_{0,i,t-1} = 1$ |
| $\bv\Gamma \in \Real^{(L+1)\times K}$ | 载荷矩阵，$\bv\beta_{i,t} = \bv\Gamma' \tilde{\mathbf z}_{i,t-1}$ | 第 0 行对应常数项，第 $j$ 行对应特征 $j$；`gamma` |
| $\bv\theta \in \Real^L$ | 定价误差系数，$\alpha_{i,t} = \bv\theta' \mathbf z_{i,t-1}$ | `alpha_coef`，默认全为 0 |
| $\mathbf e_{i,t} \in \Real^L$、$\mathbf E_t \in \Real^{N\times L}$ | 特征的更新冲击向量；同一期 $N$ 只股票的冲击按行堆叠成的冲击矩阵，第 $i$ 行为 $\mathbf e_{i,t}'$ | $\mathbf Z_t = \phi\,\mathbf Z_{t-1} + \sqrt{1-\phi^2}\,\mathbf E_t$（第 0 章 0.4 节，[拆解](#ch0-unpack-char)）；$\mathbf E_t$ 是粗体正体的大写字母 E，与表示条件期望的空心体 $\E_t[\cdot]$ 不同 |
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
| $\delta$ | 主观贴现因子（Campbell 等文献的惯例），第 1 章起正式使用 | 定价误差系数因此记为 $\bv\theta$，不用 $\delta$ |
| $\gamma$ 与 $\bv\Gamma$ | 小写 $\gamma$ 为风险厌恶系数（第 1 章起）；大写粗体 $\bv\Gamma$ 为载荷矩阵 | 由大小写与字体区分；代码中 `ConsumptionParams.gamma` 与 `MarketParams.gamma` 同名，前者是标量，后者是载荷矩阵 |
| $\rho$ 与 $\rho_{cd}$ | 不带下标的 $\rho$ 为一般的相关系数与 Campbell–Shiller 对数线性化常数（第 6 章）；第 1 章消费增长与红利增长的相关系数一律记作 $\rho_{cd}$，与代码字段 `rho_cd` 对应 | 特征持续性因此记为 $\phi$；$\rho_{cd}$ 由下标 $cd$（消费、红利）与不带下标的 $\rho$ 区分 |
| $s_t$ | Campbell–Cochrane 模型中的（对数）盈余消费比 | 异质性波动率因此记为 $\sigma_\varepsilon$；最小二乘的残差方差估计记作 $\hat\sigma_\varepsilon^2$，不用 $s^2$ |
| $x_{t+1}$ | 支付 | 带常数项的特征向量记为 $\tilde{\mathbf z}_{i,t-1}$，不用 $x$ |
| $P_t$ 与 $p_t$ | 大写为价格水平，小写为对数价格 | 同理，小写 $d_t$、$c_t$、$r_t$ 为对数；面板中的 $r_{i,t}$ 是水平超额收益，是明确声明的例外 |
| $\bv\lambda$ | 因子风险溢价 | 第四部分的正则化（惩罚）系数记为 $\kappa$，或在该章另行声明，不用 $\lambda$ |
| $\mathbf u_t$、$u_{i,t}$、$u(\cdot)$ | $\mathbf u_t$ 为因子的标准化冲击向量；$u_{i,t}$ 为回归残差（标量；时间序列、横截面与混合回归通用，见[导读：几种回归](#sec-ch0-regs)与第 0 章习题）；$u(\cdot)$ 为效用函数（第 1 章） | 三者由字体、下标与括号中的自变量区分，不改 |
| $\mathbf e_{i,t}$ 与 $\varepsilon_{i,t}$ | $\mathbf e_{i,t}$ 为特征的更新冲击；$\varepsilon_{i,t}$ 为收益的异质性冲击 | 注意区分；第 1 章的 $e_t$（标量，收入）与 $\varepsilon_{t+1}$（消费冲击）不带资产下标 |
| $\mathbf E_t$ 与 $\E_t$ | 粗体正体的 $\mathbf E_t$ 为第 0 章 0.4 节特征更新冲击按行堆叠的 $N\times L$ 矩阵；空心体 $\E_t[\cdot]$ 为条件期望 | 由字体区分 |
| $a$ 与 $a_i$ | $a$ 为第 1 章的常数 $\E[m_{t+1}e^{\Delta d_{t+1}}]$；$a_i$ 为时间序列回归的截距（包括第 1 章“模拟实验”实验三中的截距） | 由下标区分：红利贴现常数 $a$ 不带下标，回归截距 $a_i$ 带资产下标 |
| $\mathbf X$ | 第 1 章 1.3 节的支付矩阵（行为资产、列为状态，见[拆解](#ch1-unpack-states)）；回归的设计矩阵（如 $(\mathbf X'\mathbf X)^{-1}\mathbf X'\mathbf y$，第 1 章“模拟实验”实验三的 $T\times 2$ 设计矩阵，代码中的 `X`） | 由上下文区分：支付矩阵只在 1.3 节局部使用 |
| 撇号 $'$ | 作用于向量、矩阵时表示转置；作用于标量函数时表示导数，如 $u'(C)$、$u''(C)$、$V'(\xi)$（第 1 章） | 由作用对象区分：本书的转置只作用于粗体的向量与矩阵 |
| $\mathbf W_t$ | 第 0 章中混合回归与 Fama–MacBeth 关系式里的 $(L+1)\times(L+1)$ 权重矩阵 | 只在第 0 章局部使用；第 11 章 GMM 的加权矩阵记为 $\mathbf W$（不带下标），二者无关 |
| $\mathbf c$ | 第 0 章中遗漏变量偏误的系数向量 | 只在第 0 章局部使用；粗体，与标量的对数消费 $c_t$ 不同 |
| $\mathbf L$ 与 $L$ | 粗体大写 $\mathbf L$ 为第 1 章 1.4 节中 $(\Delta c_{t+1}, \Delta d_{t+1})$ 协方差矩阵的 Cholesky 因子（$2\times 2$ 下三角矩阵，见[拆解](#ch1-unpack-model)）；斜体 $L$ 为特征数 | 由字体区分；$\mathbf L$ 只在 1.4 节局部使用 |

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
| `ConsumptionParams.delta`、`gamma` | $\delta$、$\gamma$ | 第 1 章；`gamma` 是标量，与 `MarketParams.gamma` 无关 |
| `mu_c`、`sigma_c`、`mu_d`、`sigma_d`、`rho_cd` | $\mu_c$、$\sigma_c$、$\mu_d$、$\sigma_d$、$\rho_{cd}$ | 年度假设参数 |
| `sdf_mean`、`risk_free` | $\E[m]$、$R^f = 1/\E[m]$ | 只读属性 |
| `dividend_discount`、`price_dividend_ratio` | $a$、$P/D = a/(1-a)$ | $a \geq 1$ 时后者抛出 `ValueError` |
| `expected_return`、`log_equity_premium` | $\E[R]$、$\log(\E[R]/R^f) = \gamma\rho_{cd}\sigma_c\sigma_d$ | 红利索取权 |
| `simulate_consumption(params, seed=0, n_years=1000).data` 的列 `dc`、`dd`、`m`、`R`、`Rf` | $\Delta c_t$、$\Delta d_t$、$m_t$、$R_t$、$R^f$ | 索引 `year` 取 $1, \dots, T$；`seed` 的规则同 `simulate_market` |

(notation-reading)=
## 符号读法

读公式的第一步是能把它念出来。下面列出本书用到的希腊字母、修饰符号与运算符号的读法和在本书中的含义。英文读法是国际通行的念法，括号中的中文只是近似读音，帮助记忆。各章出现新符号时，会补充到这里。

### 希腊字母

| 符号 | 英文读法 | 近似读音 | 本书含义 |
|---|---|---|---|
| $\alpha$ | alpha | 阿尔法 | 定价误差 $\alpha_{i,t}$ |
| $\beta$、$\bv\beta$ | beta | 贝塔 | 载荷 $\bv\beta_{i,t}$ |
| $\gamma$ | gamma（小写） | 伽马 | 风险厌恶系数 $\gamma$ |
| $\Gamma$ | Gamma（大写） | 伽马 | 载荷矩阵 $\bv\Gamma$ |
| $\delta$ | delta（小写） | 德尔塔 | 主观贴现因子 $\delta$ |
| $\Delta$ | Delta（大写） | 德尔塔 | 一期的变化，如 $\Delta c_{t+1} = c_{t+1} - c_t$ |
| $\varepsilon$ | epsilon | 艾普西隆 | 异质性冲击 $\varepsilon_{i,t}$；第 1 章的消费冲击 $\varepsilon_{t+1}$ |
| $\eta$ | eta | 伊塔 | 第 1 章红利增长的标准化冲击 $\eta_{t+1}$ |
| $\theta$ | theta | 西塔 | 定价误差系数 $\bv\theta$ |
| $\kappa$ | kappa | 卡帕 | 预留：正则化系数 |
| $\lambda$ | lambda | 兰姆达 | 因子风险溢价 $\bv\lambda$；消费增长的风险溢价 $\lambda_c$ |
| $\mu$ | mu | 缪 | 因子均值 $\bv\mu$；对数增长的均值 $\mu_c$、$\mu_d$ |
| $\nu$ | nu | 纽 | 第 1 章与消费冲击独立的红利冲击 $\nu_{t+1}$ |
| $\xi$ | xi | 克西 | 1.1 节中买入资产的份数 $\xi$ |
| $\rho$ | rho | 柔 | 相关系数（第 1 章消费增长与红利增长的相关系数带下标，记作 $\rho_{cd}$）；预留：Campbell–Shiller 常数 |
| $\sigma$ | sigma（小写） | 西格玛 | 标准差，如 $\sigma_\varepsilon$、$\bv\sigma$、$\sigma_c$、$\sigma_d$ |
| $\Sigma$ | Sigma（大写） | 西格玛 | 协方差矩阵 $\bv\Sigma_f$（作求和号时见下表） |
| $\phi$ | phi（小写） | 斐 | 特征持续性 $\phi$ |
| $\Phi$ | Phi（大写） | 斐 | 标准正态分布的分布函数 |

大小写含义不同：$\sigma$ 是标准差，$\Sigma$ 是协方差矩阵；$\phi$ 是特征持续性，$\Phi$ 是正态分布函数；$\gamma$ 是风险厌恶系数，$\bv\Gamma$ 是载荷矩阵；$\delta$ 是主观贴现因子，$\Delta$ 表示一期的变化。同一个字母加粗与否含义也不同（见本页“字体”），例如 $\beta_i$ 是单因子时的标量载荷，$\bv\beta_i$ 是载荷向量。

### 修饰符号与上下标

| 写法 | 读作 | 含义 |
|---|---|---|
| $\hat{\bv\beta}_i$ | beta hat i | 估计值：“帽子”表示从数据估计出来的量 |
| $\bar{\mathbf g}$ | g bar | 平均值：本书中多指时间平均 |
| $\tilde{\mathbf z}_{i,t-1}$ | z tilde i，t 减 1 | 变形后的量：本书中指在 $\mathbf z$ 前加常数 1 |
| $\mathbf b'$ | b prime，或 b 转置 | 转置；撇号 $'$ 作用于向量、矩阵时只表示转置 |
| $u'(C)$、$u''(C)$ | u prime C；u double prime C | 标量函数的一阶、二阶导数（第 1 章），见“已预留与已知冲突” |
| $\xi^*$ | xi star | 星号表示最优值 |
| $r_{i,t}$ | r i t（r 下标 i 逗号 t） | 下标：股票 $i$、第 $t$ 期 |
| $R^f_{t-1}$ | R f，下标 t 减 1 | 上标 $f$ 是标签（free，无风险），不是幂次 |
| $\sigma_\varepsilon^2$ | sigma epsilon 平方 | 上标 2 是幂次 |
| $\mathbf I_K$ | K 阶单位矩阵 | 下标表示维度 |
| $\mathbf 1_N$ | N 维全 1 向量 | 下标表示维度 |
| $\Real^{N\times K}$ | R N 乘 K | 所有 $N$ 行 $K$ 列实矩阵的集合 |

上标可能是幂次（$\sigma^2$），也可能是标签（$R^f$、$R^e$），要看上下文：标签通常是字母，幂次通常是数字或 $-1$。$\bv\Sigma_f^{-1}$ 读作“Sigma f 的逆”，表示逆矩阵。

### 运算符与关系符号

| 写法 | 读作 | 含义 |
|---|---|---|
| $\E[x]$ | x 的期望 | 总体均值 |
| $\E_{t-1}[x]$ | 在 t 减 1 期信息下 x 的条件期望 | 用第 $t-1$ 期末已知的信息做的最优预测 |
| $\E[x \mid \mathbf z]$ | 给定 z 时 x 的条件期望 | 竖线读作“给定” |
| $\Var(x)$、$\Cov(x, y)$ | x 的方差；x 与 y 的协方差 | |
| $\sum_{t=1}^T$ | 对 t 从 1 到 T 求和 | 大写 Sigma 作求和号 |
| $\prod_{k=1}^{j}$ | 对 k 从 1 到 j 求积 | 大写 Pi 作求积号，如多期 SDF $M_{t,t+j} = \prod_{k=1}^{j} m_{t+k}$ |
| $\partial y/\partial x$ | y 对 x 的偏导数 | $\partial$ 读作 partial，如 $\partial\log a/\partial\mu_c$ |
| $\lim_{J\to\infty}$ | J 趋于无穷时的极限 | 如第 1 章的横截性条件 |
| $\mathbf a'\mathbf b$ | a 转置乘 b | 两个列向量的内积，结果是标量 |
| $\mathbf A\mathbf B$ | A 乘 B | 矩阵乘法 |
| $\mathbf A^{-1}$ | A 的逆 | |
| $\mathbf A \otimes \mathbf B$ | A 与 B 的 Kronecker 积 | 见第 0 章“模拟实验” |
| $\operatorname{vec}(\mathbf A)$ | vec A | 把矩阵按列堆成一个长列向量 |
| $\operatorname{diag}(\cdot)$ | diag | 由向量构成对角矩阵 |
| $\in$ | 属于 | 如 $\mathbf f_t \in \Real^K$：$\mathbf f_t$ 是 $K$ 维向量 |
| $\sim$ | 服从 | 如 $\varepsilon \sim N(0, \sigma^2)$：服从正态分布 |
| $\approx$ | 约等于 | |
| $\to$、$\infty$ | 趋于；无穷 | 如 $T \to \infty$：样本期数趋于无穷 |

### 念法示范

| 公式 | 念法 |
|---|---|
| $\bv\beta_{i,t} = \bv\Gamma'\tilde{\mathbf z}_{i,t-1}$ | beta i t 等于 Gamma 转置乘 z tilde i，t 减 1 |
| $r_{i,t} = \alpha_{i,t} + \bv\beta_{i,t}'\mathbf f_t + \varepsilon_{i,t}$ | r i t 等于 alpha i t，加 beta i t 转置乘 f t，加 epsilon i t |
| $m_t = 1 - \mathbf b'(\mathbf f_t - \bv\mu)$ | m t 等于 1 减 b 转置乘括号 f t 减 mu |
| $\mathbf b = \bv\Sigma_f^{-1}\bv\mu$ | b 等于 Sigma f 的逆乘 mu |
| $\E_{t-1}[r_{i,t}] = \alpha_{i,t} + \bv\beta_{i,t}'\bv\lambda$ | 在 t 减 1 期信息下 r i t 的条件期望，等于 alpha i t 加 beta i t 转置乘 lambda |
