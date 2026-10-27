---
short_title: 导读：几种回归
kernelspec:
  name: python3
  display_name: Python 3
---

(sec-ch0-regs)=
# 导读：几种回归——时间序列、横截面、Fama–MacBeth 与混合回归

**本页的定位。** 本页是第二部分（实证计量）的预习，不属于第 0 章的必读内容，第一次阅读可以跳过，读到第 9 章之前再回来。本页用到第 0 章的基本概念、记号与模型：读之前应当读过 [0.2 基本概念速览](#sec-ch0-concepts)、[因子、特征、载荷的关系](#sec-ch0-fcl)与 [0.3～0.4 记号与模拟市场的模型](./ch00-model.md)、[0.5～0.6 SDF 的定价性质与代码验证](./ch00-pricing.md)；文中提到的误区四、交互回归与几道习题在第 0 章的 [A 股视角与模拟实验](./ch00-experiments.md)、[常见误区](./ch00-pitfalls.md)与[本章小结、习题与延伸阅读](./ch00-exercises.md)三页，遇到时回查即可。

**导读的两页。** 导读分为两页。本页介绍四种回归，逐个拆开它们的式子、标出数据矩阵的形状，最后给出对比总表（约 2.5 万字）；[下一页](#sec-ch0-regs-sim)在同一份模拟数据上跑全部四种回归，并列出第 0 章中已经用到这些回归的地方与业界的对应口径（约 2.6 万字）。字数按源文件的字符数统计，含公式与代码，不计页首元数据。

**本页准备。** 本书每个页面的代码在独立的内核中运行。下面的单元导入本页用到的模块；本页的模拟数据都在本页生成；同一份数据上的模拟实验在[下一页](#sec-ch0-regs-sim)。

```{code-cell} python
import numpy as np
import pandas as pd

from apsim import MarketParams, simulate_market, stats
```

**白话。** 实证资产定价用的数据几乎都是面板：一个维度是股票 $i$，另一个维度是时间 $t$，每个格子里是一个收益 $r_{i,t}$，同一格还附着期初特征 $\mathbf z_{i,t-1}$，同一行（同一期）共用一个因子 $\mathbf f_t$。本书出现的几种回归用的是同一张表，区别只在于**沿哪个维度跑**：固定一只股票、沿时间跑，是时间序列回归；固定一期、沿股票跑，是横截面回归；每期各跑一次横截面回归、再对系数求时间平均，是 Fama–MacBeth 程序；不分股票与时间，把全部格子放在一起跑一次，是混合回归（pooled regression）。跑的方向不同，得到的系数含义不同，回答的问题不同，标准误的算法也不同。

**示意图。** 下图是 $N$ 只股票、$T$ 期的收益面板，行为期、列为股票，与 `panel["exret"].unstack("stock")` 的排列相同。图中的 `g_t` 即正文的 $\hat{\mathbf g}_t$，`a_2`、`beta_2` 即 $\hat a_2$、$\hat{\bv\beta}_2$。

```text
        股票 i（横截面维度）
          i=1     i=2     i=3    . . .    i=N
       +-------+-------+-------+-------+-------+
 t=1   | r_1,1 | r_2,1 | r_3,1 | . . . | r_N,1 |  <- 第 1 期横截面回归，得 g_1
       +-------+-------+-------+-------+-------+
 t=2   | r_1,2 | r_2,2 | r_3,2 | . . . | r_N,2 |  <- 第 2 期横截面回归，得 g_2
       +-------+-------+-------+-------+-------+
 :     |   :   |   :   |   :   |   :   |   :   |
       +-------+-------+-------+-------+-------+
 t=T   | r_1,T | r_2,T | r_3,T | . . . | r_N,T |  <- 第 T 期横截面回归，得 g_T
       +-------+-------+-------+-------+-------+
 |                 ^
 v t               |
                   +-- 股票 2 的时间序列回归：沿这一列跑，得 a_2 与 beta_2

 纵向为时间 t，横向为股票 i

 Fama-MacBeth：对 g_1 至 g_T 求时间平均，并用它们的波动计算标准误
 混合回归：把表中全部 N x T 个格子放进同一个回归，只得到一组系数
```

下面逐一说明这四种回归。每种都按同一顺序介绍：式子与各项的维度、样本、得到什么、回答什么问题、前提、典型用途与主要陷阱。

#### 时间序列回归

对每只股票 $i$，沿 $t = 1, \dots, T$ 回归

```{math}
:enumerated: false
\underbrace{r_{i,t}}_{1\times 1} = \underbrace{a_i}_{1\times 1} + \underbrace{\bv\beta_i'}_{1\times K}\,\underbrace{\mathbf f_t}_{K\times 1} + \underbrace{u_{i,t}}_{1\times 1}, \qquad t = 1, \dots, T .
```

斜率 $\bv\beta_i \in \Real^K$ 不带下标 $t$，是一个常数；残差 $u_{i,t}$ 是标量。把因子按行叠成 $\mathbf F = (\mathbf f_1, \dots, \mathbf f_T)' \in \Real^{T\times K}$，设计矩阵就是 $[\mathbf 1_T, \mathbf F]$（$T\times(1+K)$），$N$ 只股票共用它。

(primer-unpack-ts)=
**拆开讲时间序列回归。** 这个式子与 0.2 节“用回归斜率估计载荷”中的回归完全相同：各项的含义与单位见[拆解](#ch0-unpack-ts)；因子部分 $\bv\beta_i'\mathbf f_t$ 的分量、维度与逐项展开见[拆解](#ch0-unpack-betaf)；股票 $i$ 的 $T$ 期观测堆叠成 $T\times 1$ 的收益向量、对 $T\times(1+K)$ 的设计矩阵 $[\mathbf 1_T, \mathbf F]$ 回归的写法见[记号表的范例](#notation-unpack)。下面“前提”中截距的表达式 $a_i = \E[r_{i,t}] - \bv\beta_i'\bv\lambda$ 里，$\bv\beta_i'\bv\lambda$ 的展开见[拆解](#ch0-unpack-lambda)，定价误差 α 见[拆解](#ch0-unpack-alpha)。这里只补上把 $N$ 只股票的回归并排写在一起的形式，下一页的代码正是这样一次解出 $N$ 个回归的：

```{math}
:enumerated: false
\underbrace{\begin{pmatrix} r_{1,1} & r_{2,1} & \cdots & r_{N,1} \\ r_{1,2} & r_{2,2} & \cdots & r_{N,2} \\ \vdots & \vdots & & \vdots \\ r_{1,T} & r_{2,T} & \cdots & r_{N,T} \end{pmatrix}}_{T\times N}
= \underbrace{[\mathbf 1_T, \mathbf F]}_{T\times(1+K)}\,
\underbrace{\begin{pmatrix} a_1 & a_2 & \cdots & a_N \\ \bv\beta_1 & \bv\beta_2 & \cdots & \bv\beta_N \end{pmatrix}}_{(1+K)\times N}
+ \underbrace{\begin{pmatrix} u_{1,1} & u_{2,1} & \cdots & u_{N,1} \\ \vdots & \vdots & & \vdots \\ u_{1,T} & u_{2,T} & \cdots & u_{N,T} \end{pmatrix}}_{T\times N} .
```

左边的收益矩阵与上面示意图的排列相同：第 $t$ 行是 $\mathbf r_t'$，第 $i$ 列是股票 $i$ 的时间序列。系数矩阵的第一行是 $N$ 个截距，下面 $K$ 行是 $N$ 只股票的载荷（每个 $\bv\beta_i$ 占一列中的 $K$ 行），第 $i$ 列 $(a_i, \bv\beta_i')'$ 是 $(1+K)\times 1$。**维度**：$T\times(1+K)$ 乘 $(1+K)\times N$，得 $T\times N$；取出第 $i$ 列，就是股票 $i$ 自己的回归。**数据矩阵**：$[\mathbf 1_T, \mathbf F]$ 由全部 $N$ 只股票共用，系数矩阵的每一列属于一只股票；最小二乘解的每一列都是 $([\mathbf 1_T, \mathbf F]'[\mathbf 1_T, \mathbf F])^{-1}[\mathbf 1_T, \mathbf F]'$ 乘以收益矩阵的对应列，其中 $(1+K)\times(1+K)$ 的逆矩阵也由全部股票共用。系数估计的同方差协方差矩阵为 $\hat\sigma_i^2([\mathbf 1_T, \mathbf F]'[\mathbf 1_T, \mathbf F])^{-1}$，与交互回归中的 $\sigma_\varepsilon^2(\mathbf X'\mathbf X)^{-1}$ 形式相同（[拆解](#ch0-unpack-xtx)）：矩阵各股票共用，残差方差 $\hat\sigma_i^2$ 每只股票各有一个。**特例**：$K = 1$ 时 $\mathbf F$ 只有一列 $(f_{1,1}, \dots, f_{1,T})'$，$[\mathbf 1_T, \mathbf F]'[\mathbf 1_T, \mathbf F] = \begin{pmatrix} T & \sum_t f_{1,t} \\ \sum_t f_{1,t} & \sum_t f_{1,t}^2 \end{pmatrix}$，它的逆矩阵的右下角元素等于 $1/\sum_t(f_{1,t} - \bar f_1)^2$（$\bar f_1$ 为样本均值），于是斜率的标准误为 $\hat\sigma_i/[\sum_t(f_{1,t} - \bar f_1)^2]^{1/2}$；$T$ 较大、且残差只含异质性冲击（$\hat\sigma_i \approx \sigma_\varepsilon$）时，$\sum_t(f_{1,t} - \bar f_1)^2 \approx T\sigma_1^2$，它约为 $\sigma_\varepsilon/(\sigma_1\sqrt T)$。有多个因子时，逆矩阵中对应第 $k$ 个因子的对角元精确地等于 $1/[\sum_t(f_{k,t} - \bar f_k)^2(1 - R_k^2)]$，$R_k^2$ 是样本中第 $k$ 个因子对其余因子回归的 $R^2$；样本中因子近似不相关时 $R_k^2 \approx 0$，对第 $k$ 个因子的载荷近似套用同一公式，得到下面“陷阱”中的 $\sigma_\varepsilon/(\sigma_k\sqrt T)$：$\sigma_\varepsilon$ 是异质性收益的月度标准差，$\sigma_k$ 是第 $k$ 个因子的月度标准差，$T$ 是月数，前两者之比没有单位，与载荷相同。默认参数下因子 2 的数字为 $0.10/(0.03 \times \sqrt{240}) \approx 0.22$。

```{code-cell} python
# 拆开讲斜率的标准误（K = 1）：([1_T, F]'[1_T, F])⁻¹ 的右下角元素等于 1 / Σ_t (f_{1,t} − f̄_1)²
f_ts6 = np.array([3.0, -1.0, 2.0, -4.0, 1.0, 5.0]) / 100                 # 0.2 节券商股例子的市场因子：(T,) = (6,)
D_ts6 = np.column_stack([np.ones(6), f_ts6])                              # [1_T, F]: (6, 2)
inv_ts6 = np.linalg.solve(D_ts6.T @ D_ts6, np.eye(2))                     # (2, 2)
assert np.isclose(inv_ts6[1, 1], 1 / np.sum((f_ts6 - f_ts6.mean()) ** 2))
se_rule = 0.10 / (0.03 * np.sqrt(240))                                    # σ_ε / (σ_k √T)，默认参数下的因子 2
print(f"σ_ε / (σ_k √T) = {se_rule:.3f}")
assert 0.21 < se_rule < 0.22
```

- **样本。** 一只股票的 $T$ 期观测，即图中的一列；$N$ 只股票各跑一次，共 $N$ 个回归。
- **得到。** 每只股票一个截距估计 $\hat a_i$（标量）与载荷估计 $\hat{\bv\beta}_i \in \Real^K$。
- **回答。** 这只股票对各个因子有多敏感？因子能否解释它的平均收益，即截距是否为 0？
- **前提。** 因子 $\mathbf f_t$ 事先已知、可以观测，回归只估计载荷。截距要解释为定价误差 α，还需要因子是可交易的超额收益：此时 $\bv\lambda = \E[\mathbf f_t]$，截距 $a_i = \E[r_{i,t}] - \bv\beta_i'\E[\mathbf f_t] = \E[r_{i,t}] - \bv\beta_i'\bv\lambda$ 正是 β 定价解释不了的那部分期望收益（0.2 节“风险溢价”与“定价误差”）。不可交易的因子（例如消费增长）$\bv\lambda$ 一般不等于 $\E[\mathbf f_t]$，截距就不是 α。
- **用途。** 估计载荷，它也是下文两步法的第一步；计算 Jensen 的 α [@jensen1968performance]；检验全部 $N$ 个截距同时为 0 的 GRS 检验 [@gibbons1989test]，见第 9 章。
- **陷阱。** 第一，$T$ 短时噪声大：单只股票的异质性收益很大，对第 $k$ 个因子的载荷，估计的标准误约为 $\sigma_\varepsilon / (\sigma_k\sqrt T)$ 的量级，默认参数（$\sigma_\varepsilon = 0.10$、$\sigma_2 = 0.03$、$T = 240$）下约为 0.2。第二，本章模型中载荷 $\bv\beta_{i,t}$ 时变，而回归只有一个常数斜率：它估计的是这只股票在样本期内的平均载荷，截距估计的是平均定价误差，看不到随特征变化的条件错误定价（[习题 %s](#ex-ch0-ts)）。

#### 横截面回归

对每一期 $t$，沿 $i = 1, \dots, N$ 回归，解释变量是股票层面的变量。常见的有两种：解释变量是载荷的估计值 $\hat{\bv\beta}_i$，或者是期初特征 $\tilde{\mathbf z}_{i,t-1}$：

```{math}
:enumerated: false
\underbrace{r_{i,t}}_{1\times 1} = \underbrace{\lambda_{0,t}}_{1\times 1} + \underbrace{\hat{\bv\beta}_i'}_{1\times K}\,\underbrace{\bv\lambda_t}_{K\times 1} + u_{i,t}, \qquad\qquad
\underbrace{r_{i,t}}_{1\times 1} = \underbrace{\tilde{\mathbf z}_{i,t-1}'}_{1\times(L+1)}\,\underbrace{\mathbf g_t}_{(L+1)\times 1} + u_{i,t}, \qquad i = 1, \dots, N .
```

第二种按行堆叠后的最小二乘解 $\hat{\mathbf g}_t = (\tilde{\mathbf Z}_{t-1}'\tilde{\mathbf Z}_{t-1})^{-1}\tilde{\mathbf Z}_{t-1}'\mathbf r_t$ 已在[因子、特征、载荷的关系](#sec-ch0-fcl)中给出；第一种同理，只是把 $\tilde{\mathbf Z}_{t-1}$ 换成 $[\mathbf 1_N, \hat{\mathbf B}]$，其中 $\hat{\mathbf B}$（$N\times K$）的第 $i$ 行为 $\hat{\bv\beta}_i'$。

(primer-unpack-cs)=
**拆开讲两种横截面回归。** 第二种（对特征回归）就是 0.2 节的 Barra 式回归：$\mathbf g_t$ 的分量与含义见[拆解](#ch0-unpack-g)，最小二乘解 $\hat{\mathbf g}_t$ 中各矩阵乘积的元素、特例与设计矩阵 $\tilde{\mathbf Z}_{t-1}$（$N\times(L+1)$）见[拆解](#ch0-unpack-ghat)。下面拆第一种，即对载荷估计值的回归，并说明它与第二种的不同。写出分量，

```{math}
:enumerated: false
\hat{\bv\beta}_i = \begin{pmatrix} \hat\beta_{1,i} \\ \hat\beta_{2,i} \\ \vdots \\ \hat\beta_{K,i} \end{pmatrix} , \qquad
\bv\lambda_t = \begin{pmatrix} \lambda_{1,t} \\ \lambda_{2,t} \\ \vdots \\ \lambda_{K,t} \end{pmatrix} , \qquad
r_{i,t} = \lambda_{0,t} + \hat\beta_{1,i}\lambda_{1,t} + \hat\beta_{2,i}\lambda_{2,t} + \dots + \hat\beta_{K,i}\lambda_{K,t} + u_{i,t} .
```

- $\hat\beta_{k,i}$：股票 $i$ 对第 $k$ 个因子的载荷估计值，由时间序列回归得到（例如对市场因子 MKT 的 β），没有单位
- $\lambda_{k,t}$：第 $t$ 期每单位第 $k$ 个因子载荷得到的补偿（实现值），单位与收益相同；例如 $\lambda_{1,t} = 2\%$ 表示这个月市场 β 高 1 的股票，收益平均高 2%
- $\lambda_{0,t}$：截距，即载荷全为 0 的（假想）股票在第 $t$ 期的拟合收益，是标量

**维度**：$\hat{\bv\beta}_i$ 与 $\bv\lambda_t$ 都是 $K\times 1$ 的列向量，$\hat{\bv\beta}_i'\bv\lambda_t$ 是 $1\times K$ 乘 $K\times 1$，得一个数；每期的系数共 $1+K$ 个，而对特征回归时是 $L+1$ 个。**展开运算**见上面的第三个式子，它与 $\bv\beta_i'\mathbf f_t$ 的展开形式相同（[拆解](#ch0-unpack-betaf)），区别在于角色对调：这里载荷是已知的解释变量（虽然是估计值），每期的“因子收益”$\bv\lambda_t$ 才是待估的系数。**特例**：$K = 1$ 且解释变量是市场 β 时，$r_{i,t} = \lambda_{0,t} + \hat\beta_{1,i}\lambda_{1,t} + u_{i,t}$ 是逐期估计的证券市场线（$r_{i,t}$ 须是超额收益，截距才能与 0 比较），斜率 $\lambda_{1,t}$ 是这一期每单位市场 β 的补偿；CAPM 成立、且 $\hat\beta_{1,i}$ 的估计误差可以忽略时，斜率的时间平均估计市场风险溢价，截距的时间平均应为 0；否则斜率偏向 0、截距偏正（风险溢价与平均 β 都为正时），见下文 Fama–MacBeth 程序中“两步法”的变量误差。另一个特例是用真实载荷、不加截距：$\bv\theta = \mathbf 0$ 时 $\mathbf r_t = \mathbf B_t\mathbf f_t + \bv\varepsilon_t$（[拆解](#ch0-unpack-return)），于是

```{math}
:enumerated: false
\underbrace{\hat{\bv\lambda}_t}_{K\times 1} = (\mathbf B_t'\mathbf B_t)^{-1}\mathbf B_t'\mathbf r_t = \mathbf f_t + \underbrace{(\mathbf B_t'\mathbf B_t)^{-1}}_{K\times K}\,\underbrace{\mathbf B_t'}_{K\times N}\,\underbrace{\bv\varepsilon_t}_{N\times 1} ,
```

即当期实现的因子收益加上横截面噪声。它与 $\hat{\mathbf g}_t = \mathbf g_t + (\tilde{\mathbf Z}_{t-1}'\tilde{\mathbf Z}_{t-1})^{-1}\tilde{\mathbf Z}_{t-1}'\bv\varepsilon_t$ 结构相同，只是把 $N\times(L+1)$ 的 $\tilde{\mathbf Z}_{t-1}$ 换成 $N\times K$ 的 $\mathbf B_t$；$\mathbf B_t'\mathbf B_t$ 的 $(k, l)$ 元素是 $\sum_i\beta_{k,i,t}\beta_{l,i,t}$。下面“两种解释变量下系数的含义”与下一页两步法的对照组用的都是这个式子；那里的 $\bv\Gamma\mathbf f_t + (0, \bv\theta')'$ 见[拆解](#ch0-unpack-g)，$\bv\Gamma\bv\mu$ 见[拆解](#ch0-unpack-gmu)。**数据矩阵**：

```{math}
:enumerated: false
\underbrace{[\mathbf 1_N, \hat{\mathbf B}]}_{N\times(1+K)} = \begin{pmatrix} 1 & \hat\beta_{1,1} & \cdots & \hat\beta_{K,1} \\ 1 & \hat\beta_{1,2} & \cdots & \hat\beta_{K,2} \\ \vdots & \vdots & & \vdots \\ 1 & \hat\beta_{1,N} & \cdots & \hat\beta_{K,N} \end{pmatrix} , \qquad
\underbrace{\begin{pmatrix} \hat\lambda_{0,t} \\ \hat{\bv\lambda}_t \end{pmatrix}}_{(1+K)\times 1} = \bigl([\mathbf 1_N, \hat{\mathbf B}]'[\mathbf 1_N, \hat{\mathbf B}]\bigr)^{-1}[\mathbf 1_N, \hat{\mathbf B}]'\,\underbrace{\mathbf r_t}_{N\times 1} .
```

$\hat{\mathbf B}$ 的第 $i$ 行是股票 $i$ 的 $\hat{\bv\beta}_i'$，第 $k$ 列是全部股票对第 $k$ 个因子的载荷估计，结构与 0.2 节的 $\mathbf B$ 相同（[拆解](#ch0-unpack-betaf)）。两种横截面回归的区别在设计矩阵：对特征回归时，每期有自己的 $\tilde{\mathbf Z}_{t-1}$（特征每期都在变）；对载荷估计值回归时，$\hat{\mathbf B}$ 如果用一段样本估计一次，**各期共用同一个 $N\times(1+K)$ 的设计矩阵**，每期只换左边的 $\mathbf r_t$（用滚动窗口逐期重估时，$\hat{\mathbf B}$ 也逐期变化）。两种回归都是同一期的全部股票共用一个系数向量，$(\hat\lambda_{0,t}, \hat{\bv\lambda}_t')'$ 或 $\hat{\mathbf g}_t$，每期各估一个。与时间序列回归相比，“共用什么”正好对调（[拆解](#ch0-unpack-ghat)）。设计矩阵各期不变时，下文混合回归的[结论 1](#primer-regs-conclusions) 适用：Fama–MacBeth、混合回归与“把各股票的平均收益 $\bar r_i$ 对 $[\mathbf 1_N, \hat{\mathbf B}]$ 回归一次”三者的点估计完全相同。

- **样本。** 同一期的 $N$ 只股票，即图中的一行；每期一个回归，共 $T$ 个。
- **得到。** 每期一组系数：$\hat\lambda_{0,t}$ 与 $\hat{\bv\lambda}_t \in \Real^K$，或 $\hat{\mathbf g}_t \in \Real^{L+1}$。$T$ 期合在一起，是一条系数的时间序列。
- **回答。** 这一期，载荷（或特征）高一个单位的股票，收益多了多少？
- **两种解释变量下系数的含义。** 对载荷回归时，$\hat{\bv\lambda}_t$ 是该期每单位载荷的**实现补偿**。用真实载荷看最清楚：$\bv\theta = \mathbf 0$ 时 $r_{i,t} = \bv\beta_{i,t}'\mathbf f_t + \varepsilon_{i,t}$，收益对 $\bv\beta_{i,t}$ 做不加截距的横截面回归，系数就是 $\mathbf f_t = \bv\mu + (\mathbf f_t - \bv\mu)$ 加上横截面噪声，即风险溢价 $\bv\lambda = \bv\mu$ 加上当期的意外。单期的实现补偿可正可负，它的时间平均才估计 $\bv\lambda$。对特征回归时，就是 Barra 式回归：$\hat{\mathbf g}_t$ 估计的是 $\bv\Gamma\mathbf f_t + (0, \bv\theta')'$，即经过特征到载荷的映射换算之后的因子收益，并且混有定价误差（[因子、特征、载荷的关系](#sec-ch0-fcl)）；它的时间平均估计 $\bv\Gamma\bv\mu + (0, \bv\theta')'$。
- **前提。** 解释变量在期初已知（0.3 节的时点约定）；解释变量在股票之间要有差别。例如本模拟器中所有股票对因子 1 的真实载荷都是 1，回归中若同时有截距，因子 1 的补偿就无法与截距分开。对载荷回归时，载荷要先估计出来，估计误差会带进这一步。
- **用途。** Barra 风险模型的因子收益；Fama–MacBeth 程序的第一步；业界的 [IC](#glossary-ic) 是一元横截面回归斜率的标准化版本。
- **陷阱。** 单期噪声大。噪声有两个来源：一是横截面噪声，量级为 $\sigma_\varepsilon/\sqrt N$，默认参数下约 0.45%（[因子、特征、载荷的关系](#sec-ch0-fcl)）；二是系数本身就包含当期因子的意外 $\mathbf f_t - \bv\mu$，而且通常是更大的一项。所以一期的系数几乎说明不了平均补偿。

#### Fama–MacBeth 程序

Fama–MacBeth 是一套程序，而不是一种新的回归 [@fama1973risk]。它分三步：

1. 每期做一次横截面回归，得到 $\hat{\mathbf g}_1, \dots, \hat{\mathbf g}_T$；
2. 点估计取它们的时间平均 $\underbrace{\bar{\mathbf g}}_{(L+1)\times 1} = T^{-1}\sum_{t=1}^{T}\hat{\mathbf g}_t$；
3. 标准误逐分量计算：$\operatorname{se}(\bar g_j) = \hat\sigma(\hat g_{j,t}) / \sqrt T$，其中 $\hat\sigma(\hat g_{j,t}) = \bigl[(T-1)^{-1}\sum_t(\hat g_{j,t} - \bar g_j)^2\bigr]^{1/2}$ 是第 $j$ 个系数的时间序列标准差；$t$ 值为 $\bar g_j / \operatorname{se}(\bar g_j)$。

对载荷回归时同理，把 $\hat{\mathbf g}_t$ 换成 $\hat{\bv\lambda}_t$，时间平均记作 $\bar{\bv\lambda} = T^{-1}\sum_t\hat{\bv\lambda}_t$。

(primer-unpack-fm)=
**拆开讲 Fama–MacBeth 的点估计与标准误。** 写出分量，

```{math}
:enumerated: false
\bar{\mathbf g} = \begin{pmatrix} \bar g_0 \\ \bar g_1 \\ \vdots \\ \bar g_L \end{pmatrix} = \frac{1}{T}\sum_{t=1}^{T}\begin{pmatrix} \hat g_{0,t} \\ \hat g_{1,t} \\ \vdots \\ \hat g_{L,t} \end{pmatrix} , \qquad
\bar g_j = \frac{\hat g_{j,1} + \hat g_{j,2} + \dots + \hat g_{j,T}}{T} .
```

- $\bar g_0$：全部特征处于平均水平的股票，平均每期的收益
- $\bar g_j$（$j \geq 1$）：特征 $j$ 每高 1 个标准差，股票收益平均每期高出的部分，即 $T$ 期“特征 $j$ 的回报”$\hat g_{j,t}$ 的平均（$\hat g_{j,t}$ 的含义见[拆解](#ch0-unpack-g)）

各分量的单位都与收益相同。**维度**：$T$ 个 $(L+1)\times 1$ 的向量相加再除以 $T$，仍是 $(L+1)\times 1$；对载荷回归时 $\bar{\bv\lambda} = (\bar\lambda_1, \dots, \bar\lambda_K)'$ 是 $K\times 1$。标准误与 $t$ 值是逐分量计算的标量，不需要矩阵求逆。**展开运算**，以一个数值例子说明（假设数字）：$T = 5$ 期中 $z_1$ 的系数 $\hat g_{1,t}$ 依次为 3%、$-1\%$、2%、0、1%，则 $\bar g_1 = 5\%/5 = 1\%$；离差为 2%、$-2\%$、1%、$-1\%$、0，平方和为 $10$（单位为百分点的平方），除以 $T - 1 = 4$ 得样本方差 2.5，样本标准差约 1.58%；于是 $\operatorname{se}(\bar g_1) = 1.58\%/\sqrt 5 \approx 0.71\%$，$t$ 值约为 $1/0.71 \approx 1.41$，在 5% 的水平上不显著。单期系数在 $-1\%$～3% 之间波动，标准误的大小正是由这种逐期波动决定的。**特例**：$L = 0$（只有常数项）时，$\hat g_{0,t}$ 是第 $t$ 期全部股票收益的横截面平均（[拆解](#ch0-unpack-ghat)），即等权组合的收益；$\bar g_0$ 就是等权组合收益的时间平均，FM 标准误就是这条收益序列的均值标准误，FM 的 $t$ 检验退化为“等权组合的平均收益是否为 0”的普通 $t$ 检验。$L = 1$ 且特征每期精确标准化（横截面均值 0、方差 1，方差除以 $N$）时，$\tilde{\mathbf Z}_{t-1}'\tilde{\mathbf Z}_{t-1} = N\mathbf I_2$，$\hat g_{1,t} = N^{-1}\sum_i z_{1,i,t-1}r_{i,t}$ 等于权重为 $z_{1,i,t-1}/N$ 的零成本多空组合的收益（[拆解](#ch0-unpack-ghat)），$\bar g_1$ 就等于这个多空组合的平均收益。**数据矩阵**：把 $T$ 期的系数按行堆叠成 $T\times(L+1)$ 的矩阵，第 $t$ 行为 $\hat{\mathbf g}_t'$，第 $j$ 列为第 $j$ 个系数的时间序列（下一页代码中的 `G_rg`，形状 `(T, 3)`）；$\bar{\mathbf g}$ 是各列的均值，$\operatorname{se}(\bar g_j)$ 是第 $j$ 列的样本标准差除以 $\sqrt T$（`G_rg.std(axis=0, ddof=1) / np.sqrt(T_rg)`）。每期的设计矩阵 $\tilde{\mathbf Z}_{t-1}$ 与系数 $\hat{\mathbf g}_t$ 各不相同；程序最后只得到一个 $\bar{\mathbf g}$，由全部股票、全部时期共用，而标准误只用到 $T$ 个观测（各期的系数），不是 $NT$ 个。下面一条中的 $\tilde{\mathbf z}_{i,t-1}'\bv\Gamma(\mathbf f_t - \bv\mu)$ 与 $\mathbf g_t$ 的因子部分同形（[拆解](#ch0-unpack-g)）：$(L+1)\times 1$ 的 $\bv\Gamma(\mathbf f_t - \bv\mu)$ 是第 $t$ 期所有股票共同的冲击，每只股票按自己的特征 $\tilde{\mathbf z}_{i,t-1}$ 承受它；$\tilde{\mathbf z}_{i,t-1}'\bv\Gamma\bv\mu$ 中 $\bv\Gamma\bv\mu$ 的分量见[拆解](#ch0-unpack-gmu)。

```{code-cell} python
# 拆开讲 FM 的数值例子（假设数字）：T = 5 期的 ĝ_{1,t}
g1_ex = np.array([3.0, -1.0, 2.0, 0.0, 1.0]) / 100                       # (T,) = (5,)
g1_bar = g1_ex.mean()                                                     # ḡ_1
se_g1 = g1_ex.std(ddof=1) / np.sqrt(len(g1_ex))                           # σ̂(ĝ_{1,t}) / √T
print(f"ḡ_1 = {g1_bar:.2%}，标准误 {se_g1:.2%}，t 值 {g1_bar / se_g1:.2f}")
assert np.isclose(g1_bar, 0.01) and np.isclose(g1_ex.var(ddof=1), 2.5e-4)
assert np.isclose(se_g1, np.sqrt(0.5) / 100) and np.isclose(g1_bar / se_g1, np.sqrt(2))
```

- **样本与得到。** 全部 $T$ 期的横截面回归结果；得到 $\bar{\mathbf g}$ 及其标准误。
- **回答。** 平均而言，载荷（或特征）高一个单位的股票多赚多少？这个平均补偿是否显著不为 0？
- **为什么它自动考虑了残差在横截面上的相关。** 同一期的所有股票都受同一个 $\mathbf f_t$ 冲击。以特征为解释变量时，$r_{i,t}$ 偏离平均关系 $\tilde{\mathbf z}_{i,t-1}'\bv\Gamma\bv\mu$ 的部分是 $\tilde{\mathbf z}_{i,t-1}'\bv\Gamma(\mathbf f_t - \bv\mu) + \varepsilon_{i,t}$，其中第一项在同一期的股票之间相关。Fama–MacBeth 先把每一期的 $N$ 只股票压缩成一个系数 $\hat{\mathbf g}_t$，当期的共同冲击 $\bv\Gamma(\mathbf f_t - \bv\mu)$ 全部体现在 $\hat{\mathbf g}_t$ 之中，于是 $T$ 个 $\hat{\mathbf g}_t$ 的波动自然包含了它。同期股票之间无论怎样相关，在这里都只算一个观测；程序只要求不同期的 $\hat{\mathbf g}_t$ 近似独立。
- **前提。** 各期的 $\hat{\mathbf g}_t$ 近似独立、同分布；每期的横截面回归可解（股票数多于解释变量个数，解释变量不完全共线）。
- **两步法。** 解释变量是载荷时，载荷本身未知，于是先做时间序列回归、再做横截面回归：第一步对每只股票估计 $\hat{\bv\beta}_i$；第二步逐期把 $\mathbf r_t$ 对 $\hat{\bv\beta}_i$ 做横截面回归，得到 $\hat{\bv\lambda}_t$，再按上面的三步取平均。问题在于第二步的解释变量 $\hat{\bv\beta}_i$ 是带误差的估计值，即变量误差（errors-in-variables，EIV）。它有两个后果：斜率通常偏向 0（衰减偏误）；Fama–MacBeth 标准误把 $\hat{\bv\beta}_i$ 当作已知，没有计入第一步的估计误差，Shanken 修正主要针对后者 [@shanken1992estimation]。Fama 与 MacBeth 的原文用按事先估计的 β 分组构造的组合代替个股，以减小载荷的估计误差。这些都在第 10 章展开。
- **用途。** 估计风险溢价 $\bv\lambda$（第 10 章）；检验特征在控制其他特征后能否预测收益；业界单因子测试中的“回归法”（见下一页的[“与业界的对应”](#sec-ch0-regs-industry)）。
- **陷阱。** 两步法的变量误差；$\hat{\mathbf g}_t$ 在时间上相关时（例如用月度数据计算未来 12 个月的收益，相邻各期的收益期限重叠），标准误会偏小，需要 Newey–West 一类的调整，见第 12 章。

#### 混合回归

把全部 $(i, t)$ 放在一起，只估计一个不随时间变化的系数向量 $\mathbf g \in \Real^{L+1}$：

```{math}
:enumerated: false
\underbrace{r_{i,t}}_{1\times 1} = \underbrace{\tilde{\mathbf z}_{i,t-1}'}_{1\times(L+1)}\,\underbrace{\mathbf g}_{(L+1)\times 1} + u_{i,t}, \qquad
\underbrace{\hat{\mathbf g}^{\mathrm P}}_{(L+1)\times 1} = \Bigl(\sum_{t=1}^{T}\underbrace{\tilde{\mathbf Z}_{t-1}'\tilde{\mathbf Z}_{t-1}}_{(L+1)\times(L+1)}\Bigr)^{-1}\sum_{t=1}^{T}\underbrace{\tilde{\mathbf Z}_{t-1}'}_{(L+1)\times N}\,\underbrace{\mathbf r_t}_{N\times 1} .
```

上标 P 表示 pooled。

(primer-unpack-pool)=
**拆开讲混合回归。** 第一个式子与第二种横截面回归形式相同，$\tilde{\mathbf z}_{i,t-1}'\mathbf g$ 的逐项展开见[拆解](#ch0-unpack-g)；区别只在系数：横截面回归每期一个 $\mathbf g_t$，这里全部 $T$ 期共用一个 $\mathbf g$，不带下标 $t$。第二个式子来自把全部 $(i, t)$ 叠成一个回归。按“期在外层、股票在内层”的顺序堆叠（与面板按 `(period, stock)` 的排序、下一页代码中的 `reshape(T_rg * N_rg, 3)` 一致），

```{math}
:enumerated: false
\underbrace{\begin{pmatrix} \mathbf r_1 \\ \mathbf r_2 \\ \vdots \\ \mathbf r_T \end{pmatrix}}_{TN\times 1}
= \underbrace{\begin{pmatrix} \tilde{\mathbf Z}_0 \\ \tilde{\mathbf Z}_1 \\ \vdots \\ \tilde{\mathbf Z}_{T-1} \end{pmatrix}}_{TN\times(L+1)}\,\underbrace{\mathbf g}_{(L+1)\times 1}
+ \underbrace{\begin{pmatrix} u_{1,1} \\ \vdots \\ u_{N,1} \\ \vdots \\ u_{N,T} \end{pmatrix}}_{TN\times 1} .
```

左边的长向量由 $T$ 个 $N\times 1$ 的 $\mathbf r_t$ 依次叠成：前 $N$ 个元素是第 1 期的 $N$ 只股票，接下来 $N$ 个是第 2 期，依此类推。设计矩阵由 $T$ 块 $N\times(L+1)$ 的 $\tilde{\mathbf Z}_{t-1}$ 竖着叠成（每块的结构见[拆解](#ch0-unpack-return)），每一行是一个观测 $(i, t)$ 的 $\tilde{\mathbf z}_{i,t-1}'$。按分块矩阵的乘法，

```{math}
:enumerated: false
\begin{pmatrix} \tilde{\mathbf Z}_0' & \cdots & \tilde{\mathbf Z}_{T-1}' \end{pmatrix}\begin{pmatrix} \tilde{\mathbf Z}_0 \\ \vdots \\ \tilde{\mathbf Z}_{T-1} \end{pmatrix} = \sum_{t=1}^{T}\tilde{\mathbf Z}_{t-1}'\tilde{\mathbf Z}_{t-1} , \qquad
\begin{pmatrix} \tilde{\mathbf Z}_0' & \cdots & \tilde{\mathbf Z}_{T-1}' \end{pmatrix}\begin{pmatrix} \mathbf r_1 \\ \vdots \\ \mathbf r_T \end{pmatrix} = \sum_{t=1}^{T}\tilde{\mathbf Z}_{t-1}'\mathbf r_t ,
```

代入最小二乘公式 $(\mathbf X'\mathbf X)^{-1}\mathbf X'\mathbf y$，就得到上面的 $\hat{\mathbf g}^{\mathrm P}$。**维度**：$(L+1)\times TN$ 乘 $TN\times(L+1)$，得 $(L+1)\times(L+1)$，其 $(j, l)$ 元素为 $\sum_t\sum_i\tilde z_{j,i,t-1}\tilde z_{l,i,t-1}$，即把[拆解](#ch0-unpack-ghat)中单期的求和再对 $T$ 期求和；$\sum_t\tilde{\mathbf Z}_{t-1}'\mathbf r_t$ 是 $(L+1)\times 1$，第 $j$ 个元素为 $\sum_t\sum_i\tilde z_{j,i,t-1}r_{i,t}$。这与交互回归的 $\mathbf X'\mathbf X$ 是同一种“对全部观测求和”的结构（[拆解](#ch0-unpack-xtx)），只是每个观测的解释变量从 $(L+1)K$ 个交互项换成了 $L+1$ 个特征。**特例**：$L = 0$ 时设计矩阵是 $TN\times 1$ 的全 1 向量，$\hat g^{\mathrm P}_0 = (NT)^{-1}\sum_t\sum_i r_{i,t}$ 是全部格子的总平均；每期股票数都是 $N$ 时，它等于各期横截面平均的时间平均，即 Fama–MacBeth 的 $\bar g_0$；股票数逐期不同时二者不再相等，见下文的 $\mathbf W_t$。$L = 1$ 时，斜率是 $z_1$ 与收益在全部 $NT$ 个格子上的样本协方差除以 $z_1$ 的样本方差。**数据矩阵**：设计矩阵是上面的 $TN\times(L+1)$ 矩阵，下一页代码中的 `Xp_rg` 是它的前三列（形状 `(TN, 3)`），`yp_rg` 是左边的 $TN\times 1$ 向量。**全部股票、全部时期共用一个系数向量 $\hat{\mathbf g}^{\mathrm P}$**；对比之下，横截面回归每期一个 $\hat{\mathbf g}_t$，时间序列回归每只股票一套 $(\hat a_i, \hat{\bv\beta}_i')$。下文同方差标准误 $\hat\sigma^2(\sum_t\tilde{\mathbf Z}_{t-1}'\tilde{\mathbf Z}_{t-1})^{-1}$ 的含义与交互回归中的 $\sigma_\varepsilon^2(\mathbf X'\mathbf X)^{-1}$ 相同（[拆解](#ch0-unpack-xtx)）：$\hat\sigma^2$ 是由 $NT$ 个残差估计的残差方差，整个 $(L+1)\times(L+1)$ 矩阵的第 $j$ 个对角元开平方，就是第 $j$ 个系数的同方差标准误。

- **样本与得到。** 全部 $NT$ 个格子，即整张表；得到一组系数 $\hat{\mathbf g}^{\mathrm P}$ 及其标准误。
- **回答。** 与 Fama–MacBeth 相同：平均而言，特征与收益的关系如何。
- **前提。** 系数不随时间变化，或者只关心它的平均；用同方差标准误时，还要求 $NT$ 个残差相互独立、方差相同。
- **用途。** 误区四与[习题 %s](#ex-ch0-beta) 中估计 $\bv\Gamma\bv\mu$；交互回归估计 $\bv\Gamma$；第四部分的收益预测模型，通常也是在全部股票与期数组成的混合面板上训练的。

**点估计与 Fama–MacBeth 的关系。** 第 $t$ 期横截面回归的正规方程是 $\tilde{\mathbf Z}_{t-1}'\mathbf r_t = \tilde{\mathbf Z}_{t-1}'\tilde{\mathbf Z}_{t-1}\hat{\mathbf g}_t$（假定每期的 $\tilde{\mathbf Z}_{t-1}$ 列满秩）。代入上式，

```{math}
:enumerated: false
\hat{\mathbf g}^{\mathrm P} = \sum_{t=1}^{T}\underbrace{\mathbf W_t}_{(L+1)\times(L+1)}\,\underbrace{\hat{\mathbf g}_t}_{(L+1)\times 1}, \qquad
\mathbf W_t = \Bigl(\sum_{s=1}^{T}\tilde{\mathbf Z}_{s-1}'\tilde{\mathbf Z}_{s-1}\Bigr)^{-1}\tilde{\mathbf Z}_{t-1}'\tilde{\mathbf Z}_{t-1}, \qquad \sum_{t=1}^{T}\mathbf W_t = \mathbf I_{L+1} .
```

(primer-unpack-w)=
**拆开讲 $\mathbf W_t$。** 推导只有一步：把正规方程代入 $\hat{\mathbf g}^{\mathrm P}$ 中的第二个求和，

```{math}
:enumerated: false
\hat{\mathbf g}^{\mathrm P} = \Bigl(\sum_{s=1}^{T}\tilde{\mathbf Z}_{s-1}'\tilde{\mathbf Z}_{s-1}\Bigr)^{-1}\sum_{t=1}^{T}\tilde{\mathbf Z}_{t-1}'\tilde{\mathbf Z}_{t-1}\,\hat{\mathbf g}_t
= \sum_{t=1}^{T}\underbrace{\Bigl(\sum_{s=1}^{T}\tilde{\mathbf Z}_{s-1}'\tilde{\mathbf Z}_{s-1}\Bigr)^{-1}\tilde{\mathbf Z}_{t-1}'\tilde{\mathbf Z}_{t-1}}_{\mathbf W_t}\,\hat{\mathbf g}_t .
```

求逆的那个矩阵不依赖 $t$，可以移进求和号；它的求和下标写成 $s$，是为了与外层的 $t$ 区分。**维度**：$\mathbf W_t$ 是两个 $(L+1)\times(L+1)$ 矩阵之积，仍为 $(L+1)\times(L+1)$；乘 $(L+1)\times 1$ 的 $\hat{\mathbf g}_t$，得 $(L+1)\times 1$。$\sum_t\mathbf W_t = (\sum_s\tilde{\mathbf Z}_{s-1}'\tilde{\mathbf Z}_{s-1})^{-1}\sum_t\tilde{\mathbf Z}_{t-1}'\tilde{\mathbf Z}_{t-1} = \mathbf I_{L+1}$，这是“权重之和为 1”的矩阵版本。白话：$\tilde{\mathbf Z}_{t-1}'\tilde{\mathbf Z}_{t-1}$ 度量第 $t$ 期的横截面含有多少关于系数的信息，股票越多、特征越分散，它越“大”，这一期系数估计的条件方差 $\sigma_\varepsilon^2(\tilde{\mathbf Z}_{t-1}'\tilde{\mathbf Z}_{t-1})^{-1}$ 就越小（[拆解](#ch0-unpack-ghat)）；$\mathbf W_t$ 是第 $t$ 期的信息占全部 $T$ 期信息的份额。$\mathbf W_t$ 一般不是对角矩阵（两个对称矩阵之积一般也不对称），所以 $\hat{\mathbf g}^{\mathrm P}$ 的第 $j$ 个分量不只是各期 $\hat g_{j,t}$ 的加权平均，还会混入其他分量 $\hat g_{l,t}$。**特例**：$L = 0$ 时 $\tilde{\mathbf Z}_{t-1}$ 是全 1 的列向量，$\tilde{\mathbf Z}_{t-1}'\tilde{\mathbf Z}_{t-1}$ 与 $\mathbf W_t$ 都退化为标量，前者等于第 $t$ 期的股票数，记作 $N_t$（本书的模拟中各期都是 $N$），于是 $W_t = N_t/\sum_s N_s$，权重正比于股票数。例如只有两期，第 1 期 100 只股票、横截面平均收益 2%，第 2 期 300 只股票、平均收益 0（假设数字）：$W_1 = 0.25$、$W_2 = 0.75$，混合回归得 $0.25 \times 2\% + 0.75 \times 0 = 0.5\%$，Fama–MacBeth 的等权平均得 1%。只有一个特征且每期标准化时，$\tilde{\mathbf Z}_{t-1}'\tilde{\mathbf Z}_{t-1}$ 就是[拆解](#ch0-unpack-ghat)中写出的 $2\times 2$ 矩阵，其中 $\sum_i z_{1,i,t-1} = 0$、$\sum_i z_{1,i,t-1}^2 = N$，所以它等于 $N\mathbf I_2$，这就是下面结论 1 的例子。**数据矩阵**：$\mathbf W_t$ 只依赖解释变量 $\tilde{\mathbf Z}_0, \dots, \tilde{\mathbf Z}_{T-1}$，与收益无关；每期一个，$T$ 个叠成形状为 `(T, L+1, L+1)` 的数组。下一页的代码先用 `np.einsum` 算出逐期的 $\tilde{\mathbf Z}_{t-1}'\tilde{\mathbf Z}_{t-1}$（`XtX_rg`，形状 `(T, 3, 3)`），再左乘它们总和的逆来核对这个恒等式。

```{code-cell} python
# 拆开讲 W_t 的特例（假设数字）：L = 0，第 1 期 100 只股票平均收益 2%，第 2 期 300 只平均收益 0
r_two = [np.full(100, 0.02), np.zeros(300)]                               # 两期的 r_t，长度 N_t 不同
g_pool = np.linalg.lstsq(np.ones((400, 1)), np.concatenate(r_two), rcond=None)[0][0]   # (400, 1) 的全 1 设计矩阵
g_fm = np.mean([r.mean() for r in r_two])                                 # 各期横截面平均的等权平均
W_two = np.array([len(r) for r in r_two]) / 400                           # W_t = N_t / Σ_s N_s
print(f"W_t = {W_two}，混合回归 {g_pool:.2%}，FM {g_fm:.2%}")
assert np.allclose(W_two, [0.25, 0.75]) and np.isclose(g_pool, 0.005) and np.isclose(g_fm, 0.01)
assert np.isclose(g_pool, W_two @ np.array([0.02, 0.0]))                  # ĝ^P = Σ_t W_t ĝ_t
```

(primer-regs-conclusions)=
也就是说，混合回归的点估计是各期横截面估计的**矩阵加权平均**，而 Fama–MacBeth 是等权平均（$\mathbf W_t = T^{-1}\mathbf I_{L+1}$）。由此可以得到三条有把握的结论：

1. 如果每期的 $\tilde{\mathbf Z}_{t-1}'\tilde{\mathbf Z}_{t-1}$ 都相同，则 $\mathbf W_t = T^{-1}\mathbf I_{L+1}$，二者**完全相等**。例如每期股票数相同、只有一个特征、每期把它标准化为均值 0、方差 1（方差除以 $N$），此时 $\tilde{\mathbf Z}_{t-1}'\tilde{\mathbf Z}_{t-1} = N\mathbf I_2$。
2. 有多个特征时，每期标准化只固定了 $\tilde{\mathbf Z}_{t-1}'\tilde{\mathbf Z}_{t-1}$ 的对角元，特征之间的横截面相关系数仍逐期变化，二者一般不同；这些相关系数各期都接近 0 时（如本模拟器），二者接近。
3. 每期股票数不同、或特征的横截面离散度随时间变化时，二者一般不同：混合回归给股票多、离散度大的期更大的权重。

**陷阱：同方差标准误。** 混合回归的同方差标准误 $\hat\sigma^2(\sum_t\tilde{\mathbf Z}_{t-1}'\tilde{\mathbf Z}_{t-1})^{-1}$ 把 $NT$ 个观测当作相互独立。上面说过，残差中含有 $\tilde{\mathbf z}_{i,t-1}'\bv\Gamma(\mathbf f_t - \bv\mu)$ 这样同一期内所有股票共有的成分，独立性不成立，标准误因而严重偏小，$t$ 值严重偏大（误区四）。这不是混合回归本身的问题，而是残差结构的问题：交互回归把 $\mathbf f_t$ 放进了解释变量，残差只剩 $\varepsilon_{i,t}$，同方差标准误就是对的，第 0 章“模拟实验”一节（[用交互回归恢复 Γ](#sec-ch0-interact)）的重复模拟验证了这一点。允许同一期残差任意相关的“按时间聚类”（clustered by time）标准误，见第 12 章。

#### 对比总表

| 回归 | 怎么跑 | 解释变量 | 得到什么 | 回答什么 | 前提 | 典型用途 | 主要陷阱 |
|---|---|---|---|---|---|---|---|
| 时间序列回归 | 每只股票沿时间跑一次，共 $N$ 个 | 因子 $\mathbf f_t$ | 每只股票的 $\hat a_i$ 与 $\hat{\bv\beta}_i$ | 对因子有多敏感；截距是否为 0 | 因子已知；截距作 α 需因子可交易 | 估计载荷；Jensen α；GRS 检验 | $T$ 短时噪声大；载荷时变时只得平均载荷 |
| 横截面回归 | 每期沿股票跑一次，共 $T$ 个 | 载荷估计值 $\hat{\bv\beta}_i$ 或特征 $\tilde{\mathbf z}_{i,t-1}$ | 每期一组系数 $\hat{\bv\lambda}_t$ 或 $\hat{\mathbf g}_t$ | 本期载荷或特征高的股票多赚多少 | 解释变量期初已知且在股票间有差别 | Barra 因子收益；IC | 单期系数含因子意外，噪声大 |
| Fama–MacBeth | 逐期横截面回归，再对系数求时间平均 | 同横截面回归 | $\bar{\mathbf g}$ 或 $\bar{\bv\lambda}$ 及其时间序列标准误 | 平均补偿是否不为 0 | 各期系数近似独立 | 估计风险溢价；单因子测试的回归法 | 两步法的变量误差；系数自相关 |
| 混合回归 | 全部 $NT$ 个观测跑一次 | 特征，可含特征与因子的交互项 | 一组系数 $\hat{\mathbf g}^{\mathrm P}$ | 平均而言特征与收益的关系 | 用同方差标准误时残差须独立 | 误区四；交互回归；收益预测模型 | 残差同期相关时同方差标准误严重偏小 |
