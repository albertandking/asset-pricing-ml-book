---
short_title: 0.2（续）特征、定价误差与 SDF
kernelspec:
  name: python3
  display_name: Python 3
numbering:
  enumerator: '0.%s'
  exercise:
    template: 习题 %s
---

(sec-ch0-concepts-sdf)=
# 0.2 基本概念速览（续）：特征、定价误差与随机贴现因子

**本页准备。** 本书每个页面的代码在独立的内核中运行，前面页面定义的变量在这里不可用。本页接着[上一页](#sec-ch0-concepts)，介绍特征、定价误差与随机贴现因子三个概念，介绍的顺序与上一页相同：先白话，再例子，最后公式。下面的单元导入本页用到的模块，并重建上一页“载荷”与“风险溢价”两节中四只股票的载荷矩阵 `B_ex` 与风险溢价 `lam`，数字与上一页完全相同。

```{code-cell} python
import numpy as np

from apsim import MarketParams, simulate_market

# 上一页四只股票的载荷（假设数字）：每行是一只股票的 β_i'，列依次为市场、规模、价值
B_ex = np.array([[0.9, -0.4, -0.3],     # 白酒龙头
                 [1.5,  0.2,  0.1],     # 券商股
                 [0.7, -0.5,  0.8],     # 大盘银行股
                 [1.1,  1.2, -0.2]])    # 小盘股；B: (N, K) = (4, 3)
lam = np.asarray(MarketParams().factor_mean)       # λ: (K,) = (3,)，即模拟器默认的 μ
```

### 特征

**白话。** 特征是在期初就能观察到的股票属性，例如市值、账面市值比、动量（通常是过去 12 个月、跳过最近 1 个月的累计收益）、换手率。特征本身不是收益，而是研究者据以分组、预测或解释收益的期初信息。不同特征的单位不同，常见做法是每期在横截面上把它标准化：减去当期全部股票的均值，再除以标准差。

**例子。** 假设某个月全部股票对数市值的横截面均值为 23.0、标准差为 1.0，一只小盘股的对数市值为 21.2，则它标准化后的规模特征为 $(21.2 - 23.0) / 1.0 = -1.8$，即比平均水平小 1.8 个标准差。本章模拟器中的特征服从 $N(\mathbf 0, \mathbf I_L)$，正是这种标准化之后的尺度：$z = 2$ 表示比平均水平高 2 个标准差。

**“特征决定载荷”是什么意思。** 一只股票的载荷不是生来固定的，而是随它的特征变化：一家公司从小盘股成长为大盘股，它对规模因子的载荷就会下降。传统做法是给每只股票估计一个不随时间变化的 $\bv\beta_i$，例如用过去 60 个月做时间序列回归。这样做有三个问题：公司变了，固定的 β 跟不上；单只股票的时间序列短、噪声大；$N$ 只股票要估计 $N \times K$ 个参数，新上市的股票还没有历史数据。“特征决定载荷”的做法则假定载荷是期初特征的函数，函数的形式对所有股票相同：本章取线性函数 $\bv\beta_{i,t} = \bv\Gamma'\tilde{\mathbf z}_{i,t-1}$，只需估计 $(L+1) \times K$ 的矩阵 $\bv\Gamma$，并且可以用全部股票共同估计；股票的特征一变，它的载荷随之改变，$\bv\Gamma$ 却不变。这是 IPCA 模型的设定 [@kelly2019characteristics]。

用模拟器的默认 $\bv\Gamma$ 举例。默认参数下 $L = 5$、$K = 3$，$\bv\Gamma$ 是 $6\times 3$ 矩阵：常数项一行为 $(1, 0, 0)$，$z_1$ 一行为 $(0, 0.5, 0)$，$z_2$ 一行为 $(0, 0, 0.5)$，其余三行为 0。于是

```{math}
:enumerated: false
\bv\beta_{i,t} = \bv\Gamma'\tilde{\mathbf z}_{i,t-1} = \left(1,\ 0.5\, z_{1,i,t-1},\ 0.5\, z_{2,i,t-1}\right)' .
```

设股票 A 的 $z_1 = 2$，股票 B 的 $z_1 = -2$，两者的其他特征都为 0。则 A 的载荷为 $(1, 1, 0)'$，B 的载荷为 $(1, -1, 0)'$：二者对第一个因子的载荷相同，对第二个因子的载荷相反。若本期因子为 $\mathbf f_t = (4\%, 2\%, -1\%)'$，A 的因子部分为 $4\% + 1 \times 2\% = 6\%$，B 为 $4\% - 1 \times 2\% = 2\%$；第三个因子的 $-1\%$ 对两者都没有影响，因为它们的 $z_2 = 0$。如果下个月 A 的 $z_1$ 降到 0，它对第二个因子的载荷也随之降到 0，而 $\bv\Gamma$ 没有变。

**公式。** 特征向量 $\mathbf z_{i,t-1} \in \Real^L$，带常数项的版本 $\tilde{\mathbf z}_{i,t-1} = (1, \mathbf z_{i,t-1}')' \in \Real^{L+1}$，载荷 $\bv\beta_{i,t} = \bv\Gamma'\tilde{\mathbf z}_{i,t-1}$（$K\times(L+1)$ 乘 $(L+1)\times 1$，得 $K\times 1$），见式 {eq}`eq-ch0-char` 与 {eq}`eq-ch0-beta`。面板中特征对应 `panel["z1"]`……`panel["zL"]` 列，$\bv\Gamma$ 对应 `mkt.gamma`。

(ch0-unpack-gamma)=
**拆开讲 $\bv\beta_{i,t} = \bv\Gamma'\tilde{\mathbf z}_{i,t-1}$。** 先写出两个向量的分量：

```{math}
:enumerated: false
\underbrace{\mathbf z_{i,t-1}}_{L\times 1} = \begin{pmatrix} z_{1,i,t-1} \\ z_{2,i,t-1} \\ \vdots \\ z_{L,i,t-1} \end{pmatrix} , \qquad
\underbrace{\tilde{\mathbf z}_{i,t-1}}_{(L+1)\times 1} = \begin{pmatrix} 1 \\ \mathbf z_{i,t-1} \end{pmatrix} = \begin{pmatrix} 1 \\ z_{1,i,t-1} \\ \vdots \\ z_{L,i,t-1} \end{pmatrix} .
```

$z_{j,i,t-1}$ 是股票 $i$ 在第 $t-1$ 期末的第 $j$ 个特征，单位是横截面标准差，例如 $z_1$ 可以是标准化的对数市值，$z_2$ 可以是标准化的账面市值比，其余是动量、换手率之类（模拟器中的特征是抽象的，没有名字）。$\tilde{\mathbf z}_{i,t-1}$ 在最上面多了一个常数 1，分量记作 $\tilde z_{j,i,t-1}$，$j = 0, 1, \dots, L$，其中 $\tilde z_{0,i,t-1} = 1$。载荷矩阵 $\bv\Gamma$ 有 $L+1$ 行、$K$ 列：

```{math}
:enumerated: false
\underbrace{\bv\Gamma}_{(L+1)\times K} = \begin{pmatrix} \Gamma_{0,1} & \Gamma_{0,2} & \cdots & \Gamma_{0,K} \\ \Gamma_{1,1} & \Gamma_{1,2} & \cdots & \Gamma_{1,K} \\ \vdots & \vdots & & \vdots \\ \Gamma_{L,1} & \Gamma_{L,2} & \cdots & \Gamma_{L,K} \end{pmatrix} ,
```

第 $j$ 行对应特征 $j$（第 0 行对应常数项），第 $k$ 列对应因子 $k$。$\Gamma_{j,k}$ 的含义是：特征 $j$ 每高 1 个标准差，股票对因子 $k$ 的载荷增加 $\Gamma_{j,k}$；第 0 行 $\Gamma_{0,k}$ 是全部特征都为 0（处于平均水平）的股票对因子 $k$ 的载荷。**维度**：$\bv\Gamma'$ 是 $K\times(L+1)$，乘 $(L+1)\times 1$ 的 $\tilde{\mathbf z}_{i,t-1}$，得到 $K\times 1$ 的 $\bv\beta_{i,t}$。**展开运算**：$\bv\beta_{i,t}$ 的第 $k$ 个分量是 $\bv\Gamma$ 的第 $k$ 列与 $\tilde{\mathbf z}_{i,t-1}$ 的内积，

```{math}
:enumerated: false
\beta_{k,i,t} = \Gamma_{0,k} + \Gamma_{1,k}\, z_{1,i,t-1} + \dots + \Gamma_{L,k}\, z_{L,i,t-1} = \sum_{j=0}^{L} \Gamma_{j,k}\, \tilde z_{j,i,t-1} ,
```

即“基准载荷，加上每个特征偏离平均水平的程度乘以它对这个因子的影响”。默认参数下，

```{math}
:enumerated: false
\bv\Gamma = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 0.5 & 0 \\ 0 & 0 & 0.5 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{pmatrix} , \qquad
\bv\beta_{i,t} = \begin{pmatrix} 1 \\ 0.5\, z_{1,i,t-1} \\ 0.5\, z_{2,i,t-1} \end{pmatrix} ,
```

股票 A 的 $\tilde{\mathbf z}_{A,t-1} = (1, 2, 0, 0, 0, 0)'$，于是 $\beta_{2,A,t} = 0 + 0.5 \times 2 = 1$，与上面的结果相同。**特例**：$L = 0$（没有特征）时 $\tilde{\mathbf z}_{i,t-1} = 1$，$\bv\beta_{i,t}$ 就是 $\bv\Gamma$ 唯一一行的转置，所有股票、所有时期的载荷都相同；$K = L+1$、$\bv\Gamma = \mathbf I_{L+1}$ 时载荷就是特征本身，这是业界 Barra 风险模型的口径，见[下一页](#sec-ch0-fcl)。**数据矩阵**：同一期 $N$ 只股票的 $\tilde{\mathbf z}_{i,t-1}'$ 按行堆叠成 $N\times(L+1)$ 的 $\tilde{\mathbf Z}_{t-1}$，载荷矩阵为 $\mathbf B_t = \tilde{\mathbf Z}_{t-1}\bv\Gamma$（$N\times(L+1)$ 乘 $(L+1)\times K$，得 $N\times K$），第 $i$ 行是 $\bv\beta_{i,t}'$。**$\bv\Gamma$ 由所有股票、所有时期共用，特征 $\tilde{\mathbf z}_{i,t-1}$ 每只股票每期各有一个**，这正是“只需估计 $(L+1)\times K$ 个参数”的原因。下面代码中的 `zt_AB @ gamma0` 就是两只股票时的 $\tilde{\mathbf Z}_{t-1}\bv\Gamma$。

```{code-cell} python
size_z = (21.2 - 23.0) / 1.0                               # 对数市值 21.2；横截面均值 23.0、标准差 1.0（假设数字）
print(f"标准化的规模特征：{size_z:.1f}")
assert np.isclose(size_z, -1.8)

gamma0 = MarketParams().resolved_gamma()                    # 默认 Γ: (L+1, K) = (6, 3)
zt_AB = np.array([[1.0,  2.0, 0.0, 0.0, 0.0, 0.0],         # 股票 A 的 z̃'：常数项 1，z1 = 2，其余特征为 0
                  [1.0, -2.0, 0.0, 0.0, 0.0, 0.0]])        # 股票 B：z1 = -2；zt_AB: (2, L+1) = (2, 6)
beta_AB = zt_AB @ gamma0                                    # (2, 6) @ (6, 3) -> (2, 3)：每行 β_i' = z̃_i'Γ
f_AB = np.array([0.04, 0.02, -0.01])                        # f_t: (3,)
part_AB = beta_AB @ f_AB                                    # (2, 3) @ (3,) -> (2,)：β_i'f_t
print("默认 Γ：\n", gamma0)
print("A、B 的载荷（每行一只股票）：\n", beta_AB)
print("因子部分：", part_AB.round(4))
gamma_expected = np.zeros((6, 3))                           # 正文描述的默认 Γ：常数项、z1、z2 三行非零，其余三行为 0
gamma_expected[0, 0], gamma_expected[1, 1], gamma_expected[2, 2] = 1.0, 0.5, 0.5
assert np.array_equal(gamma0, gamma_expected)
assert np.allclose(beta_AB, [[1, 1, 0], [1, -1, 0]]) and np.allclose(part_AB, [0.06, 0.02])
```

### 定价误差 $\alpha$

**白话。** 定价误差是一只股票的期望超额收益中，载荷与风险溢价解释不了的那一部分：实际的期望收益减去“应得的”风险补偿 $\bv\beta_i'\bv\lambda$。

**例子。** 上面的小盘股按载荷算出的风险补偿是每月 0.94%。假设它长期的平均超额收益是每月 1.20%，那么相对于这个三因子模型，它的定价误差为 $1.20\% - 0.94\% = 0.26\%$：每月比它承担的风险所应得的多 0.26%。同理，前面 6 个月回归中券商股的截距 0.2%，就是它相对于单因子模型的样本定价误差（只有 6 个月，噪声很大，不能据此下任何结论）。

$\alpha \neq 0$ 可能有两种截然不同的解释。一种是**风险补偿**：模型漏掉了某个因子，这只股票暴露于这个被遗漏的风险，所谓“α”其实是对它的补偿，换一个更完整的模型就会消失。另一种是**错误定价**：投资者的认知偏差使价格偏离了风险所对应的水平，而套利受到限制，偏离无法被及时纠正，α 是风险无法解释的收益。例如一只股票被过度乐观的投资者追捧而高估，A 股融券困难，看空的投资者难以通过卖空把价格压回去，高估可以持续，此后的平均收益偏低，表现为负的 α。A 股的壳价值则不宜简单归入任何一类：它源于发行上市管制，是一种与基本面无关的价值成分，既不是普通的风险补偿，也不能简单看作错误定价（见“A 股视角”一节、@liu2019size 与第 16、17 章）。区分这两种解释往往很难，是第 8 章的主题。这里只需记住两点：α 总是相对于某一个选定的因子模型而言的；一个特征能预测收益，并不等于它带来了 α（见“常见误区”一节的误区四）。

**公式。**

```{math}
:enumerated: false
\alpha_i = \E[r_i] - \bv\beta_i'\bv\lambda .
```

因子是超额收益时，它也就是时间序列回归 $r_{i,t} = a_i + \bv\beta_i'\mathbf f_t + u_{i,t}$ 的截距 $a_i$：对回归式取期望，$\E[r_i] = a_i + \bv\beta_i'\E[\mathbf f_t] = a_i + \bv\beta_i'\bv\lambda$。本章模型中定价误差由特征决定，$\alpha_{i,t} = \bv\theta'\mathbf z_{i,t-1}$，见式 {eq}`eq-ch0-alpha`，对应 `panel["alpha"]` 列，系数 $\bv\theta$ 对应 `mkt.theta`。默认 $\bv\theta = \mathbf 0$，即没有错误定价；0.6 节取 $\theta_1 = 0.01$，即 $z_1$ 每高一个标准差，定价误差每月高 1 个百分点。

(ch0-unpack-alpha)=
**拆开讲定价误差的两个式子。** $\alpha_i = \E[r_i] - \bv\beta_i'\bv\lambda$ 中三项都是标量、单位都是每月的收益：$\E[r_i]$ 是实际的期望超额收益（上例 1.20%），$\bv\beta_i'\bv\lambda$ 是风险补偿（上例 0.94%，拆解见[上文](#ch0-unpack-lambda)），二者之差 $\alpha_i$ 是定价误差（上例 0.26%）。本章模型中的 $\alpha_{i,t} = \bv\theta'\mathbf z_{i,t-1}$ 拆开是

```{math}
:enumerated: false
\underbrace{\bv\theta}_{L\times 1} = \begin{pmatrix} \theta_1 \\ \theta_2 \\ \vdots \\ \theta_L \end{pmatrix} , \qquad
\alpha_{i,t} = \underbrace{\bv\theta'}_{1\times L}\,\underbrace{\mathbf z_{i,t-1}}_{L\times 1} = \theta_1 z_{1,i,t-1} + \theta_2 z_{2,i,t-1} + \dots + \theta_L z_{L,i,t-1} .
```

$\theta_j$ 是特征 $j$ 每高 1 个标准差、定价误差每月增加的量，单位与收益相同；式中没有常数项，所以全部特征处于平均水平的股票没有定价误差。例如取 $\theta_1 = 0.01$、其余为 0 时，$z_1 = 2$ 的股票 $\alpha_{i,t} = 0.01 \times 2 = 2\%$，$z_1 = -1$ 的股票为 $-1\%$。**特例**：$\bv\theta = \mathbf 0$ 时所有股票的 $\alpha_{i,t} = 0$，模型对每只股票都正确定价，这是模拟器的默认设定。**数据矩阵**：同一期 $N$ 只股票堆叠为 $\bv\alpha_t = \mathbf Z_{t-1}\bv\theta$（$N\times L$ 乘 $L\times 1$，得 $N\times 1$），$\mathbf Z_{t-1}$ 的第 $i$ 行是 $\mathbf z_{i,t-1}'$。$\bv\theta$ 由所有股票共用，特征每只股票各有一套。

```{code-cell} python
avg_small = 0.012                                  # 小盘股的长期平均超额收益（假设数字）
comp_small = B_ex[3] @ lam                         # 风险补偿 β'λ：(3,) @ (3,) -> 标量
alpha_small = avg_small - comp_small               # α = E[r] − β'λ
print(f"风险补偿 β'λ = {comp_small:.2%}，定价误差 α = {alpha_small:.2%}")
assert np.isclose(comp_small, 0.0094) and np.isclose(alpha_small, 0.0026)
```

### 随机贴现因子

**白话。** 同样是一块钱，在不同的未来状态里价值不同。股市暴跌、经济不景气、收入可能下降的“坏时候”，多一块钱可以救急，因而更值钱；人人都在赚钱的“好时候”，多一块钱就没那么重要。随机贴现因子（SDF）给未来的每一个状态标出一个数 $m$：那个状态下一块钱在今天的价格（状态价格），除以该状态发生的概率。$m$ 越大，那个状态越是坏时候，那里的一块钱越值钱。一项资产的价格，就是它在各个状态下的支付按 $m$ 折算后的平均值。

**例子。** 假设下个月只有两种等可能的状态：大盘上涨的“好时候”与大盘下跌的“坏时候”，SDF 分别为 0.8 与 1.2（假设数字，二者的平均值为 1）。考虑三项资产：

| 资产（假设） | 好时候的支付 | 坏时候的支付 | 期望支付 | 价格 $\E[mx]$ | 期望收益率 |
|---|---|---|---|---|---|
| 无风险债券 | 100 | 100 | 100 | 100 | 0 |
| 券商股 | 115 | 85 | 100 | 97 | 约 3.09% |
| 大盘银行股 | 104 | 98 | 101 | 100.4 | 约 0.60% |

以券商股为例，价格为 $0.5 \times 0.8 \times 115 + 0.5 \times 1.2 \times 85 = 46 + 51 = 97$ 元，期望收益率为 $100 / 97 - 1 \approx 3.09\%$。券商股在好时候支付多、在坏时候支付少，而好时候的钱不值钱、坏时候的钱值钱，所以投资者只肯出 97 元买一个期望支付为 100 元的资产，它的期望收益因此高于无风险收益，高出的部分就是这项资产的风险溢价（期望超额收益），不要与每单位载荷的价格 $\bv\lambda$ 混淆。大盘银行股的支付在两种状态下相差不大，期望超额收益也小。本例中无风险收益为 0，是因为 SDF 的平均值取了 1，与模拟器的归一化 $\E[m_t] = 1$ 相同。

**公式。** 支付为 $x$ 的资产，价格为

```{math}
:enumerated: false
P = \E[m x] = \sum_{s} \pi_s m_s x_s ,
```

其中 $\pi_s$ 是状态 $s$ 的概率，$m_s$ 与 $x_s$ 是该状态下的 SDF 与支付，都是标量。一项“只在状态 $s$ 支付一块钱”的资产，今天的价格是 $\pi_s m_s$，所以 $m_s$ 就是状态 $s$ 下一块钱的价格除以该状态的概率。两边除以 $P$，得到对总收益的 $1 = \E[mR]$；无风险资产的支付确定，所以 $R^f = 1/\E[m]$；对超额收益，$0 = \E[mR^e]$。把 $\E[mR^e] = \E[m]\E[R^e] + \Cov(m, R^e)$ 代入后一式，

```{math}
:enumerated: false
\E[R^e] = -\frac{\Cov(m, R^e)}{\E[m]} .
```

在 $m$ 低的好时候收益高的资产，与 $m$ 负相关，期望超额收益为正；券商股正是这样。上例中券商股的 $\Cov(m, R^e) \approx -3.09\%$，$\E[m] = 1$，恰好给出 3.09% 的期望超额收益（由下方代码核对）。

这与 β 定价是同一件事的两种说法。如果 $m$ 是因子的线性函数，$\Cov(m, R^e)$ 就是资产与各个因子的协方差的线性组合，也就是由载荷决定，上式于是化为 $\E[R^e] = \bv\beta'\bv\lambda$。本章模拟市场的 SDF 正是这样的线性函数 $m_t = 1 - \mathbf b'(\mathbf f_t - \bv\mu)$，其中 $\mathbf b \in \Real^K$，见 0.4 节式 {eq}`eq-ch0-sdf`，对应 `mkt.sdf()` 与 `mkt.sdf_weights`；默认参数下 $\mathbf b$ 的各分量都为正，因子收益总体高于均值（$\mathbf b'(\mathbf f_t - \bv\mu) > 0$）的好时候 $m_t < 1$。例如 $\mathbf f_t = (4\%, 2\%, -1\%)'$ 时 $m_t \approx 0.92$，第一个因子下跌 6%、另两个因子为 0 时 $m_t \approx 1.19$。SDF 与 β 定价之间的关系，在模拟市场中由 0.5 节的定理与[习题 %s](#ex-ch0-beta) 严格证明，一般理论在第 1～4 章展开。

```{code-cell} python
prob = np.array([0.5, 0.5])                          # 好时候、坏时候的概率 π_s
m_state = np.array([0.8, 1.2])                       # 各状态的 SDF m_s（假设数字）
payoff = np.array([[100.0, 100.0],                   # 无风险债券
                   [115.0,  85.0],                   # 券商股
                   [104.0,  98.0]])                  # 大盘银行股；payoff 形状为 (3, 2)，行为资产、列为状态
price = payoff @ (prob * m_state)                    # (3, 2) @ (2,) -> (3,)：P = Σ_s π_s m_s x_s
R_state = payoff / price[:, None]                    # 各状态的总收益，广播 (3, 2) / (3, 1) -> (3, 2)
E_m = prob @ m_state                                 # E[m]
E_R = R_state @ prob                                 # (3, 2) @ (2,) -> (3,)：E[R]
R_f = 1 / E_m                                        # R^f = 1 / E[m]
cov_mR = (R_state * m_state) @ prob - E_m * E_R      # Cov(m, R^e) = E[mR] − E[m]E[R]（R^f 是常数）
print("价格：", price, "\n期望收益率：", (E_R - 1).round(4), "\nCov(m, R^e)：", cov_mR.round(4))
assert np.allclose(price, [100, 97, 100.4]) and np.isclose(R_f, 1)
assert np.allclose((R_state * m_state) @ prob, 1)                  # 每项资产都满足 1 = E[mR]
assert np.allclose(E_R - R_f, -cov_mR / E_m)                       # E[R^e] = −Cov(m, R^e) / E[m]
assert np.allclose(E_R - 1, [0, 3 / 97, 0.6 / 100.4])

# 模拟器默认参数下的 SDF：m_t = 1 − b'(f_t − μ)，b = Σ_f⁻¹μ
p_def = MarketParams()
mu_def = np.asarray(p_def.factor_mean)                             # μ: (K,)
b_def = np.linalg.solve(p_def.factor_cov, mu_def)                  # 解 Σ_f b = μ，b: (K,)
m_up = 1 - b_def @ (np.array([0.04, 0.02, -0.01]) - mu_def)        # (K,) @ (K,) -> 标量
m_down = 1 - b_def @ (np.array([-0.06, 0.0, 0.0]) - mu_def)
print(f"b = {b_def.round(3)}；m（好时候）= {m_up:.3f}，m（坏时候）= {m_down:.3f}")
assert np.all(b_def > 0) and 0.92 < m_up < 0.93 and 1.18 < m_down < 1.19
```

(ch0-unpack-sdf)=
**拆开讲 SDF 的式子。** 先看标量式 $P = \E[mx] = \sum_s \pi_s m_s x_s$：$P$ 与 $x_s$ 以元计（上例券商股为 97 元与 115 元、85 元）；$\pi_s$ 是概率，没有单位；$m_s$ 是“状态 $s$ 下一块钱今天值多少钱”再除以概率，也没有单位。$\E[R^e] = -\Cov(m, R^e)/\E[m]$ 中的 $R^e$ 与 $m$ 都没有单位，所以 $\Cov(m, R^e)$ 也是一个纯数（上例约 $-3.09\%$）。再看线性 SDF $m_t = 1 - \mathbf b'(\mathbf f_t - \bv\mu)$，写出分量：

```{math}
:enumerated: false
\underbrace{\mathbf b}_{K\times 1} = \begin{pmatrix} b_1 \\ b_2 \\ \vdots \\ b_K \end{pmatrix} , \qquad
\underbrace{\bv\mu}_{K\times 1} = \E[\mathbf f_t] = \begin{pmatrix} \mu_1 \\ \mu_2 \\ \vdots \\ \mu_K \end{pmatrix} , \qquad
\underbrace{\mathbf f_t - \bv\mu}_{K\times 1} = \begin{pmatrix} f_{1,t} - \mu_1 \\ f_{2,t} - \mu_2 \\ \vdots \\ f_{K,t} - \mu_K \end{pmatrix} .
```

$\mu_k$ 是第 $k$ 个因子的月度均值，$f_{k,t} - \mu_k$ 是这个月它比平常好多少；$b_k$ 是 SDF 对第 $k$ 个因子的系数，单位是“收益的倒数”，$b_k$ 越大，这个因子比平常好时 $m_t$ 下降得越多。**维度**：$\mathbf b'$ 是 $1\times K$，乘 $K\times 1$ 的 $\mathbf f_t - \bv\mu$，得到一个数，所以 $m_t$ 是标量。**展开运算**：

```{math}
:enumerated: false
m_t = 1 - b_1(f_{1,t} - \mu_1) - b_2(f_{2,t} - \mu_2) - \dots - b_K(f_{K,t} - \mu_K) .
```

即“从 1 出发，每个因子比平常好一点，$m_t$ 就按 $b_k$ 的比例下降一点”。默认参数下 $\bv\mu = (0.6\%, 0.3\%, 0.4\%)'$，因子波动率 $\bv\sigma = (5\%, 3\%, 3\%)'$，因子互不相关，$\mathbf b = \bv\Sigma_f^{-1}\bv\mu$ 逐分量化为 $b_k = \mu_k/\sigma_k^2$，即 $\mathbf b = (2.4,\ 3.33,\ 4.44)'$（后两个分量保留两位小数）。上例 $\mathbf f_t = (4\%, 2\%, -1\%)'$ 时，三项依次约为 $2.4 \times 3.4\% = 0.0816$、$3.33 \times 1.7\% \approx 0.0567$、$4.44 \times (-1.4\%) \approx -0.0622$，于是 $m_t = 1 - (0.0816 + 0.0567 - 0.0622) \approx 0.924$：前两个因子比平常好、第三个比平常差，合起来是好时候。**特例**：$K = 1$ 时 $m_t = 1 - b_1(f_{1,t} - \mu_1)$，SDF 是市场因子的线性函数，对应 CAPM。**数据矩阵**：$m_t$ 只带时间下标，同一期**所有股票面对同一个 $m_t$**，$\mathbf b$、$\bv\mu$ 也由所有股票共用；把 $T$ 期堆叠，得到 $T\times 1$ 的列向量 $\mathbf 1_T - (\mathbf F - \mathbf 1_T\bv\mu')\mathbf b$，其中 $\mathbf F - \mathbf 1_T\bv\mu'$ 是 $T\times K$ 的去均值因子矩阵（每一行减去 $\bv\mu'$）。代码中 `mkt.sdf()` 返回这 $T$ 个值，形状为 `(T,)`。

```{code-cell} python
# 拆开讲 m_t = 1 − b'(f_t − μ)：b_k = μ_k / σ_k²，逐项求和与内积一致
sigma_def = np.asarray(p_def.factor_vol)                          # σ: (K,) = (3,)
terms_up = b_def * (np.array([0.04, 0.02, -0.01]) - mu_def)       # 逐项 b_k (f_k − μ_k)，(K,)
print("b =", b_def.round(4), "；逐项", terms_up.round(4), "；m =", round(1 - terms_up.sum(), 4))
assert np.allclose(b_def, mu_def / sigma_def**2) and np.allclose(b_def, [2.4, 10 / 3, 40 / 9])
assert np.allclose(terms_up.round(4), [0.0816, 0.0567, -0.0622]) and np.isclose(1 - terms_up.sum(), m_up)
assert 0.923 < m_up < 0.925
```

### 概念—符号—模拟器字段对照表

| 概念 | 符号 | 维度 | 模拟器中的字段 |
|---|---|---|---|
| 总收益 | $R_{i,t}$ | 标量 | 不单独生成 |
| 无风险利率 | $R^f_{t-1}$ | 标量 | 不单独生成；归一化 $\E[m_t] = 1$ 相当于 $R^f = 1$ |
| 超额收益 | $r_{i,t}$（即 $R^e_{i,t}$） | 标量 | `panel["exret"]` |
| 因子 | $\mathbf f_t$ | $K\times 1$ | `mkt.factors`（列 `f1`……`fK`） |
| 载荷（暴露、β） | $\bv\beta_{i,t}$ | $K\times 1$ | `panel["beta1"]`……`panel["betaK"]` |
| 风险溢价 | $\bv\lambda$（本章模型中 $= \bv\mu$） | $K\times 1$ | `mkt.params.factor_mean` |
| 特征 | $\mathbf z_{i,t-1}$ | $L\times 1$ | `panel["z1"]`……`panel["zL"]` |
| 特征到载荷的映射 | $\bv\Gamma$ | $(L+1)\times K$ | `mkt.gamma` |
| 定价误差 | $\alpha_{i,t}$ | 标量 | `panel["alpha"]` |
| 特征到定价误差的映射 | $\bv\theta$ | $L\times 1$ | `mkt.theta`（参数名为 `alpha_coef`） |
| 随机贴现因子 | $m_t$ | 标量 | `mkt.sdf()` |
| SDF 系数 | $\mathbf b$ | $K\times 1$ | `mkt.sdf_weights` |

下面的代码单元生成一个很小的模拟市场，核对表中的字段确实存在、形状与维度一致。

```{code-cell} python
tiny = simulate_market(MarketParams(n_stocks=3, n_periods=2), seed=0)
L0, K0 = tiny.params.n_chars, tiny.params.n_factors
expected_cols = {"exret", "alpha"} | {f"beta{k}" for k in range(1, K0 + 1)} | {f"z{j}" for j in range(1, L0 + 1)}
assert expected_cols <= set(tiny.panel.columns)
assert list(tiny.factors.columns) == [f"f{k}" for k in range(1, K0 + 1)]
assert tiny.gamma.shape == (L0 + 1, K0) and tiny.theta.shape == (L0,)          # Γ: (L+1, K)，θ: (L,)
assert tiny.sdf_weights.shape == (K0,) and tiny.sdf().shape == (tiny.params.n_periods,)
print(sorted(tiny.panel.columns))
```

表中“因子”“载荷”“定价误差”等名词，在国内量化业界常有不同的口径（例如业界的“因子”多指这里的特征），读文献或与业界交流时要先确认所指，对照见[术语对照：学术与业界](../glossary.md)。

