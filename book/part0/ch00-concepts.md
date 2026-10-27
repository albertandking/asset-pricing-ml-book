---
kernelspec:
  name: python3
  display_name: Python 3
numbering:
  enumerator: '0.%s'
  exercise:
    template: 习题 %s
---

(sec-ch0-concepts)=
# 0.2 基本概念速览

**本页准备。** 本书每个页面的代码在独立的内核中运行。下面的单元导入本页用到的模块。

```{code-cell} python
import numpy as np

from apsim import MarketParams, simulate_market
```

从 0.3 节起，正文会直接使用超额收益、因子、载荷、风险溢价、特征、定价误差、随机贴现因子这些术语，0.4 节的模型就是把它们写成方程。本节先逐个讲清楚它们的含义。每个概念都按同一个顺序介绍：先用一两句白话说明它是什么，再举一个数值例子，最后给出公式，并指出它在本章模型中对应哪个符号。例子借用读者熟悉的 A 股情形，但**例子中的股票都不指具体公司，所有数字都是为说明概念而假设的**，不是实际数据。每个例子后面的代码单元用 `numpy` 复算其中的数字，并用 `assert` 核对。

本节只求建立直觉：模型的方程在 0.4 节，定理与证明在 0.5 节，系统的理论在第 1～4 章展开。

### 总收益、无风险利率与超额收益

**白话。** 总收益是期末拿回的钱（价格加上期间的红利）与期初投入的钱之比。无风险利率是把钱放在没有违约风险的资产上所得的收益，在期初就已确定。超额收益是二者之差，也就是因为承担了风险而多得（或少得）的那一部分。

**例子。** 假设某只白酒龙头股第 $t-1$ 个月末的价格为 100 元，第 $t$ 个月末涨到 102 元，期间每股派发红利 1 元。持有一个月得到的支付是 $102 + 1 = 103$ 元，总收益为 $103 / 100 = 1.03$，即收益率 3%。再假设这个月的无风险利率为 0.2%（无风险总收益为 1.002），它在月初就已知道。这只股票当月的超额收益为 $1.03 - 1.002 = 0.028$，即 2.8%。用收益率相减也得到同样的结果：$3\% - 0.2\% = 2.8\%$，因为总收益与收益率只差一个 1，相减时消去了。

**公式。** 按 0.3 节的时点约定，从第 $t-1$ 期末持有到第 $t$ 期末，

```{math}
:enumerated: false
R_{i,t} = \frac{x_{i,t}}{P_{i,t-1}}, \qquad R^e_{i,t} = R_{i,t} - R^f_{t-1}, \qquad r_{i,t} = R^e_{i,t},
```

其中 $P_{i,t-1}$ 是股票 $i$ 在第 $t-1$ 期末的价格，$x_{i,t}$ 是第 $t$ 期末的价格加上期间的红利，$R^f_{t-1}$ 是第 $t-1$ 期末已知的无风险总收益，三者与 $R_{i,t}$、$R^e_{i,t}$ 都是标量。面板与模拟部分把超额收益简记为 $r_{i,t}$。

(ch0-unpack-exret)=
**逐项看单位。** 价格 $P_{i,t-1}$ 与支付 $x_{i,t}$ 以元计（上例为 100 元与 103 元）。总收益 $R_{i,t}$ 是两笔钱之比，没有单位（上例为 1.03）；$R^f_{t-1}$ 同样是总收益，没有单位（上例为 1.002）。超额收益 $R^e_{i,t}$ 是两个总收益之差，也没有单位，习惯上写成百分数（上例为 2.8%）。本章的一期是一个月，所以这些收益都是月度收益；下标 $i$ 表示第 $i$ 只股票，$t-1$、$t$ 分别表示期初与期末。

为什么资产定价研究超额收益，而不是总收益？有三个原因。第一，无风险利率是“等待”的报酬，与承担什么风险无关，而且本身随时间变化；资产定价要回答的是不同股票的期望收益为什么不同，先减去所有资产都能拿到的无风险收益，比较才干净。第二，超额收益是一个零成本头寸的收益：按无风险利率借入 1 元买入股票，期初不花钱，期末得到 $R_{i,t} - R^f_{t-1}$。零成本头寸的价格为 0，所以下文的定价方程对超额收益写成 $0 = \E[m R^e]$，形式最简单。第三，本章的因子本身也是超额收益（见下一个概念）。股票与因子用同一种口径时，时间序列回归的截距就是定价误差 $\alpha$，并且在模型对因子自身没有定价误差时，因子的风险溢价就是它的均值 $\E[\mathbf f_t]$（条件见下文“风险溢价”与“定价误差”两个概念）。

本章的模拟器只生成超额收益 $r_{i,t}$（面板的 `panel["exret"]` 列），不单独生成总收益与无风险利率，原因见 0.4 节末尾的注释“为什么是 $\E[m_t] = 1$”。

```{code-cell} python
# 白酒龙头（假设数字）：月初价格 100 元，月末价格 102 元，期间每股红利 1 元
P_prev, P_now, div = 100.0, 102.0, 1.0
R_gross = (P_now + div) / P_prev            # 总收益 R_{i,t} = x_{i,t} / P_{i,t-1}
Rf_prev = 1.002                             # 无风险总收益 R^f_{t-1}，月初已知
r_excess = R_gross - Rf_prev                # 超额收益 r_{i,t} = R_{i,t} - R^f_{t-1}
print(f"总收益 {R_gross:.4f}，收益率 {R_gross - 1:.2%}，超额收益 {r_excess:.2%}")
assert np.isclose(R_gross, 1.03) and np.isclose(r_excess, 0.028)
assert np.isclose((R_gross - 1) - (Rf_prev - 1), r_excess)     # 用收益率相减，结果相同
```

### 因子

**白话。** 因子是许多股票共同面对的收益来源。大盘大涨的月份，多数股票跟着上涨；小盘股整体跑赢大盘股的月份，几乎所有小盘股都受益。用一个可交易组合的收益来度量这种“共同的涨跌”，就得到一个因子。本章的因子，以及第三、四部分大多数模型中的因子，都是超额收益：市场因子是市场组合的超额收益；规模、价值这类因子是“买入一组股票、同时卖空另一组股票”的多空组合收益。但并非所有因子都是资产收益：第 1、4、10 章会遇到消费增长、宏观变量这类不可交易的因子，它们的风险溢价 $\bv\lambda$ 不等于因子的均值。

**例子。** 假设某个月：市场组合的收益率为 3.2%，无风险利率为 0.2%；小市值股票组合的收益率为 5.0%，大市值组合为 2.0%；价值股组合（相对于基本面“便宜”的股票）为 2.5%，成长股组合为 3.5%。于是这个月

- 市场因子 $= 3.2\% - 0.2\% = 3.0\%$；
- 规模因子 $=$ 小市值组合 $-$ 大市值组合 $= 5.0\% - 2.0\% = 3.0\%$；
- 价值因子 $=$ 价值股组合 $-$ 成长股组合 $= 2.5\% - 3.5\% = -1.0\%$。

规模因子与价值因子不需要再减无风险利率：多空组合的买入与卖空各占 1 元，期初不花钱，两条腿的无风险利率相互抵消，多空收益本身就是超额收益。规模因子为正，说明这个月小盘股跑赢大盘股；价值因子为负，说明成长股跑赢价值股。Liu、Stambaugh 与 Yuan 为中国市场构造的三因子模型（CH3）正是由市场、规模、价值三个因子组成，其中价值按盈利价格比（EP）而不是账面市值比划分，并剔除了市值最小的 30% 的股票 [@liu2019size]；因子的具体构造见第 15、16 章。

**公式。** 记 $R_{\text{mkt},t}$、$R_{\text{small},t}$、$R_{\text{big},t}$、$R_{\text{value},t}$、$R_{\text{growth},t}$ 分别为市场组合、小市值组合、大市值组合、价值股组合与成长股组合的总收益，则

```{math}
:enumerated: false
f_{1,t} = R_{\text{mkt},t} - R^f_{t-1}, \qquad f_{2,t} = R_{\text{small},t} - R_{\text{big},t}, \qquad f_{3,t} = R_{\text{value},t} - R_{\text{growth},t}, \qquad \underbrace{\mathbf f_t}_{3\times 1} = (f_{1,t}, f_{2,t}, f_{3,t})' .
```

一般地，$K$ 个因子组成列向量 $\mathbf f_t = (f_{1,t}, \dots, f_{K,t})' \in \Real^K$。本例中 $\mathbf f_t = (3\%, 3\%, -1\%)'$。本章模型中的 $\mathbf f_t$ 由式 {eq}`eq-ch0-factor` 生成，对应 `mkt.factors` 的列 `f1`……`fK`。模拟器的因子是抽象的，没有名字；但在默认参数下，所有股票对第一个因子的载荷都是 1，所以 $f_{1,t}$ 的角色类似市场因子。

(ch0-unpack-f)=
**拆开讲因子向量 $\mathbf f_t$。** 写出分量，本例的 $\mathbf f_t$ 竖着写是

```{math}
:enumerated: false
\mathbf f_t = \begin{pmatrix} f_{1,t} \\ f_{2,t} \\ f_{3,t} \end{pmatrix} = \begin{pmatrix} 3\% \\ 3\% \\ -1\% \end{pmatrix} .
```

- $f_{1,t}$：市场因子在第 $t$ 个月的值，即市场组合的超额收益（Fama–French 记作 MKT）
- $f_{2,t}$：规模因子在第 $t$ 个月的值，即小市值组合减大市值组合（Fama–French 记作 SMB）
- $f_{3,t}$：价值因子在第 $t$ 个月的值，即价值股组合减成长股组合（Fama–French 记作 HML；CH3 中按盈利价格比划分）

三个分量都是超额收益，没有单位，通常写成百分数。**维度**：$\mathbf f_t$ 是 $3\times 1$ 的列向量，一般地是 $K\times 1$，即 $\mathbf f_t \in \Real^K$；转置后的 $\mathbf f_t' = (f_{1,t}, \dots, f_{K,t})$ 是 $1\times K$ 的行向量，正文为了节省篇幅常把列向量写成“行向量的转置”。**特例**：$K = 1$ 时只有市场因子，就是 CAPM 的设定；$K = 3$ 对应 Fama–French 三因子 [@fama1993common] 或 CH3；$K = 5$ 对应 Fama–French 五因子 [@fama2015five]。**数据矩阵**：把 $T$ 个月的因子按行堆叠，得到 $T\times K$ 的因子矩阵

```{math}
:enumerated: false
\underbrace{\mathbf F}_{T\times K} = \begin{pmatrix} \mathbf f_1' \\ \mathbf f_2' \\ \vdots \\ \mathbf f_T' \end{pmatrix} = \begin{pmatrix} f_{1,1} & f_{2,1} & \cdots & f_{K,1} \\ f_{1,2} & f_{2,2} & \cdots & f_{K,2} \\ \vdots & \vdots & & \vdots \\ f_{1,T} & f_{2,T} & \cdots & f_{K,T} \end{pmatrix} ,
```

第 $t$ 行是第 $t$ 个月的 $K$ 个因子，第 $k$ 列是第 $k$ 个因子的时间序列，$(t, k)$ 元素是 $f_{k,t}$。因子只带时间下标、不带股票下标，所以**所有股票共用同一个 $\mathbf F$**。代码中的 `mkt.factors.to_numpy()` 就是 $\mathbf F$，形状为 `(T, K)`，它的一行是公式中列向量 $\mathbf f_t$ 的转置。

```{code-cell} python
# 某个月各组合的收益率（假设数字）
R_mkt, R_free = 0.032, 0.002          # 市场组合、无风险利率
R_small, R_big = 0.050, 0.020         # 小市值组合、大市值组合
R_value, R_growth = 0.025, 0.035      # 价值股组合、成长股组合
f_month = np.array([R_mkt - R_free, R_small - R_big, R_value - R_growth])   # f_t: (K,) = (3,)
print("f_t =", f_month)
assert np.allclose(f_month, [0.03, 0.03, -0.01])
```

(sec-ch0-loading)=
### 载荷（暴露、β）

**白话。** 载荷衡量一只股票对某个因子有多敏感：因子收益每高 1 个百分点，这只股票的收益平均高多少个百分点。载荷也叫暴露（exposure）；只有一个因子时，常把它记作 β。

**单因子的例子。** 只考虑市场因子。假设大盘银行股的市场载荷为 0.7，券商股为 1.5。某个月市场因子为 $+4\%$，则由市场因子带来的那部分收益，银行股为 $0.7 \times 4\% = 2.8\%$，券商股为 $1.5 \times 4\% = 6\%$；若市场因子为 $-4\%$，二者分别为 $-2.8\%$ 与 $-6\%$。券商股是“高 β”股票：大盘涨时它涨得更多，大盘跌时它也跌得更多。这里算的只是“因子带来的部分”，每只股票还有自己的消息带来的异质性收益，下面估计载荷时会看到。

**多因子的例子。** 有多个因子时，一只股票对每个因子各有一个载荷，$K$ 个载荷组成一个向量。下表是四只股票对 CH3 三个因子的载荷（假设数字），最后一列给出与上一个例子的因子 $\mathbf f_t = (3\%, 3\%, -1\%)'$ 对应的因子部分 $\bv\beta_i'\mathbf f_t$：

| 股票（假设） | 市场 | 规模 | 价值 | 直观含义 | 因子部分 $\bv\beta_i'\mathbf f_t$ |
|---|---|---|---|---|---|
| 白酒龙头 | 0.9 | −0.4 | −0.3 | 大市值、估值高，表现像成长股 | 1.8% |
| 券商股 | 1.5 | 0.2 | 0.1 | 对大盘最敏感 | 5.0% |
| 大盘银行股 | 0.7 | −0.5 | 0.8 | 大市值、估值低，表现像价值股 | −0.2% |
| 小盘股 | 1.1 | 1.2 | −0.2 | 随小盘股整体涨跌 | 7.1% |

以大盘银行股为例：市场上涨 3% 给它带来 $0.7 \times 3\% = 2.1\%$；这个月小盘股跑赢大盘股 3%，它的规模载荷为负，损失 $0.5 \times 3\% = 1.5\%$；成长股跑赢价值股 1%，它的价值载荷为正，再损失 $0.8 \times 1\% = 0.8\%$。合计 $-0.2\%$：大盘涨了，它反而略跌。

**公式。** 股票 $i$ 的载荷向量 $\bv\beta_i = (\beta_{1,i}, \dots, \beta_{K,i})' \in \Real^K$，因子带来的收益是内积

```{math}
:enumerated: false
\underbrace{\bv\beta_i'}_{1\times K}\underbrace{\mathbf f_t}_{K\times 1} = \sum_{k=1}^{K} \beta_{k,i} f_{k,t} .
```

把 $N$ 只股票的 $\bv\beta_i'$ 按行堆叠成 $N\times K$ 矩阵 $\mathbf B$（上表的前三列数字就是 $4\times 3$ 的 $\mathbf B$），$\mathbf B\mathbf f_t$（$N\times 1$）的第 $i$ 个元素就是 $\bv\beta_i'\mathbf f_t$。本章模型中载荷随时间变化，记作 $\bv\beta_{i,t} \in \Real^K$，由式 {eq}`eq-ch0-beta` 给出，对应面板的 `panel["beta1"]`……`panel["betaK"]` 列；堆叠后的矩阵为 $\mathbf B_t$。

(ch0-unpack-betaf)=
**拆开讲 $\bv\beta_i'\mathbf f_t$。** 写出分量，以大盘银行股为例：

```{math}
:enumerated: false
\bv\beta_i = \begin{pmatrix} \beta_{1,i} \\ \beta_{2,i} \\ \beta_{3,i} \end{pmatrix} = \begin{pmatrix} 0.7 \\ -0.5 \\ 0.8 \end{pmatrix} ,
```

- $\beta_{1,i} = 0.7$：股票 $i$ 对市场因子的载荷，市场因子每高 1 个百分点，它的收益平均高 0.7 个百分点
- $\beta_{2,i} = -0.5$：对规模因子的载荷，为负，表现像大盘股
- $\beta_{3,i} = 0.8$：对价值因子的载荷，为正，表现像价值股

一般地，$\beta_{k,i}$ 是股票 $i$ 对第 $k$ 个因子的载荷（下标“因子在前、股票在后”），是“收益的百分点除以因子的百分点”，没有单位。**维度**：$\bv\beta_i$ 是 $K\times 1$ 的列向量，转置后 $\bv\beta_i' = (\beta_{1,i}, \dots, \beta_{K,i})$ 是 $1\times K$ 的行向量；$1\times K$ 乘 $K\times 1$ 得 $1\times 1$，所以 $\bv\beta_i'\mathbf f_t$ 是一个数。**展开运算**：

```{math}
:enumerated: false
\bv\beta_i'\mathbf f_t = \beta_{1,i} f_{1,t} + \beta_{2,i} f_{2,t} + \dots + \beta_{K,i} f_{K,t}
= 0.7 \times 3\% + (-0.5) \times 3\% + 0.8 \times (-1\%) = -0.2\% ,
```

也就是 $K$ 个因子共同贡献给股票 $i$ 的那部分收益，与上表最后一列相同。**特例**：$K = 1$ 时只剩 $\beta_{1,i} f_{1,t}$，就是前面单因子例子中的 $0.7 \times 4\% = 2.8\%$，即 CAPM 的写法；$K = 3$、$K = 5$ 分别对应三因子与五因子模型。**数据矩阵**：同一期的 $N$ 只股票按行堆叠，

```{math}
:enumerated: false
\underbrace{\mathbf B}_{N\times K} = \begin{pmatrix} \bv\beta_1' \\ \bv\beta_2' \\ \vdots \\ \bv\beta_N' \end{pmatrix} = \begin{pmatrix} \beta_{1,1} & \beta_{2,1} & \cdots & \beta_{K,1} \\ \beta_{1,2} & \beta_{2,2} & \cdots & \beta_{K,2} \\ \vdots & \vdots & & \vdots \\ \beta_{1,N} & \beta_{2,N} & \cdots & \beta_{K,N} \end{pmatrix} , \qquad
\underbrace{\mathbf B}_{N\times K}\,\underbrace{\mathbf f_t}_{K\times 1} = \begin{pmatrix} \bv\beta_1'\mathbf f_t \\ \bv\beta_2'\mathbf f_t \\ \vdots \\ \bv\beta_N'\mathbf f_t \end{pmatrix} ,
```

$\mathbf B$ 的第 $i$ 行是股票 $i$ 的载荷，第 $k$ 列是全部股票对第 $k$ 个因子的载荷，$(i, k)$ 元素是 $\beta_{k,i}$。同一期**所有股票面对同一个 $\mathbf f_t$，每只股票各有一行载荷**，所以 $\mathbf B\mathbf f_t$ 是 $N\times 1$ 的列向量，第 $i$ 个元素是股票 $i$ 的因子部分。估计 $\bv\beta_i$ 时则按时间堆叠：所有股票共用 $T\times K$ 的因子矩阵 $\mathbf F$ 与 $T\times(1+K)$ 的设计矩阵 $[\mathbf 1_T, \mathbf F]$，每只股票各估一套 $\bv\beta_i$，见下文“用回归斜率估计载荷”与[记号表的范例](#notation-unpack)。

```{code-cell} python
# 单因子：大盘银行股 β = 0.7，券商股 β = 1.5（假设数字）
beta_single = np.array([0.7, 1.5])                         # (2,)
for f_m in (0.04, -0.04):
    print(f"市场因子 {f_m:+.0%}：因子部分", (beta_single * f_m).round(4))
assert np.allclose(beta_single * 0.04, [0.028, 0.06])

# 多因子：每行是一只股票的载荷 β_i'，列依次为市场、规模、价值
B_ex = np.array([[0.9, -0.4, -0.3],     # 白酒龙头
                 [1.5,  0.2,  0.1],     # 券商股
                 [0.7, -0.5,  0.8],     # 大盘银行股
                 [1.1,  1.2, -0.2]])    # 小盘股；B: (N, K) = (4, 3)
factor_part = B_ex @ f_month                                # (4, 3) @ (3,) -> (4,)：第 i 个元素为 β_i'f_t
print("因子部分 β_i'f_t：", factor_part.round(4))
assert np.allclose(factor_part, [0.018, 0.050, -0.002, 0.071])
```

**用回归斜率估计载荷。** 载荷不能直接观察，需要从数据中估计。最直接的办法是把股票的超额收益对因子做时间序列回归：把每个月的（因子收益，股票收益）画成散点，拟合直线的斜率就是载荷的估计值，即因子每高 1 个百分点，股票收益平均高多少。下面是券商股与市场因子 6 个月的超额收益（假设数字）：

| 月份 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| 市场因子 | 3.0% | −1.0% | 2.0% | −4.0% | 1.0% | 5.0% |
| 券商股 | 5.2% | −1.8% | 3.2% | −5.8% | 2.2% | 7.2% |

最小二乘回归的斜率恰好为 1.5，截距为 0.2%。每个月的实际收益与拟合直线 $0.2\% + 1.5 f_t$ 相差 $\pm 0.5\%$ 或 0，这些残差就是券商股自身的消息带来的异质性收益。单因子时 $f_t$ 为标量，回归斜率等于样本协方差除以样本方差：

```{math}
:enumerated: false
\hat\beta_i = \frac{\widehat{\Cov}(r_{i,t}, f_t)}{\widehat{\Var}(f_t)} .
```

多因子时，回归模型为 $r_{i,t} = a_i + \bv\beta_i'\mathbf f_t + u_{i,t}$，其中截距 $a_i$ 与残差 $u_{i,t}$ 是标量，总体斜率为 $\bv\beta_i = \bv\Sigma_f^{-1}\Cov(\mathbf f_t, r_{i,t})$（$K\times K$ 乘 $K\times 1$，得 $K\times 1$）。每个载荷衡量的是“其他因子不变时”对这个因子的敏感度。

(ch0-unpack-ts)=
**拆开讲时间序列回归与总体斜率。** 回归式中各项都是第 $t$ 个月的标量：$r_{i,t}$ 是股票 $i$ 的超额收益；$a_i$ 是截距，不随时间变化，单位与收益相同（上例为每月 0.2%）；$\bv\beta_i'\mathbf f_t$ 是因子部分，拆解见[上文](#ch0-unpack-betaf)；$u_{i,t}$ 是残差，即股票自身消息带来的那部分收益。总体斜率中的两个量拆开是

```{math}
:enumerated: false
\underbrace{\Cov(\mathbf f_t, r_{i,t})}_{K\times 1} = \begin{pmatrix} \Cov(f_{1,t}, r_{i,t}) \\ \vdots \\ \Cov(f_{K,t}, r_{i,t}) \end{pmatrix} , \qquad
\underbrace{\bv\Sigma_f}_{K\times K} = \Var(\mathbf f_t) = \begin{pmatrix} \Var(f_{1,t}) & \cdots & \Cov(f_{1,t}, f_{K,t}) \\ \vdots & \ddots & \vdots \\ \Cov(f_{K,t}, f_{1,t}) & \cdots & \Var(f_{K,t}) \end{pmatrix} :
```

前者的第 $k$ 个元素是股票收益与第 $k$ 个因子的协方差；后者是因子的协方差矩阵，对角元是各因子的方差，$(k, l)$ 元素是第 $k$ 与第 $l$ 个因子的协方差，矩阵是对称的。$\bv\Sigma_f^{-1}$ 的作用是扣除因子之间的相关：如果因子互不相关（本章模拟器就是这样），$\bv\Sigma_f$ 是对角矩阵，斜率逐个分量化为 $\beta_{k,i} = \Cov(f_{k,t}, r_{i,t})/\Var(f_{k,t})$，每个因子各算各的；$K = 1$ 时 $\bv\Sigma_f$ 就是标量 $\Var(f_t)$，上式退化为单因子的“协方差除以方差”。估计时，股票 $i$ 的 $T$ 个月收益组成 $T\times 1$ 的列向量，对 $T\times(1+K)$ 的设计矩阵 $[\mathbf 1_T, \mathbf F]$ 回归；所有股票共用同一个设计矩阵，每只股票各得一组 $(a_i, \bv\beta_i')$，堆叠后的写法见[记号表的范例](#notation-unpack)。下面代码中的 `X6` 就是 $T = 6$、$K = 1$ 时的设计矩阵，形状为 `(6, 2)`。单因子公式中的帽子表示由样本算出的估计值：$\widehat{\Cov}$、$\widehat{\Var}$ 是这 6 个月数据的样本协方差与样本方差，$\hat\beta_i$ 是载荷的估计值；不带帽子的 $\bv\beta_i$ 是总体值。

这个例子只用了 6 个月，数字又是特意挑选的，所以斜率恰好是 1.5，残差也很整齐。真实数据中，单只股票的异质性收益很大：0.4 节会算出，在本书的模拟市场中，异质性冲击约占单只股票月度收益方差的四分之三以上，用一只股票几年的数据估计出的载荷误差可观。另外，本章模型中的载荷 $\bv\beta_{i,t}$ 随时间变化，而上面的回归只有一个常数斜率，它估计的是载荷的平均值（[习题 %s](#ex-ch0-ts)）。

```{code-cell} python
# 6 个月的市场因子与券商股超额收益（假设数字）
f_mkt6 = np.array([3.0, -1.0, 2.0, -4.0, 1.0, 5.0]) / 100      # (T,) = (6,)
r_brk6 = np.array([5.2, -1.8, 3.2, -5.8, 2.2, 7.2]) / 100      # (6,)
X6 = np.column_stack([np.ones(6), f_mkt6])                     # (6, 2)：常数项与因子
(a_brk, beta_brk), *_ = np.linalg.lstsq(X6, r_brk6, rcond=None)
slope_cov = np.cov(r_brk6, f_mkt6)[0, 1] / np.var(f_mkt6, ddof=1)    # 样本 Cov(r, f) / Var(f)
resid6 = r_brk6 - X6 @ np.array([a_brk, beta_brk])             # (6, 2) @ (2,) -> (6,)
print(f"斜率 {beta_brk:.4f}，截距 {a_brk:.4%}，Cov / Var = {slope_cov:.4f}")
print("残差（单位：%）：", (resid6 * 100).round(2) + 0.0)       # 加 0.0 把 -0.0 显示为 0.0
assert np.isclose(beta_brk, 1.5) and np.isclose(a_brk, 0.002) and np.isclose(slope_cov, beta_brk)
assert np.allclose(np.abs(resid6), [0.005, 0.005, 0, 0, 0.005, 0.005])
```

### 风险溢价 $\bv\lambda$

**白话。** 风险溢价是承担一单位某种因子暴露所得到的平均补偿，也就是风险的“单价”。如果一只股票的期望超额收益完全来自风险补偿，它就等于“承担了多少风险（载荷）”乘以“每单位风险的价格（风险溢价）”，再对所有因子加总。

**例子。** 假设市场、规模、价值三个因子的月度风险溢价分别为 0.6%、0.3%、0.4%（这恰好是本章模拟器默认的因子均值 $\bv\mu$）。上表四只股票的期望超额收益为

- 白酒龙头：$0.9 \times 0.6\% - 0.4 \times 0.3\% - 0.3 \times 0.4\% = 0.30\%$；
- 券商股：$1.5 \times 0.6\% + 0.2 \times 0.3\% + 0.1 \times 0.4\% = 1.00\%$；
- 大盘银行股：$0.7 \times 0.6\% - 0.5 \times 0.3\% + 0.8 \times 0.4\% = 0.59\%$；
- 小盘股：$1.1 \times 0.6\% + 1.2 \times 0.3\% - 0.2 \times 0.4\% = 0.94\%$。

券商股的期望收益最高，因为它承担的市场风险最多；小盘股的期望收益中，市场部分 $1.1 \times 0.6\% = 0.66\%$ 仍是大头，但规模暴露又额外带来 $1.2 \times 0.3\% = 0.36\%$，使它的市场载荷虽然比券商股低，期望收益却与券商股接近；大盘银行股的市场载荷最低，但正的价值载荷带来了补偿；白酒龙头的期望超额收益最低，因为它对规模与价值的载荷都为负，在小盘股、价值股表现差的月份它反而相对较好，起到一定的对冲作用，投资者因此愿意接受较低的期望收益。

$\E[r_i] = \bv\beta_i'\bv\lambda$ 的直观含义是：**在 β 定价模型中，股票之间期望收益的差别全部来自载荷的差别。** 两只载荷相同的股票，不论行业、题材、单只股票的总波动有多大，期望超额收益都相同；异质性收益可以通过分散化消除，得不到补偿。载荷为 $\mathbf 0$ 的资产（例如无风险资产）期望超额收益为 0，所以式中没有截距。还要注意，风险溢价是长期平均的补偿，某个月实际实现的因子收益（如上例的 $3\%$、$3\%$、$-1\%$）可以与它相差很远。

**公式。**

```{math}
:enumerated: false
\E[r_i] = \underbrace{\bv\beta_i'}_{1\times K}\underbrace{\bv\lambda}_{K\times 1} = \sum_{k=1}^{K} \beta_{k,i}\lambda_k, \qquad \bv\lambda \in \Real^K .
```

当因子本身是超额收益、并且模型为因子自身正确定价（因子的定价误差为 0）时，风险溢价就是因子的均值：$\lambda_k = \E[f_{k,t}]$。原因是把上式用于第 $k$ 个因子本身：它对自己的载荷为 1、对其他因子的载荷为 0、定价误差为 0，于是 $\E[f_{k,t}] = \lambda_k$。不可交易的因子（例如消费增长、宏观变量）不是资产收益，不能套用这一推理，$\bv\lambda$ 一般不等于因子的均值，需要另行估计（第 1、4、10 章）。本章模型中因子都是超额收益，$\bv\lambda = \bv\mu$，对应参数 `mkt.params.factor_mean`；载荷随特征变化时，上式的条件版本是 $\E_{t-1}[r_{i,t}] = \alpha_{i,t} + \bv\beta_{i,t}'\bv\lambda$，其中 $\alpha_{i,t}$ 是下文的定价误差（证明见[习题 %s](#ex-ch0-beta)）。

(ch0-unpack-lambda)=
**拆开讲 $\E[r_i] = \bv\beta_i'\bv\lambda$。** 写出分量，本例的风险溢价向量是

```{math}
:enumerated: false
\bv\lambda = \begin{pmatrix} \lambda_1 \\ \lambda_2 \\ \lambda_3 \end{pmatrix} = \begin{pmatrix} 0.6\% \\ 0.3\% \\ 0.4\% \end{pmatrix} ,
```

- $\lambda_1 = 0.6\%$：市场因子的风险溢价，市场载荷每多 1，期望超额收益每月多 0.6%
- $\lambda_2 = 0.3\%$：规模因子的风险溢价，每单位规模载荷每月补偿 0.3%
- $\lambda_3 = 0.4\%$：价值因子的风险溢价，每单位价值载荷每月补偿 0.4%

$\lambda_k$ 的单位与收益相同（这里是每月的百分数），因为载荷没有单位。左边的 $\E[r_i]$ 是股票 $i$ 的期望超额收益（每月），不带时间下标，表示长期平均。**维度**：$\bv\lambda$ 是 $K\times 1$ 的列向量，与 $\bv\beta_i$ 同维；$1\times K$ 的 $\bv\beta_i'$ 乘 $K\times 1$ 的 $\bv\lambda$，得到一个数。**展开运算**，以小盘股为例：

```{math}
:enumerated: false
\E[r_i] = \beta_{1,i}\lambda_1 + \beta_{2,i}\lambda_2 + \dots + \beta_{K,i}\lambda_K = 1.1 \times 0.6\% + 1.2 \times 0.3\% + (-0.2) \times 0.4\% = 0.94\% ,
```

即“每个因子的暴露乘以这个因子的单价，再加总”。它与 $\bv\beta_i'\mathbf f_t$ 形式相同，区别在于 $\mathbf f_t$ 是某个月实现的因子收益，$\bv\lambda$ 是长期平均的补偿。**特例**：$K = 1$ 时 $\E[r_i] = \beta_i\lambda_1$，$\lambda_1$ 是市场组合的期望超额收益，这就是 CAPM 的证券市场线：期望超额收益与市场 β 成正比。**数据矩阵**：把 $N$ 只股票的期望超额收益堆叠成 $N\times 1$ 的列向量 $(\E[r_1], \dots, \E[r_N])'$，它等于 $\mathbf B\bv\lambda$（$N\times K$ 乘 $K\times 1$，得 $N\times 1$）。**所有股票共用同一个 $\bv\lambda$**（风险的单价对谁都一样），每只股票各有一行载荷 $\bv\beta_i'$；下面代码中的 `B_ex @ lam` 计算的就是 $\mathbf B\bv\lambda$。

```{code-cell} python
lam = np.asarray(MarketParams().factor_mean)       # λ: (K,) = (3,)，即模拟器默认的 μ
exp_ret = B_ex @ lam                                # (4, 3) @ (3,) -> (4,)：E[r_i] = β_i'λ
print("期望超额收益 β_i'λ：", exp_ret.round(4))
assert np.allclose(lam, [0.006, 0.003, 0.004])
assert np.allclose(exp_ret, [0.0030, 0.0100, 0.0059, 0.0094])
```

特征、定价误差与随机贴现因子三个概念，以及概念—符号—模拟器字段对照表，见下一页[0.2 基本概念速览（续）：特征、定价误差与随机贴现因子](#sec-ch0-concepts-sdf)。
