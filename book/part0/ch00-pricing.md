---
short_title: 0.5～0.6 SDF 的定价性质
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

(sec-ch0-pricing)=
# 0.5～0.6 SDF 的定价性质与代码验证

**本页准备。** 本书每个页面的代码在独立的内核中运行，前面页面定义的变量在这里不可用。下面的单元导入本页用到的模块，设置统一的作图样式与中文字体（0.6 节的收敛图用到），并按与 0.4 节完全相同的参数和种子重建默认市场。模型的方程见上一页[0.3～0.4 记号与模拟市场的模型](./ch00-model.md)。

```{code-cell} python
import numpy as np
import pandas as pd

from apsim import MarketParams, simulate_market, plotting

cjk_font = plotting.setup()                                # 返回所选中文字体名

# 0.4 节的默认市场（种子 0）与因子均值 μ
params = MarketParams()
mkt = simulate_market(params, seed=0)
mu = np.asarray(params.factor_mean)                       # μ: (K,)
```

## 0.5 模拟市场中 SDF 的定价性质

下面的定理说明，式 {eq}`eq-ch0-sdf` 给出的 $m_t$ 确实是这个市场的随机贴现因子：它正确地为因子定价，而它为股票定价时留下的误差恰好就是 $\alpha_{i,t}$。这是全书用模拟数据检验方法的基础：只要知道真实的 $m_t$，就知道每只股票在每一期“应有的”定价误差。在 $\E[m_t] = 1$ 的归一化下，这里的定价误差与 0.2 节的定义“期望超额收益减去 $\bv\beta_{i,t}'\bv\lambda$”（取以 $\mathbf z_{i,t-1}$ 为条件的期望）是同一个量（[习题 %s](#ex-ch0-beta)）。

:::{prf:theorem} 模拟市场 SDF 的定价性质
:label: thm-ch0-sdf
在式 {eq}`eq-ch0-factor`～{eq}`eq-ch0-sdf` 与[假设 %s](#asm-ch0-indep) 下，设 $\bv\Sigma_f$ 可逆，$m_t = 1 - \mathbf b'(\mathbf f_t - \bv\mu)$，$\mathbf b = \bv\Sigma_f^{-1}\bv\mu$。则对任意 $t$ 与 $i$：

1. $\E[m_t] = 1$；
2. $\E[m_t \mathbf f_t] = \mathbf 0 \in \Real^K$，即 $m_t$ 为全部 $K$ 个因子正确定价；
3. $\E[m_t r_{i,t} \mid \mathbf z_{i,t-1}] = \alpha_{i,t} = \bv\theta' \mathbf z_{i,t-1}$。特别地，当 $\bv\theta = \mathbf 0$ 时，$\E[m_t r_{i,t} \mid \mathbf z_{i,t-1}] = 0$，即 $m_t$ 在条件意义下为每一只股票正确定价。
:::

(ch0-unpack-thm)=
**拆开讲定理的三条。** 第 1、3 条是标量等式：$\E[m_t]$ 是 SDF 的均值，没有单位；$\E[m_t r_{i,t} \mid \mathbf z_{i,t-1}]$ 是用 $m_t$ 给股票 $i$ 定价时、以期初特征为条件的定价误差，单位与收益相同（每月）；$\bv\theta'\mathbf z_{i,t-1}$ 的展开见[拆解](#ch0-unpack-alpha)。第 2 条是 $K\times 1$ 的向量等式，标量 $m_t$ 乘到 $\mathbf f_t$ 的每个分量上，逐个分量写出是

```{math}
:enumerated: false
\E[m_t \mathbf f_t] = \begin{pmatrix} \E[m_t f_{1,t}] \\ \E[m_t f_{2,t}] \\ \vdots \\ \E[m_t f_{K,t}] \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \\ \vdots \\ 0 \end{pmatrix} .
```

第 $k$ 个分量 $\E[m_t f_{k,t}] = 0$ 就是超额收益的定价方程 $0 = \E[m R^e]$ 用于第 $k$ 个因子（因子本身是可交易组合的超额收益），所以第 2 条一次写出了 $K$ 个定价方程。**特例**：$K = 1$ 时第 2 条就是 $\E[m_t f_t] = 0$，0.6 节的 `sympy` 验证的正是这一情形。

:::{prf:proof}
**第 1 条。** $m_t$ 是 $\mathbf f_t$ 的线性函数，取期望并利用 $\E[\mathbf f_t] = \bv\mu$：

```{math}
:enumerated: false
\E[m_t] = 1 - \mathbf b'\left(\E[\mathbf f_t] - \bv\mu\right) = 1 - \mathbf b' \mathbf 0 = 1.
```

式中 $\E[\mathbf f_t] - \bv\mu$ 是 $K\times 1$ 的零向量，$1\times K$ 的 $\mathbf b'$ 乘它得到数 0；逐项写就是 $\sum_k b_k(\E[f_{k,t}] - \mu_k) = \sum_k b_k \cdot 0 = 0$。

**第 2 条。** 由于 $\mathbf b'(\mathbf f_t - \bv\mu)$ 是标量，它等于自身的转置 $(\mathbf f_t - \bv\mu)'\mathbf b$，所以 $m_t \mathbf f_t = \mathbf f_t - \mathbf f_t (\mathbf f_t - \bv\mu)'\mathbf b$，其中 $\mathbf f_t (\mathbf f_t - \bv\mu)'$ 是 $K\times K$ 矩阵、$\mathbf b$ 是 $K\times 1$ 向量。取期望：

```{math}
:enumerated: false
\underbrace{\E[m_t \mathbf f_t]}_{K\times 1} = \underbrace{\E[\mathbf f_t]}_{K\times 1} - \underbrace{\E\left[\mathbf f_t (\mathbf f_t - \bv\mu)'\right]}_{K\times K} \underbrace{\mathbf b}_{K\times 1} .
```

(ch0-unpack-outer)=
这里第一次出现两个列向量的“外积”：$K\times 1$ 的 $\mathbf f_t$ 乘 $1\times K$ 的 $(\mathbf f_t - \bv\mu)'$，结果是 $K\times K$ 矩阵，而不是一个数：

```{math}
:enumerated: false
\mathbf f_t (\mathbf f_t - \bv\mu)' = \begin{pmatrix} f_{1,t}(f_{1,t}-\mu_1) & \cdots & f_{1,t}(f_{K,t}-\mu_K) \\ \vdots & \ddots & \vdots \\ f_{K,t}(f_{1,t}-\mu_1) & \cdots & f_{K,t}(f_{K,t}-\mu_K) \end{pmatrix} ,
```

$(k, l)$ 元素是 $f_{k,t}(f_{l,t}-\mu_l)$。它再乘 $K\times 1$ 的 $\mathbf b$，第 $k$ 个分量是 $f_{k,t}\sum_l (f_{l,t}-\mu_l)\, b_l = f_{k,t}\,\mathbf b'(\mathbf f_t - \bv\mu)$，与直接逐个分量写出的 $m_t f_{k,t} = f_{k,t} - f_{k,t}\,\mathbf b'(\mathbf f_t-\bv\mu)$ 一致。把数 $\mathbf b'(\mathbf f_t - \bv\mu)$ 改写成矩阵乘积，是为了把期望移到只含 $\mathbf f_t$ 的那个矩阵上，$\mathbf b$ 是常数，留在期望之外。

第二项中，把 $\mathbf f_t$ 写成 $(\mathbf f_t - \bv\mu) + \bv\mu$：

```{math}
:enumerated: false
\E\left[\mathbf f_t (\mathbf f_t - \bv\mu)'\right] = \E\left[(\mathbf f_t - \bv\mu)(\mathbf f_t - \bv\mu)'\right] + \bv\mu\, \E\left[(\mathbf f_t - \bv\mu)'\right] = \bv\Sigma_f + \bv\mu\,\mathbf 0' = \bv\Sigma_f .
```

逐个元素看，$(k, l)$ 元素是 $\E[f_{k,t}(f_{l,t}-\mu_l)] = \E[(f_{k,t}-\mu_k)(f_{l,t}-\mu_l)] + \mu_k\,\E[f_{l,t}-\mu_l] = \Cov(f_{k,t}, f_{l,t}) + 0$，正是 $\bv\Sigma_f$ 的 $(k, l)$ 元素 $\Sigma_{kl}$（$\bv\Sigma_f$ 的结构见[拆解](#ch0-unpack-ts)）。

代回并利用 $\mathbf b = \bv\Sigma_f^{-1}\bv\mu$：

```{math}
:enumerated: false
\E[m_t \mathbf f_t] = \bv\mu - \bv\Sigma_f \bv\Sigma_f^{-1} \bv\mu = \mathbf 0 .
```

逐个分量写，第 $k$ 个分量是 $\mu_k - \sum_l \Sigma_{kl}\, b_l$，由 $\bv\Sigma_f\mathbf b = \bv\mu$ 的第 $k$ 个方程（[拆解](#ch0-unpack-sdfb)），它等于 0；对角情形下就是 $\mu_k - \sigma_k^2 \cdot \mu_k/\sigma_k^2 = 0$。

这一步只用到 $\mathbf f_t$ 的一、二阶矩，既不需要正态性，也不需要 $\bv\Sigma_f$ 是对角矩阵。反过来，若要求 $1 - \mathbf b'(\mathbf f_t-\bv\mu)$ 为因子定价，由上式 $\bv\mu - \bv\Sigma_f \mathbf b = \mathbf 0$，$\mathbf b$ 必须等于 $\bv\Sigma_f^{-1}\bv\mu$，所以在这一函数形式下 $\mathbf b$ 是唯一的。

**第 3 条。** 把式 {eq}`eq-ch0-ret` 代入：

```{math}
:enumerated: false
\E[m_t r_{i,t} \mid \mathbf z_{i,t-1}]
= \E[m_t \alpha_{i,t} \mid \mathbf z_{i,t-1}] + \E[m_t \bv\beta_{i,t}' \mathbf f_t \mid \mathbf z_{i,t-1}] + \E[m_t \varepsilon_{i,t} \mid \mathbf z_{i,t-1}] .
```

右边三项都是标量，单位都是每月的收益，依次来自收益的三个部分：定价误差、因子部分与异质性冲击（[拆解](#ch0-unpack-return)）。

由式 {eq}`eq-ch0-beta` 与 {eq}`eq-ch0-alpha`，$\alpha_{i,t}$ 与 $\bv\beta_{i,t}$ 都是 $\mathbf z_{i,t-1}$ 的确定函数，可以提到条件期望之外：

```{math}
:enumerated: false
\E[m_t r_{i,t} \mid \mathbf z_{i,t-1}]
= \alpha_{i,t}\, \E[m_t \mid \mathbf z_{i,t-1}] + \bv\beta_{i,t}'\, \E[m_t \mathbf f_t \mid \mathbf z_{i,t-1}] + \E[m_t \varepsilon_{i,t} \mid \mathbf z_{i,t-1}] ,
```

其中第二项是 $(1\times K)$ 行向量与 $(K\times 1)$ 列向量之积，为标量，逐项展开为 $\sum_{k=1}^{K} \beta_{k,i,t}\,\E[m_t f_{k,t} \mid \mathbf z_{i,t-1}]$，即每个因子的载荷乘以 $m_t$ 对这个因子的条件定价误差，再加总。逐项处理：

- $m_t$ 与 $m_t \mathbf f_t$ 都只是 $\mathbf f_t$ 的函数。由[假设 %s](#asm-ch0-indep)，$\mathbf f_t$ 与 $\mathbf z_{i,t-1}$ 独立，所以条件期望等于无条件期望：$\E[m_t \mid \mathbf z_{i,t-1}] = \E[m_t] = 1$，$\E[m_t \mathbf f_t \mid \mathbf z_{i,t-1}] = \E[m_t \mathbf f_t] = \mathbf 0$（第 1、2 条）。
- $m_t \varepsilon_{i,t}$ 是 $(\mathbf f_t, \varepsilon_{i,t})$ 的函数，而 $(\mathbf f_t, \varepsilon_{i,t})$ 与 $\mathbf z_{i,t-1}$ 独立，所以 $\E[m_t \varepsilon_{i,t} \mid \mathbf z_{i,t-1}] = \E[m_t \varepsilon_{i,t}]$；再由 $\mathbf f_t$ 与 $\varepsilon_{i,t}$ 相互独立，$\E[m_t \varepsilon_{i,t}] = \E[m_t]\, \E[\varepsilon_{i,t}] = 1 \cdot 0 = 0$。

三项合起来：

```{math}
:enumerated: false
\E[m_t r_{i,t} \mid \mathbf z_{i,t-1}] = \alpha_{i,t} \cdot 1 + \bv\beta_{i,t}' \mathbf 0 + 0 = \alpha_{i,t} = \bv\theta' \mathbf z_{i,t-1} .
```

当 $\bv\theta = \mathbf 0$ 时右端为 0。
:::

证明的第 3 条中，独立性被用了两次，二者缺一不可。如果因子收益的分布依赖于特征（例如特征能预测因子收益），那么 $\E[m_t \mid \mathbf z_{i,t-1}]$ 一般不再等于 1，$\E[m_t \mathbf f_t \mid \mathbf z_{i,t-1}]$ 也不再为 $\mathbf 0$，一个只在无条件意义下为因子定价的 $m_t$，未必在条件意义下定价。这正是第 5 章条件模型要处理的问题。由于 $(\mathbf f_t, \varepsilon_{i,t})$ 与第 $t-1$ 期末的全部信息独立，同样的论证也给出 $\E_{t-1}[m_t r_{i,t}] = \alpha_{i,t}$。

由定理的第 3 条，再用一次迭代期望定律，可以得到两个在检验中很有用的无条件矩。

:::{prf:corollary} 无条件定价误差与管理组合
:label: cor-ch0-managed
在[定理 %s](#thm-ch0-sdf) 的条件下：

$$
\E[m_t r_{i,t}] = \E[\alpha_{i,t}] = 0, \qquad \underbrace{\E[m_t r_{i,t}\, \mathbf z_{i,t-1}]}_{L\times 1} = \bv\theta .
$$ (eq-ch0-managed)
:::

(ch0-unpack-managed)=
**拆开讲式 {eq}`eq-ch0-managed`。** 第一个等式是标量：$\E[m_t r_{i,t}]$ 是用 $m_t$ 给股票 $i$ 定价的无条件定价误差（每月）。第二个等式是 $L\times 1$ 的向量等式，标量 $m_t r_{i,t}$ 乘到 $\mathbf z_{i,t-1}$ 的每个分量上：

```{math}
:enumerated: false
\E[m_t r_{i,t}\, \mathbf z_{i,t-1}] = \begin{pmatrix} \E[m_t r_{i,t}\, z_{1,i,t-1}] \\ \E[m_t r_{i,t}\, z_{2,i,t-1}] \\ \vdots \\ \E[m_t r_{i,t}\, z_{L,i,t-1}] \end{pmatrix} = \begin{pmatrix} \theta_1 \\ \theta_2 \\ \vdots \\ \theta_L \end{pmatrix} .
```

$r_{i,t} z_{j,i,t-1}$ 是一个“管理组合”的收益：期初按特征 $j$ 的大小决定持有多少元股票 $i$（$z_{j,i,t-1} = 2$ 时按无风险利率借入 2 元买入，$z_{j,i,t-1} = -1$ 时卖空 1 元股票、把所得的 1 元按无风险利率借出），它仍是零成本头寸的超额收益，可以用 $m_t$ 定价；第 $j$ 个分量 $\E[m_t r_{i,t} z_{j,i,t-1}]$ 就是这个管理组合的定价误差，等于 $\theta_j$。例如 $\bv\theta = (0.01, 0, 0, 0, 0)'$ 时，按 $z_1$ 加权的管理组合每月有 1% 的定价误差，其余四个为 0。**特例**：$\bv\theta = \mathbf 0$ 时所有管理组合都被正确定价；$L = 1$ 时只有一个等式 $\E[m_t r_{i,t} z_{1,i,t-1}] = \theta_1$。**数据矩阵**：样本中用全部 $(i, t)$ 的平均 $(NT)^{-1}\sum_t\sum_i m_t r_{i,t}\mathbf z_{i,t-1}$ 估计它（[习题 %s](#ex-ch0-theta)）；$m_t$ 由同一期的所有股票共用，$\bv\theta$ 由所有股票、所有时期共用。

:::{prf:proof}
由迭代期望定律与定理第 3 条，$\E[m_t r_{i,t}] = \E\left[\E[m_t r_{i,t} \mid \mathbf z_{i,t-1}]\right] = \bv\theta' \E[\mathbf z_{i,t-1}] = \bv\theta'\mathbf 0 = 0$。同理，由于 $\mathbf z_{i,t-1}$ 在以自身为条件时可以提出，并且标量 $\bv\theta'\mathbf z_{i,t-1}$ 等于 $\mathbf z_{i,t-1}'\bv\theta$，

```{math}
:enumerated: false
\E[m_t r_{i,t}\, \mathbf z_{i,t-1}] = \E\left[\mathbf z_{i,t-1}\, \E[m_t r_{i,t} \mid \mathbf z_{i,t-1}]\right] = \underbrace{\E[\mathbf z_{i,t-1} \mathbf z_{i,t-1}']}_{L\times L}\, \underbrace{\bv\theta}_{L\times 1} = \mathbf I_L\, \bv\theta = \bv\theta ,
```

最后一步用到 $\mathbf z_{i,t-1} \sim N(\mathbf 0, \mathbf I_L)$。

其中 $\E[\mathbf z_{i,t-1}\mathbf z_{i,t-1}']$ 是 $L\times L$ 矩阵，$(j, l)$ 元素为 $\E[z_{j,i,t-1} z_{l,i,t-1}]$：$j = l$ 时是方差 1，$j \neq l$ 时是协方差 0，所以它是单位矩阵 $\mathbf I_L$；它乘 $\bv\theta$ 的第 $j$ 个分量是 $\sum_l \E[z_{j,i,t-1} z_{l,i,t-1}]\,\theta_l = \theta_j$。
:::

这个推论有一个值得记住的含义：即使存在错误定价（$\bv\theta \neq \mathbf 0$），每只股票的**无条件**定价误差仍然为 0，因为特征围绕 0 波动，错误定价时正时负、平均抵消。只有把收益按特征加权，即构造“管理组合”（managed portfolio）$r_{i,t}\, \mathbf z_{i,t-1}$，错误定价才在无条件矩中显现出来。用无条件矩检验条件模型时，检验资产的选择决定了检验能看到什么。

最后，$m_t$ 的波动也有明确的含义：$\Var(m_t) = \mathbf b'\bv\Sigma_f \mathbf b = \bv\mu'\bv\Sigma_f^{-1}\bv\mu$（三项的维度依次为 $1\times K$、$K\times K$、$K\times 1$，结果为标量），它等于用这 $K$ 个因子所能构成的组合的最大夏普比率的平方。Hansen–Jagannathan 界 [@hansen1991implications] 指出，任何为这些因子正确定价、且均值为 1 的 SDF，其标准差都不小于这一最大夏普比率；本章的 $m_t$ 是因子的线性函数，恰好达到这个下界。第 3 章将详细讨论。线性 SDF 还有一个缺点：$\mathbf f_t$ 服从正态分布，$m_t$ 以正的概率取负值。这本身并不意味着这些因子之间存在套利：无套利等价于存在**某个**严格为正的 SDF（第 2 章），而一个特定的线性 $m_t$ 取负值，并不排除另有一个正的 SDF 同样为这些因子正确定价。问题出在用这个 $m_t$ 为因子的非线性支付定价时，例如以因子为标的的期权：在 $m_t < 0$ 的状态上支付为正、其余状态支付为 0 的证券，会被 $m_t$ 赋予负的价格，而买入一个价格为负、支付非负的证券就是套利。所以线性 SDF 只适合为它所针对的线性支付定价。在默认参数下 $m_t < 0$ 的概率可以忽略，下一节会算出它。

(ch0-unpack-varm)=
**拆开讲 $\Var(m_t) = \mathbf b'\bv\Sigma_f\mathbf b = \bv\mu'\bv\Sigma_f^{-1}\bv\mu$。** 由 $m_t - 1 = -\mathbf b'(\mathbf f_t - \bv\mu)$ 与 $\E[m_t] = 1$，$\Var(m_t) = \E[\mathbf b'(\mathbf f_t-\bv\mu)(\mathbf f_t-\bv\mu)'\mathbf b] = \mathbf b'\bv\Sigma_f\mathbf b$（与定理第 2 条的证明一样，先把两个相同的数之积写成外积，见[拆解](#ch0-unpack-outer)）。$\mathbf b'\bv\Sigma_f\mathbf b$ 叫作二次型：$1\times K$、$K\times K$、$K\times 1$ 三者相乘，展开为

```{math}
:enumerated: false
\mathbf b'\bv\Sigma_f\mathbf b = \sum_{k=1}^{K}\sum_{l=1}^{K} b_k\,\Sigma_{kl}\,b_l ,
```

即“$\mathbf b$ 加权的因子组合 $\mathbf b'\mathbf f_t$ 的方差”。代入 $\mathbf b = \bv\Sigma_f^{-1}\bv\mu$ 并利用 $\bv\Sigma_f$ 对称，$\mathbf b' = \bv\mu'\bv\Sigma_f^{-1}$，于是 $\mathbf b'\bv\Sigma_f\mathbf b = \bv\mu'\bv\Sigma_f^{-1}\bv\Sigma_f\bv\Sigma_f^{-1}\bv\mu = \bv\mu'\bv\Sigma_f^{-1}\bv\mu$。因子互不相关时只剩 $k = l$ 的项，

```{math}
:enumerated: false
\Var(m_t) = \sum_{k=1}^{K} b_k^2\sigma_k^2 = \sum_{k=1}^{K} \frac{\mu_k^2}{\sigma_k^2} = \sum_{k=1}^{K} \Bigl(\frac{\mu_k}{\sigma_k}\Bigr)^2 ,
```

即各因子夏普比率的平方和：因子互不相关时，最大夏普比率的平方等于各因子夏普比率的平方和。默认参数下三个因子的月度夏普比率依次为 $0.6\%/5\% = 0.12$、$0.3\%/3\% = 0.1$、$0.4\%/3\% \approx 0.133$，于是 $\Var(m_t) \approx 0.0144 + 0.01 + 0.0178 = 0.0422$，$\text{Std}(m_t) \approx 0.205$（下方代码核对）。**特例**：$K = 1$ 时 $\Var(m_t) = (\mu/\sigma)^2$，即单个因子夏普比率的平方，与 0.6 节符号验证的输出一致。

```{code-cell} python
# 拆开讲 Var(m_t)：二次型 b'Σ_f b 等于 μ'Σ_f⁻¹μ；对角 Σ_f 下等于各因子夏普比率的平方和
sr_k = mu / np.asarray(params.factor_vol)                                  # (K,)：μ_k / σ_k
var_m_def = mkt.sdf_weights @ params.factor_cov @ mkt.sdf_weights          # (K,) @ (K, K) @ (K,) -> 标量
print("各因子夏普比率", sr_k.round(3), f"；Var(m) = {var_m_def:.4f}，Std(m) = {np.sqrt(var_m_def):.4f}")
assert np.isclose(var_m_def, np.sum(sr_k**2)) and np.isclose(var_m_def, mu @ np.linalg.solve(params.factor_cov, mu))
assert np.allclose(sr_k.round(3), [0.12, 0.1, 0.133]) and 0.0421 < var_m_def < 0.0422 and 0.205 < np.sqrt(var_m_def) < 0.206
assert 4.85 < 1 / np.sqrt(var_m_def) < 4.95                                # 0.6 节：m_t 低于均值约 4.9 个标准差才为负
```

## 0.6 代码验证

### 符号验证

先用 `sympy` 在单因子情形下验证[定理 %s](#thm-ch0-sdf) 的前两条：设 $f = \mu + \sigma u$，$u \sim N(0,1)$，$b = \mu/\sigma^2$（单因子时它们都是标量）。

```{code-cell} python
import sympy as sp
from sympy.stats import E as Ex, Normal

mu_s = sp.Symbol("mu", real=True)
sigma_s = sp.Symbol("sigma", positive=True)
u = Normal("u", 0, 1)

f_s = mu_s + sigma_s * u
b_s = mu_s / sigma_s**2
m_s = 1 - b_s * (f_s - mu_s)

E_m = sp.simplify(Ex(m_s))
E_mf = sp.simplify(Ex(m_s * f_s))
Var_m = sp.simplify(Ex((m_s - 1) ** 2))
print("E[m]   =", E_m)
print("E[m f] =", E_mf)
print("Var(m) =", Var_m, "  （即夏普比率 μ/σ 的平方）")
assert E_m == 1 and E_mf == 0 and sp.simplify(Var_m - mu_s**2 / sigma_s**2) == 0
```

再验证第 2 条不依赖于 $\bv\Sigma_f$ 为对角矩阵。证明中真正用到的只是 $\E[\mathbf f(\mathbf f-\bv\mu)'] = \bv\Sigma_f$ 与 $\mathbf b = \bv\Sigma_f^{-1}\bv\mu$，这是纯代数，取 $K=2$ 与一般的对称协方差矩阵验证。`sympy` 的 `Matrix` 与公式一样区分行、列：`mu_v` 是 $2\times 1$ 列向量，`b_v.T` 是 $1\times 2$ 行向量。

```{code-cell} python
m1, m2 = sp.symbols("mu1 mu2", real=True)
s11, s22 = sp.symbols("s11 s22", positive=True)
s12 = sp.Symbol("s12", real=True)
mu_v = sp.Matrix([m1, m2])                         # μ: 2×1
Sigma = sp.Matrix([[s11, s12], [s12, s22]])        # Σ_f: 2×2

b_v = Sigma.LUsolve(mu_v)                          # 解 Σ_f b = μ，b: 2×1
E_mf_v = mu_v - Sigma * b_v                        # E[m f] = μ − Σ_f b: 2×1
var_m = sp.simplify((b_v.T * Sigma * b_v)[0])      # b'Σ_f b: (1×2)(2×2)(2×1) = 1×1
print("E[m f] =", sp.simplify(E_mf_v).T)
print("Var(m) − μ'Σ⁻¹μ =", sp.simplify(var_m - (mu_v.T * Sigma.LUsolve(mu_v))[0]))
assert sp.simplify(E_mf_v) == sp.zeros(2, 1)
```

### 模拟验证

第 3 条涉及条件期望，用模拟数据验证。我们取 $\bv\theta = (0.01, 0, 0, 0, 0)'$，即第一个特征每高一个标准差，每月有 1% 的错误定价；用 $T = 20000$ 期、$N = 20$ 只股票的长样本，使样本矩足够接近总体矩。验证内容是：$\E[m_t] = 1$、$\E[m_t \mathbf f_t] = \mathbf 0$、无条件 $\E[m_t r_{i,t}] = 0$、管理组合 $\E[m_t r_{i,t} z_{1,i,t-1}] = \theta_1$，以及条件版本——把观测按 $z_{1,i,t-1}$ 分成五组，组内 $m_t r_{i,t}$ 的均值应当等于组内 $\alpha_{i,t}$ 的均值。

下面的代码用 `unstack("stock")` 把面板列变成 $T\times N$ 的数组（行为期、列为股票），再用广播（broadcasting）把长度为 $T$ 的 `m` 乘到每一列上：`m[:, None]` 把形状 `(T,)` 变成 `(T, 1)`，与 `(T, N)` 相乘时这一列被自动复制 $N$ 次，结果的 $(t, i)$ 元素就是 $m_t r_{i,t}$。

```{code-cell} python
long_params = MarketParams(n_stocks=20, n_periods=20_000, alpha_coef=(0.01, 0.0, 0.0, 0.0, 0.0))
long_mkt = simulate_market(long_params, seed=2026)

m = long_mkt.sdf().to_numpy()                       # (T,)
F = long_mkt.factors.to_numpy()                     # (T, K)，第 t 行为 f_t'
R = long_mkt.panel["exret"].unstack("stock").to_numpy()   # (T, N)
Z1 = long_mkt.panel["z1"].unstack("stock").to_numpy()     # (T, N)
A = long_mkt.panel["alpha"].unstack("stock").to_numpy()   # (T, N)
MR = m[:, None] * R                                 # 广播 (T, 1) * (T, N) -> (T, N)：m_t r_{i,t}
mu_long = np.asarray(long_params.factor_mean)

rows = [
    ("E[m]", m.mean(), 1.0, m.std()),
    *[(f"E[m f{k + 1}]", v, 0.0, s) for k, (v, s) in
      enumerate(zip((m[:, None] * F).mean(axis=0), (m[:, None] * F).std(axis=0)))],
    ("E[m r]", MR.mean(), 0.0, MR.mean(axis=1).std()),
    ("E[m r z1]", (MR * Z1).mean(), long_params.alpha_coef[0], (MR * Z1).mean(axis=1).std()),
]
moments = pd.DataFrame(rows, columns=["矩", "样本值", "理论值", "逐期标准差"]).set_index("矩")
moments["标准化偏差"] = (moments["样本值"] - moments["理论值"]) / (moments["逐期标准差"] / np.sqrt(len(m)))
sd_m_theory = np.sqrt(long_mkt.sdf_weights @ mu_long)          # √(b'μ) = √(μ'Σ_f⁻¹μ)
print(moments.drop(columns="逐期标准差").round(5).to_string())
print(f"Std(m)：样本值 {m.std():.5f}，理论值 {sd_m_theory:.5f}")
assert np.all(np.abs(moments["标准化偏差"]) < 3) and abs(m.std() / sd_m_theory - 1) < 0.02
```

表中“标准化偏差”是样本值与理论值之差除以标准误，标准误按“逐期标准差 $/\sqrt{T}$”计算：先把每一期的 $N$ 个观测平均，再看这些平均值的波动（原因见下文的收敛图）。代码末尾的 `assert` 保证每个矩的标准化偏差都在 3 以内，Std(m) 与理论值 $\sqrt{\bv\mu'\bv\Sigma_f^{-1}\bv\mu} = \sqrt{\mathbf b'\bv\mu}$ 相差不到 2%。条件版本按 $z_1$ 的五分位分组比较：

```{code-cell} python
from scipy.stats import norm

groups = pd.qcut(Z1.ravel(), 5, labels=["最低", "2", "3", "4", "最高"])
cond = pd.DataFrame({"m r": MR.ravel(), "alpha": A.ravel(), "组": groups}).groupby("组", observed=True).mean()
cond["差"] = cond["m r"] - cond["alpha"]
print(cond.round(5))

# 标准正态分布最高五分位的条件均值 φ(z_0.8)/0.2，乘以 θ1 即最高组 α 的理论值
alpha_top = long_params.alpha_coef[0] * norm.pdf(norm.ppf(0.8)) / 0.2
print(f"最高组 α 的理论值 {alpha_top:.4f}（最低组为其相反数）")
assert np.all(np.abs(cond["差"]) < 0.002)
assert np.allclose(cond["alpha"].iloc[[0, -1]], [-alpha_top, alpha_top], atol=5e-4)
```

五个组中，$m_t r_{i,t}$ 的组内均值与 $\alpha_{i,t}$ 的组内均值相差都在千分之二以内（由 `assert` 保证），而组间 $\alpha$ 的差别达到 $\pm 1.4\%$ 左右——这个数不依赖种子：标准正态分布最高五分位的条件均值约为 1.40，乘以 $\theta_1 = 0.01$ 即得。条件定价误差由 $\bv\theta' \mathbf z_{i,t-1}$ 决定，这正是定理第 3 条。

最后，用默认参数核对 $m_t$ 的标准差等于因子的最大夏普比率（与随机抽取的 20 万个因子组合比较），并算出 $m_t$ 取负值的概率。由于 $m_t \sim N(1, \bv\mu'\bv\Sigma_f^{-1}\bv\mu)$，$P(m_t < 0) = \Phi\left(-1/\text{Std}(m_t)\right)$，其中 $\Phi$ 是标准正态分布函数。代码中 `einsum("nk,kl,nl->n", w, Sigma_d, w)` 对 `w` 的每一行 $\mathbf w_n'$ 计算二次型 $\mathbf w_n'\bv\Sigma_f\mathbf w_n$，即组合方差。

(ch0-unpack-sharpe)=
**拆开讲这几个式子。** $P(m_t < 0) = \Phi(-1/\text{Std}(m_t))$ 是一次标准化：$(m_t - 1)/\text{Std}(m_t)$ 服从标准正态分布，$m_t < 0$ 等价于 $(m_t-1)/\text{Std}(m_t) < -1/\text{Std}(m_t)$。默认参数下 $1/\text{Std}(m_t) \approx 1/0.205 \approx 4.9$，$m_t$ 要比均值低约 4.9 个标准差才会取负值。组合权重 $\mathbf w_n \in \Real^K$ 的第 $k$ 个分量 $w_{n,k}$ 是组合在因子 $k$ 上的头寸；它是 $K$ 个因子组成的组合的权重，与记号表中 $N$ 只股票组成的组合的权重 $\mathbf w_t \in \Real^N$ 不是同一个量；组合的期望超额收益 $\mathbf w_n'\bv\mu = \sum_k w_{n,k}\mu_k$ 与方差 $\mathbf w_n'\bv\Sigma_f\mathbf w_n = \sum_k\sum_l w_{n,k}\,\Sigma_{kl}\,w_{n,l}$（二次型的展开见[拆解](#ch0-unpack-varm)）都是标量，夏普比率 $\mathbf w_n'\bv\mu/\sqrt{\mathbf w_n'\bv\Sigma_f\mathbf w_n}$ 是期望超额收益除以波动率。把 $\mathbf w_n$ 整体放大或缩小（乘以正数），夏普比率不变，所以只有权重的方向要紧。切点组合 $\mathbf w \propto \bv\Sigma_f^{-1}\bv\mu = \mathbf b$（$\propto$ 表示“成比例”），对角情形下 $w_k \propto \mu_k/\sigma_k^2$，它的夏普比率为 $\mathbf b'\bv\mu/\sqrt{\mathbf b'\bv\Sigma_f\mathbf b} = \bv\mu'\bv\Sigma_f^{-1}\bv\mu/\sqrt{\bv\mu'\bv\Sigma_f^{-1}\bv\mu} = \sqrt{\bv\mu'\bv\Sigma_f^{-1}\bv\mu} = \text{Std}(m_t)$。**数据矩阵**：代码中 `w` 的形状为 `(n, K)`，第 $n$ 行是 $\mathbf w_n'$；20 万个组合共用同一个 $\bv\mu$ 与 $\bv\Sigma_f$。

```{code-cell} python
Sigma_d = params.factor_cov                                               # Σ_f: (K, K)
sd_m = np.sqrt(mkt.sdf_weights @ mu)                                      # √(b'μ)：(K,) @ (K,) -> 标量

w = np.random.default_rng(3).standard_normal((200_000, len(mu)))          # 随机组合权重，每行一个 w_n'：(n, K)
sharpe = (w @ mu) / np.sqrt(np.einsum("nk,kl,nl->n", w, Sigma_d, w))     # (n,)：w_n'μ / √(w_n'Σ_f w_n)
w_tan = np.linalg.solve(Sigma_d, mu)                                      # 切点组合 ∝ Σ_f⁻¹μ：(K,)
sr_tan = (w_tan @ mu) / np.sqrt(w_tan @ Sigma_d @ w_tan)
p_neg = norm.cdf(-1 / sd_m)
print(f"Std(m) = {sd_m:.4f}，切点组合夏普比率 {sr_tan:.4f}，随机组合中的最大夏普比率 {sharpe.max():.4f}")
print(f"P(m < 0) = Φ(−1/Std(m)) = {p_neg:.2e}")
assert np.isclose(sd_m, sr_tan) and sharpe.max() <= sr_tan + 1e-12 and p_neg < 1e-5
```

### 收敛到真值

样本矩只是总体矩的估计。下图展示随着样本期数 $T$ 增大，样本矩如何收敛到理论值。左图是 $m_t f_{1,t}$ 的样本均值（理论值 0），右图是管理组合矩 $m_t r_{i,t} z_{1,i,t-1}$ 的样本均值（理论值 $\theta_1 = 0.01$）。灰色区域是理论值加减两倍标准误，标准误按 $\text{Std}/\sqrt{T}$ 计算；右图先对每一期求 $N$ 只股票的横截面平均，再用这些平均值的标准差除以 $\sqrt{T}$，这样同期股票之间通过因子产生的相关会自动计入（[习题 %s](#ex-ch0-theta) 中会看到，忽略这种相关会严重低估标准误）。

```{code-cell} python
#| label: fig-ch0-convergence
#| caption: 样本矩随样本期数 T 收敛到理论值（对数横轴）
import matplotlib.pyplot as plt

T_grid = np.unique(np.geomspace(20, len(m), 60).astype(int))
mf1 = m * F[:, 0]                                    # (T,)
mrz = (MR * Z1).mean(axis=1)                         # (T, N) -> (T,)：每期横截面平均
run_mf1 = np.cumsum(mf1)[T_grid - 1] / T_grid        # 前 T 期的累计均值
run_mrz = np.cumsum(mrz)[T_grid - 1] / T_grid
se_mf1 = mf1.std() / np.sqrt(T_grid)
se_mrz = mrz.std() / np.sqrt(T_grid)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharex=True)
for ax, run, se, truth, title in [
    (axes[0], run_mf1, se_mf1, 0.0, "m·f1 的样本均值"),
    (axes[1], run_mrz, se_mrz, 0.01, "m·r·z1 的样本均值"),
]:
    ax.fill_between(T_grid, truth - 2 * se, truth + 2 * se, color="0.85", label="理论值 ± 2 倍标准误")
    ax.axhline(truth, color="0.3", lw=1, ls="--", label=f"理论值 {truth:g}")
    ax.plot(T_grid, run, color="C0", label="样本值")
    ax.set_xscale("log")
    ax.set_xlabel("样本期数 T")
    ax.set_title(title)
for ax in axes:
    ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
plt.show()

z_mf1 = (run_mf1 - 0.0) / se_mf1                     # 每个 T 上的标准化偏差
z_mrz = (run_mrz - 0.01) / se_mrz
for name, zz, run, truth in [("左图", z_mf1, run_mf1, 0.0), ("右图", z_mrz, run_mrz, 0.01)]:
    early = np.abs(run - truth)[T_grid <= 200].max()
    late = np.abs(run - truth)[T_grid >= 5000].max()
    print(f"{name}：{len(T_grid)} 个 T 中有 {np.sum(np.abs(zz) > 2)} 个落在带外；"
          f"全样本（T = {len(m)}）的标准化偏差 {zz[-1]:.2f}；"
          f"|样本值 − 理论值| 的最大值：T ≤ 200 时 {early:.4f}，T ≥ 5000 时 {late:.4f}")
    assert late < early / 5 and abs(zz[-1]) < 1

out_right = T_grid[np.abs(z_mrz) > 2]                  # 右图越出带的格点
print("右图越出带的 T：", out_right)
assert np.sum(np.abs(z_mf1) > 2) == 0 and np.all(run_mf1[T_grid < 200] < 0)
assert len(out_right) == 1 and 2000 < out_right[0] < 3500                 # 右图恰好一个格点越出带
assert z_mrz[T_grid == out_right[0]][0] > 2                                # 越出的是带的上沿
assert np.all(z_mrz[(T_grid > 2000) & (T_grid < 3500)] > 1)               # 右图 2000～3500 持续偏高
z_mid = z_mf1[(T_grid >= 1200) & (T_grid <= 15000)]                        # 左图中段的标准化偏差
assert np.all(z_mid > 0.5) and np.mean(z_mid > 1) > 0.8                    # 持续偏高，大多数格点在带的上半部分
```

两个样本均值都随 $T$ 增大而收敛到理论值：样本值与理论值的最大偏离，从 $T \leq 200$ 的小样本到 $T \geq 5000$ 的大样本缩小到五分之一以下（具体数值见图下输出），与偏离按 $1/\sqrt{T}$ 收窄的速度相符——$T$ 扩大 25 倍，$1/\sqrt{T}$ 缩小到五分之一。但两条路径并不是一直贴着理论值：左图的路径在 $T < 200$ 的小样本中始终低于理论值 0，而在 $T$ 约 1200～15000 的很长一段区间里又持续高于 0，其中绝大多数格点位于带的上半部分（标准化偏差大于 1），但始终没有越出灰色带；右图的路径在 $T$ 约 2000～3500 之间持续偏高，其中恰好一个格点越出了带的上沿，随后回到带内并逐渐贴近理论值。全样本的标准化偏差都小于 1（以上描述都由代码中的 `assert` 守护）。路径在某些区段偏离较多、甚至越出带，是预期之中的现象，并不说明理论有误。灰色带是逐点的“理论值 ± 2 倍标准误”，只保证在任意一个固定的 $T$ 上，样本均值落在带外的概率约为 5%；它不是同时成立的置信带，不保证整条路径都在带内。而同一条路径上相邻的累计均值共享绝大部分数据、高度序列相关，一旦在某个 $T$ 偏高或偏低，就会持续一段时间（左图先持续偏低、后持续偏高就是这样），所以整条路径在某些区段越出逐点的带，比 60 个独立抽样时常见得多。本节开头的表格与五分位分组的结果也都与理论一致。由此也可以读出本书所有模拟验证的基本逻辑：**单次模拟只给出带有抽样误差的样本值，判断“是否恢复了真值”时，要把偏离与标准误相比较，而不是要求两者完全相等。**

