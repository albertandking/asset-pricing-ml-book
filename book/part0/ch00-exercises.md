---
kernelspec:
  name: python3
  display_name: Python 3
numbering:
  enumerator: '0.%s'
  exercise:
    template: 习题 %s
---

# 本章小结、习题与延伸阅读

**本页准备。** 本书每个页面的代码在独立的内核中运行，前面页面定义的变量在这里不可用。下面的单元导入本页用到的模块，并重新定义上一页“模拟实验”一节的两个函数 `interaction_design` 与 `ols`。

```{code-cell} python
import numpy as np
import pandas as pd

from apsim import MarketParams, simulate_market

K = MarketParams().n_factors                                # 因子个数 K = 3，与 0.4 节相同


# 以下两个函数与上一页“模拟实验”一节中的定义相同
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
```

## 本章小结

1. 本书用 Jupyter Book 2 构建，`sh scripts/build.sh --clean` 会清除执行缓存并执行全部代码；全书的数字与图形都可以在本地复现，正文引用的数字由代码中的 `assert` 守护。
2. 基本概念（0.2 节）：资产定价研究超额收益；本章的因子是可交易组合的超额收益；载荷 $\bv\beta_{i,t}$ 是股票对因子的敏感度，可以用回归斜率估计；风险溢价 $\bv\lambda$ 是每单位载荷的补偿，$\E[r_i] = \bv\beta_i'\bv\lambda$；本章模型中载荷由特征决定，$\bv\beta_{i,t} = \bv\Gamma'\tilde{\mathbf z}_{i,t-1}$；定价误差 $\alpha$ 是载荷解释不了的期望收益；SDF $m$ 衡量各状态下一块钱的价值，$P = \E[mx]$。
3. 全书记号见[记号表](../notation.md)。标量用斜体，向量用小写粗体（列向量），矩阵用大写粗体；代码数组“观测在行”，一行是公式中列向量的转置。时点约定是：下标 $t$ 表示第 $t$ 期末已知；面板中第 $t$ 期的一行存放期初特征 $\mathbf z_{i,t-1}$ 与本期收益 $r_{i,t}$，特征比收益早一期。`period` 从 1 开始编号，`stock` 从 0 开始编号。
4. 模拟市场由式 {eq}`eq-ch0-factor`～{eq}`eq-ch0-sdf` 定义：独立同分布的因子、平稳的 AR(1) 特征、由特征决定的载荷 $\bv\beta_{i,t} = \bv\Gamma' \tilde{\mathbf z}_{i,t-1}$ 与定价误差 $\alpha_{i,t} = \bv\theta' \mathbf z_{i,t-1}$；按横截面堆叠即 $\mathbf r_t = \mathbf Z_{t-1}\bv\theta + \tilde{\mathbf Z}_{t-1}\bv\Gamma\mathbf f_t + \bv\varepsilon_t$。
5. SDF $m_t = 1 - \mathbf b'(\mathbf f_t - \bv\mu)$，$\mathbf b = \bv\Sigma_f^{-1}\bv\mu$ 为因子正确定价，并使 $\E[m_t r_{i,t} \mid \mathbf z_{i,t-1}] = \alpha_{i,t}$（[定理 %s](#thm-ch0-sdf)）。证明的关键是 $\mathbf f_t$、$\varepsilon_{i,t}$ 与 $\mathbf z_{i,t-1}$ 相互独立。模拟器中因子相互独立，$\mathbf b$ 化为 $b_k = \mu_k / \sigma_k^2$。
6. 存在错误定价时，单只股票的无条件定价误差仍可能为 0，错误定价要通过以特征加权的管理组合才能看到（[推论 %s](#cor-ch0-managed)）。
7. 交互回归可以恢复 $\bv\Gamma$，其系数向量是 $\operatorname{vec}(\bv\Gamma')$，对应 NumPy 的行优先展开；存在错误定价而不加入特征本身时，估计有遗漏变量偏误，其大小由样本因子路径决定，大样本下方向趋于 SDF 系数 $\mathbf b$ 的方向。
8. 检验“是否恢复真值”时，要把偏离与标准误比较，必要时用多次重复模拟检查标准误本身是否正确。

## 习题

### 概念题

:::{exercise} 时点错位
:label: ex-ch0-timing
某研究者在第 $t$ 期末按当时的市值把股票分为五组，然后计算各组在第 $t$ 期的收益，发现大市值组的收益显著高于小市值组，于是断言“大市值股票有更高的期望收益”。

1. 这个做法错在哪里？用本章的时点约定说明。
2. 在本章的模拟器中犯同样的错误（用 $\mathbf z_{i,t}$ 代替 $\mathbf z_{i,t-1}$），会产生虚假的显著结果吗？为什么真实数据中会？
:::

:::{solution} ex-ch0-timing
:class: dropdown
1. 第 $t$ 期末的市值是价格 $P_{i,t}$ 乘以股本，而 $P_{i,t}$ 已经包含了第 $t$ 期的收益。按它分组，再看同一期的收益，相当于用结果去解释结果：第 $t$ 期上涨的股票市值变大、下跌的变小，于是“小市值组”机械地集中了本期下跌的股票。正确的做法是按第 $t-1$ 期末的市值分组，计算第 $t$ 期的收益；在本书的面板中，就是用同一行的 `z` 列解释 `exret` 列。研究者发现的“大市值组收益更高”，正是这种机械效应的结果，与期望收益无关。
2. 不会。模拟器中 $\mathbf z_{i,t} = \phi \mathbf z_{i,t-1} + \sqrt{1-\phi^2} \mathbf e_{i,t}$，更新冲击 $\mathbf e_{i,t}$ 与 $r_{i,t}$ 独立，所以 $\mathbf z_{i,t}$ 只是 $\mathbf z_{i,t-1}$ 的一个含噪声的版本，错用它只会把真实的关系衰减为原来的 $\phi$ 倍（见“常见误区”一节的推导与代码）。真实数据中，特征的更新与当期收益相关（市值、动量、换手率、波动率等都直接由当期价格或成交量计算），所以错用同期特征会制造虚假的预测力。这也提醒我们：模拟只能检验我们放进模型的机制。
:::

:::{exercise} 无条件定价误差为零，是否说明没有错误定价？
:label: ex-ch0-uncond
设 $\bv\theta = (0.01, 0, 0, 0, 0)'$。某研究者用真实的 $m_t$ 对每只股票计算 $T^{-1}\sum_t m_t r_{i,t}$，发现在 $T$ 很大时它们都趋于 0，于是断言“该 SDF 为全部股票正确定价，市场中不存在错误定价”。这一断言对吗？应当怎样设计检验资产？
:::

:::{solution} ex-ch0-uncond
:class: dropdown
不对。由[推论 %s](#cor-ch0-managed)，$\E[m_t r_{i,t}] = \bv\theta' \E[\mathbf z_{i,t-1}] = 0$：在本模拟器中，每只股票的特征都围绕 0 波动，错误定价时正时负，长期平均恰好抵消，所以无条件定价误差为 0。但条件定价误差 $\E[m_t r_{i,t} \mid \mathbf z_{i,t-1}] = 0.01\, z_{1,i,t-1}$ 并不为 0：按 $z_1$ 做多高、做空低的策略可以获得 SDF 不能解释的收益。

要让无条件矩检验看到条件错误定价，应当把检验资产换成以期初信息加权的管理组合，例如 $r_{i,t} z_{1,i,t-1}$，或者按 $z_1$ 排序构造的组合。由推论，$\E[m_t r_{i,t} \mathbf z_{i,t-1}] = \bv\theta \neq \mathbf 0$。这正是第 5 章“条件模型与管理组合”的核心思想：条件矩 $\E[m_t r_{i,t} \mid \mathbf z_{i,t-1}] = 0$ 等价于对所有 $\mathbf z_{i,t-1}$ 的（标量）函数 $g$ 都有 $\E[m_t r_{i,t}\, g(\mathbf z_{i,t-1})] = 0$，检验时只能选有限个 $g$，选择决定了检验能看到什么。（真实数据中，特征通常有持续的个股差异，例如某些股票长期是小市值，此时单只股票的无条件定价误差一般不为 0，但同样的论证仍然成立。）
:::

### 推导题

:::{exercise} 时间序列回归的截距
:label: ex-ch0-ts
对固定的股票 $i$，考虑总体意义上的时间序列回归

```{math}
:enumerated: false
r_{i,t} = a_i + \mathbf h_i' \mathbf f_t + u_{i,t}, \qquad \E[u_{i,t}] = 0,\ \Cov(\mathbf f_t, u_{i,t}) = \mathbf 0 ,
```

其中截距 $a_i$ 是标量，斜率 $\mathbf h_i \in \Real^K$。设 $\bv\theta \neq \mathbf 0$，且特征处于平稳分布。证明：斜率 $\mathbf h_i = \E[\bv\beta_{i,t}]$，截距 $a_i = \E[\alpha_{i,t}]$。注意载荷 $\bv\beta_{i,t}$ 是时变的，而回归只有一个常数斜率；说明证明在哪一步用到了 $\mathbf z_{i,t-1}$ 与 $\mathbf f_t$ 的独立性。在本模拟器中 $a_i$ 等于多少？这对用时间序列回归检验定价模型意味着什么？
:::

:::{solution-start} ex-ch0-ts
:class: dropdown
:::

总体最小二乘的斜率与截距为 $\mathbf h_i = \Var(\mathbf f_t)^{-1}\Cov(\mathbf f_t, r_{i,t})$（$K\times K$ 乘 $K\times 1$），$a_i = \E[r_{i,t}] - \mathbf h_i'\bv\mu$。把 $r_{i,t} = \alpha_{i,t} + \bv\beta_{i,t}' \mathbf f_t + \varepsilon_{i,t}$ 代入，逐项计算。

**协方差。** $\Cov(\mathbf f_t, r_{i,t}) = \Cov(\mathbf f_t, \alpha_{i,t}) + \Cov(\mathbf f_t, \bv\beta_{i,t}' \mathbf f_t) + \Cov(\mathbf f_t, \varepsilon_{i,t})$，每一项都是 $K\times 1$ 向量。

- $\alpha_{i,t}$ 是 $\mathbf z_{i,t-1}$ 的函数，与 $\mathbf f_t$ 独立，故 $\Cov(\mathbf f_t, \alpha_{i,t}) = \mathbf 0$；$\varepsilon_{i,t}$ 与 $\mathbf f_t$ 独立，故第三项为 $\mathbf 0$。
- 第二项中 $\bv\beta_{i,t}' \mathbf f_t = \mathbf f_t' \bv\beta_{i,t}$ 是标量，

  ```{math}
  :enumerated: false
  \Cov(\mathbf f_t, \mathbf f_t'\bv\beta_{i,t}) = \E[\mathbf f_t \mathbf f_t' \bv\beta_{i,t}] - \E[\mathbf f_t]\,\E[\mathbf f_t'\bv\beta_{i,t}] .
  ```

  由于 $\bv\beta_{i,t}$ 是 $\mathbf z_{i,t-1}$ 的函数、与 $\mathbf f_t$ 独立，期望可以分开：$\E[\mathbf f_t \mathbf f_t' \bv\beta_{i,t}] = \E[\mathbf f_t \mathbf f_t']\,\E[\bv\beta_{i,t}] = (\bv\Sigma_f + \bv\mu\bv\mu')\,\E[\bv\beta_{i,t}]$，$\E[\mathbf f_t'\bv\beta_{i,t}] = \bv\mu'\E[\bv\beta_{i,t}]$。于是该项等于 $(\bv\Sigma_f + \bv\mu\bv\mu')\E[\bv\beta_{i,t}] - \bv\mu\bv\mu'\E[\bv\beta_{i,t}] = \bv\Sigma_f\, \E[\bv\beta_{i,t}]$。

因此 $\mathbf h_i = \bv\Sigma_f^{-1}\bv\Sigma_f\,\E[\bv\beta_{i,t}] = \E[\bv\beta_{i,t}]$。

**截距。** $\E[r_{i,t}] = \E[\alpha_{i,t}] + \E[\bv\beta_{i,t}' \mathbf f_t] + 0 = \E[\alpha_{i,t}] + \E[\bv\beta_{i,t}]'\bv\mu$（同样用到独立性）。所以

```{math}
:enumerated: false
a_i = \E[\alpha_{i,t}] + \E[\bv\beta_{i,t}]'\bv\mu - \E[\bv\beta_{i,t}]'\bv\mu = \E[\alpha_{i,t}] .
```

独立性用在两处：把 $\E[\mathbf f_t \mathbf f_t'\bv\beta_{i,t}]$ 与 $\E[\bv\beta_{i,t}' \mathbf f_t]$ 拆成乘积，以及 $\Cov(\mathbf f_t, \alpha_{i,t}) = \mathbf 0$。如果特征能预测因子收益（$\mathbf z$ 与 $\mathbf f$ 相关），这两步都不成立，常数斜率的回归会把载荷的时变与因子均值的时变混在一起，截距不再等于平均定价误差。另外注意，回归残差 $u_{i,t} = \alpha_{i,t} - \E[\alpha_{i,t}] + (\bv\beta_{i,t} - \E[\bv\beta_{i,t}])'\mathbf f_t + \varepsilon_{i,t}$ 与 $\mathbf f_t$ 不相关，但并不独立于 $\mathbf f_t$（它的方差依赖于 $\mathbf f_t$），这影响标准误，不影响上面的总体系数。

**在本模拟器中**，$\E[\bv\beta_{i,t}] = \bv\Gamma'\E[\tilde{\mathbf z}_{i,t-1}] = \bv\Gamma'(1, 0, \dots, 0)'$，即 $\bv\Gamma$ 第 0 行的转置；$a_i = \bv\theta'\E[\mathbf z_{i,t-1}] = 0$。也就是说，即使存在错误定价，每只股票时间序列回归的截距在总体上也是 0，GRS 一类的时间序列检验（第 9 章）检验的只是平均定价误差，看不到这里的条件错误定价——与[习题 %s](#ex-ch0-uncond) 的结论一致。

在有限样本中，股票 $i$ 的特征路径是一次具体的实现，样本截距近似等于该路径上的平均定价误差 $T^{-1}\sum_t \alpha_{i,t}$。下面的代码同时验证总体结论与这一有限样本含义：取 $\phi = 0.98$，使各股票的特征路径均值有明显差异，并把异质性波动 $\sigma_\varepsilon$ 降为 0.05 以减小噪声。代码对 $N$ 只股票一次完成 $N$ 个时间序列回归：`lstsq` 的右端是 $T\times N$ 矩阵时，对每一列分别求解，系数矩阵的第 $i$ 列为 $(a_i, \mathbf h_i')'$。

```{code-cell} python
ts_params = MarketParams(n_stocks=300, n_periods=600, char_persistence=0.98, idio_vol=0.05,
                         alpha_coef=(0.02, 0.0, 0.0, 0.0, 0.0))
ts = simulate_market(ts_params, seed=7)
F_ts = ts.factors.to_numpy()                                # (T, K)
R_ts = ts.panel["exret"].unstack("stock").to_numpy()        # (T, N)
A_ts = ts.panel["alpha"].unstack("stock").to_numpy()        # (T, N)
design_ts = np.column_stack([np.ones(len(F_ts)), F_ts])     # (T, 1+K)
coef_ts, *_ = np.linalg.lstsq(design_ts, R_ts, rcond=None)  # (1+K, N)：第 i 列为 (a_i, h_i')'
a_hat, h_hat = coef_ts[0], coef_ts[1:]                      # (N,)、(K, N)
alpha_path_mean = A_ts.mean(axis=0)                         # (N,)

print("斜率的横截面平均", h_hat.mean(axis=1).round(4), "；理论值 E[β] =", ts.gamma[0])
print(f"截距的横截面平均 {a_hat.mean():.5f}（理论值 E[α] = 0）")
slope, intercept = np.polyfit(alpha_path_mean, a_hat, 1)
corr_ts = np.corrcoef(alpha_path_mean, a_hat)[0, 1]
print(f"截距对路径平均 α 的回归：斜率 {slope:.3f}（理论值 1），截距 {intercept:.5f}（理论值 0），"
      f"相关系数 {corr_ts:.3f}")
assert np.allclose(h_hat.mean(axis=1), ts.gamma[0], atol=0.05) and abs(a_hat.mean()) < 0.002
assert abs(slope - 1) < 0.1 and abs(intercept) < 0.002 and corr_ts > 0.9
```

截距的横截面平均接近 0，斜率的横截面平均接近 $\bv\Gamma$ 第 0 行的转置；而单只股票的样本截距几乎一对一地反映了它在样本期内的平均定价误差：截距对路径平均 $\alpha$ 回归的斜率接近 1，相关系数高于 0.9（均由 `assert` 保证）。
:::{solution-end}
:::

:::{exercise} 从 SDF 到 β 定价
:label: ex-ch0-beta
利用[定理 %s](#thm-ch0-sdf) 第 3 条与条件协方差的分解 $\E[m_t r_{i,t} \mid \mathbf z] = \E[m_t \mid \mathbf z]\,\E[r_{i,t} \mid \mathbf z] + \Cov(m_t, r_{i,t} \mid \mathbf z)$（$\mathbf z$ 是 $\mathbf z_{i,t-1}$ 的简写），证明

```{math}
:enumerated: false
\E[r_{i,t} \mid \mathbf z_{i,t-1}] = \alpha_{i,t} + \bv\beta_{i,t}'\bv\lambda, \qquad \bv\lambda = \bv\mu .
```

解释为什么因子的风险溢价恰好等于因子的均值，并用模拟数据验证：混合回归 $r_{i,t}$ 对 $\tilde{\mathbf z}_{i,t-1}$ 的系数应为 $\bv\Gamma\bv\mu + (0, \bv\theta')'$。
:::

:::{solution-start} ex-ch0-beta
:class: dropdown
:::

由定理，左端 $\E[m_t r_{i,t} \mid \mathbf z] = \alpha_{i,t}$；又因 $\mathbf f_t$ 与 $\mathbf z$ 独立，$\E[m_t \mid \mathbf z] = 1$。只需计算条件协方差。以 $\mathbf z$ 为条件时，$\bv\beta_{i,t}$ 是常数向量，$m_t - \E[m_t \mid \mathbf z] = -\mathbf b'(\mathbf f_t - \bv\mu)$，$r_{i,t} - \E[r_{i,t} \mid \mathbf z] = \bv\beta_{i,t}'(\mathbf f_t - \bv\mu) + \varepsilon_{i,t}$，二者都是标量。把第二个写成 $(\mathbf f_t - \bv\mu)'\bv\beta_{i,t} + \varepsilon_{i,t}$，由于 $(\mathbf f_t, \varepsilon_{i,t})$ 与 $\mathbf z$ 独立、$\mathbf f_t$ 与 $\varepsilon_{i,t}$ 独立，

```{math}
:enumerated: false
\Cov(m_t, r_{i,t} \mid \mathbf z) = -\underbrace{\mathbf b'}_{1\times K}\, \underbrace{\E\left[(\mathbf f_t - \bv\mu)(\mathbf f_t - \bv\mu)'\right]}_{K\times K} \underbrace{\bv\beta_{i,t}}_{K\times 1} - \mathbf b'\,\E\left[(\mathbf f_t-\bv\mu)\varepsilon_{i,t}\right] = -\mathbf b'\bv\Sigma_f\, \bv\beta_{i,t} = -\bv\mu'\bv\beta_{i,t} ,
```

最后一步用到 $\mathbf b'\bv\Sigma_f = (\bv\Sigma_f\mathbf b)' = \bv\mu'$（$\bv\Sigma_f$ 对称）。代入分解式：$\alpha_{i,t} = 1 \cdot \E[r_{i,t} \mid \mathbf z] - \bv\mu'\bv\beta_{i,t}$，即 $\E[r_{i,t} \mid \mathbf z] = \alpha_{i,t} + \bv\beta_{i,t}'\bv\mu$。

一般地，β 定价中的风险溢价为 $\bv\lambda = -\Cov(\mathbf f_t, m_t)/\E[m_t] = \bv\Sigma_f \mathbf b / \E[m_t]$（$K\times 1$）。当因子本身是超额收益、且被 $m_t$ 正确定价时，把这一关系用于因子自身（其对自己的载荷矩阵为 $\mathbf I_K$、定价误差为 $\mathbf 0$），立即得到 $\bv\lambda = \E[\mathbf f_t] = \bv\mu$。第 4 章会在更一般的条件下讨论 SDF 表示与 β 表示的等价性。

验证：由 $\E[r_{i,t} \mid \mathbf z] = \bv\theta'\mathbf z + \tilde{\mathbf z}'\bv\Gamma\bv\mu = \tilde{\mathbf z}'\left[\bv\Gamma\bv\mu + (0, \bv\theta')'\right]$ 与 $\E[\tilde{\mathbf z}\tilde{\mathbf z}'] = \mathbf I_{L+1}$，混合回归 $r_{i,t}$ 对 $\tilde{\mathbf z}_{i,t-1}$ 的总体系数为 $\E[\tilde{\mathbf z}\tilde{\mathbf z}']^{-1}\E[\tilde{\mathbf z} r_{i,t}] = \bv\Gamma\bv\mu + (0, \bv\theta')'$，是 $L+1$ 维向量。

```{code-cell} python
bp_params = MarketParams(n_stocks=100, n_periods=5_000, alpha_coef=(0.01, -0.005, 0.0, 0.0, 0.0))
bp = simulate_market(bp_params, seed=12)
Xb = np.column_stack([np.ones(len(bp.panel)), bp.panel[[f"z{j}" for j in range(1, 6)]].to_numpy()])   # (TN, L+1)
yb = bp.panel["exret"].to_numpy()                                                                   # (TN,)
coef_b = np.linalg.lstsq(Xb, yb, rcond=None)[0]
Xb3, yb3 = Xb.reshape(5_000, 100, -1), yb.reshape(5_000, 100)                                       # 按期切开：(T, N, L+1)、(T, N)
per_b = np.array([np.linalg.lstsq(Xb3[t], yb3[t], rcond=None)[0] for t in range(5_000)])            # (T, L+1)
se_b = per_b.std(axis=0, ddof=1) / np.sqrt(5_000)
truth_b = bp.gamma @ np.asarray(bp_params.factor_mean) + np.r_[0.0, bp.theta]                       # Γμ + (0, θ')'：(L+1,)
print(pd.DataFrame({"理论值 Γμ + (0, θ')'": truth_b, "估计值": coef_b, "逐期回归标准误": se_b,
                    "标准化偏差": (coef_b - truth_b) / se_b},
                   index=["常数", "z1", "z2", "z3", "z4", "z5"]).round(5))
assert np.all(np.abs(coef_b - truth_b) < 2 * se_b)
```

各系数与理论值的偏离都在两倍逐期回归标准误以内（由 `assert` 保证，标准化偏差见上表）。
:::{solution-end}
:::

### 编程题

:::{exercise} 特征持续性与 Γ 的估计误差
:label: ex-ch0-phi
取 $N = 200$、$T = 120$，令 $\phi \in \{0, 0.5, 0.9, 0.99\}$，对每个 $\phi$ 用 20 个不同的种子重复模拟，用交互回归估计 $\bv\Gamma$，计算估计误差的均方根 $\left[\frac{1}{(L+1)K}\sum_{j,k}(\hat\Gamma_{j,k} - \Gamma_{j,k})^2\right]^{1/2}$ 的平均值。持续性对估计误差有多大影响？把 $N$ 降到 10 再做一次，结论有变化吗？给出解释。
:::

:::{solution-start} ex-ch0-phi
:class: dropdown
:::

```{code-cell} python
def gamma_rmse(n_stocks, phi, n_rep, base_seed):
    errs = []
    for r in range(n_rep):
        rep = simulate_market(MarketParams(n_stocks=n_stocks, n_periods=120, char_persistence=phi),
                              seed=base_seed + r)
        Xr, _ = interaction_design(rep)
        g_hat = np.linalg.lstsq(Xr, rep.panel["exret"].to_numpy(), rcond=None)[0].reshape(-1, K)   # vec(Γ̂') -> Γ̂
        errs.append(np.sqrt(np.mean((g_hat - rep.gamma) ** 2)))
    errs = np.asarray(errs)
    return errs.mean(), errs.std(ddof=1) / np.sqrt(n_rep)


rows = []
for n_stocks, n_rep, base in [(200, 20, 3000), (10, 50, 4000)]:
    for phi_ in (0.0, 0.5, 0.9, 0.99):
        mean_err, se_err = gamma_rmse(n_stocks, phi_, n_rep, base)
        rows.append({"N": n_stocks, "φ": phi_, "重复次数": n_rep, "平均 RMSE": mean_err, "其标准误": se_err})
rmse = pd.DataFrame(rows)
print(rmse.round(4).to_string(index=False))
big, small_n = rmse[rmse["N"] == 200], rmse[rmse["N"] == 10]
spread = big["平均 RMSE"].max() - big["平均 RMSE"].min()
assert spread / big["平均 RMSE"].min() < 0.05 and spread < 2 * big["其标准误"].max()
e0, e99 = small_n["平均 RMSE"].iloc[0], small_n["平均 RMSE"].iloc[-1]
assert e99 > 1.15 * e0 and e99 - e0 > 3 * np.hypot(*small_n["其标准误"].iloc[[0, -1]])
```

$N = 200$ 时，四种持续性下的平均误差几乎相同：最大值与最小值相差不到 5%，也小于两倍蒙特卡洛标准误（由 `assert` 保证）。原因是：在 $\varepsilon_{i,t}$ 与解释变量独立的条件下，最小二乘估计的条件协方差为 $\sigma_\varepsilon^2 (\mathbf X'\mathbf X)^{-1}$，而由式 {eq}`eq-ch0-kron` 与混合乘积规则，$\mathbf X'\mathbf X = \sum_t \left(\sum_i \tilde{\mathbf z}_{i,t-1}\tilde{\mathbf z}_{i,t-1}'\right) \otimes \mathbf f_t\mathbf f_t'$，这是 $(L+1)\times(L+1)$ 矩阵与 $K\times K$ 矩阵的 Kronecker 积之和，为 $(L+1)K\times(L+1)K$ 矩阵。特征的边际分布始终是 $N(\mathbf 0, \mathbf I_L)$，与 $\phi$ 无关；$\phi$ 只决定同一只股票的特征在不同期之间是否相似。当 $N$ 足够大时，每一期的横截面矩 $N^{-1}\sum_i \tilde{\mathbf z}_{i,t-1}\tilde{\mathbf z}_{i,t-1}'$ 都接近 $\mathbf I_{L+1}$，$\mathbf X'\mathbf X$ 也就几乎不受 $\phi$ 影响。

$N = 10$ 时，$\phi = 0.99$ 的误差明显变大：比 $\phi = 0$ 时高出 15% 以上，差值超过三倍蒙特卡洛标准误（由 `assert` 保证）。此时每一期只有 10 个特征观测，而且它们在整个样本期内几乎不变，样本矩 $\sum_t\sum_i \tilde{\mathbf z}_{i,t-1}\tilde{\mathbf z}_{i,t-1}'$ 实际上只由少数几个独立抽样决定，可能明显偏离 $TN \cdot \mathbf I_{L+1}$，某些特征方向上的信息很少；$(\mathbf X'\mathbf X)^{-1}$ 的期望因此变大。换句话说，持续性降低的是“有效样本量”，当横截面足够大时这一影响被横截面上的独立变化所弥补。

需要强调的是，本题中持续性不会造成偏误，只影响方差。持续性造成偏误的典型情形是预测回归：被解释变量对一个高度持续的滞后变量回归，而该变量的更新冲击与收益冲击相关，此时最小二乘斜率有小样本偏误 [@stambaugh1999predictive]。本模拟器中 $\mathbf e_{i,t}$ 与 $\varepsilon_{i,t}$ 独立，所以不会出现这种偏误。第 13 章讨论这一问题。
:::{solution-end}
:::

:::{exercise} 用管理组合恢复 θ
:label: ex-ch0-theta
取默认的 $N = 500$、$T = 240$ 与 $\bv\theta = (0.01, -0.005, 0, 0, 0)'$。利用[推论 %s](#cor-ch0-managed)，用真实的 $m_t$ 计算 $\hat{\bv\theta} = (NT)^{-1}\sum_{t}\sum_{i} m_t r_{i,t} \mathbf z_{i,t-1}$（$L$ 维），并给出各分量的标准误。标准误应当怎样计算才能反映同期股票之间的相关？把结果与“模拟实验”一节中加入特征本身的交互回归所得的 $\hat{\bv\theta}$ 比较。
:::

:::{solution-start} ex-ch0-theta
:class: dropdown
:::

先对每一期计算横截面平均 $\mathbf g_t = N^{-1}\sum_i m_t r_{i,t} \mathbf z_{i,t-1} \in \Real^L$，再对 $\mathbf g_t$ 求时间序列均值。由于因子冲击在同一期内影响所有股票，$\mathbf g_t$ 已经把同期相关“吸收”进去了，而不同期的 $\mathbf g_t$ 近似独立：$\mathbf f_t$、$\varepsilon_{i,t}$ 跨期独立；$\mathbf g_t$ 的条件均值 $N^{-1}\sum_i \mathbf z_{i,t-1} \mathbf z_{i,t-1}'\bv\theta$（$L\times L$ 乘 $L\times 1$）随特征缓慢变化，但其波动只有 $N^{-1/2}$ 的量级，远小于因子冲击带来的波动。所以第 $j$ 个分量的标准误可用 $\text{Std}(g_{j,t})/\sqrt{T}$，代码中同时计算 $g_{1,t}$ 的一阶自相关加以确认；单个样本的自相关本身有约 $1/\sqrt{T}$ 的抽样误差，所以再用 100 个种子重复，看其平均值是否接近 0。把 $NT$ 个观测当作独立样本计算的标准误，会低估同期相关较强的那些矩的不确定性。

```{code-cell} python
th_params = MarketParams(alpha_coef=(0.01, -0.005, 0.0, 0.0, 0.0))
th = simulate_market(th_params, seed=21)
m_th = th.sdf().to_numpy()                                                    # (T,)
R_th = th.panel["exret"].unstack("stock").to_numpy()                          # (T, N)
Zs_th = [th.panel[f"z{j}"].unstack("stock").to_numpy() for j in range(1, 6)]  # L 个 (T, N)
mrz_th = [m_th[:, None] * R_th * Zj for Zj in Zs_th]                          # m_t r_{i,t} z_{j,i,t-1}：L 个 (T, N)
g = np.stack([v.mean(axis=1) for v in mrz_th], axis=1)                        # (T, L)：第 t 行为 g_t'
theta_mom = g.mean(axis=0)                                                    # (L,)
se_mom = g.std(axis=0, ddof=1) / np.sqrt(len(g))
se_naive = np.array([v.std() for v in mrz_th]) / np.sqrt(R_th.size)

X_th, Z_th = interaction_design(th)
coef_th, se_th = ols(np.column_stack([X_th, Z_th]), th.panel["exret"].to_numpy())
theta_reg, se_reg = coef_th[X_th.shape[1]:], se_th[X_th.shape[1]:]


def g1_autocorr(market):
    m_ = market.sdf().to_numpy()
    g1 = (m_[:, None] * market.panel["exret"].unstack("stock").to_numpy()
          * market.panel["z1"].unstack("stock").to_numpy()).mean(axis=1)
    return np.corrcoef(g1[1:], g1[:-1])[0, 1]


ac_here = g1_autocorr(th)
ac_reps = np.array([g1_autocorr(simulate_market(th_params, seed=500 + r)) for r in range(100)])
ac_se = ac_reps.std(ddof=1) / np.sqrt(len(ac_reps))
print(f"本样本 g_1t 的一阶自相关 {ac_here:.3f}（单个样本的抽样误差约 1/√T = {1 / np.sqrt(len(g)):.3f}）；"
      f"100 次重复的平均 {ac_reps.mean():.3f}（标准误 {ac_se:.3f}）")
print(pd.DataFrame({"真值": th.theta, "矩估计": theta_mom, "按期标准误": se_mom, "独立假设标准误": se_naive,
                    "交互回归估计": theta_reg, "回归标准误": se_reg},
                   index=[f"θ{j}" for j in range(1, 6)]).round(4))
print("按期标准误 / 独立假设标准误：", (se_mom / se_naive).round(2))
assert abs(ac_reps.mean()) < 3 * ac_se and abs(ac_reps.mean()) < 0.03     # 平均自相关量级很小
assert np.all(np.abs(theta_mom - th.theta) < 2 * se_mom) and np.all(np.abs(theta_reg - th.theta) < 2 * se_reg)
assert np.all(se_mom[:2] / se_naive[:2] > 2) and np.all(np.abs(se_mom[2:] / se_naive[2:] - 1) < 0.2)
assert np.all(se_reg[:2] < 0.5 * se_mom[:2])
```

单个样本的自相关有约 $1/\sqrt{T} \approx 0.065$ 的抽样误差，所以单个样本中的自相关即使达到 0.1 以上也不奇怪（本样本的数值见上方输出）；而 100 次重复的平均自相关量级很小，与 0 的距离在其标准误的 3 倍以内（由 `assert` 保证），说明 $g_{1,t}$ 的序列相关可以忽略，按 $\text{Std}(g_{j,t})/\sqrt{T}$ 计算标准误是合理的。两种方法都恢复了 $\bv\theta$：各分量的偏离都在两倍标准误以内（由 `assert` 保证）。$\theta_1$、$\theta_2$ 的按期标准误是独立假设标准误的 2 倍以上（比值见上方输出），$\theta_3$～$\theta_5$ 则二者相近：$z_1$、$z_2$ 决定了对第二、三个因子的载荷，$m_t r_{i,t} z_{1,i,t-1}$ 中含有 $0.5\, m_t f_{2,t} z_{1,i,t-1}^2$ 这样在同一期内所有股票共有的项，横截面平均无法把它平均掉；$z_3$～$z_5$ 与载荷无关，没有这样的共同项。矩估计直接利用了真实 SDF，回归方法则利用了“载荷是特征的线性函数”这一结构；对 $\theta_1$、$\theta_2$，后者的标准误小得多，因为它把因子冲击带来的收益波动作为解释变量吸收掉了，残差只剩 $\varepsilon_{i,t}$。真实研究中 SDF 未知，必须估计，这正是第二部分 GMM 与第四部分 SDF 估计方法要解决的问题。
:::{solution-end}
:::

## 文献与延伸阅读

- 随机贴现因子、基本定价方程 $P_t = \E_t[m_{t+1}x_{t+1}]$ 以及线性因子模型与 SDF 的对应关系，系统的教材处理见 @cochrane2005asset。第一部分将以它为主要参照，用本书自己的记号重新推导；注意该书不给向量加粗，阅读时要从上下文区分标量与向量。
- 期望收益随时间与资产变化，以及“哪些特征真正提供了独立的定价信息”这一问题，@cochrane2011presidential 给出了提纲挈领的讨论，是理解全书动机的好起点。
- 模拟器中“载荷是特征的线性函数”与“特征驱动的截距”的设定，来自 IPCA 模型 @kelly2019characteristics，第 23 章将完整介绍其估计方法。
- SDF 波动率与最大夏普比率的关系，即 Hansen–Jagannathan 界，原始文献为 @hansen1991implications，第 3 章详细讨论。
- 持续回归变量的小样本偏误见 @stambaugh1999predictive，第 13 章详细讨论。
- 机器学习预测收益的基准框架见 @gu2020empirical；中国市场的对应研究见 @leippold2022machine；中国市场的规模、价值因子与壳价值问题见 @liu2019size。它们分别是第 22 章与第 16、17 章的主要文献。vec 算子、Kronecker 积与矩阵求导的系统介绍见附录 A。
