---
short_title: 0.3～0.6 模型与 SDF 定价
kernelspec:
  name: python3
  display_name: Python 3
numbering:
  enumerator: '0.%s'
  exercise:
    template: 习题 %s
---

# 0.3～0.6 记号、模型与 SDF 的定价性质

**本页准备。** 本书每个页面的代码在独立的内核中运行。下面的单元导入本页用到的模块，并设置统一的作图样式与中文字体（0.6 节的收敛图用到）。

```{code-cell} python
import numpy as np
import pandas as pd

from apsim import MarketParams, simulate_market, plotting

cjk_font = plotting.setup()                                # 返回所选中文字体名
```

## 0.3 记号与时点约定

全书记号集中列在[记号表](../notation.md)中。这里先强调两条：字体约定与时点约定。

**字体与维度。** 标量用斜体，例如收益 $r_{i,t}$、SDF $m_t$；向量用小写粗体，一律是列向量，例如因子 $\mathbf f_t \in \Real^K$、特征 $\mathbf z_{i,t-1} \in \Real^L$、载荷 $\bv\beta_{i,t} \in \Real^K$；矩阵用大写粗体，例如 $\bv\Gamma \in \Real^{(L+1)\times K}$。转置记为 $'$，行向量总是写成列向量的转置。代码中的数据数组则是“观测在行”：因子数组的形状为 `(T, K)`，它的第 $t$ 行是 $\mathbf f_t'$。公式与数组形状的完整对应见记号表的“向量与矩阵约定”，本章所有矩阵运算的代码都在关键行注明形状。

**时点。** 我们用下标 $t$ 表示“在第 $t$ 期末已知”。收益 $r_{i,t}$ 是从第 $t-1$ 期末持有到第 $t$ 期末实现的超额收益；特征 $\mathbf z_{i,t-1}$ 是第 $t-1$ 期末、也就是第 $t$ 期开始时已经知道的变量。一个研究者在第 $t-1$ 期末能够据以形成组合或做出预测的，只有 $\mathbf z_{i,t-1}$ 以及更早的信息。用时间轴表示：

```text
      A                                              B
      |<------------- holding period t ------------->|
   end of t-1                                    end of t

  A（第 t-1 期末）：已知 z_{i,t-1}，因而已知 β_{i,t} = Γ'(1, z_{i,t-1}')'
                    与 α_{i,t} = θ'z_{i,t-1}；据此形成组合或做出预测。
  B（第 t 期末）  ：实现 f_t 与 ε_{i,t}，从而实现 r_{i,t} 与 m_t；
                    同时特征更新为 z_{i,t}，供第 t+1 期使用。
```

注意载荷 $\bv\beta_{i,t}$ 与定价误差 $\alpha_{i,t}$ 虽然带下标 $t$（它们作用于第 $t$ 期的收益），却在第 $t-1$ 期末就已确定。这是条件资产定价文献的通行写法：下标标明“作用于哪一期”，可测性（在哪个时点已知）要另行说明。同理，适用于第 $t$ 期的无风险利率在第 $t-1$ 期末已知，记作 $R^f_{t-1}$。

`apsim` 的面板严格遵守这一约定：面板中索引为 `(t, i)` 的一行，特征列 `z1`……`zL` 存放的是 $\mathbf z_{i,t-1}'$ 的各分量，`exret` 存放的是 $r_{i,t}$。因此，用同一行的特征去解释同一行的收益，在时点上是正确的，不需要再做移位。还有两个容易绊倒人的细节：`period` 从 1 开始编号（$t = 1, \dots, T$），而 `stock` 从 0 开始编号（$i = 0, \dots, N-1$，与 Python 的习惯一致）。`simulate_market` 的随机种子必须以关键字 `seed=` 传入，不能省略。

```{code-cell} python
demo = simulate_market(MarketParams(n_stocks=4, n_periods=3), seed=0)
demo.panel.round(4)
```

理论章节中常用的写法是 $P_t = \E_t[m_{t+1}x_{t+1}]$（价格 $P_t$、支付 $x_{t+1}$ 都是标量），即从 $t$ 到 $t+1$；面板与模拟部分则写成从 $t-1$ 到 $t$，例如 $\E_{t-1}[m_t r_{i,t}]$。两者只差一个时间平移。

## 0.4 模拟市场的模型

本节逐条写出 `apsim.simulate_market` 实现的模型。收益均为月度超额收益。因子、载荷、特征、定价误差与随机贴现因子的含义与数值例子见 0.2 节，这里只写方程。

**因子。** $K$ 个因子的超额收益独立同分布：

$$
\mathbf f_t = \bv\mu + \bv\Sigma_f^{1/2} \mathbf u_t, \qquad \mathbf u_t \overset{\text{iid}}{\sim} N(\mathbf 0, \mathbf I_K), \qquad \bv\Sigma_f = \operatorname{diag}(\sigma_1^2, \dots, \sigma_K^2),
$$ (eq-ch0-factor)

其中 $\mathbf f_t, \bv\mu, \mathbf u_t \in \Real^K$，$\bv\Sigma_f \in \Real^{K\times K}$。因此 $\E[\mathbf f_t] = \bv\mu$，$\Var(\mathbf f_t) = \bv\Sigma_f$。模拟器中各因子相互独立，$\bv\Sigma_f$ 是对角矩阵；后文的理论结论只用到 $\bv\Sigma_f$ 可逆，不依赖对角性。我们把因子理解为可交易的多空组合的超额收益，所以它们本身也是需要被定价的资产。

**特征。** 每只股票有 $L$ 个特征，各自服从平稳的一阶自回归：

$$
\mathbf z_{i,t} = \phi\, \mathbf z_{i,t-1} + \sqrt{1-\phi^2}\; \mathbf e_{i,t}, \qquad \mathbf e_{i,t} \overset{\text{iid}}{\sim} N(\mathbf 0, \mathbf I_L), \qquad \mathbf z_{i,0} \sim N(\mathbf 0, \mathbf I_L),
$$ (eq-ch0-char)

其中 $\mathbf z_{i,t}, \mathbf e_{i,t} \in \Real^L$，持续性 $\phi$ 是标量。系数 $\sqrt{1-\phi^2}$ 的作用是让平稳分布恰好为 $N(\mathbf 0, \mathbf I_L)$：若 $\mathbf z_{i,t-1} \sim N(\mathbf 0, \mathbf I_L)$，则 $\Var(\mathbf z_{i,t}) = \phi^2 \mathbf I_L + (1-\phi^2) \mathbf I_L = \mathbf I_L$。由于初值 $\mathbf z_{i,0}$ 直接从平稳分布抽取，每一期的特征都服从 $N(\mathbf 0, \mathbf I_L)$，不存在需要丢弃的“预热期”。特征的 $k$ 阶自相关为 $\phi^k$。

**因子载荷。** 载荷是期初特征的线性函数：

$$
\underbrace{\bv\beta_{i,t}}_{K\times 1} = \underbrace{\bv\Gamma'}_{K\times(L+1)}\, \underbrace{\tilde{\mathbf z}_{i,t-1}}_{(L+1)\times 1}, \qquad \tilde{\mathbf z}_{i,t-1} = (1, \mathbf z_{i,t-1}')' .
$$ (eq-ch0-beta)

$\bv\Gamma$ 是 $(L+1)\times K$ 矩阵：第 0 行是所有股票共同的载荷，第 $j$ 行刻画特征 $j$ 每增加一个标准差，载荷变化多少；元素 $\Gamma_{j,k}$ 是特征 $j$（$j = 0$ 为常数项）对第 $k$ 个因子载荷的贡献。这与 Kelly、Pruitt 与 Su 的 IPCA 模型对载荷的设定一致 [@kelly2019characteristics]。

**定价误差。** 定价误差也是期初特征的线性函数：

$$
\alpha_{i,t} = \bv\theta' \mathbf z_{i,t-1}, \qquad \bv\theta \in \Real^L .
$$ (eq-ch0-alpha)

默认 $\bv\theta = \mathbf 0$，即市场中没有错误定价。设 $\bv\theta \neq \mathbf 0$ 时，特征除了通过载荷影响收益外，还带来一部分与风险无关的收益。代码中 $\bv\theta$ 由参数 `alpha_coef` 设定，模拟结果的属性 `theta` 返回它。

**超额收益。**

$$
r_{i,t} = \alpha_{i,t} + \bv\beta_{i,t}' \mathbf f_t + \varepsilon_{i,t}, \qquad \varepsilon_{i,t} \overset{\text{iid}}{\sim} N(0, \sigma_\varepsilon^2),
$$ (eq-ch0-ret)

式中每一项都是标量：$\bv\beta_{i,t}'\mathbf f_t$ 是 $(1\times K)(K\times 1)$ 的内积。把同一期的 $N$ 只股票按行堆叠，令 $\tilde{\mathbf Z}_{t-1} \in \Real^{N\times(L+1)}$ 的第 $i$ 行为 $\tilde{\mathbf z}_{i,t-1}'$，$\mathbf Z_{t-1} \in \Real^{N\times L}$ 的第 $i$ 行为 $\mathbf z_{i,t-1}'$，则载荷矩阵 $\mathbf B_t = \tilde{\mathbf Z}_{t-1}\bv\Gamma$ 的第 $i$ 行是 $\tilde{\mathbf z}_{i,t-1}'\bv\Gamma = (\bv\Gamma'\tilde{\mathbf z}_{i,t-1})' = \bv\beta_{i,t}'$，于是式 {eq}`eq-ch0-alpha`、{eq}`eq-ch0-ret` 可以一次写出：

$$
\underbrace{\mathbf r_t}_{N\times 1} = \underbrace{\mathbf Z_{t-1}}_{N\times L}\underbrace{\bv\theta}_{L\times 1} + \underbrace{\tilde{\mathbf Z}_{t-1}}_{N\times(L+1)}\underbrace{\bv\Gamma}_{(L+1)\times K}\underbrace{\mathbf f_t}_{K\times 1} + \underbrace{\bv\varepsilon_t}_{N\times 1} .
$$ (eq-ch0-stacked)

这正是 `simulate_market` 的写法：`beta = x @ gamma`（形状 `(T, N, L+1) @ (L+1, K) -> (T, N, K)`，对每一期计算 $\tilde{\mathbf Z}_{t-1}\bv\Gamma$），`alpha = z_lag @ theta`（`(T, N, L) @ (L,) -> (T, N)`），以及 `np.einsum("tnk,tk->tn", beta, f)`，它对每个 $(t, i)$ 把 $\beta_{k,i,t} f_{k,t}$ 沿下标 `k` 求和，得到 $\bv\beta_{i,t}'\mathbf f_t$。`einsum` 的下标字符串按“输入下标 `->` 输出下标”读：出现在输入、却不在输出中的下标被求和。

**随机贴现因子。**

$$
m_t = 1 - \mathbf b'(\mathbf f_t - \bv\mu), \qquad \underbrace{\mathbf b}_{K\times 1} = \underbrace{\bv\Sigma_f^{-1}}_{K\times K}\underbrace{\bv\mu}_{K\times 1} .
$$ (eq-ch0-sdf)

$m_t$ 是标量。代码中 `SimulatedMarket.sdf_weights` 用 `np.linalg.solve(params.factor_cov, mu)` 解线性方程组 $\bv\Sigma_f \mathbf b = \bv\mu$ 得到 $\mathbf b$，而不是先求逆矩阵再相乘（原因见记号表：解方程更快，舍入误差也更小）。模拟器中 $\bv\Sigma_f$ 为对角矩阵，于是 $\mathbf b$ 的第 $k$ 个分量化为 $b_k = \mu_k / \sigma_k^2$；但代码与本章的推导都按一般的 $\bv\Sigma_f$ 写，以后改用相关因子时只需修改 `MarketParams.factor_cov` 与因子的生成方式。

最后是全部随机性的来源。

:::{prf:assumption} 独立性
:label: asm-ch0-indep
冲击 $\{\mathbf u_t\}$、$\{\mathbf e_{i,t}\}$、$\{\varepsilon_{i,t}\}$ 与初值 $\{\mathbf z_{i,0}\}$ 相互独立，且各自在 $t$ 与 $i$ 上独立同分布。
:::

[假设 %s](#asm-ch0-indep) 有两个直接推论，证明中会反复用到：

- $\mathbf z_{i,t-1}$ 只是 $\mathbf z_{i,0}$ 与 $\mathbf e_{i,1}, \dots, \mathbf e_{i,t-1}$ 的函数，所以 $(\mathbf f_t, \varepsilon_{i,t})$ 与 $\mathbf z_{i,t-1}$ 独立，且 $\mathbf f_t$ 与 $\varepsilon_{i,t}$ 也相互独立；
- 更一般地，$(\mathbf f_t, \varepsilon_{i,t})$ 与第 $t-1$ 期末之前的全部信息（所有特征、所有过去的因子与收益）独立。

`simulate_market` 用 `np.random.SeedSequence` 为因子、初始特征、特征冲击、异质性冲击各派生一个独立的随机子流，与这一假设一一对应；以后给模拟器增加新的随机部分时，只追加新的子流，已有部分的抽样不变。

下面的代码单元打印默认参数与默认载荷矩阵，并逐列核对面板确实由式 {eq}`eq-ch0-beta`～{eq}`eq-ch0-ret` 生成。

```{code-cell} python
params = MarketParams()
mkt = simulate_market(params, seed=0)
print(params)
print("默认 Γ（行：常数项、z1……z5；列：f1, f2, f3）：")
print(mkt.gamma)
mu = np.asarray(params.factor_mean)                       # μ: (K,)
print("Σ_f =")
print(params.factor_cov)                                  # Σ_f: (K, K)
print("b = Σ_f⁻¹ μ =", mkt.sdf_weights.round(4))           # 解 Σ_f b = μ，b: (K,)
assert np.allclose(params.factor_cov @ mkt.sdf_weights, mu)
```

默认的 $\bv\Gamma$ 含义很直观：所有股票对第一个因子的载荷都是 1（类似市场因子）；第一个特征决定对第二个因子的载荷，$\beta_{2,i,t} = 0.5\, z_{1,i,t-1}$；第二个特征决定对第三个因子的载荷，$\beta_{3,i,t} = 0.5\, z_{2,i,t-1}$；其余三个特征与载荷无关，是“噪声特征”。真实研究中候选特征往往成百上千，其中多数与风险无关，第四部分的方法必须学会区分它们。

下一个代码单元把面板的各列取成“观测在行”的数组：`panel` 有 $TN$ 行，每行是一个 $(t, i)$。于是 `zt` 的每一行是 $\tilde{\mathbf z}_{i,t-1}'$，`zt @ mkt.gamma` 的每一行是 $\tilde{\mathbf z}_{i,t-1}'\bv\Gamma = \bv\beta_{i,t}'$，即式 {eq}`eq-ch0-beta` 的转置；`z @ mkt.theta` 的每个元素是 $\mathbf z_{i,t-1}'\bv\theta = \alpha_{i,t}$。

```{code-cell} python
panel = mkt.panel
L, K = params.n_chars, params.n_factors
z = panel[[f"z{j}" for j in range(1, L + 1)]].to_numpy()             # (TN, L)，每行 z_{i,t-1}'
zt = np.column_stack([np.ones(len(panel)), z])                         # (TN, L+1)，每行 z̃_{i,t-1}'
beta = panel[[f"beta{k}" for k in range(1, K + 1)]].to_numpy()       # (TN, K)，每行 β_{i,t}'
f_row = mkt.factors.reindex(panel.index.get_level_values("period")).to_numpy()   # (TN, K)，每行 f_t'

# 载荷与定价误差是期初特征的确定函数：载荷方程 eq-ch0-beta 与定价误差方程 eq-ch0-alpha
assert np.allclose(beta, zt @ mkt.gamma)                               # (TN, L+1) @ (L+1, K) -> (TN, K)
assert np.allclose(panel["alpha"], z @ mkt.theta)                      # (TN, L) @ (L,) -> (TN,)

# 收益减去 α 与 β'f 后，剩下的应是标准差为 σ_ε 的白噪声；β'f 按行求内积：(TN, K) -> (TN,)
eps = panel["exret"].to_numpy() - panel["alpha"].to_numpy() - np.sum(beta * f_row, axis=1)
print(f"残差标准差 {eps.std():.4f}（真值 σ_ε = {params.idio_vol}），残差均值 {eps.mean():.5f}")

# 特征的平稳分布为 N(0, I)，一阶自相关为 φ
z1 = panel["z1"].unstack("stock").to_numpy()                           # (T, N)
phi_hat = np.corrcoef(z1[1:].ravel(), z1[:-1].ravel())[0, 1]
print(f"z1 的均值 {z1.mean():.4f}、方差 {z1.var():.4f}、一阶自相关 {phi_hat:.4f}（真值 0、1、{params.char_persistence}）")

# 单只股票收益的方差分解：Var(r) = Var(β'f) + σ_ε²。默认 Γ 下 β'f = f1 + 0.5 z1 f2 + 0.5 z2 f3
sig = np.asarray(params.factor_vol)
var_sys = sig[0] ** 2 + 0.25 * (sig[1] ** 2 + mu[1] ** 2) + 0.25 * (sig[2] ** 2 + mu[2] ** 2)
idio_share = params.idio_vol**2 / (var_sys + params.idio_vol**2)
print(f"单只股票月度超额收益的标准差 {panel['exret'].std():.4f}（理论值 {np.sqrt(var_sys + params.idio_vol**2):.4f}）")
print(f"异质性冲击占收益方差的比例：理论值 {idio_share:.3f}，样本值 {eps.var() / panel['exret'].var():.3f}")
assert abs(eps.std() - params.idio_vol) < 0.002 and abs(phi_hat - params.char_persistence) < 0.01
assert 0.75 < idio_share < 0.8 and abs(eps.var() / panel["exret"].var() - idio_share) < 0.02
```

单只股票的月度收益波动中，异质性冲击 $\varepsilon_{i,t}$ 占了绝大部分：按默认参数计算，它约占收益方差的四分之三以上（理论值见上方输出，不依赖种子）。这意味着用单只股票的时间序列去估计它的载荷或定价误差，噪声会非常大；后面的估计几乎都要借助横截面上的大量股票。

:::{note} 为什么是 $\E[m_t] = 1$
本书的模拟市场只生成超额收益。超额收益的定价方程 $0 = \E[m_t r_{i,t}]$ 对 $m_t$ 乘以任何正常数都成立，所以 SDF 的均值在只有超额收益时无法识别，可以自由归一化。我们取 $\E[m_t] = 1$，相当于令无风险总收益 $R^f = 1/\E[m_t] = 1$。需要无风险利率的场合（第 1 章起），再显式引入 $R^f_{t-1}$（面板写法）或 $R^f_t$（理论写法）。
:::

## 0.5 模拟市场中 SDF 的定价性质

下面的定理说明，式 {eq}`eq-ch0-sdf` 给出的 $m_t$ 确实是这个市场的随机贴现因子：它正确地为因子定价，而它为股票定价时留下的误差恰好就是 $\alpha_{i,t}$。这是全书用模拟数据检验方法的基础：只要知道真实的 $m_t$，就知道每只股票在每一期“应有的”定价误差。在 $\E[m_t] = 1$ 的归一化下，这里的定价误差与 0.2 节的定义“期望超额收益减去 $\bv\beta_{i,t}'\bv\lambda$”（取以 $\mathbf z_{i,t-1}$ 为条件的期望）是同一个量（[习题 %s](#ex-ch0-beta)）。

:::{prf:theorem} 模拟市场 SDF 的定价性质
:label: thm-ch0-sdf
在式 {eq}`eq-ch0-factor`～{eq}`eq-ch0-sdf` 与[假设 %s](#asm-ch0-indep) 下，设 $\bv\Sigma_f$ 可逆，$m_t = 1 - \mathbf b'(\mathbf f_t - \bv\mu)$，$\mathbf b = \bv\Sigma_f^{-1}\bv\mu$。则对任意 $t$ 与 $i$：

1. $\E[m_t] = 1$；
2. $\E[m_t \mathbf f_t] = \mathbf 0 \in \Real^K$，即 $m_t$ 为全部 $K$ 个因子正确定价；
3. $\E[m_t r_{i,t} \mid \mathbf z_{i,t-1}] = \alpha_{i,t} = \bv\theta' \mathbf z_{i,t-1}$。特别地，当 $\bv\theta = \mathbf 0$ 时，$\E[m_t r_{i,t} \mid \mathbf z_{i,t-1}] = 0$，即 $m_t$ 在条件意义下为每一只股票正确定价。
:::

:::{prf:proof}
**第 1 条。** $m_t$ 是 $\mathbf f_t$ 的线性函数，取期望并利用 $\E[\mathbf f_t] = \bv\mu$：

```{math}
:enumerated: false
\E[m_t] = 1 - \mathbf b'\left(\E[\mathbf f_t] - \bv\mu\right) = 1 - \mathbf b' \mathbf 0 = 1.
```

**第 2 条。** 由于 $\mathbf b'(\mathbf f_t - \bv\mu)$ 是标量，它等于自身的转置 $(\mathbf f_t - \bv\mu)'\mathbf b$，所以 $m_t \mathbf f_t = \mathbf f_t - \mathbf f_t (\mathbf f_t - \bv\mu)'\mathbf b$，其中 $\mathbf f_t (\mathbf f_t - \bv\mu)'$ 是 $K\times K$ 矩阵、$\mathbf b$ 是 $K\times 1$ 向量。取期望：

```{math}
:enumerated: false
\underbrace{\E[m_t \mathbf f_t]}_{K\times 1} = \underbrace{\E[\mathbf f_t]}_{K\times 1} - \underbrace{\E\left[\mathbf f_t (\mathbf f_t - \bv\mu)'\right]}_{K\times K} \underbrace{\mathbf b}_{K\times 1} .
```

第二项中，把 $\mathbf f_t$ 写成 $(\mathbf f_t - \bv\mu) + \bv\mu$：

```{math}
:enumerated: false
\E\left[\mathbf f_t (\mathbf f_t - \bv\mu)'\right] = \E\left[(\mathbf f_t - \bv\mu)(\mathbf f_t - \bv\mu)'\right] + \bv\mu\, \E\left[(\mathbf f_t - \bv\mu)'\right] = \bv\Sigma_f + \bv\mu\,\mathbf 0' = \bv\Sigma_f .
```

代回并利用 $\mathbf b = \bv\Sigma_f^{-1}\bv\mu$：

```{math}
:enumerated: false
\E[m_t \mathbf f_t] = \bv\mu - \bv\Sigma_f \bv\Sigma_f^{-1} \bv\mu = \mathbf 0 .
```

这一步只用到 $\mathbf f_t$ 的一、二阶矩，既不需要正态性，也不需要 $\bv\Sigma_f$ 是对角矩阵。反过来，若要求 $1 - \mathbf b'(\mathbf f_t-\bv\mu)$ 为因子定价，由上式 $\bv\mu - \bv\Sigma_f \mathbf b = \mathbf 0$，$\mathbf b$ 必须等于 $\bv\Sigma_f^{-1}\bv\mu$，所以在这一函数形式下 $\mathbf b$ 是唯一的。

**第 3 条。** 把式 {eq}`eq-ch0-ret` 代入：

```{math}
:enumerated: false
\E[m_t r_{i,t} \mid \mathbf z_{i,t-1}]
= \E[m_t \alpha_{i,t} \mid \mathbf z_{i,t-1}] + \E[m_t \bv\beta_{i,t}' \mathbf f_t \mid \mathbf z_{i,t-1}] + \E[m_t \varepsilon_{i,t} \mid \mathbf z_{i,t-1}] .
```

由式 {eq}`eq-ch0-beta` 与 {eq}`eq-ch0-alpha`，$\alpha_{i,t}$ 与 $\bv\beta_{i,t}$ 都是 $\mathbf z_{i,t-1}$ 的确定函数，可以提到条件期望之外：

```{math}
:enumerated: false
\E[m_t r_{i,t} \mid \mathbf z_{i,t-1}]
= \alpha_{i,t}\, \E[m_t \mid \mathbf z_{i,t-1}] + \bv\beta_{i,t}'\, \E[m_t \mathbf f_t \mid \mathbf z_{i,t-1}] + \E[m_t \varepsilon_{i,t} \mid \mathbf z_{i,t-1}] ,
```

其中第二项是 $(1\times K)$ 行向量与 $(K\times 1)$ 列向量之积，为标量。逐项处理：

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

:::{prf:proof}
由迭代期望定律与定理第 3 条，$\E[m_t r_{i,t}] = \E\left[\E[m_t r_{i,t} \mid \mathbf z_{i,t-1}]\right] = \bv\theta' \E[\mathbf z_{i,t-1}] = \bv\theta'\mathbf 0 = 0$。同理，由于 $\mathbf z_{i,t-1}$ 在以自身为条件时可以提出，并且标量 $\bv\theta'\mathbf z_{i,t-1}$ 等于 $\mathbf z_{i,t-1}'\bv\theta$，

```{math}
:enumerated: false
\E[m_t r_{i,t}\, \mathbf z_{i,t-1}] = \E\left[\mathbf z_{i,t-1}\, \E[m_t r_{i,t} \mid \mathbf z_{i,t-1}]\right] = \underbrace{\E[\mathbf z_{i,t-1} \mathbf z_{i,t-1}']}_{L\times L}\, \underbrace{\bv\theta}_{L\times 1} = \mathbf I_L\, \bv\theta = \bv\theta ,
```

最后一步用到 $\mathbf z_{i,t-1} \sim N(\mathbf 0, \mathbf I_L)$。
:::

这个推论有一个值得记住的含义：即使存在错误定价（$\bv\theta \neq \mathbf 0$），每只股票的**无条件**定价误差仍然为 0，因为特征围绕 0 波动，错误定价时正时负、平均抵消。只有把收益按特征加权，即构造“管理组合”（managed portfolio）$r_{i,t}\, \mathbf z_{i,t-1}$，错误定价才在无条件矩中显现出来。用无条件矩检验条件模型时，检验资产的选择决定了检验能看到什么。

最后，$m_t$ 的波动也有明确的含义：$\Var(m_t) = \mathbf b'\bv\Sigma_f \mathbf b = \bv\mu'\bv\Sigma_f^{-1}\bv\mu$（三项的维度依次为 $1\times K$、$K\times K$、$K\times 1$，结果为标量），它等于用这 $K$ 个因子所能构成的组合的最大夏普比率的平方。Hansen–Jagannathan 界 [@hansen1991implications] 指出，任何为这些因子正确定价、且均值为 1 的 SDF，其标准差都不小于这一最大夏普比率；本章的 $m_t$ 是因子的线性函数，恰好达到这个下界。第 3 章将详细讨论。线性 SDF 还有一个缺点：$\mathbf f_t$ 服从正态分布，$m_t$ 以正的概率取负值。这本身并不意味着这些因子之间存在套利：无套利等价于存在**某个**严格为正的 SDF（第 2 章），而一个特定的线性 $m_t$ 取负值，并不排除另有一个正的 SDF 同样为这些因子正确定价。问题出在用这个 $m_t$ 为因子的非线性支付定价时，例如以因子为标的的期权：在 $m_t < 0$ 的状态上支付为正、其余状态支付为 0 的证券，会被 $m_t$ 赋予负的价格，而买入一个价格为负、支付非负的证券就是套利。所以线性 SDF 只适合为它所针对的线性支付定价。在默认参数下 $m_t < 0$ 的概率可以忽略，下一节会算出它。

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

