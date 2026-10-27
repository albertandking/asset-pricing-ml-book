---
short_title: 0.3～0.4 记号与模型
kernelspec:
  name: python3
  display_name: Python 3
numbering:
  enumerator: '0.%s'
  exercise:
    template: 习题 %s
---

# 0.3～0.4 记号与模拟市场的模型

**本页准备。** 本书每个页面的代码在独立的内核中运行。下面的单元导入本页用到的模块，并设置统一的作图样式与中文字体（与 0.6 节所在的下一页相同）。

```{code-cell} python
import numpy as np
import pandas as pd

from apsim import MarketParams, simulate_market, plotting

cjk_font = plotting.setup()                                # 返回所选中文字体名
```

## 0.3 记号与时点约定

全书记号集中列在[记号表](../notation.md)中。这里先强调两条：字体约定与时点约定。

**字体与维度。** 标量用斜体，例如收益 $r_{i,t}$、SDF $m_t$；向量用小写粗体，一律是列向量，例如因子 $\mathbf f_t \in \Real^K$、特征 $\mathbf z_{i,t-1} \in \Real^L$、载荷 $\bv\beta_{i,t} \in \Real^K$；矩阵用大写粗体，例如 $\bv\Gamma \in \Real^{(L+1)\times K}$。转置记为 $'$，行向量总是写成列向量的转置。代码中的数据数组则是“观测在行”：因子数组的形状为 `(T, K)`，它的第 $t$ 行是 $\mathbf f_t'$。公式与数组形状的完整对应见记号表的“向量与矩阵约定”，本章所有矩阵运算的代码都在关键行注明形状。

这几个向量与矩阵在 0.2 节已经逐一拆开讲过，这里不再重复：$\mathbf f_t$ 的分量与因子矩阵 $\mathbf F$ 见[拆解](#ch0-unpack-f)，$\bv\beta_i'\mathbf f_t$ 见[记号表的范例](#notation-unpack)，$\mathbf z_{i,t-1}$、$\tilde{\mathbf z}_{i,t-1}$ 与 $\bv\Gamma$ 见[拆解](#ch0-unpack-gamma)。下文第一次出现的向量与矩阵，在出现的地方就地拆开。

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

(ch0-unpack-factor)=
**拆开讲式 {eq}`eq-ch0-factor`。** 式中新出现的是标准正态冲击 $\mathbf u_t$、协方差矩阵 $\bv\Sigma_f$ 与它的平方根 $\bv\Sigma_f^{1/2}$（$\mathbf f_t$ 的分量见[拆解](#ch0-unpack-f)，$\bv\mu$ 的分量见 SDF 的[拆解](#ch0-unpack-sdf)）。写出分量：

```{math}
:enumerated: false
\underbrace{\mathbf u_t}_{K\times 1} = \begin{pmatrix} u_{1,t} \\ u_{2,t} \\ \vdots \\ u_{K,t} \end{pmatrix} , \qquad
\underbrace{\bv\Sigma_f}_{K\times K} = \begin{pmatrix} \sigma_1^2 & 0 & \cdots & 0 \\ 0 & \sigma_2^2 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & \sigma_K^2 \end{pmatrix} , \qquad
\underbrace{\bv\Sigma_f^{1/2}}_{K\times K} = \begin{pmatrix} \sigma_1 & 0 & \cdots & 0 \\ 0 & \sigma_2 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & \sigma_K \end{pmatrix} .
```

$u_{k,t}$ 是第 $k$ 个因子在第 $t$ 期的“意外”，以标准差为单位，本身没有单位，均值为 0、方差为 1；$N(\mathbf 0, \mathbf I_K)$ 中的 $\mathbf I_K$ 是 $K\times K$ 单位矩阵，表示各分量方差为 1、两两协方差为 0，再加上正态性，$K$ 个分量相互独立。$\sigma_k$ 是第 $k$ 个因子的月度波动率（标准差），单位与收益相同，$\sigma_k^2$ 是方差。$\operatorname{diag}(\cdot)$ 表示把括号里的数依次放在对角线上、其余元素为 0 的方阵。一般的 $\bv\Sigma_f$ 是对称矩阵，$(k, l)$ 元素是第 $k$ 与第 $l$ 个因子的协方差（见[拆解](#ch0-unpack-ts)）；模拟器中因子互不相关，非对角元都为 0。$\bv\Sigma_f^{1/2}$ 是满足 $\bv\Sigma_f^{1/2}(\bv\Sigma_f^{1/2})' = \bv\Sigma_f$ 的矩阵，对角情形下就是把每个方差开平方。**维度**：$K\times K$ 的 $\bv\Sigma_f^{1/2}$ 乘 $K\times 1$ 的 $\mathbf u_t$ 得 $K\times 1$，与 $\bv\mu$ 相加仍是 $K\times 1$。**展开运算**：对角情形下矩阵乘法只剩对角元，逐个分量是

```{math}
:enumerated: false
f_{k,t} = \mu_k + \sigma_k u_{k,t} , \qquad k = 1, \dots, K ,
```

即“每个因子等于它的均值，加上它的波动率乘以一个标准正态的意外”。默认参数下 $\bv\mu = (0.6\%, 0.3\%, 0.4\%)'$、$\bv\sigma = (5\%, 3\%, 3\%)'$；若某个月 $\mathbf u_t = (1, -0.5, 0)'$，则 $\mathbf f_t = (0.6\% + 5\%,\ 0.3\% - 1.5\%,\ 0.4\%)' = (5.6\%, -1.2\%, 0.4\%)'$（下方代码核对）。**特例**：$K = 1$ 时就是 $f_t = \mu + \sigma u_t$，0.6 节的符号验证用的正是这个形式。因子相关时（$\bv\Sigma_f$ 非对角），可以取 $\bv\Sigma_f$ 的 Cholesky 分解，即满足 $\mathbf C\mathbf C' = \bv\Sigma_f$ 的下三角矩阵 $\mathbf C$，作为 $\bv\Sigma_f^{1/2}$，这时 $f_{k,t}$ 混入了其他因子的冲击；模拟器目前不用这一形式。**数据矩阵**：$T$ 期按行堆叠为 $\mathbf F = \mathbf 1_T\bv\mu' + \mathbf U(\bv\Sigma_f^{1/2})'$，其中 $\mathbf U$ 是 $T\times K$ 的冲击矩阵，第 $t$ 行为 $\mathbf u_t'$；所有股票共用同一个 $\mathbf F$。`simulate_market` 中的 `factor_mean + factor_vol * standard_normal((T, K))` 用广播实现对角情形：形状为 `(K,)` 的 `factor_vol` 逐元素乘到 `(T, K)` 冲击矩阵的每一行上。

```{code-cell} python
# 拆开讲 eq-ch0-factor：对角 Σ_f 下 f_t = μ + Σ_f^{1/2} u_t 逐分量为 f_k = μ_k + σ_k u_k
p_ex = MarketParams()
mu_ex, sig_ex = np.asarray(p_ex.factor_mean), np.asarray(p_ex.factor_vol)   # μ、σ: (K,)
u_ex = np.array([1.0, -0.5, 0.0])                                            # 假设的一个月的 u_t: (K,)
f_ex = mu_ex + np.diag(sig_ex) @ u_ex                                        # (K,) + (K, K) @ (K,) -> (K,)
print("f_t =", f_ex)
assert np.allclose(f_ex, mu_ex + sig_ex * u_ex) and np.allclose(f_ex, [0.056, -0.012, 0.004])
assert np.allclose(np.diag(sig_ex) @ np.diag(sig_ex).T, p_ex.factor_cov)    # Σ_f^{1/2}(Σ_f^{1/2})' = Σ_f
```

**特征。** 每只股票有 $L$ 个特征，各自服从平稳的一阶自回归：

$$
\mathbf z_{i,t} = \phi\, \mathbf z_{i,t-1} + \sqrt{1-\phi^2}\; \mathbf e_{i,t}, \qquad \mathbf e_{i,t} \overset{\text{iid}}{\sim} N(\mathbf 0, \mathbf I_L), \qquad \mathbf z_{i,0} \sim N(\mathbf 0, \mathbf I_L),
$$ (eq-ch0-char)

其中 $\mathbf z_{i,t}, \mathbf e_{i,t} \in \Real^L$，持续性 $\phi$ 是标量。系数 $\sqrt{1-\phi^2}$ 的作用是让平稳分布恰好为 $N(\mathbf 0, \mathbf I_L)$：若 $\mathbf z_{i,t-1} \sim N(\mathbf 0, \mathbf I_L)$，则 $\Var(\mathbf z_{i,t}) = \phi^2 \mathbf I_L + (1-\phi^2) \mathbf I_L = \mathbf I_L$。由于初值 $\mathbf z_{i,0}$ 直接从平稳分布抽取，每一期的特征都服从 $N(\mathbf 0, \mathbf I_L)$，不存在需要丢弃的“预热期”。特征的 $k$ 阶自相关为 $\phi^k$。

(ch0-unpack-char)=
**拆开讲式 {eq}`eq-ch0-char`。** $\mathbf z_{i,t}$ 的分量见[拆解](#ch0-unpack-gamma)；新出现的是特征的更新冲击

```{math}
:enumerated: false
\underbrace{\mathbf e_{i,t}}_{L\times 1} = \begin{pmatrix} e_{1,i,t} \\ e_{2,i,t} \\ \vdots \\ e_{L,i,t} \end{pmatrix} ,
```

$e_{j,i,t}$ 是第 $t$ 期股票 $i$ 的第 $j$ 个特征收到的新消息，以标准差为单位，均值为 0、方差为 1，与其他特征、其他股票、其他时期的冲击都独立。$\phi$ 是持续性，没有单位，所有股票、所有特征共用同一个 $\phi$（默认 0.9）。**维度**：标量 $\phi$ 乘 $L\times 1$ 的向量仍是 $L\times 1$，两项相加得 $L\times 1$ 的 $\mathbf z_{i,t}$。**展开运算**：标量乘向量就是乘到每个分量上，所以 $L$ 个特征各自独立地演化，

```{math}
:enumerated: false
z_{j,i,t} = \phi\, z_{j,i,t-1} + \sqrt{1-\phi^2}\; e_{j,i,t} , \qquad j = 1, \dots, L ,
```

即“这个月的特征等于上个月特征的 $\phi$ 倍，加上一点新消息”。默认 $\phi = 0.9$ 时 $\sqrt{1-\phi^2} = \sqrt{0.19} \approx 0.436$：某只股票上月 $z_1 = 2$，本月若没有新消息（$e_{1,i,t} = 0$），$z_1$ 回落到 1.8；若新消息为 $e_{1,i,t} = 1$，则为 $1.8 + 0.436 = 2.236$。$\phi < 1$ 使特征逐渐回到平均水平 0。**特例**：$\phi = 0$ 时 $\mathbf z_{i,t} = \mathbf e_{i,t}$，每期的特征都是全新的独立抽样，与上期无关；$\phi$ 趋于 1 时特征几乎不变，接近每只股票固定不变的属性（[习题 %s](#ex-ch0-phi) 比较了不同的 $\phi$）。**数据矩阵**：同一期 $N$ 只股票的 $\mathbf z_{i,t-1}'$ 按行堆叠为 $N\times L$ 的 $\mathbf Z_{t-1}$（见[拆解](#ch0-unpack-alpha)），于是 $\mathbf Z_t = \phi\,\mathbf Z_{t-1} + \sqrt{1-\phi^2}\,\mathbf E_t$，其中 $\mathbf E_t$ 是 $N\times L$ 的冲击矩阵，第 $i$ 行为 $\mathbf e_{i,t}'$（它是粗体正体的大写字母 E，与表示条件期望的空心体 $\E_t[\cdot]$ 无关）；面板一共 $TN$ 行，每行存放一个 $\mathbf z_{i,t-1}'$。$\phi$ 由所有股票共用，每只股票有自己的一条特征路径。

**因子载荷。** 载荷是期初特征的线性函数：

$$
\underbrace{\bv\beta_{i,t}}_{K\times 1} = \underbrace{\bv\Gamma'}_{K\times(L+1)}\, \underbrace{\tilde{\mathbf z}_{i,t-1}}_{(L+1)\times 1}, \qquad \tilde{\mathbf z}_{i,t-1} = (1, \mathbf z_{i,t-1}')' .
$$ (eq-ch0-beta)

$\bv\Gamma$ 是 $(L+1)\times K$ 矩阵：第 0 行是所有股票共同的载荷，第 $j$ 行刻画特征 $j$ 每增加一个标准差，载荷变化多少；元素 $\Gamma_{j,k}$ 是特征 $j$（$j = 0$ 为常数项）对第 $k$ 个因子载荷的贡献。这与 Kelly、Pruitt 与 Su 的 IPCA 模型对载荷的设定一致 [@kelly2019characteristics]。

式 {eq}`eq-ch0-beta` 的分量、维度、逐项展开、特例与数据矩阵，以及默认 $\bv\Gamma$ 下的数值例子，见 0.2 节“特征”一节的[拆解](#ch0-unpack-gamma)。

**定价误差。** 定价误差也是期初特征的线性函数：

$$
\alpha_{i,t} = \bv\theta' \mathbf z_{i,t-1}, \qquad \bv\theta \in \Real^L .
$$ (eq-ch0-alpha)

默认 $\bv\theta = \mathbf 0$，即市场中没有错误定价。设 $\bv\theta \neq \mathbf 0$ 时，特征除了通过载荷影响收益外，还带来一部分与风险无关的收益。代码中 $\bv\theta$ 由参数 `alpha_coef` 设定，模拟结果的属性 `theta` 返回它。

式 {eq}`eq-ch0-alpha` 中 $\bv\theta$ 的分量、$\bv\theta'\mathbf z_{i,t-1}$ 的逐项展开与数值例子，见 0.2 节“定价误差”一节的[拆解](#ch0-unpack-alpha)。

**超额收益。**

$$
r_{i,t} = \alpha_{i,t} + \bv\beta_{i,t}' \mathbf f_t + \varepsilon_{i,t}, \qquad \varepsilon_{i,t} \overset{\text{iid}}{\sim} N(0, \sigma_\varepsilon^2),
$$ (eq-ch0-ret)

式中每一项都是标量：$\bv\beta_{i,t}'\mathbf f_t$ 是 $(1\times K)(K\times 1)$ 的内积。把同一期的 $N$ 只股票按行堆叠，令 $\tilde{\mathbf Z}_{t-1} \in \Real^{N\times(L+1)}$ 的第 $i$ 行为 $\tilde{\mathbf z}_{i,t-1}'$，$\mathbf Z_{t-1} \in \Real^{N\times L}$ 的第 $i$ 行为 $\mathbf z_{i,t-1}'$，则载荷矩阵 $\mathbf B_t = \tilde{\mathbf Z}_{t-1}\bv\Gamma$ 的第 $i$ 行是 $\tilde{\mathbf z}_{i,t-1}'\bv\Gamma = (\bv\Gamma'\tilde{\mathbf z}_{i,t-1})' = \bv\beta_{i,t}'$，于是式 {eq}`eq-ch0-alpha`、{eq}`eq-ch0-ret` 可以一次写出：

$$
\underbrace{\mathbf r_t}_{N\times 1} = \underbrace{\mathbf Z_{t-1}}_{N\times L}\underbrace{\bv\theta}_{L\times 1} + \underbrace{\tilde{\mathbf Z}_{t-1}}_{N\times(L+1)}\underbrace{\bv\Gamma}_{(L+1)\times K}\underbrace{\mathbf f_t}_{K\times 1} + \underbrace{\bv\varepsilon_t}_{N\times 1} .
$$ (eq-ch0-stacked)

这正是 `simulate_market` 的写法：`beta = x @ gamma`（形状 `(T, N, L+1) @ (L+1, K) -> (T, N, K)`，对每一期计算 $\tilde{\mathbf Z}_{t-1}\bv\Gamma$），`alpha = z_lag @ theta`（`(T, N, L) @ (L,) -> (T, N)`），以及 `np.einsum("tnk,tk->tn", beta, f)`，它对每个 $(t, i)$ 把 $\beta_{k,i,t} f_{k,t}$ 沿下标 `k` 求和，得到 $\bv\beta_{i,t}'\mathbf f_t$。`einsum` 的下标字符串按“输入下标 `->` 输出下标”读：出现在输入、却不在输出中的下标被求和。

(ch0-unpack-stacked)=
**拆开讲式 {eq}`eq-ch0-ret` 与 {eq}`eq-ch0-stacked`。** 式 {eq}`eq-ch0-ret` 的四项及其单位见 0.2 节的[拆解](#ch0-unpack-return)。式 {eq}`eq-ch0-stacked` 是把它对同一期的 $N$ 只股票竖着排起来：

```{math}
:enumerated: false
\underbrace{\begin{pmatrix} r_{1,t} \\ r_{2,t} \\ \vdots \\ r_{N,t} \end{pmatrix}}_{\mathbf r_t}
= \underbrace{\begin{pmatrix} \mathbf z_{1,t-1}'\bv\theta \\ \mathbf z_{2,t-1}'\bv\theta \\ \vdots \\ \mathbf z_{N,t-1}'\bv\theta \end{pmatrix}}_{\mathbf Z_{t-1}\bv\theta}
+ \underbrace{\begin{pmatrix} \tilde{\mathbf z}_{1,t-1}'\bv\Gamma\mathbf f_t \\ \tilde{\mathbf z}_{2,t-1}'\bv\Gamma\mathbf f_t \\ \vdots \\ \tilde{\mathbf z}_{N,t-1}'\bv\Gamma\mathbf f_t \end{pmatrix}}_{\tilde{\mathbf Z}_{t-1}\bv\Gamma\mathbf f_t}
+ \underbrace{\begin{pmatrix} \varepsilon_{1,t} \\ \varepsilon_{2,t} \\ \vdots \\ \varepsilon_{N,t} \end{pmatrix}}_{\bv\varepsilon_t} .
```

第 $i$ 行就是股票 $i$ 的式 {eq}`eq-ch0-ret`：$\mathbf z_{i,t-1}'\bv\theta = \alpha_{i,t}$，$\tilde{\mathbf z}_{i,t-1}'\bv\Gamma\mathbf f_t = \bv\beta_{i,t}'\mathbf f_t$。$\mathbf Z_{t-1}$ 的结构见[拆解](#ch0-unpack-alpha)，$\tilde{\mathbf Z}_{t-1}$ 的结构见[拆解](#ch0-unpack-return)，$\bv\varepsilon_t$ 的第 $i$ 个分量是股票 $i$ 的异质性冲击。**维度**：$N\times L$ 乘 $L\times 1$ 得 $N\times 1$；$N\times(L+1)$ 乘 $(L+1)\times K$ 得 $N\times K$ 的载荷矩阵 $\mathbf B_t$，再乘 $K\times 1$ 的 $\mathbf f_t$ 得 $N\times 1$。三项都是 $N\times 1$，才能相加。矩阵乘法满足结合律，$(\tilde{\mathbf Z}_{t-1}\bv\Gamma)\mathbf f_t = \tilde{\mathbf Z}_{t-1}(\bv\Gamma\mathbf f_t)$：前一种算法先求每只股票的载荷、再乘因子，后一种先求 $(L+1)\times 1$ 的 $\bv\Gamma\mathbf f_t$，即 0.2 节横截面回归中系数向量 $\mathbf g_t$ 的因子部分（[拆解](#ch0-unpack-g)），再与每只股票的特征做内积。**特例**：$N = 1$ 时就是式 {eq}`eq-ch0-ret` 本身；$\bv\theta = \mathbf 0$ 时第一项消失。**数据矩阵**：本式已经是一期的数据矩阵形式；$T$ 期各有一个，代码把它们叠成三维数组，`x` 的形状为 `(T, N, L+1)`，`x[t-1]` 就是 $\tilde{\mathbf Z}_{t-1}$。$\bv\theta$、$\bv\Gamma$ 由所有股票、所有时期共用，$\mathbf f_t$ 由同一期的所有股票共用，$\mathbf Z_{t-1}$、$\tilde{\mathbf Z}_{t-1}$ 与 $\bv\varepsilon_t$ 的每一行属于一只股票。

**随机贴现因子。**

$$
m_t = 1 - \mathbf b'(\mathbf f_t - \bv\mu), \qquad \underbrace{\mathbf b}_{K\times 1} = \underbrace{\bv\Sigma_f^{-1}}_{K\times K}\underbrace{\bv\mu}_{K\times 1} .
$$ (eq-ch0-sdf)

$m_t$ 是标量。代码中 `SimulatedMarket.sdf_weights` 用 `np.linalg.solve(params.factor_cov, mu)` 解线性方程组 $\bv\Sigma_f \mathbf b = \bv\mu$ 得到 $\mathbf b$，而不是先求逆矩阵再相乘（原因见记号表：解方程更快，舍入误差也更小）。模拟器中 $\bv\Sigma_f$ 为对角矩阵，于是 $\mathbf b$ 的第 $k$ 个分量化为 $b_k = \mu_k / \sigma_k^2$；但代码与本章的推导都按一般的 $\bv\Sigma_f$ 写，以后改用相关因子时只需修改 `MarketParams.factor_cov` 与因子的生成方式。

(ch0-unpack-sdfb)=
**拆开讲式 {eq}`eq-ch0-sdf`。** $m_t = 1 - \mathbf b'(\mathbf f_t - \bv\mu)$ 的分量、维度、逐项展开、默认参数下的数值与 $T$ 期堆叠形式，见 0.2 节“随机贴现因子”一节的[拆解](#ch0-unpack-sdf)。这里补上 $\mathbf b = \bv\Sigma_f^{-1}\bv\mu$ 在一般 $\bv\Sigma_f$ 下的含义。它等价于线性方程组 $\bv\Sigma_f\mathbf b = \bv\mu$，写出分量是

```{math}
:enumerated: false
\underbrace{\begin{pmatrix} \Sigma_{11} & \Sigma_{12} & \cdots & \Sigma_{1K} \\ \Sigma_{21} & \Sigma_{22} & \cdots & \Sigma_{2K} \\ \vdots & \vdots & \ddots & \vdots \\ \Sigma_{K1} & \Sigma_{K2} & \cdots & \Sigma_{KK} \end{pmatrix}}_{\bv\Sigma_f:\ K\times K}
\underbrace{\begin{pmatrix} b_1 \\ b_2 \\ \vdots \\ b_K \end{pmatrix}}_{\mathbf b:\ K\times 1}
= \underbrace{\begin{pmatrix} \mu_1 \\ \mu_2 \\ \vdots \\ \mu_K \end{pmatrix}}_{\bv\mu:\ K\times 1} ,
\qquad \text{即} \quad \sum_{l=1}^{K} \Sigma_{kl}\, b_l = \mu_k , \quad k = 1, \dots, K ,
```

其中 $\Sigma_{kl}$ 是 $\bv\Sigma_f$ 的 $(k, l)$ 元素，即第 $k$ 与第 $l$ 个因子的协方差，$\Sigma_{kk} = \sigma_k^2$。第 $k$ 个方程的左边是因子 $k$ 与“$\mathbf b$ 加权的因子组合” $\mathbf b'\mathbf f_t$ 的协方差，右边是因子 $k$ 的均值：$\mathbf b$ 的选取恰好使每个因子与这个组合的协方差等于它的均值，0.5 节定理第 2 条的证明用的正是这 $K$ 个方程。**特例**：$\bv\Sigma_f$ 为对角矩阵时，第 $k$ 个方程只剩一项 $\sigma_k^2 b_k = \mu_k$，即 $b_k = \mu_k/\sigma_k^2$；$K = 2$ 且两个因子相关时，解出 $b_1 = (\Sigma_{22}\mu_1 - \Sigma_{12}\mu_2)/(\Sigma_{11}\Sigma_{22} - \Sigma_{12}^2)$，一个因子的系数还取决于另一个因子的均值以及二者的协方差（下方代码核对）。**数据矩阵**：$\mathbf b$ 只有 $K$ 个数，由所有股票、所有时期共用；代码用 `np.linalg.solve` 直接解这 $K$ 个方程。

```{code-cell} python
# 拆开讲 b = Σ_f⁻¹μ：K = 2、两个因子相关时（假设数字），solve 的结果与 2×2 的显式解一致
Sig2 = np.array([[0.0025, 0.0006],
                 [0.0006, 0.0009]])                              # Σ_f: (2, 2)，协方差 Σ_12 = 0.0006
mu2 = np.array([0.006, 0.003])                                   # μ: (2,)
b2 = np.linalg.solve(Sig2, mu2)                                  # 解 Σ_f b = μ，b: (2,)
b1_formula = (Sig2[1, 1] * mu2[0] - Sig2[0, 1] * mu2[1]) / (Sig2[0, 0] * Sig2[1, 1] - Sig2[0, 1] ** 2)
print("b =", b2.round(4), "；显式解 b1 =", round(b1_formula, 4), "；不考虑相关时 μ1/σ1² =", mu2[0] / Sig2[0, 0])
assert np.isclose(b2[0], b1_formula) and np.allclose(Sig2 @ b2, mu2)
```

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

0.5 节的定理与推论、0.6 节的代码验证见下一页[0.5～0.6 SDF 的定价性质与代码验证](#sec-ch0-pricing)。
