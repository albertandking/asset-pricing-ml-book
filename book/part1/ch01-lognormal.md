---
short_title: 1.4 幂效用与对数正态消费
kernelspec:
  name: python3
  display_name: Python 3
numbering:
  enumerator: '1.%s'
  equation:
    continue: true
  figure:
    continue: true
  proof:proposition:
    continue: true
  exercise:
    template: 习题 %s
---

(sec-ch1-lognormal)=
# 1.4 幂效用与对数正态消费

**本页准备。** 本书每个页面的代码在独立的内核中运行，前面页面定义的变量在这里不可用。下面的单元导入本页用到的模块，并取 `ConsumptionParams` 的默认参数（本节的全部数值例子都用它）。

```{code-cell} python
import math

import numpy as np
import pandas as pd
import sympy as sp

from apsim import ConsumptionParams

p = ConsumptionParams()                                    # 默认的年度假设参数
```

1.1～1.3 节的结论对任意分布都成立，但没有告诉我们无风险利率、风险溢价具体是多少，取决于哪些参数。本节加上两条假设：效用是幂效用 {eq}`eq-ch1-power`，消费增长与红利增长服从联合正态分布。在这两条假设下，所有量都有闭式解。这与 Hansen 与 Singleton 检验消费模型时采用的对数正态设定同类（他们假设消费增长与资产收益联合对数正态，并允许条件均值随时间变化）[@hansen1983stochastic]，本书的模拟器 `apsim.consumption` 也按它实现。

## 对数记号

**白话。** 消费、红利、价格都是“水平”，它们的增长是乘法关系：明年的消费等于今年的消费乘以增长因子。取对数之后，乘法变成加法，增长因子的对数就是“对数增长”，它近似等于增长率，又便于套用正态分布。

**例子。** 消费从 100 增长到 102，增长率为 2%，对数增长为 $\log 102 - \log 100 = \log 1.02 \approx 0.0198$，与 2% 很接近。增长率越小，二者越接近；增长率为 −30% 时，对数增长约为 −0.357，差别就明显了。

**公式。** 本书沿用[记号表](../notation.md)的约定，小写字母表示对应大写变量的（自然）对数：

```{math}
:enumerated: false
c_t = \log C_t, \qquad d_t = \log D_t, \qquad \Delta c_{t+1} = c_{t+1} - c_t = \log\frac{C_{t+1}}{C_t}, \qquad \Delta d_{t+1} = d_{t+1} - d_t ,
```

其中 $D_t$ 是第 $t$ 期的红利。于是 $C_{t+1}/C_t = e^{\Delta c_{t+1}}$，幂效用的 SDF {eq}`eq-ch1-power-sdf` 写成 $m_{t+1} = \delta e^{-\gamma\Delta c_{t+1}}$。注意第 0 章面板中的 $r_{i,t}$ 是水平的超额收益，是这一约定声明过的例外；为了避免混淆，本章不用小写 $r$ 表示对数收益，而是直接写 $\log R_{t+1}$。

## 模型

对数消费增长与对数红利增长独立同分布：

$$
\Delta c_{t+1} = \mu_c + \sigma_c\varepsilon_{t+1}, \qquad \Delta d_{t+1} = \mu_d + \sigma_d\eta_{t+1}, \qquad \eta_{t+1} = \rho_{cd}\,\varepsilon_{t+1} + \sqrt{1-\rho_{cd}^2}\;\nu_{t+1},
$$ (eq-ch1-model)

其中 $\varepsilon_{t+1}$、$\nu_{t+1}$ 是相互独立的标准正态冲击，各自在时间上独立同分布，所有量都是标量。$\eta_{t+1}$ 也是标准正态的：$\Var(\eta) = \rho_{cd}^2 + (1 - \rho_{cd}^2) = 1$，并且 $\Cov(\varepsilon, \eta) = \rho_{cd}$。所以 $\rho_{cd}$ 是消费增长与红利增长的相关系数，$\Cov(\Delta c, \Delta d) = \rho_{cd}\sigma_c\sigma_d$（下标 $cd$ 指消费 $c$ 与红利 $d$，与代码字段 `rho_cd` 对应；[记号表](../notation.md)中不带下标的 $\rho$ 留给一般的相关系数与第 6 章的 Campbell–Shiller 常数）。

红利与消费是两个不同的序列：总消费主要由劳动收入等来源支撑，红利只是其中一小部分，波动也大得多。所以模型中的“股票”是一项**红利索取权**（dividend claim），即获得红利流 $D_{t+1}, D_{t+2}, \dots$ 的资产，红利与消费相关但不相等。$\rho_{cd} = 1$、$\sigma_d = \sigma_c$、$\mu_d = \mu_c$ 时红利增长与消费增长完全相同，这是 1.5 节的 Lucas 树。

| 记号 | 含义 | 默认值（假设参数） | `ConsumptionParams` 字段 |
|---|---|---|---|
| $\delta$ | 主观贴现因子 | 0.99 | `delta` |
| $\gamma$ | 风险厌恶系数 | 2 | `gamma` |
| $\mu_c$、$\sigma_c$ | 对数消费增长的均值、标准差 | 0.02、0.02 | `mu_c`、`sigma_c` |
| $\mu_d$、$\sigma_d$ | 对数红利增长的均值、标准差 | 0.02、0.12 | `mu_d`、`sigma_d` |
| $\rho_{cd}$ | 消费增长与红利增长的相关系数 | 0.5 | `rho_cd` |

参数都是年度的，只为说明理论，不是对任何市场的校准。

```{code-cell} python
print(p)
assert (p.delta, p.gamma, p.mu_c, p.sigma_c, p.mu_d, p.sigma_d, p.rho_cd) == (0.99, 2.0, 0.02, 0.02, 0.02, 0.12, 0.5)
```

(ch1-unpack-model)=
**拆开讲式 {eq}`eq-ch1-model`。** 本章的一期是一年。逐项看单位与时点：

- $\Delta c_{t+1}$：第 $t+1$ 年的对数消费增长，没有单位，近似等于消费增长率；在第 $t$ 期末未知
- $\mu_c$、$\sigma_c$：对数消费增长的年度均值与标准差（默认都是 0.02），是常数
- $\Delta d_{t+1}$、$\mu_d$、$\sigma_d$：对数红利增长及其年度均值、标准差（默认 0.02、0.12），含义同上
- $\varepsilon_{t+1}$、$\nu_{t+1}$：第 $t+1$ 年的两个标准正态冲击，以标准差为单位，彼此独立，在第 $t$ 期末未知
- $\eta_{t+1}$：红利增长的标准化冲击，由 $\varepsilon_{t+1}$ 与 $\nu_{t+1}$ 按权重 $\rho_{cd}$ 与 $\sqrt{1-\rho_{cd}^2}$ 合成
- $\rho_{cd}$：相关系数，没有单位（默认 0.5）

例如某年 $\varepsilon_{t+1} = 1$、$\nu_{t+1} = 0$（对数消费增长比均值高一个标准差，红利没有独立的消息）：对数消费增长 $\Delta c_{t+1} = 0.02 + 0.02 = 0.04$（水平增长 $e^{0.04} - 1 \approx 4.08\%$），$\eta_{t+1} = 0.5$，对数红利增长 $\Delta d_{t+1} = 0.02 + 0.12 \times 0.5 = 0.08$（水平增长约 8.33%）。若 $\varepsilon_{t+1} = 0$、$\nu_{t+1} = 1$，对数消费增长恰为均值 0.02（水平增长约 2.02%），而 $\eta_{t+1} = \sqrt{0.75} \approx 0.866$，对数红利增长 $\Delta d_{t+1} \approx 0.1239$（水平增长约 13.2%）。**向量写法**：把两个增长排成 $2\times 1$ 的列向量，式 {eq}`eq-ch1-model` 就是

```{math}
:enumerated: false
\underbrace{\begin{pmatrix} \Delta c_{t+1} \\ \Delta d_{t+1} \end{pmatrix}}_{2\times 1}
= \underbrace{\begin{pmatrix} \mu_c \\ \mu_d \end{pmatrix}}_{2\times 1}
+ \underbrace{\begin{pmatrix} \sigma_c & 0 \\ \rho_{cd}\sigma_d & \sigma_d\sqrt{1-\rho_{cd}^2} \end{pmatrix}}_{\mathbf L\ (2\times 2)}
\underbrace{\begin{pmatrix} \varepsilon_{t+1} \\ \nu_{t+1} \end{pmatrix}}_{2\times 1} , \qquad
\underbrace{\mathbf L\mathbf L'}_{2\times 2} = \begin{pmatrix} \sigma_c^2 & \rho_{cd}\sigma_c\sigma_d \\ \rho_{cd}\sigma_c\sigma_d & \sigma_d^2 \end{pmatrix} .
```

$2\times 2$ 的 $\mathbf L$ 乘 $2\times 1$ 的冲击向量得 $2\times 1$；展开第二行就是 $\Delta d_{t+1} = \mu_d + \rho_{cd}\sigma_d\varepsilon_{t+1} + \sigma_d\sqrt{1-\rho_{cd}^2}\,\nu_{t+1} = \mu_d + \sigma_d\eta_{t+1}$。下三角矩阵 $\mathbf L$ 是协方差矩阵 $\mathbf L\mathbf L'$ 的 Cholesky 分解，即第 0 章的 $\mathbf C$（第 0 章[拆解](#ch0-unpack-factor)中提到的写法）；$\mathbf L\mathbf L'$ 的对角元是两个增长的方差，非对角元是协方差 $\rho_{cd}\sigma_c\sigma_d$，默认为 $0.5 \times 0.02 \times 0.12 = 0.0012$。**特例**：$\rho_{cd} = 0$ 时 $\mathbf L$ 是对角矩阵，红利与消费独立；$\rho_{cd} = 1$、$\sigma_d = \sigma_c$、$\mu_d = \mu_c$ 时第二行与第一行完全相同，就是 Lucas 树。**数据矩阵**：模拟 $T$ 年，把每年的 $(\Delta c_t, \Delta d_t)$ 作为一行，得到 $T\times 2$ 的矩阵，即 `simulate_consumption` 返回值的 `data` 中的 `dc`、`dd` 两列；“模拟实验”一节的实验三中，五项资产共用同一列 `dc`，各有自己的一列 `dd`。

```{code-cell} python
# 拆开讲 eq-ch1-model：L = [[σ_c, 0], [ρσ_d, σ_d √(1−ρ²)]]，L L' 为 (Δc, Δd) 的协方差矩阵
L_chol = np.array([[p.sigma_c, 0.0],
                   [p.rho_cd * p.sigma_d, p.sigma_d * math.sqrt(1 - p.rho_cd**2)]])     # L: (2, 2)
mu_cd = np.array([p.mu_c, p.mu_d])                                                       # (μ_c, μ_d): (2,)
growth_10 = mu_cd + L_chol @ np.array([1.0, 0.0])                                       # (2,) + (2, 2) @ (2,) -> (2,)
growth_01 = mu_cd + L_chol @ np.array([0.0, 1.0])
print("(ε, ν) = (1, 0)：(Δc, Δd) =", growth_10.round(4), "；(ε, ν) = (0, 1)：", growth_01.round(4))
assert np.allclose(L_chol @ L_chol.T, [[0.0004, 0.0012], [0.0012, 0.0144]])
assert np.allclose(growth_10, [0.04, 0.08]) and np.allclose(growth_01, [0.02, 0.1239], atol=5e-5)
```

## 对数正态分布的期望

**白话。** 如果 $X$ 服从正态分布，$e^X$ 就服从**对数正态分布**。$e^X$ 的期望不等于 $e^{\E[X]}$，而是还要乘以一个大于 1 的因子：指数函数是凸的，$X$ 向上波动时 $e^X$ 增加得多，向下波动时减少得少，平均下来比 $e^{\E[X]}$ 大。$X$ 的方差越大，差别越大。

**例子。** 设某只股票对数红利增长的均值为 0、标准差为 0.12，即 $X \sim N(0, 0.12^2)$。$e^{\E[X]} = 1$，好像红利平均不增长；但 $\E[e^X] = e^{0.12^2/2} = e^{0.0072} \approx 1.0072$，红利水平的期望增长约 0.72%。

**公式。** 若 $X \sim N(\mu, \sigma^2)$，则

$$
\E\left[e^X\right] = e^{\mu + \sigma^2/2}, \qquad \log\E\left[e^X\right] = \E[X] + \tfrac12\Var(X) .
$$ (eq-ch1-lognormal)

推导只需配方：$X = \mu + \sigma Z$，$Z \sim N(0, 1)$，$\E[e^{\sigma Z}] = \int e^{\sigma z}\frac{1}{\sqrt{2\pi}}e^{-z^2/2}\,dz = e^{\sigma^2/2}\int \frac{1}{\sqrt{2\pi}}e^{-(z-\sigma)^2/2}\,dz = e^{\sigma^2/2}$，最后一个积分是 $N(\sigma, 1)$ 的密度的积分，等于 1。式中的 $\sigma^2/2$ 就是本章反复出现的“二阶项”。下面用 `sympy` 求出标准正态分布的矩母函数 $\E[e^{uZ}] = e^{u^2/2}$，后面的推导都用它。

```{code-cell} python
u, z = sp.symbols("u z", real=True)
mgf = sp.simplify(sp.integrate(sp.exp(u * z - z**2 / 2) / sp.sqrt(2 * sp.pi), (z, -sp.oo, sp.oo)))
print("E[exp(u Z)] =", mgf)
assert sp.simplify(mgf - sp.exp(u**2 / 2)) == 0

# 例子：X ~ N(0, 0.12²)；用 80 个节点的 Gauss–Hermite 求积核对 E[e^X]
nodes, weights = np.polynomial.hermite_e.hermegauss(80)
weights = weights / weights.sum()                          # 标准正态分布的求积节点与权重
EeX = weights @ np.exp(0.12 * nodes)
print(f"E[e^X] = {EeX:.6f}，公式 e^(σ²/2) = {math.exp(0.12**2 / 2):.6f}")
assert math.isclose(EeX, math.exp(0.0072), rel_tol=1e-12) and 1.0072 < EeX < 1.0073
```

(ch1-unpack-lognormal)=
**拆开讲式 {eq}`eq-ch1-lognormal`。** $X$ 是一个对数量（本章中是对数增长或对数 SDF），$\mu$、$\sigma^2$ 是它的均值与方差，单位分别与 $X$ 和 $X$ 的平方相同；$e^X$ 是对应的水平量（总增长因子），没有单位。第一式说：水平的期望等于“均值的指数”再乘以凸性修正 $e^{\sigma^2/2}$。第二式是两边取对数：期望水平的对数，等于对数的期望加上半个方差。例如对数红利增长 $\Delta d \sim N(0.02, 0.12^2)$ 时，$\E[e^{\Delta d}] = e^{0.02 + 0.0072} = e^{0.0272} \approx 1.0276$：红利水平的期望增长约 2.76%，比 $e^{0.02} - 1 \approx 2.02\%$ 多出的约 0.74 个百分点全部来自方差。本章用到它的地方有三类：$X = \Delta d$（红利水平的期望增长，1.5 节记作 $G$）；$X = -\gamma\Delta c$（求 $\E[m]$，下一小节）；$X = \log m + \log R$（风险溢价的证明）。**特例**：$\sigma = 0$ 时没有不确定性，$\E[e^X] = e^{\mu}$，修正项消失。

```{code-cell} python
# 拆开讲 eq-ch1-lognormal：Δd ~ N(0.02, 0.12²) 时 E[e^{Δd}] = e^{0.0272}
EeDd = weights @ np.exp(p.mu_d + p.sigma_d * nodes)                 # (80,) @ (80,)：按求积权重取期望
print(f"E[e^Δd] = {EeDd:.6f}，e^μ_d = {math.exp(p.mu_d):.6f}")
assert math.isclose(EeDd, math.exp(0.0272), rel_tol=1e-12) and 1.0275 < EeDd < 1.0276
assert 0.0073 < EeDd - math.exp(p.mu_d) < 0.0075
```

## 无风险利率

:::{prf:proposition} 无风险利率
:label: prop-ch1-rf
在幂效用 {eq}`eq-ch1-power` 与模型 {eq}`eq-ch1-model` 下，$\E_t[m_{t+1}] = \delta\exp\left(-\gamma\mu_c + \gamma^2\sigma_c^2/2\right)$，无风险利率是常数，

$$
\log R^f = -\log\delta + \gamma\mu_c - \frac{\gamma^2\sigma_c^2}{2} .
$$ (eq-ch1-logrf)
:::

:::{prf:proof}
$m_{t+1} = \delta e^{-\gamma\Delta c_{t+1}}$，其中 $-\gamma\Delta c_{t+1} \sim N(-\gamma\mu_c, \gamma^2\sigma_c^2)$。由于冲击独立同分布，以第 $t$ 期末信息为条件的分布与无条件分布相同。由式 {eq}`eq-ch1-lognormal`，

```{math}
:enumerated: false
\E_t[m_{t+1}] = \delta\, \E\left[e^{-\gamma\Delta c_{t+1}}\right] = \delta \exp\left(-\gamma\mu_c + \tfrac12\gamma^2\sigma_c^2\right) .
```

它不随 $t$ 变化。由式 {eq}`eq-ch1-rf`，$R^f = 1/\E_t[m_{t+1}]$，取对数即得式 {eq}`eq-ch1-logrf`。
:::

下面用 `sympy` 推出同一结果：把 $m$ 写成 $\delta e^{-\gamma\mu_c}e^{-\gamma\sigma_c\varepsilon}$，用矩母函数求期望，再取 $-\log$ 并展开。

```{code-cell} python
delta_s, gamma_s, sc_s, sd_s = sp.symbols("delta gamma sigma_c sigma_d", positive=True)
muc_s, mud_s, rho_s = sp.symbols("mu_c mu_d rho", real=True)
Em_sym = delta_s * sp.exp(-gamma_s * muc_s) * mgf.subs(u, -gamma_s * sc_s)        # E[m] = δ e^{−γμ_c} E[e^{−γσ_c ε}]
logRf_sym = sp.expand(sp.expand_log(-sp.log(Em_sym), force=True))
print("log R^f =", logRf_sym)
assert sp.simplify(logRf_sym - (-sp.log(delta_s) + gamma_s * muc_s - gamma_s**2 * sc_s**2 / 2)) == 0
assert math.isclose(math.log(p.risk_free), -math.log(p.delta) + p.gamma * p.mu_c - p.gamma**2 * p.sigma_c**2 / 2,
                    rel_tol=1e-14)
```

式 {eq}`eq-ch1-logrf` 的三项各有含义。下表用默认参数把对数无风险利率拆开，代码随后核对。

| 项 | 默认参数下的值 | 名称 | 含义 |
|---|---|---|---|
| $-\log\delta$ | 约 1.005% | 耐心 | 越没耐心（$\delta$ 越小），越想现在消费，要让人储蓄，利率必须越高；没有增长、没有风险时 $R^f = 1/\delta$ |
| $\gamma\mu_c$ | 4% | 跨期替代 | 预期明年更富，人们想借钱提前消费；但代表性投资者整体无法借入，利率上升到没人想借为止 |
| $-\gamma^2\sigma_c^2/2$ | −0.08% | 预防性储蓄 | 未来越不确定，越想多存一点以备坏时候，储蓄需求压低利率 |
| 合计 $\log R^f$ | 约 4.925% | | $R^f \approx 1.0505$ |

第二项中，对数消费增长的均值 $\mu_c$ 乘的是 $\gamma$：$\gamma$ 越大，投资者越不愿意让各期消费不均，预期增长对利率的推动越强。$1/\gamma$ 称为**跨期替代弹性**（elasticity of intertemporal substitution，EIS），它衡量利率每变化 1 个百分点，消费增长愿意变化多少个百分点。幂效用下它与风险厌恶系数互为倒数，被同一个 $\gamma$ 绑定（见“常见误区”一节）。第三项在默认参数下很小，因为年度对数消费增长的标准差只有 0.02；但它与 $\gamma^2$ 成正比，$\gamma$ 较大时迅速变大。

(ch1-unpack-logrf)=
**拆开讲式 {eq}`eq-ch1-logrf`。** 左边的 $\log R^f$ 是对数无风险利率，即连续复利的年利率（默认约 4.925%）；通常说的年利率是 $R^f - 1$（默认约 5.05%），二者只差一点，利率越低差得越少。右边各项都是“对数的年利率”，三项的名称与含义见上表，这里补上单位：$-\log\delta$ 在 $\delta$ 接近 1 时约等于 $1 - \delta$（$\delta = 0.99$ 时约 1.005%）；$\gamma\mu_c$ 中 $\mu_c$ 是对数消费增长的年度均值，$\gamma$ 没有单位；$\gamma^2\sigma_c^2/2$ 中 $\sigma_c^2$ 是对数消费增长的年度方差（默认 0.0004）。$R^f$ 不带时间下标，因为在独立同分布的模型中每年都一样：它在每个第 $t$ 期末已知，并且总是同一个常数；[习题 %s](#ex-ch1-logrf) 让预期增长随时间变化，那时 $R^f_t$ 才随 $t$ 变化。**特例**：$\sigma_c = 0$（没有消费风险）时预防性储蓄项消失，$\log R^f = -\log\delta + \gamma\mu_c$；再取 $\mu_c = 0$，只剩耐心项，$R^f = 1/\delta$（1.2 节的[拆解](#ch1-unpack-rf)）；$\gamma = 1$（对数效用）时 $\log R^f = -\log\delta + \mu_c - \sigma_c^2/2$。

```{code-cell} python
terms = pd.Series({"耐心 −log δ": -math.log(p.delta),
                   "跨期替代 γμ_c": p.gamma * p.mu_c,
                   "预防性储蓄 −γ²σ_c²/2": -p.gamma**2 * p.sigma_c**2 / 2})
print((100 * terms).round(4).to_string(), f"\n合计 log R^f = {100 * terms.sum():.4f}%，R^f = {p.risk_free:.4f}")
assert math.isclose(terms.sum(), math.log(p.risk_free), rel_tol=1e-14)
assert np.allclose(terms, [0.01005, 0.04, -0.0008], atol=1e-5) and 1.0504 < p.risk_free < 1.0506
# 比较静态：μ_c 提高 1 个百分点，log R^f 提高 γ × 1%；σ_c 加倍，预防性储蓄项变为 4 倍
up_mu = ConsumptionParams(mu_c=0.03)
assert math.isclose(math.log(up_mu.risk_free) - math.log(p.risk_free), p.gamma * 0.01, rel_tol=1e-10)
up_sd = ConsumptionParams(sigma_c=0.04)
assert math.isclose(math.log(up_sd.risk_free) - math.log(p.risk_free), -3 * p.gamma**2 * p.sigma_c**2 / 2, rel_tol=1e-10)
```

```{code-cell} python
# 拆开讲 eq-ch1-logrf：连续复利 log R^f 与年利率 R^f − 1；−log δ ≈ 1 − δ；σ_c = 0 时预防性储蓄项消失
print(f"log R^f = {math.log(p.risk_free):.4%}，R^f − 1 = {p.risk_free - 1:.4%}，−log δ = {-math.log(p.delta):.4%}")
assert 0.0492 < math.log(p.risk_free) < 0.0493 and 0.0504 < p.risk_free - 1 < 0.0506
assert abs(-math.log(p.delta) - (1 - p.delta)) < 1e-4
no_risk = ConsumptionParams(sigma_c=0.0)
assert math.isclose(math.log(no_risk.risk_free), -math.log(p.delta) + p.gamma * p.mu_c, rel_tol=1e-12)
```

## 风险溢价

:::{prf:proposition} 对数正态下的风险溢价
:label: prop-ch1-premium
设 $\log m_{t+1}$ 与某项资产的对数总收益 $\log R_{t+1}$ 服从联合正态分布，则

$$
\log\frac{\E_t[R_{t+1}]}{R^f_t} = -\Cov_t(\log m_{t+1}, \log R_{t+1}) = \gamma\,\Cov_t(\Delta c_{t+1}, \log R_{t+1}) .
$$ (eq-ch1-logprem)

对本章模型中的红利索取权，1.5 节将证明 $\log R_{t+1} = \Delta d_{t+1} - \log a$（$a$ 是常数），于是 $\log(\E[R]/R^f) = \gamma\Cov(\Delta c, \Delta d) = \gamma\rho_{cd}\sigma_c\sigma_d$。
:::

:::{prf:proof}
以下省略下标，$\E$、$\Var$、$\Cov$ 都以第 $t$ 期末信息为条件。三个关系都来自式 {eq}`eq-ch1-lognormal`。

$\log m$ 与 $\log R$ 都是第 $t+1$ 期才揭晓的随机变量（对数 SDF 与连续复利的年收益），$\E$、$\Var$、$\Cov$ 是它们在第 $t$ 期末已知的条件矩。第 1 条把式 {eq}`eq-ch1-lognormal` 用于 $X = \log m + \log R$，它的方差按 $\Var(Y + Z) = \Var(Y) + \Var(Z) + 2\Cov(Y, Z)$ 展开；第 2、3 条分别取 $X = \log m$ 与 $X = \log R$。

1. 由 $1 = \E[mR] = \E[e^{\log m + \log R}]$，而 $\log m + \log R$ 是正态的：$0 = \E[\log m] + \E[\log R] + \tfrac12\left[\Var(\log m) + \Var(\log R) + 2\Cov(\log m, \log R)\right]$。
2. 由 $R^f = 1/\E[m] = 1/\E[e^{\log m}]$：$\log R^f = -\E[\log m] - \tfrac12\Var(\log m)$。
3. 由 $\E[R] = \E[e^{\log R}]$：$\log\E[R] = \E[\log R] + \tfrac12\Var(\log R)$。

把第 1 条改写为 $\E[\log R] + \tfrac12\Var(\log R) = -\E[\log m] - \tfrac12\Var(\log m) - \Cov(\log m, \log R)$，左边由第 3 条是 $\log\E[R]$，右边前两项由第 2 条是 $\log R^f$，所以 $\log\E[R] - \log R^f = -\Cov(\log m, \log R)$。最后，$\log m = \log\delta - \gamma\Delta c$，$\log\delta$ 是常数，$\Cov(\log m, \log R) = -\gamma\Cov(\Delta c, \log R)$。

对红利索取权，$\log R = \Delta d - \log a$，减去常数不改变协方差，$\Cov(\Delta c, \log R) = \Cov(\Delta c, \Delta d) = \rho_{cd}\sigma_c\sigma_d$。
:::

在联合正态的假设下，式 {eq}`eq-ch1-logprem` 是精确的，没有任何近似。它与式 {eq}`eq-ch1-premium` 说的是同一件事：只有与 $m$ 的协方差得到补偿，在幂效用下就是与消费增长的协方差。不过左边是**期望总收益之比的对数**，下文简称**对数溢价**，它不是期望超额收益本身，二者的关系是 $\E[R] - R^f = R^f\left(e^{\gamma\rho_{cd}\sigma_c\sigma_d} - 1\right)$。

(ch1-unpack-logprem)=
**拆开讲式 {eq}`eq-ch1-logprem`。** 式中三个量都是“对数”的。$\log(\E_t[R_{t+1}]/R^f_t)$ 是期望总收益与无风险总收益之比的对数，即对数溢价，没有单位，是年度的量（默认 0.24%），在第 $t$ 期末已知。$\log m_{t+1}$ 与 $\log R_{t+1}$ 是 SDF 与总收益的对数，$\log R_{t+1}$ 就是连续复利的年收益；二者在第 $t$ 期末都是随机的，但它们的条件协方差是已知的数。第二个等号用到 $\log m_{t+1} = \log\delta - \gamma\Delta c_{t+1}$：常数 $\log\delta$ 不影响协方差，$-\gamma$ 提到协方差之外。代入红利索取权：$\Cov(\Delta c, \log R) = \Cov(\Delta c, \Delta d) = 0.5 \times 0.02 \times 0.12 = 0.0012$，乘以 $\gamma = 2$ 得 0.0024，即 0.24%；换成期望超额收益，$\E[R] - R^f = 1.0505 \times (e^{0.0024} - 1) \approx 0.2524\%$。**特例**：与消费增长不相关的资产（$\rho_{cd} = 0$），对数溢价为 0，期望收益等于无风险利率；把式子用于无风险资产本身，$\log R^f$ 是常数，协方差为 0，左边 $\log(R^f/R^f) = 0$，前后一致。

下面先用 `sympy` 核对红利索取权的结果。1.5 节将推出 $a = \E[m e^{\Delta d}]$ 与 $\E[R] = \E[e^{\Delta d}]/a$；这里先按定义用矩母函数求出它们（$\varepsilon$、$\nu$ 独立，期望可以分开），再核对 $\log(\E[R]/R^f) = \gamma\rho_{cd}\sigma_c\sigma_d$，并与模拟器的属性 `expected_return`、`risk_free`、`log_equity_premium` 比较。

```{code-cell} python
# log(m e^{Δd}) = log δ − γμ_c + μ_d + (ρσ_d − γσ_c) ε + σ_d √(1−ρ²) ν
a_sym = (delta_s * sp.exp(mud_s - gamma_s * muc_s)
         * mgf.subs(u, rho_s * sd_s - gamma_s * sc_s) * mgf.subs(u, sd_s * sp.sqrt(1 - rho_s**2)))
ER_sym = sp.exp(mud_s + sd_s**2 / 2) / a_sym                                     # E[R] = E[e^{Δd}] / a
log_prem_sym = sp.simplify(sp.expand_log(sp.log(ER_sym * Em_sym), force=True))  # log(E[R] / R^f) = log(E[R] E[m])
print("log(E[R]/R^f) =", log_prem_sym)
assert sp.simplify(log_prem_sym - gamma_s * rho_s * sc_s * sd_s) == 0

log_prem = math.log(p.expected_return / p.risk_free)
arith_prem = p.expected_return - p.risk_free
print(f"log(E[R]/R^f) = {log_prem:.6f}，γρσ_cσ_d = {p.gamma * p.rho_cd * p.sigma_c * p.sigma_d:.6f}，"
      f"属性 log_equity_premium = {p.log_equity_premium:.6f}")
print(f"E[R] = {p.expected_return:.6f}，R^f = {p.risk_free:.6f}，E[R] − R^f = {arith_prem:.6f}")
assert math.isclose(log_prem, p.log_equity_premium, rel_tol=1e-12) and math.isclose(p.log_equity_premium, 0.0024)
assert math.isclose(arith_prem, p.risk_free * math.expm1(p.log_equity_premium), rel_tol=1e-12)
assert 0.00252 < arith_prem < 0.00253
```

默认参数下，红利索取权的期望收益只比无风险利率高约 0.25 个百分点：对数消费增长的标准差只有 0.02，红利与消费的相关系数为 0.5，$\gamma = 2$，三者相乘再乘以红利的波动 12%，协方差很小。这个数小，是因为默认参数小，并不是对任何市场的判断；合理的 $\gamma$ 能否产生数据中的股权溢价，是第 7 章“股权溢价之谜” [@mehra1985equity] 的主题。

(sec-ch1-ccapm)=
## 消费 CAPM：以消费增长为因子的 β 模型

**白话。** 式 {eq}`eq-ch1-premium` 说期望超额收益由“收益与 $m$ 的协方差”决定，而 $m$ 几乎就是消费增长的线性函数。把这个协方差拆成乘积，就得到第 0 章熟悉的 β 定价形式：资产对消费增长的敏感度 $\beta_{\Delta c}$（消费增长每高 1 个百分点，收益平均高多少个百分点）乘以每单位敏感度的补偿 $\lambda_c$。这就是只有一个因子的**消费 CAPM**（consumption CAPM）。与第 0 章的根本区别是：消费增长是**不可交易的因子**，它不是任何一项资产的收益，市场上没有一项零成本头寸恰好支付 $\Delta c$。

**例子。** 默认参数下，红利索取权收益对消费增长的回归系数 $\beta_{\Delta c} = \Cov(R, \Delta c)/\Var(\Delta c) \approx 3.16$：消费增长每高 1 个百分点，它的收益平均高约 3.16 个百分点。每单位 β 的补偿是 $\lambda_c = \gamma\Var(\Delta c) = 2 \times 0.02^2 = 0.0008$，即每年 0.08%。二者相乘得 $3.16 \times 0.08\% \approx 0.2527\%$，与上面的精确值 0.2524% 只差约 0.0003 个百分点。要注意 $\lambda_c$ 只有 0.08%，而对数消费增长的均值 $\E[\Delta c] = 0.02$（即 2%），后者是前者的 25 倍；如果误把 $\E[\Delta c]$ 当作风险溢价，预测的期望超额收益就成了 $3.16 \times 2\% \approx 6.3\%$，错了 25 倍。

**公式。** 把 SDF 写成 $m = \E[m]\exp\left(-\gamma(\Delta c - \mu_c) - \gamma^2\sigma_c^2/2\right)$（两边取期望可以验证），在 $\Delta c = \mu_c$ 处对指数做一阶展开，并略去二阶小量 $\gamma^2\sigma_c^2/2$：

```{math}
:enumerated: false
m_{t+1} \approx \E[m]\left[1 - \gamma(\Delta c_{t+1} - \mu_c)\right] .
```

这是消费 CAPM 推导中唯一的近似，1.6 节会把它与第 0 章的线性 SDF 逐项对照（各项的含义与单位见那里的[拆解](#ch1-unpack-linear)）。代入式 {eq}`eq-ch1-premium`，常数项与 $\mu_c$ 不影响协方差，$\Cov(m, R^e) \approx -\E[m]\,\gamma\,\Cov(\Delta c, R^e)$，再利用 $R^f\E[m] = 1$：

$$
\E[R^e] \approx \gamma\,\Cov(\Delta c, R^e) = \underbrace{\frac{\Cov(R^e, \Delta c)}{\Var(\Delta c)}}_{\beta_{\Delta c}}\;\underbrace{\gamma\Var(\Delta c)}_{\lambda_c} = \beta_{\Delta c}\,\lambda_c ,
$$ (eq-ch1-ccapm)

其中 $\beta_{\Delta c}$ 是超额收益对消费增长做时间序列回归的斜率，$\lambda_c = \gamma\sigma_c^2$ 是消费增长的风险溢价，都是标量。不同资产的 $\lambda_c$ 相同，期望超额收益的差别全部来自 $\beta_{\Delta c}$ 的差别。在连续时间中，这一关系是精确的，见 Breeden 的消费 CAPM [@breeden1979intertemporal]；离散时间下幂效用的定价公式见 Rubinstein [@rubinstein1976valuation]。

第 0 章 0.2 节说过，当因子本身是超额收益、并且模型为因子自身正确定价时，风险溢价就是因子的均值，$\lambda_k = \E[f_{k,t}]$，理由是把 β 定价式用到因子本身：它对自己的载荷为 1；那里同时说过，消费增长这类不可交易的因子不能套用这一推理，$\bv\lambda$ 一般不等于因子的均值。现在可以看清原因：$\Delta c$ 不是资产收益，定价方程 $0 = \E[m R^e]$ 不能用到 $\Delta c$ 本身，没有任何条件把 $\lambda_c$ 与 $\E[\Delta c]$ 联系起来。$\E[\Delta c] = \mu_c$ 是经济增长的速度，$\lambda_c = \gamma\sigma_c^2$ 由风险厌恶与消费的波动决定，改变 $\mu_c$ 不影响 $\lambda_c$。$\lambda_c$ 的含义是：$\beta_{\Delta c} = 1$ 的资产的期望超额收益。用资产收益构造一个尽量贴近消费增长的组合（模拟组合，factor-mimicking portfolio），可以把不可交易因子转为可交易因子，这是第 4、10 章的内容。

(ch1-unpack-ccapm)=
**拆开讲式 {eq}`eq-ch1-ccapm`。** 它是第 0 章 $\E[r_i] = \bv\beta_i'\bv\lambda$ 只有一个因子（$K = 1$）时的形式，向量写法的分量、维度与逐项展开见第 0 章的[拆解](#ch0-unpack-lambda)；这里逐项看标量的单位与时点。

- $\E[R^e]$：某项资产（这里是红利索取权）的期望超额收益，年度，是水平而不是对数，默认约 0.2524%
- $\Delta c$：年度对数消费增长；$\Var(\Delta c) = \sigma_c^2 = 0.0004$，单位是对数增长的平方
- $\beta_{\Delta c} = \Cov(R^e, \Delta c)/\Var(\Delta c)$：消费增长每高 1 个百分点，超额收益平均高多少个百分点，没有单位（默认约 3.16）；这是总体值，用数据估计时由时间序列回归的斜率 $\hat\beta_{\Delta c}$ 代替
- $\lambda_c = \gamma\Var(\Delta c)$：每单位 $\beta_{\Delta c}$ 得到的年度补偿，单位与收益相同（默认 $2 \times 0.0004 = 0.0008$，即 0.08%）

式中没有时间下标，因为独立同分布时条件矩与无条件矩相同。**展开运算**：$\beta_{\Delta c}\lambda_c \approx 3.16 \times 0.0008 \approx 0.2527\%$，它也等于 $\gamma\Cov(\Delta c, R^e) = 2 \times (3.16 \times 0.0004)$，即“风险厌恶乘以与消费增长的协方差”。**特例**：$\beta_{\Delta c} = 0$ 的资产期望超额收益为 0；$\beta_{\Delta c} = 1$ 的资产期望超额收益恰为 $\lambda_c$。**数据矩阵**：有 $N$ 项资产时，$N\times 1$ 的期望超额收益向量约等于 $N\times 1$ 的载荷列 $\mathbf B$ 乘以标量 $\lambda_c$；$\mathbf B$ 就是 $K = 1$ 时的载荷矩阵（只有一列，记法见[记号表](../notation.md)），第 $i$ 个元素是第 $i$ 项资产的 $\beta_{\Delta c,i}$，它是标量，不加粗；**所有资产共用同一个 $\lambda_c$ 与同一条消费增长序列**，每项资产各有一个 $\beta_{\Delta c,i}$。估计时的设计矩阵与横截面回归见“模拟实验”一节实验三的[拆解](#ch1-unpack-tsdc)。

下面的代码计算红利索取权的 $\beta_{\Delta c}$。对联合正态的 $(X, Y)$，$\Cov(e^X, Y) = \E[e^X]\Cov(X, Y)$（Stein 引理的一个特例），而 $R = e^{\Delta d}/a$，所以 $\Cov(R, \Delta c) = \E[R]\,\rho_{cd}\sigma_c\sigma_d$。代码先用二维 Gauss–Hermite 求积核对这个协方差：`np.meshgrid(nodes, nodes, indexing="ij")` 生成两个 `(80, 80)` 的网格，`E1[i, j]` 取 $\varepsilon$ 的第 $i$ 个节点、`N1[i, j]` 取 $\nu$ 的第 $j$ 个节点；`np.outer(weights, weights)` 是 `(80, 80)` 的权重矩阵，元素为 $w_i w_j$（$\varepsilon$、$\nu$ 独立，联合权重是边际权重之积）。于是 $\E[g(\varepsilon, \nu)] \approx \sum_{i,j} w_i w_j\, g(\varepsilon_i, \nu_j)$，即代码中的 `(W2 * g).sum()`。随后再比较 $\beta_{\Delta c}\lambda_c$、精确的期望超额收益与“误用 $\E[\Delta c]$”的结果。

```{code-cell} python
E1, N1 = np.meshgrid(nodes, nodes, indexing="ij")                     # 两个独立标准正态冲击 ε、ν 的节点 (80, 80)
W2 = np.outer(weights, weights)                                        # 对应的权重 (80, 80)
dc_q = p.mu_c + p.sigma_c * E1                                         # Δc 的节点值
dd_q = p.mu_d + p.sigma_d * (p.rho_cd * E1 + math.sqrt(1 - p.rho_cd**2) * N1)   # Δd 的节点值
R_q = np.exp(dd_q) / p.dividend_discount                               # R = e^{Δd} / a
cov_R_dc = (W2 * R_q * dc_q).sum() - (W2 * R_q).sum() * p.mu_c         # Cov(R, Δc) = E[R Δc] − E[R] E[Δc]
assert math.isclose(cov_R_dc, p.expected_return * p.rho_cd * p.sigma_c * p.sigma_d, rel_tol=1e-10)

beta_dc = cov_R_dc / p.sigma_c**2                                      # β_Δc = Cov(R^e, Δc) / Var(Δc)
lam_c = p.gamma * p.sigma_c**2                                         # λ_c = γ Var(Δc)
exact = p.expected_return - p.risk_free
print(f"β_Δc = {beta_dc:.4f}，λ_c = {lam_c:.4%}，E[Δc] = {p.mu_c:.2%}")
print(f"β_Δc·λ_c = {beta_dc * lam_c:.4%}，精确值 E[R] − R^f = {exact:.4%}，误用 β_Δc·E[Δc] = {beta_dc * p.mu_c:.2%}")
assert 3.15 < beta_dc < 3.17 and math.isclose(lam_c, 0.0008) and math.isclose(p.mu_c / lam_c, 25.0)
assert abs(beta_dc * lam_c - exact) < 5e-6 and 0.063 < beta_dc * p.mu_c < 0.064
```

(ch1-unpack-quad)=
**拆开讲二维求积的矩阵写法。** 记 80 个节点的权重组成的列向量为 $\mathbf w = (w_1, \dots, w_{80})'$（$80\times 1$），则 `np.outer(weights, weights)` 是外积 $\mathbf w\mathbf w'$：$80\times 1$ 乘 $1\times 80$，得 $80\times 80$ 的矩阵，$(i, j)$ 元素为 $w_iw_j$（外积的写法见第 0 章的[拆解](#ch0-unpack-outer)）。把被积函数在网格上的取值排成 $80\times 80$ 的矩阵 $\mathbf G$，它的 $(i, j)$ 元素是 $g(\varepsilon_i, \nu_j)$，行对应 $\varepsilon$ 的节点、列对应 $\nu$ 的节点，则

```{math}
:enumerated: false
\E[g(\varepsilon, \nu)] \approx \sum_{i=1}^{80}\sum_{j=1}^{80} w_i\, w_j\, g(\varepsilon_i, \nu_j) = \underbrace{\mathbf w'}_{1\times 80}\,\underbrace{\mathbf G}_{80\times 80}\,\underbrace{\mathbf w}_{80\times 1} ,
```

结果是标量。代码中的 `(W2 * g).sum()` 先逐元素相乘、再全部加总，与 `weights @ g @ weights` 相同。**特例**：$g$ 只依赖 $\varepsilon$ 时，$\mathbf G$ 的每一列都相同，由于权重之和为 1，二维求积退化为一维的 $\sum_i w_i\, g(\varepsilon_i)$，例如上面的 `(W2 * dc_q).sum()` 就是 $\E[\Delta c] = \mu_c$。

```{code-cell} python
# 拆开讲二维求积：(W2 * g).sum() 等于 w' G w；g 只依赖 ε 时退化为一维求积
assert math.isclose((W2 * R_q).sum(), weights @ R_q @ weights, rel_tol=1e-12)          # (80,) @ (80, 80) @ (80,) -> 标量
assert math.isclose((W2 * dc_q).sum(), weights @ (p.mu_c + p.sigma_c * nodes), rel_tol=1e-12)
assert math.isclose((W2 * dc_q).sum(), p.mu_c, rel_tol=1e-12)
print("二维求积与 w'Gw 一致；E[Δc] 的二维求积 =", round((W2 * dc_q).sum(), 6))
```

(sec-ch1-approx)=
## 哪些关系是近似

本节的公式中，有的精确成立，有的是近似。下表按“在哪一步引入了近似”整理。

| 关系 | 成立条件 | 近似从哪一步引入 |
|---|---|---|
| $P = \E[mx]$、$R^f = 1/\E[m]$、$\E[R^e] = -R^f\Cov(m, R^e)$ | 任意分布，精确 | 无 |
| $\log R^f = -\log\delta + \gamma\mu_c - \gamma^2\sigma_c^2/2$、$\log(\E[R]/R^f) = \gamma\Cov(\Delta c, \log R)$ | 联合正态时精确 | 冲击不是正态时，相当于只保留到方差（二阶项），略去了偏度、峰度等更高阶的项 |
| $\E[R^e] \approx \beta_{\Delta c}\lambda_c = \gamma\Cov(\Delta c, R^e)$ | 近似 | 把 $m$ 在 $\Delta c = \mu_c$ 处线性化，略去 $(\Delta c - \mu_c)$ 的二次及更高次项 |
| $\E[R]/R^f - 1 \approx \gamma\rho_{cd}\sigma_c\sigma_d$ | 近似 | 用 $\log(1 + y) \approx y$ 把对数溢价当作算术溢价 |

**线性化的误差。** 在本章模型中两种近似的误差都可以精确算出。记 $x = \gamma\rho_{cd}\sigma_c\sigma_d$。由 $\log(\E[R]/R^f) = x$，精确的期望超额收益为 $\E[R] - R^f = \E[R](1 - e^{-x})$；由 $\Cov(R, \Delta c) = \E[R]\rho_{cd}\sigma_c\sigma_d$，线性化的结果为 $\beta_{\Delta c}\lambda_c = \gamma\Cov(R, \Delta c) = \E[R]\,x$。二者之比为 $x/(1 - e^{-x}) \approx 1 + x/2$，所以线性化**高估**期望超额收益，相对误差约为 $x/2$；同理，把 $x$ 当作 $\E[R]/R^f - 1$ 则低估，相对误差约为 $-x/2$。误差随 $\gamma$ 线性增长。下表的代码对几个 $\gamma$ 计算这两种误差。

```{code-cell} python
rows = []
for g in (2, 5, 10, 25):
    q = ConsumptionParams(gamma=g)
    x_g = q.log_equity_premium                                         # x = γρσ_cσ_d
    ex = q.expected_return - q.risk_free                               # 精确的 E[R] − R^f
    lin = g * q.expected_return * q.rho_cd * q.sigma_c * q.sigma_d     # β_Δc λ_c = γ Cov(R, Δc)
    rows.append({"γ": g, "x = γρσ_cσ_d": x_g, "精确 E[R]−R^f": ex, "β·λ_c": lin,
                 "线性化相对误差": lin / ex - 1,
                 "log(1+y)≈y 相对误差": q.risk_free * x_g / ex - 1})   # E[R] − R^f ≈ R^f x
approx = pd.DataFrame(rows).set_index("γ")
print(approx.round(5).to_string())
assert np.all(approx["线性化相对误差"] > 0) and np.all(approx["log(1+y)≈y 相对误差"] < 0)
assert np.allclose(approx["线性化相对误差"], approx["x = γρσ_cσ_d"] / 2, rtol=0.02)
assert np.allclose(approx["log(1+y)≈y 相对误差"], -approx["x = γρσ_cσ_d"] / 2, rtol=0.02)
assert approx.loc[2, "线性化相对误差"] < 0.0013 and 0.014 < approx.loc[25, "线性化相对误差"] < 0.016
```

默认参数（$\gamma = 2$）下线性化的相对误差约为 0.12%，可以忽略；$\gamma = 25$ 时约为 1.5%，仍然不大。原因是年度消费增长的波动很小，$m$ 在它的取值范围内几乎是线性的。

**非正态冲击的误差。** 对数正态的公式在冲击服从正态分布时精确，否则只是保留到二阶项的近似。为了看清误差有多大，把 $\varepsilon$ 换成一个均值仍为 0、方差仍为 1、但有一个“灾难”尾部的混合分布（假设参数）：以 98% 的概率来自 $N(m_1, s^2)$，以 2% 的概率来自 $N(-5, s^2)$；$m_1$ 与 $s^2$ 由“均值为 0、方差为 1”两个条件确定。灾难状态下对数消费增长 $\Delta c$ 的均值约为 $0.02 - 5 \times 0.02 = -0.08$（水平下降约 7.7%）。均值、方差不变，式 {eq}`eq-ch1-logrf` 给出的无风险利率也就不变；精确值由混合正态的矩母函数算出：$\E[e^{u\varepsilon}] = 0.98\,e^{u m_1 + u^2 s^2/2} + 0.02\,e^{-5u + u^2 s^2/2}$。

```{code-cell} python
p_dis, m_dis = 0.02, -5.0                                  # 灾难成分的概率与均值（假设参数）
m_1 = -p_dis * m_dis / (1 - p_dis)                         # 均值为 0：(1 − p) m_1 + p m_dis = 0
s2 = 1 - (1 - p_dis) * m_1**2 - p_dis * m_dis**2           # 方差为 1：s² + (1 − p) m_1² + p m_dis² = 1
assert math.isclose((1 - p_dis) * m_1 + p_dis * m_dis, 0.0, abs_tol=1e-15) and s2 > 0
mix_mgf = lambda v: (1 - p_dis) * math.exp(v * m_1 + v**2 * s2 / 2) + p_dis * math.exp(v * m_dis + v**2 * s2 / 2)

rows = []
for g in (2, 5, 10):
    q = ConsumptionParams(gamma=g)
    logRf_normal = math.log(q.risk_free)                                           # 式 eq-ch1-logrf：只用到均值与方差
    logRf_mix = -math.log(q.delta * math.exp(-g * q.mu_c) * mix_mgf(-g * q.sigma_c))  # 混合分布下的精确值
    rows.append({"γ": g, "对数正态公式": logRf_normal, "混合分布精确值": logRf_mix, "差": logRf_mix - logRf_normal})
nonnormal = pd.DataFrame(rows).set_index("γ")
print(f"m_1 = {m_1:.4f}，s² = {s2:.4f}")
print(nonnormal.round(5).to_string())
assert np.all(nonnormal["差"] < 0) and np.all(np.diff(nonnormal["差"].abs()) > 0)
assert abs(nonnormal.loc[2, "差"]) < 5e-5 and 0.004 < abs(nonnormal.loc[10, "差"]) < 0.0045
```

在 $\gamma = 2$ 时，灾难尾部只使对数无风险利率降低约 0.003 个百分点，可以忽略；在 $\gamma = 10$ 时降低约 0.43 个百分点，已不可忽略。原因是灾难状态下消费骤降，$m = \delta e^{-\gamma\Delta c}$ 在那里取很大的值，$\gamma$ 越大越明显；这部分贡献来自偏度与峰度，对数正态公式只保留方差，看不到它。本章其余部分都在正态假设下进行，这些公式因而是精确的。
