---
short_title: 本章小结、习题与延伸阅读
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

# 本章小结、习题与延伸阅读

**本页准备。** 本书每个页面的代码在独立的内核中运行，前面页面定义的变量在这里不可用。下面的单元导入本页用到的模块，并取 `ConsumptionParams` 的默认参数。

```{code-cell} python
import math

import numpy as np
import pandas as pd
import sympy as sp
from scipy.optimize import brentq
from scipy.stats import norm

from apsim import ConsumptionParams, simulate_consumption

p = ConsumptionParams()                                    # 默认的年度假设参数
```

(sec-ch1-summary)=
## 本章小结

1. 两期消费—储蓄问题的一阶条件 $P_t u'(C_t) = \delta\E_t[u'(C_{t+1})x_{t+1}]$：边际代价等于边际收益，对投资者能自由买卖的每一项资产都成立（1.1 节）。
2. 由此得到基本定价方程 $P_t = \E_t[m_{t+1}x_{t+1}]$，$m_{t+1} = \delta u'(C_{t+1})/u'(C_t)$；幂效用下 $m_{t+1} = \delta(C_{t+1}/C_t)^{-\gamma}$。特例：$R^f_t = 1/\E_t[m_{t+1}]$、$1 = \E_t[m_{t+1}R_{t+1}]$、$0 = \E_t[m_{t+1}R^e_{t+1}]$。理论写法 $R^f_t$ 与第 0 章的面板写法 $R^f_{t-1}$ 只差时点标号；第 0 章的 $m_t$ 对应本章的 $R^f_t m_{t+1}$（1.2 节）。
3. 价格等于不考虑风险的价值加上风险调整，$P = \E[x]/R^f + \Cov(m, x)$；期望超额收益 $\E[R^e] = -R^f\Cov(m, R^e)$。坏时候 $m$ 高，在坏时候赚钱的资产像保险，期望收益低，甚至低于无风险利率；与 $m$ 不相关的波动得不到补偿（1.3 节）。
4. 对数正态下，$\log R^f = -\log\delta + \gamma\mu_c - \gamma^2\sigma_c^2/2$，三项分别对应耐心、跨期替代与预防性储蓄；$\log(\E[R]/R^f) = \gamma\Cov(\Delta c, \log R)$，对红利索取权为 $\gamma\rho\sigma_c\sigma_d$。二者在联合正态时精确，冲击非正态时只是保留到二阶项的近似（1.4 节）。
5. 消费 CAPM：$\E[R^e] \approx \beta_{\Delta c}\lambda_c$，$\lambda_c = \gamma\Var(\Delta c)$，来自 SDF 的线性化，默认参数下相对误差约 0.12%。消费增长是不可交易的因子，$\lambda_c$ 不等于 $\E[\Delta c]$（1.4 节、模拟实验三）。
6. 多期定价：$P_t = \E_t\sum_{j\ge1}M_{t,t+j}D_{t+j}$，需要横截性条件排除泡沫。独立同分布时价格—红利比为常数 $a/(1-a)$，$a = \E[m e^{\Delta d}] < 1$ 等价于期望收益高于红利的期望增长；收益 $R_{t+1} = e^{\Delta d_{t+1}}/a$ 不可预测。Lucas 树是红利等于消费的特例，$\gamma > 1$ 时预期增长提高反而降低价格—红利比（1.5 节）。
7. 第 0 章的线性 SDF 是消费 SDF 的一阶近似 $m/\E[m] \approx 1 - \gamma(\Delta c - \mu_c)$；两种情形都满足 $\bv\lambda = \bv\Sigma_f\mathbf b$，区别在于可交易因子必须为自身定价，从而 $\mathbf b = \bv\Sigma_f^{-1}\bv\mu$、$\bv\lambda = \bv\mu$（1.6 节）。
8. 中国的消费数据频率低、样本短、口径多、会修订，直接用消费数据检验定价模型很困难；在本章的默认参数下，几十年的样本甚至无法把风险溢价与 0 区分开（A 股视角、[习题 %s](#ex-ch1-short)）。

## 习题

### 概念题

:::{exercise} 期望收益低于无风险利率的资产
:label: ex-ch1-insurance
某研究者说：“1.3 节的保险型资产期望收益率只有约 3.02%，低于无风险利率 3.51%，但它是有风险的（两种状态下的支付不同），这违背了‘风险越高收益越高’。”这一说法错在哪里？在幂效用、对数正态的模型中，什么样的红利索取权期望超额收益为负？什么样的为 0？
:::

:::{solution} ex-ch1-insurance
:class: dropdown
“风险越高收益越高”中的风险，只能是与 $m$ 相关的那部分风险。由式 {eq}`eq-ch1-premium`，$\E[R^e] = -R^f\Cov(m, R^e)$：期望超额收益的符号由收益与 $m$ 的协方差决定，与收益自身的方差无关。保险型资产在坏状态（$m$ 高）支付多，$\Cov(m, R^e) > 0$，期望超额收益为负：持有它降低了消费的波动，投资者愿意为此接受低于无风险利率的期望收益，正如为保险支付保费。

在幂效用、对数正态的模型中，由式 {eq}`eq-ch1-logprem`，红利索取权的 $\log(\E[R]/R^f) = \gamma\rho\sigma_c\sigma_d$。所以 $\rho < 0$（红利在消费增长低时增长高）的资产期望超额收益为负；$\rho = 0$ 的资产，不论 $\sigma_d$ 多大，期望超额收益都恰好为 0（模拟实验三的第二项资产）。
:::

:::{exercise} 提高风险厌恶的代价
:label: ex-ch1-gamma
在默认参数下，红利索取权的对数溢价只有 0.24%。某研究者想把 $\gamma$ 从 2 提高到 10，让溢价变大。说明这样做对无风险利率的三项（式 {eq}`eq-ch1-logrf`）各有什么影响，对数溢价变为多少，并解释这一做法在幂效用下会带来什么新问题。
:::

:::{solution-start} ex-ch1-gamma
:class: dropdown
:::

对数溢价 $\gamma\rho\sigma_c\sigma_d$ 与 $\gamma$ 成正比，从 0.24% 变为 1.2%。无风险利率的三项中，耐心项 $-\log\delta$ 不变；跨期替代项 $\gamma\mu_c$ 从 4% 变为 20%；预防性储蓄项 $-\gamma^2\sigma_c^2/2$ 从 −0.08% 变为 −2%。合计对数无风险利率从约 4.9% 升到约 19%。

问题在于幂效用把风险厌恶与跨期替代弹性绑在一起：$\gamma = 10$ 意味着跨期替代弹性只有 0.1，投资者极不愿意让各期消费不均，预期消费增长 2% 时，必须有很高的利率才能让他不借钱提前消费。用提高 $\gamma$ 解释较高的风险溢价，同时会产生过高的无风险利率。这类问题的定量讨论见第 7 章（股权溢价之谜与无风险利率之谜）。

```{code-cell} python
def rf_terms(q):
    """式 eq-ch1-logrf 的三项：耐心、跨期替代、预防性储蓄。"""
    return pd.Series({"耐心": -math.log(q.delta), "跨期替代": q.gamma * q.mu_c,
                      "预防性储蓄": -q.gamma**2 * q.sigma_c**2 / 2})


both = pd.DataFrame({f"γ = {g}": rf_terms(ConsumptionParams(gamma=g)) for g in (2, 10)})
both.loc["合计 log R^f"] = both.sum()
both.loc["对数溢价"] = [ConsumptionParams(gamma=g).log_equity_premium for g in (2, 10)]
print((100 * both).round(3).to_string())
assert np.allclose(both["γ = 10"], [0.01005, 0.20, -0.02, 0.19005, 0.012], atol=1e-5)
assert math.isclose(both.loc["合计 log R^f", "γ = 10"], math.log(ConsumptionParams(gamma=10).risk_free))
```
:::{solution-end}
:::

### 推导题

:::{exercise} 时变的预期增长与无风险利率
:label: ex-ch1-logrf
设对数消费增长为 $\Delta c_{t+1} = \mu_{c,t} + \sigma_c\varepsilon_{t+1}$，其中 $\mu_{c,t}$ 在第 $t$ 期末已知（例如由当期的经济状况决定），$\varepsilon_{t+1} \sim N(0, 1)$ 与第 $t$ 期末的信息独立。效用为幂效用。

1. 推导 $\log R^f_t$ 的闭式解。为什么这里必须写成 $R^f_t$ 而不能写成常数 $R^f$？
2. 如果 $\mu_{c,t}$ 在经济衰退时较低，衰退时的无风险利率比平时高还是低？用“跨期替代”一项解释。
:::

:::{solution-start} ex-ch1-logrf
:class: dropdown
:::

1. 以第 $t$ 期末信息为条件，$\mu_{c,t}$ 是已知常数，$-\gamma\Delta c_{t+1} \mid \mathcal F_t \sim N(-\gamma\mu_{c,t}, \gamma^2\sigma_c^2)$，其中 $\mathcal F_t$ 表示第 $t$ 期末的信息。由式 {eq}`eq-ch1-lognormal`，$\E_t[m_{t+1}] = \delta\exp(-\gamma\mu_{c,t} + \gamma^2\sigma_c^2/2)$，由式 {eq}`eq-ch1-rf`，

   ```{math}
   :enumerated: false
   \log R^f_t = -\log\delta + \gamma\mu_{c,t} - \frac{\gamma^2\sigma_c^2}{2} .
   ```

   与[命题 %s](#prop-ch1-rf) 的推导相同，只是条件期望不再等于无条件期望。$R^f_t$ 依赖于 $\mu_{c,t}$，随时间变化，并且在第 $t$ 期末已知、适用于从 $t$ 到 $t+1$，正是 1.2 节理论写法 $R^f_t$ 的含义。在独立同分布的模型中它恰好是常数，下标可以省略。
2. 衰退时 $\mu_{c,t}$ 低，跨期替代项 $\gamma\mu_{c,t}$ 小，无风险利率低。预期未来消费增长慢时，投资者想把今天的资源留到未来，储蓄意愿上升，均衡利率因此下降。$\gamma$ 越大（跨期替代弹性越小），利率对预期增长越敏感。

下面用 `sympy` 核对第 1 问：把 $\mu_{c,t}$ 当作一个符号常数，用矩母函数求条件期望。

```{code-cell} python
u, z = sp.symbols("u z", real=True)
mgf = sp.simplify(sp.integrate(sp.exp(u * z - z**2 / 2) / sp.sqrt(2 * sp.pi), (z, -sp.oo, sp.oo)))   # e^{u²/2}
d_s, g_s, sc_s = sp.symbols("delta gamma sigma_c", positive=True)
muct = sp.Symbol("mu_ct", real=True)                                     # μ_{c,t}：第 t 期末已知
Em_t = d_s * sp.exp(-g_s * muct) * mgf.subs(u, -g_s * sc_s)              # E_t[m_{t+1}]
logRf_t = sp.expand(sp.expand_log(-sp.log(Em_t), force=True))
print("log R^f_t =", logRf_t, "；对 μ_{c,t} 的导数 =", sp.diff(logRf_t, muct))
assert sp.simplify(logRf_t - (-sp.log(d_s) + g_s * muct - g_s**2 * sc_s**2 / 2)) == 0
assert sp.diff(logRf_t, muct) == g_s
```
:::{solution-end}
:::

:::{exercise} Lucas 树的价格—红利比
:label: ex-ch1-lucas
在 Lucas 树中红利等于消费，$\Delta d_{t+1} = \Delta c_{t+1} = \mu_c + \sigma_c\varepsilon_{t+1}$，效用为幂效用。

1. 从 $P_t = \E_t\sum_{j\ge1}M_{t,t+j}D_{t+j}$ 出发，推导 $P_t/D_t$ 的闭式解，写出价格有限的条件。
2. 证明对数效用（$\gamma = 1$）下 $P_t/D_t = \delta/(1-\delta)$，与 $\mu_c$、$\sigma_c$ 都无关，并解释原因。
3. 求 $\partial\log a/\partial\mu_c$，说明 $\gamma > 1$ 与 $\gamma < 1$ 时预期增长对价格—红利比的影响为何相反。
4. 证明总收益 $R_{t+1} = e^{\Delta c_{t+1}}/a$，并由此求 $\log(\E[R]/R^f)$。
:::

:::{solution-start} ex-ch1-lucas
:class: dropdown
:::

1. $M_{t,t+j}D_{t+j}/D_t = \delta^j(C_{t+j}/C_t)^{-\gamma}(C_{t+j}/C_t) = \delta^j\exp\left((1-\gamma)\sum_{k=1}^{j}\Delta c_{t+k}\right)$。各年的 $\Delta c$ 独立同分布，乘积的期望等于期望的乘积，每一个期望由式 {eq}`eq-ch1-lognormal` 为 $a = \delta\exp\left((1-\gamma)\mu_c + (1-\gamma)^2\sigma_c^2/2\right)$，所以 $\E_t[M_{t,t+j}D_{t+j}]/D_t = a^j$。求和得 $P_t/D_t = \sum_{j\ge1}a^j = a/(1-a)$，条件是 $a < 1$，即 $\log\delta + (1-\gamma)\mu_c + (1-\gamma)^2\sigma_c^2/2 < 0$；此时横截性条件 $\E_t[M_{t,t+J}P_{t+J}] = (P/D)\,D_t\,a^J \to 0$ 也成立。这与[命题 %s](#prop-ch1-pd) 取 $\rho = 1$、$\sigma_d = \sigma_c$、$\mu_d = \mu_c$ 的结果一致。
2. $\gamma = 1$ 时 $(1-\gamma) = 0$，$a = \delta$，$P/D = \delta/(1-\delta)$。原因是对数效用下 $m_{t+1}D_{t+1} = \delta (C_t/C_{t+1}) C_{t+1} = \delta C_t$ 是确定的：增长带来更多红利的“现金流效应”与增长使边际效用下降的“折现效应”恰好抵消，红利的任何变化都被边际效用的变化完全抵消。
3. $\partial\log a/\partial\mu_c = 1 - \gamma$。$P/D = a/(1-a)$ 是 $a$ 的增函数，所以 $\gamma > 1$ 时预期增长提高使价格—红利比下降，$\gamma < 1$ 时使它上升。直观上，$\mu_c$ 提高使未来红利增加（系数 1），同时由式 {eq}`eq-ch1-logrf` 使利率上升（系数 $\gamma$），净效应为 $1 - \gamma$。
4. 由 $P_t = vD_t$，$v = a/(1-a)$，$R_{t+1} = (v+1)/v \cdot D_{t+1}/D_t = e^{\Delta c_{t+1}}/a$（式 {eq}`eq-ch1-return`）。于是 $\log R_{t+1} = \Delta c_{t+1} - \log a$，由[命题 %s](#prop-ch1-premium)，$\log(\E[R]/R^f) = \gamma\Cov(\Delta c, \log R) = \gamma\sigma_c^2$。默认的 $\gamma = 2$、$\sigma_c = 2\%$ 下为 0.08%，恰好等于 $\lambda_c$：Lucas 树对消费增长的 β 近似为 1。

```{code-cell} python
g_s, mu_s, s_s, d_s = sp.symbols("gamma mu_c sigma_c delta", positive=True)
a_tree = d_s * sp.exp((1 - g_s) * mu_s + (1 - g_s) ** 2 * s_s**2 / 2)
print("∂log a/∂μ_c =", sp.simplify(sp.diff(sp.log(a_tree), mu_s)))
assert sp.simplify(sp.diff(sp.log(a_tree), mu_s) - (1 - g_s)) == 0
assert sp.simplify((a_tree / (1 - a_tree)).subs(g_s, 1) - d_s / (1 - d_s)) == 0          # γ = 1：P/D = δ/(1−δ)

for g in (1.0, 2.0, 5.0):                                                           # 与模拟器的属性比较
    tree = ConsumptionParams(gamma=g, mu_d=p.mu_c, sigma_d=p.sigma_c, rho_cd=1.0)
    a_num = float(a_tree.subs({g_s: g, mu_s: p.mu_c, s_s: p.sigma_c, d_s: p.delta}))
    assert math.isclose(tree.dividend_discount, a_num, rel_tol=1e-12)
    assert math.isclose(tree.log_equity_premium, g * p.sigma_c**2, rel_tol=1e-12)  # γσ_c²
tree2 = ConsumptionParams(mu_d=p.mu_c, sigma_d=p.sigma_c, rho_cd=1.0)
print(f"γ = 2 的 Lucas 树：P/D = {tree2.price_dividend_ratio:.2f}，log(E[R]/R^f) = {tree2.log_equity_premium:.4%}")
assert math.isclose(tree2.log_equity_premium, p.gamma * p.sigma_c**2)
```
:::{solution-end}
:::

### 编程题

:::{exercise} 多大的风险厌恶才能产生给定的溢价
:label: ex-ch1-grid
取默认参数，只改变 $\gamma$。

1. 对 $\gamma = 1, 2, \dots, 30$ 计算对数无风险利率、对数溢价 $\log(\E[R]/R^f)$ 与价格—红利比，列表或作图。
2. 假设我们希望红利索取权的对数溢价达到 3%（这只是一个假设的目标，不是对任何市场的估计），用求根算法求出所需的 $\gamma$，并报告此时的无风险利率与价格—红利比。你对这样的参数有什么看法？
:::

:::{solution-start} ex-ch1-grid
:class: dropdown
:::

```{code-cell} python
grid = pd.DataFrame(
    [{"γ": g, "log R^f": math.log(q.risk_free), "对数溢价": q.log_equity_premium, "P/D": q.price_dividend_ratio}
     for g in range(1, 31) for q in [ConsumptionParams(gamma=float(g))]]
).set_index("γ")
print(grid.loc[[1, 2, 5, 10, 15, 20, 25, 30]].round(4).to_string())
assert np.all(np.diff(grid["对数溢价"]) > 0) and np.all(np.diff(grid["log R^f"]) > 0)

target = 0.03
g_star = brentq(lambda g: ConsumptionParams(gamma=g).log_equity_premium - target, 0.1, 100)
q_star = ConsumptionParams(gamma=g_star)
print(f"对数溢价达到 {target:.0%} 需要 γ = {g_star:.2f}；此时 log R^f = {math.log(q_star.risk_free):.4f}，"
      f"P/D = {q_star.price_dividend_ratio:.2f}")
assert math.isclose(g_star, target / (p.rho_cd * p.sigma_c * p.sigma_d), rel_tol=1e-8) and math.isclose(g_star, 25.0)
assert 0.38 < math.log(q_star.risk_free) < 0.39 and 2.1 < q_star.price_dividend_ratio < 2.2
```

对数溢价与 $\gamma$ 成正比，$\gamma\rho\sigma_c\sigma_d = 0.03$ 给出 $\gamma = 0.03/(0.5 \times 0.02 \times 0.12) = 25$，求根结果与此一致。但此时对数无风险利率约为 38%，价格—红利比只有 2 倍多，都远远不合常理：$\gamma = 25$ 意味着跨期替代弹性只有 0.04，预期消费增长 2% 就要求极高的利率；高利率又把所有未来红利的现值压得很低。在幂效用、低消费波动的模型中，单靠提高 $\gamma$ 无法同时得到较高的风险溢价与合理的无风险利率。这正是第 7 章要讨论的问题。
:::{solution-end}
:::

:::{exercise} 短样本中的风险溢价
:label: ex-ch1-short
取默认参数，用种子 10000～10499 各模拟一条 50 年的路径。

1. 对每条路径计算样本平均超额收益 $\bar R^e$，报告它在 500 条路径上的均值、标准差，以及 $\bar R^e < 0$ 的比例。与闭式解 $\E[R] - R^f$ 比较。
2. 由对数正态的性质，$\Var(R) = \E[R]^2(e^{\sigma_d^2} - 1)$。据此算出 50 年样本均值的理论标准误，并与第 1 问的标准差比较。要使理论标准误小于闭式解的一半，需要多少年？
3. 结合“A 股视角”一节，说明这对用消费数据检验定价模型意味着什么。
:::

:::{solution-start} ex-ch1-short
:class: dropdown
:::

```{code-cell} python
n_rep, n_years = 500, 50
truth = p.expected_return - p.risk_free
Rebar = np.array([(lambda d: (d["R"] - d["Rf"]).mean())(simulate_consumption(p, seed=10_000 + s, n_years=n_years).data)
                  for s in range(n_rep)])                             # (500,)：每条路径的样本平均超额收益
sd_R = p.expected_return * math.sqrt(math.expm1(p.sigma_d**2))      # Std(R) = E[R] √(e^{σ_d²} − 1)
se_theory = sd_R / math.sqrt(n_years)
years_needed = math.ceil((sd_R / (truth / 2)) ** 2)                 # se < truth/2 ⟺ T > (2 Std(R) / truth)²
print(f"闭式解 {truth:.4f}；500 条路径：均值 {Rebar.mean():.4f}，标准差 {Rebar.std(ddof=1):.4f}，"
      f"为负的比例 {np.mean(Rebar < 0):.1%}")
print(f"理论标准误 {se_theory:.4f}（闭式解的 {se_theory / truth:.1f} 倍）；"
      f"理论上为负的概率约 Φ(−E[R^e]/se) = {norm.cdf(-truth / se_theory):.1%}；需要约 {years_needed:,} 年")
assert abs(Rebar.mean() - truth) < 3 * Rebar.std(ddof=1) / math.sqrt(n_rep)
assert abs(Rebar.std(ddof=1) / se_theory - 1) < 0.1 and se_theory / truth > 6 and Rebar.std(ddof=1) / truth > 6
assert 0.35 < np.mean(Rebar < 0) < 0.5 and 10_000 < years_needed < 11_000
```

1. 500 条路径上样本平均超额收益的均值与闭式解相差不到 3 倍蒙特卡洛标准误，说明样本均值是无偏的；但它的标准差是闭式解的 6 倍以上，有将近一半的路径得到负的样本溢价（数值见上方输出，均由 `assert` 保证）。
2. 理论标准误 $\text{Std}(R)/\sqrt{50}$ 与第 1 问的标准差相差不到 10%。要使标准误小于闭式解的一半，需要 $T > (2\,\text{Std}(R)/(\E[R] - R^f))^2$，约一万年。
3. 默认参数下风险溢价很小、收益波动很大，任何现实长度的样本都无法可靠地估计它，更不用说估计它与消费 β 的关系：年度消费数据同样只有几十个观测，还有口径、修订、时间加总等问题。要检验定价模型，需要利用横截面上大量资产的信息（第二、三部分）或多个矩的共同约束（第 11 章的 GMM），并始终把估计值与标准误放在一起看。
:::{solution-end}
:::

(sec-ch1-refs)=
## 文献与延伸阅读

- 消费模型、基本定价方程与 SDF 的系统处理见 @cochrane2005asset 的前两章；本章的推导顺序与它相近，但用本书的记号重新写出，并逐步配了代码验证。
- 交换经济中的资产定价与 Lucas 树的原始文献是 @lucas1978asset。离散时间下幂效用的定价公式见 @rubinstein1976valuation；连续时间中精确成立的消费 CAPM 见 @breeden1979intertemporal。
- 在对数正态假设下检验消费模型的经典文献是 @hansen1983stochastic；用广义矩方法估计偏好参数的方法在第 11 章介绍。消费数据的时间加总问题与消费 CAPM 的检验见 @breeden1989empirical。
- 幂效用模型与数据的定量矛盾，即股权溢价之谜，原始文献为 @mehra1985equity，第 7 章讨论；把风险厌恶与跨期替代弹性分开的递归效用见 @epstein1989substitution。
- 估值比率随时间变化、预测收益的证据与“折现率”的含义见 @cochrane2011presidential，第 6 章展开；SDF 的波动与 Hansen–Jagannathan 界见 @hansen1991implications，第 3 章展开。
