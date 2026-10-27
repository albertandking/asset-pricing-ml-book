---
short_title: A 股视角、模拟实验与常见误区
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

# A 股视角、模拟实验与常见误区

**本页准备。** 本书每个页面的代码在独立的内核中运行，前面页面定义的变量在这里不可用。下面的单元导入本页用到的模块，设置作图样式，并生成本页共用的长样本：默认假设参数、种子 2026、20 万年。

```{code-cell} python
import math

import numpy as np
import pandas as pd

from apsim import ConsumptionParams, plotting, simulate_consumption
from apsim.stats import ols

cjk_font = plotting.setup()
p = ConsumptionParams()                                       # 默认的年度假设参数
T_long = 200_000
sim = simulate_consumption(p, seed=2026, n_years=T_long).data  # 列 dc、dd、m、R、Rf，索引 year 从 1 到 T
```

(sec-ch1-ashare)=
## A 股视角

把本章的模型搬到中国的数据上，首先遇到的是数据问题。下面只讨论定性的特点，不涉及任何估计结果。

**口径。** 理论中的 $C_t$ 是边际投资者的人均实际消费，通常取非耐用品与服务。国家统计局发布的与消费有关的序列有好几种，口径各不相同：社会消费品零售总额按月发布，统计的是商品零售与餐饮收入，其中包括对单位的销售，大部分服务消费不在其中，所以它不等于居民消费；住户调查中的居民人均消费支出按季度发布，口径更接近理论，但历史较短；支出法国内生产总值中的最终消费支出包括居民消费与政府消费，以年度为主。研究者要先决定用哪一个口径，再做价格平减与人口调整，每一步都会影响消费增长的波动与它和收益的协方差。

**频率与样本长度。** 消费模型的检验依赖消费增长与资产收益的协方差，而可用的消费数据是季度或年度的。即使用年度数据，改革开放以来也只有几十个观测；城乡一体化的住户调查从 2013 年前后才开始（待核实），此前城镇与农村分别调查，口径不一致。“模拟实验”一节与[习题 %s](#ex-ch1-short) 会看到，在本章的默认参数下，50 年的样本根本不足以把风险溢价与 0 区分开。

**修订与时点。** 宏观数据会修订，例如经济普查之后会系统调整历史数据。回测中如果用今天看到的修订后数据，就等于使用了当时还不知道的信息。这与第 0 章强调的时点错位是同一类问题，正式研究应当使用当时实际发布的数据。

**时间加总。** 统计数据中的消费是一整个季度或一整年的流量，而收益是从一个时点到另一个时点的变化。前者相当于对期内消费求平均，会使测得的消费增长更平滑、带有序列相关，从而改变它与收益的协方差。Breeden、Gibbons 与 Litzenberger 讨论过这一问题及其修正 [@breeden1989empirical]。

**参与。** 并非所有家庭都持有股票，不持有股票的家庭的消费，与股票定价中的边际效用没有直接关系；总消费把二者混在一起，削弱了它与收益的联系。

这些困难并不否定消费模型的理论价值：它说明了 SDF 的经济含义，以及无风险利率与风险溢价由哪些因素决定。但在 A 股的实证研究中，直接用消费数据估计和检验定价模型困难很大，更常见的做法是像第 0 章那样用可交易的因子构造 SDF（第三部分），把消费模型当作理解因子“为什么得到补偿”的理论背景。消费模型与数据的定量比较（股权溢价之谜）在第 7 章讨论，估计偏好参数的方法在第 11 章讨论。

(sec-ch1-sim)=
## 模拟实验

本节用 `simulate_consumption` 生成的长样本检验本章的三项结论：欧拉方程、风险溢价的闭式解、消费 CAPM 的 β 表示。模拟器的模型就是式 {eq}`eq-ch1-model`，真值全部由闭式解给出，所以每一项检验都有明确的“正确答案”。

### 实验一：欧拉方程随样本收敛

欧拉方程 $\E[m_{t+1}R_{t+1}] = 1$ 与 $\E[m_{t+1}]R^f = 1$ 是总体矩，样本均值只是它们的估计。下图画出前 $T$ 年的样本均值 $T^{-1}\sum_t m_t R_t$ 与 $T^{-1}\sum_t m_t R^f$ 随 $T$ 的变化，灰色区域是理论值 1 加减两倍标准误，标准误按“逐年标准差 $/\sqrt{T}$”计算（模型中各年独立同分布）。

```{code-cell} python
#| label: fig-ch1-euler
#| caption: 欧拉方程的样本矩随样本年数 T 收敛到 1（对数横轴；默认假设参数，种子 2026）
import matplotlib.pyplot as plt

mR = (sim["m"] * sim["R"]).to_numpy()                       # (T,)：m_t R_t
mRf = (sim["m"] * sim["Rf"]).to_numpy()                     # (T,)：m_t R^f
T_grid = np.unique(np.geomspace(10, T_long, 60).astype(int))

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharex=True)
summary = []
for ax, y, title in [(axes[0], mR, "m·R 的样本均值"), (axes[1], mRf, "m·Rf 的样本均值")]:
    run = np.cumsum(y)[T_grid - 1] / T_grid                 # 前 T 年的累计均值
    se = y.std() / np.sqrt(T_grid)
    ax.fill_between(T_grid, 1 - 2 * se, 1 + 2 * se, color="0.85", label="理论值 ± 2 倍标准误")
    ax.axhline(1.0, color="0.3", lw=1, ls="--", label="理论值 1")
    ax.plot(T_grid, run, color="C0", label="样本均值")
    ax.set_xscale("log")
    ax.set_xlabel("样本年数 T")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    summary.append({"矩": title, "全样本偏差": run[-1] - 1, "全样本标准化偏差": (run[-1] - 1) / se[-1],
                    "T ≤ 100 时的最大偏差": np.abs(run - 1)[T_grid <= 100].max(),
                    "T ≥ 20000 时的最大偏差": np.abs(run - 1)[T_grid >= 20_000].max()})
fig.tight_layout()
plt.show()

euler = pd.DataFrame(summary).set_index("矩")
print(euler.to_string(float_format=lambda v: f"{v:.2e}"))
assert np.all(np.abs(euler["全样本标准化偏差"]) < 3)
assert np.all(euler["T ≥ 20000 时的最大偏差"] < euler["T ≤ 100 时的最大偏差"] / 10)
```

两个样本矩都收敛到 1：全样本的标准化偏差都在 3 以内，$T \geq 20000$ 时的最大偏差不到 $T \leq 100$ 时的十分之一（均由 `assert` 保证，数值见图下输出）。$m_t R_t$ 的样本均值收敛得慢，因为红利索取权的收益波动远大于 $m$ 本身的波动；$m_t R^f$ 中 $R^f$ 是常数，只有 $m$ 的波动。

### 实验二：风险溢价的闭式解与样本均值

由 1.4 节，红利索取权的期望超额收益精确等于 $\E[R] - R^f = R^f(e^{\gamma\rho\sigma_c\sigma_d} - 1)$，它由 `expected_return` 与 `risk_free` 两个属性给出。用同一个长样本的样本均值检验它：

```{code-cell} python
Re = (sim["R"] - sim["Rf"]).to_numpy()                       # (T,)：超额收益 R_t − R^f
truth = p.expected_return - p.risk_free
se_Re = Re.std(ddof=1) / math.sqrt(T_long)
z_prem = (Re.mean() - truth) / se_Re
print(f"闭式解 {truth:.5f}，样本均值 {Re.mean():.5f}，标准误 {se_Re:.5f}，标准化偏差 {z_prem:.2f}")
print(f"闭式解除以标准误 = {truth / se_Re:.1f}")
print(f"log(样本均值 R / R^f) = {math.log(sim['R'].mean() / p.risk_free):.5f}，γρσ_cσ_d = {p.log_equity_premium:.5f}")
assert abs(z_prem) < 4 and truth / se_Re > 5
```

阈值取 4 倍标准误的理由是：若闭式解正确，由中心极限定理，标准化偏差近似服从标准正态分布，绝对值超过 4 的概率约为万分之零点六，所以断言失败几乎只可能说明闭式解或模拟器有误。检验还要有区分力：样本必须足以把真值与 0 区分开，所以代码同时断言闭式解超过 5 倍标准误。20 万年的样本满足这一点；50 年的样本，标准误约为闭式解的 6 倍以上（[习题 %s](#ex-ch1-short)）。

### 实验三：用消费增长做时间序列回归

消费 CAPM {eq}`eq-ch1-ccapm` 说，期望超额收益近似等于 $\beta_{\Delta c}\lambda_c$，$\lambda_c = \gamma\sigma_c^2$。为了得到一个横截面，构造五项红利索取权，它们的红利增长与消费增长的相关系数 $\rho$、红利波动 $\sigma_d$ 不同。由于 `simulate_consumption` 按名称为消费冲击与红利冲击各派生一个随机子流，用同一个种子模拟时，五项资产的消费增长路径完全相同（代码中由 `assert` 核对），它们就像同一个经济中的五项资产。

对每项资产做时间序列回归 $R^e_t = c_0 + \beta_{\Delta c}\Delta c_t + u_t$，用 `apsim.stats.ols`（第 0 章“模拟实验”一节中内联实现并讲解过的最小二乘函数）求出 $\hat\beta_{\Delta c}$ 与它的标准误。与真值 $\beta_{\Delta c} = \E[R]\rho\sigma_c\sigma_d/\sigma_c^2$（1.4 节）、$\beta_{\Delta c}\lambda_c$ 与精确的期望超额收益比较。

```{code-cell} python
lam_c = p.gamma * p.sigma_c**2                                # λ_c = γ σ_c²
specs = [(-0.5, 0.12), (0.0, 0.12), (0.5, 0.12), (1.0, 0.12), (0.5, 0.24)]   # (ρ, σ_d)
rows, Re_cols, beta_hat = [], [], []
for rho, sd in specs:
    q = ConsumptionParams(rho_cd=rho, sigma_d=sd)
    dq = simulate_consumption(q, seed=2026, n_years=T_long).data
    assert np.array_equal(dq["dc"].to_numpy(), sim["dc"].to_numpy())          # 同一种子：消费路径相同
    y = (dq["R"] - dq["Rf"]).to_numpy()                                      # (T,)
    X = np.column_stack([np.ones(T_long), dq["dc"].to_numpy()])              # (T, 2)：常数项与 Δc_t
    coef, se = ols(X, y)                                                     # coef: (2,) = (c_0, β̂)
    beta_true = q.expected_return * rho * q.sigma_c * q.sigma_d / q.sigma_c**2
    exact = q.expected_return - q.risk_free
    rows.append({"ρ": rho, "σ_d": sd, "β̂": coef[1], "se(β̂)": se[1], "β 真值": beta_true,
                 "β 真值·λ_c": beta_true * lam_c, "精确 E[R^e]": exact, "样本均值": y.mean()})
    Re_cols.append(y)
    beta_hat.append(coef[1])
ccapm = pd.DataFrame(rows)
print(ccapm.round(5).to_string(index=False))
assert np.all(np.abs(ccapm["β̂"] - ccapm["β 真值"]) < 4 * ccapm["se(β̂)"])
assert np.allclose(ccapm["β 真值·λ_c"], ccapm["精确 E[R^e]"], rtol=0.005, atol=1e-12)    # 线性化误差小于 0.5%

# 横截面：样本均值对 β̂ 做不带截距的回归，估计 λ_c；标准误用逐年的横截面估计计算
Re_mat = np.column_stack(Re_cols)                                 # (T, 5)
b_vec = np.asarray(beta_hat)                                      # (5,)
lam_t = Re_mat @ b_vec / (b_vec @ b_vec)                          # (T, 5) @ (5,) -> (T,)：每年的 λ̂_t
lam_hat, lam_se = lam_t.mean(), lam_t.std(ddof=1) / math.sqrt(T_long)
print(f"λ̂ = {lam_hat:.5f}（标准误 {lam_se:.5f}）；λ_c = γσ_c² = {lam_c:.5f}；E[Δc] = {p.mu_c:.5f}")
assert abs(lam_hat - lam_c) < 3 * lam_se and abs(lam_hat - p.mu_c) > 20 * lam_se
```

五项资产的 $\hat\beta_{\Delta c}$ 都在真值的 4 倍标准误以内；$\beta_{\Delta c}\lambda_c$ 与精确的期望超额收益相对误差小于 0.5%，这就是 1.4 节算出的线性化误差。$\rho < 0$ 的资产在消费增长低时收益高，$\beta_{\Delta c} < 0$，期望超额收益为负，是一种“保险”；$\rho = 0$ 的资产与消费无关，波动虽大，期望超额收益却为 0。最后两项资产的 $\sigma_d$ 与 $\rho$ 不同，但 $\rho\sigma_d$ 相同，因而 β 与期望超额收益的真值完全相同（估计值因抽样误差略有差别）；后者的总波动大一倍，却没有得到更多补偿：决定补偿的是与消费的协方差，不是总波动。

代码最后用五项资产的样本均值对 $\hat\beta_{\Delta c}$ 做横截面回归（不带截距）估计 $\lambda_c$：估计值与 $\lambda_c = 0.0008$ 相差不到 3 倍标准误，而与 $\E[\Delta c] = 0.02$ 相差超过 20 倍标准误（均由 `assert` 保证）。数据本身告诉我们，不可交易因子的风险溢价不是它的均值。这里把 $\hat\beta$ 当作已知来计算标准误，忽略了 $\hat\beta$ 的估计误差；考虑这一误差的 Shanken 修正见第 10 章。

(sec-ch1-pitfalls)=
## 常见误区

**误区一：不加说明地把 $\gamma$ 同时当作风险厌恶系数与跨期替代弹性的倒数。** 在幂效用下，相对风险厌恶系数是 $\gamma$，跨期替代弹性是 $1/\gamma$，这是函数形式强加的，不是经验事实。风险厌恶针对同一时期不同状态之间的消费波动，跨期替代针对不同时期之间的消费不均，二者在概念上不同。用幂效用解释较高的风险溢价需要较大的 $\gamma$，同时就意味着很低的跨期替代弹性和很高的无风险利率：由式 {eq}`eq-ch1-logrf`，其余参数不变而 $\gamma = 10$ 时，对数无风险利率约为 19%（下方代码）。报告“$\gamma$ 的估计值”时，要说明效用函数的形式与识别它的矩。Epstein 与 Zin 的递归效用把二者分开 [@epstein1989substitution]，第 7 章介绍。

```{code-cell} python
log_rf_10 = math.log(ConsumptionParams(gamma=10).risk_free)
print(f"γ = 10（跨期替代弹性 0.1）时 log R^f = {log_rf_10:.4f}")
assert 0.18 < log_rf_10 < 0.20
```

**误区二：把消费增长的风险溢价 $\lambda_c$ 当作消费增长的均值。** $\lambda_c = \gamma\sigma_c^2$ 是每单位消费 β 的补偿，默认参数下为 0.08%；$\E[\Delta c] = 2\%$ 是经济增长的速度，二者毫无关系。“因子风险溢价等于因子均值”只对本身是超额收益、并为自身正确定价的因子成立（第 0 章 0.2 节，本章 1.4 节与 1.6 节）。误用 $\E[\Delta c]$，在默认参数下会把期望超额收益高估 25 倍（1.4 节）。对消费增长、宏观变量一类的因子，风险溢价要用横截面回归或模拟组合另行估计（上一节的实验三、第 4 章与第 10 章）。

**误区三：忽视 $a < 1$ 的条件。** 价格—红利比 $a/(1-a)$ 只在 $a < 1$ 时有意义。参数选得不当（例如 $\gamma < 1$ 且增长较快），$a \geq 1$，红利的贴现和发散，不存在有限的价格。这类错误在数值计算中很隐蔽：如果用有限期的和近似无穷级数，截断在 100 年时会得到一个看起来正常的数，截断在 1000 年时却是它的两百多倍。下面以 1.5 节中 $\gamma = 0.5$、$\mu_c = 3\%$ 的 Lucas 树为例，模拟器在这种参数下直接抛出 `ValueError`，而不是返回一个错误的数。

```{code-cell} python
bad = ConsumptionParams(gamma=0.5, mu_c=0.03, mu_d=0.03, sigma_d=0.02, rho_cd=1.0)   # Lucas 树，a > 1
a_bad = bad.dividend_discount
partial = {J: sum(a_bad**j for j in range(1, J + 1)) for J in (100, 1000)}
print(f"a = {a_bad:.4f}；截断到 100 年的和 {partial[100]:.1f}，截断到 1000 年的和 {partial[1000]:.3e}")
try:
    simulate_consumption(bad, seed=0, n_years=10)
    raised = False
except ValueError as err:
    raised = True
    print("simulate_consumption 抛出 ValueError：", err)
assert a_bad > 1 and 100 < partial[100] < 200 and partial[1000] > 200 * partial[100] and raised
```

**误区四：混淆对数溢价与算术溢价。** 由式 {eq}`eq-ch1-logprem`，$\log(\E[R]/R^f) = \gamma\rho\sigma_c\sigma_d$；而由 $\log\E[R] = \E[\log R] + \sigma_d^2/2$，**对数收益的期望**比无风险利率的对数高出 $\gamma\rho\sigma_c\sigma_d - \sigma_d^2/2$。默认参数下前者为 0.24%，后者为 $0.24\% - 0.72\% = -0.48\%$：红利索取权的算术期望收益高于无风险利率，对数收益的期望却低于它。二者并不矛盾。长期持有时，财富的对数是各年对数收益之和，所以“典型”路径（中位数）上，红利索取权的累积财富增长得比无风险资产慢；但少数路径上它增长得极快，把期望值拉高。比较溢价时一定要说明是哪一种：对数还是算术，期望收益之比的对数还是对数收益的期望。用长样本核对：

```{code-cell} python
log_ex = (np.log(sim["R"]) - math.log(p.risk_free)).to_numpy()       # (T,)：log R_t − log R^f
theory = p.log_equity_premium - p.sigma_d**2 / 2
se_log = log_ex.std(ddof=1) / math.sqrt(T_long)
print(f"E[log R] − log R^f：样本均值 {log_ex.mean():.5f}，理论值 γρσ_cσ_d − σ_d²/2 = {theory:.5f}；"
      f"算术溢价 E[R] − R^f = {p.expected_return - p.risk_free:.5f}")
assert math.isclose(theory, -0.0048) and abs(log_ex.mean() - theory) < 4 * se_log and log_ex.mean() < 0
```
