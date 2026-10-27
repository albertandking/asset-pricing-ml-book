---
short_title: 1.5～1.6 Lucas 树与回到第 0 章
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

(sec-ch1-lucas)=
# 1.5～1.6 Lucas 树与回到第 0 章

**本页准备。** 本书每个页面的代码在独立的内核中运行，前面页面定义的变量在这里不可用。下面的单元导入本页用到的模块，设置统一的作图样式与中文字体（本页的路径图用到），并取 `ConsumptionParams` 的默认参数。

```{code-cell} python
import math

import numpy as np
import pandas as pd
from scipy.stats import norm

from apsim import ConsumptionParams, MarketParams, plotting, simulate_consumption

cjk_font = plotting.setup()                                # 返回所选中文字体名
p = ConsumptionParams()                                    # 默认的年度假设参数
nodes, weights = np.polynomial.hermite_e.hermegauss(80)
weights = weights / weights.sum()                          # 标准正态分布的 Gauss–Hermite 节点与权重（与 1.4 节相同）
```

## 1.5 多期定价与 Lucas 树

### 价格等于未来红利的贴现和

**白话。** 股票明年的支付是“明年的价格加明年的红利”，而明年的价格又由后年的价格与红利决定，后年的价格又由大后年的决定……一路代下去，今天的价格就等于未来每一年的红利按 SDF 折算到今天之后的总和。最后剩下的“无穷远处的价格折算到今天”要求趋于 0，意思是价格中没有一部分只靠“将来有人以更高的价格接手”支撑，也就是没有泡沫。这个要求称为**横截性条件**（transversality condition）。

**例子。** 先看确定性的情形（假设数字）：每年的 SDF 都等于 0.95，红利今年为 1 元、此后每年增长 2%。明年的红利今天值 $0.95 \times 1.02 = 0.969$ 元，后年的红利值 $0.969^2$ 元，第 $j$ 年的红利值 $0.969^j$ 元，总和是等比级数 $0.969/(1 - 0.969) \approx 31.26$ 元。如果红利每年增长 6%，公比变成 $0.95 \times 1.06 = 1.007 > 1$，越远的红利今天反而越值钱，总和发散，价格无穷大。公比小于 1，正是下面 Lucas 树中的条件 $a < 1$。

**公式。** 从 $P_t = \E_t[m_{t+1}(P_{t+1} + D_{t+1})]$ 出发，把同一个方程用于第 $t+1$ 期，$P_{t+1} = \E_{t+1}[m_{t+2}(P_{t+2} + D_{t+2})]$，代入后用迭代期望定律（$m_{t+1}$ 在第 $t+1$ 期末已知，可以移进 $\E_{t+1}$）：

```{math}
:enumerated: false
P_t = \E_t[m_{t+1}D_{t+1}] + \E_t\left[m_{t+1}\,\E_{t+1}[m_{t+2}(P_{t+2} + D_{t+2})]\right] = \E_t[m_{t+1}D_{t+1}] + \E_t[m_{t+1}m_{t+2}D_{t+2}] + \E_t[m_{t+1}m_{t+2}P_{t+2}] .
```

定义从第 $t$ 期到第 $t+j$ 期的**多期 SDF**

$$
M_{t,t+j} = m_{t+1}m_{t+2}\cdots m_{t+j} = \delta^j\left(\frac{C_{t+j}}{C_t}\right)^{-\gamma}, \qquad M_{t,t} = 1 ,
$$ (eq-ch1-multi)

第二个等号是幂效用下各期之比相乘、中间项相消的结果。$M_{t,t+j}$ 为第 $t+j$ 期的支付定价：第 $t+j$ 期支付 $x_{t+j}$ 的资产今天的价格是 $\E_t[M_{t,t+j}x_{t+j}]$。重复上面的代入 $J$ 次（用归纳法），

```{math}
:enumerated: false
P_t = \sum_{j=1}^{J}\E_t\left[M_{t,t+j}D_{t+j}\right] + \E_t\left[M_{t,t+J}P_{t+J}\right] .
```

若横截性条件

$$
\lim_{J\to\infty}\E_t\left[M_{t,t+J}P_{t+J}\right] = 0
$$ (eq-ch1-tvc)

成立，令 $J \to \infty$，得到

$$
P_t = \E_t\sum_{j=1}^{\infty} M_{t,t+j}D_{t+j} .
$$ (eq-ch1-pv)

横截性条件排除的是**理性泡沫**。设 $P^*_t$ 满足式 {eq}`eq-ch1-pv`，再任取一个满足 $B_t = \E_t[m_{t+1}B_{t+1}]$ 的正过程 $B_t$，则 $P_t = P^*_t + B_t$ 同样满足一期的定价方程 $P_t = \E_t[m_{t+1}(P_{t+1} + D_{t+1})]$；但 $\E_t[M_{t,t+J}B_{t+J}] = B_t$ 对所有 $J$ 成立，不趋于 0，违反式 {eq}`eq-ch1-tvc`。所以一期的欧拉方程本身不足以确定价格，还需要横截性条件；本书总是假设它成立，即价格等于红利的贴现和。

```{code-cell} python
g_sdf = 0.95 * 1.02                                         # 确定性例子：每年的 SDF 0.95 乘以红利增长 1.02
pv_finite = sum(g_sdf**j for j in range(1, 2001))           # 前 2000 年红利的贴现和
print(f"公比 {g_sdf:.3f}；等比级数求和 {g_sdf / (1 - g_sdf):.2f}，前 2000 年之和 {pv_finite:.2f}")
assert math.isclose(g_sdf, 0.969) and math.isclose(pv_finite, g_sdf / (1 - g_sdf), rel_tol=1e-12)
assert 31.25 < g_sdf / (1 - g_sdf) < 31.27 and 0.95 * 1.06 > 1           # 增长 6% 时公比大于 1，级数发散
```

(ch1-unpack-multi)=
**拆开讲式 {eq}`eq-ch1-multi`。** $M_{t,t+j}$ 把第 $t+j$ 年的 1 元折算成今天的元，单位是“今天的元/第 $t+j$ 年的元”，数值上没有单位。它是 $j$ 个一年期 SDF 的连乘：第一个 $m_{t+1}$ 在第 $t+1$ 期末揭晓，最后一个 $m_{t+j}$ 在第 $t+j$ 期末揭晓，所以在第 $t$ 期末，$M_{t,t+j}$ 是随机的。$C_{t+j}/C_t$ 是 $j$ 年的累计总消费增长，$\delta^j$ 是 $j$ 年累计的耐心折扣。以 $j = 2$ 为例展开：

```{math}
:enumerated: false
M_{t,t+2} = m_{t+1}m_{t+2} = \delta\left(\frac{C_{t+1}}{C_t}\right)^{-\gamma}\cdot\delta\left(\frac{C_{t+2}}{C_{t+1}}\right)^{-\gamma} = \delta^2\left(\frac{C_{t+2}}{C_t}\right)^{-\gamma} ,
```

中间的 $C_{t+1}$ 一个在分子、一个在分母，相互抵消。例如上面确定性的例子中每年的 SDF 都是 0.95，则 $M_{t,t+3} = 0.95^3 \approx 0.857$：3 年后的 1 元今天值约 0.857 元。**特例**：$j = 1$ 时 $M_{t,t+1} = m_{t+1}$；$j = 0$ 时约定 $M_{t,t} = 1$，今天的 1 元就值 1 元。

(ch1-unpack-tvc)=
**拆开讲式 {eq}`eq-ch1-tvc` 与 {eq}`eq-ch1-pv`。** $P_{t+J}$ 是第 $t+J$ 期末（当年红利派发之后）的价格，以元计，在第 $t$ 期末是随机的；$M_{t,t+J}P_{t+J}$ 是“到第 $t+J$ 期末把资产卖掉”这笔钱折算到今天的值，$\E_t$ 对它取期望。横截性条件要求这笔远期卖出价的现值随 $J$ 增大而趋于 0。式 {eq}`eq-ch1-pv` 中的 $D_{t+j}$ 是第 $t+j$ 年派发的红利（元），在第 $t$ 期末随机；求和从 $j = 1$ 开始，因为 $P_t$ 是今年的红利 $D_t$ 派发之后的价格，买入者拿不到 $D_t$。逐项写出是

```{math}
:enumerated: false
P_t = \E_t[m_{t+1}D_{t+1}] + \E_t[m_{t+1}m_{t+2}D_{t+2}] + \E_t[m_{t+1}m_{t+2}m_{t+3}D_{t+3}] + \cdots ,
```

第 $j$ 项是只获得第 $t+j$ 年红利的资产今天的价格。在上面的确定性例子中（$D_t = 1$），前三项依次为 $0.969$、$0.969^2 \approx 0.939$、$0.969^3 \approx 0.910$ 元……合计约 31.26 元；$J$ 年后的价格为 $31.26 \times 1.02^J$ 元，折算到今天是 $0.95^J \times 31.26 \times 1.02^J = 31.26 \times 0.969^J$，$J = 100$ 时约为 1.34 元，$J \to \infty$ 时趋于 0，横截性条件成立。**特例**：每年的 SDF 都等于常数 $1/R^f$、红利每年确定地按总增长率 $G$ 增长时，式 {eq}`eq-ch1-pv` 化为等比级数 $P_t = D_t\,G/(R^f - G)$（要求 $G < R^f$），这是 Gordon 增长模型的确定性版本；上例 $1/R^f = 0.95$、$G = 1.02$，$1.02/(1/0.95 - 1.02) \approx 31.26$。

```{code-cell} python
# 拆开讲 eq-ch1-multi、eq-ch1-tvc 与 eq-ch1-pv：确定性例子中的多期 SDF、各年红利的价格与远期卖出价的现值
pv_det = g_sdf / (1 - g_sdf)                                 # P_t（D_t = 1）
strips_det = np.array([g_sdf**j for j in (1, 2, 3)])         # 第 1、2、3 年红利今天的价格 (3,)
tvc_100 = 0.95**100 * pv_det * 1.02**100                     # E_t[M_{t,t+100} P_{t+100}]
print("前三项", strips_det.round(3), f"；J = 100 时远期卖出价的现值 {tvc_100:.2f}")
assert math.isclose(0.95**3, 0.857375) and np.allclose(strips_det, [0.969, 0.939, 0.910], atol=5e-4)
assert math.isclose(tvc_100, pv_det * g_sdf**100) and 1.33 < tvc_100 < 1.35
assert math.isclose(1.02 / (1 / 0.95 - 1.02), pv_det, rel_tol=1e-12)    # Gordon：D_t G / (R^f − G)
```

### Lucas 树与价格—红利比

**价格—红利比。** 本节的核心变量是**价格—红利比** $P_t/D_t$（price-dividend ratio，简记 $P/D$）：价格是当年红利的多少倍。例如股价 40 元、每股年度红利 1 元，价格—红利比为 40。它不是市盈率。市盈率是价格与盈利之比 $P/E$（E 指每股收益），而公司通常只把盈利的一部分作为红利分配，其余留存。设上例中每股收益为 2 元、分红比例 50%，市盈率为 20，价格—红利比为 40。A 股更常用的是**股息率** $D/P$，即价格—红利比的倒数，上例中为 2.5%。

**Lucas 树。** Lucas 的交换经济 [@lucas1978asset] 是这一节的原型：经济中有一棵（或许多棵）果树，每年结出的果实是唯一的消费品，不能储存。代表性投资者持有果树，均衡时他吃掉全部果实，所以消费等于红利，$C_t = D_t$。价格调整到让投资者恰好既不多买也不少买果树，于是欧拉方程中的 $m$ 由等于红利的消费决定。在模型 {eq}`eq-ch1-model` 中取 $\rho_{cd} = 1$、$\sigma_d = \sigma_c$、$\mu_d = \mu_c$ 就得到 Lucas 树；一般情形下红利与消费不同，但只要二者的增长都独立同分布，推导完全一样。

:::{prf:proposition} 独立同分布时的价格—红利比
:label: prop-ch1-pd
在幂效用 {eq}`eq-ch1-power` 与模型 {eq}`eq-ch1-model` 下，令

$$
a = \E\left[m_{t+1}e^{\Delta d_{t+1}}\right] = \delta\exp\left(\mu_d - \gamma\mu_c + \tfrac12\left(\sigma_d^2 + \gamma^2\sigma_c^2 - 2\gamma\rho_{cd}\sigma_c\sigma_d\right)\right) .
$$ (eq-ch1-a)

若 $a < 1$，满足横截性条件的红利索取权价格为

$$
\frac{P_t}{D_t} = \frac{a}{1 - a} ,
$$ (eq-ch1-pd)

价格—红利比是常数。若 $a \geq 1$，式 {eq}`eq-ch1-pv` 中的级数发散，不存在有限的价格。
:::

:::{prf:proof}
**第一步：$j$ 期红利的价格。** 由 $D_{t+j} = D_t\, e^{\Delta d_{t+1} + \cdots + \Delta d_{t+j}}$ 与式 {eq}`eq-ch1-multi`，

```{math}
:enumerated: false
\frac{\E_t\left[M_{t,t+j}D_{t+j}\right]}{D_t} = \E_t\left[\prod_{k=1}^{j} m_{t+k}e^{\Delta d_{t+k}}\right] = \prod_{k=1}^{j}\E\left[m_{t+k}e^{\Delta d_{t+k}}\right] = a^j .
```

式中 $\prod_{k=1}^{j}$ 表示把 $k = 1, \dots, j$ 的 $j$ 个因子连乘，例如 $j = 2$ 时为 $m_{t+1}e^{\Delta d_{t+1}} \cdot m_{t+2}e^{\Delta d_{t+2}}$；$e^{\Delta d_{t+k}} = D_{t+k}/D_{t+k-1}$ 是第 $t+k$ 年的总红利增长，所以连乘中的 $e^{\Delta d}$ 合起来就是 $D_{t+j}/D_t$，各个 $m$ 合起来就是 $M_{t,t+j}$。

第二个等号用到：$(\Delta c_{t+k}, \Delta d_{t+k})$ 在不同的 $k$ 之间独立，且与第 $t$ 期末的信息独立，所以各因子 $m_{t+k}e^{\Delta d_{t+k}}$ 相互独立，乘积的期望等于期望的乘积；它们同分布，每个期望都是 $a$。

**第二步：求和。** 由式 {eq}`eq-ch1-pv`，$P_t/D_t = \sum_{j\ge1}a^j$。$a > 0$，当 $a < 1$ 时等比级数收敛于 $a/(1-a)$；当 $a \geq 1$ 时每一项都不小于 1，级数发散。

**第三步：横截性条件。** 当 $a < 1$ 时，令 $v = a/(1-a)$，$P_{t+J} = vD_{t+J}$，由第一步，$\E_t[M_{t,t+J}P_{t+J}] = v\,D_t\,a^J \to 0$，式 {eq}`eq-ch1-tvc` 成立。

**第四步：计算 $a$。** $\log(m_{t+1}e^{\Delta d_{t+1}}) = \log\delta - \gamma\Delta c_{t+1} + \Delta d_{t+1} = \log\delta - \gamma\mu_c + \mu_d + (\rho_{cd}\sigma_d - \gamma\sigma_c)\varepsilon_{t+1} + \sigma_d\sqrt{1-\rho_{cd}^2}\,\nu_{t+1}$，是正态的，方差为 $(\rho_{cd}\sigma_d - \gamma\sigma_c)^2 + \sigma_d^2(1 - \rho_{cd}^2) = \sigma_d^2 + \gamma^2\sigma_c^2 - 2\gamma\rho_{cd}\sigma_c\sigma_d$。由式 {eq}`eq-ch1-lognormal` 即得式 {eq}`eq-ch1-a`。
:::

另一种常见的推导是“猜测并验证”：猜 $P_t = vD_t$，代入 $P_t = \E_t[m_{t+1}(P_{t+1} + D_{t+1})]$，两边除以 $D_t$，得 $v = \E_t[m_{t+1}(v + 1)e^{\Delta d_{t+1}}] = a(v + 1)$，解出 $v = a/(1-a)$。这种方法更快，但只找到了一个解，排除泡沫解要靠横截性条件（上一小节）。$a < 1$ 也正是这个解为正的条件：$a > 1$ 时 $v = a/(1-a) < 0$，不是有意义的价格。

**$a$ 的含义。** 由第一步，$a$ 是“只获得明年红利”这项资产今天的价格，以当前红利为单位；同理，$a^j$ 是只获得第 $j$ 年红利增长因子 $D_{t+j}/D_t$ 的索取权今天的价格，即只获得第 $j$ 年红利的资产的价格（以当前红利为单位）。价格—红利比 $v = \sum_{j\ge1}a^j$ 就是这些价格之和：$a < 1$ 时越远的红利今天越不值钱，总价值才是有限的；$a \geq 1$ 时级数发散。所以 $a < 1$ 恰好是价格 $v$ 为正且有限的条件。$a$ 还可以写成 $a = G/\E[R]$，其中 $G = \E[e^{\Delta d}] = e^{\mu_d + \sigma_d^2/2}$ 是红利的期望总增长（下面的收益公式 {eq}`eq-ch1-return` 给出 $\E[R] = G/a$）。于是 $a < 1$ 等价于 $\E[R] > G$：期望收益（折现率）高于红利的期望增长；而式 {eq}`eq-ch1-pd` 化为 $P/D = G/(\E[R] - G)$，这就是熟悉的 Gordon 增长模型。默认参数下 $a \approx 0.9758$，价格—红利比约为 40.4，股息率约为 2.5%。

下面的代码用三种方法核对 $a$：闭式解（模拟器的属性 `dividend_discount`）、二维 Gauss–Hermite 求积、长样本模拟。模拟部分验证第一步的乘法规则 $\E[M_{t,t+j}D_{t+j}/D_t] = a^j$：把一条 40 万年的模拟路径切成互不重叠的、长度为 $j$ 的区段，每个区段给出一个 $\prod_k m_k e^{\Delta d_k}$ 的独立抽样。

```{code-cell} python
a = p.dividend_discount
pd_ratio = p.price_dividend_ratio
E1, N1 = np.meshgrid(nodes, nodes, indexing="ij")                      # ε、ν 的节点 (80, 80)
W2 = np.outer(weights, weights)
dc_q = p.mu_c + p.sigma_c * E1
dd_q = p.mu_d + p.sigma_d * (p.rho_cd * E1 + math.sqrt(1 - p.rho_cd**2) * N1)
a_quad = (W2 * p.delta * np.exp(-p.gamma * dc_q + dd_q)).sum()          # a = E[m e^{Δd}]
G = math.exp(p.mu_d + p.sigma_d**2 / 2)                                 # 红利的期望总增长 E[e^{Δd}]
print(f"a：闭式解 {a:.8f}，求积 {a_quad:.8f}；P/D = a/(1−a) = {pd_ratio:.3f}，股息率 D/P = {1 / pd_ratio:.2%}")
print(f"Gordon 形式 G/(E[R] − G) = {G / (p.expected_return - G):.3f}，a = G/E[R] = {G / p.expected_return:.8f}")
assert math.isclose(a, a_quad, rel_tol=1e-12) and 0.9758 < a < 0.9759 and 40.3 < pd_ratio < 40.5
assert math.isclose(pd_ratio, G / (p.expected_return - G), rel_tol=1e-12) and math.isclose(a, G / p.expected_return)
assert 0.0247 < 1 / pd_ratio < 0.0248

sim = simulate_consumption(p, seed=2026, n_years=400_000).data
mg = (sim["m"] * np.exp(sim["dd"])).to_numpy()                          # (T,)：m_t e^{Δd_t}
rows = []
for j in (1, 5, 20):
    blocks = mg[: len(mg) // j * j].reshape(-1, j).prod(axis=1)         # (T/j, j) -> (T/j,)：每个区段的乘积
    se = blocks.std(ddof=1) / math.sqrt(len(blocks))
    rows.append({"j": j, "区段数": len(blocks), "样本均值": blocks.mean(), "a^j": a**j,
                 "标准化偏差": (blocks.mean() - a**j) / se})
strips = pd.DataFrame(rows).set_index("j")
print(strips.round(5).to_string())
assert np.all(np.abs(strips["标准化偏差"]) < 4)
```

表中的“标准化偏差”是样本均值与 $a^j$ 之差除以标准误；`assert` 要求它们都在 4 以内。在 $a^j$ 是真值的前提下，标准化偏差近似服从标准正态分布，绝对值超过 4 的概率不到万分之一，所以断言失败几乎只可能说明推导或代码有误，而不是运气不好。

(ch1-unpack-a)=
**拆开讲式 {eq}`eq-ch1-a`。** $a$ 没有单位：它是“只获得明年红利”的资产今天的价格除以今年的红利 $D_t$（默认约 0.9758，即今年每 1 元红利，对应的明年那笔红利今天值约 0.976 元）。独立同分布时它不随 $t$ 变化，所以期望不带下标。$m_{t+1}e^{\Delta d_{t+1}}$ 是“明年的 SDF 乘以明年的总红利增长”，在第 $t$ 期末随机。把 $\log a$ 的六项按含义重新排列，逐项看（默认参数，都是年度的对数量）：

| 项 | 默认值 | 含义 |
|---|---|---|
| $\log\delta$ | 约 −1.005% | 耐心：明年的钱打折扣 |
| $-\gamma\mu_c$ | −4% | 预期消费增长使明年的边际效用较低，折扣更大 |
| $\tfrac12\gamma^2\sigma_c^2$ | 0.08% | $m$ 的凸性修正，与无风险利率中的预防性储蓄项对应 |
| $\mu_d$ | 2% | 红利的预期对数增长 |
| $\tfrac12\sigma_d^2$ | 0.72% | 红利水平的凸性修正 |
| $-\gamma\rho_{cd}\sigma_c\sigma_d$ | −0.24% | 风险调整：红利与消费同涨同跌，价格被压低 |
| 合计 $\log a$ | 约 −2.445% | $a = e^{-0.02445} \approx 0.9758$ |

前三项之和是 $-\log R^f$（式 {eq}`eq-ch1-logrf`），中间两项之和是 $\log G$（$G = e^{\mu_d + \sigma_d^2/2}$ 为红利的期望总增长），最后一项是对数溢价的相反数，所以 $\log a = \log G - \log R^f - \gamma\rho_{cd}\sigma_c\sigma_d$，即 $a = G/(R^f e^{\gamma\rho_{cd}\sigma_c\sigma_d}) = G/\E[R]$，与上面“$a$ 的含义”一段的结论一致。**特例**：$\sigma_c = \sigma_d = 0$（没有风险）时 $a = \delta e^{\mu_d - \gamma\mu_c}$；Lucas 树的特例见下文。

(ch1-unpack-pd)=
**拆开讲式 {eq}`eq-ch1-pd`。** $P_t$ 是第 $t$ 期末、当年红利派发之后的价格（元），$D_t$ 是第 $t$ 年派发的红利（元），二者在第 $t$ 期末都已知；$P_t/D_t$ 是价格相当于年度红利的多少倍，没有单位。展开 $a/(1-a) = a + a^2 + a^3 + \cdots$，每一项是一年红利的价格（以 $D_t$ 为单位）。默认参数下 $0.9758/(1 - 0.9758) \approx 40.4$ 倍，年度股息率 $D/P \approx 2.48\%$。**特例**：$a \to 1$ 时 $P/D \to \infty$；对数效用的 Lucas 树中 $a = \delta$，$P/D = \delta/(1-\delta) = 99$（见下文的表）。

```{code-cell} python
# 拆开讲 eq-ch1-a 与 eq-ch1-pd：log a 的六项；log a = log G − log R^f − γρσ_cσ_d
a_terms = pd.Series({"log δ": math.log(p.delta), "−γμ_c": -p.gamma * p.mu_c,
                     "γ²σ_c²/2": p.gamma**2 * p.sigma_c**2 / 2, "μ_d": p.mu_d, "σ_d²/2": p.sigma_d**2 / 2,
                     "−γρσ_cσ_d": -p.gamma * p.rho_cd * p.sigma_c * p.sigma_d})
print((100 * a_terms).round(4).to_string(), f"\n合计 log a = {100 * a_terms.sum():.4f}%")
assert math.isclose(a_terms.sum(), math.log(a), rel_tol=1e-12) and -0.02446 < a_terms.sum() < -0.02444
assert math.isclose(a_terms.iloc[:3].sum(), -math.log(p.risk_free), rel_tol=1e-12)
assert math.isclose(math.log(a), math.log(G) - math.log(p.risk_free) - p.log_equity_premium, rel_tol=1e-12)
assert round(100 / pd_ratio, 2) == 2.48 and 40.3 < a / (1 - a) < 40.5
```

**Lucas 树的特例。** 代入 $\rho_{cd} = 1$、$\sigma_d = \sigma_c$、$\mu_d = \mu_c$，式 {eq}`eq-ch1-a` 中 $\sigma_d^2 + \gamma^2\sigma_c^2 - 2\gamma\rho_{cd}\sigma_c\sigma_d = (1 - \gamma)^2\sigma_c^2$，于是

```{math}
:enumerated: false
a = \delta\exp\left((1 - \gamma)\mu_c + \tfrac12(1 - \gamma)^2\sigma_c^2\right) .
```

默认的 $\delta = 0.99$、$\gamma = 2$、$\mu_c = \sigma_c = 2\%$ 下，$a \approx 0.9706$，价格—红利比约为 33.0。上式有一个出人意料的含义：**预期增长提高，价格—红利比不一定上升。** 增长 $\mu_c$ 提高有两个相反的作用：未来的红利更多，抬高价格；同时由式 {eq}`eq-ch1-logrf`，无风险利率上升 $\gamma$ 倍于增长的幅度，压低价格。二者合在指数中的 $(1 - \gamma)\mu_c$ 里：$\gamma > 1$ 时折现的作用占优，增长越快价格—红利比越低；$\gamma = 1$（对数效用）时两种作用恰好抵消，$P/D = \delta/(1 - \delta)$ 与增长无关；$\gamma < 1$ 时红利的作用占优，增长足够快时 $a \geq 1$，价格无穷大。下表的代码验证这些结论，$a \geq 1$ 时模拟器抛出 `ValueError`。

```{code-cell} python
rows = []
for g in (0.5, 1.0, 2.0, 5.0):
    row = {"γ": g}
    for mu in (0.02, 0.03):
        tree = ConsumptionParams(gamma=g, mu_c=mu, mu_d=mu, sigma_d=p.sigma_c, rho_cd=1.0)   # Lucas 树：红利 = 消费
        a_tree = tree.delta * math.exp((1 - g) * mu + 0.5 * (1 - g) ** 2 * tree.sigma_c**2)
        assert math.isclose(tree.dividend_discount, a_tree, rel_tol=1e-12)
        try:
            row[f"μ_c = {mu:.0%}"] = tree.price_dividend_ratio
        except ValueError:                                  # a ≥ 1：价格无穷大
            row[f"μ_c = {mu:.0%}"] = math.inf
    rows.append(row)
lucas = pd.DataFrame(rows).set_index("γ")
print(lucas.to_string(float_format=lambda v: f"{v:,.1f}"))
assert lucas.loc[1.0].nunique() == 1 and math.isclose(lucas.loc[1.0].iloc[0], 0.99 / 0.01)
assert 32.9 < lucas.loc[2.0, "μ_c = 2%"] < 33.1 and lucas.loc[2.0, "μ_c = 3%"] < lucas.loc[2.0, "μ_c = 2%"]
assert lucas.loc[5.0, "μ_c = 3%"] < lucas.loc[5.0, "μ_c = 2%"]
assert lucas.loc[0.5, "μ_c = 2%"] > 1e6 and lucas.loc[0.5, "μ_c = 3%"] == math.inf
```

$\gamma = 0.5$、$\mu_c = 2\%$ 时 $a$ 只比 1 小约 $3 \times 10^{-7}$，价格—红利比接近 300 万，已经近乎无穷大；$\mu_c = 3\%$ 时 $a > 1$，不存在有限的价格。这样的参数组合不能用来讨论价格，“常见误区”一节还会回到这一点。

### 收益与红利增长

由式 {eq}`eq-ch1-pd`，令 $v = a/(1-a)$，则 $(v + 1)/v = 1/a$，红利索取权的总收益为

$$
R_{t+1} = \frac{P_{t+1} + D_{t+1}}{P_t} = \frac{v + 1}{v}\cdot\frac{D_{t+1}}{D_t} = \frac{e^{\Delta d_{t+1}}}{a}, \qquad \log R_{t+1} = \Delta d_{t+1} - \log a, \qquad \E[R] = \frac{G}{a} .
$$ (eq-ch1-return)

价格—红利比不变，价格与红利以同样的速度增长：资本利得 $P_{t+1}/P_t = e^{\Delta d_{t+1}}$，再加上股息率带来的部分。例如某年红利增长 10%，价格也上涨 10%，加上红利 $D_{t+1}/P_t = 1.10/40.4 \approx 2.72\%$，当年的总收益率约为 12.72%，等于 $1.10/a - 1$。对数收益就是对数红利增长加一个常数，所以它与红利增长一样独立同分布，这也给出了 1.4 节[命题 %s](#prop-ch1-premium) 中用到的 $\log R = \Delta d - \log a$。

(ch1-unpack-return)=
**拆开讲式 {eq}`eq-ch1-return`。** $R_{t+1}$ 是红利索取权第 $t+1$ 年的总收益，没有单位（例如 1.1272，即收益率 12.72%），在第 $t$ 期末随机；$P_{t+1} + D_{t+1}$ 是第 $t+1$ 期末的支付（元），$P_t$ 在第 $t$ 期末已知。第二个等号把 $P_{t+1} = vD_{t+1}$、$P_t = vD_t$ 代入：$(vD_{t+1} + D_{t+1})/(vD_t) = \frac{v+1}{v}\cdot\frac{D_{t+1}}{D_t}$，其中 $D_{t+1}/D_t = e^{\Delta d_{t+1}}$ 是总红利增长（随机），$(v+1)/v = 1/a$ 是常数（默认 $1/0.9758 \approx 1.0248$）。拆成两部分更直观：$R_{t+1} = P_{t+1}/P_t + D_{t+1}/P_t = e^{\Delta d_{t+1}} + e^{\Delta d_{t+1}}/v$，前者是资本利得，后者是股息带来的部分。$\log R_{t+1} = \Delta d_{t+1} - \log a$ 是对数（连续复利）收益，等于对数红利增长加上常数 $-\log a \approx 2.445\%$。$\E[R] = G/a \approx 1.0276/0.9758 \approx 1.0530$ 是期望总收益，不带时间下标，因为每年都一样。**特例**：$\sigma_d = 0$ 时红利确定地增长，$R_{t+1} = e^{\mu_d}/a$ 是常数；Lucas 树中 $\Delta d = \Delta c$，收益就是 $e^{\Delta c_{t+1}}/a$（[习题 %s](#ex-ch1-lucas) 第 4 问）。

```{code-cell} python
# 拆开讲 eq-ch1-return：R = 资本利得 + 股息部分，(v + 1)/v = 1/a，E[R] = G/a
v = pd_ratio
assert math.isclose(1.10 + 1.10 / v, 1.10 / a, rel_tol=1e-12) and math.isclose((v + 1) / v, 1 / a, rel_tol=1e-12)
assert 1.0247 < 1 / a < 1.0248 and 0.02444 < -math.log(a) < 0.02446 and 1.0530 < G / a < 1.0531
assert math.isclose(G / a, p.expected_return, rel_tol=1e-12)
print(f"1/a = {1 / a:.4f}，−log a = {-math.log(a):.4%}，E[R] = G/a = {G / a:.4f}")
```

下面用 `simulate_consumption` 模拟 60 年（种子 2026），令 $D_0 = 1$，由 $\Delta d$ 累积出红利路径，再由 $P_t = vD_t$ 得到价格路径，核对由价格与红利算出的收益等于模拟器给出的 `R` 列，并画出路径。

```{code-cell} python
#| label: fig-ch1-lucas-path
#| caption: 一条 60 年的模拟路径（种子 2026，默认假设参数）。左：红利与价格（对数纵轴），两条线平行，间距为价格—红利比的对数；右：红利索取权的年度收益率与无风险利率
import matplotlib.pyplot as plt

path = simulate_consumption(p, seed=2026, n_years=60).data
D = np.concatenate([[1.0], np.exp(np.cumsum(path["dd"].to_numpy()))])    # (T+1,)：D_0 = 1，D_t = exp(Δd_1 + Δd_2 + ⋯ + Δd_t)
Pp = pd_ratio * D                                                         # (T+1,)：P_t = v D_t
R_from_prices = (Pp[1:] + D[1:]) / Pp[:-1]                               # (T,)：(P_t + D_t) / P_{t−1}
assert np.allclose(R_from_prices, path["R"].to_numpy(), rtol=1e-12)
assert math.isclose(1.10 / a - 1, 0.10 + 1.10 / pd_ratio, rel_tol=1e-12) and 0.1271 < 1.10 / a - 1 < 0.1273   # 正文中红利增长 10% 的例子

years = np.arange(len(D))
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
ax = axes[0]
ax.plot(years, Pp, color="C0", label="价格")
ax.plot(years, D, color="C1", label="红利")
ax.set_yscale("log")
ax.set_xlabel("年")
ax.set_title("红利与价格（对数纵轴）")
ax.text(years[-1] + 1, Pp[-1], "价格", va="center", fontsize=9, color="0.25")
ax.text(years[-1] + 1, D[-1], "红利", va="center", fontsize=9, color="0.25")
ax.set_xlim(0, years[-1] + 9)
ax = axes[1]
ax.bar(path.index, 100 * (path["R"] - 1), color="C0", width=0.8, label="年度收益率")
ax.axhline(100 * (p.risk_free - 1), color="0.3", lw=1, ls="--", label="无风险利率")
ax.axhline(0, color="0.6", lw=0.8)
ax.set_xlabel("年")
ax.set_ylabel("%")
ax.set_title("红利索取权的年度收益率")
ax.legend(loc="lower left", fontsize=8)
fig.tight_layout()
plt.show()
```

左图中两条线始终平行：对数纵轴上，价格与红利的间距是 $\log(P/D) = \log 40.4$，不随时间变化。右图中收益率的起伏完全来自红利增长的起伏，与过去的收益、与当时的价格—红利比都无关。

### 独立同分布模型的局限

价格—红利比是常数，这是独立同分布假设的直接结果：每一期面对的未来都一样，没有理由给同样的红利不同的倍数。于是价格—红利比（或股息率）不能预测未来的收益，因为它根本不变；收益独立同分布，过去的收益也不能预测未来的收益。而在实际数据中，市场整体的估值比率波动很大，并且对未来的收益有一定的预测力，见 @cochrane2011presidential 与[术语对照页](#glossary-discount-rate)的“折现率”一条。要让模型产生时变的价格—红利比，必须让期望收益（折现率）或期望红利增长随时间变化。第 6 章引入时变的贴现率，并用 Campbell–Shiller 分解把价格—红利比的变动拆成两部分；第 7 章讨论能够产生时变风险溢价的消费模型。

(sec-ch1-linear)=
## 1.6 回到第 0 章：线性 SDF 是消费 SDF 的一阶近似

**白话。** 第 0 章的 SDF 是因子的线性函数，本章的 SDF 是消费增长的指数函数。年度消费增长的波动很小，在它通常的取值范围内，指数曲线几乎是一条直线，所以用直线近似它误差很小。这条直线就是第 0 章那种线性 SDF 的一维版本，只不过“因子”换成了消费增长。

**例子。** 默认参数下 $\E[m] \approx 0.9519$。若某年对数消费增长 $\Delta c = 0.04$（水平增长约 4.08%，比均值高一个标准差），精确的 $m = 0.99\,e^{-2 \times 0.04} \approx 0.91389$，线性近似 $\E[m]\left[1 - 2 \times (0.04 - 0.02)\right] \approx 0.91387$，几乎相同。若某年对数消费增长 $\Delta c = -0.08$（水平下降约 7.7%，低于均值 5 个标准差），精确的 $m = 0.99\,e^{0.16} \approx 1.162$，线性近似 $0.9519 \times (1 + 0.2) \approx 1.142$，低了约 1.7%：离均值越远，直线与指数曲线的差别越大。

**公式。** 由 $\E[m] = \delta\exp(-\gamma\mu_c + \gamma^2\sigma_c^2/2)$（[命题 %s](#prop-ch1-rf)），SDF 可以写成

```{math}
:enumerated: false
m_{t+1} = \delta e^{-\gamma\Delta c_{t+1}} = \E[m]\exp\left(-\gamma(\Delta c_{t+1} - \mu_c) - \tfrac12\gamma^2\sigma_c^2\right) .
```

对 $e^y$ 在 $y = 0$ 处做一阶展开，$e^y \approx 1 + y$，并略去与 $(\Delta c - \mu_c)$ 无关的二阶小量 $\gamma^2\sigma_c^2/2$：

$$
m_{t+1} \approx \E[m]\left[1 - \gamma(\Delta c_{t+1} - \mu_c)\right], \qquad \frac{m_{t+1}}{\E[m]} = R^f m_{t+1} \approx 1 - \gamma(\Delta c_{t+1} - \mu_c) .
$$ (eq-ch1-linear)

这个近似的均值恰好等于 $\E[m]$，所以无风险利率不变。一种常见的、更粗略的写法是 $m \approx \delta(1 - \gamma\Delta c)$，它是在 $\Delta c = 0$ 处展开的结果，均值 $\delta(1 - \gamma\mu_c)$ 与 $\E[m]$ 不同；在均值处展开并按 $\E[m]$ 归一化更准确，下文都用式 {eq}`eq-ch1-linear`。

(ch1-unpack-linear)=
**拆开讲式 {eq}`eq-ch1-linear`。** $m_{t+1}$ 是年度 SDF，没有单位，在第 $t$ 期末随机；$\E[m] = 1/R^f$（默认约 0.9519）是常数；$\Delta c_{t+1} - \mu_c$ 是第 $t+1$ 年的对数消费增长偏离均值的幅度，随机（例如对数消费增长为 0.04 时为 $+0.02$）；$\gamma$ 把这一偏离换算成 SDF 的相对变化：对数消费增长比均值高 0.01，$m_{t+1}$ 比 $\E[m]$ 约低 $\gamma\%$。第二式两边除以 $\E[m]$，$R^f m_{t+1}$ 的均值为 1，正是第 0 章的归一化（1.2 节）。代入上面的例子：$1 - 2 \times (0.04 - 0.02) = 0.96$，乘以 0.9519 得约 0.91387。**特例**：$\gamma = 0$ 时 $m_{t+1} = \E[m]$ 是常数，投资者风险中性。第二式就是第 0 章线性 SDF $m_t = 1 - \mathbf b'(\mathbf f_t - \bv\mu)$（[拆解](#ch0-unpack-sdf)）在 $K = 1$ 时的形式：唯一的因子 $f_{1,t}$ 换成 $\Delta c_{t+1}$，均值 $\mu_1$ 换成 $\mu_c$，系数 $b_1$ 换成 $\gamma$，下表逐项对照。**数据矩阵**：把 $T$ 年的 $R^f m_t$ 排成 $T\times 1$ 的列向量，第 $t$ 个元素约为 $1 - \gamma(\Delta c_t - \mu_c)$；它只带时间下标，所有资产共用这一条序列。

把式 {eq}`eq-ch1-linear` 的第二式与第 0 章的式 {eq}`eq-ch0-sdf` $m_t = 1 - \mathbf b'(\mathbf f_t - \bv\mu)$ 对照，二者形式完全相同：

| 第 0 章 | 本章（一阶近似） | 说明 |
|---|---|---|
| $m_t$，归一化 $\E[m_t] = 1$ | $R^f m_{t+1} = m_{t+1}/\E[m]$ | 同一种归一化（1.2 节） |
| $K$ 个因子 | 1 个因子 | |
| 因子 $\mathbf f_t \in \Real^K$ | 消费增长 $\Delta c_{t+1}$ | 第 0 章是可交易的超额收益；本章不可交易 |
| $\bv\mu = \E[\mathbf f_t]$ | $\mu_c = \E[\Delta c]$ | |
| $\bv\Sigma_f = \Var(\mathbf f_t)$ | $\sigma_c^2 = \Var(\Delta c)$ | |
| SDF 系数 $\mathbf b = \bv\Sigma_f^{-1}\bv\mu$ | $\gamma$ | 第 0 章由“为因子自身定价”唯一确定；本章是偏好参数 |
| 风险溢价 $\bv\lambda = \bv\Sigma_f\mathbf b = \bv\mu$ | $\lambda_c = \sigma_c^2\gamma \neq \mu_c$ | 都满足“风险溢价 = 因子方差 × SDF 系数” |
| $\E[r_i] = \bv\beta_i'\bv\lambda$ | $\E[R^e] \approx \beta_{\Delta c}\lambda_c$ | 式 {eq}`eq-ch1-ccapm` |

表中倒数第二行的关系对两种情形都成立，推导只需一行。对线性 SDF $m = 1 - \mathbf b'(\mathbf f - \bv\mu)$（$\E[m] = 1$）与任一超额收益 $R^e$，由 $0 = \E[mR^e]$ 得 $\E[R^e] = -\Cov(m, R^e) = \mathbf b'\Cov(\mathbf f, R^e)$，其中 $\Cov(\mathbf f, R^e) \in \Real^K$。令载荷 $\bv\beta = \bv\Sigma_f^{-1}\Cov(\mathbf f, R^e)$（$K\times K$ 乘 $K\times 1$），则 $\Cov(\mathbf f, R^e) = \bv\Sigma_f\bv\beta$，于是

```{math}
:enumerated: false
\E[R^e] = \underbrace{\mathbf b'}_{1\times K}\underbrace{\bv\Sigma_f}_{K\times K}\underbrace{\bv\beta}_{K\times 1} = \bv\beta'\bv\lambda, \qquad \underbrace{\bv\lambda}_{K\times 1} = \underbrace{\bv\Sigma_f}_{K\times K}\underbrace{\mathbf b}_{K\times 1} ,
```

这里用到 $\bv\Sigma_f$ 对称。两种情形的区别只在于 $\mathbf b$ 由什么决定。第 0 章的因子是超额收益，必须为自身定价：把上式用于因子本身（$\bv\beta = \mathbf I_K$），得 $\bv\mu = \bv\lambda = \bv\Sigma_f\mathbf b$，所以 $\mathbf b = \bv\Sigma_f^{-1}\bv\mu$，这就是第 0 章[定理 %s](#thm-ch0-sdf) 中唯一的 $\mathbf b$。本章的 $\Delta c$ 不是收益，没有这个约束，系数 $\gamma$ 由偏好决定，风险溢价 $\lambda_c = \gamma\sigma_c^2$ 与 $\mu_c$ 无关。所以看到一个线性 SDF，要先问因子是不是可交易的超额收益：是，风险溢价可以用因子的均值估计；不是，则需要另行估计（第 4、10 章）。

(ch1-unpack-bsb)=
**拆开讲 $\E[R^e] = \mathbf b'\bv\Sigma_f\bv\beta = \bv\beta'\bv\lambda$。** $\mathbf b$、$\bv\mu$、$\mathbf f$ 的分量见第 0 章的[拆解](#ch0-unpack-sdf)，$\bv\Sigma_f$ 的结构见[拆解](#ch0-unpack-ts)，$\bv\beta'\bv\lambda$ 的逐项展开见[拆解](#ch0-unpack-lambda)。这里新出现的是协方差向量与这项资产的载荷向量：

```{math}
:enumerated: false
\underbrace{\Cov(\mathbf f, R^e)}_{K\times 1} = \begin{pmatrix} \Cov(f_1, R^e) \\ \Cov(f_2, R^e) \\ \vdots \\ \Cov(f_K, R^e) \end{pmatrix} , \qquad
\underbrace{\bv\beta}_{K\times 1} = \bv\Sigma_f^{-1}\Cov(\mathbf f, R^e) = \begin{pmatrix} \beta_1 \\ \beta_2 \\ \vdots \\ \beta_K \end{pmatrix} .
```

- $\Cov(f_k, R^e)$：这项资产的超额收益与第 $k$ 个因子的协方差，单位是收益的平方
- $\beta_k$：这项资产对第 $k$ 个因子的载荷，即超额收益对 $K$ 个因子做多元回归时第 $k$ 个斜率，没有单位

**维度**：$\mathbf b'$ 是 $1\times K$，$\bv\Sigma_f$ 是 $K\times K$，$\bv\beta$ 是 $K\times 1$，三者相乘得一个数；$\bv\lambda = \bv\Sigma_f\mathbf b$ 是 $K\times K$ 乘 $K\times 1$，得 $K\times 1$。**展开运算**：记 $\Sigma_{kl}$ 为 $\bv\Sigma_f$ 的 $(k, l)$ 元素，

```{math}
:enumerated: false
\mathbf b'\bv\Sigma_f\bv\beta = \sum_{k=1}^{K}\sum_{l=1}^{K} b_k\,\Sigma_{kl}\,\beta_l = \sum_{l=1}^{K}\Bigl(\sum_{k=1}^{K}\Sigma_{lk}\,b_k\Bigr)\beta_l = \sum_{l=1}^{K}\lambda_l\,\beta_l , \qquad \lambda_l = \sum_{k=1}^{K}\Sigma_{lk}\,b_k ,
```

第二步用到对称性 $\Sigma_{kl} = \Sigma_{lk}$。白话：第 $l$ 个因子的风险溢价 $\lambda_l$ 是这个因子与“$\mathbf b$ 加权的因子组合” $\mathbf b'\mathbf f$ 的协方差；资产的期望超额收益是它对各因子的载荷乘以各因子的风险溢价，再加总。因子互不相关时 $\lambda_l = \sigma_l^2 b_l$，“风险溢价 = 因子方差 × SDF 系数”逐个因子成立。**特例**：$K = 1$ 时 $\bv\Sigma_f$ 就是一个方差，$\lambda_1 = \Var(f_1)\, b_1$；取 $f_1 = \Delta c$、$b_1 = \gamma$，就是本章的 $\lambda_c = \sigma_c^2\gamma$（上表倒数第二行）。把关系用于第 0 章的 $K$ 个因子本身：第 $k$ 个因子对自己的载荷为 1、对其他因子为 0，把 $K$ 个因子的载荷 $\bv\beta'$ 按行堆叠，就是 $K\times K$ 的单位矩阵 $\mathbf I_K$，于是 $K$ 个因子的期望超额收益为 $\bv\mu = \mathbf I_K\bv\lambda = \bv\lambda$。**数据矩阵**：有 $N$ 项资产时，把它们的 $\bv\beta'$ 按行堆叠成 $N\times K$ 的载荷矩阵 $\mathbf B$，$N$ 项资产的期望超额收益组成的列向量为 $\mathbf B\bv\lambda$（$N\times 1$）；**所有资产共用 $\mathbf b$、$\bv\Sigma_f$ 与 $\bv\lambda$**，每项资产各有一行载荷。第 0 章的模拟市场是**月度**的，$K = 3$，月度协方差矩阵 $\bv\Sigma_f = \operatorname{diag}(0.05^2, 0.03^2, 0.03^2)$，$\mathbf b = (2.4,\ 3.33,\ 4.44)'$（$b_k$ 的单位是“月度收益的倒数”），于是月度风险溢价 $\bv\lambda = \bv\Sigma_f\mathbf b = (0.0025 \times 2.4,\ 0.0009 \times 3.33,\ 0.0009 \times 4.44)' \approx (0.6\%, 0.3\%, 0.4\%)' = \bv\mu$，即每月 0.6%、0.3%、0.4%（下方第二个代码单元核对）；本章的 $\lambda_c = 0.08\%$ 则是年度的，二者不能直接比较大小。

线性近似还继承了第 0 章线性 SDF 的缺点：它可能取负值。$1 - \gamma(\Delta c - \mu_c) < 0$ 的概率为 $\Phi(-1/(\gamma\sigma_c))$（$\Phi$ 是标准正态分布函数），默认参数下 $1/(\gamma\sigma_c) = 25$，概率可以忽略；$\gamma = 25$ 时 $1/(\gamma\sigma_c) = 2$，概率约为 2.3%，不能再忽略。精确的 $m = \delta e^{-\gamma\Delta c}$ 则总是正的。

下面的代码核对上面的例子，用二维求积比较精确 SDF 与线性 SDF 对红利索取权的定价，并在第 0 章的模拟市场参数下核对 $\bv\lambda = \bv\Sigma_f\mathbf b = \bv\mu$。

```{code-cell} python
Em = p.sdf_mean
m_exact = lambda dc: p.delta * np.exp(-p.gamma * dc)
m_lin = lambda dc: Em * (1 - p.gamma * (dc - p.mu_c))                    # 式 eq-ch1-linear
for dc_val in (0.04, -0.08):
    print(f"Δc = {dc_val:+.0%}：精确 m = {m_exact(dc_val):.5f}，线性近似 {m_lin(dc_val):.5f}，"
          f"相对差 {m_lin(dc_val) / m_exact(dc_val) - 1:+.2%}")
assert 0.9519 < Em < 0.9520 and abs(m_lin(0.04) - m_exact(0.04)) < 5e-5 and -0.018 < m_lin(-0.08) / m_exact(-0.08) - 1 < -0.016

# 二维求积：精确 SDF 与线性 SDF 对红利索取权超额收益的定价
Re_q = np.exp(dd_q) / a - p.risk_free                                    # 各节点上的 R^e = e^{Δd}/a − R^f
E_mRe_exact = (W2 * m_exact(dc_q) * Re_q).sum()
E_mRe_lin = (W2 * m_lin(dc_q) * Re_q).sum()
print(f"E[m R^e]：精确 SDF {E_mRe_exact:.2e}，线性 SDF {E_mRe_lin:.2e}；E[m_lin] − E[m] = {(W2 * m_lin(dc_q)).sum() - Em:.1e}")
assert abs(E_mRe_exact) < 1e-14 and abs((W2 * m_lin(dc_q)).sum() - Em) < 1e-14
assert -5e-6 < E_mRe_lin < 0                                             # 线性 SDF 留下很小的定价误差

# 线性 SDF 取负值的概率 Φ(−1/(γσ_c))
for g in (2, 25):
    print(f"γ = {g}：P(线性 SDF < 0) = {norm.cdf(-1 / (g * p.sigma_c)):.2e}")
assert norm.cdf(-1 / (2 * p.sigma_c)) < 1e-100 and 0.022 < norm.cdf(-1 / (25 * p.sigma_c)) < 0.023

# 第 0 章的模拟市场：b = Σ_f⁻¹μ，λ = Σ_f b = μ
mp = MarketParams()
b0 = np.linalg.solve(mp.factor_cov, np.asarray(mp.factor_mean))         # (K, K) b = (K,)
assert np.allclose(mp.factor_cov @ b0, mp.factor_mean)                  # (K, K) @ (K,) -> (K,)：λ = Σ_f b = μ
assert math.isclose(p.sigma_c**2 * p.gamma, 0.0008) and p.sigma_c**2 * p.gamma != p.mu_c   # 本章：λ_c = σ_c²γ ≠ μ_c
```

```{code-cell} python
# 拆开讲 E[R^e] = b'Σ_f β = β'λ：K = 3 时双重求和与矩阵乘法一致；因子自身的载荷矩阵为 I_K
lam0 = mp.factor_cov @ b0                                              # (K, K) @ (K,) -> (K,)：λ = Σ_f b
beta_ex = np.array([0.9, -0.4, -0.3])                                  # 某项资产的载荷 β: (K,)（假设数字）
double_sum = sum(b0[k] * mp.factor_cov[k, l] * beta_ex[l] for k in range(3) for l in range(3))
print("b =", b0.round(4), "；λ = Σ_f b =", lam0.round(4), f"；b'Σ_f β = {double_sum:.4%}")
assert np.allclose(b0, [2.4, 10 / 3, 40 / 9]) and math.isclose(double_sum, beta_ex @ lam0)
B_self = np.linalg.solve(mp.factor_cov, mp.factor_cov)                 # 因子本身的载荷 Σ_f⁻¹Cov(f, f') (K, K)
assert np.allclose(B_self, np.eye(3)) and np.allclose(B_self @ lam0, mp.factor_mean)   # B = I_K，E[f] = Bλ = μ
```

线性 SDF 对红利索取权留下的定价误差 $\E[m_{\text{lin}}R^e]$ 是负的，量级只有 $10^{-6}$（由 `assert` 保证），对应 1.4 节“哪些关系是近似”中线性化把期望超额收益高估约 0.12%。

1.6 节回答了第 0 章留下的一部分问题：线性 SDF 从何而来（边际效用的一阶近似），SDF 系数与风险溢价之间的关系 $\bv\lambda = \bv\Sigma_f\mathbf b$，以及因子可交易与否带来的区别。更一般的理论在后面几章展开：第 2 章说明无套利等价于存在严格为正的 SDF；第 3 章讨论均值—方差前沿、把 SDF 投影到资产收益张成的空间，以及 Hansen–Jagannathan 界；第 4 章讨论 SDF 表示与 β 表示的一般等价性，以及包括不可交易因子与模拟组合在内的因子模型。
