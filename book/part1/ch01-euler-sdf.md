---
short_title: 第 1 章　消费与 SDF
kernelspec:
  name: python3
  display_name: Python 3
numbering:
  enumerator: '1.%s'
  exercise:
    template: 习题 %s
---

(sec-ch1-euler-sdf)=
# 第 1 章　消费、欧拉方程与随机贴现因子

**本页准备。** 本书每个页面的代码在独立的内核中运行。下面的单元导入本页用到的模块。

```{code-cell} python
import numpy as np
import sympy as sp
from scipy.optimize import brentq, minimize_scalar
```

## 学习目标

读完本章，读者应当能够：

1. 写出两期消费—储蓄问题，推出一阶条件（欧拉方程），并说明它为什么对投资者能自由买卖的每一项资产都成立；
2. 从欧拉方程得到基本定价方程 $P_t = \E_t[m_{t+1}x_{t+1}]$，说出随机贴现因子 $m_{t+1} = \delta u'(C_{t+1})/u'(C_t)$ 中每一项的含义，并把它用于股票、无风险债券、总收益与超额收益；
3. 用协方差分解解释“在坏时候赚钱的资产期望收益低”，并能用两状态的数值例子复算；
4. 在幂效用与对数正态消费下推出无风险利率与风险溢价的闭式解，说出其中哪些关系是精确的、哪些是近似，近似误差有多大；
5. 理解消费 CAPM 是以不可交易因子为因子的 β 模型，说明为什么消费增长的风险溢价 $\lambda_c$ 不等于消费增长的均值；
6. 推出独立同分布时 Lucas 树的价格—红利比（$P/D$，价格与红利之比，不是市盈率），说明 $a < 1$ 与横截性条件的关系，以及这一模型为什么不能产生收益的可预测性；
7. 说明第 0 章的线性 SDF $m_t = 1 - \mathbf b'(\mathbf f_t - \bv\mu)$ 与消费 SDF 的关系。

## 动机

第 0 章的随机贴现因子（SDF）是“给定”的：模拟市场直接设定 $m_t = 1 - \mathbf b'(\mathbf f_t - \bv\mu)$（各分量、维度与逐项展开见第 0 章的[拆解](#ch0-unpack-sdf)），再由因子的均值与协方差反推出 $\mathbf b$。这足以检验估计方法，却回答不了一个更基本的问题：**$m$ 从哪里来？为什么某些状态下的一块钱更值钱？**

本章给出资产定价理论中最基本的回答：$m$ 来自投资者的边际效用。投资者在今天消费与为明天储蓄之间权衡，如果某项资产的价格偏离了它对投资者的边际价值，投资者就会调整持仓，直到二者相等。由此得到的欧拉方程，同时决定了无风险利率、各项资产的期望收益与价格水平。“坏时候”也有了具体的含义：消费低、边际效用高的状态。

从消费出发有三个好处。第一，风险溢价可以追溯到偏好参数（耐心 $\delta$、风险厌恶 $\gamma$）与消费的分布。第二，在幂效用与对数正态分布下所有量都有闭式解，可以用代码直接核对，模拟器 `apsim.consumption` 正是按这些闭式解实现的。第三，第 0 章的线性 SDF 是消费 SDF 的一阶近似（1.6 节），由此可以看清“因子是超额收益”与“因子是消费增长”的区别，为第 3、4 章的一般理论做准备。

本章只讲理论与模拟，不讨论实证数字：消费模型在数据中遇到的困难（股权溢价之谜）留到第 7 章，估计偏好参数的广义矩方法（GMM）留到第 11 章。本章所有数值例子都使用**年度的假设参数**，不是对任何市场的校准。

(sec-ch1-path)=
## 本章各页与阅读路径

本章分为以下几页。第一次阅读时，标为“必读”的部分构成本章的主线；1.4 节末尾的“哪些关系是近似”与“模拟实验”可以先跳过，读到后文引用时再回来。

| 页面 | 内容 | 必读或选读 | 约多少字 |
|---|---|---|---|
| 本页 | 学习目标、动机、1.1 两期消费—储蓄问题、1.2 欧拉方程与 SDF、1.3 风险调整 | 必读 | 2.8 万 |
| [1.4 幂效用与对数正态消费](./ch01-lognormal.md) | 无风险利率与风险溢价的闭式解、消费 CAPM、近似误差 | 必读（“哪些关系是近似”一节选读，约 4000 字） | 2.7 万 |
| [1.5～1.6 Lucas 树与回到第 0 章](./ch01-lucas.md) | 多期定价与横截性条件、价格—红利比、收益与红利增长、线性 SDF 是一阶近似 | 必读 | 2.8 万 |
| [A 股视角、模拟实验与常见误区](./ch01-experiments.md) | 中国消费数据的特点、三项模拟验证、四个常见误区 | 选读（常见误区建议读） | 1.4 万 |
| [本章小结、习题与延伸阅读](./ch01-exercises.md) | 本章小结；概念题、推导题、编程题；文献与延伸阅读 | 读小结（必读）；习题与延伸阅读选读 | 全页 1.4 万，其中小结 1700 |

只读必读部分约 8.1 万字（全章各页约 11.2 万字）。字数按源文件的字符数统计，含公式与代码，不计页首元数据。

(sec-ch1-two-period)=
## 1.1 两期消费—储蓄问题

**白话。** 设想今天少吃一碗饭，把省下的 1 元钱拿去买某项资产，明年得到支付 $x$。值不值得？代价是今天少消费 1 元所损失的满足感；收益是明年多出的 $x$ 元带来的满足感，但人们更看重眼前，明年的满足感要打一个折扣，而且明年有好有坏，要对各种情形取平均。收益大于代价就多买，代价大于收益就少买甚至卖出。买得越多，今天越拮据、明年越宽裕，代价上升、收益下降，买到二者恰好相等时停下，这就是最优的储蓄。

关键在于满足感不随钱数线性增加：钱少时多一元钱很重要，钱多时就没那么重要。经济学用**边际效用**描述这一点：效用函数 $u(C)$ 衡量消费 $C$ 带来的满足感，导数 $u'(C)$ 是多消费一元带来的满足感，它随消费增加而递减。

**例子。** 假设参数：一位投资者今年收入 $e_t = 100$ 元；明年有两种等可能的状态，好状态收入 106 元，坏状态收入 96 元。效用函数为 $u(C) = C^{1-\gamma}/(1-\gamma)$，取 $\gamma = 2$，于是边际效用 $u'(C) = C^{-2}$；主观贴现因子 $\delta = 0.99$，表示明年的满足感按 0.99 折算成今天的满足感。市场上有一项资产，今天的价格 $P_t = 1$ 元，明年好状态支付 1.20 元、坏状态支付 0.90 元。

先看要不要买第一份。少消费 1 元的代价是 $u'(100) = 100^{-2} = 1.0000 \times 10^{-4}$ 个效用单位。明年的收益是

```{math}
:enumerated: false
\delta\left[0.5 \times u'(106) \times 1.20 + 0.5 \times u'(96) \times 0.90\right] = 0.99 \times \left[0.5 \times \frac{1.20}{106^2} + 0.5 \times \frac{0.90}{96^2}\right] \approx 1.0121 \times 10^{-4} ,
```

比代价高约 1.2%，所以应当买入。买入 $\xi$ 份之后，今天消费 $100 - \xi$，明年两种状态下的消费分别为 $106 + 1.20\xi$ 与 $96 + 0.90\xi$。继续买，代价上升、收益下降，在 $\xi^* \approx 0.294$ 份时二者相等（下方代码求出）。注意坏状态下的支付虽然少，单位支付带来的边际效用却高：$1/96^2$ 比 $1/106^2$ 高约 22%。

**公式。** 投资者选择买入的份数 $\xi$（标量，可以为负，表示卖出或卖空），求解

$$
\max_{\xi}\; u(C_t) + \delta\, \E_t\left[u(C_{t+1})\right] \qquad \text{s.t.}\quad C_t = e_t - P_t\xi, \qquad C_{t+1} = e_{t+1} + x_{t+1}\xi ,
$$ (eq-ch1-problem)

其中 $C_t$、$C_{t+1}$ 是今天与明年的消费，$e_t$、$e_{t+1}$ 是不买资产时的收入（禀赋），$P_t$ 是资产今天的价格，$x_{t+1}$ 是它明年的支付，$\delta \in (0, 1)$ 是**主观贴现因子**（subjective discount factor），度量投资者的耐心：$\delta$ 越小，越看重眼前。效用函数 $u$ 严格递增、严格凹，即 $u' > 0$、$u'' < 0$。时点约定与第 0 章的理论写法相同：下标 $t$ 表示第 $t$ 期末已知，$\E_t[\cdot]$ 是以第 $t$ 期末信息为条件的期望；$e_{t+1}$ 与 $x_{t+1}$ 在第 $t$ 期末未知，是随机变量。

(ch1-unpack-problem)=
**拆开讲式 {eq}`eq-ch1-problem`。** 本章的一期是一年。式中的量可以按“第 $t$ 期末已知”与“到第 $t+1$ 期才揭晓”分成两组。

- 第 $t$ 期末已知，或由投资者此刻选定：
  - $\xi$：买入的份数，单位是“份”，是投资者此刻要选的决策变量；上例的最优值约为 0.294 份
  - $e_t$：今年的收入，以元计（上例为 100 元）
  - $P_t$：资产今天的价格，单位是“元/份”（上例为 1 元）
  - $C_t = e_t - P_t\xi$：今年的消费，以元计；收入中拿出 $P_t\xi$ 元买资产，其余用于消费
  - $\delta$：主观贴现因子，没有单位，是每年的折扣：明年的 1 个效用单位只相当于今天的 $\delta$ 个（上例为 0.99）
- 第 $t$ 期末未知，随明年的状态而定（随机变量）：
  - $e_{t+1}$：明年的收入，以元计（上例中好状态为 106 元、坏状态为 96 元）
  - $x_{t+1}$：每份资产明年的支付，单位是“元/份”，是钱数而不是收益率；对股票而言，$x_{t+1}$ 是明年的价格加上红利（上例为 1.20 元或 0.90 元）
  - $C_{t+1} = e_{t+1} + x_{t+1}\xi$：明年的消费，以元计，等于明年的收入加上 $\xi$ 份资产带来的支付

效用 $u(\cdot)$ 的单位记作“效用单位”。它的绝对大小没有经济含义：把 $u$ 乘以一个正数或加上一个常数，最优的 $\xi$ 不变；有意义的是效用的比较，以及边际效用 $u'(C)$，即每多消费 1 元增加的效用单位。$\E_t[\cdot]$ 对明年的各种状态按概率取平均，概率以第 $t$ 期末的信息为准。**展开运算**：两状态（好、坏，概率为 $\pi_{\text{好}}$、$\pi_{\text{坏}}$）时，目标函数逐项写出是

```{math}
:enumerated: false
u(e_t - P_t\xi) + \delta\left[\pi_{\text{好}}\, u(e_{\text{好}} + x_{\text{好}}\xi) + \pi_{\text{坏}}\, u(e_{\text{坏}} + x_{\text{坏}}\xi)\right] ,
```

代入上例的数字是 $u(100 - \xi) + 0.99\left[0.5\, u(106 + 1.20\xi) + 0.5\, u(96 + 0.90\xi)\right]$：今年少消费 $\xi$ 元，换来明年好状态多消费 $1.20\xi$ 元、坏状态多消费 $0.90\xi$ 元。下方 `sympy` 代码中的 `pi_1`、`e_g`、`x_g` 等符号就是这里的 $\pi_{\text{好}}$、$e_{\text{好}}$、$x_{\text{好}}$。**特例**：$\xi = 0$ 时不交易，两期的消费就等于收入；收入与支付都没有不确定性（只有一种状态）时，期望可以去掉，问题化为确定性的两期储蓄问题。

**推导。** 把两个约束代入目标函数，问题化为对一个变量 $\xi$ 的无约束最大化：

```{math}
:enumerated: false
V(\xi) = u(e_t - P_t\xi) + \delta\, \E_t\left[u(e_{t+1} + x_{t+1}\xi)\right] .
```

对 $\xi$ 求导。第一项由链式法则得 $u'(e_t - P_t\xi) \cdot (-P_t) = -P_t u'(C_t)$。第二项中，求导与期望可以交换次序（状态有限时期望是有限项的加权和，逐项求导即可；一般情形需要可积性条件，本书默认满足），$\partial u(e_{t+1} + x_{t+1}\xi)/\partial\xi = u'(C_{t+1})\, x_{t+1}$。于是

```{math}
:enumerated: false
V'(\xi) = -P_t u'(C_t) + \delta\, \E_t\left[u'(C_{t+1})\, x_{t+1}\right] .
```

令 $V'(\xi^*) = 0$，得到一阶条件

$$
\underbrace{P_t\, u'(C_t)}_{\text{少消费 } P_t \text{ 元的代价}} = \underbrace{\delta\, \E_t\left[u'(C_{t+1})\, x_{t+1}\right]}_{\text{明年多得 } x_{t+1} \text{ 的收益}} ,
$$ (eq-ch1-foc)

其中 $C_t$、$C_{t+1}$ 按最优的 $\xi^*$ 计算。再看二阶导数：$V''(\xi) = P_t^2 u''(C_t) + \delta\, \E_t[u''(C_{t+1})\, x_{t+1}^2]$，由 $u'' < 0$，第一项在 $P_t \neq 0$ 时总是负的，第二项不为正（只要 $x_{t+1}$ 不恒为 0 就严格为负），所以 $V$ 严格凹，一阶条件既是必要条件也是充分条件，满足它的 $\xi^*$ 就是唯一的最大值点。

式 {eq}`eq-ch1-foc` 有两点值得强调。第一，它只涉及“再多买一点点”的边际权衡，并不要求投资者大量持有这项资产：只要投资者能以价格 $P_t$ 自由地多买或少买一点，这个等式就必须成立，否则调整持仓能提高效用。第二，有多项资产时，对每一项资产的份数分别求导，得到的每一个一阶条件都是式 {eq}`eq-ch1-foc` 的形式，只是 $P_t$、$x_{t+1}$ 换成该资产的价格与支付，$C_t$、$C_{t+1}$ 则是同一个最优消费计划。

下面先用 `sympy` 对一般的效用函数 $u$ 核对求导的结果（两状态，$u$ 是未指定的函数），再用上面的数值例子求出 $\xi^*$，并用直接最大化 $V(\xi)$ 的结果核对。

```{code-cell} python
xi, P, e0, d = sp.symbols("xi P e_0 delta", positive=True)
pi1, pi2, e1g, e1b, x1g, x1b = sp.symbols("pi_1 pi_2 e_g e_b x_g x_b", positive=True)
u, u_prime = sp.Function("u"), sp.Function("u'")                               # 未指定的 u 与记号 u'
V = u(e0 - P * xi) + d * (pi1 * u(e1g + x1g * xi) + pi2 * u(e1b + x1b * xi))   # 两状态的 V(ξ)
# sympy 把 u 在点 c 处的导数写成 Subs(Derivative(u(z), z), z, c)，这里统一改写为 u'(c)
dV = V.diff(xi).replace(lambda e: isinstance(e, sp.Subs), lambda e: u_prime(e.point[0]))
foc_formula = -P * u_prime(e0 - P * xi) + d * (pi1 * u_prime(e1g + x1g * xi) * x1g
                                               + pi2 * u_prime(e1b + x1b * xi) * x1b)
print("V'(ξ) =", dV)
print("V'(ξ) − [−P u'(C_t) + δ E(u'(C_{t+1}) x)] =", sp.simplify(dV - foc_formula))
assert sp.simplify(dV - foc_formula) == 0

# 数值例子（假设参数）：幂效用 γ = 2，δ = 0.99
delta, gamma = 0.99, 2.0
prob = np.array([0.5, 0.5])                        # 好状态、坏状态的概率
e_now, e_next = 100.0, np.array([106.0, 96.0])     # 今年收入；明年两种状态的收入 (2,)
price, payoff = 1.0, np.array([1.20, 0.90])        # 资产今天的价格；明年两种状态的支付 (2,)
u_pow = lambda c: c ** (1 - gamma) / (1 - gamma)   # u(C) = C^{1−γ}/(1−γ)
mu_pow = lambda c: c ** (-gamma)                   # u'(C) = C^{−γ}

cost0 = price * mu_pow(e_now)                                  # ξ = 0 时第一份的代价
gain0 = delta * prob @ (mu_pow(e_next) * payoff)               # ξ = 0 时第一份的收益
foc = lambda s: -price * mu_pow(e_now - price * s) + delta * prob @ (mu_pow(e_next + payoff * s) * payoff)
xi_star = brentq(foc, -50, 50)                                 # 解 V'(ξ) = 0
direct = minimize_scalar(lambda s: -(u_pow(e_now - price * s) + delta * prob @ u_pow(e_next + payoff * s)),
                         bounds=(-50, 50), method="bounded", options={"xatol": 1e-10})
print(f"第一份的代价 {cost0:.4e}，收益 {gain0:.4e}，收益/代价 = {gain0 / cost0:.4f}")
print(f"一阶条件的解 ξ* = {xi_star:.6f}；直接最大化 V(ξ) 得 {direct.x:.6f}")
assert np.isclose(cost0, 1e-4) and np.isclose(gain0, 1.0121e-4, rtol=1e-4)
assert abs(xi_star - direct.x) < 1e-6 and 0.293 < xi_star < 0.295
assert np.isclose(mu_pow(96.0) / mu_pow(106.0), 1.2192, rtol=1e-4)     # 坏状态的边际效用高约 22%
```

(ch1-unpack-foc)=
**拆开讲式 {eq}`eq-ch1-foc`。** 两边的单位都是“效用单位/份”。左边：多买一份，今天要少消费 $P_t$ 元，每少 1 元损失 $u'(C_t)$ 个效用单位，乘积是多买一份的边际代价，在第 $t$ 期末已知。右边：这一份明年带来 $x_{t+1}$ 元，每多 1 元增加 $u'(C_{t+1})$ 个效用单位；二者都随明年的状态变化，是随机的，所以先在每个状态上相乘，再用 $\E_t$ 取平均，最后乘以 $\delta$ 折回今天。两状态时，右边逐项写出是 $\delta\left[\pi_{\text{好}}\, u'(C_{\text{好}})\, x_{\text{好}} + \pi_{\text{坏}}\, u'(C_{\text{坏}})\, x_{\text{坏}}\right]$。上例在 $\xi^*$ 处，左边为 $1 \times 99.71^{-2} \approx 1.0059 \times 10^{-4}$，右边为 $0.99 \times \left[0.5 \times 1.20/106.35^2 + 0.5 \times 0.90/96.26^2\right] \approx 1.0059 \times 10^{-4}$，二者相等（下方代码核对）。注意乘积 $u'(C_{t+1})\, x_{t+1}$ 的期望不能拆成 $\E_t[u'(C_{t+1})]\,\E_t[x_{t+1}]$：支付恰好在边际效用高的状态多还是少，正是 1.3 节风险调整的来源。**特例**：若 $u$ 是线性的，$u'$ 恒为常数（投资者风险中性，相当于下面幂效用中 $\gamma \to 0$ 的极限），两边约去边际效用，得 $P_t = \delta\,\E_t[x_{t+1}]$：价格就是期望支付按 $\delta$ 折现，不需要风险调整。**多项资产**：第 $j$ 项资产的一阶条件是 $P_{j,t}\,u'(C_t) = \delta\,\E_t[u'(C_{t+1})\, x_{j,t+1}]$；$u'(C_t)$ 与 $u'(C_{t+1})$ 由同一个消费计划决定，为所有资产共用，每项资产只有自己的价格 $P_{j,t}$ 与支付 $x_{j,t+1}$。把多项资产的价格排成列向量的写法见 1.3 节的[拆解](#ch1-unpack-states)。

```{code-cell} python
# 拆开讲 eq-ch1-foc：在 ξ* 处，边际代价 P_t u'(C_t) 等于边际收益 δ E_t[u'(C_{t+1}) x_{t+1}]
C_t_star = e_now - price * xi_star                                   # 今天的消费
C_next_star = e_next + payoff * xi_star                              # 明年两种状态的消费 (2,)
cost_star = price * mu_pow(C_t_star)
gain_star = delta * prob @ (mu_pow(C_next_star) * payoff)            # (2,) @ (2,)：两状态按概率加权求和
print(f"ξ* 处：边际代价 {cost_star:.4e}，边际收益 {gain_star:.4e}")
assert np.isclose(cost_star, gain_star) and np.isclose(cost_star, 1.0059e-4, rtol=1e-4)
```

**幂效用。** 本章始终使用上例中的幂效用（power utility）

$$
u(C) = \frac{C^{1-\gamma}}{1-\gamma} \quad (\gamma > 0,\ \gamma \neq 1), \qquad u(C) = \log C \quad (\gamma = 1), \qquad u'(C) = C^{-\gamma} .
$$ (eq-ch1-power)

$\gamma = 1$ 的对数效用是 $\gamma \to 1$ 时的极限（减去常数 $1/(1-\gamma)$ 后取极限），边际效用同样是 $C^{-\gamma}$。$\gamma$ 决定边际效用随消费下降得多快：$-C u''(C)/u'(C) = \gamma$，这个量称为**相对风险厌恶系数**，$\gamma$ 越大，投资者越不愿意承担消费的波动，所以本书称 $\gamma$ 为风险厌恶系数。1.4 节会看到，同一个 $\gamma$ 还决定投资者愿意在不同时期之间调配消费的程度，这是幂效用的一个重要限制（见“常见误区”一节）。

(ch1-unpack-power)=
**拆开讲式 {eq}`eq-ch1-power`。** $C$ 是消费的水平，以元计，不是对数；$\gamma$ 没有单位；$u(C)$ 以效用单位计，$u'(C) = C^{-\gamma}$ 的单位是“效用单位/元”。分母 $1-\gamma$ 只是为了让导数恰好等于 $C^{-\gamma} > 0$：$\gamma > 1$ 时 $u(C)$ 本身是负数（上例 $u(100) = 100^{-1}/(-1) = -0.01$），但它仍随 $C$ 递增，正负号没有含义。再求一次导数，$u''(C) = -\gamma C^{-\gamma-1}$，所以 $-Cu''(C)/u'(C) = \gamma C^{-\gamma}/C^{-\gamma} = \gamma$，与消费水平无关，这就是“常数相对风险厌恶”的意思。**展开**：两个消费水平上的边际效用之比只取决于二者之比，$u'(C_1)/u'(C_2) = (C_1/C_2)^{-\gamma}$；上例 $u'(96)/u'(106) = (106/96)^2 \approx 1.219$，坏状态多 1 元的价值高约 22%。消费不论以元还是万元计，这个比值都不变，所以下一节的 SDF 只依赖消费增长 $C_{t+1}/C_t$。**特例**：$\gamma \to 0$ 时 $u(C) \to C$，边际效用恒为 1，投资者风险中性；$\gamma = 1$ 是对数效用；本章的数值例子取 $\gamma = 2$。$\gamma$ 越大，同样的消费差距对应的边际效用之比越大，例如 $\gamma = 5$ 时 $(106/96)^5 \approx 1.64$。

```{code-cell} python
# 拆开讲 eq-ch1-power：γ = 2 时 u(100) = −0.01；边际效用之比只取决于消费之比
print(f"u(100) = {u_pow(100.0):.4f}，u'(96)/u'(106) = {mu_pow(96.0) / mu_pow(106.0):.4f}，γ = 5 时 {(106 / 96) ** 5:.4f}")
assert np.isclose(u_pow(100.0), -0.01) and np.isclose(mu_pow(96.0) / mu_pow(106.0), (106 / 96) ** 2)
assert np.isclose(mu_pow(9600.0) / mu_pow(10600.0), mu_pow(96.0) / mu_pow(106.0))   # 消费单位放大 100 倍，比值不变
assert np.isclose((106 / 96) ** 5, 1.6412, rtol=1e-4)
```

(sec-ch1-sdf)=
## 1.2 欧拉方程与随机贴现因子

**白话。** 把式 {eq}`eq-ch1-foc` 两边除以 $u'(C_t)$，左边只剩价格 $P_t$，右边是支付 $x_{t+1}$ 乘以一个随状态变化的“汇率”再取期望。它回答的是：明年某个状态下的 1 元，按投资者今天的感受折合今天的多少元？它由耐心 $\delta$ 与“明年那个状态的边际效用除以今天的边际效用”相乘得到，明年消费低的状态，边际效用高，那里的 1 元折合今天的钱就多。这就是随机贴现因子 $m_{t+1}$：它把明年的钱折算成今天的钱，所以叫贴现因子；它的取值取决于明年实现的状态，所以是随机的。

**例子。** 回到 1.1 节的例子。在最优持仓 $\xi^* \approx 0.294$ 处，今天消费约 99.71 元，明年好状态消费约 106.35 元、坏状态约 96.26 元，于是

```{math}
:enumerated: false
m_{\text{好}} = 0.99 \times \left(\frac{106.35}{99.71}\right)^{-2} \approx 0.870, \qquad m_{\text{坏}} = 0.99 \times \left(\frac{96.26}{99.71}\right)^{-2} \approx 1.062 .
```

坏状态的 $m$ 比好状态高约 22%。用它给资产定价：$0.5 \times 0.870 \times 1.20 + 0.5 \times 1.062 \times 0.90 = 1.000$，恰好等于价格 1 元，这正是一阶条件。$m$ 的期望 $\E_t[m_{t+1}] \approx 0.966$：明年确定得到的 1 元，今天只值约 0.966 元，对应约 3.51% 的无风险利率（下方代码核对）。

**公式。** 由式 {eq}`eq-ch1-foc`，

$$
P_t = \E_t\left[m_{t+1}\, x_{t+1}\right], \qquad m_{t+1} = \delta\, \frac{u'(C_{t+1})}{u'(C_t)} ,
$$ (eq-ch1-pricing)

其中价格 $P_t$、支付 $x_{t+1}$ 与 SDF $m_{t+1}$ 都是标量；$\delta$ 与 $u'(C_t)$ 在第 $t$ 期末已知，可以移进或移出 $\E_t$。这就是第 0 章 0.2 节“随机贴现因子”一段中的 $P = \E[mx]$，现在 $m$ 有了来源。在幂效用 {eq}`eq-ch1-power` 下，

$$
m_{t+1} = \delta \left(\frac{C_{t+1}}{C_t}\right)^{-\gamma} .
$$ (eq-ch1-power-sdf)

(ch1-unpack-pricing)=
**拆开讲式 {eq}`eq-ch1-pricing`。** $P = \E[mx] = \sum_s \pi_s m_s x_s$ 的逐状态写法与各项单位见第 0 章的[拆解](#ch0-unpack-sdf)，这里补上时点与 $m_{t+1}$ 的来源。$P_t$ 以元计，在第 $t$ 期末已知；$x_{t+1}$ 以元计，在第 $t$ 期末是随机的；$m_{t+1}$ 也是随机的，因为分子 $u'(C_{t+1})$ 取决于明年的消费，而 $\delta$ 与分母 $u'(C_t)$ 在第 $t$ 期末已知。$m_{t+1}$ 的单位是“效用单位/明年的元”除以“效用单位/今天的元”，即“今天的元/明年的元”，像一个汇率，数值上没有单位；本章的一期是一年，所以它把明年的钱折成今天的钱。两状态时逐项写出是

```{math}
:enumerated: false
P_t = \pi_{\text{好}}\, m_{\text{好}}\, x_{\text{好}} + \pi_{\text{坏}}\, m_{\text{坏}}\, x_{\text{坏}} ,
```

上面例子中的定价算式就是这一写法：代入那里算出的 $m_{\text{好}} \approx 0.870$、$m_{\text{坏}} \approx 1.062$，两项约为 0.522 与 0.478，合计 1.000（下方代码核对）。其中 $\pi_s m_s$ 是“只在状态 $s$ 支付 1 元”的证券今天的价格，称为**状态价格**（state price），上例为 $0.5 \times 0.870 \approx 0.435$ 元与 $0.5 \times 1.062 \approx 0.531$ 元：坏状态的 1 元更贵。**特例**：明年的消费在各状态都相同（没有消费风险）时，$m_{t+1}$ 是一个确定的数，可以提到期望之外，$P_t = m_{t+1}\,\E_t[x_{t+1}]$：价格是期望支付乘以一个共同的折扣，没有风险调整；$m_{t+1}$ 在各状态之间差得越多，价格偏离“期望支付乘以平均折扣”就可能越远（1.3 节）。

(ch1-unpack-powersdf)=
**拆开讲式 {eq}`eq-ch1-power-sdf`。** $C_{t+1}/C_t$ 是**总**消费增长，即两个消费水平之比，不是对数，也不是减去 1 之后的增长率，没有单位：上例好状态为 $106.35/99.71 \approx 1.0667$（增长 6.67%），坏状态为 $96.26/99.71 \approx 0.9655$（下降 3.45%）。$C_t$ 在第 $t$ 期末已知，$C_{t+1}$ 随机，所以 $m_{t+1}$ 随机。指数 $-\gamma$ 的含义是：消费增长每高 1%，$m_{t+1}$ 大约低 $\gamma\%$（$\log m_{t+1}$ 对 $\log(C_{t+1}/C_t)$ 的斜率为 $-\gamma$）。上面例子中的 $m_{\text{好}} \approx 0.870$、$m_{\text{坏}} \approx 1.062$，就是把这两个总消费增长代入本式、取 $\delta = 0.99$、$\gamma = 2$ 得到的（下方代码核对）。**特例**：$\gamma \to 0$ 时 $m_{t+1} \to \delta$，是常数，投资者风险中性，所有资产都按 $\delta$ 折现；在消费不增长（$C_{t+1} = C_t$）的状态上，$m_{t+1} = \delta$。进入 $m_{t+1}$ 的只有增长率，没有消费水平，这是幂效用的性质（见上一节的[拆解](#ch1-unpack-power)中“边际效用之比只取决于消费之比”）。

同一个 $m_{t+1}$ 为投资者能自由买卖的每一项资产定价（1.1 节末尾的第二点）。式 {eq}`eq-ch1-pricing` 有四种常用的特例。

1. **股票。** 今天以 $P_t$ 买入，明年得到价格 $P_{t+1}$ 加红利 $D_{t+1}$，支付为 $x_{t+1} = P_{t+1} + D_{t+1}$：

   ```{math}
   :enumerated: false
   P_t = \E_t\left[m_{t+1}(P_{t+1} + D_{t+1})\right] .
   ```

   1.5 节把这个式子向前迭代，得到“价格等于未来红利的贴现和”。

   式中 $P_t$ 是今年年末、今年的红利 $D_t$ 派发之后的价格（买入者拿不到 $D_t$），在第 $t$ 期末已知；$P_{t+1}$ 是明年年末的价格，$D_{t+1}$ 是明年派发的红利，都以元计，在第 $t$ 期末都是随机的。

2. **无风险债券。** 今天花 1 元买入，明年确定得到 $R^f_t$ 元，$R^f_t$ 是无风险总收益。支付在第 $t$ 期末已知，可以提到期望之外：$1 = \E_t[m_{t+1}]\, R^f_t$，即

   $$
   R^f_t = \frac{1}{\E_t[m_{t+1}]} .
   $$ (eq-ch1-rf)

3. **总收益。** 式 {eq}`eq-ch1-pricing` 两边除以 $P_t$（它在第 $t$ 期末已知），令 $R_{t+1} = x_{t+1}/P_t$：

   ```{math}
   :enumerated: false
   1 = \E_t\left[m_{t+1} R_{t+1}\right] .
   ```

4. **超额收益。** 对任意总收益 $R_{t+1}$，令 $R^e_{t+1} = R_{t+1} - R^f_t$，把第 3 条与第 2 条相减：

   ```{math}
   :enumerated: false
   0 = \E_t\left[m_{t+1} R_{t+1}\right] - \E_t\left[m_{t+1}\right] R^f_t = \E_t\left[m_{t+1} R^e_{t+1}\right] .
   ```

   超额收益是零成本头寸（借入 1 元、买入 1 元资产）的收益，价格为 0，所以定价方程的左边是 0。

(ch1-unpack-rf)=
**拆开讲式 {eq}`eq-ch1-rf` 与第 3、4 条。** $R^f_t$ 是**总**收益：今天的 1 元明年变成 $R^f_t$ 元，是年度的量，没有单位（可以理解为“明年的元/今天的元”）；它减去 1 才是通常说的利率。上例 $R^f_t \approx 1.0351$，即年利率约 3.51%。下标 $t$ 表示它在第 $t$ 期末已知，适用于从 $t$ 到 $t+1$ 的这一年。$\E_t[m_{t+1}]$ 是明年**确定**得到 1 元的资产今天的价格，也就是各状态价格之和：上例为 $0.435 + 0.531 \approx 0.966$ 元，取倒数得 $1/0.966 \approx 1.0351$。第 3 条中的 $R_{t+1} = x_{t+1}/P_t$ 同样是总收益（上例的资产为 1.20 或 0.90，而不是 20% 或 −10%），在第 $t$ 期末随机；第 4 条中的 $R^e_{t+1} = R_{t+1} - R^f_t$ 是两个总收益之差，也等于两个收益率之差（上例为 $1.20 - 1.0351 \approx 0.165$ 或 $0.90 - 1.0351 \approx -0.135$），同样在第 $t$ 期末随机，是年度的量，没有单位。**特例**：没有不确定性、消费也不增长时 $m_{t+1} = \delta$，$R^f_t = 1/\delta$，上例为 $1/0.99 \approx 1.0101$，利率约 1.01%，只由“耐心”决定；1.4 节把对数利率拆成耐心、跨期替代与预防性储蓄三项。

**与第 0 章的对应。** 第 0 章的定价方程是 $0 = \E_{t-1}[m_t r_{i,t}]$（$r_{i,t}$ 的单位与时点见第 0 章的[拆解](#ch0-unpack-exret)），它就是上面的第 4 条，只有两处写法不同。

- **时点标号。** 第 0 章的面板写法是从 $t-1$ 到 $t$，适用于第 $t$ 期的无风险利率在第 $t-1$ 期末已知，记作 $R^f_{t-1}$，超额收益 $r_{i,t} = R_{i,t} - R^f_{t-1}$；本章的理论写法是从 $t$ 到 $t+1$，同一个量记作 $R^f_t$，$R^e_{t+1} = R_{t+1} - R^f_t$。两种写法只差一个时间平移，下标都标明“在哪个时点已知”，见[记号表](../notation.md)的“时点约定”。本书的理论章节使用 $R^f_t$，面板与模拟数据部分使用 $R^f_{t-1}$。
- **均值的归一化。** 第 0 章取 $\E[m_t] = 1$。原因是第 0 章的模拟市场只有超额收益，而 $0 = \E[m R^e]$ 对 $m$ 乘以任何正常数都成立，SDF 的均值无法识别。本章有了无风险债券，$\E_t[m_{t+1}] = 1/R^f_t$ 由利率决定。把本章的 $m_{t+1}$ 乘以 $R^f_t$，得到 $\tilde m_{t+1} = R^f_t m_{t+1}$，它满足 $\E_t[\tilde m_{t+1}] = 1$ 与 $0 = \E_t[\tilde m_{t+1} R^e_{t+1}]$，就是第 0 章归一化下的 SDF。

```{code-cell} python
C_now = e_now - price * xi_star                    # 最优持仓下今天的消费
C_next = e_next + payoff * xi_star                 # 明年两种状态的消费 (2,)
m = delta * (C_next / C_now) ** (-gamma)           # m_{t+1} = δ (C_{t+1}/C_t)^{−γ}，(2,)
E_m = prob @ m
R_f = 1 / E_m                                      # 第 2 条：R^f_t = 1 / E_t[m_{t+1}]
R_stock = payoff / price                           # 该资产的总收益 (2,)
print("消费：今天", round(C_now, 2), "，明年", C_next.round(2), "；m =", m.round(4))
print(f"E[m x] = {prob @ (m * payoff):.6f}（价格 1），E[m] = {E_m:.4f}，R^f = {R_f:.4f}")
assert np.isclose(prob @ (m * payoff), price)                       # 一阶条件：P_t = E_t[m x]
assert np.allclose(m, [0.870, 1.062], atol=5e-4) and np.isclose(m[1] / m[0], 1.2206, rtol=1e-3)
assert np.isclose(E_m, 0.9661, atol=1e-4) and np.isclose(R_f, 1.0351, atol=1e-4)
assert np.isclose(prob @ (m * R_stock), 1.0)                        # 第 3 条：1 = E[m R]
assert np.isclose(prob @ (m * (R_stock - R_f)), 0.0)                # 第 4 条：0 = E[m R^e]
m_tilde = R_f * m                                                   # 第 0 章的归一化：E[m̃] = 1
assert np.isclose(prob @ m_tilde, 1.0) and np.isclose(prob @ (m_tilde * (R_stock - R_f)), 0.0)
```

下面的单元核对上面几段拆解中的数字：状态价格、总消费增长与超额收益。

```{code-cell} python
# 拆开讲 eq-ch1-pricing、eq-ch1-power-sdf 与 eq-ch1-rf：状态价格 π_s m_s、总消费增长与超额收益
q_state = prob * m                                  # 状态价格 (2,)：只在好、坏状态支付 1 元的证券今天的价格
growth = C_next / C_now                             # 总消费增长 C_{t+1}/C_t (2,)
print("状态价格", q_state.round(4), "；总消费增长", growth.round(4), "；超额收益", (R_stock - R_f).round(4))
assert np.allclose(q_state, [0.435, 0.531], atol=5e-4) and np.isclose(q_state.sum(), E_m)
assert np.allclose(q_state * payoff, [0.522, 0.478], atol=5e-4)
assert np.allclose(growth, [1.0667, 0.9655], atol=5e-5) and np.allclose(delta * growth ** (-gamma), m)
assert np.allclose(R_stock - R_f, [0.1649, -0.1351], atol=5e-5) and np.isclose(1 / delta, 1.0101, atol=5e-5)
```

(sec-ch1-risk)=
## 1.3 风险调整：协方差决定期望收益

**白话。** 一项资产的价格可以拆成两部分：一部分是把期望支付按无风险利率折现，好像投资者不在乎风险；另一部分是风险调整，取决于支付与 $m$ 的协方差。$m$ 在坏时候高，所以在坏时候支付多的资产（支付与 $m$ 正相关）像一份保险：它在最需要钱的时候给钱，投资者愿意多付，价格高于不考虑风险时的价值，期望收益因此低，甚至低于无风险利率。在好时候支付多的资产则相反，投资者只肯少付，期望收益因此高。决定期望收益的不是资产自身的波动有多大，而是它“在什么时候”赚钱。

**例子。** 仍用 1.1 节的投资者，$m_{\text{好}} \approx 0.870$、$m_{\text{坏}} \approx 1.062$，无风险总收益 $R^f \approx 1.0351$。比较两项**期望支付相同**的资产（假设数字）：

| 资产（假设） | 好状态支付 | 坏状态支付 | 期望支付 | 不考虑风险的价值 $\E[x]/R^f$ | $\Cov(m, x)$ | 价格 $\E[mx]$ | 期望收益率 | 风险溢价 $\E[R] - R^f$ |
|---|---|---|---|---|---|---|---|---|
| 股票型资产 | 1.20 | 0.90 | 1.05 | 1.0144 | −0.0144 | 1.0000 | 5.00% | 约 1.49% |
| 保险型资产 | 1.00 | 1.10 | 1.05 | 1.0144 | 0.0048 | 1.0192 | 约 3.02% | 约 −0.49% |
| 无风险债券 | 1 | 1 | 1 | 0.9661 | 0 | 0.9661 | 约 3.51% | 0 |

两项资产的期望支付都是 1.05 元，不考虑风险时都值 1.0144 元。股票型资产在好时候支付多，与 $m$ 负相关，价格被压低 0.0144 元，期望收益比无风险利率高约 1.49 个百分点；保险型资产在坏时候支付多，与 $m$ 正相关，价格被抬高 0.0048 元，期望收益反而比无风险利率低约 0.49 个百分点：投资者为它在坏时候提供的保障付了“保费”。这里投资者并不持有保险型资产与债券，表中二者的价格是在当前消费计划下、持有量为 0 时投资者对它们的边际估值（影子价格）：以这个价格，他既不想买也不想卖。

**公式。** 由协方差的定义 $\Cov(m, x) = \E[mx] - \E[m]\E[x]$，即 $\E[mx] = \E[m]\E[x] + \Cov(m, x)$。把它代入式 {eq}`eq-ch1-pricing`，并利用 $\E_t[m_{t+1}] = 1/R^f_t$：

$$
P_t = \underbrace{\frac{\E_t[x_{t+1}]}{R^f_t}}_{\text{不考虑风险的价值}} + \underbrace{\Cov_t(m_{t+1}, x_{t+1})}_{\text{风险调整}} ,
$$ (eq-ch1-cov-price)

其中 $\Cov_t$ 是以第 $t$ 期末信息为条件的协方差。对超额收益，同样展开 $0 = \E_t[m_{t+1} R^e_{t+1}]$：

```{math}
:enumerated: false
0 = \E_t[m_{t+1}]\,\E_t[R^e_{t+1}] + \Cov_t(m_{t+1}, R^e_{t+1}) ,
```

两边除以 $\E_t[m_{t+1}] = 1/R^f_t$，移项得

$$
\E_t[R^e_{t+1}] = -\frac{\Cov_t(m_{t+1}, R^e_{t+1})}{\E_t[m_{t+1}]} = -R^f_t\, \Cov_t(m_{t+1}, R^e_{t+1}) .
$$ (eq-ch1-premium)

由于 $R^f_t$ 在第 $t$ 期末已知，$\Cov_t(m_{t+1}, R^e_{t+1}) = \Cov_t(m_{t+1}, R_{t+1})$，上式中的超额收益可以换成总收益。第一个等号就是[术语对照页](#glossary-risk)中“风险”一条与第 0 章 0.2 节写过的 $\E[R^e] = -\Cov(m, R^e)/\E[m]$，第 0 章取 $\E[m] = 1$，所以那里分母为 1；现在 $m$ 来自边际效用，分母 $\E_t[m_{t+1}]$ 就是 $1/R^f_t$。

在幂效用下，$m_{t+1} = \delta(C_{t+1}/C_t)^{-\gamma}$ 是消费增长的减函数：消费增长高的状态 $m$ 低。所以，收益随消费增长同涨同跌的资产，$\Cov(m, R^e) < 0$，期望超额收益为正；在消费增长低时收益高的资产像保险，期望超额收益为负。收益中与 $m$ 不相关的部分，不论波动多大，都不影响期望收益。这就是术语对照页所说“只有系统性风险得到补偿”的来源：系统性，就是与 $m$ 相关，也就是与边际效用相关。

(ch1-unpack-covprice)=
**拆开讲式 {eq}`eq-ch1-cov-price`。** 三项都以元计。$\E_t[x_{t+1}]$ 是期望支付，除以总收益 $R^f_t$ 就是按无风险利率折现一年的值；$\Cov_t(m_{t+1}, x_{t+1})$ 中 $m$ 没有单位、$x$ 以元计，所以它也以元计，是对风险的调整，可正可负。$m_{t+1}$ 与 $x_{t+1}$ 在第 $t$ 期末都是随机的，但它们的条件期望与条件协方差在第 $t$ 期末都是已知的数，所以式中三项都在第 $t$ 期末已知。两状态时，协方差逐项写出是

```{math}
:enumerated: false
\Cov_t(m_{t+1}, x_{t+1}) = \pi_{\text{好}}\,(m_{\text{好}} - \E_t[m_{t+1}])(x_{\text{好}} - \E_t[x_{t+1}]) + \pi_{\text{坏}}\,(m_{\text{坏}} - \E_t[m_{t+1}])(x_{\text{坏}} - \E_t[x_{t+1}]) .
```

以股票型资产为例：$m$ 在好状态比均值低约 0.096、在坏状态比均值高约 0.096，支付在好状态比均值高 0.15 元、在坏状态比均值低 0.15 元，于是协方差为 $0.5 \times (-0.096) \times 0.15 + 0.5 \times 0.096 \times (-0.15) \approx -0.0144$ 元，与上表一致：两个状态上 $m$ 与支付的偏离方向相反，乘积都为负。**特例**：支付确定（如无风险债券，$x_{t+1} = 1$）或与 $m$ 不相关时，协方差为 0，价格就是 $\E_t[x_{t+1}]/R^f_t$；表中无风险债券一行正是 $1/1.0351 \approx 0.9661$。

(ch1-unpack-premium)=
**拆开讲式 {eq}`eq-ch1-premium`。** $\E_t[R^e_{t+1}]$ 是期望超额收益，即期望收益率减去无风险利率，是年度的量，没有单位，常写成百分数（上表股票型资产约为 1.49%）；$\Cov_t(m_{t+1}, R^e_{t+1})$ 是两个无单位的量的协方差，也没有单位；$R^f_t$ 是总收益（约 1.0351），在第 $t$ 期末已知，所以可以写在 $\Cov_t$ 外面。股票型资产在两个状态下的超额收益约为 $1.20 - 1.0351 \approx 0.165$ 与 $0.90 - 1.0351 \approx -0.135$，各自偏离均值 0.15；由于价格恰为 1 元，超额收益与支付只差常数，它与 $m$ 的协方差就是上一段算出的 $\Cov_t(m_{t+1}, x_{t+1}) \approx -0.0144$，于是 $\E_t[R^e_{t+1}] \approx -1.0351 \times (-0.0144) \approx 1.49\%$。保险型资产的协方差为正：$\Cov(m, R^e) = \Cov(m, R) = \Cov(m, x)/P \approx 0.0048/1.0192 \approx 0.0047$（上表中的 0.0048 是对支付 $x$ 算的 $\Cov(m, x)$，这里对收益算，要除以价格），期望超额收益约为 $-0.49\%$。**特例**：在第 0 章的归一化 $\E[m] = 1$ 下，分母为 1，上式化为 $\E[R^e] = -\Cov(m, R^e)$（第 0 章的[拆解](#ch0-unpack-sdf)）；与 $m$ 不相关的资产期望超额收益为 0，不论它自身的波动有多大。

下面先用 `sympy` 对一般的两状态分布核对式 {eq}`eq-ch1-cov-price` 与 {eq}`eq-ch1-premium`（概率、$m$ 与支付都是符号），再复算上表。复算时用到 NumPy 的广播（broadcasting）：`P_tab` 的形状是 `(3,)`，`P_tab[:, None]` 在末尾插入一个长度为 1 的维度，变成 `(3, 1)` 的列；它与 `(3, 2)` 的 `pay` 相除时，这一列被自动复制到两个状态上，于是第 $i$ 行的两个支付都除以资产 $i$ 的价格，即 $R_{i,s} = x_{i,s}/P_i$。

```{code-cell} python
p1 = sp.Symbol("p_1", positive=True)
mg, mb, xg, xb = sp.symbols("m_g m_b x_g x_b", positive=True)
Ex = lambda a, b: p1 * a + (1 - p1) * b                        # 两状态期望：E[y] = p_1 y_g + (1 − p_1) y_b
Em_s = Ex(mg, mb)
cov_mx = Ex(mg * xg, mb * xb) - Em_s * Ex(xg, xb)               # Cov(m, x)
Pr = Ex(mg * xg, mb * xb)                                       # P = E[m x]
Rf_s = 1 / Em_s
Re_g, Re_b = xg / Pr - Rf_s, xb / Pr - Rf_s                     # 各状态的超额收益
cov_mRe = Ex(mg * Re_g, mb * Re_b) - Em_s * Ex(Re_g, Re_b)
print("P − [E(x)/R^f + Cov(m,x)] =", sp.simplify(Pr - (Ex(xg, xb) / Rf_s + cov_mx)))
print("E[R^e] + R^f Cov(m, R^e) =", sp.simplify(Ex(Re_g, Re_b) + Rf_s * cov_mRe))
assert sp.simplify(Pr - (Ex(xg, xb) / Rf_s + cov_mx)) == 0
assert sp.simplify(Ex(Re_g, Re_b) + Rf_s * cov_mRe) == 0

# 复算上表：行是资产，列是状态
pay = np.array([[1.20, 0.90],                      # 股票型资产
                [1.00, 1.10],                      # 保险型资产
                [1.00, 1.00]])                     # 无风险债券；pay: (3, 2)
P_tab = pay @ (prob * m)                           # (3, 2) @ (2,) -> (3,)：P = E[m x]
E_x = pay @ prob                                   # (3,)：E[x]
cov_tab = (pay * m) @ prob - E_m * E_x             # (3,)：Cov(m, x) = E[m x] − E[m]E[x]
R_tab = pay / P_tab[:, None]                       # 广播 (3, 2) / (3, 1) -> (3, 2)：各状态的总收益
E_R = R_tab @ prob                                 # (3,)：E[R]
cov_mR = (R_tab * m) @ prob - E_m * E_R            # (3,)：Cov(m, R) = Cov(m, R^e)
print("不考虑风险的价值", (E_x / R_f).round(4), "\nCov(m, x)", cov_tab.round(4), "\n价格", P_tab.round(4))
print("期望收益率", (E_R - 1).round(4), "\n风险溢价", (E_R - R_f).round(4))
assert np.allclose(P_tab, E_x / R_f + cov_tab)                     # 式 eq-ch1-cov-price：P = E[x]/R^f + Cov(m, x)
assert np.allclose(E_R - R_f, -R_f * cov_mR)                       # 式 eq-ch1-premium：E[R^e] = −R^f Cov(m, R^e)
assert np.allclose(E_x / R_f, [1.0144, 1.0144, 0.9661], atol=1e-4)
assert np.allclose(cov_tab, [-0.0144, 0.0048, 0.0], atol=1e-4) and np.allclose(P_tab, [1.0, 1.0192, 0.9661], atol=1e-4)
assert np.allclose(E_R - 1, [0.0500, 0.0302, 0.0351], atol=1e-4) and np.allclose(E_R - R_f, [0.0149, -0.0049, 0], atol=1e-4)
```

(ch1-unpack-states)=
**拆开讲向量写法 $\mathbf P = \mathbf X\mathbf q$。** 上面代码中的 `P_tab = pay @ (prob * m)` 一次给三项资产定价。写成公式，是把三项资产的价格排成列向量 $\mathbf P$（[记号表](../notation.md)中多项资产的价格向量），把两个状态的状态价格排成列向量 $\mathbf q$，把支付排成矩阵 $\mathbf X$。写出分量：

```{math}
:enumerated: false
\underbrace{\mathbf q}_{2\times 1} = \begin{pmatrix} q_{\text{好}} \\ q_{\text{坏}} \end{pmatrix} = \begin{pmatrix} \pi_{\text{好}}\, m_{\text{好}} \\ \pi_{\text{坏}}\, m_{\text{坏}} \end{pmatrix} \approx \begin{pmatrix} 0.435 \\ 0.531 \end{pmatrix} , \qquad
\underbrace{\mathbf X}_{3\times 2} = \begin{pmatrix} x_{1,\text{好}} & x_{1,\text{坏}} \\ x_{2,\text{好}} & x_{2,\text{坏}} \\ x_{3,\text{好}} & x_{3,\text{坏}} \end{pmatrix} = \begin{pmatrix} 1.20 & 0.90 \\ 1.00 & 1.10 \\ 1 & 1 \end{pmatrix} .
```

- $q_s = \pi_s m_s$：只在状态 $s$ 支付 1 元的证券（阿罗—德布鲁证券，Arrow–Debreu security）今天的价格，单位是“今天的元/状态 $s$ 下的元”；$q_{\text{坏}} > q_{\text{好}}$，坏状态的 1 元更贵
- $\mathbf X$ 的第 $i$ 行：资产 $i$ 在两个状态下的支付（元），三行依次为股票型资产、保险型资产与无风险债券
- $\mathbf X$ 的第 $s$ 列：全部资产在状态 $s$ 下的支付；$(i, s)$ 元素 $x_{i,s}$ 是资产 $i$ 在状态 $s$ 下的支付

这里的 $\mathbf X$ 是支付矩阵，与回归中的设计矩阵用同一个字母，含义不同。**维度**：$\mathbf X$ 是 $3\times 2$（资产数 × 状态数），$\mathbf q$ 是 $2\times 1$，乘积 $\mathbf P = \mathbf X\mathbf q$ 是 $3\times 1$，第 $i$ 个元素是资产 $i$ 的价格；转置 $\mathbf X'$ 是 $2\times 3$，它的第 $s$ 行是状态 $s$ 下三项资产的支付。**展开运算**：

```{math}
:enumerated: false
\underbrace{\mathbf P}_{3\times 1} = \begin{pmatrix} x_{1,\text{好}}\, q_{\text{好}} + x_{1,\text{坏}}\, q_{\text{坏}} \\ x_{2,\text{好}}\, q_{\text{好}} + x_{2,\text{坏}}\, q_{\text{坏}} \\ x_{3,\text{好}}\, q_{\text{好}} + x_{3,\text{坏}}\, q_{\text{坏}} \end{pmatrix} \approx \begin{pmatrix} 1.20 \times 0.435 + 0.90 \times 0.531 \\ 1.00 \times 0.435 + 1.10 \times 0.531 \\ 0.435 + 0.531 \end{pmatrix} \approx \begin{pmatrix} 1.0000 \\ 1.0192 \\ 0.9661 \end{pmatrix} ,
```

每一行都是“各状态的支付乘以该状态的价格，再加总”，也就是这项资产的 $\E[mx]$，与上表的价格一列相同。**特例**：只有一项资产时 $\mathbf X$ 只有一行，$\mathbf P = \mathbf X\mathbf q$ 退化为标量的式 {eq}`eq-ch1-pricing`；无风险债券一行全是 1，价格等于 $q_{\text{好}} + q_{\text{坏}} = \E[m] = 1/R^f$，即式 {eq}`eq-ch1-rf`；有 $S$ 个状态、$N$ 项资产时，$\mathbf X$ 是 $N\times S$，$\mathbf q$ 是 $S\times 1$。**数据矩阵**：$\mathbf X$ 按“资产在行、状态在列”排列；**所有资产共用同一个状态价格向量 $\mathbf q$**（同一个投资者、同一个 $m$），每项资产只有自己的一行支付。代码中 `pay` 就是 $\mathbf X$，形状为 `(3, 2)`；`prob * m` 是 $\mathbf q$，形状为 `(2,)`；`P_tab` 是 $\mathbf P$，形状为 `(3,)`。总收益数组 `R_tab` 的 $(i, s)$ 元素是 $x_{i,s}/P_i$，即 $\mathbf X$ 的第 $i$ 行除以 $P_i$，这正是上面说明过的广播。

```{code-cell} python
# 拆开讲 P = X q：pay 为 (3, 2) 的支付矩阵 X，q = (π_s m_s) 为 (2,) 的状态价格向量
q_vec = prob * m                                          # q: (2,)
P_vec = pay @ q_vec                                       # (3, 2) @ (2,) -> (3,)
P_rows = pay[:, 0] * q_vec[0] + pay[:, 1] * q_vec[1]      # 逐行展开：x_{i,好} q_好 + x_{i,坏} q_坏
print("q =", q_vec.round(4), "；P = X q =", P_vec.round(4))
assert np.allclose(P_vec, P_rows) and np.allclose(P_vec, P_tab)
assert np.isclose(P_vec[2], q_vec.sum()) and np.isclose(P_vec[2], 1 / R_f)     # 债券：P = q_好 + q_坏 = 1/R^f
assert np.allclose(m - E_m, [-0.096, 0.096], atol=5e-4) and np.isclose(cov_mR[1], 0.0047, atol=5e-5)
```
