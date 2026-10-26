---
kernelspec:
  name: python3
  display_name: Python 3
numbering:
  enumerator: '0.%s'
  exercise:
    template: 习题 %s
---

# 第 0 章　如何使用本书；记号约定；模拟市场生成器入门

## 学习目标

读完本章，读者应当能够：

1. 安装本书的运行环境，构建全书，并知道构建失败时从哪里查起；
2. 理解全书的记号与时点约定，特别是横截面面板中“特征比收益早一期”的含义；
3. 用 `apsim.simulate_market` 生成一个模拟市场，读懂返回结果中的每一列，并说出它对应模型中的哪个变量；
4. 写出模拟市场的全部模型方程，证明其中的随机贴现因子对因子定价、对股票的定价误差恰好是 $\alpha_{i,t}$；
5. 掌握“已知真值”的检验思路：先在模拟数据上确认一个方法能恢复真值，再把它用到真实数据上。

## 动机

实证资产定价研究中，最常见、也最难被审稿人发现的错误大致有两类。

第一类是**时点错位**。一个特征在第 $t$ 期末才能观测到，却被用来解释第 $t$ 期的收益；或者财务报表在会计期末之后几个月才披露，研究者却按会计期末对齐。这类错误不会让代码报错，往往还会让结果“更好看”。

第二类是**对估计量性质的误解**。把样本均值当成总体期望，把一次回归的系数当成真值，把在样本内构造出来、因而必然成立的等式当成对理论的检验。这类错误同样不会报错。

这两类错误都有一个共同点：只看真实数据，无法判断结果是对是错，因为真实数据的“正确答案”是未知的。模拟是检查它们最便宜的办法——我们自己设定真实的数据生成过程，知道每个参数的真值，于是任何一个估计步骤、任何一行对齐代码，都可以拿来与真值对照。本章介绍全书使用的模拟市场生成器 `apsim`，写出它背后的模型，并证明这个模型中的随机贴现因子具有我们需要的定价性质。之后的每一章，都会用这个生成器检验该章的方法。

## 0.1 环境与构建

本书的源文件是 MyST Markdown（`.md` 文件，其中的 `code-cell` 是可执行代码），由 Jupyter Book 2 构建为静态网页。构建时会执行全部代码，所以网页上看到的每一个数字与图形，都是当次构建实际算出来的。

需要的工具有三个：

- **uv**：管理 Python 版本与依赖。项目根目录下的 `pyproject.toml` 与 `uv.lock` 固定了全部依赖的版本。
- **Python 3.13**：由 uv 自动安装，不需要手动配置。
- **Node.js**：Jupyter Book 2 的构建工具链运行在 Node.js 上，需要预先安装（本项目在 v24 下测试；最低版本要求以 Jupyter Book 官方文档为准）。

在项目根目录下依次运行：

```bash
uv sync               # 创建 .venv 并安装依赖（含 jupyter-book 与 ipykernel）
uv run pytest         # 运行 apsim 的单元测试，确认环境可用
sh scripts/build.sh   # 构建全书：执行全部代码，输出到 book/_build/html
sh scripts/serve.sh   # 在 http://localhost:8000 提供网页
```

写作或修改时，可以在 `book/` 目录下运行 `uv run jupyter book start --execute` 获得实时预览。

常见问题：

1. **找不到 `node` 或构建工具报 Node.js 相关错误。** 先在终端运行 `node --version` 确认 Node.js 已安装且在 `PATH` 中。
2. **执行代码时找不到内核或找不到 `apsim`。** 每章开头的元数据声明了内核 `python3`。务必通过 `uv run` 启动构建（`scripts/build.sh` 已经这样做），这样使用的是项目虚拟环境中的 `ipykernel`，而 `apsim` 以可编辑方式安装在同一环境中。
3. **修改代码后网页上的结果没有变。** 构建会缓存代码单元的输出，只有代码单元本身改动时才重新执行。如果改的是 `src/apsim` 中的源码，而章节中的代码单元没有变，缓存的输出就会过期。此时删除 `book/_build/execute` 目录再构建即可（不要删除整个 `book/_build`，其中还缓存了网页主题模板）。
4. **图中中文显示为方框。** `apsim.plotting.setup()` 会在系统中查找可用的中文字体（Windows 下为微软雅黑，macOS 下为苹方等），找不到时给出警告。安装任一候选字体后重新构建即可。

下面的代码单元确认环境可用。

```{code-cell} python
import sys

import numpy as np
import pandas as pd

import apsim
from apsim import MarketParams, simulate_market, plotting

print("Python", sys.version.split()[0])
print("numpy", np.__version__, "| pandas", pd.__version__, "| apsim", apsim.__version__)
print("中文字体：", plotting.setup())
```

## 0.2 记号与时点约定

全书记号集中列在[记号表](../notation.md)中。这里只强调最容易出错的一条：**时点约定**。

我们用下标 $t$ 表示“在第 $t$ 期末已知”。收益 $r_{i,t}$ 是从第 $t-1$ 期末持有到第 $t$ 期末实现的超额收益；特征 $z_{i,t-1}$ 是第 $t-1$ 期末、也就是第 $t$ 期开始时已经知道的变量。一个研究者在第 $t-1$ 期末能够据以形成组合或做出预测的，只有 $z_{i,t-1}$ 以及更早的信息。用时间轴表示：

```text
      A                                         B
      |<------------- holding period t ------------->|
   end of t-1                                   end of t

  A（第 t-1 期末）：已知 z_{i,t-1}，因而已知 β_{i,t} = Γ'x_{i,t-1} 与 α_{i,t} = δ'z_{i,t-1}；
                  据此形成组合或做出预测。
  B（第 t 期末）  ：实现 f_t 与 ε_{i,t}，从而实现 r_{i,t} 与 m_t；
                  同时特征更新为 z_{i,t}，供第 t+1 期使用。
```

注意载荷 $\beta_{i,t}$ 与定价误差 $\alpha_{i,t}$ 虽然带下标 $t$（它们作用于第 $t$ 期的收益），却在第 $t-1$ 期末就已确定。这是条件资产定价文献的通行写法：下标标明“作用于哪一期”，可测性（在哪个时点已知）要另行说明。

`apsim` 的面板严格遵守这一约定：面板中索引为 `(t, i)` 的一行，特征列 `z1`……`zL` 存放的是 $z_{i,t-1}$，`exret` 存放的是 $r_{i,t}$。因此，用同一行的特征去解释同一行的收益，在时点上是正确的，不需要再做移位。还有两个容易绊倒人的细节：`period` 从 1 开始编号（$t = 1, \dots, T$），而 `stock` 从 0 开始编号（$i = 0, \dots, N-1$，与 Python 的习惯一致）。

```{code-cell} python
demo = simulate_market(MarketParams(n_stocks=4, n_periods=3), seed=0)
demo.panel.round(4)
```

理论章节中常用的写法是 $p_t = \E_t[m_{t+1}x_{t+1}]$，即从 $t$ 到 $t+1$；面板与模拟部分则写成从 $t-1$ 到 $t$，例如 $\E_{t-1}[m_t r_{i,t}]$。两者只差一个时间平移。

## 0.3 模拟市场的模型

本节逐条写出 `apsim.simulate_market` 实现的模型。收益均为月度超额收益。

**因子。** $K$ 个因子的超额收益独立同分布：

$$
f_t = \mu + \Sigma_f^{1/2} u_t, \qquad u_t \overset{\text{iid}}{\sim} N(0, I_K), \qquad \Sigma_f = \operatorname{diag}(\sigma_1^2, \dots, \sigma_K^2).
$$ (eq-ch0-factor)

因此 $\E[f_t] = \mu$，$\Var(f_t) = \Sigma_f$。模拟器中各因子相互独立，$\Sigma_f$ 是对角矩阵；后文的理论结论只用到 $\Sigma_f$ 可逆，不依赖对角性。我们把因子理解为可交易的多空组合的超额收益，所以它们本身也是需要被定价的资产。

**特征。** 每只股票有 $L$ 个特征，各自服从平稳的一阶自回归：

$$
z_{i,t} = \rho\, z_{i,t-1} + \sqrt{1-\rho^2}\; e_{i,t}, \qquad e_{i,t} \overset{\text{iid}}{\sim} N(0, I_L), \qquad z_{i,0} \sim N(0, I_L).
$$ (eq-ch0-char)

系数 $\sqrt{1-\rho^2}$ 的作用是让平稳分布恰好为 $N(0, I_L)$：若 $z_{i,t-1} \sim N(0, I_L)$，则 $\Var(z_{i,t}) = \rho^2 I_L + (1-\rho^2) I_L = I_L$。由于初值 $z_{i,0}$ 直接从平稳分布抽取，每一期的特征都服从 $N(0, I_L)$，不存在需要丢弃的“预热期”。特征的 $k$ 阶自相关为 $\rho^k$。

**因子载荷。** 载荷是期初特征的线性函数：

$$
\beta_{i,t} = \Gamma' x_{i,t-1}, \qquad x_{i,t-1} = (1, z_{i,t-1}')'.
$$ (eq-ch0-beta)

$\Gamma$ 是 $(L+1)\times K$ 矩阵：第 0 行是所有股票共同的载荷，第 $j$ 行刻画特征 $j$ 每增加一个标准差，载荷变化多少。这与 Kelly、Pruitt 与 Su 的 IPCA 模型对载荷的设定一致 [@kelly2019characteristics]。

**定价误差。** 定价误差也是期初特征的线性函数：

$$
\alpha_{i,t} = \delta' z_{i,t-1}.
$$ (eq-ch0-alpha)

默认 $\delta = 0$，即市场中没有错误定价。设 $\delta \neq 0$ 时，特征除了通过载荷影响收益外，还带来一部分与风险无关的收益。

**超额收益。**

$$
r_{i,t} = \alpha_{i,t} + \beta_{i,t}' f_t + \varepsilon_{i,t}, \qquad \varepsilon_{i,t} \overset{\text{iid}}{\sim} N(0, s^2).
$$ (eq-ch0-ret)

**随机贴现因子。**

$$
m_t = 1 - b'(f_t - \mu), \qquad b = \Sigma_f^{-1}\mu.
$$ (eq-ch0-sdf)

在模拟器中 $\Sigma_f$ 为对角矩阵，于是 $b$ 的第 $k$ 个分量化为 $b_k = \mu_k / \sigma_k^2$，这正是 `SimulatedMarket.sdf_weights` 的计算方式。读者在后续章节中改用相关因子时，必须回到一般公式 $b = \Sigma_f^{-1}\mu$。

最后是全部随机性的来源。

:::{prf:assumption} 独立性
:label: asm-ch0-indep
冲击 $\{u_t\}$、$\{e_{i,t}\}$、$\{\varepsilon_{i,t}\}$ 与初值 $\{z_{i,0}\}$ 相互独立，且各自在 $t$ 与 $i$ 上独立同分布。
:::

[假设 %s](#asm-ch0-indep) 有两个直接推论，证明中会反复用到：

- $z_{i,t-1}$ 只是 $z_{i,0}$ 与 $e_{i,1}, \dots, e_{i,t-1}$ 的函数，所以 $(f_t, \varepsilon_{i,t})$ 与 $z_{i,t-1}$ 独立，且 $f_t$ 与 $\varepsilon_{i,t}$ 也相互独立；
- 更一般地，$(f_t, \varepsilon_{i,t})$ 与第 $t-1$ 期末之前的全部信息（所有特征、所有过去的因子与收益）独立。

下面的代码单元打印默认参数与默认载荷矩阵，并逐列核对面板确实由式 {eq}`eq-ch0-beta`～{eq}`eq-ch0-ret` 生成。

```{code-cell} python
params = MarketParams()
mkt = simulate_market(params, seed=0)
print(params)
print("默认 Γ（行：常数项、z1……z5；列：f1, f2, f3）：")
print(mkt.gamma)
print("b = Σ_f^{-1} μ =", mkt.sdf_weights.round(4))
```

默认的 $\Gamma$ 含义很直观：所有股票对第一个因子的载荷都是 1（类似市场因子）；第一个特征决定对第二个因子的载荷，$\beta_{2,i,t} = 0.5\, z_{1,i,t-1}$；第二个特征决定对第三个因子的载荷，$\beta_{3,i,t} = 0.5\, z_{2,i,t-1}$；其余三个特征与载荷无关，是“噪声特征”。真实研究中候选特征往往成百上千，其中多数与风险无关，第四部分的方法必须学会区分它们。

```{code-cell} python
panel = mkt.panel
L, K = params.n_chars, params.n_factors
z = panel[[f"z{j}" for j in range(1, L + 1)]].to_numpy()
x = np.column_stack([np.ones(len(panel)), z])
beta = panel[[f"beta{k}" for k in range(1, K + 1)]].to_numpy()
f_row = mkt.factors.reindex(panel.index.get_level_values("period")).to_numpy()

# 载荷与定价误差是期初特征的确定函数：β = Γ'x，α = δ'z
assert np.allclose(beta, x @ mkt.gamma)
assert np.allclose(panel["alpha"], z @ mkt.delta)

# 收益减去 α 与 β'f 后，剩下的应是标准差为 s 的白噪声
eps = panel["exret"].to_numpy() - panel["alpha"].to_numpy() - np.sum(beta * f_row, axis=1)
print(f"残差标准差 {eps.std():.4f}（真值 s = {params.idio_vol}），残差均值 {eps.mean():.5f}")

# 特征的平稳分布为 N(0, I)，一阶自相关为 ρ
z1 = panel["z1"].unstack("stock").to_numpy()
rho_hat = np.corrcoef(z1[1:].ravel(), z1[:-1].ravel())[0, 1]
print(f"z1 的均值 {z1.mean():.4f}、方差 {z1.var():.4f}、一阶自相关 {rho_hat:.4f}（真值 0、1、{params.char_persistence}）")
print(f"单只股票月度超额收益的标准差 {panel['exret'].std():.4f}")
```

单只股票的月度收益波动中，异质性冲击 $\varepsilon_{i,t}$ 占了绝大部分。这意味着用单只股票的时间序列去估计它的载荷或定价误差，噪声会非常大；后面的估计几乎都要借助横截面上的大量股票。

:::{note} 为什么是 $\E[m_t] = 1$
本书的模拟市场只生成超额收益。超额收益的定价方程 $0 = \E[m_t r_{i,t}]$ 对 $m_t$ 乘以任何正常数都成立，所以 SDF 的均值在只有超额收益时无法识别，可以自由归一化。我们取 $\E[m_t] = 1$，相当于令无风险总收益 $R^f = 1/\E[m_t] = 1$。需要无风险利率的场合（第 1 章起），再显式引入 $R^f_t$。
:::

## 0.4 模拟市场中 SDF 的定价性质

下面的定理说明，式 {eq}`eq-ch0-sdf` 给出的 $m_t$ 确实是这个市场的随机贴现因子：它正确地为因子定价，而它为股票定价时留下的误差恰好就是 $\alpha_{i,t}$。这是全书用模拟数据检验方法的基础：只要知道真实的 $m_t$，就知道每只股票在每一期“应有的”定价误差。

:::{prf:theorem} 模拟市场 SDF 的定价性质
:label: thm-ch0-sdf
在式 {eq}`eq-ch0-factor`～{eq}`eq-ch0-sdf` 与[假设 %s](#asm-ch0-indep) 下，设 $\Sigma_f$ 可逆，$m_t = 1 - b'(f_t - \mu)$，$b = \Sigma_f^{-1}\mu$。则对任意 $t$ 与 $i$：

1. $\E[m_t] = 1$；
2. $\E[m_t f_t] = 0$，即 $m_t$ 为全部 $K$ 个因子正确定价；
3. $\E[m_t r_{i,t} \mid z_{i,t-1}] = \alpha_{i,t} = \delta' z_{i,t-1}$。特别地，当 $\delta = 0$ 时，$\E[m_t r_{i,t} \mid z_{i,t-1}] = 0$，即 $m_t$ 在条件意义下为每一只股票正确定价。
:::

:::{prf:proof}
**第 1 条。** $m_t$ 是 $f_t$ 的线性函数，取期望并利用 $\E[f_t] = \mu$：

```{math}
:enumerated: false
\E[m_t] = 1 - b'\left(\E[f_t] - \mu\right) = 1 - b' \cdot 0 = 1.
```

**第 2 条。** 展开 $m_t f_t$：

```{math}
:enumerated: false
\E[m_t f_t] = \E[f_t] - \E\left[f_t (f_t - \mu)'\right] b .
```

第二项中，把 $f_t$ 写成 $(f_t - \mu) + \mu$：

```{math}
:enumerated: false
\E\left[f_t (f_t - \mu)'\right] = \E\left[(f_t - \mu)(f_t - \mu)'\right] + \mu\, \E\left[(f_t - \mu)'\right] = \Sigma_f + 0 = \Sigma_f .
```

代回并利用 $b = \Sigma_f^{-1}\mu$：

```{math}
:enumerated: false
\E[m_t f_t] = \mu - \Sigma_f \Sigma_f^{-1} \mu = 0 .
```

这一步只用到 $f_t$ 的一、二阶矩，既不需要正态性，也不需要 $\Sigma_f$ 是对角矩阵。反过来，若要求 $1 - b'(f_t-\mu)$ 为因子定价，由上式 $\mu - \Sigma_f b = 0$，$b$ 必须等于 $\Sigma_f^{-1}\mu$，所以在这一函数形式下 $b$ 是唯一的。

**第 3 条。** 把式 {eq}`eq-ch0-ret` 代入：

```{math}
:enumerated: false
\E[m_t r_{i,t} \mid z_{i,t-1}]
= \E[m_t \alpha_{i,t} \mid z_{i,t-1}] + \E[m_t \beta_{i,t}' f_t \mid z_{i,t-1}] + \E[m_t \varepsilon_{i,t} \mid z_{i,t-1}] .
```

由式 {eq}`eq-ch0-beta` 与 {eq}`eq-ch0-alpha`，$\alpha_{i,t}$ 与 $\beta_{i,t}$ 都是 $z_{i,t-1}$ 的确定函数，可以提到条件期望之外：

```{math}
:enumerated: false
\E[m_t r_{i,t} \mid z_{i,t-1}]
= \alpha_{i,t}\, \E[m_t \mid z_{i,t-1}] + \beta_{i,t}'\, \E[m_t f_t \mid z_{i,t-1}] + \E[m_t \varepsilon_{i,t} \mid z_{i,t-1}] .
```

逐项处理：

- $m_t$ 与 $m_t f_t$ 都只是 $f_t$ 的函数。由[假设 %s](#asm-ch0-indep)，$f_t$ 与 $z_{i,t-1}$ 独立，所以条件期望等于无条件期望：$\E[m_t \mid z_{i,t-1}] = \E[m_t] = 1$，$\E[m_t f_t \mid z_{i,t-1}] = \E[m_t f_t] = 0$（第 1、2 条）。
- $m_t \varepsilon_{i,t}$ 是 $(f_t, \varepsilon_{i,t})$ 的函数，而 $(f_t, \varepsilon_{i,t})$ 与 $z_{i,t-1}$ 独立，所以 $\E[m_t \varepsilon_{i,t} \mid z_{i,t-1}] = \E[m_t \varepsilon_{i,t}]$；再由 $f_t$ 与 $\varepsilon_{i,t}$ 相互独立，$\E[m_t \varepsilon_{i,t}] = \E[m_t]\, \E[\varepsilon_{i,t}] = 1 \cdot 0 = 0$。

三项合起来：

```{math}
:enumerated: false
\E[m_t r_{i,t} \mid z_{i,t-1}] = \alpha_{i,t} \cdot 1 + \beta_{i,t}' \cdot 0 + 0 = \alpha_{i,t} = \delta' z_{i,t-1} .
```

当 $\delta = 0$ 时右端为 0。
:::

证明的第 3 条中，独立性被用了两次，二者缺一不可。如果因子收益的分布依赖于特征（例如特征能预测因子收益），那么 $\E[m_t \mid z_{i,t-1}]$ 一般不再等于 1，$\E[m_t f_t \mid z_{i,t-1}]$ 也不再为 0，一个只在无条件意义下为因子定价的 $m_t$，未必在条件意义下定价。这正是第 5 章条件模型要处理的问题。由于 $(f_t, \varepsilon_{i,t})$ 与第 $t-1$ 期末的全部信息独立，同样的论证也给出 $\E_{t-1}[m_t r_{i,t}] = \alpha_{i,t}$。

由定理的第 3 条，再用一次迭代期望定律，可以得到两个在检验中很有用的无条件矩。

:::{prf:corollary} 无条件定价误差与管理组合
:label: cor-ch0-managed
在[定理 %s](#thm-ch0-sdf) 的条件下：

$$
\E[m_t r_{i,t}] = \E[\alpha_{i,t}] = 0, \qquad \E[m_t r_{i,t}\, z_{i,t-1}] = \delta .
$$ (eq-ch0-managed)
:::

:::{prf:proof}
由迭代期望定律与定理第 3 条，$\E[m_t r_{i,t}] = \E\left[\E[m_t r_{i,t} \mid z_{i,t-1}]\right] = \delta' \E[z_{i,t-1}] = 0$。同理，由于 $z_{i,t-1}$ 在以自身为条件时可以提出，

```{math}
:enumerated: false
\E[m_t r_{i,t}\, z_{i,t-1}] = \E\left[z_{i,t-1}\, \E[m_t r_{i,t} \mid z_{i,t-1}]\right] = \E[z_{i,t-1} z_{i,t-1}']\, \delta = I_L\, \delta = \delta ,
```

最后一步用到 $z_{i,t-1} \sim N(0, I_L)$。
:::

这个推论有一个值得记住的含义：即使存在错误定价（$\delta \neq 0$），每只股票的**无条件**定价误差仍然为 0，因为特征围绕 0 波动，错误定价时正时负、平均抵消。只有把收益按特征加权，即构造“管理组合”（managed portfolio）$r_{i,t}\, z_{i,t-1}$，错误定价才在无条件矩中显现出来。用无条件矩检验条件模型时，检验资产的选择决定了检验能看到什么。

最后，$m_t$ 的波动也有明确的含义：$\Var(m_t) = b'\Sigma_f b = \mu'\Sigma_f^{-1}\mu$，它等于用这 $K$ 个因子所能构成的组合的最大夏普比率的平方。Hansen–Jagannathan 界 [@hansen1991implications] 指出，任何为这些因子正确定价、且均值为 1 的 SDF，其标准差都不小于这一最大夏普比率；本章的 $m_t$ 是因子的线性函数，恰好达到这个下界。第 3 章将详细讨论。线性 SDF 还有一个缺点：$f_t$ 服从正态分布，$m_t$ 以正的概率取负值，而负的 SDF 意味着存在套利机会（第 2 章）。在默认参数下这个概率可以忽略，下一节会算出它。

## 0.5 代码验证

### 符号验证

先用 `sympy` 在单因子情形下验证[定理 %s](#thm-ch0-sdf) 的前两条：设 $f = \mu + \sigma u$，$u \sim N(0,1)$，$b = \mu/\sigma^2$。

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

再验证第 2 条不依赖于 $\Sigma_f$ 为对角矩阵。证明中真正用到的只是 $\E[f(f-\mu)'] = \Sigma_f$ 与 $b = \Sigma_f^{-1}\mu$，这是纯代数，取 $K=2$ 与一般的对称协方差矩阵验证：

```{code-cell} python
m1, m2 = sp.symbols("mu1 mu2", real=True)
s11, s22 = sp.symbols("s11 s22", positive=True)
s12 = sp.Symbol("s12", real=True)
mu_v = sp.Matrix([m1, m2])
Sigma = sp.Matrix([[s11, s12], [s12, s22]])

b_v = Sigma.inv() * mu_v
E_mf_v = mu_v - Sigma * b_v                 # E[m f] = μ − E[f (f−μ)'] b = μ − Σ_f b
var_m = sp.simplify((b_v.T * Sigma * b_v)[0])
print("E[m f] =", sp.simplify(E_mf_v).T)
print("Var(m) − μ'Σ⁻¹μ =", sp.simplify(var_m - (mu_v.T * Sigma.inv() * mu_v)[0]))
assert sp.simplify(E_mf_v) == sp.zeros(2, 1)
```

### 模拟验证

第 3 条涉及条件期望，用模拟数据验证。我们取 $\delta = (0.01, 0, 0, 0, 0)'$，即第一个特征每高一个标准差，每月有 1% 的错误定价；用 $T = 20000$ 期、$N = 20$ 只股票的长样本，使样本矩足够接近总体矩。验证内容是：$\E[m_t] = 1$、$\E[m_t f_t] = 0$、无条件 $\E[m_t r_{i,t}] = 0$、管理组合 $\E[m_t r_{i,t} z_{1,i,t-1}] = \delta_1$，以及条件版本——把观测按 $z_{1,i,t-1}$ 分成五组，组内 $m_t r_{i,t}$ 的均值应当等于组内 $\alpha_{i,t}$ 的均值。

```{code-cell} python
long_params = MarketParams(n_stocks=20, n_periods=20_000, alpha_delta=(0.01, 0.0, 0.0, 0.0, 0.0))
long_mkt = simulate_market(long_params, seed=2026)

m = long_mkt.sdf().to_numpy()                       # (T,)
F = long_mkt.factors.to_numpy()                     # (T, K)
R = long_mkt.panel["exret"].unstack("stock").to_numpy()   # (T, N)
Z1 = long_mkt.panel["z1"].unstack("stock").to_numpy()
A = long_mkt.panel["alpha"].unstack("stock").to_numpy()
MR = m[:, None] * R                                 # m_t r_{i,t}

rows = [
    ("E[m]", m.mean(), 1.0),
    *[(f"E[m f{k + 1}]", v, 0.0) for k, v in enumerate((m[:, None] * F).mean(axis=0))],
    ("E[m r]", MR.mean(), 0.0),
    ("E[m r z1]", (MR * Z1).mean(), long_params.alpha_delta[0]),
    ("Std(m)", m.std(), np.sqrt(np.sum(long_mkt.sdf_weights * np.asarray(long_params.factor_mean)))),
]
pd.DataFrame(rows, columns=["矩", "样本值", "理论值"]).set_index("矩").round(5)
```

表中 Std(m) 的理论值按 $\sqrt{\mu'\Sigma_f^{-1}\mu} = \sqrt{\sum_k b_k \mu_k}$ 计算。条件版本按 $z_1$ 的五分位分组比较：

```{code-cell} python
groups = pd.qcut(Z1.ravel(), 5, labels=["最低", "2", "3", "4", "最高"])
cond = pd.DataFrame({"m r": MR.ravel(), "alpha": A.ravel(), "组": groups}).groupby("组", observed=True).mean()
cond["差"] = cond["m r"] - cond["alpha"]
print(cond.round(5))
assert np.all(np.abs(cond["差"]) < 0.002)
```

五个组中，$m_t r_{i,t}$ 的组内均值与 $\alpha_{i,t}$ 的组内均值相差都在千分之二以内，而组间 $\alpha$ 的差别达到 $\pm 1.4\%$ 左右：条件定价误差由 $\delta' z_{i,t-1}$ 决定，这正是定理第 3 条。最后，用默认参数核对 $m_t$ 的标准差等于因子的最大夏普比率（与随机抽取的 20 万个因子组合比较），并算出 $m_t$ 取负值的概率。由于 $m_t \sim N(1, \mu'\Sigma_f^{-1}\mu)$，

```{code-cell} python
from scipy.stats import norm

mu_d = np.asarray(params.factor_mean)
Sigma_d = np.diag(np.asarray(params.factor_vol) ** 2)
sd_m = np.sqrt(mkt.sdf_weights @ mu_d)

w = np.random.default_rng(3).standard_normal((200_000, len(mu_d)))      # 随机组合权重
sharpe = (w @ mu_d) / np.sqrt(np.einsum("nk,kl,nl->n", w, Sigma_d, w))
w_tan = np.linalg.solve(Sigma_d, mu_d)                                    # 切点组合 ∝ Σ_f⁻¹μ
sr_tan = (w_tan @ mu_d) / np.sqrt(w_tan @ Sigma_d @ w_tan)
print(f"Std(m) = {sd_m:.4f}，切点组合夏普比率 {sr_tan:.4f}，随机组合中的最大夏普比率 {sharpe.max():.4f}")
print(f"P(m < 0) = Φ(−1/Std(m)) = {norm.cdf(-1 / sd_m):.2e}")
assert np.isclose(sd_m, sr_tan) and sharpe.max() <= sr_tan + 1e-12
```

### 收敛到真值

样本矩只是总体矩的估计。下图展示随着样本期数 $T$ 增大，样本矩如何收敛到理论值。左图是 $m_t f_{1,t}$ 的样本均值（理论值 0），右图是管理组合矩 $m_t r_{i,t} z_{1,i,t-1}$ 的样本均值（理论值 $\delta_1 = 0.01$）。灰色区域是理论值加减两倍标准误，标准误按 $\text{Std}/\sqrt{T}$ 计算；右图先对每一期求 $N$ 只股票的横截面平均，再用这些平均值的标准差除以 $\sqrt{T}$，这样同期股票之间通过因子产生的相关会自动计入（[习题 %s](#ex-ch0-delta) 中会看到，忽略这种相关会严重低估标准误）。

```{code-cell} python
#| label: fig-ch0-convergence
#| caption: 样本矩随样本期数 T 收敛到理论值（对数横轴）
import matplotlib.pyplot as plt

T_grid = np.unique(np.geomspace(20, len(m), 60).astype(int))
mf1 = m * F[:, 0]
mrz = (MR * Z1).mean(axis=1)                         # 每期横截面平均
run_mf1 = np.cumsum(mf1)[T_grid - 1] / T_grid
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
print(f"全样本（T = {len(m)}）的标准化偏差：左图 {(run_mf1[-1] - 0) / se_mf1[-1]:.2f}，"
      f"右图 {(run_mrz[-1] - 0.01) / se_mrz[-1]:.2f}")
```

两个样本均值都随 $T$ 增大而收敛到理论值，偏离的幅度大致按 $1/\sqrt{T}$ 收窄。左图的样本路径始终在带内；右图的路径在 $T$ 较小时越出带外，此后长时间贴着带的上沿，全样本的标准化偏差略大于 2（见图下打印的数值）。这并不说明理论有误：在理论正确时，偏离超过 2.19 倍标准误的双侧概率约为 3%，单次模拟出现并不罕见；本节开头的表格与五分位分组的结果也都与理论一致。至于路径“越出带外”：灰色带是逐点的“理论值 ± 2 倍标准误”，而同一条路径上相邻的累计均值共享绝大部分数据、高度相关，早期一旦偏高，就会持续偏高一段时间，所以一条路径在某一段越出逐点的带，比独立抽样时常见得多。由此也可以读出本书所有模拟验证的基本逻辑：**单次模拟只给出带有抽样误差的样本值，判断“是否恢复了真值”时，要把偏离与标准误相比较，而不是要求两者完全相等。**

## A 股视角

本章的模拟市场刻意保持简单，以便先把定价方程、时点约定与“已知真值”的检验逻辑讲清楚。与真实的 A 股市场相比，它至少省略了以下几类特征：

1. **涨跌停。** A 股对单日涨跌幅设有限制，不同板块与 ST 股票的限制比例不同，并随制度改革调整。涨跌停会截断日收益、可能推迟价格对信息的反映，也让触及涨跌停的股票在当日实际上难以按收盘价成交。第 15 章在 `apsim.frictions` 中加入这一机制，第 17 章讨论它对资产定价检验的影响。
2. **停牌。** 停牌期间没有成交价，复牌后的收益集中体现了停牌期间积累的信息。月度数据中如何处理停牌股票的收益与特征，会直接影响组合收益，详见第 15 章。
3. **卖空限制。** 本章把因子当作可以无成本构造的多空组合。A 股融券规模小、成本高，许多“空头腿”在现实中难以实现，这既影响因子收益的可实现性（第 26 章），也是错误定价能够持续存在的一个原因（第 8 章）。
4. **壳价值。** 在发行上市受到严格管制的时期，小市值公司作为借壳上市的潜在对象而具有额外价值，这使市值最小的股票的收益不能用一般的风险补偿解释。Liu、Stambaugh 与 Yuan 构造中国市场的规模与价值因子时，因此剔除了市值最小的 30% 的股票 [@liu2019size]。第 16、17 章讨论这一处理。
5. **财报披露滞后。** 会计期末的财务数据要到数月之后才公开（例如年报通常在次年 4 月底前披露）。如果按会计期末而不是实际披露日对齐，特征 $z_{i,t-1}$ 中就混入了第 $t-1$ 期末尚不可得的信息，即本章强调的时点错位。正式研究必须使用带有公告日期的时点数据，第 15 章给出对齐规则。

此外，模拟器中的因子收益是独立同分布的正态变量，特征服从相互独立的正态 AR(1)，异质性冲击在股票之间不相关，股票数量在样本期内不变。真实数据中的厚尾、波动聚集、行业内的同期相关、新股上市与退市，都会在相应章节逐步加入。中国股市中机器学习方法的实证表现，可先参阅 Leippold、Wang 与 Zhou 的研究 [@leippold2022machine]，第四部分会详细讨论。

## 模拟实验：用交互回归恢复 Γ

### 方法

把式 {eq}`eq-ch0-beta` 代入式 {eq}`eq-ch0-ret`，在 $\delta = 0$ 时：

$$
r_{i,t} = x_{i,t-1}' \Gamma f_t + \varepsilon_{i,t} = \sum_{j=0}^{L} \sum_{k=1}^{K} \Gamma_{j,k}\, \left(x_{j,i,t-1} f_{k,t}\right) + \varepsilon_{i,t},
$$ (eq-ch0-interact)

其中 $x_{0,i,t-1} = 1$，$x_{j,i,t-1} = z_{j,i,t-1}$（$j \geq 1$）。这是一个以 $(L+1)K$ 个“特征 × 因子”交互项为解释变量、以 $\Gamma$ 的各元素为系数的线性回归。因子可观测时，把全部 $(i,t)$ 观测混合起来做最小二乘，就能估计 $\Gamma$。由于 $\varepsilon_{i,t}$ 与 $(z_{i,t-1}, f_t)$ 独立，解释变量严格外生，最小二乘估计量无偏，其条件协方差矩阵为 $s^2 (X'X)^{-1}$，其中 $X$ 是交互项构成的设计矩阵。下面用默认参数（$N = 500$、$T = 240$）检验这一点。

```{code-cell} python
def interaction_design(market):
    """返回交互项设计矩阵 X（行：(t, i)；列：x_j f_k，j 在外层、k 在内层）与期初特征矩阵 Z。"""
    pnl = market.panel
    n_chars = market.params.n_chars
    Z = pnl[[f"z{j}" for j in range(1, n_chars + 1)]].to_numpy()
    xx = np.column_stack([np.ones(len(pnl)), Z])
    ff = market.factors.reindex(pnl.index.get_level_values("period")).to_numpy()
    X = (xx[:, :, None] * ff[:, None, :]).reshape(len(pnl), -1)
    return X, Z


def ols(X, y):
    """最小二乘系数与同方差标准误。"""
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    s2 = resid @ resid / (len(y) - X.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    return coef, se


X, _ = interaction_design(mkt)
coef, se = ols(X, mkt.panel["exret"].to_numpy())
K = params.n_factors
names = [f"Γ[{j},{k + 1}]" for j in range(params.n_chars + 1) for k in range(K)]
table = pd.DataFrame(
    {"真值": mkt.gamma.ravel(), "估计值": coef, "标准误": se, "标准化偏差": (coef - mkt.gamma.ravel()) / se},
    index=names,
)
print(table.round(4).to_string())
assert np.all(np.abs(table["标准化偏差"]) < 4)
```

表中“标准化偏差”为 $(\hat\Gamma_{j,k} - \Gamma_{j,k}) / \text{se}$。18 个系数的估计值都落在真值附近。在这个种子下，有几个标准化偏差的绝对值超过 2，多于 18 个系数中平均约 1 个的预期。这是抽样误差，还是标准误算错了？单次模拟无法回答，要靠重复模拟：如果估计量无偏、标准误正确，标准化偏差在多次重复中应当近似服从标准正态分布。

```{code-cell} python
zs = []
for r in range(30):
    rep = simulate_market(params, seed=1000 + r)
    Xr, _ = interaction_design(rep)
    cr, ser = ols(Xr, rep.panel["exret"].to_numpy())
    zs.append((cr - rep.gamma.ravel()) / ser)
zs = np.concatenate(zs)
print(f"30 次重复、共 {zs.size} 个标准化偏差：均值 {zs.mean():.3f}，标准差 {zs.std():.3f}，"
      f"绝对值超过 2 的比例 {np.mean(np.abs(zs) > 2):.3f}（标准正态分布下为 0.046）")
```

重复模拟的结果与标准正态分布相符：估计量无偏，同方差标准误也是正确的，种子 0 下偏多的大偏差只是抽样误差。另外，第一个因子一列（$k = 1$）的标准误明显小于另外两列：$\sigma_1 = 0.05$ 较大，交互项 $x_j f_{1,t}$ 的变化也更大，因而包含更多关于 $\Gamma_{j,1}$ 的信息。

### 当存在错误定价时

如果 $\delta \neq 0$，式 {eq}`eq-ch0-interact` 遗漏了 $\alpha_{i,t} = \delta' z_{i,t-1}$。遗漏项与解释变量相关吗？交互项 $z_{j,i,t-1} f_{k,t}$ 与 $z_{j,i,t-1}$ 的协方差为 $\E[z_{j,i,t-1}^2]\,\E[f_{k,t}] = \mu_k \neq 0$，所以答案是肯定的：只用交互项的回归有遗漏变量偏误。

偏误的大小可以算出来。以因子路径 $F = (f_1, \dots, f_T)'$ 为条件，令 $N \to \infty$。每一期的横截面平均满足 $N^{-1}\sum_i x_{i,t-1} x_{i,t-1}' \to I_{L+1}$、$N^{-1}\sum_i x_{i,t-1} z_{i,t-1}' \delta \to (0, \delta')'$。交互项回归的设计行可写成 Kronecker 积 $x_{i,t-1} \otimes f_t$，于是

$$
\operatorname{plim}_{N\to\infty} \operatorname{vec}(\hat\Gamma')
= \operatorname{vec}(\Gamma') + \left[I_{L+1} \otimes F'F\right]^{-1} \left[\begin{pmatrix} 0 \\ \delta \end{pmatrix} \otimes F'\iota_T\right]
= \operatorname{vec}(\Gamma') + \begin{pmatrix} 0 \\ \delta \end{pmatrix} \otimes c ,
\qquad c = (F'F)^{-1} F'\iota_T ,
$$ (eq-ch0-bias)

其中 $\iota_T$ 是 $T$ 维全 1 向量，$\operatorname{vec}(\Gamma')$ 按行把 $\Gamma$ 排成向量（与代码中 `reshape(L+1, K)` 的顺序一致）。也就是说，常数项一行没有偏误，第 $j$ 个特征一行的偏误是 $\delta_j c$；$c$ 恰好是把常数 1 对因子做无截距回归的系数。当 $T \to \infty$ 时，$F'F/T \to \Sigma_f + \mu\mu'$，$F'\iota_T/T \to \mu$，由 Sherman–Morrison 公式，

$$
c \to (\Sigma_f + \mu\mu')^{-1}\mu = \frac{\Sigma_f^{-1}\mu}{1 + \mu'\Sigma_f^{-1}\mu} = \frac{b}{1 + \mu'\Sigma_f^{-1}\mu} .
$$ (eq-ch0-bias-limit)

偏误的方向正是 SDF 系数 $b$ 的方向。解决办法很简单：在回归中加入特征本身作为解释变量，其系数估计的就是 $\delta$。IPCA 在载荷之外允许一个由特征驱动的截距，也是出于同样的考虑 [@kelly2019characteristics]。下面的代码在 $\delta = (0.01, 0, 0, 0, 0)'$ 下比较两种回归。两者使用同一份数据，二者之差几乎不含异质性冲击带来的抽样噪声，可以直接与式 {eq}`eq-ch0-bias` 的 $\delta_1 c$ 比较。

```{code-cell} python
mis_params = MarketParams(alpha_delta=(0.01, 0.0, 0.0, 0.0, 0.0))
mis = simulate_market(mis_params, seed=0)
y = mis.panel["exret"].to_numpy()
X, Z = interaction_design(mis)
coef_short, _ = ols(X, y)                                  # 只有交互项
coef_long, se_long = ols(np.column_stack([X, Z]), y)       # 交互项 + 特征本身
G_short = coef_short.reshape(-1, K)
G_long = coef_long[: X.shape[1]].reshape(-1, K)
delta_hat, delta_se = coef_long[X.shape[1]:], se_long[X.shape[1]:]

F_path = mis.factors.to_numpy()
c = np.linalg.solve(F_path.T @ F_path, F_path.sum(axis=0))
print("z1 一行：只用交互项", G_short[1].round(4), "| 加入特征", G_long[1].round(4), "| 真值", mis.gamma[1])
print("常数项一行之差      ", (G_short[0] - G_long[0]).round(4))
print("z1 一行之差         ", (G_short[1] - G_long[1]).round(4))
print("偏误公式预测 δ1·c   ", (0.01 * c).round(4))
print("δ 的估计", delta_hat.round(4), "标准误", delta_se.round(4), "真值", mis.delta)

# Sherman–Morrison：T → ∞ 时 c 的极限
mu_v, Sig = np.asarray(mis_params.factor_mean), np.diag(np.asarray(mis_params.factor_vol) ** 2)
lhs = np.linalg.solve(Sig + np.outer(mu_v, mu_v), mu_v)
rhs = mis.sdf_weights / (1 + mu_v @ mis.sdf_weights)
print("c 的极限", lhs.round(4), "=", rhs.round(4))
assert np.allclose(G_short[1] - G_long[1], 0.01 * c, atol=0.003) and np.allclose(lhs, rhs)
```

两种回归之差与式 {eq}`eq-ch0-bias` 的预测相符；有限 $N$ 下横截面平均与总体矩之间仍有小的差别，所以二者不完全相等。还要注意，本例中 $c$ 与它的极限 {eq}`eq-ch0-bias-limit` 差别很大：240 个月的样本中，因子样本均值本身就有可观的抽样误差，而偏误取决于样本中实际出现的因子路径。

## 常见误区

**误区一：把同期特征当作期初特征。** 用 $z_{i,t}$ 而不是 $z_{i,t-1}$ 去解释 $r_{i,t}$，在真实数据中可能造成严重的前视偏差：许多特征（市值、动量、换手率）与当期收益机械相关，例如第 $t$ 期末的市值本身就包含了第 $t$ 期的收益。在本章的模拟器中，特征更新的冲击 $e_{i,t}$ 与收益无关，所以错用同期特征不会产生虚假的预测力，只会削弱真实的关系：由式 {eq}`eq-ch0-char` 与 $e_{i,t}$ 的独立性，

```{math}
:enumerated: false
\E[m_t r_{i,t} z_{i,t}] = \rho\,\E[m_t r_{i,t} z_{i,t-1}] + \sqrt{1-\rho^2}\,\E[m_t r_{i,t} e_{i,t}] = \rho\,\delta + 0 .
```

这个例子说明了模拟的局限：模拟器不会自动暴露所有错误，只能暴露我们放进模型的那些机制。下面用 $\rho = 0.9$ 的长样本验证。两个样本矩都带有抽样误差，而且误差高度相关；所以除了把它们分别与理论值比较，还检验二者之差是否满足推导给出的关系：$\overline{m r z_t} - \rho\, \overline{m r z_{t-1}}$ 只剩下 $\sqrt{1-\rho^2}\, \overline{m r e}$，理论值为 0。

```{code-cell} python
Z1_now = long_mkt.panel["z1"].unstack("stock").shift(-1).to_numpy()   # 第 t 行放 z_{i,t}
valid = ~np.isnan(Z1_now[:, 0])                                        # 最后一期没有 z_{i,T}
now = (MR[valid] * Z1_now[valid]).mean()
lag = (MR[valid] * Z1[valid]).mean()
rho = long_params.char_persistence
print(f"用期初特征 z_(t-1)：{lag:.5f}（理论值 δ1 = {long_params.alpha_delta[0]}）")
print(f"用同期特征 z_t    ：{now:.5f}（理论值 ρδ1 = {rho * 0.01:.4f}）")
resid_term = MR[valid] * (Z1_now[valid] - rho * Z1[valid])
print(f"now − ρ·lag = {now - rho * lag:.5f}，标准误约 {resid_term.std() / np.sqrt(resid_term.size):.5f}（理论值 0）")
```

**误区二：把样本均值当作总体值。** 默认参数下第一个因子的月均值 $\mu_1 = 0.6\%$、波动率 $5\%$，240 个月样本均值的标准误为 $0.05/\sqrt{240} \approx 0.32\%$，样本均值完全可能偏离真值一半以上。更隐蔽的错误是用样本矩构造 SDF，再在同一样本中“验证”它。若 $\hat b = \hat\Sigma_f^{-1}\bar f$（$\hat\Sigma_f$ 为除以 $T$ 的样本协方差），$\hat m_t = 1 - \hat b'(f_t - \bar f)$，则把[定理 %s](#thm-ch0-sdf) 第 2 条证明中的总体矩逐一换成样本矩，可得样本内 $T^{-1}\sum_t \hat m_t f_t = \bar f - \hat\Sigma_f \hat b = 0$ **恒成立**，无论模型对错。这样的“检验”什么也没有检验。

```{code-cell} python
F240 = mkt.factors.to_numpy()
f_bar = F240.mean(axis=0)
Sigma_hat = np.cov(F240, rowvar=False, ddof=0)
b_hat = np.linalg.solve(Sigma_hat, f_bar)
m_hat = 1 - (F240 - f_bar) @ b_hat
print("因子样本均值", f_bar.round(4), "真值", params.factor_mean, "标准误", (np.asarray(params.factor_vol) / np.sqrt(len(F240))).round(4))
print("b 的样本估计", b_hat.round(2), "真值", mkt.sdf_weights.round(2))
print("样本内 mean(m̂ f) =", (m_hat[:, None] * F240).mean(axis=0))
print("真实 m 的 mean(m f) =", (mkt.sdf().to_numpy()[:, None] * F240).mean(axis=0).round(5))
```

样本内的 $T^{-1}\sum_t \hat m_t f_t$ 在机器精度内为 0，而真实 SDF 在同一样本中的样本定价误差并不为 0——后者才反映了抽样误差的真实大小。真正的检验需要样本外数据，或者把估计误差计入检验统计量，这是第二部分的主题。

**误区三：忽视随机种子。** `simulate_market(params, seed=None)` 每次运行给出不同的结果，书中的数字因此无法复现。反过来的错误同样常见：在蒙特卡洛循环中每次都用同一个种子，得到的“独立重复”其实完全相同，估计量的抽样方差被算成 0。

```{code-cell} python
small = MarketParams(n_stocks=50, n_periods=60)
first = simulate_market(small, seed=None).panel["exret"].iloc[0]
second = simulate_market(small, seed=None).panel["exret"].iloc[0]
print(f"seed=None 两次运行的第一个收益：{first:.4f} 与 {second:.4f}")

same = [simulate_market(small, seed=0).factors["f1"].mean() for _ in range(5)]
diff = [simulate_market(small, seed=100 + r).factors["f1"].mean() for r in range(5)]
print("同一种子重复 5 次，f1 样本均值的标准差：", np.std(same))
print("不同种子重复 5 次，f1 样本均值的标准差：", round(float(np.std(diff)), 4))
```

（第一行的两个数字每次重新执行都会变化，这正是问题所在。）本书的约定是：单次演示用固定的整数种子；蒙特卡洛重复用“基础种子 + 重复编号”，保证各次重复互不相同，而整体可以复现。

**误区四：把可预测性等同于错误定价。** 即使 $\delta = 0$，特征也能预测收益。由式 {eq}`eq-ch0-ret` 与独立性，$\E[r_{i,t} \mid z_{i,t-1}] = \alpha_{i,t} + \beta_{i,t}'\mu$，而 $\beta_{i,t}$ 随特征变化。默认参数下 $\E[r_{i,t} \mid z_{i,t-1}] = \mu_1 + 0.5\mu_2\, z_{1,i,t-1} + 0.5\mu_3\, z_{2,i,t-1}$：特征 $z_1$、$z_2$ 能预测收益，完全是因为它们决定了风险暴露，这正是“特征即协方差”的含义 [@kelly2019characteristics]。判断一个能预测收益的特征是风险补偿还是错误定价，要看它的预测力能否被载荷与因子风险溢价解释，而不能只看预测力本身（第 8、23 章）。下面用 $\delta = 0$ 的长样本做 $r_{i,t}$ 对 $x_{i,t-1}$ 的混合回归，系数的理论值为 $\Gamma\mu$。

```{code-cell} python
pred = simulate_market(MarketParams(n_stocks=100, n_periods=5_000), seed=11)
Xp = np.column_stack([np.ones(len(pred.panel)), pred.panel[[f"z{j}" for j in range(1, 6)]].to_numpy()])
coef_p, se_p = ols(Xp, pred.panel["exret"].to_numpy())
truth_p = pred.gamma @ np.asarray(pred.params.factor_mean)

# 逐期横截面回归，用各期系数的时间序列标准差计算标准误（Fama–MacBeth 的做法，第 10 章详述）
n_t, n_i = pred.params.n_periods, pred.params.n_stocks
Xp3 = Xp.reshape(n_t, n_i, -1)
yp3 = pred.panel["exret"].to_numpy().reshape(n_t, n_i)
per_period = np.array([np.linalg.lstsq(Xp3[t], yp3[t], rcond=None)[0] for t in range(n_t)])
se_fm = per_period.std(axis=0, ddof=1) / np.sqrt(n_t)
print(pd.DataFrame({"理论值 Γμ": truth_p, "混合回归估计": coef_p, "同方差标准误": se_p, "逐期回归标准误": se_fm},
                   index=["常数", "z1", "z2", "z3", "z4", "z5"]).round(5))
```

系数的估计值与理论值 $\Gamma\mu$ 相符（偏离在两倍逐期回归标准误以内）：$z_1$、$z_2$ 的系数显著为正，而 $\delta = 0$，市场中没有任何错误定价。表中还有一个第二部分会反复出现的问题：“同方差标准误”低估了真实的不确定性。同一期的所有股票共同受到 $f_t$ 的冲击，残差在横截面上相关，混合回归把 $NT$ 个观测当作相互独立，高估了信息量。按期做横截面回归、再看各期系数的时间序列波动，得到的标准误更接近真实的抽样误差（第 10、12 章）：常数项与 $z_1$、$z_2$ 的逐期回归标准误明显大于同方差标准误，因为它们的系数随 $f_t$ 逐期波动；$z_3$～$z_5$ 与因子无关，两种标准误相近。估计值与理论值的偏离，应当按这种更大的标准误来判断。

## 本章小结

1. 本书用 Jupyter Book 2 构建，`sh scripts/build.sh` 会执行全部代码；全书的数字与图形都可以在本地复现。
2. 全书记号见[记号表](../notation.md)。时点约定是：下标 $t$ 表示第 $t$ 期末已知；面板中第 $t$ 期的一行存放期初特征 $z_{i,t-1}$ 与本期收益 $r_{i,t}$，特征比收益早一期。`period` 从 1 开始编号，`stock` 从 0 开始编号。
3. 模拟市场由式 {eq}`eq-ch0-factor`～{eq}`eq-ch0-sdf` 定义：独立同分布的因子、平稳的 AR(1) 特征、由特征决定的载荷 $\beta_{i,t} = \Gamma' x_{i,t-1}$ 与定价误差 $\alpha_{i,t} = \delta' z_{i,t-1}$。
4. SDF $m_t = 1 - b'(f_t - \mu)$，$b = \Sigma_f^{-1}\mu$ 为因子正确定价，并使 $\E[m_t r_{i,t} \mid z_{i,t-1}] = \alpha_{i,t}$（[定理 %s](#thm-ch0-sdf)）。证明的关键是 $f_t$、$\varepsilon_{i,t}$ 与 $z_{i,t-1}$ 相互独立。模拟器中因子相互独立，$b$ 化为 $\mu_k / \sigma_k^2$。
5. 存在错误定价时，单只股票的无条件定价误差仍可能为 0，错误定价要通过以特征加权的管理组合才能看到（[推论 %s](#cor-ch0-managed)）。
6. 交互回归可以恢复 $\Gamma$；存在错误定价而不加入特征本身时，估计有遗漏变量偏误，其大小由样本因子路径决定，大样本下方向趋于 SDF 系数 $b$ 的方向。
7. 检验“是否恢复真值”时，要把偏离与标准误比较，必要时用多次重复模拟检查标准误本身是否正确。

## 习题

### 概念题

:::{exercise} 时点错位
:label: ex-ch0-timing
某研究者在第 $t$ 期末按当时的市值把股票分为五组，然后计算各组在第 $t$ 期的收益，发现大市值组的收益显著高于小市值组，于是断言“大市值股票有更高的期望收益”。

1. 这个做法错在哪里？用本章的时点约定说明。
2. 在本章的模拟器中犯同样的错误（用 $z_{i,t}$ 代替 $z_{i,t-1}$），会产生虚假的显著结果吗？为什么真实数据中会？
:::

:::{solution} ex-ch0-timing
:class: dropdown
1. 第 $t$ 期末的市值是 $p_{i,t}$ 乘以股本，而 $p_{i,t}$ 已经包含了第 $t$ 期的收益。按它分组，再看同一期的收益，相当于用结果去解释结果：第 $t$ 期上涨的股票市值变大、下跌的变小，于是“小市值组”机械地集中了本期下跌的股票。正确的做法是按第 $t-1$ 期末的市值分组，计算第 $t$ 期的收益；在本书的面板中，就是用同一行的 `z` 列解释 `exret` 列。研究者发现的“大市值组收益更高”，正是这种机械效应的结果，与期望收益无关。
2. 不会。模拟器中 $z_{i,t} = \rho z_{i,t-1} + \sqrt{1-\rho^2} e_{i,t}$，更新冲击 $e_{i,t}$ 与 $r_{i,t}$ 独立，所以 $z_{i,t}$ 只是 $z_{i,t-1}$ 的一个含噪声的版本，错用它只会把真实的关系衰减为原来的 $\rho$ 倍（见“常见误区”一节的推导与代码）。真实数据中，特征的更新与当期收益相关（市值、动量、换手率、波动率等都直接由当期价格或成交量计算），所以错用同期特征会制造虚假的预测力。这也提醒我们：模拟只能检验我们放进模型的机制。
:::

:::{exercise} 无条件定价误差为零，是否说明没有错误定价？
:label: ex-ch0-uncond
设 $\delta = (0.01, 0, 0, 0, 0)'$。某研究者用真实的 $m_t$ 对每只股票计算 $T^{-1}\sum_t m_t r_{i,t}$，发现在 $T$ 很大时它们都趋于 0，于是断言“该 SDF 为全部股票正确定价，市场中不存在错误定价”。这一断言对吗？应当怎样设计检验资产？
:::

:::{solution} ex-ch0-uncond
:class: dropdown
不对。由[推论 %s](#cor-ch0-managed)，$\E[m_t r_{i,t}] = \delta' \E[z_{i,t-1}] = 0$：在本模拟器中，每只股票的特征都围绕 0 波动，错误定价时正时负，长期平均恰好抵消，所以无条件定价误差为 0。但条件定价误差 $\E[m_t r_{i,t} \mid z_{i,t-1}] = 0.01\, z_{1,i,t-1}$ 并不为 0：按 $z_1$ 做多高、做空低的策略可以获得 SDF 不能解释的收益。

要让无条件矩检验看到条件错误定价，应当把检验资产换成以期初信息加权的管理组合，例如 $r_{i,t} z_{1,i,t-1}$，或者按 $z_1$ 排序构造的组合。由推论，$\E[m_t r_{i,t} z_{i,t-1}] = \delta \neq 0$。这正是第 5 章“条件模型与管理组合”的核心思想：条件矩 $\E[m_t r_{i,t} \mid z_{i,t-1}] = 0$ 等价于对所有 $z_{i,t-1}$ 的函数 $g$ 都有 $\E[m_t r_{i,t}\, g(z_{i,t-1})] = 0$，检验时只能选有限个 $g$，选择决定了检验能看到什么。（真实数据中，特征通常有持续的个股差异，例如某些股票长期是小市值，此时单只股票的无条件定价误差一般不为 0，但同样的论证仍然成立。）
:::

### 推导题

:::{exercise} 时间序列回归的截距
:label: ex-ch0-ts
对固定的股票 $i$，考虑总体意义上的时间序列回归

```{math}
:enumerated: false
r_{i,t} = a_i + h_i' f_t + u_{i,t}, \qquad \E[u_{i,t}] = 0,\ \Cov(f_t, u_{i,t}) = 0 .
```

设 $\delta \neq 0$，且特征处于平稳分布。证明：斜率 $h_i = \E[\beta_{i,t}]$，截距 $a_i = \E[\alpha_{i,t}]$。注意载荷 $\beta_{i,t}$ 是时变的，而回归只有一个常数斜率；说明证明在哪一步用到了 $z_{i,t-1}$ 与 $f_t$ 的独立性。在本模拟器中 $a_i$ 等于多少？这对用时间序列回归检验定价模型意味着什么？
:::

:::{solution-start} ex-ch0-ts
:class: dropdown
:::

总体最小二乘的斜率与截距为 $h_i = \Var(f_t)^{-1}\Cov(f_t, r_{i,t})$，$a_i = \E[r_{i,t}] - h_i'\mu$。把 $r_{i,t} = \alpha_{i,t} + \beta_{i,t}' f_t + \varepsilon_{i,t}$ 代入，逐项计算。

**协方差。** $\Cov(f_t, r_{i,t}) = \Cov(f_t, \alpha_{i,t}) + \Cov(f_t, \beta_{i,t}' f_t) + \Cov(f_t, \varepsilon_{i,t})$。

- $\alpha_{i,t}$ 是 $z_{i,t-1}$ 的函数，与 $f_t$ 独立，故 $\Cov(f_t, \alpha_{i,t}) = 0$；$\varepsilon_{i,t}$ 与 $f_t$ 独立，故第三项为 0。
- 第二项中 $\beta_{i,t}' f_t = f_t' \beta_{i,t}$ 是标量，

  ```{math}
  :enumerated: false
  \Cov(f_t, f_t'\beta_{i,t}) = \E[f_t f_t' \beta_{i,t}] - \E[f_t]\,\E[f_t'\beta_{i,t}] .
  ```

  由于 $\beta_{i,t}$ 是 $z_{i,t-1}$ 的函数、与 $f_t$ 独立，期望可以分开：$\E[f_t f_t' \beta_{i,t}] = \E[f_t f_t']\,\E[\beta_{i,t}] = (\Sigma_f + \mu\mu')\,\E[\beta_{i,t}]$，$\E[f_t'\beta_{i,t}] = \mu'\E[\beta_{i,t}]$。于是该项等于 $\Sigma_f\, \E[\beta_{i,t}]$。

因此 $h_i = \Sigma_f^{-1}\Sigma_f\,\E[\beta_{i,t}] = \E[\beta_{i,t}]$。

**截距。** $\E[r_{i,t}] = \E[\alpha_{i,t}] + \E[\beta_{i,t}' f_t] + 0 = \E[\alpha_{i,t}] + \E[\beta_{i,t}]'\mu$（同样用到独立性）。所以

```{math}
:enumerated: false
a_i = \E[\alpha_{i,t}] + \E[\beta_{i,t}]'\mu - \E[\beta_{i,t}]'\mu = \E[\alpha_{i,t}] .
```

独立性用在两处：把 $\E[f_t f_t'\beta_{i,t}]$ 与 $\E[\beta_{i,t}' f_t]$ 拆成乘积，以及 $\Cov(f_t, \alpha_{i,t}) = 0$。如果特征能预测因子收益（$z$ 与 $f$ 相关），这两步都不成立，常数斜率的回归会把载荷的时变与因子均值的时变混在一起，截距不再等于平均定价误差。另外注意，回归残差 $u_{i,t} = \alpha_{i,t} - \E[\alpha_{i,t}] + (\beta_{i,t} - \E[\beta_{i,t}])'f_t + \varepsilon_{i,t}$ 与 $f_t$ 不相关，但并不独立于 $f_t$（它的方差依赖于 $f_t$），这影响标准误，不影响上面的总体系数。

**在本模拟器中**，$\E[\beta_{i,t}] = \Gamma'\E[x_{i,t-1}] = \Gamma'(1, 0, \dots, 0)'$，即 $\Gamma$ 的第 0 行；$a_i = \delta'\E[z_{i,t-1}] = 0$。也就是说，即使存在错误定价，每只股票时间序列回归的截距在总体上也是 0，GRS 一类的时间序列检验（第 9 章）检验的只是平均定价误差，看不到这里的条件错误定价——与[习题 %s](#ex-ch0-uncond) 的结论一致。

在有限样本中，股票 $i$ 的特征路径是一次具体的实现，样本截距近似等于该路径上的平均定价误差 $T^{-1}\sum_t \alpha_{i,t}$。下面的代码同时验证总体结论与这一有限样本含义：取 $\rho = 0.98$，使各股票的特征路径均值有明显差异，并把异质性波动降为 0.05 以减小噪声。

```{code-cell} python
ts_params = MarketParams(n_stocks=300, n_periods=600, char_persistence=0.98, idio_vol=0.05,
                         alpha_delta=(0.02, 0.0, 0.0, 0.0, 0.0))
ts = simulate_market(ts_params, seed=7)
F_ts = ts.factors.to_numpy()
R_ts = ts.panel["exret"].unstack("stock").to_numpy()      # (T, N)
A_ts = ts.panel["alpha"].unstack("stock").to_numpy()
coef_ts, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(F_ts)), F_ts]), R_ts, rcond=None)
a_hat, h_hat = coef_ts[0], coef_ts[1:]
alpha_path_mean = A_ts.mean(axis=0)

print("斜率的横截面平均", h_hat.mean(axis=1).round(4), "；理论值 E[β] =", ts.gamma[0])
print(f"截距的横截面平均 {a_hat.mean():.5f}（理论值 E[α] = 0）")
slope, intercept = np.polyfit(alpha_path_mean, a_hat, 1)
print(f"截距对路径平均 α 的回归：斜率 {slope:.3f}（理论值 1），截距 {intercept:.5f}（理论值 0），"
      f"相关系数 {np.corrcoef(alpha_path_mean, a_hat)[0, 1]:.3f}")
```

截距的横截面平均接近 0，斜率的平均接近 $\Gamma$ 的第 0 行；而单只股票的样本截距几乎一对一地反映了它在样本期内的平均定价误差。
:::{solution-end}
:::

:::{exercise} 从 SDF 到 β 定价
:label: ex-ch0-beta
利用[定理 %s](#thm-ch0-sdf) 第 3 条与条件协方差的分解 $\E[m_t r_{i,t} \mid z] = \E[m_t \mid z]\,\E[r_{i,t} \mid z] + \Cov(m_t, r_{i,t} \mid z)$（$z$ 是 $z_{i,t-1}$ 的简写），证明

```{math}
:enumerated: false
\E[r_{i,t} \mid z_{i,t-1}] = \alpha_{i,t} + \beta_{i,t}'\lambda, \qquad \lambda = \mu .
```

解释为什么因子的风险溢价恰好等于因子的均值，并用模拟数据验证：混合回归 $r_{i,t}$ 对 $x_{i,t-1}$ 的系数应为 $\Gamma\mu + (0, \delta')'$。
:::

:::{solution-start} ex-ch0-beta
:class: dropdown
:::

由定理，左端 $\E[m_t r_{i,t} \mid z] = \alpha_{i,t}$；又因 $f_t$ 与 $z$ 独立，$\E[m_t \mid z] = 1$。只需计算条件协方差。以 $z$ 为条件时，$\beta_{i,t}$ 是常数，$m_t - \E[m_t \mid z] = -b'(f_t - \mu)$，$r_{i,t} - \E[r_{i,t} \mid z] = \beta_{i,t}'(f_t - \mu) + \varepsilon_{i,t}$。由于 $(f_t, \varepsilon_{i,t})$ 与 $z$ 独立、$f_t$ 与 $\varepsilon_{i,t}$ 独立，

```{math}
:enumerated: false
\Cov(m_t, r_{i,t} \mid z) = -b'\, \E\left[(f_t - \mu)(f_t - \mu)'\right] \beta_{i,t} - b'\,\E\left[(f_t-\mu)\varepsilon_{i,t}\right] = -b'\Sigma_f\, \beta_{i,t} = -\mu'\beta_{i,t} .
```

代入分解式：$\alpha_{i,t} = 1 \cdot \E[r_{i,t} \mid z] - \mu'\beta_{i,t}$，即 $\E[r_{i,t} \mid z] = \alpha_{i,t} + \beta_{i,t}'\mu$。

一般地，β 定价中的风险溢价为 $\lambda = -\Cov(m, f)/\E[m] = \Sigma_f b / \E[m]$。当因子本身是超额收益、且被 $m$ 正确定价时，把这一关系用于因子自身（其对自己的载荷为单位矩阵、定价误差为 0），立即得到 $\lambda = \E[f_t] = \mu$。第 4 章会在更一般的条件下讨论 SDF 表示与 β 表示的等价性。

验证：由 $\E[r_{i,t} \mid z] = \delta'z + x'\Gamma\mu$ 与 $\E[x x'] = I_{L+1}$，混合回归 $r_{i,t}$ 对 $x_{i,t-1}$ 的总体系数为 $\Gamma\mu + (0, \delta')'$。

```{code-cell} python
bp_params = MarketParams(n_stocks=100, n_periods=5_000, alpha_delta=(0.01, -0.005, 0.0, 0.0, 0.0))
bp = simulate_market(bp_params, seed=12)
Xb = np.column_stack([np.ones(len(bp.panel)), bp.panel[[f"z{j}" for j in range(1, 6)]].to_numpy()])
yb = bp.panel["exret"].to_numpy()
coef_b = np.linalg.lstsq(Xb, yb, rcond=None)[0]
per_b = np.array([np.linalg.lstsq(Xb.reshape(5_000, 100, -1)[t], yb.reshape(5_000, 100)[t], rcond=None)[0]
                  for t in range(5_000)])
truth_b = bp.gamma @ np.asarray(bp_params.factor_mean) + np.r_[0.0, bp.delta]
print(pd.DataFrame({"理论值 Γμ + (0, δ')'": truth_b, "估计值": coef_b,
                    "逐期回归标准误": per_b.std(axis=0, ddof=1) / np.sqrt(5_000)},
                   index=["常数", "z1", "z2", "z3", "z4", "z5"]).round(5))
```

各系数与理论值的偏离都在两倍标准误左右以内。
:::{solution-end}
:::

### 编程题

:::{exercise} 特征持续性与 Γ 的估计误差
:label: ex-ch0-rho
取 $N = 200$、$T = 120$，令 $\rho \in \{0, 0.5, 0.9, 0.99\}$，对每个 $\rho$ 用 20 个不同的种子重复模拟，用交互回归估计 $\Gamma$，计算估计误差的均方根 $\left[\frac{1}{(L+1)K}\sum_{j,k}(\hat\Gamma_{j,k} - \Gamma_{j,k})^2\right]^{1/2}$ 的平均值。持续性对估计误差有多大影响？把 $N$ 降到 10 再做一次，结论有变化吗？给出解释。
:::

:::{solution-start} ex-ch0-rho
:class: dropdown
:::

```{code-cell} python
def gamma_rmse(n_stocks, rho, n_rep, base_seed):
    errs = []
    for r in range(n_rep):
        rep = simulate_market(MarketParams(n_stocks=n_stocks, n_periods=120, char_persistence=rho),
                              seed=base_seed + r)
        Xr, _ = interaction_design(rep)
        g_hat = np.linalg.lstsq(Xr, rep.panel["exret"].to_numpy(), rcond=None)[0].reshape(-1, K)
        errs.append(np.sqrt(np.mean((g_hat - rep.gamma) ** 2)))
    errs = np.asarray(errs)
    return errs.mean(), errs.std(ddof=1) / np.sqrt(n_rep)


rows = []
for n_stocks, n_rep, base in [(200, 20, 3000), (10, 50, 4000)]:
    for rho in (0.0, 0.5, 0.9, 0.99):
        mean_err, se_err = gamma_rmse(n_stocks, rho, n_rep, base)
        rows.append({"N": n_stocks, "ρ": rho, "重复次数": n_rep, "平均 RMSE": mean_err, "其标准误": se_err})
pd.DataFrame(rows).round(4)
```

$N = 200$ 时，四种持续性下的平均误差相差不到 10%，与蒙特卡洛标准误相比差别不大，也没有随 $\rho$ 单调变化的趋势。原因是：在 $\varepsilon_{i,t}$ 与解释变量独立的条件下，最小二乘估计的条件协方差为 $s^2 (X'X)^{-1}$，而 $X'X = \sum_t \left(\sum_i x_{i,t-1}x_{i,t-1}'\right) \otimes f_t f_t'$。特征的边际分布始终是 $N(0, I_L)$，与 $\rho$ 无关；$\rho$ 只决定同一只股票的特征在不同期之间是否相似。当 $N$ 足够大时，每一期的横截面矩 $N^{-1}\sum_i x x'$ 都接近 $I_{L+1}$，$X'X$ 也就几乎不受 $\rho$ 影响。

$N = 10$ 时，$\rho = 0.99$ 的误差明显变大。此时每一期只有 10 个特征观测，而且它们在整个样本期内几乎不变，样本矩 $\sum_t\sum_i x x'$ 实际上只由少数几个独立抽样决定，可能明显偏离 $T N \cdot I$，某些特征方向上的信息很少；$(X'X)^{-1}$ 的期望因此变大。换句话说，持续性降低的是“有效样本量”，当横截面足够大时这一影响被横截面上的独立变化所弥补。

需要强调的是，本题中持续性不会造成偏误，只影响方差。持续性造成偏误的典型情形是预测回归：被解释变量对一个高度持续的滞后变量回归，而该变量的更新冲击与收益冲击相关，此时最小二乘斜率有小样本偏误 [@stambaugh1999predictive]。本模拟器中 $e_{i,t}$ 与 $\varepsilon_{i,t}$ 独立，所以不会出现这种偏误。第 13 章讨论这一问题。
:::{solution-end}
:::

:::{exercise} 用管理组合恢复 δ
:label: ex-ch0-delta
取默认的 $N = 500$、$T = 240$ 与 $\delta = (0.01, -0.005, 0, 0, 0)'$。利用[推论 %s](#cor-ch0-managed)，用真实的 $m_t$ 计算 $\hat\delta_j = (NT)^{-1}\sum_{t}\sum_{i} m_t r_{i,t} z_{j,i,t-1}$，并给出标准误。标准误应当怎样计算才能反映同期股票之间的相关？把结果与“模拟实验”一节中加入特征本身的交互回归所得的 $\hat\delta$ 比较。
:::

:::{solution-start} ex-ch0-delta
:class: dropdown
:::

先对每一期计算横截面平均 $g_{j,t} = N^{-1}\sum_i m_t r_{i,t} z_{j,i,t-1}$，再对 $g_{j,t}$ 求时间序列均值。由于因子冲击在同一期内影响所有股票，$g_{j,t}$ 已经把同期相关“吸收”进去了，而不同期的 $g_{j,t}$ 近似独立：$f_t$、$\varepsilon_{i,t}$ 跨期独立；$g_{j,t}$ 的条件均值 $N^{-1}\sum_i z_{j,i,t-1} z_{i,t-1}'\delta$ 随特征缓慢变化，但其波动只有 $N^{-1/2}$ 的量级，远小于因子冲击带来的波动。所以标准误可用 $\text{Std}(g_{j,t})/\sqrt{T}$，代码中同时计算 $g_{1,t}$ 的一阶自相关加以确认；单个样本的自相关本身有约 $1/\sqrt{T}$ 的抽样误差，所以再用 100 个种子重复，看其平均值是否接近 0。把 $NT$ 个观测当作独立样本计算的标准误，会低估同期相关较强的那些矩的不确定性。

```{code-cell} python
dl_params = MarketParams(alpha_delta=(0.01, -0.005, 0.0, 0.0, 0.0))
dl = simulate_market(dl_params, seed=21)
m_dl = dl.sdf().to_numpy()
R_dl = dl.panel["exret"].unstack("stock").to_numpy()
g = np.stack([(m_dl[:, None] * R_dl * dl.panel[f"z{j}"].unstack("stock").to_numpy()).mean(axis=1)
              for j in range(1, 6)], axis=1)                       # (T, L)：每期横截面平均
delta_mom = g.mean(axis=0)
se_mom = g.std(axis=0, ddof=1) / np.sqrt(len(g))
se_naive = np.array([(m_dl[:, None] * R_dl * dl.panel[f"z{j}"].unstack("stock").to_numpy()).std()
                     for j in range(1, 6)]) / np.sqrt(R_dl.size)

X_dl, Z_dl = interaction_design(dl)
coef_dl, se_dl = ols(np.column_stack([X_dl, Z_dl]), dl.panel["exret"].to_numpy())


def g1_autocorr(market):
    m_ = market.sdf().to_numpy()
    g1 = (m_[:, None] * market.panel["exret"].unstack("stock").to_numpy()
          * market.panel["z1"].unstack("stock").to_numpy()).mean(axis=1)
    return np.corrcoef(g1[1:], g1[:-1])[0, 1]


ac_reps = [g1_autocorr(simulate_market(dl_params, seed=500 + r)) for r in range(100)]
print(f"本样本 g_1t 的一阶自相关 {g1_autocorr(dl):.3f}（单个样本的抽样误差约 1/√T = {1 / np.sqrt(len(g)):.3f}）；"
      f"100 次重复的平均 {np.mean(ac_reps):.3f}（标准误 {np.std(ac_reps, ddof=1) / 10:.3f}）")
print(pd.DataFrame({"真值": dl.delta, "矩估计": delta_mom, "按期标准误": se_mom, "独立假设标准误": se_naive,
                    "交互回归估计": coef_dl[X_dl.shape[1]:], "回归标准误": se_dl[X_dl.shape[1]:]},
                   index=[f"δ{j}" for j in range(1, 6)]).round(4))
```

本样本的自相关为 −0.147，约为其抽样误差的 2 倍，看起来不小；但重复模拟的平均自相关只有 −0.02 左右，说明这主要是抽样误差，按 $\text{Std}(g_{j,t})/\sqrt{T}$ 计算标准误时可以忽略。两种方法都恢复了 $\delta$（偏离均在两倍标准误以内）。$\delta_1$、$\delta_2$ 的按期标准误约为独立假设标准误的 3 倍，$\delta_3$～$\delta_5$ 则二者相近：$z_1$、$z_2$ 决定了对第二、三个因子的载荷，$m_t r_{i,t} z_{1,i,t-1}$ 中含有 $0.5\, m_t f_{2,t} z_{1,i,t-1}^2$ 这样在同一期内所有股票共有的项，横截面平均无法把它平均掉；$z_3$～$z_5$ 与载荷无关，没有这样的共同项。矩估计直接利用了真实 SDF，回归方法则利用了“载荷是特征的线性函数”这一结构；后者的标准误更小，因为它把因子冲击带来的收益波动作为解释变量吸收掉了，残差只剩 $\varepsilon_{i,t}$。真实研究中 SDF 未知，必须估计，这正是第二部分 GMM 与第四部分 SDF 估计方法要解决的问题。
:::{solution-end}
:::

## 文献与延伸阅读

- 随机贴现因子、基本定价方程 $p_t = \E_t[m_{t+1}x_{t+1}]$ 以及线性因子模型与 SDF 的对应关系，系统的教材处理见 @cochrane2005asset。第一部分将以它为主要参照，用本书自己的记号重新推导。
- 期望收益随时间与资产变化，以及“哪些特征真正提供了独立的定价信息”这一问题，@cochrane2011presidential 给出了提纲挈领的讨论，是理解全书动机的好起点。
- 模拟器中“载荷是特征的线性函数”与“特征驱动的截距”的设定，来自 IPCA 模型 @kelly2019characteristics，第 23 章将完整介绍其估计方法。
- SDF 波动率与最大夏普比率的关系，即 Hansen–Jagannathan 界，原始文献为 @hansen1991implications，第 3 章详细讨论。
- 持续回归变量的小样本偏误见 @stambaugh1999predictive，第 13 章详细讨论。
- 机器学习预测收益的基准框架见 @gu2020empirical；中国市场的对应研究见 @leippold2022machine；中国市场的规模、价值因子与壳价值问题见 @liu2019size。它们分别是第 22 章与第 16、17 章的主要文献。
