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
2. 用白话和数值例子说明超额收益、因子、载荷、风险溢价、特征、定价误差与随机贴现因子的含义，并说出它们在本章模型中的符号与 `apsim` 中的字段；
3. 理解全书的记号与时点约定，特别是横截面面板中“特征比收益早一期”的含义，以及公式中的向量、矩阵与代码中数组形状的对应；
4. 用 `apsim.simulate_market` 生成一个模拟市场，读懂返回结果中的每一列，并说出它对应模型中的哪个变量；
5. 写出模拟市场的全部模型方程，证明其中的随机贴现因子对因子定价、对股票的定价误差恰好是 $\alpha_{i,t}$；
6. 掌握“已知真值”的检验思路：先在模拟数据上确认一个方法能恢复真值，再把它用到真实数据上。

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
- **Node.js**：Jupyter Book 2 的构建工具链运行在 Node.js 上，需要预先安装（版本不低于 18；本项目在 v24 下测试）。首次构建时还需要联网下载网页主题模板，详见项目根目录的 `README.md`。

在项目根目录下依次运行：

```bash
uv sync                       # 创建 .venv 并安装依赖（含 jupyter-book 与 ipykernel）
uv run pytest                 # 运行 apsim 的单元测试，确认环境可用
sh scripts/build.sh --clean   # 冷构建：清除执行缓存，执行全部代码，输出到 book/_build/html
sh scripts/serve.sh           # 在 http://localhost:8000 提供网页
```

`scripts/build.sh` 以 `--strict` 模式构建：任何一个代码单元报错（包括正文数字所依赖的 `assert` 不成立），构建都会失败。写作或修改时，可以在 `book/` 目录下运行 `uv run jupyter book start --execute` 获得实时预览。

常见问题：

1. **找不到 `node` 或构建工具报 Node.js 相关错误。** 先在终端运行 `node --version` 确认 Node.js 已安装且在 `PATH` 中。
2. **执行代码时找不到内核或找不到 `apsim`。** 每章开头的元数据声明了内核 `python3`。务必通过 `uv run` 启动构建（`scripts/build.sh` 已经这样做），这样使用的是项目虚拟环境中的 `ipykernel`，而 `apsim` 以可编辑方式安装在同一环境中。
3. **修改代码后网页上的结果没有变。** 构建会缓存代码单元的输出，只有代码单元本身改动时才重新执行。如果改的是 `src/apsim` 中的源码，而章节中的代码单元没有变，缓存的输出就会过期。此时运行 `sh scripts/build.sh --clean`，它只删除执行缓存 `book/_build/execute` 再构建（不要删除整个 `book/_build`，其中还缓存了网页主题模板）。
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

(sec-ch0-concepts)=
## 0.2 基本概念速览

从 0.3 节起，正文会直接使用超额收益、因子、载荷、风险溢价、特征、定价误差、随机贴现因子这些术语，0.4 节的模型就是把它们写成方程。本节先逐个讲清楚它们的含义。每个概念都按同一个顺序介绍：先用一两句白话说明它是什么，再举一个数值例子，最后给出公式，并指出它在本章模型中对应哪个符号。例子借用读者熟悉的 A 股情形，但**例子中的股票都不指具体公司，所有数字都是为说明概念而假设的**，不是实际数据。每个例子后面的代码单元用 `numpy` 复算其中的数字，并用 `assert` 核对。

本节只求建立直觉：模型的方程在 0.4 节，定理与证明在 0.5 节，系统的理论在第 1～4 章展开。

### 总收益、无风险利率与超额收益

**白话。** 总收益是期末拿回的钱（价格加上期间的红利）与期初投入的钱之比。无风险利率是把钱放在没有违约风险的资产上所得的收益，在期初就已确定。超额收益是二者之差，也就是因为承担了风险而多得（或少得）的那一部分。

**例子。** 假设某只白酒龙头股第 $t-1$ 个月末的价格为 100 元，第 $t$ 个月末涨到 102 元，期间每股派发红利 1 元。持有一个月得到的支付是 $102 + 1 = 103$ 元，总收益为 $103 / 100 = 1.03$，即收益率 3%。再假设这个月的无风险利率为 0.2%（无风险总收益为 1.002），它在月初就已知道。这只股票当月的超额收益为 $1.03 - 1.002 = 0.028$，即 2.8%。用收益率相减也得到同样的结果：$3\% - 0.2\% = 2.8\%$，因为总收益与收益率只差一个 1，相减时消去了。

**公式。** 按 0.3 节的时点约定，从第 $t-1$ 期末持有到第 $t$ 期末，

```{math}
:enumerated: false
R_{i,t} = \frac{x_{i,t}}{P_{i,t-1}}, \qquad R^e_{i,t} = R_{i,t} - R^f_{t-1}, \qquad r_{i,t} = R^e_{i,t},
```

其中 $P_{i,t-1}$ 是股票 $i$ 在第 $t-1$ 期末的价格，$x_{i,t}$ 是第 $t$ 期末的价格加上期间的红利，$R^f_{t-1}$ 是第 $t-1$ 期末已知的无风险总收益，三者与 $R_{i,t}$、$R^e_{i,t}$ 都是标量。面板与模拟部分把超额收益简记为 $r_{i,t}$。

为什么资产定价研究超额收益，而不是总收益？有三个原因。第一，无风险利率是“等待”的报酬，与承担什么风险无关，而且本身随时间变化；资产定价要回答的是不同股票的期望收益为什么不同，先减去所有资产都能拿到的无风险收益，比较才干净。第二，超额收益是一个零成本头寸的收益：按无风险利率借入 1 元买入股票，期初不花钱，期末得到 $R_{i,t} - R^f_{t-1}$。零成本头寸的价格为 0，所以下文的定价方程对超额收益写成 $0 = \E[m R^e]$，形式最简单。第三，本章的因子本身也是超额收益（见下一个概念）。股票与因子用同一种口径时，时间序列回归的截距就是定价误差 $\alpha$，并且在模型对因子自身没有定价误差时，因子的风险溢价就是它的均值 $\E[\mathbf f_t]$（条件见下文“风险溢价”与“定价误差”两个概念）。

本章的模拟器只生成超额收益 $r_{i,t}$（面板的 `panel["exret"]` 列），不单独生成总收益与无风险利率，原因见 0.4 节末尾的注释“为什么是 $\E[m_t] = 1$”。

```{code-cell} python
# 白酒龙头（假设数字）：月初价格 100 元，月末价格 102 元，期间每股红利 1 元
P_prev, P_now, div = 100.0, 102.0, 1.0
R_gross = (P_now + div) / P_prev            # 总收益 R_{i,t} = x_{i,t} / P_{i,t-1}
Rf_prev = 1.002                             # 无风险总收益 R^f_{t-1}，月初已知
r_excess = R_gross - Rf_prev                # 超额收益 r_{i,t} = R_{i,t} - R^f_{t-1}
print(f"总收益 {R_gross:.4f}，收益率 {R_gross - 1:.2%}，超额收益 {r_excess:.2%}")
assert np.isclose(R_gross, 1.03) and np.isclose(r_excess, 0.028)
assert np.isclose((R_gross - 1) - (Rf_prev - 1), r_excess)     # 用收益率相减，结果相同
```

### 因子

**白话。** 因子是许多股票共同面对的收益来源。大盘大涨的月份，多数股票跟着上涨；小盘股整体跑赢大盘股的月份，几乎所有小盘股都受益。用一个可交易组合的收益来度量这种“共同的涨跌”，就得到一个因子。本章的因子，以及第三、四部分大多数模型中的因子，都是超额收益：市场因子是市场组合的超额收益；规模、价值这类因子是“买入一组股票、同时卖空另一组股票”的多空组合收益。但并非所有因子都是资产收益：第 1、4、10 章会遇到消费增长、宏观变量这类不可交易的因子，它们的风险溢价 $\bv\lambda$ 不等于因子的均值。

**例子。** 假设某个月：市场组合的收益率为 3.2%，无风险利率为 0.2%；小市值股票组合的收益率为 5.0%，大市值组合为 2.0%；价值股组合（相对于基本面“便宜”的股票）为 2.5%，成长股组合为 3.5%。于是这个月

- 市场因子 $= 3.2\% - 0.2\% = 3.0\%$；
- 规模因子 $=$ 小市值组合 $-$ 大市值组合 $= 5.0\% - 2.0\% = 3.0\%$；
- 价值因子 $=$ 价值股组合 $-$ 成长股组合 $= 2.5\% - 3.5\% = -1.0\%$。

规模因子与价值因子不需要再减无风险利率：多空组合的买入与卖空各占 1 元，期初不花钱，两条腿的无风险利率相互抵消，多空收益本身就是超额收益。规模因子为正，说明这个月小盘股跑赢大盘股；价值因子为负，说明成长股跑赢价值股。Liu、Stambaugh 与 Yuan 为中国市场构造的三因子模型（CH3）正是由市场、规模、价值三个因子组成，其中价值按盈利价格比（EP）而不是账面市值比划分，并剔除了市值最小的 30% 的股票 [@liu2019size]；因子的具体构造见第 15、16 章。

**公式。** 记 $R_{\text{mkt},t}$、$R_{\text{small},t}$、$R_{\text{big},t}$、$R_{\text{value},t}$、$R_{\text{growth},t}$ 分别为市场组合、小市值组合、大市值组合、价值股组合与成长股组合的总收益，则

```{math}
:enumerated: false
f_{1,t} = R_{\text{mkt},t} - R^f_{t-1}, \qquad f_{2,t} = R_{\text{small},t} - R_{\text{big},t}, \qquad f_{3,t} = R_{\text{value},t} - R_{\text{growth},t}, \qquad \underbrace{\mathbf f_t}_{3\times 1} = (f_{1,t}, f_{2,t}, f_{3,t})' .
```

一般地，$K$ 个因子组成列向量 $\mathbf f_t = (f_{1,t}, \dots, f_{K,t})' \in \Real^K$。本例中 $\mathbf f_t = (3\%, 3\%, -1\%)'$。本章模型中的 $\mathbf f_t$ 由式 {eq}`eq-ch0-factor` 生成，对应 `mkt.factors` 的列 `f1`……`fK`。模拟器的因子是抽象的，没有名字；但在默认参数下，所有股票对第一个因子的载荷都是 1，所以 $f_{1,t}$ 的角色类似市场因子。

```{code-cell} python
# 某个月各组合的收益率（假设数字）
R_mkt, R_free = 0.032, 0.002          # 市场组合、无风险利率
R_small, R_big = 0.050, 0.020         # 小市值组合、大市值组合
R_value, R_growth = 0.025, 0.035      # 价值股组合、成长股组合
f_month = np.array([R_mkt - R_free, R_small - R_big, R_value - R_growth])   # f_t: (K,) = (3,)
print("f_t =", f_month)
assert np.allclose(f_month, [0.03, 0.03, -0.01])
```

### 载荷（暴露、β）

**白话。** 载荷衡量一只股票对某个因子有多敏感：因子收益每高 1 个百分点，这只股票的收益平均高多少个百分点。载荷也叫暴露（exposure）；只有一个因子时，常把它记作 β。

**单因子的例子。** 只考虑市场因子。假设大盘银行股的市场载荷为 0.7，券商股为 1.5。某个月市场因子为 $+4\%$，则由市场因子带来的那部分收益，银行股为 $0.7 \times 4\% = 2.8\%$，券商股为 $1.5 \times 4\% = 6\%$；若市场因子为 $-4\%$，二者分别为 $-2.8\%$ 与 $-6\%$。券商股是“高 β”股票：大盘涨时它涨得更多，大盘跌时它也跌得更多。这里算的只是“因子带来的部分”，每只股票还有自己的消息带来的异质性收益，下面估计载荷时会看到。

**多因子的例子。** 有多个因子时，一只股票对每个因子各有一个载荷，$K$ 个载荷组成一个向量。下表是四只股票对 CH3 三个因子的载荷（假设数字），最后一列给出与上一个例子的因子 $\mathbf f_t = (3\%, 3\%, -1\%)'$ 对应的因子部分 $\bv\beta_i'\mathbf f_t$：

| 股票（假设） | 市场 | 规模 | 价值 | 直观含义 | 因子部分 $\bv\beta_i'\mathbf f_t$ |
|---|---|---|---|---|---|
| 白酒龙头 | 0.9 | −0.4 | −0.3 | 大市值、估值高，表现像成长股 | 1.8% |
| 券商股 | 1.5 | 0.2 | 0.1 | 对大盘最敏感 | 5.0% |
| 大盘银行股 | 0.7 | −0.5 | 0.8 | 大市值、估值低，表现像价值股 | −0.2% |
| 小盘股 | 1.1 | 1.2 | −0.2 | 随小盘股整体涨跌 | 7.1% |

以大盘银行股为例：市场上涨 3% 给它带来 $0.7 \times 3\% = 2.1\%$；这个月小盘股跑赢大盘股 3%，它的规模载荷为负，损失 $0.5 \times 3\% = 1.5\%$；成长股跑赢价值股 1%，它的价值载荷为正，再损失 $0.8 \times 1\% = 0.8\%$。合计 $-0.2\%$：大盘涨了，它反而略跌。

**公式。** 股票 $i$ 的载荷向量 $\bv\beta_i = (\beta_{1,i}, \dots, \beta_{K,i})' \in \Real^K$，因子带来的收益是内积

```{math}
:enumerated: false
\underbrace{\bv\beta_i'}_{1\times K}\underbrace{\mathbf f_t}_{K\times 1} = \sum_{k=1}^{K} \beta_{k,i} f_{k,t} .
```

把 $N$ 只股票的 $\bv\beta_i'$ 按行堆叠成 $N\times K$ 矩阵 $\mathbf B$（上表的前三列数字就是 $4\times 3$ 的 $\mathbf B$），$\mathbf B\mathbf f_t$（$N\times 1$）的第 $i$ 个元素就是 $\bv\beta_i'\mathbf f_t$。本章模型中载荷随时间变化，记作 $\bv\beta_{i,t} \in \Real^K$，由式 {eq}`eq-ch0-beta` 给出，对应面板的 `panel["beta1"]`……`panel["betaK"]` 列；堆叠后的矩阵为 $\mathbf B_t$。

```{code-cell} python
# 单因子：大盘银行股 β = 0.7，券商股 β = 1.5（假设数字）
beta_single = np.array([0.7, 1.5])                         # (2,)
for f_m in (0.04, -0.04):
    print(f"市场因子 {f_m:+.0%}：因子部分", (beta_single * f_m).round(4))
assert np.allclose(beta_single * 0.04, [0.028, 0.06])

# 多因子：每行是一只股票的载荷 β_i'，列依次为市场、规模、价值
B_ex = np.array([[0.9, -0.4, -0.3],     # 白酒龙头
                 [1.5,  0.2,  0.1],     # 券商股
                 [0.7, -0.5,  0.8],     # 大盘银行股
                 [1.1,  1.2, -0.2]])    # 小盘股；B: (N, K) = (4, 3)
factor_part = B_ex @ f_month                                # (4, 3) @ (3,) -> (4,)：第 i 个元素为 β_i'f_t
print("因子部分 β_i'f_t：", factor_part.round(4))
assert np.allclose(factor_part, [0.018, 0.050, -0.002, 0.071])
```

**用回归斜率估计载荷。** 载荷不能直接观察，需要从数据中估计。最直接的办法是把股票的超额收益对因子做时间序列回归：把每个月的（因子收益，股票收益）画成散点，拟合直线的斜率就是载荷的估计值，即因子每高 1 个百分点，股票收益平均高多少。下面是券商股与市场因子 6 个月的超额收益（假设数字）：

| 月份 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| 市场因子 | 3.0% | −1.0% | 2.0% | −4.0% | 1.0% | 5.0% |
| 券商股 | 5.2% | −1.8% | 3.2% | −5.8% | 2.2% | 7.2% |

最小二乘回归的斜率恰好为 1.5，截距为 0.2%。每个月的实际收益与拟合直线 $0.2\% + 1.5 f_t$ 相差 $\pm 0.5\%$ 或 0，这些残差就是券商股自身的消息带来的异质性收益。单因子时 $f_t$ 为标量，回归斜率等于样本协方差除以样本方差：

```{math}
:enumerated: false
\hat\beta_i = \frac{\widehat{\Cov}(r_{i,t}, f_t)}{\widehat{\Var}(f_t)} .
```

多因子时，回归模型为 $r_{i,t} = a_i + \bv\beta_i'\mathbf f_t + u_{i,t}$，其中截距 $a_i$ 与残差 $u_{i,t}$ 是标量，总体斜率为 $\bv\beta_i = \bv\Sigma_f^{-1}\Cov(\mathbf f_t, r_{i,t})$（$K\times K$ 乘 $K\times 1$，得 $K\times 1$）。每个载荷衡量的是“其他因子不变时”对这个因子的敏感度。

这个例子只用了 6 个月，数字又是特意挑选的，所以斜率恰好是 1.5，残差也很整齐。真实数据中，单只股票的异质性收益很大：0.4 节会算出，在本书的模拟市场中，异质性冲击约占单只股票月度收益方差的四分之三以上，用一只股票几年的数据估计出的载荷误差可观。另外，本章模型中的载荷 $\bv\beta_{i,t}$ 随时间变化，而上面的回归只有一个常数斜率，它估计的是载荷的平均值（[习题 %s](#ex-ch0-ts)）。

```{code-cell} python
# 6 个月的市场因子与券商股超额收益（假设数字）
f_mkt6 = np.array([3.0, -1.0, 2.0, -4.0, 1.0, 5.0]) / 100      # (T,) = (6,)
r_brk6 = np.array([5.2, -1.8, 3.2, -5.8, 2.2, 7.2]) / 100      # (6,)
X6 = np.column_stack([np.ones(6), f_mkt6])                     # (6, 2)：常数项与因子
(a_brk, beta_brk), *_ = np.linalg.lstsq(X6, r_brk6, rcond=None)
slope_cov = np.cov(r_brk6, f_mkt6)[0, 1] / np.var(f_mkt6, ddof=1)    # 样本 Cov(r, f) / Var(f)
resid6 = r_brk6 - X6 @ np.array([a_brk, beta_brk])             # (6, 2) @ (2,) -> (6,)
print(f"斜率 {beta_brk:.4f}，截距 {a_brk:.4%}，Cov / Var = {slope_cov:.4f}")
print("残差（单位：%）：", (resid6 * 100).round(2) + 0.0)       # 加 0.0 把 -0.0 显示为 0.0
assert np.isclose(beta_brk, 1.5) and np.isclose(a_brk, 0.002) and np.isclose(slope_cov, beta_brk)
assert np.allclose(np.abs(resid6), [0.005, 0.005, 0, 0, 0.005, 0.005])
```

### 风险溢价 $\bv\lambda$

**白话。** 风险溢价是承担一单位某种因子暴露所得到的平均补偿，也就是风险的“单价”。如果一只股票的期望超额收益完全来自风险补偿，它就等于“承担了多少风险（载荷）”乘以“每单位风险的价格（风险溢价）”，再对所有因子加总。

**例子。** 假设市场、规模、价值三个因子的月度风险溢价分别为 0.6%、0.3%、0.4%（这恰好是本章模拟器默认的因子均值 $\bv\mu$）。上表四只股票的期望超额收益为

- 白酒龙头：$0.9 \times 0.6\% - 0.4 \times 0.3\% - 0.3 \times 0.4\% = 0.30\%$；
- 券商股：$1.5 \times 0.6\% + 0.2 \times 0.3\% + 0.1 \times 0.4\% = 1.00\%$；
- 大盘银行股：$0.7 \times 0.6\% - 0.5 \times 0.3\% + 0.8 \times 0.4\% = 0.59\%$；
- 小盘股：$1.1 \times 0.6\% + 1.2 \times 0.3\% - 0.2 \times 0.4\% = 0.94\%$。

券商股的期望收益最高，因为它承担的市场风险最多；小盘股的期望收益中，市场部分 $1.1 \times 0.6\% = 0.66\%$ 仍是大头，但规模暴露又额外带来 $1.2 \times 0.3\% = 0.36\%$，使它的市场载荷虽然比券商股低，期望收益却与券商股接近；大盘银行股的市场载荷最低，但正的价值载荷带来了补偿；白酒龙头的期望超额收益最低，因为它对规模与价值的载荷都为负，在小盘股、价值股表现差的月份它反而相对较好，起到一定的对冲作用，投资者因此愿意接受较低的期望收益。

$\E[r_i] = \bv\beta_i'\bv\lambda$ 的直观含义是：**在 β 定价模型中，股票之间期望收益的差别全部来自载荷的差别。** 两只载荷相同的股票，不论行业、题材、单只股票的总波动有多大，期望超额收益都相同；异质性收益可以通过分散化消除，得不到补偿。载荷为 $\mathbf 0$ 的资产（例如无风险资产）期望超额收益为 0，所以式中没有截距。还要注意，风险溢价是长期平均的补偿，某个月实际实现的因子收益（如上例的 $3\%$、$3\%$、$-1\%$）可以与它相差很远。

**公式。**

```{math}
:enumerated: false
\E[r_i] = \underbrace{\bv\beta_i'}_{1\times K}\underbrace{\bv\lambda}_{K\times 1} = \sum_{k=1}^{K} \beta_{k,i}\lambda_k, \qquad \bv\lambda \in \Real^K .
```

当因子本身是超额收益、并且模型为因子自身正确定价（因子的定价误差为 0）时，风险溢价就是因子的均值：$\lambda_k = \E[f_{k,t}]$。原因是把上式用于第 $k$ 个因子本身：它对自己的载荷为 1、对其他因子的载荷为 0、定价误差为 0，于是 $\E[f_{k,t}] = \lambda_k$。不可交易的因子（例如消费增长、宏观变量）不是资产收益，不能套用这一推理，$\bv\lambda$ 一般不等于因子的均值，需要另行估计（第 1、4、10 章）。本章模型中因子都是超额收益，$\bv\lambda = \bv\mu$，对应参数 `mkt.params.factor_mean`；载荷随特征变化时，上式的条件版本是 $\E_{t-1}[r_{i,t}] = \alpha_{i,t} + \bv\beta_{i,t}'\bv\lambda$，其中 $\alpha_{i,t}$ 是下文的定价误差（证明见[习题 %s](#ex-ch0-beta)）。

```{code-cell} python
lam = np.asarray(MarketParams().factor_mean)       # λ: (K,) = (3,)，即模拟器默认的 μ
exp_ret = B_ex @ lam                                # (4, 3) @ (3,) -> (4,)：E[r_i] = β_i'λ
print("期望超额收益 β_i'λ：", exp_ret.round(4))
assert np.allclose(lam, [0.006, 0.003, 0.004])
assert np.allclose(exp_ret, [0.0030, 0.0100, 0.0059, 0.0094])
```

### 特征

**白话。** 特征是在期初就能观察到的股票属性，例如市值、账面市值比、动量（通常是过去 12 个月、跳过最近 1 个月的累计收益）、换手率。特征本身不是收益，而是研究者据以分组、预测或解释收益的期初信息。不同特征的单位不同，常见做法是每期在横截面上把它标准化：减去当期全部股票的均值，再除以标准差。

**例子。** 假设某个月全部股票对数市值的横截面均值为 23.0、标准差为 1.0，一只小盘股的对数市值为 21.2，则它标准化后的规模特征为 $(21.2 - 23.0) / 1.0 = -1.8$，即比平均水平小 1.8 个标准差。本章模拟器中的特征服从 $N(\mathbf 0, \mathbf I_L)$，正是这种标准化之后的尺度：$z = 2$ 表示比平均水平高 2 个标准差。

**“特征决定载荷”是什么意思。** 一只股票的载荷不是生来固定的，而是随它的特征变化：一家公司从小盘股成长为大盘股，它对规模因子的载荷就会下降。传统做法是给每只股票估计一个不随时间变化的 $\bv\beta_i$，例如用过去 60 个月做时间序列回归。这样做有三个问题：公司变了，固定的 β 跟不上；单只股票的时间序列短、噪声大；$N$ 只股票要估计 $N \times K$ 个参数，新上市的股票还没有历史数据。“特征决定载荷”的做法则假定载荷是期初特征的函数，函数的形式对所有股票相同：本章取线性函数 $\bv\beta_{i,t} = \bv\Gamma'\tilde{\mathbf z}_{i,t-1}$，只需估计 $(L+1) \times K$ 的矩阵 $\bv\Gamma$，并且可以用全部股票共同估计；股票的特征一变，它的载荷随之改变，$\bv\Gamma$ 却不变。这是 IPCA 模型的设定 [@kelly2019characteristics]。

用模拟器的默认 $\bv\Gamma$ 举例。默认参数下 $L = 5$、$K = 3$，$\bv\Gamma$ 是 $6\times 3$ 矩阵：常数项一行为 $(1, 0, 0)$，$z_1$ 一行为 $(0, 0.5, 0)$，$z_2$ 一行为 $(0, 0, 0.5)$，其余三行为 0。于是

```{math}
:enumerated: false
\bv\beta_{i,t} = \bv\Gamma'\tilde{\mathbf z}_{i,t-1} = \left(1,\ 0.5\, z_{1,i,t-1},\ 0.5\, z_{2,i,t-1}\right)' .
```

设股票 A 的 $z_1 = 2$，股票 B 的 $z_1 = -2$，两者的其他特征都为 0。则 A 的载荷为 $(1, 1, 0)'$，B 的载荷为 $(1, -1, 0)'$：二者对第一个因子的载荷相同，对第二个因子的载荷相反。若本期因子为 $\mathbf f_t = (4\%, 2\%, -1\%)'$，A 的因子部分为 $4\% + 1 \times 2\% = 6\%$，B 为 $4\% - 1 \times 2\% = 2\%$；第三个因子的 $-1\%$ 对两者都没有影响，因为它们的 $z_2 = 0$。如果下个月 A 的 $z_1$ 降到 0，它对第二个因子的载荷也随之降到 0，而 $\bv\Gamma$ 没有变。

**公式。** 特征向量 $\mathbf z_{i,t-1} \in \Real^L$，带常数项的版本 $\tilde{\mathbf z}_{i,t-1} = (1, \mathbf z_{i,t-1}')' \in \Real^{L+1}$，载荷 $\bv\beta_{i,t} = \bv\Gamma'\tilde{\mathbf z}_{i,t-1}$（$K\times(L+1)$ 乘 $(L+1)\times 1$，得 $K\times 1$），见式 {eq}`eq-ch0-char` 与 {eq}`eq-ch0-beta`。面板中特征对应 `panel["z1"]`……`panel["zL"]` 列，$\bv\Gamma$ 对应 `mkt.gamma`。

```{code-cell} python
size_z = (21.2 - 23.0) / 1.0                               # 对数市值 21.2；横截面均值 23.0、标准差 1.0（假设数字）
print(f"标准化的规模特征：{size_z:.1f}")
assert np.isclose(size_z, -1.8)

gamma0 = MarketParams().resolved_gamma()                    # 默认 Γ: (L+1, K) = (6, 3)
zt_AB = np.array([[1.0,  2.0, 0.0, 0.0, 0.0, 0.0],         # 股票 A 的 z̃'：常数项 1，z1 = 2，其余特征为 0
                  [1.0, -2.0, 0.0, 0.0, 0.0, 0.0]])        # 股票 B：z1 = -2；zt_AB: (2, L+1) = (2, 6)
beta_AB = zt_AB @ gamma0                                    # (2, 6) @ (6, 3) -> (2, 3)：每行 β_i' = z̃_i'Γ
f_AB = np.array([0.04, 0.02, -0.01])                        # f_t: (3,)
part_AB = beta_AB @ f_AB                                    # (2, 3) @ (3,) -> (2,)：β_i'f_t
print("默认 Γ：\n", gamma0)
print("A、B 的载荷（每行一只股票）：\n", beta_AB)
print("因子部分：", part_AB.round(4))
gamma_expected = np.zeros((6, 3))                           # 正文描述的默认 Γ：常数项、z1、z2 三行非零，其余三行为 0
gamma_expected[0, 0], gamma_expected[1, 1], gamma_expected[2, 2] = 1.0, 0.5, 0.5
assert np.array_equal(gamma0, gamma_expected)
assert np.allclose(beta_AB, [[1, 1, 0], [1, -1, 0]]) and np.allclose(part_AB, [0.06, 0.02])
```

### 定价误差 $\alpha$

**白话。** 定价误差是一只股票的期望超额收益中，载荷与风险溢价解释不了的那一部分：实际的期望收益减去“应得的”风险补偿 $\bv\beta_i'\bv\lambda$。

**例子。** 上面的小盘股按载荷算出的风险补偿是每月 0.94%。假设它长期的平均超额收益是每月 1.20%，那么相对于这个三因子模型，它的定价误差为 $1.20\% - 0.94\% = 0.26\%$：每月比它承担的风险所应得的多 0.26%。同理，前面 6 个月回归中券商股的截距 0.2%，就是它相对于单因子模型的样本定价误差（只有 6 个月，噪声很大，不能据此下任何结论）。

$\alpha \neq 0$ 可能有两种截然不同的解释。一种是**风险补偿**：模型漏掉了某个因子，这只股票暴露于这个被遗漏的风险，所谓“α”其实是对它的补偿，换一个更完整的模型就会消失。另一种是**错误定价**：投资者的认知偏差使价格偏离了风险所对应的水平，而套利受到限制，偏离无法被及时纠正，α 是风险无法解释的收益。例如一只股票被过度乐观的投资者追捧而高估，A 股融券困难，看空的投资者难以通过卖空把价格压回去，高估可以持续，此后的平均收益偏低，表现为负的 α。A 股的壳价值则不宜简单归入任何一类：它源于发行上市管制，是一种与基本面无关的价值成分，既不是普通的风险补偿，也不能简单看作错误定价（见“A 股视角”一节、@liu2019size 与第 16、17 章）。区分这两种解释往往很难，是第 8 章的主题。这里只需记住两点：α 总是相对于某一个选定的因子模型而言的；一个特征能预测收益，并不等于它带来了 α（见“常见误区”一节的误区四）。

**公式。**

```{math}
:enumerated: false
\alpha_i = \E[r_i] - \bv\beta_i'\bv\lambda .
```

因子是超额收益时，它也就是时间序列回归 $r_{i,t} = a_i + \bv\beta_i'\mathbf f_t + u_{i,t}$ 的截距 $a_i$：对回归式取期望，$\E[r_i] = a_i + \bv\beta_i'\E[\mathbf f_t] = a_i + \bv\beta_i'\bv\lambda$。本章模型中定价误差由特征决定，$\alpha_{i,t} = \bv\theta'\mathbf z_{i,t-1}$，见式 {eq}`eq-ch0-alpha`，对应 `panel["alpha"]` 列，系数 $\bv\theta$ 对应 `mkt.theta`。默认 $\bv\theta = \mathbf 0$，即没有错误定价；0.6 节取 $\theta_1 = 0.01$，即 $z_1$ 每高一个标准差，定价误差每月高 1 个百分点。

```{code-cell} python
avg_small = 0.012                                  # 小盘股的长期平均超额收益（假设数字）
comp_small = B_ex[3] @ lam                         # 风险补偿 β'λ：(3,) @ (3,) -> 标量
alpha_small = avg_small - comp_small               # α = E[r] − β'λ
print(f"风险补偿 β'λ = {comp_small:.2%}，定价误差 α = {alpha_small:.2%}")
assert np.isclose(comp_small, 0.0094) and np.isclose(alpha_small, 0.0026)
```

### 随机贴现因子

**白话。** 同样是一块钱，在不同的未来状态里价值不同。股市暴跌、经济不景气、收入可能下降的“坏时候”，多一块钱可以救急，因而更值钱；人人都在赚钱的“好时候”，多一块钱就没那么重要。随机贴现因子（SDF）给未来的每一个状态标出一个数 $m$：那个状态下一块钱在今天的价格（状态价格），除以该状态发生的概率。$m$ 越大，那个状态越是坏时候，那里的一块钱越值钱。一项资产的价格，就是它在各个状态下的支付按 $m$ 折算后的平均值。

**例子。** 假设下个月只有两种等可能的状态：大盘上涨的“好时候”与大盘下跌的“坏时候”，SDF 分别为 0.8 与 1.2（假设数字，二者的平均值为 1）。考虑三项资产：

| 资产（假设） | 好时候的支付 | 坏时候的支付 | 期望支付 | 价格 $\E[mx]$ | 期望收益率 |
|---|---|---|---|---|---|
| 无风险债券 | 100 | 100 | 100 | 100 | 0 |
| 券商股 | 115 | 85 | 100 | 97 | 约 3.09% |
| 大盘银行股 | 104 | 98 | 101 | 100.4 | 约 0.60% |

以券商股为例，价格为 $0.5 \times 0.8 \times 115 + 0.5 \times 1.2 \times 85 = 46 + 51 = 97$ 元，期望收益率为 $100 / 97 - 1 \approx 3.09\%$。券商股在好时候支付多、在坏时候支付少，而好时候的钱不值钱、坏时候的钱值钱，所以投资者只肯出 97 元买一个期望支付为 100 元的资产，它的期望收益因此高于无风险收益，高出的部分就是这项资产的风险溢价（期望超额收益），不要与每单位载荷的价格 $\bv\lambda$ 混淆。大盘银行股的支付在两种状态下相差不大，期望超额收益也小。本例中无风险收益为 0，是因为 SDF 的平均值取了 1，与模拟器的归一化 $\E[m_t] = 1$ 相同。

**公式。** 支付为 $x$ 的资产，价格为

```{math}
:enumerated: false
P = \E[m x] = \sum_{s} \pi_s m_s x_s ,
```

其中 $\pi_s$ 是状态 $s$ 的概率，$m_s$ 与 $x_s$ 是该状态下的 SDF 与支付，都是标量。一项“只在状态 $s$ 支付一块钱”的资产，今天的价格是 $\pi_s m_s$，所以 $m_s$ 就是状态 $s$ 下一块钱的价格除以该状态的概率。两边除以 $P$，得到对总收益的 $1 = \E[mR]$；无风险资产的支付确定，所以 $R^f = 1/\E[m]$；对超额收益，$0 = \E[mR^e]$。把 $\E[mR^e] = \E[m]\E[R^e] + \Cov(m, R^e)$ 代入后一式，

```{math}
:enumerated: false
\E[R^e] = -\frac{\Cov(m, R^e)}{\E[m]} .
```

在 $m$ 低的好时候收益高的资产，与 $m$ 负相关，期望超额收益为正；券商股正是这样。上例中券商股的 $\Cov(m, R^e) \approx -3.09\%$，$\E[m] = 1$，恰好给出 3.09% 的期望超额收益（由下方代码核对）。

这与 β 定价是同一件事的两种说法。如果 $m$ 是因子的线性函数，$\Cov(m, R^e)$ 就是资产与各个因子的协方差的线性组合，也就是由载荷决定，上式于是化为 $\E[R^e] = \bv\beta'\bv\lambda$。本章模拟市场的 SDF 正是这样的线性函数 $m_t = 1 - \mathbf b'(\mathbf f_t - \bv\mu)$，其中 $\mathbf b \in \Real^K$，见 0.4 节式 {eq}`eq-ch0-sdf`，对应 `mkt.sdf()` 与 `mkt.sdf_weights`；默认参数下 $\mathbf b$ 的各分量都为正，因子收益总体高于均值（$\mathbf b'(\mathbf f_t - \bv\mu) > 0$）的好时候 $m_t < 1$。例如 $\mathbf f_t = (4\%, 2\%, -1\%)'$ 时 $m_t \approx 0.92$，第一个因子下跌 6%、另两个因子为 0 时 $m_t \approx 1.19$。SDF 与 β 定价之间的关系，在模拟市场中由 0.5 节的定理与[习题 %s](#ex-ch0-beta) 严格证明，一般理论在第 1～4 章展开。

```{code-cell} python
prob = np.array([0.5, 0.5])                          # 好时候、坏时候的概率 π_s
m_state = np.array([0.8, 1.2])                       # 各状态的 SDF m_s（假设数字）
payoff = np.array([[100.0, 100.0],                   # 无风险债券
                   [115.0,  85.0],                   # 券商股
                   [104.0,  98.0]])                  # 大盘银行股；payoff 形状为 (3, 2)，行为资产、列为状态
price = payoff @ (prob * m_state)                    # (3, 2) @ (2,) -> (3,)：P = Σ_s π_s m_s x_s
R_state = payoff / price[:, None]                    # 各状态的总收益，广播 (3, 2) / (3, 1) -> (3, 2)
E_m = prob @ m_state                                 # E[m]
E_R = R_state @ prob                                 # (3, 2) @ (2,) -> (3,)：E[R]
R_f = 1 / E_m                                        # R^f = 1 / E[m]
cov_mR = (R_state * m_state) @ prob - E_m * E_R      # Cov(m, R^e) = E[mR] − E[m]E[R]（R^f 是常数）
print("价格：", price, "\n期望收益率：", (E_R - 1).round(4), "\nCov(m, R^e)：", cov_mR.round(4))
assert np.allclose(price, [100, 97, 100.4]) and np.isclose(R_f, 1)
assert np.allclose((R_state * m_state) @ prob, 1)                  # 每项资产都满足 1 = E[mR]
assert np.allclose(E_R - R_f, -cov_mR / E_m)                       # E[R^e] = −Cov(m, R^e) / E[m]
assert np.allclose(E_R - 1, [0, 3 / 97, 0.6 / 100.4])

# 模拟器默认参数下的 SDF：m_t = 1 − b'(f_t − μ)，b = Σ_f⁻¹μ
p_def = MarketParams()
mu_def = np.asarray(p_def.factor_mean)                             # μ: (K,)
b_def = np.linalg.solve(p_def.factor_cov, mu_def)                  # 解 Σ_f b = μ，b: (K,)
m_up = 1 - b_def @ (np.array([0.04, 0.02, -0.01]) - mu_def)        # (K,) @ (K,) -> 标量
m_down = 1 - b_def @ (np.array([-0.06, 0.0, 0.0]) - mu_def)
print(f"b = {b_def.round(3)}；m（好时候）= {m_up:.3f}，m（坏时候）= {m_down:.3f}")
assert np.all(b_def > 0) and 0.92 < m_up < 0.93 and 1.18 < m_down < 1.19
```

### 概念—符号—模拟器字段对照表

| 概念 | 符号 | 维度 | 模拟器中的字段 |
|---|---|---|---|
| 总收益 | $R_{i,t}$ | 标量 | 不单独生成 |
| 无风险利率 | $R^f_{t-1}$ | 标量 | 不单独生成；归一化 $\E[m_t] = 1$ 相当于 $R^f = 1$ |
| 超额收益 | $r_{i,t}$（即 $R^e_{i,t}$） | 标量 | `panel["exret"]` |
| 因子 | $\mathbf f_t$ | $K\times 1$ | `mkt.factors`（列 `f1`……`fK`） |
| 载荷（暴露、β） | $\bv\beta_{i,t}$ | $K\times 1$ | `panel["beta1"]`……`panel["betaK"]` |
| 风险溢价 | $\bv\lambda$（本章模型中 $= \bv\mu$） | $K\times 1$ | `mkt.params.factor_mean` |
| 特征 | $\mathbf z_{i,t-1}$ | $L\times 1$ | `panel["z1"]`……`panel["zL"]` |
| 特征到载荷的映射 | $\bv\Gamma$ | $(L+1)\times K$ | `mkt.gamma` |
| 定价误差 | $\alpha_{i,t}$ | 标量 | `panel["alpha"]` |
| 特征到定价误差的映射 | $\bv\theta$ | $L\times 1$ | `mkt.theta`（参数名为 `alpha_coef`） |
| 随机贴现因子 | $m_t$ | 标量 | `mkt.sdf()` |
| SDF 系数 | $\mathbf b$ | $K\times 1$ | `mkt.sdf_weights` |

下面的代码单元生成一个很小的模拟市场，核对表中的字段确实存在、形状与维度一致。

```{code-cell} python
tiny = simulate_market(MarketParams(n_stocks=3, n_periods=2), seed=0)
L0, K0 = tiny.params.n_chars, tiny.params.n_factors
expected_cols = {"exret", "alpha"} | {f"beta{k}" for k in range(1, K0 + 1)} | {f"z{j}" for j in range(1, L0 + 1)}
assert expected_cols <= set(tiny.panel.columns)
assert list(tiny.factors.columns) == [f"f{k}" for k in range(1, K0 + 1)]
assert tiny.gamma.shape == (L0 + 1, K0) and tiny.theta.shape == (L0,)          # Γ: (L+1, K)，θ: (L,)
assert tiny.sdf_weights.shape == (K0,) and tiny.sdf().shape == (tiny.params.n_periods,)
print(sorted(tiny.panel.columns))
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

# 载荷与定价误差是期初特征的确定函数：式 (0.3) 与 (0.4)
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
assert len(out_right) >= 1 and np.all((out_right > 2000) & (out_right < 3500))
assert np.all(z_mf1[(T_grid >= 1200) & (T_grid <= 15000)] > 0.5)          # 左图中段持续偏高
```

两个样本均值都随 $T$ 增大而收敛到理论值：样本值与理论值的最大偏离，从 $T \leq 200$ 的小样本到 $T \geq 5000$ 的大样本缩小到五分之一以下（具体数值见图下输出），与偏离按 $1/\sqrt{T}$ 收窄的速度相符——$T$ 扩大 25 倍，$1/\sqrt{T}$ 缩小到五分之一。但两条路径并不是一直贴着理论值：左图的路径在 $T < 200$ 的小样本中始终低于理论值 0，而在 $T$ 约 1200～15000 的很长一段区间里又持续高于 0、靠近带的上沿，但始终没有越出灰色带；右图的路径在 $T$ 约 2000～3500 之间持续偏高，其中一个格点越出了带的上沿，随后回到带内并逐渐贴近理论值。全样本的标准化偏差都小于 1（以上描述都由代码中的 `assert` 守护）。路径在某些区段偏离较多、甚至越出带，是预期之中的现象，并不说明理论有误。灰色带是逐点的“理论值 ± 2 倍标准误”，只保证在任意一个固定的 $T$ 上，样本均值落在带外的概率约为 5%；它不是同时成立的置信带，不保证整条路径都在带内。而同一条路径上相邻的累计均值共享绝大部分数据、高度序列相关，一旦在某个 $T$ 偏高或偏低，就会持续一段时间（左图先持续偏低、后持续偏高就是这样），所以整条路径在某些区段越出逐点的带，比 60 个独立抽样时常见得多。本节开头的表格与五分位分组的结果也都与理论一致。由此也可以读出本书所有模拟验证的基本逻辑：**单次模拟只给出带有抽样误差的样本值，判断“是否恢复了真值”时，要把偏离与标准误相比较，而不是要求两者完全相等。**

## A 股视角

本章的模拟市场刻意保持简单，以便先把定价方程、时点约定与“已知真值”的检验逻辑讲清楚。与真实的 A 股市场相比，它至少省略了以下几类特征：

1. **涨跌停。** A 股对单日涨跌幅设有限制，不同板块与 ST 股票的限制比例不同，并随制度改革调整。涨跌停会截断日收益、可能推迟价格对信息的反映，也让触及涨跌停的股票在当日实际上难以按收盘价成交。第 15 章在 `apsim.frictions` 中加入这一机制，第 17 章讨论它对资产定价检验的影响。
2. **停牌。** 停牌期间没有成交价，复牌后的收益集中体现了停牌期间积累的信息。月度数据中如何处理停牌股票的收益与特征，会直接影响组合收益，详见第 15 章。
3. **卖空限制。** 本章把因子当作可以无成本构造的多空组合。A 股融券规模小、成本高，许多“空头腿”在现实中难以实现，这既影响因子收益的可实现性（第 26 章），也是错误定价能够持续存在的一个原因（第 8 章）。
4. **壳价值。** 在发行上市受到严格管制的时期，小市值公司作为借壳上市的潜在对象而具有额外价值，这使市值最小的股票的收益不能用一般的风险补偿解释。Liu、Stambaugh 与 Yuan 构造中国市场的规模与价值因子时，因此剔除了市值最小的 30% 的股票 [@liu2019size]。第 16、17 章讨论这一处理。
5. **财报披露滞后。** 会计期末的财务数据要到数月之后才公开（例如年报通常在次年 4 月底前披露）。如果按会计期末而不是实际披露日对齐，特征 $\mathbf z_{i,t-1}$ 中就混入了第 $t-1$ 期末尚不可得的信息，即本章强调的时点错位。正式研究必须使用带有公告日期的时点数据，第 15 章给出对齐规则。

此外，模拟器中的因子收益是独立同分布的正态变量，特征服从相互独立的正态 AR(1)，异质性冲击在股票之间不相关，股票数量在样本期内不变。真实数据中的厚尾、波动聚集、行业内的同期相关、新股上市与退市，都会在相应章节逐步加入。中国股市中机器学习方法的实证表现，可先参阅 Leippold、Wang 与 Zhou 的研究 [@leippold2022machine]，第四部分会详细讨论。

## 模拟实验：用交互回归恢复 Γ

### 方法

把式 {eq}`eq-ch0-beta` 代入式 {eq}`eq-ch0-ret`，在 $\bv\theta = \mathbf 0$ 时：

$$
r_{i,t} = \underbrace{\tilde{\mathbf z}_{i,t-1}'}_{1\times(L+1)} \underbrace{\bv\Gamma}_{(L+1)\times K} \underbrace{\mathbf f_t}_{K\times 1} + \varepsilon_{i,t} = \sum_{j=0}^{L} \sum_{k=1}^{K} \Gamma_{j,k}\, \left(\tilde z_{j,i,t-1} f_{k,t}\right) + \varepsilon_{i,t},
$$ (eq-ch0-interact)

其中 $\tilde z_{0,i,t-1} = 1$，$\tilde z_{j,i,t-1} = z_{j,i,t-1}$（$j \geq 1$）。这是一个以 $(L+1)K$ 个“特征 × 因子”交互项为解释变量、以 $\bv\Gamma$ 的各元素为系数的线性回归。

**写成标准的回归形式。** 要把 $\bv\Gamma$ 的元素排成一个系数向量，需要用到 vec 算子与 Kronecker 积。$\operatorname{vec}(\mathbf A)$ 把矩阵 $\mathbf A$ 的各列从左到右依次叠成一个长的列向量（列优先）；Kronecker 积 $\mathbf a \otimes \mathbf c$ 把 $\mathbf a$ 的每个元素 $a_j$ 换成块 $a_j \mathbf c$。二者由恒等式

$$
\operatorname{vec}(\mathbf A \mathbf B \mathbf C) = (\mathbf C' \otimes \mathbf A)\operatorname{vec}(\mathbf B)
$$ (eq-ch0-vec)

联系起来（$\mathbf A$、$\mathbf B$、$\mathbf C$ 为任意可以相乘的矩阵）。标量 $\tilde{\mathbf z}_{i,t-1}'\bv\Gamma\mathbf f_t$ 等于它的转置 $\mathbf f_t'\bv\Gamma'\tilde{\mathbf z}_{i,t-1}$，取 $\mathbf A = \mathbf f_t'$（$1\times K$）、$\mathbf B = \bv\Gamma'$（$K\times(L+1)$）、$\mathbf C = \tilde{\mathbf z}_{i,t-1}$（$(L+1)\times 1$），一个标量的 vec 就是它自己，于是

$$
r_{i,t} = \underbrace{(\tilde{\mathbf z}_{i,t-1} \otimes \mathbf f_t)'}_{1\times(L+1)K}\; \underbrace{\operatorname{vec}(\bv\Gamma')}_{(L+1)K\times 1} + \varepsilon_{i,t} .
$$ (eq-ch0-kron)

这里用到 $(\tilde{\mathbf z}_{i,t-1}' \otimes \mathbf f_t') = (\tilde{\mathbf z}_{i,t-1} \otimes \mathbf f_t)'$。系数向量 $\operatorname{vec}(\bv\Gamma')$ 按列叠放 $\bv\Gamma'$，也就是按行叠放 $\bv\Gamma$：先是第 0 行 $(\Gamma_{0,1}, \dots, \Gamma_{0,K})$，再是第 1 行，依此类推，$j$ 在外层、$k$ 在内层；设计矩阵的第 $(i,t)$ 行 $(\tilde{\mathbf z}_{i,t-1} \otimes \mathbf f_t)'$ 的元素也按同样的顺序排列为 $\tilde z_{j,i,t-1} f_{k,t}$。把全部 $TN$ 个观测按行叠成 $\mathbf X \in \Real^{TN\times(L+1)K}$ 与 $\mathbf y \in \Real^{TN}$，因子可观测时，对 $\mathbf y$ 关于 $\mathbf X$ 做最小二乘，就能估计 $\operatorname{vec}(\bv\Gamma')$。

**与代码的对应。** 下面的 `interaction_design` 用广播构造 $\mathbf X$：`zt[:, :, None] * ff[:, None, :]` 的形状为 `(TN, L+1, K)`，其 `[n, j, k]` 元素是 $\tilde z_j f_k$；NumPy 的 `reshape` 默认按行优先（C 顺序）展开，最后一维 `k` 变化最快，所以展开后每一行正是 $(\tilde{\mathbf z} \otimes \mathbf f)'$。同理，`gamma.ravel()`（行优先）就是 $\operatorname{vec}(\bv\Gamma')$，而列优先的 `gamma.ravel(order="F")` 才是 $\operatorname{vec}(\bv\Gamma)$；系数向量用 `coef.reshape(L+1, K)` 还原为 $\hat{\bv\Gamma}$。数学上的 vec 是列优先，NumPy 的默认展开是行优先，两者差一个转置，这是最容易出错的地方，下面的代码单元用 `np.kron` 与 `assert` 核对这一对应。

由于 $\varepsilon_{i,t}$ 与 $(\mathbf z_{i,t-1}, \mathbf f_t)$ 独立，解释变量严格外生，最小二乘估计量无偏，其条件协方差矩阵为 $\sigma_\varepsilon^2 (\mathbf X'\mathbf X)^{-1}$（$(L+1)K \times (L+1)K$）。函数 `ols` 用 `lstsq` 求系数、用 `solve` 求 $(\mathbf X'\mathbf X)^{-1}$，而不调用 `inv`。下面用默认参数（$N = 500$、$T = 240$）检验这一点。

```{code-cell} python
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


X, _ = interaction_design(mkt)
# 核对 reshape 顺序：第 n 行等于 z̃ ⊗ f，且 X @ vec(Γ') 等于 β'f
zt_row = np.r_[1.0, mkt.panel.iloc[7][[f"z{j}" for j in range(1, L + 1)]].to_numpy()]
f_row7 = mkt.factors.loc[mkt.panel.index[7][0]].to_numpy()
assert np.allclose(X[7], np.kron(zt_row, f_row7))
assert np.allclose(X @ mkt.gamma.ravel(), np.sum(beta * f_row, axis=1))       # (TN, (L+1)K) @ ((L+1)K,)
assert not np.allclose(X @ mkt.gamma.ravel(order="F"), np.sum(beta * f_row, axis=1))

coef, se = ols(X, mkt.panel["exret"].to_numpy())
names = [f"Γ[{j},{k + 1}]" for j in range(params.n_chars + 1) for k in range(K)]
table = pd.DataFrame(
    {"真值": mkt.gamma.ravel(), "估计值": coef, "标准误": se, "标准化偏差": (coef - mkt.gamma.ravel()) / se},
    index=names,
)
print(table.round(4).to_string())
n_big = int(np.sum(np.abs(table["标准化偏差"]) > 2))
print(f"标准化偏差绝对值超过 2 的系数：{n_big} 个（标准正态分布下 18 个中平均约 {18 * 0.0455:.1f} 个）")
assert np.all(np.abs(table["标准化偏差"]) < 4)
```

表中“标准化偏差”为 $(\hat\Gamma_{j,k} - \Gamma_{j,k}) / \text{se}$。18 个系数的估计值都落在真值附近（`assert` 保证每个标准化偏差的绝对值小于 4）。单次模拟中，绝对值超过 2 的标准化偏差有几个，取决于种子（见上方输出）；若估计量无偏、标准误正确，18 个系数中平均约有 0.8 个。单次模拟无法区分“抽样误差”与“标准误算错了”，要靠重复模拟：如果估计量无偏、标准误正确，标准化偏差在多次重复中应当近似服从标准正态分布。

```{code-cell} python
zs = []
for r in range(30):
    rep = simulate_market(params, seed=1000 + r)
    Xr, _ = interaction_design(rep)
    cr, ser = ols(Xr, rep.panel["exret"].to_numpy())
    zs.append((cr - rep.gamma.ravel()) / ser)
zs = np.concatenate(zs)
frac_big = np.mean(np.abs(zs) > 2)
print(f"30 次重复、共 {zs.size} 个标准化偏差：均值 {zs.mean():.3f}，标准差 {zs.std():.3f}，"
      f"绝对值超过 2 的比例 {frac_big:.3f}（标准正态分布下为 0.046）")
# 540 个独立的标准正态抽样：均值、标准差与比例的抽样误差分别约为 0.043、0.030、0.009
assert abs(zs.mean()) < 0.13 and abs(zs.std() - 1) < 0.09 and abs(frac_big - 0.0455) < 0.027
se_grid = se.reshape(-1, K)                                                    # (L+1, K)，与 Γ 同形
assert np.all(se_grid[:, 0] < se_grid[:, 1:].min(axis=1))
```

重复模拟的结果与标准正态分布相符（代码中的 `assert` 要求均值、标准差与大偏差比例都落在约 3 倍抽样误差以内）：估计量无偏，同方差标准误也是正确的，单次模拟中大偏差的个数只反映抽样误差。另外，第一个因子一列（$k = 1$）的标准误小于另外两列：$\sigma_1 = 0.05$ 较大，交互项 $\tilde z_j f_{1,t}$ 的变化也更大，因而包含更多关于 $\Gamma_{j,1}$ 的信息。

### 当存在错误定价时

如果 $\bv\theta \neq \mathbf 0$，式 {eq}`eq-ch0-interact` 遗漏了 $\alpha_{i,t} = \bv\theta' \mathbf z_{i,t-1}$。遗漏项与解释变量相关吗？交互项 $z_{j,i,t-1} f_{k,t}$ 与 $z_{j,i,t-1}$ 的协方差为 $\E[z_{j,i,t-1}^2]\,\E[f_{k,t}] = \mu_k \neq 0$，所以答案是肯定的：只用交互项的回归有遗漏变量偏误。

偏误的大小可以算出来。记 $\mathbf F = (\mathbf f_1, \dots, \mathbf f_T)' \in \Real^{T\times K}$（第 $t$ 行为 $\mathbf f_t'$），$\mathbf 1_T$ 为 $T$ 维全 1 向量。以因子路径 $\mathbf F$ 为条件，令 $N \to \infty$。每一期的横截面平均满足 $N^{-1}\sum_i \tilde{\mathbf z}_{i,t-1} \tilde{\mathbf z}_{i,t-1}' \to \mathbf I_{L+1}$、$N^{-1}\sum_i \tilde{\mathbf z}_{i,t-1} \mathbf z_{i,t-1}' \bv\theta \to (0, \bv\theta')'$。由式 {eq}`eq-ch0-kron`，设计矩阵的行是 $(\tilde{\mathbf z}_{i,t-1} \otimes \mathbf f_t)'$；利用 Kronecker 积的混合乘积规则 $(\mathbf A \otimes \mathbf B)(\mathbf C \otimes \mathbf D) = \mathbf A\mathbf C \otimes \mathbf B\mathbf D$，

```{math}
:enumerated: false
\frac{1}{N}\mathbf X'\mathbf X = \sum_t \Bigl(\frac{1}{N}\sum_i \tilde{\mathbf z}_{i,t-1}\tilde{\mathbf z}_{i,t-1}'\Bigr) \otimes \mathbf f_t\mathbf f_t' \to \mathbf I_{L+1} \otimes \mathbf F'\mathbf F, \qquad
\frac{1}{N}\mathbf X'\bv\alpha \to \sum_t \begin{pmatrix} 0 \\ \bv\theta \end{pmatrix} \otimes \mathbf f_t = \begin{pmatrix} 0 \\ \bv\theta \end{pmatrix} \otimes \mathbf F'\mathbf 1_T ,
```

其中 $\bv\alpha$ 是全部 $TN$ 个 $\alpha_{i,t}$ 叠成的向量；前者为 $(L+1)K \times (L+1)K$，后者为 $(L+1)K \times 1$。于是

$$
\operatorname{plim}_{N\to\infty} \operatorname{vec}(\hat{\bv\Gamma}')
= \operatorname{vec}(\bv\Gamma') + \left[\mathbf I_{L+1} \otimes \mathbf F'\mathbf F\right]^{-1} \left[\begin{pmatrix} 0 \\ \bv\theta \end{pmatrix} \otimes \mathbf F'\mathbf 1_T\right]
= \operatorname{vec}(\bv\Gamma') + \begin{pmatrix} 0 \\ \bv\theta \end{pmatrix} \otimes \mathbf c ,
\qquad \underbrace{\mathbf c}_{K\times 1} = \underbrace{(\mathbf F'\mathbf F)^{-1}}_{K\times K} \underbrace{\mathbf F'\mathbf 1_T}_{K\times 1} ,
$$ (eq-ch0-bias)

第二个等号再次用到混合乘积规则，以及 $[\mathbf I_{L+1} \otimes \mathbf F'\mathbf F]^{-1} = \mathbf I_{L+1} \otimes (\mathbf F'\mathbf F)^{-1}$。按 $\operatorname{vec}(\bv\Gamma')$ 的排列顺序读，常数项一行没有偏误，第 $j$ 个特征一行的偏误是 $\theta_j \mathbf c'$；$\mathbf c$ 恰好是把常数 1 对因子做无截距回归的系数。当 $T \to \infty$ 时，$\mathbf F'\mathbf F/T \to \bv\Sigma_f + \bv\mu\bv\mu'$，$\mathbf F'\mathbf 1_T/T \to \bv\mu$，由 Sherman–Morrison 公式 $(\mathbf A + \mathbf u\mathbf v')^{-1} = \mathbf A^{-1} - \mathbf A^{-1}\mathbf u\mathbf v'\mathbf A^{-1}/(1 + \mathbf v'\mathbf A^{-1}\mathbf u)$（取 $\mathbf A = \bv\Sigma_f$，$\mathbf u = \mathbf v = \bv\mu$），

$$
\mathbf c \to (\bv\Sigma_f + \bv\mu\bv\mu')^{-1}\bv\mu = \bv\Sigma_f^{-1}\bv\mu - \frac{\bv\Sigma_f^{-1}\bv\mu\,(\bv\mu'\bv\Sigma_f^{-1}\bv\mu)}{1 + \bv\mu'\bv\Sigma_f^{-1}\bv\mu} = \frac{\bv\Sigma_f^{-1}\bv\mu}{1 + \bv\mu'\bv\Sigma_f^{-1}\bv\mu} = \frac{\mathbf b}{1 + \bv\mu'\bv\Sigma_f^{-1}\bv\mu} ,
$$ (eq-ch0-bias-limit)

分母中的 $\bv\mu'\bv\Sigma_f^{-1}\bv\mu$ 是标量。偏误的方向正是 SDF 系数 $\mathbf b$ 的方向。解决办法很简单：在回归中加入特征本身作为解释变量，其系数估计的就是 $\bv\theta$。IPCA 在载荷之外允许一个由特征驱动的截距，也是出于同样的考虑 [@kelly2019characteristics]。下面的代码在 $\bv\theta = (0.01, 0, 0, 0, 0)'$ 下比较两种回归。两者使用同一份数据，二者之差几乎不含异质性冲击带来的抽样噪声，可以直接与式 {eq}`eq-ch0-bias` 的 $\theta_1 \mathbf c'$ 比较。

```{code-cell} python
mis_params = MarketParams(alpha_coef=(0.01, 0.0, 0.0, 0.0, 0.0))
mis = simulate_market(mis_params, seed=0)
y = mis.panel["exret"].to_numpy()
X, Z = interaction_design(mis)                              # X: (TN, (L+1)K)，Z: (TN, L)
coef_short, _ = ols(X, y)                                  # 只有交互项
coef_long, se_long = ols(np.column_stack([X, Z]), y)       # 交互项 + 特征本身：(TN, (L+1)K + L)
G_short = coef_short.reshape(-1, K)                        # vec(Γ̂') -> Γ̂：((L+1)K,) -> (L+1, K)
G_long = coef_long[: X.shape[1]].reshape(-1, K)
theta_hat, theta_se = coef_long[X.shape[1]:], se_long[X.shape[1]:]

F_path = mis.factors.to_numpy()                            # F: (T, K)
c = np.linalg.solve(F_path.T @ F_path, F_path.sum(axis=0)) # 解 (F'F) c = F'1_T：(K, K) c = (K,)
print("z1 一行：只用交互项", G_short[1].round(4), "| 加入特征", G_long[1].round(4), "| 真值", mis.gamma[1])
print("常数项一行之差      ", (G_short[0] - G_long[0]).round(4))
print("z1 一行之差         ", (G_short[1] - G_long[1]).round(4))
print("偏误公式预测 θ1·c   ", (0.01 * c).round(4))
print("θ 的估计", theta_hat.round(4), "标准误", theta_se.round(4), "真值", mis.theta)

# Sherman–Morrison：T → ∞ 时 c 的极限
mu_v, Sig = np.asarray(mis_params.factor_mean), mis_params.factor_cov
lhs = np.linalg.solve(Sig + np.outer(mu_v, mu_v), mu_v)    # (Σ_f + μμ')⁻¹μ
rhs = mis.sdf_weights / (1 + mu_v @ mis.sdf_weights)       # b / (1 + μ'Σ_f⁻¹μ)
print("c 的极限", lhs.round(4), "=", rhs.round(4), "；本样本的 c", c.round(4))
assert np.allclose(lhs, rhs) and np.allclose(G_short[0] - G_long[0], 0, atol=0.003)   # 常数项一行几乎无偏误
rel_gap = (G_short[1] - G_long[1]) / (0.01 * c) - 1                # 实际之差相对公式预测的偏离
c_dev = c / lhs - 1                                        # 本样本 c 相对极限的偏离
print("之差相对公式预测的偏离", rel_gap.round(3), "；c 相对极限的偏离", c_dev.round(3))
assert np.all(np.abs(rel_gap) < 0.15)
f_bar_mis = F_path.mean(axis=0)
print("本样本的因子均值", f_bar_mis.round(4), "真值", mis_params.factor_mean)
assert np.all(c_dev > 0.15) and np.argmax(c_dev) == 1 and 0.3 < c_dev[1] < 0.5
assert f_bar_mis[1] > 1.2 * mu_v[1]
```

两种回归之差与式 {eq}`eq-ch0-bias` 的预测相符：$z_1$ 一行之差与 $\theta_1 \mathbf c'$ 的各分量相对差别在 15% 以内，常数项一行之差接近 0（均由 `assert` 保证）；有限 $N$ 下横截面平均与总体矩之间仍有小的差别，所以二者不完全相等。还要注意，本例中 $\mathbf c$ 与它的极限 {eq}`eq-ch0-bias-limit` 有明显差别：三个分量都比极限大，其中第二个分量偏离最大，约大四成（相对偏离见上方输出，由 `assert` 守护）。原因是 240 个月的样本中，因子样本均值本身就有可观的抽样误差——本样本中第二个因子的样本均值比真值高出两成以上——而偏误取决于样本中实际出现的因子路径，而不是总体矩。

## 常见误区

**误区一：把同期特征当作期初特征。** 用 $\mathbf z_{i,t}$ 而不是 $\mathbf z_{i,t-1}$ 去解释 $r_{i,t}$，在真实数据中可能造成严重的前视偏差：许多特征（市值、动量、换手率）与当期收益机械相关，例如第 $t$ 期末的市值本身就包含了第 $t$ 期的收益。在本章的模拟器中，特征更新的冲击 $\mathbf e_{i,t}$ 与收益无关，所以错用同期特征不会产生虚假的预测力，只会削弱真实的关系：由式 {eq}`eq-ch0-char` 与 $\mathbf e_{i,t}$ 的独立性，

```{math}
:enumerated: false
\underbrace{\E[m_t r_{i,t} \mathbf z_{i,t}]}_{L\times 1} = \phi\,\E[m_t r_{i,t} \mathbf z_{i,t-1}] + \sqrt{1-\phi^2}\,\E[m_t r_{i,t} \mathbf e_{i,t}] = \phi\,\bv\theta + \mathbf 0 .
```

这个例子说明了模拟的局限：模拟器不会自动暴露所有错误，只能暴露我们放进模型的那些机制。下面用 $\phi = 0.9$ 的长样本验证。两个样本矩都带有抽样误差，而且误差高度相关；所以除了把它们分别与理论值比较，还检验二者之差是否满足推导给出的关系：用 $\overline{m r z_1}$ 表示 $m_t r_{i,t} z_{1,i,t-1}$ 对全部 $(i,t)$ 的样本均值、$\overline{m r z_1^{+}}$ 表示把 $z_{1,i,t-1}$ 换成同期 $z_{1,i,t}$ 后的样本均值，则 $\overline{m r z_1^{+}} - \phi\, \overline{m r z_1}$ 只剩下 $\sqrt{1-\phi^2}\, \overline{m r e_1}$（$e_{1,i,t}$ 为 $\mathbf e_{i,t}$ 的第一个分量），理论值为 0。

```{code-cell} python
Z1_now = long_mkt.panel["z1"].unstack("stock").shift(-1).to_numpy()   # (T, N)，第 t 行放 z_{1,i,t}
valid = ~np.isnan(Z1_now[:, 0])                                        # 最后一期没有 z_{i,T}
now = (MR[valid] * Z1_now[valid]).mean()
lag = (MR[valid] * Z1[valid]).mean()
phi = long_params.char_persistence
print(f"用期初特征 z_(t-1)：{lag:.5f}（理论值 θ1 = {long_params.alpha_coef[0]}）")
print(f"用同期特征 z_t    ：{now:.5f}（理论值 φθ1 = {phi * 0.01:.4f}）")
resid_term = (MR[valid] * (Z1_now[valid] - phi * Z1[valid])).mean(axis=1)   # 每期横截面平均：(T-1,)
se_resid = resid_term.std() / np.sqrt(resid_term.size)
print(f"now − φ·lag = {now - phi * lag:.5f}，标准误约 {se_resid:.5f}（理论值 0）")
assert abs(now - phi * lag) < 3 * se_resid
```

二者之差与 0 的距离在 3 倍标准误以内（由 `assert` 保证）。这里的标准误同样先对每期求横截面平均再计算。

**误区二：把样本均值当作总体值。** 默认参数下第一个因子的月均值 $\mu_1 = 0.6\%$、波动率 $5\%$，240 个月样本均值的标准误为 $0.05/\sqrt{240} \approx 0.32\%$，样本均值完全可能偏离真值一半以上。更隐蔽的错误是用样本矩构造 SDF，再在同一样本中“验证”它。若 $\hat{\mathbf b} = \hat{\bv\Sigma}_f^{-1}\bar{\mathbf f}$（$\hat{\bv\Sigma}_f$ 为除以 $T$ 的样本协方差矩阵，$\bar{\mathbf f}$ 为样本均值向量），$\hat m_t = 1 - \hat{\mathbf b}'(\mathbf f_t - \bar{\mathbf f})$，则把[定理 %s](#thm-ch0-sdf) 第 2 条证明中的总体矩逐一换成样本矩，可得样本内 $T^{-1}\sum_t \hat m_t \mathbf f_t = \bar{\mathbf f} - \hat{\bv\Sigma}_f \hat{\mathbf b} = \mathbf 0$ **恒成立**，无论模型对错。这样的“检验”什么也没有检验。

```{code-cell} python
F240 = mkt.factors.to_numpy()                              # (T, K)
f_bar = F240.mean(axis=0)                                  # (K,)
Sigma_hat = np.cov(F240, rowvar=False, ddof=0)             # (K, K)
b_hat = np.linalg.solve(Sigma_hat, f_bar)                  # 解 Σ̂_f b̂ = f̄
m_hat = 1 - (F240 - f_bar) @ b_hat                         # (T, K) @ (K,) -> (T,)
in_sample = (m_hat[:, None] * F240).mean(axis=0)
true_pricing = (mkt.sdf().to_numpy()[:, None] * F240).mean(axis=0)
print("因子样本均值", f_bar.round(4), "真值", params.factor_mean, "标准误", (np.asarray(params.factor_vol) / np.sqrt(len(F240))).round(4))
print("b 的样本估计", b_hat.round(2), "真值", mkt.sdf_weights.round(2))
print("样本内 mean(m̂ f) =", in_sample)
print("真实 m 的 mean(m f) =", true_pricing.round(5))
assert np.max(np.abs(in_sample)) < 1e-12 and np.max(np.abs(true_pricing)) > 1e-4
```

样本内的 $T^{-1}\sum_t \hat m_t \mathbf f_t$ 在机器精度内为 $\mathbf 0$，而真实 SDF 在同一样本中的样本定价误差并不为 0——后者才反映了抽样误差的真实大小。真正的检验需要样本外数据，或者把估计误差计入检验统计量，这是第二部分的主题。

**误区三：忽视随机种子。** 没有固定种子的随机数每次运行都不同，书中的数字因此无法复现。为了避免这种错误，`apsim.simulate_market` 要求必须以关键字传入 `seed`，省略它或写成 `seed=None` 都会报错；确实需要不可复现的随机数时，只能显式传入一个未设种子的 `np.random.default_rng()`。反过来的错误同样常见：在蒙特卡洛循环中每次都用同一个种子，得到的“独立重复”其实完全相同，估计量的抽样方差被算成 0。

```{code-cell} python
small = MarketParams(n_stocks=50, n_periods=60)
try:
    simulate_market(small)
except TypeError as err:
    print("省略 seed：TypeError：", err)

first = simulate_market(small, seed=np.random.default_rng()).panel["exret"].iloc[0]
second = simulate_market(small, seed=np.random.default_rng()).panel["exret"].iloc[0]
print(f"未设种子的 Generator 两次运行的第一个收益：{first:.4f} 与 {second:.4f}")

same = [simulate_market(small, seed=0).factors["f1"].mean() for _ in range(5)]
diff = [simulate_market(small, seed=100 + r).factors["f1"].mean() for r in range(5)]
print("同一种子重复 5 次，f1 样本均值的标准差：", np.std(same))
print("不同种子重复 5 次，f1 样本均值的标准差：", round(float(np.std(diff)), 4))
assert np.std(same) == 0 and np.std(diff) > 0
```

（第二行的两个数字每次重新执行都会变化，这正是问题所在。）本书的约定是：单次演示用固定的整数种子；蒙特卡洛重复用“基础种子 + 重复编号”，保证各次重复互不相同，而整体可以复现。另一种同样可复现的做法，是新建一个带种子的 `rng = np.random.default_rng(2026)`（2026 为基础种子），在循环中反复传入 `seed=rng`：`simulate_market` 每次都从它派生新的子流，各次重复互不相同。

**误区四：把可预测性等同于错误定价。** 即使 $\bv\theta = \mathbf 0$，特征也能预测收益。由式 {eq}`eq-ch0-ret` 与独立性，$\E[r_{i,t} \mid \mathbf z_{i,t-1}] = \alpha_{i,t} + \bv\beta_{i,t}'\bv\mu$，而 $\bv\beta_{i,t}$ 随特征变化。默认参数下 $\E[r_{i,t} \mid \mathbf z_{i,t-1}] = \mu_1 + 0.5\mu_2\, z_{1,i,t-1} + 0.5\mu_3\, z_{2,i,t-1}$：特征 $z_1$、$z_2$ 能预测收益，完全是因为它们决定了风险暴露，这正是“特征即协方差”的含义 [@kelly2019characteristics]。判断一个能预测收益的特征是风险补偿还是错误定价，要看它的预测力能否被载荷与因子风险溢价解释，而不能只看预测力本身（第 8、23 章）。下面用 $\bv\theta = \mathbf 0$ 的长样本做 $r_{i,t}$ 对 $\tilde{\mathbf z}_{i,t-1}$ 的混合回归。由于 $\E[r_{i,t} \mid \mathbf z_{i,t-1}] = \bv\beta_{i,t}'\bv\mu = \tilde{\mathbf z}_{i,t-1}'\bv\Gamma\bv\mu$，系数的理论值为 $\bv\Gamma\bv\mu$（$(L+1)\times K$ 乘 $K\times 1$，为 $L+1$ 维向量）。

逐期回归的代码用 `reshape(n_t, n_i, -1)` 把 $TN$ 行的设计矩阵还原为“期 × 股票 × 变量”的三维数组。面板按 `(period, stock)` 排序、期在外层，所以前 $N$ 行是第 1 期，接下来的 $N$ 行是第 2 期，行优先的 `reshape` 恰好按期切开。

```{code-cell} python
pred = simulate_market(MarketParams(n_stocks=100, n_periods=5_000), seed=11)
Xp = np.column_stack([np.ones(len(pred.panel)), pred.panel[[f"z{j}" for j in range(1, 6)]].to_numpy()])  # (TN, L+1)
coef_p, se_p = ols(Xp, pred.panel["exret"].to_numpy())
truth_p = pred.gamma @ np.asarray(pred.params.factor_mean)               # Γμ：(L+1, K) @ (K,) -> (L+1,)

# 逐期横截面回归，用各期系数的时间序列标准差计算标准误（Fama–MacBeth 的做法，第 10 章详述）
n_t, n_i = pred.params.n_periods, pred.params.n_stocks
Xp3 = Xp.reshape(n_t, n_i, -1)                                           # (TN, L+1) -> (T, N, L+1)
yp3 = pred.panel["exret"].to_numpy().reshape(n_t, n_i)                   # (TN,) -> (T, N)
per_period = np.array([np.linalg.lstsq(Xp3[t], yp3[t], rcond=None)[0] for t in range(n_t)])   # (T, L+1)
se_fm = per_period.std(axis=0, ddof=1) / np.sqrt(n_t)
print(pd.DataFrame({"理论值 Γμ": truth_p, "混合回归估计": coef_p, "同方差标准误": se_p, "逐期回归标准误": se_fm},
                   index=["常数", "z1", "z2", "z3", "z4", "z5"]).round(5))
ratio = se_fm / se_p
print("逐期回归标准误 / 同方差标准误：", ratio.round(2))
assert np.all(np.abs(coef_p - truth_p) < 2 * se_fm)                    # 偏离在两倍逐期回归标准误以内
assert np.all(coef_p[1:3] / se_fm[1:3] > 3)                            # z1、z2 的系数显著为正
assert np.all(ratio[:3] > 1.3) and np.all(np.abs(ratio[3:] - 1) < 0.2)
```

系数的估计值与理论值 $\bv\Gamma\bv\mu$ 相符（偏离都在两倍逐期回归标准误以内）：$z_1$、$z_2$ 的系数显著为正（均由 `assert` 保证），而 $\bv\theta = \mathbf 0$，市场中没有任何错误定价。表中还有一个第二部分会反复出现的问题：“同方差标准误”低估了真实的不确定性。同一期的所有股票共同受到 $\mathbf f_t$ 的冲击，残差在横截面上相关，混合回归把 $NT$ 个观测当作相互独立，高估了信息量。按期做横截面回归、再看各期系数的时间序列波动，得到的标准误更接近真实的抽样误差（第 10、12 章）：常数项与 $z_1$、$z_2$ 的逐期回归标准误明显大于同方差标准误（比值见上方输出），因为它们的系数随 $\mathbf f_t$ 逐期波动；$z_3$～$z_5$ 与因子无关，两种标准误相近。估计值与理论值的偏离，应当按这种更大的标准误来判断。

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
