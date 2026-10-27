# 第 1 章《消费、欧拉方程与随机贴现因子》实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 写完第 1 章，并配套写好模拟器 `apsim.consumption` 及其测试。

**Architecture:** 新建 `apsim._rng` 统一随机子流的派生；`apsim.consumption` 实现幂效用、对数正态消费与红利的 Lucas 树，所有闭式解作为 `ConsumptionParams` 的属性；正文页 `book/part1/ch01-*.md` 调用这些属性与模拟函数验证推导；最后更新记号表、术语页、前言进度与前向引用台账。

**Tech Stack:** Python 3.13、numpy、pandas、sympy、pytest、Jupyter Book 2.1.7（mystmd），用 uv 管理。

**Spec:** `docs/specs/2026-10-26-book-design.md`（第 4、5、6、7、8 节）。用户已批准的本章大纲见下文“本章大纲”。

## Global Constraints

- 写作原则见设计文档第 7 节第 1～16 条，全部适用。重点如下：
  - 先白话、后例子、再公式（第 11 条）。
  - 正文中依赖随机种子的数字，必须由 assert 守护（第 8 条）。
  - 矩阵公式标注维度，代码注释形状（第 9 条）。
  - 估计工具第一次出现时内联实现，并与 `apsim.stats` 比对（第 15 条）。
  - 超过约 3 万字符就拆分子页面（第 14 条）。
  - 写作前后都要核对前向引用台账（第 16 条）。
- 记号：标量用斜体，向量用粗体小写，矩阵用粗体大写；向量一律为列向量，转置记作 ′；希腊字母加粗用 `\bv{}`。主观贴现因子记为 $\delta$，风险厌恶系数记为 $\gamma$，对数用小写（$c_t = \log C_t$，$\Delta c_{t+1} = c_{t+1} - c_t$），无风险利率按理论写法记为 $R^f_t$（第 $t$ 期已知，适用于 $t \to t+1$）。$\rho$ 预留给相关系数。
- 术语：$P/D$ 称“价格—红利比”，不是“市盈率”（市盈率是 $P/E$，价格与盈利之比）；第一次出现时说明二者的区别，A 股常用的股息率为 $D/P$。
- 排版：
  - 中文用全角标点，引号用“”，数值范围用～，省略号用……，破折号用——。
  - 中文与英文、数字、行内公式之间留一个半角空格。
  - 表格单元格中的短语不加句号。
- 数值例子一律使用年度参数，并在正文中注明“假设参数”。不写股权溢价之谜的实证数字（留到第 7 章），不讲 GMM 估计（留到第 11 章）。
- 本章篇幅目标为 1.2～1.5 万中文字（不含代码）。
- 环境：Git Bash on Windows，使用 `uv run ...`；Python 写文件时用 `encoding="utf-8"`、`newline="\n"`；不要删除 `book/_build/templates`。
- 验收必须用 `sh scripts/build.sh --clean`（已开启 `--strict`），构建日志中不得有本书页面的警告。

## 本章大纲（用户已批准）

| 节 | 内容 | 兑现的台账承诺 |
|---|---|---|
| 1.1 两期消费—储蓄问题 | 用白话例子引入（今天少吃一碗饭、买股票），推出一阶条件 | |
| 1.2 欧拉方程与 SDF | $P_t = \E_t[m_{t+1}x_{t+1}]$，$m_{t+1} = \delta u'(C_{t+1})/u'(C_t)$，应用于股票、债券、总收益与超额收益 | 引入 $R^f_t$ 的理论写法（`ch00-model.md:180`） |
| 1.3 风险调整 | $P = \E[x]/R^f + \Cov(m,x)$，$\E[R^e] = -R^f\Cov(m,R^e)$，说明“坏时候赚钱”的资产为什么期望收益低 | `glossary.md:209` |
| 1.4 幂效用与对数正态消费 | 无风险利率与风险溢价的闭式解；消费 CAPM 是以不可交易因子为因子的 β 模型，$\lambda \neq \E[\Delta c]$ | `ch00-concepts.md:58`、`:185` |
| 1.5 多期与 Lucas 树（展开写） | 价格等于未来支付的贴现和，横截性条件；独立同分布时价格—红利比为常数，收益与红利增长的关系 | |
| 1.6 回到第 0 章 | 第 0 章的线性 SDF 是消费 SDF 的一阶近似 $m \approx \delta(1 - \gamma\Delta c)$ | `ch00-concepts.md:24`、`:289` |
| A 股视角、模拟实验、常见误区、小结、习题（概念题、推导题、编程题各 2 道）、文献 | | |

另有一条：`notation.md:114` 中对数记号的惯例，在本章首次正式使用。

---

## 文件结构

| 文件 | 职责 |
|---|---|
| `src/apsim/_rng.py`（新建） | 把 seed（整数或 Generator）派生为 n 条独立子流；`market` 与 `consumption` 共用 |
| `src/apsim/market.py`（修改） | 改为调用 `_rng.substreams`，输出逐位不变 |
| `src/apsim/consumption.py`（新建） | 消费模型的参数、闭式解与模拟 |
| `src/apsim/__init__.py`（修改） | 导出 `ConsumptionParams`、`ConsumptionPath`、`simulate_consumption` |
| `tests/test_rng.py`、`tests/test_consumption.py`（新建） | 单元测试 |
| `book/part1/ch01-*.md`（新建） | 第 1 章正文；超过 3 万字符时按节拆页 |
| `book/myst.yml`（修改） | 目录中加入“第一部分　资产定价理论” |
| `book/notation.md`、`book/glossary.md`、`book/index.md`、`docs/promises.md`（修改） | 记号、术语、进度与台账 |

---

### Task 1: `apsim._rng` 与 `apsim.consumption`

**Files:**
- Create: `src/apsim/_rng.py`, `src/apsim/consumption.py`, `tests/test_rng.py`, `tests/test_consumption.py`
- Modify: `src/apsim/market.py`（`_substreams` 改为调用 `_rng.substreams`）, `src/apsim/__init__.py`

**Interfaces:**
- Produces:
  - `apsim._rng.substreams(seed: int | np.random.Generator, n: int) -> list[np.random.Generator]`
  - `ConsumptionParams`（frozen dataclass），字段为 `delta=0.99, gamma=2.0, mu_c=0.02, sigma_c=0.02, mu_d=0.02, sigma_d=0.12, rho_cd=0.5`；属性有 `sdf_mean`、`risk_free`、`dividend_discount`、`price_dividend_ratio`、`expected_return`、`log_equity_premium`
  - `ConsumptionPath`（frozen dataclass），字段为 `params`、`data`；`data` 的索引为 `year`（取 1…T），列为 `dc, dd, m, R, Rf`
  - `simulate_consumption(params: ConsumptionParams | None = None, *, seed: int | np.random.Generator, n_years: int = 1000) -> ConsumptionPath`
  - 以上三个名称从 `apsim` 顶层导出

**模型（写进 consumption.py 的模块 docstring，第 1 章的正文也以此为准）：**
- 对数消费增长：$\Delta c_{t+1} = \mu_c + \sigma_c\varepsilon_{t+1}$，$\varepsilon \sim N(0,1)$，独立同分布。
- 对数红利增长：$\Delta d_{t+1} = \mu_d + \sigma_d\eta_{t+1}$，$\eta = \rho\varepsilon + \sqrt{1-\rho^2}\,\nu$，$\nu \sim N(0,1)$，且与 $\varepsilon$ 独立。
- SDF：$m_{t+1} = \delta\exp(-\gamma\Delta c_{t+1})$。
- $\E[m] = \delta\exp(-\gamma\mu_c + \gamma^2\sigma_c^2/2)$，$R^f = 1/\E[m]$。
- $a = \E[m e^{\Delta d}] = \delta\exp\big(\mu_d - \gamma\mu_c + (\sigma_d^2 + \gamma^2\sigma_c^2 - 2\gamma\rho\sigma_c\sigma_d)/2\big)$，要求 $a < 1$；价格—红利比 $P/D = a/(1-a)$。
- 总收益：$R_{t+1} = e^{\Delta d_{t+1}}/a$；$\E[R] = e^{\mu_d+\sigma_d^2/2}/a$；$\log(\E[R]/R^f) = \gamma\rho\sigma_c\sigma_d$（精确成立）。

- [ ] **Step 1：写 `tests/test_rng.py` 和 `src/apsim/_rng.py`，并让 market 改用它**

先读 `src/apsim/market.py` 中现有的 `_substreams`，原样搬到 `_rng.substreams(seed, n)`：整数走 `np.random.SeedSequence(seed).spawn(n)`，Generator 走 `seed.spawn(1)[0].spawn(n)`。凡是 market 原本的逻辑，以 market 为准。再把 market 的 `_substreams(seed)` 改成 `return substreams(seed, len(_STREAMS))`。测试如下：

```python
import numpy as np

from apsim._rng import substreams


def test_int_seed_matches_seedsequence_spawn():
    gens = substreams(7, 3)
    ref = [np.random.default_rng(s) for s in np.random.SeedSequence(7).spawn(3)]
    for g, r in zip(gens, ref):
        assert g.standard_normal() == r.standard_normal()


def test_appending_stream_keeps_existing_ones():
    a = [g.standard_normal(5) for g in substreams(11, 2)]
    b = [g.standard_normal(5) for g in substreams(11, 3)]
    for x, y in zip(a, b[:2]):
        np.testing.assert_array_equal(x, y)


def test_generator_seed_consumes_one_key_per_call():
    rng = np.random.default_rng(5)
    substreams(rng, 4)
    assert rng.bit_generator.seed_seq.n_children_spawned == 1
```

运行 `uv run pytest tests/test_rng.py tests/test_market.py -v`，应全部 PASS。market 的金值测试必须通过，不得修改。

- [ ] **Step 2：写失败的测试 `tests/test_consumption.py`**

```python
import math

import numpy as np
import pandas as pd
import pytest

from apsim import ConsumptionParams, simulate_consumption

hermegauss = np.polynomial.hermite_e.hermegauss


def _nodes(n):
    x, w = hermegauss(n)
    return x, w / w.sum()


def test_sdf_mean_and_risk_free_match_quadrature():
    p = ConsumptionParams()
    x, w = _nodes(80)
    m = p.delta * np.exp(-p.gamma * (p.mu_c + p.sigma_c * x))
    assert math.isclose(p.sdf_mean, float(w @ m), rel_tol=1e-12)
    assert math.isclose(p.risk_free * p.sdf_mean, 1.0, rel_tol=1e-15)


def test_dividend_discount_matches_2d_quadrature():
    p = ConsumptionParams(rho_cd=0.3)
    x, w = _nodes(40)
    e, n = np.meshgrid(x, x, indexing="ij")
    weights = np.outer(w, w)
    eta = p.rho_cd * e + math.sqrt(1 - p.rho_cd**2) * n
    val = p.delta * np.exp(-p.gamma * (p.mu_c + p.sigma_c * e) + p.mu_d + p.sigma_d * eta)
    assert math.isclose(p.dividend_discount, float((weights * val).sum()), rel_tol=1e-10)


def test_log_equity_premium_is_exact():
    p = ConsumptionParams(gamma=5.0, rho_cd=0.4)
    assert math.isclose(math.log(p.expected_return / p.risk_free), p.log_equity_premium, rel_tol=1e-12)
    assert math.isclose(p.log_equity_premium, 5.0 * 0.4 * p.sigma_c * p.sigma_d, rel_tol=1e-15)


def test_return_identity_from_prices():
    p = ConsumptionParams()
    data = simulate_consumption(p, seed=0, n_years=50).data
    dividends = np.concatenate([[1.0], np.exp(np.cumsum(data["dd"].to_numpy()))])  # D_0 = 1
    prices = p.price_dividend_ratio * dividends
    np.testing.assert_allclose(data["R"].to_numpy(), (prices[1:] + dividends[1:]) / prices[:-1], rtol=1e-12)


def test_euler_equations_hold_in_simulation():
    data = simulate_consumption(seed=1, n_years=400_000).data
    assert abs((data["m"] * data["R"]).mean() - 1.0) < 0.003  # 约 15 倍标准误
    assert abs(data["m"].mean() * data["Rf"].iloc[0] - 1.0) < 0.001


def test_shock_correlation():
    data = simulate_consumption(ConsumptionParams(rho_cd=0.6), seed=2, n_years=200_000).data
    assert abs(np.corrcoef(data["dc"], data["dd"])[0, 1] - 0.6) < 0.01


def test_infinite_price_raises():
    p = ConsumptionParams(delta=0.999, gamma=0.5, mu_d=0.08)
    assert p.dividend_discount >= 1
    with pytest.raises(ValueError):
        p.price_dividend_ratio
    with pytest.raises(ValueError):
        simulate_consumption(p, seed=0)


def test_layout_and_reproducibility():
    a = simulate_consumption(seed=3, n_years=5).data
    assert list(a.columns) == ["dc", "dd", "m", "R", "Rf"]
    assert a.index.name == "year" and list(a.index) == [1, 2, 3, 4, 5]
    pd.testing.assert_frame_equal(a, simulate_consumption(seed=3, n_years=5).data)
    rng = np.random.default_rng(3)
    g1 = simulate_consumption(seed=rng, n_years=5).data
    g2 = simulate_consumption(seed=rng, n_years=5).data
    assert not np.allclose(g1["dc"], g2["dc"])
    with pytest.raises(TypeError):
        simulate_consumption()


@pytest.mark.parametrize(
    "kw",
    [dict(gamma=0.0), dict(delta=0.0), dict(sigma_c=-0.1), dict(rho_cd=1.5),
     dict(mu_c=float("nan")), dict(gamma=True)],
)
def test_invalid_params(kw):
    with pytest.raises(ValueError):
        ConsumptionParams(**kw)


@pytest.mark.parametrize("n", [0, 2.5, True])
def test_invalid_n_years(n):
    with pytest.raises(ValueError):
        simulate_consumption(seed=0, n_years=n)
```

运行 `uv run pytest tests/test_consumption.py -v`。预期结果：FAIL，报 `ImportError: cannot import name 'ConsumptionParams'`。

- [ ] **Step 3：写 `src/apsim/consumption.py`**

```python
"""消费型资产定价模型：幂效用、对数正态消费与红利增长、Lucas 树（第 1 章）。

模型（年度，参数均为假设值）
----
- 对数消费增长：Δc_{t+1} = μ_c + σ_c ε_{t+1}，ε ~ N(0, 1) 独立同分布。
- 对数红利增长：Δd_{t+1} = μ_d + σ_d η_{t+1}，η = ρ ε + sqrt(1 − ρ²) ν，ν ~ N(0, 1) 与 ε 独立。
- 幂效用的 SDF：m_{t+1} = δ exp(−γ Δc_{t+1})。
- E[m] = δ exp(−γ μ_c + γ² σ_c² / 2)，R^f = 1 / E[m]（常数）。
- 红利索取权：增长独立同分布时价格—红利比为常数 P/D = a / (1 − a)，
  a = E[m e^{Δd}] = δ exp(μ_d − γ μ_c + (σ_d² + γ² σ_c² − 2γρσ_cσ_d) / 2)，须 a < 1。
- 总收益 R_{t+1} = e^{Δd_{t+1}} / a；E[R] = e^{μ_d + σ_d²/2} / a；
  对数股权溢价 log(E[R] / R^f) = γ ρ σ_c σ_d（精确成立）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from apsim._rng import substreams

# 随机子流按名称固定顺序；以后新增随机部分只在末尾追加，不扰动已有抽样
_STREAMS = ("consumption_shock", "dividend_shock")
_FIELDS = ("delta", "gamma", "mu_c", "sigma_c", "mu_d", "sigma_d", "rho_cd")


@dataclass(frozen=True)
class ConsumptionParams:
    """消费模型的参数（年度）。默认值是教学用的假设参数，不是对任何市场的校准。"""

    delta: float = 0.99
    gamma: float = 2.0
    mu_c: float = 0.02
    sigma_c: float = 0.02
    mu_d: float = 0.02
    sigma_d: float = 0.12
    rho_cd: float = 0.5

    def __post_init__(self) -> None:
        for name in _FIELDS:
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)):
                raise ValueError(f"{name} 须为实数")
            if not math.isfinite(v):
                raise ValueError(f"{name} 须为有限值")
            object.__setattr__(self, name, float(v))
        if self.delta <= 0:
            raise ValueError("delta 须为正")
        if self.gamma <= 0:
            raise ValueError("gamma 须为正")
        if self.sigma_c < 0 or self.sigma_d < 0:
            raise ValueError("波动率须非负")
        if not -1.0 <= self.rho_cd <= 1.0:
            raise ValueError("rho_cd 须在 [-1, 1] 内")

    @property
    def sdf_mean(self) -> float:
        """E[m] = δ exp(−γ μ_c + γ² σ_c² / 2)。"""
        return self.delta * math.exp(-self.gamma * self.mu_c + 0.5 * self.gamma**2 * self.sigma_c**2)

    @property
    def risk_free(self) -> float:
        """总无风险利率 R^f = 1 / E[m]。"""
        return 1.0 / self.sdf_mean

    @property
    def dividend_discount(self) -> float:
        """a = E[m e^{Δd}]：明年一元红利增长因子在今天的价格。"""
        g, sc, sd, r = self.gamma, self.sigma_c, self.sigma_d, self.rho_cd
        return self.delta * math.exp(self.mu_d - g * self.mu_c + 0.5 * (sd**2 + g**2 * sc**2 - 2 * g * r * sc * sd))

    @property
    def price_dividend_ratio(self) -> float:
        """价格—红利比 P/D = a / (1 − a)；a ≥ 1 时价格无穷大，抛出 ValueError。"""
        a = self.dividend_discount
        if a >= 1.0:
            raise ValueError(f"a = E[m e^Δd] = {a:.4f} ≥ 1，红利索取权的价格无穷大（横截性条件不成立）")
        return a / (1.0 - a)

    @property
    def expected_return(self) -> float:
        """红利索取权的期望总收益 E[R] = e^{μ_d + σ_d²/2} / a。"""
        self.price_dividend_ratio  # 价格有限时收益才有定义
        return math.exp(self.mu_d + 0.5 * self.sigma_d**2) / self.dividend_discount

    @property
    def log_equity_premium(self) -> float:
        """log(E[R] / R^f) = γ ρ σ_c σ_d。"""
        return self.gamma * self.rho_cd * self.sigma_c * self.sigma_d


@dataclass(frozen=True)
class ConsumptionPath:
    """一次模拟：data 的索引 year 取 1…T，列为 dc、dd、m、R、Rf。"""

    params: ConsumptionParams
    data: pd.DataFrame


def simulate_consumption(
    params: ConsumptionParams | None = None,
    *,
    seed: int | np.random.Generator,
    n_years: int = 1000,
) -> ConsumptionPath:
    """模拟 n_years 年的消费增长、红利增长、SDF 与红利索取权的收益。

    seed 必须以关键字传入，可以是整数或 np.random.Generator（规则同 simulate_market）。
    """
    p = params if params is not None else ConsumptionParams()
    if isinstance(n_years, bool) or not isinstance(n_years, (int, np.integer)) or n_years < 1:
        raise ValueError("n_years 须为正整数")
    a = p.dividend_discount
    p.price_dividend_ratio  # a ≥ 1 时抛出 ValueError
    rng_c, rng_d = substreams(seed, len(_STREAMS))
    eps = rng_c.standard_normal(n_years)
    nu = rng_d.standard_normal(n_years)
    eta = p.rho_cd * eps + math.sqrt(1.0 - p.rho_cd**2) * nu
    dc = p.mu_c + p.sigma_c * eps
    dd = p.mu_d + p.sigma_d * eta
    data = pd.DataFrame(
        {
            "dc": dc,
            "dd": dd,
            "m": p.delta * np.exp(-p.gamma * dc),
            "R": np.exp(dd) / a,
            "Rf": np.full(n_years, p.risk_free),
        },
        index=pd.RangeIndex(1, n_years + 1, name="year"),
    )
    return ConsumptionPath(params=p, data=data)
```

然后在 `src/apsim/__init__.py` 中导出 `ConsumptionParams`、`ConsumptionPath`、`simulate_consumption`，并把它们加入 `__all__`。

- [ ] **Step 4：运行测试**

运行 `uv run pytest -v`，预期全部 PASS。如果某个统计容差测试失败，先对照模型逐行检查实现，不要放宽容差。

- [ ] **Step 5：加金值测试**

在 `tests/test_consumption.py` 末尾加 `test_golden_values`：固定 `simulate_consumption(seed=0, n_years=3).data` 中 `dc`、`dd` 两列的数值，用 `np.testing.assert_allclose(..., rtol=0, atol=1e-12)` 比较。数值在 Step 4 通过后生成，直接写进测试。

- [ ] **Step 6：提交**

```bash
git add src/apsim tests
git commit -m "feat(apsim): 新增 consumption 模块（幂效用、对数正态消费、Lucas 树），随机子流统一由 _rng 派生"
```

---

### Task 2: 第 1 章正文

**Files:**
- Create: `book/part1/ch01-*.md`。先写成一个文件 `book/part1/ch01-euler-sdf.md`；超过约 3 万字符时，按节拆成首页加若干子页面，结构参照 `book/part0/` 的拆法，首页同样附阅读路径表。
- Modify: `book/myst.yml`，在目录中新增“第一部分　资产定价理论”，第 1 章放在它下面；`book/references.bib` 如需补充文献也在此修改。

**Interfaces:**
- Consumes：Task 1 的 `ConsumptionParams` 及其属性、`simulate_consumption`；已有的 `apsim.plotting.setup()`；若用到回归，调用 `apsim.stats.ols`。
- Produces：页面标签 `sec-ch1-*`，公式标签 `eq-ch1-*`，习题标签 `ex-ch1-*`。页首设置 `numbering: enumerator: '1.%s'`，习题模板为 `习题 %s`，写法与第 0 章各页相同。

**内容要求：**按上文“本章大纲”，每节都按设计文档第 6 节的模板写。

- [ ] **Step 1：写作前核对台账**

运行 `uv run python scripts/promises.py`，读 `docs/promises.md` 中“第 1 章”一组。上表“兑现的台账承诺”一列列出了每条承诺由哪一节兑现，逐条落实。如果台账里还有表中没列的条目，也要处理：在正文中兑现，或者回到原处改写措辞。

- [ ] **Step 2：写 1.1～1.3**

写作要求：
- **1.1**：先讲白话例子。今天少消费 1 元买入资产，明年得到支付 $x$；什么时候值得买？然后写出两期问题和一阶条件，推导要完整。
- **1.2**：给出 $P_t = \E_t[m_{t+1}x_{t+1}]$ 和 $m_{t+1}$ 的定义，并逐一写出四种特例：股票（$x = P_{t+1}+D_{t+1}$）、无风险债券（$R^f_t = 1/\E_t[m_{t+1}]$）、总收益（$1 = \E[mR]$）、超额收益（$0 = \E[mR^e]$）。与第 0 章的 $0 = \E[m r]$ 对应起来，并说明面板写法 $R^f_{t-1}$ 与理论写法 $R^f_t$ 只是时点标号不同，引到记号表。
- **1.3**：推导协方差分解 $P = \E[x]/R^f + \Cov(m,x)$ 和 $\E[R^e] = -R^f\Cov(m, R^e)$。白话解释：坏时候 $m$ 高，在坏时候赚钱的资产像保险，所以期望收益低。举一个两状态的数值例子，配代码验证。

每个推导配 sympy 或数值验证。

- [ ] **Step 3：写 1.4～1.6**

- **1.4**：
  - 用 sympy 推出 $\log R^f = -\log\delta + \gamma\mu_c - \gamma^2\sigma_c^2/2$，逐项解释：耐心、跨期替代、预防性储蓄。
  - 推出 $\log(\E[R]/R^f) = \gamma\Cov(\Delta c, \Delta d)$，用 `ConsumptionParams` 的属性核对。
  - 消费 CAPM：把 $\Delta c$ 当作因子，写出 $\E[R^e] \approx \beta_{\Delta c}\lambda_c$，其中 $\lambda_c = \gamma\Var(\Delta c)$。说明 $\lambda_c$ 不等于 $\E[\Delta c]$，因为消费增长不是资产收益。这一点兑现第 0 章的承诺，正文要写明“第 0 章 0.2 节说过……”。
  - 说明哪些关系是近似：对数正态的二阶项和线性化各在哪一步引入。用模拟或数值积分量化近似误差，并用 assert 守护。
- **1.5**（展开写）：
  - 多期迭代：$P_t = \E_t\sum_{j\ge1} M_{t,t+j}D_{t+j}$，并写出横截性条件。
  - Lucas 树：红利等于消费（$\rho = 1$、$\sigma_d = \sigma_c$、$\mu_d = \mu_c$）是特例。一般情形给出 $P/D = a/(1-a)$，推导要完整，并解释 $a < 1$ 的含义：横截性条件、价格有限。
  - 用 `simulate_consumption` 画一条模拟的红利、价格和收益路径。可以画图，作图前调用 `apsim.plotting.setup()`，线条较少时直接标注。
  - 说明独立同分布模型的局限：价格—红利比不变，所以收益不可预测；第 6 章再引入时变的贴现率。
- **1.6**：把 $m = \delta e^{-\gamma\Delta c}$ 在 $\mu_c$ 处线性化，得到 $m \approx \E[m](1 - \gamma(\Delta c - \mu_c))$，这就是第 0 章 $m = 1 - \mathbf b'(\mathbf f - \bv\mu)$ 的一维形式（按 $\E[m]$ 归一化）。逐项对照第 0 章的记号，并说明第 0 章“因子是超额收益”与这里“因子是消费增长”的区别，引出第 3、4 章。

- [ ] **Step 4：写 A 股视角、模拟实验、常见误区、小结、习题、文献**

- **A 股视角**：只写定性内容：中国的消费数据是季度或年度的、样本短、有修订；用消费数据检验定价模型的困难。没有把握的经验结论不写，必须写时标“（待核实）”。
- **模拟实验**：用 `simulate_consumption` 验证三点：
  - 欧拉方程 $\E[mR] = 1$ 与 $\E[m]R^f = 1$，样本值随年数 $T$ 收敛；
  - 风险溢价闭式解与模拟样本均值一致，用 assert 守护，阈值要写明理由；
  - 用 $\Delta c$ 做时间序列回归求 β 再乘 $\lambda_c$ 的消费 CAPM 近似，与精确值比较。回归用 `apsim.stats.ols`。
- **常见误区**：至少 3 条，例如：把 $\gamma$ 同时当作风险厌恶与跨期替代弹性的倒数而不加说明；把 $\lambda_c$ 当作 $\E[\Delta c]$；忽视 $a < 1$ 的条件。
- **习题**：概念题 2 道、推导题 2 道（其中 1 道推导 $\log R^f$ 的闭式解，或推导 Lucas 树的价格—红利比）、编程题 2 道。全部给出折叠的解答，数值答案用 assert 核对。
- **文献**：只引用 `references.bib` 中有的条目，以及确定无误的经典文献（例如 Lucas 1978 Econometrica 46(6):1429–1445、Hansen–Singleton 1983 JPE 91(2):249–265、Breeden 1979 JFE 7(3):265–296、Rubinstein 1976 Bell J. Econ. 7(2):407–425），卷、期、页码必须核对；不确定的不加。

- [ ] **Step 5：构建与自查**

依次运行：
- `sh scripts/build.sh --clean`：退出码 0，构建日志中本书页面没有警告；
- `uv run pytest`；
- 排版正则：`grep -nP '[，。；：？！] |[\x{4e00}-\x{9fff}][,;:?!()]|\.\.\.|(?<!…)…(?!…)' book/part1/*.md`，无输出；
- 统计中文字数，在 1.2～1.5 万之间；超过 3 万字符就拆页。

- [ ] **Step 6：提交**

提交信息为 `docs(book): 第 1 章《消费、欧拉方程与随机贴现因子》`。

---

### Task 3: 记号、术语、进度与台账

**Files:** Modify `book/notation.md`, `book/glossary.md`, `book/index.md`, `docs/promises.md`, `README.md`（如需要）

- [ ] **Step 1：更新记号表**

新增以下记号：$C_t$、$c_t$、$\Delta c_{t+1}$、$D_t$、$d_t$、$\delta$、$\gamma$、$u(\cdot)$、$M_{t,t+j}$、$a$、$\mu_c$、$\sigma_c$、$\mu_d$、$\sigma_d$、$\rho$（本章指相关系数）、$\lambda_c$，以及它们与 `apsim.consumption` 字段的对应关系。在“已预留与已知冲突”表中补两条：$\gamma$（小写）是风险厌恶系数，$\bv\Gamma$（大写粗体）是载荷矩阵；$\delta$ 已正式使用。“符号读法”表中补上 $\gamma$、$\delta$（读法改为“已用”）、$\rho$、$\nabla$ 或 $\partial$（如用到）。

- [ ] **Step 2：更新术语页**

把本章新出现、学术与业界含义不同的术语补入 `book/glossary.md`（设计文档第 7 节第 12 条），至少包括“贴现因子与折现率”的补充说明，以及“风险厌恶”或“消费 CAPM”。条目放在合适的分组中，总表同步加行。

- [ ] **Step 3：更新前言进度**

在 `book/index.md` 的写作进度表中，把第 1 章标为“已完成”。

- [ ] **Step 4：重新生成台账并核对**

运行 `uv run python scripts/promises.py`，确认“第 1 章”一组已经消失（`written()` 能识别 `book/part1/ch01-*.md`）。如果仍有残留，逐条处理。

- [ ] **Step 5：构建、测试、提交**

运行 `sh scripts/build.sh --clean` 和 `uv run pytest`，两者都通过后提交：`docs(book): 第 1 章配套更新记号表、术语页、进度与台账`。
