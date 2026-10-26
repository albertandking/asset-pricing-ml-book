# 电子书地基（项目骨架、apsim 最小版、第 0 章）实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 搭好可构建、可本地阅读的 Jupyter Book 2 电子书骨架，实现模拟市场生成器 `apsim` 的最小版本，并写完前言、记号表和第 0 章。

**Architecture:** uv 管理的 Python 项目，`src/apsim` 为可测试的模拟与工具包；`book/` 为 MyST Markdown 源文件，构建时执行代码并调用 `apsim`；`scripts/` 提供构建与阅读脚本。

**Tech Stack:** Python 3.13、uv、numpy、pandas、scipy、matplotlib、sympy、statsmodels、pytest、Jupyter Book 2.1.7（mystmd）、ipykernel、Node.js v24。

**Spec:** `docs/specs/2026-10-26-book-design.md`

## Global Constraints

- 书籍框架：Jupyter Book 2.1.7，章节格式为 MyST Markdown（`.md` + `code-cell`），构建命令 `jupyter book build --html --execute`。
- 文件名用英文，章节标题用中文。
- 正文中文全角标点（GB/T 15834-2011）；数值范围用“～”；引号用“”。代码中保留半角符号。
- 网页字体：中文微软雅黑，英文和数字 Arial，代码 Consolas；需提供 macOS 回退字体。
- 所有随机过程通过显式 `seed`／`numpy.random.Generator` 控制。
- 没有把握的出处、年份、数字在正文中标注“待核实”。
- 不转写教材原文；不使用真实市场数据；不部署到互联网。
- 本地 git 提交，不推送远程。
- 本机 Python 为 Windows 版 CPython（uv 管理），shell 为 Git Bash：命令用 `uv run …`，不要假设 `.venv/bin` 存在（实际为 `.venv/Scripts`）。

---

## 文件结构

| 文件 | 职责 |
|---|---|
| `pyproject.toml` | 项目元数据、依赖、pytest 配置 |
| `.python-version`、`.gitignore`、`.gitattributes` | 环境与仓库规范 |
| `README.md` | 如何安装、构建、阅读 |
| `参考资料/参考.md` | 原根目录文件移入 |
| `src/apsim/__init__.py` | 包入口，导出公开接口 |
| `src/apsim/market.py` | 横截面面板模拟：因子、特征、载荷、定价误差、SDF |
| `src/apsim/plotting.py` | matplotlib 统一样式与中文字体 |
| `tests/test_market.py`、`tests/test_plotting.py` | 单元测试 |
| `book/myst.yml` | 书籍配置与目录 |
| `book/_static/custom.css` | 中文排版样式 |
| `book/references.bib` | 参考文献 |
| `book/index.md` | 前言 |
| `book/notation.md` | 全书记号表 |
| `book/part0/ch00-guide.md` | 第 0 章 |
| `scripts/build.sh`、`scripts/serve.sh` | 构建与阅读脚本 |

---

### Task 1: 整理目录与项目骨架

**Files:**
- Move: `参考.md` → `参考资料/参考.md`
- Create: `pyproject.toml`、`.python-version`、`.gitignore`、`.gitattributes`、`README.md`、`src/apsim/__init__.py`、`tests/test_package.py`

**Interfaces:**
- Produces: 可导入的包 `apsim`，`apsim.__version__ == "0.1.0"`；`uv run pytest` 可运行。

- [ ] **Step 1: 移动参考文件**

```bash
cd <项目根目录>
mkdir -p 参考资料 && git mv -f 参考.md 参考资料/参考.md 2>/dev/null || mv 参考.md 参考资料/参考.md
```

（`参考.md` 尚未被 git 跟踪，`git mv` 会失败并回退到 `mv`。）

- [ ] **Step 2: 写 `.python-version`、`.gitignore`、`.gitattributes`**

`.python-version`：
```
3.13
```

`.gitignore`：
```
.venv/
__pycache__/
*.pyc
.pytest_cache/
book/_build/
.ipynb_checkpoints/
```

`.gitattributes`：
```
* text=auto eol=lf
```

- [ ] **Step 3: 写 `pyproject.toml`**

```toml
[project]
name = "apsim"
version = "0.1.0"
description = "《资产定价与机器学习：从理论到 A 股实证》配套模拟与工具包"
requires-python = ">=3.13"
dependencies = [
    "numpy>=2.1",
    "pandas>=2.2",
    "scipy>=1.14",
    "matplotlib>=3.9",
    "sympy>=1.13",
    "statsmodels>=0.14",
]

[dependency-groups]
dev = [
    "pytest>=8.3",
    "ipykernel>=6.29",
    "jupyter-book==2.1.7",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/apsim"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 4: 写失败的测试 `tests/test_package.py`**

```python
import apsim


def test_version():
    assert apsim.__version__ == "0.1.0"
```

- [ ] **Step 5: 同步环境并确认测试失败**

Run: `uv sync && uv run pytest tests/test_package.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'apsim'` 或包内无 `__version__`）

- [ ] **Step 6: 写 `src/apsim/__init__.py`**

```python
"""《资产定价与机器学习：从理论到 A 股实证》配套模拟与工具包。"""

__version__ = "0.1.0"
```

- [ ] **Step 7: 确认测试通过**

Run: `uv sync && uv run pytest tests/test_package.py -v`
Expected: PASS

- [ ] **Step 8: 写 `README.md`**

````markdown
# 资产定价与机器学习：从理论到 A 股实证

本地阅读的自包含教材，配套模拟市场生成器 `apsim`。

## 安装

```bash
uv sync
```

## 构建与阅读

```bash
sh scripts/build.sh   # 构建全书（执行全部代码）
sh scripts/serve.sh   # 启动本地服务器，浏览器打开 http://localhost:8000
```

写作时可用实时预览：`cd book && uv run jupyter book start --execute`。

## 测试

```bash
uv run pytest
```

## 目录

- `book/`：书的源文件（MyST Markdown）
- `src/apsim/`：模拟市场生成器与工具函数
- `tests/`：单元测试
- `docs/specs/`：设计文档
- `参考资料/`：参考材料
````

- [ ] **Step 9: 提交**

```bash
git add -A
git commit -m "chore: 整理目录并搭建 uv 项目骨架"
```

---

### Task 2: 模拟市场生成器 `apsim.market`

**Files:**
- Create: `src/apsim/market.py`
- Modify: `src/apsim/__init__.py`
- Test: `tests/test_market.py`

**模型（写进 docstring，第 0 章也要讲）：**

- 因子超额收益：$f_t = \mu + \operatorname{diag}(\sigma)\,u_t$，$u_t \sim N(0, I_K)$，各因子独立。
- 特征：$z_{i,t} = \rho z_{i,t-1} + \sqrt{1-\rho^2}\,e_{i,t}$，平稳分布为 $N(0, I_L)$。
- 时点约定：面板中第 $t$ 期的行，特征列为 $z_{i,t-1}$（期初已知），收益列为第 $t$ 期实现的超额收益。
- 因子载荷：$\beta_{i,t} = \Gamma' x_{i,t-1}$，$x_{i,t-1} = (1, z_{i,t-1}')'$，$\Gamma$ 为 $(L+1)\times K$。
- 定价误差：$\alpha_{i,t} = \delta' z_{i,t-1}$（默认 $\delta = 0$）。
- 超额收益：$r_{i,t} = \alpha_{i,t} + \beta_{i,t}' f_t + \varepsilon_{i,t}$，$\varepsilon_{i,t} \sim N(0, s^2)$。
- 真实 SDF：$m_t = 1 - b'(f_t - \mu)$，$b = \Sigma_f^{-1}\mu$。于是 $\mathbb{E}[m_t] = 1$，$\mathbb{E}[m_t f_t] = 0$，且 $\mathbb{E}[m_t r_{i,t} \mid z_{i,t-1}] = \alpha_{i,t}$。

**Interfaces:**
- Produces:
  - `default_gamma(n_chars: int, n_factors: int) -> np.ndarray`
  - `MarketParams`（冻结数据类）：`n_stocks: int = 500`、`n_periods: int = 240`、`factor_mean: tuple[float, ...] = (0.006, 0.003, 0.004)`、`factor_vol: tuple[float, ...] = (0.05, 0.03, 0.03)`、`n_chars: int = 5`、`char_persistence: float = 0.9`、`idio_vol: float = 0.10`、`gamma: np.ndarray | None = None`、`alpha_delta: tuple[float, ...] | None = None`；属性 `n_factors`；方法 `resolved_gamma() -> np.ndarray`、`resolved_delta() -> np.ndarray`
  - `SimulatedMarket`（冻结数据类）：`params`、`factors: pd.DataFrame`（索引 `period` 为 1～T，列 `f1`…`fK`）、`panel: pd.DataFrame`（MultiIndex `(period, stock)`，列 `z1`…`zL`、`beta1`…`betaK`、`alpha`、`exret`）、`gamma: np.ndarray`、`delta: np.ndarray`；属性 `sdf_weights -> np.ndarray`；方法 `sdf() -> pd.Series`（索引 `period`，名称 `m`）
  - `simulate_market(params: MarketParams | None = None, seed: int | None = 0) -> SimulatedMarket`
  - 以上从 `apsim` 顶层导出。

- [ ] **Step 1: 写失败的测试 `tests/test_market.py`**

```python
import numpy as np
import pandas as pd
import pytest

from apsim import MarketParams, SimulatedMarket, default_gamma, simulate_market


def test_default_gamma_layout():
    g = default_gamma(n_chars=5, n_factors=3)
    assert g.shape == (6, 3)
    assert g[0, 0] == 1.0
    assert g[1, 1] == 0.5 and g[2, 2] == 0.5
    assert np.count_nonzero(g) == 3


def test_shapes_and_columns():
    p = MarketParams(n_stocks=50, n_periods=24)
    mkt = simulate_market(p, seed=1)
    assert isinstance(mkt, SimulatedMarket)
    assert mkt.factors.shape == (24, 3)
    assert list(mkt.factors.columns) == ["f1", "f2", "f3"]
    assert mkt.factors.index.name == "period"
    assert len(mkt.panel) == 24 * 50
    assert mkt.panel.index.names == ["period", "stock"]
    expected = ["z1", "z2", "z3", "z4", "z5", "beta1", "beta2", "beta3", "alpha", "exret"]
    assert list(mkt.panel.columns) == expected
    assert mkt.sdf().shape == (24,)
    assert mkt.sdf().name == "m"


def test_reproducible_with_seed():
    p = MarketParams(n_stocks=20, n_periods=12)
    a = simulate_market(p, seed=7)
    b = simulate_market(p, seed=7)
    c = simulate_market(p, seed=8)
    pd.testing.assert_frame_equal(a.panel, b.panel)
    assert not np.allclose(a.panel["exret"], c.panel["exret"])


def test_sdf_has_mean_one_and_prices_factors():
    mkt = simulate_market(MarketParams(n_stocks=5, n_periods=20000), seed=0)
    m = mkt.sdf()
    assert abs(m.mean() - 1.0) < 0.01
    pricing = mkt.factors.mul(m, axis=0).mean()
    assert np.all(np.abs(pricing) < 0.0015)


def test_stocks_priced_when_no_alpha():
    mkt = simulate_market(MarketParams(n_stocks=20, n_periods=20000), seed=0)
    m = mkt.sdf()
    mr = mkt.panel["exret"] * m.reindex(mkt.panel.index.get_level_values("period")).to_numpy()
    assert abs(mr.mean()) < 0.0015


def test_alpha_is_pricing_error():
    p = MarketParams(n_stocks=200, n_periods=5000, alpha_delta=(0.01, 0.0, 0.0, 0.0, 0.0))
    mkt = simulate_market(p, seed=0)
    m = mkt.sdf().reindex(mkt.panel.index.get_level_values("period")).to_numpy()
    y = mkt.panel["exret"].to_numpy() * m
    z1 = mkt.panel["z1"].to_numpy()
    slope = np.cov(y, z1)[0, 1] / np.var(z1, ddof=1)
    assert abs(slope - 0.01) < 0.002


def test_characteristic_persistence():
    p = MarketParams(n_stocks=200, n_periods=100, char_persistence=0.9)
    z1 = simulate_market(p, seed=0).panel["z1"].unstack("stock")
    rho_hat = np.corrcoef(z1.iloc[1:].to_numpy().ravel(), z1.iloc[:-1].to_numpy().ravel())[0, 1]
    assert abs(rho_hat - 0.9) < 0.02


def test_gamma_recovered_by_interaction_regression():
    p = MarketParams(n_stocks=200, n_periods=600)
    mkt = simulate_market(p, seed=0)
    panel = mkt.panel
    L, K = p.n_chars, p.n_factors
    x = np.column_stack([np.ones(len(panel)), panel[[f"z{j}" for j in range(1, L + 1)]].to_numpy()])
    f = mkt.factors.reindex(panel.index.get_level_values("period")).to_numpy()
    design = (x[:, :, None] * f[:, None, :]).reshape(len(panel), (L + 1) * K)
    coef, *_ = np.linalg.lstsq(design, panel["exret"].to_numpy(), rcond=None)
    np.testing.assert_allclose(coef.reshape(L + 1, K), mkt.gamma, atol=0.05)


def test_invalid_params_raise():
    with pytest.raises(ValueError):
        MarketParams(factor_mean=(0.01, 0.02), factor_vol=(0.05,))
    with pytest.raises(ValueError):
        MarketParams(char_persistence=1.0)
    with pytest.raises(ValueError):
        MarketParams(alpha_delta=(0.01,))
    with pytest.raises(ValueError):
        MarketParams(gamma=np.zeros((2, 2)))
```

- [ ] **Step 2: 确认测试失败**

Run: `uv run pytest tests/test_market.py -v`
Expected: FAIL（`ImportError: cannot import name 'MarketParams'`）

- [ ] **Step 3: 写 `src/apsim/market.py`**

```python
"""模拟市场生成器：已知真实 SDF、因子与特征的横截面面板。

模型
----
- 因子超额收益：f_t = mu + diag(sigma) u_t，u_t ~ N(0, I_K)。
- 特征：z_{i,t} = rho z_{i,t-1} + sqrt(1 - rho^2) e_{i,t}，平稳分布为 N(0, I_L)。
- 因子载荷：beta_{i,t} = Gamma' x_{i,t-1}，x_{i,t-1} = (1, z_{i,t-1}')'。
- 定价误差：alpha_{i,t} = delta' z_{i,t-1}。
- 超额收益：r_{i,t} = alpha_{i,t} + beta_{i,t}' f_t + eps_{i,t}，eps_{i,t} ~ N(0, s^2)。
- 真实 SDF：m_t = 1 - b'(f_t - mu)，b = Sigma_f^{-1} mu。

时点约定：面板中第 t 期的行，特征列为 z_{i,t-1}（期初已知），exret 为第 t 期实现的超额收益。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def default_gamma(n_chars: int, n_factors: int) -> np.ndarray:
    """默认载荷矩阵：截距只载荷第一个因子，前 K-1 个特征各载荷一个其余因子，其余特征无关。"""
    gamma = np.zeros((n_chars + 1, n_factors))
    gamma[0, 0] = 1.0
    for j in range(1, min(n_chars, n_factors - 1) + 1):
        gamma[j, j] = 0.5
    return gamma


@dataclass(frozen=True)
class MarketParams:
    """模拟市场的参数。收益均为月度超额收益。"""

    n_stocks: int = 500
    n_periods: int = 240
    factor_mean: tuple[float, ...] = (0.006, 0.003, 0.004)
    factor_vol: tuple[float, ...] = (0.05, 0.03, 0.03)
    n_chars: int = 5
    char_persistence: float = 0.9
    idio_vol: float = 0.10
    gamma: np.ndarray | None = field(default=None, compare=False)
    alpha_delta: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.n_stocks < 1 or self.n_periods < 1 or self.n_chars < 0:
            raise ValueError("n_stocks、n_periods 须为正，n_chars 须非负")
        if len(self.factor_mean) != len(self.factor_vol) or len(self.factor_mean) == 0:
            raise ValueError("factor_mean 与 factor_vol 长度须相同且非空")
        if not 0.0 <= self.char_persistence < 1.0:
            raise ValueError("char_persistence 须在 [0, 1) 内")
        if self.idio_vol < 0 or min(self.factor_vol) <= 0:
            raise ValueError("波动率须为正")
        if self.gamma is not None and np.shape(self.gamma) != (self.n_chars + 1, self.n_factors):
            raise ValueError(f"gamma 形状须为 {(self.n_chars + 1, self.n_factors)}")
        if self.alpha_delta is not None and len(self.alpha_delta) != self.n_chars:
            raise ValueError(f"alpha_delta 长度须为 n_chars={self.n_chars}")

    @property
    def n_factors(self) -> int:
        return len(self.factor_mean)

    def resolved_gamma(self) -> np.ndarray:
        if self.gamma is None:
            return default_gamma(self.n_chars, self.n_factors)
        return np.asarray(self.gamma, dtype=float)

    def resolved_delta(self) -> np.ndarray:
        if self.alpha_delta is None:
            return np.zeros(self.n_chars)
        return np.asarray(self.alpha_delta, dtype=float)


@dataclass(frozen=True)
class SimulatedMarket:
    """一次模拟的结果，附带全部真实参数。"""

    params: MarketParams
    factors: pd.DataFrame
    panel: pd.DataFrame
    gamma: np.ndarray
    delta: np.ndarray

    @property
    def sdf_weights(self) -> np.ndarray:
        """总体 SDF 系数 b = Sigma_f^{-1} mu。"""
        mu = np.asarray(self.params.factor_mean)
        var = np.asarray(self.params.factor_vol) ** 2
        return mu / var

    def sdf(self) -> pd.Series:
        """真实 SDF 的实现值 m_t = 1 - b'(f_t - mu)。"""
        mu = np.asarray(self.params.factor_mean)
        m = 1.0 - (self.factors.to_numpy() - mu) @ self.sdf_weights
        return pd.Series(m, index=self.factors.index, name="m")


def simulate_market(params: MarketParams | None = None, seed: int | None = 0) -> SimulatedMarket:
    """按 `MarketParams` 生成一次模拟市场。相同的 seed 给出相同的结果。"""
    p = params if params is not None else MarketParams()
    rng = np.random.default_rng(seed)
    T, N, L, K = p.n_periods, p.n_stocks, p.n_chars, p.n_factors
    gamma, delta = p.resolved_gamma(), p.resolved_delta()

    f = np.asarray(p.factor_mean) + np.asarray(p.factor_vol) * rng.standard_normal((T, K))

    z = np.empty((T + 1, N, L))
    z[0] = rng.standard_normal((N, L))
    shock_scale = np.sqrt(1.0 - p.char_persistence**2)
    for t in range(1, T + 1):
        z[t] = p.char_persistence * z[t - 1] + shock_scale * rng.standard_normal((N, L))
    z_lag = z[:-1]

    x = np.concatenate([np.ones((T, N, 1)), z_lag], axis=2)
    beta = x @ gamma
    alpha = z_lag @ delta
    eps = p.idio_vol * rng.standard_normal((T, N))
    exret = alpha + np.einsum("tnk,tk->tn", beta, f) + eps

    periods = pd.RangeIndex(1, T + 1, name="period")
    factors = pd.DataFrame(f, index=periods, columns=[f"f{k}" for k in range(1, K + 1)])
    index = pd.MultiIndex.from_product([periods, range(N)], names=["period", "stock"])
    columns = {f"z{j}": z_lag[:, :, j - 1].ravel() for j in range(1, L + 1)}
    columns |= {f"beta{k}": beta[:, :, k - 1].ravel() for k in range(1, K + 1)}
    columns |= {"alpha": alpha.ravel(), "exret": exret.ravel()}
    panel = pd.DataFrame(columns, index=index)

    return SimulatedMarket(params=p, factors=factors, panel=panel, gamma=gamma, delta=delta)
```

- [ ] **Step 4: 更新 `src/apsim/__init__.py`**

```python
"""《资产定价与机器学习：从理论到 A 股实证》配套模拟与工具包。"""

from apsim.market import MarketParams, SimulatedMarket, default_gamma, simulate_market

__version__ = "0.1.0"

__all__ = ["MarketParams", "SimulatedMarket", "default_gamma", "simulate_market"]
```

- [ ] **Step 5: 确认全部测试通过**

Run: `uv run pytest -v`
Expected: 全部 PASS。若某个统计容差测试失败，先检查实现是否与上面的模型一致，不要放宽容差。

- [ ] **Step 6: 提交**

```bash
git add src/apsim tests/test_market.py
git commit -m "feat(apsim): 模拟市场生成器最小版本"
```

---

### Task 3: 作图样式 `apsim.plotting`

执行前先加载 `dataviz` 技能，按其配色与图形规范确定默认颜色循环；下面代码中的颜色循环如与该技能冲突，以技能为准。

**Files:**
- Create: `src/apsim/plotting.py`
- Modify: `src/apsim/__init__.py`
- Test: `tests/test_plotting.py`

**Interfaces:**
- Produces: `apsim.plotting.setup() -> str | None`（设置 matplotlib 全局样式，返回所选中文字体名；找不到中文字体时返回 `None` 并发出 `UserWarning`）；常量 `CJK_FONT_CANDIDATES: tuple[str, ...]`。

- [ ] **Step 1: 写失败的测试 `tests/test_plotting.py`**

```python
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from apsim import plotting


def test_setup_configures_fonts():
    chosen = plotting.setup()
    assert plt.rcParams["axes.unicode_minus"] is False
    sans = plt.rcParams["font.sans-serif"]
    assert sans[0] == "Arial"
    if chosen is not None:
        assert chosen in plotting.CJK_FONT_CANDIDATES
        assert chosen in sans


def test_setup_is_idempotent():
    first = plotting.setup()
    second = plotting.setup()
    assert first == second
    assert plt.rcParams["font.sans-serif"].count("Arial") == 1
```

- [ ] **Step 2: 确认测试失败**

Run: `uv run pytest tests/test_plotting.py -v`
Expected: FAIL（`ImportError: cannot import name 'plotting'`）

- [ ] **Step 3: 写 `src/apsim/plotting.py`**

```python
"""全书统一的 matplotlib 样式：英文和数字用 Arial，中文用系统可用的黑体类字体。"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
from matplotlib import font_manager

CJK_FONT_CANDIDATES: tuple[str, ...] = (
    "Microsoft YaHei",
    "PingFang SC",
    "Hiragino Sans GB",
    "Heiti SC",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "SimHei",
    "Arial Unicode MS",
)


def _available_cjk_font() -> str | None:
    installed = {f.name for f in font_manager.fontManager.ttflist}
    return next((name for name in CJK_FONT_CANDIDATES if name in installed), None)


def setup() -> str | None:
    """设置全局作图样式，返回所选中文字体名。"""
    chosen = _available_cjk_font()
    if chosen is None:
        warnings.warn("未找到中文字体，图中的中文可能无法显示", UserWarning, stacklevel=2)
    sans = ["Arial"] + ([chosen] if chosen else []) + ["DejaVu Sans"]
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": sans,
            "axes.unicode_minus": False,
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "figure.figsize": (7.0, 4.0),
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return chosen
```

- [ ] **Step 4: 在 `src/apsim/__init__.py` 中导出子模块**

在 `from apsim.market import …` 下一行加入：

```python
from apsim import plotting
```

并把 `"plotting"` 加入 `__all__`。

- [ ] **Step 5: 确认测试通过，并检查本机选到的字体**

Run: `uv run pytest -v && uv run python -c "from apsim import plotting; print(plotting.setup())"`
Expected: 全部 PASS；打印出一个中文字体名（如 `Microsoft YaHei`）。若打印 `None`，在 Task 3 报告中说明，并把本机实际存在的中文字体名加入候选列表。

- [ ] **Step 6: 提交**

```bash
git add src/apsim tests/test_plotting.py
git commit -m "feat(apsim): 统一作图样式与中文字体"
```

---

### Task 4: 书籍骨架、排版样式与构建脚本

**Files:**
- Create: `book/myst.yml`、`book/_static/custom.css`、`book/references.bib`、`book/index.md`、`book/notation.md`、`book/part0/ch00-guide.md`（本任务只放最小可执行内容，Task 5 写正文）、`scripts/build.sh`、`scripts/serve.sh`

**Interfaces:**
- Consumes: `apsim.simulate_market`、`apsim.plotting.setup`
- Produces: `sh scripts/build.sh` 生成 `book/_build/html/index.html`；数学宏 `\E`、`\Var`、`\Cov`、`\R` 全书可用；引用键见 `references.bib`。

- [ ] **Step 1: 写 `book/myst.yml`**

```yaml
version: 1
project:
  title: 资产定价与机器学习：从理论到 A 股实证
  subtitle: 自包含教材
  language: zh
  bibliography:
    - references.bib
  math:
    '\E': '\mathbb{E}'
    '\Var': '\operatorname{Var}'
    '\Cov': '\operatorname{Cov}'
    '\R': '\mathbb{R}'
  numbering:
    equation: true
    figure: true
    table: true
  toc:
    - file: index.md
    - file: notation.md
    - title: 第零部分　准备
      children:
        - file: part0/ch00-guide.md
site:
  template: book-theme
  options:
    style: _static/custom.css
```

- [ ] **Step 2: 写 `book/_static/custom.css`**

```css
/* 屏幕阅读：英文和数字 Arial，中文微软雅黑；macOS 回退到苹方、冬青黑体 */
html,
body,
article {
  font-family: Arial, "Microsoft YaHei", "PingFang SC", "Hiragino Sans GB", sans-serif !important;
}

/* 代码：英文 Consolas，中文回退到微软雅黑或苹方 */
code,
pre,
kbd,
samp {
  font-family: Consolas, "Microsoft YaHei", "PingFang SC", monospace !important;
}

/* 中文正文行距略放宽 */
article p,
article li {
  line-height: 1.85;
}
```

- [ ] **Step 3: 写 `book/references.bib`**（只放有把握的条目，后续章节逐步增加）

```bibtex
@book{cochrane2005asset,
  author    = {Cochrane, John H.},
  title     = {Asset Pricing},
  edition   = {Revised},
  publisher = {Princeton University Press},
  year      = {2005}
}

@article{cochrane2011presidential,
  author  = {Cochrane, John H.},
  title   = {Presidential Address: Discount Rates},
  journal = {The Journal of Finance},
  volume  = {66},
  number  = {4},
  pages   = {1047--1108},
  year    = {2011}
}

@article{gu2020empirical,
  author  = {Gu, Shihao and Kelly, Bryan and Xiu, Dacheng},
  title   = {Empirical Asset Pricing via Machine Learning},
  journal = {The Review of Financial Studies},
  volume  = {33},
  number  = {5},
  pages   = {2223--2273},
  year    = {2020}
}

@article{kelly2019characteristics,
  author  = {Kelly, Bryan T. and Pruitt, Seth and Su, Yinan},
  title   = {Characteristics Are Covariances: A Unified Model of Risk and Return},
  journal = {Journal of Financial Economics},
  volume  = {134},
  number  = {3},
  pages   = {501--524},
  year    = {2019}
}

@article{liu2019size,
  author  = {Liu, Jianan and Stambaugh, Robert F. and Yuan, Yu},
  title   = {Size and Value in China},
  journal = {Journal of Financial Economics},
  volume  = {134},
  number  = {1},
  pages   = {48--69},
  year    = {2019}
}

@article{leippold2022machine,
  author  = {Leippold, Markus and Wang, Qian and Zhou, Wenyu},
  title   = {Machine Learning in the {Chinese} Stock Market},
  journal = {Journal of Financial Economics},
  volume  = {145},
  number  = {2},
  pages   = {64--82},
  year    = {2022}
}
```

注意：卷期页码在 Task 5 写作时逐条核对；无法核对的条目在正文引用处标“待核实”。

- [ ] **Step 4: 写最小的 `book/index.md`、`book/notation.md`、`book/part0/ch00-guide.md`**

`book/index.md`：
```markdown
# 前言

本书正在写作中。
```

`book/notation.md`：
```markdown
# 记号表

记号表在第 0 章完成时确定。
```

`book/part0/ch00-guide.md`：
````markdown
---
kernelspec:
  name: python3
  display_name: Python 3
---

# 第 0 章　如何使用本书

```{code-cell} python
from apsim import MarketParams, simulate_market, plotting

plotting.setup()
mkt = simulate_market(MarketParams(n_stocks=100, n_periods=120), seed=0)
mkt.panel.head()
```
````

- [ ] **Step 5: 写 `scripts/build.sh` 与 `scripts/serve.sh`**

`scripts/build.sh`：
```sh
#!/usr/bin/env sh
# 构建全书：执行全部代码单元，输出静态网页到 book/_build/html
set -e
cd "$(dirname "$0")/../book"
uv run jupyter book build --html --execute
```

`scripts/serve.sh`：
```sh
#!/usr/bin/env sh
# 在 http://localhost:8000 提供构建好的网页
set -e
cd "$(dirname "$0")/../book/_build/html"
uv run python -m http.server 8000
```

- [ ] **Step 6: 构建并检查**

Run: `sh scripts/build.sh 2>&1 | tee /tmp/build.log; ls book/_build/html/index.html; grep -iE "error|traceback" /tmp/build.log || echo "无错误"`
Expected: 生成 `book/_build/html/index.html`；日志中没有执行错误。
若 `myst.yml` 某个键不被接受（构建日志会给出警告），查阅 mystmd 文档修正后重试，并把修正记录到任务报告中。

- [ ] **Step 7: 检查执行结果与样式确实进入产物**

Run: `grep -rl "exret" book/_build/html | head -3; grep -rl "Microsoft YaHei" book/_build/html | head -3`
Expected: 两条命令都至少输出一个文件（第一条说明代码单元的输出已嵌入，第二条说明自定义样式已打包）。

- [ ] **Step 8: 提交**

```bash
git add book scripts
git commit -m "feat(book): 书籍骨架、中文排版样式与构建脚本"
```

---

### Task 5: 前言、记号表与第 0 章正文

**Files:**
- Modify: `book/index.md`、`book/notation.md`、`book/part0/ch00-guide.md`、`book/references.bib`（如需增加条目）

**Interfaces:**
- Consumes: `apsim.MarketParams`、`apsim.simulate_market`、`SimulatedMarket.sdf()`、`SimulatedMarket.sdf_weights`、`apsim.plotting.setup()`

**内容要求（正文全部用中文写作，遵守 Global Constraints 中的排版规则）：**

`book/index.md`（前言，约 1500～2500 字）：
1. 本书写给谁：资产定价方向、需要系统补理论与计量、目标是 ML 资产定价研究的读者。
2. 本书的定位：自包含教材；与 Cochrane 等经典教材及原始文献的关系；错误可能存在，读者应以原文核对，正文中“待核实”标记的含义。
3. 全书结构：四个部分各解决什么问题，部分之间的依赖关系（用一个简单的列表或表格表示）。
4. 每章的固定结构（9 项模板）以及习题三级的含义。
5. 为什么全书用模拟数据：已知真值、可检验方法能否恢复真值；以及正式研究必须换用 CSMAR／Wind 时点数据。
6. 如何构建与阅读（引用 README 中的命令）。

`book/notation.md`（记号表）：用表格列出，至少包括：
- 时间与截面下标 $t$、$i$；期数 $T$、股票数 $N$、因子数 $K$、特征数 $L$
- 总收益 $R_{t+1}$、无风险利率 $R^f_t$、超额收益 $R^e_{t+1}$（正文中横截面面板用 $r_{i,t}$ 表示超额收益，需在表中说明两种记号的关系）
- 价格 $p_t$、支付 $x_{t+1}$、随机贴现因子 $m_{t+1}$、基本定价方程 $p_t = \E_t[m_{t+1}x_{t+1}]$
- 因子 $f_t$、因子均值 $\mu$、因子协方差 $\Sigma_f$、SDF 系数 $b$、因子载荷 $\beta_{i,t}$、风险溢价 $\lambda$、定价误差 $\alpha_{i,t}$
- 特征 $z_{i,t}$、载荷矩阵 $\Gamma$、特征持续性 $\rho$、异质性波动 $s$
- 期望与条件期望 $\E[\cdot]$、$\E_t[\cdot]$，方差与协方差 $\Var$、$\Cov$
- 时点约定：下标 $t$ 表示第 $t$ 期末已知的信息；面板中特征比收益早一期
- 与 `apsim` 代码字段的对应关系（如 `exret` ↔ $r_{i,t}$，`z1` ↔ $z_{1,i,t-1}$）

`book/part0/ch00-guide.md`（第 0 章，约 8000～12000 字），按每章模板组织：
1. **学习目标**：会安装与构建本书；理解全书记号与时点约定；会用 `simulate_market` 生成数据并读懂每一列；理解“已知真值”的检验思路。
2. **动机**：研究中最常见的错误来自时点错位与对估计量性质的误解，模拟是检查这两类错误最便宜的手段。
3. **正文**：
   - 3.1 环境与构建（命令、常见问题：Node.js、内核）
   - 3.2 记号与时点约定（引用 `notation.md`，用一个时间轴的文字示意说明 $z_{i,t-1}$ 与 $r_{i,t}$ 的关系）
   - 3.3 模拟市场的模型：逐条写出 Task 2 中的模型方程，并编号
   - 3.4 定理：$m_t = 1 - b'(f_t-\mu)$、$b=\Sigma_f^{-1}\mu$ 满足 $\E[m_t]=1$、$\E[m_t f_t]=0$，并且 $\E[m_t r_{i,t}\mid z_{i,t-1}] = \alpha_{i,t}$。用 `proof:theorem` 与 `proof:proof` 写出逐步证明（需用到 $f_t$ 与 $z_{i,t-1}$、$\varepsilon_{i,t}$ 独立）。
   - 3.5 代码验证：用 `sympy` 对单因子情形符号验证 $\E[m f]=0$；用模拟数据验证三条结论，并展示样本值随 $T$ 增大收敛到真值（作一张收敛图）
4. **A 股视角**：本章的模拟市场与真实 A 股相比省略了什么（涨跌停、停牌、卖空限制、壳价值、财报披露滞后），这些在哪些章节加入。
5. **模拟实验**：用交互回归恢复 $\Gamma$（与 `tests/test_market.py` 中的做法一致），把估计值与真值并排列表。
6. **常见误区**：至少 3 条，如把同期特征当期初特征、把样本均值当总体值、忽视种子导致结果不可复现。
7. **本章小结**
8. **习题**：概念题 2 道、推导题 2 道、编程题 2 道，每道用 `exercise` 加 `solution`（`:class: dropdown`）。推导题之一：证明当 $\delta\neq 0$ 时用因子定价的时间序列回归截距等于 $\alpha_{i,t}$ 的期望。编程题之一：改变 $\rho$，观察特征持续性对 $\Gamma$ 估计误差的影响。
9. **文献与延伸阅读**：引用 `references.bib` 中的条目（如 Cochrane 2005、Kelly–Pruitt–Su 2019），用 `{cite:t}` 或 `@key` 语法。

- [ ] **Step 1: 写 `book/notation.md`**
- [ ] **Step 2: 写 `book/index.md`**
- [ ] **Step 3: 写 `book/part0/ch00-guide.md` 第 1～3 节**，每写完一个代码单元，用 `uv run python -c` 或在构建中确认可执行
- [ ] **Step 4: 写第 4～9 节**
- [ ] **Step 5: 构建并检查**

Run: `sh scripts/build.sh 2>&1 | tee /tmp/build.log; grep -iE "error|traceback|warning" /tmp/build.log || echo "无错误或警告"`
Expected: 构建成功；没有执行错误；如有引用或交叉引用警告，逐一修正。

- [ ] **Step 6: 排版自查**

Run: `grep -nP '[，。；：？！] |[\x{4e00}-\x{9fff}][,;:?!()]|\.\.\.|…(?!…)' book/index.md book/notation.md book/part0/ch00-guide.md | head -30`
Expected: 无输出，或只剩代码块与公式中的合法半角符号（逐条确认）。
同时人工确认：引号用“”；数值范围用“～”；省略号用“……”；破折号用“——”。

- [ ] **Step 7: 运行全部测试**

Run: `uv run pytest -v`
Expected: 全部 PASS

- [ ] **Step 8: 提交**

```bash
git add book
git commit -m "docs(book): 前言、记号表与第 0 章"
```

- [ ] **Step 9: 交给用户阅读**：运行 `sh scripts/serve.sh`，请用户在浏览器中阅读第 0 章并反馈。第 1 章的实施计划在用户确认第 0 章之后另行制定。
