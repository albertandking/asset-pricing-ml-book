"""消费型资产定价模型：幂效用、对数正态消费与红利增长、Lucas 树（第 1 章）。

模型（年度，参数均为假设值）
----
- 对数消费增长：Δc_{t+1} = μ_c + σ_c ε_{t+1}，ε ~ N(0, 1) 独立同分布。
- 对数红利增长：Δd_{t+1} = μ_d + σ_d η_{t+1}，η = ρ ε + sqrt(1 − ρ²) ν，ν ~ N(0, 1) 与 ε 独立。
- 幂效用的 SDF：m_{t+1} = δ exp(−γ Δc_{t+1})。
- E[m] = δ exp(−γ μ_c + γ² σ_c² / 2)，R^f = 1 / E[m]（常数）。
- 红利索取权：增长独立同分布时市盈率为常数 P/D = a / (1 − a)，
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
        """市盈率 P/D = a / (1 − a)；a ≥ 1 时价格无穷大，抛出 ValueError。"""
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
