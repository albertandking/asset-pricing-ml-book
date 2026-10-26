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
