"""模拟市场生成器：已知真实 SDF、因子与特征的横截面面板。

模型（记号见 book/notation.md；公式中向量均为列向量）
----
- 因子超额收益：f_t = mu + diag(sigma) u_t，u_t ~ N(0, I_K)，f_t 为 K 维。
- 特征：z_{i,t} = phi z_{i,t-1} + sqrt(1 - phi^2) e_{i,t}，平稳分布为 N(0, I_L)。
- 因子载荷：beta_{i,t} = Gamma' z~_{i,t-1}，z~_{i,t-1} = (1, z_{i,t-1}')'，Gamma 为 (L+1, K)。
- 定价误差：alpha_{i,t} = theta' z_{i,t-1}。
- 超额收益：r_{i,t} = alpha_{i,t} + beta_{i,t}' f_t + eps_{i,t}，eps_{i,t} ~ N(0, sigma_eps^2)。
- 真实 SDF：m_t = 1 - b'(f_t - mu)，b = Sigma_f^{-1} mu。

时点约定：面板中第 t 期的行，特征列为 z_{i,t-1}（期初已知），exret 为第 t 期实现的超额收益。
编号约定：period 从 1 开始（t = 1, ..., T），stock 从 0 开始（i = 0, ..., N-1）。

数组约定：数据数组“观测在行”，因子为 (T, K)，特征为 (T, N, L)；数组的一行是公式中列向量的转置，
所以 `x @ gamma` 按行计算的是 z~_{i,t-1}' Gamma = beta_{i,t}'。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# simulate_market 的随机子流，按此顺序由 SeedSequence.spawn 分配。
# 以后新增随机部分时只在末尾追加，不改变已有子流的抽样。
_STREAMS = ("factors", "initial_chars", "char_shocks", "idio_shocks")


def default_gamma(n_chars: int, n_factors: int) -> np.ndarray:
    """默认载荷矩阵 Gamma，形状 (L+1, K)。

    第 0 行（截距）只载荷第一个因子，取 1；第 j 行（j = 1, ..., min(L, K-1)）只载荷第 j+1 个因子，取 0.5；
    其余元素为 0。因此 L > K-1 时，第 K 个以后的特征是与载荷无关的“噪声特征”；
    L < K-1 时，第 L+2……K 个因子只有零载荷（对应的列全为 0），这些因子不进入任何股票的收益。
    """
    gamma = np.zeros((n_chars + 1, n_factors))
    gamma[0, 0] = 1.0
    for j in range(1, min(n_chars, n_factors - 1) + 1):
        gamma[j, j] = 0.5
    return gamma


def _float_tuple(values, name: str) -> tuple[float, ...]:
    out = tuple(float(v) for v in np.asarray(values, dtype=float).ravel())
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} 须全为有限数（不能含 NaN 或 inf）")
    return out


def _read_only(values) -> np.ndarray:
    arr = np.array(values, dtype=float, copy=True)
    arr.setflags(write=False)
    return arr


@dataclass(frozen=True)
class MarketParams:
    """模拟市场的参数。收益均为月度超额收益。

    构造时 `factor_mean`、`factor_vol`、`alpha_coef` 统一转为 float 元组；`gamma` 被复制为只读的 float 数组，
    之后修改传入的数组不影响本对象。相等判断与哈希把 `gamma` 的取值考虑在内。
    """

    n_stocks: int = 500
    n_periods: int = 240
    factor_mean: tuple[float, ...] = (0.006, 0.003, 0.004)
    factor_vol: tuple[float, ...] = (0.05, 0.03, 0.03)
    n_chars: int = 5
    char_persistence: float = 0.9
    idio_vol: float = 0.10
    gamma: np.ndarray | None = field(default=None, compare=False)
    alpha_coef: tuple[float, ...] | None = None
    _gamma_key: tuple[tuple[float, ...], ...] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        set_ = object.__setattr__
        if self.n_stocks < 1 or self.n_periods < 1 or self.n_chars < 0:
            raise ValueError("n_stocks、n_periods 须为正，n_chars 须非负")
        set_(self, "factor_mean", _float_tuple(self.factor_mean, "factor_mean"))
        set_(self, "factor_vol", _float_tuple(self.factor_vol, "factor_vol"))
        if len(self.factor_mean) != len(self.factor_vol) or len(self.factor_mean) == 0:
            raise ValueError("factor_mean 与 factor_vol 长度须相同且非空")
        if min(self.factor_vol) <= 0:
            raise ValueError("factor_vol 须为正")
        if not np.isfinite(self.char_persistence) or not 0.0 <= self.char_persistence < 1.0:
            raise ValueError("char_persistence 须在 [0, 1) 内")
        if not np.isfinite(self.idio_vol) or self.idio_vol < 0:
            raise ValueError("idio_vol 须为有限数且须非负")
        set_(self, "char_persistence", float(self.char_persistence))
        set_(self, "idio_vol", float(self.idio_vol))
        if self.gamma is not None:
            gamma = _read_only(self.gamma)
            if gamma.shape != (self.n_chars + 1, self.n_factors):
                raise ValueError(f"gamma 形状须为 {(self.n_chars + 1, self.n_factors)}")
            if not np.all(np.isfinite(gamma)):
                raise ValueError("gamma 须全为有限数（不能含 NaN 或 inf）")
            set_(self, "gamma", gamma)
            set_(self, "_gamma_key", tuple(tuple(float(v) for v in row) for row in gamma))
        if self.alpha_coef is not None:
            set_(self, "alpha_coef", _float_tuple(self.alpha_coef, "alpha_coef"))
            if len(self.alpha_coef) != self.n_chars:
                raise ValueError(f"alpha_coef 长度须为 n_chars={self.n_chars}")

    @property
    def n_factors(self) -> int:
        return len(self.factor_mean)

    @property
    def factor_cov(self) -> np.ndarray:
        """因子协方差矩阵 Sigma_f，形状 (K, K)。

        当前因子相互独立，Sigma_f = diag(sigma^2)。以后加入相关因子时，只需改这里与 simulate_market 中的因子生成，
        `sdf_weights` 等下游计算都通过本属性取 Sigma_f，无需改动。
        """
        return np.diag(np.asarray(self.factor_vol) ** 2)

    def resolved_gamma(self) -> np.ndarray:
        """实际使用的载荷矩阵 Gamma（只读副本，形状 (L+1, K)）；`gamma` 为 None 时取 `default_gamma`。"""
        if self.gamma is None:
            return _read_only(default_gamma(self.n_chars, self.n_factors))
        return _read_only(self.gamma)

    def resolved_theta(self) -> np.ndarray:
        """实际使用的定价误差系数 theta（只读副本，形状 (L,)）；`alpha_coef` 为 None 时取零向量。"""
        if self.alpha_coef is None:
            return _read_only(np.zeros(self.n_chars))
        return _read_only(self.alpha_coef)


@dataclass(frozen=True, eq=False)
class SimulatedMarket:
    """一次模拟的结果，附带全部真实参数。

    `factors`：(T, K) 的 DataFrame，行索引 period = 1, ..., T。
    `panel`：(T·N) 行的 DataFrame，索引 (period, stock)，stock = 0, ..., N-1；列 z1…zL 为期初特征 z_{i,t-1}。
    `gamma`（(L+1, K)）与 `theta`（(L,)）是只读数组，不与用户传入的数组共享内存。
    """

    params: MarketParams
    factors: pd.DataFrame
    panel: pd.DataFrame
    gamma: np.ndarray
    theta: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "gamma", _read_only(self.gamma))
        object.__setattr__(self, "theta", _read_only(self.theta))

    @property
    def sdf_weights(self) -> np.ndarray:
        """总体 SDF 系数 b，解线性方程组 Sigma_f b = mu 得到，形状 (K,)。"""
        mu = np.asarray(self.params.factor_mean)                  # (K,)
        return np.linalg.solve(self.params.factor_cov, mu)        # (K, K) b = (K,) -> b: (K,)

    def sdf(self) -> pd.Series:
        """真实 SDF 的实现值 m_t = 1 - b'(f_t - mu)，t = 1, ..., T。"""
        mu = np.asarray(self.params.factor_mean)
        m = 1.0 - (self.factors.to_numpy() - mu) @ self.sdf_weights   # (T, K) @ (K,) -> (T,)
        return pd.Series(m, index=self.factors.index, name="m")


def _substreams(seed: int | np.random.Generator) -> dict[str, np.random.Generator]:
    if isinstance(seed, np.random.Generator):
        children = seed.spawn(len(_STREAMS))
    elif isinstance(seed, (int, np.integer)) and not isinstance(seed, bool):
        children = [np.random.default_rng(s) for s in np.random.SeedSequence(int(seed)).spawn(len(_STREAMS))]
    else:
        raise TypeError(
            "seed 须为非负整数或 np.random.Generator；"
            "确实需要不可复现的随机数时，显式传入 seed=np.random.default_rng()"
        )
    return dict(zip(_STREAMS, children))


def simulate_market(params: MarketParams | None = None, *, seed: int | np.random.Generator) -> SimulatedMarket:
    """按 `MarketParams` 生成一次模拟市场。

    `seed` 必须以关键字传入，没有默认值：
    - 整数：由 `np.random.SeedSequence(seed)` 派生四个独立子流，依次用于因子、初始特征、特征冲击、异质性冲击。
      相同的整数给出完全相同的结果。
    - `np.random.Generator`：用 `Generator.spawn(4)` 从它派生同样四个子流。同一个 Generator 连续传入多次，
      每次得到不同的结果（适合蒙特卡洛重复）；两个以相同种子新建的 Generator 给出相同结果。
    以后新增随机部分时只追加子流，已有部分的抽样不受影响。
    """
    p = params if params is not None else MarketParams()
    rng = _substreams(seed)
    T, N, L, K = p.n_periods, p.n_stocks, p.n_chars, p.n_factors
    gamma, theta = p.resolved_gamma(), p.resolved_theta()        # (L+1, K)、(L,)

    # 因子：每行 f_t' = mu' + (sigma ∘ u_t)'，广播 (K,) * (T, K) -> (T, K)
    f = np.asarray(p.factor_mean) + np.asarray(p.factor_vol) * rng["factors"].standard_normal((T, K))

    # 特征：z[t] 存放 z_{.,t}（t = 0, ..., T），每期为 (N, L)
    z = np.empty((T + 1, N, L))
    z[0] = rng["initial_chars"].standard_normal((N, L))
    shocks = rng["char_shocks"].standard_normal((T, N, L))       # shocks[t-1] 为 e_{.,t}
    phi = p.char_persistence
    shock_scale = np.sqrt(1.0 - phi**2)
    for t in range(1, T + 1):
        z[t] = phi * z[t - 1] + shock_scale * shocks[t - 1]
    z_lag = z[:-1]                                                # (T, N, L)：第 t 期的行用 z_{.,t-1}

    x = np.concatenate([np.ones((T, N, 1)), z_lag], axis=2)      # (T, N, L+1)：每行为 z~_{i,t-1}'
    beta = x @ gamma                                              # (T, N, L+1) @ (L+1, K) -> (T, N, K)：每行 beta_{i,t}'
    alpha = z_lag @ theta                                         # (T, N, L) @ (L,) -> (T, N)：theta' z_{i,t-1}
    eps = p.idio_vol * rng["idio_shocks"].standard_normal((T, N))
    # einsum：对每个 (t, n) 求 sum_k beta[t, n, k] f[t, k]，即 beta_{i,t}' f_t；(T, N, K), (T, K) -> (T, N)
    exret = alpha + np.einsum("tnk,tk->tn", beta, f) + eps

    periods = pd.RangeIndex(1, T + 1, name="period")
    factors = pd.DataFrame(f, index=periods, columns=[f"f{k}" for k in range(1, K + 1)])
    index = pd.MultiIndex.from_product([periods, range(N)], names=["period", "stock"])
    columns = {f"z{j}": z_lag[:, :, j - 1].ravel() for j in range(1, L + 1)}
    columns |= {f"beta{k}": beta[:, :, k - 1].ravel() for k in range(1, K + 1)}
    columns |= {"alpha": alpha.ravel(), "exret": exret.ravel()}
    panel = pd.DataFrame(columns, index=index)

    return SimulatedMarket(params=p, factors=factors, panel=panel, gamma=gamma, theta=theta)
