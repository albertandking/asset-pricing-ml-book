from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import apsim
from apsim import MarketParams, simulate_market
from apsim.stats import FamaMacBethResult, fama_macbeth, interaction_design, ols


def test_stats_exported_from_package():
    assert apsim.stats.ols is ols and "stats" in apsim.__all__


def test_ols_hand_computed():
    """一元回归 y = a + b x 的手算结果。

    x = (0, 1, 2, 3)，y = (1, 3, 2, 5)：x̄ = 1.5，ȳ = 2.75，Sxx = 5，Sxy = 5.5，
    b = 1.1，a = 2.75 − 1.1 × 1.5 = 1.1；残差 (−0.1, 0.8, −1.3, 0.6)，残差平方和 2.7，σ̂² = 2.7 / (4 − 2) = 1.35；
    X'X = [[4, 6], [6, 14]]，(X'X)⁻¹ = [[14, −6], [−6, 4]] / 20，对角元 0.7、0.2。
    """
    X = np.column_stack([np.ones(4), np.arange(4.0)])
    y = np.array([1.0, 3.0, 2.0, 5.0])
    coef, se = ols(X, y)
    np.testing.assert_allclose(coef, [1.1, 1.1], rtol=0, atol=1e-12)
    np.testing.assert_allclose(se, [np.sqrt(1.35 * 0.7), np.sqrt(1.35 * 0.2)], rtol=0, atol=1e-12)


def test_ols_exact_fit_and_shapes():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 3))
    b = np.array([0.5, -1.0, 2.0])
    coef, se = ols(X, X @ b)
    assert coef.shape == (3,) and se.shape == (3,)
    np.testing.assert_allclose(coef, b, rtol=0, atol=1e-12)
    np.testing.assert_allclose(se, 0.0, atol=1e-12)


def _toy_market():
    """手工构造的小市场：T = 2 期、N = 2 只股票、L = 1 个特征、K = 2 个因子。"""
    periods = pd.RangeIndex(1, 3, name="period")
    factors = pd.DataFrame([[2.0, 3.0], [-1.0, 4.0]], index=periods, columns=["f1", "f2"])
    index = pd.MultiIndex.from_product([periods, range(2)], names=["period", "stock"])
    panel = pd.DataFrame({"z1": [5.0, -2.0, 0.5, 1.0], "exret": [0.0, 0.0, 0.0, 0.0]}, index=index)
    return SimpleNamespace(panel=panel, factors=factors, params=SimpleNamespace(n_chars=1))


def test_interaction_design_hand_computed():
    # 每行为 (z̃ ⊗ f)' = (1·f1, 1·f2, z1·f1, z1·f2)，z̃ = (1, z1)'
    X, Z = interaction_design(_toy_market())
    expected = [
        [2.0, 3.0, 10.0, 15.0],      # 第 1 期，股票 0：z1 = 5，f = (2, 3)
        [2.0, 3.0, -4.0, -6.0],      # 第 1 期，股票 1：z1 = −2
        [-1.0, 4.0, -0.5, 2.0],      # 第 2 期，股票 0：z1 = 0.5，f = (−1, 4)
        [-1.0, 4.0, -1.0, 4.0],      # 第 2 期，股票 1：z1 = 1
    ]
    np.testing.assert_array_equal(X, expected)
    np.testing.assert_array_equal(Z, [[5.0], [-2.0], [0.5], [1.0]])


def test_interaction_design_matches_kron_and_vec():
    mkt = simulate_market(MarketParams(n_stocks=4, n_periods=3, n_chars=2), seed=1)
    X, Z = interaction_design(mkt)
    L, K = mkt.params.n_chars, mkt.params.n_factors
    assert X.shape == (12, (L + 1) * K) and Z.shape == (12, L)
    zt = np.column_stack([np.ones(12), Z])                                   # (TN, L+1)
    f_row = mkt.factors.reindex(mkt.panel.index.get_level_values("period")).to_numpy()
    for n in range(12):
        np.testing.assert_allclose(X[n], np.kron(zt[n], f_row[n]))
    beta = mkt.panel[[f"beta{k}" for k in range(1, K + 1)]].to_numpy()
    np.testing.assert_allclose(X @ mkt.gamma.ravel(), np.sum(beta * f_row, axis=1))   # X vec(Γ') = β'f


def test_fama_macbeth_hand_computed():
    """每期精确拟合，各期系数已知：(1, 2)、(3, 0)、(2, 4)。

    均值 (2, 2)；截距的离差 (−1, 1, 0)，样本方差 2/2 = 1，标准误 1/√3；
    斜率的离差 (0, −2, 2)，样本方差 8/2 = 4，标准误 2/√3。
    """
    x = np.array([0.0, 1.0, 2.0])
    X = np.broadcast_to(np.column_stack([np.ones(3), x]), (3, 3, 2))          # (T, N, p)
    coefs_true = np.array([[1.0, 2.0], [3.0, 0.0], [2.0, 4.0]])               # (T, p)
    y = coefs_true[:, :1] + coefs_true[:, 1:] * x                              # (T, N)
    res = fama_macbeth(X, y)
    assert isinstance(res, FamaMacBethResult)
    np.testing.assert_allclose(res.coefs, coefs_true, rtol=0, atol=1e-12)
    np.testing.assert_allclose(res.mean, [2.0, 2.0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(res.se, [1 / np.sqrt(3), 2 / np.sqrt(3)], rtol=0, atol=1e-12)
    coefs, mean, se = res                                                      # 可以直接解包
    assert coefs.shape == (3, 2) and mean.shape == (2,) and se.shape == (2,)


def test_fama_macbeth_matches_per_period_ols():
    rng = np.random.default_rng(3)
    T, N, p = 6, 20, 3
    X = np.concatenate([np.ones((T, N, 1)), rng.standard_normal((T, N, p - 1))], axis=2)
    y = rng.standard_normal((T, N))
    res = fama_macbeth(X, y)
    per_period = np.array([ols(X[t], y[t])[0] for t in range(T)])
    np.testing.assert_allclose(res.coefs, per_period, rtol=0, atol=1e-12)
    np.testing.assert_allclose(res.mean, per_period.mean(axis=0), rtol=0, atol=1e-12)
    np.testing.assert_allclose(res.se, per_period.std(axis=0, ddof=1) / np.sqrt(T), rtol=0, atol=1e-12)


def test_fama_macbeth_rejects_bad_shapes():
    with pytest.raises(ValueError):
        fama_macbeth(np.ones((3, 4)), np.ones((3, 4)))                         # X 须为 3 维
    with pytest.raises(ValueError):
        fama_macbeth(np.ones((3, 4, 2)), np.ones((3, 5)))                      # y 须为 (T, N)
    with pytest.raises(ValueError):
        fama_macbeth(np.ones((1, 4, 2)), np.ones((1, 4)))                      # 至少两期才能算标准误
