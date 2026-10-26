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
