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


def test_golden_values():
    data = simulate_consumption(seed=0, n_years=3).data
    np.testing.assert_allclose(
        data["dc"].to_numpy(), [0.04887381909396252, 0.002081080472285172, 0.03471911340354076], rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        data["dd"].to_numpy(), [0.19028880952934726, -0.23246378128956882, -0.2992251090598254], rtol=0, atol=1e-12
    )
