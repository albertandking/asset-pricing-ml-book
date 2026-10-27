import numpy as np
import pandas as pd
import pytest

from apsim import MarketParams, SimulatedMarket, default_gamma, market, simulate_market


def test_default_gamma_layout():
    g = default_gamma(n_chars=5, n_factors=3)
    assert g.shape == (6, 3)
    assert g[0, 0] == 1.0
    assert g[1, 1] == 0.5 and g[2, 2] == 0.5
    assert np.count_nonzero(g) == 3


def test_default_gamma_few_chars():
    # L < K-1：只有前 L 个特征分别载荷因子 2……L+1，因子 L+2……K 的整列为 0
    g = default_gamma(n_chars=1, n_factors=4)
    assert g.shape == (2, 4)
    np.testing.assert_array_equal(g, [[1.0, 0.0, 0.0, 0.0], [0.0, 0.5, 0.0, 0.0]])
    assert default_gamma(n_chars=0, n_factors=2).tolist() == [[1.0, 0.0]]


def test_shapes_and_columns():
    p = MarketParams(n_stocks=50, n_periods=24)
    mkt = simulate_market(p, seed=1)
    assert isinstance(mkt, SimulatedMarket)
    assert mkt.factors.shape == (24, 3)
    assert list(mkt.factors.columns) == ["f1", "f2", "f3"]
    assert mkt.factors.index.name == "period"
    assert list(mkt.factors.index) == list(range(1, 25))
    assert len(mkt.panel) == 24 * 50
    assert mkt.panel.index.names == ["period", "stock"]
    assert mkt.panel.index.get_level_values("stock").min() == 0
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


def test_seed_is_required_keyword():
    p = MarketParams(n_stocks=5, n_periods=3)
    with pytest.raises(TypeError):
        simulate_market(p)
    with pytest.raises(TypeError):
        simulate_market(p, 0)
    with pytest.raises(TypeError):
        simulate_market(p, seed=None)


def test_generator_seed():
    p = MarketParams(n_stocks=5, n_periods=6)
    a = simulate_market(p, seed=np.random.default_rng(5))
    b = simulate_market(p, seed=np.random.default_rng(5))
    pd.testing.assert_frame_equal(a.panel, b.panel)
    rng = np.random.default_rng(5)
    first, second = simulate_market(p, seed=rng), simulate_market(p, seed=rng)
    assert not np.allclose(first.panel["exret"], second.panel["exret"])


def test_generator_seed_stable_when_streams_appended(monkeypatch):
    """同一个 Generator 连续调用两次：在子流列表末尾追加一条子流后，两次调用的结果都不变。"""
    p = MarketParams(n_stocks=5, n_periods=6)

    def two_calls():
        rng = np.random.default_rng(5)
        return simulate_market(p, seed=rng), simulate_market(p, seed=rng)

    before = two_calls()
    monkeypatch.setattr(market, "_STREAMS", (*market._STREAMS, "future_stream"))
    assert len(market._STREAMS) == 5
    after = two_calls()
    for old, new in zip(before, after):
        pd.testing.assert_frame_equal(old.factors, new.factors)
        pd.testing.assert_frame_equal(old.panel, new.panel)


def test_generator_seed_takes_one_spawn_key_per_call():
    """每次调用只从父 Generator 取 1 个 spawn 键。"""
    rng = np.random.default_rng(5)
    simulate_market(MarketParams(n_stocks=5, n_periods=6), seed=rng)
    assert rng.bit_generator.seed_seq.n_children_spawned == 1


def test_substreams_independent_of_other_parts():
    # 因子只来自自己的子流：改变特征与异质性部分的参数，因子路径不变
    a = simulate_market(MarketParams(n_stocks=5, n_periods=10), seed=4)
    b = simulate_market(MarketParams(n_stocks=50, n_periods=10, n_chars=2, idio_vol=0.3), seed=4)
    pd.testing.assert_frame_equal(a.factors, b.factors)


def test_panel_timing_alignment():
    """面板第 t 行的 z 是 z_{t-1}：第 1 行等于初值 z_0，各行按 AR(1) 递推，收益由同一行的 z 生成。"""
    p = MarketParams(n_stocks=7, n_periods=6, idio_vol=0.0, char_persistence=0.8)
    seed = 3
    mkt = simulate_market(p, seed=seed)
    N, T, L, K = p.n_stocks, p.n_periods, p.n_chars, p.n_factors
    Z = np.stack([mkt.panel[f"z{j}"].unstack("stock").to_numpy() for j in range(1, L + 1)], axis=2)  # (T, N, L)
    F = mkt.factors.to_numpy()                                    # (T, K)
    R = mkt.panel["exret"].unstack("stock").to_numpy()            # (T, N)

    rng_f, rng_z0, rng_e, _ = (np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4))
    expected_f = np.asarray(p.factor_mean) + np.asarray(p.factor_vol) * rng_f.standard_normal((T, K))
    np.testing.assert_array_equal(F, expected_f)
    z0 = rng_z0.standard_normal((N, L))
    e = rng_e.standard_normal((T, N, L))
    np.testing.assert_array_equal(Z[0], z0)                       # 第 1 期的行存放 z_0
    phi = p.char_persistence
    for t in range(1, T):                                         # 第 t+1 期的行 = φ·(第 t 期的行) + 冲击
        np.testing.assert_allclose(Z[t], phi * Z[t - 1] + np.sqrt(1 - phi**2) * e[t - 1], rtol=0, atol=1e-14)

    X = np.concatenate([np.ones((T, N, 1)), Z], axis=2)          # (T, N, L+1)
    B = X @ mkt.gamma                                             # (T, N, L+1) @ (L+1, K) -> (T, N, K)
    np.testing.assert_allclose(mkt.panel[[f"beta{k}" for k in range(1, K + 1)]].to_numpy(), B.reshape(T * N, K))
    np.testing.assert_allclose(R, np.einsum("tnk,tk->tn", B, F), rtol=0, atol=1e-14)
    # 错位一期（用下一行的 z 解释本行收益）必然对不上
    B_next = X[1:] @ mkt.gamma
    assert not np.allclose(R[:-1], np.einsum("tnk,tk->tn", B_next, F[:-1]))


GOLDEN_F1 = [0.07818454773490628, -0.02387837929157224, 0.02607867010531114]
GOLDEN_EXRET = [-0.060073704950775894, 0.09470697034304888, 0.15273663241422036, 0.12631859109045895]
GOLDEN_Z1 = [0.8050894723742356, -0.34795066393722024, 0.9635636709238318]


def test_golden_values():
    """固定随机流：一旦抽样顺序或子流分配改变，这里立即失败。"""
    mkt = simulate_market(MarketParams(n_stocks=3, n_periods=4), seed=0)
    np.testing.assert_allclose(mkt.factors.iloc[0].to_numpy(), GOLDEN_F1, rtol=0, atol=1e-12)
    np.testing.assert_allclose(mkt.panel["exret"].to_numpy()[:4], GOLDEN_EXRET, rtol=0, atol=1e-12)
    np.testing.assert_allclose(mkt.panel["z1"].to_numpy()[:3], GOLDEN_Z1, rtol=0, atol=1e-12)


def test_sdf_weights_solve_pricing_equation():
    p = MarketParams(n_stocks=3, n_periods=5)
    mkt = simulate_market(p, seed=0)
    np.testing.assert_allclose(p.factor_cov, np.diag(np.asarray(p.factor_vol) ** 2))
    np.testing.assert_allclose(p.factor_cov @ mkt.sdf_weights, np.asarray(p.factor_mean), rtol=1e-12)


def test_theta_attribute():
    theta = (0.01, -0.005, 0.0, 0.0, 0.0)
    mkt = simulate_market(MarketParams(n_stocks=3, n_periods=5, alpha_coef=theta), seed=0)
    np.testing.assert_array_equal(mkt.theta, theta)
    default = simulate_market(MarketParams(n_stocks=3, n_periods=5), seed=0)
    np.testing.assert_array_equal(default.theta, np.zeros(5))


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
    p = MarketParams(n_stocks=200, n_periods=5000, alpha_coef=(0.01, 0.0, 0.0, 0.0, 0.0))
    mkt = simulate_market(p, seed=0)
    m = mkt.sdf().reindex(mkt.panel.index.get_level_values("period")).to_numpy()
    y = mkt.panel["exret"].to_numpy() * m
    z1 = mkt.panel["z1"].to_numpy()
    slope = np.cov(y, z1)[0, 1] / np.var(z1, ddof=1)
    assert abs(slope - 0.01) < 0.002


def test_characteristic_persistence():
    p = MarketParams(n_stocks=200, n_periods=100, char_persistence=0.9)
    z1 = simulate_market(p, seed=0).panel["z1"].unstack("stock")
    phi_hat = np.corrcoef(z1.iloc[1:].to_numpy().ravel(), z1.iloc[:-1].to_numpy().ravel())[0, 1]
    assert abs(phi_hat - 0.9) < 0.02


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


def test_params_normalized_and_gamma_in_equality():
    g = default_gamma(5, 3)
    a = MarketParams(gamma=g)
    b = MarketParams(gamma=g * 2)
    assert a != b
    assert a == MarketParams(gamma=g.copy())
    assert hash(a) == hash(MarketParams(gamma=g.copy()))
    assert a.gamma.dtype == float and not a.gamma.flags.writeable
    assert isinstance(a.factor_mean, tuple) and all(type(v) is float for v in a.factor_mean)
    c = MarketParams(factor_mean=[1, 2, 3], factor_vol=np.array([0.1, 0.2, 0.3]), alpha_coef=[0, 0, 0, 0, 1])
    assert c.factor_mean == (1.0, 2.0, 3.0) and type(c.factor_mean[0]) is float
    assert c.factor_vol == (0.1, 0.2, 0.3) and type(c.factor_vol[0]) is float
    assert c.alpha_coef == (0.0, 0.0, 0.0, 0.0, 1.0) and type(c.alpha_coef[0]) is float
    hash(c)


def test_gamma_copied_and_read_only():
    g = default_gamma(5, 3)
    theta = np.array([0.01, 0.0, 0.0, 0.0, 0.0])
    p = MarketParams(n_stocks=3, n_periods=4, gamma=g, alpha_coef=theta)
    mkt = simulate_market(p, seed=0)
    g[1, 1] = 99.0
    theta[0] = 99.0
    assert p.gamma[1, 1] == 0.5 and mkt.gamma[1, 1] == 0.5
    assert p.alpha_coef[0] == 0.01 and mkt.theta[0] == 0.01
    assert not np.shares_memory(mkt.gamma, g)
    with pytest.raises(ValueError):
        mkt.gamma[0, 0] = 2.0
    with pytest.raises(ValueError):
        mkt.theta[0] = 2.0
    with pytest.raises(ValueError):
        p.gamma[0, 0] = 2.0
    default_mkt = simulate_market(MarketParams(n_stocks=3, n_periods=4), seed=0)
    with pytest.raises(ValueError):
        default_mkt.gamma[0, 0] = 2.0


def test_invalid_params_raise():
    with pytest.raises(ValueError):
        MarketParams(factor_mean=(0.01, 0.02), factor_vol=(0.05,))
    with pytest.raises(ValueError):
        MarketParams(char_persistence=1.0)
    with pytest.raises(ValueError):
        MarketParams(alpha_coef=(0.01,))
    with pytest.raises(ValueError):
        MarketParams(gamma=np.zeros((2, 2)))
    with pytest.raises(ValueError, match="须非负"):
        MarketParams(idio_vol=-0.1)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_params_raise(bad):
    with pytest.raises(ValueError):
        MarketParams(factor_mean=(bad, 0.003, 0.004))
    with pytest.raises(ValueError):
        MarketParams(factor_vol=(0.05, bad, 0.03))
    with pytest.raises(ValueError):
        MarketParams(idio_vol=bad)
    with pytest.raises(ValueError):
        MarketParams(char_persistence=bad)
    with pytest.raises(ValueError):
        MarketParams(alpha_coef=(bad, 0.0, 0.0, 0.0, 0.0))
    g = default_gamma(5, 3)
    g[2, 1] = bad
    with pytest.raises(ValueError):
        MarketParams(gamma=g)


def test_default_gamma_equals_explicit_default():
    """gamma=None 与显式传入默认矩阵：比较相等、哈希相同，模拟结果也相同。"""
    a = MarketParams()
    b = MarketParams(gamma=default_gamma(5, 3))
    assert a == b and hash(a) == hash(b)
    assert a.gamma is None                                         # 不改变用户看到的字段
    assert MarketParams(n_chars=2) == MarketParams(n_chars=2, gamma=default_gamma(2, 3))
    assert a != MarketParams(gamma=default_gamma(5, 3) * 2)
    pd.testing.assert_frame_equal(simulate_market(a, seed=0).panel, simulate_market(b, seed=0).panel)


@pytest.mark.parametrize("name", ["n_stocks", "n_periods", "n_chars"])
@pytest.mark.parametrize("bad", [5.0, 5.5, True, "5", None])
def test_integer_params_reject_non_integers(name, bad):
    with pytest.raises(TypeError):
        MarketParams(**{name: bad})


@pytest.mark.parametrize("name", ["n_stocks", "n_periods", "n_chars"])
def test_integer_params_accept_numpy_integers(name):
    p = MarketParams(**{name: np.int64(4)})
    assert getattr(p, name) == 4 and type(getattr(p, name)) is int
    assert p == MarketParams(**{name: 4}) and hash(p) == hash(MarketParams(**{name: 4}))
