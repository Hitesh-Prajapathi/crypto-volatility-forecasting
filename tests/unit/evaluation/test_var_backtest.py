"""Tests for volforecast.evaluation.var_backtest."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from volforecast.evaluation.var_backtest import (
    acerbi_szekely_pvalue,
    acerbi_szekely_z2,
    backtest_summary,
    basel_traffic_light,
    christoffersen_cc,
    christoffersen_independence,
    expected_shortfall,
    fit_student_t_nu,
    hits,
    kupiec_pof,
    rolling_traffic_light,
    standardized_es,
    standardized_quantile,
    value_at_risk,
)


def _unit_t(rng: np.random.Generator, nu: float, size: int | tuple[int, ...]) -> np.ndarray:
    return rng.standard_t(nu, size=size) * np.sqrt((nu - 2.0) / nu)


# ── standardized distribution ────────────────────────────────────────────────
def test_standardized_normal_quantile() -> None:
    assert standardized_quantile(0.05) == pytest.approx(-1.6448536269514722)
    assert standardized_quantile(0.01) == pytest.approx(-2.3263478740408408)


def test_standardized_t_quantile_formula() -> None:
    nu = 5.0
    expected = stats.t.ppf(0.01, nu) * np.sqrt(3.0 / 5.0)
    assert standardized_quantile(0.01, "t", nu=nu) == pytest.approx(expected)


@pytest.mark.parametrize("alpha", [0.05, 0.01])
def test_standardized_t_simulation_unit_variance_and_quantile(alpha: float) -> None:
    nu = 8.0
    rng = np.random.default_rng(20)
    z = _unit_t(rng, nu, 2_000_000)
    assert z.var() == pytest.approx(1.0, abs=0.02)
    assert np.quantile(z, alpha) == pytest.approx(
        standardized_quantile(alpha, "t", nu=nu), abs=0.02
    )


@pytest.mark.parametrize(("dist", "nu"), [("normal", None), ("t", 5.0)])
def test_closed_form_es_matches_monte_carlo(dist: str, nu: float | None) -> None:
    rng = np.random.default_rng(21)
    z = rng.standard_normal(3_000_000) if dist == "normal" else _unit_t(rng, 5.0, 3_000_000)
    for alpha in (0.05, 0.01):
        q = standardized_quantile(alpha, dist, nu=nu)  # type: ignore[arg-type]
        mc = -z[z <= q].mean()
        assert standardized_es(alpha, dist, nu=nu) == pytest.approx(mc, rel=0.03)  # type: ignore[arg-type]


def test_fhs_quantile_and_es_from_residuals() -> None:
    rng = np.random.default_rng(22)
    z = _unit_t(rng, 6.0, 5000)
    q = standardized_quantile(0.05, "fhs", std_resid=z)
    assert q == pytest.approx(np.quantile(z, 0.05))
    es = standardized_es(0.05, "fhs", std_resid=z)
    assert es == pytest.approx(-z[z <= q].mean())
    assert es > -q
    with pytest.raises(ValueError, match="std_resid"):
        standardized_quantile(0.05, "fhs")


def test_argument_validation() -> None:
    with pytest.raises(ValueError, match="nu"):
        standardized_quantile(0.05, "t")
    with pytest.raises(ValueError, match="nu must be > 2"):
        standardized_quantile(0.05, "t", nu=2.0)
    with pytest.raises(ValueError, match="alpha"):
        standardized_quantile(1.5)
    with pytest.raises(ValueError, match="strictly positive"):
        value_at_risk(np.array([0.01, 0.0]), 0.05)


def test_fit_student_t_nu_recovers_parameter() -> None:
    rng = np.random.default_rng(23)
    nu_hat = fit_student_t_nu(_unit_t(rng, 5.0, 30_000))
    assert nu_hat == pytest.approx(5.0, abs=0.35)
    assert fit_student_t_nu(rng.standard_normal(30_000)) > 30.0
    assert 2.05 <= fit_student_t_nu(_unit_t(rng, 2.2, 5_000)) <= 3.5


# ── VaR / ES ─────────────────────────────────────────────────────────────────
def test_var_and_es_normal_closed_form() -> None:
    h = np.array([0.0004, 0.0009])  # daily vols 2% and 3%
    var = value_at_risk(h, 0.05)
    np.testing.assert_allclose(var, 1.6448536269514722 * np.sqrt(h))
    es = expected_shortfall(h, 0.05)
    np.testing.assert_allclose(es, stats.norm.pdf(1.6448536269514722) / 0.05 * np.sqrt(h))
    assert np.all(var > 0)


def test_var_with_mu_and_vectorization() -> None:
    h = np.array([0.0004, 0.0009, 0.0016])
    var = value_at_risk(h, 0.01, mu=0.001)
    np.testing.assert_allclose(var, -(0.001 + stats.norm.ppf(0.01) * np.sqrt(h)))
    var_arr_mu = value_at_risk(h, 0.01, mu=np.array([0.0, 0.001, 0.002]))
    assert var_arr_mu.shape == (3,)


@pytest.mark.parametrize("alpha", [0.05, 0.01])
@pytest.mark.parametrize("mu", [0.0, 0.002, -0.001])
@pytest.mark.parametrize("dist", ["normal", "t", "fhs"])
def test_es_at_least_var(alpha: float, mu: float, dist: str) -> None:
    rng = np.random.default_rng(24)
    h = np.exp(rng.normal(-8.0, 1.0, 500))
    resid = _unit_t(rng, 5.0, 4000)
    kw = {"nu": 5.0, "std_resid": resid}
    var = value_at_risk(h, alpha, dist, mu=mu, **kw)  # type: ignore[arg-type]
    es = expected_shortfall(h, alpha, dist, mu=mu, **kw)  # type: ignore[arg-type]
    assert np.all(es >= var)


def test_t_es_exceeds_normal_es_for_fat_tails_at_1pct() -> None:
    h = np.array([0.0004])
    assert expected_shortfall(h, 0.01, "t", nu=4.0)[0] > expected_shortfall(h, 0.01)[0]


def test_hits() -> None:
    r = np.array([-0.05, -0.02, 0.01, -0.0301])
    var = np.array([0.03, 0.03, 0.03, 0.03])
    np.testing.assert_array_equal(hits(r, var), [True, False, False, True])
    assert hits(r, var).dtype == np.bool_
    assert not hits(np.array([-0.03]), np.array([0.03]))[0]  # strict inequality
    with pytest.raises(ValueError, match="shapes"):
        hits(r, var[:2])


# ── Kupiec ───────────────────────────────────────────────────────────────────
def _hits_from_positions(n: int, positions: list[int]) -> np.ndarray:
    h = np.zeros(n, dtype=bool)
    h[positions] = True
    return h


def test_kupiec_matches_hand_computation() -> None:
    n, x, alpha = 100, 3, 0.05
    h = _hits_from_positions(n, [10, 11, 50])
    ll0 = (n - x) * np.log(1 - alpha) + x * np.log(alpha)
    ll1 = (n - x) * np.log(1 - x / n) + x * np.log(x / n)
    lr = -2.0 * (ll0 - ll1)
    out = kupiec_pof(h, alpha)
    assert out.statistic == pytest.approx(lr)
    assert out.statistic == pytest.approx(0.9769, abs=1e-3)  # independent numeric value
    assert out.pvalue == pytest.approx(stats.chi2.sf(lr, df=1))
    assert out.detail is not None
    assert out.detail["hits"] == 3
    assert out.detail["n"] == 100


def test_kupiec_zero_and_all_hits_edge_cases() -> None:
    n, alpha = 250, 0.01
    zero = kupiec_pof(np.zeros(n, dtype=bool), alpha)
    assert zero.statistic == pytest.approx(-2.0 * n * np.log(1 - alpha))
    assert np.isfinite(zero.pvalue)
    allh = kupiec_pof(np.ones(n, dtype=bool), alpha)
    assert allh.statistic == pytest.approx(-2.0 * n * np.log(alpha))
    assert allh.pvalue < 1e-100 or allh.pvalue == 0.0
    perfect = kupiec_pof(_hits_from_positions(100, [3, 20, 40, 60, 80]), 0.05)
    assert perfect.statistic == pytest.approx(0.0, abs=1e-12)
    assert perfect.pvalue == pytest.approx(1.0)


def test_kupiec_accepts_zero_one_ints_and_rejects_other_values() -> None:
    h = _hits_from_positions(100, [1, 2, 3])
    assert kupiec_pof(h.astype(int), 0.05).statistic == pytest.approx(kupiec_pof(h, 0.05).statistic)
    with pytest.raises(ValueError, match="boolean"):
        kupiec_pof(np.array([0, 2, 1]), 0.05)


# ── Christoffersen ───────────────────────────────────────────────────────────
def test_christoffersen_independence_matches_hand_computation() -> None:
    n, alpha = 100, 0.05
    h = _hits_from_positions(n, [10, 11, 50])
    # explicit transition counts over t = 1..99:
    #   01: (9->10) and (49->50) -> 2;  11: (10->11) -> 1;
    #   10: (11->12),(50->51) -> 2;  00: 99-5 = 94
    n00, n01, n10, n11 = 94, 2, 2, 1
    pi01, pi11, pi = n01 / (n00 + n01), n11 / (n10 + n11), (n01 + n11) / 99
    ll0 = (n00 + n10) * np.log(1 - pi) + (n01 + n11) * np.log(pi)
    ll1 = n00 * np.log(1 - pi01) + n01 * np.log(pi01) + n10 * np.log(1 - pi11) + n11 * np.log(pi11)
    lr_ind = -2.0 * (ll0 - ll1)
    ind = christoffersen_independence(h)
    assert ind.detail == {"n00": n00, "n01": n01, "n10": n10, "n11": n11}
    assert ind.statistic == pytest.approx(lr_ind)
    assert ind.pvalue == pytest.approx(stats.chi2.sf(lr_ind, df=1))

    cc = christoffersen_cc(h, alpha)
    assert cc.statistic == pytest.approx(lr_ind + kupiec_pof(h, alpha).statistic)
    assert cc.pvalue == pytest.approx(stats.chi2.sf(cc.statistic, df=2))


def test_christoffersen_clustered_hits_are_rejected() -> None:
    h = np.zeros(500, dtype=bool)
    h[100:110] = True
    h[300:310] = True  # 20 hits (4%), all clustered
    assert christoffersen_independence(h).pvalue < 0.01
    assert kupiec_pof(h, 0.04).pvalue == pytest.approx(1.0)
    assert christoffersen_cc(h, 0.04).pvalue < 0.01


def test_christoffersen_edge_cases() -> None:
    zero = christoffersen_independence(np.zeros(50, dtype=bool))
    assert zero.statistic == pytest.approx(0.0, abs=1e-12)
    assert zero.pvalue == pytest.approx(1.0)
    allh = christoffersen_independence(np.ones(50, dtype=bool))
    assert allh.statistic == pytest.approx(0.0, abs=1e-12)
    # single hit: no 11 transitions, finite statistic
    one = christoffersen_independence(_hits_from_positions(50, [25]))
    assert np.isfinite(one.statistic)
    assert one.statistic >= 0
    # alternating: strongly negative dependence, still well-defined
    alt = christoffersen_independence(np.tile([True, False], 50))
    assert np.isfinite(alt.statistic)
    assert alt.pvalue < 0.01
    cc0 = christoffersen_cc(np.zeros(250, dtype=bool), 0.01)
    assert np.isfinite(cc0.pvalue)
    with pytest.raises(ValueError, match="at least 2"):
        christoffersen_independence(np.array([True]))


# ── Basel traffic light ──────────────────────────────────────────────────────
@pytest.mark.parametrize(
    ("x", "zone"),
    [(0, "green"), (4, "green"), (5, "yellow"), (9, "yellow"), (10, "red"), (25, "red")],
)
def test_basel_traffic_light_250_99(x: int, zone: str) -> None:
    assert basel_traffic_light(x) == zone


def test_basel_traffic_light_follows_binomial_cdf() -> None:
    cdf = stats.binom.cdf
    assert cdf(4, 250, 0.01) < 0.95 <= cdf(5, 250, 0.01)
    assert cdf(9, 250, 0.01) < 0.9999 <= cdf(10, 250, 0.01)
    assert basel_traffic_light(3, n_obs=500, coverage=0.99) == "green"
    assert basel_traffic_light(0, n_obs=250, coverage=0.975) == "green"
    with pytest.raises(ValueError, match="n_breaches"):
        basel_traffic_light(300, n_obs=250)


def test_rolling_traffic_light() -> None:
    h = np.zeros(400, dtype=bool)
    h[300:305] = True  # 5 hits -> yellow while all inside a window
    h[305:310] = True  # 10 hits -> red
    idx = pd.date_range("2024-01-01", periods=400, freq="D")
    out = rolling_traffic_light(pd.Series(h, index=idx), window=250)
    assert out.index.equals(idx)
    assert out.iloc[:249].isna().all()
    assert out.iloc[249] == "green"
    assert out.iloc[304] == "yellow"  # 5 hits
    assert out.iloc[308] == "yellow"  # 9 hits
    assert out.iloc[309] == "red"  # 10 hits
    assert out.iloc[399] == "red"  # all 10 hits still within [150, 399]
    assert list(out.cat.categories) == ["green", "yellow", "red"]
    # window leaves the hits behind
    longer = rolling_traffic_light(np.r_[h, np.zeros(200, dtype=bool)], window=250)
    assert longer.iloc[-1] == "green"


# ── Acerbi-Szekely ───────────────────────────────────────────────────────────
def test_z2_hand_computed() -> None:
    r = np.array([-0.10, 0.0, 0.02, -0.05])
    var = np.full(4, 0.04)
    es = np.full(4, 0.06)
    # breaches: -0.10 and -0.05;  Z2 = (-0.10/0.06 - 0.05/0.06) / (4 * 0.5) + 1 = -0.25
    assert acerbi_szekely_z2(r, var, es, alpha=0.5) == pytest.approx(-0.25)


def test_z2_zero_when_no_breaches_is_one() -> None:
    r = np.array([0.01, -0.01, 0.02])
    assert acerbi_szekely_z2(r, np.full(3, 0.05), np.full(3, 0.07), 0.05) == pytest.approx(1.0)


def test_z2_centered_under_correct_model() -> None:
    rng = np.random.default_rng(30)
    h = np.exp(rng.normal(-8.0, 0.5, 20_000))
    r = np.sqrt(h) * rng.standard_normal(h.size)
    z2 = acerbi_szekely_z2(r, value_at_risk(h, 0.05), expected_shortfall(h, 0.05), 0.05)
    assert abs(z2) < 0.08


def test_z2_negative_and_rejected_when_tails_are_fatter_than_forecast() -> None:
    rng = np.random.default_rng(31)
    n, alpha = 2500, 0.01
    h = np.full(n, 4e-4)
    r = np.sqrt(h) * _unit_t(rng, 3.5, n)  # truth: fat tails; forecast: normal
    res = acerbi_szekely_pvalue(r, h, alpha, dist="normal", n_sim=1000, seed=1)
    assert res.statistic < 0
    assert res.pvalue < 0.05


def test_z2_pvalue_not_small_under_correct_model_and_is_seeded() -> None:
    rng = np.random.default_rng(32)
    n = 2000
    h = np.exp(rng.normal(-8.0, 0.5, n))
    r = np.sqrt(h) * _unit_t(rng, 6.0, n)
    a = acerbi_szekely_pvalue(r, h, 0.05, dist="t", nu=6.0, n_sim=500, seed=3)
    b = acerbi_szekely_pvalue(r, h, 0.05, dist="t", nu=6.0, n_sim=500, seed=3)
    assert a.pvalue == b.pvalue
    assert a.pvalue > 0.01
    assert a.statistic == pytest.approx(
        acerbi_szekely_z2(
            r,
            value_at_risk(h, 0.05, "t", nu=6.0),
            expected_shortfall(h, 0.05, "t", nu=6.0),
            0.05,
        )
    )
    # FHS null distribution also works
    resid = _unit_t(rng, 6.0, 3000)
    f = acerbi_szekely_pvalue(r, h, 0.05, dist="fhs", std_resid=resid, n_sim=200, seed=3)
    assert 0.0 < f.pvalue <= 1.0


# ── Coverage simulations and summary ─────────────────────────────────────────
def _sim_returns(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    log_h = np.empty(n)
    log_h[0] = np.log(4e-4)
    for t in range(1, n):
        log_h[t] = np.log(4e-4) + 0.95 * (log_h[t - 1] - np.log(4e-4)) + 0.2 * rng.standard_normal()
    h = np.exp(log_h)
    return np.sqrt(h) * rng.standard_normal(n), h


@pytest.mark.parametrize("alpha", [0.05, 0.01])
def test_coverage_of_correct_normal_forecasts(alpha: float) -> None:
    r, h = _sim_returns(200_000, seed=40)
    hit = hits(r, value_at_risk(h, alpha))
    assert hit.mean() == pytest.approx(alpha, rel=0.05)
    rejections = 0
    reps = 200
    for i in range(reps):
        ri, hi = _sim_returns(1000, seed=1000 + i)
        rejections += kupiec_pof(hits(ri, value_at_risk(hi, alpha)), alpha).pvalue < 0.05
    assert rejections / reps < 0.12


def test_misspecified_variance_is_rejected() -> None:
    r, h = _sim_returns(3000, seed=41)
    hit = hits(r, value_at_risk(h * 0.4, 0.05))  # variance understated by 60%
    assert kupiec_pof(hit, 0.05).pvalue < 1e-6


def test_backtest_summary_structure_and_values() -> None:
    r, h = _sim_returns(1500, seed=42)
    summ = backtest_summary(r, h)
    assert list(summ.index) == [0.05, 0.01]
    assert list(summ.columns) == [
        "n_obs",
        "expected_breaches",
        "breaches",
        "breach_rate",
        "kupiec_p",
        "christoffersen_ind_p",
        "christoffersen_cc_p",
        "es_z2",
        "traffic_light",
    ]
    row = summ.loc[0.05]
    hit = hits(r, value_at_risk(h, 0.05))
    assert row["n_obs"] == 1500
    assert row["expected_breaches"] == pytest.approx(75.0)
    assert row["breaches"] == hit.sum()
    assert row["breach_rate"] == pytest.approx(hit.mean())
    assert row["kupiec_p"] == pytest.approx(kupiec_pof(hit, 0.05).pvalue)
    assert row["christoffersen_ind_p"] == pytest.approx(christoffersen_independence(hit).pvalue)
    assert row["christoffersen_cc_p"] == pytest.approx(christoffersen_cc(hit, 0.05).pvalue)
    assert pd.isna(row["traffic_light"])
    last250 = hits(r, value_at_risk(h, 0.01))[-250:]
    assert summ.loc[0.01, "traffic_light"] == basel_traffic_light(int(last250.sum()), 250, 0.99)


def test_backtest_summary_t_and_series_inputs() -> None:
    rng = np.random.default_rng(43)
    n = 1200
    idx = pd.date_range("2022-01-01", periods=n, freq="D")
    h = pd.Series(np.full(n, 4e-4), index=idx)
    r = pd.Series(np.sqrt(4e-4) * _unit_t(rng, 5.0, n), index=idx)
    resid = _unit_t(rng, 5.0, 5000)
    s_norm = backtest_summary(r, h)
    s_t = backtest_summary(r, h, dist="t", std_resid=resid)  # nu fitted by MLE
    s_fhs = backtest_summary(r.iloc[::-1], h, dist="fhs", std_resid=resid)  # index-aligned
    # normal forecasts are too thin in the 1% tail for t(5) data
    assert s_norm.loc[0.01, "breach_rate"] > s_t.loc[0.01, "breach_rate"] - 0.01
    assert s_t.loc[0.01, "kupiec_p"] > 0.01
    assert s_fhs.loc[0.05, "n_obs"] == n
    with pytest.raises(ValueError, match="lengths differ"):
        backtest_summary(r.to_numpy(), h.to_numpy()[:-1])
