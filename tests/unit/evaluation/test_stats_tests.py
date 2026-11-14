"""Tests for volforecast.evaluation.stats_tests (DM, Mincer-Zarnowitz, MCS)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats

from volforecast.evaluation import stats_tests
from volforecast.evaluation.stats_tests import (
    diebold_mariano,
    hln_factor,
    mincer_zarnowitz,
    mincer_zarnowitz_log,
    model_confidence_set,
)


def _ar1_losses(n: int, shift: float, seed: int, phi: float = 0.4) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n)
    d = np.empty(n)
    d[0] = e[0]
    for t in range(1, n):
        d[t] = phi * d[t - 1] + e[t]
    base = 5.0 + rng.standard_normal(n) ** 2
    return base + d + shift, base


def _manual_newey_west_t(d: np.ndarray, lags: int) -> float:
    n = len(d)
    dc = d - d.mean()
    lrv = sum(dc[t] ** 2 for t in range(n)) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        gamma = sum(dc[t] * dc[t - k] for t in range(k, n)) / n
        lrv += 2.0 * w * gamma
    return float(d.mean() / np.sqrt(lrv / n))


def test_dm_h1_matches_statsmodels_ols_t_stat() -> None:
    a, b = _ar1_losses(400, shift=-0.1, seed=3)
    d = a - b
    res = sm.OLS(d, np.ones_like(d)).fit(cov_type="HAC", cov_kwds={"maxlags": 0})
    out = diebold_mariano(a, b, h=1, hln=False)
    assert out.statistic == pytest.approx(float(res.tvalues[0]), rel=1e-10)
    assert out.detail is not None
    assert out.detail["mean_d"] == pytest.approx(d.mean())
    assert out.detail["n"] == 400
    assert out.detail["lags"] == 0
    assert out.detail["se"] == pytest.approx(float(res.bse[0]), rel=1e-10)
    assert out.pvalue == pytest.approx(stats.norm.cdf(out.statistic))


def test_dm_h5_matches_manual_newey_west_and_statsmodels() -> None:
    a, b = _ar1_losses(500, shift=-0.05, seed=4, phi=0.7)
    d = a - b
    out = diebold_mariano(a, b, h=5, hln=False)
    assert out.detail is not None
    assert out.detail["lags"] == 4
    assert out.statistic == pytest.approx(_manual_newey_west_t(d, 4), rel=1e-10)
    res = sm.OLS(d, np.ones_like(d)).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
    assert out.statistic == pytest.approx(float(res.tvalues[0]), rel=1e-10)


def test_dm_lags_override() -> None:
    a, b = _ar1_losses(300, shift=0.0, seed=5)
    out = diebold_mariano(a, b, h=1, hln=False, lags=7)
    assert out.detail is not None
    assert out.detail["lags"] == 7
    assert out.statistic == pytest.approx(_manual_newey_west_t(a - b, 7), rel=1e-10)


def test_hln_factor_values() -> None:
    assert hln_factor(100, 1) == pytest.approx(np.sqrt(99 / 100))
    assert hln_factor(100, 5) == pytest.approx(np.sqrt((100 + 1 - 10 + 20 / 100) / 100))
    assert hln_factor(50, 10) == pytest.approx(np.sqrt((50 + 1 - 20 + 90 / 50) / 50))
    with pytest.raises(ValueError, match="HLN"):
        hln_factor(10, 10)


def test_dm_hln_scaling_and_t_pvalue() -> None:
    a, b = _ar1_losses(120, shift=-0.2, seed=6)
    raw = diebold_mariano(a, b, h=5, hln=False)
    adj = diebold_mariano(a, b, h=5, hln=True)
    assert adj.statistic == pytest.approx(raw.statistic * hln_factor(120, 5))
    assert adj.pvalue == pytest.approx(stats.t.cdf(adj.statistic, df=119))
    assert adj.detail is not None
    assert adj.detail["mean_d"] / adj.detail["se"] == pytest.approx(adj.statistic)


def test_dm_alternative_direction() -> None:
    rng = np.random.default_rng(7)
    b = 5.0 + rng.standard_normal(300) ** 2
    a = b - 0.5 + 0.2 * rng.standard_normal(300)  # model a clearly better
    less = diebold_mariano(a, b, alternative="less")
    greater = diebold_mariano(a, b, alternative="greater")
    two = diebold_mariano(a, b, alternative="two-sided")
    assert less.statistic < 0
    assert less.pvalue < 1e-6
    assert greater.pvalue > 0.999
    assert less.pvalue + greater.pvalue == pytest.approx(1.0)
    assert two.pvalue == pytest.approx(2 * min(less.pvalue, greater.pvalue))
    # swapping the models flips the direction
    assert diebold_mariano(b, a, alternative="less").pvalue == pytest.approx(greater.pvalue)


def test_dm_equal_loss_pvalues_average_half() -> None:
    rng = np.random.default_rng(8)
    pvals = []
    for _ in range(400):
        a = rng.chisquare(3, 200)
        b = rng.chisquare(3, 200)
        pvals.append(diebold_mariano(a, b, h=1, alternative="less").pvalue)
    assert np.mean(pvals) == pytest.approx(0.5, abs=0.05)
    assert 0.02 < np.mean(np.asarray(pvals) < 0.05) < 0.10  # roughly correct size


def test_dm_identical_losses_degenerate() -> None:
    x = np.linspace(1.0, 2.0, 50)
    out = diebold_mariano(x, x)
    assert np.isnan(out.statistic)
    assert out.pvalue == 1.0
    assert out.detail is not None
    assert out.detail["variance_fallback"] is True


def test_dm_nonpositive_variance_falls_back_to_lag0(monkeypatch: pytest.MonkeyPatch) -> None:
    a, b = _ar1_losses(200, shift=-0.1, seed=9)
    real = stats_tests.hac_long_run_variance

    def fake(x: np.ndarray, lags: int) -> float:
        return -1.0 if lags > 0 else real(x, 0)

    monkeypatch.setattr(stats_tests, "hac_long_run_variance", fake)
    out = diebold_mariano(a, b, h=3, hln=False)
    assert out.detail is not None
    assert out.detail["variance_fallback"] is True
    assert out.statistic == pytest.approx(_manual_newey_west_t(a - b, 0))


def test_dm_input_validation() -> None:
    with pytest.raises(ValueError, match="equal length"):
        diebold_mariano(np.ones(10), np.ones(11))
    with pytest.raises(ValueError, match="finite"):
        diebold_mariano(np.array([1.0, np.nan, 2.0, 3.0]), np.ones(4))
    with pytest.raises(ValueError, match="alternative"):
        diebold_mariano(np.arange(10.0), np.ones(10), alternative="x")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="too few"):
        diebold_mariano(np.arange(4.0), np.ones(4), h=10)


# ── Mincer-Zarnowitz ─────────────────────────────────────────────────────────
def _sim_forecasts(
    n: int, seed: int, slope: float = 1.0, intercept_share: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    log_h = np.empty(n)
    log_h[0] = 0.0
    for t in range(1, n):
        log_h[t] = 0.9 * log_h[t - 1] + 0.3 * rng.standard_normal()
    h = np.exp(log_h) * 1e-3
    noise = np.exp(0.4 * rng.standard_normal(n) - 0.08)  # E[noise] = 1
    mean = intercept_share * h.mean() + slope * h
    return mean * noise, h


def test_mz_matches_statsmodels_fit() -> None:
    rv, h = _sim_forecasts(600, seed=10)
    out = mincer_zarnowitz(rv, h)
    lags = int(4 * (600 / 100) ** (2 / 9))
    res = sm.OLS(rv, sm.add_constant(h)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    assert out["hac_lags"] == lags
    assert out["a"] == pytest.approx(res.params[0])
    assert out["b"] == pytest.approx(res.params[1])
    assert out["se_a"] == pytest.approx(res.bse[0])
    assert out["se_b"] == pytest.approx(res.bse[1])
    assert out["r2"] == pytest.approx(res.rsquared)
    wald = res.wald_test((np.eye(2), np.array([0.0, 1.0])), use_f=False, scalar=True)
    assert out["wald_stat"] == pytest.approx(float(wald.statistic))
    assert out["wald_pvalue"] == pytest.approx(float(wald.pvalue))
    assert out["wald_df"] == 2


def test_mz_hac_lags_override_and_series_input() -> None:
    rv, h = _sim_forecasts(300, seed=11)
    out = mincer_zarnowitz(pd.Series(rv), pd.Series(h), hac_lags=3)
    res = sm.OLS(rv, sm.add_constant(h)).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    assert out["hac_lags"] == 3
    assert out["se_b"] == pytest.approx(res.bse[1])


def test_mz_unbiased_forecasts_do_not_reject_on_average() -> None:
    pvals = np.array(
        [mincer_zarnowitz(*_sim_forecasts(1000, seed=100 + i))["wald_pvalue"] for i in range(150)]
    )
    assert np.mean(pvals < 0.05) < 0.12
    assert np.mean(pvals) > 0.35


def test_mz_rejects_when_slope_is_half() -> None:
    out = [
        mincer_zarnowitz(*_sim_forecasts(1000, seed=300 + i, slope=0.5, intercept_share=0.5))
        for i in range(20)
    ]
    assert all(o["wald_pvalue"] < 0.01 for o in out)
    assert all(o["b"] == pytest.approx(0.5, abs=0.2) for o in out)


def test_mz_log_matches_statsmodels_and_validates() -> None:
    rv, h = _sim_forecasts(400, seed=12)
    out = mincer_zarnowitz_log(rv, h)
    res = sm.OLS(np.log(rv), sm.add_constant(np.log(h))).fit(
        cov_type="HAC", cov_kwds={"maxlags": out["hac_lags"]}
    )
    assert out["a"] == pytest.approx(res.params[0])
    assert out["b"] == pytest.approx(res.params[1])
    assert out["se_b"] == pytest.approx(res.bse[1])
    with pytest.raises(ValueError, match="strictly positive"):
        mincer_zarnowitz_log(np.r_[rv[:-1], 0.0], h)
    with pytest.raises(ValueError, match="strictly positive"):
        mincer_zarnowitz(rv, np.r_[h[:-1], -1.0])


# ── MCS ──────────────────────────────────────────────────────────────────────
def test_mcs_excludes_clearly_worse_model() -> None:
    rng = np.random.default_rng(13)
    t = 400
    common = rng.standard_normal(t)
    losses = pd.DataFrame(
        {
            "best": 1.00 + 0.2 * common + 0.1 * rng.standard_normal(t),
            "close": 1.01 + 0.2 * common + 0.1 * rng.standard_normal(t),
            "worse": 1.60 + 0.2 * common + 0.1 * rng.standard_normal(t),
        }
    )
    out = model_confidence_set(losses, size=0.10, reps=500, seed=0)
    assert "best" in out["included"]
    assert "close" in out["included"]
    assert out["excluded"] == ["worse"]
    pv = out["pvalues"]
    assert isinstance(pv, pd.Series)
    assert list(pv.index) == list(losses.columns)
    assert pv["worse"] < 0.10 <= pv["best"]
    assert out["block_size"] == round(t ** (1 / 3))
    # deterministic given the seed
    again = model_confidence_set(losses, size=0.10, reps=500, seed=0)
    pd.testing.assert_series_equal(again["pvalues"], pv)


def test_mcs_validation() -> None:
    with pytest.raises(ValueError, match="two models"):
        model_confidence_set(pd.DataFrame({"a": np.ones(50)}))
