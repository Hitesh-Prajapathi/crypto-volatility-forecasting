"""HAR-family models: coefficient recovery, target alignment, filters, timing."""

from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd
import pytest

from tests.unit.models.helpers import panel_from_rv, simulate_har_rv
from volforecast.contracts import COL_COMPLETE, COL_RS_NEG, COL_RS_POS, COL_RV, target_column
from volforecast.features.har import har_aggregates, make_targets
from volforecast.logging import get_logger
from volforecast.models.har import HAR, HARQ, LogHAR

log = get_logger(__name__)


def test_har_recovers_known_coefficients() -> None:
    rv = simulate_har_rv(30_000, b0=1e-4, bd=0.4, bw=0.3, bm=0.2, seed=11)
    m = HAR().fit(panel_from_rv(rv), 1)
    coef = m.get_params()["coef"]
    assert coef["const"] == pytest.approx(1e-4, abs=1e-4)
    assert coef["rv_d"] == pytest.approx(0.4, abs=0.05)
    assert coef["rv_w"] == pytest.approx(0.3, abs=0.07)
    assert coef["rv_m"] == pytest.approx(0.2, abs=0.07)


def test_har_h5_uses_five_day_sum_target() -> None:
    rv = simulate_har_rv(6000, seed=5)
    p = panel_from_rv(rv)
    h = 5
    m = HAR().fit(p, h)
    agg = har_aggregates(p)
    tgt = make_targets(p, (h,))[target_column(h)]
    X = pd.concat([agg["rv_d"], agg["rv_w"], agg["rv_m"]], axis=1)
    ok = X.notna().all(axis=1) & tgt.notna()
    design = np.column_stack([np.ones(ok.sum()), X[ok].to_numpy()])
    beta = np.linalg.lstsq(design, tgt[ok].to_numpy(), rcond=None)[0]
    coef = m.get_params()["coef"]
    np.testing.assert_allclose(
        [coef["const"], coef["rv_d"], coef["rv_w"], coef["rv_m"]], beta, rtol=1e-8, atol=1e-12
    )
    # five-day-sum projection: loadings sum to ~ 5x one-day persistence-weighted level
    assert sum(beta[1:]) > 1.0
    assert m.get_params()["n_train"] == int(ok.sum())


def test_incomplete_target_windows_are_dropped(synthetic_panel: pd.DataFrame) -> None:
    p = synthetic_panel.iloc[:500].copy()
    k = 300
    p.iloc[k, p.columns.get_loc(COL_COMPLETE)] = False
    n = len(p)
    for h in (1, 5):
        kept = HAR().fit(p, h).get_params()["n_train"]
        keep_all = HAR(drop_incomplete_targets=False).fit(p, h).get_params()["n_train"]
        assert keep_all == n - 21 - h
        assert kept == n - 21 - h - h  # the h windows that contain day k are removed


def test_nan_rv_rows_never_enter_training(synthetic_panel: pd.DataFrame) -> None:
    p = synthetic_panel.iloc[:500].copy()
    p.iloc[250, p.columns.get_loc(COL_RV)] = np.nan
    m = HAR().fit(p, 1)
    coef = m.get_params()["coef"]
    assert all(math.isfinite(v) for v in coef.values())
    # unusable: first 21 rows (window), row 249 (NaN target), rows 250..271 (NaN regressors),
    # last row (no target)
    assert m.get_params()["n_train"] == len(p) - 21 - 1 - 22 - 1
    assert m.predict_variance(p, 1) > 0


def test_insanity_filter_counts_and_replaces(synthetic_panel: pd.DataFrame) -> None:
    train = synthetic_panel.iloc[:600]
    m = HAR().fit(train, 1)
    mean = m.get_params()["train_target_mean"]
    wild = synthetic_panel.iloc[:650].copy()
    wild.iloc[-1, wild.columns.get_loc(COL_RV)] = 1e3  # forecast far above training max
    v = m.predict_variance(wild, 1)
    assert v == pytest.approx(mean)
    assert m.get_params()["n_insanity_filtered"] == 1
    m.predict_variance(synthetic_panel.iloc[:650], 1)
    assert m.get_params()["n_insanity_filtered"] == 1  # normal forecast untouched
    m.fit(train, 1)
    assert m.get_params()["n_insanity_filtered"] == 0  # reset on refit


def test_log_har_bias_correction(synthetic_panel: pd.DataFrame) -> None:
    p = synthetic_panel.iloc[:900]
    m = LogHAR().fit(p, 1)
    params = m.get_params()
    c = params["coef"]
    agg = har_aggregates(p, log=True)
    last = agg.iloc[-1]
    lp = (
        c["const"]
        + c["log_rv_d"] * last["log_rv_d"]
        + c["log_rv_w"] * last["log_rv_w"]
        + c["log_rv_m"] * last["log_rv_m"]
    )
    f = m.predict_variance(p, 1)
    assert params["sigma2"] > 0
    assert f > math.exp(lp)
    assert f == pytest.approx(math.exp(lp + params["sigma2"] / 2), rel=1e-10)


def test_harq_demeans_with_training_mean() -> None:
    rv = simulate_har_rv(3000, seed=2)
    p = panel_from_rv(rv)
    m = HARQ().fit(p, 1)
    params = m.get_params()
    assert params["rq_sqrt_train_mean"] > 0
    assert "rv_d_x_rq_sqrt_dm" in params["coef"]
    # forecast at the fit origin is reproducible from the stored coefficients
    c = params["coef"]
    a = har_aggregates(p)
    last = a.iloc[-1]
    rq = math.sqrt(p["rq"].iloc[-1])
    expected = (
        c["const"]
        + c["rv_d"] * last["rv_d"]
        + c["rv_d_x_rq_sqrt_dm"] * last["rv_d"] * (rq - params["rq_sqrt_train_mean"])
        + c["rv_w"] * last["rv_w"]
        + c["rv_m"] * last["rv_m"]
    )
    lo, hi = params["train_target_min"], params["train_target_max"]
    assert lo <= expected <= hi
    assert m.predict_variance(p, 1) == pytest.approx(expected, rel=1e-10)


def test_wls_estimator_runs_and_differs() -> None:
    rv = simulate_har_rv(5000, seed=9)
    p = panel_from_rv(rv)
    ols = HAR().fit(p, 1)
    wls = HAR(estimator="wls").fit(p, 1)
    assert wls.get_params()["coef"] != ols.get_params()["coef"]
    assert wls.get_params()["coef"]["rv_d"] == pytest.approx(0.4, abs=0.1)
    assert wls.predict_variance(p, 1) > 0
    with pytest.raises(ValueError, match="estimator"):
        HAR(estimator="gls")


def test_har_timing_2500_rows() -> None:
    p = panel_from_rv(simulate_har_rv(2500, seed=1))
    times = []
    for _ in range(7):
        t0 = time.perf_counter()
        m = HAR().fit(p, 5)
        m.predict_variance(p, 5)
        times.append(time.perf_counter() - t0)
    best, median = min(times), float(np.median(times))
    log.info("har fit+predict timing (2500 rows)", best_ms=best * 1e3, median_ms=median * 1e3)
    print(f"HAR fit+predict 2500 rows: best {best * 1e3:.2f} ms, median {median * 1e3:.2f} ms")
    assert median < 0.2


def _collapse_panel(n: int = 900, seed: int = 4) -> pd.DataFrame:
    """Panel where next-day RV loads positively on rs_neg and strongly NEGATIVELY on rs_pos."""
    rng = np.random.default_rng(seed)
    base = 4e-4
    rs_pos = np.full(n, base / 2)
    rs_neg = np.full(n, base / 2)
    rv = np.full(n, base)
    for t in range(n - 1):
        rs_pos[t] = base / 2 * rng.lognormal(0.0, 0.5)
        rs_neg[t] = base / 2 * rng.lognormal(0.0, 0.5)
        rv[t] = rs_pos[t] + rs_neg[t]
    # tomorrow's RV: strong positive loading on rs_neg, negative on rs_pos (so the fitted
    # level SHAR can be driven to ~0 or below by an extreme rs_pos day)
    nxt = np.empty(n)
    nxt[0] = base
    nxt[1:] = np.clip(
        2e-4 + 1.6 * rs_neg[:-1] - 0.9 * rs_pos[:-1] + 3e-5 * rng.standard_normal(n - 1),
        0.02 * base,
        None,
    )
    p = panel_from_rv(rv, seed=seed)
    p[COL_RS_POS] = rs_pos
    p[COL_RS_NEG] = rs_neg
    p[COL_RV] = nxt
    # keep rv consistent with the semivariances it is built from only through the target;
    # the regressors seen by SHAR are rs_pos / rs_neg (daily) and trailing mean rv
    return p


def test_level_shar_collapse_is_caught_by_trailing_floor() -> None:
    from volforecast.models.har import SHAR

    p = _collapse_panel()
    train = p.iloc[:700]
    # origin: huge positive semivariance, tiny negative semivariance, normal trailing RV
    origin = p.iloc[:760].copy()
    legacy = SHAR(floor_frac=0.0).fit(train, 1)
    coef = legacy.get_params()["coef"]
    assert coef["rs_pos_d"] < 0 < coef["rs_neg_d"]
    recent = float(origin[COL_RV].iloc[-22:].mean())
    origin.iloc[-1, origin.columns.get_loc(COL_RS_NEG)] = 1e-6
    origin.iloc[-1, origin.columns.get_loc(COL_RS_POS)] = 0.0
    lp0 = legacy._linear_predictor(origin)
    assert lp0 is not None
    # choose the positive semivariance that puts the linear predictor at 5% of the run-rate
    origin.iloc[-1, origin.columns.get_loc(COL_RS_POS)] = (lp0 - 0.05 * recent) / -coef["rs_pos_d"]

    # original filter (training min only): the collapsed forecast sails through
    lp = legacy._linear_predictor(origin)
    assert lp is not None
    y_lo = legacy.get_params()["train_target_min"]
    assert y_lo <= lp < 0.2 * recent  # collapse: legit w.r.t. the old bounds, far below run-rate
    assert legacy.predict_variance(origin, 1) == pytest.approx(lp)
    assert legacy.get_params()["n_insanity_filtered"] == 0

    # with the floor, the forecast is replaced by the fallback and the replacement is counted
    robust = SHAR().fit(train, 1)
    assert robust.floor_frac == 0.2
    f = robust.predict_variance(origin, 1)
    assert f == pytest.approx(robust.get_params()["train_target_mean"])
    assert f >= 0.2 * recent
    prm = robust.get_params()
    assert prm["n_insanity_filtered"] == 1
    assert prm["n_insanity_below_floor"] == 1
    assert prm["n_insanity_above_max"] == 0

    # horizon scaling of the floor, and normal origins are untouched
    normal = p.iloc[:760]
    assert robust.predict_variance(normal, 1) > 0
    assert robust.get_params()["n_insanity_filtered"] == 1


def test_floor_scales_with_horizon_and_validates() -> None:
    from volforecast.models.har import SHAR

    p = _collapse_panel()
    m5 = SHAR(floor_frac=0.2).fit(p.iloc[:700], 5)
    recent = float(p[COL_RV].iloc[:760].iloc[-22:].mean())
    assert m5._floor(p.iloc[:760], 5) >= 0.2 * 5 * recent - 1e-15
    with pytest.raises(ValueError, match="floor_frac"):
        HAR(floor_frac=-0.1)
    assert HAR(floor_frac=0.3).get_params()["floor_frac"] == 0.3
