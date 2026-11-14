"""Scale-calibrated VaR distributions (normal_cal, t_cal): scaling and information set."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from tests.unit.test_risk import ASSET, MIN_RESID, _forecasts, _var
from volforecast.contracts import COL_RET, COL_RV, FORECAST_COLUMNS
from volforecast.evaluation.risk import build_var_forecasts


@pytest.fixture
def fc(synthetic_panel: pd.DataFrame) -> pd.DataFrame:
    return _forecasts(synthetic_panel, {"a": 1.0, "b": 1.3})


def _cal_var(fc: pd.DataFrame, panel: pd.DataFrame, dist: str, alpha: float = 0.05) -> pd.DataFrame:
    return _var(fc, panel, dists=(dist,), alphas=(alpha,))


def test_normal_cal_scales_by_expanding_second_moment_of_past_residuals(
    fc: pd.DataFrame, synthetic_panel: pd.DataFrame
) -> None:
    out = _cal_var(fc, synthetic_panel, "normal_cal")
    g = out[out["model"] == "a"].sort_values("origin_date").reset_index(drop=True)
    z = (g["ret"] / np.sqrt(g["forecast_var"])).to_numpy()
    assert (g.loc[: MIN_RESID - 1, "dist_used"] == "normal").all()
    assert (g.loc[MIN_RESID:, "dist_used"] == "normal_cal").all()
    for i in (MIN_RESID, MIN_RESID + 9, len(g) - 1):
        c = np.mean(z[:i] ** 2)  # origins strictly before row i
        want = -norm.ppf(0.05) * np.sqrt(c * g.loc[i, "forecast_var"])
        assert g.loc[i, "var"] == pytest.approx(want)
    warm = g.loc[: MIN_RESID - 1]
    np.testing.assert_allclose(warm["var"], -norm.ppf(0.05) * np.sqrt(warm["forecast_var"]))


def test_calibration_restores_unit_variance_scale() -> None:
    """Forecasts 4x too large in variance: plain normal is too wide, the _cal dists are not."""
    rng = np.random.default_rng(0)
    n = 1500
    idx = pd.date_range("2020-01-01", periods=n, freq="D", name="date")
    sig2 = 1e-3
    panel = pd.DataFrame(
        {COL_RET: np.sqrt(sig2) * rng.standard_normal(n), COL_RV: np.full(n, sig2)}, index=idx
    )
    f = pd.DataFrame(
        {
            "asset": ASSET,
            "model": "big",
            "horizon": 1,
            "origin_date": idx[:-1],
            "target_date": idx[:-1] + pd.Timedelta(days=1),
            "forecast_var": 4 * sig2,
            "realized_var": sig2,
            "model_version": "dev",
            "data_hash": "h",
        }
    )[list(FORECAST_COLUMNS)]
    out = build_var_forecasts(
        f, {ASSET: panel}, (0.05,), ("normal", "normal_cal", "t_cal"), min_resid=100
    )
    post = out[out["n_resid"] >= 100]
    rate = post.groupby("dist")["hit"].mean()
    assert rate["normal"] < 0.02  # far too conservative
    assert abs(rate["normal_cal"] - 0.05) < 0.02
    assert abs(rate["t_cal"] - 0.05) < 0.025


@pytest.mark.parametrize("dist", ["normal_cal", "t_cal"])
def test_calibrated_dists_ignore_future_returns(
    fc: pd.DataFrame, synthetic_panel: pd.DataFrame, dist: str
) -> None:
    base = _cal_var(fc, synthetic_panel, dist)
    cutoff = pd.Timestamp("2021-06-01")
    mutated_panel = synthetic_panel.copy()
    future = mutated_panel.index > cutoff
    mutated_panel.loc[future, COL_RET] = mutated_panel.loc[future, COL_RET] * 7.0 + 0.05
    mutated = _cal_var(fc, mutated_panel, dist)
    keys = ["model", "alpha", "origin_date"]
    m = base.merge(mutated, on=keys, suffixes=("", "_m"))
    past = m[m["origin_date"] <= cutoff]
    later = m[m["origin_date"] > cutoff + pd.Timedelta(days=2)]
    assert len(past) > 100
    np.testing.assert_array_equal(past["var"], past["var_m"])
    np.testing.assert_array_equal(past["es"], past["es_m"])
    assert not np.allclose(later["var"], later["var_m"])
