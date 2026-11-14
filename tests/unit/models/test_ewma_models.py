"""EWMA models: recursion, NaN handling and lambda estimation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from volforecast.contracts import COL_RET, COL_RV
from volforecast.models.ewma import EWMARealized, EWMAReturns, ewma_path


def _manual_ewma(x: np.ndarray, lam: float, init: float) -> float:
    s = init
    for v in x:
        s = lam * s + (1 - lam) * v
    return s


def test_ewma_returns_matches_manual_loop(synthetic_panel: pd.DataFrame) -> None:
    p = synthetic_panel.iloc[:400].copy()
    p.iloc[0, p.columns.get_loc(COL_RET)] = np.nan
    r = p[COL_RET].dropna().to_numpy()
    init = float(np.var(r[:30], ddof=1))
    expected = 5 * _manual_ewma(r**2, 0.94, init)
    m = EWMAReturns().fit(p, 5)
    assert np.isclose(m.predict_variance(p, 5), expected, rtol=1e-10)


def test_ewma_realized_fixed_lambda_matches_manual_loop(synthetic_panel: pd.DataFrame) -> None:
    p = synthetic_panel.iloc[:400]
    rv = p[COL_RV].to_numpy()
    expected = 1 * _manual_ewma(rv, 0.9, float(rv[:30].mean()))
    m = EWMARealized(lam=0.9).fit(p, 1)
    assert np.isclose(m.predict_variance(p, 1), expected, rtol=1e-10)
    assert m.get_params()["lam_fitted"] == 0.9


def test_ewma_path_skips_nan() -> None:
    x = np.array([np.nan, 1.0, np.nan, 2.0, 3.0])
    out = ewma_path(x, 0.5, 4.0)
    ref = ewma_path(np.array([1.0, 2.0, 3.0]), 0.5, 4.0)
    assert out[0] == 4.0  # before first valid observation
    assert np.isclose(out[-1], ref[-1])
    assert np.isclose(out[2], out[1])  # carried forward over the NaN


def test_lambda_estimation_in_bounds(synthetic_panel: pd.DataFrame) -> None:
    p = synthetic_panel.iloc[:1000]
    est = EWMARealized(estimate_lambda=True).fit(p, 1)
    params = est.get_params()
    assert 0.5 <= params["lam_fitted"] <= 0.995
    assert params["lam_fitted"] != 0.94
    assert params["train_qlike"] is not None
    fixed = EWMARealized(lam=0.94, estimate_lambda=False).fit(p, 1)
    assert fixed.get_params()["lam_fitted"] == 0.94
    assert fixed.get_params()["train_qlike"] is None
    assert est.predict_variance(p, 1) > 0


def test_lambda_estimation_persistent_vs_noisy() -> None:
    """A persistent latent variance with little noise favours a larger lambda."""
    from tests.unit.models.helpers import panel_from_rv

    rng = np.random.default_rng(3)
    n = 2000
    latent = np.exp(np.log(1e-3) + 0.4 * np.cumsum(rng.standard_normal(n)) * 0.02)
    smooth = panel_from_rv(latent * np.exp(0.05 * rng.standard_normal(n)))
    noisy = panel_from_rv(latent * np.exp(0.8 * rng.standard_normal(n)))
    lam_smooth = EWMARealized(estimate_lambda=True).fit(smooth, 1).get_params()["lam_fitted"]
    lam_noisy = EWMARealized(estimate_lambda=True).fit(noisy, 1).get_params()["lam_fitted"]
    assert 0.5 <= lam_smooth <= 0.995
    assert 0.5 <= lam_noisy <= 0.995
    assert lam_noisy > lam_smooth  # noisy proxy needs more smoothing
