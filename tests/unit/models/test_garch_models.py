"""GARCH / GJR-GARCH: parameter recovery, fixed-parameter filtering, fallbacks."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import pytest
from arch import arch_model
from arch.univariate import StudentsT

from tests.unit.models.helpers import panel_from_rv
from volforecast.contracts import COL_RET
from volforecast.models.garch import GarchModel


def _simulate(params: list[float], o: int, nobs: int, seed: int) -> pd.DataFrame:
    """Panel whose returns come from a known (GJR-)GARCH(1,1)-t process (arch's `simulate`)."""
    am = arch_model(None, mean="Constant", vol="GARCH", p=1, o=o, q=1, dist="t", rescale=False)
    am.distribution = StudentsT(seed=seed)
    sim = am.simulate(params, nobs=nobs, burn=500)
    ret = sim["data"].to_numpy() / 100.0
    return panel_from_rv(ret**2 + 1e-8, ret=ret)


def _arch_forecast(panel: pd.DataFrame, o: int, dist: str, h: int) -> float:
    y = panel[COL_RET].dropna() * 100.0
    res = arch_model(y, mean="Constant", vol="GARCH", p=1, o=o, q=1, dist=dist, rescale=False).fit(
        disp="off", show_warning=False
    )
    return float(res.forecast(horizon=h, reindex=False).variance.to_numpy()[-1].sum() / 1e4)


def test_garch_recovers_alpha_beta() -> None:
    p = _simulate([0.0, 0.05, 0.10, 0.85, 8.0], o=0, nobs=6000, seed=1)
    m = GarchModel(o=0, dist="t").fit(p, 1)
    par: dict[str, Any] = m.get_params()["params"]
    assert par["alpha[1]"] == pytest.approx(0.10, abs=0.04)
    assert par["beta[1]"] == pytest.approx(0.85, abs=0.06)
    assert par["alpha[1]"] + par["beta[1]"] == pytest.approx(0.95, abs=0.03)
    assert m.get_params()["last_fit_converged"] is True


def test_gjr_asymmetry_sign() -> None:
    p = _simulate([0.0, 0.05, 0.04, 0.15, 0.85, 8.0], o=1, nobs=6000, seed=2)
    m = GarchModel(o=1, dist="t").fit(p, 1)
    par = m.get_params()["params"]
    assert m.name == "gjr_garch"
    assert par["gamma[1]"] > 0.05
    assert par["gamma[1]"] == pytest.approx(0.15, abs=0.08)
    assert par["beta[1]"] == pytest.approx(0.85, abs=0.06)


@pytest.mark.parametrize(("o", "dist", "h"), [(0, "t", 1), (0, "t", 5), (1, "normal", 5)])
def test_fixed_params_forecast_equals_arch_forecast(o: int, dist: str, h: int) -> None:
    p = _simulate([0.0, 0.05, 0.05, 0.10, 0.85, 8.0], o=1, nobs=1500, seed=3)
    m = GarchModel(o=o, dist=dist).fit(p, h)
    ours = m.predict_variance(p, h)
    assert ours == pytest.approx(_arch_forecast(p, o, dist, h), rel=1e-6)


def test_forecast_filters_new_data_with_fixed_params() -> None:
    p = _simulate([0.0, 0.05, 0.10, 0.85, 8.0], o=0, nobs=1500, seed=4)
    m = GarchModel().fit(p.iloc[:1000], 1)
    before = m.get_params()["params"]
    v_fit = m.predict_variance(p.iloc[:1000], 1)
    v_later = m.predict_variance(p.iloc[:1200], 1)
    assert v_fit != v_later
    assert m.get_params()["params"] == before


def test_fit_failure_falls_back_to_ewma_then_previous_params() -> None:
    p = _simulate([0.0, 0.05, 0.10, 0.85, 8.0], o=0, nobs=800, seed=5)
    short = p.iloc[:40]
    m = GarchModel().fit(short, 1)  # too few observations -> no params -> EWMA
    assert m.get_params()["params"] is None
    assert m.get_params()["n_fit_failures"] == 1
    v = m.predict_variance(short, 1)
    assert math.isfinite(v)
    assert v > 0
    assert m.get_params()["n_fallback_predictions"] == 1

    m.fit(p, 1)
    good = m.get_params()["params"]
    assert good is not None
    m.fit(short, 1)  # failing refit keeps previous parameters
    assert m.get_params()["params"] == good
    assert m.get_params()["n_fit_failures"] == 2
    assert m.predict_variance(p, 1) > 0


def test_garch_skips_nan_returns() -> None:
    p = _simulate([0.0, 0.05, 0.10, 0.85, 8.0], o=0, nobs=700, seed=6)
    p.iloc[100, p.columns.get_loc(COL_RET)] = np.nan
    m = GarchModel().fit(p, 5)
    assert m.predict_variance(p, 5) > 0
    assert m.get_params()["params"] is not None
