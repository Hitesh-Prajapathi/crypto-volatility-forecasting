"""Contract tests every factory model must satisfy."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from volforecast.contracts import COL_COMPLETE, COL_RV, VolModel
from volforecast.models.base import ModelInputError, ModelNotFittedError
from volforecast.models.factory import MODEL_FACTORY, build_model, list_models

ALL_MODELS = list_models()
EXPECTED = {
    "naive",
    "ewma",
    "ewma_rv",
    "garch",
    "gjr_garch",
    "har",
    "log_har",
    "har_cj",
    "shar",
    "harq",
    "lgbm",
    "xgb",
    "ens_eq",
    "ens_inv",
}


def _garbage_after(panel: pd.DataFrame, t: pd.Timestamp) -> pd.DataFrame:
    bad = panel.copy()
    future = bad.index > t
    for col in bad.columns:
        if col == COL_COMPLETE:
            bad.loc[future, col] = False
        else:
            bad.loc[future, col] = 1e6
    return bad


def test_factory_names() -> None:
    assert set(MODEL_FACTORY) == EXPECTED
    assert set(list_models()) == EXPECTED


def test_unknown_model() -> None:
    with pytest.raises(ValueError, match="unknown model"):
        build_model("nope")


def test_gjr_name_and_override() -> None:
    assert build_model("gjr_garch").get_params()["o"] == 1
    assert build_model("garch", dist="normal").get_params()["dist"] == "normal"


@pytest.mark.parametrize("h", [1, 5])
@pytest.mark.parametrize("name", ALL_MODELS)
def test_fit_predict_contract(name: str, h: int, synthetic_panel: pd.DataFrame) -> None:
    m = build_model(name)
    assert isinstance(m, VolModel)
    assert m.name == name
    train = synthetic_panel.iloc[:800]
    assert m.fit(train, h) is m
    for end in (800, 900):
        v = m.predict_variance(synthetic_panel.iloc[:end], h)
        assert isinstance(v, float)
        assert math.isfinite(v)
        assert v > 0.0
    json.dumps(m.get_params())


@pytest.mark.parametrize("name", ALL_MODELS)
def test_errors_before_fit_and_wrong_h(name: str, synthetic_panel: pd.DataFrame) -> None:
    m = build_model(name)
    train = synthetic_panel.iloc[:500]
    with pytest.raises(ModelNotFittedError):
        m.predict_variance(train, 1)
    m.fit(train, 1)
    with pytest.raises(ModelNotFittedError, match="h=1"):
        m.predict_variance(train, 5)


@pytest.mark.parametrize("name", ALL_MODELS)
def test_no_lookahead(name: str, synthetic_panel: pd.DataFrame) -> None:
    """Fit and forecast depend only on rows <= the cut-off, whatever sits after it."""
    h = 5
    t0, t = synthetic_panel.index[600], synthetic_panel.index[650]
    clean = build_model(name).fit(synthetic_panel.loc[:t0], h)
    garbage = _garbage_after(synthetic_panel, t0)
    dirty = build_model(name).fit(garbage.loc[:t0], h)
    assert clean.get_params() == dirty.get_params()

    garbage_t = _garbage_after(synthetic_panel, t)
    a = clean.predict_variance(synthetic_panel.loc[:t], h)
    b = clean.predict_variance(garbage_t.loc[:t], h)
    assert a == b


@pytest.mark.parametrize("name", ALL_MODELS)
def test_fit_once_predict_later(name: str, synthetic_panel: pd.DataFrame) -> None:
    h = 1
    t0, t1 = synthetic_panel.index[700], synthetic_panel.index[760]
    m = build_model(name).fit(synthetic_panel.loc[:t0], h)
    params_before = json.dumps(m.get_params(), sort_keys=True)
    f0 = m.predict_variance(synthetic_panel.loc[:t0], h)
    f1 = m.predict_variance(synthetic_panel.loc[:t1], h)
    assert f0 != f1
    assert json.dumps(m.get_params(), sort_keys=True) == params_before


@pytest.mark.parametrize("name", ["har", "garch", "naive"])
def test_input_validation(name: str, synthetic_panel: pd.DataFrame) -> None:
    m = build_model(name)
    with pytest.raises(ModelInputError, match="missing"):
        m.fit(synthetic_panel.drop(columns=[COL_RV]), 1)
    gap = synthetic_panel.drop(synthetic_panel.index[100])
    with pytest.raises(ModelInputError, match="one-day"):
        m.fit(gap, 1)
    with pytest.raises(ModelInputError, match="DatetimeIndex"):
        m.fit(synthetic_panel.reset_index(drop=True), 1)
    with pytest.raises(ModelInputError, match="horizon"):
        m.fit(synthetic_panel, 0)


@pytest.mark.parametrize("name", ["naive", "ewma_rv", "har", "log_har"])
def test_nonpositive_forecast_is_repaired(name: str, synthetic_panel: pd.DataFrame) -> None:
    m = build_model(name).fit(synthetic_panel.iloc[:600], 1)
    bad = synthetic_panel.iloc[:650].copy()
    bad.iloc[-60:, bad.columns.get_loc(COL_RV)] = 0.0  # RV collapses to zero
    v = m.predict_variance(bad, 1)
    assert math.isfinite(v)
    assert v > 0.0


def test_naive_uses_last_valid_rv(synthetic_panel: pd.DataFrame) -> None:
    p = synthetic_panel.iloc[:300].copy()
    m = build_model("naive").fit(p, 5)
    expected = 5 * p[COL_RV].iloc[-1]
    assert np.isclose(m.predict_variance(p, 5), expected)
    p.iloc[-1, p.columns.get_loc(COL_RV)] = np.nan
    assert np.isclose(m.predict_variance(p, 5), 5 * p[COL_RV].iloc[-2])
