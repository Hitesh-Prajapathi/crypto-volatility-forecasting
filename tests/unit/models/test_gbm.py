"""LightGBM / XGBoost challengers: causal features, determinism, bias correction, ext. hook."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from volforecast.contracts import COL_RV, VolModel
from volforecast.models.base import ModelInputError
from volforecast.models.factory import build_model
from volforecast.models.gbm import (
    EXOG_FEATURES,
    LightGBMVol,
    XGBoostVol,
    gbm_features,
    register_exog_features,
)

BOTH = ["lgbm", "xgb"]


def test_features_are_causal(synthetic_panel: pd.DataFrame) -> None:
    """Row i of the feature matrix equals the last row computed from data truncated at i."""
    p = synthetic_panel.iloc[:200]
    full = gbm_features(p)
    for i in (30, 77, 150, 199):
        trunc = gbm_features(p.iloc[: i + 1]).iloc[-1]
        np.testing.assert_array_equal(full.iloc[i].to_numpy(), trunc.to_numpy())
    # weekday feature is the weekday of t+1
    assert full["dow_next"].iloc[5] == (p.index[5].day_of_week + 1) % 7


def test_feature_set_matches_spec(synthetic_panel: pd.DataFrame) -> None:
    cols = set(gbm_features(synthetic_panel.iloc[:100]).columns)
    for base in ("log_rv", "log_bpv", "log_rs_pos", "log_rs_neg", "sqrt_rq"):
        assert {f"{base}_d", f"{base}_w", f"{base}_m"} <= cols
    assert {"log_jump", "abs_ret", "ret", "neg_ret", "dow_next"} <= cols
    assert {"d_log_rv_1", "d_log_rv_5", "d_log_rv_22"} <= cols


@pytest.mark.parametrize("name", BOTH)
def test_contract_and_bias_correction(name: str, synthetic_panel: pd.DataFrame) -> None:
    m = build_model(name)
    assert isinstance(m, VolModel)
    p = synthetic_panel.iloc[:700]
    m.fit(p, 5)
    v = m.predict_variance(p, 5)
    prm = m.get_params()
    assert math.isfinite(v)
    assert v > 0
    assert prm["sigma2"] > 0
    assert prm["sigma2_oof_rows"] > 100  # out-of-fold residuals
    assert abs(sum(prm["feature_importance_gain"].values()) - 1.0) < 1e-9
    json.dumps(prm)
    # exp(pred + sigma2/2): removing the correction scales the forecast by exp(-sigma2/2)
    x = gbm_features(p.iloc[-60:]).to_numpy(dtype=float)[-1:]
    raw = math.exp(float(m._predict_log(m._booster, x, m._n_rounds)[0]))  # type: ignore[attr-defined]
    assert v == pytest.approx(raw * math.exp(prm["sigma2"] / 2), rel=1e-12)


@pytest.mark.parametrize("name", BOTH)
def test_deterministic(name: str, synthetic_panel: pd.DataFrame) -> None:
    p = synthetic_panel.iloc[:600]
    a = build_model(name).fit(p, 1)
    b = build_model(name).fit(p, 1)
    assert a.predict_variance(p, 1) == b.predict_variance(p, 1)
    assert a.get_params() == b.get_params()


@pytest.mark.parametrize("cls", [LightGBMVol, XGBoostVol])
def test_no_lookahead_in_fit_or_predict(cls: type, synthetic_panel: pd.DataFrame) -> None:
    """Rows after the origin never reach the model: the engine hands over `.loc[:origin]`."""
    t = synthetic_panel.index[650]
    clean = synthetic_panel.copy()
    dirty = synthetic_panel.copy()
    fut = dirty.index > t
    dirty.loc[fut, COL_RV] = 1e3
    dirty.loc[fut, "ret"] = 5.0
    a = cls().fit(clean.loc[:t], 5)
    b = cls().fit(dirty.loc[:t], 5)
    assert a.predict_variance(clean.loc[:t], 5) == b.predict_variance(dirty.loc[:t], 5)


@pytest.mark.parametrize("cls", [LightGBMVol, XGBoostVol])
def test_training_targets_respect_embargo(cls: type, synthetic_panel: pd.DataFrame) -> None:
    """Rows whose h-day target is unobserved are not trained on (n_train shrinks with h)."""
    p = synthetic_panel.iloc[:500]
    n1 = cls().fit(p, 1).get_params()["n_train"]
    n5 = cls().fit(p, 5).get_params()["n_train"]
    assert n1 - n5 == 4


def test_early_stopping_uses_inner_validation_only(synthetic_panel: pd.DataFrame) -> None:
    p = synthetic_panel.iloc[:700]
    for cls in (LightGBMVol, XGBoostVol):
        m = cls(early_stopping_rounds=20, n_estimators=300).fit(p, 1)
        assert 20 <= m.get_params()["n_rounds_used"] <= 300
        assert m.predict_variance(p, 1) > 0


def test_too_little_data_raises(synthetic_panel: pd.DataFrame) -> None:
    with pytest.raises(ModelInputError, match="usable training rows"):
        LightGBMVol().fit(synthetic_panel.iloc[:100], 1)


def test_exogenous_feature_hook(synthetic_panel: pd.DataFrame) -> None:
    def lagged_rv_mean(panel: pd.DataFrame) -> pd.DataFrame:
        # stand-in for an external series (e.g. DVOL): already known at the end of day t
        return pd.DataFrame({"toy": panel[COL_RV].rolling(10, min_periods=1).mean()})

    register_exog_features("toy", lagged_rv_mean)
    try:
        p = synthetic_panel.iloc[:600]
        m = LightGBMVol(exog="toy").fit(p, 1)
        assert "exog_toy" in m.get_params()["features"]
        assert m.predict_variance(p, 1) > 0
        with pytest.raises(ValueError, match="unknown exogenous"):
            LightGBMVol(exog="nope")
    finally:
        EXOG_FEATURES.pop("toy", None)
