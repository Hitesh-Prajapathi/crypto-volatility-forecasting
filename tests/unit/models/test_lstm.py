"""LSTM challenger: contract, determinism, train-window-only scaler, causality, factory."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd
import pytest

# Import through volforecast.models.lstm (not torch directly): that module loads lightgbm/xgboost
# before torch, the only import order that works on macOS when both are used in one process.
pytest.importorskip("volforecast.models.lstm")

from tests.property.helpers import assert_truncation_invariant
from volforecast.contracts import COL_COMPLETE, VolModel
from volforecast.models.base import ModelInputError, ModelNotFittedError
from volforecast.models.factory import build_model, list_models
from volforecast.models.lstm import FEATURE_NAMES, LSTMVol, lstm_features

FAST: dict[str, Any] = {"max_epochs": 3, "hidden_size": 8}
N_TRAIN = 700


def _garbage_after(panel: pd.DataFrame, t: pd.Timestamp) -> pd.DataFrame:
    bad = panel.copy()
    future = bad.index > t
    for col in bad.columns:
        bad.loc[future, col] = False if col == COL_COMPLETE else 1e6
    return bad


def test_factory_registers_lstm() -> None:
    assert "lstm" in list_models()
    m = build_model("lstm", **FAST)
    assert isinstance(m, LSTMVol)
    assert isinstance(m, VolModel)
    assert m.name == "lstm"


def test_core_package_imports_without_torch() -> None:
    """The factory must not import torch at import time (torch is an optional extra)."""
    code = (
        "import sys, volforecast.models.factory as f; f.list_models(); "
        "assert 'torch' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("h", [1, 5])
def test_contract_positive_finite(h: int, synthetic_panel: pd.DataFrame) -> None:
    m = build_model("lstm", **FAST)
    p = synthetic_panel.iloc[:N_TRAIN]
    assert m.fit(p, h) is m
    for end in (N_TRAIN, N_TRAIN + 50):
        v = m.predict_variance(synthetic_panel.iloc[:end], h)
        assert isinstance(v, float)
        assert math.isfinite(v)
        assert v > 0.0
    prm = m.get_params()
    json.dumps(prm)
    assert prm["seed"] == 0
    assert prm["features"] == list(FEATURE_NAMES)
    assert prm["n_val"] > 0


def test_fit_state_machine(synthetic_panel: pd.DataFrame) -> None:
    m = build_model("lstm", **FAST)
    p = synthetic_panel.iloc[:N_TRAIN]
    with pytest.raises(ModelNotFittedError):
        m.predict_variance(p, 1)
    m.fit(p, 1)
    with pytest.raises(ModelNotFittedError, match="h=1"):
        m.predict_variance(p, 5)


def test_input_validation(synthetic_panel: pd.DataFrame) -> None:
    for bad in ({"lookback": 10}, {"lookback": 100}, {"num_layers": 3}, {"val_frac": 0.7}):
        with pytest.raises(ValueError, match="must"):
            build_model("lstm", **bad)
    with pytest.raises(ModelInputError, match="usable training rows"):
        build_model("lstm", **FAST).fit(synthetic_panel.iloc[:100], 1)
    m = build_model("lstm", **FAST).fit(synthetic_panel.iloc[:N_TRAIN], 1)
    with pytest.raises(ModelInputError, match="history"):
        m.predict_variance(synthetic_panel.iloc[:10], 1)


@pytest.mark.slow
def test_seed_determinism(synthetic_panel: pd.DataFrame) -> None:
    p = synthetic_panel.iloc[:N_TRAIN]
    a = build_model("lstm", seed=3, **FAST).fit(p, 1)
    b = build_model("lstm", seed=3, **FAST).fit(p, 1)
    c = build_model("lstm", seed=4, **FAST).fit(p, 1)
    va, vb, vc = (x.predict_variance(p, 1) for x in (a, b, c))
    assert va == vb
    assert a.get_params() == b.get_params()
    assert va != vc


def test_global_rng_state_untouched(synthetic_panel: pd.DataFrame) -> None:
    import random

    import torch

    p = synthetic_panel.iloc[:N_TRAIN]
    before = (random.getstate(), np.random.get_state()[1].copy(), torch.get_rng_state().clone())  # noqa: NPY002
    build_model("lstm", **FAST).fit(p, 1).predict_variance(p, 1)
    after = (random.getstate(), np.random.get_state()[1].copy(), torch.get_rng_state().clone())  # noqa: NPY002
    assert before[0] == after[0]
    assert np.array_equal(before[1], after[1])
    assert torch.equal(before[2], after[2])
    assert torch.are_deterministic_algorithms_enabled() is False


@pytest.mark.slow
def test_scaler_fit_on_training_window_only(synthetic_panel: pd.DataFrame) -> None:
    """Scaler stats equal the training-window stats and predict never re-estimates them."""
    train = synthetic_panel.iloc[:N_TRAIN]
    later = synthetic_panel.iloc[: N_TRAIN + 300]  # longer, different-distribution history
    m = build_model("lstm", **FAST).fit(train, 1)
    assert isinstance(m, LSTMVol)
    feats = lstm_features(train)
    np.testing.assert_allclose(m._mu, np.nanmean(feats, axis=0))
    np.testing.assert_allclose(m._sd, np.nanstd(feats, axis=0))
    # the full-sample statistics differ, so a leaky (full-sample) scaler would be caught
    assert not np.allclose(np.nanmean(lstm_features(later), axis=0), m._mu)

    mu0, sd0 = m._mu.copy(), m._sd.copy()  # type: ignore[union-attr]
    v0 = m.predict_variance(later, 1)
    np.testing.assert_array_equal(m._mu, mu0)
    np.testing.assert_array_equal(m._sd, sd0)
    # predict really consumes the stored scaler: shifting it changes the forecast
    m._mu = mu0 + 1.0
    assert m.predict_variance(later, 1) != v0
    m._mu = mu0
    assert m.predict_variance(later, 1) == v0
    # a refit on the longer window re-estimates the scaler (training-window stats move)
    m2 = build_model("lstm", **FAST).fit(later, 1)
    assert isinstance(m2, LSTMVol)
    assert not np.allclose(m2._mu, mu0)  # type: ignore[arg-type]


@pytest.mark.slow
def test_validation_block_is_chronologically_last(
    synthetic_panel: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = 5
    p = synthetic_panel.iloc[:N_TRAIN]
    seen: dict[str, Any] = {}
    orig = LSTMVol._train

    def spy(self: LSTMVol, xtr: Any, ytr: Any, xva: Any, yva: Any) -> Any:
        seen.update(xtr=xtr, xva=xva)
        return orig(self, xtr, ytr, xva, yva)

    monkeypatch.setattr(LSTMVol, "_train", spy)
    m = LSTMVol(**FAST).fit(p, h)
    scaled = m._scale(lstm_features(p))

    def last_rows(x: np.ndarray) -> np.ndarray:
        return np.array([np.flatnonzero((scaled == r).all(axis=1))[0] for r in x[:, -1, :]])

    tr_rows, va_rows = last_rows(seen["xtr"]), last_rows(seen["xva"])
    assert np.all(np.diff(va_rows) > 0)
    assert tr_rows.max() + h <= va_rows.min()  # chronological split with an h-day embargo
    frac = va_rows.size / (va_rows.size + tr_rows.size)
    assert 0.12 < frac < 0.16


@pytest.mark.slow
def test_bias_correction(synthetic_panel: pd.DataFrame) -> None:
    p = synthetic_panel.iloc[:N_TRAIN]
    m = LSTMVol(**FAST).fit(p, 1)
    sigma2 = m.get_params()["sigma2"]
    assert sigma2 > 0
    v = m.predict_variance(p, 1)
    m._sigma2 = 0.0
    assert v == pytest.approx(m.predict_variance(p, 1) * math.exp(sigma2 / 2), rel=1e-12)


def test_features_truncation_invariant(synthetic_panel: pd.DataFrame) -> None:
    def build(p: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(lstm_features(p), index=p.index, columns=list(FEATURE_NAMES))

    cuts = [synthetic_panel.index[k] for k in (100, 400, 900)]
    assert_truncation_invariant(build, synthetic_panel.iloc[:1000], cuts)


@pytest.mark.slow
def test_no_lookahead_in_fit_or_predict(synthetic_panel: pd.DataFrame) -> None:
    """Fit and forecast at origin t are identical whether or not rows after t exist."""
    h = 5
    t0, t = synthetic_panel.index[N_TRAIN - 1], synthetic_panel.index[N_TRAIN + 49]
    clean = build_model("lstm", **FAST).fit(synthetic_panel.loc[:t0], h)
    garbage = _garbage_after(synthetic_panel, t0)
    dirty = build_model("lstm", **FAST).fit(garbage.loc[:t0], h)
    assert clean.get_params() == dirty.get_params()
    garbage_t = _garbage_after(synthetic_panel, t)
    a = clean.predict_variance(synthetic_panel.loc[:t], h)
    b = clean.predict_variance(garbage_t.loc[:t], h)
    c = clean.predict_variance(synthetic_panel.loc[:t].copy(), h)
    assert a == b == c
