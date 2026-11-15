from __future__ import annotations

from typing import Any, Self

import numpy as np
import pandas as pd
import pytest

from volforecast.contracts import FORECAST_COLUMNS
from volforecast.evaluation.walkforward import (
    ModelSpec,
    WalkForwardConfig,
    realized_targets,
    run_many,
    run_walkforward,
)


class SpyModel:
    """Records the last date it was shown; forecasts h * mean(last 5 RV)."""

    name = "spy"
    seen_max: list[pd.Timestamp] = []  # noqa: RUF012 - shared across instances on purpose
    fit_calls: int = 0

    def fit(self, panel: pd.DataFrame, h: int) -> Self:
        SpyModel.fit_calls += 1
        SpyModel.seen_max.append(panel.index[-1])
        self.h = h
        return self

    def predict_variance(self, panel: pd.DataFrame, h: int) -> float:
        SpyModel.seen_max.append(panel.index[-1])
        return float(h * panel["rv"].iloc[-5:].mean())

    def get_params(self) -> dict[str, Any]:
        return {}


@pytest.fixture(autouse=True)
def _reset_spy() -> None:
    SpyModel.seen_max = []
    SpyModel.fit_calls = 0


def test_never_sees_future_and_columns(synthetic_panel: pd.DataFrame) -> None:
    cfg = WalkForwardConfig(first_origin=synthetic_panel.index[400], min_train=100)
    out, stats = run_walkforward(synthetic_panel, ModelSpec("spy", SpyModel), cfg, 1, "SYN")
    assert list(out.columns) == list(FORECAST_COLUMNS)
    # Each call's max date equals the origin of that step (never later).
    assert stats.n_origins == len(out)
    origins = list(out["origin_date"])
    # predict calls are every other element after each fit; check all seen <= last origin
    assert max(SpyModel.seen_max) <= origins[-1]
    assert (out["target_date"] - out["origin_date"]).dt.days.eq(1).all()


def test_refit_cadence(synthetic_panel: pd.DataFrame) -> None:
    cfg = WalkForwardConfig(first_origin=synthetic_panel.index[300], refit_every=7, min_train=100)
    out, stats = run_walkforward(synthetic_panel, ModelSpec("spy", SpyModel), cfg, 1, "SYN")
    assert stats.n_fits == int(np.ceil(len(out) / 7))


def test_realized_targets_and_incomplete(synthetic_panel: pd.DataFrame) -> None:
    panel = synthetic_panel.copy()
    panel.iloc[50, panel.columns.get_loc("complete")] = False
    tg = realized_targets(panel, 5)
    rv = panel["rv"].to_numpy()
    assert np.isclose(tg.iloc[10], rv[11:16].sum())
    assert tg.iloc[45:50].isna().all()  # windows covering day 50
    assert np.isfinite(tg.iloc[50])  # origin 50 covers days 51..55
    assert tg.iloc[-5:].isna().all()


def test_forecast_realized_alignment(synthetic_panel: pd.DataFrame) -> None:
    cfg = WalkForwardConfig(first_origin=synthetic_panel.index[300], min_train=100)
    out, _ = run_walkforward(synthetic_panel, ModelSpec("spy", SpyModel), cfg, 5, "SYN")
    row = out.iloc[0]
    pos = synthetic_panel.index.get_loc(row["origin_date"])
    assert np.isclose(row["realized_var"], synthetic_panel["rv"].iloc[pos + 1 : pos + 6].sum())
    assert np.isclose(row["forecast_var"], 5 * synthetic_panel["rv"].iloc[pos - 4 : pos + 1].mean())


def test_rolling_window(synthetic_panel: pd.DataFrame) -> None:
    cfg = WalkForwardConfig(
        first_origin=synthetic_panel.index[400], window="rolling", rolling_size=200, min_train=100
    )
    out, _ = run_walkforward(synthetic_panel, ModelSpec("spy", SpyModel), cfg, 1, "SYN")
    assert len(out) == len(synthetic_panel) - 400


def test_run_many_shape(synthetic_panel: pd.DataFrame) -> None:
    cfg = WalkForwardConfig(first_origin=synthetic_panel.index[1400], min_train=100)
    out, stats = run_many(
        synthetic_panel, [ModelSpec("a", SpyModel), ModelSpec("b", SpyModel)], cfg, "SYN", n_jobs=1
    )
    assert set(stats) == {("a", 1), ("a", 5), ("b", 1), ("b", 5)}
    assert len(out) == 4 * 100


# --------------------------------------------------------------------- reuse_instance
class _Tracked(SpyModel):
    """SpyModel that records every instance the builder hands out."""

    built: list[_Tracked] = []  # noqa: RUF012 - shared on purpose

    def __init__(self) -> None:
        self.n_fit = 0
        _Tracked.built.append(self)

    def fit(self, panel: pd.DataFrame, h: int) -> Self:
        self.n_fit += 1
        return super().fit(panel, h)


def test_reuse_instance_false_builds_fresh_model_per_refit(synthetic_panel: pd.DataFrame) -> None:
    _Tracked.built = []
    cfg = WalkForwardConfig(first_origin=synthetic_panel.index[300], refit_every=7, min_train=100)
    spec = ModelSpec("t", _Tracked)
    assert spec.reuse_instance is False  # default
    out, stats = run_walkforward(synthetic_panel, spec, cfg, 1, "SYN")
    assert stats.n_fits == len(_Tracked.built) == int(np.ceil(len(out) / 7))
    assert all(m.n_fit == 1 for m in _Tracked.built)  # every refit used a brand-new instance
    assert len({id(m) for m in _Tracked.built}) == len(_Tracked.built)


def test_reuse_instance_true_refits_the_same_object(synthetic_panel: pd.DataFrame) -> None:
    _Tracked.built = []
    cfg = WalkForwardConfig(first_origin=synthetic_panel.index[300], refit_every=7, min_train=100)
    spec = ModelSpec("t", _Tracked, reuse_instance=True)
    out, stats = run_walkforward(synthetic_panel, spec, cfg, 1, "SYN")
    assert len(_Tracked.built) == 1  # built once ...
    assert _Tracked.built[0].n_fit == stats.n_fits == int(np.ceil(len(out) / 7))  # ... refit often


def test_reuse_instance_does_not_change_forecasts_of_stateless_model(
    synthetic_panel: pd.DataFrame,
) -> None:
    cfg = WalkForwardConfig(first_origin=synthetic_panel.index[300], refit_every=7, min_train=100)
    a, _ = run_walkforward(synthetic_panel, ModelSpec("spy", SpyModel), cfg, 5, "SYN")
    b, _ = run_walkforward(
        synthetic_panel, ModelSpec("spy", SpyModel, reuse_instance=True), cfg, 5, "SYN"
    )
    pd.testing.assert_frame_equal(a, b)


class _Member:
    """Deterministic stub member for ensembles: forecast = k * h * RV_t."""

    def __init__(self, name: str, k: float) -> None:
        self.name, self.k = name, k

    def fit(self, panel: pd.DataFrame, h: int) -> Self:
        return self

    def predict_variance(self, panel: pd.DataFrame, h: int) -> float:
        return float(self.k * h * panel["rv"].iloc[-1])

    def get_params(self) -> dict[str, Any]:
        return {}


def test_ensemble_keeps_forecast_history_across_engine_refits(
    synthetic_panel: pd.DataFrame,
) -> None:
    from volforecast.models.ensemble import InverseLossEnsemble

    h = 1
    built: list[InverseLossEnsemble] = []

    def build() -> InverseLossEnsemble:
        e = InverseLossEnsemble([_Member("a", 1.0), _Member("b", 3.0)], window=60, min_obs=20)
        built.append(e)
        return e

    cfg = WalkForwardConfig(first_origin=synthetic_panel.index[300], refit_every=7, min_train=100)

    reuse_out, stats = run_walkforward(
        synthetic_panel, ModelSpec("ens", build, reuse_instance=True), cfg, h, "SYN"
    )
    assert len(built) == 1
    ens = built[0]
    assert stats.n_fits > 10  # many engine refits ...
    n_hist = len(ens._hist)
    assert n_hist >= 60  # ... yet member forecasts of earlier origins survived (trimmed to window)
    w = ens.weight_history()
    assert len(w) == stats.n_origins
    assert not np.allclose(w.iloc[-1].to_numpy(), 0.5)  # weights moved away from equal

    # reference: a single engine fit (huge refit_every, the old workaround) gives identical output
    built.clear()
    once = WalkForwardConfig(
        first_origin=synthetic_panel.index[300], refit_every=10**6, min_train=100
    )
    ref_out, _ = run_walkforward(synthetic_panel, ModelSpec("ens", build), once, h, "SYN")
    np.testing.assert_allclose(reuse_out["forecast_var"], ref_out["forecast_var"], rtol=1e-12)

    # and without reuse the history is lost at each refit (weights restart from equal weights)
    built.clear()
    fresh_out, _ = run_walkforward(synthetic_panel, ModelSpec("ens", build), cfg, h, "SYN")
    assert len(built) == stats.n_fits
    assert not np.allclose(fresh_out["forecast_var"], reuse_out["forecast_var"])
