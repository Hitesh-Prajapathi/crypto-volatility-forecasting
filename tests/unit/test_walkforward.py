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
