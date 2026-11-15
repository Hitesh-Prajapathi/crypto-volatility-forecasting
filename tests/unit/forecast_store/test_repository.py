"""Forecast store: upserts are idempotent, types are sanitized, read helpers behave."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import inspect

from tests.conftest import make_synthetic_panel
from volforecast.forecast_store import (
    BENCHMARK_MODEL_NAME,
    BENCHMARK_VERSION,
    ForecastRepository,
    clean_json,
)


@pytest.fixture
def repo(tmp_path: Path) -> ForecastRepository:
    return ForecastRepository(f"sqlite:///{tmp_path / 'sub' / 'store.db'}")


def _row(**kw: object) -> dict[str, object]:
    base: dict[str, object] = {
        "asset": "BTCUSDT",
        "horizon": 1,
        "model_name": "log_har",
        "model_version": "1",
        "role": "champion",
        "origin_date": date(2024, 1, 10),
        "target_date": date(2024, 1, 11),
        "forecast_var": 4.0e-4,
        "var95": 0.03,
        "var99": 0.05,
        "es975": 0.045,
        "dist": "fhs",
        "data_hash": "abc",
    }
    base.update(kw)
    return base


def test_creates_sqlite_parent_dir_and_all_tables(repo: ForecastRepository) -> None:
    names = set(inspect(repo.engine).get_table_names())
    assert {
        "forecasts",
        "realized",
        "pipeline_runs",
        "promotion_decisions",
        "monitoring_metrics",
    } <= names


def test_upsert_is_idempotent_and_updates(repo: ForecastRepository) -> None:
    assert repo.upsert_forecasts([_row()]) == 1
    repo.upsert_forecasts([_row(forecast_var=5.0e-4, var99=0.06)])
    df = repo.forecasts_df("BTCUSDT", 1)
    assert len(df) == 1
    assert df.loc[0, "forecast_var"] == pytest.approx(5.0e-4)
    assert df.loc[0, "var99"] == pytest.approx(0.06)
    # a different version on the same origin is a different row
    repo.upsert_forecasts([_row(model_version="2", role="challenger")])
    assert len(repo.forecasts_df("BTCUSDT", 1)) == 2


def test_numpy_and_pandas_values_are_sanitized(repo: ForecastRepository) -> None:
    rec = _row(
        forecast_var=np.float64(3.0e-4),
        var95=float("nan"),
        horizon=np.int64(1),
        origin_date=pd.Timestamp("2024-02-01"),
        target_date=pd.Timestamp("2024-02-02"),
    )
    repo.upsert_forecasts([rec])
    out = repo.latest_forecast("BTCUSDT", 1)
    assert out is not None
    assert out["var95"] is None
    assert out["origin_date"] == date(2024, 2, 1)
    assert isinstance(out["forecast_var"], float)


def test_clean_json_handles_nan_numpy_and_dates() -> None:
    out = clean_json({"a": np.float64("nan"), "b": [np.int64(3), pd.Timestamp("2024-01-02")]})
    assert out == {"a": None, "b": [3, "2024-01-02"]}


def test_latest_forecast_ignores_shadow_and_benchmark(repo: ForecastRepository) -> None:
    repo.upsert_forecasts(
        [
            _row(origin_date=date(2024, 1, 10)),
            _row(
                model_version="2",
                role="challenger",
                origin_date=date(2024, 1, 11),
                target_date=date(2024, 1, 12),
            ),
            _row(
                model_name=BENCHMARK_MODEL_NAME,
                model_version=BENCHMARK_VERSION,
                role="benchmark",
                origin_date=date(2024, 1, 11),
                target_date=date(2024, 1, 12),
            ),
        ]
    )
    latest = repo.latest_forecast("BTCUSDT", 1)
    assert latest is not None
    assert latest["origin_date"] == date(2024, 1, 10)
    assert latest["model_version"] == "1"
    assert repo.latest_forecast("ETHUSDT", 1) is None
    summary = repo.models_summary()
    assert summary[0]["challenger_version"] == "2"
    assert summary[0]["model_version"] == "1"


def test_realized_roundtrip_and_forecast_history_join(repo: ForecastRepository) -> None:
    panel = make_synthetic_panel(30, start="2024-01-01")
    repo.upsert_realized("BTCUSDT", panel)
    repo.upsert_realized("BTCUSDT", panel)  # idempotent
    got = repo.get_realized("BTCUSDT")
    assert len(got) == 30
    np.testing.assert_allclose(got["rv"].to_numpy(), panel["rv"].to_numpy())
    assert repo.last_realized_date("BTCUSDT") == date(2024, 1, 30)

    repo.upsert_forecasts(
        [
            _row(
                horizon=5,
                origin_date=date(2024, 1, 10),
                target_date=date(2024, 1, 15),
                model_version="7",
            ),
            _row(
                horizon=5,
                origin_date=date(2024, 1, 29),
                target_date=date(2024, 2, 3),
                model_version="7",
            ),
        ]
    )
    hist = repo.forecast_history("BTCUSDT", 5)
    assert len(hist) == 2
    expected = panel["rv"].loc["2024-01-11":"2024-01-15"].sum()
    assert hist.loc[0, "realized_var"] == pytest.approx(expected)
    assert np.isnan(hist.loc[1, "realized_var"])  # target days not yet observed
    assert hist.loc[0, "realized_ret"] == pytest.approx(panel["ret"].loc["2024-01-11"])


def test_pipeline_runs_and_last_success(repo: ForecastRepository) -> None:
    rid = repo.start_run("forecast", date(2024, 1, 10), {"x": np.float64(1.5)})
    repo.finish_run(rid, "success", {"rows": 4})
    rid2 = repo.start_run("evaluate", date(2024, 1, 10))
    repo.finish_run(rid2, "failed", {"error": "boom"})
    ok = repo.last_success()
    assert set(ok) == {"forecast"}
    runs = repo.pipeline_runs("forecast")
    assert runs[0]["detail"] == {"x": 1.5, "rows": 4}


def test_metrics_upsert_drops_nan_and_replaces(repo: ForecastRepository) -> None:
    key = {"asset": "BTCUSDT", "horizon": 1, "model": "champion", "window": 30}
    n = repo.upsert_metrics(
        [
            {**key, "metric": "rolling_qlike", "as_of_date": "2024-01-10", "value": 0.2},
            {**key, "metric": "rolling_qlike", "as_of_date": "2024-01-11", "value": float("nan")},
        ]
    )
    assert n == 1
    repo.upsert_metrics(
        [{**key, "metric": "rolling_qlike", "as_of_date": "2024-01-10", "value": 0.3}]
    )
    df = repo.get_metrics("rolling_qlike", asset="BTCUSDT")
    assert len(df) == 1
    assert df.loc[0, "value"] == pytest.approx(0.3)
