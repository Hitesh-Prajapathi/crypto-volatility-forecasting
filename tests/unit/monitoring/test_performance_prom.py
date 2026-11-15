"""Performance triggers and Prometheus exposition."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from prometheus_client import generate_latest

from tests.conftest import make_synthetic_panel
from volforecast.forecast_store import ForecastRepository
from volforecast.monitoring import prom
from volforecast.monitoring.performance import (
    breach_trigger,
    evaluate_asset,
    ratio_trigger,
    rolling_frame,
)
from volforecast.registry.config import MonitoringConfig

ASSET = "BTCUSDT"


def test_ratio_trigger_needs_ten_consecutive_days_above_one() -> None:
    s = pd.Series([0.9] * 5 + [1.1] * 9)
    assert not ratio_trigger(s, 1.0, 10)
    s2 = pd.Series([0.9] * 5 + [1.1] * 10)
    assert ratio_trigger(s2, 1.0, 10)
    assert not ratio_trigger(pd.Series([1.5] * 9), 1.0, 10)  # not enough history


def test_rolling_frame_waits_for_a_full_window() -> None:
    idx = pd.date_range("2024-01-01", periods=40)
    a, b = pd.Series(1.0, index=idx), pd.Series(0.5, index=idx)
    roll = rolling_frame(a, b, 30)
    assert len(roll) == 11
    assert roll["ratio"].iloc[-1] == pytest.approx(2.0)


def test_breach_trigger_requires_a_full_250_window() -> None:
    few = pd.Series([True, True, False, False, True])  # 3/5 would be "red" with n=5
    assert not breach_trigger(few, 250, 0.99).fired
    hits = pd.Series([False] * 245 + [True] * 5)
    st = breach_trigger(hits, 250, 0.99)
    assert (st.breaches, st.zone, st.fired) == (5, "yellow", True)
    assert not breach_trigger(pd.Series([False] * 246 + [True] * 4), 250, 0.99).fired


def _seed_store(repo: ForecastRepository, n: int, scale: float) -> pd.DataFrame:
    panel = make_synthetic_panel(n + 60, seed=4, start="2024-01-01")
    repo.upsert_realized(ASSET, panel)
    origins = panel.index[30 : 30 + n]
    rows: list[dict[str, Any]] = []
    for o in origins:
        base = float(panel["rv"].rolling(5).mean().loc[o])
        for role, ver, name, mult in (
            ("champion", "1", "log_har", scale),
            ("benchmark", "har_benchmark", "har_benchmark", 1.0),
        ):
            rows.append(
                {
                    "asset": ASSET,
                    "horizon": 1,
                    "model_name": name,
                    "model_version": ver,
                    "role": role,
                    "origin_date": o,
                    "target_date": o + pd.Timedelta(days=1),
                    "forecast_var": base * mult,
                    "var99": float(np.sqrt(base * mult) * 2.0) if role == "champion" else None,
                }
            )
    repo.upsert_forecasts(rows)
    return panel


def test_evaluate_asset_fires_on_degraded_champion_but_not_on_matching_one(tmp_path: Path) -> None:
    cfg = MonitoringConfig()
    good = ForecastRepository(f"sqlite:///{tmp_path / 'good.db'}")
    _seed_store(good, 45, scale=1.0)
    assert evaluate_asset(good, ASSET, [1], cfg).retrain_horizons == []

    bad = ForecastRepository(f"sqlite:///{tmp_path / 'bad.db'}")
    _seed_store(bad, 45, scale=0.1)
    res = evaluate_asset(bad, ASSET, [1], cfg)
    assert res.retrain_horizons == [1]
    assert "QLIKE ratio" in res.triggers[0]
    ratio = bad.get_metrics("qlike_ratio_vs_har", asset=ASSET)
    assert (ratio["value"] > 1.0).all()
    assert len(bad.get_metrics("var99_hit", asset=ASSET)) > 0


def test_evaluate_asset_tolerates_no_scored_rows(tmp_path: Path) -> None:
    repo = ForecastRepository(f"sqlite:///{tmp_path / 'empty.db'}")
    res = evaluate_asset(repo, ASSET, [1, 5], MonitoringConfig())
    assert res.retrain_horizons == []


def test_prometheus_metric_names_and_labels(tmp_path: Path) -> None:
    repo = ForecastRepository(f"sqlite:///{tmp_path / 'p.db'}")
    _seed_store(repo, 45, scale=0.1)
    evaluate_asset(repo, ASSET, [1], MonitoringConfig())
    repo.upsert_metrics(
        [
            {"asset": ASSET, "metric": "drift_share", "as_of_date": date(2024, 3, 1), "value": 0.5},
            {
                "asset": ASSET,
                "metric": "data_last_bar_ts",
                "as_of_date": date(2024, 3, 1),
                "value": prom.last_bar_timestamp("2024-03-01"),
            },
        ]
    )
    rid = repo.start_run("forecast", date(2024, 3, 1))
    repo.finish_run(rid, "success")
    metrics = prom.PipelineMetrics()
    prom.export_from_store(repo, metrics)
    text = generate_latest(metrics.registry).decode()
    for needle in (
        'volforecast_model_info{asset="BTCUSDT",horizon="1",version="1"} 1.0',
        'volforecast_forecast_var{asset="BTCUSDT",horizon="1"}',
        'volforecast_realized_var{asset="BTCUSDT"}',
        'volforecast_rolling_qlike{asset="BTCUSDT",horizon="1",model="champion",window="30"}',
        'volforecast_qlike_ratio_vs_har{asset="BTCUSDT",horizon="1"}',
        'volforecast_var_breaches_total{asset="BTCUSDT",level="99"}',
        'volforecast_data_last_bar_timestamp_seconds{asset="BTCUSDT"}',
        'volforecast_pipeline_last_success_timestamp_seconds{flow="forecast"}',
        'volforecast_drift_share{asset="BTCUSDT"} 0.5',
    ):
        assert needle in text, needle
    # re-export does not duplicate or leave stale label sets behind
    prom.export_from_store(repo, metrics)
    assert generate_latest(metrics.registry).decode().count("volforecast_model_info{") == 1


def test_push_job_metrics_is_a_noop_without_gateway(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(prom.PUSHGATEWAY_ENV, raising=False)
    assert prom.push_job_metrics("forecast") is False

    calls: list[dict[str, Any]] = []
    monkeypatch.setenv(prom.PUSHGATEWAY_ENV, "http://pushgateway:9091")
    monkeypatch.setattr(
        prom, "push_to_gateway", lambda url, job, registry: calls.append({"url": url, "job": job})
    )
    repo = ForecastRepository(f"sqlite:///{tmp_path / 'q.db'}")
    assert prom.push_job_metrics("forecast", repo) is True
    assert calls == [{"url": "http://pushgateway:9091", "job": "volforecast_forecast"}]
