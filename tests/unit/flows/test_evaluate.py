"""Evaluate flow: scoring stored forecasts, drift, and the retrain decision."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from tests.unit.flows.conftest import ASSET, seed_champion, write_synthetic_data_root
from volforecast.config import Settings
from volforecast.flows.common import Runtime
from volforecast.flows.evaluate import (
    RetrainDecision,
    decide_retrain,
    evaluate_flow,
    evaluate_one,
)
from volforecast.flows.evaluate import main as evaluate_main
from volforecast.flows.forecast import forecast_asset
from volforecast.forecast_store import (
    BENCHMARK_MODEL_NAME,
    BENCHMARK_VERSION,
    ROLE_BENCHMARK,
    ROLE_CHAMPION,
    ForecastRepository,
)
from volforecast.monitoring.drift import DriftReport
from volforecast.monitoring.performance import AssetEvaluation
from volforecast.pipeline.panel import build_panel
from volforecast.registry.config import MonitoringConfig, load_production_config


def _store_live_rows(
    rt: Runtime, panel: pd.DataFrame, n: int, champion_factor: float, har_factor: float = 1.0
) -> None:
    """Insert ``n`` days of live-looking forecasts: forecast = factor x realized next-day RV."""
    origins = panel.index[-n - 1 : -1]
    rv_next = panel["rv"].shift(-1).reindex(origins).to_numpy()
    rows = []
    for o, y in zip(origins, rv_next, strict=True):
        common = {
            "asset": ASSET,
            "horizon": 1,
            "origin_date": o.date(),
            "target_date": (o + pd.Timedelta(days=1)).date(),
        }
        rows.append(
            {**common, "model_name": "log_har", "model_version": "7", "role": ROLE_CHAMPION,
             "forecast_var": y * champion_factor, "var99": 0.2, "var95": 0.1}
        )  # fmt: skip
        rows.append(
            {**common, "model_name": BENCHMARK_MODEL_NAME, "model_version": BENCHMARK_VERSION,
             "role": ROLE_BENCHMARK, "forecast_var": y * har_factor}
        )  # fmt: skip
    rt.repo.upsert_forecasts(rows)


def test_evaluate_scores_live_forecasts_and_writes_metrics(
    runtime: Runtime, synthetic_panel_400: pd.DataFrame
) -> None:
    seed_champion(runtime, synthetic_panel_400, 330)
    forecast_asset(runtime, ASSET, replay_days=40)
    ev, report = evaluate_one(runtime, ASSET)
    info = ev.summary["h1"]
    assert info["n_scored_champion"] >= 38
    ratio = runtime.repo.get_metrics("qlike_ratio_vs_har", asset=ASSET, horizon=1)
    assert len(ratio) >= 29  # 10-day rolling window over ~39 common origins
    assert np.isfinite(ratio["value"]).all()
    hits = runtime.repo.get_metrics("var99_hit", asset=ASSET)
    assert set(hits["value"]) <= {0.0, 1.0}
    assert report.status == "ok"
    drift = runtime.repo.get_metrics("drift_share", asset=ASSET)
    assert len(drift) == 1
    files = list((runtime.settings.drift_reports_dir / ASSET).glob("*.json"))
    assert len(files) == 1
    assert json.loads(files[0].read_text())["asset"] == ASSET


def test_bad_champion_triggers_a_retrain_request(
    runtime: Runtime, synthetic_panel_400: pd.DataFrame
) -> None:
    _store_live_rows(runtime, synthetic_panel_400, 40, champion_factor=8.0)  # HAR is exact
    runtime.repo.upsert_realized(ASSET, synthetic_panel_400)
    ev, report = evaluate_one(runtime, ASSET)
    assert ev.retrain_horizons == [1]
    assert any("QLIKE ratio vs HAR" in t for t in ev.triggers)
    decision = decide_retrain([ev], {ASSET: report}, runtime.cfg.monitoring, [1])
    assert decision.retrain
    assert decision.requests == {ASSET: [1]}


def test_good_champion_does_not_trigger(
    runtime: Runtime, synthetic_panel_400: pd.DataFrame
) -> None:
    _store_live_rows(runtime, synthetic_panel_400, 40, champion_factor=1.0, har_factor=3.0)
    ev, report = evaluate_one(runtime, ASSET)
    assert ev.retrain_horizons == []
    # (the synthetic variance process is persistent, so a 20-day window may sit in another regime:
    # only the performance triggers are asserted here)
    perf_only = runtime.cfg.monitoring.model_copy(update={"drift_share_trigger": None})
    assert not decide_retrain([ev], {ASSET: report}, perf_only, [1]).retrain


def test_data_drift_triggers_all_horizons() -> None:
    cfg = MonitoringConfig()
    calm = AssetEvaluation(asset=ASSET)

    def report(share: float, status: str = "ok") -> DriftReport:
        return DriftReport(ASSET, "2025-01-01", None, 300, 30, 0.25, 0.5, 5, status, {}, share)

    assert not decide_retrain([calm], {ASSET: report(0.33)}, cfg, [1, 5]).retrain
    assert not decide_retrain(
        [calm], {ASSET: report(1.0, "insufficient_data")}, cfg, [1, 5]
    ).retrain
    hit = decide_retrain([calm], {ASSET: report(0.5)}, cfg, [1, 5])
    assert hit.requests == {ASSET: [1, 5]}
    assert "data drift" in hit.reasons[0]
    off = MonitoringConfig(drift_share_trigger=None)
    assert not decide_retrain([calm], {ASSET: report(1.0)}, off, [1, 5]).retrain


def test_performance_and_drift_triggers_are_merged() -> None:
    ev = AssetEvaluation(asset=ASSET, retrain_horizons=[5], triggers=["h=5: ratio"])
    rep = DriftReport(ASSET, "2025-01-01", None, 300, 30, 0.25, 0.5, 5, "ok", {}, 0.9)
    d = decide_retrain([ev], {ASSET: rep}, MonitoringConfig(), [1, 5])
    assert d.requests == {ASSET: [1, 5]}
    assert len(d.reasons) == 2
    assert isinstance(d, RetrainDecision)
    assert d.to_dict()["retrain"] is True


def test_evaluate_flow_records_the_decision_in_the_run_row(
    tmp_path: Path, tiny_config: Path, prefect_server: None
) -> None:
    root = tmp_path / "data"
    write_synthetic_data_root(root, 200)
    url = f"sqlite:///{tmp_path / 'store.db'}"
    kw = {
        "data_dir": str(root),
        "database_url": url,
        "tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}",
        "config_path": str(tiny_config),
    }
    # fabricate a clearly failing champion in the store (no MLflow model needed for evaluate)
    repo = ForecastRepository(url)
    panel, _ = build_panel(root, ASSET)
    repo.upsert_realized(ASSET, panel)
    rt = Runtime(settings=Settings(), cfg=load_production_config(tiny_config), repo=repo)
    _store_live_rows(rt, panel, 40, champion_factor=8.0)

    result = evaluate_flow(**kw)
    assert result["retrain_decision"]["retrain"] is True
    run = repo.pipeline_runs("evaluate")[0]
    assert run["status"] == "success"
    assert run["detail"]["retrain"] is True
    assert run["detail"]["retrain_requests"] == {ASSET: [1]}
    assert run["detail"]["retrain_reasons"]
    code = evaluate_main(
        [
            "--data-dir", kw["data_dir"], "--database-url", url,
            "--tracking-uri", kw["tracking_uri"], "--config", kw["config_path"],
        ]
    )  # fmt: skip
    assert code == 0
