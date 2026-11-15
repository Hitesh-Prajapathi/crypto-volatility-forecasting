"""End-to-end daily cycle on synthetic data: seed champion -> forecast -> ingest -> forecast ->
evaluate -> retrain, with a temp sqlite forecast store and a temp sqlite MLflow registry.

No network, no servers (Prefect runs against its in-process test harness).
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date
from pathlib import Path

import mlflow
import pandas as pd
import pytest
from prometheus_client import generate_latest

from tests.conftest import make_synthetic_klines
from tests.unit.flows.conftest import ASSET, TINY_CONFIG, seed_champion
from tests.unit.flows.test_ingest import FakeSource
from volforecast.data.store import write_klines
from volforecast.flows import ingest as ingest_mod
from volforecast.flows.common import Runtime
from volforecast.flows.evaluate import evaluate_flow
from volforecast.flows.forecast import forecast_flow
from volforecast.flows.ingest import ingest_flow
from volforecast.flows.retrain import retrain_flow
from volforecast.forecast_store import (
    ROLE_BENCHMARK,
    ROLE_CHAMPION,
    ForecastRepository,
)
from volforecast.monitoring.prom import PipelineMetrics, export_from_store
from volforecast.registry.client import CHALLENGER, CHAMPION, alias_version, registered_model_name

N_STORED, N_TOTAL = 330, 360  # 2024-01-01 + 330 days -> last stored day 2024-11-26
REPLAY = 40


@pytest.fixture(scope="module")
def prefect_server() -> Iterator[None]:
    from prefect.testing.utilities import prefect_test_harness

    with prefect_test_harness():
        yield


def test_full_daily_cycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prefect_server: None
) -> None:
    import yaml

    cfg_path = tmp_path / "production.yaml"
    cfg_path.write_text(yaml.safe_dump(TINY_CONFIG))
    root = tmp_path / "data"
    full = make_synthetic_klines(N_TOTAL, seed=11, start="2024-01-01")
    cut = pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(days=N_STORED)
    write_klines(full[full["open_time"] < cut], root, ASSET)
    db_url = f"sqlite:///{tmp_path / 'store.db'}"
    kw = {
        "data_dir": str(root),
        "database_url": db_url,
        "tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}",
        "config_path": str(cfg_path),
    }
    rt = Runtime.create(**kw)  # type: ignore[arg-type]
    repo = ForecastRepository(db_url)
    reg_name = registered_model_name(ASSET, 1)

    # -- seed a champion trained 60 days before the end of the stored data ---------------------
    panel, _ = ingest_mod.load_panel(rt, ASSET)
    v0 = seed_champion(rt, panel, len(panel) - 60)
    assert str(alias_version(rt.client, reg_name, CHAMPION).version) == v0  # type: ignore[union-attr]

    # -- forecast: champion by alias, benchmark alongside, idempotent upsert -------------------
    res = forecast_flow(replay_days=REPLAY, **kw)  # type: ignore[arg-type]
    assert res["results"][0]["rows"] == 2 * REPLAY
    rows = repo.forecasts_df(ASSET, 1)
    assert len(rows) == 2 * REPLAY
    assert set(rows[rows["role"] == ROLE_CHAMPION]["model_version"]) == {v0}
    assert (rows["role"] == ROLE_BENCHMARK).sum() == REPLAY
    last_origin = rows["origin_date"].max().date()
    forecast_flow(replay_days=REPLAY, **kw)  # type: ignore[arg-type]
    assert len(repo.forecasts_df(ASSET, 1)) == 2 * REPLAY  # no duplicates after a rerun
    latest = repo.latest_forecast(ASSET, 1)
    assert latest is not None
    assert latest["origin_date"] == last_origin
    assert latest["model_version"] == v0
    assert latest["var99"] > latest["var95"] > 0

    # -- ingest 30 new days, then forecast the new origin --------------------------------------
    monkeypatch.setattr(ingest_mod, "_SOURCE_FACTORY", lambda name, d: FakeSource(full))
    ing = ingest_flow(end="2024-12-25", **kw)  # type: ignore[arg-type]
    assert ing["results"][0]["status"] == "ok"
    assert repo.last_realized_date(ASSET) == date(2024, 12, 25)
    forecast_flow(**kw)  # type: ignore[arg-type]
    assert repo.latest_forecast(ASSET, 1)["origin_date"] == date(2024, 12, 25)  # type: ignore[index]
    assert len(repo.forecasts_df(ASSET, 1)) == 2 * REPLAY + 2

    # -- evaluate: live rows scored, drift computed, decision recorded -------------------------
    ev = evaluate_flow(**kw)  # type: ignore[arg-type]
    info = ev["summaries"][ASSET]["h1"]
    assert info["n_scored_champion"] >= REPLAY - 1
    assert not repo.get_metrics("qlike_ratio_vs_har", asset=ASSET, horizon=1).empty
    assert not repo.get_metrics("var99_hit", asset=ASSET).empty
    assert len(repo.get_metrics("drift_share", asset=ASSET)) == 1
    run = repo.pipeline_runs("evaluate")[0]
    assert run["status"] == "success"
    assert isinstance(run["detail"]["retrain"], bool)
    n_metrics = len(repo.get_metrics("qlike_ratio_vs_har"))
    evaluate_flow(**kw)  # type: ignore[arg-type]
    assert len(repo.get_metrics("qlike_ratio_vs_har")) == n_metrics  # idempotent upserts

    # -- retrain: challenger -> gate -> aliases consistent with the decision -------------------
    rt_out = retrain_flow(**kw)  # type: ignore[arg-type]
    outcome = rt_out["outcomes"][0]
    assert outcome["status"] in {"promoted", "rejected"}
    decisions = repo.promotion_decisions(ASSET, 1)
    assert len(decisions) == 1
    assert decisions[0]["champion_version"] == v0
    assert decisions[0]["promoted"] is (outcome["status"] == "promoted")
    assert decisions[0]["mlflow_run_id"]
    champion = alias_version(rt.client, reg_name, CHAMPION)
    assert champion is not None
    if outcome["status"] == "promoted":
        assert str(champion.version) == outcome["challenger_version"] != v0
        assert alias_version(rt.client, reg_name, CHALLENGER) is None
    else:
        assert str(champion.version) == v0
        challenger = alias_version(rt.client, reg_name, CHALLENGER)
        assert challenger is not None
        assert str(challenger.version) == outcome["challenger_version"]
    again = retrain_flow(**kw)  # type: ignore[arg-type]
    assert again["outcomes"][0]["status"] == "skipped"
    assert len(repo.promotion_decisions(ASSET, 1)) == 1

    # -- the next forecast uses whatever the registry now says is champion ---------------------
    forecast_flow(**kw)  # type: ignore[arg-type]
    served = repo.latest_forecast(ASSET, 1)
    assert served is not None
    now = alias_version(rt.client, reg_name, CHAMPION)
    assert now is not None
    assert served["model_version"] == str(now.version)

    # -- bookkeeping: every flow succeeded; metrics are exportable ---------------------------
    assert set(repo.last_success()) == {"ingest", "forecast", "evaluate", "retrain"}
    metrics = PipelineMetrics()
    export_from_store(repo, metrics)
    text = generate_latest(metrics.registry).decode()
    assert f'volforecast_forecast_var{{asset="{ASSET}",horizon="1"}}' in text
    assert "volforecast_pipeline_last_success_timestamp_seconds" in text
    assert "volforecast_drift_share" in text
    mlflow.set_tracking_uri("sqlite:///:memory:")
