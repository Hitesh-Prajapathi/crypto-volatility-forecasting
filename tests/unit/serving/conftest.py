"""Fixtures for the API tests: a file-based sqlite store seeded with fixture rows (no network)."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from volforecast.forecast_store import ForecastRepository
from volforecast.serving.app import create_app
from volforecast.serving.settings import ApiSettings

ORIGINS = [date(2024, 1, 8), date(2024, 1, 9), date(2024, 1, 10)]


def fc_row(
    asset: str = "BTCUSDT",
    horizon: int = 1,
    origin: date = date(2024, 1, 10),
    role: str = "champion",
    version: str = "3",
    model: str = "log_har",
    var: float = 4.0e-4,
    **kw: Any,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "asset": asset,
        "horizon": horizon,
        "model_name": model,
        "model_version": version,
        "role": role,
        "origin_date": origin,
        "target_date": origin + pd.Timedelta(days=1),
        "forecast_var": var,
        "var95": 0.031 if horizon == 1 else None,
        "var99": 0.046 if horizon == 1 else None,
        "es975": 0.044 if horizon == 1 else None,
        "dist": "fhs" if horizon == 1 else None,
        "data_hash": "sha256:abc123",
    }
    row.update(kw)
    return row


def seed(repo: ForecastRepository) -> None:
    """BTC h1 (3 days) + h5, ETH h1 fallback only, a shadow challenger, HAR benchmark."""
    rows: list[dict[str, Any]] = []
    for i, o in enumerate(ORIGINS):
        rows.append(fc_row(origin=o, var=4.0e-4 * (1 + i)))
        rows.append(
            fc_row(origin=o, role="benchmark", version="har_benchmark", model="har_benchmark")
        )
    rows.append(fc_row(horizon=5, var=2.0e-3))
    rows.append(fc_row(role="challenger", version="4", model="harq"))
    rows.append(fc_row(asset="ETHUSDT", role="fallback", version="fallback:log_har", var=9.0e-4))
    repo.upsert_forecasts(rows)
    # realized: known for the first two target days, unknown for the last origin's next day
    days = pd.to_datetime(["2024-01-09", "2024-01-10"])
    panel = pd.DataFrame(
        {"rv": [3.0e-4, 5.0e-4], "ret": [-0.01, 0.02], "complete": [True, True]}, index=days
    )
    repo.upsert_realized("BTCUSDT", panel)
    run = repo.start_run("forecast", date(2024, 1, 10))
    repo.finish_run(run, "success")
    with repo.session() as s:
        repo.add_promotion_decision(
            s,
            asset="BTCUSDT",
            horizon=1,
            as_of_date=date(2024, 1, 7),
            registered_model="volforecast-btcusdt-h1",
            challenger_version="3",
            challenger_model="log_har",
            champion_version="2",
            promoted=True,
            reasons=["all gates passed"],
            metrics={"qlike_ratio_vs_har": 0.93},
            mlflow_run_id="abc",
        )
    repo.upsert_metrics(
        [
            {
                "asset": "BTCUSDT",
                "horizon": 1,
                "model": "log_har",
                "metric": "rolling_qlike",
                "window": 30,
                "as_of_date": date(2024, 1, 10),
                "value": 0.2,
            }
        ]
    )


def make_settings(tmp_path: Path, **kw: Any) -> ApiSettings:
    base: dict[str, Any] = {
        "database_url": f"sqlite:///{tmp_path / 'store.db'}",
        "results_path": tmp_path / "results.json",
        "reload_interval_s": 3600.0,  # reloads are explicit (refresh_sync) unless a test says so
        "not_ready_retry_s": 3600.0,
        "rate_limit_rps": 0.0,
    }
    base.update(kw)
    return ApiSettings(**base)


@pytest.fixture
def repo(tmp_path: Path) -> ForecastRepository:
    return ForecastRepository(f"sqlite:///{tmp_path / 'store.db'}")


@pytest.fixture
def seeded(repo: ForecastRepository) -> ForecastRepository:
    seed(repo)
    return repo


def wait_loaded(app: FastAPI) -> None:
    """Block until the first (background) reload attempt finished."""
    import time

    deadline = time.time() + 10
    cache = app.state.cache
    while time.time() < deadline:
        if cache.healthy or cache.last_error:
            return
        time.sleep(0.01)
    raise AssertionError("the initial reload did not finish")


@contextmanager
def _client(tmp_path: Path, **kw: Any) -> Iterator[TestClient]:
    app = create_app(make_settings(tmp_path, **kw))
    with TestClient(app) as c:
        wait_loaded(app)
        yield c


@pytest.fixture
def client(tmp_path: Path, seeded: ForecastRepository) -> Iterator[TestClient]:
    with _client(tmp_path) as c:
        yield c


@pytest.fixture
def client_factory(tmp_path: Path) -> Callable[..., Any]:
    """``with client_factory(**api_settings) as c`` (store must be seeded by the caller)."""

    def factory(**kw: Any) -> Any:
        return _client(tmp_path, **kw)

    return factory


@pytest.fixture
def results_file(tmp_path: Path) -> Path:
    doc = {
        "meta": {"models": ["har", "log_har"], "seed": 0},
        "assets": {"BTCUSDT": {"horizons": {"1": {"qlike": 0.2}}}},
        "var": [{"x": 1}],
    }
    p = tmp_path / "results.json"
    p.write_text(json.dumps(doc))
    return p


def cache_of(c: TestClient) -> Any:
    return c.app.state.cache  # type: ignore[attr-defined]
