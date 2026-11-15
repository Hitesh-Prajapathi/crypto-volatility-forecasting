# ruff: noqa: PT018
"""Endpoint behaviour: happy paths, 404/422, error envelope, readiness, hot reload."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.unit.serving.conftest import cache_of, fc_row, make_settings, seed, wait_loaded
from volforecast.contracts import ANNUALIZATION_DAYS
from volforecast.forecast_store import ForecastRepository
from volforecast.serving.app import create_app


def err(resp: Any) -> dict[str, Any]:
    body: dict[str, Any] = resp.json()
    assert set(body) == {"error"}
    assert {"code", "message", "request_id"} <= set(body["error"])
    return body["error"]  # type: ignore[no-any-return]


# ------------------------------------------------------------------ /v1/forecast
def test_forecast_happy_path(client: TestClient) -> None:
    r = client.get("/v1/forecast", params={"asset": "BTCUSDT", "horizon": 1})
    assert r.status_code == 200
    b = r.json()
    assert (b["asset"], b["horizon"], b["role"], b["model_version"]) == (
        "BTCUSDT",
        1,
        "champion",
        "3",
    )
    assert b["origin_date"] == "2024-01-10" and b["target_date"] == "2024-01-11"
    assert b["forecast_var"] == pytest.approx(1.2e-3)  # newest origin: 4e-4 * 3
    assert b["daily_vol"] == pytest.approx(math.sqrt(1.2e-3))
    assert b["annualized_vol"] == pytest.approx(math.sqrt(1.2e-3 * ANNUALIZATION_DAYS))
    assert (b["var95"], b["var99"], b["es975"], b["var_dist"]) == (0.031, 0.046, 0.044, "fhs")
    assert b["data_hash"] == "sha256:abc123"
    assert b["generated_at"].endswith("Z") or b["generated_at"].endswith("+00:00")
    assert r.headers["x-request-id"]
    assert r.headers["cache-control"] == "public, max-age=60"


def test_forecast_h5_divides_variance_by_horizon_and_has_no_var(client: TestClient) -> None:
    b = client.get("/v1/forecast?asset=btcusdt&horizon=5").json()  # case-insensitive
    assert b["asset"] == "BTCUSDT"
    assert b["daily_vol"] == pytest.approx(math.sqrt(2.0e-3 / 5))
    assert b["var99"] is None and b["es975"] is None and b["var_dist"] is None


def test_forecast_defaults_and_fallback_role(client: TestClient) -> None:
    assert client.get("/v1/forecast").json()["asset"] == "BTCUSDT"
    eth = client.get("/v1/forecast?asset=ETHUSDT").json()
    assert eth["role"] == "fallback" and eth["model_version"] == "fallback:log_har"


def test_forecast_never_serves_shadow_or_benchmark_rows(client: TestClient) -> None:
    b = client.get("/v1/forecast?asset=BTCUSDT&horizon=1").json()
    assert b["model_name"] == "log_har" and b["role"] == "champion"


@pytest.mark.parametrize(
    ("query", "code"),
    [
        ("asset=DOGEUSDT&horizon=1", "unknown_asset"),
        ("asset=BTCUSDT&horizon=7", "no_data"),
        ("asset=ETHUSDT&horizon=5", "no_data"),
    ],
)
def test_forecast_404(client: TestClient, query: str, code: str) -> None:
    r = client.get(f"/v1/forecast?{query}")
    assert r.status_code == 404
    assert err(r)["code"] == code


@pytest.mark.parametrize(
    "query",
    [
        "horizon=0",
        "horizon=-1",
        "horizon=abc",
        "horizon=1000",
        "asset=BTC/USDT",
        "asset=a",
        "asset=",
    ],
)
def test_forecast_422(client: TestClient, query: str) -> None:
    r = client.get(f"/v1/forecast?{query}")
    assert r.status_code == 422
    e = err(r)
    assert e["code"] == "validation_error" and e["details"]


def test_unmatched_route_and_wrong_method_use_the_envelope(client: TestClient) -> None:
    r = client.get("/nope")
    assert r.status_code == 404 and err(r)["code"] == "not_found"
    r = client.post("/v1/forecast")
    assert r.status_code == 405 and err(r)["code"] == "method_not_allowed"


# ------------------------------------------------------------------ history
def test_history_joins_realized_values(client: TestClient) -> None:
    b = client.get("/v1/forecast/history?asset=BTCUSDT&horizon=1").json()
    assert (b["total"], b["limit"], b["offset"], b["next_offset"]) == (3, 100, 0, None)
    items = b["items"]
    assert [i["origin_date"] for i in items] == ["2024-01-10", "2024-01-09", "2024-01-08"]
    # origin 01-08: realized rv(01-09) known; origin 01-09: rv(01-10) known; origin 01-10: unknown
    assert items[2]["realized_var"] == pytest.approx(3.0e-4)
    assert items[2]["realized_ret"] == pytest.approx(-0.01)
    assert items[1]["realized_var"] == pytest.approx(5.0e-4)
    assert items[0]["realized_var"] is None and items[0]["realized_ret"] is None
    assert all(i["role"] == "champion" for i in items)
    assert items[0]["annualized_vol"] == pytest.approx(
        math.sqrt(items[0]["forecast_var"] * ANNUALIZATION_DAYS)
    )


def test_history_paging_and_range(client: TestClient) -> None:
    p1 = client.get("/v1/forecast/history?asset=BTCUSDT&horizon=1&limit=2").json()
    assert [i["origin_date"] for i in p1["items"]] == ["2024-01-10", "2024-01-09"]
    assert p1["next_offset"] == 2 and p1["total"] == 3
    p2 = client.get("/v1/forecast/history?asset=BTCUSDT&horizon=1&limit=2&offset=2").json()
    assert [i["origin_date"] for i in p2["items"]] == ["2024-01-08"] and p2["next_offset"] is None
    rng = client.get(
        "/v1/forecast/history?asset=BTCUSDT&horizon=1&start=2024-01-09&end=2024-01-09"
    ).json()
    assert rng["total"] == 1 and rng["start"] == "2024-01-09"
    empty = client.get("/v1/forecast/history?asset=BTCUSDT&horizon=1&start=2030-01-01")
    assert empty.status_code == 200 and empty.json()["items"] == []


def test_history_errors(client: TestClient) -> None:
    assert client.get("/v1/forecast/history?asset=NOPEUSDT&horizon=1").status_code == 404
    for q in ("limit=0", "limit=100000", "offset=-1", "start=notadate", "horizon=0"):
        r = client.get(f"/v1/forecast/history?asset=BTCUSDT&{q}")
        assert r.status_code == 422, q
    r = client.get("/v1/forecast/history?asset=BTCUSDT&start=2024-02-01&end=2024-01-01")
    assert r.status_code == 422 and err(r)["code"] == "validation_error"


# ------------------------------------------------------------------ models
def test_models(client: TestClient) -> None:
    models = client.get("/v1/models").json()["models"]
    by_key = {(m["asset"], m["horizon"]): m for m in models}
    assert set(by_key) == {("BTCUSDT", 1), ("BTCUSDT", 5), ("ETHUSDT", 1)}
    btc = by_key[("BTCUSDT", 1)]
    assert btc["champion"]["model_version"] == "3" and btc["champion"]["role"] == "champion"
    assert btc["champion"]["n_forecasts"] == 3
    assert btc["challenger"] == {"model_name": "harq", "model_version": "4"}
    promo = btc["latest_promotion"]
    assert promo["promoted"] is True and promo["champion_version"] == "2"
    assert promo["metrics"] == {"qlike_ratio_vs_har": 0.93} and promo["reasons"] == [
        "all gates passed"
    ]
    assert by_key[("BTCUSDT", 5)]["challenger"] is None
    assert by_key[("BTCUSDT", 5)]["latest_promotion"] is None
    assert by_key[("ETHUSDT", 1)]["champion"]["role"] == "fallback"


# ------------------------------------------------------------------ backtest summary
def test_backtest_summary_404_when_absent(client: TestClient) -> None:
    r = client.get("/v1/backtest/summary")
    assert r.status_code == 404 and err(r)["code"] == "no_results"


def test_backtest_summary_served_from_configured_path(
    client_factory: Callable[..., Any], seeded: ForecastRepository, results_file: Path
) -> None:
    with client_factory() as c:
        r = c.get("/v1/backtest/summary")
        assert r.status_code == 200
        assert r.json()["meta"]["seed"] == 0 and "BTCUSDT" in r.json()["assets"]
        assert r.headers["etag"]
        # file replaced -> new content and new ETag, no restart
        old = r.headers["etag"]
        results_file.write_text('{"meta": {"seed": 1, "pad": "xxxxxxxx"}, "assets": {}}')
        r2 = c.get("/v1/backtest/summary")
        assert r2.json()["meta"]["seed"] == 1 and r2.headers["etag"] != old


def test_backtest_summary_invalid_file_is_a_clean_500(
    client_factory: Callable[..., Any], seeded: ForecastRepository, results_file: Path
) -> None:
    results_file.write_text("{not json")
    with client_factory() as c:
        r = c.get("/v1/backtest/summary")
    assert r.status_code == 500 and err(r)["code"] == "results_unreadable"


# ------------------------------------------------------------------ probes / readiness
def test_healthz_is_always_ok_even_without_a_store(tmp_path: Path) -> None:
    app = create_app(make_settings(tmp_path / "missing" / "nested"))
    with TestClient(app) as c:
        assert c.get("/healthz").json() == {"status": "ok"}


def test_readyz_503_until_champion_forecast_then_200(
    tmp_path: Path, repo: ForecastRepository
) -> None:
    app = create_app(make_settings(tmp_path))
    with TestClient(app) as c:
        wait_loaded(app)
        r = c.get("/readyz")  # tables exist but are empty
        assert r.status_code == 503
        assert r.json()["status"] == "not_ready" and r.json()["champion_forecasts"] == 0
        assert c.get("/healthz").status_code == 200
        assert c.get("/v1/forecast").status_code == 404  # loaded, but no data yet
        repo.upsert_forecasts([fc_row()])
        assert cache_of(c).refresh_sync() is True
        r = c.get("/readyz")
        assert r.status_code == 200
        assert r.json()["status"] == "ready" and r.json()["champion_forecasts"] == 1
        assert r.json()["database"] is True


def test_readyz_503_when_database_is_unreachable_or_uninitialised(tmp_path: Path) -> None:
    # no tables at all (the API never runs DDL): reload fails -> not ready, no crash
    app = create_app(make_settings(tmp_path))
    with TestClient(app) as c:
        wait_loaded(app)
        r = c.get("/readyz")
        assert r.status_code == 503 and r.json()["database"] is False
        assert r.json()["detail"]
        assert c.get("/v1/forecast").status_code == 503
        assert c.get("/v1/models").status_code == 503
        assert c.get("/healthz").status_code == 200


def test_fallback_only_store_is_not_ready_unless_allowed(
    tmp_path: Path, repo: ForecastRepository
) -> None:
    repo.upsert_forecasts([fc_row(role="fallback", version="fallback:log_har")])
    strict = create_app(make_settings(tmp_path))
    with TestClient(strict) as c:
        wait_loaded(strict)
        assert c.get("/readyz").status_code == 503
        assert c.get("/v1/forecast").status_code == 200  # still servable
    lenient = create_app(make_settings(tmp_path, ready_allow_fallback=True))
    with TestClient(lenient) as c:
        wait_loaded(lenient)
        assert c.get("/readyz").status_code == 200


# ------------------------------------------------------------------ hot reload
def test_hot_reload_picks_up_new_forecast_without_restart(
    client: TestClient, seeded: ForecastRepository
) -> None:
    assert client.get("/v1/forecast?asset=BTCUSDT&horizon=1").json()["origin_date"] == "2024-01-10"
    etag = client.get("/v1/forecast?asset=BTCUSDT&horizon=1").headers["etag"]
    seeded.upsert_forecasts([fc_row(origin=date(2024, 1, 11), version="5", var=7.0e-4)])
    seeded.upsert_forecasts([fc_row(asset="SOLUSDT", var=1.0e-3)])
    # not visible before the reload: requests never touch the DB
    assert client.get("/v1/forecast?asset=BTCUSDT&horizon=1").json()["origin_date"] == "2024-01-10"
    assert client.get("/v1/forecast?asset=SOLUSDT").status_code == 404
    cache_of(client).refresh_sync()
    b = client.get("/v1/forecast?asset=BTCUSDT&horizon=1")
    assert b.json()["origin_date"] == "2024-01-11" and b.json()["model_version"] == "5"
    assert b.headers["etag"] != etag
    assert client.get("/v1/forecast?asset=SOLUSDT").status_code == 200
    models = {(m["asset"], m["horizon"]): m for m in client.get("/v1/models").json()["models"]}
    assert models[("BTCUSDT", 1)]["champion"]["model_version"] == "5"


def test_background_loop_reloads_by_itself(tmp_path: Path, seeded: ForecastRepository) -> None:
    app = create_app(make_settings(tmp_path, reload_interval_s=0.05, not_ready_retry_s=0.05))
    with TestClient(app) as c:
        wait_loaded(app)
        seeded.upsert_forecasts([fc_row(origin=date(2024, 1, 12), version="6")])
        deadline = time.time() + 10
        origin = ""
        while time.time() < deadline:
            origin = c.get("/v1/forecast?asset=BTCUSDT&horizon=1").json()["origin_date"]
            if origin == "2024-01-12":
                break
            time.sleep(0.02)
        assert origin == "2024-01-12"


def test_failed_reload_keeps_serving_the_last_snapshot(
    client: TestClient, seeded: ForecastRepository
) -> None:
    cache = cache_of(client)
    original = cache.repo.models_summary

    def boom() -> Any:
        raise RuntimeError("db down")

    cache.repo.models_summary = boom
    try:
        assert cache.refresh_sync() is False
        assert client.get("/readyz").status_code == 503  # DB not reachable
        assert client.get("/v1/forecast?asset=BTCUSDT&horizon=1").status_code == 200  # stale OK
        detail = client.get("/readyz").json()["detail"]
        assert detail == "store reload failed: RuntimeError"  # class only, no SQL / messages
    finally:
        cache.repo.models_summary = original
    assert cache.refresh_sync() is True
    assert client.get("/readyz").status_code == 200


def test_seed_helper_is_idempotent(seeded: ForecastRepository) -> None:
    seed(seeded)
    assert len(seeded.models_summary()) == 3
