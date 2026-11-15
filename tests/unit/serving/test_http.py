# ruff: noqa: PT018
"""Cross-cutting HTTP behaviour: ETag/304, rate limiting, CORS, request ids, OpenAPI, metrics."""

from __future__ import annotations

import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.unit.serving.conftest import make_settings
from volforecast.forecast_store import ForecastRepository
from volforecast.monitoring.prom import PipelineMetrics
from volforecast.serving.app import create_app
from volforecast.serving.metrics import RecordingMetrics
from volforecast.serving.ratelimit import TokenBucketLimiter

CACHEABLE = [
    "/v1/forecast?asset=BTCUSDT&horizon=1",
    "/v1/forecast/history?asset=BTCUSDT&horizon=1",
    "/v1/models",
    "/v1/backtest/summary",
]


# ------------------------------------------------------------------ ETag / 304
@pytest.mark.parametrize("path", CACHEABLE)
def test_etag_and_304(
    client_factory: Callable[..., Any], seeded: ForecastRepository, results_file: Path, path: str
) -> None:
    with client_factory() as c:
        r = c.get(path)
        assert r.status_code == 200
        etag = r.headers["etag"]
        assert re.fullmatch(r'"[0-9a-f]{32}"', etag)
        assert r.headers["cache-control"] == "public, max-age=60"
        for header in (etag, f"W/{etag}", f'"nope", {etag}', "*"):
            r304 = c.get(path, headers={"If-None-Match": header})
            assert r304.status_code == 304, header
            assert r304.content == b""
            assert r304.headers["etag"] == etag
            assert r304.headers["cache-control"] == "public, max-age=60"
        assert c.get(path, headers={"If-None-Match": '"stale"'}).status_code == 200
        assert c.get(path).headers["etag"] == etag  # deterministic


def test_etag_changes_with_the_data_and_max_age_is_configurable(
    client_factory: Callable[..., Any], seeded: ForecastRepository
) -> None:
    with client_factory(cache_max_age_s=5) as c:
        r = c.get("/v1/forecast?asset=BTCUSDT&horizon=1")
        assert r.headers["cache-control"] == "public, max-age=5"
        assert c.get("/v1/forecast?asset=BTCUSDT&horizon=5").headers["etag"] != r.headers["etag"]


def test_errors_and_probes_are_not_cacheable(client: TestClient) -> None:
    assert "etag" not in client.get("/v1/forecast?asset=NOPE1").headers
    assert "etag" not in client.get("/healthz").headers


# ------------------------------------------------------------------ rate limiting
class FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def test_token_bucket_unit() -> None:
    clock = FakeClock()
    lim = TokenBucketLimiter(rate=2.0, burst=3, clock=clock)
    assert [lim.acquire("a") for _ in range(3)] == [0.0, 0.0, 0.0]
    wait = lim.acquire("a")
    assert wait == pytest.approx(0.5)  # one token every 0.5 s
    assert lim.acquire("b") == 0.0  # other clients are independent
    clock.t += 0.5
    assert lim.acquire("a") == 0.0
    assert lim.acquire("a") > 0
    clock.t += 1000  # bucket refills only up to the burst
    assert [lim.acquire("a") for _ in range(3)] == [0.0, 0.0, 0.0]
    assert lim.acquire("a") > 0


def test_token_bucket_evicts_idle_clients_and_disabled_mode() -> None:
    clock = FakeClock()
    lim = TokenBucketLimiter(rate=1.0, burst=2, clock=clock)
    for i in range(100):
        lim.acquire(f"ip{i}")
    assert len(lim) == 100
    clock.t += 60
    lim.acquire("fresh")
    assert len(lim) <= 2
    off = TokenBucketLimiter(rate=0, burst=1, clock=clock)
    assert all(off.acquire("x") == 0.0 for _ in range(50))


def test_rate_limit_429_with_retry_after(
    client_factory: Callable[..., Any], seeded: ForecastRepository
) -> None:
    clock = FakeClock()
    with client_factory(rate_limit_rps=1.0, rate_limit_burst=3) as c:
        c.app.state.limiter = TokenBucketLimiter(1.0, 3, clock=clock)  # type: ignore[attr-defined]
        codes = [c.get("/v1/models").status_code for _ in range(5)]
        assert codes == [200, 200, 200, 429, 429]
        r = c.get("/v1/models")
        assert r.status_code == 429
        assert int(r.headers["retry-after"]) >= 1
        assert r.json()["error"]["code"] == "rate_limited"
        assert r.json()["error"]["request_id"] == r.headers["x-request-id"]
        # probes and metrics are never limited
        assert all(c.get(p).status_code == 200 for p in ["/healthz", "/readyz", "/metrics"] * 5)
        clock.t += 2
        assert c.get("/v1/models").status_code == 200
        # the 429s are counted under the route template, not "unmatched"
        text = c.get("/metrics").text
        assert 'http_requests_total{method="GET",route="/v1/models",status="429"}' in text


def test_rate_limit_is_per_client_ip_and_xff_only_when_trusted(
    client_factory: Callable[..., Any], seeded: ForecastRepository
) -> None:
    clock = FakeClock()
    with client_factory(rate_limit_rps=1.0, rate_limit_burst=1, trusted_proxy_hops=0) as c:
        c.app.state.limiter = TokenBucketLimiter(1.0, 1, clock=clock)  # type: ignore[attr-defined]
        assert c.get("/v1/models", headers={"X-Forwarded-For": "1.1.1.1"}).status_code == 200
        # header ignored (untrusted): the spoofed address does not buy a fresh bucket
        assert c.get("/v1/models", headers={"X-Forwarded-For": "2.2.2.2"}).status_code == 429
    with client_factory(rate_limit_rps=1.0, rate_limit_burst=1, trusted_proxy_hops=1) as c:
        c.app.state.limiter = TokenBucketLimiter(1.0, 1, clock=clock)  # type: ignore[attr-defined]
        # one trusted proxy appends the real peer; entries to its left are client-controlled
        hdr = {"X-Forwarded-For": "6.6.6.6, 1.1.1.1"}
        assert c.get("/v1/models", headers=hdr).status_code == 200
        spoof = {"X-Forwarded-For": "7.7.7.7, 1.1.1.1"}
        assert c.get("/v1/models", headers=spoof).status_code == 429
        other = {"X-Forwarded-For": "6.6.6.6, 2.2.2.2"}
        assert c.get("/v1/models", headers=other).status_code == 200


# ------------------------------------------------------------------ CORS
def test_cors_allowlist(client_factory: Callable[..., Any], seeded: ForecastRepository) -> None:
    allowed = "https://dash.example.com"
    with client_factory(cors_origins=f"{allowed}, https://other.example.com") as c:
        ok = c.get("/v1/models", headers={"Origin": allowed})
        assert ok.headers["access-control-allow-origin"] == allowed
        assert "ETag" in ok.headers["access-control-expose-headers"]
        bad = c.get("/v1/models", headers={"Origin": "https://evil.example.com"})
        assert bad.status_code == 200 and "access-control-allow-origin" not in bad.headers
        pre = c.options(
            "/v1/models",
            headers={"Origin": allowed, "Access-Control-Request-Method": "GET"},
        )
        assert pre.status_code == 200 and pre.headers["access-control-allow-origin"] == allowed
        pre_bad = c.options(
            "/v1/models",
            headers={"Origin": "https://evil.example.com", "Access-Control-Request-Method": "GET"},
        )
        assert "access-control-allow-origin" not in pre_bad.headers


def test_cors_is_off_by_default_and_never_wildcard(client: TestClient) -> None:
    r = client.get("/v1/models", headers={"Origin": "https://dash.example.com"})
    assert "access-control-allow-origin" not in r.headers


def test_cors_origins_parse_json_and_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from volforecast.serving.settings import ApiSettings

    monkeypatch.setenv("VOLF_API_CORS_ORIGINS", "https://a.example,https://b.example")
    assert ApiSettings().cors_origins == ["https://a.example", "https://b.example"]
    monkeypatch.setenv("VOLF_API_CORS_ORIGINS", '["https://c.example"]')
    assert ApiSettings().cors_origins == ["https://c.example"]
    monkeypatch.delenv("VOLF_API_CORS_ORIGINS")
    monkeypatch.setenv("VOLF_RESULTS_PATH", str(tmp_path / "r.json"))
    assert ApiSettings().results_file == tmp_path / "r.json"


# ------------------------------------------------------------------ request id
def test_request_id_accepted_echoed_or_generated(client: TestClient) -> None:
    r = client.get("/healthz", headers={"X-Request-ID": "abc-123.X_y"})
    assert r.headers["x-request-id"] == "abc-123.X_y"
    r = client.get("/v1/forecast?horizon=0", headers={"X-Request-ID": "req-42"})
    assert r.status_code == 422 and r.json()["error"]["request_id"] == "req-42"
    bad = client.get("/healthz", headers={"X-Request-ID": "bad id\twith spaces"})
    assert re.fullmatch(r"[0-9a-f]{32}", bad.headers["x-request-id"])
    assert re.fullmatch(r"[0-9a-f]{32}", client.get("/healthz").headers["x-request-id"])


def test_unhandled_exception_is_a_500_envelope_and_counted(
    client_factory: Callable[..., Any], seeded: ForecastRepository
) -> None:
    with client_factory() as c:
        c.app.state.cache = None  # type: ignore[attr-defined]  # break the handler on purpose
        quiet = TestClient(c.app, raise_server_exceptions=False)
        r = quiet.get("/v1/models", headers={"X-Request-ID": "boom-1"})
        assert r.status_code == 500
        assert r.json()["error"]["code"] == "internal_error"
        assert r.json()["error"]["request_id"] == "boom-1"
        assert 'route="/v1/models",status="500"' in quiet.get("/metrics").text


# ------------------------------------------------------------------ OpenAPI
def test_openapi_documents_everything(client: TestClient) -> None:
    spec = client.get("/openapi.json").json()
    paths = spec["paths"]
    for p in [
        "/v1/forecast",
        "/v1/forecast/history",
        "/v1/models",
        "/v1/backtest/summary",
        "/healthz",
        "/readyz",
        "/metrics",
    ]:
        assert p in paths and "get" in paths[p]
    for p in ["/v1/forecast", "/v1/models", "/v1/backtest/summary", "/v1/forecast/history"]:
        responses = next(iter(paths[p.split("?")[0]].values()))["responses"]
        assert {"200", "304", "429"} <= set(responses), p
        assert "ETag" in responses["200"]["headers"]
    fc = paths["/v1/forecast"]["get"]["responses"]
    assert {"404", "422", "503"} <= set(fc)
    assert fc["422"]["content"]["application/json"]["schema"]["$ref"].endswith("/ErrorResponse")
    assert "HTTPValidationError" not in client.get("/openapi.json").text
    props = spec["components"]["schemas"]["ForecastResponse"]["properties"]
    assert {"forecast_var", "annualized_vol", "model_version", "data_hash", "generated_at"} <= set(
        props
    )
    assert all(p["description"] for p in paths["/v1/forecast"]["get"]["parameters"])


# ------------------------------------------------------------------ metrics
def _labels(text: str, metric: str) -> list[dict[str, str]]:
    out = []
    for line in text.splitlines():
        if line.startswith(metric + "{"):
            out.append(dict(re.findall(r'(\w+)="([^"]*)"', line.split("}")[0])))
    return out


def test_metrics_labels_are_route_templates_not_raw_paths(client: TestClient) -> None:
    client.get("/v1/forecast?asset=BTCUSDT&horizon=1")
    client.get("/v1/forecast?asset=DOGEUSDT&horizon=1")
    client.get("/v1/forecast/history?asset=BTCUSDT&horizon=1")
    client.get("/nope/123456")
    client.get("/v1/forecast/987654321")
    client.post("/v1/models")
    text = client.get("/metrics").text
    assert "123456" not in text and "987654321" not in text and "DOGEUSDT" not in text
    routes = {lab["route"] for lab in _labels(text, "http_requests_total")}
    # the registry is process-wide (earlier tests contribute): only templates may ever appear
    templates = {"/v1/forecast", "/v1/forecast/history", "/v1/models", "/v1/backtest/summary"}
    templates |= {"/healthz", "/readyz", "/metrics", "unmatched"}
    assert (
        {"/v1/forecast", "/v1/forecast/history", "unmatched", "/v1/models"} <= routes <= templates
    )
    seen = {
        (lab["route"], lab["method"], lab["status"]) for lab in _labels(text, "http_requests_total")
    }
    assert ("/v1/forecast", "GET", "200") in seen and ("/v1/forecast", "GET", "404") in seen
    assert ("unmatched", "GET", "404") in seen and ("/v1/models", "POST", "405") in seen
    assert "http_request_duration_seconds_bucket" in text


def test_metrics_expose_pipeline_and_model_gauges_from_the_store(client: TestClient) -> None:
    text = client.get("/metrics").text
    assert 'volforecast_model_info{asset="BTCUSDT",horizon="1",version="3"} 1.0' in text
    assert 'volforecast_forecast_var{asset="BTCUSDT",horizon="1"} 0.0012' in text
    assert 'volforecast_pipeline_last_success_timestamp_seconds{flow="forecast"}' in text
    assert 'volforecast_pipeline_last_success_age_seconds{flow="forecast"}' in text
    assert 'volforecast_forecast_age_seconds{asset="BTCUSDT",horizon="1"}' in text
    assert (
        'volforecast_rolling_qlike{asset="BTCUSDT",horizon="1",model="log_har",window="30"}' in text
    )
    assert "volforecast_api_ready 1.0" in text
    ages = {
        (lab["asset"], lab["horizon"]) for lab in _labels(text, "volforecast_forecast_age_seconds")
    }
    assert ages == {("BTCUSDT", "1"), ("BTCUSDT", "5"), ("ETHUSDT", "1")}


def test_recording_metrics_names_match_prom_pipeline_metrics() -> None:
    real = {m.name for m in PipelineMetrics().registry.collect()}
    ours = {m.name for m in RecordingMetrics().families()}
    assert ours == real


# ------------------------------------------------------------------ isolation
def test_api_import_needs_only_the_serve_extra() -> None:
    """The API image has no mlflow; nothing on the serving path may import it (or flows/models)."""
    code = (
        "import sys; import volforecast.serving.app; "
        "bad = [m for m in sys.modules "
        "if m.split('.')[0] in ('mlflow', 'prefect', 'torch', 'boto3') "
        "or m.startswith(('volforecast.registry', 'volforecast.flows', 'volforecast.models'))]; "
        "print(bad); sys.exit(1 if bad else 0)"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stdout + r.stderr


def test_entrypoint_prepares_multiprocess_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from volforecast.serving.__main__ import prepare_multiprocess_dir

    monkeypatch.delenv("PROMETHEUS_MULTIPROC_DIR", raising=False)
    assert prepare_multiprocess_dir(1) is None
    d = tmp_path / "prom"
    d.mkdir()
    (d / "counter_1.db").write_text("stale")
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(d))
    assert prepare_multiprocess_dir(2) == str(d)
    assert list(d.iterdir()) == []
    monkeypatch.delenv("PROMETHEUS_MULTIPROC_DIR")
    got = prepare_multiprocess_dir(2, default_dir=str(tmp_path / "dflt"))
    assert got == str(tmp_path / "dflt") and (tmp_path / "dflt").is_dir()
    monkeypatch.delenv("PROMETHEUS_MULTIPROC_DIR")


def test_make_settings_helper_defaults(tmp_path: Path) -> None:
    s = make_settings(tmp_path)
    assert s.results_file == tmp_path / "results.json" and s.cors_origins == []
    assert create_app(s).title == "volforecast API"
