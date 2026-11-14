from __future__ import annotations

from fastapi.testclient import TestClient

from volforecast.serving.app import app


def test_health_ready_metrics_and_forecast_stub() -> None:
    with TestClient(app) as client:
        assert client.get("/healthz").json() == {"status": "ok"}
        assert client.get("/readyz").json() == {"status": "ready", "stub": True}
        assert client.get("/v1/forecast?asset=BTCUSDT&horizon=1").status_code == 503
        body = client.get("/metrics").text
    assert "http_request_duration_seconds_bucket" in body
    assert 'http_requests_total{method="GET",route="/healthz",status="200"}' in body
