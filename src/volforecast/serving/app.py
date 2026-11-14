"""Placeholder ASGI app (platform contract stub).

A later milestone replaces this module with the real forecasting API. The platform
(Dockerfile, Helm chart, Prometheus, k6) is built against exactly these endpoints:

* ``GET /healthz``  liveness
* ``GET /readyz``   readiness
* ``GET /metrics``  Prometheus exposition
* ``GET /v1/forecast``  currently 503
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)

REGISTRY = CollectorRegistry(auto_describe=True)

HTTP_REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds.",
    ["route", "method", "status"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 2.5, 5.0),
    registry=REGISTRY,
)
HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests.",
    ["route", "method", "status"],
    registry=REGISTRY,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.ready = True
    yield
    app.state.ready = False


app = FastAPI(title="volforecast (stub)", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def record_metrics(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    start = time.perf_counter()
    status = "500"
    try:
        response = await call_next(request)
        status = str(response.status_code)
        return response
    finally:
        route = request.scope.get("route")
        path = getattr(route, "path", None) or "unmatched"  # low-cardinality label
        labels = {"route": path, "method": request.method, "status": status}
        HTTP_REQUEST_DURATION.labels(**labels).observe(time.perf_counter() - start)
        HTTP_REQUESTS_TOTAL.labels(**labels).inc()


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
async def readyz() -> dict[str, object]:
    return {"status": "ready", "stub": True}


@app.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


@app.get("/v1/forecast")
async def forecast(asset: str = "BTCUSDT", horizon: int = 1) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": "not implemented"})
