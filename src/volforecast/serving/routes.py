"""HTTP routes. ``/v1/*`` is rate limited; probes and ``/metrics`` are not."""

from __future__ import annotations

import hashlib
import json
import math
import time
from datetime import date
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
from fastapi import APIRouter, Depends, Query, Request, Response
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
from starlette.concurrency import run_in_threadpool

from volforecast.forecast_store import ForecastRepository
from volforecast.logging import get_logger
from volforecast.serving.errors import ApiError
from volforecast.serving.metrics import StoreCollector, render_metrics
from volforecast.serving.ratelimit import enforce_rate_limit
from volforecast.serving.schemas import (
    BacktestSummary,
    ErrorResponse,
    ForecastResponse,
    HealthResponse,
    HistoryItem,
    HistoryResponse,
    ModelsResponse,
    ReadyResponse,
)
from volforecast.serving.settings import ApiSettings
from volforecast.serving.snapshot import (
    CachedBody,
    SnapshotCache,
    as_utc,
    dump_json,
    finite_or_none,
    make_etag,
    vol_fields,
)

log = get_logger(__name__)

MAX_HISTORY_LIMIT = 1000

Resp = dict[int | str, dict[str, Any]]
_ERR = {"model": ErrorResponse}
_RESP_404: Resp = {404: {**_ERR, "description": "Unknown asset, or no data for the asset/horizon."}}
_RESP_422: Resp = {422: {**_ERR, "description": "Invalid query parameters."}}
_RESP_429: Resp = {
    429: {
        **_ERR,
        "description": "Rate limit exceeded.",
        "headers": {"Retry-After": {"schema": {"type": "integer"}, "description": "Seconds."}},
    }
}
_RESP_503: Resp = {503: {**_ERR, "description": "Forecast store not loaded yet or unreachable."}}
_RESP_304: Resp = {304: {"description": "Not modified (If-None-Match matched the current ETag)."}}
_CACHE_HEADERS = {
    "ETag": {"schema": {"type": "string"}, "description": "Strong validator of the body."},
    "Cache-Control": {"schema": {"type": "string"}},
}


def _ok(model: type[Any], description: str) -> Resp:
    return {200: {"model": model, "description": description, "headers": _CACHE_HEADERS}}


Asset = Annotated[
    str,
    Query(
        pattern=r"^[A-Za-z0-9]{2,20}$",
        description="Trading pair, case-insensitive (e.g. BTCUSDT).",
        examples=["BTCUSDT"],
    ),
]
Horizon = Annotated[int, Query(ge=1, le=60, description="Forecast horizon in days.", examples=[1])]


def etag_matches(header: str | None, etag: str) -> bool:
    """Weak comparison of an ``If-None-Match`` header (list, ``*`` and ``W/`` prefixes)."""
    if not header:
        return False
    if header.strip() == "*":
        return True
    wanted = etag.removeprefix("W/")
    return any(part.strip().removeprefix("W/") == wanted for part in header.split(","))


def cached_json(
    request: Request, cached: CachedBody, media_type: str = "application/json"
) -> Response:
    """200 with ETag + Cache-Control, or an empty 304 when the client's copy is current."""
    api: ApiSettings = request.app.state.api
    headers = {"ETag": cached.etag, "Cache-Control": f"public, max-age={api.cache_max_age_s}"}
    if etag_matches(request.headers.get("if-none-match"), cached.etag):
        return Response(status_code=304, headers=headers)
    return Response(cached.body, media_type=media_type, headers=headers)


def _cache(request: Request) -> SnapshotCache:
    cache: SnapshotCache = request.app.state.cache
    return cache


v1 = APIRouter(prefix="/v1", dependencies=[Depends(enforce_rate_limit)], tags=["v1"])
ops = APIRouter(tags=["ops"])


@v1.get(
    "/forecast",
    response_model=ForecastResponse,
    summary="Latest forecast",
    responses={
        **_ok(ForecastResponse, "Latest served (champion) forecast."),
        **_RESP_304,
        **_RESP_404,
        **_RESP_422,
        **_RESP_429,
        **_RESP_503,
    },
)
async def get_forecast(
    request: Request, asset: Asset = "BTCUSDT", horizon: Horizon = 1
) -> Response:
    """The most recent precomputed forecast (ADR-0009: nothing is computed per request)."""
    asset = asset.upper()
    snap = _cache(request).snapshot
    if snap.loaded_at is None:
        raise ApiError(503, "not_loaded", "forecast store not loaded yet")
    if asset not in snap.assets:
        raise ApiError(404, "unknown_asset", f"no forecasts for asset {asset!r}")
    cached = snap.forecasts.get((asset, horizon))
    if cached is None:
        raise ApiError(404, "no_data", f"no forecast for {asset} at horizon {horizon}")
    return cached_json(request, cached)


def _opt_str(x: Any) -> str | None:
    return x if isinstance(x, str) else None


def _history_items(df: pd.DataFrame, horizon: int) -> list[HistoryItem]:
    items: list[HistoryItem] = []
    for r in df.to_dict("records"):
        var = finite_or_none(r["forecast_var"])
        if var is None:
            continue
        daily, annual = vol_fields(var, horizon)
        items.append(
            HistoryItem(
                origin_date=pd.Timestamp(r["origin_date"]).date(),
                target_date=pd.Timestamp(r["target_date"]).date(),
                forecast_var=var,
                daily_vol=daily,
                annualized_vol=annual,
                var95=finite_or_none(r["var95"]),
                var99=finite_or_none(r["var99"]),
                es975=finite_or_none(r["es975"]),
                var_dist=_opt_str(r["dist"]),
                model_name=str(r["model_name"]),
                model_version=str(r["model_version"]),
                role=str(r["role"]),
                data_hash=_opt_str(r["data_hash"]),
                generated_at=as_utc(pd.Timestamp(r["created_at"]).to_pydatetime()),
                realized_var=finite_or_none(r["realized_var"]),
                realized_ret=finite_or_none(r["realized_ret"]),
            )
        )
    return items


@v1.get(
    "/forecast/history",
    response_model=HistoryResponse,
    summary="Forecast history with realized values",
    responses={
        **_ok(HistoryResponse, "Served forecasts, newest first, joined with realized values."),
        **_RESP_304,
        **_RESP_404,
        **_RESP_422,
        **_RESP_429,
        **_RESP_503,
    },
)
async def get_history(
    request: Request,
    asset: Asset = "BTCUSDT",
    horizon: Horizon = 1,
    start: Annotated[date | None, Query(description="First origin date (inclusive).")] = None,
    end: Annotated[date | None, Query(description="Last origin date (inclusive).")] = None,
    limit: Annotated[int, Query(ge=1, le=MAX_HISTORY_LIMIT, description="Page size.")] = 100,
    offset: Annotated[int, Query(ge=0, description="Rows to skip (newest first).")] = 0,
) -> Response:
    """Paged history of served forecasts. ``realized_*`` are null until the outcome is known."""
    asset = asset.upper()
    if start is not None and end is not None and start > end:
        raise ApiError(
            422,
            "validation_error",
            "start must not be after end",
            details=[{"loc": ["query", "start"], "msg": "start > end", "type": "value_error"}],
        )
    repo: ForecastRepository = request.app.state.repo
    snap = _cache(request).snapshot
    try:
        df = await run_in_threadpool(repo.forecast_history, asset, horizon, start, end)
    except SQLAlchemyError as exc:
        log.warning("history_query_failed", err=repr(exc))
        raise ApiError(503, "unavailable", "forecast store unreachable") from exc
    if df.empty and (asset, horizon) not in snap.forecasts:
        raise ApiError(404, "no_data", f"no forecasts for {asset} at horizon {horizon}")
    df = df.sort_values("origin_date", ascending=False)
    total = len(df)
    page = _history_items(df.iloc[offset : offset + limit], horizon)
    nxt = offset + limit
    body = HistoryResponse(
        asset=asset,
        horizon=horizon,
        start=start,
        end=end,
        limit=limit,
        offset=offset,
        total=total,
        next_offset=nxt if nxt < total else None,
        items=page,
    )
    raw = dump_json(body)
    return cached_json(request, CachedBody(raw, make_etag(raw)))


@v1.get(
    "/models",
    response_model=ModelsResponse,
    summary="Champion / challenger per asset and horizon",
    responses={
        **_ok(ModelsResponse, "Served model, shadow challenger and latest promotion decision."),
        **_RESP_304,
        **_RESP_429,
        **_RESP_503,
    },
)
async def get_models(request: Request) -> Response:
    snap = _cache(request).snapshot
    if snap.models is None:
        raise ApiError(503, "not_loaded", "forecast store not loaded yet")
    return cached_json(request, snap.models)


class _ResultsFile:
    """Reads ``results.json`` once per (mtime, size); serves the validated bytes."""

    def __init__(self) -> None:
        self._key: tuple[int, int] | None = None
        self._cached: CachedBody | None = None

    def load(self, path: Path) -> CachedBody | None:
        try:
            st = path.stat()
        except OSError:
            return None
        key = (st.st_mtime_ns, st.st_size)
        if key != self._key or self._cached is None:
            raw = path.read_bytes()
            doc = json.loads(raw)
            if not isinstance(doc, dict) or "meta" not in doc or "assets" not in doc:
                raise ValueError("results file lacks 'meta'/'assets'")
            etag = '"' + hashlib.sha256(raw).hexdigest()[:32] + '"'
            self._cached, self._key = CachedBody(raw, etag), key
        return self._cached


_results = _ResultsFile()


@v1.get(
    "/backtest/summary",
    response_model=BacktestSummary,
    summary="Official walk-forward backtest summary",
    responses={
        **_ok(BacktestSummary, "Contents of the configured results.json."),
        **_RESP_304,
        **_RESP_404,
        **_RESP_429,
    },
)
async def get_backtest_summary(request: Request) -> Response:
    """Served as-is from ``VOLF_RESULTS_PATH`` (default ``<reports_dir>/results.json``)."""
    api: ApiSettings = request.app.state.api
    try:
        cached = await run_in_threadpool(_results.load, api.results_file)
    except (ValueError, OSError) as exc:
        log.error("results_unreadable", path=str(api.results_file), err=repr(exc))
        raise ApiError(500, "results_unreadable", "results file is not valid JSON") from exc
    if cached is None:
        raise ApiError(404, "no_results", "no backtest results are configured/available")
    return cached_json(request, cached)


# --------------------------------------------------------------------------- ops
@ops.get("/healthz", response_model=HealthResponse, summary="Liveness")
async def healthz() -> HealthResponse:
    """Always 200 while the process runs; touches nothing else."""
    return HealthResponse()


@ops.get(
    "/readyz",
    response_model=ReadyResponse,
    summary="Readiness",
    responses={503: {"model": ReadyResponse, "description": "Not ready."}},
)
async def readyz(request: Request) -> Response:
    """200 once the store is reachable and a champion forecast is loaded, else 503."""
    cache = _cache(request)
    snap = cache.snapshot
    ready = cache.ready
    age = None if snap.loaded_at is None else max(0.0, time.time() - snap.loaded_at)
    detail = None
    if not ready:
        detail = cache.last_error or (
            "no champion forecast in the store" if cache.healthy else "store not loaded yet"
        )
    body = ReadyResponse(
        status="ready" if ready else "not_ready",
        database=cache.healthy,
        champion_forecasts=snap.ready_pairs,
        snapshot_age_seconds=None if age is None or not math.isfinite(age) else round(age, 3),
        detail=detail,
    )
    return JSONResponse(body.model_dump(mode="json"), status_code=200 if ready else 503)


@ops.get("/metrics", summary="Prometheus metrics", response_class=Response)
async def metrics(request: Request) -> Response:
    collector: StoreCollector = request.app.state.store_collector
    body, ctype = render_metrics(collector)
    return Response(body, media_type=ctype)
