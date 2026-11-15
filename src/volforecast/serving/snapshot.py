"""Immutable snapshot of what the API serves, refreshed in the background (hot reload).

Requests only read ``SnapshotCache.snapshot`` (one attribute read, no lock, no I/O). A reload task
builds a complete new :class:`Snapshot` in a worker thread (about 15 small queries) and swaps the
reference atomically. A failed reload keeps the last good snapshot and marks the cache unhealthy
(``/readyz`` turns 503 until a reload succeeds again; the loop retries faster meanwhile).

Cached bodies are the serialized JSON plus its ETag. The ETag is a hash of the body bytes, so every
worker process derives the same value for the same data.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any, cast

from pydantic import BaseModel

from volforecast.contracts import ANNUALIZATION_DAYS
from volforecast.forecast_store import ROLE_CHAMPION, SERVED_ROLES, ForecastRepository
from volforecast.logging import get_logger
from volforecast.monitoring.prom import PipelineMetrics, export_from_store
from volforecast.serving.metrics import RecordingMetrics
from volforecast.serving.schemas import (
    ChampionInfo,
    ForecastResponse,
    ModelEntry,
    ModelRef,
    ModelsResponse,
    PromotionInfo,
)
from volforecast.serving.settings import ApiSettings

log = get_logger(__name__)


def as_utc(t: datetime) -> datetime:
    """sqlite returns naive datetimes, Postgres aware ones: normalise to aware UTC."""
    return t.replace(tzinfo=UTC) if t.tzinfo is None else t.astimezone(UTC)


def finite_or_none(x: Any) -> float | None:
    if x is None:
        return None
    f = float(x)
    return f if math.isfinite(f) else None


def vol_fields(forecast_var: float, horizon: int) -> tuple[float, float]:
    """(daily vol, annualized vol) of an h-day variance forecast (sum over h days)."""
    daily_var = max(forecast_var, 0.0) / horizon
    return math.sqrt(daily_var), math.sqrt(daily_var * ANNUALIZATION_DAYS)


def dump_json(model: BaseModel) -> bytes:
    """Deterministic JSON bytes of a response model (stable ETags across workers)."""
    return json.dumps(
        model.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def make_etag(body: bytes) -> str:
    return '"' + hashlib.sha256(body).hexdigest()[:32] + '"'


@dataclass(frozen=True)
class CachedBody:
    body: bytes
    etag: str

    @classmethod
    def of(cls, model: BaseModel) -> CachedBody:
        body = dump_json(model)
        return cls(body, make_etag(body))


def forecast_response(row: dict[str, Any]) -> ForecastResponse:
    """Map a ``forecasts`` row (``ForecastRepository.latest_forecast``) to the API model."""
    h = int(row["horizon"])
    var = float(row["forecast_var"])
    daily, annual = vol_fields(var, h)
    return ForecastResponse(
        asset=row["asset"],
        horizon=h,
        origin_date=row["origin_date"],
        target_date=row["target_date"],
        forecast_var=var,
        daily_vol=daily,
        annualized_vol=annual,
        var95=finite_or_none(row.get("var95")),
        var99=finite_or_none(row.get("var99")),
        es975=finite_or_none(row.get("es975")),
        var_dist=row.get("dist"),
        model_name=row["model_name"],
        model_version=str(row["model_version"]),
        role=row["role"],
        data_hash=row.get("data_hash"),
        generated_at=as_utc(row["created_at"]),
    )


@dataclass(frozen=True)
class Snapshot:
    loaded_at: float | None = None  # unix time of the successful build
    forecasts: dict[tuple[str, int], CachedBody] = field(default_factory=dict)
    models: CachedBody | None = None
    assets: frozenset[str] = frozenset()
    origins: dict[tuple[str, int], date] = field(default_factory=dict)
    ready_pairs: int = 0
    metrics: RecordingMetrics = field(default_factory=RecordingMetrics)


def build_snapshot(repo: ForecastRepository, api: ApiSettings) -> Snapshot:
    """Query the store and assemble a complete snapshot (blocking; run in a thread)."""
    roles = SERVED_ROLES if api.ready_allow_fallback else (ROLE_CHAMPION,)
    forecasts: dict[tuple[str, int], CachedBody] = {}
    origins: dict[tuple[str, int], date] = {}
    entries: list[ModelEntry] = []
    ready_pairs = 0
    for s in repo.models_summary():
        asset, h = str(s["asset"]), int(s["horizon"])
        latest = repo.latest_forecast(asset, h)
        if latest is None:
            continue
        try:
            resp = forecast_response(latest)
        except ValueError as exc:  # NaN variance etc.: skip the pair rather than fail the reload
            log.warning("snapshot_row_invalid", asset=asset, horizon=h, err=str(exc))
            continue
        forecasts[(asset, h)] = CachedBody.of(resp)
        origins[(asset, h)] = latest["origin_date"]
        if latest["role"] in roles:
            ready_pairs += 1
        promo = repo.promotion_decisions(asset, h, limit=1)
        entries.append(
            ModelEntry(
                asset=asset,
                horizon=h,
                champion=ChampionInfo(
                    model_name=s["model_name"],
                    model_version=str(s["model_version"]),
                    role=s["role"],
                    latest_origin_date=s["latest_origin_date"],
                    n_forecasts=int(s["n_forecasts"]),
                ),
                challenger=(
                    None
                    if s["challenger_version"] is None
                    else ModelRef(
                        model_name=s["challenger_model"],
                        model_version=str(s["challenger_version"]),
                    )
                ),
                latest_promotion=_promotion(promo[0]) if promo else None,
            )
        )
    recorder = RecordingMetrics()
    export_from_store(repo, cast(PipelineMetrics, recorder))
    return Snapshot(
        loaded_at=time.time(),
        forecasts=forecasts,
        models=CachedBody.of(ModelsResponse(models=entries)),
        assets=frozenset(a for a, _ in forecasts),
        origins=origins,
        ready_pairs=ready_pairs,
        metrics=recorder,
    )


def _promotion(row: dict[str, Any]) -> PromotionInfo:
    return PromotionInfo(
        decided_at=as_utc(row["created_at"]),
        as_of_date=row["as_of_date"],
        promoted=bool(row["promoted"]),
        challenger_version=str(row["challenger_version"]),
        challenger_model=row["challenger_model"],
        champion_version=None if row["champion_version"] is None else str(row["champion_version"]),
        reasons=[str(r) for r in row["reasons"]],
        metrics=dict(row["metrics"]),
        mlflow_run_id=row["mlflow_run_id"],
    )


class SnapshotCache:
    """Holds the current snapshot and refreshes it periodically without blocking requests."""

    def __init__(self, repo: ForecastRepository, api: ApiSettings) -> None:
        self.repo = repo
        self.api = api
        self.snapshot = Snapshot()
        #: True while the last reload succeeded (False before the first one).
        self.healthy = False
        self.last_error: str | None = None
        self._lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self.healthy and self.snapshot.ready_pairs > 0

    def refresh_sync(self) -> bool:
        """Reload now (blocking). Returns True if a new snapshot was installed."""
        if not self._lock.acquire(blocking=False):  # a reload is already running
            return False
        try:
            try:
                snap = build_snapshot(self.repo, self.api)
            except Exception as exc:
                self.healthy = False
                # /readyz is unauthenticated: expose the error class only, details go to the log
                self.last_error = f"store reload failed: {type(exc).__name__}"
                log.warning("snapshot_reload_failed", err=repr(exc)[:500])
                return False
            old = self.snapshot
            self.snapshot = snap  # atomic reference swap
            self.healthy = True
            self.last_error = None
            if {k: v.etag for k, v in snap.forecasts.items()} != {
                k: v.etag for k, v in old.forecasts.items()
            }:
                log.info("snapshot_reloaded", pairs=len(snap.forecasts), ready=snap.ready_pairs)
            return True
        finally:
            self._lock.release()

    async def refresh_now(self) -> bool:
        return await asyncio.to_thread(self.refresh_sync)

    async def run(self) -> None:
        """Background loop; never raises (except cancellation)."""
        while True:
            await self.refresh_now()
            delay = self.api.reload_interval_s if self.ready else self.api.not_ready_retry_s
            await asyncio.sleep(min(delay, self.api.reload_interval_s))
