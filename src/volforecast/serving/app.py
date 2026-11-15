"""The volforecast API (ASGI app ``volforecast.serving.app:app``).

Serves precomputed forecasts from the forecast store (ADR-0009); it never loads models. See
``docs/api.md`` for endpoints, configuration, caching, rate limits and readiness semantics.

Only ``serve``-extra dependencies may be imported here (no mlflow, no ``registry``/``flows``/
``models`` packages): the API image does not contain them.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

from volforecast.forecast_store import ForecastRepository
from volforecast.logging import configure_logging, get_logger
from volforecast.serving.errors import install_error_handlers
from volforecast.serving.metrics import StoreCollector, StoreView
from volforecast.serving.middleware import (
    REQUEST_ID_HEADER,
    MetricsMiddleware,
    RequestContextMiddleware,
)
from volforecast.serving.ratelimit import TokenBucketLimiter
from volforecast.serving.routes import ops, v1
from volforecast.serving.settings import ApiSettings
from volforecast.serving.snapshot import SnapshotCache

log = get_logger(__name__)

API_VERSION = "1.0.0"

_DESCRIPTION = """\
Daily realized-volatility forecasts for crypto assets. Forecasts are computed in batch by the
`forecast` flow and read from the forecast store; the API never runs a model per request.

All errors use `{"error": {"code", "message", "request_id"}}`. GET responses of `/v1` carry an
`ETag` and `Cache-Control`; send `If-None-Match` to get `304`. `/v1` is rate limited per client IP
(`429` + `Retry-After`).
"""


def create_app(api: ApiSettings | None = None) -> FastAPI:
    """Build the app. ``api`` defaults to ``ApiSettings()`` (environment)."""
    api = api or ApiSettings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        from volforecast.config import Settings

        core = Settings()
        configure_logging(core.log_level, core.log_json)
        # create=False: the API never runs DDL; the flows own the schema.
        repo = ForecastRepository(api.database_url, create=False)
        cache = SnapshotCache(repo, api)
        app.state.repo = repo
        app.state.cache = cache
        app.state.store_collector = StoreCollector(
            lambda: StoreView(
                metrics=cache.snapshot.metrics,
                origins=cache.snapshot.origins,
                ready=cache.ready,
                loaded_at=cache.snapshot.loaded_at,
            )
        )
        task = asyncio.create_task(cache.run(), name="snapshot-reload")
        log.info("api_started", reload_interval_s=api.reload_interval_s)
        try:
            yield
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            repo.engine.dispose()
            log.info("api_stopped")

    middleware: list[Middleware] = []
    if api.cors_origins:
        middleware.append(
            Middleware(
                CORSMiddleware,
                allow_origins=api.cors_origins,
                allow_methods=["GET", "OPTIONS"],
                allow_headers=["If-None-Match", REQUEST_ID_HEADER, "Content-Type"],
                expose_headers=["ETag", "Retry-After", REQUEST_ID_HEADER, "Cache-Control"],
                max_age=600,
            )
        )
    middleware += [Middleware(MetricsMiddleware), Middleware(RequestContextMiddleware)]

    app = FastAPI(
        title="volforecast API",
        version=API_VERSION,
        description=_DESCRIPTION,
        lifespan=lifespan,
        middleware=middleware,
    )
    app.state.api = api
    app.state.limiter = TokenBucketLimiter(api.rate_limit_rps, api.rate_limit_burst)
    install_error_handlers(app)
    app.include_router(v1)
    app.include_router(ops)
    return app


app = create_app()
