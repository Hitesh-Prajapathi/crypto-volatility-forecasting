"""API settings: ``VOLF_API_*`` environment variables (plus ``VOLF_RESULTS_PATH``).

Paths, the database URL and logging come from the shared :class:`volforecast.config.Settings`
(``VOLF_DATABASE_URL``, ``VOLF_DATA_DIR``, ``VOLF_LOG_JSON``...); this class only adds what the
API needs. See ``docs/api.md`` for the full table.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from volforecast.config import Settings


class ApiSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VOLF_API_", env_file=".env", extra="ignore", populate_by_name=True
    )

    #: SQLAlchemy URL of the forecast store; empty -> ``VOLF_DATABASE_URL`` / the core default.
    database_url: str = ""
    #: Official backtest summary (``reports/results.json``); empty -> ``<reports_dir>/results.json``.
    results_path: Path | None = Field(
        default=None, validation_alias=AliasChoices("VOLF_RESULTS_PATH", "VOLF_API_RESULTS_PATH")
    )
    #: Seconds between snapshot reloads (forecasts, models, pipeline gauges).
    reload_interval_s: float = Field(default=60.0, gt=0)
    #: Retry interval while the snapshot is not ready (DB down / no champion forecast yet).
    not_ready_retry_s: float = Field(default=5.0, gt=0)
    #: ``Cache-Control: max-age`` of cacheable GET responses.
    cache_max_age_s: int = Field(default=60, ge=0)
    #: Per-client-IP token bucket: sustained requests/second (0 disables) and burst size.
    #: The bucket lives in each worker process, so the effective limit is WEB_CONCURRENCY times
    #: higher in the worst case.
    rate_limit_rps: float = Field(default=20.0, ge=0)
    rate_limit_burst: int = Field(default=40, ge=1)
    #: Number of trusted reverse proxies appending to ``X-Forwarded-For`` (0 = ignore the header).
    trusted_proxy_hops: int = Field(default=0, ge=0)
    #: Allowed CORS origins (comma separated or JSON list). Empty = CORS disabled.
    cors_origins: Annotated[list[str], NoDecode] = Field(default_factory=list)
    #: Count a ``fallback`` forecast as "champion loadable" for ``/readyz``.
    ready_allow_fallback: bool = False
    #: Max rows per history page.
    max_history_limit: int = Field(default=1000, ge=1)

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, v: Any) -> Any:
        if isinstance(v, str):
            text = v.strip()
            if text.startswith("["):
                return json.loads(text)
            return [p.strip() for p in text.split(",") if p.strip()]
        return v

    @model_validator(mode="after")
    def _fill_defaults(self) -> ApiSettings:
        core = Settings()
        if not self.database_url:
            self.database_url = core.database_url
        if self.results_path is None:
            self.results_path = core.reports_dir / "results.json"
        return self

    @property
    def results_file(self) -> Path:
        assert self.results_path is not None  # filled by the validator
        return self.results_path
