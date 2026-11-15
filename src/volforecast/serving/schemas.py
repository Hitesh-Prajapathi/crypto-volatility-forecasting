"""Pydantic v2 response models and the error envelope of the ``/v1`` API."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ErrorBody(BaseModel):
    code: str = Field(description="Stable machine-readable error code.", examples=["not_found"])
    message: str = Field(description="Human-readable explanation.")
    request_id: str | None = Field(default=None, description="Echo of the X-Request-ID header.")
    details: list[dict[str, Any]] | None = Field(
        default=None, description="Per-field validation problems (422 only)."
    )


class ErrorResponse(BaseModel):
    """Every non-2xx response of the API has exactly this shape."""

    error: ErrorBody


class ForecastCore(BaseModel):
    """Fields shared by the latest-forecast and the history rows."""

    origin_date: date = Field(description="Last day of the information set (UTC).")
    target_date: date = Field(description="First day the forecast applies to (UTC).")
    forecast_var: float = Field(
        description="Forecast of the sum of squared 5-minute log returns over the horizon."
    )
    daily_vol: float = Field(description="sqrt(forecast_var / horizon): per-day volatility.")
    annualized_vol: float = Field(description="daily_vol * sqrt(365).")
    var95: float | None = Field(
        default=None, description="1-day 95% VaR (positive loss magnitude); horizon 1 only."
    )
    var99: float | None = Field(
        default=None, description="1-day 99% VaR (positive loss magnitude); horizon 1 only."
    )
    es975: float | None = Field(
        default=None,
        description="1-day 97.5% expected shortfall (positive loss magnitude); horizon 1 only.",
    )
    var_dist: str | None = Field(
        default=None, description="Distribution behind VaR/ES: 'fhs' or 'normal'."
    )
    model_name: str
    model_version: str = Field(description="Registry version, 'fallback:<model>' or similar.")
    role: str = Field(description="'champion' or 'fallback'.")
    data_hash: str | None = Field(
        default=None, description="Hash of the data manifest the forecast was computed from."
    )
    generated_at: datetime = Field(description="When the forecast row was written (UTC).")


class ForecastResponse(ForecastCore):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "asset": "BTCUSDT",
                "horizon": 1,
                "origin_date": "2026-01-30",
                "target_date": "2026-01-31",
                "forecast_var": 0.0004,
                "daily_vol": 0.02,
                "annualized_vol": 0.382,
                "var95": 0.031,
                "var99": 0.046,
                "es975": 0.044,
                "var_dist": "fhs",
                "model_name": "log_har",
                "model_version": "3",
                "role": "champion",
                "data_hash": "sha256:ab12...",
                "generated_at": "2026-01-31T00:35:12Z",
            }
        }
    )

    asset: str
    horizon: int


class HistoryItem(ForecastCore):
    realized_var: float | None = Field(
        default=None,
        description="Realized variance over the target days; null until all are known.",
    )
    realized_ret: float | None = Field(
        default=None, description="Log return of the day after the origin; null until known."
    )


class HistoryResponse(BaseModel):
    asset: str
    horizon: int
    start: date | None = None
    end: date | None = None
    limit: int
    offset: int
    total: int = Field(description="Rows matching the filter (all pages).")
    next_offset: int | None = Field(
        default=None, description="Offset of the next page; null on the last page."
    )
    items: list[HistoryItem] = Field(description="Newest origin date first.")


class ModelRef(BaseModel):
    model_name: str
    model_version: str


class ChampionInfo(ModelRef):
    role: str = Field(description="'champion' or 'fallback' (no champion registered).")
    latest_origin_date: date
    n_forecasts: int = Field(description="Served forecasts stored for this asset and horizon.")


class PromotionInfo(BaseModel):
    decided_at: datetime
    as_of_date: date
    promoted: bool
    challenger_version: str
    challenger_model: str
    champion_version: str | None = None
    reasons: list[str]
    metrics: dict[str, Any]
    mlflow_run_id: str | None = None


class ModelEntry(BaseModel):
    asset: str
    horizon: int
    champion: ChampionInfo
    challenger: ModelRef | None = Field(
        default=None, description="Shadow challenger forecasting next to the champion, if any."
    )
    latest_promotion: PromotionInfo | None = Field(
        default=None, description="Most recent champion/challenger gate decision."
    )


class ModelsResponse(BaseModel):
    models: list[ModelEntry]


class BacktestSummary(BaseModel):
    """The official walk-forward results (``results.json``); nested parts are left untyped."""

    model_config = ConfigDict(extra="allow")

    meta: dict[str, Any] = Field(description="Window, models, seeds, code and data hashes.")
    assets: dict[str, Any] = Field(description="Per-asset, per-horizon metrics.")


class HealthResponse(BaseModel):
    status: str = "ok"


class ReadyResponse(BaseModel):
    status: str = Field(description="'ready' or 'not_ready'.")
    database: bool = Field(description="The last store reload succeeded.")
    champion_forecasts: int = Field(description="Asset x horizon pairs with a loadable forecast.")
    snapshot_age_seconds: float | None = Field(
        default=None, description="Seconds since the last successful reload."
    )
    detail: str | None = None
