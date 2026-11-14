"""Forecast store: SQLAlchemy tables + repository (ADR-0009).

Typical use::

    repo = ForecastRepository(settings.database_url)   # creates missing tables
    repo.upsert_forecasts(rows)
    repo.latest_forecast("BTCUSDT", 1)
"""

from __future__ import annotations

from volforecast.forecast_store.models import (
    BENCHMARK_MODEL_NAME,
    BENCHMARK_VERSION,
    ROLE_BENCHMARK,
    ROLE_CHALLENGER,
    ROLE_CHAMPION,
    ROLE_FALLBACK,
    SERVED_ROLES,
    Base,
    Forecast,
    MonitoringMetric,
    PipelineRun,
    PromotionDecision,
    Realized,
)
from volforecast.forecast_store.repository import (
    ForecastRepository,
    clean_json,
    clean_value,
    init_db,
    make_engine,
    realized_horizon_sum,
    to_date,
)

__all__ = [
    "BENCHMARK_MODEL_NAME",
    "BENCHMARK_VERSION",
    "ROLE_BENCHMARK",
    "ROLE_CHALLENGER",
    "ROLE_CHAMPION",
    "ROLE_FALLBACK",
    "SERVED_ROLES",
    "Base",
    "Forecast",
    "ForecastRepository",
    "MonitoringMetric",
    "PipelineRun",
    "PromotionDecision",
    "Realized",
    "clean_json",
    "clean_value",
    "init_db",
    "make_engine",
    "realized_horizon_sum",
    "to_date",
]
