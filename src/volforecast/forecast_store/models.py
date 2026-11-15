"""SQLAlchemy 2.0 table definitions of the forecast store (ADR-0009).

Schema management: ``Base.metadata.create_all`` (see :func:`volforecast.forecast_store.init_db`).
It only creates missing tables and never alters existing ones; additive changes need a manual
``ALTER TABLE`` until Alembic migrations are introduced (future work).

Conventions: dates are calendar days (UTC), variances are sums of squared 5-minute log returns
over the h target days, VaR/ES are positive loss magnitudes (1-day, h=1 rows only).
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

#: ``forecasts.role`` values.
ROLE_CHAMPION = "champion"  # served: the registry @champion model
ROLE_FALLBACK = "fallback"  # served: config model fit on full history (no champion exists)
ROLE_CHALLENGER = "challenger"  # shadow: the registry @challenger model, never served
ROLE_BENCHMARK = "benchmark"  # HAR benchmark used by evaluate (model_name "har_benchmark")
SERVED_ROLES: tuple[str, ...] = (ROLE_CHAMPION, ROLE_FALLBACK)

BENCHMARK_MODEL_NAME = "har_benchmark"
BENCHMARK_VERSION = "har_benchmark"


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class Forecast(Base):
    """One h-day variance forecast (+ 1-day VaR/ES for h=1) made at the end of ``origin_date``.

    Unique on ``(asset, horizon, model_version, origin_date)`` so re-running a day is an
    idempotent upsert. ``model_version`` is the registry version number ("3"),
    ``"fallback:<model>"`` or ``"har_benchmark"``; ``role`` says how the row is used.
    """

    __tablename__ = "forecasts"
    __table_args__ = (
        UniqueConstraint(
            "asset", "horizon", "model_version", "origin_date", name="uq_forecast_key"
        ),
        Index("ix_forecasts_lookup", "asset", "horizon", "role", "origin_date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    asset: Mapped[str] = mapped_column(String(32))
    horizon: Mapped[int] = mapped_column(Integer)
    model_name: Mapped[str] = mapped_column(String(64))
    model_version: Mapped[str] = mapped_column(String(64))
    role: Mapped[str] = mapped_column(String(16), default=ROLE_CHAMPION)
    origin_date: Mapped[date] = mapped_column(Date)
    target_date: Mapped[date] = mapped_column(Date)
    forecast_var: Mapped[float] = mapped_column(Float)
    var95: Mapped[float | None] = mapped_column(Float, nullable=True)
    var99: Mapped[float | None] = mapped_column(Float, nullable=True)
    es975: Mapped[float | None] = mapped_column(Float, nullable=True)
    dist: Mapped[str | None] = mapped_column(String(16), nullable=True)  # VaR distribution used
    data_hash: Mapped[str | None] = mapped_column(String(80), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Realized(Base):
    """Daily realized measures per asset (the drift reference and the scoring target)."""

    __tablename__ = "realized"

    asset: Mapped[str] = mapped_column(String(32), primary_key=True)
    date: Mapped[date] = mapped_column(Date, primary_key=True)
    rv: Mapped[float | None] = mapped_column(Float, nullable=True)
    bpv: Mapped[float | None] = mapped_column(Float, nullable=True)
    rs_pos: Mapped[float | None] = mapped_column(Float, nullable=True)
    rs_neg: Mapped[float | None] = mapped_column(Float, nullable=True)
    rq: Mapped[float | None] = mapped_column(Float, nullable=True)
    ret: Mapped[float | None] = mapped_column(Float, nullable=True)
    complete: Mapped[bool] = mapped_column(Boolean, default=True)


class PipelineRun(Base):
    """One execution of a flow (``status``: running / success / failed / skipped)."""

    __tablename__ = "pipeline_runs"
    __table_args__ = (Index("ix_pipeline_runs_flow", "flow", "status", "finished_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    flow: Mapped[str] = mapped_column(String(32))
    run_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="running")
    detail: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)


class PromotionDecision(Base):
    """Audit row of one champion/challenger gate evaluation (also logged to MLflow)."""

    __tablename__ = "promotion_decisions"
    __table_args__ = (Index("ix_promotion_lookup", "asset", "horizon", "as_of_date"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    asset: Mapped[str] = mapped_column(String(32))
    horizon: Mapped[int] = mapped_column(Integer)
    as_of_date: Mapped[date] = mapped_column(Date)  # last origin of the evaluation window
    registered_model: Mapped[str] = mapped_column(String(96))
    challenger_version: Mapped[str] = mapped_column(String(32))
    challenger_model: Mapped[str] = mapped_column(String(64))
    champion_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    promoted: Mapped[bool] = mapped_column(Boolean)
    reasons: Mapped[list[str]] = mapped_column(JSON)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON)
    mlflow_run_id: Mapped[str | None] = mapped_column(String(64), nullable=True)


class MonitoringMetric(Base):
    """Time series of monitoring numbers written by ``evaluate`` (and ``ingest``/``forecast``).

    Unique on ``(asset, horizon, model, metric, window, as_of_date)``; the "not applicable"
    values of ``horizon`` / ``model`` / ``window`` are ``0`` / ``""`` / ``0`` (never NULL, so the
    key behaves identically on sqlite and Postgres).

    Metrics in use: ``rolling_qlike`` (model = champion|har_benchmark, window = 30),
    ``qlike_ratio_vs_har`` (window = 30), ``var99_hit`` (0/1 per target day: the breach log),
    ``drift_share`` (share of drifted features), ``data_last_bar_ts`` (epoch seconds).
    """

    __tablename__ = "monitoring_metrics"
    __table_args__ = (
        UniqueConstraint(
            "asset", "horizon", "model", "metric", "window", "as_of_date", name="uq_metric_key"
        ),
        Index("ix_metrics_lookup", "metric", "asset", "as_of_date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    asset: Mapped[str] = mapped_column(String(32))
    horizon: Mapped[int] = mapped_column(Integer, default=0)
    model: Mapped[str] = mapped_column(String(64), default="")
    metric: Mapped[str] = mapped_column(String(48))
    window: Mapped[int] = mapped_column(Integer, default=0)
    as_of_date: Mapped[date] = mapped_column(Date)
    value: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
