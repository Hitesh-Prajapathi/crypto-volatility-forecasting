"""Prometheus metrics for the pipeline and the model (platform contract names).

Metrics live in a dedicated ``CollectorRegistry`` (``METRICS.registry``) so repeated imports and
tests never hit duplicate-registration errors; the API serves ``generate_latest(METRICS.registry)``
next to its own HTTP metrics. :func:`export_from_store` refreshes everything from the database
(the DB is the source of truth), :func:`push_job_metrics` pushes the same numbers to a
Pushgateway for short-lived jobs when ``VOLF_PUSHGATEWAY_URL`` is set.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from datetime import UTC, datetime, time
from typing import Any

import pandas as pd
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
from prometheus_client.core import CounterMetricFamily, Metric
from prometheus_client.registry import Collector

from volforecast.forecast_store import ForecastRepository
from volforecast.logging import get_logger

log = get_logger(__name__)

PUSHGATEWAY_ENV = "VOLF_PUSHGATEWAY_URL"


class _BreachCollector(Collector):
    """``volforecast_var_breaches_total{asset,level}``: a real counter whose value comes from the
    store (a prometheus Counter cannot be set to a value)."""

    def __init__(self) -> None:
        self.values: dict[tuple[str, str], float] = {}

    def collect(self) -> Iterable[Metric]:
        fam = CounterMetricFamily(
            "volforecast_var_breaches",
            "Cumulative VaR breaches of the served forecasts",
            labels=["asset", "level"],
        )
        for (asset, level), v in sorted(self.values.items()):
            fam.add_metric([asset, level], v)
        yield fam


class PipelineMetrics:
    """All pipeline/model metrics registered in one registry."""

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry()
        r = self.registry
        self.model_info = Gauge(
            "volforecast_model_info",
            "Served model version per asset and horizon (value 1)",
            ["asset", "horizon", "version"],
            registry=r,
        )
        self.forecast_var = Gauge(
            "volforecast_forecast_var",
            "Latest served h-day variance forecast",
            ["asset", "horizon"],
            registry=r,
        )
        self.realized_var = Gauge(
            "volforecast_realized_var", "Latest realized daily variance", ["asset"], registry=r
        )
        self.rolling_qlike = Gauge(
            "volforecast_rolling_qlike",
            "Rolling mean QLIKE of live forecasts",
            ["asset", "horizon", "model", "window"],
            registry=r,
        )
        self.qlike_ratio = Gauge(
            "volforecast_qlike_ratio_vs_har",
            "Rolling QLIKE of the champion divided by that of HAR",
            ["asset", "horizon"],
            registry=r,
        )
        self.breaches = _BreachCollector()
        r.register(self.breaches)
        self.last_bar = Gauge(
            "volforecast_data_last_bar_timestamp_seconds",
            "Open time (unix seconds) of the newest ingested 5-minute bar",
            ["asset"],
            registry=r,
        )
        self.last_success = Gauge(
            "volforecast_pipeline_last_success_timestamp_seconds",
            "Unix time of the last successful run of a flow",
            ["flow"],
            registry=r,
        )
        self.drift_share = Gauge(
            "volforecast_drift_share",
            "Share of drifted features (PSI > 0.2)",
            ["asset"],
            registry=r,
        )


#: process-wide metrics used by the API's /metrics endpoint
METRICS = PipelineMetrics()


def _latest(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    return df.sort_values("as_of_date").groupby(keys, as_index=False).tail(1)


def export_from_store(repo: ForecastRepository, metrics: PipelineMetrics = METRICS) -> None:
    """Refresh every gauge from the database (stale label sets are removed)."""
    for g in (
        metrics.model_info,
        metrics.forecast_var,
        metrics.realized_var,
        metrics.rolling_qlike,
        metrics.qlike_ratio,
        metrics.last_bar,
        metrics.last_success,
        metrics.drift_share,
    ):
        g.clear()
    for row in repo.models_summary():
        asset, h = str(row["asset"]), str(row["horizon"])
        metrics.model_info.labels(asset, h, str(row["model_version"])).set(1)
        latest = repo.latest_forecast(asset, int(h))
        if latest is not None:
            metrics.forecast_var.labels(asset, h).set(float(latest["forecast_var"]))
    assets = sorted({str(r["asset"]) for r in repo.models_summary()})
    for asset in assets:
        real = repo.get_realized(asset)
        real = real[real["complete"]]
        if not real.empty:
            metrics.realized_var.labels(asset).set(float(real["rv"].iloc[-1]))

    q = _latest(repo.get_metrics("rolling_qlike"), ["asset", "horizon", "model", "window"])
    for r in q.to_dict("records"):
        metrics.rolling_qlike.labels(
            r["asset"], str(r["horizon"]), r["model"], str(r["window"])
        ).set(r["value"])
    ratio = _latest(repo.get_metrics("qlike_ratio_vs_har"), ["asset", "horizon"])
    for r in ratio.to_dict("records"):
        metrics.qlike_ratio.labels(r["asset"], str(r["horizon"])).set(r["value"])
    for level in ("95", "99"):
        hits = repo.get_metrics(f"var{level}_hit")
        for name, grp in hits.groupby("asset"):
            metrics.breaches.values[(str(name), level)] = float(grp["value"].sum())
    bar = _latest(repo.get_metrics("data_last_bar_ts"), ["asset"])
    for r in bar.to_dict("records"):
        metrics.last_bar.labels(r["asset"]).set(r["value"])
    drift_df = _latest(repo.get_metrics("drift_share"), ["asset"])
    for r in drift_df.to_dict("records"):
        metrics.drift_share.labels(r["asset"]).set(r["value"])
    for flow, ts in repo.last_success().items():
        metrics.last_success.labels(flow).set(ts.timestamp())


def last_bar_timestamp(last_day: Any) -> float:
    """Open time (unix s) of the last 5-minute bar of UTC day ``last_day`` (23:55)."""
    d = pd.Timestamp(last_day).date()
    return datetime.combine(d, time(23, 55), tzinfo=UTC).timestamp()


def push_job_metrics(flow: str, repo: ForecastRepository | None = None) -> bool:
    """Push the current metrics (with ``flow`` marked successful now) to the Pushgateway.

    No-op (returns False) when ``VOLF_PUSHGATEWAY_URL`` is unset; push failures are logged and
    never fail the job.
    """
    url = os.environ.get(PUSHGATEWAY_ENV, "").strip()
    if not url:
        return False
    reg = CollectorRegistry()
    metrics = PipelineMetrics(reg)
    try:
        if repo is not None:
            export_from_store(repo, metrics)
        metrics.last_success.labels(flow).set(datetime.now(UTC).timestamp())
        push_to_gateway(url, job=f"volforecast_{flow}", registry=reg)
    except Exception as exc:
        log.warning("pushgateway_failed", url=url, err=repr(exc))
        return False
    return True
