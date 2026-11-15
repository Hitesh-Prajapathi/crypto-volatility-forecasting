"""Prometheus exposition of the API: HTTP metrics + store-derived pipeline/model gauges.

* HTTP metrics (``http_requests_total``, ``http_request_duration_seconds``) are real
  ``prometheus_client`` metrics. With several workers the container entrypoint sets
  ``PROMETHEUS_MULTIPROC_DIR`` (under ``/tmp``) *before* Python starts, and ``/metrics`` aggregates
  the per-process files with ``MultiProcessCollector``.
* The pipeline/model gauges come from the forecast store. ``monitoring.prom.export_from_store``
  does the querying, but it writes into ``Gauge`` objects, which in multiprocess mode would leave
  file-backed series behind (and ``Gauge.clear()`` is not supported there). So the reload loop
  hands it a :class:`RecordingMetrics`, a drop-in that records plain dicts, and the scrape turns
  the latest recording into metric families. ``tests/unit/serving/test_metrics.py`` pins the
  metric names against :class:`volforecast.monitoring.prom.PipelineMetrics`, so they cannot drift.
* Scrape-time derived gauges (age of the last successful run per flow, age of the forecast per
  asset x horizon) are API additions: ``platform.md`` has timestamps only.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, time as dtime

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily, Metric
from prometheus_client.registry import Collector

MULTIPROC_ENV = "PROMETHEUS_MULTIPROC_DIR"
MULTIPROCESS = bool(os.environ.get(MULTIPROC_ENV))

#: Registry that holds the HTTP metrics in single-process mode.
HTTP_REGISTRY = CollectorRegistry(auto_describe=True)
_METRIC_REGISTRY = None if MULTIPROCESS else HTTP_REGISTRY

HTTP_REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds.",
    ["route", "method", "status"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 2.5, 5.0),
    registry=_METRIC_REGISTRY,
)
HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests.",
    ["route", "method", "status"],
    registry=_METRIC_REGISTRY,
)

UNMATCHED_ROUTE = "unmatched"


def observe_request(route: str, method: str, status: int, seconds: float) -> None:
    labels = {"route": route, "method": method, "status": str(status)}
    HTTP_REQUEST_DURATION.labels(**labels).observe(seconds)
    HTTP_REQUESTS_TOTAL.labels(**labels).inc()


# --------------------------------------------------------------------------- store gauges
class _Child:
    def __init__(self, gauge: RecordedGauge, key: tuple[str, ...]) -> None:
        self._gauge = gauge
        self._key = key

    def set(self, value: float) -> None:
        self._gauge.samples[self._key] = float(value)


class RecordedGauge:
    """Minimal stand-in for ``Gauge`` that keeps its samples in a dict."""

    def __init__(self, name: str, doc: str, labelnames: tuple[str, ...]) -> None:
        self.name = name
        self.doc = doc
        self.labelnames = labelnames
        self.samples: dict[tuple[str, ...], float] = {}

    def clear(self) -> None:
        self.samples.clear()

    def labels(self, *values: str) -> _Child:
        return _Child(self, tuple(str(v) for v in values))

    def family(self) -> Metric:
        fam = GaugeMetricFamily(self.name, self.doc, labels=list(self.labelnames))
        for key, v in sorted(self.samples.items()):
            fam.add_metric(list(key), v)
        return fam


class _Breaches:
    def __init__(self) -> None:
        self.values: dict[tuple[str, str], float] = {}

    def family(self) -> Metric:
        fam = CounterMetricFamily(
            "volforecast_var_breaches",
            "Cumulative VaR breaches of the served forecasts",
            labels=["asset", "level"],
        )
        for (asset, level), v in sorted(self.values.items()):
            fam.add_metric([asset, level], v)
        return fam


class RecordingMetrics:
    """Same attributes as ``monitoring.prom.PipelineMetrics``, backed by dicts."""

    def __init__(self) -> None:
        self.model_info = RecordedGauge(
            "volforecast_model_info",
            "Served model version per asset and horizon (value 1)",
            ("asset", "horizon", "version"),
        )
        self.forecast_var = RecordedGauge(
            "volforecast_forecast_var",
            "Latest served h-day variance forecast",
            ("asset", "horizon"),
        )
        self.realized_var = RecordedGauge(
            "volforecast_realized_var", "Latest realized daily variance", ("asset",)
        )
        self.rolling_qlike = RecordedGauge(
            "volforecast_rolling_qlike",
            "Rolling mean QLIKE of live forecasts",
            ("asset", "horizon", "model", "window"),
        )
        self.qlike_ratio = RecordedGauge(
            "volforecast_qlike_ratio_vs_har",
            "Rolling QLIKE of the champion divided by that of HAR",
            ("asset", "horizon"),
        )
        self.breaches = _Breaches()
        self.last_bar = RecordedGauge(
            "volforecast_data_last_bar_timestamp_seconds",
            "Open time (unix seconds) of the newest ingested 5-minute bar",
            ("asset",),
        )
        self.last_success = RecordedGauge(
            "volforecast_pipeline_last_success_timestamp_seconds",
            "Unix time of the last successful run of a flow",
            ("flow",),
        )
        self.drift_share = RecordedGauge(
            "volforecast_drift_share",
            "Share of drifted features (PSI and Wasserstein criteria, see monitoring.drift)",
            ("asset",),
        )

    def families(self) -> list[Metric]:
        gauges = (
            self.model_info,
            self.forecast_var,
            self.realized_var,
            self.rolling_qlike,
            self.qlike_ratio,
            self.last_bar,
            self.last_success,
            self.drift_share,
        )
        return [g.family() for g in gauges] + [self.breaches.family()]


def origin_end_ts(origin: date) -> float:
    """Unix time at which the information set of ``origin`` (a UTC day) closed."""
    day_end = datetime.combine(origin, dtime(0, 0), tzinfo=UTC).timestamp() + 86400.0
    return day_end


class StoreCollector(Collector):
    """Pipeline/model gauges of the current snapshot, plus scrape-time ages."""

    def __init__(self, get_state: Callable[[], StoreView], clock: Callable[[], float] = time.time):
        self._get_state = get_state
        self._clock = clock

    def collect(self) -> Iterable[Metric]:
        view = self._get_state()
        now = self._clock()
        yield from view.metrics.families()

        flow_age = GaugeMetricFamily(
            "volforecast_pipeline_last_success_age_seconds",
            "Seconds since the last successful run of a flow (derived at scrape time)",
            labels=["flow"],
        )
        for (flow,), ts in sorted(view.metrics.last_success.samples.items()):
            flow_age.add_metric([flow], max(0.0, now - ts))
        yield flow_age

        fc_age = GaugeMetricFamily(
            "volforecast_forecast_age_seconds",
            "Seconds since the information set of the served forecast closed "
            "(origin day end, UTC); forecast freshness per asset and horizon",
            labels=["asset", "horizon"],
        )
        for (asset, horizon), origin in sorted(view.origins.items()):
            fc_age.add_metric([asset, str(horizon)], max(0.0, now - origin_end_ts(origin)))
        yield fc_age

        ready = GaugeMetricFamily(
            "volforecast_api_ready", "1 when the API would answer /readyz with 200"
        )
        ready.add_metric([], 1.0 if view.ready else 0.0)
        yield ready
        snap_age = GaugeMetricFamily(
            "volforecast_api_snapshot_age_seconds", "Seconds since the last successful reload"
        )
        if view.loaded_at is not None:
            snap_age.add_metric([], max(0.0, now - view.loaded_at))
        yield snap_age


@dataclass(frozen=True)
class StoreView:
    """What :class:`StoreCollector` needs from the cache (kept free of import cycles)."""

    metrics: RecordingMetrics
    origins: dict[tuple[str, int], date]
    ready: bool
    loaded_at: float | None


def render_metrics(collector: StoreCollector) -> tuple[bytes, str]:
    """Exposition text of the HTTP metrics (all workers) plus the store gauges."""
    http: CollectorRegistry
    if MULTIPROCESS:
        from prometheus_client import multiprocess

        http = CollectorRegistry()
        multiprocess.MultiProcessCollector(http)
    else:
        http = HTTP_REGISTRY
    store = CollectorRegistry()
    store.register(collector)
    body = generate_latest(http) + generate_latest(store)
    return body, CONTENT_TYPE_LATEST
