# Service level objectives

Metric names are those used in `deploy/helm/volforecast/rules/volforecast-rules.yaml` (the single
rules file of the compose stack and the Helm chart) and `docs/platform.md`. The SLO 1 alerts below
are proposals; the SLO 2 (freshness) rules are implemented in that file and unit-tested. Targets are
design targets, not measurements; reassess after the first month of real traffic.

## SLO 1: availability of `GET /v1/forecast`

Target: 99.5% of requests succeed over a rolling 30 days. Good = not 5xx (4xx are client errors).

**SLI**

```promql
1 - (
  sum(increase(http_requests_total{route="/v1/forecast", status=~"5.."}[30d]))
  /
  clamp_min(sum(increase(http_requests_total{route="/v1/forecast"}[30d])), 1)
)
```

**Error budget**: 0.5% of requests, 5 failed per 1,000. As time, about 3.6 hours of total outage
per 30 days. Budget remaining = `1 - error_ratio_30d / 0.005`.

**Burn-rate alerts (multiwindow, multi-burn-rate).** Let
`err(w) = sum(rate(http_requests_total{route="/v1/forecast",status=~"5.."}[w])) / clamp_min(sum(rate(http_requests_total{route="/v1/forecast"}[w])), 1e-9)`.

| Severity | Burn rate | Long / short window | Condition | Budget spent |
|----------|-----------|---------------------|-----------|--------------|
| page | 14.4 | 1h / 5m | `err(1h) > 14.4 * 0.005 and err(5m) > 14.4 * 0.005` | 2% in 1h |
| page | 6 | 6h / 30m | `err(6h) > 6 * 0.005 and err(30m) > 6 * 0.005` | 5% in 6h |
| ticket | 1 | 3d / 6h | `err(3d) > 0.005 and err(6h) > 0.005` | 10% in 3d |

The short window makes the alert stop soon after the incident ends. The existing `APIErrorRate`
(5xx above 1% for 10m, all routes) remains as a fast symptom alert.

**When the budget is exhausted**: freeze feature releases and model promotions until the 30-day
ratio recovers; only reliability fixes and rollbacks ship ([runbook](runbook.md) section 4). Write a
short post-incident review naming the dominant failure mode and prioritize work against it.

## SLO 2: forecast freshness

Target: the forecast for the next day (t+1, made from the UTC day t that just closed) is available by
01:00 UTC on 99% of days over a rolling 30 days.

Schedule: ingest 00:20 and forecast 00:35 UTC (CronJobs in `values.yaml`), leaving 25 minutes of
slack.

**SLI**: a day is good if the forecast for t+1 is being served at 01:00 UTC. The metric is
`volforecast_forecast_age_seconds{horizon="1"}`, the age of the end of the served forecast's origin
day (UTC). The forecast made from the day that closes at 00:00 is about 1 h old at 01:00 and a missed
run leaves the previous one, at least 24 h old, so "fresh at 01:00" is `age < 86400`. The API
publishes the gauge from its store snapshot (reload every 60 s), so the SLI measures what clients
get, not just that the flow ran. (`volforecast_pipeline_last_success_timestamp_seconds` was
considered and rejected: it says the flow succeeded, not that the API serves the result.)

Rules (group `volforecast.slo`, evaluated every minute; `deploy/helm/volforecast/rules/volforecast-rules.yaml`):

| Rule | Meaning |
|------|---------|
| `volforecast:forecast_fresh_at_0100:bool` | per asset, 1 = fresh at 01:00 UTC, 0 = missed. Exactly one sample per day (the 01:00 minute: `and on() (hour() == 1 and minute() == 0)`); the series is empty the rest of the day, so every query on it must use `*_over_time` |
| `volforecast:forecast_fresh_at_0100:ratio_30d` | SLI: `avg_over_time(...:bool[30d])`, good if >= 0.99 |
| `volforecast:forecast_fresh_at_0100:misses_30d` | missed days in the window (`count_over_time - sum_over_time`) |

A missing series counts as a miss, not as a missing sample: an asset whose gauge vanished during the
last day gets 0, and with no series at all (API down) one label-less 0 is recorded. Consequence: an
outage of the API at 01:00 burns budget even though the flow may have run. The flip side is that the
SLI cannot distinguish a failed flow from an unreachable API; use `PipelineFailed` / `DataStale`
and `volforecast_api_ready` to tell them apart. Supporting data check at the same moment:
`time() - max by (asset) (volforecast_data_last_bar_timestamp_seconds) < 2 * 3600`.

The 30-day window needs 30 days of Prometheus retention. The compose stack keeps 15 days
(`--storage.tsdb.retention.time=15d`), so there the ratio covers the retained days only.

**Error budget**: 1% of days, i.e. 0.3 days per 30. In practice one missed day already breaches the
30-day target (1/30 = 3.3%), so treat it as a count: the budget is 1 missed day per 100. A single
miss therefore spends more than the whole 30-day budget (3.3x) and keeps `ratio_30d` below target for
30 days.

**Alerts** (daily samples, so only slow windows are meaningful):

| Alert | Severity | Condition |
|-------|----------|-----------|
| `ForecastLate` | warning | the served forecast is older than 25 h 5 min (still the previous day's at 01:05 UTC); clears when the new forecast is served |
| `ForecastFreshnessConsecutiveMisses` | critical (page) | two consecutive daily samples at 0 for any asset: `min(max_over_time(...:bool[23h])) == 0 and min(max_over_time(...:bool[23h] offset 1d)) == 0` (each 23 h window holds only that day's 01:00 sample); active from the second miss until midnight |
| `ForecastFreshnessBudgetBurn` | warning (ticket) | `ratio_30d < 0.99`: error budget exhausted |

`PipelineFailed` and `DataStale` (26 h) are unchanged and remain the slow backstops. The three
alerts and the recording rules are tested with `promtool test rules`
(`deploy/monitoring/alerts/tests/freshness-slo.yaml`): good day, missed days, no sample outside
01:00, vanished series, total outage, and each alert firing and clearing.
The consecutive-miss alert pairs "any asset today" with "any asset yesterday" (aggregated with
`min`), so two different assets missing on consecutive days also pages.

**When the budget is exhausted**: pause changes to the ingest and forecast flows, their schedules and
the data source, except fixes; review the timing slack (start earlier, tune retries) and consider a
documented fallback (serve the previous forecast, flagged stale).

## Dashboard

The Grafana Service dashboard (`deploy/monitoring/grafana/dashboards/service.json`) has two stat
panels for SLO 2 (30-day SLI against the 99% target and the number of missed days). An SLO 1
error-budget panel is still to be added from the SLI queries above.
