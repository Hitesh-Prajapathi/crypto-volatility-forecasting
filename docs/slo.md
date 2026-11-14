# Service level objectives

Metric names are those used in `deploy/monitoring/alerts/volforecast-rules.yaml` and
`docs/platform.md`. The SLO alerts below are proposals; they are not yet in the rules file. Targets
are design targets, not measurements; reassess after the first month of real traffic.

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

**SLI**: a day is good if the `forecast` flow succeeded after 00:00 and before 01:00 UTC. The metric
`volforecast_pipeline_last_success_timestamp_seconds{flow="forecast"}` holds only the latest
success, so record a daily sample at 01:00 UTC (recording rule evaluated then, or a
scheduled query):

```promql
# volforecast:forecast_fresh_at_0100 : 1 if today's (UTC) forecast run has succeeded, else 0
clamp_max(
  (max(volforecast_pipeline_last_success_timestamp_seconds{flow="forecast"}) > bool floor(time() / 86400) * 86400),
  1)
```

SLI over the window: `avg_over_time(volforecast:forecast_fresh_at_0100[30d])`, good if >= 0.99.
Supporting data check: `time() - max by (asset) (volforecast_data_last_bar_timestamp_seconds) < 2 * 3600`
at the same moment.

**Error budget**: 1% of days, i.e. 0.3 days per 30. In practice one missed day already breaches the
30-day target (1/30 = 3.3%), so treat it as a count: the budget is 1 missed day per 100. Track over
a 90-day window for a budget of 0.9 days, and report the number of misses.

**Burn-rate alert proposal** (daily samples, so only slow windows are meaningful):

| Severity | Condition |
|----------|-----------|
| page | two consecutive misses: `sum_over_time((1 - volforecast:forecast_fresh_at_0100)[2d:1d]) >= 2` |
| ticket | any miss in the last 7 days: `min_over_time(volforecast:forecast_fresh_at_0100[7d]) == 0` |
| ticket | 90-day SLI below target: `avg_over_time(volforecast:forecast_fresh_at_0100[90d]) < 0.99` |

Also consider an early `ForecastLate` alert that fires at 01:05 UTC when today's sample is 0, since
`PipelineFailed` and `DataStale` only fire after 26 hours.

**When the budget is exhausted**: pause changes to the ingest and forecast flows, their schedules and
the data source, except fixes; review the timing slack (start earlier, tune retries) and consider a
documented fallback (serve the previous forecast, flagged stale).

## Dashboard

Add an error-budget panel to the Grafana Service dashboard from the SLI queries above.
