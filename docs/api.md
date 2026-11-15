# API reference

FastAPI service `volforecast.serving.app:app` (port 8000). It serves **precomputed** forecasts from the
forecast store ([ADR-0009](adr/0009-batch-inference-forecast-store.md)); it never loads a model or
computes features per request. Interactive docs: `GET /docs` (Swagger UI), schema: `GET /openapi.json`.

Run locally (after the `forecast` flow has filled the store):

```bash
uv run python -m volforecast.serving --port 8000         # WEB_CONCURRENCY=2 for two workers
VOLF_DATABASE_URL=sqlite:///artifacts/volforecast.db uv run uvicorn volforecast.serving.app:app
```

Only `serve`-extra dependencies are imported (no mlflow, prefect, torch, `registry`, `flows`,
`models`); a unit test pins this because the API image does not contain them.

## 1. Endpoints

| Endpoint | Purpose | Cached (ETag) | Rate limited |
|----------|---------|---------------|--------------|
| `GET /v1/forecast?asset=BTCUSDT&horizon=1` | Latest served forecast | yes | yes |
| `GET /v1/forecast/history?asset=&horizon=&start=&end=&limit=&offset=` | Paged history joined with realized values | yes | yes |
| `GET /v1/models` | Champion / challenger per asset x horizon + latest promotion decision | yes | yes |
| `GET /v1/backtest/summary` | Official walk-forward results (`results.json`) | yes | yes |
| `GET /healthz` | Liveness: always 200, touches nothing | no | no |
| `GET /readyz` | Readiness (see section 5) | no | no |
| `GET /metrics` | Prometheus exposition | no | no |

### `GET /v1/forecast`

Query: `asset` (default `BTCUSDT`, `[A-Za-z0-9]{2,20}`, case-insensitive), `horizon` (default `1`,
integer 1..60). `forecast_var` is the forecast of the **sum** of squared 5-minute log returns over the
`horizon` days (the store's convention). Derived: `daily_vol = sqrt(forecast_var / horizon)`,
`annualized_vol = daily_vol * sqrt(365)`. `var95`, `var99`, `es975` are 1-day positive loss
magnitudes and are `null` for `horizon != 1`. `var_dist` is `fhs` or `normal` (the latter is the
unscaled fallback used before 250 residuals exist; see `docs/mlops.md`).

```bash
$ curl -s 'localhost:8000/v1/forecast?asset=BTCUSDT&horizon=1'
{"annualized_vol":0.5305,"asset":"BTCUSDT","daily_vol":0.02777,"data_hash":"3100d754...","es975":0.0649,
 "forecast_var":0.000771,"generated_at":"2026-11-15T08:19:36.602060Z","horizon":1,"model_name":"log_har",
 "model_version":"7","origin_date":"2026-11-13","role":"champion","target_date":"2026-11-14",
 "var95":0.0457,"var99":0.0646,"var_dist":"normal"}
```

* `404 unknown_asset`: no served forecast exists for that asset. `404 no_data`: asset known, but not at that horizon.
* `422 validation_error`: bad `asset` / `horizon` (also `limit`, `offset`, dates on the history route).
* `503 not_loaded`: the first store load has not succeeded yet (the API also answers `/readyz` 503).

### `GET /v1/forecast/history`

Served forecasts (champion/fallback; shadow and benchmark rows excluded), **newest origin first**, one
row per origin date, with `realized_var` (sum of realized variance over the target days) and
`realized_ret` (log return of the day after the origin). Both are `null` until the outcome is known and
complete. `start` / `end` filter on the origin date (inclusive, `YYYY-MM-DD`; `start > end` is a 422).
Paging: `limit` 1..1000 (default 100), `offset` >= 0; the response has `total` and `next_offset`
(`null` on the last page). An empty page for a known asset/horizon is a 200 with `items: []`.
Unlike the other routes this one queries the store on each (uncached) request, in a worker thread.

### `GET /v1/models`

One entry per asset x horizon: `champion` (the served model: name, version, `role` = `champion`, or
`fallback` when no champion is registered, latest origin date, number of stored forecasts),
`challenger` (the shadow model forecasting next to it, or `null`) and `latest_promotion` (the most
recent gate decision with `promoted`, `reasons`, `metrics`, `mlflow_run_id`). Everything comes from the
store (`models_summary`, `promotion_decisions`); the API does not talk to MLflow, because the API image
has no MLflow client. MLflow alias lookups are future work.

### `GET /v1/backtest/summary`

The file at `VOLF_RESULTS_PATH` (default `<reports_dir>/results.json`, produced by `make report`),
served as-is with `meta` and `assets` documented in the schema. It is re-read when its mtime/size
change. `404 no_results` if the file does not exist; `500 results_unreadable` if it is not valid JSON.
The file is about 400 KB (it includes the VaR table): rely on the ETag. The container does not ship
`reports/`: mount it (compose does: `./reports:/reports:ro`, `VOLF_RESULTS_PATH=/reports/results.json`)
or deliver the file through a ConfigMap/volume in Kubernetes. Without it the endpoint answers 404.

### Errors

Every non-2xx response, including unmatched routes (404), wrong methods (405) and unexpected
failures (500), has this shape (`ErrorResponse` in the OpenAPI schema):

```json
{"error": {"code": "validation_error", "message": "invalid query.horizon: Input should be greater than or equal to 1",
           "request_id": "f22e19b2...", "details": [{"loc": ["query", "horizon"], "msg": "...", "type": "greater_than_equal"}]}}
```

Codes: `validation_error` (422), `unknown_asset` / `no_data` / `no_results` / `not_found` (404),
`method_not_allowed` (405), `rate_limited` (429), `not_loaded` / `unavailable` (503),
`results_unreadable` / `internal_error` (500). The `/readyz` 503 body is a `ReadyResponse`, not an error.

### Request ids and logging

`X-Request-ID` is accepted when it matches `[A-Za-z0-9._-]{1,128}` (otherwise a UUID is generated),
echoed on every response including errors, and bound to the structlog context. One `request` log line
per request (method, path, route template, status, duration_ms, client); probes/metrics are logged at
debug. `VOLF_LOG_JSON=true` for JSON, `VOLF_LOG_LEVEL` for the level.

## 2. Configuration

Shared settings (`volforecast.config.Settings`): `VOLF_DATABASE_URL` (store; sqlite or
`postgresql+psycopg://`), `VOLF_DATA_DIR`, `VOLF_LOG_JSON`, `VOLF_LOG_LEVEL`. API settings
(`volforecast.serving.settings.ApiSettings`):

| Variable | Default | Meaning |
|----------|---------|---------|
| `VOLF_API_DATABASE_URL` | `VOLF_DATABASE_URL` | Override the store URL for the API only |
| `VOLF_RESULTS_PATH` (or `VOLF_API_RESULTS_PATH`) | `<reports_dir>/results.json` | File behind `/v1/backtest/summary` |
| `VOLF_API_RELOAD_INTERVAL_S` | `60` | Seconds between snapshot reloads |
| `VOLF_API_NOT_READY_RETRY_S` | `5` | Reload interval while not ready (bounded by the interval above) |
| `VOLF_API_CACHE_MAX_AGE_S` | `60` | `Cache-Control: max-age` of cacheable GETs |
| `VOLF_API_RATE_LIMIT_RPS` | `20` | Sustained requests/second per client IP and worker; `0` disables |
| `VOLF_API_RATE_LIMIT_BURST` | `40` | Bucket size (burst) |
| `VOLF_API_TRUSTED_PROXY_HOPS` | `0` | Trusted reverse proxies appending to `X-Forwarded-For` (0 = header ignored) |
| `VOLF_API_CORS_ORIGINS` | empty | Allowed origins, comma separated or a JSON list. Empty = no CORS headers; no wildcard unless you set `*` yourself |
| `VOLF_API_READY_ALLOW_FALLBACK` | `false` | Count a `fallback` forecast as "champion loadable" for `/readyz` |
| `WEB_CONCURRENCY` | `1` (image: `2`) | uvicorn worker processes |
| `VOLF_API_PORT`, `VOLF_API_HOST` | `8000`, `0.0.0.0` | Listen address |
| `VOLF_API_GRACEFUL_TIMEOUT_S` | `25` | Max seconds to drain in-flight requests on SIGTERM (chart sets 20) |

## 3. Caching

The cacheable routes return a strong `ETag` (SHA-256 of the exact body, truncated to 128 bits) and
`Cache-Control: public, max-age=N`. The body is serialized deterministically (sorted keys) and carries
no per-request values, so all workers and replicas derive the same ETag for the same data. Send
`If-None-Match` (a list, `*` and `W/` prefixes are understood) and get an empty `304` with the same
`ETag` / `Cache-Control` when it still matches. 304s are counted in the HTTP metrics. Errors and probes
are never cached. Since a forecast changes once a day, a client or CDN may use a larger `max-age`.

## 4. Rate limiting

An in-process token bucket per client IP (`rate_limit_burst` tokens, refilled at `rate_limit_rps`),
applied to `/v1/*` only; `/healthz`, `/readyz` and `/metrics` are never limited. An exhausted bucket
gives `429` with `Retry-After` (whole seconds) and the standard error body. Idle buckets are evicted.

* The state is per worker process: the worst-case effective limit is `WEB_CONCURRENCY x rps` per
  replica, and `replicas x` that cluster-wide. Treat it as abuse protection, not a quota.
* Behind a reverse proxy every request has the proxy's address. Set `VOLF_API_TRUSTED_PROXY_HOPS=N`
  for exactly N proxies that each append the peer address to `X-Forwarded-For`; the client is then the
  Nth entry from the right (entries further left are client-controlled and ignored). The default `0`
  ignores the header, so it cannot be spoofed when there is no proxy. The cloud values overlay sets 1.
* Load tests come from one IP: run the target with `VOLF_API_RATE_LIMIT_RPS=0` (`compose`:
  `API_RATE_LIMIT_RPS=0`). `loadtest/k6/forecast.js` counts a 429 as a failure.

## 5. Readiness, hot reload, shutdown

**Snapshot.** Requests for forecasts and models read an immutable in-memory snapshot (one reference
read, no I/O): this is what keeps p95 latency in the low milliseconds. A background task rebuilds the
snapshot every `VOLF_API_RELOAD_INTERVAL_S` in a worker thread (about 15 small queries: latest
forecast per asset x horizon, models summary, latest promotion, the pipeline gauges) and swaps it in
atomically, so a newly written forecast or champion appears without a restart and requests never wait
on a reload. A failed reload keeps the last good snapshot and logs a warning. While the API is not
ready, the loop retries every `VOLF_API_NOT_READY_RETRY_S`.

**`/readyz`** answers from the loader state (it does not query the DB inline, so it cannot exceed a
probe timeout). `200` when the **last reload succeeded** (DB reachable, tables present) **and at least
one asset x horizon has a `champion` forecast**; otherwise `503` with a `ReadyResponse` body
(`database`, `champion_forecasts`, `snapshot_age_seconds`, `detail`: the exception class only). Notes:

* A store holding only `fallback` rows (no registered `@champion`, a warning in the forecast flow) is
  **not ready** unless `VOLF_API_READY_ALLOW_FALLBACK=true`; `/v1/forecast` still serves those rows.
* There is deliberately **no data-freshness gate**: stale data would pull every replica out of rotation
  at once. Freshness is an alert (`volforecast_forecast_age_seconds`, `DataStale`, `PipelineFailed`).
  (`docs/platform.md` still mentions "data fresh" in the readiness row; this is the implemented rule.)
* If the DB goes away after start-up the pod turns unready (reload fails) while still serving its last
  snapshot to any traffic that reaches it; `/v1/forecast/history` returns `503 unavailable`.
* The API never runs DDL (`ForecastRepository(create=False)`): a store without tables is "not ready",
  not an error at start-up. The flows create the schema.

**Shutdown.** `python -m volforecast.serving` runs uvicorn with `--timeout-graceful-shutdown`
(`VOLF_API_GRACEFUL_TIMEOUT_S`). On SIGTERM it stops accepting connections, drains in-flight requests,
then the lifespan cancels the reload task and disposes the DB engine. Keep the value below
`terminationGracePeriodSeconds` minus the chart's 5 s `preStop` sleep.

**Container.** Runs as UID 10001 with a read-only root filesystem; writes only to `/tmp` (the
Prometheus multiprocess directory `/tmp/volforecast-prom`, wiped at start) and `VOLF_DATA_DIR`.
Probes: liveness/startup `/healthz`, readiness `/readyz` (Helm values `api.probes`, Dockerfile
`HEALTHCHECK` `/healthz`, compose healthcheck `/readyz`).

## 6. Metrics

HTTP metrics (names fixed by `docs/platform.md`): `http_requests_total` and
`http_request_duration_seconds` with labels `route`, `method`, `status`. `route` is the **route
template** (`/v1/forecast/history`), and `unmatched` for anything that matched no route, so raw paths
and ids never become label values. 429s, 304s and 500s are counted under their route.

Pipeline/model gauges come from the store through `monitoring.prom.export_from_store` (reload loop,
not per scrape): `volforecast_model_info` (champion version), `volforecast_forecast_var`,
`volforecast_realized_var`, `volforecast_rolling_qlike`, `volforecast_qlike_ratio_vs_har`,
`volforecast_var_breaches_total`, `volforecast_data_last_bar_timestamp_seconds`,
`volforecast_pipeline_last_success_timestamp_seconds`, `volforecast_drift_share`. The recording is
done into plain dicts rather than `Gauge` objects, because gauges are not clearable in Prometheus
multiprocess mode; a unit test pins the names against `PipelineMetrics`.

API-owned additions (not in `platform.md`), computed at scrape time from the snapshot:

| Metric | Labels | Meaning |
|--------|--------|---------|
| `volforecast_forecast_age_seconds` | `asset, horizon` | Seconds since the served forecast's information set closed (end of the origin day, UTC): forecast freshness |
| `volforecast_pipeline_last_success_age_seconds` | `flow` | Age of the last successful run per flow |
| `volforecast_api_ready` | none | 1 when `/readyz` would be 200 |
| `volforecast_api_snapshot_age_seconds` | none | Seconds since the last successful reload |

With `WEB_CONCURRENCY > 1` the entrypoint sets `PROMETHEUS_MULTIPROC_DIR` (before Python imports
`prometheus_client`) and `/metrics` aggregates the per-worker HTTP counters/histograms, so any worker
answers a scrape with cluster-of-workers totals. Run with `python -m volforecast.serving`; a bare
`uvicorn --workers N` does not set the directory and would export per-worker numbers.

## 7. Examples

```bash
# conditional GET
etag=$(curl -sI 'localhost:8000/v1/models' | awk -F': ' 'tolower($1)=="etag"{print $2}' | tr -d '\r')
curl -s -o /dev/null -w '%{http_code}\n' -H "If-None-Match: $etag" 'localhost:8000/v1/models'   # 304

# last 30 forecasts with realized variance, then the next page
curl -s 'localhost:8000/v1/forecast/history?asset=ETHUSDT&horizon=1&limit=30'
curl -s 'localhost:8000/v1/forecast/history?asset=ETHUSDT&horizon=1&limit=30&offset=30'

# range
curl -s 'localhost:8000/v1/forecast/history?asset=BTCUSDT&horizon=5&start=2026-01-01&end=2026-03-31'

# readiness and a 429
curl -s localhost:8000/readyz
curl -si localhost:8000/v1/models | head -1       # HTTP/1.1 429 Too Many Requests, Retry-After: 1
```

Tests: `tests/unit/serving/` (TestClient on a temp sqlite store; no network). Load test:
`loadtest/k6/forecast.js` (see `docs/platform.md` section 8 and the header of the script).
