# MLOps: the daily cycle

How a forecast gets produced, monitored and replaced. Companion to `docs/platform.md` (containers,
metric names, Kubernetes) and `docs/runbook.md` (operations). Decisions: ADR-0006 (aliases),
ADR-0007 (CronJob triggers, Prefect executes), ADR-0009 (batch inference, forecast store),
ADR-0011 (in-house drift).

## 1. Architecture

```
 K8s CronJob (when)            Prefect flow (how)                      State
 ───────────────────           ─────────────────────────────────       ──────────────────────────
 00:20  ingest   ──────────►   flows.ingest   ──► parquet klines/panel  data/ (volume)
                                                └► realized table      ┐
 00:35  forecast ──────────►   flows.forecast ◄── MLflow @champion      │  forecast store
                                                └► forecasts table     │  (Postgres; sqlite locally)
 01:00  evaluate ──────────►   flows.evaluate ──► monitoring_metrics    │
                                                └► retrain decision    ┘
 Sun 02:00 retrain ────────►   flows.retrain  ──► MLflow runs + registry (@challenger/@champion)
                                                └► promotion_decisions
                                                                   API  GET /v1/forecast reads the store
                                                                        GET /metrics (Prometheus)
```

* **Schedules live in the Helm chart** (`cronJobs` in `deploy/helm/volforecast/values.yaml`); no
  Prefect worker or deployment exists. Each CronJob runs `python -m volforecast.flows.<name>` to
  completion; Prefect (ephemeral API, or `PREFECT_API_URL`) gives task structure, retries (ingest
  fetch: 2 retries) and run history.
* Each flow module has plain functions that take a `Runtime` (settings, `configs/production.yaml`,
  forecast store, lazily the MLflow client), thin `@task` wrappers, and a `@flow`. A flow
  processes every asset, isolates per-asset failures, then raises if any failed, so the process
  exits non-zero and Kubernetes retries (`backoffLimit: 2`).
* Every run writes a `pipeline_runs` row (`running` -> `success` | `failed` | `skipped`, with a JSON
  `detail`). On success the pipeline metrics are pushed to a Pushgateway when
  `VOLF_PUSHGATEWAY_URL` is set (`monitoring/prom.py`); the API serves the same gauges from the
  store.
* Models are loaded by alias: `models:/volforecast-{asset}-h{h}@champion` (asset lower-cased).
  Promotion is one `set_registered_model_alias` call; rollback is moving the alias back.

## 2. Flows

| Flow | Does | Writes | Idempotent by |
|------|------|--------|---------------|
| `ingest` | Re-fetches the last stored day through yesterday (UTC), merges with the stored months, validates, writes monthly parquet partitions + `manifest.json`, rebuilds the daily panel, syncs `realized`, records `data_last_bar_ts` | parquet store, `realized`, `monitoring_metrics`, `pipeline_runs` | merge on `open_time`; no new bars -> status `skipped` |
| `forecast` | Information set = latest complete day. Loads `@champion` (else fits `fallback_model` on all history and warns), forecasts every horizon; h=1 also 99%/95% VaR and 97.5% ES. Also writes the HAR benchmark row and, if registered, a `@challenger` shadow row | `forecasts`, `realized` | upsert on `(asset, horizon, model_version, origin_date)` |
| `evaluate` | Joins realized RV to stored live forecasts: rolling QLIKE vs HAR, VaR breach log and Basel zone, Kupiec p; feature drift of the last 30 days vs the champion's training window; decides retrain | `monitoring_metrics`, drift report files, `pipeline_runs.detail` | upsert on the metric key |
| `retrain` | Walk-forward of all candidates, refit of the best, register as `@challenger`, promotion gate, alias move | MLflow runs/versions/tags, `promotion_decisions` | skipped if a decision exists for `(asset, horizon, as-of day)` unless `--force` |

Only *served* rows (`champion`, `fallback`) and the `benchmark` rows are scored; shadow
(`challenger`) rows are never served.

### Running locally

All flows read `VOLF_DATA_DIR`, `VOLF_DATABASE_URL`, `MLFLOW_TRACKING_URI` (defaults: sqlite files
under `artifacts/`) and accept `--data-dir`, `--database-url`, `--tracking-uri`, `--config`,
`--assets`, `--horizons`.

```bash
export MLFLOW_DISABLE_AGENT_HINT=1
uv run python -m volforecast.flows.ingest   --assets BTCUSDT [--end 2026-01-31] [--source coinbase]
uv run python -m volforecast.flows.retrain  --assets BTCUSDT [--as-of 2026-01-31] [--force]   # first: creates @champion
uv run python -m volforecast.flows.forecast --assets BTCUSDT [--as-of D] [--replay-days 60]
uv run python -m volforecast.flows.evaluate --assets BTCUSDT
uv run python -m volforecast.flows.retrain  --only-if-triggered       # what evaluate asked for
```

`ingest` needs an existing store (bootstrap with `python -m volforecast.data ingest --symbol ...
--start ...`, or pass `--start`). `forecast --replay-days N` forecasts each of the last N days with
the frozen champion; it is the way to fill the store when bootstrapping or to exercise the cycle,
and it is a genuine out-of-sample test only if the champion was trained before those days. Port
5000 is taken by AirPlay on macOS: use 5050 for an MLflow server, or the default sqlite URI.

## 3. Forecast store schema

SQLAlchemy 2.0 models in `forecast_store/models.py`; `create_all` runs on connect (it creates
missing tables and never alters existing ones; migrations are future work). Dates are UTC calendar
days. VaR/ES are positive loss magnitudes.

| Table | Key | Columns |
|-------|-----|---------|
| `forecasts` | unique `(asset, horizon, model_version, origin_date)` | `asset, horizon, model_name, model_version, role, origin_date, target_date, forecast_var, var95, var99, es975, dist, data_hash, created_at`. `role` in `champion`, `fallback`, `challenger`, `benchmark`. `model_version` is the registry version number, `fallback:<model>` or `har_benchmark`. `forecast_var` = E[sum of RV over the next h days]. VaR/ES only for h=1. `dist` = `fhs` or `normal`. |
| `realized` | `(asset, date)` | `rv, bpv, rs_pos, rs_neg, rq, ret, complete` |
| `pipeline_runs` | `id` | `flow, run_date, started_at, finished_at, status, detail (JSON)` |
| `promotion_decisions` | `id` | `created_at, asset, horizon, as_of_date, registered_model, challenger_version, challenger_model, champion_version, promoted, reasons (JSON), metrics (JSON), mlflow_run_id` |
| `monitoring_metrics` | unique `(asset, horizon, model, metric, window, as_of_date)` | `value, created_at`. Metrics: `rolling_qlike`, `qlike_ratio_vs_har`, `var99_hit`, `var95_hit`, `drift_share`, `data_last_bar_ts` (non-applicable `horizon`/`window` are 0, `model` is `""`) |

VaR in the forecast flow: filtered historical simulation. The standardized residuals are
`z_s = ret_{s+1} / sqrt(f_s)` from the champion's stored walk-forward forecasts plus its own live
forecasts (only those whose next-day return is known at the origin); the quantile of `z` times
`sqrt(f)` is the VaR. With fewer than `promotion.min_resid` (250) residuals (fallback model), the
normal distribution is used and recorded in `dist`; since `f` forecasts realized variance rather
than `E[ret^2]`, that fallback VaR is not scale-calibrated (see `evaluation/risk.py`).

## 4. Promotion gate

`registry/promotion.py` (config: `promotion:` in `configs/production.yaml`). The challenger is the
walk-forward winner (lowest mean QLIKE over the last `window` = 365 common origins among the
candidates `log_har`, `harq`, `har` plus any of `lgbm*`, `xgb*`, `ens_*` that builds), refit on the
whole information set. It is promoted only if **all** hold on the last 365 origins:

| | Criterion |
|--|-----------|
| (a) | Mean QLIKE <= champion's, and the Diebold-Mariano test "challenger worse than champion" has p > 0.10. Skipped when there is no champion. |
| (b) | Mean QLIKE ratio vs HAR < 1 - `epsilon` (0.0). Vacuous when the challenger is the HAR benchmark itself. |
| (c) | h=1 only: FHS 99% VaR passes Kupiec and Christoffersen CC at 5% and is in the Basel green zone (last 250 obs); at least 90% of the window rows used FHS. |
| (d) | All window forecasts finite and positive; the latest forecast lies within [min/10, max*10] of the trailing h-day RV range. |

Fewer than `min_eval_obs` (60) scored origins fails the gate. Every decision, promoted or not, is
an MLflow run tagged `promotion_decision`, a `promotion_decisions` row and tags on the model
version. The DB row and the `@champion` move share one transaction: if the alias call fails, no
decision row is committed and the alias is restored. On promotion the old version is kept in the
`previous_champion` tag and `@challenger` is dropped; on rejection `@challenger` stays on the
rejected version.

## 5. Drift and retrain triggers

**Performance drift** (`monitoring/performance.py`; the trigger spec):

* the champion's rolling-30-day QLIKE divided by HAR's is > 1.0 on each of the last 10 days
  (per horizon, scored on common live origins); or
* the 99% VaR breach count over the trailing 250 observations reaches the Basel yellow zone. It
  only fires once a full 250-observation window exists (with few observations a single breach
  would already read "red"). It requests all horizons of the asset.

**Data drift** (`monitoring/drift.py`, ADR-0011): per feature (log RV, log BPV, log RS+, log RS-,
sqrt RQ, |return|), the current window (last 30 complete days) is compared with the reference
(every complete day up to the champion's `train_end_date`, before the current window).

* PSI on **5** quantile bins taken from the reference, empty bins floored at 1e-4.
* Wasserstein-1 distance, normalized by the reference standard deviation.
* A feature is drifted when **PSI > 0.25 and normalized Wasserstein > 0.5**. `drift_share` is the
  share of drifted features. Needs >= 60 reference and >= 10 current days, otherwise
  `insufficient_data`.

Why these thresholds: PSI is chi-square-like; under no drift its mean is about
`(B-1)(1/n_cur + 1/n_ref)`. With the textbook B=10 bins and 30 current days that is 0.33, above the
usual 0.2-0.25 cut-offs, so a stationary series was flagged on most features (84% of 200 simulated
seeds had half or more of the features flagged at PSI > 0.2 / 10 bins). With B=5 the null mean is
0.14. 0.25 is the conventional "significant shift" level. PSI alone is still noisy at n=30, so a
practical-significance guard is added: half a reference standard deviation (Cohen's "medium"
effect). An ad hoc simulation (not committed; i.i.d. log-normal data, 400 days, 30-day window, 300
seeds) gave about 1% of features flagged on stationary data (1% of seeds had half or more
flagged), versus 80% of features flagged for a +0.75 s.d. shift in log RV. What is committed in
`tests/unit/monitoring/test_drift.py` is weaker: over 40 stationary seeds the mean share flagged
is < 0.1 and at most 10% of seeds flag half the features, a +1 s.d. shift flags at least half, and
a 50x level shift flags every feature. The six features are all functions of the same
intraday returns and are strongly correlated, so `drift_share` is effectively one noisy indicator,
not six independent tests. These numbers come from simulated i.i.d. data; real series are
autocorrelated, so false positives in persistent-volatility regimes will be higher (the synthetic
panel used in the flow tests, with AR(1) log-variance, shows it).

**Retrain decision** (`flows/evaluate.py: decide_retrain`): request a retrain for an asset's
horizons when a performance trigger fired, or when `drift_share >= drift_share_trigger` (0.5; null
disables) which requests all horizons. The decision is returned by the flow, logged, and stored in
`pipeline_runs.detail` (`retrain`, `retrain_requests`, `retrain_reasons`). Data drift alone is a
weak signal in crypto, which is why it is a config switch. Nothing starts the retrain
automatically: the weekly `retrain` CronJob runs for every pair, and `retrain --only-if-triggered`
(run manually or from an extra CronJob) retrains only what the latest successful `evaluate` asked
for. A promotion is always gated, so a spurious trigger costs compute, not a worse champion.

Drift reports are written to `artifacts/reports/drift/<asset>/<date>.{json,html}`
(`VOLF_ARTIFACTS_DIR`).

## 6. Metrics

Names are those of `docs/platform.md` section 1 (`volforecast_forecast_var`,
`volforecast_qlike_ratio_vs_har`, `volforecast_var_breaches_total`, `volforecast_drift_share`, ...);
`monitoring/prom.py` rebuilds them from the store (`export_from_store`), so the database is the
source of truth and the API and the Pushgateway publish identical numbers.

## 7. Demo: one cycle on the real BTCUSDT data

Run once on a copy of the repository's `data/` (BTCUSDT 5-minute bars 2017-08-17 to 2026-11-13,
3343 complete days) with a temp sqlite forecast store and a temp sqlite MLflow registry outside the
repository. The production config (`configs/production.yaml`, all candidates) was used unchanged.
`ingest` was skipped (no network). One asset, both horizons, single process on a laptop:

```bash
export D=/tmp/volf-demo VOLF_ARTIFACTS_DIR=/tmp/volf-demo/artifacts
A="--assets BTCUSDT --data-dir $D/data --database-url sqlite:///$D/store.db --tracking-uri sqlite:///$D/mlflow.db"
uv run python -m volforecast.flows.retrain  $A --as-of 2026-09-14      # 1. bootstrap champions (3m44s)
uv run python -m volforecast.flows.forecast $A --replay-days 60        # 2. 60 days with the frozen champions (9s)
uv run python -m volforecast.flows.evaluate $A                         # 3. score + drift + decision
uv run python -m volforecast.flows.retrain  $A --only-if-triggered     # 4. what evaluate asked for (3m40s)
uv run python -m volforecast.flows.forecast $A                         # 5. latest forecast (run twice)
```

**1. Bootstrap, trained on data up to 2026-09-14** (mean QLIKE over the last 365 walk-forward
origins; lower is better):

| h | log_har | harq | har | lgbm | xgb | ens_eq | ens_inv | Best | Decision |
|---|---------|------|-----|------|-----|--------|---------|------|----------|
| 1 | 0.302 | 0.312 | 0.350 | 0.204 | 0.206 | 0.263 | 0.263 | lgbm | **rejected**: (c) 7/365 VaR breaches, Kupiec p=0.118, Basel yellow |
| 5 | 0.209 | 0.235 | 0.253 | 0.198 | 0.190 | 0.200 | 0.200 | xgb | **promoted** (QLIKE ratio vs HAR 0.753; no champion before) |

So after step 1 there was an `@champion` for h=5 (xgb v1) and none for h=1; the rejected lgbm v1
stayed on `@challenger`.

**2. Replay of the last 60 days (2026-09-15 .. 2026-11-13)**: 300 rows. h=5 used the xgb
champion; h=1, having no champion, used the `fallback` model (`log_har`, refit at each origin,
normal-distribution VaR), plus the lgbm shadow rows and the HAR benchmark.

**3. Evaluate** (decision, as returned and stored in `pipeline_runs.detail`):

```
retrain: True   requests: {BTCUSDT: [1, 5]}   reasons: ["BTCUSDT: data drift, 67% of features drifted (trigger 50%)"]
h1: 59 scored, rolling-30d QLIKE ratio vs HAR 0.857 (latest), 99% VaR breaches 1/59 (Kupiec p=0.63; Basel zone n/a: <250 obs)
h5: 55 scored, rolling-30d QLIKE ratio vs HAR 0.261 (latest)
```

Performance triggers did not fire; the data-drift trigger did. Drift (reference = every complete
day up to the champion's training end 2026-09-14, n=3283; current = last 30 days to 2026-11-13):

| feature | PSI (5 bins) | W / ref s.d. | drifted |
|---------|-------------:|-------------:|---------|
| log_rv | 3.26 | 0.695 | yes |
| log_bpv | 2.03 | 0.677 | yes |
| log_rs_pos | 2.09 | 0.709 | yes |
| log_rs_neg | 1.72 | 0.652 | yes |
| sqrt_rq | 1.82 | 0.256 | no (effect-size guard) |
| abs_ret | 0.23 | 0.332 | no |

The last 30 days' mean realized daily variance is 3.9e-4 against 16e-4 over the full history, i.e.
this is a long calm spell relative to the 2017-2026 reference; the detector is doing what it is
specified to do, and illustrates that a long reference makes "drift" a regime comparison (see
limitations).

**4. Retrain `--only-if-triggered`, as of 2026-11-13** (both horizons were requested):

| h | Best | QLIKE (best) | Gate | Decision |
|---|------|-------------:|------|----------|
| 1 | xgb | 0.149 (har 0.304, lgbm 0.154) | (a) skip (no champion); (b) ratio 0.490; (c) 5/365 breaches, Kupiec p=0.501, CC p=0.744, Basel green; (d) pass | **promoted**: `@champion` = v2 |
| 5 | xgb | 0.147 (har 0.216, lgbm 0.155) | (a) 0.14661 vs 0.14661, DM p=1.0; (b) ratio 0.678; (d) pass | **promoted**: `@champion` v1 -> v2 |

At h=5, check (a) compares the challenger with a walk-forward of the champion's own spec
(`model_name` + params tags). The retrained best candidate had the same spec, so the losses tie
exactly and the promotion is a refresh of the fitted parameters on newer data, not an improvement.

**5. Latest forecast** (origin 2026-11-13, served by `xgb` v2 for both horizons; `forecasts` table):

| h | target | forecast_var | implied daily vol (annualized) | var95 | var99 | es975 | dist |
|---|--------|-------------:|--------------------------------|------:|------:|------:|------|
| 1 | 2026-11-14 | 2.76e-4 | 1.66% (31.7%) | 2.46% | 4.39% | 4.31% | fhs |
| 5 | 2026-11-18 | 3.64e-3 | 2.70% per day (51.5%) | | | | |

The second run of step 5 reported the same 4 rows and left the store unchanged (upsert). The
`pipeline_runs` table ended with 7 rows (retrain, forecast, evaluate, evaluate, retrain, forecast,
forecast), all `success`, and 4 `promotion_decisions` rows.

Caveats: this is one asset and one date. Step 2 evaluates frozen champions on 60 days, and the
numbers in step 3 come from 55-59 scored origins (the h=5 targets overlap), so the ratios are
noisy; they show the pipeline works end to end, not that the model beats HAR. The identical QLIKE of
`ens_eq` and `ens_inv` in the tables was not investigated.


## 8. Known limitations

* **Nothing starts retrain automatically** after `evaluate` flags a trigger (section 5).
* **Prefect has no worker, schedule or deployment.** Without `PREFECT_API_URL` each run starts an
  ephemeral local API (a few seconds of overhead, state under `~/.prefect`; with the read-only
  container filesystem set `HOME=/tmp`, as the platform contract does).
* **Live scoring needs history.** The QLIKE ratio needs 30 common live origins before the first
  value and 10 more consecutive days to trigger; the Basel trigger needs 250 scored days. A fresh
  deployment cannot trigger on performance for roughly 40 days (`forecast --replay-days` shortens
  this for a champion trained earlier).
* **Residual history for FHS is the champion's walk-forward plus live forecasts.** The fallback
  model has none and uses the normal distribution (not scale-calibrated).
* **Models are refit only by `retrain`.** A champion keeps its fitted parameters and only updates
  its state (e.g. HAR regressors) with new data between retrains.
* **Single process per asset loop.** Assets and horizons are processed sequentially; the full
  `retrain` with the optional `lgbm`/`xgb`/`ens_*` candidates takes minutes per pair (see the
  demo for a measured duration).
* **Only sqlite was exercised.** All tests and the demo ran against sqlite stores; the Postgres
  path (`postgresql+psycopg://`) has its own upsert branch in `forecast_store/repository.py` that
  has not been run here.
* **Jobs image extras.** The flows import `sqlalchemy` and `prometheus_client`, which pyproject
  lists under the `serve` extra, while `docker/jobs.Dockerfile` defaults to `UV_EXTRAS="--extra
  mlops"` (mlflow, prefect, boto3). Build the jobs image with
  `--build-arg UV_EXTRAS="--extra mlops --extra serve"` (or move those packages into the `mlops`
  extra); otherwise the CronJobs fail on import.
* **Schema changes** need manual `ALTER TABLE` (no Alembic yet).
* **Drift thresholds were tuned on simulated i.i.d. data**, see section 5. With a reference that
  spans the whole history, a persistent volatility regime reads as drift (the demo flagged 67% of
  features after a calm month), so the data-drift trigger will request retrains more often than a
  rolling reference would; the promotion gate keeps that from degrading the champion.
* **Gate check (a) ties when the retrained best has the champion's own spec** (demo, h=5):
  promotion then only refreshes parameters.
* `ingest` re-fetches the last stored day on every run and relies on the Binance Vision archive,
  which publishes a day's file after the day ends; an unavailable day is skipped, not failed
  (the `DataStale` alert catches a persistent gap).
