# Runbook

Operational procedures. Each entry is symptom, diagnosis, action. Commands are copied from the
`Makefile`, `docs/platform.md` and the CLIs (`python -m volforecast`, `python -m volforecast.data`).
The four batch flows (`python -m volforecast.flows.<ingest|forecast|evaluate|retrain>`) are
implemented; how they work, the promotion gate and the drift triggers are in
[mlops.md](mlops.md) and are not repeated here. Section 4 (cluster deploy) has not been run on a
real cluster yet. Section 10 covers the Pages workflow, section 11 the Kaggle LSTM runs.

Alert names below match `deploy/helm/volforecast/rules/volforecast-rules.yaml` (the single rules file, shared by compose and the Helm PrometheusRule).

## 1. Local stack up and down

- Symptom: need the full stack (API, MLflow, Prefect, MinIO, Postgres, Prometheus, Grafana) locally.
- Action:
  ```bash
  cp .env.example .env                           # dev-only credentials; .env stays out of git
  make up                                        # docker compose up -d --build
  docker compose --profile jobs run --rm jobs    # JOB_FLOW=forecast|ingest|evaluate|retrain
  make down                                      # docker compose down (use `docker compose down -v` to drop volumes)
  ```
- Symptom: a port is already in use. Diagnosis: `docker compose ps`, `lsof -i :<port>`. Action: set
  `POSTGRES_PORT`, `MLFLOW_PORT=5050` (macOS AirPlay holds 5000), etc. in `.env`, then `make up`.
- Symptom: API `/readyz` returns 503. Diagnosis: readiness needs the DB, a loaded champion and fresh
  data: `curl -s localhost:8000/readyz`, `docker compose logs api`. Action: fix the failing
  dependency (the real API is pending, M6).
- URLs: API :8000, MLflow :5000, Prefect :4200, MinIO console :9001, Prometheus :9090, Grafana :3000.

## 2. Re-ingest or backfill a day

- Symptom: a day is missing or incomplete, or was ingested from a bad file.
- Diagnosis:
  ```bash
  uv run python -m volforecast.data quality --symbol BTCUSDT   # lists days with < 280 of 288 bars
  ```
  Also inspect `klines/<symbol>/<interval>/manifest.json` under the data dir (`snapped_bars`,
  `dropped_bars`).
- Action (single day; `--end` is inclusive, default yesterday):
  ```bash
  uv run python -m volforecast.data ingest --symbol BTCUSDT --start 2026-11-10 --end 2026-11-10
  ```
  Repeat per symbol (`BTCUSDT`, `ETHUSDT`, `SOLUSDT`). `--source coinbase` exists as an alternative
  source but changes the venue: do not mix it into the research dataset without a decision. Then
  re-run the downstream flows (section 8) or the backtest (section 3).
- Flow-level backfill (also updates the realized table and the daily panel; `--start` defaults to
  the last stored day, `--end` to yesterday UTC):
  ```bash
  export MLFLOW_DISABLE_AGENT_HINT=1
  uv run python -m volforecast.flows.ingest --assets BTCUSDT --start 2026-11-01 --end 2026-11-10
  ```
  To backfill forecasts for past days with the current champion (no look-ahead only if the
  champion was trained before those days, see [mlops.md](mlops.md) section 2):
  `uv run python -m volforecast.flows.forecast --assets BTCUSDT --replay-days 30`.
- Note: the execution plan mentions `backfill --date`; no such command exists. Use `ingest`.

## 3. Re-run the backtest and report

- Symptom: data or config changed; `reports/` is stale.
- Action:
  ```bash
  make backtest                                   # configs/backtest.yaml -> artifacts/backtest
  make backtest ASSETS="SOLUSDT" DATA_ROOT=data   # optional subset / data root
  make report                                     # -> reports/tables.md, results.json, figures/
  ```
  `uv run python -m volforecast report --run artifacts/backtest --out reports --inject-readme README.md`
  additionally rewrites the README RESULTS block; only do that when the numbers are final.
- Diagnosis of failures: `artifacts/backtest/run_meta.json` has `n_failures`, timings and data
  hashes. An asset not in the config makes `backtest` exit with code 2.

## 4. Deploy via Argo CD and roll back

Background: [ADR-0010](adr/0010-gitops-argocd-pull.md), `docs/platform.md` sections 4, 6, 7.
The manifests have not yet been run on a cluster; treat names below as intended.

- Deploy: CD builds and signs the images, then opens a PR bumping `image.tag` in
  `values-kind.yaml` and `values-cloud.yaml`. Merging that PR is the deploy; Argo CD syncs. CI never
  runs `kubectl`.
  ```bash
  argocd app get volforecast
  argocd app sync volforecast          # only if auto-sync is off
  kubectl -n volforecast get pods
  ```
- Symptom: after a release `APIErrorRate` or `APILatencyP95` fires, or pods are not Ready.
- Action, preferred: `git revert <merge-commit-of-the-image-tag-bump>` and merge; Argo CD syncs the
  previous tag (Deployment `maxUnavailable: 0`).
- Action, emergency:
  ```bash
  argocd app set volforecast --sync-policy none      # selfHeal would otherwise undo the rollback
  argocd app history volforecast
  argocd app rollback volforecast <history-id>
  ```
  Then revert in git and re-enable auto-sync (restore the `syncPolicy` of
  `deploy/argocd/apps/volforecast.yaml`).
- An image rollback does not change the model. For the model see section 5.

## 5. Roll back the champion (MLflow alias)

Background: [ADR-0006](adr/0006-mlflow-aliases-over-stages.md), [mlops.md](mlops.md) section 4.
Registered names are `volforecast-{asset}-h{h}` with the asset lower-cased (e.g.
`volforecast-btcusdt-h1`). Serving and the `forecast` flow read only `models:/<name>@champion`.

- Symptom: `ModelUnderperformsHAR` or `VaRBreachCluster` fires, or a promotion was wrong.
- Diagnosis: in the MLflow UI compare the versions' `wf_qlike` and promotion tags; the version
  promoted last carries `previous_champion`, which is the version to go back to. Every decision is
  also a row in `promotion_decisions` (forecast store DB).
- Action:
  ```python
  from mlflow import MlflowClient

  c = MlflowClient()  # needs MLFLOW_TRACKING_URI
  name = "volforecast-btcusdt-h1"
  mv = c.get_model_version_by_alias(name, "champion")
  print(mv.version, c.get_model_version(name, mv.version).tags.get("previous_champion"))
  c.set_registered_model_alias(name, "champion", "<previous_version>")
  c.set_model_version_tag(name, "<bad_version>", "rolled_back", "<reason>")
  ```
  Repeat for each affected asset and horizon. Then re-run `forecast` (section 8) so the store holds
  forecasts from the restored model: the upsert key includes `model_version`, so the new rows sit
  next to the old ones and the API serves the champion's. The next scheduled `retrain` can promote the rolled-back version again if it still passes the
  gate; if the gate itself is the problem, fix it through git before the Sunday run.

## 6. Binance outage or incomplete day

Policy: [ADR-0003](adr/0003-missing-bar-policy.md). Up to 3 consecutive missing bars are forward
filled as zero returns; a day with fewer than 280 of 288 bars, or a longer gap, is incomplete and
is excluded from evaluation targets (it stays usable as feature history).

- Symptom: `DataStale` or `PipelineFailed` fires, or the `quality` report lists days.
- Diagnosis: `uv run python -m volforecast.data quality --symbol BTCUSDT`; check whether the
  data.binance.vision archive for the day exists yet (daily archives post with a delay).
- Action: wait and re-ingest (section 2). If the archive is permanently short, leave the day
  flagged incomplete; do not interpolate or lower the 280 threshold (changing it needs a new ADR).
  If the freshness SLO ([slo.md](slo.md)) will be missed, record an incident and let the budget
  absorb it.
- Off-grid timestamps (the Feb 2018 files) are snapped on ingest; counts are in the manifest.

## 7. Rotate secrets

Secrets: `VOLF_DATABASE_URL`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, in the Secret
referenced by `secrets.existingSecret` (e.g. `volforecast-secrets`) or produced by an ExternalSecret.
Never commit plaintext; `secrets.create` is a dev-only escape hatch.

- Sealed Secrets (kind):
  ```bash
  kubectl -n volforecast create secret generic volforecast-secrets \
    --from-literal=VOLF_DATABASE_URL=... --from-literal=AWS_ACCESS_KEY_ID=... \
    --from-literal=AWS_SECRET_ACCESS_KEY=... --dry-run=client -o yaml \
    | kubeseal --format yaml > sealed-volforecast-secrets.yaml
  ```
  Commit the sealed file through a PR; Argo CD applies it.
- External Secrets (cloud): rotate the value in the remote store configured under `externalSecret`
  in the values; the operator refreshes it. To force:
  `kubectl -n volforecast annotate externalsecret <name> force-sync=$(date +%s) --overwrite`.
- Then restart consumers: `kubectl -n volforecast rollout restart deploy/<api-deployment>`; CronJobs
  pick up the new Secret on their next run. Verify `/readyz` and the next `forecast` run, then revoke
  the old credential.

## 8. Flows: run, rerun, triggers

CronJobs (UTC, from `deploy/helm/volforecast/values.yaml`): `ingest` 00:20, `forecast` 00:35,
`evaluate` 01:00, `retrain` Sundays 02:00 (each runs `python -m volforecast.flows.<flow>`,
`backoffLimit: 2`). Config of *what* the flows do: `configs/production.yaml`.

Run or rerun one on the cluster (the Job reuses the CronJob's env and Secret):
```bash
kubectl -n volforecast create job --from=cronjob/<name> <name>-manual
kubectl -n volforecast logs job/<name>-manual
```
Locally / in compose: `docker compose --profile jobs run --rm jobs` with `JOB_FLOW` set in `.env`,
or the CLI below. All flows take `--assets`, `--horizons`, `--data-dir`, `--database-url`,
`--tracking-uri`, `--config`; each run writes a `pipeline_runs` row (`success`, `failed`,
`skipped`; the JSON `detail` has the per-asset outcome).

| Flow | Rerun / options | Idempotent |
|------|-----------------|------------|
| `ingest` | `--start D --end D --source binance` | yes, merge on `open_time`; no new bars -> `skipped` |
| `forecast` | `--as-of D`, `--replay-days N` | yes, upsert per `(asset, h, model_version, origin_date)` |
| `evaluate` | none | yes, upsert per metric key |
| `retrain` | `--as-of D`, `--force`, `--only-if-triggered` | skipped if a decision exists for `(asset, h, as-of)`; `--force` reruns |

Order after a data fix: `ingest` -> `forecast` -> `evaluate`. Flags (`--as-of`, `--force`, ...) are
not in the CronJob; run the CLI from a shell with the same env, or edit the Job's `args`:
```bash
export MLFLOW_DISABLE_AGENT_HINT=1
uv run python -m volforecast.flows.forecast --assets BTCUSDT --as-of 2026-11-10
uv run python -m volforecast.flows.retrain --assets BTCUSDT --horizons 1 --force
```

### Drift or performance trigger fired
- Symptom: `evaluate` logged `retrain_decision` with requests (also in the latest
  `pipeline_runs.detail`: `retrain`, `retrain_requests`, `retrain_reasons`), or
  `ModelUnderperformsHAR` / `VaRBreachCluster` fired. Nothing retrains automatically on a trigger.
- Diagnosis: `retrain_reasons` says performance (QLIKE ratio, VaR breaches) or data drift
  (`drift_share`, report under `artifacts/reports/drift/<asset>/<date>.html`). Data drift alone is
  a weak signal in crypto ([mlops.md](mlops.md) section 5).
- Action: `python -m volforecast.flows.retrain --only-if-triggered` (retrains only the pairs the
  latest successful `evaluate` asked for; `skipped` if none). The result is gated, so a spurious
  trigger costs compute only. If the gate rejects, `@challenger` stays on the rejected version and
  the champion is unchanged; if the performance problem is acute, roll back (section 5).

### Stale forecast (freshness SLO, `PipelineFailed`)
- Symptom: the forecast for today is not in the store by 01:00 UTC ([slo.md](slo.md) SLO 2), or
  `PipelineFailed`.
- Diagnosis: latest `ingest` and `forecast` Jobs and their `pipeline_runs` rows. `forecast`
  forecasts the latest *complete* day, so a failed or `skipped` `ingest` (archive not posted yet,
  section 6) leaves yesterday's forecast in place. No `@champion` is not fatal: the flow fits
  `fallback_model` and warns; check the log for the fallback.
- Action: fix the cause, then `ingest` (if data is behind) followed by `forecast`, via manual Jobs
  as above. A late run that finishes after 01:00 still counts against the SLO.

## 9. Alerts

### DataStale (critical)
- Symptom: no new bar for an asset for more than 26h (`volforecast_data_last_bar_timestamp_seconds`).
- Diagnosis: `kubectl -n volforecast get jobs`; logs of the latest `ingest` job; section 6.
- Action: fix the cause (network, archive delay, egress NetworkPolicy), then re-ingest (section 2).

### PipelineFailed (critical)
- Symptom: no successful `forecast` flow run in 26h.
- Diagnosis: `kubectl -n volforecast get jobs,pods`; `kubectl -n volforecast logs job/<forecast-job>`;
  Prefect UI. Jobs retry with `backoffLimit: 2`. Typical causes: stale data (see DataStale), missing
  `@champion` alias, Postgres or MLflow unreachable.
- Action: fix, then `kubectl -n volforecast create job --from=cronjob/forecast forecast-manual`.

### APIErrorRate (critical)
- Symptom: 5xx ratio above 1% for 10m.
- Diagnosis: `kubectl -n volforecast logs deploy/<api-deployment>`; `/readyz`; DB connectivity; did a
  deploy just happen?
- Action: roll back the image (section 4); if the forecast store is empty or stale, rerun `forecast`.

### APILatencyP95 (warning)
- Symptom: P95 above 200ms for 10m (excluding `/metrics`, `/healthz`, `/readyz`).
- Diagnosis: Grafana Service dashboard; `kubectl -n volforecast get hpa`; CPU throttling; DB latency.
- Action: raise replicas or HPA limits through git, fix slow queries; roll back if it correlates
  with a deploy.

### ModelUnderperformsHAR (warning)
- Symptom: the QLIKE ratio vs HAR stayed above 1 for 10 days.
- Diagnosis: Grafana Model dashboard; check for a regime shift, data issues and drift
  (`volforecast_drift_share`).
- Action: if the degradation is real, move `@champion` back (section 5) and open a retrain or
  investigation task.

### VaRBreachCluster (warning)
- Symptom: 3 or more 99% VaR breaches in 10 days for an asset.
- Diagnosis: compare with the expected breach rate; compare forecast vs realized variance
  (underprediction in a stress episode is plausible); see the RV vs close-to-close scale caveat in
  [model_card.md](model_card.md).
- Action: review the VaR distribution and the champion; roll back (section 5) if a model change
  preceded the cluster.

### ForecastLate (warning)
- Symptom: the h=1 forecast for an asset is more than 25h05m old, i.e. tomorrow's forecast was not
  served 5 minutes after 01:00 UTC (`volforecast_forecast_age_seconds`).
- Diagnosis: the latest `forecast` CronJob and flow logs; whether `PipelineFailed` or `DataStale`
  also fire; `volforecast_api_ready`. See "Stale forecast" in section 8.
- Action: fix the cause and rerun `forecast`; the alert clears once the new forecast is served.

### ForecastFreshnessConsecutiveMisses (critical)
- Symptom: the 01:00 UTC freshness sample (`volforecast:forecast_fresh_at_0100:bool`) missed on two
  consecutive days.
- Action: treat it as an incident. Fix ingest or forecast, then rerun `forecast` for the missed day.

### ForecastFreshnessBudgetBurn (warning)
- Symptom: the 30-day freshness ratio is below 99%. One miss keeps it firing for 30 days.
- Action: freeze changes to the ingest and forecast flows as [slo.md](slo.md) describes, and review
  the misses (`volforecast:forecast_fresh_at_0100:misses_30d`).

### Stateful champion loaded without its learned state
- Symptom: the forecast flow raises `stateful model 'ens_inv' was loaded without its learned state`.
- Cause: the `@champion` artifact predates the warm-start fix or lost its history.
- Action: `python -m volforecast.flows.retrain --assets <asset> --horizons <h> --force`.

## 10. GitHub Pages dashboard workflow

Background: [ADR-0013](adr/0013-github-pages-dashboard.md), `docs/platform.md` section 10.

- Symptom: the site is stale or missing, or the `pages` run failed (red in the Actions tab).
- Rerun: Actions, `pages`, "Run workflow" (`workflow_dispatch`), or `gh workflow run pages.yml`
  then `gh run watch`. It also runs daily at 01:30 UTC; runs queue and are not cancelled. The first
  run after enabling Pages bootstraps the full history and is slow (timeout 90 minutes).
- Diagnosis: open the failed run and find the first red step.
  - `deploy` fails with a Pages error: Settings, Pages, Source must be "GitHub Actions".
  - Ingest fails: the Binance Vision archive is not posted yet or unreachable (section 6); rerun later.
  - Forecast fails: read its log. The CI job has no `@champion` and uses `fallback:xgb` by design.
  - The restore step logs "no previous store": the cache was evicted and the live site had no
    `data/store.db`, so the track record restarts (the page says so).
- A failed run never replaces the published site; the previous version stays up. After fixing the
  cause, rerun the workflow. To build locally use `make site`; pass `--demo` for a scratch store so
  it cannot pass as a live record.

## 11. Rerun the LSTM seed backtests on Kaggle

Background: [ADR-0012](adr/0012-lstm-challenger.md), results in
[results/lstm_seeds.md](results/lstm_seeds.md).

- Symptom: the LSTM code, config or data changed and the five-seed table is stale.
- Action: follow [scripts/kaggle/README.md](../scripts/kaggle/README.md) (private dataset, private
  GPU kernel, per-seed reports, aggregation); the steps are not repeated here. The upload must come
  from a committed tree, and the `CODE_SHA` goes into the result.
- GPU and CPU runs are not bit-identical, so compare against the GPU numbers only.
