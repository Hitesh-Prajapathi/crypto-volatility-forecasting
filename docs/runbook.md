# Runbook

Operational procedures. Each entry is symptom, diagnosis, action. Commands are copied from the
`Makefile`, `docs/platform.md` and the CLIs (`python -m volforecast`, `python -m volforecast.data`).
Sections marked **being implemented in M5** depend on code that does not exist yet
(`src/volforecast/flows/`, `registry/` and `monitoring/` are empty packages today); the commands
there are the intended contract from `docs/platform.md` and may change.

Alert names below match `deploy/monitoring/alerts/volforecast-rules.yaml`.

## 1. Local stack up and down

- Symptom: need the full stack (API, MLflow, Prefect, MinIO, Postgres, Prometheus, Grafana) locally.
- Action:
  ```bash
  cp .env.example .env                           # dev-only credentials; .env stays out of git
  make up                                        # docker compose up -d --build
  docker compose --profile jobs run --rm jobs    # JOB_FLOW=forecast|ingest|evaluate|retrain (M5)
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

## 5. Force a champion rollback (MLflow alias) - being implemented in M5

Background: [ADR-0006](adr/0006-mlflow-aliases-over-stages.md). Registered names follow
`volforecast-{asset}-h{h}` (execution plan example: `volforecast-btc-h1`; confirm the asset casing
once `registry/` lands). Serving reads only `models:/<name>@champion`.

- Symptom: `ModelUnderperformsHAR` or `VaRBreachCluster` fires, or a promotion was wrong.
- Diagnosis: in the MLflow UI compare version metrics and the promotion tags.
- Action:
  ```python
  from mlflow import MlflowClient

  c = MlflowClient()  # needs MLFLOW_TRACKING_URI
  name = "volforecast-btc-h1"
  print(c.get_model_version_by_alias(name, "champion").version)
  c.set_registered_model_alias(name, "champion", "<previous_version>")
  ```
  Repeat for each affected asset and horizon. Then re-run the `forecast` flow so the store holds
  forecasts from the restored model, and record the reason as a tag on the version.

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

## 8. Alerts

Jobs run `python -m volforecast.flows.<flow>` for `ingest`, `forecast`, `evaluate`, `retrain`
(**being implemented in M5**). CronJob schedule (UTC): ingest 00:20, forecast 00:35, evaluate 01:00,
retrain Sundays 02:00. Manual run:
`kubectl -n volforecast create job --from=cronjob/<name> <name>-manual`.

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
