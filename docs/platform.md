# Platform guide: containers, local stack, Kubernetes, GitOps

## 1. Runtime contracts

Everything in `docker/`, `compose.yaml`, `deploy/`, and `loadtest/` is built against these.
Application code must satisfy them.

**API** (ASGI app `volforecast.serving.app:app`, port 8000)

| Endpoint | Contract |
|----------|----------|
| `GET /healthz` | Liveness. 200 as long as the process runs. No dependencies. |
| `GET /readyz` | Readiness. 503 until the DB is reachable and at least one champion forecast is cached (no freshness gate: stale data is alerted on, not used to pull replicas). See [api.md](api.md). |
| `GET /metrics` | Prometheus exposition. |
| `GET /v1/forecast?asset=BTCUSDT&horizon=1` | Latest champion forecast. |

**Jobs**: `python -m volforecast.flows.<flow>` for `flow` in `ingest`, `forecast`, `evaluate`, `retrain`.
A job exits 0 on success and non-zero on failure (Kubernetes retries `backoffLimit: 2`).

**Environment**

| Variable | Source | Purpose |
|----------|--------|---------|
| `VOLF_DATA_DIR` | ConfigMap | Data directory (`/data`, a volume) |
| `VOLF_LOG_JSON` | ConfigMap | `true`: JSON logs |
| `VOLF_DATABASE_URL` | Secret | SQLAlchemy URL, e.g. `postgresql+psycopg://...` |
| `MLFLOW_TRACKING_URI` | ConfigMap | MLflow server |
| `MLFLOW_S3_ENDPOINT_URL` | ConfigMap | MinIO/S3 endpoint |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | Secret | S3 credentials |
| `PREFECT_API_URL` | ConfigMap | Prefect server |
| `WEB_CONCURRENCY` | ConfigMap | uvicorn workers (default 2) |

The containers run with a **read-only root filesystem**: write only to `/tmp` and the data
volume. `HOME=/tmp`, `MPLCONFIGDIR=/tmp/matplotlib`.

**Prometheus metrics**

| Metric | Labels |
|--------|--------|
| `http_request_duration_seconds` (histogram) | `route, method, status` |
| `http_requests_total` | `route, method, status` |
| `volforecast_model_info` (gauge = 1) | `asset, horizon, version` |
| `volforecast_forecast_var` | `asset, horizon` |
| `volforecast_realized_var` | `asset` |
| `volforecast_rolling_qlike` | `asset, horizon, model, window` |
| `volforecast_qlike_ratio_vs_har` | `asset, horizon` |
| `volforecast_var_breaches_total` | `asset, level` (`level` is `99` or `0.99` for the 99% alert) |
| `volforecast_data_last_bar_timestamp_seconds` | `asset` |
| `volforecast_pipeline_last_success_timestamp_seconds` | `flow` |
| `volforecast_pipeline_last_success_age_seconds` | `flow` |
| `volforecast_forecast_age_seconds` | `asset, horizon` |
| `volforecast_api_ready` | (none) |
| `volforecast_api_snapshot_age_seconds` | (none) |
| `volforecast_drift_share` | `asset` |

`route` must be the route template (`/v1/forecast`), never the raw path, to bound cardinality.
Pipeline metrics must be visible to the scraped `/metrics` of the API (or a Pushgateway/exporter
added to the scrape config); jobs are short-lived and cannot be scraped directly.

## 2. Images

```bash
docker build -f docker/api.Dockerfile  -t volforecast-api:dev  --build-arg REVISION=$(git rev-parse --short HEAD) .
docker build -f docker/jobs.Dockerfile -t volforecast-jobs:dev --build-arg REVISION=$(git rev-parse --short HEAD) .
docker run --rm --read-only --tmpfs /tmp -p 8000:8000 volforecast-api:dev
```

Multi-stage: a builder with the pinned `uv` binary runs `uv sync --frozen --no-dev` into
`/opt/venv`; the runtime is `python:3.12-slim-bookworm` + `libgomp1` (lightgbm/xgboost), UID/GID
10001, OCI labels (`revision` from `--build-arg REVISION`). The API image uses
`--extra serve`. CUDA wheels pulled in by xgboost on Linux (`site-packages/nvidia`, about 200 MB)
are deleted: inference is CPU-only. If a flow needs more packages (prefect, mlflow, boto3), add a
`jobs` extra in `pyproject.toml` and build with `--build-arg UV_EXTRAS="--extra jobs"`.

## 3. Local stack (Docker Compose)

```bash
cp .env.example .env            # dev-only credentials; .env must stay out of git
docker compose build api mlflow
docker compose up -d
docker compose --profile jobs run --rm jobs      # JOB_FLOW=forecast|ingest|evaluate|retrain
docker compose down -v
```

| Service | URL |
|---------|-----|
| API | http://localhost:8000 (`/healthz`, `/readyz`, `/metrics`) |
| MLflow | http://localhost:5000 (on macOS, port 5000 is taken by AirPlay: set `MLFLOW_PORT=5050`) |
| Prefect | http://localhost:4200 |
| MinIO console | http://localhost:9001 |
| Prometheus | http://localhost:9090 (targets, rules from `deploy/monitoring/alerts`) |
| Grafana | http://localhost:3000 (dashboards under the `volforecast` folder) |

Postgres hosts three databases (`volforecast`, `mlflow`, `prefect`; created by
`deploy/compose/postgres-init.sh`). MinIO buckets `mlflow` and `reports` are created by
`minio-init`. MLflow proxies artifacts (`--serve-artifacts`), so clients need no S3 access to log
artifacts. Override host ports in `.env` (`POSTGRES_PORT`, ...) when something is already bound.

## 4. Local Kubernetes (kind + Argo CD)

Requirements: docker, kind, kubectl, helm (kubeseal optional).

```bash
REPO_URL=https://github.com/<you>/crypto-volatility-forecasting.git deploy/kind/bootstrap.sh
```

The script creates the 3-node cluster (`deploy/kind/cluster.yaml`), installs ingress-nginx,
metrics-server (`--kubelet-insecure-tls`), sealed-secrets, kube-prometheus-stack and Argo CD
(pinned chart versions), builds and `kind load`s the images, creates `volforecast-secrets`
(sealed when `kubeseal` exists, otherwise a dev Secret that is never committed) and applies
`deploy/argocd/root-app.yaml`. The placeholder `repoURL`
(`https://github.com/Hitesh-Prajapathi/crypto-volatility-forecasting.git`) in `deploy/argocd/**` must point at
the fork Argo CD can read; `REPO_URL` rewrites it for the root app, but commit the real URL in
`deploy/argocd/apps/*.yaml` so the app-of-apps resolves.

Without Argo CD: `helm upgrade --install volforecast deploy/helm/volforecast -n volforecast -f
deploy/helm/volforecast/values-kind.yaml`. The Postgres/MLflow/MinIO/Prefect backends are not
part of the chart (they are platform dependencies); on kind either install them or point the
`config` values and `networkPolicy.backends` at them.

## 5. Helm values reference

See `deploy/helm/volforecast/README.md` and the commented `values.yaml`. Overlays:
`values-kind.yaml` (small resources, NodePort 30080, ingress `volforecast.localtest.me`,
existing sealed secret, monitoring on) and `values-cloud.yaml` (cert-manager TLS ingress,
ExternalSecret, HPA 3-12, RWX PVC, retrain RBAC).

## 6. CD flow (`.github/workflows/cd.yml`)

1. Push to `main`/`master` or tag `v*`.
2. Per image (`api`, `jobs`): buildx multi-arch (amd64, arm64) with the GHA cache, push to
   `ghcr.io/<owner>/<repo>-api|-jobs` tagged `sha-<short>`, semver, `latest` (default branch).
3. Trivy scan by digest (fail on CRITICAL), SBOM (anchore/syft, SPDX), cosign keyless signature
   (GitHub OIDC), SBOM attestation, and SLSA build-provenance attestation.
4. `gitops-bump` opens a PR that sets `image.tag` in `values-kind.yaml` and `values-cloud.yaml`
   (`sha-<short>`, or the version on tags). Merging it is the deploy; Argo CD syncs. CI never runs
   `kubectl`.

Verify a signature:
`cosign verify ghcr.io/<owner>/<repo>-api@<digest> --certificate-identity-regexp 'https://github.com/<owner>/<repo>/.*' --certificate-oidc-issuer https://token.actions.githubusercontent.com`.

## 7. Rollback

* **Preferred (GitOps)**: `git revert <bump-PR-merge-commit>` and merge. Argo CD syncs the previous
  tag; the Deployment rolls back with `maxUnavailable: 0`.
* **Emergency**: `argocd app rollback volforecast <history-id>` (disable auto-sync first because
  `selfHeal` would revert it: `argocd app set volforecast --sync-policy none`), then revert in git
  and re-enable auto-sync.
* **Model rollback** is separate from image rollback: move the MLflow `@champion` alias back.

## 8. Load test

```bash
k6 run -e BASE_URL=http://localhost:8000 --summary-export=reports/k6-summary.json loadtest/k6/forecast.js
# script validation only (20 s, 10 rps, /healthz):
docker run --rm -v "$PWD/loadtest/k6":/s -e SMOKE=1 -e SMOKE_RPS=10 -e BASE_URL=http://host.docker.internal:8000 grafana/k6 run /s/forecast.js
```

Scenarios: 50 rps for 5 minutes, then a ramp to 300 rps. Thresholds: `http_req_duration`
p(95) < 100 ms, `http_req_failed` rate < 0.5%.

## 9. Security posture

* Non-root (UID/GID 10001), `runAsNonRoot`, read-only root filesystem, `allowPrivilegeEscalation:
  false`, all capabilities dropped, seccomp `RuntimeDefault`; writable paths are an `emptyDir`
  `/tmp` and the data volume only. Compose enforces `read_only`, `cap_drop: ALL`, `no-new-privileges`.
* API pods mount no ServiceAccount token. Only the retrain CronJob gets a token and a Role to
  create Jobs, and only when `rbac.retrainCanCreateJobs` is on.
* NetworkPolicy: namespace default-deny ingress; API accepts 8000 only from `ingress-nginx` and
  `monitoring`; egress limited to DNS and the Postgres/MLflow/MinIO/Prefect backends; jobs add 443
  for market data. (kind's default CNI does not enforce policies; Calico/Cilium do.)
* Supply chain: pinned base images and `uv` version, frozen lockfile, Trivy gate, SPDX SBOM,
  keyless cosign signatures, provenance attestation, GHCR with least-privilege `GITHUB_TOKEN`.
* No plaintext secrets in git: Sealed Secrets (kind) or External Secrets (cloud). The chart's
  `secrets.create` is a dev-only escape hatch and refuses empty values. `.env.example` holds
  dev-only defaults; `.env` must be git-ignored.
