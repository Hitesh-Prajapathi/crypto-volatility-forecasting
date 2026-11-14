# volforecast Helm chart

Deploys the volatility-forecast API (Deployment, Service, optional Ingress, HPA, PDB), four
scheduled flow CronJobs, NetworkPolicies, config/secret plumbing, and optional Prometheus
Operator resources (ServiceMonitor + PrometheusRule).

```bash
# from the repo root
helm lint deploy/helm/volforecast -f deploy/helm/volforecast/values-kind.yaml
helm template volforecast deploy/helm/volforecast -n volforecast -f deploy/helm/volforecast/values-kind.yaml
helm upgrade --install volforecast deploy/helm/volforecast -n volforecast --create-namespace \
  -f deploy/helm/volforecast/values-kind.yaml
helm test volforecast -n volforecast
```

## Secrets

The pods read `VOLF_DATABASE_URL`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` from one Secret:

| Mode | How |
|------|-----|
| Pre-existing (preferred) | `secrets.existingSecret: <name>` (Sealed Secrets on kind) |
| External Secrets Operator | `externalSecret.enabled: true` + `secretStoreRef` + `data` (cloud) |
| Dev only | `secrets.create: true` + `secrets.dev.*` (never with real values in git) |

## Key values

See the commented [values.yaml](values.yaml). Overlays: [values-kind.yaml](values-kind.yaml),
[values-cloud.yaml](values-cloud.yaml). `image.tag` is bumped by the CD pipeline.

| Value | Default | Notes |
|-------|---------|-------|
| `api.replicas` | 2 | ignored when `autoscaling.enabled` |
| `autoscaling.*` | CPU 60%, 2-8, scale-down window 300 s | requires metrics-server |
| `cronJobs[]` | ingest, forecast, evaluate, retrain | `python -m volforecast.flows.<flow>` |
| `cronJobDefaults.*` | Forbid, backoffLimit 2, deadlines, TTL, `Etc/UTC` | per-entry override |
| `dataVolume.type` | `emptyDir` | `pvc` for persistence (RWX if several API replicas) |
| `networkPolicy.*` | default-deny ingress + per-workload allow rules | `backends` = egress targets |
| `rbac.retrainCanCreateJobs` | false | only the retrain SA may create batch Jobs |
| `monitoring.enabled` | false | needs Prometheus Operator CRDs |
| `ingress.clusterIssuer` | "" | adds the cert-manager annotation |

Changing the ConfigMap changes the `checksum/config` pod annotation, which rolls the Deployment.
