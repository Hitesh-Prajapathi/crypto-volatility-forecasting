# Architecture Decision Records

Decisions are recorded using [the template](0000-template.md). Never delete an ADR; supersede it.

| ADR | Title | Status |
|-----|-------|--------|
| [0001](0001-record-architecture-decisions.md) | Record architecture decisions | Accepted |
| [0002](0002-legacy-results-retracted.md) | Legacy results retracted: audit findings A1-A7 | Accepted |
| [0003](0003-missing-bar-policy.md) | Missing-bar and incomplete-day policy | Accepted |
| [0004](0004-parquet-manifest-over-dvc.md) | Partitioned Parquet + manifest content hashes instead of DVC | Accepted |
| [0005](0005-tuning-inside-walk-forward.md) | Hyperparameters tuned only inside walk-forward training windows | Accepted |
| [0006](0006-mlflow-aliases-over-stages.md) | MLflow aliases (@champion / @challenger) instead of stages | Accepted |
| [0007](0007-cronjob-triggers-prefect-executes.md) | Kubernetes CronJobs trigger, Prefect executes | Accepted |
| [0009](0009-batch-inference-forecast-store.md) | Batch inference + forecast store instead of on-request inference | Accepted |
| [0010](0010-gitops-argocd-pull.md) | GitOps with Argo CD pull instead of CI push | Accepted |

Planned, written in their milestones (see the ADR index in the [execution plan](../execution_plan.md)): 0008 Kubernetes resource choices (HPA/PDB/NetworkPolicy), 0011 Public dashboard on GitHub Pages.
