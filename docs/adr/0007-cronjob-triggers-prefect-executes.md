# 0007. Kubernetes CronJobs trigger, Prefect executes

- Status: Accepted
- Date: 2026-11-13

## Context

Daily ingest, feature build, forecast, evaluation and retrain steps need scheduling, retries, and visibility into runs. Kubernetes offers CronJobs; Prefect offers flow orchestration with retries and a UI.

## Decision

Schedules live in Kubernetes CronJobs, which are the single source of truth for "when". Each CronJob starts a short-lived container that runs one Prefect flow (`ingest`, `forecast`, `evaluate`, `retrain`) to completion. Prefect provides task structure, retries, logging and run history; it does not own the schedule, and no long-running Prefect worker is required in the cluster.

## Consequences

- Scheduling is declarative, lives in the Helm chart, and is reviewed through GitOps.
- Flows are also runnable locally and in Docker Compose without Kubernetes.
- Two systems are involved; the boundary is explicit (K8s decides when, Prefect decides how).

## Alternatives considered

- Prefect deployments with schedules and a worker: another always-on component and a second place where schedules live.
- Bare CronJobs running scripts: no retries or run history.
- Airflow: heavier than needed for a handful of daily flows.
