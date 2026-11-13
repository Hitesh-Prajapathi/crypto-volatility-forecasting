# 0010. GitOps with Argo CD pull instead of CI push

- Status: Accepted
- Date: 2026-11-13

## Context

Deployments to Kubernetes can be pushed from CI (`kubectl` or `helm` with cluster credentials stored in CI) or pulled by an in-cluster controller that reconciles from git.

## Decision

CI builds, tests and publishes signed images and updates the desired image tag in git. Argo CD, running in the cluster, pulls the Helm chart and values from git and reconciles. CI holds no cluster credentials.

## Consequences

- Git is the source of truth; drift is detected and self-healed; rollback is a git revert.
- Smaller attack surface: no cluster credentials in CI.
- One more component to run (Argo CD) and a two-step flow (image tag commit, then sync).

## Alternatives considered

- CI push with `helm upgrade`: simpler to start, but needs cluster credentials in CI and has no drift detection.
- Flux: comparable; Argo CD chosen for its UI and wider familiarity.
