# 0006. MLflow aliases (@champion / @challenger) instead of stages

- Status: Accepted
- Date: 2026-11-13

## Context

The MLflow Model Registry stages (Staging, Production, Archived) are deprecated in favour of model version aliases and tags. We need a promotion mechanism and a trivial rollback.

## Decision

Use registered-model aliases: `@champion` marks the model serving forecasts, `@challenger` marks a candidate under evaluation. Promotion is a gated step that moves the `@champion` alias when the challenger passes the evaluation criteria. Consumers load models by alias (`models:/<name>@champion`). Rollback means moving the alias back.

## Consequences

- Uses the supported API; no dependence on deprecated stages.
- Rollback is a single alias move and is documented in the runbook.
- Aliases are mutable pointers, so the promotion decision and metrics are also logged as tags for auditability.

## Alternatives considered

- Stages: deprecated.
- Tags only with custom lookup logic: reinvents aliases.
- Model files in object storage with a naming convention: loses registry lineage.
