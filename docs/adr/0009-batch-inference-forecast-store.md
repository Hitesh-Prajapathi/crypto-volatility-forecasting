# 0009. Batch inference and a forecast store instead of on-request inference

- Status: Accepted
- Date: 2026-11-13

## Context

The target is a daily forecast of variance for t+1 that depends only on data up to the end of UTC day t. The answer for a given asset and date does not change between requests, and it is read far more often than it is produced.

## Decision

A daily forecast job loads the `@champion` model, computes forecasts for all assets, and writes them to a Postgres forecast store, together with the model version, the data manifest hash and the information-set date. The FastAPI service only reads from the store. It does not load models or compute features on request.

## Consequences

- Read latency is low and predictable; the API is stateless and scales horizontally.
- Every served forecast is auditable and reproducible from stored metadata.
- Forecasts are as fresh as the last job run; freshness is an SLO (see the execution plan, M8).
- No arbitrary what-if inference through the API.

## Alternatives considered

- Load the model in the API and compute on request: heavier images, slower cold starts, duplicated feature logic, no audit trail.
- A separate model server: unnecessary for one forecast per asset per day.
