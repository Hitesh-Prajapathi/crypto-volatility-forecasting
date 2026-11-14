# ADR-0011: Drift monitoring implemented in-house (scipy) instead of Evidently

- **Status:** Accepted (2026-11-14)

## Context
The plan specified Evidently for feature-drift reports. The current Evidently release imports a private pandas API (`pandas.core.dtypes.common.infer_dtype_from_object`) that was removed in pandas 3, which this project uses. Pinning pandas < 3 would affect every module.

## Decision
Implement drift metrics in `volforecast.monitoring.drift` with numpy/scipy: Population Stability Index and Wasserstein distance per feature (reference = training window, current = trailing 30 days), share of drifted features, and the performance-drift trigger (rolling QLIKE ratio vs HAR). Reports are written as JSON + a static HTML table and exported as Prometheus metrics.

## Consequences
+ No heavy dependency; metrics are transparent and unit-tested against hand computations.
− No Evidently UI; revisit when Evidently supports pandas 3.
