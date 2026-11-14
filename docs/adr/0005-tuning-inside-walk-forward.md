# 0005. Hyperparameters tuned only inside walk-forward training windows

- Status: Accepted
- Date: 2026-11-13

## Context

Tuning hyperparameters on the full sample and then evaluating walk-forward leaks future information into model selection and inflates results. It is the same class of error as the legacy leakage (see [ADR-0002](0002-legacy-results-retracted.md)).

## Decision

Hyperparameter search runs only on data inside each walk-forward training window, with its own time-ordered validation split and an embargo between train and validation. The chosen parameters are then used to forecast the following out-of-sample block. Search budgets and spaces are fixed in configuration before evaluation.

## Consequences

- Reported out-of-sample performance is not contaminated by selection on the test period.
- Higher compute cost: tuning repeats per refit, so search budgets are kept small and refit cadence is configurable.
- Results can differ from a "tune once" run; that difference is the honest one.

## Alternatives considered

- Tune once on the full sample: simple but leaky.
- Tune once on an initial hold-out and freeze: cheaper, but parameters go stale across regime changes.
