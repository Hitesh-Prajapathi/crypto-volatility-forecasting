# 0012. LSTM challenger for h-day realized variance

- Status: Accepted
- Date: 2026-11-16

## Context

The plan lists a deep model as a challenger to the HAR family and the gradient-boosted trees. The
legacy LSTM was retracted partly because its scaler was fitted on the full sample
([ADR-0002](0002-legacy-results-retracted.md), item A5), so a new one has to be leakage-safe by
construction. Neural networks are slow to fit, non-deterministic unless handled, and sensitive to
seeds, so the evaluation protocol matters as much as the architecture.

## Decision

We add `lstm` (`volforecast.models.lstm.LSTMVol`) behind the common `VolModel` / `BaseVolModel`
interface:

- Same target as the GBMs and log-HAR: `log(sum_{k=1..h} RV_{t+k})`, one model per horizon, forecast
  `exp(f + sigma^2 / 2)`. `sigma^2` is the mean squared log residual on the early-stopping block, not
  a separately trained out-of-fold model (one network per fit instead of two or three).
- Inputs are sequences of per-day realized measures (log RV, BPV, jump, RS+/RS-, sqrt RQ, returns,
  weekday) over a configurable lookback of 22-66 days (default 22).
- Feature and target standardisation are computed once per `fit` from the training window and stored
  on the instance; `predict_variance` only applies them. A unit test checks that the stored
  statistics equal the training-window statistics and that predict neither changes them nor
  re-estimates them.
- Early stopping uses the chronologically last 15% of the usable training rows, with an h-day
  embargo between the blocks; only minibatches within the training block are shuffled.
- Hyperparameters are fixed a priori (hidden size 32, one layer, dropout 0.1, Adam 1e-3, at most 50
  epochs, patience 5). They are not tuned on the test period: a search against full-sample results
  would be data snooping ([ADR-0005](0005-tuning-inside-walk-forward.md)).
- Determinism: a per-instance `seed` seeds weights, minibatch order and dropout; global torch, numpy
  and python RNG states are restored after each call; deterministic algorithms are requested; torch
  runs on CPU with one thread by default.
- The model is registered lazily in the factory (`torch` is imported only when `lstm` is built) so
  the core package and the serving image do not need the `dl` extra.
- It is not in `configs/backtest.yaml`; `configs/backtest_lstm.yaml` runs it against `har`,
  `log_har_dow` and `xgb`. The pipeline requires unique model names, so every seed is a separate
  backtest run; results are reported as mean and standard deviation over seeds.

## Consequences

+ A deep challenger exists under the same leakage and reproducibility rules as everything else.
+ The default backtest runtime is unchanged.
- Runtime: a refit trains a network (a few seconds), hence `refit_every: 30`.
- On macOS, torch and lightgbm/xgboost ship different OpenMP runtimes. Loading torch before them
  makes later lightgbm/xgboost training abort, and multi-threaded torch inference can hang once they
  are loaded. `volforecast.models.lstm` therefore imports lightgbm and xgboost before torch and runs
  torch inference under the configured thread count (default 1). Test modules that need the LSTM
  must import it through `volforecast.models.lstm`, never `import torch` first.
- `lstm` is deliberately not a production retrain candidate (`optional_prefixes` does not match it).

## Alternatives considered

- Multi-output head for both horizons: couples the horizons' early stopping and diverges from the
  GBM convention of one fit per horizon.
- A separate out-of-fold model for `sigma^2` as in the GBMs: roughly doubles the cost of a refit for a
  second-order correction.
- Hyperparameter search: rejected, see the decision on data snooping above.
- Seed ensembling inside the model: would hide the seed variance that the evaluation should report.
