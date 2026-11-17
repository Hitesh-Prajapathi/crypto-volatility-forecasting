# 0012. LSTM challenger for h-day realized variance

- Status: Accepted
- Date: 2026-11-17

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

## Results

Five seeds (0-4), expanding-window walk-forward from 2021-01-01, BTC/ETH/SOL, h = 1 and 5, run on a
Kaggle T4 at code SHA 708fbb2 with no failures. Full table, definitions and the commands to
reproduce it: [docs/results/lstm_seeds.md](../results/lstm_seeds.md). QLIKE ratios, mean over seeds
(seed std is 0.003-0.018 in every cell):

| asset | h | LSTM/HAR | LSTM/log_har_dow | LSTM/xgb | DM p vs log_har_dow (seeds < 0.05) | LSTM in MCS 90% |
|---|---|---|---|---|---|---|
| BTCUSDT | 1 | 0.744 | 0.919 | 1.042 | 0.008 (5/5) | 5/5 |
| BTCUSDT | 5 | 0.753 | 0.900 | 0.944 | 0.035 (5/5) | 5/5 |
| ETHUSDT | 1 | 0.687 | 0.895 | 0.967 | 0.024 (5/5) | 5/5 |
| ETHUSDT | 5 | 0.705 | 0.911 | 0.939 | 0.171 (0/5) | 5/5 |
| SOLUSDT | 1 | 0.689 | 0.979 | 1.008 | 0.200 (0/5) | 5/5 |
| SOLUSDT | 5 | 0.744 | 1.071 | 0.968 | 0.973 (0/5) | 1/5 |

- Against HAR the LSTM has 25-31% lower QLIKE in all six cells, significant (one-sided DM p < 0.05) in
  every seed. This is shared with the other challengers: xgb and log_har_dow are also 20-32% and 16-31% below HAR.
- Against log_har_dow, the stronger benchmark, the LSTM is lower in five of six cells (by 2-11%) but
  significantly so only on BTC at h = 1 and h = 5 and ETH at h = 1 (all five seeds each; the BTC h = 5
  and ETH h = 1 mean p-values, 0.035 and 0.024, are close to the threshold). ETH at h = 5 is lower
  but not significant, SOL at h = 1 is a tie, and SOL at h = 5 is **worse** (ratio 1.071).
- Against xgb, the LSTM is better at h = 5 on all three assets (ratios 0.94-0.97) and on ETH at
  h = 1 (0.967), and worse on BTC at h = 1 (1.042) and SOL at h = 1 (1.008). The table does not test
  this comparison, so a supplementary one-sided DM test of lstm vs xgb on the same QLIKE losses (HLN
  correction, per seed, from `forecasts.parquet`) is quoted here. It is not significant on average
  anywhere: the best cells are BTC h = 5 (mean p 0.081, 2/5 seeds below 0.05) and ETH h = 5 (mean
  p 0.077, 3/5 seeds), and on BTC h = 1, where xgb is ahead, the two-sided p is 0.15-0.33. There
  is no multiplicity correction across the six cells.
- Seed spread is small relative to the effect sizes (std/mean at most about 2%); the five seeds
  produce highly correlated forecasts (log-forecast correlation 0.992-0.993 on BTC h = 1).
- The seed-independent models (har, log_har_dow, xgb) have bit-identical forecasts in all five seeds,
  and the LSTM forecasts are finite and positive with the same origins as the other models. A local
  CPU seed-0 run on BTC h = 1 gave 0.742 and 0.916 for LSTM/HAR and LSTM/log_har_dow, consistent
  with the GPU numbers; xgb/HAR there was 0.717 against 0.714 here, a small cross-platform difference.

Caveats: one venue and one test period with a few dominant episodes; six cells; hyperparameters
fixed a priori, not tuned, so the numbers say nothing about a tuned network; the GPU runs are not
bit-identical to CPU runs.

## Decision outcome

The LSTM is a competitive challenger but not a better one. It clearly beats HAR, and it beats
log_har_dow with statistical support on BTC (h = 1, 5) and ETH (h = 1) at gains of about 8-11% in
QLIKE, but not on ETH at h = 5 or SOL, where it ties or loses (SOL at h = 5 is 7% worse). Against xgb
the LSTM is ahead at h = 5 and behind at h = 1 on BTC and SOL, with no significant difference in
either direction, and it is not cheaper than xgb in these runs (115-320 s of walk-forward per asset and horizon on a
T4, against 250-380 s for xgb on four CPU cores) while adding an optional torch dependency. **Production stays
xgb**: nothing here justifies replacing it, and `lstm` is not a retrain candidate. The model stays in
the repository as a documented challenger, evaluated by `configs/backtest_lstm.yaml` and the
five-seed table. A change of this decision would need a result that holds against xgb, for example
a significant h = 5 improvement on a longer sample or with a pre-registered tuning protocol inside the
walk-forward ([ADR-0005](0005-tuning-inside-walk-forward.md)).

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
