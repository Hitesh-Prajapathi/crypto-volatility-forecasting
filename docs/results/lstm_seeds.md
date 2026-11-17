# LSTM challenger: five-seed walk-forward results

QLIKE ratios of the LSTM challenger ([ADR-0012](../adr/0012-lstm-challenger.md)) against HAR,
log_har_dow and xgb, per asset and horizon, aggregated over five LSTM seeds (0-4).

- Ratio columns are QLIKE(model) / QLIKE(benchmark); below 1 means the model has the lower loss.
  `LSTM/HAR` and `LSTM/log_har_dow` are mean ± sample std (ddof=1) over seeds. `LSTM/xgb` is the
  per-seed ratio of the two QLIKE values. `xgb/HAR` and `log_har_dow/HAR` are seed-independent
  models, so their std must be 0.000.
- `DM p` is the one-sided Diebold-Mariano p-value (HLN-corrected, `alternative="less"` on the QLIKE
  differential: a small p means the LSTM is significantly better than the benchmark), averaged over
  seeds; the bracket counts the seeds with p < 0.05 out of 5. Averaging p-values is descriptive, not
  a test; read it together with the bracket.
- `LSTM in MCS 90%` counts the seeds in which the LSTM is in the 90% model confidence set of
  {har, log_har_dow, xgb, lstm}.
- Setup: expanding-window walk-forward from 2021-01-01 (min train 365 days), BTC/ETH/SOL (Binance
  5-minute realized measures), h = 1 and 5, `configs/backtest_lstm.yaml` (LSTM refit every 30
  origins, hyperparameters fixed a priori). `n` is the number of scored origins.
- Produced on a Kaggle Tesla T4 (torch 2.11, CUDA 13.0), one backtest per seed, 0 failures, code SHA
  708fbb2 (the `code_sha` in `artifacts/kaggle/timings.json`). Total wall time for the five seeds:
  5224 s (about 87 minutes; 1038-1057 s per seed). GPU runs are repeatable on one stack but not
  bit-identical to CPU runs, so compare statistically.

## Reproduce

The raw outputs (`artifacts/kaggle/backtest_lstm_s<k>/`) are gitignored; the Kaggle procedure is in
`scripts/kaggle/README.md`. From those outputs:

```bash
for k in 0 1 2 3 4; do
  uv run python -m volforecast report --run artifacts/kaggle/backtest_lstm_s$k \
      --out artifacts/kaggle/reports_s$k --benchmark2 log_har_dow
done
uv run python scripts/kaggle/aggregate_seeds.py artifacts/kaggle/reports_s{0,1,2,3,4}
```

## Table

| asset | h | n | LSTM/HAR | LSTM/log_har_dow | DM p vs HAR (seeds p<0.05) | DM p vs log_har_dow (seeds p<0.05) | LSTM/xgb | xgb/HAR | log_har_dow/HAR | LSTM in MCS 90% |
|---|---|---|---|---|---|---|---|---|---|---|
| BTCUSDT | 1 | 2135 | 0.744 ± 0.006 | 0.919 ± 0.008 | 0.000 (5/5) | 0.008 (5/5) | 1.042 ± 0.009 | 0.714 ± 0.000 | 0.810 ± 0.000 | 5/5 |
| BTCUSDT | 5 | 2103 | 0.753 ± 0.011 | 0.900 ± 0.013 | 0.000 (5/5) | 0.035 (5/5) | 0.944 ± 0.014 | 0.798 ± 0.000 | 0.837 ± 0.000 | 5/5 |
| ETHUSDT | 1 | 2135 | 0.687 ± 0.005 | 0.895 ± 0.007 | 0.000 (5/5) | 0.024 (5/5) | 0.967 ± 0.007 | 0.710 ± 0.000 | 0.768 ± 0.000 | 5/5 |
| ETHUSDT | 5 | 2103 | 0.705 ± 0.014 | 0.911 ± 0.018 | 0.000 (5/5) | 0.171 (0/5) | 0.939 ± 0.018 | 0.751 ± 0.000 | 0.774 ± 0.000 | 5/5 |
| SOLUSDT | 1 | 1917 | 0.689 ± 0.005 | 0.979 ± 0.008 | 0.000 (5/5) | 0.200 (0/5) | 1.008 ± 0.008 | 0.684 ± 0.000 | 0.704 ± 0.000 | 5/5 |
| SOLUSDT | 5 | 1903 | 0.744 ± 0.003 | 1.071 ± 0.004 | 0.000 (5/5) | 0.973 (0/5) | 0.968 ± 0.004 | 0.768 ± 0.000 | 0.695 ± 0.000 | 1/5 |
