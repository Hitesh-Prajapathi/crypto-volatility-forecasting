# Model card

## Intended use

A research and risk-monitoring demonstration of daily realized-volatility forecasting for BTC, ETH
and SOL. It is **not investment advice**, not a trading signal and not a production risk system.
Forecasts are statistical estimates that can be wrong, especially in regime changes and stress.

## Data

- Source: Binance spot 5-minute klines (BTCUSDT, ETHUSDT, SOLUSDT) from data.binance.vision, with
  checksum verification; stored as partitioned Parquet with a manifest of content hashes
  ([ADR-0004](adr/0004-parquet-manifest-over-dvc.md)).
- Completeness policy ([ADR-0003](adr/0003-missing-bar-policy.md)): up to 3 consecutive missing bars
  are forward-filled as zero returns; a day with fewer than 280 of 288 bars or a longer gap is
  incomplete and is excluded from evaluation targets (still allowed as feature history).
- Grid snapping: bars with off-grid `open_time` (for example Feb 2018 BTCUSDT/ETHUSDT) are floored to
  their grid slot, timestamps only; counts are recorded in the manifest.
- Information set: a forecast for day t+1 uses only data up to the end of UTC day t
  ([architecture](architecture.md)).

## Target

Daily realized variance (RV, sum of squared 5-minute log returns over a UTC day). Horizon h=1 is the
next day's RV; h=5 is the cumulative variance over the next five days (`target_rv_h{h}`).
Losses are computed on variance.

## Models evaluated

Baselines configured in `configs/backtest.yaml`: naive, EWMA, EWMA on RV, GARCH and GJR-GARCH
(Student-t), HAR, log-HAR, HAR-CJ, SHAR, HAR-Q; HAR is the benchmark. Gradient boosting and an
ensemble are in progress (M3); deep models are in the plan.

## Evaluation protocol

- Expanding-window walk-forward from 2021-01-01, minimum 365 training days; tuning happens inside the
  training window ([ADR-0005](adr/0005-tuning-inside-walk-forward.md)).
- Point forecasts: QLIKE (primary) and MSE/MAE, Diebold-Mariano tests with the Harvey-Leybourne-Newbold
  correction, Mincer-Zarnowitz regressions, and the Model Confidence Set.
- Risk: VaR (5% and 1%, normal, Student-t and filtered historical simulation) backtests
  (coverage and independence tests); ES backtests.

## Results

Results are generated, not written here. See `reports/` (tables, `results.json`, figures), produced
by `make backtest` then `make report`. This document intentionally contains no numbers.

## Limitations

- Single venue (Binance); other venues and the broader market may differ.
- 5-minute RV is affected by microstructure noise (bid-ask bounce, thin liquidity), more so for
  SOL and in early years.
- Scale gap: RV built from 5-minute returns is not the variance of close-to-close returns (noise,
  jumps and sampling differ), so a VaR built directly from an RV forecast and applied to
  close-to-close returns can be mis-scaled. Calibrated distributions (Student-t, filtered
  historical simulation) address this.
- Regime dependence: performance differs across calm and stressed periods; the evaluation window is
  short and dominated by a few episodes.
- Survivorship: the three assets were chosen because they are large and still listed; results do
  not generalize to assets that failed or have short histories.
- Earlier results of this project were retracted ([ADR-0002](adr/0002-legacy-results-retracted.md)).

## Ethical and financial considerations

Not investment advice and no guarantee of future performance. Do not use the outputs to size real
positions without independent validation. Misread volatility forecasts can lead to under-hedged
risk. The data is public market data; no personal data is used.
