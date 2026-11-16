# LSTM challenger seed runs on a Kaggle GPU

Reproduces the five-seed LSTM evaluation of [ADR-0012](../../docs/adr/0012-lstm-challenger.md)
on a Kaggle T4 GPU instead of a laptop. Everything stays **private**: a private dataset and a
private script kernel.

## What is uploaded

One private dataset (`volforecast-lstm-inputs`):

* `code/` - a clean `git archive` of the repo subset the runner needs (`src/`,
  `configs/backtest_lstm.yaml`, `pyproject.toml`) plus a `CODE_SHA` file. No git history, no
  credentials, no `.env`.
* `daily/symbol=<S>/panel.parquet` + `panel.meta.json` - the precomputed daily realized-measure
  panels (BTCUSDT, ETHUSDT, SOLUSDT) built from public Binance 5-minute bars. About 0.7 MB in
  total; the raw klines are not needed because the backtest runs from the panels
  (`volforecast.pipeline.panel.load_cached_panel` re-hashes each panel and checks it against
  `panel.meta.json`; the hashes equal the ones in the local runs).

## Reproduce

1. Build the staging directories (the upload must come from a committed tree):

   ```bash
   mkdir -p stage/dataset/code && git archive HEAD src configs/backtest_lstm.yaml pyproject.toml \
       | tar -x -C stage/dataset/code
   git rev-parse HEAD > stage/dataset/code/CODE_SHA
   cp -R data/daily stage/dataset/daily          # panels only, no raw klines
   ```

2. Create the private dataset (add a `dataset-metadata.json` with `"isPrivate": true`, title
   `volforecast lstm inputs`, id `<you>/volforecast-lstm-inputs`):

   ```bash
   uvx --from kaggle --with six kaggle datasets create -p stage/dataset --dir-mode zip
   ```

3. Fill in `kernel-metadata.json` (username, dataset slug), copy it next to `run_lstm_seeds.py`
   and push with the GPU:

   ```bash
   mkdir stage/kernel && cp scripts/kaggle/run_lstm_seeds.py scripts/kaggle/kernel-metadata.json stage/kernel/
   uvx --from kaggle --with six kaggle kernels push -p stage/kernel --accelerator NvidiaTeslaT4
   uvx --from kaggle --with six kaggle kernels status <you>/volforecast-lstm-seeds
   uvx --from kaggle --with six kaggle kernels output <you>/volforecast-lstm-seeds -p kaggle_out
   ```

4. Build the reports locally (cheap) and aggregate over seeds:

   ```bash
   for k in 0 1 2 3 4; do
     uv run python -m volforecast report --run kaggle_out/backtest_lstm_s$k --out reports_lstm_s$k
   done
   ```

## What the runner does

`run_lstm_seeds.py` locates the code and panels under `/kaggle/input`, pip-installs only the
dependencies the image lacks (torch is never reinstalled; the kernel needs internet for this),
checks that CUDA is available (it refuses to fall back to CPU) and logs the GPU name. For each
seed it loads `configs/backtest_lstm.yaml`, sets the `lstm` model's `params.seed` to the seed and
`params.device` to `cuda`, and runs the walk-forward backtest for all assets, writing
`/kaggle/working/backtest_lstm_s<k>/{forecasts.parquet,run_meta.json,panels/}`. `run_meta.json`
gets a `kaggle_run` block (GPU name, torch/CUDA versions, wall time); `timings.json` collects the
per-seed timings. The top-level backtest `seed` of the config is left alone; only the LSTM seed
varies. Seeds that already have output are skipped, so a re-run resumes.

The `har`, `log_har_dow` and `xgb` baselines run in every seed's backtest (CPU, deterministic,
seed-independent) because the report's Diebold-Mariano tests need them next to the LSTM.
`--xgb-seeds` can restrict the xgb baseline to some seeds if runtime matters.

## Determinism caveat

CUDA runs are repeatable on one hardware/software stack (cudnn deterministic, `benchmark=False`,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`) but are not bit-identical to CPU runs: reduction order and the
dropout RNG streams differ. Compare GPU and CPU results statistically (across seeds), not by
equality. Hyperparameters are fixed a priori in `configs/backtest_lstm.yaml`.

Local smoke test (CPU, tiny): `uv run python scripts/kaggle/run_lstm_seeds.py --code . --data <data>
--out /tmp/out --device cpu --seeds 0 --assets BTCUSDT --last-origin 2021-02-20
--lstm-param max_epochs=3 --skip-install`.
