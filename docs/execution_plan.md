# Execution Plan — Crypto Volatility Forecasting → Production MLOps System

> Companion to the project goals document (the *what*; kept outside this repo). This file is the *how*: ordered milestones, PR-sized tasks, file layout, formulas, acceptance criteria, and the decisions behind them.
> Repo: `crypto-volatility-forecasting/` (git, remote `origin`, branch `master`, 1 commit).

---

## 0. Audit of the current code (read this first)

These were found by reading the code and the saved artifacts (13 Nov 2026). Several of them **invalidate the current results**. An interviewer who opens the repo will find them, so fixing them comes before any new feature work.

| # | Severity | Finding | Where |
|---|----------|---------|-------|
| A1 | 🔴 Critical | **Target leakage.** The target is `rv_30d`, but the feature set still contains `rv_30d_ann` (= `rv_30d × √252`, correlation 1.000) and `var_30d` (= `rv_30d²`, correlation 1.000). `get_feature_columns` excludes only `TARGET_COLUMN`. The tree models are reading the answer off a rescaled copy of it. That explains RF MAE = 1.9e-5 / MAPE 0.12% in `models/metrics.csv`. | `src/features.py` (`get_feature_columns`), `src/preprocessing.py` |
| A2 | 🔴 Critical | **It is not a forecast.** Features at row *t* predict `rv_30d` at row *t*, and that target is built from returns up to and including *t*. The target is never shifted forward, so this is nowcasting a lagging statistic. | `src/train.py` step 4 |
| A3 | 🔴 Critical | **The published numbers can't be reproduced.** The README table says XGBoost is best (MAE 0.001234, MAPE 8.45%). `models/metrics.csv` says Random Forest is best (MAE 0.0000188). The README numbers don't come from any artifact. | `README.md:219-224` vs `models/metrics.csv` |
| A4 | 🟠 High | **GARCH is compared against the wrong target.** GARCH's 1-day conditional σ is scored against a 30-day rolling std. It is also aligned with `train_size` taken from `df_feat`, while it indexes `df_proc`, which has about 20 more rows, so the test windows are offset. | `src/train.py` 5a, `src/models.py` |
| A5 | 🟠 High | **LSTM scaler leakage.** `MinMaxScaler` is fit on the full dataset, test set included. `forecast.py` refits new scalers at inference, so the scaling no longer matches what the model was trained on. | `src/train.py` 5d, `scripts/forecast.py` |
| A6 | 🟠 High | **`bfill()` on `log_var_30d`** copies future values backward. | `src/preprocessing.py` |
| A7 | 🟡 Medium | A 30-day rolling-std target is so smooth that persistence alone scores very well. It is the wrong target (see Phase 1). Crypto trades 365 days a year, but the code annualizes with 252. | `src/config.py` |
| A8 | 🟡 Medium | `/predict` returns a slice of stored **test-set** predictions, not a forecast. `/train` runs a multi-minute pipeline inside an HTTP request. CORS is `*`. It uses the deprecated `@app.on_event`. There are no response schemas. | `api/main.py` |
| A9 | 🟡 Medium | No tests, no lockfile, unpinned dependencies. Both `torch` and `tensorflow` are installed (TF is unused). `pmdarima` is unused. `sys.path.insert` hacks everywhere. `print` instead of logging. Two launchers (`launch.py`, 414 lines, plus `scripts/launch.py`). | repo-wide |
| A10 | 🟡 Medium | `.gitignore` excludes `*.pkl` but not `*.pt` and `*.joblib`. Model binaries are mixed into the source tree. | `.gitignore`, `models/` |

**What this means for the resume.** None of the current numbers can go on a resume. Phase 0 removes them. Phase 2 produces the first numbers you can defend.

---

## 1. Guiding principles (the rules every PR follows)

1. **No hand-typed numbers.** Every table in the README and docs is generated from a results artifact by `make report`. This prevents a repeat of A3.
2. **Strict information sets.** A forecast for day *t+1* may use only data timestamped ≤ end of day *t* (UTC). A test enforces this (§P1.6).
3. **Every model predicts the same thing:** variance of the next *h* days. Each model reports through one interface, so the evaluation code never needs model-specific branches.
4. **Baselines first.** HAR-RV is the benchmark everything is measured against. If HAR wins, the README says so.
5. **Training is a job, not an endpoint.** The API serves forecasts and nothing else. Jobs train, evaluate, and promote models.
6. **Everything runs locally first** (docker compose, then kind) before any cloud deployment. Cloud deployment is optional, has a cost cap, and can be torn down.

---

## 2. Target repository layout

```
crypto-volatility-forecasting/
├── pyproject.toml            # uv-managed; ruff, mypy, pytest config
├── uv.lock
├── Makefile                  # setup, lint, test, data, backtest, report, up, kind-up, deploy
├── configs/
│   ├── base.yaml             # paths, horizons, walk-forward params, VaR levels
│   ├── assets/{btc,eth,sol}.yaml
│   └── models/{har,garch,lgbm,lstm,patchtst}.yaml
├── src/volforecast/
│   ├── config.py             # pydantic-settings, loads YAML + env overrides
│   ├── logging.py            # structlog JSON logging
│   ├── data/
│   │   ├── binance.py        # data.binance.vision downloader + CHECKSUM verification
│   │   ├── schemas.py        # Pandera schemas (klines, daily RV, features)
│   │   └── store.py          # partitioned Parquet + manifest (sha256), DuckDB reads
│   ├── features/
│   │   ├── realized.py       # RV, bipower variation, jumps, semivariances, quarticity
│   │   ├── har.py            # HAR d/w/m aggregates (+ log variants)
│   │   └── exogenous.py      # Deribit DVOL, funding rate, open interest (optional)
│   ├── models/
│   │   ├── base.py           # VolModel Protocol: fit(), predict_variance(h)
│   │   ├── naive.py  ewma.py  garch.py  har.py
│   │   ├── gbm.py            # LightGBM/XGBoost on HAR+ features
│   │   ├── lstm.py  patchtst.py
│   │   └── ensemble.py       # forecast combination (equal / inverse-QLIKE weights)
│   ├── evaluation/
│   │   ├── walkforward.py    # expanding/rolling engine, refit cadence, embargo
│   │   ├── losses.py         # QLIKE, MSE, MAE on variance
│   │   ├── stats_tests.py    # Diebold-Mariano (HAC + HLN), Mincer-Zarnowitz, MCS
│   │   ├── var_backtest.py   # VaR/ES, Kupiec, Christoffersen, Basel traffic light
│   │   └── report.py         # results JSON → markdown tables + figures
│   ├── registry/
│   │   ├── tracking.py       # MLflow logging helpers
│   │   └── promotion.py      # champion/challenger gate
│   ├── monitoring/
│   │   ├── drift.py          # Evidently reports + breach logic
│   │   └── prom.py           # Prometheus metric definitions
│   ├── flows/                # Prefect flows
│   │   ├── ingest.py  build_features.py  forecast.py  evaluate.py  retrain.py
│   └── serving/
│       ├── app.py            # FastAPI app factory, lifespan
│       ├── schemas.py        # pydantic v2 request/response models
│       └── store.py          # reads latest forecasts (Parquet/Postgres)
├── tests/{unit,property,integration,e2e}/  + tests/fixtures/btc_5m_sample.parquet
├── docker/{api.Dockerfile, jobs.Dockerfile}
├── compose.yaml              # api, mlflow, postgres, minio, prefect, prometheus, grafana
├── deploy/
│   ├── helm/volforecast/     # chart: api, cronjobs, configmap, hpa, pdb, netpol, servicemonitor
│   ├── helm/values-{kind,cloud}.yaml
│   ├── argocd/               # Application manifests (app-of-apps)
│   ├── kind/cluster.yaml
│   └── monitoring/{dashboards/*.json, alerts/*.yaml}
├── loadtest/k6/forecast.js
├── docs/{architecture.md, model_card.md, runbook.md, adr/NNNN-*.md}
├── reports/                  # GENERATED: results.json, tables.md, figures/
└── .github/workflows/{ci.yml, cd.yml, nightly.yml, pages.yml}
```

---

## 3. Milestones and time estimates

Estimates assume about 15–20 hours a week. Each milestone ships as one or more PRs into `main`.

| Milestone | Theme | Est. | Leaves the project in a state that is… |
|-----------|-------|------|------------------------------------|
| **M0** | Stop the bleeding + repo foundation | 3–4 d | honest (no false claims), tested, lintable |
| **M1** | Intraday data + realized measures | 4–5 d | built on the correct target |
| **M2** | Baselines + walk-forward + statistical tests | 6–8 d | **interview-defensible** (first resume numbers) |
| **M3** | ML/DL challengers + ensemble + regime analysis | 5–7 d | complete as research |
| **M4** | VaR / ES risk application | 3–4 d | "real-world" for risk teams |
| **M5** | MLOps: MLflow, Prefect, data versioning, drift | 6–8 d | an automated ML system |
| **M6** | Serving hardening + local stack | 3–4 d | production-shaped API |
| **M7** | CI/CD + Kubernetes + Helm + GitOps | 7–10 d | **backing for the K8s/CI/CD skills** |
| **M8** | Observability, load test, SRE docs | 4–5 d | operable |
| **M9** | Presentation + live dashboard | 3–4 d | ready to share |

Total: about 7–9 weeks. **M2 is the minimum interview-ready state.** If time runs short, ship M0–M2 + M4 + a slim M7 (CI + Helm on kind) before anything else.

---

## M0 — Stop the bleeding + repo foundation

**Branch:** `refine/m0-foundation`

### Tasks
1. **Retract the false numbers.** Delete the results table in `README.md`. Replace it with "Results being regenerated under walk-forward evaluation; see `reports/`". Remove `models/metrics.csv` and `models/predictions.joblib` from git.
2. **Tag the old state:** `git tag v0-legacy` so the original work is preserved and can be referenced.
3. **Packaging:** move to `src/volforecast/` with `pyproject.toml` managed by **uv**, and commit `uv.lock`. Remove every `sys.path.insert`. Drop `tensorflow`, `pmdarima`, and `yfinance` (yfinance stays only for an optional sanity cross-check). Install the **CPU-only torch** wheel in Docker (the image goes from several GB to about 1 GB).
4. **Tooling:** `ruff` (lint + format), `mypy --strict` on `features/`, `evaluation/`, and `models/base.py`, `pytest` + `pytest-cov`, `hypothesis`, and `pre-commit` (ruff, mypy, gitleaks, end-of-file fixes, `nbstripout` for notebooks).
5. **Config:** a `pydantic-settings` `Settings` loaded from `configs/*.yaml` and `VOLF_*` env vars. Remove the global `os.environ` mutation from `config.py` and set thread env vars in the Dockerfile and entrypoints instead.
6. **Logging:** `structlog` JSON logs carrying `run_id`, `asset`, `model`, and `horizon`. No emoji prints in library code.
7. **Artifacts out of the source tree:** `models/`, `data/`, and `figures/` go into `.gitignore` in full. Artifacts live in MLflow/MinIO (M5) and local `artifacts/` until then.
8. **Makefile** targets: `setup lint typecheck test data backtest report up down`.
9. Delete `launch.py`, `launch.sh`, and `scripts/launch.py`; `make up` (docker compose, M6) replaces them. Archive the notebooks to `notebooks/archive/`.
10. **`docs/adr/0001-record-architecture-decisions.md`**, plus ADR-0002 "Legacy results retracted: target leakage". Writing down that you found and fixed your own leakage is a strong interview story.

### Acceptance
- `make lint typecheck test` passes from a clean clone with `uv sync`.
- The README makes no performance claims.
- `git grep -n "sys.path.insert"` returns nothing.

---

## M1 — Intraday data and realized measures (correct target)

**Branch:** `refine/m1-intraday-rv`

### P1.1 Ingestion: `data/binance.py`
- Source: `https://data.binance.vision/data/spot/{monthly,daily}/klines/{SYMBOL}/5m/`. Use monthly zips for history and daily zips for the trailing month.
- Symbols: `BTCUSDT` (from 2017-08), `ETHUSDT`, `SOLUSDT` (from 2020-08).
- **Verify the `.CHECKSUM` (sha256) file for every zip** and refuse unverified data.
- Idempotent and resumable. Write to `data/raw/klines/symbol=BTCUSDT/interval=5m/year=YYYY/month=MM/part.parquet`.
- Normalize timestamps: Binance spot switched kline timestamps from milliseconds to **microseconds starting 2025-01-01**. Detect the unit per file and normalize to UTC `datetime64[ns]`. Add a unit test for this specific case.
- **Fallback / cross-check source:** Binance's public site may be geo-restricted in some regions. Add a pluggable `Source` interface with a second implementation for Coinbase Exchange public candles (5-min, paginated). Use it once in M2 to show that RV from the two venues correlates above 0.95. That doubles as a robustness result.

### P1.2 Validation: `data/schemas.py` (Pandera)
- Klines: monotonic unique `open_time`, 5-min spacing, `high ≥ max(open, close)`, `low ≤ min(open, close)`, `volume ≥ 0`, price > 0.
- **Completeness:** a day with fewer than 280/288 bars is flagged. Policy: forward-fill the price for gaps of 3 bars or fewer (which contributes zero return); otherwise mark the day `incomplete` and **exclude it from the evaluation target**, while still allowing it as model input. Record the policy in ADR-0003. Expect to see the known Binance outages (for example 2021-05-19) in the incomplete-days report.
- Daily-RV schema: `rv > 0`, no NaNs, an extreme-value report (z-score of log-RV above 6 is logged, never silently clipped).

### P1.3 Realized measures: `features/realized.py`
Days run 00:00–24:00 UTC. With 5-min log returns $r_{t,i}$, $i = 1..M$, $M = 288$:
- **Realized variance:** $RV_t = \sum_i r_{t,i}^2$. **The forecast target is variance.** Report volatility = √RV, annualized with √365.
- **Bipower variation** (robust to jumps): $BPV_t = \frac{\pi}{2}\sum_{i=2}^{M} |r_{t,i}||r_{t,i-1}|$, and the jump component $J_t = \max(RV_t - BPV_t, 0)$.
- **Realized semivariances:** $RS^+_t = \sum r^2 \mathbb{1}[r>0]$ and $RS^-_t = \sum r^2 \mathbb{1}[r<0]$, used by the SHAR model.
- **Realized quarticity:** $RQ_t = \frac{M}{3}\sum r^4$, used by the HARQ model.
- Optional: a sampling-frequency **signature plot** (1/5/15/30-min RV) to justify the choice of 5-min sampling. It makes a good README figure.

### P1.4 Targets and horizons
- $h = 1$: target $RV_{t+1}$.
- $h = 5$: target $RV_{t+1:t+5} = \sum_{k=1}^{5} RV_{t+k}$ (cumulative variance, which is what a 5-day VaR needs).
- The 1-hour horizon is deferred (non-goal for v1, listed under Limitations).

### P1.5 Storage: `data/store.py`
- Partitioned Parquet, with a `manifest.json` per dataset build: source file checksums, row counts, date range, code git SHA, and a **content hash** (sha256 of the sorted canonical Parquet).
- `DuckDB` is the read layer (`SELECT ... FROM read_parquet('data/**/*.parquet')`).
- ADR-0004 explains why this setup is used instead of DVC: it is simpler, cloud-portable, and the hashes provide lineage. Add DVC with a MinIO remote only if there is time.

### P1.6 Leakage guard tests (the most valuable tests in the repo)
- **Truncation-invariance test:** for random cut points *t*, `build_features(data[:t]).iloc[-1] == build_features(data).loc[t]`. If any feature looks ahead, this test fails.
- **Target-alignment test:** `target[t] == RV[t+1]` exactly, for each horizon.
- **Hypothesis property tests:** RV ≥ 0, BPV ≥ 0, $RS^+ + RS^- = RV$, and RV is invariant to price scaling.

### Acceptance
- `make data ASSET=btc` builds daily RV from 2017-08 to yesterday, with a manifest and an incomplete-day report.
- All P1.6 tests pass. The fixture is a committed sample of about 60 days × 288 bars (around 1 MB).

---

## M2 — Baselines, walk-forward evaluation, statistical tests

**Branch:** `refine/m2-evaluation`. **This milestone produces the first resume-grade numbers.**

### P2.1 The model interface: `models/base.py`
```python
class VolModel(Protocol):
    name: str
    def fit(self, history: pd.DataFrame) -> "VolModel": ...        # history: daily panel up to t
    def predict_variance(self, history: pd.DataFrame, h: int) -> float:  # forecast of RV_{t+1:t+h}
    def get_params(self) -> dict[str, Any]: ...
```

### P2.2 Baselines
| Model | Specification |
|-------|---------------|
| Naive | $\hat{RV}_{t+1} = RV_t$; for h=5: $5 \cdot RV_t$ |
| EWMA (RiskMetrics) | $\sigma^2_{t+1} = \lambda\sigma^2_t + (1-\lambda) r_t^2$ with λ = 0.94. **Also an EWMA on RV** (smoothing applied to RV instead of squared returns), because it is the fairer benchmark. |
| GARCH(1,1) | `arch`, daily close-to-close returns ×100, Student-t errors. h-step variance from `res.forecast(horizon=h)` summed. |
| GJR-GARCH(1,1,1) | `o=1`, captures leverage/asymmetry |
| **HAR-RV** | OLS: $RV_{t+1} = \beta_0 + \beta_d RV_t + \beta_w \overline{RV}_{t-4:t} + \beta_m \overline{RV}_{t-21:t} + \varepsilon$ |
| log-HAR | Same model on log RV. Bias-correct when going back to levels: $\exp(\hat\mu + \hat\sigma^2/2)$ |
| HAR-J / HAR-CJ, SHAR, **HARQ** | The extended HAR family, using the BPV, semivariance, and RQ features from M1. These are the strongest econometric baselines. HARQ (Bollerslev, Patton & Quaedvlieg 2016) is often hard to beat. |

The GARCH models are **refit weekly**. Between refits, the filter updates daily with fixed parameters. The engine supports this through `refit_every`. The cost drops from ~2,500 fits to ~360 per asset, and the setting is recorded.

### P2.3 Walk-forward engine: `evaluation/walkforward.py`
- **Expanding window** is the default; **rolling window** (for example 3 years) is the robustness variant.
- First forecast origin: 2020-01-01. That leaves about 2.3 years of burn-in for BTC, and the out-of-sample period includes COVID-2020, the 2021 bull run, LUNA, FTX, and the 2024 ETF period.
- `refit_every`: 1 day for HAR (cheap OLS), 7 days for GARCH and GBMs, 30 days for DL. All are configurable.
- **Embargo** for h=5: when fitting at origin *t*, drop the last *h−1* training rows, whose targets overlap the future.
- **Nested tuning only:** hyperparameters are tuned with Optuna **inside** each training window, using a time-ordered inner validation split, or tuned once on the burn-in period and frozen. The test window is never used for tuning. Record the choice in ADR-0005.
- Output: a long-format `forecasts.parquet` with columns `(asset, model, horizon, origin_date, target_date, forecast_var, realized_var, model_version, data_hash)`. This one table drives all evaluation, VaR, the dashboard, and the regression gate.
- Parallelize across `(asset, model, horizon)` with `joblib`. Enable caching so reruns only compute new origins.

### P2.4 Losses and tests: `evaluation/losses.py`, `stats_tests.py`
- **QLIKE (primary):** $L(RV, h) = \frac{RV}{h} - \ln\frac{RV}{h} - 1$, the normalized form from Patton (2011), always on **variance**. It is robust to a noisy proxy, and a perfect forecast scores 0.
- **MSE on variance** (also robust under Patton 2011) and MAE on volatility, for readability. **MAPE is dropped**, with a footnote explaining why.
- **Mincer-Zarnowitz:** regress $RV_{t+1} = a + b\,\hat h_t + u$. Report R² and a joint Wald test of $(a, b) = (0, 1)$ with **Newey-West HAC** standard errors.
- **Diebold-Mariano** against HAR-RV: $d_t = L^{model}_t - L^{HAR}_t$, HAC variance with h−1 lags (Bartlett kernel), and the **Harvey-Leybourne-Newbold** small-sample correction. Report a one-sided p-value for "model beats HAR".
- **Model Confidence Set** via `arch.bootstrap.MCS` (block bootstrap, 90% level). Report which models make the superior set. It handles the multiple-comparisons question better than pairwise DM tests.
- **Sub-period table:** QLIKE by year and by **volatility regime** (terciles of trailing 22-day RV). Interviewers often ask whether a model only wins in calm markets.

### P2.5 Reporting: `evaluation/report.py`
- `make report` reads `forecasts.parquet` and writes `reports/results.json` and `reports/tables.md`: the QLIKE ratio vs HAR, DM p-value, MZ R², and MCS membership for each asset × horizon × model. Figures: forecast vs realized for 2022 (LUNA + FTX) and cumulative QLIKE difference vs HAR over time.
- The README embeds `reports/tables.md` between markers (`<!-- RESULTS:START -->`). Only the script writes that section.

### Acceptance
- `make backtest ASSET=btc HORIZON=1,5` finishes in under 20 minutes on a laptop for the baselines.
- The DM and MZ implementations are **cross-checked against a reference**: MZ against `statsmodels` OLS with HAC, and DM against a hand-computed small example. Each check has a unit test.
- `reports/tables.md` exists and is the only source of the numbers.

---

## M3 — ML/DL challengers, ensemble, interpretation

**Branch:** `refine/m3-challengers`

1. **LightGBM (primary GBM) and XGBoost** on the HAR+ features (lags of RV, BPV, J, RS±, RQ, and log versions), plus:
   - return-based features: |r_t|, r_t (leverage), and a weekday dummy (weekends matter in crypto);
   - **optional exogenous features (strong differentiator):** **Deribit DVOL** (the 30-day implied-vol index, from the public API) as a forward-looking signal, plus Binance perpetual **funding rate** and **open interest** from `data.binance.vision/data/futures/um/`. Always lag them to respect the information set. Run an ablation with and without them.
   - Fit on **log RV** with an MSE objective and bias-correct back to levels. Also try a custom QLIKE-like objective, and report whether it helps.
2. **LSTM:** sequences of the daily realized-measure vector (lookback 22–66 days). The scaler is fit **per training window only**, and the scaler is saved with the model. Early stopping uses the inner validation split. Seeds are fixed and the model runs 5 times; report mean ± std of QLIKE, since DL variance between runs is part of the honest story.
3. **PatchTST (optional):** via `neuralforecast` or a small custom implementation. Refit monthly. Include it only if it runs within the compute budget; otherwise list it as future work.
4. **Forecast combination:** an equal-weight mean of the HAR family + GBM, and an inverse-trailing-QLIKE weighted combination. Combinations often beat every individual model, which is a credible headline result.
5. **Panel / cross-asset model:** one GBM pooled across BTC, ETH, and SOL with an asset id. Test whether pooling helps the shorter SOL history.
6. **Interpretation:** SHAP summary for the GBM. Check whether its top features line up with the HAR d/w/m structure, which ties the ML result back to the econometrics.

### Acceptance
- Every challenger goes through the same engine and produces rows in `forecasts.parquet`.
- The results table includes all models for BTC h=1/h=5 and at least ETH h=1.
- The README "Findings" paragraph is written whatever the outcome, **including "HARQ is not beaten significantly"** if that is what the data shows.

---

## M4 — Risk application: VaR / ES backtesting

**Branch:** `refine/m4-var`

1. **1-day VaR** at 95% and 99% from each model's variance forecast:
   $VaR_{\alpha,t+1} = -(\hat\mu + q_\alpha \sqrt{\hat h_{t+1}})$, where $q_\alpha$ comes from:
   - (a) a Gaussian;
   - (b) a **Student-t** with ν estimated on standardized in-window returns $r_t/\sqrt{\hat h_t}$;
   - (c) **filtered historical simulation**: the empirical quantile of standardized in-window residuals.
   Returns are daily close-to-close at 00:00 UTC, matching the RV day.
2. **Expected Shortfall** (97.5%, the current Basel FRTB standard) from (b) and (c).
3. **Backtests** in `var_backtest.py`:
   - **Kupiec POF** (unconditional coverage, LR ~ χ²(1));
   - **Christoffersen independence** and **conditional coverage** (χ²(1), χ²(2));
   - **Basel traffic light** (green/yellow/red zones over 250-day windows);
   - optional: the **Acerbi-Székely Z2** ES backtest, or a **dynamic quantile** test (Engle-Manganelli).
4. **Output:** a table per model with breaches vs expected, p-values, and traffic-light zone. Plot returns with the VaR bands and breach markers for 2022.
5. **Tests:** cross-check the LR statistics against hand-computed textbook examples.

### Acceptance
- A VaR table exists for BTC with all baselines + the best ML model + the ensemble. The README resume bullet is filled in from it automatically.

---

## M5 — MLOps: tracking, registry, orchestration, data versioning, drift

**Branch:** `refine/m5-mlops`

### P5.1 MLflow (tracking + registry)
- Self-hosted in compose and K8s: **Postgres** backend store and **MinIO** (S3-compatible) artifact store.
- Each walk-forward run logs: the config, git SHA, data content hash, per-period losses, the summary table, plots, and the `forecasts.parquet` artifact.
- Each production model is logged as an **MLflow pyfunc** with a signature and an input example. Registered names follow `volforecast-{asset}-h{h}` (for example `volforecast-btc-h1`). The pyfunc packages the feature builder, so serving and training share the same code (no training/serving skew).
- **Use aliases, not stages.** MLflow deprecated model stages (`Staging`/`Production`) in 2.9. Use `@challenger` and `@champion` aliases instead. Note this in ADR-0006; it shows you are current.

### P5.2 Champion/challenger promotion: `registry/promotion.py`
A challenger is promoted to `@champion` only if **all** of these hold on the latest evaluation window (last 365 days, walk-forward):
1. QLIKE(challenger) ≤ QLIKE(champion), **and** the DM test against the champion is not significantly worse (p > 0.10 for "worse");
2. QLIKE(challenger) < QLIKE(HAR-RV) − ε (it must stay ahead of the baseline);
3. 99% VaR passes Kupiec and Christoffersen at 5% and stays in the Basel green zone;
4. the data validation suite passed, and the forecast's sanity checks passed (positive, finite, within 10× of trailing RV range).

Every decision, including rejections, is logged as an MLflow run tagged `promotion_decision` with the reasons. Optionally, run the challenger in **shadow mode** for 14 days (it produces forecasts without serving them) before it can be promoted.

### P5.3 Orchestration (Prefect 3)
Flows (each a thin wrapper over library functions, so they are testable without Prefect):
| Flow | Schedule (UTC) | Steps |
|------|----------------|-------|
| `ingest` | daily 00:20 | download yesterday's daily zip → checksum → validate → append partition → update manifest |
| `build_features` | after ingest | recompute the last 66 days of realized measures → validate |
| `forecast` | after features | load `@champion` (+ `@challenger` shadow) → forecast h=1,5 → VaR → write to the forecast store |
| `evaluate` | daily 01:00 | score yesterday's forecasts against realized RV → update rolling QLIKE, VaR breach log → drift report |
| `retrain` | weekly Sun 02:00 **+ triggered** | walk-forward refresh on latest window → register challenger → promotion gate |

**Decision (ADR-0007): "Kubernetes CronJobs trigger, Prefect executes."** The K8s CronJobs (M7) are the schedulers and run `python -m volforecast.flows.<flow>`. The flows report to a self-hosted Prefect server for UI, retries, task-level state, and run history. The drift-triggered retrain calls the `retrain` flow from `evaluate`. This avoids having two schedulers (Prefect deployments and CronJobs) that can disagree, and it exercises both skills properly. Retries: 3 with exponential backoff on network tasks, plus idempotency keys on the date partition.

### P5.4 Drift monitoring (Evidently)
- **Feature drift:** reference = training window; current = trailing 30 days. Per-feature Wasserstein / PSI on log-RV features. Drift is expected in a regime change, so a drift alert alone **does not** retrain.
- **Performance drift (the real trigger):** rolling 30-day QLIKE of the champion vs HAR. Trigger a retrain if the champion's QLIKE ratio vs HAR goes above 1.0 for 10 consecutive days, **or** if the 99% VaR breaches exceed the Basel yellow-zone rate over the trailing 250 days.
- Save the HTML reports to MinIO and link them from Grafana. Export summary numbers as Prometheus metrics (M8).

### P5.5 Forecast store
- A Postgres table `forecasts(asset, horizon, model_version, origin_date, target_date, forecast_var, var95, var99, es975, created_at)` with a unique key on `(asset, horizon, model_version, origin_date)`, so writes are idempotent upserts.
- `realized(asset, date, rv, bpv, rs_pos, rs_neg, rq, complete)`.

### Acceptance
- On `docker compose up`, a full daily cycle (ingest → forecast → evaluate) runs end to end against live Binance data, with lineage visible in MLflow and run history in Prefect.
- Forcing a performance breach (a test fixture with a degraded champion) triggers retraining and a logged promotion decision.

---

## M6 — Serving hardening and local stack

**Branch:** `refine/m6-serving`

### API: FastAPI, `/v1`, pydantic v2 schemas
| Endpoint | Purpose |
|----------|---------|
| `GET /v1/forecast?asset=btc&horizon=1` | Latest champion forecast: `forecast_var`, `forecast_vol_ann`, `var95`, `var99`, `es975`, `model_version`, `origin_date`, `data_as_of` |
| `GET /v1/forecast/history?asset=btc&horizon=1&start=&end=` | Time series of forecasts vs realized, paginated |
| `GET /v1/models` | Champion/challenger versions and their latest evaluation metrics |
| `GET /v1/backtest/summary` | The generated results table (read from `reports/results.json`) |
| `GET /healthz` | Liveness, process only |
| `GET /readyz` | Readiness: DB reachable, champion model loaded, data freshness < 26 h |
| `GET /metrics` | Prometheus |

- **`/train` is removed** (principle 5). Retraining goes through the retrain flow, which holds the credentials.
- Use the `lifespan` context instead of `on_event`. Load the champion model at startup and poll the alias every 5 minutes for **hot reload** with zero downtime.
- Errors use RFC 7807 problem+json. The CORS allowlist comes from config. **Rate limiting** (slowapi) on the public endpoints. Response caching: forecasts change once a day, so serve `ETag` / `Cache-Control: max-age=300`.
- Run `uvicorn` with `--workers` set by env var, or gunicorn + uvicorn workers. Graceful shutdown.
- Optional: **OpenTelemetry** tracing (FastAPI + SQLAlchemy instrumentation), exported to Grafana Tempo or Jaeger in compose.

### Containers
- `docker/api.Dockerfile` and `docker/jobs.Dockerfile`: **multi-stage** builds, `python:3.12-slim`, uv to install from the lockfile, **non-root UID 10001**, read-only filesystem compatible, `HEALTHCHECK`, and OCI labels (`org.opencontainers.image.revision`). The API image does not include torch unless the champion needs it; serve DL models from the jobs image or a separate image.
- `compose.yaml`: api, dashboard, mlflow, postgres, minio (+ bucket init), prefect-server, prometheus, grafana (provisioned dashboards), and a one-shot `jobs` service. `make up` starts the whole platform.

### Dashboard
- Rewrite Streamlit against `/v1` only (no direct file reads): forecast vs realized, VaR bands with breaches, model leaderboard, and drift status.

### Acceptance
- `make up && make smoke` passes (smoke test: health, ready, forecast schema validation).
- Integration tests run the API against a Postgres testcontainer.

---

## M7 — CI/CD, Kubernetes, Helm, GitOps

**Branch:** `refine/m7-platform`. This milestone is the evidence for the K8s and CI/CD skills, so every manifest has to be justified (ADR-0008).

### P7.1 GitHub Actions
**`ci.yml`** (on PR and push):
1. `lint` (ruff), `typecheck` (mypy), and `test` (pytest with a coverage gate of 80% on `features/` and `evaluation/`). uv cache. Python 3.12 matrix (+3.13 allowed to fail).
2. `security`: `gitleaks`, `pip-audit`, and `trivy fs`.
3. **`model-regression-gate`:** run walk-forward on the **frozen fixture snapshot** (committed ~1 year of daily realized measures, which needs no network) for HAR + the current champion's model class. Assert that (a) the HAR QLIKE matches a stored golden value to 1e-8 (determinism), (b) the champion-class QLIKE ratio vs HAR is ≤ the stored threshold, and (c) the VaR backtests pass. A PR that silently degrades the model or changes the methodology fails CI.
4. `helm`: `helm lint`, `helm template | kubeconform -strict`, and `chart-testing` (`ct lint`).
5. `e2e` (on push to main): create a **kind** cluster in the runner → `helm install` with `values-kind.yaml` → wait for rollout → run the smoke test + a 30-second k6 test → collect logs on failure.

**`cd.yml`** (on push to `main` and tags):
- Build both images with Buildx, multi-arch (`linux/amd64,linux/arm64`), push to **GHCR** tagged `sha-<short>`, plus `vX.Y.Z` on tags.
- **Trivy image scan** (fail on CRITICAL), **SBOM** (syft, attached as an attestation), and **cosign keyless signing** through GitHub OIDC. Sigstore signing is a strong production-hardening point.
- **GitOps bump:** open an automated PR (or commit) updating `deploy/helm/values-*.yaml` `image.tag`. Argo CD syncs it. CI never runs `kubectl apply` against the cluster; Argo CD pulls.

**`nightly.yml`:** a scheduled full walk-forward on fresh data. It regenerates `reports/` and opens a PR if the results changed. This keeps the README numbers current and auditable.

Add **Renovate** or **Dependabot** for dependencies, Actions, and base images.

### P7.2 Helm chart `deploy/helm/volforecast/`
| Resource | Notes |
|----------|-------|
| `Deployment` (api) | 2 replicas; `RollingUpdate maxUnavailable=0`; `readinessProbe /readyz`, `livenessProbe /healthz`, `startupProbe`; resources requests/limits; `securityContext`: runAsNonRoot, readOnlyRootFilesystem, drop ALL caps, seccomp RuntimeDefault; `topologySpreadConstraints` |
| `Service` + `Ingress` | ingress-nginx; TLS through cert-manager on the cloud cluster |
| `HorizontalPodAutoscaler` | autoscaling/v2, CPU 60% + optional custom metric (RPS via prometheus-adapter); min 2 / max 8; `behavior` scale-down stabilization 300 s |
| `PodDisruptionBudget` | `minAvailable: 1` |
| `NetworkPolicy` | default deny; api → postgres/mlflow only; ingress from ingress-nginx; jobs → egress to Binance + MinIO/MLflow |
| `ConfigMap` | non-secret config (`configs/base.yaml` rendered) |
| `Secret` | DB/MinIO creds via **Sealed Secrets** (kind) or **External Secrets Operator** (cloud). **No plaintext secrets in git.** |
| `CronJob` ×4 | ingest+features+forecast (00:20), evaluate (01:00), retrain (Sun 02:00), `concurrencyPolicy: Forbid`, `startingDeadlineSeconds`, `backoffLimit: 2`, `activeDeadlineSeconds`, `ttlSecondsAfterFinished`; `timeZone: Etc/UTC` |
| `ServiceAccount` + RBAC | least privilege. Only the retrain job's SA may create Jobs, if the drift trigger launches a Job. |
| `ServiceMonitor` + `PrometheusRule` | scraped by kube-prometheus-stack; alert rules from M8 |
| Dependencies | Bitnami `postgresql`, `minio` (or the MinIO operator), MLflow (community chart or own Deployment), Prefect server chart, `kube-prometheus-stack` installed separately as a platform app |

- `values-kind.yaml`: small resources, NodePort/port-forward, sealed secrets.
- `values-cloud.yaml`: ingress + TLS, external secrets, larger HPA ceiling.
- A `helm test` hook pod runs the smoke test.

### P7.3 Clusters
- **Local:** `make kind-up` creates a kind cluster (`deploy/kind/cluster.yaml`, 1 control plane + 2 workers so the topology spread and PDB actually mean something), installs ingress-nginx, metrics-server, sealed-secrets, kube-prometheus-stack, and Argo CD, then deploys the app-of-apps.
- **Cloud (optional, with a cost cap):** **GKE Autopilot** or **AKS** free control plane, with a single small node pool. Provision with **Terraform** (`deploy/terraform/`, remote state in a bucket) so `make cloud-up` and `make cloud-down` are one command each. Set a budget alert at $10 and document the teardown. Keep the cluster up only for screenshots and demos. The always-on public dashboard does not depend on it (M9).

### P7.4 GitOps with Argo CD
- `deploy/argocd/root-app.yaml` (app-of-apps): `platform` (monitoring, sealed-secrets, ingress) and `volforecast` (the chart + env values).
- Sync policy: automated, `prune: true`, `selfHeal: true`. A **rollback** is a `git revert` of the tag bump; document it in the runbook and rehearse it once, with a screenshot.
- Optional: **Argo Rollouts** canary for the API, with an analysis template on error rate and P95 latency. It's a strong addition if time allows.

### Acceptance
- A PR to `main` produces a signed image in GHCR, Argo CD syncs the new tag to the kind cluster, and the rollout completes with zero failed requests during a k6 run (verified once and recorded in `docs/runbook.md`).
- The CronJobs run on schedule in the cluster, and their runs appear in Prefect and MLflow.

---

## M8 — Observability, load testing, SRE docs

**Branch:** `refine/m8-observability`

### Metrics (`monitoring/prom.py`)
- Serving: `http_request_duration_seconds` histogram (by route/status), `http_requests_total`, `model_info{asset,horizon,version}` (gauge = 1), and `model_reload_total`.
- Pipeline (pushed through Pushgateway from jobs, or exposed by a tiny exporter reading the DB): `pipeline_last_success_timestamp{flow}`, `data_last_bar_timestamp{asset}`, `forecast_var{asset,h}`, `realized_var{asset}`, `rolling_qlike{model,window}`, `qlike_ratio_vs_har`, `var_breaches_total{level}`, and `drift_share_of_drifted_features`.

### Grafana (dashboards as JSON in git, provisioned)
1. **Service:** RPS, P50/P95/P99 latency, error rate, pod count / HPA replicas, CPU/memory.
2. **Model:** forecast vs realized vol (annualized), rolling QLIKE champion vs HAR, VaR bands + breach markers, cumulative breaches vs expected.
3. **Data and pipelines:** data freshness, last successful run per flow, job durations, incomplete-day count.

### Alerts (PrometheusRule → Alertmanager → email/Slack/Discord webhook)
| Alert | Condition |
|-------|-----------|
| `DataStale` | `time() - data_last_bar_timestamp > 26h` |
| `PipelineFailed` | no successful `forecast` flow in 26 h |
| `APIErrorRate` | 5xx > 1% for 10 m |
| `APILatencyP95` | P95 > 200 ms for 10 m |
| `ModelUnderperformsHAR` | `qlike_ratio_vs_har > 1.0` for 10 d |
| `VaRBreachCluster` | ≥ 3 breaches of 99% VaR in 10 days |

Define two **SLOs** in `docs/slo.md`: availability 99.5% on `/v1/forecast`, and freshness (a forecast for day *t+1* is available by 01:00 UTC on 99% of days). Add an error-budget panel.

### Load test (k6, `loadtest/k6/forecast.js`)
- Scenarios: a constant 50 RPS for 5 minutes, then ramping to 300 RPS. Thresholds: `p(95) < 100ms`, `http_req_failed < 0.5%`.
- Measure **HPA scale-out time** (from the first threshold crossing to the new pod being Ready) from the Grafana annotations.
- Results go into `reports/loadtest.md` (machine, cluster size, replicas, RPS, P95, P99), so the resume bullet comes from data.

### Docs
- `docs/runbook.md`: deploy, roll back, re-run a failed day (`backfill --date`), rotate secrets, handle a Binance outage, and force a champion rollback (move the `@champion` alias back).
- `docs/model_card.md`: intended use, data, evaluation, limitations, and ethical/financial disclaimers (not investment advice).

---

## M9 — Presentation and live dashboard

**Branch:** `refine/m9-presentation`

1. **README rewrite** (top to bottom):
   - A one-paragraph pitch and the headline results (generated).
   - An **architecture diagram** (Mermaid in the README plus an exported SVG): Binance → ingest/validate → Parquet + manifest → features → walk-forward/training → MLflow registry (`@champion`) → forecast job → Postgres → FastAPI → dashboard; with the side channels Prometheus/Grafana, Evidently, Prefect, Argo CD, and GitHub Actions.
   - Results: the QLIKE table with DM p-values and MCS membership, the MZ table, the VaR backtest table, the regime table, and the forecast-vs-realized figure.
   - Screenshots: Grafana, Argo CD sync, the Prefect flow graph, the MLflow model page.
   - "How to reproduce" in 3 commands, plus "Design decisions" (links to the ADRs).
   - **Limitations:** a single venue (Binance), noise in 5-min RV, the h=1h horizon not done, survivorship of the assets chosen, regime dependence, and no transaction or market-impact modeling. Not investment advice.
2. **Always-on public dashboard, independent of the cluster:** a scheduled GitHub Action (`pages.yml`, daily 01:30 UTC) runs ingest → forecast with the champion pulled from the model artifact, writes `forecasts.json`, and publishes a static dashboard (Plotly or Observable) to **GitHub Pages**. It costs $0, it stays up when the cloud cluster is torn down, and it builds a public track record of out-of-sample forecasts over time, which is stronger evidence than any backtest.
3. **Rewrite `INTERVIEW_PREPARATION_GUIDE.md`** around the new system. Include the leakage story (A1/A2), why QLIKE, why HAR is hard to beat, the DM vs MCS trade-off, VaR backtest interpretation, aliases vs stages, CronJob + Prefect, and the HPA scale-out numbers.
4. Release `v1.0.0` with a changelog.

---

## 4. PR roadmap (order of merges)

| PR | Content | Depends on |
|----|---------|-----------|
| #1 | Retract results, tag `v0-legacy`, ADR-0002 | — |
| #2 | uv/pyproject, src layout, ruff/mypy/pytest/pre-commit, structlog, Makefile, CI skeleton (lint+test) | #1 |
| #3 | Binance ingestion + checksums + Pandera + Parquet store/manifest | #2 |
| #4 | Realized measures + targets + **leakage guard tests** | #3 |
| #5 | Model protocol + naive/EWMA/GARCH/GJR + HAR family | #4 |
| #6 | Walk-forward engine + losses + DM/MZ/MCS + report generator + README markers | #5 |
| #7 | CI regression gate on the frozen fixture | #6 |
| #8 | LightGBM/XGB + exogenous ablation + SHAP | #6 |
| #9 | LSTM (+ PatchTST optional) + ensemble + panel model | #6 |
| #10 | VaR/ES + Kupiec/Christoffersen/traffic light | #6 |
| #11 | MLflow tracking + registry + promotion gate | #6, #10 |
| #12 | Prefect flows + forecast store + drift monitoring | #11 |
| #13 | API v1 rewrite + Dockerfiles + compose stack + Streamlit on API | #12 |
| #14 | Helm chart + kind + kubeconform/ct in CI + e2e kind job | #13 |
| #15 | CD: GHCR, Trivy, SBOM, cosign, GitOps bump + Argo CD | #14 |
| #16 | Prometheus metrics, Grafana dashboards, alerts, SLOs | #14 |
| #17 | k6 load tests + HPA measurement + runbook | #15, #16 |
| #18 | Terraform cloud cluster (optional) | #15 |
| #19 | GitHub Pages live dashboard + README rewrite + interview guide + v1.0.0 | all |

---

## 5. ADR index (one page each: context → decision → consequences)

0001 Record decisions · 0002 Legacy results retracted (leakage) · 0003 Missing-bar and incomplete-day policy · 0004 Parquet + manifest over DVC · 0005 Tuning protocol inside walk-forward · 0006 MLflow aliases instead of stages · 0007 CronJob triggers, Prefect executes · 0008 Kubernetes resource choices (HPA/PDB/NetworkPolicy) · 0009 Batch inference + forecast store instead of on-request inference · 0010 GitOps (Argo CD pull) instead of CI push · 0011 Public dashboard on GitHub Pages, decoupled from the cluster

---

## 6. Risks and mitigations

| Risk | Mitigation |
|------|-----------|
| data.binance.vision is blocked or rate-limited in your region | `Source` abstraction + Coinbase fallback; cache raw zips; CI uses the frozen fixture only |
| Binance timestamp unit change (ms → µs from 2025) | detection + unit test (P1.1) |
| Walk-forward compute grows too large (GARCH/DL refits) | the `refit_every` cadence, joblib parallelism, incremental caching of origins, DL refit monthly |
| The DL models lose to HAR | this is expected and credible: report it, and make the ensemble and exogenous ablation the ML story |
| Cloud costs | Terraform up/down, budget alert, Autopilot/free tier, the Pages dashboard keeps the public demo alive |
| Scope creep (PatchTST, Rollouts, OTel, Terraform) | these are marked optional; the minimum interview-ready path is M0–M2 + M4 + CI/Helm/kind |
| Secrets leaking into a public repo | Sealed Secrets/ESO, gitleaks in pre-commit and CI, `.env` gitignored |

---

## 7. Updated definition of done

**Research correctness**
- [ ] Leakage fixed; truncation-invariance and target-alignment tests in CI
- [ ] Daily RV/BPV/RS±/RQ from Binance 5-min data for BTC, ETH, SOL with checksummed, validated, hashed snapshots
- [ ] Naive, EWMA, GARCH, GJR, HAR, log-HAR, HAR-CJ, SHAR, HARQ baselines
- [ ] Walk-forward results (h=1, h=5): QLIKE, MSE, MZ R² + Wald, DM vs HAR (HAC + HLN), MCS membership, regime breakdown
- [ ] GBM (+ exogenous ablation), LSTM (5 seeds), ensemble; SHAP
- [ ] VaR 95/99 and ES 97.5: Kupiec, Christoffersen CC, Basel traffic light

**MLOps**
- [ ] MLflow tracking + registry with `@champion`/`@challenger` aliases and a logged promotion gate
- [ ] Prefect flows (ingest → features → forecast → evaluate → retrain) with retries and idempotency
- [ ] Evidently feature drift + performance-drift-triggered retraining, tested with a forced breach

**Platform**
- [ ] CI: lint, mypy, tests + coverage, security scans, Helm validation, **model regression gate**, kind e2e
- [ ] CD: multi-arch signed images on GHCR, SBOM, GitOps deploy through Argo CD, rehearsed rollback
- [ ] Helm: Deployment, Service, Ingress, HPA, PDB, NetworkPolicy, ConfigMap, sealed/external Secrets, CronJobs, ServiceMonitor, PrometheusRule
- [ ] Grafana dashboards + alerts + SLOs; k6 results with P95 and HPA scale-out time

**Presentation**
- [ ] README generated tables, architecture diagram, screenshots, limitations, ADRs, model card, runbook
- [ ] Public GitHub Pages dashboard with a growing out-of-sample track record

---

## 8. Resume bullets (fill only from `reports/`)

- "Built a walk-forward volatility forecasting system for BTC/ETH/SOL on 5-min Binance data (~[N]M bars). [Model] reduced out-of-sample QLIKE [X]% vs HAR-RV (Diebold-Mariano p=[Y]; in the 90% Model Confidence Set) across [N] days including LUNA/FTX."
- "Turned variance forecasts into 99% VaR/97.5% ES that passed Kupiec and Christoffersen backtests ([X] breaches vs [Y] expected, Basel green zone)."
- "Found and removed target leakage in the original pipeline. Added truncation-invariance tests and a CI model-regression gate that blocks PRs which degrade QLIKE."
- "Deployed on Kubernetes with Helm (HPA, PDB, NetworkPolicy, CronJobs) through GitOps (Argo CD) and GitHub Actions CI/CD with signed images (cosign) and SBOMs. MLflow champion/challenger registry, Prefect orchestration, Evidently drift-triggered retraining. P95 [X] ms at [Y] RPS; HPA scale-out in [Z] s."

---

## 9. Day-1 checklist (start here)

1. `cd crypto-volatility-forecasting && git checkout -b refine/m0-foundation && git tag v0-legacy`
2. Remove the README results table; `git rm --cached models/*.csv models/*.joblib models/*.pt`; update `.gitignore`.
3. Write ADR-0002 (leakage findings A1–A6 from §0).
4. `uv init --package`, move `src/` → `src/volforecast/`, add ruff/mypy/pytest, and get `make test` passing with a first unit test.
5. Open PR #1/#2. Then start PR #3 (Binance ingestion), which everything else depends on.
