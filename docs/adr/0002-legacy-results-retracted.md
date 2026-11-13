# 0002. Legacy results retracted: audit findings A1-A7

- Status: Accepted
- Date: 2026-11-13

## Context

The first version of this repository (preserved at git tag `v0-legacy`, code under [`legacy/`](../../legacy/README.md)) reported strong results for Bitcoin volatility forecasting. A code audit before the refinement found methodological defects that invalidate those results. They are listed here with evidence so the retraction is transparent.

| # | Finding | Evidence |
|---|---------|----------|
| A1 | Target leakage. The target was `rv_30d`, but `rv_30d_ann` (= `rv_30d * sqrt(252)`) and `var_30d` (= `rv_30d^2`) stayed in the feature set. `get_feature_columns` excluded only the target column. Their correlation with the target is 1.000 (verified on the legacy data). | `legacy/src/features.py::get_feature_columns`. This explains the Random Forest MAE of 1.9e-5 in the committed `models/metrics.csv`. |
| A2 | Not a forecast. Features at row t predicted `rv_30d` at row t; the target was never shifted forward, so the task was nowcasting a lagging statistic. | `legacy/src/train.py` (target construction and split). |
| A3 | Published numbers were not reproducible. The README reported XGBoost as best (MAE 0.001234, MAPE 8.45%), while the committed `models/metrics.csv` showed Random Forest as best (MAE 1.88e-5). | README results table vs `models/metrics.csv` at tag `v0-legacy`. |
| A4 | GARCH comparison was invalid. GARCH's 1-day sigma was compared against a 30-day rolling standard deviation, and the test windows were misaligned. | GARCH evaluation code in `legacy/src/`. |
| A5 | LSTM scaler leakage. The `MinMaxScaler` was fit on the full dataset, test period included, and refit at inference time. | LSTM training and inference code in `legacy/src/`. |
| A6 | `bfill()` on `log_var_30d` pulled future values backwards into earlier rows. | Feature engineering in `legacy/src/features.py`. |
| A7 | Annualization by 252 trading days for a market that trades 365 days a year. | Feature engineering code in `legacy/src/`. |

## Decision

1. All legacy performance numbers are retracted. The README no longer contains them, and `models/metrics.csv` and `models/predictions.joblib` are removed from the working tree.
2. The legacy state stays reachable at tag `v0-legacy` and in `legacy/`, excluded from lint and type checks, for transparency.
3. New results are produced only by the walk-forward pipeline (HAR-RV baseline, QLIKE loss, Diebold-Mariano, Model Confidence Set, VaR backtests) and written by `make report`. No number appears in the README unless it comes from `reports/`.
4. Leakage guards (future-shift tests, information-set tests) are part of the test suite from M1 onward.
5. Annualization uses 365.

## Consequences

- The project starts from an honest baseline; the audit and fix are themselves part of the engineering story.
- Early milestones show no headline numbers. This is intentional.
- Legacy code is frozen; fixes happen in `src/volforecast`, not in `legacy/`.

## Alternatives considered

- Quietly fix the code and keep the numbers: dishonest, and the old numbers do not correspond to any artifact anyway.
- Delete the legacy code: loses the audit trail. A tag plus `legacy/` is cheap and transparent.
