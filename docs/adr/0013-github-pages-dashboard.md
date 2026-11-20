# 0013. Public dashboard as a static GitHub Pages site built daily in CI

- Status: Accepted
- Date: 2026-11-20

## Context

The project needs a public, always-on view of the forecasts and of the live out-of-sample track
record. The production design (batch inference into a forecast store, a FastAPI service,
MLflow-registry champions; [ADR-0009](0009-batch-inference-forecast-store.md),
[ADR-0006](0006-mlflow-aliases-over-stages.md)) assumes a cluster and an MLflow server, neither of
which is reachable from GitHub Actions or a free public host. The page must also be honest: no
invented numbers, and a clear statement that this is a volatility forecast, not investment advice.

## Decision

We publish a fully static site, built and deployed by `.github/workflows/pages.yml`, with the
builder in `src/volforecast/site/` (`python -m volforecast.site`, `make site`).

What it builds (`build_site`): `index.html`, `app.js`, `style.css`, `.nojekyll`,
`data/forecasts.json` and, when `reports/tables.md` exists, `data/backtest_summary.json`. Plotly is
loaded by the page from a pinned CDN URL; there is no build step. `forecasts.json` holds, per asset
and horizon (h = 1 and 5 by default), the latest forecast (variance, annualized volatility, and
VaR/ES fields when the store has them), up to 540 days of forecast-vs-realized history with the HAR
benchmark forecast, and the cumulative QLIKE of the served model against HAR (meaningful once at
least 20 realized pairs exist). `backtest_summary.json` is parsed from the committed
`reports/tables.md` and carries a short caveat that most of the gain over HAR is a weekend effect.

Schedule and flow: the workflow runs daily at 01:30 UTC and on `workflow_dispatch`. A `build` job
restores state, ingests the newest 5-minute bars (`volforecast.flows.ingest`; a full bootstrap from
2017-08-17 on a cold start), runs `volforecast.flows.forecast`, saves state, builds the site with
`--publish-store`, and uploads the Pages artifact. A separate `deploy` job (`needs: build`,
environment `github-pages`) deploys it. A `concurrency` group serialises runs without cancelling
one in progress, and the build job has a 90 minute timeout.

Model: the job has no MLflow server, so the registry has no `@champion`. The forecast flow's
documented fallback is used instead: a CI-only copy of `configs/production.yaml` sets
`fallback_model: xgb`, and that model is refit on the full ingested history at every origin. Rows
are stored with role `fallback` and `model_version` `fallback:xgb`, and the page labels them so. The
dashboard is therefore a track record of a daily-refit XGBoost, not of a registry-promoted model.

State: nothing is committed to git. The sqlite forecast store and the parquet bars live in
`actions/cache` (`pages-state-<run id>`, restored through the `pages-state-` prefix). If the cache
is gone, the job downloads `data/store.db` from the live site (published by `--publish-store`, a
`VACUUM INTO` copy) and checks the SQLite header; only the bars are then re-bootstrapped. If both
are lost, the record restarts and the page says "track record starts <date>".

What it does not show: price or direction forecasts, the registry champion, challenger models
(including the LSTM of [ADR-0012](0012-lstm-challenger.md)), or promotion state. Only the stored
forecasts and the committed backtest tables are displayed. An empty store renders a message and no
numbers. `--demo` marks the payload and shows a "DEMO DATA" banner; the workflow never passes it.

Failure behaviour: the steps run in order with default failure semantics, so a failed restore,
ingest, forecast or build fails the `build` job, and because `deploy` needs `build`, nothing is
deployed and the previously published site stays up unchanged. The state cache is saved right after
the forecast step, so a later failure (build or upload) does not lose that day's forecasts. A
missing `reports/tables.md` only omits the backtest section. There is no alerting beyond GitHub's
failed-run notification.

## Consequences

+ A public dashboard with no server, no secrets and no cluster, at no cost.
+ The track record accumulates automatically; each payload carries the code SHA.
- The CI forecast can differ from what a served champion would say: the champion is whatever the
  promotion gate selected on the cluster ([ADR-0006](0006-mlflow-aliases-over-stages.md)), while the
  dashboard always shows a refit `xgb`. They coincide only while xgb is the champion.
- Persistence is best effort: a cache that is evicted after 7 unused days, plus the last deployed
  artifact. Losing both resets the track record.
- The first run bootstraps the whole history and is slow; it must finish inside the 90 minute limit.
- Pages must be enabled by hand: Settings, Pages, Source = "GitHub Actions". Until then the
  `deploy` job fails and the site does not exist; run the workflow again (`workflow_dispatch`) after
  enabling it.
- The page depends on Binance Vision availability and on a CDN for Plotly.

## Alternatives considered

- Streamlit Community Cloud or another hosted app: needs a running server and a reachable forecast
  store and registry, and adds an availability burden for a read-only record.
- A hosted API behind the page (the FastAPI service from M6): same reachability problem, plus
  hosting cost.
- Committing generated artifacts (the sqlite store or JSON) to the repository or a data branch:
  adds daily bot commits and history growth; rejected in favour of cache plus the published
  `store.db`, at the cost of weaker durability.
- Pointing CI at a remote MLflow registry: no registry is reachable from Actions, and exposing one
  publicly for this purpose was rejected.
