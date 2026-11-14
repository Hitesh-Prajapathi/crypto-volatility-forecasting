"""Shared computation for the model regression gate and its golden-value updater.

The gate runs HAR, log-HAR and HARQ (h=1, refit every origin, expanding window) over a fixed
window of a FROZEN real-data daily panel and computes mean QLIKE plus the 1% normal-VaR Kupiec
test of HAR. Nothing here touches the network or the data directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from volforecast.evaluation.losses import qlike
from volforecast.evaluation.risk import build_var_forecasts, var_backtest_table
from volforecast.evaluation.walkforward import ModelSpec, WalkForwardConfig, run_walkforward
from volforecast.models.factory import build_model

HERE = Path(__file__).resolve().parent
FIXTURE = HERE.parent / "fixtures" / "daily_panel_btcusdt.parquet"
GOLDEN = HERE / "golden.json"

ASSET = "BTCUSDT"
FIXTURE_START = "2021-01-01"
FIXTURE_END = "2023-12-31"
FIRST_ORIGIN = "2022-01-01"
MIN_TRAIN = 365
MODELS = ("har", "log_har", "harq")
BENCHMARK = "har"
THRESHOLD_MARGIN = 0.02


def load_fixture() -> pd.DataFrame:
    return pd.read_parquet(FIXTURE)


def compute_metrics() -> dict[str, Any]:
    """Walk-forward the gate models on the fixture; return QLIKE, ratios and the VaR test."""
    panel = load_fixture()
    cfg = WalkForwardConfig(
        first_origin=pd.Timestamp(FIRST_ORIGIN), horizons=(1,), min_train=MIN_TRAIN
    )
    frames = []
    for name in MODELS:
        spec = ModelSpec(name=name, builder=lambda n=name: build_model(n), refit_every=1)
        fc, _ = run_walkforward(panel, spec, cfg, h=1, asset=ASSET)
        frames.append(fc)
    forecasts = pd.concat(frames, ignore_index=True)
    wide = forecasts.pivot(index="origin_date", columns="model", values="forecast_var")
    rv = forecasts[forecasts["model"] == BENCHMARK].set_index("origin_date")["realized_var"]
    ok = rv.notna() & wide.notna().all(axis=1)
    mean_qlike = {
        m: float(qlike(rv[ok].to_numpy(), wide.loc[ok, m].to_numpy()).mean()) for m in MODELS
    }
    ratio = {m: mean_qlike[m] / mean_qlike[BENCHMARK] for m in MODELS}

    var_df = build_var_forecasts(
        forecasts[forecasts["model"] == BENCHMARK],
        {ASSET: panel},
        alphas=(0.01,),
        dists=("normal",),
    )
    row = var_backtest_table(var_df).iloc[0]
    return {
        "n_origins_scored": int(ok.sum()),
        "qlike": mean_qlike,
        "qlike_ratio_vs_har": ratio,
        "var99_normal_har": {
            "n_obs": int(row["n_obs"]),
            "breaches": int(row["breaches"]),
            "kupiec_p": float(row["kupiec_p"]),
        },
    }
