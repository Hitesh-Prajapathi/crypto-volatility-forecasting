"""Helpers shared by the pipeline / risk / report tests (small, fast, synthetic)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

from volforecast.evaluation.walkforward import run_many
from volforecast.pipeline.backtest import model_specs, walkforward_config
from volforecast.pipeline.config import BacktestConfig

FAST_MODELS = ("har", "log_har", "naive", "ewma")


def make_cfg(
    assets: Sequence[str] = ("SYN",),
    models: Sequence[str] = FAST_MODELS,
    first_origin: str = "2022-01-01",
    horizons: Sequence[int] = (1, 5),
    **extra: Any,
) -> BacktestConfig:
    return BacktestConfig.model_validate(
        {
            "assets": list(assets),
            "horizons": list(horizons),
            "first_origin": first_origin,
            "min_train": 365,
            "models": [{"name": m, "refit_every": 1} for m in models],
            "benchmark": "har",
            "n_jobs": 1,
            "mcs_reps": 100,
            "var": {"alphas": [0.05, 0.01], "dists": ["normal", "t", "fhs"], "min_resid": 50},
            **extra,
        }
    )


def quick_forecasts(panel: pd.DataFrame, cfg: BacktestConfig, asset: str = "SYN") -> pd.DataFrame:
    fc, _ = run_many(
        panel, model_specs(cfg), walkforward_config(cfg), asset=asset, data_hash="h", n_jobs=1
    )
    return fc
