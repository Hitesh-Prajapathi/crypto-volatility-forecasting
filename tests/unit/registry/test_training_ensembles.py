"""The retrain path must give ens_inv its stateful walk-forward (regression: ens_inv == ens_eq)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from volforecast.models.factory import is_stateful_model
from volforecast.registry.config import CandidateConfig, ProductionConfig, RetrainConfig
from volforecast.registry.training import walkforward_candidate

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "daily_panel_btcusdt.parquet"
MEMBERS = {"members": ["log_har", "har"]}  # cheap, deterministic members (no GBM)


def _forecast(name: str, refit_every: int) -> pd.Series:
    panel = pd.read_parquet(FIXTURE)
    cfg = ProductionConfig(
        assets=["BTCUSDT"],
        horizons=[1],
        retrain=RetrainConfig(candidates=[CandidateConfig(name="har")], wf_days=100, min_train=365),
    )
    cand = CandidateConfig(name=name, params=MEMBERS, refit_every=refit_every)
    fc = walkforward_candidate(panel, "BTCUSDT", 1, cand, cfg, "hash")
    return fc.set_index("origin_date")["forecast_var"]


def test_ens_inv_differs_from_ens_eq_in_retrain_path() -> None:
    eq = _forecast("ens_eq", 20)
    inv = _forecast("ens_inv", 20)
    assert len(eq) == len(inv) > 50
    assert not np.allclose(eq.to_numpy(), inv.to_numpy(), rtol=1e-12)


def test_stateful_flag() -> None:
    assert is_stateful_model("ens_inv")
    assert not is_stateful_model("ens_eq")
    assert not is_stateful_model("har")
