"""Test helpers: panels built from simulated variance / return series."""

from __future__ import annotations

import numpy as np
import pandas as pd

from volforecast.contracts import (
    COL_BPV,
    COL_COMPLETE,
    COL_JUMP,
    COL_N_BARS,
    COL_RET,
    COL_RQ,
    COL_RS_NEG,
    COL_RS_POS,
    COL_RV,
    DATE_INDEX,
    PANEL_COLUMNS,
)


def panel_from_rv(rv: np.ndarray, ret: np.ndarray | None = None, seed: int = 0) -> pd.DataFrame:
    """Complete daily panel whose RV column is `rv` (other columns derived from it)."""
    rng = np.random.default_rng(seed)
    n = rv.size
    bpv = rv * rng.uniform(0.8, 1.0, n)
    share = rng.uniform(0.35, 0.65, n)
    if ret is None:
        ret = np.sqrt(rv) * rng.standard_normal(n)
    idx = pd.date_range("2018-01-01", periods=n, freq="D", name=DATE_INDEX)
    df = pd.DataFrame(
        {
            COL_RV: rv,
            COL_BPV: bpv,
            COL_JUMP: np.maximum(rv - bpv, 0.0),
            COL_RS_POS: rv * share,
            COL_RS_NEG: rv * (1 - share),
            COL_RQ: rv**2,
            COL_RET: ret,
            COL_N_BARS: np.full(n, 288, dtype="int64"),
            COL_COMPLETE: np.ones(n, dtype=bool),
        },
        index=idx,
    )
    return df[list(PANEL_COLUMNS)]


def simulate_har_rv(
    n: int,
    b0: float = 1e-4,
    bd: float = 0.4,
    bw: float = 0.3,
    bm: float = 0.2,
    noise_sd: float = 0.3,
    seed: int = 0,
) -> np.ndarray:
    """RV_{t+1} = (b0 + bd RV_t + bw mean(RV_{t-4:t}) + bm mean(RV_{t-21:t})) * eps, E[eps]=1."""
    rng = np.random.default_rng(seed)
    mean = b0 / (1 - bd - bw - bm)
    rv = np.full(n, mean)
    eps = np.exp(noise_sd * rng.standard_normal(n) - 0.5 * noise_sd**2)
    for t in range(22, n - 1):
        rv[t + 1] = (
            b0 + bd * rv[t] + bw * rv[t - 4 : t + 1].mean() + bm * rv[t - 21 : t + 1].mean()
        ) * eps[t + 1]
    return rv
