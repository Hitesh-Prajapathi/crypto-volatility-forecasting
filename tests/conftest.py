"""Shared synthetic fixtures (no network). Real-data fixtures live in tests/fixtures/."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from volforecast.contracts import (
    BARS_PER_DAY_5M,
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


def simulate_daily_variance(n_days: int, seed: int = 0) -> np.ndarray:
    """Persistent log-variance process (AR(1) on log RV) around ~(2.5% daily vol)^2."""
    rng = np.random.default_rng(seed)
    log_v = np.empty(n_days)
    mu, phi, sigma = np.log(0.025**2), 0.97, 0.25
    log_v[0] = mu
    for t in range(1, n_days):
        log_v[t] = mu + phi * (log_v[t - 1] - mu) + sigma * rng.standard_normal()
    return np.exp(log_v)


def make_synthetic_klines(
    n_days: int = 40, seed: int = 0, start: str = "2024-01-01"
) -> pd.DataFrame:
    """5-minute klines following the contracts.KLINE_COLUMNS schema."""
    rng = np.random.default_rng(seed)
    daily_var = simulate_daily_variance(n_days, seed)
    per_bar_sd = np.repeat(np.sqrt(daily_var / BARS_PER_DAY_5M), BARS_PER_DAY_5M)
    r = per_bar_sd * rng.standard_t(df=5, size=per_bar_sd.size) / np.sqrt(5 / 3)
    close = 40_000.0 * np.exp(np.cumsum(r))
    open_ = np.concatenate([[40_000.0], close[:-1]])
    spread = np.abs(r) * close * 0.5
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    open_time = pd.date_range(start, periods=close.size, freq="5min", tz="UTC")
    vol = rng.lognormal(3, 0.5, close.size)
    return pd.DataFrame(
        {
            "open_time": open_time,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": vol,
            "quote_volume": vol * close,
            "n_trades": rng.integers(100, 5000, close.size).astype("int64"),
        }
    )


def make_synthetic_panel(
    n_days: int = 1500, seed: int = 0, start: str = "2019-01-01"
) -> pd.DataFrame:
    """Daily realized panel following contracts.PANEL_COLUMNS (index named 'date')."""
    rng = np.random.default_rng(seed + 1)
    rv = simulate_daily_variance(n_days, seed)
    rv = rv * rng.lognormal(0, 0.3, n_days)  # measurement noise vs latent variance
    bpv = rv * rng.uniform(0.8, 1.0, n_days)
    share_pos = rng.uniform(0.35, 0.65, n_days)
    ret = np.sqrt(rv) * rng.standard_t(df=5, size=n_days) / np.sqrt(5 / 3)
    idx = pd.date_range(start, periods=n_days, freq="D", name=DATE_INDEX)
    panel = pd.DataFrame(
        {
            COL_RV: rv,
            COL_BPV: bpv,
            COL_JUMP: np.maximum(rv - bpv, 0.0),
            COL_RS_POS: rv * share_pos,
            COL_RS_NEG: rv * (1 - share_pos),
            COL_RQ: rv**2,  # E[RQ] = IV^2 under Gaussian intraday returns
            COL_RET: ret,
            COL_N_BARS: np.full(n_days, BARS_PER_DAY_5M, dtype="int64"),
            COL_COMPLETE: np.ones(n_days, dtype=bool),
        },
        index=idx,
    )
    return panel[list(PANEL_COLUMNS)]


@pytest.fixture
def synthetic_klines() -> pd.DataFrame:
    return make_synthetic_klines()


@pytest.fixture
def synthetic_panel() -> pd.DataFrame:
    return make_synthetic_panel()
