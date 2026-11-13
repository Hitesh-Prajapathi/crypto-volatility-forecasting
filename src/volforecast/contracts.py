"""Shared data contracts.

Every module codes against the names and shapes defined here. Change this file only
through a deliberate, reviewed PR: data, features, models, evaluation and serving all
depend on it.

Conventions
-----------
* All timestamps are UTC. A "day" is the UTC calendar day [00:00, 24:00).
* Variance is the unit of forecasting. Volatility = sqrt(variance). Annualize with
  sqrt(ANNUALIZATION_DAYS) (crypto trades 365 days a year).
* Returns are log returns, NOT multiplied by 100 (models that need %-scale, e.g. `arch`,
  rescale internally and convert back).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Self, runtime_checkable

import pandas as pd

ANNUALIZATION_DAYS = 365
BARS_PER_DAY_5M = 288
DEFAULT_HORIZONS: tuple[int, ...] = (1, 5)

# ── 1. Intraday klines (output of volforecast.data) ──────────────────────────
# One row per 5-minute bar, sorted by open_time ascending, unique open_time.
# open_time: pandas datetime64[ns, UTC]. Prices float64 > 0. RangeIndex.
KLINE_COLUMNS: tuple[str, ...] = (
    "open_time",  # datetime64[ns, UTC], bar open
    "open",
    "high",
    "low",
    "close",
    "volume",  # base-asset volume
    "quote_volume",
    "n_trades",  # int64
)

# ── 2. Daily realized panel (output of volforecast.features.realized) ────────
# Index: DatetimeIndex named "date", tz-naive, normalized to midnight, each label is the
# UTC calendar day the measures cover. Sorted ascending, unique, daily frequency
# (missing days are NOT dropped from the index: they appear with complete=False).
DATE_INDEX = "date"
COL_RV = "rv"  # realized variance: sum of squared 5-min log returns in the day
COL_BPV = "bpv"  # bipower variation: (pi/2) * sum |r_i||r_{i-1}|
COL_JUMP = "jump"  # max(rv - bpv, 0)
COL_RS_POS = "rs_pos"  # positive realized semivariance: sum r_i^2 1[r_i > 0]
COL_RS_NEG = "rs_neg"  # negative realized semivariance: sum r_i^2 1[r_i < 0]
COL_RQ = "rq"  # realized quarticity: (M/3) * sum r_i^4, M = number of returns
COL_RET = "ret"  # daily close-to-close log return (last close of day t vs day t-1)
COL_N_BARS = "n_bars"  # number of observed (non-filled) bars in the day
COL_COMPLETE = "complete"  # bool: day passes the completeness policy

PANEL_COLUMNS: tuple[str, ...] = (
    COL_RV,
    COL_BPV,
    COL_JUMP,
    COL_RS_POS,
    COL_RS_NEG,
    COL_RQ,
    COL_RET,
    COL_N_BARS,
    COL_COMPLETE,
)


def target_column(h: int) -> str:
    """Name of the h-day cumulative realized variance target: sum_{k=1..h} RV_{t+k}."""
    return f"target_rv_h{h}"


# ── 3. Forecast records (output of the walk-forward engine / forecast job) ───
FORECAST_COLUMNS: tuple[str, ...] = (
    "asset",  # e.g. "BTCUSDT"
    "model",  # model name, e.g. "har"
    "horizon",  # int h
    "origin_date",  # last date in the information set (forecast made at end of this day)
    "target_date",  # origin_date + h days (last day covered by the target)
    "forecast_var",  # forecast of sum_{k=1..h} RV_{origin+k}
    "realized_var",  # realized value of that sum (NaN until known)
    "model_version",  # str
    "data_hash",  # str: content hash of the dataset snapshot used
)


@runtime_checkable
class VolModel(Protocol):
    """Interface every volatility model implements.

    The walk-forward engine passes `panel.loc[:origin]`, i.e. ONLY rows whose date is
    <= the forecast origin. A model must never require rows after the origin. A model
    builds its own regressors/targets from that history. Training targets that would need
    rows past the end of the history are simply unavailable (this gives the h-1 embargo
    for free).

    `fit` is called every `refit_every` origins. `predict_variance` is called at every
    origin with the (longer) up-to-date history, and must use the parameters from the last
    `fit` while filtering/updating state on the new rows (e.g. GARCH recursion, HAR
    regressors from the latest RV values).
    """

    name: str

    def fit(self, panel: pd.DataFrame, h: int) -> Self:
        """Estimate parameters on the history in `panel` for horizon `h`."""
        ...

    def predict_variance(self, panel: pd.DataFrame, h: int) -> float:
        """Forecast sum_{k=1..h} RV_{t+k}, where t = panel.index[-1]. Must be > 0, finite."""
        ...

    def get_params(self) -> dict[str, Any]:
        """JSON-serializable hyperparameters + fitted coefficients (for tracking)."""
        ...


@dataclass(frozen=True)
class TestResult:
    """Generic statistical-test result used across evaluation modules."""

    statistic: float
    pvalue: float
    detail: dict[str, Any] | None = None

    __test__ = False  # stop pytest from trying to collect this class
