"""Realized volatility measures from 5-minute klines.

Pipeline: klines -> full 5-minute UTC grid (bounded forward fill of closes) -> intraday log
returns -> one row per UTC calendar day (``contracts.PANEL_COLUMNS``).

Conventions
-----------
* A bar's ``close`` is the price at ``open_time + 5min``; return ``r_i = log(close_i /
  close_{i-1})``. The first return of day *t* uses the last close of day *t-1*, so a full day
  has ``M = 288`` returns that sum exactly to the daily close-to-close log return.
* Gap filling is *causal at day granularity*: whether a missing bar is filled depends only
  on data up to the end of the bar's own UTC day (see :func:`_fill_close`). Consequently a
  panel row never changes when later days are appended (checked by the leakage-guard tests).
* The implementation is fully vectorized: closes live on a ``(n_days, 288)`` array.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import pandas as pd

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
from volforecast.logging import get_logger

logger = get_logger(__name__)

BAR_NS = 5 * 60 * 1_000_000_000
DAY_NS = BARS_PER_DAY_5M * BAR_NS
DEFAULT_SIGNATURE_INTERVALS: tuple[str, ...] = ("5min", "10min", "15min", "30min", "60min")

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.int64]


class _Grid(NamedTuple):
    """Closes on the full 5-minute grid, shaped ``(n_days, 288)``."""

    first_day: pd.Timestamp  # tz-naive midnight of the first day
    close: FloatArray  # forward-filled closes (NaN where unavailable)
    observed: BoolArray  # True where a real bar existed


def _empty_panel() -> pd.DataFrame:
    idx = pd.DatetimeIndex([], name=DATE_INDEX)
    panel = pd.DataFrame(
        {
            **{c: pd.Series([], dtype="float64", index=idx) for c in PANEL_COLUMNS},
        }
    )
    panel[COL_N_BARS] = panel[COL_N_BARS].astype("int64")
    panel[COL_COMPLETE] = panel[COL_COMPLETE].astype(bool)
    return panel[list(PANEL_COLUMNS)]


def _fill_close(close: FloatArray, observed: BoolArray, max_ffill: int) -> FloatArray:
    """Forward-fill ``close`` over gaps of at most ``max_ffill`` consecutive missing bars.

    Gaps longer than ``max_ffill`` stay NaN *entirely* (``Series.ffill(limit=)`` would fill
    their first bars, so the mask is built by hand). A gap's length is measured up to the end
    of the missing bar's own UTC day, i.e. a gap straddling midnight is judged by what is
    observable by the end of day *t* for the day-*t* bars. This keeps every day's values a
    function of data up to the end of that day only (no look-ahead into day t+1).
    """
    n = close.size
    idx = np.arange(n, dtype=np.int64)
    last_obs = np.maximum.accumulate(np.where(observed, idx, -1))
    big = np.int64(n + BARS_PER_DAY_5M)
    next_obs = np.minimum.accumulate(np.where(observed, idx, big)[::-1])[::-1]
    day_end = (idx // BARS_PER_DAY_5M + 1) * BARS_PER_DAY_5M
    gap_len = np.minimum(next_obs, day_end) - last_obs - 1
    fill = ~observed & (last_obs >= 0) & (gap_len <= max_ffill)
    out = np.where(observed, close, np.nan)
    out[fill] = close[last_obs[fill]]
    return out


def _build_grid(klines: pd.DataFrame, max_ffill: int) -> _Grid | None:
    """Place klines on the full 5-minute UTC grid; ``None`` if there are no usable rows."""
    missing = {"open_time", "close"} - set(klines.columns)
    if missing:
        raise ValueError(f"klines is missing columns: {sorted(missing)}")
    if max_ffill < 0:
        raise ValueError("max_ffill must be >= 0")
    if len(klines) == 0:
        return None

    open_time = pd.DatetimeIndex(klines["open_time"])
    if open_time.tz is not None:
        open_time = open_time.tz_convert("UTC").tz_localize(None)
    ts = open_time.to_numpy().astype("datetime64[ns]").astype(np.int64)
    if np.any(ts % BAR_NS != 0):
        raise ValueError("open_time values must lie on the 5-minute grid")

    first_day_ns = int(ts.min() // DAY_NS) * DAY_NS
    last_day_ns = int(ts.max() // DAY_NS) * DAY_NS
    n_days = (last_day_ns - first_day_ns) // DAY_NS + 1
    pos = (ts - first_day_ns) // BAR_NS
    if np.unique(pos).size != pos.size:
        raise ValueError("duplicate open_time values in klines")

    px = klines["close"].to_numpy(dtype=np.float64)
    usable = np.isfinite(px) & (px > 0)
    n = n_days * BARS_PER_DAY_5M
    close = np.full(n, np.nan)
    observed = np.zeros(n, dtype=bool)
    close[pos[usable]] = px[usable]
    observed[pos[usable]] = True

    filled = _fill_close(close, observed, max_ffill)
    first_day = pd.Timestamp(first_day_ns, unit="ns")
    shape = (n_days, BARS_PER_DAY_5M)
    return _Grid(first_day, filled.reshape(shape), observed.reshape(shape))


def _log_returns(close: FloatArray) -> FloatArray:
    """Log returns on the flattened grid; first element (no previous close) is NaN."""
    flat = close.reshape(-1)
    with np.errstate(invalid="ignore"):
        lc = np.log(flat)
    r = np.empty_like(lc)
    r[0] = np.nan
    r[1:] = lc[1:] - lc[:-1]
    return r.reshape(close.shape)


def intraday_log_returns(klines: pd.DataFrame, max_ffill: int = 3) -> pd.DataFrame:
    """5-minute log returns on the full UTC grid with bounded forward-filling of closes.

    The grid runs from 00:00 of the first day to 23:55 of the last day (UTC). Gaps of at most
    ``max_ffill`` consecutive missing bars are forward-filled (each filled bar yields a zero
    return); longer gaps stay NaN (see :func:`_fill_close` for the day-boundary rule).

    Returns a DataFrame indexed by ``open_time`` (UTC) with columns ``close`` (filled),
    ``log_return`` (``log(close_i / close_{i-1})``, so the first return of a day uses the
    previous day's last close), ``observed`` (bar existed vs. filled) and ``date`` (tz-naive
    UTC day of the bar's open_time).
    """
    grid = _build_grid(klines, max_ffill)
    if grid is None:
        index = pd.DatetimeIndex([], tz="UTC", name="open_time")
        return pd.DataFrame(
            {
                "close": pd.Series([], dtype="float64", index=index),
                "log_return": pd.Series([], dtype="float64", index=index),
                "observed": pd.Series([], dtype="bool", index=index),
                "date": pd.Series([], dtype="datetime64[ns]", index=index),
            }
        )
    n_days = grid.close.shape[0]
    n = grid.close.size
    index = pd.date_range(grid.first_day, periods=n, freq="5min", tz="UTC", name="open_time")
    date = np.repeat(
        pd.date_range(grid.first_day, periods=n_days, freq="D").to_numpy(), BARS_PER_DAY_5M
    )
    return pd.DataFrame(
        {
            "close": grid.close.reshape(-1),
            "log_return": _log_returns(grid.close).reshape(-1),
            "observed": grid.observed.reshape(-1),
            "date": date,
        },
        index=index,
    )


def daily_realized_measures(
    klines: pd.DataFrame, min_bars: int = 280, max_ffill: int = 3
) -> pd.DataFrame:
    """Daily realized measures panel (``contracts.PANEL_COLUMNS``) from 5-minute klines.

    Per UTC day, with ``r_i`` the valid intraday log returns and ``M`` their count:

    * ``rv = sum r_i^2``; ``bpv = (pi/2) sum |r_i||r_{i-1}|`` over consecutive valid pairs
      within the day; ``jump = max(rv - bpv, 0)``;
    * ``rs_pos / rs_neg`` = sum of ``r_i^2`` over positive / negative returns (zeros go to
      neither, so ``rs_pos + rs_neg == rv`` exactly up to float summation order);
    * ``rq = (M / 3) sum r_i^4``;
    * ``ret = log(last close of day t / last close of day t-1)`` (closes after gap filling);
    * ``n_bars`` = number of observed (non-filled) bars;
    * ``complete = (n_bars >= min_bars) and (no NaN return in the day)``.

    Incomplete-but-nonempty days are computed over their valid returns (``complete=False``);
    days without any valid return have NaN measures. Missing calendar days are kept in the
    index (``n_bars=0``, ``complete=False``). The first day has no previous close, so its first
    return is NaN and it is never complete.
    """
    grid = _build_grid(klines, max_ffill)
    if grid is None:
        return _empty_panel()

    r = _log_returns(grid.close)  # (D, 288)
    valid = np.isfinite(r)
    r0 = np.where(valid, r, 0.0)
    m = valid.sum(axis=1)
    has_ret = m > 0

    r2 = r0 * r0
    rv = r2.sum(axis=1)
    rs_pos = np.where(r0 > 0, r2, 0.0).sum(axis=1)
    rs_neg = np.where(r0 < 0, r2, 0.0).sum(axis=1)
    rq = m / 3.0 * (r2 * r2).sum(axis=1)

    pair_valid = valid[:, 1:] & valid[:, :-1]
    abs_r = np.abs(r0)
    bpv = (np.pi / 2.0) * np.where(pair_valid, abs_r[:, 1:] * abs_r[:, :-1], 0.0).sum(axis=1)
    has_pair = pair_valid.any(axis=1)

    nan = np.float64(np.nan)
    rv, rs_pos, rs_neg, rq = (np.where(has_ret, a, nan) for a in (rv, rs_pos, rs_neg, rq))
    bpv = np.where(has_pair, bpv, nan)
    jump = np.maximum(rv - bpv, 0.0)  # NaN propagates

    with np.errstate(invalid="ignore"):
        last_log_close = np.log(grid.close[:, -1])
    ret = np.concatenate([[np.nan], np.diff(last_log_close)])

    n_bars = grid.observed.sum(axis=1).astype(np.int64)
    complete = (n_bars >= min_bars) & valid.all(axis=1)

    index = pd.date_range(grid.first_day, periods=r.shape[0], freq="D", name=DATE_INDEX)
    panel = pd.DataFrame(
        {
            COL_RV: rv,
            COL_BPV: bpv,
            COL_JUMP: jump,
            COL_RS_POS: rs_pos,
            COL_RS_NEG: rs_neg,
            COL_RQ: rq,
            COL_RET: ret,
            COL_N_BARS: n_bars,
            COL_COMPLETE: complete,
        },
        index=index,
    )[list(PANEL_COLUMNS)]
    logger.info(
        "realized_panel_built",
        n_days=len(panel),
        n_complete=int(complete.sum()),
        min_bars=min_bars,
        max_ffill=max_ffill,
    )
    return panel


def signature_rv(
    klines: pd.DataFrame,
    intervals: tuple[str, ...] = DEFAULT_SIGNATURE_INTERVALS,
    max_ffill: int = 3,
) -> pd.DataFrame:
    """Average daily RV by sampling interval (data for a volatility signature plot).

    Closes are (forward-filled and) sampled every ``interval``; RV of a day is the sum of its
    squared log returns at that frequency, the first return of the day using the previous
    day's last close. Only days whose sampled returns are all valid enter the average.
    Each interval must be a multiple of 5 minutes dividing one day (e.g. ``"15min"``).

    Returns a DataFrame indexed by interval string with columns ``mean_rv`` and ``n_days``.
    """
    grid = _build_grid(klines, max_ffill)
    rows: dict[str, tuple[float, int]] = {}
    for interval in intervals:
        minutes = pd.Timedelta(interval) / pd.Timedelta(minutes=1)
        step = int(minutes) // 5
        if minutes % 5 != 0 or step < 1 or BARS_PER_DAY_5M % step != 0:
            raise ValueError(f"interval {interval!r} must be a multiple of 5min dividing 1 day")
        if grid is None:
            rows[interval] = (float("nan"), 0)
            continue
        with np.errstate(invalid="ignore"):
            lc = np.log(grid.close.reshape(-1)[step - 1 :: step])
        r = np.concatenate([[np.nan], np.diff(lc)]).reshape(grid.close.shape[0], -1)
        ok = np.isfinite(r).all(axis=1)
        daily = (r[ok] ** 2).sum(axis=1)
        rows[interval] = (float(daily.mean()) if daily.size else float("nan"), int(ok.sum()))
    return pd.DataFrame.from_dict(rows, orient="index", columns=["mean_rv", "n_days"]).astype(
        {"mean_rv": "float64", "n_days": "int64"}
    )
