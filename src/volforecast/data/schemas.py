"""Pandera schemas for klines and the daily realized panel, plus a kline quality report."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors

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
    KLINE_COLUMNS,
    PANEL_COLUMNS,
)

#: Relative tolerance for OHLC consistency and the semivariance identity.
REL_TOL = 1e-9
BAR_NS = 5 * 60 * 10**9


class DataValidationError(ValueError):
    """Raised by :func:`validate_klines` / :func:`validate_panel` listing every failure."""


# ── Klines ───────────────────────────────────────────────────────────────────────────
def _on_grid(s: pd.Series) -> pd.Series:
    ns = s.dt.as_unit("ns").astype("int64").to_numpy()
    return pd.Series(ns % BAR_NS == 0, index=s.index)


def _utc_datetime(s: pd.Series) -> bool:
    """tz-aware UTC datetimes. Any resolution is accepted (pandas 3 may infer us, not ns)."""
    dtype = s.dtype
    return isinstance(dtype, pd.DatetimeTZDtype) and str(dtype.tz) == "UTC"


def _unique(s: pd.Series) -> bool:
    return bool(s.is_unique)


def _monotonic(s: pd.Series) -> bool:
    return bool(s.is_monotonic_increasing)


def _high_ok(df: pd.DataFrame) -> pd.Series:
    top = df[["open", "close"]].max(axis=1)
    return df["high"] >= top * (1 - REL_TOL)


def _low_ok(df: pd.DataFrame) -> pd.Series:
    bottom = df[["open", "close"]].min(axis=1)
    return df["low"] <= bottom * (1 + REL_TOL)


def _range_index(df: pd.DataFrame) -> bool:
    return isinstance(df.index, pd.RangeIndex) and df.index.start == 0 and df.index.step == 1


_price = pa.Column(float, pa.Check.gt(0), nullable=False)

KLINES_SCHEMA = pa.DataFrameSchema(
    {
        "open_time": pa.Column(
            checks=[
                pa.Check(
                    _utc_datetime, name="open_time_utc", error="open_time is not tz-aware UTC"
                ),
                pa.Check(_unique, name="open_time_unique", error="open_time is not unique"),
                pa.Check(
                    _monotonic, name="open_time_sorted", error="open_time is not sorted ascending"
                ),
                pa.Check(_on_grid, name="open_time_on_5m_grid", error="not aligned to 5m grid"),
            ],
            nullable=False,
        ),
        "open": _price,
        "high": _price,
        "low": _price,
        "close": _price,
        "volume": pa.Column(float, pa.Check.ge(0), nullable=False),
        "quote_volume": pa.Column(float, pa.Check.ge(0), nullable=False),
        "n_trades": pa.Column("int64", pa.Check.ge(0), nullable=False),
    },
    checks=[
        pa.Check(_high_ok, name="high>=max(open,close)", error="high below max(open, close)"),
        pa.Check(_low_ok, name="low<=min(open,close)", error="low above min(open, close)"),
        pa.Check(_range_index, name="range_index", error="index is not a 0-based RangeIndex"),
    ],
    strict=True,
    ordered=True,
    coerce=False,
)


# ── Daily panel ──────────────────────────────────────────────────────────────────────
def _complete_only(df: pd.DataFrame, cond: pd.Series) -> pd.Series:
    """Rows pass if incomplete, or if ``cond`` holds (NaN comparisons count as failures)."""
    return ~df[COL_COMPLETE].astype(bool) | cond.fillna(False)


def _semivar_identity(df: pd.DataFrame) -> pd.Series:
    gap = (df[COL_RS_POS] + df[COL_RS_NEG] - df[COL_RV]).abs()
    return _complete_only(df, gap <= REL_TOL * df[COL_RV])


def _index_ok(df: pd.DataFrame) -> bool:
    idx = df.index
    return bool(isinstance(idx, pd.DatetimeIndex) and idx.tz is None and idx.is_unique)


def _index_normalized(df: pd.DataFrame) -> bool:
    idx = df.index
    return bool(isinstance(idx, pd.DatetimeIndex) and (idx == idx.normalize()).all())


def _index_sorted(df: pd.DataFrame) -> bool:
    return bool(df.index.is_monotonic_increasing)


def _consecutive_days(df: pd.DataFrame) -> bool:
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex) or idx.tz is not None or len(idx) < 2:
        return True  # other checks report a bad index type
    diffs = np.diff(idx.to_numpy().astype("datetime64[D]").astype("int64"))
    return bool((diffs == 1).all())


def _index_name(df: pd.DataFrame) -> bool:
    return bool(df.index.name == DATE_INDEX)


def _measure(name: str, strict_positive: bool = False) -> pa.Column:
    op = (lambda s: s > 0) if strict_positive else (lambda s: s >= 0)
    return pa.Column(
        float,
        pa.Check(
            lambda s, _op=op: s.isna() | _op(s),
            name=f"{name}_{'positive' if strict_positive else 'nonnegative'}_when_present",
            error=f"{name} violates sign constraint",
        ),
        nullable=True,
    )


PANEL_SCHEMA = pa.DataFrameSchema(
    {
        COL_RV: _measure(COL_RV, strict_positive=True),
        COL_BPV: _measure(COL_BPV),
        COL_JUMP: _measure(COL_JUMP),
        COL_RS_POS: _measure(COL_RS_POS),
        COL_RS_NEG: _measure(COL_RS_NEG),
        COL_RQ: _measure(COL_RQ),
        COL_RET: pa.Column(float, nullable=True),
        COL_N_BARS: pa.Column(
            "int64",
            pa.Check.in_range(0, BARS_PER_DAY_5M),
            nullable=False,
        ),
        COL_COMPLETE: pa.Column(bool, nullable=False),
    },
    checks=[
        # Sign constraints apply to *complete* rows only; they must also be non-NaN there.
        pa.Check(
            lambda df: _complete_only(df, df[COL_RV] > 0),
            name="complete_rv_positive",
            error="complete day with rv missing or <= 0",
        ),
        pa.Check(
            lambda df: _complete_only(df, df[COL_BPV] >= 0),
            name="complete_bpv_nonnegative",
            error="complete day with bpv missing or < 0",
        ),
        pa.Check(
            lambda df: _complete_only(df, df[COL_JUMP] >= 0),
            name="complete_jump_nonnegative",
            error="complete day with jump missing or < 0",
        ),
        pa.Check(
            lambda df: _complete_only(df, (df[COL_RS_POS] >= 0) & (df[COL_RS_NEG] >= 0)),
            name="complete_semivariances_nonnegative",
            error="complete day with rs_pos/rs_neg missing or < 0",
        ),
        pa.Check(
            lambda df: _complete_only(df, df[COL_RQ] >= 0),
            name="complete_rq_nonnegative",
            error="complete day with rq missing or < 0",
        ),
        pa.Check(
            _semivar_identity,
            name="rs_pos+rs_neg==rv",
            error="|rs_pos + rs_neg - rv| > 1e-9 * rv on a complete day",
        ),
        pa.Check(_index_name, name="index_named_date", error="index must be named 'date'"),
        pa.Check(
            _index_ok, name="index_naive_unique", error="index must be tz-naive with unique labels"
        ),
        pa.Check(_index_normalized, name="index_normalized", error="index labels must be midnight"),
        pa.Check(_index_sorted, name="index_sorted", error="index must be sorted ascending"),
        pa.Check(
            _consecutive_days,
            name="index_consecutive_days",
            error="index must contain consecutive calendar days",
        ),
    ],
    strict=True,
    ordered=True,
    coerce=False,
)

assert tuple(KLINES_SCHEMA.columns) == KLINE_COLUMNS
assert tuple(PANEL_SCHEMA.columns) == PANEL_COLUMNS


def _format_failures(name: str, exc: SchemaErrors) -> str:
    """One line per failed check: affected rows, columns (if few) and example rows."""
    cases = exc.failure_cases
    grouped = list(cases.groupby("check", dropna=False, sort=False))
    lines = [f"{name} validation failed: {len(grouped)} check(s) failed"]
    for check, group in grouped:
        rows = group["index"].dropna().unique()
        n_rows = len(rows) if len(rows) else len(group)
        columns = [str(c) for c in group["column"].dropna().unique()]
        where = f" in column(s) {', '.join(columns)}" if 0 < len(columns) <= 3 else ""
        example_rows = [str(r) for r in rows[:3]]
        if example_rows:
            at = f" (rows e.g. {', '.join(example_rows)})"
        else:  # schema-level failure (missing column, dtype): show the offending values
            at = f" ({', '.join(str(v) for v in group['failure_case'].head(3))})"
        lines.append(f"  - {check}: {n_rows} case(s){where}{at}")
    return "\n".join(lines)


def validate_klines(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a kline frame against the contract; raise :class:`DataValidationError`.

    All failures are collected (``lazy=True``) and listed in the error message.
    """
    try:
        return KLINES_SCHEMA.validate(df, lazy=True)
    except SchemaErrors as exc:
        raise DataValidationError(_format_failures("klines", exc)) from exc


def validate_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a daily realized panel against the contract; raise :class:`DataValidationError`."""
    try:
        return PANEL_SCHEMA.validate(df, lazy=True)
    except SchemaErrors as exc:
        raise DataValidationError(_format_failures("panel", exc)) from exc


# ── Quality report ───────────────────────────────────────────────────────────────────
def kline_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-UTC-day completeness of a 5-minute kline frame.

    Covers every calendar day from the first to the last bar (days with no bars at all appear
    with ``n_bars == 0``). Columns: ``n_bars`` (observed bars), ``n_missing``
    (``288 - n_bars``) and ``longest_gap_bars`` (longest run of consecutive missing bars on
    the day's 288-bar grid). Index is the tz-naive ``date``.
    """
    cols = ["n_bars", "n_missing", "longest_gap_bars"]
    if df.empty:
        return pd.DataFrame(
            {c: pd.Series([], dtype="int64") for c in cols},
            index=pd.DatetimeIndex([], name=DATE_INDEX),
        )
    t = df["open_time"]
    first = t.min().floor("D")
    last = t.max().floor("D")
    n_days = int((last - first) / pd.Timedelta(days=1)) + 1
    slot = ((t - first) // pd.Timedelta(minutes=5)).to_numpy().astype("int64")
    flat = np.zeros(n_days * BARS_PER_DAY_5M, dtype=bool)
    flat[slot] = True
    present = flat.reshape(n_days, BARS_PER_DAY_5M)

    missing = ~present
    csum = np.cumsum(missing, axis=1)
    reset = np.maximum.accumulate(np.where(present, csum, 0), axis=1)
    longest = (csum - reset).max(axis=1)
    n_bars = present.sum(axis=1)

    index = pd.DatetimeIndex(
        pd.date_range(first.tz_localize(None), periods=n_days, freq="D"), name=DATE_INDEX
    )
    return pd.DataFrame(
        {
            "n_bars": n_bars.astype("int64"),
            "n_missing": (BARS_PER_DAY_5M - n_bars).astype("int64"),
            "longest_gap_bars": longest.astype("int64"),
        },
        index=index,
    )
