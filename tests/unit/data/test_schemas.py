"""Pandera schemas and the kline quality report."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from volforecast.contracts import (
    COL_BPV,
    COL_COMPLETE,
    COL_N_BARS,
    COL_RS_NEG,
    COL_RS_POS,
    COL_RV,
)
from volforecast.data.schemas import (
    DataValidationError,
    kline_quality_report,
    validate_klines,
    validate_panel,
)


# ── klines ───────────────────────────────────────────────────────────────────────────
def test_synthetic_klines_pass(synthetic_klines: pd.DataFrame) -> None:
    validate_klines(synthetic_klines)


def test_real_fixture_passes() -> None:
    from pathlib import Path

    df = pd.read_parquet(Path(__file__).parents[2] / "fixtures" / "btcusdt_5m_sample.parquet")
    validate_klines(df)
    assert len(df) == 17_280


def test_empty_klines_fail_gracefully_or_pass() -> None:
    from volforecast.data.binance import empty_klines

    validate_klines(empty_klines())


def test_duplicate_open_time_fails(synthetic_klines: pd.DataFrame) -> None:
    bad = pd.concat([synthetic_klines, synthetic_klines.iloc[[5]]], ignore_index=True)
    with pytest.raises(DataValidationError, match="open_time is not unique"):
        validate_klines(bad)


def test_unsorted_fails(synthetic_klines: pd.DataFrame) -> None:
    bad = synthetic_klines.iloc[::-1].reset_index(drop=True)
    with pytest.raises(DataValidationError, match="not sorted ascending"):
        validate_klines(bad)


def test_off_grid_fails(synthetic_klines: pd.DataFrame) -> None:
    bad = synthetic_klines.copy()
    bad.loc[10, "open_time"] += pd.Timedelta(seconds=30)
    with pytest.raises(DataValidationError, match="aligned to 5m grid"):
        validate_klines(bad)


def test_non_positive_price_fails(synthetic_klines: pd.DataFrame) -> None:
    bad = synthetic_klines.copy()
    bad.loc[3, "close"] = 0.0
    with pytest.raises(DataValidationError, match="close"):
        validate_klines(bad)


def test_high_below_close_fails(synthetic_klines: pd.DataFrame) -> None:
    bad = synthetic_klines.copy()
    bad.loc[7, "high"] = bad.loc[7, "close"] * 0.99
    with pytest.raises(DataValidationError, match="high below max"):
        validate_klines(bad)


def test_low_above_open_fails(synthetic_klines: pd.DataFrame) -> None:
    bad = synthetic_klines.copy()
    bad.loc[7, "low"] = bad.loc[7, "open"] * 1.01
    with pytest.raises(DataValidationError, match="low above min"):
        validate_klines(bad)


def test_ohlc_relative_tolerance_allows_rounding(synthetic_klines: pd.DataFrame) -> None:
    ok = synthetic_klines.copy()
    ok.loc[7, "high"] = max(ok.loc[7, "open"], ok.loc[7, "close"]) * (1 - 1e-10)
    validate_klines(ok)


def test_negative_volume_fails(synthetic_klines: pd.DataFrame) -> None:
    bad = synthetic_klines.copy()
    bad.loc[2, "volume"] = -1.0
    with pytest.raises(DataValidationError, match="volume"):
        validate_klines(bad)


def test_wrong_dtype_and_missing_column_fail(synthetic_klines: pd.DataFrame) -> None:
    bad = synthetic_klines.copy()
    bad["n_trades"] = bad["n_trades"].astype("float64")
    with pytest.raises(DataValidationError, match="n_trades"):
        validate_klines(bad)
    with pytest.raises(DataValidationError, match="quote_volume"):
        validate_klines(synthetic_klines.drop(columns="quote_volume"))


def test_tz_naive_open_time_fails(synthetic_klines: pd.DataFrame) -> None:
    bad = synthetic_klines.copy()
    bad["open_time"] = bad["open_time"].dt.tz_localize(None)
    with pytest.raises(DataValidationError, match="open_time"):
        validate_klines(bad)


def test_error_lists_all_failures(synthetic_klines: pd.DataFrame) -> None:
    bad = synthetic_klines.copy()
    bad.loc[3, "close"] = -1.0
    bad.loc[4, "volume"] = -5.0
    with pytest.raises(DataValidationError) as exc:
        validate_klines(bad)
    msg = str(exc.value)
    assert "close" in msg
    assert "volume" in msg


# ── panel ────────────────────────────────────────────────────────────────────────────
def test_synthetic_panel_passes(synthetic_panel: pd.DataFrame) -> None:
    validate_panel(synthetic_panel)


def test_incomplete_rows_may_hold_nan(synthetic_panel: pd.DataFrame) -> None:
    p = synthetic_panel.copy()
    p.iloc[10, p.columns.get_loc(COL_COMPLETE)] = False  # type: ignore[index]
    p.iloc[10, [p.columns.get_loc(c) for c in (COL_RV, COL_BPV, COL_RS_POS, COL_RS_NEG)]] = np.nan  # type: ignore[index]
    p.iloc[10, p.columns.get_loc(COL_N_BARS)] = 100  # type: ignore[index]
    validate_panel(p)


def test_complete_row_with_nan_rv_fails(synthetic_panel: pd.DataFrame) -> None:
    p = synthetic_panel.copy()
    p.iloc[10, p.columns.get_loc(COL_RV)] = np.nan  # type: ignore[index]
    with pytest.raises(DataValidationError, match="complete day with rv"):
        validate_panel(p)


@pytest.mark.parametrize(
    ("col", "value", "check"),
    [
        ("rv", 0.0, "rv"),
        ("bpv", -1e-6, "bpv"),
        ("jump", -1e-6, "jump"),
        ("rs_pos", -1e-6, "rs_pos"),
        ("rs_neg", -1e-6, "rs_neg"),
        ("rq", -1e-6, "rq"),
    ],
)
def test_negative_measures_fail(
    synthetic_panel: pd.DataFrame, col: str, value: float, check: str
) -> None:
    p = synthetic_panel.copy()
    p.iloc[20, p.columns.get_loc(col)] = value  # type: ignore[index]
    with pytest.raises(DataValidationError, match=check):
        validate_panel(p)


def test_semivariance_identity_fails(synthetic_panel: pd.DataFrame) -> None:
    p = synthetic_panel.copy()
    p.iloc[5, p.columns.get_loc(COL_RS_POS)] *= 1.01  # type: ignore[index]
    with pytest.raises(DataValidationError, match=r"rs_pos \+ rs_neg - rv"):
        validate_panel(p)


@pytest.mark.parametrize("bad_n", [-1, 289])
def test_n_bars_out_of_range_fails(synthetic_panel: pd.DataFrame, bad_n: int) -> None:
    p = synthetic_panel.copy()
    p.iloc[3, p.columns.get_loc(COL_N_BARS)] = bad_n  # type: ignore[index]
    with pytest.raises(DataValidationError, match=COL_N_BARS):
        validate_panel(p)


def test_complete_must_be_bool(synthetic_panel: pd.DataFrame) -> None:
    p = synthetic_panel.copy()
    p[COL_COMPLETE] = p[COL_COMPLETE].astype("int64")
    with pytest.raises(DataValidationError, match=COL_COMPLETE):
        validate_panel(p)


def test_panel_index_gap_fails(synthetic_panel: pd.DataFrame) -> None:
    p = synthetic_panel.drop(synthetic_panel.index[50])
    with pytest.raises(DataValidationError, match="consecutive"):
        validate_panel(p)


def test_panel_index_duplicate_unsorted_naive_named(synthetic_panel: pd.DataFrame) -> None:
    dup = pd.concat([synthetic_panel, synthetic_panel.iloc[[3]]])
    with pytest.raises(DataValidationError, match="unique"):
        validate_panel(dup)
    with pytest.raises(DataValidationError, match="sorted"):
        validate_panel(synthetic_panel.iloc[::-1])
    tz = synthetic_panel.copy()
    tz.index = tz.index.tz_localize("UTC")
    with pytest.raises(DataValidationError, match="tz-naive"):
        validate_panel(tz)
    unnamed = synthetic_panel.rename_axis(None)
    with pytest.raises(DataValidationError, match="named 'date'"):
        validate_panel(unnamed)
    shifted = synthetic_panel.copy()
    shifted.index = shifted.index + pd.Timedelta(hours=3)
    with pytest.raises(DataValidationError, match="midnight"):
        validate_panel(shifted)


# ── quality report ───────────────────────────────────────────────────────────────────
def test_quality_report_full_days(synthetic_klines: pd.DataFrame) -> None:
    rep = kline_quality_report(synthetic_klines)
    assert len(rep) == 40
    assert (rep["n_bars"] == 288).all()
    assert (rep["n_missing"] == 0).all()
    assert (rep["longest_gap_bars"] == 0).all()
    assert rep.index.name == "date"
    assert rep.index.tz is None


def test_quality_report_gaps_and_empty_days(synthetic_klines: pd.DataFrame) -> None:
    df = synthetic_klines.copy()
    day0 = df["open_time"].dt.floor("D") == df["open_time"].iloc[0].floor("D")
    day1 = df["open_time"].dt.floor("D") == df["open_time"].iloc[0].floor("D") + pd.Timedelta(
        days=1
    )
    day3 = df["open_time"].dt.floor("D") == df["open_time"].iloc[0].floor("D") + pd.Timedelta(
        days=3
    )
    drop = pd.Series(False, index=df.index)
    d0 = df.index[day0]
    drop.loc[d0[10:13]] = True  # gap of 3
    drop.loc[d0[100:102]] = True  # gap of 2
    drop.loc[day1] = True  # whole day missing
    drop.loc[df.index[day3][-4:]] = True  # trailing gap of 4 at the end of the day
    rep = kline_quality_report(df[~drop].reset_index(drop=True))
    assert rep.iloc[0].to_dict() == {"n_bars": 283, "n_missing": 5, "longest_gap_bars": 3}
    assert rep.iloc[1].to_dict() == {"n_bars": 0, "n_missing": 288, "longest_gap_bars": 288}
    assert rep.iloc[2]["n_missing"] == 0
    assert rep.iloc[3].to_dict() == {"n_bars": 284, "n_missing": 4, "longest_gap_bars": 4}
    assert int((rep["n_bars"] < 280).sum()) == 1


def test_quality_report_empty() -> None:
    from volforecast.data.binance import empty_klines

    assert kline_quality_report(empty_klines()).empty
