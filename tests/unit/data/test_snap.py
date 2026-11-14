"""Grid snapping of off-grid Binance bars (e.g. the Feb 2018 post-outage block)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from tests.unit.data.test_binance import FakeArchive, csv_rows, daily_url, make_source, make_zip
from volforecast.data.binance import empty_klines, interval_timedelta, snap_to_grid
from volforecast.data.schemas import validate_klines


def _frame(times: list[str]) -> pd.DataFrame:
    n = len(times)
    return pd.DataFrame(
        {
            "open_time": pd.to_datetime(times, utc=True, format="ISO8601").astype(
                "datetime64[ns, UTC]"
            ),
            "open": [float(i + 1) for i in range(n)],
            "high": [float(i + 2) for i in range(n)],
            "low": [float(i) + 0.5 for i in range(n)],
            "close": [float(i + 1) for i in range(n)],
            "volume": [1.0] * n,
            "quote_volume": [1.0] * n,
            "n_trades": [1] * n,
        }
    )


def test_snap_noop_when_on_grid() -> None:
    df = _frame(["2018-02-09 00:00", "2018-02-09 00:05"])
    out, snapped, dropped = snap_to_grid(df, "5m")
    assert (snapped, dropped) == (0, 0)
    pd.testing.assert_frame_equal(out, df)


def test_snap_offset_block_floors_to_grid() -> None:
    # 241 consecutive bars offset by +194.789 s (the Feb 2018 post-outage block).
    start = pd.Timestamp("2018-02-09 09:58:14.789", tz="UTC")
    block = [start + pd.Timedelta(minutes=5 * i) for i in range(241)]
    first = pd.Timestamp("2018-02-09 09:45", tz="UTC")
    before = [first + pd.Timedelta(minutes=5 * i) for i in range(2)]
    df = _frame([str(t) for t in [*before, *block]])
    out, snapped, dropped = snap_to_grid(df, "5m")
    assert (snapped, dropped) == (241, 0)
    assert len(out) == 243
    assert out["open_time"].is_unique
    assert out["open_time"].is_monotonic_increasing
    assert out["open_time"].iloc[2] == pd.Timestamp("2018-02-09 09:55", tz="UTC")
    assert (out["open_time"].dt.as_unit("ns").astype("int64") % (300 * 10**9) == 0).all()
    validate_klines(out)  # the full contract holds after snapping
    assert out["close"].tolist() == df["close"].tolist()  # only timestamps move


def test_snap_collision_keeps_on_grid_bar() -> None:
    df = _frame(["2018-02-09 10:00:00", "2018-02-09 10:01:30", "2018-02-09 10:05:00"])
    out, snapped, dropped = snap_to_grid(df, "5m")
    assert (snapped, dropped) == (0, 1)
    assert out["close"].tolist() == [1.0, 3.0]  # the on-grid 10:00 bar survived


def test_snap_collision_on_grid_bar_listed_after_offgrid() -> None:
    df = _frame(["2018-02-09 10:01:30", "2018-02-09 10:00:00"])
    out, _, dropped = snap_to_grid(df, "5m")
    assert dropped == 1
    assert out["close"].tolist() == [2.0]


def test_snap_collision_between_two_off_grid_bars_keeps_first() -> None:
    df = _frame(["2018-02-09 10:01:00", "2018-02-09 10:03:00"])
    out, snapped, dropped = snap_to_grid(df, "5m")
    assert (snapped, dropped) == (1, 1)
    assert out["close"].tolist() == [1.0]
    assert out["open_time"].iloc[0] == pd.Timestamp("2018-02-09 10:00", tz="UTC")


def test_snap_empty_and_interval_parsing() -> None:
    out, snapped, dropped = snap_to_grid(empty_klines())
    assert out.empty
    assert (snapped, dropped) == (0, 0)
    assert interval_timedelta("1h") == pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match="unsupported"):
        interval_timedelta("abc")


def test_source_snaps_offset_file_and_counts(tmp_path: Path) -> None:
    arch = FakeArchive()
    lines = csv_rows("2018-02-09", 288).strip().split("\n")
    for i in range(100, 130):  # 30 consecutive bars offset by +194789 ms
        parts = lines[i].split(",")
        parts[0] = str(int(parts[0]) + 194_789)
        lines[i] = ",".join(parts)
    arch.add(daily_url("BTCUSDT", "2018-02-09"), make_zip("\n".join(lines) + "\n"))
    src = make_source(arch, tmp_path)
    df = src.fetch_klines("BTCUSDT", date(2018, 2, 9), date(2018, 2, 9))
    assert len(df) == 288
    assert (src.snapped_bars, src.dropped_bars) == (30, 0)
    validate_klines(df)
