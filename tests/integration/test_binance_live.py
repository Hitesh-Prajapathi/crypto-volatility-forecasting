"""Live Binance Vision test (requires internet): one daily file, checksum-verified."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from volforecast.contracts import KLINE_COLUMNS
from volforecast.data.binance import BinanceVisionSource
from volforecast.data.schemas import validate_klines


@pytest.mark.network
def test_fetch_single_daily_file(tmp_path: Path) -> None:
    src = BinanceVisionSource(tmp_path)
    # Daily archive of a past day in the microsecond era; the end date lies inside its month,
    # so the daily URL is used.
    df = src.fetch_klines("BTCUSDT", date(2025, 3, 5), date(2025, 3, 5))
    assert tuple(df.columns) == KLINE_COLUMNS
    assert len(df) == 288
    assert df["open_time"].min() == pd.Timestamp("2025-03-05", tz="UTC")
    assert df["open_time"].max() == pd.Timestamp("2025-03-05 23:55", tz="UTC")
    validate_klines(df)
    assert list(src.source_checksums) == ["BTCUSDT-5m-2025-03-05.zip"]
    assert len(next(iter(src.source_checksums.values()))) == 64
