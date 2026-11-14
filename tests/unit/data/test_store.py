"""Parquet/DuckDB store, content hash and manifest."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from volforecast.contracts import KLINE_COLUMNS, PANEL_COLUMNS
from volforecast.data.binance import normalize_klines
from volforecast.data.schemas import validate_klines, validate_panel
from volforecast.data.store import (
    dataset_hash,
    klines_dir,
    read_klines,
    read_panel,
    write_klines,
    write_manifest,
    write_panel,
)


@pytest.fixture
def synthetic_klines(synthetic_klines: pd.DataFrame) -> pd.DataFrame:
    """conftest klines under pandas 3 may carry us resolution; the contract is ns."""
    return normalize_klines(synthetic_klines)


def test_klines_roundtrip_and_partitioning(tmp_path: Path, synthetic_klines: pd.DataFrame) -> None:
    # 40 days from 2024-01-01 -> January + February partitions.
    paths = write_klines(synthetic_klines, tmp_path, "BTCUSDT")
    base = tmp_path / "raw" / "klines" / "symbol=BTCUSDT" / "interval=5m"
    assert sorted(p.relative_to(base).as_posix() for p in paths) == [
        "year=2024/month=01/part.parquet",
        "year=2024/month=02/part.parquet",
    ]
    out = read_klines(tmp_path, "BTCUSDT")
    pd.testing.assert_frame_equal(out, synthetic_klines)
    assert tuple(out.columns) == KLINE_COLUMNS
    validate_klines(out)


def test_read_klines_date_filter_inclusive(tmp_path: Path, synthetic_klines: pd.DataFrame) -> None:
    write_klines(synthetic_klines, tmp_path, "BTCUSDT")
    out = read_klines(tmp_path, "BTCUSDT", date(2024, 1, 31), date(2024, 2, 2))
    assert out["open_time"].min() == pd.Timestamp("2024-01-31", tz="UTC")
    assert out["open_time"].max() == pd.Timestamp("2024-02-02 23:55", tz="UTC")
    assert len(out) == 3 * 288
    only_start = read_klines(tmp_path, "BTCUSDT", start=date(2024, 2, 5))
    assert only_start["open_time"].min() == pd.Timestamp("2024-02-05", tz="UTC")
    assert read_klines(tmp_path, "BTCUSDT", start=date(2030, 1, 1)).empty


def test_write_klines_is_idempotent(tmp_path: Path, synthetic_klines: pd.DataFrame) -> None:
    write_klines(synthetic_klines, tmp_path, "BTCUSDT")
    write_klines(synthetic_klines, tmp_path, "BTCUSDT")
    out = read_klines(tmp_path, "BTCUSDT")
    pd.testing.assert_frame_equal(out, synthetic_klines)
    assert len(list(klines_dir(tmp_path, "BTCUSDT").rglob("part.parquet"))) == 2
    assert not list(tmp_path.rglob("*.tmp"))


def test_partition_is_rewritten_fully(tmp_path: Path, synthetic_klines: pd.DataFrame) -> None:
    write_klines(synthetic_klines, tmp_path, "BTCUSDT")
    jan = synthetic_klines[synthetic_klines["open_time"] < pd.Timestamp("2024-01-10", tz="UTC")]
    write_klines(jan, tmp_path, "BTCUSDT")  # replaces the January partition only
    out = read_klines(tmp_path, "BTCUSDT")
    assert len(out) == len(jan) + int(
        (synthetic_klines["open_time"] >= pd.Timestamp("2024-02-01", tz="UTC")).sum()
    )


def test_symbols_are_isolated(tmp_path: Path, synthetic_klines: pd.DataFrame) -> None:
    write_klines(synthetic_klines, tmp_path, "BTCUSDT")
    with pytest.raises(FileNotFoundError):
        read_klines(tmp_path, "ETHUSDT")


def test_real_fixture_roundtrip(tmp_path: Path) -> None:
    fx = pd.read_parquet(Path(__file__).parents[2] / "fixtures" / "btcusdt_5m_sample.parquet")
    write_klines(fx, tmp_path, "BTCUSDT")
    pd.testing.assert_frame_equal(read_klines(tmp_path, "BTCUSDT"), fx)


def test_panel_roundtrip_preserves_index(tmp_path: Path, synthetic_panel: pd.DataFrame) -> None:
    path = write_panel(synthetic_panel, tmp_path, "BTCUSDT")
    assert path == tmp_path / "daily" / "symbol=BTCUSDT" / "panel.parquet"
    out = read_panel(tmp_path, "BTCUSDT")
    assert out.index.name == "date"
    assert tuple(out.columns) == PANEL_COLUMNS
    pd.testing.assert_frame_equal(out, synthetic_panel, check_freq=False)
    validate_panel(out)


def test_read_panel_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_panel(tmp_path, "BTCUSDT")


def test_hash_ignores_datetime_resolution(synthetic_klines: pd.DataFrame) -> None:
    coarse = synthetic_klines.copy()
    coarse["open_time"] = coarse["open_time"].dt.as_unit("us")
    assert dataset_hash(coarse) == dataset_hash(synthetic_klines)


def test_hash_deterministic(synthetic_klines: pd.DataFrame) -> None:
    h = dataset_hash(synthetic_klines)
    assert h == dataset_hash(synthetic_klines.copy())
    assert len(h) == 64
    assert h == dataset_hash(pd.DataFrame(synthetic_klines))


def test_hash_survives_storage_roundtrip(tmp_path: Path, synthetic_klines: pd.DataFrame) -> None:
    write_klines(synthetic_klines, tmp_path, "BTCUSDT")
    assert dataset_hash(read_klines(tmp_path, "BTCUSDT")) == dataset_hash(synthetic_klines)


def test_hash_sensitivity(synthetic_klines: pd.DataFrame) -> None:
    h = dataset_hash(synthetic_klines)
    changed = synthetic_klines.copy()
    changed.loc[100, "close"] = changed.loc[100, "close"] * (1 + 1e-12)
    assert dataset_hash(changed) != h
    assert dataset_hash(synthetic_klines.iloc[:-1]) != h
    assert dataset_hash(synthetic_klines.rename(columns={"close": "px"})) != h
    retyped = synthetic_klines.copy()
    retyped["n_trades"] = retyped["n_trades"].astype("int32")
    assert dataset_hash(retyped) != h
    swapped = synthetic_klines.copy()
    swapped.loc[[0, 1], "close"] = swapped.loc[[1, 0], "close"].to_numpy()
    assert dataset_hash(swapped) != h


def test_hash_includes_panel_index(synthetic_panel: pd.DataFrame) -> None:
    shifted = synthetic_panel.copy()
    shifted.index = shifted.index + pd.Timedelta(days=1)
    assert dataset_hash(shifted) != dataset_hash(synthetic_panel)


def test_manifest_fields(tmp_path: Path, synthetic_klines: pd.DataFrame) -> None:
    path = tmp_path / "sub" / "manifest.json"
    returned = write_manifest(
        path,
        dataset="klines/BTCUSDT/5m",
        df=synthetic_klines,
        source_checksums={"b.zip": "2" * 64, "a.zip": "1" * 64},
        extra={"source": "binance"},
    )
    on_disk = json.loads(path.read_text())
    assert on_disk == returned
    assert on_disk["dataset"] == "klines/BTCUSDT/5m"
    assert on_disk["rows"] == len(synthetic_klines)
    assert on_disk["min_time"] == "2024-01-01T00:00:00+00:00"
    assert pd.Timestamp(on_disk["max_time"]) == synthetic_klines["open_time"].max()
    assert on_disk["content_hash"] == dataset_hash(synthetic_klines)
    assert on_disk["source_checksums"] == {"a.zip": "1" * 64, "b.zip": "2" * 64}
    assert on_disk["extra"] == {"source": "binance"}
    assert on_disk["volforecast_version"] == "0.1.0"
    assert on_disk["git_sha"] == "unknown" or len(on_disk["git_sha"]) == 40
    created = pd.Timestamp(on_disk["created_at"])
    assert created.tzinfo is not None


def test_manifest_for_panel_uses_index_bounds(
    tmp_path: Path, synthetic_panel: pd.DataFrame
) -> None:
    m = write_manifest(
        tmp_path / "m.json", dataset="panel", df=synthetic_panel, source_checksums={}, extra=None
    )
    assert m["min_time"].startswith("2019-01-01")
    assert m["extra"] == {}
