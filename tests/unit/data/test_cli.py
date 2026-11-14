"""CLI ingest/quality with a fake source (no network)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from volforecast.data import __main__ as cli
from volforecast.data.store import read_klines


class FakeSource:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        self.source_checksums = {"fake.zip": "f" * 64}
        self.snapped_bars = 7

    def fetch_klines(
        self, symbol: str, start: date, end: date, interval: str = "5m"
    ) -> pd.DataFrame:
        return self.df


def test_ingest_writes_partitions_and_manifest(
    tmp_path: Path,
    synthetic_klines: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("VOLF_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(cli, "configure_logging", lambda *a, **k: None)  # keep global state
    # Remove 30 bars from one day so it counts as incomplete (< 280).
    df = synthetic_klines.drop(index=range(300, 330)).reset_index(drop=True)
    monkeypatch.setattr(cli, "BinanceVisionSource", lambda cache_dir: FakeSource(df))
    n_incomplete = cli.run_ingest("BTCUSDT", date(2024, 1, 1), date(2024, 2, 9))
    assert n_incomplete == 1
    base = tmp_path / "raw" / "klines" / "symbol=BTCUSDT" / "interval=5m"
    manifest = json.loads((base / "manifest.json").read_text())
    assert manifest["rows"] == len(df)
    assert manifest["extra"]["snapped_bars"] == 7
    assert manifest["source_checksums"] == {"fake.zip": "f" * 64}
    assert len(read_klines(tmp_path, "BTCUSDT")) == len(df)

    assert cli.main(["quality", "--symbol", "BTCUSDT"]) == 0
    out = capsys.readouterr().out
    assert "1 day(s) with < 280 bars" in out
    assert "2024-01-02" in out


def test_ingest_empty_result_exits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_klines: pd.DataFrame
) -> None:
    monkeypatch.setenv("VOLF_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(cli, "BinanceVisionSource", lambda cache_dir: FakeSource(synthetic_klines.iloc[:0]))
    with pytest.raises(SystemExit):
        cli.run_ingest("BTCUSDT", date(2024, 1, 1), date(2024, 1, 2))
