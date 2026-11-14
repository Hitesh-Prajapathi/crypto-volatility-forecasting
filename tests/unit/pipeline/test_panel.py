from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from tests.conftest import make_synthetic_klines
from volforecast.contracts import KLINE_COLUMNS, PANEL_COLUMNS
from volforecast.features.realized import daily_realized_measures
from volforecast.pipeline import panel as panel_mod
from volforecast.pipeline.panel import build_panel, content_hash, load_klines, panel_paths

SYMBOL = "TESTUSDT"


@pytest.fixture
def hive_root(tmp_path: Path) -> tuple[Path, pd.DataFrame]:
    """Synthetic klines written as a hive-partitioned tree, one parquet per month."""
    kl = make_synthetic_klines(n_days=45, seed=3, start="2024-01-01")
    base = tmp_path / "raw" / "klines" / f"symbol={SYMBOL}" / "interval=5m"
    month = kl["open_time"].dt.month
    for m, part in kl.groupby(month):
        d = base / "year=2024" / f"month={m:02d}"
        d.mkdir(parents=True)
        part.to_parquet(d / "part.parquet", index=False)
    return tmp_path, kl


def test_load_klines_matches_contract_schema(hive_root: tuple[Path, pd.DataFrame]) -> None:
    root, kl = hive_root
    got = load_klines(root, SYMBOL)
    assert tuple(got.columns) == KLINE_COLUMNS
    assert str(got["open_time"].dtype) == "datetime64[ns, UTC]"
    assert got["n_trades"].dtype == "int64"
    assert got["open_time"].is_monotonic_increasing
    assert got["open_time"].is_unique
    assert isinstance(got.index, pd.RangeIndex)
    want = kl[list(KLINE_COLUMNS)].copy()
    want["open_time"] = want["open_time"].astype("datetime64[ns, UTC]")
    pd.testing.assert_frame_equal(got, want, check_freq=False)


def test_load_klines_missing_symbol_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_klines(tmp_path, "NOPEUSDT")


def test_build_panel_caches_and_hashes(
    hive_root: tuple[Path, pd.DataFrame], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, kl = hive_root
    calls: list[int] = []
    real = daily_realized_measures

    def spy(klines: pd.DataFrame) -> pd.DataFrame:
        calls.append(len(klines))
        return real(klines)

    monkeypatch.setattr(panel_mod, "daily_realized_measures", spy)

    panel, digest = build_panel(root, SYMBOL)
    assert list(panel.columns) == list(PANEL_COLUMNS)
    pd.testing.assert_frame_equal(panel, real(kl), check_freq=False)
    assert digest == content_hash(panel)
    assert len(digest) == 64
    pq, meta = panel_paths(root, SYMBOL)
    assert pq.exists()
    assert json.loads(meta.read_text())["hash"] == digest
    assert calls == [len(kl)]

    panel2, digest2 = build_panel(root, SYMBOL)  # cache hit: no recompute
    assert digest2 == digest
    assert len(calls) == 1
    pd.testing.assert_frame_equal(panel2, panel, check_freq=False)

    build_panel(root, SYMBOL, force=True)
    assert len(calls) == 2


def test_cache_invalidated_when_raw_changes_or_cache_corrupt(
    hive_root: tuple[Path, pd.DataFrame], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _kl = hive_root
    _, digest = build_panel(root, SYMBOL)
    calls: list[int] = []
    real = daily_realized_measures
    monkeypatch.setattr(
        panel_mod, "daily_realized_measures", lambda k: (calls.append(1), real(k))[1]
    )

    # corrupt the cached parquet: hash mismatch triggers a rebuild with the original hash
    pq, _ = panel_paths(root, SYMBOL)
    tampered = pd.read_parquet(pq)
    tampered.iloc[3, 0] = 123.0
    tampered.to_parquet(pq)
    _, digest2 = build_panel(root, SYMBOL)
    assert len(calls) == 1
    assert digest2 == digest

    # change the raw data: signature changes, content hash changes
    base = root / "raw" / "klines" / f"symbol={SYMBOL}" / "interval=5m"
    last = sorted(base.rglob("*.parquet"))[-1]
    df = pd.read_parquet(last)
    df["close"] = df["close"] * 1.01
    df.iloc[len(df) // 2 :, df.columns.get_loc("close")] *= 1.02
    df.to_parquet(last, index=False)
    _, digest3 = build_panel(root, SYMBOL)
    assert len(calls) == 2
    assert digest3 != digest


def test_content_hash_is_sensitive_and_stable(synthetic_panel: pd.DataFrame) -> None:
    h1 = content_hash(synthetic_panel)
    assert h1 == content_hash(synthetic_panel.copy())
    changed = synthetic_panel.copy()
    changed.iloc[10, 0] += 1e-12
    assert content_hash(changed) != h1
