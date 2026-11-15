"""Ingest flow with an in-memory source (no network)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from tests.conftest import make_synthetic_klines
from tests.unit.flows.conftest import ASSET
from volforecast.config import Settings
from volforecast.data.store import read_klines, write_klines
from volforecast.flows import ingest as ingest_mod
from volforecast.flows.common import Runtime
from volforecast.flows.ingest import ingest_asset, ingest_flow, last_stored_day
from volforecast.flows.ingest import main as ingest_main
from volforecast.forecast_store import ForecastRepository
from volforecast.pipeline.panel import build_panel
from volforecast.registry.config import load_production_config

N_STORED, N_TOTAL = 120, 139  # 2024-01-01 + 120 days -> last stored day 2024-04-29


class FakeSource:
    """Serves slices of one synthetic kline series, like the archive would."""

    def __init__(self, full: pd.DataFrame) -> None:
        self.full = full
        self.calls: list[tuple[date, date]] = []
        self.source_checksums: dict[str, str] = {}

    def fetch_klines(
        self, symbol: str, start: date, end: date, interval: str = "5m"
    ) -> pd.DataFrame:
        self.calls.append((start, end))
        lo = pd.Timestamp(start, tz="UTC")
        hi = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
        t = self.full["open_time"]
        return self.full[(t >= lo) & (t < hi)].reset_index(drop=True)


@pytest.fixture
def rt_and_source(tmp_path: Path, tiny_config: Path) -> tuple[Runtime, FakeSource, pd.DataFrame]:
    full = make_synthetic_klines(N_TOTAL, seed=5, start="2024-01-01")
    root = tmp_path / "data"
    cut = pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(days=N_STORED)
    write_klines(full[full["open_time"] < cut], root, ASSET)
    settings = Settings(
        data_dir=root,
        artifacts_dir=tmp_path / "artifacts",
        database_url=f"sqlite:///{tmp_path / 'store.db'}",
        mlflow_tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        production_config=tiny_config,
    )
    rt = Runtime(
        settings, load_production_config(tiny_config), ForecastRepository(settings.database_url)
    )
    return rt, FakeSource(full), full


def test_ingest_appends_new_days_and_syncs_realized(
    rt_and_source: tuple[Runtime, FakeSource, pd.DataFrame],
) -> None:
    rt, source, full = rt_and_source
    assert last_stored_day(rt.data_dir, ASSET) == date(2024, 4, 29)
    _, hash_before = build_panel(rt.data_dir, ASSET)

    out = ingest_asset(rt, ASSET, source, end=date(2024, 5, 18))

    assert out["status"] == "ok"
    assert source.calls == [(date(2024, 4, 29), date(2024, 5, 18))]  # last day is re-fetched
    stored = read_klines(rt.data_dir, ASSET)
    assert len(stored) == len(full)
    assert stored["open_time"].is_unique
    assert stored["open_time"].is_monotonic_increasing
    assert out["new_bars"] == len(full) - 120 * 288
    assert last_stored_day(rt.data_dir, ASSET) == date(2024, 5, 18)
    assert out["panel_hash"] != hash_before
    assert rt.repo.last_realized_date(ASSET) == date(2024, 5, 18)
    bar = rt.repo.get_metrics("data_last_bar_ts", asset=ASSET)
    assert float(bar["value"].max()) == pd.Timestamp("2024-05-18 23:55", tz="UTC").timestamp()
    assert (rt.data_dir / "raw/klines/symbol=BTCUSDT/interval=5m/manifest.json").exists()


def test_second_ingest_is_idempotent(
    rt_and_source: tuple[Runtime, FakeSource, pd.DataFrame],
) -> None:
    rt, source, _ = rt_and_source
    ingest_asset(rt, ASSET, source, end=date(2024, 5, 18))
    before = read_klines(rt.data_dir, ASSET)
    out = ingest_asset(rt, ASSET, source, end=date(2024, 5, 18))
    assert out["new_bars"] == 0
    pd.testing.assert_frame_equal(read_klines(rt.data_dir, ASSET), before)
    # nothing newer than the last stored day -> no fetch at all
    calls = len(source.calls)
    assert ingest_asset(rt, ASSET, source, end=date(2024, 5, 17))["status"] == "up_to_date"
    assert len(source.calls) == calls


def test_empty_fetch_is_reported_not_failed(
    rt_and_source: tuple[Runtime, FakeSource, pd.DataFrame],
) -> None:
    rt, _, full = rt_and_source
    empty = FakeSource(full.iloc[:0])
    out = ingest_asset(rt, ASSET, empty, end=date(2024, 5, 18))
    assert out["status"] == "no_new_data"


def test_missing_store_requires_an_explicit_start(
    rt_and_source: tuple[Runtime, FakeSource, pd.DataFrame], tmp_path: Path
) -> None:
    rt, source, _ = rt_and_source
    rt.settings = rt.settings.model_copy(update={"data_dir": tmp_path / "nothing"})
    with pytest.raises(RuntimeError, match="no stored data"):
        ingest_asset(rt, ASSET, source, end=date(2024, 5, 18))
    out = ingest_asset(rt, ASSET, source, start=date(2024, 1, 1), end=date(2024, 1, 31))
    assert out["status"] == "ok"
    assert out["new_bars"] == 31 * 288


def test_ingest_flow_records_a_pipeline_run(
    rt_and_source: tuple[Runtime, FakeSource, pd.DataFrame],
    monkeypatch: pytest.MonkeyPatch,
    prefect_server: None,
    tiny_config: Path,
) -> None:
    rt, source, _ = rt_and_source
    monkeypatch.setattr(ingest_mod, "_SOURCE_FACTORY", lambda name, root: source)
    kw = {
        "data_dir": str(rt.data_dir),
        "database_url": rt.settings.database_url,
        "tracking_uri": rt.settings.mlflow_tracking_uri,
        "config_path": str(tiny_config),
    }
    res = ingest_flow(end="2024-05-18", **kw)
    assert res["results"][0]["status"] == "ok"
    runs = rt.repo.pipeline_runs("ingest")
    assert runs[0]["status"] == "success"
    assert runs[0]["detail"]["results"][0]["asset"] == ASSET
    # a no-op run is recorded as skipped; the CLI maps success to exit code 0
    code = ingest_main(
        [
            "--end", "2024-05-18", "--data-dir", kw["data_dir"],
            "--database-url", kw["database_url"],
            "--tracking-uri", kw["tracking_uri"], "--config", kw["config_path"],
        ]
    )  # fmt: skip
    assert code == 0
    assert rt.repo.pipeline_runs("ingest")[0]["status"] == "skipped"


def test_failure_marks_the_run_failed_and_exits_nonzero(
    rt_and_source: tuple[Runtime, FakeSource, pd.DataFrame],
    monkeypatch: pytest.MonkeyPatch,
    prefect_server: None,
    tiny_config: Path,
) -> None:
    rt, _, _ = rt_and_source

    class Broken:
        def fetch_klines(self, *a: object, **k: object) -> pd.DataFrame:
            raise ConnectionError("archive unreachable")

    monkeypatch.setattr(ingest_mod, "_SOURCE_FACTORY", lambda name, root: Broken())
    monkeypatch.setattr(ingest_mod.ingest_asset_task, "retries", 0)
    code = ingest_main(
        [
            "--end", "2024-05-18", "--data-dir", str(rt.data_dir),
            "--database-url", rt.settings.database_url,
            "--tracking-uri", rt.settings.mlflow_tracking_uri, "--config", str(tiny_config),
        ]
    )  # fmt: skip
    assert code == 1
    run = rt.repo.pipeline_runs("ingest")[0]
    assert run["status"] == "failed"
    assert "archive unreachable" in run["detail"]["failures"][0]
