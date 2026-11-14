from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from tests.unit.pipeline.helpers import make_cfg
from volforecast.cli import main
from volforecast.contracts import FORECAST_COLUMNS
from volforecast.models import factory
from volforecast.pipeline import backtest as bt


@pytest.fixture
def patched_panels(monkeypatch: pytest.MonkeyPatch, synthetic_panel: pd.DataFrame) -> None:
    def fake_build(root: Path, symbol: str) -> tuple[pd.DataFrame, str]:
        return synthetic_panel, f"hash-{symbol}"

    monkeypatch.setattr(bt, "build_panel", fake_build)


def test_run_backtest_end_to_end(patched_panels: None, tmp_path: Path) -> None:
    cfg = make_cfg(assets=("AAA", "BBB"), models=("har", "naive", "log_har"))
    out = tmp_path / "run"
    fc = bt.run_backtest(cfg, tmp_path, out)

    disk = pd.read_parquet(out / "forecasts.parquet")
    assert tuple(disk.columns) == FORECAST_COLUMNS
    assert len(disk) == len(fc)
    assert set(disk["asset"]) == {"AAA", "BBB"}
    assert set(disk["model"]) == {"har", "naive", "log_har"}
    assert set(disk["horizon"]) == {1, 5}
    assert set(disk[disk["asset"] == "AAA"]["data_hash"]) == {"hash-AAA"}
    # one row per (asset, model, horizon, origin)
    assert not disk.duplicated(["asset", "model", "horizon", "origin_date"]).any()
    assert (disk["origin_date"] >= pd.Timestamp("2022-01-01")).all()
    assert disk["forecast_var"].notna().all()

    meta = json.loads((out / "run_meta.json").read_text())
    assert meta["config"]["benchmark"] == "har"
    assert meta["data_hashes"] == {"AAA": "hash-AAA", "BBB": "hash-BBB"}
    assert meta["n_failures"] == 0
    assert meta["n_forecast_rows"] == len(disk)
    assert "sha" in meta["git"]
    assert set(meta["run_stats"]["AAA"]) == {f"{m}|h{h}" for m in cfg.model_names for h in (1, 5)}
    st = meta["run_stats"]["AAA"]["har|h1"]
    assert st["n_fits"] == st["n_origins"]
    assert st["failures"] == []
    assert meta["timings_seconds"]["total"] > 0
    assert (out / "panels" / "AAA.parquet").exists()


def test_assets_subset_and_refit_every_passed_through(patched_panels: None, tmp_path: Path) -> None:
    cfg = make_cfg(assets=("AAA", "BBB"), models=("har",), horizons=(1,))
    cfg.models[0].refit_every = 7
    out = tmp_path / "run"
    bt.run_backtest(cfg, tmp_path, out, assets=["BBB"])
    meta = json.loads((out / "run_meta.json").read_text())
    assert meta["assets_run"] == ["BBB"]
    st = meta["run_stats"]["BBB"]["har|h1"]
    assert st["n_fits"] == -(-st["n_origins"] // 7)


def test_failures_are_recorded(
    patched_panels: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Boom:
        name = "boom"

        def fit(self, panel: pd.DataFrame, h: int) -> Boom:
            raise RuntimeError("no fit")

        def predict_variance(self, panel: pd.DataFrame, h: int) -> float:
            return 1.0

        def get_params(self) -> dict[str, object]:
            return {}

    monkeypatch.setitem(factory.MODEL_FACTORY, "naive", lambda **_: Boom())
    cfg = make_cfg(models=("har", "naive"), horizons=(1,))
    out = tmp_path / "run"
    fc = bt.run_backtest(cfg, tmp_path, out)
    meta = json.loads((out / "run_meta.json").read_text())
    assert meta["n_failures"] > 0
    rec = meta["run_stats"]["SYN"]["naive|h1"]
    assert rec["n_fit_failures"] > 0
    assert rec["failures"]
    assert fc[fc["model"] == "naive"]["forecast_var"].isna().all()


def test_cli_backtest_then_report(
    patched_panels: None,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # structlog caches loggers bound to the (pytest-captured) stderr: keep global config intact
    monkeypatch.setattr("volforecast.cli.configure_logging", lambda *a, **k: None)
    cfg = make_cfg(assets=("AAA", "BBB"), models=("har", "naive", "log_har"))
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg.model_dump(mode="json")))
    run, rep = tmp_path / "run", tmp_path / "rep"

    assert (
        main(
            [
                "backtest",
                "--config",
                str(cfg_path),
                "--assets",
                "AAA",
                "--out",
                str(run),
                "--data-root",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert "backtest done" in capsys.readouterr().out
    assert main(["backtest", "--config", str(cfg_path), "--assets", "ZZZ", "--out", str(run)]) == 2

    assert main(["report", "--run", str(run), "--out", str(rep)]) == 0
    assert (rep / "results.json").exists()
    assert (rep / "tables.md").exists()
    results = json.loads((rep / "results.json").read_text())
    assert list(results["assets"]) == ["AAA"]
