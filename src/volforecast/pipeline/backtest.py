"""Config-driven walk-forward backtest over assets (plan M2 P2.3).

Outputs, under ``out_dir``:

* ``forecasts.parquet`` - long table with ``contracts.FORECAST_COLUMNS`` (all assets);
* ``panels/<ASSET>.parquet`` - the daily panels the forecasts were produced from (the report
  needs ``ret`` and ``rv`` for VaR and regime tables);
* ``run_meta.json`` - config, git SHA, data hashes, per-(asset, model, h) run stats incl.
  failures, and timings.
"""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd

from volforecast.contracts import FORECAST_COLUMNS
from volforecast.evaluation.walkforward import ModelSpec, WalkForwardConfig, run_many
from volforecast.logging import get_logger
from volforecast.models.factory import build_model
from volforecast.pipeline.config import BacktestConfig
from volforecast.pipeline.panel import build_panel

log = get_logger(__name__)

MAX_RECORDED_FAILURES = 20

PanelBuilder = Callable[[Path, str], tuple[pd.DataFrame, str]]


def git_info(cwd: Path | None = None) -> dict[str, Any]:
    """Current git SHA and dirty flag (``unknown`` outside a repository)."""

    def _run(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()

    try:
        return {"sha": _run("rev-parse", "HEAD"), "dirty": bool(_run("status", "--porcelain"))}
    except (OSError, subprocess.SubprocessError):
        return {"sha": "unknown", "dirty": None}


def model_specs(cfg: BacktestConfig) -> list[ModelSpec]:
    """Translate the config's model list into engine ``ModelSpec`` objects."""
    return [
        ModelSpec(
            name=m.name,
            builder=partial(build_model, m.name, **m.params),
            refit_every=m.refit_every,
            version=m.version,
            reuse_instance=m.reuse_instance,
        )
        for m in cfg.models
    ]


def walkforward_config(cfg: BacktestConfig) -> WalkForwardConfig:
    return WalkForwardConfig(
        first_origin=pd.Timestamp(cfg.first_origin),
        last_origin=None if cfg.last_origin is None else pd.Timestamp(cfg.last_origin),
        horizons=tuple(cfg.horizons),
        window=cfg.window,
        rolling_size=cfg.rolling_size,
        min_train=cfg.min_train,
    )


def run_backtest(
    cfg: BacktestConfig,
    data_root: str | Path,
    out_dir: str | Path,
    assets: Sequence[str] | None = None,
    panel_builder: PanelBuilder | None = None,
) -> pd.DataFrame:
    """Run the walk-forward backtest for every asset and write the run artifacts.

    ``assets`` optionally restricts the configured assets (a subset of ``cfg.assets``).
    ``panel_builder`` maps ``(data_root, symbol)`` to ``(panel, content_hash)``; the default builds
    (or cache-reads) from raw klines, ``panel.load_cached_panel`` reads the daily cache only.
    Returns the combined forecasts table.
    """
    root, out = Path(data_root), Path(out_dir)
    build = build_panel if panel_builder is None else panel_builder  # late-bound (patchable)
    chosen = list(cfg.assets) if assets is None else list(assets)
    if not chosen:
        raise ValueError("no assets selected")
    out.mkdir(parents=True, exist_ok=True)
    (out / "panels").mkdir(exist_ok=True)

    specs = model_specs(cfg)
    wf_cfg = walkforward_config(cfg)
    t_start = time.perf_counter()
    frames: list[pd.DataFrame] = []
    data_hashes: dict[str, str] = {}
    run_stats: dict[str, dict[str, dict[str, Any]]] = {}
    timings: dict[str, float] = {}

    for asset in chosen:
        t0 = time.perf_counter()
        panel, data_hash = build(root, asset)
        panel.to_parquet(out / "panels" / f"{asset}.parquet")
        t_panel = time.perf_counter() - t0
        t1 = time.perf_counter()
        fc, stats = run_many(
            panel, specs, wf_cfg, asset=asset, data_hash=data_hash, n_jobs=cfg.n_jobs
        )
        frames.append(fc)
        data_hashes[asset] = data_hash
        run_stats[asset] = {}
        for (model, h), st in stats.items():
            rec = asdict(st)
            rec["n_failures_total"] = len(st.failures)
            rec["failures"] = st.failures[:MAX_RECORDED_FAILURES]
            run_stats[asset][f"{model}|h{h}"] = rec
        timings[asset] = round(time.perf_counter() - t0, 3)
        timings[f"{asset}:panel"] = round(t_panel, 3)
        timings[f"{asset}:walkforward"] = round(time.perf_counter() - t1, 3)
        log.info("asset_done", asset=asset, rows=len(fc), seconds=timings[asset])

    forecasts = pd.concat(frames, ignore_index=True)[list(FORECAST_COLUMNS)]
    forecasts.to_parquet(out / "forecasts.parquet", index=False)

    n_failures = sum(
        r["n_fit_failures"] + r["n_predict_failures"]
        for a in run_stats.values()
        for r in a.values()
    )
    meta = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": git_info(Path(__file__).resolve().parent),
        "config": cfg.model_dump(mode="json"),
        "assets_run": chosen,
        "data_root": str(root),
        "data_hashes": data_hashes,
        "n_forecast_rows": len(forecasts),
        "n_failures": n_failures,
        "run_stats": run_stats,
        "timings_seconds": {**timings, "total": round(time.perf_counter() - t_start, 3)},
    }
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    log.info("backtest_done", rows=len(forecasts), failures=n_failures, out=str(out))
    return forecasts
