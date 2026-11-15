"""Shared plumbing of the four flows: runtime wiring, run tracking, panel loading, CLI.

Every flow module exposes (a) plain-Python functions that take a :class:`Runtime` (unit-testable
with no Prefect server), (b) thin ``@task`` wrappers around them and (c) a ``@flow`` that
orchestrates the tasks. ``python -m volforecast.flows.<name>`` runs the flow and exits non-zero
when it raises (Kubernetes retries the Job, ADR-0007).

The runtime is configured from :class:`volforecast.config.Settings` (``VOLF_*`` environment
variables, see docs/platform.md); every setting can be overridden per call or with a CLI flag.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from functools import cached_property
from pathlib import Path
from typing import Any

import pandas as pd
from mlflow import MlflowClient

from volforecast.config import Settings, get_settings
from volforecast.contracts import COL_COMPLETE
from volforecast.forecast_store import ForecastRepository, clean_json
from volforecast.logging import configure_logging, get_logger
from volforecast.monitoring.prom import last_bar_timestamp, push_job_metrics
from volforecast.pipeline.panel import build_panel
from volforecast.registry.client import configure_mlflow
from volforecast.registry.config import ProductionConfig, load_production_config

log = get_logger(__name__)


@dataclass
class Runtime:
    """Everything a flow needs: settings, production config, forecast store, MLflow client."""

    settings: Settings
    cfg: ProductionConfig
    repo: ForecastRepository
    #: optional replacement of the parquet-store panel loader (tests / alternative stores)
    panel_loader: Callable[[str], tuple[pd.DataFrame, str]] | None = None

    @classmethod
    def create(
        cls,
        *,
        data_dir: str | Path | None = None,
        database_url: str | None = None,
        tracking_uri: str | None = None,
        config_path: str | Path | None = None,
        assets: Sequence[str] | None = None,
        horizons: Sequence[int] | None = None,
    ) -> Runtime:
        """Build a runtime from the environment, overriding what is passed explicitly."""
        updates: dict[str, Any] = {}
        if data_dir is not None:
            updates["data_dir"] = Path(data_dir)
        if database_url:
            updates["database_url"] = database_url
        if tracking_uri:
            updates["mlflow_tracking_uri"] = tracking_uri
        if config_path is not None:
            updates["production_config"] = Path(config_path)
        settings = get_settings().model_copy(update=updates)
        cfg = load_production_config(settings.production_config)
        overrides: dict[str, Any] = {}
        if assets:
            overrides["assets"] = list(assets)
        if horizons:
            overrides["horizons"] = [int(h) for h in horizons]
        if overrides:
            cfg = cfg.model_copy(update=overrides)
        return cls(settings=settings, cfg=cfg, repo=ForecastRepository(settings.database_url))

    @cached_property
    def client(self) -> MlflowClient:
        """MLflow client (also selects the tracking/registry URI and experiment)."""
        return configure_mlflow(
            self.settings.mlflow_tracking_uri,
            self.settings.mlflow_experiment,
            self.settings.mlflow_artifact_root,
        )

    @property
    def data_dir(self) -> Path:
        return self.settings.data_dir


# ---------------------------------------------------------------------------------------
# data access
# ---------------------------------------------------------------------------------------
def load_panel(rt: Runtime, asset: str) -> tuple[pd.DataFrame, str]:
    """Daily realized panel of ``asset`` from the parquet store and its content hash."""
    if rt.panel_loader is not None:
        return rt.panel_loader(asset)
    return build_panel(rt.data_dir, asset)


def last_complete_day(panel: pd.DataFrame, as_of: date | str | None = None) -> pd.Timestamp:
    """Latest complete day of ``panel`` (at or before ``as_of`` if given)."""
    ok = panel.index[panel[COL_COMPLETE].astype(bool).to_numpy()]
    if as_of is not None:
        ok = ok[ok <= pd.Timestamp(as_of)]
    if len(ok) == 0:
        raise ValueError("no complete day in the panel" + (f" up to {as_of}" if as_of else ""))
    return pd.Timestamp(ok[-1])


def sync_realized(rt: Runtime, asset: str, panel: pd.DataFrame, tail_days: int = 60) -> int:
    """Upsert the panel's realized measures into the store (full history on the first call,
    afterwards only the last ``tail_days`` rows) and record the newest-bar metric."""
    last = rt.repo.last_realized_date(asset)
    part = panel if last is None else panel.loc[pd.Timestamp(last) - pd.Timedelta(days=tail_days) :]
    n = rt.repo.upsert_realized(asset, part)
    bars = panel.index[(panel["n_bars"] > 0).to_numpy()]
    if len(bars):
        ts = last_bar_timestamp(bars[-1])
        existing = rt.repo.get_metrics("data_last_bar_ts", asset=asset)
        if existing.empty or ts > float(existing["value"].max()):
            rt.repo.upsert_metrics(
                [
                    {
                        "asset": asset,
                        "metric": "data_last_bar_ts",
                        "as_of_date": pd.Timestamp(bars[-1]).date(),
                        "value": ts,
                    }
                ]
            )
    return n


# ---------------------------------------------------------------------------------------
# run tracking
# ---------------------------------------------------------------------------------------
class RunRecorder:
    """Handle yielded by :func:`tracked_run`; ``detail`` is stored in ``pipeline_runs``."""

    def __init__(self) -> None:
        self.detail: dict[str, Any] = {}
        self.status = "success"


@contextmanager
def tracked_run(
    rt: Runtime, flow: str, run_date: date | str | None = None, **detail: Any
) -> Iterator[RunRecorder]:
    """Record a ``pipeline_runs`` row: ``running`` -> ``success`` (or ``failed`` + re-raise).

    Set ``rec.status = "skipped"`` for a no-op run. Pipeline metrics are pushed/exported at the
    end of a successful run.
    """
    run_id = rt.repo.start_run(flow, run_date, detail)
    rec = RunRecorder()
    try:
        yield rec
    except BaseException as exc:
        rt.repo.finish_run(run_id, "failed", {**rec.detail, "error": repr(exc)})
        raise
    rt.repo.finish_run(run_id, rec.status, clean_json(rec.detail))
    if rec.status == "success":
        push_job_metrics(flow, rt.repo)


# ---------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------
def base_parser(flow: str, description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=f"python -m volforecast.flows.{flow}", description=description)
    p.add_argument("--assets", nargs="+", default=None, help="default: from production.yaml")
    p.add_argument("--horizons", nargs="+", type=int, default=None)
    p.add_argument("--data-dir", type=Path, default=None, help="default: $VOLF_DATA_DIR")
    p.add_argument("--database-url", default=None, help="default: $VOLF_DATABASE_URL")
    p.add_argument("--tracking-uri", default=None, help="default: $MLFLOW_TRACKING_URI")
    p.add_argument("--config", type=Path, default=None, help="default: configs/production.yaml")
    return p


def common_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Flow keyword arguments shared by every flow (the runtime overrides)."""
    return {
        "assets": args.assets,
        "horizons": args.horizons,
        "data_dir": None if args.data_dir is None else str(args.data_dir),
        "database_url": args.database_url,
        "tracking_uri": args.tracking_uri,
        "config_path": None if args.config is None else str(args.config),
    }


def runtime_from_kwargs(kw: dict[str, Any]) -> Runtime:
    return Runtime.create(
        data_dir=kw.get("data_dir"),
        database_url=kw.get("database_url"),
        tracking_uri=kw.get("tracking_uri"),
        config_path=kw.get("config_path"),
        assets=kw.get("assets"),
        horizons=kw.get("horizons"),
    )


def run_cli(flow_name: str, runner: Callable[[], Any]) -> int:
    """Configure logging, run, and map exceptions to a non-zero exit code."""
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    try:
        result = runner()
    except Exception as exc:
        log.error("flow_failed", flow=flow_name, err=repr(exc))
        print(f"{flow_name} failed: {exc!r}", file=sys.stderr)
        return 1
    log.info("flow_done", flow=flow_name)
    print(f"{flow_name} ok: {clean_json(result)}")
    return 0
