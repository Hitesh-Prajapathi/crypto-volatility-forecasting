"""Command line interface: ``python -m volforecast {backtest,report}``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from volforecast.config import get_settings
from volforecast.logging import configure_logging, get_logger

log = get_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="volforecast", description="Volatility forecasting CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    bt = sub.add_parser("backtest", help="run the walk-forward backtest")
    bt.add_argument("--config", type=Path, default=Path("configs/backtest.yaml"))
    bt.add_argument("--assets", nargs="+", default=None, help="subset of the configured assets")
    bt.add_argument("--out", type=Path, default=Path("artifacts/backtest"))
    bt.add_argument(
        "--data-root", type=Path, default=None, help="default: $VOLF_DATA_DIR or ./data"
    )
    bt.add_argument("--n-jobs", type=int, default=None, help="override the config's n_jobs")

    rp = sub.add_parser("report", help="build tables and figures from a backtest run")
    rp.add_argument("--run", type=Path, default=Path("artifacts/backtest"))
    rp.add_argument("--out", type=Path, default=Path("reports"))
    rp.add_argument(
        "--benchmark2",
        default="log_har_dow",
        help="secondary benchmark for extra ratio/DM columns (omitted if absent; 'none' disables)",
    )
    rp.add_argument(
        "--inject-readme",
        type=Path,
        default=None,
        help="also replace the RESULTS block of this README with the headline tables",
    )
    return parser


def _cmd_backtest(args: argparse.Namespace) -> int:
    from volforecast.pipeline.backtest import run_backtest
    from volforecast.pipeline.config import load_config

    cfg = load_config(args.config)
    if args.n_jobs is not None:
        cfg = cfg.model_copy(update={"n_jobs": args.n_jobs})
    if args.assets:
        unknown = sorted(set(args.assets) - set(cfg.assets))
        if unknown:
            print(f"error: assets not in config: {unknown}", file=sys.stderr)
            return 2
    data_root = args.data_root or get_settings().data_dir
    forecasts = run_backtest(cfg, data_root, args.out, assets=args.assets)
    meta = json.loads((args.out / "run_meta.json").read_text())
    print(
        f"backtest done: {len(forecasts)} forecast rows, {meta['n_failures']} model failures, "
        f"{meta['timings_seconds']['total']:.1f}s -> {args.out}"
    )
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    from volforecast.evaluation.report import build_report, inject_readme
    from volforecast.pipeline.config import BacktestConfig

    meta = json.loads((args.run / "run_meta.json").read_text())
    cfg = BacktestConfig.model_validate(meta["config"])
    forecasts = pd.read_parquet(args.run / "forecasts.parquet")
    assets = [a for a in meta.get("assets_run", cfg.assets) if a in set(forecasts["asset"])]
    panels = {a: pd.read_parquet(args.run / "panels" / f"{a}.parquet") for a in assets}
    cfg = cfg.model_copy(update={"assets": assets})
    run_info = {"git": meta.get("git"), "data_hashes": meta.get("data_hashes")}
    bm2 = None if args.benchmark2.lower() == "none" else args.benchmark2
    build_report(forecasts, panels, cfg, args.out, run_meta=run_info, benchmark2=bm2)
    if args.inject_readme is not None:
        inject_readme(args.inject_readme, (args.out / "tables_readme.md").read_text())
    print(f"report written to {args.out} (tables.md, results.json, figures/)")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    configure_logging(get_settings().log_level)
    if args.command == "backtest":
        return _cmd_backtest(args)
    return _cmd_report(args)


if __name__ == "__main__":
    raise SystemExit(main())
