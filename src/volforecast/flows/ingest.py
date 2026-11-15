"""Ingest flow: pull the newest 5-minute bars into the parquet store and refresh the panel.

For each asset: find the last stored day, fetch from that day (re-fetched, it may have been
partial) through yesterday (UTC) from the source, merge with the stored rows of the touched
months (``write_klines`` rewrites whole monthly partitions), validate, write, refresh the
manifest, rebuild the daily panel and sync the ``realized`` table of the forecast store.

    uv run python -m volforecast.flows.ingest [--assets BTCUSDT] [--source binance|coinbase]
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from prefect import flow, task
from prefect.cache_policies import NO_CACHE

from volforecast.data.binance import BinanceVisionSource, CoinbaseSource, Source
from volforecast.data.schemas import validate_klines
from volforecast.data.store import klines_dir, read_klines, write_klines, write_manifest
from volforecast.flows.common import (
    Runtime,
    base_parser,
    common_kwargs,
    load_panel,
    run_cli,
    runtime_from_kwargs,
    sync_realized,
    tracked_run,
)
from volforecast.logging import get_logger

log = get_logger(__name__)

SourceFactory = Callable[[Path], Source]


def make_source(name: str, data_dir: Path) -> Source:
    if name == "coinbase":
        return CoinbaseSource()
    return BinanceVisionSource(cache_dir=data_dir / "cache")


def last_stored_day(data_dir: Path, asset: str, interval: str = "5m") -> date | None:
    """UTC day of the newest stored bar (reads only the newest monthly partition)."""
    parts = sorted(klines_dir(data_dir, asset, interval).glob("year=*/month=*/part.parquet"))
    if not parts:
        return None
    newest = pd.read_parquet(parts[-1], columns=["open_time"])["open_time"].max()
    return pd.Timestamp(newest).date()


def ingest_asset(
    rt: Runtime,
    asset: str,
    source: Source,
    *,
    end: date | None = None,
    start: date | None = None,
    interval: str = "5m",
) -> dict[str, Any]:
    """Fetch, merge, validate and store new bars of ``asset``; returns a summary dict."""
    data_dir = rt.data_dir
    end = end or (datetime.now(UTC).date() - timedelta(days=1))
    last = last_stored_day(data_dir, asset, interval)
    first = start or last
    if first is None:
        raise RuntimeError(
            f"no stored data for {asset} under {data_dir}: bootstrap once with "
            "`python -m volforecast.data ingest --symbol ... --start ...` or pass --start"
        )
    out: dict[str, Any] = {"asset": asset, "last_stored_day": str(last), "fetch_end": str(end)}
    if first > end:
        out.update(status="up_to_date", new_bars=0)
        return out
    fetched = source.fetch_klines(asset, first, end, interval)
    if fetched.empty:
        log.warning("ingest_no_new_data", asset=asset, start=str(first), end=str(end))
        out.update(status="no_new_data", new_bars=0)
        return out
    # merge with the stored rows of every touched month (whole-partition rewrite)
    month_start = fetched["open_time"].min().date().replace(day=1)
    try:
        stored = read_klines(data_dir, asset, start=month_start, interval=interval)
    except FileNotFoundError:
        stored = fetched.iloc[:0]
    n_before = len(stored)
    merged = (
        pd.concat([stored, fetched], ignore_index=True)
        .drop_duplicates("open_time", keep="last")
        .sort_values("open_time")
        .reset_index(drop=True)
    )
    if len(merged) == n_before:  # the re-fetched last day brought nothing new
        out.update(status="up_to_date", new_bars=0, fetched_bars=len(fetched))
        return out
    validate_klines(merged)
    write_klines(merged, data_dir, asset, interval)
    full = read_klines(data_dir, asset, interval=interval)
    manifest = write_manifest(
        klines_dir(data_dir, asset, interval) / "manifest.json",
        dataset=f"klines/{asset}/{interval}",
        df=full,
        source_checksums=getattr(source, "source_checksums", {}),
        extra={"source": type(source).__name__, "symbol": asset, "interval": interval},
    )
    panel, data_hash = load_panel(rt, asset)
    n_real = sync_realized(rt, asset, panel)
    ts = pd.Timestamp(full["open_time"].max())
    rt.repo.upsert_metrics(
        [
            {
                "asset": asset,
                "metric": "data_last_bar_ts",
                "as_of_date": ts.date(),
                "value": ts.timestamp(),
            }
        ]
    )
    out.update(
        status="ok",
        new_bars=int(len(merged) - n_before),
        fetched_bars=len(fetched),
        last_bar=ts.isoformat(),
        rows_total=int(manifest["rows"]),
        content_hash=manifest["content_hash"],
        panel_hash=data_hash,
        realized_rows_upserted=n_real,
    )
    return out


@task(name="ingest-asset", cache_policy=NO_CACHE, persist_result=False, retries=2,
      retry_delay_seconds=30)  # fmt: skip
def ingest_asset_task(
    rt: Runtime, asset: str, source_name: str, end: str | None, start: str | None
) -> dict[str, Any]:
    source = _SOURCE_FACTORY(source_name, rt.data_dir)
    return ingest_asset(
        rt,
        asset,
        source,
        end=None if end is None else date.fromisoformat(end),
        start=None if start is None else date.fromisoformat(start),
    )


#: indirection so tests can swap the network source
_SOURCE_FACTORY: Callable[[str, Path], Source] = make_source


def set_source_factory(factory: Callable[[str, Path], Source]) -> None:
    """Replace the source constructor (tests inject a fake archive)."""
    global _SOURCE_FACTORY
    _SOURCE_FACTORY = factory


@flow(name="ingest", persist_result=False)
def ingest_flow(
    assets: list[str] | None = None,
    source: str = "binance",
    end: str | None = None,
    start: str | None = None,
    data_dir: str | None = None,
    database_url: str | None = None,
    tracking_uri: str | None = None,
    config_path: str | None = None,
    horizons: list[int] | None = None,
) -> dict[str, Any]:
    rt = runtime_from_kwargs(
        dict(
            assets=assets,
            horizons=horizons,
            data_dir=data_dir,
            database_url=database_url,
            tracking_uri=tracking_uri,
            config_path=config_path,
        )
    )
    with tracked_run(rt, "ingest", end, source=source) as rec:
        results: list[dict[str, Any]] = []
        failures: list[str] = []
        for asset in rt.cfg.assets:
            try:
                results.append(ingest_asset_task(rt, asset, source, end, start))
            except Exception as exc:
                log.error("ingest_asset_failed", asset=asset, err=repr(exc))
                failures.append(f"{asset}: {exc!r}")
        rec.detail.update(results=results, failures=failures)
        if failures:
            raise RuntimeError("ingest failed for " + "; ".join(failures))
        if all(r["status"] in ("up_to_date", "no_new_data") for r in results):
            rec.status = "skipped"
    return {"results": results}


def main(argv: Sequence[str] | None = None) -> int:
    p = base_parser("ingest", __doc__.splitlines()[0] if __doc__ else "")
    p.add_argument("--source", choices=("binance", "coinbase"), default="binance")
    p.add_argument("--end", default=None, help="last UTC day to fetch (default: yesterday)")
    p.add_argument("--start", default=None, help="first day (default: last stored day)")
    args = p.parse_args(argv)
    kw = common_kwargs(args)
    return run_cli(
        "ingest",
        lambda: ingest_flow(source=args.source, end=args.end, start=args.start, **kw),
    )


if __name__ == "__main__":
    sys.exit(main())
