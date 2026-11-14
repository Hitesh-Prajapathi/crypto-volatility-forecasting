"""CLI: ``python -m volforecast.data ingest|quality``.

Examples::

    uv run python -m volforecast.data ingest --symbol BTCUSDT --start 2017-08-17
    uv run python -m volforecast.data quality --symbol BTCUSDT
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta

from volforecast.config import get_settings
from volforecast.data.binance import BinanceVisionSource, CoinbaseSource, Source
from volforecast.data.schemas import kline_quality_report, validate_klines
from volforecast.data.store import klines_dir, read_klines, write_klines, write_manifest
from volforecast.logging import configure_logging, get_logger

log = get_logger(__name__)

#: A day with fewer bars than this is reported as incomplete (full day = 288).
INCOMPLETE_BARS = 280


def _parse_date(text: str) -> date:
    return date.fromisoformat(text)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m volforecast.data", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="download klines and write hive-partitioned Parquet")
    ingest.add_argument("--symbol", required=True)
    ingest.add_argument("--start", required=True, type=_parse_date, help="YYYY-MM-DD (UTC)")
    ingest.add_argument(
        "--end", type=_parse_date, default=None, help="YYYY-MM-DD inclusive (default: yesterday)"
    )
    ingest.add_argument("--source", choices=("binance", "coinbase"), default="binance")
    ingest.add_argument("--interval", default=None, help="default: settings.interval (5m)")

    quality = sub.add_parser("quality", help="print the incomplete-days report")
    quality.add_argument("--symbol", required=True)
    quality.add_argument("--interval", default=None)
    return parser


def _incomplete_days(symbol: str, interval: str) -> tuple[int, str]:
    """Return (count, printable table) of days with fewer than INCOMPLETE_BARS bars."""
    settings = get_settings()
    report = kline_quality_report(read_klines(settings.data_dir, symbol, interval=interval))
    bad = report[report["n_bars"] < INCOMPLETE_BARS]
    return len(bad), bad.to_string() if len(bad) else "(none)"


def run_ingest(
    symbol: str,
    start: date,
    end: date | None,
    source_name: str = "binance",
    interval: str | None = None,
) -> int:
    """Fetch, validate and store klines. Returns the number of incomplete days."""
    settings = get_settings()
    interval = interval or settings.interval
    end = end or (datetime.now(UTC).date() - timedelta(days=1))
    root = settings.data_dir
    source: Source
    if source_name == "coinbase":
        source = CoinbaseSource()
    else:
        source = BinanceVisionSource(cache_dir=root / "cache")

    log.info("ingest_start", symbol=symbol, start=str(start), end=str(end), source=source_name)
    df = source.fetch_klines(symbol, start, end, interval)
    if df.empty:
        raise SystemExit(f"no data returned for {symbol} {start}..{end} from {source_name}")
    validate_klines(df)
    write_klines(df, root, symbol, interval)

    checksums: dict[str, str] = getattr(source, "source_checksums", {})
    manifest = write_manifest(
        klines_dir(root, symbol, interval) / "manifest.json",
        dataset=f"klines/{symbol}/{interval}",
        df=df,
        source_checksums=checksums,
        extra={
            "source": source_name,
            "symbol": symbol,
            "interval": interval,
            "requested_start": start.isoformat(),
            "requested_end": end.isoformat(),
            "snapped_bars": int(getattr(source, "snapped_bars", 0)),
            "dropped_bars": int(getattr(source, "dropped_bars", 0)),
        },
    )
    report = kline_quality_report(df)
    n_incomplete = int((report["n_bars"] < INCOMPLETE_BARS).sum())
    log.info(
        "ingest_done",
        symbol=symbol,
        rows=len(df),
        first=manifest["min_time"],
        last=manifest["max_time"],
        days=len(report),
        days_under_280_bars=n_incomplete,
        snapped_bars=getattr(source, "snapped_bars", 0),
        content_hash=manifest["content_hash"],
    )
    return n_incomplete


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = _build_parser().parse_args(argv)
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    interval = args.interval or settings.interval
    if args.command == "ingest":
        run_ingest(args.symbol, args.start, args.end, args.source, interval)
    else:
        count, table = _incomplete_days(args.symbol, interval)
        sys.stdout.write(
            f"{args.symbol} {interval}: {count} day(s) with < {INCOMPLETE_BARS} bars\n{table}\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
