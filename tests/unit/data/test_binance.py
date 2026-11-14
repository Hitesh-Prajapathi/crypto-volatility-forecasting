"""BinanceVisionSource / CoinbaseSource with httpx.MockTransport (no network)."""

from __future__ import annotations

import hashlib
import io
import zipfile
from datetime import date
from pathlib import Path

import httpx
import pandas as pd
import pytest

from volforecast.contracts import KLINE_COLUMNS
from volforecast.data.binance import (
    BinanceVisionSource,
    ChecksumError,
    CoinbaseSource,
    DownloadError,
    coinbase_product,
    epoch_to_datetime,
    parse_binance_csv,
    parse_checksum_file,
)

BASE = "https://data.binance.vision"
HEADER = (
    "open_time,open,high,low,close,volume,close_time,quote_volume,count,"
    "taker_buy_base,taker_buy_quote,ignore\n"
)


def csv_rows(start: str, n: int, *, unit: str = "ms", price: float = 100.0) -> str:
    """Binance-format CSV rows (no header) for ``n`` consecutive 5m bars."""
    t0 = pd.Timestamp(start, tz="UTC")
    mult = {"ms": 10**3, "us": 10**6}[unit]
    lines = []
    for i in range(n):
        open_s = int((t0 + pd.Timedelta(minutes=5 * i)).timestamp()) * mult
        close_s = open_s + 5 * 60 * mult - 1
        p = price + i
        lines.append(
            f"{open_s},{p:.8f},{p + 1:.8f},{p - 1:.8f},{p + 0.5:.8f},10.0,{close_s},"
            f"{(p + 0.5) * 10:.8f},{100 + i},5.0,{p * 5:.8f},0"
        )
    return "\n".join(lines) + "\n"


def make_zip(csv: str, name: str = "x.csv") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(name, csv)
    return buf.getvalue()


class FakeArchive:
    """URL -> (status, body) routing table that records every request."""

    def __init__(self) -> None:
        self.routes: dict[str, tuple[int, bytes]] = {}
        self.requests: list[httpx.Request] = []

    def add(self, url: str, zip_bytes: bytes, *, checksum: str | None = None) -> None:
        digest = checksum or hashlib.sha256(zip_bytes).hexdigest()
        name = url.rsplit("/", 1)[1]
        self.routes[url] = (200, zip_bytes)
        self.routes[url + ".CHECKSUM"] = (200, f"{digest}  {name}\n".encode())

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        status, body = self.routes.get(str(request.url), (404, b""))
        return httpx.Response(status, content=body)

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handler))

    def urls(self) -> list[str]:
        return [str(r.url) for r in self.requests]


def monthly_url(symbol: str, stamp: str, interval: str = "5m") -> str:
    return f"{BASE}/data/spot/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{stamp}.zip"


def daily_url(symbol: str, stamp: str, interval: str = "5m") -> str:
    return f"{BASE}/data/spot/daily/klines/{symbol}/{interval}/{symbol}-{interval}-{stamp}.zip"


def make_source(archive: FakeArchive, tmp_path: Path) -> BinanceVisionSource:
    return BinanceVisionSource(
        tmp_path / "cache", client=archive.client(), sleep=lambda _s: None, backoff_seconds=0.0
    )


# ── parsing ──────────────────────────────────────────────────────────────────────────
def test_epoch_ms_and_us_parse_to_same_instant() -> None:
    ms = pd.Series([1_740_787_200_000, 1_740_787_500_000])
    us = pd.Series([1_740_787_200_000_000, 1_740_787_500_000_000])
    a, b = epoch_to_datetime(ms), epoch_to_datetime(us)
    assert a.equals(b)
    assert str(a.dtype) == "datetime64[ns, UTC]"
    assert a.iloc[0] == pd.Timestamp("2025-03-01 00:00", tz="UTC")


def test_microsecond_file_has_no_1970_or_far_future_dates() -> None:
    df = parse_binance_csv(csv_rows("2025-03-01", 12, unit="us"))
    assert df["open_time"].min() == pd.Timestamp("2025-03-01", tz="UTC")
    assert df["open_time"].max() == pd.Timestamp("2025-03-01 00:55", tz="UTC")
    assert df["open_time"].dt.year.unique().tolist() == [2025]


def test_millisecond_file_parses() -> None:
    df = parse_binance_csv(csv_rows("2024-02-01", 12, unit="ms"))
    assert df["open_time"].min() == pd.Timestamp("2024-02-01", tz="UTC")
    assert df["open_time"].dt.year.unique().tolist() == [2024]


def test_parse_matches_contract_dtypes() -> None:
    df = parse_binance_csv(csv_rows("2024-02-01", 5))
    assert tuple(df.columns) == KLINE_COLUMNS
    assert str(df["open_time"].dtype) == "datetime64[ns, UTC]"
    assert str(df["n_trades"].dtype) == "int64"
    assert all(str(df[c].dtype) == "float64" for c in ("open", "high", "low", "close", "volume"))
    assert isinstance(df.index, pd.RangeIndex)
    assert df.loc[0, "n_trades"] == 100
    assert df.loc[0, "quote_volume"] == pytest.approx(1005.0)


def test_header_and_headerless_csv_equal() -> None:
    body = csv_rows("2024-02-01", 6)
    pd.testing.assert_frame_equal(parse_binance_csv(body), parse_binance_csv(HEADER + body))


def test_parse_sorts_and_dedupes() -> None:
    body = csv_rows("2024-02-01", 4)
    lines = body.strip().split("\n")
    shuffled = "\n".join([lines[2], lines[0], lines[0], lines[1], lines[3]]) + "\n"
    df = parse_binance_csv(shuffled)
    assert len(df) == 4
    assert df["open_time"].is_monotonic_increasing


def test_parse_checksum_file() -> None:
    digest = "a" * 64
    assert parse_checksum_file(f"{digest}  file.zip\n") == digest
    with pytest.raises(ChecksumError):
        parse_checksum_file("not a checksum")


# ── URL selection ────────────────────────────────────────────────────────────────────
def test_full_months_use_monthly_archives(tmp_path: Path) -> None:
    arch = FakeArchive()
    arch.add(monthly_url("BTCUSDT", "2024-01"), make_zip(csv_rows("2024-01-01", 288 * 31)))
    arch.add(monthly_url("BTCUSDT", "2024-02"), make_zip(csv_rows("2024-02-01", 288 * 29)))
    df = make_source(arch, tmp_path).fetch_klines("BTCUSDT", date(2024, 1, 1), date(2024, 2, 29))
    assert len(df) == 288 * 60
    assert not any("/daily/" in u for u in arch.urls())
    assert monthly_url("BTCUSDT", "2024-01") in arch.urls()


def test_partial_last_month_uses_daily_archives(tmp_path: Path) -> None:
    arch = FakeArchive()
    arch.add(monthly_url("BTCUSDT", "2024-01"), make_zip(csv_rows("2024-01-01", 288 * 31)))
    for d in ("2024-02-01", "2024-02-02", "2024-02-03"):
        arch.add(daily_url("BTCUSDT", d), make_zip(csv_rows(d, 288)))
    # A monthly archive for the partial month must never be requested (it would be a bad cut).
    df = make_source(arch, tmp_path).fetch_klines("BTCUSDT", date(2024, 1, 1), date(2024, 2, 3))
    assert len(df) == 288 * 34
    assert monthly_url("BTCUSDT", "2024-02") not in arch.urls()
    assert daily_url("BTCUSDT", "2024-02-03") in arch.urls()
    assert df["open_time"].max() == pd.Timestamp("2024-02-03 23:55", tz="UTC")


def test_unpublished_monthly_falls_back_to_daily(tmp_path: Path) -> None:
    arch = FakeArchive()  # monthly 2024-02 -> 404
    for i in range(1, 30):
        d = f"2024-02-{i:02d}"
        arch.add(daily_url("BTCUSDT", d), make_zip(csv_rows(d, 288)))
    df = make_source(arch, tmp_path).fetch_klines("BTCUSDT", date(2024, 2, 1), date(2024, 2, 29))
    assert len(df) == 288 * 29
    assert monthly_url("BTCUSDT", "2024-02") + ".CHECKSUM" in arch.urls()


def test_range_is_clipped_to_start_and_end(tmp_path: Path) -> None:
    arch = FakeArchive()
    arch.add(monthly_url("BTCUSDT", "2024-02"), make_zip(csv_rows("2024-02-01", 288 * 29)))
    df = make_source(arch, tmp_path).fetch_klines("BTCUSDT", date(2024, 2, 10), date(2024, 2, 29))
    assert df["open_time"].min() == pd.Timestamp("2024-02-10", tz="UTC")
    assert len(df) == 288 * 20


def test_year_boundary_months(tmp_path: Path) -> None:
    arch = FakeArchive()
    arch.add(monthly_url("BTCUSDT", "2024-12"), make_zip(csv_rows("2024-12-01", 288 * 31)))
    arch.add(
        monthly_url("BTCUSDT", "2025-01"), make_zip(csv_rows("2025-01-01", 288 * 31, unit="us"))
    )
    df = make_source(arch, tmp_path).fetch_klines("BTCUSDT", date(2024, 12, 1), date(2025, 1, 31))
    assert len(df) == 288 * 62
    assert df["open_time"].is_monotonic_increasing
    assert df["open_time"].is_unique
    assert df["open_time"].max() == pd.Timestamp("2025-01-31 23:55", tz="UTC")


def test_404_everywhere_returns_empty_contract_frame(tmp_path: Path) -> None:
    arch = FakeArchive()
    df = make_source(arch, tmp_path).fetch_klines("BTCUSDT", date(2024, 2, 1), date(2024, 2, 2))
    assert df.empty
    assert tuple(df.columns) == KLINE_COLUMNS
    assert str(df["open_time"].dtype) == "datetime64[ns, UTC]"


def test_missing_daily_is_skipped_gracefully(tmp_path: Path) -> None:
    arch = FakeArchive()
    arch.add(daily_url("BTCUSDT", "2024-02-01"), make_zip(csv_rows("2024-02-01", 288)))
    arch.add(daily_url("BTCUSDT", "2024-02-03"), make_zip(csv_rows("2024-02-03", 288)))
    df = make_source(arch, tmp_path).fetch_klines("BTCUSDT", date(2024, 2, 1), date(2024, 2, 3))
    assert len(df) == 288 * 2


# ── checksums / caching ──────────────────────────────────────────────────────────────
def test_checksum_verified_and_recorded(tmp_path: Path) -> None:
    arch = FakeArchive()
    payload = make_zip(csv_rows("2024-02-01", 288 * 29))
    arch.add(monthly_url("BTCUSDT", "2024-02"), payload)
    src = make_source(arch, tmp_path)
    src.fetch_klines("BTCUSDT", date(2024, 2, 1), date(2024, 2, 29))
    assert src.source_checksums == {"BTCUSDT-5m-2024-02.zip": hashlib.sha256(payload).hexdigest()}
    cache = tmp_path / "cache" / "_zips" / "BTCUSDT" / "5m"
    assert (cache / "BTCUSDT-5m-2024-02.zip").exists()
    assert (cache / "BTCUSDT-5m-2024-02.zip.CHECKSUM").exists()


def test_checksum_mismatch_raises_and_file_is_not_cached(tmp_path: Path) -> None:
    arch = FakeArchive()
    arch.add(
        monthly_url("BTCUSDT", "2024-02"),
        make_zip(csv_rows("2024-02-01", 288 * 29)),
        checksum="0" * 64,
    )
    src = make_source(arch, tmp_path)
    with pytest.raises(ChecksumError):
        src.fetch_klines("BTCUSDT", date(2024, 2, 1), date(2024, 2, 29))
    assert not list((tmp_path / "cache").rglob("*.zip"))
    assert src.source_checksums == {}


def test_cache_hit_skips_network(tmp_path: Path) -> None:
    arch = FakeArchive()
    arch.add(monthly_url("BTCUSDT", "2024-02"), make_zip(csv_rows("2024-02-01", 288 * 29)))
    make_source(arch, tmp_path).fetch_klines("BTCUSDT", date(2024, 2, 1), date(2024, 2, 29))
    n_first = len(arch.requests)
    again = make_source(arch, tmp_path)
    df = again.fetch_klines("BTCUSDT", date(2024, 2, 1), date(2024, 2, 29))
    assert len(arch.requests) == n_first
    assert len(df) == 288 * 29
    assert "BTCUSDT-5m-2024-02.zip" in again.source_checksums


def test_corrupted_cache_is_redownloaded(tmp_path: Path) -> None:
    arch = FakeArchive()
    arch.add(monthly_url("BTCUSDT", "2024-02"), make_zip(csv_rows("2024-02-01", 288 * 29)))
    make_source(arch, tmp_path).fetch_klines("BTCUSDT", date(2024, 2, 1), date(2024, 2, 29))
    cached = tmp_path / "cache" / "_zips" / "BTCUSDT" / "5m" / "BTCUSDT-5m-2024-02.zip"
    cached.write_bytes(b"corrupt")
    n_before = len(arch.requests)
    df = make_source(arch, tmp_path).fetch_klines("BTCUSDT", date(2024, 2, 1), date(2024, 2, 29))
    assert len(df) == 288 * 29
    assert len(arch.requests) > n_before


# ── retries ──────────────────────────────────────────────────────────────────────────
def test_retries_on_server_error_then_succeeds(tmp_path: Path) -> None:
    arch = FakeArchive()
    arch.add(daily_url("BTCUSDT", "2024-02-01"), make_zip(csv_rows("2024-02-01", 288)))
    failures = {"left": 2}
    delays: list[float] = []

    def flaky(request: httpx.Request) -> httpx.Response:
        if failures["left"] > 0:
            failures["left"] -= 1
            return httpx.Response(503)
        return arch.handler(request)

    src = BinanceVisionSource(
        tmp_path / "cache",
        client=httpx.Client(transport=httpx.MockTransport(flaky)),
        sleep=delays.append,
        backoff_seconds=1.0,
    )
    df = src.fetch_klines("BTCUSDT", date(2024, 2, 1), date(2024, 2, 1))
    assert len(df) == 288
    assert delays == [1.0, 2.0]  # exponential backoff


def test_gives_up_after_three_retries(tmp_path: Path) -> None:
    calls = {"n": 0}

    def always_down(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ConnectError("down", request=request)

    src = BinanceVisionSource(
        tmp_path / "cache",
        client=httpx.Client(transport=httpx.MockTransport(always_down)),
        sleep=lambda _s: None,
    )
    with pytest.raises(DownloadError):
        src.fetch_klines("BTCUSDT", date(2024, 2, 1), date(2024, 2, 1))
    assert calls["n"] == 4  # 1 attempt + 3 retries


def test_user_agent_header_is_generic(tmp_path: Path) -> None:
    arch = FakeArchive()
    arch.add(daily_url("BTCUSDT", "2024-02-01"), make_zip(csv_rows("2024-02-01", 288)))
    make_source(arch, tmp_path).fetch_klines("BTCUSDT", date(2024, 2, 1), date(2024, 2, 1))
    assert {r.headers["User-Agent"] for r in arch.requests} == {"volforecast/0.1"}


# ── Coinbase ─────────────────────────────────────────────────────────────────────────
def test_coinbase_product_mapping() -> None:
    assert coinbase_product("BTCUSDT") == "BTC-USD"
    assert coinbase_product("SOLUSDT") == "SOL-USD"
    assert coinbase_product("ETHUSD") == "ETH-USD"
    with pytest.raises(ValueError, match="cannot map"):
        coinbase_product("BTCEUR")


def test_coinbase_source_pages_and_maps_columns() -> None:
    t0 = int(pd.Timestamp("2024-02-01", tz="UTC").timestamp())
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        start = pd.Timestamp(request.url.params["start"])
        end = pd.Timestamp(request.url.params["end"])
        n = int((end - start) / pd.Timedelta(minutes=5)) + 1
        base = int(start.timestamp())
        # newest first, [time, low, high, open, close, volume]
        rows = [[base + 300 * i, 9.0, 12.0, 10.0, 11.0, 2.0] for i in reversed(range(n))]
        return httpx.Response(200, json=rows)

    src = CoinbaseSource(
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        min_interval_seconds=0.0,
        sleep=lambda _s: None,
    )
    df = src.fetch_klines("BTCUSDT", date(2024, 2, 1), date(2024, 2, 2))
    assert "/products/BTC-USD/candles" in str(seen[0].url)
    assert seen[0].url.params["granularity"] == "300"
    assert len(df) == 288 * 2
    assert len(seen) == 2  # 576 bars -> two windows of <= 300 candles
    assert tuple(df.columns) == KLINE_COLUMNS
    assert df["open_time"].iloc[0] == pd.Timestamp(t0, unit="s", tz="UTC")
    assert df["open_time"].is_monotonic_increasing
    assert df.loc[0, "open"] == 10.0
    assert df.loc[0, "high"] == 12.0
    assert df.loc[0, "low"] == 9.0
    assert df.loc[0, "quote_volume"] == pytest.approx(22.0)  # volume * close
    assert (df["n_trades"] == 0).all()


def test_coinbase_rejects_unknown_interval() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        CoinbaseSource().fetch_klines("BTCUSDT", date(2024, 1, 1), date(2024, 1, 2), "7m")
