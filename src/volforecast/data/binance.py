"""Kline ingestion from public sources (Binance Vision archives, Coinbase Exchange API).

All sources return a frame matching :data:`volforecast.contracts.KLINE_COLUMNS` exactly:
``open_time`` as ``datetime64[ns, UTC]``, float64 prices/volumes, int64 ``n_trades``, a
``RangeIndex``, sorted ascending with unique ``open_time``.
"""

from __future__ import annotations

import hashlib
import io
import re
import time
import zipfile
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Protocol

import httpx
import numpy as np
import pandas as pd

from volforecast.contracts import KLINE_COLUMNS
from volforecast.logging import get_logger

log = get_logger(__name__)

USER_AGENT = "volforecast/0.1"
BINANCE_VISION_BASE = "https://data.binance.vision"
COINBASE_BASE = "https://api.exchange.coinbase.com"

#: Binance raw kline columns, in file order (12 columns).
BINANCE_RAW_COLUMNS: tuple[str, ...] = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
)

#: Timestamps above this are microseconds, below are milliseconds. Epoch-ms values are
#: ~1.5e12 until year 2286, epoch-us values are ~1.7e15 (Binance spot switched on 2025-01-01).
_MICROSECOND_THRESHOLD = 10**14

_FLOAT_COLUMNS = ("open", "high", "low", "close", "volume", "quote_volume")


class ChecksumError(RuntimeError):
    """A downloaded archive does not match its published SHA-256 checksum."""


class DownloadError(RuntimeError):
    """A download failed after all retries."""


class Source(Protocol):
    """A provider of historical klines."""

    def fetch_klines(
        self, symbol: str, start: date, end: date, interval: str = "5m"
    ) -> pd.DataFrame:
        """Klines for UTC days ``start`` through ``end`` inclusive, as the contract frame."""
        ...


def empty_klines() -> pd.DataFrame:
    """An empty frame with the exact contract schema."""
    return pd.DataFrame(
        {
            "open_time": pd.Series([], dtype="datetime64[ns, UTC]"),
            "open": pd.Series([], dtype="float64"),
            "high": pd.Series([], dtype="float64"),
            "low": pd.Series([], dtype="float64"),
            "close": pd.Series([], dtype="float64"),
            "volume": pd.Series([], dtype="float64"),
            "quote_volume": pd.Series([], dtype="float64"),
            "n_trades": pd.Series([], dtype="int64"),
        }
    )[list(KLINE_COLUMNS)]


def normalize_klines(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce to the contract dtypes, sort, drop duplicate bars, reset to a RangeIndex."""
    out = df.loc[:, list(KLINE_COLUMNS)].copy()
    out["open_time"] = pd.to_datetime(out["open_time"], utc=True).astype("datetime64[ns, UTC]")
    for col in _FLOAT_COLUMNS:
        out[col] = out[col].astype("float64")
    out["n_trades"] = out["n_trades"].astype("int64")
    out = out.sort_values("open_time", kind="stable")
    out = out.drop_duplicates(subset="open_time", keep="first")
    return out.reset_index(drop=True)


def epoch_to_datetime(values: pd.Series) -> pd.Series:
    """Convert epoch timestamps to ``datetime64[ns, UTC]``, detecting ms vs µs per call.

    Binance spot archives use milliseconds up to 2024 and microseconds from 2025-01-01.
    The unit is decided from the magnitude of the values (> 1e14 means microseconds), so a
    single file is always parsed consistently.
    """
    ints = values.astype("int64")
    if ints.empty:
        return pd.Series([], dtype="datetime64[ns, UTC]")
    if int(ints.max()) > _MICROSECOND_THRESHOLD:
        parsed = pd.to_datetime(ints, unit="us", utc=True)
    else:
        parsed = pd.to_datetime(ints, unit="ms", utc=True)
    return pd.Series(parsed.astype("datetime64[ns, UTC]"), index=values.index)


def parse_binance_csv(raw: bytes | str) -> pd.DataFrame:
    """Parse one Binance kline CSV (with or without a header row) to the contract frame."""
    text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    first_field = text.split(",", 1)[0].strip()
    has_header = not first_field.lstrip("-").isdigit()
    df = pd.read_csv(
        io.StringIO(text),
        header=0 if has_header else None,
        names=list(BINANCE_RAW_COLUMNS),
        usecols=range(len(BINANCE_RAW_COLUMNS)),
        dtype={
            "open_time": "int64",
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "volume": "float64",
            "quote_volume": "float64",
            "count": "int64",
        },
    )
    out = pd.DataFrame(
        {
            "open_time": epoch_to_datetime(df["open_time"]),
            "open": df["open"],
            "high": df["high"],
            "low": df["low"],
            "close": df["close"],
            "volume": df["volume"],
            "quote_volume": df["quote_volume"],
            "n_trades": df["count"],
        }
    )
    return normalize_klines(out)


def interval_timedelta(interval: str) -> pd.Timedelta:
    """``'5m'`` -> 5 minutes, ``'1h'`` -> 1 hour, ``'1d'`` -> 1 day."""
    units = {"m": "min", "h": "h", "d": "D"}
    if len(interval) < 2 or interval[-1] not in units or not interval[:-1].isdigit():
        raise ValueError(f"unsupported interval {interval!r}")
    return pd.Timedelta(f"{int(interval[:-1])}{units[interval[-1]]}")


def snap_to_grid(df: pd.DataFrame, interval: str = "5m") -> tuple[pd.DataFrame, int, int]:
    """Floor off-grid bar timestamps to their grid slot.

    Binance occasionally publishes bars whose ``open_time`` is offset from the interval grid
    (e.g. after the February 2018 outage, 241 bars are +194.8 s off). Bars offset by less than
    one interval are floored to the slot they fall in. If a snapped bar collides with another
    bar in the same slot, the on-grid bar wins (otherwise the first snapped bar) and the other
    is dropped.

    Returns ``(frame, n_snapped, n_dropped)`` where ``n_snapped`` counts the off-grid bars
    that survive (with moved timestamps) and ``n_dropped`` the bars removed by collisions.
    """
    if df.empty:
        return df, 0, 0
    step = interval_timedelta(interval)
    floored = df["open_time"].dt.floor(step)
    off_grid = (df["open_time"] != floored).to_numpy()
    if not off_grid.any():
        return df, 0, 0
    work = df.copy()
    work["open_time"] = floored
    work["_off_grid"] = off_grid
    work = work.sort_values(["open_time", "_off_grid"], kind="stable")
    work = work.drop_duplicates(subset="open_time", keep="first")
    n_dropped = len(df) - len(work)
    n_snapped = int(work["_off_grid"].sum())
    out = normalize_klines(work.drop(columns="_off_grid"))
    return out, n_snapped, n_dropped


def parse_checksum_file(text: str) -> str:
    """Extract the SHA-256 hex digest from a ``<sha256>  <filename>`` checksum file."""
    token = text.strip().split()[0].lower() if text.strip() else ""
    if not re.fullmatch(r"[0-9a-f]{64}", token):
        raise ChecksumError(f"malformed checksum file: {text[:80]!r}")
    return token


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _month_bounds(year: int, month: int) -> tuple[date, date]:
    first = date(year, month, 1)
    nxt = date(year + (month == 12), month % 12 + 1, 1)
    return first, nxt - timedelta(days=1)


class BinanceVisionSource:
    """Binance public data archive (``data.binance.vision``), spot klines.

    Full months inside the requested range use monthly archives; the (partial) month that
    contains ``end`` -- and any month whose monthly archive is not published yet (404) --
    uses daily archives. Every archive's sibling ``.zip.CHECKSUM`` is downloaded and
    verified before the zip is used; a mismatch raises :class:`ChecksumError` and the file
    is discarded. Verified files are cached under ``<cache_dir>/_zips/{symbol}/{interval}/``
    and never re-downloaded.
    """

    def __init__(
        self,
        cache_dir: Path,
        *,
        client: httpx.Client | None = None,
        base_url: str = BINANCE_VISION_BASE,
        max_retries: int = 3,
        backoff_seconds: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self._sleep = sleep
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(30.0, connect=10.0),
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
        )
        #: archive file name -> verified SHA-256, for every archive used by this instance.
        self.source_checksums: dict[str, str] = {}
        #: off-grid bars floored to the grid / bars dropped because of slot collisions.
        self.snapped_bars = 0
        self.dropped_bars = 0

    # -- URLs / cache -----------------------------------------------------------------
    def archive_url(self, symbol: str, interval: str, period: date, *, monthly: bool) -> str:
        """URL of the monthly (``YYYY-MM``) or daily (``YYYY-MM-DD``) archive."""
        stamp = period.strftime("%Y-%m") if monthly else period.isoformat()
        kind = "monthly" if monthly else "daily"
        name = f"{symbol}-{interval}-{stamp}.zip"
        return f"{self.base_url}/data/spot/{kind}/klines/{symbol}/{interval}/{name}"

    def _zip_dir(self, symbol: str, interval: str) -> Path:
        return self.cache_dir / "_zips" / symbol / interval

    # -- HTTP -------------------------------------------------------------------------
    def _get(self, url: str) -> bytes | None:
        """GET with retries and exponential backoff. Returns None on 404."""
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.get(url, headers={"User-Agent": USER_AGENT})
            except httpx.TransportError as exc:
                last_error = exc
            else:
                if resp.status_code == 404:
                    return None
                if resp.status_code == 200:
                    return resp.content
                if resp.status_code < 500 and resp.status_code != 429:
                    raise DownloadError(f"GET {url} -> HTTP {resp.status_code}")
                last_error = DownloadError(f"GET {url} -> HTTP {resp.status_code}")
            if attempt < self.max_retries:
                delay = self.backoff_seconds * 2**attempt
                log.warning("download_retry", url=url, attempt=attempt + 1, delay=delay)
                self._sleep(delay)
        raise DownloadError(f"GET {url} failed after {self.max_retries} retries") from last_error

    def _load_archive(self, symbol: str, interval: str, url: str) -> bytes | None:
        """Verified zip bytes for ``url`` (cache first); None if not published (404)."""
        name = url.rsplit("/", 1)[1]
        cache = self._zip_dir(symbol, interval)
        zip_path, sum_path = cache / name, cache / f"{name}.CHECKSUM"

        if zip_path.exists() and sum_path.exists():
            try:
                expected = parse_checksum_file(sum_path.read_text())
                data = zip_path.read_bytes()
                if _sha256(data) == expected:
                    self.source_checksums[name] = expected
                    return data
            except ChecksumError:
                pass
            log.warning("cache_invalid", file=name)
            zip_path.unlink(missing_ok=True)
            sum_path.unlink(missing_ok=True)

        checksum_raw = self._get(url + ".CHECKSUM")
        if checksum_raw is None:
            return None
        expected = parse_checksum_file(checksum_raw.decode("utf-8", "replace"))
        payload = self._get(url)
        if payload is None:
            return None
        data = payload
        actual = _sha256(data)
        if actual != expected:
            raise ChecksumError(f"{name}: sha256 {actual} != published {expected}; file discarded")

        cache.mkdir(parents=True, exist_ok=True)
        tmp = zip_path.with_suffix(".zip.tmp")
        tmp.write_bytes(data)
        tmp.replace(zip_path)
        sum_path.write_bytes(checksum_raw)
        self.source_checksums[name] = expected
        return data

    def _read_zip(self, data: bytes, interval: str, name: str = "") -> pd.DataFrame:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            member = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
            df = parse_binance_csv(zf.read(member))
        df, n_snapped, n_dropped = snap_to_grid(df, interval)
        if n_snapped or n_dropped:
            log.warning("snapped_to_grid", file=name, snapped=n_snapped, dropped=n_dropped)
            self.snapped_bars += n_snapped
            self.dropped_bars += n_dropped
        return df

    # -- public API -------------------------------------------------------------------
    def fetch_klines(
        self, symbol: str, start: date, end: date, interval: str = "5m"
    ) -> pd.DataFrame:
        """Klines for UTC days ``start``..``end`` inclusive. Unavailable days are skipped."""
        if end < start:
            raise ValueError(f"end {end} is before start {start}")
        frames: list[pd.DataFrame] = []
        n_missing_days = 0
        self.snapped_bars = self.dropped_bars = 0  # counters describe the latest fetch
        year, month = start.year, start.month
        while date(year, month, 1) <= end:
            first, last = _month_bounds(year, month)
            lo, hi = max(first, start), min(last, end)
            monthly_frame: pd.DataFrame | None = None
            if last <= end:  # full month inside the requested range
                data = self._load_archive(
                    symbol, interval, self.archive_url(symbol, interval, first, monthly=True)
                )
                if data is not None:
                    monthly_frame = self._read_zip(data, interval, f"{symbol}-{first:%Y-%m}")
            if monthly_frame is not None:
                frames.append(monthly_frame)
            else:
                n_missing_days += self._fetch_daily(symbol, interval, lo, hi, frames)
            year, month = (year + 1, 1) if month == 12 else (year, month + 1)

        if not frames:
            log.warning("no_data", symbol=symbol, start=str(start), end=str(end))
            return empty_klines()
        df = normalize_klines(pd.concat(frames, ignore_index=True))
        lo_ts = pd.Timestamp(start, tz="UTC")
        hi_ts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
        df = df[(df["open_time"] >= lo_ts) & (df["open_time"] < hi_ts)].reset_index(drop=True)
        if n_missing_days:
            log.warning("archives_unavailable", symbol=symbol, days=n_missing_days)
        return df

    def _fetch_daily(
        self, symbol: str, interval: str, lo: date, hi: date, frames: list[pd.DataFrame]
    ) -> int:
        """Append daily archives for ``lo``..``hi``; return the number of unavailable days."""
        missing = 0
        for offset in range((hi - lo).days + 1):
            day = lo + timedelta(days=offset)
            data = self._load_archive(
                symbol, interval, self.archive_url(symbol, interval, day, monthly=False)
            )
            if data is None:
                missing += 1
                log.info("archive_not_available", symbol=symbol, day=day.isoformat())
            else:
                frames.append(self._read_zip(data, interval, f"{symbol}-{day}"))
        return missing


_COINBASE_GRANULARITY = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "6h": 21600, "1d": 86400}
_QUOTE_SUFFIXES = ("USDT", "USDC", "BUSD", "USD")


def coinbase_product(symbol: str) -> str:
    """Map a Binance-style symbol to a Coinbase product id (BTCUSDT -> BTC-USD)."""
    for suffix in _QUOTE_SUFFIXES:
        if symbol.endswith(suffix) and len(symbol) > len(suffix):
            return f"{symbol[: -len(suffix)]}-USD"
    raise ValueError(f"cannot map symbol {symbol!r} to a Coinbase product")


class CoinbaseSource:
    """Coinbase Exchange public candles API -- a fallback / cross-check source.

    Differences from Binance data (documented approximations):

    * ``quote_volume`` is ``volume * close`` (the API does not report quote volume).
    * ``n_trades`` is 0 (the API does not report trade counts).
    * Prices are USD, not USDT, and bars with no trades are absent (gaps).

    The API returns at most 300 candles per request, newest first, as
    ``[time, low, high, open, close, volume]``; requests are paged in 300-bar windows with a
    gentle client-side rate limit.
    """

    MAX_CANDLES = 300

    def __init__(
        self,
        *,
        client: httpx.Client | None = None,
        base_url: str = COINBASE_BASE,
        min_interval_seconds: float = 0.4,
        max_retries: int = 3,
        backoff_seconds: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.min_interval_seconds = min_interval_seconds
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self._sleep = sleep
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(30.0, connect=10.0),
            headers={"User-Agent": USER_AGENT},
        )
        self._last_request = 0.0
        self.source_checksums: dict[str, str] = {}

    def _request(self, url: str, params: dict[str, str | int]) -> list[list[float]]:
        for attempt in range(self.max_retries + 1):
            wait = self.min_interval_seconds - (time.monotonic() - self._last_request)
            if wait > 0:
                self._sleep(wait)
            self._last_request = time.monotonic()
            try:
                resp = self._client.get(url, params=params, headers={"User-Agent": USER_AGENT})
            except httpx.TransportError:
                resp = None
            if resp is not None and resp.status_code == 200:
                payload: list[list[float]] = resp.json()
                return payload
            if resp is not None and resp.status_code < 500 and resp.status_code != 429:
                raise DownloadError(f"GET {url} -> HTTP {resp.status_code}: {resp.text[:200]}")
            if attempt < self.max_retries:
                self._sleep(self.backoff_seconds * 2**attempt)
        raise DownloadError(f"GET {url} failed after {self.max_retries} retries")

    def fetch_klines(
        self, symbol: str, start: date, end: date, interval: str = "5m"
    ) -> pd.DataFrame:
        """Candles for UTC days ``start``..``end`` inclusive, as the contract frame."""
        if interval not in _COINBASE_GRANULARITY:
            raise ValueError(f"unsupported Coinbase interval {interval!r}")
        if end < start:
            raise ValueError(f"end {end} is before start {start}")
        gran = _COINBASE_GRANULARITY[interval]
        url = f"{self.base_url}/products/{coinbase_product(symbol)}/candles"
        window = timedelta(seconds=gran * self.MAX_CANDLES)
        cursor = datetime(start.year, start.month, start.day, tzinfo=UTC)
        stop = datetime(end.year, end.month, end.day, tzinfo=UTC) + timedelta(days=1)
        rows: list[list[float]] = []
        while cursor < stop:
            nxt = min(cursor + window, stop)
            params: dict[str, str | int] = {
                "granularity": gran,
                "start": cursor.isoformat(),
                # The API treats `end` as inclusive: stop one bar early to avoid overlap.
                "end": (nxt - timedelta(seconds=gran)).isoformat(),
            }
            rows.extend(self._request(url, params))
            cursor = nxt
        return coinbase_rows_to_klines(rows)


def coinbase_rows_to_klines(rows: list[list[float]]) -> pd.DataFrame:
    """Convert Coinbase ``[time, low, high, open, close, volume]`` rows to the contract frame."""
    if not rows:
        return empty_klines()
    arr = np.asarray(rows, dtype="float64")
    close = arr[:, 4]
    volume = arr[:, 5]
    df = pd.DataFrame(
        {
            "open_time": pd.to_datetime(arr[:, 0].astype("int64"), unit="s", utc=True),
            "open": arr[:, 3],
            "high": arr[:, 2],
            "low": arr[:, 1],
            "close": close,
            "volume": volume,
            "quote_volume": volume * close,
            "n_trades": np.zeros(len(arr), dtype="int64"),
        }
    )
    return normalize_klines(df)
