"""Parquet/DuckDB storage for klines and daily panels, content hashing and run manifests.

Layout under the data root::

    raw/klines/symbol=X/interval=5m/year=YYYY/month=MM/part.parquet   (zstd, hive-partitioned)
    raw/klines/symbol=X/interval=5m/manifest.json
    daily/symbol=X/panel.parquet
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

import volforecast
from volforecast.contracts import KLINE_COLUMNS
from volforecast.data.binance import empty_klines, normalize_klines
from volforecast.logging import get_logger

log = get_logger(__name__)


def klines_dir(root: Path, symbol: str, interval: str = "5m") -> Path:
    """Directory holding the hive partitions for one symbol/interval."""
    return Path(root) / "raw" / "klines" / f"symbol={symbol}" / f"interval={interval}"


def panel_path(root: Path, symbol: str) -> Path:
    """Path of the daily panel Parquet for ``symbol``."""
    return Path(root) / "daily" / f"symbol={symbol}" / "panel.parquet"


def _atomic_parquet(df: pd.DataFrame, path: Path, *, index: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, engine="pyarrow", compression="zstd", index=index)
    tmp.replace(path)


def write_klines(df: pd.DataFrame, root: Path, symbol: str, interval: str = "5m") -> list[Path]:
    """Write klines as zstd Parquet, partitioned by UTC year/month.

    Idempotent: every partition touched by ``df`` is rewritten completely from the rows of
    ``df`` that fall in that month (so pass whole months, or accept that a partial month
    replaces the stored one). Partitions not touched are left alone. Returns written paths.
    """
    frame = normalize_klines(df)
    if frame.empty:
        return []
    t = frame["open_time"]
    ym = (t.dt.year * 100 + t.dt.month).to_numpy()
    base = klines_dir(root, symbol, interval)
    written: list[Path] = []
    for code in np.unique(ym):
        year, month = divmod(int(code), 100)
        part = frame.loc[ym == code].reset_index(drop=True)
        path = base / f"year={year:04d}" / f"month={month:02d}" / "part.parquet"
        _atomic_parquet(part, path, index=False)
        written.append(path)
    log.info("klines_written", symbol=symbol, interval=interval, partitions=len(written))
    return written


def read_klines(
    root: Path,
    symbol: str,
    start: date | None = None,
    end: date | None = None,
    interval: str = "5m",
) -> pd.DataFrame:
    """Read klines via DuckDB; ``start``/``end`` are inclusive UTC calendar days."""
    base = klines_dir(root, symbol, interval)
    if not base.exists():
        raise FileNotFoundError(f"no klines stored for {symbol} {interval} under {base}")
    glob = str(base / "year=*" / "month=*" / "part.parquet")
    cols = ", ".join(KLINE_COLUMNS)
    where: list[str] = []
    params: list[Any] = [glob]
    if start is not None:
        where.append("open_time::TIMESTAMPTZ >= ?::TIMESTAMPTZ")
        params.append(f"{start.isoformat()} 00:00:00+00")
    if end is not None:
        where.append("open_time::TIMESTAMPTZ < (?::TIMESTAMPTZ + INTERVAL 1 DAY)")
        params.append(f"{end.isoformat()} 00:00:00+00")
    clause = f"WHERE {' AND '.join(where)}" if where else ""
    sql = f"SELECT {cols} FROM read_parquet(?, hive_partitioning=true) {clause} ORDER BY open_time"
    con = duckdb.connect()
    try:
        con.execute("SET TimeZone='UTC'")
        out = con.execute(sql, params).fetch_df()
    finally:
        con.close()
    if out.empty:
        return empty_klines()
    return normalize_klines(out)


def write_panel(panel: pd.DataFrame, root: Path, symbol: str) -> Path:
    """Write the daily panel (the ``date`` index is preserved) to ``daily/symbol=X``."""
    path = panel_path(root, symbol)
    _atomic_parquet(panel, path, index=True)
    log.info("panel_written", symbol=symbol, rows=len(panel))
    return path


def read_panel(root: Path, symbol: str) -> pd.DataFrame:
    """Read the daily panel written by :func:`write_panel`."""
    path = panel_path(root, symbol)
    if not path.exists():
        raise FileNotFoundError(f"no panel stored for {symbol}: {path}")
    return pd.read_parquet(path, engine="pyarrow")


def _canonical_units(df: pd.DataFrame) -> pd.DataFrame:
    """Cast datetime columns/index to ns resolution so hashes don't depend on pandas' unit."""
    out = df.copy()
    for col in out.columns:
        if isinstance(out[col].dtype, pd.DatetimeTZDtype) or out[col].dtype.kind == "M":
            out[col] = out[col].dt.as_unit("ns")
    if isinstance(out.index, pd.DatetimeIndex):
        out.index = out.index.as_unit("ns")
    return out


def dataset_hash(df: pd.DataFrame) -> str:
    """Deterministic SHA-256 content hash of a frame (values, index, column names, dtypes)."""
    df = _canonical_units(df)
    h = hashlib.sha256()
    spec = {
        "columns": [[str(c), str(df[c].dtype)] for c in df.columns],
        "index": [str(df.index.name), str(df.index.dtype)],
        "rows": len(df),
    }
    h.update(json.dumps(spec, sort_keys=True).encode())
    row_hashes = pd.util.hash_pandas_object(df, index=True).to_numpy(dtype="<u8")
    h.update(row_hashes.tobytes())
    return h.hexdigest()


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() or "unknown"


def _time_bounds(df: pd.DataFrame) -> tuple[str | None, str | None]:
    if df.empty:
        return None, None
    values = df["open_time"] if "open_time" in df.columns else df.index.to_series()
    return pd.Timestamp(values.min()).isoformat(), pd.Timestamp(values.max()).isoformat()


def write_manifest(
    path: Path,
    *,
    dataset: str,
    df: pd.DataFrame,
    source_checksums: Mapping[str, str],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a JSON provenance manifest for ``df`` to ``path`` and return it."""
    min_time, max_time = _time_bounds(df)
    manifest: dict[str, Any] = {
        "dataset": dataset,
        "rows": len(df),
        "min_time": min_time,
        "max_time": max_time,
        "content_hash": dataset_hash(df),
        "source_checksums": dict(sorted(source_checksums.items())),
        "git_sha": _git_sha(),
        "created_at": datetime.now(UTC).isoformat(),
        "volforecast_version": volforecast.__version__,
        "extra": dict(extra or {}),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    tmp.replace(path)
    return manifest
