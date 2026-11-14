"""Kline loading and cached daily realized-panel building.

Layout (hive): ``<data_root>/raw/klines/symbol=<S>/interval=5m/year=<Y>/month=<M>/*.parquet``;
the daily cache lives at ``<data_root>/daily/symbol=<S>/panel.parquet`` with a JSON sidecar
``panel.meta.json`` holding the content hash and a signature of the raw files it came from.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from volforecast.contracts import KLINE_COLUMNS, PANEL_COLUMNS
from volforecast.features.realized import daily_realized_measures
from volforecast.logging import get_logger

log = get_logger(__name__)

INTERVAL = "5m"


def _interval_dir(data_root: Path, symbol: str) -> Path:
    return data_root / "raw" / "klines" / f"symbol={symbol}" / f"interval={INTERVAL}"


def kline_files(data_root: Path, symbol: str) -> list[Path]:
    return sorted(_interval_dir(data_root, symbol).rglob("*.parquet"))


def load_klines(data_root: str | Path, symbol: str) -> pd.DataFrame:
    """Read all 5-minute klines of ``symbol`` (contract schema, sorted, UTC ``open_time``).

    TODO: switch to ``volforecast.data.read_klines`` once the data module is merged.
    """
    root = Path(data_root)
    if not kline_files(root, symbol):
        raise FileNotFoundError(f"no kline parquet files for {symbol} under {root / 'raw'}")
    cols = ", ".join(c for c in KLINE_COLUMNS if c != "open_time")
    # epoch_ns avoids duckdb's pytz-dependent TIMESTAMPTZ -> pandas conversion.
    query = (
        f"SELECT epoch_ns(open_time) AS open_ns, {cols} "
        "FROM read_parquet(?, hive_partitioning=true) ORDER BY open_ns"
    )
    glob = str(_interval_dir(root, symbol) / "**" / "*.parquet")
    con = duckdb.connect()
    try:
        df = con.execute(query, [glob]).fetchdf()
    finally:
        con.close()
    open_time = pd.to_datetime(df.pop("open_ns"), unit="ns", utc=True).astype("datetime64[ns, UTC]")
    df.insert(0, "open_time", open_time)
    df = df.drop_duplicates("open_time", keep="last").reset_index(drop=True)
    df["n_trades"] = df["n_trades"].astype("int64")
    return df[list(KLINE_COLUMNS)]


def content_hash(panel: pd.DataFrame) -> str:
    """sha256 over row hashes (index included), column names and dtypes."""
    h = hashlib.sha256()
    h.update(",".join(f"{c}:{panel[c].dtype}" for c in panel.columns).encode())
    h.update(pd.util.hash_pandas_object(panel, index=True).to_numpy(dtype=np.uint64).tobytes())
    return h.hexdigest()


def _source_signature(files: list[Path], root: Path) -> str:
    h = hashlib.sha256()
    for f in files:
        st = f.stat()
        h.update(f"{f.relative_to(root)}|{st.st_size}|{st.st_mtime_ns}".encode())
    return h.hexdigest()


def panel_paths(data_root: Path, symbol: str) -> tuple[Path, Path]:
    d = data_root / "daily" / f"symbol={symbol}"
    return d / "panel.parquet", d / "panel.meta.json"


def build_panel(
    data_root: str | Path, symbol: str, force: bool = False
) -> tuple[pd.DataFrame, str]:
    """Daily realized panel for ``symbol`` and its content hash, cached on disk.

    The cache is reused only while the raw kline files (path, size, mtime) are unchanged and
    the stored hash matches the parquet content.
    """
    root = Path(data_root)
    pq, meta_path = panel_paths(root, symbol)
    files = kline_files(root, symbol)
    signature = _source_signature(files, root)
    if not force and pq.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("source_signature") == signature:
            cached = pd.read_parquet(pq)
            if content_hash(cached) == meta.get("hash"):
                log.info("panel_cache_hit", symbol=symbol, hash=str(meta["hash"])[:12])
                return cached, str(meta["hash"])
            log.warning("panel_cache_corrupt", symbol=symbol)
    panel = daily_realized_measures(load_klines(root, symbol))[list(PANEL_COLUMNS)]
    digest = content_hash(panel)
    pq.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(pq)
    meta_path.write_text(
        json.dumps(
            {
                "symbol": symbol,
                "hash": digest,
                "source_signature": signature,
                "n_days": len(panel),
                "first_date": str(panel.index[0].date()) if len(panel) else None,
                "last_date": str(panel.index[-1].date()) if len(panel) else None,
                "n_source_files": len(files),
            },
            indent=2,
        )
    )
    log.info("panel_built", symbol=symbol, n_days=len(panel), hash=digest[:12])
    return panel, digest
