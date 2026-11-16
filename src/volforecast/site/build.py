"""Build the static dashboard: ``index.html`` + assets + ``data/forecasts.json``.

Inputs: a forecast store (the live out-of-sample track record, read only through the repository
helpers) and the committed backtest tables (``reports/tables.md``). Output is plain static files
(Plotly is loaded from a pinned CDN URL by the page), so any static host serves it.

The page never contains invented numbers: with an empty store it says the track record starts
on the first forecast date. ``--demo`` flags the payload (and the page shows a banner) so a
scratch store filled by replay can never pass as a live record.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import text

from volforecast.config import PROJECT_ROOT
from volforecast.contracts import ANNUALIZATION_DAYS
from volforecast.evaluation.losses import qlike
from volforecast.forecast_store import ROLE_BENCHMARK, SERVED_ROLES, ForecastRepository
from volforecast.logging import get_logger

log = get_logger(__name__)

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
STATIC_FILES = ("index.html", "app.js", "style.css")
SCHEMA_VERSION = 1
DISCLAIMER = (
    "This site forecasts the volatility (realized variance) of crypto assets, not their price "
    "direction. It is a research project, not investment advice."
)
#: Days of history written per (asset, horizon) to keep forecasts.json small.
HISTORY_DAYS = 540
#: Realized-forecast pairs needed before the cumulative QLIKE chart is meaningful.
MIN_QLIKE_DAYS = 20


# ---------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------
def _num(x: Any) -> float | None:
    """Finite float or None (JSON has no NaN)."""
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _iso(d: Any) -> str | None:
    if d is None or d is pd.NaT:
        return None
    return pd.Timestamp(d).date().isoformat()


def _ann_vol(var: float | None, h: int) -> float | None:
    """Annualized volatility of an h-day variance forecast/realization (365 days)."""
    if var is None or var < 0:
        return None
    return math.sqrt(var / h * ANNUALIZATION_DAYS)


def git_sha(root: Path = PROJECT_ROOT) -> str:
    """Short git SHA (``$GITHUB_SHA`` first, then ``git rev-parse``), or ``"unknown"``."""
    env = os.environ.get("GITHUB_SHA")
    if env:
        return env[:7]
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


# ---------------------------------------------------------------------------------------
# forecast store -> JSON
# ---------------------------------------------------------------------------------------
def _horizon_block(
    repo: ForecastRepository, asset: str, h: int, summary: dict[str, Any] | None
) -> tuple[dict[str, Any], date] | None:
    latest = repo.latest_forecast(asset, h)
    if latest is None:
        return None
    hist = repo.forecast_history(asset, h)
    bench = repo.forecasts_df(asset, h, roles=(ROLE_BENCHMARK,))
    if bench.empty:
        hist["har_var"] = np.nan
    else:
        har = bench.drop_duplicates("origin_date", keep="last").set_index("origin_date")
        hist["har_var"] = hist["origin_date"].map(har["forecast_var"]).astype(float)
    rows: list[dict[str, Any]] = []
    for r in hist.tail(HISTORY_DAYS).itertuples(index=False):
        fv, rv = _num(r.forecast_var), _num(r.realized_var)
        rows.append(
            {
                "origin_date": _iso(r.origin_date),
                "target_date": _iso(r.target_date),
                "forecast_var": fv,
                "forecast_vol": _ann_vol(fv, h),
                "realized_var": rv,
                "realized_vol": _ann_vol(rv, h),
                "har_var": _num(r.har_var),
                "var99": _num(r.var99),
            }
        )
    # cumulative QLIKE, champion vs HAR, on origins where both forecasts and the outcome exist
    known = hist.dropna(subset=["realized_var", "har_var"])
    known = known[
        (known["realized_var"] > 0) & (known["forecast_var"] > 0) & (known["har_var"] > 0)
    ]
    cum: dict[str, Any] = {"n": len(known), "min_days": MIN_QLIKE_DAYS, "dates": []}
    if len(known):
        rvv = known["realized_var"].to_numpy(float)
        lc = np.cumsum(qlike(rvv, known["forecast_var"].to_numpy(float)))
        lh = np.cumsum(qlike(rvv, known["har_var"].to_numpy(float)))
        cum.update(
            dates=[_iso(d) for d in known["target_date"]],
            champion=[_num(v) for v in lc],
            har=[_num(v) for v in lh],
            ratio_vs_har=_num(lc[-1] / lh[-1]) if lh[-1] > 0 else None,
        )
    fv0 = _num(latest["forecast_var"])
    block = {
        "horizon": h,
        "model_name": latest["model_name"],
        "model_version": latest["model_version"],
        "role": latest["role"],
        "n_forecasts": len(hist),
        "n_realized": int(hist["realized_var"].notna().sum()),
        "challenger_version": None if summary is None else summary["challenger_version"],
        "latest": {
            "origin_date": _iso(latest["origin_date"]),
            "target_date": _iso(latest["target_date"]),
            "forecast_var": fv0,
            "forecast_vol": _ann_vol(fv0, h),
            "var95": _num(latest.get("var95")),
            "var99": _num(latest.get("var99")),
            "es975": _num(latest.get("es975")),
            "dist": latest.get("dist"),
        },
        "history": rows,
        "cum_qlike": cum,
    }
    first = pd.Timestamp(hist["origin_date"].min()).date()
    return block, first


def build_forecasts_payload(
    repo: ForecastRepository,
    *,
    assets: Sequence[str],
    horizons: Sequence[int],
    sha: str,
    generated_at: datetime,
    demo: bool = False,
) -> dict[str, Any]:
    """The ``data/forecasts.json`` document (pure data, JSON-safe)."""
    summaries = {(s["asset"], s["horizon"]): s for s in repo.models_summary()}
    out_assets: dict[str, Any] = {}
    starts: list[date] = []
    for asset in assets:
        blocks: dict[str, Any] = {}
        for h in horizons:
            res = _horizon_block(repo, asset, h, summaries.get((asset, h)))
            if res is None:
                continue
            blocks[str(h)] = res[0]
            starts.append(res[1])
        out_assets[asset] = {"horizons": blocks}
    start = min(starts) if starts else None
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at.astimezone(UTC).isoformat(timespec="seconds"),
        "git_sha": sha,
        "demo": demo,
        "disclaimer": DISCLAIMER,
        "annualization_days": ANNUALIZATION_DAYS,
        "served_roles": list(SERVED_ROLES),
        "horizons": [int(h) for h in horizons],
        "assets": out_assets,
        "track_record_start": None if start is None else start.isoformat(),
        "empty": not any(a["horizons"] for a in out_assets.values()),
    }


# ---------------------------------------------------------------------------------------
# reports/tables.md -> backtest summary
# ---------------------------------------------------------------------------------------
_ASSET_RE = re.compile(r"^## ([A-Z0-9]+)\s*$")
_H_RE = re.compile(r"^### h=(\d+) \(n=(\d+), origins (\S+) to (\S+)\)")


def _cell(s: str) -> float | None:
    try:
        return _num(float(s.strip()))
    except ValueError:
        return None


def parse_backtest_tables(md: str) -> dict[str, Any]:
    """Parse the per-asset/horizon main tables of ``reports/tables.md``.

    Returns ``{"tables": {asset: {h: {n, first, last, rows: [...]}}}}``; rows carry QLIKE and
    the ratios / DM p-values the report renders. Other sections are ignored.
    """
    tables: dict[str, dict[str, Any]] = {}
    asset: str | None = None
    block: dict[str, Any] | None = None
    header: list[str] | None = None
    for line in md.splitlines():
        m = _ASSET_RE.match(line)
        if m:
            asset, block, header = m.group(1), None, None
            continue
        if line.startswith("## "):
            asset, block, header = None, None, None
            continue
        m = _H_RE.match(line)
        if m and asset:
            block = {"n": int(m.group(2)), "first": m.group(3), "last": m.group(4), "rows": []}
            tables.setdefault(asset, {})[m.group(1)] = block
            header = None
            continue
        if line.startswith("#"):
            block = None  # a sub-table (by year / regime ...) ends the main table
            continue
        if block is not None and header is not None and not line.startswith("|"):
            block = None  # the first table ended; later sub-tables (by year, ...) are ignored
            continue
        if block is None or not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if header is None:
            header = cells
            continue
        if set(line.replace("|", "").strip()) <= {"-", " "}:
            continue
        rec = dict(zip(header, cells, strict=False))
        name = rec.get("model", "")
        block["rows"].append(
            {
                "model": name.replace(" (benchmark)", ""),
                "benchmark": "(benchmark)" in name,
                "qlike": _cell(rec.get("QLIKE", "")),
                "ratio_vs_har": _cell(rec.get("ratio vs HAR", "")),
                "dm_p_vs_har": _cell(rec.get("DM p (beats HAR)", "")),
                "ratio_vs_log_har_dow": _cell(rec.get("ratio vs log_har_dow", "")),
                "dm_p_vs_log_har_dow": _cell(rec.get("DM p (beats log_har_dow)", "")),
                "mcs_90": rec.get("MCS 90%") == "yes",
            }
        )
    return {"tables": {a: {h: b for h, b in hs.items() if b["rows"]} for a, hs in tables.items()}}


def build_backtest_payload(
    reports_dir: Path, sha: str, generated_at: datetime
) -> dict[str, Any] | None:
    """The ``data/backtest_summary.json`` document, or None if ``tables.md`` is absent."""
    path = reports_dir / "tables.md"
    if not path.is_file():
        return None
    parsed = parse_backtest_tables(path.read_text(encoding="utf-8"))
    if not any(parsed["tables"].values()):
        return None
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at.astimezone(UTC).isoformat(timespec="seconds"),
        "git_sha": sha,
        "source": "reports/tables.md (walk-forward, expanding window, benchmark HAR)",
        "note": (
            "Most of the gain over HAR is a weekend (calendar) effect: crypto weekend variance "
            "is far lower than weekday variance. Against the weekend-aware log-HAR-DoW benchmark "
            "the XGBoost edge is statistically significant only at h=1 on BTC and marginal on "
            "ETH; at h=5 on SOL the linear model is better."
        ),
        **parsed,
    }


# ---------------------------------------------------------------------------------------
# site assembly
# ---------------------------------------------------------------------------------------
def _write_json(path: Path, doc: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, allow_nan=False, indent=1) + "\n", encoding="utf-8")


def publish_store_copy(repo: ForecastRepository, dest: Path) -> None:
    """Write a compacted copy of a sqlite store (``VACUUM INTO``) to ``dest``."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.unlink(missing_ok=True)
    with repo.engine.connect() as conn:
        conn.execute(text("VACUUM INTO :p"), {"p": str(dest)})


def build_site(
    out_dir: str | Path,
    database_url: str | Path,
    *,
    reports_dir: str | Path | None = None,
    assets: Sequence[str] = ("BTCUSDT", "ETHUSDT", "SOLUSDT"),
    horizons: Sequence[int] = (1, 5),
    sha: str | None = None,
    generated_at: datetime | None = None,
    demo: bool = False,
    publish_store: bool = False,
) -> Path:
    """Write the static site to ``out_dir`` and return it. Works on an empty/new store."""
    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    now = generated_at or datetime.now(UTC)
    sha = sha or git_sha()
    url = str(database_url)
    if "://" not in url:
        url = f"sqlite:///{Path(url).resolve()}"
    repo = ForecastRepository(url)
    for name in STATIC_FILES:
        shutil.copyfile(TEMPLATE_DIR / name, out / name)
    (out / ".nojekyll").write_text("", encoding="utf-8")
    payload = build_forecasts_payload(
        repo, assets=assets, horizons=horizons, sha=sha, generated_at=now, demo=demo
    )
    _write_json(out / "data" / "forecasts.json", payload)
    rdir = Path(reports_dir) if reports_dir is not None else PROJECT_ROOT / "reports"
    backtest = build_backtest_payload(rdir, sha, now)
    if backtest is not None:
        _write_json(out / "data" / "backtest_summary.json", backtest)
    if publish_store and url.startswith("sqlite"):
        publish_store_copy(repo, out / "data" / "store.db")
    log.info("site_built", out=str(out), empty=payload["empty"], backtest=backtest is not None)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    from volforecast.config import get_settings

    p = argparse.ArgumentParser(prog="python -m volforecast.site", description=__doc__)
    p.add_argument("--out", type=Path, default=Path("_site"), help="output directory")
    p.add_argument("--database-url", default=None, help="forecast store (URL or sqlite path)")
    p.add_argument("--reports-dir", type=Path, default=None, help="default: <repo>/reports")
    p.add_argument("--assets", nargs="+", default=None)
    p.add_argument("--horizons", nargs="+", type=int, default=None)
    p.add_argument("--demo", action="store_true", help="mark the page as DEMO data (not live)")
    p.add_argument(
        "--publish-store",
        action="store_true",
        help="also write data/store.db (restores the track record when the CI cache is lost)",
    )
    args = p.parse_args(argv)
    settings = get_settings()
    out = build_site(
        args.out,
        args.database_url or settings.database_url,
        reports_dir=args.reports_dir,
        assets=args.assets or settings.assets,
        horizons=args.horizons or settings.horizons,
        demo=args.demo,
        publish_store=args.publish_store,
    )
    print(f"site written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
