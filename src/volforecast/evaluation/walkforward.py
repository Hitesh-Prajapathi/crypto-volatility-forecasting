"""Walk-forward (out-of-sample) forecasting engine.

At every forecast origin t the model sees ONLY ``panel`` rows dated <= t. Models are
refit every ``refit_every`` origins; between refits ``predict_variance`` filters the new
rows with the last fitted parameters. The output is a long table with
``contracts.FORECAST_COLUMNS``, which drives all downstream evaluation, VaR and reporting.

Realized targets are attached for scoring only (never shown to the model): the realized
h-day variance at origin t is sum_{k=1..h} RV_{t+k}. It is NaN if any of those days is
incomplete (ADR-0003) or not yet observed.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from volforecast.contracts import COL_COMPLETE, COL_RV, FORECAST_COLUMNS, VolModel
from volforecast.logging import get_logger

log = get_logger(__name__)

ModelBuilder = Callable[[], VolModel]


@dataclass(frozen=True)
class WalkForwardConfig:
    """Walk-forward protocol. All dates are panel index labels (tz-naive UTC days)."""

    first_origin: pd.Timestamp
    last_origin: pd.Timestamp | None = None
    horizons: tuple[int, ...] = (1, 5)
    refit_every: int = 1
    window: Literal["expanding", "rolling"] = "expanding"
    rolling_size: int | None = None
    min_train: int = 250

    def __post_init__(self) -> None:
        if self.refit_every < 1:
            raise ValueError("refit_every must be >= 1")
        if self.window == "rolling" and not self.rolling_size:
            raise ValueError("rolling window requires rolling_size")
        if any(h < 1 for h in self.horizons):
            raise ValueError("horizons must be >= 1")


@dataclass(frozen=True)
class ModelSpec:
    """A named model to evaluate. `builder` must return a fresh, unfitted model."""

    name: str
    builder: ModelBuilder
    refit_every: int | None = None  # overrides WalkForwardConfig.refit_every
    version: str = "dev"


@dataclass
class RunStats:
    n_origins: int = 0
    n_fits: int = 0
    n_fit_failures: int = 0
    n_predict_failures: int = 0
    seconds: float = 0.0
    failures: list[str] = field(default_factory=list)


def realized_targets(panel: pd.DataFrame, h: int) -> pd.Series:
    """sum_{k=1..h} RV_{t+k}, NaN if any of those days is incomplete or unobserved."""
    rv = panel[COL_RV].astype(float)
    ok = panel[COL_COMPLETE].astype(bool)
    total = pd.Series(0.0, index=panel.index)
    all_ok = pd.Series(True, index=panel.index)
    for k in range(1, h + 1):
        total = total + rv.shift(-k)
        all_ok &= ok.shift(-k, fill_value=False).astype(bool)
    return total.where(all_ok)


def _origin_positions(panel: pd.DataFrame, cfg: WalkForwardConfig) -> np.ndarray:
    idx = panel.index
    if not idx.is_monotonic_increasing or not idx.is_unique:
        raise ValueError("panel index must be unique and increasing")
    start = int(idx.searchsorted(pd.Timestamp(cfg.first_origin), side="left"))
    end = (
        len(idx) - 1
        if cfg.last_origin is None
        else int(idx.searchsorted(pd.Timestamp(cfg.last_origin), side="right") - 1)
    )
    start = max(start, cfg.min_train - 1)
    if start > end:
        raise ValueError("no forecast origins in range (check first_origin/min_train)")
    return np.arange(start, end + 1)


def run_walkforward(
    panel: pd.DataFrame,
    spec: ModelSpec,
    cfg: WalkForwardConfig,
    h: int,
    asset: str,
    data_hash: str = "",
) -> tuple[pd.DataFrame, RunStats]:
    """Run one model at one horizon over all origins. Returns (forecasts, stats)."""
    refit_every = spec.refit_every or cfg.refit_every
    targets = realized_targets(panel, h).to_numpy()
    positions = _origin_positions(panel, cfg)
    stats = RunStats(n_origins=len(positions))
    model: VolModel | None = None
    fitted = False
    rows: list[tuple[object, ...]] = []
    t0 = time.perf_counter()

    for k, pos in enumerate(positions):
        lo = 0 if cfg.window == "expanding" else max(0, pos + 1 - int(cfg.rolling_size or 0))
        history = panel.iloc[lo : pos + 1]  # rows <= origin ONLY
        origin = panel.index[pos]
        if k % refit_every == 0 or not fitted:
            candidate = spec.builder()
            try:
                candidate.fit(history, h)
                model, fitted = candidate, True
                stats.n_fits += 1
            except Exception as exc:  # keep the previous fit, but record it
                stats.n_fit_failures += 1
                stats.failures.append(f"fit@{origin.date()}: {exc!r}")
                log.warning("fit_failed", model=spec.name, h=h, origin=str(origin), err=repr(exc))
        forecast = np.nan
        if fitted and model is not None:
            try:
                forecast = float(model.predict_variance(history, h))
                if not np.isfinite(forecast) or forecast <= 0:
                    raise ValueError(f"invalid forecast {forecast}")
            except Exception as exc:
                forecast = np.nan
                stats.n_predict_failures += 1
                stats.failures.append(f"predict@{origin.date()}: {exc!r}")
        rows.append(
            (
                asset,
                spec.name,
                h,
                origin,
                origin + pd.Timedelta(days=h),
                forecast,
                targets[pos],
                spec.version,
                data_hash,
            )
        )

    stats.seconds = time.perf_counter() - t0
    out = pd.DataFrame(rows, columns=list(FORECAST_COLUMNS))
    log.info(
        "walkforward_done",
        model=spec.name,
        h=h,
        asset=asset,
        origins=stats.n_origins,
        fits=stats.n_fits,
        fit_failures=stats.n_fit_failures,
        predict_failures=stats.n_predict_failures,
        seconds=round(stats.seconds, 2),
    )
    return out, stats


def run_many(
    panel: pd.DataFrame,
    specs: Sequence[ModelSpec],
    cfg: WalkForwardConfig,
    asset: str,
    data_hash: str = "",
    horizons: Iterable[int] | None = None,
    n_jobs: int = -1,
) -> tuple[pd.DataFrame, dict[tuple[str, int], RunStats]]:
    """Run every (model, horizon) pair in parallel; returns one long forecast table."""
    hs = tuple(horizons) if horizons is not None else cfg.horizons
    jobs = [(spec, h) for spec in specs for h in hs]
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(run_walkforward)(panel, spec, cfg, h, asset, data_hash) for spec, h in jobs
    )
    frames = [r[0] for r in results]
    stats = {(spec.name, h): r[1] for (spec, h), r in zip(jobs, results, strict=True)}
    return pd.concat(frames, ignore_index=True), stats
