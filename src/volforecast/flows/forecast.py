"""Forecast flow: batch inference of the champion into the forecast store (ADR-0009).

For each (asset, horizon): build the information set up to the latest complete UTC day, load the
registry ``@champion`` (``models:/volforecast-{asset}-h{h}@champion``; a ``fallback_model`` fit on
the full history when no champion exists, logged as a warning), forecast the h-day variance and,
for h=1, the 1-day 99%/95% VaR and 97.5% ES. The HAR benchmark is forecast next to it (it is the
denominator of the QLIKE ratio monitored by ``evaluate``), and a ``@challenger`` shadow row is
written when one is registered. Rows are upserted on ``(asset, horizon, model_version,
origin_date)``: re-running a day never duplicates.

    uv run python -m volforecast.flows.forecast [--assets BTCUSDT] [--as-of 2026-01-31]
        [--replay-days 60]

``--replay-days N`` forecasts each of the last N complete days with the (frozen) champion. It
exists to exercise the pipeline and to fill the store when bootstrapping; it is only an
out-of-sample test of the champion if the champion was trained before those days.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from prefect import flow, task
from prefect.cache_policies import NO_CACHE

from volforecast.contracts import COL_RET, VolModel
from volforecast.evaluation.var_backtest import standardized_es, standardized_quantile
from volforecast.flows.common import (
    Runtime,
    base_parser,
    common_kwargs,
    last_complete_day,
    load_panel,
    run_cli,
    runtime_from_kwargs,
    sync_realized,
    tracked_run,
)
from volforecast.forecast_store import (
    BENCHMARK_MODEL_NAME,
    BENCHMARK_VERSION,
    ROLE_BENCHMARK,
    ROLE_CHALLENGER,
    ROLE_CHAMPION,
    ROLE_FALLBACK,
    SERVED_ROLES,
)
from volforecast.logging import get_logger
from volforecast.models.factory import build_model
from volforecast.registry.client import (
    CHALLENGER,
    CHAMPION,
    alias_version,
    model_uri,
    registered_model_name,
)
from volforecast.registry.pyfunc import VolForecastPyfunc

log = get_logger(__name__)

VAR_ALPHAS = {"var99": 0.01, "var95": 0.05, "es975": 0.025}


@dataclass
class ServedModel:
    """A model that produces forecast rows (registry version or fallback)."""

    model: VolModel
    model_name: str
    version: str
    role: str
    oos: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["origin_date"]))


def load_registered(rt: Runtime, asset: str, h: int, alias: str, role: str) -> ServedModel | None:
    """The model behind ``@alias`` of ``volforecast-{asset}-h{h}``, or None if there is none."""
    name = registered_model_name(asset, h)
    mv = alias_version(rt.client, name, alias)
    if mv is None:
        return None
    loaded = mlflow.pyfunc.load_model(model_uri(asset, h, alias))
    inner = loaded.unwrap_python_model()
    assert isinstance(inner, VolForecastPyfunc)
    return ServedModel(
        model=inner.model,
        model_name=str(inner.metadata.get("model_name", "unknown")),
        version=str(mv.version),
        role=role,
        oos=inner.oos_forecasts,
    )


def fallback_model(rt: Runtime, h: int, panel: pd.DataFrame) -> ServedModel:
    """``fallback_model`` fit on the whole information set (used only without a champion)."""
    name = rt.cfg.fallback_model
    model = build_model(name).fit(panel, h)
    return ServedModel(model=model, model_name=name, version=f"fallback:{name}", role=ROLE_FALLBACK)


def standardized_residuals(oos: pd.DataFrame, panel: pd.DataFrame, origin: pd.Timestamp) -> Any:
    """``z_s = ret_{s+1} / sqrt(f_s)`` for stored forecasts whose next-day return is known at
    ``origin`` (``s + 1 <= origin``): the FHS residual history (same definition as
    ``evaluation.risk``)."""
    if oos.empty:
        return np.empty(0)
    s = pd.DatetimeIndex(pd.to_datetime(oos["origin_date"]))
    f = oos["forecast_var"].to_numpy(float)
    keep = (s + pd.Timedelta(days=1)) <= origin
    s, f = s[keep], f[keep]
    r = panel[COL_RET].reindex(s + pd.Timedelta(days=1)).to_numpy(float)
    ok = np.isfinite(r) & np.isfinite(f) & (f > 0)
    return r[ok] / np.sqrt(f[ok])


def risk_measures(
    f: float, z: np.ndarray[Any, Any], min_resid: int
) -> dict[str, float | str | None]:
    """1-day VaR/ES (positive loss magnitudes) from a variance forecast ``f``.

    Filtered historical simulation (empirical quantiles of ``z``) once ``min_resid`` residuals
    exist, otherwise the normal distribution (recorded in ``dist``). Note that ``f`` forecasts
    the 5-minute realized variance, not E[ret^2]; FHS corrects the scale, the normal does not
    (see ``evaluation.risk``).
    """
    use_fhs = z.size >= min_resid
    dist = "fhs" if use_fhs else "normal"
    kw: dict[str, Any] = {"std_resid": z} if use_fhs else {}
    sd = float(np.sqrt(f))
    return {
        "var99": -standardized_quantile(0.01, dist, **kw) * sd,  # type: ignore[arg-type]
        "var95": -standardized_quantile(0.05, dist, **kw) * sd,  # type: ignore[arg-type]
        "es975": standardized_es(0.025, dist, **kw) * sd,  # type: ignore[arg-type]
        "dist": dist,
    }


def _row(
    served: ServedModel,
    asset: str,
    h: int,
    origin: pd.Timestamp,
    f: float,
    data_hash: str,
    risk: dict[str, float | str | None] | None,
) -> dict[str, Any]:
    if not (np.isfinite(f) and f > 0):
        raise ValueError(f"{served.model_name} produced an invalid forecast {f} ({asset} h={h})")
    return {
        "asset": asset,
        "horizon": h,
        "model_name": served.model_name,
        "model_version": served.version,
        "role": served.role,
        "origin_date": origin.date(),
        "target_date": (origin + timedelta(days=h)).date(),
        "forecast_var": float(f),
        "data_hash": data_hash,
        **(risk or {}),
    }


def forecast_origins(
    rt: Runtime,
    asset: str,
    h: int,
    panel: pd.DataFrame,
    data_hash: str,
    origins: Sequence[pd.Timestamp],
) -> list[dict[str, Any]]:
    """Forecast rows for ``(asset, h)`` at each origin; models are loaded once."""
    served = load_registered(rt, asset, h, CHAMPION, ROLE_CHAMPION)
    if served is None:
        log.warning("no_champion_using_fallback", asset=asset, h=h, fallback=rt.cfg.fallback_model)
    shadow = load_registered(rt, asset, h, CHALLENGER, ROLE_CHALLENGER)
    live_oos: pd.DataFrame | None = None
    if served is not None and h == 1:
        live = rt.repo.forecasts_df(asset, h, roles=SERVED_ROLES, model_version=served.version)
        live_oos = live[["origin_date", "forecast_var"]]
    min_resid = rt.cfg.promotion.min_resid
    rows: list[dict[str, Any]] = []
    for origin in origins:
        pnl = panel.loc[:origin]
        cur = served or fallback_model(rt, h, pnl)  # fallback is refit on each information set
        risk = None
        if h == 1:
            oos = cur.oos
            if live_oos is not None and not live_oos.empty:
                oos = pd.concat([oos, live_oos], ignore_index=True).drop_duplicates(
                    "origin_date", keep="first"
                )
            z = standardized_residuals(oos, panel, origin)
            f_now = float(cur.model.predict_variance(pnl, h))
            risk = risk_measures(f_now, z, min_resid)
        else:
            f_now = float(cur.model.predict_variance(pnl, h))
        rows.append(_row(cur, asset, h, origin, f_now, data_hash, risk))
        if shadow is not None:
            rows.append(
                _row(shadow, asset, h, origin, float(shadow.model.predict_variance(pnl, h)),
                     data_hash, None)
            )  # fmt: skip
        bench = build_model(rt.cfg.benchmark).fit(pnl, h)
        bench_served = ServedModel(
            model=bench, model_name=BENCHMARK_MODEL_NAME, version=BENCHMARK_VERSION,
            role=ROLE_BENCHMARK,
        )  # fmt: skip
        rows.append(
            _row(bench_served, asset, h, origin, float(bench.predict_variance(pnl, h)),
                 data_hash, None)
        )  # fmt: skip
    return rows


def forecast_asset(
    rt: Runtime,
    asset: str,
    *,
    as_of: str | None = None,
    replay_days: int = 0,
) -> dict[str, Any]:
    """Forecast every configured horizon of ``asset``; writes to the store, returns a summary."""
    panel, data_hash = load_panel(rt, asset)
    sync_realized(rt, asset, panel)
    last = last_complete_day(panel, as_of)
    complete = panel.index[panel["complete"].astype(bool).to_numpy()]
    complete = complete[complete <= last]
    origins = list(complete[-max(1, replay_days) :])
    n_rows = 0
    latest: dict[str, Any] = {}
    for h in rt.cfg.horizons:
        rows = forecast_origins(rt, asset, h, panel, data_hash, origins)
        n_rows += rt.repo.upsert_forecasts(rows)
        served_now = [
            r for r in rows if r["role"] in SERVED_ROLES and r["origin_date"] == last.date()
        ]
        if served_now:
            r = served_now[0]
            latest[f"h{h}"] = {
                k: r.get(k)
                for k in ("model_name", "model_version", "role", "forecast_var", "var99", "dist")
            }
    return {
        "asset": asset,
        "origin_date": str(last.date()),
        "n_origins": len(origins),
        "rows": n_rows,
        "latest": latest,
        "data_hash": data_hash,
    }


@task(name="forecast-asset", cache_policy=NO_CACHE, persist_result=False)
def forecast_asset_task(
    rt: Runtime, asset: str, as_of: str | None, replay_days: int
) -> dict[str, Any]:
    return forecast_asset(rt, asset, as_of=as_of, replay_days=replay_days)


@flow(name="forecast", persist_result=False)
def forecast_flow(
    assets: list[str] | None = None,
    horizons: list[int] | None = None,
    as_of: str | None = None,
    replay_days: int = 0,
    data_dir: str | None = None,
    database_url: str | None = None,
    tracking_uri: str | None = None,
    config_path: str | None = None,
) -> dict[str, Any]:
    rt = runtime_from_kwargs(
        {
            "assets": assets,
            "horizons": horizons,
            "data_dir": data_dir,
            "database_url": database_url,
            "tracking_uri": tracking_uri,
            "config_path": config_path,
        }
    )
    with tracked_run(rt, "forecast", as_of, replay_days=replay_days) as rec:
        results: list[dict[str, Any]] = []
        failures: list[str] = []
        for asset in rt.cfg.assets:
            try:
                results.append(forecast_asset_task(rt, asset, as_of, replay_days))
            except Exception as exc:
                log.error("forecast_asset_failed", asset=asset, err=repr(exc))
                failures.append(f"{asset}: {exc!r}")
        rec.detail.update(results=results, failures=failures)
        if failures:
            raise RuntimeError("forecast failed for " + "; ".join(failures))
    return {"results": results}


def main(argv: Sequence[str] | None = None) -> int:
    p = base_parser("forecast", __doc__.splitlines()[0] if __doc__ else "")
    p.add_argument("--as-of", default=None, help="forecast as of this UTC day (default: latest)")
    p.add_argument("--replay-days", type=int, default=0, help="forecast the last N days")
    args = p.parse_args(argv)
    kw = common_kwargs(args)
    return run_cli(
        "forecast", lambda: forecast_flow(as_of=args.as_of, replay_days=args.replay_days, **kw)
    )


if __name__ == "__main__":
    sys.exit(main())
