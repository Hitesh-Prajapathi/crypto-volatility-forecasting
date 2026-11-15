"""Performance monitoring from the forecast store: rolling QLIKE vs HAR, VaR breach log, triggers.

Only *live* rows of the store are scored (served champion/fallback rows and the stored HAR
benchmark), never walk-forward backfills. Retrain triggers (plan P5.4):

* the champion's rolling-30-day QLIKE ratio vs HAR is > 1.0 on each of the last 10 days; or
* the 99% VaR breach count over the trailing 250 observations reaches the Basel yellow zone
  (only once a *full* 250-observation window exists: with tiny n a single breach is "red").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from volforecast.evaluation.losses import qlike
from volforecast.evaluation.var_backtest import basel_traffic_light
from volforecast.forecast_store import (
    ROLE_BENCHMARK,
    SERVED_ROLES,
    ForecastRepository,
    realized_horizon_sum,
)
from volforecast.logging import get_logger
from volforecast.registry.config import MonitoringConfig

log = get_logger(__name__)

CHAMPION_LABEL = "champion"
BENCHMARK_LABEL = "har_benchmark"


def _latest_per_origin(fc: pd.DataFrame) -> pd.DataFrame:
    if fc.empty:
        return fc
    out = fc.sort_values(["origin_date", "created_at"]).drop_duplicates("origin_date", keep="last")
    return out.reset_index(drop=True)


def qlike_by_origin(fc: pd.DataFrame, real: pd.DataFrame, h: int) -> pd.Series:
    """QLIKE per origin date for rows whose h-day target is realized (index = origin_date)."""
    if fc.empty or real.empty:
        return pd.Series(dtype=float)
    y = realized_horizon_sum(real, fc["origin_date"], h)
    f = fc["forecast_var"].to_numpy(float)
    ok = np.isfinite(y) & np.isfinite(f) & (y > 0) & (f > 0)
    return pd.Series(
        qlike(y[ok], f[ok]), index=pd.DatetimeIndex(fc["origin_date"][ok]), dtype=float
    )


def rolling_frame(champion: pd.Series, har: pd.Series, window: int) -> pd.DataFrame:
    """Rolling mean QLIKE of both models on common origins, plus their ratio.

    Rows exist only once ``window`` common scored origins have accumulated.
    """
    both = pd.concat({"champion": champion, "har": har}, axis=1, join="inner").sort_index()
    roll = both.rolling(window, min_periods=window).mean().dropna()
    roll["ratio"] = roll["champion"] / roll["har"]
    return roll


def ratio_trigger(ratio: pd.Series, threshold: float, days: int) -> bool:
    """True when the last ``days`` rolling ratios all exceed ``threshold``."""
    if len(ratio) < days:
        return False
    return bool((ratio.iloc[-days:] > threshold).all())


@dataclass
class BreachState:
    n: int
    breaches: int
    window: int
    zone: str | None
    fired: bool


def breach_trigger(hits: pd.Series, window: int, coverage: float) -> BreachState:
    """Basel zone of the breach count over the trailing ``window`` observations."""
    h = hits.to_numpy(dtype=bool)[-window:]
    n, k = int(h.size), int(h.sum())
    if n < window:  # not enough history for the traffic light to mean anything
        return BreachState(n, k, window, None, False)
    zone = basel_traffic_light(k, n, coverage)
    return BreachState(n, k, window, zone, zone != "green")


def hit_series(fc: pd.DataFrame, real: pd.DataFrame, column: str) -> pd.Series:
    """Breach indicator per *target* day: ``ret_{t+1} < -VaR_t`` (index = target date)."""
    if fc.empty or real.empty or column not in fc:
        return pd.Series(dtype=bool)
    sub = fc.dropna(subset=[column])
    nxt = pd.DatetimeIndex(sub["origin_date"]) + pd.Timedelta(days=1)
    ret = real["ret"].reindex(nxt).to_numpy(float)
    ok = np.isfinite(ret)
    var = sub[column].to_numpy(float)[ok]
    return pd.Series(ret[ok] < -var, index=nxt[ok], dtype=bool)


@dataclass
class AssetEvaluation:
    asset: str
    retrain_horizons: list[int] = field(default_factory=list)
    triggers: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


def evaluate_asset(
    repo: ForecastRepository,
    asset: str,
    horizons: list[int],
    cfg: MonitoringConfig,
) -> AssetEvaluation:
    """Score stored forecasts of ``asset``, persist monitoring metrics and decide triggers."""
    out = AssetEvaluation(asset=asset)
    real = repo.get_realized(asset)
    for h in horizons:
        served = _latest_per_origin(repo.forecasts_df(asset, h, roles=SERVED_ROLES))
        bench = _latest_per_origin(repo.forecasts_df(asset, h, roles=(ROLE_BENCHMARK,)))
        ql_c, ql_b = qlike_by_origin(served, real, h), qlike_by_origin(bench, real, h)
        info: dict[str, Any] = {"n_scored_champion": len(ql_c), "n_scored_har": len(ql_b)}
        roll = rolling_frame(ql_c, ql_b, cfg.rolling_window)
        rows: list[dict[str, Any]] = []
        base = {"asset": asset, "horizon": h, "window": cfg.rolling_window}
        for ts, r in roll.iterrows():
            d = pd.Timestamp(str(ts)).date()
            rows += [
                {**base, "metric": "rolling_qlike", "model": CHAMPION_LABEL, "as_of_date": d,
                 "value": r["champion"]},
                {**base, "metric": "rolling_qlike", "model": BENCHMARK_LABEL, "as_of_date": d,
                 "value": r["har"]},
                {**base, "metric": "qlike_ratio_vs_har", "as_of_date": d, "value": r["ratio"]},
            ]  # fmt: skip
        if not roll.empty:
            info["rolling_ratio_last"] = float(roll["ratio"].iloc[-1])
            if ratio_trigger(roll["ratio"], cfg.ratio_threshold, cfg.ratio_consecutive_days):
                out.retrain_horizons.append(h)
                out.triggers.append(
                    f"h={h}: rolling-{cfg.rolling_window}d QLIKE ratio vs HAR > "
                    f"{cfg.ratio_threshold} for {cfg.ratio_consecutive_days} consecutive days"
                )
        if h == 1:
            for level, col in (("99", "var99"), ("95", "var95")):
                hits = hit_series(served, real, col)
                rows += [
                    {"asset": asset, "model": CHAMPION_LABEL, "metric": f"var{level}_hit",
                     "as_of_date": pd.Timestamp(str(ts)).date(), "value": float(v)}
                    for ts, v in hits.items()
                ]  # fmt: skip
                if level == "99":
                    st = breach_trigger(hits, cfg.breach_window, cfg.breach_coverage)
                    info["var99_breaches_window"] = st.breaches
                    info["var99_obs"] = st.n
                    info["basel_zone"] = st.zone
                    if st.fired:
                        out.triggers.append(
                            f"99% VaR breaches {st.breaches}/{st.n} -> Basel {st.zone} zone"
                        )
                        for hh in horizons:
                            if hh not in out.retrain_horizons:
                                out.retrain_horizons.append(hh)
        repo.upsert_metrics(rows)
        out.summary[f"h{h}"] = info
    out.retrain_horizons.sort()
    log.info("asset_evaluated", asset=asset, summary=out.summary, triggers=out.triggers)
    return out


def as_date(x: Any) -> date:
    return pd.Timestamp(x).date()
