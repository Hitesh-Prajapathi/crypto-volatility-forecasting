"""Champion/challenger promotion gate (plan P5.2, ADR-0006).

:func:`evaluate_gate` is a pure function (frames in, decision + reasons out). A challenger is
promoted only if ALL of these hold on the last ``window`` walk-forward origins:

(a) mean QLIKE(challenger) <= mean QLIKE(champion) and the Diebold-Mariano test "challenger is
    worse than champion" has p > ``dm_alpha`` (skipped when there is no champion);
(b) mean QLIKE ratio vs HAR < 1 - epsilon (vacuous when the challenger *is* the HAR benchmark and
    ``allow_benchmark_champion``);
(c) its 1-day 99% VaR (``var_dist``, default FHS) passes Kupiec and Christoffersen CC at the
    ``significance`` level and is in the Basel green zone (h=1 only: VaR is a 1-day measure);
(d) all forecasts are finite and positive (and the latest one is within ``sanity_factor`` of the
    trailing h-day RV range).

:func:`apply_decision` logs the decision as an MLflow run tagged ``promotion_decision``, writes the
``promotion_decisions`` row and (if promoted) moves the ``@champion`` alias, all-or-nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient

from volforecast.evaluation.losses import qlike
from volforecast.evaluation.risk import build_var_forecasts
from volforecast.evaluation.stats_tests import diebold_mariano
from volforecast.evaluation.var_backtest import (
    basel_traffic_light,
    christoffersen_cc,
    kupiec_pof,
)
from volforecast.forecast_store.repository import ForecastRepository, clean_json
from volforecast.logging import get_logger
from volforecast.registry.client import (
    CHALLENGER,
    CHAMPION,
    alias_version,
    drop_alias,
    registered_model_name,
    set_alias,
    tag_version,
)
from volforecast.registry.config import PromotionConfig

log = get_logger(__name__)


@dataclass
class Check:
    name: str
    status: str  # "pass" | "fail" | "skip"
    detail: str
    values: dict[str, Any] = field(default_factory=dict)


@dataclass
class GateResult:
    promoted: bool
    checks: list[Check]
    metrics: dict[str, Any]

    @property
    def reasons(self) -> list[str]:
        return [f"({c.name}) {c.status.upper()}: {c.detail}" for c in self.checks]


@dataclass
class GateInputs:
    """Walk-forward products on a common origin grid (frames: origin_date, forecast_var,
    realized_var; ``realized_var`` NaN while unobserved)."""

    asset: str
    h: int
    challenger_model: str
    challenger: pd.DataFrame
    har: pd.DataFrame
    panel: pd.DataFrame  # daily panel (needs ``ret``) used for the VaR backtest
    champion: pd.DataFrame | None = None
    challenger_is_benchmark: bool = False
    #: latest point forecast of the refit challenger (for the sanity check), if available
    latest_forecast: float | None = None


def _series(df: pd.DataFrame, col: str) -> pd.Series:
    out: pd.Series = df.set_index(pd.DatetimeIndex(df["origin_date"]))[col].astype(float)
    return out


def _valid(df: pd.DataFrame) -> pd.DataFrame:
    out: pd.DataFrame = df[np.isfinite(df["realized_var"]) & (df["realized_var"] > 0)]
    return out


def evaluate_gate(inp: GateInputs, cfg: PromotionConfig) -> GateResult:
    """Apply checks (a)-(d); never raises on bad data (that is a failed check)."""
    metrics: dict[str, Any] = {"asset": inp.asset, "horizon": inp.h, "window": cfg.window}
    checks: list[Check] = []

    # ---- (d) finite and positive (first: the other checks need valid numbers) ----------
    ch_all = inp.challenger
    realized_ok = _valid(ch_all)
    ch_window_origins = pd.DatetimeIndex(realized_ok["origin_date"]).sort_values()[-cfg.window :]
    f_ch = _series(ch_all, "forecast_var").reindex(ch_window_origins)
    bad = int((~np.isfinite(f_ch) | (f_ch <= 0)).sum())
    d_detail = f"{len(f_ch) - bad}/{len(f_ch)} window forecasts finite and positive"
    d_pass = bad == 0 and len(f_ch) > 0
    if inp.latest_forecast is not None:
        lf = inp.latest_forecast
        if not (np.isfinite(lf) and lf > 0):
            d_pass, d_detail = False, d_detail + f"; latest forecast {lf} invalid"
        else:
            roll = inp.panel["rv"].astype(float).rolling(inp.h).sum().dropna().iloc[-cfg.window :]
            lo, hi = float(roll.min()) / cfg.sanity_factor, float(roll.max()) * cfg.sanity_factor
            metrics["sanity_range"] = [lo, hi]
            if not lo <= lf <= hi:
                d_pass, d_detail = (
                    False,
                    d_detail + f"; latest forecast {lf:.3g} outside [{lo:.3g}, {hi:.3g}]",
                )
            else:
                d_detail += f"; latest forecast {lf:.3g} within 10x trailing RV range"
    checks.append(Check("d", "pass" if d_pass else "fail", d_detail))

    # ---- common evaluation grid ---------------------------------------------------------
    y = _series(ch_all, "realized_var")
    f_har = _series(inp.har, "forecast_var")
    grid = ch_window_origins.intersection(pd.DatetimeIndex(f_har.index))
    f_champ: pd.Series | None = None
    if inp.champion is not None:
        f_champ = _series(inp.champion, "forecast_var")
        grid = grid.intersection(pd.DatetimeIndex(f_champ.index))
    n_eval = len(grid)
    metrics["n_eval"] = n_eval
    if not d_pass or n_eval < cfg.min_eval_obs:
        if d_pass:
            checks.append(
                Check(
                    "a", "fail", f"only {n_eval} scored origins < min_eval_obs={cfg.min_eval_obs}"
                )
            )
        return GateResult(False, checks, metrics)
    metrics["window_start"] = str(grid[0].date())
    metrics["window_end"] = str(grid[-1].date())
    yy = y.reindex(grid).to_numpy()
    ql_ch = qlike(yy, f_ch.reindex(grid).to_numpy())
    ql_har = qlike(yy, f_har.reindex(grid).to_numpy())
    m_ch, m_har = float(ql_ch.mean()), float(ql_har.mean())
    metrics.update(qlike_challenger=m_ch, qlike_har=m_har)

    # ---- (a) vs champion ----------------------------------------------------------------
    if f_champ is None:
        checks.append(Check("a", "skip", "no champion yet"))
    else:
        ql_champ = qlike(yy, f_champ.reindex(grid).to_numpy())
        m_champ = float(ql_champ.mean())
        # H1: champion loss < challenger loss, i.e. the challenger is significantly worse
        dm = diebold_mariano(ql_champ, ql_ch, h=inp.h, alternative="less")
        p = float(dm.pvalue)
        p = 1.0 if not np.isfinite(p) else p
        metrics.update(qlike_champion=m_champ, dm_stat=dm.statistic, dm_p_challenger_worse=p)
        ok = m_ch <= m_champ and p > cfg.dm_alpha
        checks.append(
            Check(
                "a",
                "pass" if ok else "fail",
                f"QLIKE challenger {m_ch:.5f} vs champion {m_champ:.5f}; "
                f"DM p(challenger worse)={p:.3f} (need > {cfg.dm_alpha})",
            )
        )

    # ---- (b) vs HAR -----------------------------------------------------------------------
    ratio = m_ch / m_har if m_har > 0 else (1.0 if m_ch == 0 else float("inf"))
    metrics["qlike_ratio_vs_har"] = ratio
    if inp.challenger_is_benchmark and cfg.allow_benchmark_champion:
        checks.append(Check("b", "pass", "challenger is the HAR benchmark (ratio 1.0 by identity)"))
    else:
        ok = ratio < 1.0 - cfg.epsilon
        checks.append(
            Check(
                "b",
                "pass" if ok else "fail",
                f"QLIKE ratio vs HAR {ratio:.4f} (need < {1.0 - cfg.epsilon:.4f})",
            )
        )

    # ---- (c) 99% VaR ----------------------------------------------------------------------
    checks.append(_var_check(inp, cfg, grid, metrics))
    return GateResult(all(c.status != "fail" for c in checks), checks, metrics)


def _var_check(
    inp: GateInputs, cfg: PromotionConfig, grid: pd.DatetimeIndex, metrics: dict[str, Any]
) -> Check:
    if inp.h != 1:
        return Check("c", "skip", "VaR is a 1-day measure: only the h=1 model is VaR-tested")
    fc = inp.challenger[["origin_date", "forecast_var"]].copy()
    fc = fc[np.isfinite(fc["forecast_var"]) & (fc["forecast_var"] > 0)]
    fc["asset"], fc["model"], fc["horizon"] = inp.asset, inp.challenger_model, 1
    var_df = build_var_forecasts(
        fc,
        {inp.asset: inp.panel},
        alphas=(cfg.var_alpha,),
        dists=(cfg.var_dist,),
        min_resid=cfg.min_resid,
        align=False,
    )
    if var_df.empty:
        return Check("c", "fail", "no VaR rows could be built")
    var_df = var_df.set_index(pd.DatetimeIndex(var_df["origin_date"]))
    win = var_df.reindex(grid).dropna(subset=["hit"])
    if len(win) < cfg.min_eval_obs:
        return Check("c", "fail", f"only {len(win)} VaR observations in the window")
    share = float((win["dist_used"] == cfg.var_dist).mean())
    hits = win["hit"].to_numpy(dtype=bool)
    n, k = hits.size, int(hits.sum())
    kup = kupiec_pof(hits, cfg.var_alpha)
    cc = christoffersen_cc(hits, cfg.var_alpha)
    last = hits[-250:]
    zone = basel_traffic_light(int(last.sum()), last.size, 1.0 - cfg.var_alpha)
    metrics.update(
        var_breaches=k,
        var_n=n,
        var_breach_rate=k / n,
        kupiec_p=kup.pvalue,
        christoffersen_cc_p=cc.pvalue,
        basel_zone=zone,
        var_dist_share=share,
    )
    problems = []
    if share < cfg.min_dist_share:
        problems.append(
            f"only {share:.0%} of rows used {cfg.var_dist} (need {cfg.min_dist_share:.0%})"
        )
    if not kup.pvalue > cfg.significance:
        problems.append(f"Kupiec p={kup.pvalue:.3f}")
    if not cc.pvalue > cfg.significance:
        problems.append(f"Christoffersen CC p={cc.pvalue:.3f}")
    if zone != "green":
        problems.append(f"Basel zone {zone}")
    detail = (
        f"{cfg.var_dist} 99% VaR: {k}/{n} breaches, Kupiec p={kup.pvalue:.3f}, "
        f"CC p={cc.pvalue:.3f}, Basel {zone}"
    )
    return Check(
        "c",
        "fail" if problems else "pass",
        detail + (f" -> {'; '.join(problems)}" if problems else ""),
    )


# --------------------------------------------------------------------------------------------
# side effects
# --------------------------------------------------------------------------------------------
def apply_decision(
    *,
    repo: ForecastRepository,
    client: MlflowClient,
    asset: str,
    h: int,
    as_of_date: date | str,
    challenger_version: str,
    challenger_model: str,
    result: GateResult,
    cfg: PromotionConfig,
    extra_tags: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Log the decision (MLflow run + DB row) and move aliases if promoted.

    The DB row and the alias move share one transaction: if moving the alias fails, no
    decision row is committed (and the MLflow run is tagged ``failed``). ``@champion`` is moved
    with a single registry call; the old version is kept in the ``previous_champion`` tag so a
    rollback is one alias move.
    """
    name = registered_model_name(asset, h)
    prev = alias_version(client, name, CHAMPION)
    prev_version = None if prev is None else str(prev.version)
    decision = "promoted" if result.promoted else "rejected"
    with mlflow.start_run(run_name=f"promotion-{asset}-h{h}-{as_of_date}") as run:
        run_id = str(run.info.run_id)
        mlflow.set_tags(
            {
                "promotion_decision": decision,
                "asset": asset,
                "horizon": str(h),
                "registered_model": name,
                "challenger_version": challenger_version,
                "champion_version": prev_version or "none",
                **(extra_tags or {}),
            }
        )
        mlflow.log_params({"window": cfg.window, "epsilon": cfg.epsilon, "dm_alpha": cfg.dm_alpha})
        numeric = {
            k: float(v)
            for k, v in result.metrics.items()
            if isinstance(v, int | float | np.floating) and np.isfinite(v)
        }
        mlflow.log_metrics(numeric)
        mlflow.log_dict(
            {
                "decision": decision,
                "reasons": result.reasons,
                "metrics": clean_json(result.metrics),
            },
            "decision.json",
        )
        moved = False
        try:
            with repo.session() as s:
                repo.add_promotion_decision(
                    s,
                    asset=asset,
                    horizon=h,
                    as_of_date=as_of_date,
                    registered_model=name,
                    challenger_version=challenger_version,
                    challenger_model=challenger_model,
                    champion_version=prev_version,
                    promoted=result.promoted,
                    reasons=result.reasons,
                    metrics=result.metrics,
                    mlflow_run_id=run_id,
                )
                if result.promoted:
                    set_alias(client, name, CHAMPION, challenger_version)
                    moved = True
                    # (the commit happens when the session block exits)
        except Exception:
            if moved:  # restore the old pointer; the decision row was rolled back
                if prev_version is not None:
                    set_alias(client, name, CHAMPION, prev_version)
                else:
                    drop_alias(client, name, CHAMPION)
            mlflow.set_tag("promotion_decision", "failed")
            raise
    tags = {
        "promotion_decision": decision,
        "promotion_as_of": str(as_of_date),
        "promotion_run_id": run_id,
        "promotion_reasons": " | ".join(result.reasons)[:4000],
    }
    if result.promoted:
        tags["previous_champion"] = prev_version or "none"
        cur = alias_version(client, name, CHALLENGER)
        if cur is not None and str(cur.version) == challenger_version:
            drop_alias(client, name, CHALLENGER)
    tag_version(client, name, challenger_version, tags)
    log.info(
        "promotion_decision",
        asset=asset,
        h=h,
        decision=decision,
        challenger=challenger_version,
        champion=prev_version,
        reasons=result.reasons,
    )
    return {
        "promoted": result.promoted,
        "decision": decision,
        "reasons": result.reasons,
        "mlflow_run_id": run_id,
        "champion_version": challenger_version if result.promoted else prev_version,
    }
