"""Evaluate flow: score stored forecasts against realized variance, monitor drift, decide retrain.

For each asset: sync the ``realized`` table from the panel, then (1) join realized RV to the
stored live forecasts and compute the rolling QLIKE vs the HAR benchmark and the 99% VaR breach
log (``monitoring.performance``), and (2) compute feature drift of the trailing window against
the champion's training window (``monitoring.drift``). Results go to ``monitoring_metrics`` and
to the run record. The flow *returns and records* a retrain decision:

* performance drift: the rolling-30d QLIKE ratio vs HAR is above 1 for 10 consecutive days
  (that horizon), or the 99% VaR Basel zone is yellow/red (all horizons of the asset);
* data drift: the share of drifted features reaches ``monitoring.drift_share_trigger``
  (all horizons of the asset).

    uv run python -m volforecast.flows.evaluate [--assets BTCUSDT]

The decision is consumed by ``retrain --only-if-triggered`` (the retrain CronJob also runs on a
schedule, see ADR-0007).
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from prefect import flow, task
from prefect.cache_policies import NO_CACHE

from volforecast.flows.common import (
    Runtime,
    base_parser,
    common_kwargs,
    load_panel,
    run_cli,
    runtime_from_kwargs,
    sync_realized,
    tracked_run,
)
from volforecast.logging import get_logger
from volforecast.monitoring.drift import DriftReport, compute_drift, write_drift_report
from volforecast.monitoring.performance import AssetEvaluation, evaluate_asset
from volforecast.registry.client import CHAMPION, alias_version, registered_model_name
from volforecast.registry.config import MonitoringConfig

log = get_logger(__name__)


@dataclass
class RetrainDecision:
    """Outcome of an evaluation: which (asset, horizon) pairs should be retrained and why."""

    retrain: bool = False
    #: asset -> horizons to retrain
    requests: dict[str, list[int]] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def decide_retrain(
    evaluations: Sequence[AssetEvaluation],
    drift: dict[str, DriftReport],
    cfg: MonitoringConfig,
    horizons: Sequence[int],
) -> RetrainDecision:
    """Combine the performance triggers and the data-drift trigger (pure function)."""
    out = RetrainDecision()
    for ev in evaluations:
        hs = set(ev.retrain_horizons)
        reasons = [f"{ev.asset}: {t}" for t in ev.triggers]
        rep = drift.get(ev.asset)
        trig = cfg.drift_share_trigger
        if (
            trig is not None
            and rep is not None
            and rep.status == "ok"
            and rep.share_drifted >= trig
        ):
            hs |= set(horizons)
            reasons.append(
                f"{ev.asset}: data drift, {rep.share_drifted:.0%} of features drifted "
                f"(trigger {trig:.0%})"
            )
        if hs:
            out.requests[ev.asset] = sorted(hs)
            out.reasons += reasons
    out.retrain = bool(out.requests)
    return out


def champion_train_end(rt: Runtime, asset: str) -> str | None:
    """``train_end_date`` tag of the first horizon that has a champion (the drift reference
    window end), or None if there is no champion (or the registry is unreachable)."""
    for h in rt.cfg.horizons:
        try:
            mv = alias_version(rt.client, registered_model_name(asset, h), CHAMPION)
        except Exception as exc:  # registry unreachable: drift falls back to "all earlier days"
            log.warning("champion_lookup_failed", asset=asset, err=repr(exc))
            return None
        if mv is not None and (mv.tags or {}).get("train_end_date"):
            return str(mv.tags["train_end_date"])
    return None


def evaluate_one(rt: Runtime, asset: str) -> tuple[AssetEvaluation, DriftReport]:
    """Performance evaluation + drift report of one asset (persisted to the store)."""
    panel, _ = load_panel(rt, asset)
    sync_realized(rt, asset, panel)
    mon = rt.cfg.monitoring
    ev = evaluate_asset(rt.repo, asset, list(rt.cfg.horizons), mon)
    real = rt.repo.get_realized(asset)
    report = compute_drift(
        real,
        asset,
        train_end=champion_train_end(rt, asset),
        current_window=mon.drift_window,
        psi_threshold=mon.psi_threshold,
        wasserstein_threshold=mon.wasserstein_threshold,
        n_bins=mon.psi_bins,
    )
    write_drift_report(report, rt.settings.drift_reports_dir)
    if report.status == "ok" and report.as_of_date:
        rt.repo.upsert_metrics(
            [
                {
                    "asset": asset,
                    "metric": "drift_share",
                    "as_of_date": report.as_of_date,
                    "value": report.share_drifted,
                }
            ]
        )
    ev.summary["drift"] = {
        "status": report.status,
        "share_drifted": report.share_drifted,
        "as_of_date": report.as_of_date,
        "n_reference": report.n_reference,
        "n_current": report.n_current,
        "drifted_features": [k for k, v in report.features.items() if v["drifted"]],
    }
    return ev, report


@task(name="evaluate-asset", cache_policy=NO_CACHE, persist_result=False)
def evaluate_asset_task(rt: Runtime, asset: str) -> tuple[AssetEvaluation, DriftReport]:
    return evaluate_one(rt, asset)


@flow(name="evaluate", persist_result=False)
def evaluate_flow(
    assets: list[str] | None = None,
    horizons: list[int] | None = None,
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
    with tracked_run(rt, "evaluate") as rec:
        evals: list[AssetEvaluation] = []
        drift: dict[str, DriftReport] = {}
        failures: list[str] = []
        for asset in rt.cfg.assets:
            try:
                ev, rep = evaluate_asset_task(rt, asset)
                evals.append(ev)
                drift[asset] = rep
            except Exception as exc:
                log.error("evaluate_asset_failed", asset=asset, err=repr(exc))
                failures.append(f"{asset}: {exc!r}")
        decision = decide_retrain(evals, drift, rt.cfg.monitoring, rt.cfg.horizons)
        rec.detail.update(
            summaries={e.asset: e.summary for e in evals},
            retrain=decision.retrain,
            retrain_requests=decision.requests,
            retrain_reasons=decision.reasons,
            failures=failures,
        )
        log.info("retrain_decision", **decision.to_dict())
        if failures:
            raise RuntimeError("evaluate failed for " + "; ".join(failures))
    return {
        "retrain_decision": decision.to_dict(),
        "summaries": {e.asset: e.summary for e in evals},
    }


def main(argv: Sequence[str] | None = None) -> int:
    p = base_parser("evaluate", __doc__.splitlines()[0] if __doc__ else "")
    args = p.parse_args(argv)
    return run_cli("evaluate", lambda: evaluate_flow(**common_kwargs(args)))


if __name__ == "__main__":
    sys.exit(main())
