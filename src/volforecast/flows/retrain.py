"""Retrain flow: fit a challenger, log it to MLflow, run the promotion gate, move aliases.

For each (asset, horizon) this calls ``registry.training.retrain_one``: walk-forward all
candidates (``configs/production.yaml``), refit the best one on the full information set,
register it (``@challenger``), run the gate (``registry.promotion``) and, only when every check
passes, move ``@champion``. Every decision is written to ``promotion_decisions`` and to an MLflow
run. Re-running for the same as-of day is a no-op unless ``--force`` is given.

    uv run python -m volforecast.flows.retrain [--assets BTCUSDT] [--horizons 1]
        [--as-of 2026-01-31] [--force] [--only-if-triggered]

``--only-if-triggered`` retrains only the (asset, horizon) pairs requested by the latest
successful ``evaluate`` run (performance or data drift); without it every configured pair is
retrained (the weekly schedule).
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

from prefect import flow, task
from prefect.cache_policies import NO_CACHE

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
from volforecast.logging import get_logger
from volforecast.registry.training import RetrainOutcome, retrain_one

log = get_logger(__name__)


def requested_by_evaluate(rt: Runtime) -> dict[str, list[int]]:
    """``retrain_requests`` of the newest evaluate run (empty if it failed or none exists)."""
    runs = rt.repo.pipeline_runs("evaluate", limit=1)
    if not runs or runs[0]["status"] != "success":
        return {}
    detail = runs[0].get("detail") or {}
    return {
        str(a): [int(h) for h in hs] for a, hs in (detail.get("retrain_requests") or {}).items()
    }


def retrain_pair(
    rt: Runtime,
    asset: str,
    h: int,
    *,
    as_of: str | None = None,
    force: bool = False,
    reason: str = "scheduled",
) -> RetrainOutcome:
    """One (asset, horizon) retrain + gate; the panel is truncated at ``as_of`` if given."""
    panel, data_hash = load_panel(rt, asset)
    sync_realized(rt, asset, panel)
    if as_of is not None:
        panel = panel.loc[: last_complete_day(panel, as_of)]
        data_hash = f"{data_hash}@{as_of}"
    return retrain_one(
        asset=asset,
        h=h,
        panel=panel,
        data_hash=data_hash,
        cfg=rt.cfg,
        repo=rt.repo,
        client=rt.client,
        force=force,
        reason=reason,
    )


@task(name="retrain-pair", cache_policy=NO_CACHE, persist_result=False)
def retrain_pair_task(
    rt: Runtime, asset: str, h: int, as_of: str | None, force: bool, reason: str
) -> RetrainOutcome:
    return retrain_pair(rt, asset, h, as_of=as_of, force=force, reason=reason)


@flow(name="retrain", persist_result=False)
def retrain_flow(
    assets: list[str] | None = None,
    horizons: list[int] | None = None,
    as_of: str | None = None,
    force: bool = False,
    only_if_triggered: bool = False,
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
    with tracked_run(rt, "retrain", as_of, force=force, only_if_triggered=only_if_triggered) as rec:
        requests = requested_by_evaluate(rt) if only_if_triggered else {}
        reason = "triggered" if only_if_triggered else "scheduled"
        outcomes: list[RetrainOutcome] = []
        failures: list[str] = []
        for asset in rt.cfg.assets:
            for h in rt.cfg.horizons:
                if only_if_triggered and h not in requests.get(asset, []):
                    continue
                try:
                    outcomes.append(retrain_pair_task(rt, asset, h, as_of, force, reason))
                except Exception as exc:
                    log.error("retrain_failed", asset=asset, h=h, err=repr(exc))
                    failures.append(f"{asset} h={h}: {exc!r}")
        summary = [
            {k: v for k, v in asdict(o).items() if k != "candidate_qlike"}
            | {"candidate_qlike": o.candidate_qlike}
            for o in outcomes
        ]
        rec.detail.update(outcomes=summary, failures=failures, requests=requests)
        if only_if_triggered and not requests:
            rec.status = "skipped"
        if failures:
            raise RuntimeError("retrain failed for " + "; ".join(failures))
    return {"outcomes": summary}


def main(argv: Sequence[str] | None = None) -> int:
    p = base_parser("retrain", __doc__.splitlines()[0] if __doc__ else "")
    p.add_argument("--as-of", default=None, help="train on data up to this UTC day")
    p.add_argument("--force", action="store_true", help="re-run even if decided for this day")
    p.add_argument("--only-if-triggered", action="store_true")
    args = p.parse_args(argv)
    kw = common_kwargs(args)
    return run_cli(
        "retrain",
        lambda: retrain_flow(
            as_of=args.as_of, force=args.force, only_if_triggered=args.only_if_triggered, **kw
        ),
    )


if __name__ == "__main__":
    sys.exit(main())
