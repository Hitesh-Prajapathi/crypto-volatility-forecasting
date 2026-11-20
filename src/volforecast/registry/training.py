"""Retrain core: walk-forward the candidates, refit the best, register it, run the gate.

One call of :func:`retrain_one` handles one (asset, horizon). The flow in
``volforecast.flows.retrain`` is a thin loop over this function.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient

from volforecast.contracts import COL_COMPLETE, VolModel
from volforecast.evaluation.losses import qlike
from volforecast.evaluation.walkforward import ModelSpec, WalkForwardConfig, run_walkforward
from volforecast.forecast_store.repository import ForecastRepository
from volforecast.logging import get_logger
from volforecast.models.ensemble import InverseLossEnsemble
from volforecast.models.factory import build_model, is_stateful_model, list_models
from volforecast.pipeline.backtest import git_info
from volforecast.registry.client import (
    CHALLENGER,
    CHAMPION,
    alias_version,
    registered_model_name,
    set_alias,
    tag_version,
)
from volforecast.registry.config import CandidateConfig, ProductionConfig
from volforecast.registry.promotion import GateInputs, GateResult, apply_decision, evaluate_gate
from volforecast.registry.pyfunc import log_volforecast_model
from volforecast.registry.tracking import log_backtest_run

log = get_logger(__name__)


@dataclass
class RetrainOutcome:
    asset: str
    horizon: int
    status: str  # promoted | rejected | skipped | failed
    as_of_date: str | None = None
    model_name: str | None = None
    challenger_version: str | None = None
    champion_version: str | None = None
    candidate_qlike: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    mlflow_run_id: str | None = None
    error: str | None = None


def resolve_candidates(cfg: ProductionConfig) -> list[CandidateConfig]:
    """Static candidates plus any optional model present in the factory *at runtime*.

    Optional ones (``lgbm``, ``xgb``, ``ens_*`` ... by prefix) are probed with a default
    construction; ones that fail to build are skipped with a warning.
    """
    out = list(cfg.retrain.candidates)
    have = {c.name for c in out}
    for name in list_models():
        if name in have or not any(name.startswith(p) for p in cfg.retrain.optional_prefixes):
            continue
        try:
            build_model(name)
        except Exception as exc:
            log.warning("optional_candidate_skipped", model=name, err=repr(exc))
            continue
        out.append(CandidateConfig(name=name, refit_every=cfg.retrain.optional_refit_every))
    return out


def last_complete_date(panel: pd.DataFrame) -> pd.Timestamp:
    ok = panel.index[panel[COL_COMPLETE].astype(bool).to_numpy()]
    if len(ok) == 0:
        raise ValueError("panel has no complete day")
    return pd.Timestamp(ok[-1])


def walkforward_candidate_with_model(
    panel: pd.DataFrame,
    asset: str,
    h: int,
    cand: CandidateConfig,
    cfg: ProductionConfig,
    data_hash: str,
) -> tuple[pd.DataFrame, VolModel | None]:
    """Walk-forward over the last ``wf_days`` origins, plus the walked-forward instance.

    The instance is returned only for stateful models (``reuse_instance``: one object lives
    through the whole walk-forward, so it holds the out-of-sample history the gate scored); it is
    None for all others, whose engine instances are rebuilt at every refit.
    """
    n = len(panel)
    first = panel.index[max(cfg.retrain.min_train - 1, n - cfg.retrain.wf_days)]
    wf = WalkForwardConfig(first_origin=first, horizons=(h,), min_train=cfg.retrain.min_train)
    stateful = is_stateful_model(cand.name)
    built: list[VolModel] = []

    def builder() -> VolModel:
        model = build_model(cand.name, **cand.params)
        built.append(model)
        return model

    spec = ModelSpec(
        name=cand.name,
        builder=builder,
        refit_every=cand.refit_every,
        # stateful models (ens_inv) keep their OOS forecast history inside the instance
        reuse_instance=stateful,
    )
    fc, _ = run_walkforward(panel, spec, wf, h, asset, data_hash)
    return fc, (built[0] if stateful and built else None)


def walkforward_candidate(
    panel: pd.DataFrame,
    asset: str,
    h: int,
    cand: CandidateConfig,
    cfg: ProductionConfig,
    data_hash: str,
) -> pd.DataFrame:
    """Walk-forward over the last ``wf_days`` origins (FORECAST_COLUMNS table)."""
    return walkforward_candidate_with_model(panel, asset, h, cand, cfg, data_hash)[0]


def fit_final_model(
    best: CandidateConfig, panel: pd.DataFrame, h: int, walked: VolModel | None
) -> VolModel:
    """The model that gets logged: ``best`` fitted on the full information set.

    A stateful model continues its walk-forward instance (``fit`` on the full history is exactly
    one more engine refit at ``as_of``), so the artifact carries the out-of-sample history its
    weights were learned from. Building a fresh instance here would serve equal weights.
    """
    if is_stateful_model(best.name):
        if walked is None:
            raise RuntimeError(f"{best.name}: no walked-forward instance to warm-start from")
        return walked.fit(panel, h)
    return build_model(best.name, **best.params).fit(panel, h)


def warm_start_metadata(model: VolModel) -> dict[str, Any]:
    """Artifact metadata describing the persisted learned state ({} for stateless models)."""
    if isinstance(model, InverseLossEnsemble):
        return {"warm_start": model.warm_start_summary()}
    return {}


def _gate_frame(fc: pd.DataFrame) -> pd.DataFrame:
    return fc[["origin_date", "forecast_var", "realized_var"]].reset_index(drop=True)


def _mean_qlike(fc: pd.DataFrame, origins: pd.DatetimeIndex) -> float:
    f = fc.set_index(pd.DatetimeIndex(fc["origin_date"]))
    sub = f.reindex(origins)
    y, p = sub["realized_var"].to_numpy(float), sub["forecast_var"].to_numpy(float)
    ok = np.isfinite(y) & np.isfinite(p) & (y > 0) & (p > 0)
    if ok.sum() < len(origins):  # any invalid forecast disqualifies the candidate
        return float("inf")
    return float(qlike(y, p).mean())


def champion_spec(client: MlflowClient, asset: str, h: int) -> tuple[str, CandidateConfig] | None:
    """(version, spec) of the current @champion from its version tags, or None."""
    mv = alias_version(client, registered_model_name(asset, h), CHAMPION)
    if mv is None:
        return None
    tags = mv.tags or {}
    name = tags.get("model_name")
    if not name or name not in list_models():
        return None
    params = json.loads(tags.get("model_params", "{}"))
    return str(mv.version), CandidateConfig(
        name=name, params=params, refit_every=int(tags.get("refit_every", 5))
    )


def retrain_one(
    *,
    asset: str,
    h: int,
    panel: pd.DataFrame,
    data_hash: str,
    cfg: ProductionConfig,
    repo: ForecastRepository,
    client: MlflowClient,
    force: bool = False,
    reason: str = "scheduled",
) -> RetrainOutcome:
    """Walk-forward candidates -> refit best -> register -> ``@challenger`` -> promotion gate."""
    as_of = last_complete_date(panel)
    out = RetrainOutcome(asset=asset, horizon=h, status="skipped", as_of_date=str(as_of.date()))
    if not force and repo.has_decision(asset, h, as_of.date()):
        log.info("retrain_skipped_idempotent", asset=asset, h=h, as_of=out.as_of_date)
        return out
    pnl = panel.loc[:as_of]
    name = registered_model_name(asset, h)
    cands = resolve_candidates(cfg)
    bench = next((c for c in cands if c.name == cfg.benchmark), None) or CandidateConfig(
        name=cfg.benchmark
    )
    champ = champion_spec(client, asset, h)
    specs: dict[str, CandidateConfig] = {c.name: c for c in cands}
    specs.setdefault(bench.name, bench)

    def key(c: CandidateConfig) -> str:
        return f"{c.name}|{json.dumps(c.params, sort_keys=True)}"

    with mlflow.start_run(run_name=f"retrain-{asset}-h{h}-{as_of.date()}") as run:
        out.mlflow_run_id = str(run.info.run_id)
        mlflow.set_tags({"asset": asset, "horizon": str(h), "trigger": reason, "kind": "retrain"})
        results: dict[str, pd.DataFrame] = {}
        walked: dict[str, VolModel] = {}  # stateful candidates' walk-forward instances
        for cand in specs.values():
            try:
                results[key(cand)], inst = walkforward_candidate_with_model(
                    pnl, asset, h, cand, cfg, data_hash
                )
                if inst is not None:
                    walked[key(cand)] = inst
            except Exception as exc:
                log.warning("candidate_failed", asset=asset, h=h, model=cand.name, err=repr(exc))
        champ_fc: pd.DataFrame | None = None
        if champ is not None:
            ck = key(champ[1])
            if ck not in results:
                try:
                    results[ck] = walkforward_candidate(pnl, asset, h, champ[1], cfg, data_hash)
                except Exception as exc:
                    log.warning("champion_wf_failed", asset=asset, h=h, err=repr(exc))
            champ_fc = results.get(ck)
        bench_key = key(bench)
        if bench_key not in results:
            raise RuntimeError(f"benchmark {bench.name} walk-forward failed for {asset} h={h}")

        # common scored origins -> candidate ranking by mean QLIKE over the gate window
        scored = None
        for fc in results.values():
            ok = fc[np.isfinite(fc["realized_var"])]["origin_date"]
            scored = pd.DatetimeIndex(ok) if scored is None else scored.intersection(ok)
        assert scored is not None
        origins = scored.sort_values()[-cfg.promotion.window :]
        if len(origins) < cfg.promotion.min_eval_obs:
            raise RuntimeError(f"only {len(origins)} scored origins for {asset} h={h}")
        by_cand: dict[str, float] = {}
        for c in cands:
            if key(c) in results:
                by_cand[c.name] = _mean_qlike(results[key(c)], origins)
        by_cand[bench.name] = _mean_qlike(results[bench_key], origins)
        out.candidate_qlike = {k: v for k, v in by_cand.items() if np.isfinite(v)}
        mlflow.log_metrics({f"qlike.{k}": v for k, v in out.candidate_qlike.items()})
        if not out.candidate_qlike:
            raise RuntimeError(f"no valid candidate for {asset} h={h}")
        best_name = min(out.candidate_qlike, key=lambda k: out.candidate_qlike[k])
        best = specs[best_name]
        best_fc = results[key(best)]
        out.model_name = best_name
        mlflow.set_tags({"selected_model": best_name})

        all_fc = pd.concat(results.values(), ignore_index=True)
        log_backtest_run(
            all_fc,
            config={"asset": asset, "horizon": h, "candidates": [c.model_dump() for c in cands]},
            data_hashes={asset: data_hash},
            run_name=f"walkforward-{asset}-h{h}",
            tags={"parent": out.mlflow_run_id, "as_of": out.as_of_date or ""},
            nested=True,
        )

        model = fit_final_model(best, pnl, h, walked.get(key(best)))
        latest = float(model.predict_variance(pnl, h))
        oos = best_fc[np.isfinite(best_fc["forecast_var"])][["origin_date", "forecast_var"]]
        info = log_volforecast_model(
            model,
            asset=asset,
            h=h,
            model_name=best.name,
            panel=pnl,
            data_hash=data_hash,
            oos=oos,
            params=best.params,
            extra_metadata={
                "git_sha": git_info()["sha"],
                "trigger": reason,
                **warm_start_metadata(model),
            },
        )
        version = str(info.registered_model_version)
        out.challenger_version = version
        tag_version(
            client,
            name,
            version,
            {
                "model_name": best.name,
                "model_params": json.dumps(best.params, sort_keys=True),
                "refit_every": best.refit_every,
                "train_end_date": as_of.date(),
                "data_hash": data_hash,
                "git_sha": git_info()["sha"],
                "wf_qlike": out.candidate_qlike[best_name],
                "wf_window": len(origins),
            },
        )
        set_alias(client, name, CHALLENGER, version)

    gate_in = GateInputs(
        asset=asset,
        h=h,
        challenger_model=best.name,
        challenger=_gate_frame(best_fc),
        har=_gate_frame(results[bench_key]),
        panel=pnl,
        champion=None if champ_fc is None else _gate_frame(champ_fc),
        challenger_is_benchmark=best.name == bench.name,
        latest_forecast=latest,
    )
    result: GateResult = evaluate_gate(gate_in, cfg.promotion)
    decision: dict[str, Any] = apply_decision(
        repo=repo,
        client=client,
        asset=asset,
        h=h,
        as_of_date=as_of.date(),
        challenger_version=version,
        challenger_model=best.name,
        result=result,
        cfg=cfg.promotion,
        extra_tags={"trigger": reason},
    )
    out.status = str(decision["decision"])
    out.reasons = list(decision["reasons"])
    out.champion_version = decision["champion_version"]
    out.mlflow_run_id = decision["mlflow_run_id"]
    return out
