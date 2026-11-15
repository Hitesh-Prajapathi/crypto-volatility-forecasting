"""Log walk-forward backtest runs to MLflow (config, git SHA, data hashes, metrics, artifacts)."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd

from volforecast.evaluation.losses import mse, qlike
from volforecast.forecast_store.repository import clean_json
from volforecast.pipeline.backtest import git_info


def summarize_forecasts(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Mean QLIKE / MSE and scored count per (asset, model, horizon) of a FORECAST_COLUMNS table."""
    rows: list[dict[str, Any]] = []
    for (asset, model, h), g in forecasts.groupby(["asset", "model", "horizon"], sort=True):
        f = g["forecast_var"].to_numpy(dtype=float)
        y = g["realized_var"].to_numpy(dtype=float)
        ok = np.isfinite(f) & np.isfinite(y) & (f > 0) & (y > 0)
        n = int(ok.sum())
        rows.append(
            {
                "asset": str(asset),
                "model": str(model),
                "horizon": int(str(h)),
                "n_origins": len(g),
                "n_scored": n,
                "qlike": float(qlike(y[ok], f[ok]).mean()) if n else float("nan"),
                "mse": float(mse(y[ok], f[ok]).mean()) if n else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _flat(prefix: str, d: Mapping[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, Mapping):
            out.update(_flat(key, v))
        else:
            out[key] = json.dumps(v, default=str) if isinstance(v, list | tuple) else str(v)
    return out


def log_backtest_run(
    forecasts: pd.DataFrame,
    *,
    config: Mapping[str, Any],
    data_hashes: Mapping[str, str],
    run_name: str = "backtest",
    tags: Mapping[str, str] | None = None,
    artifact_files: Sequence[str | Path] = (),
    artifact_dirs: Sequence[str | Path] = (),
    nested: bool = False,
) -> str:
    """Log one walk-forward run and return its MLflow run id.

    Logs: flattened ``config`` params, git SHA / dirty flag and data hashes as params+tags, one
    metric per (asset, model, horizon) ``<asset>.<model>.h<h>.{qlike,mse,n_scored}``, the summary
    table and ``forecasts.parquet`` as artifacts, plus any extra report files / directories.
    """
    summary = summarize_forecasts(forecasts)
    git = git_info()
    with mlflow.start_run(run_name=run_name, nested=nested) as run:
        mlflow.set_tags(
            {"git_sha": str(git["sha"]), "git_dirty": str(git["dirty"]), **(dict(tags or {}))}
        )
        params = _flat("config", clean_json(dict(config)))
        # MLflow limits param values to 500 chars (6000 on newer servers): truncate defensively.
        mlflow.log_params({k: v[:500] for k, v in params.items()})
        mlflow.log_params({f"data_hash.{a}": h for a, h in data_hashes.items()})
        for r in summary.to_dict("records"):
            base = f"{r['asset']}.{r['model']}.h{r['horizon']}"
            mlflow.log_metric(f"{base}.n_scored", float(r["n_scored"]))
            if np.isfinite(r["qlike"]):
                mlflow.log_metric(f"{base}.qlike", float(r["qlike"]))
                mlflow.log_metric(f"{base}.mse", float(r["mse"]))
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            forecasts.to_parquet(t / "forecasts.parquet", index=False)
            summary.to_csv(t / "summary.csv", index=False)
            mlflow.log_artifacts(str(t))
        for f in artifact_files:
            mlflow.log_artifact(str(f))
        for d in artifact_dirs:
            mlflow.log_artifacts(str(d), artifact_path=Path(d).name)
        return str(run.info.run_id)


def log_backtest_dir(out_dir: str | Path, run_name: str = "backtest") -> str:
    """Log the output directory of ``pipeline.backtest.run_backtest`` as one MLflow run."""
    out = Path(out_dir)
    meta = json.loads((out / "run_meta.json").read_text())
    forecasts = pd.read_parquet(out / "forecasts.parquet")
    files = [out / "run_meta.json"]
    files += sorted(p for p in out.glob("*.md")) + sorted(p for p in out.glob("*.json"))
    return log_backtest_run(
        forecasts,
        config=meta.get("config", {}),
        data_hashes=meta.get("data_hashes", {}),
        run_name=run_name,
        tags={"git_sha_run": str(meta.get("git", {}).get("sha", "unknown"))},
        artifact_files=sorted(set(files)),
    )
