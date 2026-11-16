"""Kaggle runner: LSTM challenger seed backtests on a GPU (see scripts/kaggle/README.md).

For each seed k it runs the walk-forward backtest of ``configs/backtest_lstm.yaml`` over the
configured assets with ``params.seed = k`` and ``params.device = cuda`` (one run per seed because
model names must be unique), and writes ``<out>/backtest_lstm_s<k>/{forecasts.parquet,
run_meta.json, panels/}`` - the layout ``python -m volforecast report --run`` expects.

Inputs (private Kaggle dataset, found by globbing, or passed explicitly):

* a ``src/volforecast`` tree (the repo code, a clean ``git archive`` subset);
* precomputed daily panels ``daily/symbol=<S>/panel.parquet`` + ``panel.meta.json`` (public
  Binance bars, already aggregated; no raw klines are uploaded).

It also works locally (``--device cpu``) for smoke tests.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# import name -> pip requirement, for what Kaggle images may lack (torch is NEVER reinstalled)
NEEDED = {
    "structlog": "structlog>=24.1",
    "pydantic_settings": "pydantic-settings>=2.2",
    "pandera": "pandera[pandas]>=0.24",
    "arch": "arch>=7.0",
    "duckdb": "duckdb>=1.0",
    "yaml": "pyyaml>=6.0",
    "statsmodels": "statsmodels>=0.14",
    "lightgbm": "lightgbm>=4.3",
    "xgboost": "xgboost>=2.0",
    "sklearn": "scikit-learn>=1.4",
    "joblib": "joblib>=1.3",
    "pyarrow": "pyarrow>=15",
    "scipy": "scipy>=1.12",
    "matplotlib": "matplotlib>=3.8",
    "pydantic": "pydantic>=2.6",
    "httpx": "httpx>=0.27",
}


def _find_code(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    hits = sorted(Path("/kaggle/input").glob("**/src/volforecast/__init__.py"))
    if not hits:
        raise SystemExit("repo code not found under /kaggle/input (pass --code)")
    return hits[0].parents[2]


def _find_data(explicit: Path | None) -> Path:
    """Return a data root holding ``daily/symbol=<S>/panel.{parquet,meta.json}``."""
    if explicit is not None:
        return explicit
    metas = sorted(Path("/kaggle/input").glob("**/panel.meta.json"))
    if not metas:
        raise SystemExit("daily panels not found under /kaggle/input (pass --data)")
    # rebuild the hive layout in a scratch dir in case the upload flattened directory names
    root = Path("/kaggle/working/_data")
    for meta in metas:
        sym = meta.parent.name.removeprefix("symbol=")
        dst = root / "daily" / f"symbol={sym}"
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy(meta, dst / "panel.meta.json")
        shutil.copy(meta.with_name("panel.parquet"), dst / "panel.parquet")
    return root


def _ensure_deps() -> None:
    missing = []
    for mod, req in NEEDED.items():
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(req)
    if missing:
        print(f"[setup] pip install {missing}", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)


def _parse_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--code", type=Path, default=None, help="repo root (has src/volforecast)")
    ap.add_argument("--data", type=Path, default=None, help="data root (has daily/)")
    ap.add_argument("--out", type=Path, default=Path("/kaggle/working"))
    ap.add_argument(
        "--config", type=Path, default=None, help="default: <code>/configs/backtest_lstm.yaml"
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--assets", nargs="+", default=None, help="default: the config's assets")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--n-jobs", type=int, default=None, help="override config n_jobs")
    ap.add_argument(
        "--xgb-seeds",
        type=int,
        nargs="*",
        default=None,
        help="seeds whose run keeps the (seed-independent, CPU) xgb baseline; default all. "
        "har/log_har_dow are always kept (the report needs them).",
    )
    ap.add_argument("--last-origin", default=None, help="smoke tests only")
    ap.add_argument(
        "--lstm-param",
        action="append",
        default=[],
        metavar="KEY=JSON",
        help="override (smoke tests)",
    )
    ap.add_argument("--skip-install", action="store_true")
    ap.add_argument("--force", action="store_true", help="rerun seeds that already have output")
    args = ap.parse_args(argv)

    code = _find_code(args.code)
    data = _find_data(args.data)
    sys.path.insert(0, str(code / "src"))
    # loky workers are fresh interpreters: give them the path and the cuBLAS determinism setting
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(code / "src"), *filter(None, [os.environ.get("PYTHONPATH")])]
    )
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    if not args.skip_install:
        _ensure_deps()

    import torch  # after deps (lstm.py itself imports lightgbm/xgboost before torch)

    from volforecast.pipeline.backtest import run_backtest
    from volforecast.pipeline.config import BacktestConfig, load_config
    from volforecast.pipeline.panel import load_cached_panel

    env: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_requested": args.device,
    }
    if torch.cuda.is_available():
        env["gpu_count"] = torch.cuda.device_count()
        env["gpu_name"] = torch.cuda.get_device_name()
    print("[env]", json.dumps(env, indent=2), flush=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda but CUDA is not available: refusing to fall back to CPU")
    if args.device == "cuda":
        # warm the GPU so the first fit does not pay context creation in its timing
        torch.zeros(1, device="cuda").add_(1).cpu()

    cfg_path = args.config or (code / "configs" / "backtest_lstm.yaml")
    base = load_config(cfg_path).model_dump()
    if args.last_origin:
        base["last_origin"] = args.last_origin
    if args.n_jobs is not None:
        base["n_jobs"] = args.n_jobs
    overrides = {k: _parse_value(v) for k, _, v in (s.partition("=") for s in args.lstm_param)}
    xgb_seeds = set(args.seeds) if args.xgb_seeds is None else set(args.xgb_seeds)
    code_sha = (
        (code / "CODE_SHA").read_text().strip() if (code / "CODE_SHA").exists() else "unknown"
    )

    args.out.mkdir(parents=True, exist_ok=True)
    timings: dict[str, Any] = {"env": env, "code_sha": code_sha, "seeds": {}}
    t_all = time.perf_counter()
    for seed in args.seeds:
        out = args.out / f"backtest_lstm_s{seed}"
        if (
            not args.force
            and (out / "forecasts.parquet").exists()
            and (out / "run_meta.json").exists()
        ):
            print(f"[seed {seed}] already done, skipping", flush=True)
            continue
        raw = json.loads(json.dumps(base, default=str))  # deep copy
        models = []
        for m in raw["models"]:
            if m["name"] == "xgb" and seed not in xgb_seeds:
                continue
            if m["name"] == "lstm":
                m["params"] = {**m["params"], **overrides, "seed": seed, "device": args.device}
            models.append(m)
        raw["models"] = models
        cfg = BacktestConfig.model_validate(raw)
        print(
            f"[seed {seed}] models={[m.name for m in cfg.models]} device={args.device}", flush=True
        )
        t0 = time.perf_counter()
        run_backtest(cfg, data, out, assets=args.assets, panel_builder=load_cached_panel)
        wall = round(time.perf_counter() - t0, 1)
        meta_path = out / "run_meta.json"
        meta = json.loads(meta_path.read_text())
        meta["kaggle_run"] = {**env, "seed": seed, "wall_seconds": wall, "code_sha": code_sha}
        meta_path.write_text(json.dumps(meta, indent=2, default=str))
        lstm_secs = {
            k: round(v["seconds"], 1)
            for a in meta["run_stats"].values()
            for k, v in a.items()
            if k.startswith("lstm")
        }
        timings["seeds"][str(seed)] = {
            "wall_seconds": wall,
            "n_failures": meta["n_failures"],
            "timings_seconds": meta["timings_seconds"],
            "lstm_seconds": lstm_secs,
        }
        timings["total_seconds"] = round(time.perf_counter() - t_all, 1)
        (args.out / "timings.json").write_text(json.dumps(timings, indent=2))
        print(
            f"[seed {seed}] done in {wall}s, failures={meta['n_failures']}, lstm={lstm_secs}",
            flush=True,
        )
    shutil.rmtree(args.out / "_data", ignore_errors=True)
    print(f"[all] {round(time.perf_counter() - t_all, 1)}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
