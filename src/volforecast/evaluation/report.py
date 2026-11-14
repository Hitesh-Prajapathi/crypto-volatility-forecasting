"""Results report: tables (markdown + JSON) and figures from ``forecasts.parquet``.

For every asset x horizon the models are compared on the COMMON valid origins (any origin where
a compared model's forecast or the realized variance is NaN is dropped; ``n`` is reported).
Metrics: mean QLIKE, QLIKE ratio vs the benchmark (HAR), one-sided Diebold-Mariano p-value for
"model beats benchmark" (HLN-corrected, h-aware), MSE, Mincer-Zarnowitz R^2 and Wald p, and
90% Model Confidence Set membership. Sub-period tables split by calendar year of the origin and
by terciles of the trailing 22-day RV measured AT the origin (information-set safe; the tercile
cut points themselves are taken over the evaluated sample, a purely descriptive grouping).
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from volforecast.contracts import ANNUALIZATION_DAYS, COL_RV
from volforecast.evaluation.losses import loss_frame
from volforecast.evaluation.risk import build_var_forecasts, var_backtest_table
from volforecast.evaluation.stats_tests import (
    diebold_mariano,
    mincer_zarnowitz,
    model_confidence_set,
)
from volforecast.logging import get_logger
from volforecast.pipeline.config import BacktestConfig

log = get_logger(__name__)

README_START = "<!-- RESULTS:START -->"
README_END = "<!-- RESULTS:END -->"
REGIME_LABELS = ("low", "mid", "high")
REGIME_WINDOW = 22
MIN_EVAL_OBS = 30
SHOWCASE_YEAR = 2022


# ── alignment ────────────────────────────────────────────────────────────────
def align_models(
    forecasts: pd.DataFrame, asset: str, h: int, models: list[str]
) -> tuple[pd.Series, pd.DataFrame, list[str]]:
    """Realized variance and forecast matrix on common valid origins.

    Models with no finite forecast at all are excluded (and returned in the third element);
    every other model must be valid, and so must ``realized_var``, at each retained origin.
    """
    sub = forecasts[(forecasts["asset"] == asset) & (forecasts["horizon"] == h)]
    sub = sub[sub["model"].isin(models)]
    if sub.empty:
        return pd.Series(dtype=float), pd.DataFrame(), list(models)
    fc = sub.pivot(index="origin_date", columns="model", values="forecast_var").sort_index()
    rv = sub.groupby("origin_date")["realized_var"].first().reindex(fc.index)
    present = [m for m in models if m in fc.columns and np.isfinite(fc[m]).any()]
    excluded = [m for m in models if m not in present]
    fc = fc[present]
    ok = np.isfinite(rv) & (rv > 0) & fc.gt(0).all(axis=1) & np.isfinite(fc).all(axis=1)
    return rv[ok].astype(float), fc[ok].astype(float), excluded


# ── metrics ──────────────────────────────────────────────────────────────────
def _nan_to_none(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _nan_to_none(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_nan_to_none(v) for v in x]
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, pd.Timestamp):
        return str(x.date())
    return x


def _ratio_vs(loss: pd.DataFrame, benchmark: str) -> pd.Series:
    return loss.mean() / float(loss[benchmark].mean())


def evaluate_block(
    rv: pd.Series, fc: pd.DataFrame, h: int, benchmark: str, mcs_reps: int, seed: int
) -> dict[str, Any]:
    """All headline metrics for one asset x horizon (inputs already aligned)."""
    qlike = loss_frame(rv, fc, "qlike")
    mse = loss_frame(rv, fc, "mse")
    mae = loss_frame(rv, fc, "mae_vol")
    ratio = _ratio_vs(qlike, benchmark)
    models: dict[str, dict[str, Any]] = {}
    for m in fc.columns:
        rec: dict[str, Any] = {
            "qlike": float(qlike[m].mean()),
            "qlike_ratio": float(ratio[m]),
            "mse": float(mse[m].mean()),
            "mae_vol": float(mae[m].mean()),
            "dm_p": None,
            "dm_stat": None,
            "mz_r2": None,
            "mz_wald_p": None,
            "mz_a": None,
            "mz_b": None,
        }
        if m != benchmark:
            dm = diebold_mariano(qlike[m], qlike[benchmark], h=h, alternative="less")
            rec["dm_p"], rec["dm_stat"] = dm.pvalue, dm.statistic
        mz = mincer_zarnowitz(rv.to_numpy(), fc[m].to_numpy())
        rec.update(mz_r2=mz["r2"], mz_wald_p=mz["wald_pvalue"], mz_a=mz["a"], mz_b=mz["b"])
        models[m] = rec
    mcs: dict[str, Any] | None = None
    try:
        res = model_confidence_set(qlike, size=0.10, reps=mcs_reps, seed=seed)
        mcs = {
            "size": 0.10,
            "reps": mcs_reps,
            "seed": seed,
            "included": res["included"],
            "excluded": res["excluded"],
            "pvalues": {k: float(v) for k, v in res["pvalues"].items()},
        }
        for m in fc.columns:
            models[m]["in_mcs90"] = m in set(res["included"])
            models[m]["mcs_p"] = mcs["pvalues"][m]
    except Exception as exc:  # MCS can fail on degenerate loss matrices; report, don't crash
        log.warning("mcs_failed", h=h, err=repr(exc))
        for m in fc.columns:
            models[m]["in_mcs90"] = None
            models[m]["mcs_p"] = None
    return {"models": models, "mcs": mcs}


def subperiod_table(rv: pd.Series, fc: pd.DataFrame, benchmark: str) -> dict[str, Any]:
    """QLIKE ratio vs benchmark by calendar year of the origin."""
    qlike = loss_frame(rv, fc, "qlike")
    years = pd.DatetimeIndex(qlike.index).year
    ratios: dict[str, dict[str, float]] = {}
    counts: dict[str, int] = {}
    for y in sorted(set(years)):
        block = qlike[years == y]
        counts[str(y)] = len(block)
        ratios[str(y)] = {str(k): float(v) for k, v in _ratio_vs(block, benchmark).items()}
    return {"ratio": ratios, "n": counts}


def trailing_rv(panel: pd.DataFrame) -> pd.Series:
    """Trailing 22-day mean RV at each date, using rows dated <= that date only."""
    return panel[COL_RV].rolling(REGIME_WINDOW, min_periods=REGIME_WINDOW).mean()


def regime_table(
    rv: pd.Series, fc: pd.DataFrame, panel: pd.DataFrame, benchmark: str
) -> dict[str, Any]:
    """QLIKE ratio vs benchmark by tercile of the trailing 22-day RV at the origin."""
    level = trailing_rv(panel).reindex(rv.index)
    valid = level.notna()
    if valid.sum() < MIN_EVAL_OBS:
        return {"ratio": {}, "n": {}, "cutpoints": []}
    cuts = level[valid].quantile([1 / 3, 2 / 3]).to_numpy()
    bucket = pd.Series(np.digitize(level[valid], cuts), index=level[valid].index)
    qlike = loss_frame(rv, fc, "qlike").loc[bucket.index]
    ratios: dict[str, dict[str, float]] = {}
    counts: dict[str, int] = {}
    for i, label in enumerate(REGIME_LABELS):
        block = qlike[bucket.to_numpy() == i]
        counts[label] = len(block)
        if len(block) >= 2 and float(block[benchmark].mean()) > 0:
            ratios[label] = {str(k): float(v) for k, v in _ratio_vs(block, benchmark).items()}
    return {"ratio": ratios, "n": counts, "cutpoints": [float(c) for c in cuts]}


# ── figures ──────────────────────────────────────────────────────────────────
def _plt() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _ann_vol(var: pd.Series | np.ndarray, h: int) -> Any:
    return np.sqrt(np.asarray(var, dtype=float) / h * ANNUALIZATION_DAYS)


def figure_forecast_vs_realized(
    rv: pd.Series, fc: pd.DataFrame, asset: str, benchmark: str, best: str, out: Path
) -> Path | None:
    """Annualized forecast vs realized vol over the 2022 LUNA/FTX year (h=1)."""
    sel = pd.DatetimeIndex(rv.index).year == SHOWCASE_YEAR
    if sel.sum() < 10:
        log.warning("figure_skipped_no_2022", asset=asset)
        return None
    plt = _plt()
    x = pd.DatetimeIndex(
        rv.index[sel].to_numpy() + np.timedelta64(1, "D")
    )  # forecast made at t covers day t+1
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(x, _ann_vol(rv[sel], 1), color="#9aa0a6", lw=1.0, label="realized (5-min RV)")
    ax.plot(x, _ann_vol(fc.loc[sel, benchmark], 1), color="#1f77b4", lw=1.2, label=benchmark)
    if best != benchmark:
        ax.plot(x, _ann_vol(fc.loc[sel, best], 1), color="#d62728", lw=1.2, label=best)
    for date, label in (("2022-05-09", "LUNA"), ("2022-11-08", "FTX")):
        ax.axvline(pd.Timestamp(date), color="k", ls=":", lw=0.8)
        ax.annotate(label, (pd.Timestamp(date), ax.get_ylim()[1]), fontsize=8, va="top")
    ax.set_ylabel("annualized volatility")
    ax.set_title(f"{asset}: 1-day-ahead volatility forecast vs realized, {SHOWCASE_YEAR}")
    ax.legend(frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0, -0.1))
    fig.tight_layout()
    path = out / f"forecast_vs_realized_{asset}_h1_{SHOWCASE_YEAR}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def figure_cum_qlike_diff(
    blocks: Mapping[int, tuple[pd.Series, pd.DataFrame]], asset: str, benchmark: str, out: Path
) -> Path | None:
    """Cumulative QLIKE(model) - QLIKE(benchmark) over time; below zero = ahead of HAR."""
    if not blocks:
        return None
    plt = _plt()
    fig, axes = plt.subplots(len(blocks), 1, figsize=(10, 3.6 * len(blocks)), squeeze=False)
    for ax, (h, (rv, fc)) in zip(axes[:, 0], sorted(blocks.items()), strict=True):
        qlike = loss_frame(rv, fc, "qlike")
        diff = qlike.sub(qlike[benchmark], axis=0).drop(columns=[benchmark]).cumsum()
        for m in diff.columns:
            ax.plot(diff.index, diff[m], lw=1.1, label=m)
        ax.axhline(0.0, color="k", lw=0.7)
        ax.set_title(
            f"{asset} h={h}: cumulative QLIKE difference vs {benchmark} (below 0 = better)"
        )
        ax.set_ylabel("cumulative QLIKE diff")
        ax.legend(frameon=False, ncol=5, fontsize=7, loc="upper left")
    fig.tight_layout()
    path = out / f"cum_qlike_diff_{asset}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ── markdown ─────────────────────────────────────────────────────────────────
def _fmt(x: Any, spec: str = ".4f") -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "-"
    return format(x, spec)


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _main_table(block: dict[str, Any], benchmark: str) -> str:
    rows = []
    for m, r in block["models"].items():
        mcs = r.get("in_mcs90")
        rows.append(
            [
                f"{m} (benchmark)" if m == benchmark else m,
                _fmt(r["qlike"], ".4f"),
                _fmt(r["qlike_ratio"], ".3f"),
                "-" if m == benchmark else _fmt(r["dm_p"], ".3f"),
                _fmt(r["mse"], ".3e"),
                _fmt(r["mz_r2"], ".3f"),
                _fmt(r["mz_wald_p"], ".3f"),
                "-" if mcs is None else ("yes" if mcs else "no"),
            ]
        )
    return md_table(
        [
            "model",
            "QLIKE",
            "ratio vs HAR",
            "DM p (beats HAR)",
            "MSE",
            "MZ R2",
            "MZ Wald p",
            "MCS 90%",
        ],
        rows,
    )


def _split_table(split: dict[str, Any], label: str) -> str:
    cols = list(split["ratio"].keys())
    if not cols:
        return "_not enough observations_"
    models = list(next(iter(split["ratio"].values())).keys())
    headers = ["model", *[f"{c} (n={split['n'][c]})" for c in cols]]
    rows = [[m, *[_fmt(split["ratio"][c].get(m), ".3f") for c in cols]] for m in models]
    return f"QLIKE ratio vs HAR by {label}\n\n" + md_table(headers, rows)


def _var_tables(var_tab: pd.DataFrame) -> str:
    out: list[str] = []
    for asset in var_tab["asset"].unique():
        for alpha in sorted(var_tab["alpha"].unique()):
            g = var_tab[(var_tab["asset"] == asset) & (var_tab["alpha"] == alpha)]
            if g.empty:
                continue
            rows = [
                [
                    str(r.model),
                    str(r.dist),
                    str(int(r.n_obs)),
                    f"{int(r.breaches)} / {r.expected_breaches:.1f}",
                    _fmt(100 * r.breach_rate, ".2f"),
                    _fmt(r.kupiec_p, ".3f"),
                    _fmt(r.christoffersen_cc_p, ".3f"),
                    str(r.traffic_light) if isinstance(r.traffic_light, str) else "-",
                    _fmt(r.es_z2, ".3f"),
                    _fmt(100 * r.frac_fallback, ".0f"),
                ]
                for r in g.itertuples()
            ]
            out.append(
                f"#### {asset}, VaR/ES level {100 * alpha:g}%\n\n"
                + md_table(
                    [
                        "model",
                        "dist",
                        "n",
                        "breaches / expected",
                        "rate %",
                        "Kupiec p",
                        "CC p",
                        "Basel (last 250)",
                        "ES Z2",
                        "normal-fallback %",
                    ],
                    rows,
                )
            )
    return "\n\n".join(out)


def render_markdown(results: dict[str, Any], detail: bool = True) -> str:
    """Markdown for ``reports/tables.md`` (``detail=False``: headline tables for the README)."""
    bm = results["benchmark"]
    parts = ["# Out-of-sample results", ""]
    meta = results["meta"]
    parts.append(
        f"Walk-forward, {meta['window']} window, first origin {meta['first_origin']}, "
        f"min_train {meta['min_train']}; benchmark {bm.upper()}; QLIKE on variance (lower is "
        f"better); ratio < 1 beats {bm.upper()}. Models are compared on common valid origins "
        "(n shown per table). DM: one-sided HLN-corrected Diebold-Mariano p-value for "
        f"'model beats {bm.upper()}'. MCS: Hansen-Lunde-Nason model confidence set at the 90% "
        f"level ({meta['mcs_reps']} bootstrap reps, seed {meta['seed']})."
    )
    for asset, ablock in results["assets"].items():
        parts += ["", f"## {asset}"]
        for h, block in ablock["horizons"].items():
            parts += [
                "",
                f"### h={h} (n={block['n']}, origins {block['start']} to {block['end']})",
                "",
                _main_table(block, bm),
            ]
            if block["excluded_models"]:
                parts += [
                    "",
                    f"Excluded (no valid forecasts): {', '.join(block['excluded_models'])}",
                ]
            if detail:
                parts += ["", _split_table(block["subperiod"], "calendar year"), ""]
                parts += [_split_table(block["regime"], "trailing-22d RV tercile at origin")]
    if detail and results.get("var"):
        parts += ["", "## 1-day VaR / ES backtest (h=1)", ""]
        parts.append(
            "Zero-mean; t and FHS use standardized residuals from strictly earlier origins "
            f"(expanding), falling back to normal until {results['var_min_resid']} are "
            "available. Basel zone is for the last 250 observations (1% level only)."
        )
        parts += ["", results["var_markdown"]]
    return "\n".join(parts).rstrip() + "\n"


# ── README injection ─────────────────────────────────────────────────────────
def inject_readme(readme_path: str | Path, tables_md: str) -> None:
    """Replace the text between the RESULTS markers in ``readme_path`` with ``tables_md``."""
    path = Path(readme_path)
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(re.escape(README_START) + r".*?" + re.escape(README_END), re.DOTALL)
    if not pattern.search(text):
        raise ValueError(f"{path}: markers {README_START} / {README_END} not found")
    replacement = f"{README_START}\n{tables_md.strip()}\n{README_END}"
    path.write_text(pattern.sub(lambda _: replacement, text, count=1), encoding="utf-8")


# ── orchestration ────────────────────────────────────────────────────────────
def build_report(
    forecasts: pd.DataFrame,
    panels: Mapping[str, pd.DataFrame],
    cfg: BacktestConfig,
    out_dir: str | Path,
    run_meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write ``results.json``, ``tables.md`` and figures under ``out_dir``; return the results."""
    out = Path(out_dir)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    bm = cfg.benchmark
    assets: dict[str, Any] = {}
    figures: list[str] = []

    for asset in [a for a in cfg.assets if a in set(forecasts["asset"])]:
        horizons: dict[str, Any] = {}
        aligned: dict[int, tuple[pd.Series, pd.DataFrame]] = {}
        for h in cfg.horizons:
            rv, fc, excluded = align_models(forecasts, asset, h, cfg.model_names)
            if bm not in fc.columns or len(rv) < MIN_EVAL_OBS:
                log.warning("block_skipped", asset=asset, h=h, n=len(rv))
                continue
            block = evaluate_block(rv, fc, h, bm, cfg.mcs_reps, cfg.seed)
            block.update(
                n=len(rv),
                start=str(rv.index[0].date()),
                end=str(rv.index[-1].date()),
                excluded_models=excluded,
                subperiod=subperiod_table(rv, fc, bm),
                regime=regime_table(rv, fc, panels[asset], bm),
            )
            horizons[str(h)] = block
            aligned[h] = (rv, fc)
        if not horizons:
            continue
        assets[asset] = {"horizons": horizons}
        p = figure_cum_qlike_diff(aligned, asset, bm, fig_dir)
        if p:
            figures.append(str(p.relative_to(out)))
        if 1 in aligned:
            rv1, fc1 = aligned[1]
            others = {m: v["qlike"] for m, v in horizons["1"]["models"].items() if m != bm}
            best = min(others, key=others.__getitem__) if others else bm
            assets[asset]["best_model_h1"] = best
            p = figure_forecast_vs_realized(rv1, fc1, asset, bm, best, fig_dir)
            if p:
                figures.append(str(p.relative_to(out)))

    results: dict[str, Any] = {
        "benchmark": bm,
        "meta": {
            "window": cfg.window,
            "first_origin": str(cfg.first_origin),
            "min_train": cfg.min_train,
            "horizons": cfg.horizons,
            "models": cfg.model_names,
            "mcs_reps": cfg.mcs_reps,
            "seed": cfg.seed,
            "run": dict(run_meta) if run_meta else None,
        },
        "assets": assets,
        "figures": figures,
        "var": [],
        "var_min_resid": cfg.var.min_resid,
        "var_markdown": "",
    }
    if 1 in cfg.horizons and assets:
        var_df = build_var_forecasts(
            forecasts, panels, cfg.var.alphas, cfg.var.dists, cfg.var.min_resid
        )
        if len(var_df):
            var_tab = var_backtest_table(var_df)
            results["var"] = var_tab.to_dict(orient="records")
            results["var_markdown"] = _var_tables(var_tab)

    clean = _nan_to_none(results)
    (out / "results.json").write_text(json.dumps(clean, indent=2))
    (out / "tables.md").write_text(render_markdown(results, detail=True), encoding="utf-8")
    (out / "tables_readme.md").write_text(render_markdown(results, detail=False), encoding="utf-8")
    log.info("report_written", out=str(out), assets=list(assets), figures=len(figures))
    return clean  # type: ignore[no-any-return]
