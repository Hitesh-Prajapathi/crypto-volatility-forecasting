"""Feature drift (ADR-0011: in-house, numpy/scipy, no Evidently).

Per feature on the log realized measures (log rv, log bpv, log rs_pos, log rs_neg, sqrt rq,
|ret|): the Population Stability Index with quantile bins taken from the *reference* sample
(the training window) and the 1-D Wasserstein distance between reference and the *current*
sample (trailing 30 days). The report holds the share of drifted features.

Detector design (sensitivity fix). PSI is a chi-square-like statistic: under *no* drift its mean
is about ``(B - 1) * (1/n_cur + 1/n_ref)`` for ``B`` bins. With the textbook ``B = 10`` and
``n_cur = 30`` that is ~0.33, above the conventional 0.2-0.25 threshold, so a stationary series
was flagged on most features (measured: 83% of features on one seed, P(share >= 0.5) = 0.84 over
200 seeds). Two changes make the rule usable for a 30-day window:

* ``B = 5`` quantile bins taken from the reference (about 6 current observations per bin; null
  mean ~0.14), and
* a feature is "drifted" only when PSI > 0.25 (the conventional "significant shift" level) **and**
  the Wasserstein distance normalized by the reference standard deviation exceeds 0.5 (half a
  standard deviation, Cohen's "medium" effect). PSI alone is noisy with few observations; the
  effect-size guard demands a practically relevant move of the location/scale as well.

On i.i.d. log-normal data this gives ~1% flagged features (P(share >= 0.5) ~ 1% over 300 seeds)
and flags ~80% of features when log-RV shifts by 0.75 reference standard deviations. The six
features are all functions of the same intraday returns and are strongly correlated, so
``share_drifted`` behaves like a noisy single indicator rather than six independent tests.

Data drift alone is a weak signal in crypto (regimes change); the retrain decision combines it
with the performance monitor in :mod:`volforecast.monitoring.performance` (see flows.evaluate).
"""

from __future__ import annotations

import html
import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats

from volforecast.contracts import COL_COMPLETE
from volforecast.forecast_store.repository import clean_json
from volforecast.logging import get_logger

log = get_logger(__name__)

DEFAULT_PSI_THRESHOLD = 0.25
DEFAULT_WASSERSTEIN_THRESHOLD = 0.5
DEFAULT_N_BINS = 5
PSI_EPS = 1e-4  # empty-bin floor so the log ratio is finite
FEATURES: tuple[str, ...] = (
    "log_rv",
    "log_bpv",
    "log_rs_pos",
    "log_rs_neg",
    "sqrt_rq",
    "abs_ret",
)
MIN_REFERENCE = 60
MIN_CURRENT = 10
_FLOOR = 1e-12

FloatArray = npt.NDArray[np.float64]


def drift_features(realized: pd.DataFrame) -> pd.DataFrame:
    """Drift features from a realized-measures frame (columns rv, bpv, rs_pos, rs_neg, rq, ret)."""
    f = pd.DataFrame(index=realized.index)
    f["log_rv"] = np.log(realized["rv"].clip(lower=_FLOOR))
    f["log_bpv"] = np.log(realized["bpv"].clip(lower=_FLOOR))
    f["log_rs_pos"] = np.log(realized["rs_pos"].clip(lower=_FLOOR))
    f["log_rs_neg"] = np.log(realized["rs_neg"].clip(lower=_FLOOR))
    f["sqrt_rq"] = np.sqrt(realized["rq"].clip(lower=0.0))
    f["abs_ret"] = realized["ret"].abs()
    return f


def psi(reference: npt.ArrayLike, current: npt.ArrayLike, n_bins: int = DEFAULT_N_BINS) -> float:
    """Population Stability Index ``sum_i (q_i - p_i) ln(q_i / p_i)``.

    Bin edges are the ``n_bins`` quantiles of ``reference`` (duplicates merged, outer bins open),
    ``p`` / ``q`` the reference / current bin proportions (floored at ``PSI_EPS``). A bin is
    closed on the right: ``(e_{i-1}, e_i]``.
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref, cur = ref[np.isfinite(ref)], cur[np.isfinite(cur)]
    if ref.size == 0 or cur.size == 0:
        return float("nan")
    inner = np.unique(np.quantile(ref, np.linspace(0.0, 1.0, n_bins + 1))[1:-1])
    p = np.bincount(np.searchsorted(inner, ref, side="left"), minlength=inner.size + 1) / ref.size
    q = np.bincount(np.searchsorted(inner, cur, side="left"), minlength=inner.size + 1) / cur.size
    p, q = np.clip(p, PSI_EPS, None), np.clip(q, PSI_EPS, None)
    return float(np.sum((q - p) * np.log(q / p)))


def wasserstein(reference: npt.ArrayLike, current: npt.ArrayLike) -> float:
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref, cur = ref[np.isfinite(ref)], cur[np.isfinite(cur)]
    if ref.size == 0 or cur.size == 0:
        return float("nan")
    return float(stats.wasserstein_distance(ref, cur))


@dataclass
class DriftReport:
    asset: str
    as_of_date: str
    train_end_date: str | None
    n_reference: int
    n_current: int
    psi_threshold: float
    wasserstein_threshold: float
    n_bins: int
    status: str  # ok | insufficient_data
    features: dict[str, dict[str, float | bool]] = field(default_factory=dict)
    share_drifted: float = float("nan")

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = clean_json(asdict(self))
        return out


def compute_drift(
    realized: pd.DataFrame,
    asset: str,
    *,
    train_end: date | str | pd.Timestamp | None = None,
    current_window: int = 30,
    psi_threshold: float = DEFAULT_PSI_THRESHOLD,
    wasserstein_threshold: float = DEFAULT_WASSERSTEIN_THRESHOLD,
    n_bins: int = DEFAULT_N_BINS,
) -> DriftReport:
    """Drift of the trailing ``current_window`` complete days vs the training window.

    The reference is every complete day up to ``train_end`` (and before the current window);
    with ``train_end=None`` it is every complete day before the current window. A feature is
    drifted when ``PSI > psi_threshold`` and ``W / std(reference) > wasserstein_threshold``.
    """
    real = realized[realized[COL_COMPLETE].astype(bool)] if COL_COMPLETE in realized else realized
    real = real.dropna(subset=["rv", "bpv", "rs_pos", "rs_neg", "rq", "ret"])
    feats = drift_features(real)
    as_of = "" if feats.empty else str(feats.index[-1].date())
    cur = feats.iloc[-current_window:]
    cutoff = cur.index[0] - pd.Timedelta(days=1) if len(cur) else None
    if train_end is not None and cutoff is not None:
        cutoff = min(cutoff, pd.Timestamp(train_end))
    ref = feats.loc[:cutoff] if cutoff is not None else feats.iloc[:0]
    report = DriftReport(
        asset=asset,
        as_of_date=as_of,
        train_end_date=None if train_end is None else str(pd.Timestamp(train_end).date()),
        n_reference=len(ref),
        n_current=len(cur),
        psi_threshold=psi_threshold,
        wasserstein_threshold=wasserstein_threshold,
        n_bins=n_bins,
        status="ok",
    )
    if len(ref) < MIN_REFERENCE or len(cur) < MIN_CURRENT:
        report.status = "insufficient_data"
        log.warning("drift_insufficient_data", asset=asset, n_ref=len(ref), n_cur=len(cur))
        return report
    for name in FEATURES:
        r, c = ref[name].to_numpy(float), cur[name].to_numpy(float)
        value = psi(r, c, n_bins)
        w = wasserstein(r, c)
        sd = float(np.std(r))
        w_norm = w / sd if sd > 0 else float("nan")
        report.features[name] = {
            "psi": value,
            "wasserstein": w,
            "wasserstein_norm": w_norm,
            "drifted": bool(value > psi_threshold and w_norm > wasserstein_threshold),
        }
    report.share_drifted = float(np.mean([bool(f["drifted"]) for f in report.features.values()]))
    return report


def _html(report: DriftReport) -> str:
    rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{v['psi']:.4f}</td><td>{v['wasserstein']:.4f}</td>"
        f"<td>{v['wasserstein_norm']:.3f}</td><td>{'DRIFT' if v['drifted'] else 'ok'}</td></tr>"
        for name, v in report.features.items()
    )
    share = "n/a" if not np.isfinite(report.share_drifted) else f"{report.share_drifted:.0%}"
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>Drift {html.escape(report.asset)} {report.as_of_date}</title>"
        "<style>body{font-family:sans-serif}td,th{border:1px solid #ccc;padding:4px 10px;"
        "text-align:right}th{background:#eee}</style></head><body>"
        f"<h2>Feature drift: {html.escape(report.asset)} as of {report.as_of_date}</h2>"
        f"<p>status: {report.status}; reference n={report.n_reference} (up to "
        f"{report.train_end_date or 'start of current window'}), current n={report.n_current}; "
        f"drifted features (PSI &gt; {report.psi_threshold} and W/std &gt; "
        f"{report.wasserstein_threshold}): {share}</p>"
        "<table><tr><th>feature</th><th>PSI</th><th>Wasserstein</th><th>W / ref std</th>"
        f"<th>flag</th></tr>{rows}</table></body></html>"
    )


def write_drift_report(report: DriftReport, base_dir: str | Path) -> tuple[Path, Path]:
    """Write ``<base_dir>/<asset>/<date>.{json,html}``; upload to the active MLflow run if any."""
    out_dir = Path(base_dir) / report.asset
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = report.as_of_date or "unknown"
    json_path, html_path = out_dir / f"{stem}.json", out_dir / f"{stem}.html"
    json_path.write_text(json.dumps(report.to_dict(), indent=2))
    html_path.write_text(_html(report))
    if mlflow.active_run() is not None:
        for p in (json_path, html_path):
            mlflow.log_artifact(str(p), artifact_path=f"drift/{report.asset}")
    return json_path, html_path
