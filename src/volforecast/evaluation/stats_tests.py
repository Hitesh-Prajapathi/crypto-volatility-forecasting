"""Forecast-comparison and forecast-quality tests.

* :func:`diebold_mariano` - equal predictive accuracy (Diebold & Mariano, 1995) with the
  Harvey-Leybourne-Newbold (1997) small-sample correction.
* :func:`mincer_zarnowitz` / :func:`mincer_zarnowitz_log` - forecast-optimality
  regressions with HAC inference (Mincer & Zarnowitz, 1969).
* :func:`model_confidence_set` - Hansen, Lunde & Nason (2011) MCS via ``arch``.

Everything is a pure function of arrays / Series.  Test outcomes use the shared
:class:`volforecast.contracts.TestResult`.

References
----------
Diebold, F. X. & Mariano, R. S. (1995). Comparing predictive accuracy. JBES 13(3).
Harvey, D., Leybourne, S. & Newbold, P. (1997). Testing the equality of prediction mean
    squared errors. International Journal of Forecasting 13(2), 281-291.
Mincer, J. & Zarnowitz, V. (1969). The evaluation of economic forecasts. NBER.
Newey, W. K. & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity
    and autocorrelation consistent covariance matrix. Econometrica 55(3), 703-708.
Hansen, P. R., Lunde, A. & Nason, J. M. (2011). The model confidence set. Econometrica
    79(2), 453-497.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
import statsmodels.api as sm
from arch.bootstrap import MCS
from scipy import stats

from volforecast.contracts import TestResult
from volforecast.evaluation.losses import FloatArray, validate_positive_finite
from volforecast.logging import get_logger

log = get_logger(__name__)

Alternative = Literal["less", "greater", "two-sided"]


def _finite_1d(x: npt.ArrayLike, name: str) -> FloatArray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite (found NaN or inf)")
    return arr


def hac_long_run_variance(x: npt.ArrayLike, lags: int) -> float:
    """Bartlett-kernel (Newey-West, 1987) long-run variance of the series ``x``.

    Formula: ``gamma_0 + 2 * sum_{k=1..L} (1 - k/(L+1)) * gamma_k`` with
    ``gamma_k = (1/n) sum_{t=k+1..n} (x_t - xbar)(x_{t-k} - xbar)``.
    The variance of the sample mean is ``hac_long_run_variance(x, L) / n``.
    """
    arr = _finite_1d(x, "x")
    n = arr.size
    if lags < 0 or lags >= n:
        raise ValueError(f"lags must be in [0, n-1]; got lags={lags}, n={n}")
    xc = arr - arr.mean()
    lrv = float(xc @ xc) / n
    for k in range(1, lags + 1):
        gamma_k = float(xc[k:] @ xc[:-k]) / n
        lrv += 2.0 * (1.0 - k / (lags + 1.0)) * gamma_k
    return lrv


def hln_factor(n: int, h: int) -> float:
    """Harvey-Leybourne-Newbold (1997) small-sample factor.

    Formula: ``sqrt((n + 1 - 2h + h(h-1)/n) / n)``.
    """
    inner = (n + 1 - 2 * h + h * (h - 1) / n) / n
    if inner <= 0.0:
        raise ValueError(f"HLN correction undefined for n={n}, h={h} (n too small)")
    return float(np.sqrt(inner))


def diebold_mariano(
    loss_a: npt.ArrayLike,
    loss_b: npt.ArrayLike,
    h: int = 1,
    alternative: Alternative = "less",
    hln: bool = True,
    lags: int | None = None,
) -> TestResult:
    """Diebold-Mariano test of equal expected loss.

    With loss differential ``d_t = loss_a_t - loss_b_t`` and sample mean ``dbar``::

        DM = dbar / sqrt(LRV / n),   LRV = Bartlett-HAC long-run variance of d_t,

    using ``lags = h - 1`` by default (an h-step forecast error is MA(h-1)).  When
    ``hln`` is true the statistic is multiplied by ``sqrt((n+1-2h+h(h-1)/n)/n)``
    (Harvey, Leybourne & Newbold, 1997) and p-values use Student-t with ``n-1`` degrees
    of freedom; otherwise the standard normal is used.

    ``alternative="less"`` means H1: E[loss_a] < E[loss_b] (model a is better), so a
    significantly negative statistic yields a small p-value.  ``"greater"`` is the
    mirror image and ``"two-sided"`` tests E[d] != 0.

    A non-positive long-run variance estimate falls back to the lag-0 variance
    (``detail["variance_fallback"] = True``); if that is also zero (identical loss
    series) the statistic is NaN with p-value 1.

    ``detail`` holds ``mean_d``, ``n``, ``lags``, ``se``, ``h``, ``hln``,
    ``alternative`` and ``variance_fallback``.
    """
    a = _finite_1d(loss_a, "loss_a")
    b = _finite_1d(loss_b, "loss_b")
    if a.shape != b.shape:
        raise ValueError(f"loss_a and loss_b must have equal length, got {a.size} and {b.size}")
    if h < 1:
        raise ValueError(f"h must be >= 1, got {h}")
    if alternative not in ("less", "greater", "two-sided"):
        raise ValueError(f"alternative must be 'less', 'greater' or 'two-sided', got {alternative}")
    d = a - b
    n = d.size
    n_lags = h - 1 if lags is None else int(lags)
    if n < 3 or n_lags >= n:
        raise ValueError(f"too few observations (n={n}) for lags={n_lags}")

    dbar = float(d.mean())
    lrv = hac_long_run_variance(d, n_lags)
    fallback = False
    if lrv <= 0.0:
        fallback = True
        lrv = hac_long_run_variance(d, 0)
        log.warning("dm_nonpositive_lrv", lags=n_lags, n=n, fallback_lrv=lrv)

    detail: dict[str, Any] = {
        "mean_d": dbar,
        "n": n,
        "lags": n_lags,
        "h": h,
        "hln": hln,
        "alternative": alternative,
        "variance_fallback": fallback,
    }
    if lrv <= 0.0:
        detail["se"] = 0.0
        return TestResult(statistic=float("nan"), pvalue=1.0, detail=detail)

    se = float(np.sqrt(lrv / n))
    stat = dbar / se
    if hln:
        c = hln_factor(n, h)
        stat *= c
        se /= c  # so that stat == mean_d / se in both variants
    detail["se"] = se

    if hln:
        dist: Any = stats.t(df=n - 1)
    else:
        dist = stats.norm
    if alternative == "less":
        p = float(dist.cdf(stat))
    elif alternative == "greater":
        p = float(dist.sf(stat))
    else:
        p = float(2.0 * dist.sf(abs(stat)))
    return TestResult(statistic=float(stat), pvalue=p, detail=detail)


def _default_hac_lags(n: int) -> int:
    """Newey-West (1994) rule of thumb: ``int(4 * (n / 100) ** (2/9))``."""
    return int(4 * (n / 100.0) ** (2.0 / 9.0))


def _mz_regression(y: FloatArray, x: FloatArray, hac_lags: int | None) -> dict[str, Any]:
    if y.shape != x.shape:
        raise ValueError(f"y and x must have equal length, got {y.size} and {x.size}")
    n = y.size
    if n < 10:
        raise ValueError(f"need at least 10 observations, got {n}")
    lags = _default_hac_lags(n) if hac_lags is None else int(hac_lags)
    res = sm.OLS(y, sm.add_constant(x)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    beta = np.asarray(res.params, dtype=np.float64)
    cov = np.asarray(res.cov_params(), dtype=np.float64)
    diff = beta - np.array([0.0, 1.0])
    wald = float(diff @ np.linalg.solve(cov, diff))  # chi2(2) under H0: (a, b) = (0, 1)
    se = np.sqrt(np.diag(cov))
    return {
        "a": float(beta[0]),
        "b": float(beta[1]),
        "se_a": float(se[0]),
        "se_b": float(se[1]),
        "r2": float(res.rsquared),
        "wald_stat": wald,
        "wald_pvalue": float(stats.chi2.sf(wald, 2)),
        "wald_df": 2,
        "n": n,
        "hac_lags": lags,
    }


def mincer_zarnowitz(
    rv: npt.ArrayLike, h: npt.ArrayLike, hac_lags: int | None = None
) -> dict[str, Any]:
    """Mincer-Zarnowitz regression ``rv_t = a + b * h_t + u_t`` with HAC inference.

    An optimal (unbiased, efficient) forecast has ``(a, b) = (0, 1)``.  The joint test
    is the Wald statistic ``(beta - (0,1))' V^{-1} (beta - (0,1)) ~ chi2(2)`` with V the
    Newey-West covariance (statsmodels ``cov_type="HAC"``, Bartlett kernel; default
    ``maxlags = int(4 (n/100)^(2/9))``).

    Returns a dict with ``a, b, se_a, se_b, r2, wald_stat, wald_pvalue, wald_df, n,
    hac_lags``.
    """
    y = validate_positive_finite(rv, "rv")
    x = validate_positive_finite(h, "h")
    return _mz_regression(y, x, hac_lags)


def mincer_zarnowitz_log(
    rv: npt.ArrayLike, h: npt.ArrayLike, hac_lags: int | None = None
) -> dict[str, Any]:
    """Log-scale Mincer-Zarnowitz regression ``ln rv_t = a + b * ln h_t + u_t``.

    Less sensitive to outliers than the level regression; same return structure as
    :func:`mincer_zarnowitz`.  Note that by Jensen's inequality ``E[ln rv]`` is below
    ``ln E[rv]``, so an unbiased variance forecast typically has ``a < 0`` here.
    """
    y = np.log(validate_positive_finite(rv, "rv"))
    x = np.log(validate_positive_finite(h, "h"))
    return _mz_regression(y, x, hac_lags)


def model_confidence_set(
    losses: pd.DataFrame,
    size: float = 0.10,
    reps: int = 2000,
    block_size: int | None = None,
    method: str = "R",
    seed: int = 0,
) -> dict[str, Any]:
    """Model Confidence Set of Hansen, Lunde & Nason (2011).

    Wraps :class:`arch.bootstrap.MCS` (stationary bootstrap).  ``losses`` is a T x k frame
    of per-period losses (columns = models, e.g. from
    :func:`volforecast.evaluation.losses.loss_frame`).  Models are eliminated
    sequentially until the equal-predictive-ability null is not rejected at ``size``.

    Parameters
    ----------
    block_size:
        Mean block length of the stationary bootstrap; default ``round(T^(1/3))``.
    method:
        ``"R"`` (range statistic) or ``"max"`` (max-t statistic), case-insensitive.

    Returns
    -------
    dict with ``included`` (list of model names in the MCS), ``excluded``,
    ``pvalues`` (Series of MCS p-values indexed by model, in column order), plus the
    settings used.
    """
    if losses.shape[1] < 2:
        raise ValueError("MCS needs at least two models (columns)")
    if losses.shape[0] < 3:
        raise ValueError("MCS needs at least three observations")
    if not np.all(np.isfinite(losses.to_numpy(dtype=np.float64))):
        raise ValueError("losses must be finite")
    t = losses.shape[0]
    bs = max(1, round(t ** (1.0 / 3.0))) if block_size is None else int(block_size)
    mcs = MCS(
        losses,
        size=size,
        reps=reps,
        block_size=bs,
        method=method.lower(),  # type: ignore[arg-type]
        bootstrap="stationary",
        seed=seed,
    )
    mcs.compute()
    pv = mcs.pvalues["Pvalue"].reindex(losses.columns).astype(np.float64)
    pv.name = "mcs_pvalue"
    included = [c for c in losses.columns if c in set(mcs.included)]
    excluded = [c for c in losses.columns if c in set(mcs.excluded)]
    return {
        "included": included,
        "excluded": excluded,
        "pvalues": pv,
        "size": size,
        "reps": reps,
        "block_size": bs,
        "method": method.upper(),
    }
