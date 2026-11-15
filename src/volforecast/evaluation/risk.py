"""1-day VaR / ES application of the h=1 walk-forward variance forecasts (plan M4).

Information set
---------------
The forecast made at origin ``t`` (end of day ``t``) is ``f_t = E_t[RV_{t+1}]`` and is paired
with the realized close-to-close log return ``ret`` of day ``t+1``. For the Student-t and
filtered-historical-simulation (FHS) distributions the standardized residuals are

    z_s = ret_{s+1} / sqrt(f_s),

and the quantile/ES used at origin ``t`` is computed from the z_s with ``s < t`` only
(expanding window; ``ret_{s+1}`` is observed by the end of day ``t`` iff ``s + 1 <= t``).
Until ``min_resid`` such residuals exist the normal distribution is used; the distribution
actually used is recorded per row in ``dist_used``. The mean return is taken to be zero.

Note that ``f_t`` forecasts the 5-minute realized variance, not ``E[ret^2]``, so ``z`` need
not have unit variance (for level-HAR forecasts var(z) is well below 1: forecasts are
mean-based and RV-driven, while the typical squared return is much smaller). Only FHS
rescales for this, because it uses the empirical quantile of z directly. The normal and the
unit-variance Student-t (nu fitted on z, but variance fixed at 1) cannot, so they can be
systematically too conservative, or too liberal for forecasts with var(z) > 1 (e.g. EWMA).

Scale calibration
-----------------
``dist="normal_cal"`` and ``"t_cal"`` correct that scale mismatch parametrically: the forecast
variance is multiplied by ``c_t = mean(z_s^2)`` over the same strictly-past residuals
(``s < t``, expanding window, zero-mean returns as elsewhere in this module), i.e. the
distribution is applied to ``ret_{t+1} / sqrt(c_t f_t)``. For ``t_cal`` the degrees of freedom
are fitted on the rescaled residuals ``z_s / sqrt(mean(z_{<=i}^2))`` so that the fitted
unit-variance t is consistent with the scale it is applied at. Until ``min_resid`` residuals
exist the plain normal is used (``dist_used == "normal"``, ``c_t = 1``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd

from volforecast.contracts import COL_RET
from volforecast.evaluation.var_backtest import (
    acerbi_szekely_z2,
    basel_traffic_light,
    christoffersen_cc,
    christoffersen_independence,
    fit_student_t_nu,
    kupiec_pof,
    standardized_es,
    standardized_quantile,
)
from volforecast.logging import get_logger

log = get_logger(__name__)

#: ``normal_cal`` / ``t_cal`` are ``normal`` / ``t`` with the variance scaled by the expanding
#: estimate of E[z^2] from origins < t (see the module docstring).
RiskDist = Literal["normal", "t", "fhs", "normal_cal", "t_cal"]
CALIBRATED: frozenset[str] = frozenset({"normal_cal", "t_cal"})

VAR_COLUMNS: tuple[str, ...] = (
    "asset",
    "model",
    "origin_date",
    "target_date",
    "alpha",
    "dist",  # requested distribution
    "dist_used",  # distribution actually used (normal while warming up)
    "forecast_var",
    "ret",  # realized log return on target_date
    "var",  # positive loss magnitude
    "es",  # positive loss magnitude, same alpha as VaR
    "hit",
    "n_resid",  # number of standardized residuals available at the origin
    "nu",  # fitted t degrees of freedom (NaN unless dist_used is "t" / "t_cal")
)


def _origin_frame(
    forecasts: pd.DataFrame, panel: pd.DataFrame, asset: str, align: bool
) -> tuple[pd.DataFrame, pd.Series]:
    """Wide h=1 forecasts (origin x model) restricted to usable rows, and next-day returns."""
    f1 = forecasts[(forecasts["asset"] == asset) & (forecasts["horizon"] == 1)]
    wide = f1.pivot(index="origin_date", columns="model", values="forecast_var").sort_index()
    ret_next = panel[COL_RET].shift(-1).reindex(wide.index)
    valid_f = np.isfinite(wide) & (wide > 0)
    ok_ret = np.isfinite(ret_next)
    if align:
        mask = valid_f.all(axis=1) & ok_ret
        return wide[mask], ret_next[mask]
    return wide.where(valid_f), ret_next


def _t_params(
    z: np.ndarray, first: int, nu_refit_every: int, standardize: bool = False
) -> np.ndarray:
    """nu estimate per row: refit on z[:i] every ``nu_refit_every`` rows from row ``first``."""
    n = z.size
    nu = np.full(n, np.nan)
    current = np.nan
    for i in range(first, n):
        if (i - first) % nu_refit_every == 0:
            past = z[:i]
            if standardize:
                past = past / np.sqrt(np.mean(past**2))
            current = fit_student_t_nu(past)
        nu[i] = current
    return nu


def _model_rows(
    z: np.ndarray,
    f: np.ndarray,
    alpha: float,
    dist: RiskDist,
    min_resid: int,
    nu_refit_every: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Standardized quantile / ES per row, dist used, nu and variance scale, for one model.

    The scale is 1 except for the calibrated distributions, where row ``i`` uses
    ``mean(z[:i] ** 2)`` (residuals of origins strictly before row ``i``).
    """
    n = f.size
    q = np.empty(n)
    es_std = np.empty(n)
    used = np.full(n, "normal", dtype=object)
    nu = np.full(n, np.nan)
    scale = np.ones(n)
    q0 = standardized_quantile(alpha, "normal")
    e0 = standardized_es(alpha, "normal")
    q[:], es_std[:] = q0, e0
    if dist != "normal" and n > min_resid:
        first = min_resid
        cal = dist in CALIBRATED
        base = "t" if dist in ("t", "t_cal") else dist
        if cal:
            # c_i = mean(z[:i]^2): expanding second moment of strictly past residuals
            c = np.cumsum(z**2)[first - 1 : n - 1] / np.arange(first, n)
            scale[first:] = c
        if base == "t":
            nu = _t_params(z, first, nu_refit_every, standardize=cal)
            cache: dict[float, tuple[float, float]] = {}
            for i in range(first, n):
                key = float(nu[i])
                if key not in cache:
                    cache[key] = (
                        standardized_quantile(alpha, "t", nu=key),
                        standardized_es(alpha, "t", nu=key),
                    )
                q[i], es_std[i] = cache[key]
                used[i] = dist
        elif base == "fhs":
            for i in range(first, n):
                past = z[:i]  # residuals of origins strictly before row i
                # same as standardized_quantile/_es(dist="fhs"), minus its per-call tail warning
                q[i] = np.quantile(past, alpha)
                es_std[i] = -past[past <= q[i]].mean()
                used[i] = "fhs"
        else:  # normal_cal: q, es stay at the normal values, only the scale differs
            used[first:] = dist
    return q, es_std, used, nu, scale


def build_var_forecasts(
    forecasts: pd.DataFrame,
    panels: Mapping[str, pd.DataFrame],
    alphas: Sequence[float] = (0.05, 0.01),
    dists: Sequence[RiskDist] = ("normal", "t", "fhs"),
    min_resid: int = 250,
    nu_refit_every: int = 20,
    align: bool = True,
) -> pd.DataFrame:
    """1-day VaR/ES rows (``VAR_COLUMNS``) per asset x model x dist x alpha x origin.

    With ``align=True`` (default) only origins where every model of the asset has a valid
    forecast are kept, so models are compared on the same days. ``nu_refit_every`` is how
    often (in origins) the t degrees of freedom are re-estimated from past residuals.
    """
    frames: list[pd.DataFrame] = []
    for asset in sorted(forecasts["asset"].unique()):
        if asset not in panels:
            raise KeyError(f"no panel supplied for asset {asset!r}")
        wide, ret_next = _origin_frame(forecasts, panels[asset], str(asset), align)
        if wide.empty:
            log.warning("var_no_origins", asset=asset)
            continue
        origins = wide.index
        for model in wide.columns:
            f_all = wide[model]
            keep = f_all.notna().to_numpy()
            idx = origins[keep]
            f = f_all.to_numpy(dtype=np.float64)[keep]
            r = ret_next.reindex(idx).to_numpy(dtype=np.float64)
            ok = np.isfinite(r)
            idx, f, r = idx[ok], f[ok], r[ok]
            if f.size == 0:
                continue
            z = r / np.sqrt(f)
            sd = np.sqrt(f)
            for dist in dists:
                for alpha in alphas:
                    q, es_std, used, nu, scale = _model_rows(
                        z, f, alpha, dist, min_resid, nu_refit_every
                    )
                    sd_eff = sd * np.sqrt(scale)
                    var = -(q * sd_eff)
                    es = es_std * sd_eff
                    frames.append(
                        pd.DataFrame(
                            {
                                "asset": asset,
                                "model": model,
                                "origin_date": idx,
                                "target_date": pd.DatetimeIndex(
                                    idx.to_numpy() + np.timedelta64(1, "D")
                                ),
                                "alpha": alpha,
                                "dist": dist,
                                "dist_used": used,
                                "forecast_var": f,
                                "ret": r,
                                "var": var,
                                "es": es,
                                "hit": r < -var,
                                "n_resid": np.arange(f.size),
                                "nu": nu,
                            }
                        )
                    )
    if not frames:
        return pd.DataFrame({c: [] for c in VAR_COLUMNS})
    return pd.concat(frames, ignore_index=True)[list(VAR_COLUMNS)]


def var_backtest_table(var_df: pd.DataFrame) -> pd.DataFrame:
    """Coverage / ES backtests per asset x model x dist x alpha (tidy, one row each).

    Columns: ``n_obs, expected_breaches, breaches, breach_rate, kupiec_p,
    christoffersen_ind_p, christoffersen_cc_p, traffic_light`` (Basel zone of the last 250
    days at coverage ``1 - alpha``; only for alpha = 0.01), ``es_z2`` (Acerbi-Szekely),
    ``frac_fallback`` (share of rows where the requested distribution was not yet available).
    """
    keys = ["asset", "model", "dist", "alpha"]
    rows: list[dict[str, Any]] = []
    for key, g in var_df.groupby(keys, sort=False, observed=True):
        asset, model, dist, alpha_raw = key  # type: ignore[misc]
        alpha = float(str(alpha_raw))
        g = g.sort_values("origin_date")
        hits = g["hit"].to_numpy(dtype=bool)
        n, k = hits.size, int(hits.sum())
        light: str | None = None
        if np.isclose(alpha, 0.01):
            last = hits[-250:]
            light = basel_traffic_light(int(last.sum()), last.size, 1.0 - alpha)
        rows.append(
            {
                "asset": asset,
                "model": model,
                "dist": dist,
                "alpha": alpha,
                "n_obs": n,
                "expected_breaches": n * alpha,
                "breaches": k,
                "breach_rate": k / n,
                "kupiec_p": kupiec_pof(hits, alpha).pvalue,
                "christoffersen_ind_p": christoffersen_independence(hits).pvalue,
                "christoffersen_cc_p": christoffersen_cc(hits, alpha).pvalue,
                "traffic_light": light,
                "es_z2": acerbi_szekely_z2(
                    g["ret"].to_numpy(), g["var"].to_numpy(), g["es"].to_numpy(), alpha
                ),
                "frac_fallback": float((g["dist_used"] != g["dist"]).mean()),
            }
        )
    return pd.DataFrame(rows)
