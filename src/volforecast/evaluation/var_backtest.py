"""Value-at-Risk / Expected-Shortfall forecasting and backtesting library.

Conventions
-----------
* **VaR and ES are POSITIVE loss magnitudes.**  A VaR of 0.05 means "a 5% loss".
* ``returns`` are 1-day *log* returns ``r_t`` and ``variance`` is the one-day-ahead
  conditional variance forecast ``h_t`` made *before* ``r_t`` is observed.
* ``alpha`` is the tail probability (0.05 or 0.01), i.e. the VaR confidence level is
  ``1 - alpha``.
* The return is modelled as ``r_t = mu + sqrt(h_t) * z_t`` where ``z_t`` is a
  zero-mean, UNIT-variance standardized innovation with quantile ``q_alpha`` (negative
  for small alpha), hence::

      VaR_alpha,t = -(mu + q_alpha * sqrt(h_t))
      ES_alpha,t  = -(mu + E[z | z <= q_alpha] * sqrt(h_t))   (>= VaR_alpha,t)

* A *hit* (breach / exceedance) is ``r_t < -VaR_t``.
* Standardized distributions: ``"normal"``; ``"t"`` (Student-t rescaled to unit variance,
  ``t_nu * sqrt((nu-2)/nu)``, requires nu > 2); ``"fhs"`` (filtered historical
  simulation: empirical quantile / tail mean of supplied standardized residuals;
  Barone-Adesi, Giannopoulos & Vosper, 1999).

Tests
-----
Kupiec (1995) proportion-of-failures, Christoffersen (1998) independence and
conditional-coverage, the Basel traffic-light zones (BCBS, 1996) and the Acerbi-Szekely
(2014) Z2 ES test.  All LR statistics treat ``0 * log(0) = 0``.

References
----------
Kupiec, P. (1995). Techniques for verifying the accuracy of risk measurement models.
    Journal of Derivatives 3(2), 73-84.
Christoffersen, P. (1998). Evaluating interval forecasts. International Economic
    Review 39(4), 841-862.
Basel Committee on Banking Supervision (1996). Supervisory framework for the use of
    "backtesting" in conjunction with the internal models approach to market risk
    capital requirements.
Acerbi, C. & Szekely, B. (2014). Back-testing expected shortfall. Risk, December 2014.
Barone-Adesi, G., Giannopoulos, K. & Vosper, L. (1999). VaR without correlations for
    portfolios of derivative securities. Journal of Futures Markets 19(5), 583-602.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import optimize, special, stats

from volforecast.contracts import TestResult
from volforecast.evaluation.losses import FloatArray, validate_positive_finite
from volforecast.logging import get_logger

log = get_logger(__name__)

Dist = Literal["normal", "t", "fhs"]
Zone = Literal["green", "yellow", "red"]

NU_MIN = 2.05
NU_MAX = 100.0
_MIN_FHS_OBS = 20


# ── helpers ──────────────────────────────────────────────────────────────────
def _check_alpha(alpha: float) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")


def _finite_array(x: npt.ArrayLike, name: str) -> FloatArray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite (found NaN or inf)")
    return arr


def _as_hits(hits_: npt.ArrayLike) -> npt.NDArray[np.bool_]:
    arr = np.asarray(hits_)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("hits must be a non-empty 1-D array")
    if arr.dtype != np.bool_:
        if not np.all(np.isin(arr, (0, 1))):
            raise ValueError("hits must be boolean or 0/1")
        arr = arr.astype(bool)
    return arr


def _check_nu(nu: float | None) -> float:
    if nu is None:
        raise ValueError("nu is required for dist='t'")
    if not nu > 2.0:
        raise ValueError(f"nu must be > 2 for a unit-variance Student-t, got {nu}")
    return float(nu)


def _check_resid(std_resid: npt.ArrayLike | None, alpha: float) -> FloatArray:
    if std_resid is None:
        raise ValueError("std_resid is required for dist='fhs'")
    z = _finite_array(std_resid, "std_resid")
    if z.size < _MIN_FHS_OBS:
        raise ValueError(f"need at least {_MIN_FHS_OBS} standardized residuals for FHS")
    if z.size * alpha < 5:
        log.warning("fhs_few_tail_obs", n=int(z.size), alpha=alpha, expected_tail=z.size * alpha)
    return z


# ── standardized distribution ────────────────────────────────────────────────
def standardized_quantile(
    alpha: float,
    dist: Dist = "normal",
    nu: float | None = None,
    std_resid: npt.ArrayLike | None = None,
) -> float:
    """alpha-quantile ``q_alpha`` of a zero-mean, unit-variance distribution.

    * normal: ``Phi^{-1}(alpha)``
    * t: ``t.ppf(alpha, nu) * sqrt((nu - 2) / nu)``
    * fhs: empirical alpha-quantile of ``std_resid`` (linear interpolation).
    """
    _check_alpha(alpha)
    if dist == "normal":
        return float(stats.norm.ppf(alpha))
    if dist == "t":
        v = _check_nu(nu)
        return float(stats.t.ppf(alpha, v) * np.sqrt((v - 2.0) / v))
    if dist == "fhs":
        return float(np.quantile(_check_resid(std_resid, alpha), alpha))
    raise ValueError(f"unknown dist {dist!r}; choose 'normal', 't' or 'fhs'")


def standardized_es(
    alpha: float,
    dist: Dist = "normal",
    nu: float | None = None,
    std_resid: npt.ArrayLike | None = None,
) -> float:
    """Positive expected shortfall ``-E[z | z <= q_alpha]`` of the standardized law.

    * normal: ``phi(q_alpha) / alpha``
    * t (unit variance): ``s * f_nu(t_a) / alpha * (nu + t_a^2) / (nu - 1)`` with
      ``t_a = t.ppf(alpha, nu)`` and ``s = sqrt((nu-2)/nu)`` (McNeil, Frey &
      Embrechts, 2015, eq. 2.26).
    * fhs: ``-mean(z[z <= q_alpha])`` over the residuals.
    """
    _check_alpha(alpha)
    if dist == "normal":
        return float(stats.norm.pdf(stats.norm.ppf(alpha)) / alpha)
    if dist == "t":
        v = _check_nu(nu)
        t_a = stats.t.ppf(alpha, v)
        s = np.sqrt((v - 2.0) / v)
        return float(s * stats.t.pdf(t_a, v) / alpha * (v + t_a**2) / (v - 1.0))
    if dist == "fhs":
        z = _check_resid(std_resid, alpha)
        q = np.quantile(z, alpha)
        return float(-z[z <= q].mean())
    raise ValueError(f"unknown dist {dist!r}; choose 'normal', 't' or 'fhs'")


def fit_student_t_nu(std_resid: npt.ArrayLike) -> float:
    """MLE of the degrees of freedom of a unit-variance standardized Student-t.

    Density of ``x = s * T``, ``s = sqrt((nu-2)/nu)``, ``T ~ t_nu``:
    ``f(x) = f_nu(x / s) / s``.  Maximises the log-likelihood over ``nu`` in
    ``[2.05, 100]`` (bounded Brent); data that look Gaussian return a value near the
    upper bound.
    """
    z = _finite_array(std_resid, "std_resid")
    if z.size < 10:
        raise ValueError("need at least 10 residuals to fit nu")

    def neg_loglik(nu: float) -> float:
        s = np.sqrt((nu - 2.0) / nu)
        return float(-(stats.t.logpdf(z / s, nu) - np.log(s)).sum())

    res = optimize.minimize_scalar(
        neg_loglik, bounds=(NU_MIN, NU_MAX), method="bounded", options={"xatol": 1e-6}
    )
    return float(res.x)


# ── VaR / ES ─────────────────────────────────────────────────────────────────
def _scale(variance: npt.ArrayLike, mu: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
    var_a = validate_positive_finite(variance, "variance")
    mu_a = np.asarray(mu, dtype=np.float64)
    if not np.all(np.isfinite(mu_a)):
        raise ValueError("mu must be finite")
    return np.sqrt(var_a), mu_a


def value_at_risk(
    variance: npt.ArrayLike,
    alpha: float,
    dist: Dist = "normal",
    mu: npt.ArrayLike = 0.0,
    nu: float | None = None,
    std_resid: npt.ArrayLike | None = None,
) -> FloatArray:
    """One-day VaR as a positive loss: ``-(mu + q_alpha * sqrt(h))``, vectorized in h."""
    sd, mu_a = _scale(variance, mu)
    q = standardized_quantile(alpha, dist, nu, std_resid)
    out: FloatArray = -(mu_a + q * sd)
    return out


def expected_shortfall(
    variance: npt.ArrayLike,
    alpha: float,
    dist: Dist = "normal",
    mu: npt.ArrayLike = 0.0,
    nu: float | None = None,
    std_resid: npt.ArrayLike | None = None,
) -> FloatArray:
    """One-day ES as a positive loss: ``-mu + sqrt(h) * ES_std`` (see ``standardized_es``)."""
    sd, mu_a = _scale(variance, mu)
    es_std = standardized_es(alpha, dist, nu, std_resid)
    out: FloatArray = -mu_a + es_std * sd
    return out


def hits(returns: npt.ArrayLike, var: npt.ArrayLike) -> npt.NDArray[np.bool_]:
    """Breach indicator ``r_t < -VaR_t`` (VaR a positive loss magnitude)."""
    r = _finite_array(returns, "returns")
    v = _finite_array(var, "var")
    if r.shape != v.shape and v.size != 1:
        raise ValueError(f"returns and var shapes differ: {r.shape} vs {v.shape}")
    out: npt.NDArray[np.bool_] = r < -v
    return out


# ── coverage tests ───────────────────────────────────────────────────────────
def _pof_lr(x: int, n: int, alpha: float) -> float:
    """Kupiec LR with explicit counts; xlogy gives 0*log(0) = 0."""
    pi_hat = x / n
    ll0 = special.xlogy(n - x, 1.0 - alpha) + special.xlogy(x, alpha)
    ll1 = special.xlogy(n - x, 1.0 - pi_hat) + special.xlogy(x, pi_hat)
    return max(0.0, float(-2.0 * (ll0 - ll1)))


def kupiec_pof(hits_: npt.ArrayLike, alpha: float) -> TestResult:
    """Kupiec (1995) proportion-of-failures test, H0: P(hit) = alpha.

    ``LR_POF = -2 ln[(1-a)^(n-x) a^x] + 2 ln[(1-x/n)^(n-x) (x/n)^x] ~ chi2(1)`` with
    ``x`` hits in ``n`` observations.  Zero-hit and all-hit samples are handled by
    ``0 ln 0 = 0``.
    """
    _check_alpha(alpha)
    h_ = _as_hits(hits_)
    n, x = int(h_.size), int(h_.sum())
    lr = _pof_lr(x, n, alpha)
    return TestResult(
        statistic=lr,
        pvalue=float(stats.chi2.sf(lr, 1)),
        detail={"n": n, "hits": x, "hit_rate": x / n, "alpha": alpha},
    )


def _transition_counts(h_: npt.NDArray[np.bool_]) -> tuple[int, int, int, int]:
    prev, cur = h_[:-1], h_[1:]
    n00 = int(np.sum(~prev & ~cur))
    n01 = int(np.sum(~prev & cur))
    n10 = int(np.sum(prev & ~cur))
    n11 = int(np.sum(prev & cur))
    return n00, n01, n10, n11


def _ind_lr(n00: int, n01: int, n10: int, n11: int) -> float:
    n_tot = n00 + n01 + n10 + n11
    pi = (n01 + n11) / n_tot
    pi01 = n01 / (n00 + n01) if n00 + n01 > 0 else 0.0
    pi11 = n11 / (n10 + n11) if n10 + n11 > 0 else 0.0
    ll0 = special.xlogy(n00 + n10, 1.0 - pi) + special.xlogy(n01 + n11, pi)
    ll1 = (
        special.xlogy(n00, 1.0 - pi01)
        + special.xlogy(n01, pi01)
        + special.xlogy(n10, 1.0 - pi11)
        + special.xlogy(n11, pi11)
    )
    return max(0.0, float(-2.0 * (ll0 - ll1)))


def christoffersen_independence(hits_: npt.ArrayLike) -> TestResult:
    """Christoffersen (1998) first-order Markov independence test.

    H0: ``pi_01 = pi_11`` (the hit probability does not depend on yesterday's hit)::

        LR_IND = -2 ln[(1-pi)^(n00+n10) pi^(n01+n11)]
                 + 2 ln[(1-pi01)^n00 pi01^n01 (1-pi11)^n10 pi11^n11]  ~ chi2(1)

    with ``n_ij`` the number of transitions i -> j, ``pi01 = n01/(n00+n01)``,
    ``pi11 = n11/(n10+n11)`` and ``pi = (n01+n11)/n_tot``.  Empty conditioning rows
    contribute 0 (``0 ln 0 = 0``).  The first observation is used only as a
    conditioning state.
    """
    h_ = _as_hits(hits_)
    if h_.size < 2:
        raise ValueError("need at least 2 observations for the independence test")
    n00, n01, n10, n11 = _transition_counts(h_)
    lr = _ind_lr(n00, n01, n10, n11)
    return TestResult(
        statistic=lr,
        pvalue=float(stats.chi2.sf(lr, 1)),
        detail={"n00": n00, "n01": n01, "n10": n10, "n11": n11},
    )


def christoffersen_cc(hits_: npt.ArrayLike, alpha: float) -> TestResult:
    """Christoffersen conditional-coverage test: ``LR_CC = LR_POF + LR_IND ~ chi2(2)``."""
    pof = kupiec_pof(hits_, alpha)
    ind = christoffersen_independence(hits_)
    lr = pof.statistic + ind.statistic
    return TestResult(
        statistic=lr,
        pvalue=float(stats.chi2.sf(lr, 2)),
        detail={"lr_pof": pof.statistic, "lr_ind": ind.statistic, "alpha": alpha},
    )


# ── Basel traffic light ──────────────────────────────────────────────────────
_GREEN_CUM = 0.95
_YELLOW_CUM = 0.9999


def _zone_from_cdf(cdf: float) -> Zone:
    if cdf < _GREEN_CUM:
        return "green"
    if cdf < _YELLOW_CUM:
        return "yellow"
    return "red"


def basel_traffic_light(n_breaches: int, n_obs: int = 250, coverage: float = 0.99) -> Zone:
    """Basel (1996) traffic-light zone from the binomial CDF.

    With ``X ~ Bin(n_obs, 1 - coverage)``: green if ``P(X <= x) < 0.95``, yellow if
    ``< 0.9999``, red otherwise.  For 250 observations at 99%: green 0-4, yellow 5-9,
    red >= 10.
    """
    if n_breaches < 0 or n_breaches > n_obs:
        raise ValueError(f"n_breaches must be in [0, n_obs]; got {n_breaches} (n_obs={n_obs})")
    if not 0.0 < coverage < 1.0:
        raise ValueError(f"coverage must be in (0, 1), got {coverage}")
    return _zone_from_cdf(float(stats.binom.cdf(n_breaches, n_obs, 1.0 - coverage)))


def rolling_traffic_light(
    hits_: npt.ArrayLike | pd.Series, window: int = 250, coverage: float = 0.99
) -> pd.Series:
    """Traffic-light zone of the breach count in each trailing ``window``.

    Returns a categorical Series (categories green < yellow < red, ordered) aligned to
    ``hits_`` (its index if a Series); entries before the first full window are NaN.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    h_ = _as_hits(hits_)
    idx = hits_.index if isinstance(hits_, pd.Series) else pd.RangeIndex(h_.size)
    counts = pd.Series(h_.astype(np.int64), index=idx).rolling(window).sum()
    cdf_by_count = stats.binom.cdf(np.arange(window + 1), window, 1.0 - coverage)
    lookup = np.array([_zone_from_cdf(float(c)) for c in cdf_by_count], dtype=object)
    zones = pd.Series(pd.NA, index=idx, dtype=object)
    valid = counts.notna().to_numpy()
    zones.iloc[np.flatnonzero(valid)] = lookup[counts.to_numpy()[valid].astype(np.int64)]
    return zones.astype(pd.CategoricalDtype(["green", "yellow", "red"], ordered=True))


# ── Expected-shortfall test ──────────────────────────────────────────────────
def acerbi_szekely_z2(
    returns: npt.ArrayLike, var: npt.ArrayLike, es: npt.ArrayLike, alpha: float
) -> float:
    """Acerbi-Szekely (2014) Z2 statistic (VaR and ES as positive losses).

    ``Z2 = (1/(N alpha)) * sum_t [ r_t * I_t / ES_t ] + 1`` with
    ``I_t = 1[r_t < -VaR_t]``; equivalently ``1 - (1/(N alpha)) sum_t I_t * loss_t / ES_t``
    with ``loss_t = -r_t``.  Under H0 (the forecast ES is the true conditional tail mean)
    ``E[Z2] = 0``; a significantly negative value means ES is *underestimated* (realised tail
    losses exceed the forecast).
    """
    _check_alpha(alpha)
    r = _finite_array(returns, "returns")
    v = _finite_array(var, "var")
    e = validate_positive_finite(es, "es")
    if not (r.shape == np.broadcast_shapes(r.shape, v.shape, e.shape)):
        raise ValueError("returns, var and es must have compatible shapes")
    indicator = r < -v
    e_b = np.broadcast_to(e, r.shape)
    return float(np.sum(r[indicator] / e_b[indicator]) / (r.size * alpha) + 1.0)


def acerbi_szekely_pvalue(
    returns: npt.ArrayLike,
    variance: npt.ArrayLike,
    alpha: float,
    dist: Dist = "normal",
    mu: npt.ArrayLike = 0.0,
    nu: float | None = None,
    std_resid: npt.ArrayLike | None = None,
    n_sim: int = 5000,
    seed: int = 0,
    chunk: int = 500,
) -> TestResult:
    """Simulated one-sided p-value of the Z2 test, H1: ES underestimated.

    VaR and ES are computed from the forecast distribution; the null distribution of
    Z2 is obtained by simulating ``r*_t = mu_t + sqrt(h_t) z*_t`` with ``z*`` drawn from
    the same standardized law (normal / unit-variance t / bootstrap of ``std_resid``),
    ``n_sim`` times.  ``p = (1 + #{Z2* <= Z2_obs}) / (n_sim + 1)``.
    """
    r = _finite_array(returns, "returns")
    var_a = validate_positive_finite(variance, "variance")
    if r.shape != var_a.shape or r.ndim != 1:
        raise ValueError("returns and variance must be 1-D arrays of equal length")
    mu_a = np.broadcast_to(np.asarray(mu, dtype=np.float64), r.shape)
    var_q = value_at_risk(var_a, alpha, dist, mu_a, nu, std_resid)
    es_q = expected_shortfall(var_a, alpha, dist, mu_a, nu, std_resid)
    z2_obs = acerbi_szekely_z2(r, var_q, es_q, alpha)

    rng = np.random.default_rng(seed)
    sd = np.sqrt(var_a)
    resid = _check_resid(std_resid, alpha) if dist == "fhs" else None
    sims = np.empty(n_sim)
    done = 0
    while done < n_sim:
        m = min(chunk, n_sim - done)
        if dist == "normal":
            z = rng.standard_normal((m, r.size))
        elif dist == "t":
            v = _check_nu(nu)
            z = rng.standard_t(v, size=(m, r.size)) * np.sqrt((v - 2.0) / v)
        elif resid is not None:
            z = rng.choice(resid, size=(m, r.size), replace=True)
        else:
            raise ValueError(f"unknown dist {dist!r}")
        r_sim = mu_a + sd * z
        ind = r_sim < -var_q
        ratio = np.where(ind, r_sim / es_q, 0.0)
        sims[done : done + m] = ratio.sum(axis=1) / (r.size * alpha) + 1.0
        done += m
    p = float((1 + np.sum(sims <= z2_obs)) / (n_sim + 1))
    return TestResult(
        statistic=z2_obs,
        pvalue=p,
        detail={"n_sim": n_sim, "seed": seed, "sim_mean": float(sims.mean()), "alpha": alpha},
    )


# ── summary ──────────────────────────────────────────────────────────────────
def backtest_summary(
    returns: npt.ArrayLike | pd.Series,
    variance: npt.ArrayLike | pd.Series,
    alphas: tuple[float, ...] = (0.05, 0.01),
    dist: Dist = "normal",
    mu: float = 0.0,
    nu: float | None = None,
    std_resid: npt.ArrayLike | None = None,
) -> pd.DataFrame:
    """One row per alpha summarising a VaR (and ES) backtest.

    Columns: ``n_obs``, ``expected_breaches`` (= n * alpha), ``breaches``,
    ``breach_rate``, ``kupiec_p``, ``christoffersen_ind_p``, ``christoffersen_cc_p``,
    ``es_z2`` (Acerbi-Szekely) and ``traffic_light`` (Basel zone of the breaches in the
    last min(n, 250) observations at coverage ``1 - alpha``; only reported for
    ``alpha = 0.01``, ``None`` otherwise).  Series inputs are inner-joined on their
    index.  For ``dist="t"`` with ``nu=None`` and ``std_resid`` given, nu is fitted by
    MLE.
    """
    if isinstance(returns, pd.Series) and isinstance(variance, pd.Series):
        both = pd.concat([returns, variance], axis=1, join="inner")
        r = both.iloc[:, 0].to_numpy(dtype=np.float64)
        v = both.iloc[:, 1].to_numpy(dtype=np.float64)
    else:
        r = _finite_array(returns, "returns")
        v = np.asarray(variance, dtype=np.float64)
    if r.shape != v.shape:
        raise ValueError(f"returns and variance lengths differ: {r.shape} vs {v.shape}")
    if dist == "t" and nu is None and std_resid is not None:
        nu = fit_student_t_nu(std_resid)
    n = r.size
    rows: dict[float, dict[str, Any]] = {}
    for alpha in alphas:
        var_q = value_at_risk(v, alpha, dist, mu, nu, std_resid)
        es_q = expected_shortfall(v, alpha, dist, mu, nu, std_resid)
        h_ = hits(r, var_q)
        k = int(h_.sum())
        light: str | None = None
        if np.isclose(alpha, 0.01):
            last = h_[-250:]
            light = basel_traffic_light(int(last.sum()), last.size, 1.0 - alpha)
        rows[alpha] = {
            "n_obs": n,
            "expected_breaches": n * alpha,
            "breaches": k,
            "breach_rate": k / n,
            "kupiec_p": kupiec_pof(h_, alpha).pvalue,
            "christoffersen_ind_p": christoffersen_independence(h_).pvalue,
            "christoffersen_cc_p": christoffersen_cc(h_, alpha).pvalue,
            "es_z2": acerbi_szekely_z2(r, var_q, es_q, alpha),
            "traffic_light": light,
        }
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "alpha"
    return out
