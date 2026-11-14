"""HAR-family direct h-step regressions on realized variance.

Every model projects the h-day cumulative target ``target_rv_h{h}[t] = sum_{k=1..h} RV_{t+k}``
directly on a small set of regressors dated <= t (the "direct" approach), estimated by OLS
(``numpy.linalg.lstsq``) on the rows where the target and all regressors are available.
Daily/weekly/monthly aggregates are 1/5/22-day means from
`volforecast.features.har.har_aggregates`.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from volforecast.contracts import (
    COL_BPV,
    COL_JUMP,
    COL_RQ,
    COL_RS_NEG,
    COL_RS_POS,
    COL_RV,
)
from volforecast.features.har import har_aggregates
from volforecast.logging import get_logger
from volforecast.models.base import BaseVolModel, BoolArray, FloatArray, ModelInputError

log = get_logger(__name__)

#: rows of history needed to evaluate the regressors at the forecast origin (22-day window
#: plus slack to locate the latest valid row if the last regressors are missing).
PREDICT_LOOKBACK = 60
_WM = {"w": 5, "m": 22}


def _agg(panel: pd.DataFrame, col: str, log_: bool = False) -> pd.DataFrame:
    with np.errstate(divide="ignore", invalid="ignore"):
        return har_aggregates(panel, col=col, log=log_, windows=_WM)


class _LinearVolModel(BaseVolModel):
    """Direct h-step linear projection with optional log target and insanity filter."""

    feature_names: tuple[str, ...] = ()
    _log_target = False
    _insanity_filter = False
    floor_frac: float = 0.0  # only meaningful for level models (see `_LevelModel`)

    def __init__(self, estimator: str = "ols", drop_incomplete_targets: bool = True) -> None:
        super().__init__(drop_incomplete_targets)
        if estimator not in ("ols", "wls"):
            raise ValueError("estimator must be 'ols' or 'wls'")
        self.estimator = estimator
        self._reset()

    def _reset(self) -> None:
        self._beta: FloatArray | None = None
        self._sigma2 = float("nan")
        self._n_train = 0
        self._y_lo = float("nan")
        self._y_hi = float("nan")
        self._y_mean = float("nan")
        self._n_insanity = 0
        self._n_below_floor = 0
        self._n_above_max = 0
        self._n_stale = 0

    # ------------------------------------------------------ subclass specification
    def _raw_design(self, panel: pd.DataFrame) -> FloatArray:
        """Regressors (without constant), one row per panel row, NaN when unavailable."""
        raise NotImplementedError

    def _learn_transform(self, raw_train: FloatArray) -> None:
        """Learn state of `_transform` from the usable training rows (default: none)."""

    def _transform(self, raw: FloatArray) -> FloatArray:
        return raw

    # --------------------------------------------------------------------- fitting
    def _fit(self, panel: pd.DataFrame, h: int, y: FloatArray, ok: BoolArray) -> None:
        self._reset()
        raw = self._raw_design(panel)
        use = ok & np.isfinite(raw).all(axis=1)
        if self._log_target:
            use &= y > 0.0
        k = len(self.feature_names) + 1
        n = int(use.sum())
        if n < max(30, 5 * k):
            raise ModelInputError(f"{self.name}: only {n} usable training rows (need >= 30)")
        self._learn_transform(raw[use])
        X = np.column_stack([np.ones(n), self._transform(raw[use])])
        yt = y[use]
        z = np.log(yt) if self._log_target else yt
        beta = np.linalg.lstsq(X, z, rcond=None)[0]
        if self.estimator == "wls":
            fitted = np.clip(X @ beta, max(float(np.min(yt)) * 0.1, 1e-12), None)
            sw = 1.0 / fitted
            beta = np.linalg.lstsq(X * sw[:, None], z * sw, rcond=None)[0]
        resid = z - X @ beta
        self._beta = beta
        self._sigma2 = float(resid @ resid / max(n - k, 1))
        self._n_train = n
        self._y_lo, self._y_hi, self._y_mean = float(yt.min()), float(yt.max()), float(yt.mean())
        self._fallback = self._y_mean

    # ------------------------------------------------------------------ prediction
    def _current_regressors(self, panel: pd.DataFrame) -> FloatArray | None:
        X = self._transform(self._raw_design(panel.iloc[-PREDICT_LOOKBACK:]))
        good = np.flatnonzero(np.isfinite(X).all(axis=1))
        if good.size == 0:
            return None
        if good[-1] != len(X) - 1:
            self._n_stale += 1
            log.warning(
                "regressors missing at origin; using latest valid row",
                model=self.name,
                age_days=int(len(X) - 1 - good[-1]),
            )
        row: FloatArray = X[good[-1]]
        return row

    def _linear_predictor(self, panel: pd.DataFrame) -> float | None:
        """x_t' beta (log scale for log models); None if regressors are unavailable."""
        assert self._beta is not None
        x = self._current_regressors(panel)
        if x is None:
            return None
        return float(self._beta[0] + x @ self._beta[1:])

    def _predict(self, panel: pd.DataFrame, h: int) -> float:
        lp = self._linear_predictor(panel)
        if lp is None:
            return float("nan")  # _finalize -> training mean
        if self._log_target:
            return math.exp(lp + 0.5 * self._sigma2)
        if self._insanity_filter:
            return self._sanity_check(lp, panel, h)
        return lp

    def _floor(self, panel: pd.DataFrame, h: int) -> float:
        """Lower bound of an acceptable level forecast at the origin of `panel`.

        ``max(train_min, floor_frac * h * mean(RV over the last 22 days))``: a level HAR
        forecast far below the recent run-rate of realized variance is an extrapolation
        failure (a negative loading times an extreme regressor), not information.
        """
        floor = self._y_lo
        if self.floor_frac > 0.0:
            recent = float(np.nanmean(panel[COL_RV].to_numpy(dtype=float)[-_WM["m"] :]))
            if math.isfinite(recent):
                floor = max(floor, self.floor_frac * h * recent)
        return floor

    def _sanity_check(self, lp: float, panel: pd.DataFrame, h: int) -> float:
        """Insanity filter (BPQ 2016, with a trailing-RV floor): replace by the training mean."""
        if not math.isfinite(lp) or lp <= 0.0 or lp < self._floor(panel, h):
            self._n_insanity += 1
            self._n_below_floor += 1
            return self._y_mean
        if lp > self._y_hi:
            self._n_insanity += 1
            self._n_above_max += 1
            return self._y_mean
        return lp

    def get_params(self) -> dict[str, Any]:
        coef = (
            None
            if self._beta is None
            else {
                n: float(b) for n, b in zip(("const", *self.feature_names), self._beta, strict=True)
            }
        )
        return {
            **self._base_params(),
            "estimator": self.estimator,
            "feature_names": list(self.feature_names),
            "coef": coef,
            "sigma2": self._sigma2 if math.isfinite(self._sigma2) else None,
            "n_train": self._n_train,
            "train_target_min": self._y_lo if math.isfinite(self._y_lo) else None,
            "train_target_max": self._y_hi if math.isfinite(self._y_hi) else None,
            "train_target_mean": self._y_mean if math.isfinite(self._y_mean) else None,
            "floor_frac": self.floor_frac,
            "n_insanity_filtered": self._n_insanity,
            "n_insanity_below_floor": self._n_below_floor,
            "n_insanity_above_max": self._n_above_max,
            "n_stale_inputs": self._n_stale,
        }


class _LevelModel(_LinearVolModel):
    """Level (non-log) HAR variants: forecasts pass through the insanity filter.

    A forecast is replaced by the training-target mean when it is non-finite or non-positive,
    below ``max(train_min, floor_frac * h * mean(RV, last 22 days))``, or above the training
    maximum. ``floor_frac=0`` reduces the floor to the training minimum (the original filter).
    Replacements are counted in ``get_params()["n_insanity_filtered"]`` (split into
    ``n_insanity_below_floor`` / ``n_insanity_above_max``) and reset on every refit.
    """

    _insanity_filter = True

    def __init__(
        self,
        estimator: str = "ols",
        drop_incomplete_targets: bool = True,
        floor_frac: float = 0.2,
    ) -> None:
        super().__init__(estimator, drop_incomplete_targets)
        if not math.isfinite(floor_frac) or floor_frac < 0.0:
            raise ValueError("floor_frac must be a finite number >= 0")
        self.floor_frac = float(floor_frac)


class HAR(_LevelModel):
    r"""HAR-RV (Corsi, 2009, J. Financial Econometrics 7(2), 174-196), direct h-step form.

    .. math::
        \sum_{k=1}^h RV_{t+k} = \beta_0 + \beta_d RV_t + \beta_w \overline{RV}_{t-4:t}
        + \beta_m \overline{RV}_{t-21:t} + \varepsilon_{t+h}

    Forecasts outside the [min, max] of the training targets (or <= 0) are replaced by the
    training-target mean, the "insanity filter" of Bollerslev, Patton & Quaedvlieg (2016).
    ``estimator="wls"`` re-weights by 1/fitted^2 from a first-pass OLS (BPQ, 2016).
    """

    name = "har"
    feature_names = ("rv_d", "rv_w", "rv_m")
    required_columns = (COL_RV,)

    def _raw_design(self, panel: pd.DataFrame) -> FloatArray:
        a = _agg(panel, COL_RV)
        return np.column_stack(
            [panel[COL_RV].to_numpy(dtype=float), a["rv_w"].to_numpy(), a["rv_m"].to_numpy()]
        )


class LogHAR(_LinearVolModel):
    r"""Log-HAR (Corsi, 2009; Andersen, Bollerslev, Diebold & Labys, 2003, Econometrica).

    .. math::
        \ln \sum_{k=1}^h RV_{t+k} = \beta_0 + \beta_d \ln RV_t + \beta_w \overline{\ln RV}_{t-4:t}
        + \beta_m \overline{\ln RV}_{t-21:t} + \varepsilon

    Forecast with the log-normal bias correction
    :math:`\exp(\hat y_t + \hat\sigma^2/2)`, :math:`\hat\sigma^2` = residual variance.
    """

    name = "log_har"
    feature_names = ("log_rv_d", "log_rv_w", "log_rv_m")
    required_columns = (COL_RV,)
    _log_target = True

    def __init__(self, drop_incomplete_targets: bool = True) -> None:
        super().__init__("ols", drop_incomplete_targets)

    def _raw_design(self, panel: pd.DataFrame) -> FloatArray:
        a = _agg(panel, COL_RV, log_=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            d = np.log(panel[COL_RV].to_numpy(dtype=float))
        return np.column_stack([d, a["log_rv_w"].to_numpy(), a["log_rv_m"].to_numpy()])


class HARCJ(_LevelModel):
    r"""HAR-CJ (Andersen, Bollerslev & Diebold, 2007, Review of Economics and Statistics).

    With continuous component :math:`C_t` = bipower variation and jump
    :math:`J_t=\max(RV_t-BPV_t,0)`:

    .. math::
        \sum_{k=1}^h RV_{t+k} = \beta_0 + \beta_{C,d} C_t + \beta_{C,w}\overline{C}_{t-4:t}
        + \beta_{C,m}\overline{C}_{t-21:t} + \beta_J J_t + \varepsilon
    """

    name = "har_cj"
    feature_names = ("bpv_d", "bpv_w", "bpv_m", "jump_d")
    required_columns = (COL_RV, COL_BPV, COL_JUMP)

    def _raw_design(self, panel: pd.DataFrame) -> FloatArray:
        a = _agg(panel, COL_BPV)
        return np.column_stack(
            [
                panel[COL_BPV].to_numpy(dtype=float),
                a["bpv_w"].to_numpy(),
                a["bpv_m"].to_numpy(),
                panel[COL_JUMP].to_numpy(dtype=float),
            ]
        )


class SHAR(_LevelModel):
    r"""Semivariance HAR (Patton & Sheppard, 2015, Review of Economics and Statistics).

    .. math::
        \sum_{k=1}^h RV_{t+k} = \beta_0 + \beta^+ RS^+_t + \beta^- RS^-_t
        + \beta_w \overline{RV}_{t-4:t} + \beta_m \overline{RV}_{t-21:t} + \varepsilon
    """

    name = "shar"
    feature_names = ("rs_pos_d", "rs_neg_d", "rv_w", "rv_m")
    required_columns = (COL_RV, COL_RS_POS, COL_RS_NEG)

    def _raw_design(self, panel: pd.DataFrame) -> FloatArray:
        a = _agg(panel, COL_RV)
        return np.column_stack(
            [
                panel[COL_RS_POS].to_numpy(dtype=float),
                panel[COL_RS_NEG].to_numpy(dtype=float),
                a["rv_w"].to_numpy(),
                a["rv_m"].to_numpy(),
            ]
        )


class HARQ(_LevelModel):
    r"""HARQ (Bollerslev, Patton & Quaedvlieg, 2016, J. Econometrics 192(1), 1-18).

    Interacts the daily lag with realized-quarticity (measurement-error) information:

    .. math::
        \sum_{k=1}^h RV_{t+k} = \beta_0 + (\beta_d + \beta_Q (RQ_t^{1/2}-\overline{RQ^{1/2}}))
        RV_t + \beta_w \overline{RV}_{t-4:t} + \beta_m \overline{RV}_{t-21:t} + \varepsilon

    where :math:`\overline{RQ^{1/2}}` is the training-sample mean of :math:`RQ_t^{1/2}`.
    """

    name = "harq"
    feature_names = ("rv_d", "rv_d_x_rq_sqrt_dm", "rv_w", "rv_m")
    required_columns = (COL_RV, COL_RQ)

    def _reset(self) -> None:
        super()._reset()
        self._rq_mean = float("nan")

    def _raw_design(self, panel: pd.DataFrame) -> FloatArray:
        a = _agg(panel, COL_RV)
        with np.errstate(invalid="ignore"):
            rq_sqrt = np.sqrt(panel[COL_RQ].to_numpy(dtype=float))
        return np.column_stack(
            [
                panel[COL_RV].to_numpy(dtype=float),
                rq_sqrt,
                a["rv_w"].to_numpy(),
                a["rv_m"].to_numpy(),
            ]
        )

    def _learn_transform(self, raw_train: FloatArray) -> None:
        self._rq_mean = float(raw_train[:, 1].mean())

    def _transform(self, raw: FloatArray) -> FloatArray:
        out = raw.copy()
        out[:, 1] = raw[:, 0] * (raw[:, 1] - self._rq_mean)
        return out

    def get_params(self) -> dict[str, Any]:
        p = super().get_params()
        p["rq_sqrt_train_mean"] = self._rq_mean if math.isfinite(self._rq_mean) else None
        return p
