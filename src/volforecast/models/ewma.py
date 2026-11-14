"""Exponentially weighted moving-average variance models (RiskMetrics family)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.signal import lfilter

from volforecast.contracts import COL_RET, COL_RV
from volforecast.logging import get_logger
from volforecast.models.base import BaseVolModel, BoolArray, FloatArray, ModelInputError

log = get_logger(__name__)

DEFAULT_LAMBDA = 0.94
LAMBDA_BOUNDS = (0.5, 0.995)
INIT_WINDOW = 30


def ewma_path(x: FloatArray, lam: float, init: float) -> FloatArray:
    r"""EWMA state after each observation: :math:`s_t = \lambda s_{t-1} + (1-\lambda) x_t`.

    Non-finite observations are skipped (the state is carried forward). Positions before the
    first valid observation hold `init`. ``s[-1]`` is the one-step-ahead variance forecast.
    """
    valid = np.isfinite(x)
    out = np.full(x.shape, init, dtype=float)
    if not valid.any():
        return out
    xv = x[valid]
    sv, _ = lfilter([1.0 - lam], [1.0, -lam], xv, zi=[lam * init])
    pos = np.cumsum(valid) - 1
    started = pos >= 0
    out[started] = sv[pos[started]]
    return out


def _qlike(realized: FloatArray, forecast: FloatArray) -> float:
    r"""Mean QLIKE loss :math:`RV/F - \ln(RV/F) - 1` (Patton, 2011, J. Econometrics)."""
    ratio = realized / forecast
    return float(np.mean(ratio - np.log(ratio) - 1.0))


class EWMAReturns(BaseVolModel):
    r"""RiskMetrics EWMA on squared daily returns (J.P. Morgan/Reuters, 1996).

    .. math::
        \sigma^2_{t+1} = \lambda \sigma^2_t + (1-\lambda) r_t^2, \qquad
        \hat F_t(h) = h \, \sigma^2_{t+1}

    with :math:`\lambda=0.94` and :math:`\sigma^2_0` the sample variance (ddof=1) of the first
    30 returns. NaN returns are skipped.
    """

    name = "ewma"
    required_columns = (COL_RV, COL_RET)

    def __init__(
        self,
        lam: float = DEFAULT_LAMBDA,
        init_window: int = INIT_WINDOW,
        drop_incomplete_targets: bool = True,
    ) -> None:
        super().__init__(drop_incomplete_targets)
        if not 0.0 < lam < 1.0:
            raise ValueError("lam must be in (0, 1)")
        self.lam = lam
        self.init_window = init_window

    def _fit(self, panel: pd.DataFrame, h: int, y: FloatArray, ok: BoolArray) -> None:
        return None  # lambda is fixed (RiskMetrics)

    def _predict(self, panel: pd.DataFrame, h: int) -> float:
        r = panel[COL_RET].to_numpy(dtype=float)
        rv = r[np.isfinite(r)][: self.init_window]
        if rv.size < 2:
            raise ModelInputError("ewma: need at least 2 valid returns to initialize")
        s = ewma_path(r * r, self.lam, float(np.var(rv, ddof=1)))
        return float(h * s[-1])

    def get_params(self) -> dict[str, Any]:
        return {**self._base_params(), "lam": self.lam, "init_window": self.init_window}


class EWMARealized(BaseVolModel):
    r"""EWMA applied to daily realized variance.

    .. math:: s_t = \lambda s_{t-1} + (1-\lambda) RV_t, \qquad \hat F_t(h) = h \, s_t

    :math:`s_0` is the mean of the first 30 RVs. With ``estimate_lambda=True`` the decay is
    chosen in `fit` by minimizing in-sample QLIKE of :math:`(\sum_{k=1}^h RV_{t+k})/h`
    against :math:`s_t` over :math:`\lambda\in[0.5, 0.995]` (bounded scalar minimization),
    a data-driven version of RiskMetrics (loss: Patton, 2011, J. Econometrics 160(1)).
    """

    name = "ewma_rv"
    required_columns = (COL_RV,)

    def __init__(
        self,
        lam: float = DEFAULT_LAMBDA,
        estimate_lambda: bool = False,
        lam_bounds: tuple[float, float] = LAMBDA_BOUNDS,
        init_window: int = INIT_WINDOW,
        drop_incomplete_targets: bool = True,
    ) -> None:
        super().__init__(drop_incomplete_targets)
        if not 0.0 < lam < 1.0:
            raise ValueError("lam must be in (0, 1)")
        self.lam = lam
        self.estimate_lambda = estimate_lambda
        self.lam_bounds = lam_bounds
        self.init_window = init_window
        self.lam_: float = lam  # decay actually used (fitted if estimate_lambda)
        self._train_qlike: float | None = None

    def _init(self, rv: FloatArray) -> float:
        xv = rv[np.isfinite(rv)][: self.init_window]
        if xv.size < 1:
            raise ModelInputError("ewma_rv: no valid RV to initialize")
        return float(np.mean(xv))

    def _fit(self, panel: pd.DataFrame, h: int, y: FloatArray, ok: BoolArray) -> None:
        self.lam_ = self.lam
        self._train_qlike = None
        if not self.estimate_lambda:
            return
        rv = panel[COL_RV].to_numpy(dtype=float)
        s0 = self._init(rv)
        pos = ok & (y > 0.0)
        if pos.sum() < 30:
            log.warning("ewma_rv: too few training rows to estimate lambda", n=int(pos.sum()))
            return
        target = y[pos] / h

        def loss(lam: float) -> float:
            s = ewma_path(rv, lam, s0)[pos]
            if not np.all(s > 0.0):
                return float("inf")
            return _qlike(target, s)

        res = minimize_scalar(
            loss, bounds=self.lam_bounds, method="bounded", options={"xatol": 1e-4}
        )
        self.lam_ = float(res.x)
        self._train_qlike = float(res.fun)

    def _predict(self, panel: pd.DataFrame, h: int) -> float:
        rv = panel[COL_RV].to_numpy(dtype=float)
        s = ewma_path(rv, self.lam_, self._init(rv))
        return float(h * s[-1])

    def get_params(self) -> dict[str, Any]:
        return {
            **self._base_params(),
            "lam": self.lam,
            "estimate_lambda": self.estimate_lambda,
            "lam_fitted": self.lam_,
            "train_qlike": self._train_qlike,
            "init_window": self.init_window,
        }
