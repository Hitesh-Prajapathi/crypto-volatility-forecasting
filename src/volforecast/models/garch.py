"""GARCH(1,1) and GJR-GARCH(1,1) forecasts of cumulative variance via `arch`."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from arch import arch_model

from volforecast.contracts import COL_RET, COL_RV
from volforecast.logging import get_logger
from volforecast.models.base import BaseVolModel, BoolArray, FloatArray
from volforecast.models.ewma import DEFAULT_LAMBDA, INIT_WINDOW, ewma_path

log = get_logger(__name__)

PCT = 100.0  # arch works on percent returns; variances convert back with /PCT**2


class GarchModel(BaseVolModel):
    r"""(GJR-)GARCH(1,1) with Constant mean on percent log returns.

    With :math:`y_t = 100\, r_t`:

    .. math::
        y_t = \mu + \varepsilon_t, \quad \varepsilon_t = \sigma_t z_t, \quad
        \sigma^2_t = \omega + \alpha \varepsilon^2_{t-1}
                     + \gamma \varepsilon^2_{t-1} \mathbf 1[\varepsilon_{t-1}<0]
                     + \beta \sigma^2_{t-1}

    ``o=0`` is GARCH (Bollerslev, 1986, J. Econometrics 31), ``o=1`` is GJR-GARCH (Glosten,
    Jagannathan & Runkle, 1993, J. Finance 48); :math:`z_t` is Normal or Student-t.

    The h-day forecast is :math:`\hat F_t(h) = 10^{-4}\sum_{k=1}^h E_t[\sigma^2_{t+k}]`.
    `fit` estimates the parameters by (quasi-)ML; `predict_variance` rebuilds the model on
    the supplied history and calls ``model.fix(params)`` so that the filter runs over the new
    data with the *fitted* parameters. On non-convergence or error the previous parameters are
    kept; with none available the model falls back to RiskMetrics EWMA (lambda=0.94).
    """

    required_columns = (COL_RV, COL_RET)

    def __init__(
        self,
        o: int = 0,
        dist: str = "t",
        mean: str = "Constant",
        min_obs: int = 100,
        drop_incomplete_targets: bool = True,
    ) -> None:
        super().__init__(drop_incomplete_targets)
        if o not in (0, 1):
            raise ValueError("o must be 0 (GARCH) or 1 (GJR-GARCH)")
        self.o = o
        self.dist = dist
        self.mean = mean
        self.min_obs = min_obs
        self.name = "gjr_garch" if o > 0 else "garch"
        self._params: pd.Series | None = None
        self._n_fit_failures = 0
        self._n_fallback_predictions = 0
        self._last_fit_converged: bool | None = None

    # ------------------------------------------------------------------ internals
    @staticmethod
    def _returns_pct(panel: pd.DataFrame) -> pd.Series:
        return panel[COL_RET].dropna().astype(float) * PCT

    def _build(self, y: pd.Series) -> Any:
        # arch types these as Literals; our values are validated in __init__.
        mean: Any = self.mean
        dist: Any = self.dist
        return arch_model(y, mean=mean, vol="garch", p=1, o=self.o, q=1, dist=dist, rescale=False)

    def _ewma_fallback(self, panel: pd.DataFrame, h: int) -> float:
        r = panel[COL_RET].to_numpy(dtype=float)
        init = r[np.isfinite(r)][:INIT_WINDOW]
        if init.size < 2:
            return float("nan")  # _finalize repairs with the training mean
        s = ewma_path(r * r, DEFAULT_LAMBDA, float(np.var(init, ddof=1)))
        return float(h * s[-1])

    # ---------------------------------------------------------------------- API
    def _fit(self, panel: pd.DataFrame, h: int, y: FloatArray, ok: BoolArray) -> None:
        self._last_fit_converged = False
        ret = self._returns_pct(panel)
        if len(ret) < self.min_obs:
            log.warning("garch: too few returns to fit", model=self.name, n=len(ret))
            self._n_fit_failures += 1
            return
        try:
            res = self._build(ret).fit(disp="off", show_warning=False)
            params = res.params
            if res.convergence_flag != 0 or not np.isfinite(params.to_numpy()).all():
                raise RuntimeError(f"optimizer did not converge (flag={res.convergence_flag})")
        except Exception as exc:  # arch raises assorted errors on degenerate data
            self._n_fit_failures += 1
            log.warning(
                "garch fit failed; keeping previous parameters",
                model=self.name,
                error=str(exc),
                have_previous=self._params is not None,
            )
            return
        self._params = params.copy()
        self._last_fit_converged = True

    def _predict(self, panel: pd.DataFrame, h: int) -> float:
        ret = self._returns_pct(panel)
        if self._params is not None and len(ret) >= self.min_obs:
            try:
                fixed = self._build(ret).fix(self._params)
                fc = fixed.forecast(horizon=h, reindex=False)
                var_pct2 = float(np.sum(fc.variance.to_numpy()[-1]))
                return var_pct2 / PCT**2
            except Exception as exc:
                log.warning("garch forecast failed; using EWMA", model=self.name, error=str(exc))
        self._n_fallback_predictions += 1
        return self._ewma_fallback(panel, h)

    def get_params(self) -> dict[str, Any]:
        params = (
            {str(k): float(v) for k, v in self._params.items()}
            if self._params is not None
            else None
        )
        return {
            **self._base_params(),
            "o": self.o,
            "dist": self.dist,
            "mean": self.mean,
            "params": params,
            "last_fit_converged": self._last_fit_converged,
            "n_fit_failures": self._n_fit_failures,
            "n_fallback_predictions": self._n_fallback_predictions,
        }
