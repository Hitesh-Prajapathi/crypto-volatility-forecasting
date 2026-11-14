"""Shared base class for volatility models.

All models forecast the h-day cumulative realized variance

    F_t(h) = E_t[ sum_{k=1..h} RV_{t+k} ],        t = panel.index[-1],

and are fitted on the daily realized panel (`volforecast.contracts.PANEL_COLUMNS`).

`BaseVolModel` provides the pieces every model needs:

* panel validation (DatetimeIndex, strictly increasing, one-day spacing, required columns);
* the "fitted for horizon h" state machine (``fit`` must precede ``predict_variance`` and the
  horizon must match);
* training-row selection: rows whose target ``sum_{k=1..h} RV_{t+k}`` is NaN are dropped and,
  with ``drop_incomplete_targets=True`` (default), so are rows whose target window contains a
  day with ``complete == False``. Incomplete days remain valid *inputs*;
* ``_finalize``: guarantees a strictly positive, finite forecast (logged repair otherwise).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Self

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from volforecast.contracts import COL_COMPLETE, COL_RV, target_column
from volforecast.features.har import make_targets
from volforecast.logging import get_logger

log = get_logger(__name__)

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


class ModelNotFittedError(RuntimeError):
    """Raised when `predict_variance` is called before `fit` (or with a different horizon)."""


class ModelInputError(ValueError):
    """Raised for malformed panels or horizons."""


class BaseVolModel(ABC):
    """Abstract base implementing the `contracts.VolModel` protocol plumbing."""

    name: str = "base"
    #: Columns (besides `rv` and `complete`, added automatically) the model reads.
    required_columns: tuple[str, ...] = (COL_RV,)

    def __init__(self, drop_incomplete_targets: bool = True) -> None:
        self.drop_incomplete_targets = drop_incomplete_targets
        self._h: int | None = None
        self._fallback: float | None = None  # h-day scale repair value (training mean target)
        self._n_repaired = 0

    # ------------------------------------------------------------------ public API
    def fit(self, panel: pd.DataFrame, h: int) -> Self:
        """Estimate parameters on `panel` for horizon `h` (h-day cumulative variance)."""
        self._check_horizon(h)
        self._validate_panel(panel)
        self._h = None
        self._n_repaired = 0
        y, ok = self._training_targets(panel, h)
        self._fallback = float(y[ok].mean()) if ok.any() else None
        self._fit(panel, h, y, ok)
        self._h = h
        return self

    def predict_variance(self, panel: pd.DataFrame, h: int) -> float:
        """Forecast sum_{k=1..h} RV_{t+k} at t = panel.index[-1]; strictly positive, finite.

        Uses only rows <= t and the parameters from the last `fit` (state/regressors are
        rebuilt from the supplied, possibly longer, history).
        """
        self._check_fitted(h)
        self._validate_panel(panel)
        return self._finalize(self._predict(panel, h))

    @abstractmethod
    def get_params(self) -> dict[str, Any]:
        """JSON-serializable hyperparameters and fitted coefficients."""

    # --------------------------------------------------------- hooks for subclasses
    @abstractmethod
    def _fit(self, panel: pd.DataFrame, h: int, y: FloatArray, ok: BoolArray) -> None:
        """Estimate parameters. `y[i]` is the h-day target at row i, `ok[i]` its usability."""

    @abstractmethod
    def _predict(self, panel: pd.DataFrame, h: int) -> float:
        """Raw (unrepaired) h-day variance forecast at the last row of `panel`."""

    # ------------------------------------------------------------------- helpers
    @staticmethod
    def _check_horizon(h: int) -> None:
        if not isinstance(h, int | np.integer) or isinstance(h, bool) or h < 1:
            raise ModelInputError(f"horizon must be a positive integer, got {h!r}")

    def _check_fitted(self, h: int) -> None:
        if self._h is None:
            raise ModelNotFittedError(f"{self.name}: call fit() before predict_variance()")
        if h != self._h:
            raise ModelNotFittedError(
                f"{self.name}: fitted for h={self._h} but predict_variance called with h={h}"
            )

    def _validate_panel(self, panel: pd.DataFrame) -> None:
        idx = panel.index
        if not isinstance(idx, pd.DatetimeIndex):
            raise ModelInputError("panel index must be a DatetimeIndex")
        if len(idx) == 0:
            raise ModelInputError("panel is empty")
        needed = set(self.required_columns) | {COL_RV}
        if self.drop_incomplete_targets:
            needed.add(COL_COMPLETE)
        missing = sorted(needed - set(panel.columns))
        if missing:
            raise ModelInputError(f"panel is missing required columns: {missing}")
        if len(idx) > 1:
            steps = np.diff(idx.to_numpy())
            if not (steps == np.timedelta64(1, "D")).all():
                raise ModelInputError(
                    "panel index must be strictly increasing with exactly one-day spacing"
                )

    def _training_targets(self, panel: pd.DataFrame, h: int) -> tuple[FloatArray, BoolArray]:
        """Return (target_h, usable) arrays aligned with `panel` rows."""
        y = make_targets(panel, (h,))[target_column(h)].to_numpy(dtype=float)
        ok: BoolArray = np.isfinite(y)
        if self.drop_incomplete_targets:
            bad = pd.DataFrame(
                {"bad": (~panel[COL_COMPLETE].astype(bool)).to_numpy(dtype=float)},
                index=panel.index,
            )
            window_bad = make_targets(bad, (h,), col="bad")[target_column(h)].to_numpy()
            ok = ok & (window_bad == 0.0)
        return y, ok

    def _finalize(self, x: float) -> float:
        """Return `x` if strictly positive and finite, else a logged repair value."""
        if math.isfinite(x) and x > 0.0:
            return float(x)
        self._n_repaired += 1
        fb = self._fallback
        log.warning("forecast repaired", model=self.name, raw=float(x), fallback=fb, h=self._h)
        if fb is not None and math.isfinite(fb) and fb > 0.0:
            return fb
        raise ModelInputError(
            f"{self.name}: forecast {x!r} is not positive/finite and no fallback is available"
        )

    def _base_params(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "h": self._h,
            "drop_incomplete_targets": self.drop_incomplete_targets,
            "n_repaired": self._n_repaired,
        }
