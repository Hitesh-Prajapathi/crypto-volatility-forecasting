"""Random-walk ("naive") realized-variance benchmark."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from volforecast.contracts import COL_RV
from volforecast.models.base import BaseVolModel, BoolArray, FloatArray, ModelInputError


class NaiveRV(BaseVolModel):
    r"""Persistence forecast.

    .. math:: \hat F_t(h) = h \cdot RV_t

    If :math:`RV_t` is NaN the last valid RV is used. This is the random-walk benchmark on
    realized variance (cf. Corsi, 2009, J. Financial Econometrics 7(2), 174-196).
    """

    name = "naive"
    required_columns = (COL_RV,)

    def _fit(self, panel: pd.DataFrame, h: int, y: FloatArray, ok: BoolArray) -> None:
        return None  # nothing to estimate

    def _predict(self, panel: pd.DataFrame, h: int) -> float:
        rv = panel[COL_RV].to_numpy(dtype=float)
        valid = np.flatnonzero(np.isfinite(rv))
        if valid.size == 0:
            raise ModelInputError("naive: panel has no valid RV observation")
        return float(h * rv[valid[-1]])

    def get_params(self) -> dict[str, Any]:
        return self._base_params()
