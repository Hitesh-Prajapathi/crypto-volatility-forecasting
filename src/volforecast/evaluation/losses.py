"""Loss functions for variance forecasts.

All functions operate on **variance** forecasts ``h`` and realized variance ``rv`` (the
unit of forecasting in this repo, see :mod:`volforecast.contracts`).  Inputs must be
strictly positive and finite; anything else raises :class:`ValueError` with a message
that names the offending argument.

Loss functions
--------------
QLIKE (Patton, 2011, normalised so that the loss is 0 at a perfect forecast)::

    QLIKE(rv, h) = rv / h - ln(rv / h) - 1  >= 0

QLIKE is *robust* in the sense of Patton (2011): the ranking of forecasts is not
distorted by noise in the realized-variance proxy.  The squared error ``(rv - h)^2`` is
also robust, but is dominated by a few high-variance days; the absolute error on
volatility ``|sqrt(rv) - sqrt(h)|`` is *not* robust (reported for interpretability only).

References
----------
Patton, A. J. (2011). Volatility forecast comparison using imperfect volatility proxies.
    Journal of Econometrics, 160(1), 246-256.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

LossName = Literal["qlike", "mse", "mae_vol"]
FloatArray = npt.NDArray[np.float64]

LOSS_NAMES: tuple[LossName, ...] = ("qlike", "mse", "mae_vol")


def validate_positive_finite(x: npt.ArrayLike, name: str) -> FloatArray:
    """Return ``x`` as a float64 array after checking it is finite and strictly positive."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite (found NaN or inf)")
    if np.any(arr <= 0.0):
        raise ValueError(f"{name} must be strictly positive (found non-positive values)")
    return arr


def _validated_pair(rv: npt.ArrayLike, h: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
    rv_a = validate_positive_finite(rv, "rv")
    h_a = validate_positive_finite(h, "h")
    try:
        rv_b, h_b = np.broadcast_arrays(rv_a, h_a)
    except ValueError as exc:
        raise ValueError(f"rv and h have incompatible shapes {rv_a.shape} and {h_a.shape}") from exc
    return rv_b, h_b


def qlike(rv: npt.ArrayLike, h: npt.ArrayLike) -> FloatArray:
    """Elementwise normalised QLIKE loss (Patton, 2011).

    Formula: ``rv / h - ln(rv / h) - 1``.  It is zero iff ``h == rv`` and strictly
    positive otherwise (``x - ln x - 1 >= 0`` for x > 0).
    """
    rv_a, h_a = _validated_pair(rv, h)
    ratio = rv_a / h_a
    out: FloatArray = ratio - np.log(ratio) - 1.0
    return out


def mse(rv: npt.ArrayLike, h: npt.ArrayLike) -> FloatArray:
    """Elementwise squared error on variance: ``(rv - h)^2``."""
    rv_a, h_a = _validated_pair(rv, h)
    out: FloatArray = (rv_a - h_a) ** 2
    return out


def mae_vol(rv: npt.ArrayLike, h: npt.ArrayLike) -> FloatArray:
    """Elementwise absolute error on volatility: ``|sqrt(rv) - sqrt(h)|``."""
    rv_a, h_a = _validated_pair(rv, h)
    out: FloatArray = np.abs(np.sqrt(rv_a) - np.sqrt(h_a))
    return out


_LOSS_FUNCS: dict[str, Callable[[npt.ArrayLike, npt.ArrayLike], FloatArray]] = {
    "qlike": qlike,
    "mse": mse,
    "mae_vol": mae_vol,
}


def _loss_func(loss: str) -> Callable[[npt.ArrayLike, npt.ArrayLike], FloatArray]:
    try:
        return _LOSS_FUNCS[loss]
    except KeyError:
        raise ValueError(f"unknown loss {loss!r}; choose from {LOSS_NAMES}") from None


def _align(rv: pd.Series, forecasts: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Inner-join ``rv`` and ``forecasts`` on their index."""
    if not isinstance(rv, pd.Series):
        rv = pd.Series(np.asarray(rv, dtype=np.float64), index=forecasts.index)
    idx = rv.index.intersection(forecasts.index)
    if len(idx) == 0:
        raise ValueError("rv and forecasts share no index labels")
    return rv.loc[idx], forecasts.loc[idx]


def loss_frame(rv: pd.Series, forecasts: pd.DataFrame, loss: LossName = "qlike") -> pd.DataFrame:
    """Per-observation loss of every model.

    Parameters
    ----------
    rv:
        Realized variance (aligned to ``forecasts`` on the index; inner join).
    forecasts:
        One column per model holding variance forecasts.
    loss:
        ``"qlike"`` (default), ``"mse"`` or ``"mae_vol"``.

    Returns
    -------
    DataFrame with the same columns as ``forecasts`` and the common index.
    """
    func = _loss_func(loss)
    rv_a, f_a = _align(rv, forecasts)
    rv_np = rv_a.to_numpy(dtype=np.float64)
    out = {col: func(rv_np, f_a[col].to_numpy(dtype=np.float64)) for col in f_a.columns}
    return pd.DataFrame(out, index=rv_a.index)


def summary_table(rv: pd.Series, forecasts: pd.DataFrame, benchmark: str) -> pd.DataFrame:
    """Mean losses per model and the QLIKE ratio versus a benchmark model.

    Returns a DataFrame indexed by model with columns ``qlike``, ``mse``, ``mae_vol``
    (sample means) and ``qlike_ratio`` = mean QLIKE(model) / mean QLIKE(benchmark);
    values below 1 mean the model beats the benchmark.
    """
    if benchmark not in forecasts.columns:
        raise ValueError(f"benchmark {benchmark!r} not in forecasts columns")
    table = pd.DataFrame({name: loss_frame(rv, forecasts, name).mean() for name in LOSS_NAMES})
    table["qlike_ratio"] = table["qlike"] / float(table["qlike"].loc[benchmark])
    return table
