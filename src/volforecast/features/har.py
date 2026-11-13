"""HAR (Heterogeneous Autoregressive) aggregates and forecast targets.

HAR regressors at date t use only data up to and including t:
    x_d(t) = X_t
    x_w(t) = mean(X_{t-4..t})
    x_m(t) = mean(X_{t-21..t})
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from volforecast.contracts import COL_RV, DEFAULT_HORIZONS, target_column

HAR_WINDOWS: dict[str, int] = {"d": 1, "w": 5, "m": 22}


def har_aggregates(
    panel: pd.DataFrame,
    col: str = COL_RV,
    log: bool = False,
    windows: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Return HAR daily/weekly/monthly aggregates of `panel[col]`.

    Columns are named ``{col}_d``, ``{col}_w``, ``{col}_m`` (prefixed ``log_`` if `log`).
    With ``log=True`` the aggregates are means of log values (the log-HAR convention).
    Rows without a full window are NaN. Strictly backward-looking.
    """
    windows = windows or HAR_WINDOWS
    x = panel[col].astype(float)
    name = col
    if log:
        x = pd.Series(np.log(x.to_numpy()), index=x.index)
        name = f"log_{col}"
    out = {f"{name}_{k}": x.rolling(w, min_periods=w).mean() for k, w in windows.items()}
    return pd.DataFrame(out, index=panel.index)


def make_targets(
    panel: pd.DataFrame,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    col: str = COL_RV,
) -> pd.DataFrame:
    """Forward cumulative-variance targets: target_rv_h{h}[t] = sum_{k=1..h} X_{t+k}.

    The last h rows are NaN (their future isn't observed in `panel`).
    """
    x = panel[col].astype(float)
    out = {}
    for h in horizons:
        # sum of the next h values: reverse rolling via shift
        fwd = sum(x.shift(-k) for k in range(1, h + 1))
        out[target_column(h)] = fwd
    return pd.DataFrame(out, index=panel.index)
