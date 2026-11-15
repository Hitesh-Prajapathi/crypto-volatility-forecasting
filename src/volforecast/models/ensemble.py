"""Forecast combinations of volatility models.

`EqualWeightEnsemble` averages the member forecasts. `InverseLossEnsemble` weights them by the
inverse of their trailing mean QLIKE (Patton, 2011), computed from each member's OWN past
out-of-sample forecasts against realized targets that are already known at the forecast origin.

Information set
---------------
At origin ``t`` with horizon ``h`` the forecast made at origin ``s`` is scored against
``sum_{k=1..h} RV_{s+k}``, which is observed at the end of day ``s + h``. Only origins with
``s <= t - h`` therefore enter the weights at ``t`` (the ``h`` most recent origins have
unrealized targets and are ignored), and only the last ``window`` of them.

Statefulness and the walk-forward engine
----------------------------------------
The engine calls ``fit(history, h)`` every ``refit_every`` origins and
``predict_variance(history, h)`` at every origin. By default the engine builds a *fresh*
instance (``spec.builder()``) for each fit; the ensemble keeps its own state:

* the members are (re)fitted by the ensemble, not by the engine: ``fit`` fits all of them on the
  supplied history, and ``predict_variance`` refits member ``j`` once ``member_refit_every[j]``
  days have passed since its last fit, always on the (<= t) history it is given;
* every ``predict_variance`` call records the members' forecasts for that origin, so the
  trailing losses are genuine out-of-sample numbers (never in-sample fits).

Configure ensembles that need the history (``ens_inv``) with ``reuse_instance: true`` on the
engine's ``ModelSpec``: the engine then builds the ensemble once and calls ``fit`` on the same
instance at every engine refit, so the forecast history survives (it is reset only when the
horizon changes). Each ``fit`` refits all members on the supplied history; between engine
refits ``member_refit_every`` sets the members' own cadence. Without instance reuse the engine
would discard the object at every refit and the weights would restart from equal weights.
"""

from __future__ import annotations

import math
from abc import abstractmethod
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from volforecast.contracts import VolModel
from volforecast.logging import get_logger
from volforecast.models.base import BaseVolModel, BoolArray, FloatArray, ModelInputError

log = get_logger(__name__)

MIN_QLIKE = 1e-12


class _Ensemble(BaseVolModel):
    """Common member management; subclasses define the combination weights."""

    name = "ensemble"

    def __init__(
        self,
        members: Sequence[VolModel],
        member_refit_every: int | Sequence[int] = 1,
        drop_incomplete_targets: bool = True,
    ) -> None:
        super().__init__(drop_incomplete_targets)
        if len(members) < 2:
            raise ValueError("an ensemble needs at least two members")
        names = [m.name for m in members]
        if len(set(names)) != len(names):
            raise ValueError(f"member names must be unique, got {names}")
        self.members: list[VolModel] = list(members)
        if isinstance(member_refit_every, int):
            cadence = [member_refit_every] * len(members)
        else:
            cadence = list(member_refit_every)
        if len(cadence) != len(members) or any(c < 1 for c in cadence):
            raise ValueError("member_refit_every must be >= 1, one value per member")
        self.member_refit_every = cadence
        self._fit_date: list[pd.Timestamp | None] = [None] * len(members)
        self._last_fit_date: pd.Timestamp | None = None
        self._hist: dict[pd.Timestamp, FloatArray] = {}
        self._weight_log: dict[pd.Timestamp, FloatArray] = {}
        self._hist_h: int | None = None
        self._member_snapshot: list[dict[str, Any]] = []

    # ------------------------------------------------------------------- fitting
    @property
    def member_names(self) -> list[str]:
        return [m.name for m in self.members]

    def _fit(self, panel: pd.DataFrame, h: int, y: FloatArray, ok: BoolArray) -> None:
        if self._hist_h != h:
            self._hist.clear()
            self._weight_log.clear()
            self._hist_h = h
        now = panel.index[-1]
        fitted = 0
        for j, m in enumerate(self.members):
            try:
                m.fit(panel, h)
                self._fit_date[j] = now
                fitted += 1
            except Exception as exc:
                self._fit_date[j] = None
                log.warning("ensemble_member_fit_failed", member=m.name, h=h, err=repr(exc))
        if fitted == 0:
            raise ModelInputError(f"{self.name}: no member could be fitted")
        self._last_fit_date = now
        self._member_snapshot = [{"name": m.name, **m.get_params()} for m in self.members]

    def _refit_due_members(self, panel: pd.DataFrame, h: int) -> None:
        t = panel.index[-1]
        for j, m in enumerate(self.members):
            last = self._fit_date[j]
            if last is None or t <= last:
                continue
            if (t - last).days >= self.member_refit_every[j]:
                try:
                    m.fit(panel, h)
                    self._fit_date[j] = t
                except Exception as exc:  # keep the previous fit
                    log.warning("ensemble_member_refit_failed", member=m.name, err=repr(exc))

    # ---------------------------------------------------------------- prediction
    def _member_forecasts(self, panel: pd.DataFrame, h: int) -> FloatArray:
        out = np.full(len(self.members), np.nan)
        for j, m in enumerate(self.members):
            if self._fit_date[j] is None:
                continue
            try:
                v = float(m.predict_variance(panel, h))
            except Exception as exc:
                log.warning("ensemble_member_predict_failed", member=m.name, err=repr(exc))
                continue
            if math.isfinite(v) and v > 0.0:
                out[j] = v
        return out

    @abstractmethod
    def _weights(self, panel: pd.DataFrame, h: int, t: pd.Timestamp) -> FloatArray:
        """Non-negative combination weights (one per member) using information <= t only."""

    def _predict(self, panel: pd.DataFrame, h: int) -> float:
        t = panel.index[-1]
        self._refit_due_members(panel, h)
        f = self._member_forecasts(panel, h)
        valid = np.isfinite(f)
        if not valid.any():
            return float("nan")  # -> training-mean repair in `_finalize`
        # weights from forecasts of origins <= t - h (the current origin is not in them yet)
        w = self._weights(panel, h, t)
        # a forecast made by a model fitted after `t` would not be out-of-sample: record only
        # origins at or after the last (re)fit date
        if self._last_fit_date is not None and t >= self._last_fit_date:
            self._hist[t] = f
        w = np.where(valid, w, 0.0)
        if w.sum() <= 0.0:
            w = valid.astype(float)
        w = w / w.sum()
        self._weight_log[t] = w
        self._trim(t, h)
        return float(np.dot(w[valid], f[valid]))

    def _trim(self, t: pd.Timestamp, h: int) -> None:
        keep = self._keep_origins()
        if keep is None or len(self._hist) <= keep + h + 5:
            return
        for s in sorted(self._hist)[: len(self._hist) - (keep + h + 5)]:
            self._hist.pop(s, None)

    def _keep_origins(self) -> int | None:
        return None

    # ----------------------------------------------------------------- reporting
    @property
    def last_weights(self) -> dict[str, float]:
        """Weights used at the most recent forecast origin."""
        if not self._weight_log:
            return {}
        last = self._weight_log[max(self._weight_log)]
        return {n: float(w) for n, w in zip(self.member_names, last, strict=True)}

    def weight_history(self) -> pd.DataFrame:
        """Weights per forecast origin (rows) and member (columns), for diagnostics."""
        idx = sorted(self._weight_log)
        data = np.vstack([self._weight_log[s] for s in idx]) if idx else np.empty((0, 0))
        return pd.DataFrame(data, index=pd.DatetimeIndex(idx), columns=self.member_names)

    def get_params(self) -> dict[str, Any]:
        """Hyperparameters and the members' parameters as of the last ``fit``."""
        return {
            **self._base_params(),
            "members": self.member_names,
            "member_refit_every": list(self.member_refit_every),
            "member_params_at_fit": self._member_snapshot,
            **self._ensemble_params(),
        }

    def _ensemble_params(self) -> dict[str, Any]:
        return {}


class EqualWeightEnsemble(_Ensemble):
    """Arithmetic mean of the member forecasts."""

    name = "ens_eq"

    def _weights(self, panel: pd.DataFrame, h: int, t: pd.Timestamp) -> FloatArray:
        return np.ones(len(self.members))


class InverseLossEnsemble(_Ensemble):
    """Members weighted by 1 / (trailing mean QLIKE of their own past forecasts).

    ``window`` is the number of most recent *evaluable* origins (``s <= t - h``) used; below
    ``min_obs`` evaluable origins for any member the weights are equal.
    """

    name = "ens_inv"

    def __init__(
        self,
        members: Sequence[VolModel],
        window: int = 250,
        min_obs: int = 20,
        member_refit_every: int | Sequence[int] = 1,
        drop_incomplete_targets: bool = True,
    ) -> None:
        super().__init__(members, member_refit_every, drop_incomplete_targets)
        if window < 1 or min_obs < 1:
            raise ValueError("window and min_obs must be >= 1")
        self.window = window
        self.min_obs = min_obs

    def _keep_origins(self) -> int | None:
        return self.window

    def trailing_qlike(self, panel: pd.DataFrame, h: int, t: pd.Timestamp) -> FloatArray:
        """Mean QLIKE per member over the last ``window`` origins ``s <= t - h`` (NaN if short)."""
        n = len(self.members)
        cutoff = t - pd.Timedelta(days=h)
        origins = sorted(s for s in self._hist if s <= cutoff)[-self.window :]
        if len(origins) < self.min_obs:
            return np.full(n, np.nan)
        tail = panel.iloc[-(self.window + 3 * h + 5) :]
        y, ok = self._training_targets(tail, h)
        pos = tail.index.get_indexer(pd.DatetimeIndex(origins))
        good = pos >= 0
        pos = pos[good]
        F = np.vstack([self._hist[s] for s, g in zip(origins, good, strict=True) if g])
        yy = y[pos]
        use = ok[pos] & np.isfinite(yy) & (yy > 0.0)
        out = np.full(n, np.nan)
        for j in range(n):
            fj = F[:, j]
            m = use & np.isfinite(fj) & (fj > 0.0)
            if m.sum() >= self.min_obs:
                r = yy[m] / fj[m]
                out[j] = float(np.mean(r - np.log(r) - 1.0))
        return out

    def _weights(self, panel: pd.DataFrame, h: int, t: pd.Timestamp) -> FloatArray:
        loss = self.trailing_qlike(panel, h, t)
        if not np.isfinite(loss).all():
            return np.ones(len(self.members))
        inv = 1.0 / np.maximum(loss, MIN_QLIKE)
        return inv / inv.sum()

    def _ensemble_params(self) -> dict[str, Any]:
        return {"window": self.window, "min_obs": self.min_obs}
