"""Name -> model constructor registry for the econometric baselines."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from volforecast.contracts import VolModel
from volforecast.models.ensemble import EqualWeightEnsemble, InverseLossEnsemble
from volforecast.models.ewma import EWMARealized, EWMAReturns
from volforecast.models.garch import GarchModel
from volforecast.models.gbm import LightGBMVol, XGBoostVol
from volforecast.models.har import HAR, HARCJ, HARQ, SHAR, HARDow, LogHAR, LogHARDow
from volforecast.models.naive import NaiveRV


def _garch(**params: Any) -> VolModel:
    return GarchModel(**params)


def _lstm(**params: Any) -> VolModel:
    """LSTM challenger; torch (the optional ``dl`` extra) is imported only when it is built."""
    from volforecast.models.lstm import LSTMVol

    return LSTMVol(**params)


def _gjr_garch(**params: Any) -> VolModel:
    params.setdefault("o", 1)
    return GarchModel(**params)


#: Default ensemble members and their internal refit cadence (days). The GBM is refitted weekly,
#: matching the ``refit_every: 7`` of the standalone ``lgbm`` entry in ``configs/backtest.yaml``.
ENSEMBLE_MEMBERS: tuple[str, ...] = ("log_har", "harq", "lgbm")
ENSEMBLE_REFIT: dict[str, int] = {"log_har": 1, "harq": 1, "lgbm": 7, "xgb": 7}


def _build_members(
    members: Sequence[str], member_params: dict[str, dict[str, Any]] | None
) -> tuple[list[VolModel], list[int]]:
    params = member_params or {}
    built = [build_model(m, **params.get(m, {})) for m in members]
    return built, [ENSEMBLE_REFIT.get(m, 1) for m in members]


def _ens_eq(
    members: Sequence[str] = ENSEMBLE_MEMBERS,
    member_params: dict[str, dict[str, Any]] | None = None,
    **params: Any,
) -> VolModel:
    built, refit = _build_members(members, member_params)
    params.setdefault("member_refit_every", refit)
    return EqualWeightEnsemble(built, **params)


def _ens_inv(
    members: Sequence[str] = ENSEMBLE_MEMBERS,
    member_params: dict[str, dict[str, Any]] | None = None,
    **params: Any,
) -> VolModel:
    built, refit = _build_members(members, member_params)
    params.setdefault("member_refit_every", refit)
    return InverseLossEnsemble(built, **params)


MODEL_FACTORY: dict[str, Callable[..., VolModel]] = {
    "naive": NaiveRV,
    "ewma": EWMAReturns,
    "ewma_rv": EWMARealized,
    "garch": _garch,
    "gjr_garch": _gjr_garch,
    "har": HAR,
    "log_har": LogHAR,
    "har_dow": HARDow,
    "log_har_dow": LogHARDow,
    "har_cj": HARCJ,
    "shar": SHAR,
    "harq": HARQ,
    "lgbm": LightGBMVol,
    "xgb": XGBoostVol,
    "lstm": _lstm,
    "ens_eq": _ens_eq,
    "ens_inv": _ens_inv,
}


def list_models() -> list[str]:
    """Names accepted by `build_model`, in registration order."""
    return list(MODEL_FACTORY)


def build_model(name: str, **params: Any) -> VolModel:
    """Instantiate the model registered under `name` with constructor keyword `params`."""
    try:
        ctor = MODEL_FACTORY[name]
    except KeyError:
        raise ValueError(f"unknown model {name!r}; available: {list_models()}") from None
    return ctor(**params)


#: Models that accumulate out-of-sample state across walk-forward refits. Any walk-forward
#: driver (backtest, retrain) must run them with ``ModelSpec(reuse_instance=True)``; otherwise
#: the state is discarded at every refit (``ens_inv`` then silently degrades to ``ens_eq``).
STATEFUL_MODELS: frozenset[str] = frozenset({"ens_inv"})


def is_stateful_model(name: str) -> bool:
    """True if `name` must be walked forward on one reused instance (see ``STATEFUL_MODELS``)."""
    return name in STATEFUL_MODELS


def require_warm_state(model: VolModel, name: str) -> None:
    """Raise if a stateful model comes back from the registry without its learned state.

    A cold ``ens_inv`` silently serves equal weights (= ``ens_eq``) while the gate scored the
    weighted one; this is the load-time tripwire for that failure.
    """
    if not is_stateful_model(name):
        return
    n = getattr(model, "n_recorded_origins", 0)
    if not isinstance(n, int) or n <= 0:
        raise RuntimeError(
            f"stateful model {name!r} was loaded without its learned state (no recorded "
            "out-of-sample forecasts); refusing to serve equal weights. Retrain it."
        )
