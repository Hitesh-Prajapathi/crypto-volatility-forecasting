"""Name -> model constructor registry for the econometric baselines."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from volforecast.contracts import VolModel
from volforecast.models.ewma import EWMARealized, EWMAReturns
from volforecast.models.garch import GarchModel
from volforecast.models.har import HAR, HARCJ, HARQ, SHAR, LogHAR
from volforecast.models.naive import NaiveRV


def _garch(**params: Any) -> VolModel:
    return GarchModel(**params)


def _gjr_garch(**params: Any) -> VolModel:
    params.setdefault("o", 1)
    return GarchModel(**params)


MODEL_FACTORY: dict[str, Callable[..., VolModel]] = {
    "naive": NaiveRV,
    "ewma": EWMAReturns,
    "ewma_rv": EWMARealized,
    "garch": _garch,
    "gjr_garch": _gjr_garch,
    "har": HAR,
    "log_har": LogHAR,
    "har_cj": HARCJ,
    "shar": SHAR,
    "harq": HARQ,
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
