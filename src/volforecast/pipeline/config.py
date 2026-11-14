"""Pydantic schema for the backtest configuration (``configs/backtest.yaml``)."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from volforecast.models.factory import list_models


class ModelConfig(BaseModel):
    """One model to evaluate: factory key, refit cadence (in origins) and constructor params."""

    model_config = ConfigDict(extra="forbid")

    name: str
    refit_every: int = Field(default=1, ge=1)
    params: dict[str, Any] = Field(default_factory=dict)
    version: str = "dev"

    @field_validator("name")
    @classmethod
    def _known_model(cls, v: str) -> str:
        if v not in list_models():
            raise ValueError(f"unknown model {v!r}; available: {list_models()}")
        return v


DistName = Literal["normal", "t", "fhs", "normal_cal", "t_cal"]
DEFAULT_DISTS: tuple[DistName, ...] = ("normal", "t", "fhs")


class VarConfig(BaseModel):
    """1-day VaR/ES application settings (plan M4)."""

    model_config = ConfigDict(extra="forbid")

    alphas: list[float] = Field(default_factory=lambda: [0.05, 0.01], min_length=1)
    dists: list[DistName] = Field(default_factory=lambda: list(DEFAULT_DISTS), min_length=1)
    min_resid: int = Field(default=250, ge=20)

    @field_validator("alphas")
    @classmethod
    def _alphas_in_unit_interval(cls, v: list[float]) -> list[float]:
        if any(not 0.0 < a < 1.0 for a in v):
            raise ValueError("alphas must lie in (0, 1)")
        return v


class BacktestConfig(BaseModel):
    """Full backtest protocol. Dates are tz-naive UTC days."""

    model_config = ConfigDict(extra="forbid")

    assets: list[str] = Field(min_length=1)
    horizons: list[int] = Field(default_factory=lambda: [1, 5], min_length=1)
    first_origin: date = date(2021, 1, 1)
    last_origin: date | None = None
    min_train: int = Field(default=365, ge=30)
    window: Literal["expanding", "rolling"] = "expanding"
    rolling_size: int | None = Field(default=None, ge=30)
    models: list[ModelConfig] = Field(min_length=1)
    benchmark: str = "har"
    var: VarConfig = Field(default_factory=VarConfig)
    n_jobs: int = -1
    mcs_reps: int = Field(default=1000, ge=100)
    seed: int = 0

    @field_validator("assets")
    @classmethod
    def _unique_assets(cls, v: list[str]) -> list[str]:
        if len(set(v)) != len(v):
            raise ValueError("assets must be unique")
        return v

    @field_validator("horizons")
    @classmethod
    def _valid_horizons(cls, v: list[int]) -> list[int]:
        if any(h < 1 for h in v) or len(set(v)) != len(v):
            raise ValueError("horizons must be unique integers >= 1")
        return v

    @model_validator(mode="after")
    def _consistent(self) -> BacktestConfig:
        names = [m.name for m in self.models]
        if len(set(names)) != len(names):
            raise ValueError("model names must be unique")
        if self.benchmark not in names:
            raise ValueError(f"benchmark {self.benchmark!r} is not in the model list {names}")
        if self.window == "rolling" and self.rolling_size is None:
            raise ValueError("window='rolling' requires rolling_size")
        if self.last_origin is not None and self.last_origin < self.first_origin:
            raise ValueError("last_origin must be >= first_origin")
        return self

    @property
    def model_names(self) -> list[str]:
        return [m.name for m in self.models]


def load_config(path: str | Path) -> BacktestConfig:
    """Parse and validate a YAML config file."""
    with Path(path).open(encoding="utf-8") as fh:
        raw: Any = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    return BacktestConfig.model_validate(raw)
