"""Schema of ``configs/production.yaml`` (retrain candidates, promotion gate, monitoring)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from volforecast.models.factory import list_models


class CandidateConfig(BaseModel):
    """A model that competes in the retrain walk-forward (factory key + constructor params)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    params: dict[str, Any] = Field(default_factory=dict)
    refit_every: int = Field(default=5, ge=1)


class RetrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    #: Always evaluated; names unknown to the factory are a configuration error.
    candidates: list[CandidateConfig] = Field(min_length=1)
    #: Name prefixes of *optional* candidates (e.g. ``lgbm``, ``ens_``). Resolved against
    #: ``models.factory.list_models()`` at runtime; absent or failing ones are skipped.
    optional_prefixes: list[str] = Field(default_factory=list)
    optional_refit_every: int = Field(default=20, ge=1)
    #: Number of walk-forward origins: gate window + FHS residual burn-in + margin.
    wf_days: int = Field(default=650, ge=60)
    min_train: int = Field(default=365, ge=30)
    #: Joblib workers for the walk-forward (1 = sequential, in-process).
    n_jobs: int = 1

    @model_validator(mode="after")
    def _known(self) -> RetrainConfig:
        known = set(list_models())
        bad = [c.name for c in self.candidates if c.name not in known]
        if bad:
            raise ValueError(f"unknown candidate models {bad}; available: {sorted(known)}")
        return self


class PromotionConfig(BaseModel):
    """The champion/challenger gate (plan P5.2), evaluated on the last ``window`` origins."""

    model_config = ConfigDict(extra="forbid")

    window: int = Field(default=365, ge=20)
    #: (b) require QLIKE ratio vs HAR < 1 - epsilon.
    epsilon: float = Field(default=0.0, ge=0.0, lt=1.0)
    #: (a) challenger counts as significantly worse than the champion when DM p <= dm_alpha.
    dm_alpha: float = 0.10
    #: (c) distribution of the 99% VaR that is backtested.
    var_dist: Literal["normal", "t", "fhs"] = "fhs"
    var_alpha: float = 0.01
    #: (c) Kupiec / Christoffersen tests pass when p > significance.
    significance: float = 0.05
    min_resid: int = Field(default=250, ge=20)
    #: (c) at least this share of the window must have used the requested VaR distribution.
    min_dist_share: float = 0.9
    #: (d) the refit model's latest forecast must lie within [min/f, max*f] of trailing h-day RV.
    sanity_factor: float = 10.0
    #: Minimum scored origins in the window to take any decision.
    min_eval_obs: int = Field(default=60, ge=10)
    #: When the best candidate *is* the HAR benchmark, check (b) is vacuous and passes.
    allow_benchmark_champion: bool = True
    benchmark: str = "har"


class MonitoringConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rolling_window: int = Field(default=30, ge=5)
    ratio_threshold: float = 1.0
    ratio_consecutive_days: int = Field(default=10, ge=1)
    breach_window: int = Field(default=250, ge=20)
    breach_coverage: float = 0.99
    #: Basel-zone trigger only fires once ``breach_window`` observations exist.
    drift_window: int = Field(default=30, ge=5)
    psi_threshold: float = 0.2
    psi_bins: int = Field(default=10, ge=2)


class ProductionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assets: list[str] = Field(default_factory=lambda: ["BTCUSDT"], min_length=1)
    horizons: list[int] = Field(default_factory=lambda: [1, 5], min_length=1)
    #: Model fit on the full history when no @champion exists (logged loudly).
    fallback_model: str = "log_har"
    benchmark: str = "har"
    retrain: RetrainConfig
    promotion: PromotionConfig = Field(default_factory=PromotionConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)

    @model_validator(mode="after")
    def _check(self) -> ProductionConfig:
        known = set(list_models())
        for name in (self.fallback_model, self.benchmark):
            if name not in known:
                raise ValueError(f"unknown model {name!r}; available: {sorted(known)}")
        if any(h < 1 for h in self.horizons):
            raise ValueError("horizons must be >= 1")
        return self


def load_production_config(path: str | Path) -> ProductionConfig:
    with Path(path).open(encoding="utf-8") as fh:
        raw: Any = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    return ProductionConfig.model_validate(raw)
