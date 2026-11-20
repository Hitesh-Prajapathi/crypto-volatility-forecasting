"""ens_inv end to end: retrain -> registry artifact -> load. The served model keeps its weights.

Regression for the original defect: the logged model was a fresh ``build_model(...).fit(...)``,
so a loaded ``ens_inv`` served equal weights (identical to ``ens_eq``) while the gate had scored
the weighted one.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient

from tests.conftest import make_synthetic_panel
from volforecast.flows.common import Runtime
from volforecast.flows.forecast import load_registered
from volforecast.forecast_store import ROLE_CHAMPION, ForecastRepository
from volforecast.models.ensemble import InverseLossEnsemble
from volforecast.models.factory import build_model
from volforecast.registry.client import (
    CHAMPION,
    configure_mlflow,
    registered_model_name,
    set_alias,
)
from volforecast.registry.config import (
    CandidateConfig,
    ProductionConfig,
    PromotionConfig,
    RetrainConfig,
    load_production_config,
)
from volforecast.registry.pyfunc import log_volforecast_model
from volforecast.registry.training import (
    fit_final_model,
    resolve_candidates,
    retrain_one,
    walkforward_candidate_with_model,
)

ASSET = "BTCUSDT"
H = 1
MEMBERS = {"members": ["log_har", "har"]}


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return make_synthetic_panel(900, seed=2)


@pytest.fixture(scope="module")
def cfg() -> ProductionConfig:
    return ProductionConfig(
        assets=[ASSET],
        horizons=[H],
        benchmark="naive",  # clearly worse than the ensemble: ens_inv is the selected model
        retrain=RetrainConfig(
            candidates=[CandidateConfig(name="ens_inv", params=MEMBERS, refit_every=20)],
            wf_days=300,
            min_train=365,
        ),
        promotion=PromotionConfig(window=120, min_eval_obs=60, min_resid=60),
    )


@pytest.fixture
def env(tmp_path: Path) -> Iterator[tuple[MlflowClient, ForecastRepository]]:
    client = configure_mlflow(f"sqlite:///{tmp_path / 'mlflow.db'}", artifact_root=tmp_path / "art")
    repo = ForecastRepository(f"sqlite:///{tmp_path / 'store.db'}")
    yield client, repo
    mlflow.set_tracking_uri("sqlite:///:memory:")


def _loaded(version: str) -> tuple[Any, InverseLossEnsemble]:
    py = mlflow.pyfunc.load_model(f"models:/{registered_model_name(ASSET, H)}/{version}")
    inner = py.unwrap_python_model()
    assert isinstance(inner.model, InverseLossEnsemble)
    return inner, inner.model


def test_retrained_ens_inv_is_served_with_its_learned_weights(
    panel: pd.DataFrame, cfg: ProductionConfig, env: tuple[MlflowClient, ForecastRepository]
) -> None:
    client, repo = env
    out = retrain_one(
        asset=ASSET, h=H, panel=panel, data_hash="d", cfg=cfg, repo=repo, client=client
    )
    assert out.model_name == "ens_inv", out.candidate_qlike
    inner, served = _loaded(str(out.challenger_version))

    # the persisted state is the walk-forward history the gate scored, not an empty one
    meta = inner.metadata["warm_start"]
    assert meta["n_recorded_origins"] == served.n_recorded_origins >= 250
    assert meta["history_end"] == str(panel.index[-1].date()) == str(served.history_end.date())  # type: ignore[union-attr]
    assert meta["last_weights"] == served.last_weights
    w = np.array(list(served.last_weights.values()))
    assert w.sum() == pytest.approx(1.0)
    assert abs(w[0] - w[1]) > 1e-3

    # served forecast at as_of == weighted combination, != the equal-weight (cold) one
    f_served = served.predict_variance(panel, H)
    cold = build_model("ens_inv", **MEMBERS).fit(panel, H)
    f_cold = cold.predict_variance(panel, H)  # no history -> equal weights
    f_eq = build_model("ens_eq", **MEMBERS).fit(panel, H).predict_variance(panel, H)
    assert f_cold == pytest.approx(f_eq, rel=1e-12)
    assert f_served != pytest.approx(f_cold, rel=1e-6)


def test_final_model_continues_the_walked_forward_instance(
    panel: pd.DataFrame, cfg: ProductionConfig
) -> None:
    cand = cfg.retrain.candidates[0]
    fc, walked = walkforward_candidate_with_model(panel, ASSET, H, cand, cfg, "d")
    assert isinstance(walked, InverseLossEnsemble)
    assert walked.n_recorded_origins > 0
    n_before = walked.n_recorded_origins
    final = fit_final_model(cand, panel, H, walked)
    assert final is walked
    assert final.n_recorded_origins == n_before
    assert np.isfinite(fc["forecast_var"]).sum() > 100
    with pytest.raises(RuntimeError, match="warm-start"):
        fit_final_model(cand, panel, H, None)
    # stateless candidates are rebuilt fresh and need no walked instance
    har = fit_final_model(CandidateConfig(name="har"), panel, H, None)
    assert har.predict_variance(panel, H) > 0


def test_forecast_flow_refuses_a_cold_champion(
    panel: pd.DataFrame, env: tuple[MlflowClient, ForecastRepository]
) -> None:
    client, _ = env
    name = registered_model_name(ASSET, H)
    rt = cast(Runtime, SimpleNamespace(client=client))

    def register(model: Any) -> str:
        with mlflow.start_run():
            info = log_volforecast_model(
                model, asset=ASSET, h=H, model_name="ens_inv", panel=panel, data_hash="d"
            )
        version = str(info.registered_model_version)
        set_alias(client, name, CHAMPION, version)
        return version

    cold = build_model("ens_inv", **MEMBERS).fit(panel, H)  # fitted but never walked forward
    register(cold)
    with pytest.raises(RuntimeError, match="learned state"):
        load_registered(rt, ASSET, H, CHAMPION, ROLE_CHAMPION)

    cfg_wf = ProductionConfig(
        assets=[ASSET],
        horizons=[H],
        retrain=RetrainConfig(candidates=[CandidateConfig(name="har")], wf_days=100, min_train=365),
    )
    _, walked = walkforward_candidate_with_model(
        panel,
        ASSET,
        H,
        CandidateConfig(name="ens_inv", params=MEMBERS, refit_every=20),
        cfg_wf,
        "d",
    )
    assert walked is not None
    version = register(walked.fit(panel, H))
    served = load_registered(rt, ASSET, H, CHAMPION, ROLE_CHAMPION)
    assert served is not None
    assert served.version == version
    assert isinstance(served.model, InverseLossEnsemble)
    assert served.model.last_weights == walked.last_weights


def test_ens_inv_is_a_production_candidate() -> None:
    prod = load_production_config(Path(__file__).parents[3] / "configs" / "production.yaml")
    names = {c.name for c in resolve_candidates(prod)}
    assert {"ens_eq", "ens_inv"} <= names
