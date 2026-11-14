"""MLflow-backed registry behaviour on a tmp sqlite tracking store (no network, no Docker)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient

from tests.conftest import make_synthetic_panel
from volforecast.forecast_store import ForecastRepository
from volforecast.models.factory import build_model
from volforecast.registry import promotion as promotion_mod
from volforecast.registry.client import (
    CHAMPION,
    alias_version,
    configure_mlflow,
    model_uri,
    registered_model_name,
)
from volforecast.registry.config import (
    CandidateConfig,
    ProductionConfig,
    PromotionConfig,
    RetrainConfig,
    load_production_config,
)
from volforecast.registry.pyfunc import log_volforecast_model, panel_to_frame
from volforecast.registry.tracking import log_backtest_run, summarize_forecasts
from volforecast.registry.training import resolve_candidates, retrain_one

ASSET = "BTCUSDT"


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return make_synthetic_panel(1500, seed=0)


@pytest.fixture(scope="module")
def cfg() -> ProductionConfig:
    return ProductionConfig(
        assets=[ASSET],
        horizons=[1],
        retrain=RetrainConfig(
            candidates=[CandidateConfig(name=n, refit_every=20) for n in ("log_har", "harq", "har")]
        ),
        promotion=PromotionConfig(),
    )


@pytest.fixture
def env(tmp_path: Path) -> Iterator[tuple[MlflowClient, ForecastRepository]]:
    client = configure_mlflow(f"sqlite:///{tmp_path / 'mlflow.db'}", artifact_root=tmp_path / "art")
    repo = ForecastRepository(f"sqlite:///{tmp_path / 'store.db'}")
    yield client, repo
    mlflow.set_tracking_uri("sqlite:///:memory:")


def test_registered_model_name_is_lowercase() -> None:
    assert registered_model_name("BTCUSDT", 5) == "volforecast-btcusdt-h5"
    assert model_uri("ETHUSDT", 1, "champion") == "models:/volforecast-ethusdt-h1@champion"


@pytest.mark.parametrize("name", ["log_har", "harq", "har"])
def test_pyfunc_roundtrip_matches_direct_prediction(
    name: str, panel: pd.DataFrame, env: tuple[MlflowClient, ForecastRepository]
) -> None:
    client, _ = env
    h = 5
    model = build_model(name).fit(panel.iloc[:600], h)
    direct = model.predict_variance(panel.iloc[:600], h)
    oos = pd.DataFrame({"origin_date": panel.index[400:600], "forecast_var": 1e-3})
    with mlflow.start_run():
        info = log_volforecast_model(
            model,
            asset=ASSET,
            h=h,
            model_name=name,
            panel=panel.iloc[:600],
            data_hash="h0",
            oos=oos,
        )
    loaded = mlflow.pyfunc.load_model(
        f"models:/{registered_model_name(ASSET, h)}/{info.registered_model_version}"
    )
    out = loaded.predict(panel_to_frame(panel.iloc[:600]))
    assert float(out["forecast_var"].iloc[0]) == pytest.approx(direct, rel=1e-12)
    inner = loaded.unwrap_python_model()
    assert inner.metadata["asset"] == ASSET
    assert inner.metadata["train_end_date"] == str(panel.index[599].date())
    assert len(inner.oos_forecasts) == 200
    assert loaded.metadata.signature is not None
    assert loaded.metadata.saved_input_example_info is not None
    # alias loading works
    client.set_registered_model_alias(
        registered_model_name(ASSET, h), "champion", str(info.registered_model_version)
    )
    again = mlflow.pyfunc.load_model(model_uri(ASSET, h, "champion"))
    assert float(
        again.predict(panel_to_frame(panel.iloc[:600]))["forecast_var"].iloc[0]
    ) == pytest.approx(direct)


def test_log_backtest_run_records_metrics_and_provenance(
    panel: pd.DataFrame, env: tuple[MlflowClient, ForecastRepository]
) -> None:
    client, _ = env
    fc = pd.DataFrame(
        {
            "asset": ASSET,
            "model": "har",
            "horizon": 1,
            "origin_date": panel.index[:100],
            "target_date": panel.index[:100] + pd.Timedelta(days=1),
            "forecast_var": panel["rv"].to_numpy()[:100] * 1.1,
            "realized_var": panel["rv"].to_numpy()[:100],
            "model_version": "dev",
            "data_hash": "abc",
        }
    )
    assert summarize_forecasts(fc).loc[0, "qlike"] > 0
    with mlflow.start_run() as parent:
        pass
    rid = log_backtest_run(fc, config={"a": {"b": 1}}, data_hashes={ASSET: "abc"}, run_name="bt")
    run = client.get_run(rid)
    assert "git_sha" in run.data.tags
    assert run.data.params["data_hash.BTCUSDT"] == "abc"
    assert run.data.params["config.a.b"] == "1"
    assert f"{ASSET}.har.h1.qlike" in run.data.metrics
    arts = {a.path for a in client.list_artifacts(rid)}
    assert {"forecasts.parquet", "summary.csv"} <= arts
    assert parent.info.run_id != rid


def test_retrain_bootstraps_champion_then_gates_against_it(
    panel: pd.DataFrame, cfg: ProductionConfig, env: tuple[MlflowClient, ForecastRepository]
) -> None:
    client, repo = env
    name = registered_model_name(ASSET, 1)
    first = retrain_one(
        asset=ASSET, h=1, panel=panel, data_hash="d1", cfg=cfg, repo=repo, client=client
    )
    assert first.status == "promoted", first.reasons
    champ = alias_version(client, name, CHAMPION)
    assert champ is not None
    assert str(champ.version) == "1"

    # idempotent for the same as-of date ...
    skipped = retrain_one(
        asset=ASSET, h=1, panel=panel, data_hash="d1", cfg=cfg, repo=repo, client=client
    )
    assert skipped.status == "skipped"
    # ... but forced reruns evaluate against the existing champion (check (a) is now active)
    second = retrain_one(
        asset=ASSET, h=1, panel=panel, data_hash="d1", cfg=cfg, repo=repo, client=client, force=True
    )
    assert any(r.startswith("(a) ") and "SKIP" not in r for r in second.reasons)

    decisions = repo.promotion_decisions(ASSET, 1)
    assert len(decisions) == 2
    assert decisions[-1]["champion_version"] is None
    assert decisions[-1]["promoted"] is True
    runs = mlflow.search_runs(
        experiment_names=["volforecast"],
        filter_string="tags.promotion_decision != ''",
        output_format="pandas",
    )
    assert len(runs) == 2
    assert set(runs["tags.promotion_decision"]) <= {"promoted", "rejected"}
    mv = client.get_model_version(name, "1")
    assert mv.tags["model_name"] in {"log_har", "harq", "har"}
    assert mv.tags["promotion_decision"] == "promoted"


def test_alias_move_failure_rolls_back_the_decision_row(
    panel: pd.DataFrame,
    cfg: ProductionConfig,
    env: tuple[MlflowClient, ForecastRepository],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, repo = env

    def boom(*_: Any, **__: Any) -> None:
        raise RuntimeError("registry down")

    monkeypatch.setattr(promotion_mod, "set_alias", boom)
    with pytest.raises(RuntimeError, match="registry down"):
        retrain_one(
            asset=ASSET, h=1, panel=panel, data_hash="d1", cfg=cfg, repo=repo, client=client
        )
    assert repo.promotion_decisions(ASSET, 1) == []
    assert alias_version(client, registered_model_name(ASSET, 1), CHAMPION) is None


def test_resolve_candidates_adds_optional_models_only_if_present(cfg: ProductionConfig) -> None:
    base = {c.name for c in resolve_candidates(cfg)}
    assert {"log_har", "harq", "har"} <= base
    with_opt = cfg.model_copy(deep=True)
    with_opt.retrain.optional_prefixes = ["naive"]  # an existing factory model
    assert "naive" in {c.name for c in resolve_candidates(with_opt)}
    with_none = cfg.model_copy(deep=True)
    with_none.retrain.optional_prefixes = ["does_not_exist_"]
    assert {c.name for c in resolve_candidates(with_none)} == base


def test_shipped_production_yaml_parses() -> None:
    cfg = load_production_config(Path(__file__).parents[3] / "configs" / "production.yaml")
    assert [c.name for c in cfg.retrain.candidates] == ["log_har", "harq", "har"]
    assert cfg.promotion.window == 365
    assert np.isclose(cfg.promotion.epsilon, 0.0)
