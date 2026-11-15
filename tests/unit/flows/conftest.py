"""Fixtures shared by the flow tests: tiny configs, synthetic data roots, temp stores."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import mlflow
import pandas as pd
import pytest
import yaml

from tests.conftest import make_synthetic_klines, make_synthetic_panel
from volforecast.config import Settings
from volforecast.data.store import write_klines
from volforecast.flows.common import Runtime
from volforecast.forecast_store import ForecastRepository
from volforecast.registry.config import load_production_config

ASSET = "BTCUSDT"

#: Small-sample production config: same schema as configs/production.yaml, windows shrunk so the
#: gate and the walk-forward run on a few hundred synthetic days.
TINY_CONFIG = {
    "assets": [ASSET],
    "horizons": [1],
    "fallback_model": "log_har",
    "benchmark": "har",
    "retrain": {
        "candidates": [{"name": "log_har", "refit_every": 30}, {"name": "har", "refit_every": 30}],
        "optional_prefixes": [],
        "wf_days": 150,
        "min_train": 60,
    },
    "promotion": {"window": 60, "min_resid": 30, "min_eval_obs": 30},
    "monitoring": {
        "rolling_window": 10,
        "ratio_consecutive_days": 3,
        "breach_window": 40,
        "drift_window": 20,
    },
}


@pytest.fixture(autouse=True)
def _isolated_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Flows built from the environment must never write MLflow artifacts or drift reports
    into the repository's ``artifacts/`` directory."""
    monkeypatch.setenv("VOLF_ARTIFACTS_DIR", str(tmp_path / "artifacts"))


@pytest.fixture
def tiny_config(tmp_path: Path) -> Path:
    path = tmp_path / "production.yaml"
    path.write_text(yaml.safe_dump(TINY_CONFIG))
    return path


@pytest.fixture
def synthetic_panel_400() -> pd.DataFrame:
    return make_synthetic_panel(400, seed=3, start="2024-01-01")


@pytest.fixture
def runtime(
    tmp_path: Path, tiny_config: Path, synthetic_panel_400: pd.DataFrame
) -> Iterator[Runtime]:
    """Runtime over a temp sqlite store and file MLflow store, panel served from memory."""
    settings = Settings(
        data_dir=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
        database_url=f"sqlite:///{tmp_path / 'store.db'}",
        mlflow_tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        production_config=tiny_config,
    )
    panel = synthetic_panel_400

    def loader(asset: str) -> tuple[pd.DataFrame, str]:
        return panel, "synthetic-400"

    rt = Runtime(
        settings=settings,
        cfg=load_production_config(tiny_config),
        repo=ForecastRepository(settings.database_url),
        panel_loader=loader,
    )
    yield rt
    mlflow.set_tracking_uri("sqlite:///:memory:")


def write_synthetic_data_root(root: Path, n_days: int, seed: int = 3) -> None:
    """Hive-partitioned 5-minute klines for ``n_days`` synthetic days starting 2024-01-01."""
    write_klines(make_synthetic_klines(n_days, seed=seed, start="2024-01-01"), root, ASSET)


@pytest.fixture(scope="session")
def prefect_server() -> Iterator[None]:
    """Temporary Prefect API (only the tests that call a ``@flow`` need it)."""
    from prefect.testing.utilities import prefect_test_harness

    with prefect_test_harness():
        yield


def seed_champion(rt: Runtime, panel: pd.DataFrame, end: int, name: str = "log_har") -> str:
    """Register ``name`` fit on ``panel.iloc[:end]`` as ``@champion`` (h=1); returns the version.

    The stored "out-of-sample" forecasts are a trailing-mean RV rule: they only provide the
    standardized-residual history the forecast flow needs for FHS VaR.
    """
    from volforecast.models.factory import build_model
    from volforecast.registry.client import (
        CHAMPION,
        registered_model_name,
        set_alias,
        tag_version,
    )
    from volforecast.registry.pyfunc import log_volforecast_model

    pnl = panel.iloc[:end]
    model = build_model(name).fit(pnl, 1)
    oos = pnl["rv"].rolling(5).mean().dropna().iloc[-120:]
    oos_frame = pd.DataFrame({"origin_date": oos.index, "forecast_var": oos.to_numpy()})
    rt.client  # noqa: B018  (selects the tracking/registry URI and experiment)
    with mlflow.start_run(run_name="seed"):
        info = log_volforecast_model(
            model, asset=ASSET, h=1, model_name=name, panel=pnl, data_hash="seed", oos=oos_frame
        )
    version = str(info.registered_model_version)
    reg = registered_model_name(ASSET, 1)
    tag_version(
        rt.client,
        reg,
        version,
        {
            "model_name": name,
            "model_params": "{}",
            "refit_every": 30,
            "train_end_date": pnl.index[-1].date(),
        },
    )
    set_alias(rt.client, reg, CHAMPION, version)
    return version
