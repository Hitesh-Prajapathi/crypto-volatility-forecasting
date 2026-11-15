"""Forecast flow: fallback vs champion, idempotent upsert, VaR measures, replay."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.unit.flows.conftest import ASSET, seed_champion, write_synthetic_data_root
from volforecast.flows.common import Runtime
from volforecast.flows.forecast import (
    forecast_asset,
    forecast_flow,
    risk_measures,
    standardized_residuals,
)
from volforecast.flows.forecast import main as forecast_main
from volforecast.forecast_store import (
    BENCHMARK_MODEL_NAME,
    ROLE_BENCHMARK,
    ROLE_CHAMPION,
    ROLE_FALLBACK,
)


def test_without_champion_the_fallback_model_is_served(
    runtime: Runtime, synthetic_panel_400: pd.DataFrame
) -> None:
    out = forecast_asset(runtime, ASSET)
    last = synthetic_panel_400.index[-1].date()
    assert out["origin_date"] == str(last)
    latest = runtime.repo.latest_forecast(ASSET, 1)
    assert latest is not None
    assert latest["role"] == ROLE_FALLBACK
    assert latest["model_version"] == "fallback:log_har"
    assert latest["origin_date"] == last
    assert latest["target_date"] == last + pd.Timedelta(days=1)
    assert latest["forecast_var"] > 0
    assert latest["dist"] == "normal"  # no residual history for a fallback
    assert 0 < latest["var95"] < latest["var99"] < latest["es975"] * 2
    bench = runtime.repo.forecasts_df(ASSET, 1, roles=(ROLE_BENCHMARK,))
    assert list(bench["model_name"]) == [BENCHMARK_MODEL_NAME]
    assert runtime.repo.last_realized_date(ASSET) == last


def test_champion_is_loaded_by_alias_and_rerun_is_idempotent(
    runtime: Runtime, synthetic_panel_400: pd.DataFrame
) -> None:
    version = seed_champion(runtime, synthetic_panel_400, 300)
    forecast_asset(runtime, ASSET, replay_days=5)
    first = runtime.repo.forecasts_df(ASSET, 1)
    assert len(first) == 10  # 5 origins x (champion + HAR benchmark)
    champ = first[first["role"] == ROLE_CHAMPION]
    assert set(champ["model_version"]) == {version}
    assert set(champ["model_name"]) == {"log_har"}
    assert set(champ["dist"]) == {"fhs"}  # 120 stored OOS residuals >= min_resid (30)
    assert (champ["var99"] > champ["var95"]).all()

    forecast_asset(runtime, ASSET, replay_days=5)
    again = runtime.repo.forecasts_df(ASSET, 1)
    assert len(again) == len(first)
    assert not again.duplicated(["model_version", "origin_date", "role"]).any()
    assert np.allclose(
        again.sort_values(["role", "origin_date"])["forecast_var"],
        first.sort_values(["role", "origin_date"])["forecast_var"],
    )


def test_as_of_limits_the_information_set(
    runtime: Runtime, synthetic_panel_400: pd.DataFrame
) -> None:
    as_of = str(synthetic_panel_400.index[-11].date())
    out = forecast_asset(runtime, ASSET, as_of=as_of)
    assert out["origin_date"] == as_of
    assert runtime.repo.forecasts_df(ASSET, 1)["origin_date"].max() == pd.Timestamp(as_of)


def test_risk_measures_normal_vs_fhs() -> None:
    f = 4e-4
    normal = risk_measures(f, np.empty(0), min_resid=50)
    assert normal["dist"] == "normal"
    assert normal["var99"] == pytest.approx(2.3263479 * np.sqrt(f), rel=1e-6)
    z = np.random.default_rng(0).standard_t(4, 2000)
    fhs = risk_measures(f, z, min_resid=50)
    assert fhs["dist"] == "fhs"
    assert fhs["var99"] == pytest.approx(-np.quantile(z, 0.01) * np.sqrt(f), rel=1e-9)
    assert fhs["es975"] > fhs["var95"]  # type: ignore[operator]


def test_standardized_residuals_use_only_returns_known_at_the_origin(
    synthetic_panel_400: pd.DataFrame,
) -> None:
    panel = synthetic_panel_400
    oos = pd.DataFrame({"origin_date": panel.index[100:200], "forecast_var": 4e-4})
    origin = panel.index[150]
    z = standardized_residuals(oos, panel, origin)
    # forecasts made at s <= origin - 1 have their next-day return known at the origin
    assert len(z) == 50
    expected = panel["ret"].iloc[101:151].to_numpy() / np.sqrt(4e-4)
    assert np.allclose(z, expected)


def test_forecast_flow_and_cli_end_to_end(
    tmp_path: Path, tiny_config: Path, prefect_server: None
) -> None:
    root = tmp_path / "data"
    write_synthetic_data_root(root, 200)
    kw = {
        "data_dir": str(root),
        "database_url": f"sqlite:///{tmp_path / 'store.db'}",
        "tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}",
        "config_path": str(tiny_config),
    }
    res = forecast_flow(**kw)
    assert res["results"][0]["asset"] == ASSET
    from volforecast.forecast_store import ForecastRepository

    repo = ForecastRepository(kw["database_url"])
    runs = repo.pipeline_runs("forecast")
    assert [r["status"] for r in runs] == ["success"]
    n = len(repo.forecasts_df(ASSET, 1))
    code = forecast_main(
        [
            "--data-dir", kw["data_dir"], "--database-url", kw["database_url"],
            "--tracking-uri", kw["tracking_uri"], "--config", kw["config_path"],
        ]
    )  # fmt: skip
    assert code == 0
    assert len(repo.forecasts_df(ASSET, 1)) == n  # idempotent
    assert [r["status"] for r in repo.pipeline_runs("forecast")] == ["success", "success"]


def test_cli_exits_nonzero_and_records_failure_when_data_is_missing(
    tmp_path: Path, tiny_config: Path, prefect_server: None
) -> None:
    url = f"sqlite:///{tmp_path / 'store.db'}"
    code = forecast_main(
        [
            "--data-dir", str(tmp_path / "empty"), "--database-url", url,
            "--tracking-uri", f"sqlite:///{tmp_path / 'mlflow.db'}",
            "--config", str(tiny_config),
        ]
    )  # fmt: skip
    assert code == 1
    from volforecast.forecast_store import ForecastRepository

    run = ForecastRepository(url).pipeline_runs("forecast")[0]
    assert run["status"] == "failed"
    assert run["detail"]["failures"]
