from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from volforecast.pipeline.config import BacktestConfig, load_config

REPO = Path(__file__).resolve().parents[3]


def _base() -> dict[str, Any]:
    return {"assets": ["BTCUSDT"], "models": [{"name": "har"}, {"name": "naive"}]}


def test_shipped_config_is_valid_and_matches_plan() -> None:
    cfg = load_config(REPO / "configs" / "backtest.yaml")
    assert cfg.horizons == [1, 5]
    assert str(cfg.first_origin) == "2021-01-01"
    assert cfg.min_train == 365
    assert cfg.window == "expanding"
    assert cfg.benchmark == "har"
    refit = {m.name: m.refit_every for m in cfg.models}
    assert refit["garch"] == refit["gjr_garch"] == 7
    assert all(refit[n] == 1 for n in ("har", "log_har", "har_cj", "shar", "harq"))
    assert refit["lgbm"] == refit["xgb"] == refit["ens_eq"] == 7
    assert refit["ens_inv"] >= 10_000  # stateful: fitted once, members refit internally
    assert cfg.var.alphas == [0.05, 0.01]
    assert cfg.var.dists == ["normal", "normal_cal", "t", "t_cal", "fhs"]
    assert cfg.var.min_resid == 250


def test_defaults_applied() -> None:
    cfg = BacktestConfig.model_validate(_base())
    assert cfg.min_train == 365
    assert cfg.window == "expanding"
    assert cfg.benchmark == "har"
    assert cfg.model_names == ["har", "naive"]


@pytest.mark.parametrize(
    ("patch", "msg"),
    [
        ({"models": [{"name": "nope"}]}, "unknown model"),
        ({"benchmark": "ewma"}, "benchmark"),
        ({"models": [{"name": "har"}, {"name": "har"}]}, "unique"),
        ({"horizons": [0]}, "horizons"),
        ({"horizons": [1, 1]}, "horizons"),
        ({"assets": []}, "at least 1"),
        ({"assets": ["A", "A"]}, "unique"),
        ({"window": "rolling"}, "rolling_size"),
        ({"var": {"alphas": [1.5]}}, "alphas"),
        ({"var": {"dists": ["laplace"]}}, "dists"),
        ({"models": [{"name": "har", "refit_every": 0}]}, "refit_every"),
        ({"first_origin": "2021-01-01", "last_origin": "2020-01-01"}, "last_origin"),
        ({"unexpected": 1}, "unexpected"),
    ],
)
def test_invalid_configs_rejected(patch: dict[str, Any], msg: str) -> None:
    with pytest.raises(ValidationError, match=msg):
        BacktestConfig.model_validate({**_base(), **patch})


def test_load_config_rejects_non_mapping(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("- just\n- a list\n")
    with pytest.raises(ValueError, match="mapping"):
        load_config(p)
