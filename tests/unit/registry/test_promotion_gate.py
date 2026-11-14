"""Pure promotion-gate logic (no MLflow): each of the checks (a)-(d) can fail on its own."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.conftest import make_synthetic_panel
from volforecast.registry.config import PromotionConfig
from volforecast.registry.promotion import GateInputs, evaluate_gate

N = 500
CFG = PromotionConfig(window=300, min_resid=100, min_eval_obs=60)


def _panel() -> pd.DataFrame:
    return make_synthetic_panel(N, seed=3, start="2022-01-01")


def _frame(panel: pd.DataFrame, forecast: np.ndarray, h: int = 1) -> pd.DataFrame:
    rv = panel["rv"]
    realized = sum(rv.shift(-k) for k in range(1, h + 1))
    return pd.DataFrame(
        {"origin_date": panel.index, "forecast_var": forecast, "realized_var": realized.to_numpy()}
    )


def _inputs(
    panel: pd.DataFrame,
    f_ch: np.ndarray,
    f_har: np.ndarray,
    f_champ: np.ndarray | None = None,
    h: int = 1,
    **kw: object,
) -> GateInputs:
    return GateInputs(
        asset="BTCUSDT",
        h=h,
        challenger_model="log_har",
        challenger=_frame(panel, f_ch, h),
        har=_frame(panel, f_har, h),
        panel=panel,
        champion=None if f_champ is None else _frame(panel, f_champ, h),
        **kw,  # type: ignore[arg-type]
    )


def _noisy(x: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    return x * np.exp(sigma * np.random.default_rng(seed).standard_normal(x.size))


def _status(res: object, name: str) -> str:
    return next(c.status for c in res.checks if c.name == name)  # type: ignore[attr-defined]


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return _panel()


def test_bootstrap_passes_and_skips_champion_check(panel: pd.DataFrame) -> None:
    rv_next = panel["rv"].shift(-1).bfill().to_numpy()
    res = evaluate_gate(_inputs(panel, _noisy(rv_next, 0.2, 1), _noisy(rv_next, 0.6, 2)), CFG)
    assert res.promoted, res.reasons
    assert _status(res, "a") == "skip"
    assert _status(res, "c") == "pass"


def test_b_fails_when_not_better_than_har(panel: pd.DataFrame) -> None:
    rv_next = panel["rv"].shift(-1).bfill().to_numpy()
    res = evaluate_gate(_inputs(panel, _noisy(rv_next, 0.6, 2), _noisy(rv_next, 0.2, 1)), CFG)
    assert not res.promoted
    assert _status(res, "b") == "fail"


def test_b_is_vacuous_for_the_benchmark_itself(panel: pd.DataFrame) -> None:
    rv_next = panel["rv"].shift(-1).bfill().to_numpy()
    f = rv_next * 1.2
    strict = evaluate_gate(
        _inputs(panel, f, f),
        PromotionConfig(**{**CFG.model_dump(), "allow_benchmark_champion": False}),
    )
    assert _status(strict, "b") == "fail"  # ratio is exactly 1 -> not < 1 - eps
    lenient = evaluate_gate(_inputs(panel, f, f, challenger_is_benchmark=True), CFG)
    assert _status(lenient, "b") == "pass"


def test_a_fails_when_challenger_worse_than_champion(panel: pd.DataFrame) -> None:
    rv_next = panel["rv"].shift(-1).bfill().to_numpy()
    res = evaluate_gate(_inputs(panel, rv_next * 1.5, rv_next * 3.0, f_champ=rv_next * 1.1), CFG)
    assert _status(res, "a") == "fail"
    assert not res.promoted


def test_a_passes_when_equal_to_champion(panel: pd.DataFrame) -> None:
    rv_next = panel["rv"].shift(-1).bfill().to_numpy()
    f = rv_next * 1.3
    res = evaluate_gate(_inputs(panel, f, rv_next * 3.0, f_champ=f), CFG)
    assert _status(res, "a") == "pass"


def test_c_fails_for_underestimated_var(panel: pd.DataFrame) -> None:
    rv_next = panel["rv"].shift(-1).bfill().to_numpy()
    # variance 100x too low -> constant FHS scaling still adapts, so break calibration by
    # making the forecast scale drift: first half correct, second half 10x too low
    f = rv_next.copy()
    f[N // 2 :] *= 0.01
    res = evaluate_gate(_inputs(panel, f, rv_next * 5.0), CFG)
    assert _status(res, "c") == "fail"
    assert not res.promoted


def test_c_is_skipped_for_h5(panel: pd.DataFrame) -> None:
    f = _frame(panel, panel["rv"].to_numpy() * 5, h=5)["realized_var"].bfill().to_numpy()
    res = evaluate_gate(_inputs(panel, f * 1.1, f * 2.0, h=5), CFG)
    assert _status(res, "c") == "skip"
    assert res.promoted, res.reasons


def test_d_fails_on_nonpositive_or_nan_forecasts(panel: pd.DataFrame) -> None:
    rv_next = panel["rv"].shift(-1).bfill().to_numpy()
    f = rv_next.copy()
    f[-10] = -1.0
    res = evaluate_gate(_inputs(panel, f, rv_next * 2.0), CFG)
    assert _status(res, "d") == "fail"
    f2 = rv_next.copy()
    f2[-20] = np.nan
    assert _status(evaluate_gate(_inputs(panel, f2, rv_next * 2.0), CFG), "d") == "fail"


def test_d_sanity_range_on_latest_forecast(panel: pd.DataFrame) -> None:
    rv_next = panel["rv"].shift(-1).bfill().to_numpy()
    res = evaluate_gate(_inputs(panel, rv_next, rv_next * 2.0, latest_forecast=1e3), CFG)
    assert _status(res, "d") == "fail"


def test_too_few_observations_is_a_rejection(panel: pd.DataFrame) -> None:
    rv_next = panel["rv"].shift(-1).bfill().to_numpy()
    res = evaluate_gate(_inputs(panel.iloc[:50], rv_next[:50], rv_next[:50] * 2), CFG)
    assert not res.promoted
    assert any("min_eval_obs" in r for r in res.reasons)
