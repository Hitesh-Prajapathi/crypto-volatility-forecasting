"""Tests for volforecast.evaluation.losses."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from volforecast.evaluation.losses import loss_frame, mae_vol, mse, qlike, summary_table


def test_qlike_zero_at_perfect_forecast() -> None:
    rv = np.array([1e-4, 4e-4, 9e-4])
    np.testing.assert_allclose(qlike(rv, rv), 0.0, atol=1e-15)


def test_qlike_formula() -> None:
    rv, h = 2.0, 4.0
    expected = rv / h - np.log(rv / h) - 1.0
    assert qlike(rv, h) == pytest.approx(expected)


@pytest.mark.parametrize("rv", [1e-4, 1.0, 37.5])
def test_qlike_minimized_at_rv_numerically(rv: float) -> None:
    grid = rv * np.exp(np.linspace(-1.5, 1.5, 3001))  # contains h = rv at the centre
    losses = qlike(np.full_like(grid, rv), grid)
    assert grid[np.argmin(losses)] == pytest.approx(rv)
    assert np.all(losses >= -1e-12)
    assert losses.min() == pytest.approx(0.0, abs=1e-12)


def test_qlike_asymmetric_penalises_under_prediction_more() -> None:
    rv = 1.0
    assert qlike(rv, 0.5) > qlike(rv, 2.0)


@pytest.mark.parametrize("func", [qlike, mse, mae_vol])
@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
def test_errors_on_invalid_forecast(func, bad: float) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match=r"\bh\b"):
        func(np.array([1.0, 1.0]), np.array([1.0, bad]))


@pytest.mark.parametrize("func", [qlike, mse, mae_vol])
@pytest.mark.parametrize("bad", [0.0, -1e-6, np.nan])
def test_errors_on_invalid_rv(func, bad: float) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match="rv"):
        func(np.array([1.0, bad]), np.array([1.0, 1.0]))


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="shapes"):
        qlike(np.ones(3), np.ones(4))


def test_mse_and_mae_vol_values() -> None:
    rv, h = np.array([4.0, 9.0]), np.array([1.0, 16.0])
    np.testing.assert_allclose(mse(rv, h), [9.0, 49.0])
    np.testing.assert_allclose(mae_vol(rv, h), [1.0, 1.0])


def _frame() -> tuple[pd.Series, pd.DataFrame]:
    rng = np.random.default_rng(1)
    idx = pd.date_range("2024-01-01", periods=300, freq="D", name="date")
    truth = np.exp(rng.normal(-8, 0.5, 300))
    rv = pd.Series(truth * np.exp(rng.normal(0, 0.2, 300)), index=idx)
    fc = pd.DataFrame(
        {
            "good": truth,
            "bad": truth * 2.0,
            "bench": np.full(300, truth.mean()),
        },
        index=idx,
    )
    return rv, fc


def test_loss_frame_columns_and_values() -> None:
    rv, fc = _frame()
    lf = loss_frame(rv, fc)
    assert list(lf.columns) == ["good", "bad", "bench"]
    assert lf.index.equals(rv.index)
    np.testing.assert_allclose(lf["good"].to_numpy(), qlike(rv.to_numpy(), fc["good"].to_numpy()))
    np.testing.assert_allclose(
        loss_frame(rv, fc, "mse")["bad"].to_numpy(), mse(rv.to_numpy(), fc["bad"].to_numpy())
    )
    with pytest.raises(ValueError, match="unknown loss"):
        loss_frame(rv, fc, "nope")  # type: ignore[arg-type]


def test_loss_frame_aligns_on_index() -> None:
    rv, fc = _frame()
    lf = loss_frame(rv.iloc[10:], fc.iloc[:250])
    assert lf.index.equals(rv.index[10:250])


def test_summary_table() -> None:
    rv, fc = _frame()
    tab = summary_table(rv, fc, benchmark="bench")
    assert set(tab.columns) == {"qlike", "mse", "mae_vol", "qlike_ratio"}
    assert tab.loc["bench", "qlike_ratio"] == pytest.approx(1.0)
    assert tab.loc["good", "qlike_ratio"] < 1.0 < tab.loc["bad", "qlike_ratio"]
    assert tab.loc["good", "qlike"] == pytest.approx(
        qlike(rv.to_numpy(), fc["good"].to_numpy()).mean()
    )
    with pytest.raises(ValueError, match="benchmark"):
        summary_table(rv, fc, benchmark="missing")
