"""Hypothesis property tests for volforecast.features.realized."""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from tests.property.helpers import assert_truncation_invariant
from volforecast.contracts import BARS_PER_DAY_5M, PANEL_COLUMNS
from volforecast.features.realized import daily_realized_measures, intraday_log_returns

SETTINGS = settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)


def make_klines(close: np.ndarray, start: str = "2024-03-01") -> pd.DataFrame:
    n = close.size
    return pd.DataFrame(
        {
            "open_time": pd.date_range(start, periods=n, freq="5min", tz="UTC"),
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": np.ones(n),
            "quote_volume": close,
            "n_trades": np.ones(n, dtype="int64"),
        }
    )


@st.composite
def price_paths(draw: st.DrawFn) -> np.ndarray:
    """Random strictly positive price paths covering 2-3 full UTC days."""
    n_days = draw(st.integers(2, 3))
    n = n_days * BARS_PER_DAY_5M
    log_ret = draw(
        hnp.arrays(
            np.float64,
            n,
            elements=st.floats(-0.05, 0.05, allow_nan=False, allow_infinity=False),
        )
    )
    p0 = draw(st.floats(1e-2, 1e6))
    return np.asarray(p0 * np.exp(np.cumsum(log_ret)))


@SETTINGS
@given(price_paths())
def test_nonnegativity_and_semivariance_identity(close: np.ndarray) -> None:
    panel = daily_realized_measures(make_klines(close))
    assert tuple(panel.columns) == PANEL_COLUMNS
    for col in ("rv", "bpv", "jump", "rq", "rs_pos", "rs_neg"):
        assert (panel[col].dropna() >= 0).all(), col
    np.testing.assert_allclose(panel["rs_pos"] + panel["rs_neg"], panel["rv"], rtol=1e-12)
    assert (panel["jump"] <= panel["rv"] + 1e-18).all()


@SETTINGS
@given(price_paths())
def test_daily_return_is_sum_of_intraday_returns(close: np.ndarray) -> None:
    k = make_klines(close)
    panel = daily_realized_measures(k)
    rets = intraday_log_returns(k)
    daily = rets.groupby("date")["log_return"].sum()
    ok = panel["complete"]
    assert ok.iloc[1:].all()  # no gaps: every day but the first is complete
    np.testing.assert_allclose(daily[ok], panel.loc[ok, "ret"], atol=1e-9)


@SETTINGS
@given(price_paths(), st.floats(1e-3, 1e6))
def test_scale_invariance(close: np.ndarray, scale: float) -> None:
    a = daily_realized_measures(make_klines(close))
    b = daily_realized_measures(make_klines(close * scale))
    for col in ("rv", "bpv", "jump", "rs_pos", "rs_neg", "rq", "ret"):
        np.testing.assert_allclose(a[col], b[col], rtol=1e-6, atol=1e-12, err_msg=col)
    assert (a["complete"] == b["complete"]).all()


@SETTINGS
@given(st.floats(1e-3, 1e6), st.integers(2, 3))
def test_constant_price_has_zero_measures(price: float, n_days: int) -> None:
    panel = daily_realized_measures(make_klines(np.full(n_days * BARS_PER_DAY_5M, price)))
    for col in ("rv", "bpv", "jump", "rs_pos", "rs_neg", "rq"):
        assert (panel[col] == 0.0).all(), col
    assert (panel["ret"].iloc[1:] == 0.0).all()
    assert panel["complete"].iloc[1:].all()


@SETTINGS
@given(price_paths(), st.integers(0, 2**32 - 1), st.integers(1, 2))
def test_truncation_invariance_with_random_gaps(close: np.ndarray, seed: int, cut_day: int) -> None:
    rng = np.random.default_rng(seed)
    k = make_klines(close)
    drop = rng.random(len(k)) < 0.02
    s = int(rng.integers(0, len(k) - 12))
    drop[s : s + int(rng.integers(1, 12))] = True
    k = k[~drop].reset_index(drop=True)
    cut = pd.Timestamp("2024-03-01") + pd.Timedelta(days=cut_day)
    assert_truncation_invariant(daily_realized_measures, k, [cut], "open_time")
