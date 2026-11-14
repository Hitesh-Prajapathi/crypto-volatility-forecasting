"""Leakage guards: the most important tests in the repo.

Every feature builder must be causal: removing data from time T onwards must not change any
output row dated before T. Targets are forward-looking by construction and are checked by
explicit alignment instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.conftest import make_synthetic_klines, make_synthetic_panel
from tests.property.helpers import assert_truncation_invariant
from volforecast.contracts import BARS_PER_DAY_5M, DEFAULT_HORIZONS, target_column
from volforecast.features.har import har_aggregates, make_targets
from volforecast.features.realized import daily_realized_measures

START = pd.Timestamp("2024-01-01")


def gappy_klines(n_days: int = 30, seed: int = 0) -> pd.DataFrame:
    """Synthetic klines with scattered, clustered, midnight-straddling and whole-day gaps."""
    rng = np.random.default_rng(seed)
    k = make_synthetic_klines(n_days=n_days, seed=seed)
    drop = rng.random(len(k)) < 0.01
    for _ in range(10):
        s = int(rng.integers(0, len(k) - 12))
        drop[s : s + int(rng.integers(1, 10))] = True
    for d in (5, 11, 17):  # gaps crossing midnight of day d (1..6 bars on each side)
        if d + 1 >= n_days:
            continue
        a, b = rng.integers(1, 7, size=2)
        drop[(d + 1) * BARS_PER_DAY_5M - a : (d + 1) * BARS_PER_DAY_5M + b] = True
    drop[8 * BARS_PER_DAY_5M : 9 * BARS_PER_DAY_5M] = True  # a whole missing day
    return k[~drop].reset_index(drop=True)


# ── the helper itself must be able to fail ───────────────────────────────────


def test_helper_detects_forward_shift(synthetic_panel: pd.DataFrame) -> None:
    def leaky(p: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"x": p["rv"].shift(-1)}, index=p.index)

    with pytest.raises(AssertionError, match="look-ahead leakage"):
        assert_truncation_invariant(leaky, synthetic_panel, [synthetic_panel.index[300]])


def test_helper_detects_centered_window(synthetic_panel: pd.DataFrame) -> None:
    def leaky(p: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"x": p["rv"].rolling(5, center=True, min_periods=1).mean()})

    with pytest.raises(AssertionError, match="look-ahead leakage"):
        assert_truncation_invariant(leaky, synthetic_panel, [synthetic_panel.index[300]])


def test_helper_detects_global_normalisation(synthetic_panel: pd.DataFrame) -> None:
    def leaky(p: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"z": (p["rv"] - p["rv"].mean()) / p["rv"].std()})

    with pytest.raises(AssertionError, match="look-ahead leakage"):
        assert_truncation_invariant(leaky, synthetic_panel, [synthetic_panel.index[300]])


def test_helper_rejects_vacuous_cut(synthetic_panel: pd.DataFrame) -> None:
    with pytest.raises(AssertionError, match="vacuous"):
        assert_truncation_invariant(
            lambda p: p[["rv"]], synthetic_panel, [synthetic_panel.index[0]]
        )


def test_helper_accepts_causal_function(synthetic_panel: pd.DataFrame) -> None:
    def causal(p: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"x": p["rv"].expanding().mean()})

    assert_truncation_invariant(causal, synthetic_panel, [synthetic_panel.index[300]])


# ── realized measures ────────────────────────────────────────────────────────


def test_daily_realized_measures_truncation_invariant() -> None:
    k = gappy_klines()
    cuts = [START + pd.Timedelta(days=d) for d in (2, 6, 9, 10, 12, 18, 25, 29)]
    assert_truncation_invariant(daily_realized_measures, k, cuts, "open_time")


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_daily_realized_measures_truncation_invariant_random_gaps(seed: int) -> None:
    k = gappy_klines(n_days=14, seed=seed)
    cuts = [START + pd.Timedelta(days=d) for d in range(2, 14)]
    assert_truncation_invariant(daily_realized_measures, k, cuts, "open_time")


def test_daily_realized_measures_truncation_invariant_tz_aware_cut() -> None:
    k = gappy_klines(n_days=10)
    cuts = [pd.Timestamp("2024-01-05", tz="UTC"), pd.Timestamp("2024-01-08", tz="UTC")]
    assert_truncation_invariant(daily_realized_measures, k, cuts, "open_time")


# ── HAR aggregates ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("log", [False, True])
def test_har_aggregates_truncation_invariant(synthetic_panel: pd.DataFrame, log: bool) -> None:
    cuts = [synthetic_panel.index[i] for i in (30, 100, 400, 1000, 1499)]
    assert_truncation_invariant(lambda p: har_aggregates(p, log=log), synthetic_panel, cuts)


def test_har_aggregates_on_realized_panel_truncation_invariant() -> None:
    panel = daily_realized_measures(gappy_klines())
    cuts = [START + pd.Timedelta(days=d) for d in (5, 10, 20, 28)]
    assert_truncation_invariant(lambda p: har_aggregates(p, col="rv"), panel, cuts)


# ── targets: forward-looking by design, checked by alignment ────────────────


@pytest.mark.parametrize("h", DEFAULT_HORIZONS)
def test_make_targets_alignment(synthetic_panel: pd.DataFrame, h: int) -> None:
    tg = make_targets(synthetic_panel, horizons=(1, 5))[target_column(h)]
    rv = synthetic_panel["rv"].to_numpy()
    for t in (0, 1, 50, 777, len(rv) - h - 1):
        assert tg.iloc[t] == pytest.approx(rv[t + 1 : t + 1 + h].sum(), rel=1e-12)
    assert tg.iloc[-h:].isna().all()  # future not observed
    assert tg.iloc[:-h].notna().all()


@pytest.mark.parametrize("h", [1, 5])
def test_make_targets_depend_only_on_next_h_days(h: int) -> None:
    panel = make_synthetic_panel(n_days=60)
    base = make_targets(panel, horizons=(h,))[target_column(h)]
    t = 20
    col = panel.columns.get_loc("rv")
    # changing rv at/before t or strictly after t+h never moves target[t] ...
    for k in (*range(0, t + 1), t + h + 1, t + h + 10, 59):
        p2 = panel.copy()
        p2.iloc[k, col] *= 7.0
        assert make_targets(p2, horizons=(h,))[target_column(h)].iloc[t] == base.iloc[t], k
    # ... while changing any of the next h days does
    for k in range(t + 1, t + h + 1):
        p2 = panel.copy()
        p2.iloc[k, col] *= 7.0
        assert make_targets(p2, horizons=(h,))[target_column(h)].iloc[t] != base.iloc[t], k


@pytest.mark.parametrize("h", [1, 5])
def test_make_targets_invariant_before_cut_minus_h(synthetic_panel: pd.DataFrame, h: int) -> None:
    """Targets of rows at least h days before the cut are unaffected by truncation."""
    cut = synthetic_panel.index[500]
    last_safe = cut - pd.Timedelta(days=h + 1)  # target[t] needs rv up to t+h <= cut-1
    full = make_targets(synthetic_panel, horizons=(h,)).loc[:last_safe]
    trunc = make_targets(synthetic_panel.loc[: cut - pd.Timedelta(days=1)], horizons=(h,))
    pd.testing.assert_frame_equal(full, trunc.loc[:last_safe], check_freq=False)
