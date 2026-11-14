"""Unit tests for volforecast.features.realized."""

from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd
import pytest

from tests.conftest import make_synthetic_klines
from volforecast.contracts import BARS_PER_DAY_5M, DATE_INDEX, KLINE_COLUMNS, PANEL_COLUMNS
from volforecast.features.realized import (
    daily_realized_measures,
    intraday_log_returns,
    signature_rv,
)

MEASURES = ["rv", "bpv", "jump", "rs_pos", "rs_neg", "rq", "ret"]


def klines_from_log_prices(log_prices: np.ndarray, start: str = "2024-01-01") -> pd.DataFrame:
    """Minimal klines (all KLINE_COLUMNS) whose closes are ``100 * exp(log_prices)``."""
    close = 100.0 * np.exp(log_prices)
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


def drop_bars(klines: pd.DataFrame, positions: np.ndarray) -> pd.DataFrame:
    """Drop the rows at integer positions (positions on the 5-min grid == row positions)."""
    return klines.drop(index=klines.index[positions]).reset_index(drop=True)


# ── hand-computed example ────────────────────────────────────────────────────


def test_hand_computed_two_days() -> None:
    n = 2 * BARS_PER_DAY_5M
    log_p = np.zeros(n)
    d1 = BARS_PER_DAY_5M
    # Day 1 log-prices: first four bars 0.01, -0.01, -0.01, 0.02, then flat at 0.02.
    log_p[d1:] = 0.02
    log_p[d1 : d1 + 3] = [0.01, -0.01, -0.01]
    panel = daily_realized_measures(klines_from_log_prices(log_p))

    # Day 1 returns: r0=0.01 (vs day-0 close at 0), r1=-0.02, r2=0, r3=0.03, rest 0.
    rv = 1e-4 + 4e-4 + 9e-4
    expected = {
        "rv": rv,
        "bpv": math.pi / 2 * (0.01 * 0.02),  # |r0||r1|; |r1||r2| = |r2||r3| = 0
        "jump": rv - math.pi / 2 * 2e-4,
        "rs_pos": 1e-4 + 9e-4,
        "rs_neg": 4e-4,
        "rq": 288 / 3 * (1e-8 + 16e-8 + 81e-8),
        "ret": 0.02,
    }
    day1 = panel.iloc[1]
    for col, value in expected.items():
        assert day1[col] == pytest.approx(value, rel=1e-12, abs=1e-15), col
    assert day1["n_bars"] == BARS_PER_DAY_5M
    assert bool(day1["complete"]) is True

    # Day 0: first return has no previous close -> NaN; the other 287 returns are 0.
    day0 = panel.iloc[0]
    assert day0["rv"] == 0.0
    assert day0["bpv"] == 0.0
    assert day0["jump"] == 0.0
    assert day0["rq"] == 0.0
    assert math.isnan(day0["ret"])
    assert day0["n_bars"] == BARS_PER_DAY_5M
    assert bool(day0["complete"]) is False


def test_rq_uses_number_of_valid_returns() -> None:
    n = 2 * BARS_PER_DAY_5M
    log_p = np.zeros(n)
    log_p[BARS_PER_DAY_5M + 10 :] = 0.01  # single return of 0.01 at bar 10 of day 1
    panel = daily_realized_measures(klines_from_log_prices(log_p))
    # Day 0 has M=287 valid returns, all zero; day 1 has M=288.
    assert panel["rq"].iloc[1] == pytest.approx(288 / 3 * 0.01**4)
    assert panel["rs_neg"].iloc[1] == 0.0
    assert panel["bpv"].iloc[1] == 0.0  # lone return has zero-return neighbours


# ── identities on synthetic data ─────────────────────────────────────────────


def test_identities_on_synthetic(synthetic_klines: pd.DataFrame) -> None:
    panel = daily_realized_measures(synthetic_klines)
    ok = panel[panel["complete"]]
    assert len(ok) == len(panel) - 1  # only day 0 lacks a previous close
    np.testing.assert_allclose(ok["rs_pos"] + ok["rs_neg"], ok["rv"], rtol=1e-12)
    assert (panel["bpv"].dropna() >= 0).all()
    assert (panel["rq"].dropna() >= 0).all()
    assert (panel["jump"].dropna() >= 0).all()
    np.testing.assert_allclose(ok["jump"], np.maximum(ok["rv"] - ok["bpv"], 0.0))

    rets = intraday_log_returns(synthetic_klines)
    daily_sum = rets.groupby("date")["log_return"].sum()
    np.testing.assert_allclose(daily_sum.loc[ok.index], ok["ret"], atol=1e-12)
    # rv from the long-format returns agrees with the panel
    rv_long = (rets["log_return"] ** 2).groupby(rets["date"]).sum()
    np.testing.assert_allclose(rv_long.loc[ok.index], ok["rv"], rtol=1e-12)
    assert (ok["n_bars"] == BARS_PER_DAY_5M).all()


def test_rv_scale_invariance(synthetic_klines: pd.DataFrame) -> None:
    scaled = synthetic_klines.copy()
    for col in ("open", "high", "low", "close"):
        scaled[col] = scaled[col] * 123.456
    a = daily_realized_measures(synthetic_klines)
    b = daily_realized_measures(scaled)
    pd.testing.assert_frame_equal(a, b, rtol=1e-9, atol=1e-15)


# ── gap handling ─────────────────────────────────────────────────────────────


@pytest.fixture
def base_panel(synthetic_klines: pd.DataFrame) -> pd.DataFrame:
    return daily_realized_measures(synthetic_klines)


def test_three_missing_bars_are_forward_filled(
    synthetic_klines: pd.DataFrame, base_panel: pd.DataFrame
) -> None:
    day = 10
    gap = day * BARS_PER_DAY_5M + 100 + np.arange(3)
    k = drop_bars(synthetic_klines, gap)
    panel = daily_realized_measures(k, min_bars=280)
    row = panel.iloc[day]
    assert row["n_bars"] == BARS_PER_DAY_5M - 3
    assert bool(row["complete"]) is True
    # the day's rv is unchanged except that the gap's returns are merged into one return
    assert row["ret"] == pytest.approx(base_panel["ret"].iloc[day], abs=1e-12)

    rets = intraday_log_returns(k)
    seg = rets.iloc[gap]
    assert not seg["observed"].any()
    assert (seg["log_return"] == 0.0).all()
    assert (seg["close"] == rets["close"].iloc[gap[0] - 1]).all()
    # the first return after the gap carries the whole price move
    after = rets["log_return"].iloc[gap[-1] + 1]
    full_rets = intraday_log_returns(synthetic_klines)["log_return"].to_numpy()
    assert after == pytest.approx(full_rets[gap].sum() + full_rets[gap[-1] + 1])
    # neighbouring days are untouched
    for d in (day - 1, day + 1):
        pd.testing.assert_series_equal(panel.iloc[d], base_panel.iloc[d])


def test_three_missing_bars_break_default_min_bars(synthetic_klines: pd.DataFrame) -> None:
    gap = 5 * BARS_PER_DAY_5M + 100 + np.arange(3)
    panel = daily_realized_measures(drop_bars(synthetic_klines, gap), min_bars=288)
    assert bool(panel["complete"].iloc[5]) is False  # n_bars=285 < 288
    assert np.isfinite(panel["rv"].iloc[5])  # still computed


def test_ten_missing_bars_make_day_incomplete(
    synthetic_klines: pd.DataFrame, base_panel: pd.DataFrame
) -> None:
    day = 10
    gap = day * BARS_PER_DAY_5M + 100 + np.arange(10)
    k = drop_bars(synthetic_klines, gap)
    panel = daily_realized_measures(k)
    row = panel.iloc[day]
    assert row["n_bars"] == BARS_PER_DAY_5M - 10
    assert bool(row["complete"]) is False
    assert np.isfinite(row["rv"])  # nansum over valid returns
    assert row["rv"] <= base_panel["rv"].iloc[day]  # dropped returns only remove terms
    rets = intraday_log_returns(k)
    assert rets["log_return"].iloc[gap].isna().all()  # whole gap stays NaN
    assert rets["close"].iloc[gap].isna().all()
    # the day after is complete again: the gap is internal to day 10
    assert bool(panel["complete"].iloc[day + 1]) is True
    assert panel["ret"].iloc[day] == pytest.approx(base_panel["ret"].iloc[day], abs=1e-12)


def test_missing_whole_day(synthetic_klines: pd.DataFrame, base_panel: pd.DataFrame) -> None:
    day = 20
    day_start = pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(days=day)
    t = synthetic_klines["open_time"]
    k = synthetic_klines[(t < day_start) | (t >= day_start + pd.Timedelta(days=1))]
    k = k.reset_index(drop=True)
    panel = daily_realized_measures(k)

    assert len(panel) == len(base_panel)  # the missing date is still present
    missing = panel.iloc[day]
    assert missing["n_bars"] == 0
    assert bool(missing["complete"]) is False
    assert missing[MEASURES].isna().all()

    # day before: unchanged
    pd.testing.assert_series_equal(panel.iloc[day - 1], base_panel.iloc[day - 1])
    # day after: first return has no previous close -> incomplete, ret NaN, rv over 287 returns
    after = panel.iloc[day + 1]
    assert bool(after["complete"]) is False
    assert math.isnan(after["ret"])
    full_r = intraday_log_returns(synthetic_klines)["log_return"].to_numpy()
    first_r = full_r[(day + 1) * BARS_PER_DAY_5M]
    assert after["rv"] == pytest.approx(base_panel["rv"].iloc[day + 1] - first_r**2, rel=1e-9)
    # two days after: back to normal, including ret (close-to-close across the hole is NaN
    # only for the day directly after it)
    pd.testing.assert_series_equal(panel.iloc[day + 2], base_panel.iloc[day + 2])


def test_gap_straddling_midnight_is_causal(synthetic_klines: pd.DataFrame) -> None:
    """Day t must not change when later days are dropped, even for gaps crossing midnight."""
    day = 8
    end = (day + 1) * BARS_PER_DAY_5M
    gap = end + np.arange(-2, 2)  # 2 bars on day 8, 2 on day 9
    k = drop_bars(synthetic_klines, gap)
    full = daily_realized_measures(k)
    cut_ts = k["open_time"].iloc[0] + pd.Timedelta(days=day + 1)
    trunc = daily_realized_measures(k[k["open_time"] < cut_ts].reset_index(drop=True))
    pd.testing.assert_frame_equal(full.iloc[: day + 1], trunc, check_freq=False)


def test_first_day_is_never_complete(base_panel: pd.DataFrame) -> None:
    assert bool(base_panel["complete"].iloc[0]) is False
    assert math.isnan(base_panel["ret"].iloc[0])


# ── contract conformance ─────────────────────────────────────────────────────


def test_panel_conforms_to_contract(synthetic_klines: pd.DataFrame) -> None:
    gap = np.arange(15 * BARS_PER_DAY_5M, 17 * BARS_PER_DAY_5M)  # two missing days
    panel = daily_realized_measures(drop_bars(synthetic_klines, gap))
    assert tuple(panel.columns) == PANEL_COLUMNS
    assert panel.index.name == DATE_INDEX
    assert isinstance(panel.index, pd.DatetimeIndex)
    assert panel.index.tz is None
    assert (panel.index == panel.index.normalize()).all()
    assert panel.index.is_unique
    assert panel.index.is_monotonic_increasing
    assert (np.diff(panel.index.to_numpy()) == np.timedelta64(1, "D")).all()
    assert len(panel) == 40
    for col in PANEL_COLUMNS:
        if col == "n_bars":
            assert panel[col].dtype == np.int64
        elif col == "complete":
            assert panel[col].dtype == bool
        else:
            assert panel[col].dtype == np.float64, col


def test_empty_and_invalid_input(synthetic_klines: pd.DataFrame) -> None:
    empty = daily_realized_measures(synthetic_klines.iloc[0:0])
    assert tuple(empty.columns) == PANEL_COLUMNS
    assert len(empty) == 0
    assert intraday_log_returns(synthetic_klines.iloc[0:0]).empty
    dup = pd.concat([synthetic_klines.iloc[:10], synthetic_klines.iloc[:1]])
    with pytest.raises(ValueError, match="duplicate"):
        daily_realized_measures(dup)
    off_grid = synthetic_klines.iloc[:10].copy()
    off_grid["open_time"] = off_grid["open_time"] + pd.Timedelta(seconds=7)
    with pytest.raises(ValueError, match="5-minute grid"):
        daily_realized_measures(off_grid)
    with pytest.raises(ValueError, match="missing columns"):
        daily_realized_measures(synthetic_klines.drop(columns=["close"]))


def test_intraday_returns_layout(synthetic_klines: pd.DataFrame) -> None:
    k = synthetic_klines.iloc[100:-50]  # partial first and last day
    rets = intraday_log_returns(k)
    assert len(rets) == 40 * BARS_PER_DAY_5M
    assert rets.index[0] == pd.Timestamp("2024-01-01 00:00", tz="UTC")
    assert rets.index[-1] == pd.Timestamp("2024-02-09 23:55", tz="UTC")
    assert rets.index.name == "open_time"
    assert set(rets.columns) == {"close", "log_return", "observed", "date"}
    assert rets["observed"].sum() == len(k)
    assert (rets["date"] == rets.index.tz_localize(None).normalize()).all()
    # leading bars before the first observation stay NaN (nothing to fill from)
    assert rets["log_return"].iloc[:101].isna().all()
    assert tuple(synthetic_klines.columns) == KLINE_COLUMNS


# ── signature plot ───────────────────────────────────────────────────────────


def test_signature_rv(synthetic_klines: pd.DataFrame) -> None:
    sig = signature_rv(synthetic_klines)
    panel = daily_realized_measures(synthetic_klines)
    assert list(sig.index) == ["5min", "10min", "15min", "30min", "60min"]
    assert sig["mean_rv"].iloc[0] == pytest.approx(panel.loc[panel["complete"], "rv"].mean())
    assert (sig["n_days"] == 39).all()
    assert (sig["mean_rv"] > 0).all()
    # i.i.d.-like synthetic returns: no strong microstructure effect, RV roughly flat
    assert sig["mean_rv"].max() / sig["mean_rv"].min() < 1.5
    with pytest.raises(ValueError, match="multiple of 5min"):
        signature_rv(synthetic_klines, intervals=("7min",))


def test_signature_rv_matches_manual_resampling() -> None:
    rng = np.random.default_rng(3)
    log_p = np.cumsum(rng.normal(0, 1e-3, 3 * BARS_PER_DAY_5M))
    k = klines_from_log_prices(log_p)
    sig = signature_rv(k, intervals=("15min",))
    sampled = log_p[2::3]  # closes at the end of each 15-min interval
    per_day = 96
    r = np.concatenate([[np.nan], np.diff(sampled)]).reshape(3, per_day)
    expected = (r[1:] ** 2).sum(axis=1).mean()  # day 0 lacks a previous close
    assert sig["mean_rv"].iloc[0] == pytest.approx(expected)
    assert sig["n_days"].iloc[0] == 2


# ── brute-force reference ────────────────────────────────────────────────────


def reference_panel(
    klines: pd.DataFrame, min_bars: int, max_ffill: int
) -> dict[str, list[float | bool]]:
    """Plain-Python, per-day loop implementation of the documented rules."""
    t0 = klines["open_time"].min().normalize()
    t1 = klines["open_time"].max().normalize()
    n_days = (t1 - t0).days + 1
    n = n_days * 288
    obs: dict[int, float] = {}
    for ts, c in zip(klines["open_time"], klines["close"], strict=True):
        obs[int((ts - t0) / pd.Timedelta(minutes=5))] = float(c)

    close: list[float] = [math.nan] * n
    for i in range(n):
        if i in obs:
            close[i] = obs[i]
            continue
        j = i
        while j >= 0 and j not in obs:
            j -= 1
        k = i
        day_end = (i // 288 + 1) * 288
        while k < day_end and k not in obs:
            k += 1
        if j >= 0 and (k - j - 1) <= max_ffill:
            close[i] = obs[j]

    out: dict[str, list[float | bool]] = {c: [] for c in PANEL_COLUMNS}
    for d in range(n_days):
        r: list[float] = []
        for i in range(d * 288, (d + 1) * 288):
            if i == 0 or math.isnan(close[i]) or math.isnan(close[i - 1]):
                r.append(math.nan)
            else:
                r.append(math.log(close[i] / close[i - 1]))
        v = [x for x in r if not math.isnan(x)]
        m = len(v)
        nan = math.nan
        rv = sum(x * x for x in v) if m else nan
        pairs = [
            abs(r[i]) * abs(r[i - 1])
            for i in range(1, 288)
            if not math.isnan(r[i]) and not math.isnan(r[i - 1])
        ]
        bpv = math.pi / 2 * sum(pairs) if pairs else nan
        out["rv"].append(rv)
        out["bpv"].append(bpv)
        out["jump"].append(max(rv - bpv, 0.0) if m and pairs else nan)
        out["rs_pos"].append(sum(x * x for x in v if x > 0) if m else nan)
        out["rs_neg"].append(sum(x * x for x in v if x < 0) if m else nan)
        out["rq"].append(m / 3 * sum(x**4 for x in v) if m else nan)
        last, prev = close[(d + 1) * 288 - 1], close[d * 288 - 1] if d else nan
        out["ret"].append(
            math.log(last / prev) if not (math.isnan(last) or math.isnan(prev)) else nan
        )
        n_obs = sum(1 for i in range(d * 288, (d + 1) * 288) if i in obs)
        out["n_bars"].append(n_obs)
        out["complete"].append(n_obs >= min_bars and m == 288)
    return out


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_reference_loop_with_random_gaps(seed: int) -> None:
    rng = np.random.default_rng(seed)
    k = make_synthetic_klines(n_days=12, seed=seed)
    drop = np.zeros(len(k), dtype=bool)
    drop[rng.random(len(k)) < 0.01] = True  # scattered single-bar gaps
    for _ in range(8):  # clustered gaps of 1..8 bars, some across midnight
        s = int(rng.integers(0, len(k) - 10))
        drop[s : s + int(rng.integers(1, 9))] = True
    drop[4 * 288 + 286 : 5 * 288 + 2] = True  # gap straddling midnight
    drop[7 * 288 : 8 * 288] = True  # whole day
    k = k[~drop].reset_index(drop=True)

    got = daily_realized_measures(k, min_bars=280, max_ffill=3)
    ref = reference_panel(k, 280, 3)
    for col in PANEL_COLUMNS:
        np.testing.assert_allclose(
            got[col].to_numpy(dtype=float),
            np.asarray(ref[col], dtype=float),
            rtol=1e-10,
            atol=1e-15,
            equal_nan=True,
            err_msg=col,
        )


# ── performance ──────────────────────────────────────────────────────────────


@pytest.mark.slow
def test_ten_years_under_five_seconds() -> None:
    k = make_synthetic_klines(n_days=3650, seed=1)
    rng = np.random.default_rng(0)
    k = k[rng.random(len(k)) > 0.005].reset_index(drop=True)  # 0.5% scattered gaps
    assert len(k) > 1_000_000
    start = time.perf_counter()
    panel = daily_realized_measures(k)
    elapsed = time.perf_counter() - start
    assert len(panel) == 3650
    assert elapsed < 5.0, f"took {elapsed:.2f}s"
