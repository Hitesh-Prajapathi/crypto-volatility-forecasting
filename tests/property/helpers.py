"""Reusable leakage-guard helpers.

The central idea: a feature/panel builder is *causal* iff its output for rows dated strictly
before ``cut`` does not change when all input data from ``cut`` onwards is removed.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import pandas as pd


def _to_naive_utc(ts: pd.Timestamp) -> pd.Timestamp:
    return ts.tz_convert("UTC").tz_localize(None) if ts.tzinfo is not None else ts


def _align_cut(cut: pd.Timestamp, index: pd.Index) -> pd.Timestamp:
    """Express ``cut`` in the timezone convention of ``index`` (tz-aware or tz-naive)."""
    if isinstance(index, pd.DatetimeIndex) and index.tz is not None:
        return cut.tz_localize("UTC") if cut.tzinfo is None else cut.tz_convert(index.tz)
    return _to_naive_utc(cut)


def assert_truncation_invariant(
    build_fn: Callable[[pd.DataFrame], pd.DataFrame],
    data: pd.DataFrame,
    cut_points: Iterable[pd.Timestamp | str],
    time_col_or_index: str | None = None,
    *,
    check_exact: bool = True,
) -> None:
    """Assert ``build_fn(data[time < cut])`` equals ``build_fn(data)`` on rows before ``cut``.

    Parameters
    ----------
    build_fn:
        Maps an input frame to an output frame with a DatetimeIndex (e.g. klines -> daily
        panel, or panel -> HAR regressors).
    data:
        Full input. Truncation keeps rows whose time is strictly before the cut.
    cut_points:
        Timestamps (tz-naive means UTC) at which to truncate.
    time_col_or_index:
        Name of the input's time column, or ``None`` to use ``data.index``.
    check_exact:
        Require bit-identical values (default). Set False to compare with pandas tolerances.

    The output is compared on rows whose output index is strictly before the cut; at least one
    such row must exist, so an empty comparison can never pass silently.
    """
    full = build_fn(data)
    times = pd.DatetimeIndex(data.index if time_col_or_index is None else data[time_col_or_index])
    for raw_cut in cut_points:
        cut = pd.Timestamp(raw_cut)
        mask = times < _align_cut(cut, times)
        truncated = build_fn(data.loc[mask])
        out_cut = _align_cut(cut, full.index)
        expected = full.loc[full.index < out_cut]
        actual = truncated.loc[truncated.index < out_cut]
        assert len(expected) > 0, f"no output rows before cut {cut}: the check is vacuous"
        # A truncated input legitimately ends earlier (e.g. trailing days without any bars are
        # unknown to a panel built from truncated data): only trailing rows may be absent.
        absent = expected.index.difference(actual.index)
        assert len(actual) > 0, f"truncated output has no rows before cut {cut}"
        assert (absent > actual.index.max()).all(), (
            f"look-ahead leakage: rows before cut {cut} missing from the truncated output: "
            f"{list(absent)}"
        )
        expected = expected.loc[actual.index]
        try:
            pd.testing.assert_frame_equal(
                actual, expected, check_freq=False, check_exact=check_exact
            )
        except AssertionError as exc:
            raise AssertionError(
                f"look-ahead leakage: output before cut {cut} changed when later data was "
                f"removed.\n{exc}"
            ) from exc
