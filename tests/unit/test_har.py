import numpy as np
import pandas as pd

from volforecast.contracts import target_column
from volforecast.features.har import har_aggregates, make_targets


def test_har_aggregates_are_backward_looking(synthetic_panel: pd.DataFrame) -> None:
    full = har_aggregates(synthetic_panel)
    cut = synthetic_panel.index[300]
    truncated = har_aggregates(synthetic_panel.loc[:cut])
    pd.testing.assert_series_equal(full.loc[cut], truncated.iloc[-1], check_names=False)


def test_har_values(synthetic_panel: pd.DataFrame) -> None:
    agg = har_aggregates(synthetic_panel)
    t = 100
    rv = synthetic_panel["rv"].to_numpy()
    assert np.isclose(agg["rv_d"].iloc[t], rv[t])
    assert np.isclose(agg["rv_w"].iloc[t], rv[t - 4 : t + 1].mean())
    assert np.isclose(agg["rv_m"].iloc[t], rv[t - 21 : t + 1].mean())
    assert agg["rv_m"].iloc[:21].isna().all()


def test_targets_alignment(synthetic_panel: pd.DataFrame) -> None:
    tg = make_targets(synthetic_panel, horizons=(1, 5))
    rv = synthetic_panel["rv"].to_numpy()
    t = 50
    assert np.isclose(tg[target_column(1)].iloc[t], rv[t + 1])
    assert np.isclose(tg[target_column(5)].iloc[t], rv[t + 1 : t + 6].sum())
    assert tg[target_column(5)].iloc[-5:].isna().all()
    assert tg[target_column(1)].iloc[-1:].isna().all()
