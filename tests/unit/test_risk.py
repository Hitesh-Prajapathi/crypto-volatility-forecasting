"""VaR/ES application: pairing of forecasts with next-day returns and information-set safety."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from volforecast.contracts import COL_RET, COL_RV, FORECAST_COLUMNS
from volforecast.evaluation.risk import VAR_COLUMNS, build_var_forecasts, var_backtest_table

ASSET = "SYN"
MIN_RESID = 50


def _forecasts(panel: pd.DataFrame, models: dict[str, float]) -> pd.DataFrame:
    """h=1 forecast table: forecast = scale * trailing 5-day mean RV (info up to origin)."""
    base = panel[COL_RV].rolling(5, min_periods=5).mean()
    frames = []
    for name, scale in models.items():
        f = (base * scale).dropna()
        f = f.loc["2020-06-01":]
        frames.append(
            pd.DataFrame(
                {
                    "asset": ASSET,
                    "model": name,
                    "horizon": 1,
                    "origin_date": f.index,
                    "target_date": f.index + pd.Timedelta(days=1),
                    "forecast_var": f.to_numpy(),
                    "realized_var": panel[COL_RV].shift(-1).reindex(f.index).to_numpy(),
                    "model_version": "dev",
                    "data_hash": "h",
                }
            )[list(FORECAST_COLUMNS)]
        )
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def fc(synthetic_panel: pd.DataFrame) -> pd.DataFrame:
    return _forecasts(synthetic_panel, {"a": 1.0, "b": 1.3})


def _var(fc: pd.DataFrame, panel: pd.DataFrame, **kw: object) -> pd.DataFrame:
    kwargs: dict[str, object] = {
        "alphas": (0.05, 0.01),
        "dists": ("normal", "t", "fhs"),
        "min_resid": MIN_RESID,
        "nu_refit_every": 10,
    }
    kwargs.update(kw)
    return build_var_forecasts(fc, {ASSET: panel}, **kwargs)  # type: ignore[arg-type]


def test_shape_and_pairing_with_next_day_return(
    fc: pd.DataFrame, synthetic_panel: pd.DataFrame
) -> None:
    out = _var(fc, synthetic_panel)
    assert tuple(out.columns) == VAR_COLUMNS
    assert set(out["model"]) == {"a", "b"}
    assert len(out.groupby(["model", "dist", "alpha"])) == 2 * 3 * 2
    expected_ret = synthetic_panel[COL_RET].shift(-1).reindex(out["origin_date"]).to_numpy()
    np.testing.assert_array_equal(out["ret"].to_numpy(), expected_ret)
    assert (out["target_date"] - out["origin_date"] == pd.Timedelta(days=1)).all()
    assert (out["var"] > 0).all()
    assert (out["es"] >= out["var"]).all()
    np.testing.assert_array_equal(out["hit"], out["ret"] < -out["var"])


def test_normal_var_is_closed_form(fc: pd.DataFrame, synthetic_panel: pd.DataFrame) -> None:
    from scipy.stats import norm

    out = _var(fc, synthetic_panel, dists=("normal",), alphas=(0.01,))
    expected = -norm.ppf(0.01) * np.sqrt(out["forecast_var"])
    np.testing.assert_allclose(out["var"], expected)
    assert (out["dist_used"] == "normal").all()


def test_warmup_uses_normal_then_requested_dist(
    fc: pd.DataFrame, synthetic_panel: pd.DataFrame
) -> None:
    out = _var(fc, synthetic_panel)
    for dist in ("t", "fhs"):
        g = out[(out["dist"] == dist) & (out["model"] == "a") & (out["alpha"] == 0.05)]
        g = g.sort_values("origin_date").reset_index(drop=True)
        assert (g.loc[: MIN_RESID - 1, "dist_used"] == "normal").all()
        assert (g.loc[MIN_RESID:, "dist_used"] == dist).all()
        assert (g["n_resid"] == np.arange(len(g))).all()
    t_rows = out[out["dist_used"] == "t"]
    assert np.isfinite(t_rows["nu"]).all()
    assert (t_rows["nu"] > 2).all()


def test_fhs_uses_only_strictly_earlier_residuals(
    fc: pd.DataFrame, synthetic_panel: pd.DataFrame
) -> None:
    out = _var(fc, synthetic_panel, dists=("fhs",), alphas=(0.05,))
    g = out[out["model"] == "a"].sort_values("origin_date").reset_index(drop=True)
    z = (g["ret"] / np.sqrt(g["forecast_var"])).to_numpy()
    for i in (MIN_RESID, MIN_RESID + 17, len(g) - 1):
        q = np.quantile(z[:i], 0.05)  # origins strictly before row i
        assert g.loc[i, "var"] == pytest.approx(-q * np.sqrt(g.loc[i, "forecast_var"]))
        tail = z[:i][z[:i] <= q]
        assert g.loc[i, "es"] == pytest.approx(-tail.mean() * np.sqrt(g.loc[i, "forecast_var"]))


@pytest.mark.parametrize("dist", ["t", "fhs", "normal"])
def test_mutating_future_returns_does_not_change_past_var(
    fc: pd.DataFrame, synthetic_panel: pd.DataFrame, dist: str
) -> None:
    """VaR at origin t may use ret up to day t only: perturb ret strictly after day T."""
    base = _var(fc, synthetic_panel, dists=(dist,))
    cutoff = pd.Timestamp("2021-06-01")  # origin T; ret_{T+1} onwards is "the future"
    mutated_panel = synthetic_panel.copy()
    future = mutated_panel.index > cutoff
    mutated_panel.loc[future, COL_RET] = mutated_panel.loc[future, COL_RET] * 7.0 + 0.05
    mutated = _var(fc, mutated_panel, dists=(dist,))

    keys = ["model", "alpha", "origin_date"]
    m = base.merge(mutated, on=keys, suffixes=("", "_m"))
    past = m[m["origin_date"] <= cutoff]
    later = m[m["origin_date"] > cutoff + pd.Timedelta(days=2)]
    assert len(past) > 100
    assert len(later) > 100
    np.testing.assert_array_equal(past["var"], past["var_m"])
    np.testing.assert_array_equal(past["es"], past["es_m"])
    np.testing.assert_array_equal(past["dist_used"], past["dist_used_m"])
    if dist != "normal":  # distribution estimated from residuals: later rows must react
        assert not np.allclose(later["var"], later["var_m"])


def test_align_drops_origins_where_any_model_is_missing(
    fc: pd.DataFrame, synthetic_panel: pd.DataFrame
) -> None:
    broken = fc.copy()
    bad_dates = broken[broken["model"] == "b"]["origin_date"].iloc[100:110]
    broken.loc[(broken["model"] == "b") & broken["origin_date"].isin(bad_dates), "forecast_var"] = (
        np.nan
    )
    aligned = _var(broken, synthetic_panel, dists=("normal",), alphas=(0.05,))
    assert not set(bad_dates) & set(aligned["origin_date"])
    n_a = (aligned["model"] == "a").sum()
    assert n_a == (aligned["model"] == "b").sum()
    free = _var(broken, synthetic_panel, dists=("normal",), alphas=(0.05,), align=False)
    assert (free["model"] == "a").sum() == n_a + 10


def test_backtest_table_columns_and_calibration(
    fc: pd.DataFrame, synthetic_panel: pd.DataFrame
) -> None:
    tab = var_backtest_table(_var(fc, synthetic_panel))
    assert len(tab) == 2 * 3 * 2
    for col in (
        "n_obs",
        "expected_breaches",
        "breaches",
        "breach_rate",
        "kupiec_p",
        "christoffersen_ind_p",
        "christoffersen_cc_p",
        "traffic_light",
        "es_z2",
        "frac_fallback",
    ):
        assert col in tab.columns
    assert tab["kupiec_p"].between(0, 1).all()
    one = tab[tab["alpha"] == 0.01]
    assert set(one["traffic_light"]) <= {"green", "yellow", "red"}
    assert tab[tab["alpha"] == 0.05]["traffic_light"].isna().all()
    assert (tab[tab["dist"] == "normal"]["frac_fallback"] == 0).all()
    assert (tab[tab["dist"] == "fhs"]["frac_fallback"] > 0).all()
    # FHS on the scale-correct forecast "a" should be reasonably calibrated at 5%
    fhs5 = tab[(tab["dist"] == "fhs") & (tab["alpha"] == 0.05) & (tab["model"] == "a")].iloc[0]
    assert 0.02 < fhs5["breach_rate"] < 0.09
    # scale 1.3 forecast overstates risk: fewer breaches under the normal
    n5 = tab[(tab["dist"] == "normal") & (tab["alpha"] == 0.05)].set_index("model")
    assert n5.loc["b", "breaches"] <= n5.loc["a", "breaches"]
