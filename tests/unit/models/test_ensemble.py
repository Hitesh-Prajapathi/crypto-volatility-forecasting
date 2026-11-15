"""Forecast combinations: equal weights, inverse-QLIKE weights and their information set."""

from __future__ import annotations

from typing import Any, Self

import numpy as np
import pandas as pd
import pytest

from volforecast.contracts import COL_RV
from volforecast.models.ensemble import EqualWeightEnsemble, InverseLossEnsemble
from volforecast.models.factory import build_model


class _Scaled:
    """Stub member: forecast = k * h * RV_t (uses only the last row)."""

    def __init__(self, name: str, k: float) -> None:
        self.name, self.k = name, k
        self.n_fits = 0

    def fit(self, panel: pd.DataFrame, h: int) -> Self:
        self.n_fits += 1
        return self

    def predict_variance(self, panel: pd.DataFrame, h: int) -> float:
        return float(self.k * h * panel[COL_RV].iloc[-1])

    def get_params(self) -> dict[str, Any]:
        return {"k": self.k}


def _drive(
    ens: InverseLossEnsemble | EqualWeightEnsemble,
    panel: pd.DataFrame,
    h: int,
    start: int,
    end: int,
) -> list[float]:
    """Mimic the walk-forward engine: fit once, predict at every origin with `.iloc[:pos+1]`."""
    ens.fit(panel.iloc[: start + 1], h)
    return [ens.predict_variance(panel.iloc[: pos + 1], h) for pos in range(start, end + 1)]


def _expected_weights(
    panel: pd.DataFrame, ks: list[float], h: int, start: int, pos: int, window: int
) -> np.ndarray:
    """Independent computation: origins s in [start, pos-h], target sum RV_{s+1..s+h}."""
    rv = panel[COL_RV].to_numpy()
    origins = list(range(start, pos - h + 1))[-window:]
    losses = []
    for k in ks:
        r = np.array([rv[s + 1 : s + h + 1].sum() / (k * h * rv[s]) for s in origins])
        losses.append(np.mean(r - np.log(r) - 1.0))
    inv = 1.0 / np.array(losses)
    return inv / inv.sum()


@pytest.mark.parametrize("h", [1, 5])
def test_inverse_loss_weights_use_only_realized_targets(
    synthetic_panel: pd.DataFrame, h: int
) -> None:
    ks = [1.0, 3.0]
    window = 40
    start, end = 300, 420
    ens = InverseLossEnsemble([_Scaled("a", ks[0]), _Scaled("b", ks[1])], window=window)
    _drive(ens, synthetic_panel, h, start, end)
    hist = ens.weight_history()
    for pos in (start + 25, start + 60, end):
        t = synthetic_panel.index[pos]
        got = hist.loc[t].to_numpy()
        np.testing.assert_allclose(
            got, _expected_weights(synthetic_panel, ks, h, start, pos, window), rtol=1e-10
        )
    # before min_obs evaluable origins exist the weights are equal
    np.testing.assert_allclose(hist.iloc[0].to_numpy(), [0.5, 0.5])
    assert hist.iloc[5].tolist() == [0.5, 0.5]


def test_weights_ignore_unrealized_origins_and_future_rows(synthetic_panel: pd.DataFrame) -> None:
    h, start, pos = 5, 300, 400
    ks = [1.0, 3.0]

    def run(panel: pd.DataFrame, poison: bool) -> np.ndarray:
        ens = InverseLossEnsemble([_Scaled("a", ks[0]), _Scaled("b", ks[1])], window=60)
        ens.fit(panel.iloc[: start + 1], h)
        for p in range(start, pos):
            ens.predict_variance(panel.iloc[: p + 1], h)
        t = panel.index[pos]
        if poison:  # origins in (t-h, t] have unrealized targets: their forecasts must not matter
            for s in list(ens._hist):
                if s > t - pd.Timedelta(days=h):
                    ens._hist[s] = np.array([1e-9, 1e9])
        ens.predict_variance(panel.iloc[: pos + 1], h)
        return ens.weight_history().loc[t].to_numpy()

    base = run(synthetic_panel, poison=False)
    np.testing.assert_array_equal(base, run(synthetic_panel, poison=True))
    future = synthetic_panel.copy()
    future.loc[future.index > synthetic_panel.index[pos], COL_RV] *= 50.0
    np.testing.assert_array_equal(base, run(future, poison=False))


def test_equal_weight_is_plain_mean(synthetic_panel: pd.DataFrame) -> None:
    ens = EqualWeightEnsemble([_Scaled("a", 1.0), _Scaled("b", 3.0)])
    out = _drive(ens, synthetic_panel, 1, 300, 310)
    rv = synthetic_panel[COL_RV].to_numpy()[300:311]
    np.testing.assert_allclose(out, 2.0 * rv)


def test_members_refit_on_cadence_inside_the_ensemble(synthetic_panel: pd.DataFrame) -> None:
    a, b = _Scaled("a", 1.0), _Scaled("b", 2.0)
    ens = InverseLossEnsemble([a, b], member_refit_every=[1, 7])
    _drive(ens, synthetic_panel, 1, 300, 320)  # 21 origins
    assert a.n_fits == 21
    assert b.n_fits == 1 + 2
    # a second engine-side fit keeps the recorded history
    n = len(ens._hist)
    ens.fit(synthetic_panel.iloc[:322], 1)
    assert len(ens._hist) == n


def test_failed_member_is_skipped(synthetic_panel: pd.DataFrame) -> None:
    class Broken(_Scaled):
        def predict_variance(self, panel: pd.DataFrame, h: int) -> float:
            raise RuntimeError("boom")

    ens = EqualWeightEnsemble([_Scaled("a", 2.0), Broken("b", 1.0)])
    out = _drive(ens, synthetic_panel, 1, 300, 302)
    np.testing.assert_allclose(out, 2.0 * synthetic_panel[COL_RV].to_numpy()[300:303])


def test_factory_ensembles(synthetic_panel: pd.DataFrame) -> None:
    for name, cls in (("ens_eq", EqualWeightEnsemble), ("ens_inv", InverseLossEnsemble)):
        m = build_model(name)
        assert isinstance(m, cls)
        assert m.get_params()["members"] == ["log_har", "harq", "lgbm"]  # type: ignore[attr-defined]
        assert m.get_params()["member_refit_every"] == [1, 1, 7]  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="at least two"):
        EqualWeightEnsemble([_Scaled("a", 1.0)])
