"""ens_inv warm start: the learned state survives save/load and a restart of the walk-forward.

The walk-forward is resumed the way the retrain job resumes it: the persisted instance is
``fit`` again on the full history (an engine refit) and then keeps predicting. The persisted
state is forecasts only; losses are recomputed from the panel, so it cannot encode look-ahead.
"""

from __future__ import annotations

import cloudpickle
import numpy as np
import pandas as pd
import pytest

from tests.conftest import make_synthetic_panel
from volforecast.contracts import VolModel
from volforecast.evaluation.walkforward import ModelSpec, WalkForwardConfig, run_walkforward
from volforecast.models.ensemble import InverseLossEnsemble
from volforecast.models.factory import build_model, require_warm_state

H = 1
REFIT = 10
START, SPLIT, END = 420, 470, 530  # SPLIT - START is a multiple of REFIT: same refit phase
MIN_OBS = 20
MEMBERS = ["log_har", "har"]  # cheap deterministic members


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return make_synthetic_panel(700, seed=4)


def _walk(model: VolModel, panel: pd.DataFrame, first: int, last: int, h: int = H) -> pd.Series:
    """Walk-forward ``model`` (one reused instance) over origins ``first..last``."""
    cfg = WalkForwardConfig(
        first_origin=panel.index[first], last_origin=panel.index[last], horizons=(h,), min_train=60
    )
    spec = ModelSpec(name="m", builder=lambda: model, refit_every=REFIT, reuse_instance=True)
    fc, stats = run_walkforward(panel, spec, cfg, h, "X")
    assert stats.n_fit_failures == 0
    assert stats.n_predict_failures == 0
    return fc.set_index("origin_date")["forecast_var"]


def _ens_inv() -> InverseLossEnsemble:
    m = build_model("ens_inv", members=MEMBERS, min_obs=MIN_OBS)
    assert isinstance(m, InverseLossEnsemble)
    return m


def _roundtrip(model: InverseLossEnsemble) -> InverseLossEnsemble:
    out = cloudpickle.loads(cloudpickle.dumps(model))
    assert isinstance(out, InverseLossEnsemble)
    return out


def test_cold_instance_reports_no_state() -> None:
    m = _ens_inv()
    assert m.n_recorded_origins == 0
    assert m.history_end is None
    with pytest.raises(RuntimeError, match="learned state"):
        require_warm_state(m, "ens_inv")
    require_warm_state(build_model("har"), "har")  # stateless models are not checked


def test_save_load_roundtrip_keeps_weights_and_history(panel: pd.DataFrame) -> None:
    m = _ens_inv()
    _walk(m, panel, START, SPLIT - 1)
    weights = m.last_weights
    assert weights
    assert not np.allclose(list(weights.values()), 0.5)
    loaded = _roundtrip(m)
    assert loaded.last_weights == weights
    assert loaded.n_recorded_origins == m.n_recorded_origins > 0
    assert loaded.history_end == m.history_end == panel.index[SPLIT - 1]
    assert loaded.warm_start_summary() == m.warm_start_summary()
    for s in m._hist:
        np.testing.assert_array_equal(loaded._hist[s], m._hist[s])
    require_warm_state(loaded, "ens_inv")


def test_warm_started_forecasts_equal_an_uninterrupted_run(panel: pd.DataFrame) -> None:
    uninterrupted = _walk(_ens_inv(), panel, START, END)
    first_leg = _ens_inv()
    _walk(first_leg, panel, START, SPLIT - 1)
    resumed = _walk(_roundtrip(first_leg), panel, SPLIT, END)  # fit on full history, then predict
    expect = uninterrupted.loc[panel.index[SPLIT] :]
    assert len(expect) == len(resumed) == END - SPLIT + 1
    np.testing.assert_allclose(resumed.to_numpy(), expect.to_numpy(), rtol=1e-12, atol=0)


def test_cold_restart_differs_from_the_uninterrupted_run(panel: pd.DataFrame) -> None:
    """Negative control: without the persisted state the resumed forecasts are not the same."""
    uninterrupted = _walk(_ens_inv(), panel, START, END).loc[panel.index[SPLIT] :]
    cold = _walk(_ens_inv(), panel, SPLIT, END)
    head = slice(0, 15)  # equal weights until min_obs evaluable origins exist
    assert not np.allclose(cold.to_numpy()[head], uninterrupted.to_numpy()[head], rtol=1e-9)


def test_ens_inv_differs_from_ens_eq_after_enough_history(panel: pd.DataFrame) -> None:
    inv = _ens_inv()
    eq = build_model("ens_eq", members=MEMBERS)
    f_inv = _walk(inv, panel, START, END).to_numpy()
    f_eq = _walk(eq, panel, START, END).to_numpy()
    # the first origins (history < min_obs) are exactly equal weights ...
    np.testing.assert_allclose(f_inv[:MIN_OBS], f_eq[:MIN_OBS], rtol=1e-12)
    # ... afterwards the learned weights move the forecast
    assert not np.allclose(f_inv[MIN_OBS + H + 5 :], f_eq[MIN_OBS + H + 5 :], rtol=1e-6)
    w = np.array(list(inv.last_weights.values()))
    assert abs(w[0] - w[1]) > 1e-3
    assert w.sum() == pytest.approx(1.0)


def test_weights_at_t_use_only_losses_realized_by_t(panel: pd.DataFrame) -> None:
    """A state saved LATER (holding origins beyond t - h) must give the weights it had at t."""
    t_pos = SPLIT + 5
    t = panel.index[t_pos]
    early, late = _ens_inv(), _ens_inv()
    _walk(early, panel, START, t_pos)
    _walk(late, panel, START, END)  # also holds origins whose targets are unrealized at t
    loaded = _roundtrip(late)
    assert loaded.history_end == panel.index[END] > t
    visible = panel.iloc[: t_pos + 1]
    np.testing.assert_allclose(
        loaded.trailing_qlike(visible, H, t), early.trailing_qlike(visible, H, t), rtol=1e-12
    )
    cutoff = t - pd.Timedelta(days=H)
    assert sum(s <= cutoff for s in loaded._hist) == sum(s <= cutoff for s in early._hist)
