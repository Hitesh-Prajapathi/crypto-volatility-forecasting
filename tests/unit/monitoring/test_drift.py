"""PSI / Wasserstein drift metrics, checked against hand computations."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from volforecast.monitoring.drift import (
    FEATURES,
    compute_drift,
    psi,
    wasserstein,
    write_drift_report,
)


def test_psi_matches_hand_computation() -> None:
    reference = np.arange(1, 11, dtype=float)  # median 5.5 -> two bins of 5 obs each
    current = np.array([1, 2, 3, 4, 5, 5, 2, 9, 3, 1], dtype=float)  # 9 below, 1 above 5.5
    p = (0.5, 0.5)
    q = (0.9, 0.1)
    expected = sum((qi - pi) * math.log(qi / pi) for pi, qi in zip(p, q, strict=True))
    assert expected == pytest.approx(0.878890, abs=1e-6)
    assert psi(reference, current, n_bins=2) == pytest.approx(expected, rel=1e-12)


def test_psi_of_identical_samples_is_zero_and_floor_keeps_it_finite() -> None:
    x = np.random.default_rng(0).standard_normal(500)
    assert psi(x, x, 10) == pytest.approx(0.0, abs=1e-12)
    far = x + 100.0  # every current point falls into the last bin: empty bins hit the eps floor
    assert math.isfinite(psi(x, far, 10))
    assert psi(x, far, 10) > 0.2


def test_wasserstein_of_a_shift_is_the_shift() -> None:
    x = np.random.default_rng(1).standard_normal(1000)
    assert wasserstein(x, x + 0.5) == pytest.approx(0.5, abs=1e-12)


def _realized(n: int = 400, seed: int = 2) -> pd.DataFrame:
    """i.i.d. log-normal measures (the synthetic panel's vol is persistent, so a 30-day window
    would legitimately sit in another regime than the full history)."""
    rng = np.random.default_rng(seed)
    rv = np.exp(rng.normal(np.log(4e-4), 0.5, n))
    idx = pd.date_range("2022-01-01", periods=n, freq="D", name="date")
    return pd.DataFrame(
        {
            "rv": rv,
            "bpv": rv * rng.uniform(0.8, 1.0, n),
            "rs_pos": rv * 0.5,
            "rs_neg": rv * 0.5,
            "rq": rv**2 * 3.0,
            "ret": np.sqrt(rv) * rng.standard_normal(n),
            "complete": True,
        },
        index=idx,
    )


def test_no_drift_in_stationary_data_and_full_drift_after_a_level_shift() -> None:
    real = _realized()
    calm = compute_drift(real, "BTCUSDT", train_end=real.index[-31])
    assert calm.status == "ok"
    assert set(calm.features) == set(FEATURES)
    assert calm.share_drifted < 0.5

    shifted = real.copy()
    for c in ("rv", "bpv", "rs_pos", "rs_neg"):
        shifted.iloc[-30:, shifted.columns.get_loc(c)] *= 50.0
    shifted.iloc[-30:, shifted.columns.get_loc("rq")] *= 2500.0
    shifted.iloc[-30:, shifted.columns.get_loc("ret")] *= 7.0
    rep = compute_drift(shifted, "BTCUSDT", train_end=real.index[-31])
    assert rep.share_drifted == pytest.approx(1.0)
    assert all(v["drifted"] for v in rep.features.values())


def test_stationary_series_has_a_low_false_positive_rate_across_seeds() -> None:
    shares = []
    for seed in range(40):
        real = _realized(400, seed)
        rep = compute_drift(real, "BTCUSDT", train_end=real.index[-31])
        shares.append(rep.share_drifted)
    arr = np.asarray(shares)
    assert arr.mean() < 0.1  # share of features flagged on average
    assert (arr >= 0.5).mean() <= 0.1  # at most 10% of seeds flag half the features


def test_moderate_level_shift_is_detected() -> None:
    real = _realized(400, 3)
    shifted = real.copy()
    f = math.exp(0.5 * 1.0)  # +1 reference s.d. in log rv (sd is 0.5 by construction -> exp(0.5))
    for c in ("rv", "bpv", "rs_pos", "rs_neg"):
        shifted.iloc[-30:, shifted.columns.get_loc(c)] *= f
    shifted.iloc[-30:, shifted.columns.get_loc("rq")] *= f * f
    shifted.iloc[-30:, shifted.columns.get_loc("ret")] *= math.sqrt(f)
    rep = compute_drift(shifted, "BTCUSDT", train_end=real.index[-31])
    assert rep.share_drifted >= 0.5


def test_insufficient_reference_is_reported_not_raised() -> None:
    rep = compute_drift(_realized(40), "BTCUSDT")
    assert rep.status == "insufficient_data"
    assert math.isnan(rep.share_drifted)


def test_report_files_are_written(tmp_path: Path) -> None:
    real = _realized()
    rep = compute_drift(real, "BTCUSDT", train_end=real.index[-31])
    jp, hp = write_drift_report(rep, tmp_path / "drift")
    assert jp == tmp_path / "drift" / "BTCUSDT" / f"{rep.as_of_date}.json"
    data = json.loads(jp.read_text())
    assert data["asset"] == "BTCUSDT"
    assert "<table>" in hp.read_text()
