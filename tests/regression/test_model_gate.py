"""CI regression gate: frozen real-data panel, fixed window, golden values (plan M7 P7.1).

* HAR mean QLIKE must reproduce the golden value to 1e-9 (determinism / no silent change);
* each challenger's QLIKE ratio vs HAR must stay at or below its stored threshold;
* the 99% normal VaR from HAR must pass Kupiec at the 5% level.

Golden values are regenerated only deliberately: see ``update_golden.py``.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from tests.regression import gate_lib as g


@pytest.fixture(scope="module")
def golden() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(g.GOLDEN.read_text())
    return data


@pytest.fixture(scope="module")
def metrics() -> dict[str, Any]:
    return g.compute_metrics()


def test_har_qlike_matches_golden(metrics: dict[str, Any], golden: dict[str, Any]) -> None:
    assert metrics["n_origins_scored"] == golden["n_origins_scored"]
    assert metrics["qlike"]["har"] == pytest.approx(golden["har_qlike"], abs=1e-9, rel=0)


@pytest.mark.parametrize("challenger", ["log_har", "harq"])
def test_challenger_not_worse_than_threshold(
    metrics: dict[str, Any], golden: dict[str, Any], challenger: str
) -> None:
    ratio = metrics["qlike_ratio_vs_har"][challenger]
    limit = golden["ratio_thresholds"][challenger]
    assert ratio <= limit, (
        f"{challenger} QLIKE ratio vs HAR regressed: {ratio:.4f} > {limit:.4f} "
        f"(golden {golden['qlike_ratio_vs_har'][challenger]:.4f})"
    )


def test_har_var99_passes_kupiec(metrics: dict[str, Any]) -> None:
    assert metrics["var99_normal_har"]["kupiec_p"] > 0.05, metrics["var99_normal_har"]
