"""Unit tests for scripts/kaggle/aggregate_seeds.py (loaded by path: scripts is not a package)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "kaggle" / "aggregate_seeds.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("aggregate_seeds", _SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _block(lstm_q: float, p_har: float, p_b2: float, in_mcs: bool) -> dict[str, object]:
    har, ldow, xgb = 0.40, 0.30, 0.25
    return {
        "n": 100,
        "models": {
            "har": {"qlike": har, "qlike_ratio": 1.0, "dm_p": None, "in_mcs90": False},
            "log_har_dow": {"qlike": ldow, "qlike_ratio": ldow / har, "in_mcs90": False},
            "xgb": {"qlike": xgb, "qlike_ratio": xgb / har, "in_mcs90": True},
            "lstm": {
                "qlike": lstm_q,
                "qlike_ratio": lstm_q / har,
                "dm_p": p_har,
                "vs_benchmark2": {"ratio": lstm_q / ldow, "dm_p": p_b2},
                "in_mcs90": in_mcs,
            },
        },
    }


def _write(path: Path, blocks: dict[str, dict[str, object]]) -> Path:
    path.mkdir()
    (path / "results.json").write_text(json.dumps({"assets": {"BTCUSDT": {"horizons": blocks}}}))
    return path


def test_aggregate_synthetic_three_seeds(tmp_path: Path) -> None:
    mod = _load_script()
    lstm_q = [0.26, 0.27, 0.28]
    p_har = [1e-6, 2e-6, 3e-6]
    p_b2 = [0.01, 0.04, 0.20]  # 2 of 3 seeds significant at 5%
    mcs = [True, True, False]
    dirs = [
        _write(
            tmp_path / f"s{k}",
            {
                "1": _block(lstm_q[k], p_har[k], p_b2[k], mcs[k]),
                "5": _block(lstm_q[k], p_har[k], p_b2[k], mcs[k]),
            },
        )
        for k in range(3)
    ]
    out = mod.aggregate(dirs)
    rows = [ln for ln in out.splitlines() if ln.startswith("| BTCUSDT")]
    assert len(rows) == 2  # both horizons come from the data, not a hard-coded list
    cells = [c.strip() for c in rows[0].strip("|").split("|")]
    sd = np.std(np.array(lstm_q) / 0.40, ddof=1)  # sample std (ddof=1), not population
    assert cells[3] == f"{np.mean(lstm_q) / 0.40:.3f} ± {sd:.3f}"
    sd_b2 = np.std(np.array(lstm_q) / 0.30, ddof=1)
    assert cells[4] == f"{np.mean(lstm_q) / 0.30:.3f} ± {sd_b2:.3f}"
    # DM p is the one-sided "LSTM beats benchmark" p-value: reported as is, not flipped
    assert cells[5] == "0.000 (3/3)"
    assert cells[6] == f"{np.mean(p_b2):.3f} (2/3)"
    # LSTM / xgb ratio is lstm qlike over xgb qlike, per seed
    sd_x = np.std(np.array(lstm_q) / 0.25, ddof=1)
    assert cells[7] == f"{np.mean(lstm_q) / 0.25:.3f} ± {sd_x:.3f}"
    # seed-independent baselines show zero spread
    assert cells[8] == "0.625 ± 0.000"
    assert cells[9] == "0.750 ± 0.000"
    assert cells[10] == "2/3"


def test_single_seed_has_zero_std(tmp_path: Path) -> None:
    mod = _load_script()
    d = _write(tmp_path / "s0", {"1": _block(0.26, 1e-6, 0.5, True)})
    out = mod.aggregate([d])
    assert "0.650 ± 0.000" in out
