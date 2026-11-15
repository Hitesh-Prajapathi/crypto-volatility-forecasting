from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from volforecast.evaluation.losses import qlike
from volforecast.evaluation.report import _main_table, evaluate_block, render_markdown
from volforecast.evaluation.stats_tests import diebold_mariano


def _synthetic_block() -> tuple[pd.Series, pd.DataFrame]:
    rng = np.random.default_rng(7)
    idx = pd.date_range("2021-01-01", periods=300, freq="D")
    rv = pd.Series(np.exp(rng.normal(-6, 0.5, 300)), index=idx)
    fc = pd.DataFrame(
        {
            "har": rv * np.exp(rng.normal(0, 0.5, 300)),
            "log_har_dow": rv * np.exp(rng.normal(0, 0.4, 300)),
            "xgb": rv * np.exp(rng.normal(0, 0.3, 300)),
        },
        index=idx,
    )
    return rv, fc


def test_second_benchmark_matches_direct_calls() -> None:
    rv, fc = _synthetic_block()
    h = 5
    block = evaluate_block(rv, fc, h, "har", 50, 0, benchmark2="log_har_dow")
    ql = {m: pd.Series(qlike(rv.to_numpy(), fc[m].to_numpy())) for m in fc}
    for m in ("har", "xgb"):
        b2 = block["models"][m]["vs_benchmark2"]
        assert b2["ratio"] == pytest.approx(ql[m].mean() / ql["log_har_dow"].mean())
        dm = diebold_mariano(ql[m], ql["log_har_dow"], h=h, alternative="less")
        assert b2["dm_p"] == pytest.approx(dm.pvalue)
    own = block["models"]["log_har_dow"]["vs_benchmark2"]
    assert own["ratio"] == pytest.approx(1.0)
    assert own["dm_p"] is None

    table = _main_table(block, "har", "log_har_dow")
    assert "ratio vs log_har_dow" in table
    assert "DM p (beats log_har_dow)" in table
    row = next(r for r in table.splitlines() if r.startswith("| log_har_dow"))
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert cells[4:6] == ["-", "-"]


def test_second_benchmark_omitted_when_absent() -> None:
    rv, fc = _synthetic_block()
    fc = fc.drop(columns=["log_har_dow"])
    block = evaluate_block(rv, fc, 1, "har", 50, 0, benchmark2="log_har_dow")
    assert all("vs_benchmark2" not in r for r in block["models"].values())
    assert "log_har_dow" not in _main_table(block, "har", "log_har_dow")


def test_methodology_mentions_second_benchmark_only_when_set() -> None:
    meta = {"window": "expanding", "first_origin": "2020-01-01", "min_train": 10}
    meta |= {"mcs_reps": 5, "seed": 0}
    base = {"benchmark": "har", "meta": meta, "assets": {}}
    assert "ratio vs log_har_dow" not in render_markdown({**base, "benchmark2": None})
    assert "ratio vs log_har_dow" in render_markdown({**base, "benchmark2": "log_har_dow"})
