"""Aggregate the per-seed LSTM reports (``<dir>/results.json``) into one markdown table.

    uv run python scripts/kaggle/aggregate_seeds.py artifacts/kaggle/reports_s{0,1,2,3,4} > table.md

Each directory is the output of ``python -m volforecast report --benchmark2 log_har_dow`` for one
seed. For every asset and horizon the table gives:

* mean +/- sample std (ddof=1) over seeds of the LSTM QLIKE ratio vs HAR and vs log_har_dow
  (a ratio below 1 means the LSTM has the lower QLIKE);
* the one-sided Diebold-Mariano p-value of the report (``alternative="less"`` on the QLIKE loss
  differential, so a small p means the LSTM is significantly better than the benchmark), averaged
  over seeds, and in how many seeds p < 0.05;
* the LSTM/xgb QLIKE ratio per seed (< 1: LSTM better), mean +/- std;
* xgb/HAR and log_har_dow/HAR QLIKE ratios, mean +/- std. These models are seed-independent, so
  the std must be 0.000 (a non-zero value flags a non-reproducible baseline);
* in how many seeds the LSTM is in the 90% model confidence set.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

DM_ALPHA = 0.05


def _ms(v: list[float]) -> str:
    a = np.asarray(v, dtype=float)
    return f"{a.mean():.3f} ± {a.std(ddof=1) if len(a) > 1 else 0.0:.3f}"


def _p(v: list[float], n: int) -> str:
    return f"{np.mean(v):.3f} ({sum(p < DM_ALPHA for p in v)}/{n})"


def _load(d: Path) -> dict[str, Any]:
    return json.loads((d / "results.json").read_text())


def aggregate(dirs: list[Path]) -> str:
    runs = [_load(d) for d in dirs]
    n = len(runs)
    assets = list(runs[0]["assets"])
    lines = [
        "| asset | h | n | LSTM/HAR | LSTM/log_har_dow | DM p vs HAR (seeds p<0.05) "
        "| DM p vs log_har_dow (seeds p<0.05) | LSTM/xgb | xgb/HAR | log_har_dow/HAR "
        "| LSTM in MCS 90% |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for asset in assets:
        for h in runs[0]["assets"][asset]["horizons"]:
            ms = [r["assets"][asset]["horizons"][h]["models"] for r in runs]
            nobs = {r["assets"][asset]["horizons"][h]["n"] for r in runs}
            lstm = [m["lstm"] for m in ms]
            r_har = [m["qlike_ratio"] for m in lstm]
            r_b2 = [m["vs_benchmark2"]["ratio"] for m in lstm]
            p_har = [m["dm_p"] for m in lstm]
            p_b2 = [m["vs_benchmark2"]["dm_p"] for m in lstm]
            r_xgb = [m["lstm"]["qlike"] / m["xgb"]["qlike"] for m in ms]
            xgb_har = [m["xgb"]["qlike_ratio"] for m in ms]
            lhd_har = [m["log_har_dow"]["qlike_ratio"] for m in ms]
            mcs = sum(bool(m["in_mcs90"]) for m in lstm)
            n_txt = "/".join(map(str, sorted(nobs)))
            lines.append(
                f"| {asset} | {h} | {n_txt} | {_ms(r_har)} | {_ms(r_b2)} "
                f"| {_p(p_har, n)} | {_p(p_b2, n)} "
                f"| {_ms(r_xgb)} | {_ms(xgb_har)} | {_ms(lhd_har)} | {mcs}/{n} |"
            )
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    print(aggregate([Path(a) for a in sys.argv[1:]]))
