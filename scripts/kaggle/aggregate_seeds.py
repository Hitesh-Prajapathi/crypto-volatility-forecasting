"""Aggregate the per-seed LSTM reports (``reports_lstm_s<k>/results.json``) into one table.

    uv run python scripts/kaggle/aggregate_seeds.py reports_lstm_s0 reports_lstm_s1 ... > table.md

For every asset and horizon: mean +/- std (ddof=1) over seeds of the LSTM QLIKE ratio vs HAR and
vs log_har_dow, the DM p-values (one-sided, 'beats'; mean, and in how many seeds p < 0.05), the
LSTM/xgb QLIKE ratio (xgb is a seed-independent CPU model) and how often the LSTM is in the 90% MCS.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def _ms(v: list[float]) -> str:
    a = np.asarray(v, dtype=float)
    return f"{a.mean():.3f} ± {a.std(ddof=1) if len(a) > 1 else 0.0:.3f}"


def aggregate(dirs: list[Path]) -> str:
    runs = [json.loads((d / "results.json").read_text()) for d in dirs]
    n = len(runs)
    assets = list(runs[0]["assets"])
    lines = [
        "| asset | h | n | LSTM/HAR | LSTM/log_har_dow | DM p vs HAR | DM p vs log_har_dow "
        "(seeds p<0.05) | LSTM/xgb | xgb/HAR | log_har_dow/HAR | LSTM in MCS 90% |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for asset in assets:
        for h in ("1", "5"):
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
                f"| {np.mean(p_har):.3f} ({sum(p < 0.05 for p in p_har)}/{n}) "
                f"| {np.mean(p_b2):.3f} ({sum(p < 0.05 for p in p_b2)}/{n}) "
                f"| {_ms(r_xgb)} | {np.mean(xgb_har):.3f} | {np.mean(lhd_har):.3f} | {mcs}/{n} |"
            )
    return "\n".join(lines)


if __name__ == "__main__":
    print(aggregate([Path(a) for a in sys.argv[1:]]))
