"""Regenerate ``golden.json`` (and optionally the frozen fixture) for the model regression gate.

THIS IS A DELIBERATE, REVIEWED ACTION. The golden values pin the behaviour of HAR / log-HAR /
HARQ; updating them hides a regression unless the change in model behaviour is intended. Run it
only when a model or feature change is meant to move the numbers, and include the diff of
``golden.json`` (and the reason) in the pull request for review.

    uv run python -m tests.regression.update_golden                  # recompute golden values
    uv run python -m tests.regression.update_golden --rebuild-fixture --data-root /path/to/data

``--rebuild-fixture`` re-slices the frozen daily panel from real klines (BTCUSDT, 2021-2023)
and is even rarer: it changes the data the gate is defined on.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tests.regression import gate_lib as g


def rebuild_fixture(data_root: Path) -> None:
    from volforecast.pipeline.panel import build_panel

    panel, digest = build_panel(data_root, g.ASSET)
    sliced = panel.loc[g.FIXTURE_START : g.FIXTURE_END]
    g.FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    sliced.to_parquet(g.FIXTURE)
    print(f"fixture written: {g.FIXTURE} ({len(sliced)} rows, source panel hash {digest[:12]})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--rebuild-fixture", action="store_true")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args()
    if args.rebuild_fixture:
        rebuild_fixture(args.data_root)

    m = g.compute_metrics()
    golden = {
        "fixture": g.FIXTURE.name,
        "window": {
            "first_origin": g.FIRST_ORIGIN,
            "min_train": g.MIN_TRAIN,
            "horizon": 1,
            "refit_every": 1,
        },
        "n_origins_scored": m["n_origins_scored"],
        "har_qlike": m["qlike"][g.BENCHMARK],
        "qlike_ratio_vs_har": {
            k: v for k, v in m["qlike_ratio_vs_har"].items() if k != g.BENCHMARK
        },
        "ratio_thresholds": {
            k: v + g.THRESHOLD_MARGIN
            for k, v in m["qlike_ratio_vs_har"].items()
            if k != g.BENCHMARK
        },
        "threshold_margin": g.THRESHOLD_MARGIN,
        "var99_normal_har": m["var99_normal_har"],
    }
    g.GOLDEN.write_text(json.dumps(golden, indent=2) + "\n")
    print(json.dumps(golden, indent=2))


if __name__ == "__main__":
    main()
