from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.unit.pipeline.helpers import make_cfg, quick_forecasts
from volforecast.contracts import COL_RV
from volforecast.evaluation.report import (
    README_END,
    README_START,
    align_models,
    build_report,
    inject_readme,
    regime_table,
    subperiod_table,
    trailing_rv,
)

MODELS = ["har", "log_har", "naive", "ewma"]


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    from tests.conftest import make_synthetic_panel

    panel = make_synthetic_panel()
    cfg = make_cfg()
    fc = quick_forecasts(panel, cfg)
    out = tmp_path_factory.mktemp("reports")
    results = build_report(fc, {"SYN": panel}, cfg, out)
    return {"panel": panel, "cfg": cfg, "fc": fc, "out": out, "results": results}


def test_align_models_drops_origins_with_any_nan(run: dict[str, object]) -> None:
    fc: pd.DataFrame = run["fc"]  # type: ignore[assignment]
    rv0, f0, ex0 = align_models(fc, "SYN", 1, MODELS)
    assert ex0 == []
    assert list(f0.columns) == MODELS
    assert f0.index.equals(rv0.index)

    broken = fc.copy()
    dates = broken[broken["model"] == "naive"]["origin_date"].iloc[20:25]
    broken.loc[(broken["model"] == "naive") & broken["origin_date"].isin(dates), "forecast_var"] = (
        np.nan
    )
    d40 = broken[(broken["model"] == "har") & (broken["horizon"] == 1)]["origin_date"].iloc[40]
    # realized NaN (for every model at that origin) is dropped as well
    broken.loc[(broken["origin_date"] == d40) & (broken["horizon"] == 1), "realized_var"] = np.nan
    rv1, f1, _ = align_models(broken, "SYN", 1, MODELS)
    assert len(rv1) == len(rv0) - 5 - 1
    assert not set(dates) & set(rv1.index)
    assert np.isfinite(f1.to_numpy()).all()
    assert np.isfinite(rv1.to_numpy()).all()


def test_all_nan_model_is_excluded_not_fatal(run: dict[str, object]) -> None:
    fc: pd.DataFrame = run["fc"]  # type: ignore[assignment]
    broken = fc.copy()
    broken.loc[broken["model"] == "ewma", "forecast_var"] = np.nan
    rv, f, excluded = align_models(broken, "SYN", 5, MODELS)
    assert excluded == ["ewma"]
    assert "ewma" not in f.columns
    assert len(rv) > 0


def test_results_shapes(run: dict[str, object]) -> None:
    res: dict[str, object] = run["results"]  # type: ignore[assignment]
    block = res["assets"]["SYN"]["horizons"]  # type: ignore[index]
    assert set(block) == {"1", "5"}
    for h, b in block.items():
        assert set(b["models"]) == set(MODELS)
        assert b["n"] > 100
        har = b["models"]["har"]
        assert har["qlike_ratio"] == pytest.approx(1.0)
        assert har["dm_p"] is None
        for m in ("log_har", "naive", "ewma"):
            r = b["models"][m]
            assert 0.0 <= r["dm_p"] <= 1.0
            assert 0.0 <= r["mz_r2"] <= 1.0
            assert 0.0 <= r["mz_wald_p"] <= 1.0
        assert any(r["in_mcs90"] for r in b["models"].values())
        assert set(b["mcs"]["included"]) | set(b["mcs"]["excluded"]) == set(MODELS)
        years = b["subperiod"]["ratio"]
        assert set(years) == {"2022", "2023"}
        assert sum(b["subperiod"]["n"].values()) == b["n"]
        assert all(set(v) == set(MODELS) for v in years.values())
        reg = b["regime"]
        assert set(reg["n"]) == {"low", "mid", "high"}
        assert len(reg["cutpoints"]) == 2
        assert sum(reg["n"].values()) == b["n"]
        assert all(v["har"] == pytest.approx(1.0) for v in reg["ratio"].values())
        _ = h
    assert res["var"], "VaR table missing"
    var = pd.DataFrame(res["var"])  # type: ignore[call-overload]
    assert len(var) == len(MODELS) * 3 * 2


def test_report_files_and_figures(run: dict[str, object]) -> None:
    out: Path = run["out"]  # type: ignore[assignment]
    on_disk = json.loads((out / "results.json").read_text())
    assert on_disk["assets"].keys() == {"SYN"}
    md = (out / "tables.md").read_text()
    assert "## SYN" in md
    assert "### h=1" in md
    assert "### h=5" in md
    assert "ratio vs HAR" in md
    assert "MCS 90%" in md
    assert "1-day VaR" in md
    assert "QLIKE ratio vs HAR by calendar year" in md
    assert "tercile" in md
    assert "Kupiec p" in md
    short = (out / "tables_readme.md").read_text()
    assert "### h=1" in short
    assert "tercile" not in short
    assert "Kupiec" not in short
    figs = {p.name for p in (out / "figures").iterdir()}
    assert figs == {"cum_qlike_diff_SYN.png", "forecast_vs_realized_SYN_h1_2022.png"}
    assert all((out / "figures" / f).stat().st_size > 5_000 for f in figs)


def test_report_is_deterministic(run: dict[str, object], tmp_path: Path) -> None:
    again = build_report(
        run["fc"],
        {"SYN": run["panel"]},
        run["cfg"],
        tmp_path,  # type: ignore[arg-type]
    )
    assert again == run["results"]


def test_regime_uses_only_trailing_information(run: dict[str, object]) -> None:
    panel: pd.DataFrame = run["panel"]  # type: ignore[assignment]
    t = pd.Timestamp("2022-06-30")
    changed = panel.copy()
    changed.loc[changed.index > t, COL_RV] *= 50.0
    np.testing.assert_array_equal(
        trailing_rv(panel).loc[:t].to_numpy(), trailing_rv(changed).loc[:t].to_numpy()
    )


def test_regime_and_subperiod_standalone(run: dict[str, object]) -> None:
    fc: pd.DataFrame = run["fc"]  # type: ignore[assignment]
    panel: pd.DataFrame = run["panel"]  # type: ignore[assignment]
    rv, f, _ = align_models(fc, "SYN", 1, MODELS)
    sub = subperiod_table(rv, f, "har")
    assert sub["ratio"]["2022"]["har"] == pytest.approx(1.0)
    reg = regime_table(rv, f, panel, "har")
    assert sum(reg["n"].values()) == len(rv)
    # terciles of the trailing level: roughly equal-sized buckets
    assert max(reg["n"].values()) - min(reg["n"].values()) <= 2


# ── README injection ─────────────────────────────────────────────────────────
README = f"""# Title

intro text

## Results

{README_START}
old results
more old
{README_END}

## Limitations

keep me
"""


def test_inject_readme_replaces_only_marked_block(tmp_path: Path) -> None:
    p = tmp_path / "README.md"
    p.write_text(README)
    inject_readme(p, "| a | b |\n|---|---|\n| 1 | 2 |\n")
    text = p.read_text()
    assert "old results" not in text
    assert "more old" not in text
    assert "| a | b |" in text
    assert text.startswith("# Title\n\nintro text")
    assert text.rstrip().endswith("keep me")
    assert text.count(README_START) == 1
    assert text.count(README_END) == 1
    assert text.index(README_START) < text.index("| a | b |") < text.index(README_END)


def test_inject_readme_idempotent_and_handles_regex_chars(tmp_path: Path) -> None:
    p = tmp_path / "README.md"
    p.write_text(README)
    payload = r"cost \1 \g<0> $& (x) [y]"
    inject_readme(p, payload)
    once = p.read_text()
    inject_readme(p, payload)
    assert p.read_text() == once
    assert payload in once


def test_inject_readme_requires_markers(tmp_path: Path) -> None:
    p = tmp_path / "README.md"
    p.write_text("# no markers here\n")
    with pytest.raises(ValueError, match="markers"):
        inject_readme(p, "x")
    assert p.read_text() == "# no markers here\n"
