"""Static site builder: empty store, synthetic store, JSON hygiene, HTML ids, backtest parsing."""
# ruff: noqa: PT018, E501

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from volforecast.forecast_store import ForecastRepository
from volforecast.site import build_site, parse_backtest_tables

NOW = datetime(2026, 3, 1, 1, 30, tzinfo=UTC)
START = date(2026, 1, 1)
N_DAYS = 40

TABLES_MD = """# Out-of-sample results

## BTCUSDT

### h=1 (n=100, origins 2021-01-01 to 2026-11-12)

| model | QLIKE | ratio vs HAR | DM p (beats HAR) | ratio vs log_har_dow | DM p (beats log_har_dow) | MSE | MZ R2 | MZ Wald p | MCS 90% |
|---|---|---|---|---|---|---|---|---|---|
| har (benchmark) | 0.3628 | 1.000 | - | 1.235 | 0.995 | 4.0e-06 | 0.211 | 0.000 | no |
| log_har_dow | 0.2937 | 0.810 | 0.005 | - | - | 3.7e-06 | 0.266 | 0.322 | no |
| xgb | 0.2599 | 0.717 | 0.000 | 0.885 | 0.000 | 3.8e-06 | 0.254 | 0.000 | yes |

QLIKE ratio vs HAR by calendar year

| model | 2021 (n=359) |
|---|---|
| xgb | 0.962 |

## 1-day VaR / ES backtest (h=1)

#### BTCUSDT, VaR/ES level 1%

| model | x |
|---|---|
| xgb | 1 |
"""


def _store(tmp_path: Path, *, with_data: bool) -> ForecastRepository:
    repo = ForecastRepository(f"sqlite:///{tmp_path / 'store.db'}")
    if not with_data:
        return repo
    rng = np.random.default_rng(0)
    days = [START + timedelta(days=i) for i in range(N_DAYS)]
    rv = rng.uniform(2e-4, 6e-4, N_DAYS + 6)
    panel = pd.DataFrame(
        {
            "rv": rv,
            "bpv": rv,
            "rs_pos": rv / 2,
            "rs_neg": rv / 2,
            "rq": rv**2,
            "ret": rng.normal(0, 0.02, len(rv)),
            "complete": True,
        },
        index=pd.DatetimeIndex([START + timedelta(days=i) for i in range(len(rv))]),
    )
    repo.upsert_realized("BTCUSDT", panel)
    rows: list[dict[str, Any]] = []
    for d in days:
        base = {"asset": "BTCUSDT", "origin_date": d, "data_hash": "x"}
        for h in (1, 5):
            rows.append(
                {**base, "horizon": h, "model_name": "xgb", "model_version": "fallback:xgb",
                 "role": "fallback", "target_date": d + timedelta(days=h),
                 "forecast_var": 4e-4 * h,
                 "var99": 0.05 if h == 1 else None, "var95": 0.03 if h == 1 else None,
                 "es975": None, "dist": "normal" if h == 1 else None}
            )  # fmt: skip
            rows.append(
                {**base, "horizon": h, "model_name": "har_benchmark",
                 "model_version": "har_benchmark", "role": "benchmark",
                 "target_date": d + timedelta(days=h), "forecast_var": 5e-4 * h}
            )  # fmt: skip
    repo.upsert_forecasts(rows)
    return repo


def _walk_no_nan(x: Any) -> None:
    if isinstance(x, float):
        assert np.isfinite(x)
    elif isinstance(x, dict):
        for v in x.values():
            _walk_no_nan(v)
    elif isinstance(x, list):
        for v in x:
            _walk_no_nan(v)


def _build(tmp_path: Path, repo: ForecastRepository, reports: Path | None = None) -> Path:
    return build_site(
        tmp_path / "_site",
        repo.engine.url.render_as_string(hide_password=False),
        reports_dir=reports or tmp_path / "no_reports",
        assets=("BTCUSDT", "ETHUSDT"),
        horizons=(1, 5),
        sha="abc1234",
        generated_at=NOW,
    )


def test_empty_store_builds(tmp_path: Path) -> None:
    out = _build(tmp_path, _store(tmp_path, with_data=False))
    doc = json.loads((out / "data" / "forecasts.json").read_text())
    assert doc["empty"] is True
    assert doc["track_record_start"] is None
    assert doc["git_sha"] == "abc1234"
    assert doc["demo"] is False
    assert set(doc["assets"]) == {"BTCUSDT", "ETHUSDT"}
    assert all(a["horizons"] == {} for a in doc["assets"].values())
    assert not (out / "data" / "backtest_summary.json").exists()
    assert (out / "index.html").is_file() and (out / "app.js").is_file()


def test_synthetic_store_schema(tmp_path: Path) -> None:
    out = _build(tmp_path, _store(tmp_path, with_data=True))
    raw = (out / "data" / "forecasts.json").read_text()
    assert "NaN" not in raw and "Infinity" not in raw
    doc = json.loads(raw)
    _walk_no_nan(doc)
    assert doc["empty"] is False
    assert doc["track_record_start"] == START.isoformat()
    datetime.fromisoformat(doc["generated_at"])
    assert doc["assets"]["ETHUSDT"]["horizons"] == {}
    for h in ("1", "5"):
        blk = doc["assets"]["BTCUSDT"]["horizons"][h]
        assert {"model_name", "model_version", "role", "latest", "history", "cum_qlike"} <= set(blk)
        assert blk["model_version"] == "fallback:xgb" and blk["role"] == "fallback"
        assert blk["n_forecasts"] == N_DAYS
        assert blk["latest"]["origin_date"] == (START + timedelta(days=N_DAYS - 1)).isoformat()
        assert blk["latest"]["forecast_vol"] == pytest.approx(np.sqrt(4e-4 * 365))
        for r in blk["history"]:
            date.fromisoformat(r["origin_date"])
            date.fromisoformat(r["target_date"])
    blk1 = doc["assets"]["BTCUSDT"]["horizons"]["1"]
    assert blk1["latest"]["var99"] == pytest.approx(0.05)
    assert blk1["n_realized"] == N_DAYS
    cum = blk1["cum_qlike"]
    assert cum["n"] == N_DAYS and len(cum["champion"]) == len(cum["dates"]) == len(cum["har"])
    assert doc["assets"]["BTCUSDT"]["horizons"]["5"]["latest"]["var99"] is None


def test_unscored_forecast_gives_null_not_nan(tmp_path: Path) -> None:
    repo = _store(tmp_path, with_data=True)
    repo.upsert_forecasts(
        [{"asset": "BTCUSDT", "horizon": 1, "model_name": "xgb", "model_version": "fallback:xgb",
          "role": "fallback", "origin_date": date(2026, 6, 1), "target_date": date(2026, 6, 2),
          "forecast_var": 3e-4}]
    )  # fmt: skip
    out = _build(tmp_path, repo)
    doc = json.loads((out / "data" / "forecasts.json").read_text())
    last = doc["assets"]["BTCUSDT"]["horizons"]["1"]["history"][-1]
    assert last["realized_var"] is None and last["har_var"] is None


def test_html_has_expected_ids(tmp_path: Path) -> None:
    out = _build(tmp_path, _store(tmp_path, with_data=False))
    html = (out / "index.html").read_text()
    for i in ("disclaimer", "forecast-table", "asset-tabs", "chart-vol", "chart-qlike",
              "backtest", "meta", "demo-banner", "horizon-select"):  # fmt: skip
        assert f'id="{i}"' in html
    assert "plotly.js-basic-dist-min@2." in html  # pinned version
    assert "not investment advice" in html
    assert 'name="viewport"' in html


def test_demo_flag(tmp_path: Path) -> None:
    out = build_site(
        tmp_path / "s", f"sqlite:///{tmp_path / 'd.db'}", reports_dir=tmp_path, demo=True
    )
    assert json.loads((out / "data" / "forecasts.json").read_text())["demo"] is True


def test_backtest_parse_and_build(tmp_path: Path) -> None:
    parsed = parse_backtest_tables(TABLES_MD)["tables"]
    assert list(parsed) == ["BTCUSDT"]
    blk = parsed["BTCUSDT"]["1"]
    assert blk["n"] == 100 and len(blk["rows"]) == 3
    by = {r["model"]: r for r in blk["rows"]}
    assert by["har"]["benchmark"] is True and by["har"]["dm_p_vs_har"] is None
    assert by["xgb"]["ratio_vs_har"] == pytest.approx(0.717)
    assert by["xgb"]["dm_p_vs_log_har_dow"] == 0.0 and by["xgb"]["mcs_90"] is True
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "tables.md").write_text(TABLES_MD)
    out = _build(tmp_path, _store(tmp_path, with_data=False), reports)
    bt = json.loads((out / "data" / "backtest_summary.json").read_text())
    assert "weekend" in bt["note"] and bt["tables"]["BTCUSDT"]["1"]["n"] == 100


def test_committed_tables_parse() -> None:
    from volforecast.config import PROJECT_ROOT

    path = PROJECT_ROOT / "reports" / "tables.md"
    if not path.is_file():
        pytest.skip("reports/tables.md absent")
    t = parse_backtest_tables(path.read_text())["tables"]
    assert {"BTCUSDT", "ETHUSDT", "SOLUSDT"} <= set(t)
    assert {"1", "5"} <= set(t["BTCUSDT"])


def test_publish_store_copy(tmp_path: Path) -> None:
    repo = _store(tmp_path, with_data=True)
    out = build_site(
        tmp_path / "s",
        repo.engine.url.render_as_string(hide_password=False),
        reports_dir=tmp_path,
        publish_store=True,
    )
    copy = ForecastRepository(f"sqlite:///{out / 'data' / 'store.db'}")
    assert copy.latest_forecast("BTCUSDT", 1) is not None
