"""The Prometheus rules have ONE source: the chart's ``rules/volforecast-rules.yaml``.

Compose mounts that directory and the Helm PrometheusRule renders its groups via ``.Files.Get``
(adding only the release-namespace matcher), so the two deployments cannot diverge. These tests pin
that wiring and the shape of the freshness-SLO rules; the PromQL itself is exercised by
``promtool test rules deploy/monitoring/alerts/tests/*.yaml`` (docs/platform.md, section 5).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
RULES = ROOT / "deploy" / "helm" / "volforecast" / "rules" / "volforecast-rules.yaml"
TEMPLATE = ROOT / "deploy" / "helm" / "volforecast" / "templates" / "prometheusrule.yaml"
PROMTOOL_TESTS = ROOT / "deploy" / "monitoring" / "alerts" / "tests"

FRESH = "volforecast:forecast_fresh_at_0100:bool"
SLO_RECORDS = {FRESH, "volforecast:forecast_fresh_at_0100:ratio_30d"}
SLO_ALERTS = {"ForecastLate", "ForecastFreshnessConsecutiveMisses", "ForecastFreshnessBudgetBurn"}
LEGACY_ALERTS = {
    "DataStale",
    "PipelineFailed",
    "APIErrorRate",
    "APILatencyP95",
    "ModelUnderperformsHAR",
    "VaRBreachCluster",
}


@pytest.fixture(scope="module")
def groups() -> list[dict[str, Any]]:
    doc = yaml.safe_load(RULES.read_text())
    return list(doc["groups"])


def _rules(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for g in groups for r in g["rules"]]


def test_single_source_no_second_copy() -> None:
    assert RULES.is_file()
    assert not (ROOT / "deploy" / "monitoring" / "alerts" / "volforecast-rules.yaml").exists()


def test_helm_template_renders_the_shared_file_without_inline_rules() -> None:
    text = TEMPLATE.read_text()
    assert '.Files.Get "rules/volforecast-rules.yaml"' in text
    assert "toYaml $rules.groups" in text
    # no rule is spelled out in the template: a second copy would silently drift
    assert not re.search(r"^\s*-\s*(alert|record):", text, flags=re.M)
    assert "expr:" not in text


def test_compose_mounts_the_chart_rules_dir_and_prometheus_globs_it() -> None:
    compose = (ROOT / "compose.yaml").read_text()
    assert "./deploy/helm/volforecast/rules:/etc/prometheus/alerts:ro" in compose
    prom = yaml.safe_load((ROOT / "deploy" / "monitoring" / "prometheus.yml").read_text())
    assert prom["rule_files"] == ["/etc/prometheus/alerts/*.yaml"]
    # only rule files in the mounted dir (a stray yaml would be loaded by the glob)
    assert [p.name for p in RULES.parent.iterdir()] == [RULES.name]


def test_expected_rules_present(groups: list[dict[str, Any]]) -> None:
    alerts = {r["alert"] for r in _rules(groups) if "alert" in r}
    records = {r["record"] for r in _rules(groups) if "record" in r}
    assert alerts == LEGACY_ALERTS | SLO_ALERTS
    assert records >= SLO_RECORDS


def test_every_rule_is_well_formed(groups: list[dict[str, Any]]) -> None:
    for r in _rules(groups):
        assert bool(r.get("alert")) != bool(r.get("record")), r
        assert str(r["expr"]).strip()
        if "alert" in r:
            assert r["labels"]["severity"] in {"critical", "warning"}
            assert r["annotations"]["summary"]


def test_http_selectors_can_be_namespace_scoped_by_the_chart() -> None:
    """The template inserts ``namespace="..."`` after ``{`` of these two metrics."""
    text = RULES.read_text()
    for metric in ("http_requests_total", "http_request_duration_seconds_bucket"):
        assert re.search(rf"{metric}(?!\{{)", text) is None, f"{metric} without braces"
        assert re.search(rf"{metric}\{{", text)
    assert ",}" not in text  # the template strips the ``,}`` that ``{}`` selectors turn into


def test_freshness_recording_rule_is_a_once_a_day_sample(groups: list[dict[str, Any]]) -> None:
    slo = next(g for g in groups if g["name"] == "volforecast.slo")
    assert slo["interval"] == "1m"  # one evaluation inside the 01:00 minute -> one sample per day
    rule = next(r for r in slo["rules"] if r.get("record") == FRESH)
    expr = " ".join(str(rule["expr"]).split())
    assert "hour() == 1 and minute() == 0" in expr
    assert 'volforecast_forecast_age_seconds{horizon="1"}' in expr
    assert "< bool 86400" in expr
    assert "or on() vector(0)" in expr  # a missing series is a miss


def test_slo_alerts_use_range_selectors_on_the_sparse_series(groups: list[dict[str, Any]]) -> None:
    """The recorded series exists one minute per day: instant selectors would be empty."""
    for r in _rules(groups):
        if r.get("alert") in {"ForecastFreshnessConsecutiveMisses"}:
            expr = str(r["expr"])
            assert FRESH in expr
            assert re.search(rf"{re.escape(FRESH)}(?!\[)", expr) is None, expr


def test_promtool_unit_tests_reference_the_shared_file() -> None:
    files = sorted(PROMTOOL_TESTS.glob("*.yaml"))
    assert files
    for f in files:
        doc = yaml.safe_load(f.read_text())
        for rf in doc["rule_files"]:
            assert (f.parent / rf).resolve() == RULES


def test_docs_name_the_new_rules() -> None:
    slo = (ROOT / "docs" / "slo.md").read_text()
    for name in sorted(SLO_ALERTS | SLO_RECORDS):
        assert name in slo, name
