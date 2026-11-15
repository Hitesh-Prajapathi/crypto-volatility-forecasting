"""Retrain flow: challenger fit, gate, alias moves, trigger handling (tiny config, file MLflow)."""

from __future__ import annotations

import pandas as pd

from tests.unit.flows.conftest import ASSET, seed_champion
from volforecast.flows.common import Runtime
from volforecast.flows.retrain import requested_by_evaluate, retrain_pair
from volforecast.registry.client import CHALLENGER, CHAMPION, alias_version, registered_model_name


def test_retrain_without_champion_records_a_decision_and_sets_aliases(
    runtime: Runtime, synthetic_panel_400: pd.DataFrame
) -> None:
    out = retrain_pair(runtime, ASSET, 1)
    assert out.status in {"promoted", "rejected"}
    assert out.model_name in {"log_har", "har"}
    decisions = runtime.repo.promotion_decisions(ASSET, 1)
    assert len(decisions) == 1
    assert decisions[0]["promoted"] is (out.status == "promoted")
    assert decisions[0]["as_of_date"] == synthetic_panel_400.index[-1].date()
    name = registered_model_name(ASSET, 1)
    champ = alias_version(runtime.client, name, CHAMPION)
    chall = alias_version(runtime.client, name, CHALLENGER)
    if out.status == "promoted":
        assert champ is not None
        assert str(champ.version) == out.challenger_version
        assert chall is None  # the challenger alias is dropped once it is the champion
    else:
        assert champ is None  # nothing was promoted over "no champion"
        assert chall is not None
        assert str(chall.version) == out.challenger_version


def test_alias_only_moves_when_the_gate_passes(
    runtime: Runtime, synthetic_panel_400: pd.DataFrame
) -> None:
    seed_champion(runtime, synthetic_panel_400, 300, name="har")
    name = registered_model_name(ASSET, 1)
    out = retrain_pair(runtime, ASSET, 1)
    champ = alias_version(runtime.client, name, CHAMPION)
    assert champ is not None
    if out.status == "promoted":
        assert str(champ.version) == out.challenger_version != "1"
    else:
        assert str(champ.version) == "1"  # the seeded champion stays
    row = runtime.repo.promotion_decisions(ASSET, 1)[0]
    assert row["champion_version"] == "1"
    assert row["reasons"]


def test_rerun_for_the_same_day_is_skipped_unless_forced(
    runtime: Runtime, synthetic_panel_400: pd.DataFrame
) -> None:
    retrain_pair(runtime, ASSET, 1)
    assert retrain_pair(runtime, ASSET, 1).status == "skipped"
    assert len(runtime.repo.promotion_decisions(ASSET, 1)) == 1
    forced = retrain_pair(runtime, ASSET, 1, force=True)
    assert forced.status != "skipped"
    assert len(runtime.repo.promotion_decisions(ASSET, 1)) == 2


def test_as_of_truncates_the_training_panel(
    runtime: Runtime, synthetic_panel_400: pd.DataFrame
) -> None:
    as_of = str(synthetic_panel_400.index[-31].date())
    out = retrain_pair(runtime, ASSET, 1, as_of=as_of)
    assert out.as_of_date == as_of
    mv = runtime.client.get_model_version(registered_model_name(ASSET, 1), out.challenger_version)
    assert mv.tags["train_end_date"] == as_of


def test_requested_by_evaluate_reads_the_latest_successful_run(runtime: Runtime) -> None:
    assert requested_by_evaluate(runtime) == {}
    rid = runtime.repo.start_run("evaluate")
    runtime.repo.finish_run(rid, "success", {"retrain_requests": {ASSET: [1, 5]}})
    assert requested_by_evaluate(runtime) == {ASSET: [1, 5]}
    rid = runtime.repo.start_run("evaluate")
    runtime.repo.finish_run(rid, "failed", {"retrain_requests": {ASSET: [1]}})
    assert requested_by_evaluate(runtime) == {}
