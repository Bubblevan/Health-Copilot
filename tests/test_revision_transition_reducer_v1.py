from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

from tools.research.memory.revision_transition_reducer_v1 import (
    StateObservation,
    TransitionWitness,
    reduce_transition,
)


FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "docs/research/memory/mem3b0q_r4_reducer_controls_v1.json"
)
FIXTURE_HASH = FIXTURE.with_suffix(FIXTURE.suffix + ".sha256")


def _observation(row: dict) -> StateObservation:
    values = dict(row)
    if values["observed_at"] is not None:
        values["observed_at"] = datetime.fromisoformat(
            values["observed_at"].replace("Z", "+00:00")
        )
    return StateObservation(**values)


def _witness(row: dict | None) -> TransitionWitness | None:
    if row is None:
        return None
    values = dict(row)
    values["event_at"] = datetime.fromisoformat(values["event_at"].replace("Z", "+00:00"))
    return TransitionWitness(**values)


def test_offline_semantic_controls_are_fail_closed_and_non_mutating() -> None:
    digest, filename = FIXTURE_HASH.read_text(encoding="utf-8").strip().split()
    assert filename == FIXTURE.name
    assert digest == hashlib.sha256(FIXTURE.read_bytes()).hexdigest()

    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert fixture["status"] == "OFFLINE_SYNTHETIC_CONTROLS_NOT_MODEL_QUALIFICATION"

    results = {}
    for control in fixture["controls"]:
        decision = reduce_transition(
            _observation(control["previous"]),
            _observation(control["current"]),
            _witness(control.get("witness")),
        )
        results[control["id"]] = decision
        assert decision.decision == control["expected"], (control["id"], decision)
        assert decision.store_mutation == "NONE", control["id"]

    assert len(results) == 16


def test_same_alias_class_is_not_a_stable_object_instance() -> None:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    control = next(
        row
        for row in fixture["controls"]
        if row["id"] == "same_alias_type_without_instance_link_is_unresolved"
    )

    decision = reduce_transition(
        _observation(control["previous"]),
        _observation(control["current"]),
    )

    assert decision.decision == "UNRESOLVED"
    assert decision.reason == "object_instance_identity_missing"


def test_negated_transition_cue_is_not_a_replace_witness() -> None:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    control = next(
        row for row in fixture["controls"] if row["id"] == "negated_replace_wording_does_not_admit"
    )

    decision = reduce_transition(
        _observation(control["previous"]),
        _observation(control["current"]),
        _witness(control["witness"]),
    )

    assert decision.decision == "UNRESOLVED"
    assert decision.reason == "replace_witness_lacks_supported_explicit_pattern"


def test_quoted_change_is_not_admitted_as_owner_assertion() -> None:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    control = next(
        row for row in fixture["controls"] if row["id"] == "quoted_change_is_not_owner_assertion"
    )

    decision = reduce_transition(
        _observation(control["previous"]),
        _observation(control["current"]),
        _witness(control["witness"]),
    )

    assert decision.decision == "UNRESOLVED"
    assert decision.reason == "replace_witness_lacks_supported_explicit_pattern"


def test_backdated_event_cannot_supersede_previously_observed_state() -> None:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    control = next(
        row
        for row in fixture["controls"]
        if row["id"] == "backdated_event_does_not_supersede_later_observation"
    )

    decision = reduce_transition(
        _observation(control["previous"]),
        _observation(control["current"]),
        _witness(control["witness"]),
    )

    assert decision.decision == "UNRESOLVED"
    assert decision.reason == "event_time_not_after_previous_observation"
