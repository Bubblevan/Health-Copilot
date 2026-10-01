from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import pytest

from health_ai_copilot.research.memory.change_mention_admission_v1 import (
    ChangeMention,
    StateObservation,
    admit_change_mention,
    project_as_of,
    project_change,
    project_current,
)


SLOT = ("scope-a", "SELF", "DAILY_ROUTINE", "TEA_BREAK_TIME")
AT = datetime(2023, 5, 4, 11, 22, tzinfo=timezone.utc)
QUOTE = "I have my tea break at 2:30 pm instead of 3 pm."


def _mention(
    *, slot: tuple[str, str, str, str] = SLOT, quote: str = QUOTE
) -> ChangeMention:
    return ChangeMention(
        mention_id="change-1",
        slot_key=slot,
        old_value_text="3 pm",
        new_value_text="2:30 pm",
        occurred_at=AT,
        source_turn_id="turn-30",
        source_text=quote,
        owner_assertion_evidence=quote,
        provenance_sha256=hashlib.sha256(quote.encode("utf-8")).hexdigest(),
    )


def _prior(
    memory_id: str = "memory-1",
    *,
    slot: tuple[str, str, str, str] = SLOT,
    value: str = "3 pm",
    at: datetime = datetime(2023, 4, 1, tzinfo=timezone.utc),
) -> StateObservation:
    return StateObservation(memory_id, slot, value, at)


def test_unanchored_change_mention_updates_current_without_claiming_old_as_of() -> None:
    decision = admit_change_mention(_mention())

    assert decision.kind == "UNANCHORED_CHANGE_MENTION"
    assert project_current(decision)["value_text"] == "2:30 pm"
    assert project_as_of(decision, datetime(2023, 5, 1, tzinfo=timezone.utc)) == {
        "status": "UNRESOLVED",
        "value_text": None,
        "valid_from": None,
        "reason": "predecessor_interval_not_independently_anchored",
    }
    assert project_change(decision)["old_value_text"] == "3 pm"
    assert project_change(decision)["new_value_text"] == "2:30 pm"
    assert project_change(decision)["predecessor_status"] == "UNANCHORED_CHANGE_MENTION"


def test_unique_earlier_same_slot_observation_admits_revision_and_historical_query() -> None:
    decision = admit_change_mention(_mention(), [_prior()])

    assert decision.kind == "ANCHORED_REVISION"
    assert decision.predecessor_memory_id == "memory-1"
    historical = project_as_of(decision, datetime(2023, 5, 1, tzinfo=timezone.utc))
    assert historical["status"] == "HISTORICAL"
    assert historical["value_text"] == "3 pm"
    assert project_as_of(decision, AT)["value_text"] == "2:30 pm"


def test_wrong_scope_value_or_non_prior_observation_does_not_anchor() -> None:
    wrong_scope = _prior(slot=("scope-b", "SELF", "DAILY_ROUTINE", "TEA_BREAK_TIME"))
    wrong_value = _prior(value="4 pm")
    same_time = _prior(at=AT)

    for row in (wrong_scope, wrong_value, same_time):
        assert admit_change_mention(_mention(), [row]).kind == "UNANCHORED_CHANGE_MENTION"


def test_multiple_prior_matches_fail_closed_instead_of_choosing_arbitrarily() -> None:
    decision = admit_change_mention(_mention(), [_prior("m1"), _prior("m2")])

    assert decision.kind == "UNANCHORED_CHANGE_MENTION"
    assert decision.reason == "multiple_prior_old_value_observations_require_store_coalescing"


def test_change_mention_requires_source_ordered_values_and_valid_provenance() -> None:
    bad_hash = "0" * 64
    with pytest.raises(ValueError, match="provenance hash"):
        ChangeMention(
            "change-1", SLOT, "3 pm", "2:30 pm", AT, "turn-30", QUOTE, QUOTE, bad_hash
        )
    with pytest.raises(ValueError, match="new value before old value"):
        reversed_quote = "I have my tea break at 3 pm instead of 2:30 pm."
        ChangeMention(
            "change-2",
            SLOT,
            "3 pm",
            "2:30 pm",
            AT,
            "turn-30",
            reversed_quote,
            reversed_quote,
            hashlib.sha256(reversed_quote.encode("utf-8")).hexdigest(),
        )


def test_change_mention_rejects_third_person_evidence_for_self_slot() -> None:
    quote = "The patient now takes tea at 2:30 pm instead of 3 pm."
    with pytest.raises(ValueError, match="first-person attribution"):
        ChangeMention(
            "change-3",
            SLOT,
            "3 pm",
            "2:30 pm",
            AT,
            "turn-31",
            quote,
            quote,
            hashlib.sha256(quote.encode("utf-8")).hexdigest(),
        )


def test_first_person_in_another_sentence_cannot_authorize_third_person_change() -> None:
    quote = "I like tea. The patient now takes 2:30 pm instead of 3 pm."
    with pytest.raises(ValueError, match="first-person attribution"):
        ChangeMention(
            "change-4",
            SLOT,
            "3 pm",
            "2:30 pm",
            AT,
            "turn-32",
            quote,
            "The patient now takes 2:30 pm instead of 3 pm",
            hashlib.sha256(quote.encode("utf-8")).hexdigest(),
        )


def test_change_times_must_be_timezone_aware() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        StateObservation("m1", SLOT, "3 pm", datetime(2023, 4, 1))
