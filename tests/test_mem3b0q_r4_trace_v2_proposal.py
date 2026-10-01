from __future__ import annotations

import hashlib

from tools.research.memory.mem3b0q_r4_trace_v2_proposal import (
    TRACE_INSTRUMENTATION_VERSION,
    evaluate_prompt_witnesses,
    normalize_slots_response,
)


SOURCE_ID = "R4-P01"
PROPOSITION = "My workout plan's activity is running."


def _observe(prompt: str | None) -> dict:
    slot = {"id": 0, "id_task": 9, "is_processing": True}
    if prompt is not None:
        slot["prompt"] = prompt
    return normalize_slots_response(
        {"slots": [slot]},
        source_id=SOURCE_ID,
        proposition_text=PROPOSITION,
    )


def test_missing_debug_prompt_fails_closed_without_inventing_a_hash() -> None:
    observation = _observe(None)

    assert observation["status"] == "UNVERIFIED"
    assert observation["failures"] == ["active_prompt_missing:0"]
    assert observation["slots"][0]["prompt_sha256"] is None
    assert observation["slots"][0]["prompt_contains_source_id"] is False


def test_empty_debug_prompt_fails_closed_with_empty_string_hash() -> None:
    observation = _observe("")

    assert observation["status"] == "UNVERIFIED"
    assert observation["slots"][0]["prompt_sha256"] == hashlib.sha256(b"").hexdigest()
    assert "active_prompt_witness_mismatch:0" in observation["failures"]


def test_exact_active_prompt_witness_passes_without_retaining_prompt() -> None:
    prompt = f"source_id={SOURCE_ID}\n{PROPOSITION}"
    observation = _observe(prompt)
    aggregate = evaluate_prompt_witnesses([observation])

    assert observation["status"] == "PASS"
    assert observation["slots"][0]["prompt_sha256"] == hashlib.sha256(
        prompt.encode("utf-8")
    ).hexdigest()
    assert aggregate["status"] == "PASS"
    assert aggregate["active_slot_ids"] == [0]
    assert aggregate["active_task_ids"] == [9]
    assert prompt not in repr(observation)


def test_split_witnesses_across_different_active_slots_do_not_pass() -> None:
    first = normalize_slots_response(
        {"slots": [{"id": 0, "id_task": 9, "is_processing": True,
                    "prompt": f"source_id={SOURCE_ID}"}]},
        source_id=SOURCE_ID,
        proposition_text=PROPOSITION,
    )
    second = normalize_slots_response(
        {"slots": [{"id": 1, "id_task": 10, "is_processing": True,
                    "prompt": PROPOSITION}]},
        source_id=SOURCE_ID,
        proposition_text=PROPOSITION,
    )

    result = evaluate_prompt_witnesses([first, second])
    assert result["status"] == "UNVERIFIED"
    assert "active_slot_identity_not_unique" in result["failures"]
    assert "active_task_identity_not_unique" in result["failures"]


def test_multiple_active_slots_in_one_response_fail_closed() -> None:
    prompt = f"{SOURCE_ID}: {PROPOSITION}"
    observation = normalize_slots_response(
        {
            "slots": [
                {"id": 0, "id_task": 9, "is_processing": True, "prompt": prompt},
                {"id": 1, "id_task": 10, "is_processing": True, "prompt": prompt},
            ]
        },
        source_id=SOURCE_ID,
        proposition_text=PROPOSITION,
    )

    assert observation["status"] == "UNVERIFIED"
    assert "multiple_active_slots" in observation["failures"]
    assert evaluate_prompt_witnesses([observation])["status"] == "UNVERIFIED"


def test_idle_slot_may_omit_prompt_but_cannot_supply_active_witness() -> None:
    idle = normalize_slots_response(
        {"slots": [{"id": 0, "id_task": -1, "is_processing": False}]},
        source_id=SOURCE_ID,
        proposition_text=PROPOSITION,
    )

    assert idle["status"] == "PASS"
    assert evaluate_prompt_witnesses([idle])["status"] == "UNVERIFIED"


def test_malformed_payload_fails_closed() -> None:
    result = normalize_slots_response(
        {"unexpected": []}, source_id=SOURCE_ID, proposition_text=PROPOSITION
    )

    assert result["status"] == "UNVERIFIED"
    assert result["failures"] == ["slots_response_not_list"]
    assert result["trace_instrumentation_version"] == TRACE_INSTRUMENTATION_VERSION


def test_invalid_prompt_target_fails_closed() -> None:
    result = normalize_slots_response(
        {
            "slots": [
                {"id": 0, "id_task": 9, "is_processing": True, "prompt": ""}
            ]
        },
        source_id="",
        proposition_text="",
    )

    assert result["status"] == "UNVERIFIED"
    assert result["failures"] == ["prompt_witness_target_invalid"]


def test_active_slot_without_task_identity_fails_closed() -> None:
    result = normalize_slots_response(
        {
            "slots": [
                {
                    "id": 0,
                    "is_processing": True,
                    "prompt": f"{SOURCE_ID}: {PROPOSITION}",
                }
            ]
        },
        source_id=SOURCE_ID,
        proposition_text=PROPOSITION,
    )

    assert result["status"] == "UNVERIFIED"
    assert result["failures"] == ["slot_identity_invalid:0"]


def test_malformed_observation_fails_closed() -> None:
    result = evaluate_prompt_witnesses([None])  # type: ignore[list-item]

    assert result["status"] == "UNVERIFIED"
    assert "observation_not_object" in result["failures"]


def test_non_boolean_processing_state_fails_closed() -> None:
    result = normalize_slots_response(
        {
            "slots": [
                {
                    "id": 0,
                    "id_task": 9,
                    "is_processing": "true",
                    "prompt": f"{SOURCE_ID}: {PROPOSITION}",
                }
            ]
        },
        source_id=SOURCE_ID,
        proposition_text=PROPOSITION,
    )

    assert result["status"] == "UNVERIFIED"
    assert result["failures"] == ["slot_processing_state_invalid:0"]


def test_boolean_slot_id_is_not_accepted_as_an_integer() -> None:
    result = normalize_slots_response(
        {
            "slots": [
                {
                    "id": True,
                    "id_task": 9,
                    "is_processing": True,
                    "prompt": f"{SOURCE_ID}: {PROPOSITION}",
                }
            ]
        },
        source_id=SOURCE_ID,
        proposition_text=PROPOSITION,
    )

    assert result["status"] == "UNVERIFIED"
    assert result["failures"] == ["slot_identity_invalid:0"]
