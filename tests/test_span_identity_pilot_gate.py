from __future__ import annotations

import pytest

from tools.research.memory import span_identity_pilot_gate as gate


def _row(
    memory_id: str,
    *,
    identity_status: str = "GROUNDED",
    slot_candidate: bool = True,
    slot_key: list[str | None] | None = None,
    property_kind: str = "SINGLE_VALUE_STATE",
    attribute_key: str = "follower_count",
    validation_reasons: list[str] | None = None,
) -> dict[str, object]:
    return {
        "memory_id": memory_id,
        "identity_status": identity_status,
        "slot_candidate": slot_candidate,
        "slot_key": slot_key or ["scope", "user", "instagram", "follower_count"],
        "property_kind": property_kind,
        "attribute_key": attribute_key,
        "validation_reasons": validation_reasons or [],
    }


def test_negative_slot_control_rejects_unresolved_abstention() -> None:
    case = {"case_id": "wallet_color_vs_material", "expected": "NOT_SAME_SLOT"}
    rows = [
        _row("a", identity_status="UNRESOLVED", slot_candidate=False, slot_key=None),
        _row("b", identity_status="UNRESOLVED", slot_candidate=False, slot_key=None),
    ]
    result = gate._case_result(case, rows)
    assert result["passed"] is False
    assert result["gate_reason"] == "negative_control_contains_unresolved_identity"


def test_negative_slot_control_accepts_grounded_distinct_slots() -> None:
    case = {"case_id": "wallet_color_vs_material", "expected": "NOT_SAME_SLOT"}
    rows = [
        _row("a", slot_key=["scope", "user", "wallet", "color"]),
        _row("b", slot_key=["scope", "user", "wallet", "material"]),
    ]
    result = gate._case_result(case, rows)
    assert result["passed"] is True
    assert result["gate_reason"] == "distinct_grounded_candidate_slots"


def test_negative_slot_control_accepts_only_explicit_grounded_vetoes() -> None:
    case = {"case_id": "trip_destinations", "expected": "NOT_SAME_SLOT"}
    rows = [
        _row("a", slot_candidate=False, property_kind="MULTI_VALUE_STATE"),
        _row("b", slot_candidate=False, property_kind="MULTI_VALUE_STATE"),
    ]
    assert gate._case_result(case, rows)["passed"] is True
    rows[1]["property_kind"] = "UNKNOWN"
    assert gate._case_result(case, rows)["passed"] is False


def test_event_control_rejects_unresolved_records() -> None:
    case = {"case_id": "completed_purchase_event", "expected": "NO_SLOT"}
    rows = [
        _row("a", identity_status="UNRESOLVED", slot_candidate=False, slot_key=None, property_kind="EVENT"),
        _row("b", identity_status="UNRESOLVED", slot_candidate=False, slot_key=None, property_kind="EVENT"),
    ]
    assert gate._case_result(case, rows)["passed"] is False


def test_event_control_requires_grounded_event_veto() -> None:
    case = {"case_id": "completed_purchase_event", "expected": "NO_SLOT"}
    rows = [
        _row("a", slot_candidate=False, property_kind="EVENT"),
        _row("b", slot_candidate=False, property_kind="EVENT"),
    ]
    assert gate._case_result(case, rows)["passed"] is True
    rows[0]["property_kind"] = "UNKNOWN"
    assert gate._case_result(case, rows)["passed"] is False


def test_macrame_control_requires_grounded_rows_and_no_repeated_candidate_slot() -> None:
    case = {"case_id": "macrame_interest_not_singleton_revision", "expected": "NO_SLOT"}
    rows = [
        _row("a", slot_key=["scope", "user", "macrame", "interest"]),
        _row("b", slot_key=["scope", "user", "macrame", "skill"]),
        _row("c", slot_candidate=False, property_kind="MULTI_VALUE_STATE"),
    ]
    assert gate._case_result(case, rows)["passed"] is True
    rows[1]["slot_key"] = ["scope", "user", "macrame", "interest"]
    assert gate._case_result(case, rows)["passed"] is False


def test_generic_unresolved_control_only_passes_for_generic_attribute_failure() -> None:
    case = {"case_id": "generic_numeric_attribute", "expected": "BOTH_UNRESOLVED"}
    rows = [
        _row(
            "a",
            identity_status="UNRESOLVED",
            slot_candidate=False,
            slot_key=None,
            attribute_key="reported_value",
            validation_reasons=["attribute_key:UNSUPPORTED_KEY_TOKENS:"],
        ),
        _row(
            "b",
            identity_status="UNRESOLVED",
            slot_candidate=False,
            slot_key=None,
            attribute_key="value",
            validation_reasons=["attribute_key:UNSUPPORTED_KEY_TOKENS:"],
        ),
    ]
    assert gate._case_result(case, rows)["passed"] is True
    rows[0]["validation_reasons"] = ["value_witness_span:MISSING_REQUIRED_WITNESS"]
    assert gate._case_result(case, rows)["passed"] is False


def test_positive_control_requires_grounded_exact_candidate_tuple() -> None:
    case = {"case_id": "instagram", "expected": "SAME_SLOT"}
    rows = [
        _row("a", slot_key=["scope", "user", "instagram", "followers"]),
        _row("b", slot_key=["scope", "user", "instagram", "followers"]),
    ]
    assert gate._case_result(case, rows)["passed"] is True
    rows[1]["identity_status"] = "UNRESOLVED"
    assert gate._case_result(case, rows)["passed"] is False


def test_case_suite_requires_exact_proposal_and_case_coverage() -> None:
    cases = [
        {"case_id": "same", "expected": "SAME_SLOT", "memory_ids": ["a", "b"]}
    ]
    proposals = [
        _row("a"),
        _row("b"),
        _row("c"),
    ]
    with pytest.raises(ValueError, match="case_memory_id_coverage_mismatch"):
        gate.evaluate_cases(cases, proposals, {"a", "b", "c"})
    with pytest.raises(ValueError, match="proposal_id_coverage_mismatch"):
        gate.evaluate_cases(cases, proposals[:2], {"a", "b", "c"})
