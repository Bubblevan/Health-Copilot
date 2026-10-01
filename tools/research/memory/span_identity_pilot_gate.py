"""Strict diagnostic gate for the span-grounded identity qualification pilot."""

from __future__ import annotations

import re
from typing import Any

from tools.research.memory.span_grounded_identity import GENERIC_ATTRIBUTE_TOKENS

EXPLICIT_NON_SINGLETON_KINDS = frozenset({"MULTI_VALUE_STATE", "EVENT", "NON_STATE"})


def _same_key(rows: list[dict[str, Any]]) -> bool:
    raw_keys = [row.get("slot_key") for row in rows]
    if not raw_keys or any(not isinstance(key, list) or not key for key in raw_keys):
        return False
    keys = [tuple(key) for key in raw_keys]
    return len(set(keys)) == 1


def _generic_only_attribute_unresolved(row: dict[str, Any]) -> bool:
    key = row.get("attribute_key")
    if not isinstance(key, str):
        return False
    tokens = set(re.findall(r"[a-z0-9]+", key.casefold()))
    return (
        row.get("identity_status") == "UNRESOLVED"
        and row.get("slot_candidate") is False
        and bool(tokens)
        and tokens <= GENERIC_ATTRIBUTE_TOKENS
        and row.get("validation_reasons") == ["attribute_key:UNSUPPORTED_KEY_TOKENS:"]
    )


def _case_result(case: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected = case.get("expected")
    case_id = case.get("case_id")
    grounded = bool(rows) and all(row.get("identity_status") == "GROUNDED" for row in rows)
    candidates = all(row.get("slot_candidate") is True for row in rows)
    keys = [row.get("slot_key") for row in rows]
    passed = False
    reason = "unsupported_expectation"

    if expected == "SAME_SLOT":
        passed = grounded and candidates and all(key is not None for key in keys) and _same_key(rows)
        reason = "grounded_same_candidate_slot" if passed else "requires_grounded_equal_candidate_slots"
    elif expected == "NOT_SAME_SLOT":
        if not grounded:
            reason = "negative_control_contains_unresolved_identity"
        elif candidates:
            keys_valid = all(isinstance(key, list) and bool(key) for key in keys)
            passed = keys_valid and not _same_key(rows)
            reason = "distinct_grounded_candidate_slots" if passed else "false_same_slot_candidate"
        else:
            explicit_vetoes = all(
                row.get("slot_candidate") is False
                and row.get("property_kind") in EXPLICIT_NON_SINGLETON_KINDS
                for row in rows
            )
            passed = explicit_vetoes
            reason = "grounded_explicit_non_singleton_veto" if passed else "missing_grounded_distinction_or_veto"
    elif expected == "NO_SLOT" and case_id == "completed_purchase_event":
        passed = grounded and all(
            row.get("slot_candidate") is False and row.get("property_kind") == "EVENT"
            for row in rows
        )
        reason = "grounded_events_vetoed" if passed else "completed_event_must_be_grounded_and_vetoed"
    elif expected == "NO_SLOT" and case_id == "macrame_interest_not_singleton_revision":
        if not grounded:
            reason = "negative_control_contains_unresolved_identity"
        else:
            candidates_by_key: dict[tuple[Any, ...], int] = {}
            malformed_candidate = False
            for row in rows:
                if row.get("slot_candidate") is True:
                    raw_key = row.get("slot_key")
                    if not isinstance(raw_key, list):
                        malformed_candidate = True
                        break
                    key = tuple(raw_key)
                    candidates_by_key[key] = candidates_by_key.get(key, 0) + 1
                elif row.get("property_kind") not in EXPLICIT_NON_SINGLETON_KINDS:
                    malformed_candidate = True
                    break
            passed = not malformed_candidate and max(candidates_by_key.values(), default=0) < 2
            reason = "no_repeated_candidate_slot" if passed else "repeated_or_unclassified_candidate_slot"
    elif expected == "BOTH_UNRESOLVED":
        passed = bool(rows) and all(_generic_only_attribute_unresolved(row) for row in rows)
        reason = "generic_attribute_only_unresolved" if passed else "unresolved_not_limited_to_generic_attribute"

    return {
        "case_id": case_id,
        "expected": expected,
        "memory_ids": [row.get("memory_id") for row in rows],
        "identity_statuses": [row.get("identity_status") for row in rows],
        "property_kinds": [row.get("property_kind") for row in rows],
        "slot_keys": keys,
        "validation_reasons": [row.get("validation_reasons", []) for row in rows],
        "passed": passed,
        "gate_reason": reason,
    }


def evaluate_cases(
    cases: list[dict[str, Any]],
    proposals: list[dict[str, Any]],
    expected_memory_ids: set[str],
) -> list[dict[str, Any]]:
    proposal_by_id = {str(row.get("memory_id")): row for row in proposals}
    if len(proposal_by_id) != len(proposals) or set(proposal_by_id) != expected_memory_ids:
        raise ValueError("proposal_id_coverage_mismatch")
    case_ids = [case.get("case_id") for case in cases]
    if len(case_ids) != len(set(case_ids)) or any(not isinstance(value, str) for value in case_ids):
        raise ValueError("case_ids_not_unique")
    covered_ids: set[str] = set()
    results = []
    for case in cases:
        memory_ids = case.get("memory_ids")
        if not isinstance(memory_ids, list) or not memory_ids or any(
            not isinstance(memory_id, str) or memory_id not in proposal_by_id
            for memory_id in memory_ids
        ):
            raise ValueError(f"invalid_case_memory_ids:{case.get('case_id')}")
        covered_ids.update(memory_ids)
        results.append(_case_result(case, [proposal_by_id[memory_id] for memory_id in memory_ids]))
    if covered_ids != expected_memory_ids:
        raise ValueError("case_memory_id_coverage_mismatch")
    return results
