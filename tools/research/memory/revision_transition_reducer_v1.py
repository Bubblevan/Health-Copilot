"""Non-mutating reducer for conservative revision-transition admission experiments.

This module does not resolve entity identity or write MemoryStore state. It only
classifies a pair after those identities have been supplied explicitly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal


Cardinality = Literal[
    "SINGLE_VALUE_AT_A_TIME",
    "MULTI_VALUE_CONCURRENT",
    "EVENT_OR_NOT_STATE",
    "UNKNOWN",
]
Operation = Literal["REPLACE", "DELETE"]


@dataclass(frozen=True)
class StateObservation:
    source_id: str
    source_text: str
    scope_id: str | None
    owner_entity_id: str | None
    object_entity_id: str | None
    object_type_id: str | None
    attribute_id: str | None
    value_text: str | None
    cardinality: Cardinality
    observed_at: datetime | None


@dataclass(frozen=True)
class TransitionWitness:
    operation: Operation
    source_id: str
    evidence_span: str
    event_time_span: str
    event_at: datetime
    previous_value: str
    next_value: str | None


@dataclass(frozen=True)
class TransitionDecision:
    decision: str
    reason: str
    store_mutation: Literal["NONE"] = "NONE"


def _norm(value: str) -> str:
    return " ".join(value.casefold().split())


def _valid_time(value: datetime | None) -> bool:
    return value is not None and value.tzinfo is not None and value.utcoffset() is not None


def _value_pattern(value: str) -> str:
    return rf"(?<!\w){re.escape(_norm(value))}(?!\w)"


def _has_replace_witness(text: str, previous_value: str, current_value: str) -> bool:
    normalized = _norm(text)
    old = _value_pattern(previous_value)
    new = _value_pattern(current_value)
    patterns = (
        rf"on \d{{4}}-\d{{2}}-\d{{2}},? i (?:have )?replaced\s+{old}\s+with\s+{new}[.!?]?",
        rf"on \d{{4}}-\d{{2}}-\d{{2}},? i (?:have )?switched from\s+{old}\s+to\s+{new}[.!?]?",
        rf"on \d{{4}}-\d{{2}}-\d{{2}},? i (?:have )?changed from\s+{old}\s+to\s+{new}[.!?]?",
        rf"on \d{{4}}-\d{{2}}-\d{{2}},? i now "
        rf"(?:prefer|use|choose)\s+{new}\s+instead of\s+{old}[.!?]?",
    )
    return any(re.fullmatch(pattern, normalized) for pattern in patterns)


def _has_delete_witness(text: str, previous_value: str) -> bool:
    normalized = _norm(text)
    old = _value_pattern(previous_value)
    patterns = (
        rf"on \d{{4}}-\d{{2}}-\d{{2}},? i (?:have )?removed\s+(?:the )?{old}[.!?]?",
        rf"on \d{{4}}-\d{{2}}-\d{{2}},? i (?:have )?deleted\s+(?:the )?{old}[.!?]?",
        rf"on \d{{4}}-\d{{2}}-\d{{2}},? i no longer (?:use|have|keep)\s+{old}[.!?]?",
        rf"on \d{{4}}-\d{{2}}-\d{{2}},? i stopped using\s+{old}[.!?]?",
    )
    return any(re.fullmatch(pattern, normalized) for pattern in patterns)


def _slot_decision(
    previous: StateObservation, current: StateObservation
) -> TransitionDecision | None:
    if not previous.scope_id or not current.scope_id:
        return TransitionDecision("UNRESOLVED", "scope_identity_missing")
    if previous.scope_id != current.scope_id:
        return TransitionDecision("DISTINCT_SCOPE", "scope_id_mismatch")
    if not previous.owner_entity_id or not current.owner_entity_id:
        return TransitionDecision("UNRESOLVED", "owner_entity_identity_missing")
    if previous.owner_entity_id != current.owner_entity_id:
        return TransitionDecision("DISTINCT_SLOT", "owner_entity_id_mismatch")
    if not previous.object_entity_id or not current.object_entity_id:
        return TransitionDecision("UNRESOLVED", "object_instance_identity_missing")
    if previous.object_entity_id != current.object_entity_id:
        return TransitionDecision("DISTINCT_SLOT", "object_entity_id_mismatch")
    if (
        previous.object_type_id
        and current.object_type_id
        and previous.object_type_id != current.object_type_id
    ):
        return TransitionDecision("UNRESOLVED", "entity_type_conflicts_with_instance_link")
    if not previous.attribute_id or not current.attribute_id:
        return TransitionDecision("UNRESOLVED", "attribute_identity_missing")
    if previous.attribute_id != current.attribute_id:
        return TransitionDecision("DISTINCT_SLOT", "attribute_id_mismatch")
    return None


def _witness_error(
    previous: StateObservation,
    current: StateObservation,
    witness: TransitionWitness | None,
) -> str | None:
    if witness is None:
        return "explicit_transition_witness_missing"
    if witness.source_id != current.source_id:
        return "witness_source_id_mismatch"
    if not witness.evidence_span:
        return "witness_span_missing"
    source = current.source_text
    first = source.find(witness.evidence_span)
    if first < 0 or source.find(witness.evidence_span, first + 1) >= 0:
        return "witness_span_not_unique_exact_source_substring"
    if witness.evidence_span != source:
        return "transition_witness_must_cover_full_source_statement"
    if not witness.event_time_span or witness.event_time_span not in witness.evidence_span:
        return "event_time_not_in_transition_witness"
    date_offset = source.find(witness.event_time_span)
    if date_offset < 0 or source.find(witness.event_time_span, date_offset + 1) >= 0:
        return "event_time_span_not_unique_exact_source_substring"
    try:
        event_date = datetime.strptime(witness.event_time_span, "%Y-%m-%d").date()
    except ValueError:
        return "event_time_span_not_iso_date"
    if not _valid_time(witness.event_at):
        return "event_time_missing_or_timezone_naive"
    expected_event_at = datetime.combine(event_date, datetime.min.time(), tzinfo=timezone.utc)
    if witness.event_at != expected_event_at:
        return "event_time_not_bound_to_date_precision"
    if not _valid_time(previous.observed_at) or not _valid_time(current.observed_at):
        return "temporal_order_missing_or_timezone_naive"
    if witness.event_at <= previous.observed_at:
        return "event_time_not_after_previous_observation"
    if current.observed_at < witness.event_at:
        return "event_time_after_current_observation"
    return None


def _observation_error(observation: StateObservation) -> str | None:
    if not observation.source_id or not observation.source_text:
        return "source_provenance_missing"
    if observation.value_text is None:
        return None
    if not observation.value_text:
        return "empty_state_value"
    first = observation.source_text.find(observation.value_text)
    if first < 0 or observation.source_text.find(
        observation.value_text, first + 1
    ) >= 0:
        return "value_not_unique_exact_source_substring"
    return None


def reduce_transition(
    previous: StateObservation,
    current: StateObservation,
    witness: TransitionWitness | None = None,
) -> TransitionDecision:
    """Classify a transition candidate without changing persistent state."""
    for observation in (previous, current):
        error = _observation_error(observation)
        if error:
            return TransitionDecision("UNRESOLVED", error)

    distinct = _slot_decision(previous, current)
    if distinct is not None:
        return distinct

    if previous.cardinality != current.cardinality:
        return TransitionDecision("UNRESOLVED", "cardinality_disagreement")
    if previous.cardinality not in {
        "SINGLE_VALUE_AT_A_TIME",
        "MULTI_VALUE_CONCURRENT",
    }:
        return TransitionDecision("UNRESOLVED", "not_a_known_state_slot")

    if witness is not None and witness.operation == "DELETE":
        if current.value_text is not None:
            return TransitionDecision("UNRESOLVED", "delete_candidate_has_current_value")
        if witness.next_value is not None:
            return TransitionDecision("UNRESOLVED", "delete_witness_has_successor_value")
        if witness.previous_value != previous.value_text or previous.value_text is None:
            return TransitionDecision("UNRESOLVED", "delete_witness_value_mismatch")
        error = _witness_error(previous, current, witness)
        if error:
            return TransitionDecision("UNRESOLVED", error)
        if not _has_delete_witness(witness.evidence_span, witness.previous_value):
            return TransitionDecision(
                "UNRESOLVED", "delete_witness_lacks_supported_explicit_pattern"
            )
        return TransitionDecision("TOMBSTONE_CANDIDATE", "explicit_grounded_delete_witness")

    if current.value_text is None or previous.value_text is None:
        return TransitionDecision("UNRESOLVED", "state_value_missing")
    if _norm(previous.value_text) == _norm(current.value_text):
        return TransitionDecision("DUPLICATE_SUPPORT_NOOP", "same_slot_same_normalized_value")
    if previous.cardinality == "MULTI_VALUE_CONCURRENT":
        return TransitionDecision(
            "COEXISTING_VALUES", "concurrent_cardinality_forbids_supersession"
        )
    if previous.cardinality != "SINGLE_VALUE_AT_A_TIME":
        return TransitionDecision("UNRESOLVED", "unsupported_cardinality")

    if witness is None or witness.operation != "REPLACE":
        return TransitionDecision("UNRESOLVED", "different_values_without_replace_witness")
    if (
        witness.previous_value != previous.value_text
        or witness.next_value != current.value_text
    ):
        return TransitionDecision("UNRESOLVED", "replace_witness_value_mismatch")
    error = _witness_error(previous, current, witness)
    if error:
        return TransitionDecision("UNRESOLVED", error)
    if not _has_replace_witness(
        witness.evidence_span, witness.previous_value, witness.next_value
    ):
        return TransitionDecision("UNRESOLVED", "replace_witness_lacks_supported_explicit_pattern")
    return TransitionDecision("REVISION_CANDIDATE", "explicit_grounded_replace_witness")
