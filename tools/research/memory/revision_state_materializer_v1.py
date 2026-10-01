"""Deterministic, read-only revision materialization for MEM-3B1 research.

This module consumes already-grounded temporal propositions. It does not
extract facts, resolve entity identity, or write to the frozen M10 store.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Literal


Cardinality = Literal[
    "SINGLE_VALUE_AT_A_TIME",
    "MULTI_VALUE_CONCURRENT",
    "EVENT_OR_NOT_STATE",
    "UNKNOWN",
]
Action = Literal["ASSERT", "DELETE"]
TemporalBasis = Literal["CURRENT_SNAPSHOT", "EXPLICIT_AS_OF", "UNSPECIFIED"]
Status = Literal["ACTIVE", "SUPERSEDED", "DELETED", "CONFLICT", "EVENT", "UNRESOLVED"]

_DELETE_PATTERNS = (
    re.compile(r"\bi no longer (?:use|have|keep)\b", re.IGNORECASE),
    re.compile(r"\bi stopped using\b", re.IGNORECASE),
    re.compile(r"\bi removed\b", re.IGNORECASE),
    re.compile(r"\bi deleted\b", re.IGNORECASE),
)


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("observed_at must be timezone-aware")
    return value.astimezone(timezone.utc)


def _time_text(value: datetime | None) -> str | None:
    return value.isoformat().replace("+00:00", "Z") if value is not None else None


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TemporalProposition:
    memory_id: str
    scope_id: str
    owner_id: str | None
    object_id: str | None
    attribute_id: str | None
    value_text: str | None
    observed_at: datetime
    source_text: str
    source_session_id: str
    source_turn_ids: tuple[str, ...]
    provenance_hash: str
    cardinality: Cardinality
    temporal_basis: TemporalBasis = "UNSPECIFIED"
    explicit_valid_at: datetime | None = None
    action: Action = "ASSERT"
    delete_evidence_span: str | None = None
    deleted_value_text: str | None = None

    def __post_init__(self) -> None:
        for name in ("memory_id", "scope_id", "source_text", "source_session_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if not re.fullmatch(r"[0-9a-f]{64}", self.provenance_hash):
            raise ValueError("provenance_hash must be a lowercase SHA-256 hex digest")
        if _sha256_text(self.source_text) != self.provenance_hash:
            raise ValueError("provenance_hash does not match source_text")
        object.__setattr__(self, "observed_at", _aware_utc(self.observed_at))
        if self.temporal_basis not in {"CURRENT_SNAPSHOT", "EXPLICIT_AS_OF", "UNSPECIFIED"}:
            raise ValueError("unsupported temporal_basis")
        if self.explicit_valid_at is not None:
            object.__setattr__(self, "explicit_valid_at", _aware_utc(self.explicit_valid_at))
        if self.temporal_basis == "EXPLICIT_AS_OF" and self.explicit_valid_at is None:
            raise ValueError("EXPLICIT_AS_OF requires an evidence-bound explicit_valid_at")
        if self.temporal_basis != "EXPLICIT_AS_OF" and self.explicit_valid_at is not None:
            raise ValueError("explicit_valid_at is only valid for EXPLICIT_AS_OF")
        object.__setattr__(self, "source_turn_ids", tuple(self.source_turn_ids))
        if self.cardinality not in {
            "SINGLE_VALUE_AT_A_TIME",
            "MULTI_VALUE_CONCURRENT",
            "EVENT_OR_NOT_STATE",
            "UNKNOWN",
        }:
            raise ValueError("unsupported cardinality")
        if self.action not in {"ASSERT", "DELETE"}:
            raise ValueError("unsupported action")
        if self.action == "ASSERT":
            if not isinstance(self.value_text, str) or not self.value_text:
                raise ValueError("ASSERT requires a non-empty value_text")
            first = self.source_text.find(self.value_text)
            if first < 0 or self.source_text.find(self.value_text, first + 1) >= 0:
                raise ValueError("ASSERT value must be a unique exact source substring")
            if self.delete_evidence_span is not None:
                raise ValueError("ASSERT cannot carry delete evidence")
        else:
            if self.value_text is not None:
                raise ValueError("DELETE cannot carry a successor value")
            if self.cardinality not in {"SINGLE_VALUE_AT_A_TIME", "MULTI_VALUE_CONCURRENT"}:
                raise ValueError("DELETE requires a known mutable cardinality")
            witness = self.delete_evidence_span
            if not isinstance(witness, str) or not witness:
                raise ValueError("DELETE requires an explicit evidence span")
            first = self.source_text.find(witness)
            if first < 0 or self.source_text.find(witness, first + 1) >= 0:
                raise ValueError("delete evidence must be a unique exact source substring")
            if not any(pattern.search(witness) for pattern in _DELETE_PATTERNS):
                raise ValueError("delete evidence lacks a supported first-person cue")
            target = self.deleted_value_text
            if not isinstance(target, str) or not target:
                raise ValueError("DELETE requires a source-grounded deleted_value_text")
            target_start = self.source_text.casefold().find(target.casefold())
            if target_start < 0 or self.source_text.casefold().find(
                target.casefold(), target_start + len(target)
            ) >= 0:
                raise ValueError("deleted_value_text must be a unique exact source substring")
            if target.casefold() not in witness.casefold():
                raise ValueError("delete evidence does not bind the deleted value")
        if self.action == "ASSERT" and self.deleted_value_text is not None:
            raise ValueError("ASSERT cannot carry deleted_value_text")
        if not self.source_turn_ids:
            raise ValueError("at least one source_turn_id is required")

    @property
    def state_at(self) -> datetime:
        return self.explicit_valid_at or self.observed_at


@dataclass(frozen=True)
class RevisionAdmission:
    """Pairwise transition already accepted by the frozen reducer/harness."""

    previous_memory_id: str
    current_memory_id: str
    effective_at: datetime
    evidence_sha256: str

    def __post_init__(self) -> None:
        if not self.previous_memory_id or not self.current_memory_id:
            raise ValueError("revision admission requires both memory IDs")
        if self.previous_memory_id == self.current_memory_id:
            raise ValueError("revision admission cannot self-link")
        object.__setattr__(self, "effective_at", _aware_utc(self.effective_at))
        if not re.fullmatch(r"[0-9a-f]{64}", self.evidence_sha256):
            raise ValueError("evidence_sha256 must be a lowercase SHA-256 hex digest")


@dataclass(frozen=True)
class MaterializedRecord:
    record_id: str
    slot_key: tuple[str, str, str, str] | None
    value_text: str | None
    status: Status
    valid_from: str
    valid_until: str | None
    source_memory_ids: tuple[str, ...]
    source_session_ids: tuple[str, ...]
    source_turn_ids: tuple[str, ...]
    provenance_hashes: tuple[str, ...]
    supersedes_id: str | None = None
    supersedes_ids: tuple[str, ...] = ()
    reason: str | None = None


@dataclass(frozen=True)
class MaterializationResult:
    records: tuple[MaterializedRecord, ...]
    input_sha256: str
    output_sha256: str

    def current(self, slot_key: tuple[str, str, str, str]) -> tuple[MaterializedRecord, ...]:
        return tuple(
            row for row in self.records
            if row.slot_key == slot_key and row.status in {"ACTIVE", "CONFLICT"}
        )

    def as_of(
        self, slot_key: tuple[str, str, str, str], at: datetime
    ) -> tuple[MaterializedRecord, ...]:
        instant = _aware_utc(at)
        result = []
        for row in self.records:
            if row.slot_key != slot_key or row.status in {"EVENT", "UNRESOLVED", "DELETED"}:
                continue
            start = datetime.fromisoformat(row.valid_from.replace("Z", "+00:00"))
            end = datetime.fromisoformat(row.valid_until.replace("Z", "+00:00")) if row.valid_until else None
            if start <= instant and (end is None or instant < end):
                result.append(row)
        return tuple(result)

    def change_chain(self, slot_key: tuple[str, str, str, str]) -> tuple[MaterializedRecord, ...]:
        return tuple(
            row for row in self.records
            if row.slot_key == slot_key and row.status not in {"EVENT", "UNRESOLVED"}
        )


def _slot_key(row: TemporalProposition) -> tuple[str, str, str, str] | None:
    if not row.owner_id or not row.object_id or not row.attribute_id:
        return None
    return (row.scope_id, row.owner_id, row.object_id, row.attribute_id)


def _record_id(slot: tuple[str, str, str, str] | None, first_id: str) -> str:
    payload = json.dumps([slot, first_id], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _record(
    *,
    slot: tuple[str, str, str, str] | None,
    members: list[TemporalProposition],
    status: Status,
    valid_until: datetime | None,
    supersedes_id: str | None = None,
    supersedes_ids: tuple[str, ...] = (),
    valid_from: datetime | None = None,
    reason: str | None = None,
) -> MaterializedRecord:
    ordered = sorted(members, key=lambda row: (row.state_at, row.observed_at, row.memory_id))
    first = ordered[0]
    return MaterializedRecord(
        record_id=_record_id(slot, first.memory_id),
        slot_key=slot,
        value_text=first.value_text if first.value_text is not None else first.deleted_value_text,
        status=status,
        valid_from=_time_text(valid_from or first.state_at) or "",
        valid_until=_time_text(valid_until),
        source_memory_ids=tuple(sorted(row.memory_id for row in ordered)),
        source_session_ids=tuple(sorted({row.source_session_id for row in ordered})),
        source_turn_ids=tuple(sorted({turn for row in ordered for turn in row.source_turn_ids})),
        provenance_hashes=tuple(sorted({row.provenance_hash for row in ordered})),
        supersedes_id=(supersedes_id or (supersedes_ids[0] if len(supersedes_ids) == 1 else None)),
        supersedes_ids=tuple(sorted(set(supersedes_ids or ((supersedes_id,) if supersedes_id else ())))),
        reason=reason,
    )


def _materialize_single(
    slot: tuple[str, str, str, str],
    rows: list[TemporalProposition],
    admissions_by_current: dict[str, list[RevisionAdmission]],
) -> list[MaterializedRecord]:
    timeline: dict[datetime, list[TemporalProposition]] = defaultdict(list)
    for row in rows:
        timeline[row.state_at].append(row)
    output: list[MaterializedRecord] = []
    active: list[MaterializedRecord] = []
    last_transition_id: str | None = None
    for instant, group in sorted(timeline.items()):
        deletions = sorted((row for row in group if row.action == "DELETE"), key=lambda row: row.memory_id)
        assertions = sorted((row for row in group if row.action == "ASSERT"), key=lambda row: row.memory_id)
        distinct_values = {row.value_text for row in assertions}
        ambiguous = bool(deletions and assertions) or len(distinct_values) > 1
        if deletions and not assertions and (
            len(active) != 1
            or active[0].status != "ACTIVE"
            or deletions[0].deleted_value_text != active[0].value_text
        ):
            output.append(_record(
                slot=slot, members=deletions, status="UNRESOLVED", valid_until=None,
                reason=(
                    "delete_target_does_not_match_unambiguous_current_value"
                    if active else "delete_has_no_unambiguous_current_value"
                ),
            ))
            continue
        if ambiguous:
            for current in active:
                conflicted = MaterializedRecord(
                    **{**asdict(current), "status": "CONFLICT", "valid_until": None,
                       "reason": "different_singleton_values_not_temporally_resolved"}
                )
                output[output.index(current)] = conflicted
            active = [row for row in output if row.slot_key == slot and row.status == "CONFLICT"]
            for value in sorted(distinct_values, key=lambda value: value or ""):
                matching = [row for row in assertions if row.value_text == value]
                record = _record(
                    slot=slot, members=matching, status="CONFLICT", valid_until=None,
                    reason="different_values_or_delete_at_same_instant",
                )
                output.append(record)
                active.append(record)
            for deletion in deletions:
                tombstone = _record(
                    slot=slot, members=[deletion], status="DELETED", valid_until=None,
                    reason="explicit_delete_conflicts_with_same-time_assertion",
                )
                output.append(tombstone)
                last_transition_id = tombstone.record_id
            continue

        if deletions:
            predecessor_id = active[0].record_id if len(active) == 1 else last_transition_id
            for current in active:
                output[output.index(current)] = MaterializedRecord(
                    **{**asdict(current), "status": "SUPERSEDED", "valid_until": _time_text(instant)}
                )
            active = []
            tombstone = _record(
                slot=slot, members=deletions, status="DELETED", valid_until=None,
                supersedes_id=predecessor_id, reason="explicit_delete_tombstone",
            )
            output.append(tombstone)
            last_transition_id = tombstone.record_id
            continue
        if not assertions:
            continue

        value = next(iter(distinct_values))
        matching_active = [row for row in active if row.value_text == value]
        if matching_active:
            current = matching_active[0]
            prior_members = [row for row in rows if row.memory_id in current.source_memory_ids]
            replacement = _record(
                slot=slot, members=[*prior_members, *assertions], status=current.status,
                valid_until=current.valid_until,
                supersedes_id=current.supersedes_id,
                supersedes_ids=current.supersedes_ids,
                valid_from=datetime.fromisoformat(current.valid_from.replace("Z", "+00:00")),
                reason=current.reason,
            )
            output[output.index(current)] = replacement
            active[active.index(current)] = replacement
            continue

        target_admissions = [
            edge
            for assertion in assertions
            for edge in admissions_by_current.get(assertion.memory_id, [])
        ]
        snapshot_transition = bool(active) and all(
            assertion.temporal_basis == "CURRENT_SNAPSHOT" for assertion in assertions
        )
        active_to_supersede: list[MaterializedRecord] = []
        if snapshot_transition and all(
            datetime.fromisoformat(current.valid_from.replace("Z", "+00:00")) < instant
            for current in active
        ):
            active_to_supersede = list(active)
        else:
            for current in active:
                supports = set(current.source_memory_ids)
                matching_edges = [edge for edge in target_admissions if edge.previous_memory_id in supports]
                if not matching_edges:
                    active_to_supersede = []
                    break
                active_to_supersede.append(current)
        if active and len(active_to_supersede) != len(active):
            for current in active:
                conflicted = MaterializedRecord(
                    **{**asdict(current), "status": "CONFLICT", "valid_until": None,
                       "reason": "different_singleton_values_without_admitted_revision"}
                )
                output[output.index(current)] = conflicted
            new_record = _record(
                slot=slot, members=assertions, status="CONFLICT", valid_until=None,
                reason="different_singleton_values_without_admitted_revision",
            )
            output.append(new_record)
            active = [row for row in output if row.slot_key == slot and row.status == "CONFLICT"]
            continue

        effective_times = {
            edge.effective_at
            for edge in target_admissions
            if any(edge.previous_memory_id in current.source_memory_ids for current in active_to_supersede)
        }
        if snapshot_transition and active_to_supersede:
            effective_times = {instant}
        if active_to_supersede and len(effective_times) != 1:
            new_record = _record(
                slot=slot, members=assertions, status="UNRESOLVED", valid_until=None,
                reason="revision_admissions_disagree_on_effective_time",
            )
            output.append(new_record)
            continue
        effective_at = next(iter(effective_times)) if effective_times else instant
        for current in active_to_supersede:
            output[output.index(current)] = MaterializedRecord(
                **{**asdict(current), "status": "SUPERSEDED", "valid_until": _time_text(effective_at)}
            )
        predecessor_ids = tuple(current.record_id for current in active_to_supersede)
        predecessor_id = (
            predecessor_ids[0] if len(predecessor_ids) == 1
            else last_transition_id if not active_to_supersede
            else None
        )

        record = _record(
            slot=slot, members=assertions, status="ACTIVE", valid_until=None,
            supersedes_id=predecessor_id,
            supersedes_ids=predecessor_ids if len(predecessor_ids) > 1 else (),
            valid_from=effective_at,
        )
        output.append(record)
        active = [record]
        last_transition_id = record.record_id
    return output


def _materialize_multi(
    slot: tuple[str, str, str, str], rows: list[TemporalProposition]
) -> list[MaterializedRecord]:
    output: list[MaterializedRecord] = []
    timeline: dict[datetime, list[TemporalProposition]] = defaultdict(list)
    for row in rows:
        timeline[row.state_at].append(row)
    active: dict[str, MaterializedRecord] = {}
    last_transition_id: str | None = None
    for instant, group in sorted(timeline.items()):
        deletions = sorted((row for row in group if row.action == "DELETE"), key=lambda row: row.memory_id)
        assertions = sorted((row for row in group if row.action == "ASSERT"), key=lambda row: row.memory_id)
        new_by_value: dict[str, list[TemporalProposition]] = defaultdict(list)
        for row in assertions:
            assert row.value_text is not None
            new_by_value[row.value_text.casefold().strip()].append(row)
        if deletions:
            for deletion in deletions:
                target_key = (deletion.deleted_value_text or "").casefold().strip()
                current = active.get(target_key)
                if current is None or assertions:
                    output.append(_record(
                        slot=slot, members=[deletion], status="UNRESOLVED", valid_until=None,
                        reason="delete_target_missing_or_conflicted_at_same_time",
                    ))
                    continue
                output[output.index(current)] = MaterializedRecord(
                    **{**asdict(current), "status": "SUPERSEDED", "valid_until": _time_text(instant)}
                )
                del active[target_key]
                tombstone = _record(
                    slot=slot, members=[deletion], status="DELETED", valid_until=None,
                    supersedes_id=current.record_id, reason="explicit_value_tombstone",
                )
                output.append(tombstone)
                last_transition_id = tombstone.record_id
            continue
        for key, members in new_by_value.items():
            current = active.get(key)
            if current is not None and current.status == "ACTIVE":
                prior_members = [row for row in rows if row.memory_id in current.source_memory_ids]
                replacement = _record(
                    slot=slot, members=[*prior_members, *members], status="ACTIVE",
                    valid_until=None, supersedes_id=current.supersedes_id,
                    reason="concurrent_value",
                )
                output[output.index(current)] = replacement
                active[key] = replacement
                continue
            record = _record(
                slot=slot, members=members, status="ACTIVE", valid_until=None,
                supersedes_id=last_transition_id, reason="concurrent_value",
            )
            active[key] = record
            output.append(record)
    return output


def materialize(
    observations: list[TemporalProposition] | tuple[TemporalProposition, ...],
    revision_admissions: list[RevisionAdmission] | tuple[RevisionAdmission, ...] = (),
) -> MaterializationResult:
    """Build a stable temporal view from frozen, source-grounded observations."""
    by_id: dict[str, TemporalProposition] = {}
    for row in observations:
        previous = by_id.get(row.memory_id)
        if previous is not None and previous != row:
            raise ValueError(f"memory_id collision with different content: {row.memory_id}")
        by_id[row.memory_id] = row
    admission_keys: set[tuple[str, str]] = set()
    admissions_by_current: dict[str, list[RevisionAdmission]] = defaultdict(list)
    for admission in revision_admissions:
        pair = (admission.previous_memory_id, admission.current_memory_id)
        if pair in admission_keys:
            raise ValueError(f"duplicate revision admission: {pair}")
        admission_keys.add(pair)
        previous = by_id.get(admission.previous_memory_id)
        current = by_id.get(admission.current_memory_id)
        if previous is None or current is None:
            raise ValueError(f"revision admission references unknown memory ID: {pair}")
        if (
            previous.action != "ASSERT"
            or current.action != "ASSERT"
            or previous.cardinality != "SINGLE_VALUE_AT_A_TIME"
            or current.cardinality != "SINGLE_VALUE_AT_A_TIME"
            or _slot_key(previous) is None
            or _slot_key(previous) != _slot_key(current)
            or previous.value_text == current.value_text
        ):
            raise ValueError(f"revision admission does not connect different values in one singleton slot: {pair}")
        if not (previous.state_at < admission.effective_at <= current.state_at):
            raise ValueError(f"revision effective time is outside observation interval: {pair}")
        admissions_by_current[current.memory_id].append(admission)
    rows = sorted(by_id.values(), key=lambda row: (row.state_at, row.observed_at, row.memory_id))
    input_payload = [
        {
            **asdict(row),
            "observed_at": _time_text(row.observed_at),
            "explicit_valid_at": _time_text(row.explicit_valid_at),
        }
        for row in rows
    ]
    admissions_payload = [
        {
            "previous_memory_id": row.previous_memory_id,
            "current_memory_id": row.current_memory_id,
            "effective_at": _time_text(row.effective_at),
            "evidence_sha256": row.evidence_sha256,
        }
        for row in sorted(revision_admissions, key=lambda edge: (edge.effective_at, edge.previous_memory_id, edge.current_memory_id))
    ]
    grouped: dict[tuple[str, str, str, str], list[TemporalProposition]] = defaultdict(list)
    records: list[MaterializedRecord] = []
    for row in rows:
        slot = _slot_key(row)
        if row.cardinality == "EVENT_OR_NOT_STATE":
            records.append(_record(
                slot=slot, members=[row], status="EVENT", valid_until=None,
                reason="event_or_non_state_excluded_from_revision_state",
            ))
        elif row.cardinality == "UNKNOWN" or slot is None:
            records.append(_record(
                slot=None, members=[row], status="UNRESOLVED", valid_until=None,
                reason="unknown_cardinality_or_incomplete_slot_identity",
            ))
        else:
            grouped[slot].append(row)
    for slot, members in sorted(grouped.items()):
        cardinalities = {row.cardinality for row in members}
        if len(cardinalities) != 1:
            records.extend(
                _record(
                    slot=slot, members=[row], status="UNRESOLVED", valid_until=None,
                    reason="cardinality_disagreement_within_slot",
                )
                for row in members
            )
        elif next(iter(cardinalities)) == "SINGLE_VALUE_AT_A_TIME":
            records.extend(_materialize_single(slot, members, admissions_by_current))
        else:
            records.extend(_materialize_multi(slot, members))
    records.sort(key=lambda row: (row.valid_from, row.slot_key or ("", "", "", ""), row.record_id))
    normalized = [asdict(row) for row in records]
    canonical = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    input_canonical = json.dumps(
        {"observations": input_payload, "revision_admissions": admissions_payload},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str,
    )
    return MaterializationResult(
        records=tuple(records),
        input_sha256=_sha256_text(input_canonical),
        output_sha256=_sha256_text(canonical),
    )


__all__ = [
    "MaterializationResult",
    "MaterializedRecord",
    "RevisionAdmission",
    "TemporalProposition",
    "materialize",
]
