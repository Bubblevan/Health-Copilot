"""Conservative admission for explicit change mentions without inventing history."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal


SlotKey = tuple[str, str, str, str]
AdmissionKind = Literal["ANCHORED_REVISION", "UNANCHORED_CHANGE_MENTION"]
_CONTRAST_CUE = re.compile(r"\b(?:instead of|rather than)\b", re.IGNORECASE)
_FIRST_PERSON = re.compile(r"\b(?:I|I've|I'm|I'd|I have|I am)\b", re.IGNORECASE)


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("change times must be timezone-aware")
    return value.astimezone(timezone.utc)


def _norm(value: str) -> str:
    return " ".join(value.casefold().split())


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class StateObservation:
    memory_id: str
    slot_key: SlotKey
    value_text: str
    observed_at: datetime

    def __post_init__(self) -> None:
        if not self.memory_id or not self.value_text.strip():
            raise ValueError("state observation requires an ID and value")
        object.__setattr__(self, "observed_at", _utc(self.observed_at))


@dataclass(frozen=True)
class ChangeMention:
    """Source-grounded change candidate; nested quotation authority remains upstream."""

    mention_id: str
    slot_key: SlotKey
    old_value_text: str
    new_value_text: str
    occurred_at: datetime
    source_turn_id: str
    source_text: str
    owner_assertion_evidence: str
    provenance_sha256: str

    def __post_init__(self) -> None:
        if not self.mention_id or not self.source_turn_id or not self.source_text.strip():
            raise ValueError("change mention requires stable identity and source provenance")
        object.__setattr__(self, "occurred_at", _utc(self.occurred_at))
        if _sha256(self.source_text) != self.provenance_sha256:
            raise ValueError("change mention provenance hash does not match source text")
        if (
            not self.owner_assertion_evidence
            or self.source_text.count(self.owner_assertion_evidence) != 1
        ):
            raise ValueError("owner assertion evidence must be a unique exact source substring")
        if not _FIRST_PERSON.search(self.owner_assertion_evidence):
            raise ValueError("owner assertion evidence lacks first-person attribution")
        if (
            self.old_value_text not in self.owner_assertion_evidence
            or self.new_value_text not in self.owner_assertion_evidence
        ):
            raise ValueError("owner assertion evidence must bind both values")
        if not self.old_value_text.strip() or not self.new_value_text.strip():
            raise ValueError("change mention requires both values")
        if _norm(self.old_value_text) == _norm(self.new_value_text):
            raise ValueError("change mention values must differ")
        if not _CONTRAST_CUE.search(self.source_text):
            raise ValueError("change mention lacks a supported contrast cue")
        if self.source_text.count(self.old_value_text) != 1:
            raise ValueError("old value must be a unique exact source substring")
        if self.source_text.count(self.new_value_text) != 1:
            raise ValueError("new value must be a unique exact source substring")
        new_start = self.source_text.index(self.new_value_text)
        cue = _CONTRAST_CUE.search(self.source_text, new_start + len(self.new_value_text))
        old_start = self.source_text.index(self.old_value_text)
        if cue is None or cue.start() >= old_start:
            raise ValueError("source must place the proposed new value before old value around a contrast cue")


@dataclass(frozen=True)
class AdmissionDecision:
    mention: ChangeMention
    kind: AdmissionKind
    predecessor_memory_id: str | None
    predecessor_observed_at: datetime | None
    reason: str


def admit_change_mention(
    mention: ChangeMention,
    prior_observations: tuple[StateObservation, ...] | list[StateObservation] = (),
) -> AdmissionDecision:
    """Link only a unique, earlier, same-slot observation of the old value."""
    matches = [
        row
        for row in prior_observations
        if row.slot_key == mention.slot_key
        and _norm(row.value_text) == _norm(mention.old_value_text)
        and row.observed_at < mention.occurred_at
    ]
    if len(matches) == 1:
        predecessor = matches[0]
        return AdmissionDecision(
            mention=mention,
            kind="ANCHORED_REVISION",
            predecessor_memory_id=predecessor.memory_id,
            predecessor_observed_at=predecessor.observed_at,
            reason="unique_earlier_same_slot_old_value_observation",
        )
    if len(matches) > 1:
        return AdmissionDecision(
            mention=mention,
            kind="UNANCHORED_CHANGE_MENTION",
            predecessor_memory_id=None,
            predecessor_observed_at=None,
            reason="multiple_prior_old_value_observations_require_store_coalescing",
        )
    return AdmissionDecision(
        mention=mention,
        kind="UNANCHORED_CHANGE_MENTION",
        predecessor_memory_id=None,
        predecessor_observed_at=None,
        reason="no_unique_earlier_same_slot_old_value_observation",
    )


def project_current(decision: AdmissionDecision) -> dict[str, str | None]:
    """Project only the asserted successor into default current context."""
    return {
        "status": "CURRENT",
        "value_text": decision.mention.new_value_text,
        "valid_from": decision.mention.occurred_at.isoformat().replace("+00:00", "Z"),
        "source_turn_id": decision.mention.source_turn_id,
    }


def project_as_of(decision: AdmissionDecision, at: datetime) -> dict[str, str | None]:
    instant = _utc(at)
    mention = decision.mention
    if instant >= mention.occurred_at:
        return project_current(decision)
    if decision.kind != "ANCHORED_REVISION" or decision.predecessor_observed_at is None:
        return {
            "status": "UNRESOLVED",
            "value_text": None,
            "valid_from": None,
            "reason": "predecessor_interval_not_independently_anchored",
        }
    if instant < decision.predecessor_observed_at:
        return {
            "status": "UNRESOLVED",
            "value_text": None,
            "valid_from": None,
            "reason": "query_precedes_first_grounded_predecessor_observation",
        }
    return {
        "status": "HISTORICAL",
        "value_text": mention.old_value_text,
        "valid_from": decision.predecessor_observed_at.isoformat().replace("+00:00", "Z"),
        "valid_until": mention.occurred_at.isoformat().replace("+00:00", "Z"),
        "source_turn_id": decision.predecessor_memory_id,
    }


def project_change(decision: AdmissionDecision) -> dict[str, str | None]:
    mention = decision.mention
    return {
        "status": "CHANGE_MENTION",
        "old_value_text": mention.old_value_text,
        "new_value_text": mention.new_value_text,
        "occurred_at": mention.occurred_at.isoformat().replace("+00:00", "Z"),
        "predecessor_status": decision.kind,
        "predecessor_memory_id": decision.predecessor_memory_id,
        "source_turn_id": mention.source_turn_id,
    }
