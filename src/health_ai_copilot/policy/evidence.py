"""Typed, fail-closed contract for recovery-search authorization."""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from ..contracts import Evidence


class EvidenceDecision(StrEnum):
    SUFFICIENT = "sufficient"
    RECOVERABLE = "recoverable"
    INSUFFICIENT = "insufficient"
    CONFLICTING = "conflicting"


KNOWN_REASON_CODES = frozenset({"direct_support", "related_but_incomplete", "out_of_scope", "missing_required_evidence", "conflicting_sources", "policy_error"})


@dataclass(frozen=True)
class EvidenceAssessment:
    decision: EvidenceDecision
    supporting_source_ids: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()


class EvidencePolicy(Protocol):
    def assess(self, question: str, evidence: Sequence[Evidence], proposed_query: str) -> EvidenceAssessment:
        ...


def validate_assessment(assessment: EvidenceAssessment, evidence: Sequence[Evidence]) -> EvidenceAssessment:
    if not isinstance(assessment, EvidenceAssessment):
        raise TypeError("policy returned an invalid assessment")
    if not set(assessment.supporting_source_ids).issubset({item.source_id for item in evidence}):
        raise ValueError("policy referenced an unobserved source")
    if not set(assessment.reason_codes).issubset(KNOWN_REASON_CODES):
        raise ValueError("policy returned an unknown reason code")
    reason_codes = set(assessment.reason_codes)
    if assessment.decision == EvidenceDecision.SUFFICIENT and "out_of_scope" in reason_codes:
        raise ValueError("sufficient policy cannot be out of scope")
    if assessment.decision == EvidenceDecision.CONFLICTING and "direct_support" in reason_codes:
        raise ValueError("conflicting policy cannot claim direct support")
    return assessment
