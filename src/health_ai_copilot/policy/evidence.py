"""Typed, fail-closed contract for recovery-search authorization.

An :class:`EvidencePolicy` judges the full ``(question, current evidence,
proposed_query)`` tuple.  It is not a relevance scorer for the evidence alone:

* ``SUFFICIENT``: the current observed evidence can answer the original
  question, so the proposed search is unnecessary.
* ``RECOVERABLE``: current evidence is insufficient, but the proposed query is
  a semantically aligned, in-domain retrieval recovery; one search is allowed.
* ``INSUFFICIENT``: current evidence is insufficient and the query cannot
  reasonably recover the missing evidence within this knowledge domain (or the
  question is out of domain); the tool is denied and the pipeline abstains.
* ``CONFLICTING``: current evidence materially conflicts on a fact needed to
  answer the question; the tool is denied and the pipeline abstains.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from ..contracts import Evidence
from ..knowledge.scope import KnowledgeScope
from ..runtime.context import RunContext


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
    matched_topic_ids: tuple[str, ...] = ()


class EvidencePolicy(Protocol):
    """Authorize at most one recovery search from question, evidence, and query.

    Implementations must apply the four-state operational semantics documented
    in this module to all three inputs, rather than treating ``evidence`` as
    the only object being classified.
    """

    def assess(
        self,
        question: str,
        evidence: Sequence[Evidence],
        proposed_query: str,
        *,
        runtime: RunContext | None = None,
    ) -> EvidenceAssessment:
        ...


def validate_assessment(
    assessment: EvidenceAssessment,
    evidence: Sequence[Evidence],
    knowledge_scope: KnowledgeScope | None = None,
) -> EvidenceAssessment:
    if not isinstance(assessment, EvidenceAssessment):
        raise TypeError("policy returned an invalid assessment")
    if not set(assessment.supporting_source_ids).issubset({item.source_id for item in evidence}):
        raise ValueError("policy referenced an unobserved source")
    if not set(assessment.reason_codes).issubset(KNOWN_REASON_CODES):
        raise ValueError("policy returned an unknown reason code")
    reason_codes = set(assessment.reason_codes)
    if assessment.decision == EvidenceDecision.SUFFICIENT and not assessment.supporting_source_ids:
        raise ValueError("sufficient policy needs a supporting source")
    if assessment.decision == EvidenceDecision.SUFFICIENT and "out_of_scope" in reason_codes:
        raise ValueError("sufficient policy cannot be out of scope")
    if assessment.decision == EvidenceDecision.CONFLICTING and "direct_support" in reason_codes:
        raise ValueError("conflicting policy cannot claim direct support")
    if knowledge_scope is not None:
        unknown_topics = set(assessment.matched_topic_ids) - knowledge_scope.topic_ids
        if unknown_topics:
            raise ValueError(f"policy referenced unknown capability topics: {sorted(unknown_topics)}")
        if assessment.decision == EvidenceDecision.RECOVERABLE and not assessment.matched_topic_ids:
            raise ValueError("recoverable policy needs a matched capability topic")
    return assessment
