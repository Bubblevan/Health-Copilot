"""Deterministic citation-ID verification for M0."""

from collections.abc import Sequence
from dataclasses import dataclass

from ..contracts import Evidence, GenerationDraft


@dataclass(frozen=True)
class CitationVerification:
    valid: bool
    citation_ids: list[str]
    reasons: list[str]


def verify_citations(
    draft: GenerationDraft, evidence: Sequence[Evidence]
) -> CitationVerification:
    """Verify IDs against retrieved evidence, preserving first-seen order.

    This checks citation integrity, not semantic entailment between claims and
    excerpts. Claim-level grounding is intentionally a later milestone.
    """
    evidence_ids = {item.source_id for item in evidence}
    unique_ids = list(dict.fromkeys(draft.citation_ids))
    reasons: list[str] = []

    if not draft.abstain and not unique_ids:
        reasons.append("missing_citation")
    fabricated = [source_id for source_id in unique_ids if source_id not in evidence_ids]
    if fabricated:
        reasons.append("invalid_citation")

    return CitationVerification(
        valid=not reasons,
        citation_ids=unique_ids,
        reasons=reasons,
    )
