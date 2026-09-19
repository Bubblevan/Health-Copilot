"""Deterministic M3 rendering of already verified factual claims."""

from collections.abc import Sequence

from .grounding import GroundedClaim


def normalize_claims(claims: Sequence[GroundedClaim]) -> tuple[GroundedClaim, ...]:
    """Normalize whitespace and deduplicate exact claim text in first-seen order."""
    if not claims:
        raise ValueError("non-abstain claim set must not be empty")
    normalized: list[GroundedClaim] = []
    seen_text: set[str] = set()
    for claim in claims:
        text = " ".join(claim.text.split())
        if not text:
            raise ValueError("claim text must not be empty")
        if not claim.citation_ids:
            raise ValueError("claim citation IDs must not be empty")
        if text not in seen_text:
            normalized.append(GroundedClaim(text, tuple(claim.citation_ids)))
            seen_text.add(text)
    return tuple(normalized)


def materialize_verified_claims(claims: Sequence[GroundedClaim]) -> str:
    """Render only verified claim text, in deterministic order, without model calls."""
    return "\n".join(f"- {claim.text}" for claim in normalize_claims(claims))
