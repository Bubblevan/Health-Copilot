"""Deterministic integrity checks for model citations."""

from .citations import CitationVerification, verify_citations

__all__ = ["CitationVerification", "verify_citations"]
"""Deterministic citation and M2 claim-grounding verification."""

from .grounding import (
    ClaimResult,
    ClaimVerdict,
    GroundedClaim,
    GroundingResult,
    GroundingVerifier,
    OpenAICompatibleGroundingVerifier,
)

__all__ = [
    "ClaimResult",
    "ClaimVerdict",
    "GroundedClaim",
    "GroundingResult",
    "GroundingVerifier",
    "OpenAICompatibleGroundingVerifier",
]
