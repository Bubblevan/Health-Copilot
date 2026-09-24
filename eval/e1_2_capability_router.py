"""Question-only capability routing policy for the E1.2 medical RAG study."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

ROUTER_VERSION = "e1.2-capability-router-v1"
ROUTE_THRESHOLD = 0.5


class RetrievalAction(StrEnum):
    CLOSED_BOOK = "closed_book"
    RAG_BM25 = "rag_bm25"
    RAG_MEDCPT = "rag_medcpt"


JEV_QUESTIONS: dict[str, dict[str, Any]] = {
    "retrieval_likely_to_help": {
        "type": "noul",
        "instructions": (
            "Based only on this question, would consulting an external biomedical textbook passage "
            "likely add answer-relevant information beyond the question itself? Do not answer the "
            "medical question and do not infer an answer option."
        ),
        "criteria": {
            "true": "A relevant external biomedical passage is likely to help resolve a specific fact.",
            "false": "The question is broad or familiar enough that retrieval is unlikely to add useful information.",
        },
    },
    "requires_specialized_detail": {
        "type": "noul",
        "instructions": (
            "Based only on this question, does it require precise, specialized biomedical detail "
            "rather than a broad introductory fact? Do not answer the medical question."
        ),
        "criteria": {
            "true": "The requested fact is technical, narrow, or detail-intensive.",
            "false": "The question asks for a broad or introductory fact.",
        },
    },
}

_SPECIALIZED = re.compile(
    r"\b(?:mechanism|pathophysiolog\w*|etiolog\w*|pharmacokinetic\w*|"
    r"pharmacodynamic\w*|contraindication\w*|biomarker\w*|mutation\w*|"
    r"receptor\w*|signaling pathway\w*|sensitivity|specificity|dose[- ]response|"
    r"adverse effect\w*|gene expression|molecular|histopatholog\w*)\b",
    re.IGNORECASE,
)
_EVIDENCE_CUE = re.compile(
    r"\b(?:evidence|according to|study|studies|clinical trial|guideline|"
    r"recommend\w*|risk factor\w*|screening|treatment|diagnos\w*|prevention|"
    r"incidence|prevalence|association|which of the following)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RouteDecision:
    action: RetrievalAction
    policy_version: str
    reasons: tuple[str, ...]
    retrieval_help_probability: float | None = None
    specialized_detail_probability: float | None = None
    threshold: float = ROUTE_THRESHOLD
    fallback: bool = False
    fallback_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["action"] = self.action.value
        value["reasons"] = list(self.reasons)
        return value


def route_from_jev_probabilities(
    probabilities: Mapping[str, float], *, threshold: float = ROUTE_THRESHOLD
) -> RouteDecision:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    required = ("retrieval_likely_to_help", "requires_specialized_detail")
    parsed: dict[str, float] = {}
    for key in required:
        value = probabilities.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"Jev probability {key} must be numeric")
        parsed[key] = float(value)
        if not 0.0 <= parsed[key] <= 1.0:
            raise ValueError(f"Jev probability {key} must be between 0 and 1")

    retrieval_help = parsed["retrieval_likely_to_help"]
    specialized = parsed["requires_specialized_detail"]
    if retrieval_help < threshold:
        action = RetrievalAction.CLOSED_BOOK
        reasons = ("retrieval_help_below_threshold",)
    elif specialized >= threshold:
        action = RetrievalAction.RAG_MEDCPT
        reasons = ("retrieval_help_at_or_above_threshold", "specialized_detail_at_or_above_threshold")
    else:
        action = RetrievalAction.RAG_BM25
        reasons = ("retrieval_help_at_or_above_threshold", "specialized_detail_below_threshold")
    return RouteDecision(
        action=action,
        policy_version=ROUTER_VERSION,
        reasons=reasons,
        retrieval_help_probability=retrieval_help,
        specialized_detail_probability=specialized,
        threshold=threshold,
    )


def route_cheap(question: str, *, fallback_reason: str | None = None) -> RouteDecision:
    """A fixed lexical comparator and Jev failover; it never invokes a full team."""
    if not question.strip():
        raise ValueError("question must be non-empty")
    if _SPECIALIZED.search(question):
        action = RetrievalAction.RAG_MEDCPT
        reasons = ("fixed_specialized_lexicon_match",)
    elif _EVIDENCE_CUE.search(question):
        action = RetrievalAction.RAG_BM25
        reasons = ("fixed_evidence_cue_match",)
    else:
        action = RetrievalAction.CLOSED_BOOK
        reasons = ("no_fixed_retrieval_cue",)
    return RouteDecision(
        action=action,
        policy_version=ROUTER_VERSION,
        reasons=reasons,
        fallback=fallback_reason is not None,
        fallback_reason=fallback_reason,
    )


__all__ = [
    "JEV_QUESTIONS",
    "ROUTER_VERSION",
    "ROUTE_THRESHOLD",
    "RetrievalAction",
    "RouteDecision",
    "route_cheap",
    "route_from_jev_probabilities",
]
