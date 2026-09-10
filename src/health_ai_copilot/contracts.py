"""Typed contracts shared by the runtime, evaluation and review layers."""

from dataclasses import dataclass, field
from enum import StrEnum


class Route(StrEnum):
    ANSWER = "answer"
    URGENT_CARE = "urgent_care"
    HUMAN_REVIEW = "human_review"
    ABSTAIN = "abstain"


@dataclass(frozen=True)
class Citation:
    source_id: str
    title: str
    excerpt: str
    source_url: str = ""


@dataclass(frozen=True)
class KnowledgeCard:
    """Versioned, reviewed public-source knowledge unit."""

    id: str
    title: str
    content: str
    source_url: str
    publisher: str
    published_at: str | None
    collected_at: str
    reviewed_at: str | None
    reviewer: str
    version: str
    expires_at: str | None
    audience: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Evidence:
    """A retrieval result whose metadata comes from a KnowledgeCard."""

    source_id: str
    title: str
    excerpt: str
    source_url: str
    score: float


@dataclass(frozen=True)
class GenerationDraft:
    """The only model output accepted by the M0 pipeline."""

    answer: str
    citation_ids: list[str]
    abstain: bool = False


@dataclass(frozen=True)
class AssistantResponse:
    route: Route
    message: str
    citations: list[Citation] = field(default_factory=list)
    safety_reasons: list[str] = field(default_factory=list)
