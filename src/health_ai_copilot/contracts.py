"""Typed contracts shared by the runtime, evaluation and review layers."""

from dataclasses import dataclass, field
from enum import StrEnum


class Route(StrEnum):
    ANSWER = "answer"
    URGENT_CARE = "urgent_care"
    HUMAN_REVIEW = "human_review"


@dataclass(frozen=True)
class Citation:
    source_id: str
    title: str
    excerpt: str


@dataclass(frozen=True)
class AssistantResponse:
    route: Route
    message: str
    citations: list[Citation] = field(default_factory=list)
    safety_reasons: list[str] = field(default_factory=list)

