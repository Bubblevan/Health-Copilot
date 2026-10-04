"""Shared reasoning context and strategy contract."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

from ..harness.contracts import AnswerSchema
from ..providers.retrieval import RetrievedEvidence


@dataclass(frozen=True)
class ReasoningContext:
    query: str
    conversation_context: tuple[str, ...]
    patient_state: tuple[str, ...]
    external_evidence: tuple[RetrievedEvidence, ...]
    answer_schema: AnswerSchema
    runtime_metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("reasoning context query must not be empty")
        object.__setattr__(self, "conversation_context", tuple(self.conversation_context))
        object.__setattr__(self, "patient_state", tuple(self.patient_state))
        object.__setattr__(self, "external_evidence", tuple(self.external_evidence))
        object.__setattr__(self, "runtime_metadata", dict(self.runtime_metadata))


@dataclass(frozen=True)
class ReasoningResult:
    answer_text: str
    citation_ids: tuple[str, ...] = ()
    safety_flags: tuple[str, ...] = ()
    provider_calls: int = 0
    tool_calls: int = 0
    input_tokens: int | None = 0
    output_tokens: int | None = 0
    reasoning_events: tuple[Mapping[str, object], ...] = ()
    failure_reason: str | None = None


class ReasoningStrategy(Protocol):
    async def reason(self, context: ReasoningContext) -> ReasoningResult:
        ...
