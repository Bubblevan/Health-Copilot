"""Runtime-visible request and response contracts for Harness V1."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from math import isfinite
from typing import Any


class AnswerSchema(StrEnum):
    FREE_TEXT = "free_text"
    SINGLE_CHOICE = "single_choice"
    MULTI_SELECT = "multi_select"
    EXACT_TOKEN = "exact_token"
    ABSTAINABLE = "abstainable"


@dataclass(frozen=True)
class RuntimeResources:
    """Per-request limits; evaluator-only data is deliberately not representable here."""

    max_provider_calls: int | None = None
    max_tool_calls: int | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    deadline_ms: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "max_provider_calls", "max_tool_calls", "max_input_tokens", "max_output_tokens",
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{name} must be a nonnegative integer or None")
        if self.deadline_ms is not None and (
            isinstance(self.deadline_ms, bool)
            or not isinstance(self.deadline_ms, (int, float))
            or not isfinite(self.deadline_ms)
            or self.deadline_ms <= 0
        ):
            raise ValueError("deadline_ms must be a finite positive number")


@dataclass(frozen=True)
class HarnessRequest:
    """Information visible to runtime components; contains no gold or benchmark labels."""

    request_id: str
    query: str
    answer_schema: AnswerSchema
    conversation_context: tuple[str, ...] = ()
    subject_id: str | None = None
    as_of_time: datetime | None = None
    benchmark_case_id: str | None = None
    runtime_resources: RuntimeResources | None = None

    def __post_init__(self) -> None:
        if not self.request_id.strip():
            raise ValueError("request_id must not be empty")
        if not self.query.strip():
            raise ValueError("query must not be empty")
        if not isinstance(self.answer_schema, AnswerSchema):
            object.__setattr__(self, "answer_schema", AnswerSchema(self.answer_schema))
        context = tuple(self.conversation_context)
        if any(not isinstance(item, str) for item in context):
            raise TypeError("conversation_context must contain strings")
        object.__setattr__(self, "conversation_context", context)
        if self.subject_id is not None and not self.subject_id.strip():
            raise ValueError("subject_id must be nonempty or None")
        if self.benchmark_case_id is not None and not self.benchmark_case_id.strip():
            raise ValueError("benchmark_case_id must be nonempty or None")
        if self.as_of_time is not None and (
            self.as_of_time.tzinfo is None or self.as_of_time.utcoffset() is None
        ):
            raise ValueError("as_of_time must be timezone-aware")


@dataclass(frozen=True)
class HarnessCitation:
    evidence_id: str
    source_id: str
    source: str
    excerpt: str
    score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "source_id": self.source_id,
            "source": self.source,
            "excerpt": self.excerpt,
            "score": self.score,
        }


@dataclass(frozen=True)
class HarnessResponse:
    answer_text: str
    parsed_answer: str | tuple[str, ...] | None
    model_variant: str
    system_profile: str
    retrieval_mode: str
    memory_mode: str
    reasoning_mode: str
    citations: tuple[HarnessCitation, ...]
    safety_flags: tuple[str, ...]
    trace_id: str
    provider_calls: int
    tool_calls: int
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer_text": self.answer_text,
            "parsed_answer": list(self.parsed_answer)
            if isinstance(self.parsed_answer, tuple) else self.parsed_answer,
            "model_variant": self.model_variant,
            "system_profile": self.system_profile,
            "retrieval_mode": self.retrieval_mode,
            "memory_mode": self.memory_mode,
            "reasoning_mode": self.reasoning_mode,
            "citations": [item.to_dict() for item in self.citations],
            "safety_flags": list(self.safety_flags),
            "trace_id": self.trace_id,
            "provider_calls": self.provider_calls,
            "tool_calls": self.tool_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_ms": round(self.latency_ms, 3),
        }
