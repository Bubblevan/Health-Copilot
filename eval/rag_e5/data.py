"""Runtime-safe integration case identity and isolated evaluator metadata."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from eval.rag_e5.temporal import (
    SourceRelativeDecisionBoundary,
    parse_esl_naive_datetime,
)


class IntegrationTaskFamily(StrEnum):
    LONGITUDINAL_ONLY = "T0"
    EXTERNAL_ONLY = "T1"
    LONGITUDINAL_EXTERNAL = "T2"


@dataclass(frozen=True, slots=True)
class E5IntegrationCase:
    """Case identity and runtime inputs, without capability or teacher labels."""

    case_id: str
    user_id: str
    batch_id: str
    question: str
    decision_timestamp: str
    longitudinal_state_ref: str

    def __post_init__(self) -> None:
        for name in (
            "case_id",
            "user_id",
            "batch_id",
            "question",
            "longitudinal_state_ref",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty")
        if not isinstance(self.decision_timestamp, str):
            raise TypeError("decision_timestamp must be an ISO-8601 string")
        parse_esl_naive_datetime(self.decision_timestamp)

    @property
    def decision_boundary(self) -> SourceRelativeDecisionBoundary:
        return SourceRelativeDecisionBoundary(self.user_id, self.decision_timestamp)


@dataclass(frozen=True, slots=True)
class EvaluatorCaseMetadata:
    """Teacher-only labels/references; never serialize this into policy input."""

    case_id: str
    task_family: IntegrationTaskFamily
    required_internal_state_fields: tuple[str, ...]
    required_external_evidence_group: str | None
    evaluation_type: str
    evaluation_payload_ref: str
    generation_provenance: str

    def __post_init__(self) -> None:
        if not isinstance(self.task_family, IntegrationTaskFamily):
            object.__setattr__(self, "task_family", IntegrationTaskFamily(self.task_family))
        for name in ("case_id", "evaluation_type", "evaluation_payload_ref", "generation_provenance"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty")
        fields = tuple(sorted(set(self.required_internal_state_fields)))
        if any(not isinstance(item, str) or not item.strip() for item in fields):
            raise ValueError("required_internal_state_fields must contain non-empty strings")
        object.__setattr__(self, "required_internal_state_fields", fields)
        if self.required_external_evidence_group is not None and not isinstance(
            self.required_external_evidence_group, str
        ):
            raise TypeError("required_external_evidence_group must be a string or null")
        if self.task_family == IntegrationTaskFamily.LONGITUDINAL_ONLY:
            if not fields or self.required_external_evidence_group is not None:
                raise ValueError("T0 requires internal state and must not require external evidence")
        elif self.task_family == IntegrationTaskFamily.EXTERNAL_ONLY:
            if fields or not self.required_external_evidence_group:
                raise ValueError("T1 requires external evidence and must not require patient state")
        elif not fields or not self.required_external_evidence_group:
            raise ValueError("T2 requires both internal state and external evidence")


__all__ = ["E5IntegrationCase", "EvaluatorCaseMetadata", "IntegrationTaskFamily"]
