"""Stable API and execution contracts for the routed medical team."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from math import isfinite
from typing import Any

from ..research.integration.contracts import IntegrationEpisode
from ..research.integration.executor import ExecutionResources


class RouteMode(StrEnum):
    SINGLE = "SINGLE"
    TEAM = "TEAM"


class WorkerRole(StrEnum):
    PATIENT_CONTEXT = "patient_context"
    EVIDENCE = "evidence"
    CARE = "care"


class WorkerStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    PARTIAL = "partial"
    FAILED = "failed"
    TIMED_OUT = "timed_out"


class CoverageStatus(StrEnum):
    COVERED = "COVERED"
    PARTIAL = "PARTIAL"
    MISSING = "MISSING"


@dataclass(frozen=True)
class TriageDecision:
    """Question-only capability scores; answer generation stays with the LLM runtime."""

    need_patient_context: float
    need_external_evidence: float
    need_care_analysis: float
    complexity: float
    provider: str = "local"
    provider_version: str = "local-triage-v1"
    model: str | None = None
    fallback_reason: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "need_patient_context", "need_external_evidence",
            "need_care_analysis", "complexity",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.cost_usd is not None and (
            isinstance(self.cost_usd, bool)
            or not isinstance(self.cost_usd, (int, float))
            or not isfinite(float(self.cost_usd))
            or self.cost_usd < 0
        ):
            raise ValueError("cost_usd must be nonnegative when provided")

    def to_dict(self) -> dict[str, Any]:
        return {
            "need_patient_context": float(self.need_patient_context),
            "need_external_evidence": float(self.need_external_evidence),
            "need_care_analysis": float(self.need_care_analysis),
            "complexity": float(self.complexity),
            "provider": self.provider,
            "provider_version": self.provider_version,
            "model": self.model,
            "fallback_reason": self.fallback_reason,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_ms": round(self.latency_ms, 3),
            "cost_usd": self.cost_usd,
        }


@dataclass(frozen=True)
class PlannedAspect:
    """Untrusted Lead text before the harness attaches identities and assignments."""

    aspect: str
    worker: WorkerRole
    expected_output: str


@dataclass(frozen=True)
class TaskAspect:
    aspect_id: str
    aspect: str
    worker: WorkerRole
    expected_output: str
    acceptance_criteria: tuple[str, ...]
    status: str = "pending"

    def to_dict(self) -> dict[str, Any]:
        return {
            "aspect_id": self.aspect_id,
            "aspect": self.aspect,
            "worker": self.worker.value,
            "expected_output": self.expected_output,
            "acceptance_criteria": list(self.acceptance_criteria),
            "status": self.status,
        }


@dataclass(frozen=True)
class TaskAssignment:
    assignment_id: str
    aspect_id: str
    worker_id: str
    worker: WorkerRole
    objective: str
    status: str = "assigned"

    def to_dict(self) -> dict[str, Any]:
        return {
            "assignment_id": self.assignment_id,
            "aspect_id": self.aspect_id,
            "worker_id": self.worker_id,
            "worker": self.worker.value,
            "objective": self.objective,
            "status": self.status,
        }


@dataclass(frozen=True)
class TaskLedger:
    request_id: str
    aspects: tuple[TaskAspect, ...]
    assignments: tuple[TaskAssignment, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "aspects": [item.to_dict() for item in self.aspects],
            "assignments": [item.to_dict() for item in self.assignments],
            "acceptance_criteria": {
                item.aspect_id: list(item.acceptance_criteria) for item in self.aspects
            },
            "statuses": {item.aspect_id: item.status for item in self.aspects},
        }


@dataclass(frozen=True)
class ArtifactFact:
    fact_id: str
    text: str
    source_id: str
    tool_id: str

    def to_dict(self) -> dict[str, str]:
        return {
            "fact_id": self.fact_id,
            "text": self.text,
            "source_id": self.source_id,
            "tool_id": self.tool_id,
        }


@dataclass(frozen=True)
class WorkerArtifact:
    worker_id: str
    role: WorkerRole
    findings: tuple[str, ...] = ()
    facts: tuple[ArtifactFact, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    patient_record_refs: tuple[str, ...] = ()
    unresolved: tuple[str, ...] = ()
    answer_fragment: str = ""
    status: WorkerStatus = WorkerStatus.PENDING
    aspect_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "role": self.role.value,
            "findings": list(self.findings),
            "facts": [item.to_dict() for item in self.facts],
            "evidence_refs": list(self.evidence_refs),
            "patient_record_refs": list(self.patient_record_refs),
            "unresolved": list(self.unresolved),
            "answer_fragment": self.answer_fragment,
            "status": self.status.value,
            "aspect_ids": list(self.aspect_ids),
        }


@dataclass(frozen=True)
class CoverageItem:
    aspect_id: str
    status: CoverageStatus
    supporting_worker_ids: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "aspect_id": self.aspect_id,
            "status": self.status.value,
            "supporting_worker_ids": list(self.supporting_worker_ids),
            "evidence_refs": list(self.evidence_refs),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class CoverageLedger:
    request_id: str
    items: tuple[CoverageItem, ...]
    judge: str = "harness+bounded-model"

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "items": [item.to_dict() for item in self.items],
            "judge": self.judge,
        }


@dataclass(frozen=True)
class Citation:
    evidence_id: str
    source_id: str
    source: str
    excerpt: str
    worker_id: str
    tool_id: str

    def to_dict(self) -> dict[str, str]:
        return {
            "evidence_id": self.evidence_id,
            "source_id": self.source_id,
            "source": self.source,
            "excerpt": self.excerpt,
            "worker_id": self.worker_id,
            "tool_id": self.tool_id,
        }


@dataclass(frozen=True)
class WorkerTask:
    task_id: str
    worker_id: str
    role: WorkerRole
    objective: str
    depends_on: tuple[WorkerRole, ...] = ()
    aspect_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "worker_id": self.worker_id,
            "role": self.role.value,
            "objective": self.objective,
            "depends_on": [role.value for role in self.depends_on],
            "aspect_ids": list(self.aspect_ids),
        }


@dataclass(frozen=True)
class LeadPlan:
    mode: RouteMode
    tasks: tuple[tuple[WorkerRole, str], ...]
    source: str = "model"

    def __post_init__(self) -> None:
        if len(self.tasks) > 3:
            raise ValueError("Lead plan may contain at most three worker tasks")
        roles = [role for role, _ in self.tasks]
        if len(set(roles)) != len(roles):
            raise ValueError("Lead plan may assign each worker at most once")
        if any(not objective.strip() for _, objective in self.tasks):
            raise ValueError("worker objective must not be empty")
        if self.mode == RouteMode.SINGLE and self.tasks:
            raise ValueError("single plan must not contain worker tasks")
        if self.mode == RouteMode.TEAM and not self.tasks:
            raise ValueError("team plan must contain at least one worker task")

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "tasks": [
                {"worker": role.value, "objective": objective}
                for role, objective in self.tasks
            ],
            "source": self.source,
        }


@dataclass(frozen=True)
class WorkerReport:
    worker_id: str
    role: WorkerRole
    objective: str
    status: WorkerStatus
    answer_text: str = ""
    observed_facts: tuple[str, ...] = ()
    observed_evidence_ids: tuple[str, ...] = ()
    citations: tuple[Citation, ...] = ()
    tool_calls: int = 0
    provider_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    error: str | None = None
    model_turns: int = 0
    raw_output: str = ""
    artifact: WorkerArtifact | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "role": self.role.value,
            "objective": self.objective,
            "status": self.status.value,
            "answer_text": self.answer_text,
            "observed_facts": list(self.observed_facts),
            "observed_evidence_ids": list(self.observed_evidence_ids),
            "citations": [citation.to_dict() for citation in self.citations],
            "tool_calls": self.tool_calls,
            "provider_calls": self.provider_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_ms": round(self.latency_ms, 3),
            "error": self.error,
            "model_turns": self.model_turns,
            "raw_output": self.raw_output,
            "artifact": self.artifact.to_dict() if self.artifact else None,
        }


@dataclass(frozen=True)
class MedicalAgentRequest:
    """FastAPI-friendly request with optional harness-owned runtime bindings."""

    query: str
    conversation_context: tuple[str, ...] = ()
    request_id: str | None = None
    patient_id: str | None = None
    as_of_time: datetime | None = None
    episode: IntegrationEpisode | None = field(default=None, repr=False, compare=False)
    resources: ExecutionResources | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("query must not be empty")
        object.__setattr__(self, "conversation_context", tuple(self.conversation_context))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MedicalAgentRequest:
        return cls(
            query=str(payload["query"]),
            conversation_context=tuple(str(item) for item in payload.get("conversation_context", ())),
            request_id=str(payload["request_id"]) if payload.get("request_id") else None,
            patient_id=str(payload["patient_id"]) if payload.get("patient_id") else None,
            as_of_time=(datetime.fromisoformat(str(payload["as_of_time"]))
                        if payload.get("as_of_time") else None),
        )


@dataclass(frozen=True)
class MedicalAgentResponse:
    answer: str
    route_mode: RouteMode
    workers_used: tuple[str, ...]
    citations: tuple[Citation, ...]
    safety_flags: tuple[str, ...]
    trace_id: str
    latency_ms: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "route_mode": self.route_mode.value,
            "workers_used": list(self.workers_used),
            "citations": [citation.to_dict() for citation in self.citations],
            "safety_flags": list(self.safety_flags),
            "trace_id": self.trace_id,
            "latency_ms": self.latency_ms,
        }
