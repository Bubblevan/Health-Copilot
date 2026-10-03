"""Stable API and execution contracts for the routed medical team."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "worker_id": self.worker_id,
            "role": self.role.value,
            "objective": self.objective,
            "depends_on": [role.value for role in self.depends_on],
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
