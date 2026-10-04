"""Harness-owned request state, evidence ledger, and append-only execution trace."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import RLock
from typing import Any
from uuid import uuid4

from .contracts import (
    Citation,
    CoverageLedger,
    LeadPlan,
    TaskLedger,
    WorkerReport,
    WorkerRole,
    WorkerTask,
)


@dataclass(frozen=True)
class EvidenceLedgerEntry:
    evidence_id: str
    source_id: str
    worker_id: str
    role: str
    tool_call_id: str
    tool_id: str
    excerpt: str
    input_hash: str
    output_hash: str
    resource_versions: tuple[tuple[str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "source_id": self.source_id,
            "worker_id": self.worker_id,
            "role": self.role,
            "tool_call_id": self.tool_call_id,
            "tool_id": self.tool_id,
            "excerpt": self.excerpt,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "resource_versions": dict(self.resource_versions),
        }


@dataclass(frozen=True)
class ExecutionEvent:
    sequence: int
    event_id: str
    event_type: str
    timestamp: str
    fields: tuple[tuple[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "event_id": self.event_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            **dict(self.fields),
        }


class SharedContext:
    """Mutable request context with an explicit harness mutation surface."""

    def __init__(self, request_id: str, original_query: str) -> None:
        self.request_id = request_id
        self.original_query = original_query
        self._plan: LeadPlan | None = None
        self._task_ledger: TaskLedger | None = None
        self._coverage_ledger: CoverageLedger | None = None
        self._tasks: list[WorkerTask] = []
        self._worker_status: dict[str, str] = {}
        self._worker_outputs: list[WorkerReport] = []
        self._observed_facts: list[str] = []
        self._evidence_ledger: list[EvidenceLedgerEntry] = []
        self._citations: list[Citation] = []
        self._risk_flags: list[str] = []
        self._timeline: list[dict[str, Any]] = []
        self._events: list[ExecutionEvent] = []
        self._lock = RLock()

    @property
    def trace_id(self) -> str:
        return self.request_id

    def _harness_set_plan(self, plan: LeadPlan) -> None:
        with self._lock:
            self._plan = plan
            self._record("lead_plan", plan=plan.to_dict())

    def _harness_set_task_ledger(self, ledger: TaskLedger) -> None:
        with self._lock:
            self._task_ledger = ledger
            self._record("task_ledger_created", task_ledger=ledger.to_dict())

    def _harness_set_task_aspects(self, worker_id: str, aspect_ids: tuple[str, ...]) -> None:
        with self._lock:
            for index, task in enumerate(self._tasks):
                if task.worker_id == worker_id:
                    self._tasks[index] = WorkerTask(
                        task_id=task.task_id,
                        worker_id=task.worker_id,
                        role=task.role,
                        objective=task.objective,
                        depends_on=task.depends_on,
                        aspect_ids=aspect_ids,
                    )
                    self._record("worker_aspects_assigned", worker_id=worker_id,
                                 aspect_ids=list(aspect_ids))
                    return
            raise ValueError("task aspect assignment references an unknown worker")

    def _harness_set_coverage_ledger(self, ledger: CoverageLedger) -> None:
        with self._lock:
            self._coverage_ledger = ledger
            self._record("coverage_ledger_updated", coverage_ledger=ledger.to_dict())

    def _harness_register_task(
        self, role: str, objective: str, depends_on: tuple = (), aspect_ids: tuple[str, ...] = ()
    ) -> WorkerTask:
        with self._lock:
            normalized_role = WorkerRole(role)
            worker_id = f"worker-{role}-{uuid4().hex[:12]}"
            task = WorkerTask(
                task_id=f"task-{uuid4().hex[:12]}",
                worker_id=worker_id,
                role=normalized_role,
                objective=objective,
                depends_on=depends_on,
                aspect_ids=aspect_ids,
            )
            self._tasks.append(task)
            self._worker_status[worker_id] = "pending"
            self._record("task_registered", task=task.to_dict())
            return task

    def _harness_worker_status(self, worker_id: str, status: str, **detail: Any) -> None:
        with self._lock:
            if worker_id not in self._worker_status:
                raise ValueError("worker status update references an unknown worker")
            self._worker_status[worker_id] = status
            self._record("worker_status", worker_id=worker_id, status=status, **detail)

    def _harness_add_report(self, report: WorkerReport) -> None:
        with self._lock:
            if report.worker_id not in self._worker_status:
                raise ValueError("worker report references an unknown worker")
            self._worker_outputs.append(report)
            self._worker_status[report.worker_id] = report.status.value
            self._observed_facts.extend(report.observed_facts)
            self._citations.extend(report.citations)
            self._record("worker_report", report=report.to_dict())

    def _harness_add_evidence(
        self,
        *,
        source_id: str,
        worker_id: str,
        role: str,
        tool_call_id: str,
        tool_id: str,
        excerpt: str,
        input_hash: str,
        output_hash: str,
        resource_versions: tuple[tuple[str, str], ...],
    ) -> Citation:
        with self._lock:
            if (worker_id not in self._worker_status
                    and not worker_id.startswith(("single-", "fallback-single-"))):
                raise ValueError("evidence attribution references an unknown worker")
            if not source_id or not tool_call_id or not tool_id:
                raise ValueError("evidence provenance requires source, tool-call, and tool IDs")
            entry = EvidenceLedgerEntry(
                evidence_id=f"evidence-{uuid4().hex}",
                source_id=source_id,
                worker_id=worker_id,
                role=role,
                tool_call_id=tool_call_id,
                tool_id=tool_id,
                excerpt=excerpt,
                input_hash=input_hash,
                output_hash=output_hash,
                resource_versions=resource_versions,
            )
            self._evidence_ledger.append(entry)
            citation = Citation(
                evidence_id=entry.evidence_id,
                source_id=source_id,
                source=tool_id,
                excerpt=excerpt,
                worker_id=worker_id,
                tool_id=tool_id,
            )
            self._citations.append(citation)
            self._record("evidence_acquired", evidence=entry.to_dict())
            return citation

    def _harness_evidence_for_worker(self, worker_id: str) -> tuple[EvidenceLedgerEntry, ...]:
        with self._lock:
            return tuple(item for item in self._evidence_ledger if item.worker_id == worker_id)

    def _harness_risk_flag(self, flag: str) -> None:
        with self._lock:
            if flag not in self._risk_flags:
                self._risk_flags.append(flag)
                self._record("risk_flag", flag=flag)

    def _harness_record_call(self, **fields: Any) -> None:
        with self._lock:
            self._record("provider_call", **fields)

    def _harness_timeline(self, step: str, **fields: Any) -> None:
        with self._lock:
            row = {"step": step, **fields}
            self._timeline.append(row)
            self._record("timeline", **row)

    def _record(self, event_type: str, **fields: Any) -> None:
        self._events.append(ExecutionEvent(
            sequence=len(self._events),
            event_id=f"event-{uuid4().hex}",
            event_type=event_type,
            timestamp=datetime.now(UTC).isoformat(),
            fields=tuple(sorted(fields.items())),
        ))

    def lead_view(self) -> dict[str, Any]:
        """Return only observed context; evaluator truth has no path into this object."""
        with self._lock:
            return {
                "request_id": self.request_id,
                "original_query": self.original_query,
                "plan": self._plan.to_dict() if self._plan else None,
                "task_ledger": self._task_ledger.to_dict() if self._task_ledger else None,
                "coverage_ledger": self._coverage_ledger.to_dict() if self._coverage_ledger else None,
                "worker_status": dict(self._worker_status),
                "worker_reports": [report.to_dict() for report in self._worker_outputs],
                "worker_artifacts": [report.artifact.to_dict() for report in self._worker_outputs
                                     if report.artifact is not None],
                "evidence_ledger": [entry.to_dict() for entry in self._evidence_ledger],
                "observed_facts": list(dict.fromkeys(self._observed_facts)),
                "risk_flags": list(self._risk_flags),
            }

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return deepcopy({
                "request_id": self.request_id,
                "original_query": self.original_query,
                "plan": self._plan.to_dict() if self._plan else None,
                "task_ledger": self._task_ledger.to_dict() if self._task_ledger else None,
                "coverage_ledger": self._coverage_ledger.to_dict() if self._coverage_ledger else None,
                "tasks": [task.to_dict() for task in self._tasks],
                "worker_status": self._worker_status,
                "worker_reports": [report.to_dict() for report in self._worker_outputs],
                "worker_artifacts": [report.artifact.to_dict() for report in self._worker_outputs
                                     if report.artifact is not None],
                "observed_facts": list(dict.fromkeys(self._observed_facts)),
                "evidence_ledger": [entry.to_dict() for entry in self._evidence_ledger],
                "citations": [citation.to_dict() for citation in self._citations],
                "risk_flags": self._risk_flags,
                "timeline": self._timeline,
                "events": [event.to_dict() for event in self._events],
            })
