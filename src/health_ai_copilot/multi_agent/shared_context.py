"""Per-request adaptive-agent trace state, without task or coverage ledgers."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import RLock
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class RegisteredTask:
    task_id: str
    worker_id: str
    role: str
    objective: str

    def to_dict(self) -> dict[str, str]:
        return {
            "task_id": self.task_id,
            "worker_id": self.worker_id,
            "role": self.role,
            "objective": self.objective,
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
    """Small trace collector for one adaptive request; the Harness owns evidence."""

    def __init__(self, request_id: str, original_query: str) -> None:
        self.request_id = request_id
        self.original_query = original_query
        self._tasks: list[RegisteredTask] = []
        self._worker_status: dict[str, str] = {}
        self._timeline: list[dict[str, Any]] = []
        self._events: list[ExecutionEvent] = []
        self._lock = RLock()

    def _harness_register_task(self, role: str, objective: str) -> RegisteredTask:
        with self._lock:
            worker_id = f"worker-{role}-{uuid4().hex[:12]}"
            task = RegisteredTask(
                task_id=f"task-{uuid4().hex[:12]}",
                worker_id=worker_id,
                role=role,
                objective=objective,
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

    def _harness_timeline(self, step: str, **fields: Any) -> None:
        with self._lock:
            row = {"step": step, **fields}
            self._timeline.append(row)
            self._record("timeline", **row)

    def _harness_record_call(self, **fields: Any) -> None:
        with self._lock:
            self._record("provider_call", **fields)

    def _record(self, event_type: str, **fields: Any) -> None:
        self._events.append(ExecutionEvent(
            sequence=len(self._events),
            event_id=f"event-{uuid4().hex}",
            event_type=event_type,
            timestamp=datetime.now(UTC).isoformat(),
            fields=tuple(sorted(fields.items())),
        ))

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return deepcopy({
                "request_id": self.request_id,
                "original_query": self.original_query,
                "tasks": [task.to_dict() for task in self._tasks],
                "worker_status": dict(self._worker_status),
                "timeline": list(self._timeline),
                "events": [event.to_dict() for event in self._events],
            })
