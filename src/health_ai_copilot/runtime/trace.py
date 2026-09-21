"""Append-only runtime trace with an explicit content privacy policy."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import Any


class TraceContentPolicy(StrEnum):
    """Whether a trace can contain public, reviewed evaluation content."""

    METADATA_ONLY = "metadata_only"
    PUBLIC_EVAL_CONTENT = "public_eval_content"


class TraceEventType(StrEnum):
    RUN_START = "run_start"
    RUN_END = "run_end"
    PROVIDER_START = "provider_start"
    PROVIDER_END = "provider_end"
    PROVIDER_ERROR = "provider_error"
    POLICY_DECISION = "policy_decision"
    TOOL_START = "tool_start"
    TOOL_END = "tool_end"
    BUDGET_DENIED = "budget_denied"
    HARNESS_DISPOSITION = "harness_disposition"
    TEAM_STARTED = "team_start"
    LEAD_DECISION = "lead_decision"
    LEAD_DECISION_RESULT = "lead_decision_result"
    TEAM_LEAD_FAILURE = "team_lead_failure"
    TASK_CREATED = "task_created"
    TASK_ASSIGNED = "task_assigned"
    TASK_STARTED = "task_started"
    WORKER_STARTED = "worker_started"
    WORKER_FINISHED = "worker_finished"
    WORKER_REPORT = "worker_report"
    MESSAGE_DELIVERED = "message_delivered"
    EVIDENCE_ADDED = "evidence_added"
    TEAM_FINAL_PROPOSED = "team_final"
    TEAM_STOPPED = "team_stop"
    PERMISSION_CHECK = "permission_check"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_GRANTED = "approval_granted"
    APPROVAL_DENIED = "approval_denied"
    MCP_CALL_START = "mcp_call_start"
    MCP_CALL_END = "mcp_call_end"
    MCP_CALL_ERROR = "mcp_call_error"
    SANDBOX_DENIED = "sandbox_denied"
    SECURITY_CONTROL = "security_control"
    SESSION_LOADED = "session_loaded"
    SESSION_COMMITTED = "session_committed"
    SESSION_COMMIT_FAILED = "session_commit_failed"
    SESSION_FORKED = "session_forked"
    MEMORY_QUERY = "memory_query"
    MEMORY_SELECTED = "memory_selected"
    MEMORY_WRITE_PROPOSED = "memory_write_proposed"
    MEMORY_WRITE_COMMITTED = "memory_write_committed"
    MEMORY_SUPERSEDED = "memory_superseded"
    MEMORY_DELETED = "memory_deleted"
    CONTEXT_PLAN = "context_plan"
    CONTEXT_COMPACTED = "context_compacted"
    CONTEXT_BUDGET_DENIED = "context_budget_denied"


@dataclass(frozen=True)
class TraceEvent:
    sequence: int
    event_type: TraceEventType
    fields: Mapping[str, Any]


class RunTrace:
    """Typed JSONL trace writer; metadata mode rejects content-bearing fields."""

    _CONTENT_FIELDS = frozenset(
        {
            "question",
            "answer",
            "claims",
            "evidence",
            "messages",
            "tool_query",
            "tool_arguments",
            "tool_result",
            "provider_content",
            "provider_response",
        }
    )

    def __init__(
        self,
        path: Path | None = None,
        content_policy: TraceContentPolicy = TraceContentPolicy.METADATA_ONLY,
    ) -> None:
        self.path = path
        self.content_policy = content_policy
        self.events: list[TraceEvent] = []
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8", newline="\n")

    def emit(self, event_type: TraceEventType, **fields: Any) -> TraceEvent:
        if self.content_policy == TraceContentPolicy.METADATA_ONLY:
            forbidden = self._CONTENT_FIELDS.intersection(fields)
            if forbidden:
                names = ", ".join(sorted(forbidden))
                raise ValueError(f"metadata-only trace cannot contain: {names}")
        event = TraceEvent(len(self.events) + 1, event_type, dict(fields))
        self.events.append(event)
        if self.path is not None:
            payload = asdict(event)
            payload["event_type"] = event.event_type.value
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
        return event

    def close(self, *, status: str) -> None:
        self.emit(TraceEventType.RUN_END, status=status, event_count=len(self.events))


def canonical_json_sha256(value: object) -> str:
    """Stable hash for replay matching without putting content into metadata traces."""

    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return sha256(encoded.encode("utf-8")).hexdigest()
