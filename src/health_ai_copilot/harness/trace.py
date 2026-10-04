"""Shared execution trace schema with metadata-only default persistence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class ExecutionEvent:
    sequence: int
    kind: str
    fields: dict[str, Any]


@dataclass
class ExecutionTrace:
    trace_id: str = field(default_factory=lambda: f"harness-{uuid4().hex}")
    query_sha256: str = ""
    profile_id: str = ""
    events: list[ExecutionEvent] = field(default_factory=list)

    def emit(self, kind: str, **fields: Any) -> ExecutionEvent:
        event = ExecutionEvent(len(self.events) + 1, kind, dict(fields))
        self.events.append(event)
        return event

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "request": {"query_sha256": self.query_sha256},
            "profile_id": self.profile_id,
            "memory_events": [asdict(item) for item in self.events if item.kind.startswith("memory_")],
            "retrieval_events": [asdict(item) for item in self.events if item.kind.startswith("retrieval_")],
            "reasoning_events": [asdict(item) for item in self.events if item.kind.startswith("reasoning_")],
            "model_calls": [asdict(item) for item in self.events if item.kind.startswith("model_")],
            "tool_calls": [asdict(item) for item in self.events if item.kind.startswith("tool_")],
            "events": [asdict(item) for item in self.events],
        }

    def to_jsonl(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def query_fingerprint(query: str) -> str:
    return sha256(query.encode("utf-8")).hexdigest()


class JsonlTraceSink:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, trace: ExecutionTrace) -> None:
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(trace.to_jsonl())
            handle.write("\n")
