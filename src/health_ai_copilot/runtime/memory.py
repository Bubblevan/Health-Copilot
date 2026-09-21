"""Typed, provenance-aware long-term memory for M10.

This module intentionally implements a small deterministic baseline.  It has
no vector database, no learned write policy, and no authority over medical
evidence or the M9 security control plane.
"""

from __future__ import annotations

import copy
import json
import re
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from threading import RLock
from typing import Any, Protocol, Self, runtime_checkable
from uuid import uuid4

from .session import SCHEMA_VERSION, UnknownSchemaError, payload_sha256, utc_now_iso


class MemoryKind(StrEnum):
    PREFERENCE = "preference"
    TASK_STATE = "task_state"
    USER_ASSERTED_CONTEXT = "user_asserted_context"
    SESSION_NOTE = "session_note"


class MemorySourceType(StrEnum):
    USER_EXPLICIT = "user_explicit"
    TRUSTED_APPLICATION = "trusted_application"
    SESSION_DERIVED = "session_derived"
    ASSISTANT_GENERATED = "assistant_generated"
    TOOL_OUTPUT = "tool_output"
    MCP_OUTPUT = "mcp_output"
    RETRIEVED_WEB_TEXT = "retrieved_web_text"


class MemoryStatus(StrEnum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    DELETED = "deleted"
    EXPIRED = "expired"


class MemorySensitivity(StrEnum):
    NON_SENSITIVE = "non_sensitive"
    SENSITIVE_HEALTH = "sensitive_health"


class MemoryOperationType(StrEnum):
    ADD = "add"
    UPDATE = "update"
    DELETE = "delete"
    NOOP = "noop"


class MemoryConsentDecision(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_CONFIRMATION = "require_confirmation"


class MemoryFailure(RuntimeError):
    """Base typed failure for the memory boundary."""


class MemoryWriteDenied(MemoryFailure):
    pass


class MemoryVersionConflict(MemoryFailure):
    pass


class MemoryScopeViolation(MemoryFailure):
    pass


class MemoryNotFound(MemoryFailure):
    pass


class MemoryStoreError(MemoryFailure):
    pass


class Clock(Protocol):
    def now(self) -> datetime: ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)


class FakeClock:
    """Injectable UTC clock used by deterministic expiry tests."""

    def __init__(self, value: datetime | str) -> None:
        self._value = _as_datetime(value)

    def now(self) -> datetime:
        return self._value

    def set(self, value: datetime | str) -> None:
        self._value = _as_datetime(value)

    def advance(self, **kwargs: int) -> None:
        from datetime import timedelta

        self._value += timedelta(**kwargs)


@dataclass(frozen=True)
class ContextIntent:
    """Small optional cue for distinguishing the same entity under new goals."""

    goal: str | None = None
    action_type: str | None = None
    entity_types: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "entity_types", tuple(self.entity_types))
        for name in ("goal", "action_type"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"intent {name} must be non-empty or null")

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "action_type": self.action_type,
            "entity_types": list(self.entity_types),
        }

    @classmethod
    def from_value(cls, value: Any) -> ContextIntent | None:
        if value is None:
            return None
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("intent must be ContextIntent, object, or null")
        return cls(
            goal=value.get("goal"),
            action_type=value.get("action_type"),
            entity_types=tuple(value.get("entity_types", ())),
        )


def _as_datetime(value: datetime | str) -> datetime:
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, str):
        normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
        result = datetime.fromisoformat(normalized)
    else:
        raise TypeError("time values must be datetime or ISO-8601 strings")
    if result.tzinfo is None:
        result = result.replace(tzinfo=UTC)
    return result.astimezone(UTC)


def _iso(value: datetime | str | None) -> str | None:
    if value is None:
        return None
    return _as_datetime(value).isoformat().replace("+00:00", "Z")


def value_hash(value: Any) -> str:
    return payload_sha256(value)


@dataclass(frozen=True)
class MemoryRecord:
    memory_id: str
    scope_id: str
    key: str
    kind: MemoryKind
    value: Any
    value_sha256: str
    status: MemoryStatus = MemoryStatus.ACTIVE
    version: int = 1
    created_at: str = field(default_factory=utc_now_iso)
    valid_from: str | None = None
    valid_until: str | None = None
    expires_at: str | None = None
    source_event_ids: tuple[str, ...] = ()
    related_event_ids: tuple[str, ...] = ()
    source_run_id: str | None = None
    source_session_id: str | None = None
    objective_id: str | None = None
    intent: ContextIntent | None = None
    supersedes_id: str | None = None
    sensitivity: MemorySensitivity = MemorySensitivity.NON_SENSITIVE
    source_type: MemorySourceType = MemorySourceType.USER_EXPLICIT

    def __post_init__(self) -> None:
        for name in ("memory_id", "scope_id", "key"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"memory {name} must be non-empty")
        if self.version <= 0:
            raise ValueError("memory version must be positive")
        object.__setattr__(self, "kind", MemoryKind(self.kind))
        object.__setattr__(self, "status", MemoryStatus(self.status))
        object.__setattr__(self, "sensitivity", MemorySensitivity(self.sensitivity))
        object.__setattr__(self, "source_type", MemorySourceType(self.source_type))
        object.__setattr__(self, "source_event_ids", tuple(self.source_event_ids))
        object.__setattr__(self, "related_event_ids", tuple(self.related_event_ids))
        object.__setattr__(self, "intent", ContextIntent.from_value(self.intent))
        if self.status != MemoryStatus.DELETED and value_hash(self.value) != self.value_sha256:
            raise ValueError("memory value hash does not match value")
        for name in ("valid_from", "valid_until", "expires_at", "created_at"):
            value = getattr(self, name)
            if value is not None:
                _as_datetime(value)
        if self.valid_from and self.valid_until and _as_datetime(self.valid_until) < _as_datetime(self.valid_from):
            raise ValueError("valid_until precedes valid_from")

    @classmethod
    def create(
        cls,
        *,
        scope_id: str,
        key: str,
        kind: MemoryKind | str,
        value: Any,
        source_type: MemorySourceType | str = MemorySourceType.USER_EXPLICIT,
        sensitivity: MemorySensitivity | str = MemorySensitivity.NON_SENSITIVE,
        version: int = 1,
        memory_id: str | None = None,
        **kwargs: Any,
    ) -> MemoryRecord:
        return cls(
            memory_id=memory_id or f"memory-{uuid4().hex}",
            scope_id=scope_id,
            key=key,
            kind=MemoryKind(kind),
            value=copy.deepcopy(value),
            value_sha256=value_hash(value),
            source_type=MemorySourceType(source_type),
            sensitivity=MemorySensitivity(sensitivity),
            version=version,
            **kwargs,
        )

    def to_dict(self, *, include_value: bool = True) -> dict[str, Any]:
        value = copy.deepcopy(self.value) if include_value and self.status != MemoryStatus.DELETED else None
        return {
            "memory_id": self.memory_id,
            "scope_id": self.scope_id,
            "key": self.key,
            "kind": self.kind.value,
            "value": value,
            "value_sha256": self.value_sha256,
            "status": self.status.value,
            "version": self.version,
            "created_at": self.created_at,
            "valid_from": self.valid_from,
            "valid_until": self.valid_until,
            "expires_at": self.expires_at,
            "source_event_ids": list(self.source_event_ids),
            "related_event_ids": list(self.related_event_ids),
            "source_run_id": self.source_run_id,
            "source_session_id": self.source_session_id,
            "objective_id": self.objective_id,
            "intent": self.intent.to_dict() if self.intent else None,
            "supersedes_id": self.supersedes_id,
            "sensitivity": self.sensitivity.value,
            "source_type": self.source_type.value,
        }


@dataclass(frozen=True)
class MemoryOperation:
    operation: MemoryOperationType
    scope_id: str
    key: str = ""
    kind: MemoryKind = MemoryKind.PREFERENCE
    value: Any = None
    source_type: MemorySourceType = MemorySourceType.USER_EXPLICIT
    sensitivity: MemorySensitivity = MemorySensitivity.NON_SENSITIVE
    memory_id: str | None = None
    expected_version: int | None = None
    supersedes_id: str | None = None
    valid_from: str | None = None
    valid_until: str | None = None
    expires_at: str | None = None
    source_event_ids: tuple[str, ...] = ()
    related_event_ids: tuple[str, ...] = ()
    source_run_id: str | None = None
    source_session_id: str | None = None
    objective_id: str | None = None
    intent: ContextIntent | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation", MemoryOperationType(self.operation))
        object.__setattr__(self, "kind", MemoryKind(self.kind))
        object.__setattr__(self, "source_type", MemorySourceType(self.source_type))
        object.__setattr__(self, "sensitivity", MemorySensitivity(self.sensitivity))
        object.__setattr__(self, "source_event_ids", tuple(self.source_event_ids))
        object.__setattr__(self, "related_event_ids", tuple(self.related_event_ids))
        object.__setattr__(self, "intent", ContextIntent.from_value(self.intent))
        if not self.scope_id.strip():
            raise ValueError("memory operation scope_id must be non-empty")
        if self.operation != MemoryOperationType.NOOP and not self.key.strip() and not self.memory_id:
            raise ValueError("memory operation requires key or memory_id")

    @classmethod
    def add(cls, **kwargs: Any) -> MemoryOperation:
        return cls(MemoryOperationType.ADD, **kwargs)

    @classmethod
    def update(cls, **kwargs: Any) -> MemoryOperation:
        return cls(MemoryOperationType.UPDATE, **kwargs)

    @classmethod
    def delete(cls, **kwargs: Any) -> MemoryOperation:
        return cls(MemoryOperationType.DELETE, **kwargs)

    @classmethod
    def noop(cls, *, scope_id: str, key: str = "") -> MemoryOperation:
        return cls(MemoryOperationType.NOOP, scope_id=scope_id, key=key)

    @property
    def value_sha256(self) -> str | None:
        return value_hash(self.value) if self.operation != MemoryOperationType.DELETE else None

    def to_dict(self, *, include_value: bool = True) -> dict[str, Any]:
        return {
            "operation": self.operation.value,
            "scope_id": self.scope_id,
            "key": self.key,
            "kind": self.kind.value,
            "value": copy.deepcopy(self.value) if include_value else None,
            "source_type": self.source_type.value,
            "sensitivity": self.sensitivity.value,
            "memory_id": self.memory_id,
            "expected_version": self.expected_version,
            "supersedes_id": self.supersedes_id,
            "valid_from": self.valid_from,
            "valid_until": self.valid_until,
            "expires_at": self.expires_at,
            "source_event_ids": list(self.source_event_ids),
            "related_event_ids": list(self.related_event_ids),
            "source_run_id": self.source_run_id,
            "source_session_id": self.source_session_id,
            "objective_id": self.objective_id,
            "intent": self.intent.to_dict() if self.intent else None,
        }


MemoryWriteRequest = MemoryOperation


@dataclass(frozen=True)
class MemoryEvent:
    event_id: str
    operation: MemoryOperationType
    memory_id: str | None
    previous_memory_id: str | None
    scope_id: str
    source: MemorySourceType
    timestamp: str
    payload_sha256: str | None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "operation": self.operation.value,
            "memory_id": self.memory_id,
            "previous_memory_id": self.previous_memory_id,
            "scope_id": self.scope_id,
            "source": self.source.value,
            "timestamp": self.timestamp,
            "payload_sha256": self.payload_sha256,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class MemoryQuery:
    scope_id: str
    text: str = ""
    keys: tuple[str, ...] = ()
    objective_id: str | None = None
    intent: ContextIntent | None = None
    now: datetime | str | None = None
    top_k: int = 5
    current_values: Mapping[str, Any] = field(default_factory=dict)
    explicit_values: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.scope_id.strip():
            raise ValueError("memory query scope_id must be non-empty")
        if self.top_k <= 0:
            raise ValueError("memory query top_k must be positive")
        object.__setattr__(self, "keys", tuple(self.keys))
        object.__setattr__(self, "intent", ContextIntent.from_value(self.intent))
        merged = {**dict(self.current_values), **dict(self.explicit_values)}
        object.__setattr__(self, "current_values", merged)
        object.__setattr__(self, "explicit_values", merged)


@dataclass(frozen=True)
class MemoryMatch:
    record: MemoryRecord
    score: float
    reasons: tuple[str, ...] = ()

    @property
    def memory_id(self) -> str:
        return self.record.memory_id

    def __getattr__(self, name: str) -> Any:
        try:
            record = object.__getattribute__(self, "record")
        except AttributeError as exc:
            raise AttributeError(name) from exc
        return getattr(record, name)


@dataclass(frozen=True)
class MemorySnapshotIdentity:
    scope_id_sha256: str
    active_memory_ids: tuple[str, ...]
    active_versions: tuple[int, ...]
    active_content_hashes: tuple[str, ...]
    snapshot_sha256: str
    session_revision: int | None = None

    @classmethod
    def from_records(
        cls, scope_id: str, records: Sequence[MemoryRecord], *, session_revision: int | None = None
    ) -> MemorySnapshotIdentity:
        ordered = sorted(records, key=lambda item: item.memory_id)
        body = {
            "scope_id_sha256": sha256(scope_id.encode("utf-8")).hexdigest(),
            "active": [
                {"memory_id": r.memory_id, "version": r.version, "value_sha256": r.value_sha256}
                for r in ordered
            ],
            "session_revision": session_revision,
        }
        digest = sha256(
            json.dumps(body, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
        ).hexdigest()
        return cls(
            body["scope_id_sha256"],
            tuple(r.memory_id for r in ordered),
            tuple(r.version for r in ordered),
            tuple(r.value_sha256 for r in ordered),
            digest,
            session_revision,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope_id_sha256": self.scope_id_sha256,
            "active_memory_ids": list(self.active_memory_ids),
            "active_versions": list(self.active_versions),
            "active_content_hashes": list(self.active_content_hashes),
            "snapshot_sha256": self.snapshot_sha256,
            "session_revision": self.session_revision,
        }


class MemoryConsentProvider(Protocol):
    def decide(self, request: MemoryOperation) -> MemoryConsentDecision | str: ...


class AllowAllMemoryConsent:
    def decide(self, request: MemoryOperation) -> MemoryConsentDecision:
        return MemoryConsentDecision.ALLOW


class StaticMemoryConsentProvider:
    def __init__(self, decision: MemoryConsentDecision | str) -> None:
        self.decision = MemoryConsentDecision(decision)

    def decide(self, request: MemoryOperation) -> MemoryConsentDecision:
        return self.decision


class MemoryPolicy:
    """Deterministic validation/materialization gate; never model-owned."""

    version = "m10-policy-v1"
    allowed_kinds = frozenset(MemoryKind)
    allowed_sources = frozenset(
        {MemorySourceType.USER_EXPLICIT, MemorySourceType.TRUSTED_APPLICATION, MemorySourceType.SESSION_DERIVED}
    )

    def __init__(self, *, consent_provider: MemoryConsentProvider | None = None) -> None:
        self.consent_provider = consent_provider

    def validate(self, request: MemoryOperation, current: MemoryRecord | None = None) -> None:
        if request.operation == MemoryOperationType.NOOP:
            return
        if request.operation == MemoryOperationType.ADD and current is not None and current.status == MemoryStatus.ACTIVE:
            raise MemoryVersionConflict(
                f"active memory already exists for scope/key: {request.scope_id}/{request.key}"
            )
        if request.kind not in self.allowed_kinds:
            raise MemoryWriteDenied(f"memory kind is not allowed: {request.kind.value}")
        if request.source_type not in self.allowed_sources:
            raise MemoryWriteDenied(f"memory source is not trusted: {request.source_type.value}")
        if request.sensitivity == MemorySensitivity.SENSITIVE_HEALTH:
            if self.consent_provider is None:
                raise MemoryWriteDenied("sensitive health memory requires explicit consent")
            decision = MemoryConsentDecision(self.consent_provider.decide(request))
            if decision != MemoryConsentDecision.ALLOW:
                raise MemoryWriteDenied(f"memory consent decision: {decision.value}")
        if request.expected_version is not None:
            actual = current.version if current else None
            if actual != request.expected_version:
                raise MemoryVersionConflict(
                    f"memory version conflict: expected {request.expected_version}, current {actual}"
                )
        if request.operation in {MemoryOperationType.ADD, MemoryOperationType.UPDATE}:
            if request.value is None:
                raise MemoryWriteDenied("memory value is required for add/update")
            _validate_temporal_fields(request.valid_from, request.valid_until, request.expires_at)
        if current is not None and current.scope_id != request.scope_id:
            raise MemoryScopeViolation("memory operation scope does not match target")
        if request.supersedes_id and current and request.supersedes_id != current.memory_id:
            raise MemoryVersionConflict("supersession target does not match current memory")

    def authorize(self, request: MemoryOperation, current: MemoryRecord | None = None) -> None:
        self.validate(request, current)


def _validate_temporal_fields(valid_from: str | None, valid_until: str | None, expires_at: str | None) -> None:
    start = _as_datetime(valid_from) if valid_from else None
    end = _as_datetime(valid_until) if valid_until else None
    expiry = _as_datetime(expires_at) if expires_at else None
    if start and end and end < start:
        raise MemoryWriteDenied("valid_until precedes valid_from")
    if start and expiry and expiry < start:
        raise MemoryWriteDenied("expires_at precedes valid_from")


@runtime_checkable
class MemoryStore(Protocol):
    schema_version: str

    def apply(self, request: MemoryOperation, *, now: datetime | str | None = None) -> MemoryRecord | None: ...

    def query(self, query: MemoryQuery) -> list[MemoryRecord]: ...

    def matches(self, query: MemoryQuery) -> list[MemoryMatch]: ...

    def active_records(self, scope_id: str, *, now: datetime | str | None = None) -> list[MemoryRecord]: ...

    def history(self, scope_id: str | None = None) -> list[MemoryEvent]: ...


class _MemoryStoreCore:
    def __init__(self, policy: MemoryPolicy | None = None, clock: Clock | None = None) -> None:
        self.policy = policy or MemoryPolicy()
        self.clock = clock or SystemClock()
        self._records: dict[str, MemoryRecord] = {}
        self._active_by_key: dict[tuple[str, str], str] = {}
        self._history: list[MemoryEvent] = []
        self._lock = RLock()

    def _current(self, request: MemoryOperation) -> MemoryRecord | None:
        identifier = request.memory_id or self._active_by_key.get((request.scope_id, request.key))
        return self._records.get(identifier) if identifier else None

    def _apply_unlocked(self, request: MemoryOperation, *, now: datetime | str | None = None) -> MemoryRecord | None:
        current = self._current(request)
        self.policy.validate(request, current)
        operation = request.operation
        timestamp = _iso(now or self.clock.now()) or utc_now_iso()
        if operation == MemoryOperationType.NOOP:
            return current
        if operation == MemoryOperationType.DELETE:
            if current is None:
                raise MemoryNotFound(request.memory_id or request.key)
            tombstone = MemoryRecord(
                memory_id=current.memory_id,
                scope_id=current.scope_id,
                key=current.key,
                kind=current.kind,
                value=None,
                value_sha256=current.value_sha256,
                status=MemoryStatus.DELETED,
                version=current.version + 1,
                created_at=current.created_at,
                valid_from=current.valid_from,
                valid_until=current.valid_until,
                expires_at=current.expires_at,
                source_event_ids=current.source_event_ids,
                related_event_ids=current.related_event_ids,
                source_run_id=current.source_run_id,
                source_session_id=current.source_session_id,
                objective_id=current.objective_id,
                intent=current.intent,
                supersedes_id=current.supersedes_id,
                sensitivity=current.sensitivity,
                source_type=current.source_type,
            )
            self._records[current.memory_id] = tombstone
            self._active_by_key.pop((current.scope_id, current.key), None)
            self._history.append(
                MemoryEvent(
                    f"memory-event-{uuid4().hex}", operation, current.memory_id, current.memory_id,
                    current.scope_id, request.source_type, timestamp, current.value_sha256,
                    {"status": MemoryStatus.DELETED.value},
                )
            )
            return tombstone
        if operation == MemoryOperationType.UPDATE:
            target = current
            if target is None:
                raise MemoryNotFound(request.memory_id or request.key)
            new_record = MemoryRecord.create(
                scope_id=request.scope_id,
                key=request.key or target.key,
                kind=request.kind,
                value=request.value,
                source_type=request.source_type,
                sensitivity=request.sensitivity,
                version=target.version + 1,
                supersedes_id=target.memory_id,
                created_at=timestamp,
                valid_from=request.valid_from,
                valid_until=request.valid_until,
                expires_at=request.expires_at,
                source_event_ids=request.source_event_ids,
                related_event_ids=request.related_event_ids,
                source_run_id=request.source_run_id,
                source_session_id=request.source_session_id,
                objective_id=request.objective_id,
                intent=request.intent,
            )
            old = MemoryRecord(**{**target.to_dict(), "value": target.value, "status": MemoryStatus.SUPERSEDED.value})
            self._records[target.memory_id] = old
            self._records[new_record.memory_id] = new_record
            self._active_by_key[(new_record.scope_id, new_record.key)] = new_record.memory_id
            self._history.append(
                MemoryEvent(
                    f"memory-event-{uuid4().hex}", operation, new_record.memory_id, target.memory_id,
                    new_record.scope_id, request.source_type, timestamp, new_record.value_sha256,
                    {"superseded": target.memory_id, "version": new_record.version},
                )
            )
            return new_record
        new_record = MemoryRecord.create(
            scope_id=request.scope_id,
            key=request.key,
            kind=request.kind,
            value=request.value,
            source_type=request.source_type,
            sensitivity=request.sensitivity,
            version=1,
            created_at=timestamp,
            valid_from=request.valid_from,
            valid_until=request.valid_until,
            expires_at=request.expires_at,
            source_event_ids=request.source_event_ids,
            related_event_ids=request.related_event_ids,
            source_run_id=request.source_run_id,
            source_session_id=request.source_session_id,
            objective_id=request.objective_id,
            intent=request.intent,
        )
        self._records[new_record.memory_id] = new_record
        self._active_by_key[(new_record.scope_id, new_record.key)] = new_record.memory_id
        self._history.append(
            MemoryEvent(
                f"memory-event-{uuid4().hex}", operation, new_record.memory_id, None,
                new_record.scope_id, request.source_type, timestamp, new_record.value_sha256,
                {"version": 1},
            )
        )
        return new_record

    def _valid(self, record: MemoryRecord, now: datetime) -> bool:
        if record.status != MemoryStatus.ACTIVE:
            return False
        if record.valid_from and now < _as_datetime(record.valid_from):
            return False
        if record.valid_until and now > _as_datetime(record.valid_until):
            return False
        return not (record.expires_at and now >= _as_datetime(record.expires_at))

    def _active(self, scope_id: str, now: datetime) -> list[MemoryRecord]:
        return [
            record
            for record in self._records.values()
            if record.scope_id == scope_id and self._valid(record, now)
        ]

    def _matches(self, query: MemoryQuery) -> list[MemoryMatch]:
        now = _as_datetime(query.now) if query.now else _as_datetime(self.clock.now())
        query_keys = set(query.keys)
        matches: list[MemoryMatch] = []
        for record in self._active(query.scope_id, now):
            if query_keys and record.key not in query_keys:
                continue
            if record.key in query.current_values and record.value != query.current_values[record.key]:
                continue
            if query.objective_id and record.objective_id and query.objective_id != record.objective_id:
                continue
            if query.intent and record.intent and not _intent_compatible(query.intent, record.intent):
                continue
            tokens = _tokens(query.text)
            value_text = f"{record.key} {record.value if isinstance(record.value, str) else json.dumps(record.value, ensure_ascii=False)}"
            value_tokens = _tokens(value_text)
            overlap = len(tokens & value_tokens)
            exact_key = 1.0 if record.key in query_keys or record.key in query.text else 0.0
            score = exact_key * 10.0 + (overlap / max(len(tokens), 1))
            if not tokens and not query_keys:
                score = 0.0
            reasons = ["scope", "active", "time_valid"]
            if exact_key:
                reasons.append("exact_key")
            if query.intent:
                reasons.append("intent_compatible")
            matches.append(MemoryMatch(record, score, tuple(reasons)))
        matches.sort(key=lambda item: (-item.score, -item.record.version, item.record.created_at, item.record.memory_id))
        return matches[: query.top_k]


def _tokens(value: str) -> set[str]:
    normalized = value.lower()
    words = set(re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", normalized))
    return words


def _intent_compatible(query: ContextIntent, record: ContextIntent) -> bool:
    if query.goal and record.goal and query.goal != record.goal:
        return False
    if query.action_type and record.action_type and query.action_type != record.action_type:
        return False
    return not (
        query.entity_types
        and record.entity_types
        and not (set(query.entity_types) & set(record.entity_types))
    )


class InMemoryMemoryStore(_MemoryStoreCore):
    schema_version = SCHEMA_VERSION

    def apply(self, request: MemoryOperation, *, now: datetime | str | None = None) -> MemoryRecord | None:
        with self._lock:
            return self._apply_unlocked(request, now=now)

    write = apply

    def query(self, query: MemoryQuery) -> list[MemoryRecord]:
        return [match.record for match in self.matches(query)]

    def matches(self, query: MemoryQuery) -> list[MemoryMatch]:
        with self._lock:
            return copy.deepcopy(self._matches(query))

    def active_records(self, scope_id: str, *, now: datetime | str | None = None) -> list[MemoryRecord]:
        with self._lock:
            return copy.deepcopy(self._active(scope_id, _as_datetime(now or self.clock.now())))

    def history(self, scope_id: str | None = None) -> list[MemoryEvent]:
        with self._lock:
            return copy.deepcopy([item for item in self._history if scope_id is None or item.scope_id == scope_id])

    def snapshot(self, scope_id: str, *, session_revision: int | None = None) -> MemorySnapshotIdentity:
        return MemorySnapshotIdentity.from_records(
            scope_id, self.active_records(scope_id), session_revision=session_revision
        )

    def close(self) -> None:
        return None


class SQLiteMemoryStore(_MemoryStoreCore):
    schema_version = SCHEMA_VERSION

    def __init__(self, path: str | Path, *, policy: MemoryPolicy | None = None, clock: Clock | None = None) -> None:
        super().__init__(policy=policy, clock=clock)
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._initialize()
        self._load_state()

    def _initialize(self) -> None:
        found = self._connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='m10_schema'"
        ).fetchone()
        if found is None:
            self._connection.executescript(
                """
                CREATE TABLE m10_schema (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                INSERT INTO m10_schema(key, value) VALUES ('schema_version', 'v1');
                CREATE TABLE memory_records (
                    memory_id TEXT PRIMARY KEY,
                    scope_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    value_json TEXT,
                    value_sha256 TEXT NOT NULL,
                    status TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    valid_from TEXT,
                    valid_until TEXT,
                    expires_at TEXT,
                    source_event_ids_json TEXT NOT NULL,
                    related_event_ids_json TEXT NOT NULL,
                    source_run_id TEXT,
                    source_session_id TEXT,
                    objective_id TEXT,
                    intent_json TEXT,
                    supersedes_id TEXT,
                    sensitivity TEXT NOT NULL,
                    source_type TEXT NOT NULL
                );
                CREATE TABLE memory_events (
                    event_id TEXT PRIMARY KEY,
                    operation TEXT NOT NULL,
                    memory_id TEXT,
                    previous_memory_id TEXT,
                    scope_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    payload_sha256 TEXT,
                    metadata_json TEXT NOT NULL
                );
                """
            )
            return
        row = self._connection.execute("SELECT value FROM m10_schema WHERE key='schema_version'").fetchone()
        if row is None or row[0] != SCHEMA_VERSION:
            raise UnknownSchemaError(f"unsupported memory schema: {row[0] if row else 'missing'}")

    def _load_state(self) -> None:
        for row in self._connection.execute("SELECT * FROM memory_records"):
            record = self._row_to_record(row)
            self._records[record.memory_id] = record
            if record.status == MemoryStatus.ACTIVE:
                self._active_by_key[(record.scope_id, record.key)] = record.memory_id
        for row in self._connection.execute("SELECT * FROM memory_events ORDER BY rowid"):
            self._history.append(
                MemoryEvent(
                    row["event_id"], row["operation"], row["memory_id"], row["previous_memory_id"],
                    row["scope_id"], row["source"], row["timestamp"], row["payload_sha256"],
                    json.loads(row["metadata_json"]),
                )
            )

    def apply(self, request: MemoryOperation, *, now: datetime | str | None = None) -> MemoryRecord | None:
        with self._lock:
            current = self._current(request)
            self.policy.validate(request, current)
            with self._transaction():
                before = len(self._history)
                record = self._apply_unlocked(request, now=now)
                for item in self._history[before:]:
                    self._persist_event(item)
                if record is not None and request.operation != MemoryOperationType.NOOP:
                    self._persist_record(record)
                    if request.operation == MemoryOperationType.UPDATE and record.supersedes_id:
                        self._persist_record(self._records[record.supersedes_id])
                return record

    write = apply

    def _persist_record(self, record: MemoryRecord) -> None:
        self._connection.execute(
            """INSERT OR REPLACE INTO memory_records VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record.memory_id, record.scope_id, record.key, record.kind.value,
                None if record.status == MemoryStatus.DELETED else _json_value(record.value),
                record.value_sha256, record.status.value, record.version, record.created_at,
                record.valid_from, record.valid_until, record.expires_at,
                _json_value(list(record.source_event_ids)), _json_value(list(record.related_event_ids)),
                record.source_run_id, record.source_session_id, record.objective_id,
                _json_value(record.intent.to_dict()) if record.intent else None,
                record.supersedes_id, record.sensitivity.value, record.source_type.value,
            ),
        )

    def _persist_event(self, event: MemoryEvent) -> None:
        self._connection.execute(
            "INSERT INTO memory_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                event.event_id, event.operation.value, event.memory_id, event.previous_memory_id,
                event.scope_id, event.source.value, event.timestamp, event.payload_sha256,
                _json_value(event.metadata),
            ),
        )

    def _row_to_record(self, row: sqlite3.Row) -> MemoryRecord:
        value = json.loads(row["value_json"]) if row["value_json"] is not None else None
        return MemoryRecord(
            memory_id=row["memory_id"], scope_id=row["scope_id"], key=row["key"], kind=row["kind"],
            value=value, value_sha256=row["value_sha256"], status=row["status"], version=row["version"],
            created_at=row["created_at"], valid_from=row["valid_from"], valid_until=row["valid_until"],
            expires_at=row["expires_at"], source_event_ids=tuple(json.loads(row["source_event_ids_json"])),
            related_event_ids=tuple(json.loads(row["related_event_ids_json"])), source_run_id=row["source_run_id"],
            source_session_id=row["source_session_id"], objective_id=row["objective_id"],
            intent=json.loads(row["intent_json"]) if row["intent_json"] else None,
            supersedes_id=row["supersedes_id"], sensitivity=row["sensitivity"], source_type=row["source_type"],
        )

    def query(self, query: MemoryQuery) -> list[MemoryRecord]:
        return [match.record for match in self.matches(query)]

    def matches(self, query: MemoryQuery) -> list[MemoryMatch]:
        with self._lock:
            return copy.deepcopy(self._matches(query))

    def active_records(self, scope_id: str, *, now: datetime | str | None = None) -> list[MemoryRecord]:
        with self._lock:
            return copy.deepcopy(self._active(scope_id, _as_datetime(now or self.clock.now())))

    def history(self, scope_id: str | None = None) -> list[MemoryEvent]:
        with self._lock:
            return copy.deepcopy([item for item in self._history if scope_id is None or item.scope_id == scope_id])

    def snapshot(self, scope_id: str, *, session_revision: int | None = None) -> MemorySnapshotIdentity:
        return MemorySnapshotIdentity.from_records(
            scope_id, self.active_records(scope_id), session_revision=session_revision
        )

    class _Tx:
        def __init__(self, owner: SQLiteMemoryStore) -> None:
            self.owner = owner

        def __enter__(self):
            self.owner._connection.execute("BEGIN IMMEDIATE")
            return self.owner

        def __exit__(self, exc_type, exc, tb):
            self.owner._connection.execute("ROLLBACK" if exc_type else "COMMIT")
            return False

    def _transaction(self):
        return self._Tx(self)

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _json_value(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise TypeError("memory value must be JSON-compatible") from exc


__all__ = [
    "AllowAllMemoryConsent",
    "Clock",
    "ContextIntent",
    "FakeClock",
    "InMemoryMemoryStore",
    "MemoryConsentDecision",
    "MemoryConsentProvider",
    "MemoryEvent",
    "MemoryFailure",
    "MemoryKind",
    "MemoryMatch",
    "MemoryNotFound",
    "MemoryOperation",
    "MemoryOperationType",
    "MemoryPolicy",
    "MemoryQuery",
    "MemoryRecord",
    "MemoryScopeViolation",
    "MemorySensitivity",
    "MemorySnapshotIdentity",
    "MemorySourceType",
    "MemoryStatus",
    "MemoryStore",
    "MemoryStoreError",
    "MemoryVersionConflict",
    "MemoryWriteDenied",
    "MemoryWriteRequest",
    "SQLiteMemoryStore",
    "StaticMemoryConsentProvider",
    "SystemClock",
    "value_hash",
]
