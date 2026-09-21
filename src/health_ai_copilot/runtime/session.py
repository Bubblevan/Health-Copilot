"""Persistent-session contracts and deterministic local stores for M10.

The session store is deliberately an append-oriented interaction log.  It is
not an AgentSession (which remains an ephemeral provider-facing transcript)
and it is not a memory store.
"""

from __future__ import annotations

import copy
import json
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

SCHEMA_VERSION = "v1"


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _payload_json(payload: Any) -> str:
    _reject_hidden_reasoning(payload)
    try:
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise TypeError("session payload must be JSON-compatible") from exc


def _reject_hidden_reasoning(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key).lower() in {"cot", "chain_of_thought", "hidden_reasoning", "private_reasoning"}:
                raise ValueError("session payload must not contain hidden reasoning")
            _reject_hidden_reasoning(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _reject_hidden_reasoning(nested)


def payload_sha256(payload: Any) -> str:
    return sha256(_payload_json(payload).encode("utf-8")).hexdigest()


class SessionStatus(StrEnum):
    ACTIVE = "active"
    CLOSED = "closed"


class SessionEventType(StrEnum):
    USER_INPUT = "user_input"
    ASSISTANT_OUTPUT = "assistant_output"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    SESSION_SUMMARY = "session_summary"
    MEMORY_WRITE_REF = "memory_write_ref"
    SESSION_FORK = "session_fork"
    SESSION_CLOSE = "session_close"
    RUN_FAILED = "run_failed"


class SessionStoreError(RuntimeError):
    """Base error for session persistence failures."""


class SessionNotFoundError(SessionStoreError):
    pass


class SessionRevisionConflictError(SessionStoreError):
    def __init__(self, session_id: str, expected: int | None, actual: int) -> None:
        self.session_id = session_id
        self.expected_revision = expected
        self.actual_revision = actual
        super().__init__(
            f"session revision conflict for {session_id}: expected {expected}, current {actual}"
        )


class UnknownSchemaError(SessionStoreError):
    pass


@dataclass(frozen=True)
class SessionRecord:
    session_id: str
    parent_session_id: str | None = None
    fork_revision: int | None = None
    created_at: str = field(default_factory=utc_now_iso)
    current_revision: int = 0
    status: SessionStatus = SessionStatus.ACTIVE
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, str) or not self.session_id.strip():
            raise ValueError("session_id must be a non-empty opaque string")
        if self.parent_session_id == self.session_id:
            raise ValueError("a session cannot fork itself")
        if self.fork_revision is not None and self.fork_revision < 0:
            raise ValueError("fork_revision must be non-negative")
        if self.current_revision < 0:
            raise ValueError("current_revision must be non-negative")
        object.__setattr__(self, "status", SessionStatus(self.status))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "parent_session_id": self.parent_session_id,
            "fork_revision": self.fork_revision,
            "created_at": self.created_at,
            "current_revision": self.current_revision,
            "status": self.status.value,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SessionEvent:
    event_id: str
    session_id: str
    sequence: int
    event_type: SessionEventType
    payload: Any
    payload_sha256: str
    run_id: str | None = None
    created_at: str = field(default_factory=utc_now_iso)

    def __post_init__(self) -> None:
        if not self.event_id.strip() or not self.session_id.strip():
            raise ValueError("event_id and session_id must be non-empty")
        if self.sequence <= 0:
            raise ValueError("event sequence must be positive")
        object.__setattr__(self, "event_type", SessionEventType(self.event_type))
        expected = payload_sha256(self.payload)
        if self.payload_sha256 != expected:
            raise ValueError("session payload hash does not match payload")

    @classmethod
    def create(
        cls,
        session_id: str,
        sequence: int,
        event_type: SessionEventType | str,
        payload: Any,
        *,
        run_id: str | None = None,
        created_at: str | None = None,
    ) -> SessionEvent:
        return cls(
            event_id=f"event-{uuid4().hex}",
            session_id=session_id,
            sequence=sequence,
            event_type=SessionEventType(event_type),
            payload=copy.deepcopy(payload),
            payload_sha256=payload_sha256(payload),
            run_id=run_id,
            created_at=created_at or utc_now_iso(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "session_id": self.session_id,
            "sequence": self.sequence,
            "event_type": self.event_type.value,
            "payload": copy.deepcopy(self.payload),
            "payload_sha256": self.payload_sha256,
            "run_id": self.run_id,
            "created_at": self.created_at,
        }


def _prepare_events(
    session_id: str,
    current_revision: int,
    events: Sequence[Any],
    *,
    run_id: str | None,
) -> list[SessionEvent]:
    """Validate and materialize a batch before mutating either store."""

    prepared: list[SessionEvent] = []
    for offset, spec in enumerate(events, start=1):
        event_run_id = run_id
        if isinstance(spec, SessionEvent):
            event_type = spec.event_type
            payload = spec.payload
            event_run_id = spec.run_id or run_id
            created_at = spec.created_at
        elif isinstance(spec, Mapping):
            if "event_type" not in spec or "payload" not in spec:
                raise TypeError("session batch mapping requires event_type and payload")
            event_type = spec["event_type"]
            payload = spec["payload"]
            event_run_id = spec.get("run_id", run_id)
            created_at = spec.get("created_at")
        elif isinstance(spec, Sequence) and not isinstance(spec, (str, bytes)) and len(spec) in {2, 3}:
            event_type, payload = spec[0], spec[1]
            event_run_id = spec[2] if len(spec) == 3 else run_id
            created_at = None
        else:
            raise TypeError("session batch events must be SessionEvent, mapping, or tuple")
        prepared.append(
            SessionEvent.create(
                session_id,
                current_revision + offset,
                event_type,
                payload,
                run_id=event_run_id,
                created_at=created_at,
            )
        )
    return prepared


@runtime_checkable
class SessionStore(Protocol):
    """Append-oriented persistence boundary for one opaque session scope."""

    schema_version: str

    def create_session(
        self, *, session_id: str | None = None, metadata: Mapping[str, Any] | None = None
    ) -> SessionRecord: ...

    def resume_session(self, session_id: str) -> SessionRecord: ...

    def append(
        self,
        session_id: str,
        event_type: SessionEventType | str,
        payload: Any,
        *,
        expected_revision: int | None = None,
        run_id: str | None = None,
    ) -> SessionEvent: ...

    def append_batch(
        self,
        session_id: str,
        events: Sequence[Any],
        *,
        expected_revision: int | None = None,
        run_id: str | None = None,
    ) -> list[SessionEvent]: ...

    def commit_turn(
        self,
        session_id: str,
        user_event: Any,
        assistant_event: Any | None = None,
        *,
        tool_events: Sequence[Any] = (),
        expected_revision: int | None = None,
        run_id: str | None = None,
    ) -> list[SessionEvent]: ...

    def list_events(self, session_id: str, *, at_revision: int | None = None) -> list[SessionEvent]: ...

    def fork_session(self, parent_session_id: str, at_revision: int) -> SessionRecord: ...


class InMemorySessionStore:
    """Deterministic store used by unit tests and local experiments."""

    schema_version = SCHEMA_VERSION

    def __init__(self) -> None:
        self._sessions: dict[str, SessionRecord] = {}
        self._events: dict[str, list[SessionEvent]] = {}
        self._lock = RLock()

    def create_session(
        self, *, session_id: str | None = None, metadata: Mapping[str, Any] | None = None
    ) -> SessionRecord:
        with self._lock:
            identifier = session_id or f"session-{uuid4().hex}"
            if identifier in self._sessions:
                raise SessionStoreError(f"session already exists: {identifier}")
            record = SessionRecord(identifier, metadata=dict(metadata or {}))
            self._sessions[identifier] = record
            self._events[identifier] = []
            return record

    def resume_session(self, session_id: str) -> SessionRecord:
        with self._lock:
            try:
                return self._sessions[session_id]
            except KeyError as exc:
                raise SessionNotFoundError(session_id) from exc

    get_session = resume_session

    def append(
        self,
        session_id: str,
        event_type: SessionEventType | str,
        payload: Any,
        *,
        expected_revision: int | None = None,
        run_id: str | None = None,
    ) -> SessionEvent:
        return self.append_batch(
            session_id,
            [(event_type, payload)],
            expected_revision=expected_revision,
            run_id=run_id,
        )[0]

    def append_batch(
        self,
        session_id: str,
        events: Sequence[Any],
        *,
        expected_revision: int | None = None,
        run_id: str | None = None,
    ) -> list[SessionEvent]:
        with self._lock:
            current = self.resume_session(session_id)
            if current.status != SessionStatus.ACTIVE:
                raise SessionStoreError(f"session is closed: {session_id}")
            if expected_revision is not None and expected_revision != current.current_revision:
                raise SessionRevisionConflictError(
                    session_id, expected_revision, current.current_revision
                )
            prepared = _prepare_events(
                session_id,
                current.current_revision,
                events,
                run_id=run_id,
            )
            self._events[session_id].extend(prepared)
            self._sessions[session_id] = SessionRecord(
                **{
                    **current.to_dict(),
                    "current_revision": current.current_revision + len(prepared),
                }
            )
            return copy.deepcopy(prepared)

    def commit_turn(
        self,
        session_id: str,
        user_event: Any,
        assistant_event: Any | None = None,
        *,
        tool_events: Sequence[Any] = (),
        expected_revision: int | None = None,
        run_id: str | None = None,
    ) -> list[SessionEvent]:
        events = [user_event, *tool_events]
        if assistant_event is not None:
            events.append(assistant_event)
        return self.append_batch(
            session_id,
            events,
            expected_revision=expected_revision,
            run_id=run_id,
        )

    def list_events(self, session_id: str, *, at_revision: int | None = None) -> list[SessionEvent]:
        with self._lock:
            self.resume_session(session_id)
            events = self._events[session_id]
            if at_revision is not None:
                if at_revision < 0:
                    raise ValueError("at_revision must be non-negative")
                events = events[:at_revision]
            return copy.deepcopy(events)

    events = list_events

    def fork_session(self, parent_session_id: str, at_revision: int) -> SessionRecord:
        with self._lock:
            parent = self.resume_session(parent_session_id)
            if at_revision < 0 or at_revision > parent.current_revision:
                raise SessionStoreError("fork revision is outside the parent session")
            child = self.create_session(
                metadata={"forked_from": parent_session_id, "fork_revision": at_revision}
            )
            child = SessionRecord(
                **{
                    **child.to_dict(),
                    "parent_session_id": parent_session_id,
                    "fork_revision": at_revision,
                    "current_revision": at_revision,
                }
            )
            source_events = self._events[parent_session_id][:at_revision]
            self._events[child.session_id] = [
                SessionEvent(
                    event_id=f"event-{uuid4().hex}",
                    session_id=child.session_id,
                    sequence=event.sequence,
                    event_type=event.event_type,
                    payload=copy.deepcopy(event.payload),
                    payload_sha256=event.payload_sha256,
                    run_id=event.run_id,
                    created_at=event.created_at,
                )
                for event in source_events
            ]
            self._sessions[child.session_id] = child
            return child

    def close_session(self, session_id: str, *, expected_revision: int | None = None) -> SessionRecord:
        with self._lock:
            current = self.resume_session(session_id)
            if expected_revision is not None and expected_revision != current.current_revision:
                raise SessionRevisionConflictError(session_id, expected_revision, current.current_revision)
            closed = SessionRecord(**{**current.to_dict(), "status": SessionStatus.CLOSED.value})
            self._sessions[session_id] = closed
            return closed

    def close(self) -> None:
        return None


class SQLiteSessionStore:
    """Transactional SQLite implementation with a fail-closed schema gate."""

    schema_version = SCHEMA_VERSION

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._initialize()
        self._lock = RLock()

    def _initialize(self) -> None:
        connection = self._connection
        tables = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='m10_schema'"
        ).fetchone()
        if tables is None:
            connection.executescript(
                """
                CREATE TABLE m10_schema (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                INSERT INTO m10_schema(key, value) VALUES ('schema_version', 'v1');
                CREATE TABLE sessions (
                    session_id TEXT PRIMARY KEY,
                    parent_session_id TEXT,
                    fork_revision INTEGER,
                    created_at TEXT NOT NULL,
                    current_revision INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                );
                CREATE TABLE session_events (
                    event_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(session_id),
                    sequence INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    run_id TEXT,
                    created_at TEXT NOT NULL,
                    UNIQUE(session_id, sequence)
                );
                """
            )
            return
        version_row = connection.execute(
            "SELECT value FROM m10_schema WHERE key='schema_version'"
        ).fetchone()
        if version_row is None or version_row[0] != SCHEMA_VERSION:
            raise UnknownSchemaError(
                f"unsupported session schema: {version_row[0] if version_row else 'missing'}"
            )

    def create_session(
        self, *, session_id: str | None = None, metadata: Mapping[str, Any] | None = None
    ) -> SessionRecord:
        identifier = session_id or f"session-{uuid4().hex}"
        record = SessionRecord(identifier, metadata=dict(metadata or {}))
        with self._transaction():
            try:
                self._connection.execute(
                    "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        record.session_id,
                        record.parent_session_id,
                        record.fork_revision,
                        record.created_at,
                        record.current_revision,
                        record.status.value,
                        _payload_json(record.metadata),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise SessionStoreError(f"session already exists: {identifier}") from exc
        return record

    def resume_session(self, session_id: str) -> SessionRecord:
        row = self._connection.execute(
            "SELECT * FROM sessions WHERE session_id=?", (session_id,)
        ).fetchone()
        if row is None:
            raise SessionNotFoundError(session_id)
        return SessionRecord(
            session_id=row["session_id"],
            parent_session_id=row["parent_session_id"],
            fork_revision=row["fork_revision"],
            created_at=row["created_at"],
            current_revision=row["current_revision"],
            status=row["status"],
            metadata=json.loads(row["metadata_json"]),
        )

    get_session = resume_session

    def append(
        self,
        session_id: str,
        event_type: SessionEventType | str,
        payload: Any,
        *,
        expected_revision: int | None = None,
        run_id: str | None = None,
    ) -> SessionEvent:
        return self.append_batch(
            session_id,
            [(event_type, payload)],
            expected_revision=expected_revision,
            run_id=run_id,
        )[0]

    def append_batch(
        self,
        session_id: str,
        events: Sequence[Any],
        *,
        expected_revision: int | None = None,
        run_id: str | None = None,
    ) -> list[SessionEvent]:
        with self._transaction():
            current = self.resume_session(session_id)
            if current.status != SessionStatus.ACTIVE:
                raise SessionStoreError(f"session is closed: {session_id}")
            if expected_revision is not None and expected_revision != current.current_revision:
                raise SessionRevisionConflictError(session_id, expected_revision, current.current_revision)
            prepared = _prepare_events(
                session_id,
                current.current_revision,
                events,
                run_id=run_id,
            )
            for event in prepared:
                self._connection.execute(
                    "INSERT INTO session_events VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        event.event_id,
                        event.session_id,
                        event.sequence,
                        event.event_type.value,
                        _payload_json(event.payload),
                        event.payload_sha256,
                        event.run_id,
                        event.created_at,
                    ),
                )
            self._connection.execute(
                "UPDATE sessions SET current_revision=? WHERE session_id=?",
                (current.current_revision + len(prepared), session_id),
            )
            return prepared

    def commit_turn(
        self,
        session_id: str,
        user_event: Any,
        assistant_event: Any | None = None,
        *,
        tool_events: Sequence[Any] = (),
        expected_revision: int | None = None,
        run_id: str | None = None,
    ) -> list[SessionEvent]:
        events = [user_event, *tool_events]
        if assistant_event is not None:
            events.append(assistant_event)
        return self.append_batch(
            session_id,
            events,
            expected_revision=expected_revision,
            run_id=run_id,
        )

    def list_events(self, session_id: str, *, at_revision: int | None = None) -> list[SessionEvent]:
        self.resume_session(session_id)
        sql = "SELECT * FROM session_events WHERE session_id=?"
        args: list[Any] = [session_id]
        if at_revision is not None:
            if at_revision < 0:
                raise ValueError("at_revision must be non-negative")
            sql += " AND sequence<=?"
            args.append(at_revision)
        sql += " ORDER BY sequence"
        return [self._event_from_row(row) for row in self._connection.execute(sql, args)]

    events = list_events

    def fork_session(self, parent_session_id: str, at_revision: int) -> SessionRecord:
        with self._transaction():
            parent = self.resume_session(parent_session_id)
            if at_revision < 0 or at_revision > parent.current_revision:
                raise SessionStoreError("fork revision is outside the parent session")
            child = SessionRecord(
                session_id=f"session-{uuid4().hex}",
                parent_session_id=parent_session_id,
                fork_revision=at_revision,
                metadata={"forked_from": parent_session_id, "fork_revision": at_revision},
                current_revision=at_revision,
            )
            self._connection.execute(
                "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    child.session_id,
                    child.parent_session_id,
                    child.fork_revision,
                    child.created_at,
                    child.current_revision,
                    child.status.value,
                    _payload_json(child.metadata),
                ),
            )
            rows = self._connection.execute(
                "SELECT * FROM session_events WHERE session_id=? AND sequence<=? ORDER BY sequence",
                (parent_session_id, at_revision),
            ).fetchall()
            for row in rows:
                self._connection.execute(
                    "INSERT INTO session_events VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        f"event-{uuid4().hex}",
                        child.session_id,
                        row["sequence"],
                        row["event_type"],
                        row["payload_json"],
                        row["payload_sha256"],
                        row["run_id"],
                        row["created_at"],
                    ),
                )
            return child

    def close_session(self, session_id: str, *, expected_revision: int | None = None) -> SessionRecord:
        with self._transaction():
            current = self.resume_session(session_id)
            if expected_revision is not None and expected_revision != current.current_revision:
                raise SessionRevisionConflictError(session_id, expected_revision, current.current_revision)
            self._connection.execute(
                "UPDATE sessions SET status=? WHERE session_id=?",
                (SessionStatus.CLOSED.value, session_id),
            )
            return self.resume_session(session_id)

    def _event_from_row(self, row: sqlite3.Row) -> SessionEvent:
        return SessionEvent(
            event_id=row["event_id"],
            session_id=row["session_id"],
            sequence=row["sequence"],
            event_type=row["event_type"],
            payload=json.loads(row["payload_json"]),
            payload_sha256=row["payload_sha256"],
            run_id=row["run_id"],
            created_at=row["created_at"],
        )

    class _Tx:
        def __init__(self, owner: SQLiteSessionStore) -> None:
            self.owner = owner

        def __enter__(self):
            self.owner._lock.acquire()
            self.owner._connection.execute("BEGIN IMMEDIATE")
            return self.owner

        def __exit__(self, exc_type, exc, tb):
            try:
                self.owner._connection.execute("ROLLBACK" if exc_type else "COMMIT")
            finally:
                self.owner._lock.release()
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


__all__ = [
    "SCHEMA_VERSION",
    "InMemorySessionStore",
    "SQLiteSessionStore",
    "SessionEvent",
    "SessionEventType",
    "SessionId",
    "SessionNotFoundError",
    "SessionRecord",
    "SessionRevisionConflictError",
    "SessionStatus",
    "SessionStore",
    "SessionStoreError",
    "UnknownSchemaError",
    "payload_sha256",
]


SessionId = str
