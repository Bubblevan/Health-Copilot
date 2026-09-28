from __future__ import annotations

import sqlite3

import pytest

from health_ai_copilot.runtime.memory import (
    FakeClock,
    InMemoryMemoryStore,
    MemoryOperation,
    MemoryVersionConflict,
    SQLiteMemoryStore,
)


def test_add_honors_caller_provided_id_in_memory_and_sqlite(tmp_path):
    request = MemoryOperation.add(
        scope_id="scope-a",
        key="raw_turn:session-a:0",
        kind="session_note",
        value={"role": "user", "session_date": "2023/05/01 (Mon) 12:00", "content": "hello"},
        memory_id="m10b-stable-id",
        valid_from="2023-05-01T12:00:00Z",
        source_session_id="session-a",
    )
    in_memory = InMemoryMemoryStore(clock=FakeClock("2023-05-01T12:00:00Z"))
    sqlite_store = SQLiteMemoryStore(
        tmp_path / "memory.sqlite", clock=FakeClock("2023-05-01T12:00:00Z")
    )

    assert in_memory.apply(request, now="2023-05-01T12:00:00Z").memory_id == "m10b-stable-id"
    assert sqlite_store.apply(request, now="2023-05-01T12:00:00Z").memory_id == "m10b-stable-id"
    sqlite_store.close()


def test_add_without_id_keeps_uuid_memory_id_behavior(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite", clock=FakeClock("2023-05-01T12:00:00Z"))
    record = store.apply(
        MemoryOperation.add(scope_id="scope-a", key="note", kind="session_note", value="hello"),
        now="2023-05-01T12:00:00Z",
    )

    assert record.memory_id.startswith("memory-")
    assert len(record.memory_id.removeprefix("memory-")) == 32
    store.close()


def test_duplicate_explicit_id_hits_sqlite_constraint_and_rolls_back_memory(tmp_path):
    path = tmp_path / "memory.sqlite"
    clock = FakeClock("2023-05-01T12:00:00Z")
    writer = SQLiteMemoryStore(path, clock=clock)
    stale_writer = SQLiteMemoryStore(path, clock=clock)
    original = writer.apply(
        MemoryOperation.add(
            scope_id="scope-a", key="first", kind="session_note", value="one", memory_id="shared-id"
        ),
        now=clock.now(),
    )

    with pytest.raises(sqlite3.IntegrityError, match="memory_records.memory_id"):
        stale_writer.apply(
            MemoryOperation.add(
                scope_id="scope-a", key="second", kind="session_note", value="two", memory_id="shared-id"
            ),
            now=clock.now(),
        )

    assert [record.memory_id for record in stale_writer.active_records("scope-a", now=clock.now())] == []
    assert [record.memory_id for record in writer.active_records("scope-a", now=clock.now())] == [
        original.memory_id
    ]
    writer.close()
    stale_writer.close()


def test_duplicate_explicit_id_in_memory_store_fails_without_replacing_record():
    store = InMemoryMemoryStore(clock=FakeClock("2023-05-01T12:00:00Z"))
    original = store.apply(
        MemoryOperation.add(
            scope_id="scope-a", key="first", kind="session_note", value="one", memory_id="shared-id"
        )
    )
    store.apply(
        MemoryOperation.update(scope_id="scope-a", key="first", kind="session_note", value="updated")
    )

    with pytest.raises(MemoryVersionConflict, match="memory ID already exists"):
        store.apply(
            MemoryOperation.add(
                scope_id="scope-a", key="second", kind="session_note", value="two", memory_id="shared-id"
            )
        )

    assert store._records[original.memory_id].value == "one"
    assert [record.key for record in store.active_records("scope-a")] == ["first"]
