"""Deterministic raw-turn adapter for the MEM-2A M10-Base diagnostic."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from health_ai_copilot.runtime.builder import default_runtime_profiles
from health_ai_copilot.runtime.context_manager import (
    ContextBudget,
    ContextItemCategory,
    ContextManager,
    DeterministicTokenEstimator,
)
from health_ai_copilot.runtime.memory import (
    MemoryKind,
    MemoryOperation,
    MemoryRecord,
    MemorySensitivity,
    MemorySnapshotIdentity,
    MemorySourceType,
    MemoryStatus,
    MemoryStore,
)

TIMESTAMP_PATTERN = re.compile(
    r"^(?P<year>\d{4})[/-](?P<month>\d{1,2})[/-](?P<day>\d{1,2})"
    r"\s+\((?P<weekday>Mon|Tue|Wed|Thu|Fri|Sat|Sun)\)"
    r"\s+(?P<hour>\d{1,2}):(?P<minute>\d{2})(?::(?P<second>\d{2}))?$"
)
WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
ALLOWED_SOURCE_FIELDS = frozenset(
    {
        "question_id",
        "question",
        "question_date",
        "haystack_session_ids",
        "haystack_dates",
        "haystack_sessions",
    }
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def parse_longmemeval_timestamp(value: str) -> str:
    """Parse the benchmark's English weekday timestamp without locale state."""
    if not isinstance(value, str):
        raise TypeError("LongMemEval timestamp must be a string")
    match = TIMESTAMP_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(f"unsupported LongMemEval timestamp format: {value!r}")
    fields = match.groupdict()
    result = datetime(
        int(fields["year"]),
        int(fields["month"]),
        int(fields["day"]),
        int(fields["hour"]),
        int(fields["minute"]),
        int(fields["second"] or 0),
        tzinfo=UTC,
    )
    if WEEKDAYS[result.weekday()] != fields["weekday"]:
        raise ValueError(f"weekday does not match LongMemEval date: {value!r}")
    return result.isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class RawTurn:
    source_session_id: str
    turn_index: int
    role: str
    session_date: str
    valid_from: str
    content: str

    @property
    def key(self) -> str:
        return f"raw_turn:{self.source_session_id}:{self.turn_index}"


@dataclass(frozen=True)
class MemoryOnlyQuestion:
    question_id: str
    question: str
    question_date: str
    now: str
    turns: tuple[RawTurn, ...]

    @property
    def scope_id(self) -> str:
        return f"longmemeval:{self.question_id}"


def question_from_source_fields(source: Mapping[str, Any]) -> MemoryOnlyQuestion:
    """Read only the allowlisted memory/generation fields, never labels or categories."""
    allowed = {key: source[key] for key in ALLOWED_SOURCE_FIELDS if key in source}
    required = ALLOWED_SOURCE_FIELDS
    missing = required - allowed.keys()
    if missing:
        raise ValueError(f"LongMemEval record is missing required source fields: {sorted(missing)}")

    question_id = allowed["question_id"]
    question = allowed["question"]
    if not isinstance(question_id, str) or not question_id:
        raise ValueError("question_id must be a non-empty string")
    if not isinstance(question, str):
        raise TypeError("question must be a string")
    question_date = allowed["question_date"]
    now = parse_longmemeval_timestamp(question_date)
    session_ids = allowed["haystack_session_ids"]
    session_dates = allowed["haystack_dates"]
    sessions = allowed["haystack_sessions"]
    if not all(isinstance(items, Sequence) and not isinstance(items, (str, bytes)) for items in (session_ids, session_dates, sessions)):
        raise TypeError("LongMemEval haystack fields must be sequences")
    if not (len(session_ids) == len(session_dates) == len(sessions)):
        raise ValueError("LongMemEval session ids, dates, and turn arrays are misaligned")

    turns: list[RawTurn] = []
    for session_id, source_date, session in zip(session_ids, session_dates, sessions, strict=True):
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("source session ids must be non-empty strings")
        if not isinstance(session, Sequence) or isinstance(session, (str, bytes)):
            raise TypeError("each LongMemEval session must be a turn sequence")
        valid_from = parse_longmemeval_timestamp(source_date)
        for turn_index, turn in enumerate(session):
            if not isinstance(turn, Mapping):
                raise TypeError("each LongMemEval turn must be an object")
            role = turn.get("role")
            content = turn.get("content")
            if role not in {"user", "assistant"}:
                continue
            if not isinstance(content, str):
                raise TypeError("user/assistant turn content must be a string")
            turns.append(
                RawTurn(
                    source_session_id=session_id,
                    turn_index=turn_index,
                    role=role,
                    session_date=source_date,
                    valid_from=valid_from,
                    content=content,
                )
            )
    return MemoryOnlyQuestion(question_id, question, question_date, now, tuple(turns))


def _without_gold_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    excluded = {"answer", "answer_session_ids", "question_type", "has_answer"}
    return {key: value for key, value in pairs if key not in excluded}


def iter_json_array(
    path: str | Path, *, chunk_size: int = 1024 * 1024, include_gold: bool = False
) -> Iterator[dict[str, Any]]:
    """Stream a UTF-8 top-level JSON array, retaining at most one decoded row."""
    if chunk_size < 16:
        raise ValueError("chunk_size must be at least 16 bytes")
    decoder = json.JSONDecoder(object_pairs_hook=None if include_gold else _without_gold_fields)
    with Path(path).open("r", encoding="utf-8") as source:
        buffer = ""
        eof = False
        started = False
        finished = False
        while not finished:
            if not eof and (len(buffer) < chunk_size or not started):
                buffer += source.read(chunk_size)
                eof = source.tell() == Path(path).stat().st_size
            buffer = buffer.lstrip()
            if not started:
                if not buffer and eof:
                    raise ValueError("empty JSON input")
                if buffer:
                    if buffer[0] != "[":
                        raise ValueError("LongMemEval source must be a top-level JSON array")
                    buffer = buffer[1:]
                    started = True
                continue
            buffer = buffer.lstrip()
            if buffer.startswith(","):
                buffer = buffer[1:]
                continue
            if buffer.startswith("]"):
                finished = True
                continue
            if not buffer:
                if eof:
                    raise ValueError("unterminated top-level JSON array")
                buffer += source.read(chunk_size)
                eof = source.tell() == Path(path).stat().st_size
                continue
            try:
                value, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                if eof:
                    raise
                buffer += source.read(chunk_size)
                eof = source.tell() == Path(path).stat().st_size
                continue
            if not isinstance(value, dict):
                raise TypeError("LongMemEval array members must be objects")
            yield value
            buffer = buffer[end:]


def operation_for_turn(
    turn: RawTurn, *, question: MemoryOnlyQuestion, dataset_sha256: str
) -> MemoryOperation:
    identity = {
        "dataset_sha256": dataset_sha256,
        "question_id": question.question_id,
        "source_session_id": turn.source_session_id,
        "turn_index": turn.turn_index,
        "role": turn.role,
        "content_sha256": hashlib.sha256(turn.content.encode("utf-8")).hexdigest(),
    }
    digest = sha256_json(identity)
    value = {"role": turn.role, "session_date": turn.session_date, "content": turn.content}
    return MemoryOperation.add(
        scope_id=question.scope_id,
        key=turn.key,
        kind=MemoryKind.SESSION_NOTE,
        value=value,
        source_type=MemorySourceType.SESSION_DERIVED,
        sensitivity=MemorySensitivity.NON_SENSITIVE,
        memory_id=f"m10b-{digest}",
        valid_from=turn.valid_from,
        valid_until=None,
        expires_at=None,
        source_event_ids=(f"longmemeval-turn-{digest}",),
        source_session_id=turn.source_session_id,
    )


def expected_context_settings() -> dict[str, Any]:
    profile = default_runtime_profiles()["m10-memory-bm25-v1"]
    context = profile.config["context_manager"]
    budget = context["budget"]
    expected = {
        "max_estimated_input_tokens": 4096,
        "reserved_system_tokens": 256,
        "reserved_current_turn_tokens": 512,
        "max_memory_tokens": 1024,
        "max_history_tokens": 2048,
        "max_tool_observation_tokens": 1024,
    }
    if budget != expected or context.get("history_window") != 24 or context.get("estimator") != "chars4-cjk1-v1":
        raise RuntimeError("frozen m10-memory-bm25-v1 ContextManager settings changed")
    return {"budget": budget, "history_window": 24, "estimator": "chars4-cjk1-v1"}


def _record_to_operation_summary(operation: MemoryOperation, record: MemoryRecord, index: int) -> dict[str, Any]:
    value = record.value
    return {
        "operation_index": index,
        "operation": "ADD",
        "memory_id": record.memory_id,
        "scope_id": record.scope_id,
        "key": record.key,
        "kind": record.kind.value,
        "value": record.value,
        "value_sha256": record.value_sha256,
        "status": record.status.value,
        "version": record.version,
        "valid_from": record.valid_from,
        "valid_until": record.valid_until,
        "expires_at": record.expires_at,
        "source_event_ids": list(record.source_event_ids),
        "source_session_id": record.source_session_id,
        "turn_index": int(record.key.rsplit(":", 1)[1]),
        "role": value["role"],
        "source_date": value["session_date"],
        "content_sha256": hashlib.sha256(value["content"].encode("utf-8")).hexdigest(),
        "source_type": record.source_type.value,
        "sensitivity": record.sensitivity.value,
        "operation_identity_sha256": sha256_json(operation.to_dict()),
    }


def build_question_state(
    question: MemoryOnlyQuestion,
    *,
    store: MemoryStore,
    dataset_sha256: str,
    reader_token_counter: Callable[[str], int],
    reader_tokenizer_name: str,
) -> dict[str, Any]:
    """Materialize deterministic ADD-only turns, query native M10, and project context."""
    if not re.fullmatch(r"[0-9a-f]{64}", dataset_sha256):
        raise ValueError("dataset_sha256 must be lowercase SHA256")
    expected_context_settings()
    ingestion_started = time.perf_counter()
    operations: list[dict[str, Any]] = []
    inventory_records: list[MemoryRecord] = []
    existing = {
        record.memory_id: record
        for record in store.active_records(question.scope_id, now="9999-12-31T23:59:59Z")
    }
    expected_ids = {
        str(operation_for_turn(turn, question=question, dataset_sha256=dataset_sha256).memory_id)
        for turn in question.turns
    }
    if set(existing) - expected_ids:
        raise RuntimeError("M10-Base scope contains records outside this deterministic replay")
    for index, turn in enumerate(question.turns):
        operation = operation_for_turn(turn, question=question, dataset_sha256=dataset_sha256)
        existing_record = existing.get(str(operation.memory_id))
        if existing_record is None:
            record = store.apply(operation, now=turn.valid_from)
        else:
            expected_value = operation.value
            if (
                existing_record.scope_id != operation.scope_id
                or existing_record.key != operation.key
                or existing_record.kind != operation.kind
                or existing_record.value != expected_value
                or existing_record.valid_from != operation.valid_from
                or existing_record.valid_until is not None
                or existing_record.expires_at is not None
                or existing_record.source_event_ids != operation.source_event_ids
                or existing_record.source_session_id != operation.source_session_id
                or existing_record.source_type != operation.source_type
                or existing_record.sensitivity != operation.sensitivity
                or existing_record.status != MemoryStatus.ACTIVE
                or existing_record.version != 1
            ):
                raise RuntimeError("existing deterministic M10-Base record differs from replay input")
            record = existing_record
        if record is None or record.status != MemoryStatus.ACTIVE or record.version != 1:
            raise RuntimeError("M10-Base ADD did not materialize an active version-1 record")
        operations.append(_record_to_operation_summary(operation, record, index))
        inventory_records.append(record)

    if len({record.memory_id for record in inventory_records}) != len(inventory_records):
        raise RuntimeError("deterministic turn ids collided")
    final_ids = {
        record.memory_id
        for record in store.active_records(question.scope_id, now="9999-12-31T23:59:59Z")
    }
    if final_ids != expected_ids:
        raise RuntimeError("M10-Base replay did not produce the exact expected active inventory")
    inventory = [record.to_dict() for record in inventory_records]
    snapshot = MemorySnapshotIdentity.from_records(
        question.scope_id, inventory_records, session_revision=len(operations)
    ).to_dict()

    ingestion_ms = (time.perf_counter() - ingestion_started) * 1000
    retrieval_started = time.perf_counter()
    # The query contract intentionally exposes no keys, intent, current values, or rewriting.
    from health_ai_copilot.runtime.memory import MemoryQuery

    query = MemoryQuery(scope_id=question.scope_id, text=question.question, now=question.now, top_k=8)
    matches = store.matches(query)
    retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
    projection_started = time.perf_counter()
    manager = ContextManager(
        budget=ContextBudget(**expected_context_settings()["budget"]),
        estimator=DeterministicTokenEstimator(),
        history_window=24,
    )
    plan = manager.build_plan(
        session_id=question.scope_id,
        session_revision=len(operations),
        current_user=question.question,
        history=[],
        memory_records=[match.record for match in matches],
        retrieval_query=question.question,
    )
    projection_ms = (time.perf_counter() - projection_started) * 1000
    match_by_id = {match.record.memory_id: (rank, match) for rank, match in enumerate(matches, 1)}
    projected = [item for item in plan.items if item.category == ContextItemCategory.MEMORY]
    projected_position_by_id = {
        str(item.provenance["memory_id"]): index
        for index, item in enumerate(plan.items, 1)
        if item.category == ContextItemCategory.MEMORY
    }
    selected_ids = set(plan.selected_memory_ids)
    candidate_items = {
        str(item.provenance["memory_id"]): item
        for item in manager.priority_candidates(memory_records=[match.record for match in matches])
        if item.category == ContextItemCategory.MEMORY
    }
    items: list[dict[str, Any]] = []
    retrieval_trace: list[dict[str, Any]] = []
    for projection_rank, item in enumerate(projected, 1):
        memory_id = str(item.provenance["memory_id"])
        native_rank, match = match_by_id[memory_id]
        record = match.record
        text = canonical_json(item.content)
        items.append(
            {
                "text": text,
                "rank": projection_rank,
                "kind": item.category.value,
                "source_session_ids": [record.source_session_id] if record.source_session_id else [],
                "memory_id": memory_id,
                "native_retrieval_rank": native_rank,
                "retrieval_score": match.score,
                "status": record.status.value,
                "version": record.version,
                "source_turn_id": record.key,
                "estimated_tokens": item.estimated_tokens,
            }
        )
    for rank, match in enumerate(matches, 1):
        retrieval_trace.append(
            {
                "rank": rank,
                "memory_id": match.record.memory_id,
                "score": match.score,
                "reasons": list(match.reasons),
                "source_session_id": match.record.source_session_id,
                "source_turn_key": match.record.key,
                "turn_index": int(match.record.key.rsplit(":", 1)[1]),
                "valid_from": match.record.valid_from,
            }
        )

    context_projection = []
    for rank, match in enumerate(matches, 1):
        record = match.record
        candidate = candidate_items[record.memory_id]
        selected = record.memory_id in selected_ids
        context_projection.append(
            {
                "retrieval_rank": rank,
                "retrieval_score": match.score,
                "retrieval_reasons": list(match.reasons),
                "memory_id": record.memory_id,
                "source_session_id": record.source_session_id,
                "turn_index": int(record.key.rsplit(":", 1)[1]),
                "selected": selected,
                "context_plan_position": projected_position_by_id.get(record.memory_id),
                "m10_estimated_tokens": candidate.estimated_tokens,
                "drop_reason": None if selected else "context_manager_budget_or_total_budget",
            }
        )

    serialized_context = "\n\n".join(
        f"[Context item {item['rank']} | {item['kind']}]\n{item['text']}" for item in items
    )
    reader_tokens = int(reader_token_counter(serialized_context)) if serialized_context else 0
    if reader_tokens < 0:
        raise ValueError("reader token count cannot be negative")
    bundle_base = {
        "schema_version": 3,
        "system": "healthcopilot_m10_base_rawturn",
        "question_id": question.question_id,
        "items": [
            {key: value for key, value in item.items() if key != "estimated_tokens"}
            for item in items
        ],
        "serialized_context": serialized_context,
        "context_embedding_tokens": None,
        "context_embedding_tokenizer": "NOT_APPLICABLE_NO_EMBEDDING",
        "context_reader_tokens": reader_tokens,
        "context_reader_tokenizer": reader_tokenizer_name,
        "provenance_available": True,
        "retrieval_latency_ms": None,
        "ingestion_latency_ms": None,
    }
    bundle_sha = sha256_json(bundle_base)
    context_bundle = {**bundle_base, "context_bundle_sha256": bundle_sha}
    plan_payload = plan.identity_payload()

    return {
        "question_id": question.question_id,
        "scope_id": question.scope_id,
        "now": question.now,
        "operation_count": len(operations),
        "memory_operations": operations,
        "memory_inventory": inventory,
        "inventory_sha256": sha256_json(inventory),
        "snapshot_identity": snapshot,
        "retrieval_query": question.question,
        "native_top_k": 8,
        "retrieval_results": retrieval_trace,
        "context_projection": context_projection,
        "context_plan": plan_payload,
        "context_plan_hash": plan.plan_hash,
        "context_items": items,
        "context_bundle": context_bundle,
        "context_bundle_sha256": bundle_sha,
        "context_embedding_tokens": None,
        "context_embedding_tokenizer": "NOT_APPLICABLE_NO_EMBEDDING",
        "context_reader_tokens": reader_tokens,
        "m10_estimated_memory_tokens": plan.memory_tokens,
        "plan_estimated_total_tokens": plan.estimated_tokens,
        "timings_ms": {
            "ingestion": ingestion_ms,
            "retrieval": retrieval_ms,
            "context_manager": projection_ms,
        },
    }
