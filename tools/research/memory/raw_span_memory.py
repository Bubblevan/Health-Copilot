"""Lossless, deterministic raw-text spans for the MEM-2C ablation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from health_ai_copilot.runtime.memory import (
    MemoryKind,
    MemoryOperation,
    MemorySensitivity,
    MemorySourceType,
)
from tools.research.memory.mem2a_m10_base import MemoryOnlyQuestion, RawTurn, sha256_json

SEGMENTER_VERSION = "raw-span-segmenter-v1"
REPRESENTATION_VERSION = "raw-span-v1"


@dataclass(frozen=True)
class RawSpan:
    turn: RawTurn
    span_index: int
    char_start: int
    char_end: int
    content: str

    @property
    def parent_turn_key(self) -> str:
        return self.turn.key

    @property
    def key(self) -> str:
        return f"raw_span:{self.turn.source_session_id}:{self.turn.turn_index}:{self.span_index}"


def _is_ordered_list_marker_period(text: str, block_start: int, block_end: int, index: int) -> bool:
    cursor = block_start
    while cursor < block_end and text[cursor] in " \t":
        cursor += 1
    digit_start = cursor
    while cursor < block_end and text[cursor].isdigit():
        cursor += 1
    return (
        digit_start < cursor
        and cursor == index
        and cursor + 1 < block_end
        and text[cursor] == "."
        and text[cursor + 1].isspace()
    )


def segment_text(text: str) -> list[tuple[int, int, str]]:
    """Partition text on newlines and simple sentence punctuation without rewriting it."""
    if not isinstance(text, str):
        raise TypeError("raw-span input must be text")
    if not text:
        return []

    line_blocks: list[tuple[int, int]] = []
    block_start = 0
    for index, character in enumerate(text):
        if character == "\n":
            line_blocks.append((block_start, index + 1))
            block_start = index + 1
    if block_start < len(text):
        line_blocks.append((block_start, len(text)))

    # Whitespace-only lines join an adjacent textual block instead of becoming records.
    merged_blocks: list[tuple[int, int]] = []
    leading_whitespace_start: int | None = None
    for start, end in line_blocks:
        if text[start:end].isspace():
            if merged_blocks:
                previous_start, _ = merged_blocks[-1]
                merged_blocks[-1] = (previous_start, end)
            elif leading_whitespace_start is None:
                leading_whitespace_start = start
            continue
        merged_blocks.append(
            (leading_whitespace_start if leading_whitespace_start is not None else start, end)
        )
        leading_whitespace_start = None

    if leading_whitespace_start is not None:
        if merged_blocks:
            previous_start, _ = merged_blocks[-1]
            merged_blocks[-1] = (previous_start, len(text))
        else:
            merged_blocks.append((0, len(text)))

    spans: list[tuple[int, int, str]] = []
    for block_start, block_end in merged_blocks:
        cursor = block_start
        index = block_start
        while index < block_end:
            character = text[index]
            followed_by_whitespace_or_end = index + 1 == block_end or text[index + 1].isspace()
            is_list_marker = character == "." and _is_ordered_list_marker_period(
                text, block_start, block_end, index
            )
            if character in ".?!" and followed_by_whitespace_or_end and not is_list_marker:
                end = index + 1
                while end < block_end and text[end].isspace():
                    end += 1
                if end > cursor:
                    spans.append((cursor, end, text[cursor:end]))
                cursor = end
                index = end
                continue
            index += 1
        if cursor < block_end:
            spans.append((cursor, block_end, text[cursor:block_end]))

    if any(not content for _, _, content in spans):
        raise RuntimeError("raw-span segmenter emitted an empty span")
    if "".join(content for _, _, content in spans) != text:
        raise RuntimeError("raw-span segmenter failed exact reconstruction")
    expected_start = 0
    for start, end, content in spans:
        if start != expected_start or end <= start or text[start:end] != content:
            raise RuntimeError("raw-span offsets are not a contiguous exact partition")
        expected_start = end
    if expected_start != len(text):
        raise RuntimeError("raw-span offsets do not cover the source turn")
    return spans


def segment_turn(turn: RawTurn) -> list[RawSpan]:
    return [
        RawSpan(turn, span_index, start, end, content)
        for span_index, (start, end, content) in enumerate(segment_text(turn.content))
    ]


def span_identity(
    span: RawSpan, *, question: MemoryOnlyQuestion, dataset_sha256: str
) -> dict[str, Any]:
    return {
        "dataset_sha256": dataset_sha256,
        "question_id": question.question_id,
        "source_session_id": span.turn.source_session_id,
        "turn_index": span.turn.turn_index,
        "span_index": span.span_index,
        "role": span.turn.role,
        "char_start": span.char_start,
        "char_end": span.char_end,
        "span_content_sha256": hashlib.sha256(span.content.encode("utf-8")).hexdigest(),
    }


def operation_for_span(
    span: RawSpan, *, question: MemoryOnlyQuestion, dataset_sha256: str
) -> MemoryOperation:
    identity = span_identity(span, question=question, dataset_sha256=dataset_sha256)
    digest = sha256_json(identity)
    value = {
        "role": span.turn.role,
        "session_date": span.turn.session_date,
        "content": span.content,
        "source_turn_index": span.turn.turn_index,
        "source_span_index": span.span_index,
    }
    return MemoryOperation.add(
        scope_id=question.scope_id,
        key=span.key,
        kind=MemoryKind.SESSION_NOTE,
        value=value,
        source_type=MemorySourceType.SESSION_DERIVED,
        sensitivity=MemorySensitivity.NON_SENSITIVE,
        memory_id=f"m10c-{digest}",
        valid_from=span.turn.valid_from,
        valid_until=None,
        expires_at=None,
        source_event_ids=(f"longmemeval-raw-span-{digest}",),
        source_session_id=span.turn.source_session_id,
    )


def operation_summary(operation: MemoryOperation, *, span: RawSpan, index: int) -> dict[str, Any]:
    return {
        "operation_index": index,
        "operation": "ADD",
        "memory_id": operation.memory_id,
        "scope_id": operation.scope_id,
        "key": operation.key,
        "kind": operation.kind.value,
        "value": operation.value,
        "value_sha256": hashlib.sha256(
            json.dumps(
                operation.value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest(),
        "valid_from": operation.valid_from,
        "valid_until": operation.valid_until,
        "expires_at": operation.expires_at,
        "source_event_ids": list(operation.source_event_ids),
        "source_session_id": operation.source_session_id,
        "parent_turn_key": span.parent_turn_key,
        "source_turn_index": span.turn.turn_index,
        "source_span_index": span.span_index,
        "char_start": span.char_start,
        "char_end": span.char_end,
        "span_content_sha256": hashlib.sha256(span.content.encode("utf-8")).hexdigest(),
        "source_type": operation.source_type.value,
        "sensitivity": operation.sensitivity.value,
        "operation_identity_sha256": sha256_json(operation.to_dict()),
    }
