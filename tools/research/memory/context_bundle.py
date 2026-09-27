"""Deterministic benchmark-side representation of retrieved memory context."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Callable


@dataclass(frozen=True)
class ContextItem:
    text: str
    rank: int
    kind: str
    source_session_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ContextBundle:
    schema_version: int
    system: str
    question_id: str
    items: tuple[ContextItem, ...]
    serialized_context: str
    context_embedding_tokens: int
    context_embedding_tokenizer: str
    context_reader_tokens: int
    context_reader_tokenizer: str
    provenance_available: bool
    retrieval_latency_ms: float | None
    ingestion_latency_ms: float | None
    context_bundle_sha256: str

    def to_dict(self) -> dict:
        value = asdict(self)
        value["items"] = [
            {**asdict(item), "source_session_ids": list(item.source_session_ids)}
            for item in self.items
        ]
        return value


def _canonical_json(value: dict) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _bundle_digest(value: dict) -> str:
    content = {key: item for key, item in value.items() if key != "context_bundle_sha256"}
    return hashlib.sha256(_canonical_json(content)).hexdigest()


def build_context_bundle(
    *,
    system: str,
    question_id: str,
    items: list[dict],
    embedding_token_counter: Callable[[list[str]], int],
    embedding_tokenizer_name: str,
    reader_token_counter: Callable[[list[str]], int],
    reader_tokenizer_name: str,
    provenance_available: bool,
    retrieval_latency_ms: float | None,
    ingestion_latency_ms: float | None,
) -> ContextBundle:
    if not system or not question_id:
        raise ValueError("ContextBundle requires a system and question_id")
    if not isinstance(provenance_available, bool):
        raise TypeError("provenance_available must be boolean")

    normalized_items: list[ContextItem] = []
    for index, item in enumerate(items, 1):
        text = item.get("text")
        kind = item.get("kind")
        rank = item.get("rank", index)
        source_ids = item.get("source_session_ids", [])
        if not isinstance(text, str) or not isinstance(kind, str) or not kind:
            raise ValueError("Every context item needs string text and kind")
        if not isinstance(rank, int) or isinstance(rank, bool) or rank < 1:
            raise ValueError("Context item ranks must be positive integers")
        if not isinstance(source_ids, (list, tuple)):
            raise TypeError("source_session_ids must be a list or tuple")
        normalized_items.append(ContextItem(
            text=text,
            rank=rank,
            kind=kind,
            source_session_ids=tuple(dict.fromkeys(str(value) for value in source_ids if value)),
        ))

    serialized = "\n\n".join(
        f"[Context item {item.rank} | {item.kind}]\n{item.text}"
        for item in normalized_items
    )
    embedding_tokens = int(embedding_token_counter([serialized])) if serialized else 0
    reader_tokens = int(reader_token_counter([serialized])) if serialized else 0
    if embedding_tokens < 0 or reader_tokens < 0:
        raise ValueError("context token counts cannot be negative")
    for value in (retrieval_latency_ms, ingestion_latency_ms):
        if value is not None and value < 0:
            raise ValueError("context latencies cannot be negative")

    base = {
        "schema_version": 2,
        "system": system,
        "question_id": question_id,
        "items": [
            {**asdict(item), "source_session_ids": list(item.source_session_ids)}
            for item in normalized_items
        ],
        "serialized_context": serialized,
        "context_embedding_tokens": embedding_tokens,
        "context_embedding_tokenizer": embedding_tokenizer_name,
        "context_reader_tokens": reader_tokens,
        "context_reader_tokenizer": reader_tokenizer_name,
        "provenance_available": provenance_available,
        "retrieval_latency_ms": retrieval_latency_ms,
        "ingestion_latency_ms": ingestion_latency_ms,
    }
    return ContextBundle(
        schema_version=2,
        system=system,
        question_id=question_id,
        items=tuple(normalized_items),
        serialized_context=serialized,
        context_embedding_tokens=embedding_tokens,
        context_embedding_tokenizer=embedding_tokenizer_name,
        context_reader_tokens=reader_tokens,
        context_reader_tokenizer=reader_tokenizer_name,
        provenance_available=provenance_available,
        retrieval_latency_ms=retrieval_latency_ms,
        ingestion_latency_ms=ingestion_latency_ms,
        context_bundle_sha256=_bundle_digest(base),
    )


def verify_context_bundle(value: dict) -> bool:
    digest = value.get("context_bundle_sha256")
    return isinstance(digest, str) and digest == _bundle_digest(value)
