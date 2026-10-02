"""Structure-aware atomic chunking for canonical Huiyi documents."""

from __future__ import annotations

import hashlib
import json
import re
import statistics
from pathlib import Path
from typing import Any

import jieba

from ..retrieval.tokenizer import tokenize
from .schema import AtomicChunk, jsonl_bytes, sha256_bytes, sha256_text

CHUNKER_VERSION = "huiyi-atomic-heading-paragraph-list-v1"
MAX_CHUNK_CHARS = 1100
MAX_CHUNK_TOKENS = 420
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_LIST = re.compile(r"^(?:[-*+]\s+|\d+[.)、]\s+)(.+?)\s*$")
_SENTENCE = re.compile(r"(?<=[。！？；!?;])")


def _semantic_units(content: str) -> list[tuple[tuple[str, ...], str]]:
    sections: list[tuple[int, str]] = []
    units: list[tuple[tuple[str, ...], str]] = []
    paragraph_lines: list[str] = []

    def flush_paragraph() -> None:
        if paragraph_lines:
            text = " ".join(line.strip() for line in paragraph_lines if line.strip()).strip()
            if text:
                units.append((tuple(label for _, label in sections), text))
            paragraph_lines.clear()

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            flush_paragraph()
            continue
        heading = _HEADING.match(line)
        if heading:
            flush_paragraph()
            level = len(heading.group(1))
            while sections and sections[-1][0] >= level:
                sections.pop()
            sections.append((level, heading.group(2).strip()))
            continue
        list_item = _LIST.match(line)
        if list_item:
            flush_paragraph()
            units.append((tuple(label for _, label in sections), list_item.group(1).strip()))
            continue
        paragraph_lines.append(line)
    flush_paragraph()
    return units


def _split_long_unit(text: str) -> list[str]:
    if len(text) <= MAX_CHUNK_CHARS and len(tokenize(text)) <= MAX_CHUNK_TOKENS:
        return [text]
    sentences = [part.strip() for part in _SENTENCE.split(text) if part.strip()]
    if not sentences:
        sentences = [text]
    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current}{sentence}" if current else sentence
        if current and (len(candidate) > MAX_CHUNK_CHARS or len(tokenize(candidate)) > MAX_CHUNK_TOKENS):
            pieces.append(current)
            current = sentence
        else:
            current = candidate
        if len(current) > MAX_CHUNK_CHARS:
            # Last-resort boundary for an unusually long sentence; no semantics
            # are rephrased and the split remains deterministic by code point.
            while len(current) > MAX_CHUNK_CHARS:
                pieces.append(current[:MAX_CHUNK_CHARS])
                current = current[MAX_CHUNK_CHARS:]
    if current:
        pieces.append(current)
    return pieces


def chunk_documents(data_root: Path) -> dict[str, Any]:
    source_path = data_root / "normalized" / "documents.jsonl"
    docs = [json.loads(line) for line in source_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    chunks: list[AtomicChunk] = []
    for document in sorted(docs, key=lambda row: row["document_id"]):
        ordinal = 0
        for section_path, unit in _semantic_units(document["content"]):
            for piece in _split_long_unit(unit):
                digest = sha256_text(piece)
                chunk_id = "huiyi-chunk-" + hashlib.sha256(
                    f"{document['document_id']}\0{ordinal}\0{digest}".encode()
                ).hexdigest()[:24]
                chunks.append(AtomicChunk(
                    chunk_id=chunk_id,
                    document_id=document["document_id"],
                    source_id=document["source_id"],
                    document_type=document["document_type"],
                    department=document.get("department"),
                    topic=document.get("topic"),
                    title=document["title"],
                    section_path=section_path,
                    text=piece,
                    ordinal=ordinal,
                    token_count=len(tokenize(piece)),
                    content_sha256=digest,
                    review_status=document["review_status"],
                    freshness_class=document["freshness_class"],
                    source_url=document["source_url"],
                ))
                ordinal += 1
    chunks.sort(key=lambda item: (item.document_id, item.ordinal, item.chunk_id))
    chunk_bytes = jsonl_bytes([chunk.to_dict() for chunk in chunks])
    chunks_dir = data_root / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    (chunks_dir / "chunks.jsonl").write_bytes(chunk_bytes)
    char_lengths = [len(item.text) for item in chunks]
    token_lengths = [item.token_count for item in chunks]
    stats = {
        "chunker_version": CHUNKER_VERSION,
        "chunk_count": len(chunks),
        "chunks_jsonl_sha256": sha256_bytes(chunk_bytes),
        "source_documents_sha256": sha256_bytes(source_path.read_bytes()),
        "tokenizer": "jieba",
        "jieba_version": getattr(jieba, "__version__", "unknown"),
        "median_chars": statistics.median(char_lengths) if char_lengths else 0,
        "p95_chars": _percentile(char_lengths, 0.95),
        "median_tokens": statistics.median(token_lengths) if token_lengths else 0,
        "p95_tokens": _percentile(token_lengths, 0.95),
        "max_chunk_chars": max(char_lengths, default=0),
        "max_chunk_tokens": max(token_lengths, default=0),
    }
    (chunks_dir / "manifest.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return stats


def _percentile(values: list[int], quantile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * quantile + 0.5)))
    return ordered[index]
