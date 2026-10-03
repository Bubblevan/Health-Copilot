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

CHUNKER_VERSION = "huiyi-atomic-heading-paragraph-list-doctor-profile-v2"
MAX_CHUNK_CHARS = 1100
MAX_CHUNK_TOKENS = 420
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_LIST = re.compile(r"^(?:[-*+]\s+|\d+[.)、]\s+)(.+?)\s*$")
_SENTENCE = re.compile(r"(?<=[。！？；!?;])")
_PROFILE_SPECIALTY = re.compile(r"主要擅长|专业特长|业务专长|擅长领域|擅长")
_PROFILE_EXPERIENCE = re.compile(
    r"从事|曾在|先后|进修|毕业|工作(?:近|于|以来|超过|\d)|任职|担任|委员|会员|"
    r"获.*奖|发表|参与|经历|个人简介|工作简历"
)


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


def _is_profile_specialty(unit: tuple[tuple[str, ...], str]) -> bool:
    section_path, text = unit
    return bool(_PROFILE_SPECIALTY.search(" ".join((*section_path, text))))


def _pack_profile_units(units: list[tuple[tuple[str, ...], str]], label: str) -> list[tuple[tuple[str, ...], str]]:
    packed: list[tuple[tuple[str, ...], str]] = []
    current: list[str] = []
    for _, text in units:
        for piece in _split_long_unit(text):
            candidate = "\n".join((*current, piece))
            if current and (len(candidate) > MAX_CHUNK_CHARS or len(tokenize(candidate)) > MAX_CHUNK_TOKENS):
                packed.append(((label,), "\n".join(current)))
                current = [piece]
            else:
                current.append(piece)
    if current:
        packed.append(((label,), "\n".join(current)))
    return packed


def _doctor_profile_groups(
    units: list[tuple[tuple[str, ...], str]],
) -> list[tuple[tuple[str, ...], str]]:
    """Group short profile facts without fragmenting identity and appointments."""
    if not units:
        return []
    specialty_start = next((index for index, unit in enumerate(units) if _is_profile_specialty(unit)), len(units))
    before_specialty = units[:specialty_start]
    specialty = units[specialty_start:] if specialty_start < len(units) else []

    experience_start = next(
        (
            index
            for index, (_, text) in enumerate(before_specialty)
            if _PROFILE_EXPERIENCE.search(text) or len(text) >= 180
        ),
        len(before_specialty),
    )
    identity = before_specialty[:experience_start]
    experience = before_specialty[experience_start:]

    # A concise profile with no biography marker is still kept together; when
    # it is longer, split after the first few identity facts for readability.
    if not experience and len(identity) > 1:
        identity_size = 0
        split = 0
        for index, (_, text) in enumerate(identity):
            if split and identity_size + len(text) > 260:
                break
            identity_size += len(text)
            split = index + 1
        if split < len(identity):
            experience = identity[split:]
            identity = identity[:split]

    groups: list[tuple[tuple[str, ...], str]] = []
    groups.extend(_pack_profile_units(identity, "身份与职称"))
    groups.extend(_pack_profile_units(experience, "任职与经历"))
    groups.extend(_pack_profile_units(specialty, "主要擅长"))
    return groups


def chunk_documents(data_root: Path) -> dict[str, Any]:
    source_path = data_root / "normalized" / "documents.jsonl"
    docs = [json.loads(line) for line in source_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    chunks: list[AtomicChunk] = []
    for document in sorted(docs, key=lambda row: row["document_id"]):
        ordinal = 0
        units = _semantic_units(document["content"])
        if document["document_type"] == "doctor_profile":
            chunk_units = _doctor_profile_groups(units)
        else:
            chunk_units = units
        for section_path, unit in chunk_units:
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
                    primary_topic=document.get("primary_topic"),
                    topics=tuple(document.get("topics", [])),
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
