"""Deterministic, provenance-preserving Huiyi corpus contracts."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

from . import CORPUS_SCHEMA_VERSION

DocumentType = Literal[
    "hospital_info",
    "department",
    "doctor_profile",
    "patient_education",
    "perioperative_instruction",
    "faq",
]
ReviewStatus = Literal[
    "UNREVIEWED",
    "AUTO_ACCEPTED_PUBLIC_INFO",
    "NEEDS_HUMAN_REVIEW",
    "APPROVED",
    "REJECTED",
]
FreshnessClass = Literal[
    "STATIC_PROFILE",
    "STATIC_EDUCATION",
    "POTENTIALLY_STALE",
    "DYNAMIC",
]


def canonical_url(url: str) -> str:
    """Normalize URL identity without changing its host or path semantics."""
    parts = urlsplit(url.strip())
    if parts.scheme.lower() not in {"http", "https"} or not parts.hostname:
        raise ValueError(f"not an absolute HTTP URL: {url!r}")
    host = parts.hostname.lower()
    if parts.port and parts.port not in {80, 443}:
        host = f"{host}:{parts.port}"
    path = parts.path or "/"
    return urlunsplit((parts.scheme.lower(), host, path, parts.query, ""))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(payload)


def source_id_for_url(url: str) -> str:
    """Stable semantic source ID with a URL-derived collision-resistant suffix."""
    normalized = canonical_url(url)
    path = urlsplit(normalized).path.strip("/")
    segments = [segment for segment in re.split(r"[^A-Za-z0-9]+", path.lower()) if segment]
    slug = "-".join(segments[-3:]) or "home"
    slug = slug[:48].strip("-") or "page"
    suffix = sha256_text(normalized)[:10]
    return f"huiyi-official-{slug}-{suffix}"


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    url: str
    domain: str
    source_family: str
    expected_type: str
    ingestion_policy: str
    enabled: bool = True
    page_role: str = "document"
    title: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SourceSpec:
        source_url = canonical_url(str(value["url"]))
        source_id = str(value.get("source_id") or value.get("id") or source_id_for_url(source_url))
        return cls(
            source_id=source_id,
            url=source_url,
            domain=str(value.get("domain") or urlsplit(source_url).hostname or ""),
            source_family=str(value.get("source_family", "huiyi_official")),
            expected_type=str(value["expected_type"]),
            ingestion_policy=str(value.get("ingestion_policy", "canonical")),
            enabled=bool(value.get("enabled", True)),
            page_role=str(value.get("page_role", "document")),
            title=str(value["title"]) if value.get("title") else None,
        )


@dataclass(frozen=True)
class CanonicalDocument:
    document_id: str
    source_id: str
    source_family: str
    source_url: str
    document_type: str
    department: str | None
    topic: str | None
    title: str
    content: str
    published_at: str | None
    effective_from: str | None
    effective_until: str | None
    fetched_at: str
    review_status: str
    ingestion_policy: str
    raw_sha256: str
    content_sha256: str
    schema_version: str = CORPUS_SCHEMA_VERSION
    freshness_class: str = "STATIC_EDUCATION"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AtomicChunk:
    chunk_id: str
    document_id: str
    source_id: str
    document_type: str
    department: str | None
    topic: str | None
    title: str
    section_path: tuple[str, ...]
    text: str
    ordinal: int
    token_count: int
    content_sha256: str
    review_status: str
    freshness_class: str
    source_url: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["section_path"] = list(self.section_path)
        return value


def jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(
        (
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        for row in rows
    )
