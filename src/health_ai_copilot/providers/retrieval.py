"""Retrieval provider boundary and adapter for a qualified frozen medical KB."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from math import isfinite
from typing import Any, Protocol


@dataclass(frozen=True)
class CommonMedicalKBManifest:
    corpus_id: str
    source_list: tuple[str, ...]
    use_status: str
    document_hashes: tuple[str, ...]
    index_hash: str
    bm25_config: Mapping[str, Any]
    dense_encoder_version: str
    rrf_config: Mapping[str, Any]
    top_k: int
    status: str = "READY"

    def __post_init__(self) -> None:
        source_list = tuple(self.source_list)
        document_hashes = tuple(self.document_hashes)
        if not isinstance(self.bm25_config, Mapping) or not self.bm25_config:
            raise ValueError("KB manifest requires a BM25 index configuration")
        if not isinstance(self.rrf_config, Mapping) or not self.rrf_config:
            raise ValueError("KB manifest requires an RRF configuration")
        if self.status != "READY":
            raise ValueError("Common Medical KB is not qualified as READY")
        if (
            not self.corpus_id.strip()
            or not source_list
            or any(not item.strip() for item in source_list)
            or not document_hashes
        ):
            raise ValueError("KB manifest needs corpus, sources, and document hashes")
        if len(source_list) != len(set(source_list)) or len(document_hashes) != len(set(document_hashes)):
            raise ValueError("KB manifest sources and document hashes must be unique")
        hashes = (*document_hashes, self.index_hash)
        if any(re.fullmatch(r"[a-fA-F0-9]{64}", value or "") is None for value in hashes):
            raise ValueError("KB document and index hashes must be SHA-256 hex digests")
        if (
            not self.use_status.strip()
            or not self.dense_encoder_version.strip()
            or isinstance(self.top_k, bool)
            or not isinstance(self.top_k, int)
            or self.top_k <= 0
        ):
            raise ValueError("KB manifest is missing a qualification field")
        object.__setattr__(self, "source_list", source_list)
        object.__setattr__(self, "document_hashes", document_hashes)


@dataclass(frozen=True)
class RetrievedEvidence:
    evidence_id: str
    source_id: str
    source: str
    excerpt: str
    score: float | None = None

    def __post_init__(self) -> None:
        if not self.evidence_id.strip() or not self.source_id.strip():
            raise ValueError("retrieved evidence requires stable evidence and source IDs")
        if not isinstance(self.source, str) or not isinstance(self.excerpt, str):
            raise TypeError("retrieved evidence source and excerpt must be strings")
        if self.score is not None and not isfinite(self.score):
            raise ValueError("retrieval score must be finite or None")


@dataclass(frozen=True)
class RetrievalResult:
    evidence: tuple[RetrievedEvidence, ...]
    retrieval_trace: tuple[Mapping[str, Any], ...] = ()
    cost: Mapping[str, int | float] = field(default_factory=dict)
    corpus_id: str = ""
    index_hash: str = ""
    evidence_sha256: str = ""

    def __post_init__(self) -> None:
        evidence = tuple(self.evidence)
        ids = [item.evidence_id for item in evidence]
        if len(ids) != len(set(ids)):
            raise ValueError("retrieval evidence IDs must be unique")
        payload = [
            {
                "evidence_id": item.evidence_id,
                "source_id": item.source_id,
                "source": item.source,
                "excerpt": item.excerpt,
                "score": item.score,
            }
            for item in evidence
        ]
        fingerprint = sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            .encode("utf-8")
        ).hexdigest()
        if self.evidence_sha256 and self.evidence_sha256 != fingerprint:
            raise ValueError("evidence_sha256 does not match the returned evidence bytes")
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "evidence_sha256", fingerprint)


class RuntimeContext(Protocol):
    request_id: str
    subject_id: str | None
    as_of_time: Any


class RetrievalProvider(Protocol):
    async def retrieve(self, *, query: str, context: RuntimeContext) -> RetrievalResult:
        ...


class SearchBackend(Protocol):
    def search(self, query: str, top_k: int = 5) -> Sequence[Any]:
        ...


class FrozenMedicalRAGProvider:
    """Wrap a frozen retriever; it never creates an index or implies a corpus exists."""

    def __init__(
        self,
        retriever: SearchBackend,
        manifest: CommonMedicalKBManifest,
        *,
        top_k: int | None = None,
    ) -> None:
        self.retriever = retriever
        self.manifest = manifest
        self.top_k = manifest.top_k if top_k is None else top_k
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or self.top_k <= 0:
            raise ValueError("top_k must be positive")

    async def retrieve(self, *, query: str, context: RuntimeContext) -> RetrievalResult:
        raw = await asyncio.to_thread(self.retriever.search, query, self.top_k)
        rows = tuple(
            RetrievedEvidence(
                evidence_id=str(getattr(item, "evidence_id", None)
                                or getattr(item, "source_id", None)
                                or getattr(item, "id", "")),
                source_id=str(getattr(item, "source_id", None)
                              or getattr(item, "id", "")),
                source=str(getattr(item, "source_url", None)
                           or getattr(item, "source", None)
                           or getattr(item, "title", "")),
                excerpt=str(getattr(item, "excerpt", None)
                            or getattr(item, "content", None)
                            or getattr(item, "text", "")),
                score=(float(item.score) if getattr(item, "score", None) is not None else None),
            )
            for item in raw
        )
        if any(not row.evidence_id or not row.source_id for row in rows):
            raise ValueError("retriever returned evidence without stable IDs")
        return RetrievalResult(
            evidence=rows,
            retrieval_trace=({"event": "frozen_retriever", "top_k": self.top_k},),
            cost={"retrieval_calls": 1, "documents": len(rows)},
            corpus_id=self.manifest.corpus_id,
            index_hash=self.manifest.index_hash,
        )
