"""Small provider-independent dense retrieval with explicit index provenance."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from math import sqrt
from pathlib import Path
from typing import Protocol

from ..contracts import Evidence, KnowledgeCard
from .documents import RetrievalDocument, document_from_knowledge_card


class EmbeddingBackend(Protocol):
    identity: str

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        ...

    def embed_query(self, query: str) -> Sequence[float]:
        ...


class DenseBackendUnavailable(RuntimeError):
    """Raised when an optional local embedding dependency is not installed."""


class FakeEmbeddingBackend:
    """Deterministic test backend keyed by exact text; never used as a quality claim."""

    identity = "fake-embedding-v1"

    def __init__(self, vectors: Mapping[str, Sequence[float]]) -> None:
        self.vectors = {key: tuple(value) for key, value in vectors.items()}

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, query: str) -> Sequence[float]:
        return self._vector(query)

    def _vector(self, text: str) -> tuple[float, ...]:
        try:
            return self.vectors[text]
        except KeyError as exc:
            raise ValueError(f"fake embedding is missing text: {text}") from exc


class HashingEmbeddingBackend:
    """Runnable local character n-gram vector backend with no model download.

    This is an auditable dense-vector baseline, not a learned semantic model.
    """

    def __init__(self, dimension: int = 256) -> None:
        if dimension <= 0:
            raise ValueError("hashing embedding dimension must be positive")
        self.dimension = dimension
        self.identity = f"hashing-char-ngram-v1:{dimension}"

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, query: str) -> Sequence[float]:
        return self._embed(query)

    def _embed(self, text: str) -> tuple[float, ...]:
        vector = [0.0] * self.dimension
        padded = f" {text.strip().lower()} "
        grams = [padded[index : index + 2] for index in range(max(0, len(padded) - 1))]
        if not grams:
            raise ValueError("cannot embed empty text")
        for gram in grams:
            slot = int.from_bytes(hashlib.sha256(gram.encode("utf-8")).digest()[:8], "big") % self.dimension
            vector[slot] += 1.0
        return tuple(vector)


class SentenceTransformerEmbeddingBackend:
    """Optional local dense backend; CI never imports or downloads it by default."""

    def __init__(self, model_name: str, *, device: str | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise DenseBackendUnavailable("install the retrieval extra for sentence-transformers") from exc
        self.identity = f"sentence-transformers:{model_name}"
        self._model = SentenceTransformer(model_name, device=device)

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return self._model.encode(list(texts), normalize_embeddings=True).tolist()

    def embed_query(self, query: str) -> Sequence[float]:
        return self._model.encode([query], normalize_embeddings=True)[0].tolist()


@dataclass(frozen=True)
class DenseIndexManifest:
    embedding_identity: str
    embedding_dimension: int
    normalized: bool
    corpus_sha256: str
    knowledge_pack_version: str | None
    document_count: int
    build_commit: str


@dataclass(frozen=True)
class DenseIndex:
    documents: tuple[RetrievalDocument, ...]
    vectors: tuple[tuple[float, ...], ...]
    manifest: DenseIndexManifest

    @classmethod
    def build(
        cls,
        documents: Sequence[RetrievalDocument],
        backend: EmbeddingBackend,
        *,
        knowledge_pack_version: str | None,
        build_commit: str,
    ) -> DenseIndex:
        ordered = tuple(sorted(documents, key=lambda item: item.id))
        vectors = tuple(
            _normalize(vector) for vector in backend.embed_documents([_text(item) for item in ordered])
        )
        dimension = len(vectors[0]) if vectors else 0
        if any(len(vector) != dimension for vector in vectors):
            raise ValueError("dense embeddings must share one dimension")
        return cls(
            ordered,
            vectors,
            DenseIndexManifest(
                embedding_identity=backend.identity,
                embedding_dimension=dimension,
                normalized=True,
                corpus_sha256=_corpus_hash(ordered),
                knowledge_pack_version=knowledge_pack_version,
                document_count=len(ordered),
                build_commit=build_commit,
            ),
        )

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "index_manifest.json").write_text(
            json.dumps(asdict(self.manifest), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        (directory / "vectors.json").write_text(
            json.dumps({"documents": [asdict(item) for item in self.documents], "vectors": self.vectors}, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
            newline="\n",
        )

    @classmethod
    def load(cls, directory: Path, *, expected: DenseIndexManifest) -> DenseIndex:
        manifest = DenseIndexManifest(**json.loads((directory / "index_manifest.json").read_text(encoding="utf-8")))
        if manifest != expected:
            raise ValueError("dense index manifest does not match the active retrieval configuration")
        raw = json.loads((directory / "vectors.json").read_text(encoding="utf-8"))
        return cls(
            tuple(RetrievalDocument(**item) for item in raw["documents"]),
            tuple(tuple(vector) for vector in raw["vectors"]),
            manifest,
        )


class DenseRetriever:
    """In-memory normalized cosine retriever with deterministic ID tie breaking."""

    def __init__(self, index: DenseIndex, backend: EmbeddingBackend) -> None:
        if index.manifest.embedding_identity != backend.identity:
            raise ValueError("dense backend identity does not match index manifest")
        self.index = index
        self.backend = backend

    @classmethod
    def from_knowledge_cards(
        cls,
        cards: Sequence[KnowledgeCard],
        backend: EmbeddingBackend,
        *,
        knowledge_pack_version: str,
        build_commit: str,
    ) -> DenseRetriever:
        index = DenseIndex.build(
            [document_from_knowledge_card(card) for card in cards],
            backend,
            knowledge_pack_version=knowledge_pack_version,
            build_commit=build_commit,
        )
        return cls(index, backend)

    def search(self, query: str, top_k: int = 5) -> list[Evidence]:
        if top_k <= 0:
            return []
        query_vector = _normalize(self.backend.embed_query(query))
        if len(query_vector) != self.index.manifest.embedding_dimension:
            raise ValueError("query embedding dimension does not match index")
        ranked = sorted(
            ((_dot(query_vector, vector), document) for document, vector in zip(self.index.documents, self.index.vectors, strict=True)),
            key=lambda item: (-item[0], item[1].id),
        )
        return [
            Evidence(
                source_id=document.id,
                title=document.title or document.id,
                excerpt=document.text,
                source_url=str(document.metadata.get("source_url", "")),
                score=score,
            )
            for score, document in ranked[:top_k]
        ]


def _text(document: RetrievalDocument) -> str:
    return " ".join(part for part in (document.title or "", document.text) if part)


def _normalize(vector: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(item) for item in vector)
    norm = sqrt(sum(item * item for item in values))
    if norm == 0:
        raise ValueError("dense embeddings must not be zero vectors")
    return tuple(item / norm for item in values)


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _corpus_hash(documents: Sequence[RetrievalDocument]) -> str:
    payload = [asdict(item) for item in documents]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
