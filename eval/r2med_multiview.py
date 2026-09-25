"""Lucene-analyzed BM25, BGE cosine retrieval, and fixed-grid weighted RRF."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

TOP_K = 100
RRF_K_VALUES = (20, 60)
WEIGHT_GRID = (
    ("W0", (1, 1, 1, 1)),
    ("W1", (2, 1, 2, 1)),
    ("W2", (1, 2, 1, 2)),
    ("W3", (2, 2, 1, 1)),
    ("W4", (1, 1, 2, 2)),
)
FUSION_CONFIGS = tuple(
    {"config_id": f"k{k}-{name}", "rrf_k": k, "weights": weights}
    for k in RRF_K_VALUES
    for name, weights in WEIGHT_GRID
)
DEFAULT_JAVA_HOME = Path(r"D:\jdk-21.0.4")


def ensure_java_home() -> Path:
    configured = os.environ.get("JAVA_HOME")
    java_home = Path(configured) if configured else DEFAULT_JAVA_HOME
    java_executable = java_home / "bin" / "java.exe"
    if not java_executable.is_file():
        raise RuntimeError(f"Pyserini requires a JDK; set JAVA_HOME to a valid JDK (checked {java_home})")
    os.environ.setdefault("JAVA_HOME", str(java_home))
    return java_home


@dataclass(frozen=True)
class RankedDocument:
    doc_id: str
    score: float


def collapse_duplicate_document_scores(
    document_ids: Sequence[str], scores: Sequence[float], *, top_k: int
) -> list[RankedDocument]:
    """Reproduce pinned R2MED's score-dict collapse for repeated corpus IDs."""
    if len(document_ids) != len(scores):
        raise ValueError("BM25 document IDs and scores are misaligned")
    score_by_id: dict[str, float] = {}
    for doc_id, score in zip(document_ids, scores, strict=True):
        score_by_id[str(doc_id)] = float(score)
    ordered_ids = sorted(score_by_id, key=lambda doc_id: -score_by_id[doc_id])[:top_k]
    return [RankedDocument(doc_id, score_by_id[doc_id]) for doc_id in ordered_ids]


class LuceneBM25Index:
    """Faithful to the pinned R2MED Pyserini analyzer + Gensim LuceneBM25Model."""

    def __init__(self, documents: Sequence[tuple[str, str]]) -> None:
        ensure_java_home()
        try:
            from gensim.corpora import Dictionary
            from gensim.models import LuceneBM25Model
            from gensim.similarities import SparseMatrixSimilarity
            from pyserini import analysis
        except ImportError as exc:
            raise RuntimeError("R2MED BM25 requires the pinned Pyserini/Gensim runtime") from exc
        if not documents:
            raise ValueError("BM25 corpus must not be empty")
        self.doc_ids = [doc_id for doc_id, _ in documents]
        self.analyzer = analysis.Analyzer(analysis.get_lucene_analyzer())
        tokenized = [self.analyzer.analyze(text) for _, text in documents]
        self.dictionary = Dictionary(tokenized)
        self.model = LuceneBM25Model(dictionary=self.dictionary, k1=0.9, b=0.4)
        corpus = self.model[list(map(self.dictionary.doc2bow, tokenized))]
        self.index = SparseMatrixSimilarity(
            corpus,
            num_docs=len(tokenized),
            num_terms=len(self.dictionary),
            normalize_queries=False,
            normalize_documents=False,
        )

    def search(self, query_text: str, top_k: int = TOP_K) -> list[RankedDocument]:
        tokenized = self.analyzer.analyze(query_text)
        query = self.model[self.dictionary.doc2bow(tokenized)]
        scores = np.asarray(self.index[query], dtype=np.float32).reshape(-1)
        if len(scores) != len(self.doc_ids):
            raise ValueError("BM25 score vector does not match corpus size")
        return collapse_duplicate_document_scores(self.doc_ids, scores, top_k=top_k)


def dense_search(
    query_vector: np.ndarray,
    document_vectors: np.ndarray,
    document_ids: Sequence[str],
    top_k: int = TOP_K,
) -> list[RankedDocument]:
    if document_vectors.ndim != 2 or len(document_vectors) != len(document_ids):
        raise ValueError("dense corpus vectors and IDs are misaligned")
    query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
    docs = np.asarray(document_vectors, dtype=np.float32)
    if docs.shape[1] != len(query):
        raise ValueError("dense query and document vector dimensions differ")
    qnorm = float(np.linalg.norm(query))
    norms = np.linalg.norm(docs, axis=1)
    denominator = norms * qnorm
    scores = np.divide(docs @ query, denominator, out=np.zeros(len(docs), dtype=np.float32), where=denominator > 0)
    indices = np.argsort(-scores, kind="stable")[:top_k]
    return [RankedDocument(str(document_ids[int(index)]), float(scores[index])) for index in indices]


def dense_search_many(
    query_vectors: np.ndarray,
    document_vectors: np.ndarray,
    document_ids: Sequence[str],
    top_k: int = TOP_K,
    *,
    device: str | None = None,
) -> list[list[RankedDocument]]:
    """Batch cosine search while preserving stable corpus-order tie breaking."""
    if query_vectors.ndim != 2 or document_vectors.ndim != 2:
        raise ValueError("dense query and corpus vectors must be matrices")
    if document_vectors.shape[0] != len(document_ids):
        raise ValueError("dense corpus vectors and IDs are misaligned")
    if query_vectors.shape[1] != document_vectors.shape[1]:
        raise ValueError("dense query and corpus vector dimensions differ")
    docs = np.asarray(document_vectors, dtype=np.float32)
    queries = np.asarray(query_vectors, dtype=np.float32)
    if device is not None:
        if device != "cuda":
            raise ValueError("dense batch retrieval only supports the declared CUDA device")
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("CUDA dense retrieval requires PyTorch") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA dense retrieval was requested but CUDA is unavailable")
        query_tensor = torch.tensor(queries, dtype=torch.float32, device="cuda")
        document_tensor = torch.tensor(docs, dtype=torch.float32, device="cuda")
        denominator = torch.linalg.vector_norm(query_tensor, dim=1, keepdim=True) * torch.linalg.vector_norm(
            document_tensor, dim=1
        ).unsqueeze(0)
        scores = torch.where(
            denominator > 0,
            (query_tensor @ document_tensor.T) / denominator.clamp_min(1e-30),
            torch.zeros_like(denominator),
        )
        top_indices = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :top_k]
        top_scores = torch.gather(scores, dim=1, index=top_indices)
        index_rows = top_indices.cpu().tolist()
        score_rows = top_scores.cpu().tolist()
        return [
            [
                RankedDocument(str(document_ids[index]), float(score))
                for index, score in zip(indices, values, strict=True)
            ]
            for indices, values in zip(index_rows, score_rows, strict=True)
        ]

    doc_norms = np.linalg.norm(docs, axis=1)
    query_norms = np.linalg.norm(queries, axis=1)
    denominators = query_norms[:, None] * doc_norms[None, :]
    numerators = queries @ docs.T
    scores = np.divide(
        numerators,
        denominators,
        out=np.zeros_like(numerators, dtype=np.float32),
        where=denominators > 0,
    )
    result: list[list[RankedDocument]] = []
    for row in scores:
        indices = np.argsort(-row, kind="stable")[:top_k]
        result.append(
            [RankedDocument(str(document_ids[int(index)]), float(row[index])) for index in indices]
        )
    return result


def ranked_ids(rows: Sequence[RankedDocument | dict[str, Any] | str]) -> list[str]:
    result: list[str] = []
    for row in rows:
        if isinstance(row, RankedDocument):
            result.append(row.doc_id)
        elif isinstance(row, str):
            result.append(row)
        else:
            result.append(str(row["doc_id"]))
    return result


def weighted_rrf(
    channels: Sequence[Sequence[RankedDocument | dict[str, Any] | str]],
    *,
    rrf_k: int,
    weights: Sequence[int],
    top_k: int = TOP_K,
) -> list[RankedDocument]:
    if len(channels) != 4:
        raise ValueError("multi-view GAR requires exactly four retrieval channels")
    if len(weights) != 4 or any(weight <= 0 for weight in weights):
        raise ValueError("weighted RRF requires four positive channel weights")
    if rrf_k not in RRF_K_VALUES:
        raise ValueError("RRF k is outside the frozen configuration grid")
    scores: dict[str, float] = {}
    first_rank: dict[str, int] = {}
    for channel_index, (channel, weight) in enumerate(zip(channels, weights, strict=True)):
        ids = ranked_ids(channel)
        if len(ids) != len(set(ids)):
            raise ValueError("a retrieval channel contains duplicate document IDs")
        for rank, doc_id in enumerate(ids, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (rrf_k + rank)
            first_rank[doc_id] = min(first_rank.get(doc_id, rank), rank)
    ordered = sorted(scores, key=lambda doc_id: (-scores[doc_id], first_rank[doc_id], doc_id))[:top_k]
    return [RankedDocument(doc_id, scores[doc_id]) for doc_id in ordered]


def make_four_channels(
    *,
    bm25_original: Sequence[RankedDocument],
    bm25_bridge: Sequence[RankedDocument],
    bge_original: Sequence[RankedDocument],
    bge_generated: Sequence[RankedDocument],
) -> tuple[Sequence[RankedDocument], ...]:
    channels = (bm25_original, bm25_bridge, bge_original, bge_generated)
    if any(len(channel) > TOP_K for channel in channels):
        raise ValueError("multi-view retrieval channels must be truncated at top 100")
    return channels


def encode_bge(
    model: Any,
    texts: Sequence[str],
    *,
    batch_size: int = 32,
    show_progress_bar: bool = False,
) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    vectors = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
        normalize_embeddings=False,
        precision="float32",
    )
    return np.asarray(vectors, dtype=np.float32)


def encode_query(model: Any, text: str) -> np.ndarray:
    return encode_bge(model, [text], batch_size=1)[0]
