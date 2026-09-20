"""Rank-only reciprocal-rank-fusion hybrid retrieval."""

from collections.abc import Sequence
from dataclasses import replace

from ..contracts import Evidence


class HybridRetriever:
    """Fuse two retrievers with RRF; raw BM25 and cosine scores are never added."""

    def __init__(self, bm25, dense, *, rrf_k: int = 60) -> None:
        if rrf_k < 0:
            raise ValueError("rrf_k must be non-negative")
        self.bm25 = bm25
        self.dense = dense
        self.rrf_k = rrf_k

    def search(self, query: str, top_k: int = 5) -> list[Evidence]:
        if top_k <= 0:
            return []
        ranked_lists = (self.bm25.search(query, top_k=top_k), self.dense.search(query, top_k=top_k))
        by_id: dict[str, Evidence] = {}
        scores: dict[str, float] = {}
        for results in ranked_lists:
            for rank, evidence in enumerate(results, 1):
                by_id.setdefault(evidence.source_id, evidence)
                scores[evidence.source_id] = scores.get(evidence.source_id, 0.0) + 1 / (self.rrf_k + rank)
        return [
            replace(by_id[source_id], score=score)
            for source_id, score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        ]


class Reranker:
    """Protocol-like narrow base class retained for simple optional implementations."""

    def rerank(self, query: str, candidates: Sequence[Evidence], top_k: int) -> list[Evidence]:
        raise NotImplementedError


class FakeReranker(Reranker):
    """Deterministic test reranker ordered by a caller-supplied source-ID score."""

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.calls: list[tuple[str, tuple[str, ...], int]] = []

    def rerank(self, query: str, candidates: Sequence[Evidence], top_k: int) -> list[Evidence]:
        self.calls.append((query, tuple(item.source_id for item in candidates), top_k))
        return [
            replace(item, score=self.scores.get(item.source_id, 0.0))
            for item in sorted(candidates, key=lambda item: (-self.scores.get(item.source_id, 0.0), item.source_id))[:top_k]
        ]


class RerankedRetriever:
    def __init__(self, candidate_retriever, reranker: Reranker, *, candidate_top_k: int = 10) -> None:
        if candidate_top_k <= 0:
            raise ValueError("candidate_top_k must be positive")
        self.candidate_retriever = candidate_retriever
        self.reranker = reranker
        self.candidate_top_k = candidate_top_k

    def search(self, query: str, top_k: int = 5) -> list[Evidence]:
        if top_k <= 0:
            return []
        candidates = self.candidate_retriever.search(query, top_k=self.candidate_top_k)
        return self.reranker.rerank(query, candidates, top_k)
