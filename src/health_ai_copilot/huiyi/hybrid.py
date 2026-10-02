"""Minimal reciprocal-rank fusion over BM25 and dense candidate lists."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

RRF_K = 60
BM25_WEIGHT = 1.0
DENSE_WEIGHT = 1.0


def reciprocal_rank_fusion(
    bm25_rows: list[dict[str, Any]],
    dense_rows: list[dict[str, Any]],
    *,
    top_k: int = 5,
    rrf_k: int = RRF_K,
    bm25_weight: float = BM25_WEIGHT,
    dense_weight: float = DENSE_WEIGHT,
) -> list[dict[str, Any]]:
    scores: dict[str, float] = {}
    rows: dict[str, dict[str, Any]] = {}
    for weight, hits in ((bm25_weight, bm25_rows), (dense_weight, dense_rows)):
        for rank, row in enumerate(hits, start=1):
            chunk_id = row["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (rrf_k + rank)
            rows[chunk_id] = row
    ranked = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))
    return [{**rows[chunk_id], "score": scores[chunk_id]} for chunk_id in ranked[:top_k]]


def hybrid_search(
    query: str,
    bm25_search: Callable[[str, int], list[dict[str, Any]]],
    dense_search: Callable[[str, int], list[dict[str, Any]]],
    *,
    top_k: int = 5,
    candidate_k: int = 20,
) -> list[dict[str, Any]]:
    return reciprocal_rank_fusion(
        bm25_search(query, candidate_k),
        dense_search(query, candidate_k),
        top_k=top_k,
    )
