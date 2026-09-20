"""Retriever-only metrics for reviewed source-ID evaluation cases."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from statistics import median
from time import perf_counter
from typing import Any


def evaluate_retriever(
    cases: Sequence[Mapping[str, Any]], retriever, *, top_k: int = 5
) -> tuple[dict[str, float | int | None], list[dict[str, Any]]]:
    """Evaluate only cases with expected IDs; blank-gold cases remain diagnostics."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    scored: list[dict[str, Any]] = []
    for case in cases:
        started = perf_counter()
        evidence = retriever.search(case["question"], top_k=top_k)
        latency_ms = (perf_counter() - started) * 1000
        latencies.append(latency_ms)
        expected = tuple(case.get("expected_source_ids", ()))
        ids = [item.source_id for item in evidence]
        row = {
            "case_id": case["id"],
            "category": case.get("category"),
            "challenge_type": case.get("challenge_type"),
            "expected_source_ids": list(expected),
            "retrieved_source_ids": ids,
            "latency_ms": latency_ms,
            "candidate_count": len(evidence),
        }
        if expected:
            expected_set = set(expected)
            rank = next((index for index, source_id in enumerate(ids, 1) if source_id in expected_set), None)
            row["first_expected_rank"] = rank
            row["hit_at_1"] = rank == 1
            row["hit_at_3"] = rank is not None and rank <= 3
            row["hit_at_5"] = rank is not None and rank <= 5
            row["recall_at_1"] = len(set(ids[:1]).intersection(expected_set)) / len(expected_set)
            row["recall_at_3"] = len(set(ids[:3]).intersection(expected_set)) / len(expected_set)
            row["recall_at_5"] = len(set(ids[:5]).intersection(expected_set)) / len(expected_set)
            row["mrr"] = 1 / rank if rank else 0.0
            row["ndcg_at_5"] = _binary_ndcg(ids, expected_set, 5)
            row["ndcg_at_10"] = _binary_ndcg(ids, expected_set, 10)
            scored.append(row)
        row["failure_type"] = _failure_type(row)
        rows.append(row)
    metrics: dict[str, float | int | None] = {
        "case_count": len(cases),
        "scored_case_count": len(scored),
        "hit_at_1": _mean(scored, "hit_at_1"),
        "hit_at_3": _mean(scored, "hit_at_3"),
        "hit_at_5": _mean(scored, "hit_at_5"),
        "recall_at_1": _mean(scored, "recall_at_1"),
        "recall_at_3": _mean(scored, "recall_at_3"),
        "recall_at_5": _mean(scored, "recall_at_5"),
        "mrr": _mean(scored, "mrr"),
        "ndcg_at_5": _mean(scored, "ndcg_at_5"),
        "ndcg_at_10": _mean(scored, "ndcg_at_10"),
        "mean_latency_ms": sum(latencies) / len(latencies) if latencies else None,
        "p50_latency_ms": median(latencies) if latencies else None,
        "p95_latency_ms": _percentile(latencies, 0.95),
        "mean_candidate_count": _mean(rows, "candidate_count"),
    }
    return metrics, rows


def _failure_type(row: Mapping[str, Any]) -> str | None:
    """Apply a transparent, non-LLM retrieval failure taxonomy."""

    expected = row["expected_source_ids"]
    if not expected:
        return "corpus_uncovered"
    rank = row.get("first_expected_rank")
    if rank == 1 and row.get("recall_at_5") == 1:
        return None
    if len(expected) > 1 and row.get("recall_at_5", 0.0) < 1:
        return "multi_source_partial"
    challenge = row.get("challenge_type")
    if challenge == "source_overlap":
        return "source_overlap"
    if challenge == "jurisdiction_context":
        return "jurisdiction_confusion"
    if challenge in {"synonym_paraphrase", "contextual_paraphrase", "mechanism_paraphrase"}:
        return "semantic_miss" if rank is None else "ranking_error"
    return "lexical_miss" if rank is None else "ranking_error"


def _binary_ndcg(ids: Sequence[str], expected: set[str], k: int) -> float:
    dcg = sum(1 / math.log2(rank + 1) for rank, item in enumerate(ids[:k], 1) if item in expected)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(expected), k) + 1))
    return dcg / ideal if ideal else 0.0


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    return sum(float(row[key]) for row in rows) / len(rows) if rows else None


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]
