"""Offline R2MED retrieval metrics and paired, subset-stratified bootstrap."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from statistics import fmean
from typing import Any

METRICS = ("ndcg@10", "mrr@10", "recall@5", "recall@10")
FIXED_ARMS = ("bm25", "bge_dense", "medcpt_dense")


def query_metrics(ranked_doc_ids: Sequence[str], qrels: Mapping[str, int]) -> dict[str, float]:
    """Compute nDCG using exponential gains plus reciprocal-rank/recall metrics."""
    relevant = {str(doc_id): int(score) for doc_id, score in qrels.items() if int(score) > 0}
    if any(score < 0 for score in qrels.values()):
        raise ValueError("qrels relevance scores must be non-negative")
    if len(set(ranked_doc_ids)) != len(ranked_doc_ids):
        raise ValueError("a ranked result must not contain duplicate document IDs")

    retrieved = [str(doc_id) for doc_id in ranked_doc_ids[:10]]
    gains = [2.0**relevant.get(doc_id, 0) - 1.0 for doc_id in retrieved]
    dcg = sum(gain / math.log2(rank + 2) for rank, gain in enumerate(gains))
    ideal_gains = sorted((2.0**score - 1.0 for score in relevant.values()), reverse=True)[:10]
    ideal_dcg = sum(gain / math.log2(rank + 2) for rank, gain in enumerate(ideal_gains))
    first_relevant = next(
        (rank for rank, doc_id in enumerate(retrieved, start=1) if doc_id in relevant), None
    )
    relevant_count = len(relevant)
    return {
        "ndcg@10": dcg / ideal_dcg if ideal_dcg else 0.0,
        "mrr@10": 1.0 / first_relevant if first_relevant else 0.0,
        "recall@5": (
            len(set(retrieved[:5]) & relevant.keys()) / relevant_count if relevant_count else 0.0
        ),
        "recall@10": (
            len(set(retrieved) & relevant.keys()) / relevant_count if relevant_count else 0.0
        ),
    }


def summarize_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty retrieval result")
    by_subset: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_subset[str(row["subset"])].append(row)
    per_subset = {
        subset: {
            "queries": len(group),
            **{
                metric: fmean(float(row[metric]) for row in group)
                for metric in METRICS
            },
        }
        for subset, group in sorted(by_subset.items())
    }
    macro = {
        metric: fmean(values[metric] for values in per_subset.values())
        for metric in METRICS
    }
    return {
        "query_count": len(rows),
        "subset_count": len(per_subset),
        "macro_equal_subset_weight": macro,
        "by_subset": per_subset,
    }


def paired_macro_bootstrap(
    candidate: Mapping[str, Mapping[str, Any]],
    baseline: Mapping[str, Mapping[str, Any]],
    subset_by_query: Mapping[str, str],
    *,
    metric: str = "ndcg@10",
    resamples: int = 10_000,
    seed: int = 120_026,
) -> dict[str, float | int | str]:
    if metric not in METRICS:
        raise ValueError(f"unsupported metric: {metric}")
    query_ids = set(candidate)
    if query_ids != set(baseline) or query_ids != set(subset_by_query):
        raise ValueError("paired bootstrap requires identical query IDs and subset mappings")
    if not query_ids:
        raise ValueError("paired bootstrap requires at least one query")
    if resamples < 1_000:
        raise ValueError("at least 1,000 bootstrap resamples are required")

    deltas_by_subset: list[list[float]] = []
    for subset in sorted(set(subset_by_query.values())):
        ids = [query_id for query_id in sorted(query_ids) if subset_by_query[query_id] == subset]
        if not ids:
            raise ValueError(f"empty query stratum: {subset}")
        deltas_by_subset.append(
            [
                float(candidate[query_id][metric]) - float(baseline[query_id][metric])
                for query_id in ids
            ]
        )

    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(resamples):
        stratum_means = [fmean(rng.choices(deltas, k=len(deltas))) for deltas in deltas_by_subset]
        draws.append(fmean(stratum_means))
    ordered = sorted(draws)
    observed_subset_differences = [fmean(deltas) for deltas in deltas_by_subset]
    query_differences = [
        float(candidate[qid][metric]) - float(baseline[qid][metric]) for qid in query_ids
    ]
    return {
        "metric": metric,
        "mean_difference": fmean(observed_subset_differences),
        "macro_subset_mean_difference": fmean(observed_subset_differences),
        "micro_query_mean_difference": fmean(query_differences),
        "bootstrap_draw_mean": fmean(draws),
        "lower_95": ordered[int(0.025 * resamples)],
        "upper_95": ordered[min(resamples - 1, int(0.975 * resamples))],
        "resamples": resamples,
        "strata": len(deltas_by_subset),
    }


def select_best_fixed_baseline(dev_summaries: Mapping[str, Mapping[str, Any]]) -> str:
    """Select a fixed comparator on DEV; ties keep the predeclared arm order."""
    missing = set(FIXED_ARMS) - set(dev_summaries)
    if missing:
        raise ValueError(f"missing fixed DEV baselines: {sorted(missing)}")
    return max(
        FIXED_ARMS,
        key=lambda arm: float(dev_summaries[arm]["macro_equal_subset_weight"]["ndcg@10"]),
    )


__all__ = [
    "FIXED_ARMS",
    "METRICS",
    "paired_macro_bootstrap",
    "query_metrics",
    "select_best_fixed_baseline",
    "summarize_metrics",
]
