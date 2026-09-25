"""The R2MED sprint's only relevance-label/qrels reader and metric evaluator."""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import fmean
from typing import Any

from eval.r2med_crb_data import SOURCE_MANIFEST_PATH, _subset_manifest, load_source_manifest
from eval.r2med_multiview import FUSION_CONFIGS

METRICS = ("ndcg@10", "mrr@10", "recall@5", "recall@10", "recall@50", "recall@100")
GAR_METHOD_ORDER = ("hyde_mv", "query2doc_mv", "lamer_mv")
GAR_GENERATION_TO_MULTIVIEW = {
    "hyde": "hyde_mv",
    "query2doc": "query2doc_mv",
    "lamer": "lamer_mv",
}
GAR_MULTIVIEW_TO_GENERATION = {
    multiview: generation
    for generation, multiview in GAR_GENERATION_TO_MULTIVIEW.items()
}


def _load_qrels(
    partition: str,
    subset: str,
    *,
    source_root: Path,
    source_manifest_path: Path,
) -> dict[str, dict[str, int]]:
    """This is the sole function in the new sprint code that opens qrels.jsonl."""
    manifest = load_source_manifest(source_manifest_path)
    entry = _subset_manifest(manifest, partition, subset)
    path = source_root / entry["directory"] / "qrels.jsonl"
    identity = entry["files"]["qrels.jsonl"]
    if not path.is_file() or path.stat().st_size != identity["bytes"]:
        raise ValueError(f"qrels missing or wrong size: {partition}/{subset}")
    from eval.r2med_crb_data import sha256_file

    if sha256_file(path) != identity["sha256"]:
        raise ValueError(f"qrels hash mismatch: {partition}/{subset}")
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            query_id, doc_id, relevance = row.get("q_id"), row.get("p_id"), row.get("score")
            if not isinstance(query_id, str) or not isinstance(doc_id, str):
                raise TypeError(f"invalid qrels IDs at {subset}:{line_number}")
            if isinstance(relevance, bool) or not isinstance(relevance, int) or relevance < 0:
                raise ValueError(f"invalid qrels relevance at {subset}:{line_number}")
            if doc_id in qrels[query_id]:
                raise ValueError(f"duplicate qrels pair at {subset}:{line_number}")
            qrels[query_id][doc_id] = relevance
    return dict(qrels)


def query_metrics(ranked_doc_ids: Sequence[str], qrels: Mapping[str, int]) -> dict[str, float]:
    relevant = {str(doc_id): int(score) for doc_id, score in qrels.items() if int(score) > 0}
    if any(score < 0 for score in qrels.values()):
        raise ValueError("relevance scores must be non-negative")
    ids = [str(doc_id) for doc_id in ranked_doc_ids]
    if len(ids) != len(set(ids)):
        raise ValueError("ranking contains duplicate document IDs")
    first_rank = next((i for i, doc_id in enumerate(ids[:10], start=1) if doc_id in relevant), None)
    dcg = sum(
        (2**relevant.get(doc_id, 0) - 1) / math.log2(rank + 1)
        for rank, doc_id in enumerate(ids[:10], start=1)
    )
    ideal = sorted((2**score - 1 for score in relevant.values()), reverse=True)[:10]
    idcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(ideal, start=1))
    result = {
        "ndcg@10": dcg / idcg if idcg else 0.0,
        "mrr@10": 1.0 / first_rank if first_rank else 0.0,
    }
    relevant_ids = set(relevant)
    for cutoff in (5, 10, 50, 100):
        result[f"recall@{cutoff}"] = (
            len(set(ids[:cutoff]) & relevant_ids) / len(relevant_ids) if relevant_ids else 0.0
        )
    return result


def summarize_query_metrics(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty R2MED evaluation")
    by_subset: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_subset[str(row["subset"])].append(row)
    subset_means = {
        subset: {
            "query_count": len(group),
            **{metric: fmean(float(row[metric]) for row in group) for metric in METRICS},
        }
        for subset, group in sorted(by_subset.items())
    }
    return {
        "query_count": len(rows),
        "subset_count": len(subset_means),
        "macro_equal_subset_weight": {
            metric: fmean(values[metric] for values in subset_means.values()) for metric in METRICS
        },
        "by_subset": subset_means,
    }


def evaluate_rankings(
    partition: str,
    ranked_by_subset: Mapping[str, Mapping[str, Sequence[str]]],
    *,
    source_root: Path,
    source_manifest_path: Path = SOURCE_MANIFEST_PATH,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if partition not in {"DEV", "TEST"}:
        raise ValueError("partition must be DEV or TEST")
    metrics_rows: list[dict[str, Any]] = []
    for subset, rankings in ranked_by_subset.items():
        qrels = _load_qrels(
            partition,
            subset,
            source_root=source_root,
            source_manifest_path=source_manifest_path,
        )
        if set(rankings) != set(qrels):
            raise ValueError(f"ranking and qrels query IDs differ in {partition}/{subset}")
        for query_id, ranking in rankings.items():
            metrics_rows.append(
                {"partition": partition, "subset": subset, "query_id": query_id, **query_metrics(ranking, qrels[query_id])}
            )
    return metrics_rows, summarize_query_metrics(metrics_rows)


def select_best_fusion(
    summaries: Mapping[str, Mapping[str, Any]],
) -> tuple[str, Mapping[str, Any]]:
    if not summaries:
        raise ValueError("fusion selection requires at least one evaluated config")
    allowed = {entry["config_id"] for entry in FUSION_CONFIGS}
    if not set(summaries).issubset(allowed):
        raise ValueError("fusion selection contains a config outside the frozen grid")
    order = {entry["config_id"]: index for index, entry in enumerate(FUSION_CONFIGS)}
    config_by_id = {entry["config_id"]: entry for entry in FUSION_CONFIGS}

    def key(item: tuple[str, Mapping[str, Any]]) -> tuple[float, float, float, int, int]:
        config_id, summary = item
        metrics = summary["macro_equal_subset_weight"]
        weights = config_by_id[config_id]["weights"]
        uniform = int(len(set(weights)) == 1)
        return (
            float(metrics["ndcg@10"]),
            float(metrics["mrr@10"]),
            float(metrics["recall@10"]),
            uniform,
            -order[config_id],
        )

    return max(summaries.items(), key=key)


def select_strongest_gar(
    method_summaries: Mapping[str, Mapping[str, Any]],
) -> tuple[str, Mapping[str, Any]]:
    if set(method_summaries) != set(GAR_METHOD_ORDER):
        raise ValueError("strongest GAR selection requires HyDE-MV, Query2Doc-MV, and LameR-MV")
    return max(
        ((method, method_summaries[method]) for method in GAR_METHOD_ORDER),
        key=lambda item: (
            float(item[1]["macro_equal_subset_weight"]["ndcg@10"]),
            float(item[1]["macro_equal_subset_weight"]["mrr@10"]),
            float(item[1]["macro_equal_subset_weight"]["recall@10"]),
            -GAR_METHOD_ORDER.index(item[0]),
        ),
    )


def paired_stratified_bootstrap(
    candidate: Mapping[str, Mapping[str, Any]],
    baseline: Mapping[str, Mapping[str, Any]],
    subset_by_query: Mapping[str, str],
    *,
    metric: str = "ndcg@10",
    resamples: int = 10_000,
    seed: int = 120_026,
) -> dict[str, float | int | str]:
    if metric not in METRICS or resamples != 10_000:
        raise ValueError("the frozen TEST bootstrap uses 10,000 resamples and a supported metric")
    query_ids = set(candidate)
    if not query_ids or query_ids != set(baseline) or query_ids != set(subset_by_query):
        raise ValueError("paired bootstrap requires identical non-empty query IDs and subset mapping")
    strata = {
        subset: sorted(query_id for query_id in query_ids if subset_by_query[query_id] == subset)
        for subset in sorted(set(subset_by_query.values()))
    }
    if not strata or any(not query_ids_in_subset for query_ids_in_subset in strata.values()):
        raise ValueError("paired bootstrap has an empty subset stratum")
    deltas = {
        query_id: float(candidate[query_id][metric]) - float(baseline[query_id][metric])
        for query_id in query_ids
    }
    observed = fmean(fmean(deltas[qid] for qid in ids) for ids in strata.values())
    rng = random.Random(seed)
    draws = [
        fmean(fmean(rng.choices([deltas[qid] for qid in ids], k=len(ids))) for ids in strata.values())
        for _ in range(resamples)
    ]
    draws.sort()
    return {
        "metric": metric,
        "mean_delta": observed,
        "lower_95": draws[int(0.025 * resamples)],
        "upper_95": draws[min(resamples - 1, int(0.975 * resamples))],
        "resamples": resamples,
        "strata": len(strata),
        "subset_names": ",".join(strata),
    }


def dev_success_gate(
    crb_summary: Mapping[str, Any],
    strongest_gar_summary: Mapping[str, Any],
) -> dict[str, Any]:
    crb = crb_summary["macro_equal_subset_weight"]
    gar = strongest_gar_summary["macro_equal_subset_weight"]
    delta = float(crb["ndcg@10"]) - float(gar["ndcg@10"])
    positive_subsets = sum(
        float(crb_summary["by_subset"][subset]["ndcg@10"])
        > float(strongest_gar_summary["by_subset"][subset]["ndcg@10"])
        for subset in crb_summary["by_subset"]
    )
    return {
        "crb_macro_ndcg_at_10": float(crb["ndcg@10"]),
        "strongest_gar_macro_ndcg_at_10": float(gar["ndcg@10"]),
        "delta": delta,
        "positive_subsets": positive_subsets,
        "signal": "POSITIVE" if delta >= 0.005 and positive_subsets >= 2 else "NEGATIVE",
    }
