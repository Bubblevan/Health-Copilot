"""Zero-model DEV analysis of LameR/CRB candidate complementarity."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.r2med_candidate_union import fuse_dual_source_rrf
from eval.r2med_crb_data import (
    DEFAULT_SOURCE_ROOT,
    PARTITIONS,
    SOURCE_MANIFEST_PATH,
    _subset_manifest,
    load_source_manifest,
    sha256_file,
)
from eval.r2med_crb_evaluator import _load_qrels, query_metrics

LAMER_ROOT = Path(r"E:\Health-Copilot-RAG\r2med\rankings\dev")
CRB_ROOT = REPO_ROOT / "runs/rag_r2med_crb/compact_repair/rankings/dev"
OUTPUT_PATH = REPO_ROOT / "runs/rag_r2med_rerank/candidate_analysis.json"
LAMER_FILE = "lamer_mv_mv_best.jsonl"
CRB_FILE = "crb_q_compact_repair_mv.jsonl"
SINGLE_RECALL_REFERENCE = {"lamer_mv": 0.7075657, "compact_crb_q": 0.7105312117503061}


def _load_rankings(path: Path, subset: str) -> dict[str, list[str]]:
    rankings: dict[str, list[str]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("subset") != subset:
                raise ValueError(f"unexpected subset in {path}:{line_number}")
            query_id = row.get("query_id")
            entries = row.get("ranking")
            if not isinstance(query_id, str) or not isinstance(entries, list):
                raise TypeError(f"invalid ranking row in {path}:{line_number}")
            doc_ids = [entry.get("doc_id") for entry in entries]
            if len(doc_ids) != 100 or any(not isinstance(doc_id, str) for doc_id in doc_ids):
                raise ValueError(f"expected 100 document IDs in {path}:{line_number}")
            if len(set(doc_ids)) != len(doc_ids):
                raise ValueError(f"duplicate document ID in {path}:{line_number}")
            if query_id in rankings:
                raise ValueError(f"duplicate query ID in {path}:{line_number}")
            rankings[query_id] = doc_ids
    return rankings


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return fmean(float(row[key]) for row in rows) if rows else 0.0


def _recall_at(ranked_doc_ids: list[str], qrels: dict[str, int], cutoff: int) -> float:
    relevant = {doc_id for doc_id, score in qrels.items() if score > 0}
    return len(set(ranked_doc_ids[:cutoff]) & relevant) / len(relevant) if relevant else 0.0


def analyze_subset(
    subset: str,
    lamer: dict[str, list[str]],
    crb: dict[str, list[str]],
    qrels: dict[str, dict[str, int]],
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    if set(lamer) != set(crb) or set(lamer) != set(qrels):
        raise ValueError(f"query ID mismatch for {subset}")
    per_query: list[dict[str, Any]] = []
    category_docs: Counter[str] = Counter()
    category_queries: Counter[str] = Counter()
    rrf_rankings: dict[str, list[str]] = {}

    for query_id in sorted(qrels):
        lamer_ids, crb_ids = lamer[query_id], crb[query_id]
        lamer_set, crb_set = set(lamer_ids), set(crb_ids)
        union_set = lamer_set | crb_set
        relevant = {doc_id for doc_id, score in qrels[query_id].items() if score > 0}
        found_lamer = relevant & lamer_set
        found_crb = relevant & crb_set
        relevant_categories = {
            "lamer_only": found_lamer - found_crb,
            "crb_only": found_crb - found_lamer,
            "both": found_lamer & found_crb,
            "neither": relevant - union_set,
        }
        for category, docs in relevant_categories.items():
            category_docs[category] += len(docs)
            category_queries[f"queries_with_{category}"] += int(bool(docs))

        candidates = fuse_dual_source_rrf(lamer_ids, crb_ids, k=60)
        fused_ids = [candidate.doc_id for candidate in candidates]
        rrf_rankings[query_id] = fused_ids
        candidate_jaccard = len(lamer_set & crb_set) / len(union_set) if union_set else 1.0
        relevant_found_union = found_lamer | found_crb
        relevant_jaccard = (
            len(found_lamer & found_crb) / len(relevant_found_union)
            if relevant_found_union
            else 0.0
        )
        qrow = {
            "subset": subset,
            "query_id": query_id,
            "lamer_recall@100": len(found_lamer) / len(relevant) if relevant else 0.0,
            "crb_recall@100": len(found_crb) / len(relevant) if relevant else 0.0,
            "pool_union_recall@100": len(relevant_found_union) / len(relevant) if relevant else 0.0,
            "candidate_top100_jaccard": candidate_jaccard,
            "relevant_doc_jaccard": relevant_jaccard,
            "relevant_lamer_only": len(relevant_categories["lamer_only"]),
            "relevant_crb_only": len(relevant_categories["crb_only"]),
            "relevant_both": len(relevant_categories["both"]),
            "relevant_neither": len(relevant_categories["neither"]),
            "pool_candidate_count": len(union_set),
            **query_metrics(fused_ids, qrels[query_id]),
        }
        per_query.append(qrow)

    qcount = len(per_query)
    summary = {
        "query_count": qcount,
        "mean_top100_jaccard": _mean(per_query, "candidate_top100_jaccard"),
        "mean_relevant_doc_jaccard": _mean(per_query, "relevant_doc_jaccard"),
        "relevant_doc_counts": {
            category: category_docs[category]
            for category in ("lamer_only", "crb_only", "both", "neither")
        },
        "queries_with_relevant_by_category": {
            f"queries_with_{category}": category_queries[f"queries_with_{category}"]
            for category in ("lamer_only", "crb_only", "both", "neither")
        },
        "macro_over_queries": {
            "lamer_recall@100": _mean(per_query, "lamer_recall@100"),
            "crb_recall@100": _mean(per_query, "crb_recall@100"),
            "pool_union_recall@100": _mean(per_query, "pool_union_recall@100"),
            "rrf_union_recall@10": _mean(per_query, "recall@10"),
            "rrf_union_recall@20": fmean(
                _recall_at(rrf_rankings[row["query_id"]], qrels[row["query_id"]], 20)
                for row in per_query
            ),
            "rrf_union_recall@50": _mean(per_query, "recall@50"),
            "rrf_union_recall@100": _mean(per_query, "recall@100"),
            "mean_top100_jaccard": _mean(per_query, "candidate_top100_jaccard"),
            "mean_relevant_doc_jaccard": _mean(per_query, "relevant_doc_jaccard"),
        },
        "macro_equal_subset_weight": {
            "rrf_union_recall@10": _mean(per_query, "recall@10"),
            "rrf_union_recall@20": fmean(
                _recall_at(rrf_rankings[row["query_id"]], qrels[row["query_id"]], 20)
                for row in per_query
            ),
            "rrf_union_recall@50": _mean(per_query, "recall@50"),
            "rrf_union_recall@100": _mean(per_query, "recall@100"),
            "pool_union_recall@100": _mean(per_query, "pool_union_recall@100"),
            "mean_top100_jaccard": _mean(per_query, "candidate_top100_jaccard"),
            "mean_relevant_doc_jaccard": _mean(per_query, "relevant_doc_jaccard"),
        },
        "rrf_k": 60,
        "rrf_weights": {"lamer": 1.0, "compact_crb_q": 1.0},
    }
    return summary, {"rankings": rrf_rankings, "per_query": per_query}


def run_analysis(
    *,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    source_manifest_path: Path = SOURCE_MANIFEST_PATH,
) -> dict[str, Any]:
    manifest = load_source_manifest(source_manifest_path)
    by_subset: dict[str, Any] = {}
    all_per_query: list[dict[str, Any]] = []
    lamer_input_hashes: dict[str, str] = {}
    crb_input_hashes: dict[str, str] = {}

    for subset in PARTITIONS["DEV"]:
        lamer_path = LAMER_ROOT / subset / LAMER_FILE
        crb_path = CRB_ROOT / subset / CRB_FILE
        lamer_input_hashes[subset] = sha256_file(lamer_path)
        crb_input_hashes[subset] = sha256_file(crb_path)
        lamer = _load_rankings(lamer_path, subset)
        crb = _load_rankings(crb_path, subset)
        qrels = _load_qrels(
            "DEV", subset, source_root=source_root, source_manifest_path=source_manifest_path
        )
        result, details = analyze_subset(subset, lamer, crb, qrels)
        by_subset[subset] = result
        all_per_query.extend(details["per_query"])

    macro = {
        key: fmean(float(by_subset[subset]["macro_over_queries"][key]) for subset in PARTITIONS["DEV"])
        for key in (
            "rrf_union_recall@10",
            "rrf_union_recall@20",
            "rrf_union_recall@50",
            "rrf_union_recall@100",
            "pool_union_recall@100",
            "mean_top100_jaccard",
            "mean_relevant_doc_jaccard",
        )
    }
    lamer_r100 = fmean(
        by_subset[name]["macro_over_queries"]["lamer_recall@100"] for name in PARTITIONS["DEV"]
    )
    crb_r100 = fmean(
        by_subset[name]["macro_over_queries"]["crb_recall@100"] for name in PARTITIONS["DEV"]
    )
    strongest_single_r100 = max(lamer_r100, crb_r100)
    union_r100 = macro["pool_union_recall@100"]
    delta = union_r100 - strongest_single_r100
    return {
        "schema_version": "r2med-candidate-complementarity-v1",
        "partition": "DEV",
        "query_count": len(all_per_query),
        "test_accessed": False,
        "model_calls": 0,
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "ranking_inputs": {
            "lamer_mv": {"path": str(LAMER_ROOT), "files": lamer_input_hashes},
            "compact_crb_q": {"path": str(CRB_ROOT), "files": crb_input_hashes},
        },
        "qrels_identity": {
            subset: _subset_manifest(manifest, "DEV", subset)["files"]["qrels.jsonl"]["sha256"]
            for subset in PARTITIONS["DEV"]
        },
        "union_definition": (
            "deduplicated union of both source top-100 pools; pool recall is the relevance ceiling; "
            "ranked cutoffs use equal-weight RRF k=60, deterministic source-rank/doc-id tie-breaks"
        ),
        "single_recall_reference_from_frozen_reports": SINGLE_RECALL_REFERENCE,
        "by_subset": by_subset,
        "macro_equal_subset_weight": macro,
        "candidate_pool_union_gate": {
            "lamer_recall@100_recomputed": lamer_r100,
            "crb_recall@100_recomputed": crb_r100,
            "strongest_single_recall@100": strongest_single_r100,
            "pool_union_recall@100": union_r100,
            "delta_vs_strongest_single": delta,
            "required_delta": 0.005,
            "decision": "POSITIVE" if delta > 0.005 else "WEAK",
        },
        "gold_boundary": "qrels were accessed only by the evaluator for post-ranking metric analysis",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--source-manifest", type=Path, default=SOURCE_MANIFEST_PATH)
    args = parser.parse_args()
    report = run_analysis(source_root=args.source_root, source_manifest_path=args.source_manifest)
    serialized = json.dumps(report, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        if args.output.read_text(encoding="utf-8") != serialized:
            raise FileExistsError(f"refusing to overwrite a different analysis artifact: {args.output}")
    else:
        with args.output.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
    gate = report["candidate_pool_union_gate"]
    print(json.dumps({"output": str(args.output), "gate": gate, "macro": report["macro_equal_subset_weight"]}, indent=2))


if __name__ == "__main__":
    main()
