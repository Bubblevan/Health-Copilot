"""Zero-cost, DEV-only paired analysis of valid CRB generations vs fallback."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import fmean
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.r2med_crb_data import PARTITIONS, load_source_manifest, read_jsonl, sha256_file
from eval.r2med_crb_evaluator import _load_qrels, query_metrics
from eval.r2med_multiview import ranked_ids, weighted_rrf

DEFAULT_STORAGE_ROOT = Path(r"E:\Health-Copilot-RAG")
DEFAULT_SOURCE_ROOT = Path(r"E:\Health-Copilot-E1.2\sources")
SUBSETS = PARTITIONS["DEV"]
VARIANTS = {
    "crb_q": {"config_id": "k20-W0", "rrf_k": 20, "weights": (1, 1, 1, 1)},
    "crb_prf": {"config_id": "k20-W2", "rrf_k": 20, "weights": (1, 2, 1, 2)},
}


def _read_rankings(path: Path, expected_method: str) -> dict[str, list[str]]:
    rows = read_jsonl(path)
    result: dict[str, list[str]] = {}
    for row in rows:
        if row.get("method") != expected_method:
            raise ValueError(f"unexpected ranking method in {path}: {row.get('method')}")
        query_id = str(row["query_id"])
        if query_id in result:
            raise ValueError(f"duplicate query ranking in {path}: {query_id}")
        result[query_id] = ranked_ids(row["ranking"])
    return result


def _read_generations(path: Path, expected_method: str) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("method") != expected_method:
            raise ValueError(f"unexpected generation method in {path}: {row.get('method')}")
        query_id = str(row["query_id"])
        if query_id in result:
            raise ValueError(f"duplicate generation in {path}: {query_id}")
        valid = row.get("valid") is True
        fallback = row.get("fallback_original") is True
        if valid == fallback:
            raise ValueError(f"generation must be exactly valid or original-query fallback: {query_id}")
        result[query_id] = row
    return result


def _mean_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    return {
        metric: fmean(float(row[metric]) for row in rows)
        for metric in ("ndcg@10", "recall@100")
    }


def _analyze_variant(
    *,
    method: str,
    generation_manifest: dict[str, Any],
    storage_root: Path,
    source_root: Path,
    source_manifest_path: Path,
) -> dict[str, Any]:
    config = VARIANTS[method]
    subsets_out: dict[str, Any] = {}
    load_source_manifest(source_manifest_path)
    exact_fallback_matches = 0
    fallback_count = 0
    generation_count = 0
    fallback_truncated_count = 0
    fallback_other_count = 0
    valid_truncated_count = 0
    generation_by_subset: dict[str, dict[str, int]] = {}

    for subset in SUBSETS:
        generation_entry = next(
            entry for entry in generation_manifest["subsets"] if entry["subset"] == subset
        )
        generation_path = Path(generation_entry["artifact"])
        if not generation_path.is_file():
            raise FileNotFoundError(generation_path)
        if sha256_file(generation_path) != generation_entry["artifact_sha256"]:
            raise ValueError(f"generation artifact hash mismatch: {generation_path}")
        generations = _read_generations(generation_path, method)

        ranking_dir = storage_root / "r2med" / "rankings" / "dev" / subset
        bm25 = _read_rankings(ranking_dir / "bm25_original.jsonl", "bm25_original")
        bge = _read_rankings(ranking_dir / "bge_large_original.jsonl", "bge_large_original")
        bm25_bridge = _read_rankings(
            ranking_dir / f"{method}_bm25_bridge.jsonl", f"{method}_bm25_bridge"
        )
        bge_generated = _read_rankings(
            ranking_dir / f"{method}_bge_generated.jsonl", f"{method}_bge_generated"
        )
        actual = _read_rankings(ranking_dir / f"{method}_mv_best.jsonl", f"{method}_mv_best")
        lamer = _read_rankings(ranking_dir / "lamer_mv_mv_best.jsonl", "lamer_mv_mv_best")
        query_ids = set(generations)
        if not query_ids or query_ids != set(bm25) or query_ids != set(bge) or query_ids != set(actual):
            raise ValueError(f"generation/ranking query IDs differ for DEV/{subset}/{method}")
        if not query_ids.issubset(lamer):
            raise ValueError(f"LameR rankings do not cover CRB queries in DEV/{subset}")
        if query_ids != set(bm25_bridge) or query_ids != set(bge_generated):
            raise ValueError(f"CRB channel query IDs differ for DEV/{subset}/{method}")

        qrels = _load_qrels(
            "DEV",
            subset,
            source_root=source_root,
            source_manifest_path=source_manifest_path,
        )
        if query_ids != set(qrels):
            raise ValueError(f"DEV qrels query IDs differ for {subset}")

        rows_by_status: dict[str, list[dict[str, Any]]] = {"valid": [], "fallback": []}
        subset_outcomes = {"valid": 0, "fallback": 0, "fallback_truncated": 0, "valid_truncated": 0}
        fallback_mismatches: list[str] = []
        bm25_channel_match_count = 0
        bge_channel_match_count = 0
        for query_id, generation in generations.items():
            fallback_ranking = ranked_ids(
                weighted_rrf(
                    (bm25[query_id], bm25[query_id], bge[query_id], bge[query_id]),
                    rrf_k=int(config["rrf_k"]),
                    weights=config["weights"],
                    top_k=100,
                )
            )
            is_valid = generation["valid"] is True
            status = "valid" if is_valid else "fallback"
            subset_outcomes[status] += 1
            if generation.get("truncated") is True:
                if is_valid:
                    valid_truncated_count += 1
                    subset_outcomes["valid_truncated"] += 1
                else:
                    fallback_truncated_count += 1
                    subset_outcomes["fallback_truncated"] += 1
            if not is_valid:
                fallback_count += 1
                bm25_matches = bm25_bridge[query_id] == bm25[query_id]
                bge_matches = bge_generated[query_id] == bge[query_id]
                bm25_channel_match_count += int(bm25_matches)
                bge_channel_match_count += int(bge_matches)
                ranking_matches = actual[query_id] == fallback_ranking
                exact_fallback_matches += int(ranking_matches)
                if not ranking_matches:
                    fallback_mismatches.append(query_id)
                if generation.get("truncated") is not True:
                    fallback_other_count += 1
            actual_metrics = query_metrics(actual[query_id], qrels[query_id])
            fallback_metrics = query_metrics(fallback_ranking, qrels[query_id])
            lamer_metrics = query_metrics(lamer[query_id], qrels[query_id])
            row = {
                "subset": subset,
                "query_id": query_id,
                "actual": actual_metrics,
                "fallback": fallback_metrics,
                "lamer": lamer_metrics,
                "delta_vs_fallback": {
                    metric: actual_metrics[metric] - fallback_metrics[metric]
                    for metric in ("ndcg@10", "recall@100")
                },
                "delta_vs_lamer": {
                    metric: actual_metrics[metric] - lamer_metrics[metric]
                    for metric in ("ndcg@10", "recall@100")
                },
            }
            rows_by_status[status].append(row)

        generation_count += len(generations)
        generation_by_subset[subset] = subset_outcomes
        subsets_out.setdefault(subset, {})["fallback_reconstruction_audit"] = {
            "exact_fallback_top100_matches": subset_outcomes["fallback"] - len(fallback_mismatches),
            "fallback_query_count": subset_outcomes["fallback"],
            "bm25_bridge_equals_original_count": bm25_channel_match_count,
            "bge_generated_equals_original_count": bge_channel_match_count,
            "mismatched_query_ids": fallback_mismatches,
        }
        for status, rows in rows_by_status.items():
            subsets_out.setdefault(subset, {})[status] = {
                "query_count": len(rows),
                "crb": _mean_metrics([row["actual"] for row in rows]),
                "query_only_fallback_same_queries": _mean_metrics([row["fallback"] for row in rows]),
                "strongest_gar_lamer_mv_same_queries": _mean_metrics([row["lamer"] for row in rows]),
                "paired_delta_vs_query_only_fallback": _mean_metrics(
                    [row["delta_vs_fallback"] for row in rows]
                ),
                "paired_delta_vs_lamer_mv": _mean_metrics([row["delta_vs_lamer"] for row in rows]),
            }

    valid_subset_deltas = {
        subset: subsets_out[subset]["valid"]["paired_delta_vs_query_only_fallback"]["ndcg@10"]
        for subset in SUBSETS
    }
    valid_subset_recall_deltas = {
        subset: subsets_out[subset]["valid"]["paired_delta_vs_query_only_fallback"]["recall@100"]
        for subset in SUBSETS
    }
    positive_subsets = sum(delta > 0 for delta in valid_subset_deltas.values())
    macro_valid_delta = fmean(valid_subset_deltas.values())
    min_valid_count = min(subsets_out[subset]["valid"]["query_count"] for subset in SUBSETS)
    reconstruction_exact = exact_fallback_matches == fallback_count
    primary_gate_eligible = method == "crb_q" and min_valid_count >= 10
    return {
        "method": method,
        "selected_dev_fusion": config,
        "generation_counts": {
            "total": generation_count,
            "valid": generation_count - fallback_count,
            "fallback": fallback_count,
            "valid_rate": (generation_count - fallback_count) / generation_count,
            "fallback_truncated": fallback_truncated_count,
            "fallback_not_truncated": fallback_other_count,
            "valid_but_truncated": valid_truncated_count,
            "by_subset": generation_by_subset,
        },
        "fallback_reconstruction": {
            "query_count": fallback_count,
            "exact_top100_matches": exact_fallback_matches,
            "exact_match_rate": exact_fallback_matches / fallback_count if fallback_count else 1.0,
            "status": "EXACT" if reconstruction_exact else "MISMATCH",
            "scope_note": (
                "This audit concerns fallback-stratum rows only. The primary valid-stratum paired delta "
                "compares actual CRB rankings to original-query B0/B1 cached rankings under the same RRF config. "
                "Fallback replay mismatches are listed by subset and are not included in the valid-query effect."
            ),
        },
        "by_subset": subsets_out,
        "valid_stratum_paired_delta_vs_fallback": {
            "ndcg@10_by_subset": valid_subset_deltas,
            "recall@100_by_subset": valid_subset_recall_deltas,
            "macro_equal_subset_ndcg@10": macro_valid_delta,
            "macro_equal_subset_recall@100": fmean(valid_subset_recall_deltas.values()),
            "positive_ndcg_subsets": positive_subsets,
            "subset_count": len(SUBSETS),
            "primary_gate_eligible": primary_gate_eligible,
            "signal": (
                "POSITIVE"
                if primary_gate_eligible and macro_valid_delta > 0 and positive_subsets >= 2
                else "NEGATIVE"
                if primary_gate_eligible
                else "SUPPLEMENTARY_ONLY"
            ),
            "signal_rule": "paired valid-query delta > 0 macro and > 0 in at least 2 of 3 DEV subsets",
            "eligibility_note": (
                "Primary screening is CRB-Q only; each DEV subset must have at least 10 valid queries. "
                "CRB-PRF is supplementary. Fallback replay differences are reported separately and do not "
                "enter the valid-stratum paired effect."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-root", type=Path, default=DEFAULT_STORAGE_ROOT)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--source-manifest", type=Path, default=Path("runs/rag_r2med_crb/source_manifest.json"))
    parser.add_argument("--report", type=Path, default=Path("runs/rag_r2med_crb/dev/valid_fallback_diagnostic.json"))
    args = parser.parse_args()

    results: dict[str, Any] = {}
    generation_root = Path("runs/rag_r2med_crb/generation/dev")
    for method in VARIANTS:
        manifest_path = generation_root / method / "generation_manifest.json"
        generation_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if generation_manifest.get("partition") != "DEV" or generation_manifest.get("method") != method:
            raise ValueError(f"expected DEV generation manifest for {method}")
        results[method] = _analyze_variant(
            method=method,
            generation_manifest=generation_manifest,
            storage_root=args.storage_root,
            source_root=args.source_root,
            source_manifest_path=args.source_manifest,
        )

    report = {
        "schema_version": "r2med-crb-valid-fallback-diagnostic-v1",
        "partition": "DEV",
        "test_accessed": False,
        "model_calls": 0,
        "ranking_recomputation": False,
        "purpose": "paired valid/fallback diagnostic before any compact-schema regeneration",
        "metric_scope": "DEV qrels are consumed only by the evaluation layer; no qrels enter generation or ranking",
        "variants": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        method: {
            "valid_rate": result["generation_counts"]["valid_rate"],
            "fallback_reconstruction": result["fallback_reconstruction"],
            "valid_delta": result["valid_stratum_paired_delta_vs_fallback"],
        }
        for method, result in results.items()
    }, indent=2))
    print(f"report: {args.report.resolve()}")


if __name__ == "__main__":
    main()
