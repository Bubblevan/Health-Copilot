"""Retrieve and evaluate the one compact-schema CRB variant on DEV only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.r2med_crb import BGE_QUERY_PREFIX, crb_lexical_text
from eval.r2med_crb_data import load_partition_inputs, load_source_manifest, read_jsonl, sha256_file
from eval.r2med_crb_evaluator import dev_success_gate, evaluate_rankings
from eval.r2med_multiview import LuceneBM25Index, dense_search_many, encode_bge, weighted_rrf
from tools.generate_r2med_crb_compact_repair import GENERATION_ROOT, MANIFEST_PATH
from tools.run_r2med_baselines import _ranking_rows
from tools.run_r2med_crb_dev import (
    DEFAULT_BGE_ROOT,
    _load_or_encode_corpus,
    _read_ranking,
)
from tools.run_r2med_crb_dev import RANKING_ROOT as CACHED_RANKING_ROOT

DEFAULT_SOURCE_ROOT = Path(r"E:\Health-Copilot-E1.2\sources")
ROOT_OUTPUT = ROOT / "runs/rag_r2med_crb/compact_repair"
DEV_REPORT_PATH = ROOT / "runs/rag_r2med_crb/dev/dev_report.json"
RRF_CONFIG = {"config_id": "k20-W0", "rrf_k": 20, "weights": (1, 1, 1, 1)}
SUBSETS = ("PMC-Treatment", "PMC-Clinical", "IIYi-Clinical")


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_jsonl_once(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite compact-repair artifact: {path}")
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return sha256_file(path)


def _read_ranking_rows(path: Path, subset: str, method: str) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for row in read_jsonl(path):
        if row.get("subset") != subset or row.get("method") != method:
            raise ValueError(f"ranking identity mismatch in {path}")
        query_id = str(row["query_id"])
        ranking = row.get("ranking")
        if query_id in result or not isinstance(ranking, list) or len(ranking) > 100:
            raise ValueError(f"invalid ranking row in {path}: {query_id}")
        result[query_id] = ranking
    return result


def _load_repair_generation(subset: str, expected_ids: set[str]) -> dict[str, dict[str, Any]]:
    path = GENERATION_ROOT / f"{subset}.jsonl"
    rows = read_jsonl(path)
    result = {str(row["query_id"]): row for row in rows}
    if len(result) != len(rows) or set(result) != expected_ids:
        raise ValueError(f"compact generation query IDs differ from DEV/{subset}")
    for row in rows:
        if row.get("method") != "crb_q_compact_repair":
            raise ValueError(f"unexpected compact-generation method in {subset}")
        structured = row.get("structured")
        required = {"canonical_query", "key_concepts", "disambiguating_terms", "pseudo_evidence"}
        if not isinstance(structured, dict) or set(structured) != required:
            raise ValueError(f"invalid normalized compact generation for {row['query_id']}")
        if row.get("valid") is not (row.get("fallback_original") is False):
            raise ValueError(f"compact generation validity markers disagree for {row['query_id']}")
    return result


def _generation_manifest_identity() -> tuple[dict[str, Any], str]:
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError("compact generation must finish before DEV retrieval/evaluation")
    raw = MANIFEST_PATH.read_bytes()
    manifest = json.loads(raw)
    if manifest.get("partition") != "DEV" or manifest.get("method") != "crb_q_compact_repair":
        raise ValueError("compact repair generation manifest must be DEV-only")
    if manifest.get("test_accessed") is not False or manifest.get("attempted_query_count") != 393:
        raise ValueError("compact generation manifest failed the one-pass DEV scope check")
    for entry in manifest["subsets"]:
        path = Path(entry["artifact"])
        if not path.is_file() or sha256_file(path) != entry["artifact_sha256"]:
            raise ValueError(f"compact generation artifact hash mismatch: {path}")
    return manifest, _hash_bytes(raw)


def run_dev(source_root: Path) -> dict[str, Any]:
    generation_manifest, generation_manifest_sha = _generation_manifest_identity()
    load_source_manifest()
    source_sha = _hash_bytes((ROOT / "runs/rag_r2med_crb/source_manifest.json").read_bytes())
    if generation_manifest.get("dataset_source_manifest_sha256") != source_sha:
        raise ValueError("compact generation source manifest identity is stale")
    diagnostic = json.loads(
        (ROOT / "runs/rag_r2med_crb/dev/valid_fallback_diagnostic.json").read_text(encoding="utf-8")
    )
    if diagnostic["variants"]["crb_q"]["valid_stratum_paired_delta_vs_fallback"]["signal"] != "POSITIVE":
        raise ValueError("DEV compact repair evaluation requires the positive preregistered diagnostic signal")

    parent_report_path = DEV_REPORT_PATH
    parent_report = json.loads(parent_report_path.read_text(encoding="utf-8"))
    selected = parent_report["multi_view"]["crb_q"]["best_config"]
    if selected["config_id"] != RRF_CONFIG["config_id"]:
        raise ValueError("unexpected original CRB-Q DEV-selected fusion; compact repair will not retune it")
    strong_gar = parent_report["strongest_cost_matched_gar"]["summary"]
    original_crb = parent_report["multi_view"]["crb_q"]["best_summary"]

    inputs = load_partition_inputs("DEV", source_root=source_root)
    if tuple(subset.name for subset in inputs) != SUBSETS:
        raise ValueError("compact repair DEV subset order differs from the frozen split")

    # Fail closed if the precomputed corpus caches are missing: this runner may read E: but never writes there.
    for subset in SUBSETS:
        cache_base = Path(r"E:\Health-Copilot-RAG\r2med\bge-large")
        if not (cache_base / f"{subset}.npy").is_file() or not (cache_base / f"{subset}.json").is_file():
            raise FileNotFoundError(f"expected read-only BGE corpus cache is missing for {subset}")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(DEFAULT_BGE_ROOT), device="cuda")
    model.max_seq_length = 512
    bge_sha = parent_report["verified_models"]["bge_large"]["weights_sha256"]
    ranking_ids_by_subset: dict[str, dict[str, list[str]]] = {}
    channel_hashes: dict[str, dict[str, str]] = {}
    final_hashes: dict[str, str] = {}
    generation_counts: dict[str, dict[str, int]] = {}

    for subset in inputs:
        name = subset.name
        query_ids = [query.query_id for query in subset.queries]
        expected_ids = set(query_ids)
        generations = _load_repair_generation(name, expected_ids)
        generation_counts[name] = {
            "query_count": len(generations),
            "valid_count": sum(row["valid"] is True for row in generations.values()),
            "fallback_count": sum(row["fallback_original"] is True for row in generations.values()),
        }

        original_bm25 = _read_ranking(CACHED_RANKING_ROOT / name / "bm25_original.jsonl")
        original_bge = _read_ranking(CACHED_RANKING_ROOT / name / "bge_large_original.jsonl")
        if set(original_bm25) != expected_ids or set(original_bge) != expected_ids:
            raise ValueError(f"original B0/B1 ranking IDs differ from DEV/{name}")

        documents = [(doc.doc_id, doc.text) for doc in subset.documents]
        index = LuceneBM25Index(documents)
        bridge_texts = []
        dense_texts = []
        for query in subset.queries:
            structured = generations[query.query_id]["structured"]
            bridge_texts.append(crb_lexical_text(structured) or query.text)
            dense_texts.append(str(structured.get("pseudo_evidence", "")).strip() or query.text)
        bridge_rows = [index.search(text, top_k=100) for text in bridge_texts]
        corpus_vectors, _ = _load_or_encode_corpus(model, subset, bge_sha)
        generated_vectors = encode_bge(
            model,
            [BGE_QUERY_PREFIX + text for text in dense_texts],
            batch_size=32,
            show_progress_bar=False,
        )
        generated_dense_rows = dense_search_many(
            generated_vectors,
            corpus_vectors,
            [doc.doc_id for doc in subset.dense_documents],
            top_k=100,
            device="cuda",
        )
        bridge_by_query = {
            query_id: [item.doc_id for item in ranking]
            for query_id, ranking in zip(query_ids, bridge_rows, strict=True)
        }
        dense_by_query = {
            query_id: [item.doc_id for item in ranking]
            for query_id, ranking in zip(query_ids, generated_dense_rows, strict=True)
        }

        channels_dir = ROOT_OUTPUT / "rankings/channels/dev" / name
        bm25_channel_rows = _ranking_rows(name, "crb_q_compact_repair_bm25_bridge", query_ids, bridge_rows)
        dense_channel_rows = _ranking_rows(name, "crb_q_compact_repair_bge_generated", query_ids, generated_dense_rows)
        channel_hashes[name] = {
            "bm25_bridge": _write_jsonl_once(
                channels_dir / "bm25_bridge.jsonl", bm25_channel_rows
            ),
            "bge_generated": _write_jsonl_once(
                channels_dir / "bge_generated.jsonl", dense_channel_rows
            ),
        }

        fused = {
            query_id: weighted_rrf(
                (
                    original_bm25[query_id],
                    bridge_by_query[query_id],
                    original_bge[query_id],
                    dense_by_query[query_id],
                ),
                rrf_k=int(RRF_CONFIG["rrf_k"]),
                weights=RRF_CONFIG["weights"],
                top_k=100,
            )
            for query_id in query_ids
        }
        ranking_ids_by_subset[name] = {
            query_id: [item.doc_id for item in fused[query_id]] for query_id in query_ids
        }
        fused_rows = _ranking_rows(name, "crb_q_compact_repair_mv", query_ids, [fused[qid] for qid in query_ids])
        final_hashes[name] = _write_jsonl_once(
            ROOT_OUTPUT / "rankings/dev" / name / "crb_q_compact_repair_mv.jsonl",
            fused_rows,
        )
        print(
            f"compact DEV retrieval complete: {name}; queries={len(query_ids)}; "
            f"generation_valid={generation_counts[name]['valid_count']}",
            flush=True,
        )

    del model
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Qrels are first consumed here, by the evaluator, after all DEV rankings are materialized.
    query_metric_rows, candidate_summary = evaluate_rankings(
        "DEV", ranking_ids_by_subset, source_root=source_root
    )
    gate = dev_success_gate(candidate_summary, strong_gar)
    report = {
        "schema_version": "r2med-crb-compact-repair-dev-report-v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "partition": "DEV",
        "test_accessed": False,
        "test_run_started": False,
        "source_manifest_sha256": source_sha,
        "generation_manifest_sha256": generation_manifest_sha,
        "generation_counts": generation_counts,
        "generation_total": {
            "query_count": sum(row["query_count"] for row in generation_counts.values()),
            "valid_count": sum(row["valid_count"] for row in generation_counts.values()),
            "fallback_count": sum(row["fallback_count"] for row in generation_counts.values()),
        },
        "retrieval": {
            "bge_model_sha256": bge_sha,
            "bge_model_path": str(DEFAULT_BGE_ROOT),
            "corpus_embeddings_reused_read_only": True,
            "bm25_identity": parent_report["bm25_runtime"],
            "fusion": RRF_CONFIG,
            "top_k": 100,
            "channel_count": 4,
            "channel_artifact_sha256": channel_hashes,
            "ranking_artifact_sha256": final_hashes,
        },
        "baseline_reference": {
            "original_crb_q_macro_ndcg_at_10": original_crb["macro_equal_subset_weight"]["ndcg@10"],
            "strongest_cost_matched_gar_method": parent_report["strongest_cost_matched_gar"]["method"],
            "strongest_cost_matched_gar_macro_ndcg_at_10": strong_gar["macro_equal_subset_weight"]["ndcg@10"],
            "strongest_cost_matched_gar_config": parent_report["strongest_cost_matched_gar"]["config_id"],
        },
        "candidate_summary": candidate_summary,
        "dev_gate_vs_strongest_cost_matched_gar": gate,
        "by_subset_comparison": {
            subset: {
                "compact_crb_q_ndcg@10": candidate_summary["by_subset"][subset]["ndcg@10"],
                "original_crb_q_ndcg@10": original_crb["by_subset"][subset]["ndcg@10"],
                "strongest_gar_ndcg@10": strong_gar["by_subset"][subset]["ndcg@10"],
                "compact_crb_q_delta_vs_gar": (
                    candidate_summary["by_subset"][subset]["ndcg@10"]
                    - strong_gar["by_subset"][subset]["ndcg@10"]
                ),
            }
            for subset in SUBSETS
        },
        "per_query_metrics_sha256": _write_jsonl_once(
            ROOT_OUTPUT / "per_query_metrics_dev.jsonl", query_metric_rows
        ),
        "decision": "DEV_ONLY_STOP_AFTER_THIS_SINGLE_COMPACT_SCHEMA_VARIANT",
        "gold_boundary": "Only the evaluator received DEV qrels, after generation and ranking completed.",
    }
    output_path = ROOT_OUTPUT / "dev_report.json"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite compact repair report: {output_path}")
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "compact_macro_ndcg@10": candidate_summary["macro_equal_subset_weight"]["ndcg@10"],
        "original_crb_q_macro_ndcg@10": original_crb["macro_equal_subset_weight"]["ndcg@10"],
        "strongest_gar_macro_ndcg@10": strong_gar["macro_equal_subset_weight"]["ndcg@10"],
        "gate": gate,
        "report": str(output_path),
    }, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    args = parser.parse_args()
    run_dev(args.source_root)


if __name__ == "__main__":
    main()
