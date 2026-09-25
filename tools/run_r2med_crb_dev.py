"""Run the predeclared R2MED DEV GAR matrix and CRB selection gate."""

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

from eval.r2med_crb import BGE_QUERY_PREFIX, bm25_query_text, crb_lexical_text
from eval.r2med_crb_data import SOURCE_MANIFEST_PATH, load_partition_inputs, load_source_manifest
from eval.r2med_crb_evaluator import (
    GAR_METHOD_ORDER,
    dev_success_gate,
    evaluate_rankings,
    select_best_fusion,
    select_strongest_gar,
)
from eval.r2med_gar_generation import GENERATION_CONFIG, METHODS
from eval.r2med_multiview import (
    FUSION_CONFIGS,
    LuceneBM25Index,
    RankedDocument,
    dense_search_many,
    encode_bge,
    weighted_rrf,
)
from tools.generate_r2med_gar import DEFAULT_SERVER as _DEFAULT_SERVER
from tools.run_r2med_baselines import (
    MANIFEST_PATH as BASELINE_REPORT_PATH,
)
from tools.run_r2med_baselines import (
    RANKING_ROOT as BASELINE_RANKING_ROOT,
)
from tools.run_r2med_baselines import (
    _load_or_encode_corpus,
    _ranking_rows,
    _write_rankings,
    run_dev_baselines,
)
from tools.verify_r2med_models import E_ROOT, verify_models

DEFAULT_SOURCE_ROOT = Path(r"E:\Health-Copilot-E1.2\sources")
DEFAULT_BGE_ROOT = E_ROOT / "models/bge-large-en-v1.5"
DEFAULT_UPSTREAM = Path(r"D:\MyLab\Jianli\external\rag\R2MED")
DEFAULT_LLAMA_SERVER = Path(_DEFAULT_SERVER)
GENERATION_ROOT = ROOT / "runs/rag_r2med_crb/generation/dev"
RANKING_ROOT = E_ROOT / "r2med/rankings/dev"
REPORT_PATH = E_ROOT / "r2med/dev/reports/dev_report.json"
METHODS_WITH_SINGLE_VIEW = ("hyde", "query2doc", "lamer")
MULTIVIEW_METHODS = (*GAR_METHOD_ORDER, "crb_q", "crb_prf")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    result = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise TypeError(f"invalid JSONL object at {path}:{line_number}")
                result.append(row)
    return result


def _read_generation(method: str, subset, source_sha256: str) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    manifest_path = GENERATION_ROOT / method / "generation_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing frozen generation manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("partition") != "DEV"
        or manifest.get("method") != method
        or manifest.get("dataset_source_manifest_sha256") != source_sha256
        or manifest.get("call_count") != 393
        or manifest.get("upstream_commit") != load_source_manifest(SOURCE_MANIFEST_PATH)["upstream"]["commit"]
        or manifest.get("generator", {}).get("sha256")
        != "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785"
        or manifest.get("generation_config") != GENERATION_CONFIG
    ):
        raise ValueError(f"generation manifest identity mismatch: {method}")
    subset_entry = next((entry for entry in manifest["subsets"] if entry["subset"] == subset.name), None)
    if subset_entry is None or subset_entry["query_count"] != len(subset.queries):
        raise ValueError(f"generation manifest lacks the expected subset: {method}/{subset.name}")
    path = Path(subset_entry["artifact"])
    source_entry = next(
        item for item in load_source_manifest(SOURCE_MANIFEST_PATH)["datasets"]["DEV"]
        if item["name"] == subset.name
    )
    expected_order_sha = hashlib.sha256(
        "\n".join(query.query_id for query in subset.queries).encode("utf-8")
    ).hexdigest()
    if (
        subset_entry.get("query_file_sha256") != source_entry["files"]["query.jsonl"]["sha256"]
        or subset_entry.get("corpus_file_sha256") != source_entry["files"]["corpus.jsonl"]["sha256"]
        or subset_entry.get("query_order_sha256") != expected_order_sha
    ):
        raise ValueError(f"generation input identity mismatch: {method}/{subset.name}")
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != subset_entry["artifact_sha256"]:
        raise ValueError(f"generation artifact hash mismatch: {method}/{subset.name}")
    rows = _read_jsonl(path)
    expected_ids = [query.query_id for query in subset.queries]
    if [row.get("query_id") for row in rows] != expected_ids:
        raise ValueError(f"generation query order/count mismatch: {method}/{subset.name}")
    if any(row.get("method") != method or row.get("subset") != subset.name for row in rows):
        raise ValueError(f"generation row method/subset identity mismatch: {method}/{subset.name}")
    return {row["query_id"]: row for row in rows}, manifest


def _ensure_base_rankings(source_root: Path, bge_root: Path) -> dict[str, Any]:
    if BASELINE_REPORT_PATH.exists():
        report = json.loads(BASELINE_REPORT_PATH.read_text(encoding="utf-8"))
        expected_source_sha = hashlib.sha256(SOURCE_MANIFEST_PATH.read_bytes()).hexdigest()
        if (
            report.get("partition") != "DEV"
            or report.get("source_manifest_sha256") != expected_source_sha
            or report.get("bge_identity", {}).get("weights_sha256")
            != "45e1954914e29bd74080e6c1510165274ff5279421c89f76c418878732f64ae7"
        ):
            raise ValueError("baseline report source/model identity mismatch")
        for subset, artifacts in report.get("rank_artifacts", {}).items():
            for name, expected_hash in artifacts.items():
                filename = "bm25_original.jsonl" if name.startswith("bm25_") else "bge_large_original.jsonl"
                path = BASELINE_RANKING_ROOT / subset / filename
                if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
                    raise ValueError(f"base ranking artifact hash mismatch: {subset}/{filename}")
        return report
    partial = [
        path
        for subset in ("PMC-Treatment", "PMC-Clinical", "IIYi-Clinical")
        for path in (
            BASELINE_RANKING_ROOT / subset / "bm25_original.jsonl.partial",
            BASELINE_RANKING_ROOT / subset / "bge_large_original.jsonl.partial",
        )
        if path.exists()
    ]
    if partial:
        raise FileExistsError(f"partial base ranking file exists; inspect before resuming: {partial[0]}")
    return run_dev_baselines(source_root, bge_root)


def _ensure_generation(
    source_root: Path,
    upstream_root: Path,
    server_executable: Path,
) -> dict[str, dict[str, Any]]:
    from tools.generate_r2med_gar import generate_method

    load_source_manifest(SOURCE_MANIFEST_PATH)
    source_sha = hashlib.sha256(SOURCE_MANIFEST_PATH.read_bytes()).hexdigest()
    subset_by_name = {subset.name: subset for subset in load_partition_inputs("DEV", source_root=source_root)}
    stats: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        expected_path = GENERATION_ROOT / method / "generation_manifest.json"
        if not expected_path.exists():
            summary = generate_method(
                "DEV",
                method,
                upstream_root=upstream_root,
                source_root=source_root,
                server_executable=server_executable,
            )
        else:
            summary = json.loads(expected_path.read_text(encoding="utf-8"))
        if summary.get("dataset_source_manifest_sha256") != source_sha:
            raise ValueError(f"stale generation manifest found for {method}")
        stats[method] = {
            "call_count": summary["call_count"],
            "valid_count": summary["valid_output_count"],
            "completion_count": summary["completion_count"],
            "failure_count": summary["failure_count"],
            "truncation_count": summary["truncation_count"],
            "average_output_tokens": summary["average_output_tokens"],
            "prompt_sha256": summary["prompt_sha256"],
            "manifest_sha256": hashlib.sha256(expected_path.read_bytes()).hexdigest(),
            "subset_artifact_sha256": {
                entry["subset"]: entry["artifact_sha256"] for entry in summary["subsets"]
            },
        }
        for subset in subset_by_name.values():
            _read_generation(method, subset, source_sha)
    return stats


def _read_ranking(path: Path) -> dict[str, list[str]]:
    rows = _read_jsonl(path)
    result: dict[str, list[str]] = {}
    for row in rows:
        query_id = row["query_id"]
        ranking = row["ranking"]
        if query_id in result or not isinstance(ranking, list) or len(ranking) > 100:
            raise ValueError(f"invalid ranking artifact: {path}")
        result[query_id] = [str(item["doc_id"]) for item in ranking]
    return result


def _write_channel(subset_name: str, method: str, channel: str, query_ids: list[str], rows) -> str:
    path = RANKING_ROOT / subset_name / f"{method}_{channel}.jsonl"
    expected = _ranking_rows(subset_name, f"{method}_{channel}", query_ids, rows)
    return _write_or_verify_rankings(path, expected)


def _write_or_verify_rankings(path: Path, expected: list[dict[str, Any]]) -> str:
    if not path.exists():
        return _write_rankings(path, expected)
    existing = _read_jsonl(path)
    expected_identity = [
        (row.get("subset"), row.get("query_id"), row.get("method"), [doc["doc_id"] for doc in row["ranking"]])
        for row in expected
    ]
    existing_identity = [
        (row.get("subset"), row.get("query_id"), row.get("method"), [doc["doc_id"] for doc in row["ranking"]])
        for row in existing
    ]
    if existing_identity != expected_identity:
        raise ValueError(f"existing DEV ranking artifact differs from deterministic replay: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rankings_for_config(channels_by_subset, config: dict[str, Any]) -> dict[str, dict[str, list[str]]]:
    return {
        subset: {
            query_id: [
                row.doc_id
                for row in weighted_rrf(
                    query_channels,
                    rrf_k=config["rrf_k"],
                    weights=config["weights"],
                )
            ]
            for query_id, query_channels in query_channels_by_id.items()
        }
        for subset, query_channels_by_id in channels_by_subset.items()
    }


def _evaluate_config(channels_by_subset, config: dict[str, Any], *, source_root: Path):
    rankings = _rankings_for_config(channels_by_subset, config)
    return evaluate_rankings("DEV", rankings, source_root=source_root)


def _evaluate_ablation_single(rankings, *, source_root: Path) -> dict[str, Any]:
    _, summary = evaluate_rankings("DEV", rankings, source_root=source_root)
    return summary


def _pair_rrf(first: list[str], second: list[str], *, top_k: int = 100) -> list[str]:
    scores: dict[str, float] = {}
    for ranking in (first, second):
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (60 + rank)
    return sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))[:top_k]


def _load_base_rankings(subset_names: tuple[str, ...]) -> tuple[dict[str, dict[str, list[str]]], dict[str, dict[str, list[str]]]]:
    bm25: dict[str, dict[str, list[str]]] = {}
    bge: dict[str, dict[str, list[str]]] = {}
    for subset in subset_names:
        folder = BASELINE_RANKING_ROOT / subset
        bm25[subset] = _read_ranking(folder / "bm25_original.jsonl")
        bge[subset] = _read_ranking(folder / "bge_large_original.jsonl")
    return bm25, bge


def run_dev(
    *,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    bge_root: Path = DEFAULT_BGE_ROOT,
    upstream_root: Path = DEFAULT_UPSTREAM,
    server_executable: Path,
) -> dict[str, Any]:
    if REPORT_PATH.exists():
        raise FileExistsError(f"refusing to overwrite DEV report: {REPORT_PATH}")
    if REPORT_PATH.with_suffix(REPORT_PATH.suffix + ".partial").exists():
        raise FileExistsError(f"partial DEV report exists; inspect before resuming: {REPORT_PATH}")
    verified = verify_models(bge_root=bge_root)
    source_manifest = load_source_manifest(SOURCE_MANIFEST_PATH)
    source_sha = hashlib.sha256(SOURCE_MANIFEST_PATH.read_bytes()).hexdigest()
    subsets = load_partition_inputs("DEV", source_root=source_root)
    baseline = _ensure_base_rankings(source_root, bge_root)
    generator_stats = _ensure_generation(source_root, upstream_root, server_executable)

    os.environ["HF_HOME"] = str(E_ROOT / "cache/huggingface")
    os.environ["HF_HUB_CACHE"] = str(E_ROOT / "cache/huggingface/hub")
    os.environ["TRANSFORMERS_CACHE"] = str(E_ROOT / "cache/huggingface/transformers")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(bge_root), device="cuda", local_files_only=True)
    model.max_seq_length = 512
    model.eval()

    channels_by_method: dict[str, dict[str, dict[str, tuple[list[str], list[str], list[str], list[str]]]]] = {
        method: {} for method in MULTIVIEW_METHODS
    }
    single_rankings: dict[str, dict[str, dict[str, list[str]]]] = {
        method: {} for method in METHODS_WITH_SINGLE_VIEW
    }
    channel_artifact_hashes: dict[str, dict[str, dict[str, str]]] = {}
    generation_views: dict[str, dict[str, dict[str, dict[str, Any]]]] = {
        method: {} for method in METHODS
    }
    for method in METHODS:
        for subset in subsets:
            generation_views[method][subset.name], _ = _read_generation(method, subset, source_sha)

    for subset in subsets:
        query_ids = [query.query_id for query in subset.queries]
        query_texts = [query.text for query in subset.queries]
        docs = [(doc.doc_id, doc.text) for doc in subset.documents]
        doc_ids = [doc.doc_id for doc in subset.dense_documents]
        bm25_original = _read_ranking(BASELINE_RANKING_ROOT / subset.name / "bm25_original.jsonl")
        bge_original = _read_ranking(BASELINE_RANKING_ROOT / subset.name / "bge_large_original.jsonl")
        bm25_index = LuceneBM25Index(docs)
        corpus_vectors, _ = _load_or_encode_corpus(model, subset, verified["bge_large"]["weights_sha256"])
        original_query_vectors = encode_bge(
            model,
            [BGE_QUERY_PREFIX + text for text in query_texts],
            batch_size=32,
            show_progress_bar=False,
        )
        channel_artifact_hashes[subset.name] = {}
        for method in METHODS:
            views = generation_views[method][subset.name]
            generated_texts = [views[query_id]["generated_text"] for query_id in query_ids]
            dense_texts = (
                [
                    str((views[query_id].get("structured") or {}).get("pseudo_evidence", query_texts[index]))
                    for index, query_id in enumerate(query_ids)
                ]
                if method in {"crb_q", "crb_prf"}
                else generated_texts
            )
            generated_vectors = encode_bge(
                model,
                [BGE_QUERY_PREFIX + text for text in dense_texts],
                batch_size=32,
                show_progress_bar=False,
            )
            expanded_texts: list[str] = []
            if method in {"crb_q", "crb_prf"}:
                for query, query_id in zip(query_texts, query_ids, strict=True):
                    structured = views[query_id].get("structured") or {}
                    expanded_texts.append(crb_lexical_text(structured) or query)
            else:
                expanded_texts = [
                    bm25_query_text(method, query, generated)
                    for query, generated in zip(query_texts, generated_texts, strict=True)
                ]

            bm25_bridge_rows = [bm25_index.search(text, top_k=100) for text in expanded_texts]
            generated_dense_rows = dense_search_many(
                generated_vectors, corpus_vectors, doc_ids, top_k=100, device="cuda"
            )
            bm25_bridge = {
                query_id: [item.doc_id for item in rows]
                for query_id, rows in zip(query_ids, bm25_bridge_rows, strict=True)
            }
            bge_generated = {
                query_id: [item.doc_id for item in rows]
                for query_id, rows in zip(query_ids, generated_dense_rows, strict=True)
            }
            artifact_hashes = {
                "bm25_bridge_sha256": _write_channel(subset.name, method, "bm25_bridge", query_ids, bm25_bridge_rows),
                "bge_generated_sha256": _write_channel(subset.name, method, "bge_generated", query_ids, generated_dense_rows),
            }

            if method in METHODS_WITH_SINGLE_VIEW:
                if method == "query2doc":
                    single_texts = [
                        BGE_QUERY_PREFIX + f"{query}[SEP]{generated}"
                        for query, generated in zip(query_texts, generated_texts, strict=True)
                    ]
                    single_vectors = encode_bge(model, single_texts, batch_size=32, show_progress_bar=False)
                else:
                    single_vectors = (original_query_vectors + generated_vectors) / 2.0
                single_dense_rows = dense_search_many(
                    single_vectors, corpus_vectors, doc_ids, top_k=100, device="cuda"
                )
                single_rankings[method].setdefault(subset.name, {})
                single_rankings[method][subset.name] = {
                    query_id: [item.doc_id for item in rows]
                    for query_id, rows in zip(query_ids, single_dense_rows, strict=True)
                }
                single_bm25 = {
                    query_id: [item.doc_id for item in rows]
                    for query_id, rows in zip(query_ids, bm25_bridge_rows, strict=True)
                }
                single_rows = [
                    [RankedDocument(doc_id, 0.0) for doc_id in single_bm25[query_id][:100]]
                    for query_id in query_ids
                ]
                artifact_hashes["single_bm25_sha256"] = _write_channel(
                    subset.name, method, "single_bm25", query_ids, single_rows
                )
                dense_rows_as_docs = [
                    [RankedDocument(doc_id, 0.0) for doc_id in single_rankings[method][subset.name][query_id]]
                    for query_id in query_ids
                ]
                artifact_hashes["single_bge_sha256"] = _write_channel(
                    subset.name, method, "single_bge", query_ids, dense_rows_as_docs
                )
                single_rankings[method].setdefault("_bm25", {})
                single_rankings[method]["_bm25"][subset.name] = single_bm25

            channels_by_method[method][subset.name] = {
                query_id: (
                    bm25_original[query_id],
                    bm25_bridge[query_id],
                    bge_original[query_id],
                    bge_generated[query_id],
                )
                for query_id in query_ids
            }
            channel_artifact_hashes[subset.name][method] = artifact_hashes
            del generated_vectors, generated_dense_rows

        print(f"DEV retrieval channels complete: {subset.name} ({len(query_ids)} queries)")

    del model
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    single_results: dict[str, dict[str, Any]] = {
        "bm25": baseline["B0_BM25"]["metrics"],
        "bge_large": baseline["B1_BGE_large"]["metrics"],
    }
    for method in METHODS_WITH_SINGLE_VIEW:
        bm25_by_subset = single_rankings[method].pop("_bm25")
        _, bm25_summary = evaluate_rankings("DEV", bm25_by_subset, source_root=source_root)
        _, dense_summary = evaluate_rankings("DEV", single_rankings[method], source_root=source_root)
        names = {"hyde": "hyde", "query2doc": "query2doc", "lamer": "lamer"}
        single_results[f"{names[method]}_bm25"] = bm25_summary
        single_results[f"{names[method]}_bge_large"] = dense_summary

    multi_results: dict[str, dict[str, Any]] = {}
    best_config_by_method: dict[str, dict[str, Any]] = {}
    for method in MULTIVIEW_METHODS:
        config_summaries: dict[str, dict[str, Any]] = {}
        for config in FUSION_CONFIGS:
            _, summary = _evaluate_config(channels_by_method[method], config, source_root=source_root)
            config_summaries[config["config_id"]] = summary
        best_config_id, best_summary = select_best_fusion(config_summaries)
        config = next(item for item in FUSION_CONFIGS if item["config_id"] == best_config_id)
        best_config_by_method[method] = config
        multi_results[method] = {
            "configs": config_summaries,
            "best_config_id": best_config_id,
            "best_config": config,
            "best_summary": best_summary,
        }
        best_rankings = _rankings_for_config(channels_by_method[method], config)
        for subset_name, query_rankings in best_rankings.items():
            path = RANKING_ROOT / subset_name / f"{method}_mv_best.jsonl"
            _write_or_verify_rankings(
                path,
                [
                    {
                        "subset": subset_name,
                        "query_id": query_id,
                        "method": f"{method}_mv_best",
                        "ranking": [{"doc_id": doc_id, "score": 0.0} for doc_id in ranking],
                    }
                    for query_id, ranking in query_rankings.items()
                ],
            )

    best_gar_method, strongest_gar = select_strongest_gar(
        {method: multi_results[method]["best_summary"] for method in GAR_METHOD_ORDER}
    )
    strongest_single_name, strongest_single = max(
        single_results.items(),
        key=lambda item: (
            item[1]["macro_equal_subset_weight"]["ndcg@10"],
            item[1]["macro_equal_subset_weight"]["mrr@10"],
            item[1]["macro_equal_subset_weight"]["recall@10"],
        ),
    )
    best_crb_method, best_crb_summary = max(
        ((method, multi_results[method]["best_summary"]) for method in ("crb_q", "crb_prf")),
        key=lambda item: (
            item[1]["macro_equal_subset_weight"]["ndcg@10"],
            item[1]["macro_equal_subset_weight"]["mrr@10"],
            item[1]["macro_equal_subset_weight"]["recall@10"],
            int(item[0] == "crb_q"),
        ),
    )
    gate = dev_success_gate(best_crb_summary, strongest_gar[1])
    chosen_crb_config = best_config_by_method[best_crb_method]
    chosen_crb_rankings = _rankings_for_config(channels_by_method[best_crb_method], chosen_crb_config)

    ablation: dict[str, Any] | None = None
    if gate["signal"] == "NEGATIVE":
        ablation_rankings: dict[str, dict[str, dict[str, list[str]]]] = {
            "A0_original_bge_large": _load_base_rankings(tuple(subset.name for subset in subsets))[1],
            "A1_crb_lexical_bm25": {},
            "A2_crb_pseudo_evidence_bge": {},
            "A3_original_bm25_plus_crb_bm25": {},
            "A4_original_bge_plus_crb_bge": {},
            "A5_full_crb": chosen_crb_rankings,
        }
        for subset in subsets:
            query_ids = [query.query_id for query in subset.queries]
            channels = channels_by_method[best_crb_method][subset.name]
            ablation_rankings["A1_crb_lexical_bm25"][subset.name] = {
                query_id: channels[query_id][1] for query_id in query_ids
            }
            ablation_rankings["A2_crb_pseudo_evidence_bge"][subset.name] = {
                query_id: channels[query_id][3] for query_id in query_ids
            }
            ablation_rankings["A3_original_bm25_plus_crb_bm25"][subset.name] = {
                query_id: _pair_rrf(channels[query_id][0], channels[query_id][1])
                for query_id in query_ids
            }
            ablation_rankings["A4_original_bge_plus_crb_bge"][subset.name] = {
                query_id: _pair_rrf(channels[query_id][2], channels[query_id][3])
                for query_id in query_ids
            }
        ablation_summaries = {
            label: _evaluate_ablation_single(rankings, source_root=source_root)
            for label, rankings in ablation_rankings.items()
        }
        crb_stats = generator_stats[best_crb_method]
        quality_rate = crb_stats["valid_count"] / crb_stats["call_count"]
        lex = ablation_summaries["A1_crb_lexical_bm25"]["macro_equal_subset_weight"]
        original_bm25 = single_results["bm25"]["macro_equal_subset_weight"]
        dense = ablation_summaries["A2_crb_pseudo_evidence_bge"]["macro_equal_subset_weight"]
        original_bge = single_results["bge_large"]["macro_equal_subset_weight"]
        lexical_good = lex["ndcg@10"] > original_bm25["ndcg@10"] or lex["recall@100"] > original_bm25["recall@100"]
        dense_good = dense["ndcg@10"] > original_bge["ndcg@10"] or dense["recall@100"] > original_bge["recall@100"]
        ablation = {
            "summaries": ablation_summaries,
            "diagnostic_flags": {
                "GENERATION_BAD": quality_rate < 0.99 or crb_stats["truncation_count"] > 0,
                "LEXICAL_BRIDGE_BAD": not lexical_good,
                "DENSE_BRIDGE_BAD": not dense_good,
                "FUSION_BAD": lexical_good and dense_good and gate["signal"] == "NEGATIVE",
            },
            "notes": "A3/A4 use equal-weight two-channel RRF at k=60; all ablations are DEV-only diagnostics.",
        }

    best_crb_manifest = json.loads(
        (GENERATION_ROOT / best_crb_method / "generation_manifest.json").read_text(encoding="utf-8")
    )
    if len(best_crb_manifest["prompt_sha256"]) != 1:
        raise ValueError("CRB prompt identity must contain exactly one hash")
    report = {
        "schema_version": "r2med-crb-dev-report-v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "partition": "DEV",
        "test_status": source_manifest["test_status"],
        "source_manifest_sha256": source_sha,
        "verified_models": verified,
        "base_retrieval_manifest_sha256": hashlib.sha256(BASELINE_REPORT_PATH.read_bytes()).hexdigest(),
        "generation_config": GENERATION_CONFIG,
        "generation_audit": generator_stats,
        "single_view": single_results,
        "multi_view": multi_results,
        "strongest_single_baseline": {
            "method": strongest_single_name,
            "summary": strongest_single,
        },
        "strongest_cost_matched_gar": {
            "method": best_gar_method,
            "summary": strongest_gar[1],
            "config_id": multi_results[best_gar_method]["best_config_id"],
        },
        "best_crb": {
            "method": best_crb_method,
            "config_id": multi_results[best_crb_method]["best_config_id"],
            "prompt_sha256": best_crb_manifest["prompt_sha256"][0],
            "summary": best_crb_summary,
        },
        "gate": gate,
        "channel_artifact_hashes": channel_artifact_hashes,
        "ablation": ablation,
        "stop_rule": "No TEST unless the predeclared DEV gate is positive; a negative result ends this R2MED CRB sprint after the predefined ablation.",
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report_partial = REPORT_PATH.with_suffix(REPORT_PATH.suffix + ".partial")
    with report_partial.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    report_partial.replace(REPORT_PATH)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--bge-root", type=Path, default=DEFAULT_BGE_ROOT)
    parser.add_argument("--upstream-root", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument(
        "--llama-server",
        type=Path,
        default=DEFAULT_LLAMA_SERVER,
    )
    args = parser.parse_args()
    report = run_dev(
        source_root=args.source_root,
        bge_root=args.bge_root,
        upstream_root=args.upstream_root,
        server_executable=args.llama_server,
    )
    print(json.dumps({"gate": report["gate"], "strongest_single": report["strongest_single_baseline"], "strongest_gar": report["strongest_cost_matched_gar"], "best_crb": report["best_crb"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
