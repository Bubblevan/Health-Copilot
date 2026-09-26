"""Run the frozen R2MED DEV reranker grid; qrels are opened only for final scoring."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.r2med_candidate_union import Candidate, fuse_dual_source_rrf
from eval.r2med_crb_data import (
    DEFAULT_SOURCE_ROOT,
    PARTITIONS,
    load_subset_inputs,
    sha256_file,
)
from eval.r2med_crb_evaluator import evaluate_rankings
from eval.r2med_reranker import (
    AAR_ALPHAS,
    MODEL_ID,
    MODEL_REVISION,
    RERANK_DEPTHS,
    BGEReranker,
    aar_is_eligible,
    append_score_cache,
    load_score_cache,
    pair_cache_key,
    rerank_top_k,
    select_best_dev_arm,
)

PROTOCOL_PATH = ROOT / "runs/rag_r2med_rerank/protocol.json"
ANALYSIS_PATH = ROOT / "runs/rag_r2med_rerank/candidate_analysis.json"
LAMER_ROOT = Path(r"E:\Health-Copilot-RAG\r2med\rankings\dev")
CRB_ROOT = ROOT / "runs/rag_r2med_crb/compact_repair/rankings/dev"
MODEL_ROOT = Path(r"E:\Health-Copilot-RAG\models\bge-reranker-v2-m3")
RERANK_ROOT = Path(r"E:\Health-Copilot-RAG\r2med\reranker")
REPORT_PATH = ROOT / "runs/rag_r2med_rerank/dev_report.json"
SUBSETS = PARTITIONS["DEV"]
BASELINE_PATHS = {
    "B0_BM25": "bm25_original.jsonl",
    "B1_BGE_large": "bge_large_original.jsonl",
    "B3_LameR_single_BGE": "lamer_single_bge.jsonl",
    "B4_LameR_MV": "lamer_mv_mv_best.jsonl",
}
CRB_FILENAME = "crb_q_compact_repair_mv.jsonl"


def _read_rankings(path: Path, subset: str, *, top_n: int = 100) -> dict[str, list[Candidate]]:
    result: dict[str, list[Candidate]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("subset") != subset:
                raise ValueError(f"subset mismatch in {path}:{line_number}")
            query_id = row.get("query_id")
            entries = row.get("ranking")
            if not isinstance(query_id, str) or not isinstance(entries, list):
                raise TypeError(f"invalid ranking row in {path}:{line_number}")
            if len(entries) < top_n:
                raise ValueError(f"ranking shorter than {top_n}: {path}:{line_number}")
            doc_ids = [entry.get("doc_id") for entry in entries[:top_n]]
            if any(not isinstance(doc_id, str) or not doc_id for doc_id in doc_ids):
                raise TypeError(f"invalid document ID in {path}:{line_number}")
            if len(set(doc_ids)) != len(doc_ids):
                raise ValueError(f"duplicate document ID in {path}:{line_number}")
            if query_id in result:
                raise ValueError(f"duplicate query ID in {path}:{line_number}")
            result[query_id] = [Candidate(doc_id, rank, None, 0.0) for rank, doc_id in enumerate(doc_ids, 1)]
    return result


def _source_candidate_map(
    rankings: Mapping[str, Sequence[Candidate]], *, source: str
) -> dict[str, list[Candidate]]:
    return {
        query_id: [
            Candidate(
                candidate.doc_id,
                candidate.lamer_rank if source == "lamer" else None,
                candidate.lamer_rank if source == "crb" else None,
                candidate.score,
            )
            for candidate in candidates
        ]
        for query_id, candidates in rankings.items()
    }


def _ids_by_subset(
    rankings: Mapping[str, Mapping[str, Sequence[Candidate]]],
) -> dict[str, dict[str, list[str]]]:
    return {
        subset: {query_id: [candidate.doc_id for candidate in items] for query_id, items in per_query.items()}
        for subset, per_query in rankings.items()
    }


def _evaluate_arm(
    arm_rankings: Mapping[str, Mapping[str, Sequence[str]]], source_root: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return evaluate_rankings("DEV", arm_rankings, source_root=source_root)


def _write_json_once(path: Path, value: Mapping[str, Any]) -> None:
    content = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"refusing to overwrite a different report: {path}")
        return
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)


def _write_rankings_once(path: Path, subset: str, rankings: Mapping[str, Sequence[str]]) -> str:
    lines = "".join(
        json.dumps(
            {"subset": subset, "query_id": query_id, "ranking": [{"doc_id": doc_id} for doc_id in docs]},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
        for query_id, docs in sorted(rankings.items())
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != lines:
            raise FileExistsError(f"refusing to overwrite a different ranking artifact: {path}")
    else:
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(lines)
    return sha256_file(path)


def _write_jsonl_once(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    lines = "".join(
        json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != lines:
            raise FileExistsError(f"refusing to overwrite a different metric artifact: {path}")
    else:
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(lines)
    return sha256_file(path)


def _check_input_hashes(protocol: Mapping[str, Any], subset: str, lamer_path: Path, crb_path: Path) -> None:
    expected_lamer = protocol["candidate_sources"]["lamer"]["sha256_per_subset"][subset]
    expected_crb = protocol["candidate_sources"]["crb"]["sha256_per_subset"][subset]
    if sha256_file(lamer_path) != expected_lamer or sha256_file(crb_path) != expected_crb:
        raise ValueError(f"frozen candidate ranking identity mismatch for {subset}")
    for arm, filename in BASELINE_PATHS.items():
        if arm == "B4_LameR_MV":
            continue  # Pinned above under candidate_sources.lamer.
        if arm not in protocol["baseline_rankings"]:
            raise ValueError(f"baseline rank identity is not frozen: {arm}")
        expected = protocol["baseline_rankings"][arm]["sha256_per_subset"][subset]
        if sha256_file(LAMER_ROOT / subset / filename) != expected:
            raise ValueError(f"frozen baseline ranking identity mismatch: {arm}/{subset}")


def run_dev(
    *,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    model_root: Path = MODEL_ROOT,
    rerank_root: Path = RERANK_ROOT,
    report_path: Path = REPORT_PATH,
) -> dict[str, Any]:
    if report_path.exists():
        raise FileExistsError(f"DEV report already exists; refusing to rerun or overwrite: {report_path}")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_BEFORE_RERANKER_INFERENCE":
        raise ValueError("reranker protocol is not frozen")
    if protocol["reranker"]["name"] != MODEL_ID or protocol["reranker"]["revision"] != MODEL_REVISION:
        raise ValueError("local reranker constants do not match the frozen protocol")
    complementarity = json.loads(ANALYSIS_PATH.read_text(encoding="utf-8"))
    if complementarity["candidate_pool_union_gate"]["decision"] != "POSITIVE":
        raise ValueError("DualSource-RRF is prohibited because Phase A gate is not positive")

    # Preflight the pinned local weights before loading the model or doing inference.
    from eval.r2med_reranker import hash_model_files

    model_files = hash_model_files(model_root)
    model = BGEReranker(model_root)
    if model.model_files != model_files:
        raise RuntimeError("reranker model files changed during preflight")

    source_rankings: dict[str, dict[str, dict[str, list[Candidate]]]] = {}
    query_text_by_subset: dict[str, dict[str, str]] = {}
    document_text_by_subset: dict[str, dict[str, str]] = {}
    query_count_by_subset: dict[str, int] = {}
    dual_rankings: dict[float, dict[str, dict[str, list[Candidate]]]] = {
        0.5: {}, 1.0: {}, 2.0: {}
    }

    for subset in SUBSETS:
        lamer_path = LAMER_ROOT / subset / "lamer_mv_mv_best.jsonl"
        crb_path = CRB_ROOT / subset / CRB_FILENAME
        _check_input_hashes(protocol, subset, lamer_path, crb_path)
        lamer = _read_rankings(lamer_path, subset)
        crb = _read_rankings(crb_path, subset)
        if set(lamer) != set(crb):
            raise ValueError(f"LameR and CRB query IDs differ for {subset}")
        data = load_subset_inputs("DEV", subset, source_root=source_root)
        query_text_by_subset[subset] = {query.query_id: query.text for query in data.queries}
        document_text_by_subset[subset] = {document.doc_id: document.text for document in data.documents}
        query_count_by_subset[subset] = len(data.queries)

        baseline_rankings: dict[str, dict[str, list[Candidate]]] = {}
        for arm, filename in BASELINE_PATHS.items():
            path = LAMER_ROOT / subset / filename
            baseline_rankings[arm] = _read_rankings(path, subset)
        baseline_rankings["B5_Compact_CRB_Q"] = _source_candidate_map(crb, source="crb")

        bm25 = baseline_rankings["B0_BM25"]
        bge = baseline_rankings["B1_BGE_large"]
        if set(bm25) != set(bge):
            raise ValueError(f"BM25/BGE query IDs differ for {subset}")
        baseline_rankings["B2_BM25_BGE_RRF"] = {
            query_id: fuse_dual_source_rrf(
                [candidate.doc_id for candidate in bm25[query_id]],
                [candidate.doc_id for candidate in bge[query_id]],
                k=60,
            )[:100]
            for query_id in sorted(bm25)
        }
        baseline_rankings["B4_LameR_MV"] = _source_candidate_map(lamer, source="lamer")
        expected_query_ids = set(query_text_by_subset[subset])
        for arm, rankings in baseline_rankings.items():
            if set(rankings) != expected_query_ids:
                raise ValueError(f"{arm} query IDs differ from frozen query inputs in {subset}")
            if any(
                candidate.doc_id not in document_text_by_subset[subset]
                for items in rankings.values()
                for candidate in items
            ):
                raise ValueError(f"{arm} contains a document outside the frozen corpus in {subset}")
        source_rankings[subset] = baseline_rankings
        for lam in (0.5, 1.0, 2.0):
            dual_rankings[lam][subset] = {
                query_id: fuse_dual_source_rrf(
                    [item.doc_id for item in lamer[query_id]],
                    [item.doc_id for item in crb[query_id]],
                    crb_weight=lam,
                    k=60,
                )
                for query_id in sorted(lamer)
            }

    expected_queries = sum(query_count_by_subset.values())
    if expected_queries != 393:
        raise ValueError(f"expected 393 DEV queries; loaded {expected_queries}")

    # No qrels are read above this line. Build all score requests from query/corpus only.
    pair_records: dict[str, dict[str, str]] = {}
    cache_key_by_pair: dict[tuple[str, str, str], str] = {}
    for subset in SUBSETS:
        queries = query_text_by_subset[subset]
        documents = document_text_by_subset[subset]
        candidate_families = [
            source_rankings[subset]["B4_LameR_MV"],
            source_rankings[subset]["B5_Compact_CRB_Q"],
            *(dual_rankings[lam][subset] for lam in (0.5, 1.0, 2.0)),
        ]
        for family in candidate_families:
            for query_id, candidates in family.items():
                if query_id not in queries:
                    raise ValueError(f"candidate query absent from frozen query set: {query_id}")
                for candidate in candidates[:50]:
                    if candidate.doc_id not in documents:
                        raise ValueError(f"candidate document absent from corpus: {candidate.doc_id}")
                    key = pair_cache_key(
                        query_id=query_id,
                        doc_id=candidate.doc_id,
                        query_text=queries[query_id],
                        document_text=documents[candidate.doc_id],
                    )
                    cache_key_by_pair[(subset, query_id, candidate.doc_id)] = key
                    pair_records.setdefault(
                        key,
                        {
                            "query_id": query_id,
                            "doc_id": candidate.doc_id,
                            "query_text": queries[query_id],
                            "document_text": documents[candidate.doc_id],
                        },
                    )

    cache_path = rerank_root / "dev" / "pair_scores.jsonl"
    cache_scores = load_score_cache(cache_path)
    pending = [(key, record) for key, record in pair_records.items() if key not in cache_scores]
    newly_scored = 0
    for offset in range(0, len(pending), 256):
        batch = pending[offset : offset + 256]
        scored = model.score_pairs(
            [(record["query_text"], record["document_text"]) for _, record in batch]
        )
        if len(scored) != len(batch):
            raise RuntimeError("reranker score count differs from requested pair count")
        cache_rows = [
            {
                "cache_key": key,
                "query_id": record["query_id"],
                "doc_id": record["doc_id"],
                "model_revision": MODEL_REVISION,
                "reranker_score": score,
            }
            for (key, record), score in zip(batch, scored, strict=True)
        ]
        append_score_cache(cache_path, cache_rows, known_scores=cache_scores)
        cache_scores.update({row["cache_key"]: float(row["reranker_score"]) for row in cache_rows})
        newly_scored += len(cache_rows)

    score_for = {
        (subset, query_id, doc_id): cache_scores[key]
        for (subset, query_id, doc_id), key in cache_key_by_pair.items()
    }

    arm_rankings: dict[str, dict[str, dict[str, list[str]]]] = {}
    arm_configs: dict[str, dict[str, Any]] = {}
    for subset in SUBSETS:
        for base_arm in ("B0_BM25", "B1_BGE_large", "B2_BM25_BGE_RRF", "B3_LameR_single_BGE", "B4_LameR_MV", "B5_Compact_CRB_Q"):
            arm_rankings.setdefault(base_arm, {}).setdefault(subset, {})
            for query_id, candidates in source_rankings[subset][base_arm].items():
                arm_rankings[base_arm][subset][query_id] = [item.doc_id for item in candidates]

        for lam in (0.5, 1.0, 2.0):
            plain_name = f"DualSource_RRF_lam{lam:g}"
            arm_configs[plain_name] = {"source": "DualSource-RRF", "depth": 0, "lambda": lam}
            arm_rankings.setdefault(plain_name, {})[subset] = {
                query_id: [item.doc_id for item in items]
                for query_id, items in dual_rankings[lam][subset].items()
            }

        for arm, candidates, lam in (
            ("LameR_MV", source_rankings[subset]["B4_LameR_MV"], None),
            ("Compact_CRB_Q", source_rankings[subset]["B5_Compact_CRB_Q"], None),
            *((f"DualSource_RRF_lam{lam:g}", dual_rankings[lam][subset], lam) for lam in (0.5, 1.0, 2.0)),
        ):
            for depth in RERANK_DEPTHS:
                config_name = f"{arm}_BGE_Rerank_K{depth}"
                arm_configs[config_name] = {"source": arm, "depth": depth, "lambda": lam}
                arm_rankings.setdefault(config_name, {}).setdefault(subset, {})
                for query_id, items in candidates.items():
                    scores = {
                        item.doc_id: score_for[(subset, query_id, item.doc_id)]
                        for item in items[:depth]
                    }
                    reranked = rerank_top_k(items, scores, depth=depth)
                    arm_rankings[config_name][subset][query_id] = [
                        entry.candidate.doc_id for entry in reranked
                    ]

    # Qrels are consumed only after candidate generation and all reranker inference finish.
    per_arm: dict[str, dict[str, Any]] = {}
    query_metrics_by_arm: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    ranking_hashes: dict[str, dict[str, str]] = {}
    for arm_name, rankings in arm_rankings.items():
        metrics_rows, summary = _evaluate_arm(rankings, source_root)
        per_arm[arm_name] = summary
        query_metrics_by_arm[arm_name] = {
            (str(row["subset"]), str(row["query_id"])): row for row in metrics_rows
        }
        if arm_name in arm_configs:
            ranking_hashes[arm_name] = {}
            for subset in SUBSETS:
                path = rerank_root / "dev" / "rankings" / subset / f"{arm_name}.jsonl"
                ranking_hashes[arm_name][subset] = _write_rankings_once(path, subset, rankings[subset])

    for arm_name, reference_key in (
        ("B4_LameR_MV", "LameR-MV"),
        ("B5_Compact_CRB_Q", "Compact_CRB_Q"),
    ):
        actual = per_arm[arm_name]["macro_equal_subset_weight"]
        expected = protocol["frozen_dev_reference_metrics"][reference_key]
        for actual_key, expected_key in (("ndcg@10", "macro_ndcg@10"), ("recall@100", "macro_recall@100")):
            if abs(float(actual[actual_key]) - float(expected[expected_key])) > 1e-9:
                raise ValueError(f"recomputed frozen DEV reference does not match: {reference_key}/{actual_key}")

    base_names = [f"{arm}_BGE_Rerank_K{depth}" for arm in ("LameR_MV", "Compact_CRB_Q") for depth in RERANK_DEPTHS]
    dual_names = [f"DualSource_RRF_lam{lam:g}_BGE_Rerank_K{depth}" for lam in (0.5, 1.0, 2.0) for depth in RERANK_DEPTHS]
    lamer_best = select_best_dev_arm(
        per_arm,
        arm_configs,
        candidates=[name for name in base_names if name.startswith("LameR_MV_")],
    )
    best_dual = select_best_dev_arm(per_arm, arm_configs, candidates=dual_names)
    aar_eligible = aar_is_eligible(
        float(per_arm[best_dual]["macro_equal_subset_weight"]["ndcg@10"]),
        float(per_arm[lamer_best]["macro_equal_subset_weight"]["ndcg@10"]),
    )
    aar_names: list[str] = []
    if aar_eligible:
        parent_config = arm_configs[best_dual]
        parent_lam = float(parent_config["lambda"])
        parent_depth = int(parent_config["depth"])
        for alpha in AAR_ALPHAS:
            name = f"AAR_lam{parent_lam:g}_K{parent_depth}_alpha{alpha:g}"
            arm_configs[name] = {
                "source": "AAR",
                "lambda": parent_lam,
                "depth": parent_depth,
                "alpha": alpha,
                "parent": best_dual,
            }
            arm_rankings[name] = {}
            for subset in SUBSETS:
                arm_rankings[name][subset] = {}
                for query_id, items in dual_rankings[parent_lam][subset].items():
                    scores = {
                        item.doc_id: score_for[(subset, query_id, item.doc_id)]
                        for item in items[:parent_depth]
                    }
                    reranked = rerank_top_k(
                        items, scores, depth=parent_depth, agreement_alpha=alpha
                    )
                    arm_rankings[name][subset][query_id] = [
                        entry.candidate.doc_id for entry in reranked
                    ]
            metrics_rows, summary = _evaluate_arm(arm_rankings[name], source_root)
            per_arm[name] = summary
            query_metrics_by_arm[name] = {
                (str(row["subset"]), str(row["query_id"])): row for row in metrics_rows
            }
            aar_names.append(name)
            ranking_hashes[name] = {}
            for subset in SUBSETS:
                path = rerank_root / "dev" / "rankings" / subset / f"{name}.jsonl"
                ranking_hashes[name][subset] = _write_rankings_once(path, subset, arm_rankings[name][subset])

    per_query_path = rerank_root / "dev" / "per_query_metrics.jsonl"
    per_query_artifact = [
        {"arm": arm_name, **row}
        for arm_name in sorted(query_metrics_by_arm)
        for _, row in sorted(query_metrics_by_arm[arm_name].items())
    ]
    per_query_metrics_sha256 = _write_jsonl_once(per_query_path, per_query_artifact)

    non_ours_names = [
        "B0_BM25",
        "B1_BGE_large",
        "B2_BM25_BGE_RRF",
        "B3_LameR_single_BGE",
        "B4_LameR_MV",
        "B5_Compact_CRB_Q",
        *base_names,
    ]
    strongest_non_ours = max(
        non_ours_names,
        key=lambda name: float(per_arm[name]["macro_equal_subset_weight"]["ndcg@10"]),
    )
    plain_dual_names = [f"DualSource_RRF_lam{lam:g}" for lam in (0.5, 1.0, 2.0)]
    ours_names = plain_dual_names + dual_names + aar_names
    best_ours = select_best_dev_arm(per_arm, arm_configs, candidates=ours_names)
    overall_delta = (
        float(per_arm[best_ours]["macro_equal_subset_weight"]["ndcg@10"])
        - float(per_arm[strongest_non_ours]["macro_equal_subset_weight"]["ndcg@10"])
    )
    positive_subsets = sum(
        float(per_arm[best_ours]["by_subset"][subset]["ndcg@10"])
        >= float(per_arm[strongest_non_ours]["by_subset"][subset]["ndcg@10"])
        for subset in SUBSETS
    )
    dev_signal = overall_delta >= 0.005 and positive_subsets >= 2
    lamer_reference = per_arm["B4_LameR_MV"]["macro_equal_subset_weight"]
    strong_signal = (
        overall_delta >= 0.010
        and float(per_arm[best_ours]["macro_equal_subset_weight"]["recall@100"])
        >= float(lamer_reference["recall@100"])
    )

    report = {
        "schema_version": "r2med-reranker-dev-report-v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "partition": "DEV",
        "test_accessed": False,
        "query_count": expected_queries,
        "code_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip(),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "candidate_analysis_sha256": sha256_file(ANALYSIS_PATH),
        "reranker": {
            "name": MODEL_ID,
            "revision": MODEL_REVISION,
            "local_model_file_sha256": model_files,
            "device": str(model.device),
            "dtype": "float16",
            "max_pair_tokens": model.max_pair_tokens,
            "batch_size": model.batch_size,
        },
        "generation": {"calls": 0, "existing_generation_artifacts_modified": False},
        "scoring_cache": {
            "path": str(cache_path),
            "unique_scored_pair_count": len(cache_scores),
            "pair_requests_this_run": len(pair_records),
            "newly_scored_pair_count": newly_scored,
            "sha256": sha256_file(cache_path),
        },
        "reasonrank_skipped_resource_constraint": True,
        "arm_summaries": per_arm,
        "reranked_ranking_sha256": ranking_hashes,
        "per_query_metrics": {
            "path": str(per_query_path),
            "row_count": len(per_query_artifact),
            "sha256": per_query_metrics_sha256,
        },
        "selection": {
            "primary_metric": "equal-subset macro nDCG@10",
            "strongest_non_ours": strongest_non_ours,
            "strongest_non_ours_macro_ndcg@10": per_arm[strongest_non_ours]["macro_equal_subset_weight"]["ndcg@10"],
            "best_ours": best_ours,
            "best_ours_config": arm_configs[best_ours],
            "best_ours_macro_ndcg@10": per_arm[best_ours]["macro_equal_subset_weight"]["ndcg@10"],
            "delta_vs_strongest_non_ours": overall_delta,
            "noninferior_subsets": positive_subsets,
            "dev_signal": "POSITIVE" if dev_signal else "NEGATIVE",
            "strong_dev_signal": bool(strong_signal),
            "aar_eligible": aar_eligible,
            "aar_configs_evaluated": aar_names,
            "primary_reference_lamer_or_reranked": max(
                float(per_arm["B4_LameR_MV"]["macro_equal_subset_weight"]["ndcg@10"]),
                float(per_arm[lamer_best]["macro_equal_subset_weight"]["ndcg@10"]),
            ),
            "test_allowed": bool(dev_signal),
        },
        "gold_boundary": "Only evaluation opened DEV qrels, after query/corpus-only reranking completed.",
    }
    _write_json_once(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    parser.add_argument("--rerank-root", type=Path, default=RERANK_ROOT)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    args = parser.parse_args()
    report = run_dev(
        source_root=args.source_root,
        model_root=args.model_root,
        rerank_root=args.rerank_root,
        report_path=args.report,
    )
    print(json.dumps(report["selection"], indent=2))


if __name__ == "__main__":
    main()
