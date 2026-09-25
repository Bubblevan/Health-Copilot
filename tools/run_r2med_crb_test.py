"""Execute R2MED TEST once, and only under a committed positive DEV lock."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.r2med_crb import BGE_QUERY_PREFIX, bm25_query_text, crb_lexical_text
from eval.r2med_crb_data import (
    PARTITIONS,
    SOURCE_MANIFEST_PATH,
    SPRINT_LOCKED_PATHS,
    load_partition_inputs,
    load_source_manifest,
)
from eval.r2med_crb_evaluator import (
    GAR_GENERATION_TO_MULTIVIEW,
    GAR_MULTIVIEW_TO_GENERATION,
    evaluate_rankings,
    paired_stratified_bootstrap,
)
from eval.r2med_gar_generation import GENERATION_CONFIG, METHODS
from eval.r2med_multiview import (
    FUSION_CONFIGS,
    LuceneBM25Index,
    dense_search_many,
    encode_bge,
    weighted_rrf,
)
from tools.run_r2med_baselines import (
    _load_or_encode_corpus,
    _ranking_rows,
    _write_rankings,
    run_partition_baselines,
)
from tools.run_r2med_crb_dev import _read_jsonl, _read_ranking
from tools.verify_r2med_models import E_ROOT, verify_models

LOCK_PATH = ROOT / "runs/rag_r2med_crb/final_method_lock.json"
DEV_REPORT_PATH = E_ROOT / "r2med/dev/reports/dev_report.json"
TEST_BASELINE_REPORT = E_ROOT / "r2med/test/reports/test_base_retrieval_manifest.json"
TEST_REPORT_PATH = E_ROOT / "r2med/test/reports/test_report.json"
TEST_RUN_MARKER = E_ROOT / "r2med/test/test_run_started.json"
TEST_RANKING_ROOT = E_ROOT / "r2med/rankings/test"
TEST_GENERATION_ROOT = ROOT / "runs/rag_r2med_crb/generation/test"
TEST_GENERATED_VIEWS = E_ROOT / "r2med/generated/test"
DEFAULT_SOURCE_ROOT = Path(r"E:\Health-Copilot-E1.2\sources")
DEFAULT_BGE_ROOT = E_ROOT / "models/bge-large-en-v1.5"
DEFAULT_UPSTREAM = Path(r"D:\MyLab\Jianli\external\rag\R2MED")
DEFAULT_SERVER = Path(
    r"C:\Users\bubblevan\AppData\Local\Microsoft\WinGet\Packages\ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe\llama-server.exe"
)
GAR_METHODS = ("hyde", "query2doc", "lamer")


def resolve_dev_gar_baseline(dev_report: dict[str, Any]) -> str:
    selected = dev_report["strongest_cost_matched_gar"]["method"]
    try:
        return GAR_MULTIVIEW_TO_GENERATION[selected]
    except KeyError as error:
        raise ValueError("DEV report names an invalid strongest cost-matched GAR baseline") from error


def dev_multiview_configs(dev_report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        method: dev_report["multi_view"][GAR_GENERATION_TO_MULTIVIEW[method]]["best_config"]
        for method in GAR_METHODS
    }
def validate_test_lock(lock: dict[str, Any] | None, *, committed: bool) -> dict[str, Any]:
    if not committed or lock is None:
        raise FileNotFoundError("TEST is blocked until final_method_lock.json is committed")
    if lock.get("status") != "FROZEN_BEFORE_TEST":
        raise ValueError("TEST lock is not frozen")
    if lock.get("test_status") != "PUBLIC_BENCHMARK_REUSED":
        raise ValueError("TEST lock has an invalid public-test exposure status")
    if lock.get("crb_variant") not in {"crb_q", "crb_prf"}:
        raise ValueError("TEST lock has no supported CRB variant")
    gate = lock.get("dev_gate", {})
    if gate.get("signal") != "POSITIVE" or float(gate.get("delta", 0)) < 0.005:
        raise ValueError("TEST is prohibited because the DEV gate is negative")
    if int(gate.get("positive_subsets", 0)) < 2:
        raise ValueError("TEST is prohibited unless at least two DEV subsets improved")
    return lock


def assert_frozen_weights(lock: dict[str, Any], *, rrf_k: int, weights: tuple[int, ...]) -> None:
    allowed = {
        (int(config["rrf_k"]), tuple(config["weights"])) for config in FUSION_CONFIGS
    }
    if (rrf_k, weights) not in allowed:
        raise ValueError("TEST fusion config is outside the predeclared DEV-selection grid")
    config = lock.get("rrf", {})
    if int(config.get("k", -1)) != rrf_k or tuple(config.get("weights", ())) != weights:
        raise ValueError("TEST fusion weights differ from the DEV-selected final lock")


def read_committed_lock(repo_root: Path = ROOT) -> dict[str, Any]:
    relative = LOCK_PATH.relative_to(repo_root).as_posix()
    subprocess.run(
        ["git", "-C", str(repo_root), "ls-files", "--error-unmatch", relative],
        check=True,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain", "--", relative],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    validate_test_lock(lock, committed=not status)
    head = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not lock.get("code_commit"):
        raise ValueError("final method lock is missing its source-code commit identity")
    diff = subprocess.run(
        [
            "git", "-C", str(repo_root), "diff", "--quiet", lock["code_commit"], head,
            "--", *SPRINT_LOCKED_PATHS,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if diff.returncode != 0:
        raise ValueError("R2MED sprint source changed after the frozen method lock")
    for path in SPRINT_LOCKED_PATHS:
        subprocess.run(
            ["git", "-C", str(repo_root), "ls-files", "--error-unmatch", path],
            check=True,
            capture_output=True,
            text=True,
        )
    changed = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain", "--", *SPRINT_LOCKED_PATHS],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if changed:
        raise ValueError("R2MED sprint source files have uncommitted changes")
    baseline_path = repo_root / "runs/rag_r2med_crb/dev/base_retrieval_manifest.json"
    baseline_relative = baseline_path.relative_to(repo_root).as_posix()
    subprocess.run(
        ["git", "-C", str(repo_root), "ls-files", "--error-unmatch", baseline_relative],
        check=True,
        capture_output=True,
        text=True,
    )
    if subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain", "--", baseline_relative],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip():
        raise ValueError("committed DEV base-retrieval manifest is modified")
    if hashlib.sha256(baseline_path.read_bytes()).hexdigest() != lock.get("base_retrieval_manifest_sha256"):
        raise ValueError("committed DEV base-retrieval manifest hash differs from the lock")
    frozen_manifests = lock.get("generation_manifests", {})
    for method in METHODS:
        relative_manifest = f"runs/rag_r2med_crb/generation/dev/{method}/generation_manifest.json"
        manifest_path = repo_root / relative_manifest
        subprocess.run(
            ["git", "-C", str(repo_root), "ls-files", "--error-unmatch", relative_manifest],
            check=True,
            capture_output=True,
            text=True,
        )
        if subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain", "--", relative_manifest],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip():
            raise ValueError(f"committed DEV generation manifest is modified: {method}")
        expected_hash = frozen_manifests.get(method, {}).get("manifest_sha256")
        if not expected_hash or hashlib.sha256(manifest_path.read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"committed DEV generation manifest hash differs from the lock: {method}")
    return lock


def _read_generation(method: str, subset, source_sha256: str) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    manifest_path = TEST_GENERATION_ROOT / method / "generation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_manifest = load_source_manifest(SOURCE_MANIFEST_PATH)
    if (
        manifest.get("partition") != "TEST"
        or manifest.get("method") != method
        or manifest.get("dataset_source_manifest_sha256") != source_sha256
        or manifest.get("call_count") != 303
        or manifest.get("generation_config") != GENERATION_CONFIG
        or manifest.get("generator", {}).get("sha256")
        != "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785"
        or manifest.get("upstream_commit") != source_manifest["upstream"]["commit"]
    ):
        raise ValueError(f"TEST generation manifest identity mismatch: {method}")
    subset_entry = next((entry for entry in manifest["subsets"] if entry["subset"] == subset.name), None)
    if subset_entry is None or subset_entry["query_count"] != len(subset.queries):
        raise ValueError(f"TEST generation manifest lacks {subset.name}")
    source_entry = next(
        entry for entry in source_manifest["datasets"]["TEST"] if entry["name"] == subset.name
    )
    expected_order_sha = hashlib.sha256(
        "\n".join(query.query_id for query in subset.queries).encode("utf-8")
    ).hexdigest()
    if (
        subset_entry.get("query_file_sha256") != source_entry["files"]["query.jsonl"]["sha256"]
        or subset_entry.get("corpus_file_sha256") != source_entry["files"]["corpus.jsonl"]["sha256"]
        or subset_entry.get("query_order_sha256") != expected_order_sha
    ):
        raise ValueError(f"TEST generation input identity mismatch: {method}/{subset.name}")
    path = Path(subset_entry["artifact"])
    expected_path = TEST_GENERATED_VIEWS / method / f"{subset.name}.jsonl"
    if path.resolve() != expected_path.resolve():
        raise ValueError(f"TEST generation artifact is outside the frozen output path: {path}")
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != subset_entry["artifact_sha256"]:
        raise ValueError(f"TEST generation artifact hash mismatch: {method}/{subset.name}")
    rows = _read_jsonl(path)
    if [row.get("query_id") for row in rows] != [query.query_id for query in subset.queries]:
        raise ValueError(f"TEST generation query order mismatch: {method}/{subset.name}")
    if any(row.get("method") != method or row.get("subset") != subset.name for row in rows):
        raise ValueError(f"TEST generation method/subset mismatch: {method}/{subset.name}")
    return {row["query_id"]: row for row in rows}, manifest


def _ensure_test_generations(
    methods: tuple[str, ...],
    *,
    source_root: Path,
    upstream_root: Path,
    server_executable: Path,
    source_sha: str,
) -> dict[str, dict[str, Any]]:
    from tools.generate_r2med_gar import generate_method

    subsets = load_partition_inputs("TEST", source_root=source_root)
    stats: dict[str, dict[str, Any]] = {}
    for method in methods:
        manifest_path = TEST_GENERATION_ROOT / method / "generation_manifest.json"
        if not manifest_path.is_file():
            summary = generate_method(
                "TEST",
                method,
                upstream_root=upstream_root,
                source_root=source_root,
                server_executable=server_executable,
            )
        else:
            summary = json.loads(manifest_path.read_text(encoding="utf-8"))
        if summary.get("dataset_source_manifest_sha256") != source_sha:
            raise ValueError(f"stale TEST generation manifest: {method}")
        stats[method] = {
            "call_count": summary["call_count"],
            "valid_count": summary["valid_output_count"],
            "failure_count": summary["failure_count"],
            "completion_count": summary["completion_count"],
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "subset_artifact_sha256": {
                entry["subset"]: entry["artifact_sha256"] for entry in summary["subsets"]
            },
        }
        for subset in subsets:
            _read_generation(method, subset, source_sha)
    return stats


def _evaluate_test(
    *,
    lock: dict[str, Any],
    source_root: Path,
    bge_root: Path,
    upstream_root: Path,
    server_executable: Path,
) -> dict[str, Any]:
    if TEST_REPORT_PATH.exists() or TEST_RUN_MARKER.exists() or TEST_BASELINE_REPORT.exists():
        raise FileExistsError("R2MED TEST already started or has artifacts; refusing a second TEST execution")
    source_manifest = load_source_manifest(SOURCE_MANIFEST_PATH)
    source_sha = hashlib.sha256(SOURCE_MANIFEST_PATH.read_bytes()).hexdigest()
    dev_report = json.loads(DEV_REPORT_PATH.read_text(encoding="utf-8"))
    if hashlib.sha256(DEV_REPORT_PATH.read_bytes()).hexdigest() != lock.get("dev_result_sha256"):
        raise ValueError("DEV report no longer matches the committed final method lock")
    if lock.get("source_manifest_sha256") != source_sha:
        raise ValueError("source manifest no longer matches the committed final method lock")
    verified = verify_models(bge_root=bge_root)
    expected_test_paths = [
        TEST_RANKING_ROOT / subset / filename
        for subset in PARTITIONS["TEST"]
        for filename in ("bm25_original.jsonl", "bge_large_original.jsonl")
    ]
    if any(path.exists() for path in expected_test_paths):
        raise FileExistsError("partial TEST baseline ranking artifacts exist; inspect before proceeding")
    TEST_RUN_MARKER.parent.mkdir(parents=True, exist_ok=True)
    marker = {
        "started_at_utc": datetime.now(UTC).isoformat(),
        "final_method_lock_sha256": hashlib.sha256(LOCK_PATH.read_bytes()).hexdigest(),
        "code_commit": lock["code_commit"],
        "test_status": "PUBLIC_BENCHMARK_REUSED",
    }
    with TEST_RUN_MARKER.open("x", encoding="utf-8") as handle:
        json.dump(marker, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    run_partition_baselines("TEST", source_root, bge_root)

    gar_baseline_key = dev_report["strongest_cost_matched_gar"]["method"]
    gar_baseline_method = resolve_dev_gar_baseline(dev_report)
    crb_method = lock["crb_variant"]
    generation_methods = (*GAR_METHODS, crb_method)
    generation_stats = _ensure_test_generations(
        generation_methods,
        source_root=source_root,
        upstream_root=upstream_root,
        server_executable=server_executable,
        source_sha=source_sha,
    )
    subsets = load_partition_inputs("TEST", source_root=source_root)
    views: dict[str, dict[str, dict[str, dict[str, Any]]]] = {
        method: {} for method in generation_methods
    }
    for method in generation_methods:
        for subset in subsets:
            views[method][subset.name], _ = _read_generation(method, subset, source_sha)

    os.environ["HF_HOME"] = str(E_ROOT / "cache/huggingface")
    os.environ["HF_HUB_CACHE"] = str(E_ROOT / "cache/huggingface/hub")
    os.environ["TRANSFORMERS_CACHE"] = str(E_ROOT / "cache/huggingface/transformers")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(bge_root), device="cuda", local_files_only=True)
    model.max_seq_length = 512
    model.eval()

    single_rankings: dict[str, dict[str, dict[str, list[str]]]] = {
        name: {} for name in (
            "bm25", "bge_large", "hyde_bm25", "query2doc_bm25", "lamer_bm25",
            "hyde_bge_large", "query2doc_bge_large", "lamer_bge_large",
        )
    }
    mv_channels: dict[str, dict[str, dict[str, tuple[list[str], list[str], list[str], list[str]]]]] = {
        method: {} for method in generation_methods
    }

    for subset in subsets:
        query_ids = [query.query_id for query in subset.queries]
        query_texts = [query.text for query in subset.queries]
        documents = [(doc.doc_id, doc.text) for doc in subset.documents]
        dense_doc_ids = [doc.doc_id for doc in subset.dense_documents]
        base_root = TEST_RANKING_ROOT / subset.name
        bm25_original = _read_ranking(base_root / "bm25_original.jsonl")
        bge_original = _read_ranking(base_root / "bge_large_original.jsonl")
        single_rankings["bm25"][subset.name] = bm25_original
        single_rankings["bge_large"][subset.name] = bge_original
        bm25_index = LuceneBM25Index(documents)
        corpus_vectors, _ = _load_or_encode_corpus(
            model, subset, verified["bge_large"]["weights_sha256"], "TEST"
        )
        original_vectors = encode_bge(
            model, [BGE_QUERY_PREFIX + text for text in query_texts], batch_size=32
        )

        for method in generation_methods:
            method_views = views[method][subset.name]
            generated = [method_views[query_id]["generated_text"] for query_id in query_ids]
            if method in {"crb_q", "crb_prf"}:
                dense_texts = [
                    str((method_views[query_id].get("structured") or {}).get("pseudo_evidence", query_texts[index]))
                    for index, query_id in enumerate(query_ids)
                ]
                expanded = [
                    crb_lexical_text(method_views[query_id].get("structured") or {}) or query_texts[index]
                    for index, query_id in enumerate(query_ids)
                ]
            else:
                dense_texts = generated
                expanded = [
                    bm25_query_text(method, query, gen)
                    for query, gen in zip(query_texts, generated, strict=True)
                ]
            sparse_rows = [bm25_index.search(text, top_k=100) for text in expanded]
            gen_vectors = encode_bge(
                model, [BGE_QUERY_PREFIX + text for text in dense_texts], batch_size=32
            )
            dense_gen_rows = dense_search_many(
                gen_vectors, corpus_vectors, dense_doc_ids, top_k=100, device="cuda"
            )
            sparse_by_id = {
                query_id: [row.doc_id for row in rows]
                for query_id, rows in zip(query_ids, sparse_rows, strict=True)
            }
            dense_gen_by_id = {
                query_id: [row.doc_id for row in rows]
                for query_id, rows in zip(query_ids, dense_gen_rows, strict=True)
            }
            if method in GAR_METHODS:
                if method == "query2doc":
                    one_view_vectors = encode_bge(
                        model,
                        [
                            BGE_QUERY_PREFIX + f"{query}[SEP]{gen}"
                            for query, gen in zip(query_texts, generated, strict=True)
                        ],
                        batch_size=32,
                    )
                else:
                    one_view_vectors = (original_vectors + gen_vectors) / 2.0
                one_view_dense = dense_search_many(
                    one_view_vectors, corpus_vectors, dense_doc_ids, top_k=100, device="cuda"
                )
                prefix = method
                single_rankings[f"{prefix}_bm25"][subset.name] = sparse_by_id
                single_rankings[f"{prefix}_bge_large"][subset.name] = {
                    query_id: [row.doc_id for row in rows]
                    for query_id, rows in zip(query_ids, one_view_dense, strict=True)
                }
            mv_channels[method][subset.name] = {
                query_id: (
                    bm25_original[query_id],
                    sparse_by_id[query_id],
                    bge_original[query_id],
                    dense_gen_by_id[query_id],
                )
                for query_id in query_ids
            }
            _write_rankings(
                TEST_RANKING_ROOT / subset.name / f"{method}_bm25_bridge.jsonl",
                _ranking_rows(subset.name, f"{method}_bm25_bridge", query_ids, sparse_rows),
            )
            _write_rankings(
                TEST_RANKING_ROOT / subset.name / f"{method}_bge_generated.jsonl",
                _ranking_rows(subset.name, f"{method}_bge_generated", query_ids, dense_gen_rows),
            )
            del gen_vectors, dense_gen_rows
        print(f"TEST retrieval channels complete: {subset.name} ({len(query_ids)} queries)")

    del model
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    single_summaries: dict[str, dict[str, Any]] = {}
    single_query_rows: dict[str, list[dict[str, Any]]] = {}
    for method, rankings in single_rankings.items():
        rows, summary = evaluate_rankings("TEST", rankings, source_root=source_root)
        single_query_rows[method] = rows
        single_summaries[method] = summary

    mv_configs = dev_multiview_configs(dev_report)
    for method, config in mv_configs.items():
        allowed = next(
            (
                candidate
                for candidate in FUSION_CONFIGS
                if candidate["config_id"] == config.get("config_id")
            ),
            None,
        )
        if allowed is None or int(config["rrf_k"]) != allowed["rrf_k"] or tuple(config["weights"]) != tuple(allowed["weights"]):
            raise ValueError(f"DEV-selected TEST fusion config is invalid for {method}")
    frozen_crb = lock["rrf"]
    assert_frozen_weights(
        lock,
        rrf_k=int(frozen_crb["k"]),
        weights=tuple(int(weight) for weight in frozen_crb["weights"]),
    )
    mv_configs[crb_method] = {
        "config_id": frozen_crb["config_id"],
        "rrf_k": int(frozen_crb["k"]),
        "weights": tuple(int(weight) for weight in frozen_crb["weights"]),
    }
    mv_summaries: dict[str, dict[str, Any]] = {}
    mv_query_rows: dict[str, list[dict[str, Any]]] = {}
    for method, config in mv_configs.items():
        rankings = {
            subset: {
                query_id: [
                    doc.doc_id
                    for doc in weighted_rrf(
                        channels,
                        rrf_k=int(config["rrf_k"]),
                        weights=tuple(config["weights"]),
                    )
                ]
                for query_id, channels in per_query.items()
            }
            for subset, per_query in mv_channels[method].items()
        }
        rows, summary = evaluate_rankings("TEST", rankings, source_root=source_root)
        mv_query_rows[method] = rows
        mv_summaries[method] = summary
        for subset_name, query_rankings in rankings.items():
            _write_rankings(
                TEST_RANKING_ROOT / subset_name / f"{method}_mv_frozen.jsonl",
                [
                    {
                        "subset": subset_name,
                        "query_id": query_id,
                        "method": f"{method}_mv_frozen",
                        "ranking": [{"doc_id": doc_id, "score": 0.0} for doc_id in ranking],
                    }
                    for query_id, ranking in query_rankings.items()
                ],
            )

    primary_base_rows = {row["query_id"]: row for row in mv_query_rows[gar_baseline_method]}
    crb_rows = {row["query_id"]: row for row in mv_query_rows[crb_method]}
    subset_by_query = {row["query_id"]: row["subset"] for row in primary_base_rows.values()}
    bootstrap = paired_stratified_bootstrap(
        crb_rows,
        primary_base_rows,
        subset_by_query,
        metric="ndcg@10",
        resamples=10_000,
    )
    crb_macro = mv_summaries[crb_method]["macro_equal_subset_weight"]
    baseline_macro = mv_summaries[gar_baseline_method]["macro_equal_subset_weight"]
    delta_ndcg = float(crb_macro["ndcg@10"]) - float(baseline_macro["ndcg@10"])
    delta_mrr = float(crb_macro["mrr@10"]) - float(baseline_macro["mrr@10"])
    delta_recall10 = float(crb_macro["recall@10"]) - float(baseline_macro["recall@10"])
    point_improvement = delta_ndcg > 0
    strong_win = delta_ndcg >= 0.010 and float(bootstrap["lower_95"]) > 0

    report = {
        "schema_version": "r2med-crb-test-report-v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "partition": "TEST",
        "test_status": source_manifest["test_status"],
        "source_manifest_sha256": source_sha,
        "final_method_lock_sha256": hashlib.sha256(LOCK_PATH.read_bytes()).hexdigest(),
        "dev_report_sha256": lock["dev_result_sha256"],
        "code_commit": lock["code_commit"],
        "verified_models": verified,
        "generation_audit": generation_stats,
        "single_view": single_summaries,
        "multi_view": {
            method: {"frozen_config": mv_configs[method], "summary": summary}
            for method, summary in mv_summaries.items()
        },
        "primary_comparison": {
            "candidate": crb_method,
            "baseline": gar_baseline_key,
            "baseline_generation_method": gar_baseline_method,
            "delta_ndcg_at_10": delta_ndcg,
            "delta_mrr_at_10": delta_mrr,
            "delta_recall_at_10": delta_recall10,
            "paired_subset_stratified_bootstrap": bootstrap,
        },
        "gates": {
            "public_benchmark_point_improvement": point_improvement,
            "public_benchmark_improvement": strong_win,
            "resume_headline_ready": strong_win,
        },
        "gold_boundary": "Generation and ranking consumed only query/corpus and query-time BM25 top-10 feedback; qrels were read only by the evaluator after the DEV-locked TEST run began.",
        "test_execution_count": 1,
    }
    TEST_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEST_REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--bge-root", type=Path, default=DEFAULT_BGE_ROOT)
    parser.add_argument("--upstream-root", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--llama-server", type=Path, default=DEFAULT_SERVER)
    args = parser.parse_args()
    lock = read_committed_lock()
    if args.preflight_only:
        print(f"TEST lock verified: {lock['crb_variant']} / {lock['rrf']['config_id']}")
        return
    result = _evaluate_test(
        lock=lock,
        source_root=args.source_root,
        bge_root=args.bge_root,
        upstream_root=args.upstream_root,
        server_executable=args.llama_server,
    )
    print(json.dumps({"comparison": result["primary_comparison"], "gates": result["gates"]}, indent=2))


if __name__ == "__main__":
    main()
