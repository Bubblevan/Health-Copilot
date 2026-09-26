"""Frozen public R2MED TEST arm definitions and artifact integrity helpers."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

TEST_COUNTS = {"MedQA-Diag": 118, "MedXpertQA-Exam": 97, "Medical-Sciences": 88}
TEST_TOTAL = 303
LOCK_PATH = Path("runs/rag_r2med_final_test/final_eval_lock.json")
START_MARKER_PATH = Path("runs/rag_r2med_final_test/test_run_started.json")
REPORT_PATH = Path("runs/rag_r2med_final_test/test_report.json")
CANDIDATE_PATH = Path("runs/rag_r2med_final_test/candidate_analysis.json")
ARTIFACT_ROOT = Path(r"E:\Health-Copilot-RAG\r2med\final_test")

FROZEN_ARMS: dict[str, dict[str, Any]] = {
    "B0_BM25": {"name": "BM25", "kind": "lucene_bm25", "top_k": 100, "k1": 0.9, "b": 0.4},
    "B1_BGE_large": {
        "name": "BGE-large",
        "kind": "bge_cosine",
        "top_k": 100,
        "revision": "d4aa6901d3a41ba39fb536a557fa166f842b0e09",
        "weights_sha256": "45e1954914e29bd74080e6c1510165274ff5279421c89f76c418878732f64ae7",
    },
    "B2_BM25_BGE_RRF": {
        "name": "BM25+BGE RRF",
        "kind": "two_source_rrf",
        "input_depth": 100,
        "rrf_k": 60,
        "weights": [1, 1],
        "output_depth": 100,
    },
    "B3_LameR_MV": {
        "name": "LameR-MV",
        "kind": "four_view_gar_rrf",
        "generation_method": "lamer",
        "feedback_depth": 10,
        "top_k": 100,
        "rrf_k": 20,
        "weights": [1, 2, 1, 2],
        "config_id": "k20-W2",
    },
    "B4_Compact_CRB_Q": {
        "name": "Compact CRB-Q",
        "kind": "four_view_gar_rrf",
        "generation_method": "crb_q_compact_repair",
        "top_k": 100,
        "rrf_k": 20,
        "weights": [1, 1, 1, 1],
        "config_id": "k20-W0",
        "prompt_sha256": "7d0bb8159d4d089f723ca306c47f3585cf9cd8c946495e625336cf01bc3e3b24",
        "schema": {"required_keys": ["q", "t", "e"], "max_output_tokens": 256},
    },
    "B5_DualSource_RRF_lam0.5": {
        "name": "DualSource-RRF λ=.5",
        "kind": "two_source_rrf",
        "lamer_weight": 1.0,
        "crb_weight": 0.5,
        "rrf_k": 60,
        "input_depth": 100,
        "keep_full_distinct_union": True,
        "tie_break": ["score_desc", "lamer_rank_asc_null_last", "crb_rank_asc_null_last", "doc_id_asc"],
    },
}

QWEN_IDENTITY = {
    "repo": "Qwen/Qwen3-8B-GGUF",
    "revision": "6a569868d07d3bd59e8b97fb001bf8c0b254bb20",
    "file": "Qwen3-8B-Q4_K_M.gguf",
    "bytes": 5_027_783_488,
    "sha256": "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785",
}
GENERATION_CONFIG = {
    "temperature": 0.0,
    "reasoning": "disabled",
    "max_output_tokens": 256,
    "calls_per_query_per_method": 1,
    "retry_on_error": False,
    "endpoint_bind": "127.0.0.1",
}
BOOTSTRAP = {"resamples": 10_000, "seed": 20260926, "stratified_by": "TEST subset", "interpretation": "exploratory paired uncertainty interval"}
LOCKED_CODE_PATHS = (
    "eval/r2med_final_test.py",
    "eval/r2med_crb_data.py",
    "eval/r2med_crb_evaluator.py",
    "eval/r2med_crb.py",
    "eval/r2med_gar_generation.py",
    "eval/r2med_multiview.py",
    "eval/r2med_candidate_union.py",
    "tools/freeze_r2med_final_eval.py",
    "tools/run_r2med_final_test.py",
    "tools/analyze_r2med_final_test.py",
    "tools/run_r2med_baselines.py",
    "tools/generate_r2med_gar.py",
    "tools/generate_r2med_crb_compact_repair.py",
    "tools/verify_r2med_models.py",
    "tests/test_r2med_final_test.py",
    "runs/rag_r2med_crb/source_manifest.json",
    "runs/rag_r2med_rerank/protocol.json",
    "runs/rag_r2med_rerank/dev_report.json",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(encoded)


def validate_final_lock(lock: Mapping[str, Any]) -> None:
    if lock.get("schema_version") != "r2med-final-public-test-lock-v1":
        raise ValueError("unknown R2MED final TEST lock schema")
    if lock.get("purpose") != "PUBLIC_TEST_BASELINE_EVALUATION_ONLY":
        raise ValueError("final TEST lock purpose is not the frozen public baseline evaluation")
    if lock.get("test_status") != "PUBLIC_BENCHMARK_REUSED":
        raise ValueError("R2MED TEST must be disclosed as public and previously reused")
    if lock.get("test_counts") != {**TEST_COUNTS, "total": TEST_TOTAL}:
        raise ValueError("final TEST split counts drifted")
    if lock.get("arms") != FROZEN_ARMS:
        raise ValueError("final TEST arm configuration differs from the frozen DEV-selected recipes")
    if lock.get("generator", {}).get("identity") != QWEN_IDENTITY:
        raise ValueError("final TEST Qwen model identity drifted")
    if lock.get("generator", {}).get("config") != GENERATION_CONFIG:
        raise ValueError("final TEST generation config drifted")
    calls = lock.get("generator", {}).get("expected_calls", {})
    if calls != {"lamer": TEST_TOTAL, "compact_crb_q": TEST_TOTAL, "total": 606}:
        raise ValueError("final TEST generation call budget drifted")
    if lock.get("generator", {}).get("paid_api_calls") != 0:
        raise ValueError("final TEST must use localhost generation only")
    if not isinstance(lock.get("generator", {}).get("llama_cpp_version"), str) or not lock["generator"]["llama_cpp_version"].strip():
        raise ValueError("final TEST lock must pin the llama.cpp runtime version")
    if lock.get("bootstrap") != BOOTSTRAP:
        raise ValueError("final TEST bootstrap config drifted")
    if lock.get("evaluation", {}).get("primary_metric") != "equal-subset macro nDCG@10":
        raise ValueError("final TEST primary metric drifted")
    if lock.get("evaluation", {}).get("qrels_boundary") != "after_all_six_rankings_are_frozen":
        raise ValueError("final TEST gold-isolation boundary drifted")
    if set(lock.get("code", {}).get("files_sha256", {})) != set(LOCKED_CODE_PATHS):
        raise ValueError("final TEST lock does not bind every execution-critical file")


def read_lock(path: Path = LOCK_PATH) -> dict[str, Any]:
    lock = json.loads(path.read_text(encoding="utf-8"))
    validate_final_lock(lock)
    return lock


def append_jsonl_once(path: Path, record: Mapping[str, Any], *, expected_method: str, expected_prompt_sha: str) -> None:
    """Append one completed query result; refuses duplicate IDs and syncs before returning."""
    query_id = record.get("query_id")
    if not isinstance(query_id, str) or not query_id:
        raise ValueError("generation result requires a query_id")
    if record.get("method") != expected_method or record.get("prompt_sha256") != expected_prompt_sha:
        raise ValueError("generation checkpoint method or prompt identity drifted")
    existing = read_jsonl_records(path) if path.exists() else []
    if any(row.get("query_id") == query_id for row in existing):
        raise ValueError(f"generation query already completed; refusing a second call: {query_id}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(record), ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"invalid JSONL object at {path}:{line_number}")
            result.append(value)
    return result


def validate_complete_generation(
    rows: Sequence[Mapping[str, Any]],
    expected_query_ids: Sequence[str],
    *,
    expected_method: str,
    expected_prompt_sha: str,
) -> None:
    actual_ids = [str(row.get("query_id", "")) for row in rows]
    if actual_ids != list(expected_query_ids):
        raise ValueError("generation artifact is not complete in frozen query order")
    if any(row.get("method") != expected_method or row.get("prompt_sha256") != expected_prompt_sha for row in rows):
        raise ValueError("generation artifact contains a method/prompt mismatch")
    if any(row.get("error") not in (None, "invalid_compact_json_or_constraints") for row in rows):
        raise ValueError("generation artifact includes an infrastructure failure as a completed query")


def validate_rankings(
    rankings: Mapping[str, Sequence[str]], expected_query_ids: Iterable[str], corpus_ids: set[str]
) -> None:
    if set(rankings) != set(expected_query_ids):
        raise ValueError("ranking query IDs do not match the frozen TEST query set")
    for query_id, docs in rankings.items():
        if not docs or len(docs) != len(set(docs)):
            raise ValueError(f"ranking is empty or contains duplicate document IDs: {query_id}")
        unknown = set(docs) - corpus_ids
        if unknown:
            raise ValueError(f"ranking contains documents outside the corpus: {query_id}")


def validate_generation_budget(lamer_rows: Sequence[Mapping[str, Any]], crb_rows: Sequence[Mapping[str, Any]]) -> None:
    if len(lamer_rows) != TEST_TOTAL or len(crb_rows) != TEST_TOTAL:
        raise ValueError("the frozen TEST requires exactly 303 generated outputs per method")
    if len({row.get("query_id") for row in lamer_rows}) != TEST_TOTAL:
        raise ValueError("LameR did not produce exactly one completed row per TEST query")
    if len({row.get("query_id") for row in crb_rows}) != TEST_TOTAL:
        raise ValueError("compact CRB did not produce exactly one completed row per TEST query")


def resume_claim_gates(
    summaries: Mapping[str, Mapping[str, Any]], comparisons: Mapping[str, Mapping[str, Any]]
) -> dict[str, bool]:
    scores = {key: float(value["macro_equal_subset_weight"]["ndcg@10"]) for key, value in summaries.items()}
    dual = scores["B5_DualSource_RRF_lam0.5"]
    bm25 = scores["B0_BM25"]
    ordinary = scores["B2_BM25_BGE_RRF"]
    lamer = scores["B3_LameR_MV"]
    dual_bm25 = comparisons["DualSource_vs_BM25"]
    dual_rrf = comparisons["DualSource_vs_ordinary_RRF"]
    dual_lamer = comparisons["DualSource_vs_LameR_MV"]
    basic = dual > bm25 and dual > ordinary and dual_bm25["lower_95"] > 0 and dual_rrf["lower_95"] > 0
    strong_basic = basic and dual - ordinary >= 0.020
    point_lamer = dual > lamer
    strong_method = point_lamer and dual - lamer >= 0.005 and dual_lamer["lower_95"] > 0
    return {
        "PUBLIC_BASIC_BASELINE_IMPROVEMENT": basic,
        "STRONG_BASIC_BASELINE_IMPROVEMENT": strong_basic,
        "POINT_IMPROVEMENT_OVER_LAMER": point_lamer,
        "STRONG_BASELINE_IMPROVEMENT": strong_method,
        "PUBLIC_R2MED_TEST_EVIDENCE_READY": basic,
        "RESUME_BASIC_BASELINE_HEADLINE_READY": basic,
        "RESUME_STRONG_METHOD_HEADLINE_READY": strong_method,
        "R2MED_FINAL_CLOSEOUT": True,
    }


def generation_statistics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    if total == 0:
        raise ValueError("cannot summarize empty generation artifacts")
    return {
        "query_count": total,
        "completed_count": sum(bool(row.get("completed")) for row in rows),
        "valid_count": sum(bool(row.get("valid")) for row in rows),
        "fallback_count": sum(bool(row.get("fallback_original")) for row in rows),
        "truncation_count": sum(bool(row.get("truncated")) for row in rows),
        "valid_rate": sum(bool(row.get("valid")) for row in rows) / total,
        "calls_per_query": 1,
    }
