"""Create the R2MED final public TEST lock without opening TEST data files."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.r2med_crb_data import SOURCE_MANIFEST_PATH, load_source_manifest
from eval.r2med_final_test import (
    BOOTSTRAP,
    FROZEN_ARMS,
    GENERATION_CONFIG,
    LOCK_PATH,
    LOCKED_CODE_PATHS,
    QWEN_IDENTITY,
    TEST_COUNTS,
    TEST_TOTAL,
    canonical_json_sha256,
    sha256_bytes,
    validate_final_lock,
)
from eval.r2med_gar_generation import load_upstream_prompt_catalog, prompt_sha256
from tools.generate_r2med_crb_compact_repair import (
    COMPACT_JSON_SCHEMA,
    COMPACT_PROMPT_TEMPLATE,
    _sha256_text,
)
from tools.verify_r2med_models import E_ROOT, verify_models

DEFAULT_UPSTREAM = Path(r"D:\MyLab\Jianli\external\rag\R2MED")
DEV_RERANK_REPORT = ROOT / "runs/rag_r2med_rerank/dev_report.json"
DEV_RERANK_PROTOCOL = ROOT / "runs/rag_r2med_rerank/protocol.json"
COMPACT_DEV_MANIFEST = ROOT / "runs/rag_r2med_crb/compact_repair/generation/generation_manifest.json"


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _assert_clean_bound_files() -> None:
    status = _git("status", "--porcelain", "--", *LOCKED_CODE_PATHS)
    if status:
        raise ValueError(f"execution-critical files must be committed and clean before lock creation:\n{status}")
    missing = [path for path in LOCKED_CODE_PATHS if not (ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError(f"cannot bind missing execution-critical files: {missing}")


def _require_dev_selection(dev: dict[str, Any], protocol: dict[str, Any]) -> None:
    selection = dev.get("selection", {})
    if selection.get("best_ours_config", {}).get("lambda") != 0.5:
        raise ValueError("frozen DEV winner is not DualSource-RRF lambda 0.5")
    if selection.get("best_ours") != "DualSource_RRF_lam0.5":
        raise ValueError("unexpected frozen DEV DualSource selection")
    if selection.get("strongest_non_ours") != "B4_LameR_MV":
        raise ValueError("LameR-MV is no longer the frozen strongest reproduced method")
    if selection.get("dev_signal") != "NEGATIVE" or selection.get("test_allowed") is not False:
        raise ValueError("the old method-development gate must remain recorded as negative")
    if protocol.get("test_candidate_reproduction", {}).get("test_model_calls_expected") != 606:
        raise ValueError("the pinned DEV protocol does not declare the expected 606 local calls")
    if protocol.get("candidate_sources", {}).get("fusion", {}).get("lambda_grid") != [0.5, 1.0, 2.0]:
        raise ValueError("the recorded DEV fusion grid differs from the frozen source protocol")


def build_lock(upstream_root: Path) -> dict[str, Any]:
    _assert_clean_bound_files()
    source_bytes = SOURCE_MANIFEST_PATH.read_bytes()
    source_manifest = load_source_manifest(SOURCE_MANIFEST_PATH)
    if source_manifest.get("test_status") != "PUBLIC_BENCHMARK_REUSED":
        raise ValueError("TEST must be disclosed as public and previously reused")
    test_entries = {entry["name"]: entry for entry in source_manifest["datasets"]["TEST"]}
    if set(test_entries) != set(TEST_COUNTS):
        raise ValueError("source manifest TEST subset set does not match the frozen sprint")
    for subset, count in TEST_COUNTS.items():
        entry = test_entries[subset]
        if entry.get("query_count") != count or entry.get("role") != "TEST_UNOPENED":
            raise ValueError(f"unexpected TEST source-manifest identity for {subset}")
        if not {"corpus.jsonl", "query.jsonl"}.issubset(entry.get("files", {})):
            raise ValueError(f"source manifest lacks query/corpus identities for {subset}")

    dev_report = json.loads(DEV_RERANK_REPORT.read_text(encoding="utf-8"))
    protocol = json.loads(DEV_RERANK_PROTOCOL.read_text(encoding="utf-8"))
    _require_dev_selection(dev_report, protocol)
    compact_manifest = json.loads(COMPACT_DEV_MANIFEST.read_text(encoding="utf-8"))
    if (
        compact_manifest.get("partition") != "DEV"
        or compact_manifest.get("test_accessed") is not False
        or compact_manifest.get("compact_prompt_sha256") != FROZEN_ARMS["B4_Compact_CRB_Q"]["prompt_sha256"]
        or compact_manifest.get("generator", {}).get("sha256") != QWEN_IDENTITY["sha256"]
    ):
        raise ValueError("compact CRB-Q DEV generation does not match the frozen TEST recipe")
    if compact_manifest.get("attempted_query_count") != 393:
        raise ValueError("compact CRB-Q DEV generation count differs from the recorded split")

    verified = verify_models()
    if verified["qwen"]["verified_sha256"] != QWEN_IDENTITY["sha256"]:
        raise ValueError("local Qwen artifact does not match the frozen model identity")
    prompt_catalog = load_upstream_prompt_catalog(upstream_root)
    source_files = source_manifest["upstream"]["files_sha256"]
    lamer_prompt_sha = {
        subset: prompt_sha256(prompt_catalog["lamer"][source_manifest["prompt_family_mapping"][subset]])
        for subset in TEST_COUNTS
    }
    compact_prompt_sha = _sha256_text(COMPACT_PROMPT_TEMPLATE)
    if compact_prompt_sha != FROZEN_ARMS["B4_Compact_CRB_Q"]["prompt_sha256"]:
        raise ValueError("local compact CRB prompt differs from its frozen DEV SHA")

    files_sha256 = {
        path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in LOCKED_CODE_PATHS
    }
    test_identity = {
        subset: {
            "repo_id": entry["repo_id"],
            "revision": entry["revision"],
            "directory": entry["directory"],
            "query_count": entry["query_count"],
            "corpus_document_count": entry["corpus_document_count"],
            "query_sha256": entry["files"]["query.jsonl"]["sha256"],
            "query_bytes": entry["files"]["query.jsonl"]["bytes"],
            "corpus_sha256": entry["files"]["corpus.jsonl"]["sha256"],
            "corpus_bytes": entry["files"]["corpus.jsonl"]["bytes"],
        }
        for subset, entry in test_entries.items()
    }
    lock = {
        "schema_version": "r2med-final-public-test-lock-v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "purpose": "PUBLIC_TEST_BASELINE_EVALUATION_ONLY",
        "claim_boundary": {
            "new_method_dev_win": False,
            "allowed_question": "whether frozen pipelines improve over basic public retrieval baselines",
            "prohibited_claim": "DualSource outperforms strongest GAR unless TEST strong-method gate passes",
            "test_status": "public_benchmark_reused_not_untouched_or_confirmatory",
        },
        "test_status": "PUBLIC_BENCHMARK_REUSED",
        "test_counts": {**TEST_COUNTS, "total": TEST_TOTAL},
        "source": {
            "manifest_path": "runs/rag_r2med_crb/source_manifest.json",
            "manifest_sha256": sha256_bytes(source_bytes),
            "upstream_commit": source_manifest["upstream"]["commit"],
            "upstream_files_sha256": source_files,
            "test_query_corpus_identity": test_identity,
            "prompt_family_mapping": source_manifest["prompt_family_mapping"],
        },
        "models": {
            "generator": QWEN_IDENTITY,
            "dense_retriever": {
                "repo": "BAAI/bge-large-en-v1.5",
                "revision": verified["bge_large"]["revision"],
                "weights_sha256": verified["bge_large"]["weights_sha256"],
                "core_files": verified["bge_large"]["core_files"],
            },
        },
        "generator": {
            "identity": QWEN_IDENTITY,
            "config": GENERATION_CONFIG,
            "llama_cpp_version": compact_manifest.get("generator", {}).get("llama_cpp_version"),
            "expected_calls": {"lamer": TEST_TOTAL, "compact_crb_q": TEST_TOTAL, "total": 606},
            "paid_api_calls": 0,
        },
        "prompts": {
            "lamer_upstream_template_sha256_by_subset": lamer_prompt_sha,
            "compact_crb_template_sha256": compact_prompt_sha,
            "compact_schema_sha256": canonical_json_sha256(COMPACT_JSON_SCHEMA),
            "compact_dev_generation_manifest_sha256": hashlib.sha256(COMPACT_DEV_MANIFEST.read_bytes()).hexdigest(),
        },
        "arms": FROZEN_ARMS,
        "metrics": {
            "primary_metric": "equal-subset macro nDCG@10",
            "secondary_metrics": ["MRR@10", "Recall@5", "Recall@10", "Recall@50", "Recall@100"],
            "bootstrap": BOOTSTRAP,
            "comparison_ids": ["DualSource_vs_BM25", "DualSource_vs_ordinary_RRF", "DualSource_vs_LameR_MV"],
        },
        "bootstrap": BOOTSTRAP,
        "evaluation": {
            "primary_metric": "equal-subset macro nDCG@10",
            "qrels_boundary": "after_all_six_rankings_are_frozen",
            "sole_qrels_consumer": "eval/r2med_crb_evaluator.py",
            "candidate_analysis": "post-hoc interpretation only",
        },
        "dev_reference": {
            "source_report": "runs/rag_r2med_rerank/dev_report.json",
            "dual_source_macro_ndcg@10": dev_report["selection"]["best_ours_macro_ndcg@10"],
            "lamer_mv_macro_ndcg@10": dev_report["selection"]["strongest_non_ours_macro_ndcg@10"],
            "delta": dev_report["selection"]["delta_vs_strongest_non_ours"],
            "signal": "NEGATIVE_FOR_NEW_METHOD_CLAIM",
        },
        "storage": {"artifact_root": str(E_ROOT / "r2med/final_test"), "git_large_artifacts": False},
        "code": {"commit": _git("rev-parse", "HEAD"), "files_sha256": files_sha256},
    }
    validate_final_lock(lock)
    return lock


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--output", type=Path, default=ROOT / LOCK_PATH)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"final TEST lock already exists; refusing to replace: {args.output}")
    lock = build_lock(args.upstream)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "lock": str(args.output),
        "code_commit": lock["code"]["commit"],
        "source_manifest_sha256": lock["source"]["manifest_sha256"],
        "expected_model_calls": lock["generator"]["expected_calls"],
        "test_status": lock["test_status"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
