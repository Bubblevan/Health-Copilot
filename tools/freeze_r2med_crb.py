"""Create the immutable TEST lock from an accepted DEV-only result."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eval.r2med_crb_data import SPRINT_LOCKED_PATHS
from eval.r2med_multiview import FUSION_CONFIGS

ROOT = Path(__file__).resolve().parents[1]
E_ROOT = Path(r"E:\Health-Copilot-RAG")
DEV_REPORT = E_ROOT / "r2med/dev/reports/dev_report.json"
SOURCE_MANIFEST = ROOT / "runs/rag_r2med_crb/source_manifest.json"
VERIFIED_MODELS = E_ROOT / "models/verified_models.json"
LOCK_PATH = ROOT / "runs/rag_r2med_crb/final_method_lock.json"
BASELINE_REPORT = ROOT / "runs/rag_r2med_crb/dev/base_retrieval_manifest.json"
GENERATION_MANIFESTS = tuple(
    ROOT / f"runs/rag_r2med_crb/generation/dev/{method}/generation_manifest.json"
    for method in ("hyde", "query2doc", "lamer", "crb_q", "crb_prf")
)


def git_text(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def make_lock(
    dev_report: dict[str, Any],
    *,
    dev_report_bytes: bytes,
    verified_models: dict[str, Any],
    source_manifest_bytes: bytes,
    code_commit: str,
) -> dict[str, Any]:
    if dev_report.get("partition") != "DEV" or dev_report.get("test_status") != "PUBLIC_BENCHMARK_REUSED":
        raise ValueError("DEV report partition or public-test exposure status is invalid")
    if hashlib.sha256(source_manifest_bytes).hexdigest() != dev_report.get("source_manifest_sha256"):
        raise ValueError("DEV report and source manifest hashes do not match")
    gate = dev_report.get("gate", {})
    if gate.get("signal") != "POSITIVE" or float(gate.get("delta", 0)) < 0.005:
        raise ValueError("DEV gate failed; final method cannot be frozen and TEST must not run")
    if int(gate.get("positive_subsets", 0)) < 2:
        raise ValueError("DEV gate needs positive delta in at least two of three subsets")
    best = dev_report.get("best_crb", {})
    variant = best.get("method")
    if variant not in {"crb_q", "crb_prf"}:
        raise ValueError("DEV report has no valid selected CRB variant")
    config_id = best.get("config_id")
    config = next((entry for entry in FUSION_CONFIGS if entry["config_id"] == config_id), None)
    if config is None:
        raise ValueError("DEV report selected a fusion config outside the frozen grid")
    qwen = verified_models.get("qwen", {})
    bge = verified_models.get("bge_large", {})
    if qwen.get("sha256") != "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785":
        raise ValueError("verified-model manifest does not contain the pinned Qwen SHA")
    if bge.get("weights_sha256") != "45e1954914e29bd74080e6c1510165274ff5279421c89f76c418878732f64ae7":
        raise ValueError("verified-model manifest does not contain the pinned BGE-large weights")
    return {
        "schema_version": "r2med-crb-final-method-lock-v1",
        "status": "FROZEN_BEFORE_TEST",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "code_commit": code_commit,
        "test_status": "PUBLIC_BENCHMARK_REUSED",
        "crb_variant": variant,
        "prompt_sha256": best["prompt_sha256"],
        "generator": {
            "repo": "Qwen/Qwen3-8B-GGUF",
            "revision": qwen["revision"],
            "bytes": qwen["bytes"],
            "sha256": qwen["sha256"],
            "generation_config": dev_report["generation_config"],
        },
        "dense_retriever": {
            "repo": "BAAI/bge-large-en-v1.5",
            "revision": bge["revision"],
            "weights_sha256": bge["weights_sha256"],
        },
        "sparse_retriever": {
            "implementation": "pinned R2MED Pyserini Lucene analyzer + Gensim LuceneBM25Model",
            "k1": 0.9,
            "b": 0.4,
        },
        "retrieval_channels": [
            "BM25(original query)",
            "BM25(CRB canonical query + key concepts + disambiguating terms)",
            "BGE-large(original query)",
            "BGE-large(CRB pseudo-evidence)",
        ],
        "rrf": {"config_id": config_id, "k": config["rrf_k"], "weights": config["weights"]},
        "top_k_per_channel": 100,
        "primary_metric": "equal-weight macro nDCG@10",
        "dev_result_sha256": hashlib.sha256(dev_report_bytes).hexdigest(),
        "source_manifest_sha256": hashlib.sha256(source_manifest_bytes).hexdigest(),
        "base_retrieval_manifest_sha256": dev_report["base_retrieval_manifest_sha256"],
        "generation_manifests": dev_report["generation_audit"],
        "strongest_cost_matched_gar": {
            "method": dev_report["strongest_cost_matched_gar"]["method"],
            "config_id": dev_report["strongest_cost_matched_gar"]["config_id"],
        },
        "dev_gate": gate,
        "dev_subset_results": best["summary"]["by_subset"],
    }


def main() -> None:
    if LOCK_PATH.exists():
        raise FileExistsError(f"refusing to overwrite final method lock: {LOCK_PATH}")
    required_run_files = (BASELINE_REPORT, *GENERATION_MANIFESTS)
    required_paths = (*SPRINT_LOCKED_PATHS, *(path.relative_to(ROOT).as_posix() for path in required_run_files))
    for relative in required_paths:
        git_text("ls-files", "--error-unmatch", relative)
        if git_text("status", "--porcelain", "--", relative):
            raise ValueError(f"freeze requires the source/evidence file to be committed and clean: {relative}")
    report_bytes = DEV_REPORT.read_bytes()
    source_bytes = SOURCE_MANIFEST.read_bytes()
    report = json.loads(report_bytes)
    models = json.loads(VERIFIED_MODELS.read_text(encoding="utf-8"))
    if hashlib.sha256(BASELINE_REPORT.read_bytes()).hexdigest() != report["base_retrieval_manifest_sha256"]:
        raise ValueError("DEV report baseline manifest hash mismatch")
    for method, path in zip(("hyde", "query2doc", "lamer", "crb_q", "crb_prf"), GENERATION_MANIFESTS, strict=True):
        entry = report["generation_audit"][method]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["manifest_sha256"]:
            raise ValueError(f"DEV generation manifest hash mismatch: {method}")
    lock = make_lock(
        report,
        dev_report_bytes=report_bytes,
        verified_models=models,
        source_manifest_bytes=source_bytes,
        code_commit=git_text("rev-parse", "HEAD"),
    )
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOCK_PATH.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {LOCK_PATH}; commit it before starting TEST")


if __name__ == "__main__":
    main()
