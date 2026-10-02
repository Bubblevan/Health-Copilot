"""Verify and write the RAG-E6B method lock before reserved materialization."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any

from eval.rag_e6b.protocol import (
    BGE_REVISION,
    BGE_SHA256,
    FROZEN_DEV_COMMIT,
    FROZEN_DEV_MANIFEST_SHA256,
    LAMER_COMMIT,
    MODEL_SHA256,
    PLAN_SHA256,
    RUN_ROOT_RELATIVE,
    U2F_ROOT_RELATIVE,
    canonical_json_bytes,
    read_json,
    sha256_file,
    validate_reserved_plan,
)

FROZEN_METHOD_PATHS = (
    "eval/rag_e6/data.py",
    "eval/rag_e6/llm.py",
    "eval/rag_e6/reader.py",
    "eval/rag_e6/reader_executor.py",
    "eval/rag_e6/rsel.py",
    "eval/rag_e6/scoring.py",
    "eval/rag_e6/split.py",
    "eval/u3r_rag_transfer.py",
    "eval/r2med_multiview.py",
    "eval/r2med_crb_data.py",
    "eval/r2med_gar_generation.py",
)
E6B_PATHS = (
    "eval/rag_e6b/protocol.py",
    "eval/rag_e6b/materialize.py",
    "eval/rag_e6b/runner.py",
    "eval/rag_e6b/scoring.py",
    "tools/research/rag_e6/freeze_e6b_method.py",
    "tools/research/rag_e6/materialize_e6b_reserved.py",
    "tools/research/rag_e6/run_e6b_reserved.py",
    "tools/research/rag_e6/score_e6b_reserved.py",
    "tools/research/rag_e6/validate_e6b_execution_freeze.py",
)
QWEN_PATH = Path(r"E:\Health-Copilot-Models\models\qwen3-8b\Qwen3-8B-Q4_K_M.gguf")
BGE_PATH = Path(r"E:\Health-Copilot-Models\models\bge-large-en-v1.5\model.safetensors")
UPSTREAM_ROOT = Path(r"D:\MyLab\Jianli\external\rag\R2MED")


def _git(repository_root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repository_root, text=True).strip()


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()


def freeze_method(repository_root: Path) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    if _git(repository_root, "branch", "--show-current") != "rag-e6b-reserved-confirmation-20261002":
        raise ValueError("method freeze must run on the E6B topic branch")
    run_root = repository_root / RUN_ROOT_RELATIVE
    output_path = run_root / "method_freeze.json"
    if output_path.exists() or (run_root / "reserved").exists():
        raise FileExistsError("method lock or reserved materialization already exists")

    frozen_path = repository_root / "runs/rag_e6/frozen_dev/frozen_dev_manifest.json"
    frozen_manifest = read_json(frozen_path)
    if (
        sha256_file(frozen_path) != FROZEN_DEV_MANIFEST_SHA256
        or frozen_manifest.get("code_commit") != FROZEN_DEV_COMMIT
        or frozen_manifest.get("method") != "RSEL-v1"
        or frozen_manifest.get("evaluator_truth_opened") is not False
    ):
        raise ValueError("FROZEN_DEV RSEL reference identity failed")
    code_hashes = {
        relative: sha256_file(repository_root / relative)
        for relative in (*FROZEN_METHOD_PATHS, *E6B_PATHS)
    }
    for relative in FROZEN_METHOD_PATHS:
        expected = frozen_manifest.get("code_sha256", {}).get(relative)
        if not isinstance(expected, str) or code_hashes[relative] != expected:
            raise ValueError(f"frozen E6A method/source bytes changed: {relative}")
    if sha256_file(QWEN_PATH) != MODEL_SHA256:
        raise ValueError("Qwen3-8B GGUF SHA-256 mismatch")
    if sha256_file(BGE_PATH) != BGE_SHA256:
        raise ValueError("BGE-large model SHA-256 mismatch")
    server_path = repository_root / "runs/rag_e6/runtime/gpu_server_manifest.json"
    if sha256_file(server_path) != frozen_manifest.get("server_manifest_sha256"):
        raise ValueError("GPU server manifest differs from frozen FROZEN_DEV identity")

    u2f_root = repository_root / U2F_ROOT_RELATIVE
    source_manifest = read_json(u2f_root / "manifest.json")
    plan_path = u2f_root / "reserved_test_plan.json"
    reserved_plan = read_json(plan_path)
    if (
        source_manifest.get("dataset_root_hash")
        != "e28ea9ef9ecae47d3f27f28c68042066e9af297fe808cafebf1d3c8c80fb2134"
        or sha256_file(plan_path) != PLAN_SHA256
    ):
        raise ValueError("frozen reserved U2-F plan/source identity mismatch")
    validate_reserved_plan(reserved_plan)

    upstream_head = _git(UPSTREAM_ROOT, "rev-parse", "HEAD")
    if upstream_head != LAMER_COMMIT:
        raise ValueError("external R2MED upstream checkout is not at the frozen LameR commit")
    upstream_files = {
        relative: sha256_file(UPSTREAM_ROOT / relative)
        for relative in (
            "src/eval_BM25.py", "src/eval_retrieval.py", "src/example.py",
            "src/generate_hypothetical_doc.py", "src/instrcution.py",
        )
    }
    if upstream_files != frozen_manifest["upstream_lamer"]["verified_source_sha256"]:
        raise ValueError("pinned LameR upstream source file hashes differ from FROZEN_DEV")

    current_commit = _git(repository_root, "rev-parse", "HEAD")
    manifest = {
        "schema_version": "rag-e6b-method-freeze-v1",
        "current_git_commit": current_commit,
        "freeze_commit_ordering": "this manifest is committed before reserved materialization",
        "frozen_dev_reference_commit": FROZEN_DEV_COMMIT,
        "frozen_dev_reference_manifest_sha256": FROZEN_DEV_MANIFEST_SHA256,
        "rsel_method": "RSEL-v1",
        "rsel_source_sha256": code_hashes["eval/rag_e6/rsel.py"],
        "reader_source_sha256": code_hashes["eval/rag_e6/reader.py"],
        "frozen_e6a_source_hashes": {
            path: code_hashes[path] for path in FROZEN_METHOD_PATHS
        },
        "retrieval_source_hashes": {
            path: code_hashes[path] for path in (
                "eval/u3r_rag_transfer.py", "eval/r2med_multiview.py",
                "eval/r2med_crb_data.py", "eval/r2med_gar_generation.py",
            )
        },
        "scoring_source_hashes": {
            "eval/rag_e6/scoring.py": code_hashes["eval/rag_e6/scoring.py"],
            "eval/rag_e6b/scoring.py": code_hashes["eval/rag_e6b/scoring.py"],
        },
        "reader_executor_sha256": code_hashes["eval/rag_e6/reader_executor.py"],
        "e6b_pipeline_source_hashes": {
            path: code_hashes[path] for path in E6B_PATHS
        },
        "model_sha256": sha256_file(QWEN_PATH),
        "model_path": str(QWEN_PATH),
        "bge_revision": BGE_REVISION,
        "bge_sha256": sha256_file(BGE_PATH),
        "upstream_lamer_commit": upstream_head,
        "upstream_lamer_source_hashes": upstream_files,
        "retrieval_configuration": frozen_manifest["retrieval"]["strong"],
        "top_k": 10,
        "temperature": 0.0,
        "top_p": 1.0,
        "reasoning_enabled": False,
        "completion_ceiling": 512,
        "retry_count": 0,
        "reserved_plan_sha256": sha256_file(plan_path),
        "evaluator_truth_opened": False,
        "reserved_rows_materialized": False,
    }
    _write_new(output_path, canonical_json_bytes(manifest) + b"\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    manifest = freeze_method(args.repository_root)
    print(canonical_json_bytes({
        "status": "METHOD_FROZEN_PENDING_COMMIT",
        "code_commit": manifest["current_git_commit"],
        "model_sha256": manifest["model_sha256"],
        "reserved_plan_sha256": manifest["reserved_plan_sha256"],
        "evaluator_truth_opened": manifest["evaluator_truth_opened"],
        "reserved_rows_materialized": manifest["reserved_rows_materialized"],
    }).decode("utf-8"))


if __name__ == "__main__":
    main()
