"""Freeze B3 measurement-recovery identities after Reader V2 qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from eval.rag_e5.e5b3_reader import (
    READER_STATE_PROJECTION_SHA256,
    READER_V2_PROMPT,
    READER_V2_SCHEMA,
)
from eval.rag_e5.e5b3_recovery import (
    DEFAULT_B2_ARTIFACT_ROOT,
    DEFAULT_CORPUS_ROOT,
    DEFAULT_PRIVATE_ROOT,
    canonical_sha256,
    load_and_verify_b2_inputs,
    read_json,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[3]
QUALIFICATION_LOCK_PATH = ROOT / "runs/rag_e5/e5b3_reader_qualification_lock.json"
QUALIFICATION_REPORT_PATH = ROOT / "runs/rag_e5/e5b3_reader_qualification.json"
LOCK_PATH = ROOT / "runs/rag_e5/e5b3_recovery_lock.json"
DEFAULT_MODEL = Path(
    r"E:\Health-Copilot-Models\models\qwen3-8b\Qwen3-8B-Q4_K_M.gguf"
)
DEFAULT_SERVER = Path(
    r"C:\Users\bubblevan\AppData\Local\Microsoft\WinGet\Packages\ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe\llama-server.exe"
)
DEFAULT_EXTERNAL_ROOT = Path(r"D:\MyLab\Jianli\external\rag_e5\e5b3")
B2_CALL_LEDGER = Path(r"D:\MyLab\Jianli\external\rag_e5\e5b2\call_ledger.jsonl")
B2_REPORT_PATH = ROOT / "runs/rag_e5/e5b2_counterfactual_report.json"
EXPECTED_LLAMA_VERSION = (
    "version: 10068 (571d0d540)\nbuilt with Clang 20.1.8 for Windows x86_64"
)
RECOVERY_SOURCE_PATHS = (
    "eval/rag_e5/e5b3_reader.py",
    "eval/rag_e5/e5b3_qualification.py",
    "eval/rag_e5/e5b3_recovery.py",
    "eval/rag_e5/e5b3_evaluator.py",
    "tools/research/rag_e5/freeze_e5b3_recovery.py",
    "tools/research/rag_e5/run_e5b3_recovery.py",
    "tools/research/rag_e5/score_e5b3_recovery.py",
)


def _validate_qualification() -> tuple[dict[str, Any], dict[str, Any]]:
    qualification_lock = read_json(QUALIFICATION_LOCK_PATH)
    qualification = read_json(QUALIFICATION_REPORT_PATH)
    lock_body = {
        key: value
        for key, value in qualification_lock.items()
        if key != "qualification_lock_sha256"
    }
    report_body = {
        key: value
        for key, value in qualification.items()
        if key != "qualification_report_sha256"
    }
    if canonical_sha256(lock_body) != qualification_lock.get("qualification_lock_sha256"):
        raise ValueError("Reader qualification protocol lock self-hash mismatch")
    if canonical_sha256(report_body) != qualification.get("qualification_report_sha256"):
        raise ValueError("Reader qualification result self-hash mismatch")
    if (
        qualification.get("reader_contract_qualified") is not True
        or qualification.get("selected_output_budget") not in {256, 384, 512}
    ):
        raise ValueError("Reader V2 did not pass the independent qualification gates")
    if qualification.get("retrieval_calls") != 0 or qualification.get("bridge_calls") != 0:
        raise ValueError("Reader qualification unexpectedly ran retrieval or bridge calls")
    return qualification_lock, qualification


def freeze_recovery_protocol(
    *,
    private_root: Path,
    corpus_root: Path,
    b2_artifact_root: Path,
    model_path: Path,
    server_executable: Path,
    b2_call_ledger: Path,
    lock_path: Path,
) -> dict[str, Any]:
    if lock_path.exists():
        raise FileExistsError(f"B3 recovery lock is immutable: {lock_path}")
    qualification_lock, qualification = _validate_qualification()
    frozen = load_and_verify_b2_inputs(
        private_root=private_root,
        corpus_root=corpus_root,
        b2_artifact_root=b2_artifact_root,
    )
    if not model_path.is_file() or sha256_file(model_path) != qualification["model"]["sha256"]:
        raise ValueError("B3 Qwen model differs from the qualified model")
    version_result = subprocess.run(
        [str(server_executable), "--version"], check=True, capture_output=True, text=True
    )
    observed_version = (version_result.stdout + version_result.stderr).strip()
    if observed_version != EXPECTED_LLAMA_VERSION:
        raise ValueError("B3 llama.cpp runtime differs from the qualified runtime")
    if not b2_call_ledger.is_file():
        raise FileNotFoundError("B2 reader call ledger is required for measurement-only comparison")
    source_hashes = {relative: sha256_file(ROOT / relative) for relative in RECOVERY_SOURCE_PATHS}
    reuse_rows = [row for row in frozen.reuse_manifest_rows]
    b2_lock = frozen.lock
    body: dict[str, Any] = {
        "schema_version": "rag-e5-e5b3-recovery-lock-v1",
        "status": "FROZEN_BEFORE_B3_READER_CALLS",
        "scope": "reader measurement recovery over immutable B2 task/state/retrieval inputs",
        "task_set_sha256": b2_lock["ordered_case_ids_sha256"],
        "state_packet_set_sha256": b2_lock["state_packet_set_sha256"],
        "question_set_sha256": b2_lock["question_set_sha256"],
        "external_corpus_chunks_sha256": b2_lock["external_corpus_chunks_sha256"],
        "case_count": 60,
        "expected_arms": 180,
        "reader_calls": 180,
        "new_retrieval_calls": 0,
        "new_bridge_calls": 0,
        "actions": ["OFF", "STANDARD", "STRONG"],
        "run_id_scheme": "sha256(canonical_json([case_id, action, b3_recovery_lock_sha256]))",
        "qualification_lock_sha256": qualification_lock["qualification_lock_sha256"],
        "qualification_report_sha256": qualification["qualification_report_sha256"],
        "reader_schema_sha256": qualification["reader_schema_sha256"],
        "reader_prompt_sha256": qualification["reader_prompt_sha256"],
        "reader_state_projection_sha256": READER_STATE_PROJECTION_SHA256,
        "reader_output_budget": qualification["selected_output_budget"],
        "reader_contract_qualified": True,
        "reader_schema_identity_verified": canonical_sha256(READER_V2_SCHEMA)
        == qualification["reader_schema_sha256"],
        "reader_prompt_identity_verified": hashlib.sha256(
            READER_V2_PROMPT.encode("utf-8")
        ).hexdigest()
        == qualification["reader_prompt_sha256"],
        "model": qualification["model"],
        "llama_cpp": {
            "version": observed_version,
            "executable": str(server_executable),
            "executable_sha256": sha256_file(server_executable),
            "endpoint": "http://127.0.0.1:8093/v1",
            "server_args": qualification_lock["llama_cpp"]["server_args"],
        },
        "temperature": 0.0,
        "reasoning": "disabled",
        "retry_on_error": False,
        "b2_lock_sha256": b2_lock["counterfactual_lock_sha256"],
        "b2_lock_file_sha256": frozen.b2_lock_file_sha256,
        "b2_execution_manifest_sha256": frozen.execution_manifest[
            "execution_manifest_sha256"
        ],
        "b2_execution_manifest_file_sha256": frozen.b2_execution_manifest_file_sha256,
        "b2_artifact_set_sha256": frozen.artifact_set_sha256,
        "b2_reuse_manifest_sha256": frozen.reuse_manifest_sha256,
        "b2_reuse_manifest_rows": reuse_rows,
        "b2_call_ledger_sha256": sha256_file(b2_call_ledger),
        "b2_report_sha256": sha256_file(B2_REPORT_PATH),
        "b2_artifacts_read_only": True,
        "b2_results_rescored": False,
        "teacher_opened": False,
        "202608_opened": False,
        "retrieval_performed": False,
        "bridge_called": False,
        "scorer_version": b2_lock["scorer_version"],
        "scoring_contract_sha256": b2_lock["scoring_contract_sha256"],
        "scorer_module_sha256": b2_lock["scorer_module_sha256"],
        "source_sha256": source_hashes,
    }
    if not body["reader_schema_identity_verified"] or not body["reader_prompt_identity_verified"]:
        raise ValueError("qualified Reader V2 contract changed before B3 freeze")
    body["b3_recovery_lock_sha256"] = canonical_sha256(body)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(body, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    return body


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-root", type=Path, default=DEFAULT_PRIVATE_ROOT)
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS_ROOT)
    parser.add_argument("--b2-artifact-root", type=Path, default=DEFAULT_B2_ARTIFACT_ROOT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--server-executable", type=Path, default=DEFAULT_SERVER)
    parser.add_argument("--b2-call-ledger", type=Path, default=B2_CALL_LEDGER)
    parser.add_argument("--lock", type=Path, default=LOCK_PATH)
    args = parser.parse_args()
    lock = freeze_recovery_protocol(
        private_root=args.private_root,
        corpus_root=args.corpus_root,
        b2_artifact_root=args.b2_artifact_root,
        model_path=args.model,
        server_executable=args.server_executable,
        b2_call_ledger=args.b2_call_ledger,
        lock_path=args.lock,
    )
    print(
        json.dumps(
            {
                "b3_recovery_lock_sha256": lock["b3_recovery_lock_sha256"],
                "b2_artifact_set_sha256": lock["b2_artifact_set_sha256"],
                "b2_reuse_manifest_sha256": lock["b2_reuse_manifest_sha256"],
                "reader_output_budget": lock["reader_output_budget"],
                "model_calls_before_commit": 0,
                "source_arms_verified": len(lock["b2_reuse_manifest_rows"]),
            },
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
