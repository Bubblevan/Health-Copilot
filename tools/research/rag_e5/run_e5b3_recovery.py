"""Replay the qualified Reader V2 over immutable E5-B2 evidence artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eval.rag_e5.counterfactual import ACTION_ORDER, write_complete_arm
from eval.rag_e5.e5b3_reader import (
    READER_STATE_PROJECTION_SHA256,
    READER_V2_PROMPT,
    READER_V2_SCHEMA,
    ReaderStateView,
    parse_reader_v2_output,
    render_reader_v2_prompt,
)
from eval.rag_e5.e5b3_recovery import (
    DEFAULT_B2_ARTIFACT_ROOT,
    DEFAULT_CORPUS_ROOT,
    DEFAULT_PRIVATE_ROOT,
    b3_run_id,
    canonical_sha256,
    load_and_verify_b2_inputs,
    read_json,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LOCK = ROOT / "runs/rag_e5/e5b3_recovery_lock.json"
DEFAULT_MANIFEST = ROOT / "runs/rag_e5/e5b3_execution_manifest.json"
DEFAULT_MODEL = Path(
    r"E:\Health-Copilot-Models\models\qwen3-8b\Qwen3-8B-Q4_K_M.gguf"
)
DEFAULT_SERVER = Path(
    r"C:\Users\bubblevan\AppData\Local\Microsoft\WinGet\Packages\ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe\llama-server.exe"
)
DEFAULT_EXTERNAL_ROOT = Path(r"D:\MyLab\Jianli\external\rag_e5\e5b3")
DEFAULT_B2_CALL_LEDGER = Path(
    r"D:\MyLab\Jianli\external\rag_e5\e5b2\call_ledger.jsonl"
)
PORT = 8093
BASE_URL = f"http://127.0.0.1:{PORT}/v1"
TIMEOUT_SECONDS = 600


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _verify_lock(lock_path: Path) -> dict[str, Any]:
    lock = read_json(lock_path)
    body = {key: value for key, value in lock.items() if key != "b3_recovery_lock_sha256"}
    if canonical_sha256(body) != lock.get("b3_recovery_lock_sha256"):
        raise ValueError("B3 recovery lock self-hash mismatch")
    for relative, expected in lock.get("source_sha256", {}).items():
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"B3 recovery source changed after freeze: {relative}")
    if lock.get("status") != "FROZEN_BEFORE_B3_READER_CALLS":
        raise ValueError("B3 recovery protocol is not frozen")
    if lock.get("reader_contract_qualified") is False or lock.get("reader_output_budget") != 512:
        raise ValueError("B3 reader is not the qualified contract/budget")
    if lock.get("new_retrieval_calls") != 0 or lock.get("new_bridge_calls") != 0:
        raise ValueError("B3 lock permits a forbidden retrieval or bridge call")
    if lock.get("202608_opened") is not False or lock.get("teacher_opened") is not False:
        raise ValueError("B3 lock violates the cohort or teacher boundary")
    paths = [lock_path, *(ROOT / relative for relative in lock["source_sha256"])]
    relative_paths = [str(path.relative_to(ROOT)).replace("\\", "/") for path in paths]
    result = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--", *relative_paths],
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        raise ValueError("B3 protocol lock and locked code must be committed before model calls")
    return lock


def _validate_qualification(root: Path, lock: dict[str, Any]) -> None:
    qualification_lock = read_json(root / "runs/rag_e5/e5b3_reader_qualification_lock.json")
    report = read_json(root / "runs/rag_e5/e5b3_reader_qualification.json")
    lock_body = {
        key: value
        for key, value in qualification_lock.items()
        if key != "qualification_lock_sha256"
    }
    report_body = {
        key: value for key, value in report.items() if key != "qualification_report_sha256"
    }
    if canonical_sha256(lock_body) != qualification_lock.get("qualification_lock_sha256"):
        raise ValueError("Reader qualification lock self-hash mismatch")
    if canonical_sha256(report_body) != report.get("qualification_report_sha256"):
        raise ValueError("Reader qualification report self-hash mismatch")
    if (
        report.get("reader_contract_qualified") is not True
        or report.get("selected_output_budget") != lock.get("reader_output_budget")
        or report.get("qualification_report_sha256")
        != lock.get("qualification_report_sha256")
    ):
        raise ValueError("B3 recovery lock does not match the qualified Reader V2")
    if (
        hashlib.sha256(READER_V2_PROMPT.encode("utf-8")).hexdigest()
        != lock.get("reader_prompt_sha256")
        or canonical_sha256(READER_V2_SCHEMA) != lock.get("reader_schema_sha256")
        or READER_STATE_PROJECTION_SHA256 != lock.get("reader_state_projection_sha256")
    ):
        raise ValueError("Reader V2 prompt/schema/state projection changed after qualification")


def _verify_b2_identity(lock: dict[str, Any], frozen: Any, b2_call_ledger: Path) -> None:
    checks = {
        "b2_lock_file_sha256": frozen.b2_lock_file_sha256,
        "b2_execution_manifest_file_sha256": frozen.b2_execution_manifest_file_sha256,
        "b2_artifact_set_sha256": frozen.artifact_set_sha256,
        "b2_reuse_manifest_sha256": frozen.reuse_manifest_sha256,
    }
    for field, actual in checks.items():
        if lock.get(field) != actual:
            raise ValueError(f"B3 lock's {field} no longer matches the frozen B2 input")
    if not b2_call_ledger.is_file() or sha256_file(b2_call_ledger) != lock.get(
        "b2_call_ledger_sha256"
    ):
        raise ValueError("frozen B2 call ledger changed after B3 protocol freeze")
    if len(frozen.arm_contexts) != 180:
        raise ValueError("B3 requires exactly 180 independently verified B2 arm contexts")


def _server_ready(timeout_seconds: int = 240) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(1)
    raise TimeoutError("B3 llama.cpp CPU server did not become healthy on loopback")


def _start_server(executable: Path, model_path: Path, log_path: Path) -> tuple[Any, Any]:
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", PORT)) == 0:
            raise OSError(f"loopback port {PORT} is already in use; refusing model ambiguity")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("ab")
    args = [
        str(executable),
        "--model", str(model_path),
        "--host", "127.0.0.1",
        "--port", str(PORT),
        "--ctx-size", "16384",
        "--n-gpu-layers", "0",
        "--reasoning", "off",
        "--chat-template-kwargs", '{"enable_thinking":false}',
        "--no-cache-prompt",
        "--parallel", "1",
        "--offline",
        "--no-webui",
    ]
    process = subprocess.Popen(
        args,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if process.poll() is not None:
        log_handle.close()
        raise RuntimeError(f"B3 llama.cpp server exited with code {process.returncode}")
    return process, log_handle


def _complete(prompt: str, *, output_budget: int) -> dict[str, Any]:
    body = {
        "model": "local-qwen3-8b",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "top_p": 1,
        "max_tokens": output_budget,
        "cache_prompt": False,
        "stream": False,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "e5_reader_v2", "strict": True, "schema": READER_V2_SCHEMA},
        },
    }
    request = urllib.request.Request(
        BASE_URL + "/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        payload = json.loads(response.read())
    latency_ms = (time.perf_counter() - started) * 1000
    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("llama.cpp response has invalid choices")
    choice = choices[0]
    text = (choice.get("message") or {}).get("content")
    if not isinstance(text, str):
        raise TypeError("llama.cpp response has no text content")
    usage = payload.get("usage") or {}
    return {
        "text": text,
        "finish_reason": choice.get("finish_reason"),
        "input_tokens": usage.get("prompt_tokens"),
        "output_tokens": usage.get("completion_tokens"),
        "latency_ms": latency_ms,
    }


def _expected_specs(frozen: Any, lock_sha256: str) -> list[dict[str, str]]:
    cases = {row["case_id"] for row in frozen.runtime_cases}
    return [
        {"case_id": case_id, "action": action, "run_id": b3_run_id(case_id, action, lock_sha256)}
        for case_id in sorted(cases)
        for action in ACTION_ORDER
    ]


def project_frozen_b2_context(context: Any) -> dict[str, Any]:
    """Copy the exact retrieval evidence and bridge provenance, never the old reader result."""
    return {
        "passages": [dict(row) for row in context.supplied_chunks],
        "retrieval": {
            "channels": dict(context.retrieval_channels),
            "ranking": [dict(row) for row in context.retrieval_ranking],
            "ranking_sha256": context.retrieval_ranking_sha256,
            "supplied_chunk_ids": [row["chunk_id"] for row in context.supplied_chunks],
            "supplied_chunk_ids_sha256": context.supplied_chunk_ids_sha256,
        },
        "bridge": dict(context.bridge_artifact) if context.bridge_artifact is not None else None,
        "source_retrieval_calls": context.source_retrieval_calls,
        "source_bridge_call_count": context.source_bridge_call_count,
        "source_bridge_latency_ms": context.source_bridge_latency_ms,
        "source_bridge_input_tokens": context.source_bridge_input_tokens,
        "source_bridge_output_tokens": context.source_bridge_output_tokens,
        "source_bridge_fallback": context.source_bridge_fallback,
    }


def run_recovery(
    *,
    lock_path: Path,
    manifest_path: Path,
    private_root: Path,
    corpus_root: Path,
    b2_artifact_root: Path,
    model_path: Path,
    server_executable: Path,
    external_root: Path,
    b2_call_ledger: Path,
) -> dict[str, Any]:
    if manifest_path.exists():
        raise FileExistsError(f"B3 execution manifest is immutable: {manifest_path}")
    lock = _verify_lock(lock_path)
    _validate_qualification(ROOT, lock)
    if not model_path.is_file() or sha256_file(model_path) != lock["model"]["sha256"]:
        raise ValueError("B3 model bytes differ from the independently qualified model")
    if not server_executable.is_file() or sha256_file(server_executable) != lock["llama_cpp"][
        "executable_sha256"
    ]:
        raise ValueError("B3 llama.cpp executable differs from the qualification runtime")
    frozen = load_and_verify_b2_inputs(
        private_root=private_root,
        corpus_root=corpus_root,
        b2_artifact_root=b2_artifact_root,
    )
    _verify_b2_identity(lock, frozen, b2_call_ledger)

    b2_root = b2_artifact_root.resolve()
    output_root = external_root.resolve()
    if b2_root == output_root or b2_root in output_root.parents or output_root in b2_root.parents:
        raise ValueError("B3 output root must be separate from the read-only B2 artifact root")
    arms_root = external_root / "arms"
    ledger_path = external_root / "reader_call_ledger.jsonl"
    log_path = external_root / "logs" / "reader-recovery-20260930.log"
    if arms_root.exists() and any(arms_root.rglob("*.json")):
        raise FileExistsError("B3 arm artifacts already exist; model calls may not be repeated")
    if ledger_path.exists():
        raise FileExistsError("B3 reader call ledger already exists; model calls may not be repeated")
    external_root.mkdir(parents=True, exist_ok=True)

    case_by_id = {row["case_id"]: row for row in frozen.runtime_cases}
    lock_sha = lock["b3_recovery_lock_sha256"]
    specs = _expected_specs(frozen, lock_sha)
    contexts = {(row.case_id, row.action): row for row in frozen.arm_contexts}
    process, log_handle = _start_server(server_executable, model_path, log_path)
    artifacts: list[dict[str, Any]] = []
    try:
        _server_ready()
        for ordinal, spec in enumerate(specs, start=1):
            key = (spec["case_id"], spec["action"])
            context = contexts[key]
            case = case_by_id[context.case_id]
            replay = project_frozen_b2_context(context)
            packet_ref = case.get("state_packet_ref")
            packet = frozen.state_packets_by_ref.get(packet_ref) if packet_ref else None
            state_view = ReaderStateView.from_packet(packet) if packet is not None else None
            passages = replay["passages"]
            prompt = render_reader_v2_prompt(
                question=case["question"],
                state_view=state_view,
                passages=passages,
            )
            prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            call_id = canonical_sha256([lock_sha, context.case_id, context.action, "reader"])
            call_key = {
                "call_id": call_id,
                "case_id": context.case_id,
                "action": context.action,
                "run_id": spec["run_id"],
                "call_ordinal": ordinal,
                "phase": "reader",
                "prompt_sha256": prompt_sha,
            }
            _append_jsonl(
                ledger_path,
                {**call_key, "status": "STARTED", "timestamp_utc": datetime.now(UTC).isoformat()},
            )
            completion = _complete(prompt, output_budget=lock["reader_output_budget"])
            parsed, is_valid = parse_reader_v2_output(completion["text"])
            supplied_ids = {str(row["chunk_id"]) for row in passages}
            citations = {
                str(citation)
                for fact in (parsed or {}).get("guidance_facts", [])
                if isinstance(fact, dict)
                for citation in fact.get("citations", [])
            }
            invented = sorted(citations - supplied_ids)
            retrieval = replay["retrieval"]
            reader_response = {
                "text": completion["text"],
                "finish_reason": completion["finish_reason"],
                "input_tokens": completion["input_tokens"],
                "output_tokens": completion["output_tokens"],
                "latency_ms": completion["latency_ms"],
            }
            composed_latency = (
                context.source_retrieval_latency_ms
                + replay["source_bridge_latency_ms"]
                + float(completion["latency_ms"])
            )
            arm = {
                "schema_version": "rag-e5-e5b3-recovery-arm-v1",
                "status": "COMPLETE",
                "run_id": spec["run_id"],
                "lock_sha256": lock_sha,
                "case_id": context.case_id,
                "action": context.action,
                "question_sha256": context.question_sha256,
                "state_packet_sha256": context.state_packet_sha256,
                "reader_state_projection_sha256": READER_STATE_PROJECTION_SHA256,
                "reader_state_view": state_view.to_dict() if state_view is not None else None,
                "runtime_capability_context_sha256": context.runtime_capability_context_sha256,
                "b2_source": context.reuse_manifest_row(),
                "reader_request": {
                    "prompt": prompt,
                    "prompt_sha256": prompt_sha,
                    "schema_sha256": lock["reader_schema_sha256"],
                    "state_projection_sha256": READER_STATE_PROJECTION_SHA256,
                    "output_budget": lock["reader_output_budget"],
                    "temperature": 0.0,
                    "reasoning": "disabled",
                    "retry_on_error": False,
                },
                "reader_raw_response": reader_response,
                "reader_raw_response_sha256": canonical_sha256(reader_response),
                "reader_parsed": parsed,
                "reader_json_valid": is_valid,
                "reader_finish_reason": completion["finish_reason"],
                "reader_input_tokens": completion["input_tokens"],
                "reader_output_tokens": completion["output_tokens"],
                "reader_latency_ms": completion["latency_ms"],
                "reader_call_count": 1,
                "new_reader_calls": 1,
                "retrieval": retrieval,
                "supplied_chunks": passages,
                "bridge": replay["bridge"],
                "bridge_call_count": replay["source_bridge_call_count"],
                "bridge_latency_ms": replay["source_bridge_latency_ms"],
                "bridge_input_tokens": replay["source_bridge_input_tokens"],
                "bridge_output_tokens": replay["source_bridge_output_tokens"],
                "bridge_fallback_original_query": replay["source_bridge_fallback"],
                "retrieval_calls": replay["source_retrieval_calls"],
                "retrieval_latency_ms": context.source_retrieval_latency_ms,
                "new_retrieval_calls": 0,
                "new_bridge_calls": 0,
                "invented_citations": invented,
                "reader_contract_cost_basis": "COMPOSED_COMPONENT_COST",
                "total_latency_ms": composed_latency,
                "b2_reader_output_reused": False,
                "b2_retrieval_and_bridge_artifacts_reused": True,
            }
            arm_path = arms_root / context.case_id / context.action / f"{spec['run_id']}.json"
            completion_sha = write_complete_arm(arm_path, arm)
            _append_jsonl(
                ledger_path,
                {
                    **call_key,
                    "status": "COMPLETED",
                    "finish_reason": completion["finish_reason"],
                    "input_tokens": completion["input_tokens"],
                    "output_tokens": completion["output_tokens"],
                    "reader_json_valid": is_valid,
                    "latency_ms": completion["latency_ms"],
                    "artifact_completion_sha256": completion_sha,
                    "timestamp_utc": datetime.now(UTC).isoformat(),
                },
            )
            artifacts.append(
                {
                    **spec,
                    "path": str(arm_path.relative_to(external_root)).replace("\\", "/"),
                    "file_sha256": sha256_file(arm_path),
                    "completion_sha256": completion_sha,
                    "b2_source_arm_file_sha256": context.source_arm_file_sha256,
                    "b2_source_completion_sha256": context.source_completion_sha256,
                    "prompt_sha256": prompt_sha,
                    "reader_json_valid": is_valid,
                    "finish_reason": completion["finish_reason"],
                }
            )
    finally:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
        log_handle.close()

    if len(artifacts) != 180:
        raise ValueError("B3 execution did not complete all 180 unique reader arms")
    if not ledger_path.is_file():
        raise ValueError("B3 reader call ledger is missing")
    ledger = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
    starts = [row for row in ledger if row.get("status") == "STARTED"]
    completed = [row for row in ledger if row.get("status") == "COMPLETED"]
    if len(starts) != 180 or len(completed) != 180 or len(ledger) != 360:
        raise ValueError("B3 call ledger does not contain exactly one start/completion per call")
    manifest: dict[str, Any] = {
        "schema_version": "rag-e5-e5b3-execution-manifest-v1",
        "status": "ALL_B3_READER_ARMS_FROZEN_BEFORE_SCORING",
        "b3_recovery_lock_sha256": lock_sha,
        "b3_recovery_lock_file_sha256": sha256_file(lock_path),
        "completed_arms": len(artifacts),
        "expected_arms": 180,
        "reader_calls": 180,
        "new_reader_calls": 180,
        "new_retrieval_calls": 0,
        "new_bridge_calls": 0,
        "b2_artifacts_read_only": True,
        "b2_results_rescored": False,
        "b2_artifact_set_sha256": lock["b2_artifact_set_sha256"],
        "b2_reuse_manifest_sha256": lock["b2_reuse_manifest_sha256"],
        "qualification_lock_sha256": lock["qualification_lock_sha256"],
        "qualification_report_sha256": lock["qualification_report_sha256"],
        "reader_schema_sha256": lock["reader_schema_sha256"],
        "reader_prompt_sha256": lock["reader_prompt_sha256"],
        "reader_state_projection_sha256": lock["reader_state_projection_sha256"],
        "reader_output_budget": lock["reader_output_budget"],
        "model": lock["model"],
        "llama_cpp": lock["llama_cpp"],
        "call_ledger_path_external": str(ledger_path),
        "call_ledger_sha256": sha256_file(ledger_path),
        "artifact_set_sha256": canonical_sha256(artifacts),
        "run_id_set_sha256": canonical_sha256([row["run_id"] for row in artifacts]),
        "artifacts": artifacts,
        "teacher_opened": False,
        "202608_opened": False,
        "execution_code_commit": subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
    }
    manifest["execution_manifest_sha256"] = canonical_sha256(manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--private-root", type=Path, default=DEFAULT_PRIVATE_ROOT)
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS_ROOT)
    parser.add_argument("--b2-artifact-root", type=Path, default=DEFAULT_B2_ARTIFACT_ROOT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--server-executable", type=Path, default=DEFAULT_SERVER)
    parser.add_argument("--external-root", type=Path, default=DEFAULT_EXTERNAL_ROOT)
    parser.add_argument("--b2-call-ledger", type=Path, default=DEFAULT_B2_CALL_LEDGER)
    args = parser.parse_args()
    manifest = run_recovery(
        lock_path=args.lock,
        manifest_path=args.manifest,
        private_root=args.private_root,
        corpus_root=args.corpus_root,
        b2_artifact_root=args.b2_artifact_root,
        model_path=args.model,
        server_executable=args.server_executable,
        external_root=args.external_root,
        b2_call_ledger=args.b2_call_ledger,
    )
    print(
        json.dumps(
            {
                "execution_manifest_sha256": manifest["execution_manifest_sha256"],
                "artifact_set_sha256": manifest["artifact_set_sha256"],
                "reader_calls": manifest["reader_calls"],
                "new_retrieval_calls": manifest["new_retrieval_calls"],
                "new_bridge_calls": manifest["new_bridge_calls"],
                "teacher_opened": manifest["teacher_opened"],
            },
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
