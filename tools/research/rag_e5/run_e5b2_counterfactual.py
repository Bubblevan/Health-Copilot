"""Execute every frozen OFF/STANDARD/STRONG E5-B2 arm, teacher-blind and resumably."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.rag_e5.counterfactual import (
    LoopbackCompletionClient,
    canonical_sha256,
    expected_arm_specs,
    parse_reader_output,
    render_lamer_prompt,
    render_reader_prompt,
    verify_complete_arm,
    write_complete_arm,
)
from eval.rag_e5.overlay import READER_SCHEMA, E5RuntimeCase
from eval.rag_e5.retrieval import E5B2Retriever

PRIVATE_ROOT = Path(r"D:\MyLab\Jianli\external\rag_e5\e5b1")
CORPUS_ROOT = Path(r"D:\MyLab\Jianli\external\rag_e5\e5a3\corpus\public_health_plus_guideline")
INDEX_ROOT = Path(r"D:\MyLab\Jianli\external\rag_e5\e5a3\indexes\public_health_plus_guideline")
MODEL_ROOT = Path(r"E:\Health-Copilot-Models\models")
ARTIFACT_ROOT = Path(r"D:\MyLab\Jianli\external\rag_e5\e5b2")
LOCK_PATH = ROOT / "runs/rag_e5/e5b2_protocol_lock.json"
EXECUTION_MANIFEST_PATH = ROOT / "runs/rag_e5/e5b2_execution_manifest.json"
EXPECTED_LOCK_SHA256 = ""
MODEL_BYTES = 5_027_783_488
MODEL_SHA256 = "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785"
DEFAULT_PORT = 8093


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path.name}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_runtime_cases(path: Path) -> list[E5RuntimeCase]:
    rows = _read_jsonl(path)
    cases = [E5RuntimeCase.from_dict(row) for row in rows]
    if len(cases) != 60 or len({case.case_id for case in cases}) != 60:
        raise ValueError("runtime input must contain exactly 60 unique cases")
    return cases


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _lock_sha(lock: dict[str, Any]) -> str:
    body = {key: value for key, value in lock.items() if key != "counterfactual_lock_sha256"}
    observed = hashlib.sha256(_canonical_bytes(body)).hexdigest()
    if lock.get("counterfactual_lock_sha256") != observed:
        raise ValueError("B2 protocol lock self-hash mismatch")
    return observed


def _verify_lock_code(lock: dict[str, Any]) -> None:
    for item in lock.get("locked_code", []):
        path = ROOT / item["path"]
        if not path.is_file() or _sha256_file(path) != item["sha256"]:
            raise ValueError(f"execution code changed after lock: {item['path']}")


def _verify_inputs(
    *,
    lock: dict[str, Any],
    lock_sha: str,
    private_root: Path,
    corpus_root: Path,
    index_root: Path,
    model_root: Path,
    server_executable: Path,
) -> tuple[list[E5RuntimeCase], dict[str, dict[str, Any]], dict[str, Any], dict[str, Any], str]:
    if lock.get("status") != "FROZEN_PROTOCOL_NO_OUTCOMES" or lock.get("expected_arms") != 180:
        raise ValueError("B2 lock is not frozen for exactly 180 arms")
    if lock.get("base_main") != "56a809024e3d006451724380e94f2b4ed9c2cee7":
        raise ValueError("B2 base main SHA differs from the approved baseline")
    if lock.get("202608_opened") is not False or lock.get("outcomes_or_oracle_action_created") is not False:
        raise ValueError("B2 lock violates the frozen cohort/outcome boundary")
    _verify_lock_code(lock)

    runtime_path = private_root / "runtime_cases.jsonl"
    if _sha256_file(runtime_path) != lock["runtime_cases_file_sha256"]:
        raise ValueError("runtime case artifact hash changed after the B2 freeze")
    raw_cases = _read_jsonl(runtime_path)
    cases = [E5RuntimeCase.from_dict(row) for row in raw_cases]
    ids = [case.case_id for case in cases]
    if len(cases) != 60 or canonical_sha256(ids) != lock["ordered_case_ids_sha256"]:
        raise ValueError("runtime case identity/order differs from the B2 lock")
    question_sha = canonical_sha256(
        [{"case_id": row["case_id"], "question": row["question"]} for row in raw_cases]
    )
    if question_sha != lock["question_set_sha256"]:
        raise ValueError("runtime question set changed after the B2 freeze")

    capability_path = private_root / "capability_context.json"
    capability = _read_json(capability_path)
    if canonical_sha256(capability) != lock["runtime_capability_context_sha256"]:
        raise ValueError("runtime capability context changed after the B2 freeze")
    reader_prompt_path = private_root / "reader_prompt.txt"
    reader_schema_path = private_root / "reader_schema.json"
    if hashlib.sha256(reader_prompt_path.read_bytes()).hexdigest() != lock["reader_prompt_sha256"]:
        raise ValueError("reader prompt changed after the B2 freeze")
    if canonical_sha256(_read_json(reader_schema_path)) != lock["reader_schema_sha256"]:
        raise ValueError("reader schema changed after the B2 freeze")
    templates_path = private_root / "task_templates.json"
    if canonical_sha256(_read_json(templates_path)) != lock["task_template_sha256"]:
        raise ValueError("frozen task templates changed after the B2 freeze")
    state_by_ref: dict[str, dict[str, Any]] = {}
    state_packets_by_user: dict[str, dict[str, Any]] = {}
    for case in cases:
        if case.state_packet_ref is None:
            if case.decision_boundary is not None:
                raise ValueError("state-free case must not carry a decision boundary")
            continue
        reference = Path(case.state_packet_ref)
        if reference.is_absolute() or ".." in reference.parts:
            raise ValueError("state packet reference escapes its frozen private root")
        path = private_root / reference
        packet = _read_json(path)
        state_by_ref[case.state_packet_ref] = packet
        state_packets_by_user[packet["user_id"]] = packet
        if packet["user_id"] != case.user_id or packet["decision_boundary"] != case.decision_boundary:
            raise ValueError("runtime case and v4 state packet are not identity-aligned")
    packet_identity = [
        {
            "user_id": packet["user_id"],
            "packet_sha256": packet["packet_sha256"],
            "decision_boundary": packet["decision_boundary"]["naive_timestamp"],
        }
        for packet in sorted(state_packets_by_user.values(), key=lambda item: item["user_id"])
    ]
    if canonical_sha256(packet_identity) != lock["state_packet_set_sha256"]:
        raise ValueError("runtime v4 state packet set changed after the B2 freeze")

    chunks_path = corpus_root / "chunks.jsonl"
    if _sha256_file(chunks_path) != lock["external_corpus_chunks_sha256"]:
        raise ValueError("active corpus chunks changed after the B2 freeze")
    for relative, expected in (
        ("bm25_index_manifest.json", lock["bm25_artifacts"]["manifest_sha256"]),
        ("bm25_document_term_weights.npz", lock["bm25_artifacts"]["matrix_sha256"]),
        ("bm25_lucene_tokens.jsonl", lock["bm25_artifacts"]["token_artifact_sha256"]),
        ("bm25_dictionary_model.json", lock["bm25_artifacts"]["dictionary_model_sha256"]),
        ("dense_index_manifest.json", lock["bge_artifacts"]["dense_manifest_sha256"]),
        ("bge_large_embeddings.npy", lock["bge_artifacts"]["embedding_matrix_sha256"]),
    ):
        if _sha256_file(index_root / relative) != expected:
            raise ValueError(f"active retrieval artifact changed after the B2 freeze: {relative}")
    analyzer_class = index_root / "java-classes" / "E5LuceneAnalyzerCli.class"
    if _sha256_file(analyzer_class) != lock["bm25_artifacts"]["analyzer_cli_class_sha256"]:
        raise ValueError("compiled Lucene analyzer CLI changed after the B2 freeze")

    profiles_path = ROOT / "runs/rag_e5/retrieval_action_profiles.json"
    if _sha256_file(profiles_path) != lock["action_profiles"]["manifest_sha256"]:
        raise ValueError("retrieval action profile manifest changed after the B2 freeze")
    profiles = _read_json(profiles_path)
    profile_by_action = {row["action"]: row for row in profiles["profiles"]}
    if profile_by_action["STANDARD"]["config_sha256"] != lock["action_profiles"]["STANDARD"]:
        raise ValueError("STANDARD retrieval config hash changed")
    if profile_by_action["STRONG"]["config_sha256"] != lock["action_profiles"]["STRONG"]:
        raise ValueError("STRONG retrieval config hash changed")

    upstream_root = Path(r"D:\MyLab\Jianli\external\rag\R2MED")
    from eval.r2med_crb_data import PINNED_UPSTREAM_COMMIT
    from eval.r2med_gar_generation import (
        EXPECTED_UPSTREAM_FILES,
        load_upstream_prompt_catalog,
        prompt_sha256,
    )

    upstream = lock.get("upstream", {})
    if (
        upstream.get("commit") != PINNED_UPSTREAM_COMMIT
        or upstream.get("source_file_sha256") != EXPECTED_UPSTREAM_FILES
    ):
        raise ValueError("R2MED upstream source identity differs from the B2 lock")
    upstream_prompts = load_upstream_prompt_catalog(upstream_root)
    lamer_template = upstream_prompts["lamer"]["MedQA-Diag"]
    if prompt_sha256(lamer_template) != upstream.get("lamer_medqa_prompt_sha256"):
        raise ValueError("pinned upstream LameR prompt differs from the B2 lock")
    if upstream.get("lamer_medqa_prompt_sha256") != lock["strong_generator"]["prompt_template_sha256"]:
        raise ValueError("B2 upstream prompt and strong-action profile binding disagree")

    model_path = model_root / "qwen3-8b" / "Qwen3-8B-Q4_K_M.gguf"
    if model_path.stat().st_size != MODEL_BYTES or _sha256_file(model_path) != MODEL_SHA256:
        raise ValueError("Qwen GGUF does not match the frozen 8B model identity")
    bge_path = model_root / "bge-large-en-v1.5" / "model.safetensors"
    if _sha256_file(bge_path) != lock["bge_artifacts"]["weights_sha256"]:
        raise ValueError("BGE-large weights do not match the frozen profile")
    for relative, expected in lock["bge_artifacts"]["model_files_sha256"].items():
        if _sha256_file(bge_path.parent / relative) != expected:
            raise ValueError(f"BGE-large model metadata changed after freeze: {relative}")
    runtime = {
        "python": platform.python_version(),
        **{
            package: importlib.metadata.version(package)
            for package in ("numpy", "scipy", "torch", "transformers", "sentence-transformers")
        },
    }
    if runtime != lock["python_runtime"]:
        raise ValueError("Python/BGE runtime versions differ from the B2 protocol lock")
    if not server_executable.is_file():
        raise FileNotFoundError("llama-server executable is unavailable")
    version = subprocess.run(
        [str(server_executable), "--version"], check=True, capture_output=True, text=True
    )
    observed_version = (version.stdout or version.stderr).strip().replace("\n", " ")
    expected_version = lock["answer_model"]["llama_cpp_version"]
    if observed_version != expected_version:
        raise ValueError("llama.cpp runtime version differs from the B1 frozen version")
    return cases, state_by_ref, capability, profile_by_action, lamer_template


def _append_call_ledger(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _next_call_ordinal(path: Path) -> int:
    if not path.exists():
        return 1
    rows = _read_jsonl(path)
    ordinals = [row.get("call_ordinal") for row in rows if isinstance(row.get("call_ordinal"), int)]
    return max(ordinals, default=0) + 1


def _verify_resume_call_budget(
    *, ledger_path: Path, specs: list[dict[str, str]], completed: set[tuple[str, str]]
) -> None:
    """Permit resuming only between fully committed arms, never by repeating an inference."""
    if not ledger_path.exists():
        if completed:
            raise ValueError("completed arms exist without their immutable call ledger")
        return
    rows = _read_jsonl(ledger_path)
    starts = sum(row.get("status") == "STARTED" for row in rows)
    if starts > 240:
        raise ValueError("call ledger exceeds the frozen 240-call maximum")
    by_key: dict[tuple[str, str, str], list[str]] = {}
    for row in rows:
        key = (str(row.get("case_id")), str(row.get("action")), str(row.get("phase")))
        by_key.setdefault(key, []).append(str(row.get("status")))
    expected = {
        (spec["case_id"], spec["action"], phase)
        for spec in specs
        for phase in (("bridge", "reader") if spec["action"] == "STRONG" else ("reader",))
    }
    for key in expected:
        statuses = by_key.get(key, [])
        arm_key = key[:2]
        if arm_key not in completed and statuses:
            raise ValueError(
                "an inference was already attempted for an incomplete arm; refusing to retry"
            )
        if arm_key in completed and sorted(statuses) != ["COMPLETED", "STARTED"]:
            raise ValueError("completed arm is missing its exact one-call ledger record")
    if set(by_key) - expected:
        raise ValueError("call ledger contains an unknown or duplicate experimental call")


def _verify_committed_execution_files(lock_path: Path, lock: dict[str, Any]) -> None:
    paths = [lock_path, *(ROOT / row["path"] for row in lock["locked_code"])]
    relative = [str(path.relative_to(ROOT)).replace("\\", "/") for path in paths]
    result = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--", *relative],
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        raise ValueError("B2 execution lock/code must be committed and clean before any model call")


def _server_ready(port: int, timeout_seconds: int = 240) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(1)
    raise TimeoutError("B2 llama.cpp CPU server did not become healthy on loopback")


def _start_server(
    executable: Path, model_path: Path, port: int, log_path: Path
) -> tuple[subprocess.Popen[Any], Any]:
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", port)) == 0:
            raise OSError(f"loopback port {port} is already in use; refusing model ambiguity")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("ab")
    args = [
        str(executable),
        "--model", str(model_path),
        "--host", "127.0.0.1",
        "--port", str(port),
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
        raise RuntimeError(f"B2 llama.cpp server exited with code {process.returncode}")
    return process, log_handle


def _call_model(
    *,
    client: LoopbackCompletionClient,
    prompt: str,
    schema: dict[str, Any] | None,
    ledger_path: Path,
    case_id: str,
    action: str,
    phase: str,
    run_id: str,
    call_ordinal: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    request_record = {
        "model": "Qwen/Qwen3-8B-GGUF",
        "temperature": 0.0,
        "reasoning": "disabled",
        "max_output_tokens": 256,
        "retry_on_error": False,
        "endpoint_binding": "loopback_only",
        "stateless_request": True,
        "prompt_cache": False,
        "prompt": prompt,
        "json_schema": schema,
    }
    prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    _append_call_ledger(
        ledger_path,
        {
            "run_id": run_id,
            "case_id": case_id,
            "action": action,
            "phase": phase,
            "call_ordinal": call_ordinal,
            "prompt_sha256": prompt_sha,
            "status": "STARTED",
            "timestamp_utc": datetime.now(UTC).isoformat(),
        },
    )
    started = time.perf_counter()
    completion = client.complete(prompt, json_schema=schema)
    latency_ms = (time.perf_counter() - started) * 1000
    response_record = {
        "text": completion.text,
        "finish_reason": completion.finish_reason,
        "input_tokens": completion.input_tokens,
        "output_tokens": completion.output_tokens,
        "latency_ms": latency_ms,
    }
    _append_call_ledger(
        ledger_path,
        {
            "run_id": run_id,
            "case_id": case_id,
            "action": action,
            "phase": phase,
            "call_ordinal": call_ordinal,
            "prompt_sha256": prompt_sha,
            "status": "COMPLETED",
            "finish_reason": completion.finish_reason,
            "input_tokens": completion.input_tokens,
            "output_tokens": completion.output_tokens,
            "latency_ms": latency_ms,
            "timestamp_utc": datetime.now(UTC).isoformat(),
        },
    )
    return request_record, response_record


def _arm_path(root: Path, case_id: str, action: str, run_id: str) -> Path:
    return root / "arms" / case_id / action / f"{run_id}.json"


def _execute_arm(
    *,
    case: E5RuntimeCase,
    state_packet: dict[str, Any] | None,
    capability: dict[str, Any],
    action: str,
    run_id: str,
    lock_sha: str,
    retriever: E5B2Retriever,
    client: LoopbackCompletionClient,
    lamer_template: str,
    private_root: Path,
    artifact_root: Path,
    call_ledger: Path,
    first_call_ordinal: int,
) -> tuple[dict[str, Any], int]:
    arm_started = time.perf_counter()
    retrieval_started = time.perf_counter()
    retrieval: dict[str, Any] | None = None
    bridge: dict[str, Any] | None = None
    bridge_latency_ms = 0.0
    call_ordinal = first_call_ordinal
    if action == "OFF":
        ranking = []
        supplied_chunks: list[dict[str, Any]] = []
        retrieval_calls = 0
    elif action == "STANDARD":
        retrieval = retriever.retrieve_standard(case.question)
        ranking = retrieval["ranking"]
        supplied_chunks = retriever.supplied_chunks(ranking, top_k=5)
        retrieval_calls = 1
    elif action == "STRONG":
        feedback_rank, feedback_passages = retriever.feedback_passages(case.question, depth=10)
        bridge_prompt = render_lamer_prompt(lamer_template, case.question, feedback_passages)
        bridge_request, bridge_response = _call_model(
            client=client,
            prompt=bridge_prompt,
            schema=None,
            ledger_path=call_ledger,
            case_id=case.case_id,
            action=action,
            phase="bridge",
            run_id=run_id,
            call_ordinal=call_ordinal,
        )
        call_ordinal += 1
        bridge_latency_ms = bridge_response["latency_ms"]
        bridge_valid = bool(bridge_response["text"].strip()) and bridge_response["finish_reason"] != "length"
        bridge_text = bridge_response["text"].strip() if bridge_valid else case.question
        bridge = {
            "request": bridge_request,
            "request_sha256": hashlib.sha256(_canonical_bytes(bridge_request)).hexdigest(),
            "raw_response": bridge_response,
            "response_sha256": hashlib.sha256(_canonical_bytes(bridge_response)).hexdigest(),
            "valid": bridge_valid,
            "fallback_original_query": not bridge_valid,
            "fallback_reason": None if bridge_valid else "empty_or_truncated_generation",
            "feedback_top10_chunk_ids": [row.doc_id for row in feedback_rank],
        }
        retrieval = retriever.retrieve_strong(case.question, bridge_text)
        ranking = retrieval["ranking"]
        supplied_chunks = retriever.supplied_chunks(ranking, top_k=5)
        retrieval_calls = 1
    else:
        raise ValueError(f"unknown action: {action}")
    retrieval_latency_ms = (time.perf_counter() - retrieval_started) * 1000 - bridge_latency_ms

    reader_prompt = render_reader_prompt(
        question=case.question,
        state_packet=state_packet,
        passages=supplied_chunks,
    )
    reader_request, reader_response = _call_model(
        client=client,
        prompt=reader_prompt,
        schema=READER_SCHEMA,
        ledger_path=call_ledger,
        case_id=case.case_id,
        action=action,
        phase="reader",
        run_id=run_id,
        call_ordinal=call_ordinal,
    )
    call_ordinal += 1
    parsed, valid_json = parse_reader_output(reader_response["text"])
    supplied_ids = [row["chunk_id"] for row in supplied_chunks]
    invented_citations = (
        sorted(set(parsed.get("citations", [])) - set(supplied_ids)) if parsed else []
    )
    ranking_records = [
        {"chunk_id": row.doc_id, "score": row.score} for row in ranking
    ]
    channel_records = (
        {
            key: [{"chunk_id": row.doc_id, "score": row.score} for row in values]
            for key, values in (retrieval["channels"].items() if retrieval else [])
        }
        if retrieval
        else {}
    )
    state_sha = hashlib.sha256(_canonical_bytes(state_packet)).hexdigest() if state_packet else None
    payload = {
        "schema_version": "rag-e5-e5b2-arm-v1",
        "status": "COMPLETE",
        "case_id": case.case_id,
        "action": action,
        "run_id": run_id,
        "lock_sha256": lock_sha,
        "question_sha256": hashlib.sha256(case.question.encode("utf-8")).hexdigest(),
        "state_packet_sha256": state_sha,
        "runtime_capability_context_sha256": canonical_sha256(capability),
        "action_profile_sha256": None if action == "OFF" else retrieval["profile_sha256"],
        "retrieval_calls": retrieval_calls,
        "retrieval_latency_ms": max(retrieval_latency_ms, 0.0),
        "bridge_latency_ms": bridge_latency_ms if action == "STRONG" else 0.0,
        "reader_latency_ms": reader_response["latency_ms"],
        "total_latency_ms": (time.perf_counter() - arm_started) * 1000,
        "bridge": bridge,
        "retrieval": {
            "channels": channel_records,
            "ranking": ranking_records,
            "ranking_sha256": hashlib.sha256(_canonical_bytes(ranking_records)).hexdigest(),
            "supplied_chunk_ids": supplied_ids,
            "supplied_chunk_ids_sha256": hashlib.sha256(_canonical_bytes(supplied_ids)).hexdigest(),
        },
        "supplied_chunks": supplied_chunks,
        "reader_request": reader_request,
        "reader_request_sha256": hashlib.sha256(_canonical_bytes(reader_request)).hexdigest(),
        "reader_raw_response": reader_response,
        "reader_raw_response_sha256": hashlib.sha256(_canonical_bytes(reader_response)).hexdigest(),
        "reader_json_valid": valid_json,
        "reader_parsed": parsed,
        "invented_citations": invented_citations,
        "input_tokens": (bridge["raw_response"]["input_tokens"] if bridge else 0),
        "reader_input_tokens": reader_response["input_tokens"],
        "output_tokens": (bridge["raw_response"]["output_tokens"] if bridge else 0),
        "reader_output_tokens": reader_response["output_tokens"],
        "reader_call_count": 1,
        "bridge_call_count": int(action == "STRONG"),
        "private_input_root_identity": str(private_root),
    }
    return payload, call_ordinal


def _freeze_execution_manifest(
    *,
    lock: dict[str, Any],
    lock_sha: str,
    specs: list[dict[str, str]],
    artifact_root: Path,
) -> dict[str, Any]:
    artifacts = []
    for spec in specs:
        path = _arm_path(artifact_root, spec["case_id"], spec["action"], spec["run_id"])
        arm = verify_complete_arm(
            path, expected_run_id=spec["run_id"], expected_lock_sha256=lock_sha
        )
        artifacts.append(
            {
                "case_id": spec["case_id"],
                "action": spec["action"],
                "run_id": spec["run_id"],
                "file_sha256": _sha256_file(path),
                "completion_sha256": arm["completion_sha256"],
            }
        )
    arm_set_sha = canonical_sha256(artifacts)
    run_id_set_sha = canonical_sha256([row["run_id"] for row in specs])
    readers = sum(1 for _ in artifacts)
    bridges = sum(row["action"] == "STRONG" for row in specs)
    invalid = 0
    bridge_fallbacks = 0
    input_token_values: list[int] = []
    output_token_values: list[int] = []
    captured_token_calls = 0
    for spec in specs:
        arm = verify_complete_arm(
            _arm_path(artifact_root, spec["case_id"], spec["action"], spec["run_id"]),
            expected_run_id=spec["run_id"], expected_lock_sha256=lock_sha,
        )
        invalid += int(not arm["reader_json_valid"])
        bridge_fallbacks += int(bool(arm.get("bridge", {}).get("fallback_original_query")))
        call_rows = []
        if arm.get("bridge"):
            call_rows.append(arm["bridge"]["raw_response"])
        call_rows.append(arm["reader_raw_response"])
        for call_row in call_rows:
            prompt_tokens = call_row.get("input_tokens")
            completion_tokens = call_row.get("output_tokens")
            if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
                captured_token_calls += 1
                input_token_values.append(prompt_tokens)
                output_token_values.append(completion_tokens)
    manifest = {
        "schema_version": "rag-e5-e5b2-execution-manifest-v1",
        "status": "ALL_ARMS_FROZEN_BEFORE_SCORING",
        "expected_arms": 180,
        "completed_arms": len(artifacts),
        "case_count": 60,
        "reader_calls": readers,
        "bridge_calls": bridges,
        "model_calls": readers + bridges,
        "retrieval_action_invocations": sum(row["action"] != "OFF" for row in specs),
        "invalid_reader_outputs": invalid,
        "bridge_fallbacks": bridge_fallbacks,
        "token_usage_captured_calls": captured_token_calls,
        "input_tokens_total": sum(input_token_values) if captured_token_calls == readers + bridges else None,
        "output_tokens_total": sum(output_token_values) if captured_token_calls == readers + bridges else None,
        "run_id_set_sha256": run_id_set_sha,
        "artifact_set_sha256": arm_set_sha,
        "counterfactual_lock_sha256": lock_sha,
        "model_sha256": lock["answer_model"]["sha256"],
        "external_corpus_identity": lock["external_corpus_identity"],
        "202608_opened": False,
        "outcomes_or_oracle_action_created": False,
        "artifacts": artifacts,
    }
    manifest["execution_manifest_sha256"] = canonical_sha256(manifest)
    _write_json(EXECUTION_MANIFEST_PATH, manifest)
    return manifest


def run(
    *,
    private_root: Path = PRIVATE_ROOT,
    corpus_root: Path = CORPUS_ROOT,
    index_root: Path = INDEX_ROOT,
    model_root: Path = MODEL_ROOT,
    artifact_root: Path = ARTIFACT_ROOT,
    lock_path: Path = LOCK_PATH,
    server_executable: Path,
    port: int = DEFAULT_PORT,
    preflight_only: bool = False,
) -> dict[str, Any]:
    lock = _read_json(lock_path)
    lock_sha = _lock_sha(lock)
    _verify_lock_code(lock)
    if EXPECTED_LOCK_SHA256 and lock_sha != EXPECTED_LOCK_SHA256:
        raise ValueError("B2 lock SHA differs from runner's approved lock binding")
    cases, state_by_ref, capability, profiles, lamer_template = _verify_inputs(
        lock=lock,
        lock_sha=lock_sha,
        private_root=private_root,
        corpus_root=corpus_root,
        index_root=index_root,
        model_root=model_root,
        server_executable=server_executable,
    )
    specs = expected_arm_specs([case.case_id for case in cases], lock_sha)
    if len(specs) != 180:
        raise AssertionError("the frozen protocol must yield exactly 180 deterministic arms")
    status: dict[str, Any] = {
        "preflight": "PASS",
        "cases": len(cases),
        "arms": len(specs),
        "counterfactual_lock_sha256": lock_sha,
        "model_calls": 0,
        "retrieval_calls": 0,
        "202608_opened": False,
    }
    if preflight_only:
        smoke = E5B2Retriever(
            corpus_root=corpus_root,
            index_root=index_root,
            bge_model_root=model_root / "bge-large-en-v1.5",
            standard_profile=profiles["STANDARD"],
            strong_profile=profiles["STRONG"],
        )
        try:
            standard = smoke.retrieve_standard("blood pressure risk and prevention")
            feedback, _ = smoke.feedback_passages("blood pressure risk and prevention", depth=10)
            strong = smoke.retrieve_strong(
                "blood pressure risk and prevention", "blood pressure prevention guideline"
            )
            if (
                not standard["ranking"]
                or len(feedback) != 10
                or not strong["ranking"]
                or len(standard["channels"]) != 2
                or len(strong["channels"]) != 4
            ):
                raise ValueError("frozen retrieval actions failed their no-outcome smoke test")
        finally:
            smoke.bm25.close()
        status["retriever_preflight"] = "PASS"
        return status
    _verify_committed_execution_files(lock_path, lock)

    if EXECUTION_MANIFEST_PATH.exists():
        existing = _read_json(EXECUTION_MANIFEST_PATH)
        if existing.get("status") == "ALL_ARMS_FROZEN_BEFORE_SCORING":
            if existing.get("counterfactual_lock_sha256") != lock_sha:
                raise ValueError("a different completed B2 run is already frozen")
            return existing
        raise ValueError("unexpected existing execution manifest; refusing to overwrite")

    completed: set[tuple[str, str]] = set()
    for spec in specs:
        path = _arm_path(artifact_root, spec["case_id"], spec["action"], spec["run_id"])
        if path.exists():
            verify_complete_arm(path, expected_run_id=spec["run_id"], expected_lock_sha256=lock_sha)
            completed.add((spec["case_id"], spec["action"]))
    call_ledger = artifact_root / "call_ledger.jsonl"
    _verify_resume_call_budget(
        ledger_path=call_ledger, specs=specs, completed=completed
    )
    pending = [spec for spec in specs if (spec["case_id"], spec["action"]) not in completed]
    if not pending:
        return _freeze_execution_manifest(
            lock=lock, lock_sha=lock_sha, specs=specs, artifact_root=artifact_root
        )

    model_path = model_root / "qwen3-8b" / "Qwen3-8B-Q4_K_M.gguf"
    retriever = E5B2Retriever(
        corpus_root=corpus_root,
        index_root=index_root,
        bge_model_root=model_root / "bge-large-en-v1.5",
        standard_profile=profiles["STANDARD"],
        strong_profile=profiles["STRONG"],
    )
    runtime_by_id = {case.case_id: case for case in cases}
    process: subprocess.Popen[Any] | None = None
    log_handle: Any = None
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    try:
        process, log_handle = _start_server(
            server_executable,
            model_path,
            port,
            artifact_root / "logs" / f"e5b2-cpu-{timestamp}.log",
        )
        _server_ready(port)
        client = LoopbackCompletionClient(f"http://127.0.0.1:{port}/v1")
        call_ordinal = _next_call_ordinal(call_ledger)
        for index, spec in enumerate(specs, start=1):
            key = (spec["case_id"], spec["action"])
            if key in completed:
                continue
            case = runtime_by_id[spec["case_id"]]
            state_packet = state_by_ref.get(case.state_packet_ref) if case.state_packet_ref else None
            payload, call_ordinal = _execute_arm(
                case=case,
                state_packet=state_packet,
                capability=capability,
                action=spec["action"],
                run_id=spec["run_id"],
                lock_sha=lock_sha,
                retriever=retriever,
                client=client,
                lamer_template=lamer_template,
                private_root=private_root,
                artifact_root=artifact_root,
                call_ledger=call_ledger,
                first_call_ordinal=call_ordinal,
            )
            arm_path = _arm_path(
                artifact_root, spec["case_id"], spec["action"], spec["run_id"]
            )
            write_complete_arm(arm_path, payload)
            completed.add(key)
            print(
                json.dumps(
                    {
                        "progress_arms": len(completed),
                        "expected_arms": 180,
                        "last_action": spec["action"],
                        "reader_json_valid": payload["reader_json_valid"],
                        "bridge_fallback": bool(payload.get("bridge", {}).get("fallback_original_query")),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if len(completed) != 180:
            raise AssertionError("execution ended before all 180 frozen arms completed")
        return _freeze_execution_manifest(
            lock=lock, lock_sha=lock_sha, specs=specs, artifact_root=artifact_root
        )
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        if log_handle is not None:
            log_handle.close()
        retriever.bm25.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-root", type=Path, default=PRIVATE_ROOT)
    parser.add_argument("--corpus-root", type=Path, default=CORPUS_ROOT)
    parser.add_argument("--index-root", type=Path, default=INDEX_ROOT)
    parser.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--lock", type=Path, default=LOCK_PATH)
    parser.add_argument(
        "--server-executable", type=Path,
        default=Path(shutil.which("llama-server") or "llama-server"),
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    result = run(
        private_root=args.private_root,
        corpus_root=args.corpus_root,
        index_root=args.index_root,
        model_root=args.model_root,
        artifact_root=args.artifact_root,
        lock_path=args.lock,
        server_executable=args.server_executable,
        port=args.port,
        preflight_only=args.preflight_only,
    )
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
