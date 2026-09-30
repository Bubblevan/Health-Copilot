"""Teacher-blind B4 planning and frozen counterfactual execution."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from eval.rag_e5.b4_guidance import (CHAT_TEMPLATE_OVERHEAD_RESERVE,
                                     CONTEXT_SIZE, MAX_OUTPUT_TOKENS,
                                     LlamaServerClient, render_guidance_prompt)
from eval.rag_e5.b4_materializer import (StateClaim, classify_runtime_task,
                                         extract_citations,
                                         materialize_final_response,
                                         materialize_state_claims)
from eval.rag_e5.counterfactual import canonical_sha256, expected_arm_specs
from eval.rag_e5.e5b3_recovery import (DEFAULT_B2_ARTIFACT_ROOT,
                                       DEFAULT_CORPUS_ROOT,
                                       DEFAULT_PRIVATE_ROOT, FrozenB2Inputs,
                                       load_and_verify_b2_inputs, sha256_file)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_B4_PRIVATE_ROOT = Path(r"D:\MyLab\Jianli\external\rag_e5\e5b4")
DEFAULT_PROTOCOL_PATH = ROOT / "runs/rag_e5/e5b4_protocol_lock.json"
DEFAULT_EXECUTION_MANIFEST_PATH = ROOT / "runs/rag_e5/e5b4_execution_manifest.json"
DEFAULT_B2_LOCK_PATH = ROOT / "runs/rag_e5/e5b2_protocol_lock.json"
DEFAULT_B2_EXECUTION_PATH = ROOT / "runs/rag_e5/e5b2_execution_manifest.json"
DEFAULT_B2_REPORT_PATH = ROOT / "runs/rag_e5/e5b2_counterfactual_report.json"
DEFAULT_B3_MANIFEST_PATH = ROOT / "runs/rag_e5/e5b3_execution_manifest.json"
DEFAULT_B3_REPORT_PATH = ROOT / "runs/rag_e5/e5b3_measurement_recovery_report.json"
DEFAULT_LLAMACPP_EXE = Path(
    r"C:\Users\bubblevan\AppData\Local\Microsoft\WinGet\Packages\ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe\llama-server.exe"
)
DEFAULT_SERVER_URL = "http://127.0.0.1:8081"
MAX_PROMPT_TOKENS = CONTEXT_SIZE - MAX_OUTPUT_TOKENS - CHAT_TEMPLATE_OVERHEAD_RESERVE
ACTION_ORDER = ("OFF", "STANDARD", "STRONG")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def verify_protocol_lock(lock: dict[str, Any]) -> str:
    lock_sha = lock.get("protocol_lock_sha256")
    body = {key: value for key, value in lock.items() if key != "protocol_lock_sha256"}
    if not isinstance(lock_sha, str) or canonical_sha256(body) != lock_sha:
        raise ValueError("B4 protocol lock self-hash mismatch")
    if lock.get("status") != "FROZEN_BEFORE_B4_GUIDANCE_CALLS":
        raise ValueError("B4 protocol was not frozen before guidance calls")
    if lock.get("202608_opened") is not False or lock.get("source_cohort") != "202607_only":
        raise ValueError("B4 protocol crosses the excluded 202608 cohort")
    return lock_sha


def verify_frozen_code(lock: dict[str, Any]) -> None:
    code_rows = lock.get("code_sha256", [])
    for row in code_rows:
        relative = row.get("path")
        expected = row.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ValueError("B4 protocol contains a malformed frozen code hash")
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"B4 frozen code changed: {relative}")
    tracked = subprocess.run(
        [
            "git",
            "ls-files",
            "--error-unmatch",
            "runs/rag_e5/e5b4_protocol_lock.json",
            *(row["path"] for row in code_rows),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if tracked.returncode != 0:
        raise ValueError("B4 lock and frozen code must be committed before model calls")
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or result.stdout.strip():
        raise ValueError("B4 protocol/code must be committed and clean before execution")


def verify_server(client: LlamaServerClient, lock: dict[str, Any]) -> dict[str, Any]:
    client.health()
    props = client.props()
    expected = lock["runtime"]
    model_file = str(expected["model_path"]).replace("/", "\\").casefold()
    observed_file = str(props.get("model_path", "")).replace("/", "\\").casefold()
    if observed_file != model_file:
        raise ValueError("shared llama.cpp server is not serving the frozen Qwen3 model path")
    if props.get("build_info") != expected["llama_cpp_build_info"]:
        raise ValueError("shared llama.cpp server build differs from the frozen runtime")
    actual_ctx = props.get("default_generation_settings", {}).get("n_ctx")
    if actual_ctx != expected["server_context_capacity"]:
        raise ValueError("shared llama.cpp server context capacity changed after protocol freeze")
    if props.get("is_sleeping") is True:
        raise ValueError("shared llama.cpp server is sleeping")
    model_path = Path(expected["model_path"])
    if not model_path.is_file() or sha256_file(model_path) != lock["model"]["sha256"]:
        raise ValueError("Qwen3 model file hash changed after B4 protocol freeze")
    server_executable = Path(expected["server_executable"])
    if (
        not server_executable.is_file()
        or sha256_file(server_executable) != expected["server_executable_sha256"]
    ):
        raise ValueError("shared llama.cpp executable hash changed after B4 protocol freeze")
    port = urlparse(expected["server_url"]).port or 80
    process = _inspect_server_process(port)
    if int(process["ProcessId"]) != expected["server_pid"]:
        raise ValueError("shared llama.cpp process changed after protocol freeze")
    if str(Path(process["ExecutablePath"]).resolve()).casefold() != str(
        server_executable.resolve()
    ).casefold():
        raise ValueError("loopback endpoint is no longer owned by the frozen server binary")
    command_line = process.get("CommandLine", "")
    expected_flags = {
        "n_gpu_layers_99": "--n-gpu-layers 99",
        "flash_attention": "--flash-attn on",
        "parallel_1": "--parallel 1",
        "q4_k_cache": "--cache-type-k q4_0",
        "q4_v_cache": "--cache-type-v q4_0",
    }
    flags = expected["observed_flags"]
    if any(
        flags.get(key) is not True or expected_flag not in command_line
        for key, expected_flag in expected_flags.items()
    ):
        raise ValueError("shared GPU offload/runtime flags differ from the B4 protocol lock")
    return {
        "model_path": props.get("model_path"),
        "build_info": props.get("build_info"),
        "server_context_capacity": actual_ctx,
        "model_ftype": props.get("model_ftype"),
        "server_pid": int(process["ProcessId"]),
        "process_command_line_sha256": hashlib.sha256(command_line.encode("utf-8")).hexdigest(),
    }


def _inspect_server_process(port: int) -> dict[str, Any]:
    command = (
        f"$connection = Get-NetTCPConnection -State Listen -LocalAddress 127.0.0.1 "
        f"-LocalPort {port} -ErrorAction Stop | Select-Object -First 1; "
        "$process = Get-CimInstance Win32_Process -Filter "
        "\"ProcessId = $($connection.OwningProcess)\"; "
        "$process | Select-Object ProcessId,ExecutablePath,CommandLine | ConvertTo-Json -Compress"
    )
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError("could not verify shared llama.cpp process identity")
    value = json.loads(result.stdout)
    if not isinstance(value, dict):
        raise ValueError("shared llama.cpp process identity is malformed")
    return value


def load_verified_b2(lock: dict[str, Any]) -> FrozenB2Inputs:
    frozen = load_and_verify_b2_inputs(
        private_root=DEFAULT_PRIVATE_ROOT,
        corpus_root=DEFAULT_CORPUS_ROOT,
        b2_artifact_root=DEFAULT_B2_ARTIFACT_ROOT,
        b2_lock_path=DEFAULT_B2_LOCK_PATH,
        b2_execution_manifest_path=DEFAULT_B2_EXECUTION_PATH,
    )
    required_identities = {
        "b2_protocol_lock_sha256": frozen.lock["counterfactual_lock_sha256"],
        "b2_protocol_lock_file_sha256": frozen.b2_lock_file_sha256,
        "b2_execution_manifest_file_sha256": frozen.b2_execution_manifest_file_sha256,
        "b2_report_file_sha256": sha256_file(DEFAULT_B2_REPORT_PATH),
        "task_template_sha256": sha256_file(DEFAULT_PRIVATE_ROOT / "task_templates.json"),
        "b2_artifact_set_sha256": frozen.artifact_set_sha256,
        "b2_reuse_manifest_sha256": frozen.reuse_manifest_sha256,
        "task_set_sha256": frozen.lock["ordered_case_ids_sha256"],
        "state_set_sha256": frozen.lock["state_packet_set_sha256"],
        "corpus_chunks_sha256": frozen.lock["external_corpus_chunks_sha256"],
        "runtime_cases_sha256": frozen.lock["runtime_cases_file_sha256"],
        "question_set_sha256": frozen.lock["question_set_sha256"],
    }
    for name, expected in required_identities.items():
        if lock.get(name) != expected:
            raise ValueError(f"B4 lock no longer matches frozen B2 input: {name}")
    if sha256_file(DEFAULT_B2_REPORT_PATH) != lock.get("b2_report_file_sha256"):
        raise ValueError("frozen B2 scored report changed after B4 protocol freeze")
    if sha256_file(DEFAULT_PRIVATE_ROOT / "task_templates.json") != lock.get(
        "task_template_sha256"
    ):
        raise ValueError("frozen B2 task template changed after B4 protocol freeze")
    if frozen.artifact_set_sha256 != "e528263a5141e6ac476252a9bfb04a1a95fc53475c5617bec23fb106980168f1":
        raise ValueError("B2 artifact set differs from the user-frozen E5-B4 identity")
    if len(frozen.runtime_cases) != 60 or len(frozen.arm_contexts) != 180:
        raise ValueError("B4 requires exactly 60 cases and 180 B2 source arms")
    return frozen


def read_and_verify_b3_identity() -> dict[str, Any]:
    """Hash-check frozen B3 diagnostics without opening its teacher plane."""
    manifest = _read_json(DEFAULT_B3_MANIFEST_PATH)
    report = _read_json(DEFAULT_B3_REPORT_PATH)
    manifest_sha = manifest.get("execution_manifest_sha256")
    manifest_body = {
        key: value for key, value in manifest.items() if key != "execution_manifest_sha256"
    }
    artifacts = manifest.get("artifacts")
    if (
        not isinstance(manifest_sha, str)
        or canonical_sha256(manifest_body) != manifest_sha
        or manifest.get("status") != "ALL_B3_READER_ARMS_FROZEN_BEFORE_SCORING"
        or manifest.get("completed_arms") != 180
        or manifest.get("expected_arms") != 180
        or manifest.get("teacher_opened") is not False
        or manifest.get("202608_opened") is not False
        or not isinstance(artifacts, list)
        or len(artifacts) != 180
        or canonical_sha256(artifacts) != manifest.get("artifact_set_sha256")
    ):
        raise ValueError("frozen B3 manifest identity/integrity check failed")
    if (
        report.get("execution_manifest_sha256") != manifest_sha
        or report.get("teacher_opened") is not False
        or report.get("202608_opened") is not False
        or report.get("b2_artifact_set_sha256")
        != "e528263a5141e6ac476252a9bfb04a1a95fc53475c5617bec23fb106980168f1"
    ):
        raise ValueError("frozen B3 closeout report is inconsistent or opened privileged labels")
    call_ledger = Path(manifest["call_ledger_path_external"])
    if not call_ledger.is_file() or sha256_file(call_ledger) != manifest.get(
        "call_ledger_sha256"
    ):
        raise ValueError("frozen B3 call ledger changed")
    artifact_root = call_ledger.parent
    for row in artifacts:
        path = artifact_root / row["path"]
        if not path.is_file() or sha256_file(path) != row.get("file_sha256"):
            raise ValueError("frozen B3 arm artifact changed")
    return {
        "b3_manifest_file_sha256": sha256_file(DEFAULT_B3_MANIFEST_PATH),
        "b3_report_file_sha256": sha256_file(DEFAULT_B3_REPORT_PATH),
        "b3_artifact_set_sha256": manifest["artifact_set_sha256"],
        "b3_execution_manifest_sha256": manifest_sha,
        "b3_call_ledger_sha256": manifest["call_ledger_sha256"],
        "b3_recovery_lock_sha256": manifest["b3_recovery_lock_sha256"],
        "b3_status": report["status"],
    }


def verify_b3_identity(lock: dict[str, Any]) -> dict[str, Any]:
    identity = read_and_verify_b3_identity()
    for key, value in identity.items():
        if lock.get(key) != value:
            raise ValueError(f"frozen B3 artifacts changed after B4 protocol freeze: {key}")
    return identity


def make_execution_plan(
    *, frozen: FrozenB2Inputs, client: LlamaServerClient, private_root: Path
) -> list[dict[str, Any]]:
    cases = {row["case_id"]: row for row in frozen.runtime_cases}
    packets = frozen.state_packets_by_ref
    contexts = {(row.case_id, row.action): row for row in frozen.arm_contexts}
    specs = expected_arm_specs(sorted(cases), frozen.lock["counterfactual_lock_sha256"])
    plan: list[dict[str, Any]] = []
    kind_counts: Counter[str] = Counter()
    for spec in specs:
        case = cases[spec["case_id"]]
        task_kind = classify_runtime_task(case)
        kind_counts[task_kind] += 1
        state_packet = packets.get(case.get("state_packet_ref"))
        state_claims = materialize_state_claims(case["question"], state_packet)
        if task_kind == "STATE_ONLY" and len(state_claims) != 1:
            raise ValueError("T0 materializer failed its one-claim contract")
        if task_kind == "STATE_AND_GUIDANCE" and len(state_claims) != 1:
            raise ValueError("T2 materializer failed its one-claim contract")
        source = contexts[(spec["case_id"], spec["action"])]
        guidance_needed = task_kind != "STATE_ONLY"
        evidence = (
            list(source.supplied_chunks)
            if guidance_needed and spec["action"] != "OFF"
            else []
        )
        prompt: str | None = None
        aliases: list[dict[str, Any]] = []
        prompt_tokens = 0
        prompt_sha: str | None = None
        if guidance_needed:
            prompt, aliases = render_guidance_prompt(case["question"], evidence)
            prompt_tokens = client.count_prompt_tokens(prompt)
            if prompt_tokens > MAX_PROMPT_TOKENS:
                raise ValueError(f"B4 prompt exceeds frozen 32768-context contract: {spec['case_id']}")
            prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        plan.append(
            {
                **spec,
                "task_kind": task_kind,
                "question": case["question"],
                "question_sha256": hashlib.sha256(case["question"].encode("utf-8")).hexdigest(),
                "state_packet_sha256": (
                    state_packet.get("packet_sha256") if state_packet is not None else None
                ),
                "state_claims": [claim.to_dict() for claim in state_claims],
                "evidence_chunks": evidence,
                "evidence_aliases": aliases,
                "guidance_prompt": prompt,
                "guidance_prompt_sha256": prompt_sha,
                "preflight_prompt_tokens": prompt_tokens,
                "model_call_count": int(guidance_needed),
                "new_retrieval_calls": 0,
                "new_bridge_calls": 0,
                "b2_source": source.reuse_manifest_row(),
                "b2_retrieval_ranking_sha256": source.retrieval_ranking_sha256,
                "b2_supplied_chunks_sha256": source.supplied_chunks_sha256,
                "b2_supplied_chunk_ids_sha256": source.supplied_chunk_ids_sha256,
                "b2_historical_retrieval_calls": source.source_retrieval_calls,
                "b2_historical_bridge_calls": source.source_bridge_call_count,
                "b2_historical_bridge_input_tokens": source.source_bridge_input_tokens,
                "b2_historical_bridge_output_tokens": source.source_bridge_output_tokens,
                "b2_historical_retrieval_latency_ms": source.source_retrieval_latency_ms,
                "b2_historical_bridge_latency_ms": source.source_bridge_latency_ms,
            }
        )
    # The plan is arm-level: each of 20 cases per task kind appears under all three actions.
    expected = {"STATE_ONLY": 60, "GUIDANCE_ONLY": 60, "STATE_AND_GUIDANCE": 60}
    if kind_counts != expected:
        raise ValueError(f"runtime-visible tasks do not match the frozen 20/20/20 structure: {kind_counts}")
    if sum(row["model_call_count"] for row in plan) != 120:
        raise AssertionError("frozen B4 plan must contain exactly 120 guidance calls")
    if any(row["new_retrieval_calls"] or row["new_bridge_calls"] for row in plan):
        raise AssertionError("B4 plan must not schedule new retrieval or bridge calls")
    plan_path = private_root / "execution_plan.json"
    plan_sha = canonical_sha256(plan)
    if plan_path.exists():
        current = _read_json(plan_path)
        if current.get("plan_sha256") != plan_sha:
            raise FileExistsError("B4 execution plan already exists with a different identity")
    else:
        _write_json(plan_path, {"plan": plan, "plan_sha256": plan_sha})
    return plan


def _wait_for_idle(client: LlamaServerClient, timeout_seconds: int = 1800) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        slots = client.slots()
        if slots and all(row.get("is_processing") is False for row in slots):
            return
        time.sleep(2)
    raise TimeoutError("shared llama.cpp slot remained busy; no generation call was made")


def _append_call(
    ledger: Path,
    *,
    call_id: str,
    plan_row: dict[str, Any],
    status: str,
    started_at: str,
    **extra: Any,
) -> None:
    _append_jsonl(
        ledger,
        {
            "call_id": call_id,
            "case_id": plan_row["case_id"],
            "action": plan_row["action"],
            "run_id": plan_row["run_id"],
            "prompt_sha256": plan_row["guidance_prompt_sha256"],
            "call_ordinal": plan_row["call_ordinal"],
            "status": status,
            "started_at": started_at,
            "recorded_at": datetime.now(UTC).isoformat(),
            **extra,
        },
    )


def _arm_path(private_root: Path, plan_row: dict[str, Any]) -> Path:
    return (
        private_root
        / "arms"
        / plan_row["case_id"]
        / plan_row["action"]
        / f"{plan_row['run_id']}.json"
    )


def _complete_arm(
    *,
    plan_row: dict[str, Any],
    lock_sha: str,
    client: LlamaServerClient,
    private_root: Path,
    call_ledger: Path,
) -> dict[str, Any]:
    claims = plan_row["state_claims"]
    raw_response: dict[str, Any] | None = None
    request: dict[str, Any] | None = None
    completion_latency_ms = 0.0
    if plan_row["model_call_count"]:
        call_id = canonical_sha256([lock_sha, plan_row["case_id"], plan_row["action"]])
        started_at = datetime.now(UTC).isoformat()
        plan_row["call_ordinal"] = plan_row.pop("_call_ordinal")
        _wait_for_idle(client)
        _append_call(
            call_ledger,
            call_id=call_id,
            plan_row=plan_row,
            status="STARTED",
            started_at=started_at,
        )
        try:
            completion = client.complete(plan_row["guidance_prompt"])
        except Exception as exc:
            _append_call(
                call_ledger,
                call_id=call_id,
                plan_row=plan_row,
                status="FAILED",
                started_at=started_at,
                error_type=type(exc).__name__,
            )
            _write_json(
                private_root / "execution_failure.json",
                {
                    "call_id": call_id,
                    "case_id": plan_row["case_id"],
                    "action": plan_row["action"],
                    "failure_type": type(exc).__name__,
                    "timestamp_utc": datetime.now(UTC).isoformat(),
                    "retry_performed": False,
                },
            )
            raise
        completion_latency_ms = completion.latency_ms
        request = dict(completion.request_payload)
        raw_response = {
            "text": completion.text,
            "finish_reason": completion.finish_reason,
            "input_tokens": completion.input_tokens,
            "output_tokens": completion.output_tokens,
            "latency_ms": completion.latency_ms,
        }
        _append_call(
            call_ledger,
            call_id=call_id,
            plan_row=plan_row,
            status="COMPLETED",
            started_at=started_at,
            finish_reason=completion.finish_reason,
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
            latency_ms=completion.latency_ms,
            empty_output=not bool(completion.text.strip()),
        )
    else:
        plan_row["call_ordinal"] = None

    guidance_text = raw_response["text"].strip() if raw_response else ""
    citations, invented_aliases = extract_citations(guidance_text, plan_row["evidence_aliases"])
    final_response = (
        materialize_final_response(
            task_kind=plan_row["task_kind"],
            state_claims=[],
            guidance_text=guidance_text,
            alias_map=plan_row["evidence_aliases"],
        )
        if plan_row["task_kind"] == "GUIDANCE_ONLY"
        else None
    )
    if plan_row["task_kind"] == "STATE_ONLY":
        state_objects = [StateClaim(**row) for row in claims]
        final_response = materialize_final_response(
            task_kind=plan_row["task_kind"],
            state_claims=state_objects,
            guidance_text="",
            alias_map=[],
        )
    elif plan_row["task_kind"] == "STATE_AND_GUIDANCE":
        state_objects = [StateClaim(**row) for row in claims]
        final_response = materialize_final_response(
            task_kind=plan_row["task_kind"],
            state_claims=state_objects,
            guidance_text=guidance_text,
            alias_map=plan_row["evidence_aliases"],
        )
    payload = {
        "schema_version": "rag-e5-e5b4-counterfactual-arm-v1",
        "status": "COMPLETE",
        "case_id": plan_row["case_id"],
        "action": plan_row["action"],
        "run_id": plan_row["run_id"],
        "protocol_lock_sha256": lock_sha,
        "question_sha256": plan_row["question_sha256"],
        "state_packet_sha256": plan_row["state_packet_sha256"],
        "runtime_task_kind": plan_row["task_kind"],
        "state_claims": claims,
        "guidance_request": request,
        "guidance_request_sha256": (
            canonical_sha256(request) if request is not None else None
        ),
        "guidance_prompt_sha256": plan_row["guidance_prompt_sha256"],
        "preflight_prompt_tokens": plan_row["preflight_prompt_tokens"],
        "guidance_raw_response": raw_response,
        "guidance_raw_response_sha256": (
            canonical_sha256(raw_response) if raw_response is not None else None
        ),
        "evidence_aliases": plan_row["evidence_aliases"],
        "evidence_chunks": plan_row["evidence_chunks"],
        "resolved_citation_chunk_ids": citations,
        "invented_evidence_aliases": invented_aliases,
        "final_materialized_response": final_response,
        "new_model_calls": plan_row["model_call_count"],
        "new_retrieval_calls": 0,
        "new_bridge_calls": 0,
        "b2_source": plan_row["b2_source"],
        "b2_retrieval_ranking_sha256": plan_row["b2_retrieval_ranking_sha256"],
        "b2_supplied_chunks_sha256": plan_row["b2_supplied_chunks_sha256"],
        "b2_historical_retrieval_calls": plan_row["b2_historical_retrieval_calls"],
        "b2_historical_bridge_calls": plan_row["b2_historical_bridge_calls"],
        "b2_historical_bridge_input_tokens": plan_row["b2_historical_bridge_input_tokens"],
        "b2_historical_bridge_output_tokens": plan_row["b2_historical_bridge_output_tokens"],
        "b2_historical_retrieval_latency_ms": plan_row[
            "b2_historical_retrieval_latency_ms"
        ],
        "b2_historical_bridge_latency_ms": plan_row["b2_historical_bridge_latency_ms"],
        "guidance_input_tokens": (
            raw_response.get("input_tokens") if raw_response is not None else 0
        ),
        "guidance_output_tokens": (
            raw_response.get("output_tokens") if raw_response is not None else 0
        ),
        "guidance_latency_ms": completion_latency_ms,
        "total_latency_ms": completion_latency_ms,
    }
    return payload


def _write_arm(path: Path, payload: dict[str, Any]) -> str:
    if path.exists():
        raise FileExistsError(f"B4 arm is immutable: {path}")
    digest = canonical_sha256(payload)
    _write_json(path, {**payload, "completion_sha256": digest})
    return digest


def run_smoke(
    *,
    protocol_path: Path = DEFAULT_PROTOCOL_PATH,
    private_root: Path = DEFAULT_B4_PRIVATE_ROOT,
) -> dict[str, Any]:
    """Run at most one small inference for each frozen action; never score quality."""
    lock = _read_json(protocol_path)
    lock_sha = verify_protocol_lock(lock)
    verify_frozen_code(lock)
    client = LlamaServerClient(lock["runtime"]["server_url"])
    server = verify_server(client, lock)
    b3_identity = verify_b3_identity(lock)
    frozen = load_verified_b2(lock)
    plan = make_execution_plan(frozen=frozen, client=client, private_root=private_root)
    sample = next(row for row in plan if row["task_kind"] == "GUIDANCE_ONLY")
    outputs: list[dict[str, Any]] = []
    for action in ACTION_ORDER:
        row = next(
            item
            for item in plan
            if item["case_id"] == sample["case_id"] and item["action"] == action
        )
        if not row["model_call_count"]:
            raise AssertionError("smoke case must be a guidance-only task")
        path = private_root / "smoke" / f"{action}.json"
        if path.exists():
            raise FileExistsError("B4 smoke calls are immutable; refusing a second smoke run")
        _wait_for_idle(client)
        completion = client.complete(row["guidance_prompt"])
        if not completion.text.strip():
            raise ValueError("B4 smoke output is empty")
        citations, invented = extract_citations(completion.text, row["evidence_aliases"])
        payload = {
            "action_label_for_smoke_only": action,
            "case_id": row["case_id"],
            "prompt_sha256": row["guidance_prompt_sha256"],
            "request": dict(completion.request_payload),
            "response": {
                "text": completion.text,
                "finish_reason": completion.finish_reason,
                "input_tokens": completion.input_tokens,
                "output_tokens": completion.output_tokens,
                "latency_ms": completion.latency_ms,
            },
            "alias_map": row["evidence_aliases"],
            "resolved_citation_chunk_ids": citations,
            "invented_evidence_aliases": invented,
        }
        _write_json(path, payload)
        outputs.append(
            {
                "action": action,
                "non_empty": True,
                "finish_reason": completion.finish_reason,
                "alias_map_rows": len(row["evidence_aliases"]),
                "citation_regex_exercised": True,
                "output_file_sha256": sha256_file(path),
            }
        )
    result = {
        "schema_version": "rag-e5-e5b4-smoke-v1",
        "status": "PASS_PATH_ONLY_NOT_QUALITY",
        "protocol_lock_sha256": lock_sha,
        "server": server,
        "b3_unchanged_identity": b3_identity,
        "model_calls": len(outputs),
        "outputs": outputs,
        "quality_scored": False,
        "retrieval_calls": 0,
        "bridge_calls": 0,
        "teacher_opened": False,
        "202608_opened": False,
    }
    _write_json(private_root / "smoke" / "smoke_report.json", result)
    return result


def run_counterfactual(
    *,
    protocol_path: Path = DEFAULT_PROTOCOL_PATH,
    manifest_path: Path = DEFAULT_EXECUTION_MANIFEST_PATH,
    private_root: Path = DEFAULT_B4_PRIVATE_ROOT,
) -> dict[str, Any]:
    lock = _read_json(protocol_path)
    lock_sha = verify_protocol_lock(lock)
    verify_frozen_code(lock)
    if manifest_path.exists():
        raise FileExistsError("B4 execution manifest already exists; refusing to rerun calls")
    smoke_path = private_root / "smoke" / "smoke_report.json"
    if not smoke_path.is_file():
        raise ValueError("complete the frozen three-call path-only smoke before B4 execution")
    smoke = _read_json(smoke_path)
    if (
        smoke.get("status") != "PASS_PATH_ONLY_NOT_QUALITY"
        or smoke.get("protocol_lock_sha256") != lock_sha
        or smoke.get("model_calls") != 3
        or smoke.get("quality_scored") is not False
    ):
        raise ValueError("B4 smoke must pass its path-only checks under the frozen protocol")
    client = LlamaServerClient(lock["runtime"]["server_url"])
    server = verify_server(client, lock)
    b3_identity = verify_b3_identity(lock)
    frozen = load_verified_b2(lock)
    plan = make_execution_plan(frozen=frozen, client=client, private_root=private_root)
    ledger = private_root / "call_ledger.jsonl"
    if ledger.exists() or (private_root / "execution_failure.json").exists():
        raise FileExistsError("B4 call history already exists; retries are forbidden")
    arm_records: list[dict[str, Any]] = []
    counters = Counter()
    input_tokens: list[int] = []
    output_tokens: list[int] = []
    latencies: list[float] = []
    call_ordinal = 0
    for plan_row in plan:
        if plan_row["model_call_count"]:
            call_ordinal += 1
            plan_row["_call_ordinal"] = call_ordinal
        arm = _complete_arm(
            plan_row=plan_row,
            lock_sha=lock_sha,
            client=client,
            private_root=private_root,
            call_ledger=ledger,
        )
        arm_path = _arm_path(private_root, plan_row)
        arm_sha = _write_arm(arm_path, arm)
        arm_records.append(
            {
                "case_id": arm["case_id"],
                "action": arm["action"],
                "run_id": arm["run_id"],
                "relative_path": str(arm_path.relative_to(private_root)).replace("\\", "/"),
                "file_sha256": sha256_file(arm_path),
                "completion_sha256": arm_sha,
                "new_model_calls": arm["new_model_calls"],
                "b2_historical_retrieval_calls": arm["b2_historical_retrieval_calls"],
                "b2_historical_bridge_calls": arm["b2_historical_bridge_calls"],
                "b2_historical_bridge_input_tokens": arm[
                    "b2_historical_bridge_input_tokens"
                ],
                "b2_historical_bridge_output_tokens": arm[
                    "b2_historical_bridge_output_tokens"
                ],
                "b2_historical_retrieval_latency_ms": arm[
                    "b2_historical_retrieval_latency_ms"
                ],
                "b2_historical_bridge_latency_ms": arm[
                    "b2_historical_bridge_latency_ms"
                ],
            }
        )
        counters["completed_arms"] += 1
        counters["guidance_calls"] += arm["new_model_calls"]
        counters["empty_outputs"] += int(
            arm["guidance_raw_response"] is not None
            and not arm["guidance_raw_response"]["text"].strip()
        )
        counters["finish_reason_length"] += int(
            arm["guidance_raw_response"] is not None
            and arm["guidance_raw_response"].get("finish_reason") == "length"
        )
        if isinstance(arm["guidance_input_tokens"], int):
            input_tokens.append(arm["guidance_input_tokens"])
        if isinstance(arm["guidance_output_tokens"], int):
            output_tokens.append(arm["guidance_output_tokens"])
        latencies.append(float(arm["guidance_latency_ms"]))
        print(
            json.dumps(
                {
                    "completed_arms": counters["completed_arms"],
                    "expected_arms": 180,
                    "guidance_calls": counters["guidance_calls"],
                    "expected_guidance_calls": 120,
                    "last_action": arm["action"],
                    "last_guidance_call": bool(arm["new_model_calls"]),
                    "empty": bool(
                        arm["guidance_raw_response"] is not None
                        and not arm["guidance_raw_response"]["text"].strip()
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if counters["completed_arms"] != 180 or counters["guidance_calls"] != 120:
        raise AssertionError("B4 run did not complete its frozen 180-arm/120-call plan")
    # Revalidate the immutable B2 sources after generation and before freezing B4 outputs.
    frozen_after = load_verified_b2(lock)
    if frozen_after.artifact_set_sha256 != frozen.artifact_set_sha256:
        raise ValueError("B2 artifacts changed during B4 execution")
    if verify_b3_identity(lock) != b3_identity:
        raise ValueError("B3 artifacts changed during B4 execution")
    call_rows = [
        json.loads(line)
        for line in ledger.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    starts = [row for row in call_rows if row["status"] == "STARTED"]
    completed_calls = [row for row in call_rows if row["status"] == "COMPLETED"]
    if len(starts) != 120 or len(completed_calls) != 120:
        raise ValueError("B4 guidance ledger must contain exactly 120 unique completed calls")
    if len({row["call_id"] for row in starts}) != 120:
        raise ValueError("B4 guidance ledger has repeated call identities")
    manifest_body = {
        "schema_version": "rag-e5-e5b4-execution-manifest-v1",
        "status": "ALL_B4_ARMS_FROZEN_BEFORE_SCORING",
        "protocol_lock_sha256": lock_sha,
        "protocol_lock_file_sha256": sha256_file(protocol_path),
        "protocol_commit_sha": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip(),
        "task_set_sha256": lock["task_set_sha256"],
        "state_set_sha256": lock["state_set_sha256"],
        "b2_artifact_set_sha256": frozen.artifact_set_sha256,
        "b2_artifact_reuse_manifest_sha256": frozen.reuse_manifest_sha256,
        **b3_identity,
        "smoke_report_sha256": sha256_file(smoke_path),
        "shared_server_runtime": server,
        "arms_expected": 180,
        "arms_completed": 180,
        "guidance_calls_expected": 120,
        "guidance_calls_completed": 120,
        "new_retrieval_calls": 0,
        "new_bridge_calls": 0,
        "b2_historical_retrieval_calls_reused": sum(
            row["b2_historical_retrieval_calls"] for row in arm_records
        ),
        "b2_historical_bridge_calls_reused": sum(
            row["b2_historical_bridge_calls"] for row in arm_records
        ),
        "b2_historical_bridge_input_tokens_reused": sum(
            row["b2_historical_bridge_input_tokens"] or 0 for row in arm_records
        ),
        "b2_historical_bridge_output_tokens_reused": sum(
            row["b2_historical_bridge_output_tokens"] or 0 for row in arm_records
        ),
        "b2_historical_retrieval_latency_ms_reused": sum(
            row["b2_historical_retrieval_latency_ms"] for row in arm_records
        ),
        "b2_historical_bridge_latency_ms_reused": sum(
            row["b2_historical_bridge_latency_ms"] for row in arm_records
        ),
        "empty_outputs": counters["empty_outputs"],
        "finish_reason_length": counters["finish_reason_length"],
        "transport_failures": 0,
        "total_input_tokens": sum(input_tokens) if len(input_tokens) == 120 else None,
        "total_output_tokens": sum(output_tokens) if len(output_tokens) == 120 else None,
        "token_usage_complete": len(input_tokens) == len(output_tokens) == 120,
        "mean_generation_latency_ms": sum(latencies) / len(latencies),
        "call_ledger_sha256": sha256_file(ledger),
        "external_arm_root": str(private_root),
        "teacher_opened": False,
        "202608_opened": False,
        "artifacts": arm_records,
    }
    manifest_body["artifact_set_sha256"] = canonical_sha256(arm_records)
    manifest_body["execution_manifest_sha256"] = canonical_sha256(manifest_body)
    _write_json(manifest_path, manifest_body)
    return manifest_body


__all__ = [
    "ACTION_ORDER",
    "DEFAULT_B4_PRIVATE_ROOT",
    "DEFAULT_EXECUTION_MANIFEST_PATH",
    "DEFAULT_PROTOCOL_PATH",
    "MAX_PROMPT_TOKENS",
    "load_verified_b2",
    "make_execution_plan",
    "run_counterfactual",
    "run_smoke",
    "read_and_verify_b3_identity",
    "verify_frozen_code",
    "verify_protocol_lock",
    "verify_server",
    "verify_b3_identity",
]
