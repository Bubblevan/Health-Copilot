"""Issue exactly the frozen R4-P01 sampler-init gate request on loopback."""

from __future__ import annotations

import datetime as dt
import ctypes
import hashlib
import http.client
import json
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from tools.research.memory import mem3b0q_r4

ROOT = Path(__file__).resolve().parents[3]
MEMORY_DOCS = ROOT / "docs" / "research" / "memory"
RUN_DIR = ROOT / "runs" / "memory" / "mem3" / "mem3b0q-r4-identity-qualification-v1"
GATE_DIR = RUN_DIR / "pass1_r4-p01_sampler_init_gate"
LOG_PATH = RUN_DIR / "llama-server.log"
PACK_PATH = MEMORY_DOCS / "mem3b0q_r4_control_pack_v1.json"
SCHEMA_PATH = MEMORY_DOCS / "mem3b0q_r4_response.schema.json"
MANIFEST_PATH = MEMORY_DOCS / "mem3b0q_r4_freeze_manifest.json"
MANIFEST_SIDECAR_PATH = MEMORY_DOCS / "mem3b0q_r4_freeze_manifest.sha256"
RUNNER_LOCK_PATH = MEMORY_DOCS / "mem3b0q_r4_gate_runner_lock_v1.json"
TRACE_CONTRACT_PATH = MEMORY_DOCS / "mem3b0q_r4_sampler_trace_instrumentation_v1.md"
HOST = "127.0.0.1"
PORT = 8081
ENDPOINT_PATH = "/v1/chat/completions"
SERVER_PATH = (
    r"C:\Users\bubblevan\AppData\Local\Microsoft\WinGet\Packages"
    r"\ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe\llama-server.exe"
)
MODEL_PATH = r"E:\Health-Copilot-Models\models\qwen3-8b\Qwen3-8B-Q4_K_M.gguf"
SERVER_SHA256 = "3a8aea5f889c4b4c2ec41c98f4e1ed484bb7a40c4096883acb23d3cfe26b59fb"
MODEL_SHA256 = "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785"
SCHEMA_SHA256 = "5d0b1ed81fa8c208e4db19dba19a9bf11d8746bbc723f0b2e5876df392ade840"
SCHEMA_CANONICAL_SHA256 = "10f976e77594c8e7a31f81ee73d93e964963e2e9e6bdf571ed697d726b246cbc"
FROZEN_MANIFEST_SHA256 = "278dd2079d0196a5d7b58111b3604a5a11ebe92bfdc22bc5fe868e8d3b547e1f"
REQUIRED_FROZEN_ARTIFACTS = frozenset(
    {
        "docs/research/memory/mem3b0q_r4_control_pack_v1.json",
        "docs/research/memory/mem3b0q_r4_control_pack_draft.json",
        "docs/research/memory/mem3b0q_r4_response.schema.json",
        "docs/research/memory/mem_3b0q_r4_protocol_v1.md",
        "docs/research/memory/mem_3b0q_r4_protocol_draft.md",
        "docs/research/memory/mem_3b0q_r4_witness_contract_amendment.md",
        "docs/research/memory/mem3b0q_r4_overlap_audit.json",
        "docs/research/memory/mem3b0q_r4_offline_preflight.json",
        "tools/research/memory/mem3b0q_r4.py",
        "tests/test_mem3b0q_r4_contract.py",
    }
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    return _sha256(path.read_bytes())


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _ps_process_snapshot() -> dict[str, Any]:
    script = rf"""
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [Console]::OutputEncoding
$listeners = @(Get-NetTCPConnection -LocalPort {PORT} -State Listen)
if ($listeners.Count -ne 1) {{ throw 'expected_exactly_one_listener' }}
$ownerPid = [int]$listeners[0].OwningProcess
$proc = Get-CimInstance Win32_Process -Filter "ProcessId = $ownerPid"
if (-not $proc -or $proc.ExecutablePath -ne '{SERVER_PATH}') {{ throw 'listener_process_path_mismatch' }}
function Get-FileSha256([string]$path) {{
  $algorithm = [System.Security.Cryptography.SHA256]::Create()
  $stream = [System.IO.File]::OpenRead($path)
  try {{
    return [BitConverter]::ToString($algorithm.ComputeHash($stream)).Replace('-', '').ToLowerInvariant()
  }} finally {{
    $stream.Dispose()
    $algorithm.Dispose()
  }}
}}
$binaryHash = Get-FileSha256 $proc.ExecutablePath
$modelHash = Get-FileSha256 '{MODEL_PATH}'
[pscustomobject]@{{
  pid = $ownerPid
  addresses = @($listeners | ForEach-Object {{ $_.LocalAddress }})
  executable_path = $proc.ExecutablePath
  executable_sha256 = $binaryHash
  model_path = '{MODEL_PATH}'
  model_sha256 = $modelHash
  command_line = $proc.CommandLine
}} | ConvertTo-Json -Compress
"""
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8-sig",
        timeout=90,
    )
    if result.returncode != 0:
        raise RuntimeError(f"process_preflight_failed:{result.stderr.strip()}")
    try:
        snapshot = json.loads(result.stdout.strip())
    except json.JSONDecodeError as exc:
        raise RuntimeError("process_preflight_invalid_json") from exc
    return snapshot


def _http_json(path: str, *, timeout: int = 15) -> tuple[int, bytes, Any]:
    connection = http.client.HTTPConnection(HOST, PORT, timeout=timeout)
    try:
        connection.request("GET", path, headers={"Accept": "application/json"})
        response = connection.getresponse()
        body = response.read()
        parsed = json.loads(body.decode("utf-8")) if body else None
        return response.status, body, parsed
    finally:
        connection.close()


def _service_snapshot() -> dict[str, Any]:
    health_status, health_raw, health = _http_json("/health")
    props_status, props_raw, props = _http_json("/props")
    models_status, models_raw, models = _http_json("/v1/models")
    slots_status, slots_raw, slots = _http_json("/slots")
    if any(status != 200 for status in (health_status, props_status, models_status, slots_status)):
        raise RuntimeError("local_endpoint_preflight_http_failure")
    return {
        "health": health,
        "props": props,
        "models": models,
        "slots": slots,
        "raw_sha256": {
            "/health": _sha256(health_raw),
            "/props": _sha256(props_raw),
            "/v1/models": _sha256(models_raw),
            "/slots": _sha256(slots_raw),
        },
    }


def _windows_argv(command_line: str) -> list[str]:
    argc = ctypes.c_int()
    shell32 = ctypes.windll.shell32
    shell32.CommandLineToArgvW.argtypes = [
        ctypes.c_wchar_p,
        ctypes.POINTER(ctypes.c_int),
    ]
    shell32.CommandLineToArgvW.restype = ctypes.POINTER(ctypes.c_wchar_p)
    argv = shell32.CommandLineToArgvW(command_line, ctypes.byref(argc))
    if not argv:
        raise RuntimeError("windows_command_line_parse_failed")
    try:
        return [argv[index] for index in range(argc.value)]
    finally:
        kernel32 = ctypes.windll.kernel32
        kernel32.LocalFree.argtypes = [ctypes.c_void_p]
        kernel32.LocalFree.restype = ctypes.c_void_p
        kernel32.LocalFree(ctypes.cast(argv, ctypes.c_void_p))


def _validate_server_command_line(command_line: str) -> None:
    argv = _windows_argv(command_line)
    expected_options = {
        "-m": MODEL_PATH,
        "--host": "127.0.0.1",
        "--port": "8081",
        "--ctx-size": "131072",
        "--n-predict": "8192",
        "--rope-scaling": "yarn",
        "--rope-scale": "4",
        "--yarn-orig-ctx": "32768",
        "--override-kv": "qwen3.context_length=int:131072",
        "--cache-type-k": "q4_0",
        "--cache-type-v": "q4_0",
        "--n-gpu-layers": "99",
        "--flash-attn": "on",
        "--parallel": "1",
        "--log-file": str(LOG_PATH),
        "--verbosity": "5",
    }
    if not argv or argv[0].casefold() != SERVER_PATH.casefold():
        raise RuntimeError("server_command_executable_mismatch")
    actual_options: dict[str, str] = {}
    if len(argv[1:]) % 2:
        raise RuntimeError("server_command_line_ambiguous")
    for index in range(1, len(argv), 2):
        option, value = argv[index : index + 2]
        normalized_option = option.casefold()
        if normalized_option not in expected_options:
            raise RuntimeError("server_command_line_unknown_argument")
        if normalized_option in actual_options:
            raise RuntimeError("server_command_line_duplicate_argument")
        actual_options[normalized_option] = value
    expected_options = {key.casefold(): value for key, value in expected_options.items()}
    actual_options = {key: value.casefold() for key, value in actual_options.items()}
    expected_options = {key: value.casefold() for key, value in expected_options.items()}
    if actual_options != expected_options:
        raise RuntimeError("server_command_line_mismatch")


def _validate_preflight(process: dict[str, Any], service: dict[str, Any]) -> None:
    if process.get("addresses") != [HOST]:
        raise RuntimeError("listener_not_loopback_only")
    if process.get("executable_sha256", "").lower() != SERVER_SHA256:
        raise RuntimeError("server_binary_hash_mismatch")
    if process.get("model_sha256", "").lower() != MODEL_SHA256:
        raise RuntimeError("model_hash_mismatch")
    if process.get("model_path") != MODEL_PATH:
        raise RuntimeError("process_model_path_mismatch")
    _validate_server_command_line(process.get("command_line", ""))

    health = service["health"]
    props = service["props"]
    models = service["models"].get("data", [])
    if health.get("status") != "ok":
        raise RuntimeError("server_unhealthy")
    if props.get("build_info") != "b10068-571d0d540":
        raise RuntimeError("server_build_mismatch")
    if props.get("model_path") != MODEL_PATH or props.get("model_alias") != MODEL_PATH:
        raise RuntimeError("served_model_path_mismatch")
    if props.get("default_generation_settings", {}).get("n_ctx") != 131072:
        raise RuntimeError("server_context_mismatch")
    if props.get("total_slots") != 1 or not models or models[0].get("id") != MODEL_PATH:
        raise RuntimeError("served_model_or_slot_mismatch")
    slot = service["slots"]
    if isinstance(slot, list):
        if len(slot) != 1:
            raise RuntimeError("slot_count_mismatch")
        slot = slot[0]
    if slot.get("n_ctx") != 131072 or slot.get("is_processing"):
        raise RuntimeError("slot_not_idle_or_context_mismatch")
    params = props.get("default_generation_settings", {}).get("params", {})
    if (
        params.get("top_k") != 40
        or abs(float(params.get("top_p", 0.0)) - 0.95) > 0.001
        or abs(float(params.get("min_p", 0.0)) - 0.05) > 0.001
        or abs(float(params.get("repeat_penalty", 0.0)) - 1.0) > 0.001
    ):
        raise RuntimeError("server_sampler_defaults_mismatch")


def _load_frozen_inputs() -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]
]:
    manifest_bytes = MANIFEST_PATH.read_bytes()
    manifest_hash = _sha256(manifest_bytes)
    sidecar_tokens = MANIFEST_SIDECAR_PATH.read_text(encoding="ascii").split()
    if (
        manifest_hash != FROZEN_MANIFEST_SHA256
        or not sidecar_tokens
        or sidecar_tokens[0].lower() != FROZEN_MANIFEST_SHA256
    ):
        raise RuntimeError("frozen_manifest_digest_mismatch")
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    if (
        manifest.get("status") != "FROZEN"
        or manifest.get("protocol_lock") != "MEM3B0Q_R4_PROTOCOL_LOCK=YES"
        or manifest.get("qualification_status")
        != "MEM3B0Q_R4_IDENTITY_QUALIFICATION=PENDING"
        or manifest.get("b1_status") != "MEM3B0Q_MEM3B1_READY=NO"
    ):
        raise RuntimeError("frozen_manifest_state_mismatch")
    verification = manifest.get("verification_state", {})
    if (
        verification.get("schema_converter") != "VERIFIED"
        or verification.get("sampler_parser") != "NOT_VERIFIED"
        or verification.get("r4_inference_started") is not False
        or verification.get("b1_started") is not False
    ):
        raise RuntimeError("frozen_verification_state_mismatch")
    artifacts = manifest.get("artifacts", {})
    if set(artifacts) != REQUIRED_FROZEN_ARTIFACTS:
        raise RuntimeError("frozen_artifact_set_mismatch")
    for relative, expected_hash in artifacts.items():
        actual_hash = _sha256((ROOT / relative).read_bytes())
        if actual_hash != expected_hash:
            raise RuntimeError(f"frozen_artifact_hash_mismatch:{relative}")
    runner_hash = _file_sha256(Path(__file__).resolve())
    runner_lock = json.loads(RUNNER_LOCK_PATH.read_text(encoding="utf-8"))
    if (
        runner_lock.get("status") != "FROZEN"
        or runner_lock.get("manifest_sha256") != FROZEN_MANIFEST_SHA256
        or runner_lock.get("runner_path") != "tools/research/memory/run_mem3b0q_r4_gate.py"
        or runner_lock.get("runner_sha256") != runner_hash
        or runner_lock.get("trace_contract_path")
        != "docs/research/memory/mem3b0q_r4_sampler_trace_instrumentation_v1.md"
        or runner_lock.get("trace_contract_sha256")
        != _file_sha256(TRACE_CONTRACT_PATH)
        or runner_lock.get("sampler_parser_before") != "NOT_VERIFIED"
        or runner_lock.get("b1_status") != "MEM3B0Q_MEM3B1_READY=NO"
    ):
        raise RuntimeError("runner_lock_mismatch")
    pack_bytes = PACK_PATH.read_bytes()
    schema_bytes = SCHEMA_PATH.read_bytes()
    if _sha256(schema_bytes) != SCHEMA_SHA256:
        raise RuntimeError("schema_file_hash_mismatch")
    pack = json.loads(pack_bytes.decode("utf-8"))
    schema = json.loads(schema_bytes.decode("utf-8"))
    if pack.get("status") != "FROZEN" or pack.get("scope_id") != mem3b0q_r4.FROZEN_SCOPE_ID:
        raise RuntimeError("frozen_pack_identity_mismatch")
    if mem3b0q_r4.canonical_json_sha256(schema) != SCHEMA_CANONICAL_SHA256:
        raise RuntimeError("canonical_schema_hash_mismatch")
    return manifest, pack, schema, runner_lock


def _capture_log_delta(log_offset: int, result: dict[str, Any]) -> None:
    delta = LOG_PATH.read_bytes()[log_offset:]
    (GATE_DIR / "server_log_delta.txt").write_bytes(delta)
    log_text = delta.decode("utf-8", errors="replace")
    result["server_log_delta_sha256"] = _sha256(delta)
    result["server_log_candidate_lines"] = [
        line
        for line in log_text.splitlines()
        if re.search(r"schema|grammar|sampler", line, flags=re.IGNORECASE)
    ]
    result["server_log_error_lines"] = [
        line
        for line in log_text.splitlines()
        if re.search(r"schema|grammar|sampler", line, flags=re.IGNORECASE)
        and re.search(r"error|failed|invalid|reject", line, flags=re.IGNORECASE)
    ]


def _observe_active_slots(
    stop_event: threading.Event,
    source_id: str,
    proposition_text: str,
    observations: list[dict[str, Any]],
    failures: list[str],
) -> None:
    while not stop_event.is_set():
        try:
            status, _, slots = _http_json("/slots", timeout=3)
            if status != 200:
                raise RuntimeError(f"slots_http_status:{status}")
            if isinstance(slots, dict):
                slots = slots.get("slots", [])
            if not isinstance(slots, list):
                raise RuntimeError("slots_response_not_list")
            observations.append(
                {
                    "captured_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "slots": [
                        {
                            "slot_id": slot.get("id"),
                            "task_id": slot.get("id_task"),
                            "is_processing": slot.get("is_processing") is True,
                            "prompt_sha256": _sha256(
                                str(slot.get("prompt", "")).encode("utf-8")
                            ),
                            "prompt_contains_source_id": source_id
                            in str(slot.get("prompt", "")),
                            "prompt_contains_proposition": proposition_text
                            in str(slot.get("prompt", "")),
                        }
                        for slot in slots
                    ],
                }
            )
        except Exception as exc:
            failures.append(f"{type(exc).__name__}:{exc}")
        stop_event.wait(0.05)


def _evaluate_sampler_evidence(
    log_text: str,
    observations: list[dict[str, Any]],
    *,
    source_id: str,
    proposition_text: str,
    observer_failures: list[str],
) -> dict[str, Any]:
    active = [
        slot
        for observation in observations
        for slot in observation.get("slots", [])
        if slot.get("is_processing")
    ]
    task_ids = {slot.get("task_id") for slot in active}
    slot_ids = {slot.get("slot_id") for slot in active}
    input_witnessed = bool(active) and all(
        slot.get("prompt_contains_source_id")
        and slot.get("prompt_contains_proposition")
        for slot in active
    )
    lines = log_text.splitlines()
    launch_rows = [
        (index, int(slot), int(task))
        for index, line in enumerate(lines)
        if (match := re.search(
            r"slot\s+\S+:\s+id\s+(\d+)\s+\|\s+task\s+(-?\d+)\s+\|\s+launching slot\s*:",
            line,
            flags=re.IGNORECASE,
        ))
        for slot, task in [match.groups()]
    ]
    init_rows = [
        (index, int(slot), int(task))
        for index, line in enumerate(lines)
        if (match := re.search(
            r"slot\s+init_sampler:\s+id\s+(\d+)\s+\|\s+task\s+(\d+)\s+\|\s+init sampler",
            line,
            flags=re.IGNORECASE,
        ))
        for slot, task in [match.groups()]
    ]
    processing_rows = [
        (index, int(slot), int(task))
        for index, line in enumerate(lines)
        if (match := re.search(
            r"slot\s+\S+:\s+id\s+(\d+)\s+\|\s+task\s+(\d+)\s+\|\s+processing task",
            line,
            flags=re.IGNORECASE,
        ))
        for slot, task in [match.groups()]
    ]
    grammar_prefill_positions = [
        index
        for index, line in enumerate(lines)
        if "grammar accepted prefill token" in line.casefold()
    ]
    error_lines = [
        line
        for line in log_text.splitlines()
        if re.search(r"grammar|sampler", line, flags=re.IGNORECASE)
        and re.search(r"error|failed|invalid|reject", line, flags=re.IGNORECASE)
    ]

    task_id = next(iter(task_ids)) if len(task_ids) == 1 else None
    slot_id = next(iter(slot_ids)) if len(slot_ids) == 1 else None
    matching_init = [
        (slot, task)
        for _, slot, task in init_rows
        if task_id is not None and slot == slot_id and task == task_id
    ]
    all_init = [(slot, task) for _, slot, task in init_rows]
    launch_processing_interval = None
    correlated_prefill_positions: list[int] = []
    uncorrelated_prefill_positions = grammar_prefill_positions.copy()
    if len(launch_rows) == 1 and len(processing_rows) == 1:
        launch_index, launch_slot, launch_task = launch_rows[0]
        processing_index, processing_slot, processing_task = processing_rows[0]
        launch_processing_interval = [launch_index, processing_index]
        if launch_index < processing_index:
            correlated_prefill_positions = [
                index
                for index in grammar_prefill_positions
                if launch_index < index < processing_index
            ]
            uncorrelated_prefill_positions = [
                index
                for index in grammar_prefill_positions
                if not launch_index < index < processing_index
            ]
    else:
        launch_slot = launch_task = processing_slot = processing_task = None
    init_after_processing = (
        len(init_rows) == 1
        and len(processing_rows) == 1
        and init_rows[0][0] > processing_rows[0][0]
    )
    passed = (
        input_witnessed
        and not observer_failures
        and len(task_ids) == 1
        and len(slot_ids) == 1
        and task_id is not None
        and slot_id is not None
        and len(launch_rows) == 1
        and launch_slot == slot_id
        and launch_task == -1
        and len(processing_rows) == 1
        and processing_slot == slot_id
        and processing_task == task_id
        and matching_init == [(slot_id, task_id)]
        and all_init == matching_init
        and init_after_processing
        and bool(correlated_prefill_positions)
        and not uncorrelated_prefill_positions
        and not error_lines
    )
    return {
        "status": "PASS" if passed else "UNVERIFIED",
        "source_id": source_id,
        "observed_task_id": task_id,
        "observed_slot_id": slot_id,
        "active_task_ids": sorted(task_ids, key=lambda value: str(value)),
        "active_slot_ids": sorted(slot_ids, key=lambda value: str(value)),
        "active_prompt_witnesses": len(active),
        "input_witnessed_in_active_prompt": input_witnessed,
        "launch_rows": [list(row) for row in launch_rows],
        "launch_processing_interval": launch_processing_interval,
        "matching_init_sampler_rows": matching_init,
        "processing_task_ids": [str(task) for _, _, task in processing_rows],
        "grammar_prefill_line_count": len(grammar_prefill_positions),
        "grammar_prefill_correlated_line_count": len(correlated_prefill_positions),
        "grammar_prefill_uncorrelated_line_indices": uncorrelated_prefill_positions,
        "observer_failures": observer_failures,
        "error_lines": error_lines,
    }


def _apply_sampler_evidence(
    result: dict[str, Any], evidence: dict[str, Any]
) -> dict[str, Any]:
    result["sampler_evidence_status"] = evidence["status"]
    if evidence["status"] != "PASS":
        result["server_log_attribution"] = "UNVERIFIED"
        result["sampler_parser_after"] = "NOT_VERIFIED"
        result["sampler_gate"] = "INFRA_FAILURE_UNCORRELATED_EVIDENCE"
        if result.get("status") != "QUALITY_FAILURE":
            result["status"] = "INFRA_FAILURE"
            result["failure"] = "sampler_init_gate_unverified"
        return result

    result["server_log_attribution"] = "REQUEST_TASK_CORRELATED"
    result["sampler_parser_after"] = "VERIFIED"
    if (
        result.get("http_status") != 200
        or result.get("http_envelope_validation") != "PASS"
    ):
        result["sampler_gate"] = "INFRA_FAILURE_REQUEST_COMPLETION"
        result["status"] = "INFRA_FAILURE"
        result.setdefault("failure", "request_completion_failed_after_sampler_init")
    elif result.get("structural_validation") != "PASS":
        result["sampler_gate"] = "QUALITY_FAILURE_RESPONSE_CONTRACT"
        result["status"] = "QUALITY_FAILURE"
        result["failure"] = "response_contract_validation_failed_after_sampler_init"
    else:
        result["sampler_gate"] = "PASS"
    return result


def run_gate() -> dict[str, Any]:
    if GATE_DIR.exists():
        raise RuntimeError("gate_attempt_directory_exists_refusing_retry")
    GATE_DIR.mkdir(parents=True)
    result: dict[str, Any] = {
        "protocol": "mem3b0q-r4-protocol-v1",
        "case_id": "R4-P01",
        "pass": 1,
        "request_attempts": 0,
        "retry_count": 0,
        "hosted_calls": 0,
        "memory_store_mutations": 0,
        "revision_edges": 0,
        "sampler_parser_before": "NOT_VERIFIED",
        "sampler_gate": "PENDING_REQUEST_CORRELATION_AND_POSITIVE_SIGNAL",
        "server_log_attribution": "TASK_CORRELATED_SLOT_TRACE_REQUIRED",
        "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    request_issued = False
    log_offset: int | None = None
    try:
        manifest, pack, schema, runner_lock = _load_frozen_inputs()
        result["protocol_manifest_sha256"] = FROZEN_MANIFEST_SHA256
        result["runner_sha256"] = runner_lock["runner_sha256"]
        result["runner_lock_sha256"] = _file_sha256(RUNNER_LOCK_PATH)
        proposition = pack["propositions"][0]
        if proposition["source_id"] != "R4-P01":
            raise RuntimeError("first_frozen_case_mismatch")
        pre_process = _ps_process_snapshot()
        pre_service = _service_snapshot()
        _validate_preflight(pre_process, pre_service)
        _write_json(
            GATE_DIR / "preflight_snapshots.json",
            {"process": pre_process, "service": pre_service},
        )
        if not LOG_PATH.exists():
            raise RuntimeError("server_log_missing")
        log_offset = LOG_PATH.stat().st_size

        request = mem3b0q_r4.build_request_payload(
            proposition, schema, model=MODEL_PATH
        )
        request_bytes = json.dumps(
            request, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        nested_schema = request["response_format"]["json_schema"]["schema"]
        if request["response_format"]["type"] != "json_schema":
            raise RuntimeError("strict_json_schema_mode_missing")
        if mem3b0q_r4.canonical_json_sha256(nested_schema) != SCHEMA_CANONICAL_SHA256:
            raise RuntimeError("outgoing_schema_hash_mismatch")

        (GATE_DIR / "request_body.json").write_bytes(request_bytes)
        request_hash = _sha256(request_bytes)
        _write_json(
            GATE_DIR / "request_issued.json",
            {"request_sha256": request_hash, "issued_at_utc": dt.datetime.now(dt.timezone.utc).isoformat()},
        )
        request_issued = True
        result["request_attempts"] = 1
        result["request_sha256"] = request_hash
        result["canonical_schema_sha256"] = SCHEMA_CANONICAL_SHA256

        slot_observations: list[dict[str, Any]] = []
        slot_observation_failures: list[str] = []
        stop_slot_observer = threading.Event()
        slot_observer = threading.Thread(
            target=_observe_active_slots,
            args=(
                stop_slot_observer,
                proposition["source_id"],
                proposition["proposition_text"],
                slot_observations,
                slot_observation_failures,
            ),
            daemon=True,
        )
        slot_observer.start()
        connection = http.client.HTTPConnection(HOST, PORT, timeout=240)
        request_started = time.perf_counter()
        try:
            connection.request(
                "POST",
                ENDPOINT_PATH,
                body=request_bytes,
                headers={"Content-Type": "application/json", "Accept": "application/json"},
            )
            response = connection.getresponse()
            response_bytes = response.read()
            result["http_status"] = response.status
            result["response_sha256"] = _sha256(response_bytes)
            (GATE_DIR / "response_body.json").write_bytes(response_bytes)
        finally:
            result["request_latency_ms"] = round(
                (time.perf_counter() - request_started) * 1000, 3
            )
            connection.close()
            stop_slot_observer.set()
            slot_observer.join(timeout=5)
            if slot_observer.is_alive():
                slot_observation_failures.append("observer_join_timeout")
            _write_json(
                GATE_DIR / "slot_observations.json",
                {
                    "observations": slot_observations,
                    "failures": slot_observation_failures,
                },
            )

        _capture_log_delta(log_offset, result)
        post_service = _service_snapshot()
        post_process = _ps_process_snapshot()
        _validate_preflight(post_process, post_service)
        _write_json(
            GATE_DIR / "postflight_snapshots.json",
            {"process": post_process, "service": post_service},
        )
        if pre_process != post_process:
            raise RuntimeError("server_process_changed_during_gate")
        result["process_snapshot_sha256"] = _sha256(
            json.dumps(pre_process, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        result["pre_service_raw_sha256"] = pre_service["raw_sha256"]
        result["post_service_raw_sha256"] = post_service["raw_sha256"]
        log_text = (GATE_DIR / "server_log_delta.txt").read_text(
            encoding="utf-8", errors="replace"
        )
        if response.status != 200:
            result["status"] = "INFRA_FAILURE"
            result["failure"] = "strict_schema_request_http_failure"
        else:
            try:
                response_json = json.loads(response_bytes.decode("utf-8"))
                if not isinstance(response_json, dict):
                    raise TypeError("response_envelope_not_object")
                content = response_json["choices"][0]["message"]["content"]
                if not isinstance(content, str):
                    raise TypeError("response_content_not_string")
                result["http_envelope_validation"] = "PASS"
            except (UnicodeDecodeError, json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
                result["http_envelope_validation"] = "FAIL"
                result["status"] = "INFRA_FAILURE"
                result["failure"] = f"invalid_http_response_envelope:{type(exc).__name__}:{exc}"
            else:
                result["content_sha256"] = _sha256(content.encode("utf-8"))
                try:
                    validated = mem3b0q_r4.validate_content(
                        content, proposition, scope_id=pack["scope_id"]
                    )
                    result["structural_validation"] = "PASS"
                    result["oracle_failures"] = mem3b0q_r4.compare_to_oracle(
                        validated, proposition["expected"]
                    )
                    result["quality_status"] = (
                        "PASS" if not result["oracle_failures"] else "QUALITY_FAILURE"
                    )
                    (GATE_DIR / "normalized_proposal.json").write_text(
                        json.dumps(validated, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                        encoding="utf-8",
                    )
                except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
                    result["quality_status"] = "QUALITY_FAILURE"
                    result["quality_failure"] = f"{type(exc).__name__}:{exc}"
                    result["structural_validation"] = "FAIL"
                result["status"] = "PENDING_SERVER_LOG_REVIEW"

        sampler_evidence = _evaluate_sampler_evidence(
            log_text,
            slot_observations,
            source_id=proposition["source_id"],
            proposition_text=proposition["proposition_text"],
            observer_failures=slot_observation_failures,
        )
        sampler_evidence["request_sha256"] = request_hash
        _write_json(GATE_DIR / "sampler_evidence.json", sampler_evidence)
        _apply_sampler_evidence(result, sampler_evidence)
    except Exception as exc:  # The attempt is terminal; never retry a potentially delivered request.
        result["status"] = "INFRA_FAILURE" if request_issued else "PREFLIGHT_FAILURE"
        result["failure"] = f"{type(exc).__name__}:{exc}"
        result["delivery_state"] = "UNKNOWN" if request_issued else "NOT_SENT"
        if request_issued and log_offset is not None and "server_log_delta_sha256" not in result:
            try:
                _capture_log_delta(log_offset, result)
            except OSError as log_exc:
                result["server_log_capture_failure"] = f"{type(log_exc).__name__}:{log_exc}"
    result["request_attempts"] = max(result["request_attempts"], int(request_issued))
    result["retry_count"] = 0
    result["finished_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    _write_json(GATE_DIR / "gate_attempt.json", result)
    gate_artifacts = {
        path.name: _file_sha256(path)
        for path in sorted(GATE_DIR.iterdir())
        if path.is_file() and path.name != "gate_run_manifest.json"
    }
    gate_run_manifest = {
        "manifest_id": "mem3b0q-r4-gate-run-manifest-v1",
        "status": "PENDING_REVIEW",
        "protocol_manifest_sha256": result.get("protocol_manifest_sha256"),
        "runner_lock_sha256": result.get("runner_lock_sha256"),
        "runner_sha256": result.get("runner_sha256"),
        "branch": "mem3b0q-span-grounded-identity-20261001",
        "repository_head": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "case_id": "R4-P01",
        "request_attempts": result["request_attempts"],
        "retry_count": result["retry_count"],
        "sampler_parser": result.get("sampler_parser_after", "NOT_VERIFIED"),
        "b1_status": "MEM3B0Q_MEM3B1_READY=NO",
        "gate_attempt_status": result.get("status"),
        "quality_status": result.get("quality_status"),
        "artifact_sha256": gate_artifacts,
    }
    _write_json(GATE_DIR / "gate_run_manifest.json", gate_run_manifest)
    return result


if __name__ == "__main__":
    print(json.dumps(run_gate(), ensure_ascii=False, sort_keys=True, indent=2))
