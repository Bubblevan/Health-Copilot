"""One-shot R4C-04 development smoke for event-value wording v3."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import sys
import time
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, ValidationError

from tools.research.memory import mem3b0q_r4 as frozen_r4
from tools.research.memory import mem3b0q_r4_candidate_request_v3 as request_v3
from tools.research.memory import run_mem3b0q_r4_atomwise_candidate_devset_v1 as v1_runner
from tools.research.memory import run_mem3b0q_r4_cardinality_devset_v2 as v2_runner
from tools.research.memory import run_mem3b0q_r4_candidate_local_smoke_v1 as smoke
from tools.research.memory import run_mem3b0q_r4_gate as frozen_gate
from tools.research.memory.preflight_mem3b0q_r4_runtime_readonly_v1 import (
    validate_runtime_snapshot,
)


ROOT = Path(__file__).resolve().parents[3]
DOCS = ROOT / "docs" / "research" / "memory"
CASE_ID = "R4C-04"
PROTOCOL_PATH = DOCS / "mem3b0q_r4_event_value_smoke_v3_protocol.md"
LOCK_PATH = DOCS / "mem3b0q_r4_event_value_smoke_v3_lock.json"
LOCK_SHA_PATH = LOCK_PATH.with_suffix(LOCK_PATH.suffix + ".sha256")
FREEZE_PATH = ROOT / "tools" / "research" / "memory" / "freeze_mem3b0q_r4_event_value_smoke_v3.py"
OUTPUT_ROOT = ROOT / "runs" / "memory" / "mem3" / "mem3b0q-r4-event-value-smoke-v3"
DEPENDENCY_PATHS = (
    "docs/research/memory/mem3b0q_r4_event_value_smoke_v3_protocol.md",
    "docs/research/memory/mem3b0q_r4_candidate_control_pack_v1.json",
    "tools/research/memory/mem3b0q_r4.py",
    "tools/research/memory/mem3b0q_r4_atomwise_admission_v1.py",
    "tools/research/memory/mem3b0q_r4_candidate_binding_guard_v1.py",
    "tools/research/memory/mem3b0q_r4_candidate_builder_v1.py",
    "tools/research/memory/mem3b0q_r4_candidate_request_v1.py",
    "tools/research/memory/mem3b0q_r4_candidate_request_v2.py",
    "tools/research/memory/mem3b0q_r4_candidate_request_v3.py",
    "tools/research/memory/mem3b0q_r4_joint_binding_guard_v2.py",
    "tools/research/memory/mem3b0q_r4_joint_binding_guard_v3.py",
    "tools/research/memory/preflight_mem3b0q_r4_runtime_readonly_v1.py",
    "tools/research/memory/run_mem3b0q_r4_candidate_local_smoke_v1.py",
    "tools/research/memory/run_mem3b0q_r4_gate.py",
    "tools/research/memory/run_mem3b0q_r4_atomwise_candidate_devset_v1.py",
    "tools/research/memory/run_mem3b0q_r4_cardinality_devset_v2.py",
    "tools/research/memory/run_mem3b0q_r4_event_value_smoke_v3.py",
    "tools/research/memory/freeze_mem3b0q_r4_event_value_smoke_v3.py",
)
LOCK_STATIC_FIELDS = {
    **v2_runner.LOCK_STATIC_FIELDS,
    "lock_id": "mem3b0q-r4-event-value-smoke-v3-lock",
    "run_id": OUTPUT_ROOT.name,
    "case_ids": [CASE_ID],
    "max_completion_posts": 1,
    "adapter_version": "mem3b0q-r4-harness-cardinality+event-value-prompt-v3-development",
}
LOCK_DYNAMIC_FIELDS = frozenset(
    {"runner_sha256", "freeze_script_sha256", "dependencies", "requests"}
)


class EventSmokeError(RuntimeError):
    """Frozen one-shot request or runtime validation failed."""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _strict_json(raw: bytes | str) -> Any:
    try:
        return v1_runner._strict_json(raw)
    except Exception as exc:
        raise EventSmokeError(str(exc)) from exc


def _write_exclusive(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json_exclusive(path: Path, payload: Any) -> None:
    _write_exclusive(path, _canonical_json(payload) + b"\n")


def _build_request() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    pack, v1_requests = v1_runner._build_requests()
    case = v1_runner._case_by_id(pack, CASE_ID)
    request = request_v3.build_candidate_request(
        case, model=frozen_gate.MODEL_PATH
    )
    schema = request["response_format"]["json_schema"]["schema"]
    atom_schema = schema["properties"]["atoms"]["items"]
    if "cardinality_proposal" in atom_schema["properties"]:
        raise EventSmokeError("model_cardinality_field_not_removed")
    Draft202012Validator.check_schema(schema)
    body = json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    record = {
        "request": request,
        "body": body,
        "request_sha256": _sha(body),
        "schema_sha256": _sha(_canonical_json(schema)),
    }
    return case, record, v1_requests[CASE_ID]


def build_lock_payload() -> dict[str, Any]:
    _, record, _ = _build_request()
    return {
        **LOCK_STATIC_FIELDS,
        "runner_sha256": _sha(Path(__file__).resolve().read_bytes()),
        "freeze_script_sha256": _sha(FREEZE_PATH.read_bytes()),
        "dependencies": {
            relative: _sha((ROOT / relative).read_bytes())
            for relative in DEPENDENCY_PATHS
        },
        "requests": {
            CASE_ID: {
                "request_sha256": record["request_sha256"],
                "schema_sha256": record["schema_sha256"],
            }
        },
    }


def _load_lock() -> tuple[dict[str, Any], str, dict[str, Any], dict[str, Any], dict[str, Any]]:
    data = LOCK_PATH.read_bytes()
    digest = _sha(data)
    sidecar = LOCK_SHA_PATH.read_text(encoding="ascii").split()
    if not sidecar or sidecar[0].lower() != digest:
        raise EventSmokeError("lock_sidecar_mismatch")
    lock = _strict_json(data)
    case, request, v1_request = _build_request()
    if set(lock) != set(LOCK_STATIC_FIELDS) | LOCK_DYNAMIC_FIELDS:
        raise EventSmokeError("lock_shape_mismatch")
    for key, value in LOCK_STATIC_FIELDS.items():
        if _canonical_json(lock.get(key)) != _canonical_json(value):
            raise EventSmokeError(f"lock_field_mismatch:{key}")
    if sys.version.split()[0] != LOCK_STATIC_FIELDS["python_version"]:
        raise EventSmokeError("python_version_mismatch")
    if package_version("jsonschema") != LOCK_STATIC_FIELDS["jsonschema_version"]:
        raise EventSmokeError("jsonschema_version_mismatch")
    expected = build_lock_payload()
    for key in LOCK_DYNAMIC_FIELDS:
        if lock.get(key) != expected[key]:
            raise EventSmokeError(f"lock_field_mismatch:{key}")
    if OUTPUT_ROOT.exists():
        raise EventSmokeError("output_directory_already_exists")
    frozen_gate._load_frozen_inputs()
    return lock, digest, case, request, v1_request


def run_once() -> dict[str, Any]:
    lock, lock_sha, case, request, v1_request = _load_lock()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    case_dir = OUTPUT_ROOT / CASE_ID
    case_dir.mkdir()
    _write_json_exclusive(
        case_dir / "reservation.json",
        {"status": "RESERVED_NO_RETRY", "run_id": OUTPUT_ROOT.name, "case_id": CASE_ID, "request_sha256": request["request_sha256"]},
    )
    _write_exclusive(case_dir / "request.json", request["body"])
    guard = smoke.LoopbackOneShotGuard(request["body"])
    response_bytes = None
    latency_ms = None
    http_status = None
    try:
        with guard:
            process = frozen_gate._ps_process_snapshot()
            service = frozen_gate._service_snapshot()
            validate_runtime_snapshot(process, service)
            guard.arm_after_preflight(process, service)
            connection = http.client.HTTPConnection(
                frozen_gate.HOST, frozen_gate.PORT, timeout=300
            )
            try:
                start = time.perf_counter()
                connection.request(
                    "POST",
                    frozen_gate.ENDPOINT_PATH,
                    body=request["body"],
                    headers={"Content-Type": "application/json", "Accept": "application/json"},
                )
                response = connection.getresponse()
                http_status = response.status
                response_bytes = response.read()
                latency_ms = round((time.perf_counter() - start) * 1000, 3)
            finally:
                connection.close()
            guard.verify_postflight()
        _write_exclusive(case_dir / "response_body.bin", response_bytes or b"")
        if http_status != 200 or response_bytes is None:
            row = {"case_id": CASE_ID, "status": "INFRA_FAILURE", "failure": "completion_http_failure"}
        else:
            envelope, content = smoke._response_content(response_bytes)
            finish_reason = envelope.get("choices", [{}])[0].get("finish_reason")
            row = v2_runner._score_response(
                case,
                content,
                v2_request=request["request"],
                v1_request=v1_request["request"],
                response_sha256=_sha(response_bytes),
                usage=envelope.get("usage"),
                latency_ms=latency_ms,
                finish_reason=finish_reason,
            )
        row.update(
            {
                "run_id": OUTPUT_ROOT.name,
                "request_sha256": request["request_sha256"],
                "http_status": http_status,
                "completion_post_attempts": guard.post_attempts,
                "retry_count": 0,
                "hosted_calls": 0,
                "memory_store_mutations": 0,
                "runtime_checkpoints": guard.events,
                "evidence_type": "NEW_LOCAL_ONE_SHOT_POST",
            }
        )
    except Exception as exc:
        row = {
            "case_id": CASE_ID,
            "run_id": OUTPUT_ROOT.name,
            "status": "INFRA_FAILURE" if guard.post_attempts else "PREFLIGHT_FAILURE",
            "failure": f"{type(exc).__name__}:{exc}",
            "completion_post_attempts": guard.post_attempts,
            "retry_count": 0,
            "hosted_calls": 0,
            "memory_store_mutations": 0,
            "runtime_checkpoints": guard.events,
        }
        if response_bytes is not None and not (case_dir / "response_body.bin").exists():
            _write_exclusive(case_dir / "response_body.bin", response_bytes)
    _write_json_exclusive(case_dir / "case_result.json", row)
    result = {
        "schema_version": 1,
        "run_id": OUTPUT_ROOT.name,
        "lock_sha256": lock_sha,
        "status": "COMPLETE" if row.get("status") in {"QUALITY_PASS", "QUALITY_FAILURE"} else "INFRA_INCOMPLETE",
        "evaluation_scope": "SINGLE_CASE_PROMPT_TUNING_SMOKE_ONLY",
        "reader_model_path": frozen_gate.MODEL_PATH,
        "reader_model_sha256": frozen_gate.MODEL_SHA256,
        "hosted_api": "NONE",
        "required_api_key": "NONE",
        "aggregate": {
            "case_count": 1,
            "exact_cases": int(row.get("exact_expected_atom_multiset_match") is True),
            "matched_atoms": row.get("matched_expected_atoms", 0),
            "expected_atoms": row.get("expected_atoms", len(case["expected_atoms"])),
            "atom_precision": row.get("atom_precision"),
            "atom_recall": row.get("atom_recall"),
            "infra_failures": int(row.get("status") in {"INFRA_FAILURE", "PREFLIGHT_FAILURE"}),
        },
        "case": row,
        "limitations": [
            "R4C-04 was already inspected in v1 and v2; this is prompt tuning, not independent evidence.",
            "No new benchmark or generalization claim; the event guard was not changed.",
            "The temporal phrase is not independently normalized into a date.",
        ],
    }
    _write_json_exclusive(OUTPUT_ROOT / "smoke_result.json", result)
    artifacts = {
        str(path.relative_to(OUTPUT_ROOT)).replace("\\", "/"): _sha(path.read_bytes())
        for path in sorted(OUTPUT_ROOT.rglob("*"))
        if path.is_file() and path.name != "run_manifest.json"
    }
    _write_json_exclusive(
        OUTPUT_ROOT / "run_manifest.json",
        {"manifest_id": "mem3b0q-r4-event-value-smoke-v3", "status": result["status"], "lock_sha256": lock_sha, "runner_sha256": lock["runner_sha256"], "case_ids": [CASE_ID], "artifact_sha256": artifacts},
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--run-r4c04-once", action="store_true")
    args = parser.parse_args()
    if not args.run_r4c04_once:
        parser.error("explicit --run-r4c04-once is required")
    result = run_once()
    return 0 if result.get("status") == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())

