"""Run a fresh three-case event-value confirmation pack on local Qwen."""

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

from jsonschema import Draft202012Validator

from tools.research.memory import mem3b0q_r4_candidate_request_v1 as request_v1
from tools.research.memory import mem3b0q_r4_candidate_request_v3 as request_v3
from tools.research.memory import run_mem3b0q_r4_atomwise_candidate_devset_v1 as v1_runner
from tools.research.memory import run_mem3b0q_r4_cardinality_devset_v2 as v2_runner
from tools.research.memory import run_mem3b0q_r4_candidate_local_smoke_v1 as smoke
from tools.research.memory import run_mem3b0q_r4_event_value_smoke_v3 as v3_runner
from tools.research.memory import run_mem3b0q_r4_gate as frozen_gate
from tools.research.memory.preflight_mem3b0q_r4_runtime_readonly_v1 import (
    validate_runtime_snapshot,
)


ROOT = Path(__file__).resolve().parents[3]
DOCS = ROOT / "docs" / "research" / "memory"
PACK_PATH = DOCS / "mem3b0q_r4_event_value_confirmation_pack_v1.json"
PROTOCOL_PATH = DOCS / "mem3b0q_r4_event_value_confirmation_v1_protocol.md"
LOCK_PATH = DOCS / "mem3b0q_r4_event_value_confirmation_v1_lock.json"
LOCK_SHA_PATH = LOCK_PATH.with_suffix(LOCK_PATH.suffix + ".sha256")
FREEZE_PATH = ROOT / "tools" / "research" / "memory" / "freeze_mem3b0q_r4_event_value_confirmation_v1.py"
OUTPUT_ROOT = ROOT / "runs" / "memory" / "mem3" / "mem3b0q-r4-event-value-confirmation-v1"
CASE_IDS = ("EVTCONF-01", "EVTCONF-02", "EVTCONF-03")
DEPENDENCY_PATHS = (
    "docs/research/memory/mem3b0q_r4_event_value_confirmation_pack_v1.json",
    "docs/research/memory/mem3b0q_r4_event_value_confirmation_v1_protocol.md",
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
    "tools/research/memory/run_mem3b0q_r4_event_value_confirmation_v1.py",
    "tools/research/memory/freeze_mem3b0q_r4_event_value_confirmation_v1.py",
)
LOCK_STATIC_FIELDS = {
    **v3_runner.LOCK_STATIC_FIELDS,
    "lock_id": "mem3b0q-r4-event-value-confirmation-v1-lock",
    "run_id": OUTPUT_ROOT.name,
    "case_ids": list(CASE_IDS),
    "max_completion_posts": len(CASE_IDS),
    "adapter_version": "mem3b0q-r4-harness-cardinality+event-value-prompt-v3-confirmation",
}
LOCK_DYNAMIC_FIELDS = frozenset(
    {"runner_sha256", "freeze_script_sha256", "dependencies", "requests", "dataset_sha256"}
)


class ConfirmationRunError(RuntimeError):
    """A frozen dataset, request, runtime, or output invariant failed closed."""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _strict_json(raw: bytes | str) -> Any:
    try:
        return v1_runner._strict_json(raw)
    except Exception as exc:
        raise ConfirmationRunError(str(exc)) from exc


def _write_exclusive(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json_exclusive(path: Path, value: Any) -> None:
    _write_exclusive(path, _canonical_json(value) + b"\n")


def _build_inputs() -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    pack_bytes = PACK_PATH.read_bytes()
    pack = _strict_json(pack_bytes)
    if (
        pack.get("pack_id") != "mem3b0q-r4-event-value-confirmation-pack-v1"
        or pack.get("status") != "FROZEN_BEFORE_REQUESTS_NO_INFERENCE_AUTHORIZATION"
        or pack.get("clinical_content") is not False
        or [row.get("case_id") for row in pack.get("cases", [])] != list(CASE_IDS)
    ):
        raise ConfirmationRunError("confirmation_pack_identity_mismatch")
    prior_pack, _ = v1_runner._build_requests()
    prior_texts = {row["proposition_text"] for row in prior_pack["cases"]}
    if any(row["proposition_text"] in prior_texts for row in pack["cases"]):
        raise ConfirmationRunError("confirmation_text_overlap_with_tuning_pack")

    requests: dict[str, dict[str, Any]] = {}
    scoring_requests: dict[str, dict[str, Any]] = {}
    for case in pack["cases"]:
        request = request_v3.build_candidate_request(
            case, model=frozen_gate.MODEL_PATH
        )
        schema = request["response_format"]["json_schema"]["schema"]
        if "cardinality_proposal" in schema["properties"]["atoms"]["items"]["properties"]:
            raise ConfirmationRunError("model_cardinality_field_not_removed")
        Draft202012Validator.check_schema(schema)
        payload = _strict_json(request["messages"][1]["content"])
        if set(payload) != {"source_id", "proposition_text", "typed_candidates"}:
            raise ConfirmationRunError("gold_leak_or_prompt_shape_mismatch")
        if payload["source_id"] != case["case_id"]:
            raise ConfirmationRunError("request_case_id_mismatch")
        body = json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        requests[case["case_id"]] = {
            "request": request,
            "body": body,
            "request_sha256": _sha(body),
            "schema_sha256": _sha(_canonical_json(schema)),
        }
        scoring_requests[case["case_id"]] = request_v1.build_candidate_request(
            case, model=frozen_gate.MODEL_PATH
        )
    return pack, requests, scoring_requests


def build_lock_payload() -> dict[str, Any]:
    pack, requests, _ = _build_inputs()
    return {
        **LOCK_STATIC_FIELDS,
        "runner_sha256": _sha(Path(__file__).resolve().read_bytes()),
        "freeze_script_sha256": _sha(FREEZE_PATH.read_bytes()),
        "dependencies": {
            relative: _sha((ROOT / relative).read_bytes())
            for relative in DEPENDENCY_PATHS
        },
        "dataset_sha256": _sha(PACK_PATH.read_bytes()),
        "requests": {
            case_id: {"request_sha256": row["request_sha256"], "schema_sha256": row["schema_sha256"]}
            for case_id, row in requests.items()
        },
    }


def _load_lock() -> tuple[dict[str, Any], str, dict[str, Any], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    data = LOCK_PATH.read_bytes()
    digest = _sha(data)
    sidecar = LOCK_SHA_PATH.read_text(encoding="ascii").split()
    if not sidecar or sidecar[0].lower() != digest:
        raise ConfirmationRunError("lock_sidecar_mismatch")
    lock = _strict_json(data)
    pack, requests, scoring_requests = _build_inputs()
    if set(lock) != set(LOCK_STATIC_FIELDS) | LOCK_DYNAMIC_FIELDS:
        raise ConfirmationRunError("lock_shape_mismatch")
    for key, value in LOCK_STATIC_FIELDS.items():
        if _canonical_json(lock.get(key)) != _canonical_json(value):
            raise ConfirmationRunError(f"lock_field_mismatch:{key}")
    if sys.version.split()[0] != LOCK_STATIC_FIELDS["python_version"]:
        raise ConfirmationRunError("python_version_mismatch")
    if package_version("jsonschema") != LOCK_STATIC_FIELDS["jsonschema_version"]:
        raise ConfirmationRunError("jsonschema_version_mismatch")
    expected = build_lock_payload()
    for key in LOCK_DYNAMIC_FIELDS:
        if lock.get(key) != expected[key]:
            raise ConfirmationRunError(f"lock_field_mismatch:{key}")
    if OUTPUT_ROOT.exists():
        raise ConfirmationRunError("output_directory_already_exists")
    frozen_gate._load_frozen_inputs()
    return lock, digest, pack, requests, scoring_requests


def _run_case(
    case: dict[str, Any],
    request: dict[str, Any],
    scoring_request: dict[str, Any],
) -> dict[str, Any]:
    case_id = case["case_id"]
    case_dir = OUTPUT_ROOT / case_id
    case_dir.mkdir(parents=True, exist_ok=False)
    _write_json_exclusive(
        case_dir / "reservation.json",
        {"status": "RESERVED_NO_RETRY", "run_id": OUTPUT_ROOT.name, "case_id": case_id, "request_sha256": request["request_sha256"]},
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
            connection = http.client.HTTPConnection(frozen_gate.HOST, frozen_gate.PORT, timeout=300)
            try:
                started = time.perf_counter()
                connection.request(
                    "POST",
                    frozen_gate.ENDPOINT_PATH,
                    body=request["body"],
                    headers={"Content-Type": "application/json", "Accept": "application/json"},
                )
                response = connection.getresponse()
                http_status = response.status
                response_bytes = response.read()
                latency_ms = round((time.perf_counter() - started) * 1000, 3)
            finally:
                connection.close()
            guard.verify_postflight()
        if response_bytes is not None:
            _write_exclusive(case_dir / "response_body.bin", response_bytes)
        if response_bytes is None or http_status != 200:
            row = {"case_id": case_id, "status": "INFRA_FAILURE", "failure": "completion_http_failure"}
        else:
            envelope, content = smoke._response_content(response_bytes)
            finish_reason = envelope.get("choices", [{}])[0].get("finish_reason")
            row = v2_runner._score_response(
                case,
                content,
                v2_request=request["request"],
                v1_request=scoring_request["request"],
                response_sha256=_sha(response_bytes),
                usage=envelope.get("usage"),
                latency_ms=latency_ms,
                finish_reason=finish_reason,
            )
    except Exception as exc:
        row = {
            "case_id": case_id,
            "status": "INFRA_FAILURE" if guard.post_attempts else "PREFLIGHT_FAILURE",
            "failure": f"{type(exc).__name__}:{exc}",
            "delivery_state": "UNKNOWN" if guard.post_attempts else "NOT_SENT",
        }
        if response_bytes is not None and not (case_dir / "response_body.bin").exists():
            _write_exclusive(case_dir / "response_body.bin", response_bytes)
    row.update(
        {
            "run_id": OUTPUT_ROOT.name,
            "request_sha256": request["request_sha256"],
            "http_status": http_status,
            "latency_ms": latency_ms,
            "completion_post_attempts": guard.post_attempts,
            "retry_count": 0,
            "hosted_calls": 0,
            "memory_store_mutations": 0,
            "runtime_checkpoints": guard.events,
            "evidence_type": "NEW_LOCAL_ONE_SHOT_POST",
        }
    )
    if row.get("admitted_normalized_atoms"):
        _write_json_exclusive(
            case_dir / "admitted_atoms.json",
            {"source_id": case_id, "scope_id": v1_runner.frozen_r4.FROZEN_SCOPE_ID, "atoms": row["admitted_normalized_atoms"]},
        )
    _write_json_exclusive(case_dir / "case_result.json", row)
    return row


def run_once() -> dict[str, Any]:
    lock, lock_sha, pack, requests, scoring_requests = _load_lock()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    _write_json_exclusive(
        OUTPUT_ROOT / "run_reservation.json",
        {"status": "RESERVED_NO_RETRY", "run_id": OUTPUT_ROOT.name, "lock_sha256": lock_sha, "case_ids": list(CASE_IDS)},
    )
    cases = {row["case_id"]: row for row in pack["cases"]}
    rows = []
    stopped = False
    for case_id in CASE_IDS:
        if stopped:
            rows.append({"case_id": case_id, "status": "NOT_RUN_AFTER_INFRA_FAILURE", "completion_post_attempts": 0, "retry_count": 0, "hosted_calls": 0, "memory_store_mutations": 0})
            continue
        row = _run_case(cases[case_id], requests[case_id], scoring_requests[case_id])
        rows.append(row)
        stopped = row.get("status") in {"INFRA_FAILURE", "PREFLIGHT_FAILURE"}
    aggregate = v2_runner._aggregate(rows)
    result = {
        "schema_version": 1,
        "run_id": OUTPUT_ROOT.name,
        "lock_sha256": lock_sha,
        "status": "COMPLETE" if aggregate["completed_quality_cases"] == len(CASE_IDS) else "INFRA_INCOMPLETE",
        "evaluation_scope": "FRESH_PROJECT_AUTHORED_SYNTHETIC_EVENT_CONFIRMATION_ONLY",
        "reader_model_path": frozen_gate.MODEL_PATH,
        "reader_model_sha256": frozen_gate.MODEL_SHA256,
        "prompt_version": request_v3.REQUEST_BUILDER_VERSION,
        "hosted_api": "NONE",
        "required_api_key": "NONE",
        "aggregate": aggregate,
        "cases": rows,
        "limitations": [
            "The three source texts and gold labels were frozen before inference and do not overlap with the R4C tuning texts.",
            "The set is small, synthetic, and project-authored; it is not a public benchmark or unbiased generalization estimate.",
            "The event-value prompt was selected after observing R4C-04; this is a fresh confirmation set, not a blinded study.",
            "Relative/absolute temporal expressions are diagnostic distractors only and are not normalized or scored.",
            "No MemoryStore mutation, revision materialization, temporal query, or clinical workflow was evaluated.",
        ],
    }
    _write_json_exclusive(OUTPUT_ROOT / "confirmation_result.json", result)
    artifacts = {
        str(path.relative_to(OUTPUT_ROOT)).replace("\\", "/"): _sha(path.read_bytes())
        for path in sorted(OUTPUT_ROOT.rglob("*"))
        if path.is_file() and path.name != "run_manifest.json"
    }
    _write_json_exclusive(
        OUTPUT_ROOT / "run_manifest.json",
        {"manifest_id": "mem3b0q-r4-event-value-confirmation-v1", "status": result["status"], "lock_sha256": lock_sha, "runner_sha256": lock["runner_sha256"], "dataset_sha256": lock["dataset_sha256"], "case_ids": list(CASE_IDS), "aggregate": aggregate, "artifact_sha256": artifacts},
    )
    print(json.dumps({"status": result["status"], "aggregate": aggregate, "cases": [{"case_id": row["case_id"], "status": row["status"], "matched": row.get("matched_expected_atoms"), "expected": row.get("expected_atoms"), "failure": row.get("failure")} for row in rows]}, ensure_ascii=False, sort_keys=True, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--run-confirmation-once", action="store_true")
    args = parser.parse_args()
    if not args.run_confirmation_once:
        parser.error("explicit --run-confirmation-once is required")
    result = run_once()
    return 0 if result.get("status") == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
