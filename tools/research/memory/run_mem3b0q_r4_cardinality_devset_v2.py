"""Run the R4 v2 cardinality-authority diagnostic on local Qwen only."""

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
from tools.research.memory import mem3b0q_r4_candidate_request_v1 as request_v1
from tools.research.memory import mem3b0q_r4_candidate_request_v2 as request_v2
from tools.research.memory import run_mem3b0q_r4_atomwise_candidate_devset_v1 as v1_runner
from tools.research.memory import run_mem3b0q_r4_candidate_local_smoke_v1 as smoke
from tools.research.memory import run_mem3b0q_r4_gate as frozen_gate
from tools.research.memory.preflight_mem3b0q_r4_runtime_readonly_v1 import (
    validate_runtime_snapshot,
)


ROOT = Path(__file__).resolve().parents[3]
DOCS = ROOT / "docs" / "research" / "memory"
PACK_PATH = DOCS / "mem3b0q_r4_candidate_control_pack_v1.json"
PROTOCOL_PATH = DOCS / "mem3b0q_r4_cardinality_authority_devset_v2_protocol.md"
LOCK_PATH = DOCS / "mem3b0q_r4_cardinality_authority_devset_v2_lock.json"
LOCK_SHA_PATH = LOCK_PATH.with_suffix(LOCK_PATH.suffix + ".sha256")
FREEZE_PATH = ROOT / "tools" / "research" / "memory" / "freeze_mem3b0q_r4_cardinality_devset_v2.py"
OUTPUT_ROOT = ROOT / "runs" / "memory" / "mem3" / "mem3b0q-r4-cardinality-devset-v2"
CASE_IDS = tuple(f"R4C-{index:02d}" for index in range(1, 7))
DEPENDENCY_PATHS = (
    "docs/research/memory/mem3b0q_r4_cardinality_authority_devset_v2_protocol.md",
    "docs/research/memory/mem3b0q_r4_candidate_control_pack_v1.json",
    "tools/research/memory/mem3b0q_r4.py",
    "tools/research/memory/mem3b0q_r4_atomwise_admission_v1.py",
    "tools/research/memory/mem3b0q_r4_candidate_binding_guard_v1.py",
    "tools/research/memory/mem3b0q_r4_candidate_builder_v1.py",
    "tools/research/memory/mem3b0q_r4_candidate_request_v1.py",
    "tools/research/memory/mem3b0q_r4_candidate_request_v2.py",
    "tools/research/memory/mem3b0q_r4_joint_binding_guard_v2.py",
    "tools/research/memory/mem3b0q_r4_joint_binding_guard_v3.py",
    "tools/research/memory/preflight_mem3b0q_r4_runtime_readonly_v1.py",
    "tools/research/memory/run_mem3b0q_r4_candidate_local_smoke_v1.py",
    "tools/research/memory/run_mem3b0q_r4_gate.py",
    "tools/research/memory/run_mem3b0q_r4_atomwise_candidate_devset_v1.py",
    "tools/research/memory/run_mem3b0q_r4_cardinality_devset_v2.py",
    "tools/research/memory/freeze_mem3b0q_r4_cardinality_devset_v2.py",
)
REQUEST_CONTROL_FIELDS = {
    "temperature": 0,
    "seed": 42,
    "max_tokens": 256,
    "stream": False,
    "thinking": False,
    "response_format": "strict_json_schema_without_cardinality",
}
LOCK_STATIC_FIELDS: dict[str, Any] = {
    "schema_version": 1,
    "status": "FROZEN",
    "lock_id": "mem3b0q-r4-cardinality-authority-devset-v2-lock",
    "run_id": OUTPUT_ROOT.name,
    "case_ids": list(CASE_IDS),
    "max_completion_posts": len(CASE_IDS),
    "max_retries": 0,
    "hosted_api": "NONE",
    "required_api_key": "NONE",
    "memory_store_mutations": 0,
    "network_policy": {
        "transport": "direct_http_client_no_proxy_resolution",
        "allowed_host": frozen_gate.HOST,
        "allowed_port": frozen_gate.PORT,
        "allowed_get_paths": sorted(smoke.ALLOWED_GET_PATHS),
        "allowed_post_path": frozen_gate.ENDPOINT_PATH,
        "https_outbound": "DENY",
        "other_hosts": "DENY",
        "ambient_proxy_used": False,
    },
    "reader_model_path": frozen_gate.MODEL_PATH,
    "reader_model_sha256": frozen_gate.MODEL_SHA256,
    "server_path": frozen_gate.SERVER_PATH,
    "server_sha256": frozen_gate.SERVER_SHA256,
    "llama_build": "b10068-571d0d540",
    "runtime": {
        "host": frozen_gate.HOST,
        "port": frozen_gate.PORT,
        "context_tokens": 131072,
        "gpu_layers": 99,
        "flash_attention": True,
        "kv_cache": "q4_0",
        "slots": 1,
    },
    "request_controls": REQUEST_CONTROL_FIELDS,
    "adapter_version": "mem3b0q-r4-harness-cardinality-v2-development",
    "quality_metric": "exact_normalized_atom_multiset_precision_recall",
    "python_version": "3.12.4",
    "jsonschema_version": "4.19.2",
    "output_policy": "append_never_overwrite_never_retry",
}
LOCK_DYNAMIC_FIELDS = frozenset(
    {"runner_sha256", "freeze_script_sha256", "dependencies", "requests"}
)


class CardinalityDevsetError(RuntimeError):
    """A frozen request, runtime, or run-artifact invariant failed closed."""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _strict_json(raw: bytes | str) -> Any:
    try:
        return v1_runner._strict_json(raw)
    except Exception as exc:
        raise CardinalityDevsetError(str(exc)) from exc


def _write_exclusive(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json_exclusive(path: Path, value: Any) -> None:
    _write_exclusive(path, _canonical_json(value) + b"\n")


def _build_requests() -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    pack, v1_requests = v1_runner._build_requests()
    if [row.get("case_id") for row in pack["cases"]] != list(CASE_IDS):
        raise CardinalityDevsetError("control_case_ids_mismatch")
    records: dict[str, dict[str, Any]] = {}
    for case in pack["cases"]:
        request = request_v2.build_candidate_request(
            case, model=frozen_gate.MODEL_PATH
        )
        if request.get("model") != frozen_gate.MODEL_PATH:
            raise CardinalityDevsetError("reader_model_not_frozen_local_model")
        if (
            request.get("temperature") != 0
            or request.get("seed") != 42
            or request.get("max_tokens") != 256
            or request.get("stream") is not False
            or request.get("chat_template_kwargs") != {"enable_thinking": False}
        ):
            raise CardinalityDevsetError("request_controls_mismatch")
        schema = request["response_format"]["json_schema"]["schema"]
        atom_schema = schema["properties"]["atoms"]["items"]
        if "cardinality_proposal" in atom_schema["properties"]:
            raise CardinalityDevsetError("model_cardinality_field_not_removed")
        Draft202012Validator.check_schema(schema)
        if len(request.get("messages", [])) != 2:
            raise CardinalityDevsetError("request_message_count_mismatch")
        user_payload = _strict_json(request["messages"][1]["content"])
        if set(user_payload) != {"source_id", "proposition_text", "typed_candidates"}:
            raise CardinalityDevsetError("oracle_or_unexpected_data_in_prompt")
        if user_payload["source_id"] != case["case_id"]:
            raise CardinalityDevsetError("request_case_id_mismatch")
        body = json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        records[case["case_id"]] = {
            "request": request,
            "body": body,
            "request_sha256": _sha(body),
            "schema_sha256": _sha(_canonical_json(schema)),
        }
    return pack, records, v1_requests


def build_lock_payload() -> dict[str, Any]:
    _, requests, _ = _build_requests()
    lock = {
        **LOCK_STATIC_FIELDS,
        "runner_sha256": _sha(Path(__file__).resolve().read_bytes()),
        "freeze_script_sha256": _sha(FREEZE_PATH.read_bytes()),
        "dependencies": {
            relative: _sha((ROOT / relative).read_bytes())
            for relative in DEPENDENCY_PATHS
        },
        "requests": {
            case_id: {
                "request_sha256": row["request_sha256"],
                "schema_sha256": row["schema_sha256"],
            }
            for case_id, row in requests.items()
        },
    }
    return lock


def _load_lock() -> tuple[dict[str, Any], str, dict[str, Any], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    lock_bytes = LOCK_PATH.read_bytes()
    lock_sha = _sha(lock_bytes)
    sidecar = LOCK_SHA_PATH.read_text(encoding="ascii").split()
    if not sidecar or sidecar[0].lower() != lock_sha:
        raise CardinalityDevsetError("lock_sidecar_mismatch")
    lock = _strict_json(lock_bytes)
    pack, requests, v1_requests = _build_requests()
    if set(lock) != set(LOCK_STATIC_FIELDS) | LOCK_DYNAMIC_FIELDS:
        raise CardinalityDevsetError("lock_shape_mismatch")
    for key, expected in LOCK_STATIC_FIELDS.items():
        if _canonical_json(lock.get(key)) != _canonical_json(expected):
            raise CardinalityDevsetError(f"lock_field_mismatch:{key}")
    if sys.version.split()[0] != LOCK_STATIC_FIELDS["python_version"]:
        raise CardinalityDevsetError("python_version_mismatch")
    if package_version("jsonschema") != LOCK_STATIC_FIELDS["jsonschema_version"]:
        raise CardinalityDevsetError("jsonschema_version_mismatch")
    expected = build_lock_payload()
    for key in LOCK_DYNAMIC_FIELDS:
        if lock.get(key) != expected[key]:
            raise CardinalityDevsetError(f"lock_field_mismatch:{key}")
    if OUTPUT_ROOT.exists():
        raise CardinalityDevsetError("output_directory_already_exists")
    frozen_gate._load_frozen_inputs()
    return lock, lock_sha, pack, requests, v1_requests


def _case_by_id(pack: dict[str, Any], case_id: str) -> dict[str, Any]:
    matches = [row for row in pack["cases"] if row.get("case_id") == case_id]
    if len(matches) != 1:
        raise CardinalityDevsetError(f"case_not_unique:{case_id}")
    return matches[0]


def _score_response(
    case: dict[str, Any],
    model_content: str,
    *,
    v2_request: dict[str, Any],
    v1_request: dict[str, Any],
    response_sha256: str,
    usage: Any,
    latency_ms: float,
    finish_reason: Any,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "case_id": case["case_id"],
        "status": "QUALITY_FAILURE",
        "expected_atoms": len(case["expected_atoms"]),
        "completion_tokens": usage.get("completion_tokens") if isinstance(usage, dict) else None,
        "prompt_tokens": usage.get("prompt_tokens") if isinstance(usage, dict) else None,
        "latency_ms": latency_ms,
        "response_sha256": response_sha256,
        "finish_reason": finish_reason,
        "hosted_calls": 0,
        "memory_store_mutations": 0,
        "retry_count": 0,
        "failure": None,
    }
    try:
        payload = _strict_json(model_content)
        Draft202012Validator(
            v2_request["response_format"]["json_schema"]["schema"]
        ).validate(payload)
    except (CardinalityDevsetError, ValidationError, TypeError) as exc:
        result["schema_validation"] = "FAIL"
        result["failure"] = (
            "generation_truncated_at_token_cap"
            if finish_reason == "length"
            else f"schema_or_json:{type(exc).__name__}:{exc}"
        )
        result["generation_truncated"] = finish_reason == "length"
        return result
    result["schema_validation"] = "PASS"
    proposition = {
        "source_id": case["case_id"],
        "proposition_text": case["proposition_text"],
    }
    try:
        adapted = request_v2.materialize_cardinality(
            model_content, proposition, scope_id=frozen_r4.FROZEN_SCOPE_ID
        )
    except (ValueError, TypeError) as exc:
        result["failure"] = f"harness_materialization:{type(exc).__name__}:{exc}"
        result["cardinality_materialization"] = "FAIL"
        return result
    result["cardinality_materialization"] = "PASS"
    result["adapted_proposal_sha256"] = _sha(adapted.encode("utf-8"))
    scored = v1_runner._validate_proposal(
        case,
        adapted,
        request=v1_request,
        response_sha256=response_sha256,
        http_status=200,
        attempt_count=1,
        retry_count=0,
        latency_ms=latency_ms,
        usage=usage,
    )
    result.update(scored)
    result["case_id"] = case["case_id"]
    result["finish_reason"] = finish_reason
    result["generation_truncated"] = False
    result["cardinality_authority"] = "HARNESS_FROZEN_SLOT_POLICY"
    return result


def _run_case(
    case: dict[str, Any],
    v2_record: dict[str, Any],
    v1_request: dict[str, Any],
) -> dict[str, Any]:
    case_id = case["case_id"]
    case_dir = OUTPUT_ROOT / case_id
    case_dir.mkdir(parents=True, exist_ok=False)
    body = v2_record["body"]
    request_sha = v2_record["request_sha256"]
    _write_json_exclusive(
        case_dir / "reservation.json",
        {"status": "RESERVED_NO_RETRY", "run_id": OUTPUT_ROOT.name, "case_id": case_id, "request_sha256": request_sha},
    )
    _write_exclusive(case_dir / "request.json", body)
    guard = smoke.LoopbackOneShotGuard(body)
    response_bytes = None
    status: int | None = None
    latency_ms = None
    result: dict[str, Any] = {
        "run_id": OUTPUT_ROOT.name,
        "case_id": case_id,
        "request_sha256": request_sha,
        "completion_post_attempts": 0,
        "retry_count": 0,
        "hosted_calls": 0,
        "memory_store_mutations": 0,
        "status": "PREFLIGHT_FAILURE",
        "evidence_type": "NEW_LOCAL_ONE_SHOT_POST",
    }
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
                    body=body,
                    headers={"Content-Type": "application/json", "Accept": "application/json"},
                )
                response = connection.getresponse()
                status = response.status
                response_bytes = response.read()
                latency_ms = round((time.perf_counter() - start) * 1000, 3)
            finally:
                connection.close()
            guard.verify_postflight()
        result["runtime_checkpoints"] = guard.events
        result["completion_post_attempts"] = guard.post_attempts
        result["http_status"] = status
        if response_bytes is not None:
            _write_exclusive(case_dir / "response_body.bin", response_bytes)
        if status != 200 or response_bytes is None:
            result.update({"status": "INFRA_FAILURE", "failure": "completion_http_failure"})
        else:
            envelope, model_content = smoke._response_content(response_bytes)
            finish_reason = (
                envelope.get("choices", [{}])[0].get("finish_reason")
                if isinstance(envelope, dict)
                else None
            )
            scored = _score_response(
                case,
                model_content,
                v2_request=v2_record["request"],
                v1_request=v1_request,
                response_sha256=_sha(response_bytes),
                usage=envelope.get("usage"),
                latency_ms=latency_ms,
                finish_reason=finish_reason,
            )
            result.update(scored)
            result["completion_post_attempts"] = guard.post_attempts
            result["runtime_checkpoints"] = guard.events
            result["http_status"] = status
            if scored.get("admitted_normalized_atoms"):
                _write_json_exclusive(
                    case_dir / "admitted_atoms.json",
                    {"source_id": case_id, "scope_id": frozen_r4.FROZEN_SCOPE_ID, "atoms": scored["admitted_normalized_atoms"]},
                )
    except Exception as exc:
        result.update(
            {
                "completion_post_attempts": guard.post_attempts,
                "runtime_checkpoints": guard.events,
                "status": "INFRA_FAILURE" if guard.post_attempts else "PREFLIGHT_FAILURE",
                "failure": f"{type(exc).__name__}:{exc}",
                "delivery_state": "UNKNOWN" if guard.post_attempts else "NOT_SENT",
            }
        )
        if response_bytes is not None and not (case_dir / "response_body.bin").exists():
            _write_exclusive(case_dir / "response_body.bin", response_bytes)
    _write_json_exclusive(case_dir / "case_result.json", result)
    return result


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if row.get("status") in {"QUALITY_PASS", "QUALITY_FAILURE"}]
    expected = sum(row.get("expected_atoms", 0) for row in completed)
    admitted = sum(row.get("admitted_atoms", 0) for row in completed)
    matched = sum(row.get("matched_expected_atoms", 0) for row in completed)
    return {
        "case_count": len(rows),
        "completed_quality_cases": len(completed),
        "quality_pass_cases": sum(row.get("status") == "QUALITY_PASS" for row in rows),
        "quality_failure_cases": sum(row.get("status") == "QUALITY_FAILURE" for row in rows),
        "infra_or_preflight_failures": sum(row.get("status") in {"INFRA_FAILURE", "PREFLIGHT_FAILURE"} for row in rows),
        "exact_atom_match_cases": sum(row.get("exact_expected_atom_multiset_match") is True for row in rows),
        "expected_atoms": expected,
        "matched_expected_atoms": matched,
        "admitted_atoms": admitted,
        "atom_precision": matched / admitted if admitted else None,
        "atom_recall": matched / expected if expected else None,
        "new_post_attempts": sum(row.get("completion_post_attempts", 0) for row in rows),
        "hosted_calls": sum(row.get("hosted_calls", 0) for row in rows),
        "retries": sum(row.get("retry_count", 0) for row in rows),
        "memory_store_mutations": sum(row.get("memory_store_mutations", 0) for row in rows),
        "generation_truncated_cases": sum(row.get("generation_truncated") is True for row in rows),
    }


def run_once() -> dict[str, Any]:
    lock, lock_sha, pack, requests, v1_requests = _load_lock()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    _write_json_exclusive(
        OUTPUT_ROOT / "run_reservation.json",
        {"status": "RESERVED_NO_RETRY", "run_id": OUTPUT_ROOT.name, "lock_sha256": lock_sha, "case_ids": list(CASE_IDS)},
    )
    rows = []
    infra_stop = False
    for case_id in CASE_IDS:
        if infra_stop:
            rows.append({"case_id": case_id, "status": "NOT_RUN_AFTER_INFRA_FAILURE", "completion_post_attempts": 0, "retry_count": 0, "hosted_calls": 0, "memory_store_mutations": 0})
            continue
        rows.append(
            _run_case(
                _case_by_id(pack, case_id), requests[case_id], v1_requests[case_id]["request"]
            )
        )
        if rows[-1].get("status") in {"INFRA_FAILURE", "PREFLIGHT_FAILURE"}:
            infra_stop = True
    aggregate = _aggregate(rows)
    result = {
        "schema_version": 1,
        "run_id": OUTPUT_ROOT.name,
        "lock_sha256": lock_sha,
        "status": "COMPLETE" if aggregate["completed_quality_cases"] == len(CASE_IDS) else "INFRA_INCOMPLETE",
        "evaluation_scope": "SYNTHETIC_PROMPTED_DEVELOPMENT_DIAGNOSTIC_ONLY",
        "reader_model_path": frozen_gate.MODEL_PATH,
        "reader_model_sha256": frozen_gate.MODEL_SHA256,
        "cardinality_authority": "HARNESS_FROZEN_SLOT_POLICY",
        "hosted_api": "NONE",
        "required_api_key": "NONE",
        "aggregate": aggregate,
        "cases": rows,
        "limitations": [
            "Project-authored prompted synthetic controls; not a public benchmark or generalization estimate.",
            "All six cases receive one fresh request under the v2 no-cardinality schema.",
            "The prompt names distractors; this measures prompted contract behavior only.",
            "No MemoryStore mutation, revision materialization, temporal query, or clinical workflow was evaluated.",
            "This protocol was designed after inspecting v1 DEV results; it is a development iteration, not an independent test.",
        ],
    }
    _write_json_exclusive(OUTPUT_ROOT / "devset_result.json", result)
    artifacts = {
        str(path.relative_to(OUTPUT_ROOT)).replace("\\", "/"): _sha(path.read_bytes())
        for path in sorted(OUTPUT_ROOT.rglob("*"))
        if path.is_file() and path.name != "run_manifest.json"
    }
    _write_json_exclusive(
        OUTPUT_ROOT / "run_manifest.json",
        {"manifest_id": "mem3b0q-r4-cardinality-devset-run-v2", "status": result["status"], "lock_sha256": lock_sha, "runner_sha256": lock["runner_sha256"], "case_ids": list(CASE_IDS), "aggregate": aggregate, "artifact_sha256": artifacts},
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--run-six-case-v2-once", action="store_true")
    args = parser.parse_args()
    if not args.run_six_case_v2_once:
        parser.error("explicit --run-six-case-v2-once is required")
    result = run_once()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if result.get("status") == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())

