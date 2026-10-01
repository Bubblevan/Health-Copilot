"""Run the frozen six-case synthetic candidate diagnostic on local Qwen only."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import sys
import time
from collections import Counter
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, ValidationError

from tools.research.memory import mem3b0q_r4 as frozen_r4
from tools.research.memory import mem3b0q_r4_candidate_binding_guard_v1 as locality_guard
from tools.research.memory import mem3b0q_r4_candidate_builder_v1 as candidate_builder
from tools.research.memory import mem3b0q_r4_candidate_request_v1 as request_builder
from tools.research.memory import mem3b0q_r4_joint_binding_guard_v2 as joint_guard_v2
from tools.research.memory import mem3b0q_r4_joint_binding_guard_v3 as joint_guard_v3
from tools.research.memory import run_mem3b0q_r4_candidate_local_smoke_v1 as smoke
from tools.research.memory import run_mem3b0q_r4_gate as frozen_gate
from tools.research.memory.mem3b0q_r4_atomwise_admission_v1 import (
    AtomwiseAdmissionError,
    admit_atoms_independently,
)
from tools.research.memory.preflight_mem3b0q_r4_runtime_readonly_v1 import (
    validate_runtime_snapshot,
)


ROOT = Path(__file__).resolve().parents[3]
DOCS = ROOT / "docs" / "research" / "memory"
PACK_PATH = DOCS / "mem3b0q_r4_candidate_control_pack_v1.json"
PROTOCOL_PATH = DOCS / "mem3b0q_r4_atomwise_candidate_devset_protocol_v1.md"
LOCK_PATH = DOCS / "mem3b0q_r4_atomwise_candidate_devset_lock_v1.json"
LOCK_SHA_PATH = LOCK_PATH.with_suffix(LOCK_PATH.suffix + ".sha256")
OUTPUT_ROOT = (
    ROOT
    / "runs"
    / "memory"
    / "mem3"
    / "mem3b0q-r4-atomwise-candidate-devset-v1"
)
PRIOR_DIR = (
    ROOT / "runs" / "memory" / "mem3" / "mem3b0q-r4-candidate-local-smoke-v1"
)
CASE_IDS = tuple(f"R4C-{index:02d}" for index in range(1, 7))
POST_CASE_IDS = tuple(case_id for case_id in CASE_IDS if case_id != "R4C-05")
REPLAY_CASE_ID = "R4C-05"

DEPENDENCY_PATHS = (
    "docs/research/memory/mem3b0q_r4_atomwise_candidate_devset_protocol_v1.md",
    "docs/research/memory/mem3b0q_r4_candidate_control_pack_v1.json",
    "docs/research/memory/mem3b0q_r4_candidate_local_smoke_protocol_v1.md",
    "docs/research/memory/mem3b0q_r4_candidate_local_smoke_runner_v1.md",
    "docs/research/memory/mem3b0q_r4_candidate_local_smoke_lock_v1.json",
    "docs/research/memory/mem3b0q_r4_candidate_local_smoke_lock_v1.json.sha256",
    "docs/research/memory/mem3b0q_r4_candidate_local_smoke_result_v1.md",
    "tools/research/memory/mem3b0q_r4.py",
    "tools/research/memory/mem3b0q_r4_atomwise_admission_v1.py",
    "tools/research/memory/mem3b0q_r4_candidate_binding_guard_v1.py",
    "tools/research/memory/mem3b0q_r4_candidate_builder_v1.py",
    "tools/research/memory/mem3b0q_r4_candidate_request_v1.py",
    "tools/research/memory/mem3b0q_r4_joint_binding_guard_v2.py",
    "tools/research/memory/mem3b0q_r4_joint_binding_guard_v3.py",
    "tools/research/memory/preflight_mem3b0q_r4_runtime_readonly_v1.py",
    "tools/research/memory/run_mem3b0q_r4_candidate_local_smoke_v1.py",
    "tools/research/memory/run_mem3b0q_r4_gate.py",
    "runs/memory/mem3/mem3b0q-r4-candidate-local-smoke-v1/request.json",
    "runs/memory/mem3/mem3b0q-r4-candidate-local-smoke-v1/reservation.json",
    "runs/memory/mem3/mem3b0q-r4-candidate-local-smoke-v1/response_body.bin",
    "runs/memory/mem3/mem3b0q-r4-candidate-local-smoke-v1/result.json",
)

EXPECTED_LOCK_FIELDS = {
    "schema_version": 1,
    "status": "FROZEN",
    "lock_id": "mem3b0q-r4-atomwise-candidate-devset-lock-v1",
    "run_id": OUTPUT_ROOT.name,
    "case_ids": list(CASE_IDS),
    "new_post_case_ids": list(POST_CASE_IDS),
    "replayed_case_id": REPLAY_CASE_ID,
    "max_new_completion_posts": len(POST_CASE_IDS),
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
    "request_controls": {
        "temperature": 0,
        "seed": 42,
        "max_tokens": 256,
        "stream": False,
        "thinking": False,
        "response_format": "strict_json_schema",
    },
    "adapter_version": "mem3b0q-r4-atomwise-admission-v1-development",
    "quality_metric": "exact_normalized_atom_multiset_precision_recall",
    "python_version": "3.12.4",
    "jsonschema_version": "4.19.2",
    "output_policy": "append_never_overwrite_never_retry",
}
LOCK_EXTRA_FIELDS = frozenset(
    {"runner_sha256", "dependencies", "requests", "replayed_case_artifacts"}
)


class DevsetRunError(RuntimeError):
    """A frozen request, runtime, or run-artifact invariant failed closed."""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _strict_json(raw: bytes | str) -> Any:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise DevsetRunError(f"duplicate_json_key:{key}")
            result[key] = value
        return result

    try:
        text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        return json.loads(text, object_pairs_hook=unique_object)
    except DevsetRunError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
        raise DevsetRunError("malformed_json") from exc


def _write_exclusive(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json_exclusive(path: Path, value: Any) -> None:
    _write_exclusive(path, _canonical_json(value) + b"\n")


def _build_requests() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    pack = _strict_json(PACK_PATH.read_bytes())
    if (
        pack.get("status") != "FROZEN_PROTOCOL_NO_INFERENCE_AUTHORIZATION"
        or pack.get("pack_id") != "mem3b0q-r4-candidate-control-pack-v1"
        or pack.get("clinical_content") is not False
    ):
        raise DevsetRunError("candidate_control_pack_identity_mismatch")
    cases = pack.get("cases")
    if not isinstance(cases, list) or [case.get("case_id") for case in cases] != list(CASE_IDS):
        raise DevsetRunError("candidate_control_case_ids_mismatch")

    request_records: dict[str, dict[str, Any]] = {}
    for case in cases:
        request = request_builder.build_candidate_request(
            case, model=frozen_gate.MODEL_PATH
        )
        if request.get("model") != frozen_gate.MODEL_PATH:
            raise DevsetRunError("reader_model_not_frozen_local_model")
        if (
            request.get("temperature") != 0
            or request.get("seed") != 42
            or request.get("max_tokens") != 256
            or request.get("stream") is not False
            or request.get("chat_template_kwargs") != {"enable_thinking": False}
        ):
            raise DevsetRunError("request_controls_mismatch")
        if len(request.get("messages", [])) != 2:
            raise DevsetRunError("request_message_count_mismatch")
        user_payload = _strict_json(request["messages"][1]["content"])
        if set(user_payload) != {"source_id", "proposition_text", "typed_candidates"}:
            raise DevsetRunError("oracle_or_unexpected_data_in_reader_prompt")
        if user_payload["source_id"] != case["case_id"]:
            raise DevsetRunError("request_case_id_mismatch")
        schema = request["response_format"]["json_schema"]["schema"]
        Draft202012Validator.check_schema(schema)
        body = json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
        request_records[case["case_id"]] = {
            "request": request,
            "body": body,
            "request_sha256": _sha(body),
            "schema_sha256": _sha(_canonical_json(schema)),
        }
    return pack, request_records


def _prior_case_artifacts() -> dict[str, str]:
    paths = {
        "request_sha256": PRIOR_DIR / "request.json",
        "reservation_sha256": PRIOR_DIR / "reservation.json",
        "response_sha256": PRIOR_DIR / "response_body.bin",
        "result_sha256": PRIOR_DIR / "result.json",
    }
    return {key: _sha(path.read_bytes()) for key, path in paths.items()}


def _load_lock() -> tuple[dict[str, Any], str, dict[str, Any], dict[str, dict[str, Any]]]:
    lock_bytes = LOCK_PATH.read_bytes()
    lock_sha = _sha(lock_bytes)
    sidecar = LOCK_SHA_PATH.read_text(encoding="ascii").split()
    if not sidecar or sidecar[0].lower() != lock_sha:
        raise DevsetRunError("lock_sidecar_mismatch")
    lock = _strict_json(lock_bytes)
    pack, requests = _build_requests()
    if set(lock) != set(EXPECTED_LOCK_FIELDS) | LOCK_EXTRA_FIELDS:
        raise DevsetRunError("lock_shape_mismatch")
    for key, expected in EXPECTED_LOCK_FIELDS.items():
        if _canonical_json(lock.get(key)) != _canonical_json(expected):
            raise DevsetRunError(f"lock_field_mismatch:{key}")
    if sys.version.split()[0] != EXPECTED_LOCK_FIELDS["python_version"]:
        raise DevsetRunError("python_version_mismatch")
    if package_version("jsonschema") != EXPECTED_LOCK_FIELDS["jsonschema_version"]:
        raise DevsetRunError("jsonschema_version_mismatch")
    if lock.get("runner_sha256") != _sha(Path(__file__).resolve().read_bytes()):
        raise DevsetRunError("runner_hash_mismatch")
    expected_dependencies = {
        relative: _sha((ROOT / relative).read_bytes())
        for relative in DEPENDENCY_PATHS
    }
    if lock.get("dependencies") != expected_dependencies:
        raise DevsetRunError("dependency_hash_mismatch")
    expected_requests = {
        case_id: {
            "request_sha256": record["request_sha256"],
            "schema_sha256": record["schema_sha256"],
        }
        for case_id, record in requests.items()
    }
    if lock.get("requests") != expected_requests:
        raise DevsetRunError("request_hash_map_mismatch")
    if lock.get("replayed_case_artifacts") != _prior_case_artifacts():
        raise DevsetRunError("replayed_case_artifact_hash_mismatch")
    if OUTPUT_ROOT.exists():
        raise DevsetRunError("output_directory_already_exists")
    frozen_gate._load_frozen_inputs()
    return lock, lock_sha, pack, requests


def _case_by_id(pack: dict[str, Any], case_id: str) -> dict[str, Any]:
    matches = [case for case in pack["cases"] if case.get("case_id") == case_id]
    if len(matches) != 1:
        raise DevsetRunError(f"case_not_unique:{case_id}")
    return matches[0]


def _validate_proposal(
    case: dict[str, Any],
    model_content: str,
    *,
    request: dict[str, Any],
    response_sha256: str,
    http_status: int,
    attempt_count: int,
    retry_count: int,
    latency_ms: float | None,
    usage: Any = None,
) -> dict[str, Any]:
    proposition = {
        "source_id": case["case_id"],
        "proposition_text": case["proposition_text"],
    }
    result: dict[str, Any] = {
        "case_id": case["case_id"],
        "http_status": http_status,
        "completion_post_attempts": attempt_count,
        "retry_count": retry_count,
        "hosted_calls": 0,
        "response_sha256": response_sha256,
        "latency_ms": latency_ms,
        "usage": usage,
        "prompt_tokens": usage.get("prompt_tokens")
        if isinstance(usage, dict)
        else None,
        "completion_tokens": usage.get("completion_tokens")
        if isinstance(usage, dict)
        else None,
        "expected_atoms": len(case["expected_atoms"]),
        "admitted_atoms": 0,
        "matched_expected_atoms": 0,
        "false_positive_atoms": 0,
        "false_negative_atoms": len(case["expected_atoms"]),
        "atom_precision": None,
        "atom_recall": 0.0 if case["expected_atoms"] else None,
        "status": "QUALITY_FAILURE",
        "failure": None,
    }
    try:
        proposal_object = _strict_json(model_content)
        Draft202012Validator(
            request["response_format"]["json_schema"]["schema"]
        ).validate(proposal_object)
    except (DevsetRunError, ValidationError, TypeError) as exc:
        result["failure"] = f"schema_or_json:{type(exc).__name__}:{exc}"
        result["schema_validation"] = "FAIL"
        return result
    result["schema_validation"] = "PASS"

    try:
        strict = joint_guard_v3.validate_joint_bound_candidate_proposal(
            model_content,
            proposition,
            scope_id=frozen_r4.FROZEN_SCOPE_ID,
        )
    except (
        candidate_builder.CandidateBuildError,
        locality_guard.CandidateBindingError,
        joint_guard_v2.JointBindingError,
        joint_guard_v3.JointBindingError,
    ) as exc:
        strict_accepts = False
        strict_error = f"{type(exc).__name__}:{exc}"
        strict_atoms: list[dict[str, Any]] = []
    else:
        strict_accepts = True
        strict_error = None
        strict_atoms = strict["atoms"]
    result["whole_document_guard_accepts"] = strict_accepts
    result["whole_document_guard_error"] = strict_error

    try:
        admission = admit_atoms_independently(
            model_content,
            proposition,
            scope_id=frozen_r4.FROZEN_SCOPE_ID,
        )
    except AtomwiseAdmissionError as exc:
        result["failure"] = f"atomwise_envelope:{exc}"
        result["atomwise_status"] = "REJECTED"
        return result

    admitted = [row["atom"] for row in admission["admitted"]]
    normalized = {
        "source_id": case["case_id"],
        "scope_id": frozen_r4.FROZEN_SCOPE_ID,
        "atoms": admitted,
    }
    expected = Counter(
        request_builder._expected_atom_signature(
            case, row, scope_id=frozen_r4.FROZEN_SCOPE_ID
        )
        for row in case["expected_atoms"]
    )
    actual = Counter(request_builder.normalized_atom_signature(row) for row in admitted)
    matched = sum((expected & actual).values())
    exact = expected == actual
    admitted_count = sum(actual.values())
    expected_count = sum(expected.values())
    result.update(
        {
            "atomwise_status": admission["status"],
            "schema_validation": "PASS",
            "expected_atoms": expected_count,
            "admitted_atoms": admitted_count,
            "matched_expected_atoms": matched,
            "quarantined_atoms": admission["quarantined"],
            "false_positive_atoms": admitted_count - matched,
            "false_negative_atoms": expected_count - matched,
            "atom_precision": matched / admitted_count if admitted_count else None,
            "atom_recall": matched / expected_count if expected_count else None,
            "exact_expected_atom_multiset_match": exact,
            "strict_document_admitted_atom_count": len(strict_atoms),
            "admitted_normalized_atoms": admitted,
            "status": "QUALITY_PASS" if exact else "QUALITY_FAILURE",
            "failure": None if exact else "exact_expected_atom_multiset_mismatch",
        }
    )
    return result


def _load_replayed_case(
    pack: dict[str, Any], requests: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    case = _case_by_id(pack, REPLAY_CASE_ID)
    old_request = (PRIOR_DIR / "request.json").read_bytes()
    if old_request != requests[REPLAY_CASE_ID]["body"]:
        raise DevsetRunError("r4c05_replay_request_not_byte_identical")
    old_result = _strict_json((PRIOR_DIR / "result.json").read_bytes())
    if (
        old_result.get("case_id") != REPLAY_CASE_ID
        or old_result.get("status") != "QUALITY_FAILURE"
        or old_result.get("completion_post_attempts") != 1
        or old_result.get("retry_count") != 0
        or old_result.get("hosted_calls") != 0
        or old_result.get("request_sha256") != requests[REPLAY_CASE_ID]["request_sha256"]
    ):
        raise DevsetRunError("r4c05_replay_result_mismatch")
    response_bytes = (PRIOR_DIR / "response_body.bin").read_bytes()
    envelope, model_content = smoke._response_content(response_bytes)
    replay_result = _validate_proposal(
        case,
        model_content,
        request=requests[REPLAY_CASE_ID]["request"],
        response_sha256=_sha(response_bytes),
        http_status=old_result["http_status"],
        attempt_count=0,
        retry_count=0,
        latency_ms=None,
        usage=envelope.get("usage"),
    )
    replay_result.update(
        {
            "evidence_type": "REPLAY_PRIOR_SINGLE_CASE_RESPONSE",
            "historical_post_attempts": old_result["completion_post_attempts"],
            "request_sha256": requests[REPLAY_CASE_ID]["request_sha256"],
            "prior_smoke_result_sha256": _sha(
                (PRIOR_DIR / "result.json").read_bytes()
            ),
            "new_model_call": False,
        }
    )
    return replay_result


def _run_one_case(
    case: dict[str, Any], request_record: dict[str, Any]
) -> dict[str, Any]:
    case_id = case["case_id"]
    case_dir = OUTPUT_ROOT / case_id
    case_dir.mkdir(parents=True, exist_ok=False)
    body = request_record["body"]
    request_sha = request_record["request_sha256"]
    reservation = {
        "status": "RESERVED_NO_RETRY",
        "run_id": OUTPUT_ROOT.name,
        "case_id": case_id,
        "request_sha256": request_sha,
    }
    _write_json_exclusive(case_dir / "reservation.json", reservation)
    _write_exclusive(case_dir / "request.json", body)
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
    response_bytes: bytes | None = None
    response_status: int | None = None
    request_latency_ms: float | None = None
    guard = smoke.LoopbackOneShotGuard(body)
    post_started: float | None = None
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
                post_started = time.perf_counter()
                connection.request(
                    "POST",
                    frozen_gate.ENDPOINT_PATH,
                    body=body,
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                    },
                )
                response = connection.getresponse()
                response_status = response.status
                response_bytes = response.read()
                request_latency_ms = round(
                    (time.perf_counter() - post_started) * 1000, 3
                )
            finally:
                connection.close()
            guard.verify_postflight()
        result["runtime_checkpoints"] = guard.events
        result["completion_post_attempts"] = guard.post_attempts
        result["http_status"] = response_status
        result["latency_ms"] = request_latency_ms
        if response_bytes is not None:
            _write_exclusive(case_dir / "response_body.bin", response_bytes)
        if response_status != 200 or response_bytes is None:
            result.update(
                {
                    "status": "INFRA_FAILURE",
                    "failure": "completion_http_failure",
                }
            )
        else:
            envelope, model_content = smoke._response_content(response_bytes)
            scored = _validate_proposal(
                case,
                model_content,
                request=request_record["request"],
                response_sha256=_sha(response_bytes),
                http_status=response_status,
                attempt_count=guard.post_attempts,
                retry_count=0,
                latency_ms=result["latency_ms"],
                usage=envelope.get("usage"),
            )
            result.update(scored)
            if scored.get("admitted_normalized_atoms"):
                _write_json_exclusive(
                    case_dir / "admitted_atoms.json",
                    {
                        "source_id": case_id,
                        "scope_id": frozen_r4.FROZEN_SCOPE_ID,
                        "atoms": scored["admitted_normalized_atoms"],
                    },
                )
    except Exception as exc:
        result.update(
            {
                "completion_post_attempts": guard.post_attempts,
                "runtime_checkpoints": guard.events,
                "status": "INFRA_FAILURE"
                if guard.post_attempts
                else "PREFLIGHT_FAILURE",
                "failure": f"{type(exc).__name__}:{exc}",
                "delivery_state": "UNKNOWN" if guard.post_attempts else "NOT_SENT",
            }
        )
        if response_bytes is not None and not (case_dir / "response_body.bin").exists():
            _write_exclusive(case_dir / "response_body.bin", response_bytes)
    _write_json_exclusive(case_dir / "case_result.json", result)
    return result


def _aggregate(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [
        row
        for row in case_results
        if row.get("status") in {"QUALITY_PASS", "QUALITY_FAILURE"}
    ]
    expected_count = sum(row.get("expected_atoms", 0) for row in completed)
    admitted_count = sum(row.get("admitted_atoms", 0) for row in completed)
    matched_count = sum(row.get("matched_expected_atoms", 0) for row in completed)
    quarantine_count = sum(len(row.get("quarantined_atoms", [])) for row in completed)
    exact_count = sum(
        row.get("exact_expected_atom_multiset_match") is True for row in completed
    )
    strict_doc_count = sum(
        row.get("strict_document_admitted_atom_count", 0) for row in completed
    )
    return {
        "case_count": len(case_results),
        "completed_quality_cases": len(completed),
        "quality_pass_cases": sum(row.get("status") == "QUALITY_PASS" for row in completed),
        "quality_failure_cases": sum(
            row.get("status") == "QUALITY_FAILURE" for row in completed
        ),
        "infra_or_preflight_failures": sum(
            row.get("status") in {"INFRA_FAILURE", "PREFLIGHT_FAILURE"}
            for row in case_results
        ),
        "exact_expected_atom_multiset_match_cases": exact_count,
        "expected_atoms": expected_count,
        "matched_expected_atoms": matched_count,
        "admitted_atoms": admitted_count,
        "quarantined_atoms": quarantine_count,
        "atom_precision": matched_count / admitted_count if admitted_count else None,
        "atom_recall": matched_count / expected_count if expected_count else None,
        "strict_document_admitted_atoms": strict_doc_count,
        "valid_atom_retention_vs_strict": (
            matched_count / expected_count if expected_count else None
        ),
        "new_post_attempts": sum(row.get("completion_post_attempts", 0) for row in case_results),
        "new_hosted_calls": sum(row.get("hosted_calls", 0) for row in case_results),
        "new_retries": sum(row.get("retry_count", 0) for row in case_results),
        "memory_store_mutations": sum(
            row.get("memory_store_mutations", 0) for row in case_results
        ),
    }


def run_devset_once() -> dict[str, Any]:
    lock, lock_sha, pack, requests = _load_lock()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    _write_json_exclusive(
        OUTPUT_ROOT / "run_reservation.json",
        {
            "status": "RESERVED_NO_RETRY",
            "run_id": lock["run_id"],
            "lock_sha256": lock_sha,
            "new_post_case_ids": list(POST_CASE_IDS),
            "replayed_case_id": REPLAY_CASE_ID,
        },
    )
    case_results = [_load_replayed_case(pack, requests)]
    infra_stop = False
    for case_id in POST_CASE_IDS:
        if infra_stop:
            case_results.append(
                {
                    "case_id": case_id,
                    "status": "NOT_RUN_AFTER_INFRA_FAILURE",
                    "completion_post_attempts": 0,
                    "retry_count": 0,
                    "hosted_calls": 0,
                    "memory_store_mutations": 0,
                }
            )
            continue
        result = _run_one_case(
            _case_by_id(pack, case_id), requests[case_id]
        )
        case_results.append(result)
        if result.get("status") in {"INFRA_FAILURE", "PREFLIGHT_FAILURE"}:
            infra_stop = True

    # Keep case order stable and make the prior-response replay explicit.
    by_id = {row["case_id"]: row for row in case_results}
    ordered = [by_id[case_id] for case_id in CASE_IDS]
    aggregate = _aggregate(ordered)
    result = {
        "schema_version": 1,
        "run_id": lock["run_id"],
        "lock_sha256": lock_sha,
        "status": "COMPLETE" if aggregate["completed_quality_cases"] == len(CASE_IDS) else "INFRA_INCOMPLETE",
        "evaluation_scope": "SYNTHETIC_PROMPTED_DEV_DIAGNOSTIC_ONLY",
        "reader_model_path": frozen_gate.MODEL_PATH,
        "reader_model_sha256": frozen_gate.MODEL_SHA256,
        "hosted_api": "NONE",
        "required_api_key": "NONE",
        "aggregate": aggregate,
        "cases": ordered,
        "limitations": [
            "This small project-authored synthetic control pack is not a public benchmark or unbiased generalization estimate.",
            "R4C-05 reuses its previously recorded local response; no duplicate request was sent.",
            "The prompt itself names the tablet and basket distractors, so the result measures prompted compliance only.",
            "No revision materializer, MemoryStore write, CURRENT/AS_OF/CHANGE query, or clinical workflow was evaluated.",
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
        {
            "manifest_id": "mem3b0q-r4-atomwise-candidate-devset-run-v1",
            "status": result["status"],
            "lock_sha256": lock_sha,
            "runner_sha256": lock["runner_sha256"],
            "case_ids": list(CASE_IDS),
            "aggregate": aggregate,
            "artifact_sha256": artifacts,
        },
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--run-five-post-devset-once",
        action="store_true",
        help="Run the five remaining frozen synthetic cases once; reuse R4C-05.",
    )
    args = parser.parse_args()
    if not args.run_five_post_devset_once:
        parser.error("explicit --run-five-post-devset-once is required")
    result = run_devset_once()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if result.get("status") == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
