"""One-shot, hash-locked local proposal smoke for the frozen R4C-05 control."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import sys
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from importlib.metadata import version as package_version

from jsonschema import Draft202012Validator

from tools.research.memory import mem3b0q_r4 as frozen_r4
from tools.research.memory import mem3b0q_r4_candidate_request_v1 as request_builder
from tools.research.memory import mem3b0q_r4_joint_binding_guard_v3 as guard_v3
from tools.research.memory import run_mem3b0q_r4_gate as frozen_gate
from tools.research.memory.preflight_mem3b0q_r4_runtime_readonly_v1 import (
    validate_runtime_snapshot,
)


ROOT = Path(__file__).resolve().parents[3]
DOCS = ROOT / "docs" / "research" / "memory"
PACK_PATH = DOCS / "mem3b0q_r4_candidate_control_pack_v1.json"
LOCK_PATH = DOCS / "mem3b0q_r4_candidate_local_smoke_lock_v1.json"
LOCK_SHA_PATH = LOCK_PATH.with_suffix(LOCK_PATH.suffix + ".sha256")
OUTPUT_DIR = ROOT / "runs" / "memory" / "mem3" / "mem3b0q-r4-candidate-local-smoke-v1"
CASE_ID = "R4C-05"
ALLOWED_GET_PATHS = frozenset({"/health", "/props", "/v1/models", "/slots"})
EXPECTED_DEPENDENCIES = (
    "docs/research/memory/mem3b0q_r4_candidate_control_pack_v1.json",
    "docs/research/memory/mem3b0q_r4_development_control_manifest_v1.json",
    "docs/research/memory/mem3b0q_r4_control_pack_v1.json",
    "docs/research/memory/mem3b0q_r4_candidate_local_smoke_protocol_v1.md",
    "docs/research/memory/mem3b0q_r4_candidate_local_smoke_runner_v1.md",
    "docs/research/memory/mem3b0q_r4_candidate_local_smoke_grammar_preflight_v1.json",
    "docs/research/memory/mem3b0q_r4_candidate_local_smoke_grammar_preflight_v1.json.sha256",
    "tools/research/memory/mem3b0q_r4_candidate_request_v1.py",
    "tools/research/memory/mem3b0q_r4_candidate_builder_v1.py",
    "tools/research/memory/mem3b0q_r4_candidate_binding_guard_v1.py",
    "tools/research/memory/mem3b0q_r4_joint_binding_guard_v2.py",
    "tools/research/memory/mem3b0q_r4_joint_binding_guard_v3.py",
    "tools/research/memory/mem3b0q_r4.py",
    "tools/research/memory/run_mem3b0q_r4_gate.py",
    "tools/research/memory/preflight_mem3b0q_r4_runtime_readonly_v1.py",
    "tools/research/memory/preflight_mem3b0q_r4_candidate_local_smoke_grammar_v1.py",
)
EXPECTED_LOCK_FIELDS = {
    "schema_version": 1,
    "status": "FROZEN",
    "lock_id": "mem3b0q-r4-candidate-local-smoke-lock-v1",
    "case_id": CASE_ID,
    "run_id": OUTPUT_DIR.name,
    "runner_path": "tools/research/memory/run_mem3b0q_r4_candidate_local_smoke_v1.py",
    "host": frozen_gate.HOST,
    "port": frozen_gate.PORT,
    "endpoint_path": frozen_gate.ENDPOINT_PATH,
    "request_method": "POST",
    "max_completion_posts": 1,
    "max_retries": 0,
    "hosted_api": "NONE",
    "required_api_key": "NONE",
    "embedding_model": "NONE",
    "judge_model": "NONE",
    "memory_store_mutations": 0,
    "network_policy": {
        "transport": "direct_http_client_no_proxy_resolution",
        "allowed_host": frozen_gate.HOST,
        "allowed_port": frozen_gate.PORT,
        "allowed_get_paths": sorted(ALLOWED_GET_PATHS),
        "allowed_post_path": frozen_gate.ENDPOINT_PATH,
        "https_outbound": "DENY",
        "other_hosts": "DENY",
        "ambient_proxy_used": False,
    },
    "model_role": "local_proposition_extractor",
    "proposal_model": {
        "model_path": frozen_gate.MODEL_PATH,
        "model_sha256": frozen_gate.MODEL_SHA256,
        "server_path": frozen_gate.SERVER_PATH,
        "server_sha256": frozen_gate.SERVER_SHA256,
        "llama_build": "b10068-571d0d540",
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
    "acceptance": "exact_normalized_atom_multiset",
    "output_policy": "append_never_overwrite_never_retry",
    "inference_authorization": "REQUIRES_SEPARATE_EXPLICIT_REVIEW",
    "python_version": "3.12.4",
    "jsonschema_version": "4.19.2",
}
EXTRA_LOCK_FIELDS = frozenset(
    {
        "runner_sha256",
        "dependencies",
        "request_sha256",
        "schema_sha256",
        "grammar_preflight",
    }
)


class CandidateSmokeError(RuntimeError):
    """A lock, runtime, request, or output invariant failed closed."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _strict_json_loads(raw: bytes | str) -> Any:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CandidateSmokeError(f"duplicate_json_key:{key}")
            result[key] = value
        return result

    return json.loads(raw, object_pairs_hook=reject_duplicate_keys)


def _write_exclusive(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "wb") as output:
        output.write(data)
        output.flush()
        os.fsync(output.fileno())


def _load_case_and_request() -> tuple[dict[str, Any], dict[str, Any], bytes]:
    pack = _strict_json_loads(PACK_PATH.read_text(encoding="utf-8"))
    if pack.get("status") != "FROZEN_PROTOCOL_NO_INFERENCE_AUTHORIZATION":
        raise CandidateSmokeError("candidate_control_pack_not_frozen")
    cases = pack.get("cases")
    if not isinstance(cases, list) or len(cases) != 6:
        raise CandidateSmokeError("candidate_control_pack_shape_mismatch")
    matches = [case for case in cases if case.get("case_id") == CASE_ID]
    if len(matches) != 1:
        raise CandidateSmokeError("candidate_smoke_case_not_unique")
    case = matches[0]
    if len(case.get("expected_atoms", [])) != 1:
        raise CandidateSmokeError("candidate_smoke_expected_atom_count_mismatch")
    request = request_builder.build_candidate_request(
        case, model=frozen_gate.MODEL_PATH
    )
    body = json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    return case, request, body


def _load_and_validate_lock() -> tuple[dict[str, Any], str, dict[str, Any], bytes]:
    lock_bytes = LOCK_PATH.read_bytes()
    lock_sha = _sha256(lock_bytes)
    sidecar = LOCK_SHA_PATH.read_text(encoding="ascii").split()
    if not sidecar or sidecar[0].lower() != lock_sha:
        raise CandidateSmokeError("run_lock_sidecar_mismatch")
    lock = _strict_json_loads(lock_bytes.decode("utf-8"))
    case, request, body = _load_case_and_request()
    schema = request["response_format"]["json_schema"]["schema"]
    Draft202012Validator.check_schema(schema)
    schema_sha = _sha256(_canonical_json(schema))
    grammar_path = DOCS / "mem3b0q_r4_candidate_local_smoke_grammar_preflight_v1.json"
    grammar_bytes = grammar_path.read_bytes()
    grammar_sha = _sha256(grammar_bytes)
    grammar = _strict_json_loads(grammar_bytes)
    grammar_sidecar = (
        DOCS / "mem3b0q_r4_candidate_local_smoke_grammar_preflight_v1.json.sha256"
    ).read_text(encoding="ascii").split()
    grammar_lock = {
        "path": "docs/research/memory/mem3b0q_r4_candidate_local_smoke_grammar_preflight_v1.json",
        "sha256": grammar_sha,
        "status": "PASS_GRAMMAR_CONVERSION_PRE_MODEL_LOAD",
        "schema_sha256": schema_sha,
        "cli_sha256": "48566cf6e2969464b799dbcac7393b3549f9efd6884688074dc803125dbafa85",
        "cli_version": "10068 (571d0d540)",
        "sampler_initialization": "NOT_VERIFIED",
    }
    valid_control = grammar.get("valid_schema_control", {})
    malformed_control = grammar.get("malformed_schema_negative_control", {})
    cli = grammar.get("llama_cli", {})

    if set(lock) != set(EXPECTED_LOCK_FIELDS) | EXTRA_LOCK_FIELDS:
        raise CandidateSmokeError("run_lock_shape_mismatch")
    if any(
        _canonical_json(lock.get(key)) != _canonical_json(value)
        for key, value in EXPECTED_LOCK_FIELDS.items()
    ):
        raise CandidateSmokeError("run_lock_mismatch")
    if sys.version.split()[0] != EXPECTED_LOCK_FIELDS["python_version"]:
        raise CandidateSmokeError("python_runtime_version_mismatch")
    if package_version("jsonschema") != EXPECTED_LOCK_FIELDS["jsonschema_version"]:
        raise CandidateSmokeError("jsonschema_runtime_version_mismatch")
    if (
        lock.get("runner_sha256") != _sha256(Path(__file__).resolve().read_bytes())
        or lock.get("dependencies")
        != {
            relative: _sha256((ROOT / relative).read_bytes())
            for relative in EXPECTED_DEPENDENCIES
        }
        or lock.get("request_sha256") != _sha256(body)
        or lock.get("schema_sha256") != schema_sha
        or lock.get("grammar_preflight") != grammar_lock
        or not grammar_sidecar
        or grammar_sidecar[0].lower() != grammar_sha
        or grammar.get("status") != grammar_lock["status"]
        or grammar.get("case_id") != CASE_ID
        or grammar.get("schema_sha256") != schema_sha
        or grammar.get("grammar_conversion") != "VERIFIED_PRE_MODEL_LOAD"
        or grammar.get("sampler_initialization") != "NOT_VERIFIED"
        or grammar.get("inference_performed") is not False
        or grammar.get("model_loaded") is not False
        or grammar.get("local_endpoint_called") is not False
        or grammar.get("hosted_api_called") is not False
        or cli.get("sha256") != grammar_lock["cli_sha256"]
        or cli.get("version") != grammar_lock["cli_version"]
        or valid_control.get("status") != "PASS"
        or valid_control.get("missing_model_failure_detected") is not True
        or valid_control.get("schema_or_grammar_error_detected") is not False
        or malformed_control.get("status") != "PASS"
        or malformed_control.get("missing_model_failure_detected") is not False
        or malformed_control.get("schema_or_grammar_error_detected") is not True
    ):
        raise CandidateSmokeError("run_lock_digest_mismatch")
    frozen_gate._load_frozen_inputs()
    return lock, lock_sha, case, body


class LoopbackOneShotGuard:
    """Allow only pinned runtime GETs and the one byte-identical local POST."""

    def __init__(self, expected_body: bytes) -> None:
        self.expected_body = expected_body
        self.baseline_process: dict[str, Any] | None = None
        self.events: list[dict[str, Any]] = []
        self.post_attempts = 0
        self._lock = threading.Lock()
        self._http_class = http.client.HTTPConnection
        self._https_class = http.client.HTTPSConnection
        self._original_http_request = self._http_class.request
        self._original_https_request = self._https_class.request
        self._installed = False
        self._request_wrapper = self._wrap_request()

    def _wrap_request(self):
        def guarded_request(
            connection: Any,
            method: str,
            url: str,
            body: Any = None,
            headers: Any = None,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            return self._request(
                connection, method, url, body, headers, *args, **kwargs
            )

        return guarded_request

    def _snapshot_and_validate(self, checkpoint: str) -> tuple[dict[str, Any], dict[str, Any]]:
        process = frozen_gate._ps_process_snapshot()
        service = frozen_gate._service_snapshot()
        validate_runtime_snapshot(process, service)
        if self.baseline_process is not None and process != self.baseline_process:
            raise CandidateSmokeError("server_process_changed_during_smoke")
        self.events.append(
            {
                "checkpoint": checkpoint,
                "status": "PASS",
                "process_sha256": _sha256(_canonical_json(process)),
                "service_raw_sha256": service.get("raw_sha256", {}),
            }
        )
        return process, service

    def arm_after_preflight(
        self, process: dict[str, Any], service: dict[str, Any]
    ) -> None:
        validate_runtime_snapshot(process, service)
        if self.baseline_process is not None:
            raise CandidateSmokeError("duplicate_preflight")
        self.baseline_process = dict(process)
        self.events.append(
            {
                "checkpoint": "initial_preflight",
                "status": "PASS",
                "process_sha256": _sha256(_canonical_json(process)),
                "service_raw_sha256": service.get("raw_sha256", {}),
            }
        )

    def _request(
        self,
        connection: Any,
        method: str,
        url: str,
        body: Any = None,
        headers: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if (
            getattr(connection, "host", None) != frozen_gate.HOST
            or getattr(connection, "port", None) != frozen_gate.PORT
        ):
            raise CandidateSmokeError("connection_not_frozen_loopback")
        if not isinstance(method, str):
            raise CandidateSmokeError("http_method_invalid")
        method = method.upper()
        if method == "GET":
            if (
                url not in ALLOWED_GET_PATHS
                or body is not None
                or not isinstance(headers, Mapping)
                or {
                    str(key).casefold(): str(value).casefold()
                    for key, value in headers.items()
                }
                != {"accept": "application/json"}
            ):
                raise CandidateSmokeError("get_path_not_allowlisted")
            return self._original_http_request(
                connection, method, url, body, headers, *args, **kwargs
            )
        if method != "POST" or url != frozen_gate.ENDPOINT_PATH:
            raise CandidateSmokeError("http_method_or_path_not_allowlisted")
        if body != self.expected_body:
            raise CandidateSmokeError("completion_body_not_exact_locked_request")
        if not isinstance(headers, Mapping) or {
            str(key).casefold(): str(value).casefold()
            for key, value in headers.items()
        } != {"content-type": "application/json", "accept": "application/json"}:
            raise CandidateSmokeError("completion_headers_not_exact")

        with self._lock:
            if self.post_attempts != 0:
                raise CandidateSmokeError("completion_post_limit_exceeded")
            if self.baseline_process is None:
                raise CandidateSmokeError("post_before_preflight")
            self._snapshot_and_validate("immediately_before_post")
            self.post_attempts = 1
        return self._original_http_request(
            connection, method, url, body, headers, *args, **kwargs
        )

    @staticmethod
    def _deny_https(*_args: Any, **_kwargs: Any) -> None:
        raise CandidateSmokeError("https_outbound_forbidden")

    def install(self) -> None:
        if self._installed:
            raise CandidateSmokeError("request_guard_already_installed")
        self._http_class.request = self._request_wrapper
        self._https_class.request = self._deny_https
        self._installed = True

    def restore(self) -> None:
        if self._installed:
            self._http_class.request = self._original_http_request
            self._https_class.request = self._original_https_request
            self._installed = False

    def __enter__(self) -> LoopbackOneShotGuard:
        self.install()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.restore()

    def verify_postflight(self) -> None:
        if self.post_attempts != 1:
            raise CandidateSmokeError("completion_post_count_mismatch")
        self._snapshot_and_validate("postflight")


def _response_content(raw: bytes) -> tuple[dict[str, Any], str]:
    try:
        envelope = _strict_json_loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CandidateSmokeError("response_envelope_unavailable") from exc
    if not isinstance(envelope, dict):
        raise CandidateSmokeError("response_envelope_not_object")
    choices = envelope.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise CandidateSmokeError("response_content_unavailable")
    message = choices[0].get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise CandidateSmokeError("response_content_unavailable")
    return envelope, message["content"]


def _run_once() -> dict[str, Any]:
    lock, lock_sha, case, body = _load_and_validate_lock()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    request_sha = _sha256(body)
    _write_exclusive(
        OUTPUT_DIR / "reservation.json",
        _canonical_json(
            {
                "status": "RESERVED_NO_RETRY",
                "run_id": lock["run_id"],
                "case_id": CASE_ID,
                "lock_sha256": lock_sha,
                "request_sha256": request_sha,
            }
        ),
    )
    _write_exclusive(OUTPUT_DIR / "request.json", body)

    result: dict[str, Any] = {
        "run_id": lock["run_id"],
        "case_id": CASE_ID,
        "lock_sha256": lock_sha,
        "request_sha256": request_sha,
        "completion_post_attempts": 0,
        "retry_count": 0,
        "hosted_calls": 0,
        "memory_store_mutations": 0,
        "sampler_initialization_evidence": "NOT_INFERRED_FROM_OFFLINE_GRAMMAR_PREFLIGHT",
    }
    guard = LoopbackOneShotGuard(body)
    response_body: bytes | None = None
    response_status: int | None = None
    try:
        with guard:
            process = frozen_gate._ps_process_snapshot()
            service = frozen_gate._service_snapshot()
            guard.arm_after_preflight(process, service)

            connection = http.client.HTTPConnection(
                frozen_gate.HOST, frozen_gate.PORT, timeout=300
            )
            try:
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
                result["response_content_type"] = response.getheader("Content-Type")
                response_body = response.read()
            finally:
                connection.close()
            guard.verify_postflight()

        result["completion_post_attempts"] = guard.post_attempts
        result["runtime_checkpoints"] = guard.events
        result["http_status"] = response_status
        if response_body is not None:
            _write_exclusive(OUTPUT_DIR / "response_body.bin", response_body)
        if (
            response_status != 200
            or response_body is None
            or not str(result.get("response_content_type", ""))
            .casefold()
            .startswith("application/json")
        ):
            raise CandidateSmokeError("completion_http_failure")
        envelope, content = _response_content(response_body)
        result["response_envelope_sha256"] = _sha256(_canonical_json(envelope))
        result["model_content_sha256"] = _sha256(content.encode("utf-8"))
        try:
            proposal = _strict_json_loads(content)
            Draft202012Validator(
                request_builder.build_candidate_request(
                    case, model=frozen_gate.MODEL_PATH
                )["response_format"]["json_schema"]["schema"]
            ).validate(proposal)
            normalized = guard_v3.validate_joint_bound_candidate_proposal(
                content,
                {"source_id": case["case_id"], "proposition_text": case["proposition_text"]},
                scope_id=frozen_r4.FROZEN_SCOPE_ID,
            )
        except Exception as exc:
            result.update(
                {
                    "status": "QUALITY_FAILURE",
                    "failure": f"{type(exc).__name__}:{exc}",
                }
            )
        else:
            _write_exclusive(
                OUTPUT_DIR / "normalized_proposal.json", _canonical_json(normalized)
            )
            exact = request_builder.exact_expected_atom_multiset_match(case, normalized)
            result["exact_expected_atom_multiset_match"] = exact
            result["validated_atom_count"] = len(normalized["atoms"])
            result["status"] = "QUALITY_PASS" if exact else "QUALITY_FAILURE"
            if not exact:
                result["failure"] = "normalized_atom_multiset_mismatch"
    except Exception as exc:
        result["completion_post_attempts"] = guard.post_attempts
        result["runtime_checkpoints"] = guard.events
        result["status"] = "INFRA_FAILURE"
        result["failure"] = f"{type(exc).__name__}:{exc}"
        if response_body is not None and not (OUTPUT_DIR / "response_body.bin").exists():
            _write_exclusive(OUTPUT_DIR / "response_body.bin", response_body)
    _write_exclusive(OUTPUT_DIR / "result.json", _canonical_json(result))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--run-r4c-05-once",
        action="store_true",
        help="Consume the frozen one-case local smoke authorization exactly once.",
    )
    args = parser.parse_args()
    if not args.run_r4c_05_once:
        parser.error("explicit --run-r4c-05-once is required")
    result = _run_once()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if result.get("status") == "QUALITY_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
