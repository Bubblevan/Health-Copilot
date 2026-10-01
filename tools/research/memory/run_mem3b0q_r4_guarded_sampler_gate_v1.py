"""Hash-pinned one-shot strict overlay for the frozen R4-P01 gate."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import threading
from pathlib import Path
from typing import Any
from unittest import mock

from tools.research.memory import mem3b0q_r4 as frozen_r4
from tools.research.memory import run_mem3b0q_r4_gate as frozen_gate
from tools.research.memory.preflight_mem3b0q_r4_runtime_readonly_v1 import (
    RuntimePreflightError,
    validate_runtime_snapshot,
)


OVERLAY_LOCK_PATH = frozen_gate.MEMORY_DOCS / "mem3b0q_r4_one_shot_overlay_lock_v1.json"
OVERLAY_LOCK_SIDECAR_PATH = frozen_gate.MEMORY_DOCS / "mem3b0q_r4_one_shot_overlay_lock_v1.sha256"
ALLOWED_GET_PATHS = frozenset({"/health", "/props", "/v1/models", "/slots"})


class OverlayGuardError(RuntimeError):
    """A request did not satisfy the exact one-shot R4-P01 boundary."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_overlay_lock() -> tuple[dict[str, Any], str]:
    lock_bytes = OVERLAY_LOCK_PATH.read_bytes()
    lock_sha256 = _sha256(lock_bytes)
    sidecar_tokens = OVERLAY_LOCK_SIDECAR_PATH.read_text(encoding="ascii").split()
    if not sidecar_tokens or sidecar_tokens[0].lower() != lock_sha256:
        raise OverlayGuardError("overlay_lock_sidecar_mismatch")
    lock = json.loads(lock_bytes.decode("utf-8"))
    overlay_sha256 = _sha256(Path(__file__).resolve().read_bytes())
    if (
        lock.get("status") != "FROZEN"
        or lock.get("overlay_sha256") != overlay_sha256
        or lock.get("runner_sha256") != frozen_gate._file_sha256(Path(frozen_gate.__file__))
        or lock.get("manifest_sha256") != frozen_gate.FROZEN_MANIFEST_SHA256
        or lock.get("runtime_amendment_sha256")
        != frozen_gate._file_sha256(frozen_gate.RUNTIME_PREFLIGHT_AMENDMENT_PATH)
        or lock.get("case_id") != "R4-P01"
        or lock.get("max_completion_posts") != 1
    ):
        raise OverlayGuardError("overlay_lock_mismatch")
    return lock, lock_sha256


def _expected_p01_request() -> tuple[dict[str, Any], bytes]:
    _, pack, schema, _ = frozen_gate._load_frozen_inputs()
    proposition = pack["propositions"][0]
    if proposition.get("source_id") != "R4-P01":
        raise OverlayGuardError("first_frozen_case_mismatch")
    payload = frozen_r4.build_request_payload(
        proposition, schema, model=frozen_gate.MODEL_PATH
    )
    if (
        payload.get("response_format", {}).get("type") != "json_schema"
        or frozen_r4.canonical_json_sha256(
            payload["response_format"]["json_schema"]["schema"]
        )
        != frozen_gate.SCHEMA_CANONICAL_SHA256
    ):
        raise OverlayGuardError("frozen_request_schema_mismatch")
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return payload, body


class GuardedSamplerGateOverlay:
    """Temporarily harden runner checkpoints and guard its only POST boundary."""

    def __init__(self, expected_body: bytes) -> None:
        self.expected_body = expected_body
        self.baseline_process: dict[str, Any] | None = None
        self.events: list[dict[str, Any]] = []
        self.post_attempts = 0
        self._request_lock = threading.Lock()
        self._original_validator = frozen_gate._validate_preflight
        self._connection_class = http.client.HTTPConnection
        self._original_request = self._connection_class.request
        self._installed = False

        def guarded_class_request(
            connection: Any,
            method: str,
            url: str,
            body: Any = None,
            headers: Any = None,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            return self._guard_request(
                connection, method, url, body, headers, *args, **kwargs
            )

        self._request_wrapper = guarded_class_request

    def _validate_checkpoint(
        self,
        process: dict[str, Any],
        service: dict[str, Any],
        checkpoint: str,
    ) -> None:
        try:
            with mock.patch.object(
                frozen_gate, "_validate_preflight", self._original_validator
            ):
                validate_runtime_snapshot(process, service)
            if checkpoint == "runner_preflight":
                if self.baseline_process is not None:
                    raise OverlayGuardError("duplicate_runner_preflight")
                self.baseline_process = dict(process)
            elif self.baseline_process is None:
                raise OverlayGuardError("runner_preflight_not_seen")
            elif dict(process) != self.baseline_process:
                raise OverlayGuardError("server_process_changed_during_gate")
        except Exception as exc:
            self.events.append(
                {
                    "checkpoint": checkpoint,
                    "status": "FAIL",
                    "error": f"{type(exc).__name__}:{exc}",
                }
            )
            raise
        self.events.append(
            {
                "checkpoint": checkpoint,
                "status": "PASS",
                "process_sha256": _sha256(
                    json.dumps(process, sort_keys=True, separators=(",", ":")).encode()
                ),
                "service_raw_sha256": service.get("raw_sha256", {}),
            }
        )

    def _strict_runner_validator(
        self, process: dict[str, Any], service: dict[str, Any]
    ) -> None:
        checkpoint = (
            "runner_preflight" if self.baseline_process is None else "runner_postflight"
        )
        self._validate_checkpoint(process, service, checkpoint)

    def _guard_request(
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
            raise OverlayGuardError("connection_not_frozen_loopback")
        if not isinstance(method, str):
            raise OverlayGuardError("http_method_invalid")
        normalized_method = method.upper()
        if normalized_method == "GET":
            if url not in ALLOWED_GET_PATHS:
                raise OverlayGuardError("get_path_not_allowlisted")
            return self._original_request(
                connection, method, url, body, headers, *args, **kwargs
            )
        if normalized_method != "POST" or url != frozen_gate.ENDPOINT_PATH:
            raise OverlayGuardError("http_method_or_path_not_allowlisted")
        if body != self.expected_body:
            raise OverlayGuardError("completion_body_not_exact_frozen_p01")

        with self._request_lock:
            if self.post_attempts != 0:
                raise OverlayGuardError("completion_post_limit_exceeded")
            if self.baseline_process is None:
                raise OverlayGuardError("post_before_runner_preflight")
            process = frozen_gate._ps_process_snapshot()
            service = frozen_gate._service_snapshot()
            self._validate_checkpoint(process, service, "immediately_before_post")
            self.post_attempts = 1

        return self._original_request(
            connection, method, url, body, headers, *args, **kwargs
        )

    def install(self) -> None:
        if self._installed:
            raise OverlayGuardError("overlay_already_installed")
        frozen_gate._validate_preflight = self._strict_runner_validator
        self._connection_class.request = self._request_wrapper
        self._installed = True

    def restore(self) -> None:
        if not self._installed:
            return
        frozen_gate._validate_preflight = self._original_validator
        self._connection_class.request = self._original_request
        self._installed = False

    def __enter__(self) -> GuardedSamplerGateOverlay:
        self.install()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.restore()

    def evidence(self) -> dict[str, Any]:
        return {
            "strict_overlay_sha256": _sha256(Path(__file__).resolve().read_bytes()),
            "expected_request_sha256": _sha256(self.expected_body),
            "completion_post_attempts_forwarded": self.post_attempts,
            "max_completion_posts": 1,
            "checkpoints": list(self.events),
        }


def run_guarded_sampler_gate() -> dict[str, Any]:
    if frozen_gate.GATE_DIR.exists():
        raise OverlayGuardError("gate_attempt_directory_exists_refusing_retry")
    lock, lock_sha256 = _load_overlay_lock()
    _, expected_body = _expected_p01_request()
    overlay = GuardedSamplerGateOverlay(expected_body)
    with overlay:
        result = frozen_gate.run_gate()
    evidence = overlay.evidence()
    evidence.update(
        {
            "overlay_lock_sha256": lock_sha256,
            "frozen_runner_sha256": lock["runner_sha256"],
            "frozen_manifest_sha256": lock["manifest_sha256"],
            "result_status": result.get("status"),
            "sampler_gate": result.get("sampler_gate"),
            "inference_scope": "R4-P01_PASS1_SAMPLER_INIT_GATE_ONLY",
            "retry_count": result.get("retry_count", 0),
            "b1_status": "MEM3B0Q_MEM3B1_READY=NO",
        }
    )
    if frozen_gate.GATE_DIR.exists():
        frozen_gate._write_json(
            frozen_gate.GATE_DIR / "strict_overlay_manifest.json", evidence
        )
    return {"gate_result": result, "overlay_evidence": evidence}


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--run-p01-sampler-init-gate",
        action="store_true",
        help="Issue the one pre-registered local R4-P01 sampler-init request.",
    )
    args = parser.parse_args()
    if not args.run_p01_sampler_init_gate:
        parser.error("explicit --run-p01-sampler-init-gate is required")
    print(
        json.dumps(
            run_guarded_sampler_gate(), ensure_ascii=False, sort_keys=True, indent=2
        )
    )


if __name__ == "__main__":
    main()
