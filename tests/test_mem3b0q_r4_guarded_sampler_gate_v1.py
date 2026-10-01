from __future__ import annotations

import copy
import hashlib
import http.client
from pathlib import Path
import sys

import pytest

from tools.research.memory import run_mem3b0q_r4_gate as frozen_gate
from tools.research.memory import run_mem3b0q_r4_guarded_sampler_gate_v1 as overlay_module
from tools.research.memory.run_mem3b0q_r4_guarded_sampler_gate_v1 import (
    GuardedSamplerGateOverlay,
    OverlayGuardError,
)
from tools.research.memory.preflight_mem3b0q_r4_runtime_readonly_v1 import (
    RuntimePreflightError,
)


def _process() -> dict:
    return {
        "pid": 100,
        "addresses": ["127.0.0.1"],
        "executable_path": frozen_gate.SERVER_PATH,
        "executable_sha256": frozen_gate.SERVER_SHA256,
        "model_path": frozen_gate.MODEL_PATH,
        "model_sha256": frozen_gate.MODEL_SHA256,
        "command_line": "pinned command line",
    }


def _service(*, top_p: object = 0.95, processing: object = False) -> dict:
    return {
        "health": {"status": "ok"},
        "props": {
            "build_info": "b10068-571d0d540",
            "model_path": frozen_gate.MODEL_PATH,
            "model_alias": frozen_gate.MODEL_PATH,
            "total_slots": 1,
            "default_generation_settings": {
                "n_ctx": 131072,
                "params": {
                    "top_k": 40,
                    "top_p": top_p,
                    "min_p": 0.05,
                    "repeat_penalty": 1.0,
                },
            },
        },
        "models": {"data": [{"id": frozen_gate.MODEL_PATH}]},
        "slots": [{"n_ctx": 131072, "is_processing": processing}],
        "raw_sha256": {"/props": "a" * 64, "/slots": "b" * 64},
    }


def _overlay(monkeypatch, *, post_service: dict | None = None):
    sent = []

    class FakeConnection:
        def __init__(self, host, port, timeout=None):
            self.host = host
            self.port = port
            self.timeout = timeout

        def request(self, method, url, body=None, headers=None, *args, **kwargs):
            sent.append((method, url, body, headers))

    monkeypatch.setattr(http.client, "HTTPConnection", FakeConnection)
    base_validation_calls = []
    monkeypatch.setattr(
        frozen_gate,
        "_validate_preflight",
        lambda process, service: base_validation_calls.append((process, service)),
    )
    monkeypatch.setattr(frozen_gate, "_ps_process_snapshot", _process)
    monkeypatch.setattr(
        frozen_gate,
        "_service_snapshot",
        lambda: copy.deepcopy(post_service or _service()),
    )
    overlay = GuardedSamplerGateOverlay(b'{"frozen":"P01"}')
    return overlay, sent, base_validation_calls


def _simulate_runner(
    body: bytes,
    *,
    postflight_service: dict | None = None,
    second_post: bool = False,
) -> dict:
    try:
        frozen_gate._validate_preflight(_process(), _service())
        connection = http.client.HTTPConnection(frozen_gate.HOST, frozen_gate.PORT)
        connection.request(
            "POST",
            frozen_gate.ENDPOINT_PATH,
            body=body,
            headers={"Content-Type": "application/json"},
        )
        if second_post:
            connection.request(
                "POST",
                frozen_gate.ENDPOINT_PATH,
                body=body,
                headers={"Content-Type": "application/json"},
            )
        frozen_gate._validate_preflight(
            _process(), postflight_service or _service()
        )
    except (OverlayGuardError, RuntimePreflightError) as exc:
        return {"status": "INFRA_FAILURE", "failure": str(exc)}
    return {"status": "PENDING_SERVER_LOG_REVIEW"}


def test_overlay_enforces_strict_pre_and_postflight_and_one_post(monkeypatch) -> None:
    overlay, sent, base_calls = _overlay(monkeypatch)
    with overlay:
        result = _simulate_runner(overlay.expected_body)

    assert result["status"] == "PENDING_SERVER_LOG_REVIEW"
    assert [call[0] for call in sent] == ["POST"]
    assert overlay.post_attempts == 1
    assert [event["checkpoint"] for event in overlay.events] == [
        "runner_preflight",
        "immediately_before_post",
        "runner_postflight",
    ]
    assert all(event["status"] == "PASS" for event in overlay.events)
    assert len(base_calls) == 3


def test_bad_fresh_pre_post_snapshot_blocks_network_post(monkeypatch) -> None:
    overlay, sent, _ = _overlay(monkeypatch, post_service=_service(top_p=float("nan")))
    with overlay:
        result = _simulate_runner(overlay.expected_body)

    assert result["status"] == "INFRA_FAILURE"
    assert "top_p_not_finite_number" in result["failure"]
    assert sent == []
    assert overlay.post_attempts == 0
    assert overlay.events[-1]["status"] == "FAIL"


def test_bad_postflight_snapshot_fails_after_exactly_one_attempt(monkeypatch) -> None:
    overlay, sent, _ = _overlay(monkeypatch)
    with overlay:
        result = _simulate_runner(
            overlay.expected_body, postflight_service=_service(processing=None)
        )

    assert result["status"] == "INFRA_FAILURE"
    assert "slot_processing_state_invalid" in result["failure"]
    assert len(sent) == 1
    assert overlay.post_attempts == 1
    assert overlay.events[-1]["checkpoint"] == "runner_postflight"
    assert overlay.events[-1]["status"] == "FAIL"


def test_second_completion_post_is_blocked_before_forwarding(monkeypatch) -> None:
    overlay, sent, _ = _overlay(monkeypatch)
    with overlay:
        result = _simulate_runner(overlay.expected_body, second_post=True)

    assert result["status"] == "INFRA_FAILURE"
    assert "completion_post_limit_exceeded" in result["failure"]
    assert len(sent) == 1
    assert overlay.post_attempts == 1


def test_changed_body_path_or_connection_is_blocked(monkeypatch) -> None:
    overlay, sent, _ = _overlay(monkeypatch)
    with overlay:
        connection = http.client.HTTPConnection(frozen_gate.HOST, frozen_gate.PORT)
        with pytest.raises(OverlayGuardError, match="completion_body_not_exact"):
            connection.request("POST", frozen_gate.ENDPOINT_PATH, body=b"different")
        with pytest.raises(OverlayGuardError, match="http_method_or_path"):
            connection.request("POST", "/other", body=overlay.expected_body)
        connection.host = "example.com"
        with pytest.raises(OverlayGuardError, match="connection_not_frozen_loopback"):
            connection.request("POST", frozen_gate.ENDPOINT_PATH, body=overlay.expected_body)

    assert sent == []
    assert overlay.post_attempts == 0


def test_overlay_leaves_frozen_runner_and_manifest_bytes_unchanged(monkeypatch) -> None:
    paths = [
        frozen_gate.MANIFEST_PATH,
        frozen_gate.MANIFEST_SIDECAR_PATH,
        Path(frozen_gate.__file__),
        frozen_gate.RUNTIME_PREFLIGHT_AMENDMENT_PATH,
        frozen_gate.RUNNER_LOCK_PATH,
    ]
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    overlay, _, _ = _overlay(monkeypatch)
    with overlay:
        frozen_gate._validate_preflight(_process(), _service())
    after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}

    assert after == before


def test_overlay_evidence_records_expected_single_p01_post() -> None:
    overlay = GuardedSamplerGateOverlay(b'{"frozen":"P01"}')

    evidence = overlay.evidence()

    assert evidence["max_completion_posts"] == 1
    assert evidence["completion_post_attempts_forwarded"] == 0
    assert evidence["expected_request_sha256"] == hashlib.sha256(
        overlay.expected_body
    ).hexdigest()


@pytest.mark.parametrize(
    "argv",
    [[], ["--run-p01-sampler-init-gat"]],
    ids=["no-arguments", "abbreviated-flag"],
)
def test_cli_rejects_without_exact_flag_and_never_invokes_runner(
    monkeypatch, argv: list[str]
) -> None:
    invoked = []
    monkeypatch.setattr(overlay_module, "run_guarded_sampler_gate", lambda: invoked.append(True))
    monkeypatch.setattr(sys, "argv", ["guarded-r4", *argv])

    with pytest.raises(SystemExit):
        overlay_module.main()

    assert invoked == []
