from __future__ import annotations

import hashlib
import http.client
import json
import sys

import pytest

from tools.research.memory import run_mem3b0q_r4_gate as frozen_gate
from tools.research.memory import run_mem3b0q_r4_p01_posthoc_repeat_v1 as repeat_module


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


def _service() -> dict:
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
                    "top_p": 0.95,
                    "min_p": 0.05,
                    "repeat_penalty": 1.0,
                },
            },
        },
        "models": {"data": [{"id": frozen_gate.MODEL_PATH}]},
        "slots": [{"n_ctx": 131072, "is_processing": False}],
        "raw_sha256": {"/props": "a" * 64, "/slots": "b" * 64},
    }


@pytest.mark.parametrize(
    "argv",
    [[], ["--run-p01-posthoc-repea"]],
    ids=["no-arguments", "abbreviated-flag"],
)
def test_cli_requires_exact_explicit_flag_without_invoking_repeat(
    monkeypatch, argv: list[str]
) -> None:
    invoked = []
    monkeypatch.setattr(repeat_module, "run_posthoc_repeat", lambda: invoked.append(True))
    monkeypatch.setattr(sys, "argv", ["posthoc-repeat", *argv])

    with pytest.raises(SystemExit):
        repeat_module.main()

    assert invoked == []


def test_posthoc_repeat_uses_separate_dir_one_post_and_restores_frozen_path(
    monkeypatch, tmp_path
) -> None:
    expected_body = b'{"frozen":"P01"}'
    historical_dir = tmp_path / "historical"
    historical_dir.mkdir()
    repeat_dir = tmp_path / "posthoc-repeat"
    sent = []

    class FakeConnection:
        def __init__(self, host, port, timeout=None):
            self.host = host
            self.port = port
            self.timeout = timeout

        def request(self, method, url, body=None, headers=None, *args, **kwargs):
            sent.append((method, url, body, headers))

    monkeypatch.setattr(http.client, "HTTPConnection", FakeConnection)
    monkeypatch.setattr(repeat_module, "REPEAT_DIR", repeat_dir)
    monkeypatch.setattr(frozen_gate, "GATE_DIR", historical_dir)
    monkeypatch.setattr(
        repeat_module,
        "_load_lock",
        lambda: (
            {
                "historical_gate_attempt_sha256": "1" * 64,
                "historical_request_sha256": "2" * 64,
                "historical_response_sha256": "3" * 64,
                "repeat_request_sha256": hashlib.sha256(expected_body).hexdigest(),
            },
            "4" * 64,
        ),
    )
    monkeypatch.setattr(
        repeat_module,
        "_sha256",
        lambda data: hashlib.sha256(data).hexdigest(),
    )
    monkeypatch.setattr(
        repeat_module.strict_overlay,
        "_expected_p01_request",
        lambda: ({"model": frozen_gate.MODEL_PATH}, expected_body),
    )
    monkeypatch.setattr(repeat_module.strict_overlay, "_load_overlay_lock", lambda: ({}, "6" * 64))
    monkeypatch.setattr(frozen_gate, "_validate_preflight", lambda process, service: None)
    monkeypatch.setattr(frozen_gate, "_ps_process_snapshot", _process)
    monkeypatch.setattr(frozen_gate, "_service_snapshot", _service)

    def fake_run_gate() -> dict:
        repeat_dir.mkdir()
        frozen_gate._validate_preflight(_process(), _service())
        connection = http.client.HTTPConnection(frozen_gate.HOST, frozen_gate.PORT)
        connection.request(
            "POST",
            frozen_gate.ENDPOINT_PATH,
            body=expected_body,
            headers={"Content-Type": "application/json"},
        )
        frozen_gate._validate_preflight(_process(), _service())
        return {
            "sampler_gate": "INFRA_FAILURE_UNCORRELATED_EVIDENCE",
            "quality_status": "QUALITY_FAILURE",
            "quality_failure": "diagnostic fixture",
        }

    monkeypatch.setattr(frozen_gate, "run_gate", fake_run_gate)

    result = repeat_module.run_posthoc_repeat()

    assert frozen_gate.GATE_DIR == historical_dir
    assert repeat_dir.is_dir()
    assert [row[0] for row in sent] == ["POST"]
    assert result["classification"] == "POSTHOC_EXPLORATORY_REPEAT_NOT_QUALIFICATION"
    assert result["overlay_evidence"]["completion_posts_forwarded"] == 1
    assert result["overlay_evidence"]["qualification_effect"] == "NONE_R4_V1_REMAINS_NO"
    manifest = json.loads(
        (repeat_dir / "posthoc_repeat_manifest.json").read_text(encoding="utf-8")
    )
    manifest_path = repeat_dir / "posthoc_repeat_manifest.json"
    sidecar_hash = (repeat_dir / "posthoc_repeat_manifest.sha256").read_text(
        encoding="ascii"
    ).split()[0]
    assert sidecar_hash == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert manifest["completion_posts_forwarded"] == 1
    assert manifest["sampler_attribution"] == "INFRA_FAILURE_UNCORRELATED_EVIDENCE"
    assert manifest["artifact_sha256"]["strict_overlay_manifest.json"] == hashlib.sha256(
        (repeat_dir / "strict_overlay_manifest.json").read_bytes()
    ).hexdigest()


def test_existing_posthoc_repeat_directory_refuses_before_lock_or_request(
    monkeypatch, tmp_path
) -> None:
    repeat_dir = tmp_path / "already-used"
    repeat_dir.mkdir()
    monkeypatch.setattr(repeat_module, "REPEAT_DIR", repeat_dir)
    monkeypatch.setattr(
        repeat_module,
        "_load_lock",
        lambda: pytest.fail("lock must not be loaded for a used repeat directory"),
    )

    with pytest.raises(repeat_module.PosthocRepeatError, match="posthoc_repeat_dir_exists"):
        repeat_module.run_posthoc_repeat()
