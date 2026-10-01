from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tools.research.memory import run_mem3b0q_r4_gate as gate


@pytest.mark.parametrize(
    ("response_body", "expected_quality"),
    [
        (
            json.dumps({"choices": [{"message": {"content": "not-json"}}]}).encode(),
            "QUALITY_FAILURE",
        ),
        (b"not-json", None),
    ],
)
def test_completion_failures_are_classified_without_retry(
    monkeypatch,
    tmp_path: Path,
    response_body: bytes,
    expected_quality: str | None,
) -> None:
    gate_dir = tmp_path / "gate"
    log_path = tmp_path / "llama-server.log"
    runner_lock_path = tmp_path / "runner-lock.json"
    log_path.write_text("server ready\n", encoding="utf-8")
    monkeypatch.setattr(gate, "GATE_DIR", gate_dir)
    monkeypatch.setattr(gate, "LOG_PATH", log_path)
    monkeypatch.setattr(gate, "RUNNER_LOCK_PATH", runner_lock_path)
    runner_lock_path.write_text("{}\n", encoding="utf-8")

    argv = [
        gate.SERVER_PATH,
        "-m",
        gate.MODEL_PATH,
        "--host",
        "127.0.0.1",
        "--port",
        "8081",
        "--ctx-size",
        "131072",
        "--n-predict",
        "8192",
        "--rope-scaling",
        "yarn",
        "--rope-scale",
        "4",
        "--yarn-orig-ctx",
        "32768",
        "--override-kv",
        "qwen3.context_length=int:131072",
        "--cache-type-k",
        "q4_0",
        "--cache-type-v",
        "q4_0",
        "--n-gpu-layers",
        "99",
        "--flash-attn",
        "on",
        "--parallel",
        "1",
        "--log-file",
        str(log_path),
        "--verbosity",
        "5",
    ]
    process = {
        "addresses": [gate.HOST],
        "executable_sha256": gate.SERVER_SHA256,
        "model_sha256": gate.MODEL_SHA256,
        "model_path": gate.MODEL_PATH,
        "command_line": subprocess.list2cmdline(argv),
    }
    gate._validate_server_command_line(process["command_line"])
    wrong_gpu_count = argv.copy()
    wrong_gpu_count[wrong_gpu_count.index("--n-gpu-layers") + 1] = "990"
    with pytest.raises(RuntimeError, match="server_command_line_mismatch"):
        gate._validate_server_command_line(subprocess.list2cmdline(wrong_gpu_count))
    duplicate_option = argv + ["--parallel", "1"]
    with pytest.raises(RuntimeError, match="server_command_line_duplicate_argument"):
        gate._validate_server_command_line(subprocess.list2cmdline(duplicate_option))

    service = {
        "health": {"status": "ok"},
        "props": {
            "build_info": "b10068-571d0d540",
            "model_path": gate.MODEL_PATH,
            "model_alias": gate.MODEL_PATH,
            "default_generation_settings": {
                "n_ctx": 131072,
                "params": {
                    "top_k": 40,
                    "top_p": 0.95,
                    "min_p": 0.05,
                    "repeat_penalty": 1.0,
                },
            },
            "total_slots": 1,
        },
        "models": {"data": [{"id": gate.MODEL_PATH}]},
        "slots": [
            {
                "n_ctx": 131072,
                "is_processing": False,
            }
        ],
        "raw_sha256": {},
    }
    frozen_pack = json.loads(gate.PACK_PATH.read_text("utf-8"))
    frozen_schema = json.loads(gate.SCHEMA_PATH.read_text("utf-8"))
    monkeypatch.setattr(
        gate,
        "_load_frozen_inputs",
        lambda: (
            {},
            frozen_pack,
            frozen_schema,
            {"runner_sha256": "frozen-runner-hash"},
        ),
    )
    monkeypatch.setattr(gate, "_ps_process_snapshot", lambda: process)
    monkeypatch.setattr(gate, "_service_snapshot", lambda: service)

    def observe_slots(
        _stop_event,
        source_id,
        proposition_text,
        observations,
        _failures,
    ) -> None:
        observations.append(
            {
                "slots": [
                    {
                        "slot_id": 0,
                        "task_id": 42,
                        "is_processing": True,
                        "prompt_contains_source_id": source_id == "R4-P01",
                        "prompt_contains_proposition": proposition_text
                        == frozen_pack["propositions"][0]["proposition_text"],
                    }
                ]
            }
        )

    monkeypatch.setattr(gate, "_observe_active_slots", observe_slots)

    class Response:
        status = 200

        @staticmethod
        def read() -> bytes:
            return response_body

    class Connection:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def request(self, *_args, **_kwargs) -> None:
            with log_path.open("a", encoding="utf-8") as log:
                log.write(
                    "slot launch_slot_: id 0 | task -1 | launching slot : {}\n"
                    "common_sampler_init: grammar accepted prefill token (1)\n"
                    "slot launch_slot_: id 0 | task 42 | processing task, is_child = 0\n"
                    "slot init_sampler: id 0 | task 42 | init sampler, took 0.1 ms\n"
                )

        @staticmethod
        def getresponse() -> Response:
            return Response()

        @staticmethod
        def close() -> None:
            pass

    monkeypatch.setattr(gate.http.client, "HTTPConnection", Connection)
    result = gate.run_gate()

    effective_status = (
        "QUALITY_FAILURE" if expected_quality else "INFRA_FAILURE"
    )
    assert result["status"] == effective_status
    if expected_quality:
        assert result["sampler_gate"] == "QUALITY_FAILURE_RESPONSE_CONTRACT"
    else:
        assert result["sampler_gate"] == "INFRA_FAILURE_REQUEST_COMPLETION"
    assert result["request_attempts"] == 1
    assert result["retry_count"] == 0
    assert result["hosted_calls"] == 0
    assert result["memory_store_mutations"] == 0
    assert result["revision_edges"] == 0
    assert (gate_dir / "request_body.json").exists()
    assert (gate_dir / "response_body.json").read_bytes() == response_body
    if expected_quality:
        assert result["quality_status"] == expected_quality
        assert result["structural_validation"] == "FAIL"
    else:
        assert "quality_status" not in result
        assert result["failure"].startswith("invalid_http_response_envelope:")
    assert result["server_log_attribution"] == "REQUEST_TASK_CORRELATED"
    run_manifest = json.loads((gate_dir / "gate_run_manifest.json").read_text("utf-8"))
    assert run_manifest["sampler_parser"] == "VERIFIED"


def test_missing_p01_active_prompt_witness_cannot_promote_gate() -> None:
    log = """\
slot launch_slot_: id 0 | task -1 | launching slot : {}
common_sampler_init: grammar accepted prefill token (1)
slot launch_slot_: id 0 | task 42 | processing task, is_child = 0
slot init_sampler: id 0 | task 42 | init sampler, took 0.1 ms
"""
    uncorrelated_observation = {
        "slots": [
            {
                "slot_id": 0,
                "task_id": 42,
                "is_processing": True,
                "prompt_contains_source_id": False,
                "prompt_contains_proposition": True,
            }
        ]
    }
    result = gate._evaluate_sampler_evidence(
        log,
        [uncorrelated_observation],
        source_id="R4-P01",
        proposition_text="My workout plan activity is running.",
        observer_failures=[],
    )
    assert result["status"] == "UNVERIFIED"
    assert result["input_witnessed_in_active_prompt"] is False
    attempt = {
        "status": "PENDING_SERVER_LOG_REVIEW",
        "http_status": 200,
        "http_envelope_validation": "PASS",
        "structural_validation": "PASS",
    }
    gate._apply_sampler_evidence(attempt, result)
    assert attempt["sampler_parser_after"] == "NOT_VERIFIED"
    assert attempt["sampler_gate"] != "PASS"
    assert attempt["status"] == "INFRA_FAILURE"


def test_only_prefill_inside_unique_launch_processing_interval_is_correlated() -> None:
    log = """\
common_sampler_init: grammar accepted prefill token (999)
slot launch_slot_: id 0 | task -1 | launching slot : {}
common_sampler_init: grammar accepted prefill token (1)
slot launch_slot_: id 0 | task 42 | processing task, is_child = 0
slot init_sampler: id 0 | task 42 | init sampler, took 0.1 ms
"""
    observed_prompt = {
        "slots": [
            {
                "slot_id": 0,
                "task_id": 42,
                "is_processing": True,
                "prompt_contains_source_id": True,
                "prompt_contains_proposition": True,
            }
        ]
    }
    evidence = gate._evaluate_sampler_evidence(
        log,
        [observed_prompt],
        source_id="R4-P01",
        proposition_text="My workout plan activity is running.",
        observer_failures=[],
    )
    assert evidence["grammar_prefill_correlated_line_count"] == 1
    assert evidence["grammar_prefill_uncorrelated_line_indices"] == [0]
    assert evidence["status"] == "UNVERIFIED"

    attempt = {
        "status": "PENDING_SERVER_LOG_REVIEW",
        "http_status": 200,
        "http_envelope_validation": "PASS",
        "structural_validation": "PASS",
    }
    gate._apply_sampler_evidence(attempt, evidence)
    assert attempt["sampler_parser_after"] == "NOT_VERIFIED"
    assert attempt["sampler_gate"] != "PASS"
    assert attempt["status"] == "INFRA_FAILURE"


def test_source_ordered_sampler_trace_passes_with_one_correlated_prefill() -> None:
    log = """\
slot launch_slot_: id 0 | task -1 | launching slot : {}
common_sampler_init: grammar accepted prefill token (1)
slot launch_slot_: id 0 | task 42 | processing task, is_child = 0
slot init_sampler: id 0 | task 42 | init sampler, took 0.1 ms
"""
    observed_prompt = {
        "slots": [
            {
                "slot_id": 0,
                "task_id": 42,
                "is_processing": True,
                "prompt_contains_source_id": True,
                "prompt_contains_proposition": True,
            }
        ]
    }
    evidence = gate._evaluate_sampler_evidence(
        log,
        [observed_prompt],
        source_id="R4-P01",
        proposition_text="My workout plan activity is running.",
        observer_failures=[],
    )
    assert evidence["status"] == "PASS"
    assert evidence["launch_rows"] == [[0, 0, -1]]
    assert evidence["launch_processing_interval"] == [0, 2]
    assert evidence["grammar_prefill_correlated_line_count"] == 1
    assert evidence["grammar_prefill_uncorrelated_line_indices"] == []


@pytest.mark.parametrize(
    "log",
    [
        """\
slot launch_slot_: id 0 | task -1 | launching slot : {}
common_sampler_init: grammar accepted prefill token (1)
slot launch_slot_: id 0 | task 42 | processing task, is_child = 0
slot init_sampler: id 0 | task 42 | init sampler, took 0.1 ms
slot launch_slot_: id 0 | task -1 | launching slot : {}
""",
        """\
slot launch_slot_: id 0 | task -1 | launching slot : {}
common_sampler_init: grammar accepted prefill token (1)
slot launch_slot_: id 0 | task 42 | processing task, is_child = 0
slot init_sampler: id 0 | task 42 | init sampler, took 0.1 ms
slot launch_slot_: id 0 | task 42 | processing task, is_child = 0
""",
        """\
slot launch_slot_: id 0 | task -1 | launching slot : {}
common_sampler_init: grammar accepted prefill token (1)
slot launch_slot_: id 0 | task 42 | processing task, is_child = 0
slot init_sampler: id 0 | task 42 | init sampler, took 0.1 ms
slot init_sampler: id 0 | task 42 | init sampler, took 0.1 ms
""",
        """\
slot launch_slot_: id 0 | task -1 | launching slot : {}
common_sampler_init: grammar accepted prefill token (1)
slot launch_slot_: id 0 | task 43 | processing task, is_child = 0
slot init_sampler: id 0 | task 42 | init sampler, took 0.1 ms
""",
    ],
)
def test_duplicate_or_task_mismatched_trace_fails_closed(log: str) -> None:
    observed_prompt = {
        "slots": [
            {
                "slot_id": 0,
                "task_id": 42,
                "is_processing": True,
                "prompt_contains_source_id": True,
                "prompt_contains_proposition": True,
            }
        ]
    }
    evidence = gate._evaluate_sampler_evidence(
        log,
        [observed_prompt],
        source_id="R4-P01",
        proposition_text="My workout plan activity is running.",
        observer_failures=[],
    )
    assert evidence["status"] == "UNVERIFIED"


def test_frozen_manifest_digest_is_pinned_before_artifact_map_is_trusted(
    monkeypatch, tmp_path: Path
) -> None:
    manifest_path = tmp_path / "freeze-manifest.json"
    sidecar_path = tmp_path / "freeze-manifest.sha256"
    manifest_path.write_text('{"artifacts":{}}\n', encoding="utf-8")
    pinned_hash = "0" * 64
    sidecar_path.write_text(f"{pinned_hash}  freeze-manifest.json\n", encoding="ascii")
    monkeypatch.setattr(gate, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(gate, "MANIFEST_SIDECAR_PATH", sidecar_path)
    monkeypatch.setattr(gate, "FROZEN_MANIFEST_SHA256", pinned_hash)

    with pytest.raises(RuntimeError, match="frozen_manifest_digest_mismatch"):
        gate._load_frozen_inputs()
