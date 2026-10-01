from __future__ import annotations

import copy

import pytest

from tools.research.memory import run_mem3b0q_r4_gate as frozen_gate
from tools.research.memory.preflight_mem3b0q_r4_runtime_readonly_v1 import (
    RuntimePreflightError,
    run_readonly_preflight,
    validate_runtime_snapshot,
)


def _snapshots() -> tuple[dict, dict]:
    process = {
        "addresses": ["127.0.0.1"],
        "executable_sha256": frozen_gate.SERVER_SHA256,
        "model_sha256": frozen_gate.MODEL_SHA256,
        "model_path": frozen_gate.MODEL_PATH,
        "command_line": "pinned command line",
    }
    service = {
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
    }
    return process, service


def test_valid_runtime_snapshot_passes_frozen_validator(monkeypatch) -> None:
    process, service = _snapshots()
    seen = []
    monkeypatch.setattr(
        frozen_gate,
        "_validate_preflight",
        lambda p, s: seen.append((p, s)),
    )

    validate_runtime_snapshot(process, service)

    assert seen == [(process, service)]


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("props", "default_generation_settings", "params", "top_p"), float("nan")),
        (("props", "default_generation_settings", "params", "top_p"), float("inf")),
        (("props", "default_generation_settings", "params", "top_p"), 10**1000),
        (("props", "default_generation_settings", "params", "top_p"), "0.95"),
        (("props", "default_generation_settings", "params", "repeat_penalty"), True),
        (("props", "default_generation_settings", "params", "top_k"), True),
    ],
)
def test_malformed_sampler_values_fail_closed(monkeypatch, path, value) -> None:
    process, service = _snapshots()
    cursor = service
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    monkeypatch.setattr(frozen_gate, "_validate_preflight", lambda *_: None)

    with pytest.raises(RuntimePreflightError):
        validate_runtime_snapshot(process, service)


@pytest.mark.parametrize(
    ("slots", "error"),
    [
        ({"n_ctx": 131072, "is_processing": False}, "slots_shape_or_count_mismatch"),
        ([{"n_ctx": 131072, "is_processing": None}], "slot_processing_state_invalid"),
        ([{"n_ctx": 131072, "is_processing": 0}], "slot_processing_state_invalid"),
        ([{"n_ctx": True, "is_processing": False}], "slot_context_mismatch"),
        ([{"n_ctx": 131072, "is_processing": False},
          {"n_ctx": 131072, "is_processing": False}], "slots_shape_or_count_mismatch"),
    ],
)
def test_malformed_slot_shapes_and_states_fail_closed(monkeypatch, slots, error) -> None:
    process, service = _snapshots()
    service["slots"] = copy.deepcopy(slots)
    monkeypatch.setattr(frozen_gate, "_validate_preflight", lambda *_: None)

    with pytest.raises(RuntimePreflightError, match=error):
        validate_runtime_snapshot(process, service)


def test_preflight_entrypoint_only_uses_read_snapshots(monkeypatch) -> None:
    process, service = _snapshots()
    calls = []
    monkeypatch.setattr(
        frozen_gate,
        "_load_frozen_inputs",
        lambda: (
            {"b1_status": "MEM3B0Q_MEM3B1_READY=NO"},
            {},
            {},
            {"runtime_preflight_amendment_sha256": "a" * 64},
        ),
    )
    monkeypatch.setattr(
        frozen_gate,
        "_ps_process_snapshot",
        lambda: calls.append("process_get") or process,
    )
    monkeypatch.setattr(
        frozen_gate,
        "_service_snapshot",
        lambda: calls.append("http_get_snapshot") or service,
    )
    monkeypatch.setattr(frozen_gate, "_validate_preflight", lambda *_: None)

    report = run_readonly_preflight()

    assert calls == ["process_get", "http_get_snapshot"]
    assert report["status"] == "PASS_READ_ONLY_PREFLIGHT"
    assert report["request_method_scope"] == "GET_ONLY"
    assert report["inference_performed"] is False
