"""GET-only R4 runtime preflight; never submits a completion request."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any

from tools.research.memory import run_mem3b0q_r4_gate as frozen_gate


class RuntimePreflightError(ValueError):
    """A frozen R4 runtime invariant is absent, malformed, or mismatched."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimePreflightError(f"{label}_not_object")
    return value


def _integer(value: Any, expected: int, label: str) -> None:
    if type(value) is not int or value != expected:
        raise RuntimePreflightError(f"{label}_mismatch")


def _finite_number(value: Any, expected: float, label: str) -> None:
    if type(value) not in (int, float):
        raise RuntimePreflightError(f"{label}_not_finite_number")
    try:
        numeric = float(value)
    except (OverflowError, ValueError):
        raise RuntimePreflightError(f"{label}_not_finite_number") from None
    if not math.isfinite(numeric):
        raise RuntimePreflightError(f"{label}_not_finite_number")
    if abs(numeric - expected) > 0.001:
        raise RuntimePreflightError(f"{label}_mismatch")


def validate_runtime_snapshot(
    process: Mapping[str, Any], service: Mapping[str, Any]
) -> None:
    """Fail closed on malformed JSON shapes/types, then apply the frozen checks."""
    service = _mapping(service, "service")
    health = _mapping(service.get("health"), "health")
    props = _mapping(service.get("props"), "props")
    settings = _mapping(
        props.get("default_generation_settings"), "default_generation_settings"
    )
    params = _mapping(settings.get("params"), "sampler_params")
    models_payload = _mapping(service.get("models"), "models")
    models = models_payload.get("data")
    slots = service.get("slots")

    if health.get("status") != "ok":
        raise RuntimePreflightError("server_unhealthy")
    if not isinstance(models, list) or len(models) != 1:
        raise RuntimePreflightError("served_models_shape_mismatch")
    model = _mapping(models[0], "served_model")
    if model.get("id") != frozen_gate.MODEL_PATH:
        raise RuntimePreflightError("served_model_id_mismatch")

    if not isinstance(slots, list) or len(slots) != 1:
        raise RuntimePreflightError("slots_shape_or_count_mismatch")
    slot = _mapping(slots[0], "slot")
    if type(slot.get("is_processing")) is not bool or slot["is_processing"] is not False:
        raise RuntimePreflightError("slot_processing_state_invalid")
    _integer(slot.get("n_ctx"), 131072, "slot_context")
    _integer(props.get("total_slots"), 1, "server_total_slots")
    _integer(settings.get("n_ctx"), 131072, "server_context")
    _integer(params.get("top_k"), 40, "top_k")
    _finite_number(params.get("top_p"), 0.95, "top_p")
    _finite_number(params.get("min_p"), 0.05, "min_p")
    _finite_number(params.get("repeat_penalty"), 1.0, "repeat_penalty")

    frozen_gate._validate_preflight(dict(process), dict(service))


def run_readonly_preflight() -> dict[str, Any]:
    """Read pinned inputs and take process/GET snapshots without any POST path."""
    manifest, _, _, runner_lock = frozen_gate._load_frozen_inputs()
    process = frozen_gate._ps_process_snapshot()
    service = frozen_gate._service_snapshot()
    validate_runtime_snapshot(process, service)
    return {
        "preflight_id": "mem3b0q-r4-runtime-readonly-preflight-v1",
        "status": "PASS_READ_ONLY_PREFLIGHT",
        "inference_performed": False,
        "request_method_scope": "GET_ONLY",
        "protocol_manifest_sha256": frozen_gate._sha256(
            frozen_gate.MANIFEST_PATH.read_bytes()
        ),
        "runtime_preflight_amendment_sha256": runner_lock[
            "runtime_preflight_amendment_sha256"
        ],
        "process": process,
        "service": service,
        "b1_status": manifest["b1_status"],
    }


if __name__ == "__main__":
    print(
        json.dumps(
            run_readonly_preflight(), ensure_ascii=False, sort_keys=True, indent=2
        )
    )
