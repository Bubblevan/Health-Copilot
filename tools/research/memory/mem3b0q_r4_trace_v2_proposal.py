"""Offline proposal for a prompt-witness observer; not wired into R4 v1."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any


TRACE_INSTRUMENTATION_VERSION = "mem3b0q-r4-trace-instrumentation-v2-proposal"


def normalize_slots_response(
    payload: Any, *, source_id: str, proposition_text: str
) -> dict[str, Any]:
    """Keep only prompt hashes and witness booleans from one /slots response."""
    if (
        not isinstance(source_id, str)
        or not source_id
        or not isinstance(proposition_text, str)
        or not proposition_text
    ):
        return {
            "trace_instrumentation_version": TRACE_INSTRUMENTATION_VERSION,
            "status": "UNVERIFIED",
            "failures": ["prompt_witness_target_invalid"],
            "slots": [],
        }

    if isinstance(payload, Mapping):
        slots = payload.get("slots")
    else:
        slots = payload

    failures: list[str] = []
    normalized: list[dict[str, Any]] = []
    if not isinstance(slots, list):
        return {
            "trace_instrumentation_version": TRACE_INSTRUMENTATION_VERSION,
            "status": "UNVERIFIED",
            "failures": ["slots_response_not_list"],
            "slots": [],
        }

    active_count = 0
    for index, slot in enumerate(slots):
        if not isinstance(slot, Mapping):
            failures.append(f"slot_not_object:{index}")
            continue

        processing_state = slot.get("is_processing")
        active = processing_state is True
        active_count += int(active)
        slot_id = slot.get("id")
        task_id = slot.get("id_task")
        if type(processing_state) is not bool:
            failures.append(f"slot_processing_state_invalid:{index}")
        if type(slot_id) is not int or type(task_id) is not int:
            failures.append(f"slot_identity_invalid:{index}")

        prompt_present = "prompt" in slot and isinstance(slot["prompt"], str)
        prompt = slot["prompt"] if prompt_present else ""
        contains_source_id = prompt_present and source_id in prompt
        contains_proposition = prompt_present and proposition_text in prompt

        normalized.append(
            {
                "slot_id": slot_id,
                "task_id": task_id,
                "is_processing": active,
                "prompt_present": prompt_present,
                "prompt_sha256": (
                    hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                    if prompt_present
                    else None
                ),
                "prompt_contains_source_id": contains_source_id,
                "prompt_contains_proposition": contains_proposition,
            }
        )

        if active and not prompt_present:
            failures.append(f"active_prompt_missing:{index}")
        elif active and not (contains_source_id and contains_proposition):
            failures.append(f"active_prompt_witness_mismatch:{index}")
    if active_count > 1:
        failures.append("multiple_active_slots")

    return {
        "trace_instrumentation_version": TRACE_INSTRUMENTATION_VERSION,
        "status": "UNVERIFIED" if failures else "PASS",
        "failures": failures,
        "slots": normalized,
    }


def evaluate_prompt_witnesses(observations: list[dict[str, Any]]) -> dict[str, Any]:
    """Require every observed active slot to witness one consistent task."""
    if not isinstance(observations, list):
        observations = []
        failures = ["observations_not_list"]
    else:
        failures = []
    active = [
        slot
        for observation in observations
        if isinstance(observation, Mapping)
        for slot in observation.get("slots", [])
        if isinstance(slot, Mapping)
        if slot.get("is_processing") is True
    ]
    slot_ids = {slot.get("slot_id") for slot in active}
    task_ids = {slot.get("task_id") for slot in active}
    for observation in observations:
        if not isinstance(observation, Mapping):
            failures.append("observation_not_object")
            continue
        if observation.get("status") != "PASS":
            failures.extend(observation.get("failures", []))
            if not observation.get("failures"):
                failures.append("observation_not_verified")

    if not active:
        failures.append("no_active_slot_observed")
    if len(slot_ids) != 1:
        failures.append("active_slot_identity_not_unique")
    if None in slot_ids:
        failures.append("active_slot_identity_missing")
    if len(task_ids) != 1 or None in task_ids:
        failures.append("active_task_identity_not_unique")
    if any(type(slot_id) is not int for slot_id in slot_ids):
        failures.append("active_slot_identity_invalid")
    if any(type(task_id) is not int for task_id in task_ids):
        failures.append("active_task_identity_invalid")
    if any(
        slot.get("prompt_present") is not True
        or slot.get("prompt_contains_source_id") is not True
        or slot.get("prompt_contains_proposition") is not True
        for slot in active
    ):
        failures.append("active_prompt_witness_incomplete")

    return {
        "trace_instrumentation_version": TRACE_INSTRUMENTATION_VERSION,
        "status": "UNVERIFIED" if failures else "PASS",
        "active_slot_samples": len(active),
        "active_slot_ids": sorted(slot_ids, key=str),
        "active_task_ids": sorted(task_ids, key=str),
        "failures": sorted(set(failures)),
    }
