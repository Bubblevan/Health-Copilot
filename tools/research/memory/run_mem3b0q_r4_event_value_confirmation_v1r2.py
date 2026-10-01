"""Fresh zero-method-change attempt with OS process-query permission."""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from typing import Iterator

from tools.research.memory import run_mem3b0q_r4_event_value_confirmation_v1 as engine
from tools.research.memory import run_mem3b0q_r4_gate as frozen_gate
from tools.research.memory.mem3b0q_windows_listener_snapshot_v1 import (
    netstat_cim_process_snapshot,
    snapshot_with_permission_fallback,
)


ROOT = engine.ROOT
OUTPUT_ROOT = ROOT / "runs" / "memory" / "mem3" / "mem3b0q-r4-event-value-confirmation-v1r2"
LOCK_PATH = ROOT / "docs" / "research" / "memory" / "mem3b0q_r4_event_value_confirmation_v1r2_lock.json"
LOCK_SHA_PATH = LOCK_PATH.with_suffix(LOCK_PATH.suffix + ".sha256")
PROTOCOL_PATH = ROOT / "docs" / "research" / "memory" / "mem3b0q_r4_event_value_confirmation_v1r2_protocol.md"
FREEZE_PATH = ROOT / "tools" / "research" / "memory" / "freeze_mem3b0q_r4_event_value_confirmation_v1r2.py"

BASE_V1_LOCK_PATH = ROOT / "docs" / "research" / "memory" / "mem3b0q_r4_event_value_confirmation_v1_lock.json"
BASE_V1_LOCK = json.loads(BASE_V1_LOCK_PATH.read_text(encoding="utf-8"))
BASE_V1_STATIC_FIELDS = {
    key: value
    for key, value in BASE_V1_LOCK.items()
    if key not in engine.LOCK_DYNAMIC_FIELDS
}
BASE_V1_DEPENDENCIES = tuple(BASE_V1_LOCK["dependencies"])
PRIMARY_PROCESS_SNAPSHOT = frozen_gate._ps_process_snapshot

# The v1r1 import preserves its original process function but also configures
# shared module globals. Capture its audit lock, then restore the v1 core state.
from tools.research.memory import run_mem3b0q_r4_event_value_confirmation_v1r1 as retry

R1_LOCK_PATH = ROOT / "docs" / "research" / "memory" / "mem3b0q_r4_event_value_confirmation_v1r1_lock.json"
R1_LOCK = json.loads(R1_LOCK_PATH.read_text(encoding="utf-8"))
R1_STATIC_FIELDS = {
    key: value for key, value in R1_LOCK.items() if key not in engine.LOCK_DYNAMIC_FIELDS
}
R1_DEPENDENCIES = tuple(R1_LOCK["dependencies"])
PRIMARY_PROCESS_SNAPSHOT = retry.PRIMARY_PROCESS_SNAPSHOT
engine.OUTPUT_ROOT = ROOT / "runs" / "memory" / "mem3" / "mem3b0q-r4-event-value-confirmation-v1"
engine.LOCK_PATH = BASE_V1_LOCK_PATH
engine.LOCK_SHA_PATH = BASE_V1_LOCK_PATH.with_suffix(BASE_V1_LOCK_PATH.suffix + ".sha256")
engine.PROTOCOL_PATH = ROOT / "docs" / "research" / "memory" / "mem3b0q_r4_event_value_confirmation_v1_protocol.md"
engine.FREEZE_PATH = ROOT / "tools" / "research" / "memory" / "freeze_mem3b0q_r4_event_value_confirmation_v1.py"
engine.LOCK_STATIC_FIELDS = BASE_V1_STATIC_FIELDS
engine.DEPENDENCY_PATHS = BASE_V1_DEPENDENCIES
frozen_gate._ps_process_snapshot = PRIMARY_PROCESS_SNAPSHOT

LOCK_STATIC_FIELDS = {
    **R1_STATIC_FIELDS,
    "lock_id": "mem3b0q-r4-event-value-confirmation-v1r2-lock",
    "run_id": OUTPUT_ROOT.name,
    "process_snapshot_execution": "elevated_read_only_listener_process_query_then_same_frozen_validators",
}
DEPENDENCY_PATHS = tuple(
    dict.fromkeys(
        (
            *R1_DEPENDENCIES,
            "docs/research/memory/mem3b0q_r4_event_value_confirmation_v1r2_protocol.md",
            "docs/research/memory/mem3b0q_r4_event_value_confirmation_v1r2_preflight.md",
            "tools/research/memory/run_mem3b0q_r4_event_value_confirmation_v1r2.py",
            "tools/research/memory/freeze_mem3b0q_r4_event_value_confirmation_v1r2.py",
            "docs/research/memory/mem3b0q_r4_event_value_confirmation_v1r1_lock.json",
            "docs/research/memory/mem3b0q_r4_event_value_confirmation_v1r1_lock.json.sha256",
            "docs/research/memory/mem3b0q_r4_event_value_confirmation_v1r1_preflight_failure.md",
            "runs/memory/mem3/mem3b0q-r4-event-value-confirmation-v1r1/confirmation_result.json",
            "runs/memory/mem3/mem3b0q-r4-event-value-confirmation-v1r1/run_manifest.json",
            "runs/memory/mem3/mem3b0q-r4-event-value-confirmation-v1r1/run_reservation.json",
            "runs/memory/mem3/mem3b0q-r4-event-value-confirmation-v1r1/EVTCONF-01/request.json",
            "runs/memory/mem3/mem3b0q-r4-event-value-confirmation-v1r1/EVTCONF-01/reservation.json",
            "runs/memory/mem3/mem3b0q-r4-event-value-confirmation-v1r1/EVTCONF-01/case_result.json",
            "runs/memory/mem3/mem3b0q-r4-event-value-confirmation-v1r1/confirmation_result.json",
            "runs/memory/mem3/mem3b0q-r4-event-value-confirmation-v1r1/run_manifest.json",
            "runs/memory/mem3/mem3b0q-r4-event-value-confirmation-v1r1/run_reservation.json",
            "runs/memory/mem3/mem3b0q-r4-event-value-confirmation-v1r1/EVTCONF-01/request.json",
            "runs/memory/mem3/mem3b0q-r4-event-value-confirmation-v1r1/EVTCONF-01/reservation.json",
            "runs/memory/mem3/mem3b0q-r4-event-value-confirmation-v1r1/EVTCONF-01/case_result.json",
            "docs/research/memory/mem3b0q_r4_event_value_confirmation_v1r1_preflight_failure.md",
        )
    )
)


def _verified_process_snapshot() -> dict:
    snapshot, method = snapshot_with_permission_fallback(
        PRIMARY_PROCESS_SNAPSHOT,
        lambda: netstat_cim_process_snapshot(
            port=frozen_gate.PORT,
            server_path=frozen_gate.SERVER_PATH,
            model_path=frozen_gate.MODEL_PATH,
        ),
    )
    snapshot["snapshot_method"] = method
    return snapshot


@contextmanager
def _configured_engine() -> Iterator[None]:
    names = (
        "OUTPUT_ROOT",
        "LOCK_PATH",
        "LOCK_SHA_PATH",
        "PROTOCOL_PATH",
        "FREEZE_PATH",
        "LOCK_STATIC_FIELDS",
        "DEPENDENCY_PATHS",
    )
    saved = {name: getattr(engine, name) for name in names}
    saved_snapshot = frozen_gate._ps_process_snapshot
    try:
        engine.OUTPUT_ROOT = OUTPUT_ROOT
        engine.LOCK_PATH = LOCK_PATH
        engine.LOCK_SHA_PATH = LOCK_SHA_PATH
        engine.PROTOCOL_PATH = PROTOCOL_PATH
        engine.FREEZE_PATH = FREEZE_PATH
        engine.LOCK_STATIC_FIELDS = LOCK_STATIC_FIELDS
        engine.DEPENDENCY_PATHS = DEPENDENCY_PATHS
        frozen_gate._ps_process_snapshot = _verified_process_snapshot
        yield
    finally:
        for name, value in saved.items():
            setattr(engine, name, value)
        frozen_gate._ps_process_snapshot = saved_snapshot


def build_lock_payload() -> dict:
    with _configured_engine():
        return engine.build_lock_payload()


def _load_lock():
    with _configured_engine():
        return engine._load_lock()


def run_once() -> dict:
    with _configured_engine():
        return engine.run_once()


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--run-confirmation-once", action="store_true")
    args = parser.parse_args()
    if not args.run_confirmation_once:
        parser.error("explicit --run-confirmation-once is required")
    result = run_once()
    print(json.dumps({"status": result["status"], "aggregate": result["aggregate"], "cases": [{"case_id": row["case_id"], "status": row["status"], "matched": row.get("matched_expected_atoms"), "expected": row.get("expected_atoms"), "failure": row.get("failure")} for row in result["cases"]]}, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if result.get("status") == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
