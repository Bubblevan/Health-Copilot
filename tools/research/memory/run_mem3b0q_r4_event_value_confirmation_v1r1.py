"""Retry a zero-POST event-confirmation run with verified netstat fallback."""

from __future__ import annotations

import argparse
import json

from tools.research.memory import run_mem3b0q_r4_event_value_confirmation_v1 as engine
from tools.research.memory import run_mem3b0q_r4_gate as frozen_gate
from tools.research.memory.mem3b0q_windows_listener_snapshot_v1 import (
    netstat_cim_process_snapshot,
    snapshot_with_permission_fallback,
)


ROOT = engine.ROOT
OUTPUT_ROOT = ROOT / "runs" / "memory" / "mem3" / "mem3b0q-r4-event-value-confirmation-v1r1"
LOCK_PATH = ROOT / "docs" / "research" / "memory" / "mem3b0q_r4_event_value_confirmation_v1r1_lock.json"
LOCK_SHA_PATH = LOCK_PATH.with_suffix(LOCK_PATH.suffix + ".sha256")
PROTOCOL_PATH = ROOT / "docs" / "research" / "memory" / "mem3b0q_r4_event_value_confirmation_v1r1_protocol.md"
FREEZE_PATH = ROOT / "tools" / "research" / "memory" / "freeze_mem3b0q_r4_event_value_confirmation_v1r1.py"
PRIMARY_PROCESS_SNAPSHOT = frozen_gate._ps_process_snapshot


def _process_snapshot() -> tuple[dict, str]:
    return snapshot_with_permission_fallback(
        PRIMARY_PROCESS_SNAPSHOT,
        lambda: netstat_cim_process_snapshot(
            port=frozen_gate.PORT,
            server_path=frozen_gate.SERVER_PATH,
            model_path=frozen_gate.MODEL_PATH,
        ),
    )


def _verified_process_snapshot() -> dict:
    snapshot, method = _process_snapshot()
    snapshot["snapshot_method"] = method
    return snapshot


frozen_gate._ps_process_snapshot = _verified_process_snapshot
engine.OUTPUT_ROOT = OUTPUT_ROOT
engine.LOCK_PATH = LOCK_PATH
engine.LOCK_SHA_PATH = LOCK_SHA_PATH
engine.PROTOCOL_PATH = PROTOCOL_PATH
engine.FREEZE_PATH = FREEZE_PATH
engine.LOCK_STATIC_FIELDS = {
    **engine.LOCK_STATIC_FIELDS,
    "lock_id": "mem3b0q-r4-event-value-confirmation-v1r1-lock",
    "run_id": OUTPUT_ROOT.name,
    "process_snapshot_policy": "primary_Get-NetTCPConnection; fallback_only_on_PermissionDenied_to_netstat_plus_CIM_then_same_frozen_validators",
}
engine.DEPENDENCY_PATHS = tuple(
    dict.fromkeys(
        (
            *engine.DEPENDENCY_PATHS,
            "docs/research/memory/mem3b0q_r4_event_value_confirmation_v1r1_protocol.md",
            "docs/research/memory/mem3b0q_r4_event_value_confirmation_v1_preflight_failure.md",
            "tools/research/memory/mem3b0q_windows_listener_snapshot_v1.py",
            "tools/research/memory/run_mem3b0q_r4_event_value_confirmation_v1r1.py",
            "tools/research/memory/freeze_mem3b0q_r4_event_value_confirmation_v1r1.py",
            "runs/memory/mem3/mem3b0q-r4-event-value-confirmation-v1/confirmation_result.json",
            "runs/memory/mem3/mem3b0q-r4-event-value-confirmation-v1/run_manifest.json",
            "runs/memory/mem3/mem3b0q-r4-event-value-confirmation-v1/run_reservation.json",
            "runs/memory/mem3/mem3b0q-r4-event-value-confirmation-v1/EVTCONF-01/request.json",
            "runs/memory/mem3/mem3b0q-r4-event-value-confirmation-v1/EVTCONF-01/reservation.json",
            "runs/memory/mem3/mem3b0q-r4-event-value-confirmation-v1/EVTCONF-01/case_result.json",
        )
    )
)

build_lock_payload = engine.build_lock_payload
_load_lock = engine._load_lock
run_once = engine.run_once


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
