"""Bind the frozen factorized case registry before invoking the shared engine."""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from typing import Any, Iterator

from tools.research.memory import run_mem3b0q_r4_factorized_event_slot_confirmation_v1r1 as v1r1


ROOT = v1r1.ROOT
OUTPUT_ROOT = ROOT / "runs" / "memory" / "mem3" / "mem3b0q-r4-factorized-event-slot-confirmation-v1r2"
LOCK_PATH = ROOT / "docs" / "research" / "memory" / "mem3b0q_r4_factorized_event_slot_confirmation_v1r2_lock.json"
LOCK_SHA_PATH = LOCK_PATH.with_suffix(LOCK_PATH.suffix + ".sha256")
PROTOCOL_PATH = ROOT / "docs" / "research" / "memory" / "mem3b0q_r4_factorized_event_slot_confirmation_v1r2_protocol.md"
FREEZE_PATH = ROOT / "tools" / "research" / "memory" / "freeze_mem3b0q_r4_factorized_event_slot_confirmation_v1r2.py"
LOCK_STATIC_FIELDS = {
    **v1r1.LOCK_STATIC_FIELDS,
    "lock_id": "mem3b0q-r4-factorized-event-slot-confirmation-v1r2-lock",
    "run_id": OUTPUT_ROOT.name,
    "process_snapshot_execution": "elevated_read_only_listener_process_query_with_locked_case_registry",
}
DEPENDENCY_PATHS = tuple(
    dict.fromkeys(
        (
            *v1r1.DEPENDENCY_PATHS,
            "docs/research/memory/mem3b0q_r4_factorized_event_slot_confirmation_v1r1_preflight_failure.md",
            "docs/research/memory/mem3b0q_r4_factorized_event_slot_confirmation_v1r2_protocol.md",
            "tools/research/memory/run_mem3b0q_r4_factorized_event_slot_confirmation_v1r2.py",
            "tools/research/memory/freeze_mem3b0q_r4_factorized_event_slot_confirmation_v1r2.py",
            "docs/research/memory/mem3b0q_r4_factorized_event_slot_confirmation_v1r1_lock.json",
            "docs/research/memory/mem3b0q_r4_factorized_event_slot_confirmation_v1r1_lock.json.sha256",
            "runs/memory/mem3/mem3b0q-r4-factorized-event-slot-confirmation-v1r1/run_reservation.json",
        )
    )
)


@contextmanager
def _configured_v1r1() -> Iterator[None]:
    names = ("OUTPUT_ROOT", "LOCK_PATH", "LOCK_SHA_PATH", "PROTOCOL_PATH", "FREEZE_PATH", "LOCK_STATIC_FIELDS", "DEPENDENCY_PATHS")
    saved = {name: getattr(v1r1, name) for name in names}
    try:
        v1r1.OUTPUT_ROOT = OUTPUT_ROOT
        v1r1.LOCK_PATH = LOCK_PATH
        v1r1.LOCK_SHA_PATH = LOCK_SHA_PATH
        v1r1.PROTOCOL_PATH = PROTOCOL_PATH
        v1r1.FREEZE_PATH = FREEZE_PATH
        v1r1.LOCK_STATIC_FIELDS = LOCK_STATIC_FIELDS
        v1r1.DEPENDENCY_PATHS = DEPENDENCY_PATHS
        yield
    finally:
        for name, value in saved.items():
            setattr(v1r1, name, value)


def build_lock_payload() -> dict[str, Any]:
    with _configured_v1r1():
        return v1r1.build_lock_payload()


def run_once() -> dict[str, Any]:
    with _configured_v1r1():
        with v1r1._configured_base():
            engine = v1r1.base.engine
            previous_case_ids = engine.CASE_IDS
            engine.CASE_IDS = v1r1.base.CASE_IDS
            try:
                return v1r1.base._run_once()
            finally:
                engine.CASE_IDS = previous_case_ids


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--run-confirmation-once", action="store_true")
    args = parser.parse_args()
    if not args.run_confirmation_once:
        parser.error("explicit --run-confirmation-once is required")
    result = run_once()
    print(
        json.dumps(
            {
                "status": result["status"],
                "aggregate": result["aggregate"],
                "cases": [
                    {
                        "case_id": row["case_id"],
                        "status": row["status"],
                        "matched": row.get("matched_expected_atoms"),
                        "failure": row.get("failure"),
                        "audit": row.get("factorized_proposal_audit"),
                    }
                    for row in result["cases"]
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )
    return 0 if result.get("status") == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
