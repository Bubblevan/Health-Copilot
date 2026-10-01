"""Dispatch the factorized runner through its own locked build-lock function."""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from tools.research.memory import run_mem3b0q_r4_factorized_event_slot_confirmation_v1 as base


ROOT = base.ROOT
OUTPUT_ROOT = ROOT / "runs" / "memory" / "mem3" / "mem3b0q-r4-factorized-event-slot-confirmation-v1r1"
LOCK_PATH = ROOT / "docs" / "research" / "memory" / "mem3b0q_r4_factorized_event_slot_confirmation_v1r1_lock.json"
LOCK_SHA_PATH = LOCK_PATH.with_suffix(LOCK_PATH.suffix + ".sha256")
PROTOCOL_PATH = ROOT / "docs" / "research" / "memory" / "mem3b0q_r4_factorized_event_slot_confirmation_v1r1_protocol.md"
FREEZE_PATH = ROOT / "tools" / "research" / "memory" / "freeze_mem3b0q_r4_factorized_event_slot_confirmation_v1r1.py"
LOCK_STATIC_FIELDS = {
    **base.LOCK_STATIC_FIELDS,
    "lock_id": "mem3b0q-r4-factorized-event-slot-confirmation-v1r1-lock",
    "run_id": OUTPUT_ROOT.name,
    "process_snapshot_execution": "elevated_read_only_listener_process_query_with_factorized_lock_dispatch",
}
DEPENDENCY_PATHS = tuple(
    dict.fromkeys(
        (
            *base.DEPENDENCY_PATHS,
            "docs/research/memory/mem3b0q_r4_factorized_event_slot_confirmation_v1_preflight_failure.md",
            "docs/research/memory/mem3b0q_r4_factorized_event_slot_confirmation_v1r1_protocol.md",
            "tools/research/memory/run_mem3b0q_r4_factorized_event_slot_confirmation_v1r1.py",
            "tools/research/memory/freeze_mem3b0q_r4_factorized_event_slot_confirmation_v1r1.py",
            "docs/research/memory/mem3b0q_r4_factorized_event_slot_confirmation_v1_lock.json",
            "docs/research/memory/mem3b0q_r4_factorized_event_slot_confirmation_v1_lock.json.sha256",
        )
    )
)


@contextmanager
def _configured_base() -> Iterator[None]:
    names = ("OUTPUT_ROOT", "LOCK_PATH", "LOCK_SHA_PATH", "PROTOCOL_PATH", "FREEZE_PATH", "LOCK_STATIC_FIELDS", "DEPENDENCY_PATHS")
    saved = {name: getattr(base, name) for name in names}
    saved_builder = base.engine.build_lock_payload
    try:
        base.OUTPUT_ROOT = OUTPUT_ROOT
        base.LOCK_PATH = LOCK_PATH
        base.LOCK_SHA_PATH = LOCK_SHA_PATH
        base.PROTOCOL_PATH = PROTOCOL_PATH
        base.FREEZE_PATH = FREEZE_PATH
        base.LOCK_STATIC_FIELDS = LOCK_STATIC_FIELDS
        base.DEPENDENCY_PATHS = DEPENDENCY_PATHS
        base.engine.build_lock_payload = base.build_lock_payload
        yield
    finally:
        base.engine.build_lock_payload = saved_builder
        for name, value in saved.items():
            setattr(base, name, value)


def build_lock_payload() -> dict[str, Any]:
    with _configured_base():
        return base.build_lock_payload()


def run_once() -> dict[str, Any]:
    with _configured_base():
        return base._run_once()


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
