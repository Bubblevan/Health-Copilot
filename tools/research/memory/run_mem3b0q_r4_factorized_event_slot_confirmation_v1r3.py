"""Isolate legacy overlap-audit IDs for the factorized confirmation run."""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from typing import Any, Iterator

from tools.research.memory import run_mem3b0q_r4_factorized_event_slot_confirmation_v1 as base
from tools.research.memory import run_mem3b0q_r4_factorized_event_slot_confirmation_v1r2 as v1r2


ROOT = base.ROOT
OUTPUT_ROOT = ROOT / "runs" / "memory" / "mem3" / "mem3b0q-r4-factorized-event-slot-confirmation-v1r3"
LOCK_PATH = ROOT / "docs" / "research" / "memory" / "mem3b0q_r4_factorized_event_slot_confirmation_v1r3_lock.json"
LOCK_SHA_PATH = LOCK_PATH.with_suffix(LOCK_PATH.suffix + ".sha256")
PROTOCOL_PATH = ROOT / "docs" / "research" / "memory" / "mem3b0q_r4_factorized_event_slot_confirmation_v1r3_protocol.md"
FREEZE_PATH = ROOT / "tools" / "research" / "memory" / "freeze_mem3b0q_r4_factorized_event_slot_confirmation_v1r3.py"
LOCK_STATIC_FIELDS = {
    **v1r2.LOCK_STATIC_FIELDS,
    "lock_id": "mem3b0q-r4-factorized-event-slot-confirmation-v1r3-lock",
    "run_id": OUTPUT_ROOT.name,
    "process_snapshot_execution": "elevated_read_only_listener_process_query_with_legacy_overlap_id_isolation",
}
DEPENDENCY_PATHS = tuple(
    dict.fromkeys(
        (
            *v1r2.DEPENDENCY_PATHS,
            "docs/research/memory/mem3b0q_r4_factorized_event_slot_confirmation_v1r2_preflight_failure.md",
            "docs/research/memory/mem3b0q_r4_factorized_event_slot_confirmation_v1r3_protocol.md",
            "tools/research/memory/run_mem3b0q_r4_factorized_event_slot_confirmation_v1r3.py",
            "tools/research/memory/freeze_mem3b0q_r4_factorized_event_slot_confirmation_v1r3.py",
            "docs/research/memory/mem3b0q_r4_factorized_event_slot_confirmation_v1r2_lock.json",
            "docs/research/memory/mem3b0q_r4_factorized_event_slot_confirmation_v1r2_lock.json.sha256",
        )
    )
)
LEGACY_CASE_IDS = tuple(base.engine.CASE_IDS)
LEGACY_BUILD_INPUTS = base.BASE_BUILD_INPUTS


@contextmanager
def _configured_v1r2() -> Iterator[None]:
    names = ("OUTPUT_ROOT", "LOCK_PATH", "LOCK_SHA_PATH", "PROTOCOL_PATH", "FREEZE_PATH", "LOCK_STATIC_FIELDS", "DEPENDENCY_PATHS")
    saved = {name: getattr(v1r2, name) for name in names}
    saved_builder = base.BASE_BUILD_INPUTS
    try:
        v1r2.OUTPUT_ROOT = OUTPUT_ROOT
        v1r2.LOCK_PATH = LOCK_PATH
        v1r2.LOCK_SHA_PATH = LOCK_SHA_PATH
        v1r2.PROTOCOL_PATH = PROTOCOL_PATH
        v1r2.FREEZE_PATH = FREEZE_PATH
        v1r2.LOCK_STATIC_FIELDS = LOCK_STATIC_FIELDS
        v1r2.DEPENDENCY_PATHS = DEPENDENCY_PATHS

        def isolated_legacy_builder():
            engine = base.engine
            active_ids = engine.CASE_IDS
            engine.CASE_IDS = LEGACY_CASE_IDS
            try:
                return LEGACY_BUILD_INPUTS()
            finally:
                engine.CASE_IDS = active_ids

        base.BASE_BUILD_INPUTS = isolated_legacy_builder
        yield
    finally:
        base.BASE_BUILD_INPUTS = saved_builder
        for name, value in saved.items():
            setattr(v1r2, name, value)


def build_lock_payload() -> dict[str, Any]:
    with _configured_v1r2():
        return v1r2.build_lock_payload()


def run_once() -> dict[str, Any]:
    with _configured_v1r2():
        return v1r2.run_once()


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--run-confirmation-once", action="store_true")
    args = parser.parse_args()
    if not args.run_confirmation_once:
        parser.error("explicit --run-confirmation-once is required")
    result = run_once()
    print(json.dumps({"status": result["status"], "aggregate": result["aggregate"], "cases": [{"case_id": row["case_id"], "status": row["status"], "matched": row.get("matched_expected_atoms"), "failure": row.get("failure"), "audit": row.get("factorized_proposal_audit")} for row in result["cases"]]}, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if result.get("status") == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
