from __future__ import annotations

import json

from tools.research.memory import run_mem3b0q_r4_factorized_event_slot_confirmation_v1r1 as v1r1


def test_lock_dispatch_repair_preserves_frozen_data_and_request_bytes() -> None:
    original = json.loads(
        (v1r1.ROOT / "docs/research/memory/mem3b0q_r4_factorized_event_slot_confirmation_v1_lock.json")
        .read_text(encoding="utf-8")
    )
    repaired = v1r1.build_lock_payload()

    assert repaired["dataset_sha256"] == original["dataset_sha256"]
    assert repaired["requests"] == original["requests"]
    assert repaired["case_ids"] == original["case_ids"]
    assert repaired["run_id"] == v1r1.OUTPUT_ROOT.name
    assert "tools/research/memory/run_mem3b0q_r4_factorized_event_slot_confirmation_v1r1.py" in repaired["dependencies"]


def test_wrapper_routes_engine_lock_validation_to_factorized_builder(monkeypatch) -> None:
    unused_output = v1r1.ROOT / "runs/memory/mem3/.v1r1-lock-validation-unused"
    assert not unused_output.exists()
    monkeypatch.setattr(v1r1, "OUTPUT_ROOT", unused_output)
    with v1r1._configured_base():
        with v1r1.base._configured_engine():
            lock, lock_sha, *_ = v1r1.base.engine._load_lock()

    assert lock["lock_id"] == "mem3b0q-r4-factorized-event-slot-confirmation-v1r1-lock"
    assert lock_sha == v1r1.base._sha(v1r1.LOCK_PATH.read_bytes())
