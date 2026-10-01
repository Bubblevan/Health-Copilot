from __future__ import annotations

import json

from tools.research.memory import run_mem3b0q_r4_factorized_event_slot_confirmation_v1r3 as v1r3


def test_overlap_audit_isolated_lock_keeps_frozen_requests() -> None:
    previous = json.loads(
        (v1r3.ROOT / "docs/research/memory/mem3b0q_r4_factorized_event_slot_confirmation_v1r2_lock.json")
        .read_text(encoding="utf-8")
    )
    current = v1r3.build_lock_payload()

    assert current["dataset_sha256"] == previous["dataset_sha256"]
    assert current["requests"] == previous["requests"]
    assert current["case_ids"] == previous["case_ids"]
    assert current["run_id"] == v1r3.OUTPUT_ROOT.name


def test_factorized_registry_can_load_lock_without_mutating_legacy_audit(monkeypatch) -> None:
    base = v1r3.base
    unused_output = v1r3.ROOT / "runs/memory/mem3/.v1r3-lock-validation-unused"
    assert not unused_output.exists()
    monkeypatch.setattr(v1r3, "OUTPUT_ROOT", unused_output)
    with v1r3._configured_v1r2():
        with v1r3.v1r2._configured_v1r1():
            with v1r3.v1r2.v1r1._configured_base():
                with base._configured_engine():
                    previous_ids = base.engine.CASE_IDS
                    base.engine.CASE_IDS = base.CASE_IDS
                    try:
                        lock, lock_sha, *_ = base.engine._load_lock()
                    finally:
                        base.engine.CASE_IDS = previous_ids

    assert lock["case_ids"] == list(base.CASE_IDS)
    assert lock_sha == base._sha(v1r3.LOCK_PATH.read_bytes())
