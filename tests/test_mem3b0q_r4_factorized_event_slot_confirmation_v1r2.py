from __future__ import annotations

import json

from tools.research.memory import run_mem3b0q_r4_factorized_event_slot_confirmation_v1r2 as v1r2


def test_registry_wrapper_preserves_requests_and_changes_only_run_binding() -> None:
    prior = json.loads(
        (v1r2.ROOT / "docs/research/memory/mem3b0q_r4_factorized_event_slot_confirmation_v1_lock.json")
        .read_text(encoding="utf-8")
    )
    current = v1r2.build_lock_payload()

    assert current["dataset_sha256"] == prior["dataset_sha256"]
    assert current["requests"] == prior["requests"]
    assert current["case_ids"] == prior["case_ids"]
    assert current["run_id"] == v1r2.OUTPUT_ROOT.name


def test_run_once_binds_case_ids_for_shared_engine(monkeypatch) -> None:
    observed = []
    monkeypatch.setattr(v1r2.v1r1.base, "_run_once", lambda: observed.extend(v1r2.v1r1.base.engine.CASE_IDS) or {})

    v1r2.run_once()

    assert tuple(observed) == v1r2.v1r1.base.CASE_IDS
    assert tuple(v1r2.v1r1.base.engine.CASE_IDS) != tuple(observed)
