from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.research.memory import run_mem3b0q_r4_event_value_confirmation_v1r1 as v1r1
from tools.research.memory.mem3b0q_windows_listener_snapshot_v1 import (
    snapshot_with_permission_fallback,
)

ROOT = Path(__file__).resolve().parents[1]


def test_fallback_is_only_used_for_explicit_permission_denial() -> None:
    calls = []

    def denied():
        raise RuntimeError("Get-NetTCPConnection PermissionDenied")

    def fallback():
        calls.append("fallback")
        return {"addresses": ["127.0.0.1"]}

    snapshot, method = snapshot_with_permission_fallback(denied, fallback)

    assert snapshot["addresses"] == ["127.0.0.1"]
    assert method == "NETSTAT_CIM_PERMISSION_FALLBACK"
    assert calls == ["fallback"]

    with pytest.raises(RuntimeError, match="binary_hash_mismatch"):
        snapshot_with_permission_fallback(
            lambda: (_ for _ in ()).throw(RuntimeError("binary_hash_mismatch")),
            fallback,
        )
    assert calls == ["fallback"]


def test_retry_lock_preserves_dataset_and_exact_request_hashes() -> None:
    old_lock_path = (
        ROOT
        / "docs/research/memory/mem3b0q_r4_event_value_confirmation_v1_lock.json"
    )
    old_lock = json.loads(old_lock_path.read_text(encoding="utf-8"))
    retry_lock = v1r1.build_lock_payload()

    assert retry_lock["dataset_sha256"] == old_lock["dataset_sha256"]
    assert retry_lock["requests"] == old_lock["requests"]
    assert retry_lock["case_ids"] == old_lock["case_ids"]
    assert retry_lock["process_snapshot_policy"].startswith("primary_Get-NetTCPConnection")
    assert retry_lock["run_id"] == v1r1.OUTPUT_ROOT.name
    assert all(
        relative.startswith("runs/memory/mem3/mem3b0q-r4-event-value-confirmation-v1/")
        for relative in retry_lock["dependencies"]
        if "event-value-confirmation-v1/" in relative
    )
