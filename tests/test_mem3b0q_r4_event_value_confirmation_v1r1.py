from __future__ import annotations

import json
import hashlib
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
    lock_path = (
        ROOT
        / "docs/research/memory/mem3b0q_r4_event_value_confirmation_v1r1_lock.json"
    )
    frozen_lock = json.loads(lock_path.read_text(encoding="utf-8"))
    pack, requests, _ = v1r1.engine._build_inputs()
    pack_path = ROOT / "docs/research/memory/mem3b0q_r4_event_value_confirmation_pack_v1.json"
    request_map = {
        case_id: {
            "request_sha256": row["request_sha256"],
            "schema_sha256": row["schema_sha256"],
        }
        for case_id, row in requests.items()
    }

    assert len(pack["cases"]) == 3
    assert hashlib.sha256(pack_path.read_bytes()).hexdigest() == frozen_lock["dataset_sha256"]
    assert request_map == frozen_lock["requests"]
    assert frozen_lock["case_ids"] == ["EVTCONF-01", "EVTCONF-02", "EVTCONF-03"]
    assert frozen_lock["process_snapshot_policy"].startswith("primary_Get-NetTCPConnection")
    assert frozen_lock["run_id"] == "mem3b0q-r4-event-value-confirmation-v1r1"
