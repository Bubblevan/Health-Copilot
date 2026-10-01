from __future__ import annotations

import json
from pathlib import Path

from tools.research.memory import run_mem3b0q_r4_event_value_confirmation_v1r2 as v1r2


ROOT = Path(__file__).resolve().parents[1]


def test_elevated_retry_does_not_change_dataset_or_requests() -> None:
    prior = json.loads(
        (ROOT / "docs/research/memory/mem3b0q_r4_event_value_confirmation_v1_lock.json")
        .read_text(encoding="utf-8")
    )
    current = v1r2.build_lock_payload()

    assert current["dataset_sha256"] == prior["dataset_sha256"]
    assert current["requests"] == prior["requests"]
    assert current["case_ids"] == prior["case_ids"]
    assert current["run_id"] == v1r2.OUTPUT_ROOT.name
    assert current["process_snapshot_execution"].startswith("elevated_read_only")
