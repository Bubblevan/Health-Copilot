from __future__ import annotations

import json

from tools.research.memory import run_mem3b0q_r4_event_value_confirmation_v1 as runner


def test_confirmation_requests_and_dataset_are_frozen_without_gold_leak() -> None:
    pack, requests, scoring = runner._build_inputs()
    lock = runner.build_lock_payload()

    assert tuple(row["case_id"] for row in pack["cases"]) == runner.CASE_IDS
    assert tuple(requests) == runner.CASE_IDS
    assert tuple(scoring) == runner.CASE_IDS
    assert lock["case_ids"] == list(runner.CASE_IDS)
    assert len(lock["dataset_sha256"]) == 64
    assert len(lock["requests"]) == 3
    assert lock["hosted_api"] == "NONE"
    cases = {row["case_id"]: row for row in pack["cases"]}
    for case_id, record in requests.items():
        prompt = record["request"]["messages"][1]["content"]
        payload = json.loads(prompt)
        assert '"expected_atoms"' not in prompt
        assert '"temporal_expression_for_diagnostic_only"' not in prompt
        assert payload["source_id"] == case_id
        assert payload["proposition_text"] == cases[case_id]["proposition_text"]
