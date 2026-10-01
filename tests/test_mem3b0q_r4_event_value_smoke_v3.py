from __future__ import annotations

from tools.research.memory import run_mem3b0q_r4_event_value_smoke_v3 as runner


def test_v3_locks_only_the_observed_event_tuning_case() -> None:
    case, request, v1 = runner._build_request()
    lock = runner.build_lock_payload()

    assert case["case_id"] == "R4C-04"
    assert request["request"]["max_tokens"] == 256
    assert lock["case_ids"] == ["R4C-04"]
    assert lock["max_completion_posts"] == 1
    assert lock["adapter_version"].endswith("development")
    assert v1["request"]["response_format"]["json_schema"]["schema"]
