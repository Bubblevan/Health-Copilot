from __future__ import annotations

import pytest

from tools.research.rag_e5.repair_e5b4_cpu_usage_manifest import \
    derive_usage_totals


def test_usage_aggregation_ignores_zero_call_arms() -> None:
    model_arm = {
        "case_id": "case-1",
        "action": "STANDARD",
        "run_id": "run-1",
        "new_model_calls": 1,
        "guidance_input_tokens": 11,
        "guidance_output_tokens": 7,
        "guidance_raw_response": {"input_tokens": 11, "output_tokens": 7},
    }
    no_call_arms = [
        {"case_id": "case-2", "action": action, "run_id": action, "new_model_calls": 0}
        for action in ("OFF", "STANDARD", "STRONG")
    ]
    start = {
        "call_id": "call-1",
        "case_id": "case-1",
        "action": "STANDARD",
        "run_id": "run-1",
        "prompt_sha256": "prompt-sha",
        "call_ordinal": 1,
        "status": "STARTED",
    }
    done = {
        **start,
        "status": "COMPLETED",
        "input_tokens": 11,
        "output_tokens": 7,
    }

    result = derive_usage_totals(
        [*no_call_arms, model_arm], [start, done], expected_calls=1
    )

    assert result == {
        "model_calls": 1,
        "input_tokens": 11,
        "output_tokens": 7,
        "token_usage_complete": True,
    }


def test_usage_aggregation_rejects_missing_model_usage() -> None:
    arm = {
        "case_id": "case-1",
        "action": "OFF",
        "run_id": "run-1",
        "new_model_calls": 1,
        "guidance_raw_response": {"input_tokens": None, "output_tokens": 7},
    }
    common = {
        "call_id": "call-1",
        "case_id": "case-1",
        "action": "OFF",
        "run_id": "run-1",
        "prompt_sha256": "prompt-sha",
        "call_ordinal": 1,
    }
    ledger = [
        {**common, "status": "STARTED"},
        {**common, "status": "COMPLETED", "input_tokens": None, "output_tokens": 7},
    ]
    with pytest.raises(ValueError, match="missing token usage"):
        derive_usage_totals([arm], ledger, expected_calls=1)
