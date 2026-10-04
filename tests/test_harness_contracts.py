from dataclasses import fields
from datetime import UTC, datetime

import pytest

from health_ai_copilot.harness.contracts import AnswerSchema, HarnessRequest, RuntimeResources


def test_harness_request_has_runtime_fields_and_no_gold_plane() -> None:
    request = HarnessRequest(
        "r1", "question", AnswerSchema.FREE_TEXT,
        as_of_time=datetime.now(UTC), runtime_resources=RuntimeResources(max_tool_calls=0),
    )
    names = {item.name for item in fields(request)}
    assert "gold" not in names and "evaluator_truth" not in names
    assert request.answer_schema is AnswerSchema.FREE_TEXT


def test_runtime_integer_limits_reject_floats_and_booleans() -> None:
    with pytest.raises(ValueError):
        RuntimeResources(max_provider_calls=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        RuntimeResources(max_tool_calls=True)  # type: ignore[arg-type]
