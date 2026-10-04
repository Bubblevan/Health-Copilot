from datetime import UTC, datetime

import pytest

from health_ai_copilot.harness.contracts import AnswerSchema, HarnessRequest


def test_context_identity_requires_timezone_aware_as_of_time() -> None:
    with pytest.raises(ValueError):
        HarnessRequest("r", "q", AnswerSchema.FREE_TEXT, subject_id="p",
                       as_of_time=datetime(2026, 1, 1))  # noqa: DTZ001 - assert naive timestamps fail
    request = HarnessRequest("r", "q", AnswerSchema.FREE_TEXT, subject_id="p",
                             as_of_time=datetime(2026, 1, 1, tzinfo=UTC))
    assert request.subject_id == "p"
