from health_ai_copilot.evaluation.parser import canonical_option_set
from health_ai_copilot.harness.contracts import AnswerSchema
from health_ai_copilot.harness.verification import parse_answer


def test_multiselect_accepts_compact_and_separated_labels() -> None:
    assert parse_answer("AC", AnswerSchema.MULTI_SELECT) == ("A", "C")
    assert parse_answer("A, C because ...", AnswerSchema.MULTI_SELECT) == ("A", "C")
    assert canonical_option_set("AC") == ("A", "C")
