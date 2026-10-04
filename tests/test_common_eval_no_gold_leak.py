from dataclasses import asdict

from health_ai_copilot.evaluation.contracts import EvalCase
from health_ai_copilot.harness.contracts import AnswerSchema, HarnessRequest


def test_gold_stays_on_eval_case_and_is_absent_from_runtime_request() -> None:
    case = EvalCase("c1", "question", AnswerSchema.SINGLE_CHOICE, "A")
    request = HarnessRequest("r1", case.query, case.answer_schema, benchmark_case_id=case.case_id)
    assert case.gold == "A"
    assert "gold" not in asdict(request)
