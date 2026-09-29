from health_ai_copilot.research.integration.counterfactual import CounterfactualRunner
from health_ai_copilot.research.integration.fixtures import build_synthetic_cases
from health_ai_copilot.research.integration.replay import semantic_execution_hash


def test_repeated_counterfactual_bundle_is_byte_stable() -> None:
    case = next(item for item in build_synthetic_cases() if item.case_id == "U1-ALL")
    runner = CounterfactualRunner()
    first = runner.run(case.episode, case.evaluation, case.resources)
    second = runner.run(case.episode, case.evaluation, case.resources)
    assert semantic_execution_hash(first) == semantic_execution_hash(second)
    assert [row.identity for row in first.arms] == [row.identity for row in second.arms]
