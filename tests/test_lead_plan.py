import pytest

from health_ai_copilot.multi_agent.contracts import RouteMode
from health_ai_copilot.multi_agent.routing import (
    MedicalRouter,
    deterministic_fallback_plan,
    parse_lead_plan,
)


def test_lead_plan_accepts_bounded_distinct_workers() -> None:
    plan = parse_lead_plan(
        '{"mode":"team","tasks":['
        '{"worker":"patient_context","objective":"查患者纵向变化"},'
        '{"worker":"evidence","objective":"检索支持证据"}]}'
    )
    assert plan.mode == RouteMode.TEAM
    assert len(plan.tasks) == 2


@pytest.mark.parametrize("payload", [
    '{"mode":"team","tasks":[]}',
    ('{"mode":"team","tasks":[{"worker":"care","objective":"x"},'
     '{"worker":"care","objective":"y"}]}'),
    '{"mode":"team","tasks":[{"worker":"other","objective":"x"}]}',
    ('{"mode":"team","tasks":[{"worker":"care","objective":"x"},'
     '{"worker":"patient_context","objective":"x"},'
     '{"worker":"evidence","objective":"x"},'
     '{"worker":"care","objective":"x"}]}'),
])
def test_lead_plan_rejects_malformed_or_over_budget_tasks(payload: str) -> None:
    with pytest.raises(ValueError):
        parse_lead_plan(payload)


def test_harness_plan_fallback_stays_inside_worker_budget() -> None:
    query = "compare my prior history with current evidence"
    plan = deterministic_fallback_plan(MedicalRouter().decide(query), query)
    assert plan.mode == RouteMode.TEAM
    assert 2 <= len(plan.tasks) <= 3
    assert len({role for role, _ in plan.tasks}) == len(plan.tasks)
