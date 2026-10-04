from types import SimpleNamespace

from health_ai_copilot.multi_agent.contracts import RouteMode
from health_ai_copilot.multi_agent.evaluation import _route_usage_flags
from health_ai_copilot.multi_agent.routing import MedicalRouter


def test_simple_education_uses_single_fast_path() -> None:
    decision = MedicalRouter().decide("What is a simple medical education question?")
    assert decision.mode == RouteMode.SINGLE
    assert decision.unique_key_count == 0


def test_cross_source_longitudinal_question_routes_to_team() -> None:
    decision = MedicalRouter().decide(
        "How has my previous record changed, and what do current guidelines recommend?"
    )
    assert decision.mode == RouteMode.TEAM
    assert {role.value for role in decision.predicted_capabilities} == {
        "patient_context", "evidence",
    }


def test_distinct_synthetic_keys_are_counted_once_each() -> None:
    decision = MedicalRouter().decide(
        "Report SYNKEY-ABC12345 and include SYNKEY-DEF67890; repeat SYNKEY-ABC12345."
    )
    assert decision.unique_key_count == 2
    assert decision.mode == RouteMode.TEAM


def test_lead_may_decline_team_without_counting_team_activation() -> None:
    execution = SimpleNamespace(
        route_decision=SimpleNamespace(mode=RouteMode.TEAM),
        response=SimpleNamespace(route_mode=RouteMode.SINGLE),
    )
    assert _route_usage_flags(execution) == {
        "single_fast_path": False,
        "router_team_candidate": True,
        "team_activated": False,
    }
