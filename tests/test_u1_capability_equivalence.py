from dataclasses import replace

from health_ai_copilot.research.integration.actions import (
    ACTION_BY_KEY,
    ActionKey,
    capability_equivalence_report,
)
from health_ai_copilot.research.integration.executor import DeterministicIntegrationExecutor
from health_ai_copilot.research.integration.fixtures import build_synthetic_cases


def test_all_synthetic_single_team_unions_match() -> None:
    for case in build_synthetic_cases():
        report = capability_equivalence_report(case.episode)
        assert report.equivalent, (case.case_id, report.mismatches)


def test_worker_capability_violation_fails_before_any_read() -> None:
    case = next(item for item in build_synthetic_cases() if item.case_id == "U1-MEM-TEAM")
    surface = case.episode.tool_surface_ref
    bad_surface = replace(surface, workers=(surface.workers[0],))
    bad_episode = replace(case.episode, tool_surface_ref=bad_surface)
    report = capability_equivalence_report(bad_episode)
    assert not report.equivalent
    result = DeterministicIntegrationExecutor().execute(
        bad_episode,
        ACTION_BY_KEY[ActionKey.MEMORY_TEAM],
        case.resources,
    )
    assert [event.kind.value for event in result.trace] == ["ACTION_REJECTED"]
    assert "SINGLE_TEAM_CAPABILITY_MISMATCH" in result.trace[0].detail
