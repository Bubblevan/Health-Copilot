"""Metric helpers for supplied M1 runs; this module never invents live results."""

from collections.abc import Iterable, Mapping
from typing import Any

from ..agent.loop import AgentRunResult
from ..contracts import AssistantResponse, Route


def summarize_m1_runs(
    cases: Iterable[Mapping[str, Any]],
    runs: Mapping[str, AgentRunResult],
    responses: Mapping[str, AssistantResponse] | None = None,
    *,
    safety_short_circuits: int = 0,
) -> dict[str, float | int]:
    """Aggregate only metrics supported by the run traces supplied by the caller.

    The evaluator deliberately does not call a model and does not infer a
    recovery rewrite. A missing run is excluded from run-dependent metrics.
    """
    case_list = list(cases)
    run_list = [runs[case["id"]] for case in case_list if case.get("id") in runs]
    run_count = len(run_list)
    tool_activated = sum(run.state.tool_calls_used > 0 for run in run_list)
    budget_exhausted = sum(
        run.stop_reason is not None
        and run.stop_reason.value in {"max_model_turns", "max_tool_calls"}
        for run in run_list
    )
    expected_cases = [
        case
        for case in case_list
        if case.get("expected_source_ids") and case.get("id") in runs
    ]
    recovery_hits = sum(
        bool(
            {
                item.source_id for item in runs[case["id"]].observed_evidence
            }.intersection(case["expected_source_ids"])
        )
        for case in expected_cases
    )
    direct_hit_cases = [
        case
        for case in case_list
        if case.get("category") == "direct_hit" and case.get("id") in runs
    ]
    unnecessary_recoveries = sum(
        runs[case["id"]].state.tool_calls_used > 0 for case in direct_hit_cases
    )

    metrics: dict[str, float | int] = {
        "run_cases": run_count,
        "tool_activation_rate": tool_activated / run_count if run_count else 0.0,
        "mean_tool_calls": (
            sum(run.state.tool_calls_used for run in run_list) / run_count
            if run_count
            else 0.0
        ),
        "budget_exhaustion_rate": budget_exhausted / run_count if run_count else 0.0,
        "recovery_hit_at_3": (
            recovery_hits / len(expected_cases) if expected_cases else 0.0
        ),
        "unnecessary_recovery_rate": (
            unnecessary_recoveries / len(direct_hit_cases) if direct_hit_cases else 0.0
        ),
        "safety_short_circuit_rate": (
            safety_short_circuits / len(case_list) if case_list else 0.0
        ),
    }

    if responses is not None:
        answer_cases = [
            case
            for case in case_list
            if case.get("id") in runs
            and case.get("id") in responses
            and runs[case["id"]].draft is not None
            and not runs[case["id"]].draft.abstain
        ]
        citation_passes = sum(
            responses[case["id"]].route == Route.ANSWER
            for case in answer_cases
        )
        metrics["citation_integrity_pass_rate"] = (
            citation_passes / len(answer_cases) if answer_cases else 0.0
        )
    return metrics
