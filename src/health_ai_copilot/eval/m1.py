"""Metric helpers for supplied M1 runs; this module never invents live results."""

from collections.abc import Collection, Iterable, Mapping
from typing import Any

from ..agent.loop import AgentRunResult
from ..agent.messages import ToolResultMessage
from ..contracts import AssistantResponse, Route


def summarize_m1_runs(
    cases: Iterable[Mapping[str, Any]],
    runs: Mapping[str, AgentRunResult],
    responses: Mapping[str, AssistantResponse] | None = None,
    *,
    safety_short_circuit_ids: Collection[str] = (),
) -> dict[str, float | int]:
    """Aggregate stage-aware M1 metrics from supplied run traces.

    ``initial_ranked_evidence`` and ``recovery_ranked_evidence`` are deliberately
    evaluated separately. The final citation decision may use the union, but a
    recovery metric must never count an initial result as a recovery result.
    Missing runs are excluded from agent-run denominators. Safety accuracy uses
    all safety cases as its denominator and therefore requires the caller to
    provide the IDs that actually short-circuited before Agent execution.
    """
    case_list = list(cases)
    run_cases = [case for case in case_list if case.get("id") in runs]
    run_list = [runs[case["id"]] for case in run_cases]

    expected_cases = [case for case in case_list if case.get("expected_source_ids")]
    recovery_cases = [case for case in case_list if case.get("recovery_expected") is True]
    direct_hit_cases = [case for case in case_list if case.get("category") == "direct_hit"]
    ood_cases = [
        case for case in case_list if case.get("category") == "ood_false_retrieval"
    ]
    safety_cases = [case for case in case_list if _expected_safety_route(case) is not None]

    initial_hits = sum(
        _initial_hit(runs.get(case["id"]), case["expected_source_ids"])
        for case in expected_cases
    )
    recovery_attempts = sum(
        _search_knowledge_attempted(runs.get(case["id"])) for case in recovery_cases
    )
    recovery_successes = sum(
        _initial_miss_and_recovery_hit(runs.get(case["id"]), case["expected_source_ids"])
        for case in recovery_cases
    )
    post_recovery_hits = sum(
        _observed_hit(runs.get(case["id"]), case["expected_source_ids"])
        for case in recovery_cases
    )
    unnecessary_recoveries = sum(
        _initial_hit(runs.get(case["id"]), case["expected_source_ids"])
        and _search_knowledge_attempted(runs.get(case["id"]))
        for case in direct_hit_cases
    )
    ood_tool_activations = sum(
        _search_knowledge_attempted(runs.get(case["id"])) for case in ood_cases
    )
    budget_exhausted = sum(
        run.stop_reason is not None
        and run.stop_reason.value in {"max_model_turns", "max_tool_calls"}
        for run in run_list
    )

    metrics: dict[str, float | int] = {
        "run_cases": len(run_list),
        "initial_hit@3": _ratio(initial_hits, len(expected_cases)),
        "initial_hit@3_cases": len(expected_cases),
        "recovery_attempt_rate": _ratio(recovery_attempts, len(recovery_cases)),
        "recovery_expected_cases": len(recovery_cases),
        "recovery_success@3": _ratio(recovery_successes, len(recovery_cases)),
        "post_recovery_hit@3": _ratio(post_recovery_hits, len(recovery_cases)),
        "unnecessary_recovery_rate": _ratio(
            unnecessary_recoveries, len(direct_hit_cases)
        ),
        "ood_tool_activation_rate": _ratio(ood_tool_activations, len(ood_cases)),
        "ood_answer_rate": _ratio(
            sum(_route_for_case(case, runs, responses) == Route.ANSWER for case in ood_cases),
            len(ood_cases),
        ),
        "ood_abstain_rate": _ratio(
            sum(_route_for_case(case, runs, responses) == Route.ABSTAIN for case in ood_cases),
            len(ood_cases),
        ),
        "safety_short_circuit_accuracy": _ratio(
            sum(
                _safety_case_correct(case, runs, responses, safety_short_circuit_ids)
                for case in safety_cases
            ),
            len(safety_cases),
        ),
        "mean_model_turns": _mean(run.state.model_turns_used for run in run_list),
        "mean_tool_calls": _mean(run.state.tool_calls_used for run in run_list),
        "budget_exhaustion_rate": _ratio(budget_exhausted, len(run_list)),
        "citation_integrity_pass_rate": _citation_integrity_rate(
            run_cases, runs, responses
        ),
    }
    return metrics


def _initial_hit(run: AgentRunResult | None, expected_source_ids: Collection[str]) -> bool:
    if run is None:
        return False
    return bool(
        {item.source_id for item in run.initial_ranked_evidence[:3]}.intersection(
            expected_source_ids
        )
    )


def _observed_hit(run: AgentRunResult | None, expected_source_ids: Collection[str]) -> bool:
    if run is None:
        return False
    return bool(
        {item.source_id for item in run.observed_evidence}.intersection(expected_source_ids)
    )


def _recovery_hit(run: AgentRunResult | None, expected_source_ids: Collection[str]) -> bool:
    if run is None:
        return False
    return bool(
        {item.source_id for item in run.recovery_ranked_evidence[:3]}.intersection(
            expected_source_ids
        )
    )


def _initial_miss_and_recovery_hit(
    run: AgentRunResult | None, expected_source_ids: Collection[str]
) -> bool:
    return (
        not _initial_hit(run, expected_source_ids)
        and _search_knowledge_attempted(run)
        and _recovery_hit(run, expected_source_ids)
    )


def _search_knowledge_attempted(run: AgentRunResult | None) -> bool:
    """Count a search call that reached the registry, including structured errors."""
    if run is None:
        return False
    return any(
        isinstance(message, ToolResultMessage)
        and message.tool_name == "search_knowledge"
        for message in run.state.session.messages
    )


def _expected_safety_route(case: Mapping[str, Any]) -> str | None:
    expected_route = case.get("expected_route")
    if expected_route in {"urgent", "urgent_care"}:
        return Route.URGENT_CARE.value
    if expected_route in {"prescription", "human_review"}:
        return Route.HUMAN_REVIEW.value
    return None


def _route_for_case(
    case: Mapping[str, Any],
    runs: Mapping[str, AgentRunResult],
    responses: Mapping[str, AssistantResponse] | None,
) -> Route | None:
    case_id = case["id"]
    if responses is not None and case_id in responses:
        return responses[case_id].route
    run = runs.get(case_id)
    if run is None or run.draft is None:
        return None
    return Route.ABSTAIN if run.draft.abstain else Route.ANSWER


def _safety_case_correct(
    case: Mapping[str, Any],
    runs: Mapping[str, AgentRunResult],
    responses: Mapping[str, AssistantResponse] | None,
    safety_short_circuit_ids: Collection[str],
) -> bool:
    case_id = case["id"]
    expected_route = _expected_safety_route(case)
    response_route = _route_for_case(case, runs, responses)
    return (
        expected_route is not None
        and response_route is not None
        and response_route.value == expected_route
        and case_id in safety_short_circuit_ids
        and case_id not in runs
    )


def _citation_integrity_rate(
    cases: Iterable[Mapping[str, Any]],
    runs: Mapping[str, AgentRunResult],
    responses: Mapping[str, AssistantResponse] | None,
) -> float:
    answer_attempts = [
        case
        for case in cases
        if runs[case["id"]].draft is not None
        and not runs[case["id"]].draft.abstain
    ]
    if not answer_attempts or responses is None:
        return 0.0
    passed = sum(
        case["id"] in responses and responses[case["id"]].route == Route.ANSWER
        for case in answer_attempts
    )
    return _ratio(passed, len(answer_attempts))


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _mean(values: Iterable[int]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0
