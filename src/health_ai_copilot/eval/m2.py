"""Stage-aware M2 metric helpers; no metric is medical accuracy."""

from collections.abc import Iterable, Mapping
from typing import Any

from ..agent.loop import AgentRunResult
from ..contracts import AssistantResponse, Route


def summarize_m2_runs(
    cases: Iterable[Mapping[str, Any]],
    runs: Mapping[str, AgentRunResult],
    responses: Mapping[str, AssistantResponse],
) -> dict[str, float | int]:
    rows = list(cases)
    run_rows = [row for row in rows if row.get("id") in runs]
    run_list = [runs[row["id"]] for row in run_rows]
    ood = [row for row in rows if row.get("category") == "ood_false_retrieval"]
    answerable = [row for row in rows if row.get("expected_route") == "answer"]
    direct = [row for row in rows if row.get("category") == "direct_hit"]
    route = lambda row: responses.get(row["id"], AssistantResponse(Route.ABSTAIN, "")).route
    proposals = lambda row: runs.get(row["id"]).state.tool_proposals_used if row.get("id") in runs else 0
    executions = lambda row: runs.get(row["id"]).state.tool_calls_used if row.get("id") in runs else 0
    vetoes = sum(proposals(row) > executions(row) for row in run_rows)
    policy_truth = [row for row in rows if row.get("expected_decision")]
    policy_correct = sum(
        getattr(runs.get(row["id"]), "state", None) is not None
        and getattr(runs[row["id"]].state, "policy_decision", None) == row["expected_decision"]
        for row in policy_truth
    )
    return {
        "run_cases": len(run_list),
        "policy_accuracy": _ratio(policy_correct, len(policy_truth)),
        "tool_proposal_rate": _ratio(sum(proposals(row) > 0 for row in run_rows), len(run_rows)),
        "tool_execution_rate": _ratio(sum(executions(row) > 0 for row in run_rows), len(run_rows)),
        "policy_veto_rate": _ratio(vetoes, len(run_rows)),
        "ood_tool_proposal_rate": _ratio(sum(proposals(row) > 0 for row in ood), len(ood)),
        "ood_tool_execution_rate": _ratio(sum(executions(row) > 0 for row in ood), len(ood)),
        "ood_answer_rate": _ratio(sum(route(row) == Route.ANSWER for row in ood), len(ood)),
        "ood_abstain_rate": _ratio(sum(route(row) == Route.ABSTAIN for row in ood), len(ood)),
        "expected_answer_rate": _ratio(sum(route(row) == Route.ANSWER for row in answerable), len(answerable)),
        "unexpected_abstain_rate": _ratio(sum(route(row) == Route.ABSTAIN for row in answerable), len(answerable)),
        "direct_hit_policy_false_veto_rate": _ratio(sum(proposals(row) > executions(row) for row in direct), len(direct)),
        "mean_model_turns": _mean(run.state.model_turns_used for run in run_list),
        "mean_tool_proposals": _mean(run.state.tool_proposals_used for run in run_list),
        "mean_tool_executions": _mean(run.state.tool_calls_used for run in run_list),
        "mean_policy_calls": _mean(run.state.policy_calls_used for run in run_list),
        "mean_verifier_calls": _mean(run.state.verifier_calls_used for run in run_list),
        "budget_exhaustion_rate": _ratio(sum(run.stop_reason and run.stop_reason.value in {"max_model_turns", "max_tool_calls"} for run in run_list), len(run_list)),
        "grounding_rejection_rate": _ratio(sum(response.safety_reasons == ["grounding_failed"] for response in responses.values()), len(run_rows)),
    }


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _mean(values: Iterable[int]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0
