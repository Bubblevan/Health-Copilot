"""Gold-blind-at-runtime scoring and aggregate product metrics."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from statistics import mean
from typing import Any

from ..research.integration.owned_universe.evaluator_truth import evaluate_structured
from ..research.integration.owned_universe.schema import FactLocation, StructuredAnswerType
from .contracts import CoverageStatus, RouteMode, WorkerRole
from .data import EvaluationRecord
from .runtime import RuntimeExecution

_VALUE_TOKEN = re.compile(r"SYNVAL-[0-9A-F]{10}", re.IGNORECASE)
_NUMBER = re.compile(r"(?<![\w.-])-?\d+(?![\w.])")
_COMPLEX_FAMILIES = frozenset({
    "MEMORY_EXTERNAL_JOIN", "MEMORY_EXTERNAL_CONFLICT", "EXTERNAL_MULTI_SOURCE",
    "MEMORY_MULTI_RECORD", "COMPOSITIONAL_MULTI_FACT",
})
_SIMPLE_FAMILIES = frozenset({"CURRENT_ONLY", "MEMORY_LOOKUP", "EXTERNAL_LOOKUP"})


def score_execution(
    record: EvaluationRecord,
    execution: RuntimeExecution,
    system: str,
) -> dict[str, Any]:
    case = record.case
    truth = record.truth
    observed_ids = tuple(dict.fromkeys(
        citation.source_id for citation in execution.response.citations
    ))
    structured = evaluate_structured(case, execution.response.answer, observed_ids)
    answer_exact = _answer_exact(case, execution.response.answer)
    predicted_values = _extract_values(case, execution.response.answer)
    expected_values = tuple(case.scenario.answer_values)
    answer_coverage = (
        sum(value in predicted_values for value in expected_values) / len(expected_values)
        if expected_values else float(answer_exact)
    )
    required_mem_ids = set(truth.get("required_memory_record_ids", ()))
    required_ext_ids = set(truth.get("required_external_evidence_ids", ()))
    required_fact_ids = set(truth.get("answer_fact_ids", ()))
    memory_values = tuple(
        case.scenario.world.graph.fact_index[fact_id].value
        for fact_id in required_fact_ids
        if (fact_id in case.scenario.world.graph.fact_index
            and case.scenario.world.graph.fact_index[fact_id].location == FactLocation.PATIENT_STATE)
    )
    observed = set(observed_ids)
    memory_fact_coverage = (
        round(sum(value in predicted_values for value in memory_values) / len(memory_values), 6)
        if memory_values else None
    )
    memory_record_coverage = _coverage(required_mem_ids, observed)
    ext_coverage = _coverage(required_ext_ids, observed)
    expected_caps = _expected_capabilities(truth)
    predicted_caps = _predicted_capabilities(execution)
    family = record.scenario_family
    citation_ids = {citation.source_id for citation in execution.response.citations}
    citation_valid = (
        citation_ids.issubset(observed)
        and required_ext_ids.issubset(citation_ids)
    )
    grounding_pass = required_mem_ids.issubset(observed) and required_ext_ids.issubset(observed)
    scenario_metadata = truth.get("structural_evaluator_metadata", {})
    status_counts = defaultdict(int)
    for report in execution.worker_reports:
        status_counts[report.status.value] += 1
    incorporated = sum(
        _report_incorporated(report.answer_text, execution.response.answer)
        for report in execution.worker_reports
        if report.answer_text
    )
    nonempty_reports = sum(bool(report.answer_text) for report in execution.worker_reports)
    useful_workers = sum(_worker_useful(report) for report in execution.worker_reports)
    failed_workers = sum(report.status.value in {"failed", "timed_out"}
                         for report in execution.worker_reports)
    coverage_items = execution.coverage_ledger.items if execution.coverage_ledger else ()
    coverage_counts = {
        status.value: sum(item.status == status for item in coverage_items)
        for status in CoverageStatus
    }
    return {
        "episode_id": record.episode_id,
        "system": system,
        "scenario_family": family,
        "slice_complex": family in _COMPLEX_FAMILIES,
        "slice_simple": family in _SIMPLE_FAMILIES,
        "task_success": bool(structured.success),
        "grounded_task_success": bool(structured.success and grounding_pass and citation_valid),
        "answer_accuracy": answer_exact,
        "required_fact_coverage": round(answer_coverage, 6),
        "external_evidence_coverage": ext_coverage,
        "patient_state_fact_coverage": memory_fact_coverage,
        "patient_state_record_coverage": memory_record_coverage,
        "grounding_pass": grounding_pass,
        "correct_abstention": (
            not case.scenario.oracle.answerability
            and execution.response.answer.strip() == "INSUFFICIENT_EVIDENCE"
        ),
        "citation_validity": bool(citation_valid),
        "route_mode": execution.response.route_mode.value,
        "initial_route_mode": execution.route_decision.mode.value,
        "route_reason": execution.route_decision.reason,
        "triage_decision": (execution.triage_decision.to_dict()
                            if execution.triage_decision is not None else None),
        "triage_provider": (execution.triage_decision.provider
                            if execution.triage_decision is not None else None),
        "triage_input_tokens": (execution.triage_decision.input_tokens
                                if execution.triage_decision is not None else 0),
        "triage_output_tokens": (execution.triage_decision.output_tokens
                                 if execution.triage_decision is not None else 0),
        "triage_latency_ms": (execution.triage_decision.latency_ms
                              if execution.triage_decision is not None else 0.0),
        "triage_cost_usd": (execution.triage_decision.cost_usd
                            if execution.triage_decision is not None else None),
        "expected_capabilities": sorted(role.value for role in expected_caps),
        "predicted_capabilities": sorted(role.value for role in predicted_caps),
        "exact_worker_set_match": expected_caps == predicted_caps,
        "route_precision_numerator": len(expected_caps & predicted_caps),
        "route_precision_denominator": len(predicted_caps),
        "route_recall_numerator": len(expected_caps & predicted_caps),
        "route_recall_denominator": len(expected_caps),
        "over_activation_count": len(predicted_caps - expected_caps),
        "under_activation_count": len(expected_caps - predicted_caps),
        **_route_usage_flags(execution),
        "repair_wave_used": bool(execution.repair_wave.get("invoked")),
        "coverage_aspect_counts": coverage_counts,
        "coverage_ledger_covered_rate": (
            sum(item.status == CoverageStatus.COVERED for item in coverage_items) / len(coverage_items)
            if coverage_items else None
        ),
        "workers_assigned": len(execution.worker_reports),
        "workers_completed": sum(report.status.value == "complete"
                                  for report in execution.worker_reports),
        "workers_useful": useful_workers,
        "worker_status_counts": dict(status_counts),
        "lead_incorporated_workers": incorporated,
        "nonempty_worker_reports": nonempty_reports,
        "worker_completion_rate_numerator": sum(
            report.status.value in {"complete", "partial"} for report in execution.worker_reports
        ),
        "worker_completion_rate_denominator": len(execution.worker_reports),
        "useful_worker_rate_numerator": useful_workers,
        "useful_worker_rate_denominator": len(execution.worker_reports),
        "lead_incorporation_numerator": incorporated,
        "lead_incorporation_denominator": nonempty_reports,
        "parallel_speedup": (
            execution.sequential_worker_latency_ms / execution.worker_wave_wall_ms
            if execution.worker_wave_wall_ms > 0 else None
        ),
        "partial_failure_recovered": bool(
            failed_workers > 0 and (execution.partial_failure_recovered or execution.response.answer)
        ),
        "failed_worker_count": failed_workers,
        "provider_calls": execution.provider_calls,
        "tool_calls": execution.tool_calls,
        "input_tokens": execution.input_tokens,
        "output_tokens": execution.output_tokens,
        "total_tokens": execution.input_tokens + execution.output_tokens,
        "latency_ms": execution.response.latency_ms,
        "worker_wave_wall_ms": round(execution.worker_wave_wall_ms, 3),
        "sequential_worker_latency_ms": round(execution.sequential_worker_latency_ms, 3),
        "workers_used": list(execution.response.workers_used),
        "trace_id": execution.response.trace_id,
        "answer": execution.response.answer,
        "observed_resource_count": len(observed),
        "expected_resource_count": len(required_mem_ids | required_ext_ids),
        "structural_metadata": scenario_metadata,
    }


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"episode_count": 0}
    values: dict[str, Any] = {"episode_count": len(rows)}
    for key in (
        "task_success", "grounded_task_success", "answer_accuracy", "grounding_pass",
        "correct_abstention", "citation_validity", "single_fast_path",
        "router_team_candidate", "team_activated", "repair_wave_used",
    ):
        selected = [bool(row[key]) for row in rows]
        values[f"{key}_numerator"] = sum(selected)
        values[f"{key}_denominator"] = len(selected)
        values[key] = round(mean(selected), 6)
    for key in (
        "required_fact_coverage", "external_evidence_coverage", "patient_state_fact_coverage",
        "patient_state_record_coverage",
        "parallel_speedup",
        "coverage_ledger_covered_rate",
    ):
        selected = [float(row[key]) for row in rows if row[key] is not None]
        values[key] = round(mean(selected), 6) if selected else None
        values[f"{key}_denominator"] = len(selected)
    values.update(_micro_set_metrics(rows))
    assigned = sum(row["worker_completion_rate_denominator"] for row in rows)
    values["worker_completion_rate"] = (
        round(sum(row["worker_completion_rate_numerator"] for row in rows) / assigned, 6)
        if assigned else None
    )
    useful_denominator = sum(row["useful_worker_rate_denominator"] for row in rows)
    values["useful_worker_rate"] = (
        round(sum(row["useful_worker_rate_numerator"] for row in rows) / useful_denominator, 6)
        if useful_denominator else None
    )
    incorporated_denominator = sum(row["lead_incorporation_denominator"] for row in rows)
    values["lead_incorporation_rate"] = (
        round(sum(row["lead_incorporation_numerator"] for row in rows) / incorporated_denominator, 6)
        if incorporated_denominator else None
    )
    team_rows = [row for row in rows if row["team_activated"]]
    values["average_workers_per_team_request"] = (
        round(mean(row["workers_assigned"] for row in team_rows), 6) if team_rows else 0.0
    )
    failure_rows = [row for row in rows if row["failed_worker_count"]]
    values["partial_failure_recovery_rate"] = (
        round(mean(row["partial_failure_recovered"] for row in failure_rows), 6)
        if failure_rows else None
    )
    values["partial_failure_recovery_denominator"] = len(failure_rows)
    latency = [int(row["latency_ms"]) for row in rows]
    values["mean_latency_ms"] = round(mean(latency), 3)
    values["p50_latency_ms"] = _percentile(latency, 0.50)
    values["p95_latency_ms"] = _percentile(latency, 0.95)
    values["provider_calls"] = sum(int(row["provider_calls"]) for row in rows)
    values["tool_calls"] = sum(int(row["tool_calls"]) for row in rows)
    values["input_tokens"] = sum(int(row["input_tokens"]) for row in rows)
    values["output_tokens"] = sum(int(row["output_tokens"]) for row in rows)
    values["total_tokens"] = values["input_tokens"] + values["output_tokens"]
    triage_costs = [float(row["triage_cost_usd"]) for row in rows
                    if row["triage_cost_usd"] is not None]
    values["triage_cost_usd"] = round(sum(triage_costs), 8) if triage_costs else None
    values["triage_cost_reported_calls"] = len(triage_costs)
    values["triage_input_tokens"] = sum(int(row["triage_input_tokens"]) for row in rows)
    values["triage_output_tokens"] = sum(int(row["triage_output_tokens"]) for row in rows)
    values["triage_mean_latency_ms"] = round(mean(
        float(row["triage_latency_ms"]) for row in rows
    ), 3)
    values["triage_provider_counts"] = dict(Counter(
        str(row["triage_provider"]) for row in rows if row["triage_provider"] is not None
    ))
    return values


def compare_systems(single: dict[str, Any], team: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"strong_single": single, "routed_medical_team": team, "delta": {}}
    for metric in (
        "task_success", "grounded_task_success", "required_fact_coverage", "answer_accuracy",
        "external_evidence_coverage", "patient_state_fact_coverage",
    ):
        left = single.get(metric)
        right = team.get(metric)
        result["delta"][metric] = round(right - left, 6) if left is not None and right is not None else None
    return result


def _micro_set_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    intersection = sum(row["route_precision_numerator"] for row in rows)
    predicted = sum(row["route_precision_denominator"] for row in rows)
    expected = sum(row["route_recall_denominator"] for row in rows)
    return {
        "route_precision_numerator": intersection,
        "route_precision_denominator": predicted,
        "route_precision": round(intersection / predicted, 6) if predicted else None,
        "route_recall_numerator": intersection,
        "route_recall_denominator": expected,
        "route_recall": round(intersection / expected, 6) if expected else None,
        "exact_worker_set_match_numerator": sum(row["exact_worker_set_match"] for row in rows),
        "exact_worker_set_match_denominator": len(rows),
        "exact_worker_set_match": round(mean(row["exact_worker_set_match"] for row in rows), 6),
        "over_activation_count": sum(row["over_activation_count"] for row in rows),
        "under_activation_count": sum(row["under_activation_count"] for row in rows),
        "over_activation_rate": round(mean(row["over_activation_count"] > 0 for row in rows), 6),
        "under_activation_rate": round(mean(row["under_activation_count"] > 0 for row in rows), 6),
    }


def _expected_capabilities(truth: dict[str, Any]) -> frozenset[WorkerRole]:
    oracle = truth.get("capability_requirement_oracle", {})
    expected = set()
    if oracle.get("memory_required"):
        expected.add(WorkerRole.PATIENT_CONTEXT)
    if oracle.get("external_retrieval_required"):
        expected.add(WorkerRole.EVIDENCE)
    if not oracle.get("answerability", True) and not expected:
        expected.add(WorkerRole.CARE)
    return frozenset(expected)


def _predicted_capabilities(execution: RuntimeExecution) -> frozenset[WorkerRole]:
    if execution.triage_decision is not None:
        return frozenset(execution.route_decision.predicted_capabilities)
    if execution.response.route_mode == RouteMode.TEAM and execution.plan is not None:
        return frozenset(role for role, _objective in execution.plan.tasks)
    if execution.route_decision.mode == RouteMode.SINGLE:
        return frozenset(execution.route_decision.predicted_capabilities)
    return frozenset()


def _route_usage_flags(execution: RuntimeExecution) -> dict[str, bool]:
    return {
        "single_fast_path": execution.route_decision.mode == RouteMode.SINGLE,
        "router_team_candidate": execution.route_decision.mode == RouteMode.TEAM,
        "team_activated": execution.response.route_mode == RouteMode.TEAM,
    }


def _answer_exact(case: EvaluationRecord | Any, answer: str) -> bool:
    scenario = case.case.scenario if isinstance(case, EvaluationRecord) else case.scenario
    if not scenario.oracle.answerability:
        return answer.strip() == "INSUFFICIENT_EVIDENCE"
    expected = scenario.answer_values
    if scenario.world.answer_type == StructuredAnswerType.BOOLEAN:
        observed = tuple(value.upper() for value in re.findall(
            r"\b(?:TRUE|FALSE)\b", answer, re.IGNORECASE
        ))
    elif expected and all(value in {"UP", "DOWN", "STABLE"} for value in expected):
        observed = tuple(value.upper() for value in re.findall(
            r"\b(?:UP|DOWN|STABLE)\b", answer, re.IGNORECASE
        ))
    else:
        observed = tuple(_VALUE_TOKEN.findall(answer))
        if expected and any(value.isdigit() for value in expected):
            observed = (*observed, *tuple(_NUMBER.findall(answer)))
    if scenario.world.answer_type == StructuredAnswerType.ORDERED_SEQUENCE:
        return len(observed) == len(expected) and observed == expected
    if scenario.world.answer_type == StructuredAnswerType.EXACT_TOKEN:
        return len(expected) == 1 and observed == expected
    return set(observed) == set(expected) and len(observed) == len(set(observed))


def _extract_values(case, answer: str) -> tuple[str, ...]:
    expected = case.scenario.answer_values
    if not case.scenario.oracle.answerability:
        return ("INSUFFICIENT_EVIDENCE",) if answer.strip() == "INSUFFICIENT_EVIDENCE" else ()
    if case.scenario.world.answer_type == StructuredAnswerType.BOOLEAN:
        return tuple(value.upper() for value in re.findall(
            r"\b(?:TRUE|FALSE)\b", answer, re.IGNORECASE
        ))
    if expected and all(value in {"UP", "DOWN", "STABLE"} for value in expected):
        return tuple(value.upper() for value in re.findall(
            r"\b(?:UP|DOWN|STABLE)\b", answer, re.IGNORECASE
        ))
    values = tuple(_VALUE_TOKEN.findall(answer))
    if expected and any(value.isdigit() for value in expected):
        values = (*values, *tuple(_NUMBER.findall(answer)))
    return values


def _coverage(required: set[str], observed: set[str]) -> float | None:
    return round(len(required & observed) / len(required), 6) if required else None


def _worker_useful(report) -> bool:
    if report.status.value not in {"complete", "partial"}:
        return False
    if report.observed_evidence_ids or _VALUE_TOKEN.search(report.answer_text):
        return True
    return report.role == WorkerRole.CARE and bool(report.answer_text.strip())


def _report_incorporated(worker_answer: str, final_answer: str) -> bool:
    tokens = set(_VALUE_TOKEN.findall(worker_answer))
    if tokens:
        return bool(tokens & set(_VALUE_TOKEN.findall(final_answer)))
    normalized = {token for token in re.findall(r"[A-Za-z0-9_-]{4,}", worker_answer.casefold())}
    final = set(re.findall(r"[A-Za-z0-9_-]{4,}", final_answer.casefold()))
    return bool(normalized & final)


def _percentile(values: Iterable[int], fraction: float) -> float:
    rows = sorted(values)
    if not rows:
        return 0.0
    index = min(len(rows) - 1, max(0, int((len(rows) - 1) * fraction)))
    return float(rows[index])
