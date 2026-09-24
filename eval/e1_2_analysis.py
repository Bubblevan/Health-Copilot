"""Held-out paired analysis and preregistered headline gate for E1.2."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from eval.e1_2_runner import summarize_results

FIXED_ARMS = ("closed_book", "rag_bm25", "rag_medcpt")
ADAPTIVE_ARMS = ("cheap_router", "jev_router")
HEADLINE_ARMS = ("jev_router",)
COST_REFERENCE_ARMS = ("rag_bm25", "rag_medcpt")


def _case_map(rows: list[dict[str, Any]], arm: str) -> dict[str, dict[str, Any]]:
    result = {str(row.get("case_id")): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate case IDs in {arm}")
    return result


def paired_bootstrap_difference(
    candidate: Mapping[str, Mapping[str, Any]],
    baseline: Mapping[str, Mapping[str, Any]],
    *,
    subset_by_case: Mapping[str, str],
    resamples: int = 10_000,
    seed: int = 120_026,
) -> dict[str, float]:
    if set(candidate) != set(baseline) or set(candidate) != set(subset_by_case):
        raise ValueError("paired bootstrap requires identical case IDs and subset mappings")
    if resamples < 100:
        raise ValueError("at least 100 paired resamples are required")
    import numpy as np

    deltas_by_subset: dict[str, Any] = {}
    for subset in sorted(set(subset_by_case.values())):
        case_ids = [case_id for case_id, value in subset_by_case.items() if value == subset]
        deltas_by_subset[subset] = np.asarray(
            [
                int(candidate[case_id].get("is_correct") is True)
                - int(baseline[case_id].get("is_correct") is True)
                for case_id in case_ids
            ],
            dtype=np.int8,
        )
    total = len(candidate)
    rng = np.random.default_rng(seed)
    draws: list[float] = []
    batch_size = 250
    for start in range(0, resamples, batch_size):
        batch_count = min(batch_size, resamples - start)
        sums = np.zeros(batch_count, dtype=np.int64)
        for deltas in deltas_by_subset.values():
            sample_indices = rng.integers(0, len(deltas), size=(batch_count, len(deltas)))
            sums += deltas[sample_indices].sum(axis=1)
        draws.extend((sums / total).tolist())
    ordered = sorted(float(value) for value in draws)
    lower = ordered[int(0.025 * resamples)]
    upper = ordered[min(resamples - 1, int(0.975 * resamples))]
    return {"lower_95": lower, "upper_95": upper, "resamples": resamples}


def route_error_analysis(
    fixed: Mapping[str, Mapping[str, Mapping[str, Any]]],
    adaptive: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    case_ids = set(fixed["closed_book"])
    if any(set(fixed[name]) != case_ids for name in FIXED_ARMS[1:]) or set(adaptive) != case_ids:
        raise ValueError("route analysis requires identical fixed and adaptive case IDs")
    opportunity = 0
    routed = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0
    regret_sum = 0
    oracle_correct = 0
    uplift_sum = 0
    for case_id in case_ids:
        closed_correct = fixed["closed_book"][case_id].get("is_correct") is True
        retrieval_correct = any(
            fixed[name][case_id].get("is_correct") is True for name in ("rag_bm25", "rag_medcpt")
        )
        has_retrieval_opportunity = retrieval_correct and not closed_correct
        action = str(adaptive[case_id].get("selected_action", "closed_book"))
        chose_retrieval = action != "closed_book"
        router_correct = adaptive[case_id].get("is_correct") is True
        best_fixed_correct = any(fixed[name][case_id].get("is_correct") is True for name in FIXED_ARMS)
        opportunity += int(has_retrieval_opportunity)
        routed += int(chose_retrieval)
        true_positive += int(has_retrieval_opportunity and chose_retrieval)
        false_positive += int(not has_retrieval_opportunity and chose_retrieval)
        false_negative += int(has_retrieval_opportunity and not chose_retrieval)
        true_negative += int(not has_retrieval_opportunity and not chose_retrieval)
        oracle_correct += int(best_fixed_correct)
        regret_sum += max(int(best_fixed_correct) - int(router_correct), 0)
        uplift_sum += int(router_correct) - int(closed_correct)
    count = len(case_ids)
    return {
        "cases": count,
        "retrieval_opportunity_cases": opportunity,
        "router_retrieval_cases": routed,
        "opportunity_precision": true_positive / routed if routed else None,
        "opportunity_recall": true_positive / opportunity if opportunity else None,
        "false_positive_rate": false_positive / (false_positive + true_negative)
        if false_positive + true_negative
        else None,
        "false_negative_rate": false_negative / (false_negative + true_positive)
        if false_negative + true_positive
        else None,
        "posthoc_fixed_arm_oracle_accuracy": oracle_correct / count if count else None,
        "mean_nonnegative_regret_vs_fixed_oracle": regret_sum / count if count else None,
        "mean_accuracy_uplift_vs_closed_book": uplift_sum / count if count else None,
        "label_use": "analysis only; never supplied to router",
    }


def _arm_cost(summary: Mapping[str, Any]) -> dict[str, float | None]:
    cases = int(summary.get("cases", 0))
    if not cases:
        return {
            "retrieval_calls_per_case": None,
            "answer_input_tokens_per_case": None,
            "p95_component_latency_proxy_ms": None,
        }
    latency = summary.get("p95_component_latency_proxy_ms")
    latency_coverage = summary.get("component_latency_proxy_measurement_coverage")
    token_total = summary.get("answer_input_tokens")
    token_coverage = summary.get("answer_input_token_measurement_coverage")
    return {
        "retrieval_calls_per_case": float(summary.get("retrieval_calls", 0)) / cases,
        "answer_input_tokens_per_case": (
            float(token_total) / cases
            if token_total is not None and token_coverage == 1.0
            else None
        ),
        "p95_component_latency_proxy_ms": (
            float(latency) if latency is not None and latency_coverage == 1.0 else None
        ),
    }


def select_dev_retrieval_cost_reference(
    dev_arm_rows: Mapping[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    if not set(COST_REFERENCE_ARMS).issubset(dev_arm_rows):
        raise ValueError("DEV cost-reference selection requires both fixed retrieval arms")
    maps = {name: _case_map(dev_arm_rows[name], name) for name in COST_REFERENCE_ARMS}
    if set(maps[COST_REFERENCE_ARMS[0]]) != set(maps[COST_REFERENCE_ARMS[1]]):
        raise ValueError("DEV cost-reference selection requires paired fixed-arm cases")
    summaries = {name: summarize_results(dev_arm_rows[name]) for name in COST_REFERENCE_ARMS}
    selected = max(
        COST_REFERENCE_ARMS,
        key=lambda name: float(summaries[name]["accuracy_fixed_denominator"]),
    )
    return {
        "selected_retrieval_cost_reference": selected,
        "selection_metric": "DEV fixed-denominator exact-answer accuracy",
        "selection_rule": "Choose the higher-accuracy fixed retrieval arm on DEV; ties favor rag_bm25.",
        "dev_case_count": len(maps[selected]),
        "dev_arm_summaries": summaries,
    }


def analyze_arms(
    arm_rows: Mapping[str, list[dict[str, Any]]],
    *,
    dev_selected_retrieval_cost_reference: str,
    resamples: int = 10_000,
) -> dict[str, Any]:
    required = set(FIXED_ARMS) | {"random_context", "cheap_router"}
    if not required.issubset(arm_rows):
        raise ValueError(f"missing required arms: {sorted(required - set(arm_rows))}")
    if dev_selected_retrieval_cost_reference not in COST_REFERENCE_ARMS:
        raise ValueError("cost reference must be selected from the fixed DEV retrieval arms")
    maps = {name: _case_map(rows, name) for name, rows in arm_rows.items()}
    case_ids = set(maps["closed_book"])
    if any(set(rows) != case_ids for rows in maps.values()):
        raise ValueError("all analyzed arms must contain the same frozen TEST case IDs")
    subset_by_case = {
        case_id: str(maps["closed_book"][case_id].get("subdataset")) for case_id in case_ids
    }
    if any(
        {
            case_id: str(rows[case_id].get("subdataset"))
            for case_id in case_ids
        }
        != subset_by_case
        for rows in maps.values()
    ):
        raise ValueError("all analyzed arms must preserve the frozen subdataset per case")
    summaries = {name: summarize_results(rows) for name, rows in arm_rows.items()}
    paired: dict[str, Any] = {}
    for adaptive_name in ADAPTIVE_ARMS:
        if adaptive_name not in maps:
            continue
        paired[adaptive_name] = {
            fixed_name: {
                "accuracy_difference": summaries[adaptive_name]["accuracy_fixed_denominator"]
                - summaries[fixed_name]["accuracy_fixed_denominator"],
                "paired_bootstrap_95": paired_bootstrap_difference(
                    maps[adaptive_name],
                    maps[fixed_name],
                    subset_by_case=subset_by_case,
                    resamples=resamples,
                ),
            }
            for fixed_name in FIXED_ARMS
        }

    fixed_accuracy = {
        name: float(summaries[name]["accuracy_fixed_denominator"]) for name in FIXED_ARMS
    }
    best_fixed = max(FIXED_ARMS, key=lambda name: fixed_accuracy[name])
    best_fixed_rag = dev_selected_retrieval_cost_reference
    headline: dict[str, Any] = {}
    for adaptive_name in HEADLINE_ARMS:
        if adaptive_name not in summaries:
            continue
        adaptive_accuracy = float(summaries[adaptive_name]["accuracy_fixed_denominator"])
        intervals = paired[adaptive_name]
        quality_win = all(
            adaptive_accuracy > fixed_accuracy[baseline]
            and intervals[baseline]["paired_bootstrap_95"]["lower_95"] > 0
            for baseline in FIXED_ARMS
        )
        reference_cost = _arm_cost(summaries[best_fixed_rag])
        adaptive_cost = _arm_cost(summaries[adaptive_name])
        reference_p95 = reference_cost["p95_component_latency_proxy_ms"]
        adaptive_p95 = adaptive_cost["p95_component_latency_proxy_ms"]
        reference_retrieval = reference_cost["retrieval_calls_per_case"]
        adaptive_retrieval = adaptive_cost["retrieval_calls_per_case"]
        reference_tokens = reference_cost["answer_input_tokens_per_case"]
        adaptive_tokens = adaptive_cost["answer_input_tokens_per_case"]
        proxy_latency_reduction = (
            reference_p95 is not None
            and adaptive_p95 is not None
            and reference_p95 > 0
            and adaptive_p95 <= 0.85 * reference_p95
        )
        cost_reductions = {
            "retrieval_calls_at_least_25_percent": (
                reference_retrieval is not None
                and adaptive_retrieval is not None
                and reference_retrieval > 0
                and adaptive_retrieval <= 0.75 * reference_retrieval
            ),
            "answer_input_tokens_at_least_20_percent": (
                reference_tokens is not None
                and adaptive_tokens is not None
                and reference_tokens > 0
                and adaptive_tokens <= 0.80 * reference_tokens
            ),
            "component_latency_proxy_p95_at_least_15_percent": proxy_latency_reduction,
        }
        tradeoff = (
            adaptive_accuracy >= fixed_accuracy[best_fixed] - 0.005
            and any(cost_reductions.values())
        )
        headline[adaptive_name] = {
            "eligible": quality_win or tradeoff,
            "quality_win": quality_win,
            "quality_cost_tradeoff": tradeoff,
            "best_fixed_test_accuracy_arm": best_fixed,
            "dev_selected_retrieval_cost_reference": best_fixed_rag,
            "cost_reductions_vs_best_accuracy_fixed_retrieval": cost_reductions,
        }

    fixed = {name: maps[name] for name in FIXED_ARMS}
    routes = {
        name: route_error_analysis(fixed, maps[name])
        for name in ADAPTIVE_ARMS
        if name in maps
    }
    return {
        "schema_version": "e1-2-test-analysis-v1",
        "case_count": len(case_ids),
        "arm_summaries": summaries,
        "paired_differences": paired,
        "route_analysis": routes,
        "headline_eligibility": headline,
        "latency_semantics": "Component-summed latency proxy; retrieval/answer/router components were timed in isolated phases, not as a single production end-to-end request.",
        "bootstrap_semantics": "Paired resampling within each clean subdataset; fixed subset sizes; the same sampled case indices are used for each arm pair.",
    }


def load_completed_test_arms(
    scratch_root: Path,
    config_sha256: str,
    expected_case_count: int,
) -> dict[str, list[dict[str, Any]]]:
    test_root = scratch_root / "runs" / "e1_2" / "test"
    result: dict[str, list[dict[str, Any]]] = {}
    for arm in (*FIXED_ARMS, "random_context", *ADAPTIVE_ARMS):
        manifest_path = test_root / arm / "manifest.json"
        rows_path = test_root / arm / "case_results.jsonl"
        if not manifest_path.is_file() or not rows_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("status") != "COMPLETED"
            or manifest.get("partition") != "TEST"
            or manifest.get("arm") != arm
            or manifest.get("config_sha256") != config_sha256
            or manifest.get("case_count") != expected_case_count
        ):
            raise ValueError(f"test arm {arm} is incomplete or belongs to another config")
        identity = manifest.get("result_identity")
        rows = [
            json.loads(line)
            for line in rows_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not isinstance(identity, str) or not identity or len(rows) != expected_case_count or any(
            row.get("result_identity") != identity
            or row.get("partition") != "TEST"
            or row.get("arm") != arm
            for row in rows
        ):
            raise ValueError(f"test arm {arm} rows do not match its completed manifest")
        result[arm] = rows
    return result


__all__ = [
    "ADAPTIVE_ARMS",
    "COST_REFERENCE_ARMS",
    "FIXED_ARMS",
    "HEADLINE_ARMS",
    "analyze_arms",
    "load_completed_test_arms",
    "paired_bootstrap_difference",
    "route_error_analysis",
    "select_dev_retrieval_cost_reference",
]
