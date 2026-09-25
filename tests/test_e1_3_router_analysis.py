import pytest

from eval.e1_3_learned_router import validate_oof_prediction_coverage
from tools.analyze_e1_3_router import pareto_frontier, relative_cost, summarize_rows


def test_relative_cost_reports_signed_change_and_positive_savings() -> None:
    candidate = {
        "accuracy": 0.6204,
        "retrieval_calls": 108,
        "answer_input_tokens": 1_237_336,
        "component_latency_proxy_p95_ms": 431.0,
    }
    baseline = {
        "accuracy": 0.6223,
        "retrieval_calls": 2_311,
        "answer_input_tokens": 3_640_366,
        "component_latency_proxy_p95_ms": 2_267.3,
    }

    delta = relative_cost(candidate, baseline)["delta"]

    assert delta["accuracy_pp"] < 0
    assert delta["retrieval_calls_percent_change"] < -95
    assert delta["retrieval_calls_reduction_percent"] > 95
    assert delta["answer_input_tokens_percent_change"] < -65
    assert delta["answer_input_tokens_reduction_percent"] > 65
    assert delta["component_latency_proxy_p95_percent_change"] < 0


def test_pareto_frontier_requires_one_strict_improvement() -> None:
    metrics = {
        "closed": {"accuracy": 0.6, "retrieval_rate": 0.0},
        "cheap": {"accuracy": 0.62, "retrieval_rate": 0.4},
        "dominated": {"accuracy": 0.6, "retrieval_rate": 0.4},
        "tie": {"accuracy": 0.6, "retrieval_rate": 0.0},
    }

    assert pareto_frontier(metrics) == ["cheap", "closed", "tie"]


def test_policy_metric_summary_keeps_fixed_denominator_and_missing_cost_coverage() -> None:
    rows = [
        {
            "selected_arm_correct": True,
            "retrieval_calls": 0,
            "answer_input_tokens": 100,
            "context_characters": 0,
            "component_latency_proxy_ms": 20,
        },
        {
            "selected_arm_correct": False,
            "retrieval_calls": 1,
            "answer_input_tokens": None,
            "context_characters": 10,
            "component_latency_proxy_ms": 40,
        },
    ]

    metrics = summarize_rows(rows)

    assert metrics["cases"] == 2
    assert metrics["accuracy"] == 0.5
    assert metrics["retrieval_calls"] == 1
    assert metrics["answer_input_tokens"] == 100
    assert metrics["answer_input_token_measurement_coverage"] == 0.5


def test_oof_coverage_rejects_duplicate_case_policy_rows() -> None:
    rows = [
        {"case_id": "case-1", "policy": "tfidf_direct"},
        {"case_id": "case-1", "policy": "tfidf_direct"},
    ]

    with pytest.raises(ValueError, match="Duplicate OOF prediction"):
        validate_oof_prediction_coverage(rows, ["case-1"])
