import json

import pytest

from eval.e1_2_analysis import ADAPTIVE_ARMS, FIXED_ARMS, analyze_arms, load_completed_test_arms
from eval.e1_2_runner import summarize_results


def _rows(*, correct: list[bool], retrieval_calls: int, answer_tokens: int) -> list[dict]:
    return [
        {
            "case_id": f"case-{index}",
            "subdataset": "medqa" if index % 2 == 0 else "mmlu",
            "status": "completed",
            "is_correct": value,
            "retrieval_calls": retrieval_calls,
            "answer_input_tokens": answer_tokens,
            "component_latency_proxy_ms": None,
        }
        for index, value in enumerate(correct)
    ]


def _arms(*, jev_correct: list[bool], cheap_correct: list[bool]) -> dict[str, list[dict]]:
    fixed = [False, False, False, False]
    return {
        "closed_book": _rows(correct=fixed, retrieval_calls=0, answer_tokens=10),
        "rag_bm25": _rows(correct=fixed, retrieval_calls=1, answer_tokens=10),
        "rag_medcpt": _rows(correct=fixed, retrieval_calls=1, answer_tokens=10),
        "random_context": _rows(correct=fixed, retrieval_calls=1, answer_tokens=10),
        "cheap_router": _rows(correct=cheap_correct, retrieval_calls=1, answer_tokens=10),
        "jev_router": _rows(correct=jev_correct, retrieval_calls=1, answer_tokens=10),
    }


def test_random_context_is_required_for_analysis():
    arms = _arms(jev_correct=[False] * 4, cheap_correct=[False] * 4)
    del arms["random_context"]

    with pytest.raises(ValueError, match="random_context"):
        analyze_arms(
            arms,
            dev_selected_retrieval_cost_reference="rag_bm25",
            resamples=100,
        )


def test_only_jev_is_headline_eligible_and_component_proxy_is_named_honestly():
    arms = _arms(jev_correct=[True] * 4, cheap_correct=[True] * 4)

    analysis = analyze_arms(
        arms,
        dev_selected_retrieval_cost_reference="rag_bm25",
        resamples=100,
    )

    assert set(analysis["headline_eligibility"]) == {"jev_router"}
    assert analysis["headline_eligibility"]["jev_router"]["quality_win"] is True
    assert "component_latency_proxy_p95_at_least_15_percent" in analysis[
        "headline_eligibility"
    ]["jev_router"]["cost_reductions_vs_best_accuracy_fixed_retrieval"]
    assert analysis["latency_semantics"].startswith("Component-summed latency proxy")


def test_missing_latency_proxy_does_not_create_a_cost_win():
    arms = _arms(jev_correct=[False] * 4, cheap_correct=[False] * 4)

    analysis = analyze_arms(
        arms,
        dev_selected_retrieval_cost_reference="rag_bm25",
        resamples=100,
    )

    gate = analysis["headline_eligibility"]["jev_router"]
    assert gate["eligible"] is False
    assert gate["cost_reductions_vs_best_accuracy_fixed_retrieval"][
        "component_latency_proxy_p95_at_least_15_percent"
    ] is False


def test_partial_latency_measurements_do_not_create_a_cost_win():
    arms = _arms(jev_correct=[False] * 4, cheap_correct=[False] * 4)
    for row in arms["rag_bm25"]:
        row["component_latency_proxy_ms"] = 1000
    arms["jev_router"][0]["component_latency_proxy_ms"] = 1

    analysis = analyze_arms(
        arms,
        dev_selected_retrieval_cost_reference="rag_bm25",
        resamples=100,
    )

    gate = analysis["headline_eligibility"]["jev_router"]
    assert gate["cost_reductions_vs_best_accuracy_fixed_retrieval"][
        "component_latency_proxy_p95_at_least_15_percent"
    ] is False
    assert gate["eligible"] is False


def test_analysis_rejects_case_subdataset_mismatch_between_arms():
    arms = _arms(jev_correct=[False] * 4, cheap_correct=[False] * 4)
    arms["rag_medcpt"][0]["subdataset"] = "unexpected"

    with pytest.raises(ValueError, match="subdataset per case"):
        analyze_arms(
            arms,
            dev_selected_retrieval_cost_reference="rag_bm25",
            resamples=100,
        )


def test_missing_token_measurements_do_not_count_as_zero_cost():
    arms = _arms(jev_correct=[False] * 4, cheap_correct=[False] * 4)
    for rows in arms.values():
        for row in rows:
            row.pop("answer_input_tokens")

    analysis = analyze_arms(
        arms,
        dev_selected_retrieval_cost_reference="rag_bm25",
        resamples=100,
    )

    gate = analysis["headline_eligibility"]["jev_router"]
    assert gate["cost_reductions_vs_best_accuracy_fixed_retrieval"][
        "answer_input_tokens_at_least_20_percent"
    ] is False
    assert gate["eligible"] is False
    assert gate["eligible"] is False


def test_dev_cost_reference_selection_is_paired_and_ties_favor_bm25():
    from eval.e1_2_analysis import select_dev_retrieval_cost_reference

    rows = _rows(correct=[True, False, True, False], retrieval_calls=1, answer_tokens=10)
    selection = select_dev_retrieval_cost_reference({"rag_bm25": rows, "rag_medcpt": rows})

    assert selection["selected_retrieval_cost_reference"] == "rag_bm25"
    assert selection["dev_case_count"] == 4


def test_summary_reports_component_latency_and_random_context_match_rates():
    summary = summarize_results(
        [
            {
                "case_id": "a",
                "subdataset": "medqa",
                "status": "completed",
                "is_correct": True,
                "component_latency_proxy_ms": 100,
                "random_context_doc_count_matched": True,
                "random_context_char_budget_matched": True,
            },
            {
                "case_id": "b",
                "subdataset": "medqa",
                "status": "completed",
                "is_correct": False,
                "component_latency_proxy_ms": 200,
                "random_context_doc_count_matched": True,
                "random_context_char_budget_matched": False,
            },
        ]
    )

    assert summary["p50_component_latency_proxy_ms"] == 100
    assert summary["p95_component_latency_proxy_ms"] == 200
    assert summary["random_context_doc_count_match_rate"] == 1.0
    assert summary["random_context_char_budget_match_rate"] == 0.5
    assert summary["component_latency_proxy_measurement_coverage"] == 1.0


def test_completed_test_arm_loader_checks_identity_partition_and_case_count(tmp_path):
    for arm in (*FIXED_ARMS, "random_context", *ADAPTIVE_ARMS):
        directory = tmp_path / "runs" / "e1_2" / "test" / arm
        directory.mkdir(parents=True)
        identity = f"identity-{arm}"
        (directory / "manifest.json").write_text(
            json.dumps(
                {
                    "status": "COMPLETED",
                    "partition": "TEST",
                    "arm": arm,
                    "config_sha256": "config-hash",
                    "case_count": 1,
                    "result_identity": identity,
                }
            ),
            encoding="utf-8",
        )
        (directory / "case_results.jsonl").write_text(
            json.dumps(
                {
                    "case_id": "q1",
                    "result_identity": identity,
                    "partition": "TEST",
                    "arm": arm,
                }
            )
            + "\n",
            encoding="utf-8",
        )

    rows = load_completed_test_arms(tmp_path, "config-hash", 1)

    assert set(rows) == {*FIXED_ARMS, "random_context", *ADAPTIVE_ARMS}
