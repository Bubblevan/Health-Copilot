"""Validation and construction of the immutable E6A method lock."""

from __future__ import annotations

from typing import Any

LOCKED_BUILD_GATES = (
    "primary_build_point_delta_at_least_10pp",
    "primary_build_ci_lower_above_zero",
    "grounding_degradation_no_more_than_1pp",
    "utilization_build_point_delta_at_least_10pp",
)


def create_protocol_lock(
    *,
    build_manifest: dict[str, Any],
    build_report: dict[str, Any],
    build_manifest_sha256: str,
    build_report_sha256: str,
    scored_rows_sha256: str,
    method_code_commit: str,
) -> dict[str, Any]:
    """Freeze RSEL-v1 only after its BUILD gate passes."""
    gate = build_report.get("development_gate", {})
    if (
        build_manifest.get("partition") != "BUILD"
        or build_manifest.get("episode_count") != 818
        or build_manifest.get("all_partition_episodes_executed") is not True
        or build_manifest.get("evaluator_truth_opened") is not False
        or build_report.get("partition") != "BUILD"
        or build_report.get("episode_count") != 818
        or build_report.get("frozen_dev_truth_opened") is not False
        or any(gate.get(name) is not True for name in LOCKED_BUILD_GATES)
    ):
        raise ValueError("RSEL-v1 cannot be frozen unless the complete BUILD gate passes")

    generator = build_manifest.get("generator", {})
    attempted = int(generator.get("generation_calls_attempted", 0))
    nonempty = int(generator.get("generation_calls_nonempty", 0))
    if attempted <= 0 or nonempty / attempted < 0.99:
        raise ValueError("BUILD generation health is below the 99% freeze threshold")
    truth_access = build_report.get("truth_access", {})
    if (
        truth_access.get("build_truth_rows_decoded") != 818
        or truth_access.get("frozen_dev_truth_rows_decoded") != 0
        or truth_access.get("future_train_truth_rows_decoded") != 0
        or truth_access.get("reserved_test_ood_opened") is not False
    ):
        raise ValueError("BUILD scoring crossed a protected evaluator-truth boundary")

    comparison = build_report.get("primary_comparison", {})
    delta = comparison.get("grounded_task_success_delta", {})
    utilization = comparison.get("utilization_given_full_evidence_task_success_delta", {})
    if (
        comparison.get("candidate") != "RSEL_STRONG"
        or comparison.get("baseline") != "VANILLA_STRONG"
        or comparison.get("slice") != "RAG"
        or delta.get("delta") is None
        or not isinstance(delta.get("ci95"), list)
        or len(delta["ci95"]) != 2
        or utilization.get("delta") is None
    ):
        raise ValueError("BUILD report does not match the pre-registered primary comparison")

    return {
        "schema_version": "rag-e6a-protocol-lock-v1",
        "selected_method": "RSEL-v1",
        "primary_arm": "RSEL_STRONG",
        "primary_comparator": "VANILLA_STRONG",
        "secondary_arm": "RSEL_STANDARD",
        "secondary_comparator": "VANILLA_STANDARD",
        "control_arm": "VANILLA_OFF",
        "method_code_commit": method_code_commit,
        "build_selection_evidence": {
            "build_manifest_sha256": build_manifest_sha256,
            "build_report_sha256": build_report_sha256,
            "scored_rows_sha256": scored_rows_sha256,
            "primary_grounded_task_success_delta": delta,
            "utilization_given_full_evidence_delta": utilization,
            "development_gate": gate,
            "generation_health": {
                "attempted": attempted,
                "nonempty": nonempty,
                "nonempty_rate": nonempty / attempted,
                "truncated": int(generator.get("generation_calls_truncated", 0)),
            },
        },
        "runtime_code_sha256": build_manifest["code_sha256"],
        "retrieval_identity": build_manifest["retrieval"],
        "generator_identity": {
            key: generator[key]
            for key in (
                "model_name", "model_api_id", "model_sha256", "context_ceiling",
                "completion_ceiling", "temperature", "top_p", "reasoning_enabled",
                "retry_count", "actual_backend",
            )
        },
        "execution_graph": [
            "same_vanilla_draft_and_frozen_top10_evidence",
            "harness_extracts_exact_query_key_to_value_relations",
            "answer_values_and_aliases_derived_only_from_visible_relations",
            "no_match_falls_back_to_unchanged_vanilla_answer",
            "used_evidence_and_provenance_are_harness_owned",
        ],
        "primary_metric": "RAG-slice grounded_task_success",
        "utilization_metric": "task_success when both arms retrieve all required external evidence in top-10",
        "bootstrap": {"unit": "subject_id", "resamples": 10000, "seed": 20260930},
        "reserved_test_ood_materialized": False,
        "frozen_dev_truth_opened": False,
    }
