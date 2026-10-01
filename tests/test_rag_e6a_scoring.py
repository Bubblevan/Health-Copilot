from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.rag_e6.protocol import create_protocol_lock
from eval.rag_e6.scoring import (
    _load_build_truth_slice,
    _load_partition_truth_slice,
    _paired_cluster_bootstrap,
    _retrieval_contract_failed,
    _score_arm,
)


def test_build_truth_reader_skips_non_build_rows_without_json_decode(tmp_path) -> None:
    train_root = tmp_path / "train"
    train_root.mkdir()
    runtime_path = train_root / "episodes.jsonl"
    truth_path = train_root / "evaluator_truth.jsonl"
    expected_ids = {f"EP-{index}" for index in range(818)}
    with runtime_path.open("w", encoding="utf-8") as runtime, truth_path.open(
        "w", encoding="utf-8"
    ) as truth:
        for index in range(4096):
            subject = "BUILD-SUBJECT" if index < 818 else "HIDDEN-SUBJECT"
            runtime.write(json.dumps({"episode_id": f"EP-{index}", "subject_id": subject}) + "\n")
            if index < 818:
                truth.write(json.dumps({
                    "episode_id": f"EP-{index}",
                    "answer_type": "EXACT_TOKEN",
                    "answer_values": ["SYNVAL-0123456789"],
                    "capability_requirement_oracle": {"answerability": True},
                    "required_memory_record_ids": [],
                    "required_external_evidence_ids": ["EVIDENCE-1"],
                }) + "\n")
            else:
                truth.write("deliberately-not-json-and-must-remain-opaque\n")

    loaded, audit = _load_build_truth_slice(
            u2f_root=Path(tmp_path),
            split_manifest={
                "subjects_by_partition": {"BUILD": [{"subject_id": "BUILD-SUBJECT"}]}
            },
        expected_ids=expected_ids,
    )
    assert set(loaded) == expected_ids
    assert audit["build_truth_rows_decoded"] == 818
    assert audit["non_build_truth_rows_skipped_without_json_decode"] == 3278
    assert audit["frozen_dev_truth_rows_decoded"] == 0
    assert audit["future_train_truth_rows_decoded"] == 0


def test_frozen_dev_truth_reader_skips_other_partitions_without_json_decode(tmp_path) -> None:
    train_root = tmp_path / "train"
    train_root.mkdir()
    runtime_path = train_root / "episodes.jsonl"
    truth_path = train_root / "evaluator_truth.jsonl"
    frozen_subjects = [{"subject_id": f"DEV-{index}"} for index in range(128)]
    expected_ids = {f"DEV-EP-{index}" for index in range(1628)}
    with runtime_path.open("w", encoding="utf-8") as runtime, truth_path.open(
        "w", encoding="utf-8"
    ) as truth:
        for index in range(4096):
            is_dev = index < 1628
            episode_id = f"DEV-EP-{index}" if is_dev else f"HIDDEN-EP-{index}"
            subject_id = f"DEV-{index % 128}" if is_dev else f"HIDDEN-{index}"
            runtime.write(json.dumps({"episode_id": episode_id, "subject_id": subject_id}) + "\n")
            if is_dev:
                truth.write(json.dumps({
                    "episode_id": episode_id,
                    "answer_type": "EXACT_TOKEN",
                    "answer_values": ["SYNVAL-0123456789"],
                    "capability_requirement_oracle": {"answerability": True},
                    "required_memory_record_ids": [],
                    "required_external_evidence_ids": ["EVIDENCE-1"],
                }) + "\n")
            else:
                truth.write("deliberately-not-json-and-must-remain-opaque\n")

    loaded, audit = _load_partition_truth_slice(
        u2f_root=Path(tmp_path),
        split_manifest={"subjects_by_partition": {"FROZEN_DEV": frozen_subjects}},
        expected_ids=expected_ids,
        partition="FROZEN_DEV",
    )
    assert set(loaded) == expected_ids
    assert audit["frozen_dev_truth_rows_decoded"] == 1628
    assert audit["non_frozen_dev_truth_rows_skipped_without_json_decode"] == 2468
    assert audit["build_truth_rows_decoded"] == 0
    assert audit["future_train_truth_rows_decoded"] == 0


def test_protocol_lock_requires_every_prespecified_build_gate() -> None:
    build_manifest = {
        "partition": "BUILD",
        "episode_count": 818,
        "all_partition_episodes_executed": True,
        "evaluator_truth_opened": False,
        "generator": {
            "generation_calls_attempted": 100,
            "generation_calls_nonempty": 99,
            "generation_calls_truncated": 0,
            "model_name": "qwen",
            "model_api_id": "qwen-local",
            "model_sha256": "model-sha",
            "context_ceiling": 65536,
            "completion_ceiling": 512,
            "temperature": 0,
            "top_p": 1,
            "reasoning_enabled": False,
            "retry_count": 0,
            "actual_backend": "Vulkan1",
        },
        "code_sha256": {"reader.py": "reader-sha"},
        "retrieval": {"top_k": 10},
    }
    build_report = {
        "partition": "BUILD",
        "episode_count": 818,
        "frozen_dev_truth_opened": False,
        "development_gate": {
            "primary_build_point_delta_at_least_10pp": True,
            "primary_build_ci_lower_above_zero": True,
            "grounding_degradation_no_more_than_1pp": True,
            "utilization_build_point_delta_at_least_10pp": True,
        },
        "truth_access": {
            "build_truth_rows_decoded": 818,
            "frozen_dev_truth_rows_decoded": 0,
            "future_train_truth_rows_decoded": 0,
            "reserved_test_ood_opened": False,
        },
        "primary_comparison": {
            "candidate": "CFEC_STRONG",
            "baseline": "VANILLA_STRONG",
            "slice": "RAG",
            "grounded_task_success_delta": {"delta": 0.2, "ci95": [0.1, 0.3]},
            "utilization_given_full_evidence_task_success_delta": {"delta": 0.2},
        },
    }
    lock = create_protocol_lock(
        build_manifest=build_manifest,
        build_report=build_report,
        build_manifest_sha256="manifest-sha",
        build_report_sha256="report-sha",
        scored_rows_sha256="scores-sha",
        method_code_commit="abc123",
    )
    assert lock["selected_method"] == "CFEC-v1.4"
    assert lock["primary_arm"] == "CFEC_STRONG"
    assert lock["frozen_dev_truth_opened"] is False

    build_manifest["generator"]["generation_calls_truncated"] = 1
    lock_with_truncation = create_protocol_lock(
        build_manifest=build_manifest,
        build_report=build_report,
        build_manifest_sha256="manifest-sha",
        build_report_sha256="report-sha",
        scored_rows_sha256="scores-sha",
        method_code_commit="abc123",
    )
    assert lock_with_truncation["build_selection_evidence"]["generation_health"][
        "truncated"
    ] == 1
    build_manifest["generator"]["generation_calls_truncated"] = 0

    build_report["development_gate"]["primary_build_ci_lower_above_zero"] = False
    with pytest.raises(ValueError, match="BUILD gate"):
        create_protocol_lock(
            build_manifest=build_manifest,
            build_report=build_report,
            build_manifest_sha256="manifest-sha",
            build_report_sha256="report-sha",
            scored_rows_sha256="scores-sha",
            method_code_commit="abc123",
        )


def test_numeric_answer_parser_ignores_alias_digits_and_scores_provenance() -> None:
    truth = {
        "answer_values": ["12"],
        "answer_type": "NUMERIC",
        "capability_requirement_oracle": {"answerability": True},
        "required_memory_record_ids": [],
        "required_external_evidence_ids": ["DOC-1"],
    }
    scored = _score_arm(
        truth=truth,
        runtime_arm={
            "answer": "The value is 12 [E1].",
            "used_evidence_ids": ["DOC-1"],
            "ranked_evidence_ids": ["DOC-1"],
        },
        visible_ids={"DOC-1"},
    )
    assert scored["answer_value_correct"]
    assert scored["task_success"]
    assert scored["grounded_task_success"]


def test_grounding_fails_closed_for_missing_required_evidence_and_unknown_sources() -> None:
    truth = {
        "answer_values": ["SYNVAL-0123456789"],
        "answer_type": "EXACT_TOKEN",
        "capability_requirement_oracle": {"answerability": True},
        "required_memory_record_ids": [],
        "required_external_evidence_ids": ["DOC-1", "DOC-2"],
    }
    scored = _score_arm(
        truth=truth,
        runtime_arm={
            "answer": "SYNVAL-0123456789 [E1]",
            "used_evidence_ids": ["DOC-1", "NOT-VISIBLE"],
            "ranked_evidence_ids": ["DOC-1"],
        },
        visible_ids={"DOC-1"},
    )
    assert scored["answer_value_correct"]
    assert not scored["task_success"]
    assert not scored["provenance_pass"]
    assert not scored["grounding_pass"]


def test_output_contract_failure_cannot_count_as_task_success() -> None:
    truth = {
        "answer_values": ["SYNVAL-0123456789"],
        "answer_type": "EXACT_TOKEN",
        "capability_requirement_oracle": {"answerability": True},
        "required_memory_record_ids": [],
        "required_external_evidence_ids": ["DOC-1"],
    }
    scored = _score_arm(
        truth=truth,
        runtime_arm={
            "answer": "SYNVAL-0123456789 [E1]",
            "used_evidence_ids": ["DOC-1"],
            "ranked_evidence_ids": ["DOC-1"],
            "output_contract_failure": True,
            "unknown_aliases": ["[E11]"],
        },
        visible_ids={"DOC-1"},
    )
    assert scored["answer_value_correct"]
    assert not scored["output_contract_pass"]
    assert not scored["task_success"]
    assert not scored["grounded_task_success"]


def test_failed_retrieval_contract_cannot_count_as_task_success() -> None:
    truth = {
        "answer_values": ["SYNVAL-0123456789"],
        "answer_type": "EXACT_TOKEN",
        "capability_requirement_oracle": {"answerability": True},
        "required_memory_record_ids": [],
        "required_external_evidence_ids": ["DOC-1"],
    }
    scored = _score_arm(
        truth=truth,
        runtime_arm={
            "answer": "SYNVAL-0123456789 [E1]",
            "used_evidence_ids": ["DOC-1"],
            "ranked_evidence_ids": ["DOC-1"],
        },
        visible_ids={"DOC-1"},
        retrieval_contract_failure=True,
    )
    assert scored["answer_value_correct"]
    assert scored["output_contract_pass"]
    assert scored["retrieval_contract_failure"]
    assert not scored["execution_contract_pass"]
    assert not scored["task_success"]
    assert not scored["grounded_task_success"]


def test_invalid_lamer_bridge_fails_only_strong_retrieval_arms() -> None:
    invalid_bridge = {
        "valid": False,
        "completed": False,
        "fallback_original_query": True,
        "truncated": True,
    }
    assert _retrieval_contract_failed("VANILLA_STRONG", invalid_bridge)
    assert _retrieval_contract_failed("CFEC_STRONG", invalid_bridge)
    assert not _retrieval_contract_failed("VANILLA_STANDARD", invalid_bridge)
    assert not _retrieval_contract_failed("CFEC_STANDARD", invalid_bridge)


def test_subject_cluster_bootstrap_is_paired_and_reproducible() -> None:
    rows = []
    for subject_index in range(5):
        for episode_index in range(2):
            rows.append({
                "subject_id": f"S-{subject_index}",
                "arms": {
                    "VANILLA_STRONG": {"task_success": False},
                    "CFEC_STRONG": {"task_success": bool(episode_index)},
                },
            })
    first = _paired_cluster_bootstrap(
        rows, metric="task_success", arm_a="VANILLA_STRONG", arm_b="CFEC_STRONG",
        resamples=200, seed=7,
    )
    second = _paired_cluster_bootstrap(
        rows, metric="task_success", arm_a="VANILLA_STRONG", arm_b="CFEC_STRONG",
        resamples=200, seed=7,
    )
    assert first == second
    assert first["delta"] == 0.5
    assert first["subjects"] == 5
