from __future__ import annotations

import json
from pathlib import Path

from eval.rag_e6.scoring import (
    _load_build_truth_slice,
    _paired_cluster_bootstrap,
    _score_arm,
)


def test_build_truth_reader_skips_non_build_rows_without_json_decode(tmp_path) -> None:
    train_root = tmp_path / "train"
    train_root.mkdir()
    runtime_path = train_root / "episodes.jsonl"
    truth_path = train_root / "evaluator_truth.jsonl"
    with runtime_path.open("w", encoding="utf-8") as runtime, truth_path.open(
        "w", encoding="utf-8"
    ) as truth:
        for index in range(4096):
            subject = "BUILD-SUBJECT" if index == 0 else "HIDDEN-SUBJECT"
            runtime.write(json.dumps({"episode_id": f"EP-{index}", "subject_id": subject}) + "\n")
            if index == 0:
                truth.write(json.dumps({
                    "episode_id": "EP-0",
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
        expected_ids={"EP-0"},
    )
    assert set(loaded) == {"EP-0"}
    assert audit["build_truth_rows_decoded"] == 1
    assert audit["non_build_truth_rows_skipped_without_json_decode"] == 4095
    assert audit["frozen_dev_truth_rows_decoded"] == 0
    assert audit["future_train_truth_rows_decoded"] == 0


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
