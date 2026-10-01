from tools.research.memory.audit_mem3b1_dev_alternative_cues_v1 import (
    attach_manual_labels,
    score_candidate_labels,
)
from tools.research.memory.audit_mem3b1_dev_alternative_cues_v2 import (
    _amend_annotations,
    _score_mentions,
)


def test_multiple_cues_in_one_turn_share_source_label_without_extra_turns() -> None:
    candidates = [
        {"question_id": "q1", "source_position": 2, "match_index_in_turn": 0},
        {"question_id": "q1", "source_position": 2, "match_index_in_turn": 1},
        {"question_id": "q2", "source_position": 7, "match_index_in_turn": 0},
    ]
    labels = {
        "q1:2": {"label": "EXPLICIT_STATE_REVISION", "rationale": "adopted routine"},
        "q2:7": {"label": "NON_SELF_OR_IMPORTED_CONTENT", "rationale": "embedded text"},
    }

    labeled = attach_manual_labels(candidates, labels)

    assert [row["label"] for row in labeled] == [
        "EXPLICIT_STATE_REVISION",
        "EXPLICIT_STATE_REVISION",
        "NON_SELF_OR_IMPORTED_CONTENT",
    ]
    assert score_candidate_labels(labeled)["unique_source_turn_count"] == 2


def test_task_plan_precision_is_reported_separately_from_personal_state_precision() -> None:
    candidates = [
        {"label": "EXPLICIT_STATE_REVISION", "question_id": "q1", "source_position": 1},
        {"label": "EXPLICIT_TASK_PLAN_REVISION", "question_id": "q2", "source_position": 2},
        {"label": "PROPOSED_NOT_ADOPTED", "question_id": "q3", "source_position": 3},
        {"label": "NON_SELF_OR_IMPORTED_CONTENT", "question_id": "q4", "source_position": 4},
    ]

    score = score_candidate_labels(candidates)

    assert score["adopted_personal_state_revision_precision"] == 0.25
    assert score["state_or_task_revision_precision"] == 0.5
    assert score["label_counts_by_occurrence"]["EXPLICIT_TASK_PLAN_REVISION"] == 1


def test_v2_amendment_renames_change_mentions_without_claiming_revision_chains() -> None:
    annotations = {
        "labels": {
            "q1:1": {"label": "EXPLICIT_STATE_REVISION", "rationale": "state change"},
            "q2:2": {"label": "EXPLICIT_TASK_PLAN_REVISION", "rationale": "plan change"},
        }
    }
    amendment = {
        "label_renames": {
            "EXPLICIT_STATE_REVISION": "EXPLICIT_STATE_CHANGE_MENTION",
            "EXPLICIT_TASK_PLAN_REVISION": "EXPLICIT_TASK_PLAN_CHANGE_MENTION",
        }
    }

    corrected = _amend_annotations(annotations, amendment)
    score = _score_mentions(
        [
            {"question_id": "q1", "source_position": 1, **corrected["labels"]["q1:1"]},
            {"question_id": "q2", "source_position": 2, **corrected["labels"]["q2:2"]},
        ]
    )

    assert corrected["labels"]["q1:1"]["label"] == "EXPLICIT_STATE_CHANGE_MENTION"
    assert score["explicit_adopted_state_change_mentions"] == 1
    assert score["revision_chain_quality_scored"] is False
