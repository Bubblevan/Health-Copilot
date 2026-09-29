import json

from health_ai_copilot.research.integration.counterfactual import CounterfactualRunner
from health_ai_copilot.research.integration.fixtures import build_synthetic_cases
from health_ai_copilot.research.integration.training_views import (
    StudentActionSource,
    build_training_views,
)


def _serialized(value) -> str:
    return json.dumps(value.to_dict(), ensure_ascii=False, sort_keys=True)


def test_sft_candidate_excludes_gold_future_outcomes_and_teacher_analysis() -> None:
    case = next(item for item in build_synthetic_cases() if item.case_id == "U1-MEM")
    bundle = CounterfactualRunner().run(case.episode, case.evaluation, case.resources)
    views = build_training_views(
        case.episode, bundle, student_action="NONE",
        student_action_source=StudentActionSource.SCRIPTED_PROBE,
    )
    serialized = _serialized(views.sft_candidate)
    for forbidden in ("gold_answer", "future_patient_state", "counterfactual_outcomes",
                      "failure_attribution", "privileged_teacher", "raw_outcome"):
        assert forbidden not in serialized
    assert views.sft_candidate.acceptable_actions == ("MEMORY",)


def test_student_packet_does_not_embed_privileged_teacher_fields() -> None:
    case = next(item for item in build_synthetic_cases() if item.case_id == "U1-MEM")
    bundle = CounterfactualRunner().run(case.episode, case.evaluation, case.resources)
    views = build_training_views(
        case.episode, bundle, student_action="NONE",
        student_action_source=StudentActionSource.SCRIPTED_PROBE,
    )
    student_json = _serialized(views.student_packet)
    teacher_json = _serialized(views.teacher_packet)
    for forbidden in ("task_success", "safety_pass", "grounding_pass", "gold_answer",
                      "counterfactual_action_outcomes", "minimal_successful_action_set",
                      "failure_attribution", "privileged_plane"):
        assert forbidden not in student_json
    assert "counterfactual_action_outcomes" in teacher_json
    assert "arm_failure_categories" in teacher_json


def test_grpo_group_contains_same_episode_actions_and_outcomes() -> None:
    case = next(item for item in build_synthetic_cases() if item.case_id == "U1-MEM-RAG")
    bundle = CounterfactualRunner().run(case.episode, case.evaluation, case.resources)
    group = build_training_views(
        case.episode, bundle, student_action="NONE",
        student_action_source=StudentActionSource.SCRIPTED_PROBE,
    ).grpo_group.to_dict()
    assert group["episode_id"] == case.case_id
    assert len(group["actions"]) == len(group["outcomes"]) == len(group["group_rewards"])
    assert {"MEMORY+RAG", "ALL"}.issubset(group["actions"])
    assert group["rollout_source"] == "COUNTERFACTUAL_ENUMERATION"
