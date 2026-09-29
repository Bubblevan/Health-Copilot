"""Fail-closed checks between runtime policy observations and teacher data."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from health_ai_copilot.execution_policy import ExecutionPolicyObservation

EVALUATOR_ONLY_FIELDS = frozenset(
    {
        "task_family",
        "required_internal_state_fields",
        "required_external_evidence_group",
        "gold_evidence_group",
        "gold_external_source",
        "correct_answer",
        "answer_key",
        "expected_value",
        "key_points",
        "official_difficulty",
        "official_capability_dimension",
        "outcome_off",
        "outcome_standard",
        "outcome_strong",
        "oracle_action",
        "action_was_correct",
        "retrieval_hit_relevance",
        "retrieved_gold_doc_count",
        "post_retrieval_evidence_score",
    }
)


def reject_evaluator_fields(value: object, *, path: str = "observation") -> None:
    """Reject teacher-only field names at any mapping depth."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            if key in EVALUATOR_ONLY_FIELDS:
                raise ValueError(f"evaluator-only field is forbidden in runtime input: {path}.{key}")
            reject_evaluator_fields(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            reject_evaluator_fields(item, path=f"{path}[{index}]")


def policy_observation_from_payload(value: Mapping[str, Any]) -> ExecutionPolicyObservation:
    reject_evaluator_fields(value)
    return ExecutionPolicyObservation.from_dict(value)


__all__ = ["EVALUATOR_ONLY_FIELDS", "policy_observation_from_payload", "reject_evaluator_fields"]
