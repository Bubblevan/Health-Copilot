from __future__ import annotations

import pytest

from eval.rag_e5.data import EvaluatorCaseMetadata, IntegrationTaskFamily
from health_ai_copilot.e5_runtime_context import E5RuntimeCapabilityContext
from health_ai_copilot.retrieval_capabilities import RetrievalAction


def _environment_context() -> E5RuntimeCapabilityContext:
    return E5RuntimeCapabilityContext.from_environment(
        environment_source_families=("public_health", "reviewed_guideline"),
        permission_actions=(RetrievalAction.OFF, RetrievalAction.STANDARD, RetrievalAction.STRONG),
        budget_snapshot={
            "remaining_provider_budget": 2,
            "remaining_tool_budget": 3,
            "remaining_token_budget": 4096,
            "deadline_remaining_ms": 60_000,
        },
        active_corpus_identity="a" * 64,
    )


def test_t0_t1_t2_share_same_v1_capability_context() -> None:
    teachers = (
        EvaluatorCaseMetadata(
            case_id="t0",
            task_family=IntegrationTaskFamily.LONGITUDINAL_ONLY,
            required_internal_state_fields=("timeline",),
            required_external_evidence_group=None,
            evaluation_type="synthetic",
            evaluation_payload_ref="fixture://t0",
            generation_provenance="fixture-v1",
        ),
        EvaluatorCaseMetadata(
            case_id="t1",
            task_family=IntegrationTaskFamily.EXTERNAL_ONLY,
            required_internal_state_fields=(),
            required_external_evidence_group="reviewed_guideline",
            evaluation_type="synthetic",
            evaluation_payload_ref="fixture://t1",
            generation_provenance="fixture-v1",
        ),
        EvaluatorCaseMetadata(
            case_id="t2",
            task_family=IntegrationTaskFamily.LONGITUDINAL_EXTERNAL,
            required_internal_state_fields=("timeline", "profile"),
            required_external_evidence_group="public_health",
            evaluation_type="synthetic",
            evaluation_payload_ref="fixture://t2",
            generation_provenance="fixture-v1",
        ),
    )

    contexts = [_environment_context() for _teacher in teachers]

    assert len({context.provenance_sha256 for context in contexts}) == 1
    assert all(
        context.available_source_families == ("public_health", "reviewed_guideline")
        for context in contexts
    )
    assert all(
        context.available_retrieval_actions
        == (RetrievalAction.OFF, RetrievalAction.STANDARD, RetrievalAction.STRONG)
        for context in contexts
    )
    assert all(context.capability_context_source == "environment" for context in contexts)


def test_available_source_families_not_derived_from_required_group() -> None:
    t1 = EvaluatorCaseMetadata(
        case_id="t1",
        task_family=IntegrationTaskFamily.EXTERNAL_ONLY,
        required_internal_state_fields=(),
        required_external_evidence_group="reviewed_guideline",
        evaluation_type="synthetic",
        evaluation_payload_ref="fixture://t1",
        generation_provenance="fixture-v1",
    )
    context = _environment_context()

    assert t1.required_external_evidence_group == "reviewed_guideline"
    assert context.available_source_families == ("public_health", "reviewed_guideline")
    assert context.available_source_families != (t1.required_external_evidence_group,)


def test_available_actions_not_derived_from_oracle_action() -> None:
    context = _environment_context()
    teacher_outcome = {"oracle_action": "OFF", "counterfactual_outcome": 1}

    assert teacher_outcome["oracle_action"] == "OFF"
    assert context.available_retrieval_actions == (
        RetrievalAction.OFF,
        RetrievalAction.STANDARD,
        RetrievalAction.STRONG,
    )


def test_runtime_context_fails_closed_on_unapproved_guideline_or_bad_provenance() -> None:
    with pytest.raises(ValueError, match="active corpus identity"):
        E5RuntimeCapabilityContext.from_environment(
            environment_source_families=("public_health", "reviewed_guideline"),
            permission_actions=(RetrievalAction.OFF, RetrievalAction.STANDARD),
            budget_snapshot={
                "remaining_provider_budget": 1,
                "remaining_tool_budget": 1,
                "remaining_token_budget": 100,
                "deadline_remaining_ms": None,
            },
            active_corpus_identity=None,
        )
    with pytest.raises(ValueError, match="unsupported fields"):
        E5RuntimeCapabilityContext.from_environment(
            environment_source_families=("public_health",),
            permission_actions=(RetrievalAction.OFF,),
            budget_snapshot={
                "remaining_provider_budget": 1,
                "remaining_tool_budget": 1,
                "remaining_token_budget": 100,
                "deadline_remaining_ms": None,
                "oracle_action": "OFF",
            },
            active_corpus_identity=None,
        )


def test_policy_fields_exclude_task_labels_and_keep_provenance_separate() -> None:
    context = _environment_context()
    fields = context.to_policy_fields()

    assert "task_family" not in fields
    assert "required_external_evidence_group" not in fields
    assert "oracle_action" not in fields
    assert context.capability_context_source == "environment"
    assert len(context.provenance_sha256) == 64
