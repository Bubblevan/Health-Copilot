from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from eval.rag_e5.data import E5IntegrationCase, EvaluatorCaseMetadata, IntegrationTaskFamily
from eval.rag_e5.leakage import policy_observation_from_payload
from health_ai_copilot.capabilities import (
    CapabilityEligibility,
    CapabilitySourceCatalog,
    E2WorkerRole,
    production_worker_capabilities,
)
from health_ai_copilot.execution_policy import (
    ExecutionPolicyObservation,
    require_policy_observation,
)
from health_ai_copilot.knowledge.loader import load_knowledge_cards
from health_ai_copilot.knowledge.scope import load_knowledge_scope
from health_ai_copilot.retrieval_capabilities import (
    BGE_LARGE_SHA256,
    QWEN3_8B_SHA256,
    R2MED_FINAL_LOCK_SHA256,
    RetrievalAction,
    require_action_available,
    retrieval_action_specs,
)
from tools.research.rag_e5.audit_esl_source import BATCHES, build_manifest


def test_policy_observation_round_trips_only_runtime_fields() -> None:
    observation = ExecutionPolicyObservation(
        query_text="Summarize the recent trend and relevant public guidance.",
        history_turn_count=8,
        history_time_span_days=42,
        recent_event_count=2,
        exam_record_count=1,
        active_condition_count=1,
        memory_available=True,
        memory_evidence_count=3,
        memory_source_types=("timeline", "profile"),
        memory_temporal_span_days=42,
        available_source_families=("public_health",),
        available_retrieval_actions=(RetrievalAction.OFF, RetrievalAction.STANDARD),
        remaining_provider_budget=2,
        remaining_tool_budget=3,
        remaining_token_budget=1200,
        deadline_remaining_ms=5000,
        state_summary="Synthetic bounded state summary.",
    )

    restored = ExecutionPolicyObservation.from_dict(observation.to_dict())

    assert restored == observation
    assert set(observation.to_dict()) == {
        "query_text",
        "history_turn_count",
        "history_time_span_days",
        "recent_event_count",
        "exam_record_count",
        "active_condition_count",
        "memory_available",
        "memory_evidence_count",
        "memory_source_types",
        "memory_temporal_span_days",
        "memory_conflict_flag",
        "available_source_families",
        "available_retrieval_actions",
        "remaining_provider_budget",
        "remaining_tool_budget",
        "remaining_token_budget",
        "deadline_remaining_ms",
        "previous_tool_failures",
        "state_summary",
    }


@pytest.mark.parametrize(
    "payload",
    [
        {"query_text": "question", "oracle_action": "STRONG"},
        {"query_text": "question", "metadata": {"outcome_standard": 1}},
        {"query_text": "question", "gold_evidence_group": "guideline"},
    ],
)
def test_teacher_fields_fail_closed_at_policy_boundary(payload: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="evaluator-only|non-runtime"):
        policy_observation_from_payload(payload)


def test_policy_api_refuses_evaluation_case_objects() -> None:
    case = E5IntegrationCase(
        case_id="synthetic-1",
        user_id="synthetic-user-1",
        batch_id="synthetic",
        question="Use my trend and public guidance.",
        decision_timestamp="2026-01-01T00:00:00Z",
        longitudinal_state_ref="synthetic://state/1",
    )
    teacher = EvaluatorCaseMetadata(
        case_id="synthetic-1",
        task_family=IntegrationTaskFamily.LONGITUDINAL_EXTERNAL,
        required_internal_state_fields=("timeline",),
        required_external_evidence_group="public_health",
        evaluation_type="synthetic_exact_match",
        evaluation_payload_ref="synthetic://evaluation/1",
        generation_provenance="fixture-v1",
    )

    assert teacher.task_family == IntegrationTaskFamily.LONGITUDINAL_EXTERNAL
    assert "allowed_external_source_families" not in E5IntegrationCase.__dataclass_fields__
    assert not hasattr(case, "allowed_external_source_families")
    with pytest.raises(TypeError, match="only ExecutionPolicyObservation"):
        require_policy_observation(case)
    with pytest.raises(ValueError, match="evaluator-only"):
        policy_observation_from_payload(
            {
                "query_text": case.question,
                "task_family": teacher.task_family.value,
            }
        )


def test_action_space_is_frozen_and_bound_to_r2med_profiles() -> None:
    off, standard, strong = retrieval_action_specs()

    assert [item.action for item in (off, standard, strong)] == [
        RetrievalAction.OFF,
        RetrievalAction.STANDARD,
        RetrievalAction.STRONG,
    ]
    assert off.retriever_profile_id is None
    assert off.max_external_calls == 0 and not off.requires_generator
    assert standard.frozen_config["rrf"] == {
        "k": 60,
        "weights": [1, 1],
        "output_depth": 100,
    }
    assert strong.frozen_config["method"] == "LameR-MV"
    assert strong.frozen_config["feedback_depth"] == 10
    assert strong.frozen_config["rrf"]["weights"] == [1, 2, 1, 2]
    assert strong.requires_generator
    assert standard.config_sha256 == "6ddb91bb0c31f5bc5b69372df6a3bdd3b4f71d70cdbcece8ef22c5bfd9a94330"
    assert strong.config_sha256 == "d9e3de9bf986a1f08b1c217153422d6e86ad0a84bc1982d90bade897c9274a8b"
    assert standard.provenance_lock_sha256 == R2MED_FINAL_LOCK_SHA256
    assert strong.provenance_lock_sha256 == R2MED_FINAL_LOCK_SHA256
    assert strong.frozen_config["generator"]["sha256"] == QWEN3_8B_SHA256
    assert strong.frozen_config["dense"]["weights_sha256"] == BGE_LARGE_SHA256
    assert len(off.config_sha256) == 64


def test_machine_profile_and_feature_contracts_match_runtime_types() -> None:
    profile_payload = json.loads(
        Path("runs/rag_e5/retrieval_action_profiles.json").read_text(encoding="utf-8")
    )
    feature_payload = json.loads(
        Path("runs/rag_e5/feature_contract.json").read_text(encoding="utf-8")
    )

    assert profile_payload["profiles"] == [item.to_dict() for item in retrieval_action_specs()]
    assert profile_payload["external_corpus_identity"] == (
        "9b19ad467f47641032277707cb1cfdb1d05c2fb39c558039b180efc7394692bd"
    )
    assert [item["name"] for item in feature_payload["features"]] == [
        item.name for item in ExecutionPolicyObservation.__dataclass_fields__.values()
    ]
    assert all(
        item[flag] is False
        for item in feature_payload["features"]
        for flag in ("uses_external_retrieval", "uses_gold", "uses_counterfactual")
    )
    assert all(item["available_at_decision_time"] is True for item in feature_payload["features"])
    assert feature_payload["observation_views"]["V0_question_only"] == ["query_text"]
    assert feature_payload["observation_views"]["V2_question_metadata_and_state_summary"] == [
        item.name for item in ExecutionPolicyObservation.__dataclass_fields__.values()
    ]


def test_current_reviewed_source_catalog_blocks_guideline_and_literature() -> None:
    cards = load_knowledge_cards("data/knowledge_cards")
    scope = load_knowledge_scope("data/knowledge_scope.json", cards)
    catalog = CapabilitySourceCatalog.from_reviewed_knowledge(
        cards,
        scope,
        publisher_families={
            "World Health Organization": "public_health",
            "Centers for Disease Control and Prevention": "public_health",
            "国家卫生健康委员会": "public_health",
        },
    )
    counts = Counter(item.source_family for item in catalog.sources)
    by_role = {item.role: item for item in production_worker_capabilities()}

    assert len(cards) == 30
    assert counts == {"public_health": 30}
    assert catalog.catalog_hash == "7cef21fceb5c04577fed2541dbefd2798900299825b711ab06c8c14e41b29c19"
    assert by_role[E2WorkerRole.PUBLIC_HEALTH].eligibility == CapabilityEligibility.ELIGIBLE
    assert by_role[E2WorkerRole.GUIDELINE].eligibility == CapabilityEligibility.NOT_YET_ELIGIBLE
    assert by_role[E2WorkerRole.LITERATURE].eligibility == CapabilityEligibility.NOT_YET_ELIGIBLE


def test_e5_v1_policy_observation_rejects_unscoped_literature_family() -> None:
    with pytest.raises(ValueError, match="outside E5 v1"):
        ExecutionPolicyObservation(
            query_text="synthetic question",
            available_source_families=("scholarly_literature",),
        )


def test_external_action_resolution_requires_active_action_and_source() -> None:
    standard = require_action_available(
        RetrievalAction.STANDARD,
        (RetrievalAction.OFF, RetrievalAction.STANDARD),
        ("public_health",),
    )
    assert standard.action == RetrievalAction.STANDARD
    with pytest.raises(ValueError, match="not available"):
        require_action_available(
            RetrievalAction.STRONG,
            (RetrievalAction.OFF, RetrievalAction.STANDARD),
            ("public_health",),
        )
    with pytest.raises(ValueError, match="no eligible source"):
        require_action_available(
            RetrievalAction.STANDARD,
            (RetrievalAction.OFF, RetrievalAction.STANDARD),
            (),
        )


def test_esl_source_audit_is_user_disjoint_and_reads_only_state_files(tmp_path: Path) -> None:
    source = tmp_path / "ESL-Bench"
    (source / "data").mkdir(parents=True)
    manifest = {
        "version": 1,
        "updated_at": "2026-09-01T00:00:00Z",
        "batches": {
            "202607": {
                "users": list(BATCHES["202607"]),
                "checksum": "sha256:dev-fixture",
                "eval_dataset": "dev-fixture",
            },
            "202608": {
                "users": list(BATCHES["202608"]),
                "checksum": "sha256:holdout-fixture",
                "eval_dataset": "holdout-fixture",
            },
        },
    }
    (source / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for batch_id, users in BATCHES.items():
        for user_id in users:
            user_dir = source / "data" / batch_id / user_id
            user_dir.mkdir(parents=True)
            (user_dir / "profile.json").write_text(
                json.dumps(
                    {
                        "metadata": {},
                        "demographics": {"age": 60, "gender": "synthetic"},
                        "personality": {},
                        "health_profile": {
                            "chronic_conditions": [],
                            "past_medical_history": [],
                            "lifestyle": {},
                        },
                    }
                ),
                encoding="utf-8",
            )
            (user_dir / "timeline.json").write_text(
                json.dumps(
                    {
                        "user_id": user_id,
                        "generated_at": "synthetic",
                        "entry_count": 1,
                            "entries": [
                                {"time": "synthetic", "entry_type": "fixture", "event": "fixture"}
                            ],
                    }
                ),
                encoding="utf-8",
            )
            (user_dir / "exam_data.json").write_text(
                    json.dumps(
                        [
                            {
                                "exam_date": "synthetic",
                                "exam_type": "fixture",
                                "indicators": {},
                            }
                        ]
                    ),
                encoding="utf-8",
            )
    # The native holdout question file is deliberately invalid JSON. An audit
    # that opens it would fail; the permitted state-only audit must not.
    holdout_dir = source / "data" / "202608"
    (holdout_dir / "sample320-20260830.jsonl").write_text("not json", encoding="utf-8")

    result = build_manifest(source)

    assert result["batches"]["202607"]["state_file_count"] == 60
    assert result["batches"]["202608"]["state_file_count"] == 60
    assert result["split"]["user_overlap"] == []
    assert result["split"]["user_level_split"] is True
    assert result["question_or_answer_files_opened"] is False
    assert (source / "data" / "202608" / "sample320-20260830.jsonl").read_text() == "not json"


def test_committed_esl_pin_has_disjoint_users_and_no_native_outcome_access() -> None:
    payload = json.loads(
        Path("runs/rag_e5/esl_source_manifest.json").read_text(encoding="utf-8")
    )

    assert payload["manifest_version"] == 16
    assert payload["question_or_answer_files_opened"] is False
    assert payload["split"]["user_overlap"] == []
    assert payload["batches"]["202607"]["state_file_count"] == 60
    assert payload["batches"]["202608"]["state_file_count"] == 60
    assert payload["upstream_git_commit"] is None
    dev_metadata = payload["batches"]["202607"]["schemas"]["profile.json:metadata"][0]["fields"]
    holdout_metadata = payload["batches"]["202608"]["schemas"]["profile.json:metadata"][0]["fields"]
    assert "generation_params" in dev_metadata
    assert "generation_params" not in holdout_metadata
    dev_exam = payload["batches"]["202607"]["schemas"]["exam_data.json:record"][0]["fields"]
    holdout_exam = payload["batches"]["202608"]["schemas"]["exam_data.json:record"][0]["fields"]
    assert "overall_assessment" in dev_exam
    assert "overall_assessment" not in holdout_exam


def test_evaluator_metadata_has_no_runtime_serializer_or_oracle_action_field() -> None:
    assert "oracle_action" not in EvaluatorCaseMetadata.__dataclass_fields__
    assert "required_external_evidence_group" not in ExecutionPolicyObservation.__dataclass_fields__
    assert "task_family" not in ExecutionPolicyObservation.__dataclass_fields__
