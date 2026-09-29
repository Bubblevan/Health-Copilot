from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from eval.rag_e5.overlay import (
    CAPABILITY_CONTEXT,
    READER_SCHEMA,
    E5RuntimeCase,
    audit_questions,
    build_overlay_cases,
    score_case,
)
from tools.research.rag_e5.build_e5b1_state_packets import (
    DEVELOPMENT_USERS,
    _validate_source_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
USER_IDS = tuple(f"user{number}_AT_demo" for number in range(5200, 5220))


def _state_packets(users: tuple[str, ...] = USER_IDS) -> list[dict[str, object]]:
    packets = []
    for index, user_id in enumerate(users):
        timestamp = f"2026-08-{(index % 28) + 1:02d}T12:00:00"
        packets.append(
            {
                "user_id": user_id,
                "summary_builder_version": "e5-longitudinal-state-v4",
                "decision_boundary": {
                    "user_id": user_id,
                    "naive_timestamp": timestamp,
                    "semantics_id": "SOURCE_RELATIVE_NAIVE_CIVIL_TIME_V1",
                },
                "recent_measurement_trends": [{"metric": "body_weight", "trend": "rising"}],
            }
        )
    return packets


def _eligible_chunks() -> list[dict[str, str]]:
    return [
        {
            "chunk_id": "pa-adults",
            "source_id": "who-physical-activity-sedentary-2020",
            "recommendation_id": "adults_18_64_physical_activity",
        },
        {
            "chunk_id": "fat-core",
            "source_id": "who-total-fat-weight-gain-2023",
            "recommendation_id": "recommendation-1",
        },
        {
            "chunk_id": "fat-caveat",
            "source_id": "who-total-fat-weight-gain-2023",
            "recommendation_id": "recommendation-1",
        },
    ]


def _build():
    return build_overlay_cases(state_packets=_state_packets(), eligible_chunks=_eligible_chunks())


def test_task_ids_are_deterministic_and_bind_boundary_for_all_families() -> None:
    first, _ = _build()
    second, _ = _build()

    assert [case.case_id for case in first] == [case.case_id for case in second]
    assert first[1].decision_boundary is None  # T1 has no runtime temporal dependency.
    changed_packets = _state_packets()
    changed_packets[0]["decision_boundary"]["naive_timestamp"] = "2026-08-31T12:00:00"  # type: ignore[index]
    changed, _ = build_overlay_cases(state_packets=changed_packets, eligible_chunks=_eligible_chunks())
    assert first[1].case_id != changed[1].case_id


def test_exact_three_tasks_per_user_and_family_counts() -> None:
    runtime, teacher = _build()
    counts = {family: sum(case.task_family == family for case in teacher) for family in ("T0", "T1", "T2")}

    assert len(runtime) == len(teacher) == 60
    assert counts == {"T0": 20, "T1": 20, "T2": 20}
    assert all(sum(case.user_id == user_id for case in teacher) == 3 for user_id in USER_IDS)


def test_t0_t1_t2_requirements_and_runtime_teacher_separation() -> None:
    runtime, teacher = _build()
    runtime_by_id = {case.case_id: case for case in runtime}

    for case in teacher:
        runtime_case = runtime_by_id[case.case_id]
        if case.task_family == "T0":
            assert case.expected_state_fields
            assert case.required_external_source_id is None
            assert not case.required_recommendation_ids
        elif case.task_family == "T1":
            assert not case.expected_state_fields
            assert case.required_external_source_id is not None
            assert runtime_case.state_packet_ref is None
            assert runtime_case.decision_boundary is None
        else:
            assert case.expected_state_fields
            assert case.required_external_source_id == "who-total-fat-weight-gain-2023"
            assert case.required_recommendation_ids == ("recommendation-1",)
            assert runtime_case.state_packet_ref is not None

    expected_runtime_keys = {
        "case_id", "user_id", "question", "decision_boundary", "state_packet_ref",
        "runtime_capability_context_ref",
    }
    assert set(runtime[0].to_dict()) == expected_runtime_keys
    assert E5RuntimeCase.from_dict(runtime[0].to_dict()) == runtime[0]
    forbidden = {"task_family", "expected_state_fields", "required_external_source_id", "oracle_action"}
    assert not forbidden.intersection(runtime[0].to_dict())
    with pytest.raises(ValueError, match="forbidden fields"):
        E5RuntimeCase.from_dict({**runtime[0].to_dict(), "task_family": "T0"})


def test_t1_source_assignment_alternates_only_between_eligible_sources() -> None:
    _, teacher = _build()
    t1 = [case for case in teacher if case.task_family == "T1"]

    assert [case.required_external_source_id for case in t1[::2]] == [
        "who-physical-activity-sedentary-2020"
    ] * 10
    assert [case.required_external_source_id for case in t1[1::2]] == [
        "who-total-fat-weight-gain-2023"
    ] * 10
    manifest = json.loads((ROOT / "runs/rag_e5/external_corpus_manifest.json").read_text())
    hypertension = next(
        source for source in manifest["guideline_candidates"]
        if source["source_id"] == "who-hypertension-pharmacological-2021"
    )
    assert hypertension["task_authoring_eligible"] is False
    assert all(case.required_external_source_id != hypertension["source_id"] for case in teacher)


def test_question_has_no_family_action_or_source_leakage_and_context_is_identical() -> None:
    runtime, _ = _build()
    audit = audit_questions([case.question for case in runtime])

    assert audit["question_leakage_gate"] == "PASS"
    assert audit["forbidden_token_counts"] == {}
    assert {case.runtime_capability_context_ref for case in runtime} == {"capability_context.json"}
    assert CAPABILITY_CONTEXT["available_retrieval_actions"] == ["OFF", "STANDARD", "STRONG"]
    assert CAPABILITY_CONTEXT["external_corpus_identity"] == (
        "9b19ad467f47641032277707cb1cfdb1d05c2fb39c558039b180efc7394692bd"
    )


def test_latest_boundary_missing_required_metric_is_not_rescued() -> None:
    bad_packet = _state_packets((USER_IDS[0],))[0]
    bad_packet["recent_measurement_trends"] = []

    with pytest.raises(ValueError, match="coverage failed; metric fallback is forbidden"):
        build_overlay_cases(state_packets=[bad_packet], eligible_chunks=_eligible_chunks())


def test_state_builder_source_manifest_must_match_the_pinned_202607_users() -> None:
    _validate_source_manifest(
        {
            "batches": {
                "202607": {
                    "role": "development",
                    "user_count": 20,
                    "users": list(DEVELOPMENT_USERS),
                }
            }
        }
    )
    wrong_users = list(DEVELOPMENT_USERS)
    wrong_users[-1] = "unapproved-user"
    with pytest.raises(ValueError, match="identity changed"):
        _validate_source_manifest(
            {
                "batches": {
                    "202607": {
                        "role": "development",
                        "user_count": 20,
                        "users": wrong_users,
                    }
                }
            }
        )


def _assert_overlay_has_no_model_or_retriever_calls() -> None:
    source = (ROOT / "eval/rag_e5/overlay.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    imports.extend(alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names)
    calls = [
        node.func.id if isinstance(node.func, ast.Name) else node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
    ]

    assert not any("retriev" in name.casefold() or "model" in name.casefold() for name in imports if name)
    assert not any("retriev" in name.casefold() or "model" in name.casefold() for name in calls)
    assert len(_build()[0]) == 60


def test_task_construction_does_not_call_model() -> None:
    _assert_overlay_has_no_model_or_retriever_calls()


def test_task_construction_does_not_call_retriever() -> None:
    _assert_overlay_has_no_model_or_retriever_calls()


def test_t2_maximum_drops_without_state_or_evidence_for_every_user() -> None:
    runtime, teachers = _build()
    runtime_by_id = {case.case_id: case for case in runtime}
    t2 = [case for case in teachers if case.task_family == "T2"]
    chunk = _eligible_chunks()[1]
    perfect_output = {
        "state_facts": [{"field": "body_weight", "value": "rising"}],
        "guidance_facts": [
            {"statement": "Adults aged 20 years and older should limit total fat to 30% of total energy or less."},
            {"statement": "Adults already below 30% should not increase fat intake to reach the threshold."},
        ],
        "citations": ["fat-core"],
        "answer": "Separated state and guidance.",
    }

    assert READER_SCHEMA["additionalProperties"] is False
    for teacher in t2:
        assert runtime_by_id[teacher.case_id].state_packet_ref
        full = score_case(
            teacher=teacher.to_dict(), reader_output=perfect_output, supplied_chunks=[chunk]
        )
        without_state = score_case(
            teacher=teacher.to_dict(), reader_output=perfect_output, supplied_chunks=[chunk], remove_state=True
        )
        without_evidence = score_case(
            teacher=teacher.to_dict(), reader_output=perfect_output, supplied_chunks=[chunk],
            remove_evidence_and_citations=True,
        )
        assert full["quality"] == 1.0
        assert without_state["quality"] <= 0.5
        assert without_evidence["quality"] <= 0.5


def test_frozen_manifests_contain_no_counterfactual_outcomes() -> None:
    task_manifest = json.loads((ROOT / "runs/rag_e5/e5b1_task_manifest.json").read_text())
    lock = json.loads((ROOT / "runs/rag_e5/e5b_counterfactual_lock.json").read_text())
    audit = json.loads((ROOT / "runs/rag_e5/e5b1_leakage_audit.json").read_text())

    assert task_manifest["task_family_counts"] == {"T0": 20, "T1": 20, "T2": 20}
    assert task_manifest["counterfactual_outcomes_created"] is False
    assert task_manifest["model_calls"] == task_manifest["retrieval_calls"] == 0
    assert lock["status"] == "FROZEN_PROTOCOL_NO_OUTCOMES"
    assert lock["outcomes_or_oracle_action_created"] is False
    assert audit["leakage_gate"] == "PASS"
    assert audit["202608_profile_timeline_exam_questions_answers_opened"] is False
