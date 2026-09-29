from __future__ import annotations

import copy

import pytest

from eval.rag_e5.longitudinal_state import (
    PROFILE_TEMPORAL_CONTRACT_SHA256,
    STATE_PACKET_BUILDER_VERSION,
    build_longitudinal_state_packet,
    build_longitudinal_state_packet_from_payload,
)
from eval.rag_e5.temporal import SourceRelativeDecisionBoundary


def _fixture() -> tuple[dict[str, object], dict[str, str]]:
    payload: dict[str, object] = {
        "decision_boundary": {
            "user_id": "synthetic-user",
            "naive_timestamp": "2026-01-03T12:00:00",
            "semantics_id": "SOURCE_RELATIVE_NAIVE_CIVIL_TIME_V1",
        },
        "profile": {
            "demographics": {"age": 62},
            "health_profile": {
                "chronic_conditions": ["hypertension"],
                "patient_narrative": "private narrative must not enter the state summary",
                "lifestyle": {"diet_narrative": "not copied to the summary"},
            },
        },
        "timeline": {
            "user_id": "synthetic-user",
            "generated_at": "2026-01-05T12:00:00",
            "entries": [
                {
                    "time": "2026-01-01T08:00:00",
                    "entry_type": "device_indicator",
                    "indicator": "SystolicBloodPressure",
                    "value": 120,
                    "unit": "mmHg",
                    "event": "ignored narrative",
                },
                {
                    "time": "2026-01-02T08:00:00",
                    "entry_type": "device_indicator",
                    "indicator": "SystolicBloodPressure",
                    "value": 130,
                    "unit": "mmHg",
                },
                {
                    "time": "2026-01-03T12:00:00",
                    "entry_type": "device_indicator",
                    "indicator": "SystolicBloodPressure",
                    "value": 210,
                },
                {
                    "time": "2026-01-04T08:00:00",
                    "entry_type": "device_indicator",
                    "indicator": "SystolicBloodPressure",
                    "value": 220,
                    "unit": "mmHg",
                },
            ]
        },
        "exam_data": [
            {
                "exam_date": "2026-01-01",
                "exam_type": "cardiovascular exam",
                "indicators": {"SystolicBloodPressure": 120},
            },
            {
                "exam_date": "2026-01-03",
                "exam_type": "future-on-cutoff-date exam",
                "indicators": {"BloodGlucose": 999},
            },
            {
                "exam_date": "2026-01-05",
                "exam_type": "future exam",
                "indicators": {"BloodGlucose": 999},
            },
        ],
    }
    hashes = {
        "profile.json": "a" * 64,
        "timeline.json": "b" * 64,
        "exam_data.json": "c" * 64,
    }
    return payload, hashes


def _build(payload: dict[str, object], hashes: dict[str, str]):
    return build_longitudinal_state_packet_from_payload(payload, source_file_hashes=hashes)


def test_state_packet_respects_decision_timestamp_and_excludes_future_events() -> None:
    payload, hashes = _fixture()

    packet = _build(payload, hashes)

    assert dict(packet.included_record_counts) == {"exam_data": 1, "profile": 1, "timeline": 2}
    assert packet.recent_event_count == 2
    assert packet.recent_measurement_trends == (("systolic_blood_pressure", "rising"),)
    assert "glucose" not in packet.state_summary.casefold()
    assert len(packet.included_record_ids) == 3


def test_future_timeline_event_cannot_change_history_or_state_summary() -> None:
    payload, hashes = _fixture()
    baseline = _build(payload, hashes)
    changed = copy.deepcopy(payload)
    changed["timeline"]["entries"][3]["value"] = -999  # type: ignore[index]
    changed["timeline"]["entries"][3]["time"] = "2030-01-01T00:00:00"  # type: ignore[index]

    future_changed = _build(changed, hashes)

    assert future_changed.state_summary == baseline.state_summary
    assert future_changed.packet_sha256 == baseline.packet_sha256
    assert future_changed.included_record_counts == baseline.included_record_counts


def test_future_exam_is_excluded_and_same_day_date_only_exam_is_conservative() -> None:
    payload, hashes = _fixture()
    packet = _build(payload, hashes)

    assert dict(packet.included_record_counts)["exam_data"] == 1
    assert packet.available_exam_categories == ("cardiovascular", "hypertension")


def test_state_summary_is_deterministic_bounded_and_does_not_copy_prose_or_values() -> None:
    payload, hashes = _fixture()

    first = _build(payload, hashes)
    second = _build(copy.deepcopy(payload), hashes)

    assert first == second
    assert len(first.state_summary) <= 2400
    assert "private narrative" not in first.state_summary
    assert "ignored narrative" not in first.state_summary
    assert "120" not in first.state_summary
    assert first.summary_builder_version == STATE_PACKET_BUILDER_VERSION
    assert first.summary_builder_version == "e5-longitudinal-state-v3"
    assert len(first.summary_config_sha256) == 64
    assert first.profile_temporal_contract_sha256 == PROFILE_TEMPORAL_CONTRACT_SHA256
    assert len(first.packet_sha256) == 64


def test_profile_clinical_fields_do_not_change_runtime_packet() -> None:
    payload, hashes = _fixture()
    baseline = _build(payload, hashes)
    changed = copy.deepcopy(payload)
    changed["profile"]["health_profile"].update(  # type: ignore[index]
        {
            "chronic_conditions": ["diabetes", "kidney disease"],
            "past_medical_history": ["stroke"],
            "summary": "profile summary must not affect decision-time state",
            "patient_narrative": "profile narrative must not affect decision-time state",
            "mental_health": {"depression": True},
            "family_history": ["heart disease"],
        }
    )

    changed_packet = _build(changed, hashes)

    assert changed_packet.condition_categories == baseline.condition_categories
    assert changed_packet.state_summary == baseline.state_summary
    assert changed_packet.packet_sha256 == baseline.packet_sha256


def test_condition_categories_use_only_pre_cutoff_structured_records() -> None:
    payload, hashes = _fixture()
    payload["profile"]["health_profile"]["chronic_conditions"] = [  # type: ignore[index]
        "hypertension",
        "diabetes",
    ]
    payload["timeline"]["entries"] = [  # type: ignore[index]
        {
            "time": "2026-01-01T08:00:00",
            "entry_type": "device_indicator",
            "indicator": "DailySteps",
            "value": 4000,
        },
        {
            "time": "2026-01-03T12:00:00",
            "entry_type": "device_indicator",
            "indicator": "SystolicBloodPressure",
            "value": 220,
        },
    ]
    payload["exam_data"] = [  # type: ignore[assignment]
        {
            "exam_date": "2026-01-02",
            "exam_type": "routine exam",
            "indicators": {"BodyWeight": 70},
        },
        {
            "exam_date": "2026-01-05",
            "exam_type": "hypertension exam",
            "indicators": {"BloodGlucose": 999},
        },
    ]

    packet = _build(payload, hashes)

    assert "hypertension" not in packet.condition_categories
    assert "diabetes" not in packet.condition_categories
    assert "physical_activity" in packet.condition_categories
    assert "weight_metabolic" in packet.condition_categories


def test_profile_metadata_is_not_a_policy_shortcut() -> None:
    payload, hashes = _fixture()
    baseline = _build(payload, hashes)
    changed = copy.deepcopy(payload)
    changed["profile"]["metadata"] = {  # type: ignore[index]
        "generation_type": "202607-only marker",
        "use_case": "future batch discriminator",
        "profile_id": "synthetic-id",
    }

    changed_packet = _build(changed, hashes)

    assert changed_packet.state_summary == baseline.state_summary
    assert changed_packet.packet_sha256 == baseline.packet_sha256


def test_timeline_generated_at_is_metadata_not_a_temporal_feature() -> None:
    payload, hashes = _fixture()
    baseline = _build(payload, hashes)
    changed = copy.deepcopy(payload)
    changed["timeline"]["generated_at"] = "not a usable timestamp"  # type: ignore[index]

    changed_packet = _build(changed, hashes)

    assert changed_packet.state_summary == baseline.state_summary
    assert changed_packet.packet_sha256 == baseline.packet_sha256


@pytest.mark.parametrize(
    "field,value",
    [
        ("task_family", "T2"),
        ("oracle_action", "STRONG"),
        ("required_external_evidence_group", "guideline"),
        ("counterfactual_outcome", {"standard": 1}),
    ],
)
def test_teacher_metadata_cannot_enter_state_packet(field: str, value: object) -> None:
    payload, hashes = _fixture()
    payload[field] = value

    with pytest.raises(ValueError, match="non-state fields"):
        _build(payload, hashes)


def test_nested_external_evidence_is_rejected_from_state_inputs() -> None:
    payload, hashes = _fixture()
    payload["timeline"]["entries"][0]["external_evidence"] = "retrieved guideline text"  # type: ignore[index]

    with pytest.raises(ValueError, match="teacher/evaluator field"):
        _build(payload, hashes)


def test_state_summary_uses_no_external_retrieval_or_teacher_inputs() -> None:
    payload, hashes = _fixture()
    packet = _build(payload, hashes)
    serialized = str(packet.to_dict()).casefold()

    assert "external_evidence" not in serialized
    assert "retrieval_result" not in serialized
    assert "teacher" not in serialized
    assert set(dict(packet.source_file_hashes)) == {"profile.json", "timeline.json", "exam_data.json"}


def test_state_packet_rejects_ambiguous_timestamps_and_incomplete_hashes() -> None:
    payload, hashes = _fixture()
    payload["timeline"]["entries"][0]["time"] = "2026-01-01T08:00:00Z"  # type: ignore[index]
    with pytest.raises(ValueError, match="without a zone"):
        _build(payload, hashes)

    payload, _ = _fixture()
    with pytest.raises(ValueError, match="exactly the three allowed"):
        build_longitudinal_state_packet(
            decision_boundary=SourceRelativeDecisionBoundary(
                user_id="synthetic-user", naive_timestamp="2026-01-03T12:00:00"
            ),
            profile=payload["profile"],  # type: ignore[arg-type]
            timeline=payload["timeline"],  # type: ignore[arg-type]
            exam_data=payload["exam_data"],  # type: ignore[arg-type]
            source_file_hashes={"profile.json": hashes["profile.json"]},
        )
