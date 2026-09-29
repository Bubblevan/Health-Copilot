from __future__ import annotations

import pytest

from eval.rag_e5.longitudinal_state import build_longitudinal_state_packet
from eval.rag_e5.temporal import (
    TEMPORAL_SEMANTICS,
    TEMPORAL_SEMANTICS_ID,
    TEMPORAL_SEMANTICS_SHA256,
    SourceRelativeDecisionBoundary,
    candidate_decision_boundaries,
    date_only_exam_is_visible,
    parse_esl_naive_datetime,
    sanitize_visible_event_entry,
)


def test_parser_accepts_only_naive_source_civil_times() -> None:
    parsed = parse_esl_naive_datetime("2026-03-04T05:06:07.123456")

    assert parsed.isoformat() == "2026-03-04T05:06:07.123456"
    assert parsed.tzinfo is None
    assert parse_esl_naive_datetime("2026-03-04 05:06:07").isoformat() == (
        "2026-03-04T05:06:07"
    )
    for value in ("2026-03-04T05:06:07Z", "2026-03-04T05:06:07+08:00"):
        with pytest.raises(ValueError, match="without a zone"):
            parse_esl_naive_datetime(value)


def test_temporal_manifest_freezes_relative_only_scope() -> None:
    manifest = TEMPORAL_SEMANTICS.to_dict()

    assert manifest == {
        "semantics_id": TEMPORAL_SEMANTICS_ID,
        "timezone_interpretation": "NONE",
        "cross_user_ordering_allowed": False,
        "cross_user_duration_allowed": False,
        "same_user_ordering_allowed": True,
        "date_only_same_day_visibility": "EXCLUDE",
        "equal_timestamp_visibility": "EXCLUDE",
    }
    assert len(TEMPORAL_SEMANTICS_SHA256) == 64


def test_source_relative_ordering_is_same_user_only() -> None:
    boundary = SourceRelativeDecisionBoundary("user-a", "2026-03-04T12:00:00")

    assert boundary.is_strictly_after_source_timestamp(
        source_user_id="user-a", source_timestamp="2026-03-04T11:59:59"
    )
    assert not boundary.is_strictly_after_source_timestamp(
        source_user_id="user-a", source_timestamp="2026-03-04T12:00:00"
    )
    with pytest.raises(ValueError, match="cross-user"):
        boundary.is_strictly_after_source_timestamp(
            source_user_id="user-b", source_timestamp="2026-03-04T11:59:59"
        )


def test_candidate_boundaries_group_equal_times_and_ignore_generated_at() -> None:
    timeline = {
        "user_id": "user-a",
        "generated_at": "2099-01-01T00:00:00",
        "entries": [
            {"time": "2026-03-03T10:00:00"},
            {"time": "2026-03-03T10:00:00"},
            {"time": "2026-03-04T10:00:00"},
        ],
    }

    boundaries = candidate_decision_boundaries(user_id="user-a", timeline=timeline)

    assert [item.naive_timestamp for item in boundaries] == [
        "2026-03-03T10:00:00",
        "2026-03-04T10:00:00",
    ]
    assert all(item.semantics_id == TEMPORAL_SEMANTICS_ID for item in boundaries)


def test_date_only_exam_requires_strictly_earlier_calendar_date() -> None:
    boundary = SourceRelativeDecisionBoundary("user-a", "2026-03-04T00:00:00")

    assert date_only_exam_is_visible("2026-03-03", boundary)
    assert not date_only_exam_is_visible("2026-03-04", boundary)
    assert not date_only_exam_is_visible("2026-03-05", boundary)


def test_event_sanitizer_redacts_future_completion_fields() -> None:
    boundary = SourceRelativeDecisionBoundary("user-a", "2026-03-04T12:00:00")
    ongoing = {
        "entry_type": "event",
        "time": "2026-03-01T08:00:00",
        "end_time": "2026-03-06T08:00:00",
        "event": {
            "description": "must never be copied",
            "duration_days": 5,
            "interrupted": True,
        },
        "private_payload": "also never copied",
    }

    visible = sanitize_visible_event_entry(
        ongoing, source_user_id="user-a", boundary=boundary
    )

    assert visible == {
        "entry_type": "event",
        "start_time": "2026-03-01T08:00:00",
        "status_at_boundary": "ongoing_or_unknown",
    }
    assert "2026-03-06" not in str(visible)
    assert "duration_days" not in visible
    assert "interrupted" not in visible
    assert "description" not in visible


def test_completed_event_fields_visible_only_after_end() -> None:
    event = {
        "entry_type": "event",
        "time": "2026-03-01T08:00:00",
        "end_time": "2026-03-02T08:00:00",
        "event": {"duration_days": 1, "interrupted": False},
    }
    at_end = SourceRelativeDecisionBoundary("user-a", "2026-03-02T08:00:00")
    after_end = SourceRelativeDecisionBoundary("user-a", "2026-03-03T08:00:00")

    assert "end_time" not in sanitize_visible_event_entry(
        event, source_user_id="user-a", boundary=at_end
    )
    assert sanitize_visible_event_entry(
        event, source_user_id="user-a", boundary=after_end
    ) == {
        "entry_type": "event",
        "start_time": "2026-03-01T08:00:00",
        "status_at_boundary": "completed",
        "end_time": "2026-03-02T08:00:00",
        "duration_days": 1,
        "interrupted": False,
    }


def test_state_packet_excludes_every_record_at_equal_boundary_and_is_order_invariant() -> None:
    boundary = SourceRelativeDecisionBoundary("user-a", "2026-03-03T10:00:00")
    rows = [
        {
            "time": "2026-03-01T10:00:00",
            "entry_type": "device_indicator",
            "indicator": "SystolicBloodPressure",
            "value": 120,
        },
        {
            "time": "2026-03-02T10:00:00",
            "entry_type": "device_indicator",
            "indicator": "SystolicBloodPressure",
            "value": 130,
        },
        {
            "time": "2026-03-03T10:00:00",
            "entry_type": "device_indicator",
            "indicator": "SystolicBloodPressure",
            "value": 999,
        },
        {
            "time": "2026-03-03T10:00:00",
            "entry_type": "event",
            "event": "same-time event must be excluded",
        },
    ]
    kwargs = {
        "decision_boundary": boundary,
        "profile": {"demographics": {"age": 50}},
        "timeline": {"user_id": "user-a", "entries": rows},
        "exam_data": [],
        "source_file_hashes": {
            "profile.json": "a" * 64,
            "timeline.json": "b" * 64,
            "exam_data.json": "c" * 64,
        },
    }

    first = build_longitudinal_state_packet(**kwargs)
    second = build_longitudinal_state_packet(
        **{**kwargs, "timeline": {"user_id": "user-a", "entries": list(reversed(rows))}}
    )

    assert dict(first.included_record_counts)["timeline"] == 2
    assert first == second
    assert first.temporal_semantics_id == TEMPORAL_SEMANTICS_ID
    assert first.temporal_semantics_sha256 == TEMPORAL_SEMANTICS_SHA256
    assert first.to_dict()["decision_boundary"]["naive_timestamp"] == "2026-03-03T10:00:00"
    assert "generated_at" not in str(first.to_dict())


def test_ambiguous_same_time_metric_group_does_not_create_a_trend() -> None:
    boundary = SourceRelativeDecisionBoundary("user-a", "2026-03-03T10:00:00")
    rows = [
        {
            "time": "2026-03-01T10:00:00",
            "entry_type": "device_indicator",
            "indicator": "SystolicBloodPressure",
            "value": 80,
        },
        {
            "time": "2026-03-01T10:00:00",
            "entry_type": "device_indicator",
            "indicator": "SystolicBloodPressure",
            "value": 180,
        },
        {
            "time": "2026-03-02T10:00:00",
            "entry_type": "device_indicator",
            "indicator": "SystolicBloodPressure",
            "value": 100,
        },
        {
            "time": "2026-03-03T10:00:00",
            "entry_type": "event",
            "event": "decision boundary source timestamp",
        },
    ]
    packet = build_longitudinal_state_packet(
        decision_boundary=boundary,
        profile={"demographics": {}},
        timeline={"user_id": "user-a", "entries": rows},
        exam_data=[],
        source_file_hashes={
            "profile.json": "a" * 64,
            "timeline.json": "b" * 64,
            "exam_data.json": "c" * 64,
        },
    )

    assert packet.recent_measurement_trends == ()


def test_recent_record_cap_never_splits_a_timestamp_group() -> None:
    boundary = SourceRelativeDecisionBoundary("user-a", "2026-03-03T10:00:00")
    rows = [
        {
            "time": "2026-03-02T10:00:00",
            "entry_type": "event",
            "event": f"opaque event {index}",
        }
        for index in range(40)
    ]
    packet = build_longitudinal_state_packet(
        decision_boundary=boundary,
        profile={"demographics": {}},
        timeline={"user_id": "user-a", "entries": [*rows, {"time": boundary.naive_timestamp,
              "entry_type": "event", "event": "boundary"}]},
        exam_data=[],
        source_file_hashes={
            "profile.json": "a" * 64,
            "timeline.json": "b" * 64,
            "exam_data.json": "c" * 64,
        },
    )

    assert dict(packet.included_record_counts)["timeline"] == 40
    assert len(packet.included_record_ids) == 40


def test_boundary_rejects_unknown_semantics_and_foreign_timeline() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        SourceRelativeDecisionBoundary("user-a", "2026-03-03T10:00:00", "UTC")
    with pytest.raises(ValueError, match="timeline user_id"):
        candidate_decision_boundaries(
            user_id="user-a",
            timeline={"user_id": "user-b", "entries": [{"time": "2026-03-03T10:00:00"}]},
        )
