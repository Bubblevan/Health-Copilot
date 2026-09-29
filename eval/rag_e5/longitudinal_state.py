"""Build bounded, deterministic, decision-time state packets from pinned ESL state."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from hashlib import sha256
from typing import Any

STATE_PACKET_BUILDER_VERSION = "e5-longitudinal-state-v2"
PROFILE_TEMPORAL_CONTRACT = {
    "schema_version": "rag-e5-profile-temporal-contract-v1",
    "time_invariant_safe": {
        "demographics.age": "materialize_only_as_age_bucket",
    },
    "baseline_safe_if_proven": [],
    "temporally_unsafe": [
        "health_profile.chronic_conditions",
        "health_profile.past_medical_history",
        "health_profile.summary",
        "health_profile.patient_narrative",
        "health_profile.mental_health",
        "health_profile.family_history",
    ],
    "excluded_from_policy_features": ["profile.metadata"],
    "condition_categories_source": [
        "timeline.entry_type",
        "timeline.indicator",
        "exam_data.exam_type",
        "exam_data.indicators.keys",
    ],
    "condition_category_cutoff": "record_timestamp <= decision_timestamp",
    "date_only_exam_on_decision_date": "exclude_conservatively",
}
PROFILE_TEMPORAL_CONTRACT_SHA256 = sha256(
    json.dumps(PROFILE_TEMPORAL_CONTRACT, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()
STATE_PACKET_CONFIG = {
    "age_buckets": [18, 40, 60, 75],
    "max_condition_categories": 12,
    "max_exam_categories": 12,
    "max_recent_event_ids": 32,
    "max_trend_samples": 5,
    "recent_window_days": 30,
    "trend_tolerance_relative": 0.02,
    "profile_temporal_contract_sha256": PROFILE_TEMPORAL_CONTRACT_SHA256,
    "condition_categories_source": "pre-cutoff structured timeline/exam labels only",
}
STATE_PACKET_CONFIG_SHA256 = sha256(
    json.dumps(STATE_PACKET_CONFIG, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()

_FORBIDDEN_KEYS = frozenset(
    {
        "answer",
        "answer_key",
        "correct_answer",
        "counterfactual_outcome",
        "external_evidence",
        "gold",
        "gold_evidence_group",
        "gold_source",
        "oracle_action",
        "outcome_off",
        "outcome_standard",
        "outcome_strong",
        "required_external_evidence_group",
        "required_internal_state_fields",
        "task_family",
        "teacher_action",
        "teacher_metadata",
    }
)
_DOMAIN_PATTERNS = {
    "hypertension": re.compile(r"hypertension|blood[ _-]?pressure|systolic|diastolic", re.IGNORECASE),
    "diabetes": re.compile(r"diabetes|glucose|hba1c|glycated|insulin|blood[ _-]?sugar", re.IGNORECASE),
    "weight_metabolic": re.compile(r"weight|bmi|body[ _-]?mass|obes|overweight|waist|lipid", re.IGNORECASE),
    "physical_activity": re.compile(r"physical[ _-]?activity|exercise|steps|sedentary|activity", re.IGNORECASE),
    "nutrition": re.compile(r"diet|nutrition|food|calorie|carbohydrate|fiber|fibre|fat[ _-]?intake", re.IGNORECASE),
    "cardiovascular": re.compile(r"cardio|heart|cardiac|ecg|pulse|heart[ _-]?rate", re.IGNORECASE),
    "kidney": re.compile(r"kidney|renal|creatinine|egfr|albuminuria", re.IGNORECASE),
    "respiratory": re.compile(r"respirat|pulmonary|lung|spo2|oxygen|peak[ _-]?flow", re.IGNORECASE),
    "sleep": re.compile(r"sleep|insomnia|apnea|apnoea", re.IGNORECASE),
}
_METRIC_PATTERNS = (
    ("systolic_blood_pressure", re.compile(r"systolic", re.IGNORECASE)),
    ("diastolic_blood_pressure", re.compile(r"diastolic", re.IGNORECASE)),
    ("body_weight", re.compile(r"body[ _-]?weight|weight", re.IGNORECASE)),
    ("body_mass_index", re.compile(r"body[ _-]?mass[ _-]?index|\bbmi\b", re.IGNORECASE)),
    ("glycemic_measure", re.compile(r"glucose|hba1c|glycated", re.IGNORECASE)),
    ("daily_steps", re.compile(r"step[ _-]?count", re.IGNORECASE)),
    ("exercise_duration", re.compile(r"exercise[ _-]?duration|exercise[ _-]?minutes", re.IGNORECASE)),
)
_MAX_SUMMARY_CHARS = 2400


@dataclass(frozen=True, slots=True)
class LongitudinalStatePacket:
    """Bounded runtime summary with deterministic source and record provenance."""

    user_id: str
    decision_timestamp: str
    source_file_hashes: tuple[tuple[str, str], ...]
    included_record_counts: tuple[tuple[str, int], ...]
    included_record_ids: tuple[str, ...]
    age_bucket: str
    condition_categories: tuple[str, ...]
    recent_measurement_trends: tuple[tuple[str, str], ...]
    available_exam_categories: tuple[str, ...]
    history_span_days: int
    recent_event_count: int
    state_summary: str
    summary_builder_version: str
    summary_config_sha256: str
    profile_temporal_contract_sha256: str
    packet_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "decision_timestamp": self.decision_timestamp,
            "source_file_hashes": dict(self.source_file_hashes),
            "included_record_counts": dict(self.included_record_counts),
            "included_record_ids": list(self.included_record_ids),
            "age_bucket": self.age_bucket,
            "condition_categories": list(self.condition_categories),
            "recent_measurement_trends": [
                {"metric": metric, "trend": trend}
                for metric, trend in self.recent_measurement_trends
            ],
            "available_exam_categories": list(self.available_exam_categories),
            "history_span_days": self.history_span_days,
            "recent_event_count": self.recent_event_count,
            "state_summary": self.state_summary,
            "summary_builder_version": self.summary_builder_version,
            "summary_config_sha256": self.summary_config_sha256,
            "profile_temporal_contract_sha256": self.profile_temporal_contract_sha256,
            "packet_sha256": self.packet_sha256,
        }


def build_longitudinal_state_packet(
    *,
    user_id: str,
    decision_timestamp: str,
    profile: Mapping[str, Any],
    timeline: Mapping[str, Any],
    exam_data: Sequence[Mapping[str, Any]],
    source_file_hashes: Mapping[str, str],
) -> LongitudinalStatePacket:
    """Summarize only allowlisted longitudinal fields up to the decision cutoff.

    Event prose, patient narrative, retrieved material, and evaluator annotations
    are never copied into the summary. Date-only exam records are conservatively
    excluded on the decision date because their time-of-day is unknown.
    """
    if not isinstance(user_id, str):
        raise TypeError("user_id must be a string")
    if not user_id.strip():
        raise ValueError("user_id must be non-empty")
    cutoff = _parse_timestamp(decision_timestamp)
    if not isinstance(profile, Mapping) or not isinstance(timeline, Mapping):
        raise TypeError("profile and timeline must be mappings")
    if not isinstance(exam_data, Sequence) or isinstance(exam_data, (str, bytes)):
        raise TypeError("exam_data must be a sequence of exam records")
    _reject_forbidden_keys({"profile": profile, "timeline": timeline, "exam_data": exam_data})
    hashes = _normalize_source_hashes(source_file_hashes)

    demographics = profile.get("demographics", {})
    if not isinstance(demographics, Mapping):
        raise TypeError("profile.demographics must be a mapping")
    age_bucket = _age_bucket(demographics.get("age"))

    entries = timeline.get("entries", ())
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        raise TypeError("timeline.entries must be a sequence")
    included_events: list[tuple[datetime, int, Mapping[str, Any]]] = []
    trend_samples: dict[str, list[tuple[datetime, float]]] = {}
    for index, row in enumerate(entries):
        if not isinstance(row, Mapping):
            raise TypeError("timeline entries must be objects")
        timestamp = _parse_timestamp(row.get("time"))
        if timestamp > cutoff:
            continue
        included_events.append((timestamp, index, row))
        metric = _metric_bucket(row.get("indicator"))
        numeric = _numeric_value(row.get("value"))
        if metric is not None and numeric is not None:
            samples = trend_samples.setdefault(metric, [])
            samples.append((timestamp, numeric))
            samples.sort(key=lambda sample: sample[0])
            del samples[:-int(STATE_PACKET_CONFIG["max_trend_samples"])]

    exams_included: list[tuple[datetime, int, Mapping[str, Any]]] = []
    exam_categories: set[str] = set()
    for index, exam in enumerate(exam_data):
        if not isinstance(exam, Mapping):
            raise TypeError("exam_data rows must be objects")
        exam_timestamp, date_only = _parse_exam_date(exam.get("exam_date"))
        if exam_timestamp > cutoff or (date_only and exam_timestamp.date() == cutoff.date()):
            continue
        exams_included.append((exam_timestamp, index, exam))
        exam_categories.update(_categories_from_label(exam.get("exam_type")))
        indicators = exam.get("indicators", {})
        if isinstance(indicators, Mapping):
            for indicator in indicators:
                exam_categories.update(_categories_from_label(indicator))

    condition_categories = _time_bounded_condition_categories(included_events, exams_included)

    included_events.sort(key=lambda item: (item[0], item[1]))
    exams_included.sort(key=lambda item: (item[0], item[1]))
    event_times = [item[0] for item in included_events]
    history_span_days = (
        max(0, (event_times[-1].date() - event_times[0].date()).days) if event_times else 0
    )
    recent_cutoff = cutoff - timedelta(days=int(STATE_PACKET_CONFIG["recent_window_days"]))
    recent_event_count = sum(timestamp >= recent_cutoff for timestamp in event_times)
    trends = tuple(
        sorted(
            (metric, _trend(samples))
            for metric, samples in trend_samples.items()
            if len(samples) >= 2
        )
    )

    max_ids = int(STATE_PACKET_CONFIG["max_recent_event_ids"])
    recent_events = included_events[-max_ids:]
    record_ids = [
        _record_id("timeline", index, timestamp)
        for timestamp, index, _row in recent_events
    ]
    record_ids.extend(
        _record_id("exam_data", index, timestamp)
        for timestamp, index, _exam in exams_included[-max_ids:]
    )

    condition_categories = condition_categories[: int(STATE_PACKET_CONFIG["max_condition_categories"])]
    available_exam_categories = tuple(
        sorted(exam_categories)[: int(STATE_PACKET_CONFIG["max_exam_categories"])]
    )
    summary_payload = {
        "age_bucket": age_bucket,
        "condition_categories": list(condition_categories),
        "recent_measurement_trends": [
            {"metric": metric, "trend": trend} for metric, trend in trends
        ],
        "available_exam_categories": list(available_exam_categories),
        "history_span_days": history_span_days,
        "recent_event_count": recent_event_count,
    }
    state_summary = json.dumps(
        summary_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    if len(state_summary) > _MAX_SUMMARY_CHARS:
        raise ValueError("deterministic state summary exceeded its frozen character budget")

    packet_without_hash: dict[str, Any] = {
        "user_id": user_id,
        "decision_timestamp": cutoff.isoformat().replace("+00:00", "Z"),
        "source_file_hashes": dict(hashes),
        "included_record_counts": {
            "profile": 1,
            "timeline": len(included_events),
            "exam_data": len(exams_included),
        },
        "included_record_ids": record_ids,
        **summary_payload,
        "state_summary": state_summary,
        "summary_builder_version": STATE_PACKET_BUILDER_VERSION,
        "summary_config_sha256": STATE_PACKET_CONFIG_SHA256,
        "profile_temporal_contract_sha256": PROFILE_TEMPORAL_CONTRACT_SHA256,
    }
    packet_sha256 = sha256(
        json.dumps(packet_without_hash, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return LongitudinalStatePacket(
        user_id=user_id,
        decision_timestamp=packet_without_hash["decision_timestamp"],
        source_file_hashes=hashes,
        included_record_counts=tuple(sorted(packet_without_hash["included_record_counts"].items())),
        included_record_ids=tuple(record_ids),
        age_bucket=age_bucket,
        condition_categories=tuple(condition_categories),
        recent_measurement_trends=trends,
        available_exam_categories=available_exam_categories,
        history_span_days=history_span_days,
        recent_event_count=recent_event_count,
        state_summary=state_summary,
        summary_builder_version=STATE_PACKET_BUILDER_VERSION,
        summary_config_sha256=STATE_PACKET_CONFIG_SHA256,
        profile_temporal_contract_sha256=PROFILE_TEMPORAL_CONTRACT_SHA256,
        packet_sha256=packet_sha256,
    )


def build_longitudinal_state_packet_from_payload(
    payload: Mapping[str, Any], *, source_file_hashes: Mapping[str, str]
) -> LongitudinalStatePacket:
    """Fail closed on unknown task/evaluator payload keys before state building."""
    if not isinstance(payload, Mapping):
        raise TypeError("state payload must be a mapping")
    allowed = {"user_id", "decision_timestamp", "profile", "timeline", "exam_data"}
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"state payload contains non-state fields: {sorted(unknown)}")
    missing = allowed - set(payload)
    if missing:
        raise ValueError(f"state payload is missing fields: {sorted(missing)}")
    return build_longitudinal_state_packet(
        user_id=payload["user_id"],
        decision_timestamp=payload["decision_timestamp"],
        profile=payload["profile"],
        timeline=payload["timeline"],
        exam_data=payload["exam_data"],
        source_file_hashes=source_file_hashes,
    )


def _reject_forbidden_keys(value: Any, path: str = "state") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            normalized = key.casefold()
            if normalized in _FORBIDDEN_KEYS:
                raise ValueError(f"teacher/evaluator field is forbidden in state input: {path}.{key}")
            _reject_forbidden_keys(nested, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, nested in enumerate(value):
            _reject_forbidden_keys(nested, f"{path}[{index}]")


def _normalize_source_hashes(value: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    expected = {"profile.json", "timeline.json", "exam_data.json"}
    if set(value) != expected:
        raise ValueError("source_file_hashes must pin exactly the three allowed state files")
    for name, digest in value.items():
        if not isinstance(digest, str) or len(digest) != 64 or any(
            char not in "0123456789abcdef" for char in digest
        ):
            raise ValueError(f"invalid SHA-256 for {name}")
    return tuple(sorted(value.items()))


def _parse_timestamp(value: object) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("timestamps must be non-empty ISO-8601 strings")
    try:
        parsed = datetime.fromisoformat(value.strip())
    except ValueError as exc:
        raise ValueError("timestamps must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise ValueError("timestamps must include a timezone")
    return parsed.astimezone(UTC)


def _parse_exam_date(value: object) -> tuple[datetime, bool]:
    if isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value.strip()):
        return datetime.combine(date.fromisoformat(value.strip()), time.min, UTC), True
    return _parse_timestamp(value), False


def _age_bucket(value: object) -> str:
    if isinstance(value, bool):
        return "unknown"
    try:
        age = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if not 0 <= age <= 125:
        return "unknown"
    if age < 18:
        return "under_18"
    if age < 40:
        return "18_39"
    if age < 60:
        return "40_59"
    if age < 75:
        return "60_74"
    return "75_plus"


def _time_bounded_condition_categories(
    timeline_entries: Sequence[tuple[datetime, int, Mapping[str, Any]]],
    exam_records: Sequence[tuple[datetime, int, Mapping[str, Any]]],
) -> tuple[str, ...]:
    """Derive coarse observed domains from structured, pre-cutoff record labels only."""
    categories: set[str] = set()
    for _timestamp, _index, row in timeline_entries:
        categories.update(_categories_from_label(row.get("entry_type")))
        categories.update(_categories_from_label(row.get("indicator")))
    for _timestamp, _index, exam in exam_records:
        categories.update(_categories_from_label(exam.get("exam_type")))
        indicators = exam.get("indicators", {})
        if isinstance(indicators, Mapping):
            for indicator in indicators:
                categories.update(_categories_from_label(indicator))
    return tuple(sorted(categories))


def _category_labels(value: Any) -> set[str]:
    labels: set[str] = set()
    if isinstance(value, str):
        labels.add(value)
    elif isinstance(value, Mapping):
        for key, nested in value.items():
            if isinstance(key, str):
                labels.add(key)
            if isinstance(nested, (str, Mapping, list, tuple)):
                labels.update(_category_labels(nested))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for nested in value:
            labels.update(_category_labels(nested))
    return labels


def _categories_from_label(value: object) -> set[str]:
    if not isinstance(value, str):
        return set()
    return {domain for domain, pattern in _DOMAIN_PATTERNS.items() if pattern.search(value)}


def _metric_bucket(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    for metric, pattern in _METRIC_PATTERNS:
        if pattern.search(value):
            return metric
    return None


def _numeric_value(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _trend(samples: list[tuple[datetime, float]]) -> str:
    ordered = sorted(samples, key=lambda sample: sample[0])
    first = ordered[0][1]
    last = ordered[-1][1]
    tolerance = max(abs(first) * float(STATE_PACKET_CONFIG["trend_tolerance_relative"]), 1e-9)
    if last - first > tolerance:
        return "rising"
    if first - last > tolerance:
        return "falling"
    return "stable"


def _record_id(source: str, index: int, timestamp: datetime) -> str:
    identity = f"{source}\n{index}\n{timestamp.isoformat()}"
    return sha256(identity.encode("utf-8")).hexdigest()


__all__ = [
    "PROFILE_TEMPORAL_CONTRACT",
    "PROFILE_TEMPORAL_CONTRACT_SHA256",
    "STATE_PACKET_BUILDER_VERSION",
    "STATE_PACKET_CONFIG_SHA256",
    "LongitudinalStatePacket",
    "build_longitudinal_state_packet",
    "build_longitudinal_state_packet_from_payload",
]
