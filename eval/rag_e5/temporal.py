"""Source-relative temporal semantics for the ESL-derived E5 development cohort."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date, datetime
from hashlib import sha256
from typing import Any

TEMPORAL_SEMANTICS_ID = "SOURCE_RELATIVE_NAIVE_CIVIL_TIME_V1"
_NAIVE_DATETIME_PATTERN = re.compile(
    r"\A\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?\Z"
)
_DATE_PATTERN = re.compile(r"\A\d{4}-\d{2}-\d{2}\Z")


@dataclass(frozen=True, slots=True)
class TemporalSemanticsManifest:
    semantics_id: str = TEMPORAL_SEMANTICS_ID
    timezone_interpretation: str = "NONE"
    cross_user_ordering_allowed: bool = False
    cross_user_duration_allowed: bool = False
    same_user_ordering_allowed: bool = True
    date_only_same_day_visibility: str = "EXCLUDE"
    equal_timestamp_visibility: str = "EXCLUDE"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return sha256(encoded).hexdigest()


TEMPORAL_SEMANTICS = TemporalSemanticsManifest()
TEMPORAL_SEMANTICS_SHA256 = TEMPORAL_SEMANTICS.sha256


def parse_esl_naive_datetime(value: object) -> datetime:
    """Parse source-native civil timestamps without assigning or converting zones."""
    if not isinstance(value, str) or not _NAIVE_DATETIME_PATTERN.fullmatch(value):
        raise ValueError(
            "ESL timestamps must be YYYY-MM-DD[T ]HH:MM:SS[.ffffff] without a zone"
        )
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("ESL timestamp is not a valid calendar datetime") from exc
    if parsed.tzinfo is not None:
        raise ValueError("ESL source-relative timestamps must not include a timezone")
    return parsed


def parse_esl_date(value: object) -> date:
    """Parse an exact date-only source value; do not synthesize a time of day."""
    if not isinstance(value, str) or not _DATE_PATTERN.fullmatch(value):
        raise ValueError("ESL date-only values must use YYYY-MM-DD")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("ESL date-only value is not a valid calendar date") from exc


@dataclass(frozen=True, slots=True)
class SourceRelativeDecisionBoundary:
    user_id: str
    naive_timestamp: str
    semantics_id: str = TEMPORAL_SEMANTICS_ID

    def __post_init__(self) -> None:
        if not isinstance(self.user_id, str) or not self.user_id.strip():
            raise ValueError("decision boundary user_id must be non-empty")
        if self.semantics_id != TEMPORAL_SEMANTICS_ID:
            raise ValueError("decision boundary uses an unsupported temporal semantics id")
        parsed = parse_esl_naive_datetime(self.naive_timestamp)
        object.__setattr__(self, "naive_timestamp", parsed.isoformat())

    @property
    def timestamp(self) -> datetime:
        return parse_esl_naive_datetime(self.naive_timestamp)

    def is_strictly_after_source_timestamp(
        self, *, source_user_id: str, source_timestamp: object
    ) -> bool:
        """Whether a same-user record is strictly visible before this boundary."""
        if source_user_id != self.user_id:
            raise ValueError("cross-user temporal comparison is forbidden")
        if self.semantics_id != TEMPORAL_SEMANTICS_ID:
            raise ValueError("temporal semantics mismatch")
        return parse_esl_naive_datetime(source_timestamp) < self.timestamp

    def to_dict(self) -> dict[str, str]:
        return {
            "user_id": self.user_id,
            "naive_timestamp": self.timestamp.isoformat(),
            "semantics_id": self.semantics_id,
        }


def candidate_decision_boundaries(
    *, user_id: str, timeline: Mapping[str, Any]
) -> tuple[SourceRelativeDecisionBoundary, ...]:
    """Return sorted, unique observed timestamps; generated_at is never consulted."""
    if timeline.get("user_id") != user_id:
        raise ValueError("timeline user_id must match the requested source user")
    entries = timeline.get("entries")
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        raise TypeError("timeline.entries must be a sequence")
    observed: set[datetime] = set()
    for row in entries:
        if not isinstance(row, Mapping):
            raise TypeError("timeline entries must be objects")
        observed.add(parse_esl_naive_datetime(row.get("time")))
    return tuple(
        SourceRelativeDecisionBoundary(user_id, timestamp.isoformat())
        for timestamp in sorted(observed)
    )


def latest_decision_boundary(
    *, user_id: str, timeline: Mapping[str, Any]
) -> SourceRelativeDecisionBoundary:
    """Select the latest unique observed timestamp; never rescue from another source."""
    boundaries = candidate_decision_boundaries(user_id=user_id, timeline=timeline)
    if not boundaries:
        raise ValueError("timeline has no observed decision boundary")
    return boundaries[-1]


def date_only_exam_is_visible(exam_date: object, boundary: SourceRelativeDecisionBoundary) -> bool:
    """A date-only exam is visible only on a strictly earlier calendar date."""
    return parse_esl_date(exam_date) < boundary.timestamp.date()


def sanitize_visible_event_entry(
    entry: Mapping[str, Any], *, source_user_id: str, boundary: SourceRelativeDecisionBoundary
) -> dict[str, Any] | None:
    """Expose only safe event timing; redact final fields until the event is complete."""
    if entry.get("entry_type") != "event":
        raise ValueError("event sanitizer accepts only event entries")
    start = parse_esl_naive_datetime(entry.get("time"))
    if not boundary.is_strictly_after_source_timestamp(
        source_user_id=source_user_id, source_timestamp=entry.get("time")
    ):
        return None

    nested_event = entry.get("event")
    if not isinstance(nested_event, Mapping):
        nested_event = {}
    raw_end = entry.get("end_time", nested_event.get("end_time"))
    end = parse_esl_naive_datetime(raw_end) if raw_end is not None else None
    if end is not None and end < start:
        raise ValueError("event end_time cannot precede its start time")

    completed = end is not None and end < boundary.timestamp
    safe: dict[str, Any] = {
        "entry_type": "event",
        "start_time": start.isoformat(),
        "status_at_boundary": "completed" if completed else "ongoing_or_unknown",
    }
    if completed:
        safe["end_time"] = end.isoformat()
        for field in ("duration_days", "interrupted"):
            value = entry.get(field, nested_event.get(field))
            if value is not None:
                if field == "duration_days" and (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or value < 0
                ):
                    raise ValueError("completed event duration_days must be non-negative numeric")
                if field == "interrupted" and not isinstance(value, bool):
                    raise ValueError("completed event interrupted must be boolean")
                safe[field] = value
    return safe


__all__ = [
    "TEMPORAL_SEMANTICS",
    "TEMPORAL_SEMANTICS_ID",
    "TEMPORAL_SEMANTICS_SHA256",
    "SourceRelativeDecisionBoundary",
    "TemporalSemanticsManifest",
    "candidate_decision_boundaries",
    "date_only_exam_is_visible",
    "latest_decision_boundary",
    "parse_esl_date",
    "parse_esl_naive_datetime",
    "sanitize_visible_event_entry",
]
