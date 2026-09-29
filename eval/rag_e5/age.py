"""Decision-time age derivation for E5 longitudinal state."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import date
from hashlib import sha256
from typing import Any

from eval.rag_e5.temporal import SourceRelativeDecisionBoundary, parse_esl_date

AGE_TEMPORAL_CONTRACT = {
    "contract_id": "AGE_TEMPORAL_CONTRACT_V1",
    "allowed_birth_date_fields": [
        "profile.demographics.date_of_birth",
        "profile.demographics.birth_date",
        "profile.demographics.birthday",
    ],
    "allowed_birth_date_format": "YYYY-MM-DD with a four-digit birth year",
    "decision_date_basis": "same_user_source_relative_decision_boundary.calendar_date",
    "age_calculation": "completed_years_at_decision_date",
    "excluded_dynamic_fields": ["profile.demographics.age"],
    "fallback": "unknown",
    "conflicting_or_invalid_birth_dates": "unknown",
    "timezone_interpretation": "NONE",
}
AGE_TEMPORAL_CONTRACT_SHA256 = sha256(
    json.dumps(AGE_TEMPORAL_CONTRACT, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()
_ALLOWED_DATE_KEYS = frozenset({"date_of_birth", "birth_date", "birthday"})


def age_bucket_at_decision(
    demographics: Mapping[str, Any], boundary: SourceRelativeDecisionBoundary
) -> str:
    """Use only a full, unambiguous DOB; never use snapshot age directly."""
    available = [(key, demographics[key]) for key in _ALLOWED_DATE_KEYS if key in demographics]
    if not available:
        return "unknown"
    parsed_dates: set[date] = set()
    for _key, value in available:
        try:
            parsed_dates.add(parse_esl_date(value))
        except ValueError:
            return "unknown"
    if len(parsed_dates) != 1:
        return "unknown"
    birth_date = next(iter(parsed_dates))
    decision_date = boundary.timestamp.date()
    if birth_date > decision_date:
        return "unknown"
    age_years = decision_date.year - birth_date.year
    if (decision_date.month, decision_date.day) < (birth_date.month, birth_date.day):
        age_years -= 1
    return age_bucket(age_years)


def age_bucket(age: object) -> str:
    """Map an already decision-time-safe completed age to its fixed bucket."""
    if type(age) is not int or not 0 <= age <= 125:
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


__all__ = [
    "AGE_TEMPORAL_CONTRACT",
    "AGE_TEMPORAL_CONTRACT_SHA256",
    "age_bucket",
    "age_bucket_at_decision",
]
