"""Small deterministic canonicalizer for typed revision-slot proposals."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

WEEKLY_EVENT_UNIT_ALIASES = frozenset(
    {
        "session_per_week",
        "sessions_per_week",
        "time_per_week",
        "times_per_week",
        "visit_per_week",
        "visits_per_week",
        "workout_per_week",
        "workouts_per_week",
    }
)
WEEKLY_FREQUENCY_DIMENSIONS = frozenset(
    {
        "frequency",
        "schedule",
        "weekly_frequency",
        "recurring_frequency",
        "exercise_frequency",
        "activity_frequency",
    }
)
FITNESS_OBJECT_TOKENS = frozenset(
    {"gym", "workout", "workouts", "exercise", "exercises", "fitness"}
)


def _label(value: str) -> str:
    return "_".join(re.findall(r"[a-z0-9]+", value.casefold()))


def canonical_unit(value: str) -> str:
    unit = _label(value)
    return "sessions_per_week" if unit in WEEKLY_EVENT_UNIT_ALIASES else unit


def canonical_dimension(value: str, unit: str) -> str:
    dimension = _label(value)
    normalized_unit = canonical_unit(unit)
    if (
        normalized_unit == "sessions_per_week"
        and dimension in WEEKLY_FREQUENCY_DIMENSIONS
    ):
        return "recurring_frequency"
    return dimension


def canonical_object(value: str, *, dimension: str, unit: str) -> str:
    object_label = _label(value)
    tokens = set(object_label.split("_"))
    if (
        canonical_dimension(dimension, unit) == "recurring_frequency"
        and tokens.intersection(FITNESS_OBJECT_TOKENS)
    ):
        return "fitness_activity"
    return object_label


def canonical_slot_key(
    scope_id: str,
    proposal: Mapping[str, Any],
) -> tuple[str, str, str, str, str, str]:
    entity = _label(str(proposal["entity"]))
    dimension = canonical_dimension(
        str(proposal["state_dimension"]), str(proposal["unit"])
    )
    object_qualifier = canonical_object(
        str(proposal["object_qualifier"]),
        dimension=str(proposal["state_dimension"]),
        unit=str(proposal["unit"]),
    )
    unit = canonical_unit(str(proposal["unit"]))
    cardinality = _label(str(proposal["cardinality"]))
    return (
        scope_id,
        entity,
        dimension,
        object_qualifier,
        unit,
        cardinality,
    )


def same_revision_slot(
    scope_a: str,
    proposal_a: Mapping[str, Any],
    scope_b: str,
    proposal_b: Mapping[str, Any],
) -> bool:
    key_a = canonical_slot_key(scope_a, proposal_a)
    key_b = canonical_slot_key(scope_b, proposal_b)
    return key_a == key_b and key_a[-1] == "single_value_at_a_time"
