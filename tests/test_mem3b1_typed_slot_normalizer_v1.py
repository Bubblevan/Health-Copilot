from tools.research.memory.typed_slot_normalizer_v1 import (
    canonical_slot_key,
    canonical_unit,
    same_revision_slot,
)


def _proposal(dimension, obj, unit, cardinality="SINGLE_VALUE_AT_A_TIME"):
    return {
        "entity": "SELF",
        "state_dimension": dimension,
        "object_qualifier": obj,
        "unit": unit,
        "cardinality": cardinality,
    }


def test_schedule_and_frequency_labels_normalize_to_same_weekly_activity_slot():
    old = _proposal("schedule", "gym visits", "sessions_per_week")
    new = _proposal("frequency", "gym routine", "times_per_week")

    assert canonical_slot_key("scope-a", old) == canonical_slot_key("scope-a", new)
    assert same_revision_slot("scope-a", old, "scope-a", new)


def test_different_active_search_targets_do_not_become_one_slot():
    jewelry = _proposal("search intent", "jewelry store", "searches", "MULTI_VALUE_OR_SET")
    shoes = _proposal("search intent", "running shoes", "searches", "MULTI_VALUE_OR_SET")

    assert not same_revision_slot("scope-a", jewelry, "scope-a", shoes)


def test_scope_is_part_of_revision_identity():
    first = _proposal("follower count", "Instagram account", "followers")
    second = _proposal("follower count", "Instagram account", "followers")

    assert not same_revision_slot("user-a", first, "user-b", second)


def test_weekly_unit_aliases_are_canonicalized_only_for_known_frequency_units():
    assert canonical_unit("times per week") == "sessions_per_week"
    assert canonical_unit("sessions_per_week") == "sessions_per_week"
    assert canonical_unit("searches") == "searches"


def test_frequency_alias_does_not_normalize_non_weekly_schedule():
    days = _proposal("schedule", "work", "weekdays")
    frequency = _proposal("frequency", "work", "per_day")

    assert canonical_slot_key("scope-a", days) != canonical_slot_key("scope-a", frequency)
