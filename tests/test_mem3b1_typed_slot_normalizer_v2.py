from tools.research.memory.typed_slot_normalizer_v2 import (
    canonical_slot_key,
    same_revision_slot,
)


def _proposal(obj_type, qualifier, dimension="interest", unit="none", cardinality="SINGLE_VALUE_AT_A_TIME"):
    return {
        "entity": "SELF",
        "state_dimension": dimension,
        "object_type": obj_type,
        "object_qualifier": qualifier,
        "value": "some value",
        "value_qualifiers": ["descriptive qualifier"],
        "unit": unit,
        "cardinality": cardinality,
    }


def test_value_qualifiers_do_not_change_the_slot_identity():
    left = _proposal("attractions", "generic")
    right = _proposal("attraction", "generic")
    right["value_qualifiers"] = ["suitable for younger children"]

    assert canonical_slot_key("scope-a", left) == canonical_slot_key("scope-a", right)
    assert same_revision_slot("scope-a", left, "scope-a", right)


def test_distinct_concrete_object_types_remain_separate():
    jewelry = _proposal("jewelry store", "generic", cardinality="MULTI_VALUE_OR_SET")
    shoes = _proposal("running shoes", "generic", cardinality="MULTI_VALUE_OR_SET")

    assert not same_revision_slot("scope-a", jewelry, "scope-a", shoes)


def test_weekly_frequency_units_remain_compatible_after_axis_normalization():
    old = _proposal("fitness activity", "gym", "schedule", "sessions_per_week")
    new = _proposal("fitness activity", "gym", "frequency", "times_per_week")

    assert same_revision_slot("scope-a", old, "scope-a", new)


def test_unknown_slot_component_fails_closed():
    known = _proposal("attraction", "generic")
    unknown = _proposal("unknown", "generic")

    assert not same_revision_slot("scope-a", known, "scope-a", unknown)


def test_scope_is_part_of_slot_identity():
    left = _proposal("attraction", "generic")
    right = _proposal("attraction", "generic")

    assert not same_revision_slot("user-a", left, "user-b", right)
