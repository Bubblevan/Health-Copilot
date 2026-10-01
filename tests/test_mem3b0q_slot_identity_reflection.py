from __future__ import annotations

from tools.research.memory.mem3b0q_slot_identity_reflection import (
    _components,
    literal_key_anchor,
)


def test_literal_attribute_anchor_accepts_directly_supported_key() -> None:
    assert literal_key_anchor(
        "instagram_followers", "The user has reached 600 followers on Instagram."
    ) == (True, ["instagram", "followers"])


def test_literal_attribute_anchor_rejects_key_not_supported_by_proposition() -> None:
    grounded, _ = literal_key_anchor(
        "workout_preference", "The user is planning to build a gaming PC."
    )
    assert not grounded


def test_literal_attribute_anchor_is_not_paraphrase_aware() -> None:
    grounded, _ = literal_key_anchor(
        "software_version", "The user has node version v19.7.0."
    )
    assert not grounded


def test_components_are_deterministic_and_keep_disconnected_groups_separate() -> None:
    assert _components([("b", "a"), ("a", "c"), ("x", "y")]) == [
        ["a", "b", "c"],
        ["x", "y"],
    ]
