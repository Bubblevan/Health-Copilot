"""Conservative offline joint-binding guard for R4 candidate proposals.

This wrapper narrows candidate-builder v1's independent field cross-product.
It does not infer revisions, write memory, or authorize inference.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from tools.research.memory import mem3b0q_r4 as frozen_r4
from tools.research.memory import mem3b0q_r4_candidate_binding_guard_v1 as locality_guard
from tools.research.memory import mem3b0q_r4_candidate_builder_v1 as candidate_builder


JOINT_GUARD_VERSION = "mem3b0q-r4-joint-binding-guard-v2-proposal"

# Closed policy for the frozen synthetic R4 vocabulary. Unknown pairs fail closed.
CARDINALITY_POLICY = {
    ("EXERCISE_PLAN", "ACTIVITY"): "SINGLE_VALUE_AT_A_TIME",
    ("WORK_LAPTOP", "OPERATING_SYSTEM"): "SINGLE_VALUE_AT_A_TIME",
    ("PERSONAL_LAPTOP", "OPERATING_SYSTEM"): "SINGLE_VALUE_AT_A_TIME",
    ("WALLET", "COLOR"): "SINGLE_VALUE_AT_A_TIME",
    ("WALLET", "MATERIAL"): "SINGLE_VALUE_AT_A_TIME",
    ("FAVORITE_FRUIT_SET", "MEMBERSHIP"): "MULTI_VALUE_CONCURRENT",
    ("WEDDING_TRIP_PLAN", "DESTINATION"): "SINGLE_VALUE_AT_A_TIME",
    ("HIKING_TRIP_PLAN", "DESTINATION"): "SINGLE_VALUE_AT_A_TIME",
    ("BICYCLE", "PURCHASE_EVENT"): "EVENT_OR_NOT_STATE",
}
EVENT_RELATION_PATTERNS = {
    ("BICYCLE", "PURCHASE_EVENT"): re.compile(
        r"purchase\s+of\s+my\s+bicycle", re.IGNORECASE
    ),
}
EVENT_VALUE_PATTERNS = {
    ("BICYCLE", "PURCHASE_EVENT"): re.compile(
        r"completed\s+the\s+purchase", re.IGNORECASE
    ),
}


class JointBindingError(ValueError):
    """A selected field tuple is not supported by the frozen local rules."""


def _nearest_before(
    candidates: list[Mapping[str, Any]], position: int
) -> Mapping[str, Any] | None:
    preceding = [row for row in candidates if row["end"] <= position]
    if not preceding:
        return None
    return max(preceding, key=lambda row: (row["end"], row["start"]))


def _nearest_after(
    candidates: list[Mapping[str, Any]], position: int
) -> Mapping[str, Any] | None:
    following = [row for row in candidates if row["start"] >= position]
    if not following:
        return None
    return min(following, key=lambda row: (row["start"], row["end"]))


def _selected_candidate(
    field: str,
    atom: Mapping[str, Any],
    candidates: list[Mapping[str, Any]],
) -> Mapping[str, Any]:
    witness = atom["witnesses"][f"{field}_span"]
    canonical_id = atom[f"{field}_id"]
    matches = [
        row
        for row in candidates
        if row["canonical_id"] == canonical_id
        and row["start"] == witness["start"]
        and row["end"] == witness["end"]
        and row["source_span"] == witness["text"]
    ]
    if len(matches) != 1:
        raise JointBindingError(f"{field}_candidate_not_unique")
    return matches[0]


def validate_joint_bound_candidate_proposal(
    content: bytes | str,
    proposition: Mapping[str, Any],
    *,
    scope_id: str,
) -> dict[str, Any]:
    """Apply local owner/object/attribute binding after the unchanged v1 checks.

    Mutable-state attributes bind to the nearest preceding object candidate.
    The selected owner must be the nearest owner before that object, and no
    new owner may intervene before the attribute. Event attributes are allowed
    to precede their object only when a frozen relation pattern binds the pair.
    Repeated mentions of one canonical object type are
    unresolved because this fixture cannot prove physical-instance identity.
    """
    normalized = locality_guard.validate_bound_candidate_proposal(
        content, proposition, scope_id=scope_id
    )
    source = proposition["proposition_text"]
    manifest = candidate_builder.build_candidates(
        source, frozen_r4.ALIASES, source_id=proposition["source_id"]
    )
    candidates = manifest["candidates"]

    for atom_index, atom in enumerate(normalized["atoms"]):
        owner = _selected_candidate("owner", atom, candidates["owner"])
        obj = _selected_candidate("object", atom, candidates["object"])
        attribute = _selected_candidate(
            "attribute", atom, candidates["attribute"]
        )

        policy = CARDINALITY_POLICY.get((atom["object_id"], atom["attribute_id"]))
        if policy is None:
            raise JointBindingError(f"atom_{atom_index}:typed_slot_policy_unknown")
        if atom["cardinality"] != policy:
            raise JointBindingError(f"atom_{atom_index}:cardinality_policy_mismatch")

        if policy == "EVENT_OR_NOT_STATE":
            value_pattern = EVENT_VALUE_PATTERNS.get(
                (atom["object_id"], atom["attribute_id"])
            )
            if value_pattern is None:
                raise JointBindingError(f"atom_{atom_index}:event_value_policy_unknown")
            if value_pattern.fullmatch(atom["value_text"].strip()) is None:
                raise JointBindingError(f"atom_{atom_index}:event_value_unproven")

        same_type_mentions = [
            row
            for row in candidates["object"]
            if row["canonical_id"] == atom["object_id"]
        ]
        if len(same_type_mentions) != 1:
            raise JointBindingError(f"atom_{atom_index}:object_instance_ambiguous")

        nearest_owner = _nearest_before(candidates["owner"], obj["start"])
        if nearest_owner != owner:
            raise JointBindingError(f"atom_{atom_index}:owner_not_nearest_before_object")

        if policy == "EVENT_OR_NOT_STATE":
            nearest_object = _nearest_after(
                candidates["object"], attribute["end"]
            )
            if nearest_object != obj:
                raise JointBindingError(f"atom_{atom_index}:event_object_not_nearest")
            relation_pattern = EVENT_RELATION_PATTERNS.get(
                (atom["object_id"], atom["attribute_id"])
            )
            if relation_pattern is None:
                raise JointBindingError(f"atom_{atom_index}:event_relation_policy_unknown")
            relation_span = source[attribute["start"] : obj["end"]].strip()
            if relation_pattern.fullmatch(relation_span) is None:
                raise JointBindingError(f"atom_{atom_index}:event_object_relation_unproven")
        else:
            nearest_object = _nearest_before(
                candidates["object"], attribute["start"]
            )
            if nearest_object != obj:
                raise JointBindingError(f"atom_{atom_index}:object_not_nearest_attribute")

            owner_intervenes = any(
                row["start"] >= obj["end"]
                and row["start"] < attribute["start"]
                for row in candidates["owner"]
            )
            if owner_intervenes:
                raise JointBindingError(f"atom_{atom_index}:owner_boundary_crossed")

    return normalized
