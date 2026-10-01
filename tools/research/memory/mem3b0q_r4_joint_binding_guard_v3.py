"""Offline clause-local extension for the R4 joint-binding proposal."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from tools.research.memory import mem3b0q_r4 as frozen_r4
from tools.research.memory import mem3b0q_r4_candidate_builder_v1 as candidate_builder
from tools.research.memory import mem3b0q_r4_joint_binding_guard_v2 as joint_guard_v2


JOINT_GUARD_VERSION = "mem3b0q-r4-joint-binding-guard-v3-proposal"


class JointBindingError(ValueError):
    """A selected field tuple is not supported by the frozen local rules."""


def _clause_id(
    boundaries: list[int], span: Mapping[str, Any]
) -> int:
    if span["start"] < 0 or span["end"] <= span["start"]:
        raise JointBindingError("anchor_span_invalid")
    return sum(boundary <= span["start"] for boundary in boundaries)


def _anchor_clause_boundaries(source: str) -> list[int]:
    boundaries = []
    for index, character in enumerate(source):
        if character in ";!?\n":
            boundaries.append(index + 1)
        elif character == ".":
            previous_is_digit = index > 0 and source[index - 1].isdigit()
            next_is_digit = index + 1 < len(source) and source[index + 1].isdigit()
            terminal = index + 1 == len(source) or source[index + 1].isspace()
            if terminal and not (previous_is_digit and next_is_digit):
                boundaries.append(index + 1)
    return boundaries


def _reject_unknown_determiner_between(
    atom_index: int,
    source: str,
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    candidates: Mapping[str, list[Mapping[str, Any]]],
    relation: str,
) -> None:
    gap_start, gap_end = left["end"], right["start"]
    if gap_end <= gap_start:
        return
    gap = source[gap_start:gap_end]
    for match in re.finditer(
        r"\b(?:a|an|the|another|this|that|these|those|some|each)\s+([a-z][\w-]*)\b",
        gap,
        re.IGNORECASE,
    ):
        start = gap_start + match.start()
        end = gap_start + match.end()
        overlaps_registered_anchor = any(
            start < candidate["end"] and candidate["start"] < end
            for field in ("owner", "object", "attribute")
            for candidate in candidates[field]
        )
        if not overlaps_registered_anchor:
            raise JointBindingError(
                f"atom_{atom_index}:unknown_entity_between_{relation}"
            )


def validate_joint_bound_candidate_proposal(
    content: bytes | str,
    proposition: Mapping[str, Any],
    *,
    scope_id: str,
) -> dict[str, Any]:
    """Add owner/object/attribute clause locality over the reviewed v2 guard.

    Anchor clause boundaries are punctuation-only. A separate conservative
    determiner+noun check rejects an unregistered noun phrase between anchors.
    This applies only to mutable-state slots; event relations retain their
    separately frozen, fixture-scoped rule.
    """
    normalized = joint_guard_v2.validate_joint_bound_candidate_proposal(
        content, proposition, scope_id=scope_id
    )
    source = proposition["proposition_text"]
    candidates = candidate_builder.build_candidates(
        source, frozen_r4.ALIASES, source_id=proposition["source_id"]
    )["candidates"]
    boundaries = _anchor_clause_boundaries(source)

    for atom_index, atom in enumerate(normalized["atoms"]):
        policy = joint_guard_v2.CARDINALITY_POLICY.get(
            (atom["object_id"], atom["attribute_id"])
        )
        if policy == "EVENT_OR_NOT_STATE":
            continue

        witnesses = atom["witnesses"]
        owner_clause = _clause_id(boundaries, witnesses["owner_span"])
        object_clause = _clause_id(boundaries, witnesses["object_span"])
        attribute_clause = _clause_id(boundaries, witnesses["attribute_span"])
        if owner_clause != object_clause:
            raise JointBindingError(f"atom_{atom_index}:owner_object_cross_clause")
        if object_clause != attribute_clause:
            raise JointBindingError(f"atom_{atom_index}:object_attribute_cross_clause")
        _reject_unknown_determiner_between(
            atom_index,
            source,
            witnesses["owner_span"],
            witnesses["object_span"],
            candidates,
            "owner_object",
        )
        _reject_unknown_determiner_between(
            atom_index,
            source,
            witnesses["object_span"],
            witnesses["attribute_span"],
            candidates,
            "object_attribute",
        )

    return normalized
