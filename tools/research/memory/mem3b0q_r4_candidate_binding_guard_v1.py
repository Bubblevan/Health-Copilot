"""Conservative clause-local guard for R4 candidate attribute/value bindings.

This is a non-mutating proposal validator. It is deliberately narrower than a
semantic parser and must not be treated as proof of a valid memory revision.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from collections.abc import Mapping
from typing import Any

from tools.research.memory import mem3b0q_r4 as frozen_r4
from tools.research.memory import mem3b0q_r4_candidate_builder_v1 as candidate_builder


class CandidateBindingError(ValueError):
    """The selected attribute and value are separated by a hard clause boundary."""


def _clause_boundaries(
    source: str, attribute_candidates: list[Mapping[str, Any]]
) -> list[int]:
    boundaries = set()
    for index, character in enumerate(source):
        if character in ";!?\n":
            boundaries.add(index + 1)
            continue
        if character != ".":
            continue
        previous_is_digit = index > 0 and source[index - 1].isdigit()
        next_is_digit = index + 1 < len(source) and source[index + 1].isdigit()
        period_is_terminal = index + 1 == len(source) or source[index + 1].isspace()
        if period_is_terminal and not (previous_is_digit and next_is_digit):
            boundaries.add(index + 1)
    for candidate in attribute_candidates:
        start = candidate["start"]
        if not isinstance(start, int):
            raise CandidateBindingError("attribute_candidate_offset_invalid")
        if re.search(r"\band\s+$", source[:start], flags=re.IGNORECASE):
            boundaries.add(start)
    return sorted(boundaries)


def _clause_index(boundaries: list[int], start: int, end: int) -> int:
    if start < 0 or end <= start:
        raise CandidateBindingError("witness_span_invalid")
    first = bisect_right(boundaries, start)
    if first < len(boundaries) and boundaries[first] < end:
        raise CandidateBindingError("witness_crosses_clause_boundary")
    return first


def validate_bound_candidate_proposal(
    content: bytes | str,
    proposition: Mapping[str, Any],
    *,
    scope_id: str,
) -> dict[str, Any]:
    """Validate candidate IDs, then reject attribute/value cross-clause pairs.

    Clause boundaries are ``;!?``/line breaks, a terminal-looking period that
    is not between digits, and ``and`` immediately before another registered
    attribute alias. This catches the frozen P15 color/material cross-binding
    without splitting decimal/version spans or value lists such as
    ``apples and pears``. Abbreviations can still look terminal; this is not a
    sentence parser and does not prove semantic binding within a clause.
    """
    normalized = candidate_builder.validate_candidate_proposal(
        content, proposition, scope_id=scope_id
    )
    source = proposition["proposition_text"]
    candidate_manifest = candidate_builder.build_candidates(
        source, frozen_r4.ALIASES, source_id=proposition["source_id"]
    )
    boundaries = _clause_boundaries(
        source, candidate_manifest["candidates"]["attribute"]
    )

    for index, atom in enumerate(normalized["atoms"]):
        witnesses = atom["witnesses"]
        attribute = witnesses["attribute_span"]
        value = witnesses["value_span"]
        attribute_clause = _clause_index(
            boundaries, attribute["start"], attribute["end"]
        )
        value_clause = _clause_index(boundaries, value["start"], value["end"])
        if attribute_clause != value_clause:
            raise CandidateBindingError(f"atom_{index}:attribute_value_cross_clause")
    return normalized
