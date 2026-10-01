"""Deterministically project licensed event cues to provenance spans."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from tools.research.memory import mem3b0q_r4 as frozen_r4
from tools.research.memory import mem3b0q_r4_candidate_builder_v1 as candidate_builder


POLICY_VERSION = "mem3b0q-r4-factorized-event-slot-projection-v1"
EVENT_CUE_POLICY = {
    ("BICYCLE", "PURCHASE_EVENT"): re.compile(r"\bcompleted\s+the\s+purchase\b", re.IGNORECASE),
}
CLAUSE_DELIMITERS = frozenset(";!?\n")


class EventProjectionError(ValueError):
    """A deterministic event cue cannot be safely projected from the source."""


def _clause_bounds(source: str, position: int) -> tuple[int, int]:
    left = 0
    right = len(source)
    for index, character in enumerate(source):
        terminal_period = character == "." and (
            index + 1 == len(source) or source[index + 1].isspace()
        )
        if character in CLAUSE_DELIMITERS or terminal_period:
            if index < position:
                left = index + 1
            elif index >= position:
                right = index
                break
    return left, right


def project_event_value_spans(
    content: str,
    proposition: Mapping[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    """Replace only recognized event-slot values with exact source cue spans.

    The model proposes object and attribute candidates. For known event slots,
    frozen harness rules select the nearest preceding owner mention and source
    event cue, then leave the existing admission guards unchanged.
    """
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise EventProjectionError("proposal_json_invalid") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("atoms"), list):
        raise EventProjectionError("proposal_shape_invalid")

    source = proposition.get("proposition_text")
    source_id = proposition.get("source_id")
    if not isinstance(source, str) or not isinstance(source_id, str):
        raise EventProjectionError("proposition_invalid")
    candidates = candidate_builder.build_candidates(
        source, frozen_r4.ALIASES, source_id=source_id
    )["candidates"]
    by_id = {
        field: {row["candidate_id"]: row for row in candidates[field]}
        for field in ("owner", "object", "attribute")
    }
    audit: list[dict[str, Any]] = []
    for index, atom in enumerate(payload["atoms"]):
        if not isinstance(atom, dict):
            raise EventProjectionError(f"atom_{index}_invalid")
        slot = (atom.get("object_candidate_id", "").split(":")[1:2], atom.get("attribute_candidate_id", "").split(":")[1:2])
        object_id = slot[0][0] if slot[0] else None
        attribute_id = slot[1][0] if slot[1] else None
        cue = EVENT_CUE_POLICY.get((object_id, attribute_id))
        if cue is None:
            continue
        object_candidate = by_id["object"].get(atom.get("object_candidate_id"))
        attribute_candidate = by_id["attribute"].get(atom.get("attribute_candidate_id"))
        if object_candidate is None or attribute_candidate is None:
            raise EventProjectionError(f"atom_{index}_candidate_unknown")
        clause_left, clause_right = _clause_bounds(source, attribute_candidate["start"])
        if not clause_left <= object_candidate["start"] < clause_right:
            raise EventProjectionError(f"atom_{index}_object_cross_clause")
        owner_left, owner_right = _clause_bounds(source, object_candidate["start"])
        if (owner_left, owner_right) != (clause_left, clause_right):
            raise EventProjectionError(f"atom_{index}_object_attribute_cross_clause")
        preceding_owners = [
            row
            for row in candidates["owner"]
            if owner_left <= row["start"]
            and row["end"] <= object_candidate["start"]
        ]
        if not preceding_owners:
            raise EventProjectionError(f"atom_{index}_owner_not_found")
        nearest_owner = max(preceding_owners, key=lambda row: (row["end"], row["start"]))
        matches = [
            match
            for match in cue.finditer(source, clause_left, clause_right)
            if match.start() < clause_right and clause_left <= match.start()
        ]
        if len(matches) != 1:
            raise EventProjectionError(f"atom_{index}_event_cue_not_unique")
        match = matches[0]
        source_span = source[match.start() : match.end()]
        prior_owner = atom.get("owner_candidate_id")
        prior_value = atom.get("value_span")
        atom["owner_candidate_id"] = nearest_owner["candidate_id"]
        atom["value_span"] = source_span
        audit.append(
            {
                "atom_index": index,
                "object_id": object_id,
                "attribute_id": attribute_id,
                "prior_model_owner_candidate_id": prior_owner,
                "projected_owner_candidate_id": nearest_owner["candidate_id"],
                "prior_model_value_span": prior_value,
                "projected_value_span": source_span,
                "source_start": match.start(),
                "source_end": match.end(),
                "policy": POLICY_VERSION,
            }
        )
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")), audit
