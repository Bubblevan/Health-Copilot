"""Typed slot-key normalization with explicit object type and value qualifiers."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from tools.research.memory.typed_slot_normalizer_v1 import (
    canonical_dimension,
    canonical_unit,
)


def _label(value: str) -> str:
    return "_".join(re.findall(r"[a-z0-9]+", value.casefold()))


def _singularize(value: str) -> str:
    tokens = _label(value).split("_")
    if tokens and len(tokens[-1]) > 3 and tokens[-1].endswith("s"):
        tokens[-1] = tokens[-1][:-1]
    return "_".join(tokens)


def canonical_slot_key(
    scope_id: str,
    proposal: Mapping[str, Any],
) -> tuple[str, str, str, str, str, str, str]:
    unit = canonical_unit(str(proposal["unit"]))
    dimension = canonical_dimension(str(proposal["state_dimension"]), unit)
    object_type = _singularize(str(proposal["object_type"]))
    object_qualifier = _label(str(proposal["object_qualifier"]))
    if object_qualifier in {"", "none", "generic"}:
        object_qualifier = "generic"
    return (
        scope_id,
        _label(str(proposal["entity"])),
        dimension,
        object_type,
        object_qualifier,
        unit,
        _label(str(proposal["cardinality"])),
    )


def same_revision_slot(
    scope_a: str,
    proposal_a: Mapping[str, Any],
    scope_b: str,
    proposal_b: Mapping[str, Any],
) -> bool:
    for proposal in (proposal_a, proposal_b):
        if any(
            _label(str(proposal[field])) in {"", "unknown", "unspecified"}
            for field in ("entity", "state_dimension", "object_type", "object_qualifier", "unit")
        ):
            return False
    key_a = canonical_slot_key(scope_a, proposal_a)
    key_b = canonical_slot_key(scope_b, proposal_b)
    return key_a == key_b and key_a[-1] == "single_value_at_a_time"
