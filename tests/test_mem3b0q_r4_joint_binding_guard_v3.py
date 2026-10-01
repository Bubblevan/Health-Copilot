from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.research.memory import mem3b0q_r4 as frozen_r4
from tools.research.memory.mem3b0q_r4_candidate_builder_v1 import build_candidates
from tools.research.memory.mem3b0q_r4_joint_binding_guard_v3 import (
    JointBindingError,
    validate_joint_bound_candidate_proposal,
)

ROOT = Path(__file__).resolve().parents[1]
PACK_PATH = ROOT / "docs/research/memory/mem3b0q_r4_control_pack_v1.json"


def _proposal(source_id: str, source: str, *, attribute_occurrence: int, value: str) -> str:
    candidates = build_candidates(source, frozen_r4.ALIASES, source_id=source_id)[
        "candidates"
    ]

    def candidate(field: str, canonical_id: str, occurrence: int = 0) -> str:
        rows = [
            row
            for row in candidates[field]
            if row["canonical_id"] == canonical_id
        ]
        return rows[occurrence]["candidate_id"]

    atom = {
        "owner_candidate_id": candidate("owner", "SELF"),
        "object_candidate_id": candidate("object", "WORK_LAPTOP"),
        "attribute_candidate_id": candidate(
            "attribute", "OPERATING_SYSTEM", attribute_occurrence
        ),
        "value_span": value,
        "cardinality_proposal": "SINGLE_VALUE_AT_A_TIME",
    }
    return json.dumps(
        {"source_id": source_id, "atoms": [atom], "abstention_reason": "NONE"}
    )


@pytest.mark.parametrize(
    ("source", "error"),
    [
        (
            "My work laptop's operating system is Linux. A tablet's operating system is Windows.",
            "object_attribute_cross_clause",
        ),
        (
            "My work laptop has operating system Linux and a tablet has operating system Windows.",
            "unknown_entity_between_object_attribute",
        ),
    ],
)
def test_unknown_object_distractor_cannot_reuse_known_object(
    source: str, error: str
) -> None:
    source_id = "R4-X9"
    proposal = _proposal(
        source_id, source, attribute_occurrence=1, value="Windows"
    )

    with pytest.raises(JointBindingError, match=error):
        validate_joint_bound_candidate_proposal(
            proposal,
            {"source_id": source_id, "proposition_text": source},
            scope_id=frozen_r4.FROZEN_SCOPE_ID,
        )


def test_known_object_attribute_and_value_in_same_clause_pass() -> None:
    source_id = "R4-X10"
    source = "My work laptop's operating system is Linux."
    result = validate_joint_bound_candidate_proposal(
        _proposal(source_id, source, attribute_occurrence=0, value="Linux"),
        {"source_id": source_id, "proposition_text": source},
        scope_id=frozen_r4.FROZEN_SCOPE_ID,
    )

    assert len(result["atoms"]) == 1


def test_determiner_before_registered_attribute_is_not_unknown_entity() -> None:
    source_id = "R4-X11"
    source = "My work laptop has the operating system Linux."
    result = validate_joint_bound_candidate_proposal(
        _proposal(source_id, source, attribute_occurrence=0, value="Linux"),
        {"source_id": source_id, "proposition_text": source},
        scope_id=frozen_r4.FROZEN_SCOPE_ID,
    )

    assert len(result["atoms"]) == 1


def _oracle_proposal(proposition: dict) -> str:
    source_id = proposition["source_id"]
    source = proposition["proposition_text"]
    candidates = build_candidates(source, frozen_r4.ALIASES, source_id=source_id)[
        "candidates"
    ]
    atoms = []
    for expected in proposition["expected"]["atoms"]:
        atom = {}
        for field in ("owner", "object", "attribute"):
            canonical_id = expected[f"{field}_id"]
            span = expected[f"{field}_span"]
            match = next(
                row
                for row in candidates[field]
                if row["canonical_id"] == canonical_id
                and row["source_span"].casefold() == span.casefold()
            )
            atom[f"{field}_candidate_id"] = match["candidate_id"]
        atom["value_span"] = expected["value_span"]
        atom["cardinality_proposal"] = expected["cardinality"]
        atoms.append(atom)
    return json.dumps(
        {
            "source_id": source_id,
            "atoms": atoms,
            "abstention_reason": proposition["expected"].get("abstention") or "NONE",
        }
    )


def test_all_frozen_oracle_atoms_pass_v3_guard() -> None:
    pack = json.loads(PACK_PATH.read_text(encoding="utf-8"))
    for proposition in pack["propositions"]:
        result = validate_joint_bound_candidate_proposal(
            _oracle_proposal(proposition),
            proposition,
            scope_id=frozen_r4.FROZEN_SCOPE_ID,
        )
        assert len(result["atoms"]) == len(proposition["expected"]["atoms"])
