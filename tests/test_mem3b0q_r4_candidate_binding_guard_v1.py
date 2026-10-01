from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.research.memory import mem3b0q_r4 as frozen_r4
from tools.research.memory.mem3b0q_r4_candidate_binding_guard_v1 import (
    CandidateBindingError,
    validate_bound_candidate_proposal,
)
from tools.research.memory.mem3b0q_r4_candidate_builder_v1 import build_candidates


ROOT = Path(__file__).resolve().parents[1]
PACK_PATH = ROOT / "docs/research/memory/mem3b0q_r4_control_pack_v1.json"


def _proposal(source_id: str, source_text: str, attribute: str, value: str) -> str:
    candidates = build_candidates(source_text, frozen_r4.ALIASES, source_id=source_id)
    attribute_candidate = next(
        row
        for row in candidates["candidates"]["attribute"]
        if row["canonical_id"] == attribute
    )
    atom = {
        f"{field}_candidate_id": candidates["candidates"][field][0]["candidate_id"]
        for field in ("owner", "object")
    }
    atom.update(
        {
            "attribute_candidate_id": attribute_candidate["candidate_id"],
            "value_span": value,
            "cardinality_proposal": "SINGLE_VALUE_AT_A_TIME",
        }
    )
    return json.dumps(
        {"source_id": source_id, "atoms": [atom], "abstention_reason": "NONE"}
    )


def _oracle_proposal(proposition: dict) -> str:
    source_id = proposition["source_id"]
    source_text = proposition["proposition_text"]
    candidates = build_candidates(source_text, frozen_r4.ALIASES, source_id=source_id)
    expected_atoms = proposition["expected"]["atoms"]
    if isinstance(expected_atoms, dict):
        expected_atoms = [expected_atoms]

    atoms = []
    for expected in expected_atoms:
        atom = {}
        for field, id_key, span_key in (
            ("owner", "owner_id", "owner_span"),
            ("object", "object_id", "object_span"),
            ("attribute", "attribute_id", "attribute_span"),
        ):
            match = next(
                row
                for row in candidates["candidates"][field]
                if row["canonical_id"] == expected[id_key]
                and row["source_span"].casefold() == expected[span_key].casefold()
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


def test_p15_rejects_color_bound_to_material_value() -> None:
    pack = json.loads(PACK_PATH.read_text(encoding="utf-8"))
    proposition = next(
        row for row in pack["propositions"] if row["source_id"] == "R4-P15"
    )
    payload = _proposal(
        proposition["source_id"],
        proposition["proposition_text"],
        "COLOR",
        "leather",
    )

    with pytest.raises(CandidateBindingError, match="attribute_value_cross_clause"):
        validate_bound_candidate_proposal(
            payload, proposition, scope_id=frozen_r4.FROZEN_SCOPE_ID
        )


@pytest.mark.parametrize(
    ("source_id", "attribute", "value"),
    [("R4-P15", "COLOR", "black"), ("R4-P15", "MATERIAL", "leather")],
)
def test_p15_accepts_clause_local_attribute_value_pairs(
    source_id: str, attribute: str, value: str
) -> None:
    pack = json.loads(PACK_PATH.read_text(encoding="utf-8"))
    proposition = next(
        row for row in pack["propositions"] if row["source_id"] == source_id
    )
    result = validate_bound_candidate_proposal(
        _proposal(source_id, proposition["proposition_text"], attribute, value),
        proposition,
        scope_id=frozen_r4.FROZEN_SCOPE_ID,
    )

    assert result["atoms"][0]["attribute_id"] == attribute
    assert result["atoms"][0]["value_text"] == value


def test_value_list_conjunction_is_not_misread_as_attribute_boundary() -> None:
    pack = json.loads(PACK_PATH.read_text(encoding="utf-8"))
    proposition = next(
        row for row in pack["propositions"] if row["source_id"] == "R4-P09"
    )
    candidates = build_candidates(
        proposition["proposition_text"],
        frozen_r4.ALIASES,
        source_id=proposition["source_id"],
    )
    atom = {
        f"{field}_candidate_id": candidates["candidates"][field][0]["candidate_id"]
        for field in ("owner", "object", "attribute")
    }
    atom.update(
        {
            "value_span": "apples and pears",
            "cardinality_proposal": "MULTI_VALUE_CONCURRENT",
        }
    )
    payload = json.dumps(
        {
            "source_id": proposition["source_id"],
            "atoms": [atom],
            "abstention_reason": "NONE",
        }
    )

    result = validate_bound_candidate_proposal(
        payload, proposition, scope_id=frozen_r4.FROZEN_SCOPE_ID
    )

    assert result["atoms"][0]["value_text"] == "apples and pears"


def test_decimal_point_inside_value_is_not_a_clause_boundary() -> None:
    proposition = {
        "source_id": "R4-D1",
        "proposition_text": "My wallet's color is 3.14.",
    }
    result = validate_bound_candidate_proposal(
        _proposal("R4-D1", proposition["proposition_text"], "COLOR", "3.14"),
        proposition,
        scope_id=frozen_r4.FROZEN_SCOPE_ID,
    )

    assert result["atoms"][0]["value_text"] == "3.14"


def test_frozen_control_oracle_representations_pass_the_locality_guard() -> None:
    pack = json.loads(PACK_PATH.read_text(encoding="utf-8"))
    for proposition in pack["propositions"]:
        result = validate_bound_candidate_proposal(
            _oracle_proposal(proposition),
            proposition,
            scope_id=frozen_r4.FROZEN_SCOPE_ID,
        )
        expected_atoms = proposition["expected"]["atoms"]
        expected_count = len(expected_atoms) if isinstance(expected_atoms, list) else 1
        assert len(result["atoms"]) == expected_count, proposition["source_id"]
