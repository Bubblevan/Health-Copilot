from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.research.memory import mem3b0q_r4 as frozen_r4
from tools.research.memory.mem3b0q_r4_candidate_builder_v1 import build_candidates
from tools.research.memory.mem3b0q_r4_joint_binding_guard_v2 import (
    JointBindingError,
    validate_joint_bound_candidate_proposal,
)


ROOT = Path(__file__).resolve().parents[1]
PACK_PATH = ROOT / "docs/research/memory/mem3b0q_r4_control_pack_v1.json"


def _candidate(
    source_id: str,
    source_text: str,
    field: str,
    canonical_id: str,
    occurrence: int = 0,
) -> dict:
    rows = build_candidates(
        source_text, frozen_r4.ALIASES, source_id=source_id
    )["candidates"][field]
    matches = [row for row in rows if row["canonical_id"] == canonical_id]
    return matches[occurrence]


def _proposal(
    source_id: str,
    source_text: str,
    *,
    owner_id: str = "SELF",
    object_id: str,
    attribute_id: str,
    value: str,
    cardinality: str = "SINGLE_VALUE_AT_A_TIME",
    owner_occurrence: int = 0,
    object_occurrence: int = 0,
    attribute_occurrence: int = 0,
) -> str:
    atom = {
        "owner_candidate_id": _candidate(
            source_id, source_text, "owner", owner_id, owner_occurrence
        )["candidate_id"],
        "object_candidate_id": _candidate(
            source_id, source_text, "object", object_id, object_occurrence
        )["candidate_id"],
        "attribute_candidate_id": _candidate(
            source_id,
            source_text,
            "attribute",
            attribute_id,
            attribute_occurrence,
        )["candidate_id"],
        "value_span": value,
        "cardinality_proposal": cardinality,
    }
    return json.dumps(
        {"source_id": source_id, "atoms": [atom], "abstention_reason": "NONE"}
    )


def _oracle_proposal(proposition: dict) -> str:
    source_id = proposition["source_id"]
    source_text = proposition["proposition_text"]
    candidates = build_candidates(
        source_text, frozen_r4.ALIASES, source_id=source_id
    )["candidates"]
    atoms = []
    for expected in proposition["expected"]["atoms"]:
        atom = {}
        for field, id_key, span_key in (
            ("owner", "owner_id", "owner_span"),
            ("object", "object_id", "object_span"),
            ("attribute", "attribute_id", "attribute_span"),
        ):
            match = next(
                row
                for row in candidates[field]
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


def test_all_frozen_oracle_atom_bindings_pass_offline_guard() -> None:
    pack = json.loads(PACK_PATH.read_text(encoding="utf-8"))
    for proposition in pack["propositions"]:
        result = validate_joint_bound_candidate_proposal(
            _oracle_proposal(proposition),
            proposition,
            scope_id=frozen_r4.FROZEN_SCOPE_ID,
        )
        assert len(result["atoms"]) == len(proposition["expected"]["atoms"])


def test_p01_candidate_ids_prevent_free_form_span_boundary_drift() -> None:
    source_id = "R4-P01"
    source = "My workout plan's activity is running."
    candidates = build_candidates(source, frozen_r4.ALIASES, source_id=source_id)

    assert [row["source_span"] for row in candidates["candidates"]["owner"]] == ["My"]
    assert [row["source_span"] for row in candidates["candidates"]["object"]] == [
        "workout plan"
    ]
    assert [row["source_span"] for row in candidates["candidates"]["attribute"]] == [
        "activity"
    ]


def test_cross_clause_laptop_mix_is_rejected_and_local_atoms_pass() -> None:
    source_id = "R4-X1"
    source = (
        "My work laptop's operating system is Linux, and my personal laptop's "
        "operating system is Windows."
    )
    proposition = {"source_id": source_id, "proposition_text": source}

    for owner_occurrence, object_id, object_occurrence, attr_occurrence, value in (
        (0, "WORK_LAPTOP", 0, 0, "Linux"),
        (1, "PERSONAL_LAPTOP", 0, 1, "Windows"),
    ):
        result = validate_joint_bound_candidate_proposal(
            _proposal(
                source_id,
                source,
                object_id=object_id,
                attribute_id="OPERATING_SYSTEM",
                value=value,
                owner_occurrence=owner_occurrence,
                object_occurrence=object_occurrence,
                attribute_occurrence=attr_occurrence,
            ),
            proposition,
            scope_id=frozen_r4.FROZEN_SCOPE_ID,
        )
        assert len(result["atoms"]) == 1

    cross_bound = _proposal(
        source_id,
        source,
        object_id="WORK_LAPTOP",
        attribute_id="OPERATING_SYSTEM",
        value="Windows",
        attribute_occurrence=1,
    )
    with pytest.raises(JointBindingError, match="object_not_nearest_attribute"):
        validate_joint_bound_candidate_proposal(
            cross_bound, proposition, scope_id=frozen_r4.FROZEN_SCOPE_ID
        )


def test_repeated_same_type_object_mentions_fail_closed() -> None:
    source_id = "R4-X2"
    source = "My wallet's color is black, and my wallet's color is blue."
    proposition = {"source_id": source_id, "proposition_text": source}
    proposal = _proposal(
        source_id,
        source,
        object_id="WALLET",
        attribute_id="COLOR",
        value="black",
    )

    with pytest.raises(JointBindingError, match="object_instance_ambiguous"):
        validate_joint_bound_candidate_proposal(
            proposal, proposition, scope_id=frozen_r4.FROZEN_SCOPE_ID
        )


def test_cardinality_is_validated_by_frozen_slot_policy() -> None:
    source_id = "R4-P09"
    source = "My favorite-fruit set has apples and pears as members."
    proposition = {"source_id": source_id, "proposition_text": source}
    wrong = _proposal(
        source_id,
        source,
        object_id="FAVORITE_FRUIT_SET",
        attribute_id="MEMBERSHIP",
        value="apples and pears",
        cardinality="SINGLE_VALUE_AT_A_TIME",
    )

    with pytest.raises(JointBindingError, match="cardinality_policy_mismatch"):
        validate_joint_bound_candidate_proposal(
            wrong, proposition, scope_id=frozen_r4.FROZEN_SCOPE_ID
        )


@pytest.mark.parametrize(
    "source",
    [
        "I completed the purchase. My bicycle is blue.",
        "I completed the purchase and my bicycle is red.",
    ],
)
def test_event_is_not_bound_to_unrelated_later_object(source: str) -> None:
    source_id = "R4-X4"
    proposition = {"source_id": source_id, "proposition_text": source}
    proposal = _proposal(
        source_id,
        source,
        object_id="BICYCLE",
        attribute_id="PURCHASE_EVENT",
        value="completed the purchase",
        cardinality="EVENT_OR_NOT_STATE",
    )

    with pytest.raises(JointBindingError, match="event_object_relation_unproven"):
        validate_joint_bound_candidate_proposal(
            proposal, proposition, scope_id=frozen_r4.FROZEN_SCOPE_ID
        )


def test_event_value_cannot_borrow_an_unrelated_same_clause_attribute() -> None:
    source_id = "R4-X5"
    source = "I completed the purchase of my bicycle, which is blue."
    proposition = {"source_id": source_id, "proposition_text": source}
    proposal = _proposal(
        source_id,
        source,
        object_id="BICYCLE",
        attribute_id="PURCHASE_EVENT",
        value="blue",
        cardinality="EVENT_OR_NOT_STATE",
    )

    with pytest.raises(JointBindingError, match="event_value_unproven"):
        validate_joint_bound_candidate_proposal(
            proposal, proposition, scope_id=frozen_r4.FROZEN_SCOPE_ID
        )


def test_unknown_slot_pair_is_not_admitted_by_default() -> None:
    source_id = "R4-X3"
    source = "My work laptop's color is green."
    proposition = {"source_id": source_id, "proposition_text": source}
    proposal = _proposal(
        source_id,
        source,
        object_id="WORK_LAPTOP",
        attribute_id="COLOR",
        value="green",
    )

    with pytest.raises(JointBindingError, match="typed_slot_policy_unknown"):
        validate_joint_bound_candidate_proposal(
            proposal, proposition, scope_id=frozen_r4.FROZEN_SCOPE_ID
        )
