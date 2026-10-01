from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.research.memory import mem3b0q_r4 as frozen_r4
from tools.research.memory.mem3b0q_r4_candidate_builder_v1 import (
    CandidateBuildError,
    build_candidates,
    build_candidate_schema,
    validate_candidate_proposal,
)


ROOT = Path(__file__).resolve().parents[1]
PACK_PATH = ROOT / "docs/research/memory/mem3b0q_r4_control_pack_v1.json"


def _candidate_ids(result: dict, field: str) -> list[str]:
    return [row["candidate_id"] for row in result["candidates"][field]]


def test_p01_typed_candidates_separate_possessor_object_and_attribute() -> None:
    result = build_candidates(
        "My workout plan's activity is running.", frozen_r4.ALIASES
    )

    assert [
        (row["source_span"], row["canonical_id"])
        for row in result["candidates"]["owner"]
    ] == [("My", "SELF")]
    assert [
        (row["source_span"], row["canonical_id"])
        for row in result["candidates"]["object"]
    ] == [("workout plan", "EXERCISE_PLAN")]
    assert [
        (row["source_span"], row["canonical_id"])
        for row in result["candidates"]["attribute"]
    ] == [("activity", "ACTIVITY")]


def test_longest_owner_alias_suppresses_nested_my_candidate() -> None:
    result = build_candidates(
        "My sister's exercise plan lists running as its activity.",
        frozen_r4.ALIASES,
    )

    assert [row["source_span"] for row in result["candidates"]["owner"]] == [
        "My sister"
    ]
    assert result["candidates"]["owner"][0]["canonical_id"] == "SISTER"


def test_longest_attribute_alias_suppresses_member_subspan() -> None:
    result = build_candidates(
        "My favorite-fruit set has apples and pears as members.",
        frozen_r4.ALIASES,
    )

    assert [row["source_span"] for row in result["candidates"]["attribute"]] == [
        "members"
    ]


def test_repeated_aliases_remain_distinct_source_candidates() -> None:
    result = build_candidates("My wallet and my wallet", frozen_r4.ALIASES)

    assert [row["source_span"] for row in result["candidates"]["object"]] == [
        "wallet",
        "wallet",
    ]
    assert len(set(_candidate_ids(result, "object"))) == 2


def test_codepoint_offsets_reconstruct_casefolded_source_span() -> None:
    registry = {"owner": {"ss": "SELF"}, "object": {}, "attribute": {}}
    result = build_candidates("ß", registry)
    owner = result["candidates"]["owner"][0]

    assert (owner["source_span"], owner["start"], owner["end"]) == ("ß", 0, 1)


def test_alias_substrings_inside_larger_words_are_not_candidates() -> None:
    result = build_candidates("dummy running", frozen_r4.ALIASES)

    assert result["candidates"]["owner"] == []


def test_candidate_manifest_is_deterministic() -> None:
    first = build_candidates("MY wallet's color is black.", frozen_r4.ALIASES)
    second = build_candidates("MY wallet's color is black.", frozen_r4.ALIASES)

    assert first == second
    assert first["candidate_manifest_sha256"] == second["candidate_manifest_sha256"]
    assert first["candidates"]["owner"][0]["source_span"] == "MY"


def test_candidate_manifest_binds_alias_registry_even_when_output_is_unchanged() -> None:
    base = {"owner": {"my": "SELF"}, "object": {}, "attribute": {}}
    extended = {
        "owner": {"my": "SELF", "myself": "SELF"},
        "object": {},
        "attribute": {},
    }

    first = build_candidates("my wallet", base)
    second = build_candidates("my wallet", extended)

    assert first["candidates"] == second["candidates"]
    assert first["alias_registry_sha256"] != second["alias_registry_sha256"]
    assert first["candidate_manifest_sha256"] != second["candidate_manifest_sha256"]


def test_alias_registry_hash_preserves_exact_alias_spelling() -> None:
    title_case = {"owner": {"My": "SELF"}, "object": {}, "attribute": {}}
    lower_case = {"owner": {"my": "SELF"}, "object": {}, "attribute": {}}

    first = build_candidates("MY plan", title_case)
    second = build_candidates("MY plan", lower_case)

    assert first["candidates"] == second["candidates"]
    assert first["alias_registry_sha256"] != second["alias_registry_sha256"]


def test_exact_alias_collision_and_partial_overlap_fail_closed() -> None:
    with pytest.raises(CandidateBuildError, match="alias_casefold_collision"):
        build_candidates(
            "foo",
            {"owner": {"Foo": "A", "foo": "B"}, "object": {}, "attribute": {}},
        )

    with pytest.raises(CandidateBuildError, match="partial_alias_overlap"):
        build_candidates(
            "foo bar",
            {"owner": {"foo ": "A", " bar": "B"}, "object": {}, "attribute": {}},
        )


def test_candidate_builder_rejects_malformed_source_and_registry() -> None:
    with pytest.raises(CandidateBuildError, match="source_text_invalid"):
        build_candidates("", frozen_r4.ALIASES)
    with pytest.raises(CandidateBuildError, match="alias_entry_invalid"):
        build_candidates("My wallet", {"owner": {"": "SELF"}, "object": {}, "attribute": {}})


def test_frozen_pack_accepted_atoms_have_unique_typed_alias_candidates() -> None:
    pack = json.loads(PACK_PATH.read_text(encoding="utf-8"))
    for proposition in pack["propositions"]:
        result = build_candidates(proposition["proposition_text"], frozen_r4.ALIASES)
        for atom in proposition["expected"]["atoms"]:
            for field, canonical_key, span_key in (
                ("owner", "owner_id", "owner_span"),
                ("object", "object_id", "object_span"),
                ("attribute", "attribute_id", "attribute_span"),
            ):
                matching = [
                    row
                    for row in result["candidates"][field]
                    if row["canonical_id"] == atom[canonical_key]
                    and row["source_span"].casefold() == atom[span_key].casefold()
                ]
                assert len(matching) == 1, (proposition["source_id"], field)


def test_abstention_cases_never_acquire_invented_typed_aliases() -> None:
    pack = json.loads(PACK_PATH.read_text(encoding="utf-8"))
    expected_empty_fields = {
        "R4-P13": {"object", "attribute"},
        "R4-P16": {"owner", "object", "attribute"},
        "R4-P17": {"owner"},
        "R4-P18": {"attribute"},
        "R4-P19": {"attribute"},
        "R4-P20": {"object"},
    }
    for proposition in pack["propositions"]:
        source_id = proposition["source_id"]
        if source_id not in expected_empty_fields:
            continue
        result = build_candidates(proposition["proposition_text"], frozen_r4.ALIASES)
        for field in expected_empty_fields[source_id]:
            assert result["candidates"][field] == [], (source_id, field)


def test_p01_schema_enums_are_exactly_the_harness_candidates() -> None:
    proposition = "My workout plan's activity is running."
    candidates = build_candidates(proposition, frozen_r4.ALIASES, source_id="R4-P01")
    schema = build_candidate_schema("R4-P01", proposition)
    atom_properties = schema["properties"]["atoms"]["items"]["properties"]

    for field in ("owner", "object", "attribute"):
        assert atom_properties[f"{field}_candidate_id"]["enum"] == _candidate_ids(
            candidates, field
        )
    assert schema["properties"]["source_id"]["enum"] == ["R4-P01"]
    assert schema["additionalProperties"] is False
    assert candidates["source_id"] == "R4-P01"
    assert candidates["source_sha256"]


def test_empty_candidate_schema_keeps_empty_array_abstention_path() -> None:
    proposition = "It is now blue."
    candidates = build_candidates(proposition, frozen_r4.ALIASES, source_id="R4-P16")
    schema = build_candidate_schema("R4-P16", proposition)
    atoms_schema = schema["properties"]["atoms"]
    atom_properties = atoms_schema["items"]["properties"]

    assert all(
        not candidates["candidates"][field]
        for field in ("owner", "object", "attribute")
    )
    assert atoms_schema["maxItems"] == 0
    assert all(
        "enum" not in atom_properties[f"{field}_candidate_id"]
        for field in ("owner", "object", "attribute")
    )


def test_candidate_schema_is_bound_to_its_source_case() -> None:
    proposition = "My workout plan's activity is running."
    schema = build_candidate_schema("R4-P01", proposition)
    validator = pytest.importorskip("jsonschema").Draft202012Validator(schema)
    other_case = {
        "source_id": "R4-P02",
        "atoms": [],
        "abstention_reason": "MISSING_OWNER",
    }

    assert not validator.is_valid(other_case)


def test_jsonschema_semantics_keep_abstention_valid_and_empty_candidate_atoms_invalid() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = build_candidate_schema("R4-P16", "It is now blue.")
    validator = jsonschema.Draft202012Validator(schema)

    valid_abstention = {
        "source_id": "R4-P16",
        "atoms": [],
        "abstention_reason": "AMBIGUOUS_REFERENCE",
    }
    invalid_guessed_atom = {
        "source_id": "R4-P16",
        "atoms": [
            {
                "owner_candidate_id": "invented-owner",
                "object_candidate_id": "invented-object",
                "attribute_candidate_id": "invented-attribute",
                "value_span": "blue",
                "cardinality_proposal": "SINGLE_VALUE_AT_A_TIME",
            }
        ],
        "abstention_reason": "NONE",
    }

    assert validator.is_valid(valid_abstention)
    assert not validator.is_valid(invalid_guessed_atom)


def test_all_frozen_control_case_schemas_are_valid_json_schema() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    pack = json.loads(PACK_PATH.read_text(encoding="utf-8"))
    for proposition in pack["propositions"]:
        schema = build_candidate_schema(
            proposition["source_id"], proposition["proposition_text"]
        )
        jsonschema.Draft202012Validator.check_schema(schema)


def test_candidate_proposal_resolves_case_bound_ids_to_exact_witnesses() -> None:
    proposition = {
        "source_id": "R4-P01",
        "proposition_text": "My workout plan's activity is running.",
    }
    candidates = build_candidates(
        proposition["proposition_text"], frozen_r4.ALIASES, source_id="R4-P01"
    )
    atom = {
        f"{field}_candidate_id": candidates["candidates"][field][0]["candidate_id"]
        for field in ("owner", "object", "attribute")
    }
    atom.update({"value_span": "running", "cardinality_proposal": "SINGLE_VALUE_AT_A_TIME"})
    payload = {
        "source_id": "R4-P01",
        "atoms": [atom],
        "abstention_reason": "NONE",
    }

    result = validate_candidate_proposal(
        json.dumps(payload), proposition, scope_id=frozen_r4.FROZEN_SCOPE_ID
    )

    assert result["atoms"][0]["slot_key"] == [
        frozen_r4.FROZEN_SCOPE_ID,
        "SELF",
        "EXERCISE_PLAN",
        "ACTIVITY",
    ]
    assert result["atoms"][0]["witnesses"]["object_span"] == {
        "text": "workout plan",
        "start": 3,
        "end": 15,
    }


def test_candidate_proposal_rejects_cross_case_source_id() -> None:
    proposition = {
        "source_id": "R4-P02",
        "proposition_text": "My workout plan's activity is running.",
    }
    payload = {
        "source_id": "R4-P01",
        "atoms": [],
        "abstention_reason": "MISSING_OWNER",
    }

    with pytest.raises(CandidateBuildError, match="source_id_mismatch"):
        validate_candidate_proposal(
            json.dumps(payload), proposition, scope_id=frozen_r4.FROZEN_SCOPE_ID
        )


def test_candidate_proposal_rejects_empty_atoms_with_none_abstention() -> None:
    proposition = {"source_id": "R4-P16", "proposition_text": "It is now blue."}
    payload = {
        "source_id": "R4-P16",
        "atoms": [],
        "abstention_reason": "NONE",
    }

    with pytest.raises(CandidateBuildError, match="atoms_abstention_inconsistent"):
        validate_candidate_proposal(
            json.dumps(payload), proposition, scope_id=frozen_r4.FROZEN_SCOPE_ID
        )


def test_candidate_proposal_uses_occurrence_ids_to_preserve_repeated_alias_offsets() -> None:
    proposition = {
        "source_id": "R4-P07",
        "proposition_text": "My wallet and wallet have color black.",
    }
    candidates = build_candidates(
        proposition["proposition_text"], frozen_r4.ALIASES, source_id="R4-P07"
    )
    atom = {
        "owner_candidate_id": candidates["candidates"]["owner"][0]["candidate_id"],
        "object_candidate_id": candidates["candidates"]["object"][1]["candidate_id"],
        "attribute_candidate_id": candidates["candidates"]["attribute"][0]["candidate_id"],
        "value_span": "black",
        "cardinality_proposal": "SINGLE_VALUE_AT_A_TIME",
    }
    payload = {"source_id": "R4-P07", "atoms": [atom], "abstention_reason": "NONE"}

    result = validate_candidate_proposal(
        json.dumps(payload), proposition, scope_id=frozen_r4.FROZEN_SCOPE_ID
    )

    assert result["atoms"][0]["witnesses"]["object_span"]["start"] == 14


def test_candidate_proposal_rejects_duplicate_json_keys() -> None:
    proposition = {"source_id": "R4-P16", "proposition_text": "It is now blue."}
    content = (
        '{"source_id":"R4-P16","atoms":[],"atoms":[],'
        '"abstention_reason":"AMBIGUOUS_REFERENCE"}'
    )

    with pytest.raises(CandidateBuildError, match="duplicate_json_key:atoms"):
        validate_candidate_proposal(
            content, proposition, scope_id=frozen_r4.FROZEN_SCOPE_ID
        )


def test_frozen_control_pack_oracle_round_trips_through_candidate_adapter() -> None:
    pack = json.loads(PACK_PATH.read_text(encoding="utf-8"))
    for proposition in pack["propositions"]:
        source_id = proposition["source_id"]
        source_text = proposition["proposition_text"]
        candidate_manifest = build_candidates(
            source_text, frozen_r4.ALIASES, source_id=source_id
        )
        expected = proposition["expected"]
        expected_atoms = expected["atoms"]
        if isinstance(expected_atoms, dict):
            expected_atoms = [expected_atoms]

        proposal_atoms = []
        for expected_atom in expected_atoms:
            proposal_atom = {}
            for field, id_key, span_key in (
                ("owner", "owner_id", "owner_span"),
                ("object", "object_id", "object_span"),
                ("attribute", "attribute_id", "attribute_span"),
            ):
                matches = [
                    row
                    for row in candidate_manifest["candidates"][field]
                    if row["canonical_id"] == expected_atom[id_key]
                    and row["source_span"].casefold()
                    == expected_atom[span_key].casefold()
                ]
                assert len(matches) == 1, (source_id, field)
                proposal_atom[f"{field}_candidate_id"] = matches[0]["candidate_id"]
            proposal_atom["value_span"] = expected_atom["value_span"]
            proposal_atom["cardinality_proposal"] = expected_atom["cardinality"]
            proposal_atoms.append(proposal_atom)

        payload = {
            "source_id": source_id,
            "atoms": proposal_atoms,
            "abstention_reason": expected.get("abstention") or "NONE",
        }
        result = validate_candidate_proposal(
            json.dumps(payload),
            proposition,
            scope_id=frozen_r4.FROZEN_SCOPE_ID,
        )

        assert len(result["atoms"]) == len(expected_atoms), source_id
        for actual, expected_atom in zip(result["atoms"], expected_atoms, strict=True):
            assert (
                actual["owner_id"],
                actual["object_id"],
                actual["attribute_id"],
                actual["value_text"],
                actual["cardinality"],
            ) == (
                expected_atom["owner_id"],
                expected_atom["object_id"],
                expected_atom["attribute_id"],
                expected_atom["value_span"],
                expected_atom["cardinality"],
            ), source_id
