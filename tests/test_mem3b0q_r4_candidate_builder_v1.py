from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.research.memory import mem3b0q_r4 as frozen_r4
from tools.research.memory.mem3b0q_r4_candidate_builder_v1 import (
    CandidateBuildError,
    build_candidates,
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
