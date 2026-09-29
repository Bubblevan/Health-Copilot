from __future__ import annotations

import pytest

from tools.research.memory.flatprop_ledger_repair import restore_source_session_identity
from tools.research.memory.flatprop_recursive_recovery import (
    aggregate_exact_leaf_propositions,
    assert_primary_leaf_coverage,
    balanced_partition,
    balanced_split_index,
    overflow_disposition,
    recursive_child_identity,
    resolve_recursive_overflow,
)


def _child_identity(
    parent: str,
    depth: int,
    ordinals: list[int],
    session_identity: str = "session-identity",
) -> tuple[str, dict[str, object]]:
    return recursive_child_identity(
        {
            "split_contract_sha256": "split-contract",
            "session_identity_sha256": session_identity,
            "root_initial_chunk_id": "104d270a9c92d9d464dc5eb07adf1eac6881d5edd6a1d437fc040ec814508bc9",
            "parent_chunk_id": parent,
            "recursion_depth": depth,
            "primary_unit_kind": "turn",
            "primary_unit_ordinals": ordinals,
            "primary_span_ordinals": ordinals,
            "overlap_turn_ordinals": [],
            "overlap_span_ordinals": [],
            "rendered_source_payload_sha256": f"payload-{ordinals}",
            "dynamic_schema_sha256": f"schema-{ordinals}",
        }
    )


def test_failed_historical_root_has_two_deterministic_child_identities():
    first_left = _child_identity(
        "104d270a9c92d9d464dc5eb07adf1eac6881d5edd6a1d437fc040ec814508bc9", 1, [0, 1, 2, 3]
    )
    first_right = _child_identity(
        "104d270a9c92d9d464dc5eb07adf1eac6881d5edd6a1d437fc040ec814508bc9", 1, [4, 5, 6, 7]
    )
    replay_left = _child_identity(
        "104d270a9c92d9d464dc5eb07adf1eac6881d5edd6a1d437fc040ec814508bc9", 1, [0, 1, 2, 3]
    )
    replay_right = _child_identity(
        "104d270a9c92d9d464dc5eb07adf1eac6881d5edd6a1d437fc040ec814508bc9", 1, [4, 5, 6, 7]
    )

    assert first_left[0] == replay_left[0]
    assert first_right[0] == replay_right[0]
    assert first_left[0] != first_right[0]


def test_balanced_split_is_deterministic_and_ties_choose_earliest_boundary():
    assert balanced_split_index([1, 2, 1]) == 1
    assert balanced_partition(["a", "b", "c"], [1, 2, 1]) == (["a"], ["b", "c"])
    assert balanced_partition(["a", "b", "c"], [1, 2, 1]) == (
        ["a"],
        ["b", "c"],
    )


def test_restores_source_session_without_changing_recursive_cache_identity():
    root_id = "104d270a9c92d9d464dc5eb07adf1eac6881d5edd6a1d437fc040ec814508bc9"
    chunk_id, identity = _child_identity(
        root_id, 1, [0, 1, 2, 3], session_identity="source-session"
    )
    attempt = {
        "chunk_id": chunk_id,
        "root_initial_chunk_id": root_id,
        "recursion_depth": 1,
        "chunk_identity": identity,
        "session_identity_sha256": chunk_id,
    }

    restored = restore_source_session_identity(attempt, {root_id: "source-session"})

    assert restored["session_identity_sha256"] == "source-session"
    assert restored["session_identity_sha256_for_cache"] == chunk_id
    assert restored["session_identity_sha256_provider_cache_key"] == chunk_id
    assert attempt["session_identity_sha256"] == chunk_id


def test_rejects_recursive_attempt_with_wrong_embedded_source_session():
    root_id = "104d270a9c92d9d464dc5eb07adf1eac6881d5edd6a1d437fc040ec814508bc9"
    chunk_id, identity = _child_identity(root_id, 1, [0, 1, 2, 3])
    attempt = {
        "chunk_id": chunk_id,
        "root_initial_chunk_id": root_id,
        "recursion_depth": 1,
        "chunk_identity": identity,
    }

    with pytest.raises(ValueError, match="source_session_identity_mismatch"):
        restore_source_session_identity(attempt, {root_id: "different-session"})


def test_nested_length_overflow_splits_recursively_and_discards_parent_output():
    nodes = {
        "root": {"chunk_id": "root", "primary_span_ordinals": [0, 1, 2, 3]},
        "left": {"chunk_id": "left", "primary_span_ordinals": [0, 1]},
        "right": {"chunk_id": "right", "primary_span_ordinals": [2, 3]},
        "left_a": {"chunk_id": "left_a", "primary_span_ordinals": [0]},
        "left_b": {"chunk_id": "left_b", "primary_span_ordinals": [1]},
    }
    split_map = {"root": ("left", "right"), "left": ("left_a", "left_b")}
    responses = {
        "root": {
            "finish_reason": "length",
            "success": False,
            "normalized_packet": {
                "propositions": [{"proposition_text": "partial", "evidence_refs": ["S0"]}]
            },
        },
        "left": {
            "finish_reason": "length",
            "success": False,
            "normalized_packet": {
                "propositions": [{"proposition_text": "partial too", "evidence_refs": ["S0"]}]
            },
        },
        "left_a": {
            "finish_reason": "stop",
            "success": True,
            "normalized_packet": {"propositions": []},
        },
        "left_b": {
            "finish_reason": "stop",
            "success": True,
            "normalized_packet": {"propositions": []},
        },
        "right": {
            "finish_reason": "stop",
            "success": True,
            "normalized_packet": {"propositions": []},
        },
    }
    leaves, attempts = resolve_recursive_overflow(
        nodes["root"],
        execute=lambda node: responses[node["chunk_id"]],
        split=lambda node: tuple(nodes[key] for key in split_map[node["chunk_id"]]),
    )

    assert [row["chunk_id"] for row in leaves] == ["left_a", "left_b", "right"]
    assert [row["status"] for row in attempts[:2]] == ["OVERFLOW_PARENT", "OVERFLOW_PARENT"]
    assert all(row["semantic_output_used"] is False for row in attempts[:2])
    assert not any("partial" in str(row.get("normalized_packet")) for row in leaves)


def test_successful_sibling_is_reused_when_other_sibling_overflows():
    nodes = {
        "root": {"chunk_id": "root", "primary_span_ordinals": [0, 1, 2, 3]},
        "left": {"chunk_id": "left", "primary_span_ordinals": [0, 1]},
        "right": {"chunk_id": "right", "primary_span_ordinals": [2, 3]},
        "right_a": {"chunk_id": "right_a", "primary_span_ordinals": [2]},
        "right_b": {"chunk_id": "right_b", "primary_span_ordinals": [3]},
    }
    split_map = {"root": ("left", "right"), "right": ("right_a", "right_b")}
    results: dict[str, dict[str, object]] = {}
    provider_calls: dict[str, int] = {}

    def execute(node: dict[str, object]) -> dict[str, object]:
        key = str(node["chunk_id"])
        if key in results:
            return results[key]
        provider_calls[key] = provider_calls.get(key, 0) + 1
        result = (
            {"finish_reason": "length", "success": False}
            if key in {"root", "right"}
            else {
                "finish_reason": "stop",
                "success": True,
                "normalized_packet": {"propositions": []},
            }
        )
        results[key] = result
        return result

    def split(node: dict[str, object]) -> tuple[dict[str, object], dict[str, object]]:
        return tuple(nodes[key] for key in split_map[str(node["chunk_id"])])

    first_leaves, _ = resolve_recursive_overflow(nodes["root"], execute=execute, split=split)
    second_leaves, _ = resolve_recursive_overflow(nodes["root"], execute=execute, split=split)

    assert [row["chunk_id"] for row in first_leaves] == ["left", "right_a", "right_b"]
    assert [row["chunk_id"] for row in second_leaves] == ["left", "right_a", "right_b"]
    assert provider_calls["left"] == 1
    assert provider_calls["right"] == 1
    assert results["left"]["success"] is True


def test_single_rawspan_overflow_is_irreducible():
    with pytest.raises(RuntimeError, match="IRREDUCIBLE_WRITER_OVERFLOW"):
        resolve_recursive_overflow(
            {"chunk_id": "leaf", "primary_span_ordinals": [4]},
            execute=lambda node: {"finish_reason": "length", "success": False},
            split=lambda node: pytest.fail("single RawSpan must not be split"),
        )
    assert overflow_disposition(1) == "IRREDUCIBLE_WRITER_OVERFLOW"
    assert overflow_disposition(2) == "SUBDIVISION"


def test_only_exact_same_text_and_canonical_evidence_identity_collapses():
    catalog = [
        {"evidence_ref": "S0"},
        {"evidence_ref": "S1"},
    ]
    leaves = [
        {
            "chunk_id": "leaf-a",
            "success": True,
            "terminal": True,
            "primary_start_ordinal": 0,
            "primary_end_ordinal": 0,
            "primary_span_ids": ["S0"],
            "overlap_span_ids": [],
            "normalized_packet": {
                "propositions": [
                    {"proposition_text": "Same claim.", "evidence_refs": ["S0"]},
                    {"proposition_text": "Same claim.", "evidence_refs": ["S0"]},
                    {"proposition_text": "Same claim!", "evidence_refs": ["S0"]},
                ],
            },
        },
        {
            "chunk_id": "leaf-b",
            "success": True,
            "terminal": True,
            "primary_start_ordinal": 1,
            "primary_end_ordinal": 1,
            "primary_span_ids": ["S1"],
            "overlap_span_ids": [],
            "normalized_packet": {
                "propositions": [
                    {"proposition_text": "Same claim.", "evidence_refs": ["S0"]},
                    {"proposition_text": "Same claim.", "evidence_refs": ["S1"]},
                ],
            },
        },
    ]

    merged, diagnostics = aggregate_exact_leaf_propositions(catalog=catalog, leaf_outputs=leaves)

    assert len(merged) == 3
    assert diagnostics["exact_duplicates_removed_within_leaf"] == 1
    assert diagnostics["exact_duplicates_removed_across_leaves"] == 1
    assert diagnostics["near_duplicate_pairs_retained"] >= 1
    assert diagnostics["near_duplicates_merged"] is False


def test_duplicate_emissions_from_overlap_are_attributed_but_not_primary_ownership():
    catalog = [{"evidence_ref": "S0"}, {"evidence_ref": "S1"}]
    outputs = [
        {
            "chunk_id": "left",
            "success": True,
            "terminal": True,
            "primary_start_ordinal": 0,
            "primary_end_ordinal": 0,
            "primary_span_ids": ["S0"],
            "overlap_span_ids": [],
            "normalized_packet": {
                "propositions": [{"proposition_text": "Fact.", "evidence_refs": ["S0"]}]
            },
        },
        {
            "chunk_id": "right",
            "success": True,
            "terminal": True,
            "primary_start_ordinal": 1,
            "primary_end_ordinal": 1,
            "primary_span_ids": ["S1"],
            "overlap_span_ids": ["S0"],
            "normalized_packet": {
                "propositions": [{"proposition_text": "Fact.", "evidence_refs": ["S0"]}]
            },
        },
    ]

    merged, diagnostics = aggregate_exact_leaf_propositions(catalog=catalog, leaf_outputs=outputs)

    assert len(merged) == 1
    assert diagnostics["exact_duplicates_attributed_to_overlap"] == 1


def test_primary_leaf_ownership_requires_exactly_once_chronological_coverage():
    assert_primary_leaf_coverage([0, 1, 2], [[0], [1, 2]])
    with pytest.raises(ValueError, match="exactly_once"):
        assert_primary_leaf_coverage([0, 1, 2], [[0, 1], [1, 2]])
