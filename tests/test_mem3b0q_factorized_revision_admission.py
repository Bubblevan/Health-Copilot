from __future__ import annotations

from tools.research.memory import factorized_revision_admission as admission
from tools.research.memory import run_mem3b0q_factorized_revision_admission as runner
from tools.research.memory.revision_pairwise_admission import jsonl_projection


def test_forbidden_timestamp_sentinel_is_never_decoded() -> None:
    sentinel = "9" * 5000
    raw = (
        '{"memory_id":"m1","scope_id":"s1","proposition_text":"prefers tea",'
        '"source_authority":"user","observed_at":{"nested":[{"sentinel":' + sentinel + "}]}}"
    )
    assert jsonl_projection(
        raw, frozenset({"memory_id", "scope_id", "proposition_text", "source_authority"})
    ) == {
        "memory_id": "m1",
        "scope_id": "s1",
        "proposition_text": "prefers tea",
        "source_authority": "user",
    }


def test_factorized_instagram_historical_observations_are_one_state_dimension() -> None:
    content = (
        '{"same_state_dimension":"YES","state_cardinality":"SINGLE_VALUE_AT_A_TIME",'
        '"value_relation":"DIFFERENT"}'
    )
    verdict, error = admission.parse_factorized_output(content, "stop")
    assert error is None
    assert admission.is_positive_singleton_edge(verdict)


def test_completed_event_is_not_a_state_even_when_dimension_matches() -> None:
    content = (
        '{"same_state_dimension":"YES","state_cardinality":"NOT_A_STATE","value_relation":"SAME"}'
    )
    verdict, error = admission.parse_factorized_output(content, "stop")
    assert error is None
    assert not admission.is_positive_singleton_edge(verdict)


def test_multi_value_preference_does_not_form_single_value_edge() -> None:
    verdict, error = admission.parse_factorized_output(
        '{"same_state_dimension":"YES","state_cardinality":"MULTI_VALUE_OR_SET",'
        '"value_relation":"DIFFERENT"}',
        "stop",
    )
    assert error is None
    assert not admission.is_positive_singleton_edge(verdict)


def test_malformed_output_is_local_unknown_fallback() -> None:
    verdict, error = admission.parse_factorized_output("not json", "stop")
    assert verdict == admission.UNKNOWN_VERDICT
    assert verdict["pair_origin"] == "HARNESS_UNKNOWN_FALLBACK"
    assert error == "MALFORMED_JSON_OR_DUPLICATE_KEY"


def test_request_projection_contains_propositions_only() -> None:
    pair = {
        "memory_id_a": "a",
        "memory_id_b": "b",
        "scope_id": "s",
        "candidate_sources": ["EXACT_HINT"],
        "proposition_a": "likes tea",
        "proposition_b": "likes coffee",
    }
    assert admission.request_projection(pair) == {
        "proposition_a": "likes tea",
        "proposition_b": "likes coffee",
    }


def test_closure_discovers_missing_intra_component_pair() -> None:
    eligible = [
        {
            "memory_id": item,
            "scope_id": "s",
            "subject_key": "u",
            "attribute_key": "x",
            "proposition_text": item,
        }
        for item in ("a", "b", "c")
    ]
    contract_sha = "f" * 64
    pairs = [
        {
            "pair_id": admission.pair_id("a", "b", contract_sha),
            "memory_id_a": "a",
            "memory_id_b": "b",
        },
        {
            "pair_id": admission.pair_id("b", "c", contract_sha),
            "memory_id_a": "b",
            "memory_id_b": "c",
        },
    ]
    verdicts = {
        pair["pair_id"]: {
            "same_state_dimension": "YES",
            "state_cardinality": "SINGLE_VALUE_AT_A_TIME",
        }
        for pair in pairs
    }
    closure, oversized = admission.closure_pairs(eligible, pairs, verdicts, contract_sha)
    assert [(row["memory_id_a"], row["memory_id_b"]) for row in closure] == [("a", "c")]
    assert oversized == []


def test_exact_and_semantic_candidate_sources_union_without_cross_scope_pairs() -> None:
    eligible = [
        {
            "memory_id": "a",
            "scope_id": "s1",
            "subject_key": "u",
            "attribute_key": "sport",
            "proposition_text": "likes running",
        },
        {
            "memory_id": "b",
            "scope_id": "s1",
            "subject_key": "u",
            "attribute_key": "sport",
            "proposition_text": "likes swimming",
        },
        {
            "memory_id": "c",
            "scope_id": "s1",
            "subject_key": "u",
            "attribute_key": "other",
            "proposition_text": "likes cycling",
        },
        {
            "memory_id": "d",
            "scope_id": "s2",
            "subject_key": "u",
            "attribute_key": "sport",
            "proposition_text": "likes running",
        },
    ]
    vectors = {
        "a": [1.0, 0.0],
        "b": [0.0, 1.0],
        "c": [0.999949, 0.0101005],
        "d": [1.0, 0.0],
    }
    rows, stats = admission.candidate_pairs(eligible, vectors, "f" * 64, top_k=1)
    by_pair = {(row["memory_id_a"], row["memory_id_b"]): row for row in rows}
    assert by_pair[("a", "b")]["candidate_sources"] == ["EXACT_HINT"]
    assert by_pair[("a", "c")]["candidate_sources"] == ["SEMANTIC_NEIGHBOR"]
    assert not any("d" in pair for pair in by_pair)
    assert stats["union_seed_pairs"] == len(rows)


def test_pair_id_and_slot_id_are_deterministic() -> None:
    contract_sha = "1" * 64
    assert admission.pair_id("a", "b", contract_sha) == admission.pair_id("b", "a", contract_sha)
    expected = admission.sha256_bytes(("scopeab" + contract_sha).encode("utf-8"))
    assert admission.revision_slot_id("scope", ["b", "a"], contract_sha) == expected


def test_positive_positive_negative_chain_never_becomes_one_slot() -> None:
    eligible = [
        {
            "memory_id": item,
            "scope_id": "s",
            "subject_key": "u",
            "attribute_key": "x",
            "proposition_text": item,
        }
        for item in ("a", "b", "c")
    ]
    contract_sha = "f" * 64
    all_pairs = []
    verdicts = {}
    for first, second, positive in (("a", "b", True), ("b", "c", True), ("a", "c", False)):
        pid = admission.pair_id(first, second, contract_sha)
        all_pairs.append({"pair_id": pid, "memory_id_a": first, "memory_id_b": second})
        verdicts[pid] = {
            "same_state_dimension": "YES" if positive else "NO",
            "state_cardinality": "SINGLE_VALUE_AT_A_TIME",
            "value_relation": "DIFFERENT",
        }
    graph, slots, _ = admission.build_slot_manifest(eligible, all_pairs, verdicts, contract_sha)
    assert slots == []
    assert graph["maximal_cliques"] == [
        {
            "member_memory_ids": ["a", "b"],
            "scope_id": "s",
            "status": "AMBIGUOUS_OVERLAPPING_CLIQUE",
            "ambiguous_members": ["b"],
            "revision_slot_id": None,
        },
        {
            "member_memory_ids": ["b", "c"],
            "scope_id": "s",
            "status": "AMBIGUOUS_OVERLAPPING_CLIQUE",
            "ambiguous_members": ["b"],
            "revision_slot_id": None,
        },
    ]


def test_slot_id_does_not_depend_on_b0r_key_spelling() -> None:
    first = admission.revision_slot_id("scope", ["a", "b"], "c" * 64)
    _different_diagnostic_keys = {"subject_key": "user_pref", "attribute_key": "exercise"}
    second = admission.revision_slot_id("scope", ["b", "a"], "c" * 64)
    assert first == second


def test_protocol_sentinel_gate_and_frozen_runtime_contracts() -> None:
    assert runner._self_test_forbidden_metadata_skip() == {
        "forbidden_metadata_decode_count": 0,
        "sentinel_projection_passed": True,
    }
    _candidate, factor_contract, slot_contract, *_ = runner._candidate_contracts()
    assert factor_contract == runner.FACTOR_CONTRACT
    assert set(factor_contract["output_schema"]["required"]) == {
        "same_state_dimension",
        "state_cardinality",
        "value_relation",
    }
    assert slot_contract["state_mutations"] == {
        "ADD": 0,
        "UPDATE": 0,
        "DELETE": 0,
        "SUPERSEDED": 0,
    }


def test_large_component_is_blocked_without_closure_expansion() -> None:
    eligible = [
        {
            "memory_id": f"m{index:02}",
            "scope_id": "s",
            "subject_key": "u",
            "attribute_key": "x",
            "proposition_text": str(index),
        }
        for index in range(admission.MAX_COMPONENT_SIZE + 1)
    ]
    contract_sha = "f" * 64
    pairs = []
    verdicts = {}
    for index in range(admission.MAX_COMPONENT_SIZE):
        first, second = f"m{index:02}", f"m{index + 1:02}"
        pid = admission.pair_id(first, second, contract_sha)
        pairs.append({"pair_id": pid, "memory_id_a": first, "memory_id_b": second})
        verdicts[pid] = {
            "same_state_dimension": "YES",
            "state_cardinality": "SINGLE_VALUE_AT_A_TIME",
        }
    closure, oversized = admission.closure_pairs(eligible, pairs, verdicts, contract_sha)
    assert closure == []
    assert oversized == [[row["memory_id"] for row in eligible]]
