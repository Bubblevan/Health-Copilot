from __future__ import annotations

import copy
import os

import pytest

from tools.research.memory import revision_pairwise_admission as admission


def _identity(memory_id: str, *, subject: str = "user", attribute: str = "preferred_exercise", value: str = "running", kind: str = "SINGLETON_STATE", origin: str = "MODEL_VALIDATED") -> dict[str, str]:
    return {
        "memory_id": memory_id,
        "revision_kind": kind,
        "subject_key": subject,
        "attribute_key": attribute,
        "value_text": value,
        "identity_origin": origin,
    }


def _source(memory_id: str, text: str, *, authority: str = "user", scope: str = "scope-a") -> dict[str, str]:
    return {
        "memory_id": memory_id,
        "scope_id": scope,
        "source_authority": authority,
        "proposition_text": text,
    }


def test_numeric_value_mismatch_is_blocked_deterministically() -> None:
    assert admission.grounding_result("My Instagram has 600 followers", "500 followers") == (
        "BLOCKED",
        "IDENTITY_HINT_BLOCKED_VALUE_NUMERIC_MISMATCH",
    )


def test_zero_informative_overlap_is_blocked_deterministically() -> None:
    assert admission.grounding_result("I prefer swimming on weekends", "likes jazz music") == (
        "BLOCKED",
        "IDENTITY_HINT_BLOCKED_VALUE_UNGROUNDED",
    )


def test_grounding_uses_nfkc_lowercase_and_underscore_normalization() -> None:
    assert admission.grounding_result("ＦＡＶＯＲＩＴＥ_ＣＯＬＯＲ is teal", "teal") == (
        "PASS",
        "VALUE_LOCALLY_GROUNDED",
    )


def test_jsonl_projection_discards_time_and_benchmark_fields() -> None:
    row = admission.jsonl_projection(
        '{"memory_id":"m1","scope_id":"s","proposition_text":"prefers tea",'
        '"source_authority":"user","observed_at":"do-not-load",'
        '"question":"do-not-load","gold":"do-not-load"}',
        admission.PROPOSAL_SOURCE_FIELDS,
    )
    assert row == {
        "memory_id": "m1",
        "scope_id": "s",
        "proposition_text": "prefers tea",
        "source_authority": "user",
    }
    assert not admission.TEMPORAL_OR_BENCHMARK_FIELDS.intersection(row)


def test_jsonl_projection_skips_non_allowlisted_values_without_decoding() -> None:
    huge_numeric_timestamp = "9" * 5000
    line = (
        '{"memory_id":"m1","observed_at":'
        + huge_numeric_timestamp
        + ',"nested":{"session_date":"unread"},"source_authority":"user"}'
    )
    assert admission.jsonl_projection(line, admission.PROPOSAL_SOURCE_FIELDS) == {
        "memory_id": "m1",
        "source_authority": "user",
    }


def test_assistant_origin_never_becomes_user_revision_candidate() -> None:
    identities = [_identity("m1"), _identity("m2")]
    sources = [
        _source("m1", "I prefer running", authority="assistant"),
        _source("m2", "I prefer swimming", authority="assistant"),
    ]
    universe, decisions, _ = admission.build_candidate_universe(identities, sources)
    assert universe["repeated_candidate_groups"] == []
    assert all(row["revision_eligibility"] == "INELIGIBLE_ASSISTANT_ORIGIN" for row in decisions)


def test_candidate_keys_only_create_exact_candidate_groups() -> None:
    identities = [
        _identity("m1", value="running"),
        _identity("m2", value="swimming"),
        _identity("m3", attribute="workout_days", value="three days"),
    ]
    sources = [
        _source("m1", "I prefer running"),
        _source("m2", "I prefer swimming"),
        _source("m3", "I exercise three days each week"),
    ]
    universe, _, _ = admission.build_candidate_universe(identities, sources)
    assert len(universe["repeated_candidate_groups"]) == 1
    assert len(universe["singletons_without_revision_history"]) == 1
    assert universe["repeated_candidate_groups"][0]["diagnostic_attribute_key"] == "preferred_exercise"


def test_singleton_identity_hint_does_not_trigger_pair_or_slot() -> None:
    universe, _, _ = admission.build_candidate_universe(
        [_identity("m1")], [_source("m1", "I prefer running")]
    )
    assert admission.pairwise_records(universe, "a" * 64) == []
    overlay, slots = admission.build_admission_overlay(universe, [], "b" * 64)
    assert overlay[0]["admission_status"] == "NO_REVISION_HISTORY"
    assert overlay[0]["revision_slot_id"] is None
    assert slots == []


def test_pair_manifest_enumerates_every_unordered_pair_stably() -> None:
    identities = [_identity(f"m{i}", value=value) for i, value in enumerate(("running", "swimming", "cycling"), 1)]
    sources = [
        _source("m1", "I prefer running"),
        _source("m2", "I now prefer swimming"),
        _source("m3", "I choose cycling now"),
    ]
    universe, _, _ = admission.build_candidate_universe(identities, sources)
    rows = admission.pairwise_records(universe, "c" * 64)
    assert len(rows) == 3
    assert len({row["pair_id"] for row in rows}) == 3
    reverse_rows = admission.pairwise_records(
        {**universe, "repeated_candidate_groups": list(reversed(universe["repeated_candidate_groups"]))},
        "c" * 64,
    )
    assert rows == reverse_rows


def test_pairwise_request_contains_only_the_two_propositions() -> None:
    request = admission.pairwise_request_projection(
        {
            "pair_id": "x",
            "group_id": "g",
            "memory_id_a": "a",
            "memory_id_b": "b",
            "proposition_a": "Prefers tea",
            "proposition_b": "Now prefers coffee",
            "observed_at": "not allowed",
            "subject_key": "not allowed",
        }
    )
    assert request == {
        "proposition_a": "Prefers tea",
        "proposition_b": "Now prefers coffee",
    }


@pytest.mark.parametrize(
    ("content", "finish_reason", "expected"),
    [
        ('{"verdict":"SAME_MUTABLE_SLOT"}', "stop", ("SAME_MUTABLE_SLOT", None)),
        ('{"verdict":"NOT_A_VERDICT"}', "stop", ("UNKNOWN", "ILLEGAL_VERDICT")),
        ('{"verdict":"UNRELATED","key":"assistant"}', "stop", ("UNKNOWN", "ILLEGAL_RESPONSE_SHAPE")),
        ('{"verdict":"UNRELATED"', "stop", ("UNKNOWN", "MALFORMED_JSON")),
        ('{"verdict":"UNRELATED"}', "length", ("UNKNOWN", "COMPLETION_TRUNCATED")),
        (None, "stop", ("UNKNOWN", "MALFORMED_ASSISTANT_CONTENT")),
    ],
)
def test_bad_or_illegal_model_output_is_terminal_unknown(
    content: str | None, finish_reason: str, expected: tuple[str, str | None]
) -> None:
    assert admission.parse_pairwise_output(content, finish_reason) == expected


def test_malformed_json_duplicate_key_is_unknown() -> None:
    assert admission.parse_pairwise_output(
        '{"verdict":"SAME_MUTABLE_SLOT","verdict":"UNRELATED"}', "stop"
    ) == ("UNKNOWN", "MALFORMED_JSON")


def test_one_unrelated_pair_blocks_entire_group() -> None:
    universe, _, _ = admission.build_candidate_universe(
        [_identity("m1"), _identity("m2", value="swimming"), _identity("m3", value="cycling")],
        [
            _source("m1", "I prefer running"),
            _source("m2", "I prefer swimming"),
            _source("m3", "I prefer cycling"),
        ],
    )
    pairs = admission.pairwise_records(universe, "c" * 64)
    pair_rows = [
        {**row, "verdict": "SAME_MUTABLE_SLOT" if index < 2 else "UNRELATED"}
        for index, row in enumerate(pairs)
    ]
    overlay, slots = admission.build_admission_overlay(universe, pair_rows, "d" * 64)
    assert overlay[0]["admission_status"] == "REVISION_BLOCKED_PAIRWISE_INCONSISTENCY"
    assert overlay[0]["revision_slot_id"] is None
    assert slots == []


def test_unknown_pair_blocks_entire_group() -> None:
    universe, _, _ = admission.build_candidate_universe(
        [_identity("m1"), _identity("m2", value="swimming")],
        [_source("m1", "I prefer running"), _source("m2", "I prefer swimming")],
    )
    pair = admission.pairwise_records(universe, "c" * 64)[0]
    overlay, slots = admission.build_admission_overlay(
        universe, [{**pair, "verdict": "UNKNOWN"}], "d" * 64
    )
    assert overlay[0]["admission_status"] == "REVISION_BLOCKED_PAIRWISE_INCONSISTENCY"
    assert slots == []


def test_all_same_pairs_allow_harness_admission_and_slot_id() -> None:
    universe, _, _ = admission.build_candidate_universe(
        [_identity("m1"), _identity("m2", value="swimming")],
        [_source("m1", "I prefer running"), _source("m2", "I prefer swimming")],
    )
    pair = admission.pairwise_records(universe, "c" * 64)[0]
    overlay, slots = admission.build_admission_overlay(
        universe, [{**pair, "verdict": "SAME_MUTABLE_SLOT"}], "d" * 64
    )
    assert overlay[0]["admission_status"] == "REVISION_ADMITTED_PAIRWISE_ALL_SAME"
    assert overlay[0]["revision_slot_id"] == slots[0]["revision_slot_id"]
    assert len(overlay[0]["revision_slot_id"]) == 64


def test_opaque_slot_id_ignores_diagnostic_subject_and_attribute_keys() -> None:
    universe, _, _ = admission.build_candidate_universe(
        [_identity("m1"), _identity("m2", value="swimming")],
        [_source("m1", "I prefer running"), _source("m2", "I prefer swimming")],
    )
    pair = admission.pairwise_records(universe, "c" * 64)[0]
    pair_rows = [{**pair, "verdict": "SAME_MUTABLE_SLOT"}]
    first, _ = admission.build_admission_overlay(universe, pair_rows, "d" * 64)
    changed = copy.deepcopy(universe)
    changed["repeated_candidate_groups"][0]["diagnostic_subject_key"] = "edited-hint"
    changed["repeated_candidate_groups"][0]["diagnostic_attribute_key"] = "edited-label"
    second, _ = admission.build_admission_overlay(changed, pair_rows, "d" * 64)
    assert first[0]["revision_slot_id"] == second[0]["revision_slot_id"]


def test_timestamp_access_helper_fails_closed_until_semantic_freeze() -> None:
    with pytest.raises(RuntimeError, match="semantic_freeze_required"):
        admission.require_semantic_freeze(None)
    with pytest.raises(RuntimeError, match="semantic_freeze_marker_invalid"):
        admission.require_semantic_freeze(b'{"status":"CANDIDATES_FROZEN"}')
    assert admission.require_semantic_freeze(b'{"status":"SEMANTIC_DECISIONS_FROZEN"}')[
        "status"
    ] == "SEMANTIC_DECISIONS_FROZEN"


def test_memory_store_mutations_are_fixed_at_zero() -> None:
    assert admission.admission_contract()["memory_store_mutations"] == {
        "ADD": 0,
        "UPDATE": 0,
        "DELETE": 0,
        "SUPERSEDED": 0,
    }


def test_pairwise_contract_is_local_without_openai_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    contract = admission.pairwise_contract()
    assert os.environ.get("OPENAI_API_KEY") is None
    assert contract["model"]["provider"] == "local_loopback_llama_cpp"
    assert contract["hosted_fallback"] is False
    assert contract["retries"] == 0
