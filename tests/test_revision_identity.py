from __future__ import annotations

import json

import pytest

from tools.research.memory.revision_identity import (
    IdentityContractError,
    IdentityLengthStop,
    build_request,
    candidate_groups,
    execute_with_bisection,
    fragmentation_diagnostics,
    normalize_key,
    response_schema,
    split_batch,
    validate_identity_response,
)


def source_row(memory_id: str, text: str = "User likes apples", **extra):
    return {
        "memory_id": memory_id,
        "proposition_text": text,
        "source_authority": "user",
        "scope_id": "longmemeval:case-a",
        "question_id": "must-not-enter-request",
        "observed_at": "2023-01-01T00:00:00Z",
        **extra,
    }


def model_row(memory_id: str, **overrides):
    return {
        "memory_id": memory_id,
        "revision_kind": "SINGLETON_STATE",
        "subject_key": "User",
        "attribute_key": "favorite-food",
        "value_text": "apples",
        **overrides,
    }


def response_bytes(rows):
    return json.dumps({"identities": rows}).encode("utf-8")


def test_request_projects_only_memory_side_fields_and_dynamic_ids():
    request = build_request(
        [source_row("m1"), source_row("m2", "User likes pears")],
        model="health-memory-qwen3-8b",
        max_tokens=8192,
    )
    payload = json.loads(request["messages"][1]["content"])
    assert set(payload) == {"memories"}
    assert [set(row) for row in payload["memories"]] == [
        {"memory_id", "proposition_text", "source_authority", "scope_id"},
        {"memory_id", "proposition_text", "source_authority", "scope_id"},
    ]
    assert "must-not-enter-request" not in request["messages"][1]["content"]
    schema = request["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["identities"]["items"]["properties"]["memory_id"]["enum"] == [
        "m1",
        "m2",
    ]
    assert request["temperature"] == 0
    assert request["seed"] == 42
    assert request["max_tokens"] == 8192
    assert request["chat_template_kwargs"] == {"enable_thinking": False}


def test_response_requires_exact_coverage_and_canonical_input_order():
    batch = [source_row("m2"), source_row("m1")]
    payload = response_bytes([model_row("m1"), model_row("m2", value_text="pears")])
    result = validate_identity_response(payload, batch)["identities"]
    assert [row["memory_id"] for row in result] == ["m2", "m1"]
    assert result[0]["subject_key"] == "user"
    assert result[0]["attribute_key"] == "favorite_food"


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([model_row("m1")], "missing_memory_ids"),
        ([model_row("m1"), model_row("m1")], "duplicate_memory_id"),
        ([model_row("m1"), model_row("m3")], "unknown_memory_id"),
        ([{**model_row("m1"), "current": True}], "invalid_identity_row_fields"),
        ([model_row("m1", revision_kind="UPDATE")], "illegal_revision_kind"),
        ([model_row("m1", attribute_key="old_address")], "forbidden_temporal_key_token"),
    ],
)
def test_response_rejects_invalid_rows(rows, message):
    with pytest.raises(IdentityContractError, match=message):
        validate_identity_response(response_bytes(rows), [source_row("m1"), source_row("m2")])


def test_response_rejects_extra_root_field_and_invalid_key_syntax():
    with pytest.raises(IdentityContractError, match="invalid_identity_root"):
        validate_identity_response(
            json.dumps({"identities": [model_row("m1")], "timestamp": "today"}).encode(),
            [source_row("m1")],
        )
    with pytest.raises(IdentityContractError, match="invalid_key_syntax"):
        validate_identity_response(
            response_bytes([model_row("m1", subject_key="user/account")]), [source_row("m1")]
        )


def test_key_normalization_is_deterministic_and_keeps_allowed_current_slot_example():
    assert normalize_key("  Current--Employer ", field="attribute_key", memory_id="m1") == (
        "current_employer"
    )
    assert normalize_key("USER   MOTHER", field="subject_key", memory_id="m1") == "user_mother"


def test_dynamic_schema_rejects_duplicate_batch_ids():
    with pytest.raises(IdentityContractError, match="unique_batch_ids"):
        response_schema(["m1", "m1"])


def test_candidate_groups_use_exact_scope_subject_attribute_and_singleton_only():
    identities = [
        {
            "memory_id": "m1",
            "revision_kind": "SINGLETON_STATE",
            "subject_key": "user",
            "attribute_key": "city",
            "value_text": "Boston",
        },
        {
            "memory_id": "m2",
            "revision_kind": "SINGLETON_STATE",
            "subject_key": "user",
            "attribute_key": "city",
            "value_text": "Paris",
        },
        {
            "memory_id": "m3",
            "revision_kind": "SINGLETON_STATE",
            "subject_key": "user",
            "attribute_key": "city",
            "value_text": "Rome",
        },
        {
            "memory_id": "m4",
            "revision_kind": "SET_STATE",
            "subject_key": "user",
            "attribute_key": "city",
            "value_text": "Tokyo",
        },
    ]
    source = [
        {
            **source_row("m1"),
            "reader_value": {"observed_at": "2023-01-01T00:00:00Z"},
            "source_session_id": "s1",
        },
        {
            **source_row("m2"),
            "reader_value": {"observed_at": "2023-02-01T00:00:00Z"},
            "source_session_id": "s2",
        },
        {
            **source_row("m3"),
            "scope_id": "longmemeval:case-b",
            "reader_value": {"observed_at": "2023-03-01T00:00:00Z"},
            "source_session_id": "s3",
        },
        source_row("m4"),
    ]
    groups = candidate_groups(identities, source)
    assert len(groups) == 1
    assert groups[0]["memory_ids"] == ["m1", "m2"]
    assert groups[0]["distinct_value_text_count"] == 2
    assert set(groups[0]["observations"][0]) == {"memory_id", "value_text"}
    assert "source_sessions" not in groups[0]
    assert "observed_timestamps" not in groups[0]


def test_length_stop_bisects_deterministically_and_reuses_successful_siblings():
    batch = [source_row(f"m{i}") for i in range(4)]
    assert [[row["memory_id"] for row in part] for part in split_batch(batch)] == [
        ["m0", "m1"],
        ["m2", "m3"],
    ]
    cache = {}
    provider_calls = []

    def invoke(rows):
        ids = tuple(row["memory_id"] for row in rows)
        if ids not in cache:
            provider_calls.append(ids)
            if ids == ("m0", "m1", "m2", "m3"):
                cache[ids] = IdentityLengthStop({"batch_id": "root", "finish_reason": "length"})
            elif ids == ("m0", "m1"):
                cache[ids] = (
                    [model_row("m0"), model_row("m1")],
                    {"batch_id": "left", "provider_calls": 1},
                )
            else:
                cache[ids] = (
                    [model_row("m2"), model_row("m3")],
                    {"batch_id": "right", "provider_calls": 1},
                )
        result = cache[ids]
        if isinstance(result, IdentityLengthStop):
            raise result
        rows_out, ledger = result
        return rows_out, {**ledger, "reused_frozen_result": True}

    first_rows, first_attempts = execute_with_bisection(batch, invoke)
    first_call_count = len(provider_calls)
    second_rows, second_attempts = execute_with_bisection(batch, invoke)
    assert len(first_rows) == len(second_rows) == 4
    assert [row["memory_id"] for row in first_rows] == ["m0", "m1", "m2", "m3"]
    assert [row["status"] for row in first_attempts] == ["OVERFLOW_PARENT", "COMPLETE", "COMPLETE"]
    assert len(provider_calls) == first_call_count == 3
    assert all(row.get("reused_frozen_result") for row in second_attempts[1:])


def test_single_record_length_stop_is_fatal():
    with pytest.raises(IdentityContractError, match="IRREDUCIBLE_IDENTITY_OUTPUT_FAILURE"):
        execute_with_bisection(
            [source_row("m1")],
            lambda _: (_ for _ in ()).throw(IdentityLengthStop({"finish_reason": "length"})),
        )


def test_fragmentation_is_heuristic_and_never_merges():
    identities = [
        {
            "memory_id": "m1",
            "revision_kind": "SINGLETON_STATE",
            "subject_key": "user",
            "attribute_key": "instagram_follower_count",
            "value_text": "500",
        },
        {
            "memory_id": "m2",
            "revision_kind": "SINGLETON_STATE",
            "subject_key": "user",
            "attribute_key": "instagram_followers_count",
            "value_text": "600",
        },
    ]
    source = [source_row("m1"), source_row("m2")]
    result = fragmentation_diagnostics(identities, source)
    assert result["flagged_pair_count"] == 1
    assert result["heuristic"] is True
    assert result["automatic_merges"] == 0
