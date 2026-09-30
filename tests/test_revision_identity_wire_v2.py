from __future__ import annotations

import json

import pytest

from tools.research.memory.revision_identity_wire_v2 import (
    GlobalIdentityFailure,
    KeyedIdentityContractError,
    LocalBatchFailure,
    batch_id,
    build_request,
    is_candidate_singleton,
    recover_batch,
    require_exact_terminal_coverage,
    response_schema,
    validate_response,
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


def model_payload(value: str = "apples", **overrides):
    return {
        "revision_kind": "SINGLETON_STATE",
        "subject_key": "user",
        "attribute_key": "favorite-food",
        "value_text": value,
        **overrides,
    }


def keyed_bytes(rows: dict[str, dict[str, str]]) -> bytes:
    return json.dumps({"identities": rows}).encode("utf-8")


def test_keyed_schema_requires_exact_dynamic_id_set_and_no_array():
    schema = response_schema(["m1", "m2"])
    identities = schema["properties"]["identities"]
    assert identities["type"] == "object"
    assert identities["required"] == ["m1", "m2"]
    assert identities["additionalProperties"] is False
    assert set(identities["properties"]) == {"m1", "m2"}
    assert "memory_id" not in identities["properties"]["m1"]["properties"]


def test_request_keeps_four_memory_fields_and_uses_keyed_map_schema():
    request = build_request(
        [source_row("m1"), source_row("m2", "User likes pears")],
        model="health-memory-qwen3-8b",
        max_tokens=8192,
    )
    memories = json.loads(request["messages"][1]["content"])["memories"]
    assert all(
        set(row) == {"memory_id", "proposition_text", "source_authority", "scope_id"}
        for row in memories
    )
    assert "must-not-enter-request" not in request["messages"][1]["content"]
    assert request["temperature"] == 0
    assert request["seed"] == 42
    assert request["max_tokens"] == 8192


def test_validator_returns_exact_coverage_in_canonical_input_order():
    batch = [source_row("m2"), source_row("m1")]
    raw = keyed_bytes({"m1": model_payload("pears"), "m2": model_payload("apples")})
    records = validate_response(raw, batch)
    assert [row["memory_id"] for row in records] == ["m2", "m1"]
    assert records[0]["value_text"] == "apples"
    assert records[0]["identity_origin"] == "MODEL_VALIDATED"
    require_exact_terminal_coverage(["m2", "m1"], records)


@pytest.mark.parametrize(
    ("raw", "code"),
    [
        (b'{"identities":{"m1":{},"m1":{}}}', "DUPLICATE_JSON_OBJECT_KEY"),
        (keyed_bytes({"m1": model_payload()}), "MISSING_IDENTITY_KEY"),
        (
            keyed_bytes({"m1": model_payload(), "m2": model_payload(), "m3": model_payload()}),
            "UNEXPECTED_IDENTITY_KEY",
        ),
        (b"not json", "MALFORMED_ASSISTANT_JSON"),
    ],
)
def test_validator_rejects_duplicate_raw_keys_and_id_set_errors(raw, code):
    batch = [source_row("m1"), source_row("m2")]
    with pytest.raises(KeyedIdentityContractError) as exc:
        validate_response(raw, batch)
    assert exc.value.code == code


@pytest.mark.parametrize(
    ("override", "code"),
    [
        ({"revision_kind": "UPDATE"}, "ILLEGAL_REVISION_KIND"),
        ({"subject_key": "user/account"}, "INVALID_SUBJECT_KEY"),
        ({"attribute_key": "old_address"}, "INVALID_ATTRIBUTE_KEY"),
        ({"extra": "no"}, "UNEXPECTED_IDENTITY_FIELD"),
        ({"value_text": "  "}, "EMPTY_VALUE_TEXT"),
    ],
)
def test_validator_rejects_invalid_identity_payloads(override, code):
    with pytest.raises(KeyedIdentityContractError) as exc:
        validate_response(keyed_bytes({"m1": model_payload(**override)}), [source_row("m1")])
    assert exc.value.code == code


def _valid_records(batch):
    return [
        {
            "memory_id": row["memory_id"],
            **model_payload(row["proposition_text"].split()[-1]),
            "identity_origin": "MODEL_VALIDATED",
            "fallback_reason": None,
        }
        for row in batch
    ]


def test_local_batch_failure_subdivides_left_then_right_and_reuses_successful_sibling():
    batch = [source_row(f"m{i}") for i in range(4)]
    cache = {}
    provider_batches = []

    def invoke(rows):
        key = tuple(row["memory_id"] for row in rows)
        if key not in cache:
            provider_batches.append(key)
            if key == ("m0", "m1", "m2", "m3"):
                cache[key] = LocalBatchFailure("MALFORMED_ASSISTANT_JSON", {"calls": 1})
            elif key == ("m0", "m1"):
                cache[key] = (_valid_records(rows), {"provider_calls": 1})
            elif key == ("m2", "m3"):
                cache[key] = LocalBatchFailure("MISSING_IDENTITY_KEY", {"calls": 1})
            else:
                cache[key] = (_valid_records(rows), {"provider_calls": 1})
        result = cache[key]
        if isinstance(result, LocalBatchFailure):
            raise result
        return result

    records, calls, lineage = recover_batch(batch, invoke)
    first_call_count = len(provider_batches)
    replay_records, replay_calls, replay_lineage = recover_batch(batch, invoke)
    assert [row["memory_id"] for row in records] == [f"m{i}" for i in range(4)]
    assert [row["memory_id"] for row in replay_records] == [f"m{i}" for i in range(4)]
    assert first_call_count == len(provider_batches) == 5
    assert [row["outcome"] for row in lineage] == [
        "LOCAL_FAILURE_SUBDIVIDED",
        "MODEL_VALIDATED",
        "LOCAL_FAILURE_SUBDIVIDED",
        "MODEL_VALIDATED",
        "MODEL_VALIDATED",
    ]
    assert len(calls) == len(replay_calls) == len(lineage) == len(replay_lineage) == 5
    assert lineage[0]["child_batch_ids"] == [batch_id(batch[:2]), batch_id(batch[2:])]


@pytest.mark.parametrize("reason", ["ILLEGAL_REVISION_KIND", "COMPLETION_TRUNCATED"])
def test_single_item_local_failure_becomes_unknown_fallback(reason):
    row = source_row("m1", "User works at Example Health")

    def invoke(batch):
        raise LocalBatchFailure(reason, {"provider_calls": 1})

    records, calls, lineage = recover_batch([row], invoke)
    fallback = records[0]
    assert fallback == {
        "memory_id": "m1",
        "revision_kind": "UNKNOWN",
        "subject_key": "unknown",
        "attribute_key": "unknown",
        "value_text": "User works at Example Health",
        "identity_origin": "HARNESS_UNKNOWN_FALLBACK",
        "fallback_reason": reason,
    }
    assert not is_candidate_singleton(fallback)
    assert calls[0]["terminal_identities"] is False
    assert lineage[0]["outcome"] == "HARNESS_UNKNOWN_FALLBACK"


def test_global_failure_is_not_converted_to_unknown_fallback():
    row = source_row("m1")
    with pytest.raises(GlobalIdentityFailure, match="frozen_flatprop_sha_mismatch"):
        recover_batch(
            [row],
            lambda _: (_ for _ in ()).throw(
                GlobalIdentityFailure("frozen_flatprop_sha_mismatch")
            ),
        )


def test_illegal_revision_kind_in_multi_item_batch_is_recovered_by_subdivision():
    batch = [source_row("m1"), source_row("m2")]
    calls_seen = []

    def invoke(rows):
        ids = [row["memory_id"] for row in rows]
        calls_seen.append(ids)
        if len(rows) == 2:
            raise LocalBatchFailure("ILLEGAL_REVISION_KIND", {"provider_calls": 1})
        return _valid_records(rows), {"provider_calls": 1}

    records, calls, lineage = recover_batch(batch, invoke)
    assert calls_seen == [["m1", "m2"], ["m1"], ["m2"]]
    assert [row["identity_origin"] for row in records] == [
        "MODEL_VALIDATED",
        "MODEL_VALIDATED",
    ]
    assert lineage[0]["outcome"] == "LOCAL_FAILURE_SUBDIVIDED"
    assert len(calls) == 3


def test_unknown_fallback_cannot_enter_candidate_singleton_groups():
    model_singleton = {
        "identity_origin": "MODEL_VALIDATED",
        "revision_kind": "SINGLETON_STATE",
    }
    model_unknown = {"identity_origin": "MODEL_VALIDATED", "revision_kind": "UNKNOWN"}
    fallback_singleton = {
        "identity_origin": "HARNESS_UNKNOWN_FALLBACK",
        "revision_kind": "SINGLETON_STATE",
    }
    fallback_unknown = {
        "identity_origin": "HARNESS_UNKNOWN_FALLBACK",
        "revision_kind": "UNKNOWN",
    }
    assert is_candidate_singleton(model_singleton)
    assert not is_candidate_singleton(model_unknown)
    assert not is_candidate_singleton(fallback_singleton)
    assert not is_candidate_singleton(fallback_unknown)


def test_terminal_coverage_requires_exactly_once_identity_for_every_input():
    records = _valid_records([source_row("m1"), source_row("m2")])
    require_exact_terminal_coverage(["m1", "m2"], records)
    with pytest.raises(GlobalIdentityFailure, match="not_unique"):
        require_exact_terminal_coverage(["m1", "m2"], [records[0], records[0]])
    with pytest.raises(GlobalIdentityFailure, match="incomplete"):
        require_exact_terminal_coverage(["m1", "m2"], records[:1])
    with pytest.raises(GlobalIdentityFailure, match="not_unique"):
        require_exact_terminal_coverage(["m1", "m1"], records[:1])
