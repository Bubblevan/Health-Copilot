from __future__ import annotations

import json

import pytest

from tools.research.memory import span_grounded_identity as identity


def _proposal(
    *,
    principal: str = "user",
    principal_witness: str | None = "The user",
    object_key: str | None = "instagram_account",
    object_witness: str | None = "Instagram",
    attribute: str = "follower_count",
    attribute_witness: str | None = "followers",
    value: str = "500",
    value_witness: str | None = "500",
    property_kind: str = "SINGLE_VALUE_STATE",
    change_cue: str | None = None,
) -> dict[str, object]:
    return {
        "principal_key": principal,
        "principal_witness_span": principal_witness,
        "object_key": object_key,
        "object_witness_span": object_witness,
        "attribute_key": attribute,
        "attribute_witness_span": attribute_witness,
        "value_text": value,
        "value_witness_span": value_witness,
        "property_kind": property_kind,
        "change_cue_span": change_cue,
    }


def _validate(proposals: dict[str, dict[str, object]], texts: dict[str, str]) -> list[dict[str, object]]:
    batch = [{"memory_id": mid, "proposition_text": text} for mid, text in texts.items()]
    sources = {
        mid: {"memory_id": mid, "scope_id": "scope-a", "proposition_text": text}
        for mid, text in texts.items()
    }
    raw = json.dumps({"identities": proposals}).encode()
    return identity.validate_response(raw, batch, sources)


def test_request_contains_only_id_and_proposition_not_eval_labels() -> None:
    request = identity.build_request(
        [
            {
                "memory_id": "m1",
                "proposition_text": "The user has reached 500 followers on Instagram.",
                "scope_id": "private-scope-not-for-model",
                "decision": "do-not-send",
            }
        ],
        model="local-qwen",
    )
    user_payload = json.loads(request["messages"][1]["content"])
    assert user_payload == {
        "memories": [
            {
                "memory_id": "m1",
                "proposition_text": "The user has reached 500 followers on Instagram.",
            }
        ]
    }
    assert request["temperature"] == 0
    assert request["seed"] == 42


def test_grounded_scalar_proposal_gets_harness_owned_exact_slot_key() -> None:
    records = _validate(
        {"m1": _proposal()},
        {"m1": "The user has reached 500 followers on Instagram."},
    )
    record = records[0]
    assert record["identity_status"] == "GROUNDED"
    assert record["slot_candidate"] is True
    assert record["slot_key"] == [
        "scope-a",
        "user",
        "instagram_account",
        "follower_count",
    ]


def test_equal_grounded_tuple_groups_values_without_deciding_revision() -> None:
    records = _validate(
        {
            "m1": _proposal(value="500", value_witness="500"),
            "m2": _proposal(value="600", value_witness="600"),
        },
        {
            "m1": "The user has reached 500 followers on Instagram.",
            "m2": "The user has reached 600 followers on Instagram.",
        },
    )
    groups = identity.exact_slot_groups(records)
    assert len(groups) == 1
    assert groups[0]["memory_ids"] == ["m1", "m2"]
    assert "revision" not in groups[0]


def test_value_number_must_be_supported_by_the_witness_and_source() -> None:
    records = _validate(
        {"m1": _proposal(value="600", value_witness="500")},
        {"m1": "The user has reached 500 followers on Instagram."},
    )
    assert records[0]["identity_status"] == "UNRESOLVED"
    assert any("value_text:NOT_GROUNDED" in reason for reason in records[0]["validation_reasons"])


def test_generic_reported_value_cannot_define_a_memory_slot() -> None:
    records = _validate(
        {
            "m1": _proposal(
                object_key=None,
                object_witness=None,
                attribute="reported_value",
                attribute_witness="value",
                value="0",
                value_witness="0",
            )
        },
        {"m1": "The user reported a value of 0 on 18/11/2020."},
    )
    assert records[0]["identity_status"] == "UNRESOLVED"
    assert records[0]["slot_candidate"] is False


def test_object_and_attribute_identity_separate_wallet_color_from_material() -> None:
    records = _validate(
        {
            "m1": _proposal(
                object_key="wallet",
                object_witness="wallet",
                attribute="color",
                attribute_witness="black leather wallet",
                value="black",
                value_witness="black",
            ),
            "m2": _proposal(
                object_key="wallet",
                object_witness="wallet",
                attribute="material",
                attribute_witness="leather material",
                value="leather",
                value_witness="leather",
            ),
        },
        {
            "m1": "The user decided to go with a black leather wallet.",
            "m2": "The user prefers leather material for a wallet.",
        },
    )
    assert all(record["identity_status"] == "GROUNDED" for record in records)
    assert identity.exact_slot_groups(records) == []


def test_event_and_set_state_are_vetoed_from_revision_slot_candidates() -> None:
    records = _validate(
        {
            "m1": _proposal(property_kind="EVENT"),
            "m2": _proposal(property_kind="MULTI_VALUE_STATE"),
        },
        {
            "m1": "The user has reached 500 followers on Instagram.",
            "m2": "The user has reached 500 followers on Instagram.",
        },
    )
    assert all(record["identity_status"] == "GROUNDED" for record in records)
    assert all(record["slot_candidate"] is False for record in records)
    assert identity.exact_slot_groups(records) == []


def test_witness_must_be_unique_exact_source_substring() -> None:
    records = _validate(
        {"m1": _proposal()},
        {"m1": "The user said The user reached 500 followers on Instagram."},
    )
    assert records[0]["identity_status"] == "UNRESOLVED"
    assert any(
        "principal_witness_span:WITNESS_NOT_UNIQUE_SOURCE_SUBSTRING" in reason
        for reason in records[0]["validation_reasons"]
    )


def test_duplicate_json_keys_are_rejected() -> None:
    raw = b'{"identities":{"m1":{},"m1":{}}}'
    with pytest.raises(identity.SpanIdentityError, match="duplicate_json_key"):
        identity.validate_response(
            raw,
            [{"memory_id": "m1", "proposition_text": "x"}],
            {"m1": {"scope_id": "scope-a", "proposition_text": "x"}},
        )
