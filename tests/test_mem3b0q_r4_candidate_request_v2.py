from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from tools.research.memory import mem3b0q_r4 as frozen_r4
from tools.research.memory import mem3b0q_r4_candidate_builder_v1 as candidate_builder
from tools.research.memory import mem3b0q_r4_candidate_request_v2 as request_v2


ROOT = Path(__file__).resolve().parents[1]
PACK = json.loads(
    (ROOT / "docs/research/memory/mem3b0q_r4_candidate_control_pack_v1.json")
    .read_text(encoding="utf-8")
)


def _case(case_id: str) -> dict:
    return next(row for row in PACK["cases"] if row["case_id"] == case_id)


def _candidate_id(
    field: str, canonical_id: str, source: str, occurrence: int = 0
) -> str:
    rows = candidate_builder.build_candidates(
        source, frozen_r4.ALIASES
    )["candidates"][field]
    matches = [row for row in rows if row["canonical_id"] == canonical_id]
    return matches[occurrence]["candidate_id"]


def test_schema_excludes_cardinality_and_request_remains_case_bounded() -> None:
    case = _case("R4C-03")
    request = request_v2.build_candidate_request(case, model="frozen-local-model")
    schema = request["response_format"]["json_schema"]["schema"]
    atom = schema["properties"]["atoms"]["items"]

    Draft202012Validator.check_schema(schema)
    assert "cardinality_proposal" not in atom["properties"]
    assert "cardinality_proposal" not in atom["required"]
    assert "Do not output cardinality" in request["messages"][0]["content"]
    assert request["max_tokens"] == 256
    assert "expected_atoms" not in request["messages"][1]["content"]


def test_harness_derives_multivalue_and_single_value_from_frozen_slot_policy() -> None:
    case = _case("R4C-03")
    source = case["proposition_text"]
    raw = {
        "source_id": case["case_id"],
        "atoms": [
            {
                "owner_candidate_id": _candidate_id("owner", "SELF", source),
                "object_candidate_id": _candidate_id(
                    "object", "FAVORITE_FRUIT_SET", source
                ),
                "attribute_candidate_id": _candidate_id(
                    "attribute", "MEMBERSHIP", source
                ),
                "value_span": "pears",
            },
            {
                "owner_candidate_id": _candidate_id("owner", "SELF", source, 1),
                "object_candidate_id": _candidate_id(
                    "object", "WEDDING_TRIP_PLAN", source
                ),
                "attribute_candidate_id": _candidate_id(
                    "attribute", "DESTINATION", source
                ),
                "value_span": "Lisbon",
            },
        ],
        "abstention_reason": "NONE",
    }
    adapted = json.loads(
        request_v2.materialize_cardinality(
            json.dumps(raw),
            {"source_id": case["case_id"], "proposition_text": source},
            scope_id=frozen_r4.FROZEN_SCOPE_ID,
        )
    )

    assert [row["cardinality_proposal"] for row in adapted["atoms"]] == [
        "MULTI_VALUE_CONCURRENT",
        "SINGLE_VALUE_AT_A_TIME",
    ]


def test_unknown_policy_pair_fails_closed(monkeypatch) -> None:
    case = _case("R4C-03")
    source = case["proposition_text"]
    raw = {
        "source_id": case["case_id"],
        "atoms": [
            {
                "owner_candidate_id": _candidate_id("owner", "SELF", source),
                "object_candidate_id": _candidate_id(
                    "object", "WEDDING_TRIP_PLAN", source
                ),
                "attribute_candidate_id": _candidate_id(
                    "attribute", "DESTINATION", source
                ),
                "value_span": "Lisbon",
            }
        ],
        "abstention_reason": "NONE",
    }
    monkeypatch.setattr(request_v2.joint_guard, "CARDINALITY_POLICY", {})

    with pytest.raises(ValueError, match="typed_slot_policy_unknown"):
        request_v2.materialize_cardinality(
            json.dumps(raw),
            {"source_id": case["case_id"], "proposition_text": source},
            scope_id=frozen_r4.FROZEN_SCOPE_ID,
        )


def test_model_cannot_supply_cardinality_field() -> None:
    case = _case("R4C-03")
    raw = {
        "source_id": case["case_id"],
        "atoms": [{"cardinality_proposal": "SINGLE_VALUE_AT_A_TIME"}],
        "abstention_reason": "NONE",
    }

    with pytest.raises(ValueError, match="keys_mismatch"):
        request_v2.materialize_cardinality(
            json.dumps(raw),
            {
                "source_id": case["case_id"],
                "proposition_text": case["proposition_text"],
            },
            scope_id=frozen_r4.FROZEN_SCOPE_ID,
        )
