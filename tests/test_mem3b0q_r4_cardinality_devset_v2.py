from __future__ import annotations

import json

from tools.research.memory import mem3b0q_r4_candidate_builder_v1 as candidate_builder
from tools.research.memory import mem3b0q_r4_candidate_request_v2 as request_v2
from tools.research.memory import run_mem3b0q_r4_cardinality_devset_v2 as runner


def test_six_new_requests_are_frozen_local_and_cardinality_free() -> None:
    pack, requests, _ = runner._build_requests()

    assert tuple(row["case_id"] for row in pack["cases"]) == runner.CASE_IDS
    assert tuple(requests) == runner.CASE_IDS
    for record in requests.values():
        request = record["request"]
        atom_schema = request["response_format"]["json_schema"]["schema"]["properties"]["atoms"]["items"]
        payload = json.loads(request["messages"][1]["content"])
        assert request["model"] == runner.frozen_gate.MODEL_PATH
        assert request["max_tokens"] == 256
        assert "cardinality_proposal" not in atom_schema["properties"]
        assert set(payload) == {"source_id", "proposition_text", "typed_candidates"}
        assert "expected_atoms" not in record["body"].decode("utf-8")


def test_v2_response_derives_cardinality_then_reuses_frozen_atomwise_guards() -> None:
    pack, requests, v1_requests = runner._build_requests()
    case = next(row for row in pack["cases"] if row["case_id"] == "R4C-03")
    source = case["proposition_text"]
    candidates = candidate_builder.build_candidates(
        source, candidate_builder.frozen_r4.ALIASES
    )["candidates"]

    def candidate(field: str, canonical_id: str, occurrence: int = 0) -> str:
        matches = [row for row in candidates[field] if row["canonical_id"] == canonical_id]
        return matches[occurrence]["candidate_id"]

    raw = {
        "source_id": "R4C-03",
        "atoms": [
            {
                "owner_candidate_id": candidate("owner", "SELF"),
                "object_candidate_id": candidate("object", "FAVORITE_FRUIT_SET"),
                "attribute_candidate_id": candidate("attribute", "MEMBERSHIP"),
                "value_span": "pears",
            },
            {
                "owner_candidate_id": candidate("owner", "SELF", 1),
                "object_candidate_id": candidate("object", "WEDDING_TRIP_PLAN"),
                "attribute_candidate_id": candidate("attribute", "DESTINATION"),
                "value_span": "Lisbon",
            },
        ],
        "abstention_reason": "NONE",
    }
    scored = runner._score_response(
        case,
        json.dumps(raw),
        v2_request=requests["R4C-03"]["request"],
        v1_request=v1_requests["R4C-03"]["request"],
        response_sha256="0" * 64,
        usage={"prompt_tokens": 1, "completion_tokens": 1},
        latency_ms=1.0,
        finish_reason="stop",
    )

    assert scored["cardinality_materialization"] == "PASS"
    assert scored["cardinality_authority"] == "HARNESS_FROZEN_SLOT_POLICY"
    assert scored["status"] == "QUALITY_PASS"
    assert scored["matched_expected_atoms"] == 2


def test_truncation_is_quality_failure_not_infrastructure_failure() -> None:
    pack, requests, v1_requests = runner._build_requests()
    case = next(row for row in pack["cases"] if row["case_id"] == "R4C-02")
    scored = runner._score_response(
        case,
        '{"source_id":"R4C-02","atoms":[',
        v2_request=requests["R4C-02"]["request"],
        v1_request=v1_requests["R4C-02"]["request"],
        response_sha256="0" * 64,
        usage={"prompt_tokens": 1, "completion_tokens": 256},
        latency_ms=1.0,
        finish_reason="length",
    )

    assert scored["status"] == "QUALITY_FAILURE"
    assert scored["generation_truncated"] is True
    assert scored["failure"] == "generation_truncated_at_token_cap"


def test_lock_payload_pins_new_v2_schema_without_touching_runtime() -> None:
    lock = runner.build_lock_payload()

    assert lock["status"] == "FROZEN"
    assert lock["case_ids"] == list(runner.CASE_IDS)
    assert lock["max_completion_posts"] == 6
    assert lock["request_controls"]["max_tokens"] == 256
    assert len(lock["requests"]) == 6
    assert lock["hosted_api"] == "NONE"
