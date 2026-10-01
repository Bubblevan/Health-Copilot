from __future__ import annotations

import json

from tools.research.memory import run_mem3b0q_r4_factorized_event_slot_confirmation_v1 as runner


def test_model_request_contains_only_object_and_attribute_slots() -> None:
    pack, requests, scoring = runner._build_inputs()

    assert tuple(row["case_id"] for row in pack["cases"]) == runner.CASE_IDS
    assert tuple(requests) == runner.CASE_IDS
    assert tuple(scoring) == runner.CASE_IDS
    for case_id, request_row in runner.MODEL_REQUESTS.items():
        schema = request_row["schema"]
        slot_properties = schema["properties"]["slots"]["items"]["properties"]
        assert set(slot_properties) == {"object_candidate_id", "attribute_candidate_id"}
        user_payload = json.loads(request_row["request"]["messages"][1]["content"])
        assert set(user_payload["typed_candidates"]) == {"object", "attribute"}
        assert "expected_atoms" not in request_row["request"]["messages"][1]["content"]
        assert "temporal_expression_for_diagnostic_only" not in request_row["request"]["messages"][1]["content"]


def test_response_adapter_projects_owner_and_value_without_model_fields() -> None:
    runner._build_inputs()
    case_id = runner.CASE_IDS[0]
    candidates = json.loads(runner.MODEL_REQUESTS[case_id]["request"]["messages"][1]["content"])["typed_candidates"]
    proposal = {
        "source_id": case_id,
        "slots": [{
            "object_candidate_id": candidates["object"][0]["candidate_id"],
            "attribute_candidate_id": candidates["attribute"][0]["candidate_id"],
        }],
        "abstention_reason": "NONE",
    }
    response = json.dumps({"choices": [{"message": {"content": json.dumps(proposal)}}]}).encode()
    runner.CURRENT_CASE_ID = case_id
    try:
        _, normalized = runner._project_model_response(response)
    finally:
        runner.CURRENT_CASE_ID = None

    result = json.loads(normalized)
    assert set(result["atoms"][0]) == {
        "owner_candidate_id", "object_candidate_id", "attribute_candidate_id", "value_span"
    }
    assert result["atoms"][0]["owner_candidate_id"] == "owner:SELF:56:58"
    assert result["atoms"][0]["value_span"] == "completed the purchase"
    assert runner.CURRENT_MODEL_AUDIT[case_id]["model_schema_validation"] == "PASS"
