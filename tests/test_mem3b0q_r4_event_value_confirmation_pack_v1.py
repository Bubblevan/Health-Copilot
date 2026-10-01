from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.research.memory import mem3b0q_r4 as frozen_r4
from tools.research.memory import mem3b0q_r4_candidate_builder_v1 as candidate_builder
from tools.research.memory import mem3b0q_r4_candidate_request_v2 as request_v2
from tools.research.memory import mem3b0q_r4_candidate_request_v3 as request_v3
from tools.research.memory import mem3b0q_r4_joint_binding_guard_v3 as guard_v3


PACK = json.loads(
    (
        Path(__file__).resolve().parents[1]
        / "docs/research/memory/mem3b0q_r4_event_value_confirmation_pack_v1.json"
    ).read_text(encoding="utf-8")
)


@pytest.mark.parametrize("case", PACK["cases"], ids=lambda row: row["case_id"])
def test_confirmation_case_spans_and_frozen_guard_contract(case: dict) -> None:
    source = case["proposition_text"]
    atom = case["expected_atoms"][0]
    candidates = candidate_builder.build_candidates(
        source, frozen_r4.ALIASES, source_id=case["case_id"]
    )["candidates"]
    candidate_ids = {}
    for field in ("owner", "object", "attribute"):
        matches = [
            row
            for row in candidates[field]
            if row["canonical_id"] == atom[f"{field}_id"]
            and row["source_span"].casefold() == atom[f"{field}_span"].casefold()
        ]
        occurrence = atom.get(f"{field}_occurrence", 0)
        assert occurrence < len(matches)
        candidate_ids[f"{field}_candidate_id"] = matches[occurrence]["candidate_id"]
    value = atom["value_span"]
    assert source.count(value) == 1
    assert case["temporal_expression_for_diagnostic_only"] in source
    assert value != case["temporal_expression_for_diagnostic_only"]
    request = request_v3.build_candidate_request(case, model="local-qwen")
    prompt_payload = json.loads(request["messages"][1]["content"])
    assert prompt_payload["proposition_text"] == source
    assert set(prompt_payload) == {"source_id", "proposition_text", "typed_candidates"}
    assert "expected_atoms" not in request["messages"][1]["content"]
    assert "temporal_expression_for_diagnostic_only" not in request["messages"][1]["content"]
    assert "completed the purchase" in source
    assert "bicycle" in source
    assert "purchase" in source
    raw_proposal = {
        "source_id": case["case_id"],
        "atoms": [{**candidate_ids, "value_span": value}],
        "abstention_reason": "NONE",
    }
    adapted = request_v2.materialize_cardinality(
        json.dumps(raw_proposal),
        {"source_id": case["case_id"], "proposition_text": source},
        scope_id=frozen_r4.FROZEN_SCOPE_ID,
    )
    validated = guard_v3.validate_joint_bound_candidate_proposal(
        adapted,
        {"source_id": case["case_id"], "proposition_text": source},
        scope_id=frozen_r4.FROZEN_SCOPE_ID,
    )
    assert validated["atoms"][0]["value_text"] == "completed the purchase"
