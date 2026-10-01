from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

from tools.research.memory import mem3b0q_r4_candidate_request_v2 as request_v2
from tools.research.memory import mem3b0q_r4_candidate_request_v3 as request_v3


ROOT = Path(__file__).resolve().parents[1]
PACK = json.loads(
    (ROOT / "docs/research/memory/mem3b0q_r4_candidate_control_pack_v1.json")
    .read_text(encoding="utf-8")
)


def test_event_prompt_refinement_does_not_change_schema_or_model_authority() -> None:
    case = next(row for row in PACK["cases"] if row["case_id"] == "R4C-04")
    base = request_v2.build_candidate_request(case, model="local-qwen")
    refined = request_v3.build_candidate_request(case, model="local-qwen")
    base_schema = base["response_format"]["json_schema"]["schema"]
    refined_schema = refined["response_format"]["json_schema"]["schema"]

    Draft202012Validator.check_schema(refined_schema)
    assert base_schema == refined_schema
    assert "not a phrase that only says when" in refined["messages"][0]["content"]
    assert "do not invent one" in refined["messages"][0]["content"]
    atom = refined_schema["properties"]["atoms"]["items"]
    assert "cardinality_proposal" not in atom["properties"]
