from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from eval.rag_e5.e5b3_qualification import (
    Q1_CHUNK_IDS,
    Q2_LONG_PASSAGE_IDS,
    build_qualification_fixtures,
    evaluate_qualification_case,
    select_smallest_passing_budget,
    summarize_budget,
)
from eval.rag_e5.e5b3_reader import (
    READER_STATE_PROJECTION_SPEC,
    READER_V2_SCHEMA,
    ReaderStateView,
    parse_reader_v2_output,
    render_reader_v2_prompt,
)


def _fixture_corpus() -> list[dict[str, str]]:
    return [
        {"chunk_id": chunk_id, "text": f"Approved public guideline passage {index}."}
        for index, chunk_id in enumerate(dict.fromkeys((*Q1_CHUNK_IDS, *Q2_LONG_PASSAGE_IDS)))
    ]


def _packet() -> dict[str, Any]:
    return {
        "summary_builder_version": "e5-longitudinal-state-v4",
        "age_bucket": "40-59",
        "condition_categories": ["hypertension"],
        "recent_measurement_trends": [{"metric": "body_weight", "trend": "rising"}],
        "available_exam_categories": ["blood_pressure"],
        "history_span_days": 512,
        "recent_event_count": 34,
        "user_id": "must-not-project",
        "teacher_expected_state": {"body_weight": "rising"},
        "future_event": {"metric": "body_weight", "trend": "falling"},
        "state_summary": "non-allowlisted prose must not be projected",
    }


def test_reader_v2_has_no_freeform_answer_or_top_level_citations() -> None:
    assert READER_V2_SCHEMA["required"] == ["state_facts", "guidance_facts"]
    assert set(READER_V2_SCHEMA["properties"]) == {"state_facts", "guidance_facts"}
    assert READER_V2_SCHEMA["additionalProperties"] is False
    assert parse_reader_v2_output(
        json.dumps({"state_facts": [], "guidance_facts": [], "answer": "extra"})
    ) == (None, False)


def test_state_field_and_trend_are_global_enums() -> None:
    state_item = READER_V2_SCHEMA["properties"]["state_facts"]["items"]["properties"]
    assert state_item["field"]["enum"] == ["body_weight", "body_mass_index"]
    assert state_item["value"]["enum"] == ["rising", "falling", "stable"]
    assert parse_reader_v2_output(
        json.dumps(
            {
                "state_facts": [{"field": "body_weight_trend", "value": "rising"}],
                "guidance_facts": [],
            }
        )
    ) == (None, False)


def test_guidance_citations_are_claim_local_and_bounded() -> None:
    fact = READER_V2_SCHEMA["properties"]["guidance_facts"]["items"]
    assert fact["required"] == ["statement", "citations"]
    assert fact["properties"]["citations"]["maxItems"] == 4
    parsed, valid = parse_reader_v2_output(
        json.dumps(
            {
                "state_facts": [],
                "guidance_facts": [
                    {"statement": "A supported claim.", "citations": ["chunk-1"]}
                ],
            }
        )
    )
    assert valid and parsed is not None
    assert parsed["guidance_facts"][0]["citations"] == ["chunk-1"]


def test_reader_state_view_is_projection_only() -> None:
    view = ReaderStateView.from_packet(_packet()).to_dict()
    assert set(view) == set(READER_STATE_PROJECTION_SPEC["fields"])
    assert view["recent_measurement_trends"] == [
        {"metric": "body_weight", "trend": "rising"}
    ]
    assert "user_id" not in view
    assert "teacher_expected_state" not in view
    assert "future_event" not in view
    assert "state_summary" not in view


def test_qualification_fixtures_have_independent_balanced_strata_and_q2_passages() -> None:
    fixtures = build_qualification_fixtures(_fixture_corpus())
    assert len(fixtures) == 24
    assert [sum(row["stratum"] == group for row in fixtures) for group in (
        "Q0_STATE_ONLY",
        "Q1_EVIDENCE_ONLY",
        "Q2_STATE_AND_EVIDENCE",
    )] == [8, 8, 8]
    assert all(len(row["passages"]) == 5 for row in fixtures[-8:])
    assert {row["chunk_id"] for row in fixtures[-8:][0]["passages"]} == set(
        Q2_LONG_PASSAGE_IDS
    )
    state_pairs = {
        (fact["field"], fact["value"])
        for row in (*fixtures[:8], *fixtures[-8:])
        for fact in row["expected_state_facts"]
    }
    assert state_pairs == {
        (metric, trend)
        for metric in ("body_weight", "body_mass_index")
        for trend in ("rising", "falling", "stable")
    }


def test_qualification_uses_no_b2_outputs_or_202608(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    original_open = Path.open

    def guarded_open(path: Path, *args: Any, **kwargs: Any):
        normalized = str(path).casefold()
        assert "e5b2" not in normalized
        assert "202608" not in normalized
        assert path.name != "teacher_cases.jsonl"
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    fixtures = build_qualification_fixtures(_fixture_corpus())
    prompt = render_reader_v2_prompt(
        question=fixtures[-1]["question"],
        state_view=fixtures[-1]["state_view"],
        passages=fixtures[-1]["passages"],
    )
    assert len(fixtures) == 24
    assert "teacher_expected_state" not in prompt
    assert "202608" not in prompt


def test_budget_ladder_selects_smallest_passing_budget() -> None:
    summaries = [
        {"output_budget": 256, "passed": False},
        {"output_budget": 384, "passed": True},
        {"output_budget": 512, "passed": True},
    ]
    assert select_smallest_passing_budget(summaries) == 384
    assert select_smallest_passing_budget(summaries[:1]) is None


def test_qualification_gate_rejects_truncation_alias_and_invented_citations() -> None:
    fixture = {
        "stratum": "Q2_STATE_AND_EVIDENCE",
        "expected_state_facts": [{"field": "body_weight", "value": "rising"}],
        "passages": [{"chunk_id": "approved-1"}],
        "evidence_required": True,
    }
    result = evaluate_qualification_case(
        fixture=fixture,
        text=json.dumps(
            {
                "state_facts": [{"field": "body_weight_trend", "value": "increasing"}],
                "guidance_facts": [
                    {"statement": "A claim.", "citations": ["invented-1"]}
                ],
            }
        ),
        finish_reason="length",
        output_tokens=256,
        output_cap=256,
    )
    assert result["valid_json"] is False
    assert result["length_failure"] is True
    assert result["citation_contract_match"] is False


def test_budget_summary_enforces_exact_qualification_gates() -> None:
    rows = []
    for index in range(24):
        stratum = ("Q0_STATE_ONLY", "Q1_EVIDENCE_ONLY", "Q2_STATE_AND_EVIDENCE")[index // 8]
        evidence_required = stratum != "Q0_STATE_ONLY"
        rows.append(
            {
                "fixture_id": f"fixture-{index}",
                "stratum": stratum,
                "valid_json": True,
                "length_failure": False,
                "state_contract_match": stratum != "Q1_EVIDENCE_ONLY",
                "evidence_required": evidence_required,
                "citation_contract_match": True,
                "invented_citations": [],
                "output_tokens": 100,
            }
        )
    summary = summarize_budget(budget=256, case_results=rows)
    assert summary["passed"] is True
    assert summary["output_tokens"]["p95_nearest_rank"] == 100
    assert all(summary["gates"].values())
