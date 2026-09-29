"""Independent, non-retrieval fixtures and gates for Reader V2 qualification."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from statistics import median
from typing import Any

from eval.rag_e5.e5b3_reader import ReaderStateView, parse_reader_v2_output

Q1_CHUNK_IDS = (
    "who-guideline-4f47a8d684098dc3c22b7ad5",
    "who-guideline-33511bbfac7ce4586cb7aadc",
    "who-guideline-e6e84b1a717d8c85db5f0454",
    "who-guideline-9c2ad627f2731d22c33e010d",
    "who-guideline-5f9f38ca0abb571b9420ada2",
    "who-guideline-a88b56abfce6c49800316477",
    "who-guideline-7f385b69c8e66434d2fefc4c",
    "who-guideline-5ca2f3d427b204da9c293c0f",
)
Q2_LONG_PASSAGE_IDS = Q1_CHUNK_IDS[:5]
TREND_VALUES = ("rising", "falling", "stable")
METRIC_FIELDS = ("body_weight", "body_mass_index")


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_qualification_fixtures(
    approved_chunks: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build 8 state-only, 8 public-evidence, and 8 mixed synthetic fixtures."""
    chunks_by_id = {
        row.get("chunk_id"): row
        for row in approved_chunks
        if isinstance(row.get("chunk_id"), str) and isinstance(row.get("text"), str)
    }
    required_ids = (*Q1_CHUNK_IDS, *Q2_LONG_PASSAGE_IDS)
    missing = sorted(set(required_ids) - set(chunks_by_id))
    if missing:
        raise ValueError(f"approved qualification passages missing from corpus: {missing}")
    if any(not chunks_by_id[chunk_id]["text"].strip() for chunk_id in required_ids):
        raise ValueError("qualification passage cannot be empty")

    fixtures: list[dict[str, Any]] = []
    state_patterns = [
        ("body_weight", "rising"),
        ("body_weight", "falling"),
        ("body_weight", "stable"),
        ("body_mass_index", "rising"),
        ("body_mass_index", "falling"),
        ("body_mass_index", "stable"),
        ("body_weight", "stable"),
        ("body_mass_index", "rising"),
    ]
    for index, (metric, trend) in enumerate(state_patterns, start=1):
        view = ReaderStateView(
            age_bucket="40-59",
            condition_categories=("hypertension",),
            recent_measurement_trends=((metric, trend),),
            available_exam_categories=("blood_pressure",),
            history_span_days=512,
            recent_event_count=34,
        )
        fixtures.append(
            {
                "fixture_id": f"Q0-{index:02d}",
                "stratum": "Q0_STATE_ONLY",
                "question": "Describe only the recorded measurement trend, using its exact metric name.",
                "state_view": view.to_dict(),
                "expected_state_facts": [{"field": metric, "value": trend}],
                "passages": [],
                "evidence_required": False,
            }
        )

    for index, chunk_id in enumerate(Q1_CHUNK_IDS, start=1):
        fixtures.append(
            {
                "fixture_id": f"Q1-{index:02d}",
                "stratum": "Q1_EVIDENCE_ONLY",
                "question": "State one concise general recommendation supported by the supplied public guideline excerpt.",
                "state_view": None,
                "expected_state_facts": [],
                "passages": [dict(chunks_by_id[chunk_id])],
                "evidence_required": True,
            }
        )

    long_passages = [dict(chunks_by_id[chunk_id]) for chunk_id in Q2_LONG_PASSAGE_IDS]
    for index, (metric, trend) in enumerate(state_patterns, start=1):
        view = ReaderStateView(
            age_bucket="60-74",
            condition_categories=("hypertension", "overweight"),
            recent_measurement_trends=((metric, trend),),
            available_exam_categories=("blood_pressure", "body_composition"),
            history_span_days=730,
            recent_event_count=52,
        )
        fixtures.append(
            {
                "fixture_id": f"Q2-{index:02d}",
                "stratum": "Q2_STATE_AND_EVIDENCE",
                "question": "Describe the recorded measurement trend and state one general recommendation supported by the passages.",
                "state_view": view.to_dict(),
                "expected_state_facts": [{"field": metric, "value": trend}],
                "passages": [dict(row) for row in long_passages],
                "evidence_required": True,
            }
        )
    return fixtures


def fixture_set_sha256(fixtures: Sequence[Mapping[str, Any]]) -> str:
    canonical: list[dict[str, Any]] = []
    for fixture in fixtures:
        canonical.append(
            {
                "fixture_id": fixture["fixture_id"],
                "stratum": fixture["stratum"],
                "question": fixture["question"],
                "state_view": fixture["state_view"],
                "expected_state_facts": fixture["expected_state_facts"],
                "evidence_required": fixture["evidence_required"],
                "passages": [
                    {
                        "chunk_id": row["chunk_id"],
                        "text_sha256": hashlib.sha256(row["text"].encode("utf-8")).hexdigest(),
                    }
                    for row in fixture["passages"]
                ],
            }
        )
    return canonical_sha256(canonical)


def evaluate_qualification_case(
    *,
    fixture: Mapping[str, Any],
    text: str,
    finish_reason: str | None,
    output_tokens: int | None,
    output_cap: int,
) -> dict[str, Any]:
    parsed, valid = parse_reader_v2_output(text)
    supplied_ids = {str(row["chunk_id"]) for row in fixture["passages"]}
    if parsed is None:
        state_match = False
        cited_ids: set[str] = set()
    else:
        expected_state = sorted(
            (row["field"], row["value"]) for row in fixture["expected_state_facts"]
        )
        actual_state = sorted(
            (row["field"], row["value"]) for row in parsed["state_facts"]
        )
        state_match = actual_state == expected_state
        cited_ids = {
            citation
            for fact in parsed["guidance_facts"]
            for citation in fact["citations"]
        }
    invented = sorted(cited_ids - supplied_ids)
    valid_citations = sorted(cited_ids & supplied_ids)
    return {
        "valid_json": valid,
        "finish_reason": finish_reason,
        "length_failure": finish_reason == "length",
        "state_contract_match": state_match,
        "evidence_required": bool(fixture["evidence_required"]),
        "invented_citations": invented,
        "valid_supplied_citations": valid_citations,
        "citation_contract_match": not invented
        and (not fixture["evidence_required"] or bool(valid_citations)),
        "output_tokens": output_tokens,
        "output_cap": output_cap,
        "parsed_output": parsed,
    }


def summarize_budget(
    *, budget: int, case_results: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    results = list(case_results)
    token_counts = [
        value for row in results if isinstance((value := row.get("output_tokens")), int)
    ]
    p95 = _nearest_rank_percentile(token_counts, 0.95) if token_counts else None
    q0q2 = [row for row in results if row["stratum"] in {"Q0_STATE_ONLY", "Q2_STATE_AND_EVIDENCE"}]
    evidence = [row for row in results if row["evidence_required"]]
    valid_count = sum(bool(row["valid_json"]) for row in results)
    length_count = sum(bool(row["length_failure"]) for row in results)
    state_match_count = sum(bool(row["state_contract_match"]) for row in q0q2)
    citation_match_count = sum(bool(row["citation_contract_match"]) for row in evidence)
    invented_count = sum(len(row["invented_citations"]) for row in evidence)
    margin_pass = p95 is not None and p95 <= 0.75 * budget
    gates = {
        "valid_json_24_of_24": len(results) == 24 and valid_count == 24,
        "zero_finish_reason_length": length_count == 0,
        "q0_q2_canonical_state_100_percent": len(q0q2) == 16 and state_match_count == 16,
        "q1_q2_citation_contract": (
            len(evidence) == 16 and invented_count == 0 and citation_match_count == 16
        ),
        "p95_output_tokens_at_most_75_percent_cap": margin_pass,
    }
    return {
        "output_budget": budget,
        "cases": len(results),
        "valid_json": valid_count,
        "finish_reason_length": length_count,
        "q0_q2_state_contract_match": {"matched": state_match_count, "total": len(q0q2)},
        "q1_q2_citation_contract_match": {
            "matched": citation_match_count,
            "total": len(evidence),
            "invented_citation_count": invented_count,
        },
        "output_tokens": {
            "count": len(token_counts),
            "p50": median(token_counts) if token_counts else None,
            "p95_nearest_rank": p95,
            "max": max(token_counts) if token_counts else None,
            "p95_cap": 0.75 * budget,
        },
        "gates": gates,
        "passed": all(gates.values()),
        "case_results": results,
    }


def select_smallest_passing_budget(
    summaries: Sequence[Mapping[str, Any]], candidate_budgets: Sequence[int] = (256, 384, 512)
) -> int | None:
    """Return the first passing budget in the preregistered ascending ladder."""
    by_budget = {row.get("output_budget"): row for row in summaries}
    for budget in candidate_budgets:
        row = by_budget.get(budget)
        if row is not None and row.get("passed") is True:
            return budget
    return None


def _nearest_rank_percentile(values: Sequence[int], percentile: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


__all__ = [
    "METRIC_FIELDS",
    "Q1_CHUNK_IDS",
    "Q2_LONG_PASSAGE_IDS",
    "TREND_VALUES",
    "build_qualification_fixtures",
    "canonical_sha256",
    "evaluate_qualification_case",
    "fixture_set_sha256",
    "select_smallest_passing_budget",
    "summarize_budget",
]
