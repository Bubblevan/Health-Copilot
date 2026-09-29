from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.research.memory import flat_proposition_writer_v2 as v2
from tools.research.memory import flatprop_writer_v4_chunked as v4
from tools.research.memory.flatprop_chunking import (
    aggregate_chunk_propositions,
    plan_session_chunks,
)


def _catalog(turn_costs: list[list[int]]) -> list[dict[str, object]]:
    catalog = []
    for turn_index, costs in enumerate(turn_costs):
        for span_index, cost in enumerate(costs):
            catalog.append(
                {
                    "evidence_ref": f"S{len(catalog):04d}",
                    "source_turn_index": turn_index,
                    "source_span_index": span_index,
                    "role": "user",
                    "content": f"turn {turn_index} span {span_index}",
                    "_cost": cost,
                }
            )
    return catalog


def _measure(spans: list[dict[str, object]]) -> dict[str, object]:
    refs = [str(span["evidence_ref"]) for span in spans]
    source_sha = v2.sha256_bytes(v2.canonical_json(refs))
    schema_sha = v2.sha256_bytes(v2.canonical_json({"refs": refs}))
    budget = 1 + sum(int(span["_cost"]) for span in spans)
    return {
        "request_budget_tokens": budget,
        "rendered_chat_prompt_tokens": budget - 1,
        "dynamic_schema_tokens": 1,
        "rendered_chat_prompt_sha256": source_sha,
        "request_budget_view_sha256": v2.sha256_bytes(f"{source_sha}:{schema_sha}".encode()),
        "rendered_source_payload_sha256": source_sha,
        "dynamic_schema_sha256": schema_sha,
        "request_sha256": v2.sha256_bytes(f"request:{source_sha}:{schema_sha}".encode()),
    }


def _plan(costs: list[list[int]], *, limit: int, session: str = "session-1") -> dict[str, object]:
    return plan_session_chunks(
        session_identity_sha256=session,
        catalog=_catalog(costs),
        chunking_contract_sha256="contract-sha",
        max_prompt_tokens=limit,
        measure_spans=_measure,
    )


def test_greedily_groups_complete_turns_in_source_order():
    plan = _plan([[2, 2], [2, 2], [2, 2], [2, 2]], limit=9)

    assert [chunk["primary_turn_indices"] for chunk in plan["chunks"]] == [[0, 1], [2, 3]]
    assert all(chunk["request_budget_tokens"] <= 9 for chunk in plan["chunks"])
    assert plan["primary_span_coverage_exactly_once"] is True


def test_oversized_turn_is_partitioned_only_between_rawspans():
    plan = _plan([[3, 3, 3, 3, 3]], limit=7)

    chunks = plan["chunks"]
    assert [chunk["primary_span_ids"] for chunk in chunks] == [
        ["S0000", "S0001"],
        ["S0002", "S0003"],
        ["S0004"],
    ]
    assert all(chunk["oversized_turn_split"] for chunk in chunks)
    assert all(chunk["request_budget_tokens"] <= 7 for chunk in chunks)


def test_overlap_prefers_preceding_then_uses_following_only_if_it_fits():
    plan = _plan([[2], [4, 4], [2]], limit=7)

    middle_chunks = [chunk for chunk in plan["chunks"] if chunk["primary_turn_indices"] == [1]]
    assert len(middle_chunks) == 2
    assert all(chunk["overlap_turn_indices"] == [0] for chunk in middle_chunks)
    assert all(chunk["request_budget_tokens"] == 7 for chunk in middle_chunks)


def test_overlap_includes_both_neighbors_when_the_complete_budget_fits():
    plan = _plan([[1], [4, 4], [1]], limit=7)

    middle_chunks = [chunk for chunk in plan["chunks"] if chunk["primary_turn_indices"] == [1]]
    assert len(middle_chunks) == 2
    assert all(chunk["overlap_turn_indices"] == [0, 2] for chunk in middle_chunks)
    assert all(chunk["request_budget_tokens"] == 7 for chunk in middle_chunks)


def test_if_preceding_overlap_does_not_fit_omit_all_overlap_even_if_following_fits():
    plan = _plan([[4], [3, 3], [1]], limit=5)

    middle_chunks = [chunk for chunk in plan["chunks"] if chunk["primary_turn_indices"] == [1]]
    assert len(middle_chunks) == 2
    assert all(chunk["overlap_turn_indices"] == [] for chunk in middle_chunks)
    assert all(chunk["request_budget_tokens"] == 4 for chunk in middle_chunks)


def test_a_single_rawspan_that_exceeds_limit_is_fatal():
    with pytest.raises(ValueError, match="rawspan_prompt_exceeds_limit"):
        _plan([[8]], limit=7)


def test_catalog_source_order_is_required():
    catalog = _catalog([[1], [1]])
    catalog[1]["source_turn_index"] = 0
    catalog[1]["source_span_index"] = 0
    with pytest.raises(ValueError, match="chunk_plan_catalog_must_preserve_source_order"):
        plan_session_chunks(
            session_identity_sha256="session-1",
            catalog=catalog,
            chunking_contract_sha256="contract-sha",
            max_prompt_tokens=10,
            measure_spans=_measure,
        )


def test_chunk_identity_is_deterministic_and_bound_to_session_and_payload():
    costs = [[2], [3, 3]]
    first = _plan(costs, limit=5)
    replay = _plan(costs, limit=5)
    other_session = _plan(costs, limit=5, session="session-2")

    assert [row["chunk_id"] for row in first["chunks"]] == [
        row["chunk_id"] for row in replay["chunks"]
    ]
    assert [row["chunk_id"] for row in first["chunks"]] != [
        row["chunk_id"] for row in other_session["chunks"]
    ]


def test_aggregation_collapses_exact_cross_chunk_only_and_retains_lineage():
    catalog = _catalog([[1, 1]])
    outputs = [
        {
            "chunk_index": 0,
            "chunk_id": "chunk-a",
            "normalized_packet": {
                "propositions": [
                    {"proposition_text": "Same claim.", "evidence_refs": ["S0000"]},
                    {"proposition_text": "Same claim.", "evidence_refs": ["S0000"]},
                    {"proposition_text": "Same claim.", "evidence_refs": ["S0001"]},
                ]
            },
        },
        {
            "chunk_index": 1,
            "chunk_id": "chunk-b",
            "normalized_packet": {
                "propositions": [
                    {"proposition_text": "Same claim.", "evidence_refs": ["S0000"]},
                    {"proposition_text": "Same claim.", "evidence_refs": ["S0001"]},
                    {"proposition_text": "Distinct claim.", "evidence_refs": ["S0000"]},
                ]
            },
        },
    ]

    merged, diagnostics = aggregate_chunk_propositions(catalog=catalog, chunk_outputs=outputs)

    assert len(merged) == 4
    assert diagnostics["exact_cross_chunk_duplicates_removed"] == 2
    assert merged[0]["source_chunk_ids"] == ["chunk-a", "chunk-b"]
    assert merged[1]["source_chunk_ids"] == ["chunk-a"]
    assert merged[2]["evidence_refs"] == ["S0001"]
    assert merged[2]["source_chunk_ids"] == ["chunk-a", "chunk-b"]
    assert merged[3]["proposition_text"] == "Distinct claim."


def test_v4_writer_keeps_minimal_v3_semantics_and_uses_8192_capacity():
    catalog = v2.build_source_span_catalog(
        [
            {
                "source_turn_index": 0,
                "source_span_index": 0,
                "char_start": 0,
                "char_end": len("I like tea."),
                "role": "user",
                "content": "I like tea.",
            }
        ]
    )
    request = v4.writer_request(
        session_date="2026-09-29",
        catalog=catalog,
        system_prompt="frozen v3 prompt",
        model_alias="local-qwen",
    )
    schema = request["response_format"]["json_schema"]["schema"]

    assert request["max_tokens"] == 8192
    assert request["response_format"]["json_schema"]["name"] == (
        "flat_proposition_packet_v4_chunked"
    )
    assert set(schema["properties"]["propositions"]["items"]["properties"]) == {
        "proposition_text",
        "evidence_refs",
    }
    assert (
        schema["properties"]["propositions"]["items"]["properties"]["evidence_refs"]["uniqueItems"]
        is True
    )
    assert set(
        json.loads(json.dumps(request))["response_format"]["json_schema"]["schema"]["properties"]
    ) == {"propositions"}


def test_chunking_contract_freezes_exact_schema_delimiters_and_diagnostic_threshold():
    contract_path = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "research"
        / "memory"
        / "flatprop_chunking_v1.json"
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))

    assert contract["tokenizer_budget_view"]["schema_prefix"] == ("\n\n<FLATPROP_DYNAMIC_SCHEMA>\n")
    assert contract["tokenizer_budget_view"]["schema_suffix"] == ("\n</FLATPROP_DYNAMIC_SCHEMA>")
    assert contract["duplicate_diagnostics"]["high_similarity_threshold"] == 0.9
    assert contract["duplicate_diagnostics"]["purpose"].startswith("diagnostic only")
