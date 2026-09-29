"""Deterministic primary-turn and oversized-turn RawSpan chunk planning."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

try:
    from .flat_proposition_writer_v2 import canonical_json, sha256_bytes
except ImportError:
    from flat_proposition_writer_v2 import canonical_json, sha256_bytes


TokenMeasurement = dict[str, Any]
MeasureSpans = Callable[[list[dict[str, Any]]], TokenMeasurement]


def plan_session_chunks(
    *,
    session_identity_sha256: str,
    catalog: list[dict[str, Any]],
    chunking_contract_sha256: str,
    max_prompt_tokens: int,
    measure_spans: MeasureSpans,
) -> dict[str, Any]:
    if not catalog or max_prompt_tokens <= 0:
        raise ValueError("chunk_plan_requires_catalog_and_positive_token_limit")
    refs = [span.get("evidence_ref") for span in catalog]
    if any(not isinstance(ref, str) or not ref for ref in refs) or len(set(refs)) != len(refs):
        raise ValueError("chunk_plan_requires_unique_ordered_evidence_refs")

    spans_by_turn: dict[int, list[int]] = defaultdict(list)
    previous_turn_index = -1
    previous_span_index = -1
    for ordinal, span in enumerate(catalog):
        turn_index = span.get("source_turn_index")
        span_index = span.get("source_span_index")
        if type(turn_index) is not int or turn_index < 0:
            raise ValueError("chunk_plan_source_turn_index_invalid")
        if type(span_index) is not int or span_index < 0:
            raise ValueError("chunk_plan_source_span_index_invalid")
        if turn_index < previous_turn_index or (
            turn_index == previous_turn_index and span_index <= previous_span_index
        ):
            raise ValueError("chunk_plan_catalog_must_preserve_source_order")
        if turn_index != previous_turn_index:
            previous_span_index = -1
        previous_turn_index = turn_index
        previous_span_index = span_index
        spans_by_turn[turn_index].append(ordinal)
    turn_indices = sorted(spans_by_turn)

    def measured(ordinals: list[int]) -> TokenMeasurement:
        if not ordinals or ordinals != sorted(set(ordinals)):
            raise ValueError("chunk_candidate_ordinals_must_be_nonempty_ordered_unique")
        result = measure_spans([catalog[ordinal] for ordinal in ordinals])
        token_count = result.get("request_budget_tokens")
        if type(token_count) is not int or token_count < 0:
            raise ValueError("chunk_measurement_missing_request_budget_tokens")
        return result

    def spans_for_turn_ordinals(ordinals: list[int]) -> list[int]:
        return [
            span_ordinal
            for turn_ordinal in ordinals
            for span_ordinal in spans_by_turn[turn_indices[turn_ordinal]]
        ]

    primary_groups: list[dict[str, Any]] = []
    current_turn_ordinals: list[int] = []

    def append_complete_turn_group(turn_ordinals: list[int]) -> None:
        primary_groups.append(
            {
                "primary_turn_ordinals": list(turn_ordinals),
                "primary_span_ordinals": spans_for_turn_ordinals(turn_ordinals),
                "oversized_turn_split": False,
            }
        )

    for turn_ordinal, turn_index in enumerate(turn_indices):
        turn_span_ordinals = spans_by_turn[turn_index]
        if current_turn_ordinals:
            candidate_turns = [*current_turn_ordinals, turn_ordinal]
            candidate_spans = spans_for_turn_ordinals(candidate_turns)
            if measured(candidate_spans)["request_budget_tokens"] <= max_prompt_tokens:
                current_turn_ordinals = candidate_turns
                continue
            append_complete_turn_group(current_turn_ordinals)
            current_turn_ordinals = []

        if measured(turn_span_ordinals)["request_budget_tokens"] <= max_prompt_tokens:
            current_turn_ordinals = [turn_ordinal]
            continue

        span_group: list[int] = []
        for span_ordinal in turn_span_ordinals:
            candidate_spans = [*span_group, span_ordinal]
            if measured(candidate_spans)["request_budget_tokens"] <= max_prompt_tokens:
                span_group = candidate_spans
                continue
            if not span_group:
                span = catalog[span_ordinal]
                raise ValueError(
                    "rawspan_prompt_exceeds_limit:"
                    f"turn={turn_index}:span={span.get('evidence_ref')}"
                )
            primary_groups.append(
                {
                    "primary_turn_ordinals": [turn_ordinal],
                    "primary_span_ordinals": span_group,
                    "oversized_turn_split": True,
                }
            )
            span_group = [span_ordinal]
            if measured(span_group)["request_budget_tokens"] > max_prompt_tokens:
                span = catalog[span_ordinal]
                raise ValueError(
                    "rawspan_prompt_exceeds_limit:"
                    f"turn={turn_index}:span={span.get('evidence_ref')}"
                )
        if span_group:
            primary_groups.append(
                {
                    "primary_turn_ordinals": [turn_ordinal],
                    "primary_span_ordinals": span_group,
                    "oversized_turn_split": True,
                }
            )

    if current_turn_ordinals:
        append_complete_turn_group(current_turn_ordinals)

    turn_spans = {
        turn_ordinal: spans_by_turn[turn_index]
        for turn_ordinal, turn_index in enumerate(turn_indices)
    }
    chunks = []
    for chunk_index, group in enumerate(primary_groups):
        primary_ordinals = group["primary_span_ordinals"]
        primary_measurement = measured(primary_ordinals)
        primary_turn_ordinals = group["primary_turn_ordinals"]
        previous_ordinal = min(primary_turn_ordinals) - 1
        following_ordinal = max(primary_turn_ordinals) + 1
        previous_available = previous_ordinal >= 0
        following_available = following_ordinal < len(turn_indices)
        selected_overlap: list[int] = []

        def overlap_candidate(turn_ordinals: list[int]) -> tuple[list[int], TokenMeasurement]:
            overlap_spans = [
                span_ordinal
                for turn_ordinal in turn_ordinals
                for span_ordinal in turn_spans[turn_ordinal]
            ]
            visible = sorted([*primary_ordinals, *overlap_spans])
            return visible, measured(visible)

        both_turns = []
        if previous_available:
            both_turns.append(previous_ordinal)
        if following_available:
            both_turns.append(following_ordinal)
        if both_turns:
            _, both_measurement = overlap_candidate(both_turns)
            if both_measurement["request_budget_tokens"] <= max_prompt_tokens:
                selected_overlap = both_turns
                final_measurement = both_measurement
            elif previous_available:
                previous_ordinals, previous_measurement = overlap_candidate([previous_ordinal])
                if previous_measurement["request_budget_tokens"] <= max_prompt_tokens:
                    selected_overlap = [previous_ordinal]
                    final_measurement = previous_measurement
                    if following_available:
                        preferred_ordinals, preferred_measurement = overlap_candidate(
                            [previous_ordinal, following_ordinal]
                        )
                        if preferred_measurement["request_budget_tokens"] <= max_prompt_tokens:
                            selected_overlap = [previous_ordinal, following_ordinal]
                            final_measurement = preferred_measurement
                else:
                    final_measurement = primary_measurement
            else:
                following_ordinals, following_measurement = overlap_candidate([following_ordinal])
                if following_measurement["request_budget_tokens"] <= max_prompt_tokens:
                    selected_overlap = [following_ordinal]
                    final_measurement = following_measurement
                else:
                    final_measurement = primary_measurement
        else:
            final_measurement = primary_measurement

        overlap_span_ordinals = [
            span_ordinal
            for turn_ordinal in selected_overlap
            for span_ordinal in turn_spans[turn_ordinal]
        ]
        visible_ordinals = sorted([*primary_ordinals, *overlap_span_ordinals])
        if final_measurement["request_budget_tokens"] > max_prompt_tokens:
            raise ValueError("final_chunk_prompt_exceeds_limit")

        identity_body = {
            "chunking_contract_sha256": chunking_contract_sha256,
            "session_identity_sha256": session_identity_sha256,
            "primary_start_ordinal": primary_ordinals[0],
            "primary_end_ordinal": primary_ordinals[-1],
            "overlap_ordinals": {
                "turn_ordinals": selected_overlap,
                "span_ordinals": overlap_span_ordinals,
            },
            "rendered_source_payload_sha256": final_measurement["rendered_source_payload_sha256"],
            "dynamic_schema_sha256": final_measurement["dynamic_schema_sha256"],
        }
        chunk_id = sha256_bytes(canonical_json(identity_body))
        chunks.append(
            {
                "chunk_index": chunk_index,
                "chunk_id": chunk_id,
                "chunk_identity": identity_body,
                "primary_start_ordinal": primary_ordinals[0],
                "primary_end_ordinal": primary_ordinals[-1],
                "primary_turn_ordinals": primary_turn_ordinals,
                "primary_turn_indices": [turn_indices[i] for i in primary_turn_ordinals],
                "primary_span_ordinals": primary_ordinals,
                "primary_span_ids": [catalog[i]["evidence_ref"] for i in primary_ordinals],
                "overlap_turn_ordinals": selected_overlap,
                "overlap_turn_indices": [turn_indices[i] for i in selected_overlap],
                "overlap_span_ordinals": overlap_span_ordinals,
                "overlap_span_ids": [catalog[i]["evidence_ref"] for i in overlap_span_ordinals],
                "visible_span_ordinals": visible_ordinals,
                "visible_span_ids": [catalog[i]["evidence_ref"] for i in visible_ordinals],
                "oversized_turn_split": group["oversized_turn_split"],
                "primary_request_budget_tokens": primary_measurement["request_budget_tokens"],
                **final_measurement,
            }
        )

    primary_coverage = [ordinal for chunk in chunks for ordinal in chunk["primary_span_ordinals"]]
    if sorted(primary_coverage) != list(range(len(catalog))) or len(primary_coverage) != len(
        set(primary_coverage)
    ):
        raise ValueError("primary_rawspan_coverage_not_exactly_once")

    turn_assignments = []
    for turn_ordinal, turn_index in enumerate(turn_indices):
        turn_chunks = [chunk for chunk in chunks if turn_ordinal in chunk["primary_turn_ordinals"]]
        covered = [
            ordinal
            for chunk in turn_chunks
            for ordinal in chunk["primary_span_ordinals"]
            if catalog[ordinal]["source_turn_index"] == turn_index
        ]
        if sorted(covered) != turn_spans[turn_ordinal] or len(covered) != len(set(covered)):
            raise ValueError("primary_turn_coverage_not_exactly_once")
        split = len(turn_chunks) > 1
        if split and any(
            not chunk["oversized_turn_split"] or chunk["primary_turn_ordinals"] != [turn_ordinal]
            for chunk in turn_chunks
        ):
            raise ValueError("only_oversized_turns_may_span_primary_chunks")
        turn_assignments.append(
            {
                "turn_ordinal": turn_ordinal,
                "source_turn_index": turn_index,
                "primary_chunk_ids": [chunk["chunk_id"] for chunk in turn_chunks],
                "split_across_rawspan_chunks": split,
            }
        )

    return {
        "session_identity_sha256": session_identity_sha256,
        "source_turn_count": len(turn_indices),
        "source_span_count": len(catalog),
        "chunk_count": len(chunks),
        "oversized_turn_split_count": sum(
            assignment["split_across_rawspan_chunks"] for assignment in turn_assignments
        ),
        "primary_span_coverage_exactly_once": True,
        "primary_turn_coverage_exactly_once_or_oversized_span_partition": True,
        "turn_assignments": turn_assignments,
        "chunks": chunks,
    }


def aggregate_chunk_propositions(
    *,
    catalog: list[dict[str, Any]],
    chunk_outputs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    catalog_order = {span["evidence_ref"]: index for index, span in enumerate(catalog)}
    merged: list[dict[str, Any]] = []
    positions_by_identity: dict[str, list[int]] = defaultdict(list)
    chunk_emission_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    exact_cross_chunk_duplicates = 0

    for chunk in sorted(chunk_outputs, key=lambda row: row["chunk_index"]):
        chunk_id = chunk["chunk_id"]
        for output_index, prop in enumerate(chunk["normalized_packet"]["propositions"]):
            text = prop["proposition_text"].strip()
            raw_refs = prop["evidence_refs"]
            if not text or not raw_refs or any(ref not in catalog_order for ref in raw_refs):
                raise ValueError("chunk_aggregate_proposition_is_invalid_or_out_of_catalog")
            refs = sorted(set(raw_refs), key=catalog_order.__getitem__)
            identity_body = {"proposition_text": text, "evidence_refs": refs}
            identity_sha = sha256_bytes(canonical_json(identity_body))
            prior_positions = positions_by_identity[identity_sha]
            repeated_in_same_chunk = chunk_emission_counts[identity_sha][chunk_id] > 0
            chunk_emission_counts[identity_sha][chunk_id] += 1
            cross_chunk_match = (
                prior_positions[0]
                if prior_positions
                and not repeated_in_same_chunk
                and any(
                    prior_chunk_id != chunk_id
                    for prior_chunk_id in chunk_emission_counts[identity_sha]
                    if chunk_emission_counts[identity_sha][prior_chunk_id]
                )
                else None
            )
            if cross_chunk_match is not None:
                target = merged[cross_chunk_match]
                if chunk_id not in target["source_chunk_ids"]:
                    target["source_chunk_ids"].append(chunk_id)
                target["chunk_emissions"].append(
                    {"chunk_id": chunk_id, "chunk_proposition_index": output_index}
                )
                target["duplicate_emission_count"] += 1
                exact_cross_chunk_duplicates += 1
                continue

            normalized = {
                **prop,
                "proposition_text": text,
                "evidence_refs": refs,
                "source_chunk_ids": [chunk_id],
                "chunk_emissions": [
                    {"chunk_id": chunk_id, "chunk_proposition_index": output_index}
                ],
                "duplicate_emission_count": 0,
                "logical_proposition_identity_sha256": identity_sha,
            }
            merged.append(normalized)
            prior_positions.append(len(merged) - 1)

    for proposition_index, prop in enumerate(merged):
        prop["proposition_index"] = proposition_index
    return merged, {
        "input_chunk_proposition_emissions": sum(
            len(chunk["normalized_packet"]["propositions"]) for chunk in chunk_outputs
        ),
        "aggregated_proposition_count": len(merged),
        "exact_cross_chunk_duplicates_removed": exact_cross_chunk_duplicates,
        "duplicate_identity_algorithm": "sha256(canonical_json(stripped proposition_text, catalog-ordered evidence_refs))",
        "paraphrases_merged": False,
    }
