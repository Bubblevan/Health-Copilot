"""Deterministic overflow subdivision and exact FlatProp identity recovery."""

from __future__ import annotations

from collections.abc import Callable
from difflib import SequenceMatcher
from typing import Any, Sequence, TypeVar

try:
    from .flat_proposition_writer_v2 import canonical_json, sha256_bytes
except ImportError:
    from flat_proposition_writer_v2 import canonical_json, sha256_bytes

T = TypeVar("T")


def balanced_split_index(token_masses: Sequence[int]) -> int:
    """Return the earliest legal split with minimum left/right token imbalance."""
    if len(token_masses) < 2:
        raise ValueError("recursive_split_requires_at_least_two_primary_units")
    if any(type(mass) is not int or mass < 0 for mass in token_masses):
        raise ValueError("recursive_split_token_masses_must_be_nonnegative_integers")

    total = sum(token_masses)
    left = 0
    best_index = 1
    best_key: tuple[int, int] | None = None
    for index, mass in enumerate(token_masses[:-1], 1):
        left += mass
        key = (abs(left - (total - left)), index)
        if best_key is None or key < best_key:
            best_key = key
            best_index = index
    return best_index


def balanced_partition(units: Sequence[T], token_masses: Sequence[int]) -> tuple[list[T], list[T]]:
    if len(units) != len(token_masses):
        raise ValueError("recursive_split_units_and_token_masses_length_mismatch")
    index = balanced_split_index(token_masses)
    return list(units[:index]), list(units[index:])


def recursive_child_identity(identity_body: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    required = {
        "split_contract_sha256",
        "session_identity_sha256",
        "root_initial_chunk_id",
        "parent_chunk_id",
        "recursion_depth",
        "primary_unit_kind",
        "primary_unit_ordinals",
        "primary_span_ordinals",
        "overlap_turn_ordinals",
        "overlap_span_ordinals",
        "rendered_source_payload_sha256",
        "dynamic_schema_sha256",
    }
    if set(identity_body) != required:
        raise ValueError("recursive_child_identity_fields_differ_from_contract")
    if identity_body["primary_unit_kind"] not in {"turn", "rawspan"}:
        raise ValueError("recursive_child_primary_unit_kind_invalid")
    if type(identity_body["recursion_depth"]) is not int or identity_body["recursion_depth"] < 1:
        raise ValueError("recursive_child_depth_must_be_positive")
    return sha256_bytes(canonical_json(identity_body)), identity_body


def overflow_disposition(primary_rawspan_count: int) -> str:
    if type(primary_rawspan_count) is not int or primary_rawspan_count < 1:
        raise ValueError("overflow_parent_primary_rawspan_count_invalid")
    return "SUBDIVISION" if primary_rawspan_count >= 2 else "IRREDUCIBLE_WRITER_OVERFLOW"


def overflow_requires_subdivision(finish_reason: str | None) -> bool:
    return finish_reason == "length"


def resolve_recursive_overflow(
    node: dict[str, Any],
    *,
    execute: Callable[[dict[str, Any]], dict[str, Any]],
    split: Callable[[dict[str, Any]], tuple[dict[str, Any], dict[str, Any]]],
    prepare: Callable[[dict[str, Any]], list[dict[str, Any]]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return terminal outputs and all attempts; overflow parents never become leaves."""
    pending = [node]
    leaves: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    while pending:
        current = pending.pop()
        result = execute(current)
        if overflow_requires_subdivision(result.get("finish_reason")):
            span_count = len(current.get("primary_span_ordinals", []))
            if overflow_disposition(span_count) == "IRREDUCIBLE_WRITER_OVERFLOW":
                raise RuntimeError("IRREDUCIBLE_WRITER_OVERFLOW")
            attempts.append(
                {
                    **result,
                    "chunk_id": current["chunk_id"],
                    "status": "OVERFLOW_PARENT",
                    "semantic_output_used": False,
                }
            )
            left, right = split(current)
            left_nodes = prepare(left) if prepare is not None else [left]
            right_nodes = prepare(right) if prepare is not None else [right]
            pending.extend(reversed([*left_nodes, *right_nodes]))
            continue
        if result.get("success") is not True:
            raise RuntimeError("recursive_writer_non_length_failure_is_fatal")
        terminal = {**result, "chunk_id": current["chunk_id"], "status": "TERMINAL_LEAF"}
        leaves.append(terminal)
        attempts.append(terminal)
    return leaves, attempts


def assert_primary_leaf_coverage(
    expected_ordinals: Sequence[int], leaf_ordinals: Sequence[Sequence[int]]
) -> None:
    expected = list(expected_ordinals)
    actual = [ordinal for leaf in leaf_ordinals for ordinal in leaf]
    if actual != sorted(actual):
        raise ValueError("recursive_primary_leaf_order_is_not_chronological")
    if actual != expected or len(actual) != len(set(actual)):
        raise ValueError("recursive_primary_leaf_coverage_not_exactly_once")


def logical_proposition_identity(proposition_text: str, evidence_refs: Sequence[str]) -> str:
    text = proposition_text.strip()
    refs = list(evidence_refs)
    if not text or not refs or len(refs) != len(set(refs)):
        raise ValueError("logical_proposition_identity_requires_text_and_evidence")
    return sha256_bytes(
        canonical_json({"proposition_text": text, "evidence_refs": refs})
    )


def aggregate_exact_leaf_propositions(
    *, catalog: list[dict[str, Any]], leaf_outputs: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Collapse exact identities globally within a session, retaining all emissions."""
    catalog_order = {span["evidence_ref"]: index for index, span in enumerate(catalog)}
    merged: list[dict[str, Any]] = []
    positions: dict[str, int] = {}
    counts = {
        "raw_generated_emissions": 0,
        "final_logical_propositions": 0,
        "exact_duplicates_removed_total": 0,
        "exact_duplicates_removed_within_leaf": 0,
        "exact_duplicates_removed_across_leaves": 0,
        "exact_duplicates_attributed_to_overlap": 0,
        "near_duplicate_pairs_retained": 0,
    }

    ordered_outputs = sorted(
        leaf_outputs,
        key=lambda row: (
            row["primary_start_ordinal"],
            row["primary_end_ordinal"],
            row["chunk_id"],
        ),
    )
    for leaf in ordered_outputs:
        if leaf.get("terminal") is False or leaf.get("success") is False:
            raise ValueError("nonterminal_or_failed_output_cannot_enter_flatprop_aggregate")
        chunk_id = leaf["chunk_id"]
        overlap_refs = set(leaf.get("overlap_span_ids", []))
        primary_refs = set(leaf.get("primary_span_ids", []))
        for output_index, prop in enumerate(leaf["normalized_packet"]["propositions"]):
            text = prop["proposition_text"].strip()
            raw_refs = prop["evidence_refs"]
            if not text or not raw_refs or any(ref not in catalog_order for ref in raw_refs):
                raise ValueError("leaf_proposition_invalid_or_out_of_session_catalog")
            refs = sorted(set(raw_refs), key=catalog_order.__getitem__)
            identity = logical_proposition_identity(text, refs)
            counts["raw_generated_emissions"] += 1
            existing_index = positions.get(identity)
            emission = {
                "chunk_id": chunk_id,
                "chunk_proposition_index": output_index,
                "primary_span_ids": sorted(primary_refs, key=catalog_order.__getitem__),
                "overlap_span_ids": sorted(overlap_refs, key=catalog_order.__getitem__),
            }
            if existing_index is not None:
                existing = merged[existing_index]
                cross_leaf = chunk_id not in existing["source_chunk_ids"]
                existing["chunk_emissions"].append(emission)
                existing["duplicate_emission_count"] += 1
                if not cross_leaf:
                    counts["exact_duplicates_removed_within_leaf"] += 1
                else:
                    existing["source_chunk_ids"].append(chunk_id)
                    counts["exact_duplicates_removed_across_leaves"] += 1
                if cross_leaf and any(
                    (set(prior["primary_span_ids"]) & set(refs) & overlap_refs)
                    or (set(prior["overlap_span_ids"]) & set(refs) & primary_refs)
                    for prior in existing["chunk_emissions"][:-1]
                    if prior["chunk_id"] != chunk_id
                ):
                    counts["exact_duplicates_attributed_to_overlap"] += 1
                continue

            merged.append(
                {
                    **prop,
                    "proposition_text": text,
                    "evidence_refs": refs,
                    "source_chunk_ids": [chunk_id],
                    "chunk_emissions": [emission],
                    "duplicate_emission_count": 0,
                    "logical_proposition_identity_sha256": identity,
                }
            )
            positions[identity] = len(merged) - 1

    for index, prop in enumerate(merged):
        prop["proposition_index"] = index
    counts["final_logical_propositions"] = len(merged)
    counts["exact_duplicates_removed_total"] = (
        counts["raw_generated_emissions"] - len(merged)
    )
    if counts["exact_duplicates_removed_total"] != (
        counts["exact_duplicates_removed_within_leaf"]
        + counts["exact_duplicates_removed_across_leaves"]
    ):
        raise ValueError("exact_duplicate_accounting_inconsistent")
    counts["identity_definition"] = (
        "sha256(canonical_json(stripped proposition_text, catalog-ordered canonical evidence_refs))"
    )
    counts["near_duplicates_merged"] = False
    for index, left in enumerate(merged):
        left_text = left["proposition_text"].casefold()
        for right in merged[index + 1 :]:
            right_text = right["proposition_text"].casefold()
            if min(len(left_text), len(right_text)) / max(
                1, max(len(left_text), len(right_text))
            ) < 0.90:
                continue
            if SequenceMatcher(None, left_text, right_text, autojunk=False).ratio() >= 0.90:
                counts["near_duplicate_pairs_retained"] += 1
    return merged, counts
