"""Offline-only exact alias candidate builder for the R4 redesign proposal."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


CANDIDATE_BUILDER_VERSION = "mem3b0q-r4-candidate-builder-v1-proposal"
FIELDS = ("owner", "object", "attribute")


class CandidateBuildError(ValueError):
    """The supplied source or typed alias registry is unsafe to resolve."""


def _validate_registry(
    alias_registry: Mapping[str, Mapping[str, str]],
) -> dict[str, dict[str, str]]:
    if not isinstance(alias_registry, Mapping):
        raise CandidateBuildError("alias_registry_not_mapping")

    registry: dict[str, dict[str, str]] = {}
    for field in FIELDS:
        aliases = alias_registry.get(field)
        if not isinstance(aliases, Mapping):
            raise CandidateBuildError(f"alias_field_not_mapping:{field}")
        normalized: dict[str, str] = {}
        for alias, canonical_id in aliases.items():
            if (
                not isinstance(alias, str)
                or not alias
                or not isinstance(canonical_id, str)
                or not canonical_id
            ):
                raise CandidateBuildError(f"alias_entry_invalid:{field}")
            key = alias.casefold()
            previous = normalized.get(key)
            if previous is not None and previous != canonical_id:
                raise CandidateBuildError(f"alias_casefold_collision:{field}:{alias}")
            normalized[key] = canonical_id
        registry[field] = normalized
    return registry


def _fold_with_source_offsets(source: str) -> tuple[str, list[int]]:
    folded: list[str] = []
    original_index: list[int] = []
    for index, character in enumerate(source):
        for folded_character in character.casefold():
            folded.append(folded_character)
            original_index.append(index)
    return "".join(folded), original_index


def _is_word_character(character: str) -> bool:
    return character.isalnum() or character in "-_"


def _matches_for_field(
    source: str,
    folded_source: str,
    original_index: list[int],
    field: str,
    aliases: Mapping[str, str],
) -> list[dict[str, Any]]:
    raw: dict[tuple[int, int, str], dict[str, Any]] = {}
    for folded_alias, canonical_id in aliases.items():
        search_from = 0
        while True:
            folded_start = folded_source.find(folded_alias, search_from)
            if folded_start < 0:
                break
            folded_end = folded_start + len(folded_alias)
            start = original_index[folded_start]
            end = original_index[folded_end - 1] + 1
            if (
                start > 0
                and _is_word_character(folded_alias[0])
                and _is_word_character(source[start - 1])
            ):
                search_from = folded_start + 1
                continue
            if (
                end < len(source)
                and _is_word_character(folded_alias[-1])
                and _is_word_character(source[end])
            ):
                search_from = folded_start + 1
                continue
            source_span = source[start:end]
            if source_span.casefold() != folded_alias:
                raise CandidateBuildError("casefold_offset_reconstruction_mismatch")
            key = (start, end, canonical_id)
            raw[key] = {
                "field": field,
                "canonical_id": canonical_id,
                "source_span": source_span,
                "start": start,
                "end": end,
            }
            search_from = folded_start + 1

    spans: dict[tuple[int, int], set[str]] = {}
    for start, end, canonical_id in raw:
        spans.setdefault((start, end), set()).add(canonical_id)
    if any(len(canonical_ids) > 1 for canonical_ids in spans.values()):
        raise CandidateBuildError(f"typed_alias_collision:{field}")

    ordered = sorted(
        raw.values(),
        key=lambda row: (row["start"], -(row["end"] - row["start"]), row["canonical_id"]),
    )
    selected: list[dict[str, Any]] = []
    for candidate in ordered:
        start, end = candidate["start"], candidate["end"]
        overlaps = [
            prior
            for prior in selected
            if start < prior["end"] and prior["start"] < end
        ]
        if not overlaps:
            selected.append(candidate)
            continue
        if all(
            prior["start"] <= start
            and end <= prior["end"]
            and (prior["end"] - prior["start"]) > (end - start)
            for prior in overlaps
        ):
            continue
        raise CandidateBuildError(f"partial_alias_overlap:{field}")

    for candidate in selected:
        candidate["candidate_id"] = (
            f"{field}:{candidate['canonical_id']}:{candidate['start']}:{candidate['end']}"
        )
    return sorted(
        selected,
        key=lambda row: (row["start"], row["end"], row["canonical_id"]),
    )


def build_candidates(
    source_text: str,
    alias_registry: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    """Enumerate maximal exact typed aliases; do not infer missing aliases."""
    if not isinstance(source_text, str) or not source_text:
        raise CandidateBuildError("source_text_invalid")
    registry = _validate_registry(alias_registry)
    registry_manifest = {
        field: sorted(
            (
                {"alias": alias, "canonical_id": canonical_id}
                for alias, canonical_id in alias_registry[field].items()
            ),
            key=lambda row: (row["alias"].casefold(), row["alias"], row["canonical_id"]),
        )
        for field in FIELDS
    }
    registry_bytes = json.dumps(
        registry_manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    folded_source, original_index = _fold_with_source_offsets(source_text)
    candidates = {
        field: _matches_for_field(
            source_text,
            folded_source,
            original_index,
            field,
            registry[field],
        )
        for field in FIELDS
    }
    manifest = {
        "candidate_builder_version": CANDIDATE_BUILDER_VERSION,
        "alias_registry_sha256": hashlib.sha256(registry_bytes).hexdigest(),
        "source_sha256": hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
        "candidates": candidates,
    }
    canonical = json.dumps(
        manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    manifest["candidate_manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    return manifest
