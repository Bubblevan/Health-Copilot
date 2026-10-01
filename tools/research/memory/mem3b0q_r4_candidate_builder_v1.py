"""Offline-only exact alias candidate builder for the R4 redesign proposal."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from tools.research.memory import mem3b0q_r4 as frozen_r4


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
    *,
    source_id: str | None = None,
) -> dict[str, Any]:
    """Enumerate maximal exact typed aliases; do not infer missing aliases."""
    if not isinstance(source_text, str) or not source_text:
        raise CandidateBuildError("source_text_invalid")
    if source_id is not None and (not isinstance(source_id, str) or not source_id):
        raise CandidateBuildError("source_id_invalid")
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
        "source_id": source_id,
        "alias_registry_sha256": hashlib.sha256(registry_bytes).hexdigest(),
        "source_sha256": hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
        "candidates": candidates,
    }
    # This digest detects accidental drift; consumers must rebuild from trusted inputs.
    canonical = json.dumps(
        manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    manifest["candidate_manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    return manifest


def build_candidate_schema(source_id: str, source_text: str) -> dict[str, Any]:
    """Derive a case-bound schema from the source and frozen typed alias registry."""
    if not isinstance(source_id, str) or not source_id:
        raise CandidateBuildError("schema_source_id_invalid")
    manifest = build_candidates(source_text, frozen_r4.ALIASES, source_id=source_id)
    candidate_ids = {
        field: [row["candidate_id"] for row in manifest["candidates"][field]]
        for field in FIELDS
    }

    atom_properties = {
        f"{field}_candidate_id": (
            {"type": "string", "enum": candidate_ids[field]}
            if candidate_ids[field]
            else {"type": "string"}
        )
        for field in FIELDS
    }
    atom_properties.update(
        {
            "value_span": {"type": "string", "minLength": 1, "maxLength": 40},
                "cardinality_proposal": {
                    "type": "string",
                    "enum": sorted(frozen_r4.CARDINALITIES),
                },
        }
    )
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "MEM-3B0Q-R4 Candidate-Bounded Proposal",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "source_id": {"type": "string", "enum": [source_id]},
            "atoms": {
                "type": "array",
                "maxItems": 4 if all(candidate_ids.values()) else 0,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": atom_properties,
                    "required": list(atom_properties),
                },
            },
            "abstention_reason": {
                "type": "string",
                "enum": sorted(frozen_r4.ABSTENTION_REASONS),
            },
        },
        "required": ["source_id", "atoms", "abstention_reason"],
    }


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CandidateBuildError(f"duplicate_json_key:{key}")
        result[key] = value
    return result


def _parse_candidate_proposal(content: bytes | str) -> dict[str, Any]:
    if not isinstance(content, (bytes, str)):
        raise CandidateBuildError("content_type_invalid")
    try:
        decoded = content.decode("utf-8") if isinstance(content, bytes) else content
        payload = json.loads(decoded, object_pairs_hook=_unique_object)
    except CandidateBuildError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
        raise CandidateBuildError("malformed_json") from exc
    if not isinstance(payload, dict):
        raise CandidateBuildError("root_not_object")
    return payload


def validate_candidate_proposal(
    content: bytes | str,
    proposition: Mapping[str, Any],
    *,
    scope_id: str,
) -> dict[str, Any]:
    """Resolve candidate IDs against this case; reject inconsistent abstention."""
    if not isinstance(proposition, Mapping):
        raise CandidateBuildError("source_record_invalid")
    source_id = proposition.get("source_id")
    source_text = proposition.get("proposition_text")
    if not isinstance(source_id, str) or not source_id:
        raise CandidateBuildError("source_id_invalid")
    if not isinstance(source_text, str) or not source_text:
        raise CandidateBuildError("source_text_invalid")
    if not isinstance(scope_id, str) or not scope_id:
        raise CandidateBuildError("scope_id_invalid")

    manifest = build_candidates(source_text, frozen_r4.ALIASES, source_id=source_id)
    by_id = {
        field: {row["candidate_id"]: row for row in manifest["candidates"][field]}
        for field in FIELDS
    }
    payload = _parse_candidate_proposal(content)
    if set(payload) != {"source_id", "atoms", "abstention_reason"}:
        raise CandidateBuildError("root_keys_mismatch")
    if payload["source_id"] != source_id:
        raise CandidateBuildError("source_id_mismatch")
    atoms = payload["atoms"]
    reason = payload["abstention_reason"]
    if not isinstance(atoms, list) or len(atoms) > 4:
        raise CandidateBuildError("atoms_invalid")
    if not isinstance(reason, str) or reason not in frozen_r4.ABSTENTION_REASONS:
        raise CandidateBuildError("abstention_reason_invalid")
    if bool(atoms) != (reason == "NONE"):
        raise CandidateBuildError("atoms_abstention_inconsistent")

    normalized_atoms = []
    expected_atom_keys = {
        "owner_candidate_id",
        "object_candidate_id",
        "attribute_candidate_id",
        "value_span",
        "cardinality_proposal",
    }
    for index, atom in enumerate(atoms):
        if not isinstance(atom, dict) or set(atom) != expected_atom_keys:
            raise CandidateBuildError(f"atom_{index}:keys_mismatch")
        selected: dict[str, Mapping[str, Any]] = {}
        for field in FIELDS:
            candidate_id = atom[f"{field}_candidate_id"]
            if not isinstance(candidate_id, str) or candidate_id not in by_id[field]:
                raise CandidateBuildError(f"atom_{index}:{field}_candidate_invalid")
            selected[field] = by_id[field][candidate_id]

        value = atom["value_span"]
        if not isinstance(value, str) or not 1 <= len(value) <= 40:
            raise CandidateBuildError(f"atom_{index}:value_span_invalid")
        value_start = source_text.find(value)
        if value_start < 0 or source_text.find(value, value_start + 1) >= 0:
            raise CandidateBuildError(f"atom_{index}:value_span_not_unique")
        cardinality = atom["cardinality_proposal"]
        if not isinstance(cardinality, str) or cardinality not in frozen_r4.CARDINALITIES:
            raise CandidateBuildError(f"atom_{index}:cardinality_invalid")

        span_names = {
            "owner": "owner_span",
            "object": "object_span",
            "attribute": "attribute_span",
        }
        witnesses = {
            span_names[field]: {
                "text": selected[field]["source_span"],
                "start": selected[field]["start"],
                "end": selected[field]["end"],
            }
            for field in FIELDS
        }
        witnesses["value_span"] = {
            "text": value,
            "start": value_start,
            "end": value_start + len(value),
        }
        owner_id = selected["owner"]["canonical_id"]
        object_id = selected["object"]["canonical_id"]
        attribute_id = selected["attribute"]["canonical_id"]
        slot_candidate = cardinality in {
            "SINGLE_VALUE_AT_A_TIME",
            "MULTI_VALUE_CONCURRENT",
        }
        normalized_atoms.append(
            {
                "atom_index": index,
                "owner_id": owner_id,
                "object_id": object_id,
                "attribute_id": attribute_id,
                "value_text": value,
                "cardinality": cardinality,
                "scope_id": scope_id,
                "slot_key": [scope_id, owner_id, object_id, attribute_id]
                if slot_candidate
                else None,
                "slot_candidate": slot_candidate,
                "witnesses": witnesses,
            }
        )
    return {
        "source_id": source_id,
        "scope_id": scope_id,
        "abstention_reason": reason,
        "atoms": normalized_atoms,
    }
