"""Span-witnessed per-proposition identity proposals for MEM-3B0Q-R2."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any

from tools.research.memory import revision_identity, revision_pairwise_admission

PROPERTY_KINDS = (
    "SINGLE_VALUE_STATE",
    "MULTI_VALUE_STATE",
    "EVENT",
    "NON_STATE",
    "UNKNOWN",
)
IDENTITY_FIELDS = (
    "principal_key",
    "principal_witness_span",
    "object_key",
    "object_witness_span",
    "attribute_key",
    "attribute_witness_span",
    "value_text",
    "value_witness_span",
    "property_kind",
    "change_cue_span",
)
GENERIC_ATTRIBUTE_TOKENS = frozenset(
    {
        "context",
        "event",
        "fact",
        "information",
        "interest",
        "number",
        "plan",
        "planning",
        "preference",
        "reported",
        "status",
        "thing",
        "value",
    }
)
GENERIC_KEY_TOKENS = GENERIC_ATTRIBUTE_TOKENS | frozenset(
    {
        "account",
        "and",
        "at",
        "by",
        "count",
        "current",
        "for",
        "from",
        "in",
        "item",
        "object",
        "of",
        "on",
        "or",
        "record",
        "state",
        "to",
        "user",
        "with",
    }
)
LEXICAL_ALIASES = {
    "color": frozenset(
        {"black", "blue", "brown", "gray", "grey", "green", "neutral", "red", "white"}
    ),
    "colour": frozenset(
        {"black", "blue", "brown", "gray", "grey", "green", "neutral", "red", "white"}
    ),
    "follower": frozenset({"followers"}),
    "follower_count": frozenset({"follower", "followers"}),
    "nodejs": frozenset({"node", "node.js"}),
    "software": frozenset({"node", "node.js"}),
}

SYSTEM_PROMPT = """Extract a typed identity proposal for each supplied memory independently.
Do not compare memories, group them, choose which one is newer, or decide whether an
update should be applied. Do not infer missing details from outside the proposition.

Return exactly one identity object for every supplied memory ID. Keys are concise
lower-snake labels for the principal, optional object/target, and mutable attribute.
For each key and value, also return the shortest exact substring from that same
proposition that supports it. Witness spans must be copied verbatim, including case.
Use null when an object/target or explicit change cue is absent. Never invent a span.

property_kind:
- SINGLE_VALUE_STATE: one concrete attribute that normally has one active value at a time.
- MULTI_VALUE_STATE: values or members can coexist, such as interests or owned items.
- EVENT: a completed occurrence that remains historically true.
- NON_STATE: no useful mutable state slot.
- UNKNOWN: insufficient evidence.

Do not put dates, values, session IDs, or memory IDs in keys. value_text may be
normalized, but value_witness_span must be an exact source substring. A change cue is
only a verbatim phrase that explicitly corrects, replaces, or deletes prior state.
Return only JSON matching the supplied schema."""


class SpanIdentityError(ValueError):
    """The proposal envelope or record violates the frozen pilot contract."""


class DuplicateJsonKey(ValueError):
    """The raw JSON contained a repeated object key."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateJsonKey(key)
        result[key] = value
    return result


def response_schema(memory_ids: list[str]) -> dict[str, Any]:
    if not memory_ids or len(memory_ids) != len(set(memory_ids)):
        raise SpanIdentityError("response_schema_requires_unique_ids")
    nullable_string = {
        "anyOf": [{"type": "string"}, {"type": "null"}],
    }
    identity_schema = {
        "type": "object",
        "required": list(IDENTITY_FIELDS),
        "additionalProperties": False,
        "properties": {
            "principal_key": {"type": "string", "minLength": 1},
            "principal_witness_span": nullable_string,
            "object_key": nullable_string,
            "object_witness_span": nullable_string,
            "attribute_key": {"type": "string", "minLength": 1},
            "attribute_witness_span": nullable_string,
            "value_text": {"type": "string", "minLength": 1},
            "value_witness_span": nullable_string,
            "property_kind": {"type": "string", "enum": list(PROPERTY_KINDS)},
            "change_cue_span": nullable_string,
        },
    }
    return {
        "type": "object",
        "required": ["identities"],
        "additionalProperties": False,
        "properties": {
            "identities": {
                "type": "object",
                "required": list(memory_ids),
                "additionalProperties": False,
                "properties": {memory_id: identity_schema for memory_id in memory_ids},
            }
        },
    }


def build_request(
    batch: list[dict[str, str]], *, model: str, max_tokens: int = 8192
) -> dict[str, Any]:
    if not batch or len({row.get("memory_id") for row in batch}) != len(batch):
        raise SpanIdentityError("request_requires_unique_memory_ids")
    projected = []
    for row in batch:
        memory_id, proposition = row.get("memory_id"), row.get("proposition_text")
        if not isinstance(memory_id, str) or not memory_id.strip():
            raise SpanIdentityError("request_memory_id_invalid")
        if not isinstance(proposition, str) or not proposition.strip():
            raise SpanIdentityError("request_proposition_invalid")
        projected.append({"memory_id": memory_id, "proposition_text": proposition})
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps({"memories": projected}, ensure_ascii=False, indent=2),
            },
        ],
        "temperature": 0,
        "seed": 42,
        "max_tokens": max_tokens,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "span_grounded_memory_identity_v1",
                "strict": True,
                "schema": response_schema([row["memory_id"] for row in projected]),
            },
        },
    }


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.casefold()))


def normalize_key(value: str, *, field: str, memory_id: str) -> str:
    try:
        return revision_identity.normalize_key(value, field=field, memory_id=memory_id)
    except revision_identity.IdentityContractError as exc:
        raise SpanIdentityError(f"invalid_{field}:{exc}") from exc


def _span_status(value: Any, source_text: str, *, required: bool) -> tuple[str | None, str | None]:
    if value is None:
        return (None, "MISSING_REQUIRED_WITNESS" if required else None)
    if not isinstance(value, str) or not value.strip():
        return (None, "INVALID_WITNESS_TYPE")
    span = value.strip()
    occurrences = source_text.count(span)
    if occurrences != 1:
        return (span, "WITNESS_NOT_UNIQUE_SOURCE_SUBSTRING")
    return span, None


def _supported_tokens(key: str, witness: str, *, field: str) -> tuple[bool, list[str]]:
    key_tokens = _tokens(key.replace("_", " ")) - GENERIC_KEY_TOKENS
    witness_tokens = _tokens(witness)
    if field == "principal_key" and key == "user" and "user" in witness_tokens:
        return True, []
    unsupported = []
    for token in sorted(key_tokens):
        aliases = LEXICAL_ALIASES.get(token, frozenset())
        if token not in witness_tokens and not (aliases & witness_tokens):
            unsupported.append(token)
    return bool(key_tokens) and not unsupported, unsupported


def validate_response(
    raw_bytes: bytes, batch: list[dict[str, str]], source_by_id: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    expected_ids = [row["memory_id"] for row in batch]
    if not expected_ids or len(expected_ids) != len(set(expected_ids)):
        raise SpanIdentityError("input_ids_not_unique")
    try:
        payload = json.loads(raw_bytes.decode("utf-8"), object_pairs_hook=_unique_object)
    except DuplicateJsonKey as exc:
        raise SpanIdentityError(f"duplicate_json_key:{exc}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SpanIdentityError("malformed_json") from exc
    if not isinstance(payload, dict) or set(payload) != {"identities"}:
        raise SpanIdentityError("invalid_root_shape")
    identities = payload["identities"]
    if not isinstance(identities, dict) or set(identities) != set(expected_ids):
        raise SpanIdentityError("identity_id_coverage_mismatch")

    records = []
    for memory_id in expected_ids:
        raw = identities[memory_id]
        if not isinstance(raw, dict) or set(raw) != set(IDENTITY_FIELDS):
            raise SpanIdentityError(f"invalid_identity_shape:{memory_id}")
        if raw["property_kind"] not in PROPERTY_KINDS:
            raise SpanIdentityError(f"invalid_property_kind:{memory_id}")
        source = source_by_id.get(memory_id)
        proposition = source.get("proposition_text") if source else None
        scope_id = source.get("scope_id") if source else None
        if not isinstance(proposition, str) or not isinstance(scope_id, str):
            raise SpanIdentityError(f"source_record_missing:{memory_id}")

        reasons: list[str] = []
        normalized: dict[str, Any] = {"memory_id": memory_id}
        for key_field in ("principal_key", "attribute_key"):
            try:
                normalized[key_field] = normalize_key(
                    raw[key_field], field=key_field, memory_id=memory_id
                )
            except SpanIdentityError as exc:
                reasons.append(str(exc))
                normalized[key_field] = None
        object_key = raw["object_key"]
        if object_key is None:
            normalized["object_key"] = None
            if raw["object_witness_span"] is not None:
                reasons.append("object_witness_span:PRESENT_WITHOUT_OBJECT_KEY")
        else:
            try:
                normalized["object_key"] = normalize_key(
                    object_key, field="object_key", memory_id=memory_id
                )
            except SpanIdentityError as exc:
                reasons.append(str(exc))
                normalized["object_key"] = None

        spans: dict[str, str | None] = {}
        span_requirements = {
            "principal_witness_span": (True, raw["principal_key"]),
            "object_witness_span": (raw["object_key"] is not None, raw["object_key"]),
            "attribute_witness_span": (True, raw["attribute_key"]),
            "value_witness_span": (True, raw["value_text"]),
            "change_cue_span": (False, None),
        }
        for span_field, (required, _) in span_requirements.items():
            span, reason = _span_status(raw[span_field], proposition, required=required)
            spans[span_field] = span
            if reason:
                reasons.append(f"{span_field}:{reason}")

        for key_field, witness_field in (
            ("principal_key", "principal_witness_span"),
            ("object_key", "object_witness_span"),
            ("attribute_key", "attribute_witness_span"),
        ):
            key = normalized.get(key_field)
            witness = spans.get(witness_field)
            if key is None or witness is None:
                continue
            supported, unsupported = _supported_tokens(key, witness, field=key_field)
            if not supported:
                reasons.append(f"{key_field}:UNSUPPORTED_KEY_TOKENS:{','.join(unsupported)}")

        value_text = raw["value_text"]
        if not isinstance(value_text, str) or not value_text.strip():
            reasons.append("value_text:EMPTY")
            value_text = None
        elif spans.get("value_witness_span") is not None:
            grounding, grounding_reason = revision_pairwise_admission.grounding_result(
                proposition, value_text
            )
            value_tokens = _tokens(value_text)
            witness_tokens = _tokens(spans["value_witness_span"])
            if grounding != "PASS" or not value_tokens.intersection(witness_tokens):
                reasons.append(f"value_text:NOT_GROUNDED:{grounding_reason}")
        normalized.update(spans)
        normalized["value_text"] = value_text.strip() if value_text else None
        normalized["property_kind"] = raw["property_kind"]
        normalized["validation_reasons"] = sorted(set(reasons))
        normalized["identity_status"] = "UNRESOLVED" if reasons else "GROUNDED"
        normalized["scope_id"] = scope_id
        if normalized["identity_status"] == "GROUNDED":
            normalized["slot_key"] = [
                scope_id,
                normalized["principal_key"],
                normalized["object_key"],
                normalized["attribute_key"],
            ]
            if normalized["property_kind"] == "SINGLE_VALUE_STATE":
                normalized["slot_candidate"] = True
            else:
                normalized["slot_candidate"] = False
                normalized["validation_reasons"].append(
                    f"PROPERTY_KIND_VETO:{normalized['property_kind']}"
                )
        else:
            normalized["slot_key"] = None
            normalized["slot_candidate"] = False
        records.append(normalized)
    return records


def exact_slot_groups(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str | None, str], list[str]] = defaultdict(list)
    for record in records:
        if record.get("slot_candidate") is not True:
            continue
        key = tuple(record["slot_key"])
        groups[key].append(str(record["memory_id"]))
    return [
        {"slot_key": list(key), "memory_ids": sorted(memory_ids)}
        for key, memory_ids in sorted(
            groups.items(), key=lambda item: json.dumps(item[0], ensure_ascii=False)
        )
        if len(memory_ids) > 1
    ]
