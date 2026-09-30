"""Deterministic validation and diagnostics for MEM-3B0 revision identity."""

from __future__ import annotations

import difflib
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from itertools import combinations
from typing import Any

REVISION_KINDS = (
    "SINGLETON_STATE",
    "SET_STATE",
    "EVENT",
    "NON_REVISIONAL",
    "UNKNOWN",
)
INPUT_FIELDS = ("memory_id", "proposition_text", "source_authority", "scope_id")
OUTPUT_FIELDS = ("memory_id", "revision_kind", "subject_key", "attribute_key", "value_text")
KEY_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
FORBIDDEN_KEY_TOKENS = frozenset(
    {"old", "new", "stale", "superseded", "previous", "latest", "update", "delete", "noop"}
)
GENERIC_TOKENS = frozenset(
    {
        "a",
        "about",
        "after",
        "also",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "being",
        "by",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "his",
        "i",
        "in",
        "is",
        "it",
        "its",
        "me",
        "my",
        "of",
        "on",
        "or",
        "our",
        "she",
        "that",
        "the",
        "their",
        "them",
        "they",
        "this",
        "to",
        "was",
        "we",
        "were",
        "will",
        "with",
        "you",
        "your",
    }
)

SYSTEM_PROMPT = """You assign semantic identity to each proposition in a memory system.

For each supplied memory, return exactly one row with the same memory_id.
Use only the supplied scope_id, source_authority, and proposition_text. Do not infer
chronology, validity, or which observation should be preferred. The input contains
no timestamps by design. Do not emit current/old/new/stale/previous/latest status,
supersession, UPDATE, DELETE, NOOP, or any revision decision.

revision_kind meanings:
- SINGLETON_STATE: a mutable attribute normally has one value at a time for a subject
  (for example employer, home city, follower count, body weight, gym frequency).
- SET_STATE: multiple members can coexist; adding one does not replace another
  (for example likes, owned devices, skills, favorite genres).
- EVENT: a historical occurrence that remains true as an event even after later events.
- NON_REVISIONAL: a useful proposition that is not naturally a mutable state slot,
  collection member, or historical event.
- UNKNOWN: semantic certainty is insufficient. Prefer UNKNOWN to forcing a slot.

subject_key must name the stable referent, not its wording in this sentence. Use
concise lower-snake semantic identifiers (examples: user, assistant, user_mother,
user_dog, instagram_account). attribute_key identifies only the slot, never its
value or observation (examples: instagram_follower_count, gym_frequency_per_week,
current_employer, home_city, body_weight). Do not include dates, values, session IDs,
or memory IDs in either key. Do not encode which observation is newer.

value_text must preserve the value meaning supported by this proposition. Do not
normalize units, infer equivalence, consult outside knowledge, or use benchmark answers.
Return valid JSON exactly matching the supplied response schema, with no prose."""


class IdentityContractError(ValueError):
    """An identity input or model response violates the frozen contract."""


class IdentityLengthStop(RuntimeError):
    def __init__(self, ledger: dict[str, Any]):
        self.ledger = ledger
        super().__init__("finish_reason=length")


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def project_input_row(row: dict[str, Any]) -> dict[str, str]:
    """Project a frozen inventory row to the only fields allowed into identity calls."""
    projected: dict[str, str] = {}
    for field in INPUT_FIELDS:
        value = row.get(field)
        if not isinstance(value, str) or not value.strip():
            raise IdentityContractError(f"invalid_input_field:{field}")
        projected[field] = value if field == "proposition_text" else value.strip()
    return projected


def normalize_key(value: Any, *, field: str, memory_id: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise IdentityContractError(f"invalid_key:{field}")
    normalized = unicodedata.normalize("NFKC", value).strip().lower()
    normalized = re.sub(r"[\s-]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized)
    if not KEY_PATTERN.fullmatch(normalized):
        raise IdentityContractError(f"invalid_key_syntax:{field}:{value!r}")
    if FORBIDDEN_KEY_TOKENS.intersection(normalized.split("_")):
        raise IdentityContractError(f"forbidden_temporal_key_token:{field}:{normalized}")
    flattened_id = re.sub(r"[^a-z0-9]", "", memory_id.casefold())
    if flattened_id and flattened_id in normalized.replace("_", ""):
        raise IdentityContractError(f"memory_id_encoded_in_key:{field}")
    return normalized


def response_schema(memory_ids: list[str]) -> dict[str, Any]:
    if not memory_ids or len(memory_ids) != len(set(memory_ids)):
        raise IdentityContractError("response_schema_requires_unique_batch_ids")
    return {
        "type": "object",
        "required": ["identities"],
        "additionalProperties": False,
        "properties": {
            "identities": {
                "type": "array",
                "minItems": len(memory_ids),
                "maxItems": len(memory_ids),
                "items": {
                    "type": "object",
                    "required": list(OUTPUT_FIELDS),
                    "additionalProperties": False,
                    "properties": {
                        "memory_id": {"type": "string", "enum": list(memory_ids)},
                        "revision_kind": {"type": "string", "enum": list(REVISION_KINDS)},
                        "subject_key": {"type": "string", "minLength": 1},
                        "attribute_key": {"type": "string", "minLength": 1},
                        "value_text": {"type": "string", "minLength": 1},
                    },
                },
            }
        },
    }


def build_request(batch: list[dict[str, str]], *, model: str, max_tokens: int) -> dict[str, Any]:
    projected = [project_input_row(row) for row in batch]
    ids = [row["memory_id"] for row in projected]
    schema = response_schema(ids)
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
            "json_schema": {"name": "revision_identity_v1", "strict": True, "schema": schema},
        },
    }


def validate_identity_response(
    raw_bytes: bytes, batch: list[dict[str, str]]
) -> dict[str, list[dict[str, str]]]:
    try:
        payload = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IdentityContractError("malformed_identity_json") from exc
    if not isinstance(payload, dict) or set(payload) != {"identities"}:
        raise IdentityContractError("invalid_identity_root")
    rows = payload["identities"]
    if not isinstance(rows, list):
        raise IdentityContractError("identity_rows_not_array")
    expected_ids = [row["memory_id"] for row in batch]
    if len(expected_ids) != len(set(expected_ids)):
        raise IdentityContractError("duplicate_input_memory_id")
    source_by_id = {row["memory_id"]: row for row in batch}
    seen: set[str] = set()
    normalized_by_id: dict[str, dict[str, str]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != set(OUTPUT_FIELDS):
            raise IdentityContractError(f"invalid_identity_row_fields:{index}")
        memory_id = row["memory_id"]
        if not isinstance(memory_id, str) or memory_id not in source_by_id:
            raise IdentityContractError(f"unknown_memory_id:{index}")
        if memory_id in seen:
            raise IdentityContractError(f"duplicate_memory_id:{memory_id}")
        seen.add(memory_id)
        kind = row["revision_kind"]
        if not isinstance(kind, str) or kind not in REVISION_KINDS:
            raise IdentityContractError(f"illegal_revision_kind:{memory_id}")
        subject = normalize_key(row["subject_key"], field="subject_key", memory_id=memory_id)
        attribute = normalize_key(row["attribute_key"], field="attribute_key", memory_id=memory_id)
        value = row["value_text"]
        if not isinstance(value, str) or not value.strip():
            raise IdentityContractError(f"invalid_value_text:{memory_id}")
        normalized_by_id[memory_id] = {
            "memory_id": memory_id,
            "revision_kind": kind,
            "subject_key": subject,
            "attribute_key": attribute,
            "value_text": value.strip(),
        }
    missing = set(expected_ids) - seen
    if missing:
        raise IdentityContractError(f"missing_memory_ids:{','.join(sorted(missing))}")
    if len(rows) != len(expected_ids):
        raise IdentityContractError("identity_row_count_mismatch")
    return {"identities": [normalized_by_id[memory_id] for memory_id in expected_ids]}


def split_batch(batch: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if len(batch) < 2:
        raise IdentityContractError("IRREDUCIBLE_IDENTITY_OUTPUT_FAILURE")
    middle = len(batch) // 2
    return batch[:middle], batch[middle:]


def execute_with_bisection(
    batch: list[dict[str, str]],
    invoke: Any,
    *,
    depth: int = 0,
    parent_batch_id: str | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    """Invoke once; on a length stop, discard that payload and recurse left then right."""
    try:
        identities, ledger = invoke(batch)
    except IdentityLengthStop as exc:
        attempt = {
            **exc.ledger,
            "status": "OVERFLOW_PARENT",
            "batch_size": len(batch),
            "depth": depth,
            "parent_batch_id": parent_batch_id,
            "semantic_output_used": False,
            "retry_count": 0,
        }
        if len(batch) == 1:
            raise IdentityContractError("IRREDUCIBLE_IDENTITY_OUTPUT_FAILURE") from exc
        left, right = split_batch(batch)
        left_rows, left_attempts = execute_with_bisection(
            left, invoke, depth=depth + 1, parent_batch_id=attempt.get("batch_id")
        )
        right_rows, right_attempts = execute_with_bisection(
            right, invoke, depth=depth + 1, parent_batch_id=attempt.get("batch_id")
        )
        return left_rows + right_rows, [attempt, *left_attempts, *right_attempts]
    attempt = {
        **ledger,
        "status": "COMPLETE",
        "batch_size": len(batch),
        "depth": depth,
        "parent_batch_id": parent_batch_id,
        "semantic_output_used": True,
        "retry_count": 0,
    }
    return identities, [attempt]


def candidate_groups(
    identities: list[dict[str, str]], source_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    source_by_id = {row["memory_id"]: row for row in source_rows}
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for identity in identities:
        if identity["revision_kind"] != "SINGLETON_STATE":
            continue
        source = source_by_id[identity["memory_id"]]
        key = (source["scope_id"], identity["subject_key"], identity["attribute_key"])
        grouped[key].append(
            {
                "memory_id": identity["memory_id"],
                "value_text": identity["value_text"],
            }
        )
    output = []
    for (scope_id, subject_key, attribute_key), observations in sorted(grouped.items()):
        if len(observations) < 2:
            continue
        observations.sort(key=lambda row: row["memory_id"])
        output.append(
            {
                "group_id": sha256_bytes(canonical_json([scope_id, subject_key, attribute_key])),
                "scope_id": scope_id,
                "subject_key": subject_key,
                "attribute_key": attribute_key,
                "proposition_count": len(observations),
                "distinct_value_text_count": len({row["value_text"] for row in observations}),
                "memory_ids": [row["memory_id"] for row in observations],
                "observations": observations,
            }
        )
    return output


def identity_statistics(
    identities: list[dict[str, str]],
    source_rows: list[dict[str, Any]],
    groups: list[dict[str, Any]],
) -> dict[str, Any]:
    counts = Counter(row["revision_kind"] for row in identities)
    source_by_id = {row["memory_id"]: row for row in source_rows}
    singleton_rows = [row for row in identities if row["revision_kind"] == "SINGLETON_STATE"]
    sizes = Counter(
        (source_by_id[row["memory_id"]]["scope_id"], row["subject_key"], row["attribute_key"])
        for row in singleton_rows
    )
    orphan_count = sum(size == 1 for size in sizes.values())
    multi_value = [group for group in groups if group["distinct_value_text_count"] > 1]
    multi_session = [group for group in groups if len(group["source_sessions"]) > 1]
    total = len(identities)
    return {
        "total_propositions": total,
        "counts_by_revision_kind": {kind: counts.get(kind, 0) for kind in REVISION_KINDS},
        "singleton_state_count": counts.get("SINGLETON_STATE", 0),
        "set_state_count": counts.get("SET_STATE", 0),
        "event_count": counts.get("EVENT", 0),
        "non_revisional_count": counts.get("NON_REVISIONAL", 0),
        "unknown_count": counts.get("UNKNOWN", 0),
        "unique_subject_keys": len({row["subject_key"] for row in identities}),
        "unique_attribute_keys": len({row["attribute_key"] for row in identities}),
        "candidate_singleton_groups": len(groups),
        "singleton_groups_size_at_least_two": len(groups),
        "singleton_groups_with_multiple_values": len(multi_value),
        "singleton_groups_spanning_multiple_sessions": len(multi_session),
        "singleton_orphan_count": orphan_count,
        "singleton_orphan_rate": orphan_count / len(singleton_rows) if singleton_rows else 0.0,
        "unknown_rate": counts.get("UNKNOWN", 0) / total if total else 0.0,
        "candidate_groups": groups,
    }


def fragmentation_diagnostics(
    identities: list[dict[str, str]],
    source_rows: list[dict[str, Any]],
    *,
    threshold: float = 0.82,
) -> dict[str, Any]:
    source_by_id = {row["memory_id"]: row for row in source_rows}
    slots: dict[str, dict[tuple[str, str], list[str]]] = defaultdict(lambda: defaultdict(list))
    for row in identities:
        if row["revision_kind"] != "SINGLETON_STATE":
            continue
        scope = source_by_id[row["memory_id"]]["scope_id"]
        slots[scope][(row["subject_key"], row["attribute_key"])].append(row["memory_id"])
    flagged: list[dict[str, Any]] = []
    compared = 0
    for scope_id, slot_map in sorted(slots.items()):
        slot_keys = sorted(slot_map)
        for left, right in combinations(slot_keys, 2):
            left_surface = f"{left[0]} {left[1]}"
            right_surface = f"{right[0]} {right[1]}"
            if len(left_surface) * 0.8 > len(right_surface) or len(right_surface) * 0.8 > len(
                left_surface
            ):
                continue
            matcher = difflib.SequenceMatcher(None, left_surface, right_surface, autojunk=False)
            if matcher.real_quick_ratio() < threshold:
                continue
            compared += 1
            similarity = matcher.ratio()
            if similarity >= threshold:
                flagged.append(
                    {
                        "scope_id": scope_id,
                        "left_subject_key": left[0],
                        "left_attribute_key": left[1],
                        "right_subject_key": right[0],
                        "right_attribute_key": right[1],
                        "surface_similarity": round(similarity, 6),
                        "left_memory_ids": sorted(slot_map[left]),
                        "right_memory_ids": sorted(slot_map[right]),
                        "heuristic": True,
                        "merged": False,
                    }
                )
    flagged.sort(
        key=lambda row: (
            -row["surface_similarity"],
            row["scope_id"],
            row["left_subject_key"],
            row["left_attribute_key"],
            row["right_subject_key"],
            row["right_attribute_key"],
        )
    )
    return {
        "method": "SequenceMatcher over subject_key + attribute_key within the same scope",
        "threshold": threshold,
        "candidate_key_pairs": compared,
        "flagged_pair_count": len(flagged),
        "flagged_pairs": flagged,
        "heuristic": True,
        "automatic_merges": 0,
    }


def _content_tokens(text: str) -> set[str]:
    return {
        token for token in TOKEN_PATTERN.findall(text.casefold()) if token not in GENERIC_TOKENS
    }


def collision_diagnostics(
    groups: list[dict[str, Any]], *, threshold: float = 0.08
) -> dict[str, Any]:
    flagged = []
    for group in groups:
        observations = group["observations"]
        suspicious_pairs = []
        for left, right in combinations(observations, 2):
            if left["value_text"] == right["value_text"]:
                continue
            left_tokens = _content_tokens(left["proposition_text"])
            right_tokens = _content_tokens(right["proposition_text"])
            union = left_tokens | right_tokens
            similarity = len(left_tokens & right_tokens) / len(union) if union else 1.0
            if similarity <= threshold:
                suspicious_pairs.append(
                    {
                        "left_memory_id": left["memory_id"],
                        "right_memory_id": right["memory_id"],
                        "left_proposition_text": left["proposition_text"],
                        "right_proposition_text": right["proposition_text"],
                        "content_token_jaccard": round(similarity, 6),
                        "heuristic": True,
                    }
                )
        if suspicious_pairs:
            suspicious_pairs.sort(
                key=lambda row: (
                    row["content_token_jaccard"],
                    row["left_memory_id"],
                    row["right_memory_id"],
                )
            )
            flagged.append(
                {
                    "group_id": group["group_id"],
                    "scope_id": group["scope_id"],
                    "subject_key": group["subject_key"],
                    "attribute_key": group["attribute_key"],
                    "suspicious_pair_count": len(suspicious_pairs),
                    "pairs": suspicious_pairs,
                    "heuristic": True,
                    "automatically_split": False,
                }
            )
    flagged.sort(
        key=lambda row: (
            -row["suspicious_pair_count"],
            row["scope_id"],
            row["subject_key"],
            row["attribute_key"],
        )
    )
    return {
        "method": "content token Jaccard over proposition text for distinct values in an exact candidate group",
        "threshold_at_or_below": threshold,
        "flagged_group_count": len(flagged),
        "flagged_groups": flagged,
        "heuristic": True,
        "automatic_splits": 0,
    }
