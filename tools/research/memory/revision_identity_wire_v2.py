"""Keyed-map identity wire contract and conservative local batch recovery."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from tools.research.memory import revision_identity as semantic_v1

WIRE_CONTRACT_ID = "revision-identity-wire-v2-keyed-map"
VALIDATOR_ID = "revision-identity-validator-v2-keyed-map"
LOCAL_FAILURE_CODES = frozenset(
    {
        "DUPLICATE_JSON_OBJECT_KEY",
        "MALFORMED_ASSISTANT_JSON",
        "INVALID_IDENTITY_ROOT",
        "IDENTITIES_NOT_OBJECT",
        "MISSING_IDENTITY_KEY",
        "UNEXPECTED_IDENTITY_KEY",
        "IDENTITY_PAYLOAD_NOT_OBJECT",
        "MISSING_IDENTITY_FIELD",
        "UNEXPECTED_IDENTITY_FIELD",
        "ILLEGAL_REVISION_KIND",
        "INVALID_SUBJECT_KEY",
        "INVALID_ATTRIBUTE_KEY",
        "EMPTY_VALUE_TEXT",
        "COMPLETION_TRUNCATED",
    }
)


class KeyedIdentityContractError(ValueError):
    def __init__(self, code: str, detail: str | None = None):
        self.code = code
        self.detail = detail or code
        super().__init__(self.detail)


class DuplicateJsonObjectKey(ValueError):
    def __init__(self, key: str):
        self.key = key
        super().__init__(key)


class LocalBatchFailure(RuntimeError):
    def __init__(self, code: str, ledger: dict[str, Any], detail: str | None = None):
        self.code = code
        self.ledger = ledger
        self.detail = detail or code
        super().__init__(f"{code}:{self.detail}")


class GlobalIdentityFailure(RuntimeError):
    """A failure that invalidates the stage rather than one model batch."""


def _batch_ids(batch: list[dict[str, str]]) -> list[str]:
    ids = [row["memory_id"] for row in batch]
    if not ids or len(ids) != len(set(ids)):
        raise GlobalIdentityFailure("input_batch_memory_ids_not_unique")
    return ids


def response_schema(memory_ids: list[str]) -> dict[str, Any]:
    if not memory_ids or len(memory_ids) != len(set(memory_ids)):
        raise GlobalIdentityFailure("response_schema_requires_unique_batch_ids")
    identity_payload = {
        "type": "object",
        "required": ["revision_kind", "subject_key", "attribute_key", "value_text"],
        "additionalProperties": False,
        "properties": {
            "revision_kind": {"type": "string", "enum": list(semantic_v1.REVISION_KINDS)},
            "subject_key": {"type": "string", "minLength": 1},
            "attribute_key": {"type": "string", "minLength": 1},
            "value_text": {"type": "string", "minLength": 1},
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
                "properties": {memory_id: identity_payload for memory_id in memory_ids},
            }
        },
    }


def build_request(batch: list[dict[str, str]], *, model: str, max_tokens: int) -> dict[str, Any]:
    projected = [semantic_v1.project_input_row(row) for row in batch]
    ids = _batch_ids(projected)
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": semantic_v1.SYSTEM_PROMPT},
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
                "name": "revision_identity_v2_keyed_map",
                "strict": True,
                "schema": response_schema(ids),
            },
        },
    }


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise DuplicateJsonObjectKey(key)
        output[key] = value
    return output


def validate_response(raw_bytes: bytes, batch: list[dict[str, str]]) -> list[dict[str, str | None]]:
    expected_ids = _batch_ids(batch)
    try:
        payload = json.loads(raw_bytes.decode("utf-8"), object_pairs_hook=_unique_object)
    except DuplicateJsonObjectKey as exc:
        raise KeyedIdentityContractError("DUPLICATE_JSON_OBJECT_KEY", exc.key) from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise KeyedIdentityContractError("MALFORMED_ASSISTANT_JSON") from exc

    if not isinstance(payload, dict) or set(payload) != {"identities"}:
        raise KeyedIdentityContractError("INVALID_IDENTITY_ROOT")
    identities = payload["identities"]
    if not isinstance(identities, dict):
        raise KeyedIdentityContractError("IDENTITIES_NOT_OBJECT")
    expected_set = set(expected_ids)
    actual_set = set(identities)
    if expected_set - actual_set:
        raise KeyedIdentityContractError(
            "MISSING_IDENTITY_KEY", ",".join(sorted(expected_set - actual_set))
        )
    if actual_set - expected_set:
        raise KeyedIdentityContractError(
            "UNEXPECTED_IDENTITY_KEY", ",".join(sorted(actual_set - expected_set))
        )

    records = []
    for memory_id in expected_ids:
        row = identities[memory_id]
        if not isinstance(row, dict):
            raise KeyedIdentityContractError("IDENTITY_PAYLOAD_NOT_OBJECT", memory_id)
        required = {"revision_kind", "subject_key", "attribute_key", "value_text"}
        if required - set(row):
            raise KeyedIdentityContractError("MISSING_IDENTITY_FIELD", memory_id)
        if set(row) - required:
            raise KeyedIdentityContractError("UNEXPECTED_IDENTITY_FIELD", memory_id)
        kind = row["revision_kind"]
        if not isinstance(kind, str) or kind not in semantic_v1.REVISION_KINDS:
            raise KeyedIdentityContractError("ILLEGAL_REVISION_KIND", memory_id)
        try:
            subject = semantic_v1.normalize_key(
                row["subject_key"], field="subject_key", memory_id=memory_id
            )
        except semantic_v1.IdentityContractError as exc:
            raise KeyedIdentityContractError("INVALID_SUBJECT_KEY", f"{memory_id}:{exc}") from exc
        try:
            attribute = semantic_v1.normalize_key(
                row["attribute_key"], field="attribute_key", memory_id=memory_id
            )
        except semantic_v1.IdentityContractError as exc:
            raise KeyedIdentityContractError("INVALID_ATTRIBUTE_KEY", f"{memory_id}:{exc}") from exc
        value = row["value_text"]
        if not isinstance(value, str) or not value.strip():
            raise KeyedIdentityContractError("EMPTY_VALUE_TEXT", memory_id)
        records.append(
            {
                "memory_id": memory_id,
                "revision_kind": kind,
                "subject_key": subject,
                "attribute_key": attribute,
                "value_text": value.strip(),
                "identity_origin": "MODEL_VALIDATED",
                "fallback_reason": None,
            }
        )
    return records


def unknown_fallback(memory: dict[str, str], reason: str) -> dict[str, str | None]:
    return {
        "memory_id": memory["memory_id"],
        "revision_kind": "UNKNOWN",
        "subject_key": "unknown",
        "attribute_key": "unknown",
        "value_text": memory["proposition_text"],
        "identity_origin": "HARNESS_UNKNOWN_FALLBACK",
        "fallback_reason": reason,
    }


def is_candidate_singleton(record: dict[str, Any]) -> bool:
    return (
        record.get("identity_origin") == "MODEL_VALIDATED"
        and record.get("revision_kind") == "SINGLETON_STATE"
    )


def batch_id(batch: list[dict[str, str]]) -> str:
    return semantic_v1.sha256_bytes(semantic_v1.canonical_json([row["memory_id"] for row in batch]))


def split_batch(batch: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if len(batch) < 2:
        raise GlobalIdentityFailure("cannot_subdivide_single_item_batch")
    middle = len(batch) // 2
    return batch[:middle], batch[middle:]


def recover_batch(
    batch: list[dict[str, str]],
    invoke: Callable[[list[dict[str, str]]], tuple[list[dict[str, Any]], dict[str, Any]]],
    *,
    depth: int = 0,
    parent_batch_id: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    current_id = batch_id(batch)
    try:
        records, ledger = invoke(batch)
    except LocalBatchFailure as exc:
        call_record = {
            **exc.ledger,
            "batch_id": current_id,
            "parent_batch_id": parent_batch_id,
            "depth": depth,
            "failure_code": exc.code,
            "failure_detail": exc.detail,
            "input_memory_ids": [row["memory_id"] for row in batch],
            "terminal_identities": False,
            "retry_count": 0,
        }
        if len(batch) == 1:
            memory = batch[0]
            record = unknown_fallback(memory, exc.code)
            lineage = {
                "batch_id": current_id,
                "parent_batch_id": parent_batch_id,
                "depth": depth,
                "failure_code": exc.code,
                "input_memory_ids": [memory["memory_id"]],
                "child_batch_ids": [],
                "outcome": "HARNESS_UNKNOWN_FALLBACK",
                "terminal_identity_ids": [memory["memory_id"]],
                "retry_count": 0,
            }
            return [record], [call_record], [lineage]
        left, right = split_batch(batch)
        child_ids = [batch_id(left), batch_id(right)]
        lineage = {
            "batch_id": current_id,
            "parent_batch_id": parent_batch_id,
            "depth": depth,
            "failure_code": exc.code,
            "input_memory_ids": [row["memory_id"] for row in batch],
            "child_batch_ids": child_ids,
            "outcome": "LOCAL_FAILURE_SUBDIVIDED",
            "terminal_identity_ids": [],
            "retry_count": 0,
        }
        left_rows, left_calls, left_lineage = recover_batch(
            left,
            invoke,
            depth=depth + 1,
            parent_batch_id=current_id,
        )
        right_rows, right_calls, right_lineage = recover_batch(
            right,
            invoke,
            depth=depth + 1,
            parent_batch_id=current_id,
        )
        return (
            left_rows + right_rows,
            [call_record, *left_calls, *right_calls],
            [lineage, *left_lineage, *right_lineage],
        )

    ids = [row["memory_id"] for row in batch]
    returned_ids = [row.get("memory_id") for row in records]
    if returned_ids != ids:
        raise GlobalIdentityFailure(f"invoke_returned_noncanonical_or_incomplete_ids:{current_id}")
    call_record = {
        **ledger,
        "batch_id": current_id,
        "parent_batch_id": parent_batch_id,
        "depth": depth,
        "failure_code": None,
        "input_memory_ids": ids,
        "terminal_identities": True,
        "retry_count": 0,
    }
    lineage = {
        "batch_id": current_id,
        "parent_batch_id": parent_batch_id,
        "depth": depth,
        "failure_code": None,
        "input_memory_ids": ids,
        "child_batch_ids": [],
        "outcome": "MODEL_VALIDATED",
        "terminal_identity_ids": ids,
        "retry_count": 0,
    }
    return records, [call_record], [lineage]


def require_exact_terminal_coverage(input_ids: list[str], records: list[dict[str, Any]]) -> None:
    output_ids = [row.get("memory_id") for row in records]
    if len(input_ids) != len(set(input_ids)):
        raise GlobalIdentityFailure("frozen_input_memory_ids_not_unique")
    if len(output_ids) != len(set(output_ids)):
        raise GlobalIdentityFailure("terminal_identity_ids_not_unique")
    if set(input_ids) != set(output_ids) or len(input_ids) != len(output_ids):
        raise GlobalIdentityFailure("terminal_identity_coverage_incomplete")
