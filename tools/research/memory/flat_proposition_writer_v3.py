"""Minimal proposition packet schema with Harness-derived provenance and authority."""

from __future__ import annotations

import json
from typing import Any

try:
    from .flat_proposition_writer_v2 import (
        ExtractionValidationError,
        canonical_json,
        sha256_bytes,
        writer_payload,
    )
except ImportError:
    from flat_proposition_writer_v2 import (
        ExtractionValidationError,
        canonical_json,
        sha256_bytes,
        writer_payload,
    )


def dynamic_output_schema(catalog: list[dict[str, Any]]) -> dict[str, Any]:
    refs = [span["evidence_ref"] for span in catalog]
    if not refs or len(refs) != len(set(refs)):
        raise ValueError("dynamic_schema_requires_unique_session_refs")
    return {
        "type": "object",
        "required": ["propositions"],
        "additionalProperties": False,
        "properties": {
            "propositions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["proposition_text", "evidence_refs"],
                    "additionalProperties": False,
                    "properties": {
                        "proposition_text": {"type": "string", "minLength": 1},
                        "evidence_refs": {
                            "type": "array",
                            "minItems": 1,
                            "uniqueItems": True,
                            "items": {"type": "string", "enum": refs},
                        },
                    },
                },
            }
        },
    }


def writer_request(
    *, session_date: str, catalog: list[dict[str, Any]], system_prompt: str, model_alias: str
) -> dict[str, Any]:
    return {
        "model": model_alias,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(
                    writer_payload(session_date, catalog),
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                ),
            },
        ],
        "temperature": 0,
        "seed": 42,
        "max_tokens": 4096,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "flat_proposition_packet_v3_minimal",
                "strict": True,
                "schema": dynamic_output_schema(catalog),
            },
        },
    }


def validate_packet(raw_bytes: bytes, catalog: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        payload = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExtractionValidationError("MALFORMED_JSON", detail=str(exc)) from exc
    if not isinstance(payload, dict):
        raise ExtractionValidationError("INVALID_ROOT", field="root", detail="expected object")
    if set(payload) != {"propositions"} or not isinstance(payload.get("propositions"), list):
        extra = set(payload) - {"propositions"}
        raise ExtractionValidationError(
            "UNEXPECTED_OUTPUT_FIELD" if extra else "INVALID_ROOT",
            field=min(extra) if extra else "propositions",
            detail="expected propositions array only",
        )

    spans = {span["evidence_ref"]: span for span in catalog}
    normalized = []
    for index, proposition in enumerate(payload["propositions"]):
        if not isinstance(proposition, dict):
            raise ExtractionValidationError(
                "INVALID_PROPOSITION", proposition_index=index, detail="expected object"
            )
        fields = set(proposition)
        if fields - {"proposition_text", "evidence_refs"}:
            field = min(fields - {"proposition_text", "evidence_refs"})
            raise ExtractionValidationError(
                "UNEXPECTED_OUTPUT_FIELD",
                proposition_index=index,
                field=field,
                detail="field is not in v3 minimal contract",
            )
        missing = {"proposition_text", "evidence_refs"} - fields
        if missing:
            raise ExtractionValidationError(
                "MISSING_FIELD", proposition_index=index, field=min(missing)
            )
        text = proposition["proposition_text"]
        if not isinstance(text, str) or not text.strip():
            raise ExtractionValidationError(
                "EMPTY_PROPOSITION", proposition_index=index, field="proposition_text"
            )
        refs = proposition["evidence_refs"]
        if not isinstance(refs, list) or not refs:
            raise ExtractionValidationError(
                "EMPTY_EVIDENCE_REFS", proposition_index=index, field="evidence_refs"
            )
        if any(not isinstance(ref, str) for ref in refs):
            raise ExtractionValidationError(
                "INVALID_EVIDENCE_REF", proposition_index=index, field="evidence_refs"
            )
        if len(refs) != len(set(refs)):
            duplicate = next(ref for pos, ref in enumerate(refs) if ref in refs[:pos])
            raise ExtractionValidationError(
                "DUPLICATE_EVIDENCE_REF",
                proposition_index=index,
                field="evidence_refs",
                evidence_ref=duplicate,
            )

        evidence = []
        for ref in refs:
            span = spans.get(ref)
            if span is None:
                raise ExtractionValidationError(
                    "UNKNOWN_EVIDENCE_REF",
                    proposition_index=index,
                    field="evidence_refs",
                    evidence_ref=ref,
                )
            content = span["content"]
            actual_sha = sha256_bytes(content.encode("utf-8"))
            if actual_sha != span["content_sha256"]:
                raise ExtractionValidationError(
                    "PROVENANCE_CONTENT_HASH_MISMATCH",
                    proposition_index=index,
                    evidence_ref=ref,
                )
            if span["char_end"] - span["char_start"] != len(content):
                raise ExtractionValidationError(
                    "PROVENANCE_OFFSET_MISMATCH",
                    proposition_index=index,
                    evidence_ref=ref,
                )
            evidence.append(
                {
                    "evidence_ref": ref,
                    "source_turn_index": span["source_turn_index"],
                    "source_span_index": span["source_span_index"],
                    "source_role": span["role"],
                    "char_start": span["char_start"],
                    "char_end": span["char_end"],
                    "evidence_quote": content,
                    "content_sha256": actual_sha,
                }
            )
        roles = {item["source_role"] for item in evidence}
        authority = next(iter(roles)) if len(roles) == 1 else "mixed"
        normalized.append(
            {
                "proposition_text": text.strip(),
                "evidence_refs": refs,
                "evidence": evidence,
                "source_authority": authority,
            }
        )
    return {"propositions": normalized}


def schema_sha256(catalog: list[dict[str, Any]]) -> str:
    return sha256_bytes(canonical_json(dynamic_output_schema(catalog)))
