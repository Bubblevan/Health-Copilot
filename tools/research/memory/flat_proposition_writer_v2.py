"""Provenance-by-reference writer contract and write-ahead response journal."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MEMORY_KINDS = (
    "user_fact",
    "user_preference",
    "user_event",
    "user_plan",
    "task_decision",
    "assistant_recommendation",
)
USER_ONLY_KINDS = frozenset(
    {"user_fact", "user_preference", "user_event", "user_plan"}
)
KEY_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
FORBIDDEN_FIELDS = frozenset(
    {
        "source_role",
        "source_turn_indices",
        "source_span_indices",
        "evidence_quotes",
        "char_start",
        "char_end",
        "operation",
        "status",
        "current",
        "stale",
        "supersedes",
    }
)
PROPOSITION_FIELDS = frozenset(
    {
        "proposition_text",
        "memory_kind",
        "entity_key_candidate",
        "attribute_key_candidate",
        "value_text",
        "evidence_refs",
    }
)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class ExtractionValidationError(Exception):
    def __init__(
        self,
        code: str,
        *,
        proposition_index: int | None = None,
        field: str | None = None,
        evidence_ref: str | None = None,
        detail: str = "",
    ) -> None:
        self.code = code
        self.proposition_index = proposition_index
        self.field = field
        self.evidence_ref = evidence_ref
        self.detail = detail
        super().__init__(f"{code}: {detail}" if detail else code)

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "proposition_index": self.proposition_index,
            "field": self.field,
            "evidence_ref": self.evidence_ref,
            "detail": self.detail,
        }


class WriterRecoveryError(RuntimeError):
    """A call cannot safely continue without issuing a second model request."""


class WriterQualificationFailure(RuntimeError):
    def __init__(self, ledger: dict[str, Any], error: dict[str, Any] | None = None):
        self.ledger = ledger
        self.error = error
        super().__init__(ledger.get("failure_code", "writer_qualification_failure"))


def build_source_span_catalog(raw_spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Assign session-local references to an already frozen RawSpan partition."""
    ordered = sorted(
        raw_spans,
        key=lambda row: (
            row["source_turn_index"],
            row["source_span_index"],
        ),
    )
    catalog: list[dict[str, Any]] = []
    previous_turn: int | None = None
    expected_span_index = 0
    expected_char_start = 0
    for row in ordered:
        turn_index = row["source_turn_index"]
        span_index = row["source_span_index"]
        role = row["role"]
        content = row["content"]
        char_start = row["char_start"]
        char_end = row["char_end"]
        if type(turn_index) is not int or turn_index < 0:
            raise ValueError("rawspan_turn_index_invalid")
        if type(span_index) is not int or span_index < 0:
            raise ValueError("rawspan_span_index_invalid")
        if role not in {"user", "assistant"}:
            raise ValueError("rawspan_role_invalid")
        if not isinstance(content, str) or not content:
            raise ValueError("rawspan_content_empty_or_invalid")
        if type(char_start) is not int or type(char_end) is not int:
            raise ValueError("rawspan_offsets_not_integer")
        if char_end - char_start != len(content):
            raise ValueError("rawspan_offsets_do_not_match_content_length")
        if turn_index != previous_turn:
            if previous_turn is not None and turn_index <= previous_turn:
                raise ValueError("rawspan_turn_order_invalid")
            previous_turn = turn_index
            expected_span_index = 0
            expected_char_start = 0
        if span_index != expected_span_index:
            raise ValueError("rawspan_span_indices_not_contiguous")
        if char_start != expected_char_start:
            raise ValueError("rawspan_character_partition_not_contiguous")
        evidence_ref = f"S{len(catalog):04d}"
        catalog.append(
            {
                "evidence_ref": evidence_ref,
                "source_turn_index": turn_index,
                "source_span_index": span_index,
                "role": role,
                "char_start": char_start,
                "char_end": char_end,
                "content": content,
                "content_sha256": sha256_bytes(content.encode("utf-8")),
            }
        )
        expected_span_index += 1
        expected_char_start = char_end
    if not catalog:
        raise ValueError("rawspan_catalog_empty")
    return catalog


def catalog_sha256(catalog: list[dict[str, Any]]) -> str:
    return sha256_bytes(canonical_json(catalog))


def writer_payload(session_date: str, catalog: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "session_date": session_date,
        "source_spans": [
            {
                "evidence_ref": span["evidence_ref"],
                "role": span["role"],
                "content": span["content"],
            }
            for span in catalog
        ],
    }


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
                    "required": [
                        "proposition_text",
                        "memory_kind",
                        "entity_key_candidate",
                        "attribute_key_candidate",
                        "value_text",
                        "evidence_refs",
                    ],
                    "additionalProperties": False,
                    "properties": {
                        "proposition_text": {"type": "string", "minLength": 1},
                        "memory_kind": {"type": "string", "enum": list(MEMORY_KINDS)},
                        "entity_key_candidate": {
                            "type": "string",
                            "pattern": KEY_PATTERN.pattern,
                        },
                        "attribute_key_candidate": {
                            "type": "string",
                            "pattern": KEY_PATTERN.pattern,
                        },
                        "value_text": {"type": "string", "minLength": 1},
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
    *,
    session_date: str,
    catalog: list[dict[str, Any]],
    system_prompt: str,
    model_alias: str,
) -> dict[str, Any]:
    payload = writer_payload(session_date, catalog)
    return {
        "model": model_alias,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2),
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
                "name": "flat_proposition_packet_v2",
                "strict": True,
                "schema": dynamic_output_schema(catalog),
            },
        },
    }


def _error(code: str, **kwargs: Any) -> ExtractionValidationError:
    return ExtractionValidationError(code, **kwargs)


def validate_packet(raw_bytes: bytes, catalog: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        text = raw_bytes.decode("utf-8")
        payload = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _error("MALFORMED_JSON", detail=str(exc)) from exc
    if not isinstance(payload, dict):
        raise _error("INVALID_ROOT", field="root", detail="expected object")
    extra_root = set(payload) - {"propositions"}
    if extra_root:
        field = min(extra_root)
        code = "FORBIDDEN_OUTPUT_FIELD" if field in FORBIDDEN_FIELDS else "UNEXPECTED_OUTPUT_FIELD"
        raise _error(code, field=field, detail="unexpected root field")
    if set(payload) != {"propositions"} or not isinstance(payload["propositions"], list):
        raise _error("INVALID_ROOT", field="propositions", detail="expected propositions array only")

    spans = {span["evidence_ref"]: span for span in catalog}
    output: list[dict[str, Any]] = []
    for index, proposition in enumerate(payload["propositions"]):
        if not isinstance(proposition, dict):
            raise _error("INVALID_PROPOSITION", proposition_index=index, detail="expected object")
        extra = set(proposition) - PROPOSITION_FIELDS
        if extra:
            field = min(extra)
            code = "FORBIDDEN_OUTPUT_FIELD" if field in FORBIDDEN_FIELDS else "UNEXPECTED_OUTPUT_FIELD"
            raise _error(code, proposition_index=index, field=field, detail="field is not in v2 contract")
        missing = PROPOSITION_FIELDS - set(proposition)
        if missing:
            field = min(missing)
            raise _error("MISSING_FIELD", proposition_index=index, field=field)
        for field in ("proposition_text", "entity_key_candidate", "attribute_key_candidate", "value_text"):
            value = proposition[field]
            if not isinstance(value, str) or not value.strip():
                raise _error("EMPTY_PROPOSITION", proposition_index=index, field=field)
        for field in ("entity_key_candidate", "attribute_key_candidate"):
            if not KEY_PATTERN.fullmatch(proposition[field]):
                raise _error("INVALID_CANDIDATE_KEY", proposition_index=index, field=field)
        kind = proposition["memory_kind"]
        if kind not in MEMORY_KINDS:
            raise _error("INVALID_MEMORY_KIND", proposition_index=index, field="memory_kind")
        refs = proposition["evidence_refs"]
        if not isinstance(refs, list) or not refs:
            raise _error("EMPTY_EVIDENCE_REFS", proposition_index=index, field="evidence_refs")
        if any(not isinstance(ref, str) for ref in refs):
            raise _error("INVALID_EVIDENCE_REF", proposition_index=index, field="evidence_refs")
        if len(refs) != len(set(refs)):
            duplicate = next(ref for pos, ref in enumerate(refs) if ref in refs[:pos])
            raise _error(
                "DUPLICATE_EVIDENCE_REF",
                proposition_index=index,
                field="evidence_refs",
                evidence_ref=duplicate,
            )
        for ref in refs:
            if ref not in spans:
                raise _error(
                    "UNKNOWN_EVIDENCE_REF",
                    proposition_index=index,
                    field="evidence_refs",
                    evidence_ref=ref,
                )
        evidence: list[dict[str, Any]] = []
        for ref in refs:
            span = spans[ref]
            content = span["content"]
            actual_sha = sha256_bytes(content.encode("utf-8"))
            if actual_sha != span["content_sha256"]:
                raise _error(
                    "PROVENANCE_CONTENT_HASH_MISMATCH",
                    proposition_index=index,
                    evidence_ref=ref,
                )
            if span["char_end"] - span["char_start"] != len(content):
                raise _error(
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
        if kind in USER_ONLY_KINDS and roles != {"user"}:
            raise _error(
                "MEMORY_KIND_ROLE_MISMATCH",
                proposition_index=index,
                field="memory_kind",
                detail=f"{kind} requires user-only evidence; got {sorted(roles)}",
            )
        if kind == "assistant_recommendation" and roles != {"assistant"}:
            raise _error(
                "MEMORY_KIND_ROLE_MISMATCH",
                proposition_index=index,
                field="memory_kind",
                detail="assistant_recommendation requires assistant-only evidence",
            )
        if kind == "task_decision" and "user" not in roles:
            raise _error(
                "MEMORY_KIND_ROLE_MISMATCH",
                proposition_index=index,
                field="memory_kind",
                detail="task_decision requires at least one user evidence span",
            )
        normalized = {
            field: proposition[field]
            for field in (
                "proposition_text",
                "memory_kind",
                "entity_key_candidate",
                "attribute_key_candidate",
                "value_text",
                "evidence_refs",
            )
        }
        normalized["evidence"] = evidence
        output.append(normalized)
    return {"propositions": output}


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


class _Journal:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        events = []
        for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise WriterRecoveryError(f"journal_corrupt_line_{line_number}") from exc
            if not isinstance(row, dict):
                raise WriterRecoveryError(f"journal_invalid_event_{line_number}")
            events.append(row)
        return events

    def append(self, event: dict[str, Any]) -> None:
        encoded = canonical_json(event) + b"\n"
        with self.path.open("ab") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())


def _check_journal(events: list[dict[str, Any]], identity_sha: str) -> None:
    allowed = ["STARTED", "RESPONSE_CAPTURED", "COMPLETE_SUCCESS", "COMPLETE_FAILURE"]
    states = [event.get("state") for event in events]
    if not states or states[0] != "STARTED" or states != allowed[: len(states)]:
        raise WriterRecoveryError("journal_state_transition_invalid")
    if any(event.get("cache_identity_sha256") != identity_sha for event in events):
        raise WriterRecoveryError("journal_cache_identity_mismatch")
    if len(states) > len(allowed) or states[-1] in {"COMPLETE_SUCCESS", "COMPLETE_FAILURE"} and len(states) != 3:
        raise WriterRecoveryError("journal_terminal_state_invalid")


def execute_or_resume(
    *,
    request: dict[str, Any],
    catalog: list[dict[str, Any]],
    session_identity_sha256: str,
    prompt_sha256: str,
    contract_sha256: str,
    local_cache_root: Path,
    provider: Callable[[dict[str, Any]], tuple[int, bytes, str | None]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run once, or recover only from the exact frozen raw response."""
    request_sha = sha256_bytes(canonical_json(request))
    identity_body = {
        "stage": "MEM-3A.1",
        "contract_sha256": contract_sha256,
        "prompt_sha256": prompt_sha256,
        "session_identity_sha256": session_identity_sha256,
        "catalog_sha256": catalog_sha256(catalog),
        "request_sha256": request_sha,
    }
    cache_identity_sha = sha256_bytes(canonical_json(identity_body))
    cache_dir = local_cache_root / contract_sha256 / cache_identity_sha
    journal = _Journal(cache_dir / "journal.jsonl")
    events = journal.read()
    if events:
        _check_journal(events, cache_identity_sha)
        state = events[-1]["state"]
        provider_called_before = True
        if state == "STARTED":
            raise WriterRecoveryError("STARTED_without_response_capture_unknown_outcome")
        if state == "COMPLETE_FAILURE":
            raise WriterQualificationFailure(events[-1].get("ledger", {}), events[-1].get("error"))
        if state == "COMPLETE_SUCCESS":
            packet = events[-1].get("normalized_packet")
            if not isinstance(packet, dict):
                raise WriterRecoveryError("completed_success_missing_packet")
            ledger = dict(events[-1]["ledger"])
            ledger["reused_frozen_result"] = True
            ledger["provider_calls_this_resume"] = 0
            return packet, ledger
        captured = events[-1]
    else:
        provider_called_before = False
        journal.append(
            {
                "state": "STARTED",
                "cache_identity_sha256": cache_identity_sha,
                "request_sha256": request_sha,
                "started_at_utc": datetime.now(UTC).isoformat(),
            }
        )
        try:
            status_code, raw_body, content_type = provider(request)
        except Exception as exc:
            raise WriterRecoveryError(f"provider_failed_after_started:{type(exc).__name__}") from exc
        if not isinstance(raw_body, bytes):
            raise WriterRecoveryError("provider_returned_non_bytes_response")
        raw_sha = sha256_bytes(raw_body)
        raw_path = cache_dir / "raw_response.txt"
        meta_path = cache_dir / "response_meta.json"
        atomic_write_bytes(raw_path, raw_body)
        metadata = {
            "response_sha256": raw_sha,
            "http_status": status_code,
            "content_type": content_type,
            "captured_at_utc": datetime.now(UTC).isoformat(),
        }
        atomic_write_bytes(meta_path, canonical_json(metadata) + b"\n")
        captured = {
            "state": "RESPONSE_CAPTURED",
            "cache_identity_sha256": cache_identity_sha,
            "request_sha256": request_sha,
            "response_sha256": raw_sha,
            "http_status": status_code,
        }
        journal.append(captured)

    raw_path = cache_dir / "raw_response.txt"
    meta_path = cache_dir / "response_meta.json"
    try:
        raw_body = raw_path.read_bytes()
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WriterRecoveryError("captured_response_artifact_missing_or_corrupt") from exc
    raw_sha = sha256_bytes(raw_body)
    if raw_sha != metadata.get("response_sha256") or raw_sha != captured.get("response_sha256"):
        raise WriterRecoveryError("captured_response_hash_mismatch")

    base_ledger = {
        "session_identity_sha256": session_identity_sha256,
        "cache_identity_sha256": cache_identity_sha,
        "request_sha256": request_sha,
        "response_sha256": raw_sha,
        "local_response_retained": True,
        "response_cache_path": str(raw_path),
        "http_status": metadata.get("http_status"),
        "provider_calls": 1,
        "reused_frozen_result": False,
        "provider_calls_this_resume": 0 if provider_called_before else 1,
    }
    if metadata.get("http_status") != 200:
        error = {
            "code": "HTTP_STATUS_FAILURE",
            "proposition_index": None,
            "field": None,
            "evidence_ref": None,
            "detail": f"HTTP {metadata.get('http_status')}",
        }
        ledger = {**base_ledger, "validation": "failed", "failure_code": error["code"]}
        journal.append(
            {
                "state": "COMPLETE_FAILURE",
                "cache_identity_sha256": cache_identity_sha,
                "error": error,
                "ledger": ledger,
            }
        )
        raise WriterQualificationFailure(ledger, error)
    try:
        packet = validate_packet(raw_body, catalog)
    except ExtractionValidationError as exc:
        error = exc.as_dict()
        ledger = {**base_ledger, "validation": "failed", "failure_code": exc.code}
        journal.append(
            {
                "state": "COMPLETE_FAILURE",
                "cache_identity_sha256": cache_identity_sha,
                "error": error,
                "ledger": ledger,
            }
        )
        raise WriterQualificationFailure(ledger, error) from exc
    ledger = {**base_ledger, "validation": "passed", "failure_code": None}
    journal.append(
        {
            "state": "COMPLETE_SUCCESS",
            "cache_identity_sha256": cache_identity_sha,
            "normalized_packet": packet,
            "ledger": ledger,
        }
    )
    return packet, ledger
