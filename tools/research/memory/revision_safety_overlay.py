"""Harness-owned, conservative revision-safety overlay for MEM-3B0S."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping
from typing import Any

AUTHORITY_NAMESPACES = {
    "user": "USER_ASSERTED",
    "mixed": "USER_MIXED",
    "assistant": "ASSISTANT_ORIGIN",
}
ELIGIBLE_AUTHORITIES = frozenset({"USER_ASSERTED", "USER_MIXED"})
ALLOWED_VERDICTS = frozenset(
    {"COHERENT_SINGLETON_SLOT", "INCOHERENT_GROUP", "UNKNOWN"}
)
FORBIDDEN_INPUT_FIELDS = frozenset(
    {
        "observed_at",
        "valid_from",
        "valid_to",
        "valid_until",
        "expires_at",
        "timestamp",
        "session_timestamp",
        "session_date",
        "question_id",
        "question",
        "question_date",
        "gold",
        "answer_session_id",
        "reader_output",
        "retrieval_rank",
    }
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _assistant_owned_subject(subject_key: str) -> bool:
    value = subject_key.strip().casefold()
    return value == "assistant" or value.startswith(
        ("assistant:", "assistant/", "assistant_", "assistant.")
    )


def derive_authority_namespace(source_authority: Any) -> str:
    if not isinstance(source_authority, str):
        return "HARNESS_UNKNOWN_FALLBACK"
    return AUTHORITY_NAMESPACES.get(source_authority, "HARNESS_UNKNOWN_FALLBACK")


def derive_record_eligibility(
    identity: Mapping[str, Any], source: Mapping[str, Any]
) -> dict[str, str]:
    """Assign eligibility without consulting time or benchmark-specific fields."""
    namespace = derive_authority_namespace(source.get("source_authority"))
    kind = identity.get("revision_kind")
    subject = identity.get("subject_key")
    origin = identity.get("identity_origin")

    if origin == "HARNESS_UNKNOWN_FALLBACK":
        eligibility, reason = "REVISION_INELIGIBLE", "UNKNOWN_IDENTITY_FALLBACK"
    elif namespace == "HARNESS_UNKNOWN_FALLBACK":
        eligibility, reason = "REVISION_INELIGIBLE", "UNKNOWN_SOURCE_AUTHORITY"
    elif kind != "SINGLETON_STATE":
        eligibility, reason = "INELIGIBLE_REVISION_KIND", "NON_SINGLETON_REVISION_KIND"
    elif namespace == "ASSISTANT_ORIGIN":
        eligibility, reason = "INELIGIBLE_ASSISTANT_ORIGIN", "ASSISTANT_SOURCE_AUTHORITY"
    elif isinstance(subject, str) and _assistant_owned_subject(subject):
        eligibility, reason = "INELIGIBLE_SUBJECT_NAMESPACE", "ASSISTANT_OWNED_SUBJECT"
    elif namespace in ELIGIBLE_AUTHORITIES and origin == "MODEL_VALIDATED":
        eligibility, reason = "ELIGIBLE_USER_STATE", "ELIGIBLE_AUTHORITY_AND_KIND"
    else:
        eligibility, reason = "REVISION_INELIGIBLE", "UNRECOGNIZED_IDENTITY_OR_AUTHORITY"

    return {
        "authority_namespace": namespace,
        "revision_eligibility": eligibility,
        "eligibility_reason": reason,
    }


def build_candidate_universe(
    identity_rows: list[dict[str, Any]], source_rows: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, int]]:
    """Join frozen inputs and project only post-gate verifier candidates.

    Only explicitly selected fields are read from source rows. Timestamp and
    benchmark metadata may be present in FlatProp, but are never projected.
    """
    identity_ids = [row.get("memory_id") for row in identity_rows]
    source_ids = [row.get("memory_id") for row in source_rows]
    if any(not isinstance(value, str) or not value for value in identity_ids + source_ids):
        raise ValueError("source_memory_id_missing")
    if len(identity_ids) != len(set(identity_ids)):
        raise ValueError("duplicate_identity_memory_id")
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("duplicate_source_memory_id")
    if set(identity_ids) != set(source_ids) or len(identity_ids) != len(source_ids):
        raise ValueError("identity_source_coverage_mismatch")

    source_by_id = {row["memory_id"]: row for row in source_rows}
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    gate_counts: Counter[str] = Counter()
    authority_counts: Counter[str] = Counter()
    retained_authority_counts: Counter[str] = Counter()
    eligible_count = 0
    for identity in identity_rows:
        memory_id = identity["memory_id"]
        source = source_by_id[memory_id]
        gate = derive_record_eligibility(identity, source)
        gate_counts[gate["revision_eligibility"]] += 1
        authority_counts[gate["authority_namespace"]] += 1
        if gate["revision_eligibility"] != "ELIGIBLE_USER_STATE":
            continue
        retained_authority_counts[gate["authority_namespace"]] += 1

        scope_id = source.get("scope_id")
        subject_key = identity.get("subject_key")
        attribute_key = identity.get("attribute_key")
        proposition_text = source.get("proposition_text")
        value_text = identity.get("value_text")
        source_authority = source.get("source_authority")
        if not all(
            isinstance(value, str) and value
            for value in (
                scope_id,
                subject_key,
                attribute_key,
                proposition_text,
                value_text,
                source_authority,
            )
        ):
            raise ValueError(f"eligible_record_missing_required_projection:{memory_id}")
        eligible_count += 1
        groups[(scope_id, subject_key, attribute_key)].append(
            {
                "memory_id": memory_id,
                "proposition_text": proposition_text,
                "value_text": value_text,
                "source_authority": source_authority,
            }
        )

    repeated_groups = []
    singleton_groups = []
    for (scope_id, subject_key, attribute_key), members in sorted(groups.items()):
        member_rows = sorted(members, key=lambda row: row["memory_id"])
        group_id = sha256_bytes(
            canonical_json_bytes([scope_id, subject_key, attribute_key])
        )
        if len(member_rows) == 1:
            singleton_groups.append(
                {
                    "group_id": group_id,
                    "scope_id": scope_id,
                    "subject_key": subject_key,
                    "attribute_key": attribute_key,
                    "member_memory_ids": [member_rows[0]["memory_id"]],
                }
            )
            continue
        repeated_groups.append(
            {
                "group_id": group_id,
                "scope_id": scope_id,
                "subject_key": subject_key,
                "attribute_key": attribute_key,
                "member_memory_ids": [row["memory_id"] for row in member_rows],
                "propositions": member_rows,
            }
        )

    universe = {
        "schema_version": 1,
        "grouping_key": ["scope_id", "subject_key", "attribute_key"],
        "singletons_without_revision_history": sorted(singleton_groups, key=lambda row: row["group_id"]),
        "repeated_candidate_groups": repeated_groups,
    }
    stats = {
        "source_propositions": len(source_rows),
        "identity_records": len(identity_rows),
        "eligible_user_state_records": eligible_count,
        "authority_gate_counts": dict(sorted(gate_counts.items())),
        "authority_namespace_counts": dict(sorted(authority_counts.items())),
        "eligible_records_by_authority_namespace": dict(sorted(retained_authority_counts.items())),
        "assistant_origin_records_excluded_from_user_state": authority_counts.get("ASSISTANT_ORIGIN", 0)
        - retained_authority_counts.get("ASSISTANT_ORIGIN", 0),
        "user_asserted_records_retained": retained_authority_counts.get("USER_ASSERTED", 0),
        "user_mixed_records_retained": retained_authority_counts.get("USER_MIXED", 0),
        "eligible_exact_groups": len(groups),
        "singletons_without_revision_history": len(singleton_groups),
        "repeated_candidate_groups": len(repeated_groups),
        "repeated_candidate_members": sum(
            len(row["member_memory_ids"]) for row in repeated_groups
        ),
    }
    assert_no_forbidden_fields(universe)
    return universe, stats


def assert_no_forbidden_fields(value: Any) -> None:
    if isinstance(value, Mapping):
        forbidden = FORBIDDEN_INPUT_FIELDS.intersection(value.keys())
        if forbidden:
            raise ValueError(f"forbidden_field_in_verifier_projection:{sorted(forbidden)}")
        for child in value.values():
            assert_no_forbidden_fields(child)
    elif isinstance(value, list):
        for child in value:
            assert_no_forbidden_fields(child)


def build_verifier_request(group: Mapping[str, Any]) -> dict[str, Any]:
    request = {
        "subject_key": group["subject_key"],
        "attribute_key": group["attribute_key"],
        "propositions": [
            {
                "memory_id": row["memory_id"],
                "proposition_text": row["proposition_text"],
                "value_text": row["value_text"],
                "source_authority": row["source_authority"],
            }
            for row in group["propositions"]
        ],
    }
    assert_no_forbidden_fields(request)
    return request


def parse_verifier_output(content: Any, finish_reason: Any) -> tuple[str | None, str | None]:
    if finish_reason == "length":
        return None, "COMPLETION_TRUNCATED"
    if finish_reason not in {None, "stop"}:
        return None, "UNEXPECTED_FINISH_REASON"
    if not isinstance(content, str):
        return None, "MALFORMED_ASSISTANT_CONTENT"

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError("DUPLICATE_JSON_OBJECT_KEY")
            output[key] = value
        return output

    try:
        payload = json.loads(content, object_pairs_hook=unique_object)
    except (json.JSONDecodeError, UnicodeError):
        return None, "MALFORMED_ASSISTANT_JSON"
    except ValueError as exc:
        return None, str(exc) if str(exc) == "DUPLICATE_JSON_OBJECT_KEY" else "MALFORMED_ASSISTANT_JSON"
    if not isinstance(payload, dict) or set(payload) != {"verdict"}:
        return None, "UNEXPECTED_OUTPUT_FIELDS"
    verdict = payload.get("verdict")
    if not isinstance(verdict, str) or verdict not in ALLOWED_VERDICTS:
        return None, "ILLEGAL_VERDICT"
    return verdict, None


def materialization_status(verdict: str | None, failure_code: str | None = None) -> str:
    if failure_code:
        return "MATERIALIZATION_BLOCKED_VERIFIER_FAILURE"
    mapping = {
        "COHERENT_SINGLETON_SLOT": "MATERIALIZATION_SAFE",
        "INCOHERENT_GROUP": "MATERIALIZATION_BLOCKED_SEMANTIC_COLLISION",
        "UNKNOWN": "MATERIALIZATION_BLOCKED_UNKNOWN",
    }
    if verdict not in mapping:
        raise ValueError("verifier_terminal_state_missing")
    return mapping[verdict]


def safety_overlay_record(
    group: Mapping[str, Any],
    verdict: str | None,
    *,
    failure_code: str | None = None,
    oversized: bool = False,
) -> dict[str, Any]:
    if oversized:
        status = "MATERIALIZATION_BLOCKED_OVERSIZED_GROUP"
    else:
        status = materialization_status(verdict, failure_code)
    return {
        "group_id": group["group_id"],
        "scope_id": group["scope_id"],
        "subject_key": group["subject_key"],
        "attribute_key": group["attribute_key"],
        "member_memory_ids": list(group["member_memory_ids"]),
        "authority_gate": "PASS",
        "revision_kind_gate": "PASS",
        "semantic_verifier": verdict if verdict is not None else "NOT_RUN",
        "verifier_failure_code": failure_code,
        "materialization_status": status,
    }
