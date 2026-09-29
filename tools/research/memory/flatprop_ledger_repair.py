"""Audit-only repairs for persisted FlatProp writer attempt metadata."""

from __future__ import annotations

from typing import Any

from tools.research.memory.flatprop_recursive_recovery import recursive_child_identity


def restore_source_session_identity(
    attempt: dict[str, Any], root_session_ids: dict[str, str]
) -> dict[str, Any]:
    """Restore semantic session attribution while retaining the chunk cache identity."""
    root_id = attempt.get("root_initial_chunk_id")
    chunk_id = attempt.get("chunk_id")
    source_session_id = root_session_ids.get(root_id)
    if not isinstance(root_id, str) or not isinstance(chunk_id, str) or not source_session_id:
        raise ValueError("attempt_source_session_identity_cannot_be_resolved")
    depth = attempt.get("recursion_depth", 0)
    if depth == 0:
        if chunk_id != root_id:
            raise ValueError("initial_attempt_chunk_must_match_root_identity")
    else:
        body = attempt.get("chunk_identity")
        if not isinstance(body, dict):
            raise ValueError("recursive_attempt_identity_body_missing")
        if (
            body.get("root_initial_chunk_id") != root_id
            or body.get("session_identity_sha256") != source_session_id
        ):
            raise ValueError("recursive_attempt_source_session_identity_mismatch")
        computed_id, _ = recursive_child_identity(body)
        if computed_id != chunk_id:
            raise ValueError("recursive_attempt_chunk_identity_hash_mismatch")
    restored = dict(attempt)
    restored["session_identity_sha256_provider_cache_key"] = attempt.get(
        "session_identity_sha256"
    )
    restored["session_identity_sha256"] = source_session_id
    restored["session_identity_sha256_for_cache"] = chunk_id
    restored["session_identity_repair_basis"] = "root_initial_chunk_id_to_frozen_initial_manifest"
    return restored
