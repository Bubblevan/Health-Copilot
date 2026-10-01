"""Offline atom-level quarantine adapter for R4 candidate proposals.

The adapter keeps the frozen per-atom validators unchanged. It changes only
failure granularity: a rejected atom is quarantined without discarding valid
siblings. It does not write to MemoryStore or infer revisions.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from tools.research.memory import mem3b0q_r4 as frozen_r4
from tools.research.memory import mem3b0q_r4_candidate_binding_guard_v1 as locality_guard
from tools.research.memory import mem3b0q_r4_candidate_builder_v1 as candidate_builder
from tools.research.memory import mem3b0q_r4_joint_binding_guard_v2 as joint_guard_v2
from tools.research.memory import mem3b0q_r4_joint_binding_guard_v3 as joint_guard_v3
from tools.research.memory.mem3b0q_r4_candidate_request_v1 import (
    normalized_atom_signature,
)


ATOMWISE_ADMISSION_VERSION = "mem3b0q-r4-atomwise-admission-v1-development"
_ROOT_KEYS = {"source_id", "atoms", "abstention_reason"}
_ATOM_REJECTION_ERRORS = (
    candidate_builder.CandidateBuildError,
    locality_guard.CandidateBindingError,
    joint_guard_v2.JointBindingError,
    joint_guard_v3.JointBindingError,
)


class AtomwiseAdmissionError(ValueError):
    """The proposal envelope cannot be safely interpreted atom by atom."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AtomwiseAdmissionError(f"duplicate_json_key:{key}")
        result[key] = value
    return result


def _parse_envelope(content: bytes | str, source_id: str) -> dict[str, Any]:
    if not isinstance(content, (bytes, str)):
        raise AtomwiseAdmissionError("content_type_invalid")
    try:
        text = content.decode("utf-8") if isinstance(content, bytes) else content
        proposal = json.loads(text, object_pairs_hook=_unique_object)
    except AtomwiseAdmissionError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
        raise AtomwiseAdmissionError("malformed_json") from exc
    if not isinstance(proposal, dict) or set(proposal) != _ROOT_KEYS:
        raise AtomwiseAdmissionError("root_keys_mismatch")
    if proposal["source_id"] != source_id:
        raise AtomwiseAdmissionError("source_id_mismatch")
    atoms = proposal["atoms"]
    if not isinstance(atoms, list) or len(atoms) > 4:
        raise AtomwiseAdmissionError("atoms_invalid")
    reason = proposal["abstention_reason"]
    if not isinstance(reason, str) or reason not in frozen_r4.ABSTENTION_REASONS:
        raise AtomwiseAdmissionError("abstention_reason_invalid")
    if bool(atoms) != (reason == "NONE"):
        raise AtomwiseAdmissionError("atoms_abstention_inconsistent")
    return proposal


def _one_atom_content(source_id: str, atom: Any) -> bytes:
    return json.dumps(
        {"source_id": source_id, "atoms": [atom], "abstention_reason": "NONE"},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def admit_atoms_independently(
    content: bytes | str,
    proposition: Mapping[str, Any],
    *,
    scope_id: str,
) -> dict[str, Any]:
    """Validate each proposal atom independently and quarantine failures.

    Envelope corruption and abstention inconsistency remain document-level
    failures. Once the envelope is valid, candidate resolution and frozen
    binding errors are isolated to their source atom. Exact duplicate admitted
    atoms are deduplicated, retaining the first occurrence.
    """
    if not isinstance(proposition, Mapping):
        raise AtomwiseAdmissionError("source_record_invalid")
    source_id = proposition.get("source_id")
    source_text = proposition.get("proposition_text")
    if not isinstance(source_id, str) or not source_id:
        raise AtomwiseAdmissionError("source_id_invalid")
    if not isinstance(source_text, str) or not source_text:
        raise AtomwiseAdmissionError("source_text_invalid")
    if not isinstance(scope_id, str) or not scope_id:
        raise AtomwiseAdmissionError("scope_id_invalid")

    proposal = _parse_envelope(content, source_id)
    if not proposal["atoms"]:
        return {
            "version": ATOMWISE_ADMISSION_VERSION,
            "source_id": source_id,
            "scope_id": scope_id,
            "status": "ABSTAINED",
            "source_abstention_reason": proposal["abstention_reason"],
            "admitted": [],
            "quarantined": [],
        }

    admitted: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []
    admitted_signatures: dict[tuple[Any, ...], int] = {}
    for source_atom_index, raw_atom in enumerate(proposal["atoms"]):
        one_atom = _one_atom_content(source_id, raw_atom)
        try:
            normalized = joint_guard_v3.validate_joint_bound_candidate_proposal(
                one_atom, proposition, scope_id=scope_id
            )
        except _ATOM_REJECTION_ERRORS as exc:
            quarantined.append(
                {
                    "source_atom_index": source_atom_index,
                    "reason": f"{type(exc).__name__}:{exc}",
                }
            )
            continue

        atom = normalized["atoms"][0]
        atom["atom_index"] = source_atom_index
        signature = normalized_atom_signature(atom)
        duplicate_of = admitted_signatures.get(signature)
        if duplicate_of is not None:
            quarantined.append(
                {
                    "source_atom_index": source_atom_index,
                    "reason": "duplicate_admitted_atom",
                    "duplicate_of_source_atom_index": duplicate_of,
                }
            )
            continue
        admitted_signatures[signature] = source_atom_index
        admitted.append(
            {"source_atom_index": source_atom_index, "atom": atom}
        )

    if admitted and quarantined:
        status = "PARTIAL"
    elif admitted:
        status = "ACCEPTED"
    else:
        status = "REJECTED"
    return {
        "version": ATOMWISE_ADMISSION_VERSION,
        "source_id": source_id,
        "scope_id": scope_id,
        "status": status,
        "source_abstention_reason": proposal["abstention_reason"],
        "admitted": admitted,
        "quarantined": quarantined,
    }
