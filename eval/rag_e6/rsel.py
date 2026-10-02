"""Harness-owned relation extraction for explicit key/value evidence records."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from typing import Any

from eval.rag_e6.reader import EvidenceAlias, evidence_identity_sha256, sha256_text
from eval.rag_e6.split import canonical_json_bytes

QUERY_KEY_PATTERN = re.compile(r"\bSYNKEY-[0-9A-F]{8}\b", re.IGNORECASE)
RELATION_PATTERN = re.compile(
    r"\b(SYNKEY-[0-9A-F]{8})\s+to\s+(SYNVAL-[0-9A-F]{10})\b",
    re.IGNORECASE,
)


def relation_ledger(
    question: str, evidence: Sequence[EvidenceAlias]
) -> tuple[dict[str, str], ...]:
    """Extract only exact key/value relations visible in the issued evidence."""
    if not question.strip():
        raise ValueError("question must be non-empty")
    requested_keys = set(QUERY_KEY_PATTERN.findall(question.upper()))
    if not requested_keys:
        return ()
    result: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in evidence:
        for raw_key, raw_value in RELATION_PATTERN.findall(item.text):
            key, value = raw_key.upper(), raw_value.upper()
            identity = (key, value, item.document_id)
            if key in requested_keys and identity not in seen:
                seen.add(identity)
                result.append({
                    "key": key,
                    "value": value,
                    "alias": item.alias,
                    "document_id": item.document_id,
                })
    return tuple(result)


def rsel_output_row(
    *,
    arm: str,
    question: str,
    evidence: Sequence[EvidenceAlias],
    baseline_row: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a deterministic evidence-bound answer; otherwise preserve Vanilla."""
    if arm not in {"RSEL_STANDARD", "RSEL_STRONG"}:
        raise ValueError("RSEL requires a STANDARD or STRONG evidence arm")
    if baseline_row.get("evidence_identity_sha256") != evidence_identity_sha256(evidence):
        raise ValueError("RSEL baseline and candidate evidence identities differ")

    ledger = relation_ledger(question, evidence)
    row = dict(baseline_row)
    row["arm"] = arm
    row["rsel_ledger"] = list(ledger)
    row["rsel_ledger_sha256"] = hashlib.sha256(
        canonical_json_bytes(list(ledger))
    ).hexdigest()
    row["baseline_answer_sha256"] = baseline_row["answer_sha256"]
    if not ledger:
        row["rsel_action"] = "FALLBACK_NO_MATCH"
        return row

    values = list(dict.fromkeys(item["value"] for item in ledger))
    aliases = list(dict.fromkeys(item["alias"] for item in ledger))
    used_ids = list(dict.fromkeys(item["document_id"] for item in ledger))
    row.update({
        "answer": " ".join((*values, *aliases)),
        "answer_sha256": sha256_text(" ".join((*values, *aliases))),
        "cited_aliases": aliases,
        "unknown_aliases": [],
        "used_evidence_ids": used_ids,
        "output_contract_failure": False,
        "rsel_action": "STRUCTURED_RELATION_LEDGER",
    })
    return row
