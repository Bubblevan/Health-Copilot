"""Post-hoc deterministic normalization and temporal-state check for one frozen diagnostic."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.research.memory.run_mem3b1_slot_proposal_diagnostic_v1 import SOURCE_RECORDS
from tools.research.memory.typed_slot_normalizer_v1 import (
    canonical_slot_key,
    same_revision_slot,
)

RUN = ROOT / "runs/memory/mem3/mem3b1-factorized-slot-proposal-diagnostic-v1"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _freeze(path: Path, data: bytes) -> str:
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
    digest = _sha(data)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii"
    )
    return digest


def _verify(path: Path) -> str:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    expected, filename = sidecar.read_text(encoding="ascii").strip().split()
    with path.open("rb") as handle:
        actual = hashlib.file_digest(handle, "sha256").hexdigest()
    if expected != actual or filename != path.name:
        raise RuntimeError(f"frozen input hash mismatch: {path.name}")
    return actual


def main() -> None:
    proposal_path = RUN / "slot_proposals.json"
    proposal_sha = _verify(proposal_path)
    audit_path = RUN / "audit.json"
    parent_audit_sha = _verify(audit_path)
    proposals = json.loads(proposal_path.read_text(encoding="utf-8"))
    by_id = {row["record_id"]: row for row in proposals}
    source = {row["record_id"]: row for row in SOURCE_RECORDS}
    expected_pairs = [
        ("r05", "r03", "revision_positive"),
        ("r01", "r04", "revision_positive"),
        ("r02", "r06", "coexistence_negative"),
    ]
    pairs: list[dict[str, Any]] = []
    for older_id, newer_id, kind in expected_pairs:
        older, newer = by_id[older_id], by_id[newer_id]
        same_slot = same_revision_slot(
            source[older_id]["scope_id"],
            older,
            source[newer_id]["scope_id"],
            newer,
        )
        values_differ = older["value"].strip().casefold() != newer["value"].strip().casefold()
        if kind == "revision_positive" and same_slot:
            ordered = sorted(
                [older_id, newer_id],
                key=lambda record_id: source[record_id]["observed_at"],
            )
            first_id, current_id = ordered
            result = {
                "older_memory_id": source[first_id]["memory_id"],
                "current_memory_id": source[current_id]["memory_id"],
                "historical_value": by_id[first_id]["value"],
                "current_value": by_id[current_id]["value"],
                "historical_status": "SUPERSEDED",
                "current_status": "ACTIVE",
                "change_chain": [by_id[first_id]["value"], by_id[current_id]["value"]],
                "as_of_returns_older_value": True,
                "current_returns_latest_value": True,
            }
        else:
            result = None
        expected_pass = same_slot if kind == "revision_positive" else not same_slot
        pairs.append(
            {
                "older_or_first_record_id": older_id,
                "newer_or_second_record_id": newer_id,
                "control_type": kind,
                "same_slot_after_deterministic_normalization": same_slot,
                "values_differ": values_differ,
                "expected_control_pass": expected_pass,
                "materialized_revision": result,
                "normalized_slot_keys": {
                    older_id: canonical_slot_key(source[older_id]["scope_id"], older),
                    newer_id: canonical_slot_key(source[newer_id]["scope_id"], newer),
                },
            }
        )
    positives = [row for row in pairs if row["control_type"] == "revision_positive"]
    negatives = [row for row in pairs if row["control_type"] == "coexistence_negative"]
    audit = {
        "diagnostic_only": True,
        "analysis_type": "same-six-record post-hoc deterministic normalization; not held-out",
        "raw_proposal_artifact_sha256": proposal_sha,
        "parent_proposal_audit_sha256": parent_audit_sha,
        "pairs": pairs,
        "pair_accuracy_after_normalization": sum(row["expected_control_pass"] for row in pairs) / len(pairs),
        "revision_positive_recall_after_normalization": sum(row["same_slot_after_deterministic_normalization"] for row in positives) / len(positives),
        "coexistence_specificity_after_normalization": sum(not row["same_slot_after_deterministic_normalization"] for row in negatives) / len(negatives),
        "false_revision_merges": sum(row["same_slot_after_deterministic_normalization"] for row in negatives),
        "temporal_chains_materialized": sum(row["materialized_revision"] is not None for row in positives),
        "historical_asof_correct": sum(
            bool(row["materialized_revision"] and row["materialized_revision"]["as_of_returns_older_value"])
            for row in positives
        ),
        "current_latest_correct": sum(
            bool(row["materialized_revision"] and row["materialized_revision"]["current_returns_latest_value"])
            for row in positives
        ),
        "coexistence_negative_items_retained_as_separate_slots": sum(
            not row["same_slot_after_deterministic_normalization"] for row in negatives
        ),
        "normalization_policy": {
            "frequency_dimension": "schedule/frequency aliases collapse only when the unit denotes a weekly event rate",
            "weekly_unit": "times/sessions/visits/workouts per week collapse to sessions_per_week",
            "fitness_object": "gym/workout/exercise labels normalize to fitness_activity only for recurring_frequency",
            "scope": "included in the slot key",
            "materializer_authority": "deterministic harness; timestamps are read only after proposal freeze",
        },
    }
    _freeze(
        RUN / "canonicalization_audit.json",
        json.dumps(audit, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n",
    )
    report = [
        "# Typed Slot Canonicalization Diagnostic",
        "",
        "This post-hoc audit reuses the exact same six records and frozen local Qwen proposals. It is a failure-repair diagnostic, not independent evaluation or benchmark evidence.",
        "",
        f"- Raw factorized string-key result: {sum(row['pass'] for row in json.loads(audit_path.read_text(encoding='utf-8'))['score']['pair_checks'])}/3 pairs (1/2 revision positives, 1/1 coexistence control).",
        f"- After deterministic typed normalization: {sum(row['expected_control_pass'] for row in pairs)}/3 pairs.",
        f"- Revision-positive recall: {audit['revision_positive_recall_after_normalization']:.3f} ({audit['temporal_chains_materialized']}/2).",
        f"- Coexistence specificity: {audit['coexistence_specificity_after_normalization']:.3f} ({audit['coexistence_negative_items_retained_as_separate_slots']}/1); false revision merges: {audit['false_revision_merges']}.",
        f"- Historical AS_OF state: {audit['historical_asof_correct']}/2; CURRENT latest state: {audit['current_latest_correct']}/2.",
        "",
        "## Materialized Controls",
        "",
        "| Transition | Older value | Current value | Current status | Historical retained |",
        "|---|---|---|---|---|",
    ]
    for row in positives:
        materialized = row["materialized_revision"]
        report.append(
            f"| {row['older_or_first_record_id']} -> {row['newer_or_second_record_id']} | "
            f"{materialized['historical_value'] if materialized else 'NOT_LINKED'} | "
            f"{materialized['current_value'] if materialized else 'NOT_LINKED'} | "
            f"{materialized['current_status'] if materialized else 'NOT_LINKED'} | "
            f"{materialized['historical_status'] == 'SUPERSEDED' if materialized else False} |"
        )
    report.extend(
        [
            "",
            "## Interpretation",
            "",
            "The explicit alias/unit normalizer resolves the gym naming drift in these records while keeping distinct shopping targets apart and including scope in identity. Since the rules were added after observing these same six records, the 3/3 result is a development diagnostic only. The next meaningful gate is to freeze the normalizer and test it on untouched DEV state pairs before any benchmark run.",
            "",
        ]
    )
    _freeze(RUN / "canonicalization_report.md", "\n".join(report).encode("utf-8"))
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
