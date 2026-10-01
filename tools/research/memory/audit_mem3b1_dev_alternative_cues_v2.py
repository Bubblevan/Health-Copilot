"""Correct v1 cue labels to distinguish change mentions from revision chains."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.research.memory.audit_mem3b1_dev_alternative_cues_v1 import (
    DATASET_PATH_CANDIDATES,
    DATASET_SHA256,
    LABELS_PATH as V1_LABELS_PATH,
    SPLIT_PATH,
    _candidate_rows,
    _sha256,
    _source_sha256,
)
from tools.research.memory.run_mem3b1_dev_revision_pair_diagnostic_v1 import (
    _freeze_artifact,
    _selected_dev_records,
    _source_turns,
)

AMENDMENT_PATH = ROOT / "docs/research/memory/mem3b1_dev_alt_cue_annotation_amendment_v2.json"
OUTPUT = ROOT / "runs/memory/mem3/mem3b1-dev-alternative-cue-audit-v2"
PRIOR_VALUE = re.compile(r"\b3\s*pm\b", re.IGNORECASE)
LABEL_RENAMES = {
    "EXPLICIT_STATE_REVISION": "EXPLICIT_STATE_CHANGE_MENTION",
    "EXPLICIT_TASK_PLAN_REVISION": "EXPLICIT_TASK_PLAN_CHANGE_MENTION",
}


def _amend_annotations(
    annotations: dict[str, Any], amendment: dict[str, Any]
) -> dict[str, Any]:
    amended = json.loads(json.dumps(annotations))
    for row in amended["labels"].values():
        row["label"] = amendment["label_renames"].get(row["label"], row["label"])
        if row["label"] == "EXPLICIT_STATE_CHANGE_MENTION":
            row["rationale"] += " This is a change mention, not proof of an independently anchored predecessor."
        elif row["label"] == "EXPLICIT_TASK_PLAN_CHANGE_MENTION":
            row["rationale"] += " This is a plan-change mention, not proof of an independently anchored predecessor."
    return amended


def _prior_old_value_matches(
    records: dict[str, dict[str, Any]],
    question_id: str,
    source_position: int,
    value_pattern: re.Pattern[str],
) -> list[dict[str, Any]]:
    record = records[question_id]
    return [
        {
            "source_position": turn["source_position"],
            "session_date": turn["session_date"],
            "source_turn_sha256": _source_sha256(str(turn["text"])),
        }
        for turn in _source_turns(record)
        if turn["source_position"] < source_position and value_pattern.search(str(turn["text"]))
    ]


def _score_mentions(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["label"] for row in candidates)
    total = len(candidates)
    if total == 0:
        raise ValueError("cannot score an empty candidate list")
    state_mentions = counts["EXPLICIT_STATE_CHANGE_MENTION"]
    state_or_task_mentions = state_mentions + counts["EXPLICIT_TASK_PLAN_CHANGE_MENTION"]
    return {
        "cue_occurrence_count": total,
        "explicit_adopted_state_change_mentions": state_mentions,
        "explicit_task_plan_change_mentions": counts["EXPLICIT_TASK_PLAN_CHANGE_MENTION"],
        "source_only_change_mention_precision": state_mentions / total,
        "source_only_state_or_task_change_mention_precision": state_or_task_mentions / total,
        "label_counts_by_occurrence": dict(sorted(counts.items())),
        "revision_chain_quality_scored": False,
    }


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"output already exists: {OUTPUT}")
    amendment_bytes = AMENDMENT_PATH.read_bytes()
    amendment = json.loads(amendment_bytes.decode("utf-8"))
    source_label_bytes = V1_LABELS_PATH.read_bytes()
    if _sha256(source_label_bytes) != amendment["source_annotation_sha256"]:
        raise RuntimeError("v1 source annotations differ from the frozen correction amendment")
    annotations = _amend_annotations(json.loads(source_label_bytes), amendment)

    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    dev_ids = set(split["dev"]["question_ids"])
    dataset_path = next((path for path in DATASET_PATH_CANDIDATES if path.is_file()), None)
    if dataset_path is None:
        raise FileNotFoundError("frozen LongMemEval-S data not found")
    records = _selected_dev_records(dataset_path, dev_ids)
    candidates = _candidate_rows(records)
    label_map = annotations["labels"]
    seen_turns = {f"{row['question_id']}:{row['source_position']}" for row in candidates}
    if seen_turns != set(label_map):
        raise RuntimeError("correction labels do not match the frozen candidate turns")
    for row in candidates:
        key = f"{row['question_id']}:{row['source_position']}"
        row.update(label_map[key])
    score = _score_mentions(candidates)
    if score["cue_occurrence_count"] != 37:
        raise RuntimeError(f"expected 37 cue occurrences, got {score['cue_occurrence_count']}")

    anchor_spec = amendment["anchor_check"]
    anchor_matches = _prior_old_value_matches(
        records,
        anchor_spec["question_id"],
        anchor_spec["source_position"],
        PRIOR_VALUE,
    )
    if len(anchor_matches) != anchor_spec["prior_exact_match_count"]:
        raise RuntimeError("the exact predecessor-anchor check differs from the annotation amendment")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    result = {
        "diagnostic_only": True,
        "dataset_sha256": DATASET_SHA256,
        "split_manifest_sha256": _sha256(SPLIT_PATH.read_bytes()),
        "source_v1_label_sha256": _sha256(source_label_bytes),
        "annotation_amendment_sha256": _sha256(amendment_bytes),
        "scope": "17 frozen DEV knowledge-update histories; user-authored source turns only",
        "question_or_gold_decoded": False,
        "model_calls": 0,
        "single_reviewer_annotations": True,
        "score": score,
        "predecessor_anchor_check": {
            "question_id": anchor_spec["question_id"],
            "source_position": anchor_spec["source_position"],
            "old_value_text": anchor_spec["old_value_text"],
            "prior_user_turns_checked": anchor_spec["source_position"],
            "prior_exact_old_value_match_count": len(anchor_matches),
            "matches": anchor_matches,
            "interpretation": "One-case exact-string scan only; not semantic predecessor recall.",
        },
        "candidates": candidates,
        "limitations": [
            "The corrected counts measure source change mentions among cue hits, not valid revision-chain admission.",
            "The exact old-value anchor check covers one case and one spelling only.",
            "Single-reviewer, non-random DEV labels are exploratory and are not benchmark gold.",
            "No end-to-end answer quality or stale-memory suppression was measured.",
        ],
    }
    payload = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    _freeze_artifact(OUTPUT / "cue_audit.json", payload)
    report = (
        "# DEV Alternative-Cue Audit v2: Change Mention vs Revision Chain\n\n"
        "This version corrects the v1 annotation vocabulary. The five personal-state labels are explicit adopted change mentions, not verified revision chains.\n\n"
        f"- Explicit personal-state change mentions: {score['explicit_adopted_state_change_mentions']}/37 "
        f"({score['source_only_change_mention_precision']:.1%} of lexical cue occurrences).\n"
        f"- Including one task-plan change mention: {score['source_only_state_or_task_change_mention_precision']:.1%}.\n"
        "- For the tea-time example, no earlier user turn contained an exact `3 pm` mention; only the current correction sentence did. "
        "Thus the extraction validator's source-grounded same-turn candidate is not by itself evidence for a historical predecessor interval.\n"
        "- Revision-chain accuracy, cue recall, benchmark QA accuracy: not measured. Questions/gold decoded: no; model calls: 0.\n"
    )
    _freeze_artifact(OUTPUT / "report.md", report.encode("utf-8"))
    print(json.dumps({"score": score, "predecessor_anchor_check": result["predecessor_anchor_check"], "output": str(OUTPUT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
