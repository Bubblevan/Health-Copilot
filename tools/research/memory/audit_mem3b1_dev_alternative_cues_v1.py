"""Triage broad replacement-language cues in frozen DEV source histories."""

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
from tools.research.memory.run_mem3b1_dev_revision_pair_diagnostic_v1 import (
    DATASET_SHA256,
    DATASET_PATH_CANDIDATES,
    SPLIT_PATH,
    _freeze_artifact,
    _selected_dev_records,
    _source_turns,
)

LABELS_PATH = ROOT / "docs/research/memory/mem3b1_dev_alt_cue_manual_labels_v1.json"
OUTPUT = ROOT / "runs/memory/mem3/mem3b1-dev-alternative-cue-audit-v1"
ALT_CUE = re.compile(r"\b(?:instead of|rather than)\b", re.IGNORECASE)
REVISION_LABELS = {"EXPLICIT_STATE_REVISION", "EXPLICIT_TASK_PLAN_REVISION"}


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _source_sha256(text: str) -> str:
    return _sha256(text.encode("utf-8"))


def _candidate_rows(records: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for question_id, record in records.items():
        if record.get("question_type") != "knowledge-update":
            continue
        for turn in _source_turns(record):
            text = str(turn["text"])
            for match_index, match in enumerate(ALT_CUE.finditer(text)):
                start = max(0, match.start() - 90)
                end = min(len(text), match.end() + 90)
                rows.append(
                    {
                        "question_id": question_id,
                        "source_position": turn["source_position"],
                        "session_date": turn["session_date"],
                        "match_index_in_turn": match_index,
                        "match_span": match.group(0),
                        "source_turn_sha256": _source_sha256(text),
                        "context_snippet": text[start:end].replace("\n", " "),
                    }
                )
    return sorted(
        rows,
        key=lambda row: (
            row["question_id"], row["source_position"], row["match_index_in_turn"]
        ),
    )


def attach_manual_labels(
    candidates: list[dict[str, Any]], label_map: dict[str, dict[str, str]]
) -> list[dict[str, Any]]:
    seen_keys: set[str] = set()
    labeled = []
    for row in candidates:
        key = f"{row['question_id']}:{row['source_position']}"
        annotation = label_map.get(key)
        if annotation is None:
            raise ValueError(f"candidate source turn lacks a manual label: {key}")
        if not annotation.get("label") or not annotation.get("rationale"):
            raise ValueError(f"manual annotation is incomplete: {key}")
        seen_keys.add(key)
        labeled.append({**row, **annotation})
    if seen_keys != set(label_map):
        raise ValueError(f"manual labels do not match candidate turns: {sorted(set(label_map) - seen_keys)}")
    return labeled


def score_candidate_labels(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["label"] for row in candidates)
    total = len(candidates)
    if total == 0:
        raise ValueError("cannot score an empty candidate list")
    state_revision_count = counts["EXPLICIT_STATE_REVISION"]
    all_revision_count = sum(counts[label] for label in REVISION_LABELS)
    return {
        "cue_occurrence_count": total,
        "unique_source_turn_count": len(
            {(row["question_id"], row["source_position"]) for row in candidates}
        ),
        "label_counts_by_occurrence": dict(sorted(counts.items())),
        "adopted_personal_state_revision_precision": state_revision_count / total,
        "state_or_task_revision_precision": all_revision_count / total,
        "revision_label_scope": "single-reviewer exploratory precision among lexical cue hits; recall is not measured",
    }


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"output already exists: {OUTPUT}")
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    dev_ids = set(split["dev"]["question_ids"])
    dataset_path = next((path for path in DATASET_PATH_CANDIDATES if path.is_file()), None)
    if dataset_path is None:
        raise FileNotFoundError("frozen LongMemEval-S data not found")
    records = _selected_dev_records(dataset_path, dev_ids)
    knowledge_update_count = sum(
        record.get("question_type") == "knowledge-update" for record in records.values()
    )
    if knowledge_update_count != 17:
        raise RuntimeError(f"expected 17 DEV knowledge-update histories, found {knowledge_update_count}")

    annotation_bytes = LABELS_PATH.read_bytes()
    annotations = json.loads(annotation_bytes.decode("utf-8"))
    candidates = attach_manual_labels(_candidate_rows(records), annotations["labels"])
    score = score_candidate_labels(candidates)
    if score["cue_occurrence_count"] != 37:
        raise RuntimeError(f"frozen review expected 37 cue occurrences, found {score['cue_occurrence_count']}")

    OUTPUT.mkdir(parents=True, exist_ok=False)
    result = {
        "diagnostic_only": True,
        "dataset_sha256": DATASET_SHA256,
        "split_manifest_sha256": _sha256(SPLIT_PATH.read_bytes()),
        "annotation_file_sha256": _sha256(annotation_bytes),
        "scope": "17 frozen DEV knowledge-update histories; user-authored source turns only",
        "cue_rule": "literal case-insensitive 'instead of' or 'rather than' occurrence",
        "question_or_gold_decoded": False,
        "model_calls": 0,
        "reviewer_count": annotations["reviewer_count"],
        "score": score,
        "candidates": candidates,
        "limitations": [
            "Manual labels are a one-reviewer exploratory source-only triage, not benchmark gold.",
            "The denominator is lexical cue occurrences, not all potential state transitions; recall is not measured.",
            "The knowledge-update histories are a small, non-random DEV slice and are not an independent test set.",
            "State/task revision precision must not be reported as LongMemEval answer accuracy or system performance.",
        ],
    }
    result_bytes = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    _freeze_artifact(OUTPUT / "cue_audit.json", result_bytes)
    report = (
        "# DEV Alternative-Cue Audit\n\n"
        "A zero-model-call, source-only review of literal `instead of` / `rather than` hits in the frozen DEV knowledge-update histories. "
        "The manually assigned labels distinguish adopted revision from proposals, questions, non-self content, and assertions without a prior value.\n\n"
        f"- Histories: {knowledge_update_count}.\n"
        f"- Cue occurrences: {score['cue_occurrence_count']} in {score['unique_source_turn_count']} turns.\n"
        f"- Adopted personal-state revisions: {score['label_counts_by_occurrence'].get('EXPLICIT_STATE_REVISION', 0)}/{score['cue_occurrence_count']} "
        f"({score['adopted_personal_state_revision_precision']:.1%} cue precision).\n"
        f"- Including one-time task-plan revision: {sum(score['label_counts_by_occurrence'].get(label, 0) for label in REVISION_LABELS)}/{score['cue_occurrence_count']} "
        f"({score['state_or_task_revision_precision']:.1%} cue precision).\n"
        "- Questions/gold decoded: no; model calls: 0.\n\n"
        "This is an exploratory false-positive audit. It measures neither recall nor end-to-end quality, and the single-reviewer labels are not benchmark gold. "
        "A lexical contrast marker alone is therefore not a revision admission rule.\n"
    )
    _freeze_artifact(OUTPUT / "report.md", report.encode("utf-8"))
    print(json.dumps({"score": score, "output": str(OUTPUT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
