"""Mine an exact, conservative same-turn time-change cue on frozen DEV only."""

from __future__ import annotations

import hashlib
import json
import re
import sys
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

OUTPUT = ROOT / "runs/memory/mem3/mem3b1-dev-explicit-time-cue-audit-v1"
TIME_VALUE = r"\d{1,2}(?::\d{2})?\s*[ap]m\b"
TIME_REVISION = re.compile(
    rf"(?P<new>{TIME_VALUE})\s+instead of\s+(?P<old>{TIME_VALUE})",
    flags=re.IGNORECASE,
)


def find_explicit_time_revision_cues(
    question_id: str,
    turns: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates = []
    for turn in turns:
        text = turn["text"]
        for match in TIME_REVISION.finditer(text):
            candidates.append(
                {
                    "question_id": question_id,
                    "source_position": turn["source_position"],
                    "session_date": turn["session_date"],
                    "old_value_text": match.group("old"),
                    "new_value_text": match.group("new"),
                    "cue_span": match.group(0),
                    "source_quote": text,
                    "cue_type": "explicit_same_turn_time_replacement",
                    "model_inference_used": False,
                }
            )
    return candidates


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"output already exists: {OUTPUT}")
    manifest = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    dev_ids = set(manifest["dev"]["question_ids"])
    dataset_path = next((path for path in DATASET_PATH_CANDIDATES if path.is_file()), None)
    if dataset_path is None:
        raise FileNotFoundError("frozen LongMemEval-S data not found")
    records = _selected_dev_records(dataset_path, dev_ids)
    knowledge_update_rows = []
    for question_id, record in records.items():
        if record.get("question_type") != "knowledge-update":
            continue
        turns = _source_turns(record)
        knowledge_update_rows.append(
            {
                "question_id": question_id,
                "source_record_sha256": record["_raw_record_sha256"],
                "user_turn_count": len(turns),
                "candidate_cues": find_explicit_time_revision_cues(question_id, turns),
            }
        )
    if len(knowledge_update_rows) != 17:
        raise RuntimeError(f"expected 17 DEV knowledge-update histories, found {len(knowledge_update_rows)}")
    cues = [cue for row in knowledge_update_rows for cue in row["candidate_cues"]]
    OUTPUT.mkdir(parents=True, exist_ok=False)
    result = {
        "diagnostic_only": True,
        "scope": "frozen LongMemEval-S DEV knowledge-update histories; no questions or gold decoded",
        "dataset_sha256": DATASET_SHA256,
        "split_manifest_sha256": _sha_file(SPLIT_PATH),
        "dev_id_count": len(dev_ids),
        "knowledge_update_history_count": len(knowledge_update_rows),
        "cue_rule": "numeric/time-of-day value followed by 'instead of' and another numeric/time-of-day value within one user turn",
        "candidate_count": len(cues),
        "candidate_question_ids": sorted({row["question_id"] for row in cues}),
        "model_calls": 0,
        "candidate_truth_scored": False,
        "histories": knowledge_update_rows,
        "candidates": cues,
        "limitations": [
            "This narrow lexical rule measures only explicit time-of-day contrast cues.",
            "It does not detect arbitrary non-numeric or cross-turn revisions.",
            "A cue is not by itself a final memory update admission.",
        ],
    }
    _freeze_artifact(
        OUTPUT / "cue_audit.json",
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n",
    )
    report = (
        "# DEV Explicit Time-Revision Cue Audit\n\n"
        "Deterministic, zero-model-call scan of the frozen DEV knowledge-update conversations. "
        "This narrow cue miner is a candidate-generation diagnostic, not a general revision detector or benchmark metric.\n\n"
        f"- Histories scanned: {len(knowledge_update_rows)}.\n"
        f"- Exact same-turn time-replacement cues: {len(cues)}.\n"
        f"- Candidate DEV IDs: {', '.join(sorted({row['question_id'] for row in cues})) or 'none'}.\n"
        "- Questions/gold decoded: no.\n"
        "- Model calls: 0.\n"
    )
    _freeze_artifact(OUTPUT / "report.md", report.encode("utf-8"))
    print(json.dumps({key: result[key] for key in (
        "knowledge_update_history_count", "candidate_count", "candidate_question_ids", "model_calls"
    )}, indent=2))


if __name__ == "__main__":
    main()
