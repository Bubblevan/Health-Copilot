"""Check exact old-value anchors before five reviewed DEV change mentions."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.research.memory.audit_mem3b1_dev_alternative_cues_v2 import (
    DATASET_PATH_CANDIDATES,
    DATASET_SHA256,
    SPLIT_PATH,
    _sha256,
)
from tools.research.memory.run_mem3b1_dev_revision_pair_diagnostic_v1 import (
    _freeze_artifact,
    _selected_dev_records,
    _source_turns,
)

OUTPUT = ROOT / "runs/memory/mem3/mem3b1-dev-change-predecessor-anchor-audit-v1"
TARGETS = (
    {"question_id": "031748ae", "source_position": 151, "old_value_text": "liquid soap", "attribute_hint": "soap_format"},
    {"question_id": "031748ae", "source_position": 151, "old_value_text": "plastic", "attribute_hint": "product_packaging"},
    {"question_id": "1cea1afa", "source_position": 43, "old_value_text": "driving", "attribute_hint": "weekend_trip_to_grocery_store"},
    {"question_id": "4d6b87c8", "source_position": 30, "old_value_text": "3 pm", "attribute_hint": "tea_break_time"},
    {"question_id": "a1eacc2a", "source_position": 156, "old_value_text": "taking the bus", "attribute_hint": "trip_to_train_station"},
)


def _audit_target(
    records: dict[str, dict[str, Any]], target: dict[str, Any]
) -> dict[str, Any]:
    turns = _source_turns(records[target["question_id"]])
    cue_turn = next(
        (row for row in turns if row["source_position"] == target["source_position"]),
        None,
    )
    if cue_turn is None:
        raise ValueError(f"change cue turn missing: {target}")
    source = str(cue_turn["text"])
    old_value = target["old_value_text"]
    if source.casefold().count(old_value.casefold()) != 1:
        raise ValueError(f"old value is not a unique exact source substring: {target}")
    earlier_matches = [
        {
            "source_position": row["source_position"],
            "session_date": row["session_date"],
            "source_text": row["text"],
        }
        for row in turns
        if row["source_position"] < target["source_position"]
        and old_value.casefold() in str(row["text"]).casefold()
    ]
    return {
        **target,
        "cue_session_date": cue_turn["session_date"],
        "cue_turn_sha256": _sha256(source.encode("utf-8")),
        "cue_source_quote": source,
        "prior_user_turn_count": target["source_position"],
        "prior_exact_literal_match_count": len(earlier_matches),
        "prior_exact_literal_matches": earlier_matches,
        "question_or_gold_used": False,
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
    rows = [_audit_target(records, target) for target in TARGETS]
    anchored = sum(row["prior_exact_literal_match_count"] > 0 for row in rows)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    result = {
        "diagnostic_only": True,
        "dataset_sha256": DATASET_SHA256,
        "split_manifest_sha256": _sha256(SPLIT_PATH.read_bytes()),
        "scope": "five source-only adopted state-change mentions manually selected from frozen DEV cue audit v2",
        "question_or_gold_decoded": False,
        "model_calls": 0,
        "exact_prior_old_value_anchor_count": anchored,
        "reviewed_change_mention_count": len(rows),
        "exact_anchor_rate": anchored / len(rows),
        "targets": rows,
        "limitations": [
            "Only exact case-insensitive literal matches in earlier user turns are counted; semantic/paraphrased anchors are not assessed.",
            "The five mentions were selected after source-only review and do not form a random or independent sample.",
            "No revision-truth recall, end-to-end metric, or benchmark accuracy is measured.",
        ],
    }
    payload = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    _freeze_artifact(OUTPUT / "anchor_audit.json", payload)
    report = (
        "# DEV Change-Mention Predecessor Anchor Audit\n\n"
        f"Among {len(rows)} explicitly adopted change mentions, {anchored} had an exact occurrence of the proposed old value in an earlier user turn "
        f"({anchored}/{len(rows)} = {anchored / len(rows):.1%} exact-anchor rate).\n\n"
        "This is a conservative literal-source check, not semantic anchor recall. The change phrases themselves can still assert that a prior behavior existed; "
        "however, these source passages do not independently establish when that prior value was observed. The result supports retaining the distinction between "
        "a change mention and a fully time-bounded revision chain. No questions/gold were decoded and no model was called.\n"
    )
    _freeze_artifact(OUTPUT / "report.md", report.encode("utf-8"))
    print(json.dumps({"exact_anchor_rate": result["exact_anchor_rate"], "anchors": anchored, "targets": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
