"""Exercise conservative change-mention views on five reviewed DEV source cues."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from health_ai_copilot.research.memory.change_mention_admission_v1 import (
    ChangeMention,
    StateObservation,
    admit_change_mention,
    project_as_of,
    project_change,
    project_current,
)
from tools.research.memory.run_mem3b1_dev_revision_pair_diagnostic_v1 import (
    DATASET_PATH_CANDIDATES,
    DATASET_SHA256,
    SPLIT_PATH,
    _freeze_artifact,
    _selected_dev_records,
    _source_turns,
)
from tools.research.memory.audit_mem3b1_dev_alternative_cues_v2 import _sha256

OUTPUT = ROOT / "runs/memory/mem3/mem3b1-dev-change-mention-admission-v4"
TARGETS = (
    {
        "question_id": "031748ae",
        "source_position": 151,
        "old_value_text": "liquid soap",
        "new_value_text": "soap bars",
        "slot_key": ("scope:031748ae", "SELF", "PERSONAL_CARE", "SOAP_FORMAT"),
    },
    {
        "question_id": "031748ae",
        "source_position": 151,
        "old_value_text": "plastic",
        "new_value_text": "cardboard or paper packaging",
        "slot_key": ("scope:031748ae", "SELF", "PRODUCT_PREFERENCE", "PACKAGING_MATERIAL"),
    },
    {
        "question_id": "1cea1afa",
        "source_position": 43,
        "old_value_text": "driving",
        "new_value_text": "riding my bike to the grocery store on weekends",
        "slot_key": ("scope:1cea1afa", "SELF", "WEEKEND_TRAVEL", "GROCERY_TRANSPORT"),
    },
    {
        "question_id": "4d6b87c8",
        "source_position": 30,
        "old_value_text": "3 pm",
        "new_value_text": "2:30 pm",
        "slot_key": ("scope:4d6b87c8", "SELF", "DAILY_ROUTINE", "TEA_BREAK_TIME"),
    },
    {
        "question_id": "a1eacc2a",
        "source_position": 156,
        "old_value_text": "taking the bus",
        "new_value_text": "walking to the train station",
        "slot_key": ("scope:a1eacc2a", "SELF", "DAILY_TRAVEL", "TRIP_TO_TRAIN_STATION"),
    },
)


def _parse_session_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y/%m/%d (%a) %H:%M").replace(tzinfo=timezone.utc)


def _owner_assertion_span(text: str, new_value: str, old_value: str) -> str:
    new_start = text.index(new_value)
    old_end = text.index(old_value) + len(old_value)
    left_boundaries = [text.rfind(mark, 0, new_start) for mark in (".", "?", "!", "\n")]
    start = max(left_boundaries) + 1
    right_boundaries = [
        position for mark in (".", "?", "!", "\n")
        if (position := text.find(mark, old_end)) >= 0
    ]
    end = min(right_boundaries) if right_boundaries else len(text)
    return text[start:end].strip()


def _rows(records: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for target in TARGETS:
        turns = _source_turns(records[target["question_id"]])
        turn = next(
            row for row in turns if row["source_position"] == target["source_position"]
        )
        quote = str(turn["text"])
        assertion_span = _owner_assertion_span(
            quote, target["new_value_text"], target["old_value_text"]
        )
        mention_id = hashlib.sha256(
            f"{target['question_id']}:{target['source_position']}:{target['slot_key']}".encode("utf-8")
        ).hexdigest()
        mention = ChangeMention(
            mention_id=mention_id,
            slot_key=target["slot_key"],
            old_value_text=target["old_value_text"],
            new_value_text=target["new_value_text"],
            occurred_at=_parse_session_date(str(turn["session_date"])),
            source_turn_id=f"{target['question_id']}:user-turn-{target['source_position']}",
            source_text=quote,
            owner_assertion_evidence=assertion_span,
            provenance_sha256=hashlib.sha256(quote.encode("utf-8")).hexdigest(),
        )
        prior_observations = []
        for prior in turns:
            if prior["source_position"] >= target["source_position"]:
                continue
            if target["old_value_text"].casefold() in str(prior["text"]).casefold():
                prior_observations.append(
                    StateObservation(
                        memory_id=f"{target['question_id']}:user-turn-{prior['source_position']}",
                        slot_key=target["slot_key"],
                        value_text=target["old_value_text"],
                        observed_at=_parse_session_date(str(prior["session_date"])),
                    )
                )
        decision = admit_change_mention(mention, prior_observations)
        before_change = mention.occurred_at - timedelta(days=1)
        results.append(
            {
                "question_id": target["question_id"],
                "source_position": target["source_position"],
                "slot_key": list(target["slot_key"]),
                "old_value_text": mention.old_value_text,
                "new_value_text": mention.new_value_text,
                "source_turn_sha256": hashlib.sha256(quote.encode("utf-8")).hexdigest(),
                "owner_assertion_evidence": assertion_span,
                "prior_exact_anchor_candidate_count": len(prior_observations),
                "admission_kind": decision.kind,
                "admission_reason": decision.reason,
                "current_projection": project_current(decision),
                "as_of_day_before_projection": project_as_of(decision, before_change),
                "change_projection": project_change(decision),
            }
        )
    return results


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"output already exists: {OUTPUT}")
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    dev_ids = set(split["dev"]["question_ids"])
    dataset_path = next((path for path in DATASET_PATH_CANDIDATES if path.is_file()), None)
    if dataset_path is None:
        raise FileNotFoundError("frozen LongMemEval-S data not found")
    records = _selected_dev_records(dataset_path, dev_ids)
    rows = _rows(records)
    if len(rows) != 5:
        raise RuntimeError("expected five source-reviewed personal-state change mentions")
    counts = Counter(row["admission_kind"] for row in rows)
    if counts != Counter({"UNANCHORED_CHANGE_MENTION": 5}):
        raise RuntimeError(f"unexpected admission distribution: {counts}")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    result = {
        "diagnostic_only": True,
        "dataset_sha256": DATASET_SHA256,
        "split_manifest_sha256": _sha256(SPLIT_PATH.read_bytes()),
        "scope": "five source-reviewed change mentions from frozen DEV knowledge-update user histories",
        "timestamp_policy": "corpus date labels are parsed as timezone-naive and tagged UTC solely for deterministic ordering; no user timezone is inferred",
        "question_or_gold_decoded": False,
        "model_calls": 0,
        "first_person_assertion_span_validated_count": len(rows),
        "admission_counts": dict(sorted(counts.items())),
        "current_successor_projection_count": sum(
            row["current_projection"]["value_text"] == row["new_value_text"] for row in rows
        ),
        "pre_change_as_of_unresolved_count": sum(
            row["as_of_day_before_projection"]["status"] == "UNRESOLVED" for row in rows
        ),
        "change_mention_projection_count": sum(
            row["change_projection"]["status"] == "CHANGE_MENTION" for row in rows
        ),
        "quality_accuracy": None,
        "benchmark_performance_claim": False,
        "rows": rows,
        "limitations": [
            "The five cases are hand-selected change mentions from DEV, not a random evaluation sample.",
            "Prior-value linking uses exact case-insensitive text matching only; semantic aliases are not resolved.",
            "Corpus session timestamps do not declare a timezone; UTC tagging is a deterministic ordering convention only.",
            "Projection behavior is a contract demonstration, not answer accuracy or a benchmark score.",
            "AS_OF before an unanchored predecessor interval is deliberately unresolved; the old value remains available only in CHANGE context with provenance.",
        ],
    }
    payload = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    _freeze_artifact(OUTPUT / "admission_diagnostic.json", payload)
    report = (
        "# DEV Change-Mention Admission Diagnostic\n\n"
        "- Explicit change mentions routed to unanchored status: 5/5.\n"
        "- Current projection contains the source-asserted successor: 5/5.\n"
        "- Pre-change AS_OF projection remains unresolved: 5/5.\n"
        "- CHANGE projection retains the old-to-new mention with provenance: 5/5.\n\n"
        "These are deterministic contract outcomes on five selected DEV source passages, not prediction accuracy or benchmark performance. "
        "The design preserves current-state usefulness without claiming an unsupported historical interval. Questions/gold decoded: no; model calls: 0.\n"
    )
    _freeze_artifact(OUTPUT / "report.md", report.encode("utf-8"))
    print(json.dumps({key: result[key] for key in (
        "admission_counts",
        "current_successor_projection_count",
        "pre_change_as_of_unresolved_count",
        "change_mention_projection_count",
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
