"""Aggregate-only temporal schema audit for ESL-Bench development batch 202607."""

from __future__ import annotations

import argparse
import json
from bisect import bisect_right
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from eval.rag_e5.temporal import (
    TEMPORAL_SEMANTICS_ID,
    TEMPORAL_SEMANTICS_SHA256,
    parse_esl_date,
    parse_esl_naive_datetime,
)

DEVELOPMENT_BATCH = "202607"
DEVELOPMENT_USERS = tuple(f"user{number}_AT_demo" for number in range(5200, 5220))
DEFAULT_SOURCE_ROOT = Path("D:/MyLab/Jianli/external/datasets/ESL-Bench")


def build_temporal_audit(source_root: Path) -> dict[str, Any]:
    """Read only the pinned 202607 state files and return aggregate metadata."""
    batch_root = source_root / "data" / DEVELOPMENT_BATCH
    counters: Counter[str] = Counter()
    duplicate_group_count = 0
    max_duplicate_group_size = 1
    events_crossing_boundary_possible = 0

    for user_id in DEVELOPMENT_USERS:
        user_root = batch_root / user_id
        timeline = _read_json(user_root / "timeline.json")
        exams = _read_json(user_root / "exam_data.json")
        if not isinstance(timeline, dict) or timeline.get("user_id") != user_id:
            raise ValueError("202607 timeline identity does not match its pinned user directory")
        entries = timeline.get("entries")
        if not isinstance(entries, list):
            raise TypeError("202607 timeline.entries must be an array")
        if not isinstance(exams, list):
            raise TypeError("202607 exam_data.json must be an array")

        counters["user_count"] += 1
        generated_at = timeline.get("generated_at")
        counters["generated_at_count"] += 1
        _count_timestamp(generated_at, counters, prefix="generated_at")

        timestamp_counts: Counter[datetime] = Counter()
        parsed_entries: list[tuple[datetime, dict[str, Any]]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                raise TypeError("202607 timeline entries must be objects")
            counters["timeline_timestamp_count"] += 1
            timestamp = _count_timestamp(entry.get("time"), counters, prefix="timeline")
            if timestamp is not None:
                timestamp_counts[timestamp] += 1
                parsed_entries.append((timestamp, entry))
            if entry.get("entry_type") == "event" and "end_time" in entry:
                counters["events_with_end_time"] += 1

        duplicate_sizes = [count for count in timestamp_counts.values() if count > 1]
        duplicate_group_count += len(duplicate_sizes)
        if duplicate_sizes:
            max_duplicate_group_size = max(max_duplicate_group_size, max(duplicate_sizes))
        candidate_times = sorted(timestamp_counts)
        for start, entry in parsed_entries:
            if entry.get("entry_type") != "event" or "end_time" not in entry:
                continue
            end = _count_timestamp(entry.get("end_time"), counters, prefix="event_end")
            if end is None:
                continue
            if end < start:
                counters["invalid_event_interval_count"] += 1
                continue
            next_boundary_index = bisect_right(candidate_times, start)
            if next_boundary_index < len(candidate_times) and candidate_times[next_boundary_index] <= end:
                events_crossing_boundary_possible += 1

        for exam in exams:
            if not isinstance(exam, dict):
                raise TypeError("202607 exam records must be objects")
            counters["exam_date_count"] += 1
            value = exam.get("exam_date")
            try:
                parse_esl_date(value)
            except ValueError:
                counters["date_only_exam_count"] += 0
                _count_timestamp(value, counters, prefix="exam_datetime")
            else:
                counters["date_only_exam_count"] += 1

    source_temporal_schema_pass = (
        counters["timeline_timestamp_count"] > 0
        and counters["timeline_naive_count"] == counters["timeline_timestamp_count"]
        and counters["timeline_aware_count"] == 0
        and counters["timeline_invalid_count"] == 0
        and counters["generated_at_count"] == len(DEVELOPMENT_USERS)
        and counters["generated_at_naive_count"]
        + counters["generated_at_aware_count"]
        + counters["generated_at_invalid_count"]
        == counters["generated_at_count"]
        and counters["exam_date_count"] > 0
        and counters["date_only_exam_count"] == counters["exam_date_count"]
        and counters["event_end_aware_count"] == 0
        and counters["event_end_invalid_count"] == 0
        and counters["invalid_event_interval_count"] == 0
    )
    result = {
        "schema_version": "rag-e5-temporal-source-audit-v1",
        "batch_id": DEVELOPMENT_BATCH,
        "user_count": counters["user_count"],
        "timeline_timestamp_count": counters["timeline_timestamp_count"],
        "naive_timeline_timestamp_count": counters["timeline_naive_count"],
        "timeline_T_separator_count": counters["timeline_separator_T_count"],
        "timeline_space_separator_count": counters["timeline_separator_space_count"],
        "aware_timeline_timestamp_count": counters["timeline_aware_count"],
        "invalid_timeline_timestamp_count": counters["timeline_invalid_count"],
        "generated_at_count": counters["generated_at_count"],
        "naive_generated_at_count": counters["generated_at_naive_count"],
        "aware_generated_at_count": counters["generated_at_aware_count"],
        "invalid_generated_at_count": counters["generated_at_invalid_count"],
        "exam_date_count": counters["exam_date_count"],
        "date_only_exam_count": counters["date_only_exam_count"],
        "duplicate_timestamp_group_count": duplicate_group_count,
        "max_duplicate_group_size": max_duplicate_group_size,
        "events_with_end_time": counters["events_with_end_time"],
        "naive_event_end_time_count": counters["event_end_naive_count"],
        "aware_event_end_time_count": counters["event_end_aware_count"],
        "invalid_event_end_time_count": counters["event_end_invalid_count"],
        "events_crossing_candidate_boundary_possible": events_crossing_boundary_possible,
        "invalid_event_interval_count": counters["invalid_event_interval_count"],
        "temporal_semantics_id": TEMPORAL_SEMANTICS_ID,
        "temporal_semantics_sha256": TEMPORAL_SEMANTICS_SHA256,
        "timezone_assumption_used": False,
        "source_relative_ordering": True,
        "same_timestamp_policy": "STRICT_EXCLUDE",
        "date_only_exam_policy": "SAME_DAY_EXCLUDE",
        "generated_at_policy_input": False,
        "audit_gate": "PASS" if source_temporal_schema_pass else "FAIL",
        "202608_opened": False,
        "patient_level_values_or_prose_included": False,
    }
    return result


def _count_timestamp(value: object, counters: Counter[str], *, prefix: str) -> datetime | None:
    if isinstance(value, str) and len(value) > 10 and value[10] in {"T", " "}:
        separator = "T" if value[10] == "T" else "space"
        counters[f"{prefix}_separator_{separator}_count"] += 1
    try:
        parsed = datetime.fromisoformat(value) if isinstance(value, str) else None
    except ValueError:
        parsed = None
    if parsed is None:
        counters[f"{prefix}_invalid_count"] += 1
        return None
    if parsed.tzinfo is not None:
        counters[f"{prefix}_aware_count"] += 1
        return None
    try:
        checked = parse_esl_naive_datetime(value)
    except ValueError:
        counters[f"{prefix}_invalid_count"] += 1
        return None
    counters[f"{prefix}_naive_count"] += 1
    return checked


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/rag_e5/e5b_temporal_semantics_audit.json"),
    )
    args = parser.parse_args()
    payload = build_temporal_audit(args.source_root)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(encoded.encode("utf-8"))
    print(encoded, end="")


if __name__ == "__main__":
    main()
