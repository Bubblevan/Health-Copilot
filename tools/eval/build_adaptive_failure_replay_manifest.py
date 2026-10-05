"""Build a gold-blind ID list for replaying frozen Adaptive runtime failures."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys_path = str(ROOT / "src")

if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

from health_ai_copilot.evaluation.manifests import file_sha256


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _trace_failure_reason(trace: dict[str, Any]) -> str | None:
    for event in trace.get("reasoning_events", ()):
        fields = event.get("fields", {})
        if fields.get("event") == "adaptive_harness_disposition":
            reason = fields.get("reason")
            return reason if isinstance(reason, str) else None
    return None


def _ids_sha256(case_ids: list[str]) -> str:
    return sha256(("\n".join(case_ids) + "\n").encode("utf-8")).hexdigest()


def build(case_file: Path, trace_file: Path, output: Path) -> dict[str, Any]:
    # Deliberately read only case_id and trace_id from case rows; evaluator gold,
    # parsed answers, and correctness are never inspected by this selector.
    case_rows = _jsonl(case_file)
    case_trace_ids = {
        str(row["response"]["trace_id"]): str(row["case_id"])
        for row in case_rows
        if isinstance(row.get("response"), dict)
        and isinstance(row["response"].get("trace_id"), str)
    }
    exact_failures: dict[str, str] = {}
    for trace in _jsonl(trace_file):
        trace_id = trace.get("trace_id")
        case_id = case_trace_ids.get(str(trace_id))
        if case_id is None:
            continue
        reason = _trace_failure_reason(trace)
        if reason:
            exact_failures[case_id] = reason

    by_reason: dict[str, list[str]] = defaultdict(list)
    ordered_case_ids = [str(row["case_id"]) for row in case_rows]
    for case_id in ordered_case_ids:
        reason = exact_failures.get(case_id)
        if reason:
            by_reason[reason].append(case_id)

    primary_reason = "TypeError:invalid_team_recruitment"
    selected = by_reason.get(primary_reason, [])
    if len(selected) != 219:
        raise ValueError(
            f"expected 219 {primary_reason} rows from frozen execution traces, found {len(selected)}"
        )
    manifest = {
        "schema_version": "adaptive-validation-failure-replay-selection-v1",
        "status": "IDS_FROZEN_DEBUG_REPLAY_NOT_RUN",
        "selection_rule": (
            "Select case IDs only by the exact Adaptive trace disposition reason; "
            "the selector reads no gold, answer text, parsed answer, or correctness field."
        ),
        "primary_failure_reason": primary_reason,
        "source_run_git_sha": "203b2cdcfe3765695f405e4afe6c0704da07d80f",
        "source_parser_revision": "deterministic-mcq-parser-v7",
        "source_case_file": str(case_file),
        "source_case_file_sha256": file_sha256(case_file),
        "source_trace_file": str(trace_file),
        "source_trace_file_sha256": file_sha256(trace_file),
        "case_count": len(selected),
        "case_ids_sha256": _ids_sha256(selected),
        "case_ids": selected,
        "failure_counts": {reason: len(ids) for reason, ids in sorted(by_reason.items())},
        "other_validation_failure_case_ids": {
            reason: ids for reason, ids in sorted(by_reason.items()) if reason != primary_reason
        },
        "debug_constraints": {
            "gold_visible_to_selector": False,
            "gold_visible_to_harness": False,
            "purpose": "reproduce and inspect Adaptive route output validation only",
            "accuracy_tuning": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case-file",
        type=Path,
        default=ROOT / "runs/common_eval/harness-v1-base-20261005/rescored-parser-v7/diagnosisarena-915/B2/cases.jsonl",
    )
    parser.add_argument(
        "--trace-file",
        type=Path,
        default=ROOT / "runs/common_eval/harness-v1-base-20261005/rescored-parser-v6/diagnosisarena-915/B2/traces.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runs/common_eval/harness-v1-base-20261005/adaptive-failure-audit-20261005/team-recruitment-failures.json",
    )
    args = parser.parse_args()
    manifest = build(args.case_file, args.trace_file, args.output)
    print(json.dumps({
        "output": str(args.output),
        "case_count": manifest["case_count"],
        "case_ids_sha256": manifest["case_ids_sha256"],
        "failure_counts": manifest["failure_counts"],
        "status": manifest["status"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
