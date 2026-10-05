"""Classify unparsed frozen Harness answers without reading evaluator labels."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from health_ai_copilot.evaluation.manifests import file_sha256
from health_ai_copilot.harness.contracts import AnswerSchema
from health_ai_copilot.harness.verification import (
    ANSWER_PARSER_REVISION,
    is_explicit_abstention,
    parse_answer,
)

DEFAULT_ROOT = ROOT / "runs/common_eval/harness-v1-base-20261005/rescored-parser-v7"
DEFAULT_RUNS = {
    "cmb-common-1024/B0": DEFAULT_ROOT / "cmb-common-1024/B0/cases.jsonl",
    "cmb-common-1024/B2": DEFAULT_ROOT / "cmb-common-1024/B2/cases.jsonl",
    "diagnosisarena-915/B0": DEFAULT_ROOT / "diagnosisarena-915/B0/cases.jsonl",
    "diagnosisarena-915/B2": DEFAULT_ROOT / "diagnosisarena-915/B2/cases.jsonl",
}

_ANSWER_PREFIX = re.compile(
    r"^\s*(?:(?:final\s+)?(?:correct\s+)?(?:answer|option|options|choice|choices)"
    r"|(?:最终答案|正确答案|答案|选项字母|正确选项|最佳选项|多选题答案))\s*[:：=]\s*(.*)$",
    re.IGNORECASE,
)
_LEADING_LABELS = re.compile(
    r"^\s*[\[({'\"]?\s*([A-E](?:\s*(?:[,，、;/&+]|\band\b)\s*[A-E])*)",
    re.IGNORECASE,
)
_ENUMERATED_LABEL = re.compile(r"(?:^|[,，、;；])\s*([A-E])\s*[.)](?=\s*\S)", re.IGNORECASE)
_JSON_ANSWER = re.compile(
    r"[\"'](?:answer|option|options|choice|choices|final_answer|multi_select)[\"']"
    r"\s*:\s*(\[[^\]]*\]|[\"'][^\"']*[\"'])",
    re.IGNORECASE,
)


def _explicit_answer_statements(text: str) -> list[tuple[str, ...]]:
    statements: list[tuple[str, ...]] = []
    candidates = list(text.splitlines())
    candidates.extend(match.group(0) for match in _JSON_ANSWER.finditer(text))
    for line in candidates:
        prefix_match = _ANSWER_PREFIX.match(line)
        value = prefix_match.group(1) if prefix_match else line
        if not prefix_match and not line.lstrip().startswith(("{", "[")):
            continue
        json_match = _JSON_ANSWER.search(value)
        if json_match:
            value = json_match.group(1).strip("[](){}\"' ")
        leading = _LEADING_LABELS.match(value)
        labels = re.findall(r"[A-E]", leading.group(1).upper()) if leading else []
        enumerated = [match.group(1).upper() for match in _ENUMERATED_LABEL.finditer(value)]
        if len(enumerated) >= 2:
            labels = enumerated
        if labels:
            statements.append(tuple(sorted(set(labels))))
    return statements


def _classify_unparsed(answer: str, safety_flags: list[str], schema: AnswerSchema) -> tuple[str, dict[str, Any]]:
    if any(flag.startswith("reasoning_failure:") for flag in safety_flags):
        return "C_ORCHESTRATOR_OR_RUNTIME_FAILURE", {}
    if any(
        flag.startswith(("urgent_marker:", "prescription_marker:", "safety_route:"))
        for flag in safety_flags
    ):
        return "B_SAFETY_ABSTENTION", {}
    if is_explicit_abstention(answer):
        return "D_MODEL_ABSTENTION_OR_NONCOMMITMENT", {}
    statements = _explicit_answer_statements(answer)
    unique = sorted(set(statements))
    if len(unique) > 1:
        return "E_AMBIGUOUS_MULTIPLE_FINAL_CHOICES", {"explicit_choice_sets": unique}
    if unique:
        labels = unique[0]
        if schema is AnswerSchema.SINGLE_CHOICE and len(labels) != 1:
            return "E_AMBIGUOUS_MULTIPLE_FINAL_CHOICES", {"explicit_choice_sets": unique}
        return "A_PARSER_MISSED_EXPLICIT_CHOICE_CANDIDATE", {"explicit_choice_sets": unique}
    return "D_MODEL_NEVER_COMMITS_TO_VALID_CHOICE", {}


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _candidate_schemas() -> tuple[dict[str, AnswerSchema], dict[str, str]]:
    prepared = json.loads((ROOT / "configs/eval/local_prepared_views_h0.json").read_text())
    schemas: dict[str, AnswerSchema] = {}
    hashes: dict[str, str] = {}
    for dataset_id, entry in prepared["datasets"].items():
        path = Path(entry["candidate_view_path"])
        hashes[dataset_id] = file_sha256(path)
        if hashes[dataset_id] != entry["candidate_view_sha256"]:
            raise ValueError(f"candidate view hash mismatch for {dataset_id}")
        if dataset_id != "cmb-common":
            continue
        for row in _read_rows(path):
            question_type = str(row.get("question_type", ""))
            schemas[str(row["id"])] = (
                AnswerSchema.MULTI_SELECT if "多项" in question_type or "多选" in question_type
                else AnswerSchema.SINGLE_CHOICE
            )
    return schemas, hashes


def _normalize_parsed(value: object) -> object:
    if isinstance(value, list):
        return tuple(sorted({str(item).upper() for item in value}))
    if isinstance(value, tuple):
        return tuple(sorted({str(item).upper() for item in value}))
    if isinstance(value, str):
        return value.upper()
    return value


def audit(runs: dict[str, Path], output: Path) -> dict[str, Any]:
    run_results: dict[str, Any] = {}
    all_candidates: list[dict[str, Any]] = []
    candidate_schemas, candidate_hashes = _candidate_schemas()
    for run_id, path in runs.items():
        category_counts: Counter[str] = Counter()
        unparsed_cases: list[dict[str, Any]] = []
        parser_mismatches: list[str] = []
        rows = _read_rows(path)
        for row in rows:
            case_id = row.get("case_id")
            response = row.get("response", {})
            if not isinstance(response, dict):
                continue
            answer = response.get("answer_text", "")
            if not isinstance(answer, str):
                answer = ""
            stored = response.get("parsed_answer")
            schema = (
                candidate_schemas.get(str(case_id), AnswerSchema.MULTI_SELECT)
                if run_id.startswith("cmb-common-1024/") else AnswerSchema.SINGLE_CHOICE
            )
            parsed = parse_answer(answer, schema)
            if _normalize_parsed(parsed) != _normalize_parsed(stored):
                parser_mismatches.append(str(case_id))
            if parsed is not None:
                continue
            flags = response.get("safety_flags", [])
            flags = [str(item) for item in flags] if isinstance(flags, list) else []
            category, details = _classify_unparsed(answer, flags, schema)
            category_counts[category] += 1
            record = {
                "case_id": str(case_id),
                "category": category,
                "safety_flags": flags,
                **details,
            }
            unparsed_cases.append(record)
            if category == "A_PARSER_MISSED_EXPLICIT_CHOICE_CANDIDATE":
                record["raw_answer_text"] = answer
                all_candidates.append({"run_id": run_id, **record})
        run_results[run_id] = {
            "case_count": len(rows),
            "parse_success": len(rows) - len(unparsed_cases),
            "parse_success_rate": (len(rows) - len(unparsed_cases)) / len(rows) if rows else None,
            "unparsed_count": len(unparsed_cases),
            "categories": dict(sorted(category_counts.items())),
            "parser_mismatches_vs_frozen_response": parser_mismatches,
            "checkpoint_sha256": file_sha256(path),
            "cases": unparsed_cases,
        }
    result = {
        "schema_version": "gold-blind-parser-failure-audit-v1",
        "status": "REVIEW_REQUIRED" if all_candidates or any(
            run["parser_mismatches_vs_frozen_response"] for run in run_results.values()
        ) else "NO_PARSER_FALSE_NEGATIVE_CANDIDATES_FOUND",
        "scoring_revision": ANSWER_PARSER_REVISION,
        "method": (
            "Read only candidate-view case IDs/question_type and case_id plus response "
            "answer_text/parsed_answer/safety_flags. "
            "Do not inspect gold, score.correct, or evaluator labels. Parser-missed candidates "
            "are surfaced for manual review; no candidate is auto-corrected."
        ),
        "runs": run_results,
        "candidate_view_sha256": candidate_hashes,
        "parser_missed_explicit_choice_candidates": all_candidates,
        "gold_accessed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/common_eval/harness-v1-base-20261005/parser-audit-v7-gold-blind/audit.json")
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="NAME=CASES_JSONL",
        help="audit a specific run; repeat for each arm. Defaults to the historical B0/B2 set.",
    )
    args = parser.parse_args()
    runs = dict(DEFAULT_RUNS)
    if args.run:
        runs = {}
        for item in args.run:
            name, separator, path = item.partition("=")
            if not separator or not name or not path:
                parser.error("--run must have NAME=CASES_JSONL form")
            if name in runs:
                parser.error(f"duplicate --run name: {name}")
            runs[name] = Path(path)
    result = audit(runs, args.output)
    print(json.dumps({
        "output": str(args.output),
        "status": result["status"],
        "runs": {
            key: {"parse_success": value["parse_success"], "unparsed_count": value["unparsed_count"], "categories": value["categories"]}
            for key, value in result["runs"].items()
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
