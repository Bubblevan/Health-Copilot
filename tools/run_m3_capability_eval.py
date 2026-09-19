"""Standalone live M3 capability-aware policy evaluation on frozen inputs."""

import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from health_ai_copilot.contracts import Evidence
from health_ai_copilot.eval.runner import load_cases
from health_ai_copilot.knowledge.loader import load_knowledge_cards
from health_ai_copilot.knowledge.scope import load_knowledge_scope
from health_ai_copilot.policy.evidence import validate_assessment
from health_ai_copilot.policy.model import OpenAICompatibleEvidencePolicy

DECISIONS = ("sufficient", "recoverable", "insufficient", "conflicting")
REQUIRED_FIELDS = frozenset(
    {
        "id",
        "question",
        "evidence_source_ids",
        "proposed_query",
        "scope_id",
        "expected_decision",
        "expected_topic_ids",
        "category",
        "status",
    }
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run standalone M3 capability evaluation")
    parser.add_argument("--knowledge-dir", default="data/knowledge_cards")
    parser.add_argument("--knowledge-scope", default="data/knowledge_scope.json")
    parser.add_argument("--dataset", default="evals/m3_capability.jsonl")
    parser.add_argument("--output-root", default="runs/m3")
    args = parser.parse_args(argv)
    cards = {card.id: card for card in load_knowledge_cards(args.knowledge_dir)}
    scope = load_knowledge_scope(args.knowledge_scope, list(cards.values()))
    cases = load_cases(args.dataset)
    for case in cases:
        _validate_case(case, cards, scope.scope_id)
    policy = OpenAICompatibleEvidencePolicy(knowledge_scope=scope)
    run_dir = Path(args.output_root) / datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    run_dir.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    for case in cases:
        evidence = _evidence(case["evidence_source_ids"], cards)
        try:
            assessment = validate_assessment(
                policy.assess(case["question"], evidence, case["proposed_query"]),
                evidence,
                scope,
            )
            predicted = assessment.decision.value
            reason_codes = list(assessment.reason_codes)
            supporting_source_ids = list(assessment.supporting_source_ids)
            matched_topic_ids = list(assessment.matched_topic_ids)
            error = None
        except Exception as exc:  # noqa: BLE001 - controlled live evaluation error row
            predicted = None
            reason_codes = []
            supporting_source_ids = []
            matched_topic_ids = []
            error = type(exc).__name__
        rows.append(
            {
                "case_id": case["id"],
                "question": case["question"],
                "evidence_source_ids": case["evidence_source_ids"],
                "proposed_query": case["proposed_query"],
                "scope_id": scope.scope_id,
                "scope_version": scope.version,
                "category": case["category"],
                "expected_decision": case["expected_decision"],
                "expected_topic_ids": case["expected_topic_ids"],
                "predicted_decision": predicted,
                "matched_topic_ids": matched_topic_ids,
                "reason_codes": reason_codes,
                "supporting_source_ids": supporting_source_ids,
                "error": error,
            }
        )
    metrics = capability_metrics(rows)
    _write_jsonl(run_dir / "decisions.jsonl", rows)
    config = _config(args, policy, scope, len(cases))
    _write_json(run_dir / "config.json", config)
    _write_json(run_dir / "metrics.json", metrics)
    (run_dir / "report.md").write_text(_report(metrics, rows), encoding="utf-8")
    print(run_dir)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


def capability_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    confusion = {
        expected: {predicted: 0 for predicted in (*DECISIONS, "error")}
        for expected in DECISIONS
    }
    for row in rows:
        confusion[row["expected_decision"]][row["predicted_decision"] or "error"] += 1
    per_class: dict[str, dict[str, float | int | None]] = {}
    f1_values: list[float] = []
    for decision in DECISIONS:
        true_positive = confusion[decision][decision]
        actual = sum(confusion[decision].values())
        predicted = sum(confusion[expected][decision] for expected in DECISIONS)
        recall = true_positive / actual if actual else None
        precision = true_positive / predicted if predicted else None
        f1 = 2 * precision * recall / (precision + recall) if precision and recall else 0.0
        per_class[decision] = {
            "case_count": actual,
            "precision": precision,
            "recall": recall,
            "f1": f1 if actual else None,
        }
        if actual:
            f1_values.append(f1)
    correct = sum(confusion[decision][decision] for decision in DECISIONS)
    topic_rows = [row for row in rows if row["predicted_decision"] is not None]
    out_of_scope = [row for row in rows if row["category"] == "insufficient_out_of_scope"]
    in_scope = [
        row
        for row in rows
        if row["category"] in {"sufficient_in_scope", "recoverable_in_scope"}
    ]
    return {
        "case_count": len(rows),
        "capability_policy_accuracy": correct / len(rows) if rows else None,
        "capability_macro_f1": sum(f1_values) / len(f1_values) if f1_values else None,
        "sufficient_recall": per_class["sufficient"]["recall"],
        "recoverable_recall": per_class["recoverable"]["recall"],
        "insufficient_recall": per_class["insufficient"]["recall"],
        "matched_topic_exact_set_accuracy": (
            sum(set(row["matched_topic_ids"]) == set(row["expected_topic_ids"]) for row in topic_rows)
            / len(topic_rows)
            if topic_rows
            else None
        ),
        "out_of_scope_false_recovery_rate": (
            sum(row["predicted_decision"] == "recoverable" for row in out_of_scope)
            / len(out_of_scope)
            if out_of_scope
            else None
        ),
        "in_scope_false_reject_rate": (
            sum(row["predicted_decision"] in {"insufficient", "conflicting", None} for row in in_scope)
            / len(in_scope)
            if in_scope
            else None
        ),
        "policy_error_rate": sum(row["error"] is not None for row in rows) / len(rows) if rows else None,
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def _validate_case(case: dict[str, Any], cards: dict[str, Any], scope_id: str) -> None:
    missing = REQUIRED_FIELDS - case.keys()
    if missing:
        raise ValueError(f"M3 capability case missing fields: {sorted(missing)}")
    if case["scope_id"] != scope_id:
        raise ValueError(f"M3 capability case {case['id']} names the wrong scope")
    if case["expected_decision"] not in DECISIONS:
        raise ValueError(f"M3 capability case {case['id']} has an invalid decision")
    if not case["proposed_query"].strip():
        raise ValueError(f"M3 capability case {case['id']} has empty proposed_query")
    unknown = set(case["evidence_source_ids"]) - cards.keys()
    if unknown:
        raise ValueError(f"M3 capability case {case['id']} has unknown evidence: {sorted(unknown)}")


def _evidence(source_ids: list[str], cards: dict[str, Any]) -> list[Evidence]:
    return [
        Evidence(cards[source_id].id, cards[source_id].title, cards[source_id].content, cards[source_id].source_url, 0.0)
        for source_id in source_ids
    ]


def _config(args, policy, scope, case_count: int) -> dict[str, Any]:
    return {
        "commit_sha": _git_sha(),
        "model": policy.model_name,
        "policy_model": policy.model_name,
        "verifier_model": None,
        "same_model_verification": None,
        "base_url": policy.base_url,
        "temperature": policy.temperature,
        "policy_temperature": policy.temperature,
        "verifier_temperature": None,
        "dataset_path": str(Path(args.dataset)),
        "dataset_sha256": _sha256(Path(args.dataset)),
        "knowledge_pack_version": scope.knowledge_pack_version,
        "knowledge_scope_id": scope.scope_id,
        "knowledge_scope_version": scope.version,
        "knowledge_scope_sha256": _sha256(Path(args.knowledge_scope)),
        "case_count": case_count,
        "trial_count": None,
        "timeout": policy.timeout_seconds,
        "retries": policy.max_retries,
        "max_model_turns": 2,
        "max_tool_calls": 1,
        "initial_top_k": 5,
        "recovery_top_k": 3,
    }


def _report(metrics: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    false_rows = [row for row in rows if row["predicted_decision"] != row["expected_decision"]]
    lines = [
        "# M3 standalone capability evaluation",
        "",
        "Frozen evidence, proposed query, and reviewed scope version are materialized directly; BM25 is not invoked.",
        "",
        "```json",
        json.dumps(metrics, ensure_ascii=False, indent=2),
        "```",
        "",
        "## False decisions",
        "",
    ]
    lines.extend(
        f"- `{row['case_id']}`: expected `{row['expected_decision']}`, predicted `{row['predicted_decision']}`, expected_topics=`{row['expected_topic_ids']}`, matched_topics=`{row['matched_topic_ids']}`."
        for row in false_rows
    )
    if not false_rows:
        lines.append("None.")
    return "\n".join(lines) + "\n"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
