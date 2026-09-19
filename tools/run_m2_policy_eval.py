"""Standalone live evaluation of M2 EvidencePolicy on reviewed policy labels."""

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
from health_ai_copilot.policy.model import OpenAICompatibleEvidencePolicy

DECISIONS = ("sufficient", "recoverable", "insufficient", "conflicting")
REQUIRED_CASE_FIELDS = frozenset(
    {
        "id",
        "question",
        "evidence_source_ids",
        "proposed_query",
        "expected_decision",
        "category",
        "status",
    }
)
KNOWLEDGE_PACK_VERSION = "m0.2-2026-09-15"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run standalone M2 policy evaluation")
    parser.add_argument("--knowledge-dir", default="data/knowledge_cards")
    parser.add_argument("--dataset", default="evals/m2_policy.jsonl")
    parser.add_argument("--output-root", default="runs/m2")
    args = parser.parse_args(argv)
    cases = load_cases(args.dataset)
    cards = {card.id: card for card in load_knowledge_cards(args.knowledge_dir)}
    for case in cases:
        _validate_case(case, cards)
    policy = OpenAICompatibleEvidencePolicy()
    run_dir = Path(args.output_root) / datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    run_dir.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    for case in cases:
        evidence = _materialize_evidence(case["evidence_source_ids"], cards)
        try:
            assessment = policy.assess(case["question"], evidence, case["proposed_query"])
            predicted = assessment.decision.value
            reason_codes = list(assessment.reason_codes)
            supporting_source_ids = list(assessment.supporting_source_ids)
            error = None
        except Exception as exc:  # noqa: BLE001 - evaluation records controlled failures
            predicted = None
            reason_codes = []
            supporting_source_ids = []
            error = type(exc).__name__
        rows.append(
            {
                "case_id": case["id"],
                "question": case["question"],
                "category": case["category"],
                "expected_decision": case["expected_decision"],
                "predicted_decision": predicted,
                "proposed_query": case["proposed_query"],
                "reason_codes": reason_codes,
                "supporting_source_ids": supporting_source_ids,
                "evidence_source_ids": [item.source_id for item in evidence],
                "error": error,
            }
        )
    metrics = policy_metrics(rows)
    _write_jsonl(run_dir / "policy_decisions.jsonl", rows)
    config = {
        "commit_sha": _git_sha(),
        "model": policy.model_name,
        "base_url": policy.base_url,
        "temperature": policy.temperature,
        "dataset_path": str(Path(args.dataset)),
        "dataset_sha256": _sha256(Path(args.dataset)),
        "knowledge_pack_version": KNOWLEDGE_PACK_VERSION,
        "case_count": len(cases),
        "timeout": policy.timeout_seconds,
        "retries": policy.max_retries,
    }
    (run_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (run_dir / "report.md").write_text(_report(metrics, rows), encoding="utf-8")
    print(run_dir)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


def policy_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    confusion = {expected: {predicted: 0 for predicted in (*DECISIONS, "error")} for expected in DECISIONS}
    for row in rows:
        expected = row["expected_decision"]
        predicted = row["predicted_decision"] or "error"
        confusion[expected][predicted] += 1
    per_class: dict[str, dict[str, float | int | None]] = {}
    f1_values: list[float] = []
    for decision in DECISIONS:
        tp = confusion[decision][decision]
        actual = sum(confusion[decision].values())
        predicted = sum(confusion[expected][decision] for expected in DECISIONS)
        recall = tp / actual if actual else None
        precision = tp / predicted if predicted else None
        f1 = 2 * precision * recall / (precision + recall) if precision and recall else 0.0
        if actual:
            f1_values.append(f1)
        per_class[decision] = {"case_count": actual, "precision": precision, "recall": recall, "f1": f1 if actual else None}
    correct = sum(confusion[item][item] for item in DECISIONS)
    return {"case_count": len(rows), "policy_accuracy": correct / len(rows) if rows else None, "macro_f1": sum(f1_values) / len(f1_values) if f1_values else None, "sufficient_recall": per_class["sufficient"]["recall"], "recoverable_recall": per_class["recoverable"]["recall"], "insufficient_recall": per_class["insufficient"]["recall"], "per_class": per_class, "confusion_matrix": confusion}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _report(metrics: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    false_rows = [row for row in rows if row["predicted_decision"] != row["expected_decision"]]
    lines = [
        "# M2 standalone policy evaluation",
        "",
        "The evaluator materializes frozen evidence_source_ids and never invokes BM25.",
        "",
        "## Metrics",
        "",
        "```json",
        json.dumps(metrics, ensure_ascii=False, indent=2),
        "```",
        "",
        "## False decisions",
        "",
    ]
    if not false_rows:
        lines.append("None.")
    else:
        for row in false_rows:
            lines.extend(
                [
                    (
                        f"- `{row['case_id']}`: expected `{row['expected_decision']}`, "
                        f"predicted `{row['predicted_decision']}`; reason_codes="
                        f"`{row['reason_codes']}`; proposed_query=`{row['proposed_query']}`; "
                        f"evidence_source_ids=`{row['evidence_source_ids']}`."
                    ),
                ]
            )
    return "\n".join(lines) + "\n"


def _validate_case(case: dict[str, Any], cards: dict[str, Any]) -> None:
    missing = REQUIRED_CASE_FIELDS - case.keys()
    if missing:
        raise ValueError(f"policy case {case.get('id', '<unknown>')} missing fields: {sorted(missing)}")
    if not case["proposed_query"].strip():
        raise ValueError(f"policy case {case['id']} has an empty proposed_query")
    if case["expected_decision"] not in DECISIONS:
        raise ValueError(f"policy case {case['id']} has an invalid expected_decision")
    unknown = set(case["evidence_source_ids"]) - cards.keys()
    if unknown:
        raise ValueError(f"policy case {case['id']} references unknown evidence: {sorted(unknown)}")


def _materialize_evidence(source_ids: list[str], cards: dict[str, Any]) -> list[Evidence]:
    return [
        Evidence(card.id, card.title, card.content, card.source_url, 0.0)
        for source_id in source_ids
        for card in (cards[source_id],)
    ]


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
