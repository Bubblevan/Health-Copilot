"""Standalone live evaluation of M2 EvidencePolicy on reviewed policy labels."""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from health_ai_copilot.eval.runner import load_cases
from health_ai_copilot.knowledge.loader import load_knowledge_cards
from health_ai_copilot.policy.model import OpenAICompatibleEvidencePolicy
from health_ai_copilot.retrieval.bm25 import BM25Retriever

DECISIONS = ("sufficient", "recoverable", "insufficient", "conflicting")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run standalone M2 policy evaluation")
    parser.add_argument("--knowledge-dir", default="data/knowledge_cards")
    parser.add_argument("--dataset", default="evals/m2_policy.jsonl")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-root", default="runs/m2")
    args = parser.parse_args(argv)
    cases = load_cases(args.dataset)
    retriever = BM25Retriever(load_knowledge_cards(args.knowledge_dir))
    policy = OpenAICompatibleEvidencePolicy()
    run_dir = Path(args.output_root) / datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    run_dir.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    for case in cases:
        evidence = retriever.search(case["question"], top_k=args.top_k)
        try:
            assessment = policy.assess(case["question"], evidence, case["question"])
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
                "reason_codes": reason_codes,
                "supporting_source_ids": supporting_source_ids,
                "evidence_source_ids": [item.source_id for item in evidence],
                "error": error,
            }
        )
    metrics = policy_metrics(rows)
    _write_jsonl(run_dir / "policy_decisions.jsonl", rows)
    (run_dir / "config.json").write_text(json.dumps({"dataset": args.dataset, "top_k": args.top_k, "case_count": len(cases), "model": policy.model_name}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (run_dir / "report.md").write_text("# M2 standalone policy evaluation\n\n```json\n" + json.dumps(metrics, ensure_ascii=False, indent=2) + "\n```\n", encoding="utf-8")
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


if __name__ == "__main__":
    raise SystemExit(main())
