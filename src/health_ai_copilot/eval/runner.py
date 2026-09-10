"""Offline M0 evaluator for safety routes and retrieval Hit@K."""

import argparse
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from ..contracts import KnowledgeCard, Route
from ..knowledge.loader import load_knowledge_cards
from ..retrieval.bm25 import BM25Retriever
from ..safety import route_question


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in {path} line {line_number}") from exc
        if not isinstance(case, dict) or not isinstance(case.get("question"), str):
            raise TypeError(f"case on line {line_number} needs a question")
        cases.append(case)
    return cases


def _predicted_safety_route(question: str) -> str:
    result = route_question(question)
    return result.route.value if result else Route.ANSWER.value


def evaluate_cases(
    cases: Iterable[dict[str, Any]], cards: Sequence[KnowledgeCard], top_k: int = 3
) -> dict[str, float | int]:
    cases = list(cases)
    retriever = BM25Retriever(cards)
    safety_route_total = 0
    safety_route_correct = 0
    hit_total = 0
    hit_count = 0

    for case in cases:
        expected_route = case.get("expected_route")
        if expected_route in {"answer", "urgent", "urgent_care", "prescription", "human_review"}:
            expected_route = {
                "urgent": Route.URGENT_CARE.value,
                "prescription": Route.HUMAN_REVIEW.value,
            }.get(expected_route, expected_route)
            safety_route_total += 1
            if _predicted_safety_route(case["question"]) == expected_route:
                safety_route_correct += 1

        expected_source_ids = case.get("expected_source_ids", [])
        if expected_source_ids:
            hit_total += 1
            returned_ids = {
                item.source_id for item in retriever.search(case["question"], top_k=top_k)
            }
            if returned_ids.intersection(expected_source_ids):
                hit_count += 1

    return {
        "num_cases": len(cases),
        "safety_route_cases": safety_route_total,
        "safety_route_accuracy": (
            safety_route_correct / safety_route_total if safety_route_total else 0.0
        ),
        "retrieval_hit_at_3_cases": hit_total,
        "retrieval_hit_at_3": hit_count / hit_total if hit_total else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run deterministic Health-Copilot M0 eval")
    parser.add_argument("--knowledge-dir", default="data/knowledge_cards")
    parser.add_argument("--dataset", default="evals/m0.jsonl")
    args = parser.parse_args(argv)
    cards = load_knowledge_cards(args.knowledge_dir)
    cases = load_cases(args.dataset)
    print(json.dumps(evaluate_cases(cases, cards), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
