"""Replay M0 and fail if its frozen deterministic metrics changed."""

import argparse
import json
import math
from pathlib import Path

from health_ai_copilot.eval.runner import evaluate_cases, load_cases
from health_ai_copilot.knowledge.loader import load_knowledge_cards


def compare_metrics(
    actual: dict[str, float | int], expected: dict[str, float | int]
) -> list[str]:
    mismatches: list[str] = []
    for key, expected_value in expected.items():
        if key not in actual:
            mismatches.append(f"missing metric: {key}")
            continue
        actual_value = actual[key]
        if isinstance(expected_value, float):
            matches = isinstance(actual_value, (int, float)) and math.isclose(
                float(actual_value), expected_value, rel_tol=0.0, abs_tol=1e-12
            )
        else:
            matches = actual_value == expected_value
        if not matches:
            mismatches.append(f"{key}: expected {expected_value!r}, got {actual_value!r}")
    return mismatches


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check the frozen deterministic M0 baseline")
    parser.add_argument("--knowledge-dir", default="data/knowledge_cards")
    parser.add_argument("--dataset", default="evals/m0.jsonl")
    parser.add_argument("--baseline", default="evals/baselines/m0.json")
    args = parser.parse_args(argv)

    cards = load_knowledge_cards(args.knowledge_dir)
    cases = load_cases(args.dataset)
    expected: dict[str, float | int] = json.loads(
        Path(args.baseline).read_text(encoding="utf-8")
    )
    actual = evaluate_cases(cases, cards)
    mismatches = compare_metrics(actual, expected)
    if mismatches:
        raise SystemExit("M0 baseline mismatch:\n- " + "\n- ".join(mismatches))
    print(json.dumps(actual, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
