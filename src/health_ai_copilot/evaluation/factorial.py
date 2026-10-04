"""Paired summaries for the frozen 2x2 Common Eval reasoning/RAG matrix."""

from __future__ import annotations

from collections.abc import Mapping
from statistics import mean
from typing import Any

CORE_PROFILE_ALIASES = ("B0", "B1", "B2", "B3")


def summarize_core_factorial(
    correctness: Mapping[str, Mapping[str, bool]],
) -> dict[str, Any]:
    """Summarize B0=Single, B1=Single+RAG, B2=Adaptive, B3=Adaptive+RAG.

    All arms must contain the same case IDs. Deltas are paired descriptive
    differences; this function does not perform inferential testing.
    """
    if set(correctness) != set(CORE_PROFILE_ALIASES):
        raise ValueError("core factorial requires exactly B0, B1, B2, and B3")
    case_ids = set(correctness["B0"])
    if not case_ids or any(set(correctness[alias]) != case_ids for alias in CORE_PROFILE_ALIASES):
        raise ValueError("core factorial arms must contain the same non-empty case set")

    accuracies = {
        alias: mean(bool(value) for value in correctness[alias].values())
        for alias in CORE_PROFILE_ALIASES
    }

    def paired(left: str, right: str) -> dict[str, int | float]:
        left_values, right_values = correctness[left], correctness[right]
        wins = sum(not left_values[key] and right_values[key] for key in case_ids)
        losses = sum(left_values[key] and not right_values[key] for key in case_ids)
        ties = len(case_ids) - wins - losses
        return {
            "left": left,
            "right": right,
            "n": len(case_ids),
            "delta_pp": round((accuracies[right] - accuracies[left]) * 100, 4),
            "right_only_correct": wins,
            "left_only_correct": losses,
            "both_same": ties,
        }

    rag_without_adaptive = accuracies["B1"] - accuracies["B0"]
    rag_with_adaptive = accuracies["B3"] - accuracies["B2"]
    adaptive_without_rag = accuracies["B2"] - accuracies["B0"]
    adaptive_with_rag = accuracies["B3"] - accuracies["B1"]
    return {
        "profile_accuracy": {key: round(value, 6) for key, value in accuracies.items()},
        "paired_comparisons": {
            "rag_without_adaptive": paired("B0", "B1"),
            "rag_with_adaptive": paired("B2", "B3"),
            "adaptive_without_rag": paired("B0", "B2"),
            "adaptive_with_rag": paired("B1", "B3"),
        },
        "factor_effects_pp": {
            "rag_main_effect": round(mean((rag_without_adaptive, rag_with_adaptive)) * 100, 4),
            "adaptive_main_effect": round(mean((adaptive_without_rag, adaptive_with_rag)) * 100, 4),
            "rag_adaptive_interaction": round(
                (rag_with_adaptive - rag_without_adaptive) * 100, 4
            ),
        },
        "case_count": len(case_ids),
        "interpretation": "paired descriptive accuracy deltas; no significance test",
    }
