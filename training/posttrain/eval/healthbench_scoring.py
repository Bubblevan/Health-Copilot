from __future__ import annotations

import math
import random
from collections.abc import Sequence
from typing import Any

LENGTH_ADJUSTMENT_CENTER = 2000
LENGTH_PENALTY_PER_500_CHARS = 0.0147
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20261004


def rubric_score(rubric_items: Sequence[dict[str, Any]], grades: Sequence[dict[str, Any]]) -> float | None:
    if len(rubric_items) != len(grades):
        raise ValueError("Rubric item and grade counts differ")
    positive_points = sum(float(item["points"]) for item in rubric_items if float(item["points"]) > 0)
    if positive_points <= 0:
        return None
    achieved_points = sum(
        float(item["points"])
        for item, grade in zip(rubric_items, grades, strict=True)
        if grade.get("criteria_met") is True
    )
    return achieved_points / positive_points


def length_adjusted_score(score: float, answer: str) -> float:
    return score - LENGTH_PENALTY_PER_500_CHARS * ((len(answer) - LENGTH_ADJUSTMENT_CENTER) / 500.0)


def clipped_mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return min(1.0, max(0.0, sum(values) / len(values)))


def clipped_bootstrap_ci(
    values: Sequence[float], *, seed: int = BOOTSTRAP_SEED, samples: int = BOOTSTRAP_SAMPLES
) -> tuple[float, float]:
    if not values:
        return (math.nan, math.nan)
    rng = random.Random(seed)
    n = len(values)
    means = sorted(min(1.0, max(0.0, sum(values[rng.randrange(n)] for _ in range(n)) / n)) for _ in range(samples))
    return means[int(0.025 * samples)], means[min(samples - 1, int(0.975 * samples))]
