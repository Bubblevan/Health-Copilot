from __future__ import annotations

import math
import random
from collections.abc import Sequence


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return (float("nan"), float("nan"))
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def bootstrap_mean_ci(values: Sequence[float], *, seed: int = 20261004, samples: int = 10_000) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(samples))
    return means[int(0.025 * samples)], means[min(samples - 1, int(0.975 * samples))]


def mcnemar_counts(base_correct: Sequence[bool], candidate_correct: Sequence[bool]) -> dict[str, int | float]:
    if len(base_correct) != len(candidate_correct):
        raise ValueError("paired result sequences must have equal length")
    wrong_to_right = sum(not a and b for a, b in zip(base_correct, candidate_correct))
    right_to_wrong = sum(a and not b for a, b in zip(base_correct, candidate_correct))
    discordant = wrong_to_right + right_to_wrong
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(min(wrong_to_right, right_to_wrong) + 1)) / (2 ** discordant)
        p_value = min(1.0, 2 * tail)
    return {
        "wrong_to_right": wrong_to_right,
        "right_to_wrong": right_to_wrong,
        "paired_delta": (wrong_to_right - right_to_wrong) / len(base_correct) if base_correct else float("nan"),
        "mcnemar_exact_p": p_value,
    }
