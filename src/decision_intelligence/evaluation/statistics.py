"""Paired comparisons between configurations.

Questions are the unit of observation. Repetitions of a deterministic
configuration produce identical rows, so they are not independent
observations and are never counted as such: comparisons use one value per
question (the first repetition, or the mean over repetitions for LLM
configurations).
"""

from __future__ import annotations

from math import comb
from typing import Any


def sign_test(wins: int, losses: int) -> float | None:
    """Two-sided exact sign test (ties excluded). ``None`` when there is no untied pair."""

    n = wins + losses
    if n == 0:
        return None
    k = max(wins, losses)
    tail: float = sum(comb(n, i) for i in range(k, n + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired_comparison(first: dict[str, float], second: dict[str, float]) -> dict[str, Any]:
    """How often ``second`` beats ``first`` on the questions both have, with the sign test."""

    shared = sorted(set(first) & set(second))
    wins = sum(second[q] > first[q] for q in shared)
    losses = sum(second[q] < first[q] for q in shared)
    p_value = sign_test(wins, losses)
    return {
        "questions": len(shared),
        "second_better": wins,
        "first_better": losses,
        "ties": len(shared) - wins - losses,
        "sign_test_p": None if p_value is None else round(p_value, 4),
    }


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float] | None:
    """95 % Wilson score interval for a proportion (``None`` when there is no observation)."""

    if total == 0:
        return None
    p = successes / total
    denominator = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denominator
    half = z * ((p * (1 - p) / total + z**2 / (4 * total**2)) ** 0.5) / denominator
    return round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)


def bootstrap_interval(
    values: list[float], seed: int = 2026, resamples: int = 2000
) -> tuple[float, float] | None:
    """Percentile bootstrap 95 % interval of a mean over questions (seeded, deterministic)."""

    if not values:
        return None
    import random

    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    return round(means[int(0.025 * resamples)], 4), round(means[int(0.975 * resamples) - 1], 4)


def paired_difference(first: dict[str, float], second: dict[str, float], seed: int = 2026) -> dict[str, Any]:
    """Mean of (second − first) over shared questions with a paired bootstrap interval."""

    shared = sorted(set(first) & set(second))
    differences = [second[q] - first[q] for q in shared]
    mean = round(sum(differences) / len(differences), 4) if differences else None
    return {"mean_difference": mean, "ci95": bootstrap_interval(differences, seed)}
