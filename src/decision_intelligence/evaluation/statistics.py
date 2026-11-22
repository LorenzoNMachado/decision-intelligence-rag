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
