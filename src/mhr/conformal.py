"""Split conformal prediction sets (least ambiguous set-valued classifier, LAC)."""

from __future__ import annotations

import numpy as np


def lac_threshold(proba_cal: np.ndarray, y_cal: np.ndarray, alpha: float) -> float:
    """Conformal quantile of the nonconformity score 1 - p(true class)."""
    scores = 1.0 - proba_cal[np.arange(len(y_cal)), y_cal]
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def lac_sets(proba: np.ndarray, qhat: float) -> np.ndarray:
    """Boolean matrix: class k is in the set when 1 - p_k <= qhat."""
    return (1.0 - proba) <= qhat


def coverage_and_size(sets: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    covered = sets[np.arange(len(y)), y]
    return float(covered.mean()), float(sets.sum(axis=1).mean())
