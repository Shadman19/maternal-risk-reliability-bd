"""Metrics used in the analysis. All functions take integer labels and probability arrays."""

from __future__ import annotations

import numpy as np


def accuracy(y: np.ndarray, proba: np.ndarray) -> float:
    return float(np.mean(proba.argmax(axis=1) == y))


def macro_f1(y: np.ndarray, proba: np.ndarray) -> float:
    pred = proba.argmax(axis=1)
    scores = []
    for k in range(proba.shape[1]):
        tp = np.sum((pred == k) & (y == k))
        fp = np.sum((pred == k) & (y != k))
        fn = np.sum((pred != k) & (y == k))
        denom = 2 * tp + fp + fn
        scores.append(0.0 if denom == 0 else 2 * tp / denom)
    return float(np.mean(scores))


def top_label_ece(y: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error of the top-label confidence, equal-width bins on [0, 1]."""
    conf = proba.max(axis=1)
    correct = (proba.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def brier(y: np.ndarray, proba: np.ndarray) -> float:
    onehot = np.eye(proba.shape[1])[y]
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def nll(y: np.ndarray, proba: np.ndarray, eps: float = 1e-12) -> float:
    return float(-np.mean(np.log(np.clip(proba[np.arange(len(y)), y], eps, 1.0))))


def recall_specificity(y_pos: np.ndarray, flag: np.ndarray) -> tuple[float, float]:
    """Recall (sensitivity) and specificity of a binary flag against a binary truth."""
    y_pos = y_pos.astype(bool)
    flag = flag.astype(bool)
    rec = float(flag[y_pos].mean()) if y_pos.any() else float("nan")
    spec = float((~flag[~y_pos]).mean()) if (~y_pos).any() else float("nan")
    return rec, spec


def bootstrap_ci(values_fn, n: int, n_boot: int = 2000, seed: int = 0, groups: np.ndarray | None = None):
    """Percentile 95% interval of values_fn(index_array). Resamples groups when given."""
    rng = np.random.default_rng(seed)
    stats = []
    if groups is None:
        for _ in range(n_boot):
            stats.append(values_fn(rng.integers(0, n, n)))
    else:
        uniq = np.unique(groups)
        members = {g: np.flatnonzero(groups == g) for g in uniq}
        for _ in range(n_boot):
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            stats.append(values_fn(np.concatenate([members[g] for g in pick])))
    lo, hi = np.nanpercentile(stats, [2.5, 97.5])
    return float(lo), float(hi)
