"""Cross-validation schemes: row-level (leaky with duplicates) and duplicate-aware."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


def row_level_folds(y: np.ndarray, n_splits: int, seed: int):
    """Standard stratified K-fold on rows. Identical records can land on both sides."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y))


def grouped_folds(y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int):
    """Stratified group K-fold: all copies of a feature vector stay in one fold."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(sgkf.split(np.zeros(len(y)), y, groups))


def test_rows_seen_in_train(X: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray) -> float:
    """Fraction of test rows whose exact feature vector also appears in the training fold."""
    seen = {tuple(r) for r in X[train_idx]}
    return float(np.mean([tuple(r) in seen for r in X[test_idx]]))
