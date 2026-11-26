import numpy as np
import pytest

from mhr import metrics


def test_accuracy_and_macro_f1():
    y = np.array([0, 1, 2, 2])
    p = np.eye(3)[[0, 1, 2, 1]]
    assert metrics.accuracy(y, p) == pytest.approx(0.75)
    # per-class F1: 1.0, 2/3, 2/3
    assert metrics.macro_f1(y, p) == pytest.approx((1 + 2 / 3 + 2 / 3) / 3)


def test_ece_zero_when_confidence_matches_accuracy():
    # 10 predictions at confidence 0.8 with 8 correct
    p = np.tile([0.8, 0.1, 0.1], (10, 1))
    y = np.array([0] * 8 + [1] * 2)
    assert metrics.top_label_ece(y, p) == pytest.approx(0.0, abs=1e-12)


def test_ece_of_overconfident_predictions():
    p = np.tile([0.9, 0.05, 0.05], (10, 1))
    y = np.array([0] * 5 + [1] * 5)
    assert metrics.top_label_ece(y, p) == pytest.approx(0.4)


def test_brier_and_nll_known_values():
    y = np.array([0, 1])
    p = np.array([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0]])
    assert metrics.brier(y, p) == pytest.approx((0.0 + 0.5) / 2)
    assert metrics.nll(y, p) == pytest.approx(-np.log(0.5) / 2)


def test_recall_specificity():
    truth = np.array([1, 1, 0, 0, 0])
    flag = np.array([1, 0, 1, 0, 0])
    rec, spec = metrics.recall_specificity(truth, flag)
    assert rec == pytest.approx(0.5)
    assert spec == pytest.approx(2 / 3)


def test_bootstrap_interval_brackets_the_mean():
    rng = np.random.default_rng(0)
    v = rng.normal(size=400)
    lo, hi = metrics.bootstrap_ci(lambda idx: v[idx].mean(), n=len(v), n_boot=500)
    assert lo < v.mean() < hi


def test_group_bootstrap_runs_with_groups():
    v = np.arange(20, dtype=float)
    groups = np.repeat(np.arange(10), 2)
    lo, hi = metrics.bootstrap_ci(lambda idx: v[idx].mean(), n=20, n_boot=300, groups=groups)
    assert lo <= hi
