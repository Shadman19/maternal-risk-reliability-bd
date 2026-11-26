import numpy as np

from mhr import conformal


def _synthetic(n, rng):
    logits = rng.normal(size=(n, 3)) * 2
    p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    y = np.array([rng.choice(3, p=row) for row in p])
    return p, y


def test_lac_coverage_is_near_target_on_exchangeable_data():
    rng = np.random.default_rng(0)
    p_cal, y_cal = _synthetic(2000, rng)
    p_test, y_test = _synthetic(4000, rng)
    q = conformal.lac_threshold(p_cal, y_cal, alpha=0.1)
    cov, size = conformal.coverage_and_size(conformal.lac_sets(p_test, q), y_test)
    assert 0.88 <= cov <= 0.93
    assert 1.0 <= size <= 3.0


def test_sets_contain_true_class_when_threshold_is_one():
    p = np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
    sets = conformal.lac_sets(p, 1.0)
    assert sets.all()
