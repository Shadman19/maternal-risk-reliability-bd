import numpy as np

from mhr import data, splits


def test_grouped_folds_never_share_a_feature_vector():
    df, X, y = data.load()
    groups = data.duplicate_groups(X)
    for tr, te in splits.grouped_folds(y, groups, n_splits=5, seed=0):
        assert splits.test_rows_seen_in_train(X, tr, te) == 0.0
        assert set(groups[tr]).isdisjoint(set(groups[te]))


def test_row_level_folds_leak_on_this_dataset():
    df, X, y = data.load()
    leaks = [splits.test_rows_seen_in_train(X, tr, te) for tr, te in splits.row_level_folds(y, 5, 0)]
    assert np.mean(leaks) > 0.5


def test_folds_cover_every_row_once():
    df, X, y = data.load()
    groups = data.duplicate_groups(X)
    test_idx = np.concatenate([te for _, te in splits.grouped_folds(y, groups, 5, 1)])
    assert sorted(test_idx.tolist()) == list(range(len(y)))
