import numpy as np

from mhr import data


def test_file_matches_recorded_checksum():
    assert data.file_sha256() == data.EXPECTED_SHA256


def test_audit_counts():
    df, X, y = data.load()
    a = data.audit(df)
    assert a["rows"] == 1014
    assert a["unique_rows"] == 452
    assert a["redundant_full_row_copies"] == 562
    assert a["unique_feature_vectors"] == 416
    assert a["rows_sharing_a_feature_vector"] == 884
    assert a["largest_group_size"] == 34
    assert a["feature_vectors_with_conflicting_labels"] == 35
    assert a["class_counts"] == {"low risk": 406, "mid risk": 336, "high risk": 272}
    assert a["heart_rate_below_40"] == 2


def test_labels_and_shapes():
    df, X, y = data.load()
    assert X.shape == (1014, len(data.FEATURES))
    assert set(np.unique(y)) == {0, 1, 2}


def test_duplicate_groups_share_ids_only_for_identical_rows():
    X = np.array([[1.0, 2.0], [1.0, 2.0], [3.0, 4.0], [1.0, 2.5]])
    g = data.duplicate_groups(X)
    assert g[0] == g[1]
    assert len({g[0], g[2], g[3]}) == 3
