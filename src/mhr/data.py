"""Loading and auditing the Maternal Health Risk data."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]
TARGET = "RiskLevel"
CLASSES = ["low risk", "mid risk", "high risk"]

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "maternal_health_risk.csv"
EXPECTED_SHA256 = "2b60ea74439e1e838c381ab81577feb2e01284dc4f3d78e014b7bc68fce32f16"


def file_sha256(path: Path = DATA_PATH) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path: Path = DATA_PATH) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Return the dataframe, the feature matrix and integer labels (0 low, 1 mid, 2 high)."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    y = df[TARGET].map({c: i for i, c in enumerate(CLASSES)})
    if y.isna().any():
        raise ValueError("unexpected label values")
    return df, df[FEATURES].to_numpy(dtype=float), y.to_numpy(dtype=int)


def duplicate_groups(X: np.ndarray) -> np.ndarray:
    """Group id per row: rows with identical feature vectors share an id."""
    _, inverse = np.unique(X, axis=0, return_inverse=True)
    return inverse.ravel()


def audit(df: pd.DataFrame) -> dict:
    """Counts that describe duplication, label conflicts and implausible values."""
    groups = df.groupby(FEATURES)[TARGET]
    sizes = groups.size()
    conflicts = int((groups.nunique() > 1).sum())
    return {
        "rows": int(len(df)),
        "unique_feature_vectors": int(len(sizes)),
        "unique_rows": int(len(df.drop_duplicates())),
        "redundant_full_row_copies": int(df.duplicated().sum()),
        "rows_sharing_a_feature_vector": int(sizes[sizes > 1].sum()),
        "largest_group_size": int(sizes.max()),
        "feature_vectors_with_conflicting_labels": conflicts,
        "class_counts": {c: int((df[TARGET] == c).sum()) for c in CLASSES},
        "heart_rate_below_40": int((df["HeartRate"] < 40).sum()),
        "age_below_15": int((df["Age"] < 15).sum()),
        "age_above_50": int((df["Age"] > 50).sum()),
    }
