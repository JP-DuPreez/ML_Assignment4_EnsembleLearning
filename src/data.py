from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def _resolve_path(csv_path: str) -> str:
    if os.path.exists(csv_path):
        return csv_path
    # Try common subdir 'Data/' relative to project root
    proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    candidate = os.path.join(proj_root, csv_path)
    if os.path.exists(candidate):
        return candidate
    # Try Data/<name>
    data_candidate = os.path.join(proj_root, "Data", os.path.basename(csv_path))
    if os.path.exists(data_candidate):
        return data_candidate
    return csv_path  # fall back; pandas will raise


def load_dataset(csv_path: str, target_col: str, label_map: Dict) -> Tuple[pd.DataFrame, pd.Series]:
    csv_path = _resolve_path(csv_path)
    # Use Python engine to auto-detect delimiters (tabs/commas) and be tolerant to spaces
    df = pd.read_csv(csv_path, sep=None, engine="python", skipinitialspace=True)
    # Normalize placeholder missing tokens like '?' to NaN
    df = df.replace(to_replace=r"^\s*\?\s*$", value=np.nan, regex=True)
    # Drop duplicates to ensure each case contributes equally
    df = df.drop_duplicates()

    # Normalize column names (strip quotes/spaces, lower-cased for matching)
    original_cols = df.columns.tolist()
    normalized = [str(c).strip().strip('"').strip("'") for c in original_cols]
    df.columns = normalized
    target_norm = target_col.strip().strip('"').strip("'")
    if target_norm not in df.columns:
        # Try case-insensitive match
        lowered = {c.lower(): c for c in df.columns}
        key = target_norm.lower()
        if key in lowered:
            target_norm = lowered[key]
        else:
            raise ValueError(f"Target column '{target_col}' not found in {csv_path}. Available: {df.columns.tolist()}")

    # Identify known categorical columns (currently 'gender' if present)
    categorical_like = []
    if "gender" in df.columns:
        categorical_like.append("gender")

    # Coerce numeric-looking object columns to numeric (except known categoricals and target)
    for c in df.columns:
        if c == target_norm or c in categorical_like:
            continue
        if df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Map labels (handle already-numeric gracefully)
    y_raw = df[target_norm]
    if y_raw.dtype.kind in {"i", "u", "f"}:
        # Assume already encoded 0/1; coerce to int
        y = y_raw.astype(float).round().astype(int)
    else:
        y = y_raw.map(label_map)
    if y.isna().any():
        missing_labels = df[target_col][y.isna()].unique()
        raise ValueError(f"Unmapped labels in target column: {missing_labels}")

    # Drop obvious non-feature identifiers if present
    X = df.drop(columns=[target_norm])
    for col in ["id", "ID", "Id"]:
        if col in X.columns:
            X = X.drop(columns=[col])

    return X, y.astype(int)


def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=seed,
    )


def get_feature_lists(
    X: pd.DataFrame,
    explicit_categorical: Optional[List[str]] = None,
) -> Tuple[List[str], List[str]]:
    # Restrict categorical encoding strictly to the explicitly provided list (e.g., ['gender']).
    # All other columns are treated as numeric features.
    explicit = [c for c in (explicit_categorical or []) if c in X.columns]
    categorical_cols = explicit
    numeric_cols = [c for c in X.columns if c not in categorical_cols]
    return numeric_cols, categorical_cols

