from __future__ import annotations

from typing import Dict, List, Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold
import numpy as np


def build_preprocessors(
    X: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> Dict[str, ColumnTransformer]:
    # For LR/SVM: clip outliers at ±3σ then scale
    class _Clipper:
        def fit(self, X, y=None):
            # compute per-feature mean/std on training fold
            X = pd.DataFrame(X)
            self.mu_ = X.mean(axis=0).to_numpy()
            self.sigma_ = X.std(axis=0, ddof=0).replace(0, 1.0).to_numpy()
            return self
        def transform(self, X):
            X = np.asarray(X)
            lower = self.mu_ - 3.0 * self.sigma_
            upper = self.mu_ + 3.0 * self.sigma_
            return np.clip(X, lower, upper)

    numeric_linear = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("varthresh", VarianceThreshold(threshold=1e-12)),
        ("clip", _Clipper()),
        ("scaler", StandardScaler()),
    ])

    # For tree-based models: no scaling
    numeric_tree = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("varthresh", VarianceThreshold(threshold=1e-12)),
    ])

    # For 'gender': binary encode if exactly two categories; otherwise one-hot
    categorical = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="if_binary")),
    ])

    preprocess_linear = ColumnTransformer(
        transformers=[
            ("num", numeric_linear, numeric_cols),
            ("cat", categorical, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    preprocess_tree = ColumnTransformer(
        transformers=[
            ("num", numeric_tree, numeric_cols),
            ("cat", categorical, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    return {"linear": preprocess_linear, "tree": preprocess_tree}


def get_feature_names_from_transformer(
    transformer: ColumnTransformer,
) -> List[str]:
    # sklearn >= 1.0 exposes get_feature_names_out on ColumnTransformer
    try:
        names = transformer.get_feature_names_out().tolist()
    except Exception:
        # Fallback: best-effort
        names = []
        for name, trans, cols in transformer.transformers_:
            if name == "remainder":
                continue
            if hasattr(trans, "get_feature_names_out"):
                try:
                    subnames = trans.get_feature_names_out(cols)
                except Exception:
                    subnames = cols
            else:
                subnames = cols
            names.extend([f"{name}__{c}" for c in subnames])
    return names

