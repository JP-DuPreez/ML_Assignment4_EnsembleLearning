from __future__ import annotations

from typing import Dict, List, Tuple
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


def build_soft_voting(best_named_estimators: List[Tuple[str, object]]) -> VotingClassifier:
    return VotingClassifier(
        estimators=best_named_estimators,
        voting="soft",
        n_jobs=-1,
        flatten_transform=False,
    )


def build_stacking(
    best_named_estimators: List[Tuple[str, object]],
    meta_C: float = 1.0,
    cv_folds: int = 5,
    seed: int = 0,
) -> StackingClassifier:
    meta = LogisticRegression(
        penalty="l2",
        C=meta_C,
        solver="liblinear",
        max_iter=1000,
        class_weight="balanced",
        random_state=seed,
    )
    return StackingClassifier(
        estimators=best_named_estimators,
        final_estimator=meta,
        stack_method="predict_proba",
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed),
        passthrough=False,
        n_jobs=None,
    )

