from __future__ import annotations

from typing import Dict, Tuple
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def rf_pipeline_and_grid(preprocess, config) -> Tuple[Pipeline, Dict]:
    rf = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        n_jobs=-1,
        oob_score=True,
        bootstrap=True,
        random_state=0,
    )
    pipe = Pipeline([
        ("preprocess", preprocess),
        ("clf", rf),
    ])

    grid = {
        "clf__n_estimators": config["models"]["rf"].get("n_estimators", [300]),
        "clf__max_depth": config["models"]["rf"].get("max_depth", [None, 8, 12]),
        "clf__max_features": config["models"]["rf"].get("max_features", ["sqrt", 0.5]),
        "clf__min_samples_split": config["models"]["rf"].get("min_samples_split", [2, 5]),
    }
    return pipe, grid


def _calibrated_pipeline(base_estimator, preprocess, calib_cfg) -> Pipeline:
    calibrator = CalibratedClassifierCV(
        estimator=base_estimator,
        method=calib_cfg.method,
        cv=calib_cfg.cv,
        n_jobs=None,
    )
    return Pipeline([
        ("preprocess", preprocess),
        ("clf", calibrator),
    ])


def lr_pipeline_and_grid(preprocess, config, calib_cfg) -> Tuple[Pipeline, Dict]:
    base = LogisticRegression(
        penalty="l2",
        solver="liblinear",
        max_iter=1000,
        class_weight="balanced",
        random_state=0,
    )
    pipe = _calibrated_pipeline(base, preprocess, calib_cfg)
    grid = {
        "clf__estimator__C": config["models"]["lr"].get("C", [0.1, 1.0, 10.0]),
    }
    return pipe, grid


def svc_pipeline_and_grid(preprocess, config, calib_cfg) -> Tuple[Pipeline, Dict]:
    base = SVC(
        kernel="rbf",
        probability=False,  # let CalibratedClassifierCV handle probabilities
        class_weight="balanced",
        random_state=0,
    )
    pipe = _calibrated_pipeline(base, preprocess, calib_cfg)
    grid = {
        "clf__estimator__C": config["models"]["svc"].get("C", [1.0, 10.0]),
        "clf__estimator__gamma": config["models"]["svc"].get("gamma", [0.01, 0.1]),
    }
    return pipe, grid


def dt_pipeline_and_grid(preprocess, config, calib_cfg) -> Tuple[Pipeline, Dict]:
    base = DecisionTreeClassifier(
        class_weight="balanced",
        random_state=0,
    )
    pipe = _calibrated_pipeline(base, preprocess, calib_cfg)
    grid = {
        "clf__estimator__max_depth": config["models"]["dt"].get("max_depth", [None, 5, 8, 12]),
        "clf__estimator__min_samples_split": config["models"]["dt"].get("min_samples_split", [2, 5, 10]),
        "clf__estimator__min_samples_leaf": config["models"]["dt"].get("min_samples_leaf", [1, 2, 5]),
        "clf__estimator__ccp_alpha": config["models"]["dt"].get("ccp_alpha", [0.0, 0.001]),
    }
    return pipe, grid

