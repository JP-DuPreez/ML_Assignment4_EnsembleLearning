from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import yaml


@dataclass
class CalibrationConfig:
    enabled: bool = True
    method: str = "sigmoid"
    cv: int = 5


@dataclass
class SplitsConfig:
    test_size: float = 0.2
    cv_folds: int = 5
    seeds: List[int] = None  # type: ignore


@dataclass
class DataConfig:
    path: str = "breastCancer.csv"
    target: str = "diagnosis"
    label_map: Dict[Any, int] = None  # type: ignore
    categorical: Optional[List[str]] = None


@dataclass
class Config:
    data: DataConfig
    splits: SplitsConfig
    calibration: CalibrationConfig
    models: Dict[str, Any]
    metrics: List[str]
    ensembles: Dict[str, Any]
    plotting: Dict[str, Any]


def load_config(path: str) -> Config:
    with open(path, "r") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)

    data = DataConfig(**raw.get("data", {}))
    splits = SplitsConfig(**raw.get("splits", {}))
    calibration = CalibrationConfig(**raw.get("calibration", {}))
    models = raw.get("models", {})
    metrics = raw.get("metrics", ["roc_auc"])  # default primary
    ensembles = raw.get("ensembles", {"soft_vote": True, "stacking": True})
    plotting = raw.get("plotting", {"enabled": True})

    # defaults if missing
    if splits.seeds is None:
        splits.seeds = [42]
    if data.label_map is None:
        data.label_map = {"B": 0, "M": 1}

    return Config(
        data=data,
        splits=splits,
        calibration=calibration,
        models=models,
        metrics=metrics,
        ensembles=ensembles,
        plotting=plotting,
    )

