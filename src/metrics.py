from __future__ import annotations

from typing import Callable, Dict, Tuple
import json
import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_proba, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "average_precision": average_precision_score(y_true, y_proba),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "accuracy": accuracy_score(y_true, y_pred),
        "brier": brier_score_loss(y_true, y_proba),
    }


def expected_calibration_error(y_true, y_proba, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.digitize(y_proba, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        mask = binids == b
        if not np.any(mask):
            continue
        conf = y_proba[mask].mean()
        acc = y_true[mask].mean()
        ece += np.abs(acc - conf) * mask.mean()
    return float(ece)


def bootstrap_ci(
    y_true,
    y_proba,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats_vec = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        s = metric_func(y_true[idx], y_proba[idx])
        stats_vec.append(s)
    low = np.percentile(stats_vec, 100 * (alpha / 2))
    high = np.percentile(stats_vec, 100 * (1 - alpha / 2))
    return float(low), float(high)


def paired_wilcoxon(x, y) -> Dict[str, float]:
    stat, p = stats.wilcoxon(x, y, zero_method="wilcox", correction=True)
    return {"stat": float(stat), "pvalue": float(p)}


def save_json(path: str, obj: Dict) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

