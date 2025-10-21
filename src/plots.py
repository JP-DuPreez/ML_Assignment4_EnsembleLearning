from __future__ import annotations

import os
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc, precision_recall_curve, roc_curve


def plot_roc_pr(y_true, y_proba, out_dir: str, title_prefix: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)

    # ROC
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{title_prefix} ROC")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{title_prefix}_roc.png"), dpi=150)
    plt.close()

    # PR
    plt.figure()
    plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{title_prefix} PR")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{title_prefix}_pr.png"), dpi=150)
    plt.close()


def plot_calibration(y_true, y_proba, out_dir: str, title_prefix: str, n_bins: int = 10) -> None:
    os.makedirs(out_dir, exist_ok=True)
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy="uniform")

    plt.figure()
    plt.plot(prob_pred, prob_true, marker="o", label="Reliability curve")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    plt.xlabel("Predicted probability")
    plt.ylabel("True probability in bin")
    plt.title(f"{title_prefix} Calibration")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{title_prefix}_calibration.png"), dpi=150)
    plt.close()

