from __future__ import annotations

import argparse
import os
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from collections import Counter

from .config import Config, load_config
from .data import get_feature_lists, load_dataset, stratified_split
from .ensembles import build_soft_voting, build_stacking
from .metrics import (
    bootstrap_ci,
    compute_metrics,
    expected_calibration_error,
    paired_wilcoxon,
    save_json,
)
from .models import (
    dt_pipeline_and_grid,
    lr_pipeline_and_grid,
    rf_pipeline_and_grid,
    svc_pipeline_and_grid,
)
from .preprocess import build_preprocessors
from .plots import plot_calibration, plot_roc_pr


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def train_and_select(
    name: str,
    pipe,
    grid: Dict,
    X_train,
    y_train,
    cv,
    scoring: str = "roc_auc",
    n_jobs: int = -1,
    verbose: int = 0,
):
    search = GridSearchCV(
        estimator=pipe,
        param_grid=grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
        return_train_score=False,
        verbose=verbose,
    )
    search.fit(X_train, y_train)
    return search


def run_seed(
    seed: int,
    cfg: Config,
    run_modes: List[str],
    base_dir: str,
):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    seed_dirname = f"{timestamp}-seed{seed}"
    homo_dir = os.path.join(base_dir, "homogeneous", seed_dirname)
    hetero_dir = os.path.join(base_dir, "heterogeneous", seed_dirname)
    ensure_dir(homo_dir)
    ensure_dir(hetero_dir)

    # Data
    X, y = load_dataset(cfg.data.path, cfg.data.target, cfg.data.label_map)
    X_train, X_test, y_train, y_test = stratified_split(X, y, cfg.splits.test_size, seed)
    num_cols, cat_cols = get_feature_lists(X_train, cfg.data.categorical)
    preprocessors = build_preprocessors(X_train, num_cols, cat_cols)

    # CV definition (same for all models)
    cv = StratifiedKFold(n_splits=cfg.splits.cv_folds, shuffle=True, random_state=seed)

    # Homogeneous RF
    rf_pipe, rf_grid = rf_pipeline_and_grid(preprocessors["tree"], cfg.__dict__)
    rf_gs = train_and_select("rf", rf_pipe, rf_grid, X_train, y_train, cv)

    # Save RF CV results
    pd.DataFrame(rf_gs.cv_results_).to_csv(os.path.join(homo_dir, "rf_cv_results.csv"), index=False)
    save_json(os.path.join(homo_dir, "best_params.json"), {
        "model": "RandomForest",
        "best_params": rf_gs.best_params_,
        "best_score_roc_auc": float(rf_gs.best_score_),
    })

    # Fold-wise CV evaluation for RF (on the selected model)
    scoring = {
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
        "f1_macro": "f1_macro",
        "accuracy": "accuracy",
        "brier": "neg_brier_score",
    }
    rf_cv = cross_validate(
        rf_gs.best_estimator_, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1, return_estimator=False
    )
    pd.DataFrame(rf_cv).to_csv(os.path.join(homo_dir, "rf_cv_fold_scores.csv"), index=False)

    # Heterogeneous bases (calibrated)
    best_estimators: List[Tuple[str, object]] = []
    # LR (linear preprocess)
    lr_pipe, lr_grid = lr_pipeline_and_grid(preprocessors["linear"], cfg.__dict__, cfg.calibration)
    lr_gs = train_and_select("lr", lr_pipe, lr_grid, X_train, y_train, cv)
    pd.DataFrame(lr_gs.cv_results_).to_csv(os.path.join(hetero_dir, "lr_cv_results.csv"), index=False)
    save_json(os.path.join(hetero_dir, "lr_best_params.json"), {
        "model": "LogisticRegression+Calibration",
        "best_params": lr_gs.best_params_,
        "best_score_roc_auc": float(lr_gs.best_score_),
    })
    best_estimators.append(("lr", lr_gs.best_estimator_))
    # Fold-wise CV evaluation for LR (on the selected model)
    lr_cv = cross_validate(
        lr_gs.best_estimator_, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1, return_estimator=False
    )
    pd.DataFrame(lr_cv).to_csv(os.path.join(hetero_dir, "lr_cv_fold_scores.csv"), index=False)

    # SVC (linear preprocess)
    svc_pipe, svc_grid = svc_pipeline_and_grid(preprocessors["linear"], cfg.__dict__, cfg.calibration)
    svc_gs = train_and_select("svc", svc_pipe, svc_grid, X_train, y_train, cv)
    pd.DataFrame(svc_gs.cv_results_).to_csv(os.path.join(hetero_dir, "svc_cv_results.csv"), index=False)
    save_json(os.path.join(hetero_dir, "svc_best_params.json"), {
        "model": "SVC(RBF)+Calibration",
        "best_params": svc_gs.best_params_,
        "best_score_roc_auc": float(svc_gs.best_score_),
    })
    best_estimators.append(("svc", svc_gs.best_estimator_))
    # Fold-wise CV evaluation for SVC (on the selected model)
    svc_cv = cross_validate(
        svc_gs.best_estimator_, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1, return_estimator=False
    )
    pd.DataFrame(svc_cv).to_csv(os.path.join(hetero_dir, "svc_cv_fold_scores.csv"), index=False)

    # Decision Tree (tree preprocess)
    dt_pipe, dt_grid = dt_pipeline_and_grid(preprocessors["tree"], cfg.__dict__, cfg.calibration)
    dt_gs = train_and_select("dt", dt_pipe, dt_grid, X_train, y_train, cv)
    pd.DataFrame(dt_gs.cv_results_).to_csv(os.path.join(hetero_dir, "dt_cv_results.csv"), index=False)
    save_json(os.path.join(hetero_dir, "dt_best_params.json"), {
        "model": "DecisionTree+Calibration",
        "best_params": dt_gs.best_params_,
        "best_score_roc_auc": float(dt_gs.best_score_),
    })
    best_estimators.append(("dt", dt_gs.best_estimator_))
    # Fold-wise CV evaluation for DT (on the selected model)
    dt_cv = cross_validate(
        dt_gs.best_estimator_, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1, return_estimator=False
    )
    pd.DataFrame(dt_cv).to_csv(os.path.join(hetero_dir, "dt_cv_fold_scores.csv"), index=False)

    # Evaluate ensembles via CV
    results_cv = {}
    if "soft" in run_modes or "all" in run_modes:
        soft = build_soft_voting(best_estimators)
        soft_cv = cross_validate(soft, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        pd.DataFrame(soft_cv).to_csv(os.path.join(hetero_dir, "soft_cv_fold_scores.csv"), index=False)
        results_cv["soft"] = soft_cv

    if "stacking" in run_modes or "all" in run_modes:
        stack = build_stacking(
            best_estimators, meta_C=float(cfg.ensembles.get("meta", {}).get("C", 1.0)), cv_folds=cfg.splits.cv_folds, seed=seed
        )
        stack_cv = cross_validate(stack, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        pd.DataFrame(stack_cv).to_csv(os.path.join(hetero_dir, "stack_cv_fold_scores.csv"), index=False)
        results_cv["stacking"] = stack_cv

    # Test-set evaluation and artifacts
    # RF already refit on full train via GridSearchCV
    rf_best = rf_gs.best_estimator_
    rf_oob = None
    try:
        rf_oob = getattr(rf_best.named_steps["clf"], "oob_score_", None)
    except Exception:
        rf_oob = None

    def eval_and_save(name: str, est, X_tr, y_tr, X_te, y_te, out_dir_for_this: str):
        # Fit if not already fitted
        try:
            getattr(est, "predict")
            # attempt to predict to confirm fitted
            est.predict(X_tr.iloc[:1])
        except Exception:
            est.fit(X_tr, y_tr)
        proba = est.predict_proba(X_te)[:, 1]
        m = compute_metrics(y_te.values, proba)
        ece = expected_calibration_error(y_te.values, proba)
        m["ece"] = ece
        save_json(os.path.join(out_dir_for_this, f"{name}_test_metrics.json"), m)
        # predictions
        pd.DataFrame({"y_true": y_te.values, "y_proba": proba}).to_csv(
            os.path.join(out_dir_for_this, f"{name}_test_predictions.csv"), index=False
        )
        # plots
        plot_roc_pr(y_te.values, proba, out_dir_for_this, title_prefix=name)
        plot_calibration(y_te.values, proba, out_dir_for_this, title_prefix=name, n_bins=10)
        # bootstrap CI for ROC-AUC
        low, high = bootstrap_ci(
            y_te.values,
            proba,
            metric_func=lambda yt, yp: __import__("sklearn.metrics").metrics.roc_auc_score(yt, yp),
            seed=seed,
        )
        save_json(os.path.join(out_dir_for_this, f"{name}_test_auc_ci.json"), {"low": low, "high": high})
        return m

    rf_test_metrics = eval_and_save("rf", rf_best, X_train, y_train, X_test, y_test, homo_dir)
    if rf_oob is not None:
        save_json(os.path.join(homo_dir, "rf_oob.json"), {"oob_score": float(rf_oob)})

    soft_test_metrics = None
    if "soft" in run_modes or "all" in run_modes:
        soft = build_soft_voting(best_estimators)
        soft_test_metrics = eval_and_save("soft", soft, X_train, y_train, X_test, y_test, hetero_dir)

    stack_test_metrics = None
    if "stacking" in run_modes or "all" in run_modes:
        stack = build_stacking(
            best_estimators, meta_C=float(cfg.ensembles.get("meta", {}).get("C", 1.0)), cv_folds=cfg.splits.cv_folds, seed=seed
        )
        stack_test_metrics = eval_and_save("stacking", stack, X_train, y_train, X_test, y_test, hetero_dir)

    # Also evaluate base models on the held-out test set
    lr_test_metrics = eval_and_save("lr", lr_gs.best_estimator_, X_train, y_train, X_test, y_test, hetero_dir)
    svc_test_metrics = eval_and_save("svc", svc_gs.best_estimator_, X_train, y_train, X_test, y_test, hetero_dir)
    dt_test_metrics = eval_and_save("dt", dt_gs.best_estimator_, X_train, y_train, X_test, y_test, hetero_dir)

    # Prepare fold-wise comparison for stats (use ROC-AUC)
    rf_auc = rf_cv["test_roc_auc"]
    agg = {"seed": seed, "rf_cv_roc_auc": rf_auc.tolist()}
    # Include base-model CV AUCs in per-seed summary
    try:
        agg["lr_cv_roc_auc"] = lr_cv["test_roc_auc"].tolist()
        agg["svc_cv_roc_auc"] = svc_cv["test_roc_auc"].tolist()
        agg["dt_cv_roc_auc"] = dt_cv["test_roc_auc"].tolist()
    except Exception:
        pass
    if "soft" in results_cv:
        agg["soft_cv_roc_auc"] = results_cv["soft"]["test_roc_auc"].tolist()
    if "stacking" in results_cv:
        agg["stacking_cv_roc_auc"] = results_cv["stacking"]["test_roc_auc"].tolist()
    # Save per-seed CV ROC-AUC summary under results/summary/seeds/
    seed_summ_dir = os.path.join(base_dir, "summary", "seeds")
    ensure_dir(seed_summ_dir)
    save_json(os.path.join(seed_summ_dir, f"{seed_dirname}.json"), agg)

    # Return necessary info for global aggregation
    return {
        "seed": seed,
        "rf_cv": rf_cv,
        "soft_cv": results_cv.get("soft"),
        "stack_cv": results_cv.get("stacking"),
        "rf_test": rf_test_metrics,
        "soft_test": soft_test_metrics,
        "stack_test": stack_test_metrics,
        "lr_cv": lr_cv,
        "svc_cv": svc_cv,
        "dt_cv": dt_cv,
        "lr_test": lr_test_metrics,
        "svc_test": svc_test_metrics,
        "dt_test": dt_test_metrics,
        "best_params": {
            "rf": {"params": rf_gs.best_params_, "score": float(rf_gs.best_score_)},
            "lr": {"params": lr_gs.best_params_, "score": float(lr_gs.best_score_)},
            "svc": {"params": svc_gs.best_params_, "score": float(svc_gs.best_score_)},
            "dt": {"params": dt_gs.best_params_, "score": float(dt_gs.best_score_)},
        },
    }


def aggregate_and_save(all_runs: List[Dict], base_dir: str) -> None:
    summ_dir = os.path.join(base_dir, "summary")
    ensure_dir(summ_dir)

    # Collect fold-wise ROC-AUC across seeds
    rf_auc = []
    soft_auc = []
    stack_auc = []
    lr_auc = []
    svc_auc = []
    dt_auc = []
    test_records = []
    for run in all_runs:
        rf_auc.extend(run["rf_cv"]["test_roc_auc"])  # type: ignore
        if run["soft_cv"] is not None:
            soft_auc.extend(run["soft_cv"]["test_roc_auc"])  # type: ignore
        if run["stack_cv"] is not None:
            stack_auc.extend(run["stack_cv"]["test_roc_auc"])  # type: ignore
        if run.get("lr_cv") is not None:
            lr_auc.extend(run["lr_cv"]["test_roc_auc"])  # type: ignore
        if run.get("svc_cv") is not None:
            svc_auc.extend(run["svc_cv"]["test_roc_auc"])  # type: ignore
        if run.get("dt_cv") is not None:
            dt_auc.extend(run["dt_cv"]["test_roc_auc"])  # type: ignore
        # test metrics snapshot
        test_records.append({
            "seed": run["seed"],
            "rf": run["rf_test"],
            "soft": run["soft_test"],
            "stack": run["stack_test"],
        })

    def mean_std(arr):
        if len(arr) == 0:
            return {"mean": None, "std": None}
        return {"mean": float(np.mean(arr)), "std": float(np.std(arr, ddof=1) if len(arr) > 1 else 0.0)}

    summary = {
        "cv_roc_auc": {
            "rf": mean_std(rf_auc),
            "lr": mean_std(lr_auc) if len(lr_auc) else None,
            "svc": mean_std(svc_auc) if len(svc_auc) else None,
            "dt": mean_std(dt_auc) if len(dt_auc) else None,
            "soft": mean_std(soft_auc) if len(soft_auc) else None,
            "stacking": mean_std(stack_auc) if len(stack_auc) else None,
        },
        "test_metrics": test_records,
    }

    # Paired Wilcoxon RF vs Soft/Stacking (if present), using fold-wise ROC-AUC
    if len(soft_auc) and len(rf_auc) and len(soft_auc) == len(rf_auc):
        summary["paired_rf_vs_soft_wilcoxon"] = paired_wilcoxon(np.array(rf_auc), np.array(soft_auc))
    if len(stack_auc) and len(rf_auc) and len(stack_auc) == len(rf_auc):
        summary["paired_rf_vs_stack_wilcoxon"] = paired_wilcoxon(np.array(rf_auc), np.array(stack_auc))

    save_json(os.path.join(summ_dir, "summary.json"), summary)

    # Aggregate best params across seeds for RF, LR, SVC, DT
    model_keys = ["rf", "lr", "svc", "dt"]
    agg_params = {}
    for key in model_keys:
        records = []
        for run in all_runs:
            if "best_params" in run and key in run["best_params"]:
                bp = run["best_params"][key]
                records.append({
                    "seed": run["seed"],
                    "best_params": bp["params"],
                    "best_score_roc_auc": bp["score"],
                })
        # summarize
        if records:
            # frequency of param dicts
            def norm_tuple(d):
                return tuple(sorted(d.items()))
            freq = Counter([norm_tuple(r["best_params"]) for r in records])
            most_common = freq.most_common(1)[0][0] if len(freq) else None
            most_common_dict = dict(most_common) if most_common else None
            score_mean = float(np.mean([r["best_score_roc_auc"] for r in records]))
            score_std = float(np.std([r["best_score_roc_auc"] for r in records], ddof=1)) if len(records) > 1 else 0.0
            agg_params[key] = {
                "per_seed": records,
                "most_frequent_params": most_common_dict,
                "best_score_roc_auc_mean": score_mean,
                "best_score_roc_auc_std": score_std,
            }
    save_json(os.path.join(summ_dir, "best_params.json"), agg_params)

    # Final metrics summary across seeds for CV and Test (per model)
    def mean_std_arr(values: List[float]):
        if not values:
            return {"mean": None, "std": None, "n": 0}
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1) if len(values) > 1 else 0.0),
            "n": int(len(values)),
        }

    cv_metric_keys = ["roc_auc", "average_precision", "f1_macro", "accuracy", "brier"]

    def gather_cv(model_key: str):
        rows = {m: [] for m in cv_metric_keys}
        for run in all_runs:
            if model_key == "rf":
                cvres = run.get("rf_cv")
            elif model_key == "soft":
                cvres = run.get("soft_cv")
            elif model_key == "stack":
                cvres = run.get("stack_cv")
            elif model_key == "lr":
                cvres = run.get("lr_cv")
            elif model_key == "svc":
                cvres = run.get("svc_cv")
            elif model_key == "dt":
                cvres = run.get("dt_cv")
            else:
                cvres = None
            if cvres is None:
                continue
            for m in cv_metric_keys:
                arr = cvres.get(f"test_{m}")
                if arr is None:
                    continue
                vals = list(arr)
                if m == "brier":
                    # stored as neg_brier_score in CV; invert sign back to Brier score
                    vals = [-v for v in vals]
                rows[m].extend([float(v) for v in vals])
        return {m: mean_std_arr(rows[m]) for m in cv_metric_keys}

    test_metric_keys = ["roc_auc", "average_precision", "f1_macro", "accuracy", "brier", "ece"]

    def gather_test(model_key: str):
        rows = {m: [] for m in test_metric_keys}
        for run in all_runs:
            if model_key == "rf":
                tres = run.get("rf_test")
            elif model_key == "soft":
                tres = run.get("soft_test")
            elif model_key == "stack":
                tres = run.get("stack_test")
            elif model_key == "lr":
                tres = run.get("lr_test")
            elif model_key == "svc":
                tres = run.get("svc_test")
            elif model_key == "dt":
                tres = run.get("dt_test")
            else:
                tres = None
            if not tres:
                continue
            for m in test_metric_keys:
                if m in tres and tres[m] is not None:
                    rows[m].append(float(tres[m]))
        return {m: mean_std_arr(rows[m]) for m in test_metric_keys}

    final_metrics = {
        "seeds": [int(run["seed"]) for run in all_runs],
        "cv": {
            "rf": gather_cv("rf"),
            "lr": gather_cv("lr"),
            "svc": gather_cv("svc"),
            "dt": gather_cv("dt"),
            "soft": gather_cv("soft"),
            "stacking": gather_cv("stack"),
        },
        "test": {
            "rf": gather_test("rf"),
            "lr": gather_test("lr"),
            "svc": gather_test("svc"),
            "dt": gather_test("dt"),
            "soft": gather_test("soft"),
            "stacking": gather_test("stack"),
        },
    }
    save_json(os.path.join(summ_dir, "final_metrics_summary.json"), final_metrics)

    # Lightweight Markdown summary
    lines = [
        "### Cross-validated ROC-AUC (mean ± std)",
        f"- RF: {summary['cv_roc_auc']['rf']}",
        f"- Soft: {summary['cv_roc_auc']['soft']}",
        f"- Stacking: {summary['cv_roc_auc']['stacking']}",
        "",
        "### Paired tests (Wilcoxon, fold-wise ROC-AUC)",
        f"- RF vs Soft: {summary.get('paired_rf_vs_soft_wilcoxon')}",
        f"- RF vs Stacking: {summary.get('paired_rf_vs_stack_wilcoxon')}",
    ]
    with open(os.path.join(summ_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Breast Cancer Ensembles Experiment Runner")
    p.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config")
    p.add_argument(
        "--run",
        type=str,
        default="all",
        choices=["all", "rf", "soft", "stacking"],
        help="Which ensemble(s) to run",
    )
    p.add_argument("--seeds", type=int, nargs="*", default=None, help="Override seeds list")
    p.add_argument("--outdir", type=str, default="results", help="Artifacts output directory")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.seeds:
        cfg.splits.seeds = list(args.seeds)

    base_dir = os.path.abspath(args.outdir)
    ensure_dir(base_dir)

    run_modes = [args.run]
    if args.run == "all":
        run_modes = ["all"]  # downstream code checks this
    elif args.run == "rf":
        run_modes = ["rf"]
    elif args.run == "soft":
        run_modes = ["soft"]
    elif args.run == "stacking":
        run_modes = ["stacking"]

    all_runs = []
    for seed in cfg.splits.seeds:
        res = run_seed(seed, cfg, run_modes, base_dir)
        all_runs.append(res)

    aggregate_and_save(all_runs, base_dir)


if __name__ == "__main__":
    main()

