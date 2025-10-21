## Breast Cancer Ensembles (Homogeneous vs Heterogeneous)

This project implements the full specification in `ProjectSpec.md` on the provided `breastCancer.csv` dataset. It compares a homogeneous Random Forest ensemble against a heterogeneous ensemble (Logistic Regression + SVM (RBF) + Decision Tree) using calibrated probabilities for soft voting and stacking with a logistic-regression meta-learner trained on out-of-fold predictions.

### Quickstart

1) Create and activate a virtual environment, then install dependencies:

```bash
cd /Users/jpdupreez/Downloads/CompSci/4.MachineLearning/Assignment_4
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```

2) Run the full experiment (uses `config.yaml` by default):

```bash
python3 -m src.train --config config.yaml
```

Flags:

- `--run rf` to run only the homogeneous Random Forest
- `--run soft` to run only the calibrated soft voting ensemble
- `--run stacking` to run only the calibrated stacking ensemble
- `--run all` (default) to run all three
- `--seeds 42 1337 2020 7 99` overrides seeds from config

Artifacts saved under `results/` are split by ensemble type:

- `results/homogeneous/<timestamp>-seed<seed>/` — Random Forest outputs
- `results/heterogeneous/<timestamp>-seed<seed>/` — LR/SVM/DT bases, soft voting, stacking
- `results/summary/` — aggregated summaries; per-seed CV ROC-AUC in `results/summary/seeds/`

Each directory also includes best hyperparameters:

- RF: `results/homogeneous/<timestamp>-seed<seed>/best_params.json`
- LR/SVM/DT: `results/heterogeneous/<timestamp>-seed<seed>/{lr,svc,dt}_best_params.json`

### What’s implemented

- Stratified 80/20 train/test with 5-fold CV for model selection (same folds across models per seed)
- Preprocessing inside Pipelines and folds (median imputation, StandardScaler for linear/kernel/knn, one-hot for categorical `gender` if present)
- Homogeneous RF with control-parameter tuning and OOB logging
- Heterogeneous base learners (LR, SVM(RBF), Decision Tree) with calibrated probabilities (Platt/sigmoid) and tuning
- Combination strategies: soft voting and stacking (meta=LogReg on OOF probabilities)
- Metrics: ROC-AUC (primary), PR-AUC, Macro-F1, Accuracy, Brier score; reliability curve and ECE
- Statistics: paired Wilcoxon test across folds; test-set bootstrap CIs
- Reproducibility: seeds, version pins, artifacts, and plots are saved

### Dataset

- Input CSV: `breastCancer.csv`
- Target column: `diagnosis` with labels {B, M} mapped to {0, 1}
- If `gender` exists, it is one-hot encoded; numeric columns are imputed (median) and scaled when required

### Notes

- Ensure your environment is activated before running any Python commands (uses python3/pip3 as requested).
- If you need to adjust grids, seeds, or metrics, edit `config.yaml`.

# ML_Assignment4_EnsembleLearning
