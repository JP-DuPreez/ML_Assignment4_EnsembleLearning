# %% [markdown]
# Data Quality Analysis (Notebook-like Script)
#
# This script uses `# %%` cells to behave like a Jupyter notebook in editors that support cell execution.
# It analyzes:
# 1) Missing values
# 2) Class imbalance
# 3) Outliers (IQR and Z-score methods)
# 4) Other data quality issues (duplicates, constant/near-constant features, mixed types, cardinality, correlations)


# %% Imports and setup
import os
import json
import warnings
from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import yaml

sns.set_theme(style="whitegrid", context="notebook")
warnings.filterwarnings("ignore", category=FutureWarning)

proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
results_dir = os.path.join(proj_root, "results", "data_quality")
os.makedirs(results_dir, exist_ok=True)


# %% Load config and resolve dataset path
cfg_path = os.path.join(proj_root, "config.yaml")
if os.path.exists(cfg_path):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    data_cfg = cfg.get("data", {})
    csv_path_cfg = data_cfg.get("path", "Data/breastCancer.csv")
    target_col = data_cfg.get("target", "diagnosis")
    label_map = data_cfg.get("label_map", {"B": 0, "M": 1})
else:
    csv_path_cfg = "Data/breastCancer.csv"
    target_col = "diagnosis"
    label_map = {"B": 0, "M": 1}

candidate_paths: List[str] = []
if os.path.isabs(csv_path_cfg):
    candidate_paths.append(csv_path_cfg)
else:
    candidate_paths.append(os.path.join(proj_root, csv_path_cfg))
    candidate_paths.append(os.path.join(proj_root, "Data", os.path.basename(csv_path_cfg)))
    candidate_paths.append(os.path.join(proj_root, "data", os.path.basename(csv_path_cfg)))

csv_path = next((p for p in candidate_paths if os.path.exists(p)), candidate_paths[0])
print(f"Using dataset: {csv_path}")
print(f"Target column: {target_col}")


# %% Load data (preserve raw for duplicate checks)
df_raw = pd.read_csv(csv_path, sep=None, engine="python", skipinitialspace=True)
df_raw = df_raw.replace(to_replace=r"^\s*\?\s*$", value=np.nan, regex=True)
df_raw.columns = [str(c).strip().strip('"').strip("'") for c in df_raw.columns]

if target_col not in df_raw.columns:
    lowered = {c.lower(): c for c in df_raw.columns}
    if target_col.lower() in lowered:
        target_col = lowered[target_col.lower()]
    else:
        raise SystemExit(f"Target column '{target_col}' not found. Available: {list(df_raw.columns)}")

X_raw = df_raw.drop(columns=[target_col])
y_raw = df_raw[target_col]

# Map labels to numeric if possible (keeps y_raw for display)
if y_raw.dtype.kind in {"i", "u", "f"}:
    y = y_raw.astype(float).round().astype(int)
else:
    y = y_raw.map(label_map)
if getattr(y, "isna")().any():
    unmapped = sorted(y_raw[y.isna()].astype(str).unique().tolist())
    print(f"Warning: unmapped labels in target; falling back to raw labels. Unmapped: {unmapped}")
    y = y_raw.astype(str)

print(f"Shape (rows, cols including target): {df_raw.shape}")


# %% Basic dtype and uniqueness overview
dtypes_summary = pd.DataFrame({
    "column": df_raw.columns,
    "dtype": df_raw.dtypes.astype(str).values,
    "n_unique": [df_raw[c].nunique(dropna=True) for c in df_raw.columns],
})
dtypes_summary.to_csv(os.path.join(results_dir, "dtypes_summary.csv"), index=False)
display_cols = ["column", "dtype", "n_unique"]
print(dtypes_summary.sort_values("column")[display_cols].to_string(index=False))


# %% Missing values analysis
n_rows = len(df_raw)
missing_counts = df_raw.isna().sum()
missing_pct = (missing_counts / max(1, n_rows)) * 100
missing_df = pd.DataFrame({
    "column": df_raw.columns,
    "dtype": df_raw.dtypes.astype(str).values,
    "n_unique": [df_raw[c].nunique(dropna=True) for c in df_raw.columns],
    "missing_count": missing_counts.values,
    "missing_pct": missing_pct.values,
}).sort_values("missing_pct", ascending=False)
missing_df.to_csv(os.path.join(results_dir, "missing_values_summary.csv"), index=False)

plt.figure(figsize=(10, max(4, 0.25 * len(missing_df))))
sns.barplot(x="missing_pct", y="column", data=missing_df, color="#4c78a8")
plt.xlabel("Missing (%)")
plt.ylabel("Column")
plt.title("Missing Values by Column")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "missing_values_bar.png"), dpi=200)
plt.close()

print("Saved missing values summary and bar plot.")


# %% Class imbalance analysis
use_y = y
class_counts = use_y.value_counts(dropna=False).sort_index()
class_pct = (class_counts / len(use_y)) * 100
class_df = pd.DataFrame({
    "class": class_counts.index.astype(str),
    "count": class_counts.values,
    "pct": class_pct.values,
})
class_df.to_csv(os.path.join(results_dir, "class_imbalance_summary.csv"), index=False)

plt.figure(figsize=(7, 4))
sns.barplot(x="class", y="count", data=class_df, color="#72b7b2")
plt.title("Class Counts")
plt.xlabel("Class")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "class_counts.png"), dpi=200)
plt.close()

imbalance_ratio = None
if class_counts.shape[0] == 2:
    minority = int(class_counts.min())
    majority = int(class_counts.max())
    imbalance_ratio = minority / majority if majority > 0 else None
    print(f"Imbalance ratio (minority/majority): {imbalance_ratio:.4f}")
else:
    print("Multiclass or missing classes detected; imbalance ratio (binary) not computed.")


# %% Prepare numeric view for outlier analysis
df_coerced = X_raw.copy()
for c in df_coerced.columns:
    if df_coerced[c].dtype == object:
        df_coerced[c] = pd.to_numeric(df_coerced[c], errors="coerce")

numeric_cols = [c for c in df_coerced.columns if pd.api.types.is_numeric_dtype(df_coerced[c])]
print(f"Numeric columns inferred: {len(numeric_cols)}")


# %% Outliers (IQR method)
iqr_rows = []
for c in numeric_cols:
    s = df_coerced[c].dropna()
    if s.shape[0] == 0:
        iqr_rows.append({"column": c, "n": 0, "outliers": 0, "outliers_pct": 0.0, "q1": np.nan, "q3": np.nan})
        continue
    q1 = float(np.percentile(s, 25))
    q3 = float(np.percentile(s, 75))
    iqr = q3 - q1
    if iqr == 0:
        outliers_mask = pd.Series(False, index=s.index)
    else:
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers_mask = (s < lower) | (s > upper)
    count = int(outliers_mask.sum())
    pct = (count / s.shape[0]) * 100
    iqr_rows.append({"column": c, "n": int(s.shape[0]), "outliers": count, "outliers_pct": pct, "q1": q1, "q3": q3})

iqr_df = pd.DataFrame(iqr_rows).sort_values("outliers_pct", ascending=False)
iqr_df.to_csv(os.path.join(results_dir, "outliers_iqr_summary.csv"), index=False)

top_iqr = iqr_df.head(20)
plt.figure(figsize=(10, max(4, 0.25 * top_iqr.shape[0])))
sns.barplot(x="outliers_pct", y="column", data=top_iqr, color="#e45756")
plt.xlabel("Outliers (%)")
plt.ylabel("Column")
plt.title("Top Outlier Columns (IQR)")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "outliers_iqr_bar.png"), dpi=200)
plt.close()

print("Saved IQR outlier summary and bar plot.")


# %% Outliers (Z-score |z| > 3)
z_rows = []
for c in numeric_cols:
    s = df_coerced[c].astype(float)
    n = int(s.dropna().shape[0])
    if n == 0:
        z_rows.append({"column": c, "n": 0, "z_outliers": 0, "z_outliers_pct": 0.0})
        continue
    mean = float(s.mean(skipna=True))
    std = float(s.std(ddof=0, skipna=True))
    if std == 0.0:
        count = 0
        pct = 0.0
    else:
        z = (s - mean) / std
        mask = z.abs() > 3.0
        count = int(mask.sum(skipna=True))
        pct = (count / n) * 100
    z_rows.append({"column": c, "n": n, "z_outliers": count, "z_outliers_pct": pct})

z_df = pd.DataFrame(z_rows).sort_values("z_outliers_pct", ascending=False)
z_df.to_csv(os.path.join(results_dir, "outliers_zscore_summary.csv"), index=False)

top_z = z_df.head(20)
plt.figure(figsize=(10, max(4, 0.25 * top_z.shape[0])))
sns.barplot(x="z_outliers_pct", y="column", data=top_z, color="#f58518")
plt.xlabel("Z-score Outliers (%)")
plt.ylabel("Column")
plt.title("Top Outlier Columns (|z| > 3)")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "outliers_zscore_bar.png"), dpi=200)
plt.close()

print("Saved Z-score outlier summary and bar plot.")


# %% Other data quality checks
summary = {}

# Duplicates
dup_count = int(df_raw.duplicated().sum())
summary["duplicate_rows"] = dup_count
if dup_count > 0:
    sample_dups = df_raw[df_raw.duplicated(keep=False)]
    sample_path = os.path.join(results_dir, "duplicate_rows_sample.csv")
    sample_dups.head(200).to_csv(sample_path, index=False)

# Identifier uniqueness if present
for id_col in ["id", "ID", "Id"]:
    if id_col in df_raw.columns:
        nunique = int(df_raw[id_col].nunique(dropna=True))
        summary[f"{id_col}_unique"] = nunique
        summary[f"{id_col}_is_unique_key"] = (nunique == len(df_raw))

# Constant and near-constant
const_cols = [c for c in X_raw.columns if df_raw[c].nunique(dropna=False) <= 1]
near_const_cols = []
for c in X_raw.columns:
    vc = df_raw[c].value_counts(dropna=False, normalize=True)
    if vc.shape[0] > 0 and float(vc.iloc[0]) >= 0.95 and df_raw[c].nunique(dropna=False) > 1:
        near_const_cols.append(c)

pd.DataFrame({"constant_col": const_cols}).to_csv(os.path.join(results_dir, "constant_columns.csv"), index=False)
pd.DataFrame({"near_constant_col": near_const_cols}).to_csv(os.path.join(results_dir, "near_constant_columns.csv"), index=False)
summary["n_constant_cols"] = len(const_cols)
summary["n_near_constant_cols"] = len(near_const_cols)

# Categorical cardinality and mixed types
cat_like = [c for c in X_raw.columns if df_raw[c].dtype == object]
card_rows = []
mixed_rows = []
for c in cat_like:
    nuniq = int(df_raw[c].nunique(dropna=True))
    ratio = nuniq / max(1, len(df_raw))
    card_rows.append({"column": c, "n_unique": nuniq, "unique_ratio": ratio})

    coerced = pd.to_numeric(df_raw[c], errors="coerce")
    frac_numeric = float(coerced.notna().mean())
    if 0.05 < frac_numeric < 0.95:
        mixed_rows.append({"column": c, "fraction_numeric_like": frac_numeric})

pd.DataFrame(card_rows).sort_values("unique_ratio", ascending=False).to_csv(
    os.path.join(results_dir, "categorical_cardinality.csv"), index=False
)
pd.DataFrame(mixed_rows).to_csv(os.path.join(results_dir, "mixed_type_suspected.csv"), index=False)
summary["n_high_cardinality_cats"] = int(sum(r["unique_ratio"] > 0.5 or r["n_unique"] > 50 for r in card_rows))
summary["n_mixed_type_suspected"] = len(mixed_rows)

# Correlations (numeric only)
if len(numeric_cols) >= 2:
    corr = df_coerced[numeric_cols].corr(method="pearson")
    corr.to_csv(os.path.join(results_dir, "correlation_matrix.csv"))
    pairs = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = float(corr.iloc[i, j])
            if abs(r) >= 0.95 and not np.isnan(r):
                pairs.append({"col1": cols[i], "col2": cols[j], "pearson_r": r})
    pd.DataFrame(pairs).sort_values("pearson_r", ascending=False, key=lambda s: s.abs()).to_csv(
        os.path.join(results_dir, "high_correlation_pairs.csv"), index=False
    )
    summary["n_high_corr_pairs_abs_ge_0_95"] = len(pairs)
else:
    summary["n_high_corr_pairs_abs_ge_0_95"] = 0


# %% Save run summary
summary.update({
    "rows": int(df_raw.shape[0]),
    "columns": int(df_raw.shape[1]),
    "target": str(target_col),
    "class_counts": {str(k): int(v) for k, v in use_y.value_counts(dropna=False).to_dict().items()},
    "imbalance_ratio_minority_majority": float(imbalance_ratio) if imbalance_ratio is not None else None,
    "missing_total_cells": int(df_raw.isna().sum().sum()),
})

with open(os.path.join(results_dir, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("Data quality analysis complete. Artifacts saved to:")
print(results_dir)



# %%
