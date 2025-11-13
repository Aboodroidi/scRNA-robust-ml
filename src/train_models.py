# src/train_models.py
import os
import scanpy as sc
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from xgboost import XGBClassifier
import joblib
from tqdm import tqdm

# ----------------------------
# Config
# ----------------------------
DATA_PATH = "data/processed/pbmc68k_labeled.h5ad"
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# progress bar setup
bar_fmt = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{percentage:3.0f}%] - {elapsed}<{remaining}"
steps = [
    "Load data",
    "Split & scale",
    "Train Logistic Regression",
    "Train XGBoost",
    "Save metrics & reports"
]
pbar = tqdm(total=len(steps), bar_format=bar_fmt, desc="Baseline pipeline")

# 1) Load data
pbar.set_description(f"1/5 {steps[0]}")
adata = sc.read_h5ad(DATA_PATH)
pbar.update(1)

# Extract features and labels
X = adata.X
y = adata.obs["cell_type"].astype("category").cat.codes
label_names = list(adata.obs["cell_type"].astype("category").cat.categories)

# 2) Split & scale
pbar.set_description(f"2/5 {steps[1]}")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
scaler = StandardScaler(with_mean=False)  # safe for sparse
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
pbar.update(1)

# 3) Train Logistic Regression
pbar.set_description(f"3/5 {steps[2]}")
lr_model = LogisticRegression(max_iter=1000, solver="saga", n_jobs=-1)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
acc_lr = accuracy_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr, average="macro")
joblib.dump(lr_model, os.path.join(OUTPUT_DIR, "logreg_model.pkl"))
pbar.set_postfix_str(f"LR acc {acc_lr:.3f}, F1 {f1_lr:.3f}")
pbar.update(1)

# 4) Train XGBoost
pbar.set_description(f"4/5 {steps[3]}")
xgb_model = XGBClassifier(
    objective="multi:softprob",
    num_class=len(label_names),
    max_depth=6,
    learning_rate=0.1,
    n_estimators=200,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
acc_xgb = accuracy_score(y_test, y_pred_xgb)
f1_xgb = f1_score(y_test, y_pred_xgb, average="macro")
joblib.dump(xgb_model, os.path.join(OUTPUT_DIR, "xgb_model.pkl"))
pbar.set_postfix_str(f"XGB acc {acc_xgb:.3f}, F1 {f1_xgb:.3f}")
pbar.update(1)

# 5) Save metrics and report
pbar.set_description(f"5/5 {steps[4]}")
metrics = pd.DataFrame({
    "Model": ["Logistic Regression", "XGBoost"],
    "Accuracy": [acc_lr, acc_xgb],
    "Macro-F1": [f1_lr, f1_xgb]
})
metrics.to_csv(os.path.join(OUTPUT_DIR, "baseline_metrics.csv"), index=False)
# print a brief report for XGB
print("\nClassification report for XGBoost:\n")
print(classification_report(y_test, y_pred_xgb, target_names=label_names))
pbar.update(1)
pbar.close()

print("\nSaved baseline metrics to results/baseline_metrics.csv")