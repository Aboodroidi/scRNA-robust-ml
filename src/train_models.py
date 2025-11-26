# src/train_models.py
import os

# ----------------------------
# Single-thread setup (Mac-friendly)
# ----------------------------
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

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
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
RANDOM_STATE = 42
TEST_SIZE = 0.2

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_data(data_path: str):
    """Load h5ad file and return (X, y, label_names)."""
    adata = sc.read_h5ad(data_path)
    X = adata.X
    cell_type_cat = adata.obs["cell_type"].astype("category")
    y = cell_type_cat.cat.codes
    label_names = list(cell_type_cat.cat.categories)
    return X, y, label_names


def split_and_scale(X, y, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE):
    """Stratified train-test split + scaling (safe for sparse matrices)."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    scaler = StandardScaler(with_mean=False)  # with_mean=False required for sparse
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_logistic_regression(X_train, y_train, X_test, y_test, output_dir: str):
    """Train Logistic Regression baseline and save model."""
    lr_model = LogisticRegression(
        max_iter=3000,      # increased from 1000 to reduce convergence warnings
        solver="saga",
        n_jobs=1            # single-threaded for macOS stability
    )
    lr_model.fit(X_train, y_train)

    y_pred = lr_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    joblib.dump(lr_model, os.path.join(output_dir, "logreg_model.pkl"))

    return lr_model, acc, macro_f1, y_pred


def train_xgboost(X_train, y_train, X_test, y_test, num_classes: int, output_dir: str):
    """Train XGBoost baseline and save model."""
    xgb_model = XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        max_depth=6,
        learning_rate=0.1,
        n_estimators=200,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=1,           # single-threaded for macOS
        tree_method="hist", # CPU-friendly & stable
        # predictor parameter removed (was ignored and caused a warning)
    )
    xgb_model.fit(X_train, y_train)

    y_pred = xgb_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    joblib.dump(xgb_model, os.path.join(output_dir, "xgb_model.pkl"))

    return xgb_model, acc, macro_f1, y_pred


def save_metrics_and_reports(
    acc_lr, f1_lr,
    acc_xgb, f1_xgb,
    y_test, y_pred_xgb,
    label_names,
    output_dir: str
):
    """Save CSV with global metrics + text classification report for XGBoost."""
    metrics = pd.DataFrame({
        "Model": ["Logistic Regression", "XGBoost"],
        "Accuracy": [acc_lr, acc_xgb],
        "Macro-F1": [f1_lr, f1_xgb],
    })
    metrics_path = os.path.join(output_dir, "baseline_metrics.csv")
    metrics.to_csv(metrics_path, index=False)

    # Detailed per-class report for XGBoost
    report = classification_report(
        y_test, y_pred_xgb, target_names=label_names
    )
    report_path = os.path.join(output_dir, "xgb_classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    print("\nClassification report for XGBoost:\n")
    print(report)
    print(f"\nSaved baseline metrics to {metrics_path}")
    print(f"Saved XGBoost classification report to {report_path}")


def main():
    steps = [
        "Load data",
        "Split & scale",
        "Train Logistic Regression",
        "Train XGBoost",
        "Save metrics & reports",
    ]
    bar_fmt = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{percentage:3.0f}%] - {elapsed}<{remaining}"
    pbar = tqdm(total=len(steps), bar_format=bar_fmt, desc="Baseline pipeline")

    # 1) Load data
    pbar.set_description(f"1/5 {steps[0]}")
    X, y, label_names = load_data(DATA_PATH)
    pbar.update(1)

    # 2) Split & scale
    pbar.set_description(f"2/5 {steps[1]}")
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)
    # Save scaler for reuse in evaluation or deployment
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))
    pbar.update(1)

    # 3) Train Logistic Regression
    pbar.set_description(f"3/5 {steps[2]}")
    lr_model, acc_lr, f1_lr, y_pred_lr = train_logistic_regression(
        X_train, y_train, X_test, y_test, OUTPUT_DIR
    )
    pbar.set_postfix_str(f"LR acc {acc_lr:.3f}, F1 {f1_lr:.3f}")
    pbar.update(1)

    # 4) Train XGBoost
    pbar.set_description(f"4/5 {steps[3]}")
    xgb_model, acc_xgb, f1_xgb, y_pred_xgb = train_xgboost(
        X_train, y_train, X_test, y_test, num_classes=len(label_names), output_dir=OUTPUT_DIR
    )
    pbar.set_postfix_str(f"XGB acc {acc_xgb:.3f}, F1 {f1_xgb:.3f}")
    pbar.update(1)

    # 5) Save metrics & reports
    pbar.set_description(f"5/5 {steps[4]}")
    save_metrics_and_reports(
        acc_lr, f1_lr,
        acc_xgb, f1_xgb,
        y_test, y_pred_xgb,
        label_names,
        OUTPUT_DIR,
    )
    pbar.update(1)
    pbar.close()


if __name__ == "__main__":
    main()