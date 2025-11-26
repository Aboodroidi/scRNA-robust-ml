# src/evaluate_models.py
import os

import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
import joblib

# ----------------------------
# Config (must match train_models.py)
# ----------------------------
DATA_PATH = "data/processed/pbmc68k_labeled.h5ad"
OUTPUT_DIR = "results"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
RANDOM_STATE = 42
TEST_SIZE = 0.2

os.makedirs(FIGURES_DIR, exist_ok=True)


def load_data_and_split():
    """Load data and reproduce the same train-test split + scaling."""
    adata = sc.read_h5ad(DATA_PATH)
    X = adata.X
    cell_type_cat = adata.obs["cell_type"].astype("category")
    y = cell_type_cat.cat.codes
    label_names = list(cell_type_cat.cat.categories)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # Either load the saved scaler or re-fit; both are equivalent if config matches
    scaler_path = os.path.join(OUTPUT_DIR, "scaler.pkl")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        scaler = StandardScaler(with_mean=False)
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, label_names


def load_models():
    lr_model = joblib.load(os.path.join(OUTPUT_DIR, "logreg_model.pkl"))
    xgb_model = joblib.load(os.path.join(OUTPUT_DIR, "xgb_model.pkl"))
    return lr_model, xgb_model


def plot_global_metrics_bar():
    """Read baseline_metrics.csv and create a bar chart for Accuracy & Macro-F1."""
    metrics_path = os.path.join(OUTPUT_DIR, "baseline_metrics.csv")
    df = pd.read_csv(metrics_path)

    x = np.arange(len(df["Model"]))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width / 2, df["Accuracy"], width, label="Accuracy")
    ax.bar(x + width / 2, df["Macro-F1"], width, label="Macro-F1")

    ax.set_xticks(x)
    ax.set_xticklabels(df["Model"], rotation=0)
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Baseline Model Performance (Accuracy & Macro-F1)")
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "baseline_metrics_bar.png"), dpi=300)
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, label_names, model_name: str):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(label_names)))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=label_names,
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(include_values=False, cmap="Blues", ax=ax, xticks_rotation=90)
    ax.set_title(f"Confusion Matrix – {model_name}")
    fig.tight_layout()
    out_path = os.path.join(
        FIGURES_DIR,
        f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    )
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_per_class_f1(y_true, y_pred, label_names, model_name: str):
    """Plot per-class F1 scores as a bar chart."""
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )

    class_f1 = [report_dict[lbl]["f1-score"] for lbl in label_names]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(np.arange(len(label_names)), class_f1)
    ax.set_xticks(np.arange(len(label_names)))
    ax.set_xticklabels(label_names, rotation=90)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("F1-score")
    ax.set_title(f"Per-class F1 – {model_name}")
    fig.tight_layout()
    out_path = os.path.join(
        FIGURES_DIR,
        f"per_class_f1_{model_name.lower().replace(' ', '_')}.png"
    )
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    # 1) Data & split
    X_train, X_test, y_train, y_test, label_names = load_data_and_split()

    # 2) Load trained models
    lr_model, xgb_model = load_models()

    # 3) Predictions
    y_pred_lr = lr_model.predict(X_test)
    y_pred_xgb = xgb_model.predict(X_test)

    # 4) Global metrics bar chart (Accuracy & Macro-F1)
    plot_global_metrics_bar()

    # 5) Confusion matrices
    plot_confusion_matrix(y_test, y_pred_lr, label_names, model_name="Logistic Regression")
    plot_confusion_matrix(y_test, y_pred_xgb, label_names, model_name="XGBoost")

    # 6) Per-class F1 bar charts
    plot_per_class_f1(y_test, y_pred_lr, label_names, model_name="Logistic Regression")
    plot_per_class_f1(y_test, y_pred_xgb, label_names, model_name="XGBoost")

    print(f"Saved plots to {FIGURES_DIR}")


if __name__ == "__main__":
    main()