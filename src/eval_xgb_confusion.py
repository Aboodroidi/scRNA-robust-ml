# src/eval_xgb_confusion.py
import os

# ----------------------------
# Mac-friendly single-thread setup
# ----------------------------
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
)
from xgboost import XGBClassifier

# ----------------------------
# Config
# ----------------------------
DATA_PATH = "data/processed/pbmc68k_labeled.h5ad"
LABEL_KEY = "cell_type"

# Feature representation (must match how you trained XGBoost)
# If your XGB was trained on HVGs (adata.X after HVG selection), set USE_PCA=False
# If you trained XGB on PCA-50, set USE_PCA=True
USE_PCA = True
PCA_KEY = "X_pca"
PCA_DIM = 50  # set None to use all dims

RANDOM_STATE = 42
TEST_SIZE = 0.2

OUTPUT_DIR = "results"
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save/load XGBoost safely (JSON)
REP_TAG = f"pca{PCA_DIM}" if USE_PCA else "x"
XGB_MODEL_JSON = os.path.join(OUTPUT_DIR, f"xgb_model_{REP_TAG}.json")

FIG_PATH = os.path.join(FIG_DIR, "confusion_matrix_xgboost.png")
METRICS_PATH = os.path.join(OUTPUT_DIR, "baseline_metrics.csv")


def shorten_labels(labels, max_len=16):
    out = []
    for s in labels:
        s = str(s)
        out.append(s if len(s) <= max_len else s[: max_len - 1] + "…")
    return out


def load_features_and_labels():
    adata = sc.read_h5ad(DATA_PATH)

    if LABEL_KEY not in adata.obs:
        raise ValueError(f"Expected adata.obs['{LABEL_KEY}'] to exist.")

    y_cat = adata.obs[LABEL_KEY].astype("category")
    y = y_cat.cat.codes.to_numpy()
    class_names = list(y_cat.cat.categories)

    if USE_PCA:
        if PCA_KEY not in adata.obsm:
            raise ValueError(f"USE_PCA=True but adata.obsm['{PCA_KEY}'] not found.")
        X = adata.obsm[PCA_KEY]
        if PCA_DIM is not None:
            if X.shape[1] < PCA_DIM:
                raise ValueError(f"{PCA_KEY} has {X.shape[1]} dims; cannot take PCA_DIM={PCA_DIM}.")
            X = X[:, :PCA_DIM]
        X = np.asarray(X, dtype=np.float32)
    else:
        X = adata.X  # can be sparse

    return X, y, class_names


def get_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    return X_train, X_test, y_train, y_test


def train_or_load_xgb(X_train, y_train, num_classes: int):
    """
    Load XGB from JSON if available; otherwise train and save.
    This avoids pickle/joblib segfault issues on macOS.
    """
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        max_depth=4,
        learning_rate=0.1,
        n_estimators=200,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=1,
        tree_method="hist",
        eval_metric="mlogloss",
    )

    if os.path.exists(XGB_MODEL_JSON):
        model.load_model(XGB_MODEL_JSON)
        return model, True

    model.fit(X_train, y_train)
    model.save_model(XGB_MODEL_JSON)
    return model, False


def plot_confusion_matrix(cm_norm, class_names, out_path):
    fig, ax = plt.subplots(figsize=(9, 8))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_norm,
        display_labels=shorten_labels(class_names),
    )
    disp.plot(
        include_values=False,
        cmap="Blues",
        ax=ax,
        colorbar=True,
    )

    ax.set_title("Confusion Matrix (Row-normalised) — XGBoost")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def update_metrics_csv(model_name, acc, macro_f1, path):
    row = pd.DataFrame([{
        "Model": model_name,
        "Accuracy": round(float(acc), 4),
        "Macro-F1": round(float(macro_f1), 4),
    }])

    if os.path.exists(path):
        df = pd.read_csv(path)
        df = df[df["Model"] != model_name]
        df = pd.concat([df, row], ignore_index=True)
    else:
        df = row

    df.to_csv(path, index=False)


def main():
    X, y, class_names = load_features_and_labels()

    print(f"[Sanity] X shape: {getattr(X, 'shape', None)}")
    print(f"[Sanity] y length: {len(y)}")
    print(f"[Sanity] #classes: {len(class_names)}")
    print(f"[Sanity] class names: {class_names}")

    X_train, X_test, y_train, y_test = get_split(X, y)
    print(f"[Sanity] Train samples: {len(y_train)} | Test samples: {len(y_test)}")

    model, loaded = train_or_load_xgb(X_train, y_train, num_classes=len(class_names))
    print(f"[Sanity] XGBoost model loaded from disk: {loaded}")
    print(f"[Sanity] Using artifact: {XGB_MODEL_JSON}")

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print(f"[Sanity] Accuracy (XGB): {acc:.4f}")
    print(f"[Sanity] Macro-F1 (XGB): {macro_f1:.4f}")

    labels = np.arange(len(class_names))
    cm_raw = confusion_matrix(y_test, y_pred, labels=labels)

    cm_norm = cm_raw.astype(np.float64)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_norm, row_sums, out=np.zeros_like(cm_norm), where=row_sums != 0)

    plot_confusion_matrix(cm_norm, class_names, FIG_PATH)
    print(f"[Sanity] Saved figure: {FIG_PATH}")

    update_metrics_csv("XGBoost", acc, macro_f1, METRICS_PATH)
    print(f"[Sanity] Updated metrics file: {METRICS_PATH}")


if __name__ == "__main__":
    main()