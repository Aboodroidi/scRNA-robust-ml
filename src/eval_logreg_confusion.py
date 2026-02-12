# src/eval_logreg_confusion.py
import os

# ----------------------------
# Mac-friendly single-thread setup
# ----------------------------
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MPLBACKEND"] = "Agg"  # safe non-GUI backend for saving figures

import joblib
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
)

# ----------------------------
# Config
# ----------------------------
DATA_PATH = "data/processed/pbmc68k_labeled.h5ad"
LABEL_KEY = "cell_type"

# Feature representation (must match what you want to evaluate)
# If you want LR on PCA (consistent with SANN): USE_PCA=True
USE_PCA = True
PCA_KEY = "X_pca"
PCA_DIM = 50  # set None to use all PCA dims

RANDOM_STATE = 42
TEST_SIZE = 0.2

OUTPUT_DIR = "results"
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save separate artifacts for the representation to avoid mismatches
REP_TAG = f"pca{PCA_DIM}" if USE_PCA else "x"
LR_MODEL_PATH = os.path.join(OUTPUT_DIR, f"logreg_model_{REP_TAG}.pkl")
SCALER_PATH = os.path.join(OUTPUT_DIR, f"scaler_{REP_TAG}.pkl")

FIG_PATH = os.path.join(FIG_DIR, "confusion_matrix_logistic_regression.png")
METRICS_PATH = os.path.join(OUTPUT_DIR, "baseline_metrics.csv")


def shorten_labels(labels, max_len=16):
    """Shorten long class names for axis readability."""
    out = []
    for s in labels:
        s = str(s)
        out.append(s if len(s) <= max_len else s[: max_len - 1] + "…")
    return out


def load_features_and_labels():
    adata = sc.read_h5ad(DATA_PATH)

    if LABEL_KEY not in adata.obs:
        raise ValueError(f"Expected adata.obs['{LABEL_KEY}'] to exist.")

    # Labels with fixed order
    y_cat = adata.obs[LABEL_KEY].astype("category")
    y = y_cat.cat.codes.to_numpy()
    class_names = list(y_cat.cat.categories)

    # Features
    if USE_PCA:
        if PCA_KEY not in adata.obsm:
            raise ValueError(
                f"USE_PCA=True but adata.obsm['{PCA_KEY}'] not found. "
                f"Either set USE_PCA=False or ensure PCA exists."
            )
        X = adata.obsm[PCA_KEY]
        if PCA_DIM is not None:
            if X.shape[1] < PCA_DIM:
                raise ValueError(f"{PCA_KEY} has {X.shape[1]} dims; cannot take PCA_DIM={PCA_DIM}.")
            X = X[:, :PCA_DIM]
        X = np.asarray(X, dtype=np.float32)
    else:
        # adata.X may be sparse
        X = adata.X

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


def train_or_load_lr(X_train, y_train, expected_n_features: int):
    """
    Load LR + scaler if they exist AND match the expected feature dimension.
    Otherwise retrain and overwrite.
    """
    loaded = False

    if os.path.exists(LR_MODEL_PATH) and os.path.exists(SCALER_PATH):
        try:
            lr = joblib.load(LR_MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)

            # Check scaler feature compatibility
            scaler_n = getattr(scaler, "n_features_in_", None)
            if scaler_n is None or scaler_n != expected_n_features:
                raise ValueError(
                    f"Saved scaler expects {scaler_n} features, but current data has {expected_n_features}."
                )

            loaded = True
            return lr, scaler, loaded
        except Exception as e:
            print(f"[Info] Existing model/scaler incompatible or failed to load: {e}")
            print("[Info] Retraining Logistic Regression for this representation...")

    # Train fresh
    scaler = StandardScaler(with_mean=False)  # safe for sparse too
    X_train_scaled = scaler.fit_transform(X_train)

    lr = LogisticRegression(
        max_iter=3000,
        solver="saga",
        multi_class="multinomial",
        n_jobs=1,
        random_state=RANDOM_STATE,
    )
    lr.fit(X_train_scaled, y_train)

    joblib.dump(lr, LR_MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    return lr, scaler, loaded


def plot_confusion_matrix(cm_norm, class_names, out_path):
    fig, ax = plt.subplots(figsize=(9, 8))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_norm,
        display_labels=shorten_labels(class_names),
    )
    disp.plot(
        include_values=False,  # cleaner for many classes
        cmap="Blues",
        ax=ax,
        colorbar=True,
    )

    ax.set_title("Confusion Matrix (Row-normalised) — Logistic Regression")
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
        df = df[df["Model"] != model_name]  # replace if exists
        df = pd.concat([df, row], ignore_index=True)
    else:
        df = row

    df.to_csv(path, index=False)


def main():
    # 1) Load
    X, y, class_names = load_features_and_labels()

    # Sanity prints
    print(f"[Sanity] X shape: {getattr(X, 'shape', None)}")
    print(f"[Sanity] y length: {len(y)}")
    print(f"[Sanity] #classes: {len(class_names)}")
    print(f"[Sanity] class names: {class_names}")

    # 2) Split
    X_train, X_test, y_train, y_test = get_split(X, y)
    print(f"[Sanity] Train samples: {len(y_train)} | Test samples: {len(y_test)}")

    # 3) Train or load LR + scaler (representation-safe)
    expected_n_features = X_train.shape[1]
    lr, scaler, loaded = train_or_load_lr(X_train, y_train, expected_n_features)
    print(f"[Sanity] Logistic Regression model loaded from disk: {loaded}")
    print(f"[Sanity] Using artifacts: {LR_MODEL_PATH} and {SCALER_PATH}")

    # 4) Predict
    X_test_scaled = scaler.transform(X_test)
    y_pred = lr.predict(X_test_scaled)

    # 5) Metrics
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print(f"[Sanity] Accuracy (LR): {acc:.4f}")
    print(f"[Sanity] Macro-F1 (LR): {macro_f1:.4f}")

    # 6) Confusion matrices (raw + row-normalised)
    labels = np.arange(len(class_names))
    cm_raw = confusion_matrix(y_test, y_pred, labels=labels)

    cm_norm = cm_raw.astype(np.float64)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_norm, row_sums, out=np.zeros_like(cm_norm), where=row_sums != 0)

    # 7) Plot + save
    plot_confusion_matrix(cm_norm, class_names, FIG_PATH)
    print(f"[Sanity] Saved figure: {FIG_PATH}")

    # 8) Save/update metrics file
    update_metrics_csv("Logistic Regression", acc, macro_f1, METRICS_PATH)
    print(f"[Sanity] Updated metrics file: {METRICS_PATH}")


if __name__ == "__main__":
    main()