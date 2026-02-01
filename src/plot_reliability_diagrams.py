import os
import joblib
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ----------------------------
# Config (match your training)
# ----------------------------
DATA_PATH = "data/processed/pbmc68k_labeled.h5ad"
OUTPUT_DIR = "results"
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")

RANDOM_STATE = 42
TEST_SIZE = 0.2

LABEL_KEY = "cell_type"
N_BINS = 10  # 0–0.1, 0.1–0.2, ..., 0.9–1.0


def load_data_split_scaled():
    """Recreate the same split and scaling used for baselines."""
    adata = sc.read_h5ad(DATA_PATH)

    if LABEL_KEY not in adata.obs:
        raise ValueError(f"Expected adata.obs['{LABEL_KEY}'] to exist.")

    X = adata.X
    y_cat = adata.obs[LABEL_KEY].astype("category")
    y = y_cat.cat.codes.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    scaler_path = os.path.join(OUTPUT_DIR, "scaler.pkl")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        X_test = scaler.transform(X_test)
    else:
        scaler = StandardScaler(with_mean=False)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_test, y_test


def bin_stats(confidence, correct, n_bins=10):
    """
    Bin by confidence and compute:
    - mean confidence per bin
    - accuracy per bin
    - counts per bin
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(confidence, bins, right=True) - 1  # 0..n_bins-1

    bin_acc = np.full(n_bins, np.nan, dtype=float)
    bin_conf = np.full(n_bins, np.nan, dtype=float)
    bin_count = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        mask = bin_ids == b
        bin_count[b] = int(mask.sum())
        if bin_count[b] > 0:
            bin_acc[b] = correct[mask].mean()
            bin_conf[b] = confidence[mask].mean()

    return bins, bin_conf, bin_acc, bin_count


def expected_calibration_error(bin_conf, bin_acc, bin_count):
    """ECE = sum_k (n_k / n) * |acc_k - conf_k| over non-empty bins."""
    n = bin_count.sum()
    ece = 0.0
    for c, a, nk in zip(bin_conf, bin_acc, bin_count):
        if nk > 0 and np.isfinite(c) and np.isfinite(a):
            ece += (nk / n) * abs(a - c)
    return float(ece)


def plot_reliability(conf, correct, model_name, out_path, n_bins=10):
    bins, bin_conf, bin_acc, bin_count = bin_stats(conf, correct, n_bins=n_bins)
    ece = expected_calibration_error(bin_conf, bin_acc, bin_count)

    fig, ax = plt.subplots(figsize=(6, 5))

    # Perfect calibration reference
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)

    # Plot only non-empty bins
    mask = bin_count > 0
    ax.plot(bin_conf[mask], bin_acc[mask], marker="o", linewidth=2)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    ax.set_xlabel("Predicted confidence (binned)")
    ax.set_ylabel("Observed accuracy")
    ax.set_title(f"Reliability Diagram — {model_name}")

    # Clean look (no clutter)
    ax.grid(False)

    # Small text note with ECE (useful for you, still clean)
    ax.text(
        0.02, 0.02,
        f"ECE = {ece:.3f}",
        transform=ax.transAxes
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {out_path} (ECE={ece:.3f})")


def get_model_probs(model, X_test):
    """
    Return predicted probabilities:
    - scikit LR: predict_proba
    - XGBoost: predict_proba
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)
    raise ValueError("Model does not support predict_proba().")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    X_test, y_test = load_data_split_scaled()

    # Load available models
    models = []
    lr_path = os.path.join(OUTPUT_DIR, "logreg_model.pkl")
    xgb_path = os.path.join(OUTPUT_DIR, "xgb_model.pkl")
    sann_path = os.path.join(OUTPUT_DIR, "sann_model.pkl")  # optional, if you save it later

    if os.path.exists(lr_path):
        models.append(("Logistic Regression", joblib.load(lr_path)))
    if os.path.exists(xgb_path):
        models.append(("XGBoost", joblib.load(xgb_path)))
    if os.path.exists(sann_path):
        models.append(("SANN", joblib.load(sann_path)))  # will only work if it has predict_proba()

    if not models:
        raise FileNotFoundError("No models found in results/. Expected logreg_model.pkl and/or xgb_model.pkl.")

    for name, model in models:
        probs = get_model_probs(model, X_test)  # shape: (n_samples, n_classes)
        pred = probs.argmax(axis=1)
        conf = probs.max(axis=1)
        correct = (pred == y_test).astype(float)

        out_path = os.path.join(FIG_DIR, f"reliability_diagram_{name.lower().replace(' ', '_')}.png")
        plot_reliability(conf, correct, name, out_path, n_bins=N_BINS)


if __name__ == "__main__":
    main()