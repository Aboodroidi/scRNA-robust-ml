import os
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc

from sklearn.metrics import confusion_matrix

# ----------------------------
# Config
# ----------------------------
ADATA_PATH = "data/processed/pbmc68k_labeled.h5ad"
LABEL_KEY = "cell_type"
FIG_OUT = "results/figures/confusion_matrices_all_models.png"

# Load saved test arrays (from results/full_train/)
LR_TRUE = "results/full_train/lr_test_true.npy"
LR_PRED = "results/full_train/lr_test_pred.npy"

XGB_TRUE = "results/full_train/xgb_test_true.npy"
XGB_PRED = "results/full_train/xgb_test_pred.npy"

SANN_TRUE = "results/full_train/sann_test_true.npy"
SANN_PRED = "results/full_train/sann_test_pred.npy"

# Style tweaks
TITLE_FONTSIZE = 14
LABEL_FONTSIZE = 12
TICK_FONTSIZE = 10

# For 14 classes, this is usually cleanest:
SHOW_VALUES = False  # keep False for dissertation cleanliness
CMAP = "Blues"       # keep your same style


def shorten_labels(labels, max_len=16):
    out = []
    for s in labels:
        s = str(s)
        out.append(s if len(s) <= max_len else s[: max_len - 1] + "…")
    return out


def row_normalise(cm):
    cm = cm.astype(np.float64)
    row_sums = cm.sum(axis=1, keepdims=True)
    return np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)


def plot_cm(ax, cm_norm, class_names, title):
    im = ax.imshow(cm_norm, vmin=0.0, vmax=1.0, cmap=CMAP)

    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Predicted", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("True", fontsize=LABEL_FONTSIZE)

    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    disp_labels = shorten_labels(class_names)
    ax.set_xticklabels(disp_labels, rotation=45, ha="right", fontsize=TICK_FONTSIZE)
    ax.set_yticklabels(disp_labels, fontsize=TICK_FONTSIZE)

    ax.grid(False)

    if SHOW_VALUES:
        # optional: annotate values (not recommended for 14x14)
        for i in range(cm_norm.shape[0]):
            for j in range(cm_norm.shape[1]):
                ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", fontsize=7)

    return im


def main():
    os.makedirs(os.path.dirname(FIG_OUT), exist_ok=True)

    # Load class names in the SAME order used across your pipeline
    adata = sc.read_h5ad(ADATA_PATH)
    if LABEL_KEY not in adata.obs:
        raise ValueError(f"Expected adata.obs['{LABEL_KEY}']")
    class_names = list(adata.obs[LABEL_KEY].astype("category").cat.categories)
    n_classes = len(class_names)

    # Load arrays
    lr_true = np.load(LR_TRUE).astype(int)
    lr_pred = np.load(LR_PRED).astype(int)

    xgb_true = np.load(XGB_TRUE).astype(int)
    xgb_pred = np.load(XGB_PRED).astype(int)

    sann_true = np.load(SANN_TRUE).astype(int)
    sann_pred = np.load(SANN_PRED).astype(int)

    # Quick sanity
    for name, y_t, y_p in [
        ("LR", lr_true, lr_pred),
        ("XGB", xgb_true, xgb_pred),
        ("SANN", sann_true, sann_pred),
    ]:
        if len(y_t) != len(y_p):
            raise ValueError(f"{name}: y_true and y_pred length mismatch")
        if y_p.min() < 0 or y_p.max() >= n_classes:
            raise ValueError(f"{name}: predictions out of range for n_classes={n_classes}")

    # Confusion matrices (raw -> row-normalised)
    labels = np.arange(n_classes)

    cm_lr = row_normalise(confusion_matrix(lr_true, lr_pred, labels=labels))
    cm_xgb = row_normalise(confusion_matrix(xgb_true, xgb_pred, labels=labels))
    cm_sann = row_normalise(confusion_matrix(sann_true, sann_pred, labels=labels))

    # Plot 1x3
    fig, axes = plt.subplots(1, 3, figsize=(22, 7), constrained_layout=True)

    im0 = plot_cm(axes[0], cm_lr, class_names, "Logistic Regression")
    im1 = plot_cm(axes[1], cm_xgb, class_names, "XGBoost")
    im2 = plot_cm(axes[2], cm_sann, class_names, "SANN")

    # ONE colorbar for all (on the right)
    cbar = fig.colorbar(im2, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label("Row-normalised proportion", fontsize=LABEL_FONTSIZE)
    cbar.ax.tick_params(labelsize=TICK_FONTSIZE)

    fig.savefig(FIG_OUT, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[Saved] {FIG_OUT}")


if __name__ == "__main__":
    main()