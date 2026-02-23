# src/plot_confusion_matrices_combined.py
import os
import argparse
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def parse_args():
    p = argparse.ArgumentParser(description="Plot 3 confusion matrices side-by-side with ONE shared colorbar.")
    p.add_argument("--data", type=str, default="data/processed/pbmc68k_labeled.h5ad")
    p.add_argument("--label_key", type=str, default="cell_type")

    p.add_argument("--lr_true", type=str, default="results/lr_test_true.npy")
    p.add_argument("--lr_pred", type=str, default="results/lr_test_pred.npy")

    p.add_argument("--xgb_true", type=str, default="results/xgb_test_true.npy")
    p.add_argument("--xgb_pred", type=str, default="results/xgb_test_pred.npy")

    p.add_argument("--sann_true", type=str, default="results/sann_test_true.npy")
    p.add_argument("--sann_pred", type=str, default="results/sann_test_pred.npy")

    p.add_argument("--out", type=str,
                   default="results/figures/confusion_matrices_all_models.png")

    p.add_argument("--dpi", type=int, default=300)

    # Bigger labels
    p.add_argument("--fig_w", type=float, default=18)
    p.add_argument("--fig_h", type=float, default=6)
    p.add_argument("--title_size", type=int, default=20)
    p.add_argument("--panel_title_size", type=int, default=17)
    p.add_argument("--tick_size", type=int, default=12)
    p.add_argument("--axis_label_size", type=int, default=14)

    p.add_argument("--max_label_len", type=int, default=16)
    return p.parse_args()


def shorten_labels(labels, max_len=16):
    out = []
    for s in labels:
        s = str(s)
        out.append(s if len(s) <= max_len else s[: max_len - 1] + "…")
    return out


def load_class_names(data_path: str, label_key: str):
    adata = sc.read_h5ad(data_path)
    y_cat = adata.obs[label_key].astype("category")
    return list(y_cat.cat.categories)


def load_arrays(true_path: str, pred_path: str):
    y_true = np.load(true_path).astype(int)
    y_pred = np.load(pred_path).astype(int)
    return y_true, y_pred


def row_norm_cm(y_true, y_pred, n_classes):
    labels = np.arange(n_classes)
    cm = confusion_matrix(y_true, y_pred, labels=labels).astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)
    return cm


def plot_panel(ax, cm, title, tick_labels, args):
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1, aspect="auto")

    ax.set_title(title, fontsize=args.panel_title_size)
    ax.set_xlabel("Predicted label", fontsize=args.axis_label_size)
    ax.set_ylabel("True label", fontsize=args.axis_label_size)

    ax.set_xticks(np.arange(len(tick_labels)))
    ax.set_yticks(np.arange(len(tick_labels)))

    ax.set_xticklabels(tick_labels, fontsize=args.tick_size, rotation=45, ha="right")
    ax.set_yticklabels(tick_labels, fontsize=args.tick_size)

    ax.grid(False)
    return im


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    class_names = load_class_names(args.data, args.label_key)
    n_classes = len(class_names)
    tick_labels = shorten_labels(class_names, args.max_label_len)

    lr_true, lr_pred = load_arrays(args.lr_true, args.lr_pred)
    xgb_true, xgb_pred = load_arrays(args.xgb_true, args.xgb_pred)
    sann_true, sann_pred = load_arrays(args.sann_true, args.sann_pred)

    cm_lr = row_norm_cm(lr_true, lr_pred, n_classes)
    cm_xgb = row_norm_cm(xgb_true, xgb_pred, n_classes)
    cm_sann = row_norm_cm(sann_true, sann_pred, n_classes)

    fig, axes = plt.subplots(1, 3, figsize=(args.fig_w, args.fig_h),
                             constrained_layout=True)

    fig.suptitle("Confusion Matrices (Row-normalised)",
                 fontsize=args.title_size)

    im0 = plot_panel(axes[0], cm_lr, "Logistic Regression",
                     tick_labels, args)
    im1 = plot_panel(axes[1], cm_xgb, "XGBoost",
                     tick_labels, args)
    im2 = plot_panel(axes[2], cm_sann, "SANN",
                     tick_labels, args)

    # ONE shared colorbar
    cbar = fig.colorbar(im2, ax=axes.ravel().tolist(),
                        shrink=0.92, pad=0.02)
    cbar.set_label("Row-normalised proportion",
                   fontsize=args.axis_label_size)
    cbar.ax.tick_params(labelsize=args.tick_size)

    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"[Saved] {args.out}")


if __name__ == "__main__":
    main()