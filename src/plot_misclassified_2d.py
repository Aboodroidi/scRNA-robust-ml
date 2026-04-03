# src/plot_misclassified_2d.py
"""
Plot misclassified cells on a 2-D UMAP embedding.
Correct cells in steelblue, misclassified cells in red (drawn on top).
"""
import os
import argparse
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# macOS safe settings
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["MPLBACKEND"] = "Agg"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def parse_args():
    p = argparse.ArgumentParser(description="Plot misclassified cells in 2-D UMAP")

    p.add_argument("--adata", default="data/processed/pbmc68k_labeled.h5ad")
    p.add_argument("--label_key", default="cell_type")
    p.add_argument("--pred", default="results/full_train/sann_all_pred.npy")
    p.add_argument("--umap_key", default="X_umap")

    p.add_argument("--out", default="results/figures/misclassified_2d.png")

    p.add_argument("--fig_w", type=float, default=8.5)
    p.add_argument("--fig_h", type=float, default=7.5)
    p.add_argument("--point_size_bg", type=float, default=6.0)
    p.add_argument("--point_size_fg", type=float, default=14.0)
    p.add_argument("--alpha_bg", type=float, default=0.20)
    p.add_argument("--alpha_fg", type=float, default=0.95)
    p.add_argument("--dpi", type=int, default=300)

    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    adata = sc.read_h5ad(args.adata)

    umap = adata.obsm[args.umap_key]
    x = umap[:, 0]
    y = umap[:, 1]

    true = adata.obs[args.label_key].astype("category")
    y_true = true.cat.codes.values
    y_pred = np.load(args.pred)

    if len(y_true) != len(y_pred):
        raise ValueError(f"Prediction length mismatch: true={len(y_true)} pred={len(y_pred)}")

    correct = y_true == y_pred
    wrong = ~correct

    n_correct = int(correct.sum())
    n_wrong = int(wrong.sum())
    acc = n_correct / len(y_true) * 100

    print(f"Total cells:   {len(y_true)}")
    print(f"Correct:       {n_correct}")
    print(f"Misclassified: {n_wrong}")
    print(f"Accuracy:      {acc:.2f}%")

    fig, ax = plt.subplots(figsize=(args.fig_w, args.fig_h))

    # Background: all cells in neutral grey
    ax.scatter(
        x, y,
        s=args.point_size_bg,
        alpha=args.alpha_bg,
        linewidths=0,
        color="#B0B0B0",
        edgecolors="none",
    )

    # Correct classifications
    ax.scatter(
        x[correct], y[correct],
        s=args.point_size_bg,
        alpha=args.alpha_bg,
        linewidths=0,
        color="steelblue",
        edgecolors="none",
    )

    # Misclassifications (foreground, on top)
    ax.scatter(
        x[wrong], y[wrong],
        s=args.point_size_fg,
        alpha=args.alpha_fg,
        linewidths=0,
        color="red",
        edgecolors="none",
    )

    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title("Misclassified Cells on UMAP (SANN)")
    ax.grid(False)

    # Legend with counts — outside to avoid covering embedding
    legend_elements = [
        Line2D([0], [0], marker="o", color="w",
               label=f"Correct ({n_correct:,})",
               markerfacecolor="steelblue", markeredgecolor="none", markersize=8),
        Line2D([0], [0], marker="o", color="w",
               label=f"Misclassified ({n_wrong:,})",
               markerfacecolor="red", markeredgecolor="none", markersize=8),
    ]

    ax.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
    )

    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
