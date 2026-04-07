# src/plot_per_class_heatmap.py
"""
3-panel heatmap: per-class F1, Precision, Recall across LR / XGBoost / SANN.
Columns ordered by ascending SANN F1 (hardest classes on the left).
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import precision_recall_fscore_support

os.environ["MPLBACKEND"] = "Agg"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


CLASS_NAMES = [
    "B cells", "CD8 T", "CL 0", "CL 12", "CL 13", "CL 15",
    "CL 16", "CL 19", "CL 6", "CL 8", "CL 9", "Mono", "NK", "Platelet",
]

MODEL_NAMES = ["LR", "XGBoost", "SANN"]
MODEL_PREFIXES = ["lr", "xgb", "sann"]


def parse_args():
    p = argparse.ArgumentParser(
        description="Per-class metrics heatmap (F1 / Precision / Recall)."
    )
    p.add_argument("--rep", choices=["hvg", "pca"], default="hvg")
    p.add_argument("--hvg_dir", default="results/full_train_all_hvg")
    p.add_argument("--pca_dir", default="results/full_train_all_pca")
    p.add_argument("--outdir", default="results/figures")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--cmap", default="Blues")
    return p.parse_args()


def load_pred_true(base_dir, prefix):
    pred = np.load(os.path.join(base_dir, f"{prefix}_test_pred.npy"))
    true = np.load(os.path.join(base_dir, f"{prefix}_test_true.npy"))
    return true, pred


def compute_per_class(y_true, y_pred, n_classes):
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n_classes)), zero_division=0.0
    )
    return p, r, f


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    base_dir = args.hvg_dir if args.rep == "hvg" else args.pca_dir
    rep_label = "HVG" if args.rep == "hvg" else "PCA"
    n_classes = len(CLASS_NAMES)

    # ------------------------------------------------------------------
    # Compute per-class metrics for each model  (rows = models)
    # ------------------------------------------------------------------
    precision = np.zeros((3, n_classes))
    recall = np.zeros((3, n_classes))
    f1 = np.zeros((3, n_classes))

    for i, prefix in enumerate(MODEL_PREFIXES):
        y_true, y_pred = load_pred_true(base_dir, prefix)
        p, r, f = compute_per_class(y_true, y_pred, n_classes)
        precision[i] = p
        recall[i] = r
        f1[i] = f

    # ------------------------------------------------------------------
    # Column order: ascending SANN F1 (hardest on the left)
    # ------------------------------------------------------------------
    sann_f1 = f1[2]  # SANN is index 2
    col_order = np.argsort(sann_f1)  # ascending

    sorted_names = [CLASS_NAMES[c] for c in col_order]
    f1_sorted = f1[:, col_order]
    prec_sorted = precision[:, col_order]
    rec_sorted = recall[:, col_order]

    panels = [
        ("F1 Score", f1_sorted),
        ("Precision", prec_sorted),
        ("Recall", rec_sorted),
    ]

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]

    fig, axes = plt.subplots(
        3, 1,
        figsize=(16, 8),
        gridspec_kw={"hspace": 0.08},
    )

    cmap = plt.get_cmap(args.cmap)
    norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)

    for idx, (metric_label, data) in enumerate(panels):
        ax = axes[idx]

        im = ax.imshow(
            data, cmap=cmap, norm=norm, aspect="auto",
        )

        # Thin white gridlines
        for edge in range(n_classes + 1):
            ax.axvline(edge - 0.5, color="white", linewidth=1.2)
        for edge in range(4):
            ax.axhline(edge - 0.5, color="white", linewidth=1.2)

        # Annotate cells
        for row in range(3):
            for col in range(n_classes):
                val = data[row, col]
                txt_color = "white" if val > 0.85 else "#222222"
                ax.text(
                    col, row, f"{val:.2f}",
                    ha="center", va="center",
                    fontsize=8.5, color=txt_color,
                    fontweight="medium",
                )

        # Y-axis: model names
        ax.set_yticks(range(3))
        ax.set_yticklabels(MODEL_NAMES, fontsize=11)

        # Metric label on the left
        ax.set_ylabel(metric_label, fontsize=13, fontweight="semibold", labelpad=12)

        # X-axis labels only on the bottom panel
        if idx == 2:
            ax.set_xticks(range(n_classes))
            ax.set_xticklabels(sorted_names, fontsize=10, rotation=45, ha="right")
        else:
            ax.set_xticks(range(n_classes))
            ax.set_xticklabels([])

        ax.tick_params(length=0)

    # Horizontal separator lines between panels
    for idx in range(2):
        line_y = axes[idx].get_position().y0
        fig.add_artist(
            plt.Line2D(
                [0.06, 0.88], [line_y, line_y],
                transform=fig.transFigure,
                color="#999999", linewidth=0.8,
            )
        )

    # Shared colourbar
    cbar_ax = fig.add_axes([0.91, 0.12, 0.015, 0.76])
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cbar_ax,
    )
    cbar.set_label("Score", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    fig.subplots_adjust(left=0.10, right=0.89, top=0.95, bottom=0.14)

    out_path = os.path.join(args.outdir, f"per_class_metrics_heatmap_{args.rep}.png")
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"[Saved] {out_path}")
    print(f"Column order (ascending SANN F1): {sorted_names}")


if __name__ == "__main__":
    main()
