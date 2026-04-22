#!/usr/bin/env python
"""
Cross-donor transfer figure (bar chart only).

Grouped bar chart of macro-F1 on the 6 cross-donor rotation pairs.
Three bars per pair (LR_PCA, XGB_PCA, SANN_PCA). Pairs sorted along the
x-axis by SANN performance, ascending. Legend in upper-left.

The 3x3 SANN heatmap and full per-model matrices live in the appendix
(transfer_appendix.csv + the earlier transfer_matrix.png).

Reads pre-computed macro-F1 from results/figures/transfer_matrix.csv
(produced by plot_transfer_matrix.py). Does not retrain anything.
"""
import os

os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec

ROOT = "/Users/Abdullah/scRNA-robust-ml"
CSV_IN = os.path.join(ROOT, "results/figures/transfer_matrix.csv")
PNG_OUT = os.path.join(ROOT, "results/figures/transfer_bar.png")
PDF_OUT = os.path.join(ROOT, "results/figures/transfer_bar.pdf")

DONORS = ["68K", "8K", "3K"]
MODELS = ["LR_PCA", "XGB_PCA", "SANN_PCA"]
MODEL_COLORS = {
    "LR_PCA":   "#4C72B0",   # blue
    "XGB_PCA":  "#DD8452",   # orange
    "SANN_PCA": "#2CA02C",   # green
}


def load_matrices(csv_path):
    df = pd.read_csv(csv_path)
    mats = {}
    for m in MODELS:
        sub = df[df["Model"] == m]
        mat = np.full((3, 3), np.nan)
        for _, r in sub.iterrows():
            i = DONORS.index(r["Train"])
            j = DONORS.index(r["Test"])
            mat[i, j] = r["MacroF1"]
        mats[m] = mat
    return mats


def plot_panel_a(ax, mat, title):
    """SANN 3x3 heatmap, diagonal distinguished with hatch + bold outline."""
    norm = Normalize(vmin=0.55, vmax=1.00)
    cmap = plt.get_cmap("viridis")
    im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="equal")

    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels([f"test\n{d}" for d in DONORS], fontsize=10)
    ax.set_yticklabels([f"train {d}" for d in DONORS], fontsize=10)
    ax.set_title(title, fontsize=12, pad=10)

    # Cell annotations + diagonal hatching/outline
    for i in range(3):
        for j in range(3):
            v = mat[i, j]
            color = "white" if v < 0.78 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color=color, fontsize=13, fontweight="bold")

            if i == j:
                # Outline diagonal cells with thick white border
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    fill=False, edgecolor="white", linewidth=3.0,
                    zorder=5,
                ))
                # Hatch overlay to distinguish within-donor
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    fill=False, edgecolor="white", linewidth=0,
                    hatch="//", zorder=4, alpha=0.35,
                ))

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    # Legend patch for diagonal convention
    legend_patch = mpatches.Patch(
        facecolor="none", edgecolor="white", linewidth=2.0, hatch="//",
        label="within-donor (val split)",
    )
    ax.legend(handles=[legend_patch], loc="upper center",
              bbox_to_anchor=(0.5, -0.12), frameon=False, fontsize=9)

    return im


def plot_panel_b(ax, mats):
    """Grouped bar chart: 6 cross-donor pairs, sorted by SANN ascending."""
    # Collect off-diagonal pairs
    pairs = []
    for i, tr in enumerate(DONORS):
        for j, te in enumerate(DONORS):
            if i == j:
                continue
            pairs.append({
                "label": f"{tr}→{te}",
                "train": tr, "test": te,
                "LR_PCA":   mats["LR_PCA"][i, j],
                "XGB_PCA":  mats["XGB_PCA"][i, j],
                "SANN_PCA": mats["SANN_PCA"][i, j],
            })
    # Sort by SANN ascending
    pairs.sort(key=lambda r: r["SANN_PCA"])

    n = len(pairs)
    x = np.arange(n)
    bar_w = 0.26
    offsets = {"LR_PCA": -bar_w, "XGB_PCA": 0.0, "SANN_PCA": +bar_w}
    labels_pretty = {"LR_PCA": "LR", "XGB_PCA": "XGB", "SANN_PCA": "SANN"}

    for m in MODELS:
        vals = [p[m] for p in pairs]
        bars = ax.bar(
            x + offsets[m], vals, width=bar_w,
            color=MODEL_COLORS[m], edgecolor="black", linewidth=0.6,
            label=labels_pretty[m],
        )
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.008,
                    f"{v:.2f}", ha="center", va="bottom",
                    fontsize=8, rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels([p["label"] for p in pairs], fontsize=10)
    ax.set_ylabel("macro-F1 (known)", fontsize=11)
    ax.set_title("Cross-donor transfer — sorted by SANN ascending",
                 fontsize=12, pad=10)
    ax.set_ylim(0.50, 1.05)
    ax.axhline(1.0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=True, fontsize=10, ncol=1)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def main():
    os.makedirs(os.path.dirname(PNG_OUT), exist_ok=True)
    mats = load_matrices(CSV_IN)

    fig, ax = plt.subplots(figsize=(10, 5.2), constrained_layout=True)
    plot_panel_b(ax, mats)

    fig.savefig(PNG_OUT, dpi=220, bbox_inches="tight")
    fig.savefig(PDF_OUT, bbox_inches="tight")
    print(f"Saved: {PNG_OUT}")
    print(f"Saved: {PDF_OUT}")

    # Appendix-ready full 3x3 table for all models
    appendix_path = os.path.join(ROOT, "results/figures/transfer_appendix.csv")
    rows = []
    for m in MODELS:
        for i, tr in enumerate(DONORS):
            for j, te in enumerate(DONORS):
                rows.append({
                    "Model": m, "Train": tr, "Test": te,
                    "MacroF1": round(float(mats[m][i, j]), 4),
                    "WithinDonor": i == j,
                })
    pd.DataFrame(rows).to_csv(appendix_path, index=False)
    print(f"Saved: {appendix_path}")


if __name__ == "__main__":
    main()
