#!/usr/bin/env python
"""
Animated grouped bar chart of cross-donor macro-F1 across the 6 train/test
rotation pairs. Bars grow in by model (LR -> XGB -> SANN) so the SANN
advantage on the harder rotations is the visual climax.

Reads results/figures/transfer_matrix.csv and writes
results/figures/transfer_bar.gif.
"""
import os

os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

ROOT = "/Users/Abdullah/scRNA-robust-ml"
CSV_IN = os.path.join(ROOT, "results/figures/transfer_matrix.csv")
GIF_OUT = os.path.join(ROOT, "results/figures/transfer_bar.gif")

DONORS = ["68K", "8K", "3K"]
MODELS = ["LR_PCA", "XGB_PCA", "SANN_PCA"]
LABELS = {"LR_PCA": "LR", "XGB_PCA": "XGB", "SANN_PCA": "SANN"}
COLORS = {"LR_PCA": "#4C72B0", "XGB_PCA": "#DD8452", "SANN_PCA": "#2CA02C"}


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


def collect_pairs(mats):
    """Off-diagonal pairs sorted by SANN ascending."""
    pairs = []
    for i, tr in enumerate(DONORS):
        for j, te in enumerate(DONORS):
            if i == j:
                continue
            pairs.append({
                "label": f"{tr}->{te}",
                "LR_PCA":   float(mats["LR_PCA"][i, j]),
                "XGB_PCA":  float(mats["XGB_PCA"][i, j]),
                "SANN_PCA": float(mats["SANN_PCA"][i, j]),
            })
    pairs.sort(key=lambda r: r["SANN_PCA"])
    return pairs


def main():
    os.makedirs(os.path.dirname(GIF_OUT), exist_ok=True)
    mats = load_matrices(CSV_IN)
    pairs = collect_pairs(mats)
    n = len(pairs)
    x = np.arange(n)
    bar_w = 0.26
    offsets = {"LR_PCA": -bar_w, "XGB_PCA": 0.0, "SANN_PCA": +bar_w}

    fig, ax = plt.subplots(figsize=(10, 5.2), constrained_layout=True)
    bar_groups = {}
    text_groups = {}
    for m in MODELS:
        bars = ax.bar(
            x + offsets[m], np.zeros(n), width=bar_w,
            color=COLORS[m], edgecolor="black", linewidth=0.6,
            label=LABELS[m],
        )
        bar_groups[m] = bars
        text_groups[m] = [
            ax.text(b.get_x() + b.get_width() / 2, 0.0, "",
                    ha="center", va="bottom", fontsize=8)
            for b in bars
        ]

    ax.set_xticks(x)
    ax.set_xticklabels([p["label"] for p in pairs], fontsize=10)
    ax.set_ylabel("macro-F1 (known)", fontsize=11)
    ax.set_title("Cross-donor transfer, sorted by SANN ascending",
                 fontsize=12, pad=10)
    ax.set_ylim(0.50, 1.05)
    ax.axhline(1.0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=True, fontsize=10, ncol=1)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    grow_frames = 22
    hold_frames = 6
    final_hold = 26
    total = (grow_frames + hold_frames) * len(MODELS) + final_hold

    def update(f):
        for k, m in enumerate(MODELS):
            seg_start = k * (grow_frames + hold_frames)
            seg_grow_end = seg_start + grow_frames
            target = np.array([p[m] for p in pairs], dtype=np.float64)
            if f < seg_start:
                heights = np.zeros(n)
            elif f < seg_grow_end:
                t = (f - seg_start + 1) / grow_frames
                t = max(0.0, min(1.0, t))
                heights = target * t
            else:
                heights = target
            for b, h, txt, tv in zip(bar_groups[m], heights,
                                     text_groups[m], target):
                b.set_height(float(h))
                if h > 0.05:
                    txt.set_position((b.get_x() + b.get_width() / 2,
                                      float(h) + 0.008))
                    txt.set_text(f"{tv:.2f}" if h >= tv - 1e-3 else "")
                else:
                    txt.set_text("")
        return list(bar_groups["LR_PCA"]) + list(bar_groups["XGB_PCA"]) \
            + list(bar_groups["SANN_PCA"])

    anim = animation.FuncAnimation(
        fig, update, frames=total, interval=int(1000 / 24),
        blit=False, repeat=True,
    )
    anim.save(GIF_OUT, writer=animation.PillowWriter(fps=24), dpi=120)
    plt.close(fig)
    print(f"Saved: {GIF_OUT}")


if __name__ == "__main__":
    main()
