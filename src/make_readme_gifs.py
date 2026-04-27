#!/usr/bin/env python
"""
Build animated GIFs for the README from already-computed results.

Outputs (saved to results/figures/):
  transfer_heatmap_cycle.gif: 3x3 train/test macro-F1 matrix, cycling
    LR_PCA -> XGB_PCA -> SANN_PCA.
  sann_training_curves.gif: val macro-F1 vs epoch, drawn in epoch by
    epoch for each saved SANN seed (8K-trained run by default).
  bootstrap_delta.gif: paired bootstrap delta (SANN_PCA - Seurat) on
    the 8K test set, histogram building up over 1000 resamples with
    the 95% CI band added at the end.

Deps: matplotlib, numpy, pandas, pillow (for GIF writer).
"""
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["MPLBACKEND"] = "Agg"

import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from sklearn.metrics import f1_score

ROOT = "/Users/Abdullah/scRNA-robust-ml"
OUT_DIR = os.path.join(ROOT, "results/figures")
os.makedirs(OUT_DIR, exist_ok=True)

DONORS = ["68K", "8K", "3K"]
MODELS = ["LR_PCA", "XGB_PCA", "SANN_PCA"]


# GIF 1: cycling 3x3 transfer heatmap
def gif_transfer_cycle(out_path, fps=1.4):
    df = pd.read_csv(os.path.join(OUT_DIR, "transfer_matrix.csv"))
    mats = {}
    for m in MODELS:
        sub = df[df["Model"] == m]
        mat = np.full((3, 3), np.nan)
        for _, r in sub.iterrows():
            i = DONORS.index(r["Train"])
            j = DONORS.index(r["Test"])
            mat[i, j] = r["MacroF1"]
        mats[m] = mat

    norm = Normalize(vmin=0.55, vmax=1.00)
    cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(5.4, 5.4), constrained_layout=True)
    im = ax.imshow(mats[MODELS[0]], cmap=cmap, norm=norm, aspect="equal")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("macro-F1", fontsize=10)

    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels([f"test {d}" for d in DONORS], fontsize=11)
    ax.set_yticklabels([f"train {d}" for d in DONORS], fontsize=11)

    title = ax.set_title(MODELS[0], fontsize=14, pad=10)
    texts = []
    for i in range(3):
        row = []
        for j in range(3):
            row.append(ax.text(j, i, "", ha="center", va="center",
                               fontsize=14, fontweight="bold"))
        texts.append(row)

    # Outline diagonals (within-donor) statically
    for i in range(3):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                                   fill=False, edgecolor="white",
                                   linewidth=2.5, zorder=5))

    # Animation: cycle each model, hold for several frames so eye can read it
    HOLD = 4   # frames per model
    sequence = [(m, k) for m in MODELS for k in range(HOLD)]

    def update(frame_idx):
        m, _hold = sequence[frame_idx]
        mat = mats[m]
        im.set_data(mat)
        title.set_text(f"{m}    (rows = train donor, cols = test donor)")
        for i in range(3):
            for j in range(3):
                v = mat[i, j]
                texts[i][j].set_text(f"{v:.2f}")
                texts[i][j].set_color("white" if v < 0.78 else "black")
        return [im, title] + [t for row in texts for t in row]

    anim = animation.FuncAnimation(
        fig, update, frames=len(sequence), interval=int(1000 / fps),
        blit=False, repeat=True,
    )

    writer = animation.PillowWriter(fps=fps)
    anim.save(out_path, writer=writer, dpi=120)
    plt.close(fig)
    print(f"Saved: {out_path}")


# GIF 2: SANN training curves drawing in
def gif_training_curves(out_path, train_dir=None, fps=20):
    if train_dir is None:
        train_dir = os.path.join(ROOT, "results/train_8k")
    seed_files = sorted(glob.glob(
        os.path.join(train_dir, "sann_model_seed*_history.csv")))
    if not seed_files:
        print(f"No seed history files in {train_dir}")
        return

    histories = []
    for fp in seed_files:
        d = pd.read_csv(fp)
        histories.append(d[["epoch", "val_macro_f1"]].values)
    max_ep = int(max(h[-1, 0] for h in histories))

    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    ax.set_xlim(0, max_ep + 2)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("epoch", fontsize=11)
    ax.set_ylabel("validation macro-F1", fontsize=11)
    title_src = os.path.basename(train_dir)
    ax.set_title(f"SANN training curves across seeds ({title_src})",
                 fontsize=12, pad=10)
    ax.grid(alpha=0.3)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    cmap = plt.get_cmap("viridis")
    colors = [cmap(0.15 + 0.7 * i / max(1, len(histories) - 1))
              for i in range(len(histories))]
    lines = []
    dots = []
    for i, h in enumerate(histories):
        ln, = ax.plot([], [], lw=1.6, color=colors[i],
                      label=f"seed {i}")
        dt, = ax.plot([], [], "o", color=colors[i], markersize=5)
        lines.append(ln)
        dots.append(dt)
    ax.legend(loc="lower right", frameon=True, fontsize=9, ncol=3)

    txt = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=11,
                  verticalalignment="top",
                  bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.85))

    n_frames = max_ep + 6
    def update(f):
        for i, h in enumerate(histories):
            mask = h[:, 0] <= f
            xs = h[mask, 0]
            ys = h[mask, 1]
            lines[i].set_data(xs, ys)
            if len(xs) > 0:
                dots[i].set_data([xs[-1]], [ys[-1]])
            else:
                dots[i].set_data([], [])
        # Show running mean across seeds at this epoch
        cur = []
        for h in histories:
            mask = h[:, 0] <= f
            if mask.sum() > 0:
                cur.append(h[mask, 1][-1])
        if cur:
            txt.set_text(f"epoch {min(int(f), max_ep)}    "
                         f"mean F1 = {np.mean(cur):.3f}    "
                         f"std = {np.std(cur):.3f}")
        return lines + dots + [txt]

    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=int(1000 / fps),
        blit=False, repeat=True,
    )
    writer = animation.PillowWriter(fps=fps)
    anim.save(out_path, writer=writer, dpi=120)
    plt.close(fig)
    print(f"Saved: {out_path}")


# GIF 3: paired bootstrap delta (SANN - Seurat), histogram building up
def gif_bootstrap_delta(out_path, n_boot=1000, fps=30):
    """
    Build a histogram of bootstrap Δ macro-F1 (SANN − Seurat) on 8K,
    revealing one resample at a time.
    """
    import scanpy as sc

    CLASSES = ["B cells", "Mono", "NK", "Platelet", "T cells"]
    CLS_IDX = {c: i for i, c in enumerate(CLASSES)}
    SANN_INT_TO_NAME = {i: c for i, c in enumerate(sorted(CLASSES))}
    LABEL_NORM = {
        "B cells": "B cells", "Mono": "Mono", "Monocytes": "Mono",
        "NK": "NK", "Platelet": "Platelet", "T cells": "T cells",
        "CD8 T": "T cells", "CD4 T": "T cells",
        "FCGR3A+ Monocytes": "Mono", "CD14+ Monocytes": "Mono",
        "Megakaryocytes": "Platelet",
    }

    def _norm(s):
        if pd.isna(s):
            return None
        s = str(s).strip()
        return LABEL_NORM.get(s, None)

    # Load 8K truth + SANN
    ad = sc.read_h5ad(os.path.join(ROOT, "data/processed/pbmc8k_labeled.h5ad"))
    raw = ad.obs["cell_type"].astype(str).to_numpy()
    mapped = np.array([_norm(s) for s in raw], dtype=object)
    known = np.array([m is not None for m in mapped])
    barcodes = ad.obs_names[known].to_numpy()
    y_true = np.array([mapped[i] for i, k in enumerate(known) if k],
                      dtype=object)

    sann_int = np.load(os.path.join(
        ROOT, "results/external_validation/pca/sann_pca_ext_pred.npy"))
    sann_pred = np.array([SANN_INT_TO_NAME[i] for i in sann_int],
                         dtype=object)

    seurat_df = pd.read_csv(os.path.join(
        ROOT, "results/comparators/seurat/seurat_8k_predictions.csv"))
    bmap = dict(zip(seurat_df["barcode"].astype(str),
                    seurat_df["predicted_label"].astype(str)))
    seurat_pred = np.array(
        [_norm(bmap.get(str(b), None)) or "OTHER" for b in barcodes],
        dtype=object)

    label_set = sorted({c for c in y_true if c in CLS_IDX})
    labels_idx = [CLS_IDX[c] for c in label_set]

    def _f1(yt, yp):
        return float(f1_score(
            [CLS_IDX[c] for c in yt],
            [CLS_IDX.get(c, -1) for c in yp],
            labels=labels_idx, average="macro", zero_division=0,
        ))

    rng = np.random.default_rng(42)
    n = len(y_true)
    deltas = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        deltas[b] = _f1(y_true[idx], sann_pred[idx]) - \
                    _f1(y_true[idx], seurat_pred[idx])

    # Build histogram animation
    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    bins = np.linspace(deltas.min() - 0.001, deltas.max() + 0.001, 41)
    ax.set_xlim(bins[0], bins[-1])
    final_count, _ = np.histogram(deltas, bins=bins)
    ax.set_ylim(0, final_count.max() * 1.15)
    ax.axvline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.set_xlabel("Δ macro-F1   (SANN_PCA − Seurat) — 8K bootstrap",
                  fontsize=11)
    ax.set_ylabel("count", fontsize=11)
    title = ax.set_title("Paired bootstrap building up …", fontsize=12, pad=10)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    bar_container = ax.bar(
        (bins[:-1] + bins[1:]) / 2,
        np.zeros(len(bins) - 1),
        width=bins[1] - bins[0],
        color="#4C72B0", edgecolor="black", linewidth=0.4,
    )

    ci_low_line = ax.axvline(np.nan, color="red", linewidth=0, alpha=0.0)
    ci_high_line = ax.axvline(np.nan, color="red", linewidth=0, alpha=0.0)
    band = None
    annot = ax.text(0.02, 0.94, "", transform=ax.transAxes, fontsize=10,
                    verticalalignment="top",
                    bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.85))

    # Animate adding ~25 samples per frame
    BATCH = 25
    n_frames = (n_boot // BATCH) + 6

    state = {"band": None}

    def update(f):
        end = min(n_boot, (f + 1) * BATCH)
        seen = deltas[:end]
        counts, _ = np.histogram(seen, bins=bins)
        for rect, c in zip(bar_container, counts):
            rect.set_height(c)

        mean_d = float(seen.mean())
        title.set_text(f"Paired bootstrap — {end}/{n_boot} resamples")
        annot.set_text(f"Δ̄ = {mean_d:+.4f}    n = {end}")

        # Once all samples drawn, show CI band
        if end == n_boot and state["band"] is None:
            lo, hi = np.percentile(deltas, [2.5, 97.5])
            ci_low_line.set_xdata([lo, lo])
            ci_low_line.set_linewidth(2)
            ci_low_line.set_alpha(1.0)
            ci_low_line.set_color("red")
            ci_high_line.set_xdata([hi, hi])
            ci_high_line.set_linewidth(2)
            ci_high_line.set_alpha(1.0)
            ci_high_line.set_color("red")
            state["band"] = ax.axvspan(lo, hi, color="red", alpha=0.12)
            annot.set_text(
                f"Δ̄ = {mean_d:+.4f}    "
                f"95% CI = [{lo:+.4f}, {hi:+.4f}]\n"
                f"CI {'excludes' if (lo > 0 or hi < 0) else 'spans'} zero"
            )
        return list(bar_container) + [ci_low_line, ci_high_line, annot, title]

    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=int(1000 / fps),
        blit=False, repeat=True,
    )
    writer = animation.PillowWriter(fps=fps)
    anim.save(out_path, writer=writer, dpi=120)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    gif_transfer_cycle(os.path.join(OUT_DIR, "transfer_heatmap_cycle.gif"))
    gif_training_curves(os.path.join(OUT_DIR, "sann_training_curves.gif"))
    gif_bootstrap_delta(os.path.join(OUT_DIR, "bootstrap_delta.gif"))


if __name__ == "__main__":
    main()
