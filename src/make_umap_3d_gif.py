#!/usr/bin/env python
"""
Render a rotating GIF of the 3D PBMC68K embedding (UMAP1, UMAP2, PC1)
coloured by SANN_PCA predicted cell type, mirroring the static figure in
results/figures/umap_3d_sann_pred_full.png.

Re-uses the saved predictions in results/full_train/sann_all_pred.npy and
the 3D coordinates derived from the h5ad file's existing UMAP and PCA.
"""
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["MPLBACKEND"] = "Agg"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--adata", default="data/processed/pbmc68k_labeled.h5ad")
    p.add_argument("--label_key", default="cell_type")
    p.add_argument("--pred_npy", default="results/full_train/sann_all_pred.npy")
    p.add_argument("--out_gif", default="results/figures/umap_3d_sann_pred_full.gif")
    p.add_argument("--n_frames", type=int, default=72)
    p.add_argument("--fps", type=int, default=18)
    p.add_argument("--elev", type=float, default=24.0)
    p.add_argument("--point_size", type=float, default=3.0)
    p.add_argument("--alpha", type=float, default=0.65)
    p.add_argument("--fig_w", type=float, default=9.0)
    p.add_argument("--fig_h", type=float, default=6.5)
    p.add_argument("--dpi", type=int, default=110)
    p.add_argument("--max_cells", type=int, default=30000,
                   help="Subsample to this many cells to keep GIF small.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_gif), exist_ok=True)

    print(f"[Loading] {args.adata}")
    adata = sc.read_h5ad(args.adata)
    if "X_umap" not in adata.obsm or "X_pca" not in adata.obsm:
        raise RuntimeError("Need both X_umap and X_pca in adata.obsm")

    umap2d = np.asarray(adata.obsm["X_umap"], dtype=np.float32)
    pca = np.asarray(adata.obsm["X_pca"], dtype=np.float32)
    z = (pca[:, 0] - pca[:, 0].mean()) / (pca[:, 0].std() + 1e-8)
    coords = np.column_stack([umap2d[:, 0], umap2d[:, 1], z]).astype(np.float32)

    y_cat = adata.obs[args.label_key].astype("category")
    class_names = list(y_cat.cat.categories)

    print(f"[Loading] {args.pred_npy}")
    y_pred = np.load(args.pred_npy)
    pred_labels = np.array([class_names[i] for i in y_pred], dtype=object)

    n = coords.shape[0]
    if args.max_cells and args.max_cells < n:
        rng = np.random.default_rng(args.seed)
        sel = rng.choice(n, size=args.max_cells, replace=False)
        coords = coords[sel]
        pred_labels = pred_labels[sel]
        print(f"[Subsample] {n} -> {args.max_cells} cells")

    cmap = plt.get_cmap("tab20")
    palette = {label: cmap(i % 20) for i, label in enumerate(class_names)}
    colors = np.array([palette[c] for c in pred_labels])

    fig = plt.figure(figsize=(args.fig_w, args.fig_h), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        coords[:, 0], coords[:, 1], coords[:, 2],
        s=args.point_size, alpha=args.alpha, c=colors, linewidths=0,
    )
    ax.set_xlabel("UMAP1", labelpad=10)
    ax.set_ylabel("UMAP2", labelpad=10)
    ax.set_zlabel("PC1", labelpad=10)
    ax.set_title("PBMC68K: SANN_PCA predicted cell type", pad=14)
    ax.set_xlim(coords[:, 0].min(), coords[:, 0].max())
    ax.set_ylim(coords[:, 1].min(), coords[:, 1].max())
    ax.set_zlim(coords[:, 2].min(), coords[:, 2].max())
    ax.grid(False)

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="None",
               markerfacecolor=palette[name], markeredgecolor="none",
               markersize=6, label=name)
        for name in class_names
    ]
    ax.legend(
        handles=legend_handles, loc="center left",
        bbox_to_anchor=(1.02, 0.5), frameon=False,
        title="Predicted", borderaxespad=0.5, fontsize=8,
    )

    azimuths = np.linspace(0, 360, args.n_frames, endpoint=False)

    def update(i):
        ax.view_init(elev=args.elev, azim=float(azimuths[i]))
        return ax,

    anim = animation.FuncAnimation(
        fig, update, frames=len(azimuths),
        interval=int(1000 / args.fps), blit=False, repeat=True,
    )
    writer = animation.PillowWriter(fps=args.fps)
    anim.save(args.out_gif, writer=writer, dpi=args.dpi)
    plt.close(fig)
    print(f"[Saved] {args.out_gif}")


if __name__ == "__main__":
    main()
