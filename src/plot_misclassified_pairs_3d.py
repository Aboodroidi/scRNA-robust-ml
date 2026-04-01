import os
import json
import argparse
from collections import Counter

import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# macOS-safe settings
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["MPLBACKEND"] = "Agg"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONUTF8"] = "1"
os.environ["LC_ALL"] = "en_US.UTF-8"
os.environ["LANG"] = "en_US.UTF-8"


def parse_args():
    p = argparse.ArgumentParser(
        description="3D embedding plot: misclassified SANN test cells coloured by top true→pred pairs."
    )
    p.add_argument("--adata", type=str, default="data/processed/pbmc68k_labeled.h5ad")
    p.add_argument("--splits", type=str, default="results/ablations/fixed_splits.json")
    p.add_argument("--true", type=str, default="results/full_train/sann_test_true.npy")
    p.add_argument("--pred", type=str, default="results/full_train/sann_test_pred.npy")
    p.add_argument("--out", type=str, default="results/figures/misclassified_pairs_3d_sann.png")

    p.add_argument("--label_key", type=str, default="cell_type")
    p.add_argument("--umap_key", type=str, default="X_umap")
    p.add_argument("--pca_key", type=str, default="X_pca")
    p.add_argument("--pca_z_index", type=int, default=0)

    p.add_argument("--top_k", type=int, default=8, help="Number of most common confusion pairs to show.")
    p.add_argument("--fig_w", type=float, default=11.0)
    p.add_argument("--fig_h", type=float, default=8.0)
    p.add_argument("--point_size_bg", type=float, default=3.0)
    p.add_argument("--point_size_fg", type=float, default=9.0)
    p.add_argument("--alpha_bg", type=float, default=0.08)
    p.add_argument("--alpha_fg", type=float, default=0.95)
    p.add_argument("--dpi", type=int, default=300)

    p.add_argument("--elev", type=float, default=24)
    p.add_argument("--azim", type=float, default=42)
    return p.parse_args()


def ensure_parent(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def shorten(s: str, max_len=28) -> str:
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


def load_test_idx(splits_path: str) -> np.ndarray:
    with open(splits_path, "r") as f:
        d = json.load(f)
    if "test_idx" not in d:
        raise ValueError(f"Expected 'test_idx' in {splits_path}. Found keys: {list(d.keys())}")
    return np.array(d["test_idx"], dtype=int)


def get_3d_coords(adata, umap_key: str, pca_key: str, pca_z_index: int):
    if umap_key not in adata.obsm:
        raise ValueError(f"Expected 2D UMAP in adata.obsm['{umap_key}']")
    if pca_key not in adata.obsm:
        raise ValueError(f"Expected PCA in adata.obsm['{pca_key}']")

    umap = np.asarray(adata.obsm[umap_key], dtype=np.float32)
    pca = np.asarray(adata.obsm[pca_key], dtype=np.float32)

    if umap.shape[1] < 2:
        raise ValueError(f"{umap_key} must have at least 2 dims. Found shape {umap.shape}")
    if pca.shape[1] <= pca_z_index:
        raise ValueError(f"{pca_key} has only {pca.shape[1]} dims; cannot use z index {pca_z_index}")

    x = umap[:, 0]
    y = umap[:, 1]
    z = pca[:, pca_z_index]

    # standardize z for nicer appearance
    z = (z - z.mean()) / (z.std() + 1e-8)

    return np.column_stack([x, y, z]).astype(np.float32)


def set_common_axes(ax, coords, elev, azim, title):
    ax.set_xlabel("UMAP1", labelpad=12)
    ax.set_ylabel("UMAP2", labelpad=12)
    ax.set_zlabel("PC1", labelpad=12)
    ax.set_title(title, pad=18)
    ax.view_init(elev=elev, azim=azim)

    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    ax.set_xlim(float(np.min(x)), float(np.max(x)))
    ax.set_ylim(float(np.min(y)), float(np.max(y)))
    ax.set_zlim(float(np.min(z)), float(np.max(z)))
    ax.grid(False)


def main():
    args = parse_args()
    ensure_parent(args.out)

    # 1) Load AnnData + check embedding
    adata = sc.read_h5ad(args.adata)

    if args.label_key not in adata.obs:
        raise ValueError(f"Label column missing: adata.obs['{args.label_key}'] not found.")

    coords = get_3d_coords(
        adata=adata,
        umap_key=args.umap_key,
        pca_key=args.pca_key,
        pca_z_index=args.pca_z_index,
    )

    # Class mapping from category order
    class_names = list(adata.obs[args.label_key].astype("category").cat.categories)
    num_classes = len(class_names)

    # 2) Load splits + test_idx
    test_idx = load_test_idx(args.splits)

    # 3) Load y_true / y_pred for test only
    y_true = np.load(args.true).astype(int)
    y_pred = np.load(args.pred).astype(int)

    print(f"[Sanity] adata.n_obs: {adata.n_obs}")
    print(f"[Sanity] 3D coords shape: {coords.shape}")
    print(f"[Sanity] #classes: {num_classes}")
    print(f"[Sanity] class names: {class_names}")
    print(f"[Sanity] len(test_idx): {len(test_idx)}")
    print(f"[Sanity] len(y_true): {len(y_true)} | len(y_pred): {len(y_pred)}")

    if len(test_idx) != len(y_true) or len(test_idx) != len(y_pred):
        raise ValueError("Mismatch: test_idx length must equal lengths of sann_test_true.npy and sann_test_pred.npy")

    if y_true.min() < 0 or y_true.max() >= num_classes:
        raise ValueError(f"y_true out of range: min={y_true.min()}, max={y_true.max()}, expected [0,{num_classes-1}]")
    if y_pred.min() < 0 or y_pred.max() >= num_classes:
        raise ValueError(f"y_pred out of range: min={y_pred.min()}, max={y_pred.max()}, expected [0,{num_classes-1}]")

    # 4) Misclassified mask (test only) + pair strings
    mis_test = (y_true != y_pred)
    n_mis = int(mis_test.sum())
    print(f"[Sanity] Misclassified test cells: {n_mis} / {len(test_idx)}")

    true_labels = [class_names[i] for i in y_true[mis_test]]
    pred_labels = [class_names[i] for i in y_pred[mis_test]]
    pairs = [f"{t} → {p}" for t, p in zip(true_labels, pred_labels)]

    # 5) Top-K pairs; rest -> Other
    counts = Counter(pairs)
    top_pairs = [p for p, _ in counts.most_common(args.top_k)]

    print(f"[Sanity] Top {args.top_k} confusion pairs:")
    for p in top_pairs:
        print(f"  {p}: {counts[p]}")

    # Full-length pair label array
    pair_full = np.array(["NA"] * adata.n_obs, dtype=object)

    mis_indices_full = test_idx[mis_test]
    mis_pair_labels = [pair if pair in top_pairs else "Other" for pair in pairs]
    pair_full[mis_indices_full] = np.array(mis_pair_labels, dtype=object)

    # Plot order
    overlay_order = top_pairs.copy()
    if np.any(pair_full == "Other"):
        overlay_order.append("Other")

    # 6) Plot
    fig = plt.figure(figsize=(args.fig_w, args.fig_h), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)

    # Background: all cells in light grey
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        s=args.point_size_bg,
        alpha=args.alpha_bg,
        linewidths=0,
        color="#B0B0B0",
        depthshade=False,
        zorder=1,
    )

    # Overlay misclassified cells by pair (plotted last, on top)
    legend_handles = []

    for i, label in enumerate(overlay_order):
        mask = (pair_full == label)
        if np.any(mask):
            scatt = ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                coords[mask, 2],
                s=args.point_size_fg,
                alpha=args.alpha_fg,
                linewidths=0,
                depthshade=False,
                zorder=10 + i,
                label=shorten(label, 28),
            )

            # build legend handle with same color
            facecolor = scatt.get_facecolor()[0]
            legend_handles.append(
                Line2D(
                    [0], [0],
                    marker="o",
                    linestyle="None",
                    markerfacecolor=facecolor,
                    markeredgecolor="none",
                    markersize=7,
                    label=shorten(label, 28),
                )
            )

    set_common_axes(ax, coords, args.elev, args.azim,
                    "3D Embedding: Misclassified Cells (SANN) — Top true→pred pairs")

    ax.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        title="True → Pred (Top)",
        borderaxespad=1.0,
    )

    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)

    # Sanity: ensure overlay points match misclassified count
    n_overlay = int(np.sum(pair_full != "NA"))
    print(f"[Sanity] Overlay points plotted: {n_overlay} (should equal misclassified test cells)")
    print(f"[Saved] {args.out}")


if __name__ == "__main__":
    main()