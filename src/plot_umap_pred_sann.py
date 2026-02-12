# src/plot_umap_pred_sann.py
import os
import json
import argparse

import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description="Plot UMAP coloured by SANN predicted labels (test cells only).")
    p.add_argument("--adata", type=str, default="data/processed/pbmc68k_labeled.h5ad")
    p.add_argument("--splits", type=str, default="results/ablations/fixed_splits.json")
    p.add_argument("--pred", type=str, default="results/sann_test_pred.npy")
    p.add_argument("--out", type=str, default="results/figures/umap_pred_sann.png")
    p.add_argument("--umap_key", type=str, default="X_umap")
    p.add_argument("--label_key", type=str, default="cell_type")
    p.add_argument("--point_size", type=float, default=6.0)
    p.add_argument("--alpha_bg", type=float, default=0.25)
    p.add_argument("--alpha_fg", type=float, default=0.90)
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def main():
    args = parse_args()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # 1) Load AnnData
    adata = sc.read_h5ad(args.adata)

    if args.umap_key not in adata.obsm:
        raise ValueError(
            f"UMAP not found: adata.obsm['{args.umap_key}'] missing. "
            f"Compute UMAP first and save it into the h5ad."
        )

    umap = adata.obsm[args.umap_key]
    if umap.shape[1] < 2:
        raise ValueError(f"UMAP embedding must have at least 2 dims. Found shape: {umap.shape}")

    # class name mapping from the dataset categories (must match your training order)
    if args.label_key not in adata.obs:
        raise ValueError(f"Label column missing: adata.obs['{args.label_key}'] not found.")

    class_names = list(adata.obs[args.label_key].astype("category").cat.categories)
    num_classes = len(class_names)

    # 2) Load split indices
    with open(args.splits, "r") as f:
        splits = json.load(f)
    if "test_idx" not in splits:
        raise ValueError(f"fixed_splits.json must contain 'test_idx'. Found keys: {list(splits.keys())}")
    test_idx = np.array(splits["test_idx"], dtype=int)

    # 3) Load SANN predictions (class indices)
    y_pred = np.load(args.pred)
    y_pred = np.asarray(y_pred).astype(int)

    # 4) Checks
    print(f"[Sanity] adata.n_obs: {adata.n_obs}")
    print(f"[Sanity] UMAP key: {args.umap_key} | shape: {umap.shape}")
    print(f"[Sanity] #classes: {num_classes}")
    print(f"[Sanity] class names: {class_names}")
    print(f"[Sanity] test_idx length: {len(test_idx)}")
    print(f"[Sanity] sann_test_pred length: {len(y_pred)}")

    if len(test_idx) != len(y_pred):
        raise ValueError(f"Mismatch: len(test_idx)={len(test_idx)} but len(pred)={len(y_pred)}")

    if y_pred.min() < 0 or y_pred.max() >= num_classes:
        raise ValueError(
            f"Pred values out of range. min={y_pred.min()}, max={y_pred.max()}, expected [0, {num_classes-1}]"
        )

    # 5) Create full-length predicted label column
    full_pred = np.array(["NA"] * adata.n_obs, dtype=object)

    pred_labels_test = np.array([class_names[i] for i in y_pred], dtype=object)
    full_pred[test_idx] = pred_labels_test

    adata.obs["sann_pred"] = pd_categorical_from_order(full_pred, class_names)

    # Confirm coloured points count
    n_coloured = np.sum(adata.obs["sann_pred"].astype(str).values != "NA")
    print(f"[Sanity] #coloured points (should equal test_idx): {n_coloured}")

    # 6) Plot UMAP:
    # background (all cells in grey), then overlay test cells colored by predicted class
    x = umap[:, 0]
    y = umap[:, 1]

    fig, ax = plt.subplots(figsize=(8.5, 7.5))

    # background
    ax.scatter(
        x, y,
        s=args.point_size,
        alpha=args.alpha_bg,
        linewidths=0,
    )

    # overlay only test cells by predicted label
    # we plot per class so we can color consistently
    # NOTE: we do NOT specify custom colors; matplotlib will cycle defaults.
    for cname in class_names:
        mask = (adata.obs["sann_pred"].astype(str).values == cname)
        if np.any(mask):
            ax.scatter(
                x[mask], y[mask],
                s=args.point_size,
                alpha=args.alpha_fg,
                linewidths=0,
                label=cname,
            )

    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title("SANN Predicted Labels")
    ax.grid(False)

    # Legend: often huge; put outside
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        title="Predicted",
        markerscale=1.2
    )

    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"[Sanity] Saved figure: {args.out}")


def pd_categorical_from_order(values: np.ndarray, class_names: list):
    """
    Create a categorical that keeps the class order consistent.
    Includes NA as a string category for plotting logic.
    """
    import pandas as pd
    cats = ["NA"] + list(class_names)
    return pd.Categorical(values, categories=cats, ordered=False)


if __name__ == "__main__":
    main()