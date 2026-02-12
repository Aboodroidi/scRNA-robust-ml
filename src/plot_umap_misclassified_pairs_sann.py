# src/plot_umap_misclassified_pairs_sann.py
import os
import json
import argparse
from collections import Counter

import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description="UMAP plot: misclassified SANN test cells coloured by top true→pred pairs.")
    p.add_argument("--adata", type=str, default="data/processed/pbmc68k_labeled.h5ad")
    p.add_argument("--splits", type=str, default="results/ablations/fixed_splits.json")
    p.add_argument("--true", type=str, default="results/sann_test_true.npy")
    p.add_argument("--pred", type=str, default="results/sann_test_pred.npy")
    p.add_argument("--out", type=str, default="results/figures/umap_misclassified_pairs_sann.png")

    p.add_argument("--umap_key", type=str, default="X_umap")
    p.add_argument("--label_key", type=str, default="cell_type")

    p.add_argument("--top_k", type=int, default=8, help="Number of most common confusion pairs to show.")
    p.add_argument("--point_size_bg", type=float, default=6.0)
    p.add_argument("--point_size_fg", type=float, default=14.0)
    p.add_argument("--alpha_bg", type=float, default=0.20)
    p.add_argument("--alpha_fg", type=float, default=0.95)
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def shorten(s: str, max_len=18) -> str:
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # 1) Load AnnData + check UMAP
    adata = sc.read_h5ad(args.adata)

    if args.umap_key not in adata.obsm:
        raise ValueError(f"UMAP not found: adata.obsm['{args.umap_key}'] missing. Compute UMAP first.")
    umap = adata.obsm[args.umap_key]
    if umap.shape[1] < 2:
        raise ValueError(f"UMAP embedding must have at least 2 dims. Found: {umap.shape}")

    if args.label_key not in adata.obs:
        raise ValueError(f"Label column missing: adata.obs['{args.label_key}'] not found.")

    # Use category order as the class mapping (same as earlier eval scripts)
    class_names = list(adata.obs[args.label_key].astype("category").cat.categories)
    num_classes = len(class_names)

    # 2) Load splits + test_idx
    with open(args.splits, "r") as f:
        splits = json.load(f)
    if "test_idx" not in splits:
        raise ValueError(f"Split file must contain 'test_idx'. Found keys: {list(splits.keys())}")
    test_idx = np.array(splits["test_idx"], dtype=int)

    # 3) Load y_true/y_pred for test only
    y_true = np.load(args.true).astype(int)
    y_pred = np.load(args.pred).astype(int)

    print(f"[Sanity] adata.n_obs: {adata.n_obs}")
    print(f"[Sanity] UMAP shape: {umap.shape}")
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

    # 4) Misclassified mask (test only) + map back to full dataset
    mis_test = (y_true != y_pred)
    mis_full = np.zeros(adata.n_obs, dtype=bool)
    mis_full[test_idx[mis_test]] = True

    n_mis = int(mis_test.sum())
    print(f"[Sanity] Misclassified test cells: {n_mis} / {len(test_idx)}")

    # Build confusion pair strings for misclassified points
    true_labels = [class_names[i] for i in y_true[mis_test]]
    pred_labels = [class_names[i] for i in y_pred[mis_test]]
    pairs = [f"{t} → {p}" for t, p in zip(true_labels, pred_labels)]

    # 5) Top-K pairs, rest -> Other
    counts = Counter(pairs)
    top_pairs = [p for p, _ in counts.most_common(args.top_k)]
    print(f"[Sanity] Top {args.top_k} confusion pairs:")
    for p in top_pairs:
        print(f"  {p}: {counts[p]}")

    # full-length pair label column (default NA)
    pair_full = np.array(["NA"] * adata.n_obs, dtype=object)

    mis_indices_full = test_idx[mis_test]  # absolute indices in adata for misclassified test cells
    mis_pair_labels = []
    for pair in pairs:
        mis_pair_labels.append(pair if pair in top_pairs else "Other")

    pair_full[mis_indices_full] = np.array(mis_pair_labels, dtype=object)

    # 6) Plot (background grey + overlay misclassified by pair)
    x = umap[:, 0]
    y = umap[:, 1]

    fig, ax = plt.subplots(figsize=(8.5, 7.5))

    # Background: all cells in neutral grey
    ax.scatter(
        x, y,
        s=args.point_size_bg,
        alpha=args.alpha_bg,
        linewidths=0,
        color="#B0B0B0",   # fixed light grey
        edgecolors="none",
    )

    # Overlay: misclassified cells, grouped by top pairs + Other
    overlay_order = top_pairs + (["Other"] if "Other" in pair_full else [])
    # But make sure "Other" appears if present
    if np.any(pair_full == "Other") and "Other" not in overlay_order:
        overlay_order.append("Other")

    # Plot each group separately so they get distinct colors automatically
    for label in overlay_order:
        mask = (pair_full == label)
        if np.any(mask):
            ax.scatter(
                x[mask], y[mask],
                s=args.point_size_fg,
                alpha=args.alpha_fg,
                linewidths=0,
                label=shorten(label, 28),
            )

    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title("Misclassified cells on UMAP (SANN) — Top true→pred pairs")
    ax.grid(False)

    # Legend outside to avoid covering the embedding
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        title="True → Pred (Top)",
        markerscale=1.0
    )

    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    # Sanity: ensure misclassified points are shown
    n_overlay = int(np.sum(pair_full != "NA"))
    print(f"[Sanity] Overlay points plotted (should equal misclassified test cells): {n_overlay}")
    print(f"[Sanity] Saved figure: {args.out}")


if __name__ == "__main__":
    main()