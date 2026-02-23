import os
import argparse

import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib as mpl


def parse_args():
    p = argparse.ArgumentParser(description="Plot UMAP coloured by TRUE labels using SAME palette + layout as SANN plot.")
    p.add_argument("--adata", type=str, default="data/processed/pbmc68k_labeled.h5ad")
    p.add_argument("--out", type=str, default="results/figures/umap_true_labels.png")
    p.add_argument("--umap_key", type=str, default="X_umap")
    p.add_argument("--label_key", type=str, default="cell_type")

    # style
    p.add_argument("--point_size", type=float, default=6.0)
    p.add_argument("--alpha_bg", type=float, default=0.20)
    p.add_argument("--alpha_fg", type=float, default=0.90)
    p.add_argument("--dpi", type=int, default=300)

    # palette
    p.add_argument("--palette", type=str, default="tab20",
                   help="Matplotlib categorical cmap (e.g. tab20, tab20b, tab20c).")

    # legend appearance
    p.add_argument("--legend_marker_size", type=float, default=70.0)  # bigger swatches
    p.add_argument("--legend_markerscale", type=float, default=2.6)
    return p.parse_args()


def build_palette(class_names, cmap_name="tab20"):
    """
    Deterministic mapping: class_name -> RGBA color.
    Uses modern matplotlib colormaps API (no deprecation warning).
    """
    cmap = mpl.colormaps.get_cmap(cmap_name).resampled(len(class_names))
    return {c: cmap(i) for i, c in enumerate(class_names)}


def set_legend_marker_style(legend, marker_size=70.0):
    """
    Matplotlib-version-safe way to edit legend marker size/alpha.
    """
    handles = getattr(legend, "legendHandles", None)
    if handles is None:
        handles = getattr(legend, "legend_handles", None)
    if handles is None:
        # fallback
        try:
            handles = legend.legend_handles
        except Exception:
            handles = []

    for h in handles:
        try:
            h.set_alpha(1.0)
        except Exception:
            pass
        try:
            # PathCollection from scatter
            h.set_sizes([marker_size])
        except Exception:
            pass


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    adata = sc.read_h5ad(args.adata)

    if args.umap_key not in adata.obsm:
        raise ValueError(f"UMAP not found: adata.obsm['{args.umap_key}'] missing.")
    if args.label_key not in adata.obs:
        raise ValueError(f"Label column missing: adata.obs['{args.label_key}'] not found.")

    umap = adata.obsm[args.umap_key]
    if umap.shape[1] < 2:
        raise ValueError(f"UMAP embedding must have at least 2 dims. Found shape: {umap.shape}")

    x = umap[:, 0]
    y = umap[:, 1]

    # lock category order
    y_cat = adata.obs[args.label_key].astype("category")
    class_names = list(y_cat.cat.categories)
    color_map = build_palette(class_names, args.palette)

    print(f"[Sanity] adata.n_obs: {adata.n_obs}")
    print(f"[Sanity] #classes: {len(class_names)}")
    print(f"[Sanity] class names: {class_names}")

    fig, ax = plt.subplots(figsize=(8.5, 7.5))

    # background GREY
    ax.scatter(
        x, y,
        s=args.point_size,
        alpha=args.alpha_bg,
        linewidths=0,
        c="lightgrey",
    )

    # overlay TRUE labels with fixed colors
    y_true = y_cat.astype(str).values
    for cname in class_names:
        mask = (y_true == cname)
        if np.any(mask):
            ax.scatter(
                x[mask], y[mask],
                s=args.point_size,
                alpha=args.alpha_fg,
                linewidths=0,
                color=color_map[cname],
                label=cname,
            )

    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title("True Labels")
    ax.grid(False)

    legend = ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        title="True",
        markerscale=args.legend_markerscale,
        scatterpoints=1,
        fontsize=11,
        title_fontsize=12,
    )
    set_legend_marker_style(legend, marker_size=args.legend_marker_size)

    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"[Saved] {args.out}")


if __name__ == "__main__":
    main()