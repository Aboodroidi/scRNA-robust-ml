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
os.environ["PYTHONUTF8"] = "1"
os.environ["LC_ALL"] = "en_US.UTF-8"
os.environ["LANG"] = "en_US.UTF-8"


def parse_args():
    p = argparse.ArgumentParser(description="Plot misclassified cells in 3D")

    p.add_argument("--adata", default="data/processed/pbmc68k_labeled.h5ad")
    p.add_argument("--label_key", default="cell_type")
    p.add_argument("--pred", default="results/full_train/sann_all_pred.npy")

    p.add_argument("--umap_key", default="X_umap")
    p.add_argument("--pca_key", default="X_pca")

    p.add_argument("--out", default="results/figures/misclassified_3d.png")

    p.add_argument("--fig_w", type=float, default=11)
    p.add_argument("--fig_h", type=float, default=8)

    p.add_argument("--point_size", type=float, default=4)

    p.add_argument("--elev", type=float, default=24)
    p.add_argument("--azim", type=float, default=42)

    return p.parse_args()


def get_3d_coords(adata, umap_key, pca_key):

    umap = adata.obsm[umap_key]
    pca = adata.obsm[pca_key]

    x = umap[:, 0]
    y = umap[:, 1]
    z = pca[:, 0]

    z = (z - z.mean()) / (z.std() + 1e-8)

    return np.column_stack([x, y, z])


def main():

    args = parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    adata = sc.read_h5ad(args.adata)

    coords = get_3d_coords(
        adata,
        args.umap_key,
        args.pca_key
    )

    true = adata.obs[args.label_key].astype("category")

    y_true = true.cat.codes.values
    y_pred = np.load(args.pred)

    if len(y_true) != len(y_pred):
        raise ValueError(f"Prediction length mismatch: true={len(y_true)} pred={len(y_pred)}")

    correct = y_true == y_pred
    wrong = ~correct

    n_correct = int(correct.sum())
    n_wrong = int(wrong.sum())

    print("Total cells:", len(y_true))
    print("Correctly classified:", n_correct)
    print("Misclassified:", n_wrong)

    fig = plt.figure(figsize=(args.fig_w, args.fig_h))

    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)

    # background
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        s=args.point_size,
        c="lightgrey",
        alpha=0.04,
        depthshade=False,
        zorder=1
    )

    # correct classifications
    ax.scatter(
        coords[correct, 0],
        coords[correct, 1],
        coords[correct, 2],
        s=1,
        c="steelblue",
        alpha=0.22,
        depthshade=False,
        zorder=2
    )

    # misclassifications (on top)
    ax.scatter(
        coords[wrong, 0],
        coords[wrong, 1],
        coords[wrong, 2],
        s=1,
        c="red",
        alpha=1.0,
        depthshade=False,
        zorder=100
    )

    ax.set_xlabel("UMAP1", labelpad=12)
    ax.set_ylabel("UMAP2", labelpad=12)
    ax.set_zlabel("PC1", labelpad=12)

    ax.set_title("3D Embedding: Misclassified Cells (SANN)", pad=18)

    ax.view_init(elev=args.elev, azim=args.azim)

    # legend with counts
    legend_elements = [
        Line2D([0], [0], marker="o", color="w",
               label=f"Correct classification ({n_correct})",
               markerfacecolor="steelblue", markeredgecolor="none", markersize=8),

        Line2D([0], [0], marker="o", color="w",
               label=f"Misclassification ({n_wrong})",
               markerfacecolor="red", markeredgecolor="none", markersize=8),
    ]

    ax.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False
    )

    plt.tight_layout()

    plt.savefig(args.out, dpi=300, bbox_inches="tight", pad_inches=0.2)

    print("Saved:", args.out)


if __name__ == "__main__":
    main()