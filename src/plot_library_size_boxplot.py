import os
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt


# ----------------------------
# Paths
# ----------------------------
RAW_MTX_PATH = "data/raw/pbmc68k/filtered_matrices_mex/hg19"
NORM_PATH = "data/processed/pbmc68k_prepped.h5ad"

OUT_DIR = "results/figures"
OUT_PATH = os.path.join(OUT_DIR, "library_size_before_after_normalisation.png")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ----------------------------
    # Load raw counts (BEFORE normalisation)
    # ----------------------------
    adata_raw = sc.read_10x_mtx(
        RAW_MTX_PATH,
        var_names="gene_symbols",
        cache=True
    )

    # ----------------------------
    # Load processed data (AFTER normalisation)
    # ----------------------------
    adata_norm = sc.read_h5ad(NORM_PATH)

    # ----------------------------
    # Compute total counts per cell
    # ----------------------------
    total_counts_before = np.array(adata_raw.X.sum(axis=1)).flatten()
    total_counts_after = np.array(adata_norm.X.sum(axis=1)).flatten()

    # ----------------------------
    # Boxplot
    # ----------------------------
    fig, ax = plt.subplots(figsize=(6, 5))

    ax.boxplot(
        [total_counts_before, total_counts_after],
        labels=["Before normalisation", "After normalisation"],
        patch_artist=True,
        boxprops=dict(facecolor="#4C72B0"),
        medianprops=dict(color="black"),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
        flierprops=dict(marker=".", markersize=3, alpha=0.4),
    )

    ax.set_ylabel("Total counts per cell")
    ax.set_title(
        "Distribution of Total Counts per Cell\nBefore vs After Normalisation"
    )

    # Log scale is essential for library sizes
    ax.set_yscale("log")
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()