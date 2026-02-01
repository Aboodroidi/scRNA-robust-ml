# src/plot_marker_dotplot.py
import os
import scanpy as sc
import matplotlib.pyplot as plt

# ----------------------------
# Config
# ----------------------------
IN_PATH = "data/processed/pbmc68k_labeled.h5ad"   # must contain 'leiden'
OUT_DIR = "results/figures"
OUT_PATH = os.path.join(OUT_DIR, "marker_expression_by_cluster_dotplot.png")

# Marker genes (edit/extend as you like)
MARKER_GENES = [
    "CD3D", "CD3E",        # T cells
    "MS4A1",               # B cells
    "NKG7", "GNLY",        # NK cells
    "LST1",                # Monocytes
    "PPBP"                 # Platelets
]

# Columns = clusters
CLUSTER_KEY = "leiden"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    adata = sc.read_h5ad(IN_PATH)

    if CLUSTER_KEY not in adata.obs:
        raise ValueError(f"Expected adata.obs['{CLUSTER_KEY}'] to exist. Run annotate.py first.")

    # Keep only genes that exist in var_names (avoid crashing)
    genes_present = [g for g in MARKER_GENES if g in adata.var_names]
    missing = [g for g in MARKER_GENES if g not in adata.var_names]

    if not genes_present:
        raise ValueError(
            "None of the marker genes were found in adata.var_names. "
            "Check whether your genes are Ensembl IDs or different symbols."
        )

    if missing:
        print("Warning: missing marker genes (not found in var_names):", missing)

    # Dot plot:
    # - color = mean expression
    # - dot size = fraction of cells expressing the gene
    # swap_axes=True -> rows=genes, cols=clusters (as requested)
    sc.pl.dotplot(
        adata,
        var_names=genes_present,
        groupby=CLUSTER_KEY,
        standard_scale="var",   # scale each gene across clusters (cleaner visual comparison)
        swap_axes=True,
        show=False
    )

    plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()