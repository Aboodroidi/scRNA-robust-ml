import scanpy as sc
import numpy as np
import pandas as pd
import os

IN_PATH  = "data/processed/pbmc68k_prepped.h5ad"
OUT_PATH = "data/processed/pbmc68k_labeled.h5ad"
PLOT_DIR = "data/processed"

# quick PBMC marker dictionary (you’ll refine)
# keys = friendly label; values = list of known marker genes
PBMC_MARKERS = {
    "T cells": ["CD3D","CD3E","TRAC","IL7R","CCR7"],
    "CD8 T":  ["CD8A","CD8B","GZMK","GZMB"],
    "NK":     ["NKG7","GNLY","PRF1","KLRD1"],
    "B cells":["MS4A1","CD79A","CD79B","CD74"],
    "Plasma": ["MZB1","SDC1","JCHAIN","XBP1"],
    "Mono":   ["LYZ","S100A8","S100A9","FCGR3A","MS4A7"],
    "DC":     ["FCER1A","CLEC10A","ITGAX"],
    "Platelet":["PPBP","PF4","NRGN"]
}

def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    sc.settings.verbosity = 2

    adata = sc.read(IN_PATH)

    # 1) cluster (on existing PCA graph from preprocess)
    sc.tl.leiden(adata, key_added="leiden", resolution=0.5)
    sc.pl.umap(adata, color=["leiden"], legend_loc="on data", show=False,
               save=None)
    sc.pl.umap(adata, color=["leiden"], legend_loc="on data", show=False)
    adata.write(OUT_PATH)  # interim save

    # 2) rank marker genes per cluster
    sc.tl.rank_genes_groups(adata, "leiden", method="wilcoxon")
    sc.pl.rank_genes_groups(adata, n_genes=20, sharey=False, show=False)
    import matplotlib.pyplot as plt
    plt.savefig(f"{PLOT_DIR}/pbmc68k_rank_genes_by_cluster.png", dpi=200, bbox_inches="tight")

    # 3) rough automatic label suggestion:
    # for each cluster, check which PBMC_MARKERS group scores highest among top ranked genes
    rg = sc.get.rank_genes_groups_df(adata, None)
    cluster_labels = {}
    for cl in sorted(adata.obs["leiden"].unique(), key=lambda x:int(x)):
        sub = rg[rg["group"]==cl].head(100)  # top 100 ranked for that cluster
        score_per_type = {ctype:0 for ctype in PBMC_MARKERS}
        tops = set(sub["names"].str.upper())
        for ctype, genes in PBMC_MARKERS.items():
            score_per_type[ctype] = len(tops.intersection({g.upper() for g in genes}))
        best = max(score_per_type, key=score_per_type.get)
        cluster_labels[cl] = best if score_per_type[best] > 0 else f"Cluster {cl}"

    # 4) attach labels and save coloured UMAP
    adata.obs["cell_type"] = adata.obs["leiden"].map(cluster_labels).astype("category")
    adata.write(OUT_PATH)

    sc.pl.umap(adata, color=["cell_type"], legend_loc="on data", show=False)
    plt.savefig(f"{PLOT_DIR}/pbmc68k_umap_by_celltype.png", dpi=200, bbox_inches="tight")

    # also save a confusion-style table: cluster -> assigned label counts
    counts = adata.obs.groupby(["leiden","cell_type"]).size().unstack(fill_value=0)
    counts.to_csv(f"{PLOT_DIR}/cluster_to_label_counts.csv")
    print("Saved:")
    print(f" - {OUT_PATH}")
    print(f" - {PLOT_DIR}/pbmc68k_umap_by_celltype.png")
    print(f" - {PLOT_DIR}/pbmc68k_rank_genes_by_cluster.png")
    print(f" - {PLOT_DIR}/cluster_to_label_counts.csv")
    print("Initial cluster→label map:", cluster_labels)

if __name__ == "__main__":
    main()