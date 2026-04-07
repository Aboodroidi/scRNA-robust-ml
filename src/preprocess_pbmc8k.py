# src/preprocess_pbmc8k.py
"""
Preprocess the PBMC 8K external validation dataset.

Key difference from the 68K preprocessing:
  - We do NOT select new HVGs. Instead we align to the same HVG set
    used by the 68K training data so the trained models can be applied directly.
  - Cell type annotation uses the same marker-based approach.
"""
import os

# Must be set BEFORE importing numpy/scanpy/sklearn to avoid macOS segfaults
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["MPLBACKEND"] = "Agg"

import argparse
import numpy as np
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Same marker dictionary used in 68K annotation
PBMC_MARKERS = {
    "T cells": ["CD3D", "CD3E", "TRAC", "IL7R", "CCR7"],
    "CD8 T":   ["CD8A", "CD8B", "GZMK", "GZMB"],
    "NK":      ["NKG7", "GNLY", "PRF1", "KLRD1"],
    "B cells": ["MS4A1", "CD79A", "CD79B", "CD74"],
    "Plasma":  ["MZB1", "SDC1", "JCHAIN", "XBP1"],
    "Mono":    ["LYZ", "S100A8", "S100A9", "FCGR3A", "MS4A7"],
    "DC":      ["FCER1A", "CLEC10A", "ITGAX"],
    "Platelet": ["PPBP", "PF4", "NRGN"],
}


def annotate_clusters(adata):
    """Assign cell type labels using marker gene scoring (same method as 68K)."""
    # Cluster
    sc.tl.leiden(adata, key_added="leiden", resolution=0.5)

    # Rank genes per cluster
    sc.tl.rank_genes_groups(adata, "leiden", method="wilcoxon")
    rg = sc.get.rank_genes_groups_df(adata, None)

    cluster_labels = {}
    for cl in sorted(adata.obs["leiden"].unique(), key=lambda x: int(x)):
        sub = rg[rg["group"] == cl].head(100)
        tops = set(sub["names"].str.upper())

        score_per_type = {}
        for ctype, genes in PBMC_MARKERS.items():
            score_per_type[ctype] = len(tops.intersection({g.upper() for g in genes}))

        best = max(score_per_type, key=score_per_type.get)
        cluster_labels[cl] = best if score_per_type[best] > 0 else f"CL {cl}"

    adata.obs["cell_type"] = adata.obs["leiden"].map(cluster_labels).astype("category")
    print(f"  Cluster → label map: {cluster_labels}")
    return adata


def main():
    p = argparse.ArgumentParser(description="Preprocess PBMC 8K for external validation")
    p.add_argument("--in_dir", default="data/raw/pbmc8k/filtered_gene_bc_matrices/GRCh38")
    p.add_argument("--ref", default="data/processed/pbmc68k_labeled.h5ad",
                   help="68K reference to extract HVG gene list")
    p.add_argument("--out", default="data/processed/pbmc8k_labeled.h5ad")
    p.add_argument("--plot_dir", default="data/processed")
    p.add_argument("--mt_pct", type=float, default=10.0)
    p.add_argument("--n_pcs", type=int, default=50)
    p.add_argument("--n_neighbors", type=int, default=15)
    args = p.parse_args()

    sc.settings.verbosity = 2
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)

    # ----------------------------------------------------------------
    # 1. Load 8K raw data
    # ----------------------------------------------------------------
    print("Loading PBMC 8K raw data...")
    adata = sc.read_10x_mtx(args.in_dir, var_names="gene_symbols", make_unique=True)
    print(f"  Raw: {adata.shape[0]} cells x {adata.shape[1]} genes")

    # ----------------------------------------------------------------
    # 2. QC filtering (same criteria as 68K)
    #    Manual implementation to avoid scanpy segfault on macOS
    # ----------------------------------------------------------------
    print("QC filtering...")
    import scipy.sparse as sp

    X = adata.X
    if sp.issparse(X):
        genes_per_cell = np.array((X > 0).sum(axis=1)).flatten()
        cells_per_gene = np.array((X > 0).sum(axis=0)).flatten()
        total_counts = np.array(X.sum(axis=1)).flatten()
    else:
        genes_per_cell = (X > 0).sum(axis=1)
        cells_per_gene = (X > 0).sum(axis=0)
        total_counts = X.sum(axis=1)

    # Mitochondrial percentage
    mt_mask = adata.var_names.str.upper().str.startswith("MT-")
    if sp.issparse(X):
        mt_counts = np.array(X[:, mt_mask].sum(axis=1)).flatten()
    else:
        mt_counts = X[:, mt_mask].sum(axis=1)
    pct_mt = mt_counts / (total_counts + 1e-8) * 100

    # Filter cells: min 200 genes, < mt_pct% mitochondrial
    cell_mask = (genes_per_cell >= 200) & (pct_mt < args.mt_pct)
    adata = adata[cell_mask].copy()

    # Filter genes: min 3 cells
    if sp.issparse(adata.X):
        cells_per_gene2 = np.array((adata.X > 0).sum(axis=0)).flatten()
    else:
        cells_per_gene2 = (adata.X > 0).sum(axis=0)
    gene_mask = cells_per_gene2 >= 3
    adata = adata[:, gene_mask].copy()

    print(f"  After QC: {adata.shape[0]} cells x {adata.shape[1]} genes")

    # ----------------------------------------------------------------
    # 3. Normalize + log (same as 68K)
    # ----------------------------------------------------------------
    print("Normalizing...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # ----------------------------------------------------------------
    # 4. Align to 68K HVG gene set (do NOT select new HVGs)
    # ----------------------------------------------------------------
    print("Loading 68K reference for HVG alignment...")
    ref = sc.read_h5ad(args.ref)
    hvg_genes = list(ref.var_names)  # these are already the HVG-subset genes
    n_ref_genes = len(hvg_genes)

    # Find overlap
    shared = [g for g in hvg_genes if g in adata.var_names]
    missing = [g for g in hvg_genes if g not in adata.var_names]
    print(f"  68K HVGs: {n_ref_genes} | shared with 8K: {len(shared)} | missing: {len(missing)}")

    if len(missing) > 0 and len(missing) <= 50:
        print(f"  Missing genes (will be zero-filled): {missing[:20]}...")
    elif len(missing) > 50:
        print(f"  WARNING: {len(missing)} genes missing — first 20: {missing[:20]}")

    # Subset to shared genes first
    adata_aligned = adata[:, shared].copy()

    # Add zero columns for missing genes (if any)
    if missing:
        import scipy.sparse as sp
        n_cells = adata_aligned.shape[0]
        zero_block = sp.csr_matrix((n_cells, len(missing)), dtype=np.float32)

        from anndata import AnnData
        zero_adata = AnnData(
            X=zero_block,
            obs=adata_aligned.obs.copy(),
            var=sc.AnnData(np.zeros((len(missing), 0))).obs,  # placeholder
        )
        zero_adata.var_names = missing

        import anndata
        adata_aligned = anndata.concat([adata_aligned, zero_adata], axis=1)

    # Reorder columns to match 68K gene order exactly
    adata_aligned = adata_aligned[:, hvg_genes].copy()

    # Mark all as highly variable (they are the HVG set)
    adata_aligned.var["highly_variable"] = True

    print(f"  Aligned shape: {adata_aligned.shape}")

    # ----------------------------------------------------------------
    # 5. Scale, PCA, neighbors, UMAP (same params as 68K)
    # ----------------------------------------------------------------
    print("Scaling + PCA + UMAP...")
    # Manual scale to avoid scanpy segfault on macOS during densification
    import scipy.sparse as sp_mod
    if sp_mod.issparse(adata_aligned.X):
        X_dense = adata_aligned.X.toarray().astype(np.float32)
    else:
        X_dense = np.asarray(adata_aligned.X, dtype=np.float32)

    mean = X_dense.mean(axis=0)
    std = X_dense.std(axis=0)
    std[std == 0] = 1.0
    X_scaled = (X_dense - mean) / std
    X_scaled = np.clip(X_scaled, -10, 10)
    adata_aligned.X = X_scaled

    sc.tl.pca(adata_aligned, n_comps=args.n_pcs)
    # Use sklearn's NearestNeighbors directly to avoid multiprocessing segfault
    from sklearn.neighbors import NearestNeighbors
    from scipy.sparse import csr_matrix

    pca_data = adata_aligned.obsm["X_pca"][:, :args.n_pcs]
    nn = NearestNeighbors(n_neighbors=args.n_neighbors, metric="euclidean", n_jobs=1)
    nn.fit(pca_data)
    distances, indices = nn.kneighbors(pca_data)

    # Build connectivity and distance matrices for scanpy
    n_cells = pca_data.shape[0]
    rows = np.repeat(np.arange(n_cells), args.n_neighbors)
    cols = indices.flatten()
    dist_vals = distances.flatten()

    dist_matrix = csr_matrix((dist_vals, (rows, cols)), shape=(n_cells, n_cells))
    conn_vals = np.ones(len(dist_vals), dtype=np.float32)
    conn_matrix = csr_matrix((conn_vals, (rows, cols)), shape=(n_cells, n_cells))

    adata_aligned.obsp["distances"] = dist_matrix
    adata_aligned.obsp["connectivities"] = conn_matrix
    adata_aligned.uns["neighbors"] = {
        "connectivities_key": "connectivities",
        "distances_key": "distances",
        "params": {"n_neighbors": args.n_neighbors, "method": "custom"},
    }

    sc.tl.umap(adata_aligned)

    # ----------------------------------------------------------------
    # 6. Annotate cell types (same marker-based approach)
    # ----------------------------------------------------------------
    print("Annotating cell types...")
    adata_aligned = annotate_clusters(adata_aligned)

    # ----------------------------------------------------------------
    # 7. Save
    # ----------------------------------------------------------------
    adata_aligned.write(args.out)
    print(f"\nSaved: {args.out}")
    print(adata_aligned)
    print(f"\nCell type distribution:")
    print(adata_aligned.obs["cell_type"].value_counts())

    # Save UMAP plot
    sc.pl.umap(
        adata_aligned,
        color=["cell_type"],
        legend_loc="on data",
        legend_fontsize=7,
        show=False,
    )
    umap_path = os.path.join(args.plot_dir, "pbmc8k_umap_by_celltype.png")
    plt.savefig(umap_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved UMAP: {umap_path}")

    # Save cluster-label counts
    counts = adata_aligned.obs.groupby(["leiden", "cell_type"]).size().unstack(fill_value=0)
    counts_path = os.path.join(args.plot_dir, "pbmc8k_cluster_to_label_counts.csv")
    counts.to_csv(counts_path)
    print(f"Saved counts: {counts_path}")


if __name__ == "__main__":
    main()
