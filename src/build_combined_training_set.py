#!/usr/bin/env python
"""
Build a combined 68K + 8K training dataset with consistent coarse labels.

Both datasets are z-scored using 68K's per-gene statistics so they share
the same feature space. The 8K labels are mapped to coarse types:
  CD8 T → T cells, B cells → B cells, Mono → Mono, NK → NK
  CL 1, DC → excluded (not in coarse label set)

Outputs:
  data/processed/combined_68k_8k.h5ad
    - X = z-scored HVG expression (2000 genes, 68K stats)
    - obs['cell_type_coarse'] = coarse labels
    - obs['dataset'] = '68k' or '8k' (for stratified splitting)
    - layers['log_norm'] = log-normalised expression
"""
import os, sys, warnings
import numpy as np
import scipy.sparse as sp

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import scanpy as sc
import anndata

warnings.filterwarnings("ignore")


def main():
    out_dir = "data/processed"
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load 68K (already preprocessed with coarse labels)
    # ------------------------------------------------------------------
    print("Loading PBMC 68K...")
    adata_68k = sc.read_h5ad("data/processed/pbmc68k_labeled.h5ad")
    print(f"  68K: {adata_68k.shape[0]} cells × {adata_68k.shape[1]} genes")

    hvg_genes = list(adata_68k.var_names)
    ref_mean = adata_68k.var["mean"].values.astype(np.float32)
    ref_std = adata_68k.var["std"].values.astype(np.float32)
    ref_std[ref_std == 0] = 1.0

    # 68K X is already z-scored with its own stats
    X_68k = adata_68k.X
    if sp.issparse(X_68k):
        X_68k = X_68k.toarray()
    X_68k = X_68k.astype(np.float32)

    labels_68k = adata_68k.obs["cell_type_coarse"].values.astype(str)
    print(f"  68K coarse labels: {dict(zip(*np.unique(labels_68k, return_counts=True)))}")

    # ------------------------------------------------------------------
    # 2. Load 8K raw → CPM → log1p → align → z-score with 68K stats
    # ------------------------------------------------------------------
    print("\nLoading PBMC 8K raw data...")
    raw_dir = "data/raw/pbmc8k/filtered_gene_bc_matrices/GRCh38"
    adata_8k_raw = sc.read_10x_mtx(raw_dir, var_names="gene_symbols", make_unique=True)

    # QC filter (same as preprocess_pbmc8k.py)
    X = adata_8k_raw.X
    genes_per_cell = np.array((X > 0).sum(axis=1)).flatten()
    total_counts = np.array(X.sum(axis=1)).flatten()
    keep = (genes_per_cell >= 200) & (genes_per_cell <= 5000) & (total_counts < 20000)
    adata_8k_raw = adata_8k_raw[keep].copy()
    print(f"  8K after QC: {adata_8k_raw.shape[0]} cells")

    # Normalise
    sc.pp.normalize_total(adata_8k_raw, target_sum=1e4)
    sc.pp.log1p(adata_8k_raw)

    # Align to 68K HVG genes
    gene_to_idx = {g: i for i, g in enumerate(adata_8k_raw.var_names)}
    X_8k_lognorm = adata_8k_raw.X
    if sp.issparse(X_8k_lognorm):
        X_8k_lognorm = X_8k_lognorm.toarray()

    n_8k = adata_8k_raw.shape[0]
    X_8k_aligned = np.zeros((n_8k, len(hvg_genes)), dtype=np.float32)
    shared = 0
    for j, g in enumerate(hvg_genes):
        if g in gene_to_idx:
            X_8k_aligned[:, j] = X_8k_lognorm[:, gene_to_idx[g]]
            shared += 1
    print(f"  Gene alignment: {shared}/{len(hvg_genes)} shared")

    # Z-score with 68K statistics
    X_8k_zscored = np.clip((X_8k_aligned - ref_mean) / ref_std, -10, 10).astype(np.float32)
    print(f"  8K z-scored: mean={X_8k_zscored.mean():.4f}, std={X_8k_zscored.std():.4f}")

    # ------------------------------------------------------------------
    # 3. Load 8K labels and map to coarse types
    # ------------------------------------------------------------------
    adata_8k_labeled = sc.read_h5ad("data/processed/pbmc8k_labeled.h5ad")

    # Match barcodes between raw (QC'd) and labeled
    raw_barcodes = list(adata_8k_raw.obs_names)
    labeled_barcodes = set(adata_8k_labeled.obs_names)

    # Map 8K labels
    ALIAS_8K = {"CD8 T": "T cells"}
    VALID_COARSE = {"T cells", "NK", "B cells", "Mono", "Platelet"}

    labels_8k = []
    keep_mask = []
    label_series = adata_8k_labeled.obs["cell_type"]

    for bc in raw_barcodes:
        if bc in labeled_barcodes:
            raw_label = str(label_series[bc])
            coarse = ALIAS_8K.get(raw_label, raw_label)
            if coarse in VALID_COARSE:
                labels_8k.append(coarse)
                keep_mask.append(True)
            else:
                keep_mask.append(False)
        else:
            keep_mask.append(False)

    keep_mask = np.array(keep_mask)
    X_8k_zscored = X_8k_zscored[keep_mask]
    X_8k_aligned_kept = X_8k_aligned[keep_mask]  # log-normalised for mask
    labels_8k = np.array(labels_8k)

    print(f"  8K with valid coarse labels: {len(labels_8k)}")
    print(f"  8K coarse labels: {dict(zip(*np.unique(labels_8k, return_counts=True)))}")

    # ------------------------------------------------------------------
    # 4. Combine into single AnnData
    # ------------------------------------------------------------------
    print("\nCombining 68K + 8K...")

    X_combined = np.vstack([X_68k, X_8k_zscored])
    labels_combined = np.concatenate([labels_68k, labels_8k])
    dataset_combined = np.array(
        ["68k"] * X_68k.shape[0] + ["8k"] * X_8k_zscored.shape[0]
    )

    # Store log-normalised for sparsity mask computation
    # 68K log-norm: un-z-score using 68K stats
    X_68k_lognorm = X_68k * ref_std + ref_mean

    X_lognorm_combined = np.vstack([X_68k_lognorm, X_8k_aligned_kept])

    adata_combined = anndata.AnnData(
        X=X_combined.astype(np.float32),
        var=adata_68k.var[[]].copy(),  # just gene names
    )
    adata_combined.var_names = hvg_genes
    adata_combined.obs["cell_type_coarse"] = labels_combined
    adata_combined.obs["cell_type_coarse"] = adata_combined.obs["cell_type_coarse"].astype("category")
    adata_combined.obs["dataset"] = dataset_combined
    adata_combined.layers["log_norm"] = X_lognorm_combined.astype(np.float32)

    # Store 68K scaling stats in var for downstream use
    adata_combined.var["mean"] = ref_mean
    adata_combined.var["std"] = ref_std

    # Mark all genes as HVG (they already are — the 2000 HVG set from 68K)
    adata_combined.var["highly_variable"] = True

    # Compute PCA on the combined z-scored expression
    print("  Computing PCA (50 components)...")
    from sklearn.decomposition import PCA
    pca = PCA(n_components=50, random_state=42)
    X_pca = pca.fit_transform(X_combined)
    adata_combined.obsm["X_pca"] = X_pca.astype(np.float32)
    print(f"  PCA variance explained: {pca.explained_variance_ratio_.sum():.3f}")

    # Save PCA model for projecting new data (e.g. 3K external test)
    import joblib
    pca_path = os.path.join(out_dir, "combined_expr_pca_model.pkl")
    joblib.dump(pca, pca_path)
    print(f"  Saved expression PCA model: {pca_path}")

    print(f"  Combined: {adata_combined.shape[0]} cells × {adata_combined.shape[1]} genes")
    print(f"  Labels: {dict(adata_combined.obs['cell_type_coarse'].value_counts())}")
    print(f"  Datasets: {dict(zip(*np.unique(dataset_combined, return_counts=True)))}")

    out_path = os.path.join(out_dir, "combined_68k_8k.h5ad")
    adata_combined.write(out_path)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
