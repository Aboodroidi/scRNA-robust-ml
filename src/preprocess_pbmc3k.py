#!/usr/bin/env python
"""
Preprocess PBMC 3K dataset (scanpy built-in) for external validation.

Steps:
  1. QC filtering (same thresholds as 68K/8K)
  2. CPM normalisation → log1p
  3. Cell type annotation via canonical marker genes
  4. Z-scoring (manual, to avoid macOS segfault)
  5. Gene alignment to 68K HVG set
  6. Save processed h5ad

Cell type annotation uses the standard scanpy PBMC 3K tutorial markers:
  - CD4 T / CD8 T / NK / B cells / Mono (CD14+, FCGR3A+) / DC / Platelet
  - We map to coarse labels matching the 68K training: T cells, NK, B cells, Mono, Platelet
"""
import os, sys, warnings
import numpy as np
import scipy.sparse as sp

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import scanpy as sc

warnings.filterwarnings("ignore")


def main():
    out_dir = "data/processed"
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load raw data
    # ------------------------------------------------------------------
    print("Loading PBMC 3K raw data...")
    adata = sc.read_h5ad("data/raw/pbmc3k/pbmc3k_raw.h5ad")
    print(f"  Raw: {adata.shape[0]} cells × {adata.shape[1]} genes")

    # Make var_names unique
    adata.var_names_make_unique()

    # ------------------------------------------------------------------
    # 2. QC filtering
    # ------------------------------------------------------------------
    print("QC filtering...")
    X = adata.X
    if sp.issparse(X):
        genes_per_cell = np.array((X > 0).sum(axis=1)).flatten()
        total_counts = np.array(X.sum(axis=1)).flatten()
    else:
        genes_per_cell = (X > 0).sum(axis=1)
        total_counts = X.sum(axis=1)

    # Mitochondrial fraction
    mito_genes = adata.var_names.str.startswith("MT-")
    if sp.issparse(X):
        mito_counts = np.array(X[:, mito_genes].sum(axis=1)).flatten()
    else:
        mito_counts = X[:, mito_genes].sum(axis=1)
    pct_mito = mito_counts / (total_counts + 1e-8) * 100

    keep = (genes_per_cell >= 200) & (genes_per_cell <= 5000) & (pct_mito < 5)
    adata = adata[keep].copy()
    print(f"  After QC: {adata.shape[0]} cells")

    # ------------------------------------------------------------------
    # 3. Normalise
    # ------------------------------------------------------------------
    print("Normalising (CPM + log1p)...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Keep a copy of log-normalised expression for later
    adata.layers["log_norm"] = adata.X.copy()

    # ------------------------------------------------------------------
    # 4. Annotate cell types using scanpy's reference-annotated PBMC 3K
    # ------------------------------------------------------------------
    print("Loading scanpy reference annotations (pbmc3k_processed)...")
    adata_ref = sc.datasets.pbmc3k_processed()

    # Map scanpy louvain labels → coarse types matching 68K training
    LOUVAIN_TO_COARSE = {
        "CD4 T cells": "T cells",
        "CD8 T cells": "T cells",
        "NK cells": "NK",
        "B cells": "B cells",
        "CD14+ Monocytes": "Mono",
        "FCGR3A+ Monocytes": "Mono",
        "Dendritic cells": "DC",
        "Megakaryocytes": "Platelet",
    }

    # Transfer labels by matching cell barcodes
    ref_labels = adata_ref.obs["louvain"].map(LOUVAIN_TO_COARSE)
    shared_barcodes = adata.obs_names.intersection(adata_ref.obs_names)
    print(f"  Shared barcodes: {len(shared_barcodes)} / {adata.shape[0]} QC-passed cells")

    # Keep only cells with reference annotations
    adata = adata[shared_barcodes].copy()
    adata.obs["cell_type"] = ref_labels.loc[shared_barcodes].values
    adata.obs["cell_type"] = adata.obs["cell_type"].astype("category")
    adata.obs["cell_type_coarse"] = adata.obs["cell_type"].copy()

    print(f"\nCell type distribution (reference-annotated):")
    print(adata.obs["cell_type_coarse"].value_counts().to_string())

    # ------------------------------------------------------------------
    # 6. Z-score the full expression matrix (using 3K's own stats, stored for reference)
    # ------------------------------------------------------------------
    print("\nZ-scoring full expression matrix...")
    X_full = adata.layers["log_norm"]
    if sp.issparse(X_full):
        X_full = X_full.toarray()
    full_mean = X_full.mean(axis=0)
    full_std = X_full.std(axis=0)
    full_std[full_std == 0] = 1.0

    adata.var["mean"] = full_mean
    adata.var["std"] = full_std

    X_zscored = np.clip((X_full - full_mean) / full_std, -10, 10).astype(np.float32)
    adata.X = X_zscored

    # ------------------------------------------------------------------
    # 7. Save
    # ------------------------------------------------------------------
    out_path = os.path.join(out_dir, "pbmc3k_labeled.h5ad")
    adata.write(out_path)
    print(f"\nSaved: {out_path}")
    print(f"  Shape: {adata.shape}")
    print(f"  Cell types: {dict(adata.obs['cell_type'].value_counts())}")


if __name__ == "__main__":
    main()
