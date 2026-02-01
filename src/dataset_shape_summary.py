import scanpy as sc
import pandas as pd
import os

# ----------------------------
# Config
# ----------------------------
IN_PATH = "data/processed/pbmc68k_labeled.h5ad"
OUT_DIR = "results/tables"
OUT_PATH = os.path.join(OUT_DIR, "dataset_shape_summary.csv")

# If you used PCA, this is usually the representation name
PCA_KEY = "X_pca"
LABEL_KEY = "cell_type"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    adata = sc.read_h5ad(IN_PATH)

    # Number of cells
    n_cells = adata.n_obs

    # Number of features
    # If PCA exists, report number of PCA components
    if PCA_KEY in adata.obsm:
        n_features = adata.obsm[PCA_KEY].shape[1]
        feature_desc = "PCA components"
    else:
        n_features = adata.n_vars
        feature_desc = "Genes (post-HVG)"

    # Number of classes
    if LABEL_KEY not in adata.obs:
        raise ValueError(f"Expected adata.obs['{LABEL_KEY}'] to exist.")
    n_classes = adata.obs[LABEL_KEY].nunique()

    # Create 1-row summary table
    summary = pd.DataFrame({
        "Number of cells": [n_cells],
        f"Number of features ({feature_desc})": [n_features],
        "Number of classes": [n_classes],
    })

    summary.to_csv(OUT_PATH, index=False)

    print("Final dataset shape summary:")
    print(summary)
    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()