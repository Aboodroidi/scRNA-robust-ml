"""
Export PBMC 3K raw counts from h5ad to 10x MEX format so SingleR can read it.
Writes matrix.mtx.gz, barcodes.tsv.gz, features.tsv.gz under
data/raw/pbmc3k/mex/
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import gzip
from pathlib import Path
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from scipy.io import mmwrite


SRC = "data/raw/pbmc3k/pbmc3k_raw.h5ad"
DST = Path("data/raw/pbmc3k/mex")
DST.mkdir(parents=True, exist_ok=True)


def main():
    print(f"Reading {SRC}…")
    adata = sc.read_h5ad(SRC)
    print(f"  shape: {adata.shape}")

    X = adata.X
    if not sp.issparse(X):
        X = sp.csr_matrix(X)

    # Transpose so rows are genes, cols are cells (10x convention)
    X_T = X.T.tocoo()

    # matrix.mtx
    mtx_path = DST / "matrix.mtx"
    print(f"  writing {mtx_path}…")
    mmwrite(str(mtx_path), X_T)

    # gzip
    with open(mtx_path, "rb") as fi, gzip.open(str(mtx_path) + ".gz", "wb") as fo:
        fo.write(fi.read())
    os.remove(mtx_path)

    # barcodes.tsv.gz
    barcodes_path = DST / "barcodes.tsv.gz"
    print(f"  writing {barcodes_path}…")
    with gzip.open(barcodes_path, "wt") as f:
        for bc in adata.obs_names:
            f.write(f"{bc}\n")

    # features.tsv.gz — three columns (ensembl_id, gene_symbol, Gene Expression)
    features_path = DST / "features.tsv.gz"
    print(f"  writing {features_path}…")
    with gzip.open(features_path, "wt") as f:
        for g in adata.var_names:
            f.write(f"{g}\t{g}\tGene Expression\n")

    print(f"\n✅ Exported to {DST}")
    print(f"   Files: matrix.mtx.gz, barcodes.tsv.gz, features.tsv.gz")


if __name__ == "__main__":
    main()
