"""
Export barcode → label CSVs for Seurat label transfer.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import pandas as pd
import scanpy as sc

OUT = Path("results/comparators/seurat/labels")
OUT.mkdir(parents=True, exist_ok=True)

datasets = [
    ("data/processed/pbmc68k_labeled.h5ad", "cell_type_coarse", OUT / "pbmc68k_coarse_labels.csv"),
    ("data/processed/pbmc8k_labeled.h5ad",  "cell_type",        OUT / "pbmc8k_labels.csv"),
    ("data/processed/pbmc3k_labeled.h5ad",  "cell_type",        OUT / "pbmc3k_labels.csv"),
]

for path, col, out in datasets:
    adata = sc.read_h5ad(path)
    df = pd.DataFrame({
        "barcode": adata.obs_names,
        "label":   adata.obs[col].astype(str).values,
    })
    df.to_csv(out, index=False)
    print(f"  {path}  →  {out}  ({len(df)} cells, {df['label'].nunique()} classes)")

print("\n✅ Label CSVs exported.")
