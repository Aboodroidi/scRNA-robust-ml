"""
scANVI comparator — Google Colab variant.

The local Mac cannot run scvi-tools (AVX / jax incompatibility), so this
script is intended to be uploaded to Colab together with a small data
bundle (`scanvi_data.tgz`).  A T4 / L4 GPU runs the full pipeline in
10–15 min.

Colab workflow
──────────────
1. On Colab, runtime → change runtime type → GPU (T4 is free)
2. Upload `scanvi_data.tgz` to the session:
       !tar xzf scanvi_data.tgz -C /content/
3. Install scvi-tools:
       !pip install -q "scvi-tools" "anndata"
4. Upload this script (or paste its contents into a cell) and run:
       !python comparator_scanvi_colab.py --use_gpu
5. The script writes two CSVs to `/content/results/`:
       scanvi_8k_predictions.csv
       scanvi_3k_predictions.csv
6. Download both files back to the local repo at:
       results/comparators/scanvi/
7. Run locally for metrics + confusion matrices:
       KMP_DUPLICATE_LIB_OK=TRUE python src/comparator_scanvi_postprocess.py

The script uses the same label CSVs already produced by Seurat
(`results/comparators/seurat/labels/*.csv`) — no need to upload the
processed 2000-HVG h5ads.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import anndata as ad


# ══════════════════════════════════════════════════
# PATHS — default to /content/ (Colab session root)
# ══════════════════════════════════════════════════
DEFAULT_ROOT = "/content"
RAW_68K      = "data/raw/pbmc68k/filtered_matrices_mex/hg19"
RAW_8K       = "data/raw/pbmc8k/filtered_gene_bc_matrices/GRCh38"
RAW_3K_MEX   = "data/raw/pbmc3k/mex"
LBL_68K      = "results/comparators/seurat/labels/pbmc68k_coarse_labels.csv"
LBL_8K       = "results/comparators/seurat/labels/pbmc8k_labels.csv"
LBL_3K       = "results/comparators/seurat/labels/pbmc3k_labels.csv"

COARSE_CLASSES = ["B cells", "Mono", "NK", "Platelet", "T cells"]


def read_mex(path):
    return sc.read_10x_mtx(path, var_names="gene_symbols", make_unique=True)


def attach_labels(adata, label_csv):
    lbl = pd.read_csv(label_csv).set_index("barcode")
    common = adata.obs_names.intersection(lbl.index)
    adata = adata[common].copy()
    adata.obs["label"] = lbl.loc[common, "label"].astype(str).values
    return adata


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",      default=DEFAULT_ROOT)
    ap.add_argument("--outdir",    default="/content/results")
    ap.add_argument("--n_hvg",     type=int, default=2000)
    ap.add_argument("--scvi_epochs",   type=int, default=100)
    ap.add_argument("--scanvi_epochs", type=int, default=50)
    ap.add_argument("--use_gpu",   action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    import scvi
    print(f"[scvi-tools version] {scvi.__version__}")
    print(f"[accelerator       ] {'gpu' if args.use_gpu else 'cpu'}")

    t_start = time.time()

    # ── 1. Load raw counts for all three donors ────────────────────
    print("\n── Loading raw 10x MEX ──")
    a68 = read_mex(str(root / RAW_68K))
    a8  = read_mex(str(root / RAW_8K))
    a3  = read_mex(str(root / RAW_3K_MEX))
    print(f"  68K: {a68.shape}   8K: {a8.shape}   3K: {a3.shape}")

    # ── 2. Attach labels via CSV lookup ────────────────────────────
    print("\n── Attaching labels ──")
    a68 = attach_labels(a68, str(root / LBL_68K))
    a8  = attach_labels(a8,  str(root / LBL_8K))
    a3  = attach_labels(a3,  str(root / LBL_3K))
    print(f"  68K labelled: {a68.shape}")
    print(f"   8K labelled: {a8.shape}")
    print(f"   3K labelled: {a3.shape}")

    # Coarsen 8K/3K labels to 5-class taxonomy so we can evaluate later
    label_map = {
        "B cells": "B cells",
        "Mono": "Mono",
        "CD8 T": "T cells",
        "T cells": "T cells",
        "NK": "NK",
        "Platelet": "Platelet",
    }
    a8.obs["true_coarse"]  = [label_map.get(l, l) for l in a8.obs["label"]]
    a3.obs["true_coarse"]  = [label_map.get(l, l) for l in a3.obs["label"]]
    a68.obs["true_coarse"] = a68.obs["label"].values  # already coarse

    # ── 3. Gene intersection ───────────────────────────────────────
    print("\n── Gene intersection ──")
    common = sorted(set(a68.var_names) & set(a8.var_names) & set(a3.var_names))
    print(f"  shared genes: {len(common)}")
    a68 = a68[:, common].copy()
    a8  = a8[:, common].copy()
    a3  = a3[:, common].copy()

    # ── 4. Concatenate with batch keys ─────────────────────────────
    print("\n── Concatenating with batch labels ──")
    a68.obs["batch"]  = "68K"
    a8.obs["batch"]   = "8K"
    a3.obs["batch"]   = "3K"
    a68.obs["labels"] = a68.obs["true_coarse"].values  # known
    a8.obs["labels"]  = "Unknown"
    a3.obs["labels"]  = "Unknown"

    adata = ad.concat(
        [a68, a8, a3], join="outer",
        label="dataset", keys=["68K", "8K", "3K"],
        index_unique="-",
    )
    adata.layers["counts"] = adata.X.copy()
    print(f"  concatenated: {adata.shape}")
    print(f"  batch sizes : {adata.obs['batch'].value_counts().to_dict()}")

    # ── 5. HVG selection (seurat_v3 on raw counts) ────────────────
    print("\n── HVG selection ──")
    sc.pp.highly_variable_genes(
        adata, n_top_genes=args.n_hvg, subset=True,
        layer="counts", flavor="seurat_v3", batch_key="batch",
    )
    print(f"  retained {adata.shape[1]} HVGs")

    # ── 6. Train SCVI then fine-tune with SCANVI ──────────────────
    print("\n── Training SCVI ──")
    scvi.model.SCVI.setup_anndata(
        adata, layer="counts", batch_key="batch", labels_key="labels",
    )
    scvi_model = scvi.model.SCVI(adata, n_layers=2, n_latent=30)
    t0 = time.time()
    scvi_model.train(
        max_epochs=args.scvi_epochs,
        accelerator="gpu" if args.use_gpu else "cpu",
    )
    scvi_time = (time.time() - t0) / 60
    print(f"  scvi training: {scvi_time:.1f} min")

    print("\n── Fine-tuning with SCANVI ──")
    scanvi_model = scvi.model.SCANVI.from_scvi_model(
        scvi_model, unlabeled_category="Unknown", labels_key="labels",
    )
    t0 = time.time()
    scanvi_model.train(
        max_epochs=args.scanvi_epochs,
        accelerator="gpu" if args.use_gpu else "cpu",
    )
    scanvi_time = (time.time() - t0) / 60
    print(f"  scanvi training: {scanvi_time:.1f} min")

    # ── 7. Predict on 8K + 3K query cells ─────────────────────────
    print("\n── Predicting ──")
    preds = scanvi_model.predict(adata)
    adata.obs["scanvi_pred"] = preds

    for ds in ["8K", "3K"]:
        sub = adata[adata.obs["batch"] == ds]
        # Barcode without the "-{ds}" suffix anndata appended
        raw_bc = [b.rsplit("-", 1)[0] if b.endswith(f"-{ds}") else b
                  for b in sub.obs_names.values]
        df = pd.DataFrame({
            "barcode":         raw_bc,
            "predicted_label": sub.obs["scanvi_pred"].astype(str).values,
            "true_raw":        sub.obs["label"].astype(str).values,
            "true_coarse":     sub.obs["true_coarse"].astype(str).values,
        })
        out = outdir / f"scanvi_{ds.lower()}_predictions.csv"
        df.to_csv(out, index=False)
        print(f"  wrote {len(df)} predictions → {out}")

    # Run info (metrics computed locally afterwards)
    run_info = pd.DataFrame([{
        "tool":            "scANVI",
        "scvi_version":    scvi.__version__,
        "n_hvg":           adata.shape[1],
        "scvi_epochs":     args.scvi_epochs,
        "scanvi_epochs":   args.scanvi_epochs,
        "scvi_time_min":   float(scvi_time),
        "scanvi_time_min": float(scanvi_time),
        "total_time_min":  float((time.time() - t_start) / 60),
        "accelerator":     "gpu" if args.use_gpu else "cpu",
    }])
    run_info.to_csv(outdir / "scanvi_run_info.csv", index=False)
    print(f"\n✅ Prediction CSVs saved to {outdir}")
    print("   Download them locally into results/comparators/scanvi/ then run:")
    print("   python src/comparator_scanvi_postprocess.py")


if __name__ == "__main__":
    main()
