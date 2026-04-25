"""
Comparator: scANVI (scvi-tools) vs SANN_PCA ensemble.

Trains SCVI on combined 68K+8K+3K with batch labels, then SCANVI fine-tunes
using 68K labels as the labelled subset. Predicts on held-out 8K and 3K.

Outputs:
    results/comparators/scanvi/scanvi_8k_predictions.csv
    results/comparators/scanvi/scanvi_3k_predictions.csv
    results/comparators/scanvi/scanvi_8k_confusion.csv
    results/comparators/scanvi/scanvi_3k_confusion.csv
    results/comparators/scanvi/scanvi_metrics.json
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import anndata as ad
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
)


# ══════════════════════════════════════════════════
# LABEL MAPPING
# ══════════════════════════════════════════════════
# Map tool-predicted labels (based on 68K training labels) to coarse 5-class taxonomy.
# The 68K has cell_type_coarse with classes: ['B cells', 'Mono', 'NK', 'Platelet', 'T cells']
# The 8K/3K use cell_type with potentially different names.
LABEL_MAP = {
    # 8K cell_type → coarse
    "B cells": "B cells",
    "Mono": "Mono",
    "CD8 T": "T cells",
    "T cells": "T cells",
    "NK": "NK",
    "Platelet": "Platelet",
    # classes to exclude from evaluation (not in model taxonomy)
    "CL 1": None,
    "DC": None,
}

COARSE_CLASSES = ["B cells", "Mono", "NK", "Platelet", "T cells"]


def map_labels(labels, mapping=LABEL_MAP):
    mapped = []
    keep = []
    for lbl in labels:
        m = mapping.get(str(lbl), str(lbl))
        if m in COARSE_CLASSES:
            mapped.append(m)
            keep.append(True)
        else:
            mapped.append(None)
            keep.append(False)
    return np.array(mapped, dtype=object), np.array(keep, dtype=bool)


# ══════════════════════════════════════════════════
# RAW COUNT LOADING
# ══════════════════════════════════════════════════
def load_68k_raw(path_mex):
    """68K raw counts from 10x mtx format (hg19)."""
    adata = sc.read_10x_mtx(path_mex, var_names="gene_symbols", make_unique=True)
    return adata


def load_8k_raw(path_mex):
    """8K raw counts from 10x mtx format (GRCh38)."""
    adata = sc.read_10x_mtx(path_mex, var_names="gene_symbols", make_unique=True)
    return adata


def load_3k_raw(path_h5ad):
    """3K raw counts from h5ad."""
    adata = sc.read_h5ad(path_h5ad)
    return adata


def basic_qc(adata, min_genes=200, mt_threshold=10.0):
    """Light QC filtering matching the main pipeline."""
    X = adata.X
    if sp.issparse(X):
        genes_per_cell = np.array((X > 0).sum(axis=1)).flatten()
        total_counts = np.array(X.sum(axis=1)).flatten()
    else:
        genes_per_cell = (X > 0).sum(axis=1)
        total_counts = X.sum(axis=1)
    mt_mask = adata.var_names.str.upper().str.startswith("MT-")
    if sp.issparse(X):
        mt_counts = np.array(X[:, mt_mask].sum(axis=1)).flatten()
    else:
        mt_counts = X[:, mt_mask].sum(axis=1)
    pct_mt = mt_counts / (total_counts + 1e-8) * 100
    keep_cells = (genes_per_cell >= min_genes) & (pct_mt < mt_threshold)
    adata = adata[keep_cells].copy()
    if sp.issparse(adata.X):
        cells_per_gene = np.array((adata.X > 0).sum(axis=0)).flatten()
    else:
        cells_per_gene = (adata.X > 0).sum(axis=0)
    adata = adata[:, cells_per_gene >= 3].copy()
    return adata


# ══════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_68k", default="data/raw/pbmc68k/filtered_matrices_mex/hg19")
    ap.add_argument("--raw_8k",  default="data/raw/pbmc8k/filtered_gene_bc_matrices/GRCh38")
    ap.add_argument("--raw_3k",  default="data/raw/pbmc3k/pbmc3k_raw.h5ad")
    ap.add_argument("--processed_68k", default="data/processed/pbmc68k_labeled.h5ad")
    ap.add_argument("--processed_8k",  default="data/processed/pbmc8k_labeled.h5ad")
    ap.add_argument("--processed_3k",  default="data/processed/pbmc3k_labeled.h5ad")
    ap.add_argument("--outdir", default="results/comparators/scanvi")
    ap.add_argument("--n_hvg", type=int, default=2000)
    ap.add_argument("--scvi_epochs", type=int, default=100)
    ap.add_argument("--scanvi_epochs", type=int, default=50)
    ap.add_argument("--use_gpu", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    # Import scvi here to avoid import overhead if only viewing --help
    import scvi
    print(f"\n[scvi-tools version]  {scvi.__version__}")
    print(f"[torch device]        {'GPU' if args.use_gpu else 'CPU'}")

    # ══════════════════════════════════════════════════
    # 1. Load raw counts + labels from the three datasets
    # ══════════════════════════════════════════════════
    print("\n── Loading datasets ──────────────────────────────────────")

    # 68K: raw counts + barcode matching to labelled processed file
    print("[68K] loading raw counts…")
    a68_raw = load_68k_raw(args.raw_68k)
    print(f"       raw:       {a68_raw.shape}")
    a68_proc = sc.read_h5ad(args.processed_68k)
    # Match barcodes of labelled cells
    common_bc = a68_raw.obs_names.intersection(a68_proc.obs_names)
    a68_raw = a68_raw[common_bc].copy()
    labels_68k = a68_proc.obs.loc[common_bc, "cell_type_coarse"].astype(str).values
    print(f"       matched:   {a68_raw.shape} cells with labels")

    print("[8K] loading raw counts…")
    a8_raw = load_8k_raw(args.raw_8k)
    a8_raw = basic_qc(a8_raw)
    a8_proc = sc.read_h5ad(args.processed_8k)
    common_bc8 = a8_raw.obs_names.intersection(a8_proc.obs_names)
    a8_raw = a8_raw[common_bc8].copy()
    labels_8k_raw = a8_proc.obs.loc[common_bc8, "cell_type"].astype(str).values
    print(f"       matched:   {a8_raw.shape}")

    print("[3K] loading raw counts…")
    a3_raw = load_3k_raw(args.raw_3k)
    a3_proc = sc.read_h5ad(args.processed_3k)
    common_bc3 = a3_raw.obs_names.intersection(a3_proc.obs_names)
    a3_raw = a3_raw[common_bc3].copy()
    labels_3k_raw = a3_proc.obs.loc[common_bc3, "cell_type"].astype(str).values
    print(f"       matched:   {a3_raw.shape}")

    # ══════════════════════════════════════════════════
    # 2. Align to common gene set
    # ══════════════════════════════════════════════════
    print("\n── Aligning genes across datasets ────────────────────────")
    common_genes = (
        set(a68_raw.var_names) &
        set(a8_raw.var_names) &
        set(a3_raw.var_names)
    )
    common_genes = sorted(common_genes)
    print(f"   common genes across 68K/8K/3K: {len(common_genes)}")

    a68_raw = a68_raw[:, common_genes].copy()
    a8_raw  = a8_raw[:, common_genes].copy()
    a3_raw  = a3_raw[:, common_genes].copy()

    # ══════════════════════════════════════════════════
    # 3. Concatenate + add batch + label metadata
    # ══════════════════════════════════════════════════
    print("\n── Concatenating with batch labels ───────────────────────")
    a68_raw.obs["batch"] = "68K"
    a8_raw.obs["batch"]  = "8K"
    a3_raw.obs["batch"]  = "3K"

    # 68K has labels; 8K and 3K are treated as unlabelled for SCANVI
    a68_raw.obs["labels"] = labels_68k
    a8_raw.obs["labels"]  = "Unknown"
    a3_raw.obs["labels"]  = "Unknown"

    # Keep ground truth for evaluation (not used in training)
    a68_raw.obs["true_label"] = labels_68k
    a8_raw.obs["true_label"]  = labels_8k_raw
    a3_raw.obs["true_label"]  = labels_3k_raw

    adata = ad.concat([a68_raw, a8_raw, a3_raw], join="outer",
                       label="dataset", keys=["68K", "8K", "3K"], index_unique="-")
    adata.layers["counts"] = adata.X.copy()
    print(f"   concatenated: {adata.shape}")
    print(f"   batch counts: {adata.obs['batch'].value_counts().to_dict()}")

    # ══════════════════════════════════════════════════
    # 4. Normalise + HVG selection (standard scvi-tools workflow)
    # ══════════════════════════════════════════════════
    print("\n── HVG selection on combined dataset ─────────────────────")
    sc.pp.highly_variable_genes(
        adata, n_top_genes=args.n_hvg, subset=True,
        layer="counts", flavor="seurat_v3", batch_key="batch",
    )
    print(f"   after HVG selection: {adata.shape}")

    # ══════════════════════════════════════════════════
    # 5. SCVI training
    # ══════════════════════════════════════════════════
    print("\n── Training SCVI ─────────────────────────────────────────")
    scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key="batch",
                                   labels_key="labels")
    scvi_model = scvi.model.SCVI(adata, n_layers=2, n_latent=30)

    t0 = time.time()
    scvi_model.train(
        max_epochs=args.scvi_epochs,
        accelerator="gpu" if args.use_gpu else "cpu",
    )
    scvi_time = time.time() - t0
    print(f"   SCVI training time: {scvi_time/60:.1f} min")

    # ══════════════════════════════════════════════════
    # 6. SCANVI fine-tuning
    # ══════════════════════════════════════════════════
    print("\n── Fine-tuning with SCANVI ───────────────────────────────")
    scanvi_model = scvi.model.SCANVI.from_scvi_model(
        scvi_model, unlabeled_category="Unknown", labels_key="labels",
    )

    t0 = time.time()
    scanvi_model.train(
        max_epochs=args.scanvi_epochs,
        accelerator="gpu" if args.use_gpu else "cpu",
    )
    scanvi_time = time.time() - t0
    print(f"   SCANVI training time: {scanvi_time/60:.1f} min")

    # ══════════════════════════════════════════════════
    # 7. Predict on 8K and 3K
    # ══════════════════════════════════════════════════
    print("\n── Predicting on 8K and 3K ───────────────────────────────")
    preds = scanvi_model.predict(adata)
    adata.obs["scanvi_pred"] = preds

    metrics = {"scvi_time_min": scvi_time / 60, "scanvi_time_min": scanvi_time / 60}

    for ds in ["8K", "3K"]:
        sub = adata[adata.obs["batch"] == ds].copy()
        true_raw = sub.obs["true_label"].astype(str).values
        true_mapped, keep = map_labels(true_raw)
        pred_raw = sub.obs["scanvi_pred"].astype(str).values

        # Keep only cells with a valid coarse label
        true_eval = true_mapped[keep]
        pred_eval = pred_raw[keep]

        n_total = len(true_raw)
        n_known = keep.sum()
        n_excluded = n_total - n_known

        # Save per-cell predictions
        pred_df = pd.DataFrame({
            "barcode": sub.obs_names.values,
            "true_raw": true_raw,
            "true_mapped": true_mapped,
            "scanvi_pred": pred_raw,
            "included_in_eval": keep,
        })
        pred_df.to_csv(outdir / f"scanvi_{ds}_predictions.csv", index=False)

        # Metrics
        acc = accuracy_score(true_eval, pred_eval)
        f1_macro = f1_score(true_eval, pred_eval, labels=COARSE_CLASSES,
                             average="macro", zero_division=0)
        f1_weighted = f1_score(true_eval, pred_eval, labels=COARSE_CLASSES,
                                average="weighted", zero_division=0)
        f1_per_class = f1_score(true_eval, pred_eval, labels=COARSE_CLASSES,
                                 average=None, zero_division=0)

        print(f"\n  [{ds}]  total={n_total}  known_types={n_known}  excluded={n_excluded}")
        print(f"       accuracy    = {acc:.4f}")
        print(f"       macro-F1    = {f1_macro:.4f}")
        print(f"       weighted-F1 = {f1_weighted:.4f}")
        print(f"       per-class F1:")
        for c, v in zip(COARSE_CLASSES, f1_per_class):
            print(f"         {c:>10s}: {v:.4f}")

        # Confusion matrix
        cm = confusion_matrix(true_eval, pred_eval, labels=COARSE_CLASSES)
        cm_df = pd.DataFrame(cm, index=COARSE_CLASSES, columns=COARSE_CLASSES)
        cm_df.to_csv(outdir / f"scanvi_{ds}_confusion.csv")

        metrics[ds] = {
            "n_total": int(n_total),
            "n_known": int(n_known),
            "n_excluded": int(n_excluded),
            "accuracy": float(acc),
            "macro_f1": float(f1_macro),
            "weighted_f1": float(f1_weighted),
            "per_class_f1": dict(zip(COARSE_CLASSES, [float(x) for x in f1_per_class])),
        }

    metrics["total_time_min"] = (time.time() - t_start) / 60
    metrics["scvi_version"] = scvi.__version__

    with open(outdir / "scanvi_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✅ Results saved to {outdir}")
    print(f"   Total time: {metrics['total_time_min']:.1f} min")


if __name__ == "__main__":
    main()
