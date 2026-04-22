#!/usr/bin/env python
"""
Build a 3x3 train-vs-test macro-F1 transfer matrix per model.

Rows = training donor (68K, 8K, 3K)
Cols = test donor    (68K, 8K, 3K)

Diagonals    = within-donor validation (stratified val split of training donor)
Off-diagonal = cross-donor transfer (external eval of model on the other donor)

Three side-by-side heatmaps: LR_PCA, XGB_PCA, SANN_PCA, sharing a colour scale.

Data sources:
  68K-trained:
    within-donor  : results/full_train_all_pca/baseline_metrics_full.csv
    → 8K          : results/external_validation/external_validation_summary.csv
    → 3K          : results/external_validation_3k/external_validation_3k_summary.csv
  8K-trained:
    within-donor  : train_summary.json (LR, SANN); XGB recomputed from model+split
    → 3K, → 68K   : results/train_8k/external_eval_summary.csv
  3K-trained:
    within-donor  : train_summary.json (LR, SANN); XGB recomputed from model+split
    → 8K, → 68K   : results/train_3k/external_eval_summary.csv
"""
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["MPLBACKEND"] = "Agg"

import json
import numpy as np
import pandas as pd
import scanpy as sc
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score

ROOT = "/Users/Abdullah/scRNA-robust-ml"
OUT_PATH = os.path.join(ROOT, "results/figures/transfer_matrix.png")

DONORS = ["68K", "8K", "3K"]  # display order (rows & cols)
MODELS = ["LR_PCA", "XGB_PCA", "SANN_PCA"]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def safe_mean_sann_val(train_summary_path):
    """Return mean best_val_f1 across SANN seeds, or None."""
    with open(train_summary_path) as f:
        entries = json.load(f)
    vals = [e["best_val_f1"] for e in entries
            if e.get("Model") == "SANN" and "best_val_f1" in e]
    return float(np.mean(vals)) if vals else None


def read_kv(train_summary_path, model_name, key="best_val_f1"):
    with open(train_summary_path) as f:
        entries = json.load(f)
    for e in entries:
        if e.get("Model") == model_name and key in e:
            return float(e[key])
    return None


def recompute_xgb_within(train_dir, h5ad_path, raw_8k_dir=None,
                         label_key="cell_type_coarse", val_frac=0.15, seed=42,
                         coarsen=False):
    """
    Reproduce the train/val split used by train_3k/train_8k scripts, project
    through the saved expr PCA, and compute XGB val macro-F1.

    coarsen: apply the same coarsening used in train_8k_eval_3k_68k.py when True.
    raw_8k_dir: when set, we rebuild the 8K training matrix from raw (to match
                the script, which derives its own z-score stats).
    """
    expr_pca = joblib.load(os.path.join(train_dir, "expr_pca_model.pkl"))
    booster = xgb.Booster()
    booster.load_model(os.path.join(train_dir, "xgb_model.json"))

    if raw_8k_dir is not None:
        # 8K pipeline: load labels, reload raw, align, z-score with 8K stats,
        # intersect with 3K genes.
        import scipy.sparse as sp

        ad8 = sc.read_h5ad(h5ad_path)
        genes_8k_full = list(ad8.var_names)

        # Raw 8K → log-norm
        adata_raw = sc.read_10x_mtx(raw_8k_dir, var_names="gene_symbols",
                                    make_unique=True)
        X = adata_raw.X
        if sp.issparse(X):
            gpc = np.array((X > 0).sum(axis=1)).flatten()
            tc = np.array(X.sum(axis=1)).flatten()
        else:
            gpc = (X > 0).sum(axis=1)
            tc = X.sum(axis=1)
        mt = adata_raw.var_names.str.upper().str.startswith("MT-")
        if sp.issparse(X):
            mc = np.array(X[:, mt].sum(axis=1)).flatten()
        else:
            mc = X[:, mt].sum(axis=1)
        pmt = mc / (tc + 1e-8) * 100
        keep = (gpc >= 200) & (pmt < 10.0)
        adata_raw = adata_raw[keep].copy()
        if sp.issparse(adata_raw.X):
            cpg = np.array((adata_raw.X > 0).sum(axis=0)).flatten()
        else:
            cpg = (adata_raw.X > 0).sum(axis=0)
        adata_raw = adata_raw[:, cpg >= 3].copy()
        sc.pp.normalize_total(adata_raw, target_sum=1e4)
        sc.pp.log1p(adata_raw)
        X_raw_dense = (adata_raw.X.toarray() if sp.issparse(adata_raw.X)
                       else np.asarray(adata_raw.X)).astype(np.float32)
        genes_raw = list(adata_raw.var_names)
        obs_raw = list(adata_raw.obs_names)

        # Align raw→8K labeled gene order
        src_idx = {g: i for i, g in enumerate(genes_raw)}
        X_log = np.zeros((X_raw_dense.shape[0], len(genes_8k_full)),
                         dtype=np.float32)
        for i, g in enumerate(genes_8k_full):
            j = src_idx.get(g)
            if j is not None:
                X_log[:, i] = X_raw_dense[:, j]

        raw_idx = {n: i for i, n in enumerate(obs_raw)}
        kept = [n for n in ad8.obs_names if n in raw_idx]
        keep_rows = np.array([raw_idx[n] for n in kept])
        X_log = X_log[keep_rows]
        ad8 = ad8[kept].copy()

        ref_mean = X_log.mean(axis=0).astype(np.float32)
        ref_std = X_log.std(axis=0).astype(np.float32)
        std_safe = np.where(ref_std == 0, 1.0, ref_std)
        X_z = np.clip((X_log - ref_mean) / std_safe, -10, 10).astype(np.float32)

        # Intersect with 3K genes (same as the script does by default)
        inter_path = os.path.join(train_dir, "intersection_genes.txt")
        if os.path.exists(inter_path):
            with open(inter_path) as f:
                inter_genes = [l.strip() for l in f if l.strip()]
            inter_set = set(inter_genes)
            keep_mask = np.array([g in inter_set for g in genes_8k_full])
            X_z = X_z[:, keep_mask]

        raw_labels = ad8.obs[label_key].astype(str).to_numpy()
        if coarsen:
            COARSE_MAP = {
                "T cells": "T cells", "CD8 T": "T cells", "CD4 T": "T cells",
                "Treg": "T cells",
                "NK": "NK", "NK cells": "NK",
                "B cells": "B cells", "B": "B cells", "Plasma": "B cells",
                "Mono": "Mono", "Monocytes": "Mono",
                "CD14+ Monocytes": "Mono", "FCGR3A+ Monocytes": "Mono",
                "DC": "DC", "Dendritic cells": "DC",
                "Platelet": "Platelet", "Megakaryocytes": "Platelet",
            }
            mapped = np.array([COARSE_MAP.get(n, None) for n in raw_labels])
            keep = np.array([m is not None for m in mapped])
            X_z = X_z[keep]
            mapped = mapped[keep].astype(str)
            class_names = sorted(set(mapped.tolist()))
            name_to_idx = {n: i for i, n in enumerate(class_names)}
            y = np.array([name_to_idx[n] for n in mapped], dtype=np.int64)
        else:
            y_cat = ad8.obs[label_key].astype("category")
            y = y_cat.cat.codes.to_numpy()
            # align to kept rows above — already done
        # Drop singleton classes
        counts = np.bincount(y)
        singleton = [i for i, c in enumerate(counts) if c < 2]
        if singleton:
            kk = ~np.isin(y, singleton)
            X_z = X_z[kk]
            y = y[kk]
            uniq = sorted(set(y.tolist()))
            remap = {o: n for n, o in enumerate(uniq)}
            y = np.array([remap[c] for c in y], dtype=np.int64)

        X_feat = X_z
    else:
        # 3K pipeline: use adata.X directly (already z-scored in 3K stats),
        # then subset to the intersection gene set saved by the training run.
        ad3 = sc.read_h5ad(h5ad_path)
        y_cat = ad3.obs[label_key].astype("category")
        y = y_cat.cat.codes.to_numpy()

        import scipy.sparse as sp
        X_feat = ad3.X.toarray() if sp.issparse(ad3.X) else np.asarray(ad3.X)
        X_feat = X_feat.astype(np.float32)

        # Apply the same gene intersection used at training time
        inter_path = os.path.join(train_dir, "intersection_genes.txt")
        if os.path.exists(inter_path):
            with open(inter_path) as f:
                inter_genes = [l.strip() for l in f if l.strip()]
            g_to_i = {g: i for i, g in enumerate(ad3.var_names)}
            cols = [g_to_i[g] for g in inter_genes if g in g_to_i]
            X_feat = X_feat[:, cols]
        # Drop singletons
        counts = np.bincount(y)
        singleton = [i for i, c in enumerate(counts) if c < 2]
        if singleton:
            kk = ~np.isin(y, singleton)
            X_feat = X_feat[kk]
            y = y[kk]
            uniq = sorted(set(y.tolist()))
            remap = {o: n for n, o in enumerate(uniq)}
            y = np.array([remap[c] for c in y], dtype=np.int64)

    # Reproduce split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    tr_idx, val_idx = next(sss.split(np.zeros(len(y)), y))
    X_val = X_feat[val_idx]
    y_val = y[val_idx]

    X_val_pca = expr_pca.transform(X_val).astype(np.float32)
    pred = booster.predict(xgb.DMatrix(X_val_pca)).argmax(axis=1)
    return float(f1_score(y_val, pred, average="macro", zero_division=0))


# -----------------------------------------------------------------------------
# Build the 3x3 matrix per model
# -----------------------------------------------------------------------------
def build_matrices():
    # Off-diagonal + diagonal (68K row) — from summary CSVs
    ext_68k_to_8k = pd.read_csv(os.path.join(
        ROOT, "results/external_validation/external_validation_summary.csv"))
    ext_68k_to_3k = pd.read_csv(os.path.join(
        ROOT, "results/external_validation_3k/external_validation_3k_summary.csv"))
    ext_3k = pd.read_csv(os.path.join(ROOT, "results/train_3k/external_eval_summary.csv"))
    ext_8k = pd.read_csv(os.path.join(ROOT, "results/train_8k/external_eval_summary.csv"))
    baseline_68k = pd.read_csv(os.path.join(
        ROOT, "results/full_train_all_pca/baseline_metrics_full.csv"))

    # 68K within-donor (diagonal) — from baseline_metrics_full
    b = baseline_68k.set_index("Model")["Macro-F1"]
    diag_68k = {
        "LR_PCA": float(b.loc["LR"]),
        "XGB_PCA": float(b.loc["XGB"]),
        "SANN_PCA": float(b.loc["SANN"]),
    }

    # 8K within-donor: LR + SANN from train_summary, XGB recomputed
    ts_8k_path = os.path.join(ROOT, "results/train_8k/train_summary.json")
    diag_8k = {
        "LR_PCA": read_kv(ts_8k_path, "LR"),
        "SANN_PCA": safe_mean_sann_val(ts_8k_path),
    }
    print("Recomputing 8K XGB within-donor val F1 ...")
    diag_8k["XGB_PCA"] = recompute_xgb_within(
        train_dir=os.path.join(ROOT, "results/train_8k"),
        h5ad_path=os.path.join(ROOT, "data/processed/pbmc8k_labeled.h5ad"),
        raw_8k_dir=os.path.join(
            ROOT, "data/raw/pbmc8k/filtered_gene_bc_matrices/GRCh38"),
        label_key="cell_type",
        coarsen=True,
    )

    # 3K within-donor
    ts_3k_path = os.path.join(ROOT, "results/train_3k/train_summary.json")
    diag_3k = {
        "LR_PCA": read_kv(ts_3k_path, "LR"),
        "SANN_PCA": safe_mean_sann_val(ts_3k_path),
    }
    print("Recomputing 3K XGB within-donor val F1 ...")
    diag_3k["XGB_PCA"] = recompute_xgb_within(
        train_dir=os.path.join(ROOT, "results/train_3k"),
        h5ad_path=os.path.join(ROOT, "data/processed/pbmc3k_labeled.h5ad"),
        raw_8k_dir=None,
        label_key="cell_type_coarse",
        coarsen=False,
    )

    # Off-diagonals from the 68K external CSVs.  These files use rows named
    # "LR_PCA", "XGB_PCA", "SANN_PCA" and column "Macro-F1 (known)".
    def _from_68k_row(df, model):
        return float(df.set_index("Model").loc[model, "Macro-F1 (known)"])

    # Off-diagonals from the 3K / 8K CSVs use rows "LR","XGB","SANN" with
    # "Macro-F1 (known)".
    def _from_3k_or_8k(df, model_short, test_set):
        r = df[(df["Model"] == model_short) & (df["TestSet"] == test_set)].iloc[0]
        return float(r["Macro-F1 (known)"])

    # Matrix per model: rows = train donor (DONORS order), cols = test donor.
    matrices = {}
    for m in MODELS:
        mat = np.full((3, 3), np.nan, dtype=np.float64)
        short = {"LR_PCA": "LR", "XGB_PCA": "XGB", "SANN_PCA": "SANN"}[m]

        # Row 0 = 68K train
        mat[0, 0] = diag_68k[m]                                       # 68K→68K
        mat[0, 1] = _from_68k_row(ext_68k_to_8k, m)                   # 68K→8K
        mat[0, 2] = _from_68k_row(ext_68k_to_3k, m)                   # 68K→3K

        # Row 1 = 8K train
        mat[1, 0] = _from_3k_or_8k(ext_8k, short, "pbmc68k")          # 8K→68K
        mat[1, 1] = diag_8k[m]                                        # 8K→8K
        mat[1, 2] = _from_3k_or_8k(ext_8k, short, "pbmc3k")           # 8K→3K

        # Row 2 = 3K train
        mat[2, 0] = _from_3k_or_8k(ext_3k, short, "pbmc68k")          # 3K→68K
        mat[2, 1] = _from_3k_or_8k(ext_3k, short, "pbmc8k")           # 3K→8K
        mat[2, 2] = diag_3k[m]                                        # 3K→3K

        matrices[m] = mat

    return matrices


# -----------------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------------
def plot_panels(matrices, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Shared colour scale from 0.5 to 1.0 for readability
    vals = np.concatenate([m.flatten() for m in matrices.values()])
    vmin = max(0.50, np.nanmin(vals) - 0.02)
    vmax = 1.0
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    cmap = plt.get_cmap("viridis")

    for ax, m in zip(axes, MODELS):
        mat = matrices[m]
        im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="equal")
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels([f"test {d}" for d in DONORS], fontsize=10)
        ax.set_yticklabels([f"train {d}" for d in DONORS], fontsize=10)
        ax.set_title(m, fontsize=12, pad=8)
        ax.set_xlabel("")
        ax.set_ylabel("")

        # Diagonal emphasis
        for i in range(3):
            ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                                       fill=False, edgecolor="white",
                                       linewidth=2.0))

        # Cell annotations
        for i in range(3):
            for j in range(3):
                v = mat[i, j]
                color = "white" if v < (vmin + vmax) / 2 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color=color, fontsize=11, fontweight="bold")

    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02,
                        label="macro-F1 (known)")
    cbar.ax.tick_params(labelsize=9)

    fig.suptitle("Cross-donor transfer: macro-F1 per model "
                 "(diagonal = within-donor val, off-diagonal = external)",
                 fontsize=12, y=1.02)

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    # Also save a CSV of the raw numbers
    csv_path = out_path.replace(".png", ".csv")
    rows = []
    for m in MODELS:
        for i, tr in enumerate(DONORS):
            for j, te in enumerate(DONORS):
                rows.append({"Model": m, "Train": tr, "Test": te,
                             "MacroF1": matrices[m][i, j]})
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")


# -----------------------------------------------------------------------------
def main():
    matrices = build_matrices()
    for m in MODELS:
        print(f"\n{m} matrix (rows train / cols test, order={DONORS}):")
        print(pd.DataFrame(matrices[m], index=DONORS, columns=DONORS).round(3))
    plot_panels(matrices, OUT_PATH)


if __name__ == "__main__":
    main()
