# src/evaluate_external_pbmc8k.py
"""
External validation: run trained LR, XGBoost, and SANN models
(from the 68K PBMC training set) on the independent 8K PBMC dataset.

KEY FIX: The 8K data must be z-scored using the 68K's per-gene mean/std
(stored in adata_68k.var['mean'] / var['std'] by sc.pp.scale).
Previously the 8K was z-scored with its OWN stats, causing distribution
mismatch with every model.

For PCA models: the 8K z-scored data is projected through the 68K's PCA
loadings (stored in adata_68k.varm['PCs']), not a fresh PCA.
"""
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["MPLBACKEND"] = "Agg"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import joblib
import xgboost as xgb

import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score, f1_score, classification_report
import glob as glob_module


def load_sann_ensemble(model_dir, model_class, X_input, model_filename="sann_model.pt"):
    """Load all SANN seed models from a directory and return averaged probabilities.
    If multiple seed models exist (sann_model_seed0.pt, seed1.pt, ...),
    loads all and averages softmax probabilities. Otherwise loads single model."""
    seed_files = sorted(glob_module.glob(os.path.join(model_dir, "sann_model_seed*.pt")))
    if len(seed_files) > 1:
        print(f"  Deep ensemble: found {len(seed_files)} seed models")
        all_probs = []
        for sf in seed_files:
            m = model_class()
            m.load_state_dict(torch.load(sf, map_location="cpu"))
            m.eval()
            with torch.no_grad():
                logits = m(torch.tensor(X_input, dtype=torch.float32)).numpy()
            probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
            all_probs.append(probs)
            print(f"    Loaded {os.path.basename(sf)}")
        avg_probs = np.mean(all_probs, axis=0)
        return avg_probs
    else:
        # Single model fallback
        m = model_class()
        m.load_state_dict(torch.load(
            os.path.join(model_dir, model_filename), map_location="cpu"
        ))
        m.eval()
        with torch.no_grad():
            logits = m(torch.tensor(X_input, dtype=torch.float32)).numpy()
        return torch.softmax(torch.tensor(logits), dim=1).numpy()


# ----------------------------
# SANN v2 architecture (must match training exactly)
# ----------------------------
def _make_norm(dim, use_batchnorm, use_layernorm):
    if use_layernorm:
        return nn.LayerNorm(dim)
    if use_batchnorm:
        return nn.BatchNorm1d(dim)
    return nn.Identity()


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1, use_batchnorm=False, use_layernorm=True):
        super().__init__()
        self.norm = _make_norm(dim, use_batchnorm, use_layernorm)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x):
        h = self.norm(x)
        h = self.act(self.fc1(h))
        h = self.drop(h)
        h = self.fc2(h)
        return x + h


class SANN(nn.Module):
    def __init__(
        self, expr_dim, mask_dim, num_classes,
        branch_hidden=512, branch_out=256, fusion_hidden=256,
        dropout=0.05, use_batchnorm=True, use_layernorm=False,
        input_noise=0.0,
    ):
        super().__init__()
        self.expr_dim = expr_dim
        self.input_noise = input_noise

        self.expr_proj = nn.Sequential(
            nn.Linear(expr_dim, branch_hidden),
            _make_norm(branch_hidden, use_batchnorm, use_layernorm),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self.expr_res1 = ResidualBlock(branch_hidden, dropout, use_batchnorm, use_layernorm)
        self.expr_res2 = ResidualBlock(branch_hidden, dropout, use_batchnorm, use_layernorm)
        self.expr_out = nn.Sequential(
            nn.Linear(branch_hidden, branch_out),
            _make_norm(branch_out, use_batchnorm, use_layernorm),
            nn.GELU(),
        )

        self.mask_proj = nn.Sequential(
            nn.Linear(mask_dim, branch_hidden),
            _make_norm(branch_hidden, use_batchnorm, use_layernorm),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self.mask_res1 = ResidualBlock(branch_hidden, dropout, use_batchnorm, use_layernorm)
        self.mask_res2 = ResidualBlock(branch_hidden, dropout, use_batchnorm, use_layernorm)
        self.mask_out = nn.Sequential(
            nn.Linear(branch_hidden, branch_out),
            _make_norm(branch_out, use_batchnorm, use_layernorm),
            nn.GELU(),
        )

        self.gate = nn.Sequential(
            nn.Linear(branch_out * 2, branch_out),
            nn.GELU(),
            nn.Linear(branch_out, branch_out),
            nn.Sigmoid(),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(branch_out, fusion_hidden),
            _make_norm(fusion_hidden, use_batchnorm, use_layernorm),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(fusion_hidden, num_classes),
        )

    def forward(self, x):
        x_expr = x[:, :self.expr_dim]
        x_mask = x[:, self.expr_dim:]

        if self.training and self.input_noise > 0:
            x_expr = x_expr + torch.randn_like(x_expr) * self.input_noise

        h_expr = self.expr_proj(x_expr)
        h_expr = self.expr_res1(h_expr)
        h_expr = self.expr_res2(h_expr)
        h_expr = self.expr_out(h_expr)

        h_mask = self.mask_proj(x_mask)
        h_mask = self.mask_res1(h_mask)
        h_mask = self.mask_res2(h_mask)
        h_mask = self.mask_out(h_mask)

        combined = torch.cat([h_expr, h_mask], dim=1)
        alpha = self.gate(combined)
        h_fused = alpha * h_expr + (1 - alpha) * h_mask

        return self.classifier(h_fused)


# ----------------------------
# Helpers
# ----------------------------
def to_dense_float32(x):
    if sp.issparse(x):
        return x.toarray().astype(np.float32)
    return np.asarray(x, dtype=np.float32)


def evaluate_model(name, y_true, y_pred, y_probs, class_names, outdir, known_class_indices=None):
    """Print metrics and save artifacts for one model.

    known_class_indices: list of class indices that exist in the 8K ground truth.
        Used to compute a fairer 'known-only' macro-F1 that excludes classes
        the 8K dataset doesn't contain (CL clusters from 68K).
    """
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    print(f"\n{'='*60}")
    print(f"  {name} — External Validation (PBMC 8K)")
    print(f"{'='*60}")
    print(f"  Accuracy (all):        {acc:.4f}")
    print(f"  Macro-F1 (all):        {f1:.4f}")

    # Known-classes-only metrics: only consider the 4 shared cell types
    # Predictions of CL clusters are treated as misclassifications
    if known_class_indices is not None:
        known_f1 = f1_score(y_true, y_pred, average="macro",
                           labels=known_class_indices, zero_division=0)
        known_wf1 = f1_score(y_true, y_pred, average="weighted",
                             labels=known_class_indices, zero_division=0)
        print(f"  Macro-F1 (known 4):    {known_f1:.4f}")
        print(f"  Weighted-F1 (known 4): {known_wf1:.4f}")
    else:
        known_f1 = f1

    # Diagnostic: predicted class distribution
    pred_counts = np.bincount(y_pred, minlength=len(class_names))
    true_counts = np.bincount(y_true, minlength=len(class_names))
    print(f"\n  Predicted class distribution:")
    for i, cn in enumerate(class_names):
        if pred_counts[i] > 0 or true_counts[i] > 0:
            print(f"    {cn:12s}: pred={pred_counts[i]:5d}  true={true_counts[i]:5d}")

    # Only use class names that actually appear in the data
    present_labels = sorted(set(y_true) | set(y_pred))
    present_names = [class_names[i] for i in present_labels if i < len(class_names)]
    report = classification_report(
        y_true, y_pred,
        labels=present_labels,
        target_names=present_names,
        zero_division=0,
    )
    print(report)

    # Save
    np.save(os.path.join(outdir, f"{name.lower()}_ext_pred.npy"), y_pred)
    np.save(os.path.join(outdir, f"{name.lower()}_ext_probs.npy"), y_probs)
    np.save(os.path.join(outdir, f"{name.lower()}_ext_true.npy"), y_true)

    # Save report as text
    with open(os.path.join(outdir, f"{name.lower()}_ext_report.txt"), "w") as f:
        f.write(f"{name} — External Validation (PBMC 8K)\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Macro-F1 (all): {f1:.4f}\n")
        f.write(f"Macro-F1 (known): {known_f1:.4f}\n\n")
        f.write(report)

    return {
        "Model": name,
        "Accuracy": float(acc),
        "Macro-F1 (all)": float(f1),
        "Macro-F1 (known)": float(known_f1),
    }


def evaluate_coarse(name, y_true_coarse, y_pred_fine, fine_to_coarse,
                    coarse_names, coarse_known_mask, outdir,
                    probs_fine=None):
    """Remap fine-grained predictions to coarse labels and evaluate.

    If probs_fine is provided (n_cells × 14), we aggregate probabilities
    by summing over fine classes that map to the same coarse class, then
    argmax over the coarse probabilities. This is much better than
    argmax-then-remap because e.g. P(T cells) = P(CD8 T) + P(CL 0) + ...
    can outrank P(NK) even when no single T subtype does.

    Only computes macro-F1 over classes with support > 0 in the ground truth.
    """
    if probs_fine is not None:
        # Aggregate fine → coarse probabilities
        n_coarse = len(coarse_names)
        probs_coarse = np.zeros((probs_fine.shape[0], n_coarse), dtype=np.float32)
        for fine_idx, coarse_idx in fine_to_coarse.items():
            probs_coarse[:, coarse_idx] += probs_fine[:, fine_idx]
        y_pred_coarse = probs_coarse.argmax(axis=1)
    else:
        # Fallback: argmax-then-remap
        y_pred_coarse = np.array([fine_to_coarse[p] for p in y_pred_fine])

    yt = y_true_coarse[coarse_known_mask]
    yp = y_pred_coarse[coarse_known_mask]

    acc = accuracy_score(yt, yp)

    # Only evaluate over classes with actual ground-truth support
    supported_labels = sorted(set(yt))
    supported_names = [coarse_names[i] for i in supported_labels]

    f1_macro = f1_score(yt, yp, average="macro", labels=supported_labels, zero_division=0)
    f1_weighted = f1_score(yt, yp, average="weighted", labels=supported_labels, zero_division=0)

    report = classification_report(
        yt, yp, labels=supported_labels, target_names=supported_names, zero_division=0,
    )

    print(f"\n  ── {name} COARSE ({len(supported_labels)}-class, supported only) ──")
    print(f"  Accuracy:     {acc:.4f}")
    print(f"  Macro-F1:     {f1_macro:.4f}")
    print(f"  Weighted-F1:  {f1_weighted:.4f}")
    print(report)

    # Save coarse report
    with open(os.path.join(outdir, f"{name.lower()}_ext_coarse_report.txt"), "w") as f:
        f.write(f"{name} — Coarse External Validation (PBMC 8K)\n")
        f.write(f"Classes evaluated: {supported_names}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Macro-F1: {f1_macro:.4f}\n")
        f.write(f"Weighted-F1: {f1_weighted:.4f}\n\n")
        f.write(report)

    return {
        "Model": name,
        "Accuracy": float(acc),
        "Macro-F1": float(f1_macro),
        "Weighted-F1": float(f1_weighted),
    }


def load_raw_8k_lognorm(raw_dir, hvg_genes):
    """
    Reload 8K raw data, QC-filter, normalize (CPM + log1p), and align
    to the 68K HVG gene set. Returns log-normalized expression BEFORE z-scoring.
    """
    print("  Loading 8K raw data → CPM → log1p (no z-scoring)...")
    adata_raw = sc.read_10x_mtx(raw_dir, var_names="gene_symbols", make_unique=True)

    # QC filtering (same as preprocessing)
    X = adata_raw.X
    if sp.issparse(X):
        genes_per_cell = np.array((X > 0).sum(axis=1)).flatten()
        total_counts = np.array(X.sum(axis=1)).flatten()
    else:
        genes_per_cell = (X > 0).sum(axis=1)
        total_counts = X.sum(axis=1)

    mt_mask = adata_raw.var_names.str.upper().str.startswith("MT-")
    if sp.issparse(X):
        mt_counts = np.array(X[:, mt_mask].sum(axis=1)).flatten()
    else:
        mt_counts = X[:, mt_mask].sum(axis=1)
    pct_mt = mt_counts / (total_counts + 1e-8) * 100

    cell_mask = (genes_per_cell >= 200) & (pct_mt < 10.0)
    adata_raw = adata_raw[cell_mask].copy()

    if sp.issparse(adata_raw.X):
        cells_per_gene = np.array((adata_raw.X > 0).sum(axis=0)).flatten()
    else:
        cells_per_gene = (adata_raw.X > 0).sum(axis=0)
    adata_raw = adata_raw[:, cells_per_gene >= 3].copy()

    # Normalize + log (same as 68K pipeline)
    sc.pp.normalize_total(adata_raw, target_sum=1e4)
    sc.pp.log1p(adata_raw)

    # Align to 68K HVG gene set
    shared = [g for g in hvg_genes if g in adata_raw.var_names]
    missing = [g for g in hvg_genes if g not in adata_raw.var_names]
    print(f"  Gene alignment: {len(shared)}/{len(hvg_genes)} shared, {len(missing)} missing (zero-filled)")

    adata_aligned = adata_raw[:, shared].copy()

    # Zero-fill missing genes
    if missing:
        import anndata
        n_cells = adata_aligned.shape[0]
        zero_block = sp.csr_matrix((n_cells, len(missing)), dtype=np.float32)
        zero_adata = anndata.AnnData(X=zero_block, obs=adata_aligned.obs.copy())
        zero_adata.var_names = missing
        adata_aligned = anndata.concat([adata_aligned, zero_adata], axis=1)

    # Reorder to match 68K gene order
    adata_aligned = adata_aligned[:, hvg_genes].copy()

    X_lognorm = to_dense_float32(adata_aligned.X)
    print(f"  Log-normalized 8K shape: {X_lognorm.shape}")
    return X_lognorm


def main():
    p = argparse.ArgumentParser(description="External validation on PBMC 8K")
    p.add_argument("--data_8k", default="data/processed/pbmc8k_labeled.h5ad")
    p.add_argument("--raw_8k_dir", default="data/raw/pbmc8k/filtered_gene_bc_matrices/GRCh38")
    p.add_argument("--data_68k", default="data/processed/pbmc68k_labeled.h5ad")
    p.add_argument("--hvg_dir", default="results/full_train_all_hvg")
    p.add_argument("--pca_dir", default="results/full_train_all_pca")
    p.add_argument("--outdir", default="results/external_validation")
    p.add_argument("--pca_dim", type=int, default=50)
    p.add_argument("--label_key", type=str, default="cell_type",
                   help="68K obs column the models were trained on (cell_type or cell_type_coarse)")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ----------------------------------------------------------------
    # Load 8K labels from preprocessed file
    # ----------------------------------------------------------------
    print("Loading PBMC 8K labels...")
    adata_8k = sc.read_h5ad(args.data_8k)
    y_true_cat = adata_8k.obs["cell_type"].astype("category")

    print("Loading 68K reference...")
    adata_68k = sc.read_h5ad(args.data_68k)

    # Model class names come from whatever label_key was used for training
    class_names_model = list(adata_68k.obs[args.label_key].astype("category").cat.categories)
    num_classes_model = len(class_names_model)
    print(f"  Model classes ({num_classes_model}): {class_names_model}")

    class_names_8k = list(y_true_cat.cat.categories)
    print(f"  8K classes ({len(class_names_8k)}): {class_names_8k}")

    # Map 8K labels to model class indices
    # For coarse models: 8K "CD8 T" → model "T cells"
    LABEL_ALIAS = {"CD8 T": "T cells"}  # only used if target exists in model classes
    label_map = {}
    for name in class_names_8k:
        if name in class_names_model:
            label_map[name] = class_names_model.index(name)
        elif name in LABEL_ALIAS and LABEL_ALIAS[name] in class_names_model:
            label_map[name] = class_names_model.index(LABEL_ALIAS[name])
        else:
            label_map[name] = -1  # unknown class

    print(f"  Label mapping: {label_map}")

    y_true_mapped = np.array([label_map[n] for n in y_true_cat])
    known_mask = y_true_mapped >= 0
    n_unknown = (~known_mask).sum()
    if n_unknown > 0:
        unknown_types = [n for n in class_names_8k if label_map[n] == -1]
        print(f"  WARNING: {n_unknown} cells have types not in model classes: {unknown_types}")
        print(f"  These cells will be excluded from accuracy/F1 computation.")

    # ----------------------------------------------------------------
    # CRITICAL FIX: Z-score 8K data using 68K's per-gene statistics
    #
    # The 68K preprocessing did: raw → CPM → log1p → sc.pp.scale
    # sc.pp.scale stores per-gene mean/std in adata.var['mean']/var['std']
    # We apply THOSE statistics to the 8K log-normalized data so it's
    # in the exact same feature space the models were trained on.
    # ----------------------------------------------------------------
    hvg_genes = list(adata_68k.var_names)

    # Extract 68K's per-gene scaling statistics
    ref_gene_mean = adata_68k.var["mean"].values.astype(np.float32)  # shape (2000,)
    ref_gene_std = adata_68k.var["std"].values.astype(np.float32)    # shape (2000,)
    ref_gene_std[ref_gene_std == 0] = 1.0  # avoid division by zero

    # Load 8K raw log-normalized expression (aligned to 68K gene order)
    X_8k_lognorm = load_raw_8k_lognorm(args.raw_8k_dir, hvg_genes)

    # Z-score 8K using 68K's statistics (same transform as sc.pp.scale on 68K)
    X_8k_zscored = ((X_8k_lognorm - ref_gene_mean) / ref_gene_std).astype(np.float32)
    X_8k_zscored = np.clip(X_8k_zscored, -10, 10)

    print(f"\n  8K z-scored with 68K stats:")
    print(f"    shape: {X_8k_zscored.shape}")
    print(f"    mean:  {X_8k_zscored.mean():.4f}  (68K was ~-0.012)")
    print(f"    std:   {X_8k_zscored.std():.4f}  (68K was ~0.553)")

    # Binary sparsity mask from z-scored data (consistent with training)
    X_mask_zscored = (X_8k_zscored != 0).astype(np.float32)
    print(f"    mask fraction non-zero: {X_mask_zscored.mean():.4f}")

    # Also keep the number of cells for known-only evaluation
    n_total = X_8k_zscored.shape[0]
    n_known = known_mask.sum()
    print(f"\n  Cells: {n_total} total, {n_known} with known types, {n_unknown} excluded")

    # Indices of classes that exist in 8K ground truth (for fairer macro-F1)
    known_class_indices = sorted([label_map[n] for n in class_names_8k if label_map[n] >= 0])
    known_class_names = [class_names_model[i] for i in known_class_indices]
    print(f"  Known shared classes: {known_class_names} (indices {known_class_indices})")

    # ----------------------------------------------------------------
    # Coarse label remapping (only needed for fine-grained 14-class models)
    # If models were trained with cell_type_coarse, they already output 5 classes.
    # ----------------------------------------------------------------
    is_coarse_model = (num_classes_model <= 5)

    if not is_coarse_model:
        COARSE_NAMES = ["B cells", "Mono", "NK", "Platelet", "T cells"]

        fine_to_coarse = {}
        for i, name in enumerate(class_names_model):
            if name in COARSE_NAMES:
                fine_to_coarse[i] = COARSE_NAMES.index(name)
            else:
                fine_to_coarse[i] = COARSE_NAMES.index("T cells")

        print(f"\n  Coarse label mapping ({num_classes_model}-class → 5 types):")
        for i, name in enumerate(class_names_model):
            print(f"    {name:12s} (idx {i:2d}) → {COARSE_NAMES[fine_to_coarse[i]]}")

        coarse_8k_map = {
            "B cells": COARSE_NAMES.index("B cells"),
            "CD8 T": COARSE_NAMES.index("T cells"),
            "Mono": COARSE_NAMES.index("Mono"),
            "NK": COARSE_NAMES.index("NK"),
        }
        y_true_coarse = np.array([coarse_8k_map.get(n, -1) for n in y_true_cat])
        coarse_known_mask = y_true_coarse >= 0
    else:
        print(f"\n  Models already trained with coarse labels — no remapping needed.")

    # ================================================================
    # HVG-based models
    # ================================================================
    print("\n" + "=" * 60)
    print("  HVG-based model evaluation")
    print("=" * 60)

    hvg_outdir = os.path.join(args.outdir, "hvg")
    os.makedirs(hvg_outdir, exist_ok=True)

    # --- LR (HVG) ---
    # Training: LR scaler was fit on 68K z-scored expression, model trained on that
    print("\nEvaluating LR (HVG)...")
    lr_model = joblib.load(os.path.join(args.hvg_dir, "lr_model.pkl"))
    lr_scaler = joblib.load(os.path.join(args.hvg_dir, "lr_scaler.pkl"))
    X_lr = lr_scaler.transform(X_8k_zscored)
    lr_probs = lr_model.predict_proba(X_lr)
    lr_pred = lr_probs.argmax(axis=1)
    lr_metrics = evaluate_model(
        "LR_HVG", y_true_mapped[known_mask], lr_pred[known_mask],
        lr_probs[known_mask], class_names_model, hvg_outdir, known_class_indices,
    )

    # --- XGBoost (HVG) ---
    # Training: XGB used 68K z-scored expression directly (no scaler)
    print("\nEvaluating XGBoost (HVG)...")
    booster = xgb.Booster()
    booster.load_model(os.path.join(args.hvg_dir, "xgb_model.json"))
    dtest = xgb.DMatrix(X_8k_zscored)
    xgb_probs = booster.predict(dtest)
    xgb_pred = xgb_probs.argmax(axis=1)
    xgb_metrics = evaluate_model(
        "XGB_HVG", y_true_mapped[known_mask], xgb_pred[known_mask],
        xgb_probs[known_mask], class_names_model, hvg_outdir, known_class_indices,
    )

    # --- SANN (HVG) ---
    # Training: SANN applied additional standardization (sann_expr_mean/std)
    # on top of z-scored data, mask from z-scored data
    print("\nEvaluating SANN (HVG)...")
    # SANN v2: no double standardization — pass z-scored expression directly
    # (same input space as LR/XGB, internal LayerNorm handles normalization)
    X_expr_for_sann = X_8k_zscored.copy()
    X_sann = np.concatenate([X_expr_for_sann, X_mask_zscored], axis=1).astype(np.float32)

    print(f"  SANN input shape: {X_sann.shape}")
    print(f"  Expression branch stats: mean={X_expr_for_sann.mean():.4f}, std={X_expr_for_sann.std():.4f}")

    # HVG SANN: LayerNorm, 256→128 branches (deep ensemble if multiple seeds)
    def make_sann_hvg():
        return SANN(
            expr_dim=2000, mask_dim=2000, num_classes=num_classes_model,
            branch_hidden=256, branch_out=128, fusion_hidden=128,
            dropout=0.4, use_batchnorm=False, use_layernorm=True,
        )
    sann_probs = load_sann_ensemble(args.hvg_dir, make_sann_hvg, X_sann)
    sann_pred = sann_probs.argmax(axis=1)
    sann_hvg_metrics = evaluate_model(
        "SANN_HVG", y_true_mapped[known_mask], sann_pred[known_mask],
        sann_probs[known_mask], class_names_model, hvg_outdir, known_class_indices,
    )

    # ================================================================
    # PCA-based models
    # ================================================================
    print("\n" + "=" * 60)
    print("  PCA-based model evaluation")
    print("=" * 60)

    pca_outdir = os.path.join(args.outdir, "pca")
    os.makedirs(pca_outdir, exist_ok=True)

    # Project 8K z-scored data through 68K's PCA loadings
    # The 68K PCA was computed by scanpy on the z-scored 68K data.
    # We need to center using the 68K z-scored column means, then multiply by PCs.
    X_68k_zscored = to_dense_float32(adata_68k.X)
    pca_center = X_68k_zscored.mean(axis=0)  # per-gene mean of z-scored 68K
    PCs = adata_68k.varm["PCs"][:, :args.pca_dim]  # (2000, 50)

    X_pca_8k = ((X_8k_zscored - pca_center) @ PCs).astype(np.float32)
    print(f"\n  8K projected through 68K PCA: {X_pca_8k.shape}")
    print(f"    PCA stats: mean={X_pca_8k.mean():.4f}, std={X_pca_8k.std():.4f}")

    # Mask PCA: compute mask from z-scored data, project through saved mask PCA
    mask_pca_model = joblib.load(os.path.join(args.pca_dir, "mask_pca_model.pkl"))
    X_mask_pca = mask_pca_model.transform(X_mask_zscored).astype(np.float32)
    print(f"  8K mask PCA: {X_mask_pca.shape}")

    # --- LR (PCA) ---
    # Training: LR scaler was fit on 68K PCA features
    print("\nEvaluating LR (PCA)...")
    lr_model_pca = joblib.load(os.path.join(args.pca_dir, "lr_model.pkl"))
    lr_scaler_pca = joblib.load(os.path.join(args.pca_dir, "lr_scaler.pkl"))
    X_lr_pca = lr_scaler_pca.transform(X_pca_8k)
    lr_probs_pca = lr_model_pca.predict_proba(X_lr_pca)
    lr_pred_pca = lr_probs_pca.argmax(axis=1)
    lr_pca_metrics = evaluate_model(
        "LR_PCA", y_true_mapped[known_mask], lr_pred_pca[known_mask],
        lr_probs_pca[known_mask], class_names_model, pca_outdir, known_class_indices,
    )

    # --- XGBoost (PCA) ---
    # Training: XGB used PCA features directly (no additional scaler)
    print("\nEvaluating XGBoost (PCA)...")
    booster_pca = xgb.Booster()
    booster_pca.load_model(os.path.join(args.pca_dir, "xgb_model.json"))
    dtest_pca = xgb.DMatrix(X_pca_8k)
    xgb_probs_pca = booster_pca.predict(dtest_pca)
    xgb_pred_pca = xgb_probs_pca.argmax(axis=1)
    xgb_pca_metrics = evaluate_model(
        "XGB_PCA", y_true_mapped[known_mask], xgb_pred_pca[known_mask],
        xgb_probs_pca[known_mask], class_names_model, pca_outdir, known_class_indices,
    )

    # --- SANN (PCA) ---
    # Training: PCA SANN used [expression_PCA, mask_PCA] directly (no sann_expr_mean/std saved)
    # The LR scaler for PCA was a StandardScaler — SANN also needs standardization.
    # We'll use the LR scaler stats as a proxy (same training data).
    print("\nEvaluating SANN (PCA)...")
    pca_mean_path = os.path.join(args.pca_dir, "sann_expr_mean.npy")
    if os.path.exists(pca_mean_path):
        pca_expr_mean = np.load(pca_mean_path)
        pca_expr_std = np.load(os.path.join(args.pca_dir, "sann_expr_std.npy"))
        pca_expr_std = np.where(pca_expr_std < 1e-6, 1.0, pca_expr_std)
        X_pca_for_sann = ((X_pca_8k - pca_expr_mean) / pca_expr_std).astype(np.float32)
    else:
        # No saved SANN stats for PCA — use PCA features directly
        # (PCA training script didn't save sann_expr_mean/std, so the SANN
        # was trained on raw PCA features without additional standardization)
        X_pca_for_sann = X_pca_8k

    X_sann_pca = np.concatenate([X_pca_for_sann, X_mask_pca], axis=1).astype(np.float32)
    print(f"  SANN PCA input shape: {X_sann_pca.shape}")

    # PCA SANN v2: BatchNorm, 512→256 branches (deep ensemble if multiple seeds)
    def make_sann_pca():
        return SANN(
            expr_dim=args.pca_dim, mask_dim=args.pca_dim, num_classes=num_classes_model,
            branch_hidden=512, branch_out=256, fusion_hidden=256,
            dropout=0.25, use_batchnorm=True, use_layernorm=False,
        )
    sann_probs_pca = load_sann_ensemble(args.pca_dir, make_sann_pca, X_sann_pca)
    sann_pred_pca = sann_probs_pca.argmax(axis=1)
    sann_pca_metrics = evaluate_model(
        "SANN_PCA", y_true_mapped[known_mask], sann_pred_pca[known_mask],
        sann_probs_pca[known_mask], class_names_model, pca_outdir, known_class_indices,
    )

    # ================================================================
    # COARSE post-hoc evaluation (only for fine-grained 14-class models)
    # ================================================================
    if not is_coarse_model:
        print("\n" + "=" * 60)
        print("  COARSE 5-CLASS EVALUATION (post-hoc probability aggregation)")
        print("=" * 60)

        coarse_results = []
        for model_name, preds, probs in [
            ("LR_HVG", lr_pred, lr_probs),
            ("XGB_HVG", xgb_pred, xgb_probs),
            ("SANN_HVG", sann_pred, sann_probs),
            ("LR_PCA", lr_pred_pca, lr_probs_pca),
            ("XGB_PCA", xgb_pred_pca, xgb_probs_pca),
            ("SANN_PCA", sann_pred_pca, sann_probs_pca),
        ]:
            out = hvg_outdir if "HVG" in model_name else pca_outdir
            row = evaluate_coarse(
                model_name, y_true_coarse, preds, fine_to_coarse,
                COARSE_NAMES, coarse_known_mask, out,
                probs_fine=probs,
            )
            coarse_results.append(row)

        summary_coarse = pd.DataFrame(coarse_results)
        coarse_path = os.path.join(args.outdir, "external_validation_coarse_summary.csv")
        summary_coarse.to_csv(coarse_path, index=False)

    # ================================================================
    # Summary
    # ================================================================
    all_metrics = [
        lr_metrics, xgb_metrics, sann_hvg_metrics,
        lr_pca_metrics, xgb_pca_metrics, sann_pca_metrics,
    ]
    summary = pd.DataFrame(all_metrics)
    summary_path = os.path.join(args.outdir, "external_validation_summary.csv")
    summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 60)
    print(f"  SUMMARY — External Validation ({num_classes_model}-class models)")
    print("=" * 60)
    print(summary.to_string(index=False))

    if not is_coarse_model:
        print("\n" + "=" * 60)
        print("  COARSE (5-class post-hoc) SUMMARY")
        print("=" * 60)
        print(summary_coarse.to_string(index=False))

    print(f"\nSaved: {summary_path}")
    print(f"Saved artifacts: {args.outdir}/")


if __name__ == "__main__":
    main()
