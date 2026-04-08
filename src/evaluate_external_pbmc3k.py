# src/evaluate_external_pbmc3k.py
"""
External validation: evaluate 68K-trained models on PBMC 3K.

The 3K data is z-scored using 68K per-gene statistics (same as training),
then passed through each model. For PCA models, the 3K is projected
through the 68K PCA loadings.
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
    """Load all SANN seed models and return averaged softmax probabilities."""
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
        m = model_class()
        m.load_state_dict(torch.load(
            os.path.join(model_dir, model_filename), map_location="cpu"
        ))
        m.eval()
        with torch.no_grad():
            logits = m(torch.tensor(X_input, dtype=torch.float32)).numpy()
        return torch.softmax(torch.tensor(logits), dim=1).numpy()


# ----------------------------
# SANN v2 architecture (must match training)
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
def evaluate_model(name, y_true, y_pred, y_probs, class_names, outdir,
                   known_class_indices=None):
    acc = accuracy_score(y_true, y_pred)
    f1_all = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"\n{'='*60}")
    print(f"  {name} — External Validation (PBMC 3K)")
    print(f"{'='*60}")
    print(f"  Accuracy:     {acc:.4f}")
    print(f"  Macro-F1:     {f1_all:.4f}")

    if known_class_indices is not None:
        f1_known = f1_score(y_true, y_pred, average="macro",
                            labels=known_class_indices, zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average="weighted",
                               labels=known_class_indices, zero_division=0)
        print(f"  Macro-F1 (known):    {f1_known:.4f}")
        print(f"  Weighted-F1 (known): {f1_weighted:.4f}")
    else:
        f1_known = f1_all
        f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # Predicted class distribution
    pred_counts = np.bincount(y_pred, minlength=len(class_names))
    true_counts = np.bincount(y_true, minlength=len(class_names))
    print(f"\n  Predicted class distribution:")
    for i, cn in enumerate(class_names):
        if pred_counts[i] > 0 or true_counts[i] > 0:
            print(f"    {cn:12s}: pred={pred_counts[i]:5d}  true={true_counts[i]:5d}")

    present_labels = sorted(set(y_true) | set(y_pred))
    present_names = [class_names[i] for i in present_labels if i < len(class_names)]
    report = classification_report(
        y_true, y_pred, labels=present_labels,
        target_names=present_names, zero_division=0,
    )
    print(report)

    # Save artifacts
    np.save(os.path.join(outdir, f"{name.lower()}_ext3k_pred.npy"), y_pred)
    np.save(os.path.join(outdir, f"{name.lower()}_ext3k_probs.npy"), y_probs)
    np.save(os.path.join(outdir, f"{name.lower()}_ext3k_true.npy"), y_true)

    with open(os.path.join(outdir, f"{name.lower()}_ext3k_report.txt"), "w") as f:
        f.write(f"{name} — External Validation (PBMC 3K)\n")
        f.write(f"Accuracy: {acc:.4f}\nMacro-F1: {f1_all:.4f}\n")
        f.write(f"Macro-F1 (known): {f1_known:.4f}\n\n{report}")

    return {
        "Model": name,
        "Accuracy": float(acc),
        "Macro-F1": float(f1_all),
        "Macro-F1 (known)": float(f1_known),
        "Weighted-F1": float(f1_weighted),
    }


def main():
    p = argparse.ArgumentParser(description="External validation on PBMC 3K")
    p.add_argument("--data_3k", default="data/processed/pbmc3k_labeled.h5ad")
    p.add_argument("--ref", default="data/processed/pbmc68k_labeled.h5ad",
                   help="68K training h5ad (for gene list and z-score stats)")
    p.add_argument("--hvg_dir", default="results/full_train_all_hvg")
    p.add_argument("--pca_dir", default="results/full_train_all_pca")
    p.add_argument("--outdir", default="results/external_validation_3k")
    p.add_argument("--pca_dim", type=int, default=50)
    p.add_argument("--label_key", type=str, default="cell_type",
                   help="68K obs column the models were trained on (cell_type or cell_type_coarse)")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ----------------------------------------------------------------
    # Load 3K labels
    # ----------------------------------------------------------------
    print("Loading PBMC 3K data...")
    adata_3k = sc.read_h5ad(args.data_3k)
    print(f"  Shape: {adata_3k.shape}")
    print(f"  Cell types: {dict(adata_3k.obs['cell_type_coarse'].value_counts())}")

    # ----------------------------------------------------------------
    # Load reference (68K training data)
    # ----------------------------------------------------------------
    print("\nLoading 68K reference...")
    adata_ref = sc.read_h5ad(args.ref)
    class_names = list(adata_ref.obs[args.label_key].astype("category").cat.categories)
    num_classes = len(class_names)
    print(f"  Model classes ({num_classes}): {class_names}")

    hvg_genes = list(adata_ref.var_names)  # 2000 HVG genes

    # ----------------------------------------------------------------
    # Map 3K labels to model class indices
    # ----------------------------------------------------------------
    LABEL_ALIAS = {"CD8 T": "T cells"}  # 3K may use different names
    y_true_cat = adata_3k.obs["cell_type_coarse"]
    label_map = {}
    for name in y_true_cat.unique():
        if name in class_names:
            label_map[name] = class_names.index(name)
        elif name in LABEL_ALIAS and LABEL_ALIAS[name] in class_names:
            label_map[name] = class_names.index(LABEL_ALIAS[name])
        else:
            label_map[name] = -1
    print(f"  Label mapping: {label_map}")

    y_true_mapped = np.array([label_map[n] for n in y_true_cat])
    known_mask = y_true_mapped >= 0
    n_unknown = (~known_mask).sum()
    if n_unknown > 0:
        unknown_types = [n for n in y_true_cat.unique() if label_map[n] == -1]
        print(f"  WARNING: {n_unknown} cells excluded (types: {unknown_types})")

    n_known = known_mask.sum()
    print(f"  Cells: {len(y_true_mapped)} total, {n_known} with known types")

    known_class_indices = sorted([label_map[n] for n in y_true_cat.unique() if label_map[n] >= 0])
    known_class_names = [class_names[i] for i in known_class_indices]
    print(f"  Known shared classes: {known_class_names}")

    # ----------------------------------------------------------------
    # Prepare 3K expression: align genes + z-score with 68K stats
    # ----------------------------------------------------------------
    print("\nPreparing 3K expression data...")
    ref_mean = adata_ref.var["mean"].values.astype(np.float32)
    ref_std = adata_ref.var["std"].values.astype(np.float32)
    ref_std[ref_std == 0] = 1.0

    # Reload 3K raw counts → normalize with target_sum=1e4 → log1p
    print("  Reloading raw 3K counts → CPM(1e4) → log1p ...")
    adata_3k_raw = sc.read_h5ad("data/raw/pbmc3k/pbmc3k_raw.h5ad")
    adata_3k_raw = adata_3k_raw[adata_3k.obs_names].copy()
    sc.pp.normalize_total(adata_3k_raw, target_sum=1e4)
    sc.pp.log1p(adata_3k_raw)
    X_3k_lognorm = adata_3k_raw.X
    genes_3k_raw = list(adata_3k_raw.var_names)

    if sp.issparse(X_3k_lognorm):
        X_3k_lognorm = X_3k_lognorm.toarray()

    # Align to HVG genes
    shared = [g for g in hvg_genes if g in genes_3k_raw]
    missing = [g for g in hvg_genes if g not in genes_3k_raw]
    print(f"  Gene alignment: {len(shared)}/{len(hvg_genes)} shared, {len(missing)} missing")

    # Build aligned matrix
    X_aligned = np.zeros((adata_3k.shape[0], len(hvg_genes)), dtype=np.float32)
    for i, g in enumerate(hvg_genes):
        if g in genes_3k_raw:
            j = genes_3k_raw.index(g)
            col = X_3k_lognorm[:, j]
            X_aligned[:, i] = col.flatten()

    # Z-score with 68K stats
    X_3k_zscored = np.clip(((X_aligned - ref_mean) / ref_std), -10, 10).astype(np.float32)
    print(f"  Z-scored: mean={X_3k_zscored.mean():.4f}, std={X_3k_zscored.std():.4f}")

    # Sparsity mask from z-scored data (consistent with training)
    X_mask = (X_3k_zscored != 0).astype(np.float32)
    print(f"  Mask non-zero fraction: {X_mask.mean():.4f}")

    # ================================================================
    # HVG models
    # ================================================================
    print("\n" + "=" * 60)
    print("  HVG-based model evaluation")
    print("=" * 60)

    hvg_outdir = os.path.join(args.outdir, "hvg")
    os.makedirs(hvg_outdir, exist_ok=True)

    # --- LR ---
    print("\nEvaluating LR (HVG)...")
    lr_model = joblib.load(os.path.join(args.hvg_dir, "lr_model.pkl"))
    lr_scaler = joblib.load(os.path.join(args.hvg_dir, "lr_scaler.pkl"))
    X_lr = lr_scaler.transform(X_3k_zscored)
    lr_probs = lr_model.predict_proba(X_lr)
    lr_pred = lr_probs.argmax(axis=1)
    lr_metrics = evaluate_model(
        "LR_HVG", y_true_mapped[known_mask], lr_pred[known_mask],
        lr_probs[known_mask], class_names, hvg_outdir, known_class_indices,
    )

    # --- XGBoost ---
    print("\nEvaluating XGBoost (HVG)...")
    booster = xgb.Booster()
    booster.load_model(os.path.join(args.hvg_dir, "xgb_model.json"))
    dtest = xgb.DMatrix(X_3k_zscored)
    xgb_probs = booster.predict(dtest)
    xgb_pred = xgb_probs.argmax(axis=1)
    xgb_metrics = evaluate_model(
        "XGB_HVG", y_true_mapped[known_mask], xgb_pred[known_mask],
        xgb_probs[known_mask], class_names, hvg_outdir, known_class_indices,
    )

    # --- SANN (HVG) ---
    print("\nEvaluating SANN (HVG)...")
    # SANN v2: no double standardization — pass z-scored expression directly
    X_sann = np.concatenate([X_3k_zscored, X_mask], axis=1).astype(np.float32)
    print(f"  SANN input: {X_sann.shape}")

    def make_sann_hvg():
        return SANN(
            expr_dim=2000, mask_dim=2000, num_classes=num_classes,
            branch_hidden=256, branch_out=128, fusion_hidden=128,
            dropout=0.4, use_batchnorm=False, use_layernorm=True,
        )
    sann_probs = load_sann_ensemble(args.hvg_dir, make_sann_hvg, X_sann)
    sann_pred = sann_probs.argmax(axis=1)
    sann_hvg_metrics = evaluate_model(
        "SANN_HVG", y_true_mapped[known_mask], sann_pred[known_mask],
        sann_probs[known_mask], class_names, hvg_outdir, known_class_indices,
    )

    # ================================================================
    # PCA models
    # ================================================================
    print("\n" + "=" * 60)
    print("  PCA-based model evaluation")
    print("=" * 60)

    pca_outdir = os.path.join(args.outdir, "pca")
    os.makedirs(pca_outdir, exist_ok=True)

    # Project 3K through 68K PCA loadings
    pca_loadings = adata_ref.varm["PCs"][:, :args.pca_dim]  # (2000, 50)
    X_pca_3k = (X_3k_zscored @ pca_loadings).astype(np.float32)
    print(f"\n  3K projected through 68K PCA: {X_pca_3k.shape}")
    print(f"    PCA stats: mean={X_pca_3k.mean():.4f}, std={X_pca_3k.std():.4f}")

    # Mask PCA
    mask_pca = joblib.load(os.path.join(args.pca_dir, "mask_pca_model.pkl"))
    X_mask_pca = mask_pca.transform(X_mask).astype(np.float32)
    print(f"  3K mask PCA: {X_mask_pca.shape}")

    # --- LR (PCA) ---
    print("\nEvaluating LR (PCA)...")
    lr_model_pca = joblib.load(os.path.join(args.pca_dir, "lr_model.pkl"))
    lr_scaler_pca = joblib.load(os.path.join(args.pca_dir, "lr_scaler.pkl"))
    X_lr_pca = lr_scaler_pca.transform(X_pca_3k)
    lr_probs_pca = lr_model_pca.predict_proba(X_lr_pca)
    lr_pred_pca = lr_probs_pca.argmax(axis=1)
    lr_pca_metrics = evaluate_model(
        "LR_PCA", y_true_mapped[known_mask], lr_pred_pca[known_mask],
        lr_probs_pca[known_mask], class_names, pca_outdir, known_class_indices,
    )

    # --- XGBoost (PCA) ---
    print("\nEvaluating XGBoost (PCA)...")
    booster_pca = xgb.Booster()
    booster_pca.load_model(os.path.join(args.pca_dir, "xgb_model.json"))
    dtest_pca = xgb.DMatrix(X_pca_3k)
    xgb_probs_pca = booster_pca.predict(dtest_pca)
    xgb_pred_pca = xgb_probs_pca.argmax(axis=1)
    xgb_pca_metrics = evaluate_model(
        "XGB_PCA", y_true_mapped[known_mask], xgb_pred_pca[known_mask],
        xgb_probs_pca[known_mask], class_names, pca_outdir, known_class_indices,
    )

    # --- SANN (PCA) ---
    print("\nEvaluating SANN (PCA)...")
    pca_mean_path = os.path.join(args.pca_dir, "sann_expr_mean.npy")
    if os.path.exists(pca_mean_path):
        pca_expr_mean = np.load(pca_mean_path)
        pca_expr_std = np.load(os.path.join(args.pca_dir, "sann_expr_std.npy"))
        pca_expr_std = np.where(pca_expr_std < 1e-6, 1.0, pca_expr_std)
        X_pca_for_sann = ((X_pca_3k - pca_expr_mean) / pca_expr_std).astype(np.float32)
    else:
        X_pca_for_sann = X_pca_3k

    X_sann_pca = np.concatenate([X_pca_for_sann, X_mask_pca], axis=1).astype(np.float32)
    print(f"  SANN PCA input: {X_sann_pca.shape}")

    def make_sann_pca():
        return SANN(
            expr_dim=args.pca_dim, mask_dim=args.pca_dim, num_classes=num_classes,
            branch_hidden=512, branch_out=256, fusion_hidden=256,
            dropout=0.25, use_batchnorm=True, use_layernorm=False,
        )
    sann_probs_pca = load_sann_ensemble(args.pca_dir, make_sann_pca, X_sann_pca)
    sann_pred_pca = sann_probs_pca.argmax(axis=1)
    sann_pca_metrics = evaluate_model(
        "SANN_PCA", y_true_mapped[known_mask], sann_pred_pca[known_mask],
        sann_probs_pca[known_mask], class_names, pca_outdir, known_class_indices,
    )

    # ================================================================
    # Summary
    # ================================================================
    all_metrics = [
        lr_metrics, xgb_metrics, sann_hvg_metrics,
        lr_pca_metrics, xgb_pca_metrics, sann_pca_metrics,
    ]
    summary = pd.DataFrame(all_metrics)
    summary_path = os.path.join(args.outdir, "external_validation_3k_summary.csv")
    summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 60)
    print(f"  SUMMARY — External Validation on PBMC 3K ({num_classes}-class)")
    print("=" * 60)
    print(summary.to_string(index=False))

    print(f"\nSaved: {summary_path}")
    print(f"Saved artifacts: {args.outdir}/")


if __name__ == "__main__":
    main()
