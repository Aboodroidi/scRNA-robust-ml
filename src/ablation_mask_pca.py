"""
Ablation: SANN_PCA with mask vs SANN_PCA without mask (no_mask).
Trains 3-seed ensemble for both, evaluates on 68K test set + 8K + 3K external.
"""
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MPLBACKEND"] = "Agg"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import time
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ══════════════════════════════════════════
# SANN v2 architecture (same as train_all_pca_models.py)
# ══════════════════════════════════════════
def _make_norm(dim, use_batchnorm=True):
    if use_batchnorm:
        return nn.BatchNorm1d(dim)
    return nn.LayerNorm(dim)


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1, use_batchnorm=True):
        super().__init__()
        self.norm = _make_norm(dim, use_batchnorm)
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
    """Full dual-encoder SANN with mask branch."""
    def __init__(self, expr_dim, mask_dim, num_classes,
                 branch_hidden=512, branch_out=256, fusion_hidden=256,
                 dropout=0.25, use_batchnorm=True, input_noise=0.05):
        super().__init__()
        self.expr_dim = expr_dim
        self.input_noise = input_noise

        self.expr_proj = nn.Sequential(
            nn.Linear(expr_dim, branch_hidden), _make_norm(branch_hidden, use_batchnorm),
            nn.GELU(), nn.Dropout(dropout) if dropout > 0 else nn.Identity())
        self.expr_res1 = ResidualBlock(branch_hidden, dropout, use_batchnorm)
        self.expr_res2 = ResidualBlock(branch_hidden, dropout, use_batchnorm)
        self.expr_out = nn.Sequential(
            nn.Linear(branch_hidden, branch_out), _make_norm(branch_out, use_batchnorm), nn.GELU())

        self.mask_proj = nn.Sequential(
            nn.Linear(mask_dim, branch_hidden), _make_norm(branch_hidden, use_batchnorm),
            nn.GELU(), nn.Dropout(dropout) if dropout > 0 else nn.Identity())
        self.mask_res1 = ResidualBlock(branch_hidden, dropout, use_batchnorm)
        self.mask_res2 = ResidualBlock(branch_hidden, dropout, use_batchnorm)
        self.mask_out = nn.Sequential(
            nn.Linear(branch_hidden, branch_out), _make_norm(branch_out, use_batchnorm), nn.GELU())

        self.gate = nn.Sequential(
            nn.Linear(branch_out * 2, branch_out), nn.GELU(),
            nn.Linear(branch_out, branch_out), nn.Sigmoid())

        self.classifier = nn.Sequential(
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(branch_out, fusion_hidden), _make_norm(fusion_hidden, use_batchnorm),
            nn.GELU(), nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(fusion_hidden, num_classes))

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


class SANN_NoMask(nn.Module):
    """Expression-only SANN — NO mask branch, NO gate. Same capacity in expr path."""
    def __init__(self, expr_dim, num_classes,
                 branch_hidden=512, branch_out=256, fusion_hidden=256,
                 dropout=0.25, use_batchnorm=True, input_noise=0.05):
        super().__init__()
        self.expr_dim = expr_dim
        self.input_noise = input_noise

        self.expr_proj = nn.Sequential(
            nn.Linear(expr_dim, branch_hidden), _make_norm(branch_hidden, use_batchnorm),
            nn.GELU(), nn.Dropout(dropout) if dropout > 0 else nn.Identity())
        self.expr_res1 = ResidualBlock(branch_hidden, dropout, use_batchnorm)
        self.expr_res2 = ResidualBlock(branch_hidden, dropout, use_batchnorm)
        self.expr_out = nn.Sequential(
            nn.Linear(branch_hidden, branch_out), _make_norm(branch_out, use_batchnorm), nn.GELU())

        self.classifier = nn.Sequential(
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(branch_out, fusion_hidden), _make_norm(fusion_hidden, use_batchnorm),
            nn.GELU(), nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(fusion_hidden, num_classes))

    def forward(self, x):
        x_expr = x[:, :self.expr_dim]
        if self.training and self.input_noise > 0:
            x_expr = x_expr + torch.randn_like(x_expr) * self.input_noise

        h = self.expr_proj(x_expr)
        h = self.expr_res1(h)
        h = self.expr_res2(h)
        h = self.expr_out(h)
        return self.classifier(h)


# ══════════════════════════════════════════
# Training helpers
# ══════════════════════════════════════════
def compute_class_weights(y, num_classes):
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    inv_freq = 1.0 / np.sqrt(counts)
    weights = inv_freq / inv_freq.sum() * num_classes
    return weights.astype(np.float32)


def l1_penalty(model):
    penalty = torch.tensor(0.0)
    for param in model.parameters():
        penalty = penalty + param.abs().sum()
    return penalty


def mixup_batch(xb, yb, alpha, num_classes):
    if alpha <= 0:
        return xb, torch.nn.functional.one_hot(yb, num_classes).float()
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)
    idx = torch.randperm(xb.size(0))
    x_mix = lam * xb + (1 - lam) * xb[idx]
    y_oh = torch.nn.functional.one_hot(yb, num_classes).float()
    y_mix = lam * y_oh + (1 - lam) * y_oh[idx]
    return x_mix, y_mix


def soft_ce(logits, soft_targets, class_weights=None):
    log_probs = torch.nn.functional.log_softmax(logits, dim=1)
    if class_weights is not None:
        log_probs = log_probs * class_weights.unsqueeze(0)
    return -(soft_targets * log_probs).sum(dim=1).mean()


def train_sann_model(model, X_train, y_train, X_val, y_val, num_classes, seed,
                     max_epochs=200, warmup_epochs=10, lr_init=3e-4, wd=1e-4,
                     mixup_alpha=0.3, l1_lambda=5e-8, patience=30):
    torch.manual_seed(seed)
    np.random.seed(seed)

    cw = compute_class_weights(y_train, num_classes)
    class_weights_t = torch.tensor(cw, dtype=torch.float32)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=1024, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_init, weight_decay=wd)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(max_epochs - warmup_epochs, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_f1 = -1.0
    best_state = None
    patience_counter = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            xb_mix, yb_soft = mixup_batch(xb, yb, mixup_alpha, num_classes)
            logits = model(xb_mix)
            loss = soft_ce(logits, yb_soft, class_weights_t) + l1_penalty(model) * l1_lambda
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t).numpy()
        val_pred = val_logits.argmax(axis=1)
        val_f1 = f1_score(y_val, val_pred, average="macro")

        if epoch % 20 == 0 or epoch == 1:
            print(f"    Epoch {epoch:03d} | val_F1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f"    Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_val_f1


def predict_probs(model, X):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32)).numpy()
    return torch.softmax(torch.tensor(logits), dim=1).numpy()


# ══════════════════════════════════════════
# External data preparation (matches eval scripts)
# ══════════════════════════════════════════
def prepare_external_hvg(adata_ext, gene_names_ref, gene_mean_ref, gene_std_ref):
    """
    Load external data already preprocessed in h5ad (z-scored X),
    align to 68K gene order.
    """
    ext_genes = list(adata_ext.var_names)
    X_ext = adata_ext.X
    if sp.issparse(X_ext):
        X_ext = X_ext.toarray()
    X_ext = np.array(X_ext, dtype=np.float32)

    # If data is already z-scored (same genes as 68K), return directly
    if set(ext_genes) == set(gene_names_ref) and len(ext_genes) == len(gene_names_ref):
        # Check if gene order matches
        if ext_genes == gene_names_ref:
            print(f"  Gene alignment: {len(gene_names_ref)}/{len(gene_names_ref)} shared, 0 missing")
            return X_ext

    # Otherwise align to 68K gene order
    ext_gene_idx = {g: i for i, g in enumerate(ext_genes)}
    X_aligned = np.zeros((X_ext.shape[0], len(gene_names_ref)), dtype=np.float32)
    matched = 0
    for j, g in enumerate(gene_names_ref):
        if g in ext_gene_idx:
            X_aligned[:, j] = X_ext[:, ext_gene_idx[g]]
            matched += 1

    print(f"  Gene alignment: {matched}/{len(gene_names_ref)} shared, "
          f"{len(gene_names_ref) - matched} missing (zero-filled)")

    return X_aligned


def prepare_external_labels(y_raw, class_names):
    """Map external labels to model class indices."""
    label_map = {}
    for i, cn in enumerate(class_names):
        label_map[cn] = i
    # Handle T cells / CD8 T mapping
    if "T cells" in class_names:
        label_map["CD8 T"] = class_names.index("T cells")
    elif "CD8 T" in class_names:
        label_map["T cells"] = class_names.index("CD8 T")

    y_encoded = np.full(len(y_raw), -1, dtype=int)
    for i, label in enumerate(y_raw):
        if label in label_map:
            y_encoded[i] = label_map[label]

    known_mask = y_encoded >= 0
    return y_encoded, known_mask


# ══════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Ablation: SANN with mask vs without mask")
    parser.add_argument("--data", default="data/processed/pbmc68k_labeled.h5ad")
    parser.add_argument("--data_8k", default="data/processed/pbmc8k_labeled.h5ad")
    parser.add_argument("--data_3k", default="data/processed/pbmc3k_labeled.h5ad")
    parser.add_argument("--splits", default="results/ablations/fixed_splits.json")
    parser.add_argument("--pca_dim", type=int, default=50)
    parser.add_argument("--mask_pca_dim", type=int, default=50)
    parser.add_argument("--label_key", type=str, default="cell_type_coarse")
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--outdir", default="results/ablation_mask")
    parser.add_argument("--mode", choices=["pca", "hvg", "both"], default="both",
                        help="Run ablation on PCA, HVG, or both")
    parser.add_argument("--pca_model_dir", default="results/full_train_all_pca_coarse",
                        help="Dir with trained v2 PCA seed models")
    parser.add_argument("--hvg_model_dir", default="results/full_train_all_hvg_coarse",
                        help="Dir with trained v2 HVG seed models")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    seeds = [42 + i for i in range(args.n_seeds)]

    # ── Load 68K ──
    print("Loading 68K dataset...")
    adata_68k = sc.read_h5ad(args.data)

    y_cat = adata_68k.obs[args.label_key].astype("category")
    y = y_cat.cat.codes.to_numpy()
    class_names = list(y_cat.cat.categories)
    num_classes = len(class_names)
    print(f"  Classes ({num_classes}): {class_names}")

    X_pca = np.asarray(adata_68k.obsm["X_pca"][:, :args.pca_dim], dtype=np.float32)

    X_raw = adata_68k.X
    if sp.issparse(X_raw):
        X_raw = X_raw.toarray()
    X_raw = np.asarray(X_raw, dtype=np.float32)

    gene_names = list(adata_68k.var_names)
    gene_mean = adata_68k.var["mean"].values.astype(np.float64)
    gene_std = adata_68k.var["std"].values.astype(np.float64)

    # ── Fixed splits ──
    with open(args.splits) as f:
        splits = json.load(f)
    train_idx_full = np.array(splits["train_idx"], dtype=int)
    test_idx = np.array(splits["test_idx"], dtype=int)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    tr_rel, val_rel = next(sss.split(np.zeros(len(train_idx_full)), y[train_idx_full]))

    train_idx = train_idx_full[tr_rel]
    val_idx = train_idx_full[val_rel]

    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    # ── Expression PCA ──
    X_pca_train = X_pca[train_idx]
    X_pca_val = X_pca[val_idx]
    X_pca_test = X_pca[test_idx]

    # ── Mask PCA ──
    X_mask_binary = (X_raw != 0).astype(np.float32)
    mask_pca = PCA(n_components=args.mask_pca_dim, random_state=42)
    mask_pca.fit(X_mask_binary[train_idx])

    X_mask_pca_train = mask_pca.transform(X_mask_binary[train_idx]).astype(np.float32)
    X_mask_pca_val = mask_pca.transform(X_mask_binary[val_idx]).astype(np.float32)
    X_mask_pca_test = mask_pca.transform(X_mask_binary[test_idx]).astype(np.float32)

    # ── SANN inputs ──
    # WITH mask: [expr_pca(50) + mask_pca(50)] = 100
    X_train_mask = np.concatenate([X_pca_train, X_mask_pca_train], axis=1)
    X_val_mask = np.concatenate([X_pca_val, X_mask_pca_val], axis=1)
    X_test_mask = np.concatenate([X_pca_test, X_mask_pca_test], axis=1)

    # WITHOUT mask: just [expr_pca(50)]
    X_train_nomask = X_pca_train
    X_val_nomask = X_pca_val
    X_test_nomask = X_pca_test

    print(f"\n  With mask input shape:    {X_train_mask.shape}")
    print(f"  Without mask input shape: {X_train_nomask.shape}")

    # ── Load external datasets ──
    # Use the SAME pipeline as evaluate_external_pbmc8k.py:
    # Load raw 8K → CPM → log1p → align to 68K HVGs → z-score with 68K stats
    print("\nLoading 8K dataset...")
    adata_8k = sc.read_h5ad(args.data_8k)

    # Labels
    ext_label_key = "cell_type" if "cell_type_coarse" not in adata_8k.obs.columns else args.label_key
    y_8k_raw = adata_8k.obs[ext_label_key].values.astype(str)
    y_8k, mask_8k = prepare_external_labels(y_8k_raw, class_names)

    # Load raw 8K counts and normalize (same as eval script)
    raw_8k_dir = "data/raw/pbmc8k/filtered_gene_bc_matrices/GRCh38"
    adata_8k_raw = sc.read_10x_mtx(raw_8k_dir, var_names="gene_symbols", make_unique=True)

    # QC filtering
    X_r = adata_8k_raw.X
    if sp.issparse(X_r):
        genes_per_cell = np.array((X_r > 0).sum(axis=1)).flatten()
        total_counts = np.array(X_r.sum(axis=1)).flatten()
    else:
        genes_per_cell = (X_r > 0).sum(axis=1)
        total_counts = X_r.sum(axis=1)
    mt_mask_8k = adata_8k_raw.var_names.str.upper().str.startswith("MT-")
    if sp.issparse(X_r):
        mt_counts = np.array(X_r[:, mt_mask_8k].sum(axis=1)).flatten()
    else:
        mt_counts = X_r[:, mt_mask_8k].sum(axis=1)
    pct_mt = mt_counts / (total_counts + 1e-8) * 100
    cell_mask_8k = (genes_per_cell >= 200) & (pct_mt < 10.0)
    adata_8k_raw = adata_8k_raw[cell_mask_8k].copy()
    if sp.issparse(adata_8k_raw.X):
        cells_per_gene = np.array((adata_8k_raw.X > 0).sum(axis=0)).flatten()
    else:
        cells_per_gene = (adata_8k_raw.X > 0).sum(axis=0)
    adata_8k_raw = adata_8k_raw[:, cells_per_gene >= 3].copy()

    sc.pp.normalize_total(adata_8k_raw, target_sum=1e4)
    sc.pp.log1p(adata_8k_raw)

    # Align to 68K HVG genes
    ext_gene_idx = {g: i for i, g in enumerate(adata_8k_raw.var_names)}
    X_8k_lognorm = np.zeros((adata_8k_raw.n_obs, len(gene_names)), dtype=np.float32)
    matched = 0
    for j, g in enumerate(gene_names):
        if g in ext_gene_idx:
            col = adata_8k_raw.X[:, ext_gene_idx[g]]
            if sp.issparse(col):
                col = col.toarray().flatten()
            X_8k_lognorm[:, j] = col
            matched += 1
    print(f"  Gene alignment: {matched}/{len(gene_names)} shared, {len(gene_names)-matched} missing")

    # Z-score with 68K stats (same as eval script)
    gene_std_safe = np.where(gene_std == 0, 1.0, gene_std)
    X_8k_hvg_zscored = ((X_8k_lognorm - gene_mean) / gene_std_safe).astype(np.float32)
    X_8k_hvg_zscored = np.clip(X_8k_hvg_zscored, -10, 10)

    # Filter to known classes
    X_8k_hvg_known = X_8k_hvg_zscored[mask_8k]
    y_8k_known = y_8k[mask_8k]

    # 8K PCA: center with 68K mean, project through 68K loadings (same as eval script)
    pca_loadings = adata_68k.varm["PCs"][:, :args.pca_dim]
    pca_center = X_raw.mean(axis=0)  # per-gene mean of z-scored 68K
    X_8k_pca = ((X_8k_hvg_known - pca_center) @ pca_loadings).astype(np.float32)

    # 8K mask PCA (from z-scored data, consistent with training)
    X_8k_mask_zscored = (X_8k_hvg_known != 0).astype(np.float32)
    X_8k_mask_pca = mask_pca.transform(X_8k_mask_zscored).astype(np.float32)

    X_8k_with_mask = np.concatenate([X_8k_pca, X_8k_mask_pca], axis=1)
    X_8k_no_mask = X_8k_pca

    print(f"  8K known: {X_8k_hvg_known.shape[0]} cells")

    print("\nLoading 3K dataset...")
    adata_3k = sc.read_h5ad(args.data_3k)

    # Load raw 3K, normalize properly (same as evaluate_external_pbmc3k.py)
    adata_3k_raw = sc.read_h5ad("data/raw/pbmc3k/pbmc3k_raw.h5ad")
    adata_3k_raw = adata_3k_raw[adata_3k.obs_names].copy()
    sc.pp.normalize_total(adata_3k_raw, target_sum=1e4)
    sc.pp.log1p(adata_3k_raw)
    X_3k_lognorm = adata_3k_raw.X
    if sp.issparse(X_3k_lognorm):
        X_3k_lognorm = X_3k_lognorm.toarray()
    genes_3k_raw = list(adata_3k_raw.var_names)

    # Align to 68K HVG gene order
    X_3k_aligned = np.zeros((adata_3k.shape[0], len(gene_names)), dtype=np.float32)
    matched = 0
    for j, g in enumerate(gene_names):
        if g in genes_3k_raw:
            idx = genes_3k_raw.index(g)
            X_3k_aligned[:, j] = np.array(X_3k_lognorm[:, idx]).flatten()
            matched += 1
    print(f"  Gene alignment: {matched}/{len(gene_names)} shared, "
          f"{len(gene_names) - matched} missing (zero-filled)")

    # Z-score with 68K stats
    std_safe = np.where(gene_std == 0, 1.0, gene_std)
    X_3k_hvg = np.clip((X_3k_aligned - gene_mean) / std_safe, -10, 10).astype(np.float32)

    ext_label_key_3k = "cell_type" if "cell_type_coarse" not in adata_3k.obs.columns else args.label_key
    y_3k_raw = adata_3k.obs[ext_label_key_3k].values.astype(str)
    y_3k, mask_3k = prepare_external_labels(y_3k_raw, class_names)
    X_3k_hvg_known = X_3k_hvg[mask_3k]
    y_3k_known = y_3k[mask_3k]

    X_3k_pca = (X_3k_hvg_known @ pca_loadings).astype(np.float32)
    X_3k_mask_binary = (X_3k_hvg_known != 0).astype(np.float32)
    X_3k_mask_pca = mask_pca.transform(X_3k_mask_binary).astype(np.float32)

    X_3k_with_mask = np.concatenate([X_3k_pca, X_3k_mask_pca], axis=1)
    X_3k_no_mask = X_3k_pca

    print(f"  3K known: {X_3k_hvg_known.shape[0]} cells")

    # ══════════════════════════════════════════
    # HVG data preparation (for HVG ablation)
    # ══════════════════════════════════════════
    if args.mode in ("hvg", "both"):
        # 68K HVG: already z-scored in h5ad
        X_hvg_train = X_raw[train_idx].astype(np.float32)
        X_hvg_val = X_raw[val_idx].astype(np.float32)
        X_hvg_test = X_raw[test_idx].astype(np.float32)

        # HVG binary masks
        X_hvg_mask_train = (X_hvg_train != 0).astype(np.float32)
        X_hvg_mask_val = (X_hvg_val != 0).astype(np.float32)
        X_hvg_mask_test = (X_hvg_test != 0).astype(np.float32)
        X_8k_hvg_mask = (X_8k_hvg_known != 0).astype(np.float32)
        X_3k_hvg_mask = (X_3k_hvg_known != 0).astype(np.float32)

        # WITH mask: [expr(2000) + mask(2000)] = 4000
        X_hvg_train_mask = np.concatenate([X_hvg_train, X_hvg_mask_train], axis=1)
        X_hvg_val_mask = np.concatenate([X_hvg_val, X_hvg_mask_val], axis=1)
        X_hvg_test_mask = np.concatenate([X_hvg_test, X_hvg_mask_test], axis=1)
        X_8k_hvg_with_mask = np.concatenate([X_8k_hvg_known, X_8k_hvg_mask], axis=1)
        X_3k_hvg_with_mask = np.concatenate([X_3k_hvg_known, X_3k_hvg_mask], axis=1)

        # WITHOUT mask: just [expr(2000)]
        X_hvg_train_nomask = X_hvg_train
        X_hvg_val_nomask = X_hvg_val
        X_hvg_test_nomask = X_hvg_test
        X_8k_hvg_no_mask = X_8k_hvg_known
        X_3k_hvg_no_mask = X_3k_hvg_known

        print(f"\n  HVG with mask input shape:    {X_hvg_train_mask.shape}")
        print(f"  HVG without mask input shape: {X_hvg_train_nomask.shape}")

    # ══════════════════════════════════════════
    # Train and evaluate all variants
    # ══════════════════════════════════════════
    results = []

    # Build list of ablation runs
    ablation_runs = []
    if args.mode in ("pca", "both"):
        ablation_runs.extend([
            ("PCA_with_mask", X_train_mask, X_val_mask, X_test_mask,
             X_8k_with_mask, X_3k_with_mask, True, "pca"),
            ("PCA_no_mask", X_train_nomask, X_val_nomask, X_test_nomask,
             X_8k_no_mask, X_3k_no_mask, False, "pca"),
        ])
    if args.mode in ("hvg", "both"):
        ablation_runs.extend([
            ("HVG_with_mask", X_hvg_train_mask, X_hvg_val_mask, X_hvg_test_mask,
             X_8k_hvg_with_mask, X_3k_hvg_with_mask, True, "hvg"),
            ("HVG_no_mask", X_hvg_train_nomask, X_hvg_val_nomask, X_hvg_test_nomask,
             X_8k_hvg_no_mask, X_3k_hvg_no_mask, False, "hvg"),
        ])

    for variant_name, X_tr, X_v, X_te, X_8k_v, X_3k_v, has_mask, rep_type in ablation_runs:
        print(f"\n{'='*60}")
        print(f"  VARIANT: SANN_{variant_name} ({args.n_seeds}-seed ensemble)")
        print(f"  Input dim: {X_tr.shape[1]}")
        print(f"{'='*60}")

        # Select hyperparameters matching trained SANN v2
        if rep_type == "pca":
            bh, bo, fh = 512, 256, 256
            drop, lr, wd = 0.25, 3e-4, 1e-4
            use_bn = True
            input_noise, mixup_alpha = 0.05, 0.3
            expr_dim = args.pca_dim
            mask_dim = args.mask_pca_dim
        else:  # hvg
            bh, bo, fh = 256, 128, 128
            drop, lr, wd = 0.4, 2e-4, 5e-4
            use_bn = False  # LayerNorm
            input_noise, mixup_alpha = 0.1, 0.4
            expr_dim = 2000
            mask_dim = 2000

        all_test_probs = []
        all_8k_probs = []
        all_3k_probs = []

        # For with_mask variants, load the already-trained v2 seed models
        # so results match the published v2 numbers exactly
        load_pretrained = False
        pretrained_dir = None
        if has_mask:
            if rep_type == "pca":
                pretrained_dir = args.pca_model_dir
            else:
                pretrained_dir = args.hvg_model_dir
            import glob as glob_module
            seed_files = sorted(glob_module.glob(os.path.join(pretrained_dir, "sann_model_seed*.pt")))
            if len(seed_files) >= len(seeds):
                load_pretrained = True
                print(f"  Loading {len(seeds)} pre-trained v2 models from {pretrained_dir}")

        for si, seed in enumerate(seeds):
            print(f"\n  Seed {seed}:")

            if load_pretrained:
                # Load the already-trained v2 model
                model = SANN(
                    expr_dim=expr_dim, mask_dim=mask_dim,
                    num_classes=num_classes,
                    branch_hidden=bh, branch_out=bo, fusion_hidden=fh,
                    dropout=drop, use_batchnorm=use_bn, input_noise=input_noise)
                seed_file = os.path.join(pretrained_dir, f"sann_model_seed{si}.pt")
                model.load_state_dict(torch.load(seed_file, map_location="cpu"))
                model.eval()
                total_params = sum(p.numel() for p in model.parameters())
                print(f"    Loaded: {seed_file}")
                print(f"    Parameters: {total_params:,}")
            elif has_mask:
                model = SANN(
                    expr_dim=expr_dim, mask_dim=mask_dim,
                    num_classes=num_classes,
                    branch_hidden=bh, branch_out=bo, fusion_hidden=fh,
                    dropout=drop, use_batchnorm=use_bn, input_noise=input_noise)
                total_params = sum(p.numel() for p in model.parameters())
                print(f"    Parameters: {total_params:,}")
                t0 = time.time()
                model, best_f1 = train_sann_model(
                    model, X_tr, y_train, X_v, y_val, num_classes, seed,
                    lr_init=lr, wd=wd, mixup_alpha=mixup_alpha)
                train_time = time.time() - t0
                print(f"    Best val F1: {best_f1:.4f} | Time: {train_time:.1f}s")
            else:
                model = SANN_NoMask(
                    expr_dim=expr_dim, num_classes=num_classes,
                    branch_hidden=bh, branch_out=bo, fusion_hidden=fh,
                    dropout=drop, use_batchnorm=use_bn, input_noise=input_noise)
                total_params = sum(p.numel() for p in model.parameters())
                print(f"    Parameters: {total_params:,}")
                t0 = time.time()
                model, best_f1 = train_sann_model(
                    model, X_tr, y_train, X_v, y_val, num_classes, seed,
                    lr_init=lr, wd=wd, mixup_alpha=mixup_alpha)
                train_time = time.time() - t0
                print(f"    Best val F1: {best_f1:.4f} | Time: {train_time:.1f}s")

            test_probs = predict_probs(model, X_te)
            probs_8k = predict_probs(model, X_8k_v)
            probs_3k = predict_probs(model, X_3k_v)

            all_test_probs.append(test_probs)
            all_8k_probs.append(probs_8k)
            all_3k_probs.append(probs_3k)

        # Ensemble average
        ens_test = np.mean(all_test_probs, axis=0)
        ens_8k = np.mean(all_8k_probs, axis=0)
        ens_3k = np.mean(all_3k_probs, axis=0)

        # Evaluate
        for dataset_name, probs, y_true in [
            ("68K_test", ens_test, y_test),
            ("8K", ens_8k, y_8k_known),
            ("3K", ens_3k, y_3k_known),
        ]:
            pred = probs.argmax(axis=1)
            acc = accuracy_score(y_true, pred)
            f1 = f1_score(y_true, pred, average="macro")
            print(f"\n  {variant_name} on {dataset_name}: Acc={acc:.4f} | F1={f1:.4f}")
            results.append({
                "variant": variant_name,
                "dataset": dataset_name,
                "accuracy": float(acc),
                "macro_f1": float(f1),
                "n_seeds": args.n_seeds,
            })

    # ── Save results ──
    df = pd.DataFrame(results)
    out_csv = os.path.join(args.outdir, "mask_ablation_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n{'='*60}")
    print(f"RESULTS SAVED: {out_csv}")
    print(f"{'='*60}")
    print(df.to_string(index=False))

    # ── Summary comparison ──
    print(f"\n{'='*60}")
    print("MASK ABLATION SUMMARY")
    print(f"{'='*60}")
    for rep in ["PCA", "HVG"]:
        wm = df[df["variant"] == f"{rep}_with_mask"]
        nm = df[df["variant"] == f"{rep}_no_mask"]
        if len(wm) == 0:
            continue
        print(f"\n  {rep}:")
        for dataset in ["68K_test", "8K", "3K"]:
            wm_f1 = wm[wm["dataset"] == dataset]["macro_f1"].values[0]
            nm_f1 = nm[nm["dataset"] == dataset]["macro_f1"].values[0]
            diff = wm_f1 - nm_f1
            print(f"    {dataset:>8s}: with_mask={wm_f1:.4f}  no_mask={nm_f1:.4f}  diff={diff:+.4f}")


if __name__ == "__main__":
    main()
