"""
Cross-donor robustness: retrain LR and SANN with 5 different seeds on 68K,
evaluate each on 8K and 3K external datasets.
Saves results to CSV for plotting.
"""
import os
import sys
import time
import json

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.decomposition import PCA
import scipy.sparse as sp


# ══════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════
SEEDS = [42, 53, 64, 75, 86]
DATA_68K = "data/processed/pbmc68k_labeled.h5ad"
DATA_8K = "data/processed/pbmc8k_labeled.h5ad"
DATA_3K = "data/processed/pbmc3k_labeled.h5ad"
OUT_CSV = "results/figures/robustness_cross_donor_splits.csv"
LABEL_KEY = "cell_type"
N_PCA = 50
MASK_PCA = 50
VAL_FRAC = 0.1

os.makedirs("results/figures", exist_ok=True)


# ══════════════════════════════════════════════════
# SANN v2 ARCHITECTURE (same as training scripts)
# ══════════════════════════════════════════════════
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
    def __init__(self, expr_dim, mask_dim, num_classes,
                 branch_hidden=512, branch_out=256, fusion_hidden=256,
                 dropout=0.05, use_batchnorm=True, use_layernorm=False,
                 input_noise=0.0):
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


# ══════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════
def compute_class_weights(y, num_classes):
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    inv = 1.0 / counts
    inv = np.sqrt(inv)
    inv /= inv.sum()
    inv *= num_classes
    return inv.astype(np.float32)


def l1_penalty(model):
    penalty = torch.tensor(0.0)
    for param in model.parameters():
        penalty = penalty + param.abs().sum()
    return penalty


def train_sann(X_train, y_train, X_val, y_val, n_expr, num_classes, is_pca, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_mask = X_train.shape[1] - n_expr
    max_epochs = 200
    warmup_epochs = 10
    l1_lambda = 5e-8

    if is_pca:
        bh, bo, fh = 512, 256, 256
        drop, lr_init, wd = 0.25, 3e-4, 1e-4
        use_bn, use_ln = True, False
        input_noise, mixup_alpha = 0.05, 0.3
    else:
        bh, bo, fh = 256, 128, 128
        drop, lr_init, wd = 0.4, 2e-4, 5e-4
        use_bn, use_ln = False, True
        input_noise, mixup_alpha = 0.1, 0.4

    model = SANN(n_expr, n_mask, num_classes,
                 branch_hidden=bh, branch_out=bo, fusion_hidden=fh,
                 dropout=drop, use_batchnorm=use_bn, use_layernorm=use_ln,
                 input_noise=input_noise)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_init, weight_decay=wd)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(max_epochs - warmup_epochs, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    cw = compute_class_weights(y_train, num_classes)
    class_weights_t = torch.tensor(cw, dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=class_weights_t, label_smoothing=0.1)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=1024, shuffle=True)

    def mixup_batch(xb, yb, alpha, nc):
        if alpha <= 0:
            return xb, torch.nn.functional.one_hot(yb, nc).float()
        lam = np.random.beta(alpha, alpha)
        lam = max(lam, 1 - lam)
        idx = torch.randperm(xb.size(0))
        x_mix = lam * xb + (1 - lam) * xb[idx]
        y_oh = torch.nn.functional.one_hot(yb, nc).float()
        y_mix = lam * y_oh + (1 - lam) * y_oh[idx]
        return x_mix, y_mix

    def soft_ce(logits, soft_targets, cw=None):
        lp = torch.nn.functional.log_softmax(logits, dim=1)
        if cw is not None:
            lp = lp * cw.unsqueeze(0)
        return -(soft_targets * lp).sum(dim=1).mean()

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

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= 30:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def evaluate_model_on_external(model, X_ext, y_ext, is_torch=False):
    if is_torch:
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(X_ext, dtype=torch.float32)).numpy()
        probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
        pred = probs.argmax(axis=1)
    else:
        pred = model.predict(X_ext)

    # Only evaluate on classes present in test set (known)
    acc = accuracy_score(y_ext, pred)
    f1 = f1_score(y_ext, pred, average="macro")
    return acc, f1


# ══════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════
def main():
    print("Loading 68K dataset...")
    adata_68k = sc.read_h5ad(DATA_68K)
    X_68k = np.array(adata_68k.X, dtype=np.float32)
    y_68k_raw = adata_68k.obs[LABEL_KEY].values.astype(str)

    le = LabelEncoder()
    y_68k = le.fit_transform(y_68k_raw)
    num_classes = len(le.classes_)
    print(f"  68K: {X_68k.shape}, {num_classes} classes: {list(le.classes_)}")

    # Pre-computed PCA from h5ad
    X_68k_pca = np.asarray(adata_68k.obsm["X_pca"][:, :N_PCA], dtype=np.float32)
    pca_loadings = adata_68k.varm["PCs"][:, :N_PCA]  # for projecting 3K
    print(f"  68K PCA: {X_68k_pca.shape}")

    # Gene info for external alignment
    gene_names_68k = list(adata_68k.var_names)
    gene_mean = adata_68k.var["mean"].values.astype(np.float64)
    gene_std = adata_68k.var["std"].values.astype(np.float64)

    # ── Load external datasets ──
    print("Loading 8K dataset...")
    adata_8k = sc.read_h5ad(DATA_8K)
    print("Loading 3K dataset...")
    adata_3k = sc.read_h5ad(DATA_3K)

    # ── Prepare external HVG data (align to 68K genes, z-score with 68K stats) ──
    def prepare_external_hvg(adata_ext, gene_names_ref, gene_mean_ref, gene_std_ref):
        ext_genes = list(adata_ext.var_names)
        X_ext_raw = adata_ext.X
        if sp.issparse(X_ext_raw):
            X_ext_raw = X_ext_raw.toarray()
        X_ext_raw = np.array(X_ext_raw, dtype=np.float64)

        # Align to 68K gene order
        ext_gene_idx = {g: i for i, g in enumerate(ext_genes)}
        X_aligned = np.zeros((X_ext_raw.shape[0], len(gene_names_ref)), dtype=np.float64)
        for j, g in enumerate(gene_names_ref):
            if g in ext_gene_idx:
                X_aligned[:, j] = X_ext_raw[:, ext_gene_idx[g]]

        # Z-score with 68K stats
        std_safe = np.where(gene_std_ref == 0, 1.0, gene_std_ref)
        X_zscored = (X_aligned - gene_mean_ref) / std_safe
        X_zscored = np.clip(X_zscored, -10, 10).astype(np.float32)

        return X_zscored

    def prepare_external_labels(adata_ext, label_encoder):
        y_raw = adata_ext.obs[LABEL_KEY].values.astype(str)
        # Map "T cells" -> "CD8 T" if needed
        y_raw = np.array(["CD8 T" if v == "T cells" else v for v in y_raw])
        known_mask = np.isin(y_raw, label_encoder.classes_)
        y_encoded = np.full(len(y_raw), -1, dtype=int)
        y_encoded[known_mask] = label_encoder.transform(y_raw[known_mask])
        return y_encoded, known_mask

    X_8k_hvg = prepare_external_hvg(adata_8k, gene_names_68k, gene_mean, gene_std)
    X_3k_hvg = prepare_external_hvg(adata_3k, gene_names_68k, gene_mean, gene_std)

    y_8k, mask_8k = prepare_external_labels(adata_8k, le)
    y_3k, mask_3k = prepare_external_labels(adata_3k, le)

    # Filter to known classes only
    X_8k_hvg_known = X_8k_hvg[mask_8k]
    y_8k_known = y_8k[mask_8k]
    X_3k_hvg_known = X_3k_hvg[mask_3k]
    y_3k_known = y_3k[mask_3k]

    # Pre-computed PCA for 8K; project 3K through 68K loadings
    X_8k_pca_all = np.asarray(adata_8k.obsm["X_pca"][:, :N_PCA], dtype=np.float32)
    X_8k_pca = X_8k_pca_all[mask_8k]
    X_3k_pca = (X_3k_hvg_known @ pca_loadings).astype(np.float32)

    print(f"  8K known: {X_8k_hvg_known.shape[0]} cells, PCA: {X_8k_pca.shape}")
    print(f"  3K known: {X_3k_hvg_known.shape[0]} cells, PCA: {X_3k_pca.shape}")

    # ── Results collection ──
    results = []

    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"SEED {seed}")
        print(f"{'='*60}")

        # Create train/val split
        sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_FRAC, random_state=seed)
        train_idx, val_idx = next(sss.split(X_68k, y_68k))

        X_train_raw = X_68k[train_idx]
        X_val_raw = X_68k[val_idx]
        y_train = y_68k[train_idx]
        y_val = y_68k[val_idx]

        # ════════════════════════════
        # HVG PATH
        # ════════════════════════════
        print(f"\n  [HVG] Training LR and SANN (seed={seed})...")

        # 68K data is ALREADY z-scored in h5ad (with full 68K stats).
        # For robustness, we DON'T re-z-score per split — we use the data as-is,
        # exactly as the main training pipeline does.
        X_train_hvg = X_train_raw.astype(np.float32)
        X_val_hvg = X_val_raw.astype(np.float32)

        # External data is also z-scored with 68K full stats — consistent
        X_8k_hvg_eval = X_8k_hvg_known.astype(np.float32)
        X_3k_hvg_eval = X_3k_hvg_known.astype(np.float32)

        # HVG masks (from z-scored data)
        X_mask_train_hvg = (X_train_hvg != 0).astype(np.float32)
        X_mask_val_hvg = (X_val_hvg != 0).astype(np.float32)
        X_mask_8k_hvg = (X_8k_hvg_eval != 0).astype(np.float32)
        X_mask_3k_hvg = (X_3k_hvg_eval != 0).astype(np.float32)

        X_train_sann_hvg = np.concatenate([X_train_hvg, X_mask_train_hvg], axis=1)
        X_val_sann_hvg = np.concatenate([X_val_hvg, X_mask_val_hvg], axis=1)
        X_8k_sann_hvg = np.concatenate([X_8k_hvg_eval, X_mask_8k_hvg], axis=1)
        X_3k_sann_hvg = np.concatenate([X_3k_hvg_eval, X_mask_3k_hvg], axis=1)

        # --- LR HVG (with StandardScaler + C grid, matching main pipeline) ---
        from sklearn.preprocessing import StandardScaler
        t0 = time.time()
        scaler_hvg = StandardScaler()
        X_train_hvg_s = scaler_hvg.fit_transform(X_train_hvg)
        X_8k_hvg_s = scaler_hvg.transform(X_8k_hvg_eval)
        X_3k_hvg_s = scaler_hvg.transform(X_3k_hvg_eval)

        best_lr_hvg = None
        best_lr_hvg_f1 = -1.0
        for C in [0.1, 1.0, 5.0]:
            lr_tmp = LogisticRegression(max_iter=3000, solver="lbfgs",
                                        C=C, random_state=seed, n_jobs=1)
            lr_tmp.fit(X_train_hvg_s, y_train)
            val_pred = lr_tmp.predict(scaler_hvg.transform(X_val_hvg))
            val_f1 = f1_score(y_val, val_pred, average="macro")
            if val_f1 > best_lr_hvg_f1:
                best_lr_hvg_f1 = val_f1
                best_lr_hvg = lr_tmp
        lr_hvg_time = time.time() - t0

        acc_8k, f1_8k = evaluate_model_on_external(best_lr_hvg, X_8k_hvg_s, y_8k_known)
        acc_3k, f1_3k = evaluate_model_on_external(best_lr_hvg, X_3k_hvg_s, y_3k_known)
        print(f"    LR_HVG  | 8K F1={f1_8k:.4f} | 3K F1={f1_3k:.4f} | {lr_hvg_time:.1f}s")
        results.append({"seed": seed, "model": "LR", "rep": "HVG", "donor": "8K", "macro_f1": f1_8k, "accuracy": acc_8k})
        results.append({"seed": seed, "model": "LR", "rep": "HVG", "donor": "3K", "macro_f1": f1_3k, "accuracy": acc_3k})

        # --- SANN HVG ---
        t0 = time.time()
        sann_hvg = train_sann(X_train_sann_hvg, y_train, X_val_sann_hvg, y_val,
                              n_expr=2000, num_classes=num_classes, is_pca=False, seed=seed)
        sann_hvg_time = time.time() - t0

        acc_8k, f1_8k = evaluate_model_on_external(sann_hvg, X_8k_sann_hvg, y_8k_known, is_torch=True)
        acc_3k, f1_3k = evaluate_model_on_external(sann_hvg, X_3k_sann_hvg, y_3k_known, is_torch=True)
        print(f"    SANN_HVG| 8K F1={f1_8k:.4f} | 3K F1={f1_3k:.4f} | {sann_hvg_time:.1f}s")
        results.append({"seed": seed, "model": "SANN", "rep": "HVG", "donor": "8K", "macro_f1": f1_8k, "accuracy": acc_8k})
        results.append({"seed": seed, "model": "SANN", "rep": "HVG", "donor": "3K", "macro_f1": f1_3k, "accuracy": acc_3k})

        # ════════════════════════════
        # PCA PATH
        # ════════════════════════════
        print(f"\n  [PCA] Training LR and SANN (seed={seed})...")

        # PCA is pre-computed in h5ad for 68K and 8K.
        # For 3K, project through 68K PCA loadings.
        X_train_pca = X_68k_pca[train_idx].astype(np.float32)
        X_val_pca = X_68k_pca[val_idx].astype(np.float32)

        # Sparsity masks from z-scored HVG data (consistent with training pipeline)
        X_mask_train_raw = (X_train_raw != 0).astype(np.float32)
        X_mask_val_raw = (X_val_raw != 0).astype(np.float32)
        X_mask_8k_raw = (X_8k_hvg_known != 0).astype(np.float32)
        X_mask_3k_raw = (X_3k_hvg_known != 0).astype(np.float32)

        # Fit mask PCA on this split's train data
        mask_pca_model = PCA(n_components=MASK_PCA, random_state=42)
        X_mask_pca_train = mask_pca_model.fit_transform(X_mask_train_raw).astype(np.float32)
        X_mask_pca_val = mask_pca_model.transform(X_mask_val_raw).astype(np.float32)
        X_mask_pca_8k = mask_pca_model.transform(X_mask_8k_raw).astype(np.float32)
        X_mask_pca_3k = mask_pca_model.transform(X_mask_3k_raw).astype(np.float32)

        X_train_sann_pca = np.concatenate([X_train_pca, X_mask_pca_train], axis=1)
        X_val_sann_pca = np.concatenate([X_val_pca, X_mask_pca_val], axis=1)
        X_8k_sann_pca = np.concatenate([X_8k_pca, X_mask_pca_8k], axis=1)
        X_3k_sann_pca = np.concatenate([X_3k_pca, X_mask_pca_3k], axis=1)

        # --- LR PCA (with StandardScaler + C grid, matching main pipeline) ---
        t0 = time.time()
        scaler_pca = StandardScaler()
        X_train_pca_s = scaler_pca.fit_transform(X_train_pca)
        X_8k_pca_s = scaler_pca.transform(X_8k_pca)
        X_3k_pca_s = scaler_pca.transform(X_3k_pca)

        best_lr_pca = None
        best_lr_pca_f1 = -1.0
        for C in [0.1, 0.5, 1.0, 2.0, 5.0]:
            lr_tmp = LogisticRegression(max_iter=3000, solver="lbfgs",
                                        C=C, random_state=seed, n_jobs=1)
            lr_tmp.fit(X_train_pca_s, y_train)
            val_pred = lr_tmp.predict(scaler_pca.transform(X_val_pca))
            val_f1 = f1_score(y_val, val_pred, average="macro")
            if val_f1 > best_lr_pca_f1:
                best_lr_pca_f1 = val_f1
                best_lr_pca = lr_tmp
        lr_pca_time = time.time() - t0

        acc_8k, f1_8k = evaluate_model_on_external(best_lr_pca, X_8k_pca_s, y_8k_known)
        acc_3k, f1_3k = evaluate_model_on_external(best_lr_pca, X_3k_pca_s, y_3k_known)
        print(f"    LR_PCA  | 8K F1={f1_8k:.4f} | 3K F1={f1_3k:.4f} | {lr_pca_time:.1f}s")
        results.append({"seed": seed, "model": "LR", "rep": "PCA", "donor": "8K", "macro_f1": f1_8k, "accuracy": acc_8k})
        results.append({"seed": seed, "model": "LR", "rep": "PCA", "donor": "3K", "macro_f1": f1_3k, "accuracy": acc_3k})

        # --- SANN PCA ---
        t0 = time.time()
        sann_pca = train_sann(X_train_sann_pca, y_train, X_val_sann_pca, y_val,
                              n_expr=N_PCA, num_classes=num_classes, is_pca=True, seed=seed)
        sann_pca_time = time.time() - t0

        acc_8k, f1_8k = evaluate_model_on_external(sann_pca, X_8k_sann_pca, y_8k_known, is_torch=True)
        acc_3k, f1_3k = evaluate_model_on_external(sann_pca, X_3k_sann_pca, y_3k_known, is_torch=True)
        print(f"    SANN_PCA| 8K F1={f1_8k:.4f} | 3K F1={f1_3k:.4f} | {sann_pca_time:.1f}s")
        results.append({"seed": seed, "model": "SANN", "rep": "PCA", "donor": "8K", "macro_f1": f1_8k, "accuracy": acc_8k})
        results.append({"seed": seed, "model": "SANN", "rep": "PCA", "donor": "3K", "macro_f1": f1_3k, "accuracy": acc_3k})

    # ── Save ──
    df = pd.DataFrame(results)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n{'='*60}")
    print(f"Saved results to {OUT_CSV}")
    print(f"{'='*60}")

    # Summary
    print("\nSummary (mean ± std across 5 seeds):")
    for rep in ["PCA", "HVG"]:
        for model in ["LR", "SANN"]:
            for donor in ["8K", "3K"]:
                vals = df[(df["model"] == model) & (df["rep"] == rep) & (df["donor"] == donor)]["macro_f1"]
                print(f"  {model}_{rep} on {donor}: {vals.mean():.4f} ± {vals.std():.4f}")


if __name__ == "__main__":
    main()
