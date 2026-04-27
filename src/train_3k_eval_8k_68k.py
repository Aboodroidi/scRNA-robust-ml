#!/usr/bin/env python
"""
Train PCA-based LR / XGBoost / SANN models on PBMC 3K, evaluate externally
on PBMC 8K and PBMC 68K.

Mirror of train_all_pca_models.py + evaluate_external_pbmc3k/8k.py, but
with 3K as the source instead of 68K.

Pipeline:
  1. Load pbmc3k_labeled.h5ad (z-scored in 3K's own gene/stat space).
  2. Stratified train/val split within 3K.
  3. Fit expression PCA + mask PCA on the train split only.
  4. Train LR, XGB, SANN on PCA features (same configs as 68K script).
  5. External eval on full 8K: align raw 8K to 3K genes, z-score with
     3K stats, project through 3K PCAs, predict.
  6. External eval on full 68K: undo 68K's own z-score to recover log-norm,
     align to 3K genes, z-score with 3K stats, project, predict.
  7. Summary CSV (macro-F1 + known-class-only macro-F1).
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

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
)
from sklearn.model_selection import StratifiedShuffleSplit

import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


# SANN architecture (identical to train_all_pca_models.py)
def _make_norm_pca(dim, use_batchnorm):
    return nn.BatchNorm1d(dim) if use_batchnorm else nn.LayerNorm(dim)


class ResidualBlockPCA(nn.Module):
    def __init__(self, dim, dropout=0.1, use_batchnorm=True):
        super().__init__()
        self.norm = _make_norm_pca(dim, use_batchnorm)
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
                 dropout=0.25, use_batchnorm=True, input_noise=0.0):
        super().__init__()
        self.expr_dim = expr_dim
        self.input_noise = input_noise

        self.expr_proj = nn.Sequential(
            nn.Linear(expr_dim, branch_hidden),
            _make_norm_pca(branch_hidden, use_batchnorm),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self.expr_res1 = ResidualBlockPCA(branch_hidden, dropout, use_batchnorm)
        self.expr_res2 = ResidualBlockPCA(branch_hidden, dropout, use_batchnorm)
        self.expr_out = nn.Sequential(
            nn.Linear(branch_hidden, branch_out),
            _make_norm_pca(branch_out, use_batchnorm),
            nn.GELU(),
        )

        self.mask_proj = nn.Sequential(
            nn.Linear(mask_dim, branch_hidden),
            _make_norm_pca(branch_hidden, use_batchnorm),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self.mask_res1 = ResidualBlockPCA(branch_hidden, dropout, use_batchnorm)
        self.mask_res2 = ResidualBlockPCA(branch_hidden, dropout, use_batchnorm)
        self.mask_out = nn.Sequential(
            nn.Linear(branch_hidden, branch_out),
            _make_norm_pca(branch_out, use_batchnorm),
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
            _make_norm_pca(fusion_hidden, use_batchnorm),
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


def l1_penalty(model):
    p = torch.tensor(0.0, device=next(model.parameters()).device)
    for param in model.parameters():
        p = p + param.abs().sum()
    return p


def compute_class_weights(y, num_classes):
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    inv = 1.0 / np.sqrt(counts)
    w = inv / inv.sum() * num_classes
    return w.astype(np.float32)


# Training
def train_lr(X_tr, y_tr, X_val, y_val, outdir, c_grid=(0.1, 1.0, 5.0)):
    print("\nTraining Logistic Regression (PCA, class_weight=balanced)...")
    t0 = time.time()
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)

    best_f1, best_C, best_model = -1.0, None, None
    for C in c_grid:
        m = LogisticRegression(C=C, max_iter=3000, tol=1e-4,
                               solver="lbfgs", n_jobs=1, random_state=42,
                               class_weight="balanced")
        m.fit(X_tr_s, y_tr)
        f1v = f1_score(y_val, m.predict(X_val_s), average="macro")
        print(f"  C={C} | val macro-F1={f1v:.4f}")
        if f1v > best_f1:
            best_f1, best_C, best_model = f1v, C, m

    train_time = time.time() - t0
    joblib.dump(best_model, os.path.join(outdir, "lr_model.pkl"))
    joblib.dump(scaler, os.path.join(outdir, "lr_scaler.pkl"))
    print(f"  best C={best_C} | val macro-F1={best_f1:.4f} | {train_time:.1f}s")
    return {"Model": "LR", "train_time": float(train_time),
            "best_C": best_C, "best_val_f1": float(best_f1)}


def train_xgb(X_tr, y_tr, X_val, y_val, num_classes, outdir):
    print("\nTraining XGBoost (PCA, class-balanced sample_weight)...")
    t0 = time.time()
    # Balanced sample weights: w_i = N / (num_classes * count[y_i])
    counts = np.bincount(y_tr, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    sw = (len(y_tr) / (num_classes * counts))[y_tr].astype(np.float32)
    dtr = xgb.DMatrix(X_tr, label=y_tr, weight=sw)
    dval = xgb.DMatrix(X_val, label=y_val)
    params = {
        "objective": "multi:softprob",
        "num_class": int(num_classes),
        "eta": 0.05,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "eval_metric": "mlogloss",
        "seed": 42,
        "nthread": 1,
    }
    evals_result = {}
    booster = xgb.train(
        params, dtr, num_boost_round=1500,
        evals=[(dtr, "train"), (dval, "valid")],
        evals_result=evals_result, verbose_eval=50,
        callbacks=[xgb.callback.EarlyStopping(rounds=50, save_best=True)],
    )
    train_time = time.time() - t0
    booster.save_model(os.path.join(outdir, "xgb_model.json"))
    print(f"  best_iter={booster.best_iteration} | {train_time:.1f}s")
    return {"Model": "XGB", "train_time": float(train_time),
            "best_iteration": int(booster.best_iteration)}


def train_sann(X_tr, y_tr, X_val, y_val, num_classes, outdir,
               n_expr_features, l1_lambda=5e-8, seed=42,
               arch="small", batch_size=128, patience=50,
               mixup_alpha=0.4, model_filename="sann_model.pt"):
    print(f"\nTraining SANN v2 (PCA dual-encoder, arch={arch}, seed={seed})...")
    t0 = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = "cpu"
    n_mask_features = X_tr.shape[1] - n_expr_features
    max_epochs, warmup = 200, 10
    if arch == "small":
        bh, bo, fh = 256, 128, 128
    else:
        bh, bo, fh = 512, 256, 256
    drop, lr_init, wd = 0.25, 3e-4, 1e-4
    input_noise = 0.05

    model = SANN(
        expr_dim=n_expr_features, mask_dim=n_mask_features,
        num_classes=num_classes, branch_hidden=bh,
        branch_out=bo, fusion_hidden=fh,
        dropout=drop, use_batchnorm=True, input_noise=input_noise,
    ).to(device)

    print(f"  params: {sum(p.numel() for p in model.parameters()):,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_init, weight_decay=wd)

    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(max_epochs - warmup, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    cw = compute_class_weights(y_tr, num_classes)
    cw_t = torch.tensor(cw, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=cw_t, label_smoothing=0.1)

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    bs = min(batch_size, len(y_tr))
    # Class-balanced sampler: each class gets equal total weight per epoch
    counts = np.bincount(y_tr, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    per_sample_w = (1.0 / counts)[y_tr]
    per_sample_w = per_sample_w / per_sample_w.sum() * len(y_tr)
    sampler = WeightedRandomSampler(
        weights=torch.tensor(per_sample_w, dtype=torch.double),
        num_samples=len(y_tr),
        replacement=True,
    )
    print(f"  [WeightedRandomSampler] class counts={counts.astype(int).tolist()}")
    train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                              batch_size=bs, sampler=sampler)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t),
                            batch_size=bs, shuffle=False)

    def mixup(xb, yb, alpha, nc):
        if alpha <= 0:
            return xb, torch.nn.functional.one_hot(yb, nc).float()
        lam = np.random.beta(alpha, alpha)
        lam = max(lam, 1 - lam)
        idx = torch.randperm(xb.size(0))
        x_mix = lam * xb + (1 - lam) * xb[idx]
        y_oh = torch.nn.functional.one_hot(yb, nc).float()
        y_mix = lam * y_oh + (1 - lam) * y_oh[idx]
        return x_mix, y_mix

    def soft_ce(logits, soft_targets, w=None):
        lp = torch.nn.functional.log_softmax(logits, dim=1)
        if w is not None:
            lp = lp * w.unsqueeze(0)
        return -(soft_targets * lp).sum(dim=1).mean()

    best_val_f1, best_state, pc = -1.0, None, 0
    history = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss_sum, train_n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            xb_m, y_soft = mixup(xb, yb, mixup_alpha, num_classes)
            logits = model(xb_m)
            loss = soft_ce(logits, y_soft, cw_t) + l1_penalty(model) * l1_lambda
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            bs = xb.size(0)
            train_loss_sum += float(loss.item()) * bs
            train_n += bs
        train_loss = train_loss_sum / max(train_n, 1)

        model.eval()
        vp, vt = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb.to(device))
                vp.append(logits.cpu().numpy())
                vt.append(yb.numpy())
        val_logits = np.vstack(vp)
        val_true = np.concatenate(vt)
        val_pred = val_logits.argmax(axis=1)
        val_f1 = f1_score(val_true, val_pred, average="macro")
        val_acc = accuracy_score(val_true, val_pred)
        scheduler.step()

        history.append({
            "epoch": epoch, "train_loss": float(train_loss),
            "val_acc": float(val_acc), "val_macro_f1": float(val_f1),
            "lr": float(optimizer.param_groups[0]["lr"]),
        })

        if epoch == 1 or epoch % 10 == 0:
            print(f"  epoch {epoch:03d} | train_loss={train_loss:.4f} | "
                  f"val macro-F1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pc = 0
        else:
            pc += 1
        if pc >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    train_time = time.time() - t0
    torch.save(model.state_dict(), os.path.join(outdir, model_filename))
    hist_filename = model_filename.replace(".pt", "_history.csv")
    pd.DataFrame(history).to_csv(os.path.join(outdir, hist_filename),
                                 index=False)
    print(f"  SANN seed={seed} done | best val F1={best_val_f1:.4f} | "
          f"{train_time:.1f}s | saved={model_filename}")
    return {"Model": "SANN", "seed": int(seed),
            "train_time": float(train_time),
            "best_val_f1": float(best_val_f1),
            "arch": arch, "branch_hidden": bh, "branch_out": bo}


# Evaluation helpers
def to_dense_f32(x):
    if sp.issparse(x):
        return x.toarray().astype(np.float32)
    return np.asarray(x, dtype=np.float32)


def predict_sann(model_dir, n_expr, n_mask, num_classes, X_input,
                 arch="small"):
    """Load all sann_model_seed*.pt in model_dir and return softmax-averaged
    probabilities. Falls back to sann_model.pt if no seed files present."""
    import glob as _glob
    seed_files = sorted(_glob.glob(os.path.join(model_dir, "sann_model_seed*.pt")))
    if not seed_files:
        seed_files = [os.path.join(model_dir, "sann_model.pt")]

    if arch == "small":
        bh, bo, fh = 256, 128, 128
    else:
        bh, bo, fh = 512, 256, 256

    all_probs = []
    for sf in seed_files:
        m = SANN(
            expr_dim=n_expr, mask_dim=n_mask, num_classes=num_classes,
            branch_hidden=bh, branch_out=bo, fusion_hidden=fh,
            dropout=0.25, use_batchnorm=True,
        )
        m.load_state_dict(torch.load(sf, map_location="cpu"))
        m.eval()
        with torch.no_grad():
            logits = m(torch.tensor(X_input, dtype=torch.float32)).numpy()
        probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
        all_probs.append(probs)
    avg = np.mean(all_probs, axis=0)
    if len(all_probs) > 1:
        print(f"  [SANN ensemble] averaged over {len(all_probs)} seeds")
    return avg


def evaluate_external(name, y_true, y_pred, y_probs, class_names,
                      outdir, tag, known_class_indices=None):
    """Same pattern as evaluate_external_pbmc8k.py's evaluate_model."""
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"\n{'=' * 60}")
    print(f"  {name} — External Validation ({tag})")
    print(f"{'=' * 60}")
    print(f"  Accuracy:           {acc:.4f}")
    print(f"  Macro-F1 (all):     {f1:.4f}")

    if known_class_indices is not None and len(known_class_indices) > 0:
        f1k = f1_score(y_true, y_pred, average="macro",
                       labels=known_class_indices, zero_division=0)
        wf1k = f1_score(y_true, y_pred, average="weighted",
                        labels=known_class_indices, zero_division=0)
        print(f"  Macro-F1 (known):    {f1k:.4f}")
        print(f"  Weighted-F1 (known): {wf1k:.4f}")
    else:
        f1k = f1
        wf1k = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    pc = np.bincount(y_pred, minlength=len(class_names))
    tc = np.bincount(y_true, minlength=len(class_names))
    print("  Predicted / true counts per class:")
    for i, cn in enumerate(class_names):
        if pc[i] > 0 or tc[i] > 0:
            print(f"    {cn:14s}: pred={pc[i]:5d}  true={tc[i]:5d}")

    pl = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    pn = [class_names[i] for i in pl if i < len(class_names)]
    rep = classification_report(y_true, y_pred, labels=pl,
                                target_names=pn, zero_division=0)
    print(rep)

    np.save(os.path.join(outdir, f"{name.lower()}_{tag}_pred.npy"), y_pred)
    np.save(os.path.join(outdir, f"{name.lower()}_{tag}_probs.npy"), y_probs)
    np.save(os.path.join(outdir, f"{name.lower()}_{tag}_true.npy"), y_true)
    with open(os.path.join(outdir, f"{name.lower()}_{tag}_report.txt"), "w") as f:
        f.write(f"{name} on {tag}\nAccuracy {acc:.4f}\nMacro-F1 {f1:.4f}\n"
                f"Macro-F1 (known) {f1k:.4f}\n\n{rep}")

    return {
        "Model": name,
        "TestSet": tag,
        "Accuracy": float(acc),
        "Macro-F1": float(f1),
        "Macro-F1 (known)": float(f1k),
        "Weighted-F1 (known)": float(wf1k),
    }


# Data alignment helpers
def align_and_zscore(X_lognorm_src, genes_src, genes_ref, mean_ref, std_ref):
    """Align log-normalized expression to ref gene set, zero-fill missing,
    then z-score with ref stats (clipped to ±10)."""
    X = X_lognorm_src
    if sp.issparse(X):
        X = X.toarray()
    X = X.astype(np.float32)

    src_idx = {g: i for i, g in enumerate(genes_src)}
    out_log = np.zeros((X.shape[0], len(genes_ref)), dtype=np.float32)
    shared = 0
    for i, g in enumerate(genes_ref):
        j = src_idx.get(g)
        if j is not None:
            out_log[:, i] = X[:, j]
            shared += 1

    std_safe = std_ref.copy()
    std_safe[std_safe == 0] = 1.0
    out_z = np.clip((out_log - mean_ref) / std_safe, -10, 10).astype(np.float32)
    return out_log, out_z, shared


def load_8k_lognorm(raw_dir):
    """Raw 8K → QC → CPM → log1p. Returns (X_lognorm, gene_list, obs_names)."""
    print(f"  Loading raw 8K from {raw_dir} ...")
    adata = sc.read_10x_mtx(raw_dir, var_names="gene_symbols", make_unique=True)
    X = adata.X
    if sp.issparse(X):
        gpc = np.array((X > 0).sum(axis=1)).flatten()
        tc = np.array(X.sum(axis=1)).flatten()
    else:
        gpc = (X > 0).sum(axis=1)
        tc = X.sum(axis=1)
    mt = adata.var_names.str.upper().str.startswith("MT-")
    if sp.issparse(X):
        mc = np.array(X[:, mt].sum(axis=1)).flatten()
    else:
        mc = X[:, mt].sum(axis=1)
    pmt = mc / (tc + 1e-8) * 100
    keep = (gpc >= 200) & (pmt < 10.0)
    adata = adata[keep].copy()

    if sp.issparse(adata.X):
        cpg = np.array((adata.X > 0).sum(axis=0)).flatten()
    else:
        cpg = (adata.X > 0).sum(axis=0)
    adata = adata[:, cpg >= 3].copy()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    X = to_dense_f32(adata.X)
    return X, list(adata.var_names), list(adata.obs_names)


def reconstruct_68k_lognorm(adata_68k):
    """Invert 68K's z-scoring (clipped) to recover approximate log-norm."""
    X = to_dense_f32(adata_68k.X)
    m = adata_68k.var["mean"].values.astype(np.float32)
    s = adata_68k.var["std"].values.astype(np.float32)
    s = np.where(s == 0, 1.0, s)
    X_lognorm = X * s + m
    # clipping-introduced approximation, fine for the vast majority of values
    return X_lognorm, list(adata_68k.var_names)


# Main
def main():
    ap = argparse.ArgumentParser(
        description="Train on PBMC 3K, eval externally on 8K + 68K"
    )
    ap.add_argument("--data_3k", default="data/processed/pbmc3k_labeled.h5ad")
    ap.add_argument("--data_8k", default="data/processed/pbmc8k_labeled.h5ad")
    ap.add_argument("--raw_8k_dir",
                    default="data/raw/pbmc8k/filtered_gene_bc_matrices/GRCh38")
    ap.add_argument("--data_68k", default="data/processed/pbmc68k_labeled.h5ad")
    ap.add_argument("--outdir", default="results/train_3k")
    ap.add_argument("--label_key", type=str, default="cell_type_coarse")
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--pca_dim", type=int, default=50)
    ap.add_argument("--mask_pca_dim", type=int, default=50)
    ap.add_argument("--models", type=str, default="lr,xgb,sann")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_sann_seeds", type=int, default=3,
                    help="Number of SANN seeds for deep ensemble "
                         "(softmax-averaged at eval time).")
    ap.add_argument("--sann_arch", type=str, default="small",
                    choices=["small", "big"],
                    help="'small' (for 3K training): 256/128/128. "
                         "'big' (mirrors 68K script): 512/256/256.")
    ap.add_argument("--sann_batch_size", type=int, default=128)
    ap.add_argument("--sann_patience", type=int, default=50)
    ap.add_argument("--sann_mixup_alpha", type=float, default=0.4)
    ap.add_argument("--no_intersection", action="store_true",
                    help="Disable 3K ∩ 68K gene-intersection filter "
                         "(default: intersection is ON to avoid zero-fill bias on 68K).")
    ap.add_argument("--prior_correction", action="store_true",
                    help="At inference, divide predicted probs by training "
                         "class prior (label-shift correction).")
    args = ap.parse_args()
    args.models = [m.strip().lower() for m in args.models.split(",")]
    use_intersection = not args.no_intersection

    os.makedirs(args.outdir, exist_ok=True)

    # 1. Load 3K
    print(f"Loading 3K: {args.data_3k}")
    ad3 = sc.read_h5ad(args.data_3k)
    print(f"  3K shape: {ad3.shape}")

    if "mean" not in ad3.var or "std" not in ad3.var:
        raise RuntimeError(
            "3K file missing var['mean']/['std']. "
            "Rerun preprocess_pbmc3k.py to produce them."
        )
    ref_mean_full = ad3.var["mean"].values.astype(np.float32)
    ref_std_full = ad3.var["std"].values.astype(np.float32)
    genes_3k_full = list(ad3.var_names)

    # Z-scored expression (for expression PCA)
    X_3k_z_full = to_dense_f32(ad3.X)
    # True log-normalized (for mask)
    if "log_norm" in ad3.layers:
        X_3k_log_full = to_dense_f32(ad3.layers["log_norm"])
    else:
        ref_std_safe_full = np.where(ref_std_full == 0, 1.0, ref_std_full)
        X_3k_log_full = X_3k_z_full * ref_std_safe_full + ref_mean_full
        print("  WARNING: layers['log_norm'] missing — reconstructing from z-scored X.")

    # 1b. (Option A) Intersect 3K genes with 68K gene set
    if use_intersection:
        print(f"\n[Option A] Intersecting 3K genes ({len(genes_3k_full)}) "
              f"with 68K gene set ...")
        ad68_header = sc.read_h5ad(args.data_68k, backed="r")
        genes_68k_set = set(ad68_header.var_names)
        ad68_header.file.close()
        keep_mask = np.array([g in genes_68k_set for g in genes_3k_full])
        n_kept = int(keep_mask.sum())
        print(f"  3K ∩ 68K: {n_kept}/{len(genes_3k_full)} genes retained")
        if n_kept < 500:
            raise RuntimeError(
                f"Too few shared genes ({n_kept}) — check gene naming "
                "(symbols vs. ensembl) in both datasets."
            )
        genes_3k = [g for g, k in zip(genes_3k_full, keep_mask) if k]
        ref_mean = ref_mean_full[keep_mask]
        ref_std = ref_std_full[keep_mask]
        X_3k_z = X_3k_z_full[:, keep_mask]
        X_3k_log = X_3k_log_full[:, keep_mask]
        # Save the intersection list for reproducibility
        with open(os.path.join(args.outdir, "intersection_genes.txt"), "w") as f:
            f.write("\n".join(genes_3k))
    else:
        print("\n[Option A DISABLED] Using all 3K genes "
              "— expect 68K external eval to suffer from zero-fill bias.")
        genes_3k = genes_3k_full
        ref_mean = ref_mean_full
        ref_std = ref_std_full
        X_3k_z = X_3k_z_full
        X_3k_log = X_3k_log_full

    ref_std_safe = np.where(ref_std == 0, 1.0, ref_std)

    y_cat = ad3.obs[args.label_key].astype("category")
    y_3k = y_cat.cat.codes.to_numpy()
    class_names = list(y_cat.cat.categories)
    num_classes = len(class_names)
    print(f"  classes ({num_classes}): {class_names}")
    print(f"  class counts: {dict(zip(class_names, np.bincount(y_3k).tolist()))}")

    # 2. Stratified train/val split (within 3K)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.val_frac,
                                 random_state=args.seed)
    tr_idx, val_idx = next(sss.split(np.zeros(len(y_3k)), y_3k))
    y_tr = y_3k[tr_idx]
    y_val = y_3k[val_idx]
    print(f"  train={len(tr_idx)} val={len(val_idx)}")

    X_tr_z = X_3k_z[tr_idx]
    X_val_z = X_3k_z[val_idx]
    X_tr_log = X_3k_log[tr_idx]
    X_val_log = X_3k_log[val_idx]

    # 3. Fit expression PCA on train split
    print("\nFitting expression PCA on train split...")
    expr_pca = PCA(n_components=args.pca_dim, random_state=args.seed)
    X_tr_pca = expr_pca.fit_transform(X_tr_z).astype(np.float32)
    X_val_pca = expr_pca.transform(X_val_z).astype(np.float32)
    print(f"  expr PCA expl. var. sum = "
          f"{expr_pca.explained_variance_ratio_.sum():.4f}")
    joblib.dump(expr_pca, os.path.join(args.outdir, "expr_pca_model.pkl"))

    # 4. Fit mask PCA on train split (mask = (log_norm != 0))
    print("\nFitting mask PCA on train split...")
    X_tr_mask = (X_tr_log != 0).astype(np.float32)
    X_val_mask = (X_val_log != 0).astype(np.float32)
    print(f"  train mask non-zero fraction: {X_tr_mask.mean():.4f}")
    mask_pca = PCA(n_components=args.mask_pca_dim, random_state=args.seed)
    X_tr_mpca = mask_pca.fit_transform(X_tr_mask).astype(np.float32)
    X_val_mpca = mask_pca.transform(X_val_mask).astype(np.float32)
    print(f"  mask PCA expl. var. sum = "
          f"{mask_pca.explained_variance_ratio_.sum():.4f}")
    joblib.dump(mask_pca, os.path.join(args.outdir, "mask_pca_model.pkl"))

    X_tr_full = np.concatenate([X_tr_pca, X_tr_mpca], axis=1).astype(np.float32)
    X_val_full = np.concatenate([X_val_pca, X_val_mpca], axis=1).astype(np.float32)
    print(f"  LR/XGB input: {X_tr_pca.shape}, SANN input: {X_tr_full.shape}")

    with open(os.path.join(args.outdir, "run_config.json"), "w") as f:
        json.dump({
            "train_source": "pbmc3k",
            "gene_intersection_with_68k": bool(use_intersection),
            "n_genes_3k_original": int(len(genes_3k_full)),
            "n_genes_used": int(len(genes_3k)),
            "n_train": int(len(tr_idx)),
            "n_val": int(len(val_idx)),
            "pca_dim": int(args.pca_dim),
            "mask_pca_dim": int(args.mask_pca_dim),
            "class_names": class_names,
            "label_key": args.label_key,
            "expr_pca_expl_var": float(expr_pca.explained_variance_ratio_.sum()),
            "mask_pca_expl_var": float(mask_pca.explained_variance_ratio_.sum()),
            "sann_arch": args.sann_arch,
            "sann_n_seeds": int(args.n_sann_seeds),
            "sann_batch_size": int(args.sann_batch_size),
            "sann_patience": int(args.sann_patience),
            "sann_mixup_alpha": float(args.sann_mixup_alpha),
            "prior_correction": bool(args.prior_correction),
            "lr_class_weight": "balanced",
            "xgb_sample_weight": "balanced (1 / class_count)",
            "sann_sampler": "WeightedRandomSampler (class-balanced)",
        }, f, indent=2)

    # 5. Train models
    train_summary = []
    if "lr" in args.models:
        train_summary.append(train_lr(X_tr_pca, y_tr, X_val_pca, y_val,
                                      args.outdir))
    if "xgb" in args.models:
        train_summary.append(train_xgb(X_tr_pca, y_tr, X_val_pca, y_val,
                                       num_classes, args.outdir))
    if "sann" in args.models:
        # Clean up any old single-seed file from previous runs
        stale = os.path.join(args.outdir, "sann_model.pt")
        if os.path.exists(stale):
            os.remove(stale)
        for i in range(args.n_sann_seeds):
            seed_i = args.seed + i
            train_summary.append(train_sann(
                X_tr_full, y_tr, X_val_full, y_val, num_classes,
                args.outdir, n_expr_features=args.pca_dim, seed=seed_i,
                arch=args.sann_arch,
                batch_size=args.sann_batch_size,
                patience=args.sann_patience,
                mixup_alpha=args.sann_mixup_alpha,
                model_filename=f"sann_model_seed{i}.pt",
            ))
    with open(os.path.join(args.outdir, "train_summary.json"), "w") as f:
        json.dump(train_summary, f, indent=2, default=str)

    # 6. Prepare 8K features (aligned to 3K genes, z-scored with 3K stats)
    print("\n" + "=" * 60)
    print("  Preparing 8K external set")
    print("=" * 60)

    X_8k_log_raw, genes_8k, obs_8k = load_8k_lognorm(args.raw_8k_dir)
    X_8k_log_ref, X_8k_z_ref, shared_8k = align_and_zscore(
        X_8k_log_raw, genes_8k, genes_3k, ref_mean, ref_std_safe
    )
    print(f"  8K aligned to 3K genes: {X_8k_z_ref.shape}, "
          f"{shared_8k}/{len(genes_3k)} shared")

    X_8k_epca = expr_pca.transform(X_8k_z_ref).astype(np.float32)
    X_8k_mask = (X_8k_log_ref != 0).astype(np.float32)
    X_8k_mpca = mask_pca.transform(X_8k_mask).astype(np.float32)
    X_8k_sann = np.concatenate([X_8k_epca, X_8k_mpca], axis=1).astype(np.float32)
    print(f"  8K expr_pca shape={X_8k_epca.shape}, "
          f"mask non-zero frac={X_8k_mask.mean():.4f}")

    # Load 8K labels from preprocessed file, restrict to raw-reload rows
    ad8 = sc.read_h5ad(args.data_8k)
    raw_idx_8k = {n: i for i, n in enumerate(obs_8k)}
    kept_8k = [n for n in ad8.obs_names if n in raw_idx_8k]
    if len(kept_8k) < ad8.shape[0]:
        print(f"  WARNING: {ad8.shape[0] - len(kept_8k)} labeled 8K cells "
              "dropped (not present in raw-reload QC survivors)")
    eval_rows_8k = np.array([raw_idx_8k[n] for n in kept_8k])
    y_8k_cat = ad8.obs.loc[kept_8k, "cell_type"].astype(str)

    LABEL_ALIAS_8K = {"CD8 T": "T cells"}
    label_map_8k = {}
    for name in y_8k_cat.unique():
        if name in class_names:
            label_map_8k[name] = class_names.index(name)
        elif name in LABEL_ALIAS_8K and LABEL_ALIAS_8K[name] in class_names:
            label_map_8k[name] = class_names.index(LABEL_ALIAS_8K[name])
        else:
            label_map_8k[name] = -1
    print(f"  8K label map: {label_map_8k}")
    y_8k_mapped = np.array([label_map_8k[n] for n in y_8k_cat])
    known_8k = y_8k_mapped >= 0
    known_cls_8k = sorted(set(int(v) for v in y_8k_mapped[known_8k].tolist()))
    print(f"  8K cells: {len(y_8k_mapped)} total, {known_8k.sum()} known")

    X_8k_epca_e = X_8k_epca[eval_rows_8k]
    X_8k_sann_e = X_8k_sann[eval_rows_8k]

    # 7. Prepare 68K features
    print("\n" + "=" * 60)
    print("  Preparing 68K external set")
    print("=" * 60)

    ad68 = sc.read_h5ad(args.data_68k)
    X_68k_log_raw, genes_68k = reconstruct_68k_lognorm(ad68)
    X_68k_log_ref, X_68k_z_ref, shared_68k = align_and_zscore(
        X_68k_log_raw, genes_68k, genes_3k, ref_mean, ref_std_safe
    )
    print(f"  68K aligned to 3K genes: {X_68k_z_ref.shape}, "
          f"{shared_68k}/{len(genes_3k)} shared")

    X_68k_epca = expr_pca.transform(X_68k_z_ref).astype(np.float32)
    X_68k_mask = (X_68k_log_ref != 0).astype(np.float32)
    X_68k_mpca = mask_pca.transform(X_68k_mask).astype(np.float32)
    X_68k_sann = np.concatenate([X_68k_epca, X_68k_mpca],
                                axis=1).astype(np.float32)

    y_68k_cat = ad68.obs["cell_type"].astype(str)
    class_names_68k = sorted(y_68k_cat.unique())
    print(f"  68K classes ({len(class_names_68k)}): {class_names_68k}")

    LABEL_ALIAS_68K = {
        "CD4 T": "T cells", "CD8 T": "T cells", "Treg": "T cells",
        "NK cells": "NK", "NK": "NK",
        "B cells": "B cells", "B": "B cells", "Plasma": "B cells",
        "Mono": "Mono", "Monocytes": "Mono", "CD14+ Monocytes": "Mono",
        "FCGR3A+ Monocytes": "Mono",
        "DC": "DC", "Dendritic": "DC", "Dendritic cells": "DC",
        "Platelet": "Platelet", "Megakaryocytes": "Platelet",
    }

    def resolve_68k(name):
        if name in class_names:
            return class_names.index(name)
        if name in LABEL_ALIAS_68K and LABEL_ALIAS_68K[name] in class_names:
            return class_names.index(LABEL_ALIAS_68K[name])
        n = str(name)
        if ("T cell" in n or "CD4" in n or "CD8" in n or n.startswith("T ")) \
                and "T cells" in class_names:
            return class_names.index("T cells")
        if "NK" in n and "NK" in class_names:
            return class_names.index("NK")
        if ("B cell" in n or n.startswith("B ") or n == "B") \
                and "B cells" in class_names:
            return class_names.index("B cells")
        if "Mono" in n and "Mono" in class_names:
            return class_names.index("Mono")
        if "DC" in n and "DC" in class_names:
            return class_names.index("DC")
        if ("Platelet" in n or "Megakary" in n) and "Platelet" in class_names:
            return class_names.index("Platelet")
        return -1

    label_map_68k = {name: resolve_68k(name) for name in class_names_68k}
    print(f"  68K label map: {label_map_68k}")
    y_68k_mapped = np.array([label_map_68k[n] for n in y_68k_cat])
    known_68k = y_68k_mapped >= 0
    known_cls_68k = sorted(set(int(v) for v in y_68k_mapped[known_68k].tolist()))
    print(f"  68K cells: {len(y_68k_mapped)} total, {known_68k.sum()} known")
    unknown_68k = [n for n in class_names_68k if label_map_68k[n] == -1]
    if unknown_68k:
        print(f"  68K types with no 3K counterpart (excluded): {unknown_68k}")

    # 8. Run predictions
    external_rows = []

    # Training class prior for optional label-shift correction
    train_prior = np.bincount(y_tr, minlength=num_classes).astype(np.float64)
    train_prior = np.maximum(train_prior, 1.0) / train_prior.sum()
    np.save(os.path.join(args.outdir, "train_prior.npy"), train_prior)
    if args.prior_correction:
        print(f"\n[Prior correction] train prior: "
              f"{dict(zip(class_names, np.round(train_prior, 4)))}")

    def _apply_prior(probs):
        if not args.prior_correction:
            return probs
        adj = probs / train_prior[None, :]
        adj = adj / adj.sum(axis=1, keepdims=True)
        return adj.astype(np.float32)

    def _run_lr(X_pca_feats, y_mapped, known_mask, known_cls, tag):
        lr_model = joblib.load(os.path.join(args.outdir, "lr_model.pkl"))
        lr_scaler = joblib.load(os.path.join(args.outdir, "lr_scaler.pkl"))
        probs = lr_model.predict_proba(lr_scaler.transform(X_pca_feats))
        probs = _apply_prior(probs)
        pred = probs.argmax(axis=1)
        return evaluate_external(
            "LR", y_mapped[known_mask], pred[known_mask],
            probs[known_mask], class_names, args.outdir, tag,
            known_class_indices=known_cls,
        )

    def _run_xgb(X_pca_feats, y_mapped, known_mask, known_cls, tag):
        booster = xgb.Booster()
        booster.load_model(os.path.join(args.outdir, "xgb_model.json"))
        probs = booster.predict(xgb.DMatrix(X_pca_feats))
        probs = _apply_prior(probs)
        pred = probs.argmax(axis=1)
        return evaluate_external(
            "XGB", y_mapped[known_mask], pred[known_mask],
            probs[known_mask], class_names, args.outdir, tag,
            known_class_indices=known_cls,
        )

    def _run_sann(X_sann_feats, y_mapped, known_mask, known_cls, tag):
        probs = predict_sann(args.outdir, args.pca_dim, args.mask_pca_dim,
                             num_classes, X_sann_feats, arch=args.sann_arch)
        probs = _apply_prior(probs)
        pred = probs.argmax(axis=1)
        return evaluate_external(
            "SANN", y_mapped[known_mask], pred[known_mask],
            probs[known_mask], class_names, args.outdir, tag,
            known_class_indices=known_cls,
        )

    if "lr" in args.models:
        external_rows.append(_run_lr(X_8k_epca_e, y_8k_mapped,
                                     known_8k, known_cls_8k, "pbmc8k"))
        external_rows.append(_run_lr(X_68k_epca, y_68k_mapped,
                                     known_68k, known_cls_68k, "pbmc68k"))
    if "xgb" in args.models:
        external_rows.append(_run_xgb(X_8k_epca_e, y_8k_mapped,
                                      known_8k, known_cls_8k, "pbmc8k"))
        external_rows.append(_run_xgb(X_68k_epca, y_68k_mapped,
                                      known_68k, known_cls_68k, "pbmc68k"))
    if "sann" in args.models:
        external_rows.append(_run_sann(X_8k_sann_e, y_8k_mapped,
                                       known_8k, known_cls_8k, "pbmc8k"))
        external_rows.append(_run_sann(X_68k_sann, y_68k_mapped,
                                       known_68k, known_cls_68k, "pbmc68k"))

    # 9. Summary
    summary = pd.DataFrame(external_rows)
    summary_path = os.path.join(args.outdir, "external_eval_summary.csv")
    summary.to_csv(summary_path, index=False)
    print("\n" + "=" * 60)
    print("  SUMMARY — 3K-trained models on external test sets")
    print("=" * 60)
    print(summary.to_string(index=False))
    print(f"\nSaved: {summary_path}")
    print(f"Artifacts: {args.outdir}/")


if __name__ == "__main__":
    main()
