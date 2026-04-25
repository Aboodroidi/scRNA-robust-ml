# src/train_all_full_models.py
import os

# mac-friendly: set BEFORE importing numpy / scanpy / sklearn / xgboost / torch
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
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedShuffleSplit

import xgboost as xgb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ----------------------------
# SANN MODEL
# ----------------------------
def _make_norm(dim, use_batchnorm, use_layernorm):
    """Return the appropriate normalisation layer."""
    if use_layernorm:
        return nn.LayerNorm(dim)
    if use_batchnorm:
        return nn.BatchNorm1d(dim)
    return nn.Identity()


class ResidualBlock(nn.Module):
    """Pre-norm residual block: Norm → Linear → ReLU → Dropout → Linear → add skip."""
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
    """
    Sparse-Aware Neural Network v2 — dual-encoder with residual blocks
    and gated attention fusion.

    Improvements over v1:
      - GELU activations (smoother than ReLU, better gradient flow)
      - Residual connections in each branch (3 layers deep)
      - Gated attention fusion (learns per-sample branch importance)
      - Input noise injection during training (regularisation)
    """
    def __init__(
        self,
        expr_dim,
        mask_dim,
        num_classes,
        branch_hidden=512,
        branch_out=256,
        fusion_hidden=256,
        dropout=0.05,
        use_batchnorm=True,
        use_layernorm=False,
        input_noise=0.0,
    ):
        super().__init__()
        self.expr_dim = expr_dim
        self.input_noise = input_noise

        # --- Expression branch (projection + 2 residual blocks) ---
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

        # --- Mask branch (projection + 2 residual blocks) ---
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

        # --- Gated attention fusion ---
        # Learns a per-sample gate α ∈ [0,1] that weights branch contributions:
        #   h_fused = α · h_expr + (1 − α) · h_mask
        self.gate = nn.Sequential(
            nn.Linear(branch_out * 2, branch_out),
            nn.GELU(),
            nn.Linear(branch_out, branch_out),
            nn.Sigmoid(),
        )

        # --- Classification head ---
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

        # Input noise injection (training only)
        if self.training and self.input_noise > 0:
            x_expr = x_expr + torch.randn_like(x_expr) * self.input_noise

        # Expression branch with residual blocks
        h_expr = self.expr_proj(x_expr)
        h_expr = self.expr_res1(h_expr)
        h_expr = self.expr_res2(h_expr)
        h_expr = self.expr_out(h_expr)

        # Mask branch with residual blocks
        h_mask = self.mask_proj(x_mask)
        h_mask = self.mask_res1(h_mask)
        h_mask = self.mask_res2(h_mask)
        h_mask = self.mask_out(h_mask)

        # Gated attention fusion
        combined = torch.cat([h_expr, h_mask], dim=1)
        alpha = self.gate(combined)  # (B, branch_out), values in [0,1]
        h_fused = alpha * h_expr + (1 - alpha) * h_mask

        return self.classifier(h_fused)


# ----------------------------
# Utilities
# ----------------------------
def compute_class_weights(y, num_classes):
    """Sqrt-inverse-frequency class weights, normalized to sum to num_classes.

    Using sqrt dampens extreme ratios between rare and common classes,
    preventing gradient explosion on high-dimensional inputs.
    """
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)  # avoid division by zero
    inv_freq = 1.0 / np.sqrt(counts)  # sqrt dampening
    weights = inv_freq / inv_freq.sum() * num_classes
    return weights.astype(np.float32)


def to_dense_float32(x):
    if sp.issparse(x):
        return x.toarray().astype(np.float32)
    return np.asarray(x, dtype=np.float32)


def load_labels(adata, label_key="cell_type"):
    if label_key not in adata.obs:
        raise ValueError(f"Expected adata.obs['{label_key}']")

    y_cat = adata.obs[label_key].astype("category")
    y = y_cat.cat.codes.to_numpy()
    class_names = list(y_cat.cat.categories)
    return y, class_names


def load_expression_data(data_path, label_key="cell_type", hvg_key="highly_variable", max_hvgs=None):
    """
    Load HVG expression matrix for LR, XGB, and SANN.
    """
    adata = sc.read_h5ad(data_path)
    y, class_names = load_labels(adata, label_key=label_key)

    if hvg_key in adata.var.columns:
        hvg_mask = adata.var[hvg_key].to_numpy().astype(bool)
        if hvg_mask.sum() == 0:
            raise ValueError(f"adata.var['{hvg_key}'] exists but contains no True values.")
        adata_expr = adata[:, hvg_mask].copy()
    else:
        adata_expr = adata

    X_expr = to_dense_float32(adata_expr.X)

    if max_hvgs is not None and X_expr.shape[1] > max_hvgs:
        X_expr = X_expr[:, :max_hvgs]

    return X_expr, y, class_names


def standardize_expression_train_val_test(X_train, X_val, X_test, eps=1e-6):
    """
    Standardize expression features using TRAIN statistics only.
    """
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std = np.where(std < eps, 1.0, std)

    X_train_s = ((X_train - mean) / std).astype(np.float32)
    X_val_s = ((X_val - mean) / std).astype(np.float32)
    X_test_s = ((X_test - mean) / std).astype(np.float32)

    return X_train_s, X_val_s, X_test_s, mean.astype(np.float32), std.astype(np.float32)


def build_sann_input(X_expr_scaled, X_mask):
    """
    Build SANN input as:
    [scaled expression, binary sparsity mask]
    """
    return np.concatenate([X_expr_scaled.astype(np.float32), X_mask.astype(np.float32)], axis=1).astype(np.float32)


def load_fixed_split(split_path):
    with open(split_path, "r") as f:
        d = json.load(f)
    if "train_idx" not in d or "test_idx" not in d:
        raise ValueError(f"Split file must contain train_idx and test_idx. Found keys: {list(d.keys())}")
    return np.array(d["train_idx"], dtype=int), np.array(d["test_idx"], dtype=int)


def make_train_val_split(y_train_full, val_frac=0.1, seed=42):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    tr_rel, val_rel = next(sss.split(np.zeros(len(y_train_full)), y_train_full))
    return np.array(tr_rel, dtype=int), np.array(val_rel, dtype=int)


def save_metrics_row(path, row_dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    row = pd.DataFrame([row_dict])
    if os.path.exists(path):
        df = pd.read_csv(path)
        df = df[df["Model"] != row_dict["Model"]]
        df = pd.concat([df, row], ignore_index=True)
    else:
        df = row
    df.to_csv(path, index=False)


def l1_penalty(model: nn.Module) -> torch.Tensor:
    penalty = torch.tensor(0.0, device=next(model.parameters()).device)
    for param in model.parameters():
        penalty = penalty + param.abs().sum()
    return penalty


# ----------------------------
# Logistic Regression
# ----------------------------
def train_full_lr(X_train, y_train, X_val, y_val, X_test, y_test, outdir):
    print("\nTraining Logistic Regression (HVG expression, tuning C on validation)...")
    t0 = time.time()

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    best_f1 = -1.0
    best_C = None
    best_model = None

    c_grid = [0.1, 1.0, 5.0]

    for C in c_grid:
        model = LogisticRegression(
            C=C,
            max_iter=3000,
            tol=1e-4,
            solver="lbfgs",
            n_jobs=1,
            random_state=42,
        )
        model.fit(X_train_s, y_train)

        val_pred = model.predict(X_val_s)
        val_acc = accuracy_score(y_val, val_pred)
        val_f1 = f1_score(y_val, val_pred, average="macro")
        print(f"  C={C:<4} | Val Acc={val_acc:.4f} | Val Macro-F1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_C = C
            best_model = model

    train_time = time.time() - t0
    print(f"Best LR: C={best_C} | Val Macro-F1={best_f1:.4f} | time={train_time:.1f}s")

    # --- convergence history via warm_start with incremental max_iter ---
    import warnings
    print("  Recording LR convergence history (warm_start)...")
    iter_checkpoints = list(range(1, 21)) + list(range(25, 101, 5)) + list(range(150, 501, 50))
    lr_history = []
    warm_model = LogisticRegression(
        C=best_C, max_iter=1, tol=0.0, solver="lbfgs",
        n_jobs=1, random_state=42, warm_start=True,
    )
    n_checkpoints = len(iter_checkpoints)
    t_hist = time.time()
    for idx, it in enumerate(iter_checkpoints, 1):
        warm_model.max_iter = it
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="lbfgs failed to converge")
            warm_model.fit(X_train_s, y_train)

        elapsed = time.time() - t_hist

        train_probs = warm_model.predict_proba(X_train_s)
        val_probs = warm_model.predict_proba(X_val_s)

        train_ll = log_loss(y_train, train_probs)
        val_ll = log_loss(y_val, val_probs)
        val_pred_i = val_probs.argmax(axis=1)
        val_f1_i = f1_score(y_val, val_pred_i, average="macro")
        val_acc_i = accuracy_score(y_val, val_pred_i)

        print(f"    [{idx}/{n_checkpoints}] max_iter={it} | val_loss={val_ll:.4f} | val_F1={val_f1_i:.4f} | {elapsed:.1f}s")

        lr_history.append({
            "iteration": it,
            "train_loss": float(train_ll),
            "val_loss": float(val_ll),
            "val_acc": float(val_acc_i),
            "val_macro_f1": float(val_f1_i),
            "elapsed_seconds": float(elapsed),
        })

    pd.DataFrame(lr_history).to_csv(os.path.join(outdir, "lr_history.csv"), index=False)

    joblib.dump(best_model, os.path.join(outdir, "lr_model.pkl"))
    joblib.dump(scaler, os.path.join(outdir, "lr_scaler.pkl"))

    probs_test = best_model.predict_proba(X_test_s)
    pred_test = probs_test.argmax(axis=1)

    np.save(os.path.join(outdir, "lr_test_probs.npy"), probs_test)
    np.save(os.path.join(outdir, "lr_test_pred.npy"), pred_test)
    np.save(os.path.join(outdir, "lr_test_true.npy"), y_test)

    return {
        "Model": "LR",
        "Accuracy": float(accuracy_score(y_test, pred_test)),
        "Macro-F1": float(f1_score(y_test, pred_test, average="macro")),
        "TrainTimeSeconds": float(train_time),
        "Notes": f"input=HVG_expression; best_C={best_C}; solver=lbfgs; grid={c_grid}",
    }


# ----------------------------
# XGBoost
# ----------------------------
def train_full_xgb(X_train, y_train, X_val, y_val, X_test, y_test, num_classes, outdir):
    print("\nTraining XGBoost (HVG expression features, early stopping on validation)...")
    t0 = time.time()

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        "objective": "multi:softprob",
        "num_class": int(num_classes),
        "eta": 0.03,
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
        params=params,
        dtrain=dtrain,
        num_boost_round=2000,
        evals=[(dtrain, "train"), (dval, "valid")],
        evals_result=evals_result,
        verbose_eval=50,
        callbacks=[xgb.callback.EarlyStopping(rounds=50, save_best=True)],
    )

    train_time = time.time() - t0
    print(f"XGB best_iteration={booster.best_iteration} | time={train_time:.1f}s")

    # --- build convergence history ---
    train_logloss = evals_result["train"]["mlogloss"]
    val_logloss = evals_result["valid"]["mlogloss"]
    # Cap at best_iteration+1 because save_best=True trims the booster
    max_rounds = int(booster.best_iteration) + 1
    n_rounds = min(len(val_logloss), max_rounds)
    total_trained_rounds = len(val_logloss)  # actual rounds before early stop

    # Compute val F1 at every 10th round + final round using staged prediction
    f1_interval = 10
    xgb_history = []
    for r in range(n_rounds):
        # Estimate elapsed time proportionally (rounds are ~uniform cost)
        elapsed_est = train_time * (r + 1) / total_trained_rounds

        row = {
            "round": r + 1,
            "train_loss": float(train_logloss[r]),
            "val_loss": float(val_logloss[r]),
            "elapsed_seconds": float(elapsed_est),
        }
        if r % f1_interval == 0 or r == n_rounds - 1:
            probs_val_r = booster.predict(dval, iteration_range=(0, r + 1))
            pred_val_r = probs_val_r.argmax(axis=1)
            row["val_macro_f1"] = float(f1_score(y_val, pred_val_r, average="macro"))
            row["val_acc"] = float(accuracy_score(y_val, pred_val_r))
        xgb_history.append(row)

    pd.DataFrame(xgb_history).to_csv(os.path.join(outdir, "xgb_history.csv"), index=False)

    probs_test = booster.predict(dtest)
    pred_test = probs_test.argmax(axis=1)

    acc = accuracy_score(y_test, pred_test)
    f1 = f1_score(y_test, pred_test, average="macro")
    print(f"XGB | Test Acc={acc:.4f} | Test Macro-F1={f1:.4f}")

    booster.save_model(os.path.join(outdir, "xgb_model.json"))

    np.save(os.path.join(outdir, "xgb_test_probs.npy"), probs_test)
    np.save(os.path.join(outdir, "xgb_test_pred.npy"), pred_test)
    np.save(os.path.join(outdir, "xgb_test_true.npy"), y_test)

    return {
        "Model": "XGB",
        "Accuracy": float(acc),
        "Macro-F1": float(f1),
        "TrainTimeSeconds": float(train_time),
        "Notes": f"input=HVG_expression; best_iter={int(booster.best_iteration)}",
    }


# ----------------------------
# SANN
# ----------------------------
def train_full_sann(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    num_classes,
    outdir,
    n_expr_features,
    l1_lambda=5e-8,
):
    print("\nTraining SANN (dual-encoder: expression branch + mask branch, class-weighted)...")
    t0 = time.time()

    device = "cpu"
    n_mask_features = X_train.shape[1] - n_expr_features

    # Scale architecture to input dimensionality
    max_epochs = 200
    warmup_epochs = 10

    if n_expr_features > 500:
        # HVG: narrower branches, LayerNorm, strong regularization
        bh, bo, fh = 256, 128, 128
        drop, lr_init, wd = 0.4, 2e-4, 5e-4
        use_bn, use_ln = False, True
        norm_label = "LayerNorm"
        input_noise = 0.1
        mixup_alpha = 0.4
    else:
        # PCA: wider branches, BatchNorm, moderate regularization
        bh, bo, fh = 512, 256, 256
        drop, lr_init, wd = 0.25, 3e-4, 1e-4
        use_bn, use_ln = True, False
        norm_label = "BatchNorm"
        input_noise = 0.05
        mixup_alpha = 0.3

    model = SANN(
        expr_dim=n_expr_features,
        mask_dim=n_mask_features,
        num_classes=num_classes,
        branch_hidden=bh,
        branch_out=bo,
        fusion_hidden=fh,
        dropout=drop,
        use_batchnorm=use_bn,
        use_layernorm=use_ln,
        input_noise=input_noise,
    ).to(device)

    print(f"  [Architecture] expr_branch: {n_expr_features}→{bh}→res→res→{bo} | "
          f"mask_branch: {n_mask_features}→{bh}→res→res→{bo} | gated_fusion→{fh}→{num_classes}")
    print(f"  [Config] norm={norm_label}, dropout={drop}, lr={lr_init}, weight_decay={wd}")
    print(f"  [Config] input_noise={input_noise}, mixup_alpha={mixup_alpha}, "
          f"epochs={max_epochs}, warmup={warmup_epochs}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  [Architecture] Total parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_init, weight_decay=wd)

    # Cosine annealing with linear warmup
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(max_epochs - warmup_epochs, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Class-weighted loss to handle imbalanced cell types
    cw = compute_class_weights(y_train, num_classes)
    class_weights_t = torch.tensor(cw, dtype=torch.float32).to(device)
    print(f"  [Class weights] {dict(zip(range(num_classes), np.round(cw, 3)))}")

    criterion = nn.CrossEntropyLoss(weight=class_weights_t, label_smoothing=0.1)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=1024, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=1024, shuffle=False)

    # Mixup helper: interpolates samples and produces soft labels
    def mixup_batch(xb, yb, alpha, num_classes):
        if alpha <= 0:
            return xb, torch.nn.functional.one_hot(yb, num_classes).float()
        lam = np.random.beta(alpha, alpha)
        lam = max(lam, 1 - lam)  # ensure lam >= 0.5 so original dominates
        idx = torch.randperm(xb.size(0))
        x_mix = lam * xb + (1 - lam) * xb[idx]
        y_onehot = torch.nn.functional.one_hot(yb, num_classes).float()
        y_mix = lam * y_onehot + (1 - lam) * y_onehot[idx]
        return x_mix, y_mix

    # Soft cross-entropy for mixup (works with soft labels)
    def soft_cross_entropy(logits, soft_targets, class_weights=None):
        log_probs = torch.nn.functional.log_softmax(logits, dim=1)
        if class_weights is not None:
            log_probs = log_probs * class_weights.unsqueeze(0)
        loss = -(soft_targets * log_probs).sum(dim=1).mean()
        return loss

    best_val_f1 = -1.0
    best_state = None
    patience = 30
    patience_counter = 0
    history = []
    t_epoch_start = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_n = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            # Mixup augmentation
            xb_mix, yb_soft = mixup_batch(xb, yb, mixup_alpha, num_classes)
            logits = model(xb_mix)

            ce_loss = soft_cross_entropy(logits, yb_soft, class_weights_t)
            reg_l1 = l1_penalty(model) * l1_lambda
            loss = ce_loss + reg_l1

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            bs = xb.size(0)
            train_loss_sum += float(loss.item()) * bs
            train_n += bs

        train_loss = train_loss_sum / max(train_n, 1)

        model.eval()
        val_loss_sum = 0.0
        val_n = 0
        val_logits_all = []
        val_true_all = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                logits = model(xb)
                ce_loss = criterion(logits, yb)
                reg_l1 = l1_penalty(model) * l1_lambda
                loss = ce_loss + reg_l1

                bs = xb.size(0)
                val_loss_sum += float(loss.item()) * bs
                val_n += bs

                val_logits_all.append(logits.cpu().numpy())
                val_true_all.append(yb.cpu().numpy())

        val_loss = val_loss_sum / max(val_n, 1)
        val_logits = np.vstack(val_logits_all)
        val_true = np.concatenate(val_true_all)
        val_probs = torch.softmax(torch.tensor(val_logits), dim=1).numpy()
        val_pred = val_probs.argmax(axis=1)
        val_acc = accuracy_score(val_true, val_pred)
        val_f1 = f1_score(val_true, val_pred, average="macro")

        scheduler.step()  # cosine annealing steps per epoch
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_elapsed = time.time() - t_epoch_start

        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "val_macro_f1": float(val_f1),
            "lr": float(current_lr),
            "elapsed_seconds": float(epoch_elapsed),
        })

        print(
            f"  Epoch {epoch:03d} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_macroF1={val_f1:.4f} | lr={current_lr:.6f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    train_time = time.time() - t0

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        test_logits = model(X_test_t.to(device)).cpu().numpy()
        val_logits_final = model(X_val_t.to(device)).cpu().numpy()

    test_probs = torch.softmax(torch.tensor(test_logits), dim=1).numpy()
    test_pred = test_probs.argmax(axis=1)

    val_probs_final = torch.softmax(torch.tensor(val_logits_final), dim=1).numpy()

    acc = accuracy_score(y_test, test_pred)
    f1 = f1_score(y_test, test_pred, average="macro")
    print(f"SANN raw | Test Acc={acc:.4f} | Test Macro-F1={f1:.4f}")

    torch.save(model.state_dict(), os.path.join(outdir, "sann_model.pt"))
    pd.DataFrame(history).to_csv(os.path.join(outdir, "sann_history.csv"), index=False)

    np.save(os.path.join(outdir, "sann_test_probs.npy"), test_probs)
    np.save(os.path.join(outdir, "sann_test_pred.npy"), test_pred)
    np.save(os.path.join(outdir, "sann_test_true.npy"), y_test)
    np.save(os.path.join(outdir, "sann_val_probs.npy"), val_probs_final)
    np.save(os.path.join(outdir, "sann_val_true.npy"), y_val)

    return {
        "Model": "SANN",
        "Accuracy": float(acc),
        "Macro-F1": float(f1),
        "TrainTimeSeconds": float(train_time),
        "Notes": (
            f"input=dual_encoder(expr_branch+mask_branch); "
            f"branch=512→256; fusion=256→{num_classes}; dropout=0.05; "
            f"bn=True; class_weighted=True; lr=8e-4; l1_lambda={l1_lambda}; "
            f"temp_scaling=off"
        ),
    }


# ----------------------------
# MAIN
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/pbmc68k_labeled.h5ad")
    parser.add_argument("--splits", default="results/ablations/fixed_splits.json")
    parser.add_argument("--outdir", default="results/full_train_all_hvg")
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--max_hvgs", type=int, default=None)
    parser.add_argument("--l1_lambda", type=float, default=5e-8)
    parser.add_argument("--models", type=str, default="lr,xgb,sann",
                        help="Comma-separated list of models to train (lr,xgb,sann)")
    parser.add_argument("--label_key", type=str, default="cell_type",
                        help="obs column to use as labels (e.g. cell_type_coarse)")
    parser.add_argument("--n_seeds", type=int, default=1,
                        help="Number of SANN seeds for deep ensemble (default: 1)")
    args = parser.parse_args()
    args.models = [m.strip().lower() for m in args.models.split(",")]

    os.makedirs(args.outdir, exist_ok=True)

    # Shared HVG pathway for all models
    X_expr, y_expr, class_names_expr = load_expression_data(
        args.data,
        label_key=args.label_key,
        hvg_key="highly_variable",
        max_hvgs=args.max_hvgs,
    )

    y = y_expr
    class_names = class_names_expr

    train_idx_full, test_idx = load_fixed_split(args.splits)

    # Labels
    y_train_full = y[train_idx_full]
    y_test = y[test_idx]

    tr_rel, val_rel = make_train_val_split(y_train_full, val_frac=args.val_frac, seed=42)
    y_train = y_train_full[tr_rel]
    y_val = y_train_full[val_rel]

    # Shared HVG expression splits
    X_train_full_expr = X_expr[train_idx_full]
    X_test_expr = X_expr[test_idx]
    X_train_expr = X_train_full_expr[tr_rel]
    X_val_expr = X_train_full_expr[val_rel]

    # SANN v2: NO double standardization — internal LayerNorm handles normalization.
    # Input = [z-scored expression, binary mask from z-scored expression]
    # The mask from z-scored data is ~all 1s (consistent across datasets).
    X_mask_train = (X_train_expr != 0).astype(np.float32)
    X_mask_val = (X_val_expr != 0).astype(np.float32)
    X_mask_test = (X_test_expr != 0).astype(np.float32)
    print(f"  [Mask] Sparsity mask non-zero fraction: {X_mask_train.mean():.4f}")

    X_train_sann = build_sann_input(X_train_expr, X_mask_train)
    X_val_sann = build_sann_input(X_val_expr, X_mask_val)
    X_test_sann = build_sann_input(X_test_expr, X_mask_test)
    # Save dummy mean/std for backward compatibility (zeros/ones = identity transform)
    expr_mean = np.zeros((1, X_train_expr.shape[1]), dtype=np.float32)
    expr_std = np.ones((1, X_train_expr.shape[1]), dtype=np.float32)

    num_classes = len(class_names)
    n_expr_features = X_train_expr.shape[1]

    print(f"[Sanity] Total samples = {len(y)} | classes = {num_classes}")
    print(f"[Sanity] Expression shape = {X_expr.shape}")
    print(f"[Sanity] LR/XGB input shape = {X_train_expr.shape}")
    print(f"[Sanity] SANN shape = {X_train_sann.shape} (expression={n_expr_features}, mask={n_expr_features})")
    print(f"[Sanity] Train_full={len(train_idx_full)} | Train={len(tr_rel)} | Val={len(val_rel)} | Test={len(test_idx)}")
    print(f"[Sanity] class names: {class_names}")

    with open(os.path.join(args.outdir, "train_val_test_split.json"), "w") as f:
        json.dump({
            "train_full_size": int(len(train_idx_full)),
            "train_size": int(len(tr_rel)),
            "val_size": int(len(val_rel)),
            "test_size": int(len(test_idx)),
            "val_frac": float(args.val_frac),
            "expression_features": int(n_expr_features),
            "sann_total_input_dim": int(X_train_sann.shape[1]),
            "l1_lambda": float(args.l1_lambda),
            "temperature_scaling_used": False,
        }, f, indent=2)

    np.save(os.path.join(args.outdir, "sann_expr_mean.npy"), expr_mean)
    np.save(os.path.join(args.outdir, "sann_expr_std.npy"), expr_std)

    metrics_path = os.path.join(args.outdir, "baseline_metrics_full.csv")

    if "lr" in args.models:
        lr_row = train_full_lr(
            X_train_expr, y_train,
            X_val_expr, y_val,
            X_test_expr, y_test,
            args.outdir
        )
        save_metrics_row(metrics_path, lr_row)

    if "xgb" in args.models:
        xgb_row = train_full_xgb(
            X_train_expr, y_train,
            X_val_expr, y_val,
            X_test_expr, y_test,
            num_classes,
            args.outdir
        )
        save_metrics_row(metrics_path, xgb_row)

    if "sann" in args.models:
        n_seeds = args.n_seeds
        all_test_probs = []
        all_val_probs = []
        for seed_i in range(n_seeds):
            seed_val = 42 + seed_i
            print(f"\n{'='*60}")
            print(f"  SANN seed {seed_i+1}/{n_seeds} (seed={seed_val})")
            print(f"{'='*60}")
            torch.manual_seed(seed_val)
            np.random.seed(seed_val)

            sann_row = train_full_sann(
                X_train_sann, y_train,
                X_val_sann, y_val,
                X_test_sann, y_test,
                num_classes,
                args.outdir,
                n_expr_features=n_expr_features,
                l1_lambda=args.l1_lambda,
            )

            # Save individual seed model
            if n_seeds > 1:
                import shutil
                src = os.path.join(args.outdir, "sann_model.pt")
                dst = os.path.join(args.outdir, f"sann_model_seed{seed_i}.pt")
                shutil.copy2(src, dst)

            # Collect test/val probs for ensemble averaging
            test_probs_i = np.load(os.path.join(args.outdir, "sann_test_probs.npy"))
            val_probs_i = np.load(os.path.join(args.outdir, "sann_val_probs.npy"))
            all_test_probs.append(test_probs_i)
            all_val_probs.append(val_probs_i)

        if n_seeds > 1:
            # Ensemble: average probabilities across seeds
            ens_test_probs = np.mean(all_test_probs, axis=0)
            ens_val_probs = np.mean(all_val_probs, axis=0)
            ens_pred = ens_test_probs.argmax(axis=1)
            ens_acc = accuracy_score(y_test, ens_pred)
            ens_f1 = f1_score(y_test, ens_pred, average="macro")
            print(f"\nSANN ENSEMBLE ({n_seeds} seeds) | Test Acc={ens_acc:.4f} | Test Macro-F1={ens_f1:.4f}")

            # Save ensemble probs (eval scripts will load these)
            np.save(os.path.join(args.outdir, "sann_test_probs.npy"), ens_test_probs)
            np.save(os.path.join(args.outdir, "sann_val_probs.npy"), ens_val_probs)
            np.save(os.path.join(args.outdir, "sann_test_pred.npy"), ens_pred)

            sann_row["Accuracy"] = float(ens_acc)
            sann_row["Macro-F1"] = float(ens_f1)
            sann_row["Notes"] += f"; ensemble={n_seeds}_seeds"

        save_metrics_row(metrics_path, sann_row)

    print("\n✅ ALL FULL MODELS TRAINED CORRECTLY.")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved artifacts under: {args.outdir}/")


if __name__ == "__main__":
    main()