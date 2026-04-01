import os
import time
import argparse

# ----------------------------
# Mac-friendly single-thread setup
# ----------------------------
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MPLBACKEND"] = "Agg"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import scanpy as sc

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

import xgboost as xgb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ----------------------------
# Models
# ----------------------------
class SANNPCA(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=256, dropout=0.1, activation="relu", batchnorm=True):
        super().__init__()
        act = nn.ReLU() if activation.lower() == "relu" else nn.GELU()

        layers = [nn.Linear(input_dim, hidden_dim)]
        if batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(act)
        if dropout and dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SANNHVG(nn.Module):
    def __init__(self, expr_dim, num_classes, hidden_dim=256, dropout=0.1, activation="relu", batchnorm=True):
        super().__init__()
        act = nn.ReLU() if activation.lower() == "relu" else nn.GELU()

        input_dim = expr_dim * 2  # expression + binary mask

        layers = [nn.Linear(input_dim, hidden_dim)]
        if batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(act)
        if dropout and dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x_expr, x_mask):
        x = torch.cat([x_expr, x_mask], dim=1)
        return self.net(x)


# ----------------------------
# Utilities
# ----------------------------
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_splits(y, n_splits: int, test_size: float, seeds):
    splits = []
    for i in range(n_splits):
        seed = seeds[i]
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(sss.split(np.zeros_like(y), y))
        splits.append({
            "split_id": f"split_{i+1:02d}",
            "seed": seed,
            "train_idx": train_idx.tolist(),
            "test_idx": test_idx.tolist()
        })
    return splits


def load_pca_xy(adata_path: str, label_key: str, pca_key: str, pca_dim: int):
    adata = sc.read_h5ad(adata_path)
    if label_key not in adata.obs:
        raise ValueError(f"Missing label column adata.obs['{label_key}']")

    y_cat = adata.obs[label_key].astype("category")
    y = y_cat.cat.codes.to_numpy()
    class_names = list(y_cat.cat.categories)

    if pca_key not in adata.obsm:
        raise ValueError(f"Missing PCA matrix adata.obsm['{pca_key}']")

    X = np.asarray(adata.obsm[pca_key][:, :pca_dim], dtype=np.float32)
    return X, y, class_names


def load_hvg_xy(adata_path: str, label_key: str):
    adata = sc.read_h5ad(adata_path)
    if label_key not in adata.obs:
        raise ValueError(f"Missing label column adata.obs['{label_key}']")

    y_cat = adata.obs[label_key].astype("category")
    y = y_cat.cat.codes.to_numpy()
    class_names = list(y_cat.cat.categories)

    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)

    return X, y, class_names


# ----------------------------
# LR
# ----------------------------
def train_eval_lr_pca(X_train, y_train, X_test, y_test, seed: int):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    Cs = [0.1, 0.5, 1.0, 2.0, 5.0]
    best = {"C": None, "macro_f1": -1.0, "acc": -1.0}

    t0 = time.time()
    for C in Cs:
        lr = LogisticRegression(
            C=C,
            max_iter=3000,
            solver="lbfgs",
            n_jobs=1,
            random_state=seed,
        )
        lr.fit(Xtr, y_train)
        pred = lr.predict(Xte)
        acc = accuracy_score(y_test, pred)
        mf1 = f1_score(y_test, pred, average="macro")
        if mf1 > best["macro_f1"]:
            best.update({"C": C, "macro_f1": mf1, "acc": acc})
    t1 = time.time()

    return best["acc"], best["macro_f1"], (t1 - t0), f"input=PCA; best_C={best['C']}"


def train_eval_lr_hvg(X_train, y_train, X_test, y_test, seed: int):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    Cs = [0.1, 1.0, 5.0]
    best = {"C": None, "macro_f1": -1.0, "acc": -1.0}

    t0 = time.time()
    for C in Cs:
        lr = LogisticRegression(
            C=C,
            max_iter=3000,
            solver="lbfgs",
            n_jobs=1,
            random_state=seed,
        )
        lr.fit(Xtr, y_train)
        pred = lr.predict(Xte)
        acc = accuracy_score(y_test, pred)
        mf1 = f1_score(y_test, pred, average="macro")
        if mf1 > best["macro_f1"]:
            best.update({"C": C, "macro_f1": mf1, "acc": acc})
    t1 = time.time()

    return best["acc"], best["macro_f1"], (t1 - t0), f"input=HVG; best_C={best['C']}"


# ----------------------------
# XGB
# ----------------------------
def train_eval_xgb_pca(X_train, y_train, X_test, y_test, num_classes: int, seed: int):
    t0 = time.time()

    dtrain = xgb.DMatrix(X_train, label=y_train)
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
        "seed": int(seed),
        "nthread": 1,
    }

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=2000,
        evals=[(dtest, "valid")],
        verbose_eval=False,
        callbacks=[xgb.callback.EarlyStopping(rounds=50, save_best=True)],
    )

    probs = booster.predict(dtest)
    pred = np.argmax(probs, axis=1)

    acc = accuracy_score(y_test, pred)
    mf1 = f1_score(y_test, pred, average="macro")
    t1 = time.time()

    note = f"input=PCA; best_iter={int(booster.best_iteration)}"
    return acc, mf1, (t1 - t0), note


def train_eval_xgb_hvg(X_train, y_train, X_test, y_test, num_classes: int, seed: int):
    t0 = time.time()

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        "objective": "multi:softprob",
        "num_class": int(num_classes),
        "eta": 0.03,
        "max_depth": 6,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "lambda": 1.0,
        "tree_method": "hist",
        "eval_metric": "mlogloss",
        "seed": int(seed),
        "nthread": 1,
    }

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=5000,
        evals=[(dtest, "valid")],
        verbose_eval=False,
        callbacks=[xgb.callback.EarlyStopping(rounds=50, save_best=True)],
    )

    probs = booster.predict(dtest)
    pred = np.argmax(probs, axis=1)

    acc = accuracy_score(y_test, pred)
    mf1 = f1_score(y_test, pred, average="macro")
    t1 = time.time()

    note = f"input=HVG; best_iter={int(booster.best_iteration)}"
    return acc, mf1, (t1 - t0), note


# ----------------------------
# SANN
# ----------------------------
def train_eval_sann_pca(
    X_train, y_train, X_test, y_test, num_classes: int, seed: int,
    hidden_dim=256, dropout=0.1, activation="relu", batchnorm=True,
    epochs=100, batch_size=512, lr=1e-3, patience=15, device="cpu"
):
    set_seed(seed)

    Xtr = torch.from_numpy(X_train).float()
    ytr = torch.from_numpy(y_train).long()
    Xte = torch.from_numpy(X_test).float()

    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)

    model = SANNPCA(
        input_dim=X_train.shape[1],
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        dropout=dropout,
        activation=activation,
        batchnorm=batchnorm,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    best_mf1 = -1.0
    best_state = None
    bad = 0

    Xte_dev = Xte.to(device)

    t0 = time.time()
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            logits = model(Xte_dev)
            pred = torch.argmax(logits, dim=1).cpu().numpy()

        mf1 = f1_score(y_test, pred, average="macro")
        if mf1 > best_mf1 + 1e-6:
            best_mf1 = mf1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        logits = model(Xte_dev)
        pred = torch.argmax(logits, dim=1).cpu().numpy()

    acc = accuracy_score(y_test, pred)
    mf1 = f1_score(y_test, pred, average="macro")
    t1 = time.time()

    note = f"input=PCA; hidden={hidden_dim} dropout={dropout} bn={batchnorm} {activation}"
    return acc, mf1, (t1 - t0), note


def train_eval_sann_hvg(
    X_train, y_train, X_test, y_test, num_classes: int, seed: int,
    hidden_dim=256, dropout=0.1, activation="relu", batchnorm=True,
    epochs=100, batch_size=512, lr=8e-4, patience=15, device="cpu"
):
    set_seed(seed)

    scaler = StandardScaler()
    Xtr_expr = scaler.fit_transform(X_train).astype(np.float32)
    Xte_expr = scaler.transform(X_test).astype(np.float32)

    Xtr_mask = (X_train > 0).astype(np.float32)
    Xte_mask = (X_test > 0).astype(np.float32)

    Xtr_expr_t = torch.from_numpy(Xtr_expr).float()
    Xtr_mask_t = torch.from_numpy(Xtr_mask).float()
    ytr_t = torch.from_numpy(y_train).long()

    Xte_expr_t = torch.from_numpy(Xte_expr).float()
    Xte_mask_t = torch.from_numpy(Xte_mask).float()

    train_loader = DataLoader(
        TensorDataset(Xtr_expr_t, Xtr_mask_t, ytr_t),
        batch_size=batch_size,
        shuffle=True
    )

    model = SANNHVG(
        expr_dim=X_train.shape[1],
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        dropout=dropout,
        activation=activation,
        batchnorm=batchnorm,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    best_mf1 = -1.0
    best_state = None
    bad = 0

    Xte_expr_dev = Xte_expr_t.to(device)
    Xte_mask_dev = Xte_mask_t.to(device)

    t0 = time.time()
    for _ in range(epochs):
        model.train()
        for xb_expr, xb_mask, yb in train_loader:
            xb_expr = xb_expr.to(device)
            xb_mask = xb_mask.to(device)
            yb = yb.to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(xb_expr, xb_mask)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            logits = model(Xte_expr_dev, Xte_mask_dev)
            pred = torch.argmax(logits, dim=1).cpu().numpy()

        mf1 = f1_score(y_test, pred, average="macro")
        if mf1 > best_mf1 + 1e-6:
            best_mf1 = mf1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        logits = model(Xte_expr_dev, Xte_mask_dev)
        pred = torch.argmax(logits, dim=1).cpu().numpy()

    acc = accuracy_score(y_test, pred)
    mf1 = f1_score(y_test, pred, average="macro")
    t1 = time.time()

    note = f"input=HVG+mask; hidden={hidden_dim} dropout={dropout} bn={batchnorm} {activation}"
    return acc, mf1, (t1 - t0), note


# ----------------------------
# Args
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Run robustness across repeated stratified splits for PCA or HVG.")
    p.add_argument("--data", type=str, default="data/processed/pbmc68k_labeled.h5ad")
    p.add_argument("--label_key", type=str, default="cell_type")

    p.add_argument("--rep", type=str, default="pca", choices=["pca", "hvg"])

    # PCA settings
    p.add_argument("--pca_key", type=str, default="X_pca")
    p.add_argument("--pca_dim", type=int, default=50)

    # split settings
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 53, 64, 75, 86])

    # output
    p.add_argument("--out_csv", type=str, default=None)

    # SANN config
    p.add_argument("--sann_hidden", type=int, default=256)
    p.add_argument("--sann_dropout", type=float, default=0.1)
    p.add_argument("--sann_activation", type=str, default="relu", choices=["relu", "gelu"])
    p.add_argument("--sann_batchnorm", action="store_true", default=True)
    p.add_argument("--sann_epochs", type=int, default=100)
    p.add_argument("--sann_patience", type=int, default=15)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()

    if args.out_csv is None:
        if args.rep == "pca":
            args.out_csv = "results/full_train_all_pca/reports/robustness_splits_full.csv"
        else:
            args.out_csv = "results/full_train_all_hvg/reports/robustness_splits_full.csv"

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    if args.rep == "pca":
        X, y, class_names = load_pca_xy(args.data, args.label_key, args.pca_key, args.pca_dim)
    else:
        X, y, class_names = load_hvg_xy(args.data, args.label_key)

    num_classes = len(class_names)

    print(f"[Sanity] rep={args.rep}")
    print(f"[Sanity] X={X.shape} | y={len(y)} | classes={num_classes}")
    print(f"[Sanity] class names: {class_names}")

    if len(args.seeds) < args.n_splits:
        raise ValueError("Need at least n_splits seeds.")

    splits = make_splits(y, args.n_splits, args.test_size, args.seeds[:args.n_splits])

    rows = []
    for sp in splits:
        split_id = sp["split_id"]
        seed = sp["seed"]
        train_idx = np.array(sp["train_idx"], dtype=int)
        test_idx = np.array(sp["test_idx"], dtype=int)

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        print(f"\n=== {split_id} (seed={seed}, rep={args.rep}) ===")
        print(f"Train={len(train_idx)} | Test={len(test_idx)}")

        # LR
        if args.rep == "pca":
            acc, mf1, tsec, note = train_eval_lr_pca(X_train, y_train, X_test, y_test, seed)
        else:
            acc, mf1, tsec, note = train_eval_lr_hvg(X_train, y_train, X_test, y_test, seed)

        rows.append([split_id, seed, "LR", mf1, acc, tsec, note])
        print(f"LR   | MacroF1={mf1:.4f} | Acc={acc:.4f} | time={tsec:.1f}s | {note}")

        # XGB
        if args.rep == "pca":
            acc, mf1, tsec, note = train_eval_xgb_pca(X_train, y_train, X_test, y_test, num_classes, seed)
        else:
            acc, mf1, tsec, note = train_eval_xgb_hvg(X_train, y_train, X_test, y_test, num_classes, seed)

        rows.append([split_id, seed, "XGB", mf1, acc, tsec, note])
        print(f"XGB  | MacroF1={mf1:.4f} | Acc={acc:.4f} | time={tsec:.1f}s | {note}")

        # SANN
        if args.rep == "pca":
            acc, mf1, tsec, note = train_eval_sann_pca(
                X_train, y_train, X_test, y_test, num_classes, seed,
                hidden_dim=args.sann_hidden,
                dropout=args.sann_dropout,
                activation=args.sann_activation,
                batchnorm=args.sann_batchnorm,
                epochs=args.sann_epochs,
                patience=args.sann_patience,
                device=args.device,
            )
        else:
            acc, mf1, tsec, note = train_eval_sann_hvg(
                X_train, y_train, X_test, y_test, num_classes, seed,
                hidden_dim=args.sann_hidden,
                dropout=args.sann_dropout,
                activation=args.sann_activation,
                batchnorm=args.sann_batchnorm,
                epochs=args.sann_epochs,
                patience=args.sann_patience,
                device=args.device,
            )

        rows.append([split_id, seed, "SANN", mf1, acc, tsec, note])
        print(f"SANN | MacroF1={mf1:.4f} | Acc={acc:.4f} | time={tsec:.1f}s | {note}")

    out = pd.DataFrame(rows, columns=[
        "split_id", "seed", "model", "macro_f1", "accuracy", "train_time_seconds", "notes"
    ])
    out.to_csv(args.out_csv, index=False)
    print(f"\n[Saved] {args.out_csv}")

    summary = out.groupby("model")["macro_f1"].agg(["mean", "std", "min", "max", "count"]).reset_index()
    print("\n[Summary] Macro F1 across splits:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()