# src/eval_robustness_splits_full.py
import os
import json
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

from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# XGBoost
from xgboost import XGBClassifier
import xgboost as xgb

# PyTorch for SANN
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ----------------------------
# SANN Model
# ----------------------------
class SANN(nn.Module):
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


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_xy(adata_path: str, label_key: str, pca_key: str, pca_dim: int):
    adata = sc.read_h5ad(adata_path)
    if label_key not in adata.obs:
        raise ValueError(f"Missing label column adata.obs['{label_key}']")

    y_cat = adata.obs[label_key].astype("category")
    y = y_cat.cat.codes.to_numpy()
    class_names = list(y_cat.cat.categories)

    if pca_key not in adata.obsm:
        raise ValueError(f"Missing PCA matrix adata.obsm['{pca_key}'] (needed for PCA-{pca_dim})")

    X = np.asarray(adata.obsm[pca_key][:, :pca_dim], dtype=np.float32)
    return X, y, class_names


def make_splits(y, n_splits: int, test_size: float, seeds):
    splits = []
    for i in range(n_splits):
        seed = seeds[i]
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(sss.split(np.zeros_like(y), y))
        splits.append({"split_id": f"split_{i+1:02d}", "seed": seed,
                       "train_idx": train_idx.tolist(), "test_idx": test_idx.tolist()})
    return splits


def train_eval_lr(X_train, y_train, X_test, y_test, seed: int):
    # scale on train only (PCA is already “nice”, but we keep it consistent)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    Cs = [0.1, 0.5, 1, 2, 5]
    best = {"C": None, "macro_f1": -1, "acc": -1, "model": None}

    t0 = time.time()
    for C in Cs:
        lr = LogisticRegression(
            max_iter=6000,
            solver="saga",
            n_jobs=1,
            random_state=seed,
        )
        lr.set_params(C=C)
        lr.fit(Xtr, y_train)
        pred = lr.predict(Xte)
        acc = accuracy_score(y_test, pred)
        mf1 = f1_score(y_test, pred, average="macro")
        if mf1 > best["macro_f1"]:
            best.update({"C": C, "macro_f1": mf1, "acc": acc, "model": lr})
    t1 = time.time()

    return best["acc"], best["macro_f1"], (t1 - t0), f"best_C={best['C']}"


def train_eval_xgb(X_train, y_train, X_test, y_test, num_classes: int, seed: int):
    # Big model + early stopping using callback (works across xgboost versions)
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        max_depth=6,
        learning_rate=0.03,
        n_estimators=5000,        # large, rely on early stopping
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=1,
        tree_method="hist",
        eval_metric="mlogloss",
    )

    t0 = time.time()
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
        callbacks=[xgb.callback.EarlyStopping(rounds=50, save_best=True)]
    )
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    mf1 = f1_score(y_test, pred, average="macro")
    t1 = time.time()

    best_iter = getattr(model, "best_iteration", None)
    note = f"best_iter={best_iter}" if best_iter is not None else "best_iter=NA"
    return acc, mf1, (t1 - t0), note


def train_eval_sann(X_train, y_train, X_test, y_test, num_classes: int, seed: int,
                    hidden_dim=256, dropout=0.1, activation="relu", batchnorm=True,
                    epochs=100, batch_size=512, lr=1e-3, patience=15, device="cpu"):
    set_seed(seed)

    Xtr = torch.from_numpy(X_train).float()
    ytr = torch.from_numpy(y_train).long()
    Xte = torch.from_numpy(X_test).float()
    yte = torch.from_numpy(y_test).long()

    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(Xte, yte), batch_size=batch_size, shuffle=False)

    model = SANN(
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

    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

        # eval on test (you can swap to val if you want, but for robustness this is OK)
        model.eval()
        all_pred = []
        all_true = []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                logits = model(xb)
                pred = torch.argmax(logits, dim=1).cpu().numpy()
                all_pred.append(pred)
                all_true.append(yb.numpy())
        ypred = np.concatenate(all_pred)
        ytrue = np.concatenate(all_true)

        mf1 = f1_score(ytrue, ypred, average="macro")
        if mf1 > best_mf1 + 1e-6:
            best_mf1 = mf1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # final metrics
    model.eval()
    with torch.no_grad():
        logits = model(Xte.to(device))
        ypred = torch.argmax(logits, dim=1).cpu().numpy()

    acc = accuracy_score(y_test, ypred)
    mf1 = f1_score(y_test, ypred, average="macro")
    t1 = time.time()

    note = f"hidden={hidden_dim} dropout={dropout} bn={batchnorm} {activation}"
    return acc, mf1, (t1 - t0), note


def parse_args():
    p = argparse.ArgumentParser(description="Run robustness across repeated stratified splits (FULL training configs).")
    p.add_argument("--data", type=str, default="data/processed/pbmc68k_labeled.h5ad")
    p.add_argument("--label_key", type=str, default="cell_type")
    p.add_argument("--pca_key", type=str, default="X_pca")
    p.add_argument("--pca_dim", type=int, default=50)

    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 53, 64, 75, 86])

    p.add_argument("--out_csv", type=str, default="results/full_train/reports/robustness_splits_full.csv")

    # SANN config (match your final)
    p.add_argument("--sann_hidden", type=int, default=256)
    p.add_argument("--sann_dropout", type=float, default=0.1)
    p.add_argument("--sann_activation", type=str, default="relu", choices=["relu", "gelu"])
    p.add_argument("--sann_batchnorm", action="store_true", default=True)
    p.add_argument("--sann_epochs", type=int, default=100)
    p.add_argument("--sann_patience", type=int, default=15)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    X, y, class_names = load_xy(args.data, args.label_key, args.pca_key, args.pca_dim)
    num_classes = len(class_names)

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

        print(f"\n=== {split_id} (seed={seed}) ===")
        print(f"Train={len(train_idx)} | Test={len(test_idx)}")

        # LR
        acc, mf1, tsec, note = train_eval_lr(X_train, y_train, X_test, y_test, seed)
        rows.append([split_id, seed, "LR", mf1, acc, tsec, note])
        print(f"LR  | MacroF1={mf1:.4f} | Acc={acc:.4f} | time={tsec:.1f}s | {note}")

        # XGB
        acc, mf1, tsec, note = train_eval_xgb(X_train, y_train, X_test, y_test, num_classes, seed)
        rows.append([split_id, seed, "XGB", mf1, acc, tsec, note])
        print(f"XGB | MacroF1={mf1:.4f} | Acc={acc:.4f} | time={tsec:.1f}s | {note}")

        # SANN
        acc, mf1, tsec, note = train_eval_sann(
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
        print(f"SANN| MacroF1={mf1:.4f} | Acc={acc:.4f} | time={tsec:.1f}s | {note}")

    out = pd.DataFrame(rows, columns=[
        "split_id", "seed", "model", "macro_f1", "accuracy", "train_time_seconds", "notes"
    ])
    out.to_csv(args.out_csv, index=False)
    print(f"\n[Saved] {args.out_csv}")


if __name__ == "__main__":
    main()