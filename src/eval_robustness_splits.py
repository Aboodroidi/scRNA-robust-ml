# src/eval_robustness_splits.py
import os
import json
import time
import argparse

# macOS-friendly single-thread settings
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import scanpy as sc

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate robustness across repeated stratified splits.")
    p.add_argument("--data", type=str, default="data/processed/pbmc68k_labeled.h5ad")
    p.add_argument("--label_key", type=str, default="cell_type")

    p.add_argument("--use_pca", action="store_true", default=True)
    p.add_argument("--pca_key", type=str, default="X_pca")
    p.add_argument("--pca_dim", type=int, default=50)

    p.add_argument("--splits", type=str, default="splits/all_splits.json")
    p.add_argument("--out_csv", type=str, default="results/reports/robustness_splits.csv")

    # LR params
    p.add_argument("--lr_max_iter", type=int, default=3000)

    # XGB params (if available)
    p.add_argument("--xgb_estimators", type=int, default=200)
    p.add_argument("--xgb_max_depth", type=int, default=4)
    p.add_argument("--xgb_lr", type=float, default=0.1)

    # SANN params (if torch available)
    p.add_argument("--sann_hidden", type=int, default=256)
    p.add_argument("--sann_dropout", type=float, default=0.1)
    p.add_argument("--sann_activation", type=str, default="relu", choices=["relu", "gelu"])
    p.add_argument("--sann_epochs", type=int, default=15)
    p.add_argument("--sann_batch_size", type=int, default=512)
    p.add_argument("--sann_lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def load_X_y(adata, label_key, use_pca, pca_key, pca_dim):
    y_cat = adata.obs[label_key].astype("category")
    y = y_cat.cat.codes.to_numpy().astype(int)
    class_names = list(y_cat.cat.categories)

    if use_pca:
        if pca_key not in adata.obsm:
            raise ValueError(f"Expected adata.obsm['{pca_key}'] for PCA features.")
        X = np.asarray(adata.obsm[pca_key][:, :pca_dim], dtype=np.float32)
    else:
        X = np.asarray(adata.X, dtype=np.float32)

    return X, y, class_names


def eval_lr(X_train, y_train, X_test, y_test, max_iter):
    t0 = time.time()
    scaler = StandardScaler(with_mean=False)
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    model = LogisticRegression(
        max_iter=max_iter,
        solver="saga",
        multi_class="multinomial",
        n_jobs=1,
        random_state=42
    )
    model.fit(Xtr, y_train)
    pred = model.predict(Xte)

    acc = accuracy_score(y_test, pred)
    mf1 = f1_score(y_test, pred, average="macro")
    return acc, mf1, time.time() - t0


def eval_xgb(X_train, y_train, X_test, y_test, n_classes, params):
    try:
        from xgboost import XGBClassifier
    except Exception as e:
        return None, None, None, f"XGB import failed: {e}"

    t0 = time.time()
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=n_classes,
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=1,
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=42
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)
    mf1 = f1_score(y_test, pred, average="macro")
    return acc, mf1, time.time() - t0, None


def eval_sann(X_train, y_train, X_test, y_test, n_classes, args):
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as e:
        return None, None, None, f"Torch import failed: {e}"

    device = args.device

    class SANN(nn.Module):
        def __init__(self, input_dim, num_classes, hidden, dropout, activation):
            super().__init__()
            if activation == "relu":
                act = nn.ReLU()
            elif activation == "gelu":
                act = nn.GELU()
            else:
                raise ValueError("activation must be relu or gelu")

            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden),
                act,
                nn.Dropout(dropout),
                nn.Linear(hidden, num_classes),
            )

        def forward(self, x):
            return self.net(x)

    # tensors
    Xtr = torch.tensor(X_train, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.long)
    Xte = torch.tensor(X_test, dtype=torch.float32)
    yte = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(Xtr, ytr),
        batch_size=args.sann_batch_size,
        shuffle=True
    )

    model = SANN(
        input_dim=X_train.shape[1],
        num_classes=n_classes,
        hidden=args.sann_hidden,
        dropout=args.sann_dropout,
        activation=args.sann_activation
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.sann_lr)

    t0 = time.time()
    model.train()
    for _ in range(args.sann_epochs):
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        logits = model(Xte.to(device))
        pred = torch.argmax(logits, dim=1).cpu().numpy()

    acc = accuracy_score(y_test, pred)
    mf1 = f1_score(y_test, pred, average="macro")
    return acc, mf1, time.time() - t0, None


def main():
    args = parse_args()
    ensure_dir(args.out_csv)

    adata = sc.read_h5ad(args.data)
    X, y, class_names = load_X_y(
        adata, args.label_key, args.use_pca, args.pca_key, args.pca_dim
    )
    n_classes = len(class_names)

    with open(args.splits, "r") as f:
        splits = json.load(f)

    rows = []

    xgb_params = {
        "n_estimators": args.xgb_estimators,
        "max_depth": args.xgb_max_depth,
        "learning_rate": args.xgb_lr
    }

    print(f"[Sanity] X shape: {X.shape} | y: {len(y)} | classes: {n_classes}")
    print(f"[Sanity] n_splits: {len(splits)} | splits file: {args.splits}")

    for s in splits:
        split_id = s["split_id"]
        seed = s["seed"]
        train_idx = np.array(s["train_idx"], dtype=int)
        test_idx = np.array(s["test_idx"], dtype=int)

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # LR
        acc, mf1, tsec = eval_lr(X_train, y_train, X_test, y_test, max_iter=args.lr_max_iter)
        rows.append({
            "split_id": split_id, "seed": seed, "model": "LR",
            "macro_f1": mf1, "accuracy": acc, "train_time_seconds": tsec
        })
        print(f"[{split_id}] LR macroF1={mf1:.4f} acc={acc:.4f} time={tsec:.1f}s")

        # XGB
        acc, mf1, tsec, err = eval_xgb(X_train, y_train, X_test, y_test, n_classes, xgb_params)
        if err is None:
            rows.append({
                "split_id": split_id, "seed": seed, "model": "XGB",
                "macro_f1": mf1, "accuracy": acc, "train_time_seconds": tsec
            })
            print(f"[{split_id}] XGB macroF1={mf1:.4f} acc={acc:.4f} time={tsec:.1f}s")
        else:
            print(f"[{split_id}] XGB skipped: {err}")

        # SANN
        acc, mf1, tsec, err = eval_sann(X_train, y_train, X_test, y_test, n_classes, args)
        if err is None:
            rows.append({
                "split_id": split_id, "seed": seed, "model": "SANN",
                "macro_f1": mf1, "accuracy": acc, "train_time_seconds": tsec
            })
            print(f"[{split_id}] SANN macroF1={mf1:.4f} acc={acc:.4f} time={tsec:.1f}s")
        else:
            print(f"[{split_id}] SANN skipped: {err}")

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    print(f"[Saved] {args.out_csv}")

    # quick summary print
    if len(df) > 0:
        summary = df.groupby("model")["macro_f1"].agg(["mean", "std", "min", "max", "count"]).reset_index()
        print("\n[Summary] Macro F1 across splits:")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()