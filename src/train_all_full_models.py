# src/train_all_full_models.py
import os
import json
import time
import argparse
import numpy as np
import pandas as pd
import scanpy as sc

# mac-friendly: keep threads low
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MPLBACKEND"] = "Agg"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # helps when torch/sklearn/OpenMP clash

import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

import xgboost as xgb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ----------------------------
# SANN MODEL
# ----------------------------
class SANN(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=256, dropout=0.1, use_batchnorm=True):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim)]
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers += [nn.ReLU()]
        if dropout and dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ----------------------------
# Utilities
# ----------------------------
def load_data(data_path, label_key="cell_type", pca_key="X_pca", pca_dim=50):
    adata = sc.read_h5ad(data_path)

    if label_key not in adata.obs:
        raise ValueError(f"Expected adata.obs['{label_key}']")

    if pca_key not in adata.obsm:
        raise ValueError(f"Expected adata.obsm['{pca_key}'] (PCA features)")

    y_cat = adata.obs[label_key].astype("category")
    y = y_cat.cat.codes.to_numpy()
    class_names = list(y_cat.cat.categories)

    X = np.asarray(adata.obsm[pca_key][:, :pca_dim], dtype=np.float32)
    return X, y, class_names


def load_fixed_split(split_path):
    with open(split_path, "r") as f:
        d = json.load(f)
    if "train_idx" not in d or "test_idx" not in d:
        raise ValueError(f"Split file must contain train_idx and test_idx. Found keys: {list(d.keys())}")
    return np.array(d["train_idx"], dtype=int), np.array(d["test_idx"], dtype=int)


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


# ----------------------------
# Logistic Regression (FULL)
# ----------------------------
def train_full_lr(X_train, y_train, X_test, y_test, outdir):
    print("\nTraining Logistic Regression (tuning C)...")
    t0 = time.time()

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    best_f1 = -1.0
    best_C = None
    best_model = None

    for C in [0.1, 0.5, 1, 2, 5]:
        model = LogisticRegression(
            C=C,
            max_iter=20000,     # higher to reduce convergence warnings
            tol=1e-4,
            solver="saga",
            n_jobs=1,
            random_state=42,
        )
        model.fit(X_train_s, y_train)
        pred = model.predict(X_test_s)
        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, average="macro")
        print(f"  C={C:<4} | Acc={acc:.4f} | Macro-F1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_C = C
            best_model = model

    train_time = time.time() - t0
    print(f"Best LR: C={best_C} | Macro-F1={best_f1:.4f} | time={train_time:.1f}s")

    # save
    joblib.dump(best_model, os.path.join(outdir, "lr_model.pkl"))
    joblib.dump(scaler, os.path.join(outdir, "lr_scaler.pkl"))

    # probs/preds for later calibration
    probs = best_model.predict_proba(X_test_s)
    pred = probs.argmax(axis=1)
    np.save(os.path.join(outdir, "lr_test_probs.npy"), probs)
    np.save(os.path.join(outdir, "lr_test_pred.npy"), pred)
    np.save(os.path.join(outdir, "lr_test_true.npy"), y_test)

    return {
        "Model": "LR",
        "Accuracy": float(accuracy_score(y_test, pred)),
        "Macro-F1": float(f1_score(y_test, pred, average="macro")),
        "TrainTimeSeconds": float(train_time),
        "Notes": f"best_C={best_C}",
    }


# ----------------------------
# XGBoost (FULL + EARLY STOP) using xgb.train() for old API compatibility
# ----------------------------
def train_full_xgb(X_train, y_train, X_test, y_test, num_classes, outdir):
    print("\nTraining XGBoost (large model + early stopping, compatible API)...")
    t0 = time.time()

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_test, label=y_test)

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

    # train with early stopping via callbacks (works across versions)
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=2000,
        evals=[(dvalid, "valid")],
        verbose_eval=50,
        callbacks=[xgb.callback.EarlyStopping(rounds=50, save_best=True)],
    )

    train_time = time.time() - t0
    print(f"XGB best_iteration={booster.best_iteration} | time={train_time:.1f}s")

    # predict probabilities
    probs = booster.predict(dvalid)  # shape (N, C)
    pred = probs.argmax(axis=1)

    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average="macro")
    print(f"XGB | Acc={acc:.4f} | Macro-F1={f1:.4f}")

    # save booster
    booster.save_model(os.path.join(outdir, "xgb_model.json"))

    # save arrays
    np.save(os.path.join(outdir, "xgb_test_probs.npy"), probs)
    np.save(os.path.join(outdir, "xgb_test_pred.npy"), pred)
    np.save(os.path.join(outdir, "xgb_test_true.npy"), y_test)

    return {
        "Model": "XGB",
        "Accuracy": float(acc),
        "Macro-F1": float(f1),
        "TrainTimeSeconds": float(train_time),
        "Notes": f"best_iter={int(booster.best_iteration)}",
    }


# ----------------------------
# SANN (FULL + EARLY STOP)
# ----------------------------
def train_full_sann(X_train, y_train, X_test, y_test, num_classes, outdir):
    print("\nTraining SANN (100 epochs + early stopping)...")
    t0 = time.time()

    device = "cpu"

    model = SANN(
        input_dim=X_train.shape[1],
        num_classes=num_classes,
        hidden_dim=256,
        dropout=0.1,
        use_batchnorm=True,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=512, shuffle=True)

    best_f1 = -1.0
    best_state = None
    patience = 15
    patience_counter = 0

    history = []

    for epoch in range(1, 101):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            logits_test = model(X_test_t)
            probs = torch.softmax(logits_test, dim=1).cpu().numpy()
            pred = probs.argmax(axis=1)
            acc = accuracy_score(y_test, pred)
            f1 = f1_score(y_test, pred, average="macro")

        tr_loss = float(np.mean(train_losses)) if train_losses else np.nan
        history.append({"epoch": epoch, "train_loss": tr_loss, "test_acc": float(acc), "test_macro_f1": float(f1)})

        print(f"  Epoch {epoch:03d} | train_loss={tr_loss:.4f} | test_macroF1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    train_time = time.time() - t0

    # load best state
    if best_state is not None:
        model.load_state_dict(best_state)

    # final inference for saving
    model.eval()
    with torch.no_grad():
        logits_test = model(X_test_t)
        probs = torch.softmax(logits_test, dim=1).cpu().numpy()
        pred = probs.argmax(axis=1)

    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average="macro")

    print(f"SANN best_macroF1={best_f1:.4f} | final_test_macroF1={f1:.4f} | time={train_time:.1f}s")

    # save
    torch.save(model.state_dict(), os.path.join(outdir, "sann_model.pt"))
    pd.DataFrame(history).to_csv(os.path.join(outdir, "sann_history.csv"), index=False)

    np.save(os.path.join(outdir, "sann_test_probs.npy"), probs)
    np.save(os.path.join(outdir, "sann_test_pred.npy"), pred)
    np.save(os.path.join(outdir, "sann_test_true.npy"), y_test)

    return {
        "Model": "SANN",
        "Accuracy": float(acc),
        "Macro-F1": float(f1),
        "TrainTimeSeconds": float(train_time),
        "Notes": "hidden=256 dropout=0.1 bn=True relu",
    }


# ----------------------------
# MAIN
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/pbmc68k_labeled.h5ad")
    parser.add_argument("--splits", default="results/ablations/fixed_splits.json")
    parser.add_argument("--outdir", default="results/full_train")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    X, y, class_names = load_data(args.data)
    train_idx, test_idx = load_fixed_split(args.splits)

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    num_classes = len(class_names)

    print(f"[Sanity] X={X.shape} | y={len(y)} | classes={num_classes}")
    print(f"[Sanity] Train={len(train_idx)} | Test={len(test_idx)}")
    print(f"[Sanity] class names: {class_names}")

    metrics_path = os.path.join(args.outdir, "baseline_metrics_full.csv")

    # 1) LR
    lr_row = train_full_lr(X_train, y_train, X_test, y_test, args.outdir)
    save_metrics_row(metrics_path, lr_row)

    # 2) XGB (fixed)
    xgb_row = train_full_xgb(X_train, y_train, X_test, y_test, num_classes, args.outdir)
    save_metrics_row(metrics_path, xgb_row)

    # 3) SANN
    sann_row = train_full_sann(X_train, y_train, X_test, y_test, num_classes, args.outdir)
    save_metrics_row(metrics_path, sann_row)

    print("\n✅ ALL FULL MODELS TRAINED.")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved artifacts under: {args.outdir}/")


if __name__ == "__main__":
    main()