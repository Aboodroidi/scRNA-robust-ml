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

import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit

import xgboost as xgb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ----------------------------
# SANN MODEL (PCA input only)
# ----------------------------
class SANN(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=256, dropout=0.1, use_batchnorm=True):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim)]
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        if dropout and dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ----------------------------
# Utilities
# ----------------------------
def load_pca_data(data_path, label_key="cell_type", pca_key="X_pca", pca_dim=50):
    adata = sc.read_h5ad(data_path)

    if label_key not in adata.obs:
        raise ValueError(f"Expected adata.obs['{label_key}']")
    if pca_key not in adata.obsm:
        raise ValueError(f"Expected adata.obsm['{pca_key}']")

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


# ----------------------------
# Logistic Regression
# ----------------------------
def train_full_lr(X_train, y_train, X_val, y_val, X_test, y_test, outdir):
    print("\nTraining Logistic Regression (PCA features, tuning C on validation)...")
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
        "Notes": f"input=PCA; best_C={best_C}; solver=lbfgs; grid={c_grid}",
    }


# ----------------------------
# XGBoost
# ----------------------------
def train_full_xgb(X_train, y_train, X_val, y_val, X_test, y_test, num_classes, outdir):
    print("\nTraining XGBoost (PCA features, early stopping on validation)...")
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

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=2000,
        evals=[(dval, "valid")],
        verbose_eval=50,
        callbacks=[xgb.callback.EarlyStopping(rounds=50, save_best=True)],
    )

    train_time = time.time() - t0
    print(f"XGB best_iteration={booster.best_iteration} | time={train_time:.1f}s")

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
        "Notes": f"input=PCA; best_iter={int(booster.best_iteration)}",
    }


# ----------------------------
# SANN
# ----------------------------
def train_full_sann(X_train, y_train, X_val, y_val, X_test, y_test, num_classes, outdir):
    print("\nTraining SANN (PCA features only, no temp scaling)...")
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
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=512, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=512, shuffle=False)

    best_val_f1 = -1.0
    best_state = None
    patience = 15
    patience_counter = 0
    history = []

    for epoch in range(1, 101):
        model.train()
        train_loss_sum = 0.0
        train_n = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
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
                loss = criterion(logits, yb)

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

        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "val_macro_f1": float(val_f1),
        })

        print(
            f"  Epoch {epoch:03d} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_macroF1={val_f1:.4f}"
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

    test_probs = torch.softmax(torch.tensor(test_logits), dim=1).numpy()
    test_pred = test_probs.argmax(axis=1)

    acc = accuracy_score(y_test, test_pred)
    f1 = f1_score(y_test, test_pred, average="macro")
    print(f"SANN raw | Test Acc={acc:.4f} | Test Macro-F1={f1:.4f}")

    torch.save(model.state_dict(), os.path.join(outdir, "sann_model.pt"))
    pd.DataFrame(history).to_csv(os.path.join(outdir, "sann_history.csv"), index=False)

    np.save(os.path.join(outdir, "sann_test_probs.npy"), test_probs)
    np.save(os.path.join(outdir, "sann_test_pred.npy"), test_pred)
    np.save(os.path.join(outdir, "sann_test_true.npy"), y_test)

    return {
        "Model": "SANN",
        "Accuracy": float(acc),
        "Macro-F1": float(f1),
        "TrainTimeSeconds": float(train_time),
        "Notes": "input=PCA; hidden=256; dropout=0.1; bn=True; relu=True; temp_scaling=off",
    }


# ----------------------------
# MAIN
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/pbmc68k_labeled.h5ad")
    parser.add_argument("--splits", default="results/ablations/fixed_splits.json")
    parser.add_argument("--outdir", default="results/full_train_all_pca")
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--pca_dim", type=int, default=50)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    X, y, class_names = load_pca_data(
        args.data,
        label_key="cell_type",
        pca_key="X_pca",
        pca_dim=args.pca_dim,
    )

    train_idx_full, test_idx = load_fixed_split(args.splits)

    X_train_full = X[train_idx_full]
    y_train_full = y[train_idx_full]
    X_test = X[test_idx]
    y_test = y[test_idx]

    tr_rel, val_rel = make_train_val_split(y_train_full, val_frac=args.val_frac, seed=42)

    X_train = X_train_full[tr_rel]
    y_train = y_train_full[tr_rel]
    X_val = X_train_full[val_rel]
    y_val = y_train_full[val_rel]

    num_classes = len(class_names)

    print(f"[Sanity] Total samples = {len(y)} | classes = {num_classes}")
    print(f"[Sanity] PCA shape = {X.shape}")
    print(f"[Sanity] Train_full={len(train_idx_full)} | Train={len(tr_rel)} | Val={len(val_rel)} | Test={len(test_idx)}")
    print(f"[Sanity] class names: {class_names}")

    with open(os.path.join(args.outdir, "train_val_test_split.json"), "w") as f:
        json.dump({
            "train_full_size": int(len(train_idx_full)),
            "train_size": int(len(tr_rel)),
            "val_size": int(len(val_rel)),
            "test_size": int(len(test_idx)),
            "val_frac": float(args.val_frac),
            "pca_dim": int(args.pca_dim),
            "temperature_scaling_used": False,
        }, f, indent=2)

    metrics_path = os.path.join(args.outdir, "baseline_metrics_full.csv")

    lr_row = train_full_lr(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        args.outdir
    )
    save_metrics_row(metrics_path, lr_row)

    xgb_row = train_full_xgb(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        num_classes,
        args.outdir
    )
    save_metrics_row(metrics_path, xgb_row)

    sann_row = train_full_sann(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        num_classes,
        args.outdir
    )
    save_metrics_row(metrics_path, sann_row)

    print("\n✅ ALL PCA MODELS TRAINED CORRECTLY.")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved artifacts under: {args.outdir}/")


if __name__ == "__main__":
    main()