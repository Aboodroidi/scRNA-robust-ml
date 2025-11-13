# src/calibrate.py
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scanpy as sc
from scipy.sparse import issparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from multiprocessing import Process

# ------------- utils -------------
def to_numpy(arr):
    return arr.toarray() if issparse(arr) else np.asarray(arr)

def ece_score(y_true, prob, n_bins=15):
    y_true = np.asarray(y_true)
    conf = prob.max(axis=1)
    pred = prob.argmax(axis=1)
    acc = (pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(conf, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.any():
            ece += abs(acc[mask].mean() - conf[mask].mean()) * mask.mean()
    return float(ece)

def reliability_plot(y_true, prob, out_path, n_bins=15, title="Reliability Diagram"):
    y_true = np.asarray(y_true)
    conf = prob.max(axis=1)
    pred = prob.argmax(axis=1)
    acc = (pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(conf, bins) - 1

    bin_conf, bin_acc = [], []
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.any():
            bin_conf.append(conf[mask].mean()); bin_acc.append(acc[mask].mean())
        else:
            bin_conf.append(np.nan); bin_acc.append(np.nan)

    fig = plt.figure(figsize=(5, 5))
    xs = np.linspace(0, 1, 100); plt.plot(xs, xs, linestyle="--")
    valid = ~np.isnan(bin_conf)
    plt.bar(np.array(bin_conf)[valid], np.array(bin_acc)[valid], width=1.0/n_bins, align="center", alpha=0.6)
    plt.xlim(0, 1); plt.ylim(0, 1)
    plt.xlabel("Confidence"); plt.ylabel("Accuracy"); plt.title(title)
    plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close(fig)

# ------------- SANN (match train_sann.py) -------------
class SANN(nn.Module):
    def __init__(self, input_dim, num_classes, hidden=256, dropout=0.3, l1_lambda=1e-5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.act1 = nn.GELU(); self.do1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.act2 = nn.GELU(); self.do2 = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, num_classes)
    def forward(self, x):
        x = self.do1(self.act1(self.bn1(self.fc1(x))))
        x = self.do2(self.act2(self.bn2(self.fc2(x))))
        return self.head(x)

# ------------- child process for XGB -------------
def _child_predict_xgb(pkl_path, json_path, X_path, out_path):
    # Set env vars BEFORE importing xgboost to reduce OpenMP issues on macOS
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    import numpy as _np
    import xgboost as _xgb
    from xgboost import XGBClassifier as _XGBClassifier
    import joblib as _joblib

    X = _np.load(X_path)

    prob = None
    try:
        if pkl_path and os.path.exists(pkl_path):
            clf: _XGBClassifier = _joblib.load(pkl_path)
            # Prefer Booster path (safer)
            booster = clf.get_booster()
            booster.set_param({"nthread": 1, "predictor": "cpu_predictor"})
            dm = _xgb.DMatrix(X)
            prob = booster.predict(dm)
        elif json_path and os.path.exists(json_path):
            booster = _xgb.Booster()
            booster.load_model(json_path)
            booster.set_param({"nthread": 1, "predictor": "cpu_predictor"})
            dm = _xgb.DMatrix(X)
            prob = booster.predict(dm)
    except Exception:
        prob = None

    if prob is not None:
        _np.save(out_path, prob)
    # If it crashes, parent will just not find out_path and skip elegantly.

# ------------- main -------------
def main():
    DATA = "data/processed/pbmc68k_labeled.h5ad"
    LABEL_COL = "cell_type"
    OUT = "results/calibration"
    os.makedirs(OUT, exist_ok=True)

    bar_fmt = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{percentage:3.0f}%] - {elapsed}<{remaining}"
    steps = ["Load & split","Scale","LR calibration","XGB calibration","SANN calibration","Save summary"]
    pbar = tqdm(total=len(steps), bar_format=bar_fmt, desc="Calibration")

    # 1) Load & split
    pbar.set_description(f"1/6 {steps[0]}")
    adata = sc.read_h5ad(DATA)
    X = adata.X
    y_cat = adata.obs[LABEL_COL].astype("category")
    y = y_cat.cat.codes
    label_names = list(y_cat.cat.categories)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    pbar.update(1)

    # 2) Scale
    pbar.set_description(f"2/6 {steps[1]}")
    scaler = StandardScaler(with_mean=not issparse(X_train))
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    X_test_np = to_numpy(X_test)
    y_test_np = np.asarray(y_test)
    pbar.update(1)

    records = []

    # 3) Logistic Regression
    pbar.set_description(f"3/6 {steps[2]}")
    lr_path = "results/logreg_model.pkl"
    if os.path.exists(lr_path):
        try:
            lr: LogisticRegression = joblib.load(lr_path)
            prob_lr = lr.predict_proba(X_test_np)
            ece_lr = ece_score(y_test_np, prob_lr)
            reliability_plot(y_test_np, prob_lr, os.path.join(OUT, "reliability_lr.png"),
                             title="Reliability: Logistic Regression")
            records.append(("Logistic Regression", ece_lr))
            pbar.set_postfix_str(f"LR ECE {ece_lr:.3f}")
        except Exception as e:
            pbar.set_postfix_str(f"LR error: {e}")
    else:
        pbar.set_postfix_str("LR model not found")
    pbar.update(1)

    # 4) XGB (run in child process; skip gracefully if it crashes)
    pbar.set_description(f"4/6 {steps[3]}")
    xgb_pkl = "results/xgb_model.pkl"
    xgb_json = "results/xgb_model.json"  # present if you saved it
    tmp_X = os.path.join(OUT, "_X_test.npy")
    tmp_prob = os.path.join(OUT, "_xgb_prob.npy")
    try:
        np.save(tmp_X, X_test_np)
        proc = Process(target=_child_predict_xgb, args=(xgb_pkl, xgb_json, tmp_X, tmp_prob))
        proc.start()
        proc.join(timeout=300)  # 5 minutes cap
        if proc.is_alive():
            proc.terminate()
        if os.path.exists(tmp_prob):
            prob_xgb = np.load(tmp_prob)
            ece_xgb = ece_score(y_test_np, prob_xgb)
            reliability_plot(y_test_np, prob_xgb, os.path.join(OUT, "reliability_xgb.png"),
                             title="Reliability: XGBoost")
            records.append(("XGBoost", ece_xgb))
            pbar.set_postfix_str(f"XGB ECE {ece_xgb:.3f}")
        else:
            pbar.set_postfix_str("XGB skipped (child segfault or timeout)")
    except Exception as e:
        pbar.set_postfix_str(f"XGB skipped: {e}")
    finally:
        for path in (tmp_X, tmp_prob):
            if os.path.exists(path):
                try: os.remove(path)
                except Exception: pass
    pbar.update(1)

    # 5) SANN
    pbar.set_description(f"5/6 {steps[4]}")
    sann_dir = "results/sann"
    model_path = os.path.join(sann_dir, "sann_final.pt")
    cfg_path = os.path.join(sann_dir, "sann_config.txt")
    if not os.path.exists(model_path):
        raise FileNotFoundError("SANN weights not found at results/sann/sann_final.pt")

    hidden, dropout = 256, 0.3
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            for line in f:
                if line.startswith("hidden="):   hidden = int(line.split("=")[1].strip())
                if line.startswith("dropout="):  dropout = float(line.split("=")[1].strip())

    input_dim = X_test_np.shape[1]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SANN(input_dim=input_dim, num_classes=len(label_names), hidden=hidden, dropout=dropout)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    with torch.no_grad():
        logits = []
        B = 2048
        for i in range(0, X_test_np.shape[0], B):
            xb = torch.tensor(X_test_np[i:i+B], dtype=torch.float32, device=device)
            logits.append(model(xb).cpu().numpy())
        logits = np.concatenate(logits, axis=0)
        prob_sann = F.softmax(torch.tensor(logits), dim=1).numpy()

    ece_sann = ece_score(y_test_np, prob_sann)
    reliability_plot(y_test_np, prob_sann, os.path.join(OUT, "reliability_sann.png"),
                     title="Reliability: SANN (Neural Network)")
    records.append(("SANN", ece_sann))
    pbar.set_postfix_str(f"SANN ECE {ece_sann:.3f}")
    pbar.update(1)

    # 6) Save summary
    pbar.set_description(f"6/6 {steps[5]}")
    df = pd.DataFrame(records, columns=["Model", "ECE"])
    df.to_csv(os.path.join(OUT, "metrics.csv"), index=False)
    pbar.update(1); pbar.close()

    print(f"\nSaved: {os.path.join(OUT, 'metrics.csv')}")
    print("Reliability plots saved in results/calibration/")

if __name__ == "__main__":
    # use "spawn" start method for safer child isolation on macOS
    try:
        import multiprocessing as _mp
        _mp.set_start_method("spawn", force=True)
    except Exception:
        pass
    main()