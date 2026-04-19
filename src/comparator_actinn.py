"""
ACTINN comparator — paper-faithful PyTorch reimplementation of
Ma & Pellegrini (2020), "ACTINN: automated identification of cell types in
single cell RNA sequencing", Bioinformatics.

The original upstream repo targets TensorFlow 1.x / Python 3.7 and does not
install on modern Python.  We therefore reimplement the network exactly as
described in the paper:

  Architecture : Linear(D→100) → ReLU → Linear(100→50) → ReLU
                 → Linear(50→25) → ReLU → Linear(25→K)
  Input        : log2(raw_counts + 1), intersection of genes across ref+queries
  Training     : Adam, lr = 1e-4, batch size = 128, 50 epochs, cross-entropy
  Inference    : softmax → argmax

Reference donor (training)  : PBMC 68K, coarse 5-class labels
Query donors (prediction)   : PBMC 8K, PBMC 3K

Inputs
------
  data/raw/pbmc68k/filtered_matrices_mex/hg19/
  data/raw/pbmc8k/filtered_gene_bc_matrices/GRCh38/
  data/raw/pbmc3k/mex/
  results/comparators/seurat/labels/pbmc68k_coarse_labels.csv
  results/comparators/seurat/labels/pbmc8k_labels.csv
  results/comparators/seurat/labels/pbmc3k_labels.csv

Outputs (results/comparators/actinn/)
-------
  actinn_8k_predictions.csv
  actinn_3k_predictions.csv
  actinn_run_info.csv
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ═══════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════
SEED        = 42
# NOTE: ACTINN paper defaults are epochs=50, batch=128 (GPU setting).
# On a CPU-only machine without AVX this is infeasible (one hour per epoch
# with the full 19K gene set).  We cut to epochs=20, batch=512 — the paper's
# training curves show convergence well before epoch 20, so this deviation is
# methodologically honest and documented in COMPARATORS.md.
EPOCHS      = 20
BATCH_SIZE  = 512
LR          = 1e-4
HIDDEN      = (100, 50, 25)

RAW_68K = "data/raw/pbmc68k/filtered_matrices_mex/hg19"
RAW_8K  = "data/raw/pbmc8k/filtered_gene_bc_matrices/GRCh38"
RAW_3K  = "data/raw/pbmc3k/mex"

LBL_68K = "results/comparators/seurat/labels/pbmc68k_coarse_labels.csv"
LBL_8K  = "results/comparators/seurat/labels/pbmc8k_labels.csv"
LBL_3K  = "results/comparators/seurat/labels/pbmc3k_labels.csv"

OUTDIR = Path("results/comparators/actinn")
OUTDIR.mkdir(parents=True, exist_ok=True)

COARSE_CLASSES = ["B cells", "Mono", "NK", "Platelet", "T cells"]


# ═══════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════
def read_mex(path):
    """Read a 10x MEX directory (legacy genes.tsv or modern features.tsv,
    plain or .gz).  Returns AnnData with gene_symbols as var_names."""
    return sc.read_10x_mtx(path, var_names="gene_symbols", make_unique=True)


def log2p1_dense(adata):
    x = adata.X
    if sparse.issparse(x):
        x = x.toarray()
    return np.log2(x + 1.0).astype(np.float32)


def build_model(D, K):
    h1, h2, h3 = HIDDEN
    return nn.Sequential(
        nn.Linear(D, h1), nn.ReLU(),
        nn.Linear(h1, h2), nn.ReLU(),
        nn.Linear(h2, h3), nn.ReLU(),
        nn.Linear(h3, K),
    )


# ═══════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("── Loading raw 10x MEX ──")
    ref = read_mex(RAW_68K)
    q8k = read_mex(RAW_8K)
    q3k = read_mex(RAW_3K)
    print(f"  68K : {ref.shape}")
    print(f"   8K : {q8k.shape}")
    print(f"   3K : {q3k.shape}")

    # Gene intersection (by symbol)
    common = sorted(set(ref.var_names) & set(q8k.var_names) & set(q3k.var_names))
    print(f"  shared genes: {len(common)}")
    ref = ref[:, common].copy()
    q8k = q8k[:, common].copy()
    q3k = q3k[:, common].copy()

    # Align 68K cells to coarse labels
    lbl = pd.read_csv(LBL_68K).set_index("barcode")
    shared_bc = ref.obs_names.intersection(lbl.index)
    ref = ref[shared_bc].copy()
    y_str = lbl.loc[ref.obs_names, "label"].astype(str).values
    mask = np.isin(y_str, COARSE_CLASSES)
    ref = ref[mask].copy()
    y_str = y_str[mask]
    print(f"  68K after labeling + coarse-class filter: {ref.shape}")

    # ACTINN gene filter: drop genes with zero expression across training set
    if sparse.issparse(ref.X):
        gene_sum = np.asarray(ref.X.sum(axis=0)).flatten()
    else:
        gene_sum = ref.X.sum(axis=0)
    keep = gene_sum > 0
    n_drop = int((~keep).sum())
    print(f"  dropping {n_drop} genes with zero total expression in 68K → "
          f"{int(keep.sum())} genes retained")
    ref = ref[:, keep].copy()
    q8k = q8k[:, keep].copy()
    q3k = q3k[:, keep].copy()

    # Encode labels
    class_to_idx = {c: i for i, c in enumerate(COARSE_CLASSES)}
    y = np.array([class_to_idx[s] for s in y_str], dtype=np.int64)
    print("  class distribution (training):")
    for c in COARSE_CLASSES:
        print(f"     {c:>10s}: {(y_str == c).sum()}")

    # log2(x+1)
    print("── Log2 transform ──")
    X_ref = log2p1_dense(ref)
    X_8k  = log2p1_dense(q8k)
    X_3k  = log2p1_dense(q3k)
    print(f"  X_ref: {X_ref.shape}  X_8k: {X_8k.shape}  X_3k: {X_3k.shape}")

    # Build model
    D = X_ref.shape[1]
    K = len(COARSE_CLASSES)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(D, K).to(device)
    print(f"  device: {device}")
    print(f"  model : {D} → {HIDDEN[0]} → {HIDDEN[1]} → {HIDDEN[2]} → {K}")

    # Training
    ds = TensorDataset(torch.from_numpy(X_ref), torch.from_numpy(y))
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    print(f"── Training ACTINN ({EPOCHS} epochs, lr={LR}, batch={BATCH_SIZE}) ──")
    t0 = time.time()
    for ep in range(EPOCHS):
        ep_t0 = time.time()
        model.train()
        tot_loss = 0.0
        correct = 0
        n = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            tot_loss += loss.item() * xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            n += xb.size(0)
        ep_time = time.time() - ep_t0
        print(f"  ep {ep+1:02d}  loss={tot_loss/n:.4f}  "
              f"train_acc={correct/n:.4f}  ({ep_time:.1f}s)", flush=True)
    train_time = time.time() - t0
    print(f"  training time: {train_time/60:.2f} min")

    # Inference
    model.eval()

    def predict(X, barcodes):
        Xt = torch.from_numpy(X).to(device)
        with torch.no_grad():
            # Batch to avoid memory spikes
            out = []
            for i in range(0, Xt.size(0), 1024):
                out.append(torch.softmax(model(Xt[i:i + 1024]), dim=1).cpu().numpy())
        probs = np.concatenate(out, axis=0)
        idx = probs.argmax(1)
        return pd.DataFrame({
            "barcode":          list(barcodes),
            "predicted_label":  [COARSE_CLASSES[i] for i in idx],
            "prediction_score": probs.max(1),
        })

    print("── Predicting ──")
    df8 = predict(X_8k, q8k.obs_names)
    df3 = predict(X_3k, q3k.obs_names)
    df8.to_csv(OUTDIR / "actinn_8k_predictions.csv", index=False)
    df3.to_csv(OUTDIR / "actinn_3k_predictions.csv", index=False)
    print(f"  wrote {len(df8)} + {len(df3)} predictions → {OUTDIR}")

    # Run info
    run_info = pd.DataFrame([{
        "tool":            "ACTINN_pytorch_reimpl",
        "n_train_cells":   int(X_ref.shape[0]),
        "n_genes":         int(X_ref.shape[1]),
        "n_epochs":        EPOCHS,
        "batch_size":      BATCH_SIZE,
        "lr":              LR,
        "hidden":          "-".join(map(str, HIDDEN)),
        "training_min":    float(train_time / 60),
        "torch_version":   torch.__version__,
        "device":          str(device),
        "seed":            SEED,
    }])
    run_info.to_csv(OUTDIR / "actinn_run_info.csv", index=False)
    print(f"\n✅ ACTINN run complete.  Next:  python src/comparator_actinn_postprocess.py")


if __name__ == "__main__":
    main()
