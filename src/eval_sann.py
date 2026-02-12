# src/eval_sann.py
import os
import json
import argparse

# macOS-friendly + avoid OpenMP crashes
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MPLBACKEND"] = "Agg"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)


# ----------------------------
# SANN Model (must match training architecture)
# ----------------------------
class SANN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        activation: str = "relu",
        use_batchnorm: bool = False,
    ):
        super().__init__()
        act = self._make_activation(activation)

        layers = [nn.Linear(input_dim, hidden_dim)]
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(act)
        if dropout and dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.net = nn.Sequential(*layers)

    @staticmethod
    def _make_activation(name: str) -> nn.Module:
        name = name.lower()
        if name == "relu":
            return nn.ReLU()
        if name == "gelu":
            return nn.GELU()
        raise ValueError(f"Unsupported activation: {name}")

    def forward(self, x):
        return self.net(x)


# ----------------------------
# Helpers
# ----------------------------
def shorten_labels(labels, max_len=16):
    out = []
    for s in labels:
        s = str(s)
        out.append(s if len(s) <= max_len else s[: max_len - 1] + "…")
    return out


def load_fixed_splits(split_path: str):
    with open(split_path, "r") as f:
        d = json.load(f)
    # support either list or numpy-like
    splits = {k: np.array(v, dtype=int) for k, v in d.items()}
    required = {"test_idx"}
    if not required.issubset(set(splits.keys())):
        raise ValueError(f"Split file must contain at least {required}. Found keys: {list(splits.keys())}")
    return splits


@torch.no_grad()
def infer(model: nn.Module, X_test: np.ndarray, batch_size: int, device: str):
    model.eval()

    ds = TensorDataset(torch.from_numpy(X_test))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    all_logits = []
    for (xb,) in dl:
        xb = xb.to(device)
        logits = model(xb)
        all_logits.append(logits.detach().cpu().numpy())

    logits = np.vstack(all_logits)
    probs = softmax_np(logits)
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    return logits, probs, pred, conf


def softmax_np(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def plot_confusion(cm_norm: np.ndarray, class_names, out_path: str, title: str):
    fig, ax = plt.subplots(figsize=(9, 8))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_norm,
        display_labels=shorten_labels(class_names),
    )
    disp.plot(
        include_values=False,
        cmap="Blues",
        ax=ax,
        colorbar=True,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def update_metrics_csv(model_name: str, acc: float, macro_f1: float, path: str):
    row = pd.DataFrame([{
        "Model": model_name,
        "Accuracy": round(float(acc), 4),
        "Macro-F1": round(float(macro_f1), 4),
    }])

    if os.path.exists(path):
        df = pd.read_csv(path)
        df = df[df["Model"] != model_name]
        df = pd.concat([df, row], ignore_index=True)
    else:
        df = row

    df.to_csv(path, index=False)


# ----------------------------
# Main
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained SANN on the fixed test split.")
    p.add_argument("--data", type=str, default="data/processed/pbmc68k_labeled.h5ad")
    p.add_argument("--label_key", type=str, default="cell_type")

    # representation
    p.add_argument("--use_pca", action="store_true", default=True)
    p.add_argument("--pca_key", type=str, default="X_pca")
    p.add_argument("--pca_dim", type=int, default=50)

    # splits
    p.add_argument("--splits", type=str, default="results/ablations/fixed_splits.json")

    # model checkpoint
    p.add_argument("--ckpt", type=str, required=True, help="Path to trained SANN checkpoint (.pt) state_dict")
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--activation", type=str, default="relu", choices=["relu", "gelu"])
    p.add_argument("--batchnorm", action="store_true", default=False)

    # eval
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--outdir", type=str, default="results")
    return p.parse_args()


def main():
    args = parse_args()

    outdir = args.outdir
    fig_dir = os.path.join(outdir, "figures")
    rep_dir = os.path.join(outdir, "reports")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(rep_dir, exist_ok=True)

    # ----------------------------
    # Step 1–2: Load data, build X/y using PCA
    # ----------------------------
    adata = sc.read_h5ad(args.data)

    if args.label_key not in adata.obs:
        raise ValueError(f"Expected adata.obs['{args.label_key}'] to exist.")

    y_cat = adata.obs[args.label_key].astype("category")
    y = y_cat.cat.codes.to_numpy()
    class_names = list(y_cat.cat.categories)

    if args.use_pca:
        if args.pca_key not in adata.obsm:
            raise ValueError(f"Expected adata.obsm['{args.pca_key}'] to exist for PCA features.")
        X = adata.obsm[args.pca_key]
        if args.pca_dim is not None:
            X = X[:, :args.pca_dim]
        X = np.asarray(X, dtype=np.float32)
    else:
        # not recommended unless your SANN was trained on adata.X
        X = np.asarray(adata.X, dtype=np.float32)

    print(f"[Sanity] X shape: {X.shape}")
    print(f"[Sanity] y length: {len(y)}")
    print(f"[Sanity] #classes: {len(class_names)}")
    print(f"[Sanity] class names: {class_names}")

    # ----------------------------
    # Step 3: Apply fixed split (preferred) else fallback split
    # ----------------------------
    if args.splits and os.path.exists(args.splits):
        splits = load_fixed_splits(args.splits)
        test_idx = splits["test_idx"]
        print(f"[Sanity] Using fixed split file: {args.splits}")
        if "train_idx" in splits:
            print(f"[Sanity] Train samples: {len(splits['train_idx'])} | Test samples: {len(test_idx)}")
        else:
            print(f"[Sanity] Test samples: {len(test_idx)}")
    else:
        # fallback: same definition as LR/XGB scripts
        _, test_idx = train_test_split(
            np.arange(len(y)),
            test_size=0.2,
            stratify=y,
            random_state=42,
        )
        print("[Sanity] fixed_splits.json not found; using fallback stratified split (seed=42).")
        print(f"[Sanity] Test samples: {len(test_idx)}")

    X_test = X[test_idx]
    y_test = y[test_idx]

    # ----------------------------
    # Step 4: Load model + inference
    # ----------------------------
    device = args.device
    num_classes = len(class_names)

    model = SANN(
        input_dim=X_test.shape[1],
        num_classes=num_classes,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        activation=args.activation,
        use_batchnorm=args.batchnorm,
    ).to(device)

    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)

    logits, probs, y_pred, conf = infer(model, X_test, batch_size=args.batch_size, device=device)

    # Save raw outputs (for calibration/error analysis)
    np.save(os.path.join(outdir, "sann_test_probs.npy"), probs)
    np.save(os.path.join(outdir, "sann_test_pred.npy"), y_pred)
    np.save(os.path.join(outdir, "sann_test_true.npy"), y_test)

    # optional: cell IDs if present
    if adata.obs_names is not None:
        cell_ids = np.array(adata.obs_names)[test_idx]
        with open(os.path.join(outdir, "sann_test_cell_ids.txt"), "w") as f:
            for cid in cell_ids:
                f.write(str(cid) + "\n")

    # ----------------------------
    # Step 5: Metrics + per-class report
    # ----------------------------
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    print(f"[Sanity] Accuracy (SANN): {acc:.4f}")
    print(f"[Sanity] Macro-F1 (SANN): {macro_f1:.4f}")

    # Per-class report (CSV)
    report_dict = classification_report(
        y_test, y_pred, target_names=class_names, output_dict=True
    )
    report_df = pd.DataFrame(report_dict).transpose()
    report_path = os.path.join(rep_dir, "sann_classification_report.csv")
    report_df.to_csv(report_path, index=True)

    # Append/update baseline_metrics.csv
    metrics_path = os.path.join(outdir, "baseline_metrics.csv")
    update_metrics_csv("SANN", acc, macro_f1, metrics_path)

    # ----------------------------
    # Step 6: Confusion matrix (row-normalised)
    # ----------------------------
    cm_raw = confusion_matrix(y_test, y_pred, labels=np.arange(num_classes))
    cm_norm = cm_raw.astype(np.float64)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_norm, row_sums, out=np.zeros_like(cm_norm), where=row_sums != 0)

    fig_path = os.path.join(fig_dir, "confusion_matrix_sann.png")
    plot_confusion(cm_norm, class_names, fig_path, "Confusion Matrix (Row-normalised) — SANN")

    # ----------------------------
    # Step 7: Sanity prints (paths)
    # ----------------------------
    print(f"[Sanity] Saved confusion matrix: {fig_path}")
    print(f"[Sanity] Saved per-class report: {report_path}")
    print(f"[Sanity] Saved probs/preds/true to: {outdir}/sann_test_probs.npy, sann_test_pred.npy, sann_test_true.npy")
    print(f"[Sanity] Updated metrics file: {metrics_path}")


if __name__ == "__main__":
    main()