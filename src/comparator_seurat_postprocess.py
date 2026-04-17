"""
Post-process Seurat label transfer predictions: compute accuracy, macro-F1,
weighted-F1, per-class F1, and confusion matrices.

Seurat already produces labels in the 5-class coarse taxonomy (since the
reference was built with coarse labels), so no label mapping needed — only
ground-truth mapping for 8K/3K.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


OUTDIR = Path("results/comparators/seurat")
COARSE_CLASSES = ["B cells", "Mono", "NK", "Platelet", "T cells"]

# Ground-truth external → coarse (same as SingleR)
GT_LABEL_MAP = {
    "B cells":  "B cells",
    "Mono":     "Mono",
    "CD8 T":    "T cells",
    "T cells":  "T cells",
    "NK":       "NK",
    "Platelet": "Platelet",
    "CL 1":     None,
    "DC":       None,
}


def map_gt(lbl):
    return GT_LABEL_MAP.get(str(lbl), None)


def evaluate(ds_name, processed_path, pred_path):
    print(f"\n── Evaluating Seurat on {ds_name} ──")
    adata = sc.read_h5ad(processed_path)
    gt_raw = adata.obs["cell_type"].astype(str).values
    barcodes = adata.obs_names.values

    pred_df = pd.read_csv(pred_path).set_index("barcode")
    shared = [b for b in barcodes if b in pred_df.index]
    pred_df = pred_df.loc[shared]
    gt_raw_matched = np.array([gt_raw[list(barcodes).index(b)] for b in shared])

    gt_coarse = np.array([map_gt(v) for v in gt_raw_matched], dtype=object)
    keep = np.array([v is not None for v in gt_coarse])
    n_total = len(gt_coarse)
    n_known = keep.sum()

    pred_coarse = pred_df["predicted_label"].astype(str).values

    y_true = gt_coarse[keep].astype(str)
    y_pred = pred_coarse[keep].astype(str)

    # Any Seurat predictions outside the 5-class taxonomy land in 'Other'
    y_pred_clean = np.array([
        p if p in COARSE_CLASSES else "Other" for p in y_pred
    ])
    n_other = int((y_pred_clean == "Other").sum())

    acc = accuracy_score(y_true, y_pred_clean)
    f1_macro = f1_score(y_true, y_pred_clean, labels=COARSE_CLASSES,
                         average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred_clean, labels=COARSE_CLASSES,
                            average="weighted", zero_division=0)
    f1_per_class = f1_score(y_true, y_pred_clean, labels=COARSE_CLASSES,
                             average=None, zero_division=0)

    cm_labels = COARSE_CLASSES + ["Other"]
    cm = confusion_matrix(y_true, y_pred_clean, labels=cm_labels)
    cm_df = pd.DataFrame(cm, index=cm_labels, columns=cm_labels)
    cm_df.to_csv(OUTDIR / f"seurat_{ds_name.lower()}_confusion.csv")

    print(f"   n_total={n_total}  n_known={n_known}  n_excluded={n_total - n_known}")
    print(f"   accuracy    = {acc:.4f}")
    print(f"   macro-F1    = {f1_macro:.4f}")
    print(f"   weighted-F1 = {f1_weighted:.4f}")
    print("   per-class F1:")
    for c, v in zip(COARSE_CLASSES, f1_per_class):
        print(f"     {c:>10s}: {v:.4f}")
    print(f"   predictions outside taxonomy: {n_other}")

    return {
        "n_total": int(n_total),
        "n_known": int(n_known),
        "n_excluded": int(n_total - n_known),
        "n_other_predictions": n_other,
        "accuracy": float(acc),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_weighted),
        "per_class_f1": dict(zip(COARSE_CLASSES, [float(x) for x in f1_per_class])),
    }


def main():
    metrics = {"tool": "Seurat_label_transfer"}
    metrics["8K"] = evaluate(
        "8K",
        "data/processed/pbmc8k_labeled.h5ad",
        OUTDIR / "seurat_8k_predictions.csv",
    )
    metrics["3K"] = evaluate(
        "3K",
        "data/processed/pbmc3k_labeled.h5ad",
        OUTDIR / "seurat_3k_predictions.csv",
    )
    run_info_path = OUTDIR / "seurat_run_info.csv"
    if run_info_path.exists():
        metrics["run_info"] = pd.read_csv(run_info_path).to_dict(orient="records")

    with open(OUTDIR / "seurat_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✅ Metrics saved to {OUTDIR / 'seurat_metrics.json'}")


if __name__ == "__main__":
    main()
