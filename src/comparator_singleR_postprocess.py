"""
Post-process SingleR predictions: map Monaco Immune labels to the 5-class
coarse taxonomy and compute accuracy, macro-F1, weighted-F1, per-class F1,
plus confusion matrices.

Inputs:
    results/comparators/singleR/singleR_8k_predictions.csv
    results/comparators/singleR/singleR_3k_predictions.csv

Outputs:
    results/comparators/singleR/singleR_8k_confusion.csv
    results/comparators/singleR/singleR_3k_confusion.csv
    results/comparators/singleR/singleR_metrics.json
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


# ══════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════
OUTDIR = Path("results/comparators/singleR")
COARSE_CLASSES = ["B cells", "Mono", "NK", "Platelet", "T cells"]

# Monaco Immune → our 5-class coarse taxonomy
# Monaco label.main values: "B cells", "T cells", "CD4+ T cells", "CD8+ T cells",
#   "NK cells", "Monocytes", "Dendritic cells", "Basophils", "Progenitors",
#   "Neutrophils"
MONACO_TO_COARSE = {
    "B cells":            "B cells",
    "T cells":            "T cells",
    "CD4+ T cells":       "T cells",
    "CD8+ T cells":       "T cells",
    "NK cells":           "NK",
    "Monocytes":          "Mono",
    # Classes NOT in our taxonomy → map to None (predicted wrong by default)
    "Dendritic cells":    "Other",
    "Basophils":          "Other",
    "Neutrophils":        "Other",
    "Progenitors":        "Other",
}

# Ground-truth external labels → coarse
GT_LABEL_MAP = {
    "B cells":   "B cells",
    "Mono":      "Mono",
    "CD8 T":     "T cells",
    "T cells":   "T cells",
    "NK":        "NK",
    "Platelet":  "Platelet",
    "CL 1":      None,
    "DC":        None,
}


def map_singleR_label(lbl):
    """Map a Monaco prediction to one of the 5 coarse classes (or 'Other')."""
    if lbl is None or pd.isna(lbl) or str(lbl) == "nan":
        return "Other"
    # SingleR's pruned labels may contain NA for low-confidence predictions;
    # treat those as 'Other' so they count as errors
    return MONACO_TO_COARSE.get(str(lbl), "Other")


def map_gt(lbl):
    v = GT_LABEL_MAP.get(str(lbl), None)
    return v  # None = exclude from evaluation


def evaluate_dataset(ds_name, processed_path, pred_path):
    print(f"\n── Evaluating SingleR on {ds_name} ──")
    # Load processed h5ad to get ground-truth labels
    adata = sc.read_h5ad(processed_path)
    gt_col = "cell_type"
    gt_raw = adata.obs[gt_col].astype(str).values
    barcodes = adata.obs_names.values

    # Load SingleR predictions
    pred_df = pd.read_csv(pred_path)
    pred_df = pred_df.set_index("barcode")

    # Match SingleR predictions to processed h5ad cells
    shared = [b for b in barcodes if b in pred_df.index]
    pred_df = pred_df.loc[shared]
    gt_raw_matched = np.array([gt_raw[list(barcodes).index(b)] for b in shared])

    # Map ground-truth to coarse, keep only evaluable cells
    gt_coarse = np.array([map_gt(v) for v in gt_raw_matched], dtype=object)
    keep = np.array([v is not None for v in gt_coarse])
    n_total = len(gt_coarse)
    n_known = keep.sum()

    # Predicted labels — use pruned_label if available, else predicted_label
    pred_raw = pred_df["pruned_label"].fillna(pred_df["predicted_label"]).values
    pred_coarse = np.array([map_singleR_label(v) for v in pred_raw], dtype=object)

    # Evaluate only known-label cells
    y_true = gt_coarse[keep].astype(str)
    y_pred = pred_coarse[keep].astype(str)

    # Compute metrics (counts "Other" predictions as misclassifications)
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, labels=COARSE_CLASSES,
                         average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, labels=COARSE_CLASSES,
                            average="weighted", zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, labels=COARSE_CLASSES,
                             average=None, zero_division=0)

    # Confusion matrix — include 'Other' column so we can see how many
    # predictions landed outside the taxonomy
    cm_labels = COARSE_CLASSES + ["Other"]
    cm = confusion_matrix(y_true, y_pred, labels=cm_labels)
    cm_df = pd.DataFrame(cm, index=cm_labels, columns=cm_labels)
    cm_df.to_csv(OUTDIR / f"singleR_{ds_name.lower()}_confusion.csv")

    print(f"   n_total={n_total}  n_known={n_known}  n_excluded={n_total - n_known}")
    print(f"   accuracy    = {acc:.4f}")
    print(f"   macro-F1    = {f1_macro:.4f}")
    print(f"   weighted-F1 = {f1_weighted:.4f}")
    print("   per-class F1:")
    for c, v in zip(COARSE_CLASSES, f1_per_class):
        print(f"     {c:>10s}: {v:.4f}")
    print(f"   predictions landing outside taxonomy: {(pred_coarse[keep] == 'Other').sum()}")

    return {
        "n_total": int(n_total),
        "n_known": int(n_known),
        "n_excluded": int(n_total - n_known),
        "n_other_predictions": int((pred_coarse[keep] == "Other").sum()),
        "accuracy": float(acc),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_weighted),
        "per_class_f1": dict(zip(COARSE_CLASSES, [float(x) for x in f1_per_class])),
    }


def main():
    metrics = {}

    metrics["8K"] = evaluate_dataset(
        "8K",
        "data/processed/pbmc8k_labeled.h5ad",
        OUTDIR / "singleR_8k_predictions.csv",
    )
    metrics["3K"] = evaluate_dataset(
        "3K",
        "data/processed/pbmc3k_labeled.h5ad",
        OUTDIR / "singleR_3k_predictions.csv",
    )

    # Runtime info from R
    run_info_path = OUTDIR / "singleR_run_info.csv"
    if run_info_path.exists():
        metrics["run_info"] = pd.read_csv(run_info_path).to_dict(orient="records")

    metrics["tool"] = "SingleR"
    metrics["reference"] = "MonacoImmuneData"
    metrics["label_mapping_notes"] = {
        "source": "Monaco Immune label.main",
        "target": "5-class coarse taxonomy",
        "mapping": MONACO_TO_COARSE,
        "unmappable_classes": ["Dendritic cells", "Basophils", "Neutrophils", "Progenitors"],
        "note": "Predictions from unmappable classes count as 'Other' and are misclassifications.",
    }

    with open(OUTDIR / "singleR_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✅ Metrics saved to {OUTDIR / 'singleR_metrics.json'}")


if __name__ == "__main__":
    main()
