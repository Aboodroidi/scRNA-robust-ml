"""
Per-class F1 with 95% bootstrap CIs across donors — SANN_PCA vs scANVI vs Seurat.

Closes the "per-class single-point-estimate" critique by resampling test cells
1,000 times and computing the 2.5 / 97.5 percentile F1 for each class × tool ×
donor. The figure is a 2-panel grouped bar chart (8K, 3K) with error bars.

Outputs:
    results/figures/per_class_f1_ci.png
    results/figures/per_class_f1_ci.csv

Run:
    python src/plot_per_class_f1_ci.py
"""
import os

os.environ["OMP_NUM_THREADS"]    = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"]    = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["MPLBACKEND"]         = "Agg"

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score


# ── Config ────────────────────────────────────────────────────────
FIG_DIR  = "results/figures"
FIG_PATH = os.path.join(FIG_DIR, "per_class_f1_ci.png")
CSV_PATH = os.path.join(FIG_DIR, "per_class_f1_ci.csv")

N_BOOT   = 1000
CI_LOW   = 2.5
CI_HIGH  = 97.5
RNG_SEED = 0

# Class list used for SANN positional-to-name mapping (alphabetical).
SANN_CLASSES = ["B cells", "Mono", "NK", "Platelet", "T cells"]

# Classes to display per donor (drop classes with zero support).
DONOR_CLASSES = {
    "8K": ["B cells", "Mono", "NK", "T cells"],          # no platelets in 8K
    "3K": ["B cells", "Mono", "NK", "Platelet", "T cells"],
}

# Coarse-label mapping (merges CD8 T → T cells, normalises SingleR's own
# reference labels, keeps truly unknown categories as-is so the filter step
# can drop them or treat them as wrong predictions).
COARSE_MAP = {
    "B cells":        "B cells",
    "CD8 T":          "T cells",
    "T cells":        "T cells",
    "Mono":           "Mono",
    "NK":             "NK",
    "Platelet":       "Platelet",
    # SingleR reference-panel labels
    "Monocytes":      "Mono",
    "NK cells":       "NK",
    "CD4+ T cells":   "T cells",
    "CD8+ T cells":   "T cells",
    # "Dendritic cells" and "Progenitors" intentionally left unmapped so they
    # count as incorrect for whichever true class they're predicted against.
}

# Three strongest tools rendered in saturated colours; SingleR + ACTINN added
# as faded bars so a reviewer can still compare all five Table-5 tools.
TOOL_COLORS = {
    "SANN_PCA": "#2E86AB",   # blue           (strong)
    "scANVI":   "#D1495B",   # red            (strong)
    "Seurat":   "#8FBC94",   # green          (strong)
    "ACTINN":   "#C9A875",   # muted gold     (faded)
    "SingleR":  "#9E9E9E",   # grey           (faded)
}
TOOL_ALPHA = {
    "SANN_PCA": 1.00,
    "scANVI":   1.00,
    "Seurat":   1.00,
    "ACTINN":   0.65,
    "SingleR":  0.65,
}
TOOL_ORDER = ["SANN_PCA", "scANVI", "Seurat", "ACTINN", "SingleR"]

DONOR_H5AD = {
    "8K": "data/processed/pbmc8k_labeled.h5ad",
    "3K": "data/processed/pbmc3k_labeled.h5ad",
}

SANN_PRED = {
    "8K": "results/external_validation/pca/sann_pca_ext_pred.npy",
    "3K": "results/external_validation_3k/pca/sann_pca_ext3k_pred.npy",
}

SCANVI_CSV = {
    "8K": "results/comparators/scanvi/scanvi_8k_predictions.csv",
    "3K": "results/comparators/scanvi/scanvi_3k_predictions.csv",
}

SEURAT_CSV = {
    "8K": "results/comparators/seurat/seurat_8k_predictions.csv",
    "3K": "results/comparators/seurat/seurat_3k_predictions.csv",
}

ACTINN_CSV = {
    "8K": "results/comparators/actinn/actinn_8k_predictions.csv",
    "3K": "results/comparators/actinn/actinn_3k_predictions.csv",
}

SINGLER_CSV = {
    "8K": "results/comparators/singleR/singleR_8k_predictions.csv",
    "3K": "results/comparators/singleR/singleR_3k_predictions.csv",
}


# ── Loaders ───────────────────────────────────────────────────────
def load_donor_true(donor: str):
    """Filter donor h5ad to known coarse classes, return (barcodes, y_true)."""
    adata = sc.read_h5ad(DONOR_H5AD[donor])
    ct = adata.obs["cell_type"].astype(str)
    mapped = ct.map(lambda x: COARSE_MAP.get(x, x))
    mask = mapped.isin(SANN_CLASSES)
    bc  = np.asarray(adata.obs.index[mask.values])
    y   = np.asarray(mapped[mask.values].values)
    return bc, y


def load_sann(donor: str, n_cells: int):
    """Load SANN_PCA integer predictions, map to class names by alphabetical order."""
    pred_int = np.load(SANN_PRED[donor])
    assert len(pred_int) == n_cells, (
        f"SANN {donor} prediction length {len(pred_int)} != {n_cells} filtered cells"
    )
    return np.array([SANN_CLASSES[i] for i in pred_int])


def load_comparator_csv(csv_path: str, barcodes: np.ndarray, donor: str):
    """Align a comparator CSV to the canonical barcode order."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().strip('"') for c in df.columns]
    df = df.set_index("barcode")
    df = df.loc[barcodes]  # reindex → same order as canonical
    preds = df["predicted_label"].astype(str).values
    # Normalize CD8 T → T cells (Seurat may emit CD8 T)
    preds = np.array([COARSE_MAP.get(p, p) for p in preds])
    return preds


# ── Bootstrap ─────────────────────────────────────────────────────
def bootstrap_per_class_f1(y_true, y_pred, classes, n_boot=N_BOOT, seed=0):
    """Return dict: class -> (point_estimate, ci_low, ci_high)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    boot_f1 = np.zeros((n_boot, len(classes)), dtype=np.float32)

    # Point estimate on full data
    point_f1 = f1_score(y_true, y_pred, labels=classes,
                        average=None, zero_division=0)

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_f1[b] = f1_score(
            y_true[idx], y_pred[idx],
            labels=classes, average=None, zero_division=0,
        )

    lo = np.percentile(boot_f1, CI_LOW,  axis=0)
    hi = np.percentile(boot_f1, CI_HIGH, axis=0)
    return {c: (float(point_f1[i]), float(lo[i]), float(hi[i]))
            for i, c in enumerate(classes)}


# ── Plot ──────────────────────────────────────────────────────────
def plot_grouped_bars(results: pd.DataFrame, out_path: str):
    """results: long-form df with columns tool, donor, cell_type, f1, ci_lo, ci_hi."""
    fig, axes = plt.subplots(
        1, 2, figsize=(13.5, 5.2),
        gridspec_kw={"width_ratios": [4, 5], "wspace": 0.22},
    )

    for ax, donor in zip(axes, ["8K", "3K"]):
        classes = DONOR_CLASSES[donor]
        n_cls   = len(classes)
        n_tool  = len(TOOL_ORDER)
        x = np.arange(n_cls)
        bar_w = 0.84 / n_tool

        sub = results[results["donor"] == donor]
        # per-class support counts (taken from any tool's rows — all tools share
        # the same filtered test set)
        support = (
            sub.groupby("cell_type")["n_class"].first().reindex(classes)
            .astype(int).to_dict()
        )

        for j, tool in enumerate(TOOL_ORDER):
            tsub = sub[sub["tool"] == tool].set_index("cell_type")
            tsub = tsub.reindex(classes)
            vals = tsub["f1"].values
            lo   = tsub["ci_lo"].values
            hi   = tsub["ci_hi"].values
            err  = np.vstack([vals - lo, hi - vals])
            offset = (j - (n_tool - 1) / 2) * bar_w
            ax.bar(
                x + offset, vals, width=bar_w,
                color=TOOL_COLORS[tool], edgecolor="black",
                linewidth=0.4, label=tool, alpha=TOOL_ALPHA[tool],
            )
            ax.errorbar(
                x + offset, vals, yerr=err,
                fmt="none", ecolor="black", capsize=2, lw=0.7,
                alpha=TOOL_ALPHA[tool],
            )
            # numeric annotation above error bar (vertical so adjacent 0.99s
            # don't collide horizontally)
            for xi, v, h in zip(x + offset, vals, hi):
                ax.text(
                    xi, h + 0.015, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=7,
                    color="black", rotation=90,
                )

        n_cells = int(sub.groupby("tool")["n_test"].first().iloc[0])
        # x-tick labels with n per class on a second line
        xtick_labels = [
            f"{c}\n(n = {support[c]:,})" for c in classes
        ]
        ax.set_xticks(x)
        ax.set_xticklabels(xtick_labels, fontsize=9.5)
        ax.set_ylim(0, 1.22)
        ax.set_ylabel("F1 score (95% bootstrap CI)" if donor == "8K" else "")
        ax.set_title(f"PBMC {donor}   (n = {n_cells:,} test cells)",
                     fontsize=11)
        ax.axhline(1.0, color="#bbbbbb", linewidth=0.5, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", linestyle=":", alpha=0.4)

        if donor == "3K":
            ax.legend(
                fontsize=8.5, frameon=True, framealpha=0.9,
                edgecolor="none", facecolor="white",
                loc="upper right",
                bbox_to_anchor=(1.0, 1.17),
                ncol=1, handlelength=1.6,
            )

    fig.suptitle(
        "Per-class F1 across donors — SANN_PCA vs scANVI vs Seurat "
        f"(bootstrap 95% CI, {N_BOOT:,} resamples)",
        fontsize=12, y=1.02,
    )
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ── Main ──────────────────────────────────────────────────────────
def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    rows = []

    for donor in ["8K", "3K"]:
        print(f"\n── {donor} ──")
        bc, y_true = load_donor_true(donor)
        print(f"  filtered to {len(bc):,} cells")

        preds = {
            "SANN_PCA": load_sann(donor, len(bc)),
            "scANVI":   load_comparator_csv(SCANVI_CSV[donor],  bc, donor),
            "Seurat":   load_comparator_csv(SEURAT_CSV[donor],  bc, donor),
            "ACTINN":   load_comparator_csv(ACTINN_CSV[donor],  bc, donor),
            "SingleR":  load_comparator_csv(SINGLER_CSV[donor], bc, donor),
        }

        classes = DONOR_CLASSES[donor]
        class_support = {c: int((y_true == c).sum()) for c in classes}
        for tool, y_pred in preds.items():
            boot = bootstrap_per_class_f1(
                y_true, y_pred, classes,
                n_boot=N_BOOT, seed=RNG_SEED,
            )
            for cls, (pt, lo, hi) in boot.items():
                rows.append({
                    "donor":     donor,
                    "tool":      tool,
                    "cell_type": cls,
                    "f1":        pt,
                    "ci_lo":     lo,
                    "ci_hi":     hi,
                    "n_test":    len(bc),
                    "n_class":   class_support[cls],
                })
                print(f"  {tool:10s} {cls:10s}  n={class_support[cls]:>4}  "
                      f"F1 = {pt:.3f}  [{lo:.3f}, {hi:.3f}]")

    results = pd.DataFrame(rows)
    results.to_csv(CSV_PATH, index=False)
    print(f"\nSaved {CSV_PATH}")

    plot_grouped_bars(results, FIG_PATH)


if __name__ == "__main__":
    main()
