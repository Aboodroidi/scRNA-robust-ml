"""
Visualise the learned gate α values of the SANN fusion layer.

INPUTS (you produce these with a short inference snippet that loads the
trained SANN checkpoint, registers a forward hook on `self.gate`, runs one
pass over the 68K test fold, and saves):

    results/gate_analysis/alpha.npy               # shape (n_test, 256), float in [0,1]
    results/gate_analysis/cell_type_per_cell.npy  # shape (n_test,), string/object

OUTPUTS:

    results/figures/gate_alpha_histogram.png          (always)
    results/figures/gate_alpha_per_celltype.png       (only if Outcome B)
    results/figures/gate_alpha_summary.csv

The histogram asks one question: is the gate actually gating? Interpretation
is printed to stdout and annotated on the figure:

    Outcome A  — mass collapses near 0.5          → learnable fusion, not selection
    Outcome B  — spread across [0,1] or bimodal   → per-dim selection, design validated
    Outcome C  — peaked near 0 or near 1          → one stream dominates (flag tension)

Run:
    python src/plot_gate_alpha_distribution.py
"""
import os

os.environ["OMP_NUM_THREADS"]     = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"]     = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["MPLBACKEND"]          = "Agg"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Paths ────────────────────────────────────────────────────────
IN_DIR        = "results/gate_analysis"
ALPHA_PATH    = os.path.join(IN_DIR, "alpha.npy")
CELLTYPE_PATH = os.path.join(IN_DIR, "cell_type_per_cell.npy")

FIG_DIR       = "results/figures"
HIST_PATH     = os.path.join(FIG_DIR, "gate_alpha_histogram.png")
BOX_PATH      = os.path.join(FIG_DIR, "gate_alpha_per_celltype.png")
SUMMARY_CSV   = os.path.join(FIG_DIR, "gate_alpha_summary.csv")

# Outcome-classification thresholds
COLLAPSED_BAND = (0.40, 0.60)    # "Outcome A" band around 0.5
DECISIVE_LO    = 0.10            # values ≤ 0.10 are "decisively mask"
DECISIVE_HI    = 0.90            # values ≥ 0.90 are "decisively expression"

COLLAPSED_FRAC_THRESHOLD = 0.60  # >60% in [0.4, 0.6] → Outcome A
DECISIVE_FRAC_THRESHOLD  = 0.25  # >25% at the edges → qualifies as Outcome B spread
SIDEDNESS_MEAN_LO        = 0.25  # mean < 0.25 → C (mask-dominated)
SIDEDNESS_MEAN_HI        = 0.75  # mean > 0.75 → C (expression-dominated)

COARSE_ORDER = ["B cells", "Mono", "NK", "Platelet", "T cells"]


def classify_outcome(alpha_flat: np.ndarray) -> tuple[str, str]:
    """Return (code, human_description)."""
    mean    = float(alpha_flat.mean())
    std     = float(alpha_flat.std())
    collapsed = float(((alpha_flat >= COLLAPSED_BAND[0]) &
                       (alpha_flat <= COLLAPSED_BAND[1])).mean())
    decisive  = float(((alpha_flat <= DECISIVE_LO) |
                       (alpha_flat >= DECISIVE_HI)).mean())

    if collapsed > COLLAPSED_FRAC_THRESHOLD:
        return ("A", "collapsed near 0.5 — gate acts as learnable fusion, "
                     "not per-dimension selection")

    if decisive > DECISIVE_FRAC_THRESHOLD or std > 0.22:
        return ("B", "spread / bimodal — gate makes per-dimension decisions "
                     "about which stream to trust")

    if mean < SIDEDNESS_MEAN_LO:
        return ("C", "peaked near 0 — mask stream dominates")
    if mean > SIDEDNESS_MEAN_HI:
        return ("C", "peaked near 1 — expression stream dominates")

    return ("B", "mild spread, no sharp mode — gate active but uncommitted")


def plot_histogram(alpha_flat: np.ndarray, stats: dict, outcome_code: str,
                   outcome_desc: str, out_path: str):
    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    ax.hist(
        alpha_flat, bins=50, range=(0.0, 1.0),
        color="#2E86AB", edgecolor="black", linewidth=0.3, alpha=0.9,
    )
    ax.axvline(0.5, color="#C44E52", linestyle="--", lw=1.2,
               label="α = 0.5 (equal fusion)")
    ax.axvline(stats["mean"], color="black", linestyle=":", lw=1.2,
               label=f"mean α = {stats['mean']:.3f}")

    # Shade "collapsed" band for reference
    ax.axvspan(COLLAPSED_BAND[0], COLLAPSED_BAND[1],
               color="#C44E52", alpha=0.08,
               label=f"collapsed band [{COLLAPSED_BAND[0]:.1f}, "
                     f"{COLLAPSED_BAND[1]:.1f}]")

    # Stats box
    textstr = "\n".join([
        f"n values   = {stats['n']:,}",
        f"mean       = {stats['mean']:.3f}",
        f"std        = {stats['std']:.3f}",
        f"median     = {stats['median']:.3f}",
        f"% in [0.4, 0.6]    = {stats['collapsed_frac']*100:5.1f}",
        f"% ≤ 0.1 or ≥ 0.9  = {stats['decisive_frac']*100:5.1f}",
        "",
        f"Outcome {outcome_code}: {outcome_desc}",
    ])
    ax.text(
        0.98, 0.97, textstr, transform=ax.transAxes,
        fontsize=9, va="top", ha="right", family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#cccccc", alpha=0.95),
    )

    ax.set_xlabel("Learned gate value α  (0 = mask stream, 1 = expression stream)",
                  fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("Distribution of learned gate α values "
                 "— all 68K test cells × 256 fusion dimensions",
                 fontsize=12)
    ax.set_xlim(0, 1)
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


def plot_per_celltype_boxplot(alpha: np.ndarray, cell_type: np.ndarray,
                              out_path: str):
    """One box per coarse cell type, y = mean α per cell (mean across 256 dims)."""
    mean_alpha_per_cell = alpha.mean(axis=1)  # (n_cells,)
    cats = [c for c in COARSE_ORDER if (cell_type == c).any()]

    data = [mean_alpha_per_cell[cell_type == c] for c in cats]

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    bp = ax.boxplot(
        data, labels=cats, patch_artist=True,
        medianprops=dict(color="black", linewidth=1.2),
        flierprops=dict(marker="o", markersize=2, alpha=0.3,
                        markerfacecolor="#888", markeredgecolor="none"),
        widths=0.55,
    )
    palette = ["#2E86AB", "#D1495B", "#55A868", "#DD8452", "#8172B3"]
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        patch.set_edgecolor("black")

    ax.axhline(0.5, color="#888", linestyle="--", lw=0.9, alpha=0.7)
    ax.set_ylabel("Mean α per cell (averaged over 256 fusion dims)",
                  fontsize=11)
    ax.set_xlabel("Cell type (coarse)", fontsize=11)
    ax.set_title("Per-cell-type mean gate α — does the gate specialise?",
                 fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    if not os.path.exists(ALPHA_PATH):
        raise FileNotFoundError(
            f"{ALPHA_PATH} not found. Generate it first by running your "
            "inference snippet that hooks self.gate and saves alpha.npy."
        )
    alpha = np.load(ALPHA_PATH)
    if alpha.ndim != 2:
        raise ValueError(f"alpha.npy must be 2D (n_cells × 256), got {alpha.shape}")
    print(f"[load] alpha shape = {alpha.shape}")

    alpha_flat = alpha.ravel()
    stats = {
        "n":              int(alpha_flat.size),
        "n_cells":        int(alpha.shape[0]),
        "n_gate_dims":    int(alpha.shape[1]),
        "mean":           float(alpha_flat.mean()),
        "std":            float(alpha_flat.std()),
        "median":         float(np.median(alpha_flat)),
        "q25":            float(np.quantile(alpha_flat, 0.25)),
        "q75":            float(np.quantile(alpha_flat, 0.75)),
        "collapsed_frac": float(((alpha_flat >= COLLAPSED_BAND[0]) &
                                 (alpha_flat <= COLLAPSED_BAND[1])).mean()),
        "decisive_frac":  float(((alpha_flat <= DECISIVE_LO) |
                                 (alpha_flat >= DECISIVE_HI)).mean()),
    }
    outcome_code, outcome_desc = classify_outcome(alpha_flat)
    stats["outcome_code"] = outcome_code
    stats["outcome_desc"] = outcome_desc

    print(f"[stats] mean={stats['mean']:.3f}  std={stats['std']:.3f}  "
          f"median={stats['median']:.3f}")
    print(f"[stats] % in [0.4, 0.6]   = {stats['collapsed_frac']*100:.1f}")
    print(f"[stats] % ≤ 0.1 or ≥ 0.9 = {stats['decisive_frac']*100:.1f}")
    print(f"[outcome] {outcome_code} — {outcome_desc}")

    pd.DataFrame([stats]).to_csv(SUMMARY_CSV, index=False)
    print(f"[saved] {SUMMARY_CSV}")

    # Always draw the histogram
    plot_histogram(alpha_flat, stats, outcome_code, outcome_desc, HIST_PATH)

    # Conditional second plot — only if Outcome B (gate is actually gating)
    if outcome_code == "B":
        if not os.path.exists(CELLTYPE_PATH):
            print(f"[skip] {CELLTYPE_PATH} not found — cannot draw per-cell-type "
                  "boxplot. Save cell_type_per_cell.npy alongside alpha.npy if "
                  "you want this plot.")
            return
        cell_type = np.load(CELLTYPE_PATH, allow_pickle=True).astype(str)
        if len(cell_type) != alpha.shape[0]:
            raise ValueError(
                f"cell_type length {len(cell_type)} != alpha rows {alpha.shape[0]}"
            )
        plot_per_celltype_boxplot(alpha, cell_type, BOX_PATH)
    else:
        print(f"[skip] Outcome {outcome_code} — per-cell-type boxplot only "
              "added for Outcome B (gate demonstrably active).")


if __name__ == "__main__":
    main()
