"""
Plot QC distribution diagnostics for all three donors (68K, 8K, 3K).

Purpose
───────
Justify the QC thresholds used in `src/preprocess*.py` by showing the
underlying empirical distributions rather than asserting the cutoffs.

Thresholds drawn:
    - n_genes_per_cell  ≥ 200          (all donors)
    - n_genes_per_cell  ≤ 5000         (3K only)
    - pct_counts_mt     < 10% (68K/8K) / < 5% (3K)
    - min_cells         ≥ 3            (gene-level filter)

Figure layout (one PNG):
    3 rows × 3 cols  — each row is a donor, columns are:
        (1) histogram of n_genes_per_cell (log-x)
        (2) histogram of pct_counts_mt
        (3) scatter total_counts vs n_genes, coloured by pct_mt (log-log)

Plus a separate small figure:
    bar chart of "cells retained vs dropped" per donor.

Outputs:
    results/figures/qc_distributions.png
    results/figures/qc_retention_bars.png
"""
import os

# Silence macOS thread warnings before heavy imports
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import scipy.sparse as sp
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


OUT_DIR = "results/figures"
FIG_PATH = os.path.join(OUT_DIR, "qc_distributions.png")
BAR_PATH = os.path.join(OUT_DIR, "qc_retention_bars.png")


# ───── Donor configuration ──────────────────────────────────────────
# Each entry: (name, loader_kind, path, min_genes, max_genes, mt_pct)
DONORS = [
    ("PBMC 68K", "mex",  "data/raw/pbmc68k/filtered_matrices_mex/hg19", 200, None, 10.0),
    ("PBMC 8K",  "mex",  "data/raw/pbmc8k/filtered_gene_bc_matrices/GRCh38", 200, None, 10.0),
    ("PBMC 3K",  "h5ad", "data/raw/pbmc3k/pbmc3k_raw.h5ad",              200, 5000, 5.0),
]

# Dissertation-level total-counts cut described in §2.2.1 (shown for
# text-figure consistency even though the code doesn't enforce it —
# 10x filtered_* matrices rarely retain cells above this anyway).
MAX_TOTAL_COUNTS = 20_000

# Shared colour scale for the scatter colourbar (% mt)
MT_CMAP_VMAX = 15.0

BLUE   = "#4C72B0"
ORANGE = "#DD8452"
RED    = "#C44E52"
GREY   = "#7F7F7F"


def load_raw(kind, path):
    if kind == "mex":
        adata = sc.read_10x_mtx(path, var_names="gene_symbols", make_unique=True)
    else:
        adata = sc.read_h5ad(path)
        adata.var_names_make_unique()
    return adata


def compute_qc(adata):
    """Return (n_genes_per_cell, total_counts, pct_mt, cells_per_gene)."""
    X = adata.X
    if sp.issparse(X):
        n_genes = np.array((X > 0).sum(axis=1)).flatten()
        total  = np.array(X.sum(axis=1)).flatten()
        cpg    = np.array((X > 0).sum(axis=0)).flatten()
    else:
        n_genes = (X > 0).sum(axis=1)
        total  = X.sum(axis=1)
        cpg    = (X > 0).sum(axis=0)

    mt_mask = np.asarray(adata.var_names.str.upper().str.startswith("MT-"))
    if sp.issparse(X):
        mt_counts = np.array(X[:, mt_mask].sum(axis=1)).flatten()
    else:
        mt_counts = X[:, mt_mask].sum(axis=1)
    pct_mt = mt_counts / (total + 1e-8) * 100.0
    return n_genes, total, pct_mt, cpg


def retained_mask(n_genes, pct_mt, min_g, max_g, mt_pct):
    m = (n_genes >= min_g) & (pct_mt < mt_pct)
    if max_g is not None:
        m = m & (n_genes <= max_g)
    return m


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ───── Load & compute ────────────────────────────────────────────
    stats = []
    for name, kind, path, min_g, max_g, mt_pct in DONORS:
        print(f"Loading {name} from {path}...")
        adata = load_raw(kind, path)
        n_genes, total, pct_mt, cpg = compute_qc(adata)
        keep = retained_mask(n_genes, pct_mt, min_g, max_g, mt_pct)
        gene_keep = cpg >= 3
        stats.append({
            "name": name, "adata": adata,
            "n_genes": n_genes, "total": total, "pct_mt": pct_mt, "cpg": cpg,
            "min_g": min_g, "max_g": max_g, "mt_pct": mt_pct,
            "n_cells_before": int(adata.shape[0]),
            "n_cells_after":  int(keep.sum()),
            "n_genes_before": int(adata.shape[1]),
            "n_genes_after":  int(gene_keep.sum()),
        })
        print(f"  raw:   {adata.shape[0]:>6d} cells × {adata.shape[1]:>6d} genes")
        print(f"  keep:  {keep.sum():>6d} cells ({100*keep.mean():.1f}%)")
        print(f"  genes: {gene_keep.sum():>6d} genes pass min_cells≥3 "
              f"({100*gene_keep.mean():.1f}%)")

    # ───── Main 3×3 figure ───────────────────────────────────────────
    fig, axes = plt.subplots(3, 3, figsize=(14, 11),
                              gridspec_kw={"wspace": 0.32, "hspace": 0.45})

    # One shared colourbar for the scatter column
    scatter_handles = []

    for row, s in enumerate(stats):
        # Col 1 — n_genes_per_cell histogram (log-x)
        ax = axes[row, 0]
        bins = np.logspace(np.log10(max(1, s["n_genes"].min())),
                           np.log10(max(10, s["n_genes"].max())), 60)
        ax.hist(s["n_genes"], bins=bins, color=BLUE, alpha=0.85,
                edgecolor="white", linewidth=0.3)
        ax.axvline(s["min_g"], color=RED, linestyle="--", linewidth=1.5,
                   label=f"min n_genes = {s['min_g']}")
        if s["max_g"] is not None:
            ax.axvline(s["max_g"], color=RED, linestyle=":", linewidth=1.5,
                       label=f"max n_genes = {s['max_g']}")
        ax.set_xscale("log")
        ax.set_xlabel("Genes detected per cell")
        ax.set_ylabel("Cells")          # consistent across rows
        ax.set_title(f"{s['name']}  ·  n_genes", fontsize=11)
        ax.legend(fontsize=8, frameon=False, loc="upper left")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Col 2 — pct_mt histogram
        ax = axes[row, 1]
        ax.hist(np.clip(s["pct_mt"], 0, 50), bins=60, color=ORANGE,
                alpha=0.85, edgecolor="white", linewidth=0.3)
        ax.axvline(s["mt_pct"], color=RED, linestyle="--", linewidth=1.5,
                   label=f"max pct_mt = {s['mt_pct']:g}%")
        ax.set_xlabel("% mitochondrial counts")
        ax.set_ylabel("Cells")          # consistent across rows
        ax.set_xlim(0, 50)
        ax.set_title(f"{s['name']}  ·  pct_mt", fontsize=11)
        ax.legend(fontsize=8, frameon=False, loc="upper right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Col 3 — scatter total_counts vs n_genes, coloured by pct_mt
        ax = axes[row, 2]
        order = np.argsort(s["pct_mt"])  # plot low pct_mt first, high on top
        sc_h = ax.scatter(
            s["total"][order], s["n_genes"][order],
            c=s["pct_mt"][order], cmap="viridis",
            vmin=0, vmax=MT_CMAP_VMAX,          # shared across rows
            s=3, alpha=0.55, edgecolors="none", rasterized=True,
        )
        scatter_handles.append(sc_h)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Total counts per cell")
        ax.set_ylabel("Genes detected per cell")

        # Horizontal threshold lines (labelled in legend)
        ax.axhline(s["min_g"], color=RED, linestyle="--", linewidth=1.0,
                   label=f"min n_genes = {s['min_g']}")
        if s["max_g"] is not None:
            ax.axhline(s["max_g"], color=RED, linestyle=":", linewidth=1.0,
                       label=f"max n_genes = {s['max_g']}")

        # Vertical total-counts reference line (§2.2.1 dissertation claim)
        ax.axvline(MAX_TOTAL_COUNTS, color="black", linestyle="--",
                   linewidth=1.0,
                   label=f"max counts = {MAX_TOTAL_COUNTS:,}")

        ax.set_title(f"{s['name']}  ·  counts vs n_genes", fontsize=11)
        ax.legend(fontsize=7, frameon=False, loc="lower right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Single shared colourbar for column 3
    cb = fig.colorbar(
        scatter_handles[0],
        ax=axes[:, 2].tolist(),
        pad=0.02, shrink=0.85, aspect=28,
    )
    cb.set_label("% mitochondrial counts", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    fig.suptitle(
        "QC distributions and applied thresholds, per donor\n"
        "Left: n_genes per cell · Middle: % mitochondrial counts · "
        "Right: joint distribution of total counts and n_genes, coloured by % mt",
        fontsize=11, y=0.995,
    )
    fig.savefig(FIG_PATH, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {FIG_PATH}")

    # ───── Retention bar chart ───────────────────────────────────────
    fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4.2))

    names   = [s["name"] for s in stats]
    before  = [s["n_cells_before"] for s in stats]
    after   = [s["n_cells_after"]  for s in stats]
    dropped = [b - a for b, a in zip(before, after)]

    x = np.arange(len(names))
    axes2[0].bar(x, after,   color=BLUE,   label="Retained", edgecolor="white")
    axes2[0].bar(x, dropped, bottom=after, color=GREY, alpha=0.7,
                 label="Dropped", edgecolor="white")
    for xi, (a, b) in enumerate(zip(after, before)):
        pct = 100 * a / b
        axes2[0].text(xi, b + 0.02 * max(before),
                      f"{a:,} / {b:,}\n({pct:.1f}% kept)",
                      ha="center", va="bottom", fontsize=9)
    axes2[0].set_xticks(x)
    axes2[0].set_xticklabels(names)
    axes2[0].set_ylabel("Cells")
    axes2[0].set_title("Cell-level QC (n_genes + pct_mt)")
    axes2[0].set_ylim(0, 1.18 * max(before))
    axes2[0].legend(frameon=False, fontsize=9)
    axes2[0].spines["top"].set_visible(False)
    axes2[0].spines["right"].set_visible(False)

    gbefore = [s["n_genes_before"] for s in stats]
    gafter  = [s["n_genes_after"]  for s in stats]
    gdrop   = [b - a for b, a in zip(gbefore, gafter)]
    axes2[1].bar(x, gafter, color=ORANGE, label="Retained (≥3 cells)",
                 edgecolor="white")
    axes2[1].bar(x, gdrop, bottom=gafter, color=GREY, alpha=0.7,
                 label="Dropped", edgecolor="white")
    for xi, (a, b) in enumerate(zip(gafter, gbefore)):
        pct = 100 * a / b
        axes2[1].text(xi, b + 0.02 * max(gbefore),
                      f"{a:,} / {b:,}\n({pct:.1f}% kept)",
                      ha="center", va="bottom", fontsize=9)
    axes2[1].set_xticks(x)
    axes2[1].set_xticklabels(names)
    axes2[1].set_ylabel("Genes")
    axes2[1].set_title("Gene-level QC (min_cells ≥ 3)")
    axes2[1].set_ylim(0, 1.18 * max(gbefore))
    axes2[1].legend(frameon=False, fontsize=9)
    axes2[1].spines["top"].set_visible(False)
    axes2[1].spines["right"].set_visible(False)

    fig2.suptitle("Retention after QC filtering", fontsize=12, y=1.02)
    fig2.tight_layout()
    fig2.savefig(BAR_PATH, dpi=250, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved {BAR_PATH}")


if __name__ == "__main__":
    main()
