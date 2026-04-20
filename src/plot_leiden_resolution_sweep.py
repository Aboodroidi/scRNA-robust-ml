"""
Leiden resolution sensitivity sweep on the PBMC 68K training donor.

Closes the "resolution 0.5 is chosen with no sensitivity analysis" critique
by showing how cluster count, label-agreement (ARI against final coarse
labels), and within-cluster silhouette score vary with resolution ∈
{0.3, 0.4, 0.5, 0.6, 0.8, 1.0}.

Outputs:
    results/figures/leiden_resolution_sweep.png
    results/figures/leiden_resolution_sweep.csv
"""
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["MPLBACKEND"] = "Agg"

import time
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)


IN_PATH   = "data/processed/pbmc68k_labeled.h5ad"
OUT_DIR   = "results/figures"
FIG_PATH  = os.path.join(OUT_DIR, "leiden_resolution_sweep.png")
CSV_PATH  = os.path.join(OUT_DIR, "leiden_resolution_sweep.csv")

RESOLUTIONS = [0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
CHOSEN_RES  = 0.5
# Number of final labelled cell types (after post-hoc merging of Leiden
# sub-clusters into named populations + CL x residuals). Used as a
# reference line in panel (a).
N_FINAL_LABELS = 14

# Silhouette is O(n^2) — subsample for tractability
SIL_N_SAMPLE = 5000
SIL_SEED     = 0

BLUE   = "#4C72B0"
ORANGE = "#DD8452"
GREEN  = "#55A868"
RED    = "#C44E52"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Loading {IN_PATH}...")
    adata = sc.read_h5ad(IN_PATH)
    print(f"  shape: {adata.shape}")
    assert "neighbors" in adata.uns, "precomputed neighbor graph required"

    # ── Reference labels (from §2.2.5 final annotation) ────────────
    ref_coarse = adata.obs["cell_type_coarse"].astype(str).values
    ref_fine   = adata.obs["cell_type"].astype(str).values

    # PCA embedding for silhouette
    X_pca = adata.obsm["X_pca"]
    rng = np.random.default_rng(SIL_SEED)
    sample_idx = rng.choice(X_pca.shape[0], size=SIL_N_SAMPLE, replace=False)
    X_sample = X_pca[sample_idx]

    # ── Run Leiden sweep ───────────────────────────────────────────
    records = []
    cluster_cols = {}
    for res in RESOLUTIONS:
        key = f"leiden_r{res}"
        t0 = time.time()
        sc.tl.leiden(adata, resolution=res, key_added=key, random_state=0)
        dt = time.time() - t0
        labels = adata.obs[key].astype(str).values
        n_clust = len(np.unique(labels))

        ari_coarse = adjusted_rand_score(ref_coarse, labels)
        nmi_coarse = normalized_mutual_info_score(ref_coarse, labels)
        ari_fine   = adjusted_rand_score(ref_fine, labels)
        nmi_fine   = normalized_mutual_info_score(ref_fine, labels)

        sample_labels = labels[sample_idx]
        if len(np.unique(sample_labels)) > 1:
            sil = silhouette_score(X_sample, sample_labels, metric="euclidean")
        else:
            sil = np.nan

        records.append({
            "resolution":   res,
            "n_clusters":   n_clust,
            "ari_coarse":   ari_coarse,
            "nmi_coarse":   nmi_coarse,
            "ari_fine":     ari_fine,
            "nmi_fine":     nmi_fine,
            "silhouette":   sil,
            "runtime_s":    dt,
        })
        cluster_cols[res] = labels
        print(f"  r={res:<3}  k={n_clust:>2}  "
              f"ARI_coarse={ari_coarse:.3f}  NMI_coarse={nmi_coarse:.3f}  "
              f"ARI_fine={ari_fine:.3f}  sil={sil:.3f}  ({dt:.1f}s)")

    df = pd.DataFrame(records)
    df.to_csv(CSV_PATH, index=False)
    print(f"\nSaved {CSV_PATH}")

    # ── Figure: 2 rows × 3 cols ────────────────────────────────────
    # Row 1: (a) n_clusters vs res, (b) ARI/NMI vs res, (c) silhouette vs res
    # Row 2: 3 UMAPs @ representative resolutions (0.3, 0.5, 1.0)
    fig = plt.figure(figsize=(14, 8.5))
    gs  = fig.add_gridspec(
        2, 3,
        height_ratios=[1, 1.15],
        wspace=0.32, hspace=0.45,
        left=0.06, right=0.985, top=0.93, bottom=0.07,
    )

    res_arr = df["resolution"].values

    # ── (a) n_clusters vs resolution ───────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(res_arr, df["n_clusters"], marker="o", color=BLUE, lw=1.8,
            label="raw Leiden clusters")
    ax.axvline(CHOSEN_RES, color=RED, linestyle="--", lw=1.2,
               label=f"chosen r = {CHOSEN_RES}")
    ax.axhline(N_FINAL_LABELS, color="#666666", linestyle=":", lw=1.2,
               label=f"final labelled types = {N_FINAL_LABELS}")
    for r, n in zip(res_arr, df["n_clusters"]):
        ax.annotate(f"{n}", (r, n), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=8, color=BLUE)
    ax.set_xlabel("Leiden resolution")
    ax.set_ylabel("Number of clusters")
    ax.set_title("(a) Cluster count vs resolution", fontsize=11)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── (b) ARI / NMI vs resolution ───────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(res_arr, df["ari_coarse"], marker="o", color=BLUE,
            lw=1.8, label="ARI  (vs 5-class coarse)")
    ax.plot(res_arr, df["nmi_coarse"], marker="s", color=GREEN,
            lw=1.8, label="NMI  (vs 5-class coarse)")
    ax.plot(res_arr, df["ari_fine"],   marker="^", color=ORANGE,
            lw=1.4, linestyle="--",
            label="ARI  (vs final fine labels)")
    ax.axvline(CHOSEN_RES, color=RED, linestyle="--", lw=1.2)
    ax.set_xlabel("Leiden resolution")
    ax.set_ylabel("Agreement score")
    ax.set_title("(b) Agreement with final labels", fontsize=11)
    # legend placed below the axis to avoid overlapping the curves
    ax.legend(
        fontsize=7.5, frameon=False,
        loc="upper center", bbox_to_anchor=(0.5, -0.22),
        ncol=1, handlelength=2.2, borderaxespad=0.0,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, 1.02)

    # ── (c) silhouette vs resolution ──────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(res_arr, df["silhouette"], marker="o", color=ORANGE, lw=1.8,
            label=f"silhouette (n={SIL_N_SAMPLE:,})")
    ax.axvline(CHOSEN_RES, color=RED, linestyle="--", lw=1.2)
    ax.set_xlabel("Leiden resolution")
    ax.set_ylabel("Mean silhouette (PCA, k=50)")
    ax.set_title("(c) Within-cluster cohesion", fontsize=11)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── UMAP grid (row 2) ──────────────────────────────────────────
    # Reuse the exact palette stored in the h5ad (same as the "True Labels"
    # reference figure) so that r = 0.5 matches the dissertation figure
    # and r = 0.3 / 1.0 use the same colour family cyclically.
    stored_palette = list(adata.uns.get("leiden_colors", []))
    if len(stored_palette) == 0:
        stored_palette = [plt.get_cmap("tab20")(i) for i in range(20)]

    X_umap = adata.obsm["X_umap"]
    picks = [0.3, 0.5, 1.0]  # sparse, chosen, over-split
    for col, res in enumerate(picks):
        ax = fig.add_subplot(gs[1, col])
        labels = cluster_cols[res]
        uniq = sorted(np.unique(labels), key=lambda x: int(x))
        for i, u in enumerate(uniq):
            m = labels == u
            ax.scatter(
                X_umap[m, 0], X_umap[m, 1],
                s=0.9, color=stored_palette[i % len(stored_palette)],
                alpha=0.6, edgecolors="none", rasterized=True,
            )
        n_c = len(uniq)
        border_color = RED if res == CHOSEN_RES else "#999999"
        border_width = 2.0 if res == CHOSEN_RES else 0.6
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(border_width)
        ax.set_xticks([])
        ax.set_yticks([])
        title = f"resolution = {res}  ·  {n_c} clusters"
        if res == CHOSEN_RES:
            title += f"  (chosen → {N_FINAL_LABELS} labels)"
        ax.set_title(title, fontsize=10,
                      color=RED if res == CHOSEN_RES else "black")

    fig.suptitle(
        "Leiden resolution sensitivity sweep — PBMC 68K",
        fontsize=13, y=0.99,
    )
    fig.savefig(FIG_PATH, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIG_PATH}")


if __name__ == "__main__":
    main()
