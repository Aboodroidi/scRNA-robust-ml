#!/usr/bin/env python
"""
Build Appendix Table B4: paired bootstrap + Stuart-Maxwell tests of
SANN_PCA against four external tools on the 8K and 3K cross-donor test
sets. Includes both scANVI variants (default + weighted seed-42).

Statistical procedure
---------------------
For each (test donor, comparator) pair:
  1. Align cell-by-cell predictions for SANN and the comparator.
  2. Paired bootstrap (B=1,000 resamples, paired indices):
       Δ_b = macro-F1(SANN, idx_b) − macro-F1(comp, idx_b)
       report mean Δ + 95% percentile CI [2.5, 97.5]
  3. Stuart–Maxwell test of marginal homogeneity on the K×K table
     of (SANN prediction class, comparator prediction class) over the
     un-resampled cells. Tests whether the two methods have different
     marginal prediction distributions; gives a single p-value.
  4. Macro-F1 is averaged only over classes present in the original
     ground truth (matches Table B3's "known-class" definition).

Outputs:
  results/figures/table_b4.md    (markdown table for Appendix B)
  results/figures/table_b4.csv   (machine-readable copy)
"""
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import f1_score
from scipy import stats

ROOT = "/Users/Abdullah/scRNA-robust-ml"
OUT_MD = os.path.join(ROOT, "results/figures/table_b4.md")
OUT_CSV = os.path.join(ROOT, "results/figures/table_b4.csv")

# Canonical 5-class coarse taxonomy
CLASSES = ["B cells", "Mono", "NK", "Platelet", "T cells"]
CLS_IDX = {c: i for i, c in enumerate(CLASSES)}
SANN_INT_TO_NAME = {i: c for i, c in enumerate(sorted(CLASSES))}

LABEL_NORM = {
    "B cells": "B cells", "B cell": "B cells", "B": "B cells",
    "Plasma": "B cells", "Plasma cells": "B cells",
    "Mono": "Mono", "Monocytes": "Mono", "Monocyte": "Mono",
    "CD14+ Monocytes": "Mono", "CD14 Monocytes": "Mono",
    "FCGR3A+ Monocytes": "Mono", "Classical monocytes": "Mono",
    "Non classical monocytes": "Mono",
    "Intermediate monocytes": "Mono",
    "NK": "NK", "NK cells": "NK", "NK cell": "NK",
    "Platelet": "Platelet", "Platelets": "Platelet",
    "Megakaryocytes": "Platelet",
    "T cells": "T cells", "T cell": "T cells",
    "CD8 T": "T cells", "CD4 T": "T cells", "Treg": "T cells",
    "T cells CD8": "T cells", "T cells CD4": "T cells",
    "T helper": "T cells",
}


def norm_label(s):
    if pd.isna(s):
        return None
    s = str(s).strip()
    if s in LABEL_NORM:
        return LABEL_NORM[s]
    for k, v in LABEL_NORM.items():
        if k.lower() == s.lower():
            return v
    return None


# Comparator catalogue. Each entry: (display name, csv path lambda(tag), col)
COMPARATORS = [
    ("Seurat",
     lambda tag: f"{ROOT}/results/comparators/seurat/seurat_{tag}_predictions.csv",
     "predicted_label"),
    ("ACTINN",
     lambda tag: f"{ROOT}/results/comparators/actinn/actinn_{tag}_predictions.csv",
     "predicted_label"),
    ("SingleR",
     lambda tag: f"{ROOT}/results/comparators/singleR/singleR_{tag}_predictions.csv",
     "pruned_label"),
    ("scANVI (default)",
     lambda tag: f"{ROOT}/results/scanvi_results/scanvi_baseline_seed42_{tag}_predictions.csv",
     "predicted_label"),
    ("scANVI (weighted, seed 42)",
     lambda tag: f"{ROOT}/results/scanvi_results/scanvi_weighted_seed42_{tag}_predictions.csv",
     "predicted_label"),
]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_sann_and_truth():
    out = {}
    for tag, h5ad, label_key, sann_pred_path in [
        ("8K",
         f"{ROOT}/data/processed/pbmc8k_labeled.h5ad",
         "cell_type",
         f"{ROOT}/results/external_validation/pca/sann_pca_ext_pred.npy"),
        ("3K",
         f"{ROOT}/data/processed/pbmc3k_labeled.h5ad",
         "cell_type_coarse",
         f"{ROOT}/results/external_validation_3k/pca/sann_pca_ext3k_pred.npy"),
    ]:
        ad = sc.read_h5ad(h5ad)
        raw = ad.obs[label_key].astype(str).to_numpy()
        mapped = np.array([norm_label(s) for s in raw], dtype=object)
        known = np.array([m is not None for m in mapped])
        barcodes = ad.obs_names[known].to_numpy()
        y_true = np.array([mapped[i] for i, k in enumerate(known) if k],
                          dtype=object)

        sann_pred_int = np.load(sann_pred_path)
        assert len(sann_pred_int) == known.sum(), (
            f"{tag}: SANN pred rows ({len(sann_pred_int)}) != known rows "
            f"({known.sum()})")
        sann_pred = np.array(
            [SANN_INT_TO_NAME[i] for i in sann_pred_int], dtype=object)

        out[tag] = {
            "barcodes": barcodes,
            "y_true": y_true,
            "SANN_PCA": sann_pred,
        }
        print(f"  {tag}: {len(barcodes)} cells")
    return out


def load_comparator(tag, barcodes, csv_path, col):
    df = pd.read_csv(csv_path)
    if col not in df.columns:
        col = "predicted_label"
    bmap = dict(zip(df["barcode"].astype(str), df[col].astype(str)))
    pred = np.array(
        [norm_label(bmap.get(str(b), None)) for b in barcodes],
        dtype=object,
    )
    pred = np.array(["OTHER" if p is None else p for p in pred],
                    dtype=object)
    return pred


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
def macro_f1(y_true, y_pred, label_set):
    labels = [CLS_IDX[c] for c in label_set]
    return float(f1_score(
        [CLS_IDX[c] for c in y_true],
        [CLS_IDX.get(c, -1) for c in y_pred],
        labels=labels, average="macro", zero_division=0,
    ))


def paired_bootstrap_delta(y_true, sann_pred, comp_pred, n_boot=1000, seed=42):
    """
    Returns (mean_delta, ci_low, ci_high, deltas_array).
    delta_b = macro_f1(SANN) − macro_f1(comp), same bootstrap idx.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    label_set = sorted({c for c in y_true if c in CLS_IDX})
    deltas = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        f1_s = macro_f1(yt, sann_pred[idx], label_set)
        f1_c = macro_f1(yt, comp_pred[idx], label_set)
        deltas[b] = f1_s - f1_c
    return (float(deltas.mean()),
            float(np.percentile(deltas, 2.5)),
            float(np.percentile(deltas, 97.5)),
            deltas)


def stuart_maxwell(pred_a, pred_b, classes):
    """
    Stuart-Maxwell test of marginal homogeneity for paired multi-class
    predictions. Uses K-1 degrees of freedom. Returns p-value.
    """
    K = len(classes)
    cls_to_i = {c: i for i, c in enumerate(classes)}
    # Build K×K contingency
    N = np.zeros((K, K), dtype=np.float64)
    for a, b in zip(pred_a, pred_b):
        ia = cls_to_i.get(a, None)
        ib = cls_to_i.get(b, None)
        if ia is None or ib is None:
            continue
        N[ia, ib] += 1.0

    row_sum = N.sum(axis=1)
    col_sum = N.sum(axis=0)
    d = row_sum - col_sum  # marginal differences (length K)

    # Drop the last category to avoid singularity (sum of d is 0)
    d_red = d[:-1].reshape(-1, 1)
    K_red = K - 1
    S = np.zeros((K_red, K_red), dtype=np.float64)
    for i in range(K_red):
        S[i, i] = row_sum[i] + col_sum[i] - 2.0 * N[i, i]
        for j in range(K_red):
            if i == j:
                continue
            S[i, j] = -(N[i, j] + N[j, i])

    try:
        S_inv = np.linalg.pinv(S)
        chi2 = float(d_red.T @ S_inv @ d_red)
        df = K_red
        p = float(stats.chi2.sf(chi2, df))
    except np.linalg.LinAlgError:
        return float("nan"), float("nan"), 0
    # Sample size used (cells where both predictions were in CLASSES)
    n_used = int(N.sum())
    return p, chi2, n_used


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading SANN + ground truth ...")
    aligned = load_sann_and_truth()

    rows = []
    for tag in ("8K", "3K"):
        bar = aligned[tag]["barcodes"]
        y_true = aligned[tag]["y_true"]
        sann = aligned[tag]["SANN_PCA"]
        n_cells = len(y_true)
        print(f"\n=== {tag} (n={n_cells}) ===")

        for name, path_fn, col in COMPARATORS:
            comp = load_comparator(tag, bar, path_fn(tag), col)

            mean_d, lo, hi, _ = paired_bootstrap_delta(
                y_true, sann, comp, n_boot=1000, seed=42)
            sm_p, sm_chi2, n_used = stuart_maxwell(sann, comp, CLASSES)

            ci_excludes_zero = (lo > 0) or (hi < 0)
            print(f"  SANN vs {name}: Δ={mean_d:+.4f} "
                  f"[{lo:+.4f}, {hi:+.4f}]  "
                  f"SM p={sm_p:.3g}  (CI excludes 0: {ci_excludes_zero})")

            rows.append({
                "Comparison": f"SANN vs {name}",
                "Test donor": tag,
                "n cells": n_cells,
                "Δ macro-F1 (mean)": mean_d,
                "CI low": lo,
                "CI high": hi,
                "McNemar p": sm_p,
                "Stuart-Maxwell chi2": sm_chi2,
                "CI excludes 0": ci_excludes_zero,
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    # ---------- Markdown table ----------
    def fmt_p(p):
        if p < 1e-3:
            return "<0.001"
        if p < 1e-2:
            return f"{p:.3f}"
        return f"{p:.3f}"

    def fmt_d(v):
        return f"{v:+.4f}"

    def comma(n):
        return f"{n:,}"

    # Pretty ordered comparator list — fix order to match user's template
    template_order = [
        "SANN vs Seurat",
        "SANN vs ACTINN",
        "SANN vs SingleR",
        "SANN vs scANVI (default)",
        "SANN vs scANVI (weighted, seed 42)",
    ]
    md_lines = [
        "**Table B4** Paired bootstrap and Stuart-Maxwell test results "
        "comparing SANN_PCA against four established cell-type "
        "classification tools on the 8K and 3K cross-donor test sets. "
        "Δ macro-F1 is computed as SANN minus baseline on each of 1,000 "
        "paired bootstrap resamples of the test cells; mean and 95% "
        "percentile CI reported. Stuart-Maxwell test (multi-class extension "
        "of McNemar) compares paired marginal prediction distributions on "
        "the same test cells. Positive Δ favours SANN_PCA; CIs excluding "
        "zero indicate the margin is robust to test-set sampling variation. "
        "**Bold** marks comparisons where the 95% CI excludes zero. scANVI "
        "was evaluated under both its default configuration and the "
        "inverse-frequency class-weighted variant introduced in §4.1.3.",
        "",
        "| Comparison | Test donor | n cells | Δ macro-F1 (mean) | 95% CI | "
        "McNemar p |",
        "|---|---|---|---|---|---|",
    ]

    for cmp in template_order:
        for tag in ("8K", "3K"):
            sub = df[(df["Comparison"] == cmp) & (df["Test donor"] == tag)]
            if sub.empty:
                continue
            r = sub.iloc[0]
            d_str = fmt_d(r["Δ macro-F1 (mean)"])
            ci_str = f"[{fmt_d(r['CI low'])}, {fmt_d(r['CI high'])}]"
            p_str = fmt_p(r["McNemar p"])
            if r["CI excludes 0"]:
                d_str = f"**{d_str}**"
                ci_str = f"**{ci_str}**"
            md_lines.append(
                f"| {cmp} | {tag} | {comma(int(r['n cells']))} | "
                f"{d_str} | {ci_str} | {p_str} |"
            )

    with open(OUT_MD, "w") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"\nSaved: {OUT_MD}")
    print(f"Saved: {OUT_CSV}")
    print("\n" + "\n".join(md_lines))


if __name__ == "__main__":
    main()
