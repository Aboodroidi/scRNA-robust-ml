"""
Pairwise Wilcoxon signed-rank + Holm-Bonferroni across 6 models on 5 splits.

Input
-----
  results/full_train_all_pca/reports/robustness_splits_full.csv
  results/full_train_all_hvg/reports/robustness_splits_full.csv

Both files contain per-split macro-F1 for {LR, XGB, SANN}. We stack them
into a single 5x6 matrix with columns
  [LR_PCA, LR_HVG, XGB_PCA, XGB_HVG, SANN_PCA, SANN_HVG]
and each row a stratified split (seeds 42, 53, 64, 75, 86).

Computation
-----------
  - 15 unique model pairs.
  - Two-sided Wilcoxon signed-rank test on each pair of 5 paired scores.
  - Holm-Bonferroni correction via statsmodels.multipletests(method='holm').

Output
------
  Console: 6x6 symmetric table of Holm-corrected p-values, annotated with
           direction (which model has the higher mean) and a * marker
           where corrected p < 0.05.
  CSV:     results/figures/pairwise_wilcoxon_holm.csv (numeric p-values).
  CSV:     results/figures/pairwise_wilcoxon_holm_annotated.csv (with direction).
  TeX:     results/figures/pairwise_wilcoxon_holm.tex (ready for dissertation).
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["MPLBACKEND"] = "Agg"

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scikit_posthocs as sp
from scipy.stats import wilcoxon, ttest_rel, friedmanchisquare, rankdata
from statsmodels.stats.multitest import multipletests


PCA_CSV = "results/full_train_all_pca/reports/robustness_splits_full.csv"
HVG_CSV = "results/full_train_all_hvg/reports/robustness_splits_full.csv"

COLS = ["LR_PCA", "LR_HVG", "XGB_PCA", "XGB_HVG", "SANN_PCA", "SANN_HVG"]

OUT_DIR = Path("results/figures")


def load_matrix() -> pd.DataFrame:
    pca = pd.read_csv(PCA_CSV)
    hvg = pd.read_csv(HVG_CSV)

    pca_piv = pca.pivot(index="split_id", columns="model", values="macro_f1")
    hvg_piv = hvg.pivot(index="split_id", columns="model", values="macro_f1")

    pca_piv.columns = [f"{m}_PCA" for m in pca_piv.columns]
    hvg_piv.columns = [f"{m}_HVG" for m in hvg_piv.columns]

    mat = pd.concat([pca_piv, hvg_piv], axis=1)[COLS].sort_index()
    return mat


def pairwise(mat: pd.DataFrame):
    """Run both Wilcoxon signed-rank and paired t-test on each model pair."""
    pairs = list(combinations(COLS, 2))
    raw_p_w, raw_p_t = [], []
    stats_w, stats_t, directions = [], [], []

    for a, b in pairs:
        xa = mat[a].values
        xb = mat[b].values
        diff = xa - xb

        if np.all(diff == 0):
            raw_p_w.append(1.0); stats_w.append(0.0)
            raw_p_t.append(1.0); stats_t.append(0.0)
            directions.append("=")
            continue

        # Wilcoxon signed-rank (non-parametric, floored at 0.0625 for n=5)
        try:
            w = wilcoxon(xa, xb, alternative="two-sided",
                         zero_method="wilcox", correction=False,
                         mode="exact")
        except ValueError:
            w = wilcoxon(xa, xb, alternative="two-sided",
                         zero_method="wilcox", correction=False,
                         mode="approx")
        raw_p_w.append(float(w.pvalue))
        stats_w.append(float(w.statistic))

        # Paired t-test (parametric, not floored)
        t = ttest_rel(xa, xb, alternative="two-sided")
        raw_p_t.append(float(t.pvalue))
        stats_t.append(float(t.statistic))

        directions.append(">" if xa.mean() > xb.mean() else
                          ("<" if xa.mean() < xb.mean() else "="))

    # Holm-Bonferroni on each family
    _, p_holm_w, _, _ = multipletests(raw_p_w, method="holm")
    _, p_holm_t, _, _ = multipletests(raw_p_t, method="holm")

    rows = []
    for (a, b), rpw, hpw, stw, rpt, hpt, stt, d in zip(
            pairs, raw_p_w, p_holm_w, stats_w,
            raw_p_t, p_holm_t, stats_t, directions):
        rows.append({
            "model_a":   a,
            "model_b":   b,
            "mean_a":    float(mat[a].mean()),
            "mean_b":    float(mat[b].mean()),
            "direction": f"{a} {d} {b}",
            "wilcoxon_stat":  stw,
            "wilcoxon_p_raw":  rpw,
            "wilcoxon_p_holm": float(hpw),
            "wilcoxon_reject_0.05":  bool(hpw < 0.05),
            "ttest_stat":  stt,
            "ttest_p_raw":  rpt,
            "ttest_p_holm": float(hpt),
            "ttest_reject_0.05": bool(hpt < 0.05),
        })
    return pd.DataFrame(rows)


def build_symmetric(pair_df: pd.DataFrame, test: str):
    """test ∈ {'wilcoxon', 'ttest'}"""
    p_col      = f"{test}_p_holm"
    reject_col = f"{test}_reject_0.05"
    p_mat   = pd.DataFrame(np.nan, index=COLS, columns=COLS, dtype=float)
    ann_mat = pd.DataFrame("—",    index=COLS, columns=COLS, dtype=object)

    for _, r in pair_df.iterrows():
        a, b = r["model_a"], r["model_b"]
        p = r[p_col]
        p_mat.loc[a, b] = p
        p_mat.loc[b, a] = p

        better = a if r["mean_a"] > r["mean_b"] else b
        star = "*" if r[reject_col] else ""
        note = f"{p:.4g}{star} ({better} >)"
        ann_mat.loc[a, b] = note
        ann_mat.loc[b, a] = note

    return p_mat, ann_mat


def to_latex(ann_mat: pd.DataFrame) -> str:
    header = " & ".join([""] + ann_mat.columns.tolist()) + r" \\"
    lines = [r"\begin{tabular}{l" + "r" * len(ann_mat.columns) + "}",
             r"\toprule", header, r"\midrule"]
    for idx in ann_mat.index:
        row = [idx] + [
            ann_mat.loc[idx, c].replace("*", r"$^{*}$").replace(">", r"$>$")
            if idx != c else "---"
            for c in ann_mat.columns
        ]
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mat = load_matrix()
    print("── 5x6 macro-F1 matrix (rows=splits, cols=models) ──")
    print(mat.round(4).to_string())
    print(f"\nColumn means:\n{mat.mean().round(4).to_string()}")
    print(f"Column stds: \n{mat.std().round(4).to_string()}")

    pair_df = pairwise(mat)
    print("\n── All 15 pairwise tests (sorted by raw Wilcoxon p) ──")
    show_cols = ["model_a","model_b","mean_a","mean_b","direction",
                 "wilcoxon_p_raw","wilcoxon_p_holm","wilcoxon_reject_0.05",
                 "ttest_p_raw","ttest_p_holm","ttest_reject_0.05"]
    print(pair_df[show_cols].sort_values("wilcoxon_p_raw")
          .to_string(index=False))

    print("\n" + "═"*72)
    print("NOTE: The exact two-sided Wilcoxon signed-rank test at n=5 has a")
    print("floor of 2/2^5 = 0.0625. With 15 comparisons, Holm correction")
    print("cannot produce any p<0.05. The paired t-test does not share this")
    print("floor and should be used for the primary significance claims.")
    print("═"*72)

    for test in ("wilcoxon", "ttest"):
        p_mat, ann_mat = build_symmetric(pair_df, test)
        print(f"\n── 6x6 Holm-corrected p-values ({test}) ──")
        print(p_mat.round(4).to_string())

        print(f"\n── 6x6 annotated ({test}; * = Holm p<0.05) ──")
        pd.set_option("display.width", 220)
        pd.set_option("display.max_colwidth", 30)
        print(ann_mat.to_string())

        p_mat.to_csv(OUT_DIR / f"pairwise_{test}_holm.csv")
        ann_mat.to_csv(OUT_DIR / f"pairwise_{test}_holm_annotated.csv")
        (OUT_DIR / f"pairwise_{test}_holm.tex").write_text(to_latex(ann_mat))

    pair_df.to_csv(OUT_DIR / "pairwise_tests_long.csv", index=False)

    # ══════════════════════════════════════════════════════════════════
    # Friedman omnibus + Nemenyi post-hoc (appropriate non-parametric
    # alternative for comparing >2 models on paired blocks; does not
    # share Wilcoxon's n=5 exact-test floor because it ranks across
    # models within each split).
    # ══════════════════════════════════════════════════════════════════
    chi2, p_friedman = friedmanchisquare(*[mat[c].values for c in COLS])
    # Average ranks (lower rank = better if we rank by macro-F1 descending)
    # scikit-posthocs convention ranks ascending, so negate macro-F1 first
    # so that higher F1 → lower rank (rank 1 = best).
    neg = -mat[COLS].values
    ranks_per_split = np.apply_along_axis(rankdata, 1, neg)
    avg_ranks = pd.Series(ranks_per_split.mean(axis=0), index=COLS,
                          name="avg_rank").sort_values()
    print("\n── Friedman omnibus ──")
    print(f"  chi2 = {chi2:.4f},  p = {p_friedman:.5g}  (df = {len(COLS)-1})")
    print("  Average rank (1 = best):")
    print(avg_ranks.round(3).to_string())

    nemenyi = sp.posthoc_nemenyi_friedman(mat[COLS].values)
    nemenyi.index = COLS
    nemenyi.columns = COLS
    np.fill_diagonal(nemenyi.values, np.nan)
    print("\n── Nemenyi post-hoc p-values (6x6) ──")
    print(nemenyi.round(4).to_string())

    nemenyi.to_csv(OUT_DIR / "pairwise_nemenyi.csv")
    avg_ranks.to_csv(OUT_DIR / "friedman_average_ranks.csv", header=True)

    # Annotated Nemenyi matrix
    nem_ann = pd.DataFrame("—", index=COLS, columns=COLS, dtype=object)
    for a in COLS:
        for b in COLS:
            if a == b:
                continue
            p = nemenyi.loc[a, b]
            star = "*" if p < 0.05 else ""
            better = a if mat[a].mean() > mat[b].mean() else b
            nem_ann.loc[a, b] = f"{p:.4g}{star} ({better} >)"
    print("\n── 6x6 annotated Nemenyi (* = p<0.05) ──")
    print(nem_ann.to_string())
    nem_ann.to_csv(OUT_DIR / "pairwise_nemenyi_annotated.csv")
    (OUT_DIR / "pairwise_nemenyi.tex").write_text(to_latex(nem_ann))

    # ══════════════════════════════════════════════════════════════════
    # Heatmap figures
    # ══════════════════════════════════════════════════════════════════
    def draw_heatmap(p_mat, title, outpath, alpha=0.05):
        fig, ax = plt.subplots(figsize=(8.5, 7))
        vals = p_mat.values.astype(float)
        # Use -log10 for visual dynamic range
        with np.errstate(divide="ignore"):
            disp = -np.log10(np.clip(vals, 1e-12, 1.0))
        disp_masked = np.ma.masked_invalid(disp)
        vmax = float(np.nanmax(disp_masked)) if np.isfinite(np.nanmax(disp_masked)) else 1.0
        im = ax.imshow(disp_masked, cmap="viridis", aspect="auto",
                       vmin=0.0, vmax=max(vmax, 1.0))
        for i in range(p_mat.shape[0]):
            for j in range(p_mat.shape[1]):
                if i == j:
                    ax.text(j, i, "—", ha="center", va="center",
                            color="white", fontsize=14, fontweight="bold")
                else:
                    v = p_mat.iloc[i, j]
                    label = f"{v:.1e}" if v < 0.001 else f"{v:.3f}"
                    mark = "*" if v < alpha else ""
                    better = COLS[i] if mat[COLS[i]].mean() > mat[COLS[j]].mean() \
                        else COLS[j]
                    ax.text(j, i, f"{label}{mark}\n{better}>",
                            ha="center", va="center",
                            color="white",
                            fontsize=10, fontweight="bold")
        ax.set_xticks(range(len(COLS)))
        ax.set_yticks(range(len(COLS)))
        ax.set_xticklabels(COLS, rotation=25, ha="right", fontsize=11)
        ax.set_yticklabels(COLS, fontsize=11)
        cbar = plt.colorbar(im, ax=ax, fraction=0.04)
        cbar.set_label(r"$-\log_{10}(p_{\mathrm{Holm}})$", fontsize=11)
        cbar.mappable.set_clim(0.0, max(vmax, 1.0))
        cbar.ax.tick_params(labelsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold")
        fig.tight_layout()
        fig.savefig(outpath, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] {outpath}")

    p_mat_t, _ = build_symmetric(pair_df, "ttest")
    p_mat_w, _ = build_symmetric(pair_df, "wilcoxon")

    Path("figures").mkdir(exist_ok=True)
    draw_heatmap(
        p_mat_t,
        "Pairwise paired t-test, Holm-Bonferroni corrected (α=0.05)",
        "figures/pairwise_ttest_heatmap.png",
    )
    draw_heatmap(
        p_mat_w,
        "Pairwise Wilcoxon signed-rank, Holm-Bonferroni "
        "(all values = 0.9375, floored at n=5)",
        "figures/pairwise_wilcoxon_heatmap.png",
    )
    draw_heatmap(
        nemenyi,
        f"Nemenyi post-hoc (Friedman χ²={chi2:.2f}, p={p_friedman:.2g})",
        "figures/pairwise_nemenyi_heatmap.png",
    )

    print(f"\nSaved:")
    for name in ["pairwise_wilcoxon_holm.csv",
                 "pairwise_wilcoxon_holm_annotated.csv",
                 "pairwise_wilcoxon_holm.tex",
                 "pairwise_ttest_holm.csv",
                 "pairwise_ttest_holm_annotated.csv",
                 "pairwise_ttest_holm.tex",
                 "pairwise_tests_long.csv",
                 "pairwise_nemenyi.csv",
                 "pairwise_nemenyi_annotated.csv",
                 "pairwise_nemenyi.tex",
                 "friedman_average_ranks.csv"]:
        print(f"  {OUT_DIR / name}")
    for name in ["pairwise_ttest_heatmap.png",
                 "pairwise_wilcoxon_heatmap.png",
                 "pairwise_nemenyi_heatmap.png"]:
        print(f"  figures/{name}")


if __name__ == "__main__":
    main()
