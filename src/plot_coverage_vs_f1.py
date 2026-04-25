"""
Coverage-vs-macro-F1 curves and per-class rejection bars for LR / XGB / SANN.

Graph 1: macro-F1 on retained cells vs coverage (fraction of cells kept),
         as the confidence threshold tau sweeps. One line per model, two
         side-by-side panels (8K external donor, 3K external donor). SANN
         probabilities are temperature-scaled using results/temp_scaling_sann.json.

Graph 2: per-class rejection rate at a fixed threshold (default tau=0.9).
         Shows WHICH classes the model refuses; ties uncertainty to the
         ambiguous CL 0 / CD8 T boundary story in the error-analysis section.

Outputs (PNG @ 300 dpi):
  figures/coverage_vs_macro_f1.png
  figures/rejection_by_class_tau{TAU}.png
Also writes the underlying tables to:
  results/figures/coverage_vs_macro_f1.csv
  results/figures/rejection_by_class_tau{TAU}.csv
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["MPLBACKEND"] = "Agg"

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score


COARSE = ["B cells", "Mono", "NK", "Platelet", "T cells"]
MODELS = ["lr", "xgb", "sann"]
MODEL_LABELS = {"lr": "LogReg", "xgb": "XGBoost", "sann": "SANN (T-scaled)"}
MODEL_COLORS = {"lr": "#1f77b4", "xgb": "#ff7f0e", "sann": "#2ca02c"}

DONOR_PATHS = {
    "8K": {
        "lr":   ("results/external_validation/pca/lr_pca_ext_probs.npy",
                 "results/external_validation/pca/lr_pca_ext_true.npy"),
        "xgb":  ("results/external_validation/pca/xgb_pca_ext_probs.npy",
                 "results/external_validation/pca/xgb_pca_ext_true.npy"),
        "sann": ("results/external_validation/pca/sann_pca_ext_probs.npy",
                 "results/external_validation/pca/sann_pca_ext_true.npy"),
    },
    "3K": {
        "lr":   ("results/external_validation_3k/pca/lr_pca_ext3k_probs.npy",
                 "results/external_validation_3k/pca/lr_pca_ext3k_true.npy"),
        "xgb":  ("results/external_validation_3k/pca/xgb_pca_ext3k_probs.npy",
                 "results/external_validation_3k/pca/xgb_pca_ext3k_true.npy"),
        "sann": ("results/external_validation_3k/pca/sann_pca_ext3k_probs.npy",
                 "results/external_validation_3k/pca/sann_pca_ext3k_true.npy"),
    },
}

# Canonical temperature for the main pipeline (matches §4.1.1 reported numbers).
TEMP_JSON = "results/full_train/sann_temperature.json"


def apply_temperature(probs: np.ndarray, T: float) -> np.ndarray:
    """Re-softmax probs with temperature T (logits = log probs, divide by T)."""
    probs = np.clip(probs, 1e-12, 1.0)
    logits = np.log(probs)
    z = logits / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def coverage_f1_curve(probs: np.ndarray, y_true: np.ndarray, n_points: int = 51):
    """Return (coverage_array, macro_f1_array) swept over thresholds on max-prob.

    Macro-F1 is computed over the classes ACTUALLY PRESENT in y_true
    (the labelled test set). This avoids dragging the 8K macro down by
    including Platelet (0 GT cells on 8K) as a forced-zero class.
    """
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    present_labels = sorted(set(y_true.tolist()))

    taus = np.unique(np.concatenate([
        np.linspace(0.0, 1.0, n_points),
        np.quantile(conf, np.linspace(0.0, 1.0, n_points)),
    ]))
    taus.sort()

    covs, f1s = [], []
    N = len(y_true)
    for t in taus:
        mask = conf >= t
        cov = mask.sum() / N
        if mask.sum() < 10:
            continue
        # Only score over classes that still have any GT cells retained,
        # otherwise a class with 0 retained GT cells forces F1=0 and creates
        # the spurious staircase in the curve.
        labels_here = [c for c in present_labels if (y_true[mask] == c).any()]
        if not labels_here:
            continue
        f1 = f1_score(
            y_true[mask], pred[mask],
            labels=labels_here,
            average="macro", zero_division=0,
        )
        covs.append(cov)
        f1s.append(f1)
    return np.array(covs), np.array(f1s)


def per_class_rejection(probs: np.ndarray, y_true: np.ndarray, tau: float):
    """Return fraction of cells of each GT class that fail the tau threshold."""
    conf = probs.max(axis=1)
    rejected = conf < tau
    out = {}
    for c_idx, c_name in enumerate(COARSE):
        mask = y_true == c_idx
        n = int(mask.sum())
        if n == 0:
            out[c_name] = (np.nan, 0)
        else:
            out[c_name] = (float(rejected[mask].mean()), n)
    return out


def load_model_probs(paths, T_sann):
    """Load all models for one donor; apply SANN temperature scaling."""
    out = {}
    y_true_ref = None
    for m in MODELS:
        p_path, t_path = paths[m]
        probs = np.load(p_path)
        y_true = np.load(t_path).astype(int)
        if m == "sann":
            probs = apply_temperature(probs, T_sann)
        if y_true_ref is None:
            y_true_ref = y_true
        else:
            assert np.array_equal(y_true, y_true_ref), \
                f"Ground-truth mismatch for {m}"
        out[m] = probs
    return out, y_true_ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau_reject", type=float, default=0.9,
                    help="Confidence threshold for the per-class rejection figure")
    ap.add_argument("--fig_coverage", type=str,
                    default="figures/coverage_vs_macro_f1.png")
    ap.add_argument("--fig_rejection", type=str,
                    default=None,
                    help="Default uses figures/rejection_by_class_tau{TAU}.png")
    ap.add_argument("--csv_out_dir", type=str, default="results/figures")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    if args.fig_rejection is None:
        tau_str = str(args.tau_reject).replace(".", "")
        args.fig_rejection = f"figures/rejection_by_class_tau{tau_str}.png"

    Path("figures").mkdir(parents=True, exist_ok=True)
    Path(args.csv_out_dir).mkdir(parents=True, exist_ok=True)

    # ── temperature ── (canonical full_train/sann_temperature.json uses key "temperature")
    with open(TEMP_JSON) as f:
        tj = json.load(f)
    T = float(tj.get("temperature", tj.get("temperature_T")))
    print(f"[SANN temperature] T = {T:.4f}  (from {TEMP_JSON})")

    # ── load everything ──
    donor_data = {}
    for donor, paths in DONOR_PATHS.items():
        probs_by_model, y_true = load_model_probs(paths, T)
        donor_data[donor] = (probs_by_model, y_true)
        print(f"[{donor}] loaded {len(y_true)} cells, classes present: "
              f"{sorted(set(y_true.tolist()))}")

    # ── sanity check: coverage=1.0 macro-F1 should match the main pipeline ──
    print("\n── SANITY CHECK (coverage=1.0 macro-F1, classes-present-only) ──")
    print("  Compare these to §4.1.1 / Table B3 before trusting the figure.")
    for donor, (probs_by_model, y_true) in donor_data.items():
        present = sorted(set(y_true.tolist()))
        for m in MODELS:
            p = probs_by_model[m]
            pred = p.argmax(axis=1)
            macro = f1_score(y_true, pred, labels=present,
                             average="macro", zero_division=0)
            acc = (pred == y_true).mean()
            print(f"  {donor} {m:<4s} acc={acc:.4f}  "
                  f"macro-F1(classes={present})={macro:.4f}")

    # ══════════════════════════════════════════════════════════════
    # Graph 1: coverage vs macro-F1
    # ══════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    curve_rows = []
    for ax, donor in zip(axes, ["8K", "3K"]):
        probs_by_model, y_true = donor_data[donor]
        for m in MODELS:
            covs, f1s = coverage_f1_curve(probs_by_model[m], y_true)
            ax.plot(covs, f1s, "-", color=MODEL_COLORS[m],
                    label=MODEL_LABELS[m], lw=2)
            for c, f in zip(covs, f1s):
                curve_rows.append({"donor": donor, "model": m,
                                   "coverage": float(c), "macro_f1": float(f)})
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Coverage (fraction of cells retained)")
        ax.set_title(f"{donor} external donor")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Macro-F1 on retained cells")
    axes[0].legend(loc="lower left", frameon=True)
    fig.suptitle("Selective prediction: macro-F1 vs coverage",
                 y=1.02, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(args.fig_coverage, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {args.fig_coverage}")

    pd.DataFrame(curve_rows).to_csv(
        Path(args.csv_out_dir) / "coverage_vs_macro_f1.csv", index=False)

    # ══════════════════════════════════════════════════════════════
    # Graph 2: per-class rejection at tau_reject
    # ══════════════════════════════════════════════════════════════
    rej_rows = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    width = 0.25
    for ax, donor in zip(axes, ["8K", "3K"]):
        probs_by_model, y_true = donor_data[donor]
        # Only show classes that actually have GT cells on this donor.
        present_names = [COARSE[i] for i in sorted(set(y_true.tolist()))]
        x = np.arange(len(present_names))
        for i, m in enumerate(MODELS):
            out = per_class_rejection(probs_by_model[m], y_true, args.tau_reject)
            heights = [out[c][0] if not np.isnan(out[c][0]) else 0.0
                       for c in present_names]
            ax.bar(
                x + (i - 1) * width, heights, width,
                color=MODEL_COLORS[m], label=MODEL_LABELS[m],
                edgecolor="black", linewidth=0.3,
            )
            for c_name, (rate, n) in out.items():
                rej_rows.append({
                    "donor": donor, "model": m, "class": c_name,
                    "tau": args.tau_reject,
                    "rejection_rate": float(rate) if not np.isnan(rate) else None,
                    "n_gt": int(n),
                })
        ax.set_xticks(x)
        ax.set_xticklabels(present_names, rotation=25, ha="right")
        ax.set_ylim(0, 1.02)
        ax.set_title(f"{donor} external donor "
                     f"(N={len(donor_data[donor][1])})")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel(f"Rejection rate at τ={args.tau_reject}")
    axes[0].legend(loc="upper right", frameon=True)
    fig.suptitle(f"Per-class rejection at τ={args.tau_reject} "
                 "(only classes with ground-truth cells shown per donor)",
                 y=1.02, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(args.fig_rejection, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {args.fig_rejection}")

    pd.DataFrame(rej_rows).to_csv(
        Path(args.csv_out_dir) / f"rejection_by_class_tau"
        f"{str(args.tau_reject).replace('.', '')}.csv",
        index=False)

    print("\nDone.")


if __name__ == "__main__":
    main()
