import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Mac-friendly + non-GUI plotting
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MPLBACKEND"] = "Agg"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def parse_args():
    p = argparse.ArgumentParser(description="Plot reliability diagrams from saved test probs (LR, XGB, SANN).")
    p.add_argument("--lr_probs", type=str, required=True)
    p.add_argument("--lr_true", type=str, required=True)

    p.add_argument("--xgb_probs", type=str, required=True)
    p.add_argument("--xgb_true", type=str, required=True)

    p.add_argument("--sann_probs", type=str, required=True)
    p.add_argument("--sann_true", type=str, required=True)

    p.add_argument("--out", type=str, default="results/figures/reliability_all_models.png")
    p.add_argument("--bins", type=int, default=10)
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def compute_reliability(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10):
    """
    Multiclass reliability using max softmax confidence.
    Always returns values for all bins (no NaNs) using fixed bin midpoints.
    Empty bins: accuracy=0, confidence=bin midpoint (for visual continuity).
    """
    probs = np.asarray(probs)
    y_true = np.asarray(y_true).astype(int)

    if probs.ndim != 2:
        raise ValueError(f"probs must be 2D (N,C). Got shape {probs.shape}")
    if probs.shape[0] != len(y_true):
        raise ValueError(f"Mismatch N: probs={probs.shape[0]} vs y_true={len(y_true)}")

    # sanity: rows sum ~1
    row_sums = probs.sum(axis=1)
    if not np.all(np.isfinite(row_sums)):
        raise ValueError("probs contains non-finite values.")
    # don't hard-fail for tiny numerical errors, but warn if very off
    if np.max(np.abs(row_sums - 1.0)) > 1e-2:
        print(f"[Warning] probs rows not summing to 1 (max |sum-1|={np.max(np.abs(row_sums-1)):.3e})")

    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(int)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    x_mid = 0.5 * (bins[:-1] + bins[1:])

    # right=True ensures conf==1.0 goes into last bin
    bin_ids = np.digitize(conf, bins, right=True) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    bin_acc = np.zeros(n_bins, dtype=float)
    bin_conf = np.zeros(n_bins, dtype=float)
    bin_counts = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        mask = (bin_ids == b)
        cnt = int(mask.sum())
        bin_counts[b] = cnt
        if cnt > 0:
            bin_acc[b] = float(correct[mask].mean())
            bin_conf[b] = float(conf[mask].mean())
        else:
            bin_acc[b] = 0.0
            bin_conf[b] = float(x_mid[b])  # midpoint

    # ECE
    n_total = len(y_true)
    ece = 0.0
    for b in range(n_bins):
        if bin_counts[b] > 0:
            w = bin_counts[b] / n_total
            ece += w * abs(bin_acc[b] - bin_conf[b])

    return x_mid, bin_acc, bin_counts, float(ece)


def plot_reliability(ax, probs, y_true, title, n_bins=10):
    x_mid, bin_acc, bin_counts, ece = compute_reliability(probs, y_true, n_bins=n_bins)

    # perfect calibration
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)

    # full curve across all bins
    ax.plot(x_mid, bin_acc, marker="o", linewidth=1.5)

    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Observed accuracy")

    ax.text(0.05, 0.08, f"ECE = {ece:.3f}", transform=ax.transAxes, fontsize=10)
    return ece


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    lr_probs = np.load(args.lr_probs)
    lr_true = np.load(args.lr_true)

    xgb_probs = np.load(args.xgb_probs)
    xgb_true = np.load(args.xgb_true)

    sann_probs = np.load(args.sann_probs)
    sann_true = np.load(args.sann_true)

    # sanity: class counts match
    C = lr_probs.shape[1]
    if xgb_probs.shape[1] != C or sann_probs.shape[1] != C:
        raise ValueError(f"Class count mismatch: LR C={C}, XGB C={xgb_probs.shape[1]}, SANN C={sann_probs.shape[1]}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True, sharey=True)

    plot_reliability(axes[0], lr_probs, lr_true, "Logistic Regression", n_bins=args.bins)
    plot_reliability(axes[1], xgb_probs, xgb_true, "XGBoost", n_bins=args.bins)
    plot_reliability(axes[2], sann_probs, sann_true, "SANN", n_bins=args.bins)

    fig.suptitle("Reliability Diagrams (Test Set, 10 Confidence Bins)", y=1.02)
    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"[Saved] {args.out}")


if __name__ == "__main__":
    main()