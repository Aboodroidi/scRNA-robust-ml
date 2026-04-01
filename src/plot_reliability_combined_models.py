import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description="Plot one combined reliability diagram (LR, XGB, SANN).")
    p.add_argument("--lr_probs", type=str, default="results/full_train/lr_test_probs.npy")
    p.add_argument("--xgb_probs", type=str, default="results/full_train/xgb_test_probs.npy")
    p.add_argument("--sann_probs", type=str, default="results/full_train/sann_test_probs.npy")

    # Use ONE consistent y_true for all models (same split)
    p.add_argument("--y_true", type=str, default="results/full_train/lr_test_true.npy")

    p.add_argument("--bins", type=int, default=10)
    p.add_argument("--out", type=str, default="results/figures/reliability_combined_models.png")
    p.add_argument("--dpi", type=int, default=300)

    # If you want the curve to include empty bins visually, turn this on.
    # This does NOT change ECE (ECE ignores empty bins anyway).
    p.add_argument("--plot_empty_bins", action="store_true", default=False)

    return p.parse_args()


def compute_reliability(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10):
    probs = np.asarray(probs, dtype=float)
    y_true = np.asarray(y_true, dtype=int)

    if probs.ndim != 2:
        raise ValueError(f"probs must be 2D (N,C). Got shape: {probs.shape}")
    if probs.shape[0] != len(y_true):
        raise ValueError(f"Mismatch: probs N={probs.shape[0]} vs y_true={len(y_true)}")

    # prediction confidence + correctness
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(int)

    # bins [0,1]
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(conf, edges, right=False) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    bin_conf = np.full(n_bins, np.nan, dtype=float)
    bin_acc = np.full(n_bins, np.nan, dtype=float)
    bin_counts = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        m = (bin_ids == b)
        cnt = int(m.sum())
        bin_counts[b] = cnt
        if cnt > 0:
            bin_conf[b] = float(conf[m].mean())
            bin_acc[b] = float(correct[m].mean())

    # ECE
    n_total = len(y_true)
    ece = 0.0
    for b in range(n_bins):
        if bin_counts[b] > 0:
            w = bin_counts[b] / n_total
            ece += w * abs(bin_acc[b] - bin_conf[b])

    # also return bin midpoints for optional plotting of empty bins
    mids = (edges[:-1] + edges[1:]) / 2.0
    return bin_conf, bin_acc, bin_counts, float(ece), edges, mids


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Load arrays
    y_true = np.load(args.y_true).astype(int)

    lr_probs = np.load(args.lr_probs)
    xgb_probs = np.load(args.xgb_probs)
    sann_probs = np.load(args.sann_probs)

    # Compute reliability per model
    lr_conf, lr_acc, lr_counts, lr_ece, edges, mids = compute_reliability(lr_probs, y_true, args.bins)
    xgb_conf, xgb_acc, xgb_counts, xgb_ece, _, _ = compute_reliability(xgb_probs, y_true, args.bins)
    sann_conf, sann_acc, sann_counts, sann_ece, _, _ = compute_reliability(sann_probs, y_true, args.bins)

    # Plot
    fig, ax = plt.subplots(figsize=(7.0, 6.0))

    # Perfect calibration reference
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)

    def plot_model(bin_conf, bin_acc, label):
        if args.plot_empty_bins:
            # show all bins on x as midpoints, and plot only where accuracy exists
            # (still real values; empties remain NaN so they won't draw points)
            x = np.where(np.isnan(bin_conf), mids, bin_conf)
            y = bin_acc
            ax.plot(x, y, marker="o", linewidth=1.8, label=label)
        else:
            # plot only bins that have data
            m = ~np.isnan(bin_conf) & ~np.isnan(bin_acc)
            ax.plot(bin_conf[m], bin_acc[m], marker="o", linewidth=1.8, label=label)

    plot_model(lr_conf, lr_acc, f"LR (ECE={lr_ece:.3f})")
    plot_model(xgb_conf, xgb_acc, f"XGB (ECE={xgb_ece:.3f})")
    plot_model(sann_conf, sann_acc, f"SANN (ECE={sann_ece:.3f})")

    # Force axes to show 0..1 and include 0 tick
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 6))  # 0.0,0.2,...,1.0
    ax.set_yticks(np.linspace(0, 1, 6))

    ax.set_title("Reliability Diagram (Test Set)")
    ax.set_xlabel("Mean Predicted Confidence")
    ax.set_ylabel("Observed Accuracy")

    ax.legend(frameon=False, loc="lower right")

    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"[Saved] {args.out}")
    print(f"[ECE] LR={lr_ece:.4f} | XGB={xgb_ece:.4f} | SANN={sann_ece:.4f}")


if __name__ == "__main__":
    main()