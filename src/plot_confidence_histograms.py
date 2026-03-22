# src/plot_confidence_histograms.py
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description="Plot confidence histograms (proportions) for LR, XGB, and SANN.")
    p.add_argument("--lr_probs", type=str, default="results/full_train/lr_test_probs.npy")
    p.add_argument("--xgb_probs", type=str, default="results/full_train/xgb_test_probs.npy")
    p.add_argument("--sann_probs", type=str, default="results/full_train/sann_test_probs.npy")
    p.add_argument("--out", type=str, default="results/figures/confidence_histograms.png")
    p.add_argument("--bins", type=int, default=10)
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def load_confidence(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    probs = np.load(path)

    if probs.ndim != 2:
        raise ValueError(f"Expected 2D probability array at {path}, got shape {probs.shape}")

    # confidence = max probability per sample
    conf = probs.max(axis=1)
    return conf


def plot_one(ax, conf, title, bins):
    edges = np.linspace(0.0, 1.0, bins + 1)

    # convert counts → proportions
    weights = np.ones_like(conf) / len(conf)

    ax.hist(conf, bins=edges, weights=weights)

    ax.set_title(title)
    ax.set_xlabel("Prediction confidence")
    ax.set_ylabel("Proportion of predictions")
    ax.set_xlim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 11))
    ax.grid(False)

    # summary stats (clean + compact)
    ax.text(
        0.03, 0.95,
        f"min={conf.min():.2f}\nmean={conf.mean():.2f}\nmax={conf.max():.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # load confidence arrays
    lr_conf = load_confidence(args.lr_probs)
    xgb_conf = load_confidence(args.xgb_probs)
    sann_conf = load_confidence(args.sann_probs)

    # sanity prints
    print(f"[Sanity] LR   range: {lr_conf.min():.4f} → {lr_conf.max():.4f} | mean={lr_conf.mean():.4f}")
    print(f"[Sanity] XGB  range: {xgb_conf.min():.4f} → {xgb_conf.max():.4f} | mean={xgb_conf.mean():.4f}")
    print(f"[Sanity] SANN range: {sann_conf.min():.4f} → {sann_conf.max():.4f} | mean={sann_conf.mean():.4f}")

    # plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)

    plot_one(axes[0], lr_conf, "Logistic Regression", args.bins)
    plot_one(axes[1], xgb_conf, "XGBoost", args.bins)
    plot_one(axes[2], sann_conf, "SANN", args.bins)

    fig.suptitle("Distribution of Prediction Confidence Across Models", y=1.02)

    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"[Saved] {args.out}")


if __name__ == "__main__":
    main()