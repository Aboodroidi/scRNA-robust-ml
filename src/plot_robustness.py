# src/plot_robustness.py
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description="Plot robustness: Macro F1 across repeated splits (mean±std + points).")
    p.add_argument("--csv", type=str, default="results/reports/robustness_splits.csv")
    p.add_argument("--out", type=str, default="results/figures/robustness_macro_f1_across_splits.png")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--padding", type=float, default=0.01, help="Y-axis padding around min/max.")
    p.add_argument("--jitter", type=float, default=0.06, help="Horizontal jitter for points.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    df = pd.read_csv(args.csv)
    if df.empty:
        raise ValueError("robustness_splits.csv is empty.")

    # keep only expected models in this order if present
    order = ["LR", "XGB", "SANN"]
    models = [m for m in order if m in set(df["model"])]

    # summary stats
    summary = (
        df.groupby("model")["macro_f1"]
          .agg(["mean", "std", "min", "max", "count"])
          .reindex(models)
          .reset_index()
    )

    # determine y-limits (zoomed)
    y_min = df["macro_f1"].min()
    y_max = df["macro_f1"].max()
    lo = max(0.0, y_min - args.padding)
    hi = min(1.0, y_max + args.padding)

    # plot
    fig, ax = plt.subplots(figsize=(8.2, 5.4))

    x = np.arange(len(models)) + 1
    means = summary["mean"].values
    stds = summary["std"].fillna(0.0).values

    # mean ± std error bars (no custom colors to respect your preference)
    ax.errorbar(
        x, means, yerr=stds,
        fmt="o", capsize=6, elinewidth=1.5, markersize=7
    )

    # overlay raw points with jitter
    rng = np.random.RandomState(42)
    for i, m in enumerate(models, start=1):
        vals = df.loc[df["model"] == m, "macro_f1"].values
        xj = rng.normal(loc=i, scale=args.jitter, size=len(vals))
        ax.scatter(xj, vals, s=28, alpha=0.85, linewidths=0)

        # annotate mean±std above
        ax.text(
            i, means[i-1] + (stds[i-1] if stds[i-1] > 0 else 0.002) + 0.002,
            f"{means[i-1]:.3f} ± {stds[i-1]:.3f}",
            ha="center", va="bottom", fontsize=10
        )

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_title("Robustness Across Repeated Stratified Splits")
    ax.set_xlabel("Model")
    ax.set_ylabel("Macro F1")
    ax.set_ylim(lo, hi)
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"[Saved] {args.out}")
    print("\n[Summary] Macro F1 across splits:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()