# src/plot_robustness.py
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description="Plot robustness: Macro F1 across repeated splits (mean±std + points).")
    p.add_argument(
        "--csv",
        type=str,
        default="results/reports/robustness_splits_metrics.csv",
        help="CSV containing at least columns: model, macro_f1 (or Macro-F1).",
    )
    p.add_argument("--out", type=str, default="results/figures/robustness_macro_f1_across_splits.png")
    p.add_argument("--csv", type=str, default="results/full_train/reports/robustness_splits_full.csv")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--padding", type=float, default=0.01, help="Y-axis padding around min/max.")
    p.add_argument("--jitter", type=float, default=0.06, help="Horizontal jitter for points.")
    return p.parse_args()


def normalise_model_name(m: str) -> str:
    m = str(m).strip()
    u = m.upper()
    if u in {"LOGISTIC REGRESSION", "LOGREG", "LR"}:
        return "LR"
    if u in {"XGBOOST", "XGB"}:
        return "XGB"
    if u in {"SANN"}:
        return "SANN"
    return m


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    df = pd.read_csv(args.csv)
    if df.empty:
        raise ValueError(f"{args.csv} is empty.")

    # ----- Column detection -----
    # support both "macro_f1" and "Macro-F1"
    if "macro_f1" in df.columns:
        f1_col = "macro_f1"
    elif "Macro-F1" in df.columns:
        f1_col = "Macro-F1"
    else:
        raise ValueError(f"Could not find Macro F1 column. Columns are: {list(df.columns)}")

    if "model" not in df.columns and "Model" in df.columns:
        df = df.rename(columns={"Model": "model"})
    if "model" not in df.columns:
        raise ValueError(f"Could not find 'model' column. Columns are: {list(df.columns)}")

    # standardise names + numeric
    df["model"] = df["model"].apply(normalise_model_name)
    df[f1_col] = df[f1_col].astype(float)

    # keep only expected models in this order if present
    order = ["LR", "XGB", "SANN"]
    models = [m for m in order if m in set(df["model"])]

    if not models:
        raise ValueError(f"No recognised models found in 'model' column. Found: {sorted(df['model'].unique())}")

    # summary stats
    summary = (
        df.groupby("model")[f1_col]
          .agg(["mean", "std", "min", "max", "count"])
          .reindex(models)
          .reset_index()
    )

    # determine y-limits (zoomed)
    y_min = df[f1_col].min()
    y_max = df[f1_col].max()
    lo = max(0.0, y_min - args.padding)
    hi = min(1.0, y_max + args.padding)

    # plot
    fig, ax = plt.subplots(figsize=(8.2, 5.4))

    x = np.arange(len(models)) + 1
    means = summary["mean"].values
    stds = summary["std"].fillna(0.0).values

    # mean ± std error bars
    ax.errorbar(
        x, means, yerr=stds,
        fmt="o", capsize=6, elinewidth=1.5, markersize=7
    )

    # overlay raw points with jitter
    rng = np.random.RandomState(42)
    for i, m in enumerate(models, start=1):
        vals = df.loc[df["model"] == m, f1_col].values
        xj = rng.normal(loc=i, scale=args.jitter, size=len(vals))
        ax.scatter(xj, vals, s=28, alpha=0.85, linewidths=0)

        # annotate mean±std above
        ax.text(
            i,
            means[i - 1] + (stds[i - 1] if stds[i - 1] > 0 else 0.002) + 0.002,
            f"{means[i - 1]:.3f} ± {stds[i - 1]:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
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