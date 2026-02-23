# src/plot_runtime.py
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description="Plot training runtime per model (points + mean±std).")
    p.add_argument("--csv", type=str, default="results/reports/robustness_splits.csv",
                   help="Long CSV with columns: model, train_time_seconds (and optionally split_id/seed).")
    p.add_argument("--out_fig", type=str, default="results/figures/runtime_time_per_model.png")
    p.add_argument("--out_summary", type=str, default="results/reports/runtime_summary.csv")
    p.add_argument("--dpi", type=int, default=300)

    p.add_argument("--unit", type=str, default="seconds", choices=["seconds", "minutes"],
                   help="Plot in seconds or minutes.")
    p.add_argument("--jitter", type=float, default=0.06, help="Horizontal jitter for points.")
    return p.parse_args()


def standardise_model_name(x: str) -> str:
    x = str(x).strip()
    x_upper = x.upper()
    if x_upper in ["LOGISTIC REGRESSION", "LOGREG", "LR"]:
        return "LR"
    if x_upper in ["XGBOOST", "XGB"]:
        return "XGB"
    if x_upper in ["SANN"]:
        return "SANN"
    return x_upper


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_fig), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_summary), exist_ok=True)

    df = pd.read_csv(args.csv)

    # sanity check columns
    required = {"model", "train_time_seconds"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns {missing}. Found columns: {list(df.columns)}")

    # clean + standardise names
    df = df.copy()
    df["model"] = df["model"].apply(standardise_model_name)
    df["train_time_seconds"] = pd.to_numeric(df["train_time_seconds"], errors="coerce")
    df = df.dropna(subset=["train_time_seconds"])
    df = df[df["train_time_seconds"] > 0]

    # keep only expected models in desired order (if present)
    order = ["LR", "XGB", "SANN"]
    models = [m for m in order if m in set(df["model"])]
    if not models:
        raise ValueError("No recognised models found after standardisation (expected LR/XGB/SANN).")

    # convert unit if needed
    scale = 60.0 if args.unit == "minutes" else 1.0
    df["time_unit"] = df["train_time_seconds"] / scale
    y_label = "Training time (minutes)" if args.unit == "minutes" else "Training time (seconds)"

    # summary stats
    summary = (
        df.groupby("model")["time_unit"]
          .agg(["mean", "std", "count", "min", "max"])
          .reindex(models)
          .reset_index()
    )
    summary.to_csv(args.out_summary, index=False)

    print("\n[Runtime Summary]")
    for _, r in summary.iterrows():
        m = r["model"]
        mean = r["mean"]
        std = r["std"] if not np.isnan(r["std"]) else 0.0
        n = int(r["count"])
        unit = "min" if args.unit == "minutes" else "s"
        print(f"  {m}: {mean:.2f} ± {std:.2f} {unit} (n={n})")
    print(f"[Saved] {args.out_summary}")

    # plot
    fig, ax = plt.subplots(figsize=(8.2, 5.4))

    x = np.arange(len(models)) + 1
    means = summary["mean"].values
    stds = summary["std"].fillna(0.0).values

    # mean ± std error bars
    ax.errorbar(x, means, yerr=stds, fmt="o", capsize=6, elinewidth=1.5, markersize=7)

    # jittered raw points
    rng = np.random.RandomState(42)
    for i, m in enumerate(models, start=1):
        vals = df.loc[df["model"] == m, "time_unit"].values
        xj = rng.normal(loc=i, scale=args.jitter, size=len(vals))
        ax.scatter(xj, vals, s=28, alpha=0.85, linewidths=0)

        # annotate mean ± std
        std_val = stds[i-1]
        offset = (std_val if std_val > 0 else (0.02 * means[i-1] if means[i-1] > 0 else 0.2))
        unit_txt = "min" if args.unit == "minutes" else "s"
        ax.text(
            i, means[i-1] + offset + 0.02 * means[i-1],
            f"{means[i-1]:.2f} ± {stds[i-1]:.2f} {unit_txt}",
            ha="center", va="bottom", fontsize=10
        )

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_title("Training Time per Model (Repeated Splits)")
    ax.set_xlabel("Model")
    ax.set_ylabel(y_label)
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(args.out_fig, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"[Saved] {args.out_fig}")


if __name__ == "__main__":
    main()