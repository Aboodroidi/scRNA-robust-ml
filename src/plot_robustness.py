 # src/plot_robustness.py
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description="Robustness plot with angled min/max labels.")
    p.add_argument("--csv", type=str,
                   default="results/full_train/reports/robustness_splits_full.csv")
    p.add_argument("--out", type=str,
                   default="results/figures/robustness_macro_f1_across_splits.png")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--padding", type=float, default=0.01)
    return p.parse_args()


def normalise_model_name(m: str) -> str:
    u = str(m).strip().upper()
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

    df = pd.read_csv(args.csv)

    if "macro_f1" in df.columns:
        f1_col = "macro_f1"
    elif "Macro-F1" in df.columns:
        f1_col = "Macro-F1"
    else:
        raise ValueError("Macro F1 column not found.")

    if "model" not in df.columns and "Model" in df.columns:
        df = df.rename(columns={"Model": "model"})

    df["model"] = df["model"].apply(normalise_model_name)
    df[f1_col] = df[f1_col].astype(float)

    model_order = ["LR", "XGB", "SANN"]
    models = [m for m in model_order if m in df["model"].unique()]

    colors = {
        "LR": "#1f77b4",
        "XGB": "#ff7f0e",
        "SANN": "#2ca02c",
    }

    stats = []
    for m in models:
        vals = df.loc[df["model"] == m, f1_col].values
        stats.append({
            "model": m,
            "mean": vals.mean(),
            "min": vals.min(),
            "max": vals.max(),
        })

    stats = pd.DataFrame(stats)

    y_min = stats["min"].min()
    y_max = stats["max"].max()
    lo = max(0.0, y_min - args.padding)
    hi = min(1.0, y_max + args.padding)

    fig, ax = plt.subplots(figsize=(9, 6))
    x_positions = np.arange(len(models)) + 1

    for i, m in enumerate(models):
        row = stats.iloc[i]
        color = colors[m]
        x = x_positions[i]

        # vertical min-max bar
        ax.vlines(
            x,
            row["min"],
            row["max"],
            color=color,
            linewidth=5,
            alpha=0.25,
            zorder=1,
        )

        # mean point
        ax.scatter(
            x,
            row["mean"],
            color=color,
            s=110,
            edgecolor="black",
            linewidth=0.6,
            zorder=3,
        )

        offset_x = 0.18
        offset_y = 0.004  # vertical separation for angled layout

        # label side logic
        if m in {"XGB", "SANN"}:
            text_x = x - offset_x
            ha = "right"
        else:
            text_x = x + offset_x
            ha = "left"

        # ----- MAX (angled up /) -----
        ax.annotate(
            f"{row['max']:.3f}",
            xy=(x, row["max"]),
            xytext=(text_x, row["max"] + offset_y),
            ha=ha,
            va="bottom",
            fontsize=10,
            arrowprops=dict(
                arrowstyle="-",
                color=color,
                linewidth=1,
            ),
        )

        # ----- MEAN (horizontal -) -----
        ax.annotate(
            rf"$\bf{{{row['mean']:.3f}}}$",
            xy=(x, row["mean"]),
            xytext=(text_x, row["mean"]),
            ha=ha,
            va="center",
            fontsize=10,
            arrowprops=dict(
                arrowstyle="-",
                color=color,
                linewidth=1,
            ),
        )

        # ----- MIN (angled down \) -----
        ax.annotate(
            f"{row['min']:.3f}",
            xy=(x, row["min"]),
            xytext=(text_x, row["min"] - offset_y),
            ha=ha,
            va="top",
            fontsize=10,
            arrowprops=dict(
                arrowstyle="-",
                color=color,
                linewidth=1,
            ),
        )

    ax.set_xticks(x_positions)
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


if __name__ == "__main__":
    main()