import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(
        description="Robustness plot comparing PCA vs HVG across repeated splits."
    )

    p.add_argument("--pca_csv", type=str, default=None)
    p.add_argument("--hvg_csv", type=str, default=None)

    p.add_argument(
        "--out",
        type=str,
        default="figures/robustness_pca_vs_hvg.png",
    )

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
    return u


def find_first_existing(candidates):
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def load_split_csv(path: str, rep: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "macro_f1" in df.columns:
        f1_col = "macro_f1"
    elif "Macro-F1" in df.columns:
        f1_col = "Macro-F1"
    else:
        raise ValueError(f"Macro F1 column not found in {path}")

    if "model" not in df.columns and "Model" in df.columns:
        df = df.rename(columns={"Model": "model"})

    df = df.copy()
    df["model"] = df["model"].apply(normalise_model_name)
    df["macro_f1"] = pd.to_numeric(df[f1_col], errors="coerce")
    df = df.dropna(subset=["macro_f1"])
    df["rep"] = rep

    return df[["model", "macro_f1", "rep"]]


def main():
    args = parse_args()

    pca_csv = find_first_existing([
        args.pca_csv,
        "results/full_train_all_pca/reports/robustness_splits_full.csv",
    ])

    hvg_csv = find_first_existing([
        args.hvg_csv,
        "results/full_train_all_hvg/reports/robustness_splits_full.csv",
    ])

    if pca_csv is None or hvg_csv is None:
        raise FileNotFoundError("Missing PCA or HVG CSV")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    df = pd.concat([
        load_split_csv(pca_csv, "PCA"),
        load_split_csv(hvg_csv, "HVG")
    ])

    models = ["LR", "XGB", "SANN"]

    colors = {
        "PCA": "#1f77b4",
        "HVG": "#ff7f0e",
    }

    stats = []
    for rep in ["PCA", "HVG"]:
        for m in models:
            vals = df[(df["model"] == m) & (df["rep"] == rep)]["macro_f1"].values
            if len(vals) == 0:
                continue
            stats.append({
                "model": m,
                "rep": rep,
                "mean": vals.mean(),
                "min": vals.min(),
                "max": vals.max(),
            })

    stats = pd.DataFrame(stats)

    lo = max(0.0, stats["min"].min() - args.padding)
    hi = min(1.0, stats["max"].max() + args.padding)

    fig, ax = plt.subplots(figsize=(10, 6))

    x_positions = []
    labels = []
    x = 1

    label_dx = 0.10
    offset_y = 0.004

    for m in models:
        for rep in ["PCA", "HVG"]:
            xp = x if rep == "PCA" else x + 0.35

            x_positions.append(xp)
            labels.append(f"{m}\n{rep}")

            row = stats[(stats["model"] == m) & (stats["rep"] == rep)]
            if row.empty:
                continue

            row = row.iloc[0]
            color = colors[rep]

            # min-max bar
            ax.vlines(xp, row["min"], row["max"], color=color, linewidth=5, alpha=0.25)

            # mean point
            ax.scatter(xp, row["mean"], color=color, s=110, edgecolor="black")

            # Projection lines
            ax.vlines(
                xp,
                lo,
                row["mean"],
                color=color,
                linestyle="--",
                alpha=0.5,
                linewidth=1.2,
                zorder=0,
            )

            ax.hlines(
                row["mean"],
                0.5,
                xp,
                color=color,
                linestyle="--",
                alpha=0.5,
                linewidth=1.2,
                zorder=0,
            )

            # labels
            text_x = xp + label_dx

            ax.annotate(f"{row['max']:.3f}", xy=(xp, row["max"]),
                        xytext=(text_x, row["max"] + offset_y),
                        ha="left", fontsize=9,
                        arrowprops=dict(arrowstyle="-", color=color))

            ax.annotate(f"{row['mean']:.3f}", xy=(xp, row["mean"]),
                        xytext=(text_x, row["mean"]),
                        ha="left", fontsize=9,
                        arrowprops=dict(arrowstyle="-", color=color))

            ax.annotate(f"{row['min']:.3f}", xy=(xp, row["min"]),
                        xytext=(text_x, row["min"] - offset_y),
                        ha="left", fontsize=9,
                        arrowprops=dict(arrowstyle="-", color=color))

        x += 1.2

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)

    ax.set_xlim(0.5, max(x_positions) + 0.6)
    ax.set_ylim(lo, hi)

    ax.set_title("Robustness Across Repeated Stratified Splits")
    ax.set_xlabel("Model and Representation")
    ax.set_ylabel("Macro-F1")

    ax.grid(False)

    handles = [
        plt.Line2D([0], [0], color=colors["PCA"], lw=3, linestyle="--", alpha=0.5, label="PCA"),
        plt.Line2D([0], [0], color=colors["HVG"], lw=3, linestyle="--", alpha=0.5, label="HVG"),
    ]
    ax.legend(handles=handles)

    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"[Saved] {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()