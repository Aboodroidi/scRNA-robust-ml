import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(
        description="Robustness plot comparing PCA vs HVG across repeated splits."
    )

    p.add_argument(
        "--pca_csv",
        type=str,
        default=None,
        help="CSV for PCA robustness results. If omitted, common paths are searched."
    )
    p.add_argument(
        "--hvg_csv",
        type=str,
        default=None,
        help="CSV for HVG robustness results. If omitted, common paths are searched."
    )
    p.add_argument(
        "--out",
        type=str,
        default="figures/robustness_pca_vs_hvg.png",
        help="Output figure path."
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

    if "model" not in df.columns:
        raise ValueError(f"'model' column not found in {path}")

    df = df.copy()
    df["model"] = df["model"].apply(normalise_model_name)
    df["macro_f1"] = pd.to_numeric(df[f1_col], errors="coerce")
    df = df.dropna(subset=["macro_f1"])
    df["rep"] = rep
    return df[["model", "macro_f1", "rep"]]


def main():
    args = parse_args()

    pca_candidates = [
        args.pca_csv,
        "results/full_train_all_pca/reports/robustness_splits_full.csv",
        "results/full_train_all_pca/reports/robustness_splits_pca.csv",
        "results/full_train_all_pca/robustness_splits_full.csv",
    ]
    hvg_candidates = [
        args.hvg_csv,
        "results/full_train_all_hvg/reports/robustness_splits_full.csv",
        "results/full_train_all_hvg/reports/robustness_splits_hvg.csv",
        "results/full_train_all_hvg/robustness_splits_full.csv",
    ]

    pca_csv = find_first_existing(pca_candidates)
    hvg_csv = find_first_existing(hvg_candidates)

    if pca_csv is None:
        raise FileNotFoundError(
            "Could not find PCA robustness CSV. Pass --pca_csv explicitly."
        )
    if hvg_csv is None:
        raise FileNotFoundError(
            "Could not find HVG robustness CSV. Pass --hvg_csv explicitly."
        )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    pca_df = load_split_csv(pca_csv, "PCA")
    hvg_df = load_split_csv(hvg_csv, "HVG")
    df = pd.concat([pca_df, hvg_df], ignore_index=True)

    model_order = ["LR", "XGB", "SANN"]
    models = [m for m in model_order if m in df["model"].unique()]

    colors = {
        "PCA": "#1f77b4",
        "HVG": "#ff7f0e",
    }

    stats = []
    for rep in ["PCA", "HVG"]:
        for m in models:
            vals = df.loc[(df["model"] == m) & (df["rep"] == rep), "macro_f1"].values
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

    y_min = stats["min"].min()
    y_max = stats["max"].max()
    lo = max(0.0, y_min - args.padding)
    hi = min(1.0, y_max + args.padding)

    fig, ax = plt.subplots(figsize=(9.2, 6))
    base_x = np.arange(len(models)) + 1
    width = 0.18

    for i, m in enumerate(models):
        for rep, shift in [("PCA", -width), ("HVG", width)]:
            row = stats[(stats["model"] == m) & (stats["rep"] == rep)]
            if row.empty:
                continue
            row = row.iloc[0]

            color = colors[rep]
            x = base_x[i] + shift

            # min-max bar
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

            # label direction:
            # PCA labels go to the left of point, HVG labels to the right
            if rep == "PCA":
                text_x = x - 0.16
                ha = "right"
            else:
                text_x = x + 0.16
                ha = "left"

            offset_y = 0.004

            # max
            ax.annotate(
                f"{row['max']:.3f}",
                xy=(x, row["max"]),
                xytext=(text_x, row["max"] + offset_y),
                ha=ha,
                va="bottom",
                fontsize=9,
                arrowprops=dict(
                    arrowstyle="-",
                    color=color,
                    linewidth=1,
                ),
            )

            # mean
            ax.annotate(
                f"{row['mean']:.3f}",
                xy=(x, row["mean"]),
                xytext=(text_x, row["mean"]),
                ha=ha,
                va="center",
                fontsize=9,
                arrowprops=dict(
                    arrowstyle="-",
                    color=color,
                    linewidth=1,
                ),
            )

            # min
            ax.annotate(
                f"{row['min']:.3f}",
                xy=(x, row["min"]),
                xytext=(text_x, row["min"] - offset_y),
                ha=ha,
                va="top",
                fontsize=9,
                arrowprops=dict(
                    arrowstyle="-",
                    color=color,
                    linewidth=1,
                ),
            )

    ax.set_xticks(base_x)
    ax.set_xticklabels(models)
    ax.set_title("Robustness Across Repeated Stratified Splits")
    ax.set_xlabel("Model")
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(lo, hi)
    ax.grid(False)

    # legend
    handles = [
        plt.Line2D([0], [0], color=colors["PCA"], lw=5, alpha=0.5, label="PCA"),
        plt.Line2D([0], [0], color=colors["HVG"], lw=5, alpha=0.5, label="HVG"),
    ]
    ax.legend(handles=handles, loc="best")

    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"[Loaded PCA] {pca_csv}")
    print(f"[Loaded HVG] {hvg_csv}")
    print(f"[Saved] {args.out}")


if __name__ == "__main__":
    main()