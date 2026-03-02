# src/plot_convergence_curves_option2.py
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot SANN convergence (normalized train loss + test Macro F1)."
    )
    p.add_argument(
        "--history",
        type=str,
        default="results/full_train/sann_history.csv",
    )
    p.add_argument(
        "--out",
        type=str,
        default="results/figures/sann_convergence_loss_f1.png",
    )
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--smooth", type=int, default=0,
                   help="Optional moving average window (0 disables).")
    return p.parse_args()


def moving_average(x, w):
    if w <= 1:
        return x
    pad = w // 2
    xpad = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(w) / w
    return np.convolve(xpad, kernel, mode="valid")


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    if not os.path.exists(args.history):
        raise FileNotFoundError(f"History CSV not found: {args.history}")

    df = pd.read_csv(args.history)
    required = {"epoch", "train_loss", "test_macro_f1"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing required columns. Found: {list(df.columns)}")

    df = df.sort_values("epoch")

    epoch = df["epoch"].to_numpy()
    train_loss = df["train_loss"].astype(float).to_numpy()
    test_mf1 = df["test_macro_f1"].astype(float).to_numpy()

    # ---- Normalize train loss to [0,1] for single-axis plotting ----
    loss_min = train_loss.min()
    loss_max = train_loss.max()
    train_loss_norm = (train_loss - loss_min) / (loss_max - loss_min)

    # Optional smoothing (plot only)
    if args.smooth > 1:
        train_loss_plot = moving_average(train_loss_norm, args.smooth)
        test_mf1_plot = moving_average(test_mf1, args.smooth)
    else:
        train_loss_plot = train_loss_norm
        test_mf1_plot = test_mf1

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(8.6, 5.2))

    ax.plot(epoch, train_loss_plot,
            color="blue", linewidth=2, label="Train Loss (normalized)")

    ax.plot(epoch, test_mf1_plot,
            color="red", linewidth=2, label="Test Macro F1")

    ax.set_title("SANN Convergence")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Normalized scale (0–1)")

    # Ensure axis starts at 0
    ax.set_ylim(0, 1)
    ax.set_xlim(epoch.min(), epoch.max())

    # Legend outside
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    ax.grid(False)

    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"[Saved] {args.out}")


if __name__ == "__main__":
    main()