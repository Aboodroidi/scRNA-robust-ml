# src/plot_convergence_curves_option2.py
"""
Produce two figures:
  1) Validation Loss  — all 3 models overlaid, 1 row per rep (HVG, PCA)
  2) Validation Macro-F1 — same layout
X-axis = wall-clock seconds.
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot convergence curves for LR, XGB, and SANN (HVG + PCA combined)."
    )
    p.add_argument("--hvg_dir", type=str, default="results/full_train_all_hvg")
    p.add_argument("--pca_dir", type=str, default="results/full_train_all_pca")
    p.add_argument("--outdir", type=str, default="results/figures")
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


def load_sann(path):
    df = pd.read_csv(path).sort_values("epoch")
    return (
        df["elapsed_seconds"].astype(float).to_numpy(),
        df["val_loss"].astype(float).to_numpy(),
        df["val_macro_f1"].astype(float).to_numpy(),
    )


def load_lr(path):
    df = pd.read_csv(path).sort_values("iteration")
    return (
        df["elapsed_seconds"].astype(float).to_numpy(),
        df["val_loss"].astype(float).to_numpy(),
        df["val_macro_f1"].astype(float).to_numpy(),
    )


def load_xgb(path):
    df = pd.read_csv(path).sort_values("round")
    t = df["elapsed_seconds"].astype(float).to_numpy()
    val_loss = df["val_loss"].astype(float).to_numpy()

    f1_raw = df["val_macro_f1"].astype(float).to_numpy()
    valid = ~np.isnan(f1_raw)
    f1_interp = np.interp(t, t[valid], f1_raw[valid])

    return t, val_loss, f1_interp


# ---------- styling ----------
COLORS = {
    "LR":   "#1b9e77",
    "XGB":  "#d95f02",
    "SANN": "#7570b3",
}

LOADERS = {
    "LR":   ("lr_history.csv",   load_lr),
    "XGB":  ("xgb_history.csv",  load_xgb),
    "SANN": ("sann_history.csv", load_sann),
}
MODEL_ORDER = ["LR", "XGB", "SANN"]


def _load_all(base_dir):
    out = {}
    for name, (fname, loader) in LOADERS.items():
        path = os.path.join(base_dir, fname)
        if os.path.exists(path):
            out[name] = loader(path)
    return out


def _fmt_time_axis(ax):
    def _fmt(x, _pos):
        if x >= 60:
            return f"{x / 60:.0f}m"
        return f"{x:.0f}s"
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(_fmt))


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Load both reps
    reps = {}
    if os.path.isdir(args.hvg_dir):
        data = _load_all(args.hvg_dir)
        if data:
            reps["HVG"] = data
    if os.path.isdir(args.pca_dir):
        data = _load_all(args.pca_dir)
        if data:
            reps["PCA"] = data

    if not reps:
        raise FileNotFoundError("No history files found in either HVG or PCA directories.")

    rep_labels = [r for r in ["HVG", "PCA"] if r in reps]
    n_rows = len(rep_labels)

    # ================================================================
    # Figure 1: Validation Loss — all models overlaid
    # ================================================================
    fig_loss, axes_loss = plt.subplots(
        n_rows, 1, figsize=(10, 4.5 * n_rows),
        squeeze=False,
    )

    for row, rep in enumerate(rep_labels):
        ax = axes_loss[row, 0]
        # Use SANN's final time as the x-axis limit
        sann_max_t = None
        if "SANN" in reps[rep]:
            sann_max_t = reps[rep]["SANN"][0][-1]

        for model in MODEL_ORDER:
            if model in reps[rep]:
                t, vl, _ = reps[rep][model]
                vl_plot = moving_average(vl, args.smooth) if args.smooth > 1 else vl
                ax.plot(t, vl_plot, color=COLORS[model], linewidth=1.8,
                        label=model)

        if sann_max_t is not None:
            ax.set_xlim(0, sann_max_t * 1.02)

        ax.set_xlabel("Wall-clock time", fontsize=11)
        ax.set_ylabel("Validation Loss", fontsize=11)
        ax.set_title(f"{rep} Representation", fontsize=13, fontweight="bold")
        ax.legend(fontsize=11, frameon=False)
        ax.grid(True, alpha=0.3)
        _fmt_time_axis(ax)

    fig_loss.suptitle("Validation Loss — Convergence Curves",
                      fontsize=15, fontweight="bold", y=1.01)
    fig_loss.tight_layout()
    out_loss = os.path.join(args.outdir, "convergence_val_loss.png")
    fig_loss.savefig(out_loss, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig_loss)
    print(f"[Saved] {out_loss}")

    # ================================================================
    # Figure 2: Validation Macro-F1 — all models overlaid
    # ================================================================
    fig_f1, axes_f1 = plt.subplots(
        n_rows, 1, figsize=(10, 4.5 * n_rows),
        squeeze=False,
    )

    for row, rep in enumerate(rep_labels):
        ax = axes_f1[row, 0]
        sann_max_t = None
        if "SANN" in reps[rep]:
            sann_max_t = reps[rep]["SANN"][0][-1]

        for model in MODEL_ORDER:
            if model in reps[rep]:
                t, _, vf = reps[rep][model]
                vf_plot = moving_average(vf, args.smooth) if args.smooth > 1 else vf
                ax.plot(t, vf_plot, color=COLORS[model], linewidth=1.8,
                        label=model)

        if sann_max_t is not None:
            ax.set_xlim(0, sann_max_t * 1.02)

        ax.set_xlabel("Wall-clock time", fontsize=11)
        ax.set_ylabel("Validation Macro-F1", fontsize=11)
        ax.set_title(f"{rep} Representation", fontsize=13, fontweight="bold")
        ax.legend(fontsize=11, frameon=False)
        ax.grid(True, alpha=0.3)
        _fmt_time_axis(ax)

    fig_f1.suptitle("Validation Macro-F1 — Convergence Curves",
                    fontsize=15, fontweight="bold", y=1.01)
    fig_f1.tight_layout()
    out_f1 = os.path.join(args.outdir, "convergence_val_f1.png")
    fig_f1.savefig(out_f1, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig_f1)
    print(f"[Saved] {out_f1}")


if __name__ == "__main__":
    main()
