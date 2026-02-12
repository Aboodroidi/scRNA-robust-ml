# src/temp_scale_sann.py
import os
import json
import argparse
import time

# macOS-friendly
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MPLBACKEND"] = "Agg"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F


def parse_args():
    p = argparse.ArgumentParser(description="Temperature scaling for SANN using a calibration split from train.")
    p.add_argument("--splits", type=str, default="results/ablations/fixed_splits.json")
    p.add_argument("--y_true_full", type=str, default="data/processed/pbmc68k_labeled.h5ad",
                   help="Not used directly; kept for consistency. We use split indices only.")
    p.add_argument("--sann_probs", type=str, default="results/sann_test_probs.npy")
    p.add_argument("--sann_true", type=str, default="results/sann_test_true.npy")

    # IMPORTANT: need logits for calibration set too.
    # Best: provide sann_train_probs.npy + sann_train_true.npy (full train split order),
    # Otherwise: load full probs for all cells (recommended) via --sann_probs_all and --sann_true_all.
    p.add_argument("--sann_probs_all", type=str, default="",
                   help="Optional: probs for ALL cells in original adata order (N x C). Best option.")
    p.add_argument("--sann_true_all", type=str, default="",
                   help="Optional: true labels for ALL cells as indices in adata order (N,). Best option.")

    # If you don't have all-cells probs, but you have train probs saved separately:
    p.add_argument("--sann_train_probs", type=str, default="",
                   help="Optional: probs for TRAIN cells only (len(train_idx) x C).")
    p.add_argument("--sann_train_true", type=str, default="",
                   help="Optional: true labels for TRAIN cells only (len(train_idx),).")

    p.add_argument("--cal_frac", type=float, default=0.2, help="Fraction of TRAIN used for calibration fit.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bins", type=int, default=10)

    p.add_argument("--out_fig", type=str, default="results/figures/temp_scaling_sann.png")
    p.add_argument("--out_json", type=str, default="results/temp_scaling_sann.json")
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def probs_to_logits(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    probs = np.clip(probs, eps, 1.0)
    return np.log(probs)


def reliability_points(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10):
    probs = np.asarray(probs)
    y_true = np.asarray(y_true).astype(int)

    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(int)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(conf, bins) - 1
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

    return bin_conf, bin_acc, bin_counts, float(ece), pred, conf


def fit_temperature(logits_cal: np.ndarray, y_cal: np.ndarray, max_iter: int = 200):
    """
    Fit scalar temperature T>0 by minimising NLL on calibration set.
    Optimise log_T to enforce positivity.
    """
    device = "cpu"
    logits = torch.tensor(logits_cal, dtype=torch.float32, device=device)
    y = torch.tensor(y_cal, dtype=torch.long, device=device)

    log_T = torch.zeros((), dtype=torch.float32, requires_grad=True, device=device)

    optimizer = torch.optim.LBFGS([log_T], lr=0.5, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        T = torch.exp(log_T)
        scaled_logits = logits / T
        loss = F.cross_entropy(scaled_logits, y)
        loss.backward()
        return loss

    start = time.time()
    loss_before = float(F.cross_entropy(logits, y).item())
    optimizer.step(closure)
    elapsed = time.time() - start

    T = float(torch.exp(log_T).detach().cpu().item())
    loss_after = float(F.cross_entropy(logits / T, y).item())

    return T, loss_before, loss_after, elapsed


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_fig), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)

    # Load splits
    with open(args.splits, "r") as f:
        splits = json.load(f)

    if "train_idx" not in splits or "test_idx" not in splits:
        raise ValueError(f"fixed_splits.json must contain train_idx and test_idx. Found keys: {list(splits.keys())}")

    train_idx = np.array(splits["train_idx"], dtype=int)
    test_idx = np.array(splits["test_idx"], dtype=int)

    # Load test outputs (already saved by eval_sann.py)
    probs_test = np.load(args.sann_probs)
    y_test = np.load(args.sann_true).astype(int)

    if probs_test.shape[0] != len(y_test):
        raise ValueError("Mismatch: sann_test_probs rows must match sann_test_true length.")
    if not np.allclose(probs_test.sum(axis=1), 1.0, atol=1e-3):
        print("[Warning] Test probs rows do not sum to 1 (within atol=1e-3). Proceeding anyway.")

    n_classes = probs_test.shape[1]
    print(f"[Sanity] Test set: N={len(y_test)}, C={n_classes}")

    # ----------------------------
    # Get calibration probs/logits from TRAIN (NOT test)
    # We support 2 ways:
    #  A) probs_all + true_all for all cells in adata order (best)
    #  B) train_probs + train_true saved aligned to train_idx (ok)
    # ----------------------------
    probs_train_full = None
    y_train_full = None

    if args.sann_probs_all and args.sann_true_all:
        probs_all = np.load(args.sann_probs_all)
        y_all = np.load(args.sann_true_all).astype(int)

        if probs_all.shape[0] != len(y_all):
            raise ValueError("Mismatch: sann_probs_all rows must match sann_true_all length.")
        if probs_all.shape[1] != n_classes:
            raise ValueError("Mismatch: sann_probs_all class count must match test probs.")
        probs_train_full = probs_all[train_idx]
        y_train_full = y_all[train_idx]
        print("[Sanity] Using all-cells SANN probs/true to build calibration set from train_idx.")

    elif args.sann_train_probs and args.sann_train_true:
        probs_train_full = np.load(args.sann_train_probs)
        y_train_full = np.load(args.sann_train_true).astype(int)

        if probs_train_full.shape[0] != len(y_train_full):
            raise ValueError("Mismatch: sann_train_probs rows must match sann_train_true length.")
        if probs_train_full.shape[1] != n_classes:
            raise ValueError("Mismatch: sann_train_probs class count must match test probs.")
        if probs_train_full.shape[0] != len(train_idx):
            print("[Warning] sann_train_probs length does not equal len(train_idx). "
                  "This is OK only if your saved train probs correspond exactly to the train split you used.")
        print("[Sanity] Using saved train-only SANN probs/true to build calibration set.")

    else:
        raise ValueError(
            "You must provide calibration data from TRAIN.\n"
            "Provide either:\n"
            "  --sann_probs_all results/sann_all_probs.npy --sann_true_all results/sann_all_true.npy\n"
            "OR\n"
            "  --sann_train_probs results/sann_train_probs.npy --sann_train_true results/sann_train_true.npy\n"
            "Because fitting T on test set would be incorrect."
        )

    # Build calibration subset from train
    rng = np.random.RandomState(args.seed)
    n_train = len(y_train_full)
    n_cal = int(round(args.cal_frac * n_train))
    if n_cal < 100:
        print(f"[Warning] calibration set very small (n_cal={n_cal}). Consider cal_frac=0.2.")
    cal_perm = rng.permutation(n_train)
    cal_sel = cal_perm[:n_cal]

    probs_cal = probs_train_full[cal_sel]
    y_cal = y_train_full[cal_sel]

    print(f"[Sanity] Train set: N={n_train} | Calibration subset: n_cal={len(y_cal)} (cal_frac={args.cal_frac})")

    # Convert probs -> logits (cal + test)
    logits_cal = probs_to_logits(probs_cal)
    logits_test = probs_to_logits(probs_test)

    # Fit temperature on calibration set
    T, nll_before, nll_after, fit_time = fit_temperature(logits_cal, y_cal, max_iter=200)
    print(f"[Sanity] Fitted T = {T:.4f}")
    print(f"[Sanity] Calibration NLL: before={nll_before:.4f} after={nll_after:.4f} | fit_time={fit_time:.2f}s")

    # Apply scaling on test set
    probs_test_before = softmax_np(logits_test)
    probs_test_after = softmax_np(logits_test / T)

    # Reliability + ECE
    x_b, y_b, _, ece_before, pred_b, conf_b = reliability_points(probs_test_before, y_test, n_bins=args.bins)
    x_a, y_a, _, ece_after,  pred_a, conf_a = reliability_points(probs_test_after,  y_test, n_bins=args.bins)

    # Sanity: accuracy should be (almost) unchanged
    acc_before = float((pred_b == y_test).mean())
    acc_after = float((pred_a == y_test).mean())
    print(f"[Sanity] Test accuracy: before={acc_before:.4f} after={acc_after:.4f}")
    print(f"[Sanity] Test ECE:      before={ece_before:.4f} after={ece_after:.4f}")

    # Plot figure (two curves)
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)

    mb = ~np.isnan(x_b) & ~np.isnan(y_b)
    ma = ~np.isnan(x_a) & ~np.isnan(y_a)

    ax.plot(x_b[mb], y_b[mb], marker="o", linewidth=1.5, label=f"Before (ECE={ece_before:.3f})")
    ax.plot(x_a[ma], y_a[ma], marker="o", linewidth=1.5, label=f"After (ECE={ece_after:.3f})")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Observed accuracy")
    ax.set_title("Temperature Scaling Effect on SANN Calibration")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(args.out_fig, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    # Save summary JSON
    out = {
        "temperature_T": T,
        "calibration_nll_before": nll_before,
        "calibration_nll_after": nll_after,
        "fit_time_seconds": fit_time,
        "test_accuracy_before": acc_before,
        "test_accuracy_after": acc_after,
        "test_ece_before": ece_before,
        "test_ece_after": ece_after,
        "bins": args.bins,
        "cal_frac": args.cal_frac,
        "seed": args.seed,
    }
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)

    print(f"[Sanity] Saved figure: {args.out_fig}")
    print(f"[Sanity] Saved summary: {args.out_json}")
    print("[Note] For dissertation: report T and ECE_before vs ECE_after.")


if __name__ == "__main__":
    main()