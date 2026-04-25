import os
import json
import argparse

# Mac-friendly + non-GUI plotting
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MPLBACKEND"] = "Agg"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot reliability diagrams (LR, XGB, SANN) side-by-side using REAL bins (quantile by default)."
    )
    p.add_argument("--data", type=str, default="data/processed/pbmc68k_labeled.h5ad")
    p.add_argument("--label_key", type=str, default="cell_type")
    p.add_argument("--splits", type=str, default="results/ablations/fixed_splits.json")

    # feature representation (match SANN)
    p.add_argument("--use_pca", action="store_true", default=True)
    p.add_argument("--pca_key", type=str, default="X_pca")
    p.add_argument("--pca_dim", type=int, default=50)

    # model artifacts / arrays
    p.add_argument("--sann_probs", type=str, required=True)
    p.add_argument("--sann_true", type=str, required=True)

    p.add_argument("--lr_model", type=str, required=True)
    p.add_argument("--lr_scaler", type=str, required=True)

    p.add_argument("--xgb_json", type=str, required=True)

    # output
    p.add_argument("--out", type=str, default="results/figures/reliability_all_models.png")
    p.add_argument("--bins", type=int, default=10)
    p.add_argument("--dpi", type=int, default=300)

    # binning strategy
    # - quantile: equal-count bins => NO empty bins (recommended for clean curve)
    # - uniform: fixed bins [0,1] => may have empty bins; we will NOT fabricate values
    p.add_argument("--binning", type=str, default="quantile", choices=["quantile", "uniform"])

    # save recomputed probs/preds/true for LR/XGB into this directory
    p.add_argument("--save_dir", type=str, default="results/full_train")
    return p.parse_args()


def ensure_dir_for_file(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def load_test_idx(path: str) -> np.ndarray:
    with open(path, "r") as f:
        d = json.load(f)
    if "test_idx" not in d:
        raise ValueError(f"Split file must contain 'test_idx'. Found keys: {list(d.keys())}")
    return np.array(d["test_idx"], dtype=int)


def get_X_y(adata, label_key: str, use_pca: bool, pca_key: str, pca_dim: int):
    if label_key not in adata.obs:
        raise ValueError(f"Expected adata.obs['{label_key}'] to exist.")

    y_cat = adata.obs[label_key].astype("category")
    y = y_cat.cat.codes.to_numpy().astype(int)
    class_names = list(y_cat.cat.categories)

    if use_pca:
        if pca_key not in adata.obsm:
            raise ValueError(f"Expected adata.obsm['{pca_key}'] for PCA features.")
        X = adata.obsm[pca_key]
        if pca_dim is not None:
            X = X[:, :pca_dim]
        X = np.asarray(X, dtype=np.float32)
    else:
        X = np.asarray(adata.X, dtype=np.float32)

    return X, y, class_names


def compute_reliability(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10, binning: str = "quantile"):
    """
    Real reliability computation.

    Returns:
      x: mean confidence per bin
      y: observed accuracy per bin
      counts: samples per bin
      ece: Expected Calibration Error (computed on bins with data)
      bin_edges: used bin edges (for debugging/reporting)
    """
    probs = np.asarray(probs, dtype=float)
    y_true = np.asarray(y_true).astype(int)

    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(int)

    if binning == "uniform":
        # Fixed bins in [0,1]
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        # Quantile bins => roughly equal number of samples per bin (avoids empty bins)
        # Use unique edges to avoid issues when many conf values are identical.
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        bin_edges = np.quantile(conf, qs)
        # Make edges strictly increasing (handle duplicates)
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 3:
            # extreme edge case: almost all confidences identical
            # fall back to uniform
            bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    # Assign bins
    # right=True makes bins (edge[i-1], edge[i]] which is usually stable with quantiles
    bin_ids = np.digitize(conf, bin_edges, right=True) - 1
    n_eff_bins = len(bin_edges) - 1
    bin_ids = np.clip(bin_ids, 0, n_eff_bins - 1)

    x = np.full(n_eff_bins, np.nan, dtype=float)
    y = np.full(n_eff_bins, np.nan, dtype=float)
    counts = np.zeros(n_eff_bins, dtype=int)

    for b in range(n_eff_bins):
        mask = (bin_ids == b)
        cnt = int(mask.sum())
        counts[b] = cnt
        if cnt > 0:
            x[b] = float(conf[mask].mean())
            y[b] = float(correct[mask].mean())

    # ECE (only bins with data)
    n_total = len(y_true)
    ece = 0.0
    for b in range(n_eff_bins):
        if counts[b] > 0:
            w = counts[b] / n_total
            ece += w * abs(y[b] - x[b])

    return x, y, counts, float(ece), bin_edges


MODEL_COLORS = {
    "LR":   "#1b9e77",
    "XGB":  "#d95f02",
    "SANN": "#7570b3",
}


COLOR_UNDERCONF = "#FFB6C1"   # pink — accuracy > confidence
COLOR_OVERCONF  = "#B0B0FF"   # light purple/blue — confidence > accuracy


def _shade_calibration_gap(ax, x_pts, y_pts):
    """Shade underconfidence pink, overconfidence purple between curve and diagonal."""
    x_pts = np.asarray(x_pts)
    y_pts = np.asarray(y_pts)

    # Underconfidence: observed accuracy > predicted confidence
    ax.fill_between(
        x_pts, x_pts, y_pts,
        where=(y_pts >= x_pts),
        interpolate=True,
        color=COLOR_UNDERCONF, alpha=0.45,
        label="Underconfidence",
    )
    # Overconfidence: predicted confidence > observed accuracy
    ax.fill_between(
        x_pts, x_pts, y_pts,
        where=(y_pts < x_pts),
        interpolate=True,
        color=COLOR_OVERCONF, alpha=0.45,
        label="Overconfidence",
    )


def plot_reliability_single(ax, probs, y_true, title, color, n_bins=10, binning="quantile"):
    """Single-model reliability diagram with under/overconfidence shading."""
    x, y, counts, ece, bin_edges = compute_reliability(probs, y_true, n_bins=n_bins, binning=binning)
    mask = counts > 0
    xm, ym = x[mask], y[mask]

    # Shaded gap regions
    _shade_calibration_gap(ax, xm, ym)

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect Calibration")

    # Calibration curve
    ax.plot(xm, ym, "o-", color="black", linewidth=2, markersize=7,
            label=f"Calibration Line (ECE={ece:.3f})")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")

    return ece


def plot_reliability_overlay(ax, model_data, n_bins=10, binning="quantile"):
    """Overlay all models on one reliability diagram."""
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect Calibration")

    for name, probs, y_true in model_data:
        color = MODEL_COLORS.get(name, "grey")
        x, y, counts, ece, _ = compute_reliability(probs, y_true, n_bins=n_bins, binning=binning)
        mask = counts > 0
        ax.plot(x[mask], y[mask], "o-", color=color, linewidth=2, markersize=6,
                label=f"{name} (ECE={ece:.3f})")
        # Shaded gap
        ax.fill_between(x[mask], x[mask], y[mask], color=color, alpha=0.12)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("Predicted Probability", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.legend(loc="upper left", fontsize=10, frameon=True, framealpha=0.9)
    ax.grid(True, alpha=0.2)


def main():
    args = parse_args()
    ensure_dir_for_file(args.out)
    os.makedirs(args.save_dir, exist_ok=True)

    adata = sc.read_h5ad(args.data)
    X, y, class_names = get_X_y(
        adata,
        label_key=args.label_key,
        use_pca=args.use_pca,
        pca_key=args.pca_key,
        pca_dim=args.pca_dim,
    )

    test_idx = load_test_idx(args.splits)
    X_test = X[test_idx]
    y_test = y[test_idx]

    print(f"[Sanity] X={X.shape} | X_test={X_test.shape}")
    print(f"[Sanity] y={len(y)} | y_test={len(y_test)} | classes={len(class_names)}")
    print(f"[Sanity] class names: {class_names}")
    print(f"[Sanity] test_idx length: {len(test_idx)}")
    print(f"[Sanity] binning strategy: {args.binning}")

    # SANN arrays
    sann_probs = np.load(args.sann_probs)
    sann_true = np.load(args.sann_true).astype(int)
    if sann_probs.shape[0] != len(test_idx) or len(sann_true) != len(test_idx):
        raise ValueError(
            f"SANN arrays must match test size. len(test_idx)={len(test_idx)}, "
            f"sann_probs={sann_probs.shape}, sann_true={len(sann_true)}"
        )

    # LR artifacts + probs
    if not (os.path.exists(args.lr_model) and os.path.exists(args.lr_scaler)):
        raise FileNotFoundError(f"LR artifacts not found: {args.lr_model}, {args.lr_scaler}")
    lr = joblib.load(args.lr_model)
    scaler = joblib.load(args.lr_scaler)

    exp = getattr(scaler, "n_features_in_", None)
    if exp is not None and exp != X_test.shape[1]:
        raise ValueError(f"LR scaler expects {exp} features, but X_test has {X_test.shape[1]}.")
    X_test_lr = scaler.transform(X_test)
    lr_probs = lr.predict_proba(X_test_lr)

    # XGB model + probs
    if not os.path.exists(args.xgb_json):
        raise FileNotFoundError(f"XGB model JSON not found: {args.xgb_json}")
    xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=len(class_names),
        n_jobs=1,
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=42,
    )
    xgb.load_model(args.xgb_json)
    xgb_probs = xgb.predict_proba(X_test)

    # Save LR/XGB arrays for later calibration/error analysis
    np.save(os.path.join(args.save_dir, "lr_test_probs.npy"), lr_probs)
    np.save(os.path.join(args.save_dir, "lr_test_true.npy"), y_test)
    np.save(os.path.join(args.save_dir, "lr_test_pred.npy"), lr_probs.argmax(axis=1))

    np.save(os.path.join(args.save_dir, "xgb_test_probs.npy"), xgb_probs)
    np.save(os.path.join(args.save_dir, "xgb_test_true.npy"), y_test)
    np.save(os.path.join(args.save_dir, "xgb_test_pred.npy"), xgb_probs.argmax(axis=1))

    print(f"[Sanity] Saved LR test arrays to {args.save_dir}/lr_test_*.npy")
    print(f"[Sanity] Saved XGB test arrays to {args.save_dir}/xgb_test_*.npy")

    # Sanity metrics
    print(f"[Sanity] LR  acc={accuracy_score(y_test, lr_probs.argmax(1)):.4f} macroF1={f1_score(y_test, lr_probs.argmax(1), average='macro'):.4f}")
    print(f"[Sanity] XGB acc={accuracy_score(y_test, xgb_probs.argmax(1)):.4f} macroF1={f1_score(y_test, xgb_probs.argmax(1), average='macro'):.4f}")
    print(f"[Sanity] SANN acc={accuracy_score(sann_true, sann_probs.argmax(1)):.4f} macroF1={f1_score(sann_true, sann_probs.argmax(1), average='macro'):.4f}")

    # ================================================================
    # PRIMARY FIGURE: all three models overlaid on one reliability diagram
    # ================================================================
    fig_overlay, ax_overlay = plt.subplots(figsize=(7, 7))

    model_data = [
        ("LR",   lr_probs,   y_test),
        ("XGB",  xgb_probs,  y_test),
        ("SANN", sann_probs, sann_true),
    ]
    plot_reliability_overlay(ax_overlay, model_data, n_bins=args.bins, binning=args.binning)
    ax_overlay.set_title(f"Reliability Diagram — All Models\n({args.bins} {args.binning} bins, test set)",
                         fontsize=13, fontweight="bold")

    fig_overlay.tight_layout()
    out_overlay = args.out
    fig_overlay.savefig(out_overlay, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig_overlay)
    print(f"[Saved] {out_overlay}")

    # ================================================================
    # SECONDARY FIGURE: per-model subplots with under/overconfidence shading
    # ================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for ax_i, (name, probs, y_true_i) in zip(axes, model_data):
        color = MODEL_COLORS[name]
        plot_reliability_single(ax_i, probs, y_true_i, name, color,
                                n_bins=args.bins, binning=args.binning)
        ax_i.set_title(name, fontsize=13, fontweight="bold")
        ax_i.set_xlabel("Predicted Probability", fontsize=10)
        ax_i.set_ylabel("Accuracy", fontsize=10)
        ax_i.legend(loc="upper left", fontsize=9, frameon=True, framealpha=0.9)
        ax_i.grid(True, alpha=0.2)

    fig.suptitle(f"Reliability Diagrams ({args.bins} {args.binning} bins)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    out_per_model = args.out.replace(".png", "_per_model.png")
    fig.savefig(out_per_model, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {out_per_model}")


if __name__ == "__main__":
    main()