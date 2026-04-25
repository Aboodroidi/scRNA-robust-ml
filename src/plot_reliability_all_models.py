import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Mac-friendly + non-GUI plotting
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MPLBACKEND"] = "Agg"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def parse_args():
    p = argparse.ArgumentParser(description="Plot reliability diagrams from saved test probs (LR, XGB, SANN).")
    p.add_argument("--model_dir", type=str, default="results/full_train_all_pca_coarse",
                   help="Directory with saved models and probs")
    p.add_argument("--data", type=str, default="data/processed/pbmc68k_labeled.h5ad")
    p.add_argument("--pca_key", type=str, default="X_pca")
    p.add_argument("--pca_dim", type=int, default=50)
    p.add_argument("--label_key", type=str, default="cell_type")
    p.add_argument("--out", type=str, default="results/figures/reliability_all_models.png")
    p.add_argument("--bins", type=int, default=10)
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def compute_reliability(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10):
    """
    Multiclass reliability using max softmax confidence.
    Uses quantile bins to match the original reliability diagrams.
    """
    probs = np.asarray(probs)
    y_true = np.asarray(y_true).astype(int)

    if probs.ndim != 2:
        raise ValueError(f"probs must be 2D (N,C). Got shape {probs.shape}")
    if probs.shape[0] != len(y_true):
        raise ValueError(f"Mismatch N: probs={probs.shape[0]} vs y_true={len(y_true)}")

    row_sums = probs.sum(axis=1)
    if not np.all(np.isfinite(row_sums)):
        raise ValueError("probs contains non-finite values.")
    if np.max(np.abs(row_sums - 1.0)) > 1e-2:
        print(f"[Warning] probs rows not summing to 1 (max |sum-1|={np.max(np.abs(row_sums-1)):.3e})")

    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(int)

    # Quantile bins (same as original plot)
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    bin_edges = np.quantile(conf, qs)
    bin_edges = np.unique(bin_edges)
    n_eff = len(bin_edges) - 1

    bin_ids = np.digitize(conf, bin_edges, right=True) - 1
    bin_ids = np.clip(bin_ids, 0, n_eff - 1)

    bin_acc = np.zeros(n_eff, dtype=float)
    bin_conf = np.zeros(n_eff, dtype=float)
    bin_counts = np.zeros(n_eff, dtype=int)

    for b in range(n_eff):
        mask = (bin_ids == b)
        cnt = int(mask.sum())
        bin_counts[b] = cnt
        if cnt > 0:
            bin_acc[b] = float(correct[mask].mean())
            bin_conf[b] = float(conf[mask].mean())

    # ECE
    n_total = len(y_true)
    ece = 0.0
    for b in range(n_eff):
        if bin_counts[b] > 0:
            w = bin_counts[b] / n_total
            ece += w * abs(bin_acc[b] - bin_conf[b])

    return bin_conf, bin_acc, bin_counts, float(ece)


def plot_reliability(ax, probs, y_true, title, n_bins=10):
    bin_conf, bin_acc, bin_counts, ece = compute_reliability(probs, y_true, n_bins=n_bins)

    # Only plot bins with data and non-zero accuracy
    mask = (bin_counts > 0) & (bin_acc > 0)
    x_plot = bin_conf[mask]
    y_plot = bin_acc[mask]

    # Perfect calibration
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect Calibration")

    # Shade overconfidence and underconfidence with darker colours
    # Need to interpolate to fill correctly between points
    if len(x_plot) > 1:
        diag = x_plot
        # Fill segment by segment to avoid fill_between interpolation issues
        for i in range(len(x_plot) - 1):
            x_seg = [x_plot[i], x_plot[i+1]]
            y_seg = [y_plot[i], y_plot[i+1]]
            d_seg = [diag[i], diag[i+1]]
            # Check if both points are on same side
            if y_seg[0] >= d_seg[0] and y_seg[1] >= d_seg[1]:
                # Underconfidence (model accuracy > confidence)
                ax.fill_between(x_seg, y_seg, d_seg,
                                color="#ff8888", alpha=0.5)
            elif y_seg[0] <= d_seg[0] and y_seg[1] <= d_seg[1]:
                # Overconfidence (model confidence > accuracy)
                ax.fill_between(x_seg, y_seg, d_seg,
                                color="#9b72cf", alpha=0.5)
            else:
                # Crossing — split at intersection
                ax.fill_between(x_seg, y_seg, d_seg,
                                where=[y >= d for y, d in zip(y_seg, d_seg)],
                                color="#ff8888", alpha=0.5, interpolate=True)
                ax.fill_between(x_seg, y_seg, d_seg,
                                where=[y <= d for y, d in zip(y_seg, d_seg)],
                                color="#9b72cf", alpha=0.5, interpolate=True)

    # Add legend patches for shading
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#ff8888", alpha=0.5, label="Underconfidence"),
        Patch(facecolor="#9b72cf", alpha=0.5, label="Overconfidence"),
        plt.Line2D([0], [0], color="k", linestyle="--", linewidth=1.5, label="Perfect Calibration"),
        plt.Line2D([0], [0], color="black", marker="o", linewidth=2, markersize=6,
                   label=f"Calibration Line (ECE={ece:.3f})"),
    ], loc="upper left", fontsize=7.5, frameon=True, framealpha=0.9)

    # Calibration curve
    ax.plot(x_plot, y_plot, "o-", color="black", linewidth=2, markersize=6)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("Predicted Probability", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.grid(True, alpha=0.2)
    return ece


def softmax_np(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def compute_sann_ensemble_probs(model_dir, data_path, pca_key, pca_dim, label_key):
    """Load SANN seed models, compute ensemble test probs."""
    import glob as glob_module
    import scanpy as sc
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.decomposition import PCA as PCASK

    seed_files = sorted(glob_module.glob(os.path.join(model_dir, "sann_model_seed*.pt")))
    if len(seed_files) <= 1:
        # Single model — use saved probs
        probs = np.load(os.path.join(model_dir, "sann_test_probs.npy"))
        true = np.load(os.path.join(model_dir, "sann_test_true.npy")).astype(int)
        return probs, true

    print(f"  Deep ensemble: found {len(seed_files)} seed models")

    is_pca = "pca" in model_dir.lower()

    adata = sc.read_h5ad(data_path)
    le = LabelEncoder()
    y_all = le.fit_transform(adata.obs[label_key].values.astype(str))
    num_classes = len(le.classes_)

    # Reconstruct split
    import json
    with open(os.path.join(model_dir, "train_val_test_split.json")) as f:
        split_info = json.load(f)

    n_total = len(y_all)
    test_frac = split_info["test_size"] / n_total
    val_frac = split_info.get("val_frac", 0.1)

    indices = np.arange(n_total)
    train_full_idx, test_idx = train_test_split(
        indices, test_size=test_frac, random_state=42, stratify=y_all)
    train_idx, val_idx = train_test_split(
        train_full_idx, test_size=val_frac, random_state=42, stratify=y_all[train_full_idx])

    if is_pca:
        X_pca = np.asarray(adata.obsm[pca_key][:, :pca_dim], dtype=np.float32)
        X_raw = np.array(adata.X, dtype=np.float32)
        X_mask_raw = (X_raw != 0).astype(np.float32)
        mask_pca = PCASK(n_components=pca_dim, random_state=42)
        mask_pca.fit(X_mask_raw[train_idx])
        X_mask_pca = mask_pca.transform(X_mask_raw).astype(np.float32)
        X_full = np.concatenate([X_pca, X_mask_pca], axis=1)
        n_expr = pca_dim
    else:
        X_raw = np.array(adata.X, dtype=np.float32)
        X_mask = (X_raw != 0).astype(np.float32)
        X_full = np.concatenate([X_raw, X_mask], axis=1)
        n_expr = X_raw.shape[1]

    X_test = X_full[test_idx]
    y_test = y_all[test_idx]

    # Build SANN architecture
    def _make_norm(dim, use_batchnorm, use_layernorm):
        if use_layernorm: return nn.LayerNorm(dim)
        if use_batchnorm: return nn.BatchNorm1d(dim)
        return nn.Identity()

    class ResidualBlock(nn.Module):
        def __init__(self, dim, dropout=0.1, use_batchnorm=False, use_layernorm=True):
            super().__init__()
            self.norm = _make_norm(dim, use_batchnorm, use_layernorm)
            self.fc1 = nn.Linear(dim, dim)
            self.fc2 = nn.Linear(dim, dim)
            self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            self.act = nn.GELU()
        def forward(self, x):
            h = self.norm(x)
            h = self.act(self.fc1(h))
            h = self.drop(h)
            h = self.fc2(h)
            return x + h

    class SANN(nn.Module):
        def __init__(self, expr_dim, mask_dim, num_classes,
                     branch_hidden=512, branch_out=256, fusion_hidden=256,
                     dropout=0.05, use_batchnorm=True, use_layernorm=False,
                     input_noise=0.0):
            super().__init__()
            self.expr_dim = expr_dim
            self.input_noise = input_noise
            self.expr_proj = nn.Sequential(
                nn.Linear(expr_dim, branch_hidden),
                _make_norm(branch_hidden, use_batchnorm, use_layernorm), nn.GELU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity())
            self.expr_res1 = ResidualBlock(branch_hidden, dropout, use_batchnorm, use_layernorm)
            self.expr_res2 = ResidualBlock(branch_hidden, dropout, use_batchnorm, use_layernorm)
            self.expr_out = nn.Sequential(
                nn.Linear(branch_hidden, branch_out),
                _make_norm(branch_out, use_batchnorm, use_layernorm), nn.GELU())
            self.mask_proj = nn.Sequential(
                nn.Linear(mask_dim, branch_hidden),
                _make_norm(branch_hidden, use_batchnorm, use_layernorm), nn.GELU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity())
            self.mask_res1 = ResidualBlock(branch_hidden, dropout, use_batchnorm, use_layernorm)
            self.mask_res2 = ResidualBlock(branch_hidden, dropout, use_batchnorm, use_layernorm)
            self.mask_out = nn.Sequential(
                nn.Linear(branch_hidden, branch_out),
                _make_norm(branch_out, use_batchnorm, use_layernorm), nn.GELU())
            self.gate = nn.Sequential(
                nn.Linear(branch_out * 2, branch_out), nn.GELU(),
                nn.Linear(branch_out, branch_out), nn.Sigmoid())
            self.classifier = nn.Sequential(
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(branch_out, fusion_hidden),
                _make_norm(fusion_hidden, use_batchnorm, use_layernorm), nn.GELU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(fusion_hidden, num_classes))
        def forward(self, x):
            x_expr = x[:, :self.expr_dim]
            x_mask = x[:, self.expr_dim:]
            if self.training and self.input_noise > 0:
                x_expr = x_expr + torch.randn_like(x_expr) * self.input_noise
            h_expr = self.expr_proj(x_expr)
            h_expr = self.expr_res1(h_expr)
            h_expr = self.expr_res2(h_expr)
            h_expr = self.expr_out(h_expr)
            h_mask = self.mask_proj(x_mask)
            h_mask = self.mask_res1(h_mask)
            h_mask = self.mask_res2(h_mask)
            h_mask = self.mask_out(h_mask)
            combined = torch.cat([h_expr, h_mask], dim=1)
            alpha = self.gate(combined)
            h_fused = alpha * h_expr + (1 - alpha) * h_mask
            return self.classifier(h_fused)

    n_mask = X_full.shape[1] - n_expr
    if is_pca:
        bh, bo, fh, drop = 512, 256, 256, 0.25
        use_bn, use_ln = True, False
    else:
        bh, bo, fh, drop = 256, 128, 128, 0.4
        use_bn, use_ln = False, True

    def make_model():
        return SANN(n_expr, n_mask, num_classes,
                    branch_hidden=bh, branch_out=bo, fusion_hidden=fh,
                    dropout=drop, use_batchnorm=use_bn, use_layernorm=use_ln)

    all_test_probs = []
    for sf in seed_files:
        m = make_model()
        m.load_state_dict(torch.load(sf, map_location="cpu"))
        m.eval()
        with torch.no_grad():
            logits = m(torch.tensor(X_test, dtype=torch.float32)).numpy()
        all_test_probs.append(softmax_np(logits))
        print(f"    Loaded {os.path.basename(sf)}")

    ensemble_probs = np.mean(all_test_probs, axis=0)
    return ensemble_probs, y_test


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    model_dir = args.model_dir
    is_pca = "pca" in model_dir.lower()
    rep_label = "PCA" if is_pca else "HVG"

    lr_probs = np.load(os.path.join(model_dir, "lr_test_probs.npy"))
    lr_true = np.load(os.path.join(model_dir, "lr_test_true.npy")).astype(int)

    xgb_probs = np.load(os.path.join(model_dir, "xgb_test_probs.npy"))
    xgb_true = np.load(os.path.join(model_dir, "xgb_test_true.npy")).astype(int)

    sann_probs = np.load(os.path.join(model_dir, "sann_test_probs.npy"))
    sann_true = np.load(os.path.join(model_dir, "sann_test_true.npy")).astype(int)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True, sharey=True)

    ece_lr = plot_reliability(axes[0], lr_probs, lr_true, "LR", n_bins=args.bins)
    ece_xgb = plot_reliability(axes[1], xgb_probs, xgb_true, "XGB", n_bins=args.bins)
    ece_sann = plot_reliability(axes[2], sann_probs, sann_true, "SANN", n_bins=args.bins)

    fig.suptitle(f"Reliability Diagrams — {rep_label} (10 quantile bins)", y=1.02, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[Saved] {args.out}")
    print(f"  LR ECE:   {ece_lr:.4f}")
    print(f"  XGB ECE:  {ece_xgb:.4f}")
    print(f"  SANN ECE: {ece_sann:.4f}")


if __name__ == "__main__":
    main()