"""
Temperature scaling for LR, XGB, and SANN — side-by-side reliability diagrams.
Uses saved test/val probs for SANN; generates val probs for LR/XGB on the fly.
"""
import os
import json
import argparse

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
import joblib


def parse_args():
    p = argparse.ArgumentParser(description="Temperature scaling for all 3 models, side-by-side reliability diagrams.")
    p.add_argument("--model_dir", type=str, default="results/full_train_all_pca_coarse",
                   help="Directory with saved models and probs")
    p.add_argument("--data", type=str, default="data/processed/pbmc68k_labeled.h5ad")
    p.add_argument("--pca_key", type=str, default="X_pca")
    p.add_argument("--pca_dim", type=int, default=50)
    p.add_argument("--label_key", type=str, default="cell_type")
    p.add_argument("--bins", type=int, default=10)
    p.add_argument("--out", type=str, default="results/figures/temp_scaling_all_models.png")
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def softmax_np(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def probs_to_logits(probs, eps=1e-12):
    return np.log(np.clip(probs, eps, 1.0))


def reliability_points(probs, y_true, n_bins=10):
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(int)

    qs = np.linspace(0.0, 1.0, n_bins + 1)
    bin_edges = np.quantile(conf, qs)
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 3:
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    bin_ids = np.digitize(conf, bin_edges, right=True) - 1
    n_eff = len(bin_edges) - 1
    bin_ids = np.clip(bin_ids, 0, n_eff - 1)

    bin_conf = np.full(n_eff, np.nan)
    bin_acc = np.full(n_eff, np.nan)
    bin_counts = np.zeros(n_eff, dtype=int)

    for b in range(n_eff):
        m = bin_ids == b
        cnt = int(m.sum())
        bin_counts[b] = cnt
        if cnt > 0:
            bin_conf[b] = conf[m].mean()
            bin_acc[b] = correct[m].mean()

    n_total = len(y_true)
    ece = sum(
        (bin_counts[b] / n_total) * abs(bin_acc[b] - bin_conf[b])
        for b in range(n_eff) if bin_counts[b] > 0
    )
    return bin_conf, bin_acc, bin_counts, float(ece)


def fit_temperature(logits_cal, y_cal, max_iter=200, n_bins=10):
    logits = torch.tensor(logits_cal, dtype=torch.float32)
    y = torch.tensor(y_cal, dtype=torch.long)

    # Step 1: LBFGS on NLL to get a good starting point
    log_T = torch.zeros((), dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_T], lr=0.5, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / torch.exp(log_T), y)
        loss.backward()
        return loss

    optimizer.step(closure)
    T_nll = float(torch.exp(log_T).detach().item())

    # Step 2: Fine grid search around NLL optimum to directly minimize ECE
    y_np = y.numpy()
    best_T = T_nll
    best_ece = float("inf")

    # Search a very wide range to find true ECE minimum
    for T_cand in np.linspace(0.1, 3.0, 10000):
        scaled_logits = logits_cal / T_cand
        probs = torch.softmax(torch.tensor(scaled_logits, dtype=torch.float32), dim=1).numpy()
        _, _, _, ece = reliability_points(probs, y_np, n_bins)
        if ece < best_ece:
            best_ece = ece
            best_T = T_cand

    return best_T


def plot_single_model(ax, probs_before, probs_after, y_true, T, model_name, n_bins=10):
    bc_b, ba_b, cnt_b, ece_b = reliability_points(probs_before, y_true, n_bins)
    bc_a, ba_a, cnt_a, ece_a = reliability_points(probs_after, y_true, n_bins)

    mask_b = (cnt_b > 0) & (ba_b > 0)
    mask_a = (cnt_a > 0) & (ba_a > 0)
    xb, yb = bc_b[mask_b], ba_b[mask_b]
    xa, ya = bc_a[mask_a], ba_a[mask_a]

    # Shade difference (temperature scaling effect)
    if len(xb) > 1 and len(xa) > 1:
        x_common = np.linspace(max(xb.min(), xa.min()), min(xb.max(), xa.max()), 200)
        yb_interp = np.interp(x_common, xb, yb)
        ya_interp = np.interp(x_common, xa, ya)
        ax.fill_between(x_common, yb_interp, ya_interp,
                        color="#DAA520", alpha=0.50, label="Temperature scaling effect")

    # Perfect calibration
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect Calibration")

    # Before — darker red
    ax.plot(xb, yb, "o-", color="#a31919", linewidth=2, markersize=6,
            label=f"Before (ECE={ece_b:.3f})")

    # After — darker green
    ax.plot(xa, ya, "s-", color="#1a7a1a", linewidth=2, markersize=6,
            label=f"After T={T:.3f} (ECE={ece_a:.3f})")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("Predicted Probability", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title(model_name, fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8, frameon=True, framealpha=0.9)
    ax.grid(True, alpha=0.2)


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    model_dir = args.model_dir

    # Determine if PCA or HVG
    is_pca = "pca" in model_dir.lower()

    # ══════════════════════════════════════════
    # For LR and XGB: no saved val probs, so we split the test set
    # into cal (30%) and eval (70%) for temperature fitting.
    # For SANN: use the saved val probs for calibration.
    # ══════════════════════════════════════════

    # ══════════════════════════════════════════
    # LR
    # ══════════════════════════════════════════
    print("Processing LR...")
    lr_test_probs = np.load(os.path.join(model_dir, "lr_test_probs.npy"))
    lr_test_true = np.load(os.path.join(model_dir, "lr_test_true.npy")).astype(int)

    # Split test into cal/eval for fitting T
    n_lr = len(lr_test_true)
    rng = np.random.RandomState(42)
    perm_lr = rng.permutation(n_lr)
    n_cal_lr = int(0.3 * n_lr)
    cal_lr = perm_lr[:n_cal_lr]
    eval_lr = perm_lr[n_cal_lr:]

    lr_logits_cal = probs_to_logits(lr_test_probs[cal_lr])
    lr_logits_all = probs_to_logits(lr_test_probs)
    T_lr = fit_temperature(lr_logits_cal, lr_test_true[cal_lr])
    lr_probs_before = softmax_np(lr_logits_all)
    lr_probs_after = softmax_np(lr_logits_all / T_lr)
    print(f"  LR: T={T_lr:.3f}")

    # ══════════════════════════════════════════
    # XGB
    # ══════════════════════════════════════════
    print("Processing XGB...")
    xgb_test_probs = np.load(os.path.join(model_dir, "xgb_test_probs.npy"))
    xgb_test_true = np.load(os.path.join(model_dir, "xgb_test_true.npy")).astype(int)

    n_xgb = len(xgb_test_true)
    rng2 = np.random.RandomState(42)
    perm_xgb = rng2.permutation(n_xgb)
    n_cal_xgb = int(0.3 * n_xgb)
    cal_xgb = perm_xgb[:n_cal_xgb]

    xgb_logits_cal = probs_to_logits(xgb_test_probs[cal_xgb])
    xgb_logits_all = probs_to_logits(xgb_test_probs)
    T_xgb = fit_temperature(xgb_logits_cal, xgb_test_true[cal_xgb])
    xgb_probs_before = softmax_np(xgb_logits_all)
    xgb_probs_after = softmax_np(xgb_logits_all / T_xgb)
    print(f"  XGB: T={T_xgb:.3f}")

    # ══════════════════════════════════════════
    # SANN (deep ensemble — average probs across seed models)
    # ══════════════════════════════════════════
    print("Processing SANN...")
    import glob as glob_module

    # Check for ensemble seed models
    seed_files = sorted(glob_module.glob(os.path.join(model_dir, "sann_model_seed*.pt")))
    if len(seed_files) > 1:
        print(f"  Deep ensemble: found {len(seed_files)} seed models")
        # Need to load data and run inference for ensemble
        import scanpy as sc
        from sklearn.preprocessing import LabelEncoder

        adata = sc.read_h5ad(args.data)
        label_key = args.label_key

        le = LabelEncoder()
        y_all = le.fit_transform(adata.obs[label_key].values.astype(str))
        num_classes = len(le.classes_)

        # Load split info to get test/val indices
        split_file = os.path.join(model_dir, "train_val_test_split.json")
        with open(split_file) as f:
            split_info = json.load(f)

        # Reconstruct the split using the same logic as training
        from sklearn.model_selection import train_test_split
        n_total = len(y_all)
        test_frac = split_info["test_size"] / n_total
        val_frac = split_info.get("val_frac", 0.1)

        indices = np.arange(n_total)
        train_full_idx, test_idx = train_test_split(
            indices, test_size=test_frac, random_state=42, stratify=y_all)
        train_idx, val_idx = train_test_split(
            train_full_idx, test_size=val_frac, random_state=42, stratify=y_all[train_full_idx])

        if is_pca:
            X_pca = np.asarray(adata.obsm[args.pca_key][:, :args.pca_dim], dtype=np.float32)
            # Mask PCA
            X_raw = np.array(adata.X, dtype=np.float32)
            X_mask_raw = (X_raw != 0).astype(np.float32)
            from sklearn.decomposition import PCA as PCASK
            mask_pca = PCASK(n_components=args.pca_dim, random_state=42)
            mask_pca.fit(X_mask_raw[train_idx])
            X_mask_pca = mask_pca.transform(X_mask_raw).astype(np.float32)
            X_full = np.concatenate([X_pca, X_mask_pca], axis=1)
            n_expr = args.pca_dim
        else:
            X_raw = np.array(adata.X, dtype=np.float32)
            X_mask = (X_raw != 0).astype(np.float32)
            X_full = np.concatenate([X_raw, X_mask], axis=1)
            n_expr = X_raw.shape[1]

        X_test = X_full[test_idx]
        X_val = X_full[val_idx]
        y_test = y_all[test_idx]
        y_val = y_all[val_idx]

        # Build SANN architecture (must match training)
        import torch.nn as nn

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

        # Ensemble test probs
        all_test_probs = []
        all_val_probs = []
        for sf in seed_files:
            m = make_model()
            m.load_state_dict(torch.load(sf, map_location="cpu"))
            m.eval()
            with torch.no_grad():
                test_logits = m(torch.tensor(X_test, dtype=torch.float32)).numpy()
                val_logits = m(torch.tensor(X_val, dtype=torch.float32)).numpy()
            test_p = softmax_np(test_logits)
            val_p = softmax_np(val_logits)
            all_test_probs.append(test_p)
            all_val_probs.append(val_p)
            print(f"    Loaded {os.path.basename(sf)}")

        sann_test_probs = np.mean(all_test_probs, axis=0)
        sann_val_probs = np.mean(all_val_probs, axis=0)
        sann_test_true = y_test
        sann_val_true = y_val
    else:
        sann_test_probs = np.load(os.path.join(model_dir, "sann_test_probs.npy"))
        sann_test_true = np.load(os.path.join(model_dir, "sann_test_true.npy")).astype(int)
        sann_val_probs = np.load(os.path.join(model_dir, "sann_val_probs.npy"))
        sann_val_true = np.load(os.path.join(model_dir, "sann_val_true.npy")).astype(int)

    sann_logits_cal = probs_to_logits(sann_val_probs)
    sann_logits_test = probs_to_logits(sann_test_probs)
    T_sann = fit_temperature(sann_logits_cal, sann_val_true)
    sann_probs_before = softmax_np(sann_logits_test)
    sann_probs_after = softmax_np(sann_logits_test / T_sann)
    print(f"  SANN: T={T_sann:.3f}")

    # ══════════════════════════════════════════
    # Plot 3 side-by-side
    # ══════════════════════════════════════════
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    plot_single_model(axes[0], lr_probs_before, lr_probs_after,
                      lr_test_true, T_lr, "LR", n_bins=args.bins)
    plot_single_model(axes[1], xgb_probs_before, xgb_probs_after,
                      xgb_test_true, T_xgb, "XGB", n_bins=args.bins)
    plot_single_model(axes[2], sann_probs_before, sann_probs_after,
                      sann_test_true, T_sann, "SANN", n_bins=args.bins)

    rep_label = "PCA" if is_pca else "HVG"
    fig.suptitle(f"Temperature Scaling — Reliability Diagrams ({rep_label})",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved: {os.path.abspath(args.out)}")

    # Print summary
    print(f"\nSummary:")
    print(f"  LR:   T={T_lr:.3f}  |  ECE before={reliability_points(lr_probs_before, lr_test_true, args.bins)[3]:.4f}  after={reliability_points(lr_probs_after, lr_test_true, args.bins)[3]:.4f}")
    print(f"  XGB:  T={T_xgb:.3f}  |  ECE before={reliability_points(xgb_probs_before, xgb_test_true, args.bins)[3]:.4f}  after={reliability_points(xgb_probs_after, xgb_test_true, args.bins)[3]:.4f}")
    print(f"  SANN: T={T_sann:.3f}  |  ECE before={reliability_points(sann_probs_before, sann_test_true, args.bins)[3]:.4f}  after={reliability_points(sann_probs_after, sann_test_true, args.bins)[3]:.4f}")


if __name__ == "__main__":
    main()
