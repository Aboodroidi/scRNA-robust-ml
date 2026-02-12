# src/plot_reliability_all_models.py
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

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier


def parse_args():
    p = argparse.ArgumentParser(description="Plot reliability diagrams (LR, XGB, SANN) side-by-side.")
    p.add_argument("--data", type=str, default="data/processed/pbmc68k_labeled.h5ad")
    p.add_argument("--label_key", type=str, default="cell_type")
    p.add_argument("--splits", type=str, default="results/ablations/fixed_splits.json")

    # feature representation (match your SANN)
    p.add_argument("--use_pca", action="store_true", default=True)
    p.add_argument("--pca_key", type=str, default="X_pca")
    p.add_argument("--pca_dim", type=int, default=50)

    # saved SANN outputs
    p.add_argument("--sann_probs", type=str, default="results/sann_test_probs.npy")
    p.add_argument("--sann_true", type=str, default="results/sann_test_true.npy")

    # LR artifacts (PCA version)
    p.add_argument("--lr_model", type=str, default="results/logreg_model_pca50.pkl")
    p.add_argument("--lr_scaler", type=str, default="results/scaler_pca50.pkl")

    # XGB artifact (JSON is most stable on macOS)
    p.add_argument("--xgb_json", type=str, default="results/xgb_model_pca50.json")

    # output
    p.add_argument("--out", type=str, default="results/figures/reliability_all_models.png")
    p.add_argument("--bins", type=int, default=10)
    p.add_argument("--dpi", type=int, default=300)

    # also save probs for LR/XGB for later analysis
    p.add_argument("--save_probs", action="store_true", default=True)
    return p.parse_args()


def compute_reliability(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10):
    """
    Returns:
      bin_centers (mean confidence in bin),
      bin_acc (observed accuracy),
      bin_counts,
      ece
    """
    probs = np.asarray(probs)
    y_true = np.asarray(y_true).astype(int)

    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(int)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(conf, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    bin_conf = np.zeros(n_bins, dtype=float)
    bin_acc = np.zeros(n_bins, dtype=float)
    bin_counts = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        mask = (bin_ids == b)
        cnt = int(mask.sum())
        bin_counts[b] = cnt
        if cnt > 0:
            bin_conf[b] = float(conf[mask].mean())
            bin_acc[b] = float(correct[mask].mean())
        else:
            bin_conf[b] = np.nan
            bin_acc[b] = np.nan

    # Expected Calibration Error (ECE)
    n_total = len(y_true)
    ece = 0.0
    for b in range(n_bins):
        if bin_counts[b] > 0:
            w = bin_counts[b] / n_total
            ece += w * abs(bin_acc[b] - bin_conf[b])

    return bin_conf, bin_acc, bin_counts, float(ece)


def plot_reliability(ax, probs, y_true, title, n_bins=10):
    bin_conf, bin_acc, bin_counts, ece = compute_reliability(probs, y_true, n_bins=n_bins)

    # perfect calibration line
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)

    # plot only bins with data
    mask = ~np.isnan(bin_conf) & ~np.isnan(bin_acc)
    ax.plot(bin_conf[mask], bin_acc[mask], marker="o", linewidth=1.5)

    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Observed accuracy")

    # ECE text
    ax.text(
        0.05, 0.08,
        f"ECE = {ece:.3f}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="bottom"
    )
    return ece


def load_test_split_indices(path: str):
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


def ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def main():
    args = parse_args()
    ensure_dir(args.out)

    # Load data + features
    adata = sc.read_h5ad(args.data)
    X, y, class_names = get_X_y(
        adata,
        label_key=args.label_key,
        use_pca=args.use_pca,
        pca_key=args.pca_key,
        pca_dim=args.pca_dim
    )

    test_idx = load_test_split_indices(args.splits)
    X_test = X[test_idx]
    y_test = y[test_idx]

    print(f"[Sanity] X shape: {X.shape} | X_test shape: {X_test.shape}")
    print(f"[Sanity] y length: {len(y)} | y_test length: {len(y_test)}")
    print(f"[Sanity] #classes: {len(class_names)}")
    print(f"[Sanity] class names: {class_names}")
    print(f"[Sanity] test_idx length: {len(test_idx)}")

    # ----------------------------
    # Load SANN outputs (probs + true)
    # ----------------------------
    sann_probs = np.load(args.sann_probs)
    sann_true = np.load(args.sann_true).astype(int)

    if len(sann_true) != len(test_idx) or sann_probs.shape[0] != len(test_idx):
        raise ValueError(
            f"SANN arrays must match test size. "
            f"len(test_idx)={len(test_idx)}, sann_true={len(sann_true)}, sann_probs={sann_probs.shape}"
        )

    # Optional consistency check: sann_true should match dataset y_test
    if not np.array_equal(sann_true, y_test):
        print("[Warning] sann_test_true.npy does not match y_test from adata + fixed_splits. "
              "Proceeding with sann_true for SANN plot, and y_test for LR/XGB.")

    # ----------------------------
    # Logistic Regression probs
    # ----------------------------
    if os.path.exists(args.lr_model) and os.path.exists(args.lr_scaler):
        lr = joblib.load(args.lr_model)
        scaler = joblib.load(args.lr_scaler)
        # check feature match
        exp = getattr(scaler, "n_features_in_", None)
        if exp is not None and exp != X_test.shape[1]:
            raise ValueError(
                f"LR scaler expects {exp} features, but X_test has {X_test.shape[1]}. "
                f"Use the PCA50 artifacts or retrain."
            )
        X_test_lr = scaler.transform(X_test)
    else:
        # Train LR quickly if not found (deterministic, but uses only train split indices if available)
        print("[Info] LR artifacts not found, training LR quickly on the fixed split.")
        # if train_idx exists, use it; else use complement of test_idx
        train_idx = None
        with open(args.splits, "r") as f:
            d = json.load(f)
        if "train_idx" in d:
            train_idx = np.array(d["train_idx"], dtype=int)
        else:
            mask = np.ones(len(y), dtype=bool)
            mask[test_idx] = False
            train_idx = np.where(mask)[0]

        X_train = X[train_idx]
        y_train = y[train_idx]

        scaler = StandardScaler(with_mean=False)
        X_train_lr = scaler.fit_transform(X_train)
        X_test_lr = scaler.transform(X_test)

        lr = LogisticRegression(
            max_iter=3000,
            solver="saga",
            multi_class="multinomial",
            n_jobs=1,
            random_state=42,
        ).fit(X_train_lr, y_train)

        joblib.dump(lr, args.lr_model)
        joblib.dump(scaler, args.lr_scaler)

    lr_probs = lr.predict_proba(X_test_lr)

    # ----------------------------
    # XGBoost probs
    # ----------------------------
    xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=len(class_names),
        n_jobs=1,
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=42,
    )
    if os.path.exists(args.xgb_json):
        xgb.load_model(args.xgb_json)
    else:
        print("[Info] XGB artifact not found, training XGB quickly on the fixed split.")
        with open(args.splits, "r") as f:
            d = json.load(f)
        if "train_idx" in d:
            train_idx = np.array(d["train_idx"], dtype=int)
        else:
            mask = np.ones(len(y), dtype=bool)
            mask[test_idx] = False
            train_idx = np.where(mask)[0]
        X_train = X[train_idx]
        y_train = y[train_idx]
        xgb.set_params(max_depth=4, learning_rate=0.1, n_estimators=200, subsample=0.8, colsample_bytree=0.8)
        xgb.fit(X_train, y_train)
        xgb.save_model(args.xgb_json)

    xgb_probs = xgb.predict_proba(X_test)

    # Save probs for later calibration/error analysis
    if args.save_probs:
        np.save("results/lr_test_probs.npy", lr_probs)
        np.save("results/lr_test_true.npy", y_test)
        np.save("results/lr_test_pred.npy", lr_probs.argmax(axis=1))

        np.save("results/xgb_test_probs.npy", xgb_probs)
        np.save("results/xgb_test_true.npy", y_test)
        np.save("results/xgb_test_pred.npy", xgb_probs.argmax(axis=1))

        print("[Sanity] Saved LR probs/preds/true to results/lr_test_*.npy")
        print("[Sanity] Saved XGB probs/preds/true to results/xgb_test_*.npy")

    # Quick sanity metrics (optional prints)
    print(f"[Sanity] LR test acc={accuracy_score(y_test, lr_probs.argmax(1)):.4f} "
          f"macroF1={f1_score(y_test, lr_probs.argmax(1), average='macro'):.4f}")
    print(f"[Sanity] XGB test acc={accuracy_score(y_test, xgb_probs.argmax(1)):.4f} "
          f"macroF1={f1_score(y_test, xgb_probs.argmax(1), average='macro'):.4f}")
    print(f"[Sanity] SANN test acc={accuracy_score(sann_true, sann_probs.argmax(1)):.4f} "
          f"macroF1={f1_score(sann_true, sann_probs.argmax(1), average='macro'):.4f}")

    # ----------------------------
    # Plot combined 1x3 reliability diagrams
    # ----------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True, sharey=True)

    plot_reliability(axes[0], lr_probs, y_test, "Logistic Regression", n_bins=args.bins)
    plot_reliability(axes[1], xgb_probs, y_test, "XGBoost", n_bins=args.bins)
    plot_reliability(axes[2], sann_probs, sann_true, "SANN", n_bins=args.bins)

    fig.suptitle("Reliability Diagrams (Test Set, 10 Confidence Bins)", y=1.02)
    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"[Sanity] Saved combined figure: {args.out}")


if __name__ == "__main__":
    main()