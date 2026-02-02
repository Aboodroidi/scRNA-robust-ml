# src/ablation_sann.py
import os
import json
import time
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

# macOS-friendly single-thread setup (important before torch import)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt


# ----------------------------
# Model
# ----------------------------
class SANN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int,
        dropout: float,
        activation: str,
        use_batchnorm: bool,
    ):
        super().__init__()

        act = self._make_activation(activation)

        layers: List[nn.Module] = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(act)
        if dropout and dropout > 0:
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dim, num_classes))
        self.net = nn.Sequential(*layers)

    @staticmethod
    def _make_activation(name: str) -> nn.Module:
        name = name.lower()
        if name == "relu":
            return nn.ReLU()
        if name == "gelu":
            return nn.GELU()
        raise ValueError(f"Unsupported activation '{name}'. Use 'relu' or 'gelu'.")

    def forward(self, x):
        return self.net(x)


# ----------------------------
# Config
# ----------------------------
@dataclass
class TrainConfig:
    # data
    pca_key: str = "X_pca"
    pca_dim: Optional[int] = 50  # if None, use whatever is in X_pca

    # split
    test_size: float = 0.2
    val_size: float = 0.1  # fraction of total (taken from train via stratified split)

    # training
    epochs: int = 20
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.0

    # model
    hidden_dim: int = 256
    dropout: float = 0.1
    activation: str = "relu"
    use_batchnorm: bool = False

    # regularization
    l1_lambda: float = 0.0

    # misc
    seed: int = 42
    device: str = "cpu"


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


# ----------------------------
# Data loading
# ----------------------------
def load_pca_data(h5ad_path: str, label_key: str, pca_key: str, pca_dim: Optional[int]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    adata = sc.read_h5ad(h5ad_path)

    if label_key not in adata.obs:
        raise ValueError(f"Expected adata.obs['{label_key}'] to exist (your labels).")

    if pca_key not in adata.obsm:
        raise ValueError(
            f"Expected adata.obsm['{pca_key}'] to exist. "
            f"Your SANN ablations should use the SAME PCA representation as train_sann.py."
        )

    Xpca = adata.obsm[pca_key]
    if pca_dim is not None:
        if Xpca.shape[1] < pca_dim:
            raise ValueError(f"{pca_key} has only {Xpca.shape[1]} dims; cannot take pca_dim={pca_dim}.")
        Xpca = Xpca[:, :pca_dim]

    y_cat = adata.obs[label_key].astype("category")
    y = y_cat.cat.codes.to_numpy()
    label_names = list(y_cat.cat.categories)

    X = np.asarray(Xpca, dtype=np.float32)
    return X, y, label_names


def make_fixed_splits(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    test_size: float,
    val_size: float
) -> Dict[str, np.ndarray]:
    """
    Fixed split:
      - test split from full data
      - val split from train portion (stratified)
    Returns dict of indices: train_idx, val_idx, test_idx
    """
    idx_all = np.arange(len(y))

    train_idx, test_idx = train_test_split(
        idx_all,
        test_size=test_size,
        stratify=y,
        random_state=seed,
    )

    # val_size is fraction of total; convert to fraction of train set:
    # val_from_train = val_size / (1 - test_size)
    val_from_train = val_size / (1.0 - test_size)
    y_train = y[train_idx]

    train_idx2, val_idx = train_test_split(
        train_idx,
        test_size=val_from_train,
        stratify=y_train,
        random_state=seed,
    )

    return {"train_idx": train_idx2, "val_idx": val_idx, "test_idx": test_idx}


# ----------------------------
# Training / evaluation
# ----------------------------
def l1_penalty(model: nn.Module) -> torch.Tensor:
    l1 = torch.tensor(0.0, device=next(model.parameters()).device)
    for p in model.parameters():
        l1 = l1 + p.abs().sum()
    return l1


@torch.no_grad()
def eval_epoch(model: nn.Module, loader: DataLoader, device: str) -> Tuple[float, float, float]:
    model.eval()
    ce = nn.CrossEntropyLoss()

    total_loss = 0.0
    all_preds = []
    all_targets = []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        logits = model(xb)
        loss = ce(logits, yb)
        total_loss += float(loss.item()) * xb.size(0)

        preds = logits.argmax(dim=1).detach().cpu().numpy()
        all_preds.append(preds)
        all_targets.append(yb.detach().cpu().numpy())

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)

    avg_loss = total_loss / len(y_true)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    return avg_loss, acc, f1


def train_one_variant(
    variant_name: str,
    cfg: TrainConfig,
    X: np.ndarray,
    y: np.ndarray,
    splits: Dict[str, np.ndarray],
    outdir: str,
    num_classes: int
) -> Dict:
    device = cfg.device
    os.makedirs(outdir, exist_ok=True)

    # tensors
    X_train = torch.from_numpy(X[splits["train_idx"]])
    y_train = torch.from_numpy(y[splits["train_idx"]]).long()
    X_val = torch.from_numpy(X[splits["val_idx"]])
    y_val = torch.from_numpy(y[splits["val_idx"]]).long()
    X_test = torch.from_numpy(X[splits["test_idx"]])
    y_test = torch.from_numpy(y[splits["test_idx"]]).long()

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=cfg.batch_size, shuffle=False)

    model = SANN(
        input_dim=X.shape[1],
        num_classes=num_classes,
        hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout,
        activation=cfg.activation,
        use_batchnorm=cfg.use_batchnorm,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    ce = nn.CrossEntropyLoss()

    history_rows = []
    best_val_f1 = -1.0
    best_epoch = -1
    best_path = os.path.join(outdir, "best_model.pt")

    start_t = time.time()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        n_seen = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()
            logits = model(xb)

            loss = ce(logits, yb)
            if cfg.l1_lambda and cfg.l1_lambda > 0:
                loss = loss + cfg.l1_lambda * l1_penalty(model)

            loss.backward()
            opt.step()

            running_loss += float(loss.item()) * xb.size(0)
            n_seen += xb.size(0)

        train_loss = running_loss / max(n_seen, 1)

        val_loss, val_acc, val_f1 = eval_epoch(model, val_loader, device)

        history_rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1,
        })

        # checkpoint by best val macro F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)

    elapsed = time.time() - start_t

    # load best and test
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_loss, test_acc, test_f1 = eval_epoch(model, test_loader, device)

    # save history
    hist_df = pd.DataFrame(history_rows)
    hist_df.to_csv(os.path.join(outdir, "history.csv"), index=False)

    metrics = {
        "variant": variant_name,
        "best_epoch": int(best_epoch),
        "best_val_macro_f1": float(best_val_f1),
        "final_test_macro_f1": float(test_f1),
        "final_test_accuracy": float(test_acc),
        "final_test_loss": float(test_loss),
        "training_time_sec": float(elapsed),
        "config": asdict(cfg),
    }
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


# ----------------------------
# Ablation definitions
# ----------------------------
def build_variants(base: TrainConfig) -> List[Tuple[str, TrainConfig]]:
    """
    Baseline reference: base config.
    Variants: change exactly one factor at a time.
    """
    variants: List[Tuple[str, TrainConfig]] = []

    # 0) baseline
    variants.append(("baseline", base))

    # 1) hidden size: smaller vs larger
    small = TrainConfig(**asdict(base))
    small.hidden_dim = max(64, base.hidden_dim // 2)
    variants.append((f"hidden_smaller_{small.hidden_dim}", small))

    large = TrainConfig(**asdict(base))
    large.hidden_dim = base.hidden_dim * 2
    variants.append((f"hidden_larger_{large.hidden_dim}", large))

    # 2) dropout: off vs on (or 0.1 vs 0.3)
    drop_off = TrainConfig(**asdict(base))
    drop_off.dropout = 0.0
    variants.append(("dropout_off", drop_off))

    drop_hi = TrainConfig(**asdict(base))
    drop_hi.dropout = 0.3
    variants.append(("dropout_0p3", drop_hi))

    # 3) activation: ReLU vs GELU
    gelu = TrainConfig(**asdict(base))
    gelu.activation = "gelu"
    variants.append(("activation_gelu", gelu))

    # 4) L1: off vs on (if base is 0, turn on; if base on, create off too)
    l1_on = TrainConfig(**asdict(base))
    if l1_on.l1_lambda <= 0:
        l1_on.l1_lambda = 1e-6  # mild L1 by default
    variants.append((f"l1_on_{l1_on.l1_lambda:g}", l1_on))

    l1_off = TrainConfig(**asdict(base))
    l1_off.l1_lambda = 0.0
    variants.append(("l1_off", l1_off))

    # 5) batch norm: off vs on
    bn_on = TrainConfig(**asdict(base))
    bn_on.use_batchnorm = True
    variants.append(("batchnorm_on", bn_on))

    bn_off = TrainConfig(**asdict(base))
    bn_off.use_batchnorm = False
    variants.append(("batchnorm_off", bn_off))

    # NOTE: That’s more than 5 “pairs” because we include both sides explicitly.
    # If you want EXACTLY 5 runs (not counting baseline), remove the extra ones.
    return variants


# ----------------------------
# CLI + main
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="SANN ablation runner (fixed split, per-epoch logging).")
    p.add_argument("--data", type=str, default="data/processed/pbmc68k_labeled.h5ad")
    p.add_argument("--outdir", type=str, default="results/ablations")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--config_json", type=str, default=None, help="Optional JSON string to override base config.")
    p.add_argument("--label_key", type=str, default="cell_type")
    p.add_argument("--pca_key", type=str, default="X_pca")
    p.add_argument("--pca_dim", type=int, default=50)
    p.add_argument("--plot_curves", action="store_true", help="Plot val macro-F1 curves for a few variants.")
    return p.parse_args()


def apply_overrides(base: TrainConfig, config_json: Optional[str], epochs: int, seed: int, device: str, pca_key: str, pca_dim: int) -> TrainConfig:
    cfg = TrainConfig(**asdict(base))
    cfg.epochs = epochs
    cfg.seed = seed
    cfg.device = device
    cfg.pca_key = pca_key
    cfg.pca_dim = pca_dim

    if config_json:
        overrides = json.loads(config_json)
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
            else:
                raise ValueError(f"Unknown config field in overrides: {k}")
    return cfg


def main():
    args = parse_args()
    set_seed(args.seed)

    base = TrainConfig()
    base = apply_overrides(base, args.config_json, args.epochs, args.seed, args.device, args.pca_key, args.pca_dim)

    X, y, label_names = load_pca_data(
        h5ad_path=args.data,
        label_key=args.label_key,
        pca_key=base.pca_key,
        pca_dim=base.pca_dim,
    )
    splits = make_fixed_splits(X, y, seed=base.seed, test_size=base.test_size, val_size=base.val_size)

    num_classes = len(label_names)

    # Build variants (baseline + ablations)
    variants = build_variants(base)

    os.makedirs(args.outdir, exist_ok=True)

    # save split indices once for reproducibility
    split_path = os.path.join(args.outdir, "fixed_splits.json")
    with open(split_path, "w") as f:
        json.dump({k: v.tolist() for k, v in splits.items()}, f, indent=2)

    results = []
    for name, cfg in variants:
        run_dir = os.path.join(args.outdir, name)
        print(f"\n=== Running variant: {name} ===")
        metrics = train_one_variant(
            variant_name=name,
            cfg=cfg,
            X=X,
            y=y,
            splits=splits,
            outdir=run_dir,
            num_classes=num_classes,
        )
        results.append({
            "variant": metrics["variant"],
            "best_epoch": metrics["best_epoch"],
            "best_val_macro_f1": metrics["best_val_macro_f1"],
            "final_test_macro_f1": metrics["final_test_macro_f1"],
            "final_test_accuracy": metrics["final_test_accuracy"],
            "training_time_sec": metrics["training_time_sec"],
            "hidden_dim": cfg.hidden_dim,
            "dropout": cfg.dropout,
            "activation": cfg.activation,
            "l1_lambda": cfg.l1_lambda,
            "batchnorm": cfg.use_batchnorm,
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
        })

    summary_df = pd.DataFrame(results).sort_values(by="best_val_macro_f1", ascending=False)
    summary_path = os.path.join(args.outdir, "ablation_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary: {summary_path}")

    # Optional: plot val macro-F1 curves for a few key variants (keep report space)
    if args.plot_curves:
        # pick top 4 by best val F1
        top_variants = summary_df["variant"].head(4).tolist()

        fig, ax = plt.subplots(figsize=(7, 5))
        for v in top_variants:
            hist_path = os.path.join(args.outdir, v, "history.csv")
            h = pd.read_csv(hist_path)
            ax.plot(h["epoch"], h["val_macro_f1"], label=v)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation Macro-F1")
        ax.set_title("Validation Macro-F1 Curves (Top Variants)")
        ax.grid(False)
        ax.legend()

        fig.tight_layout()
        out_plot = os.path.join(args.outdir, "val_macro_f1_curves_top_variants.png")
        fig.savefig(out_plot, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved curve plot: {out_plot}")


if __name__ == "__main__":
    main()