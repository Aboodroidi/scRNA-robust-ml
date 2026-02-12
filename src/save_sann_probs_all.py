# src/save_sann_probs_all.py
import os
import argparse

# macOS-friendly
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MPLBACKEND"] = "Agg"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import scanpy as sc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# --------- Model definition (must match training) ----------
class SANN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        activation: str = "relu",
        use_batchnorm: bool = False,
    ):
        super().__init__()
        act = self._make_activation(activation)

        layers = [nn.Linear(input_dim, hidden_dim)]
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
        raise ValueError(f"Unsupported activation: {name}")

    def forward(self, x):
        return self.net(x)


def parse_args():
    p = argparse.ArgumentParser(description="Run SANN inference on ALL cells and save probs + true labels.")
    p.add_argument("--data", type=str, default="data/processed/pbmc68k_labeled.h5ad")
    p.add_argument("--label_key", type=str, default="cell_type")

    # representation (match training)
    p.add_argument("--use_pca", action="store_true", default=True)
    p.add_argument("--pca_key", type=str, default="X_pca")
    p.add_argument("--pca_dim", type=int, default=50)

    # checkpoint + architecture params (MUST MATCH CKPT)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--activation", type=str, default="relu", choices=["relu", "gelu"])
    p.add_argument("--batchnorm", action="store_true", default=False)

    # inference
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--device", type=str, default="cpu")

    # outputs
    p.add_argument("--out_probs", type=str, default="results/sann_all_probs.npy")
    p.add_argument("--out_true", type=str, default="results/sann_all_true.npy")
    return p.parse_args()


@torch.no_grad()
def infer_probs(model: nn.Module, X: np.ndarray, batch_size: int, device: str) -> np.ndarray:
    model.eval()
    ds = TensorDataset(torch.from_numpy(X))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    probs_list = []
    for (xb,) in dl:
        xb = xb.to(device)
        logits = model(xb)
        probs = torch.softmax(logits, dim=1)
        probs_list.append(probs.cpu().numpy())

    return np.vstack(probs_list)


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_probs), exist_ok=True)

    adata = sc.read_h5ad(args.data)

    # X
    if args.use_pca:
        if args.pca_key not in adata.obsm:
            raise ValueError(f"Expected adata.obsm['{args.pca_key}'] for PCA features.")
        X = adata.obsm[args.pca_key]
        if args.pca_dim is not None:
            X = X[:, : args.pca_dim]
        X = np.asarray(X, dtype=np.float32)
    else:
        X = np.asarray(adata.X, dtype=np.float32)

    # y (as integer codes consistent with category order)
    if args.label_key not in adata.obs:
        raise ValueError(f"Expected adata.obs['{args.label_key}'] to exist.")
    y_cat = adata.obs[args.label_key].astype("category")
    y = y_cat.cat.codes.to_numpy().astype(int)
    class_names = list(y_cat.cat.categories)

    print(f"[Sanity] X shape: {X.shape}")
    print(f"[Sanity] y length: {len(y)}")
    print(f"[Sanity] #classes: {len(class_names)}")
    print(f"[Sanity] class names: {class_names}")

    # model
    device = args.device
    model = SANN(
        input_dim=X.shape[1],
        num_classes=len(class_names),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        activation=args.activation,
        use_batchnorm=args.batchnorm,
    ).to(device)

    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)

    # infer
    probs = infer_probs(model, X, batch_size=args.batch_size, device=device)

    # checks
    if probs.shape[0] != X.shape[0]:
        raise ValueError("Probs rows != number of cells")
    if probs.shape[1] != len(class_names):
        raise ValueError("Probs cols != number of classes")
    row_sums = probs.sum(axis=1)
    print(f"[Sanity] probs shape: {probs.shape}")
    print(f"[Sanity] probs row-sum (min/mean/max): {row_sums.min():.4f} / {row_sums.mean():.4f} / {row_sums.max():.4f}")

    np.save(args.out_probs, probs)
    np.save(args.out_true, y)

    print(f"[Sanity] Saved: {args.out_probs}")
    print(f"[Sanity] Saved: {args.out_true}")


if __name__ == "__main__":
    main()