import os

# macOS-safe settings
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["MPLBACKEND"] = "Agg"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONUTF8"] = "1"
os.environ["LC_ALL"] = "en_US.UTF-8"
os.environ["LANG"] = "en_US.UTF-8"

import argparse
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class SANN(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=256, dropout=0.1, use_batchnorm=True):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim)]
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers += [nn.ReLU()]
        if dropout and dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def parse_args():
    p = argparse.ArgumentParser(description="Create stable 3D plots using 2D UMAP + PCA1, with full true and full SANN-predicted labels.")
    p.add_argument("--adata", type=str, default="data/processed/pbmc68k_labeled.h5ad")
    p.add_argument("--label_key", type=str, default="cell_type")

    p.add_argument("--umap2d_key", type=str, default="X_umap")
    p.add_argument("--pca_key", type=str, default="X_pca")
    p.add_argument("--pca_dim", type=int, default=50)
    p.add_argument("--pca_z_index", type=int, default=0)

    p.add_argument("--sann_ckpt", type=str, default="results/full_train/sann_model.pt")
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batchnorm", action="store_true", default=True)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--device", type=str, default="cpu")

    p.add_argument("--out_true", type=str, default="results/figures/umap_3d_true_labels.png")
    p.add_argument("--out_pred", type=str, default="results/figures/umap_3d_sann_pred_full.png")
    p.add_argument("--dpi", type=int, default=300)

    p.add_argument("--fig_w", type=float, default=11.5)
    p.add_argument("--fig_h", type=float, default=8.5)
    p.add_argument("--point_size", type=float, default=4.0)
    p.add_argument("--alpha_true", type=float, default=0.65)
    p.add_argument("--alpha_pred", type=float, default=0.65)

    p.add_argument("--elev", type=float, default=24)
    p.add_argument("--azim", type=float, default=42)
    return p.parse_args()


def ensure_parent(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def build_palette(class_names):
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(len(class_names))]
    return {label: colors[i] for i, label in enumerate(class_names)}


def make_legend_handles(class_names, palette, marker_size=7):
    return [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markerfacecolor=palette[name],
            markeredgecolor="none",
            markersize=marker_size,
            label=name,
        )
        for name in class_names
    ]


def get_stable_3d_coords(adata, umap2d_key="X_umap", pca_key="X_pca", pca_z_index=0):
    if umap2d_key not in adata.obsm:
        raise ValueError(f"Expected 2D UMAP in adata.obsm['{umap2d_key}']")
    if pca_key not in adata.obsm:
        raise ValueError(f"Expected PCA in adata.obsm['{pca_key}']")

    umap2d = np.asarray(adata.obsm[umap2d_key], dtype=np.float32)
    pca = np.asarray(adata.obsm[pca_key], dtype=np.float32)

    if umap2d.shape[1] < 2:
        raise ValueError(f"{umap2d_key} must have at least 2 dimensions. Found shape {umap2d.shape}")
    if pca.shape[1] <= pca_z_index:
        raise ValueError(f"{pca_key} has only {pca.shape[1]} dims; cannot use z index {pca_z_index}")

    x = umap2d[:, 0]
    y = umap2d[:, 1]
    z = pca[:, pca_z_index]

    # scale z for nicer appearance
    z = (z - z.mean()) / (z.std() + 1e-8)
    coords = np.column_stack([x, y, z]).astype(np.float32)
    return coords


@torch.no_grad()
def infer_full_predictions(model, X_all, batch_size, device):
    model.eval()
    ds = TensorDataset(torch.from_numpy(X_all))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    preds = []
    for (xb,) in dl:
        xb = xb.to(device)
        logits = model(xb)
        pred = torch.argmax(logits, dim=1).cpu().numpy()
        preds.append(pred)

    return np.concatenate(preds)


def set_common_axes(ax, coords, elev, azim, title):
    ax.set_xlabel("UMAP1", labelpad=12)
    ax.set_ylabel("UMAP2", labelpad=12)
    ax.set_zlabel("PC1", labelpad=12)
    ax.set_title(title, pad=18)
    ax.view_init(elev=elev, azim=azim)

    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    ax.set_xlim(float(np.min(x)), float(np.max(x)))
    ax.set_ylim(float(np.min(y)), float(np.max(y)))
    ax.set_zlim(float(np.min(z)), float(np.max(z)))
    ax.grid(False)


def plot_labels(coords, labels, class_names, palette, out_path,
                fig_w, fig_h, point_size, alpha_val, elev, azim, dpi, title, legend_title):
    ensure_parent(out_path)

    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")

    for cname in class_names:
        mask = (labels == cname)
        if np.any(mask):
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                coords[mask, 2],
                s=point_size,
                alpha=alpha_val,
                color=palette[cname],
                linewidths=0,
            )

    set_common_axes(ax, coords, elev, azim, title)

    handles = make_legend_handles(class_names, palette, marker_size=7)
    ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.03, 0.5),
        frameon=False,
        title=legend_title,
        borderaxespad=1.0,
    )

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def main():
    args = parse_args()

    ensure_parent(args.out_true)
    ensure_parent(args.out_pred)

    adata = sc.read_h5ad(args.adata)

    if args.label_key not in adata.obs:
        raise ValueError(f"Expected adata.obs['{args.label_key}'] to exist.")
    if args.pca_key not in adata.obsm:
        raise ValueError(f"Expected PCA features in adata.obsm['{args.pca_key}']")
    if not os.path.exists(args.sann_ckpt):
        raise FileNotFoundError(f"SANN checkpoint not found: {args.sann_ckpt}")

    coords = get_stable_3d_coords(
        adata,
        umap2d_key=args.umap2d_key,
        pca_key=args.pca_key,
        pca_z_index=args.pca_z_index,
    )

    y_cat = adata.obs[args.label_key].astype("category")
    class_names = list(y_cat.cat.categories)
    true_labels = y_cat.astype(str).to_numpy()
    palette = build_palette(class_names)

    # True-label plot
    plot_labels(
        coords=coords,
        labels=true_labels,
        class_names=class_names,
        palette=palette,
        out_path=args.out_true,
        fig_w=args.fig_w,
        fig_h=args.fig_h,
        point_size=args.point_size,
        alpha_val=args.alpha_true,
        elev=args.elev,
        azim=args.azim,
        dpi=args.dpi,
        title="3D Embedding: True Cell-Type Labels",
        legend_title="True labels",
    )

    # Full SANN predictions
    X_all = np.asarray(adata.obsm[args.pca_key][:, :args.pca_dim], dtype=np.float32)

    model = SANN(
        input_dim=X_all.shape[1],
        num_classes=len(class_names),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        use_batchnorm=args.batchnorm,
    ).to(args.device)

    state = torch.load(args.sann_ckpt, map_location=args.device)
    model.load_state_dict(state)

    y_pred = infer_full_predictions(
        model=model,
        X_all=X_all,
        batch_size=args.batch_size,
        device=args.device,
    )
    pred_labels = np.array([class_names[i] for i in y_pred], dtype=object)

    # Save full preds for reuse if helpful
    np.save("results/full_train/sann_all_pred.npy", y_pred)

    # Predicted-label plot
    plot_labels(
        coords=coords,
        labels=pred_labels,
        class_names=class_names,
        palette=palette,
        out_path=args.out_pred,
        fig_w=args.fig_w,
        fig_h=args.fig_h,
        point_size=args.point_size,
        alpha_val=args.alpha_pred,
        elev=args.elev,
        azim=args.azim,
        dpi=args.dpi,
        title="3D Embedding: SANN Predicted Labels",
        legend_title="Predicted labels",
    )

    print(f"[Sanity] n_cells = {adata.n_obs}")
    print(f"[Sanity] full prediction count = {len(y_pred)}")
    print(f"[Sanity] stable 3D coords shape = {coords.shape}")
    print(f"[Saved] {args.out_true}")
    print(f"[Saved] {args.out_pred}")
    print("[Saved] results/full_train/sann_all_pred.npy")


if __name__ == "__main__":
    main()