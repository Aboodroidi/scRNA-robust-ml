# src/plot_umap_misclassifications.py
import os
import json
import argparse
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def parse_args():
    p = argparse.ArgumentParser(description="Plot misclassifications on UMAP (main + top-pairs + density).")
    p.add_argument("--data", type=str, default="data/processed/pbmc68k_labeled.h5ad")
    p.add_argument("--splits", type=str, default="results/ablations/fixed_splits.json")
    p.add_argument("--model", type=str, default="sann", choices=["sann", "lr", "xgb"])

    # saved predictions for TEST split only
    p.add_argument("--true", type=str, default="")
    p.add_argument("--pred", type=str, default="")

    # plotting
    p.add_argument("--outdir", type=str, default="results/figures")
    p.add_argument("--top_k", type=int, default=6, help="Top confusion pairs to colour in the pair plot.")
    p.add_argument("--dpi", type=int, default=300)

    # marker sizes
    p.add_argument("--bg_size", type=float, default=4.0, help="Background point size (test cells).")
    p.add_argument("--err_size", type=float, default=10.0, help="Misclassified point size.")
    p.add_argument("--bg_alpha", type=float, default=0.18, help="Background alpha.")
    p.add_argument("--err_alpha", type=float, default=0.85, help="Error alpha.")

    # density
    p.add_argument("--hex_gridsize", type=int, default=55)
    return p.parse_args()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def default_paths(model: str):
    if model == "sann":
        return "results/sann_test_true.npy", "results/sann_test_pred.npy"
    if model == "lr":
        return "results/lr_test_true.npy", "results/lr_test_pred.npy"
    if model == "xgb":
        return "results/xgb_test_true.npy", "results/xgb_test_pred.npy"
    raise ValueError("unknown model")


def load_split_test_idx(splits_path: str):
    with open(splits_path, "r") as f:
        d = json.load(f)
    if "test_idx" not in d:
        raise ValueError(f"Split file must contain 'test_idx'. Found keys: {list(d.keys())}")
    return np.array(d["test_idx"], dtype=int)


def class_names_from_adata(adata, label_key="cell_type"):
    # must match the encoding you used across scripts
    y_cat = adata.obs[label_key].astype("category")
    return list(y_cat.cat.categories)


def top_confusion_pairs(y_true, y_pred, class_names, top_k=6):
    """
    Return list of top_k pairs as tuples (true_idx, pred_idx) by raw confusion count, excluding diagonal.
    """
    n = len(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(n))
    cm_off = cm.copy()
    np.fill_diagonal(cm_off, 0)

    flat = np.argsort(cm_off.ravel())[::-1]
    pairs = []
    for idx in flat:
        i = idx // n
        j = idx % n
        c = cm_off[i, j]
        if c <= 0:
            break
        pairs.append((i, j, int(c)))
        if len(pairs) >= top_k:
            break
    return pairs, cm


def plot_main(ax, umap_xy, is_mis, title, bg_size, err_size, bg_alpha, err_alpha):
    # all test cells in grey
    ax.scatter(
        umap_xy[:, 0], umap_xy[:, 1],
        s=bg_size, alpha=bg_alpha, linewidths=0,
        c="#BDBDBD", label="Correct/All test"
    )
    # misclassified overlay in red
    if is_mis.any():
        ax.scatter(
            umap_xy[is_mis, 0], umap_xy[is_mis, 1],
            s=err_size, alpha=err_alpha, linewidths=0,
            c="#D32F2F", label="Misclassified"
        )

    ax.set_title(title)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.grid(False)
    ax.legend(frameon=False, loc="best")


def plot_pairs(ax, umap_xy, y_true, y_pred, class_names, title, top_k, bg_size, err_size, bg_alpha, err_alpha):
    is_mis = (y_true != y_pred)

    # background: all test cells grey
    ax.scatter(
        umap_xy[:, 0], umap_xy[:, 1],
        s=bg_size, alpha=bg_alpha, linewidths=0, c="#BDBDBD"
    )

    if not is_mis.any():
        ax.set_title(title + " (no errors)")
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        return

    pairs, _ = top_confusion_pairs(y_true, y_pred, class_names, top_k=top_k)
    top_set = {(i, j) for (i, j, _) in pairs}

    # assign label per misclassified point
    labels = []
    for t, p in zip(y_true[is_mis], y_pred[is_mis]):
        if (t, p) in top_set:
            labels.append(f"{class_names[t]} → {class_names[p]}")
        else:
            labels.append("Other")
    labels = np.array(labels)

    # plot each group separately (no custom colors needed; matplotlib cycles)
    for lab in np.unique(labels):
        mask = (labels == lab)
        ax.scatter(
            umap_xy[is_mis, 0][mask], umap_xy[is_mis, 1][mask],
            s=err_size, alpha=err_alpha, linewidths=0,
            label=lab
        )

    ax.set_title(title)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.grid(False)

    # put legend outside to avoid covering data
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)


def plot_density(ax, umap_xy, is_mis, title, bg_size, bg_alpha, gridsize):
    # background test points grey
    ax.scatter(
        umap_xy[:, 0], umap_xy[:, 1],
        s=bg_size, alpha=bg_alpha, linewidths=0, c="#BDBDBD"
    )

    if is_mis.any():
        hb = ax.hexbin(
            umap_xy[is_mis, 0], umap_xy[is_mis, 1],
            gridsize=gridsize, mincnt=1
        )
        plt.colorbar(hb, ax=ax, fraction=0.046, pad=0.04, label="Misclassified density")

    ax.set_title(title)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.grid(False)


def main():
    args = parse_args()
    ensure_dir(args.outdir)

    # resolve default y paths if not provided
    if args.true == "" or args.pred == "":
        t, p = default_paths(args.model)
        args.true = t if args.true == "" else args.true
        args.pred = p if args.pred == "" else args.pred

    # load data + check UMAP
    adata = sc.read_h5ad(args.data)
    if "X_umap" not in adata.obsm:
        raise ValueError("adata.obsm['X_umap'] not found. Compute UMAP first and save into the h5ad.")

    test_idx = load_split_test_idx(args.splits)

    # test UMAP
    umap_xy = np.asarray(adata.obsm["X_umap"][test_idx], dtype=float)

    # class names for pair labels
    class_names = class_names_from_adata(adata, label_key="cell_type")

    # load arrays
    y_true = np.load(args.true).astype(int)
    y_pred = np.load(args.pred).astype(int)

    if len(y_true) != len(test_idx) or len(y_pred) != len(test_idx):
        raise ValueError(
            f"Length mismatch: test_idx={len(test_idx)} y_true={len(y_true)} y_pred={len(y_pred)}. "
            "These must match."
        )

    is_mis = (y_true != y_pred)

    print(f"[Sanity] Model: {args.model}")
    print(f"[Sanity] Test cells: {len(test_idx)} | Misclassified: {int(is_mis.sum())}")

    # 1) Main plot
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    plot_main(
        ax,
        umap_xy,
        is_mis,
        title=f"Misclassified Cells on UMAP ({args.model.upper()})",
        bg_size=args.bg_size,
        err_size=args.err_size,
        bg_alpha=args.bg_alpha,
        err_alpha=args.err_alpha,
    )
    out1 = os.path.join(args.outdir, f"umap_misclassified_{args.model}.png")
    fig.tight_layout()
    fig.savefig(out1, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {out1}")

    # 2) Pair-coloured plot (optional strong)
    fig, ax = plt.subplots(figsize=(9.6, 6.2))
    plot_pairs(
        ax,
        umap_xy,
        y_true,
        y_pred,
        class_names,
        title=f"Misclassifications by Top Confusion Pairs ({args.model.upper()})",
        top_k=args.top_k,
        bg_size=args.bg_size,
        err_size=args.err_size,
        bg_alpha=args.bg_alpha,
        err_alpha=args.err_alpha,
    )
    out2 = os.path.join(args.outdir, f"umap_misclassified_pairs_{args.model}.png")
    fig.tight_layout()
    fig.savefig(out2, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {out2}")

    # 3) Density plot (tiny extra)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    plot_density(
        ax,
        umap_xy,
        is_mis,
        title=f"Misclassification Density on UMAP ({args.model.upper()})",
        bg_size=args.bg_size,
        bg_alpha=args.bg_alpha,
        gridsize=args.hex_gridsize,
    )
    out3 = os.path.join(args.outdir, f"umap_error_density_{args.model}.png")
    fig.tight_layout()
    fig.savefig(out3, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {out3}")


if __name__ == "__main__":
    main()